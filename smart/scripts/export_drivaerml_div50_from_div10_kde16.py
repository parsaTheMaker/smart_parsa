#!/usr/bin/env python3
"""Create total-div50 DrivAerML meshes from div10 inputs and export KDE-16 fields."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
SMART_ROOT = SCRIPT_DIR.parent
if str(SMART_ROOT) not in sys.path:
    sys.path.insert(0, str(SMART_ROOT))

from data.ahmedml_dataset_v2 import AhmedMLDatasetV2
from export_drivaerml_remesh_kde16 import (
    compute_kde16,
    read_polydata,
    resolve_device,
    validate_written_vtp,
    write_density_mesh,
)


DEFAULT_DATA_ROOT = "/mnt/ssdraid/parsa/drivaerml_preprocessed"
DEFAULT_FEATURE_ROOT = "/mnt/ssdraid/parsa/drivaerml_surface_vtp_remesh_v4/feature"
DEFAULT_QEM_ROOT = "/mnt/ssdraid/parsa/drivaerml_surface_vtp_remesh_v4/quadric"
DEFAULT_VOXEL_ROOT = "/mnt/ssdraid/parsa/drivaerml_surface_vtp_remesh_v4/voxel"
DEFAULT_OUTPUT_DIR = "/home/parsa/smart_parsa/results/final/drivaerml_run29_mesh_kde16"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", type=int, default=29)
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--feature-vtp-root", default=DEFAULT_FEATURE_ROOT)
    parser.add_argument("--qem-vtp-root", default=DEFAULT_QEM_ROOT)
    parser.add_argument("--voxel-vtp-root", default=DEFAULT_VOXEL_ROOT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--knn-k", type=int, default=16)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--feature-angle", type=float, default=30.0)
    parser.add_argument(
        "--sources",
        default="all",
        help="Comma-separated source labels to regenerate, or 'all' (default).",
    )
    return parser.parse_args()


def source_paths(args: argparse.Namespace) -> dict[str, Path]:
    run_id = int(args.run_id)
    name = f"drivaer_{run_id}_faces_div10.vtp"
    paths = {
        "feature_aware": Path(args.feature_vtp_root) / f"run_{run_id}" / name,
        "qem": Path(args.qem_vtp_root) / f"run_{run_id}" / name,
        "voxel_grid": Path(args.voxel_vtp_root) / f"run_{run_id}" / name,
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing div10 source VTPs:\n" + "\n".join(missing))
    return paths


def triangulated(vtk, polydata):
    triangle_filter = vtk.vtkTriangleFilter()
    triangle_filter.SetInputData(polydata)
    triangle_filter.PassVertsOff()
    triangle_filter.PassLinesOff()
    triangle_filter.Update()
    result = vtk.vtkPolyData()
    result.ShallowCopy(triangle_filter.GetOutput())
    return result


def decimate_feature_aware(vtk, polydata, feature_angle: float):
    decimator = vtk.vtkDecimatePro()
    decimator.SetInputData(polydata)
    decimator.SetTargetReduction(0.8)  # div10 -> div50 is an additional 5x reduction.
    decimator.PreserveTopologyOn()
    decimator.BoundaryVertexDeletionOff()
    decimator.SetFeatureAngle(float(feature_angle))
    decimator.Update()
    result = vtk.vtkPolyData()
    result.ShallowCopy(decimator.GetOutput())
    return result


def decimate_qem(vtk, polydata):
    decimator = vtk.vtkQuadricDecimation()
    decimator.SetInputData(polydata)
    decimator.SetTargetReduction(0.8)  # div10 -> div50 is an additional 5x reduction.
    decimator.AttributeErrorMetricOff()
    decimator.VolumePreservationOn()
    decimator.Update()
    result = vtk.vtkPolyData()
    result.ShallowCopy(decimator.GetOutput())
    return result


def decimate_voxel_grid(vtk, polydata, feature_angle: float):
    cluster = vtk.vtkQuadricClustering()
    cluster.SetInputData(polydata)
    # This grid is calibrated on run 29's div10 voxel source to recover an
    # approximately 50x reduction relative to the original surface.
    cluster.SetNumberOfDivisions(289, 145, 109)
    cluster.AutoAdjustNumberOfDivisionsOff()
    cluster.UseFeatureEdgesOn()
    cluster.UseFeaturePointsOn()
    cluster.SetFeaturePointsAngle(float(feature_angle))
    cluster.UseInputPointsOn()
    cluster.Update()
    result = vtk.vtkPolyData()
    result.ShallowCopy(cluster.GetOutput())
    return result


def validate_mesh(vtk, polydata, source_triangles: int) -> tuple[np.ndarray, int]:
    from vtk.util.numpy_support import vtk_to_numpy

    if polydata is None or polydata.GetPoints() is None or polydata.GetNumberOfPolys() <= 0:
        raise RuntimeError("Decimation produced an empty surface.")
    points = np.asarray(vtk_to_numpy(polydata.GetPoints().GetData()), dtype=np.float32)
    triangles = int(polydata.GetNumberOfPolys())
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise RuntimeError("Decimation produced invalid point coordinates.")
    if triangles >= source_triangles:
        raise RuntimeError("Decimation did not reduce the input mesh.")
    return np.ascontiguousarray(points), triangles


def original_triangle_count(output_dir: Path) -> int | None:
    """Reuse the original-mesh count already recorded by the KDE export."""
    path = output_dir / "kde16_export_summary.json"
    if not path.is_file():
        return None
    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
        triangles = int(summary["sources"]["original"]["triangles"])
    except (KeyError, OSError, TypeError, ValueError):
        return None
    return triangles if triangles > 0 else None


def main() -> None:
    args = parse_args()
    if int(args.knn_k) < 1:
        raise ValueError("--knn-k must be positive.")
    import vtk

    device = resolve_device(args.device)
    output_dir = Path(args.output_dir).expanduser().resolve()
    original_triangles = original_triangle_count(output_dir)
    dataset = AhmedMLDatasetV2(
        saved_folder=str(Path(args.data_root).expanduser().resolve()),
        if_test=True,
        geometry_points=0,
        surface_points=0,
        volume_points=0,
        scale_positions=False,
        require_preprocessed=True,
    )
    minimum = dataset.min_pos.detach().cpu().numpy().astype(np.float32)
    maximum = dataset.max_pos.detach().cpu().numpy().astype(np.float32)
    decimators = {
        "feature_aware": lambda poly: decimate_feature_aware(vtk, poly, args.feature_angle),
        "qem": lambda poly: decimate_qem(vtk, poly),
        "voxel_grid": lambda poly: decimate_voxel_grid(vtk, poly, args.feature_angle),
    }
    summary_path = output_dir / "div50_from_div10_kde16_summary.json"
    summary: dict[str, object] = {
        "run_id": int(args.run_id),
        "source_factor": 10,
        "target_factor": 50,
        "additional_reduction_factor": 5,
        "original_triangles": original_triangles,
        "knn_k": int(args.knn_k),
        "normalization": "global_drivaerml_training_bounds",
        "sources": {},
    }
    if summary_path.is_file():
        try:
            previous = json.loads(summary_path.read_text(encoding="utf-8"))
            if int(previous.get("run_id", -1)) == int(args.run_id):
                summary["sources"] = dict(previous.get("sources", {}))
        except (OSError, ValueError, TypeError):
            pass

    sources = source_paths(args)
    if str(args.sources).strip().lower() != "all":
        selected = [item.strip() for item in str(args.sources).split(",") if item.strip()]
        unknown = sorted(set(selected) - set(sources))
        if unknown:
            raise ValueError(f"Unknown --sources values: {unknown}. Available: {list(sources)}")
        sources = {label: sources[label] for label in selected}

    for label, source_path in sources.items():
        started = time.perf_counter()
        poly, _ = read_polydata(source_path)
        source = triangulated(vtk, poly)
        source_triangles = int(source.GetNumberOfPolys())
        output = decimators[label](source)
        points, triangles = validate_mesh(vtk, output, source_triangles)
        try:
            log_density = compute_kde16(points, minimum, maximum, int(args.knn_k), device)
        except RuntimeError as exc:
            if device.type != "cuda" or "KDE density estimation requires" not in str(exc):
                raise
            # CUDA tensor placement alone is insufficient: the KDE utility
            # requires torch-cluster for GPU kNN. Preserve the exact estimator
            # by falling back to its scikit-learn CPU implementation instead.
            print("[kde] torch-cluster unavailable; falling back to CPU kNN.", flush=True)
            log_density = compute_kde16(points, minimum, maximum, int(args.knn_k), resolve_device("cpu"))
        output_path = output_dir / f"drivaerml_run{args.run_id}_{label}_div50_from_div10_kde16.vtp"
        write_density_mesh(output, log_density, output_path)
        validate_written_vtp(output_path, points.shape[0])
        summary["sources"][label] = {
            "input": str(source_path),
            "output": str(output_path),
            "method": {
                "feature_aware": "vtkDecimatePro feature-preserving edge collapse",
                "qem": "vtkQuadricDecimation",
                "voxel_grid": "vtkQuadricClustering on calibrated total-div50 grid",
            }[label],
            "source_triangles": source_triangles,
            "triangles": triangles,
            "achieved_total_factor": float(
                (original_triangles if original_triangles is not None else source_triangles * 10.0)
                / max(triangles, 1)
            ),
            "points": int(points.shape[0]),
            "seconds": float(time.perf_counter() - started),
        }
        print(
            f"{label}: {source_triangles:,} -> {triangles:,} triangles "
            f"({summary['sources'][label]['achieved_total_factor']:.2f}x total) -> {output_path}",
            flush=True,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
