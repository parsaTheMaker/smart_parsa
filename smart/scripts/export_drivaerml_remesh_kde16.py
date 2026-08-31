#!/usr/bin/env python3
"""Export mesh-vertex KDE-16 density for one DrivAerML surface and remesh set.

Each output retains its input triangle connectivity and contains KDE-16 point
arrays computed on that output mesh's vertices.  Coordinates are normalized
with the global DrivAerML training bounds before density estimation, matching
the sampling protocol used by SMART.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
SMART_ROOT = SCRIPT_DIR.parent
if str(SMART_ROOT) not in sys.path:
    sys.path.insert(0, str(SMART_ROOT))

from data.ahmedml_dataset_v2 import AhmedMLDatasetV2
from utils.geometry_density import estimate_log_sampling_density


DEFAULT_DATA_ROOT = "/mnt/ssdraid/parsa/drivaerml_preprocessed"
DEFAULT_ORIGINAL_ROOT = "/mnt/ssdraid/parsa/drivaerml_surface_vtp"
DEFAULT_FEATURE_ROOT = "/mnt/ssdraid/parsa/drivaerml_surface_vtp_remesh_v4/feature"
DEFAULT_QEM_ROOT = "/mnt/ssdraid/parsa/drivaerml_surface_vtp_remesh_v4/quadric"
DEFAULT_VOXEL_ROOT = "/mnt/ssdraid/parsa/drivaerml_surface_vtp_remesh_v4/voxel"
DEFAULT_OUTPUT_DIR = "/home/parsa/smart_parsa/results/final/drivaerml_run29_mesh_kde16"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", type=int, default=29)
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--original-vtp-root", default=DEFAULT_ORIGINAL_ROOT)
    parser.add_argument("--feature-vtp-root", default=DEFAULT_FEATURE_ROOT)
    parser.add_argument("--qem-vtp-root", default=DEFAULT_QEM_ROOT)
    parser.add_argument("--voxel-vtp-root", default=DEFAULT_VOXEL_ROOT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--knn-k", type=int, default=16)
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument(
        "--sources",
        default="all",
        help="Comma-separated source labels to export, or 'all' (default).",
    )
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda was requested but CUDA is unavailable.")
        return torch.device("cuda")
    return torch.device("cpu")


def source_paths(run_id: int, args: argparse.Namespace) -> dict[str, Path]:
    stem = f"drivaer_{run_id}"
    roots = {
        "original": Path(args.original_vtp_root) / f"run_{run_id}" / f"{stem}.vtp",
        "feature_aware_div5": Path(args.feature_vtp_root) / f"run_{run_id}" / f"{stem}_faces_div5.vtp",
        "feature_aware_div10": Path(args.feature_vtp_root) / f"run_{run_id}" / f"{stem}_faces_div10.vtp",
        "qem_div5": Path(args.qem_vtp_root) / f"run_{run_id}" / f"{stem}_faces_div5.vtp",
        "qem_div10": Path(args.qem_vtp_root) / f"run_{run_id}" / f"{stem}_faces_div10.vtp",
        "voxel_grid_div5": Path(args.voxel_vtp_root) / f"run_{run_id}" / f"{stem}_faces_div5.vtp",
        "voxel_grid_div10": Path(args.voxel_vtp_root) / f"run_{run_id}" / f"{stem}_faces_div10.vtp",
    }
    missing = [str(path) for path in roots.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required input VTPs:\n" + "\n".join(missing))
    return roots


def read_polydata(path: Path):
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy

    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(str(path))
    reader.Update()
    poly = reader.GetOutput()
    if poly is None or poly.GetPoints() is None or poly.GetNumberOfPoints() < 2:
        raise RuntimeError(f"Invalid or empty VTP: {path}")
    if poly.GetNumberOfPolys() <= 0:
        raise RuntimeError(f"VTP contains no surface polygons: {path}")
    points = np.asarray(vtk_to_numpy(poly.GetPoints().GetData()), dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise RuntimeError(f"Invalid point coordinates in {path}")
    return poly, np.ascontiguousarray(points)


def compute_kde16(points: np.ndarray, minimum: np.ndarray, maximum: np.ndarray, knn_k: int, device: torch.device) -> np.ndarray:
    span = np.maximum(maximum - minimum, 1.0e-12)
    normalized = (points - minimum[None, :]) / span[None, :]
    if not np.isfinite(normalized).all():
        raise RuntimeError("Non-finite coordinates after DrivAerML global normalization.")
    point_tensor = torch.from_numpy(np.ascontiguousarray(normalized, dtype=np.float32)).unsqueeze(0).to(device)
    try:
        log_density = estimate_log_sampling_density(point_tensor, knn_k=int(knn_k), estimator="kde")
        result = log_density.squeeze(0).float().cpu().numpy()
    finally:
        del point_tensor
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if result.shape != (points.shape[0],) or not np.isfinite(result).all():
        raise RuntimeError("KDE-16 returned invalid point density values.")
    return np.ascontiguousarray(result, dtype=np.float32)


def write_density_mesh(poly, log_density: np.ndarray, output_path: Path) -> None:
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk

    if int(poly.GetNumberOfPoints()) != int(log_density.shape[0]):
        raise ValueError("Density length does not match mesh point count.")
    # The reader-owned polydata is no longer needed after each export. Clearing
    # old arrays keeps the output focused on the density fields and avoids
    # duplicating unrelated source fields.
    poly.GetPointData().Initialize()
    density = np.exp(np.clip(log_density.astype(np.float64), -80.0, 80.0)).astype(np.float32)
    arrays = {
        "kde16_log_density": log_density,
        "kde16_density": density,
    }
    for name, values in arrays.items():
        array = numpy_to_vtk(np.ascontiguousarray(values), deep=True)
        array.SetName(name)
        poly.GetPointData().AddArray(array)
    poly.GetPointData().SetActiveScalars("kde16_density")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial = output_path.with_suffix(output_path.suffix + ".partial")
    if partial.exists():
        partial.unlink()
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(str(partial))
    writer.SetInputData(poly)
    writer.SetDataModeToBinary()
    writer.SetCompressor(None)
    if hasattr(writer, "SetHeaderTypeToUInt32"):
        writer.SetHeaderTypeToUInt32()
    if writer.Write() != 1:
        raise RuntimeError(f"Failed to write VTP: {output_path}")
    partial.replace(output_path)


def validate_written_vtp(path: Path, expected_points: int) -> None:
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy

    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(str(path))
    reader.Update()
    poly = reader.GetOutput()
    if poly is None or poly.GetNumberOfPoints() != int(expected_points):
        raise RuntimeError(f"Written VTP failed validation: {path}")
    values = poly.GetPointData().GetArray("kde16_log_density")
    if values is None:
        raise RuntimeError(f"Written VTP has no kde16_log_density array: {path}")
    density = np.asarray(vtk_to_numpy(values), dtype=np.float32)
    if density.shape[0] != int(expected_points) or not np.isfinite(density).all():
        raise RuntimeError(f"Written VTP has invalid KDE values: {path}")


def main() -> None:
    args = parse_args()
    if args.knn_k < 1:
        raise ValueError("--knn-k must be positive.")
    device = resolve_device(args.device)
    output_dir = Path(args.output_dir).expanduser().resolve()
    sources = source_paths(int(args.run_id), args)
    if str(args.sources).strip().lower() != "all":
        requested = [item.strip() for item in str(args.sources).split(",") if item.strip()]
        unknown = sorted(set(requested) - set(sources))
        if unknown:
            raise ValueError(f"Unknown --sources values: {unknown}. Available: {list(sources)}")
        sources = {name: sources[name] for name in requested}

    # Read the exact global bounds used by the DrivAerML data loader.
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
    summary: dict[str, object] = {
        "run_id": int(args.run_id),
        "estimator": "kde",
        "knn_k": int(args.knn_k),
        "normalization": "global_drivaerml_training_bounds",
        "global_min": minimum.tolist(),
        "global_max": maximum.tolist(),
        "sources": {},
    }

    for label, source_path in sources.items():
        started = time.perf_counter()
        print(f"{label}: reading {source_path}", flush=True)
        poly, points = read_polydata(source_path)
        print(
            f"{label}: computing KDE-{args.knn_k} on {points.shape[0]:,} mesh vertices "
            f"using {device}",
            flush=True,
        )
        log_density = compute_kde16(points, minimum, maximum, args.knn_k, device)
        output_path = output_dir / f"drivaerml_run{args.run_id}_{label}_kde16.vtp"
        print(f"{label}: writing {output_path}", flush=True)
        write_density_mesh(poly, log_density, output_path)
        validate_written_vtp(output_path, points.shape[0])
        elapsed = time.perf_counter() - started
        summary["sources"][label] = {
            "input": str(source_path),
            "output": str(output_path),
            "points": int(points.shape[0]),
            "triangles": int(poly.GetNumberOfPolys()),
            "seconds": float(elapsed),
            "log_density_min": float(log_density.min()),
            "log_density_max": float(log_density.max()),
            "log_density_mean": float(log_density.mean()),
        }
        print(f"{label}: {points.shape[0]:,} points, {poly.GetNumberOfPolys():,} triangles, {elapsed:.1f}s -> {output_path}", flush=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "kde16_export_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(sources)} KDE-16 mesh VTP(s) to {output_dir}")


if __name__ == "__main__":
    main()
