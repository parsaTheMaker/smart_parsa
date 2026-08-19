#!/usr/bin/env python3
"""Export and remesh toy heat-exchange FEM surfaces for sampling tests.

The source VTPs are the exact adaptive boundary meshes stored by the FEM
generator.  Every method changes only encoder geometry; the reference query
coordinates and targets in the preprocessed benchmark remain untouched.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import signal
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from tqdm.auto import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
SMART_ROOT = SCRIPT_DIR.parent
if str(SMART_ROOT) not in sys.path:
    sys.path.insert(0, str(SMART_ROOT))

from scripts.decimate_drivaerml_vtp import (  # noqa: E402
    geometry_only,
    process_input,
    require_vtk,
    write_vtp,
)


def parse_csv_ints(value: str) -> list[int]:
    values = sorted({int(item.strip()) for item in str(value).split(",") if item.strip()})
    if not values or any(item <= 1 for item in values):
        raise ValueError("--factors must contain unique integers greater than one.")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("/mnt/ssdraid/parsa/toy_heat_exchange_fem_v1"))
    parser.add_argument("--surface-vtp-dir", type=Path, default=Path("/mnt/ssdraid/parsa/toy_heat_exchange_surface_vtp"))
    parser.add_argument("--angle-output-dir", type=Path, default=Path("/mnt/ssdraid/parsa/toy_heat_exchange_surface_vtp_angle"))
    parser.add_argument("--isotropic-output-dir", type=Path, default=Path("/mnt/ssdraid/parsa/toy_heat_exchange_surface_vtp_isotropic"))
    parser.add_argument("--voxel-output-dir", type=Path, default=Path("/mnt/ssdraid/parsa/toy_heat_exchange_surface_vtp_voxel"))
    parser.add_argument("--results-dir", type=Path, default=Path("/home/parsa/smart_parsa/results/toy_heat_exchange_remeshing"))
    parser.add_argument(
        "--case-stem",
        default="heat_exchange_case",
        help="Prefix of per-case heat-exchange surface VTPs.",
    )
    parser.add_argument("--factors", default="5,10")
    parser.add_argument("--max-cases", type=int, default=0, help="Validation cases to remesh; 0 means all.")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--feature-angle", type=float, default=30.0)
    parser.add_argument("--isotropic-iterations", type=int, default=8)
    parser.add_argument("--isotropic-iso-tries", type=int, default=5)
    parser.add_argument("--example-count", type=int, default=3, help="Number of validation cases to export as visual remeshing galleries.")
    parser.add_argument("--example-max-triangles", type=int, default=8000, help="Maximum triangles drawn per gallery panel; source VTPs remain unchanged.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def source_path(root: Path, case_id: int, case_stem: str) -> Path:
    return root / f"case_{case_id:05d}" / f"{case_stem}_{case_id:05d}_surface.vtp"


def export_surface_vtp(data_root: Path, surface_root: Path, case_id: int, case_stem: str, overwrite: bool) -> Path:
    path = source_path(surface_root, case_id, case_stem)
    if path.is_file() and not overwrite:
        return path
    vtk, _vtk_to_numpy = require_vtk()
    from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray

    case_dir = data_root / f"case_{case_id:05d}"
    points = np.asarray(np.load(case_dir / "surface_mesh_points.npy", mmap_mode="r"), dtype=np.float32)
    faces = np.asarray(np.load(case_dir / "surface_mesh_faces.npy", mmap_mode="r"), dtype=np.int64)
    if points.ndim != 2 or points.shape[1] != 3 or faces.ndim != 2 or faces.shape[1] != 3:
        raise RuntimeError(f"Malformed FEM surface arrays for case {case_id}.")
    poly = vtk.vtkPolyData()
    vtk_points = vtk.vtkPoints()
    vtk_points.SetData(numpy_to_vtk(np.ascontiguousarray(points), deep=True))
    poly.SetPoints(vtk_points)
    offsets = np.arange(0, 3 * faces.shape[0] + 1, 3, dtype=np.int64)
    cells = vtk.vtkCellArray()
    cells.SetData(
        numpy_to_vtkIdTypeArray(offsets, deep=True),
        numpy_to_vtkIdTypeArray(np.ascontiguousarray(faces.reshape(-1)), deep=True),
    )
    poly.SetPolys(cells)
    poly.BuildCells()
    write_vtp(vtk, geometry_only(vtk, poly), path)
    return path


def estimate_voxel_divisions(polydata, source_triangles: int, factor: int) -> tuple[int, int, int]:
    bounds = np.asarray(polydata.GetBounds(), dtype=np.float64).reshape(3, 2)
    extent = np.maximum(bounds[:, 1] - bounds[:, 0], 1.0e-9)
    aspect = extent / np.cbrt(np.prod(extent))
    target_cells = max(128.0, float(source_triangles) / float(factor))
    scale = np.cbrt(target_cells / max(float(np.prod(aspect)), 1.0e-12))
    return tuple(int(value) for value in np.maximum(np.rint(aspect * scale), 8))


def method_args(output_dir: Path, method: str, args: dict) -> SimpleNamespace:
    return SimpleNamespace(
        output_dir=output_dir,
        method=method,
        feature_angle=float(args["feature_angle"]),
        isotropic_iterations=int(args["isotropic_iterations"]),
        isotropic_iso_tries=int(args["isotropic_iso_tries"]),
        isotropic_seed_triangle_multiplier=1.5,
        gpu_device="cuda:0",
        gpu_adjustments=3,
        gpu_face_chunk_size=262144,
        validate_output=True,
        validate_topology=True,
        repair_degenerate_triangles=True,
        overwrite=bool(args["overwrite"]),
    )


def remesh_case(payload: tuple[int, str, dict, list[int]]) -> dict:
    case_id, source, args, factors = payload
    vtk, vtk_to_numpy = require_vtk()
    from scripts.decimate_drivaerml_vtp import read_polydata, triangulate_if_needed

    source_path_value = Path(source)
    polydata = triangulate_if_needed(vtk, read_polydata(vtk, source_path_value))
    source_triangles = int(polydata.GetNumberOfPolys())
    specs = (
        ("angle", "decimate_pro", Path(args["angle_output_dir"])),
        ("isotropic", "isotropic_remeshing", Path(args["isotropic_output_dir"])),
        ("voxel", "voxel_quadric_clustering", Path(args["voxel_output_dir"])),
    )
    records = []
    for label, method, output_dir in specs:
        calibrated = {factor: estimate_voxel_divisions(polydata, source_triangles, factor) for factor in factors}
        records.extend(
            process_input(
                vtk,
                vtk_to_numpy,
                source_path_value,
                method_args(output_dir, method, args),
                factors,
                calibrated,
            )
        )
    return {"case_id": case_id, "source": str(source_path_value), "source_triangles": source_triangles, "records": records}


def ignore_sigint() -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def export_remeshing_examples(args: argparse.Namespace, case_ids: list[int], factors: list[int]) -> list[dict]:
    """Create inspectable mesh galleries without duplicating the large VTP data."""
    if args.example_count <= 0:
        return []
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from scripts.decimate_drivaerml_vtp import read_polydata

    vtk, vtk_to_numpy = require_vtk()
    examples_root = args.results_dir.expanduser().resolve() / "examples"
    examples_root.mkdir(parents=True, exist_ok=True)
    exported = []
    for case_id in case_ids[: int(args.example_count)]:
        case_dir = examples_root / f"case_{case_id:05d}"
        case_dir.mkdir(parents=True, exist_ok=True)
        source = source_path(args.surface_vtp_dir.expanduser().resolve(), case_id, args.case_stem)
        panels = [("Original adaptive mesh", source)]
        for method, directory in (
            ("Angle", args.angle_output_dir),
            ("Isotropic", args.isotropic_output_dir),
            ("Voxel", args.voxel_output_dir),
        ):
            for factor in factors:
                path = Path(directory).expanduser().resolve() / f"case_{case_id:05d}" / f"{source.stem}_faces_div{factor}.vtp"
                if not path.is_file():
                    raise FileNotFoundError(f"Missing remeshed VTP for example export: {path}")
                panels.append((f"{method} {factor}x", path))

        sources = []
        fig = plt.figure(figsize=(16, 9), constrained_layout=True)
        for panel_index, (label, path) in enumerate(panels, start=1):
            polydata = read_polydata(vtk, path)
            points = np.asarray(vtk_to_numpy(polydata.GetPoints().GetData()), dtype=np.float32)
            raw = np.asarray(vtk_to_numpy(polydata.GetPolys().GetData()), dtype=np.int64)
            faces = raw.reshape(-1, 4)[:, 1:]
            if faces.shape[0] > int(args.example_max_triangles):
                display_ids = np.linspace(0, faces.shape[0] - 1, int(args.example_max_triangles), dtype=np.int64)
                faces = faces[display_ids]
            axis = fig.add_subplot(2, 4, panel_index, projection="3d")
            mesh = Poly3DCollection(points[faces], facecolor="#75aadb", edgecolor="#274c77", linewidth=0.05, alpha=0.94)
            axis.add_collection3d(mesh)
            lower, upper = points.min(axis=0), points.max(axis=0)
            center, radius = 0.5 * (lower + upper), 0.55 * float(np.max(upper - lower))
            axis.set_xlim(center[0] - radius, center[0] + radius)
            axis.set_ylim(center[1] - radius, center[1] + radius)
            axis.set_zlim(center[2] - radius, center[2] + radius)
            axis.set_box_aspect((1, 1, 1))
            axis.view_init(elev=18, azim=-62)
            axis.set_axis_off()
            axis.set_title(label, fontsize=13, fontweight="bold", pad=4)
            link = case_dir / f"{panel_index - 1:02d}_{label.lower().replace(' ', '_')}.vtp"
            if link.exists() or link.is_symlink():
                link.unlink()
            link.symlink_to(path)
            sources.append({"label": label, "vtp": str(path), "result_link": str(link)})
        png_path = case_dir / f"{args.case_stem}_{case_id:05d}_remeshing_gallery.png"
        fig.savefig(png_path, dpi=220, facecolor="white")
        plt.close(fig)
        (case_dir / "geometry_sources.json").write_text(json.dumps(sources, indent=2) + "\n", encoding="utf-8")
        exported.append({"case_id": case_id, "gallery": str(png_path), "sources": sources})
    return exported


def main() -> int:
    args = parse_args()
    if args.workers <= 0:
        raise ValueError("--workers must be positive.")
    if args.feature_angle <= 0.0 or args.feature_angle >= 180.0:
        raise ValueError("--feature-angle must be in (0, 180).")
    factors = parse_csv_ints(args.factors)
    data_root = args.data_root.expanduser().resolve()
    manifest = json.loads((data_root / "preprocessed_manifest.json").read_text(encoding="utf-8"))
    case_ids = [int(case_id) for case_id in manifest["validation_ids"]]
    if args.max_cases > 0:
        case_ids = case_ids[: args.max_cases]
    if not case_ids:
        raise ValueError("No validation cases selected for remeshing.")

    surface_root = args.surface_vtp_dir.expanduser().resolve()
    for directory in (surface_root, args.angle_output_dir, args.isotropic_output_dir, args.voxel_output_dir, args.results_dir):
        directory.expanduser().resolve().mkdir(parents=True, exist_ok=True)
    print(f"Exporting {len(case_ids)} FEM boundary meshes.", flush=True)
    source_paths = [
        export_surface_vtp(data_root, surface_root, case_id, args.case_stem, args.overwrite)
        for case_id in case_ids
    ]

    shared_args = {
        "angle_output_dir": str(args.angle_output_dir.expanduser().resolve()),
        "isotropic_output_dir": str(args.isotropic_output_dir.expanduser().resolve()),
        "voxel_output_dir": str(args.voxel_output_dir.expanduser().resolve()),
        "feature_angle": float(args.feature_angle),
        "isotropic_iterations": int(args.isotropic_iterations),
        "isotropic_iso_tries": int(args.isotropic_iso_tries),
        "overwrite": bool(args.overwrite),
    }
    payloads = [(case_id, str(path), shared_args, factors) for case_id, path in zip(case_ids, source_paths)]
    context = mp.get_context("spawn")
    records = []
    started = time.perf_counter()
    executor = ProcessPoolExecutor(max_workers=args.workers, mp_context=context, initializer=ignore_sigint)
    try:
        futures = [executor.submit(remesh_case, payload) for payload in payloads]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Remeshing toy FEM surfaces"):
            records.append(future.result())
    except KeyboardInterrupt:
        executor.shutdown(wait=False, cancel_futures=True)
        raise SystemExit(130)
    finally:
        executor.shutdown(wait=True, cancel_futures=False)
    records.sort(key=lambda record: int(record["case_id"]))
    summary = {
        "data_root": str(data_root), "surface_vtp_dir": str(surface_root),
        "factors": factors, "methods": ["angle", "isotropic", "voxel"],
        "source_role": "actual_adaptive_fem_boundary_mesh",
        "target_role": "unchanged area-uniform surface and tetra-volume query arrays",
        "wall_seconds": time.perf_counter() - started, "records": records,
    }
    summary["examples"] = export_remeshing_examples(args, case_ids, factors)
    summary_path = args.results_dir.expanduser().resolve() / f"{args.case_stem}_remeshing_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Summary: {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
