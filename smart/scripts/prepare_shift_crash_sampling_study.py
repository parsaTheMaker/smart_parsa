#!/usr/bin/env python3
"""Download and remesh a small SHIFT-Crash sampling study.

This utility prepares only encoder geometries.  The displacement targets and
query nodes remain in the existing SHIFT-Crash preprocessed root and are not
copied or modified.  For each selected case it writes div5/div10 outputs for:

* angle-based topology-preserving decimation (VTK DecimatePro),
* voxel/quadric clustering, and
* isotropic GPU remeshing (the CUDA voxel-centroid backend used by the
  repository's DrivAerML isotropic study).

The Hugging Face repository is gated.  Set ``HF_TOKEN`` in the environment
before running this script.  Raw downloads are retained by default so the
study can be inspected or regenerated without downloading again.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
SMART_ROOT = SCRIPT_DIR.parent
if str(SMART_ROOT) not in sys.path:
    sys.path.insert(0, str(SMART_ROOT))

from scripts.decimate_drivaerml_vtp import (  # noqa: E402
    geometry_only,
    process_input,
    read_polydata,
    require_vtk,
    triangulate_if_needed,
    write_vtp,
)


DEFAULT_REPO = "luminary-shift/SHIFT-Crash-sample"
DEFAULT_DATA_ROOT = Path("/mnt/ssdraid/parsa/shift_crash_preprocessed")
DEFAULT_DOWNLOAD_ROOT = Path("/mnt/ssdraid/parsa/shift_crash_surface_vtp_raw")
DEFAULT_STUDY_ROOT = Path("/mnt/ssdraid/parsa/shift_crash_surface_vtp_study")
DEFAULT_RESULTS_ROOT = Path("/home/parsa/smart_parsa/results/shift_crash_sampling_study")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=DEFAULT_REPO)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--download-dir", type=Path, default=DEFAULT_DOWNLOAD_ROOT)
    parser.add_argument("--study-root", type=Path, default=DEFAULT_STUDY_ROOT)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--num-cases", type=int, default=15)
    parser.add_argument(
        "--case-ids",
        default=None,
        help="Optional comma-separated case IDs. Otherwise the first available sample cases are selected.",
    )
    parser.add_argument(
        "--split",
        choices=("all", "train", "validation"),
        default="all",
        help="Restrict automatic selection to the existing preprocessed split; default keeps the first 15 sample cases.",
    )
    parser.add_argument("--factors", default="5,10")
    parser.add_argument("--feature-angle", type=float, default=30.0)
    parser.add_argument("--isotropic-device", default="cuda:7")
    parser.add_argument("--gpu-adjustments", type=int, default=3)
    parser.add_argument("--gpu-face-chunk-size", type=int, default=262144)
    parser.add_argument("--isotropic-iterations", type=int, default=5)
    parser.add_argument("--isotropic-iso-tries", type=int, default=3)
    parser.add_argument("--isotropic-seed-triangle-multiplier", type=float, default=1.25)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--keep-raw", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def parse_factors(text: str) -> list[int]:
    factors = sorted({int(item.strip()) for item in str(text).split(",") if item.strip()})
    if not factors or any(factor <= 1 for factor in factors):
        raise ValueError("--factors must contain integers greater than one.")
    return factors


def load_split_ids(data_root: Path, split: str) -> set[str]:
    if split == "all":
        return set()
    with (data_root / "splits.json").open("r", encoding="utf-8") as handle:
        splits = json.load(handle)
    return {str(case_id) for case_id in splits.get(split, ())}


def list_case_ids(repo_id: str, revision: str, token: str | None) -> list[str]:
    from huggingface_hub import HfApi

    entries = HfApi(token=token).list_repo_tree(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        recursive=True,
        expand=False,
    )
    cases = set()
    for entry in entries:
        path = str(getattr(entry, "path", ""))
        if path.startswith("yaris_x") and path.endswith("/merged_surfaces.vtp"):
            cases.add(path.split("/", 1)[0])
    return sorted(cases)


def select_cases(args: argparse.Namespace, available: list[str]) -> list[str]:
    requested = None
    if args.case_ids:
        requested = [item.strip() for item in str(args.case_ids).split(",") if item.strip()]
        unknown = sorted(set(requested) - set(available))
        if unknown:
            raise ValueError(f"Requested cases are not in the Hugging Face sample: {unknown}")
    else:
        split_ids = load_split_ids(args.data_root, args.split)
        local_ids = {
            path.name
            for path in (args.data_root / "cases").iterdir()
            if path.is_dir()
        }
        requested = [
            case_id
            for case_id in available
            if case_id in local_ids and (not split_ids or case_id in split_ids)
        ]
        requested = requested[: int(args.num_cases)]
    if len(requested) != int(args.num_cases) and not args.case_ids:
        raise ValueError(
            f"Only {len(requested)} cases satisfy the selection, but --num-cases={args.num_cases}. "
            "Use --split all or provide --case-ids explicitly."
        )
    missing = [case_id for case_id in requested if not (args.data_root / "cases" / case_id).is_dir()]
    if missing:
        raise FileNotFoundError(
            "The comparison needs preprocessed targets/features for every selected case. "
            f"Missing local cases: {missing}"
        )
    return requested


def download_case(repo_id: str, revision: str, case_id: str, download_dir: Path, token: str | None) -> Path:
    from huggingface_hub import hf_hub_download

    target = download_dir / case_id / "merged_surfaces.vtp"
    if target.is_file() and target.stat().st_size > 0:
        return target
    downloaded = hf_hub_download(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        filename=f"{case_id}/merged_surfaces.vtp",
        local_dir=str(download_dir),
        token=token,
    )
    downloaded = Path(downloaded)
    if downloaded.resolve() != target.resolve():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(downloaded, target)
    return target


def estimate_voxel_divisions(polydata, source_triangles: int, factor: int) -> tuple[int, int, int]:
    """Estimate a geometry-scaled grid instead of using DrivAerML divisions."""
    bounds = np.asarray(polydata.GetBounds(), dtype=np.float64).reshape(3, 2)
    extent = np.maximum(bounds[:, 1] - bounds[:, 0], 1.0e-9)
    aspect = extent / np.cbrt(np.prod(extent))
    target_cells = max(128.0, float(source_triangles) / float(factor))
    scale = np.cbrt(target_cells / max(float(np.prod(aspect)), 1.0e-12))
    divisions = np.maximum(np.rint(aspect * scale).astype(np.int64), 8)
    return tuple(int(value) for value in divisions)


def method_args(output_dir: Path, method: str, args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        output_dir=output_dir,
        method=method,
        feature_angle=float(args.feature_angle),
        isotropic_iterations=int(args.isotropic_iterations),
        isotropic_iso_tries=int(args.isotropic_iso_tries),
        isotropic_seed_triangle_multiplier=float(args.isotropic_seed_triangle_multiplier),
        gpu_device=str(args.isotropic_device),
        gpu_adjustments=int(args.gpu_adjustments),
        gpu_face_chunk_size=int(args.gpu_face_chunk_size),
        validate_output=False,
        validate_topology=False,
        overwrite=bool(args.overwrite),
    )


def write_legacy_vtk(vtk, polydata, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".partial")
    if temporary.exists():
        temporary.unlink()
    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(str(temporary))
    writer.SetInputData(geometry_only(vtk, polydata))
    writer.SetFileTypeToBinary()
    if writer.Write() != 1:
        raise RuntimeError(f"VTK failed to write {output_path}.")
    temporary.replace(output_path)


def process_case(case_id: str, source_path: Path, args: argparse.Namespace, vtk, vtk_to_numpy, factors: list[int]) -> dict:
    source = triangulate_if_needed(vtk, read_polydata(vtk, source_path))
    source_triangles = int(source.GetNumberOfPolys())
    outputs = {}
    records = []
    method_specs = (
        ("angle", "decimate_pro", args.study_root / "angle"),
        ("voxel", "voxel_quadric_clustering", args.study_root / "voxel"),
        ("isotropic", "isotropic_gpu", args.study_root / "isotropic"),
    )
    for label, method, output_dir in method_specs:
        config = method_args(output_dir, method, args)
        calibrated = {
            factor: estimate_voxel_divisions(source, source_triangles, factor)
            for factor in factors
        }
        records.extend(process_input(vtk, vtk_to_numpy, source_path, config, factors, calibrated))
        for factor in factors:
            output_path = output_dir / source_path.parent.name / f"{source_path.stem}_faces_div{factor}.vtp"
            if not output_path.is_file():
                raise FileNotFoundError(f"Remeshing reported success but output is missing: {output_path}")
            outputs[f"{label}_div{factor}"] = str(output_path)

    vtk_dir = args.results_dir / "representative_vtks"
    representative_keys = ["original"] + [f"{label}_div{factor}" for label, _method, _dir in method_specs for factor in factors]
    if case_id == args.selected_cases[0]:
        write_legacy_vtk(vtk, source, vtk_dir / f"{case_id}_original.vtk")
        for key in representative_keys[1:]:
            remeshed = read_polydata(vtk, Path(outputs[key]))
            write_legacy_vtk(vtk, remeshed, vtk_dir / f"{case_id}_{key}.vtk")
    return {
        "case_id": case_id,
        "source": str(source_path),
        "source_points": int(source.GetNumberOfPoints()),
        "source_triangles": source_triangles,
        "outputs": outputs,
        "records": records,
    }


def main() -> int:
    args = parse_args()
    if args.num_cases <= 0:
        raise ValueError("--num-cases must be positive.")
    if not args.data_root.is_dir():
        raise FileNotFoundError(f"SHIFT-Crash preprocessed root not found: {args.data_root}")
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    factors = parse_factors(args.factors)
    available = list_case_ids(args.repo_id, args.revision, token)
    args.selected_cases = select_cases(args, available)
    print(f"Selected cases ({len(args.selected_cases)}): {', '.join(args.selected_cases)}", flush=True)
    if args.dry_run:
        print(f"Download root: {args.download_dir}")
        print(f"Study root: {args.study_root}")
        print(f"Factors: {factors}; isotropic device: {args.isotropic_device}")
        return 0
    if not token:
        raise RuntimeError("The dataset is gated. Export HF_TOKEN before downloading the VTP files.")

    args.download_dir.mkdir(parents=True, exist_ok=True)
    args.study_root.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    vtk, vtk_to_numpy = require_vtk()
    all_cases = []
    wall_start = time.perf_counter()
    for index, case_id in enumerate(args.selected_cases, start=1):
        print(f"[{index}/{len(args.selected_cases)}] downloading {case_id}", flush=True)
        source_path = download_case(args.repo_id, args.revision, case_id, args.download_dir, token)
        print(f"[{index}/{len(args.selected_cases)}] remeshing {case_id} from {source_path}", flush=True)
        all_cases.append(process_case(case_id, source_path, args, vtk, vtk_to_numpy, factors))

    summary = {
        "repo_id": args.repo_id,
        "revision": args.revision,
        "data_root": str(args.data_root.resolve()),
        "download_dir": str(args.download_dir.resolve()),
        "study_root": str(args.study_root.resolve()),
        "results_dir": str(args.results_dir.resolve()),
        "cases": args.selected_cases,
        "factors": factors,
        "methods": ["angle", "voxel", "isotropic"],
        "isotropic_backend": "cuda_voxel_centroid",
        "isotropic_device": args.isotropic_device,
        "wall_seconds": time.perf_counter() - wall_start,
        "records": all_cases,
    }
    summary_path = args.results_dir / "shift_crash_sampling_study_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if not args.keep_raw:
        shutil.rmtree(args.download_dir)
        print(f"Removed raw downloads: {args.download_dir}", flush=True)
    print(f"Summary: {summary_path}", flush=True)
    print(f"Finished in {summary['wall_seconds'] / 3600.0:.2f} h", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
