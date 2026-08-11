#!/usr/bin/env python3
"""Download and remesh a small SHIFT-Submarine sampling study.

Only the surface solution VTP is downloaded.  The preprocessed submarine
root remains the source of the fixed query coordinates and target fields.
The remeshed files therefore change the encoder point distribution without
changing the prediction target or its normalization.

The three geometry sources are:

* ``angle``: topology-preserving VTK DecimatePro;
* ``voxel``: VTK uniform-grid quadric clustering;
* ``isotropic``: the repository CUDA voxel-centroid remesher.

The Hugging Face repository is gated.  Export ``HF_TOKEN`` before running
the actual download; the token is never written to the output metadata.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace

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


DEFAULT_REPO = "luminary-shift/Submarine-sample"
DEFAULT_DATA_ROOT = Path("/mnt/ssdraid/parsa/shift_submarine_sample_preprocessed")
DEFAULT_DOWNLOAD_ROOT = Path("/mnt/ssdraid/parsa/shift_submarine_surface_vtp_raw")
DEFAULT_STUDY_ROOT = Path("/mnt/ssdraid/parsa/shift_submarine_surface_vtp_study")
DEFAULT_RESULTS_ROOT = Path("/home/parsa/smart_parsa/results/shift_submarine_sampling_study")
CASE_RE = re.compile(r"^sample_(\d{6})$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=DEFAULT_REPO)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--download-dir", type=Path, default=DEFAULT_DOWNLOAD_ROOT)
    parser.add_argument("--study-root", type=Path, default=DEFAULT_STUDY_ROOT)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument(
        "--summary-name",
        default="shift_submarine_sampling_study_summary.json",
        help="Filename for the generated study summary inside --results-dir.",
    )
    parser.add_argument("--num-cases", type=int, default=15)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument(
        "--case-ids",
        default=None,
        help="Optional comma-separated sample IDs such as sample_000003,sample_000017.",
    )
    parser.add_argument("--factors", default="5,10")
    parser.add_argument("--feature-angle", type=float, default=30.0)
    parser.add_argument("--isotropic-device", default="cuda:0")
    parser.add_argument("--gpu-adjustments", type=int, default=3)
    parser.add_argument("--gpu-face-chunk-size", type=int, default=262144)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--keep-raw", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--merge-existing-summary",
        action="store_true",
        help="Merge newly processed records into the existing summary instead of replacing it.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def parse_factors(value: str) -> list[int]:
    factors = sorted({int(item.strip()) for item in str(value).split(",") if item.strip()})
    if not factors or any(factor <= 1 for factor in factors):
        raise ValueError("--factors must contain integers greater than one.")
    return factors


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
        parts = path.split("/", 1)
        if len(parts) == 2 and CASE_RE.match(parts[0]) and parts[1] == "merged_surfaces.vtp":
            cases.add(parts[0])
    return sorted(cases)


def local_run_id(case_id: str, data_root: Path) -> int:
    match = CASE_RE.match(case_id)
    if match is None:
        raise ValueError(f"Unexpected submarine sample ID: {case_id}")
    run_id = int(match.group(1))
    run_dir = data_root / f"run_{run_id}"
    required = (
        run_dir / "_COMPLETE.json",
        run_dir / "surface_coords.npy",
        run_dir / "surface_data.npy",
        run_dir / "volume_coords.npy",
        run_dir / "volume_data.npy",
    )
    if not all(path.is_file() for path in required):
        raise FileNotFoundError(f"Missing complete preprocessed data for {case_id}: {run_dir}")
    return run_id


def select_cases(args: argparse.Namespace, available: list[str]) -> list[str]:
    if args.case_ids:
        cases = [item.strip() for item in str(args.case_ids).split(",") if item.strip()]
        unknown = sorted(set(cases) - set(available))
        if unknown:
            raise ValueError(f"Requested cases are not in the Hugging Face sample: {unknown}")
    else:
        start = max(0, int(args.start_index))
        cases = available[start : start + int(args.num_cases)]
    if len(cases) != int(args.num_cases) and not args.case_ids:
        raise ValueError(f"Only {len(cases)} cases available for --num-cases={args.num_cases}.")
    for case_id in cases:
        local_run_id(case_id, args.data_root)
    return cases


def download_surface(repo_id: str, revision: str, case_id: str, download_dir: Path, token: str | None) -> Path:
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
    import numpy as np

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
        isotropic_iterations=5,
        isotropic_iso_tries=3,
        isotropic_seed_triangle_multiplier=1.25,
        gpu_device=str(args.isotropic_device),
        gpu_adjustments=int(args.gpu_adjustments),
        gpu_face_chunk_size=int(args.gpu_face_chunk_size),
        validate_output=False,
        validate_topology=False,
        overwrite=bool(args.overwrite),
    )


def write_legacy_vtk(vtk, polydata, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial = output_path.with_suffix(output_path.suffix + ".partial")
    if partial.exists():
        partial.unlink()
    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(str(partial))
    writer.SetInputData(geometry_only(vtk, polydata))
    writer.SetFileTypeToBinary()
    if writer.Write() != 1:
        raise RuntimeError(f"VTK failed to write {output_path}")
    partial.replace(output_path)


def process_case(case_id: str, source_path: Path, args: argparse.Namespace, vtk, vtk_to_numpy, factors: list[int], representative: bool) -> dict:
    source = triangulate_if_needed(vtk, read_polydata(vtk, source_path))
    source_triangles = int(source.GetNumberOfPolys())
    method_specs = (
        ("angle", "decimate_pro", args.study_root / "angle"),
        ("voxel", "voxel_quadric_clustering", args.study_root / "voxel"),
        ("isotropic", "isotropic_gpu", args.study_root / "isotropic"),
    )
    outputs: dict[str, str] = {}
    records: list[dict] = []
    for label, method, output_dir in method_specs:
        config = method_args(output_dir, method, args)
        calibrated = {factor: estimate_voxel_divisions(source, source_triangles, factor) for factor in factors}
        records.extend(process_input(vtk, vtk_to_numpy, source_path, config, factors, calibrated))
        for factor in factors:
            output_path = output_dir / source_path.parent.name / f"{source_path.stem}_faces_div{factor}.vtp"
            if not output_path.is_file():
                raise FileNotFoundError(f"Missing remeshed output: {output_path}")
            key = f"{label}_div{factor}"
            outputs[key] = str(output_path)

    if representative:
        vtk_dir = args.results_dir / "representative_vtks"
        write_legacy_vtk(vtk, source, vtk_dir / f"{case_id}_original.vtk")
        for key, path in outputs.items():
            write_legacy_vtk(vtk, read_polydata(vtk, Path(path)), vtk_dir / f"{case_id}_{key}.vtk")
    return {
        "case_id": case_id,
        "run_id": local_run_id(case_id, args.data_root),
        "source": str(source_path),
        "source_points": int(source.GetNumberOfPoints()),
        "source_triangles": source_triangles,
        "outputs": outputs,
        "records": records,
    }


def main() -> int:
    args = parse_args()
    if args.num_cases <= 0:
        raise ValueError("--num-cases must be positive")
    if not args.data_root.is_dir():
        raise FileNotFoundError(f"SHIFT-Submarine preprocessed root not found: {args.data_root}")
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    factors = parse_factors(args.factors)
    available = list_case_ids(args.repo_id, args.revision, token)
    selected = select_cases(args, available)
    print(f"Selected samples ({len(selected)}): {', '.join(selected)}", flush=True)
    print(f"Local target root: {args.data_root}", flush=True)
    print(f"Factors: {factors}; isotropic device: {args.isotropic_device}", flush=True)
    if args.dry_run:
        return 0
    if not token:
        raise RuntimeError("The dataset is gated. Export HF_TOKEN before downloading the VTP files.")

    args.download_dir.mkdir(parents=True, exist_ok=True)
    args.study_root.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    vtk, vtk_to_numpy = require_vtk()
    records = []
    started = time.perf_counter()
    for index, case_id in enumerate(selected, start=1):
        print(f"[{index}/{len(selected)}] downloading {case_id}", flush=True)
        source_path = download_surface(args.repo_id, args.revision, case_id, args.download_dir, token)
        print(f"[{index}/{len(selected)}] remeshing {case_id}", flush=True)
        records.append(process_case(case_id, source_path, args, vtk, vtk_to_numpy, factors, index == 1))

    summary = {
        "repo_id": args.repo_id,
        "revision": args.revision,
        "data_root": str(args.data_root.resolve()),
        "download_dir": str(args.download_dir.resolve()),
        "study_root": str(args.study_root.resolve()),
        "results_dir": str(args.results_dir.resolve()),
        "cases": selected,
        "factors": factors,
        "methods": ["angle", "voxel", "isotropic"],
        "isotropic_backend": "cuda_voxel_centroid",
        "isotropic_device": args.isotropic_device,
        "source_role": "encoder_geometry_only",
        "targets_role": "existing_preprocessed_surface_and_volume_arrays",
        "wall_seconds": time.perf_counter() - started,
        "records": records,
    }
    summary_path = args.results_dir / str(args.summary_name)
    if args.merge_existing_summary and summary_path.is_file():
        existing = json.loads(summary_path.read_text(encoding="utf-8"))
        merged_by_case = {
            str(record["case_id"]): record
            for record in existing.get("records", [])
            if record.get("case_id") is not None
        }
        merged_by_case.update({str(record["case_id"]): record for record in records})
        summary["cases"] = sorted(merged_by_case)
        summary["records"] = [merged_by_case[case_id] for case_id in summary["cases"]]
        summary["wall_seconds"] = float(existing.get("wall_seconds", 0.0)) + float(summary["wall_seconds"])
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if not args.keep_raw:
        shutil.rmtree(args.download_dir)
        print(f"Removed raw downloads: {args.download_dir}", flush=True)
    print(f"Summary: {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
