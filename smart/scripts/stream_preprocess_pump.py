#!/usr/bin/env python3
"""Stream, inspect, and preprocess the gated SHIFT-Pump sample dataset.

Pump cases contain a dense surface VTP and a large volume VTU.  This converter
keeps the solver fields that are actually present in the files:

* surface: pressure, velocity_x/y/z, wall_shear_x/y/z;
* volume: pressure, velocity_x/y/z.

The surface is retained up to ``--surface-points`` and the volume is sampled
uniformly to ``--volume-points``.  KDE k=16 is computed on the retained
surface only and saved with the exact surface point indexing used by the
training adapter.  Raw files are staged per worker and removed after each
case, so the download is bounded by the largest active case rather than the
whole dataset.

The 13 varying numeric Pump parameters are preserved and later normalized as
conditioning inputs.  Constants, strings, and null metadata are deliberately
not passed to the model.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import gc
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pyvista as pv
import torch

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

SMART_ROOT = Path(__file__).resolve().parents[1]
if str(SMART_ROOT) not in sys.path:
    sys.path.insert(0, str(SMART_ROOT))

from scripts.stream_preprocess_shift_submarine import (  # noqa: E402
    _atomic_json,
    _atomic_save_npy,
    _density_cache,
    _download_inputs,
    _load_json,
    _sample_indices,
    _sample_number,
    list_samples,
)
DEFAULT_REPO_ID = "luminary-shift/Pump-sample"
DEFAULT_REVISION = "main"
DEFAULT_OUTPUT_DIR = "/mnt/ssdraid/parsa/shift_pump_preprocessed"
DEFAULT_RESULTS_DIR = "/home/parsa/smart_parsa/results/shift_pump_preprocess"
DEFAULT_STAGING_DIR = "/mnt/ssdraid/parsa/.shift_pump_staging"
CACHE_VERSION = "pump_v1"
SURFACE_FIELDS = (
    "pressure",
    "velocity_x",
    "velocity_y",
    "velocity_z",
    "wall_shear_x",
    "wall_shear_y",
    "wall_shear_z",
)
VOLUME_FIELDS = ("pressure", "velocity_x", "velocity_y", "velocity_z")
PARAMETER_KEYS = (
    "flow_rate",
    "flow_rate_op_condition",
    "head",
    "outer_diameter_factor",
    "outlet_width_factor",
    "shroud_diameter_factor",
    "compactness",
    "leAngleDelta",
    "leRake",
    "leShapevar",
    "leWidthVar",
    "teAngleDelta",
    "teRake",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--staging-dir", default=DEFAULT_STAGING_DIR)
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--surface-points", type=int, default=1_000_000)
    parser.add_argument("--volume-points", type=int, default=1_000_000)
    parser.add_argument("--density-knn-k", type=int, default=16)
    parser.add_argument("--density-device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--density-cache-dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--download-retries", type=int, default=8)
    parser.add_argument("--download-retry-delay", type=float, default=5.0)
    parser.add_argument("--export-inspection-vtk", action="store_true")
    parser.add_argument("--inspection-sample-index", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--source-root", default="", help="Local sample root for smoke tests; skips Hub downloads.")
    return parser.parse_args()


def _normal_name(name: str) -> str:
    return "".join(char for char in str(name).lower() if char.isalnum())


def _point_data(mesh: pv.DataSet) -> pv.DataSet:
    if len(mesh.point_data) > 0:
        return mesh
    if len(mesh.cell_data) == 0:
        raise ValueError("Mesh has neither point data nor cell data.")
    return mesh.cell_data_to_point_data()


def _find_scalar(mesh: pv.DataSet, aliases: tuple[str, ...], label: str) -> tuple[np.ndarray, str]:
    names = {_normal_name(name): name for name in mesh.point_data.keys()}
    for alias in aliases:
        source = names.get(_normal_name(alias))
        if source is None:
            continue
        values = np.asarray(mesh.point_data[source]).reshape(-1)
        if np.isfinite(values).all():
            return np.ascontiguousarray(values, dtype=np.float32), source
    candidates = [
        name for name in mesh.point_data.keys()
        if any(_normal_name(alias) in _normal_name(name) for alias in aliases)
    ]
    if len(candidates) == 1:
        values = np.asarray(mesh.point_data[candidates[0]]).reshape(-1)
        if np.isfinite(values).all():
            return np.ascontiguousarray(values, dtype=np.float32), candidates[0]
    raise KeyError(f"Could not find {label}; available fields: {list(mesh.point_data.keys())}")


def _find_vector(mesh: pv.DataSet, aliases: tuple[str, ...], label: str) -> tuple[np.ndarray, str]:
    names = {_normal_name(name): name for name in mesh.point_data.keys()}
    for alias in aliases:
        source = names.get(_normal_name(alias))
        if source is None:
            continue
        values = np.asarray(mesh.point_data[source])
        if values.ndim == 2 and values.shape[1] >= 3 and np.isfinite(values[:, :3]).all():
            return np.ascontiguousarray(values[:, :3], dtype=np.float32), source
    raise KeyError(f"Could not find {label}; available fields: {list(mesh.point_data.keys())}")


def _extract_targets(surface: pv.DataSet, volume: pv.DataSet) -> tuple[np.ndarray, np.ndarray, dict]:
    pressure = ("pressure", "Pressure (Pa)", "p", "pmean", "p_rgh")
    velocity = ("velocity", "Velocity (m/s)", "U", "UMean", "velocityMean")
    shear = ("wallShearStress", "Wall Shear Stress (N/m²)", "wallShear", "wall_shear_stress", "tauWall")
    surface = _point_data(surface)
    volume = _point_data(volume)
    surface_pressure, surface_pressure_name = _find_scalar(surface, pressure, "surface pressure")
    surface_velocity, surface_velocity_name = _find_vector(surface, velocity, "surface velocity")
    surface_shear, surface_shear_name = _find_vector(surface, shear, "surface wall shear stress")
    volume_pressure, volume_pressure_name = _find_scalar(volume, pressure, "volume pressure")
    volume_velocity, volume_velocity_name = _find_vector(volume, velocity, "volume velocity")
    surface_data = np.ascontiguousarray(np.column_stack([surface_pressure, surface_velocity, surface_shear]), dtype=np.float32)
    volume_data = np.ascontiguousarray(np.column_stack([volume_pressure, volume_velocity]), dtype=np.float32)
    return surface_data, volume_data, {
        "surface": {
            "pressure": surface_pressure_name,
            "velocity": surface_velocity_name,
            "wall_shear": surface_shear_name,
        },
        "volume": {"pressure": volume_pressure_name, "velocity": volume_velocity_name},
        "available_surface_fields": list(surface.point_data.keys()),
        "available_volume_fields": list(volume.point_data.keys()),
    }


def _numeric_parameter_vector(value: object) -> np.ndarray:
    if not isinstance(value, dict):
        raise ValueError("params.json must contain a JSON object.")
    output = []
    for key in PARAMETER_KEYS:
        raw = value.get(key)
        if isinstance(raw, bool) or raw is None:
            raise ValueError(f"Missing finite numeric Pump parameter {key!r}: {raw!r}")
        try:
            number = float(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Pump parameter {key!r} is not numeric: {raw!r}") from exc
        if not np.isfinite(number):
            raise ValueError(f"Pump parameter {key!r} is non-finite.")
        output.append(number)
    return np.asarray(output, dtype=np.float32)


def _load_local_inputs(sample_name: str, source_root: str) -> dict[str, Path]:
    source = Path(source_root) / sample_name
    return {
        name: source / name
        for name in ("merged_surfaces.vtp", "merged_volumes.vtu", "params.json", "metadata.json", "forces.json")
        if (source / name).is_file()
    }


def _process_case(task: tuple[str, str, str, dict, bool]) -> dict:
    sample_name, output_root, staging_root, args, local_mode = task
    run_id = _sample_number(sample_name)
    final_dir = Path(output_root) / f"run_{run_id}"
    marker = final_dir / "_COMPLETE.json"
    if marker.is_file() and not args["overwrite"]:
        return json.loads(marker.read_text(encoding="utf-8"))
    if final_dir.exists():
        shutil.rmtree(final_dir)
    temporary = Path(output_root) / f"run_{run_id}.partial_{os.getpid()}"
    worker_stage = Path(staging_root) / f"worker_{os.getpid()}" / sample_name
    shutil.rmtree(temporary, ignore_errors=True)
    shutil.rmtree(worker_stage, ignore_errors=True)
    temporary.mkdir(parents=True, exist_ok=True)
    worker_stage.mkdir(parents=True, exist_ok=True)
    try:
        if local_mode:
            inputs = _load_local_inputs(sample_name, args["source_root"])
        else:
            inputs = _download_inputs(sample_name, worker_stage, args)
        for required in ("merged_surfaces.vtp", "merged_volumes.vtu", "params.json"):
            if required not in inputs:
                raise FileNotFoundError(f"{sample_name}: missing required {required}")
        surface = pv.read(inputs["merged_surfaces.vtp"])
        volume = pv.read(inputs["merged_volumes.vtu"])
        surface_data_full, volume_data_full, source_fields = _extract_targets(surface, volume)
        surface_points_full = np.ascontiguousarray(np.asarray(surface.points, dtype=np.float32))
        volume_points_full = np.ascontiguousarray(np.asarray(volume.points, dtype=np.float32))
        if surface_points_full.shape != (surface_data_full.shape[0], 3):
            raise ValueError(f"{sample_name}: surface point/field mismatch {surface_points_full.shape} vs {surface_data_full.shape}")
        if volume_points_full.shape != (volume_data_full.shape[0], 3):
            raise ValueError(f"{sample_name}: volume point/field mismatch {volume_points_full.shape} vs {volume_data_full.shape}")
        surface_idx = _sample_indices(surface_points_full.shape[0], int(args["surface_points"]), int(args["seed"]) + 11 * run_id)
        volume_idx = _sample_indices(volume_points_full.shape[0], int(args["volume_points"]), int(args["seed"]) + 17 * run_id)
        surface_points = np.ascontiguousarray(surface_points_full[surface_idx])
        surface_data = np.ascontiguousarray(surface_data_full[surface_idx])
        volume_points = np.ascontiguousarray(volume_points_full[volume_idx])
        volume_data = np.ascontiguousarray(volume_data_full[volume_idx])
        parameters = _numeric_parameter_vector(_load_json(inputs["params.json"]))
        density, density_info = _density_cache(
            surface_points,
            int(args["seed"]) + run_id,
            int(args["density_knn_k"]),
            str(args["density_device"]),
            str(args["density_cache_dtype"]),
        )
        density_name = f"geometry_log_density_{CACHE_VERSION}_casebbox_k{int(args['density_knn_k'])}_h1_{args['density_cache_dtype']}.npy"
        _atomic_save_npy(temporary / "surface_coords.npy", surface_points)
        _atomic_save_npy(temporary / "surface_data.npy", surface_data)
        _atomic_save_npy(temporary / "volume_coords.npy", volume_points)
        _atomic_save_npy(temporary / "volume_data.npy", volume_data)
        _atomic_save_npy(temporary / density_name, density)
        position_min = np.minimum(surface_points.min(axis=0), volume_points.min(axis=0))
        position_max = np.maximum(surface_points.max(axis=0), volume_points.max(axis=0))
        metadata = {
            "run_id": run_id,
            "sample_name": sample_name,
            "surface_points": int(surface_points.shape[0]),
            "surface_source_points": int(surface_points_full.shape[0]),
            "volume_points": int(volume_points.shape[0]),
            "volume_source_points": int(volume_points_full.shape[0]),
            "surface_fields": list(SURFACE_FIELDS),
            "volume_fields": list(VOLUME_FIELDS),
            "parameter_keys": list(PARAMETER_KEYS),
            "parameters": parameters.tolist(),
            "source_field_names": source_fields,
            "density": density_info,
            "cache_version": CACHE_VERSION,
            "surface_sum": surface_data.astype(np.float64).sum(axis=0).tolist(),
            "surface_sq_sum": np.square(surface_data.astype(np.float64)).sum(axis=0).tolist(),
            "surface_count": int(surface_data.shape[0]),
            "volume_sum": volume_data.astype(np.float64).sum(axis=0).tolist(),
            "volume_sq_sum": np.square(volume_data.astype(np.float64)).sum(axis=0).tolist(),
            "volume_count": int(volume_data.shape[0]),
            "parameter_sum": parameters.astype(np.float64).tolist(),
            "parameter_sq_sum": np.square(parameters.astype(np.float64)).tolist(),
            "parameter_count": 1,
            "position_min": position_min.tolist(),
            "position_max": position_max.tolist(),
            "source_metadata": _load_json(inputs.get("metadata.json")),
            "params": _load_json(inputs.get("params.json")),
            "forces": _load_json(inputs.get("forces.json")),
        }
        _atomic_json(temporary / "case_metadata.json", metadata)
        for path in temporary.glob("*.npy"):
            values = np.load(path, mmap_mode="r", allow_pickle=False)
            if not np.isfinite(values).all():
                raise ValueError(f"Non-finite output written to {path}")
        _atomic_json(temporary / "_COMPLETE.json", metadata)
        if final_dir.exists():
            shutil.rmtree(final_dir)
        temporary.replace(final_dir)
        return metadata
    finally:
        shutil.rmtree(temporary, ignore_errors=True)
        shutil.rmtree(worker_stage, ignore_errors=True)
        gc.collect()


def _aggregate_stats(output_root: Path, rows: list[dict], args: argparse.Namespace) -> None:
    surf_sum = np.zeros(len(SURFACE_FIELDS), dtype=np.float64)
    surf_sq = np.zeros(len(SURFACE_FIELDS), dtype=np.float64)
    vol_sum = np.zeros(len(VOLUME_FIELDS), dtype=np.float64)
    vol_sq = np.zeros(len(VOLUME_FIELDS), dtype=np.float64)
    param_sum = np.zeros(len(PARAMETER_KEYS), dtype=np.float64)
    param_sq = np.zeros(len(PARAMETER_KEYS), dtype=np.float64)
    surf_count = vol_count = param_count = 0
    position_min = np.full(3, np.inf, dtype=np.float32)
    position_max = np.full(3, -np.inf, dtype=np.float32)
    for row in rows:
        surf_sum += np.asarray(row["surface_sum"], dtype=np.float64)
        surf_sq += np.asarray(row["surface_sq_sum"], dtype=np.float64)
        vol_sum += np.asarray(row["volume_sum"], dtype=np.float64)
        vol_sq += np.asarray(row["volume_sq_sum"], dtype=np.float64)
        param_sum += np.asarray(row["parameter_sum"], dtype=np.float64)
        param_sq += np.asarray(row["parameter_sq_sum"], dtype=np.float64)
        surf_count += int(row["surface_count"])
        vol_count += int(row["volume_count"])
        param_count += int(row["parameter_count"])
        position_min = np.minimum(position_min, np.asarray(row["position_min"], dtype=np.float32))
        position_max = np.maximum(position_max, np.asarray(row["position_max"], dtype=np.float32))

    def stats(total, square_total, count):
        mean = total / float(count)
        variance = np.maximum((square_total - np.square(total) / float(count)) / max(count - 1, 1), 1.0e-12)
        return np.stack([mean, np.sqrt(variance)]).astype(np.float32)

    _atomic_save_npy(output_root / f"surface_stats_{CACHE_VERSION}.npy", stats(surf_sum, surf_sq, surf_count))
    _atomic_save_npy(output_root / f"volume_stats_{CACHE_VERSION}.npy", stats(vol_sum, vol_sq, vol_count))
    _atomic_save_npy(output_root / f"parameter_stats_{CACHE_VERSION}.npy", stats(param_sum, param_sq, param_count))
    _atomic_save_npy(output_root / f"position_stats_{CACHE_VERSION}.npy", np.stack([position_min, position_max]).astype(np.float32))
    ids = sorted(int(row["run_id"]) for row in rows)
    shuffled = np.asarray(ids, dtype=np.int64)
    np.random.default_rng(int(args.seed)).shuffle(shuffled)
    split = max(1, min(len(shuffled) - 1, int(round(0.8 * len(shuffled)))) if len(shuffled) > 1 else 1)
    train_ids = sorted(int(value) for value in shuffled[:split])
    test_ids = train_ids if len(shuffled) == 1 else sorted(int(value) for value in shuffled[split:])
    manifest = {
        "dataset": "SHIFTPumpSample",
        "source": str(args.repo_id),
        "sample_count": len(ids),
        "fields": {"surface": list(SURFACE_FIELDS), "volume": list(VOLUME_FIELDS)},
        "parameter_keys": list(PARAMETER_KEYS),
        "surface_cap": int(args.surface_points),
        "volume_points": int(args.volume_points),
        "density": {"estimator": "kde", "knn_k": int(args.density_knn_k), "surface_only": True, "normalization": "per_case_surface_bbox", "cache_dtype": args.density_cache_dtype},
        "train_ids": train_ids,
        "test_ids": test_ids,
        "runs": sorted(rows, key=lambda row: int(row["run_id"])),
    }
    _atomic_json(output_root / "preprocessed_manifest.json", manifest)
    _atomic_json(output_root / "splits.json", {"seed": int(args.seed), "train_ids": train_ids, "test_ids": test_ids})


def _export_inspection(sample_name: str, results_root: Path, staging_root: Path, args: dict) -> None:
    stage = Path(staging_root) / "inspection" / sample_name
    output = results_root / "inspection" / sample_name
    temporary = output.with_name(output.name + ".partial")
    shutil.rmtree(stage, ignore_errors=True)
    shutil.rmtree(temporary, ignore_errors=True)
    stage.mkdir(parents=True, exist_ok=True)
    temporary.mkdir(parents=True, exist_ok=True)
    try:
        inputs = (
            _load_local_inputs(sample_name, args["source_root"])
            if args.get("source_root")
            else _download_inputs(sample_name, stage, args)
        )
        surface = pv.read(inputs["merged_surfaces.vtp"])
        volume = pv.read(inputs["merged_volumes.vtu"])
        surface.save(temporary / "merged_surfaces.vtp")
        volume.save(temporary / "merged_volumes.vtu")
        for name in ("params.json", "metadata.json", "forces.json"):
            if name in inputs:
                shutil.copyfile(inputs[name], temporary / name)
        temporary.replace(output)
        print(f"[inspection] {output}", flush=True)
    finally:
        shutil.rmtree(stage.parent, ignore_errors=True)
        shutil.rmtree(temporary, ignore_errors=True)


def main() -> None:
    args = parse_args()
    if args.num_samples <= 0 or args.workers <= 0 or args.surface_points <= 0 or args.volume_points <= 0:
        raise ValueError("num-samples, workers, surface-points, and volume-points must be positive.")
    if not args.source_root:
        sample_names = list_samples(args.num_samples, args.start_index, args.repo_id, args.revision)
    else:
        root = Path(args.source_root).expanduser().resolve()
        sample_names = sorted((path.name for path in root.glob("sample_*") if path.is_dir()), key=_sample_number)
        sample_names = [name for name in sample_names if _sample_number(name) >= args.start_index][:args.num_samples]
        if len(sample_names) < args.num_samples:
            raise RuntimeError(f"Local source has only {len(sample_names)} selected cases.")
    output_root = Path(args.output_dir).expanduser().resolve()
    results_root = Path(args.results_dir).expanduser().resolve()
    staging_root = Path(args.staging_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    results_root.mkdir(parents=True, exist_ok=True)
    staging_root.mkdir(parents=True, exist_ok=True)
    density_device = args.density_device
    if density_device == "auto":
        density_device = "cpu" if args.workers > 1 else ("cuda" if torch.cuda.is_available() else "cpu")
    if density_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA density requested but CUDA is unavailable.")
    options = {
        "repo_id": args.repo_id,
        "revision": args.revision,
        "surface_points": args.surface_points,
        "volume_points": args.volume_points,
        "density_knn_k": args.density_knn_k,
        "density_device": density_device,
        "density_cache_dtype": args.density_cache_dtype,
        "seed": args.seed,
        "timeout_seconds": args.timeout_seconds,
        "download_retries": args.download_retries,
        "download_retry_delay": args.download_retry_delay,
        "overwrite": args.overwrite,
        "source_root": str(Path(args.source_root).expanduser().resolve()) if args.source_root else "",
    }
    tasks = [(name, str(output_root), str(staging_root), options, bool(args.source_root)) for name in sample_names]
    print(f"Selected {len(tasks)} Pump cases, workers={args.workers}, density_device={density_device}", flush=True)
    try:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(_process_case, task) for task in tasks]
            for index, future in enumerate(concurrent.futures.as_completed(futures), start=1):
                row = future.result()
                print(f"[{index}/{len(tasks)}] {row['sample_name']}: surface={row['surface_points']:,}, volume={row['volume_points']:,}", flush=True)
    except KeyboardInterrupt:
        print("Interrupted; completed cases remain and worker staging is cleaned.", file=sys.stderr, flush=True)
        raise
    rows = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(output_root.glob("run_*/_COMPLETE.json"), key=lambda path: _sample_number(path.parent.name))
    ]
    if len(rows) < len(tasks):
        raise RuntimeError(f"Only {len(rows)}/{len(tasks)} cases completed; aggregate files were not written.")
    _aggregate_stats(output_root, rows, args)
    _atomic_json(results_root / "first_case_summary.json", rows[0])
    if args.export_inspection_vtk:
        selected = next((name for name in sample_names if _sample_number(name) == args.inspection_sample_index), None)
        if selected is None:
            raise ValueError(f"Inspection sample {args.inspection_sample_index} was not selected.")
        _export_inspection(selected, results_root, staging_root, options)
    if staging_root.exists() and not any(staging_root.iterdir()):
        staging_root.rmdir()
    print(f"Preprocessing complete: {len(rows)} cases -> {output_root}", flush=True)


if __name__ == "__main__":
    main()
