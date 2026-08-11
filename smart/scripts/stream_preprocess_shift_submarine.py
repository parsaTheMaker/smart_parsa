#!/usr/bin/env python3
"""Download and preprocess a bounded subset of the SHIFT-Submarine dataset.

The repository is a gated Hugging Face dataset containing one directory per
case.  This script downloads only the requested files for one case per worker,
converts them immediately, and deletes the raw files after validation.

Output channels are deliberately kept compatible with point-cloud SMART
experiments while matching the available fields:

* surface: ``pressure, wall_shear_x, wall_shear_y, wall_shear_z``;
* volume: ``pressure, velocity_x, velocity_y, velocity_z``.

The volume is restricted to points within ``--volume-radius-fraction`` of the
volume-domain centre before deterministic 1M-point sampling.  Surface points
are capped deterministically at 1M.  KDE k=16 is cached on the retained
surface cloud only.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import gc
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pyvista as pv
import requests
import torch

# Xet transfers were stalling on the large LFS-backed VTP/VTU files on this
# host. Use the regular authenticated Hub/LFS HTTP path for reliable streaming.
# This must be set before importing huggingface_hub.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

_SMART_ROOT = Path(__file__).resolve().parents[1]
if str(_SMART_ROOT) not in sys.path:
    sys.path.insert(0, str(_SMART_ROOT))

try:
    from utils.geometry_density import estimate_log_sampling_density
except ImportError:  # pragma: no cover
    from smart.utils.geometry_density import estimate_log_sampling_density

try:
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import GatedRepoError
    from huggingface_hub.utils import HfHubHTTPError
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("huggingface_hub is required.") from exc


DEFAULT_REPO_ID = "luminary-shift/Submarine-sample"
DEFAULT_REVISION = "main"
DEFAULT_OUTPUT_DIR = "/mnt/ssdraid/parsa/shift_submarine_sample_preprocessed"
DEFAULT_RESULTS_DIR = "/home/parsa/smart_parsa/results/shift_submarine_sample_preprocess"
DEFAULT_STAGING_DIR = "/mnt/ssdraid/parsa/.shift_submarine_staging"
SURFACE_FIELDS = ("pressure", "wall_shear_x", "wall_shear_y", "wall_shear_z")
VOLUME_FIELDS = ("pressure", "velocity_x", "velocity_y", "velocity_z")
CACHE_VERSION = "shift_submarine_sample_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--staging-dir", default=DEFAULT_STAGING_DIR)
    parser.add_argument("--num-samples", type=int, default=400)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--surface-points", type=int, default=1_000_000)
    parser.add_argument("--volume-points", type=int, default=1_000_000)
    parser.add_argument("--volume-radius-fraction", type=float, default=0.20)
    parser.add_argument("--density-knn-k", type=int, default=16)
    parser.add_argument("--density-device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--density-cache-dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--download-retries", type=int, default=8)
    parser.add_argument("--download-retry-delay", type=float, default=5.0)
    parser.add_argument(
        "--export-inspection-vtk",
        action="store_true",
        help="Preserve one original VTP surface and VTU volume under results/inspection.",
    )
    parser.add_argument("--inspection-sample-index", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--source-root", default="", help="Local sample root for smoke tests; skips Hugging Face downloads.")
    parser.add_argument("--delete-local-source", action="store_true")
    return parser.parse_args()


def _headers() -> dict[str, str]:
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        raise RuntimeError(
            "This gated dataset requires HF_TOKEN. Export an accepted Hugging Face token before running."
        )
    return {"Authorization": f"Bearer {token}"}


def _sample_number(sample_name: str) -> int:
    match = re.search(r"(\d+)$", str(sample_name))
    if not match:
        raise ValueError(f"Invalid sample directory name: {sample_name}")
    return int(match.group(1))


def _atomic_save_npy(path: Path, array: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("wb") as handle:
        np.save(handle, np.asarray(array), allow_pickle=False)
    temporary.replace(path)


def _atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def list_samples(num_samples: int, start_index: int, repo_id: str, revision: str) -> list[str]:
    api_url = f"https://huggingface.co/api/datasets/{repo_id}/tree/{revision}"
    response = requests.get(
        api_url,
        params={"recursive": "false", "expand": "false", "limit": 1000},
        headers=_headers(),
        timeout=120,
    )
    if response.status_code == 401:
        raise RuntimeError("Hugging Face rejected the token. Accept the dataset terms and export a valid HF_TOKEN.")
    response.raise_for_status()
    rows = response.json()
    names = sorted(
        [row["path"] for row in rows if row.get("type") == "directory" and str(row.get("path", "")).startswith("sample_")],
        key=_sample_number,
    )
    selected = [name for name in names if _sample_number(name) >= int(start_index)]
    selected = selected[: int(num_samples)] if int(num_samples) > 0 else selected
    if len(selected) < int(num_samples):
        raise RuntimeError(f"Requested {num_samples} samples from index {start_index}, found {len(selected)}.")
    return selected


def _download_file(sample_name: str, filename: str, destination: Path, args: dict) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    downloaded = None
    last_error = None
    attempts = max(1, int(args["download_retries"]) + 1)
    for attempt in range(attempts):
        try:
            downloaded = hf_hub_download(
                repo_id=str(args["repo_id"]),
                filename=filename,
                subfolder=sample_name,
                repo_type="dataset",
                revision=str(args["revision"]),
                token=os.environ["HF_TOKEN"].strip(),
                cache_dir=str(destination.parent / ".hf_cache"),
                local_dir=str(destination.parent),
                force_download=False,
                etag_timeout=60,
            )
            break
        except GatedRepoError as exc:
            raise RuntimeError(
                f"Hugging Face access to {args['repo_id']} is still awaiting review "
                "from the dataset authors. Wait for approval at "
                f"https://huggingface.co/datasets/{args['repo_id']}, then rerun "
                "the same command; no downloader or token change can bypass a pending gate."
            ) from exc
        except HfHubHTTPError as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if status in {401, 403}:
                raise RuntimeError(
                    f"Hugging Face denied {sample_name}/{filename} (HTTP {status}). "
                    f"Open https://huggingface.co/datasets/{args['repo_id']}, "
                    "accept the dataset access terms, then export a read-scoped HF_TOKEN."
                ) from exc
            if status == 404:
                raise FileNotFoundError(f"Dataset file not found: {sample_name}/{filename}") from exc
            last_error = exc
        except OSError as exc:
            last_error = exc
        if attempt + 1 < attempts:
            delay = float(args["download_retry_delay"]) * min(4.0, 1.5 ** attempt)
            print(
                f"[download retry {attempt + 1}/{attempts - 1}] "
                f"{sample_name}/{filename}: {type(last_error).__name__}; sleeping {delay:.1f}s",
                flush=True,
            )
            time.sleep(delay)
    if downloaded is None:
        raise RuntimeError(
            f"Could not download {sample_name}/{filename} after {attempts} attempts."
        ) from last_error
    downloaded = Path(downloaded)
    if not downloaded.is_file():
        raise IOError(f"Hugging Face reported success but no local file exists: {downloaded}")
    if downloaded.resolve() != destination.resolve():
        temporary = destination.with_suffix(destination.suffix + ".partial")
        temporary.unlink(missing_ok=True)
        shutil.copyfile(downloaded, temporary)
        temporary.replace(destination)
        downloaded.unlink(missing_ok=True)


def _normal_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).lower())


def _point_data_with_cell_fallback(mesh: pv.DataSet) -> pv.DataSet:
    """Expose cell fields at points when a file has no point fields.

    Most Submarine files store solver outputs on points, but keeping this
    fallback makes the converter work with files exported as cell-centered
    VTK data without silently dropping the fields.
    """
    if len(mesh.point_data) > 0:
        return mesh
    if len(mesh.cell_data) == 0:
        raise ValueError("Mesh has neither point data nor cell data.")
    return mesh.cell_data_to_point_data()


def _scalar_field(mesh: pv.DataSet, aliases: tuple[str, ...], label: str) -> tuple[np.ndarray, str]:
    normalized = {_normal_name(name): name for name in mesh.point_data.keys()}
    for alias in aliases:
        source_name = normalized.get(_normal_name(alias))
        if source_name is None:
            continue
        values = np.asarray(mesh.point_data[source_name])
        if values.ndim == 1:
            values = values[:, None]
        if values.ndim == 2 and values.shape[1] == 1:
            values = values[:, 0]
            if np.isfinite(values).all():
                return np.ascontiguousarray(values.astype(np.float32)), source_name
    candidates = [name for name in mesh.point_data.keys() if any(_normal_name(alias) in _normal_name(name) for alias in aliases)]
    if len(candidates) == 1:
        values = np.asarray(mesh.point_data[candidates[0]])
        if values.ndim == 1 or (values.ndim == 2 and values.shape[1] == 1):
            values = values.reshape(-1)
            if np.isfinite(values).all():
                return np.ascontiguousarray(values.astype(np.float32)), candidates[0]
    raise KeyError(f"Could not find scalar {label}; available point fields: {list(mesh.point_data.keys())}")


def _vector_field(mesh: pv.DataSet, aliases: tuple[str, ...], label: str) -> tuple[np.ndarray, str]:
    normalized = {_normal_name(name): name for name in mesh.point_data.keys()}
    for alias in aliases:
        source_name = normalized.get(_normal_name(alias))
        if source_name is None:
            continue
        values = np.asarray(mesh.point_data[source_name])
        if values.ndim == 2 and values.shape[1] >= 3 and np.isfinite(values[:, :3]).all():
            return np.ascontiguousarray(values[:, :3].astype(np.float32)), source_name
    axis_sources = []
    for axis in ("x", "y", "z"):
        candidates = [
            name for name in mesh.point_data.keys()
            if any(_normal_name(alias) in _normal_name(name) for alias in aliases)
            and _normal_name(name).endswith(axis)
        ]
        if len(candidates) != 1:
            axis_sources = []
            break
        axis_sources.append(candidates[0])
    if len(axis_sources) == 3:
        values = np.column_stack([np.asarray(mesh.point_data[name]).reshape(-1) for name in axis_sources])
        if np.isfinite(values).all():
            return np.ascontiguousarray(values.astype(np.float32)), "/".join(axis_sources)
    raise KeyError(f"Could not find vector {label}; available point fields: {list(mesh.point_data.keys())}")


def _extract_targets(surface: pv.DataSet, volume: pv.DataSet) -> tuple[np.ndarray, np.ndarray, dict]:
    pressure_aliases = (
        "pressure",
        "Pressure (Pa)",
        "p",
        "pmean",
        "p_rgh",
        "pressuremean",
        "pressuretimeaveraged",
    )
    shear_aliases = (
        "wallShearStress",
        "Wall Shear Stress (N/m²)",
        "wallShearStressMean",
        "wallShear",
        "wall_shear_stress",
        "tauWall",
    )
    velocity_aliases = (
        "velocity",
        "Velocity (m/s)",
        "U",
        "UMean",
        "velocityMean",
        "velocityTimeAveraged",
    )

    def extract(point_mesh: pv.DataSet, cell_mesh: pv.DataSet, kind: str):
        point_mesh = _point_data_with_cell_fallback(point_mesh)
        try:
            if kind == "surface":
                pressure = _scalar_field(point_mesh, pressure_aliases, "surface pressure")
                shear = _vector_field(point_mesh, shear_aliases, "surface wall shear stress")
                return pressure, shear, point_mesh
            pressure = _scalar_field(point_mesh, pressure_aliases, "volume pressure")
            velocity = _vector_field(point_mesh, velocity_aliases, "volume velocity")
            return pressure, velocity, point_mesh
        except (KeyError, ValueError):
            # A mesh can contain unrelated point fields while the requested
            # solver fields remain cell-centered. Convert only on failure.
            if len(cell_mesh.cell_data) == 0:
                raise
            converted = cell_mesh.cell_data_to_point_data()
            if kind == "surface":
                pressure = _scalar_field(converted, pressure_aliases, "surface pressure")
                shear = _vector_field(converted, shear_aliases, "surface wall shear stress")
                return pressure, shear, converted
            pressure = _scalar_field(converted, pressure_aliases, "volume pressure")
            velocity = _vector_field(converted, velocity_aliases, "volume velocity")
            return pressure, velocity, converted

    surface_pressure_pair, surface_shear_pair, surface_fields_mesh = extract(surface, surface, "surface")
    volume_pressure_pair, volume_velocity_pair, volume_fields_mesh = extract(volume, volume, "volume")
    surface_pressure, surface_pressure_name = surface_pressure_pair
    surface_shear, surface_shear_name = surface_shear_pair
    volume_pressure, volume_pressure_name = volume_pressure_pair
    volume_velocity, volume_velocity_name = volume_velocity_pair
    surface_data = np.ascontiguousarray(np.column_stack([surface_pressure, surface_shear]))
    volume_data = np.ascontiguousarray(np.column_stack([volume_pressure, volume_velocity]))
    names = {
        "surface": {"pressure": surface_pressure_name, "wall_shear": surface_shear_name},
        "volume": {"pressure": volume_pressure_name, "velocity": volume_velocity_name},
        "available_surface_fields": list(surface_fields_mesh.point_data.keys()),
        "available_volume_fields": list(volume_fields_mesh.point_data.keys()),
    }
    return surface_data, volume_data, names


def _sample_indices(n: int, count: int, seed: int) -> np.ndarray:
    if count <= 0 or count >= int(n):
        return np.arange(int(n), dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    return np.sort(rng.choice(int(n), size=int(count), replace=False).astype(np.int64))


def _density_cache(points: np.ndarray, seed: int, knn_k: int, device_name: str, cache_dtype: str) -> tuple[np.ndarray, dict]:
    lower = points.min(axis=0)
    upper = points.max(axis=0)
    normalized = np.clip((points - lower) / np.maximum(upper - lower, 1.0e-12), 0.0, 1.0 - 1.0e-6).astype(np.float32)
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    started = time.perf_counter()
    with torch.inference_mode():
        log_density = estimate_log_sampling_density(
            torch.from_numpy(normalized).unsqueeze(0).to(device),
            knn_k=int(knn_k),
            estimator="kde",
        ).squeeze(0).float().cpu().numpy()
    if not np.isfinite(log_density).all():
        raise ValueError("Surface KDE returned non-finite values.")
    output = log_density.astype(np.float16 if cache_dtype == "float16" else np.float32, copy=False)
    info = {
        "estimator": "kde",
        "knn_k": int(knn_k),
        "normalization": "per_case_surface_bbox",
        "bbox_min": lower.tolist(),
        "bbox_max": upper.tolist(),
        "dtype": str(output.dtype),
        "seconds": float(time.perf_counter() - started),
    }
    del normalized
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return output, info


def _download_inputs(sample_name: str, staging: Path, args: dict) -> dict[str, Path]:
    staging.mkdir(parents=True, exist_ok=True)
    required = ("merged_surfaces.vtp", "merged_volumes.vtu")
    optional = ("params.json", "metadata.json", "forces.json")
    paths = {}
    for filename in required + optional:
        target = staging / filename
        try:
            _download_file(sample_name, filename, target, args)
            paths[filename] = target
        except FileNotFoundError:
            if filename in required:
                raise
        except (OSError, RuntimeError) as exc:
            if filename in required:
                raise
            print(
                f"[optional file skipped] {sample_name}/{filename}: {str(exc).splitlines()[0]}",
                flush=True,
            )
    return paths


def _export_inspection_vtk(sample_name: str, results_root: Path, staging_root: Path, args: dict) -> None:
    """Preserve the original VTK XML files for one case for visual inspection."""
    inspection_stage = staging_root / f"inspection_{os.getpid()}" / sample_name
    inspection_output = results_root / "inspection" / sample_name
    temporary_output = inspection_output.with_name(inspection_output.name + ".partial")
    if inspection_stage.exists():
        shutil.rmtree(inspection_stage)
    if temporary_output.exists():
        shutil.rmtree(temporary_output)
    inspection_stage.mkdir(parents=True, exist_ok=True)
    temporary_output.mkdir(parents=True, exist_ok=True)
    try:
        inputs = _download_inputs(sample_name, inspection_stage, args)
        # Read/write through PyVista so the saved files are valid VTK XML,
        # independent of Hugging Face's local cache/symlink layout.
        surface = pv.read(inputs["merged_surfaces.vtp"])
        volume = pv.read(inputs["merged_volumes.vtu"])
        surface.save(temporary_output / "merged_surfaces.vtp")
        volume.save(temporary_output / "merged_volumes.vtu")
        for filename in ("params.json", "metadata.json", "forces.json"):
            source = inputs.get(filename)
            if source is not None and source.is_file():
                shutil.copyfile(source, temporary_output / filename)
        if inspection_output.exists():
            shutil.rmtree(inspection_output)
        temporary_output.replace(inspection_output)
        print(
            f"[inspection] preserved {sample_name}: "
            f"surface={inspection_output / 'merged_surfaces.vtp'}, "
            f"volume={inspection_output / 'merged_volumes.vtu'}",
            flush=True,
        )
    finally:
        if temporary_output.exists():
            shutil.rmtree(temporary_output, ignore_errors=True)
        shutil.rmtree(inspection_stage.parent, ignore_errors=True)


def _load_json(path: Path | None) -> object:
    if path is None or not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _process_case(task: tuple[str, str, str, dict, bool]) -> dict:
    sample_name, output_root, staging_root, args, local_mode = task
    sample_number = _sample_number(sample_name)
    final_dir = Path(output_root) / f"run_{sample_number}"
    marker = final_dir / "_COMPLETE.json"
    if marker.is_file() and not args["overwrite"]:
        return json.loads(marker.read_text(encoding="utf-8"))
    if final_dir.exists():
        shutil.rmtree(final_dir)
    temporary_output = Path(output_root) / f"run_{sample_number}.partial_{os.getpid()}"
    if temporary_output.exists():
        shutil.rmtree(temporary_output)
    temporary_output.mkdir(parents=True, exist_ok=True)
    worker_staging = Path(staging_root) / f"worker_{os.getpid()}" / sample_name
    if worker_staging.exists():
        shutil.rmtree(worker_staging)
    worker_staging.mkdir(parents=True, exist_ok=True)
    try:
        if local_mode:
            source_dir = Path(args["source_root"]) / sample_name
            inputs = {name: source_dir / name for name in ("merged_surfaces.vtp", "merged_volumes.vtu", "params.json", "metadata.json", "forces.json") if (source_dir / name).is_file()}
        else:
            inputs = _download_inputs(sample_name, worker_staging, args)
        surface = pv.read(inputs["merged_surfaces.vtp"])
        volume = pv.read(inputs["merged_volumes.vtu"])
        surface_data, volume_data_full, field_names = _extract_targets(surface, volume)
        surface_points_full = np.ascontiguousarray(np.asarray(surface.points, dtype=np.float32))
        volume_points_full = np.ascontiguousarray(np.asarray(volume.points, dtype=np.float32))
        if surface_points_full.shape[0] != surface_data.shape[0]:
            raise ValueError(f"Surface point/field mismatch for {sample_name}: {surface_points_full.shape} vs {surface_data.shape}")
        if volume_points_full.shape[0] != volume_data_full.shape[0]:
            raise ValueError(f"Volume point/field mismatch for {sample_name}: {volume_points_full.shape} vs {volume_data_full.shape}")

        surface_indices = _sample_indices(surface_points_full.shape[0], int(args["surface_points"]), int(args["seed"]) + 11 * sample_number)
        surface_points = np.ascontiguousarray(surface_points_full[surface_indices])
        surface_data = np.ascontiguousarray(surface_data[surface_indices])
        volume_center = 0.5 * (volume_points_full.min(axis=0) + volume_points_full.max(axis=0))
        distances = np.linalg.norm(volume_points_full - volume_center[None, :], axis=1)
        domain_radius = float(distances.max())
        radius_limit = float(args["volume_radius_fraction"]) * domain_radius
        volume_keep = distances <= radius_limit
        kept_count = int(volume_keep.sum())
        if kept_count < int(args["volume_points"]):
            raise ValueError(
                f"{sample_name}: radius cutoff retained {kept_count:,} points, "
                f"fewer than requested {int(args['volume_points']):,}; increase --volume-radius-fraction."
            )
        volume_pool_points = np.ascontiguousarray(volume_points_full[volume_keep])
        volume_pool_data = np.ascontiguousarray(volume_data_full[volume_keep])
        volume_indices = _sample_indices(volume_pool_points.shape[0], int(args["volume_points"]), int(args["seed"]) + 17 * sample_number)
        volume_points = np.ascontiguousarray(volume_pool_points[volume_indices])
        volume_data = np.ascontiguousarray(volume_pool_data[volume_indices])
        density, density_info = _density_cache(
            surface_points,
            int(args["seed"]) + sample_number,
            int(args["density_knn_k"]),
            str(args["density_device"]),
            str(args["density_cache_dtype"]),
        )
        density_name = f"geometry_log_density_{CACHE_VERSION}_casebbox_k{int(args['density_knn_k'])}_h1_{args['density_cache_dtype']}.npy"
        _atomic_save_npy(temporary_output / "surface_coords.npy", surface_points)
        _atomic_save_npy(temporary_output / "surface_data.npy", surface_data)
        _atomic_save_npy(temporary_output / "volume_coords.npy", volume_points)
        _atomic_save_npy(temporary_output / "volume_data.npy", volume_data)
        _atomic_save_npy(temporary_output / density_name, density)
        metadata = {
            "run_id": sample_number,
            "sample_name": sample_name,
            "surface_points": int(surface_points.shape[0]),
            "surface_source_points": int(surface_points_full.shape[0]),
            "volume_points": int(volume_points.shape[0]),
            "volume_source_points": int(volume_points_full.shape[0]),
            "volume_cut_points": kept_count,
            "volume_center": volume_center.tolist(),
            "volume_domain_radius": domain_radius,
            "volume_radius_fraction": float(args["volume_radius_fraction"]),
            "volume_radius_limit": radius_limit,
            "surface_fields": list(SURFACE_FIELDS),
            "volume_fields": list(VOLUME_FIELDS),
            "source_field_names": field_names,
            "density": density_info,
            "cache_version": CACHE_VERSION,
            "surface_sum": surface_data.astype(np.float64).sum(axis=0).tolist(),
            "surface_sq_sum": np.square(surface_data.astype(np.float64)).sum(axis=0).tolist(),
            "surface_count": int(surface_data.shape[0]),
            "volume_sum": volume_data.astype(np.float64).sum(axis=0).tolist(),
            "volume_sq_sum": np.square(volume_data.astype(np.float64)).sum(axis=0).tolist(),
            "volume_count": int(volume_data.shape[0]),
            "position_min": np.minimum(surface_points.min(axis=0), volume_points.min(axis=0)).tolist(),
            "position_max": np.maximum(surface_points.max(axis=0), volume_points.max(axis=0)).tolist(),
            "source_metadata": _load_json(inputs.get("metadata.json")),
            "params": _load_json(inputs.get("params.json")),
            "forces": _load_json(inputs.get("forces.json")),
        }
        _atomic_json(temporary_output / "case_metadata.json", metadata)
        for path in temporary_output.glob("*.npy"):
            loaded = np.load(path, mmap_mode="r", allow_pickle=False)
            if not np.isfinite(loaded).all():
                raise ValueError(f"Non-finite output written to {path}")
        _atomic_json(temporary_output / "_COMPLETE.json", metadata)
        if final_dir.exists():
            shutil.rmtree(final_dir)
        temporary_output.replace(final_dir)
        return metadata
    finally:
        if temporary_output.exists():
            shutil.rmtree(temporary_output, ignore_errors=True)
        if not local_mode or args["delete_local_source"]:
            shutil.rmtree(worker_staging, ignore_errors=True)
        gc.collect()


def _aggregate_stats(output_root: Path, metadata_rows: list[dict], args: argparse.Namespace) -> None:
    surf_sum = np.zeros(4, dtype=np.float64)
    surf_sq = np.zeros(4, dtype=np.float64)
    surf_count = 0
    vol_sum = np.zeros(4, dtype=np.float64)
    vol_sq = np.zeros(4, dtype=np.float64)
    vol_count = 0
    position_min = np.full(3, np.inf, dtype=np.float32)
    position_max = np.full(3, -np.inf, dtype=np.float32)
    for row in metadata_rows:
        surf_sum += np.asarray(row["surface_sum"], dtype=np.float64)
        surf_sq += np.asarray(row["surface_sq_sum"], dtype=np.float64)
        surf_count += int(row["surface_count"])
        vol_sum += np.asarray(row["volume_sum"], dtype=np.float64)
        vol_sq += np.asarray(row["volume_sq_sum"], dtype=np.float64)
        vol_count += int(row["volume_count"])
        position_min = np.minimum(position_min, np.asarray(row["position_min"], dtype=np.float32))
        position_max = np.maximum(position_max, np.asarray(row["position_max"], dtype=np.float32))

    def stats(sum_, sq_sum, count):
        mean = sum_ / float(count)
        variance = np.maximum((sq_sum - np.square(sum_) / float(count)) / max(count - 1, 1), 1.0e-12)
        return np.stack([mean.astype(np.float32), np.sqrt(variance).astype(np.float32)])

    _atomic_save_npy(output_root / "surface_stats.npy", stats(surf_sum, surf_sq, surf_count))
    _atomic_save_npy(output_root / "volume_stats.npy", stats(vol_sum, vol_sq, vol_count))
    _atomic_save_npy(output_root / "position_stats.npy", np.stack([position_min, position_max]))
    ids = sorted(int(row["run_id"]) for row in metadata_rows)
    rng = np.random.default_rng(int(args.seed))
    shuffled = np.asarray(ids, dtype=np.int64)
    rng.shuffle(shuffled)
    if len(shuffled) == 1:
        train_ids = test_ids = [int(shuffled[0])]
    else:
        split = max(1, min(len(shuffled) - 1, int(round(0.8 * len(shuffled)))))
        train_ids = sorted(int(value) for value in shuffled[:split])
        test_ids = sorted(int(value) for value in shuffled[split:])
    manifest = {
        "dataset": "SHIFTSubmarineSample",
        "source": str(args.repo_id),
        "sample_count": len(ids),
        "fields": {"surface": list(SURFACE_FIELDS), "volume": list(VOLUME_FIELDS)},
        "surface_cap": int(args.surface_points),
        "volume_points": int(args.volume_points),
        "volume_radius_fraction": float(args.volume_radius_fraction),
        "density": {"estimator": "kde", "knn_k": int(args.density_knn_k), "surface_only": True, "normalization": "per_case_surface_bbox", "cache_dtype": args.density_cache_dtype},
        "train_ids": train_ids,
        "test_ids": test_ids,
        "runs": sorted(metadata_rows, key=lambda row: int(row["run_id"])),
    }
    _atomic_json(output_root / "preprocessed_manifest.json", manifest)
    _atomic_json(output_root / "splits.json", {"seed": int(args.seed), "train_ids": train_ids, "test_ids": test_ids})


def main() -> None:
    args = parse_args()
    if int(args.num_samples) <= 0 or int(args.workers) <= 0:
        raise ValueError("--num-samples and --workers must be positive.")
    if not 0.0 < float(args.volume_radius_fraction) <= 1.0:
        raise ValueError("--volume-radius-fraction must be in (0, 1].")
    if not args.source_root:
        _headers()
        sample_names = list_samples(
            int(args.num_samples),
            int(args.start_index),
            str(args.repo_id),
            str(args.revision),
        )
    else:
        source_root = Path(args.source_root).expanduser().resolve()
        sample_names = sorted([path.name for path in source_root.glob("sample_*") if path.is_dir()], key=_sample_number)
        sample_names = [name for name in sample_names if _sample_number(name) >= int(args.start_index)][: int(args.num_samples)]
        if len(sample_names) < int(args.num_samples):
            raise RuntimeError(f"Local source has only {len(sample_names)} samples.")
    output_root = Path(args.output_dir).expanduser().resolve()
    results_root = Path(args.results_dir).expanduser().resolve()
    staging_root = Path(args.staging_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    results_root.mkdir(parents=True, exist_ok=True)
    staging_root.mkdir(parents=True, exist_ok=True)
    density_device = str(args.density_device)
    if density_device == "auto":
        density_device = "cpu" if int(args.workers) > 1 else ("cuda" if torch.cuda.is_available() else "cpu")
    if density_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--density-device=cuda requested but CUDA is unavailable.")
    options = {
        "repo_id": str(args.repo_id),
        "revision": str(args.revision),
        "surface_points": int(args.surface_points),
        "volume_points": int(args.volume_points),
        "volume_radius_fraction": float(args.volume_radius_fraction),
        "density_knn_k": int(args.density_knn_k),
        "density_device": density_device,
        "density_cache_dtype": str(args.density_cache_dtype),
        "seed": int(args.seed),
        "timeout_seconds": int(args.timeout_seconds),
        "download_retries": int(args.download_retries),
        "download_retry_delay": float(args.download_retry_delay),
        "overwrite": bool(args.overwrite),
        "source_root": str(Path(args.source_root).expanduser().resolve()) if args.source_root else "",
        "delete_local_source": bool(args.delete_local_source),
    }
    tasks = [(name, str(output_root), str(staging_root), options, bool(args.source_root)) for name in sample_names]
    print(f"Selected {len(tasks)} cases, workers={args.workers}, density_device={density_device}", flush=True)
    metadata_rows = []
    try:
        with concurrent.futures.ProcessPoolExecutor(max_workers=int(args.workers)) as executor:
            futures = [executor.submit(_process_case, task) for task in tasks]
            for index, future in enumerate(concurrent.futures.as_completed(futures), start=1):
                row = future.result()
                metadata_rows.append(row)
                print(
                    f"[{index}/{len(tasks)}] {row['sample_name']}: "
                    f"surface={row['surface_points']:,}, volume={row['volume_points']:,}, "
                    f"cut_pool={row['volume_cut_points']:,}",
                    flush=True,
                )
    except KeyboardInterrupt:
        print("Interrupted. Completed cases remain; active workers clean their staged files on exit.", file=sys.stderr, flush=True)
        raise
    all_rows = []
    for marker in sorted(output_root.glob("run_*/_COMPLETE.json"), key=lambda path: _sample_number(path.parent.name)):
        all_rows.append(json.loads(marker.read_text(encoding="utf-8")))
    if len(all_rows) < len(tasks):
        raise RuntimeError(f"Only {len(all_rows)}/{len(tasks)} cases have complete markers; no aggregate manifest was written.")
    _aggregate_stats(output_root, all_rows, args)
    _atomic_json(results_root / "first_case_summary.json", all_rows[0])
    if bool(args.export_inspection_vtk):
        selected_inspection = next(
            (
                name
                for name in sample_names
                if _sample_number(name) == int(args.inspection_sample_index)
            ),
            None,
        )
        if selected_inspection is None:
            raise ValueError(
                f"Inspection sample {int(args.inspection_sample_index)} was not selected "
                f"from start_index={int(args.start_index)} and num_samples={int(args.num_samples)}."
            )
        _export_inspection_vtk(selected_inspection, results_root, staging_root, options)
    if staging_root.exists() and not any(staging_root.iterdir()):
        staging_root.rmdir()
    print(f"Preprocessing complete: {len(all_rows)} cases -> {output_root}", flush=True)


if __name__ == "__main__":
    main()
