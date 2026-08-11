#!/usr/bin/env python3
"""Export full-surface pressure predictions for SMART and SATLOSS sine views.

For each selected preprocessed DrivAerML run, this diagnostic:

* samples a training-budget encoder view with the established sine-x=1 and
  sine-y=1 sampling rules;
* keeps the complete preprocessed surface point cloud as the query cloud;
* decodes the surface in bounded chunks; and
* writes ground truth, predictions, and signed pointwise pressure errors to a
  legacy VTK vertex cloud.

The two models are placed on separate devices when two CUDA devices are
provided.  This is intentionally surface-only; no volume query is required.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import vtk
from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray

SCRIPT_DIR = Path(__file__).resolve().parent
SMART_ROOT = SCRIPT_DIR.parent
REPO_ROOT = SMART_ROOT.parent
if str(SMART_ROOT) not in sys.path:
    sys.path.insert(0, str(SMART_ROOT))

from data.ahmedml_dataset_v2 import AhmedMLDatasetV2  # noqa: E402
from scripts.compare_drivaerml_sampling_invariance import (  # noqa: E402
    build_model,
    load_cfg,
    normalize_pos,
    sample_uniform_weighted_mixture_without_replacement,
    sinusoidal_axis_probabilities,
    train_encoder_input_points,
)


DEFAULT_DATA_ROOT = Path("/mnt/ssdraid/parsa/drivaerml_preprocessed")
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "drivaerml_sine_surface_pressure_vtk"
DEFAULT_BASE_CHECKPOINT = REPO_ROOT / "checkpoints" / "smart-smart-drivaerml-131k16kwr-drivaerml-s42_best.pt"
DEFAULT_SATLOSS_CHECKPOINT = REPO_ROOT / "checkpoints" / "smart-satloss7-smart-satloss7-drivaerml-131k-drivaerml-s42_best.pt"


def parse_devices(value: str) -> list[torch.device]:
    devices = [torch.device(item.strip()) for item in str(value).split(",") if item.strip()]
    if not devices:
        raise ValueError("At least one device is required.")
    for device in devices:
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(f"Requested {device}, but CUDA is unavailable.")
    return devices


def parse_run_ids(value: str | None, available: list[int], num_runs: int, seed: int) -> list[int]:
    available_set = {int(item) for item in available}
    if value:
        run_ids = [int(item.strip()) for item in str(value).split(",") if item.strip()]
        missing = sorted(set(run_ids) - available_set)
        if missing:
            raise ValueError(f"Requested run IDs are not available: {missing}")
        return run_ids
    if num_runs <= 0:
        raise ValueError("--num-runs must be positive when --run-ids is omitted.")
    count = min(int(num_runs), len(available))
    rng = np.random.default_rng(int(seed) + 71231)
    return sorted(int(item) for item in rng.choice(np.asarray(available), size=count, replace=False))


def sample_sine_view_indices(
    surface_coords: np.ndarray,
    input_budget: int,
    axis: int,
    seed: int,
) -> np.ndarray:
    if input_budget <= 0:
        raise ValueError(f"Encoder input budget must be positive, got {input_budget}.")
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(axis), 991]))
    weights = sinusoidal_axis_probabilities(surface_coords, axis=axis)
    return sample_uniform_weighted_mixture_without_replacement(
        weights,
        int(input_budget),
        1.0,
        rng,
    )


@torch.inference_mode()
def predict_surface_pressure(
    model: torch.nn.Module,
    device: torch.device,
    geometry_norm_cpu: torch.Tensor,
    query_norm_cpu: torch.Tensor,
    mean_pressure: float,
    std_pressure: float,
    query_chunk_size: int,
    seed: int,
) -> np.ndarray:
    """Encode one view and decode the complete surface query in chunks."""
    if int(query_chunk_size) <= 0:
        raise ValueError("--query-chunk-size must be positive.")
    context = torch.cuda.device(device) if device.type == "cuda" else nullcontext()
    with context:
        if device.type == "cuda":
            torch.cuda.manual_seed(int(seed))
        else:
            torch.manual_seed(int(seed))

        geometry = geometry_norm_cpu.unsqueeze(0).to(device, non_blocking=True)
        autocast_enabled = device.type == "cuda"
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=autocast_enabled):
            intermediate, latent_pos = model.encode(geometry, None)
            prediction = np.empty((int(query_norm_cpu.shape[0]),), dtype=np.float32)
            for start in range(0, int(query_norm_cpu.shape[0]), int(query_chunk_size)):
                stop = min(start + int(query_chunk_size), int(query_norm_cpu.shape[0]))
                query = query_norm_cpu[start:stop].unsqueeze(0).to(device, non_blocking=True)
                normalized = model.decode(intermediate, latent_pos, None, query)
                prediction[start:stop] = normalized[0, :, 0].float().cpu().numpy()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    return prediction * float(std_pressure) + float(mean_pressure)


def load_pressure_case(data_root: Path, run_id: int) -> tuple[np.ndarray, np.ndarray]:
    run_dir = data_root / f"run_{int(run_id)}"
    coords_path = run_dir / "surface_coords.npy"
    pressure_path = run_dir / "surface_pMeanTrim.npy"
    if not coords_path.is_file() or not pressure_path.is_file():
        raise FileNotFoundError(f"run_{run_id} is missing {coords_path.name} or {pressure_path.name}.")
    coords = np.ascontiguousarray(np.asarray(np.load(coords_path, mmap_mode="r"), dtype=np.float32))
    pressure = np.asarray(np.load(pressure_path, mmap_mode="r"), dtype=np.float32).reshape(-1)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"run_{run_id}: surface_coords.npy must have shape [N, 3], got {coords.shape}.")
    if coords.shape[0] != pressure.shape[0]:
        raise ValueError(
            f"run_{run_id}: surface coordinate count {coords.shape[0]} does not match pressure count {pressure.shape[0]}.")
    if not np.isfinite(coords).all() or not np.isfinite(pressure).all():
        raise ValueError(f"run_{run_id}: surface coordinates or pressure contain non-finite values.")
    return coords, pressure


def write_surface_vtk(path: Path, points: np.ndarray, point_data: OrderedDict[str, np.ndarray]) -> None:
    """Write a native binary legacy VTK vertex cloud with all point arrays."""
    points = np.ascontiguousarray(np.asarray(points, dtype=np.float32))
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"VTK points must have shape [N, 3], got {points.shape}.")
    count = int(points.shape[0])
    vtk_points = vtk.vtkPoints()
    vtk_points.SetData(numpy_to_vtk(points, deep=True))

    vertices = vtk.vtkCellArray()
    vertices.SetData(
        numpy_to_vtkIdTypeArray(np.arange(count + 1, dtype=np.int64), deep=True),
        numpy_to_vtkIdTypeArray(np.arange(count, dtype=np.int64), deep=True),
    )
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(vtk_points)
    polydata.SetVerts(vertices)

    for name, values in point_data.items():
        values = np.ascontiguousarray(np.asarray(values, dtype=np.float32))
        if values.ndim == 1:
            values = values[:, None]
        if values.ndim != 2 or values.shape[0] != count:
            raise ValueError(f"Point-data '{name}' must have {count} rows, got {values.shape}.")
        vtk_array = numpy_to_vtk(values, deep=True)
        vtk_array.SetName(str(name))
        polydata.GetPointData().AddArray(vtk_array)

    path.parent.mkdir(parents=True, exist_ok=True)
    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(str(path))
    writer.SetFileTypeToBinary()
    writer.SetInputData(polydata)
    if writer.Write() != 1:
        raise IOError(f"VTK writer failed for {path}.")


def evaluate_model_views(
    model_name: str,
    model: torch.nn.Module,
    device: torch.device,
    geometry_views: dict[str, torch.Tensor],
    query_norm: torch.Tensor,
    mean_pressure: float,
    std_pressure: float,
    query_chunk_size: int,
    seed: int,
) -> dict[str, np.ndarray]:
    predictions = {}
    for shift_index, shift_name in enumerate(("sin_x_1", "sin_y_1")):
        predictions[shift_name] = predict_surface_pressure(
            model=model,
            device=device,
            geometry_norm_cpu=geometry_views[shift_name],
            query_norm_cpu=query_norm,
            mean_pressure=mean_pressure,
            std_pressure=std_pressure,
            query_chunk_size=query_chunk_size,
            seed=seed + 1009 * shift_index + (0 if model_name == "base" else 500003),
        )
    return predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--base-config", default="drivaerml")
    parser.add_argument("--satloss-config", default="drivaerml_satloss7")
    parser.add_argument("--base-checkpoint", type=Path, default=DEFAULT_BASE_CHECKPOINT)
    parser.add_argument("--satloss-checkpoint", type=Path, default=DEFAULT_SATLOSS_CHECKPOINT)
    parser.add_argument("--run-ids", default="34", help="Comma-separated run IDs. Defaults to run 34.")
    parser.add_argument("--num-runs", type=int, default=1, help="Used only when --run-ids is empty.")
    parser.add_argument("--split", choices=("test", "all"), default="test")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--query-chunk-size", type=int, default=65536)
    parser.add_argument(
        "--query-limit",
        type=int,
        default=0,
        help="Debug-only prefix limit for the surface query. Zero keeps the complete surface cloud.",
    )
    parser.add_argument("--devices", default="cuda:0,cuda:1", help="Base and SATLOSS devices in that order.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.query_limit < 0:
        raise ValueError("--query-limit must be zero or positive.")
    data_root = args.data_root.expanduser().resolve()
    base_checkpoint = args.base_checkpoint.expanduser().resolve()
    satloss_checkpoint = args.satloss_checkpoint.expanduser().resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(data_root)
    if not base_checkpoint.is_file():
        raise FileNotFoundError(base_checkpoint)
    if not satloss_checkpoint.is_file():
        raise FileNotFoundError(satloss_checkpoint)
    devices = parse_devices(args.devices)
    base_device = devices[0]
    satloss_device = devices[1] if len(devices) > 1 else devices[0]

    base_cfg = load_cfg(args.base_config)
    satloss_cfg = load_cfg(args.satloss_config)
    base_budget = train_encoder_input_points(base_cfg, "SMART")
    satloss_budget = train_encoder_input_points(satloss_cfg, "SMART_SATLOSS7")
    if base_budget <= 0 or satloss_budget <= 0:
        raise ValueError(f"Invalid encoder budgets: base={base_budget}, satloss={satloss_budget}.")
    print(f"Base config={args.base_config}, encoder budget={base_budget}, device={base_device}", flush=True)
    print(f"SATLOSS config={args.satloss_config}, encoder budget={satloss_budget}, device={satloss_device}", flush=True)

    stats_dataset = AhmedMLDatasetV2(
        saved_folder=str(data_root),
        if_test=True,
        geometry_points=0,
        surface_points=0,
        volume_points=0,
        scale_positions=bool(getattr(base_cfg, "scale_positions", False)),
        require_preprocessed=True,
    )
    available = stats_dataset.test_ids if args.split == "test" else stats_dataset.all_ids
    run_ids = parse_run_ids(args.run_ids, available, args.num_runs, args.seed)
    min_pos = stats_dataset.min_pos.cpu()
    max_pos = stats_dataset.max_pos.cpu()
    mean_pressure = float(stats_dataset.mean_surf_data[0].item())
    std_pressure = float(max(stats_dataset.std_surf_data[0].item(), 1.0e-12))

    models = {
        "base": build_model(base_cfg, str(base_checkpoint), base_device, args.query_chunk_size).to(base_device),
        "satloss": build_model(satloss_cfg, str(satloss_checkpoint), satloss_device, args.query_chunk_size).to(satloss_device),
    }
    model_devices = {"base": base_device, "satloss": satloss_device}
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "data_root": str(data_root),
        "run_ids": run_ids,
        "split": args.split,
        "seed": int(args.seed),
        "query": "complete preprocessed surface_coords.npy for each run",
        "query_chunk_size": int(args.query_chunk_size),
        "sine_sampling": "uniform weighted mixture with mix_fraction=1.0 using sin(pi*t)^2 on x or y",
        "normalization": {
            "position_min": min_pos.tolist(),
            "position_max": max_pos.tolist(),
            "pressure_mean": mean_pressure,
            "pressure_std": std_pressure,
        },
        "models": {
            "base": {"config": args.base_config, "checkpoint": str(base_checkpoint), "device": str(base_device), "encoder_budget": base_budget},
            "satloss": {"config": args.satloss_config, "checkpoint": str(satloss_checkpoint), "device": str(satloss_device), "encoder_budget": satloss_budget},
        },
        "vtk_fields": [
            "gt_pressure",
            "sin_x_1_satloss_pred", "sin_x_1_satloss_err",
            "sin_y_1_satloss_pred", "sin_y_1_satloss_err",
            "sin_x_1_base_pred", "sin_x_1_base_err",
            "sin_y_1_base_pred", "sin_y_1_base_err",
            "sin_y_1_smart_pred", "sin_y_1_smart_err",
        ],
    }

    for run_index, run_id in enumerate(run_ids):
        source_coords, source_pressure = load_pressure_case(data_root, run_id)
        if args.query_limit > 0 and args.query_limit < source_coords.shape[0]:
            query_coords = np.ascontiguousarray(source_coords[: args.query_limit]).copy()
            gt_pressure = np.ascontiguousarray(source_pressure[: args.query_limit]).copy()
        else:
            query_coords = source_coords
            gt_pressure = source_pressure
        query_norm = normalize_pos(torch.from_numpy(np.ascontiguousarray(query_coords).copy()), min_pos, max_pos).contiguous()
        geometry_views = {}
        for shift_index, (shift_name, axis) in enumerate((("sin_x_1", 0), ("sin_y_1", 1))):
            base_idx = sample_sine_view_indices(source_coords, base_budget, axis, args.seed + run_id * 100003 + shift_index)
            satloss_idx = sample_sine_view_indices(source_coords, satloss_budget, axis, args.seed + run_id * 100003 + shift_index)
            # Current SMART/SATLOSS configurations use the same 131K budget;
            # retain separate tensors so this remains correct if that changes.
            geometry_views.setdefault("base", {})[shift_name] = normalize_pos(
                torch.from_numpy(np.ascontiguousarray(source_coords[base_idx]).copy()), min_pos, max_pos
            ).contiguous()
            geometry_views.setdefault("satloss", {})[shift_name] = normalize_pos(
                torch.from_numpy(np.ascontiguousarray(source_coords[satloss_idx]).copy()), min_pos, max_pos
            ).contiguous()
            print(
                f"run_{run_id} {shift_name}: source={source_coords.shape[0]} query={query_coords.shape[0]} "
                f"base_input={base_idx.size} satloss_input={satloss_idx.size}",
                flush=True,
            )

        with ThreadPoolExecutor(max_workers=2 if len(devices) > 1 else 1) as executor:
            futures = {
                model_name: executor.submit(
                    evaluate_model_views,
                    model_name,
                    models[model_name],
                    model_devices[model_name],
                    geometry_views[model_name],
                    query_norm,
                    mean_pressure,
                    std_pressure,
                    args.query_chunk_size,
                    args.seed + run_id * 100003,
                )
                for model_name in ("base", "satloss")
            }
            predictions = {model_name: future.result() for model_name, future in futures.items()}

        fields = OrderedDict({"gt_pressure": gt_pressure.astype(np.float32, copy=False)})
        for shift_name in ("sin_x_1", "sin_y_1"):
            satloss_prediction = predictions["satloss"][shift_name]
            base_prediction = predictions["base"][shift_name]
            fields[f"{shift_name}_satloss_pred"] = satloss_prediction
            fields[f"{shift_name}_satloss_err"] = satloss_prediction - gt_pressure
            fields[f"{shift_name}_base_pred"] = base_prediction
            fields[f"{shift_name}_base_err"] = base_prediction - gt_pressure
        # Keep the requested smart spelling as an alias while the canonical
        # vanilla fields remain consistently named *_base_* above.
        fields["sin_y_1_smart_pred"] = fields["sin_y_1_base_pred"]
        fields["sin_y_1_smart_err"] = fields["sin_y_1_base_err"]
        vtk_path = output_dir / f"run_{int(run_id)}_surface_pressure_sine_endpoints.vtk"
        write_surface_vtk(vtk_path, query_coords, fields)
        print(f"[{run_index + 1}/{len(run_ids)}] wrote {vtk_path} ({query_coords.shape[0]} query points)", flush=True)

        manifest.setdefault("outputs", {})[str(run_id)] = {
            "vtk": str(vtk_path),
            "surface_points": int(query_coords.shape[0]),
            "gt_pressure_min": float(gt_pressure.min()),
            "gt_pressure_max": float(gt_pressure.max()),
        }

    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Manifest: {output_dir / 'manifest.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
