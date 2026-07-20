#!/usr/bin/env python3
"""Run SMART and SATLOSS6 SMART on external surface-only CFD folders.

Each case must contain ``surface_coords.npy`` and ``surface_pMeanTrim.npy``.
The full surface cloud is used as the query cloud.  The encoder input is a
shared random sample whose size is resolved from each model's training config:
SMART uses ``num_body_points`` and SATLOSS6 uses ``view_geometry_points`` when
its full-source setting has ``num_body_points=0``.

Coordinates are normalized with the fixed DrivAerML training frame obtained
from the configured dataset.  Per-case centering/scaling is deliberately not
performed: it would change the physical coordinate frame and scale learned by
the checkpoints.  The script reports and warns about external bounds that fall
outside that training frame.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from timeit import default_timer

import numpy as np
import torch
from omegaconf import OmegaConf


SCRIPT_DIR = Path(__file__).resolve().parent
SMART_ROOT = SCRIPT_DIR.parent
REPO_ROOT = SMART_ROOT.parent
if str(SMART_ROOT) not in sys.path:
    sys.path.insert(0, str(SMART_ROOT))

from data.datasets import get_dataset
from models.smart.smart import SMART


DEFAULT_INPUT_DIR = REPO_ROOT / "CFD_audi" / "new_cfds"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "external_surface_smart_vs_satloss6"
DEFAULT_SMART_CHECKPOINT = REPO_ROOT / "checkpoints" / "smart-smart-drivaerml-131k16kwr-drivaerml-s42_best.pt"
DEFAULT_SATLOSS6_CHECKPOINT = (
    REPO_ROOT
    / "checkpoints"
    / "smart-satloss6-smart-satloss6-drivaerml-randbeta00to05-bothviews-drivaerml-s42_best.pt"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR), help="Parent directory of external CFD case folders.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Output results directory.")
    parser.add_argument("--smart-config", default="drivaerml", help="Vanilla SMART config name without .yaml.")
    parser.add_argument("--satloss6-config", default="drivaerml_satloss6", help="SATLOSS6 config name without .yaml.")
    parser.add_argument("--smart-checkpoint", default=str(DEFAULT_SMART_CHECKPOINT), help="Vanilla SMART checkpoint path.")
    parser.add_argument(
        "--satloss6-checkpoint",
        default=str(DEFAULT_SATLOSS6_CHECKPOINT),
        help="SMART SATLOSS6 checkpoint path.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed for shared encoder-input sampling and model sampling.")
    parser.add_argument("--query-chunk-size", type=int, default=65536, help="Surface query chunk size for GPU decoding.")
    parser.add_argument(
        "--query-limit",
        type=int,
        default=0,
        help="Optional limit for debugging. Zero means the complete surface cloud.",
    )
    parser.add_argument("--device", default=None, help="Torch device, for example cuda:0 or cpu. Defaults to CUDA when available.")
    parser.add_argument(
        "--out-of-frame-tolerance",
        type=float,
        default=0.05,
        help="Warn when normalized external coordinates leave [0,1] by more than this tolerance.",
    )
    return parser.parse_args()


def load_experiment_config(config_name: str):
    config_path = SMART_ROOT / "config" / f"{config_name}.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found: {config_path}")
    return OmegaConf.load(config_path).experiment


def resolve_device(device_arg: str | None) -> torch.device:
    if device_arg:
        device = torch.device(device_arg)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Requested {device}, but CUDA is not available.")
    return device


def resolve_case_dirs(input_dir: Path) -> list[Path]:
    def valid(path: Path) -> bool:
        return (path / "surface_coords.npy").is_file() and (path / "surface_pMeanTrim.npy").is_file()

    if valid(input_dir):
        return [input_dir]
    cases = [path for path in sorted(input_dir.iterdir()) if path.is_dir() and valid(path)]
    if not cases:
        raise FileNotFoundError(
            f"No external surface cases found under {input_dir}. Expected surface_coords.npy and "
            "surface_pMeanTrim.npy in the directory or its immediate subdirectories."
        )
    return cases


def load_case(case_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    coords = np.asarray(np.load(case_dir / "surface_coords.npy"), dtype=np.float32)
    pressure = np.asarray(np.load(case_dir / "surface_pMeanTrim.npy"), dtype=np.float32).reshape(-1)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"{case_dir}: surface_coords.npy must have shape [N, 3], got {coords.shape}")
    if coords.shape[0] != pressure.shape[0]:
        raise ValueError(
            f"{case_dir}: surface_coords has {coords.shape[0]} points but surface pressure has {pressure.shape[0]} values."
        )
    finite = np.isfinite(coords).all(axis=1) & np.isfinite(pressure)
    coords = coords[finite]
    pressure = pressure[finite]
    if coords.shape[0] == 0:
        raise ValueError(f"{case_dir}: no finite surface coordinates/pressure values found.")
    return coords, pressure


def resolve_geometry_budget(config) -> int:
    configured_body_points = int(getattr(config, "num_body_points", 0))
    if configured_body_points > 0:
        return configured_body_points
    view_points = int(getattr(config, "view_geometry_points", 0))
    if view_points > 0:
        return view_points
    raise ValueError(
        f"Could not resolve an encoder input budget from config {getattr(config, 'model_name', '<unknown>')}: "
        "expected a positive num_body_points or view_geometry_points."
    )


def sample_geometry_indices(point_count: int, budget: int, generator: np.random.Generator) -> np.ndarray:
    if budget <= 0 or budget == point_count:
        return np.arange(point_count, dtype=np.int64)
    if budget < point_count:
        return generator.choice(point_count, size=budget, replace=False).astype(np.int64, copy=False)
    return generator.integers(0, point_count, size=budget, dtype=np.int64)


def normalize_positions(positions: torch.Tensor, min_pos: torch.Tensor, max_pos: torch.Tensor) -> torch.Tensor:
    denominator = (max_pos - min_pos).clamp_min(1.0e-12)
    return (positions - min_pos) / denominator


def summarize(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    return {
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "std": float(values.std()),
    }


def safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", name)


def write_polydata_vtk(path: Path, points: np.ndarray, point_data: dict[str, np.ndarray]) -> None:
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"VTK points must have shape [N, 3], got {points.shape}")
    count = int(points.shape[0])
    vertices = np.empty((count, 2), dtype=">i4")
    vertices[:, 0] = 1
    vertices[:, 1] = np.arange(count, dtype=np.int32)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(b"# vtk DataFile Version 3.0\n")
        handle.write(b"External SMART surface pressure prediction\n")
        handle.write(b"BINARY\n")
        handle.write(b"DATASET POLYDATA\n")
        handle.write(f"POINTS {count} float\n".encode("ascii"))
        handle.write(points.astype(">f4", copy=False).tobytes())
        handle.write(b"\n")
        handle.write(f"VERTICES {count} {2 * count}\n".encode("ascii"))
        handle.write(vertices.tobytes())
        handle.write(b"\n")
        handle.write(f"POINT_DATA {count}\n".encode("ascii"))
        for name, values in point_data.items():
            values = np.asarray(values, dtype=np.float32).reshape(count, -1)
            handle.write(f"SCALARS {safe_name(name)} float {values.shape[1]}\n".encode("ascii"))
            handle.write(b"LOOKUP_TABLE default\n")
            handle.write(values.astype(">f4", copy=False).tobytes())
            handle.write(b"\n")


def load_model(config, checkpoint_path: Path, spatial_dim: int, surface_channels: int, volume_channels: int, device: torch.device):
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    architecture = OmegaConf.to_container(getattr(config, "architecture", {}), resolve=True)
    model_kwargs = {
        "spatial_dim": int(spatial_dim),
        "surface_channels": int(surface_channels),
        "volume_channels": int(volume_channels),
        "parameter_channels": 0,
        **architecture,
    }
    model = SMART(**model_kwargs).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    if any(key.startswith("module.") for key in state_dict):
        state_dict = {key.removeprefix("module."): value for key, value in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model, model_kwargs


def predict_surface(
    model: SMART,
    geometry_norm: torch.Tensor,
    query_norm: torch.Tensor,
    mean_pressure: torch.Tensor,
    std_pressure: torch.Tensor,
    query_chunk_size: int,
    device: torch.device,
    dtype: torch.dtype,
    amp: bool,
    seed: int,
) -> np.ndarray:
    if query_chunk_size <= 0:
        raise ValueError("query_chunk_size must be positive.")
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))
    torch.manual_seed(int(seed))
    geometry_batch = geometry_norm.unsqueeze(0).to(device, non_blocking=True)
    predictions = []
    autocast_enabled = bool(amp and device.type == "cuda")
    with torch.inference_mode():
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=autocast_enabled):
            encoded_geometry, latent_positions = model.encode(geometry_batch, None)
        for start in range(0, int(query_norm.shape[0]), int(query_chunk_size)):
            stop = min(start + int(query_chunk_size), int(query_norm.shape[0]))
            query_batch = query_norm[start:stop].unsqueeze(0).to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=autocast_enabled):
                prediction_norm = model.decode(encoded_geometry, latent_positions, None, query_batch)
            prediction = prediction_norm[0, :, 0].float().mul(std_pressure).add(mean_pressure).cpu().numpy()
            predictions.append(prediction.astype(np.float32, copy=False))
    return np.concatenate(predictions, axis=0)


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    if int(args.query_chunk_size) <= 0:
        raise ValueError("--query-chunk-size must be positive.")
    if int(args.query_limit) < 0:
        raise ValueError("--query-limit must be nonnegative.")

    device = resolve_device(args.device)
    smart_config = load_experiment_config(args.smart_config)
    satloss6_config = load_experiment_config(args.satloss6_config)

    # The dataset loader supplies the exact fixed coordinate frame and field stats
    # used by the DrivAerML training configs; external cases are not renormalized.
    _train_data, test_data, stats, spatial_dim, surface_channels, volume_channels, _params_dim, fields = get_dataset(smart_config)
    if not fields["surface"] or fields["surface"][0] != "pressure":
        raise ValueError(f"Expected pressure as the first surface field, got {fields['surface']}")
    min_pos = test_data.min_pos.float()
    max_pos = test_data.max_pos.float()
    mean_pressure = stats[0][0].float().to(device)
    std_pressure = stats[1][0].float().to(device)
    smart_budget = resolve_geometry_budget(smart_config)
    satloss6_budget = resolve_geometry_budget(satloss6_config)
    if smart_budget != satloss6_budget:
        raise ValueError(
            f"The selected configs use different external encoder budgets: SMART={smart_budget}, "
            f"SATLOSS6={satloss6_budget}. Pass matching configs/checkpoints for a fair comparison."
        )
    input_budget = smart_budget

    smart_model, smart_kwargs = load_model(
        smart_config,
        Path(args.smart_checkpoint).expanduser().resolve(),
        spatial_dim,
        surface_channels,
        volume_channels,
        device,
    )
    satloss6_model, satloss6_kwargs = load_model(
        satloss6_config,
        Path(args.satloss6_checkpoint).expanduser().resolve(),
        spatial_dim,
        surface_channels,
        volume_channels,
        device,
    )

    print(f"Input root: {input_dir}")
    print(f"Output root: {output_dir}")
    print(f"Device: {device}")
    print(f"Training coordinate frame: min={min_pos.tolist()}, max={max_pos.tolist()}")
    print(f"Shared encoder input budget: {input_budget}; full-surface queries enabled")
    print(f"SMART checkpoint: {Path(args.smart_checkpoint).expanduser().resolve()}")
    print(f"SATLOSS6 checkpoint: {Path(args.satloss6_checkpoint).expanduser().resolve()}")

    cases = resolve_case_dirs(input_dir)
    root_generator = np.random.default_rng(int(args.seed))
    autocast_dtype = getattr(torch, str(getattr(smart_config, "precision", "float16")), torch.float16)
    use_amp = bool(getattr(smart_config, "amp", True))

    for case_index, case_dir in enumerate(cases):
        case_t0 = default_timer()
        surface_coords, surface_pressure = load_case(case_dir)
        query_count = surface_coords.shape[0]
        if int(args.query_limit) > 0:
            query_count = min(query_count, int(args.query_limit))
        query_coords = surface_coords[:query_count]
        query_pressure = surface_pressure[:query_count]

        raw_min = surface_coords.min(axis=0)
        raw_max = surface_coords.max(axis=0)
        raw_center = 0.5 * (raw_min + raw_max)
        normalized_coords = normalize_positions(torch.from_numpy(query_coords), min_pos, max_pos).numpy()
        normalized_min = normalized_coords.min(axis=0)
        normalized_max = normalized_coords.max(axis=0)
        if normalized_min.min() < -float(args.out_of_frame_tolerance) or normalized_max.max() > 1.0 + float(args.out_of_frame_tolerance):
            print(
                f"[warning] {case_dir.name}: normalized coordinates exceed the training frame: "
                f"min={normalized_min.tolist()}, max={normalized_max.tolist()}"
            )

        indices = sample_geometry_indices(surface_coords.shape[0], input_budget, root_generator)
        geometry_raw = surface_coords[indices]
        geometry_norm = normalize_positions(torch.from_numpy(geometry_raw), min_pos, max_pos)
        query_norm = torch.from_numpy(normalized_coords.astype(np.float32, copy=False))
        if device.type == "cuda":
            geometry_norm = geometry_norm.pin_memory()
            query_norm = query_norm.pin_memory()

        print(
            f"[{case_index + 1}/{len(cases)}] {case_dir.name}: source={surface_coords.shape[0]}, "
            f"input={geometry_raw.shape[0]}, queries={query_coords.shape[0]}, "
            f"bbox_center={raw_center.tolist()}"
        )
        smart_t0 = default_timer()
        smart_prediction = predict_surface(
            smart_model,
            geometry_norm,
            query_norm,
            mean_pressure,
            std_pressure,
            int(args.query_chunk_size),
            device,
            autocast_dtype,
            use_amp,
            int(args.seed) + 2 * case_index,
        )
        smart_time = default_timer() - smart_t0
        sat_t0 = default_timer()
        satloss6_prediction = predict_surface(
            satloss6_model,
            geometry_norm,
            query_norm,
            mean_pressure,
            std_pressure,
            int(args.query_chunk_size),
            device,
            autocast_dtype,
            use_amp,
            int(args.seed) + 2 * case_index + 1,
        )
        satloss6_time = default_timer() - sat_t0

        case_output = output_dir / safe_name(case_dir.name)
        case_output.mkdir(parents=True, exist_ok=True)
        np.save(case_output / "shared_geometry_indices.npy", indices)
        write_polydata_vtk(
            case_output / "smart_surface_pressure.vtk",
            query_coords,
            {
                "gt_pressure": query_pressure,
                "pred_pressure": smart_prediction,
                "abs_error_pressure": np.abs(smart_prediction - query_pressure),
            },
        )
        write_polydata_vtk(
            case_output / "smart_satloss6_surface_pressure.vtk",
            query_coords,
            {
                "gt_pressure": query_pressure,
                "pred_pressure": satloss6_prediction,
                "abs_error_pressure": np.abs(satloss6_prediction - query_pressure),
            },
        )
        write_polydata_vtk(
            case_output / "smart_vs_satloss6_surface_pressure.vtk",
            query_coords,
            {
                "gt_pressure": query_pressure,
                "smart_pred_pressure": smart_prediction,
                "satloss6_pred_pressure": satloss6_prediction,
                "smart_abs_error_pressure": np.abs(smart_prediction - query_pressure),
                "satloss6_abs_error_pressure": np.abs(satloss6_prediction - query_pressure),
                "satloss6_minus_smart_pressure": satloss6_prediction - smart_prediction,
                "absolute_model_difference_pressure": np.abs(satloss6_prediction - smart_prediction),
            },
        )
        metrics = {
            "case": case_dir.name,
            "source_surface_points": int(surface_coords.shape[0]),
            "surface_query_points": int(query_coords.shape[0]),
            "encoder_input_points": int(geometry_raw.shape[0]),
            "seed": int(args.seed),
            "training_min_pos": min_pos.tolist(),
            "training_max_pos": max_pos.tolist(),
            "external_min_pos": raw_min.tolist(),
            "external_max_pos": raw_max.tolist(),
            "external_bbox_center": raw_center.tolist(),
            "normalized_query_min_pos": normalized_min.tolist(),
            "normalized_query_max_pos": normalized_max.tolist(),
            "coordinate_transform": "fixed_drivaerml_training_min_max_no_per_case_rescale",
            "smart_checkpoint": str(Path(args.smart_checkpoint).expanduser().resolve()),
            "satloss6_checkpoint": str(Path(args.satloss6_checkpoint).expanduser().resolve()),
            "smart_config": args.smart_config,
            "satloss6_config": args.satloss6_config,
            "smart_model_kwargs": smart_kwargs,
            "satloss6_model_kwargs": satloss6_kwargs,
            "smart_seconds": float(smart_time),
            "satloss6_seconds": float(satloss6_time),
            "pressure": {
                "ground_truth": summarize(query_pressure),
                "smart": summarize(smart_prediction),
                "satloss6": summarize(satloss6_prediction),
            },
        }
        with (case_output / "surface_pressure_metrics.json").open("w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2)
        print(
            f"  saved {case_output}; SMART={smart_time:.2f}s, SATLOSS6={satloss6_time:.2f}s, "
            f"case_total={default_timer() - case_t0:.2f}s"
        )


if __name__ == "__main__":
    main()
