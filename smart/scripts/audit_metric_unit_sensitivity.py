#!/usr/bin/env python3
"""Audit physical- versus standardized-unit relative error without changing evaluators.

The script evaluates the existing SMART and DeAL checkpoints on fixed held-out
cases and identical sine-x/sine-y encoder views. Each prediction is scored both
before and after target de-standardization to quantify metric sensitivity.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm.auto import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
SMART_ROOT = SCRIPT_DIR.parent
REPO_ROOT = SMART_ROOT.parent
if str(SMART_ROOT) not in sys.path:
    sys.path.insert(0, str(SMART_ROOT))

from data.ahmedml_dataset_v2 import AhmedMLDatasetV2
from data.pump_dataset import PumpDataset
from data.toy_heat_exchange_dataset import ToyHeatExchangeDataset
from models.smart.smart import SMART


DEFAULTS = {
    "drivaerml": {
        "base_config": "drivaerml",
        "deal_config": "drivaerml_satloss7_range100",
        "base_checkpoint": REPO_ROOT / "checkpoints/smart-smart-drivaerml-131k16kwr-drivaerml-s42_best.pt",
        "deal_checkpoint": REPO_ROOT / "checkpoints/smart-satloss7-range100-smart-satloss7-range100-from-smart-150ep-drivaerml-s42_best.pt",
        "root": Path("/mnt/ssdraid/parsa/drivaerml_preprocessed"),
        "surface_channels": 7,
        "volume_channels": 4,
        "parameter_channels": 0,
    },
    "pump": {
        "base_config": "pump",
        "deal_config": "pump_deal_from_smart_full",
        "base_checkpoint": REPO_ROOT / "checkpoints/smart-pump-random1400-base-16k-pump-s42_best.pt",
        "deal_checkpoint": REPO_ROOT / "checkpoints/smart-pump-deal-random1400-from-smart-150ep-pump-s42_best.pt",
        "root": Path("/mnt/data/parsa/shift_pump_random1400_preprocessed"),
        "surface_channels": 7,
        "volume_channels": 4,
        "parameter_channels": 13,
    },
    "heat_exchanger": {
        "base_config": "toy_heat_exchange",
        "deal_config": "toy_heat_exchange_satloss7",
        "base_checkpoint": REPO_ROOT / "checkpoints/smart-toy-heat-exchange-heat-exchange-base-ratio-aligned-toyheatexchange-s42_best.pt",
        "deal_checkpoint": REPO_ROOT / "checkpoints/smart-toy-heat-exchange-satloss7-heat-exchange-satloss-ratio-aligned-toyheatexchange-s42_best.pt",
        "root": Path("/mnt/ssdraid/parsa/toy_heat_exchange_fem_v1"),
        "surface_channels": 1,
        "volume_channels": 1,
        "parameter_channels": 0,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=tuple(DEFAULTS), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cases", type=int, default=8)
    parser.add_argument("--query-points", type=int, default=32768)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def compose_config(name: str, stack: tuple[str, ...] = ()):
    if name in stack:
        raise ValueError(f"Cyclic configuration defaults: {' -> '.join((*stack, name))}")
    path = SMART_ROOT / "config" / f"{name}.yaml"
    current = OmegaConf.load(path)
    defaults = list(current.get("defaults", []))
    current.pop("defaults", None)
    resolved = OmegaConf.create()
    inserted_self = False
    for entry in defaults:
        if not isinstance(entry, str):
            # Hydra logging overrides affect only the launcher, not the model
            # configuration required by this inference-only audit.
            if OmegaConf.is_dict(entry) and all(str(key).startswith("override hydra/") for key in entry):
                continue
            raise ValueError(f"Unsupported config default in {path}: {entry!r}")
        if entry == "_self_":
            resolved = OmegaConf.merge(resolved, current)
            inserted_self = True
        elif entry:
            resolved = OmegaConf.merge(resolved, compose_config(entry, (*stack, name)))
    return resolved if inserted_self else OmegaConf.merge(resolved, current)


def load_model(config_name: str, checkpoint_path: Path, spec: dict, device: torch.device):
    config = compose_config(config_name)
    architecture = OmegaConf.to_container(config.experiment.architecture, resolve=True)
    model = SMART(
        spatial_dim=3,
        surface_channels=spec["surface_channels"],
        volume_channels=spec["volume_channels"],
        parameter_channels=spec["parameter_channels"],
        **architecture,
    ).to(device).eval()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
    state = {str(key).removeprefix("module."): value for key, value in state.items()}
    model.load_state_dict(state, strict=True)
    return model, config


def view_budget(config) -> int:
    for key in ("eval_view_geometry_points", "primary_view_geometry_points", "view_geometry_points", "num_body_points"):
        value = int(getattr(config.experiment, key, 0))
        if value > 0:
            return value
    raise ValueError("No positive encoder-view budget in configuration.")


def rel_l2(prediction: torch.Tensor, target: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(prediction - target) / torch.linalg.vector_norm(target).clamp_min(1.0e-12))


def combined_error(pred_s, target_s, pred_v, target_v) -> float:
    return 0.5 * (rel_l2(pred_s, target_s) + rel_l2(pred_v, target_v))


def choose_indices(count: int, budget: int, seed_values: tuple[int, ...]) -> np.ndarray:
    if budget <= 0 or budget >= count:
        return np.arange(count, dtype=np.int64)
    rng = np.random.default_rng(np.random.SeedSequence(seed_values))
    return rng.choice(count, size=budget, replace=False).astype(np.int64, copy=False)


def sine_view(geometry: np.ndarray, budget: int, axis: int, seed: int) -> np.ndarray:
    if budget >= geometry.shape[0]:
        return geometry
    coordinate = geometry[:, axis]
    normalized = (coordinate - coordinate.min()) / max(float(coordinate.max() - coordinate.min()), 1.0e-8)
    weights = np.sin(np.pi * normalized) ** 2 + 1.0e-6
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    indices = torch.multinomial(torch.from_numpy(weights.astype(np.float32)), budget, replacement=False, generator=generator)
    return geometry[indices.numpy()]


def normalize(points: np.ndarray, minimum: torch.Tensor, span: torch.Tensor) -> np.ndarray:
    return ((points - minimum.numpy()[None, :]) / span.numpy()[None, :]).astype(np.float32, copy=False)


def select_case_ids(ids, cases: int, seed: int) -> list[int]:
    ids = np.asarray(sorted(int(case_id) for case_id in ids), dtype=np.int64)
    count = min(int(cases), int(ids.size))
    rng = np.random.default_rng(seed)
    return sorted(int(value) for value in rng.choice(ids, size=count, replace=False))


def make_dataset(task: str, spec: dict):
    if task == "drivaerml":
        return AhmedMLDatasetV2(
            saved_folder=str(spec["root"]), if_test=True, geometry_points=0,
            surface_points=0, volume_points=0, require_preprocessed=True,
            geometry_density_knn_k=16, geometry_density_estimator="kde",
        )
    if task == "pump":
        return PumpDataset(saved_folder=str(spec["root"]), if_test=True, geometry_points=0, surface_points=0, volume_points=0)
    return ToyHeatExchangeDataset(saved_folder=str(spec["root"]), if_test=True, geometry_points=0, surface_points=0, volume_points=0)


def load_case(task: str, dataset, case_id: int, query_points: int, seed: int):
    run_dir = dataset._run_dir(case_id) if task != "drivaerml" else Path(dataset.file_path) / f"run_{case_id}"
    if task == "drivaerml":
        geometry = np.load(run_dir / "surface_coords.npy").astype(np.float32, copy=False)
        surface_coords = geometry
        surface_target = np.concatenate([
            np.load(run_dir / "surface_pMeanTrim.npy").astype(np.float32, copy=False).reshape(-1, 1),
            np.load(run_dir / "surface_normals.npy").astype(np.float32, copy=False)[:, :3],
            np.load(run_dir / "surface_wallShearStressMeanTrim_x.npy").astype(np.float32, copy=False).reshape(-1, 1),
            np.load(run_dir / "surface_wallShearStressMeanTrim_y.npy").astype(np.float32, copy=False).reshape(-1, 1),
            np.load(run_dir / "surface_wallShearStressMeanTrim_z.npy").astype(np.float32, copy=False).reshape(-1, 1),
        ], axis=1)
        volume_coords = np.load(run_dir / "volume_coords.npy").astype(np.float32, copy=False)
        volume_target = np.concatenate([
            np.load(run_dir / "volume_pMeanTrim.npy").astype(np.float32, copy=False).reshape(-1, 1),
            np.load(run_dir / "volume_UMeanTrim.npy").astype(np.float32, copy=False),
        ], axis=1)
        minimum, span = dataset.min_pos, dataset.max_pos - dataset.min_pos
        params = None
    else:
        arrays = dataset._get_arrays(case_id)
        geometry = np.asarray(arrays["geometry_coords"] if task == "heat_exchanger" else arrays["surface_coords"], dtype=np.float32)
        surface_coords = np.asarray(arrays["surface_coords"], dtype=np.float32)
        surface_target = np.asarray(arrays["surface_data"], dtype=np.float32)
        volume_coords = np.asarray(arrays["volume_coords"], dtype=np.float32)
        volume_target = np.asarray(arrays["volume_data"], dtype=np.float32)
        minimum, span = dataset.min_pos, dataset.position_span
        params = None if task == "heat_exchanger" else dataset.get_case_params(case_id)

    surf_idx = choose_indices(surface_coords.shape[0], query_points, (seed, case_id, 101))
    vol_idx = choose_indices(volume_coords.shape[0], query_points, (seed, case_id, 202))
    return {
        "geometry": normalize(geometry, minimum, span),
        "surface_q": normalize(surface_coords[surf_idx], minimum, span),
        "volume_q": normalize(volume_coords[vol_idx], minimum, span),
        "surface_raw": surface_target[surf_idx],
        "volume_raw": volume_target[vol_idx],
        "surface_mean": dataset.mean_surf_data.float(),
        "surface_std": dataset.std_surf_data.float().clamp_min(1.0e-12),
        "volume_mean": dataset.mean_vol_data.float(),
        "volume_std": dataset.std_vol_data.float().clamp_min(1.0e-12),
        "params": params,
    }


@torch.inference_mode()
def evaluate_model(model, device: torch.device, case: dict, geometry: np.ndarray, task: str, seed: int):
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(seed)
    geo = torch.from_numpy(np.ascontiguousarray(geometry)).unsqueeze(0).to(device)
    surface_q = torch.from_numpy(np.ascontiguousarray(case["surface_q"])).unsqueeze(0).to(device)
    volume_q = torch.from_numpy(np.ascontiguousarray(case["volume_q"])).unsqueeze(0).to(device)
    params = None
    if case["params"] is not None:
        params = torch.from_numpy(np.ascontiguousarray(case["params"], dtype=np.float32)).unsqueeze(0).to(device)
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
        prediction_s, prediction_v = model.inference(geo, surface_q, volume_q, params)
    prediction_s = prediction_s[0].float().cpu()
    prediction_v = prediction_v[0].float().cpu()
    target_s_raw = torch.from_numpy(np.ascontiguousarray(case["surface_raw"]))
    target_v_raw = torch.from_numpy(np.ascontiguousarray(case["volume_raw"]))
    target_s_standardized = (target_s_raw - case["surface_mean"]) / case["surface_std"]
    target_v_standardized = (target_v_raw - case["volume_mean"]) / case["volume_std"]
    prediction_s_raw = prediction_s * case["surface_std"] + case["surface_mean"]
    prediction_v_raw = prediction_v * case["volume_std"] + case["volume_mean"]
    return {
        "standardized": combined_error(prediction_s, target_s_standardized, prediction_v, target_v_standardized),
        "physical": combined_error(prediction_s_raw, target_s_raw, prediction_v_raw, target_v_raw),
    }


def summarize(rows: list[dict]) -> dict:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["model"], row["units"])].append(float(row["combined_rel_l2"]))
    result = {}
    for (model, units), values in grouped.items():
        result.setdefault(model, {})[units] = {
            "mean": float(np.mean(values)),
            "case_shift_std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            "n": len(values),
        }
    for units in ("standardized", "physical"):
        base = result["base"][units]["mean"]
        deal = result["deal"][units]["mean"]
        result.setdefault("paired_reduction", {})[units] = 100.0 * (base - deal) / base
    return result


def main() -> None:
    args = parse_args()
    spec = DEFAULTS[args.task]
    device = torch.device(args.device)
    if not all(Path(path).is_file() for path in (spec["base_checkpoint"], spec["deal_checkpoint"])):
        raise FileNotFoundError("One or more requested audit checkpoints are missing.")
    dataset = make_dataset(args.task, spec)
    case_ids = select_case_ids(dataset.test_ids, args.cases, args.seed)
    base_model, base_config = load_model(spec["base_config"], spec["base_checkpoint"], spec, device)
    deal_model, deal_config = load_model(spec["deal_config"], spec["deal_checkpoint"], spec, device)
    base_budget = view_budget(base_config)
    deal_budget = view_budget(deal_config)
    if base_budget != deal_budget:
        raise ValueError(f"The audit expects matched base/DeAL budgets, got {base_budget} and {deal_budget}.")

    rows = []
    for case_id in tqdm(case_ids, desc=f"{args.task} metric-unit audit"):
        case = load_case(args.task, dataset, case_id, args.query_points, args.seed)
        for shift_name, axis in (("sine_x", 0), ("sine_y", 1)):
            geometry = sine_view(case["geometry"], base_budget, axis, args.seed + case_id * 1009 + axis)
            for model_name, model in (("base", base_model), ("deal", deal_model)):
                errors = evaluate_model(model, device, case, geometry, args.task, args.seed + case_id * 1009 + axis)
                for units, value in errors.items():
                    rows.append({
                        "task": args.task,
                        "case_id": case_id,
                        "shift": shift_name,
                        "model": model_name,
                        "units": units,
                        "combined_rel_l2": value,
                    })
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "task": args.task,
        "cases": case_ids,
        "conditions": ["sine_x", "sine_y"],
        "query_points_per_domain": args.query_points,
        "input_budget": base_budget,
        "summary": summarize(rows),
    }
    (args.output_dir / f"{args.task}_metric_unit_audit.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.output_dir / f"{args.task}_metric_unit_rows.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
