#!/usr/bin/env python3
"""Diagnose C-core Transolver++ train/evaluation and view-sampling mismatches.

This is deliberately inference-only.  It measures the same checkpoints at the
same physical queries under (1) the base trainer's native sampled geometry,
(2) the DeAL evaluator's fixed-seed uniform view, and (3) the local
deterministic versus the official Transolver++ Gumbel slice assignment rule.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from data.datasets import get_dataset
from models.transolverpp import PhysicsAttention1DEidetic, TransolverPP
from train_consistency_common import sample_geometry_view


ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "smart" / "config"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--base-config", default="c_core_magnetic_transolverpp")
    parser.add_argument("--deal-config", default="c_core_magnetic_transolverpp_deal_from_base")
    parser.add_argument("--base-best-checkpoint", type=Path, required=True)
    parser.add_argument("--base-last-checkpoint", type=Path, required=True)
    parser.add_argument("--deal-best-checkpoint", type=Path, required=True)
    parser.add_argument("--deal-last-checkpoint", type=Path, required=True)
    parser.add_argument("--case-ids", default="256")
    parser.add_argument("--gumbel-repeats", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def compose_config(name: str):
    with initialize_config_dir(version_base="1.2", config_dir=str(CONFIG_DIR)):
        return compose(config_name=name)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def state_dict(path: Path) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    value = payload.get("model_state_dict", payload.get("model_state", payload.get("state_dict", payload)))
    if not isinstance(value, dict):
        raise TypeError(f"{path} does not contain a model state dictionary.")
    return {str(key).removeprefix("module."): tensor for key, tensor in value.items()}


def checkpoint_metadata(path: Path) -> dict[str, object]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return {
        "path": str(path.resolve()),
        "epoch": int(payload.get("epoch", -1)),
        "sha256": digest.hexdigest(),
        "best_rel_l2": float(payload.get("best_rel_l2", payload.get("best_robust_rel_l2", float("nan")))),
    }


def build_model(cfg, checkpoint: Path, device: torch.device, channels: tuple[int, int, int]) -> TransolverPP:
    architecture = OmegaConf.to_container(cfg.experiment.architecture, resolve=True)
    if not isinstance(architecture, dict):
        raise TypeError("experiment.architecture must be a mapping")
    valid = set(TransolverPP.__mro__[1].__init__.__code__.co_varnames)
    architecture = {key: value for key, value in architecture.items() if key in valid}
    model = TransolverPP(
        spatial_dim=3,
        surface_channels=channels[0],
        volume_channels=channels[1],
        parameter_channels=channels[2],
        **architecture,
    ).to(device)
    missing, unexpected = model.load_state_dict(state_dict(checkpoint), strict=False)
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint/config mismatch for {checkpoint}: missing={missing}, unexpected={unexpected}")
    return model.eval()


def relative_l2(prediction: torch.Tensor, target: torch.Tensor) -> float:
    return float(
        torch.linalg.vector_norm(prediction.float() - target.float())
        / torch.linalg.vector_norm(target.float()).clamp_min(1.0e-12)
    )


def predict(model, device: torch.device, geometry, surf_q, surf_y, vol_q, vol_y, dataset) -> dict[str, float]:
    with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
        pred_s, pred_v = model.inference(
            geometry.unsqueeze(0).to(device, non_blocking=True),
            surf_q.unsqueeze(0).to(device, non_blocking=True),
            vol_q.unsqueeze(0).to(device, non_blocking=True),
            None,
        )
    pred_s, pred_v = pred_s[0].float().cpu(), pred_v[0].float().cpu()
    surf_error = relative_l2(
        pred_s * dataset.std_surf_data.float() + dataset.mean_surf_data.float(),
        surf_y * dataset.std_surf_data.float() + dataset.mean_surf_data.float(),
    )
    vol_error = relative_l2(
        pred_v * dataset.std_vol_data.float() + dataset.mean_vol_data.float(),
        vol_y * dataset.std_vol_data.float() + dataset.mean_vol_data.float(),
    )
    return {
        "surface_physical_rel_l2": surf_error,
        "volume_physical_rel_l2": vol_error,
        "combined_physical_rel_l2": 0.5 * (surf_error + vol_error),
    }


def run_with_assignment(model, assignment: str, seed: int, *args) -> dict[str, float]:
    """Evaluate one assignment rule while preserving eval-mode dropout."""
    attentions = [module for module in model.modules() if isinstance(module, PhysicsAttention1DEidetic)]
    if not attentions:
        raise RuntimeError("No Transolver++ physics-attention module found.")
    if any(child.training for module in attentions for child in module.to_out.modules() if isinstance(child, torch.nn.Dropout)):
        raise RuntimeError("Expected eval-mode dropout before the Gumbel diagnostic.")
    original = [module.slice_assignment_mode for module in attentions]
    try:
        # Changing only this non-parameter mode isolates assignment behavior;
        # model.eval() keeps all Dropout modules disabled in both variants.
        for module in attentions:
            module.slice_assignment_mode = assignment
        seed_all(seed)
        return predict(model, *args)
    finally:
        for module, state in zip(attentions, original):
            module.slice_assignment_mode = state


def unpack(item, parameter_channels: int):
    if parameter_channels:
        geometry, surf_q, surf_y, vol_q, vol_y, _params, *optional_density = item
    else:
        geometry, surf_q, surf_y, vol_q, vol_y, *optional_density = item
    density = optional_density[0] if optional_density else None
    return geometry, surf_q, surf_y, vol_q, vol_y, density


def summarize(values: list[dict[str, float]]) -> dict[str, float]:
    output: dict[str, float] = {}
    for key in values[0]:
        array = np.asarray([value[key] for value in values], dtype=np.float64)
        output[f"{key}_mean"] = float(array.mean())
        output[f"{key}_std"] = float(array.std(ddof=1)) if len(array) > 1 else 0.0
    return output


def main() -> None:
    args = parse_args()
    args.data_root = args.data_root.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    paths = [
        args.base_best_checkpoint, args.base_last_checkpoint,
        args.deal_best_checkpoint, args.deal_last_checkpoint,
    ]
    if missing := [str(path) for path in paths if not path.is_file()]:
        raise FileNotFoundError("Missing checkpoints: " + ", ".join(missing))

    base_cfg, deal_cfg = compose_config(args.base_config), compose_config(args.deal_config)
    native_cfg = OmegaConf.create(OmegaConf.to_container(base_cfg, resolve=True))
    native_cfg.experiment.data_path = str(args.data_root)
    native_cfg.experiment.geometry_epoch_seeded_sampling = True
    full_cfg = OmegaConf.create(OmegaConf.to_container(deal_cfg, resolve=True))
    full_cfg.experiment.data_path = str(args.data_root)
    full_cfg.experiment.num_body_points = 0
    full_cfg.experiment.geometry_epoch_seeded_sampling = True

    _, native_dataset, _, spatial_dim, surf_channels, vol_channels, parameter_channels, _ = get_dataset(native_cfg.experiment)
    _, full_dataset, _, _, _, _, _, _ = get_dataset(full_cfg.experiment)
    if spatial_dim != 3 or native_dataset.data != full_dataset.data:
        raise RuntimeError("Expected identical 3D C-core validation case ordering.")
    native_dataset.set_epoch(0)
    full_dataset.set_epoch(0)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but unavailable.")
    channels = (surf_channels, vol_channels, parameter_channels)
    checkpoints = {
        "base_best": args.base_best_checkpoint,
        "base_last": args.base_last_checkpoint,
        "deal_best": args.deal_best_checkpoint,
        "deal_last": args.deal_last_checkpoint,
    }
    models = {name: build_model(base_cfg, path, device, channels) for name, path in checkpoints.items()}
    case_ids = [int(value.strip()) for value in args.case_ids.split(",") if value.strip()]
    rows: list[dict[str, object]] = []
    budget = int(getattr(deal_cfg.experiment, "eval_view_geometry_points"))
    if int(getattr(base_cfg.experiment, "num_body_points")) != budget:
        raise RuntimeError("Base and DeAL encoder budgets are not matched.")

    for case_id in case_ids:
        if case_id not in native_dataset.data:
            raise ValueError(f"Case {case_id} is not in the C-core validation split.")
        index = native_dataset.data.index(case_id)
        seed_all(args.seed + case_id * 1_000_003)
        native_geo, surf_q, surf_y, vol_q, vol_y, _ = unpack(native_dataset[index], parameter_channels)
        seed_all(args.seed + case_id * 1_000_003)
        full_geo, full_surf_q, full_surf_y, full_vol_q, full_vol_y, full_density = unpack(full_dataset[index], parameter_channels)
        if not (torch.equal(surf_q, full_surf_q) and torch.equal(vol_q, full_vol_q)):
            raise RuntimeError("Native and full datasets produced different fixed physical queries.")
        uniform_geo, _, _ = sample_geometry_view(
            full_geo.unsqueeze(0), full_density.unsqueeze(0), budget, "uniform_wor", 0.0, 0.0,
            args.seed + case_id * 10007 + 17,
        )
        views = {"base_native_sample": native_geo, "deal_uniform_view": uniform_geo[0]}
        for checkpoint_name, model in models.items():
            for view_name, geometry in views.items():
                current = run_with_assignment(
                    model,
                    "deterministic_softmax",
                    args.seed + case_id * 100_003,
                    device,
                    geometry,
                    surf_q,
                    surf_y,
                    vol_q,
                    vol_y,
                    native_dataset,
                )
                rows.append({"case_id": case_id, "checkpoint": checkpoint_name, "view": view_name, "assignment": "local_deterministic_softmax", "repeat": 0, **current})
                official = [
                    run_with_assignment(
                        model,
                        "official_gumbel",
                        args.seed + case_id * 100_003 + repeat * 101,
                        device, geometry, surf_q, surf_y, vol_q, vol_y, native_dataset,
                    )
                    for repeat in range(args.gumbel_repeats)
                ]
                for repeat, result in enumerate(official):
                    rows.append({"case_id": case_id, "checkpoint": checkpoint_name, "view": view_name, "assignment": "official_gumbel_eval", "repeat": repeat, **result})
        print(f"audited case {case_id}", flush=True)

    grouped: dict[tuple[str, str, str], list[dict[str, float]]] = {}
    for row in rows:
        grouped.setdefault((str(row["checkpoint"]), str(row["view"]), str(row["assignment"])), []).append(
            {key: float(row[key]) for key in ("surface_physical_rel_l2", "volume_physical_rel_l2", "combined_physical_rel_l2")}
        )
    summary = [
        {"checkpoint": key[0], "view": key[1], "assignment": key[2], "samples": len(values), **summarize(values)}
        for key, values in sorted(grouped.items())
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = {
        "purpose": "C-core Transolver++ inference-only regression audit",
        "config": {"base": args.base_config, "deal": args.deal_config},
        "case_ids": case_ids,
        "encoder_budget": budget,
        "query_budgets": {"surface": int(native_dataset.surface_points), "volume": int(native_dataset.volume_points)},
        "gumbel_repeats": args.gumbel_repeats,
        "checkpoints": {name: checkpoint_metadata(path) for name, path in checkpoints.items()},
        "summary": summary,
        "rows": rows,
    }
    (args.output_dir / "transolverpp_c_core_regression_audit.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    for item in summary:
        print(
            f"{item['checkpoint']:10s} | {item['view']:18s} | {item['assignment']:27s} | "
            f"{item['combined_physical_rel_l2_mean']:.5f} +/- {item['combined_physical_rel_l2_std']:.5f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
