#!/usr/bin/env python3
"""Evaluate one base/DeAL architecture pair on fixed sampling shifts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from data.datasets import get_dataset
from models.ab_upt import ABUPT
from models.geo_fno import GeoFNO
from models.lno import LNO
from models.mspt import MSPT
from models.point_transformer_v3 import PointTransformerV3
from models.pointnet2_ssg import PointNet2SSG
from models.smart.smart import SMART
from models.transolverpp import TransolverPP
from scripts.audit_new_architecture_base_sampling import (
    bounds,
    checkpoint_state,
    compose_config,
    read_vtp_mesh,
    read_vtp_points,
    rel_l2,
    sample_mesh_geometry,
    unpack_item,
)
from train_consistency_common import sample_geometry_view


CONSTRUCTORS = {
    "smart": SMART,
    "ab_upt": ABUPT,
    "geo_fno": GeoFNO,
    "pointnet2_ssg": PointNet2SSG,
    "lno": LNO,
    "mspt": MSPT,
    "transolverpp": TransolverPP,
    "point_transformer_v3": PointTransformerV3,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("drivaerml", "heat_exchanger", "pump", "c_core"), required=True)
    parser.add_argument("--model", choices=tuple(CONSTRUCTORS), required=True)
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--deal-config", required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--deal-checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--remesh-root", type=Path, required=True)
    parser.add_argument("--methods", default="feature,quadric,voxel")
    parser.add_argument("--factors", default="5,10")
    parser.add_argument("--num-cases", type=int, default=5)
    parser.add_argument("--case-ids", default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def checkpoint_metadata(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path.resolve()), "epoch": int(payload.get("epoch", -1)), "sha256": digest.hexdigest()}


def input_budget(cfg) -> int:
    exp = cfg.experiment
    for key in ("primary_view_geometry_points", "view_geometry_points", "num_body_points"):
        value = int(getattr(exp, key, 0))
        if value > 0:
            return value
    raise ValueError("Configuration has no positive encoder input budget.")


def build_model(model_name: str, cfg, checkpoint: Path, device: torch.device, channels: tuple[int, int, int]):
    architecture = OmegaConf.to_container(cfg.experiment.architecture, resolve=True)
    if not isinstance(architecture, dict):
        raise TypeError("experiment.architecture must be a mapping")
    model = CONSTRUCTORS[model_name](
        spatial_dim=3,
        surface_channels=channels[0],
        volume_channels=channels[1],
        parameter_channels=channels[2],
        **architecture,
    ).to(device)
    missing, unexpected = model.load_state_dict(checkpoint_state(checkpoint), strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"Checkpoint/config mismatch: missing={len(missing)}, unexpected={len(unexpected)}"
        )
    return model.eval()


def remesh_path(dataset: str, root: Path, case_id: int, method: str, factor: int) -> Path:
    if dataset == "drivaerml":
        return root / method / f"run_{case_id}" / f"drivaer_{case_id}_faces_div{factor}.vtp"
    if dataset == "heat_exchanger":
        return root / method / f"case_{case_id:05d}" / f"heat_exchange_case_{case_id:05d}_surface_faces_div{factor}.vtp"
    if dataset == "pump":
        return root / method / f"sample_{case_id:06d}" / f"merged_surfaces_faces_div{factor}.vtp"
    return root / method / f"case_{case_id:05d}" / f"case_{case_id:05d}_solid_surface_faces_div{factor}.vtp"


def complete_remeshes(dataset: str, root: Path, case_id: int, methods: list[str], factors: list[int]) -> bool:
    return all(remesh_path(dataset, root, case_id, method, factor).is_file() for method in methods for factor in factors)


def remesh_geometry(path: Path, budget: int, seed: int, dataset: str) -> np.ndarray:
    if dataset == "heat_exchanger":
        return sample_mesh_geometry(path, budget, seed, dataset)
    points = read_vtp_points(path)
    if not len(points):
        raise ValueError(f"{path} has no vertices")
    rng = np.random.default_rng(seed)
    return np.ascontiguousarray(
        points[rng.choice(len(points), budget, replace=len(points) < budget)],
        dtype=np.float32,
    )


def predict(model, device, geometry, surf_q, surf_y, vol_q, vol_y, params, dataset, *, amp_enabled: bool) -> dict[str, float]:
    geometry = geometry.unsqueeze(0).to(device, non_blocking=True)
    surf_q = surf_q.unsqueeze(0).to(device, non_blocking=True)
    vol_q = vol_q.unsqueeze(0).to(device, non_blocking=True)
    params = None if params is None else params.unsqueeze(0).to(device, non_blocking=True)
    with torch.inference_mode(), torch.autocast(
        device_type=device.type,
        dtype=torch.float16,
        enabled=device.type == "cuda" and amp_enabled,
    ):
        pred_s, pred_v = model.inference(geometry, surf_q, vol_q, params)
    pred_s, pred_v = pred_s[0].float().cpu(), pred_v[0].float().cpu()
    if not torch.isfinite(pred_s).all() or not torch.isfinite(pred_v).all():
        raise FloatingPointError("Inference produced non-finite predictions")
    standard_s, standard_v = rel_l2(pred_s, surf_y), rel_l2(pred_v, vol_y)
    surf_mean, surf_std = dataset.mean_surf_data.float(), dataset.std_surf_data.float()
    vol_mean, vol_std = dataset.mean_vol_data.float(), dataset.std_vol_data.float()
    physical_s = rel_l2(pred_s * surf_std + surf_mean, surf_y * surf_std + surf_mean)
    physical_v = rel_l2(pred_v * vol_std + vol_mean, vol_y * vol_std + vol_mean)
    return {
        "surface_standardized_rel_l2": standard_s,
        "volume_standardized_rel_l2": standard_v,
        "combined_standardized_rel_l2": 0.5 * (standard_s + standard_v),
        "surface_physical_rel_l2": physical_s,
        "volume_physical_rel_l2": physical_v,
        "combined_physical_rel_l2": 0.5 * (physical_s + physical_v),
    }


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metric_names = [key for key in rows[0] if key.endswith("_rel_l2")]
    conditions = sorted({str(row["condition"]) for row in rows})
    # Add the requested means over remeshing methods at each reduction factor.
    for factor in sorted({int(row["factor"]) for row in rows if int(row["factor"]) > 0}):
        conditions.append(f"remesh_mean_div{factor}")
    output = []
    for condition in conditions:
        if condition.startswith("remesh_mean_div"):
            factor = int(condition.rsplit("div", 1)[1])
            selected = [row for row in rows if int(row["factor"]) == factor]
        else:
            selected = [row for row in rows if row["condition"] == condition]
        for metric in metric_names:
            base = np.asarray([row[metric] for row in selected if row["variant"] == "base"], dtype=np.float64)
            deal = np.asarray([row[metric] for row in selected if row["variant"] == "deal"], dtype=np.float64)
            if len(base) != len(deal) or not len(base):
                raise RuntimeError(f"Unpaired rows for {condition}/{metric}")
            base_mean, deal_mean = float(base.mean()), float(deal.mean())
            output.append({
                "condition": condition,
                "metric": metric,
                "samples": int(len(base)),
                "base_mean": base_mean,
                "base_std": float(base.std(ddof=1)) if len(base) > 1 else 0.0,
                "deal_mean": deal_mean,
                "deal_std": float(deal.std(ddof=1)) if len(deal) > 1 else 0.0,
                "deal_improvement_percent": 100.0 * (base_mean - deal_mean) / max(base_mean, 1.0e-12),
            })
    return output


def main() -> None:
    args = parse_args()
    methods = [item.strip() for item in args.methods.split(",") if item.strip()]
    factors = [int(item) for item in args.factors.split(",") if item.strip()]
    device = torch.device(args.device)
    base_cfg, deal_cfg = compose_config(args.base_config), compose_config(args.deal_config)
    budget = input_budget(deal_cfg)
    dataset_cfg = OmegaConf.create(OmegaConf.to_container(deal_cfg, resolve=True))
    dataset_cfg.experiment.data_path = str(args.data_root.resolve())
    dataset_cfg.experiment.num_body_points = 0
    dataset_cfg.experiment.model_name = "PAIRED_DEAL_SAMPLING_AUDIT"
    _, dataset, _, spatial_dim, surf_channels, vol_channels, parameter_channels, _ = get_dataset(dataset_cfg.experiment)
    if spatial_dim != 3:
        raise ValueError("Only 3D datasets are supported")
    dataset.set_epoch(0)
    requested = [int(item) for item in args.case_ids.split(",") if item.strip()]
    if requested:
        indices = [dataset.data.index(case_id) for case_id in requested]
    else:
        indices = [
            index for index, case_id in enumerate(dataset.data)
            if complete_remeshes(args.dataset, args.remesh_root, int(case_id), methods, factors)
        ][: args.num_cases]
    if len(indices) != args.num_cases:
        raise RuntimeError(f"Found only {len(indices)} complete cases, requested {args.num_cases}")

    lower, span = bounds(dataset)
    cases = []
    for index in indices:
        geo, surf_q, surf_y, vol_q, vol_y, params, density = unpack_item(dataset[index], parameter_channels)
        case_id = int(dataset.data[index])
        sine_x, _, _ = sample_geometry_view(
            geo.unsqueeze(0), density.unsqueeze(0), budget, "sinusoidal_axis_mixture_wor",
            0.0, 0.0, args.seed + case_id + 101, sinusoidal_axis=0, sinusoidal_mix_fraction=1.0,
        )
        sine_y, _, _ = sample_geometry_view(
            geo.unsqueeze(0), density.unsqueeze(0), budget, "sinusoidal_axis_mixture_wor",
            0.0, 0.0, args.seed + case_id + 211, sinusoidal_axis=1, sinusoidal_mix_fraction=1.0,
        )
        original, _, _ = sample_geometry_view(
            geo.unsqueeze(0), density.unsqueeze(0), budget, "uniform_wor",
            0.0, 0.0, args.seed + case_id + 17,
        )
        conditions = [
            ("original", original[0], "", 0),
            ("sine_x", sine_x[0], "", 0),
            ("sine_y", sine_y[0], "", 0),
        ]
        for method in methods:
            for factor in factors:
                path = remesh_path(args.dataset, args.remesh_root, case_id, method, factor)
                physical = remesh_geometry(path, budget, args.seed + case_id * 1000 + factor * 10 + len(method), args.dataset)
                conditions.append((f"remesh_{method}_div{factor}", torch.from_numpy((physical - lower) / span), method, factor))
        cases.append((case_id, surf_q, surf_y, vol_q, vol_y, params, conditions))

    print(f"dataset={args.dataset} model={args.model} cases={[case[0] for case in cases]} budget={budget}", flush=True)
    rows: list[dict[str, Any]] = []
    channels = (surf_channels, vol_channels, parameter_channels)
    for variant, cfg, checkpoint in (("base", base_cfg, args.base_checkpoint), ("deal", deal_cfg, args.deal_checkpoint)):
        model = build_model(args.model, cfg, checkpoint, device, channels)
        for case_id, surf_q, surf_y, vol_q, vol_y, params, conditions in tqdm(cases, desc=variant, dynamic_ncols=True):
            for condition, geometry, method, factor in conditions:
                metrics = predict(
                    model,
                    device,
                    geometry,
                    surf_q,
                    surf_y,
                    vol_q,
                    vol_y,
                    params,
                    dataset,
                    amp_enabled=args.model != "lno",
                )
                rows.append({"case_id": case_id, "variant": variant, "condition": condition, "method": method, "factor": factor, **metrics})
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    summary = summarize(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "per_case_rows.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump({
            "dataset": args.dataset,
            "model": args.model,
            "case_ids": [case[0] for case in cases],
            "encoder_budget": budget,
            "query_budgets": {"surface": int(dataset.surface_points), "volume": int(dataset.volume_points)},
            "checkpoints": {"base": checkpoint_metadata(args.base_checkpoint), "deal": checkpoint_metadata(args.deal_checkpoint)},
            "summary": summary,
        }, handle, indent=2)
    for item in summary:
        if item["metric"] == "combined_physical_rel_l2" and item["condition"] in {"sine_x", "sine_y", "remesh_mean_div5", "remesh_mean_div10"}:
            print(
                f"{item['condition']}: base={item['base_mean']:.6f}, DeAL={item['deal_mean']:.6f}, "
                f"improvement={item['deal_improvement_percent']:+.2f}%",
                flush=True,
            )


if __name__ == "__main__":
    main()
