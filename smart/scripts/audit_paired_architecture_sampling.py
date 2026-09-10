#!/usr/bin/env python3
"""Evaluate one base/DeAL architecture pair on fixed sampling shifts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import random
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

# This filename is historical: the archived DeAL weights were trained before
# the density-sensitive PTv3 runtime changes. Sparse ordering and voxel-density
# behavior are runtime settings, so a state dict can otherwise load silently
# under the wrong configuration.
PTV3_DRIVAERML_DEAL_CHECKPOINT_PREFIX = (
    "point-transformer-v3-satloss7-ptv3-satloss7-density-sensitive-drivaerml-131k-"
)
PTV3_DRIVAERML_DEAL_RUNTIME_CONFIG = "drivaerml_point_transformer_v3_satloss7"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("drivaerml", "heat_exchanger", "pump", "c_core"), required=True)
    parser.add_argument("--model", choices=tuple(CONSTRUCTORS), required=True)
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--deal-config", required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--deal-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--checkpoint-policy",
        choices=("last", "base_best_deal_last", "any"),
        default="last",
        help=(
            "Require final-epoch checkpoints by default. "
            "base_best_deal_last permits only a best base checkpoint paired with a final DeAL checkpoint."
        ),
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--remesh-root", type=Path, required=True)
    parser.add_argument(
        "--split",
        choices=("train", "validation"),
        default="validation",
        help="Dataset split to evaluate. The default preserves validation-only audits.",
    )
    parser.add_argument("--methods", default="feature,quadric,voxel")
    parser.add_argument("--factors", default="5,10")
    parser.add_argument(
        "--conditions",
        default="original,sine_x,sine_y,remesh",
        help=(
            "Comma-separated evaluation conditions. Use 'original' for aligned-input "
            "accuracy; 'remesh' expands to every requested method and factor."
        ),
    )
    parser.add_argument("--num-cases", type=int, default=5)
    parser.add_argument("--case-ids", default="")
    parser.add_argument(
        "--cohort-registry",
        type=Path,
        help="Canonical cohort registry; when set, case IDs are loaded by dataset.",
    )
    parser.add_argument("--views-per-condition", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--transolver-slice-assignment-mode",
        choices=("official_gumbel", "deterministic_softmax"),
        help="Override Transolver++ slice assignment to reproduce an archived evaluation runtime.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def stable_seed(*parts: object) -> int:
    payload = "|".join(map(str, parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**31)


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
    constructor = CONSTRUCTORS[model_name]
    if model_name == "transolverpp":
        valid = set(inspect.signature(TransolverPP.__mro__[1].__init__).parameters) - {"self"}
        architecture = {key: value for key, value in architecture.items() if key in valid}
    elif model_name == "mspt":
        valid = set(inspect.signature(MSPT.__init__).parameters) - {"self"}
        architecture = {key: value for key, value in architecture.items() if key in valid}
    model = constructor(
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


def normalized_prediction_difference(
    first: torch.Tensor,
    second: torch.Tensor,
    target: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> float:
    numerator = torch.linalg.vector_norm((first - second) * std)
    denominator = torch.linalg.vector_norm(target * std + mean).clamp_min(1.0e-12)
    return float((numerator / denominator).item())


def predict(
    model,
    device,
    geometry,
    surf_q,
    surf_y,
    vol_q,
    vol_y,
    params,
    dataset,
    *,
    amp_enabled: bool,
) -> tuple[dict[str, float], torch.Tensor, torch.Tensor]:
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
    metrics = {
        "surface_standardized_rel_l2": standard_s,
        "volume_standardized_rel_l2": standard_v,
        "combined_standardized_rel_l2": 0.5 * (standard_s + standard_v),
        "surface_physical_rel_l2": physical_s,
        "volume_physical_rel_l2": physical_v,
        "combined_physical_rel_l2": 0.5 * (physical_s + physical_v),
    }
    return metrics, pred_s, pred_v


def canonical_case_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse views and remesh methods before any across-case statistic."""
    metric_names = [
        key for key in rows[0]
        if key.endswith("_rel_l2") or key.endswith("_rel_gt")
    ]
    grouped: dict[tuple[int, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (int(row["case_id"]), str(row["variant"]), str(row["condition"]))
        grouped.setdefault(key, []).append(row)

    output: list[dict[str, Any]] = []
    for (case_id, variant, condition), selected in sorted(grouped.items()):
        record: dict[str, Any] = {
            "case_id": case_id,
            "variant": variant,
            "condition": condition,
            "source_rows": len(selected),
        }
        for metric in metric_names:
            values = np.asarray([row[metric] for row in selected], dtype=np.float64)
            finite = values[np.isfinite(values)]
            record[metric] = float(finite.mean()) if finite.size else float("nan")
        output.append(record)

    factors = sorted({int(row["factor"]) for row in rows if int(row["factor"]) > 0})
    for case_id in sorted({int(row["case_id"]) for row in rows}):
        for variant in ("base", "deal"):
            for factor in factors:
                selected = [
                    row for row in rows
                    if int(row["case_id"]) == case_id
                    and row["variant"] == variant
                    and int(row["factor"]) == factor
                ]
                if not selected:
                    continue
                record = {
                    "case_id": case_id,
                    "variant": variant,
                    "condition": f"remesh_mean_div{factor}",
                    "source_rows": len(selected),
                }
                for metric in metric_names:
                    values = np.asarray([row[metric] for row in selected], dtype=np.float64)
                    finite = values[np.isfinite(values)]
                    record[metric] = float(finite.mean()) if finite.size else float("nan")
                output.append(record)
    return output


def summarize(case_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metric_names = [
        key for key in case_rows[0]
        if key.endswith("_rel_l2") or key.endswith("_rel_gt")
    ]
    output = []
    for condition in sorted({str(row["condition"]) for row in case_rows}):
        for metric in metric_names:
            by_variant = {
                variant: {
                    int(row["case_id"]): float(row[metric])
                    for row in case_rows
                    if row["condition"] == condition
                    and row["variant"] == variant
                    and np.isfinite(float(row[metric]))
                }
                for variant in ("base", "deal")
            }
            paired_cases = sorted(set(by_variant["base"]) & set(by_variant["deal"]))
            if not paired_cases:
                # A single-view run has no finite repeated-view disagreement.
                continue
            base = np.asarray([by_variant["base"][case] for case in paired_cases])
            deal = np.asarray([by_variant["deal"][case] for case in paired_cases])
            base_mean, deal_mean = float(base.mean()), float(deal.mean())
            output.append({
                "condition": condition,
                "metric": metric,
                "samples": int(len(paired_cases)),
                "base_mean": base_mean,
                "base_std": float(base.std(ddof=1)) if len(base) > 1 else 0.0,
                "deal_mean": deal_mean,
                "deal_std": float(deal.std(ddof=1)) if len(deal) > 1 else 0.0,
                "deal_improvement_percent": 100.0 * (base_mean - deal_mean) / max(base_mean, 1.0e-12),
            })
    return output


def registry_case_ids(path: Path, dataset: str) -> tuple[list[int], dict[str, Any]]:
    registry = json.loads(path.read_text(encoding="utf-8"))
    task_keys = {
        "drivaerml": "drivaerml",
        "heat_exchanger": "heat_exchanger",
        "pump": "pump",
        "c_core": "c_core",
    }
    task = registry.get("tasks", {}).get(task_keys[dataset])
    if not task:
        raise KeyError(f"Registry {path} has no task entry for {dataset}")
    case_ids = [int(case_id) for case_id in task["selected_case_ids"]]
    if not case_ids:
        raise ValueError(f"Registry {path} has an empty cohort for {dataset}")
    return case_ids, registry


def validate_checkpoint_runtime_config(model: str, checkpoint: Path, config_name: str) -> None:
    """Reject known shape-compatible checkpoint/config mismatches."""
    if (
        model == "point_transformer_v3"
        and checkpoint.name.startswith(PTV3_DRIVAERML_DEAL_CHECKPOINT_PREFIX)
        and config_name != PTV3_DRIVAERML_DEAL_RUNTIME_CONFIG
    ):
        raise ValueError(
            f"{checkpoint.name} requires {PTV3_DRIVAERML_DEAL_RUNTIME_CONFIG!r}; "
            f"received {config_name!r}. The density-sensitive PTv3 config is "
            "shape-compatible but produces invalid inference for this checkpoint."
        )


def main() -> None:
    args = parse_args()
    if args.views_per_condition < 1:
        raise ValueError("--views-per-condition must be positive")
    if args.checkpoint_policy == "last":
        invalid = [
            str(path)
            for path in (args.base_checkpoint, args.deal_checkpoint)
            if not path.name.endswith("_last.pt")
        ]
        if invalid:
            raise ValueError(
                "Canonical evaluations require `_last.pt` checkpoints; received: "
                + ", ".join(invalid)
            )
    elif args.checkpoint_policy == "base_best_deal_last":
        if not args.base_checkpoint.name.endswith("_best.pt"):
            raise ValueError(
                "base_best_deal_last requires a `_best.pt` base checkpoint; received: "
                + str(args.base_checkpoint)
            )
        if not args.deal_checkpoint.name.endswith("_last.pt"):
            raise ValueError(
                "base_best_deal_last requires a `_last.pt` DeAL checkpoint; received: "
                + str(args.deal_checkpoint)
            )
    validate_checkpoint_runtime_config(args.model, args.base_checkpoint, args.base_config)
    validate_checkpoint_runtime_config(args.model, args.deal_checkpoint, args.deal_config)
    requested_conditions = [item.strip() for item in args.conditions.split(",") if item.strip()]
    allowed_conditions = {"original", "sine_x", "sine_y", "remesh"}
    unknown_conditions = sorted(set(requested_conditions) - allowed_conditions)
    if unknown_conditions:
        raise ValueError(f"Unknown conditions: {unknown_conditions}")
    if not requested_conditions:
        raise ValueError("At least one evaluation condition is required")
    if any(condition != "original" for condition in requested_conditions) and "original" not in requested_conditions:
        raise ValueError("The original condition is required as the representation-drift reference")
    needs_remesh = "remesh" in requested_conditions
    methods = [item.strip() for item in args.methods.split(",") if item.strip()]
    factors = [int(item) for item in args.factors.split(",") if item.strip()]
    device = torch.device(args.device)
    base_cfg, deal_cfg = compose_config(args.base_config), compose_config(args.deal_config)
    if args.transolver_slice_assignment_mode:
        if args.model != "transolverpp":
            raise ValueError("--transolver-slice-assignment-mode applies only to Transolver++")
        for cfg in (base_cfg, deal_cfg):
            OmegaConf.update(
                cfg,
                "experiment.architecture.slice_assignment_mode",
                args.transolver_slice_assignment_mode,
                force_add=True,
            )
    budget = input_budget(deal_cfg)
    dataset_cfg = OmegaConf.create(OmegaConf.to_container(deal_cfg, resolve=True))
    dataset_cfg.experiment.data_path = str(args.data_root.resolve())
    dataset_cfg.experiment.num_body_points = 0
    dataset_cfg.experiment.model_name = "PAIRED_DEAL_SAMPLING_AUDIT"
    train_dataset, validation_dataset, _, spatial_dim, surf_channels, vol_channels, parameter_channels, _ = get_dataset(dataset_cfg.experiment)
    dataset = train_dataset if args.split == "train" else validation_dataset
    if spatial_dim != 3:
        raise ValueError("Only 3D datasets are supported")
    dataset.set_epoch(0)
    if hasattr(dataset, "geometry_epoch_seeded_sampling"):
        dataset.geometry_epoch_seeded_sampling = True
    # Pump query sampling uses an independent Generator. This optional
    # evaluation-only seed leaves training behavior unchanged.
    dataset.deterministic_evaluation_seed = args.seed
    registry = None
    requested = [int(item) for item in args.case_ids.split(",") if item.strip()]
    if args.cohort_registry:
        registry_ids, registry = registry_case_ids(args.cohort_registry, args.dataset)
        if requested and requested != registry_ids:
            raise ValueError("--case-ids does not exactly match the canonical cohort registry")
        requested = registry_ids
    if requested:
        indices = [dataset.data.index(case_id) for case_id in requested]
    else:
        indices = [
            index for index, case_id in enumerate(dataset.data)
            if not needs_remesh
            or complete_remeshes(args.dataset, args.remesh_root, int(case_id), methods, factors)
        ][: args.num_cases]
    if not requested and len(indices) != args.num_cases:
        raise RuntimeError(f"Found only {len(indices)} complete cases, requested {args.num_cases}")
    missing_remeshes = [] if not needs_remesh else [
        int(dataset.data[index]) for index in indices
        if not complete_remeshes(args.dataset, args.remesh_root, int(dataset.data[index]), methods, factors)
    ]
    if missing_remeshes:
        raise FileNotFoundError(f"Canonical cases have incomplete remeshes: {missing_remeshes}")

    lower, span = bounds(dataset)
    cases = []
    for index in indices:
        case_id = int(dataset.data[index])
        # Dataset query subsampling uses process-global RNGs. Seed by immutable
        # case identity so evaluating a subset or changing traversal order does
        # not alter the physical queries used for that case.
        case_seed = args.seed + case_id * 1_000_003
        random.seed(case_seed)
        np.random.seed(case_seed % (2**32))
        torch.manual_seed(case_seed)
        geo, surf_q, surf_y, vol_q, vol_y, params, density = unpack_item(dataset[index], parameter_channels)
        conditions = []
        for view in range(args.views_per_condition):
            view_seed = args.seed + case_id * 10_007 + view * 1_009
            if "original" in requested_conditions:
                original, _, _ = sample_geometry_view(
                    geo.unsqueeze(0), density.unsqueeze(0), budget, "uniform_wor",
                    0.0, 0.0, view_seed + 17,
                )
                conditions.append(("original", original[0], "", 0, view))
            if "sine_x" in requested_conditions:
                sine_x, _, _ = sample_geometry_view(
                    geo.unsqueeze(0), density.unsqueeze(0), budget, "sinusoidal_axis_mixture_wor",
                    0.0, 0.0, view_seed + 101, sinusoidal_axis=0, sinusoidal_mix_fraction=1.0,
                )
                conditions.append(("sine_x", sine_x[0], "", 0, view))
            if "sine_y" in requested_conditions:
                sine_y, _, _ = sample_geometry_view(
                    geo.unsqueeze(0), density.unsqueeze(0), budget, "sinusoidal_axis_mixture_wor",
                    0.0, 0.0, view_seed + 211, sinusoidal_axis=1, sinusoidal_mix_fraction=1.0,
                )
                conditions.append(("sine_y", sine_y[0], "", 0, view))
            if needs_remesh:
                for method in methods:
                    for factor in factors:
                        path = remesh_path(args.dataset, args.remesh_root, case_id, method, factor)
                        physical = remesh_geometry(
                            path,
                            budget,
                            view_seed + factor * 10 + len(method),
                            args.dataset,
                        )
                        conditions.append(
                            (
                                f"remesh_{method}_div{factor}",
                                torch.from_numpy((physical - lower) / span),
                                method,
                                factor,
                                view,
                            )
                        )
        cases.append((case_id, surf_q, surf_y, vol_q, vol_y, params, conditions))

    print(f"dataset={args.dataset} model={args.model} cases={[case[0] for case in cases]} budget={budget}", flush=True)
    rows: list[dict[str, Any]] = []
    channels = (surf_channels, vol_channels, parameter_channels)
    for variant, cfg, checkpoint in (("base", base_cfg, args.base_checkpoint), ("deal", deal_cfg, args.deal_checkpoint)):
        model = build_model(args.model, cfg, checkpoint, device, channels)
        for case_id, surf_q, surf_y, vol_q, vol_y, params, conditions in tqdm(cases, desc=variant, dynamic_ncols=True):
            original_predictions: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
            condition_references: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
            surf_mean, surf_std = dataset.mean_surf_data.float(), dataset.std_surf_data.float()
            vol_mean, vol_std = dataset.mean_vol_data.float(), dataset.std_vol_data.float()
            for condition, geometry, method, factor, view in conditions:
                inference_seed = stable_seed(args.seed, case_id, condition, view)
                random.seed(inference_seed)
                np.random.seed(inference_seed % (2**32))
                torch.manual_seed(inference_seed)
                metrics, pred_s, pred_v = predict(
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
                if condition == "original":
                    original_predictions[view] = (pred_s, pred_v)
                    representation_s = representation_v = 0.0
                else:
                    original_s, original_v = original_predictions[view]
                    representation_s = normalized_prediction_difference(
                        pred_s, original_s, surf_y, surf_mean, surf_std
                    )
                    representation_v = normalized_prediction_difference(
                        pred_v, original_v, vol_y, vol_mean, vol_std
                    )

                if condition in condition_references:
                    reference_s, reference_v = condition_references[condition]
                    view_s = normalized_prediction_difference(
                        pred_s, reference_s, surf_y, surf_mean, surf_std
                    )
                    view_v = normalized_prediction_difference(
                        pred_v, reference_v, vol_y, vol_mean, vol_std
                    )
                else:
                    condition_references[condition] = (pred_s, pred_v)
                    view_s = view_v = float("nan")

                rows.append({
                    "case_id": case_id,
                    "view": view,
                    "variant": variant,
                    "condition": condition,
                    "method": method,
                    "factor": factor,
                    "surface_representation_drift_rel_gt": representation_s,
                    "volume_representation_drift_rel_gt": representation_v,
                    "combined_representation_drift_rel_gt": 0.5 * (representation_s + representation_v),
                    "surface_view_disagreement_rel_gt": view_s,
                    "volume_view_disagreement_rel_gt": view_v,
                    "combined_view_disagreement_rel_gt": 0.5 * (view_s + view_v),
                    **metrics,
                })
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    case_rows = canonical_case_rows(rows)
    summary = summarize(case_rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "per_case_rows.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    with (args.output_dir / "case_condition_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(case_rows[0]))
        writer.writeheader(); writer.writerows(case_rows)
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump({
            "dataset": args.dataset,
            "split": args.split,
            "model": args.model,
            "configs": {"base": args.base_config, "deal": args.deal_config},
            "case_ids": [case[0] for case in cases],
            "encoder_budget": budget,
            "query_budgets": {"surface": int(dataset.surface_points), "volume": int(dataset.volume_points)},
            "views_per_condition": args.views_per_condition,
            "requested_conditions": requested_conditions,
            "transolver_slice_assignment_mode_override": args.transolver_slice_assignment_mode,
        "seed": args.seed,
        "case_seed_rule": "seed + case_id * 1000003",
            "remesh_methods": methods,
            "remesh_factors": factors,
            "data_root": str(args.data_root.resolve()),
            "remesh_root": str(args.remesh_root.resolve()),
            "aggregation_order": [
                "mean repeated views within case and condition",
                "mean remeshing methods within case and reduction factor",
                "mean and sample standard deviation across canonical cases",
            ],
            "cohort_registry": None if args.cohort_registry is None else {
                "path": str(args.cohort_registry.resolve()),
                "registry_id": registry.get("registry_id"),
                "sha256": hashlib.sha256(args.cohort_registry.read_bytes()).hexdigest(),
            },
            "checkpoints": {"base": checkpoint_metadata(args.base_checkpoint), "deal": checkpoint_metadata(args.deal_checkpoint)},
            "checkpoint_policy": args.checkpoint_policy,
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
