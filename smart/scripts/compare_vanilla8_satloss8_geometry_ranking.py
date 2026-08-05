#!/usr/bin/env python3
"""Rank held-out geometries by SATLOSS8 improvement over vanilla8.

For every available vanilla8/SATLOSS8 model pair, this script evaluates the
same held-out geometry IDs and the same 65536 surface/volume query budgets.
Each pair keeps its own training-configured encoder budget.  The geometry score
used for ranking is:

    mean_model(vanilla8 combined global Rel-L2)
    - mean_model(SATLOSS8 combined global Rel-L2)

Positive scores mean SATLOSS8 is better than vanilla8.  Model pairs are kept
resident on four independent CUDA devices; this is faster and safer than
DataParallel for inference because each model has batch size one and the model
families have different memory footprints.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import threading
from collections import OrderedDict, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

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
from scripts.compare_drivaerml_sampling_invariance import (
    build_model,
    choose_fixed_query_indices,
    compute_metrics,
    denorm_fields,
    load_cfg,
    normalize_pos,
    parse_active_shifts,
    parse_shift_betas,
    predict_view_batch,
    resolve_density_spec,
    resolve_eval_sampling_mode,
    sample_inverse_density_with_replacement,
    sample_inverse_density_without_replacement,
    sample_uniform_with_replacement,
    sample_uniform_weighted_mixture_with_replacement,
    sample_uniform_weighted_mixture_without_replacement,
    sample_uniform_without_replacement,
    sampling_mode_uses_replacement,
    sinusoidal_axis_probabilities,
    sine_mix_levels_from_shift_betas,
    train_encoder_input_points,
    train_geometry_uses_replacement,
)
from scripts.infer_smart_vanilla8_satloss8_domain import MODEL_DEFAULTS


MODEL_ORDER = (
    "smart",
    "pointnet2_ssg",
    "point_gnn",
    "lno",
    "lno2",
    "transolverpp",
    "mspt",
    "point_transformer_v3",
)
METRIC_KEYS = (
    "combined_global_rel_l2",
    "combined_physics_rel_l2",
    "surface_pressure_rel_l2",
    "volume_pressure_rel_l2",
    "surface_wss_mag_rel_l2",
    "volume_velocity_mag_rel_l2",
)
DATASET_LOAD_LOCK = threading.Lock()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split-json",
        type=Path,
        default=REPO_ROOT / "results/drivaerml_geometry_statistical_split/geometry_domain_split.json",
    )
    parser.add_argument("--num-runs", type=int, default=150)
    parser.add_argument("--run-ids", default=None, help="Optional explicit held-out cluster-1 IDs.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--surface-query-points", type=int, default=65536)
    parser.add_argument("--volume-query-points", type=int, default=65536)
    parser.add_argument("--query-chunk", type=int, default=65536)
    parser.add_argument(
        "--query-sampling-with-replacement",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use the comparison script's default AhmedMLDatasetV2 query sampling semantics.",
    )
    parser.add_argument("--shift-betas", default="0,1", help="Endpoint beta values used by the comparison protocol.")
    parser.add_argument(
        "--active-shifts",
        default="all",
        help="Comparison shifts: beta,sine_y,sine_x, or all. Only configured endpoints are evaluated.",
    )
    parser.add_argument("--views-per-mode", type=int, default=2)
    parser.add_argument("--model-repeats", type=int, default=1)
    parser.add_argument("--devices", default="cuda:0,cuda:1,cuda:2,cuda:3")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results/vanilla8_satloss8_geometry_comparison_cluster1_150runs",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def resolve_devices(text: str) -> List[torch.device]:
    devices = []
    for token in str(text).split(","):
        token = token.strip()
        if not token:
            continue
        if token.isdigit():
            token = f"cuda:{token}"
        devices.append(torch.device(token))
    if not devices:
        raise ValueError("At least one device is required.")
    if any(device.type == "cuda" for device in devices) and not torch.cuda.is_available():
        raise RuntimeError("CUDA devices were requested but CUDA is unavailable.")
    for device in devices:
        if device.type == "cuda" and device.index is not None and device.index >= torch.cuda.device_count():
            raise ValueError(f"Requested {device}, but only {torch.cuda.device_count()} CUDA devices are visible.")
    return devices


def select_run_ids(split_path: Path, args: argparse.Namespace) -> List[int]:
    with split_path.open("r", encoding="utf-8") as handle:
        split = json.load(handle)
    test_ids = [int(value) for value in split["train_cluster_0_test_cluster_1"]["test_ids"]]
    if args.run_ids:
        requested = [int(value.strip()) for value in str(args.run_ids).split(",") if value.strip()]
        missing = sorted(set(requested).difference(test_ids))
        if missing:
            raise ValueError(f"Requested IDs are not in held-out cluster 1: {missing}")
        return requested
    if int(args.num_runs) <= 0 or int(args.num_runs) >= len(test_ids):
        return sorted(test_ids)
    rng = np.random.default_rng(int(args.seed) + 7001)
    return sorted(int(value) for value in rng.choice(np.asarray(test_ids), size=int(args.num_runs), replace=False))


def checkpoint_epoch(path: Path) -> int:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    return int(checkpoint.get("epoch", -1))


def make_dataset(
    config,
    split_path: Path,
    query_s: int,
    query_v: int,
    geometry_points: int,
    density_estimator: str,
    density_knn_k: int,
    density_neighbor_hops: int,
    density_cache_dtype: str,
) -> AhmedMLDatasetV2:
    return AhmedMLDatasetV2(
        saved_folder=str(config.data_path),
        if_test=True,
        geometry_points=int(geometry_points),
        surface_points=int(query_s),
        volume_points=int(query_v),
        scale_positions=bool(config.scale_positions),
        require_preprocessed=True,
        domain_split_json=str(split_path),
        domain_split_train_cluster=0,
        domain_split_test_cluster=1,
        geometry_density_estimator=str(density_estimator),
        geometry_density_knn_k=int(density_knn_k),
        geometry_density_neighbor_hops=int(density_neighbor_hops),
        geometry_density_cache_dtype=str(density_cache_dtype),
    )


def build_mode_defs(args: argparse.Namespace) -> OrderedDict:
    """Build the same endpoint modes as compare_drivaerml_sampling_invariance."""
    shift_betas = parse_shift_betas(args.shift_betas)
    sine_mix_levels = sine_mix_levels_from_shift_betas(shift_betas)
    active_shifts = set(parse_active_shifts(args.active_shifts))
    mode_defs = OrderedDict()
    mode_defs["aligned_uniform_wor"] = {
        "kind": "uniform_wor",
        "beta": 0.0,
        "id": 0,
        "description": "Uniform comparison control.",
    }
    if "beta" in active_shifts:
        for mode_id, beta in enumerate(shift_betas, start=1):
            mode_defs[f"shifted_inverse_density_beta_{beta:.2f}"] = {
                "kind": "inverse_density_wor",
                "beta": float(beta),
                "id": mode_id,
                "description": f"Inverse-density endpoint beta={beta:.2f}.",
            }
    next_mode_id = len(mode_defs)
    for shift_idx, (shift_name, axis) in enumerate((("sine_y", 1), ("sine_x", 0))):
        if shift_name not in active_shifts:
            continue
        for mix_idx, mix_fraction in enumerate(sine_mix_levels):
            mode_defs[f"ood_{shift_name}_mix_{mix_fraction:.2f}"] = {
                "kind": "sinusoidal_axis_mixture_wor",
                "beta": math.nan,
                "mix_fraction": float(mix_fraction),
                "distribution_key": shift_name,
                "axis": axis,
                "id": next_mode_id + shift_idx * len(sine_mix_levels) + mix_idx,
                "description": f"{shift_name} endpoint intensity={mix_fraction:.2f}.",
            }
    return mode_defs


def load_raw_run(config, run_id: int) -> Dict[str, np.ndarray]:
    """Load the same raw preprocessed arrays used by the main comparison."""
    run_dir = Path(config.data_path) / f"run_{int(run_id)}"
    surf_coords = np.load(run_dir / "surface_coords.npy").astype(np.float32, copy=False)
    surf_p = np.load(run_dir / "surface_pMeanTrim.npy").astype(np.float32, copy=False).reshape(-1, 1)
    surf_n = np.load(run_dir / "surface_normals.npy").astype(np.float32, copy=False)
    surf_wx = np.load(run_dir / "surface_wallShearStressMeanTrim_x.npy").astype(np.float32, copy=False).reshape(-1, 1)
    surf_wy = np.load(run_dir / "surface_wallShearStressMeanTrim_y.npy").astype(np.float32, copy=False).reshape(-1, 1)
    surf_wz = np.load(run_dir / "surface_wallShearStressMeanTrim_z.npy").astype(np.float32, copy=False).reshape(-1, 1)
    vol_coords = np.load(run_dir / "volume_coords.npy").astype(np.float32, copy=False)
    vol_p = np.load(run_dir / "volume_pMeanTrim.npy").astype(np.float32, copy=False).reshape(-1, 1)
    vol_u = np.load(run_dir / "volume_UMeanTrim.npy").astype(np.float32, copy=False)
    return {
        "surf_coords": surf_coords,
        "surf_gt": np.concatenate([surf_p, surf_n, surf_wx, surf_wy, surf_wz], axis=1),
        "vol_coords": vol_coords,
        "vol_gt": np.concatenate([vol_p, vol_u], axis=1),
    }


def sample_comparison_indices(
    coords: np.ndarray,
    config,
    mode_info: Mapping[str, object],
    density: np.ndarray,
    sine_weights: Mapping[str, np.ndarray],
    seed_components: Sequence[int],
) -> np.ndarray:
    budget = train_encoder_input_points(config)
    replacement_default = train_geometry_uses_replacement(config, budget, int(coords.shape[0]))
    configured_mode = resolve_eval_sampling_mode(config, str(mode_info["kind"]))
    replacement = sampling_mode_uses_replacement(configured_mode, replacement_default)
    rng = np.random.default_rng(np.random.SeedSequence([int(x) for x in seed_components]))
    kind = str(mode_info["kind"])
    if kind == "uniform_wor":
        return (
            sample_uniform_with_replacement(coords.shape[0], budget, rng)
            if replacement
            else sample_uniform_without_replacement(coords.shape[0], budget, rng)
        )
    if kind == "inverse_density_wor":
        beta = float(mode_info["beta"])
        return (
            sample_inverse_density_with_replacement(density, budget, beta, rng)
            if replacement
            else sample_inverse_density_without_replacement(density, budget, beta, rng)
        )
    if kind == "sinusoidal_axis_mixture_wor":
        weights = sine_weights[str(mode_info["distribution_key"])]
        fraction = float(mode_info["mix_fraction"])
        return (
            sample_uniform_weighted_mixture_with_replacement(weights, budget, fraction, rng)
            if replacement
            else sample_uniform_weighted_mixture_without_replacement(weights, budget, fraction, rng)
        )
    raise ValueError(f"Unsupported comparison sampling mode: {kind}")


def build_pair_models(model_key: str, configs, checkpoint_paths, device: torch.device, query_chunk: int):
    vanilla_cfg = configs[model_key]["vanilla"]
    satloss8_cfg = configs[model_key]["satloss8"]
    sat_build_cfg = OmegaConf.create(OmegaConf.to_container(satloss8_cfg, resolve=True))
    sat_build_cfg.model_name = vanilla_cfg.model_name
    vanilla_model = build_model(vanilla_cfg, str(checkpoint_paths[model_key]["vanilla"]), device, query_chunk).to(device)
    satloss8_model = build_model(sat_build_cfg, str(checkpoint_paths[model_key]["satloss8"]), device, query_chunk).to(device)
    vanilla_model.eval()
    satloss8_model.eval()
    return vanilla_model, satloss8_model


def evaluate_device_group(
    device: torch.device,
    model_keys: Sequence[str],
    run_ids: Sequence[int],
    args: argparse.Namespace,
    split_path: Path,
    configs,
    checkpoint_paths,
    mode_defs: Mapping[str, Mapping[str, object]],
    mean_s: torch.Tensor,
    std_s: torch.Tensor,
    mean_v: torch.Tensor,
    std_v: torch.Tensor,
    min_pos: torch.Tensor,
    max_pos: torch.Tensor,
    geometry_points: int,
    density_spec: Sequence[object],
) -> List[Dict[str, object]]:
    if device.type == "cuda":
        torch.cuda.set_device(device)
    models = {
        model_key: build_pair_models(model_key, configs, checkpoint_paths, device, int(args.query_chunk))
        for model_key in model_keys
    }
    density_estimator, density_knn_k, density_neighbor_hops, density_cache_dtype = density_spec
    dataset = make_dataset(
        configs[model_keys[0]]["vanilla"],
        split_path,
        args.surface_query_points,
        args.volume_query_points,
        geometry_points,
        str(density_estimator),
        int(density_knn_k),
        int(density_neighbor_hops),
        str(density_cache_dtype),
    )
    rows: List[Dict[str, object]] = []
    for run_id in tqdm(run_ids, desc=f"{device} {','.join(model_keys)}", leave=False):
        # Raw arrays and fixed query indices mirror the main comparison script;
        # this deliberately does not use AhmedMLDatasetV2's ordinary sampled item.
        with DATASET_LOAD_LOCK:
            raw = load_raw_run(configs[model_keys[0]]["vanilla"], int(run_id))
            full_density = dataset._load_or_compute_full_geometry_density(
                int(run_id), expected_n=int(raw["surf_coords"].shape[0])
            ).to(dtype=torch.float32).numpy()
        surf_coords = raw["surf_coords"]
        vol_coords = raw["vol_coords"]
        max_surf_query = int(args.surface_query_points)
        max_vol_query = int(args.volume_query_points)
        surf_query_idx = choose_fixed_query_indices(
            surf_coords.shape[0], max_surf_query, [args.seed, int(run_id), 3001], args.query_sampling_with_replacement
        )
        vol_query_idx = choose_fixed_query_indices(
            vol_coords.shape[0], max_vol_query, [args.seed, int(run_id), 3002], args.query_sampling_with_replacement
        )
        surf_query_norm = normalize_pos(torch.from_numpy(surf_coords[surf_query_idx]), min_pos, max_pos)
        vol_query_norm = normalize_pos(torch.from_numpy(vol_coords[vol_query_idx]), min_pos, max_pos)
        gt_s = raw["surf_gt"][surf_query_idx]
        gt_v = raw["vol_gt"][vol_query_idx]
        sine_weights = {
            "sine_y": sinusoidal_axis_probabilities(surf_coords, axis=1),
            "sine_x": sinusoidal_axis_probabilities(surf_coords, axis=0),
        }

        for model_key in model_keys:
            for variant_index, (variant_name, variant_key) in enumerate((("vanilla8", "vanilla"), ("satloss8", "satloss8"))):
                variant_cfg = configs[model_key][variant_key]
                for mode_name, mode_info in mode_defs.items():
                    geo_views = []
                    for view_idx in range(max(1, int(args.views_per_mode))):
                        geo_idx = sample_comparison_indices(
                            surf_coords,
                            variant_cfg,
                            mode_info,
                            full_density,
                            sine_weights,
                            [args.seed, int(run_id), int(mode_info["id"]), view_idx],
                        )
                        geo_views.append(normalize_pos(torch.from_numpy(surf_coords[geo_idx]), min_pos, max_pos))
                    geo_views_norm = torch.stack(geo_views, dim=0)
                    pred_s, pred_v = predict_view_batch(
                        str(variant_cfg.model_name),
                        models[model_key][variant_index],
                        geo_views_norm,
                        surf_query_norm,
                        vol_query_norm,
                        None,
                        mean_s,
                        std_s,
                        mean_v,
                        std_v,
                        device,
                        base_seed=int(args.seed) + 100000 * int(mode_info["id"]) + 1000 * int(run_id),
                        repeats=int(args.model_repeats),
                    )
                    for view_idx in range(pred_s.shape[0]):
                        metrics = compute_metrics(gt_s, pred_s[view_idx], gt_v, pred_v[view_idx])
                        rows.append(
                            {
                                "run_id": int(run_id),
                                "model": model_key,
                                "variant": variant_name,
                                "sampling_mode": mode_name,
                                "sampling_kind": str(mode_info["kind"]),
                                "sampling_mode_id": int(mode_info["id"]),
                                "view_id": int(view_idx),
                                "input_points": int(geo_views_norm.shape[1]),
                                **{key: float(metrics[key]) for key in METRIC_KEYS},
                            }
                        )
    return rows


def write_csv(path: Path, rows: Sequence[Mapping[str, object]], fields: Sequence[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def rank_geometries(rows: Sequence[Mapping[str, object]], model_keys: Sequence[str]) -> List[Dict[str, object]]:
    """Rank run/mode pairs after averaging views for every model variant."""
    by_run_mode: Dict[tuple[int, str], Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        key = (int(row["run_id"]), str(row["sampling_mode"]))
        by_run_mode[key][f"{row['model']}_{row['variant']}"].append(
            float(row["combined_global_rel_l2"])
        )
    scores = []
    for (run_id, mode_name), run_rows in by_run_mode.items():
        vanilla_values = [float(np.mean(run_rows[f"{model}_vanilla8"])) for model in model_keys]
        satloss8_values = [float(np.mean(run_rows[f"{model}_satloss8"])) for model in model_keys]
        score = float(np.mean(vanilla_values) - np.mean(satloss8_values))
        item: Dict[str, object] = {
            "run_id": int(run_id),
            "sampling_mode": mode_name,
            "avg_vanilla8_combined_global_rel_l2": float(np.mean(vanilla_values)),
            "avg_satloss8_combined_global_rel_l2": float(np.mean(satloss8_values)),
            "avg_improvement_abs": score,
            "avg_improvement_percent": 100.0 * score / max(float(np.mean(vanilla_values)), 1.0e-12),
            "models_improved": int(sum(s < v for v, s in zip(vanilla_values, satloss8_values))),
        }
        for model, vanilla_value, satloss8_value in zip(model_keys, vanilla_values, satloss8_values):
            item[f"{model}_vanilla8"] = vanilla_value
            item[f"{model}_satloss8"] = satloss8_value
            item[f"{model}_improvement_abs"] = vanilla_value - satloss8_value
        scores.append(item)
    return sorted(scores, key=lambda item: (-float(item["avg_improvement_abs"]), str(item["sampling_mode"]), int(item["run_id"])))


def rank_overall_geometries(mode_scores: Sequence[Mapping[str, object]], model_keys: Sequence[str]) -> List[Dict[str, object]]:
    """Average the comparison-script mode scores so each geometry gets one rank."""
    by_run: Dict[int, List[Mapping[str, object]]] = defaultdict(list)
    for item in mode_scores:
        by_run[int(item["run_id"])].append(item)
    scores = []
    for run_id, items in by_run.items():
        vanilla_values = []
        satloss8_values = []
        for model in model_keys:
            vanilla_values.append(float(np.mean([float(item[f"{model}_vanilla8"]) for item in items])))
            satloss8_values.append(float(np.mean([float(item[f"{model}_satloss8"]) for item in items])))
        score = float(np.mean(vanilla_values) - np.mean(satloss8_values))
        item: Dict[str, object] = {
            "run_id": int(run_id),
            "modes_averaged": int(len(items)),
            "avg_vanilla8_combined_global_rel_l2": float(np.mean(vanilla_values)),
            "avg_satloss8_combined_global_rel_l2": float(np.mean(satloss8_values)),
            "avg_improvement_abs": score,
            "avg_improvement_percent": 100.0 * score / max(float(np.mean(vanilla_values)), 1.0e-12),
            "models_improved": int(sum(s < v for v, s in zip(vanilla_values, satloss8_values))),
        }
        for model, vanilla_value, satloss8_value in zip(model_keys, vanilla_values, satloss8_values):
            item[f"{model}_vanilla8"] = vanilla_value
            item[f"{model}_satloss8"] = satloss8_value
            item[f"{model}_improvement_abs"] = vanilla_value - satloss8_value
        scores.append(item)
    return sorted(scores, key=lambda item: (-float(item["avg_improvement_abs"]), int(item["run_id"])))


def main() -> None:
    args = parse_args()
    split_path = args.split_json.expanduser().resolve()
    run_ids = select_run_ids(split_path, args)
    devices = resolve_devices(args.devices)
    active_models = []
    configs = {}
    checkpoint_paths = {}
    for model_key in MODEL_ORDER:
        defaults = MODEL_DEFAULTS[model_key]
        vanilla_path = Path(defaults["vanilla_checkpoint"]).expanduser().resolve()
        satloss8_path = Path(defaults["satloss8_checkpoint"]).expanduser().resolve()
        if not vanilla_path.is_file() or not satloss8_path.is_file():
            print(f"[skip] {model_key}: missing vanilla8 or SATLOSS8 checkpoint")
            continue
        configs[model_key] = {
            "vanilla": load_cfg(defaults["vanilla_config"]),
            "satloss8": load_cfg(defaults["satloss8_config"]),
        }
        checkpoint_paths[model_key] = {"vanilla": vanilla_path, "satloss8": satloss8_path}
        active_models.append(model_key)
    if not active_models:
        raise RuntimeError("No complete vanilla8/SATLOSS8 checkpoint pairs were found.")
    mode_defs = build_mode_defs(args)
    if not mode_defs:
        raise RuntimeError("No comparison sampling modes were enabled.")
    query_s = int(args.surface_query_points)
    query_v = int(args.volume_query_points)
    if any(int(configs[key]["vanilla"].num_surface_points) != query_s for key in active_models):
        raise ValueError("The requested surface query budget does not match every vanilla8 config.")
    if any(int(configs[key]["vanilla"].num_volume_points) != query_v for key in active_models):
        raise ValueError("The requested volume query budget does not match every vanilla8 config.")
    if any(int(configs[key]["satloss8"].num_surface_points) != query_s for key in active_models):
        raise ValueError("The requested surface query budget does not match every SATLOSS8 config.")
    if any(int(configs[key]["satloss8"].num_volume_points) != query_v for key in active_models):
        raise ValueError("The requested volume query budget does not match every SATLOSS8 config.")
    if args.dry_run:
        print(f"Held-out geometries: {len(run_ids)}")
        print(f"Active model pairs: {', '.join(active_models)}")
        print(f"Devices: {', '.join(map(str, devices))}")
        for key in active_models:
            print(
                f"{key}: vanilla8 epoch={checkpoint_epoch(checkpoint_paths[key]['vanilla'])}, "
                f"SATLOSS8 epoch={checkpoint_epoch(checkpoint_paths[key]['satloss8'])}, "
                f"encoder={train_encoder_input_points(configs[key]['vanilla'])}/"
                f"{train_encoder_input_points(configs[key]['satloss8'])}"
            )
        print(f"Comparison modes: {', '.join(mode_defs)}")
        return

    density_spec = resolve_density_spec(configs[active_models[0]]["vanilla"])
    max_encoder_points = max(
        train_encoder_input_points(configs[key][variant])
        for key in active_models
        for variant in ("vanilla", "satloss8")
    )
    reference_dataset = make_dataset(
        configs[active_models[0]]["vanilla"],
        split_path,
        query_s,
        query_v,
        max_encoder_points,
        *density_spec,
    )
    mean_s = reference_dataset.mean_surf_data
    std_s = torch.clamp(reference_dataset.std_surf_data, min=1.0e-12)
    mean_v = reference_dataset.mean_vol_data
    std_v = torch.clamp(reference_dataset.std_vol_data, min=1.0e-12)
    min_pos = reference_dataset.min_pos
    max_pos = reference_dataset.max_pos
    groups = [[] for _ in devices]
    # Put the largest model first so the persistent model copies are balanced
    # across visible GPUs rather than filling one GPU serially.
    ordered_models = sorted(active_models, key=lambda key: key == "point_transformer_v3", reverse=True)
    for index, model_key in enumerate(ordered_models):
        groups[index % len(groups)].append(model_key)
    print(f"Evaluating {len(run_ids)} held-out geometries with {len(active_models)} model pairs on {len(devices)} GPUs.")
    print("Assignments: " + "; ".join(f"{device}: {','.join(group)}" for device, group in zip(devices, groups) if group))

    all_rows: List[Dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=len(devices)) as pool:
        futures = [
            pool.submit(
                evaluate_device_group,
                device,
                group,
                run_ids,
                args,
                split_path,
                configs,
                checkpoint_paths,
                mode_defs,
                mean_s,
                std_s,
                mean_v,
                std_v,
                min_pos,
                max_pos,
                max_encoder_points,
                density_spec,
            )
            for device, group in zip(devices, groups)
            if group
        ]
        for future in futures:
            all_rows.extend(future.result())

    all_rows.sort(
        key=lambda row: (
            int(row["run_id"]),
            int(row["sampling_mode_id"]),
            str(row["model"]),
            str(row["variant"]),
            int(row["view_id"]),
        )
    )
    mode_scores = rank_geometries(all_rows, active_models)
    overall_scores = rank_overall_geometries(mode_scores, active_models)
    top25 = overall_scores[:25]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        args.output_dir / "all_model_geometry_metrics.csv",
        all_rows,
        [
            "run_id",
            "model",
            "variant",
            "sampling_mode",
            "sampling_kind",
            "sampling_mode_id",
            "view_id",
            "input_points",
            *METRIC_KEYS,
        ],
    )
    mode_score_fields = list(mode_scores[0].keys()) if mode_scores else ["run_id"]
    overall_score_fields = list(top25[0].keys()) if top25 else ["run_id"]
    write_csv(args.output_dir / "all_geometry_scores_by_sampling_mode.csv", mode_scores, mode_score_fields)
    write_csv(args.output_dir / "all_geometry_scores.csv", overall_scores, overall_score_fields)
    write_csv(args.output_dir / "top25_satloss8_improvement_geometries.csv", top25, overall_score_fields)
    for mode_name in mode_defs:
        mode_top25 = [item for item in mode_scores if str(item["sampling_mode"]) == mode_name][:25]
        write_csv(
            args.output_dir / f"top25_{mode_name}_satloss8_improvement_geometries.csv",
            mode_top25,
            mode_score_fields,
        )
    payload = {
        "protocol": (
            "Comparison-script sampling protocol on train-cluster-0/test-cluster-1: "
            "rank geometries by average vanilla8 minus SATLOSS8 combined global Rel-L2."
        ),
        "run_ids": run_ids,
        "train_cluster": 0,
        "test_cluster": 1,
        "sampling_modes": {
            name: dict(info) for name, info in mode_defs.items()
        },
        "query_surface_points": query_s,
        "query_volume_points": query_v,
        "query_sampling_with_replacement": bool(args.query_sampling_with_replacement),
        "views_per_mode": int(args.views_per_mode),
        "model_repeats": int(args.model_repeats),
        "query_chunk": int(args.query_chunk),
        "seed": int(args.seed),
        "devices": [str(device) for device in devices],
        "models": {
            key: {
                "vanilla8_config": MODEL_DEFAULTS[key]["vanilla_config"],
                "satloss8_config": MODEL_DEFAULTS[key]["satloss8_config"],
                "vanilla8_checkpoint": str(checkpoint_paths[key]["vanilla"]),
                "satloss8_checkpoint": str(checkpoint_paths[key]["satloss8"]),
                "vanilla8_epoch": checkpoint_epoch(checkpoint_paths[key]["vanilla"]),
                "satloss8_epoch": checkpoint_epoch(checkpoint_paths[key]["satloss8"]),
                "vanilla8_encoder_points": train_encoder_input_points(configs[key]["vanilla"]),
                "satloss8_encoder_points": train_encoder_input_points(configs[key]["satloss8"]),
            }
            for key in active_models
        },
        "top25": top25,
    }
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print("\nTop geometries by average SATLOSS8 improvement across comparison modes:")
    print("rank run_id avg_vanilla avg_satloss8 improvement_pct models_improved")
    for rank, item in enumerate(top25, start=1):
        print(
            f"{rank:>4} {int(item['run_id']):>6} "
            f"{float(item['avg_vanilla8_combined_global_rel_l2']):>11.6f} "
            f"{float(item['avg_satloss8_combined_global_rel_l2']):>12.6f} "
            f"{float(item['avg_improvement_percent']):>+15.2f}% "
            f"{int(item['models_improved'])}/{len(active_models)}"
        )
    print("\nTop geometry by mode files:")
    for mode_name in mode_defs:
        print(f"  {mode_name}: top25_{mode_name}_satloss8_improvement_geometries.csv")
    print(f"Saved ranking to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
