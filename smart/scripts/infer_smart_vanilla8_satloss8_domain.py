#!/usr/bin/env python3
"""Evaluate a vanilla8/SATLOSS8 model pair on held-out geometries.

The evaluator follows the effective training configurations:

* held-out geometry IDs come from the SATLOSS8 cluster-0/train, cluster-1/test
  split;
* each checkpoint receives its own train-configured uniform encoder budget;
* both models receive the same 65536 surface and 65536 volume query points;
* the model weights are loaded from the explicitly supplied checkpoints.

This is an aligned, non-shifted evaluation. It does not mix in beta/sine
sampling modes or use the SATLOSS8 second training view.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List

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
    SURFACE_FIELDS,
    VOLUME_FIELDS,
    build_model,
    compute_metrics,
    denorm_fields,
    load_cfg,
    predict_view_batch,
    sample_uniform_without_replacement,
)


DEFAULT_SPLIT = REPO_ROOT / "results/drivaerml_geometry_statistical_split/geometry_domain_split.json"
MODEL_DEFAULTS = {
    "smart": {
        "vanilla_config": "drivaerml",
        "satloss8_config": "drivaerml_satloss8",
        "vanilla_checkpoint": REPO_ROOT / "checkpoints/smart-smart-vanilla8-domain-cluster0-drivaerml-s42_last.pt",
        "satloss8_checkpoint": REPO_ROOT / "checkpoints/smart-satloss8-smart-satloss8-domain-cluster0-from-vanilla8-100ep-drivaerml-s42_last.pt",
    },
    "pointnet2_ssg": {
        "vanilla_config": "drivaerml_pointnet2_ssg",
        "satloss8_config": "drivaerml_pointnet2_ssg_satloss8",
        "vanilla_checkpoint": REPO_ROOT / "checkpoints/pointnet2-ssg-pointnet2-ssg-vanilla8-domain-cluster0-drivaerml-s42_last.pt",
        "satloss8_checkpoint": REPO_ROOT / "checkpoints/pointnet2-ssg-satloss8-pointnet2-ssg-satloss8-domain-cluster0-from-vanilla8-100ep-drivaerml-s42_last.pt",
    },
    "point_gnn": {
        "vanilla_config": "drivaerml_point_gnn",
        "satloss8_config": "drivaerml_point_gnn_satloss8",
        "vanilla_checkpoint": REPO_ROOT / "checkpoints/point-gnn-point-gnn-vanilla8-domain-cluster0-drivaerml-s42_last.pt",
        "satloss8_checkpoint": REPO_ROOT / "checkpoints/point-gnn-satloss8-point-gnn-satloss8-domain-cluster0-from-vanilla8-100ep-drivaerml-s42_last.pt",
    },
    "lno": {
        "vanilla_config": "drivaerml_lno",
        "satloss8_config": "drivaerml_lno_satloss8",
        "vanilla_checkpoint": REPO_ROOT / "checkpoints/lno-lno-vanilla8-domain-cluster0-drivaerml-s42_last.pt",
        "satloss8_checkpoint": REPO_ROOT / "checkpoints/lno-satloss8-lno-satloss8-domain-cluster0-from-vanilla8-100ep-drivaerml-s42_last.pt",
    },
    "lno2": {
        "vanilla_config": "drivaerml_lno2",
        "satloss8_config": "drivaerml_lno2_satloss8",
        "vanilla_checkpoint": REPO_ROOT / "checkpoints/lno2-lno2-vanilla8-domain-cluster0-drivaerml-s42_last.pt",
        "satloss8_checkpoint": REPO_ROOT / "checkpoints/lno2-satloss8-lno2-satloss8-domain-cluster0-from-vanilla8-100ep-drivaerml-s42_last.pt",
    },
    "transolverpp": {
        "vanilla_config": "drivaerml_transolverpp",
        "satloss8_config": "drivaerml_transolverpp_satloss8",
        "vanilla_checkpoint": REPO_ROOT / "checkpoints/transolverpp-transolverpp-vanilla8-domain-cluster0-drivaerml-s42_last.pt",
        "satloss8_checkpoint": REPO_ROOT / "checkpoints/transolverpp-satloss8-transolverpp-satloss8-domain-cluster0-from-vanilla8-100ep-drivaerml-s42_last.pt",
    },
    "mspt": {
        "vanilla_config": "drivaerml_mspt",
        "satloss8_config": "drivaerml_mspt_satloss8",
        "vanilla_checkpoint": REPO_ROOT / "checkpoints/mspt-mspt-vanilla8-domain-cluster0-drivaerml-s42_last.pt",
        "satloss8_checkpoint": REPO_ROOT / "checkpoints/mspt-satloss8-mspt-satloss8-domain-cluster0-from-vanilla8-100ep-drivaerml-s42_last.pt",
    },
    "point_transformer_v3": {
        "vanilla_config": "drivaerml_point_transformer_v3",
        "satloss8_config": "drivaerml_point_transformer_v3_satloss8",
        "vanilla_checkpoint": REPO_ROOT / "checkpoints/point-transformer-v3-ptv3-vanilla8-domain-cluster0-drivaerml-s42_last.pt",
        "satloss8_checkpoint": REPO_ROOT / "checkpoints/point-transformer-v3-satloss8-ptv3-satloss8-domain-cluster0-from-vanilla8-100ep-drivaerml-s42_last.pt",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-json", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--model", choices=tuple(MODEL_DEFAULTS), default="smart")
    parser.add_argument("--vanilla8-checkpoint", type=Path, default=None)
    parser.add_argument("--satloss8-checkpoint", type=Path, default=None)
    parser.add_argument("--run-ids", default="1,4", help="Held-out cluster run IDs, comma-separated.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--surface-query-points", type=int, default=65536)
    parser.add_argument("--volume-query-points", type=int, default=65536)
    parser.add_argument("--encoder-input-points", type=int, default=None)
    parser.add_argument("--query-chunk", type=int, default=65536)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )
    return parser.parse_args()


def heldout_ids(split_path: Path, requested: str) -> List[int]:
    with split_path.open("r", encoding="utf-8") as handle:
        split = json.load(handle)
    direction = split.get("train_cluster_0_test_cluster_1")
    if not isinstance(direction, dict):
        raise ValueError(f"Missing train_cluster_0_test_cluster_1 in {split_path}")
    allowed = {int(value) for value in direction.get("test_ids", [])}
    run_ids = [int(value.strip()) for value in requested.split(",") if value.strip()]
    if not run_ids:
        raise ValueError("At least one run ID is required.")
    missing = sorted(set(run_ids).difference(allowed))
    if missing:
        raise ValueError(f"Requested IDs are not in held-out cluster 1: {missing}")
    return run_ids


def load_case(dataset: AhmedMLDatasetV2, run_id: int, seed: int):
    try:
        index = dataset.data.index(int(run_id))
    except ValueError as exc:
        raise ValueError(f"run_{run_id} is not available in the held-out dataset") from exc
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    item = dataset[index]
    if len(item) != 5:
        raise ValueError(f"Expected five tensors for run_{run_id}, got {len(item)}")
    return tuple(value.float() for value in item)


def metric_summary(rows: List[Dict[str, object]], model_names: tuple[str, str]) -> Dict[str, Dict[str, float]]:
    keys = [
        "combined_global_rel_l2",
        "combined_physics_rel_l2",
        "surface_global_rel_l2",
        "volume_global_rel_l2",
        "surface_pressure_rel_l2",
        "volume_pressure_rel_l2",
        "surface_wss_mag_rel_l2",
        "volume_velocity_mag_rel_l2",
    ]
    output = {}
    for model_name in model_names:
        model_rows = [row for row in rows if row["model_name"] == model_name]
        output[model_name] = {
            f"{key}_mean": float(np.mean([float(row[key]) for row in model_rows])) for key in keys
        }
        output[model_name].update(
            {f"{key}_std": float(np.std([float(row[key]) for row in model_rows])) for key in keys}
        )
        output[model_name]["run_count"] = int(len(model_rows))
    return output


def main() -> None:
    args = parse_args()
    model_defaults = MODEL_DEFAULTS[args.model]
    vanilla_checkpoint = (args.vanilla8_checkpoint or model_defaults["vanilla_checkpoint"]).expanduser().resolve()
    satloss8_checkpoint = (args.satloss8_checkpoint or model_defaults["satloss8_checkpoint"]).expanduser().resolve()
    output_dir = args.output_dir or (REPO_ROOT / f"results/{args.model}_vanilla8_vs_satloss8_unseen_2runs")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    if not vanilla_checkpoint.is_file():
        raise FileNotFoundError(vanilla_checkpoint)
    if not satloss8_checkpoint.is_file():
        raise FileNotFoundError(satloss8_checkpoint)

    split_path = args.split_json.expanduser().resolve()
    run_ids = heldout_ids(split_path, args.run_ids)
    vanilla_cfg = load_cfg(model_defaults["vanilla_config"])
    satloss8_cfg = load_cfg(model_defaults["satloss8_config"])
    vanilla_input_points = int(getattr(vanilla_cfg, "num_body_points", 0))
    satloss8_input_points = int(getattr(satloss8_cfg, "view_geometry_points", 0))
    if vanilla_input_points <= 0 or satloss8_input_points <= 0:
        raise ValueError(
            f"The vanilla8/SATLOSS8 pair does not have positive encoder budgets: "
            f"vanilla8={vanilla_input_points}, SATLOSS8={satloss8_input_points}"
        )
    if args.encoder_input_points is not None and (
        vanilla_input_points != int(args.encoder_input_points)
        or satloss8_input_points != int(args.encoder_input_points)
    ):
        raise ValueError(
            "--encoder-input-points can only be used when both selected configs "
            f"have that budget; configs require vanilla8={vanilla_input_points}, "
            f"SATLOSS8={satloss8_input_points}, requested={args.encoder_input_points}"
        )
    vanilla_eval_input_points = vanilla_input_points
    satloss8_eval_input_points = satloss8_input_points
    if args.encoder_input_points is not None:
        vanilla_eval_input_points = satloss8_eval_input_points = int(args.encoder_input_points)
    if int(vanilla_cfg.num_surface_points) != int(args.surface_query_points):
        raise ValueError("vanilla8 surface query budget does not match the requested evaluation budget")
    if int(vanilla_cfg.num_volume_points) != int(args.volume_query_points):
        raise ValueError("vanilla8 volume query budget does not match the requested evaluation budget")
    if int(satloss8_cfg.num_surface_points) != int(args.surface_query_points):
        raise ValueError("SATLOSS8 surface query budget does not match the requested evaluation budget")
    if int(satloss8_cfg.num_volume_points) != int(args.volume_query_points):
        raise ValueError("SATLOSS8 volume query budget does not match the requested evaluation budget")

    dataset = AhmedMLDatasetV2(
        saved_folder=str(vanilla_cfg.data_path),
        if_test=True,
        geometry_points=0,
        surface_points=int(args.surface_query_points),
        volume_points=int(args.volume_query_points),
        scale_positions=bool(vanilla_cfg.scale_positions),
        require_preprocessed=True,
        domain_split_json=str(split_path),
        domain_split_train_cluster=0,
        domain_split_test_cluster=1,
    )
    mean_s = dataset.mean_surf_data
    std_s = torch.clamp(dataset.std_surf_data, min=1.0e-12)
    mean_v = dataset.mean_vol_data
    std_v = torch.clamp(dataset.std_vol_data, min=1.0e-12)

    # SATLOSS8 changes the training protocol, not the tensor architecture.
    sat_build_cfg = OmegaConf.create(OmegaConf.to_container(satloss8_cfg, resolve=True))
    sat_build_cfg.model_name = vanilla_cfg.model_name
    vanilla_model = build_model(
        vanilla_cfg,
        str(vanilla_checkpoint),
        device,
        batched_query_subregion_size=int(args.query_chunk),
    ).to(device)
    satloss8_model = build_model(
        sat_build_cfg,
        str(satloss8_checkpoint),
        device,
        batched_query_subregion_size=int(args.query_chunk),
    ).to(device)

    model_names = (f"{args.model.upper()}_VANILLA8", f"{args.model.upper()}_SATLOSS8")
    rows: List[Dict[str, object]] = []
    print(
        f"Model={args.model}; held-out cluster-1 run IDs: {run_ids}; "
        f"encoder vanilla8={vanilla_eval_input_points}, SATLOSS8={satloss8_eval_input_points}; "
        f"queries surface={args.surface_query_points}, volume={args.volume_query_points}; device={device}"
    )
    for run_id in tqdm(run_ids, desc="Unseen geometries"):
        geo_full, surf_query, surf_data, vol_query, vol_data = load_case(
            dataset, run_id, int(args.seed) + int(run_id)
        )
        rng = np.random.default_rng([int(args.seed), int(run_id), 9301])
        vanilla_geo_idx = sample_uniform_without_replacement(
            int(geo_full.shape[0]), int(vanilla_eval_input_points), rng
        )
        if vanilla_eval_input_points == satloss8_eval_input_points:
            satloss8_geo_idx = vanilla_geo_idx
        else:
            satloss8_rng = np.random.default_rng([int(args.seed), int(run_id), 9302])
            satloss8_geo_idx = sample_uniform_without_replacement(
                int(geo_full.shape[0]), int(satloss8_eval_input_points), satloss8_rng
            )
        geo_inputs = {
            model_names[0]: geo_full[torch.from_numpy(vanilla_geo_idx).long()],
            model_names[1]: geo_full[torch.from_numpy(satloss8_geo_idx).long()],
        }

        gt_s = denorm_fields(surf_data, mean_s, std_s).numpy()
        gt_v = denorm_fields(vol_data, mean_v, std_v).numpy()
        for model_name, model in (
            (model_names[0], vanilla_model),
            (model_names[1], satloss8_model),
        ):
            pred_s, pred_v = predict_view_batch(
                str(vanilla_cfg.model_name),
                model,
                geo_inputs[model_name].unsqueeze(0),
                surf_query,
                vol_query,
                None,
                mean_s,
                std_s,
                mean_v,
                std_v,
                device,
                base_seed=int(args.seed) + 1009 * int(run_id),
                repeats=1,
            )
            metrics = compute_metrics(gt_s, pred_s[0], gt_v, pred_v[0])
            rows.append({"model_name": model_name, "run_id": int(run_id), **metrics})
            print(
                f"run_{run_id} {model_name}: combined_global={metrics['combined_global_rel_l2']:.6f}, "
                f"combined_physics={metrics['combined_physics_rel_l2']:.6f}, "
                f"surface_pressure={metrics['surface_pressure_rel_l2']:.6f}, "
                f"volume_pressure={metrics['volume_pressure_rel_l2']:.6f}"
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    fields = sorted(rows[0].keys())
    with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "protocol": f"{args.model} vanilla8 vs SATLOSS8 on held-out geometry cluster 1",
        "model": args.model,
        "split_json": str(split_path),
        "run_ids": run_ids,
        "encoder_input_points": {
            "vanilla8": int(vanilla_eval_input_points),
            "satloss8": int(satloss8_eval_input_points),
        },
        "surface_query_points": int(args.surface_query_points),
        "volume_query_points": int(args.volume_query_points),
        "query_chunk": int(args.query_chunk),
        "seed": int(args.seed),
        "vanilla8_checkpoint": str(vanilla_checkpoint),
        "satloss8_checkpoint": str(satloss8_checkpoint),
        "configs": {model_names[0]: model_defaults["vanilla_config"], model_names[1]: model_defaults["satloss8_config"]},
        "metrics": metric_summary(rows, model_names),
        "per_run": rows,
        "fields": {"surface": list(SURFACE_FIELDS), "volume": list(VOLUME_FIELDS)},
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(json.dumps(payload["metrics"], indent=2))
    print(f"Saved results to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
