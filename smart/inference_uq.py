#!/usr/bin/env python3
"""Full-query inference with spatially-conditioned post-hoc UQ.

This script supports both CAT and SMART checkpoints through
`experiment.model_name` in the selected config.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from data.datasets import get_dataset
from uq_common import (
    apply_runtime_overrides,
    build_model,
    cat_surface_head_with_features,
    compute_anchor_joint_mahalanobis,
    compute_channel_rel,
    compute_lll_variance_and_alpha,
    denorm_fields,
    load_config,
    load_full_preprocessed_case,
    load_model_state_dict_only,
    normalize_pos,
    project_anchor_scores_to_queries,
    rel_l2,
    resolve_surface_targets,
    sample_input_idx,
    set_seed,
    smart_head_with_features,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Full-query inference with spatially-conditioned UQ.")
    parser.add_argument("--config-name", default="drivaerml_cat", help="Config file under smart/config (without .yaml).")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint path.")
    parser.add_argument("--uq-params", required=True, help="Path produced by calibrate_uq.py.")
    parser.add_argument("--stage", type=int, default=2, choices=[1, 2], help="CAT-only flag: stage 1 skips volume prediction.")
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--output-dir", default=None, help="Output dir for arrays + metrics json.")
    parser.add_argument("--gamma", type=float, default=0.1, help="Pointwise variance inflation weight for the projected OOD field.")
    parser.add_argument("--max-runs", type=int, default=-1, help="Limit number of evaluated runs (-1 for all).")
    parser.add_argument("--batch-size", type=int, default=None, help="Optional override for experiment.batch_size.")
    parser.add_argument("--num-workers", type=int, default=None, help="Optional override for experiment.num_workers.")
    parser.add_argument("--num-body-points", type=int, default=None, help="Optional override for experiment.num_body_points.")
    parser.add_argument("--num-surface-points", type=int, default=None, help="Optional override for experiment.num_surface_points.")
    parser.add_argument("--num-volume-points", type=int, default=None, help="Optional override for experiment.num_volume_points.")
    parser.add_argument("--single-surface-input-points", type=int, default=None, help="Optional override for CAT single_surface_input_points.")
    parser.add_argument("--single-surface-query-points", type=int, default=None, help="Optional override for CAT single_surface_query_points.")
    parser.add_argument("--single-volume-query-points", type=int, default=None, help="Optional override for CAT single_volume_query_points.")
    parser.add_argument("--latent-geometry-points", type=int, default=None, help="Optional override for architecture.latent_geometry_points.")
    parser.add_argument("--subsampled-geometry-points", type=int, default=None, help="Optional override for architecture.subsampled_geometry_points.")
    parser.add_argument("--num-encoder-decoder-blocks", type=int, default=None, help="Optional override for architecture.num_encoder_decoder_blocks.")
    parser.add_argument("--latent-dim", type=int, default=None, help="Optional override for architecture.latent_dim.")
    parser.add_argument("--pos-scale-factor", type=int, default=None, help="Optional override for architecture.pos_scale_factor.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None, help="cuda:0 / cpu. Default auto.")
    parser.add_argument("--surface-chunk", type=int, default=131072, help="Surface-query chunk size.")
    parser.add_argument("--volume-chunk", type=int, default=131072, help="Volume-query chunk size.")
    return parser.parse_args()


def choose_output_dir(args, config) -> Path:
    if args.output_dir:
        return Path(args.output_dir)
    stem = Path(args.checkpoint).stem
    return Path("results") / "uq_inference" / config.dataset / stem


def encode_inference_context(model_kind: str, model, geometry_input_b: torch.Tensor, params_b: torch.Tensor | None):
    if model_kind == "cat":
        geom_latents, anchor_pos, geom_final = model._encode_stage1(geometry_input_b[..., : model.spatial_dim])
        return {
            "geom_latents": geom_latents,
            "anchor_pos": anchor_pos,
            "anchor_pos_norm": anchor_pos / float(model.pos_scale_factor),
            "anchor_features": geom_final,
            "geom_final": geom_final,
            "params": None,
        }

    geom_latents, anchor_pos, geom_final = model.encode(
        geometry_input_b[..., : model.pos_encoder.spatial_dim],
        params_b,
        return_final=True,
    )
    return {
        "geom_latents": geom_latents,
        "anchor_pos": anchor_pos,
        "anchor_pos_norm": anchor_pos / float(model.pos_scale_factor),
        "anchor_features": geom_final,
        "geom_final": geom_final,
        "params": params_b,
    }


def decode_surface_chunk(model_kind: str, model, context, query_chunk_b: torch.Tensor, surface_target_indices: list[int]):
    if model_kind == "cat":
        q = model._decode(
            query_chunk_b[..., : model.spatial_dim],
            context["geom_latents"],
            context["anchor_pos"],
            model.surface_decoder_blocks,
        )
        surface_pred, z = cat_surface_head_with_features(model, q)
        return surface_pred, z

    q = model.decode_features(
        context["geom_latents"],
        context["anchor_pos"],
        context["params"],
        query_chunk_b[..., : model.pos_encoder.spatial_dim],
    )
    pred_all, z = smart_head_with_features(model, q)
    surface_pred = pred_all[..., : model.surface_channels]
    if surface_pred.shape[-1] != len(surface_target_indices):
        s_idx = torch.tensor(surface_target_indices, dtype=torch.long, device=surface_pred.device)
        surface_pred = surface_pred.index_select(dim=2, index=s_idx)
    return surface_pred, z


def decode_smart_volume(model, context, volume_query_b: torch.Tensor, chunk_size: int) -> torch.Tensor:
    preds = []
    for start in range(0, volume_query_b.shape[1], chunk_size):
        chunk = volume_query_b[:, start : start + chunk_size, :]
        q = model.decode_features(
            context["geom_latents"],
            context["anchor_pos"],
            context["params"],
            chunk[..., : model.pos_encoder.spatial_dim],
        )
        pred_all, _z = smart_head_with_features(model, q)
        preds.append(pred_all[..., model.surface_channels :].cpu())
    return torch.cat(preds, dim=1)


def decode_cat_volume(model, context, surface_query_b: torch.Tensor, surface_pred_b: torch.Tensor, volume_query_b: torch.Tensor, chunk_size: int) -> torch.Tensor:
    geom_latents = context["geom_latents"]
    anchor_pos = context["anchor_pos"]
    geom_final = context["geom_final"]

    prev_latents, _ = model._encode_stage2(
        surface_query_b[..., : model.spatial_dim],
        surface_pred_b,
        anchor_pos,
        initial_latent=geom_final,
    )
    new_latents, _ = model._encode_stage2(
        surface_query_b[..., : model.spatial_dim],
        surface_pred_b,
        anchor_pos,
        initial_latent=None,
    )
    w_couple, w_fuse = model._compute_dynamic_skip_weights(geom_latents, prev_latents, new_latents, surface_pred_b)

    fused_latents = []
    for layer_idx in range(model.loops):
        wc = w_couple[:, layer_idx, :].unsqueeze(-1)
        wf = w_fuse[:, layer_idx, :].unsqueeze(-1)
        geom_m = model.surface_to_volume_latent_norm(geom_latents[layer_idx])
        prev_m = model.surface_to_volume_latent_norm(prev_latents[layer_idx])
        new_m = model.surface_to_volume_latent_norm(new_latents[layer_idx])
        coupled = prev_m + wc * (geom_m - prev_m)
        fused = new_m + wf * (coupled - new_m)
        fused_latents.append(fused)

    preds = []
    for start in range(0, volume_query_b.shape[1], chunk_size):
        chunk = volume_query_b[:, start : start + chunk_size, :]
        qv = model._decode(chunk[..., : model.spatial_dim], fused_latents, anchor_pos, model.volume_decoder_blocks)
        preds.append(model.volume_head(qv).cpu())
    return torch.cat(preds, dim=1)


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_config(args.config_name)
    apply_runtime_overrides(config, args)

    train_data, test_data, _stats, spatial_dim, surf_channels, vol_channels, params_dim, fields = get_dataset(config)
    if params_dim > 0:
        raise NotImplementedError("Full-query UQ inference currently supports params_dim=0 datasets only.")

    dataset = test_data if args.split == "test" else train_data
    if not getattr(dataset, "preprocessed_mode", False):
        raise RuntimeError("Expected a preprocessed DrivAerML-style dataset layout for full-query UQ inference.")

    run_ids = list(dataset.data)
    if args.max_runs > 0:
        run_ids = run_ids[: int(args.max_runs)]

    surface_target_indices, surface_target_fields = resolve_surface_targets(fields)
    model, model_kind = build_model(
        config,
        spatial_dim=spatial_dim,
        surf_channels=surf_channels,
        vol_channels=vol_channels,
        params_dim=params_dim,
        surface_target_dim=len(surface_target_indices),
    )
    model = model.to(device)

    state_dict = load_model_state_dict_only(args.checkpoint)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[checkpoint] Missing keys: {len(missing)}", flush=True)
    if unexpected:
        print(f"[checkpoint] Unexpected keys: {len(unexpected)}", flush=True)
    del state_dict
    if device.type == "cuda":
        torch.cuda.empty_cache()
    model.eval()

    uq = torch.load(args.uq_params, map_location="cpu")
    coord_mean = uq["coord_mean"].float().to(device)
    coord_std = uq["coord_std"].float().to(device)
    feature_mean = uq["feature_mean"].float().to(device)
    feature_std = uq["feature_std"].float().to(device)
    spatial_centroids = uq["spatial_centroids"].float().to(device)
    joint_region_means = uq["joint_region_means"].float().to(device)
    joint_region_inv_covs = uq["joint_region_inv_covs"].float().to(device)
    rbf_length_scale = float(uq["rbf_length_scale"].item() if isinstance(uq["rbf_length_scale"], torch.Tensor) else uq["rbf_length_scale"])
    Sigma_LLL = uq["Sigma_LLL"].float().to(device)
    V_skew = uq["V_skew"].float().to(device)
    K = float(uq["K"].item() if isinstance(uq["K"], torch.Tensor) else uq["K"])

    out_dir = choose_output_dir(args, config)
    out_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = out_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    mean_s = dataset.mean_surf_data[surface_target_indices].float()
    std_s = torch.clamp(dataset.std_surf_data[surface_target_indices].float(), min=1e-12)
    mean_v = dataset.mean_vol_data.float()
    std_v = torch.clamp(dataset.std_vol_data.float(), min=1e-12)
    min_pos = dataset.min_pos.float()
    max_pos = dataset.max_pos.float()

    metrics = {
        "runs_processed": 0,
        "surface_rel_l2": [],
        "volume_rel_l2": [],
        "surface_channel_rel_l2": {field: [] for field in surface_target_fields},
        "volume_channel_rel_l2": {field: [] for field in fields["volume"]},
        "surface_ood_mean": [],
        "surface_ood_max": [],
        "anchor_ood_mean": [],
        "anchor_ood_max": [],
    }

    with torch.inference_mode():
        for run_id in tqdm(run_ids, desc=f"UQ inference ({args.split})", dynamic_ncols=True):
            run_dir = Path(config.data_path) / f"run_{run_id}"
            surf_coords, surf_gt_all, vol_coords, vol_gt = load_full_preprocessed_case(run_dir)
            surf_gt = surf_gt_all[:, surface_target_indices]

            surf_coords_t = torch.from_numpy(surf_coords)
            vol_coords_t = torch.from_numpy(vol_coords)
            surf_query = normalize_pos(surf_coords_t, min_pos, max_pos)
            vol_query = normalize_pos(vol_coords_t, min_pos, max_pos)

            rng = np.random.default_rng(args.seed + int(run_id))
            if model_kind == "cat":
                num_input = int(getattr(config, "single_surface_input_points", getattr(config, "num_body_points", surf_coords.shape[0])))
            else:
                num_input = int(getattr(config, "num_body_points", surf_coords.shape[0]))
            input_idx = sample_input_idx(surf_coords.shape[0], num_input, rng)
            geometry_input = surf_query[torch.from_numpy(input_idx)]

            geometry_input_b = geometry_input.unsqueeze(0).to(device)
            surf_query_b = surf_query.unsqueeze(0).to(device)
            vol_query_b = vol_query.unsqueeze(0).to(device)

            context = encode_inference_context(model_kind, model, geometry_input_b, params_b=None)
            anchor_ood, anchor_region = compute_anchor_joint_mahalanobis(
                context["anchor_pos_norm"],
                context["anchor_features"],
                coord_mean,
                coord_std,
                feature_mean,
                feature_std,
                spatial_centroids,
                joint_region_means,
                joint_region_inv_covs,
            )

            surface_pred_norm_chunks = []
            surface_pred_phys_chunks = []
            surface_var_lll_chunks = []
            surface_var_final_chunks = []
            surface_alpha_chunks = []
            surface_ood_chunks = []

            for start in range(0, surf_query_b.shape[1], int(args.surface_chunk)):
                query_chunk = surf_query_b[:, start : start + int(args.surface_chunk), :]
                surface_pred_norm, z = decode_surface_chunk(
                    model_kind,
                    model,
                    context,
                    query_chunk,
                    surface_target_indices,
                )

                projected_ood = project_anchor_scores_to_queries(
                    query_chunk,
                    context["anchor_pos_norm"],
                    anchor_ood,
                    rbf_length_scale,
                )
                modulation = 1.0 + float(args.gamma) * projected_ood[0]
                variance_lll, variance_final, alpha_exact = compute_lll_variance_and_alpha(
                    z[0],
                    Sigma_LLL,
                    V_skew,
                    K,
                    modulation,
                )

                surface_pred_norm_cpu = surface_pred_norm[0].cpu()
                surface_pred_norm_chunks.append(surface_pred_norm_cpu)
                surface_pred_phys_chunks.append(denorm_fields(surface_pred_norm_cpu, mean_s, std_s))
                surface_var_lll_chunks.append(variance_lll.cpu())
                surface_var_final_chunks.append(variance_final.cpu())
                surface_alpha_chunks.append(alpha_exact.cpu())
                surface_ood_chunks.append(projected_ood[0].cpu())

            surface_pred_norm_full = torch.cat(surface_pred_norm_chunks, dim=0)
            surface_pred_phys = torch.cat(surface_pred_phys_chunks, dim=0).numpy()
            surface_var_lll = torch.cat(surface_var_lll_chunks, dim=0).numpy()
            surface_var_final = torch.cat(surface_var_final_chunks, dim=0).numpy()
            surface_alpha = torch.cat(surface_alpha_chunks, dim=0).numpy()
            surface_ood = torch.cat(surface_ood_chunks, dim=0).numpy()

            if model_kind == "smart":
                volume_pred_norm = decode_smart_volume(
                    model,
                    context,
                    vol_query_b,
                    chunk_size=int(args.volume_chunk),
                )[0]
                volume_pred_phys = denorm_fields(volume_pred_norm, mean_v, std_v).numpy()
            elif int(args.stage) == 2:
                volume_pred_norm = decode_cat_volume(
                    model,
                    context,
                    surf_query_b,
                    surface_pred_norm_full.unsqueeze(0).to(device),
                    vol_query_b,
                    chunk_size=int(args.volume_chunk),
                )[0]
                volume_pred_phys = denorm_fields(volume_pred_norm, mean_v, std_v).numpy()
            else:
                volume_pred_phys = None

            surface_rel = rel_l2(surf_gt.reshape(-1), surface_pred_phys.reshape(-1))
            surface_ch = compute_channel_rel(surf_gt, surface_pred_phys, surface_target_fields)
            metrics["surface_rel_l2"].append(surface_rel)
            for key, value in surface_ch.items():
                metrics["surface_channel_rel_l2"][key].append(value)

            metrics["surface_ood_mean"].append(float(surface_ood.mean()))
            metrics["surface_ood_max"].append(float(surface_ood.max()))
            metrics["anchor_ood_mean"].append(float(anchor_ood.mean().item()))
            metrics["anchor_ood_max"].append(float(anchor_ood.max().item()))

            if volume_pred_phys is not None:
                volume_rel = rel_l2(vol_gt.reshape(-1), volume_pred_phys.reshape(-1))
                volume_ch = compute_channel_rel(vol_gt, volume_pred_phys, fields["volume"])
                metrics["volume_rel_l2"].append(volume_rel)
                for key, value in volume_ch.items():
                    metrics["volume_channel_rel_l2"][key].append(value)

            np.savez_compressed(
                runs_dir / f"run_{run_id}_uq.npz",
                run_id=np.array([run_id], dtype=np.int64),
                surface_coords=surf_coords,
                surface_gt=surf_gt,
                surface_mu_pred=surface_pred_phys,
                surface_variance_lll=surface_var_lll,
                surface_variance_final=surface_var_final,
                surface_alpha_exact=surface_alpha,
                surface_ood_score=surface_ood,
                anchor_coords=context["anchor_pos_norm"][0].cpu().numpy(),
                anchor_ood_score=anchor_ood[0].cpu().numpy(),
                anchor_region_id=anchor_region[0].cpu().numpy(),
                spatial_centroids=spatial_centroids.cpu().numpy(),
                rbf_length_scale=np.array([rbf_length_scale], dtype=np.float32),
                volume_coords=vol_coords,
                volume_gt=vol_gt,
                volume_mu_pred=np.zeros((0, 0), dtype=np.float32) if volume_pred_phys is None else volume_pred_phys,
            )
            metrics["runs_processed"] += 1

    def mean_or_nan(values):
        return float(np.mean(values)) if values else float("nan")

    summary = {
        "config_name": args.config_name,
        "checkpoint": args.checkpoint,
        "uq_params": args.uq_params,
        "model_name": str(getattr(config, "model_name", model_kind)).upper(),
        "stage": int(args.stage) if model_kind == "cat" else None,
        "split": args.split,
        "gamma": float(args.gamma),
        "runs_processed": int(metrics["runs_processed"]),
        "surface_rel_l2_mean": mean_or_nan(metrics["surface_rel_l2"]),
        "volume_rel_l2_mean": mean_or_nan(metrics["volume_rel_l2"]),
        "surface_ood_mean": mean_or_nan(metrics["surface_ood_mean"]),
        "surface_ood_max_mean": mean_or_nan(metrics["surface_ood_max"]),
        "anchor_ood_mean": mean_or_nan(metrics["anchor_ood_mean"]),
        "anchor_ood_max_mean": mean_or_nan(metrics["anchor_ood_max"]),
        "surface_channel_rel_l2_mean": {key: mean_or_nan(value) for key, value in metrics["surface_channel_rel_l2"].items()},
        "volume_channel_rel_l2_mean": {key: mean_or_nan(value) for key, value in metrics["volume_channel_rel_l2"].items()},
    }
    (out_dir / "uq_metrics.json").write_text(json.dumps(summary, indent=2))

    print(f"Saved run arrays to: {runs_dir}")
    print(f"Saved summary metrics to: {out_dir / 'uq_metrics.json'}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
