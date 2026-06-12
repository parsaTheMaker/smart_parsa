#!/usr/bin/env python3
"""Offline calibration for spatially-conditioned post-hoc UQ.

This calibration script now supports both CAT and SMART checkpoints through
`experiment.model_name` in the selected config. It saves:
  - spatial centroids over anchor coordinates
  - regional joint coordinate-feature means and inverse covariances
  - coordinate and feature standardization statistics
  - dynamic RBF length scale
  - skew-corrected last-layer Laplace tensors for surface-query uncertainty
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from data.datasets import get_dataset
from uq_common import (
    apply_runtime_overrides,
    build_model,
    cat_surface_head_with_features,
    compute_knn_length_scale,
    extract_cat_surface_features,
    extract_smart_surface_features,
    kmeans_plus_plus,
    load_config,
    load_model_state_dict_only,
    prepare_cat_surface_batch,
    resolve_surface_targets,
    set_seed,
    smart_head_with_features,
    split_optional_params,
)


def autocast_context(config, device: torch.device):
    if not bool(getattr(config, "amp", False)):
        return torch.autocast(device_type=device.type, enabled=False)
    precisions = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    dtype = precisions.get(str(getattr(config, "precision", "float16")), torch.float16)
    return torch.autocast(device_type=device.type, dtype=dtype, enabled=(device.type != "cpu"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate spatially-conditioned UQ parameters.")
    parser.add_argument("--config-name", default="drivaerml_cat", help="Config file under smart/config (without .yaml).")
    parser.add_argument("--checkpoint", required=True, help="Path to the trained checkpoint.")
    parser.add_argument("--output", default=None, help="Output .pt path. Default: checkpoints/<ckpt_stem>_uq_params.pt")
    parser.add_argument("--max-batches", type=int, default=80, help="Number of train batches for calibration.")
    parser.add_argument("--probe-batches", type=int, default=1, help="Number of batches reserved for the skew probe.")
    parser.add_argument("--batch-size", type=int, default=None, help="Optional dataloader batch-size override.")
    parser.add_argument("--num-workers", type=int, default=0, help="Calibration dataloader workers.")
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
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--device", default=None, help="cuda:0 / cpu. Default auto.")
    parser.add_argument("--num-regions", type=int, default=10, help="Number of spatial macro-regions.")
    parser.add_argument("--kmeans-iters", type=int, default=25, help="Number of k-means refinement steps.")
    parser.add_argument("--knn-k", type=int, default=5, help="Neighbor rank used for the RBF length scale.")
    parser.add_argument("--knn-query-chunk", type=int, default=512, help="Query chunk size for exact kNN length-scale computation.")
    parser.add_argument("--rbf-scale", type=float, default=1.5, help="Multiplier applied to the median kNN distance.")
    parser.add_argument("--pinv-rcond", type=float, default=1e-6, help="rcond used for pseudo-inverses.")
    parser.add_argument("--hessian-damping", type=float, default=1e-4, help="Diagonal damping for Hessian inversion.")
    parser.add_argument("--eps", type=float, default=1e-3, help="Finite-difference step for the skew probe.")
    parser.add_argument("--surface-chunk", type=int, default=8192, help="Surface-query chunk size used during calibration/probe passes.")
    return parser.parse_args()


def output_path_from_ckpt(ckpt_path: str, explicit_output: str | None) -> Path:
    if explicit_output:
        return Path(explicit_output)
    path = Path(ckpt_path)
    stem = path.stem
    for suffix in ("_best", "_last"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    return path.parent / f"{stem}_uq_params.pt"


def prepare_surface_batch(model_kind: str, batch, config, device: torch.device, surface_target_indices: list[int]):
    if model_kind == "cat":
        surface_input_tokens, surface_query_tokens, surface_target = prepare_cat_surface_batch(
            batch,
            config,
            device,
            surface_target_indices,
        )
        return {
            "surface_input_tokens": surface_input_tokens,
            "surface_query_tokens": surface_query_tokens.cpu(),
            "surface_target": surface_target.cpu(),
            "params": None,
        }

    geo_mesh, surf_mesh, surf_data, _vol_mesh, _vol_data, params = split_optional_params(batch)
    geo_mesh = geo_mesh.to(device)
    surf_mesh = surf_mesh.cpu()
    surf_data = surf_data.cpu()
    params = None if params is None else params.to(device)
    s_idx = torch.tensor(surface_target_indices, dtype=torch.long, device=surf_data.device)
    surface_target = surf_data.index_select(dim=2, index=s_idx)
    return {
        "geo_mesh": geo_mesh,
        "surface_query_tokens": surf_mesh,
        "surface_target": surface_target,
        "params": params,
    }


def run_surface_feature_pass(model_kind: str, model, batch_inputs, surface_target_indices: list[int], config, device: torch.device):
    if model_kind == "cat":
        with autocast_context(config, device):
            return extract_cat_surface_features(
                model,
                batch_inputs["surface_input_tokens"],
                batch_inputs["surface_query_tokens"],
            )

    with autocast_context(config, device):
        surface_pred, q, z, aux = extract_smart_surface_features(
            model,
            batch_inputs["geo_mesh"],
            batch_inputs["surface_query_tokens"],
            batch_inputs["params"],
        )
    if surface_pred.shape[-1] != len(surface_target_indices):
        s_idx = torch.tensor(surface_target_indices, dtype=torch.long, device=surface_pred.device)
        surface_pred = surface_pred.index_select(dim=2, index=s_idx)
    return surface_pred, q, z, aux


def build_surface_context(model_kind: str, model, batch_inputs, config, device: torch.device):
    if model_kind == "cat":
        with autocast_context(config, device):
            surface_input_pos = batch_inputs["surface_input_tokens"][..., : model.spatial_dim]
            geom_latents, anchor_pos, geom_final = model._encode_stage1(surface_input_pos)
        return {
            "geom_latents": geom_latents,
            "anchor_pos": anchor_pos,
            "anchor_pos_norm": anchor_pos / float(model.pos_scale_factor),
            "anchor_features": geom_final,
            "params": None,
        }

    with autocast_context(config, device):
        geom_latents, anchor_pos, geom_final = model.encode(
            batch_inputs["geo_mesh"][..., : model.pos_encoder.spatial_dim],
            batch_inputs["params"],
            return_final=True,
        )
    return {
        "geom_latents": geom_latents,
        "anchor_pos": anchor_pos,
        "anchor_pos_norm": anchor_pos / float(model.pos_scale_factor),
        "anchor_features": geom_final,
        "params": batch_inputs["params"],
    }


def iter_surface_feature_chunks(model_kind: str, model, context, surface_query_tokens: torch.Tensor, surface_target_indices: list[int], chunk_size: int, config, device: torch.device):
    n_query = int(surface_query_tokens.shape[1])
    for start in range(0, n_query, max(1, int(chunk_size))):
        query_chunk = surface_query_tokens[:, start : start + int(chunk_size), :].to(device, non_blocking=True)
        if model_kind == "cat":
            with autocast_context(config, device):
                q = model._decode(
                    query_chunk[..., : model.spatial_dim],
                    context["geom_latents"],
                    context["anchor_pos"],
                    model.surface_decoder_blocks,
                )
                surface_pred, z = cat_surface_head_with_features(model, q)
            yield start, surface_pred, z
            continue

        with autocast_context(config, device):
            q = model.decode_features(
                context["geom_latents"],
                context["anchor_pos"],
                context["params"],
                query_chunk[..., : model.pos_encoder.spatial_dim],
            )
            pred_all, z = smart_head_with_features(model, q)
            surface_pred = pred_all[..., : model.surface_channels]
        if surface_pred.shape[-1] != len(surface_target_indices):
            s_idx = torch.tensor(surface_target_indices, dtype=torch.long, device=surface_pred.device)
            surface_pred = surface_pred.index_select(dim=2, index=s_idx)
        yield start, surface_pred, z


def build_probe_weight_delta(model_kind: str, final_linear: torch.nn.Linear, principal: torch.Tensor, surface_dim: int):
    weight = final_linear.weight.detach()
    delta = torch.zeros_like(weight)
    principal_row = principal.to(device=weight.device, dtype=weight.dtype).unsqueeze(0)

    if model_kind == "cat":
        delta = principal_row.repeat(weight.shape[0], 1)
    else:
        delta[:surface_dim, :] = principal_row.repeat(surface_dim, 1)

    delta = delta / torch.clamp(torch.linalg.norm(delta), min=1e-12)
    return delta


def make_train_loader(train_data, config, seed: int):
    dl_kwargs = dict(
        batch_size=int(config.batch_size),
        shuffle=True,
        num_workers=int(config.num_workers),
        pin_memory=bool(getattr(config, "pin_memory", True)),
        generator=torch.Generator().manual_seed(int(seed)),
    )
    if dl_kwargs["num_workers"] > 0:
        dl_kwargs["persistent_workers"] = True
        dl_kwargs["prefetch_factor"] = int(getattr(config, "prefetch_factor", 2))
    return torch.utils.data.DataLoader(train_data, **dl_kwargs)


def move_batch_to_cpu(batch_inputs):
    return {
        key: value.detach().cpu().clone() if torch.is_tensor(value) else value
        for key, value in batch_inputs.items()
    }


def move_batch_to_device(batch_inputs, device: torch.device):
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch_inputs.items()
    }


def finalize_region_inverse_covariances(
    region_counts: torch.Tensor,
    region_sum: torch.Tensor,
    region_sum_outer: torch.Tensor,
    region_beta_sum: torch.Tensor,
    pinv_rcond: float,
):
    num_regions, joint_dim = region_sum.shape
    joint_region_means = []
    joint_region_inv_covs = []
    eye = None

    for region_idx in range(num_regions):
        n_samples = int(region_counts[region_idx].item())
        if n_samples <= 1:
            mean = torch.zeros((joint_dim,), dtype=torch.double)
            inv_cov = torch.eye(joint_dim, dtype=torch.double)
            joint_region_means.append(mean.float())
            joint_region_inv_covs.append(inv_cov.float())
            continue

        sum_vec = region_sum[region_idx]
        sum_outer = region_sum_outer[region_idx]
        mean = sum_vec / float(n_samples)
        scatter = sum_outer - float(n_samples) * torch.outer(mean, mean)
        scatter = 0.5 * (scatter + scatter.T)
        emp_cov = scatter / float(n_samples)

        emp_cov_trace = torch.diag(emp_cov)
        mu = emp_cov_trace.sum() / float(joint_dim)
        beta_ = region_beta_sum[region_idx]
        delta_ = torch.sum(scatter * scatter) / float(n_samples * n_samples)
        beta = (beta_ / float(n_samples) - delta_) / float(joint_dim * n_samples)
        delta = (delta_ - 2.0 * mu * emp_cov_trace.sum() + joint_dim * mu * mu) / float(joint_dim)

        beta = torch.clamp(beta, min=0.0)
        if float(delta.item()) <= 0.0:
            shrinkage = 0.0
        else:
            beta = torch.minimum(beta, delta)
            shrinkage = float((beta / delta).item()) if float(beta.item()) > 0.0 else 0.0

        if eye is None:
            eye = torch.eye(joint_dim, dtype=torch.double)
        cov = (1.0 - shrinkage) * emp_cov + shrinkage * mu * eye
        inv_cov = torch.linalg.pinv(cov, rcond=float(pinv_rcond))
        joint_region_means.append(mean.float())
        joint_region_inv_covs.append(inv_cov.float())

    return torch.stack(joint_region_means, dim=0), torch.stack(joint_region_inv_covs, dim=0)


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_config(args.config_name)
    apply_runtime_overrides(config, args)

    train_data, _test_data, _stats, spatial_dim, surf_channels, vol_channels, params_dim, fields = get_dataset(config)
    surface_target_indices, surface_target_fields = resolve_surface_targets(fields)

    print("[calibrate_uq] Building model...", flush=True)
    model, model_kind = build_model(
        config,
        spatial_dim=spatial_dim,
        surf_channels=surf_channels,
        vol_channels=vol_channels,
        params_dim=params_dim,
        surface_target_dim=len(surface_target_indices),
    )
    model = model.to(device)

    print("[calibrate_uq] Loading model weights on CPU...", flush=True)
    state_dict = load_model_state_dict_only(args.checkpoint)
    print("[calibrate_uq] Transferring model weights into model...", flush=True)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[checkpoint] Missing keys: {len(missing)}", flush=True)
    if unexpected:
        print(f"[checkpoint] Unexpected keys: {len(unexpected)}", flush=True)
    del state_dict
    if device.type == "cuda":
        torch.cuda.empty_cache()
    model.eval()
    print("[calibrate_uq] Model ready. Starting streaming calibration passes...", flush=True)

    surface_dim = len(surface_target_indices)
    anchor_coord_batches = []
    probe_batches = []
    hessian = None
    processed = 0
    coord_count = 0
    feature_count = 0
    coord_sum = torch.zeros((spatial_dim,), dtype=torch.double)
    coord_sum_sq = torch.zeros((spatial_dim,), dtype=torch.double)
    feature_sum = None
    feature_sum_sq = None

    with torch.inference_mode():
        for batch in make_train_loader(train_data, config, args.seed):
            print(f"[calibrate_uq] Pass 1 batch {processed + 1}/{int(args.max_batches)}", flush=True)
            batch_inputs = prepare_surface_batch(model_kind, batch, config, device, surface_target_indices)
            context = build_surface_context(model_kind, model, batch_inputs, config, device)

            coord_batch = context["anchor_pos_norm"].reshape(-1, spatial_dim).double().cpu()
            feature_batch = context["anchor_features"].reshape(-1, context["anchor_features"].shape[-1]).double().cpu()
            if feature_sum is None:
                feature_sum = torch.zeros((feature_batch.shape[-1],), dtype=torch.double)
                feature_sum_sq = torch.zeros((feature_batch.shape[-1],), dtype=torch.double)

            anchor_coord_batches.append(coord_batch.float())
            coord_sum += coord_batch.sum(dim=0)
            coord_sum_sq += torch.sum(coord_batch * coord_batch, dim=0)
            coord_count += int(coord_batch.shape[0])

            feature_sum += feature_batch.sum(dim=0)
            feature_sum_sq += torch.sum(feature_batch * feature_batch, dim=0)
            feature_count += int(feature_batch.shape[0])

            for _start, _surface_pred, z in iter_surface_feature_chunks(
                model_kind,
                model,
                context,
                batch_inputs["surface_query_tokens"],
                surface_target_indices,
                chunk_size=int(args.surface_chunk),
                config=config,
                device=device,
            ):
                z2 = z.reshape(-1, z.shape[-1]).double().cpu()
                cur_hessian = z2.T @ z2
                hessian = cur_hessian if hessian is None else (hessian + cur_hessian)
                del z2, cur_hessian, z

            if len(probe_batches) < int(args.probe_batches):
                probe_batches.append(move_batch_to_cpu(batch_inputs))

            del context, batch_inputs
            if device.type == "cuda":
                torch.cuda.empty_cache()

            processed += 1
            if processed >= int(args.max_batches):
                break

    if processed == 0:
        raise RuntimeError("No batches were processed during calibration.")
    if hessian is None:
        raise RuntimeError("No Hessian statistics were accumulated.")
    if not probe_batches:
        raise RuntimeError("No probe batch was captured for the skew probe.")
    if feature_sum is None or feature_sum_sq is None or coord_count <= 0 or feature_count <= 0:
        raise RuntimeError("Failed to accumulate anchor moments during calibration.")

    anchor_coords = torch.cat(anchor_coord_batches, dim=0)
    coord_mean = coord_sum / float(coord_count)
    coord_var = torch.clamp(coord_sum_sq / float(coord_count) - coord_mean * coord_mean, min=1e-6)
    coord_std = torch.sqrt(coord_var)
    feature_mean = feature_sum / float(feature_count)
    feature_var = torch.clamp(feature_sum_sq / float(feature_count) - feature_mean * feature_mean, min=1e-6)
    feature_std = torch.sqrt(feature_var)
    spatial_centroids = kmeans_plus_plus(anchor_coords.float(), int(args.num_regions), num_iters=int(args.kmeans_iters)).cpu()
    rbf_length_scale = compute_knn_length_scale(
        anchor_coords,
        knn_k=int(args.knn_k),
        scale=float(args.rbf_scale),
        query_chunk_size=int(args.knn_query_chunk),
    )

    joint_dim = int(spatial_dim + feature_mean.numel())
    num_regions = int(args.num_regions)
    region_counts = torch.zeros((num_regions,), dtype=torch.long)
    region_sum = torch.zeros((num_regions, joint_dim), dtype=torch.double)
    region_sum_outer = torch.zeros((num_regions, joint_dim, joint_dim), dtype=torch.double)

    with torch.inference_mode():
        processed_second = 0
        for batch in make_train_loader(train_data, config, args.seed):
            print(f"[calibrate_uq] Pass 2 batch {processed_second + 1}/{int(args.max_batches)}", flush=True)
            batch_inputs = prepare_surface_batch(model_kind, batch, config, device, surface_target_indices)
            context = build_surface_context(model_kind, model, batch_inputs, config, device)
            coord_batch = context["anchor_pos_norm"].reshape(-1, spatial_dim).double().cpu()
            feature_batch = context["anchor_features"].reshape(-1, feature_mean.numel()).double().cpu()
            joint_batch = torch.cat(
                [
                    (coord_batch - coord_mean) / coord_std,
                    (feature_batch - feature_mean) / feature_std,
                ],
                dim=-1,
            )
            assign = torch.cdist(coord_batch.float(), spatial_centroids.float()).argmin(dim=1)
            for region_idx in range(num_regions):
                mask = assign == region_idx
                if not torch.any(mask):
                    continue
                region_joint = joint_batch[mask]
                region_counts[region_idx] += int(region_joint.shape[0])
                region_sum[region_idx] += region_joint.sum(dim=0)
                region_sum_outer[region_idx] += region_joint.T @ region_joint
            del coord_batch, feature_batch, joint_batch, assign, context, batch_inputs
            if device.type == "cuda":
                torch.cuda.empty_cache()
            processed_second += 1
            if processed_second >= int(args.max_batches):
                break

    provisional_means = region_sum / torch.clamp(region_counts.unsqueeze(-1).double(), min=1.0)
    global_joint_sum = region_sum.sum(dim=0)
    global_joint_outer = region_sum_outer.sum(dim=0)
    global_count = int(region_counts.sum().item())
    empty_regions = torch.where(region_counts == 0)[0]
    for region_idx in empty_regions.tolist():
        region_counts[region_idx] = global_count
        region_sum[region_idx] = global_joint_sum
        region_sum_outer[region_idx] = global_joint_outer
        provisional_means[region_idx] = global_joint_sum / float(max(global_count, 1))

    region_beta_sum = torch.zeros((num_regions,), dtype=torch.double)
    with torch.inference_mode():
        processed_third = 0
        for batch in make_train_loader(train_data, config, args.seed):
            print(f"[calibrate_uq] Pass 3 batch {processed_third + 1}/{int(args.max_batches)}", flush=True)
            batch_inputs = prepare_surface_batch(model_kind, batch, config, device, surface_target_indices)
            context = build_surface_context(model_kind, model, batch_inputs, config, device)
            coord_batch = context["anchor_pos_norm"].reshape(-1, spatial_dim).double().cpu()
            feature_batch = context["anchor_features"].reshape(-1, feature_mean.numel()).double().cpu()
            joint_batch = torch.cat(
                [
                    (coord_batch - coord_mean) / coord_std,
                    (feature_batch - feature_mean) / feature_std,
                ],
                dim=-1,
            )
            assign = torch.cdist(coord_batch.float(), spatial_centroids.float()).argmin(dim=1)
            for region_idx in range(num_regions):
                mask = assign == region_idx
                if not torch.any(mask):
                    continue
                centered = joint_batch[mask] - provisional_means[region_idx]
                region_beta_sum[region_idx] += torch.sum(torch.sum(centered * centered, dim=1) ** 2)
            del coord_batch, feature_batch, joint_batch, assign, context, batch_inputs
            if device.type == "cuda":
                torch.cuda.empty_cache()
            processed_third += 1
            if processed_third >= int(args.max_batches):
                break

    joint_region_means, joint_region_inv_covs = finalize_region_inverse_covariances(
        region_counts=region_counts,
        region_sum=region_sum,
        region_sum_outer=region_sum_outer,
        region_beta_sum=region_beta_sum,
        pinv_rcond=float(args.pinv_rcond),
    )

    hessian = hessian.double()
    z_dim = hessian.shape[0]
    hessian_reg = hessian + float(args.hessian_damping) * torch.eye(z_dim, dtype=torch.double)
    Sigma_LLL = torch.linalg.pinv(hessian_reg, rcond=float(args.pinv_rcond))

    eigvals, eigvecs = torch.linalg.eigh(Sigma_LLL)
    del eigvals
    principal = eigvecs[:, -1]
    principal = principal / torch.clamp(torch.linalg.norm(principal), min=1e-12)

    final_linear: torch.nn.Linear = model.stage2_head[-1] if model_kind == "cat" else model.mlp[-1]
    weight_orig = final_linear.weight.detach().clone()
    delta_w = build_probe_weight_delta(model_kind, final_linear, principal, surface_dim=surface_dim)
    eps = float(args.eps)

    probe_batch = probe_batches[0]

    def probe_loss(scale: float) -> float:
        with torch.no_grad():
            final_linear.weight.copy_(weight_orig + (scale * eps) * delta_w)
            probe_batch_device = move_batch_to_device(probe_batch, device)
            probe_context = build_surface_context(model_kind, model, probe_batch_device, config, device)
            loss_sum = 0.0
            elem_count = 0
            for start, pred, _z in iter_surface_feature_chunks(
                model_kind,
                model,
                probe_context,
                probe_batch_device["surface_query_tokens"],
                surface_target_indices,
                chunk_size=int(args.surface_chunk),
                config=config,
                device=device,
            ):
                target_chunk = probe_batch_device["surface_target"][:, start : start + pred.shape[1], :]
                loss_sum += float(F.mse_loss(pred, target_chunk, reduction="sum").item())
                elem_count += int(target_chunk.numel())
                del pred, _z, target_chunk
            del probe_context, probe_batch_device
            if device.type == "cuda":
                torch.cuda.empty_cache()
        return loss_sum / max(elem_count, 1)

    with torch.no_grad():
        f_m2 = probe_loss(-2.0)
        f_m1 = probe_loss(-1.0)
        f_p1 = probe_loss(1.0)
        f_p2 = probe_loss(2.0)
        final_linear.weight.copy_(weight_orig)

    d3 = (f_m2 - 2.0 * f_m1 + 2.0 * f_p1 - f_p2) / (2.0 * (eps ** 3))
    alpha_w = (d3 * principal).double().cpu()
    V_skew = (alpha_w @ Sigma_LLL).double().cpu()
    K = float(1.0 + (alpha_w @ Sigma_LLL @ alpha_w).item())

    out_path = output_path_from_ckpt(args.checkpoint, args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "uq_mode": "spatial_joint_local_v1",
        "model_name": str(getattr(config, "model_name", model_kind)).upper(),
        "surface_target_fields": surface_target_fields,
        "coord_mean": coord_mean.float(),
        "coord_std": coord_std.float(),
        "feature_mean": feature_mean.float(),
        "feature_std": feature_std.float(),
        "spatial_centroids": spatial_centroids.float(),
        "joint_region_means": joint_region_means.float(),
        "joint_region_inv_covs": joint_region_inv_covs.float(),
        "region_counts": region_counts,
        "rbf_length_scale": torch.tensor(float(rbf_length_scale), dtype=torch.float32),
        "Sigma_LLL": Sigma_LLL.float(),
        "alpha_w": alpha_w.float(),
        "V_skew": V_skew.float(),
        "K": torch.tensor(K, dtype=torch.float32),
        "calibration_meta": {
            "checkpoint": str(args.checkpoint),
            "config_name": args.config_name,
            "processed_batches": int(processed),
            "num_regions": int(args.num_regions),
            "kmeans_iters": int(args.kmeans_iters),
            "knn_k": int(args.knn_k),
            "knn_query_chunk": int(args.knn_query_chunk),
            "rbf_scale": float(args.rbf_scale),
            "rbf_length_scale": float(rbf_length_scale),
            "pinv_rcond": float(args.pinv_rcond),
            "hessian_damping": float(args.hessian_damping),
            "eps": float(args.eps),
            "surface_dim": int(surface_dim),
            "feature_dim": int(feature_mean.numel()),
            "z_dim": int(z_dim),
            "anchor_count": int(anchor_coords.shape[0]),
            "skew_directional_d3": float(d3),
            "K": float(K),
        },
    }
    torch.save(payload, out_path)

    print(f"Saved UQ calibration params to: {out_path}")
    print(json.dumps(payload["calibration_meta"], indent=2))


if __name__ == "__main__":
    main()
