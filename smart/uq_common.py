from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from models.smart.cat import CAT
from models.smart.smart import SMART


SURFACE_PREFERRED = [
    "pressure",
    "normal_x",
    "normal_y",
    "normal_z",
    "wall_shear_x",
    "wall_shear_y",
    "wall_shear_z",
]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(config_name: str):
    cfg_path = Path(__file__).resolve().parent / "config" / f"{config_name}.yaml"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    return OmegaConf.load(cfg_path).experiment


def load_model_state_dict_only(checkpoint_path: str):
    """Load only the model state from a training checkpoint onto CPU.

    Training checkpoints in this repo also contain optimizer/scheduler states and
    duplicated component state dicts. Loading them directly onto CUDA can exhaust
    GPU memory before calibration or inference even starts.
    """
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", mmap=True)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    return checkpoint.get("model_state_dict", checkpoint)


def apply_runtime_overrides(config, args):
    override_map = {
        "batch_size": "batch_size",
        "num_workers": "num_workers",
        "num_body_points": "num_body_points",
        "num_surface_points": "num_surface_points",
        "num_volume_points": "num_volume_points",
        "single_surface_input_points": "single_surface_input_points",
        "single_surface_query_points": "single_surface_query_points",
        "single_volume_query_points": "single_volume_query_points",
    }
    for arg_name, cfg_name in override_map.items():
        value = getattr(args, arg_name, None)
        if value is not None:
            setattr(config, cfg_name, int(value))

    arch_override_map = {
        "latent_geometry_points": "latent_geometry_points",
        "subsampled_geometry_points": "subsampled_geometry_points",
        "num_encoder_decoder_blocks": "num_encoder_decoder_blocks",
        "latent_dim": "latent_dim",
        "pos_scale_factor": "pos_scale_factor",
    }
    for arg_name, cfg_name in arch_override_map.items():
        value = getattr(args, arg_name, None)
        if value is not None:
            setattr(config.architecture, cfg_name, int(value))


def split_optional_params(batch):
    if len(batch) == 6:
        geo_mesh, surf_mesh, surf_data, vol_mesh, vol_data, params = batch
        return geo_mesh, surf_mesh, surf_data, vol_mesh, vol_data, params
    geo_mesh, surf_mesh, surf_data, vol_mesh, vol_data = batch
    return geo_mesh, surf_mesh, surf_data, vol_mesh, vol_data, None


def resolve_surface_targets(fields: Dict[str, List[str]]) -> Tuple[List[int], List[str]]:
    surface_fields = list(fields.get("surface", []))
    idx = [surface_fields.index(name) for name in SURFACE_PREFERRED if name in surface_fields]
    if not idx:
        idx = list(range(len(surface_fields)))
    return idx, [surface_fields[i] for i in idx]


def sample_indices(
    n: int,
    k: int,
    device: torch.device,
    disjoint_from: torch.Tensor | None = None,
) -> torch.Tensor:
    if k <= 0:
        return torch.empty((0,), dtype=torch.long, device=device)
    if disjoint_from is not None:
        mask = torch.ones((n,), dtype=torch.bool, device=device)
        mask[disjoint_from] = False
        candidate = torch.where(mask)[0]
        if candidate.numel() == 0:
            return torch.randint(0, n, (k,), device=device)
        if k == candidate.numel():
            return candidate
        if k <= candidate.numel():
            perm = torch.randperm(candidate.numel(), device=device)[:k]
            return candidate[perm]
        extra = candidate[torch.randint(0, candidate.numel(), (k - candidate.numel(),), device=device)]
        return torch.cat([candidate, extra], dim=0)
    if k == n:
        return torch.arange(n, device=device)
    if k <= n:
        return torch.randperm(n, device=device)[:k]
    extra = torch.randint(0, n, (k - n,), device=device)
    return torch.cat([torch.arange(n, device=device), extra], dim=0)


def gather_per_batch(x: torch.Tensor, idx_list: List[torch.Tensor]) -> torch.Tensor:
    return torch.stack([x[b, idx_list[b], :] for b in range(x.shape[0])], dim=0)


def prepare_cat_surface_batch(batch, config, device: torch.device, surface_target_indices: List[int]):
    geo_mesh, surf_mesh, surf_data, _vol_mesh, _vol_data, params = split_optional_params(batch)
    if params is not None:
        raise NotImplementedError("CAT UQ currently supports params_dim=0 datasets only.")

    geo_mesh = geo_mesh.to(device)
    surf_mesh = surf_mesh.to(device)
    surf_data = surf_data.to(device)

    bsz, ng, _ = geo_mesh.shape
    ns = surf_mesh.shape[1]
    num_body_points = int(getattr(config, "num_body_points", ng))
    s_in = int(getattr(config, "single_surface_input_points", num_body_points))
    # Match CAT training exactly: the dataset already applies any surface query cap.
    s_q = ns
    if num_body_points <= 0:
        s_in = ng
    if s_in <= 0:
        s_in = ng

    enc_idx = []
    surf_q_idx = []
    for _ in range(bsz):
        e = sample_indices(ng, s_in, device)
        disjoint = e if ng == ns else None
        sq = sample_indices(ns, s_q, device, disjoint_from=disjoint)
        enc_idx.append(e)
        surf_q_idx.append(sq)

    s_idx = torch.tensor(surface_target_indices, dtype=torch.long, device=surf_data.device)
    surface_input_tokens = gather_per_batch(geo_mesh, enc_idx)
    surface_query_tokens = gather_per_batch(surf_mesh, surf_q_idx)
    surface_target = gather_per_batch(surf_data.index_select(dim=2, index=s_idx), surf_q_idx)
    return surface_input_tokens, surface_query_tokens, surface_target


def build_model(config, spatial_dim: int, surf_channels: int, vol_channels: int, params_dim: int, surface_target_dim: int):
    arch = OmegaConf.to_container(config.architecture, resolve=True)
    model_name = str(getattr(config, "model_name", "")).strip().upper()

    if model_name == "CAT":
        arch["stage2_surface_channels"] = int(surface_target_dim)
        model = CAT(
            spatial_dim=spatial_dim,
            surface_channels=surf_channels,
            volume_channels=vol_channels,
            parameter_channels=params_dim,
            **arch,
        )
        return model, "cat"

    if model_name == "SMART":
        model = SMART(
            spatial_dim=spatial_dim,
            surface_channels=surf_channels,
            volume_channels=vol_channels,
            parameter_channels=params_dim,
            **arch,
        )
        return model, "smart"

    raise ValueError(f"Unsupported model_name for UQ: {config.model_name}")


def cat_surface_head_with_features(model: CAT, q: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    z1 = model.stage2_head[0](q)
    z2 = model.stage2_head[1](z1)
    z3 = model.stage2_head[2](z2)
    z4 = model.stage2_head[3](z3)
    pred = model.stage2_head[4](z4)
    return pred, z4


def smart_head_with_features(model: SMART, q: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    z1 = model.mlp[0](q)
    z2 = model.mlp[1](z1)
    z3 = model.mlp[2](z2)
    z4 = model.mlp[3](z3)
    pred = model.mlp[4](z4)
    return pred, z4


def extract_cat_surface_features(model: CAT, surface_input_tokens: torch.Tensor, surface_query_tokens: torch.Tensor):
    surface_input_pos = surface_input_tokens[..., : model.spatial_dim]
    surface_query_pos = surface_query_tokens[..., : model.spatial_dim]
    geom_latents, anchor_pos, geom_final = model._encode_stage1(surface_input_pos)
    q = model._decode(surface_query_pos, geom_latents, anchor_pos, model.surface_decoder_blocks)
    surface_pred, z = cat_surface_head_with_features(model, q)
    anchor_pos_norm = anchor_pos / float(model.pos_scale_factor)
    return surface_pred, q, z, {
        "anchor_pos": anchor_pos,
        "anchor_pos_norm": anchor_pos_norm,
        "anchor_features": geom_final,
        "geom_latents": geom_latents,
        "geom_final": geom_final,
    }


def extract_smart_surface_features(model: SMART, geo_mesh: torch.Tensor, surface_query_tokens: torch.Tensor, params: torch.Tensor | None):
    geom_latents, anchor_pos, geom_final = model.encode(geo_mesh[..., : model.pos_encoder.spatial_dim], params, return_final=True)
    q = model.decode_features(geom_latents, anchor_pos, params, surface_query_tokens[..., : model.pos_encoder.spatial_dim])
    pred_all, z = smart_head_with_features(model, q)
    anchor_pos_norm = anchor_pos / float(model.pos_scale_factor)
    surface_pred = pred_all[..., : model.surface_channels]
    return surface_pred, q, z, {
        "anchor_pos": anchor_pos,
        "anchor_pos_norm": anchor_pos_norm,
        "anchor_features": geom_final,
        "geom_latents": geom_latents,
        "geom_final": geom_final,
    }


def kmeans_plus_plus(points: torch.Tensor, num_clusters: int, num_iters: int = 25) -> torch.Tensor:
    if points.ndim != 2:
        raise ValueError(f"Expected points to have shape [N, D], got {tuple(points.shape)}")
    if points.shape[0] < num_clusters:
        raise ValueError(f"Need at least {num_clusters} points for k-means, got {points.shape[0]}")

    pts = points.float()
    centroids = [pts[torch.randint(0, pts.shape[0], (1,), device=pts.device)].squeeze(0)]
    closest_dist2 = torch.sum((pts - centroids[0]) ** 2, dim=-1)

    for _ in range(1, num_clusters):
        probs = closest_dist2 / torch.clamp(closest_dist2.sum(), min=1e-12)
        idx = torch.multinomial(probs, num_samples=1).item()
        new_centroid = pts[idx]
        centroids.append(new_centroid)
        dist2 = torch.sum((pts - new_centroid) ** 2, dim=-1)
        closest_dist2 = torch.minimum(closest_dist2, dist2)

    centroids = torch.stack(centroids, dim=0)

    for _ in range(num_iters):
        distances = torch.cdist(pts, centroids)
        assignment = distances.argmin(dim=1)
        assignment_oh = F.one_hot(assignment, num_classes=num_clusters).to(dtype=pts.dtype)
        counts = assignment_oh.sum(dim=0)
        safe_counts = torch.clamp(counts, min=1.0)
        new_centroids = (assignment_oh.T @ pts) / safe_counts.unsqueeze(-1)

        empty = counts == 0
        if empty.any():
            min_dist = distances.gather(1, assignment.unsqueeze(1)).squeeze(1)
            refill_idx = torch.topk(min_dist, k=int(empty.sum().item()), largest=True).indices
            new_centroids[empty] = pts[refill_idx]

        if torch.allclose(new_centroids, centroids, atol=1e-4, rtol=0.0):
            centroids = new_centroids
            break
        centroids = new_centroids

    return centroids


def estimate_ledoit_wolf_inverse_covariance(x: torch.Tensor, pinv_rcond: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """Estimate a Ledoit-Wolf shrunk covariance inverse for centered samples.

    This mirrors the non-blocked shrinkage computation used by
    `sklearn.covariance.ledoit_wolf_shrinkage` for the standard identity-target
    Ledoit-Wolf estimator:

        shrunk_cov = (1 - shrinkage) * emp_cov + shrinkage * mu * I

    where `mu = trace(emp_cov) / n_features`.
    """
    x = x.double()
    n_samples, n_features = x.shape
    if n_samples <= 1:
        mean = x.mean(dim=0) if n_samples == 1 else torch.zeros((n_features,), dtype=torch.double, device=x.device)
        inv_cov = torch.eye(n_features, dtype=torch.double, device=x.device)
        return mean, inv_cov

    mean = x.mean(dim=0)
    xc = x - mean
    emp_cov = (xc.T @ xc) / float(n_samples)

    if n_features == 1:
        inv_cov = torch.linalg.pinv(emp_cov, rcond=pinv_rcond)
        return mean, inv_cov

    x2 = xc * xc
    emp_cov_trace = x2.sum(dim=0) / float(n_samples)
    mu = emp_cov_trace.sum() / float(n_features)

    # Exact non-blocked Ledoit-Wolf shrinkage statistics.
    beta_ = torch.sum(x2.T @ x2)
    delta_ = torch.sum((xc.T @ xc) ** 2) / float(n_samples ** 2)
    beta = (beta_ / float(n_samples) - delta_) / float(n_features * n_samples)
    delta = (delta_ - 2.0 * mu * emp_cov_trace.sum() + n_features * mu * mu) / float(n_features)

    beta = torch.clamp(beta, min=0.0)
    if float(delta.item()) <= 0.0:
        shrinkage = 0.0
    else:
        beta = torch.minimum(beta, delta)
        shrinkage = float((beta / delta).item()) if float(beta.item()) > 0.0 else 0.0

    cov = (1.0 - shrinkage) * emp_cov
    cov = cov + shrinkage * mu * torch.eye(n_features, dtype=torch.double, device=x.device)
    inv_cov = torch.linalg.pinv(cov, rcond=pinv_rcond)
    return mean, inv_cov


def compute_spatial_statistics(
    anchor_coords: torch.Tensor,
    anchor_features: torch.Tensor,
    num_regions: int,
    pinv_rcond: float,
    kmeans_iters: int = 25,
    eps: float = 1e-6,
):
    coords = anchor_coords.float()
    feats = anchor_features.float()

    coord_mean = coords.mean(dim=0)
    coord_std = torch.clamp(coords.std(dim=0, unbiased=False), min=eps)
    feat_mean = feats.mean(dim=0)
    feat_std = torch.clamp(feats.std(dim=0, unbiased=False), min=eps)

    coords_std = (coords - coord_mean) / coord_std
    feats_std = (feats - feat_mean) / feat_std
    joint = torch.cat([coords_std, feats_std], dim=-1)

    centroids = kmeans_plus_plus(coords, num_regions, num_iters=kmeans_iters)
    assignment = torch.cdist(coords, centroids).argmin(dim=1)

    region_means = []
    region_inv_covs = []
    region_counts = []
    for region_idx in range(num_regions):
        joint_region = joint[assignment == region_idx]
        region_counts.append(int(joint_region.shape[0]))
        if joint_region.shape[0] == 0:
            joint_region = joint
        mean_k, inv_cov_k = estimate_ledoit_wolf_inverse_covariance(joint_region, pinv_rcond)
        region_means.append(mean_k)
        region_inv_covs.append(inv_cov_k)

    return {
        "coord_mean": coord_mean,
        "coord_std": coord_std,
        "feature_mean": feat_mean,
        "feature_std": feat_std,
        "spatial_centroids": centroids,
        "joint_region_means": torch.stack(region_means, dim=0).float(),
        "joint_region_inv_covs": torch.stack(region_inv_covs, dim=0).float(),
        "region_counts": torch.tensor(region_counts, dtype=torch.long),
    }


def compute_knn_length_scale(
    anchor_pos: torch.Tensor,
    knn_k: int = 5,
    scale: float = 1.5,
    eps: float = 1e-6,
    query_chunk_size: int = 512,
) -> float:
    if anchor_pos.ndim == 3:
        anchor_pos = anchor_pos.reshape(-1, anchor_pos.shape[-1])
    elif anchor_pos.ndim != 2:
        raise ValueError(f"Expected anchor_pos to have shape [B, N, D] or [N, D], got {tuple(anchor_pos.shape)}")

    n_anchor = anchor_pos.shape[0]
    if n_anchor <= 1:
        return float(scale)

    k_eff = min(int(knn_k), n_anchor - 1)
    coords = anchor_pos.float()
    kth_distances = []
    chunk_size = max(1, int(query_chunk_size))
    for start in range(0, n_anchor, chunk_size):
        end = min(start + chunk_size, n_anchor)
        distances = torch.cdist(coords[start:end], coords)
        row = torch.arange(end - start, device=distances.device)
        col = torch.arange(start, end, device=distances.device)
        distances[row, col] = float("inf")
        kth_distances.append(torch.topk(distances, k=k_eff, largest=False).values[:, -1].cpu())
        del distances
    knn_dist = torch.cat(kth_distances, dim=0)
    median_dist = torch.median(knn_dist)
    return float(max(scale * float(median_dist.item()), eps))


def compute_anchor_joint_mahalanobis(
    anchor_pos_norm: torch.Tensor,
    anchor_features: torch.Tensor,
    coord_mean: torch.Tensor,
    coord_std: torch.Tensor,
    feature_mean: torch.Tensor,
    feature_std: torch.Tensor,
    spatial_centroids: torch.Tensor,
    joint_region_means: torch.Tensor,
    joint_region_inv_covs: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    bsz, n_anchor, _ = anchor_pos_norm.shape
    coords_flat = anchor_pos_norm.reshape(-1, anchor_pos_norm.shape[-1])
    feats_flat = anchor_features.reshape(-1, anchor_features.shape[-1])

    coord_stdzd = (coords_flat - coord_mean) / torch.clamp(coord_std, min=1e-6)
    feat_stdzd = (feats_flat - feature_mean) / torch.clamp(feature_std, min=1e-6)
    joint = torch.cat([coord_stdzd, feat_stdzd], dim=-1)

    region_idx = torch.cdist(coords_flat, spatial_centroids).argmin(dim=1)
    means = joint_region_means.index_select(0, region_idx)
    inv_covs = joint_region_inv_covs.index_select(0, region_idx)
    delta = joint - means
    md2 = torch.einsum("ni,nij,nj->n", delta, inv_covs, delta)
    md = torch.sqrt(torch.clamp(md2, min=1e-12))
    return md.view(bsz, n_anchor), region_idx.view(bsz, n_anchor)


def project_anchor_scores_to_queries(
    query_pos_norm: torch.Tensor,
    anchor_pos_norm: torch.Tensor,
    anchor_scores: torch.Tensor,
    length_scale: float,
    eps: float = 1e-12,
) -> torch.Tensor:
    distances = torch.cdist(query_pos_norm.float(), anchor_pos_norm.float())
    ls = max(float(length_scale), 1e-6)
    weights = torch.exp(-(distances * distances) / (2.0 * ls * ls))
    weighted = torch.einsum("bqa,ba->bq", weights, anchor_scores)
    normalizer = torch.clamp(weights.sum(dim=-1), min=eps)
    return weighted / normalizer


def compute_lll_variance_and_alpha(
    z: torch.Tensor,
    Sigma_LLL: torch.Tensor,
    V_skew: torch.Tensor,
    K: float,
    modulation: torch.Tensor | float,
):
    variance_lll = torch.sum((z @ Sigma_LLL) * z, dim=-1)
    variance_lll = torch.clamp(variance_lll, min=1e-12)
    variance_final = variance_lll * modulation
    cross_term = z @ V_skew
    denom = torch.sqrt(torch.clamp(variance_final * float(K) - cross_term * cross_term, min=1e-6))
    alpha_exact = cross_term / denom
    return variance_lll, variance_final, alpha_exact


def normalize_pos(pos: torch.Tensor, min_pos: torch.Tensor, max_pos: torch.Tensor) -> torch.Tensor:
    return (pos - min_pos) / torch.clamp(max_pos - min_pos, min=1e-12)


def denorm_fields(pred_norm: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return pred_norm * std + mean


def rel_l2(gt: np.ndarray, pred: np.ndarray, eps: float = 1e-12) -> float:
    return float(np.linalg.norm(pred - gt) / max(np.linalg.norm(gt), eps))


def compute_channel_rel(gt: np.ndarray, pred: np.ndarray, names: List[str]) -> Dict[str, float]:
    return {name: rel_l2(gt[:, idx], pred[:, idx]) for idx, name in enumerate(names)}


def load_full_preprocessed_case(run_dir: Path):
    surf_coords = np.load(run_dir / "surface_coords.npy").astype(np.float32, copy=False)
    surf_p = np.load(run_dir / "surface_pMeanTrim.npy").astype(np.float32, copy=False).reshape(-1, 1)
    surf_n = np.load(run_dir / "surface_normals.npy").astype(np.float32, copy=False)
    surf_wx = np.load(run_dir / "surface_wallShearStressMeanTrim_x.npy").astype(np.float32, copy=False).reshape(-1, 1)
    surf_wy = np.load(run_dir / "surface_wallShearStressMeanTrim_y.npy").astype(np.float32, copy=False).reshape(-1, 1)
    surf_wz = np.load(run_dir / "surface_wallShearStressMeanTrim_z.npy").astype(np.float32, copy=False).reshape(-1, 1)
    surf_gt = np.concatenate([surf_p, surf_n, surf_wx, surf_wy, surf_wz], axis=1)

    vol_coords = np.load(run_dir / "volume_coords.npy").astype(np.float32, copy=False)
    vol_p = np.load(run_dir / "volume_pMeanTrim.npy").astype(np.float32, copy=False).reshape(-1, 1)
    vol_u = np.load(run_dir / "volume_UMeanTrim.npy").astype(np.float32, copy=False)
    vol_gt = np.concatenate([vol_p, vol_u], axis=1)
    return surf_coords, surf_gt, vol_coords, vol_gt


def sample_input_idx(n: int, k: int, rng: np.random.Generator) -> np.ndarray:
    if k <= 0 or k >= n:
        return np.arange(n, dtype=np.int64)
    return rng.permutation(n).astype(np.int64, copy=False)[:k]
