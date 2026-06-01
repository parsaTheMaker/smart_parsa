#!/usr/bin/env python3
"""CAT inference with hybrid OOD + UQ (MD + skew-corrected LLL).

New-file implementation that does not modify existing training/inference files.
Outputs:
  - deterministic metrics (surface/volume rel_l2 + per-channel)
  - per-run arrays with mu_pred, variance_final, alpha_exact, MD
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm.auto import tqdm

from data.datasets import get_dataset
from models.smart.cat import CAT
from utils.utils import get_model_checkpoint_name


SURFACE_PREFERRED = [
    "pressure",
    "normal_x",
    "normal_y",
    "normal_z",
    "wall_shear_x",
    "wall_shear_y",
    "wall_shear_z",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CAT inference with OOD/UQ.")
    p.add_argument("--config-name", default="drivaerml_cat", help="Config file under smart/config (without .yaml).")
    p.add_argument("--checkpoint", required=True, help="CAT checkpoint path.")
    p.add_argument("--uq-params", required=True, help="Path produced by calibrate_uq.py.")
    p.add_argument("--stage", type=int, default=2, choices=[1, 2], help="Inference mode.")
    p.add_argument("--split", default="test", choices=["train", "test"])
    p.add_argument("--output-dir", default=None, help="Output dir for arrays + metrics json.")
    p.add_argument("--gamma", type=float, default=0.1, help="Hybrid variance inflation weight for MD.")
    p.add_argument("--max-runs", type=int, default=-1, help="Limit number of runs (-1 for all).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None, help="cuda:0 / cpu; default auto.")
    p.add_argument("--cat-vol-chunk", type=int, default=131072, help="Volume chunk size for stage-2 inference.")
    return p.parse_args()


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


def resolve_surface_targets(fields: Dict[str, List[str]]) -> Tuple[List[int], List[str]]:
    surface_fields = list(fields.get("surface", []))
    idx = [surface_fields.index(name) for name in SURFACE_PREFERRED if name in surface_fields]
    if not idx:
        idx = list(range(len(surface_fields)))
    return idx, [surface_fields[i] for i in idx]


def normalize_pos(pos: torch.Tensor, min_pos: torch.Tensor, max_pos: torch.Tensor) -> torch.Tensor:
    return (pos - min_pos) / torch.clamp(max_pos - min_pos, min=1e-12)


def denorm_fields(pred_norm: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return pred_norm * std + std.new_tensor(0.0) + mean


def rel_l2(gt: np.ndarray, pred: np.ndarray, eps: float = 1e-12) -> float:
    return float(np.linalg.norm(pred - gt) / max(np.linalg.norm(gt), eps))


def compute_channel_rel(gt: np.ndarray, pred: np.ndarray, names: List[str]) -> Dict[str, float]:
    out = {}
    for i, n in enumerate(names):
        out[n] = rel_l2(gt[:, i], pred[:, i])
    return out


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
    # Match training fast behavior (with replacement).
    return rng.integers(0, n, size=k, dtype=np.int64)


def stage2_volume_predict_cached(
    model: CAT,
    surf_input_b: torch.Tensor,
    surf_query_b: torch.Tensor,
    vol_query_b: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    """Compute stage2 volume prediction with one stage1 pass + chunked volume decode."""
    surface_pred, aux_s1 = model.forward_stage1_only(surf_input_b, surf_query_b, return_aux=True)
    geom_latents = aux_s1["geom_latents"]
    anchor_pos = aux_s1["anchor_pos"]
    geom_final = aux_s1["geom_final"]

    prev_latents, _ = model._encode_stage2(surf_query_b[..., : model.spatial_dim], surface_pred, anchor_pos, initial_latent=geom_final)
    new_latents, _ = model._encode_stage2(surf_query_b[..., : model.spatial_dim], surface_pred, anchor_pos, initial_latent=None)
    w_couple, w_fuse = model._compute_dynamic_skip_weights(geom_latents, prev_latents, new_latents, surface_pred)

    fused_latents = []
    for m in range(model.loops):
        wc = w_couple[:, m, :].unsqueeze(-1)
        wf = w_fuse[:, m, :].unsqueeze(-1)
        geom_m = model.surface_to_volume_latent_norm(geom_latents[m])
        prev_m = model.surface_to_volume_latent_norm(prev_latents[m])
        new_m = model.surface_to_volume_latent_norm(new_latents[m])
        coupled = prev_m + wc * (geom_m - prev_m)
        fused = new_m + wf * (coupled - new_m)
        fused_latents.append(fused)

    preds = []
    n_vol = vol_query_b.shape[1]
    for start in range(0, n_vol, chunk_size):
        q = vol_query_b[:, start : start + chunk_size, :]
        qv = model._decode(q[..., : model.spatial_dim], fused_latents, anchor_pos, model.volume_decoder_blocks)
        preds.append(model.volume_head(qv))
    return torch.cat(preds, dim=1)


def choose_output_dir(args, config) -> Path:
    if args.output_dir:
        return Path(args.output_dir)
    stem = Path(args.checkpoint).stem
    return Path("results") / "uq_inference" / config.dataset / stem


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_config(args.config_name)

    train_data, test_data, _stats, spatial_dim, surf_channels, vol_channels, params_dim, fields = get_dataset(config)
    if params_dim > 0:
        raise NotImplementedError("This script supports params_dim=0 datasets only.")
    if not getattr(train_data, "preprocessed_mode", False):
        raise RuntimeError("Expected preprocessed DrivAerML mode for full-query inference.")

    dataset = test_data if args.split == "test" else train_data
    run_ids = list(dataset.data)
    if args.max_runs > 0:
        run_ids = run_ids[: int(args.max_runs)]

    s_idx, s_fields = resolve_surface_targets(fields)
    arch = OmegaConf.to_container(config.architecture, resolve=True)
    arch["stage2_surface_channels"] = len(s_idx)

    model = CAT(
        spatial_dim=spatial_dim,
        surface_channels=surf_channels,
        volume_channels=vol_channels,
        parameter_channels=params_dim,
        **arch,
    ).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
    model.eval()

    uq = torch.load(args.uq_params, map_location="cpu")
    mu_train = uq["mu_train"].float()
    inv_sigma_train = uq["inv_sigma_train"].float()
    Sigma_LLL = uq["Sigma_LLL"].float()
    V_skew = uq["V_skew"].float()
    K = float(uq["K"].item() if isinstance(uq["K"], torch.Tensor) else uq["K"])

    # Hooks for q, Z, and stage1 surface prediction (stage2_head output).
    cache: Dict[str, torch.Tensor] = {}

    def hook_q(module, inputs, output):
        del module, output
        cache["q"] = inputs[0].detach()

    def hook_z(module, inputs, output):
        del module, inputs
        cache["z"] = output.detach()

    def hook_surface_pred(module, inputs, output):
        del module, inputs
        cache["surface_pred"] = output.detach()

    h_q = model.stage2_head[0].register_forward_hook(hook_q)
    h_z = model.stage2_head[3].register_forward_hook(hook_z)
    h_s = model.stage2_head[4].register_forward_hook(hook_surface_pred)

    out_dir = choose_output_dir(args, config)
    out_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = out_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    mean_s = dataset.mean_surf_data[s_idx].float()
    std_s = torch.clamp(dataset.std_surf_data[s_idx].float(), min=1e-12)
    mean_v = dataset.mean_vol_data.float()
    std_v = torch.clamp(dataset.std_vol_data.float(), min=1e-12)
    min_pos = dataset.min_pos.float()
    max_pos = dataset.max_pos.float()

    # Move UQ tensors to device for fast batch algebra.
    mu_train_d = mu_train.to(device)
    inv_sigma_train_d = inv_sigma_train.to(device)
    Sigma_LLL_d = Sigma_LLL.to(device)
    V_skew_d = V_skew.to(device)

    metrics = {
        "runs_processed": 0,
        "surface_rel_l2": [],
        "volume_rel_l2": [],
        "surface_channel_rel_l2": {f: [] for f in s_fields},
        "volume_channel_rel_l2": {f: [] for f in fields["volume"]},
        "md": [],
    }

    try:
        with torch.inference_mode():
            for run_id in tqdm(run_ids, desc=f"UQ inference ({args.split})", dynamic_ncols=True):
                run_dir = Path(config.data_path) / f"run_{run_id}"
                surf_coords, surf_gt_all, vol_coords, vol_gt_all = load_full_preprocessed_case(run_dir)
                surf_gt = surf_gt_all[:, s_idx]
                vol_gt = vol_gt_all

                surf_coords_t = torch.from_numpy(surf_coords)
                vol_coords_t = torch.from_numpy(vol_coords)
                surf_query = normalize_pos(surf_coords_t, min_pos, max_pos)
                vol_query = normalize_pos(vol_coords_t, min_pos, max_pos)

                ns = surf_coords.shape[0]
                rng = np.random.default_rng(args.seed + int(run_id))
                s_in = int(getattr(config, "single_surface_input_points", getattr(config, "num_body_points", ns)))
                in_idx = sample_input_idx(ns, s_in, rng)
                surf_input = surf_query[torch.from_numpy(in_idx)]

                # Clear hooks cache
                cache.clear()

                surf_in_b = surf_input.unsqueeze(0).to(device)
                surf_q_b = surf_query.unsqueeze(0).to(device)

                if int(args.stage) == 1:
                    _ = model.forward_stage1_only(surf_in_b, surf_q_b, return_aux=False)
                    vol_pred_phys = None
                else:
                    vol_q_b = vol_query.unsqueeze(0).to(device)
                    vol_pred_norm = stage2_volume_predict_cached(
                        model,
                        surf_in_b,
                        surf_q_b,
                        vol_q_b,
                        chunk_size=int(args.cat_vol_chunk),
                    )[0]
                    vol_pred_phys = denorm_fields(vol_pred_norm.cpu(), mean_v, std_v).numpy()

                if "surface_pred" not in cache or "q" not in cache or "z" not in cache:
                    raise RuntimeError("Failed to capture surface_pred/q/Z via hooks.")

                surf_pred_norm = cache["surface_pred"][0]               # [Nsurf, Csurf_target]
                q = cache["q"]                                          # [1, Nsurf, latent_dim]
                z = cache["z"]                                          # [1, Nsurf, 64]

                # OOD score (Mahalanobis) from pooled q.
                q_global = q.mean(dim=1)                                # [1, latent_dim]
                dq = q_global - mu_train_d.unsqueeze(0)
                md2 = torch.einsum("bi,ij,bj->b", dq, inv_sigma_train_d, dq)
                md = torch.sqrt(torch.clamp(md2, min=1e-12))            # [1]

                # LLL variance per surface point.
                # variance_LLL = sum((Z @ Sigma_LLL) * Z, dim=-1)
                z2 = z[0]                                               # [Nsurf, 64]
                variance_lll = torch.sum((z2 @ Sigma_LLL_d) * z2, dim=-1)
                variance_lll = torch.clamp(variance_lll, min=1e-12)

                variance_final = variance_lll * (1.0 + float(args.gamma) * md[0])

                cross_term = z2 @ V_skew_d                               # [Nsurf]
                denom = torch.sqrt(torch.clamp(variance_final * K - cross_term * cross_term, min=1e-6))
                alpha_exact = cross_term / denom                         # [Nsurf]

                surf_pred_phys = denorm_fields(surf_pred_norm.cpu(), mean_s, std_s).numpy()
                surf_var_np = variance_final.detach().cpu().numpy()
                surf_alpha_np = alpha_exact.detach().cpu().numpy()
                md_scalar = float(md[0].item())

                surface_rel = rel_l2(surf_gt.reshape(-1), surf_pred_phys.reshape(-1))
                surface_ch = compute_channel_rel(surf_gt, surf_pred_phys, s_fields)

                metrics["surface_rel_l2"].append(surface_rel)
                for k, v in surface_ch.items():
                    metrics["surface_channel_rel_l2"][k].append(v)
                metrics["md"].append(md_scalar)

                if vol_pred_phys is not None:
                    volume_rel = rel_l2(vol_gt.reshape(-1), vol_pred_phys.reshape(-1))
                    volume_ch = compute_channel_rel(vol_gt, vol_pred_phys, fields["volume"])
                    metrics["volume_rel_l2"].append(volume_rel)
                    for k, v in volume_ch.items():
                        metrics["volume_channel_rel_l2"][k].append(v)

                out_npz = runs_dir / f"run_{run_id}_uq.npz"
                np.savez_compressed(
                    out_npz,
                    run_id=np.array([run_id], dtype=np.int64),
                    surface_coords=surf_coords,
                    surface_gt=surf_gt,
                    surface_mu_pred=surf_pred_phys,
                    surface_variance_final=surf_var_np,
                    surface_alpha_exact=surf_alpha_np,
                    md=np.array([md_scalar], dtype=np.float32),
                    volume_coords=vol_coords,
                    volume_gt=vol_gt,
                    volume_mu_pred=np.zeros((0, 0), dtype=np.float32) if vol_pred_phys is None else vol_pred_phys,
                )
                metrics["runs_processed"] += 1
    finally:
        h_q.remove()
        h_z.remove()
        h_s.remove()

    def mean_or_nan(vals):
        return float(np.mean(vals)) if len(vals) > 0 else float("nan")

    summary = {
        "config_name": args.config_name,
        "checkpoint": args.checkpoint,
        "uq_params": args.uq_params,
        "stage": int(args.stage),
        "split": args.split,
        "gamma": float(args.gamma),
        "runs_processed": int(metrics["runs_processed"]),
        "surface_rel_l2_mean": mean_or_nan(metrics["surface_rel_l2"]),
        "volume_rel_l2_mean": mean_or_nan(metrics["volume_rel_l2"]),
        "md_mean": mean_or_nan(metrics["md"]),
        "surface_channel_rel_l2_mean": {k: mean_or_nan(v) for k, v in metrics["surface_channel_rel_l2"].items()},
        "volume_channel_rel_l2_mean": {k: mean_or_nan(v) for k, v in metrics["volume_channel_rel_l2"].items()},
    }
    (out_dir / "uq_metrics.json").write_text(json.dumps(summary, indent=2))

    print(f"Saved run arrays to: {runs_dir}")
    print(f"Saved summary metrics to: {out_dir / 'uq_metrics.json'}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
