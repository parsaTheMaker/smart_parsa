#!/usr/bin/env python3
"""Offline UQ calibration for CAT (Mahalanobis + skew-corrected last-layer Laplace).

This script does NOT alter checkpoints. It reads a trained CAT checkpoint, runs
on a subset of training batches, and saves UQ calibration tensors:
  - mu_train
  - inv_sigma_train
  - Sigma_LLL
  - V_skew
  - K
to: checkpoints/<checkpoint_stem>_uq_params.pt (or --output).
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from data.datasets import get_dataset
from models.smart.cat import CAT


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
    p = argparse.ArgumentParser(description="Calibrate CAT UQ parameters.")
    p.add_argument("--config-name", default="drivaerml_cat", help="Config file under smart/config (without .yaml).")
    p.add_argument("--checkpoint", required=True, help="Path to trained CAT checkpoint (stage-2 recommended).")
    p.add_argument("--output", default=None, help="Output .pt path. Default: checkpoints/<ckpt_stem>_uq_params.pt")
    p.add_argument("--max-batches", type=int, default=80, help="Number of train batches for calibration.")
    p.add_argument("--probe-batches", type=int, default=1, help="Probe batches used for finite-difference skew probe.")
    p.add_argument("--batch-size", type=int, default=None, help="Override dataloader batch size.")
    p.add_argument("--num-workers", type=int, default=0, help="Dataloader workers for calibration.")
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    p.add_argument("--device", default=None, help="cuda:0 / cpu. Default auto.")
    p.add_argument("--pinv-rcond", type=float, default=1e-6, help="rcond for pseudo-inverse.")
    p.add_argument("--cov-shrink", type=float, default=0.05, help="Fallback covariance shrinkage factor.")
    p.add_argument("--hessian-damping", type=float, default=1e-4, help="Diagonal damping for Hessian inversion.")
    p.add_argument("--eps", type=float, default=1e-3, help="Finite-difference step for skew probe.")
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


def sample_indices(n: int, k: int, device: torch.device, disjoint_from: torch.Tensor | None = None) -> torch.Tensor:
    if k <= 0:
        return torch.empty((0,), dtype=torch.long, device=device)
    if disjoint_from is not None:
        mask = torch.ones((n,), dtype=torch.bool, device=device)
        mask[disjoint_from] = False
        candidate = torch.where(mask)[0]
        if candidate.numel() == 0:
            return torch.randint(0, n, (k,), device=device)
        if k <= candidate.numel():
            perm = torch.randperm(candidate.numel(), device=device)[:k]
            return candidate[perm]
        extra = candidate[torch.randint(0, candidate.numel(), (k - candidate.numel(),), device=device)]
        return torch.cat([candidate, extra], dim=0)
    if k <= n:
        return torch.randperm(n, device=device)[:k]
    extra = torch.randint(0, n, (k - n,), device=device)
    return torch.cat([torch.arange(n, device=device), extra], dim=0)


def gather_per_batch(x: torch.Tensor, idx_list: List[torch.Tensor]) -> torch.Tensor:
    return torch.stack([x[b, idx_list[b], :] for b in range(x.shape[0])], dim=0)


def prepare_stage1_batch(batch, config, device: torch.device, surface_target_indices: List[int]):
    geo_mesh, surf_mesh, surf_data, vol_mesh, vol_data = batch
    del geo_mesh, vol_mesh, vol_data
    surf_mesh = surf_mesh.to(device)
    surf_data = surf_data.to(device)

    bsz, ns, _ = surf_mesh.shape
    s_in = int(getattr(config, "single_surface_input_points", getattr(config, "num_body_points", ns)))
    s_q = int(getattr(config, "single_surface_query_points", getattr(config, "num_surface_points", ns)))
    if s_in <= 0:
        s_in = ns
    if s_q <= 0:
        s_q = ns

    enc_idx = []
    surf_q_idx = []
    for _ in range(bsz):
        e = sample_indices(ns, s_in, device)
        sq = sample_indices(ns, s_q, device, disjoint_from=e)
        enc_idx.append(e)
        surf_q_idx.append(sq)

    s_idx = torch.tensor(surface_target_indices, dtype=torch.long, device=surf_data.device)
    surface_input_tokens = gather_per_batch(surf_mesh, enc_idx)
    surface_query_tokens = gather_per_batch(surf_mesh, surf_q_idx)
    surface_target = gather_per_batch(surf_data.index_select(dim=2, index=s_idx), surf_q_idx)

    return surface_input_tokens, surface_query_tokens, surface_target


def estimate_inv_covariance(q_global: torch.Tensor, pinv_rcond: float, cov_shrink: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (mu, inv_cov), both float64 on CPU."""
    q = q_global.double().cpu()
    mu = q.mean(dim=0)
    x = q - mu
    n, d = x.shape

    # Prefer Ledoit-Wolf if available.
    cov = None
    try:
        from sklearn.covariance import LedoitWolf  # type: ignore

        lw = LedoitWolf().fit(q.numpy())
        cov = torch.from_numpy(lw.covariance_).double()
    except Exception:
        if n > 1:
            cov = (x.T @ x) / float(n - 1)
        else:
            cov = torch.eye(d, dtype=torch.double)
        # Simple shrinkage fallback.
        tr = torch.trace(cov) / float(d)
        cov = (1.0 - cov_shrink) * cov + cov_shrink * tr * torch.eye(d, dtype=torch.double)

    inv_cov = torch.linalg.pinv(cov, rcond=pinv_rcond)
    return mu, inv_cov


def output_path_from_ckpt(ckpt_path: str, explicit_output: str | None) -> Path:
    if explicit_output:
        return Path(explicit_output)
    p = Path(ckpt_path)
    stem = p.stem
    for suffix in ("_best", "_last"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    return p.parent / f"{stem}_uq_params.pt"


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_config(args.config_name)

    if args.batch_size is not None:
        config.batch_size = int(args.batch_size)
    config.num_workers = int(args.num_workers)

    train_data, _test_data, _stats, spatial_dim, surf_channels, vol_channels, params_dim, fields = get_dataset(config)
    if params_dim > 0:
        raise NotImplementedError("Calibration currently supports params_dim=0 datasets.")

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

    dl_kwargs = dict(
        batch_size=int(config.batch_size),
        shuffle=True,
        num_workers=int(config.num_workers),
        pin_memory=bool(getattr(config, "pin_memory", True)),
    )
    if dl_kwargs["num_workers"] > 0:
        dl_kwargs["persistent_workers"] = True
        dl_kwargs["prefetch_factor"] = int(getattr(config, "prefetch_factor", 2))
    train_loader = torch.utils.data.DataLoader(train_data, **dl_kwargs)

    cache: Dict[str, torch.Tensor] = {}

    def hook_q(module, inputs, output):
        del module, output
        cache["q"] = inputs[0].detach()

    def hook_z(module, inputs, output):
        del module, inputs
        cache["z"] = output.detach()

    h_q = model.stage2_head[0].register_forward_hook(hook_q)
    h_z = model.stage2_head[3].register_forward_hook(hook_z)

    q_globals = []
    H = None
    probe_batches = []
    processed = 0

    try:
        for batch in train_loader:
            surface_input_tokens, surface_query_tokens, surface_target = prepare_stage1_batch(batch, config, device, s_idx)
            with torch.no_grad():
                _ = model.forward_stage1_only(surface_input_tokens, surface_query_tokens, return_aux=False)
            if "q" not in cache or "z" not in cache:
                raise RuntimeError("Failed to capture q/Z via hooks. Check CAT stage2_head structure.")

            q = cache["q"].double().cpu()       # [B, N, latent_dim]
            z = cache["z"].double().cpu()       # [B, N, 64]
            q_global = q.mean(dim=1)            # [B, latent_dim]
            q_globals.append(q_global)

            z2 = z.reshape(-1, z.shape[-1])     # [B*N, 64]
            cur_H = z2.T @ z2
            H = cur_H if H is None else (H + cur_H)

            if len(probe_batches) < int(args.probe_batches):
                probe_batches.append(
                    (
                        surface_input_tokens.detach().clone(),
                        surface_query_tokens.detach().clone(),
                        surface_target.detach().clone(),
                    )
                )

            processed += 1
            if processed >= int(args.max_batches):
                break
    finally:
        h_q.remove()
        h_z.remove()

    if processed == 0:
        raise RuntimeError("No batches processed for calibration.")
    if H is None:
        raise RuntimeError("Hessian accumulator is empty.")
    if len(probe_batches) == 0:
        raise RuntimeError("No probe batch captured.")

    q_all = torch.cat(q_globals, dim=0)  # [M, latent_dim]
    mu_train, inv_sigma_train = estimate_inv_covariance(q_all, args.pinv_rcond, args.cov_shrink)

    H = H.double()
    z_dim = H.shape[0]
    H_reg = H + float(args.hessian_damping) * torch.eye(z_dim, dtype=torch.double)
    Sigma_LLL = torch.linalg.pinv(H_reg, rcond=float(args.pinv_rcond))

    # Principal direction in penultimate feature space (64-dim).
    eigvals, eigvecs = torch.linalg.eigh(Sigma_LLL)
    principal = eigvecs[:, -1]
    principal = principal / torch.clamp(torch.linalg.norm(principal), min=1e-12)

    final_linear: torch.nn.Linear = model.stage2_head[-1]
    W_orig = final_linear.weight.detach().clone()
    dW = principal.to(W_orig.device, dtype=W_orig.dtype).unsqueeze(0).repeat(W_orig.shape[0], 1)
    dW = dW / torch.clamp(torch.linalg.norm(dW), min=1e-12)
    eps = float(args.eps)

    probe_in, probe_q, probe_tgt = probe_batches[0]

    def probe_loss(scale: float) -> float:
        with torch.no_grad():
            final_linear.weight.copy_(W_orig + (scale * eps) * dW)
            pred = model.forward_stage1_only(probe_in, probe_q, return_aux=False)
            # Use MSE for stable finite-difference probing.
            loss = F.mse_loss(pred, probe_tgt)
        return float(loss.item())

    with torch.no_grad():
        f_m2 = probe_loss(-2.0)
        f_m1 = probe_loss(-1.0)
        f_p1 = probe_loss(1.0)
        f_p2 = probe_loss(2.0)
        final_linear.weight.copy_(W_orig)

    # Central finite-difference approximation of 3rd directional derivative.
    d3 = (f_m2 - 2.0 * f_m1 + 2.0 * f_p1 - f_p2) / (2.0 * (eps ** 3))
    alpha_w = (d3 * principal).double().cpu()                    # [64]
    V_skew = (alpha_w @ Sigma_LLL).double().cpu()                # [64]
    K = float(1.0 + (alpha_w @ Sigma_LLL @ alpha_w).item())      # scalar

    out_path = output_path_from_ckpt(args.checkpoint, args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "mu_train": mu_train.float(),
        "inv_sigma_train": inv_sigma_train.float(),
        "Sigma_LLL": Sigma_LLL.float(),
        "alpha_w": alpha_w.float(),
        "V_skew": V_skew.float(),
        "K": torch.tensor(K, dtype=torch.float32),
        "surface_target_fields": s_fields,
        "calibration_meta": {
            "checkpoint": str(args.checkpoint),
            "config_name": args.config_name,
            "processed_batches": int(processed),
            "max_batches": int(args.max_batches),
            "probe_batches": int(args.probe_batches),
            "eps": float(args.eps),
            "cov_shrink": float(args.cov_shrink),
            "hessian_damping": float(args.hessian_damping),
            "pinv_rcond": float(args.pinv_rcond),
            "q_dim": int(mu_train.numel()),
            "z_dim": int(Sigma_LLL.shape[0]),
            "skew_directional_d3": float(d3),
            "K": float(K),
        },
    }
    torch.save(payload, out_path)

    print(f"Saved UQ calibration params to: {out_path}")
    print(json.dumps(payload["calibration_meta"], indent=2))


if __name__ == "__main__":
    main()
