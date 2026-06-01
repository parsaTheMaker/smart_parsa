#!/usr/bin/env python3
"""Full-query DrivAerML comparison for CAT (stage1+stage2) vs SMART.

What this script does for one run_id:
1) Uses preprocessed DrivAerML files as data source.
2) Keeps encoder input sizes as configured in training configs.
3) Queries ALL coordinates in the selected run for predictions.
4) Computes pointwise errors and aggregate metrics.
5) Exports ParaView-friendly VTK point-cloud files.
6) Saves comprehensive plots + summary text + metrics JSON.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf


SCRIPT_DIR = Path(__file__).resolve().parent
SMART_ROOT = SCRIPT_DIR.parent
if str(SMART_ROOT) not in sys.path:
    sys.path.insert(0, str(SMART_ROOT))

from data.ahmedml_dataset_v2 import AhmedMLDatasetV2
from models.smart.cat import CAT
from models.smart.smart import SMART
from utils.utils import get_model_checkpoint_name


SURFACE_FIELDS = [
    "pressure",
    "normal_x",
    "normal_y",
    "normal_z",
    "wall_shear_x",
    "wall_shear_y",
    "wall_shear_z",
]
VOLUME_FIELDS = ["pressure", "velocity_x", "velocity_y", "velocity_z"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare CAT(stage1+stage2) vs SMART on full run coordinates.")
    p.add_argument("--run-id", type=int, default=None, help="Representative run id for detailed plots/vtk. Default: first test run.")
    p.add_argument("--all-test-metrics", action="store_true", help="Compute summary metrics over full test subset.")
    p.add_argument("--smart-config", default="drivaerml", help="Config file name under smart/config (without .yaml)")
    p.add_argument("--cat-config", default="drivaerml_cat", help="Config file name under smart/config (without .yaml)")
    p.add_argument("--smart-checkpoint", default=None, help="Path to SMART checkpoint (_best.pt/_last.pt).")
    p.add_argument("--cat-stage1-checkpoint", required=True, help="Path to CAT stage1 checkpoint.")
    p.add_argument("--cat-stage2-checkpoint", required=True, help="Path to CAT stage2 checkpoint.")
    p.add_argument("--output-dir", default=None, help="Output directory root.")
    p.add_argument("--device", default=None, help="Torch device, e.g. cuda:0 or cpu.")
    p.add_argument("--seed", type=int, default=42, help="Sampling seed for encoder inputs.")
    p.add_argument("--plot-max-points", type=int, default=90000, help="Max points per domain for plotting only.")
    p.add_argument("--cat-vol-chunk", type=int, default=131072, help="Stage2 volume query chunk size.")
    p.add_argument("--uq-params", default=None, help="Optional UQ params .pt from calibrate_uq.py to save uncertainty fields.")
    p.add_argument("--gamma", type=float, default=0.1, help="UQ variance inflation factor for Mahalanobis scalar.")
    return p.parse_args()


def load_cfg(name: str):
    path = SMART_ROOT / "config" / f"{name}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"Config not found: {path}")
    return OmegaConf.load(path).experiment


def resolve_device(device_arg: str | None) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def to_np(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


def normalize_pos(pos: torch.Tensor, min_pos: torch.Tensor, max_pos: torch.Tensor) -> torch.Tensor:
    return (pos - min_pos) / torch.clamp(max_pos - min_pos, min=1e-12)


def denorm_fields(pred_norm: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return pred_norm * std + mean


def sample_input_idx(n: int, k: int, rng: np.random.Generator) -> np.ndarray:
    if k <= 0 or k >= n:
        return np.arange(n, dtype=np.int64)
    # Match training default fast sampling behavior: with replacement.
    return rng.integers(0, n, size=k, dtype=np.int64)


def safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", name)


def write_polydata_vtk(path: Path, points_xyz: np.ndarray, point_data: Dict[str, np.ndarray]) -> None:
    """Write legacy VTK binary PolyData with one vertex per point."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pts = np.asarray(points_xyz, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("points_xyz must have shape [N, 3]")
    n = pts.shape[0]
    connectivity = np.empty((n, 2), dtype=">i4")
    connectivity[:, 0] = 1
    connectivity[:, 1] = np.arange(n, dtype=np.int32)

    with open(path, "wb") as f:
        f.write(b"# vtk DataFile Version 3.0\n")
        f.write(b"DrivAerML prediction comparison\n")
        f.write(b"BINARY\n")
        f.write(b"DATASET POLYDATA\n")
        f.write(f"POINTS {n} float\n".encode("ascii"))
        f.write(pts.astype(">f4", copy=False).tobytes())
        f.write(b"\n")

        f.write(f"VERTICES {n} {2*n}\n".encode("ascii"))
        f.write(connectivity.tobytes())
        f.write(b"\n")

        f.write(f"POINT_DATA {n}\n".encode("ascii"))
        for name, arr in point_data.items():
            a = np.asarray(arr, dtype=np.float32)
            if a.ndim == 1:
                a = a.reshape(-1, 1)
            if a.shape[0] != n:
                raise ValueError(f"Point-data '{name}' has {a.shape[0]} rows, expected {n}")
            nm = safe_name(name)
            comps = a.shape[1]
            if comps == 3:
                f.write(f"VECTORS {nm} float\n".encode("ascii"))
                f.write(a.astype(">f4", copy=False).tobytes())
                f.write(b"\n")
            else:
                f.write(f"SCALARS {nm} float {comps}\n".encode("ascii"))
                f.write(b"LOOKUP_TABLE default\n")
                f.write(a.astype(">f4", copy=False).tobytes())
                f.write(b"\n")


def rel_l2(gt: np.ndarray, pred: np.ndarray, eps: float = 1e-12) -> float:
    num = float(np.linalg.norm(pred - gt))
    den = float(np.linalg.norm(gt))
    return num / max(den, eps)


def compute_field_metrics(gt: np.ndarray, pred: np.ndarray, field_names: List[str]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for i, name in enumerate(field_names):
        g = gt[:, i]
        p = pred[:, i]
        e = p - g
        ae = np.abs(e)
        ss_res = float(np.sum(e**2))
        ss_tot = float(np.sum((g - g.mean()) ** 2))
        r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 1e-12 else float("nan")
        out[name] = {
            "mae": float(ae.mean()),
            "rmse": float(np.sqrt(np.mean(e**2))),
            "rel_l2": rel_l2(g, p),
            "r2": r2,
            "median_abs_err": float(np.median(ae)),
            "p95_abs_err": float(np.percentile(ae, 95)),
            "max_abs_err": float(ae.max()),
        }
    return out


def compute_global_metrics(gt: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    e = pred - gt
    ae = np.abs(e)
    ss_res = float(np.sum(e**2))
    gflat = gt.reshape(-1)
    ss_tot = float(np.sum((gflat - gflat.mean()) ** 2))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 1e-12 else float("nan")
    return {
        "mae": float(ae.mean()),
        "rmse": float(np.sqrt(np.mean(e**2))),
        "rel_l2": rel_l2(gt.reshape(-1), pred.reshape(-1)),
        "r2": r2,
        "median_abs_err": float(np.median(ae)),
        "p95_abs_err": float(np.percentile(ae, 95)),
        "max_abs_err": float(ae.max()),
    }


class RunningFieldStats:
    def __init__(self, field_names: List[str]):
        self.names = field_names
        m = len(field_names)
        self.count = np.zeros((m,), dtype=np.float64)
        self.sum_abs = np.zeros((m,), dtype=np.float64)
        self.sum_sq_err = np.zeros((m,), dtype=np.float64)
        self.sum_sq_gt = np.zeros((m,), dtype=np.float64)
        self.sum_gt = np.zeros((m,), dtype=np.float64)

    def update(self, gt: np.ndarray, pred: np.ndarray):
        err = pred - gt
        self.count += gt.shape[0]
        self.sum_abs += np.sum(np.abs(err), axis=0)
        self.sum_sq_err += np.sum(err**2, axis=0)
        self.sum_sq_gt += np.sum(gt**2, axis=0)
        self.sum_gt += np.sum(gt, axis=0)

    def summary(self):
        out = {}
        total_sse = float(np.sum(self.sum_sq_err))
        total_sst = float(np.sum(self.sum_sq_gt - (self.sum_gt**2) / np.maximum(self.count, 1.0)))
        for i, n in enumerate(self.names):
            c = max(self.count[i], 1.0)
            sse = float(self.sum_sq_err[i])
            sst = float(self.sum_sq_gt[i] - (self.sum_gt[i] ** 2) / c)
            rel = float(np.sqrt(sse) / max(np.sqrt(self.sum_sq_gt[i]), 1e-12))
            out[n] = {
                "mae": float(self.sum_abs[i] / c),
                "rmse": float(np.sqrt(sse / c)),
                "rel_l2": rel,
                "r2": (1.0 - sse / sst) if sst > 1e-12 else float("nan"),
            }
        total_count = float(np.sum(self.count))
        return {
            "fields": out,
            "global": {
                "mae": float(np.sum(self.sum_abs) / max(total_count * len(self.names), 1.0)),
                "rmse": float(np.sqrt(total_sse / max(total_count * len(self.names), 1.0))),
                "rel_l2": float(np.sqrt(total_sse) / max(np.sqrt(np.sum(self.sum_sq_gt)), 1e-12)),
                "r2": (1.0 - total_sse / total_sst) if total_sst > 1e-12 else float("nan"),
            },
        }


def choose_ckpt(config, explicit: str | None) -> str:
    if explicit:
        return explicit
    stem = get_model_checkpoint_name(config)
    best = SMART_ROOT.parent / "checkpoints" / f"{stem}_best.pt"
    last = SMART_ROOT.parent / "checkpoints" / f"{stem}_last.pt"
    if best.is_file():
        return str(best)
    if last.is_file():
        return str(last)
    raise FileNotFoundError(f"SMART checkpoint not found: {best} or {last}")


def cat_stage1_predict_full(
    model: CAT,
    surf_input_norm: torch.Tensor,
    surf_query_norm: torch.Tensor,
    chunk_size: int,
    device: torch.device,
) -> torch.Tensor:
    preds = []
    b_input = surf_input_norm.unsqueeze(0).to(device)
    n = surf_query_norm.shape[0]
    for i in range(0, n, chunk_size):
        q = surf_query_norm[i : i + chunk_size].unsqueeze(0).to(device)
        y = model.forward_stage1_only(b_input, q, return_aux=False)
        preds.append(y[0].detach().cpu())
    return torch.cat(preds, dim=0)


def cat_stage2_predict_full_cached(
    model: CAT,
    surf_input_norm: torch.Tensor,
    surf_query_norm: torch.Tensor,
    vol_query_norm: torch.Tensor,
    chunk_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Equivalent to forward_stage2_only, but caches surface-side work once."""
    b_input = surf_input_norm.unsqueeze(0).to(device)
    b_surf_q = surf_query_norm.unsqueeze(0).to(device)

    with torch.no_grad():
        surface_pred, aux_s1 = model.forward_stage1_only(b_input, b_surf_q, return_aux=True)
        geom_latents = aux_s1["geom_latents"]
        anchor_pos = aux_s1["anchor_pos"]
        geom_final = aux_s1["geom_final"]

        prev_latents, _ = model._encode_stage2(b_surf_q[..., : model.spatial_dim], surface_pred, anchor_pos, initial_latent=geom_final)
        new_latents, _ = model._encode_stage2(b_surf_q[..., : model.spatial_dim], surface_pred, anchor_pos, initial_latent=None)
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
        n = vol_query_norm.shape[0]
        for i in range(0, n, chunk_size):
            q = vol_query_norm[i : i + chunk_size].unsqueeze(0).to(device)
            qv = model._decode(q[..., : model.spatial_dim], fused_latents, anchor_pos, model.volume_decoder_blocks)
            y = model.volume_head(qv)
            preds.append(y[0].detach().cpu())
    return torch.cat(preds, dim=0)


def setup_plot_style():
    plt.rcParams.update(
        {
            "font.size": 13,
            "axes.titlesize": 15,
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
            "figure.titlesize": 18,
            "axes.grid": True,
            "grid.alpha": 0.2,
            "grid.linestyle": "--",
        }
    )


def downsample_for_plot(coords: np.ndarray, arrays: List[np.ndarray], max_points: int, seed: int):
    n = coords.shape[0]
    if n <= max_points:
        return coords, arrays
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=max_points, replace=False)
    return coords[idx], [a[idx] for a in arrays]


def plot_field_maps(
    out_path: Path,
    title: str,
    coords: np.ndarray,
    gt: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    field_names: List[str],
    model_a_name: str,
    model_b_name: str,
):
    n_fields = len(field_names)
    fig, axes = plt.subplots(n_fields, 5, figsize=(26, 4.4 * n_fields), constrained_layout=True)
    if n_fields == 1:
        axes = np.expand_dims(axes, axis=0)

    xz = coords[:, [0, 2]]
    fig.suptitle(title)
    for i, field in enumerate(field_names):
        g = gt[:, i]
        a = pred_a[:, i]
        b = pred_b[:, i]
        ea = np.abs(a - g)
        eb = np.abs(b - g)
        vmin = float(np.percentile(np.concatenate([g, a, b]), 1))
        vmax = float(np.percentile(np.concatenate([g, a, b]), 99))
        evmax = float(max(np.percentile(ea, 99), np.percentile(eb, 99)))
        evmax = evmax if evmax > 0 else 1e-12
        panels = [
            (g, "GT", "coolwarm", vmin, vmax),
            (a, model_a_name, "coolwarm", vmin, vmax),
            (b, model_b_name, "coolwarm", vmin, vmax),
            (ea, f"|{model_a_name}-GT|", "magma", 0.0, evmax),
            (eb, f"|{model_b_name}-GT|", "magma", 0.0, evmax),
        ]
        for j, (vals, subtitle, cmap, pvmin, pvmax) in enumerate(panels):
            ax = axes[i, j]
            sc = ax.scatter(xz[:, 0], xz[:, 1], c=vals, s=3, cmap=cmap, vmin=pvmin, vmax=pvmax, linewidths=0, rasterized=True)
            ax.set_title(f"{field} - {subtitle}")
            ax.set_xlabel("x")
            ax.set_ylabel("z")
            cbar = fig.colorbar(sc, ax=ax, shrink=0.8)
            cbar.ax.tick_params(labelsize=10)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_parity(
    out_path: Path,
    gt: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    field_names: List[str],
    model_a_name: str,
    model_b_name: str,
):
    n = len(field_names)
    fig, axes = plt.subplots(n, 2, figsize=(14, 4.0 * n), constrained_layout=True)
    if n == 1:
        axes = np.expand_dims(axes, axis=0)
    fig.suptitle("Parity Plots")
    for i, field in enumerate(field_names):
        g = gt[:, i]
        for j, (p, name) in enumerate([(pred_a[:, i], model_a_name), (pred_b[:, i], model_b_name)]):
            ax = axes[i, j]
            lo = float(np.percentile(np.concatenate([g, p]), 1))
            hi = float(np.percentile(np.concatenate([g, p]), 99))
            ax.scatter(g, p, s=3, alpha=0.35, rasterized=True)
            ax.plot([lo, hi], [lo, hi], "k--", linewidth=1.2)
            ax.set_title(f"{field}: {name}")
            ax.set_xlabel("GT")
            ax.set_ylabel("Prediction")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_error_hist(
    out_path: Path,
    gt: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    field_names: List[str],
    model_a_name: str,
    model_b_name: str,
):
    n = len(field_names)
    fig, axes = plt.subplots(n, 1, figsize=(12, 3.2 * n), constrained_layout=True)
    if n == 1:
        axes = [axes]
    fig.suptitle("Absolute Error Distributions")
    for i, field in enumerate(field_names):
        ea = np.abs(pred_a[:, i] - gt[:, i])
        eb = np.abs(pred_b[:, i] - gt[:, i])
        ax = axes[i]
        ax.hist(ea, bins=80, alpha=0.65, label=model_a_name, density=True)
        ax.hist(eb, bins=80, alpha=0.65, label=model_b_name, density=True)
        ax.set_title(field)
        ax.set_xlabel("Absolute Error")
        ax.set_ylabel("Density")
        ax.legend()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_grouped_metrics(
    out_path: Path,
    metric_name: str,
    metrics_a: Dict[str, Dict[str, float]],
    metrics_b: Dict[str, Dict[str, float]],
    field_names: List[str],
    model_a_name: str,
    model_b_name: str,
):
    x = np.arange(len(field_names))
    va = [metrics_a[f][metric_name] for f in field_names]
    vb = [metrics_b[f][metric_name] for f in field_names]
    width = 0.38
    fig, ax = plt.subplots(figsize=(max(10, len(field_names) * 1.35), 6), constrained_layout=True)
    ax.bar(x - width / 2, va, width, label=model_a_name)
    ax.bar(x + width / 2, vb, width, label=model_b_name)
    ax.set_xticks(x)
    ax.set_xticklabels(field_names, rotation=35, ha="right")
    ax.set_ylabel(metric_name.upper())
    ax.set_title(f"{metric_name.upper()} by Field")
    ax.legend()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main():
    args = parse_args()
    setup_plot_style()

    smart_cfg = load_cfg(args.smart_config)
    cat_cfg = load_cfg(args.cat_config)
    device = resolve_device(args.device)
    print(f"Device: {device}")

    # Dataset/stats from preprocessed-only DrivAerML.
    dataset = AhmedMLDatasetV2(
        saved_folder=str(smart_cfg.data_path),
        if_test=False,
        geometry_points=int(smart_cfg.num_body_points),
        surface_points=int(smart_cfg.num_surface_points),
        volume_points=int(smart_cfg.num_volume_points),
        scale_positions=bool(smart_cfg.scale_positions),
        require_preprocessed=True,
    )

    # Use split ids from dataset for statistically reliable summary metrics.
    eval_run_ids = list(dataset.test_ids) if args.all_test_metrics else [int(args.run_id if args.run_id is not None else dataset.test_ids[0])]
    rep_run_id = int(args.run_id if args.run_id is not None else eval_run_ids[0])

    mean_s = dataset.mean_surf_data
    std_s = torch.clamp(dataset.std_surf_data, min=1e-12)
    mean_v = dataset.mean_vol_data
    std_v = torch.clamp(dataset.std_vol_data, min=1e-12)
    min_pos = dataset.min_pos
    max_pos = dataset.max_pos

    # Build models
    smart_arch = OmegaConf.to_container(smart_cfg.architecture, resolve=True)
    smart_model = SMART(
        spatial_dim=3,
        surface_channels=len(SURFACE_FIELDS),
        volume_channels=len(VOLUME_FIELDS),
        parameter_channels=0,
        **smart_arch,
    ).to(device)
    smart_ckpt_path = choose_ckpt(smart_cfg, args.smart_checkpoint)
    smart_ckpt = torch.load(smart_ckpt_path, map_location=device)
    smart_model.load_state_dict(smart_ckpt["model_state_dict"], strict=True)
    smart_model.eval()

    cat_arch = OmegaConf.to_container(cat_cfg.architecture, resolve=True)
    cat_arch["stage2_surface_channels"] = len(SURFACE_FIELDS)
    cat_stage1 = CAT(
        spatial_dim=3,
        surface_channels=len(SURFACE_FIELDS),
        volume_channels=len(VOLUME_FIELDS),
        parameter_channels=0,
        **cat_arch,
    ).to(device)
    ckpt_s1 = torch.load(args.cat_stage1_checkpoint, map_location=device)
    cat_stage1.load_state_dict(ckpt_s1["model_state_dict"], strict=True)
    cat_stage1.eval()

    cat_stage2 = CAT(
        spatial_dim=3,
        surface_channels=len(SURFACE_FIELDS),
        volume_channels=len(VOLUME_FIELDS),
        parameter_channels=0,
        **cat_arch,
    ).to(device)
    ckpt_s2 = torch.load(args.cat_stage2_checkpoint, map_location=device)
    cat_stage2.load_state_dict(ckpt_s2["model_state_dict"], strict=True)
    cat_stage2.eval()

    # Optional uncertainty params
    uq = None
    mu_train = inv_sigma_train = Sigma_LLL = V_skew = None
    K = 1.0
    if args.uq_params:
        uq = torch.load(args.uq_params, map_location="cpu")
        mu_train = uq["mu_train"].float().to(device)
        inv_sigma_train = uq["inv_sigma_train"].float().to(device)
        Sigma_LLL = uq["Sigma_LLL"].float().to(device)
        V_skew = uq["V_skew"].float().to(device)
        K = float(uq["K"].item() if isinstance(uq["K"], torch.Tensor) else uq["K"])

    # Hooks for UQ tensors from stage1 pass.
    uq_cache: Dict[str, torch.Tensor] = {}
    h_q = h_z = None
    if uq is not None:
        def hook_q(module, inputs, output):
            del module, output
            uq_cache["q"] = inputs[0].detach()
        def hook_z(module, inputs, output):
            del module, inputs
            uq_cache["z"] = output.detach()
        h_q = cat_stage2.stage2_head[0].register_forward_hook(hook_q)
        h_z = cat_stage2.stage2_head[3].register_forward_hook(hook_z)

    # Streaming stats over full test subset.
    surf_cat_stats = RunningFieldStats(SURFACE_FIELDS)
    surf_smart_stats = RunningFieldStats(SURFACE_FIELDS)
    vol_cat_stats = RunningFieldStats(VOLUME_FIELDS)
    vol_smart_stats = RunningFieldStats(VOLUME_FIELDS)

    # Representative run buffers (for detailed files/plots)
    rep = {}

    try:
        for rid in eval_run_ids:
            run_dir = Path(smart_cfg.data_path) / f"run_{rid}"
            if not run_dir.is_dir():
                continue
            surf_coords = np.load(run_dir / "surface_coords.npy").astype(np.float32, copy=False)
            surf_p = np.load(run_dir / "surface_pMeanTrim.npy").astype(np.float32, copy=False).reshape(-1, 1)
            surf_n = np.load(run_dir / "surface_normals.npy").astype(np.float32, copy=False)
            surf_wx = np.load(run_dir / "surface_wallShearStressMeanTrim_x.npy").astype(np.float32, copy=False).reshape(-1, 1)
            surf_wy = np.load(run_dir / "surface_wallShearStressMeanTrim_y.npy").astype(np.float32, copy=False).reshape(-1, 1)
            surf_wz = np.load(run_dir / "surface_wallShearStressMeanTrim_z.npy").astype(np.float32, copy=False).reshape(-1, 1)
            surf_gt_np = np.concatenate([surf_p, surf_n, surf_wx, surf_wy, surf_wz], axis=1)

            vol_coords = np.load(run_dir / "volume_coords.npy").astype(np.float32, copy=False)
            vol_p = np.load(run_dir / "volume_pMeanTrim.npy").astype(np.float32, copy=False).reshape(-1, 1)
            vol_u = np.load(run_dir / "volume_UMeanTrim.npy").astype(np.float32, copy=False)
            vol_gt_np = np.concatenate([vol_p, vol_u], axis=1)

            surf_coords_t = torch.from_numpy(surf_coords)
            vol_coords_t = torch.from_numpy(vol_coords)
            surf_query_norm = normalize_pos(surf_coords_t, min_pos, max_pos)
            vol_query_norm = normalize_pos(vol_coords_t, min_pos, max_pos)

            rng = np.random.default_rng(args.seed + int(rid))
            smart_geo_idx = sample_input_idx(surf_coords.shape[0], int(smart_cfg.num_body_points), rng)
            cat_s_in = int(getattr(cat_cfg, "single_surface_input_points", cat_cfg.num_body_points))
            cat_in_idx = sample_input_idx(surf_coords.shape[0], cat_s_in, rng)
            smart_geo_norm = surf_query_norm[torch.from_numpy(smart_geo_idx)]
            cat_input_norm = surf_query_norm[torch.from_numpy(cat_in_idx)]

            with torch.inference_mode():
                smart_s_norm, smart_v_norm = smart_model.inference(
                    smart_geo_norm.unsqueeze(0).to(device),
                    surf_query_norm.unsqueeze(0).to(device),
                    vol_query_norm.unsqueeze(0).to(device),
                    None,
                )
                smart_s_np = to_np(denorm_fields(smart_s_norm[0].cpu(), mean_s, std_s))
                smart_v_np = to_np(denorm_fields(smart_v_norm[0].cpu(), mean_v, std_v))

                uq_cache.clear()
                cat_query_chunk = int(getattr(cat_stage1, "subregion_size", 262144))
                cat_s1_norm = cat_stage1_predict_full(
                    cat_stage1,
                    cat_input_norm,
                    surf_query_norm,
                    chunk_size=cat_query_chunk,
                    device=device,
                )
                cat_s_np = to_np(denorm_fields(cat_s1_norm, mean_s, std_s))

                cat_v_norm = cat_stage2_predict_full_cached(
                    cat_stage2,
                    cat_input_norm,
                    surf_query_norm,
                    vol_query_norm,
                    chunk_size=int(args.cat_vol_chunk),
                    device=device,
                )
                cat_v_np = to_np(denorm_fields(cat_v_norm, mean_v, std_v))

            surf_cat_stats.update(surf_gt_np, cat_s_np)
            surf_smart_stats.update(surf_gt_np, smart_s_np)
            vol_cat_stats.update(vol_gt_np, cat_v_np)
            vol_smart_stats.update(vol_gt_np, smart_v_np)

            if int(rid) == rep_run_id:
                rep = {
                    "rid": rid,
                    "surf_coords": surf_coords,
                    "vol_coords": vol_coords,
                    "surf_gt": surf_gt_np,
                    "vol_gt": vol_gt_np,
                    "smart_s": smart_s_np,
                    "smart_v": smart_v_np,
                    "cat_s": cat_s_np,
                    "cat_v": cat_v_np,
                }
                if uq is not None and "q" in uq_cache and "z" in uq_cache:
                    qg = uq_cache["q"].mean(dim=1)
                    dq = qg - mu_train.unsqueeze(0)
                    md2 = torch.einsum("bi,ij,bj->b", dq, inv_sigma_train, dq)
                    md = torch.sqrt(torch.clamp(md2, min=1e-12))[0]
                    z = uq_cache["z"][0]
                    var_lll = torch.sum((z @ Sigma_LLL) * z, dim=-1)
                    var_final = var_lll * (1.0 + float(args.gamma) * md)
                    cross = z @ V_skew
                    denom = torch.sqrt(torch.clamp(var_final * K - cross * cross, min=1e-6))
                    alpha = cross / denom
                    rep["uq_md"] = float(md.item())
                    rep["uq_var"] = var_final.detach().cpu().numpy()
                    rep["uq_alpha"] = alpha.detach().cpu().numpy()
    finally:
        if h_q is not None:
            h_q.remove()
        if h_z is not None:
            h_z.remove()

    if not rep:
        raise RuntimeError(f"Representative run {rep_run_id} not found in evaluated set.")

    surf_coords = rep["surf_coords"]
    vol_coords = rep["vol_coords"]
    surf_gt = rep["surf_gt"]
    vol_gt = rep["vol_gt"]
    smart_s_np = rep["smart_s"]
    smart_v_np = rep["smart_v"]
    cat_s_np = rep["cat_s"]
    cat_v_np = rep["cat_v"]

    surface_agg_cat = surf_cat_stats.summary()
    surface_agg_smart = surf_smart_stats.summary()
    volume_agg_cat = vol_cat_stats.summary()
    volume_agg_smart = vol_smart_stats.summary()

    surface_metrics = {
        "cat_stage1": surface_agg_cat["fields"],
        "smart": surface_agg_smart["fields"],
        "cat_stage1_global": surface_agg_cat["global"],
        "smart_global": surface_agg_smart["global"],
    }
    volume_metrics = {
        "cat_stage2": volume_agg_cat["fields"],
        "smart": volume_agg_smart["fields"],
        "cat_stage2_global": volume_agg_cat["global"],
        "smart_global": volume_agg_smart["global"],
    }

    out_root = Path(args.output_dir or (SMART_ROOT.parent / "results" / "drivaerml_full_compare" / f"run_{rep_run_id}"))
    out_root.mkdir(parents=True, exist_ok=True)

    # Save ParaView-friendly files (same points; switch arrays in ParaView).
    surf_point_data = {}
    vol_point_data = {}
    for i, f in enumerate(SURFACE_FIELDS):
        surf_point_data[f"gt_{f}"] = surf_gt[:, i]
        surf_point_data[f"cat_stage1_{f}"] = cat_s_np[:, i]
        surf_point_data[f"smart_{f}"] = smart_s_np[:, i]
        surf_point_data[f"err_cat_stage1_{f}"] = cat_s_np[:, i] - surf_gt[:, i]
        surf_point_data[f"err_smart_{f}"] = smart_s_np[:, i] - surf_gt[:, i]
        surf_point_data[f"abs_err_cat_stage1_{f}"] = np.abs(cat_s_np[:, i] - surf_gt[:, i])
        surf_point_data[f"abs_err_smart_{f}"] = np.abs(smart_s_np[:, i] - surf_gt[:, i])

    for i, f in enumerate(VOLUME_FIELDS):
        vol_point_data[f"gt_{f}"] = vol_gt[:, i]
        vol_point_data[f"cat_stage2_{f}"] = cat_v_np[:, i]
        vol_point_data[f"smart_{f}"] = smart_v_np[:, i]
        vol_point_data[f"err_cat_stage2_{f}"] = cat_v_np[:, i] - vol_gt[:, i]
        vol_point_data[f"err_smart_{f}"] = smart_v_np[:, i] - vol_gt[:, i]
        vol_point_data[f"abs_err_cat_stage2_{f}"] = np.abs(cat_v_np[:, i] - vol_gt[:, i])
        vol_point_data[f"abs_err_smart_{f}"] = np.abs(smart_v_np[:, i] - vol_gt[:, i])

    # Convenience vector arrays for quick glyph visualization.
    surf_point_data["gt_normals"] = surf_gt[:, 1:4]
    surf_point_data["cat_stage1_normals"] = cat_s_np[:, 1:4]
    surf_point_data["smart_normals"] = smart_s_np[:, 1:4]
    surf_point_data["gt_wall_shear"] = surf_gt[:, 4:7]
    surf_point_data["cat_stage1_wall_shear"] = cat_s_np[:, 4:7]
    surf_point_data["smart_wall_shear"] = smart_s_np[:, 4:7]
    if "uq_var" in rep:
        surf_point_data["uq_variance_final"] = rep["uq_var"]
        surf_point_data["uq_alpha_exact"] = rep["uq_alpha"]
        surf_point_data["uq_md_scalar"] = np.full((surf_coords.shape[0],), rep["uq_md"], dtype=np.float32)

    vol_point_data["gt_velocity"] = vol_gt[:, 1:4]
    vol_point_data["cat_stage2_velocity"] = cat_v_np[:, 1:4]
    vol_point_data["smart_velocity"] = smart_v_np[:, 1:4]

    write_polydata_vtk(out_root / "surface_predictions.vtk", surf_coords, surf_point_data)
    write_polydata_vtk(out_root / "volume_predictions.vtk", vol_coords, vol_point_data)

    # Plotting (downsampled for speed/readability only).
    surf_plot_coords, surf_plot_arrs = downsample_for_plot(
        surf_coords,
        [surf_gt, cat_s_np, smart_s_np],
        max_points=int(args.plot_max_points),
        seed=args.seed + 1000 + args.run_id,
    )
    surf_plot_gt, surf_plot_cat, surf_plot_smart = surf_plot_arrs
    vol_plot_coords, vol_plot_arrs = downsample_for_plot(
        vol_coords,
        [vol_gt, cat_v_np, smart_v_np],
        max_points=int(args.plot_max_points),
        seed=args.seed + 2000 + args.run_id,
    )
    vol_plot_gt, vol_plot_cat, vol_plot_smart = vol_plot_arrs

    plots_dir = out_root / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    plot_field_maps(
        plots_dir / "surface_field_maps.png",
        "Surface Fields: GT vs CAT Stage1 vs SMART",
        surf_plot_coords,
        surf_plot_gt,
        surf_plot_cat,
        surf_plot_smart,
        SURFACE_FIELDS,
        "CAT-S1",
        "SMART",
    )
    plot_field_maps(
        plots_dir / "volume_field_maps.png",
        "Volume Fields: GT vs CAT Stage2 vs SMART",
        vol_plot_coords,
        vol_plot_gt,
        vol_plot_cat,
        vol_plot_smart,
        VOLUME_FIELDS,
        "CAT-S2",
        "SMART",
    )
    plot_parity(plots_dir / "surface_parity.png", surf_plot_gt, surf_plot_cat, surf_plot_smart, SURFACE_FIELDS, "CAT-S1", "SMART")
    plot_parity(plots_dir / "volume_parity.png", vol_plot_gt, vol_plot_cat, vol_plot_smart, VOLUME_FIELDS, "CAT-S2", "SMART")
    plot_error_hist(plots_dir / "surface_error_hist.png", surf_plot_gt, surf_plot_cat, surf_plot_smart, SURFACE_FIELDS, "CAT-S1", "SMART")
    plot_error_hist(plots_dir / "volume_error_hist.png", vol_plot_gt, vol_plot_cat, vol_plot_smart, VOLUME_FIELDS, "CAT-S2", "SMART")
    plot_grouped_metrics(
        plots_dir / "surface_rel_l2_by_field.png",
        "rel_l2",
        surface_metrics["cat_stage1"],
        surface_metrics["smart"],
        SURFACE_FIELDS,
        "CAT-S1",
        "SMART",
    )
    plot_grouped_metrics(
        plots_dir / "volume_rel_l2_by_field.png",
        "rel_l2",
        volume_metrics["cat_stage2"],
        volume_metrics["smart"],
        VOLUME_FIELDS,
        "CAT-S2",
        "SMART",
    )

    metrics_payload = {
        "representative_run_id": int(rep_run_id),
        "evaluated_run_ids": [int(x) for x in eval_run_ids],
        "surface_points_total": int(surf_coords.shape[0]),
        "volume_points_total": int(vol_coords.shape[0]),
        "smart_checkpoint": smart_ckpt_path,
        "cat_stage1_checkpoint": args.cat_stage1_checkpoint,
        "cat_stage2_checkpoint": args.cat_stage2_checkpoint,
        "uq_params": args.uq_params,
        "surface_metrics": surface_metrics,
        "volume_metrics": volume_metrics,
    }
    (out_root / "metrics.json").write_text(json.dumps(metrics_payload, indent=2))

    summary_lines = [
        "DrivAerML Full-Query Comparison Summary",
        "======================================",
        f"Representative run id: {rep_run_id}",
        f"Evaluated runs for summary: {len(eval_run_ids)}",
        f"Surface points in representative run: {surf_coords.shape[0]}",
        f"Volume points in representative run: {vol_coords.shape[0]}",
        f"SMART checkpoint: {smart_ckpt_path}",
        f"CAT stage1 checkpoint: {args.cat_stage1_checkpoint}",
        f"CAT stage2 checkpoint: {args.cat_stage2_checkpoint}",
        f"UQ params: {args.uq_params}",
        "",
        "Global Metrics (lower is better)",
        "--------------------------------",
        f"Surface CAT-S1: {surface_metrics['cat_stage1_global']}",
        f"Surface SMART:  {surface_metrics['smart_global']}",
        f"Volume CAT-S2: {volume_metrics['cat_stage2_global']}",
        f"Volume SMART:  {volume_metrics['smart_global']}",
        "",
        "Per-field Surface rel_l2",
        "-------------------------",
    ]
    for f in SURFACE_FIELDS:
        summary_lines.append(
            f"{f:18s} CAT-S1 rel_l2={surface_metrics['cat_stage1'][f]['rel_l2']:.6f}, R2={surface_metrics['cat_stage1'][f]['r2']:.6f}"
            f" | SMART rel_l2={surface_metrics['smart'][f]['rel_l2']:.6f}, R2={surface_metrics['smart'][f]['r2']:.6f}"
        )
    summary_lines += ["", "Per-field Volume rel_l2", "------------------------"]
    for f in VOLUME_FIELDS:
        summary_lines.append(
            f"{f:18s} CAT-S2 rel_l2={volume_metrics['cat_stage2'][f]['rel_l2']:.6f}, R2={volume_metrics['cat_stage2'][f]['r2']:.6f}"
            f" | SMART rel_l2={volume_metrics['smart'][f]['rel_l2']:.6f}, R2={volume_metrics['smart'][f]['r2']:.6f}"
        )
    (out_root / "summary.txt").write_text("\n".join(summary_lines))

    print(f"Saved outputs to: {out_root}")
    print("Main files:")
    print(f"  - {out_root / 'surface_predictions.vtk'}")
    print(f"  - {out_root / 'volume_predictions.vtk'}")
    print(f"  - {out_root / 'metrics.json'}")
    print(f"  - {out_root / 'summary.txt'}")
    print(f"  - {plots_dir}")


if __name__ == "__main__":
    main()
