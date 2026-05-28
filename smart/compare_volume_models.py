#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import zlib
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm.auto import tqdm

from data.naca4_dataset import NACA4Dataset
from models.smart.smart import SMART
from models.smart.cat import CAT
from utils.utils import apply_naca4_auto_point_budget, print_point_budget


def parse_args():
    p = argparse.ArgumentParser(description="Enhanced SMART vs CAT comparison on NACA4.")
    p.add_argument("--smart-config", default="naca4")
    p.add_argument("--cat-config", default="naca4_cat")
    p.add_argument("--smart-checkpoint", required=True)
    p.add_argument("--cat-checkpoint", required=True)
    p.add_argument("--split", choices=["train", "test"], default="test")
    p.add_argument("--num-cases", type=int, default=30)
    p.add_argument("--case-ids", default=None)
    p.add_argument("--chunk-size", type=int, default=65536)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--roi", nargs=4, type=float, default=None, metavar=("XMIN", "XMAX", "YMIN", "YMAX"), help="Optional ROI for metric computation.")
    p.add_argument("--viz-roi", nargs=4, type=float, default=[-2.0, 3.0, -2.0, 2.0], metavar=("XMIN", "XMAX", "YMIN", "YMAX"), help="ROI used only for visualization.")
    p.add_argument("--viz-cases", type=int, default=5)
    p.add_argument("--max-plot-points", type=int, default=120000)
    p.add_argument("--output-dir", default="results/compare_volume")
    return p.parse_args()


def load_cfg(name: str):
    path = Path(__file__).resolve().parent / "config" / f"{name}.yaml"
    if not path.is_file():
        raise FileNotFoundError(path)
    return OmegaConf.load(path).experiment


def init_device(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def stable_seed(case_id: str) -> int:
    return zlib.adler32(case_id.encode("utf-8")) & 0xFFFFFFFF


def sample_indices(n: int, k: int, seed: int):
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    if k <= 0 or k >= n:
        return torch.arange(n)
    return torch.randint(0, n, (k,), generator=g)


def roi_mask(points: torch.Tensor, roi):
    if roi is None:
        return torch.ones((points.shape[0],), dtype=torch.bool)
    xmin, xmax, ymin, ymax = roi
    return (points[:, 0] >= xmin) & (points[:, 0] <= xmax) & (points[:, 1] >= ymin) & (points[:, 1] <= ymax)


def rel_l2(pred: np.ndarray, gt: np.ndarray, eps=1e-8):
    return float(np.linalg.norm(pred - gt) / max(np.linalg.norm(gt), eps))


def metrics(pred: np.ndarray, gt: np.ndarray):
    err = pred - gt
    abs_err = np.abs(err)
    mse = float(np.mean(err ** 2))
    gt_mean = float(np.mean(gt))
    ss_res = float(np.sum((pred - gt) ** 2))
    ss_tot = float(np.sum((gt - gt_mean) ** 2))
    r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
    return {
        "mae": float(abs_err.mean()),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "r2": float(r2),
        "rel_l2": rel_l2(pred, gt),
        "p95_abs": float(np.percentile(abs_err, 95)),
        "median_abs": float(np.median(abs_err)),
    }


def bootstrap_ci(values: np.ndarray, n_boot=2000, alpha=0.05, seed=42):
    if values.size == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    means = []
    n = len(values)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        means.append(values[idx].mean())
    lo = float(np.percentile(means, 100 * alpha / 2))
    hi = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return lo, hi


def load_smart_with_ckpt(cfg, ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt["model_state_dict"]

    surface_fields = ckpt.get("surface_fields")
    volume_fields = ckpt.get("volume_fields")
    if surface_fields is not None and volume_fields is not None:
        surf_ch = max(1, len(surface_fields))
        vol_ch = len(volume_fields)
    else:
        out_dim = int(state["mlp.4.weight"].shape[0])
        surf_ch = 1
        vol_ch = out_dim - surf_ch

    kwargs = {
        "spatial_dim": 2,
        "surface_channels": surf_ch,
        "volume_channels": vol_ch,
        "parameter_channels": 0,
    }
    if "architecture" in cfg:
        kwargs.update(OmegaConf.to_container(cfg.architecture, resolve=True))

    model = SMART(**kwargs).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, ckpt


def load_cat_with_ckpt(cfg, ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt["model_state_dict"]

    vol_ch = int(state["volume_head.4.weight"].shape[0])
    surf_out = int(state["stage2_head.4.weight"].shape[0])

    kwargs = {
        "spatial_dim": 2,
        "surface_channels": 3,
        "volume_channels": vol_ch,
        "parameter_channels": 0,
        "stage2_surface_channels": surf_out,
    }
    if "architecture" in cfg:
        kwargs.update(OmegaConf.to_container(cfg.architecture, resolve=True))

    model = CAT(**kwargs).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, ckpt


@torch.inference_mode()
def predict_smart_volume(model: SMART, geo_norm: torch.Tensor, vol_norm: torch.Tensor, chunk_size: int, device: torch.device):
    inter, latent_pos = model.encode(geo_norm, None)
    out = []
    n_surf = geo_norm.shape[1]
    surf_query = geo_norm
    for i in range(0, vol_norm.shape[0], chunk_size):
        vq = vol_norm[i:i + chunk_size].to(device).unsqueeze(0)
        q = torch.cat([surf_query, vq], dim=1)
        pred = model.decode(inter, latent_pos, None, q)
        out.append(pred[0, n_surf:, model.surface_channels:].cpu())
    return torch.cat(out, dim=0)


@torch.inference_mode()
def predict_cat_volume(model: CAT, surf_norm: torch.Tensor, vol_norm: torch.Tensor, chunk_size: int):
    out = []
    for i in range(0, vol_norm.shape[0], chunk_size):
        q = vol_norm[i:i + chunk_size].unsqueeze(0)
        pred = model.forward_stage2_only(surf_norm, surf_norm, q, return_aux=False)
        out.append(pred[0].cpu())
    return torch.cat(out, dim=0)


def _subsample_for_plot(*arrs, max_points: int, seed: int):
    n = len(arrs[0])
    if n <= max_points:
        return arrs
    idx = np.random.default_rng(seed).integers(0, n, size=max_points)
    return tuple(a[idx] for a in arrs)


def parity_plot(out_path: Path, gt: np.ndarray, a: np.ndarray, b: np.ndarray, name: str, a_name: str, b_name: str, a_stats: dict, b_stats: dict, max_points: int):
    g, pa, pb = _subsample_for_plot(gt, a, b, max_points=max_points, seed=0)
    mn = float(np.percentile(np.concatenate([g, pa, pb]), 1))
    mx = float(np.percentile(np.concatenate([g, pa, pb]), 99))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for ax, pred, label, stats in [(axes[0], pa, a_name, a_stats), (axes[1], pb, b_name, b_stats)]:
        ax.scatter(g, pred, s=1, alpha=0.22, color="#1f77b4")
        ax.plot([mn, mx], [mn, mx], "--", lw=1, color="#ff7f0e")
        ax.set_title(f"{label} vs GT ({name})")
        ax.set_xlabel("GT")
        ax.set_ylabel(label)
        ax.text(
            0.03,
            0.97,
            f"R2={stats['r2']:.4f}\nMAE={stats['mae']:.4g}\nRMSE={stats['rmse']:.4g}\nRelL2={stats['rel_l2']:.4g}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
            fontsize=9,
        )
        
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def abs_err_hist_and_cdf(out_path: Path, err_a: np.ndarray, err_b: np.ndarray, name: str, a_name: str, b_name: str):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    bins = 120
    axes[0].hist(err_a, bins=bins, density=True, alpha=0.45, label=f"{a_name} |err|", color="#1f77b4")
    axes[0].hist(err_b, bins=bins, density=True, alpha=0.45, label=f"{b_name} |err|", color="#d62728")
    axes[0].set_title(f"Abs error density ({name})")
    axes[0].set_xlabel("Absolute error")
    axes[0].set_ylabel("Density")
    axes[0].legend()

    sa = np.sort(err_a)
    sb = np.sort(err_b)
    ca = np.linspace(0, 1, sa.size, endpoint=True)
    cb = np.linspace(0, 1, sb.size, endpoint=True)
    axes[1].plot(sa, ca, label=a_name, lw=2, color="#1f77b4")
    axes[1].plot(sb, cb, label=b_name, lw=2, color="#d62728")
    axes[1].set_title(f"Abs error CDF ({name})")
    axes[1].set_xlabel("Absolute error")
    axes[1].set_ylabel("CDF")
    axes[1].legend()

    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def error_quantile_plot(out_path: Path, err_a: np.ndarray, err_b: np.ndarray, name: str, a_name: str, b_name: str):
    q = np.linspace(1, 99.9, 220)
    qa = np.percentile(err_a, q)
    qb = np.percentile(err_b, q)

    fig, ax = plt.subplots(1, 1, figsize=(8, 5), constrained_layout=True)
    ax.plot(q, qa, label=a_name, lw=2, color="#1f77b4")
    ax.plot(q, qb, label=b_name, lw=2, color="#d62728")
    ax.set_title(f"Absolute-error quantile curve ({name})")
    ax.set_xlabel("Percentile")
    ax.set_ylabel("Absolute error")
    ax.legend()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def bland_altman_plot(out_path: Path, gt: np.ndarray, pred: np.ndarray, name: str, model_name: str, max_points: int):
    diff = pred - gt
    mean = 0.5 * (pred + gt)
    mu = float(np.mean(diff))
    sd = float(np.std(diff))

    mean, diff = _subsample_for_plot(mean, diff, max_points=max_points, seed=1)

    fig, ax = plt.subplots(1, 1, figsize=(7, 5), constrained_layout=True)
    ax.scatter(mean, diff, s=1, alpha=0.2, color="#3f3f3f")
    ax.axhline(mu, color="k", lw=1.5, label="Mean diff")
    ax.axhline(mu + 1.96 * sd, color="#ff7f0e", ls="--", lw=1.2, label="±1.96σ")
    ax.axhline(mu - 1.96 * sd, color="#ff7f0e", ls="--", lw=1.2)
    ax.set_title(f"Bland-Altman ({model_name}, {name})")
    ax.set_xlabel("Mean of GT and prediction")
    ax.set_ylabel("Prediction - GT")
    ax.legend()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def delta_bar_plot(out_path: Path, rows: list[dict]):
    case_ids = [r["case_id"] for r in rows]
    dp = np.array([r["delta_pressure_rel_l2_cat_minus_smart"] for r in rows], dtype=float)
    dv = np.array([r["delta_velocity_rel_l2_cat_minus_smart"] for r in rows], dtype=float)

    x = np.arange(len(rows))
    fig, axes = plt.subplots(2, 1, figsize=(max(12, len(rows) * 0.3), 8), constrained_layout=True, sharex=True)

    axes[0].bar(x, dp, color=np.where(dp < 0, "#2ca02c", "#d62728"))
    axes[0].axhline(0.0, color="k", lw=1)
    axes[0].set_ylabel("CAT-SMART relL2")
    axes[0].set_title("Pressure per-case delta (negative = CAT better)")

    axes[1].bar(x, dv, color=np.where(dv < 0, "#2ca02c", "#d62728"))
    axes[1].axhline(0.0, color="k", lw=1)
    axes[1].set_ylabel("CAT-SMART relL2")
    axes[1].set_title("Velocity per-case delta (negative = CAT better)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(case_ids, rotation=90, fontsize=7)

    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def box_violin_delta_plot(out_path: Path, rows: list[dict]):
    dp = np.array([r["delta_pressure_rel_l2_cat_minus_smart"] for r in rows], dtype=float)
    dv = np.array([r["delta_velocity_rel_l2_cat_minus_smart"] for r in rows], dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    vp = axes[0].violinplot([dp, dv], showmeans=True, showextrema=True)
    for body in vp["bodies"]:
        body.set_facecolor("#8da0cb")
        body.set_alpha(0.6)
    axes[0].axhline(0.0, color="k", lw=1)
    axes[0].set_xticks([1, 2])
    axes[0].set_xticklabels(["Pressure Δ", "Velocity Δ"])
    axes[0].set_title("Delta relL2 distribution (CAT-SMART)")

    axes[1].boxplot([dp, dv], labels=["Pressure Δ", "Velocity Δ"], vert=True)
    axes[1].axhline(0.0, color="k", lw=1)
    axes[1].set_title("Delta relL2 boxplot")

    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def win_rate_plot(out_path: Path, rows: list[dict]):
    dp = np.array([r["delta_pressure_rel_l2_cat_minus_smart"] for r in rows], dtype=float)
    dv = np.array([r["delta_velocity_rel_l2_cat_minus_smart"] for r in rows], dtype=float)

    p_win = float(np.mean(dp < 0))
    v_win = float(np.mean(dv < 0))

    fig, ax = plt.subplots(1, 1, figsize=(6, 4), constrained_layout=True)
    labels = ["Pressure", "Velocity"]
    wins = [p_win, v_win]
    ax.bar(labels, wins, color=["#2ca02c", "#1f77b4"])
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("CAT win fraction")
    ax.set_title("Case win-rate (lower relL2 wins)")
    for i, v in enumerate(wins):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center")
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def per_case_panel(out_path: Path, xy: np.ndarray, gt: np.ndarray, p_smart: np.ndarray, p_cat: np.ndarray, title: str, airfoil_xy: np.ndarray, viz_roi):
    err_s = np.abs(p_smart - gt)
    err_c = np.abs(p_cat - gt)
    delta_err = err_c - err_s

    vmin = float(np.percentile(np.concatenate([gt, p_smart, p_cat]), 1))
    vmax = float(np.percentile(np.concatenate([gt, p_smart, p_cat]), 99))
    emax = float(np.percentile(np.concatenate([err_s, err_c]), 99))
    de = float(np.percentile(np.abs(delta_err), 99))

    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    fig.suptitle(title)

    plots = [
        (gt, "GT", "coolwarm", vmin, vmax),
        (p_smart, "SMART Pred", "coolwarm", vmin, vmax),
        (p_cat, "CAT Pred", "coolwarm", vmin, vmax),
        (err_s, "SMART |Err|", "inferno", 0.0, emax),
        (err_c, "CAT |Err|", "inferno", 0.0, emax),
        (delta_err, "(CAT|Err| - SMART|Err|)", "RdBu_r", -de, de),
    ]

    xmin, xmax, ymin, ymax = viz_roi
    for i, (vals, t, cmap, lo, hi) in enumerate(plots):
        r, c = divmod(i, 3)
        sc = axes[r, c].scatter(xy[:, 0], xy[:, 1], c=vals, s=2, cmap=cmap, vmin=lo, vmax=hi, linewidths=0)
        if airfoil_xy.size > 0:
            axes[r, c].scatter(airfoil_xy[:, 0], airfoil_xy[:, 1], s=1.5, c="black", alpha=0.9, linewidths=0)
        axes[r, c].set_xlim(xmin, xmax)
        axes[r, c].set_ylim(ymin, ymax)
        axes[r, c].set_title(t)
        axes[r, c].set_aspect("equal", adjustable="box")
        fig.colorbar(sc, ax=axes[r, c], shrink=0.8)

    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def spatial_hexbin_delta(out_path: Path, xy: np.ndarray, delta_err: np.ndarray, title: str, viz_roi):
    xmin, xmax, ymin, ymax = viz_roi
    fig, ax = plt.subplots(1, 1, figsize=(8, 6), constrained_layout=True)
    hb = ax.hexbin(xy[:, 0], xy[:, 1], C=delta_err, reduce_C_function=np.mean, gridsize=90, cmap="RdBu_r", vmin=-np.percentile(np.abs(delta_err), 98), vmax=np.percentile(np.abs(delta_err), 98))
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    cb = fig.colorbar(hb, ax=ax)
    cb.set_label("Mean (CAT|err| - SMART|err|)")
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main():
    args = parse_args()
    device = init_device(args.seed)

    smart_cfg = load_cfg(args.smart_config)
    cat_cfg = load_cfg(args.cat_config)

    ds_train = NACA4Dataset(
        smart_cfg.data_path,
        if_test=False,
        geometry_points=int(smart_cfg.num_body_points),
        surface_points=int(smart_cfg.num_surface_points),
        volume_points=int(smart_cfg.num_volume_points),
        scale_positions=bool(smart_cfg.scale_positions),
        manifest_variant=getattr(smart_cfg, "manifest_variant", "full"),
    )
    info_s = apply_naca4_auto_point_budget(smart_cfg, ds_train, for_cat=False)
    if info_s:
        print_point_budget("SMART-COMP", info_s)
    info_c = apply_naca4_auto_point_budget(cat_cfg, ds_train, for_cat=True)
    if info_c:
        print_point_budget("CAT-COMP", info_c)

    dataset = NACA4Dataset(
        smart_cfg.data_path,
        if_test=(args.split == "test"),
        geometry_points=int(smart_cfg.num_body_points),
        surface_points=int(smart_cfg.num_surface_points),
        volume_points=int(smart_cfg.num_volume_points),
        scale_positions=bool(smart_cfg.scale_positions),
        manifest_variant=getattr(smart_cfg, "manifest_variant", "full"),
    )

    smart_model, smart_ckpt_obj = load_smart_with_ckpt(smart_cfg, args.smart_checkpoint, device)
    cat_model, _ = load_cat_with_ckpt(cat_cfg, args.cat_checkpoint, device)

    case_ids = list(dataset.data)
    if args.case_ids:
        case_ids = [x.strip() for x in args.case_ids.split(",") if x.strip()]
    else:
        case_ids = case_ids[:args.num_cases]

    out_root = Path(args.output_dir) / f"smart_vs_cat_{args.split}_n{len(case_ids)}"
    out_root.mkdir(parents=True, exist_ok=True)

    rows = []
    all_gt_p, all_smart_p, all_cat_p = [], [], []
    all_gt_speed, all_smart_speed, all_cat_speed = [], [], []
    all_gt_vx, all_smart_vx, all_cat_vx = [], [], []
    all_gt_vy, all_smart_vy, all_cat_vy = [], [], []

    viz_roi = tuple(args.viz_roi)
    all_xy_pressure, all_delta_pressure = [], []
    all_xy_speed, all_delta_speed = [], []

    for i, case_id in enumerate(tqdm(case_ids, desc="Comparing cases", dynamic_ncols=True)):
        _, surf_mesh, _, vol_mesh, vol_data = dataset._load_case_arrays(case_id, write_cache=True)

        vm = roi_mask(vol_mesh, args.roi)
        vol_xy = vol_mesh[vm]
        vol_gt = vol_data[vm]
        if vol_xy.numel() == 0:
            continue

        k = min(int(smart_cfg.num_body_points), surf_mesh.shape[0]) if int(smart_cfg.num_body_points) > 0 else surf_mesh.shape[0]
        idx = sample_indices(surf_mesh.shape[0], k, seed=args.seed + stable_seed(case_id))
        surf_input = surf_mesh[idx]

        surf_norm = ((surf_input - dataset.min_pos) / (dataset.max_pos - dataset.min_pos)).to(device).unsqueeze(0)
        vol_norm = ((vol_xy - dataset.min_pos) / (dataset.max_pos - dataset.min_pos)).to(device)

        pred_smart_n = predict_smart_volume(smart_model, surf_norm, vol_norm, args.chunk_size, device)
        pred_cat_n = predict_cat_volume(cat_model, surf_norm, vol_norm, args.chunk_size)

        vol_fields = smart_ckpt_obj.get("volume_fields", ["pressure", "velocity_x", "velocity_y"])
        if len(vol_fields) == 3:
            mean_vol = torch.stack([dataset.mean_vol_data[0], dataset.mean_vol_data[2], dataset.mean_vol_data[3]])
            std_vol = torch.stack([dataset.std_vol_data[0], dataset.std_vol_data[2], dataset.std_vol_data[3]])
            gt = torch.cat([vol_gt[:, :1], vol_gt[:, 2:4]], dim=-1).numpy()
        else:
            mean_vol = dataset.mean_vol_data[: pred_smart_n.shape[-1]]
            std_vol = dataset.std_vol_data[: pred_smart_n.shape[-1]]
            gt = vol_gt[:, : pred_smart_n.shape[-1]].numpy()

        pred_smart = (pred_smart_n * std_vol + mean_vol).numpy()
        pred_cat = (pred_cat_n * std_vol + mean_vol).numpy()

        gt_p = gt[:, 0]
        s_p = pred_smart[:, 0]
        c_p = pred_cat[:, 0]

        gt_v = gt[:, 1:3]
        s_v = pred_smart[:, 1:3]
        c_v = pred_cat[:, 1:3]

        gt_speed = np.linalg.norm(gt_v, axis=1)
        s_speed = np.linalg.norm(s_v, axis=1)
        c_speed = np.linalg.norm(c_v, axis=1)

        m_sp = metrics(s_p, gt_p)
        m_cp = metrics(c_p, gt_p)
        m_sv = metrics(s_v, gt_v)
        m_cv = metrics(c_v, gt_v)

        rows.append({
            "case_id": case_id,
            "n_vol_points": int(vol_xy.shape[0]),
            "smart_pressure_rel_l2": m_sp["rel_l2"],
            "cat_pressure_rel_l2": m_cp["rel_l2"],
            "smart_velocity_rel_l2": m_sv["rel_l2"],
            "cat_velocity_rel_l2": m_cv["rel_l2"],
            "delta_pressure_rel_l2_cat_minus_smart": m_cp["rel_l2"] - m_sp["rel_l2"],
            "delta_velocity_rel_l2_cat_minus_smart": m_cv["rel_l2"] - m_sv["rel_l2"],
        })

        all_gt_p.append(gt_p); all_smart_p.append(s_p); all_cat_p.append(c_p)
        all_gt_speed.append(gt_speed); all_smart_speed.append(s_speed); all_cat_speed.append(c_speed)
        all_gt_vx.append(gt_v[:, 0]); all_smart_vx.append(s_v[:, 0]); all_cat_vx.append(c_v[:, 0])
        all_gt_vy.append(gt_v[:, 1]); all_smart_vy.append(s_v[:, 1]); all_cat_vy.append(c_v[:, 1])

        viz_mask = roi_mask(vol_xy, viz_roi).numpy()
        if viz_mask.any():
            xy_case = vol_xy.numpy()[viz_mask]
            delta_p = np.abs(c_p - gt_p)[viz_mask] - np.abs(s_p - gt_p)[viz_mask]
            delta_s = np.abs(c_speed - gt_speed)[viz_mask] - np.abs(s_speed - gt_speed)[viz_mask]
            all_xy_pressure.append(xy_case)
            all_delta_pressure.append(delta_p)
            all_xy_speed.append(xy_case)
            all_delta_speed.append(delta_s)

        if i < args.viz_cases:
            case_dir = out_root / f"case_{case_id}"
            case_dir.mkdir(parents=True, exist_ok=True)

            viz_mask_case = roi_mask(vol_xy, viz_roi).numpy()
            xy_plot = vol_xy.numpy()[viz_mask_case]
            if xy_plot.shape[0] == 0:
                continue

            gt_p_plot = gt_p[viz_mask_case]
            s_p_plot = s_p[viz_mask_case]
            c_p_plot = c_p[viz_mask_case]
            gt_speed_plot = gt_speed[viz_mask_case]
            s_speed_plot = s_speed[viz_mask_case]
            c_speed_plot = c_speed[viz_mask_case]

            airfoil_xy = surf_mesh.numpy()[roi_mask(surf_mesh, viz_roi).numpy()]
            if airfoil_xy.shape[0] > 5000:
                sub = np.random.default_rng(args.seed).integers(0, airfoil_xy.shape[0], size=5000)
                airfoil_xy = airfoil_xy[sub]

            per_case_panel(case_dir / "pressure_panel.png", xy_plot, gt_p_plot, s_p_plot, c_p_plot, f"Case {case_id} pressure (ROI zoom)", airfoil_xy, viz_roi)
            per_case_panel(case_dir / "speed_panel.png", xy_plot, gt_speed_plot, s_speed_plot, c_speed_plot, f"Case {case_id} speed (ROI zoom)", airfoil_xy, viz_roi)

    if not rows:
        raise RuntimeError("No valid cases processed.")

    gt_p = np.concatenate(all_gt_p)
    smart_p = np.concatenate(all_smart_p)
    cat_p = np.concatenate(all_cat_p)
    gt_speed = np.concatenate(all_gt_speed)
    smart_speed = np.concatenate(all_smart_speed)
    cat_speed = np.concatenate(all_cat_speed)
    gt_vx = np.concatenate(all_gt_vx)
    smart_vx = np.concatenate(all_smart_vx)
    cat_vx = np.concatenate(all_cat_vx)
    gt_vy = np.concatenate(all_gt_vy)
    smart_vy = np.concatenate(all_smart_vy)
    cat_vy = np.concatenate(all_cat_vy)

    smart_pressure_stats = metrics(smart_p, gt_p)
    cat_pressure_stats = metrics(cat_p, gt_p)
    smart_speed_stats = metrics(smart_speed, gt_speed)
    cat_speed_stats = metrics(cat_speed, gt_speed)
    smart_vx_stats = metrics(smart_vx, gt_vx)
    cat_vx_stats = metrics(cat_vx, gt_vx)
    smart_vy_stats = metrics(smart_vy, gt_vy)
    cat_vy_stats = metrics(cat_vy, gt_vy)

    parity_plot(out_root / "pressure_parity.png", gt_p, smart_p, cat_p, "pressure", "SMART", "CAT", smart_pressure_stats, cat_pressure_stats, args.max_plot_points)
    parity_plot(out_root / "speed_parity.png", gt_speed, smart_speed, cat_speed, "speed", "SMART", "CAT", smart_speed_stats, cat_speed_stats, args.max_plot_points)
    parity_plot(out_root / "vx_parity.png", gt_vx, smart_vx, cat_vx, "velocity_x", "SMART", "CAT", smart_vx_stats, cat_vx_stats, args.max_plot_points)
    parity_plot(out_root / "vy_parity.png", gt_vy, smart_vy, cat_vy, "velocity_y", "SMART", "CAT", smart_vy_stats, cat_vy_stats, args.max_plot_points)

    abs_err_hist_and_cdf(out_root / "pressure_abs_err_hist_cdf.png", np.abs(smart_p - gt_p), np.abs(cat_p - gt_p), "pressure", "SMART", "CAT")
    abs_err_hist_and_cdf(out_root / "speed_abs_err_hist_cdf.png", np.abs(smart_speed - gt_speed), np.abs(cat_speed - gt_speed), "speed", "SMART", "CAT")
    error_quantile_plot(out_root / "pressure_abs_err_quantiles.png", np.abs(smart_p - gt_p), np.abs(cat_p - gt_p), "pressure", "SMART", "CAT")
    error_quantile_plot(out_root / "speed_abs_err_quantiles.png", np.abs(smart_speed - gt_speed), np.abs(cat_speed - gt_speed), "speed", "SMART", "CAT")

    bland_altman_plot(out_root / "pressure_bland_altman_smart.png", gt_p, smart_p, "pressure", "SMART", args.max_plot_points)
    bland_altman_plot(out_root / "pressure_bland_altman_cat.png", gt_p, cat_p, "pressure", "CAT", args.max_plot_points)
    bland_altman_plot(out_root / "speed_bland_altman_smart.png", gt_speed, smart_speed, "speed", "SMART", args.max_plot_points)
    bland_altman_plot(out_root / "speed_bland_altman_cat.png", gt_speed, cat_speed, "speed", "CAT", args.max_plot_points)

    delta_bar_plot(out_root / "per_case_delta_bars.png", rows)
    box_violin_delta_plot(out_root / "delta_box_violin.png", rows)
    win_rate_plot(out_root / "cat_win_rate.png", rows)

    if all_xy_pressure and all_delta_pressure:
        xy_p = np.concatenate(all_xy_pressure, axis=0)
        de_p = np.concatenate(all_delta_pressure, axis=0)
        spatial_hexbin_delta(out_root / "spatial_delta_pressure_hexbin.png", xy_p, de_p, "Spatial error-gap map: pressure", viz_roi)
    if all_xy_speed and all_delta_speed:
        xy_s = np.concatenate(all_xy_speed, axis=0)
        de_s = np.concatenate(all_delta_speed, axis=0)
        spatial_hexbin_delta(out_root / "spatial_delta_speed_hexbin.png", xy_s, de_s, "Spatial error-gap map: speed", viz_roi)

    arr_dp = np.array([r["delta_pressure_rel_l2_cat_minus_smart"] for r in rows], dtype=float)
    arr_dv = np.array([r["delta_velocity_rel_l2_cat_minus_smart"] for r in rows], dtype=float)

    summary = {
        "num_cases": len(rows),
        "metric_roi": args.roi,
        "viz_roi": list(viz_roi),
        "pressure_delta_rel_l2_mean": float(arr_dp.mean()),
        "pressure_delta_rel_l2_ci95": bootstrap_ci(arr_dp),
        "velocity_delta_rel_l2_mean": float(arr_dv.mean()),
        "velocity_delta_rel_l2_ci95": bootstrap_ci(arr_dv),
        "pressure_cat_better_case_fraction": float(np.mean(arr_dp < 0)),
        "velocity_cat_better_case_fraction": float(np.mean(arr_dv < 0)),
        "smart_pressure_rel_l2_global": rel_l2(smart_p, gt_p),
        "cat_pressure_rel_l2_global": rel_l2(cat_p, gt_p),
        "smart_speed_rel_l2_global": rel_l2(smart_speed, gt_speed),
        "cat_speed_rel_l2_global": rel_l2(cat_speed, gt_speed),
        "smart_vx_rel_l2_global": rel_l2(smart_vx, gt_vx),
        "cat_vx_rel_l2_global": rel_l2(cat_vx, gt_vx),
        "smart_vy_rel_l2_global": rel_l2(smart_vy, gt_vy),
        "cat_vy_rel_l2_global": rel_l2(cat_vy, gt_vy),
        "smart_pressure_stats": smart_pressure_stats,
        "cat_pressure_stats": cat_pressure_stats,
        "smart_speed_stats": smart_speed_stats,
        "cat_speed_stats": cat_speed_stats,
        "smart_vx_stats": smart_vx_stats,
        "cat_vx_stats": cat_vx_stats,
        "smart_vy_stats": smart_vy_stats,
        "cat_vy_stats": cat_vy_stats,
        "smart_ckpt": args.smart_checkpoint,
        "cat_ckpt": args.cat_checkpoint,
    }

    with open(out_root / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with open(out_root / "per_case.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"Saved comparison outputs to: {out_root}")


if __name__ == "__main__":
    main()
