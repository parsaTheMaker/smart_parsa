#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
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
from models.smart.cat import CAT, LoopEncoder
from utils.utils import get_model_checkpoint_name, apply_naca4_auto_point_budget, print_point_budget


def parse_args():
    p = argparse.ArgumentParser(description="Compare SMART vs CAT-stage3 on NACA4 volume pressure/velocity.")
    p.add_argument("--smart-config", default="naca4")
    p.add_argument("--cat-config", default="naca4_cat")
    p.add_argument("--smart-checkpoint", default=None)
    p.add_argument("--cat-checkpoint", default=None)
    p.add_argument("--split", choices=["train", "test"], default="test")
    p.add_argument("--num-cases", type=int, default=20)
    p.add_argument("--case-ids", default=None)
    p.add_argument("--chunk-size", type=int, default=65536)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--roi", nargs=4, type=float, default=None, metavar=("XMIN", "XMAX", "YMIN", "YMAX"))
    p.add_argument("--viz-cases", type=int, default=3)
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


def predict_smart(model: SMART, geo_norm: torch.Tensor, vol_norm: torch.Tensor, chunk_size: int, device: torch.device):
    with torch.no_grad():
        inter, latent_pos = model.encode(geo_norm, None)
        out = []
        for i in range(0, vol_norm.shape[0], chunk_size):
            q = vol_norm[i:i + chunk_size].to(device).unsqueeze(0)
            pred = model.decode(inter, latent_pos, None, q)
            out.append(pred[0, :, model.surface_channels:].cpu())
        return torch.cat(out, dim=0)


def load_cat_with_ckpt(cfg, ckpt_path: str, device):
    kwargs = {
        "spatial_dim": 2,
        "surface_channels": 3,
        "volume_channels": 4,
        "parameter_channels": 0,
    }
    if "architecture" in cfg:
        kwargs.update(OmegaConf.to_container(cfg.architecture, resolve=True))
    model = CAT(**kwargs).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt["model_state_dict"]
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError:
        model.enable_stage3_encoder_lora(rank=16, alpha=16.0)
        model.load_state_dict(state, strict=True)
    model.eval()
    return model


def predict_cat_stage3_cached(model: CAT, surf_norm: torch.Tensor, vol_norm: torch.Tensor, chunk_size: int, device: torch.device):
    with torch.no_grad():
        n_pts = surf_norm.shape[1]
        anchor_idx = LoopEncoder._sample_anchor_idx(n_pts, model.geometry_encoder.anchors, surf_norm.device)
        geom_latents, anchor_pos, _ = model.encode_geometry(surf_norm, anchor_idx=anchor_idx)
        surf_latents, _, _ = model.encode_surface(surf_norm, anchor_idx=anchor_idx)
        fused = model.fusion(geom_latents, surf_latents, anchor_pos)

        out = []
        for i in range(0, vol_norm.shape[0], chunk_size):
            q = vol_norm[i:i + chunk_size].to(device).unsqueeze(0)
            dec = model.stage3_decoder(model._scale(q), fused, anchor_pos)
            pred = model.stage3_head(dec)
            out.append(pred[0].cpu())
        return torch.cat(out, dim=0)


def scatter_compare(out_path: Path, gt: np.ndarray, a: np.ndarray, b: np.ndarray, name: str, a_name: str, b_name: str, a_stats: dict, b_stats: dict):
    n = min(len(gt), 120000)
    idx = np.random.default_rng(0).integers(0, len(gt), size=n)
    g = gt[idx]
    pa = a[idx]
    pb = b[idx]

    mn = float(np.percentile(np.concatenate([g, pa, pb]), 1))
    mx = float(np.percentile(np.concatenate([g, pa, pb]), 99))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    axes[0].scatter(g, pa, s=1, alpha=0.25)
    axes[0].plot([mn, mx], [mn, mx], "r--", lw=1)
    axes[0].set_title(f"{a_name} vs GT ({name})")
    axes[0].text(0.03, 0.97,
                 f"R2={a_stats['r2']:.4f}\nMAE={a_stats['mae']:.4g}\nRMSE={a_stats['rmse']:.4g}\nMSE={a_stats['mse']:.4g}",
                 transform=axes[0].transAxes, va="top", ha="left",
                 bbox=dict(boxstyle="round", facecolor="white", alpha=0.8), fontsize=9)
    axes[0].set_xlabel("GT")
    axes[0].set_ylabel(a_name)

    axes[1].scatter(g, pb, s=1, alpha=0.25)
    axes[1].plot([mn, mx], [mn, mx], "r--", lw=1)
    axes[1].set_title(f"{b_name} vs GT ({name})")
    axes[1].text(0.03, 0.97,
                 f"R2={b_stats['r2']:.4f}\nMAE={b_stats['mae']:.4g}\nRMSE={b_stats['rmse']:.4g}\nMSE={b_stats['mse']:.4g}",
                 transform=axes[1].transAxes, va="top", ha="left",
                 bbox=dict(boxstyle="round", facecolor="white", alpha=0.8), fontsize=9)
    axes[1].set_xlabel("GT")
    axes[1].set_ylabel(b_name)

    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def hist_errors(out_path: Path, err_a: np.ndarray, err_b: np.ndarray, name: str, a_name: str, b_name: str):
    fig, ax = plt.subplots(1, 1, figsize=(7, 5), constrained_layout=True)
    bins = 120
    ax.hist(err_a, bins=bins, density=True, alpha=0.45, label=f"{a_name} |err|")
    ax.hist(err_b, bins=bins, density=True, alpha=0.45, label=f"{b_name} |err|")
    ax.set_title(f"Absolute error distribution ({name})")
    ax.set_xlabel("Absolute error")
    ax.set_ylabel("Density")
    ax.legend()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def per_case_panel(out_path: Path, xy: np.ndarray, gt: np.ndarray, p_smart: np.ndarray, p_cat: np.ndarray, title: str):
    err_s = np.abs(p_smart - gt)
    err_c = np.abs(p_cat - gt)

    vmin = float(np.percentile(np.concatenate([gt, p_smart, p_cat]), 1))
    vmax = float(np.percentile(np.concatenate([gt, p_smart, p_cat]), 99))
    emax = float(np.percentile(np.concatenate([err_s, err_c]), 99))

    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    fig.suptitle(title)

    plots = [
        (gt, "GT", "coolwarm", vmin, vmax),
        (p_smart, "SMART Pred", "coolwarm", vmin, vmax),
        (p_cat, "CAT Pred", "coolwarm", vmin, vmax),
        (err_s, "SMART |Err|", "magma", 0.0, emax),
        (err_c, "CAT |Err|", "magma", 0.0, emax),
    ]

    for i, (vals, t, cmap, lo, hi) in enumerate(plots):
        r, c = (0, i) if i < 3 else (1, i - 3)
        sc = axes[r, c].scatter(xy[:, 0], xy[:, 1], c=vals, s=2, cmap=cmap, vmin=lo, vmax=hi, linewidths=0)
        axes[r, c].set_title(t)
        axes[r, c].set_aspect("equal", adjustable="box")
        fig.colorbar(sc, ax=axes[r, c], shrink=0.8)

    axes[1, 2].axis("off")
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

    smart_kwargs = {
        "spatial_dim": 2,
        "surface_channels": 3,
        "volume_channels": 4,
        "parameter_channels": 0,
    }
    smart_kwargs.update(OmegaConf.to_container(smart_cfg.architecture, resolve=True))
    smart_model = SMART(**smart_kwargs).to(device).eval()

    smart_ckpt = args.smart_checkpoint or os.path.join("checkpoints", f"{get_model_checkpoint_name(smart_cfg)}_best.pt")
    if not os.path.isfile(smart_ckpt):
        smart_ckpt = os.path.join("checkpoints", f"{get_model_checkpoint_name(smart_cfg)}_last.pt")
    smart_model.load_state_dict(torch.load(smart_ckpt, map_location=device)["model_state_dict"])

    cat_ckpt = args.cat_checkpoint
    if cat_ckpt is None:
        stage = int(getattr(cat_cfg, "cat_stage", 3))
        base = get_model_checkpoint_name(cat_cfg) + f"-cat-stage{stage}"
        cat_ckpt = os.path.join("checkpoints", f"{base}_best.pt")
        if not os.path.isfile(cat_ckpt):
            cat_ckpt = os.path.join("checkpoints", f"{base}_last.pt")
    cat_model = load_cat_with_ckpt(cat_cfg, cat_ckpt, device)

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
        vol_norm = (vol_xy - dataset.min_pos) / (dataset.max_pos - dataset.min_pos)

        pred_smart_n = predict_smart(smart_model, surf_norm, vol_norm, args.chunk_size, device)
        pred_cat_n = predict_cat_stage3_cached(cat_model, surf_norm, vol_norm, args.chunk_size, device)

        pred_smart = (pred_smart_n * dataset.std_vol_data + dataset.mean_vol_data).numpy()
        pred_cat = (pred_cat_n * dataset.std_vol_data + dataset.mean_vol_data).numpy()
        gt = vol_gt.numpy()

        gt_p = gt[:, 0]
        s_p = pred_smart[:, 0]
        c_p = pred_cat[:, 0]

        gt_v = gt[:, 2:4]
        s_v = pred_smart[:, 2:4]
        c_v = pred_cat[:, 2:4]

        gt_speed = np.linalg.norm(gt_v, axis=1)
        s_speed = np.linalg.norm(s_v, axis=1)
        c_speed = np.linalg.norm(c_v, axis=1)

        m_sp = metrics(s_p, gt_p)
        m_cp = metrics(c_p, gt_p)
        m_sv = metrics(s_v, gt_v)
        m_cv = metrics(c_v, gt_v)
        m_ss = metrics(s_speed, gt_speed)
        m_cs = metrics(c_speed, gt_speed)

        rows.append({
            "case_id": case_id,
            "n_vol_points": int(vol_xy.shape[0]),
            "smart_pressure_rel_l2": m_sp["rel_l2"],
            "cat_pressure_rel_l2": m_cp["rel_l2"],
            "smart_velocity_rel_l2": m_sv["rel_l2"],
            "cat_velocity_rel_l2": m_cv["rel_l2"],
            "smart_speed_rel_l2": m_ss["rel_l2"],
            "cat_speed_rel_l2": m_cs["rel_l2"],
            "delta_pressure_rel_l2_cat_minus_smart": m_cp["rel_l2"] - m_sp["rel_l2"],
            "delta_velocity_rel_l2_cat_minus_smart": m_cv["rel_l2"] - m_sv["rel_l2"],
        })

        all_gt_p.append(gt_p); all_smart_p.append(s_p); all_cat_p.append(c_p)
        all_gt_speed.append(gt_speed); all_smart_speed.append(s_speed); all_cat_speed.append(c_speed)

        if i < args.viz_cases:
            case_dir = out_root / f"case_{case_id}"
            case_dir.mkdir(parents=True, exist_ok=True)
            per_case_panel(case_dir / "pressure_panel.png", vol_xy.numpy(), gt_p, s_p, c_p, f"Case {case_id} pressure")
            per_case_panel(case_dir / "speed_panel.png", vol_xy.numpy(), gt_speed, s_speed, c_speed, f"Case {case_id} speed")

    if not rows:
        raise RuntimeError("No valid cases processed.")

    gt_p = np.concatenate(all_gt_p)
    smart_p = np.concatenate(all_smart_p)
    cat_p = np.concatenate(all_cat_p)
    gt_speed = np.concatenate(all_gt_speed)
    smart_speed = np.concatenate(all_smart_speed)
    cat_speed = np.concatenate(all_cat_speed)

    smart_pressure_stats = metrics(smart_p, gt_p)
    cat_pressure_stats = metrics(cat_p, gt_p)
    smart_speed_stats = metrics(smart_speed, gt_speed)
    cat_speed_stats = metrics(cat_speed, gt_speed)

    scatter_compare(out_root / "pressure_parity.png", gt_p, smart_p, cat_p, "pressure", "SMART", "CAT", smart_pressure_stats, cat_pressure_stats)
    scatter_compare(out_root / "speed_parity.png", gt_speed, smart_speed, cat_speed, "speed", "SMART", "CAT", smart_speed_stats, cat_speed_stats)
    hist_errors(out_root / "pressure_abs_error_hist.png", np.abs(smart_p - gt_p), np.abs(cat_p - gt_p), "pressure", "SMART", "CAT")
    hist_errors(out_root / "speed_abs_error_hist.png", np.abs(smart_speed - gt_speed), np.abs(cat_speed - gt_speed), "speed", "SMART", "CAT")

    arr_dp = np.array([r["delta_pressure_rel_l2_cat_minus_smart"] for r in rows], dtype=float)
    arr_dv = np.array([r["delta_velocity_rel_l2_cat_minus_smart"] for r in rows], dtype=float)

    summary = {
        "num_cases": len(rows),
        "pressure_delta_rel_l2_mean": float(arr_dp.mean()),
        "pressure_delta_rel_l2_ci95": bootstrap_ci(arr_dp),
        "velocity_delta_rel_l2_mean": float(arr_dv.mean()),
        "velocity_delta_rel_l2_ci95": bootstrap_ci(arr_dv),
        "pressure_cat_better_case_fraction": float(np.mean(arr_dp < 0)),
        "velocity_cat_better_case_fraction": float(np.mean(arr_dv < 0)),
        "smart_pressure_rel_l2_global": rel_l2(smart_p, gt_p),
        "cat_pressure_rel_l2_global": rel_l2(cat_p, gt_p),
        "smart_velocity_rel_l2_global": rel_l2(np.concatenate(all_smart_speed), np.concatenate(all_gt_speed)),
        "cat_velocity_rel_l2_global": rel_l2(np.concatenate(all_cat_speed), np.concatenate(all_gt_speed)),
        "smart_pressure_stats": smart_pressure_stats,
        "cat_pressure_stats": cat_pressure_stats,
        "smart_speed_stats": smart_speed_stats,
        "cat_speed_stats": cat_speed_stats,
        "smart_ckpt": smart_ckpt,
        "cat_ckpt": cat_ckpt,
    }

    with open(out_root / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    import csv
    with open(out_root / "per_case.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"Saved comparison outputs to: {out_root}")


if __name__ == "__main__":
    main()
