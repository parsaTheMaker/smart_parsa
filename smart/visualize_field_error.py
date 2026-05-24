#!/usr/bin/env python3
"""Visualize NACA4 field errors over a region of interest.

The script loads a trained SMART checkpoint, evaluates the model on every
surface and volume point inside the requested ROI, chunks the queries using the
same scale as the training configuration, merges the chunked predictions back
into full fields, and saves detailed comparison plots plus numeric summaries.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LogNorm
from omegaconf import OmegaConf
from tqdm.auto import tqdm

from data.naca4_dataset import NACA4Dataset
from models.smart.smart import SMART
from utils.utils import get_model_checkpoint_name, apply_naca4_auto_point_budget, print_point_budget


FULL_SURFACE_FIELDS = ["pressure", "normal_x", "normal_y"]
FULL_VOLUME_FIELDS = ["pressure", "sdf", "velocity_x", "velocity_y"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize NACA4 field errors in a ROI.")
    parser.add_argument("--config-name", default="naca4", help="Hydra-style config name under smart/config.")
    parser.add_argument("--checkpoint", default=None, help="Optional explicit checkpoint path.")
    parser.add_argument("--split", default="test", choices=["train", "test"], help="Which split to visualize.")
    parser.add_argument("--num-cases", type=int, default=5, help="Number of cases to visualize.")
    parser.add_argument("--case-ids", default=None, help="Comma-separated explicit case ids to visualize.")
    parser.add_argument(
        "--roi",
        nargs=4,
        type=float,
        default=[-2.0, 2.0, -2.0, 2.0],
        metavar=("XMIN", "XMAX", "YMIN", "YMAX"),
        help="ROI bounds in raw coordinates.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Query chunk size. Defaults to the larger training query count.",
    )
    parser.add_argument("--output-dir", default=None, help="Where to save figures and arrays.")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic sampling seed for geometry subsampling.")
    return parser.parse_args()


def load_config(config_name: str):
    config_path = Path(__file__).resolve().parent / "config" / f"{config_name}.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    return OmegaConf.load(config_path)


def initialize_gpu(random_seed: int, high_precision: bool = False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if high_precision and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")

    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)

    return device


def count_model_params(model):
    params = []
    for p in model.parameters():
        if p.requires_grad:
            params.append(2 * p.numel() if torch.is_complex(p) else p.numel())
    return sum(params)


def normalize_positions(pos: torch.Tensor, min_pos: torch.Tensor, max_pos: torch.Tensor) -> torch.Tensor:
    return (pos - min_pos) / (max_pos - min_pos)


def chunk_indices(n: int, chunk_size: int) -> Iterable[Tuple[int, int]]:
    for start in range(0, n, chunk_size):
        yield start, min(start + chunk_size, n)


def stable_case_seed(case_id: str) -> int:
    """Return a deterministic seed for a case id across Python processes."""
    import zlib

    return zlib.adler32(case_id.encode("utf-8")) & 0xFFFFFFFF


def select_geometry(geo_points: torch.Tensor, target_points: int, seed: int) -> torch.Tensor:
    if target_points <= 0 or target_points >= geo_points.shape[0]:
        return geo_points

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    idx = torch.randint(0, geo_points.shape[0], (target_points,), generator=generator)
    return geo_points[idx]


def load_case_raw(dataset, case_id: str):
    geo_mesh, surf_mesh, surf_data, vol_mesh, vol_data = dataset._load_case_arrays(case_id, write_cache=True)
    return {
        "geo_mesh": geo_mesh,
        "surf_mesh": surf_mesh,
        "surf_data": surf_data,
        "vol_mesh": vol_mesh,
        "vol_data": vol_data,
    }


def roi_mask(points: torch.Tensor, roi: Sequence[float]) -> torch.Tensor:
    xmin, xmax, ymin, ymax = roi
    return (
        (points[:, 0] >= xmin)
        & (points[:, 0] <= xmax)
        & (points[:, 1] >= ymin)
        & (points[:, 1] <= ymax)
    )


def predict_chunked(
    model: SMART,
    intermediate_latent_geometries,
    latent_geo_pos: torch.Tensor,
    query_points: torch.Tensor,
    field_kind: str,
    chunk_size: int,
    device: torch.device,
) -> torch.Tensor:
    if query_points.numel() == 0:
        if field_kind == "surface":
            return query_points.new_empty((0, model.surface_channels))
        return query_points.new_empty((0, model.volume_channels))

    outputs: List[torch.Tensor] = []
    for start, end in chunk_indices(query_points.shape[0], chunk_size):
        chunk = query_points[start:end].to(device).unsqueeze(0)
        pred = model.decode(intermediate_latent_geometries, latent_geo_pos, None, chunk)
        if field_kind == "surface":
            outputs.append(pred[0, :, :model.surface_channels].detach().cpu())
        elif field_kind == "volume":
            outputs.append(pred[0, :, model.surface_channels :].detach().cpu())
        else:
            raise ValueError(f"Unknown field kind: {field_kind}")
    return torch.cat(outputs, dim=0)


def field_metrics(y_hat: np.ndarray, y: np.ndarray, eps: float = 1e-8) -> dict:
    abs_err = np.abs(y_hat - y)
    rel_err = abs_err / np.maximum(np.abs(y), eps)
    return {
        "mae": float(abs_err.mean()),
        "rmse": float(np.sqrt(np.mean((y_hat - y) ** 2))),
        "rel_l2": float(np.linalg.norm(y_hat - y) / max(np.linalg.norm(y), eps)),
        "median_abs_err": float(np.median(abs_err)),
        "p95_abs_err": float(np.percentile(abs_err, 95)),
        "max_abs_err": float(abs_err.max()),
        "median_rel_err": float(np.median(rel_err)),
        "p95_rel_err": float(np.percentile(rel_err, 95)),
        "max_rel_err": float(rel_err.max()),
    }


def robust_limits(values: np.ndarray, low: float = 1.0, high: float = 99.0, symmetric: bool = False) -> Tuple[float, float]:
    if values.size == 0:
        return -1.0, 1.0
    if symmetric:
        limit = np.percentile(np.abs(values), high)
        if not np.isfinite(limit) or limit == 0:
            limit = 1.0
        return -float(limit), float(limit)
    vmin = np.percentile(values, low)
    vmax = np.percentile(values, high)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        center = float(np.mean(values))
        spread = float(np.std(values))
        if spread == 0:
            spread = 1.0
        return center - spread, center + spread
    return float(vmin), float(vmax)


def scatter_panel(
    ax,
    xy: np.ndarray,
    values: np.ndarray,
    title: str,
    cmap: str,
    roi: Sequence[float],
    norm=None,
    vmin=None,
    vmax=None,
):
    scatter_kwargs = {
        "c": values,
        "s": 4,
        "cmap": cmap,
        "linewidths": 0,
        "rasterized": True,
    }
    if norm is not None:
        scatter_kwargs["norm"] = norm
    else:
        scatter_kwargs["vmin"] = vmin
        scatter_kwargs["vmax"] = vmax

    sc = ax.scatter(xy[:, 0], xy[:, 1], **scatter_kwargs)
    ax.set_title(title, fontsize=10)
    ax.set_xlim(float(roi[0]), float(roi[1]))
    ax.set_ylim(float(roi[2]), float(roi[3]))
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.15, linewidth=0.4)
    return sc


def save_field_figure(
    out_path: Path,
    title: str,
    xy: np.ndarray,
    gt: np.ndarray,
    pred: np.ndarray,
    field_name: str,
    field_units: str,
    roi: Sequence[float],
):
    abs_err = np.abs(pred - gt)
    rel_err = abs_err / np.maximum(np.abs(gt), 1e-8)
    gt_vmin, gt_vmax = robust_limits(np.concatenate([gt, pred]), symmetric=False)

    positive_abs = abs_err[abs_err > 0]
    err_vmin = max(np.percentile(positive_abs, 5) if positive_abs.size else 1e-8, 1e-8)
    err_vmax = float(np.percentile(abs_err, 99)) if abs_err.size else 1.0
    if not np.isfinite(err_vmax) or err_vmax <= err_vmin:
        err_vmax = err_vmin * 10.0

    positive_rel = rel_err[rel_err > 0]
    rel_vmin = max(np.percentile(positive_rel, 5) if positive_rel.size else 1e-8, 1e-8)
    rel_vmax = float(np.percentile(rel_err, 99)) if rel_err.size else 1.0
    if not np.isfinite(rel_vmax) or rel_vmax <= rel_vmin:
        rel_vmax = rel_vmin * 10.0

    fig, axes = plt.subplots(1, 4, figsize=(20, 5), constrained_layout=True)
    fig.suptitle(title, fontsize=14)

    sc0 = scatter_panel(axes[0], xy, gt, f"{field_name} GT", cmap="coolwarm", roi=roi, vmin=gt_vmin, vmax=gt_vmax)
    sc1 = scatter_panel(axes[1], xy, pred, f"{field_name} Pred", cmap="coolwarm", roi=roi, vmin=gt_vmin, vmax=gt_vmax)
    sc2 = scatter_panel(
        axes[2],
        xy,
        abs_err,
        f"{field_name} Abs Err",
        cmap="magma",
        roi=roi,
        norm=LogNorm(vmin=err_vmin, vmax=err_vmax),
    )
    sc3 = scatter_panel(
        axes[3],
        xy,
        rel_err,
        f"{field_name} Rel Err",
        cmap="viridis",
        roi=roi,
        norm=LogNorm(vmin=rel_vmin, vmax=rel_vmax),
    )

    for ax, sc, label in [
        (axes[0], sc0, f"{field_name} [{field_units}]"),
        (axes[1], sc1, f"{field_name} [{field_units}]"),
        (axes[2], sc2, f"Abs err [{field_units}]"),
        (axes[3], sc3, "Rel err"),
    ]:
        cbar = fig.colorbar(sc, ax=ax, shrink=0.82)
        cbar.ax.tick_params(labelsize=8)
        cbar.set_label(label, fontsize=9)
        ax.tick_params(labelsize=8)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def save_group_overview_figure(
    out_path: Path,
    title: str,
    xy: np.ndarray,
    gt: np.ndarray,
    pred: np.ndarray,
    field_names: Sequence[str],
    roi: Sequence[float],
):
    n_fields = len(field_names)
    fig, axes = plt.subplots(n_fields, 4, figsize=(20, 4.8 * n_fields), constrained_layout=True)
    if n_fields == 1:
        axes = np.expand_dims(axes, axis=0)
    fig.suptitle(title, fontsize=15)

    for row_idx, field_name in enumerate(field_names):
        gt_field = gt[:, row_idx]
        pred_field = pred[:, row_idx]
        abs_err = np.abs(pred_field - gt_field)
        rel_err = abs_err / np.maximum(np.abs(gt_field), 1e-8)

        gt_vmin, gt_vmax = robust_limits(np.concatenate([gt_field, pred_field]), symmetric=False)
        positive_abs = abs_err[abs_err > 0]
        err_vmin = max(float(np.percentile(positive_abs, 5)) if positive_abs.size else 1e-8, 1e-8)
        err_vmax = float(np.percentile(abs_err, 99)) if abs_err.size else 1.0
        if not np.isfinite(err_vmax) or err_vmax <= err_vmin:
            err_vmax = err_vmin * 10.0

        positive_rel = rel_err[rel_err > 0]
        rel_vmin = max(float(np.percentile(positive_rel, 5)) if positive_rel.size else 1e-8, 1e-8)
        rel_vmax = float(np.percentile(rel_err, 99)) if rel_err.size else 1.0
        if not np.isfinite(rel_vmax) or rel_vmax <= rel_vmin:
            rel_vmax = rel_vmin * 10.0

        panels = [
            (gt_field, f"{field_name} GT", "coolwarm", None),
            (pred_field, f"{field_name} Pred", "coolwarm", None),
            (abs_err, f"{field_name} Abs Err", "magma", LogNorm(vmin=err_vmin, vmax=err_vmax)),
            (rel_err, f"{field_name} Rel Err", "viridis", LogNorm(vmin=rel_vmin, vmax=rel_vmax)),
        ]
        for col_idx, (values, panel_title, cmap, norm) in enumerate(panels):
            ax = axes[row_idx, col_idx]
            if norm is None:
                vmin, vmax = gt_vmin, gt_vmax
                sc = scatter_panel(ax, xy, values, panel_title, cmap=cmap, roi=roi, vmin=vmin, vmax=vmax)
            else:
                sc = scatter_panel(ax, xy, values, panel_title, cmap=cmap, roi=roi, norm=norm)
            fig.colorbar(sc, ax=ax, shrink=0.78)
            ax.tick_params(labelsize=7)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def save_combined_study_figure(
    out_path: Path,
    title: str,
    surf_xy: np.ndarray,
    surf_gt: np.ndarray,
    surf_pred: np.ndarray,
    vol_xy: np.ndarray,
    vol_gt: np.ndarray,
    vol_pred: np.ndarray,
    roi: Sequence[float],
):
    all_rows = [
        ("surface", "pressure", surf_xy, surf_gt[:, 0], surf_pred[:, 0]),
        ("surface", "normal_x", surf_xy, surf_gt[:, 1], surf_pred[:, 1]),
        ("surface", "normal_y", surf_xy, surf_gt[:, 2], surf_pred[:, 2]),
        ("volume", "pressure", vol_xy, vol_gt[:, 0], vol_pred[:, 0]),
        ("volume", "sdf", vol_xy, vol_gt[:, 1], vol_pred[:, 1]),
        ("volume", "velocity_x", vol_xy, vol_gt[:, 2], vol_pred[:, 2]),
        ("volume", "velocity_y", vol_xy, vol_gt[:, 3], vol_pred[:, 3]),
    ]

    fig, axes = plt.subplots(len(all_rows), 4, figsize=(20, 3.6 * len(all_rows)), constrained_layout=True)
    fig.suptitle(title, fontsize=16)
    fig.text(0.005, 0.78, "Surface", rotation=90, va="center", ha="left", fontsize=14, weight="bold")
    fig.text(0.005, 0.28, "Volume", rotation=90, va="center", ha="left", fontsize=14, weight="bold")

    for row_idx, (group_name, field_name, xy, gt, pred) in enumerate(all_rows):
        abs_err = np.abs(pred - gt)
        rel_err = abs_err / np.maximum(np.abs(gt), 1e-8)

        gt_vmin, gt_vmax = robust_limits(np.concatenate([gt, pred]), symmetric=False)
        positive_abs = abs_err[abs_err > 0]
        err_vmin = max(float(np.percentile(positive_abs, 5)) if positive_abs.size else 1e-8, 1e-8)
        err_vmax = float(np.percentile(abs_err, 99)) if abs_err.size else 1.0
        if not np.isfinite(err_vmax) or err_vmax <= err_vmin:
            err_vmax = err_vmin * 10.0

        positive_rel = rel_err[rel_err > 0]
        rel_vmin = max(float(np.percentile(positive_rel, 5)) if positive_rel.size else 1e-8, 1e-8)
        rel_vmax = float(np.percentile(rel_err, 99)) if rel_err.size else 1.0
        if not np.isfinite(rel_vmax) or rel_vmax <= rel_vmin:
            rel_vmax = rel_vmin * 10.0

        panels = [
            (gt, f"{group_name} {field_name} GT", "coolwarm", None),
            (pred, f"{group_name} {field_name} Pred", "coolwarm", None),
            (abs_err, f"{group_name} {field_name} Abs Err", "magma", LogNorm(vmin=err_vmin, vmax=err_vmax)),
            (rel_err, f"{group_name} {field_name} Rel Err", "viridis", LogNorm(vmin=rel_vmin, vmax=rel_vmax)),
        ]
        for col_idx, (values, panel_title, cmap, norm) in enumerate(panels):
            ax = axes[row_idx, col_idx]
            if norm is None:
                sc = scatter_panel(ax, xy, values, panel_title, cmap=cmap, roi=roi, vmin=gt_vmin, vmax=gt_vmax)
            else:
                sc = scatter_panel(ax, xy, values, panel_title, cmap=cmap, roi=roi, norm=norm)
            fig.colorbar(sc, ax=ax, shrink=0.76)
            ax.tick_params(labelsize=7)
            if col_idx == 0:
                ax.set_ylabel(field_name, fontsize=10, rotation=0, labelpad=42, va="center")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def save_overview_figure(*args, **kwargs):
    return save_group_overview_figure(*args, **kwargs)


def roi_tag(roi: Sequence[float]) -> str:
    def fmt(value: float) -> str:
        text = f"{value:.2f}".rstrip("0").rstrip(".")
        return text.replace("-", "m").replace(".", "p")

    return f"x{fmt(roi[0])}_{fmt(roi[1])}_y{fmt(roi[2])}_{fmt(roi[3])}"


def main():
    args = parse_args()
    cfg = load_config(args.config_name)
    config = cfg.experiment

    if args.checkpoint is not None:
        config.checkpoint = args.checkpoint
    if args.output_dir is not None:
        config.output_dir = args.output_dir
    else:
        config.output_dir = None

    device = initialize_gpu(config.random_seed, high_precision=False)

    train_data = NACA4Dataset(
        config.data_path,
        if_test=False,
        geometry_points=int(config.num_body_points),
        surface_points=int(config.num_surface_points),
        volume_points=int(config.num_volume_points),
        scale_positions=bool(config.scale_positions),
        manifest_variant=getattr(config, "manifest_variant", "full"),
    )

    point_info = apply_naca4_auto_point_budget(config, train_data, for_cat=False)
    if point_info is not None:
        print_point_budget("SMART-VIZ", point_info)

    train_data = NACA4Dataset(
        config.data_path,
        if_test=False,
        geometry_points=int(config.num_body_points),
        surface_points=int(config.num_surface_points),
        volume_points=int(config.num_volume_points),
        scale_positions=bool(config.scale_positions),
        manifest_variant=getattr(config, "manifest_variant", "full"),
    )
    test_data = NACA4Dataset(
        config.data_path,
        if_test=True,
        geometry_points=int(config.num_body_points),
        surface_points=int(config.num_surface_points),
        volume_points=int(config.num_volume_points),
        scale_positions=bool(config.scale_positions),
        manifest_variant=getattr(config, "manifest_variant", "full"),
    )
    spatial_dim = 2
    surf_channels = len(SURFACE_FIELDS)
    vol_channels = len(VOLUME_FIELDS)
    params_dim = 0

    model_kwargs = {
        "spatial_dim": spatial_dim,
        "surface_channels": surf_channels,
        "volume_channels": vol_channels,
        "parameter_channels": params_dim,
    }
    if "architecture" in config:
        model_kwargs.update(OmegaConf.to_container(config.architecture, resolve=True))

    model = SMART(**model_kwargs).to(device)
    model.eval()

    checkpoint_name = get_model_checkpoint_name(config)
    checkpoint_path = args.checkpoint or os.path.join("checkpoints", f"{checkpoint_name}_best.pt")
    if not os.path.isfile(checkpoint_path):
        fallback_last = os.path.join("checkpoints", f"{checkpoint_name}_last.pt")
        if os.path.isfile(fallback_last):
            checkpoint_path = fallback_last
        else:
            raise FileNotFoundError(f"No checkpoint found at {checkpoint_path} or {fallback_last}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"Total parameters: {count_model_params(model)}")

    if model.surface_channels == 1:
        surface_fields = ["pressure"]
    else:
        surface_fields = FULL_SURFACE_FIELDS[:model.surface_channels]

    if model.volume_channels == 2:
        volume_fields = ["velocity_x", "velocity_y"]
        vol_channel_idx = [2, 3]  # from raw/full volume targets
    else:
        volume_fields = FULL_VOLUME_FIELDS[:model.volume_channels]
        vol_channel_idx = list(range(model.volume_channels))

    print(f"Visualization fields: surface={surface_fields}, volume={volume_fields}")

    dataset = test_data if args.split == "test" else train_data
    case_ids = list(dataset.data)
    if args.case_ids:
        wanted = [c.strip() for c in args.case_ids.split(",") if c.strip()]
        case_ids = wanted
    else:
        case_ids = case_ids[: args.num_cases]

    query_chunk_size = args.chunk_size or max(int(config.num_volume_points), 1)
    roi_specs = [("full", [-5.0, 5.0, -5.0, 5.0]), ("roi", args.roi)]

    root_dir = Path(config.output_dir or os.path.join("results", "field_error", config.dataset, checkpoint_name))
    root_dir.mkdir(parents=True, exist_ok=True)

    outputs = []
    with torch.inference_mode():
        for tag, roi in roi_specs:
            out_root = root_dir / f"split_{args.split}_{tag}_{roi_tag(roi)}_n{len(case_ids)}"
            out_root.mkdir(parents=True, exist_ok=True)

            summary_rows = []
            for case_id in tqdm(case_ids, desc=f"Cases [{tag}]", dynamic_ncols=True):
                raw = load_case_raw(dataset, case_id)
                geo_mesh = raw["geo_mesh"]
                surf_mesh = raw["surf_mesh"]
                surf_data = raw["surf_data"]
                vol_mesh = raw["vol_mesh"]
                vol_data = raw["vol_data"]

                surf_roi_mask = roi_mask(surf_mesh, roi)
                vol_roi_mask = roi_mask(vol_mesh, roi)

                surf_xy = surf_mesh[surf_roi_mask]
                surf_gt = surf_data[surf_roi_mask][:, :model.surface_channels]
                vol_xy = vol_mesh[vol_roi_mask]
                vol_gt = vol_data[vol_roi_mask][:, vol_channel_idx]

                if surf_xy.numel() == 0 or vol_xy.numel() == 0:
                    print(f"Skipping {case_id} [{tag}]: no points found in ROI {roi}")
                    continue

                geo_sample = select_geometry(
                    geo_mesh,
                    int(config.num_body_points),
                    seed=int(config.random_seed) + stable_case_seed(case_id),
                )

                geo_norm = normalize_positions(geo_sample, dataset.min_pos, dataset.max_pos).to(device).unsqueeze(0)
                intermediate_latent_geometries, latent_geo_pos = model.encode(geo_norm, None)

                surf_pred_norm = predict_chunked(
                    model,
                    intermediate_latent_geometries,
                    latent_geo_pos,
                    normalize_positions(surf_xy, dataset.min_pos, dataset.max_pos),
                    field_kind="surface",
                    chunk_size=query_chunk_size,
                    device=device,
                )
                vol_pred_norm = predict_chunked(
                    model,
                    intermediate_latent_geometries,
                    latent_geo_pos,
                    normalize_positions(vol_xy, dataset.min_pos, dataset.max_pos),
                    field_kind="volume",
                    chunk_size=query_chunk_size,
                    device=device,
                )

                surf_pred = (surf_pred_norm * dataset.std_surf_data[:model.surface_channels] + dataset.mean_surf_data[:model.surface_channels]).numpy()
                surf_gt_np = surf_gt.numpy()
                vol_pred = (vol_pred_norm * dataset.std_vol_data[vol_channel_idx] + dataset.mean_vol_data[vol_channel_idx]).numpy()
                vol_gt_np = vol_gt.numpy()

                surface_field_metrics = {
                    field_name: field_metrics(surf_pred[:, idx], surf_gt_np[:, idx])
                    for idx, field_name in enumerate(SURFACE_FIELDS)
                }
                volume_field_metrics = {
                    field_name: field_metrics(vol_pred[:, idx], vol_gt_np[:, idx])
                    for idx, field_name in enumerate(VOLUME_FIELDS)
                }
                surface_normals_metrics = field_metrics(surf_pred[:, 1:], surf_gt_np[:, 1:])
                volume_velocity_metrics = field_metrics(vol_pred[:, 2:], vol_gt_np[:, 2:])
                speed_pred = np.linalg.norm(vol_pred[:, 2:], axis=1)
                speed_gt = np.linalg.norm(vol_gt_np[:, 2:], axis=1)
                speed_metrics = field_metrics(speed_pred, speed_gt)

                case_dir = out_root / f"case_{case_id}"
                case_dir.mkdir(parents=True, exist_ok=True)

                np.savez_compressed(
                    case_dir / "fields.npz",
                    surf_xy=surf_xy.numpy(),
                    surf_gt=surf_gt_np,
                    surf_pred=surf_pred,
                    vol_xy=vol_xy.numpy(),
                    vol_gt=vol_gt_np,
                    vol_pred=vol_pred,
                    surface_fields=np.array(surface_fields),
                    volume_fields=np.array(volume_fields),
                    roi=np.array(roi, dtype=np.float32),
                    chunk_size=np.array([query_chunk_size], dtype=np.int64),
                    checkpoint=np.array([checkpoint_path]),
                )

                with open(case_dir / "metrics.json", "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "case_id": case_id,
                            "roi": roi,
                            "chunk_size": query_chunk_size,
                            "surface": surface_field_metrics,
                            "surface_normals": surface_normals_metrics,
                            "volume": volume_field_metrics,
                            "volume_velocity": volume_velocity_metrics,
                            "speed": speed_metrics,
                        },
                        f,
                        indent=2,
                    )

                for idx, field_name in enumerate(surface_fields):
                    save_field_figure(
                        case_dir / f"surface_{field_name}_panel.png",
                        f"Case {case_id} - Surface {field_name} over {tag}",
                        surf_xy.numpy(),
                        surf_gt_np[:, idx],
                        surf_pred[:, idx],
                        field_name=f"surface/{field_name}",
                        field_units=("Pa" if field_name == "pressure" else "unitless"),
                        roi=roi,
                    )
                for idx, field_name in enumerate(volume_fields):
                    save_field_figure(
                        case_dir / f"volume_{field_name}_panel.png",
                        f"Case {case_id} - Volume {field_name} over {tag}",
                        vol_xy.numpy(),
                        vol_gt_np[:, idx],
                        vol_pred[:, idx],
                        field_name=f"volume/{field_name}",
                        field_units=("Pa" if field_name == "pressure" else ("distance" if field_name == "sdf" else "unitless")),
                        roi=roi,
                    )
                save_field_figure(
                    case_dir / "velocity_speed_panel.png",
                    f"Case {case_id} - Velocity speed over {tag}",
                    vol_xy.numpy(),
                    speed_gt,
                    speed_pred,
                    field_name="|velocity|",
                    field_units="m/s",
                    roi=roi,
                )
                save_group_overview_figure(
                    case_dir / "surface_overview_panel.png",
                    f"Case {case_id} - Surface overview ({tag})",
                    surf_xy.numpy(),
                    surf_gt_np,
                    surf_pred,
                    surface_fields,
                    roi=roi,
                )
                save_group_overview_figure(
                    case_dir / "volume_overview_panel.png",
                    f"Case {case_id} - Volume overview ({tag})",
                    vol_xy.numpy(),
                    vol_gt_np,
                    vol_pred,
                    volume_fields,
                    roi=roi,
                )
                if model.surface_channels >= 3 and model.volume_channels >= 4:
                    save_combined_study_figure(
                        case_dir / "study_panel.png",
                        f"Case {case_id} - Combined study ({tag})",
                        surf_xy.numpy(),
                        surf_gt_np,
                        surf_pred,
                        vol_xy.numpy(),
                        vol_gt_np,
                        vol_pred,
                        roi=roi,
                    )

                summary_row = {
                    "case_id": case_id,
                    "surf_points": int(surf_xy.shape[0]),
                    "vol_points": int(vol_xy.shape[0]),
                }
                for field_name in surface_fields:
                    summary_row[f"surface_{field_name}_mae"] = surface_field_metrics[field_name]["mae"]
                    summary_row[f"surface_{field_name}_rel_l2"] = surface_field_metrics[field_name]["rel_l2"]
                for field_name in volume_fields:
                    summary_row[f"volume_{field_name}_mae"] = volume_field_metrics[field_name]["mae"]
                    summary_row[f"volume_{field_name}_rel_l2"] = volume_field_metrics[field_name]["rel_l2"]
                if "normal_x" in surface_fields and "normal_y" in surface_fields:
                    summary_row["surface_normals_rel_l2"] = surface_normals_metrics["rel_l2"]
                if "velocity_x" in volume_fields and "velocity_y" in volume_fields:
                    summary_row["volume_velocity_rel_l2"] = volume_velocity_metrics["rel_l2"]
                    summary_row["speed_mae"] = speed_metrics["mae"]
                    summary_row["speed_rel_l2"] = speed_metrics["rel_l2"]
                summary_rows.append(summary_row)

            with open(out_root / "summary.json", "w", encoding="utf-8") as f:
                json.dump(summary_rows, f, indent=2)

            with open(out_root / "summary.csv", "w", encoding="utf-8") as f:
                headers = list(summary_rows[0].keys()) if summary_rows else ["case_id"]
                f.write(",".join(headers) + "\n")
                for row in summary_rows:
                    f.write(",".join(str(row.get(h, "")) for h in headers) + "\n")

            outputs.append((tag, str(out_root)))

    for tag, path in outputs:
        print(f"Saved {tag} outputs to: {path}")


if __name__ == "__main__":
    main()
