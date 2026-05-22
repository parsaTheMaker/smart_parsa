#!/usr/bin/env python3
"""Visualize CAT field errors for NACA4, stage by stage.

Stage 1: geometry pretraining outputs surface normals + volume sdf
Stage 2: surface-field pretraining outputs surface pressure
Stage 3: volume training outputs pressure, sdf, velocity_x, velocity_y
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
from models.smart.cat import CAT
from utils.utils import get_model_checkpoint_name


SURFACE_FIELDS = ["pressure", "normal_x", "normal_y"]
VOLUME_FIELDS = ["pressure", "sdf", "velocity_x", "velocity_y"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize CAT field errors for NACA4 stage checkpoints.")
    parser.add_argument("--config-name", default="naca4_cat", help="Hydra-style config name under smart/config.")
    parser.add_argument("--cat-stage", type=int, default=None, choices=[1, 2, 3], help="Which CAT stage checkpoint to visualize. Defaults to config.cat_stage.")
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
    parser.add_argument("--chunk-size", type=int, default=None, help="Query chunk size. Defaults to the model subregion size.")
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


def normalize_positions(pos: torch.Tensor, min_pos: torch.Tensor, max_pos: torch.Tensor) -> torch.Tensor:
    return (pos - min_pos) / (max_pos - min_pos)


def to_model_positions(pos: torch.Tensor, min_pos: torch.Tensor, max_pos: torch.Tensor) -> torch.Tensor:
    # Dataset __getitem__ uses normalized [0,1] positions, but this visualizer may operate on raw cached coordinates.
    # Normalize only when values are outside the normalized range.
    pmin = float(pos.min()) if pos.numel() > 0 else 0.0
    pmax = float(pos.max()) if pos.numel() > 0 else 1.0
    if pmin < -1e-4 or pmax > 1.0001:
        return normalize_positions(pos, min_pos, max_pos)
    return pos




def sample_indices(n: int, k: int, generator: torch.Generator, disjoint_from: torch.Tensor | None = None) -> torch.Tensor:
    if k <= 0:
        return torch.empty((0,), dtype=torch.long)

    if disjoint_from is not None:
        mask = torch.ones((n,), dtype=torch.bool)
        mask[disjoint_from] = False
        candidate = torch.where(mask)[0]
        if candidate.numel() == 0:
            return torch.randint(0, n, (k,), generator=generator)
        # Pure random subsampling with replacement from the admissible set.
        pick = torch.randint(0, candidate.numel(), (k,), generator=generator)
        return candidate[pick]

    # Pure random subsampling with replacement.
    return torch.randint(0, n, (k,), generator=generator)


def make_case_generator(base_seed: int, case_id: str) -> torch.Generator:
    g = torch.Generator(device='cpu')
    g.manual_seed(int(base_seed) + stable_case_seed(case_id))
    return g

def chunk_indices(n: int, chunk_size: int) -> Iterable[Tuple[int, int]]:
    for start in range(0, n, chunk_size):
        yield start, min(start + chunk_size, n)


def stable_case_seed(case_id: str) -> int:
    import zlib

    return zlib.adler32(case_id.encode("utf-8")) & 0xFFFFFFFF


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


def robust_limits(values: np.ndarray, low: float = 1.0, high: float = 99.0) -> Tuple[float, float]:
    if values.size == 0:
        return -1.0, 1.0
    vmin = np.percentile(values, low)
    vmax = np.percentile(values, high)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        center = float(np.mean(values))
        spread = float(np.std(values))
        if spread == 0:
            spread = 1.0
        return center - spread, center + spread
    return float(vmin), float(vmax)


def roi_tag(roi: Sequence[float]) -> str:
    def fmt(value: float) -> str:
        text = f"{value:.2f}".rstrip("0").rstrip(".")
        return text.replace("-", "m").replace(".", "p")

    return f"x{fmt(roi[0])}_{fmt(roi[1])}_y{fmt(roi[2])}_{fmt(roi[3])}"


def candidate_checkpoint_stems(config, stage: int):
    model_name_variants = []
    for name in [getattr(config, "model_name", "CAT")]:
        if name not in model_name_variants:
            model_name_variants.append(name)
        lower = str(name).lower()
        upper = str(name).upper()
        for alt in (lower, upper):
            if alt not in model_name_variants:
                model_name_variants.append(alt)

    tag_variants = []
    for tag in [getattr(config, "model_tag", ""), f"stage{stage}", ""]:
        if tag not in (None, "") and tag not in tag_variants:
            tag_variants.append(tag)
    if f"stage{stage}" not in tag_variants:
        tag_variants.append(f"stage{stage}")

    dataset = getattr(config, "dataset", None)
    seed = getattr(config, 'random_seed', 'na')
    variant = getattr(config, "manifest_variant", None)
    variants = []
    if variant and variant != "full":
        variants.append(variant)

    stems = []
    for model_name in model_name_variants:
        # Base naming convention used by get_model_checkpoint_name(...) with multiple tag possibilities.
        for tag in tag_variants:
            parts = [model_name, tag, dataset]
            if variant and variant != "full":
                parts.append(variant)
            parts.append(f"s{seed}")
            stem = "-".join(str(p).lower() for p in parts if p not in (None, ""))
            if stem not in stems:
                stems.append(stem)

        # Also allow the exact stage-suffixed pattern used by CAT train script.
        for tag in tag_variants:
            parts = [model_name, tag, dataset]
            if variant and variant != "full":
                parts.append(variant)
            parts.append(f"s{seed}")
            stem = "-".join(str(p).lower() for p in parts if p not in (None, "")) + f"-cat-stage{stage}"
            if stem not in stems:
                stems.append(stem)

    return stems


def resolve_checkpoint_path(config, stage: int, explicit_checkpoint: str | None):
    candidates = []
    if explicit_checkpoint:
        candidates.append(explicit_checkpoint)
    for stem in candidate_checkpoint_stems(config, stage):
        if stem:
            candidates.append(os.path.join("checkpoints", f"{stem}_best.pt"))
            candidates.append(os.path.join("checkpoints", f"{stem}_last.pt"))

    seen = []
    for path in candidates:
        if path not in seen:
            seen.append(path)
            if os.path.isfile(path):
                return path
    raise FileNotFoundError("No checkpoint found. Tried: " + ", ".join(seen))


def scatter_panel(ax, xy, values, title, cmap, roi, norm=None, vmin=None, vmax=None):
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


def save_overview_figure(
    out_path: Path,
    title: str,
    rows: Sequence[Tuple[str, str, np.ndarray, np.ndarray, np.ndarray]],
    roi: Sequence[float],
    meta_line: str = "",
):
    n_rows = len(rows)
    fig, axes = plt.subplots(n_rows, 4, figsize=(20, 4.8 * n_rows), constrained_layout=True)
    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)
    fig.suptitle(title + (f"\n{meta_line}" if meta_line else ""), fontsize=15)

    for row_idx, (group_name, field_name, xy, gt, pred) in enumerate(rows):
        abs_err = np.abs(pred - gt)
        rel_err = abs_err / np.maximum(np.abs(gt), 1e-8)

        gt_vmin, gt_vmax = robust_limits(np.concatenate([gt, pred]))
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
            fig.colorbar(sc, ax=ax, shrink=0.78)
            ax.tick_params(labelsize=7)
            if col_idx == 0:
                ax.set_ylabel(field_name, fontsize=10, rotation=0, labelpad=42, va="center")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def predict_stage1_chunked(
    model: CAT,
    geo_norm: torch.Tensor,
    query_points_norm: torch.Tensor,
    chunk_size: int,
    device: torch.device,
) -> torch.Tensor:
    outputs: List[torch.Tensor] = []
    for start, end in chunk_indices(query_points_norm.shape[0], chunk_size):
        chunk = query_points_norm[start:end].to(device).unsqueeze(0)
        pred = model.forward_stage1(geo_norm, chunk)
        outputs.append(pred[0].detach().cpu())
    return torch.cat(outputs, dim=0) if outputs else query_points_norm.new_empty((0, 3))


def predict_stage2_chunked(
    model: CAT,
    geo_norm: torch.Tensor,
    query_points_norm: torch.Tensor,
    chunk_size: int,
    device: torch.device,
) -> torch.Tensor:
    outputs: List[torch.Tensor] = []
    for start, end in chunk_indices(query_points_norm.shape[0], chunk_size):
        chunk = query_points_norm[start:end].to(device).unsqueeze(0)
        pred = model.forward_stage2(geo_norm, chunk)
        outputs.append(pred[0].detach().cpu())
    return torch.cat(outputs, dim=0) if outputs else query_points_norm.new_empty((0, 1))


def predict_stage3_chunked(
    model: CAT,
    geo_norm: torch.Tensor,
    query_points_norm: torch.Tensor,
    chunk_size: int,
    device: torch.device,
) -> torch.Tensor:
    if query_points_norm.numel() == 0:
        return query_points_norm.new_empty((0, len(VOLUME_FIELDS)))

    outputs: List[torch.Tensor] = []
    for start, end in chunk_indices(query_points_norm.shape[0], chunk_size):
        chunk = query_points_norm[start:end].to(device).unsqueeze(0)
        pred = model.inference_stage3(geo_norm, chunk)
        outputs.append(pred[0].detach().cpu())
    return torch.cat(outputs, dim=0)


def stage_field_specs(stage: int):
    if stage == 1:
        return [
            ("surface", "normal_x", "unitless"),
            ("surface", "normal_y", "unitless"),
            ("volume", "sdf", "distance"),
        ]
    if stage == 2:
        return [("surface", "pressure", "Pa")]
    if stage == 3:
        return [
            ("volume", "pressure", "Pa"),
            ("volume", "sdf", "distance"),
            ("volume", "velocity_x", "unitless"),
            ("volume", "velocity_y", "unitless"),
        ]
    raise ValueError(f"Unsupported stage: {stage}")


def main():
    args = parse_args()
    cfg = load_config(args.config_name)
    config = cfg.experiment

    stage = int(args.cat_stage if args.cat_stage is not None else getattr(config, "cat_stage", 3))
    if stage not in (1, 2, 3):
        raise ValueError("CAT visualization only supports stages 1, 2, and 3")

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
    test_data = NACA4Dataset(
        config.data_path,
        if_test=True,
        geometry_points=int(config.num_body_points),
        surface_points=int(config.num_surface_points),
        volume_points=int(config.num_volume_points),
        scale_positions=bool(config.scale_positions),
        manifest_variant=getattr(config, "manifest_variant", "full"),
    )

    model_kwargs = {
        "spatial_dim": 2,
        "surface_channels": 3,
        "volume_channels": 4,
        "parameter_channels": 0,
    }
    if "architecture" in config:
        model_kwargs.update(OmegaConf.to_container(config.architecture, resolve=True))

    model = CAT(**model_kwargs).to(device)
    model.eval()

    checkpoint_name = get_model_checkpoint_name(config) + f"-cat-stage{stage}"
    checkpoint_path = resolve_checkpoint_path(config, stage, args.checkpoint)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    dataset = test_data if args.split == "test" else train_data
    case_ids = list(dataset.data)
    if args.case_ids:
        case_ids = [c.strip() for c in args.case_ids.split(",") if c.strip()]
    else:
        case_ids = case_ids[: args.num_cases]

    query_chunk_size = int(args.chunk_size or getattr(model, "subregion_size", max(int(config.num_volume_points), 1)))
    roi_specs = [("full", [-5.0, 5.0, -5.0, 5.0]), ("roi", args.roi)]
    root_dir = Path(args.output_dir or os.path.join("results", "field_error", config.dataset, checkpoint_name))
    root_dir.mkdir(parents=True, exist_ok=True)

    outputs = []
    with torch.inference_mode():
        for tag, roi in roi_specs:
            out_root = root_dir / f"split_{args.split}_{tag}_{roi_tag(roi)}_n{len(case_ids)}"
            out_root.mkdir(parents=True, exist_ok=True)
            summary_rows = []

            for case_id in tqdm(case_ids, desc=f"CAT stage{stage} cases [{tag}]", dynamic_ncols=True):
                raw = load_case_raw(dataset, case_id)
                surf_mesh = raw["surf_mesh"]
                surf_data = raw["surf_data"]
                vol_mesh = raw["vol_mesh"]
                vol_data = raw["vol_data"]

                surf_mask = roi_mask(surf_mesh, roi)
                vol_mask = roi_mask(vol_mesh, roi)
                surf_xy = surf_mesh[surf_mask]
                vol_xy = vol_mesh[vol_mask]
                if stage in (1, 3) and vol_xy.numel() == 0:
                    print(f"Skipping {case_id} [{tag}]: no volume points in ROI")
                    continue
                if stage in (1, 2) and surf_xy.numel() == 0:
                    print(f"Skipping {case_id} [{tag}]: no surface points in ROI")
                    continue

                rng = make_case_generator(int(config.random_seed), case_id)
                ns = surf_mesh.shape[0]
                nv = vol_mesh.shape[0]

                case_dir = out_root / f"case_{case_id}"
                case_dir.mkdir(parents=True, exist_ok=True)

                train_pts = (f"train: geo={int(config.num_body_points)}, surf={int(config.num_surface_points)}, vol={int(config.num_volume_points)}")

                if stage == 1:
                    s_in = int(getattr(config, "stage1_surface_input_points", config.num_body_points))

                    enc_idx = sample_indices(ns, s_in, rng)
                    input_points = surf_mesh[enc_idx]

                    # Query ALL available points in current view (ROI/full) and chunk only for memory.
                    surf_gt = surf_data[surf_mask][:, 1:3]
                    vol_gt = vol_data[vol_mask][:, 1:2]
                    query_points = torch.cat([surf_xy, vol_xy], dim=0)

                    input_norm = to_model_positions(input_points, dataset.min_pos, dataset.max_pos).to(device).unsqueeze(0)
                    query_norm = to_model_positions(query_points, dataset.min_pos, dataset.max_pos)
                    pred_norm = predict_stage1_chunked(model, input_norm, query_norm, query_chunk_size, device)

                    surf_pred = pred_norm[: surf_xy.shape[0], :2]
                    vol_pred = pred_norm[surf_xy.shape[0] :, 2:3]
                    # De-normalize to physical units for apples-to-apples comparison with raw targets.
                    surf_pred = surf_pred * dataset.std_surf_data[1:3] + dataset.mean_surf_data[1:3]
                    vol_pred = vol_pred * dataset.std_vol_data[1:2] + dataset.mean_vol_data[1:2]
                    surf_pred = surf_pred.numpy()
                    vol_pred = vol_pred.numpy()
                    surf_gt_np = surf_gt.numpy()
                    vol_gt_np = vol_gt.numpy()
                    infer_pts = f"infer(stage1): encoder_in={input_points.shape[0]}, surf_q={surf_xy.shape[0]}, vol_q={vol_xy.shape[0]}"
                    meta_line = train_pts + " | " + infer_pts

                    rows = [
                        ("surface", "normal_x", surf_xy.numpy(), surf_gt_np[:, 0], surf_pred[:, 0]),
                        ("surface", "normal_y", surf_xy.numpy(), surf_gt_np[:, 1], surf_pred[:, 1]),
                        ("volume", "sdf", vol_xy.numpy(), vol_gt_np[:, 0], vol_pred[:, 0]),
                    ]
                    surf_metrics = {"normal_x": field_metrics(surf_pred[:, 0], surf_gt_np[:, 0]), "normal_y": field_metrics(surf_pred[:, 1], surf_gt_np[:, 1])}
                    surf_normals_metrics = field_metrics(surf_pred, surf_gt_np)
                    vol_metrics = {"sdf": field_metrics(vol_pred[:, 0], vol_gt_np[:, 0])}

                    np.savez_compressed(
                        case_dir / "fields.npz",
                        surf_xy=surf_xy.numpy(),
                        surf_gt=surf_gt_np,
                        surf_pred=surf_pred,
                        vol_xy=vol_xy.numpy(),
                        vol_gt=vol_gt_np,
                        vol_pred=vol_pred,
                        roi=np.array(roi, dtype=np.float32),
                        chunk_size=np.array([query_chunk_size], dtype=np.int64),
                        checkpoint=np.array([checkpoint_path]),
                        stage=np.array([stage], dtype=np.int64),
                    )

                    with open(case_dir / "metrics.json", "w", encoding="utf-8") as f:
                        json.dump(
                            {
                                "case_id": case_id,
                                "roi": roi,
                                "chunk_size": query_chunk_size,
                                "surface": surf_metrics,
                                "surface_normals": surf_normals_metrics,
                                "volume": vol_metrics,
                            },
                            f,
                            indent=2,
                        )

                    save_overview_figure(case_dir / "stage1_overview_panel.png", f"CAT Stage 1 Case {case_id} - Overview ({tag})", rows, roi, meta_line=meta_line)
                    for group, field, units in stage_field_specs(stage):
                        if group == "surface":
                            idx = 0 if field == "normal_x" else 1
                            save_field_figure(
                                case_dir / f"surface_{field}_panel.png",
                                f"CAT Stage 1 Case {case_id} - Surface {field} ({tag})",
                                surf_xy.numpy(),
                                surf_gt_np[:, idx],
                                surf_pred[:, idx],
                                field_name=f"surface/{field}",
                                field_units=units,
                                roi=roi,
                                meta_line=meta_line,
                            )
                        else:
                            save_field_figure(
                                case_dir / f"volume_{field}_panel.png",
                                f"CAT Stage 1 Case {case_id} - Volume {field} ({tag})",
                                vol_xy.numpy(),
                                vol_gt_np[:, 0],
                                vol_pred[:, 0],
                                field_name=f"volume/{field}",
                                field_units=units,
                                roi=roi,
                                meta_line=meta_line,
                            )

                    summary_rows.append(
                        {
                            "case_id": case_id,
                            "surf_points": int(surf_xy.shape[0]),
                            "vol_points": int(vol_xy.shape[0]),
                            "surface_normal_x_mae": surf_metrics["normal_x"]["mae"],
                            "surface_normal_x_rel_l2": surf_metrics["normal_x"]["rel_l2"],
                            "surface_normal_y_mae": surf_metrics["normal_y"]["mae"],
                            "surface_normal_y_rel_l2": surf_metrics["normal_y"]["rel_l2"],
                            "surface_normals_rel_l2": surf_normals_metrics["rel_l2"],
                            "volume_sdf_mae": vol_metrics["sdf"]["mae"],
                            "volume_sdf_rel_l2": vol_metrics["sdf"]["rel_l2"],
                        }
                    )

                elif stage == 2:
                    s_in = int(getattr(config, "stage2_surface_input_points", config.num_body_points))

                    enc_idx = sample_indices(ns, s_in, rng)
                    input_points = surf_mesh[enc_idx]

                    # Query ALL available surface points in current view (ROI/full).
                    surf_gt = surf_data[surf_mask][:, 0:1]

                    input_norm = to_model_positions(input_points, dataset.min_pos, dataset.max_pos).to(device).unsqueeze(0)
                    query_norm = to_model_positions(surf_xy, dataset.min_pos, dataset.max_pos)
                    pred_norm = predict_stage2_chunked(model, input_norm, query_norm, query_chunk_size, device)
                    surf_pred = pred_norm * dataset.std_surf_data[0:1] + dataset.mean_surf_data[0:1]
                    surf_pred = surf_pred.numpy()
                    surf_gt_np = surf_gt.numpy()
                    infer_pts = f"infer(stage2): encoder_in={input_points.shape[0]}, surf_q={surf_xy.shape[0]}"
                    meta_line = train_pts + " | " + infer_pts

                    rows = [("surface", "pressure", surf_xy.numpy(), surf_gt_np[:, 0], surf_pred[:, 0])]
                    surf_metrics = {"pressure": field_metrics(surf_pred[:, 0], surf_gt_np[:, 0])}

                    np.savez_compressed(
                        case_dir / "fields.npz",
                        surf_xy=surf_xy.numpy(),
                        surf_gt=surf_gt_np,
                        surf_pred=surf_pred,
                        roi=np.array(roi, dtype=np.float32),
                        chunk_size=np.array([query_chunk_size], dtype=np.int64),
                        checkpoint=np.array([checkpoint_path]),
                        stage=np.array([stage], dtype=np.int64),
                    )

                    with open(case_dir / "metrics.json", "w", encoding="utf-8") as f:
                        json.dump(
                            {
                                "case_id": case_id,
                                "roi": roi,
                                "chunk_size": query_chunk_size,
                                "surface": surf_metrics,
                            },
                            f,
                            indent=2,
                        )

                    save_overview_figure(case_dir / "stage2_overview_panel.png", f"CAT Stage 2 Case {case_id} - Overview ({tag})", rows, roi, meta_line=meta_line)
                    save_field_figure(
                        case_dir / "surface_pressure_panel.png",
                        f"CAT Stage 2 Case {case_id} - Surface pressure ({tag})",
                        surf_xy.numpy(),
                        surf_gt_np[:, 0],
                        surf_pred[:, 0],
                        field_name="surface/pressure",
                        field_units="Pa",
                        roi=roi,
                        meta_line=meta_line,
                    )

                    summary_rows.append(
                        {
                            "case_id": case_id,
                            "surf_points": int(surf_xy.shape[0]),
                            "surface_pressure_mae": surf_metrics["pressure"]["mae"],
                            "surface_pressure_rel_l2": surf_metrics["pressure"]["rel_l2"],
                        }
                    )

                else:
                    s_in = int(getattr(config, "stage3_surface_input_points", config.num_body_points))

                    enc_idx = sample_indices(ns, s_in, rng)
                    input_points = surf_mesh[enc_idx]

                    # Query ALL available volume points in current view (ROI/full).
                    vol_gt = vol_data[vol_mask]

                    input_norm = to_model_positions(input_points, dataset.min_pos, dataset.max_pos).to(device).unsqueeze(0)
                    query_norm = to_model_positions(vol_xy, dataset.min_pos, dataset.max_pos)
                    pred_norm = predict_stage3_chunked(model, input_norm, query_norm, query_chunk_size, device)
                    vol_pred = pred_norm * dataset.std_vol_data + dataset.mean_vol_data
                    vol_pred = vol_pred.numpy()
                    vol_gt_np = vol_gt.numpy()
                    infer_pts = f"infer(stage3): encoder_in={input_points.shape[0]}, vol_q={vol_xy.shape[0]}"
                    meta_line = train_pts + " | " + infer_pts

                    rows = [("volume", field, vol_xy.numpy(), vol_gt_np[:, i], vol_pred[:, i]) for i, field in enumerate(VOLUME_FIELDS)]
                    vol_metrics = {field: field_metrics(vol_pred[:, i], vol_gt_np[:, i]) for i, field in enumerate(VOLUME_FIELDS)}
                    vol_velocity_metrics = field_metrics(vol_pred[:, 2:4], vol_gt_np[:, 2:4])
                    speed_pred = np.linalg.norm(vol_pred[:, 2:4], axis=1)
                    speed_gt = np.linalg.norm(vol_gt_np[:, 2:4], axis=1)
                    speed_metrics = field_metrics(speed_pred, speed_gt)

                    np.savez_compressed(
                        case_dir / "fields.npz",
                        vol_xy=vol_xy.numpy(),
                        vol_gt=vol_gt_np,
                        vol_pred=vol_pred,
                        volume_fields=np.array(VOLUME_FIELDS),
                        roi=np.array(roi, dtype=np.float32),
                        chunk_size=np.array([query_chunk_size], dtype=np.int64),
                        checkpoint=np.array([checkpoint_path]),
                        stage=np.array([stage], dtype=np.int64),
                    )

                    with open(case_dir / "metrics.json", "w", encoding="utf-8") as f:
                        json.dump(
                            {
                                "case_id": case_id,
                                "roi": roi,
                                "chunk_size": query_chunk_size,
                                "volume": vol_metrics,
                                "volume_velocity": vol_velocity_metrics,
                                "speed": speed_metrics,
                            },
                            f,
                            indent=2,
                        )

                    save_overview_figure(case_dir / "stage3_overview_panel.png", f"CAT Stage 3 Case {case_id} - Overview ({tag})", rows, roi, meta_line=meta_line)
                    for _, field, units in stage_field_specs(stage):
                        idx = VOLUME_FIELDS.index(field)
                        save_field_figure(
                            case_dir / f"volume_{field}_panel.png",
                            f"CAT Stage 3 Case {case_id} - Volume {field} ({tag})",
                            vol_xy.numpy(),
                            vol_gt_np[:, idx],
                            vol_pred[:, idx],
                            field_name=f"volume/{field}",
                            field_units=units,
                            roi=roi,
                        )

                    summary_row = {
                        "case_id": case_id,
                        "vol_points": int(vol_xy.shape[0]),
                        "volume_velocity_rel_l2": vol_velocity_metrics["rel_l2"],
                        "speed_rel_l2": speed_metrics["rel_l2"],
                    }
                    for field in VOLUME_FIELDS:
                        summary_row[f"{field}_mae"] = vol_metrics[field]["mae"]
                        summary_row[f"{field}_rel_l2"] = vol_metrics[field]["rel_l2"]
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
        print(f"Saved CAT {tag} outputs to: {path}")


def save_field_figure(
    out_path: Path,
    title: str,
    xy: np.ndarray,
    gt: np.ndarray,
    pred: np.ndarray,
    field_name: str,
    field_units: str,
    roi: Sequence[float],
    meta_line: str = "",
):
    abs_err = np.abs(pred - gt)
    rel_err = abs_err / np.maximum(np.abs(gt), 1e-8)
    gt_vmin, gt_vmax = robust_limits(np.concatenate([gt, pred]), low=1.0, high=99.0)

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
    fig.suptitle(title + (f"\n{meta_line}" if meta_line else ""), fontsize=14)

    sc0 = scatter_panel(axes[0], xy, gt, f"{field_name} GT", cmap="coolwarm", roi=roi, vmin=gt_vmin, vmax=gt_vmax)
    sc1 = scatter_panel(axes[1], xy, pred, f"{field_name} Pred", cmap="coolwarm", roi=roi, vmin=gt_vmin, vmax=gt_vmax)
    sc2 = scatter_panel(axes[2], xy, abs_err, f"{field_name} Abs Err", cmap="magma", roi=roi, norm=LogNorm(vmin=err_vmin, vmax=err_vmax))
    sc3 = scatter_panel(axes[3], xy, rel_err, f"{field_name} Rel Err", cmap="viridis", roi=roi, norm=LogNorm(vmin=rel_vmin, vmax=rel_vmax))

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


if __name__ == "__main__":
    main()
