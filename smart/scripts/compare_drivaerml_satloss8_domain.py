#!/usr/bin/env python3
"""Compare SATLOSS8 models on the held-out statistical geometry domain.

SATLOSS8 is SATLOSS7's fixed-sum, two-view training protocol with one deliberate
dataset change: training uses geometry cluster 0 and evaluation uses cluster 1
from ``geometry_domain_split.json``. This evaluator therefore does not create
beta/sine/remeshing shifts. It reports the aligned encoder-view error on both
the training-domain cluster and the held-out geometry-domain cluster, making
the cross-domain generalization gap explicit.

The geometry clustering itself is performed by
``cluster_drivaerml_geometry_statistical.py`` and is never recomputed here.
No CFD fields or model outputs are used to define the split.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm.auto import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
SMART_ROOT = SCRIPT_DIR.parent
if str(SMART_ROOT) not in sys.path:
    sys.path.insert(0, str(SMART_ROOT))

from data.ahmedml_dataset_v2 import AhmedMLDatasetV2
from models.smart.smart import SMART
from models.transolverpp import TransolverPP
from models.pointnet2_ssg import PointNet2SSG
from models.lno import LNO
from models.mspt import MSPT
from models.point_transformer_v3 import PointTransformerV3
from scripts.compare_drivaerml_sampling_invariance import (
    build_model,
    compute_metrics,
    denorm_fields,
    load_cfg,
    predict_view_batch,
    sample_uniform_without_replacement,
    train_encoder_input_points,
)


MODEL_ORDER = [
    "SMART_SATLOSS8",
    "TRANSOLVERPP_SATLOSS8",
    "POINTNET2_SSG_SATLOSS8",
    "LNO_SATLOSS8",
    "MSPT_SATLOSS8",
    "POINT_TRANSFORMER_V3_SATLOSS8",
]
MODEL_LABELS = {
    "SMART_SATLOSS8": "SMART-SATLOSS8",
    "TRANSOLVERPP_SATLOSS8": "TransolverPP-SATLOSS8",
    "POINTNET2_SSG_SATLOSS8": "PointNet++-SSG-SATLOSS8",
    "LNO_SATLOSS8": "LNO-SATLOSS8",
    "MSPT_SATLOSS8": "MSPT-SATLOSS8",
    "POINT_TRANSFORMER_V3_SATLOSS8": "PointTransformerV3-SATLOSS8",
}
MODEL_COLORS = {
    # Saturated, color-blind-friendly family colors. Domain status is encoded
    # by hatch, so the same family remains visually paired across domains.
    "SMART_SATLOSS8": "#1F77B4",
    "TRANSOLVERPP_SATLOSS8": "#FF7F0E",
    "POINTNET2_SSG_SATLOSS8": "#17BECF",
    "LNO_SATLOSS8": "#D62728",
    "MSPT_SATLOSS8": "#2CA02C",
    "POINT_TRANSFORMER_V3_SATLOSS8": "#7F3C8D",
}
BASE_MODEL_NAMES = {name: name.removesuffix("_SATLOSS8") for name in MODEL_ORDER}
# Keep the detailed metrics in the machine-readable outputs for auditability,
# but deliberately render only combined-global error in paper-facing figures.
METRIC_KEYS = [
    "combined_global_rel_l2",
    "combined_physics_rel_l2",
    "surface_global_rel_l2",
    "volume_global_rel_l2",
    "surface_pressure_rel_l2",
    "volume_pressure_rel_l2",
    "surface_wss_mag_rel_l2",
    "volume_velocity_mag_rel_l2",
]

_PLOT_FONT_SCALE = 1.0
_PLOT_BASE_FONT_SIZE = 15.0


def configure_plot_style(font_scale: float) -> None:
    global _PLOT_FONT_SCALE
    font_scale = float(font_scale)
    if not math.isfinite(font_scale) or font_scale <= 0.0:
        raise ValueError("--font-scale must be a finite positive number.")
    _PLOT_FONT_SCALE = font_scale
    font_size = 0.55 * _PLOT_BASE_FONT_SIZE * font_scale
    matplotlib.rcParams.update(
        {
            "font.size": font_size,
            "axes.titlesize": font_size,
            "axes.labelsize": font_size,
            "xtick.labelsize": font_size,
            "ytick.labelsize": font_size,
            "legend.fontsize": font_size,
            "figure.titlesize": font_size,
        }
    )


def plot_font_size(multiplier: float = 1.0) -> float:
    return 0.55 * _PLOT_BASE_FONT_SIZE * _PLOT_FONT_SCALE * float(multiplier)


def save_plot(fig: matplotlib.figure.Figure, path: Path) -> None:
    fig.set_constrained_layout(False)
    fig.savefig(path, dpi=280, bbox_inches="tight", pad_inches=0.18)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-json", default="/home/parsa/smart_parsa/results/drivaerml_geometry_statistical_split/geometry_domain_split.json")
    parser.add_argument("--data-root", default="/mnt/ssdraid/parsa/drivaerml_preprocessed")
    parser.add_argument("--output-dir", default="/home/parsa/smart_parsa/results/drivaerml_satloss8_geometry_domain_comparison")
    parser.add_argument("--train-cluster", type=int, default=0)
    parser.add_argument("--test-cluster", type=int, default=1)
    parser.add_argument("--num-runs", type=int, default=0, help="Runs per domain; 0 evaluates every run.")
    parser.add_argument("--run-ids", default=None, help="Optional comma-separated IDs, applied to both clusters where present.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--views-per-run", type=int, default=2)
    parser.add_argument("--run-batch-size", type=int, default=1)
    parser.add_argument("--surface-query-points", type=int, default=0)
    parser.add_argument("--volume-query-points", type=int, default=0)
    parser.add_argument("--batched-query-subregion-size", type=int, default=65536)
    parser.add_argument(
        "--plot-scales",
        default="log,linear",
        help="Comma-separated absolute plot scales to render: log,linear.",
    )
    parser.add_argument("--font-scale", type=float, default=1.2)
    parser.add_argument("--devices", default=None, help="Comma-separated devices, e.g. cuda:0,cuda:1,cuda:2.")
    parser.add_argument("--device", default=None, help="Single-device fallback, e.g. cuda:0.")
    parser.add_argument("--dry-run", action="store_true", help="Validate configs/checkpoints without inference.")
    parser.add_argument("--smart-config", default="drivaerml_satloss8")
    parser.add_argument("--transolverpp-config", default="drivaerml_transolverpp_satloss8")
    parser.add_argument("--pointnet2-ssg-config", default="drivaerml_pointnet2_ssg_satloss8")
    parser.add_argument("--lno-config", default="drivaerml_lno_satloss8")
    parser.add_argument("--mspt-config", default="drivaerml_mspt_satloss8")
    parser.add_argument("--point-transformer-v3-config", default="drivaerml_point_transformer_v3_satloss8")
    parser.add_argument("--smart-satloss8-checkpoint", default=None)
    parser.add_argument("--transolverpp-satloss8-checkpoint", default=None)
    parser.add_argument("--pointnet2-ssg-satloss8-checkpoint", default=None)
    parser.add_argument("--lno-satloss8-checkpoint", default=None)
    parser.add_argument("--mspt-satloss8-checkpoint", default=None)
    parser.add_argument("--point-transformer-v3-satloss8-checkpoint", default=None)
    return parser.parse_args()


def resolve_devices(args: argparse.Namespace) -> List[torch.device]:
    text = args.devices or args.device
    if text:
        values = [item.strip() for item in str(text).split(",") if item.strip()]
        devices = [torch.device(item) for item in values]
    elif torch.cuda.is_available():
        devices = [torch.device("cuda")]
    else:
        devices = [torch.device("cpu")]
    if any(device.type == "cuda" for device in devices) and not torch.cuda.is_available():
        raise RuntimeError("CUDA devices were requested but CUDA is unavailable.")
    return devices


def checkpoint_map(args: argparse.Namespace) -> Dict[str, str | None]:
    return {
        "SMART_SATLOSS8": args.smart_satloss8_checkpoint,
        "TRANSOLVERPP_SATLOSS8": args.transolverpp_satloss8_checkpoint,
        "POINTNET2_SSG_SATLOSS8": args.pointnet2_ssg_satloss8_checkpoint,
        "LNO_SATLOSS8": args.lno_satloss8_checkpoint,
        "MSPT_SATLOSS8": args.mspt_satloss8_checkpoint,
        "POINT_TRANSFORMER_V3_SATLOSS8": args.point_transformer_v3_satloss8_checkpoint,
    }


def config_map(args: argparse.Namespace) -> Dict[str, str]:
    return {
        "SMART_SATLOSS8": args.smart_config,
        "TRANSOLVERPP_SATLOSS8": args.transolverpp_config,
        "POINTNET2_SSG_SATLOSS8": args.pointnet2_ssg_config,
        "LNO_SATLOSS8": args.lno_config,
        "MSPT_SATLOSS8": args.mspt_config,
        "POINT_TRANSFORMER_V3_SATLOSS8": args.point_transformer_v3_config,
    }


def load_split(path: Path, train_cluster: int, test_cluster: int) -> tuple[List[int], List[int]]:
    with path.open("r", encoding="utf-8") as handle:
        split = json.load(handle)
    key = f"train_cluster_{int(train_cluster)}_test_cluster_{int(test_cluster)}"
    direction = split.get(key)
    if not isinstance(direction, dict):
        raise ValueError(f"Split JSON does not contain `{key}`: {path}")
    train_ids = [int(item) for item in direction.get("train_ids", [])]
    test_ids = [int(item) for item in direction.get("test_ids", [])]
    if not train_ids or not test_ids or set(train_ids).intersection(test_ids):
        raise ValueError(f"Invalid train/test IDs in `{key}`.")
    return train_ids, test_ids


def select_ids(ids: Sequence[int], args: argparse.Namespace, offset: int) -> List[int]:
    ids = sorted(int(item) for item in ids)
    if args.run_ids:
        requested = [int(item.strip()) for item in str(args.run_ids).split(",") if item.strip()]
        missing = sorted(set(requested).difference(ids))
        if missing:
            raise ValueError(f"Requested run IDs are not in this domain: {missing}")
        return requested
    if int(args.num_runs) <= 0 or int(args.num_runs) >= len(ids):
        return ids
    rng = np.random.default_rng(int(args.seed) + int(offset))
    return sorted(int(item) for item in rng.choice(np.asarray(ids), size=int(args.num_runs), replace=False))


def dataset_for_domain(config, split_json: Path, train_cluster: int, test_cluster: int, if_test: bool, surface_points: int, volume_points: int):
    return AhmedMLDatasetV2(
        saved_folder=str(config.data_path),
        if_test=bool(if_test),
        geometry_points=0,
        surface_points=int(surface_points),
        volume_points=int(volume_points),
        scale_positions=bool(config.scale_positions),
        require_preprocessed=True,
        domain_split_json=str(split_json),
        domain_split_train_cluster=int(train_cluster),
        domain_split_test_cluster=int(test_cluster),
    )


def get_case(dataset: AhmedMLDatasetV2, run_id: int, seed: int):
    try:
        index = dataset.data.index(int(run_id))
    except ValueError as exc:
        raise ValueError(f"run_{run_id} is not available in this dataset role.") from exc
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    item = dataset[index]
    if len(item) != 5:
        raise ValueError(f"Expected a five-tensor dataset item, got {len(item)} values for run_{run_id}.")
    geo, surf, surf_data, vol, vol_data = item
    return geo.float(), surf.float(), surf_data.float(), vol.float(), vol_data.float()


def build_satloss8_model(config, checkpoint: str, device: torch.device, query_chunk: int):
    # SATLOSS8 has identical tensor architecture to its vanilla model. The
    # alias is only for the comparison builder's architecture dispatch.
    build_config = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
    build_config.model_name = BASE_MODEL_NAMES[str(config.model_name)]
    return build_model(build_config, checkpoint, device, batched_query_subregion_size=int(query_chunk)).to(device)


def evaluate_model(
    model_name: str,
    model,
    device: torch.device,
    cases_by_domain: Mapping[str, Mapping[int, tuple[torch.Tensor, ...]]],
    input_budget: int,
    views_per_run: int,
    mean_s: torch.Tensor,
    std_s: torch.Tensor,
    mean_v: torch.Tensor,
    std_v: torch.Tensor,
    seed: int,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    base_name = BASE_MODEL_NAMES[model_name]
    for domain_name, cases in cases_by_domain.items():
        for run_id in tqdm(sorted(cases), desc=f"{MODEL_LABELS[model_name]} {domain_name}", leave=False):
            geo, surf_query, surf_data, vol_query, vol_data = cases[run_id]
            n_geo = int(geo.shape[0])
            views = []
            for view_idx in range(max(1, int(views_per_run))):
                rng = np.random.default_rng([int(seed), int(run_id), int(view_idx), 81173])
                index = sample_uniform_without_replacement(n_geo, int(input_budget), rng)
                views.append(geo[index])
            geo_views = torch.stack(views, dim=0)
            pred_s, pred_v = predict_view_batch(
                base_name,
                model,
                geo_views,
                surf_query,
                vol_query,
                None,
                mean_s,
                std_s,
                mean_v,
                std_v,
                device,
                base_seed=int(seed + run_id * 1009),
                repeats=1,
            )
            gt_s = denorm_fields(surf_data, mean_s, std_s).numpy()
            gt_v = denorm_fields(vol_data, mean_v, std_v).numpy()
            per_view = [
                compute_metrics(gt_s, pred_s[view_idx], gt_v, pred_v[view_idx])
                for view_idx in range(pred_s.shape[0])
            ]
            row = {"model_name": model_name, "domain": domain_name, "run_id": int(run_id)}
            for key in METRIC_KEYS:
                row[key] = float(np.mean([metrics[key] for metrics in per_view]))
            rows.append(row)
    if device.type == "cuda":
        with torch.cuda.device(device):
            torch.cuda.synchronize(device)
    return rows


def aggregate(rows: Sequence[Mapping[str, object]]) -> Dict[str, Dict[str, Dict[str, float]]]:
    output = {}
    for model_name in MODEL_ORDER:
        output[model_name] = {}
        for domain in ("train_cluster", "heldout_cluster"):
            values = [row for row in rows if row["model_name"] == model_name and row["domain"] == domain]
            if not values:
                continue
            output[model_name][domain] = {
                f"{key}_mean": float(np.mean([float(row[key]) for row in values])) for key in METRIC_KEYS
            }
            output[model_name][domain].update(
                {f"{key}_std": float(np.std([float(row[key]) for row in values])) for key in METRIC_KEYS}
            )
            output[model_name][domain]["run_count"] = int(len(values))
    return output


def save_rows(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    fields = ["model_name", "domain", "run_id", *METRIC_KEYS]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _present_models(summary: Mapping[str, Mapping[str, Mapping[str, float]]]) -> List[str]:
    return [
        name
        for name in MODEL_ORDER
        if name in summary and "train_cluster" in summary[name] and "heldout_cluster" in summary[name]
    ]


def _set_absolute_limits(ax, values: Sequence[float], log_scale: bool, pad_fraction: float = 0.10) -> None:
    finite = np.asarray([value for value in values if math.isfinite(float(value))], dtype=np.float64)
    if finite.size == 0:
        return
    low = float(np.min(finite))
    high = float(np.max(finite))
    if log_scale:
        ax.set_ylim(max(low * (1.0 - pad_fraction), 1.0e-12), high * (1.0 + pad_fraction))
    else:
        span = max(high - low, high, 1.0e-12)
        ax.set_ylim(max(0.0, low - pad_fraction * span), high + pad_fraction * span)


def _set_signed_limits(ax, values: Sequence[float], pad_fraction: float = 0.10) -> None:
    finite = np.asarray([value for value in values if math.isfinite(float(value))], dtype=np.float64)
    if finite.size == 0:
        return
    low = min(0.0, float(np.min(finite)))
    high = max(0.0, float(np.max(finite)))
    span = max(high - low, 1.0)
    pad = float(pad_fraction) * span
    ax.set_ylim(low - pad, high + pad)


def plot_absolute_bars(summary, path: Path, log_scale: bool) -> None:
    """Render the domain split as paired train/held-out bars.

    SATLOSS8 has no beta/sine or remeshing severity axis. The two bars are
    therefore the scientifically meaningful comparison: in-domain cluster 0
    versus held-out geometry cluster 1. Model color is kept constant and hatch
    carries domain status, avoiding a misleading second color family.
    """
    present = _present_models(summary)
    x = np.arange(len(present), dtype=np.float64)
    total_slots = 2
    slot_pitch = 0.44
    width = 0.38
    font_size = plot_font_size()
    fig, ax = plt.subplots(figsize=(13.4, 7.4))
    fig.subplots_adjust(left=0.12, right=0.60, bottom=0.34, top=0.84)
    values: List[float] = []
    for source_idx, (domain, label, hatch) in enumerate(
        (
            ("train_cluster", "cluster 0: training domain", ""),
            ("heldout_cluster", "cluster 1: held-out domain", "//"),
        )
    ):
        slot_offset = (source_idx - 0.5 * (total_slots - 1)) * slot_pitch
        means = [float(summary[name][domain]["combined_global_rel_l2_mean"]) for name in present]
        values.extend(means)
        ax.bar(
            x + slot_offset,
            means,
            width=width,
            color=[MODEL_COLORS[name] for name in present],
            edgecolor="#222222",
            linewidth=0.65,
            alpha=0.96,
            hatch=hatch,
            label=label,
        )
    if log_scale:
        ax.set_yscale("log")
    _set_absolute_limits(ax, values, log_scale)
    scale_name = "log" if log_scale else "linear"
    ax.set_ylabel(f"Combined-global relative L2 error ({scale_name} scale)", fontsize=font_size)
    ax.set_title("SATLOSS8 geometry-domain evaluation", fontsize=font_size, pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS[name] for name in present], rotation=24, ha="right", fontsize=font_size)
    ax.yaxis.set_label_coords(-0.095, 0.50)
    ax.tick_params(axis="both", labelsize=font_size)
    ax.grid(axis="y", which="both", alpha=0.20, linewidth=0.8)
    ax.set_axisbelow(True)
    model_legend = fig.legend(
        handles=[
            Patch(facecolor=MODEL_COLORS[name], edgecolor="#222222", label=MODEL_LABELS[name])
            for name in present
        ],
        loc="upper left",
        bbox_to_anchor=(0.625, 0.82),
        bbox_transform=fig.transFigure,
        framealpha=0.92,
        fontsize=font_size,
    )
    fig.legend(
        handles=[
            Patch(facecolor="#777777", edgecolor="#222222", alpha=0.96, label="cluster 0: training domain"),
            Patch(facecolor="#777777", edgecolor="#222222", alpha=0.96, hatch="//", label="cluster 1: held-out domain"),
        ],
        loc="upper left",
        bbox_to_anchor=(0.625, 0.43),
        bbox_transform=fig.transFigure,
        framealpha=0.92,
        fontsize=font_size,
    )
    save_plot(fig, path)
    plt.close(fig)


def plot_domain_gap(summary, path: Path) -> None:
    """Render held-out error relative to each model's own train-domain error."""
    present = _present_models(summary)
    values = []
    for name in present:
        train = float(summary[name]["train_cluster"]["combined_global_rel_l2_mean"])
        heldout = float(summary[name]["heldout_cluster"]["combined_global_rel_l2_mean"])
        values.append(100.0 * (heldout - train) / max(abs(train), 1.0e-12))
    x = np.arange(len(present), dtype=np.float64)
    font_size = plot_font_size()
    fig, ax = plt.subplots(figsize=(13.4, 7.4))
    fig.subplots_adjust(left=0.12, right=0.60, bottom=0.34, top=0.84)
    bars = ax.bar(
        x,
        values,
        width=0.525,
        color=[MODEL_COLORS[name] for name in present],
        edgecolor="#222222",
        linewidth=0.65,
        alpha=0.96,
    )
    ax.axhline(0.0, color="#303030", linewidth=1.1)
    _set_signed_limits(ax, values)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            value + (0.005 * max(max(values) - min(values), 1.0) if value >= 0.0 else -0.005 * max(max(values) - min(values), 1.0)),
            f"{value:+.1f}%",
            ha="center",
            va="bottom" if value >= 0.0 else "top",
            fontsize=font_size * 0.82,
            rotation=90 if len(f"{value:+.1f}%") > 7 else 0,
            clip_on=False,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS[name] for name in present], rotation=24, ha="right", fontsize=font_size)
    ax.set_ylabel("Held-out-domain change vs training domain (%)", fontsize=font_size)
    ax.yaxis.set_label_coords(-0.095, 0.50)
    ax.set_title("SATLOSS8 geometry-domain generalization gap", fontsize=font_size, pad=12)
    ax.tick_params(axis="both", labelsize=font_size)
    ax.grid(axis="y", alpha=0.20, linewidth=0.8)
    ax.set_axisbelow(True)
    fig.legend(
        handles=[
            Patch(facecolor=MODEL_COLORS[name], edgecolor="#222222", label=MODEL_LABELS[name])
            for name in present
        ],
        loc="upper left",
        bbox_to_anchor=(0.625, 0.82),
        bbox_transform=fig.transFigure,
        framealpha=0.92,
        fontsize=font_size,
    )
    save_plot(fig, path)
    plt.close(fig)


def write_domain_table(summary, output: Path) -> None:
    rows = []
    for name in _present_models(summary):
        train = float(summary[name]["train_cluster"]["combined_global_rel_l2_mean"])
        heldout = float(summary[name]["heldout_cluster"]["combined_global_rel_l2_mean"])
        rows.append(
            {
                "model": MODEL_LABELS[name],
                "cluster_0_train_domain_rel_l2": train,
                "cluster_1_heldout_domain_rel_l2": heldout,
                "heldout_minus_train": heldout - train,
                "heldout_change_percent": 100.0 * (heldout - train) / max(abs(train), 1.0e-12),
            }
        )
    csv_path = output / "satloss8_combined_global_domain_table.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    md_path = output / "satloss8_combined_global_domain_table.md"
    lines = [
        "# SATLOSS8 Combined-Global Geometry-Domain Results",
        "",
        "Errors are mean combined-global relative L2 values. No standard-deviation terms are used in the paper-facing plots.",
        "",
        "| Model | Cluster 0 train-domain error | Cluster 1 held-out-domain error | Held-out minus train | Held-out change |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    lines.extend(
        f"| {row['model']} | {row['cluster_0_train_domain_rel_l2']:.6f} | {row['cluster_1_heldout_domain_rel_l2']:.6f} | {row['heldout_minus_train']:+.6f} | {row['heldout_change_percent']:+.2f}% |"
        for row in rows
    )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    configure_plot_style(args.font_scale)
    plot_scales = {item.strip().lower() for item in str(args.plot_scales).split(",") if item.strip()}
    invalid_scales = plot_scales.difference({"linear", "log"})
    if invalid_scales or not plot_scales:
        raise ValueError("--plot-scales must contain one or both of: linear,log")
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    split_path = Path(args.split_json).expanduser().resolve()
    train_ids_all, heldout_ids_all = load_split(split_path, args.train_cluster, args.test_cluster)
    train_ids = select_ids(train_ids_all, args, 7001)
    heldout_ids = select_ids(heldout_ids_all, args, 7002)
    if set(train_ids).intersection(heldout_ids):
        raise ValueError("Train and held-out evaluation IDs overlap.")

    cfg_names = config_map(args)
    ckpts = checkpoint_map(args)
    active = [name for name in MODEL_ORDER if ckpts[name]]
    if not active:
        raise ValueError("Provide at least one SATLOSS8 checkpoint.")
    configs = OrderedDict((name, load_cfg(cfg_names[name])) for name in active)
    data_paths = {str(config.data_path) for config in configs.values()}
    if data_paths != {str(Path(args.data_root).expanduser().resolve())}:
        raise ValueError(f"Checkpoint configs do not all use --data-root={args.data_root}: {sorted(data_paths)}")
    devices = resolve_devices(args)
    for name in active:
        if not Path(ckpts[name]).is_file():
            raise FileNotFoundError(f"Checkpoint not found: {ckpts[name]}")
    if args.dry_run:
        print(
            f"SATLOSS8 dry run: cluster {args.train_cluster} -> cluster {args.test_cluster}; "
            f"train_runs={len(train_ids)}, heldout_runs={len(heldout_ids)}"
        )
        print(f"Devices: {', '.join(map(str, devices))}")
        print(f"Plot scales: {', '.join(sorted(plot_scales))}; font scale: {args.font_scale:.2f}; std bars: disabled")
        for name in active:
            print(
                f"{MODEL_LABELS[name]}: config={cfg_names[name]}, "
                f"checkpoint={ckpts[name]}, encoder_points={train_encoder_input_points(configs[name], name)}"
            )
        return
    query_s = int(args.surface_query_points) if int(args.surface_query_points) > 0 else min(int(configs[name].num_surface_points) for name in active)
    query_v = int(args.volume_query_points) if int(args.volume_query_points) > 0 else min(int(configs[name].num_volume_points) for name in active)
    train_dataset = dataset_for_domain(configs[active[0]], split_path, args.train_cluster, args.test_cluster, False, query_s, query_v)
    heldout_dataset = dataset_for_domain(configs[active[0]], split_path, args.train_cluster, args.test_cluster, True, query_s, query_v)
    if sorted(train_dataset.training_ids) != sorted(train_ids_all) or sorted(heldout_dataset.test_ids) != sorted(heldout_ids_all):
        raise RuntimeError("Dataset domain split does not match the requested split JSON.")
    mean_s = train_dataset.mean_surf_data
    std_s = torch.clamp(train_dataset.std_surf_data, min=1.0e-12)
    mean_v = train_dataset.mean_vol_data
    std_v = torch.clamp(train_dataset.std_vol_data, min=1.0e-12)
    print(f"SATLOSS8 domain split: train cluster {args.train_cluster} ({len(train_ids)} runs), held-out cluster {args.test_cluster} ({len(heldout_ids)} runs)")
    print(f"Query budgets: surface={query_s}, volume={query_v}")

    cases_by_domain = {"train_cluster": {}, "heldout_cluster": {}}
    for domain_name, dataset, run_ids, offset in [
        ("train_cluster", train_dataset, train_ids, 101),
        ("heldout_cluster", heldout_dataset, heldout_ids, 202),
    ]:
        for run_id in tqdm(run_ids, desc=f"Loading {domain_name}"):
            cases_by_domain[domain_name][run_id] = get_case(dataset, run_id, args.seed + offset + run_id)

    model_device = {name: devices[index % len(devices)] for index, name in enumerate(active)}
    models = {}
    for name in active:
        print(f"Loading {MODEL_LABELS[name]} from {ckpts[name]} on {model_device[name]}")
        models[name] = build_satloss8_model(configs[name], ckpts[name], model_device[name], args.batched_query_subregion_size)

    model_rows: Dict[str, List[Dict[str, object]]] = {}
    groups = {device: [name for name in active if model_device[name] == device] for device in devices}

    def evaluate_device(device: torch.device, names: Sequence[str]):
        result = {}
        for name in names:
            result[name] = evaluate_model(
                name,
                models[name],
                device,
                cases_by_domain,
                train_encoder_input_points(configs[name], name),
                args.views_per_run,
                mean_s,
                std_s,
                mean_v,
                std_v,
                args.seed,
            )
        return result

    with ThreadPoolExecutor(max_workers=len(devices)) as pool:
        futures = [pool.submit(evaluate_device, device, names) for device, names in groups.items() if names]
        for future in futures:
            model_rows.update(future.result())
    rows = [row for name in active for row in model_rows[name]]
    rows.sort(key=lambda row: (str(row["model_name"]), str(row["domain"]), int(row["run_id"])))
    save_rows(output / "satloss8_domain_metrics.csv", rows)
    summary = aggregate(rows)
    payload = {
        "protocol": "SATLOSS8 = SATLOSS7 objective/sampling with statistical geometry-domain split",
        "split_json": str(split_path),
        "train_cluster": int(args.train_cluster),
        "heldout_cluster": int(args.test_cluster),
        "train_run_ids": train_ids,
        "heldout_run_ids": heldout_ids,
        "views_per_run": int(args.views_per_run),
        "query_surface_points": query_s,
        "query_volume_points": query_v,
        "encoder_input_points": {
            name: int(train_encoder_input_points(configs[name], name)) for name in active
        },
        "plot_protocol": {
            "visible_metric": "combined_global_rel_l2",
            "plot_scales": sorted(plot_scales),
            "standard_deviation_bars": False,
            "font_scale": float(args.font_scale),
            "domain_encoding": "model family color; cluster 0 solid; cluster 1 hatched",
            "sampling_shifts": "not evaluated; SATLOSS8 is a geometry-domain split test",
        },
        "models": {name: {"config": cfg_names[name], "checkpoint": ckpts[name]} for name in active},
        "metrics": summary,
    }
    with (output / "satloss8_domain_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    write_domain_table(summary, output)
    # Remove only plots from older versions of this dedicated renderer. The
    # metrics/CSV artifacts remain untouched for reproducibility.
    for old_plot in output.glob("satloss8_*_*.png"):
        old_plot.unlink()
    if "linear" in plot_scales:
        plot_absolute_bars(
            summary,
            output / "satloss8_combined_global_domain_bars_linear.png",
            log_scale=False,
        )
    if "log" in plot_scales:
        plot_absolute_bars(
            summary,
            output / "satloss8_combined_global_domain_bars_log.png",
            log_scale=True,
        )
    plot_domain_gap(summary, output / "satloss8_combined_global_domain_gap_percentage.png")
    print(f"Wrote SATLOSS8 metrics: {output / 'satloss8_domain_metrics.json'}")
    print(f"Wrote SATLOSS8 CSV: {output / 'satloss8_domain_metrics.csv'}")


if __name__ == "__main__":
    main()
