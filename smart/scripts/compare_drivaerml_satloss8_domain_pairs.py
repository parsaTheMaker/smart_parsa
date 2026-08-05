#!/usr/bin/env python3
"""Paired vanilla8/SATLOSS8 geometry-domain comparison.

This benchmark evaluates the vanilla8 and SATLOSS8 checkpoint of each model
on the same held-out geometry-domain cases.  It deliberately does not create
beta, sine, remeshing, or masking shifts: cluster 1 is the distribution shift.

The script evaluates ``--num-runs`` held-out cluster-1 geometries, ranks them
by SATLOSS8 improvement over vanilla8, and renders paper-facing combined-global
plots for all cases and for the shared top-k cases.  With ``--positive-first``,
geometries with positive summed paired improvement across model families are ranked
first; if fewer than ``top-k`` are positive, the least-negative remaining cases
fill the list and remain visible in the signed ranking tables.  The stricter
``--positive-only`` mode is also available and refuses to fill the list with
regressions.  A separate top-k list is written for every model family so the
selection can be audited without silently using a different geometry set for
different plots.  ``--min-transolverpp-improvement-percent`` can further
restrict the candidate pool before any shared ranking is performed.
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
from scripts.compare_drivaerml_sampling_invariance import (
    SURFACE_FIELDS,
    VOLUME_FIELDS,
    build_model,
    compute_metrics,
    denorm_fields,
    load_cfg,
    predict_view_batch,
    sample_uniform_without_replacement,
    train_encoder_input_points,
)


FAMILY_ORDER = ["SMART", "TRANSOLVERPP", "POINTNET2_SSG", "LNO", "MSPT", "POINT_TRANSFORMER_V3"]
FAMILY_LABELS = {
    "SMART": "SMART",
    "TRANSOLVERPP": "TransolverPP",
    "POINTNET2_SSG": "PointNet++-SSG",
    "LNO": "LNO",
    "MSPT": "MSPT",
    "POINT_TRANSFORMER_V3": "PointTransformerV3",
}
FAMILY_COLORS = {
    "SMART": "#1F77B4",
    "TRANSOLVERPP": "#FF7F0E",
    "POINTNET2_SSG": "#17BECF",
    "LNO": "#D62728",
    "MSPT": "#2CA02C",
    "POINT_TRANSFORMER_V3": "#7F3C8D",
}
VARIANTS = ("vanilla8", "satloss8")
METRIC_KEY = "combined_global_rel_l2"
BASE_FONT_SIZE = 15.0
FONT_SCALE = 1.0


def configure_plot_style(font_scale: float) -> None:
    global FONT_SCALE
    if not math.isfinite(float(font_scale)) or float(font_scale) <= 0.0:
        raise ValueError("--font-scale must be finite and positive.")
    FONT_SCALE = float(font_scale)
    size = 0.55 * BASE_FONT_SIZE * FONT_SCALE
    matplotlib.rcParams.update(
        {
            "font.size": size,
            "axes.titlesize": size,
            "axes.labelsize": size,
            "xtick.labelsize": size,
            "ytick.labelsize": size,
            "legend.fontsize": size,
            "figure.titlesize": size,
        }
    )


def font_size(multiplier: float = 1.0) -> float:
    return 0.55 * BASE_FONT_SIZE * FONT_SCALE * float(multiplier)


def save_plot(fig: matplotlib.figure.Figure, path: Path) -> None:
    fig.savefig(path, dpi=280, bbox_inches="tight", pad_inches=0.16)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split-json",
        default="/home/parsa/smart_parsa/results/drivaerml_geometry_statistical_split/geometry_domain_split.json",
    )
    parser.add_argument("--data-root", default="/mnt/ssdraid/parsa/drivaerml_preprocessed")
    parser.add_argument(
        "--output-dir",
        default="/home/parsa/smart_parsa/results/drivaerml_satloss8_domain_pairs_40runs",
    )
    parser.add_argument("--train-cluster", type=int, default=0)
    parser.add_argument("--test-cluster", type=int, default=1)
    parser.add_argument("--num-runs", type=int, default=40)
    parser.add_argument("--run-ids", default=None, help="Optional explicit comma-separated held-out run IDs.")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--positive-only",
        action="store_true",
        help=(
            "For shared top-k selection, require positive summed paired improvement "
            "(vanilla8 error - SATLOSS8 error) across model families. Per-family "
            "top-k lists use the same positive-only filter."
        ),
    )
    parser.add_argument(
        "--positive-first",
        action="store_true",
        help=(
            "Rank positive summed paired improvement first and fill remaining top-k "
            "slots with the least-negative candidates. Cannot be combined with "
            "--positive-only."
        ),
    )
    parser.add_argument(
        "--min-transolverpp-improvement-percent",
        type=float,
        default=None,
        help=(
            "Restrict ranking candidates to geometries where TransolverPP "
            "SATLOSS8 improves over TransolverPP vanilla8 by strictly more "
            "than this relative percentage."
        ),
    )
    parser.add_argument(
        "--exclude-from-ranking",
        default="",
        help=(
            "Comma-separated family keys excluded from the shared summed ranking "
            "but still evaluated and plotted, e.g. POINT_TRANSFORMER_V3."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--views-per-run", type=int, default=2)
    parser.add_argument("--surface-query-points", type=int, default=65536)
    parser.add_argument("--volume-query-points", type=int, default=65536)
    parser.add_argument("--batched-query-subregion-size", type=int, default=65536)
    parser.add_argument("--plot-scales", default="linear,log")
    parser.add_argument("--font-scale", type=float, default=1.2)
    parser.add_argument("--devices", default=None, help="Comma-separated inference devices, e.g. cuda:0,cuda:1,cuda:2,cuda:3.")
    parser.add_argument("--device", default=None, help="Single-device fallback.")
    parser.add_argument("--dry-run", action="store_true")

    # Vanilla8 configurations were trained from the ordinary model configs;
    # the domain split is a dataset override, not a separate architecture.
    parser.add_argument("--smart-config", default="drivaerml")
    parser.add_argument("--smart-satloss8-config", default="drivaerml_satloss8")
    parser.add_argument("--transolverpp-config", default="drivaerml_transolverpp")
    parser.add_argument("--transolverpp-satloss8-config", default="drivaerml_transolverpp_satloss8")
    parser.add_argument("--pointnet2-ssg-config", default="drivaerml_pointnet2_ssg")
    parser.add_argument("--pointnet2-ssg-satloss8-config", default="drivaerml_pointnet2_ssg_satloss8")
    parser.add_argument("--lno-config", default="drivaerml_lno")
    parser.add_argument("--lno-satloss8-config", default="drivaerml_lno_satloss8")
    parser.add_argument("--mspt-config", default="drivaerml_mspt")
    parser.add_argument("--mspt-satloss8-config", default="drivaerml_mspt_satloss8")
    parser.add_argument(
        "--point-transformer-v3-config",
        default="drivaerml_point_transformer_v3_density_sensitive",
    )
    parser.add_argument(
        "--point-transformer-v3-satloss8-config",
        default="drivaerml_point_transformer_v3_satloss8_density_sensitive",
    )

    parser.add_argument("--smart-checkpoint", required=True)
    parser.add_argument("--smart-satloss8-checkpoint", required=True)
    parser.add_argument("--transolverpp-checkpoint", required=True)
    parser.add_argument("--transolverpp-satloss8-checkpoint", required=True)
    parser.add_argument("--pointnet2-ssg-checkpoint", required=True)
    parser.add_argument("--pointnet2-ssg-satloss8-checkpoint", required=True)
    parser.add_argument("--lno-checkpoint", required=True)
    parser.add_argument("--lno-satloss8-checkpoint", required=True)
    parser.add_argument("--mspt-checkpoint", required=True)
    parser.add_argument("--mspt-satloss8-checkpoint", required=True)
    parser.add_argument("--point-transformer-v3-checkpoint", required=True)
    parser.add_argument("--point-transformer-v3-satloss8-checkpoint", required=True)
    return parser.parse_args()


def resolve_devices(args: argparse.Namespace) -> List[torch.device]:
    text = args.devices or args.device
    if text:
        tokens = [item.strip() for item in str(text).split(",") if item.strip()]
        devices = [torch.device(f"cuda:{item}" if item.isdigit() else item) for item in tokens]
    elif torch.cuda.is_available():
        devices = [torch.device("cuda")]
    else:
        devices = [torch.device("cpu")]
    if any(device.type == "cuda" for device in devices) and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    for device in devices:
        if device.type == "cuda" and device.index is not None and device.index >= torch.cuda.device_count():
            raise ValueError(f"Requested {device}, but only {torch.cuda.device_count()} visible CUDA devices exist.")
    return devices


def config_and_checkpoint_maps(args: argparse.Namespace):
    configs = {
        "SMART": {"vanilla8": args.smart_config, "satloss8": args.smart_satloss8_config},
        "TRANSOLVERPP": {"vanilla8": args.transolverpp_config, "satloss8": args.transolverpp_satloss8_config},
        "POINTNET2_SSG": {"vanilla8": args.pointnet2_ssg_config, "satloss8": args.pointnet2_ssg_satloss8_config},
        "LNO": {"vanilla8": args.lno_config, "satloss8": args.lno_satloss8_config},
        "MSPT": {"vanilla8": args.mspt_config, "satloss8": args.mspt_satloss8_config},
        "POINT_TRANSFORMER_V3": {
            "vanilla8": args.point_transformer_v3_config,
            "satloss8": args.point_transformer_v3_satloss8_config,
        },
    }
    checkpoints = {
        "SMART": {"vanilla8": args.smart_checkpoint, "satloss8": args.smart_satloss8_checkpoint},
        "TRANSOLVERPP": {"vanilla8": args.transolverpp_checkpoint, "satloss8": args.transolverpp_satloss8_checkpoint},
        "POINTNET2_SSG": {"vanilla8": args.pointnet2_ssg_checkpoint, "satloss8": args.pointnet2_ssg_satloss8_checkpoint},
        "LNO": {"vanilla8": args.lno_checkpoint, "satloss8": args.lno_satloss8_checkpoint},
        "MSPT": {"vanilla8": args.mspt_checkpoint, "satloss8": args.mspt_satloss8_checkpoint},
        "POINT_TRANSFORMER_V3": {
            "vanilla8": args.point_transformer_v3_checkpoint,
            "satloss8": args.point_transformer_v3_satloss8_checkpoint,
        },
    }
    return configs, checkpoints


def load_split(path: Path, train_cluster: int, test_cluster: int) -> tuple[List[int], List[int]]:
    with path.open("r", encoding="utf-8") as handle:
        split = json.load(handle)
    key = f"train_cluster_{int(train_cluster)}_test_cluster_{int(test_cluster)}"
    entry = split.get(key)
    if not isinstance(entry, dict):
        raise ValueError(f"Missing `{key}` in {path}.")
    train_ids = sorted(int(item) for item in entry.get("train_ids", []))
    test_ids = sorted(int(item) for item in entry.get("test_ids", []))
    if not train_ids or not test_ids or set(train_ids).intersection(test_ids):
        raise ValueError(f"Invalid train/test IDs in `{key}`.")
    return train_ids, test_ids


def select_heldout_ids(test_ids: Sequence[int], args: argparse.Namespace) -> List[int]:
    available = sorted(int(item) for item in test_ids)
    if args.run_ids:
        chosen = [int(item.strip()) for item in str(args.run_ids).split(",") if item.strip()]
        missing = sorted(set(chosen).difference(available))
        if missing:
            raise ValueError(f"Requested held-out run IDs are unavailable: {missing}")
        return chosen
    if int(args.num_runs) <= 0 or int(args.num_runs) >= len(available):
        return available
    rng = np.random.default_rng(int(args.seed) + 7001)
    return sorted(int(item) for item in rng.choice(np.asarray(available), size=int(args.num_runs), replace=False))


def dataset_for_domain(config, split_path: Path, args: argparse.Namespace) -> AhmedMLDatasetV2:
    return AhmedMLDatasetV2(
        saved_folder=str(config.data_path),
        if_test=True,
        geometry_points=0,
        surface_points=int(args.surface_query_points),
        volume_points=int(args.volume_query_points),
        scale_positions=bool(config.scale_positions),
        require_preprocessed=True,
        domain_split_json=str(split_path),
        domain_split_train_cluster=int(args.train_cluster),
        domain_split_test_cluster=int(args.test_cluster),
    )


def load_case(dataset: AhmedMLDatasetV2, run_id: int, seed: int):
    try:
        index = dataset.data.index(int(run_id))
    except ValueError as exc:
        raise ValueError(f"run_{run_id} is not available in the held-out dataset.") from exc
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    item = dataset[index]
    if len(item) != 5:
        raise ValueError(f"Expected five tensors for run_{run_id}, received {len(item)}.")
    return tuple(value.float() for value in item)


def evaluate_family(
    family: str,
    models: Mapping[str, torch.nn.Module],
    configs: Mapping[str, object],
    device: torch.device,
    cases: Mapping[int, tuple[torch.Tensor, ...]],
    run_ids: Sequence[int],
    mean_s: torch.Tensor,
    std_s: torch.Tensor,
    mean_v: torch.Tensor,
    std_v: torch.Tensor,
    args: argparse.Namespace,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    budgets = {
        variant: int(train_encoder_input_points(configs[variant], family if variant == "vanilla8" else f"{family}_SATLOSS8"))
        for variant in VARIANTS
    }
    if budgets["vanilla8"] != budgets["satloss8"]:
        raise ValueError(f"{family} vanilla8/SATLOSS8 input budgets differ: {budgets}")
    input_budget = budgets["vanilla8"]
    family_seed = 100003 * (FAMILY_ORDER.index(family) + 1)
    for run_id in tqdm(run_ids, desc=f"{FAMILY_LABELS[family]} held-out", leave=False):
        geo, surf_query, surf_data, vol_query, vol_data = cases[int(run_id)]
        views = []
        for view_idx in range(max(1, int(args.views_per_run))):
            rng = np.random.default_rng([int(args.seed), int(run_id), int(view_idx), family_seed])
            indices = sample_uniform_without_replacement(int(geo.shape[0]), input_budget, rng)
            views.append(geo[indices])
        geo_views = torch.stack(views, dim=0)
        gt_s = denorm_fields(surf_data, mean_s, std_s).numpy()
        gt_v = denorm_fields(vol_data, mean_v, std_v).numpy()
        for variant in VARIANTS:
            model_name = family if variant == "vanilla8" else f"{family}_SATLOSS8"
            pred_s, pred_v = predict_view_batch(
                model_name,
                models[variant],
                geo_views,
                surf_query,
                vol_query,
                None,
                mean_s,
                std_s,
                mean_v,
                std_v,
                device,
                base_seed=int(args.seed + int(run_id) * 1009 + (0 if variant == "vanilla8" else 17)),
                repeats=1,
            )
            per_view = [
                compute_metrics(gt_s, pred_s[view_idx], gt_v, pred_v[view_idx])
                for view_idx in range(pred_s.shape[0])
            ]
            row: Dict[str, object] = {
                "family": family,
                "variant": variant,
                "run_id": int(run_id),
                "input_points": int(input_budget),
            }
            keys = sorted(per_view[0])
            for key in keys:
                row[key] = float(np.mean([metrics[key] for metrics in per_view]))
            rows.append(row)
    if device.type == "cuda":
        with torch.cuda.device(device):
            torch.cuda.synchronize(device)
    return rows


def paired_rows(rows: Sequence[Mapping[str, object]]) -> Dict[str, Dict[int, Dict[str, Mapping[str, object]]]]:
    output: Dict[str, Dict[int, Dict[str, Mapping[str, object]]]] = {}
    for row in rows:
        family = str(row["family"])
        run_id = int(row["run_id"])
        variant = str(row["variant"])
        output.setdefault(family, {}).setdefault(run_id, {})[variant] = row
    return output


def relative_gain(vanilla: float, satloss: float) -> float:
    return 100.0 * (float(vanilla) - float(satloss)) / max(abs(float(vanilla)), 1.0e-12)


def select_eligible_run_ids(
    rows: Sequence[Mapping[str, object]],
    family: str,
    minimum_improvement_percent: float,
) -> set[int]:
    paired = paired_rows(rows)
    eligible = set()
    for run_id, variants in paired.get(family, {}).items():
        if set(VARIANTS).difference(variants):
            continue
        improvement = relative_gain(
            float(variants["vanilla8"][METRIC_KEY]),
            float(variants["satloss8"][METRIC_KEY]),
        )
        if improvement > float(minimum_improvement_percent):
            eligible.add(int(run_id))
    return eligible


def parse_ranking_exclusions(text: str) -> set[str]:
    aliases = {
        "PTV3": "POINT_TRANSFORMER_V3",
        "POINTTRANSFORMERV3": "POINT_TRANSFORMER_V3",
        "POINT_TRANSFORMER_V3": "POINT_TRANSFORMER_V3",
    }
    excluded = set()
    for token in str(text).split(","):
        token = token.strip().upper().replace("-", "_")
        if not token:
            continue
        token = aliases.get(token, token)
        if token not in FAMILY_ORDER:
            raise ValueError(
                f"Unknown --exclude-from-ranking family {token!r}; "
                f"expected one of: {', '.join(FAMILY_ORDER)}"
            )
        excluded.add(token)
    if len(excluded) == len(FAMILY_ORDER):
        raise ValueError("--exclude-from-ranking cannot exclude every model family.")
    return excluded


def rank_geometries(
    rows: Sequence[Mapping[str, object]],
    top_k: int,
    positive_only: bool = False,
    positive_first: bool = False,
    eligible_run_ids: set[int] | None = None,
    ranking_families: set[str] | None = None,
):
    paired = paired_rows(rows)
    per_family: Dict[str, List[Dict[str, object]]] = {}
    consensus: Dict[int, Dict[str, object]] = {}
    for family in FAMILY_ORDER:
        entries = []
        for run_id, variants in paired.get(family, {}).items():
            if eligible_run_ids is not None and int(run_id) not in eligible_run_ids:
                continue
            if set(VARIANTS).difference(variants):
                continue
            vanilla = float(variants["vanilla8"][METRIC_KEY])
            satloss = float(variants["satloss8"][METRIC_KEY])
            item = {
                "family": family,
                "run_id": int(run_id),
                "vanilla8_error": vanilla,
                "satloss8_error": satloss,
                "absolute_improvement": vanilla - satloss,
                "relative_improvement_percent": relative_gain(vanilla, satloss),
            }
            entries.append(item)
            if ranking_families is None or family in ranking_families:
                consensus.setdefault(int(run_id), {"run_id": int(run_id), "gains": []})["gains"].append(
                    float(item["relative_improvement_percent"])
                )
        if positive_only:
            entries = [item for item in entries if float(item["relative_improvement_percent"]) > 0.0]
        entries.sort(
            key=lambda item: (
                (0 if float(item["relative_improvement_percent"]) > 0.0 else 1)
                if positive_first
                else 0,
                -float(item["relative_improvement_percent"]),
                int(item["run_id"]),
            )
        )
        per_family[family] = entries[: max(0, int(top_k))]
    consensus_rows = []
    for item in consensus.values():
        gains = item["gains"]
        mean_gain = float(np.mean(gains))
        consensus_rows.append(
            {
                "run_id": int(item["run_id"]),
                "summed_relative_improvement_percent": float(np.sum(gains)),
                "mean_relative_improvement_percent": mean_gain,
                "median_relative_improvement_percent": float(np.median(gains)),
                "families_present": int(len(gains)),
                "positive_family_count": int(sum(gain > 0.0 for gain in gains)),
                "positive_only_candidate": bool(float(np.sum(gains)) > 0.0),
            }
        )
    consensus_rows.sort(key=lambda item: (-float(item["summed_relative_improvement_percent"]), int(item["run_id"])))
    ranked_candidates = consensus_rows
    if positive_only:
        ranked_candidates = [
        item for item in consensus_rows if float(item["summed_relative_improvement_percent"]) > 0.0
        ]
    elif positive_first:
        ranked_candidates = sorted(
            consensus_rows,
            key=lambda item: (
                0 if float(item["summed_relative_improvement_percent"]) > 0.0 else 1,
                -float(item["summed_relative_improvement_percent"]),
                int(item["run_id"]),
            ),
        )
    return per_family, ranked_candidates[: max(0, int(top_k))], consensus_rows


def aggregate_family(rows: Sequence[Mapping[str, object]], selected_ids: Iterable[int] | None = None):
    selected = None if selected_ids is None else {int(item) for item in selected_ids}
    output = {}
    for family in FAMILY_ORDER:
        output[family] = {}
        for variant in VARIANTS:
            values = [
                float(row[METRIC_KEY])
                for row in rows
                if row["family"] == family
                and row["variant"] == variant
                and (selected is None or int(row["run_id"]) in selected)
            ]
            if values:
                output[family][variant] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "count": int(len(values)),
                }
    return output


def limits(ax, values: Sequence[float], log_scale: bool) -> None:
    finite = np.asarray([float(value) for value in values if math.isfinite(float(value)) and float(value) > 0.0], dtype=float)
    if finite.size == 0:
        return
    low = float(finite.min())
    high = float(finite.max())
    if log_scale:
        ax.set_ylim(max(low / 1.15, 1.0e-12), high * 1.15)
    else:
        span = max(high - low, high, 1.0e-12)
        ax.set_ylim(max(0.0, low - 0.10 * span), high + 0.10 * span)


def annotate_gain(ax, bar, gain: float, values: Sequence[float], log_scale: bool) -> None:
    if log_scale:
        y = max(float(bar.get_height()) * 1.12, 1.0e-12)
        va = "bottom"
    else:
        low = min(float(value) for value in values)
        high = max(float(value) for value in values)
        y = float(bar.get_height()) + 0.025 * max(high - low, high, 1.0e-12)
        va = "bottom"
    ax.text(
        bar.get_x() + bar.get_width() / 2.0,
        y,
        f"{gain:+.1f}%",
        ha="center",
        va=va,
        fontsize=font_size(0.78),
        rotation=90 if abs(gain) >= 100.0 else 0,
        clip_on=False,
    )


def plot_family_bars(summary, path: Path, title: str, log_scale: bool) -> None:
    families = [family for family in FAMILY_ORDER if family in summary and all(v in summary[family] for v in VARIANTS)]
    x = np.arange(len(families), dtype=float)
    width = 0.22
    offsets = {"vanilla8": -0.13, "satloss8": 0.13}
    values = []
    fig, ax = plt.subplots(figsize=(13.4, 7.4))
    fig.subplots_adjust(left=0.12, right=0.78, bottom=0.29, top=0.84)
    sat_bars = []
    for variant in VARIANTS:
        means = [float(summary[family][variant]["mean"]) for family in families]
        values.extend(means)
        bars = ax.bar(
            x + offsets[variant],
            means,
            width=width,
            color=[FAMILY_COLORS[family] for family in families],
            edgecolor="#222222",
            linewidth=0.7,
            alpha=0.96,
            hatch="///" if variant == "satloss8" else None,
            label=variant,
        )
        if variant == "satloss8":
            sat_bars = list(bars)
    for family, bar in zip(families, sat_bars):
        vanilla = float(summary[family]["vanilla8"]["mean"])
        satloss = float(summary[family]["satloss8"]["mean"])
        annotate_gain(ax, bar, relative_gain(vanilla, satloss), values, log_scale)
    if log_scale:
        ax.set_yscale("log")
    limits(ax, values, log_scale)
    size = font_size()
    ax.set_title(title, fontsize=size, pad=12)
    ax.set_ylabel("Combined-global relative L2 error", fontsize=size)
    ax.set_xticks(x)
    ax.set_xticklabels([FAMILY_LABELS[family] for family in families], rotation=24, ha="right", fontsize=size)
    ax.tick_params(axis="both", labelsize=size)
    ax.grid(axis="y", which="both", alpha=0.20, linewidth=0.8)
    ax.set_axisbelow(True)
    fig.legend(
        handles=[Patch(facecolor="#777777", edgecolor="#222222", label="vanilla8"), Patch(facecolor="#777777", edgecolor="#222222", hatch="///", label="SATLOSS8")],
        loc="upper left",
        bbox_to_anchor=(0.80, 0.77),
        framealpha=0.94,
        fontsize=size,
    )
    fig.legend(
        handles=[Patch(facecolor=FAMILY_COLORS[family], edgecolor="#222222", label=FAMILY_LABELS[family]) for family in families],
        loc="upper left",
        bbox_to_anchor=(0.80, 0.50),
        framealpha=0.94,
        fontsize=size,
    )
    save_plot(fig, path)


def plot_gain_bars(summary, path: Path, title: str) -> None:
    families = [family for family in FAMILY_ORDER if family in summary and all(v in summary[family] for v in VARIANTS)]
    gains = [
        relative_gain(summary[family]["vanilla8"]["mean"], summary[family]["satloss8"]["mean"])
        for family in families
    ]
    x = np.arange(len(families), dtype=float)
    size = font_size()
    fig, ax = plt.subplots(figsize=(13.4, 7.4))
    fig.subplots_adjust(left=0.12, right=0.78, bottom=0.29, top=0.84)
    bars = ax.bar(x, gains, width=0.42, color=[FAMILY_COLORS[family] for family in families], edgecolor="#222222", linewidth=0.7)
    ax.axhline(0.0, color="#333333", linewidth=1.1)
    span = max(max(gains, default=0.0) - min(gains, default=0.0), 1.0)
    ax.set_ylim(min(min(gains, default=0.0) - 0.10 * span, 0.0), max(max(gains, default=0.0) + 0.10 * span, 0.0))
    for bar, gain in zip(bars, gains):
        y = float(bar.get_height()) + (0.025 * span if gain >= 0.0 else -0.025 * span)
        ax.text(bar.get_x() + bar.get_width() / 2.0, y, f"{gain:+.1f}%", ha="center", va="bottom" if gain >= 0 else "top", fontsize=font_size(0.78), clip_on=False)
    ax.set_title(title, fontsize=size, pad=12)
    ax.set_ylabel("SATLOSS8 improvement relative to vanilla8 (%)", fontsize=size)
    ax.set_xticks(x)
    ax.set_xticklabels([FAMILY_LABELS[family] for family in families], rotation=24, ha="right", fontsize=size)
    ax.tick_params(axis="both", labelsize=size)
    ax.grid(axis="y", alpha=0.20, linewidth=0.8)
    ax.set_axisbelow(True)
    save_plot(fig, path)


def plot_per_family_top10(rows, family: str, selected_ids: Sequence[int], path: Path, log_scale: bool) -> None:
    selected = {int(item) for item in selected_ids}
    values = {variant: [] for variant in VARIANTS}
    gains = []
    for run_id in selected_ids:
        pair = {
            str(row["variant"]): float(row[METRIC_KEY])
            for row in rows
            if row["family"] == family and int(row["run_id"]) == int(run_id) and str(row["variant"]) in VARIANTS
        }
        if len(pair) != 2:
            continue
        values["vanilla8"].append(pair["vanilla8"])
        values["satloss8"].append(pair["satloss8"])
        gains.append(relative_gain(pair["vanilla8"], pair["satloss8"]))
    if not values["vanilla8"]:
        return
    x = np.arange(len(values["vanilla8"]), dtype=float)
    width = 0.32
    size = font_size()
    fig, ax = plt.subplots(figsize=(14.0, 7.4))
    fig.subplots_adjust(left=0.12, right=0.82, bottom=0.23, top=0.84)
    all_values = values["vanilla8"] + values["satloss8"]
    if log_scale:
        ax.set_yscale("log")
    limits(ax, all_values, log_scale)
    ax.bar(x - width / 2.0, values["vanilla8"], width=width, color=FAMILY_COLORS[family], edgecolor="#222222", label="vanilla8")
    sat_bars = ax.bar(x + width / 2.0, values["satloss8"], width=width, color=FAMILY_COLORS[family], edgecolor="#222222", hatch="///", label="SATLOSS8")
    for bar, gain in zip(sat_bars, gains):
        annotate_gain(ax, bar, gain, all_values, log_scale)
    ax.set_title(f"{FAMILY_LABELS[family]}: top-{len(values['vanilla8'])} held-out geometries", fontsize=size, pad=12)
    ax.set_ylabel("Combined-global relative L2 error", fontsize=size)
    ax.set_xlabel("Held-out geometry run ID", fontsize=size)
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(run_id)) for run_id in selected_ids[: len(values["vanilla8"])]], rotation=0, fontsize=size)
    ax.tick_params(axis="both", labelsize=size)
    ax.grid(axis="y", which="both", alpha=0.20, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 0.80), framealpha=0.94, fontsize=size)
    save_plot(fig, path)


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_summary_tables(
    output: Path,
    full_summary,
    top_summary,
    consensus_top,
    per_family_top,
    top_scope: str,
    selection_mode: str,
) -> None:
    rows = []
    for scope, summary in (("heldout_all", full_summary), (top_scope, top_summary)):
        for family in FAMILY_ORDER:
            if family not in summary or any(variant not in summary[family] for variant in VARIANTS):
                continue
            vanilla = summary[family]["vanilla8"]["mean"]
            satloss = summary[family]["satloss8"]["mean"]
            rows.append(
                {
                    "scope": scope,
                    "model": FAMILY_LABELS[family],
                    "vanilla8_combined_global_rel_l2": vanilla,
                    "satloss8_combined_global_rel_l2": satloss,
                    "satloss8_minus_vanilla8": satloss - vanilla,
                    "satloss8_improvement_percent": relative_gain(vanilla, satloss),
                    "count": summary[family]["vanilla8"]["count"],
                }
            )
    write_csv(output / "satloss8_domain_pairs_summary.csv", rows)
    lines = [
        "# Vanilla8 vs SATLOSS8 Geometry-Domain Comparison",
        "",
        "Cluster 0 is the training domain and cluster 1 is the held-out geometry domain. All paper-facing values are combined-global relative L2 means without standard-deviation bars.",
        "",
        "| Scope | Model | Vanilla8 | SATLOSS8 | SATLOSS8 improvement | N |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    lines.extend(
        f"| {row['scope']} | {row['model']} | {row['vanilla8_combined_global_rel_l2']:.6f} | {row['satloss8_combined_global_rel_l2']:.6f} | {row['satloss8_improvement_percent']:+.2f}% | {row['count']} |"
        for row in rows
    )
    (output / "satloss8_domain_pairs_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    top_rows = []
    for item in consensus_top:
        top_rows.append(dict(item))
    top_stem = f"satloss8_domain_{selection_mode}_consensus_top10"
    write_csv(output / f"{top_stem}.csv", top_rows)
    per_model_stem = f"satloss8_domain_{selection_mode}_top10_per_model.json"
    with (output / per_model_stem).open("w", encoding="utf-8") as handle:
        json.dump(per_family_top, handle, indent=2)


def main() -> None:
    args = parse_args()
    configure_plot_style(args.font_scale)
    scales = {item.strip().lower() for item in str(args.plot_scales).split(",") if item.strip()}
    if not scales or scales.difference({"linear", "log"}):
        raise ValueError("--plot-scales must contain linear, log, or both.")
    if int(args.top_k) <= 0:
        raise ValueError("--top-k must be positive.")
    if args.positive_only and args.positive_first:
        raise ValueError("--positive-only and --positive-first cannot be used together.")
    if args.min_transolverpp_improvement_percent is not None and not math.isfinite(
        float(args.min_transolverpp_improvement_percent)
    ):
        raise ValueError("--min-transolverpp-improvement-percent must be finite.")
    selection_mode = (
        "positive_only"
        if args.positive_only
        else "positive_first"
        if args.positive_first
        else "standard"
    )
    excluded_ranking_families = parse_ranking_exclusions(args.exclude_from_ranking)
    ranking_families = set(FAMILY_ORDER).difference(excluded_ranking_families)
    split_path = Path(args.split_json).expanduser().resolve()
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    train_ids, test_ids = load_split(split_path, args.train_cluster, args.test_cluster)
    selected_ids = select_heldout_ids(test_ids, args)
    if len(selected_ids) < int(args.top_k):
        raise ValueError(f"Need at least top-k={args.top_k} held-out geometries, got {len(selected_ids)}.")
    configs_by_family, checkpoints_by_family = config_and_checkpoint_maps(args)
    config_objects = {
        family: {variant: load_cfg(configs_by_family[family][variant]) for variant in VARIANTS}
        for family in FAMILY_ORDER
    }
    for family in FAMILY_ORDER:
        for variant in VARIANTS:
            path = Path(checkpoints_by_family[family][variant]).expanduser().resolve()
            if not path.is_file():
                raise FileNotFoundError(f"Missing {family} {variant} checkpoint: {path}")
            checkpoints_by_family[family][variant] = str(path)
    devices = resolve_devices(args)
    print(f"Domain split: train cluster {args.train_cluster} ({len(train_ids)}) -> held-out cluster {args.test_cluster} ({len(test_ids)})")
    print(f"Selected held-out geometries: {len(selected_ids)}; top-k visualization: {args.top_k}")
    print(f"Inference devices: {', '.join(str(device) for device in devices)}")
    for family in FAMILY_ORDER:
        for variant in VARIANTS:
            cfg = config_objects[family][variant]
            print(
                f"{family} {variant}: config={configs_by_family[family][variant]}, "
                f"input_points={train_encoder_input_points(cfg, family)}, "
                f"checkpoint={Path(checkpoints_by_family[family][variant]).name}"
            )
    if args.dry_run:
        return

    reference_cfg = config_objects["SMART"]["vanilla8"]
    if Path(str(reference_cfg.data_path)).expanduser().resolve() != Path(args.data_root).expanduser().resolve():
        raise ValueError(f"SMART config data_path does not match --data-root: {reference_cfg.data_path}")
    dataset = dataset_for_domain(reference_cfg, split_path, args)
    actual_test_ids = sorted(int(item) for item in dataset.test_ids)
    if not set(selected_ids).issubset(actual_test_ids):
        raise RuntimeError("Selected held-out IDs are not present in the dataset's cluster-1 role.")
    mean_s = dataset.mean_surf_data
    std_s = torch.clamp(dataset.std_surf_data, min=1.0e-12)
    mean_v = dataset.mean_vol_data
    std_v = torch.clamp(dataset.std_vol_data, min=1.0e-12)
    cases = {}
    for run_id in tqdm(selected_ids, desc="Loading held-out geometries"):
        cases[int(run_id)] = load_case(dataset, int(run_id), int(args.seed) + int(run_id) * 37)

    models_by_family = {}
    device_by_family = {family: devices[index % len(devices)] for index, family in enumerate(FAMILY_ORDER)}
    for family in FAMILY_ORDER:
        device = device_by_family[family]
        models_by_family[family] = {}
        for variant in VARIANTS:
            model_name = family if variant == "vanilla8" else f"{family}_SATLOSS8"
            print(f"Loading {FAMILY_LABELS[family]} {variant} on {device}")
            models_by_family[family][variant] = build_model(
                config_objects[family][variant],
                checkpoints_by_family[family][variant],
                device,
                int(args.batched_query_subregion_size),
            ).to(device)

    family_groups: Dict[torch.device, List[str]] = {device: [] for device in devices}
    for family, device in device_by_family.items():
        family_groups[device].append(family)

    def evaluate_device(device: torch.device, families: Sequence[str]):
        output_rows = []
        for family in families:
            output_rows.extend(
                evaluate_family(
                    family,
                    models_by_family[family],
                    config_objects[family],
                    device,
                    cases,
                    selected_ids,
                    mean_s,
                    std_s,
                    mean_v,
                    std_v,
                    args,
                )
            )
        return output_rows

    rows: List[Dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=len(devices)) as pool:
        futures = [pool.submit(evaluate_device, device, families) for device, families in family_groups.items() if families]
        for future in futures:
            rows.extend(future.result())
    rows.sort(key=lambda row: (str(row["family"]), int(row["run_id"]), str(row["variant"])))
    write_csv(output / "satloss8_domain_pairs_metrics.csv", rows)
    eligible_run_ids = None
    eligibility_description = "all scanned geometries"
    if args.min_transolverpp_improvement_percent is not None:
        eligible_run_ids = select_eligible_run_ids(
            rows,
            "TRANSOLVERPP",
            float(args.min_transolverpp_improvement_percent),
        )
        eligibility_description = (
            "TRANSOLVERPP SATLOSS8 relative improvement > "
            f"{float(args.min_transolverpp_improvement_percent):g}%"
        )
        print(
            f"Eligible ranking geometries: {len(eligible_run_ids)} / {len(selected_ids)} "
            f"({eligibility_description})"
        )
        if len(eligible_run_ids) < int(args.top_k):
            raise RuntimeError(
                f"Only {len(eligible_run_ids)} scanned geometries satisfy {eligibility_description}; "
                f"cannot produce top-k={args.top_k}. Increase --num-runs or lower the threshold."
            )
    per_family_top, consensus_top, consensus_all = rank_geometries(
        rows,
        int(args.top_k),
        positive_only=bool(args.positive_only),
        positive_first=bool(args.positive_first),
        eligible_run_ids=eligible_run_ids,
        ranking_families=ranking_families,
    )
    positive_candidate_count = sum(
        float(item["summed_relative_improvement_percent"]) > 0.0 for item in consensus_all
    )
    if args.positive_only and positive_candidate_count < int(args.top_k):
        raise RuntimeError(
            f"Only {positive_candidate_count} of {len(consensus_all)} scanned geometries have "
            f"positive mean SATLOSS8 improvement; cannot produce top-k={args.top_k} "
            "without including regressions. Increase --num-runs or lower --top-k."
        )
    consensus_ids = [int(item["run_id"]) for item in consensus_top]
    top_summary = aggregate_family(rows, consensus_ids)
    full_summary = aggregate_family(rows, selected_ids)
    top_scope = f"{selection_mode}_consensus_top10"
    write_summary_tables(
        output,
        full_summary,
        top_summary,
        consensus_top,
        per_family_top,
        top_scope=top_scope,
        selection_mode=selection_mode,
    )
    with (output / "satloss8_domain_pairs_protocol.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "split_json": str(split_path),
                "train_cluster": int(args.train_cluster),
                "test_cluster": int(args.test_cluster),
                "selected_heldout_run_ids": selected_ids,
                "consensus_top_run_ids": consensus_ids,
                "consensus_ranking": consensus_all,
                "top_k": int(args.top_k),
                "positive_only": bool(args.positive_only),
                "positive_first": bool(args.positive_first),
                "positive_consensus_candidate_count": int(positive_candidate_count),
                "ranking_candidate_filter": eligibility_description,
                "ranking_families": [family for family in FAMILY_ORDER if family in ranking_families],
                "excluded_from_ranking": [
                    family for family in FAMILY_ORDER if family in excluded_ranking_families
                ],
                "ranking_candidate_run_ids": (
                    sorted(int(run_id) for run_id in eligible_run_ids)
                    if eligible_run_ids is not None
                    else None
                ),
                "selection_rule": (
                    "rank summed paired relative improvement across model families after requiring "
                    "summed improvement > 0"
                    if args.positive_only
                    else "rank positive summed paired improvements first, then least-negative candidates"
                    if args.positive_first
                    else "rank summed paired relative improvement across model families"
                ),
                "views_per_run": int(args.views_per_run),
                "surface_query_points": int(args.surface_query_points),
                "volume_query_points": int(args.volume_query_points),
                "sampling_shift": "statistical geometry-domain shift only; no beta/sine/remeshing/masking shift",
                "ranking": "summed per-family relative combined-global improvement: sum(100*(vanilla8-SATLOSS8)/abs(vanilla8))",
                "models": {
                    family: {
                        variant: {
                            "config": configs_by_family[family][variant],
                            "checkpoint": checkpoints_by_family[family][variant],
                            "input_points": int(train_encoder_input_points(config_objects[family][variant], family)),
                        }
                        for variant in VARIANTS
                    }
                    for family in FAMILY_ORDER
                },
                "plot_protocol": {
                    "metric": METRIC_KEY,
                    "standard_deviation_bars": False,
                    "vanilla_style": "solid",
                    "satloss8_style": "/// hatch",
                    "family_style": "same saturated color for paired variants",
                    "plot_scales": sorted(scales),
                    "font_scale": float(args.font_scale),
                },
            },
            handle,
            indent=2,
        )
    for family in FAMILY_ORDER:
        family_ids = [int(item["run_id"]) for item in per_family_top[family]]
        prefix = f"{selection_mode}_top10"
        with (output / f"{prefix}_{family.lower()}_run_ids.json").open("w", encoding="utf-8") as handle:
            json.dump(family_ids, handle, indent=2)

    full_scope_label = f"Held-out cluster {args.test_cluster}: vanilla8 vs SATLOSS8 ({len(selected_ids)} geometries)"
    top_scope_label = (
        f"Positive-only top-{len(consensus_ids)} held-out geometries"
        if args.positive_only
        else f"Positive-first top-{len(consensus_ids)} held-out geometries"
        if args.positive_first
        else f"Consensus top-{len(consensus_ids)} held-out geometries"
    )
    full_stem = f"heldout{len(selected_ids)}"
    top_prefix = f"{selection_mode}_consensus_top10"
    if "linear" in scales:
        plot_family_bars(full_summary, output / f"{full_stem}_combined_global_bars_linear.png", full_scope_label, False)
        plot_family_bars(top_summary, output / f"{top_prefix}_combined_global_bars_linear.png", top_scope_label, False)
        for family in FAMILY_ORDER:
            plot_per_family_top10(rows, family, [int(item["run_id"]) for item in per_family_top[family]], output / f"{selection_mode}_top10_{family.lower()}_combined_global_bars_linear.png", False)
    if "log" in scales:
        plot_family_bars(full_summary, output / f"{full_stem}_combined_global_bars_log.png", full_scope_label, True)
        plot_family_bars(top_summary, output / f"{top_prefix}_combined_global_bars_log.png", top_scope_label, True)
        for family in FAMILY_ORDER:
            plot_per_family_top10(rows, family, [int(item["run_id"]) for item in per_family_top[family]], output / f"{selection_mode}_top10_{family.lower()}_combined_global_bars_log.png", True)
    plot_gain_bars(full_summary, output / f"{full_stem}_combined_global_improvement_vs_vanilla8.png", full_scope_label)
    plot_gain_bars(top_summary, output / f"{top_prefix}_combined_global_improvement_vs_vanilla8.png", top_scope_label)
    print(f"Wrote paired metrics and plots to {output}")


if __name__ == "__main__":
    main()
