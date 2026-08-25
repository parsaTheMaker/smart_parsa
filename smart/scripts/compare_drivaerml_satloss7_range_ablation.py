#!/usr/bin/env python3
"""Compare SMART against the SATLOSS7 shift-range ablation checkpoints.

The evaluated checkpoints are the actual SMART baseline and SATLOSS7 models
trained with shared beta ranges of 0.25, 0.50, 0.75, 1.00, 2.00, 3.00,
and 5.00. The sine-mixture intensity is a probability and remains bounded
to [0, 1] for every model.
Every model receives the same run, query, and encoder-view samples for a given
seed/mode. The only difference between ablation models is therefore the
checkpoint and its training protocol.

The evaluator produces:

* per-view and aggregate CSV/JSON metrics;
* absolute combined-global endpoint bar plots;
* paired percentage endpoint bars relative to SMART at the same shift;
* endpoint bar plots for every active shift.

Beta, sine-y, and sine-x are evaluated independently at the requested
severity levels. Severity zero is a shared uniform aligned control and is
reported for every active shift without a duplicate model forward. VTP
geometry-source modes are separate conditions: their remeshed coordinates are
used directly, with only uniform budget sampling and training-frame
normalization. No beta or sine transform is applied to a VTP mode.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import OrderedDict, defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Patch
from omegaconf import OmegaConf
from tqdm.auto import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
SMART_ROOT = SCRIPT_DIR.parent
REPO_ROOT = SMART_ROOT.parent
if str(SMART_ROOT) not in sys.path:
    sys.path.insert(0, str(SMART_ROOT))

from data.ahmedml_dataset_v2 import AhmedMLDatasetV2
from scripts.compare_drivaerml_sampling_invariance import (
    build_model,
    choose_fixed_query_indices,
    compute_metrics,
    denorm_fields,
    load_cfg,
    normalize_pos,
    predict_view_batch,
    sample_inverse_density_without_replacement,
    sample_uniform_weighted_mixture_without_replacement,
    sample_uniform_without_replacement,
    sinusoidal_axis_probabilities,
    geometry_source_vtp_path,
    parse_active_geometry_sources,
    parse_geometry_decimation_factors,
    read_vtp_points,
    train_encoder_input_points,
    validate_geometry_source_bbox,
)


MODEL_ORDER = (
    "SMART",
    "SMART_SATLOSS7_RANGE025",
    "SMART_SATLOSS7_RANGE050",
    "SMART_SATLOSS7_RANGE075",
    "SMART_SATLOSS7",
    "SMART_SATLOSS7_RANGE200",
    "SMART_SATLOSS7_RANGE300",
    "SMART_SATLOSS7_RANGE500",
)
MODEL_LABELS = {
    "SMART": "SMART baseline",
    "SMART_SATLOSS7_RANGE025": "SATLOSS [0, 0.25]",
    "SMART_SATLOSS7_RANGE050": "SATLOSS [0, 0.50]",
    "SMART_SATLOSS7_RANGE075": "SATLOSS [0, 0.75]",
    "SMART_SATLOSS7": "SATLOSS [0, 1.00]",
    "SMART_SATLOSS7_RANGE200": "SATLOSS [0, 2.00]",
    "SMART_SATLOSS7_RANGE300": "SATLOSS [0, 3.00]",
    "SMART_SATLOSS7_RANGE500": "SATLOSS [0, 5.00]",
}
MODEL_COLORS = {
    "SMART": "#4C78A8",
    "SMART_SATLOSS7_RANGE025": "#F2CF5B",
    "SMART_SATLOSS7_RANGE050": "#F28E2B",
    "SMART_SATLOSS7_RANGE075": "#E15759",
    "SMART_SATLOSS7": "#7A5195",
    "SMART_SATLOSS7_RANGE200": "#2F4B7C",
    "SMART_SATLOSS7_RANGE300": "#A05195",
    "SMART_SATLOSS7_RANGE500": "#D45087",
}
METRIC_KEYS = ("combined_global_rel_l2",)
SHIFT_ORDER = ("beta", "sine_y", "sine_x")
SHIFT_LABELS = {
    "beta": "Inverse-density beta",
    "sine_y": "Sinusoidal-y shift",
    "sine_x": "Sinusoidal-x shift",
}
GEOMETRY_SOURCE_LABELS = {
    f"{method}_div{factor}": f"{method.title()} div{factor}"
    for method in ("angle", "isotropic", "voxel")
    for factor in (5, 10, 20, 40)
}

_COMPUTE_PLOT_STD = True
_PLOT_FONT_SCALE = 1.0
_PLOT_BASE_FONT_SIZE = 15.0
_BAR_WIDTH_SCALE = 1.0
REFERENCE_MODEL = "SMART"
REFERENCE_MODEL_LABEL = "SMART baseline"
ABLATION_PREFIX = "range_ablation"
ABLATION_TABLE_TITLE = "SMART SATLOSS range ablation"

KDE_MODEL_ORDER = (
    "SMART",
    "SMART_SATLOSS7_KDE4",
    "SMART_SATLOSS7_KDE8",
    "SMART_SATLOSS7_KDE16",
    "SMART_SATLOSS7_KDE32",
    "SMART_SATLOSS7_KDE64",
)
KDE_MODEL_LABELS = {
    "SMART": "SMART baseline",
    "SMART_SATLOSS7_KDE4": "SATLOSS KDE k=4",
    "SMART_SATLOSS7_KDE8": "SATLOSS KDE k=8",
    "SMART_SATLOSS7_KDE16": "SATLOSS KDE k=16 (reference)",
    "SMART_SATLOSS7_KDE32": "SATLOSS KDE k=32",
    "SMART_SATLOSS7_KDE64": "SATLOSS KDE k=64",
}
KDE_MODEL_COLORS = {
    "SMART": "#4C78A8",
    "SMART_SATLOSS7_KDE4": "#F2CF5B",
    "SMART_SATLOSS7_KDE8": "#F28E2B",
    "SMART_SATLOSS7_KDE16": "#7A5195",
    "SMART_SATLOSS7_KDE32": "#59A14F",
    "SMART_SATLOSS7_KDE64": "#E15759",
}

DEAL_MODEL_ORDER = (
    "SMART",
    "DEAL_FIXED",
    "DEAL_GRADNORM",
    "DEAL_UNCERTAINTY",
    "DEAL_CONFIG",
)
DEAL_MODEL_LABELS = {
    "SMART": "SMART baseline",
    "DEAL_FIXED": "DeAL fixed weights",
    "DEAL_GRADNORM": "DeAL GradNorm",
    "DEAL_UNCERTAINTY": "DeAL uncertainty",
    "DEAL_CONFIG": "DeAL ConFIG",
}
DEAL_MODEL_COLORS = {
    "SMART": "#4C78A8",
    "DEAL_FIXED": "#7A5195",
    "DEAL_GRADNORM": "#F28E2B",
    "DEAL_UNCERTAINTY": "#59A14F",
    "DEAL_CONFIG": "#E15759",
}
DEAL_GEOMETRY_SOURCE_LABELS = {
    f"angle_div{factor}": f"Feature-aware div{factor}"
    for factor in (5, 10, 20, 40)
} | {
    f"isotropic_div{factor}": f"QEM div{factor}"
    for factor in (5, 10, 20, 40)
} | {
    f"voxel_div{factor}": f"Voxel-grid clustering div{factor}"
    for factor in (5, 10, 20, 40)
}
DEAL_GEOMETRY_METHOD_LABELS = {
    "angle": "Feature-aware",
    "isotropic": "QEM",
    "voxel": "Voxel-grid clustering",
}
GEOMETRY_METHOD_LABELS = {
    "angle": "Angle",
    "isotropic": "Isotropic",
    "voxel": "Voxel",
}
GEOMETRY_METHOD_FILE_SLUGS = {
    "angle": "angle",
    "isotropic": "isotropic",
    "voxel": "voxel",
}
DEAL_GEOMETRY_METHOD_FILE_SLUGS = {
    "angle": "feature_aware",
    "isotropic": "qem",
    "voxel": "voxel_grid",
}


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


def ablation_font_size(multiplier: float = 1.0) -> float:
    """Return the single font scale used by every ablation figure element."""
    return 0.55 * _PLOT_BASE_FONT_SIZE * float(_PLOT_FONT_SCALE) * float(multiplier)


def save_plot(fig: matplotlib.figure.Figure, output: Path) -> None:
    fig.set_constrained_layout(False)
    fig.savefig(output, dpi=280, bbox_inches="tight", pad_inches=0.18)


def apply_experiment_preset(args: argparse.Namespace) -> None:
    """Apply a named protocol without changing checkpoint/config semantics."""
    def set_default(option: str, attribute: str, value: object) -> None:
        explicit = any(
            token == option or token.startswith(f"{option}=")
            for token in sys.argv[1:]
        )
        if not explicit:
            setattr(args, attribute, value)

    preset = str(args.experiment_preset)
    if preset in {"range_ablation_vtp", "deal_weighting_ablation_vtp"}:
        set_default("--shift-levels", "shift_levels", "0,0.25,0.5,0.75,1.0")
        set_default("--active-shifts", "active_shifts", "beta,sine_y,sine_x")
        set_default("--active-geometry-sources", "active_geometry_sources", "angle,isotropic,voxel")
        set_default("--geometry-decimation-factors", "geometry_decimation_factors", "5,10")
        set_default("--num-runs", "num_runs", 25)
    elif preset == "range_ablation_shift_only":
        set_default("--shift-levels", "shift_levels", "0,0.25,0.5,0.75,1.0")
        set_default("--active-shifts", "active_shifts", "beta,sine_y,sine_x")
        set_default("--active-geometry-sources", "active_geometry_sources", "none")
        set_default("--geometry-decimation-factors", "geometry_decimation_factors", "5,10")
    elif preset == "range_ablation_vtp_only":
        set_default("--shift-levels", "shift_levels", "0,0.25,0.5,0.75,1.0")
        set_default("--active-shifts", "active_shifts", "beta")
        set_default("--active-geometry-sources", "active_geometry_sources", "angle,isotropic,voxel")
        set_default("--geometry-decimation-factors", "geometry_decimation_factors", "5,10")
    elif preset == "legacy_25runs":
        # This reproduces the previous ablation protocol's data-selection
        # scope while retaining the current checkpoint/config arguments.
        set_default("--shift-levels", "shift_levels", "0,0.25,0.5,0.75,1.0")
        set_default("--active-shifts", "active_shifts", "beta,sine_y,sine_x")
        set_default("--active-geometry-sources", "active_geometry_sources", "angle,isotropic,voxel")
        set_default("--geometry-decimation-factors", "geometry_decimation_factors", "5,10")
        set_default("--num-runs", "num_runs", 25)
    elif preset == "kde_ablation_vtp":
        set_default("--shift-levels", "shift_levels", "0,0.25,0.5,0.75,1.0")
        set_default("--active-shifts", "active_shifts", "beta,sine_y,sine_x")
        set_default("--active-geometry-sources", "active_geometry_sources", "angle,isotropic,voxel")
        set_default("--geometry-decimation-factors", "geometry_decimation_factors", "5,10")
        set_default("--num-runs", "num_runs", 25)
    else:  # pragma: no cover - argparse choices make this unreachable.
        raise ValueError(f"Unknown experiment preset: {preset}")


def parse_plot_scales(text: str) -> List[str]:
    values = [item.strip().lower() for item in str(text).split(",") if item.strip()]
    if not values or any(value not in {"log", "linear"} for value in values):
        raise ValueError("--plot-scales must contain log and/or linear.")
    return list(dict.fromkeys(values))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="/mnt/ssdraid/parsa/drivaerml_preprocessed")
    parser.add_argument(
        "--experiment-preset",
        choices=(
            "range_ablation_vtp",
            "range_ablation_shift_only",
            "range_ablation_vtp_only",
            "legacy_25runs",
            "kde_ablation_vtp",
            "deal_weighting_ablation_vtp",
        ),
        default="range_ablation_vtp",
        help="Named protocol preset; use explicit flags after choosing a preset to adapt the scope.",
    )
    parser.add_argument("--num-runs", type=int, default=25, help="Number of test geometries; 0 means all test geometries.")
    parser.add_argument("--run-ids", default=None, help="Optional comma-separated explicit test run IDs.")
    parser.add_argument(
        "--run-selection",
        choices=("random", "top_angle_div10_range_ablation", "top_pairwise_improvement", "top_deal_mean_improvement"),
        default="random",
        help="Select random geometries, legacy range selection, pairwise improvement, or mean DeAL improvement over SMART.",
    )
    parser.add_argument(
        "--top-selection-metric",
        choices=METRIC_KEYS,
        default="combined_global_rel_l2",
        help="Metric used when ranking geometries for top_angle_div10_range_ablation.",
    )
    parser.add_argument(
        "--top-selection-candidates",
        type=int,
        default=0,
        help="Candidate-pool size for top selection; 0 uses every available common candidate.",
    )
    parser.add_argument(
        "--top-selection-improved-model",
        default=None,
        help="Model key that should have lower error when --run-selection=top_pairwise_improvement.",
    )
    parser.add_argument(
        "--top-selection-reference-model",
        default=None,
        help="Model key used as the pairwise denominator when --run-selection=top_pairwise_improvement.",
    )
    parser.add_argument(
        "--top-selection-conditions",
        default="sine_y,sine_x,remeshing",
        help="Comma-separated ranking conditions: beta,sine_y,sine_x,remeshing. Remeshing includes every active VTP source/factor.",
    )
    parser.add_argument(
        "--top-selection-min-condition-improvement-percent",
        type=float,
        default=None,
        help="Optional eligibility floor applied to every ranking condition before averaging improvements.",
    )
    parser.add_argument(
        "--candidate-split",
        choices=("test", "all"),
        default="test",
        help="Candidate universe for random/top selection. `all` may include training-split geometries and is therefore not a held-out evaluation.",
    )
    parser.add_argument(
        "--screen-case-batch-size",
        type=int,
        default=1,
        help="Number of candidate geometries packed into each screening forward pass."
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shift-levels", default="0,0.25,0.5,0.75,1.0", help="Severity levels, including zero.")
    parser.add_argument(
        "--beta-levels",
        default=None,
        help="Optional beta severity levels. Overrides --shift-levels for beta and may exceed 1.",
    )
    parser.add_argument(
        "--sine-levels",
        default=None,
        help="Optional sine-mixture levels. Overrides --shift-levels for sine shifts and must be in [0, 1].",
    )
    parser.add_argument("--active-shifts", default="beta,sine_y,sine_x")
    parser.add_argument("--views-per-mode", type=int, default=2)
    parser.add_argument("--view-batch-size", type=int, default=2)
    parser.add_argument("--model-repeats", type=int, default=1)
    parser.add_argument("--surface-query-points", type=int, default=65536)
    parser.add_argument("--volume-query-points", type=int, default=65536)
    parser.add_argument("--batched-query-subregion-size", type=int, default=65536)
    parser.add_argument("--density-estimator", choices=("rk2", "tangent_cov", "kde"), default="kde")
    parser.add_argument("--density-knn-k", type=int, default=16)
    parser.add_argument("--query-sampling-with-replacement", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--devices", default="cuda:0", help="Comma-separated inference devices, e.g. cuda:0,cuda:1.")
    parser.add_argument("--active-geometry-sources", default="angle,isotropic,voxel")
    parser.add_argument("--geometry-decimation-factors", default="5,10")
    parser.add_argument("--angle-decimated-vtp-dir", default="/mnt/ssdraid/parsa/drivaerml_surface_vtp_decimated")
    parser.add_argument("--isotropic-decimated-vtp-dir", default="/mnt/ssdraid/parsa/drivaerml_surface_vtp_isotropic_gpu")
    parser.add_argument("--voxel-decimated-vtp-dir", default="/mnt/ssdraid/parsa/drivaerml_surface_vtp_voxel_quadric_clustered")
    parser.add_argument("--plot-scales", default="log,linear", help="Absolute-error plot scales, comma-separated.")
    parser.add_argument("--font-scale", type=float, default=1.2)
    parser.add_argument("--y-pad-fraction", type=float, default=0.10, help="Fractional vertical padding on every plot y-axis.")
    parser.add_argument("--no-std", action="store_true", help="Disable standard-deviation error bars in all plots.")
    parser.add_argument(
        "--compact-endpoint-summary",
        action="store_true",
        help="Add one linear chart with sine-x, sine-y, and mean div5/div10 remeshing endpoint groups.",
    )
    parser.add_argument("--smart-config", default="drivaerml")
    parser.add_argument("--range025-config", default="drivaerml_satloss7_range025")
    parser.add_argument("--range050-config", default="drivaerml_satloss7_range050")
    parser.add_argument("--range075-config", default="drivaerml_satloss7_range075")
    parser.add_argument("--satloss7-config", default="drivaerml_satloss7")
    parser.add_argument("--range200-config", default="drivaerml_satloss7_range200")
    parser.add_argument("--range300-config", default="drivaerml_satloss7_range300")
    parser.add_argument("--range500-config", default="drivaerml_satloss7_range500")
    parser.add_argument("--kde4-config", default="drivaerml_satloss7_range100_kde4_from_smart")
    parser.add_argument("--kde8-config", default="drivaerml_satloss7_range100_kde8_from_smart")
    parser.add_argument("--kde16-config", default="drivaerml_satloss7_range100")
    parser.add_argument("--kde32-config", default="drivaerml_satloss7_range100_kde32_from_smart")
    parser.add_argument("--kde64-config", default="drivaerml_satloss7_range100_kde64_from_smart")
    parser.add_argument("--deal-fixed-config", default="drivaerml_satloss7_range100")
    parser.add_argument("--deal-gradnorm-config", default="drivaerml_satloss7_gradnorm")
    parser.add_argument("--deal-uncertainty-config", default="drivaerml_satloss7_uncertainty")
    parser.add_argument("--deal-config-config", default="drivaerml_satloss7_config_full")
    parser.add_argument("--smart-checkpoint", default=None)
    parser.add_argument("--range025-checkpoint", default=None)
    parser.add_argument("--range050-checkpoint", default=None)
    parser.add_argument("--range075-checkpoint", default=None)
    parser.add_argument("--satloss7-checkpoint", default=None)
    parser.add_argument("--range200-checkpoint", default=None)
    parser.add_argument("--range300-checkpoint", default=None)
    parser.add_argument("--range500-checkpoint", default=None)
    parser.add_argument("--kde4-checkpoint", default=None)
    parser.add_argument("--kde8-checkpoint", default=None)
    parser.add_argument("--kde16-checkpoint", default=None)
    parser.add_argument("--kde32-checkpoint", default=None)
    parser.add_argument("--kde64-checkpoint", default=None)
    parser.add_argument("--deal-fixed-checkpoint", default=None)
    parser.add_argument("--deal-gradnorm-checkpoint", default=None)
    parser.add_argument("--deal-uncertainty-checkpoint", default=None)
    parser.add_argument("--deal-config-checkpoint", default=None)
    parser.add_argument(
        "--exclude-range500",
        action="store_true",
        help="Exclude the range-500 model and its plots from this comparison.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "results/drivaerml_smart_satloss7_range_ablation_25runs"),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def resolve_devices(text: str) -> List[torch.device]:
    devices: List[torch.device] = []
    for token in str(text).split(","):
        token = token.strip()
        if not token:
            continue
        if token.isdigit():
            token = f"cuda:{token}"
        devices.append(torch.device(token))
    if not devices:
        raise ValueError("At least one inference device is required.")
    if any(device.type == "cuda" for device in devices) and not torch.cuda.is_available():
        raise RuntimeError("CUDA devices were requested but CUDA is unavailable.")
    for device in devices:
        if device.type == "cuda" and device.index is not None and device.index >= torch.cuda.device_count():
            raise ValueError(f"Requested {device}, but only {torch.cuda.device_count()} visible CUDA devices exist.")
    return devices


def parse_levels(text: str) -> List[float]:
    values = sorted({round(float(item.strip()), 8) for item in str(text).split(",") if item.strip()})
    if not values or values[0] < 0.0:
        raise ValueError("Shift levels must be non-negative.")
    if values[0] != 0.0:
        values.insert(0, 0.0)
    return values


def levels_for_shift(levels: Sequence[float] | Mapping[str, Sequence[float]], shift: str) -> Sequence[float]:
    """Return the configured severity levels for one shift family."""
    if isinstance(levels, Mapping):
        return levels.get(shift, ())
    return levels


def parse_shifts(text: str) -> List[str]:
    aliases = {"sine": ("sine_y", "sine_x"), "all": SHIFT_ORDER}
    raw = [item.strip().lower().replace("-", "_") for item in str(text).split(",") if item.strip()]
    expanded: List[str] = []
    for item in raw:
        if item in aliases:
            expanded.extend(aliases[item])
        elif item in SHIFT_ORDER:
            expanded.append(item)
        else:
            raise ValueError(f"Unknown shift {item!r}; use beta, sine_y, sine_x, sine, or all.")
    selected = [shift for shift in SHIFT_ORDER if shift in set(expanded)]
    if not selected:
        raise ValueError("At least one shift must be active.")
    return selected


def select_run_ids(
    dataset: AhmedMLDatasetV2,
    args: argparse.Namespace,
    candidate_ids: Sequence[int] | None = None,
) -> List[int]:
    available = sorted(int(item) for item in (candidate_ids if candidate_ids is not None else dataset.test_ids))
    if args.run_ids:
        requested = [int(item.strip()) for item in str(args.run_ids).split(",") if item.strip()]
        missing = sorted(set(requested).difference(available))
        if missing:
            raise ValueError(f"Requested run IDs are not in the test split: {missing}")
        return requested
    if int(args.num_runs) <= 0 or int(args.num_runs) >= len(available):
        return available
    rng = np.random.default_rng(int(args.seed) + 7001)
    return sorted(int(item) for item in rng.choice(np.asarray(available), size=int(args.num_runs), replace=False))


def rank_top_range_ablation_runs(
    rows: Sequence[Mapping[str, object]],
    candidate_ids: Sequence[int],
    metric: str,
    requested_count: int,
) -> List[Dict[str, object]]:
    """Rank angle-div10 candidates by the three intermediate range models.

    A lower signed percentage means that a range checkpoint has lower error
    than SMART under the same angle-div10 geometry. The ranking score is the
    mean of the SMART-relative differences for ranges 0.25, 0.50, and 0.75;
    range 1.00 is deliberately excluded from selection.
    """
    selected_models = (
        "SMART_SATLOSS7_RANGE025",
        "SMART_SATLOSS7_RANGE050",
        "SMART_SATLOSS7_RANGE075",
    )
    groups: Dict[tuple[str, int], List[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        if str(row["shift_name"]) == "geometry_angle_div10":
            groups[(str(row["model_name"]), int(row["run_id"]))].append(row)

    ranked: List[Dict[str, object]] = []
    for run_id in sorted(int(value) for value in candidate_ids):
        smart_rows = groups.get(("SMART", run_id), [])
        if not smart_rows:
            continue
        smart_mean = float(np.mean([float(row[metric]) for row in smart_rows]))
        if not math.isfinite(smart_mean) or abs(smart_mean) < 1.0e-12:
            continue
        model_scores: Dict[str, float] = {}
        for model_name in selected_models:
            model_rows = groups.get((model_name, run_id), [])
            if not model_rows:
                break
            model_mean = float(np.mean([float(row[metric]) for row in model_rows]))
            model_scores[model_name] = 100.0 * (model_mean - smart_mean) / abs(smart_mean)
        if len(model_scores) != len(selected_models):
            continue
        ranked.append(
            {
                "run_id": run_id,
                "selection_metric": metric,
                "smart_angle_div10_mean": smart_mean,
                **{
                    f"{model_name.lower()}_relative_to_smart_pct": score
                    for model_name, score in model_scores.items()
                },
                "mean_intermediate_range_relative_to_smart_pct": float(np.mean(list(model_scores.values()))),
            }
        )
    ranked.sort(key=lambda row: (float(row["mean_intermediate_range_relative_to_smart_pct"]), int(row["run_id"])))
    if len(ranked) < int(requested_count):
        raise ValueError(
            f"Top-run selection found only {len(ranked)} complete angle_div10 candidates, "
            f"but {requested_count} were requested."
        )
    return ranked[: int(requested_count)]


def top_selection_conditions(
    conditions_text: str,
    active_shifts: Sequence[str],
    levels: Sequence[float] | Mapping[str, Sequence[float]],
    active_geometry_sources: Sequence[str],
) -> List[tuple[str, float]]:
    """Resolve endpoint conditions used only for pairwise candidate ranking."""
    valid = {"beta", "sine_y", "sine_x", "remeshing"}
    requested = [item.strip().lower().replace("-", "_") for item in str(conditions_text).split(",") if item.strip()]
    unknown = sorted(set(requested).difference(valid))
    if unknown:
        raise ValueError(f"Unknown --top-selection-conditions values: {unknown}. Valid values: {sorted(valid)}")
    selected: List[tuple[str, float]] = []
    for shift in SHIFT_ORDER:
        if shift in requested:
            if shift not in active_shifts:
                raise ValueError(f"Ranking condition `{shift}` is not active. Add it to --active-shifts.")
            selected.append((shift, float(max(levels_for_shift(levels, shift)))))
    if "remeshing" in requested:
        if not active_geometry_sources:
            raise ValueError("Ranking condition `remeshing` requires active VTP geometry sources.")
        selected.extend((f"geometry_{source}", 1.0) for source in active_geometry_sources)
    if not selected:
        raise ValueError("--top-selection-conditions resolved to no evaluated conditions.")
    return selected


def top_selection_condition_label(condition: tuple[str, float]) -> str:
    name, severity = condition
    return f"{name}_{severity:.2f}" if name in SHIFT_ORDER else name


def selection_modes(
    modes: Sequence[Mapping[str, object]],
    conditions: Sequence[tuple[str, float]],
) -> List[Mapping[str, object]]:
    """Keep only the exact endpoint modes that contribute to top-run ranking."""
    requested = {top_selection_condition_label(condition) for condition in conditions}
    selected = [mode for mode in modes if str(mode["name"]) in requested]
    selected_names = {str(mode["name"]) for mode in selected}
    if selected_names != requested:
        missing = sorted(requested.difference(selected_names))
        raise ValueError(f"Top-run selection modes are missing from the evaluation: {missing}")
    return selected


def rank_top_pairwise_improvement_runs(
    rows: Sequence[Mapping[str, object]],
    candidate_ids: Sequence[int],
    metric: str,
    requested_count: int,
    improved_model: str,
    reference_model: str,
    conditions: Sequence[tuple[str, float]],
    min_condition_improvement_percent: float | None = None,
) -> List[Dict[str, object]]:
    """Rank cases where one model most improves over another across fixed OOD modes.

    The score is the arithmetic mean of per-condition relative improvements:
    ``100 * (reference_error - improved_error) / abs(reference_error)``.
    Every selected case must have all requested sine/remeshing conditions for
    both models.  This selection criterion is saved to a sidecar file and is
    deliberately not added to plot titles.
    """
    grouped: Dict[tuple[str, int, str, float], List[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        model_name = str(row["model_name"])
        shift_name = str(row["shift_name"])
        severity = round(float(row["severity"]), 8)
        if model_name in {improved_model, reference_model} and (shift_name, severity) in conditions:
            grouped[(model_name, int(row["run_id"]), shift_name, severity)].append(row)

    ranked: List[Dict[str, object]] = []
    for run_id in sorted(int(value) for value in candidate_ids):
        improvements: Dict[str, float] = {}
        values: Dict[str, Dict[str, float]] = {}
        complete = True
        for condition in conditions:
            condition_name, condition_severity = condition
            condition_label = top_selection_condition_label(condition)
            improved_rows = grouped.get((improved_model, run_id, condition_name, round(condition_severity, 8)), [])
            reference_rows = grouped.get((reference_model, run_id, condition_name, round(condition_severity, 8)), [])
            if not improved_rows or not reference_rows:
                complete = False
                break
            improved_value = float(np.mean([float(row[metric]) for row in improved_rows]))
            reference_value = float(np.mean([float(row[metric]) for row in reference_rows]))
            if not (math.isfinite(improved_value) and math.isfinite(reference_value)) or abs(reference_value) < 1.0e-12:
                complete = False
                break
            improvements[condition_label] = 100.0 * (reference_value - improved_value) / abs(reference_value)
            values[condition_label] = {"improved": improved_value, "reference": reference_value}
        if not complete:
            continue
        if (
            min_condition_improvement_percent is not None
            and min(improvements.values()) < float(min_condition_improvement_percent)
        ):
            continue
        score = float(np.mean(list(improvements.values())))
        row: Dict[str, object] = {
            "run_id": run_id,
            "selection_metric": metric,
            "improved_model": improved_model,
            "reference_model": reference_model,
            "selection_conditions": [top_selection_condition_label(condition) for condition in conditions],
            "mean_pairwise_improvement_pct": score,
            "minimum_condition_improvement_pct": float(min(improvements.values())),
        }
        for condition, improvement in improvements.items():
            row[f"{condition}_improvement_pct"] = improvement
            row[f"{condition}_{improved_model.lower()}_error"] = values[condition]["improved"]
            row[f"{condition}_{reference_model.lower()}_error"] = values[condition]["reference"]
        ranked.append(row)
    ranked.sort(key=lambda row: (-float(row["mean_pairwise_improvement_pct"]), int(row["run_id"])))
    if len(ranked) < int(requested_count):
        qualification = (
            ""
            if min_condition_improvement_percent is None
            else f" satisfying the per-condition floor of {float(min_condition_improvement_percent):.3g}%"
        )
        raise ValueError(
            f"Pairwise top-run selection found only {len(ranked)} complete candidates{qualification}, "
            f"but {requested_count} were requested."
        )
    return ranked[: int(requested_count)]


def rank_top_deal_mean_improvement_runs(
    rows: Sequence[Mapping[str, object]],
    candidate_ids: Sequence[int],
    metric: str,
    requested_count: int,
    conditions: Sequence[tuple[str, float]],
) -> List[Dict[str, object]]:
    """Rank cases by mean SMART-relative improvement across every DeAL variant.

    Each candidate receives one relative improvement for every selected
    condition and each non-baseline DeAL model. The ranking score is their
    arithmetic mean, so a single favorable loss balancer cannot dominate
    selection. All values are computed on paired run/view samples.
    """
    deal_models = tuple(name for name in MODEL_ORDER if name != REFERENCE_MODEL)
    grouped: Dict[tuple[str, int, str, float], List[Mapping[str, object]]] = defaultdict(list)
    active_models = {REFERENCE_MODEL, *deal_models}
    for row in rows:
        model_name = str(row["model_name"])
        shift_name = str(row["shift_name"])
        severity = round(float(row["severity"]), 8)
        if model_name in active_models and (shift_name, severity) in conditions:
            grouped[(model_name, int(row["run_id"]), shift_name, severity)].append(row)

    ranked: List[Dict[str, object]] = []
    for run_id in sorted(int(value) for value in candidate_ids):
        all_improvements: List[float] = []
        model_scores: Dict[str, List[float]] = {model_name: [] for model_name in deal_models}
        condition_scores: Dict[str, List[float]] = defaultdict(list)
        complete = True
        for condition_name, condition_severity in conditions:
            key_severity = round(float(condition_severity), 8)
            smart_rows = grouped.get((REFERENCE_MODEL, run_id, condition_name, key_severity), [])
            if not smart_rows:
                complete = False
                break
            smart_value = float(np.mean([float(row[metric]) for row in smart_rows]))
            if not math.isfinite(smart_value) or abs(smart_value) < 1.0e-12:
                complete = False
                break
            condition_label = top_selection_condition_label((condition_name, condition_severity))
            for model_name in deal_models:
                deal_rows = grouped.get((model_name, run_id, condition_name, key_severity), [])
                if not deal_rows:
                    complete = False
                    break
                deal_value = float(np.mean([float(row[metric]) for row in deal_rows]))
                if not math.isfinite(deal_value):
                    complete = False
                    break
                improvement = 100.0 * (smart_value - deal_value) / abs(smart_value)
                all_improvements.append(improvement)
                model_scores[model_name].append(improvement)
                condition_scores[condition_label].append(improvement)
            if not complete:
                break
        if not complete or not all_improvements:
            continue
        item: Dict[str, object] = {
            "run_id": run_id,
            "selection_metric": metric,
            "reference_model": REFERENCE_MODEL,
            "deal_models": list(deal_models),
            "selection_conditions": [top_selection_condition_label(condition) for condition in conditions],
            "mean_deal_improvement_pct": float(np.mean(all_improvements)),
            "minimum_deal_improvement_pct": float(np.min(all_improvements)),
        }
        item.update({
            f"{model_name.lower()}_mean_improvement_pct": float(np.mean(values))
            for model_name, values in model_scores.items()
        })
        item.update({
            f"{condition}_mean_deal_improvement_pct": float(np.mean(values))
            for condition, values in condition_scores.items()
        })
        ranked.append(item)
    ranked.sort(key=lambda row: (-float(row["mean_deal_improvement_pct"]), int(row["run_id"])))
    if len(ranked) < int(requested_count):
        raise ValueError(
            f"DeAL top-run selection found only {len(ranked)} complete candidates, but {requested_count} were requested."
        )
    return ranked[: int(requested_count)]


def checkpoint_map(args: argparse.Namespace) -> OrderedDict[str, str]:
    if args.experiment_preset == "deal_weighting_ablation_vtp":
        checkpoints = OrderedDict(
            [
                ("SMART", args.smart_checkpoint),
                ("DEAL_FIXED", args.deal_fixed_checkpoint),
                ("DEAL_GRADNORM", args.deal_gradnorm_checkpoint),
                ("DEAL_UNCERTAINTY", args.deal_uncertainty_checkpoint),
                ("DEAL_CONFIG", args.deal_config_checkpoint),
            ]
        )
        missing = [name for name, path in checkpoints.items() if not path]
        if missing:
            raise ValueError("DeAL weighting ablation requires checkpoints for: " + ", ".join(missing))
        return checkpoints
    if args.experiment_preset == "kde_ablation_vtp":
        checkpoints = OrderedDict(
            [
                ("SMART", args.smart_checkpoint),
                ("SMART_SATLOSS7_KDE4", args.kde4_checkpoint),
                ("SMART_SATLOSS7_KDE8", args.kde8_checkpoint),
                ("SMART_SATLOSS7_KDE16", args.kde16_checkpoint),
                ("SMART_SATLOSS7_KDE32", args.kde32_checkpoint),
                ("SMART_SATLOSS7_KDE64", args.kde64_checkpoint),
            ]
        )
        missing = [name for name, path in checkpoints.items() if not path]
        if missing:
            raise ValueError(
                "KDE ablation requires checkpoints for: " + ", ".join(missing)
            )
        return checkpoints
    checkpoints = OrderedDict(
        [
            ("SMART", args.smart_checkpoint),
            ("SMART_SATLOSS7_RANGE025", args.range025_checkpoint),
            ("SMART_SATLOSS7_RANGE050", args.range050_checkpoint),
            ("SMART_SATLOSS7_RANGE075", args.range075_checkpoint),
            ("SMART_SATLOSS7", args.satloss7_checkpoint),
            ("SMART_SATLOSS7_RANGE200", args.range200_checkpoint),
            ("SMART_SATLOSS7_RANGE300", args.range300_checkpoint),
        ]
    )
    if not args.exclude_range500:
        if not args.range500_checkpoint:
            raise ValueError("--range500-checkpoint is required unless --exclude-range500 is set.")
        checkpoints["SMART_SATLOSS7_RANGE500"] = args.range500_checkpoint
    missing = [name for name, path in checkpoints.items() if not path]
    if missing:
        raise ValueError("Missing required checkpoint arguments for: " + ", ".join(missing))
    return checkpoints


def config_map(args: argparse.Namespace) -> OrderedDict[str, str]:
    if args.experiment_preset == "deal_weighting_ablation_vtp":
        return OrderedDict(
            [
                ("SMART", args.smart_config),
                ("DEAL_FIXED", args.deal_fixed_config),
                ("DEAL_GRADNORM", args.deal_gradnorm_config),
                ("DEAL_UNCERTAINTY", args.deal_uncertainty_config),
                ("DEAL_CONFIG", args.deal_config_config),
            ]
        )
    if args.experiment_preset == "kde_ablation_vtp":
        return OrderedDict(
            [
                ("SMART", args.smart_config),
                ("SMART_SATLOSS7_KDE4", args.kde4_config),
                ("SMART_SATLOSS7_KDE8", args.kde8_config),
                ("SMART_SATLOSS7_KDE16", args.kde16_config),
                ("SMART_SATLOSS7_KDE32", args.kde32_config),
                ("SMART_SATLOSS7_KDE64", args.kde64_config),
            ]
        )
    configs = OrderedDict(
        [
            ("SMART", args.smart_config),
            ("SMART_SATLOSS7_RANGE025", args.range025_config),
            ("SMART_SATLOSS7_RANGE050", args.range050_config),
            ("SMART_SATLOSS7_RANGE075", args.range075_config),
            ("SMART_SATLOSS7", args.satloss7_config),
            ("SMART_SATLOSS7_RANGE200", args.range200_config),
            ("SMART_SATLOSS7_RANGE300", args.range300_config),
        ]
    )
    if not args.exclude_range500:
        configs["SMART_SATLOSS7_RANGE500"] = args.range500_config
    return configs


def build_ablation_model(config, model_name: str, checkpoint: str, device: torch.device, query_chunk: int):
    build_config = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
    build_config.model_name = "SMART" if model_name == "SMART" else "SMART_SATLOSS7"
    model = build_model(build_config, checkpoint, device, batched_query_subregion_size=int(query_chunk))
    return model.to(device)


def load_case(data_root: Path, run_id: int) -> Dict[str, np.ndarray]:
    run_dir = data_root / f"run_{int(run_id)}"
    surf_coords = np.load(run_dir / "surface_coords.npy").astype(np.float32, copy=False)
    surf_gt = np.concatenate(
        [
            np.load(run_dir / "surface_pMeanTrim.npy").astype(np.float32, copy=False).reshape(-1, 1),
            np.load(run_dir / "surface_normals.npy").astype(np.float32, copy=False),
            np.load(run_dir / "surface_wallShearStressMeanTrim_x.npy").astype(np.float32, copy=False).reshape(-1, 1),
            np.load(run_dir / "surface_wallShearStressMeanTrim_y.npy").astype(np.float32, copy=False).reshape(-1, 1),
            np.load(run_dir / "surface_wallShearStressMeanTrim_z.npy").astype(np.float32, copy=False).reshape(-1, 1),
        ],
        axis=1,
    )
    vol_coords = np.load(run_dir / "volume_coords.npy").astype(np.float32, copy=False)
    vol_gt = np.concatenate(
        [
            np.load(run_dir / "volume_pMeanTrim.npy").astype(np.float32, copy=False).reshape(-1, 1),
            np.load(run_dir / "volume_UMeanTrim.npy").astype(np.float32, copy=False),
        ],
        axis=1,
    )
    if surf_coords.shape[0] != surf_gt.shape[0] or vol_coords.shape[0] != vol_gt.shape[0]:
        raise ValueError(f"run_{run_id} coordinate/field lengths do not match.")
    return {
        "surf_coords": np.ascontiguousarray(surf_coords),
        "surf_gt": np.ascontiguousarray(surf_gt),
        "vol_coords": np.ascontiguousarray(vol_coords),
        "vol_gt": np.ascontiguousarray(vol_gt),
    }


def build_modes(
    active_shifts: Sequence[str],
    levels: Sequence[float] | Mapping[str, Sequence[float]],
    active_geometry_sources: Sequence[str],
) -> List[Dict[str, object]]:
    modes: List[Dict[str, object]] = [
        {
            "name": "aligned_uniform_wor",
            "kind": "uniform_wor",
            "shift": None,
            "severity": 0.0,
            "id": 0,
        }
    ]
    mode_id = 1
    for shift in active_shifts:
        for severity in levels_for_shift(levels, shift):
            if severity <= 0.0:
                continue
            if shift == "beta":
                kind = "inverse_density_wor"
            else:
                kind = "sinusoidal_axis_mixture_wor"
            modes.append(
                {
                    "name": f"{shift}_{severity:.2f}",
                    "kind": kind,
                    "shift": shift,
                    "severity": float(severity),
                    "id": mode_id,
                }
            )
            mode_id += 1
    for source in active_geometry_sources:
        modes.append(
            {
                "name": f"geometry_{source}",
                "kind": "geometry_vtp",
                "shift": None,
                "condition": f"geometry_{source}",
                "severity": 1.0,
                "geometry_source": source,
                "id": mode_id,
            }
        )
        mode_id += 1
    return modes


def make_query_data(
    case: Mapping[str, np.ndarray],
    args: argparse.Namespace,
    min_pos: torch.Tensor,
    max_pos: torch.Tensor,
) -> Dict[str, object]:
    surf_coords = case["surf_coords"]
    vol_coords = case["vol_coords"]
    surf_idx = choose_fixed_query_indices(
        surf_coords.shape[0],
        int(args.surface_query_points),
        # Match the fixed query seeds used by the main SATLOSS7/8 evaluator.
        [int(args.seed), int(case["run_id"]), 3001],
        replace=bool(args.query_sampling_with_replacement),
    )
    vol_idx = choose_fixed_query_indices(
        vol_coords.shape[0],
        int(args.volume_query_points),
        [int(args.seed), int(case["run_id"]), 3002],
        replace=bool(args.query_sampling_with_replacement),
    )
    surf_coords_selected = surf_coords[surf_idx]
    vol_coords_selected = vol_coords[vol_idx]
    return {
        "surf_query_norm": normalize_pos(torch.from_numpy(surf_coords_selected), min_pos, max_pos),
        "vol_query_norm": normalize_pos(torch.from_numpy(vol_coords_selected), min_pos, max_pos),
        "surf_gt": case["surf_gt"][surf_idx],
        "vol_gt": case["vol_gt"][vol_idx],
        "full_surf_norm": normalize_pos(torch.from_numpy(surf_coords), min_pos, max_pos),
    }


def sample_mode_indices(
    mode: Mapping[str, object],
    surf_coords: np.ndarray,
    log_density: np.ndarray | None,
    sine_weights: Mapping[str, np.ndarray],
    budget: int,
    seed: int,
    run_id: int,
    views: int,
    geometry_source_points: Mapping[str, np.ndarray],
) -> List[np.ndarray]:
    result: List[np.ndarray] = []
    for view_id in range(int(views)):
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(run_id), int(mode["id"]), view_id]))
        kind = str(mode["kind"])
        if kind == "geometry_vtp":
            if mode.get("shift") is not None or "geometry_source" not in mode:
                raise RuntimeError("VTP geometry modes must not carry a beta/sine shift or density weighting.")
            source = geometry_source_points[str(mode["geometry_source"])]
            indices = sample_uniform_without_replacement(source.shape[0], budget, rng)
        elif kind == "uniform_wor":
            indices = sample_uniform_without_replacement(surf_coords.shape[0], budget, rng)
        elif kind == "inverse_density_wor":
            if log_density is None:
                raise RuntimeError("Inverse-density sampling requested without a density field.")
            indices = sample_inverse_density_without_replacement(
                log_density,
                budget,
                float(mode["severity"]),
                rng,
            )
        elif kind == "sinusoidal_axis_mixture_wor":
            shift = str(mode["shift"])
            indices = sample_uniform_weighted_mixture_without_replacement(
                sine_weights[shift],
                budget,
                float(mode["severity"]),
                rng,
            )
        else:
            raise ValueError(f"Unsupported mode kind: {kind}")
        result.append(indices)
    return result


def geometry_sampling_seed(global_seed: int, mode_id: int, run_id: int, view_id: int) -> int:
    """Stable SMART encoder-sampling seed for one evaluated geometry view."""
    return int(
        (int(global_seed) + 1_000_003 * int(mode_id) + 10_007 * int(run_id) + 101 * int(view_id))
        % (2**63 - 1)
    )


def evaluate_model_run(
    model_name: str,
    model,
    device: torch.device,
    case: Mapping[str, np.ndarray],
    query: Mapping[str, object],
    modes: Sequence[Mapping[str, object]],
    log_density: np.ndarray | None,
    sine_weights: Mapping[str, np.ndarray],
    input_budget: int,
    args: argparse.Namespace,
    mean_s: torch.Tensor,
    std_s: torch.Tensor,
    mean_v: torch.Tensor,
    std_v: torch.Tensor,
    geometry_source_norm: Mapping[str, torch.Tensor],
    geometry_source_points: Mapping[str, np.ndarray],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    mode_rows: Dict[str, List[Dict[str, object]]] = {}
    for mode in modes:
        indices = sample_mode_indices(
            mode,
            case["surf_coords"],
            log_density,
            sine_weights,
            int(input_budget),
            int(args.seed),
            int(case["run_id"]),
            int(args.views_per_mode),
            geometry_source_points,
        )
        mode_result: List[Dict[str, object]] = []
        for batch_start in range(0, int(args.views_per_mode), int(args.view_batch_size)):
            batch_stop = min(batch_start + int(args.view_batch_size), int(args.views_per_mode))
            batch_indices = indices[batch_start:batch_stop]
            if str(mode["kind"]) == "geometry_vtp":
                source_norm = geometry_source_norm[str(mode["geometry_source"])]
                geo_views = torch.stack(
                    [source_norm[torch.from_numpy(indexes)] for indexes in batch_indices],
                    dim=0,
                )
            else:
                geo_views = torch.stack(
                    [query["full_surf_norm"][torch.from_numpy(indexes)] for indexes in batch_indices],
                    dim=0,
                )
            pred_surf, pred_vol = predict_view_batch(
                model_name=model_name,
                model=model,
                geo_views_norm=geo_views,
                surf_query_norm=query["surf_query_norm"],
                vol_query_norm=query["vol_query_norm"],
                geo_log_density_views=None,
                mean_s=mean_s,
                std_s=std_s,
                mean_v=mean_v,
                std_v=std_v,
                device=device,
                base_seed=int(args.seed + 100000 * int(mode["id"]) + 1000 * int(case["run_id"]) + batch_start * 17),
                repeats=int(args.model_repeats),
                geometry_sampling_seeds=torch.tensor(
                    [
                        geometry_sampling_seed(args.seed, int(mode["id"]), int(case["run_id"]), view_id)
                        for view_id in range(batch_start, batch_stop)
                    ],
                    dtype=torch.long,
                ),
            )
            for local_view, view_id in enumerate(range(batch_start, batch_stop)):
                metrics = compute_metrics(
                    query["surf_gt"],
                    pred_surf[local_view],
                    query["vol_gt"],
                    pred_vol[local_view],
                )
                mode_result.append(
                    {
                        "run_id": int(case["run_id"]),
                        "view_id": int(view_id),
                        "model_name": model_name,
                        "sampling_mode": str(mode["name"]),
                        "shift_name": str(mode.get("condition", mode["shift"] or "aligned")),
                        "severity": float(mode["severity"]),
                        "sampling_kind": str(mode["kind"]),
                        "geometry_source": str(mode.get("geometry_source", "preprocessed_surface")),
                        "input_points": int(input_budget),
                        "surface_query_points": int(query["surf_gt"].shape[0]),
                        "volume_query_points": int(query["vol_gt"].shape[0]),
                        "checkpoint": str(args.checkpoints[model_name]),
                        **metrics,
                    }
                )
        mode_rows[str(mode["name"])] = mode_result

    # Pairwise candidate screening evaluates only endpoint modes.  It does not
    # need the aligned control rows that full plotting uses for severity zero.
    if "aligned_uniform_wor" not in mode_rows:
        for mode in modes:
            rows.extend(mode_rows[str(mode["name"])])
        return rows

    aligned_rows = mode_rows["aligned_uniform_wor"]
    for shift in [*args.active_shifts, *(f"geometry_{source}" for source in args.active_geometry_sources)]:
        for row in aligned_rows:
            rows.append({**row, "shift_name": shift, "severity": 0.0})
    for mode in modes:
        if str(mode["name"]) != "aligned_uniform_wor":
            rows.extend(mode_rows[str(mode["name"])])
    return rows


@torch.inference_mode()
def predict_packed_case_views(
    model,
    geo_views_norm: torch.Tensor,
    surf_query_norm: torch.Tensor,
    vol_query_norm: torch.Tensor,
    mean_s: torch.Tensor,
    std_s: torch.Tensor,
    mean_v: torch.Tensor,
    std_v: torch.Tensor,
    device: torch.device,
    base_seed: int,
    repeats: int,
    geometry_sampling_seeds: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray]:
    """Infer independently queried cases packed into one GPU batch."""
    batch_size = int(geo_views_norm.shape[0])
    if int(surf_query_norm.shape[0]) != batch_size or int(vol_query_norm.shape[0]) != batch_size:
        raise ValueError("Packed geometry and query batches must have the same batch size.")
    if int(geometry_sampling_seeds.numel()) != batch_size:
        raise ValueError("Packed geometry and sampling-seed batches must have the same batch size.")
    geo_b = geo_views_norm.to(device, non_blocking=True)
    surf_q_b = surf_query_norm.to(device, non_blocking=True)
    vol_q_b = vol_query_norm.to(device, non_blocking=True)
    geometry_sampling_b = geometry_sampling_seeds.to(device=device, dtype=torch.long, non_blocking=True)

    surf_acc = None
    vol_acc = None
    for repeat in range(int(repeats)):
        seed = int(base_seed + repeat)
        if device.type == "cuda":
            with torch.cuda.device(device):
                torch.cuda.manual_seed(seed)
        else:
            torch.manual_seed(seed)
        with (torch.cuda.device(device) if device.type == "cuda" else nullcontext()):
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                pred_s_norm, pred_v_norm = model.inference(
                    geo_b,
                    surf_q_b,
                    vol_q_b,
                    None,
                    geometry_sampling_seeds=geometry_sampling_b,
                )
        pred_s = denorm_fields(pred_s_norm.cpu(), mean_s, std_s)
        pred_v = denorm_fields(pred_v_norm.cpu(), mean_v, std_v)
        surf_acc = pred_s if surf_acc is None else (surf_acc + pred_s)
        vol_acc = pred_v if vol_acc is None else (vol_acc + pred_v)
    return (surf_acc / float(repeats)).numpy(), (vol_acc / float(repeats)).numpy()


def evaluate_model_case_batch(
    model_name: str,
    model,
    device: torch.device,
    prepared_cases: Sequence[tuple[Dict[str, np.ndarray], Dict[str, object], Dict[str, torch.Tensor], Dict[str, np.ndarray], np.ndarray | None, Dict[str, np.ndarray]]],
    modes: Sequence[Mapping[str, object]],
    input_budget: int,
    args: argparse.Namespace,
    mean_s: torch.Tensor,
    std_s: torch.Tensor,
    mean_v: torch.Tensor,
    std_v: torch.Tensor,
) -> List[Dict[str, object]]:
    """Evaluate endpoint-only modes with several independent cases per forward pass."""
    if not prepared_cases:
        return []
    if any(str(mode["name"]) == "aligned_uniform_wor" for mode in modes):
        raise ValueError("Packed candidate screening must receive endpoint modes only.")

    rows: List[Dict[str, object]] = []
    for mode in modes:
        sampled_views = []
        for case, query, source_norm, source_points, log_density, sine_weights in prepared_cases:
            del source_norm
            sampled_views.append(
                sample_mode_indices(
                    mode,
                    case["surf_coords"],
                    log_density,
                    sine_weights,
                    int(input_budget),
                    int(args.seed),
                    int(case["run_id"]),
                    int(args.views_per_mode),
                    source_points,
                )
            )

        for view_start in range(0, int(args.views_per_mode), int(args.view_batch_size)):
            view_stop = min(view_start + int(args.view_batch_size), int(args.views_per_mode))
            packed_geo: List[torch.Tensor] = []
            packed_surf_query: List[torch.Tensor] = []
            packed_vol_query: List[torch.Tensor] = []
            metadata: List[tuple[Mapping[str, np.ndarray], Mapping[str, object], int]] = []
            for prepared, indices_per_view in zip(prepared_cases, sampled_views):
                case, query, source_norm, _source_points, _log_density, _sine_weights = prepared
                for view_id in range(view_start, view_stop):
                    indices = indices_per_view[view_id]
                    if str(mode["kind"]) == "geometry_vtp":
                        geometry = source_norm[str(mode["geometry_source"])][torch.from_numpy(indices)]
                    else:
                        geometry = query["full_surf_norm"][torch.from_numpy(indices)]
                    packed_geo.append(geometry)
                    packed_surf_query.append(query["surf_query_norm"])
                    packed_vol_query.append(query["vol_query_norm"])
                    metadata.append((case, query, view_id))

            packed_run_seed = sum((index + 1) * int(case["run_id"]) for index, (case, _query, _view) in enumerate(metadata))
            pred_surf, pred_vol = predict_packed_case_views(
                model,
                torch.stack(packed_geo, dim=0),
                torch.stack(packed_surf_query, dim=0),
                torch.stack(packed_vol_query, dim=0),
                mean_s,
                std_s,
                mean_v,
                std_v,
                device,
                base_seed=int(args.seed + 100000 * int(mode["id"]) + packed_run_seed + view_start * 17),
                repeats=int(args.model_repeats),
                geometry_sampling_seeds=torch.tensor(
                    [
                        geometry_sampling_seed(args.seed, int(mode["id"]), int(case["run_id"]), view_id)
                        for case, _query, view_id in metadata
                    ],
                    dtype=torch.long,
                ),
            )
            for packed_index, (case, query, view_id) in enumerate(metadata):
                metrics = compute_metrics(
                    query["surf_gt"],
                    pred_surf[packed_index],
                    query["vol_gt"],
                    pred_vol[packed_index],
                )
                rows.append(
                    {
                        "run_id": int(case["run_id"]),
                        "view_id": int(view_id),
                        "model_name": model_name,
                        "sampling_mode": str(mode["name"]),
                        "shift_name": str(mode.get("condition", mode["shift"] or "aligned")),
                        "severity": float(mode["severity"]),
                        "sampling_kind": str(mode["kind"]),
                        "geometry_source": str(mode.get("geometry_source", "preprocessed_surface")),
                        "input_points": int(input_budget),
                        "surface_query_points": int(query["surf_gt"].shape[0]),
                        "volume_query_points": int(query["vol_gt"].shape[0]),
                        "checkpoint": str(args.checkpoints[model_name]),
                        **metrics,
                    }
                )
    return rows


def prepare_run_inputs(
    run_id: int,
    dataset: AhmedMLDatasetV2,
    args: argparse.Namespace,
    min_pos: torch.Tensor,
    max_pos: torch.Tensor,
    geometry_vtp_dirs: Mapping[str, Path],
) -> tuple[Dict[str, np.ndarray], Dict[str, object], Dict[str, torch.Tensor], Dict[str, np.ndarray], np.ndarray | None, Dict[str, np.ndarray]]:
    """Load one case once so every model receives identical CPU-side inputs."""
    case = load_case(Path(args.data_root), int(run_id))
    case["run_id"] = int(run_id)
    query = make_query_data(case, args, min_pos, max_pos)
    geometry_source_points: Dict[str, np.ndarray] = {}
    geometry_source_norm: Dict[str, torch.Tensor] = {}
    for source_name in args.active_geometry_sources:
        source_path = geometry_source_vtp_path(source_name, int(run_id), geometry_vtp_dirs)
        source_points = read_vtp_points(source_path)
        validate_geometry_source_bbox(source_points, case["surf_coords"], source_name, int(run_id))
        geometry_source_points[source_name] = source_points
        geometry_source_norm[source_name] = normalize_pos(torch.from_numpy(source_points), min_pos, max_pos)
    needs_density = "beta" in args.active_shifts
    log_density = None
    if needs_density:
        log_density = dataset._load_or_compute_full_geometry_density(
            int(run_id), expected_n=int(case["surf_coords"].shape[0])
        ).to(dtype=torch.float32).cpu().numpy()
    sine_weights = {
        "sine_y": sinusoidal_axis_probabilities(case["surf_coords"], axis=1),
        "sine_x": sinusoidal_axis_probabilities(case["surf_coords"], axis=0),
    }
    return case, query, geometry_source_norm, geometry_source_points, log_density, sine_weights


def prepare_run_batch(
    run_ids: Sequence[int],
    dataset: AhmedMLDatasetV2,
    args: argparse.Namespace,
    min_pos: torch.Tensor,
    max_pos: torch.Tensor,
    geometry_vtp_dirs: Mapping[str, Path],
) -> List[tuple[Dict[str, np.ndarray], Dict[str, object], Dict[str, torch.Tensor], Dict[str, np.ndarray], np.ndarray | None, Dict[str, np.ndarray]]]:
    return [
        prepare_run_inputs(int(run_id), dataset, args, min_pos, max_pos, geometry_vtp_dirs)
        for run_id in run_ids
    ]


def evaluate_run_group(
    run_ids: Sequence[int],
    model_names: Sequence[str],
    models: Mapping[str, torch.nn.Module],
    device: torch.device,
    dataset: AhmedMLDatasetV2,
    args: argparse.Namespace,
    modes: Sequence[Mapping[str, object]],
    budgets: Mapping[str, int],
    mean_s: torch.Tensor,
    std_s: torch.Tensor,
    mean_v: torch.Tensor,
    std_v: torch.Tensor,
    min_pos: torch.Tensor,
    max_pos: torch.Tensor,
    geometry_vtp_dirs: Mapping[str, Path],
    case_batch_size: int = 1,
    progress_label: str = "",
) -> List[Dict[str, object]]:
    """Own one GPU and overlap CPU input preparation with model inference."""
    rows: List[Dict[str, object]] = []
    if device.type == "cuda":
        torch.cuda.set_device(device)
    if progress_label:
        print(f"[{progress_label}] starting {len(run_ids)} cases", flush=True)
    if not run_ids:
        return rows
    if int(case_batch_size) < 1:
        raise ValueError("--screen-case-batch-size must be at least one.")
    run_batches = [
        list(run_ids[start : start + int(case_batch_size)])
        for start in range(0, len(run_ids), int(case_batch_size))
    ]
    # The next microbatch is read and normalized on CPU while this GPU
    # evaluates the current one, avoiding recurring host-side gaps.
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="comparison-prefetch") as prefetch:
        pending = prefetch.submit(
            prepare_run_batch,
            run_batches[0], dataset, args, min_pos, max_pos, geometry_vtp_dirs,
        )
        completed_cases = 0
        for batch_index, _run_batch in enumerate(run_batches):
            prepared_cases = pending.result()
            if batch_index + 1 < len(run_batches):
                pending = prefetch.submit(
                    prepare_run_batch,
                    run_batches[batch_index + 1], dataset, args, min_pos, max_pos, geometry_vtp_dirs,
                )
            for model_name in model_names:
                rows.extend(
                    evaluate_model_case_batch(
                        model_name, models[model_name], device, prepared_cases, modes,
                        budgets[model_name], args, mean_s, std_s, mean_v, std_v,
                    )
                )
            completed_cases += len(prepared_cases)
            if progress_label and (
                completed_cases == len(run_ids)
                or completed_cases == len(prepared_cases)
                or completed_cases % max(5, int(case_batch_size)) == 0
            ):
                print(f"[{progress_label}] {completed_cases}/{len(run_ids)} cases complete", flush=True)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return rows


def aggregate_rows(rows: Sequence[Mapping[str, object]], metric_keys: Sequence[str]) -> List[Dict[str, object]]:
    groups: Dict[tuple, List[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["model_name"]), str(row["shift_name"]), float(row["severity"]))].append(row)
    result: List[Dict[str, object]] = []
    def condition_key(shift_name: str) -> tuple[int, int, str]:
        if shift_name in SHIFT_ORDER:
            return (0, SHIFT_ORDER.index(shift_name), shift_name)
        return (1, 0, shift_name)

    for (model_name, shift_name, severity), group in sorted(
        groups.items(),
        key=lambda item: (MODEL_ORDER.index(item[0][0]), condition_key(item[0][1]), item[0][2]),
    ):
        item: Dict[str, object] = {
            "model_name": model_name,
            "model_label": MODEL_LABELS[model_name],
            "shift_name": shift_name,
            "severity": severity,
            "n": len(group),
        }
        for metric in metric_keys:
            values = np.asarray([float(row[metric]) for row in group], dtype=np.float64)
            item[f"{metric}_mean"] = float(np.mean(values))
            item[f"{metric}_std"] = float(np.std(values))
        result.append(item)
    return result


def paired_percentage_rows(rows: Sequence[Mapping[str, object]], metric_keys: Sequence[str]) -> List[Dict[str, object]]:
    lookup = {
        (str(row["model_name"]), int(row["run_id"]), int(row["view_id"]), str(row["shift_name"]), float(row["severity"])): row
        for row in rows
    }
    result: List[Dict[str, object]] = []
    for row in rows:
        severity = float(row["severity"])
        if severity <= 0.0:
            continue
        baseline = lookup.get(
            (REFERENCE_MODEL, int(row["run_id"]), int(row["view_id"]), str(row["shift_name"]), severity)
        )
        if baseline is None:
            continue
        output = {
            "run_id": int(row["run_id"]),
            "view_id": int(row["view_id"]),
            "model_name": str(row["model_name"]),
            "model_label": MODEL_LABELS[str(row["model_name"])],
            "percentage_reference_model": REFERENCE_MODEL,
            "shift_name": str(row["shift_name"]),
            "severity": severity,
        }
        for metric in metric_keys:
            denominator = max(abs(float(baseline[metric])), 1.0e-12)
            output[f"{metric}_pct_worsening"] = 100.0 * (float(row[metric]) - float(baseline[metric])) / denominator
        result.append(output)
    return result


def aggregate_percentage_rows(
    rows: Sequence[Mapping[str, object]],
    metric_keys: Sequence[str],
    absolute_rows: Sequence[Mapping[str, object]] | None = None,
) -> List[Dict[str, object]]:
    groups: Dict[tuple, List[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["model_name"]), str(row["shift_name"]), float(row["severity"]))].append(row)
    output: List[Dict[str, object]] = []
    absolute_lookup = {
        (str(row["model_name"]), str(row["shift_name"]), float(row["severity"])): row
        for row in (absolute_rows or [])
    }
    def condition_key(shift_name: str) -> tuple[int, int, str]:
        if shift_name in SHIFT_ORDER:
            return (0, SHIFT_ORDER.index(shift_name), shift_name)
        return (1, 0, shift_name)

    for (model_name, shift_name, severity), group in sorted(
        groups.items(),
        key=lambda item: (MODEL_ORDER.index(item[0][0]), condition_key(item[0][1]), item[0][2]),
    ):
        item = {
            "model_name": model_name,
            "model_label": MODEL_LABELS[model_name],
            "shift_name": shift_name,
            "severity": severity,
            "n": len(group),
        }
        for metric in metric_keys:
            values = np.asarray([float(row[f"{metric}_pct_worsening"]) for row in group], dtype=np.float64)
            current = absolute_lookup.get((model_name, shift_name, severity))
            reference = absolute_lookup.get((REFERENCE_MODEL, shift_name, severity))
            if current is not None and reference is not None:
                current_mean = float(current[f"{metric}_mean"])
                reference_mean = float(reference[f"{metric}_mean"])
                item[f"{metric}_pct_worsening_mean"] = 100.0 * (
                    current_mean - reference_mean
                ) / max(abs(reference_mean), 1.0e-12)
            else:
                item[f"{metric}_pct_worsening_mean"] = float(np.mean(values))
            item[f"{metric}_pct_worsening_std"] = float(np.std(values))
        output.append(item)
    return output


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        return
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_wide_metric_tables(
    output_dir: Path,
    aggregate: Sequence[Mapping[str, object]],
    percentage_aggregate: Sequence[Mapping[str, object]],
    metric: str,
    active_shifts: Sequence[str],
    levels: Sequence[float] | Mapping[str, Sequence[float]],
    active_geometry_sources: Sequence[str],
    include_std: bool = True,
) -> Dict[str, str]:
    """Write paper-facing absolute and paired-worsening tables."""
    conditions: List[tuple[str, str, float]] = []
    for shift in active_shifts:
        for severity in levels_for_shift(levels, shift):
            if float(severity) > 0.0:
                conditions.append((shift, f"{shift}_{severity:.2f}", float(severity)))
    for source in active_geometry_sources:
        conditions.append((f"geometry_{source}", f"geometry_{source}", 1.0))

    absolute_lookup = {
        (str(row["model_name"]), str(row["shift_name"]), float(row["severity"])): row
        for row in aggregate
    }
    percentage_lookup = {
        (str(row["model_name"]), str(row["shift_name"]), float(row["severity"])): row
        for row in percentage_aggregate
    }
    slug = metric.removesuffix("_rel_l2")
    absolute_rows: List[Dict[str, object]] = []
    percentage_rows: List[Dict[str, object]] = []
    for model_name in MODEL_ORDER:
        absolute_row: Dict[str, object] = {"model_name": model_name, "model_label": MODEL_LABELS[model_name]}
        percentage_row: Dict[str, object] = {"model_name": model_name, "model_label": MODEL_LABELS[model_name]}
        for condition, column, severity in conditions:
            absolute = absolute_lookup.get((model_name, condition, severity))
            percentage = percentage_lookup.get((model_name, condition, severity))
            absolute_row[f"{column}_mean"] = float(absolute[f"{metric}_mean"]) if absolute else math.nan
            if include_std:
                absolute_row[f"{column}_std"] = float(absolute[f"{metric}_std"]) if absolute else math.nan
            percentage_row[f"{column}_mean_pct"] = float(percentage[f"{metric}_pct_worsening_mean"]) if percentage else math.nan
            if include_std:
                percentage_row[f"{column}_std_pct"] = float(percentage[f"{metric}_pct_worsening_std"]) if percentage else math.nan
        absolute_rows.append(absolute_row)
        percentage_rows.append(percentage_row)

    absolute_csv = output_dir / f"{ABLATION_PREFIX}_{slug}_absolute_table.csv"
    percentage_csv = output_dir / f"{ABLATION_PREFIX}_{slug}_percentage_worsening_table.csv"
    write_csv(absolute_csv, absolute_rows)
    write_csv(percentage_csv, percentage_rows)

    def write_markdown(path: Path, rows: Sequence[Mapping[str, object]], percentage: bool) -> None:
        headers = ["Model"] + [column for _, column, _ in conditions]
        lines = [
                    f"# {ABLATION_TABLE_TITLE}: `{metric}`",
            "",
            (
                f"Values are means across paired runs and views. Percentage values are relative to {REFERENCE_MODEL_LABEL} under the same shift and severity."
                if not include_std
                else f"Values are mean +/- standard deviation across paired runs and views. Percentage values are relative to {REFERENCE_MODEL_LABEL} under the same shift and severity."
            ),
            "",
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join("---" for _ in headers) + " |",
        ]
        for row in rows:
            values = [str(row["model_label"])]
            for _, column, _ in conditions:
                if percentage:
                    mean = float(row[f"{column}_mean_pct"])
                    if include_std:
                        std = float(row[f"{column}_std_pct"])
                        values.append("n/a" if not np.isfinite(mean) else f"{mean:+.2f}% +/- {std:.2f}%")
                    else:
                        values.append("n/a" if not np.isfinite(mean) else f"{mean:+.2f}%")
                else:
                    mean = float(row[f"{column}_mean"])
                    if include_std:
                        std = float(row[f"{column}_std"])
                        values.append("n/a" if not np.isfinite(mean) else f"{mean:.5f} +/- {std:.5f}")
                    else:
                        values.append("n/a" if not np.isfinite(mean) else f"{mean:.5f}")
            lines.append("| " + " | ".join(values) + " |")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    absolute_md = output_dir / f"{ABLATION_PREFIX}_{slug}_absolute_table.md"
    percentage_md = output_dir / f"{ABLATION_PREFIX}_{slug}_percentage_worsening_table.md"
    write_markdown(absolute_md, absolute_rows, percentage=False)
    write_markdown(percentage_md, percentage_rows, percentage=True)
    return {
        "absolute_csv": str(absolute_csv),
        "absolute_markdown": str(absolute_md),
        "percentage_csv": str(percentage_csv),
        "percentage_markdown": str(percentage_md),
    }


def plot_endpoint_bars(
    aggregate: Sequence[Mapping[str, object]],
    metric: str,
    shift: str,
    endpoint: float,
    output: Path,
    percentage: bool,
    baseline_aggregate: Sequence[Mapping[str, object]] | None = None,
    percentage_reference_aggregate: Sequence[Mapping[str, object]] | None = None,
    log_scale: bool = False,
    show_std: bool = True,
    y_pad_fraction: float = 0.10,
) -> None:
    """Write one paper-facing endpoint bar plot.

    Absolute bars annotate every model against SMART under the same condition
    and endpoint. Percentage bars are already paired against that reference.
    This keeps the visual comparison tied to the range-0 SMART baseline rather
    than silently changing the denominator for each ablation checkpoint.
    """
    baseline_aggregate = aggregate if baseline_aggregate is None else baseline_aggregate
    percentage_reference_aggregate = (
        aggregate if percentage_reference_aggregate is None else percentage_reference_aggregate
    )
    y_pad_fraction = float(y_pad_fraction)
    if not math.isfinite(y_pad_fraction) or y_pad_fraction < 0.0:
        raise ValueError("y_pad_fraction must be finite and non-negative.")
    means: List[float] = []
    stds: List[float] = []
    baseline_means: List[float] = []
    smart_endpoint_matches = [
        row
        for row in baseline_aggregate
        if row["model_name"] == REFERENCE_MODEL
        and row["shift_name"] == shift
        and abs(float(row["severity"]) - endpoint) < 1.0e-8
    ]
    smart_endpoint = (
        float(smart_endpoint_matches[0][f"{metric}_mean"])
        if smart_endpoint_matches
        else math.nan
    )
    for model_name in MODEL_ORDER:
        matches = [
            row
            for row in aggregate
            if row["model_name"] == model_name
            and row["shift_name"] == shift
            and abs(float(row["severity"]) - endpoint) < 1.0e-8
        ]
        if not matches:
            means.append(math.nan)
            stds.append(math.nan)
            baseline_means.append(math.nan)
            continue
        row = matches[0]
        if percentage:
            means.append(float(row[f"{metric}_pct_worsening_mean"]))
            stds.append(float(row[f"{metric}_pct_worsening_std"]))
            baseline_means.append(0.0)
        else:
            means.append(float(row[f"{metric}_mean"]))
            stds.append(float(row[f"{metric}_std"]))
            baseline_means.append(smart_endpoint)

    means_array = np.asarray(means, dtype=np.float64)
    stds_array = np.asarray(stds, dtype=np.float64)
    if not percentage and log_scale:
        means_array = np.maximum(means_array, 1.0e-12)
        stds_array = np.minimum(stds_array, np.maximum(means_array - 1.0e-12, 0.0))
    error_values = stds_array if bool(show_std and _COMPUTE_PLOT_STD) else None
    x = np.arange(len(MODEL_ORDER), dtype=np.float64)
    font_size = ablation_font_size()
    fig, ax = plt.subplots(figsize=(13.4, 6.3))
    fig.subplots_adjust(left=0.12, right=0.60, bottom=0.34, top=0.84)
    bars = ax.bar(
        x,
        means_array,
        yerr=error_values,
        capsize=3 if error_values is not None else 0,
        width=0.525 * _BAR_WIDTH_SCALE,
        color=[MODEL_COLORS[name] for name in MODEL_ORDER],
        edgecolor="#222222",
        linewidth=0.65,
        alpha=0.96,
        error_kw={"elinewidth": 1.0, "capthick": 1.0},
    )
    if percentage:
        ax.axhline(0.0, color="#303030", linestyle="--", linewidth=1.1)
        ylabel = f"Relative difference from {REFERENCE_MODEL_LABEL} (%)"
    else:
        ylabel = f"Relative L2 error ({'log' if log_scale else 'linear'} scale)"
    ax.set_ylabel(ylabel, fontsize=font_size)
    condition_label = SHIFT_LABELS.get(shift, GEOMETRY_SOURCE_LABELS.get(shift.removeprefix("geometry_"), shift))
    ax.set_title(
        f"{condition_label} at severity {endpoint:.2f}: {metric.replace('_', ' ')}",
        fontsize=font_size,
        pad=12,
    )
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS[name] for name in MODEL_ORDER], rotation=24, ha="right", fontsize=font_size)
    ax.yaxis.set_label_coords(-0.095, 0.50)
    ax.tick_params(axis="both", labelsize=font_size)
    if not percentage and log_scale:
        ax.set_yscale("log")
    if not percentage:
        finite = means_array[np.isfinite(means_array)]
        if finite.size:
            low = float(np.min(finite))
            high = float(np.max(finite))
            if log_scale:
                ax.set_ylim(max(low * (1.0 - y_pad_fraction), 1.0e-12), high * (1.0 + y_pad_fraction))
            else:
                span = max(high - low, high, 1.0e-12)
                ax.set_ylim(max(0.0, low - y_pad_fraction * span), high + y_pad_fraction * span)
    if percentage:
        finite = means_array[np.isfinite(means_array)]
        if finite.size:
            low = min(0.0, float(np.min(finite)))
            high = max(0.0, float(np.max(finite)))
            span = max(high - low, 1.0)
            pad = y_pad_fraction * span
            ax.set_ylim(low - pad, high + pad)
    ax.grid(axis="y", alpha=0.20, linewidth=0.8)
    ax.set_axisbelow(True)
    fig.legend(
        handles=[Patch(facecolor=MODEL_COLORS[name], edgecolor="#222222", label=MODEL_LABELS[name]) for name in MODEL_ORDER],
        loc="upper left",
        bbox_to_anchor=(0.625, 0.82),
        bbox_transform=fig.transFigure,
        framealpha=0.92,
        fontsize=font_size,
    )

    finite_for_labels = means_array[np.isfinite(means_array)]
    label_span = (
        max(float(np.max(finite_for_labels)) - float(np.min(finite_for_labels)), 1.0)
        if finite_for_labels.size
        else 1.0
    )
    percentage_label_offset = 0.005 * label_span
    for index, bar in enumerate(bars):
        value = float(means_array[index])
        if not math.isfinite(value):
            continue
        if percentage:
            if MODEL_ORDER[index] == REFERENCE_MODEL:
                continue
            label = f"{value:+.1f}%"
            y = value + (percentage_label_offset if value >= 0.0 else -percentage_label_offset)
        else:
            if MODEL_ORDER[index] == REFERENCE_MODEL:
                continue
            percentage_matches = [
                reference_row
                for reference_row in percentage_reference_aggregate
                if reference_row["model_name"] == MODEL_ORDER[index]
                and reference_row["shift_name"] == shift
                and abs(float(reference_row["severity"]) - endpoint) < 1.0e-8
            ]
            if percentage_matches and f"{metric}_pct_worsening_mean" in percentage_matches[0]:
                relative = float(percentage_matches[0][f"{metric}_pct_worsening_mean"])
            else:
                baseline = float(baseline_means[index])
                if not math.isfinite(baseline) or abs(baseline) < 1.0e-12:
                    continue
                relative = 100.0 * (value - baseline) / abs(baseline)
            label = f"{relative:+.1f}%"
            y = value * 1.005 if log_scale else value + max(abs(value) * 0.005, 1.0e-6)
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            y,
            label,
            ha="center",
            va="bottom" if value >= 0.0 else "top",
            fontsize=font_size,
            rotation=90 if len(label) > 7 else 0,
            clip_on=False,
        )
    save_plot(fig, output)
    plt.close(fig)


def plot_combined_geometry_source_bars(
    aggregate: Sequence[Mapping[str, object]],
    percentage_aggregate: Sequence[Mapping[str, object]],
    source_modes: Sequence[str],
    output: Path,
    title: str,
    log_scale: bool,
    percentage: bool = False,
    show_std: bool = True,
    y_pad_fraction: float = 0.10,
) -> None:
    """Render global error for paired VTP sources in one grouped figure.

    Model color identifies the ablation range, source edge color identifies
    angle/isotropic/voxel remeshing, and opacity identifies div5/div10. This
    is the same visual grammar used by the broader SATLOSS and strategy
    comparison plots, but with the five range-ablation checkpoints as groups.
    """
    source_modes = list(source_modes)
    if not source_modes:
        return
    show_std = bool(show_std and _COMPUTE_PLOT_STD and not percentage)
    value_key = "combined_global_rel_l2_pct_worsening_mean" if percentage else "combined_global_rel_l2_mean"
    std_key = "combined_global_rel_l2_pct_worsening_std" if percentage else "combined_global_rel_l2_std"
    row_source = percentage_aggregate if percentage else aggregate
    row_map = {
        (str(row["model_name"]), str(row["shift_name"]), float(row["severity"])): row
        for row in row_source
    }
    present_models = [
        model_name
        for model_name in MODEL_ORDER
        if all((model_name, source_name, 1.0) in row_map for source_name in source_modes)
    ]
    if not present_models:
        return

    geometry_font_size = ablation_font_size()
    x = np.arange(len(present_models), dtype=np.float64)
    total_slots = len(source_modes)
    slot_pitch = 0.88 / float(max(total_slots, 1))
    width = 0.82 * _BAR_WIDTH_SCALE / float(max(total_slots, 1))
    factor_alphas = {5: 0.50, 10: 1.0, 20: 0.65, 40: 0.85}
    method_edgecolors = {
        "angle": "#222222",
        "isotropic": "#1B7837",
        "voxel": "#B15928",
    }
    values_for_limits: List[float] = []
    # Keep the same compact paper layout as the established comparison plots.
    # The reduced right edge leaves both short explanatory legends inside the
    # exported canvas instead of allowing them to be clipped.
    fig, ax = plt.subplots(figsize=(13.4, 7.4))
    fig.subplots_adjust(left=0.12, right=0.60, bottom=0.34, top=0.84)

    finite_source_values: List[float] = []
    for source_name in source_modes:
        for model_name in present_models:
            source_row = row_map[(model_name, source_name, 1.0)]
            source_value = float(source_row[value_key])
            if not percentage:
                source_value = max(source_value, 1.0e-12)
            if math.isfinite(source_value):
                finite_source_values.append(source_value)
    source_label_span = (
        max(max(finite_source_values) - min(finite_source_values), 1.0)
        if finite_source_values
        else 1.0
    )
    percentage_label_offset = 0.005 * source_label_span

    for source_idx, source_name in enumerate(source_modes):
        source_key = str(source_name).removeprefix("geometry_")
        method = source_key.split("_div", 1)[0]
        factor = int(source_key.rsplit("div", 1)[1])
        alpha = factor_alphas.get(factor, 1.0)
        edgecolor = method_edgecolors.get(method, "#222222")
        for model_idx, model_name in enumerate(present_models):
            row = row_map[(model_name, source_name, 1.0)]
            value = float(row[value_key])
            if not percentage:
                value = max(value, 1.0e-12)
            if not math.isfinite(value):
                continue
            error = float(row.get(std_key, 0.0)) if show_std else 0.0
            if log_scale and show_std:
                error = min(max(error, 0.0), value * 0.8)
            values_for_limits.append(value)
            slot_offset = (source_idx - 0.5 * (total_slots - 1)) * slot_pitch
            bar = ax.bar(
                x[model_idx] + slot_offset,
                value,
                width=width,
                yerr=error if show_std else None,
                capsize=3 if show_std else 0,
                color=MODEL_COLORS[model_name],
                edgecolor=edgecolor,
                linewidth=0.65,
                alpha=alpha,
                error_kw={"elinewidth": 1.0, "capthick": 1.0},
            )[0]
            if model_name != REFERENCE_MODEL:
                if percentage:
                    relative = value
                else:
                    relative_row = next(
                        (
                            candidate
                            for candidate in percentage_aggregate
                            if candidate["model_name"] == model_name
                            and candidate["shift_name"] == source_name
                            and abs(float(candidate["severity"]) - 1.0) < 1.0e-8
                        ),
                        None,
                    )
                    if relative_row is None:
                        continue
                    relative = float(relative_row["combined_global_rel_l2_pct_worsening_mean"])
                if percentage:
                    label_y = value + (percentage_label_offset if value >= 0.0 else -percentage_label_offset)
                elif log_scale:
                    label_y = value * 1.005
                else:
                    label_y = value + max(abs(value) * 0.005, 1.0e-6)
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    label_y,
                    f"{relative:+.1f}%",
                    ha="center",
                    va="bottom" if (not percentage or value >= 0.0) else "top",
                    rotation=90,
                    fontsize=geometry_font_size,
                    clip_on=False,
                )

    if percentage:
        ax.axhline(0.0, color="#303030", linestyle="--", linewidth=1.1)
        ylabel = f"Relative difference from {REFERENCE_MODEL_LABEL} (%)"
    else:
        ylabel = f"Combined-global relative L2 ({'log' if log_scale else 'linear'} scale)"
        if log_scale:
            ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [MODEL_LABELS[model_name] for model_name in present_models],
        rotation=24,
        ha="right",
        fontsize=geometry_font_size,
    )
    ax.set_ylabel(ylabel, fontsize=geometry_font_size)
    ax.yaxis.set_label_coords(-0.095, 0.50)
    ax.tick_params(axis="y", labelsize=geometry_font_size)
    ax.set_title(title, fontsize=geometry_font_size, pad=12)
    ax.grid(axis="y", which="both", alpha=0.2)
    ax.set_axisbelow(True)

    if values_for_limits:
        finite_values = np.asarray(values_for_limits, dtype=np.float64)
        if percentage:
            low = min(0.0, float(np.min(finite_values)))
            high = max(0.0, float(np.max(finite_values)))
            span = max(high - low, 1.0)
            pad = float(y_pad_fraction) * span
            ax.set_ylim(low - pad, high + pad)
        elif log_scale:
            ax.set_ylim(max(float(np.min(finite_values)) * (1.0 - y_pad_fraction), 1.0e-12), float(np.max(finite_values)) * (1.0 + y_pad_fraction))
        else:
            low = float(np.min(finite_values))
            high = float(np.max(finite_values))
            span = max(high - low, high, 1.0e-12)
            ax.set_ylim(max(0.0, low - y_pad_fraction * span), high + y_pad_fraction * span)

    model_legend = fig.legend(
        handles=[
            Patch(facecolor=MODEL_COLORS[name], edgecolor="#222222", label=MODEL_LABELS[name])
            for name in present_models
        ],
        loc="upper left",
        bbox_to_anchor=(0.625, 0.82),
        bbox_transform=fig.transFigure,
        fontsize=geometry_font_size,
        framealpha=0.92,
    )
    factor_legend = fig.legend(
        handles=[
            Patch(facecolor="#555555", edgecolor="#222222", alpha=0.50, label="div5 (50% opacity)"),
            Patch(facecolor="#555555", edgecolor="#222222", alpha=1.0, label="div10 (100% opacity)"),
        ],
        loc="upper left",
        bbox_to_anchor=(0.625, 0.50),
        bbox_transform=fig.transFigure,
        fontsize=geometry_font_size,
        framealpha=0.92,
    )
    save_plot(fig, output)
    plt.close(fig)


def average_remeshing_rows(
    rows: Sequence[Mapping[str, object]],
    source_modes: Sequence[str],
    percentage: bool,
) -> List[Dict[str, object]]:
    """Average angle/isotropic/voxel rows separately for each div factor."""
    row_map = {
        (str(row["model_name"]), str(row["shift_name"]), float(row["severity"])): row
        for row in rows
    }
    methods = ("angle", "isotropic", "voxel")
    factors = sorted(
        {
            int(str(source).removeprefix("geometry_").rsplit("div", 1)[1])
            for source in source_modes
            if "_div" in str(source)
        }
    )
    value_key = "combined_global_rel_l2_pct_worsening_mean" if percentage else "combined_global_rel_l2_mean"
    std_key = "combined_global_rel_l2_pct_worsening_std" if percentage else "combined_global_rel_l2_std"
    averaged: List[Dict[str, object]] = []
    for model_name in MODEL_ORDER:
        for factor in factors:
            source_names = [f"geometry_{method}_div{factor}" for method in methods]
            source_rows = [row_map.get((model_name, source_name, 1.0)) for source_name in source_names]
            if any(row is None for row in source_rows):
                continue
            typed_rows = [row for row in source_rows if row is not None]
            result: Dict[str, object] = dict(typed_rows[0])
            result["shift_name"] = f"geometry_average_div{factor}"
            result["model_label"] = MODEL_LABELS[model_name]
            result[value_key] = float(np.mean([float(row[value_key]) for row in typed_rows]))
            if std_key in typed_rows[0]:
                # This is only a visual aggregate; RMS preserves the scale of
                # the source uncertainty without treating source methods as
                # independent samples.
                result[std_key] = float(
                    np.sqrt(np.mean([float(row.get(std_key, 0.0)) ** 2 for row in typed_rows]))
                )
            averaged.append(result)
    return averaged


def plot_compact_endpoint_summary(
    aggregate: Sequence[Mapping[str, object]],
    percentage_aggregate: Sequence[Mapping[str, object]],
    levels: Mapping[str, Sequence[float]],
    geometry_source_modes: Sequence[str],
    output: Path,
    y_pad_fraction: float,
    show_std: bool = False,
) -> None:
    """Render the paper-style four-condition endpoint summary requested for ablations."""
    if "sine_x" not in levels or "sine_y" not in levels:
        return
    average_absolute = average_remeshing_rows(aggregate, geometry_source_modes, percentage=False)
    average_percentage = average_remeshing_rows(percentage_aggregate, geometry_source_modes, percentage=True)
    values_by_key = {
        (str(row["model_name"]), str(row["shift_name"]), round(float(row["severity"]), 8)): row
        for row in [*aggregate, *average_absolute]
    }
    percentages_by_key = {
        (str(row["model_name"]), str(row["shift_name"]), round(float(row["severity"]), 8)): row
        for row in [*percentage_aggregate, *average_percentage]
    }
    conditions = (
        ("sine_x", round(float(max(levels["sine_x"])), 8), "Sine x"),
        ("sine_y", round(float(max(levels["sine_y"])), 8), "Sine y"),
        ("geometry_average_div5", 1.0, "Mean remeshing div5"),
        ("geometry_average_div10", 1.0, "Mean remeshing div10"),
    )
    if any(
        (model_name, condition, severity) not in values_by_key
        for model_name in MODEL_ORDER
        for condition, severity, _label in conditions
    ):
        return

    metric = "combined_global_rel_l2"
    fig, axis = plt.subplots(figsize=(14.2, 6.1))
    font_size = ablation_font_size(1.0)
    x_centers = np.arange(len(conditions), dtype=np.float64)
    bar_width = 0.175 * _BAR_WIDTH_SCALE
    intra_group_gap = 0.028
    offsets = (np.arange(len(MODEL_ORDER)) - 0.5 * (len(MODEL_ORDER) - 1)) * (bar_width + intra_group_gap)
    plotted_values: List[float] = []
    plotted_lower_bounds: List[float] = []
    plotted_upper_bounds: List[float] = []
    percentage_labels: List[tuple[object, float, str, str]] = []
    for condition_index, (condition, severity, _label) in enumerate(conditions):
        smart_row = values_by_key[(REFERENCE_MODEL, condition, severity)]
        smart_value = float(smart_row[f"{metric}_mean"])
        for model_index, model_name in enumerate(MODEL_ORDER):
            row = values_by_key[(model_name, condition, severity)]
            value = float(row[f"{metric}_mean"])
            plotted_values.append(value)
            std = float(row.get(f"{metric}_std", 0.0)) if show_std else 0.0
            if not math.isfinite(std) or std < 0.0:
                std = 0.0
            plotted_lower_bounds.append(max(0.0, value - std))
            plotted_upper_bounds.append(value + std)
            bar = axis.bar(
                x_centers[condition_index] + offsets[model_index],
                value,
                width=bar_width,
                color=MODEL_COLORS[model_name],
                edgecolor="#20262d",
                linewidth=0.65,
                zorder=3,
            )[0]
            if show_std and std > 0.0:
                axis.errorbar(
                    bar.get_x() + bar.get_width() / 2.0,
                    value,
                    yerr=std,
                    fmt="none",
                    ecolor="#20262d",
                    elinewidth=1.1,
                    capsize=3.2,
                    capthick=1.1,
                    zorder=5,
                )
            if model_name == REFERENCE_MODEL:
                continue
            percentage_row = percentages_by_key.get((model_name, condition, severity))
            relative = (
                float(percentage_row[f"{metric}_pct_worsening_mean"])
                if percentage_row is not None
                else 100.0 * (value - smart_value) / max(abs(smart_value), 1.0e-12)
            )
            percentage_labels.append((bar, value + std, f"{relative:+.1f}%", MODEL_COLORS[model_name]))
    low = min(plotted_lower_bounds)
    high = max(plotted_upper_bounds)
    span = max(high - low, high, 1.0e-12)
    label_offset = max(0.014 * span, 1.0e-6)
    axis.set_ylim(max(0.0, low - float(y_pad_fraction) * span), high + max(float(y_pad_fraction), 0.14) * span + label_offset)
    for bar, top, text, color in percentage_labels:
        axis.text(
            bar.get_x() + bar.get_width() / 2.0,
            top + label_offset,
            text,
            ha="center",
            va="bottom",
            rotation=90,
            fontsize=font_size,
            fontweight="bold",
            color=color,
            clip_on=False,
        )
    axis.set_xticks(x_centers)
    axis.set_xticklabels([label for _condition, _severity, label in conditions], fontsize=font_size)
    axis.set_ylabel("Combined-global relative L2", fontsize=font_size)
    axis.yaxis.set_label_coords(-0.075, 0.5)
    axis.tick_params(axis="y", labelsize=font_size)
    axis.set_title("Combined-global endpoint error", fontsize=font_size, pad=12)
    axis.grid(axis="y", alpha=0.22, linewidth=0.8, zorder=0)
    axis.set_axisbelow(True)
    axis.legend(
        handles=[Patch(facecolor=MODEL_COLORS[name], edgecolor="#20262d", label=MODEL_LABELS[name]) for name in MODEL_ORDER],
        loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=len(MODEL_ORDER),
        frameon=False, fontsize=font_size * 0.92,
    )
    fig.subplots_adjust(left=0.105, right=0.985, bottom=0.15, top=0.87)
    save_plot(fig, output)
    plt.close(fig)


def main() -> None:
    global _COMPUTE_PLOT_STD, MODEL_ORDER, MODEL_LABELS, MODEL_COLORS
    global _BAR_WIDTH_SCALE, GEOMETRY_SOURCE_LABELS, GEOMETRY_METHOD_LABELS, GEOMETRY_METHOD_FILE_SLUGS
    global REFERENCE_MODEL, REFERENCE_MODEL_LABEL, ABLATION_PREFIX, ABLATION_TABLE_TITLE
    args = parse_args()
    apply_experiment_preset(args)
    if args.experiment_preset == "deal_weighting_ablation_vtp":
        MODEL_ORDER = DEAL_MODEL_ORDER
        MODEL_LABELS = dict(DEAL_MODEL_LABELS)
        MODEL_COLORS = dict(DEAL_MODEL_COLORS)
        GEOMETRY_SOURCE_LABELS = dict(DEAL_GEOMETRY_SOURCE_LABELS)
        GEOMETRY_METHOD_LABELS = dict(DEAL_GEOMETRY_METHOD_LABELS)
        GEOMETRY_METHOD_FILE_SLUGS = dict(DEAL_GEOMETRY_METHOD_FILE_SLUGS)
        _BAR_WIDTH_SCALE = 0.70
        REFERENCE_MODEL = "SMART"
        REFERENCE_MODEL_LABEL = "SMART baseline"
        ABLATION_PREFIX = "deal_weighting_ablation"
        ABLATION_TABLE_TITLE = "DeAL loss-balancing ablation"
    elif args.experiment_preset == "kde_ablation_vtp":
        MODEL_ORDER = KDE_MODEL_ORDER
        MODEL_LABELS = dict(KDE_MODEL_LABELS)
        MODEL_COLORS = dict(KDE_MODEL_COLORS)
        REFERENCE_MODEL = "SMART"
        REFERENCE_MODEL_LABEL = "SMART baseline"
        ABLATION_PREFIX = "kde_ablation"
        ABLATION_TABLE_TITLE = "SATLOSS KDE-neighborhood ablation"
    else:
        _BAR_WIDTH_SCALE = 1.0
        MODEL_ORDER = (
            "SMART",
            "SMART_SATLOSS7_RANGE025",
            "SMART_SATLOSS7_RANGE050",
            "SMART_SATLOSS7_RANGE075",
            "SMART_SATLOSS7",
            "SMART_SATLOSS7_RANGE200",
            "SMART_SATLOSS7_RANGE300",
            "SMART_SATLOSS7_RANGE500",
        )
        REFERENCE_MODEL = "SMART"
        REFERENCE_MODEL_LABEL = "SMART baseline"
        ABLATION_PREFIX = "range_ablation"
        ABLATION_TABLE_TITLE = "SMART SATLOSS range ablation"
    if args.exclude_range500:
        MODEL_ORDER = tuple(name for name in MODEL_ORDER if name != "SMART_SATLOSS7_RANGE500")
    configure_plot_style(args.font_scale)
    _COMPUTE_PLOT_STD = not bool(args.no_std)
    plot_scales = parse_plot_scales(args.plot_scales)
    if not math.isfinite(float(args.y_pad_fraction)) or float(args.y_pad_fraction) < 0.0:
        raise ValueError("--y-pad-fraction must be finite and non-negative.")
    args.active_shifts = parse_shifts(args.active_shifts)
    common_levels = parse_levels(args.shift_levels)
    beta_levels = parse_levels(args.beta_levels) if args.beta_levels is not None else common_levels
    sine_levels = parse_levels(args.sine_levels) if args.sine_levels is not None else common_levels
    if any(float(level) > 1.0 for level in sine_levels):
        raise ValueError(
            "Sine-mixture levels must be within [0, 1]. Use --beta-levels for beta endpoints above 1; "
            "the sine sampler is a bounded mixture fraction."
        )
    levels: Dict[str, Sequence[float]] = {
        "beta": beta_levels,
        "sine_y": sine_levels,
        "sine_x": sine_levels,
    }
    geometry_factors = parse_geometry_decimation_factors(args.geometry_decimation_factors)
    args.active_geometry_sources = parse_active_geometry_sources(
        args.active_geometry_sources,
        geometry_factors,
    )
    geometry_vtp_dirs = {
        "angle": Path(args.angle_decimated_vtp_dir).expanduser().resolve(),
        "isotropic": Path(args.isotropic_decimated_vtp_dir).expanduser().resolve(),
        "voxel": Path(args.voxel_decimated_vtp_dir).expanduser().resolve(),
    }
    devices = resolve_devices(args.devices)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    args.checkpoints = checkpoint_map(args)
    configs = OrderedDict((name, load_cfg(cfg_name)) for name, cfg_name in config_map(args).items())
    for name, checkpoint in args.checkpoints.items():
        if not Path(checkpoint).is_file():
            raise FileNotFoundError(f"Checkpoint for {name} not found: {checkpoint}")
    budgets = {name: train_encoder_input_points(configs[name], name) for name in MODEL_ORDER}
    if len(set(budgets.values())) != 1:
        raise ValueError(f"The SMART range ablation requires one shared encoder budget, got {budgets}")
    input_budget = int(next(iter(budgets.values())))
    modes = build_modes(args.active_shifts, levels, args.active_geometry_sources)
    print(f"Devices: {', '.join(str(device) for device in devices)}")
    print(f"Active shifts: {', '.join(args.active_shifts)}")
    print(
        "Active VTP geometry sources: "
        + (", ".join(args.active_geometry_sources) if args.active_geometry_sources else "none")
    )
    print(f"Geometry factors: {', '.join(str(factor) for factor in geometry_factors)}")
    print(
        "Shift levels: "
        + "; ".join(
            f"{shift}=" + ",".join(f"{level:.2f}" for level in levels_for_shift(levels, shift))
            for shift in args.active_shifts
        )
    )
    print(f"Experiment preset: {args.experiment_preset}")
    print(f"Absolute plot scales: {', '.join(plot_scales)}; standard deviations: {_COMPUTE_PLOT_STD}")
    print(f"Plot font scale: {args.font_scale:.2f}")
    print(f"Plot y-axis padding: {args.y_pad_fraction:.2%}")
    print(f"Shared training-aligned encoder input budget: {input_budget}")
    for name in MODEL_ORDER:
        print(f"{name}: config={config_map(args)[name]} checkpoint={args.checkpoints[name]}")
    if args.dry_run:
        print(f"[dry-run] modes: {', '.join(str(mode['name']) for mode in modes)}")
        return

    reference_config = configs[REFERENCE_MODEL]
    dataset = AhmedMLDatasetV2(
        saved_folder=str(args.data_root),
        if_test=True,
        geometry_points=input_budget,
        surface_points=int(args.surface_query_points),
        volume_points=int(args.volume_query_points),
        scale_positions=bool(reference_config.scale_positions),
        require_preprocessed=True,
        geometry_density_estimator=str(args.density_estimator),
        geometry_density_knn_k=int(args.density_knn_k),
        geometry_density_neighbor_hops=1,
        geometry_density_cache_dtype="float16",
    )
    candidate_universe = set(dataset.test_ids if args.candidate_split == "test" else dataset.all_ids)
    geometry_candidate_ids: set[int] | None = None
    if args.active_geometry_sources:
        geometry_candidate_ids = {int(run_id) for run_id in candidate_universe}
        for source_name in args.active_geometry_sources:
            source_ids = {
                int(path.parent.name.split("_", 1)[1])
                for path in (
                    geometry_source_vtp_path(source_name, run_id, geometry_vtp_dirs)
                    for run_id in candidate_universe
                )
                if path.is_file()
            }
            geometry_candidate_ids.intersection_update(source_ids)
        if not geometry_candidate_ids:
            raise FileNotFoundError(
                "No test runs contain every requested VTP source. "
                f"Sources: {args.active_geometry_sources}; factors: {geometry_factors}"
            )
        print(f"VTP common completed subset ({args.candidate_split} universe): {len(geometry_candidate_ids)} runs")
    if args.run_selection in {"top_angle_div10_range_ablation", "top_pairwise_improvement", "top_deal_mean_improvement"}:
        if args.run_ids:
            raise ValueError("--run-ids cannot be combined with top-run selection.")
        if args.run_selection == "top_angle_div10_range_ablation" and "angle_div10" not in args.active_geometry_sources:
            raise ValueError(
                "Top range-ablation selection requires angle_div10 to be an active VTP source. "
                "Use --active-geometry-sources angle --geometry-decimation-factors 10."
            )
        available_top_candidates = sorted(int(run_id) for run_id in (geometry_candidate_ids or candidate_universe))
        candidate_pool_size_requested = int(args.top_selection_candidates)
        if candidate_pool_size_requested < 0:
            raise ValueError("--top-selection-candidates must be non-negative.")
        if candidate_pool_size_requested > len(available_top_candidates):
            raise ValueError(
                f"Requested {candidate_pool_size_requested} top-selection candidates, "
                f"but only {len(available_top_candidates)} common candidates are available."
            )
        if candidate_pool_size_requested > 0 and candidate_pool_size_requested < len(available_top_candidates):
            candidate_rng = np.random.default_rng(int(args.seed) + 7101)
            run_ids = sorted(
                int(value)
                for value in candidate_rng.choice(
                    np.asarray(available_top_candidates),
                    size=candidate_pool_size_requested,
                    replace=False,
                )
            )
        else:
            run_ids = available_top_candidates
        if int(args.num_runs) <= 0:
            raise ValueError("--num-runs must be positive when selecting top geometries.")
        print(f"Top-run candidate pool ({args.candidate_split} universe): {len(run_ids)} geometries")
    else:
        run_ids = select_run_ids(dataset, args, candidate_ids=geometry_candidate_ids or candidate_universe)
    print(f"Evaluating run IDs: {run_ids}")
    if "beta" not in args.active_shifts:
        print(
            "Density estimation skipped: beta/inverse-density sampling is inactive for this run.",
            flush=True,
        )
    mean_s = dataset.mean_surf_data
    std_s = torch.clamp(dataset.std_surf_data, min=1.0e-12)
    mean_v = dataset.mean_vol_data
    std_v = torch.clamp(dataset.std_vol_data, min=1.0e-12)
    min_pos = dataset.min_pos
    max_pos = dataset.max_pos
    top_selection_rows: List[Dict[str, object]] = []
    candidate_pool_size = len(run_ids)

    if args.run_selection == "top_pairwise_improvement":
        if not args.top_selection_improved_model or not args.top_selection_reference_model:
            raise ValueError(
                "--run-selection top_pairwise_improvement requires --top-selection-improved-model "
                "and --top-selection-reference-model."
            )
        screen_model_names = (
            str(args.top_selection_improved_model),
            str(args.top_selection_reference_model),
        )
        if any(name not in MODEL_ORDER for name in screen_model_names):
            raise ValueError("Pairwise selection model keys must be active models in this experiment.")
        selection_conditions = top_selection_conditions(
            args.top_selection_conditions,
            args.active_shifts,
            levels,
            args.active_geometry_sources,
        )
        screen_modes = selection_modes(modes, selection_conditions)

        # The screening phase has no need to run SMART.  Replicating just the
        # two ranked checkpoints across devices lets each GPU own a disjoint
        # case shard, instead of leaving one GPU idle for 250 expensive cases.
        active_screen_devices = devices[: min(len(devices), len(run_ids))]
        screen_models_by_device: Dict[torch.device, Dict[str, torch.nn.Module]] = {}
        for device in active_screen_devices:
            screen_models_by_device[device] = {
                name: build_ablation_model(
                    configs[name], name, args.checkpoints[name], device, args.batched_query_subregion_size
                )
                for name in screen_model_names
            }
        # Contiguous shards reduce filesystem seek churn relative to round-robin
        # dispatch while retaining an equal persistent workload per GPU.
        screen_shards = [
            [int(run_id) for run_id in shard]
            for shard in np.array_split(np.asarray(run_ids, dtype=np.int64), len(active_screen_devices))
            if len(shard)
        ]
        print(
            "Pairwise screening placement: "
            + ", ".join(
                f"{device}->{'/'.join(screen_model_names)} ({len(shard)} cases)"
                for device, shard in zip(active_screen_devices, screen_shards)
            )
            + f"; ranking modes={', '.join(str(mode['name']) for mode in screen_modes)}",
            flush=True,
        )
        screening_rows: List[Dict[str, object]] = []
        with ThreadPoolExecutor(max_workers=len(active_screen_devices)) as executor:
            futures = [
                executor.submit(
                    evaluate_run_group,
                    shard,
                    screen_model_names,
                    screen_models_by_device[device],
                    device,
                    dataset,
                    args,
                    screen_modes,
                    budgets,
                    mean_s,
                    std_s,
                    mean_v,
                    std_v,
                    min_pos,
                    max_pos,
                    geometry_vtp_dirs,
                    int(args.screen_case_batch_size),
                    f"screen {device}",
                )
                for device, shard in zip(active_screen_devices, screen_shards)
            ]
            for future in futures:
                screening_rows.extend(future.result())

        top_selection_rows = rank_top_pairwise_improvement_runs(
            screening_rows,
            run_ids,
            args.top_selection_metric,
            int(args.num_runs),
            screen_model_names[0],
            screen_model_names[1],
            selection_conditions,
            args.top_selection_min_condition_improvement_percent,
        )
        selected_ids = [int(row["run_id"]) for row in top_selection_rows]
        selection_path = output_dir / "top_pairwise_improvement_selection.csv"
        write_csv(selection_path, top_selection_rows)
        with (output_dir / "top_pairwise_improvement_selection.json").open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "selection_metric": args.top_selection_metric,
                    "candidate_pool_size": candidate_pool_size,
                    "candidate_split": args.candidate_split,
                    "selected_count": len(selected_ids),
                    "selected_run_ids": selected_ids,
                    "improved_model": screen_model_names[0],
                    "reference_model": screen_model_names[1],
                    "selection_conditions": [top_selection_condition_label(condition) for condition in selection_conditions],
                    "selection_rule": "highest mean relative improvement of improved_model over reference_model across the listed conditions",
                    "rows": top_selection_rows,
                },
                handle,
                indent=2,
            )
        print(f"Selected top {len(selected_ids)} pairwise-improvement geometries: {selected_ids}", flush=True)
        run_ids = selected_ids
        del screen_models_by_device, screening_rows
        if torch.cuda.is_available():
            for device in active_screen_devices:
                with torch.cuda.device(device):
                    torch.cuda.empty_cache()

    elif args.run_selection == "top_deal_mean_improvement":
        if args.experiment_preset != "deal_weighting_ablation_vtp":
            raise ValueError("--run-selection top_deal_mean_improvement requires --experiment-preset deal_weighting_ablation_vtp.")
        selection_conditions = top_selection_conditions(
            args.top_selection_conditions,
            args.active_shifts,
            levels,
            args.active_geometry_sources,
        )
        screen_modes = selection_modes(modes, selection_conditions)
        screen_model_names = list(MODEL_ORDER)
        active_screen_devices = devices[: min(len(devices), len(screen_model_names))]
        screen_assignments: Dict[torch.device, List[str]] = defaultdict(list)
        for index, model_name in enumerate(screen_model_names):
            screen_assignments[active_screen_devices[index % len(active_screen_devices)]].append(model_name)
        screen_models_by_device: Dict[torch.device, Dict[str, torch.nn.Module]] = {}
        for device, assigned_models in screen_assignments.items():
            screen_models_by_device[device] = {
                name: build_ablation_model(
                    configs[name], name, args.checkpoints[name], device, args.batched_query_subregion_size
                )
                for name in assigned_models
            }
        print(
            "DeAL screening placement: "
            + ", ".join(
                f"{device}->{'/'.join(assigned_models)} ({len(run_ids)} cases)"
                for device, assigned_models in screen_assignments.items()
            )
            + f"; ranking modes={', '.join(str(mode['name']) for mode in screen_modes)}",
            flush=True,
        )
        screening_rows: List[Dict[str, object]] = []
        with ThreadPoolExecutor(max_workers=len(screen_assignments)) as executor:
            futures = [
                executor.submit(
                    evaluate_run_group,
                    run_ids,
                    assigned_models,
                    screen_models_by_device[device],
                    device,
                    dataset,
                    args,
                    screen_modes,
                    budgets,
                    mean_s,
                    std_s,
                    mean_v,
                    std_v,
                    min_pos,
                    max_pos,
                    geometry_vtp_dirs,
                    int(args.screen_case_batch_size),
                    f"screen {device}",
                )
                for device, assigned_models in screen_assignments.items()
            ]
            for future in futures:
                screening_rows.extend(future.result())

        top_selection_rows = rank_top_deal_mean_improvement_runs(
            screening_rows,
            run_ids,
            args.top_selection_metric,
            int(args.num_runs),
            selection_conditions,
        )
        selected_ids = [int(row["run_id"]) for row in top_selection_rows]
        selection_path = output_dir / "top_deal_mean_improvement_selection.csv"
        write_csv(selection_path, top_selection_rows)
        with (output_dir / "top_deal_mean_improvement_selection.json").open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "selection_metric": args.top_selection_metric,
                    "candidate_pool_size": candidate_pool_size,
                    "candidate_split": args.candidate_split,
                    "selected_count": len(selected_ids),
                    "selected_run_ids": selected_ids,
                    "reference_model": REFERENCE_MODEL,
                    "deal_models": [name for name in MODEL_ORDER if name != REFERENCE_MODEL],
                    "selection_conditions": [top_selection_condition_label(condition) for condition in selection_conditions],
                    "selection_rule": "highest mean SMART-relative improvement across every DeAL model and listed condition",
                    "rows": top_selection_rows,
                },
                handle,
                indent=2,
            )
        print(f"Selected top {len(selected_ids)} mean-DeAL-improvement geometries: {selected_ids}", flush=True)
        run_ids = selected_ids
        del screen_models_by_device, screening_rows
        if torch.cuda.is_available():
            for device in active_screen_devices:
                with torch.cuda.device(device):
                    torch.cuda.empty_cache()

    model_devices = {name: devices[index % len(devices)] for index, name in enumerate(MODEL_ORDER)}
    models = {
        name: build_ablation_model(configs[name], name, args.checkpoints[name], model_devices[name], args.batched_query_subregion_size)
        for name in MODEL_ORDER
    }
    print("Model placement: " + ", ".join(f"{name}->{model_devices[name]}" for name in MODEL_ORDER))
    device_groups: Dict[torch.device, List[str]] = defaultdict(list)
    for name in MODEL_ORDER:
        device_groups[model_devices[name]].append(name)

    all_rows: List[Dict[str, object]] = []
    for run_id in tqdm(run_ids, desc="Runs", dynamic_ncols=True):
        (
            case,
            query,
            geometry_source_norm,
            geometry_source_points,
            log_density,
            sine_weights,
        ) = prepare_run_inputs(
            int(run_id), dataset, args, min_pos, max_pos, geometry_vtp_dirs
        )

        def evaluate_group(group_names: Sequence[str]) -> List[Dict[str, object]]:
            group_rows: List[Dict[str, object]] = []
            group_device = model_devices[group_names[0]]
            for model_name in group_names:
                group_rows.extend(
                    evaluate_model_run(
                        model_name,
                        models[model_name],
                        group_device,
                        case,
                        query,
                        modes,
                        log_density,
                        sine_weights,
                        budgets[model_name],
                        args,
                        mean_s,
                        std_s,
                        mean_v,
                        std_v,
                        geometry_source_norm,
                        geometry_source_points,
                    )
                )
            return group_rows

        with ThreadPoolExecutor(max_workers=len(device_groups)) as executor:
            futures = [executor.submit(evaluate_group, names) for names in device_groups.values()]
            for future in futures:
                all_rows.extend(future.result())

    raw_rows = sorted(
        all_rows,
        key=lambda row: (
            MODEL_ORDER.index(str(row["model_name"])),
            int(row["run_id"]),
            str(row["shift_name"]),
            float(row["severity"]),
            int(row["view_id"]),
        ),
    )
    if args.run_selection == "top_angle_div10_range_ablation":
        top_selection_rows = rank_top_range_ablation_runs(
            raw_rows,
            run_ids,
            args.top_selection_metric,
            int(args.num_runs),
        )
        selected_ids = [int(row["run_id"]) for row in top_selection_rows]
        write_csv(output_dir / "top_angle_div10_range_ablation_selection.csv", top_selection_rows)
        with (output_dir / "top_angle_div10_range_ablation_selection.json").open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "selection_metric": args.top_selection_metric,
                    "candidate_pool_size": candidate_pool_size,
                    "selected_count": len(selected_ids),
                    "selected_run_ids": selected_ids,
                    "selection_rule": "lowest mean SMART-relative combined error across SATLOSS ranges 0.25, 0.50, and 0.75 on angle_div10",
                    "rows": top_selection_rows,
                },
                handle,
                indent=2,
            )
        print(f"Selected top {len(selected_ids)} angle_div10 geometries: {selected_ids}")
        selected_set = set(selected_ids)
        raw_rows = [row for row in raw_rows if int(row["run_id"]) in selected_set]
        run_ids = selected_ids
    elif args.run_selection == "top_pairwise_improvement" and not top_selection_rows:
        if not args.top_selection_improved_model or not args.top_selection_reference_model:
            raise ValueError(
                "--run-selection top_pairwise_improvement requires --top-selection-improved-model "
                "and --top-selection-reference-model."
            )
        if args.top_selection_improved_model not in MODEL_ORDER or args.top_selection_reference_model not in MODEL_ORDER:
            raise ValueError("Pairwise selection model keys must be active models in this experiment.")
        selection_conditions = top_selection_conditions(
            args.top_selection_conditions,
            args.active_shifts,
            levels,
            args.active_geometry_sources,
        )
        top_selection_rows = rank_top_pairwise_improvement_runs(
            raw_rows,
            run_ids,
            args.top_selection_metric,
            int(args.num_runs),
            args.top_selection_improved_model,
            args.top_selection_reference_model,
            selection_conditions,
            args.top_selection_min_condition_improvement_percent,
        )
        selected_ids = [int(row["run_id"]) for row in top_selection_rows]
        selection_path = output_dir / "top_pairwise_improvement_selection.csv"
        write_csv(selection_path, top_selection_rows)
        with (output_dir / "top_pairwise_improvement_selection.json").open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "selection_metric": args.top_selection_metric,
                    "candidate_pool_size": candidate_pool_size,
                    "candidate_split": args.candidate_split,
                    "selected_count": len(selected_ids),
                    "selected_run_ids": selected_ids,
                    "improved_model": args.top_selection_improved_model,
                    "reference_model": args.top_selection_reference_model,
                    "selection_conditions": [top_selection_condition_label(condition) for condition in selection_conditions],
                    "selection_rule": "highest mean relative improvement of improved_model over reference_model across the listed conditions",
                    "rows": top_selection_rows,
                },
                handle,
                indent=2,
            )
        print(f"Selected top {len(selected_ids)} pairwise-improvement geometries: {selected_ids}")
        selected_set = set(selected_ids)
        raw_rows = [row for row in raw_rows if int(row["run_id"]) in selected_set]
        run_ids = selected_ids
    aggregate = aggregate_rows(raw_rows, METRIC_KEYS)
    percentage_rows = paired_percentage_rows(raw_rows, METRIC_KEYS)
    percentage_aggregate = aggregate_percentage_rows(percentage_rows, METRIC_KEYS, absolute_rows=aggregate)
    write_csv(output_dir / "per_view_metrics.csv", raw_rows)
    write_csv(output_dir / "aggregate_metrics.csv", aggregate)
    write_csv(output_dir / "paired_percentage_worsening.csv", percentage_rows)
    write_csv(output_dir / "aggregate_percentage_worsening.csv", percentage_aggregate)

    # Geometry-source conditions are rendered only in the paired method plots
    # below; do not emit one redundant figure for every div5/div10 source.
    condition_names = list(args.active_shifts)
    combined_geometry_plot_paths: Dict[str, str] = {}
    for shift in condition_names:
        for metric in METRIC_KEYS:
            condition_endpoint = float(max(levels_for_shift(levels, shift)))
            for plot_scale in plot_scales:
                plot_endpoint_bars(
                    aggregate,
                    metric,
                    shift,
                    condition_endpoint,
                    output_dir / f"{shift}_{metric}_endpoint_absolute_{plot_scale}.png",
                    percentage=False,
                    baseline_aggregate=aggregate,
                    percentage_reference_aggregate=percentage_aggregate,
                    log_scale=plot_scale == "log",
                    show_std=_COMPUTE_PLOT_STD,
                    y_pad_fraction=args.y_pad_fraction,
                )
            plot_endpoint_bars(
                percentage_aggregate,
                metric,
                shift,
                condition_endpoint,
                output_dir / f"{shift}_{metric}_endpoint_percentage_worsening.png",
                percentage=True,
                baseline_aggregate=aggregate,
                show_std=_COMPUTE_PLOT_STD,
                y_pad_fraction=args.y_pad_fraction,
            )

    geometry_source_modes = [f"geometry_{source}" for source in args.active_geometry_sources]
    if geometry_source_modes:
        for method in ("angle", "isotropic", "voxel"):
            method_modes = [
                source_mode
                for source_mode in geometry_source_modes
                if source_mode.removeprefix("geometry_").startswith(f"{method}_")
            ]
            if not method_modes:
                continue
            method_slug = GEOMETRY_METHOD_FILE_SLUGS.get(method, method)
            for log_scale, scale_slug in ((True, "log"), (False, "linear")):
                output_path = output_dir / f"{ABLATION_PREFIX}_combined_global_endpoint_bars_{method_slug}_{scale_slug}.png"
                plot_combined_geometry_source_bars(
                    aggregate,
                    percentage_aggregate,
                    method_modes,
                    output_path,
                    f"Combined global error ({GEOMETRY_METHOD_LABELS.get(method, method.title())} remeshing, {scale_slug} scale)",
                    log_scale=log_scale,
                    percentage=False,
                    show_std=_COMPUTE_PLOT_STD,
                    y_pad_fraction=args.y_pad_fraction,
                )
                combined_geometry_plot_paths[f"{method_slug}_{scale_slug}"] = str(output_path)
            relative_reference_slug = "smart" if REFERENCE_MODEL == "SMART" else "reference"
            output_path = output_dir / f"{ABLATION_PREFIX}_combined_global_relative_vs_{relative_reference_slug}_{method_slug}.png"
            plot_combined_geometry_source_bars(
                aggregate,
                percentage_aggregate,
                method_modes,
                output_path,
                f"Combined global relative difference from {REFERENCE_MODEL_LABEL} ({GEOMETRY_METHOD_LABELS.get(method, method.title())} remeshing)",
                log_scale=False,
                percentage=True,
                show_std=False,
                y_pad_fraction=args.y_pad_fraction,
            )
            combined_geometry_plot_paths[f"{method_slug}_relative_vs_smart"] = str(output_path)

        average_factors = sorted(
            {
                int(source.removeprefix("geometry_").rsplit("div", 1)[1])
                for source in geometry_source_modes
                if "_div" in source
            }
        )
        if all(
            f"geometry_{method}_div{factor}" in geometry_source_modes
            for method in ("angle", "isotropic", "voxel")
            for factor in average_factors
        ):
            average_absolute = average_remeshing_rows(
                aggregate,
                geometry_source_modes,
                percentage=False,
            )
            average_percentage = average_remeshing_rows(
                percentage_aggregate,
                geometry_source_modes,
                percentage=True,
            )
            average_source_modes = [f"geometry_average_div{factor}" for factor in average_factors]
            output_path = output_dir / f"{ABLATION_PREFIX}_combined_global_endpoint_bars_remeshing_average_linear.png"
            plot_combined_geometry_source_bars(
                average_absolute,
                average_percentage,
                average_source_modes,
                output_path,
                "Combined global error (average across the three remeshing methods, linear scale)",
                log_scale=False,
                percentage=False,
                show_std=_COMPUTE_PLOT_STD,
                y_pad_fraction=args.y_pad_fraction,
            )
            combined_geometry_plot_paths["remeshing_average_linear"] = str(output_path)

    if args.compact_endpoint_summary:
        summary_path = output_dir / f"{ABLATION_PREFIX}_combined_global_endpoint_summary_linear.png"
        plot_compact_endpoint_summary(
            aggregate,
            percentage_aggregate,
            levels,
            geometry_source_modes,
            summary_path,
            args.y_pad_fraction,
        )
        if summary_path.is_file():
            combined_geometry_plot_paths["compact_endpoint_summary_linear"] = str(summary_path)

    table_paths = {
        metric: write_wide_metric_tables(
            output_dir,
            aggregate,
            percentage_aggregate,
            metric,
            args.active_shifts,
            levels,
            args.active_geometry_sources,
            include_std=_COMPUTE_PLOT_STD,
        )
        for metric in METRIC_KEYS
    }

    metadata = {
        "models": list(MODEL_ORDER),
        "labels": MODEL_LABELS,
        "configs": dict(config_map(args)),
        "checkpoints": dict(args.checkpoints),
        "active_shifts": list(args.active_shifts),
        "shift_levels": {shift: list(levels_for_shift(levels, shift)) for shift in SHIFT_ORDER},
        "run_ids": run_ids,
        "run_selection": str(args.run_selection),
        "top_selection_metric": str(args.top_selection_metric),
        "top_selection_candidate_pool_size": candidate_pool_size,
        "top_selection_candidate_pool_requested": int(args.top_selection_candidates),
        "top_selection_rows": (
            str(output_dir / "top_angle_div10_range_ablation_selection.csv")
            if args.run_selection == "top_angle_div10_range_ablation" and top_selection_rows
            else str(output_dir / "top_pairwise_improvement_selection.csv")
            if args.run_selection == "top_pairwise_improvement" and top_selection_rows
            else str(output_dir / "top_deal_mean_improvement_selection.csv")
            if args.run_selection == "top_deal_mean_improvement" and top_selection_rows
            else None
        ),
        "top_selection_improved_model": args.top_selection_improved_model,
        "top_selection_reference_model": args.top_selection_reference_model,
        "top_selection_conditions": args.top_selection_conditions,
        "top_selection_min_condition_improvement_percent": args.top_selection_min_condition_improvement_percent,
        "candidate_split": args.candidate_split,
        "seed": int(args.seed),
        "encoder_input_budget": input_budget,
        "surface_query_points": int(args.surface_query_points),
        "volume_query_points": int(args.volume_query_points),
        "views_per_mode": int(args.views_per_mode),
        "view_batch_size": int(args.view_batch_size),
        "model_repeats": int(args.model_repeats),
        "density_estimator": str(args.density_estimator),
        "density_knn_k": int(args.density_knn_k),
        "active_geometry_sources": list(args.active_geometry_sources),
        "geometry_decimation_factors": geometry_factors,
        "geometry_vtp_dirs": {key: str(value) for key, value in geometry_vtp_dirs.items()},
        "geometry_common_completed_subset_size": len(geometry_candidate_ids) if geometry_candidate_ids is not None else 0,
        "experiment_preset": str(args.experiment_preset),
        "plot_scales": plot_scales,
        "compute_plot_std": bool(_COMPUTE_PLOT_STD),
        "font_scale": float(args.font_scale),
        "y_pad_fraction": float(args.y_pad_fraction),
        "compact_endpoint_summary": bool(args.compact_endpoint_summary),
        "paper_table_paths": table_paths,
        "combined_geometry_plot_paths": combined_geometry_plot_paths,
        "reported_metric": "combined_global_rel_l2",
        "plot_style": "grouped endpoint bars only; no line plots",
        "percentage_reference": f"{REFERENCE_MODEL_LABEL} at the same shift and severity; reference bars are 0 percent",
        "percentage_aggregation": "percentage of aggregate mean errors; standard deviation is computed from paired per-view percentages",
        "protocol": f"{ABLATION_TABLE_TITLE}; identical paired views and queries across models. VTP modes use their remeshed coordinates directly and are never beta/sine reweighted.",
    }
    with (output_dir / "comparison_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    print(f"Saved comparison outputs to {output_dir}")


if __name__ == "__main__":
    main()
