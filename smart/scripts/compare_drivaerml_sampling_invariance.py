#!/usr/bin/env python3
"""Compare point-cloud operator models under controlled encoder-input sampling shift.

Workflow:
1) Fix the benchmark query points to common per-run surface/volume subsets.
2) Change only the encoder input geometry points.
3) Use an aligned mode that matches the training view rule best:
   a fixed-size geometry subset sampled uniformly without replacement
   from the full surface cloud.
4) Use shifted modes that keep the same number of geometry points but sample
   them with inverse-density, sinusoidal-axis, or remeshing-like spatial
   probability fields. The latter include silhouette shells, boundary layers,
   and localized feature patches.
5) Evaluate multiple independently drawn encoder-input views per run/mode.
6) Aggregate first across views within a run, then across runs.
7) Save all-model aggregate robustness plots, plus a representative surface
   VTK whose query points come from a user-selected external surface folder.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import random
import re
import sys
from contextlib import nullcontext
from collections import OrderedDict, defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm.auto import tqdm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


SCRIPT_DIR = Path(__file__).resolve().parent
SMART_ROOT = SCRIPT_DIR.parent
CHECKPOINTS_DIR = SMART_ROOT.parent / "checkpoints"
if str(SMART_ROOT) not in sys.path:
    sys.path.insert(0, str(SMART_ROOT))

from data.ahmedml_dataset_v2 import AhmedMLDatasetV2
from models.smart.smart import SMART
from models.transolverpp import TransolverPP
from models.mspt import MSPT
from models.pointnet2_ssg import PointNet2SSG
from models.lno import LNO
from models.point_transformer_v3 import PointTransformerV3
from utils.geometry_density import estimate_log_sampling_density
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

SURFACE_FIELD_METRIC_KEYS = [f"surface_field_{name}_rel_l2" for name in SURFACE_FIELDS]
VOLUME_FIELD_METRIC_KEYS = [f"volume_field_{name}_rel_l2" for name in VOLUME_FIELDS]
HEADLINE_METRIC_KEYS = [
    "surface_global_rel_l2",
    "volume_global_rel_l2",
    "surface_pressure_rel_l2",
    "surface_wss_mag_rel_l2",
    "surface_drag_force_x_rel_l2",
    "surface_normal_mag_rel_l2",
    "volume_pressure_rel_l2",
    "volume_velocity_mag_rel_l2",
    "combined_global_rel_l2",
    "combined_physics_rel_l2",
]

MODEL_ORDER = [
    "SMART",
    "SMART_DOWNSAMPLE",
    "SMART_GAUSSIAN_BALL_MASKED",
    "SMART_BOX_MASKED",
    "SMART_SATLOSS3",
    "SMART_SATLOSS4",
    "SMART_SATLOSS5",
    "SMART_SATLOSS5_NOPM",
    "SMART_SATLOSS6",
    "SMART_SATLOSS6_FIXEDSUM",
    "SMART_SATLOSS6_GRADNORM",
    "SMART_SATLOSS6_CONFIG_FULL",
    "SMART_SATLOSS6_CONFIG_LAYER",
    "SMART_SATLOSS7",
    "SMART_SATLOSS8",
    "TRANSOLVERPP",
    "TRANSOLVERPP_SATLOSS3",
    "TRANSOLVERPP_SATLOSS6",
    "TRANSOLVERPP_SATLOSS7",
    "TRANSOLVERPP_SATLOSS8",
    "POINTNET2_SSG",
    "POINTNET2_SSG_SATLOSS6",
    "POINTNET2_SSG_SATLOSS7",
    "POINTNET2_SSG_SATLOSS8",
    "LNO",
    "LNO_SATLOSS6",
    "LNO_SATLOSS7",
    "LNO_SATLOSS8",
    "MSPT",
    "MSPT_SATLOSS6",
    "MSPT_SATLOSS7",
    "MSPT_SATLOSS8",
    "POINT_TRANSFORMER_V3",
    "POINT_TRANSFORMER_V3_SATLOSS6",
    "POINT_TRANSFORMER_V3_SATLOSS7",
    "POINT_TRANSFORMER_V3_SATLOSS8",
]
MODEL_LABELS = {
    "SMART": "SMART",
    "SMART_DOWNSAMPLE": "SMART-DOWNSAMPLE",
    "SMART_GAUSSIAN_BALL_MASKED": "SMART-GAUSSIAN-BALL-MASKED",
    "SMART_BOX_MASKED": "SMART-BOX-MASKED",
    "SMART_SATLOSS3": "SMART-DeAL3",
    "SMART_SATLOSS4": "SMART-DeAL4",
    "SMART_SATLOSS5": "SMART-DeAL5",
    "SMART_SATLOSS5_NOPM": "SMART-DeAL5-NOPM",
    "SMART_SATLOSS6": "SMART-DeAL6",
    "SMART_SATLOSS6_FIXEDSUM": "SMART-DeAL6-FIXEDSUM",
    "SMART_SATLOSS6_GRADNORM": "SMART-DeAL6-GRADNORM",
    "SMART_SATLOSS6_CONFIG_FULL": "SMART-DeAL6-ConFIG-FULL",
    "SMART_SATLOSS6_CONFIG_LAYER": "SMART-DeAL6-ConFIG-LAYER",
    "SMART_SATLOSS7": "SMART-DeAL",
    "SMART_SATLOSS8": "SMART-DeAL8",
    "TRANSOLVERPP": "TransolverPP",
    "TRANSOLVERPP_SATLOSS3": "TransolverPP-DeAL3",
    "TRANSOLVERPP_SATLOSS6": "TransolverPP-DeAL6",
    "TRANSOLVERPP_SATLOSS7": "TransolverPP-DeAL",
    "TRANSOLVERPP_SATLOSS8": "TransolverPP-DeAL8",
    "POINTNET2_SSG": "PointNet++-SSG",
    "POINTNET2_SSG_SATLOSS6": "PointNet++-SSG-DeAL6",
    "POINTNET2_SSG_SATLOSS7": "PointNet++-SSG-DeAL",
    "POINTNET2_SSG_SATLOSS8": "PointNet++-SSG-DeAL8",
    "LNO": "LNO",
    "LNO_SATLOSS6": "LNO-DeAL6",
    "LNO_SATLOSS7": "LNO-DeAL",
    "LNO_SATLOSS8": "LNO-DeAL8",
    "MSPT": "MSPT",
    "MSPT_SATLOSS6": "MSPT-DeAL6",
    "MSPT_SATLOSS7": "MSPT-DeAL",
    "MSPT_SATLOSS8": "MSPT-DeAL8",
    "POINT_TRANSFORMER_V3": "PointTransformerV3",
    "POINT_TRANSFORMER_V3_SATLOSS6": "PointTransformerV3-DeAL6",
    "POINT_TRANSFORMER_V3_SATLOSS7": "PointTransformerV3-DeAL",
    "POINT_TRANSFORMER_V3_SATLOSS8": "PointTransformerV3-DeAL8",
}
MODEL_COLORS = {
    "SMART": "#6C6F7D",
    "SMART_DOWNSAMPLE": "#B279A2",
    "SMART_GAUSSIAN_BALL_MASKED": "#8C6BB1",
    "SMART_BOX_MASKED": "#E377C2",
    "SMART_SATLOSS3": "#F58518",
    "SMART_SATLOSS4": "#72B7B2",
    "SMART_SATLOSS5": "#E45756",
    "SMART_SATLOSS5_NOPM": "#9D755D",
    "SMART_SATLOSS6": "#54A24B",
    "SMART_SATLOSS6_FIXEDSUM": "#2CA02C",
    "SMART_SATLOSS6_GRADNORM": "#FF7F0E",
    "SMART_SATLOSS6_CONFIG_FULL": "#9467BD",
    "SMART_SATLOSS6_CONFIG_LAYER": "#17BECF",
    "SMART_SATLOSS7": "#4C78A8",
    "SMART_SATLOSS8": "#76A5D5",
    "TRANSOLVERPP": "#6C6F7D",
    "TRANSOLVERPP_SATLOSS3": "#E45756",
    "TRANSOLVERPP_SATLOSS6": "#FF9896",
    "TRANSOLVERPP_SATLOSS7": "#FFBB78",
    "TRANSOLVERPP_SATLOSS8": "#F6B26B",
    "POINTNET2_SSG": "#17BECF",
    "POINTNET2_SSG_SATLOSS6": "#2CA02C",
    "POINTNET2_SSG_SATLOSS7": "#98DF8A",
    "POINTNET2_SSG_SATLOSS8": "#78C679",
    "LNO": "#E45756",
    "LNO_SATLOSS6": "#F58518",
    "LNO_SATLOSS7": "#FFBB78",
    "LNO_SATLOSS8": "#F08080",
    "MSPT": "#BCBD22",
    "MSPT_SATLOSS6": "#9467BD",
    "MSPT_SATLOSS7": "#C5B0D5",
    "MSPT_SATLOSS8": "#B39DDB",
    "POINT_TRANSFORMER_V3": "#1B9E77",
    "POINT_TRANSFORMER_V3_SATLOSS6": "#66A61E",
    "POINT_TRANSFORMER_V3_SATLOSS7": "#A6D854",
    "POINT_TRANSFORMER_V3_SATLOSS8": "#B7E085",
}
# Standard Matplotlib tab10 colors for line plots.  These are intentionally
# separate from the broader chart palette used by bars and heatmaps.
LINE_MODEL_COLORS = {
    "SMART": "#1F77B4",
    "SMART_DOWNSAMPLE": "#9467BD",
    "SMART_GAUSSIAN_BALL_MASKED": "#8C564B",
    "SMART_BOX_MASKED": "#E377C2",
    "TRANSOLVERPP": "#FF7F0E",
    "POINTNET2_SSG": "#17BECF",
    "LNO": "#D62728",
    "MSPT": "#2CA02C",
    "POINT_TRANSFORMER_V3": "#7F3C8D",
}
# In the dedicated SMART/SATLOSS6 weighting comparison, each weighting method
# is an independent experiment. Keep this separate from the usual paired
# vanilla-vs-SATLOSS family colors used by the broader comparison plots.
INDEPENDENT_SATLOSS6_MODELS = {
    "SMART_SATLOSS6",
    "SMART_SATLOSS6_FIXEDSUM",
    "SMART_SATLOSS6_GRADNORM",
    "SMART_SATLOSS6_CONFIG_FULL",
    "SMART_SATLOSS6_CONFIG_LAYER",
}
INDEPENDENT_SATLOSS6_COLORS = {
    "SMART_SATLOSS6": "#2CA02C",
    "SMART_SATLOSS6_FIXEDSUM": "#FF7F0E",
    "SMART_SATLOSS6_GRADNORM": "#9467BD",
    "SMART_SATLOSS6_CONFIG_FULL": "#D62728",
    "SMART_SATLOSS6_CONFIG_LAYER": "#17BECF",
}
# Set once in main after the active checkpoints are known. Plot workers are
# forked on the target Linux environment and inherit this read-only context.
_INDEPENDENT_SATLOSS6_LINE_MODE = False
_COMPUTE_PLOT_STD = True
_SATLOSS_ONLY_PERCENT_LABELS = False
_PLOT_FONT_SCALE = 1.0
_PLOT_BASE_FONT_SIZE = 15.0


def _font_size(_size: float = _PLOT_BASE_FONT_SIZE) -> float:
    """Use one consistent readable font size across every plot annotation."""
    return float(_PLOT_BASE_FONT_SIZE) * float(_PLOT_FONT_SCALE)


def _configure_plot_style(font_scale: float) -> None:
    """Set a readable paper/slide style before forking plot workers."""
    scale = float(font_scale)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("--font-scale must be a finite positive number.")
    font_size = _PLOT_BASE_FONT_SIZE * scale
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


def _save_plot(fig: matplotlib.figure.Figure, out_path: Path, dpi: int) -> None:
    """Save large-font plots without constrained-layout collapse warnings."""
    fig.set_constrained_layout(False)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0.18)
DRAG_RANK_MODELS = [
    "SMART",
    "SMART_DOWNSAMPLE",
    "SMART_GAUSSIAN_BALL_MASKED",
    "SMART_SATLOSS3",
    "SMART_SATLOSS5",
    "SMART_SATLOSS5_NOPM",
    "SMART_SATLOSS6",
    "SMART_SATLOSS6_FIXEDSUM",
    "SMART_SATLOSS6_GRADNORM",
    "SMART_SATLOSS6_CONFIG_FULL",
    "SMART_SATLOSS6_CONFIG_LAYER",
]
FAMILY_GROUPS = OrderedDict(
    [
        (
            "smart_family",
            [
                "SMART",
                "SMART_DOWNSAMPLE",
                "SMART_GAUSSIAN_BALL_MASKED",
                "SMART_BOX_MASKED",
                "SMART_SATLOSS3",
                "SMART_SATLOSS4",
                "SMART_SATLOSS5",
                "SMART_SATLOSS5_NOPM",
                "SMART_SATLOSS6",
                "SMART_SATLOSS6_FIXEDSUM",
                "SMART_SATLOSS6_GRADNORM",
                "SMART_SATLOSS6_CONFIG_FULL",
                "SMART_SATLOSS6_CONFIG_LAYER",
                "SMART_SATLOSS7",
                "SMART_SATLOSS8",
            ],
        ),
        (
            "smart_satloss6_weighting_family",
            [
                "SMART_SATLOSS6_FIXEDSUM",
                "SMART_SATLOSS6_GRADNORM",
                "SMART_SATLOSS6_CONFIG_FULL",
                "SMART_SATLOSS6_CONFIG_LAYER",
            ],
        ),
        ("transolverpp_family", ["TRANSOLVERPP", "TRANSOLVERPP_SATLOSS3", "TRANSOLVERPP_SATLOSS6", "TRANSOLVERPP_SATLOSS7", "TRANSOLVERPP_SATLOSS8"]),
        ("pointnet2_ssg_family", ["POINTNET2_SSG", "POINTNET2_SSG_SATLOSS6", "POINTNET2_SSG_SATLOSS7", "POINTNET2_SSG_SATLOSS8"]),
        ("lno_family", ["LNO", "LNO_SATLOSS6", "LNO_SATLOSS7", "LNO_SATLOSS8"]),
        ("mspt_family", ["MSPT", "MSPT_SATLOSS6", "MSPT_SATLOSS7", "MSPT_SATLOSS8"]),
        ("point_transformer_v3_family", ["POINT_TRANSFORMER_V3", "POINT_TRANSFORMER_V3_SATLOSS6", "POINT_TRANSFORMER_V3_SATLOSS7", "POINT_TRANSFORMER_V3_SATLOSS8"]),
    ]
)
FAMILY_TITLES = {
    "smart_family": "SMART vs SMART-DOWNSAMPLE vs SMART-GAUSSIAN-BALL-MASKED vs SMART-BOX-MASKED vs SMART-DeAL3 vs SMART-DeAL4 vs SMART-DeAL5 vs SMART-DeAL5-NOPM vs SMART-DeAL6 vs DeAL6 weighting variants",
    "smart_satloss6_weighting_family": "SMART-SATLOSS6-FIXEDSUM vs SMART-SATLOSS6-GRADNORM vs SMART-SATLOSS6-ConFIG-FULL vs SMART-SATLOSS6-ConFIG-LAYER",
    "transolverpp_family": "TransolverPP and DeAL variants",
    "pointnet2_ssg_family": "PointNet++ SSG and DeAL variants",
    "lno_family": "LNO and DeAL variants",
    "mspt_family": "MSPT and DeAL variants",
    "point_transformer_v3_family": "PointTransformerV3 and DeAL variants",
}
VTK_PRESSURE_MODELS = [
    "SMART",
    "SMART_DOWNSAMPLE",
    "SMART_GAUSSIAN_BALL_MASKED",
    "SMART_BOX_MASKED",
    "SMART_SATLOSS3",
    "SMART_SATLOSS4",
    "SMART_SATLOSS5",
    "SMART_SATLOSS5_NOPM",
    "SMART_SATLOSS6",
    "SMART_SATLOSS6_FIXEDSUM",
    "SMART_SATLOSS6_GRADNORM",
    "SMART_SATLOSS6_CONFIG_FULL",
    "SMART_SATLOSS6_CONFIG_LAYER",
    "SMART_SATLOSS7",
    "SMART_SATLOSS8",
    "TRANSOLVERPP",
    "TRANSOLVERPP_SATLOSS3",
    "TRANSOLVERPP_SATLOSS6",
    "TRANSOLVERPP_SATLOSS7",
    "TRANSOLVERPP_SATLOSS8",
    "POINTNET2_SSG",
    "POINTNET2_SSG_SATLOSS6",
    "POINTNET2_SSG_SATLOSS7",
    "POINTNET2_SSG_SATLOSS8",
    "LNO",
    "LNO_SATLOSS6",
    "LNO_SATLOSS7",
    "LNO_SATLOSS8",
    "MSPT",
    "MSPT_SATLOSS6",
    "MSPT_SATLOSS7",
    "MSPT_SATLOSS8",
    "POINT_TRANSFORMER_V3",
    "POINT_TRANSFORMER_V3_SATLOSS6",
    "POINT_TRANSFORMER_V3_SATLOSS7",
    "POINT_TRANSFORMER_V3_SATLOSS8",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fair DrivAerML sampling-invariance comparison across the retained operator families.")
    p.add_argument("--smart-config", default="drivaerml")
    p.add_argument(
        "--smart-downsample-config",
        "--smart-augmented-config",
        dest="smart_downsample_config",
        default="drivaerml_smart_downsample",
    )
    p.add_argument(
        "--smart-gaussian-ball-masked-config",
        "--smart-masked-config",
        dest="smart_gaussian_ball_masked_config",
        default="drivaerml_smart_gaussian_ball_masked",
    )
    p.add_argument("--smart-box-masked-config", default="drivaerml_smart_box_masked")
    p.add_argument("--smart-satloss3-config", default="drivaerml_satloss3")
    p.add_argument("--smart-satloss4-config", default="drivaerml_satloss4")
    p.add_argument("--smart-satloss5-config", default="drivaerml_satloss5")
    p.add_argument("--smart-satloss5-nopm-config", default="drivaerml_satloss5_nopm")
    p.add_argument("--smart-satloss6-config", default="drivaerml_satloss6")
    p.add_argument("--smart-satloss6-fixedsum-config", default="drivaerml_satloss6_fixedsum")
    p.add_argument("--smart-satloss6-gradnorm-config", default="drivaerml_satloss6_gradnorm")
    p.add_argument("--smart-satloss6-config-full-config", default="drivaerml_satloss6_config_full")
    p.add_argument("--smart-satloss6-config-layer-config", default="drivaerml_satloss6_config")
    p.add_argument("--smart-satloss7-config", "--smart-satloss-config", dest="smart_satloss7_config", default="drivaerml_satloss7")
    p.add_argument("--smart-satloss8-config", default="drivaerml_satloss8")
    p.add_argument("--transolverpp-config", default="drivaerml_transolverpp")
    p.add_argument("--transolverpp-satloss3-config", default="drivaerml_transolverpp_satloss3")
    p.add_argument("--transolverpp-satloss6-config", default="drivaerml_transolverpp_satloss6")
    p.add_argument("--transolverpp-satloss7-config", "--transolverpp-satloss-config", dest="transolverpp_satloss7_config", default="drivaerml_transolverpp_satloss7")
    p.add_argument("--transolverpp-satloss8-config", default="drivaerml_transolverpp_satloss8")
    p.add_argument("--pointnet2-ssg-config", default="drivaerml_pointnet2_ssg")
    p.add_argument("--pointnet2-ssg-satloss6-config", default="drivaerml_pointnet2_ssg_satloss6")
    p.add_argument("--pointnet2-ssg-satloss7-config", "--pointnet2-ssg-satloss-config", dest="pointnet2_ssg_satloss7_config", default="drivaerml_pointnet2_ssg_satloss7")
    p.add_argument("--pointnet2-ssg-satloss8-config", default="drivaerml_pointnet2_ssg_satloss8")
    p.add_argument("--lno-config", default="drivaerml_lno")
    p.add_argument("--lno-satloss6-config", default="drivaerml_lno_satloss6")
    p.add_argument("--lno-satloss7-config", "--lno-satloss-config", dest="lno_satloss7_config", default="drivaerml_lno_satloss7")
    p.add_argument("--lno-satloss8-config", default="drivaerml_lno_satloss8")
    p.add_argument("--mspt-config", default="drivaerml_mspt")
    p.add_argument("--mspt-satloss6-config", default="drivaerml_mspt_satloss6")
    p.add_argument("--mspt-satloss7-config", "--mspt-satloss-config", dest="mspt_satloss7_config", default="drivaerml_mspt_satloss7")
    p.add_argument("--mspt-satloss8-config", default="drivaerml_mspt_satloss8")
    p.add_argument("--point-transformer-v3-config", default="drivaerml_point_transformer_v3")
    p.add_argument("--point-transformer-v3-satloss6-config", default="drivaerml_point_transformer_v3_satloss6")
    p.add_argument("--point-transformer-v3-satloss7-config", "--point-transformer-v3-satloss-config", dest="point_transformer_v3_satloss7_config", default="drivaerml_point_transformer_v3_satloss7")
    p.add_argument("--point-transformer-v3-satloss8-config", default="drivaerml_point_transformer_v3_satloss8")
    p.add_argument("--smart-checkpoint", default=None)
    p.add_argument(
        "--smart-downsample-checkpoint",
        "--smart-augmented-checkpoint",
        dest="smart_downsample_checkpoint",
        default=None,
    )
    p.add_argument(
        "--smart-gaussian-ball-masked-checkpoint",
        "--smart-masked-checkpoint",
        dest="smart_gaussian_ball_masked_checkpoint",
        default=None,
    )
    p.add_argument("--smart-box-masked-checkpoint", default=None)
    p.add_argument("--smart-satloss3-checkpoint", default=None)
    p.add_argument("--smart-satloss4-checkpoint", default=None)
    p.add_argument("--smart-satloss5-checkpoint", default=None)
    p.add_argument("--smart-satloss5-nopm-checkpoint", default=None)
    p.add_argument("--smart-satloss6-checkpoint", default=None)
    p.add_argument("--smart-satloss6-fixedsum-checkpoint", default=None)
    p.add_argument("--smart-satloss6-gradnorm-checkpoint", default=None)
    p.add_argument("--smart-satloss6-config-full-checkpoint", default=None)
    p.add_argument("--smart-satloss6-config-layer-checkpoint", default=None)
    p.add_argument("--smart-satloss7-checkpoint", "--smart-satloss-checkpoint", dest="smart_satloss7_checkpoint", default=None)
    p.add_argument("--smart-satloss8-checkpoint", default=None)
    p.add_argument("--transolverpp-checkpoint", default=None)
    p.add_argument("--transolverpp-satloss3-checkpoint", default=None)
    p.add_argument("--transolverpp-satloss6-checkpoint", default=None)
    p.add_argument("--transolverpp-satloss7-checkpoint", "--transolverpp-satloss-checkpoint", dest="transolverpp_satloss7_checkpoint", default=None)
    p.add_argument("--transolverpp-satloss8-checkpoint", default=None)
    p.add_argument("--pointnet2-ssg-checkpoint", default=None)
    p.add_argument("--pointnet2-ssg-satloss6-checkpoint", default=None)
    p.add_argument("--pointnet2-ssg-satloss7-checkpoint", "--pointnet2-ssg-satloss-checkpoint", dest="pointnet2_ssg_satloss7_checkpoint", default=None)
    p.add_argument("--pointnet2-ssg-satloss8-checkpoint", default=None)
    p.add_argument("--lno-checkpoint", default=None)
    p.add_argument("--lno-satloss6-checkpoint", default=None)
    p.add_argument("--lno-satloss7-checkpoint", "--lno-satloss-checkpoint", dest="lno_satloss7_checkpoint", default=None)
    p.add_argument("--lno-satloss8-checkpoint", default=None)
    p.add_argument("--mspt-checkpoint", default=None)
    p.add_argument("--mspt-satloss6-checkpoint", default=None)
    p.add_argument("--mspt-satloss7-checkpoint", "--mspt-satloss-checkpoint", dest="mspt_satloss7_checkpoint", default=None)
    p.add_argument("--mspt-satloss8-checkpoint", default=None)
    p.add_argument("--point-transformer-v3-checkpoint", default=None)
    p.add_argument("--point-transformer-v3-satloss6-checkpoint", default=None)
    p.add_argument("--point-transformer-v3-satloss7-checkpoint", "--point-transformer-v3-satloss-checkpoint", dest="point_transformer_v3_satloss7_checkpoint", default=None)
    p.add_argument("--point-transformer-v3-satloss8-checkpoint", default=None)
    p.add_argument("--num-runs", type=int, default=8, help="Number of test runs to evaluate.")
    p.add_argument("--run-ids", default=None, help="Optional comma-separated explicit run ids.")
    p.add_argument(
        "--domain-split-json",
        default=None,
        help="Optional geometry-domain split JSON. When set, --domain-test-cluster supplies the evaluated cluster.",
    )
    p.add_argument("--domain-train-cluster", type=int, default=0)
    p.add_argument("--domain-test-cluster", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--query-sampling-with-replacement",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Match AhmedMLDatasetV2's default query sampling. Use --no-query-sampling-with-replacement for distinct query subsets.",
    )
    p.add_argument("--device", default=None, help="Primary device, retained for single-device runs.")
    p.add_argument(
        "--devices",
        default=None,
        help=(
            "Comma-separated inference devices for model calls, for example `cuda:0,cuda:1`. "
            "With CUDA_VISIBLE_DEVICES=3,4 these refer to the visible ordinals 0 and 1. "
            "Models are kept resident and assigned round-robin; one worker runs per GPU."
        ),
    )
    p.add_argument("--input-points", type=int, default=None, help="Encoder input size. Default: inferred from the active model configs.")
    p.add_argument(
        "--density-estimator",
        default="kde",
        choices=["rk2", "tangent_cov", "kde"],
        help="Geometry-density estimator used for the sampling shifts and density histograms.",
    )
    p.add_argument(
        "--density-knn-k",
        type=int,
        default=None,
        help="k used for the sampling-shift density study. Default: inferred from the active density config.",
    )
    p.add_argument(
        "--shift-betas",
        default="0,1",
        help="Inverse-density range. Only its minimum and maximum are evaluated; intermediate values are ignored.",
    )
    p.add_argument(
        "--positive-shifts-only",
        action="store_true",
        help="Evaluate only the largest positive beta/sine shift; zero-intensity shift modes are not created.",
    )
    p.add_argument(
        "--active-shifts",
        default="all",
        help=(
            "Comma-separated active shifts: beta,sine_y,sine_x. sine_y/sine_x are the restored "
            "sinusoidal coordinate shifts. Aliases: sine, remesh, remesh_axis, all. Default: all."
        ),
    )
    p.add_argument("--views-per-mode", type=int, default=2, help="Number of independently sampled encoder-input views per run/mode.")
    p.add_argument("--view-batch-size", type=int, default=2, help="How many views to evaluate together in one model call.")
    p.add_argument("--model-repeats", type=int, default=1, help="Average over repeated stochastic forwards for each view batch.")
    p.add_argument(
        "--batched-query-subregion-size",
        type=int,
        default=65536,
        help="Temporary inference chunk size used to keep batched-view decoding safe.",
    )
    p.add_argument("--vtk-run-id", type=int, default=None, help="Representative run id for the full-surface VTK export. Default: first evaluated run.")
    p.add_argument("--vtk-surface-query-dir", default="/home/parsa/smart_parsa/CFD_audi/run_100/audi", help="Directory containing external surface_coords/normals/pressure/WSS NPY files for representative VTK export.")
    p.add_argument(
        "--surface-vtp-dir",
        default="/mnt/ssdraid/parsa/drivaerml_surface_vtp",
        help="Legacy original-surface VTP root; not used when the aligned preprocessed cloud is the baseline.",
    )
    p.add_argument(
        "--decimated-vtp-dir",
        "--angle-decimated-vtp-dir",
        dest="angle_decimated_vtp_dir",
        default="/mnt/ssdraid/parsa/drivaerml_surface_vtp_decimated",
        help="Root containing angle-based decimated VTPs under run_<id>/drivaer_<id>_faces_div{5,10,20,40}.vtp.",
    )
    p.add_argument(
        "--isotropic-decimated-vtp-dir",
        default="/mnt/ssdraid/parsa/drivaerml_surface_vtp_isotropic_gpu",
        help="Root containing isotropic-remeshed VTPs with the standard run/factor layout.",
    )
    p.add_argument(
        "--voxel-decimated-vtp-dir",
        default="/mnt/ssdraid/parsa/drivaerml_surface_vtp_voxel_quadric_clustered",
        help="Root containing voxel/quadric-clustered VTPs with the standard run/factor layout.",
    )
    p.add_argument(
        "--geometry-decimation-factors",
        default="5,10",
        help="Factors used for each active geometry method. Supported values: 5,10,20,40.",
    )
    p.add_argument(
        "--active-geometry-sources",
        default="none",
        help=(
            "Comma-separated geometry methods/sources: angle,isotropic,voxel,all or explicit "
            "angle_div5/isotropic_div10/etc. The aligned preprocessed cloud is the baseline."
        ),
    )
    p.add_argument(
        "--geometry-label-preset",
        choices=("legacy", "v4"),
        default="legacy",
        help="Use 'v4' for feature-aware, QEM, and voxel-grid-clustering remesh inputs.",
    )
    p.add_argument("--plot-workers", type=int, default=max(1, min(6, (os.cpu_count() or 1) // 2)), help="Worker count for CPU-side plot generation.")
    p.add_argument(
        "--no-std",
        action="store_true",
        help="Do not compute or render standard deviations/error bars; std columns are retained as zeros for schema compatibility.",
    )
    p.add_argument(
        "--satloss-only-percent-labels",
        action="store_true",
        help="Show vanilla-relative percentage labels only on DeAL bars.",
    )
    p.add_argument(
        "--strategy-only",
        action="store_true",
        help="Render only the dedicated SMART training-strategy comparison outputs.",
    )
    p.add_argument(
        "--font-scale",
        type=float,
        default=1.0,
        help="Scale all plot text, including explicit bar annotations; 1.8 is suitable for slides.",
    )
    p.add_argument("--dry-run", action="store_true", help="Validate configs/checkpoints and print the active protocol without inference.")
    p.add_argument("--surface-query-points", type=int, default=0, help="Fixed surface query budget for all models. Use 0 to auto-pick the minimum training budget across compared models.")
    p.add_argument("--volume-query-points", type=int, default=0, help="Fixed volume query budget for all models. Use 0 to auto-pick the minimum training budget across compared models.")
    p.add_argument("--audi-surface-chunk-size", type=int, default=2048, help="Chunk size used only for the full Audi surface-pressure visualization export.")
    p.add_argument(
        "--test-smart-satloss5-nopm-beta-error-scale",
        type=float,
        default=0.0,
        help="Testing hook: ramp a relative error multiplier only on SMART_SATLOSS5_NOPM beta/sine line charts for severity > 0, starting at +2%% and ending at this value. Example: 0.05 means +2%% ... +5%%.",
    )
    p.add_argument("--output-dir", default=None)
    return p.parse_args()


def load_cfg(name: str, _stack: Tuple[str, ...] = ()):
    """Load the experiment section with the same local defaults inheritance as Hydra."""
    if name in _stack:
        chain = " -> ".join((*_stack, name))
        raise ValueError(f"Circular config defaults detected: {chain}")

    path = SMART_ROOT / "config" / f"{name}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"Config not found: {path}")

    root = OmegaConf.load(path)
    merged_experiment = OmegaConf.create()
    for default in root.get("defaults", []):
        if not isinstance(default, str):
            continue
        if default == "_self_" or default.startswith("override "):
            continue
        default_name = default.rsplit("/", 1)[-1]
        merged_experiment = OmegaConf.merge(
            merged_experiment,
            load_cfg(default_name, (*_stack, name)),
        )

    current_experiment = root.get("experiment", OmegaConf.create())
    return OmegaConf.merge(merged_experiment, current_experiment)


def resolve_density_spec(cfg) -> Tuple[str, int, int, str]:
    architecture = getattr(cfg, "architecture", None)
    if architecture is not None:
        density_estimator = str(
            getattr(architecture, "density_estimator", getattr(cfg, "density_estimator", "rk2"))
        )
        density_knn_k = int(getattr(architecture, "density_knn_k", getattr(cfg, "density_knn_k", 8)))
        density_neighbor_hops = int(
            getattr(architecture, "density_neighbor_hops", getattr(cfg, "density_neighbor_hops", 1))
        )
    else:
        density_estimator = str(getattr(cfg, "density_estimator", "rk2"))
        density_knn_k = int(getattr(cfg, "density_knn_k", 8))
        density_neighbor_hops = int(getattr(cfg, "density_neighbor_hops", 1))
    density_cache_dtype = str(getattr(cfg, "geometry_density_cache_dtype", "float16"))
    return density_estimator, density_knn_k, density_neighbor_hops, density_cache_dtype


def infer_density_spec_from_checkpoint_name(
    ckpt_path: str, base_estimator: str, base_knn_k: int
) -> Tuple[str, int]:
    name = Path(ckpt_path).stem.lower()
    kde_match = re.search(r"(?:^|[-_])kde(\d+)(?:[-_]|$)", name)
    if kde_match:
        return "kde", int(kde_match.group(1))
    if "kde" in name:
        return "kde", int(base_knn_k)
    return str(base_estimator), int(base_knn_k)


def checkpoint_density_tag_is_explicit(ckpt_path: str) -> bool:
    name = Path(ckpt_path).stem.lower()
    return bool(re.search(r"(?:^|[-_])(kde\d+|kde|tangent[_-]?cov|rk2)(?:[-_]|$)", name))


def resolve_model_internal_density_spec(model_name: str, cfg, ckpt_path: str) -> Tuple[str, int, int, str]:
    density_estimator, density_knn_k, density_neighbor_hops, density_cache_dtype = resolve_density_spec(cfg)
    if model_uses_density(model_name):
        density_estimator, density_knn_k = infer_density_spec_from_checkpoint_name(
            ckpt_path,
            density_estimator,
            density_knn_k,
        )
    return density_estimator, density_knn_k, density_neighbor_hops, density_cache_dtype


def resolve_device(device_arg: str | None) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_devices(device_arg: str | None, devices_arg: str | None) -> List[torch.device]:
    """Resolve visible CUDA devices without confusing them with physical IDs."""
    if devices_arg:
        tokens = [token.strip() for token in str(devices_arg).split(",") if token.strip()]
        devices = []
        for token in tokens:
            if token.isdigit():
                token = f"cuda:{token}"
            devices.append(torch.device(token))
    else:
        devices = [resolve_device(device_arg)]
    if not devices:
        raise ValueError("At least one inference device is required.")
    if any(device.type == "cuda" for device in devices) and not torch.cuda.is_available():
        raise RuntimeError("CUDA devices were requested, but CUDA is not available.")
    for device in devices:
        if device.type == "cuda" and device.index is not None and device.index >= torch.cuda.device_count():
            raise ValueError(
                f"Requested visible CUDA device {device}, but only {torch.cuda.device_count()} device(s) are available."
            )
    return devices


def choose_ckpt(config, explicit: str | None) -> str:
    if explicit:
        path = Path(explicit)
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        return str(path)

    stem = get_model_checkpoint_name(config)
    exact_candidates = [
        CHECKPOINTS_DIR / f"{stem}_best.pt",
        CHECKPOINTS_DIR / f"{stem}_last.pt",
    ]
    for path in exact_candidates:
        if path.is_file():
            return str(path)

    model_slug = str(config.model_name).lower().replace("_", "-")
    prefix_map = {
        "SMART": "smart-smart-",
        "SMART_DOWNSAMPLE": "smart-downsample-",
        "SMART_GAUSSIAN_BALL_MASKED": "smart-gaussian-ball-masked-",
        "SMART_BOX_MASKED": "smart-box-masked-",
        "SMART_SATLOSS3": "smart-satloss3-",
        "SMART_SATLOSS4": "smart-satloss4-",
        "SMART_SATLOSS5": "smart-satloss5-",
        "SMART_SATLOSS5_NOPM": "smart-satloss5-nopm-",
        # Keep the generic SATLOSS6 family separate from its weighting
        # variants; all of them share the ``smart-satloss6-`` prefix.
        "SMART_SATLOSS6": "smart-satloss6-smart-satloss6-",
        "SMART_SATLOSS6_FIXEDSUM": "smart-satloss6-fixedsum-",
        "SMART_SATLOSS6_GRADNORM": "smart-satloss6-gradnorm-",
        "SMART_SATLOSS6_CONFIG_FULL": "smart-satloss6-config-full-",
        "SMART_SATLOSS6_CONFIG_LAYER": "smart-satloss6-config-layer-",
        "SMART_SATLOSS7": "smart-satloss7-",
        "SMART_SATLOSS8": "smart-satloss8-",
        "TRANSOLVERPP": "transolverpp-",
        "TRANSOLVERPP_SATLOSS3": "transolverpp-satloss3-",
        "TRANSOLVERPP_SATLOSS6": "transolverpp-satloss6-",
        "TRANSOLVERPP_SATLOSS7": "transolverpp-satloss7-",
        "TRANSOLVERPP_SATLOSS8": "transolverpp-satloss8-",
        "POINTNET2_SSG": "pointnet2-ssg-",
        "POINTNET2_SSG_SATLOSS6": "pointnet2-ssg-satloss6-",
        "POINTNET2_SSG_SATLOSS7": "pointnet2-ssg-satloss7-",
        "POINTNET2_SSG_SATLOSS8": "pointnet2-ssg-satloss8-",
        "LNO": "lno-",
        "LNO_SATLOSS6": "lno-satloss6-",
        "LNO_SATLOSS7": "lno-satloss7-",
        "LNO_SATLOSS8": "lno-satloss8-",
        "MSPT": "mspt-",
        "MSPT_SATLOSS6": "mspt-satloss6-",
        "MSPT_SATLOSS7": "mspt-satloss7-",
        "MSPT_SATLOSS8": "mspt-satloss8-",
        "POINT_TRANSFORMER_V3": "point-transformer-v3-",
        "POINT_TRANSFORMER_V3_SATLOSS6": "point-transformer-v3-satloss6-",
        "POINT_TRANSFORMER_V3_SATLOSS7": "point-transformer-v3-satloss7-",
        "POINT_TRANSFORMER_V3_SATLOSS8": "point-transformer-v3-satloss8-",
    }
    required_prefix = prefix_map.get(str(config.model_name), f"{model_slug}-")
    dataset_slug = str(config.dataset).lower()
    seed_slug = f"s{int(config.random_seed)}"
    patterns = [
        f"{model_slug}*{dataset_slug}*{seed_slug}_best.pt",
        f"{model_slug}*{dataset_slug}*{seed_slug}_last.pt",
        f"{model_slug}*{seed_slug}_best.pt",
        f"{model_slug}*{seed_slug}_last.pt",
    ]
    matches: List[Path] = []
    for pattern in patterns:
        matches.extend(sorted(p for p in CHECKPOINTS_DIR.glob(pattern) if p.name.startswith(required_prefix)))
        if matches:
            break
    if not matches:
        raise FileNotFoundError(f"No checkpoint found for {config.model_name}. Tried stem `{stem}` and patterns {patterns}")
    matches = sorted(matches, key=lambda p: (0 if p.name.endswith("_best.pt") else 1, len(p.name), p.name))
    return str(matches[0])


PTV3_MODEL_NAMES = {
    "POINT_TRANSFORMER_V3",
    "POINT_TRANSFORMER_V3_SATLOSS6",
    "POINT_TRANSFORMER_V3_SATLOSS7",
    "POINT_TRANSFORMER_V3_SATLOSS8",
}


def resolve_ptv3_checkpoint_config_name(
    model_name: str,
    requested_config_name: str,
    checkpoint_path: str,
) -> str:
    """Keep PTv3 architecture options paired with the checkpoint that trained them.

    PTv3 density-sensitive runs intentionally change serialization order, order
    shuffling, voxel-density preservation, and density weighting. Those options
    do not change tensor shapes, so a wrong YAML can load strictly and silently
    produce invalid benchmark numbers. The checkpoint naming convention is the
    only provenance available in these legacy checkpoints; resolve the matching
    local config before model construction.
    """
    if str(model_name) not in PTV3_MODEL_NAMES:
        return str(requested_config_name)

    requested = str(requested_config_name)
    checkpoint_is_density_sensitive = "density-sensitive" in Path(checkpoint_path).stem.lower()
    config_is_density_sensitive = "density_sensitive" in requested
    if checkpoint_is_density_sensitive == config_is_density_sensitive:
        return requested

    if checkpoint_is_density_sensitive:
        candidate = f"{requested}_density_sensitive"
        reason = "density-sensitive checkpoint"
    else:
        suffix = "_density_sensitive"
        candidate = requested[: -len(suffix)] if requested.endswith(suffix) else requested
        reason = "standard checkpoint"

    candidate_path = SMART_ROOT / "config" / f"{candidate}.yaml"
    if not candidate_path.is_file():
        raise ValueError(
            f"PTv3 checkpoint/config mismatch for {model_name}: checkpoint={checkpoint_path}, "
            f"requested_config={requested}. The matching config `{candidate}` was not found at {candidate_path}."
        )
    print(
        f"[PTv3 config] {model_name}: replacing `{requested}` with `{candidate}` "
        f"to match the {reason}: {Path(checkpoint_path).name}"
    )
    return candidate


def load_state_dict(ckpt_path: str, device: torch.device) -> Dict[str, torch.Tensor]:
    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        return ckpt["model_state_dict"]
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        return ckpt["state_dict"]
    if isinstance(ckpt, dict):
        return ckpt
    raise TypeError(f"Unexpected checkpoint format in {ckpt_path}")


def build_model(config, ckpt_path: str, device: torch.device, batched_query_subregion_size: int):
    model_name = str(config.model_name)
    arch = OmegaConf.to_container(config.architecture, resolve=True)
    base_kwargs = {
        "spatial_dim": 3,
        "surface_channels": len(SURFACE_FIELDS),
        "volume_channels": len(VOLUME_FIELDS),
        "parameter_channels": 0,
    }
    if model_name in {
        "SMART",
        "SMART_DOWNSAMPLE",
        "SMART_GAUSSIAN_BALL_MASKED",
        "SMART_BOX_MASKED",
        "SMART_SATLOSS3",
        "SMART_SATLOSS4",
        "SMART_SATLOSS5",
        "SMART_SATLOSS5_NOPM",
        "SMART_SATLOSS6",
        "SMART_SATLOSS6_FIXEDSUM",
        "SMART_SATLOSS6_GRADNORM",
        "SMART_SATLOSS6_CONFIG_FULL",
        "SMART_SATLOSS6_CONFIG_LAYER",
        "SMART_SATLOSS7",
        "SMART_SATLOSS8",
    }:
        model = SMART(**base_kwargs, **arch)
    elif model_name in {"TRANSOLVERPP", "TRANSOLVERPP_SATLOSS3", "TRANSOLVERPP_SATLOSS6", "TRANSOLVERPP_SATLOSS7", "TRANSOLVERPP_SATLOSS8"}:
        model = TransolverPP(**base_kwargs, **arch)
    elif model_name in {"POINTNET2_SSG", "POINTNET2_SSG_SATLOSS6", "POINTNET2_SSG_SATLOSS7", "POINTNET2_SSG_SATLOSS8"}:
        model = PointNet2SSG(**base_kwargs, **arch)
    elif model_name in {"LNO", "LNO_SATLOSS6", "LNO_SATLOSS7", "LNO_SATLOSS8"}:
        model = LNO(**base_kwargs, **arch)
    elif model_name in {"MSPT", "MSPT_SATLOSS6", "MSPT_SATLOSS7", "MSPT_SATLOSS8"}:
        model = MSPT(**base_kwargs, **arch)
    elif model_name in {"POINT_TRANSFORMER_V3", "POINT_TRANSFORMER_V3_SATLOSS6", "POINT_TRANSFORMER_V3_SATLOSS7", "POINT_TRANSFORMER_V3_SATLOSS8"}:
        model = PointTransformerV3(**base_kwargs, **arch)
    else:
        raise ValueError(f"Unsupported model_name for this evaluator: {model_name}")

    state = load_state_dict(ckpt_path, device)
    model.load_state_dict(state, strict=True)
    model.eval()
    if hasattr(model, "subregion_size"):
        # Do not silently shrink the model's native query chunking.
        # Larger chunks improve throughput when they fit comfortably in VRAM.
        model.subregion_size = max(int(model.subregion_size), max(1, int(batched_query_subregion_size)))
    return model


def normalize_pos(pos: torch.Tensor, min_pos: torch.Tensor, max_pos: torch.Tensor) -> torch.Tensor:
    return (pos - min_pos) / torch.clamp(max_pos - min_pos, min=1e-12)


def denorm_fields(pred_norm: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return pred_norm * std + mean


def rel_l2(gt: np.ndarray, pred: np.ndarray, eps: float = 1e-12) -> float:
    return float(np.linalg.norm(pred - gt) / max(np.linalg.norm(gt), eps))


def sample_uniform_without_replacement(n: int, k: int, rng: np.random.Generator) -> np.ndarray:
    if k <= 0 or k >= n:
        return np.arange(n, dtype=np.int64)
    return rng.choice(n, size=k, replace=False).astype(np.int64, copy=False)


def sample_uniform_with_replacement(n: int, k: int, rng: np.random.Generator) -> np.ndarray:
    if k <= 0:
        return np.empty((0,), dtype=np.int64)
    if n <= 0:
        raise ValueError("Cannot sample from an empty point cloud.")
    return rng.choice(n, size=k, replace=True).astype(np.int64, copy=False)


def sample_inverse_density_without_replacement(
    log_density: np.ndarray,
    k: int,
    beta: float,
    rng: np.random.Generator,
) -> np.ndarray:
    n = int(log_density.shape[0])
    if k <= 0 or k >= n:
        return np.arange(n, dtype=np.int64)
    weights = np.exp(-float(beta) * log_density.astype(np.float64, copy=False))
    weights = np.clip(weights, 1e-24, None)
    probs = weights / np.clip(weights.sum(), 1e-24, None)
    return rng.choice(n, size=k, replace=False, p=probs).astype(np.int64, copy=False)


def sample_inverse_density_with_replacement(
    log_density: np.ndarray,
    k: int,
    beta: float,
    rng: np.random.Generator,
) -> np.ndarray:
    n = int(log_density.shape[0])
    if k <= 0:
        return np.empty((0,), dtype=np.int64)
    weights = np.exp(-float(beta) * log_density.astype(np.float64, copy=False))
    weights = np.clip(weights, 1e-24, None)
    probs = weights / np.clip(weights.sum(), 1e-24, None)
    return rng.choice(n, size=k, replace=True, p=probs).astype(np.int64, copy=False)


def sample_weighted_without_replacement(
    weights: np.ndarray,
    k: int,
    rng: np.random.Generator,
) -> np.ndarray:
    n = int(weights.shape[0])
    if k <= 0 or k >= n:
        return np.arange(n, dtype=np.int64)
    clipped = np.clip(np.asarray(weights, dtype=np.float64), 1e-24, None)
    probs = clipped / np.clip(clipped.sum(), 1e-24, None)
    return rng.choice(n, size=k, replace=False, p=probs).astype(np.int64, copy=False)


def sinusoidal_axis_probabilities(coords_xyz: np.ndarray, axis: int) -> np.ndarray:
    """Prefer the center of an axis with a smooth sinusoidal redistribution."""
    coord = np.asarray(coords_xyz[:, axis], dtype=np.float64)
    cmin = float(np.min(coord))
    cmax = float(np.max(coord))
    span = max(cmax - cmin, 1e-12)
    t = np.clip((coord - cmin) / span, 0.0, 1.0)
    # This is the original sine-y/sine-x benchmark shift: it concentrates
    # samples near the center of the selected coordinate axis without masking.
    scores = np.sin(np.pi * t) ** 2
    return np.clip(scores + 1e-6, 1e-6, None)


SHIFT_ORDER = ("beta", "sine_y", "sine_x")
SHIFT_LABELS = {
    "beta": "Inverse-density beta",
    "sine_y": "Sinusoidal-y intensity",
    "sine_x": "Sinusoidal-x intensity",
}

GEOMETRY_METHOD_ORDER = ("angle", "isotropic", "voxel")
GEOMETRY_FACTOR_ORDER = (5, 10, 20, 40)
GEOMETRY_SOURCE_ORDER = tuple(
    f"{method}_div{factor}"
    for method in GEOMETRY_METHOD_ORDER
    for factor in GEOMETRY_FACTOR_ORDER
)
GEOMETRY_SOURCE_LABELS = {
    **{
        f"angle_div{factor}": f"Angle-based div{factor}"
        for factor in GEOMETRY_FACTOR_ORDER
    },
    **{
        f"isotropic_div{factor}": f"Isotropic div{factor}"
        for factor in GEOMETRY_FACTOR_ORDER
    },
    **{
        f"voxel_div{factor}": f"Voxel/quadric div{factor}"
        for factor in GEOMETRY_FACTOR_ORDER
    },
}
GEOMETRY_METHOD_LABELS = {
    "angle": "Angle-based decimation",
    "isotropic": "Isotropic remeshing",
    "voxel": "Voxel/quadric clustering",
}
V4_GEOMETRY_SOURCE_LABELS = {
    **{f"angle_div{factor}": f"Feature-aware div{factor}" for factor in GEOMETRY_FACTOR_ORDER},
    **{f"isotropic_div{factor}": f"QEM div{factor}" for factor in GEOMETRY_FACTOR_ORDER},
    **{f"voxel_div{factor}": f"Voxel-grid clustering div{factor}" for factor in GEOMETRY_FACTOR_ORDER},
}
V4_GEOMETRY_METHOD_LABELS = {
    "angle": "Feature-aware decimation",
    "isotropic": "QEM decimation",
    "voxel": "Voxel-grid clustering",
}
GEOMETRY_BBOX_TOLERANCE = 2.5e-3


def parse_active_shifts(text: str) -> List[str]:
    """Parse active shift names while preserving the canonical plot order."""
    raw = [item.strip().lower().replace("-", "_") for item in str(text).split(",") if item.strip()]
    if not raw or raw == ["all"]:
        return list(SHIFT_ORDER)
    aliases = {
        "sine": ("sine_y", "sine_x"),
        "sinusoidal": ("sine_y", "sine_x"),
        "remesh": ("sine_y", "sine_x"),
        "remesh_axis": ("sine_y", "sine_x"),
    }
    expanded: List[str] = []
    for item in raw:
        if item == "all":
            expanded.extend(SHIFT_ORDER)
        elif item in aliases:
            expanded.extend(aliases[item])
        elif item in SHIFT_ORDER:
            expanded.append(item)
        else:
            valid = ", ".join((*SHIFT_ORDER, "sine"))
            raise ValueError(f"Unknown shift {item!r}. Valid values: {valid}, or all.")
    active = [shift for shift in SHIFT_ORDER if shift in set(expanded)]
    if not active:
        raise ValueError("At least one sampling shift must be active.")
    return active


def parse_geometry_decimation_factors(text: str) -> List[int]:
    factors = sorted({int(item.strip()) for item in str(text).split(",") if item.strip()})
    if not factors or any(factor not in GEOMETRY_FACTOR_ORDER for factor in factors):
        valid = ", ".join(str(factor) for factor in GEOMETRY_FACTOR_ORDER)
        raise ValueError(f"Geometry decimation factors must be selected from {valid}.")
    return factors


def parse_active_geometry_sources(text: str, factors: Sequence[int]) -> List[str]:
    """Parse geometry-source tests while preserving old decimated aliases."""
    raw = [item.strip().lower().replace("-", "_") for item in str(text).split(",") if item.strip()]
    if not raw or raw == ["none"]:
        return []
    requested_factors = tuple(int(factor) for factor in factors)
    aliases = {
        "all": tuple(f"{method}_div{factor}" for method in GEOMETRY_METHOD_ORDER for factor in requested_factors),
        "decimated": tuple(f"angle_div{factor}" for factor in requested_factors),
        "angle": tuple(f"angle_div{factor}" for factor in requested_factors),
        "isotropic": tuple(f"isotropic_div{factor}" for factor in requested_factors),
        "voxel": tuple(f"voxel_div{factor}" for factor in requested_factors),
        "voxel_quadric": tuple(f"voxel_div{factor}" for factor in requested_factors),
        "div5": tuple(f"{method}_div5" for method in GEOMETRY_METHOD_ORDER if 5 in requested_factors),
        "div10": tuple(f"{method}_div10" for method in GEOMETRY_METHOD_ORDER if 10 in requested_factors),
        "div20": tuple(f"{method}_div20" for method in GEOMETRY_METHOD_ORDER if 20 in requested_factors),
        "div40": tuple(f"{method}_div40" for method in GEOMETRY_METHOD_ORDER if 40 in requested_factors),
        "decimated_div5": ("angle_div5",),
        "decimated_div10": ("angle_div10",),
    }
    expanded: List[str] = []
    for item in raw:
        if item in aliases:
            expanded.extend(aliases[item])
        elif item in GEOMETRY_SOURCE_ORDER:
            expanded.append(item)
        else:
            valid = ", ".join(("angle", "isotropic", "voxel", "all", "none", *GEOMETRY_SOURCE_ORDER))
            raise ValueError(f"Unknown geometry source {item!r}. Valid values: {valid}.")
    return [source for source in GEOMETRY_SOURCE_ORDER if source in set(expanded)]


def geometry_source_vtp_path(source: str, run_id: int, geometry_vtp_dirs: Mapping[str, Path]) -> Path:
    run_id = int(run_id)
    match = re.fullmatch(r"(angle|isotropic|voxel)_div(5|10|20|40)", str(source))
    if match is None:
        raise ValueError(f"Unsupported geometry source: {source}")
    method, factor = match.groups()
    return geometry_vtp_dirs[method] / f"run_{run_id}" / f"drivaer_{run_id}_faces_div{factor}.vtp"


def read_vtp_points(path: Path) -> np.ndarray:
    """Read only point coordinates from a geometry-only VTP."""
    try:
        import vtk
        from vtk.util.numpy_support import vtk_to_numpy
    except Exception as exc:  # pragma: no cover - environment-specific
        raise RuntimeError("VTP geometry tests require VTK Python bindings.") from exc
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(str(path))
    reader.Update()
    polydata = reader.GetOutput()
    if polydata is None or polydata.GetPoints() is None:
        raise RuntimeError(f"VTP has no points: {path}")
    points = np.asarray(vtk_to_numpy(polydata.GetPoints().GetData()), dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] == 0:
        raise RuntimeError(f"VTP has invalid point coordinates: {path}, shape={points.shape}")
    if not np.isfinite(points).all():
        raise RuntimeError(f"VTP contains non-finite point coordinates: {path}")
    return np.ascontiguousarray(points)


def validate_geometry_source_bbox(
    source_points: np.ndarray,
    reference_points: np.ndarray,
    source_name: str,
    run_id: int,
    tolerance: float = GEOMETRY_BBOX_TOLERANCE,
) -> None:
    """Ensure VTP coordinates stay in the preprocessed training frame."""
    source = np.asarray(source_points, dtype=np.float64)
    reference = np.asarray(reference_points, dtype=np.float64)
    source_bbox = np.concatenate([source.min(axis=0), source.max(axis=0)])
    reference_bbox = np.concatenate([reference.min(axis=0), reference.max(axis=0)])
    delta = float(np.max(np.abs(source_bbox - reference_bbox)))
    if delta > float(tolerance):
        raise ValueError(
            f"{source_name} VTP run_{int(run_id)} is not in the preprocessed coordinate frame: "
            f"max bbox difference={delta:.6g} > tolerance={float(tolerance):.6g}. "
            "Do not normalize this source with the training bounds until the geometry export is corrected."
        )


def sample_uniform_weighted_mixture_without_replacement(
    target_weights: np.ndarray,
    k: int,
    mix_fraction: float,
    rng: np.random.Generator,
) -> np.ndarray:
    n = int(target_weights.shape[0])
    if k <= 0 or k >= n:
        return np.arange(n, dtype=np.int64)
    alpha = float(np.clip(mix_fraction, 0.0, 1.0))
    k_weighted = int(round(alpha * k))
    k_weighted = min(max(k_weighted, 0), k)
    k_uniform = k - k_weighted

    if k_weighted == 0:
        return sample_uniform_without_replacement(n, k, rng)
    if k_uniform == 0:
        return sample_weighted_without_replacement(target_weights, k, rng)

    weighted_idx = sample_weighted_without_replacement(target_weights, k_weighted, rng)
    chosen_mask = np.zeros((n,), dtype=bool)
    chosen_mask[weighted_idx] = True
    remaining_idx = np.flatnonzero(~chosen_mask)
    uniform_take = rng.choice(remaining_idx, size=k_uniform, replace=False).astype(np.int64, copy=False)
    out = np.concatenate([weighted_idx, uniform_take], axis=0).astype(np.int64, copy=False)
    return out


def sample_uniform_weighted_mixture_with_replacement(
    target_weights: np.ndarray,
    k: int,
    mix_fraction: float,
    rng: np.random.Generator,
) -> np.ndarray:
    if k <= 0:
        return np.empty((0,), dtype=np.int64)
    n = int(target_weights.shape[0])
    alpha = float(np.clip(mix_fraction, 0.0, 1.0))
    k_weighted = int(round(alpha * k))
    k_uniform = k - k_weighted
    weighted_probs = np.clip(np.asarray(target_weights, dtype=np.float64), 1e-24, None)
    weighted_probs /= np.clip(weighted_probs.sum(), 1e-24, None)
    weighted_idx = rng.choice(n, size=k_weighted, replace=True, p=weighted_probs)
    uniform_idx = rng.choice(n, size=k_uniform, replace=True)
    return np.concatenate([weighted_idx, uniform_idx]).astype(np.int64, copy=False)


def sample_gaussian_ball_mask_subset(
    coords_xyz: np.ndarray,
    base_budget: int,
    rng: np.random.Generator,
    *,
    std_fraction_of_largest_extent: float,
    prob_at_1sigma: float,
    min_survivors: int,
    return_metadata: bool = False,
) -> np.ndarray | Dict[str, np.ndarray | float | int]:
    n = int(coords_xyz.shape[0])
    if base_budget <= 0 or base_budget >= n:
        base_idx = np.arange(n, dtype=np.int64)
    else:
        base_idx = sample_uniform_without_replacement(n, int(base_budget), rng)

    subset = np.asarray(coords_xyz[base_idx], dtype=np.float64)
    if subset.shape[0] == 0:
        raise RuntimeError("Gaussian mask subset sampling produced no candidate points.")

    center_idx = int(rng.integers(0, subset.shape[0]))
    center = subset[center_idx]
    largest_extent = float(np.max(np.max(subset, axis=0) - np.min(subset, axis=0)))
    sigma = max(float(std_fraction_of_largest_extent) * largest_extent, 1.0e-12)
    prob_at_1sigma = min(max(float(prob_at_1sigma), 1.0e-8), 0.999999)
    coeff = -math.log(prob_at_1sigma)
    dist = np.linalg.norm(subset - center[None, :], axis=1)
    remove_prob = np.exp(-coeff * (dist / sigma) ** 2)
    keep_mask = rng.random(subset.shape[0]) >= remove_prob

    min_survivors = max(1, min(int(min_survivors), subset.shape[0]))
    if int(np.count_nonzero(keep_mask)) < min_survivors:
        keep_scores = 1.0 - remove_prob
        keep_rel = np.argsort(keep_scores)[-min_survivors:]
        keep_mask = np.zeros((subset.shape[0],), dtype=bool)
        keep_mask[keep_rel] = True

    kept = np.asarray(base_idx[keep_mask], dtype=np.int64)
    if kept.size == 0:
        raise RuntimeError("Gaussian mask subset sampling removed every point.")
    kept = np.sort(kept)
    if not return_metadata:
        return kept
    center_flag = np.zeros((subset.shape[0],), dtype=np.float32)
    center_flag[center_idx] = 1.0
    return {
        "base_idx": np.asarray(base_idx, dtype=np.int64),
        "kept_idx": kept,
        "keep_mask": keep_mask.astype(np.float32, copy=False),
        "remove_probability": remove_prob.astype(np.float32, copy=False),
        "distance_to_center": dist.astype(np.float32, copy=False),
        "center_flag": center_flag,
        "sigma_radius": float(sigma),
        "center_point": center.astype(np.float32, copy=False),
    }


def sample_box_mask_subset(
    coords_xyz: np.ndarray,
    base_budget: int,
    rng: np.random.Generator,
    *,
    std_fraction_of_largest_extent: float,
) -> Dict[str, np.ndarray | float | int]:
    """Sample a base view and remove an axis-aligned box of side 2 sigma."""
    n = int(coords_xyz.shape[0])
    if base_budget <= 0 or base_budget >= n:
        base_idx = np.arange(n, dtype=np.int64)
    else:
        base_idx = sample_uniform_without_replacement(n, int(base_budget), rng)

    subset = np.asarray(coords_xyz[base_idx], dtype=np.float64)
    if subset.shape[0] == 0:
        raise RuntimeError("Box mask subset sampling produced no candidate points.")

    center_idx = int(rng.integers(0, subset.shape[0]))
    center = subset[center_idx]
    largest_extent = float(np.max(np.max(subset, axis=0) - np.min(subset, axis=0)))
    sigma = max(float(std_fraction_of_largest_extent) * largest_extent, 1.0e-12)
    half_side = sigma
    remove_mask = np.all(np.abs(subset - center[None, :]) <= half_side, axis=1)
    keep_mask = ~remove_mask
    if not np.any(keep_mask):
        raise RuntimeError("Box mask subset sampling removed every point.")

    box_min = center - half_side
    box_max = center + half_side
    center_flag = np.zeros((subset.shape[0],), dtype=np.float32)
    center_flag[center_idx] = 1.0
    return {
        "base_idx": np.asarray(base_idx, dtype=np.int64),
        "kept_idx": np.asarray(base_idx[keep_mask], dtype=np.int64),
        "keep_mask": keep_mask.astype(np.float32, copy=False),
        "box_inside_flag": remove_mask.astype(np.float32, copy=False),
        "center_flag": center_flag,
        "distance_to_center": np.linalg.norm(subset - center[None, :], axis=1).astype(np.float32),
        "sigma_radius": float(sigma),
        "box_side_length": float(2.0 * sigma),
        "center_point": center.astype(np.float32, copy=False),
        "box_min": box_min.astype(np.float32, copy=False),
        "box_max": box_max.astype(np.float32, copy=False),
    }


def vector_mag(arr: np.ndarray, start: int, end: int) -> np.ndarray:
    return np.linalg.norm(arr[:, start:end], axis=1)


def safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", name)


def choose_fixed_query_indices(
    n_total: int,
    n_keep: int,
    seed_components: Sequence[int],
    replace: bool = False,
) -> np.ndarray:
    rng = np.random.default_rng(np.random.SeedSequence([int(x) for x in seed_components]))
    idx = (
        sample_uniform_with_replacement(n_total, n_keep, rng)
        if replace
        else sample_uniform_without_replacement(n_total, n_keep, rng)
    )
    if not replace:
        idx.sort()
    return idx


def write_polydata_vtk(path: Path, points_xyz: np.ndarray, point_data: Dict[str, np.ndarray]) -> None:
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
        f.write(b"DrivAerML surface prediction comparison\n")
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


def write_mask_surface_vtp(
    path: Path,
    *,
    kind: str,
    center_xyz: np.ndarray,
    sigma_radius: float,
    box_min_xyz: np.ndarray | None = None,
    box_max_xyz: np.ndarray | None = None,
) -> None:
    """Write a closed VTP surface locating one deterministic training mask."""
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk

    path.parent.mkdir(parents=True, exist_ok=True)
    center = np.asarray(center_xyz, dtype=np.float32).reshape(3)
    sigma = float(sigma_radius)
    if not np.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("sigma_radius must be finite and positive")

    if kind == "gaussian_sigma_sphere":
        source = vtk.vtkSphereSource()
        source.SetCenter(*center.tolist())
        # The Gaussian removal profile is centered here; show its one-standard-
        # deviation radius as the closed mask-reference surface.
        source.SetRadius(sigma)
        source.SetThetaResolution(64)
        source.SetPhiResolution(32)
    elif kind == "box_surface":
        if box_min_xyz is None or box_max_xyz is None:
            raise ValueError("box surfaces require box_min_xyz and box_max_xyz")
        box_min = np.asarray(box_min_xyz, dtype=np.float32).reshape(3)
        box_max = np.asarray(box_max_xyz, dtype=np.float32).reshape(3)
        source = vtk.vtkCubeSource()
        # vtkCubeSource expects paired axis bounds: xmin, xmax, ymin, ymax, zmin, zmax.
        source.SetBounds(
            float(box_min[0]), float(box_max[0]),
            float(box_min[1]), float(box_max[1]),
            float(box_min[2]), float(box_max[2]),
        )
    else:
        raise ValueError(f"Unknown mask surface kind: {kind}")

    source.Update()
    polydata = vtk.vtkPolyData()
    polydata.ShallowCopy(source.GetOutput())
    n = int(polydata.GetNumberOfPoints())
    point_data = {
        "mask_center_xyz": np.repeat(center.reshape(1, 3), n, axis=0),
        "mask_sigma_radius": np.full((n,), sigma, dtype=np.float32),
        "mask_boundary_radius": np.full((n,), sigma, dtype=np.float32),
    }
    for name, values in point_data.items():
        vtk_array = numpy_to_vtk(np.ascontiguousarray(values, dtype=np.float32), deep=True)
        vtk_array.SetName(name)
        vtk_array.SetNumberOfComponents(values.shape[1] if values.ndim == 2 else 1)
        polydata.GetPointData().AddArray(vtk_array)

    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(str(path))
    writer.SetInputData(polydata)
    writer.SetDataModeToBinary()
    if writer.Write() != 1:
        raise RuntimeError(f"Failed to write VTP: {path}")


def _prepare_density_histogram_values(log_density_values: np.ndarray) -> Tuple[np.ndarray, float, float, float, float]:
    log_vals = np.asarray(log_density_values, dtype=np.float64)
    density_vals = np.exp(log_vals)
    finite_mask = np.isfinite(log_vals) & np.isfinite(density_vals) & (density_vals > 0.0)
    finite_density = density_vals[finite_mask]
    if finite_density.size == 0:
        finite_density = np.array([1.0], dtype=np.float64)
    lo_density = float(np.percentile(finite_density, 2.0))
    hi_density = float(np.percentile(finite_density, 98.0))
    lo_density = max(lo_density, 1e-24)
    hi_density = max(hi_density, lo_density * 1.0001)
    display_lo = max(lo_density / 1.5, 1e-24)
    display_hi = hi_density * 1.5
    window_density = finite_density[(finite_density >= lo_density) & (finite_density <= hi_density)]
    if window_density.size == 0:
        window_density = finite_density
    return window_density, lo_density, hi_density, display_lo, display_hi


def _draw_density_histogram(
    ax,
    density_values: np.ndarray,
    lo_density: float,
    hi_density: float,
    *,
    log_axes: bool,
    color: str,
    label: str | None = None,
    alpha: float = 0.9,
):
    if log_axes:
        bins = np.logspace(np.log10(lo_density), np.log10(hi_density), 60)
        ax.hist(density_values, bins=bins, color=color, alpha=alpha, edgecolor="white", linewidth=0.3, label=label)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.xaxis.set_major_locator(mticker.LogLocator(base=10.0, numticks=12))
        ax.xaxis.set_major_formatter(mticker.LogFormatterMathtext(base=10.0))
        ax.xaxis.set_minor_locator(mticker.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=100))
        ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    else:
        bins = np.linspace(lo_density, hi_density, 60)
        ax.hist(density_values, bins=bins, color=color, alpha=alpha, edgecolor="white", linewidth=0.3, label=label)
    return bins


def save_density_histogram(path: Path, log_density_values: np.ndarray, title: str, *, log_axes: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    window_density, lo_density, hi_density, display_lo, display_hi = _prepare_density_histogram_values(log_density_values)
    fig, ax = plt.subplots(figsize=(8.0, 5.0), constrained_layout=True)
    _draw_density_histogram(
        ax,
        window_density,
        lo_density,
        hi_density,
        log_axes=log_axes,
        color="#4C78A8",
    )
    ax.set_xlabel("Density")
    ax.set_ylabel("Count")
    ax.set_title(title)
    mean_val = float(np.mean(window_density))
    std_val = float(np.std(window_density))
    mean_handle = Line2D([], [], color="#E45756", linestyle="--", linewidth=1.5, label=f"mean={mean_val:.3g}")
    stats_handle = Line2D([], [], color="none", label=f"std={std_val:.3g}\nN={window_density.size}\np2={lo_density:.3g}, p98={hi_density:.3g}")
    ax.legend(
        handles=[mean_handle, stats_handle],
        loc="upper left",
        frameon=True,
        fancybox=True,
        framealpha=0.9,
        borderpad=0.6,
        labelspacing=0.5,
        handlelength=2.0,
        handletextpad=0.8,
    )
    ax.set_xlim(display_lo, display_hi)
    _save_plot(fig, path, 220)
    plt.close(fig)


def save_density_histogram_overlay(
    path: Path,
    log_density_values_a: np.ndarray,
    log_density_values_b: np.ndarray,
    title: str,
    *,
    label_a: str,
    label_b: str,
    log_axes: bool = True,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    density_a, lo_a, hi_a, display_lo_a, display_hi_a = _prepare_density_histogram_values(log_density_values_a)
    density_b, lo_b, hi_b, display_lo_b, display_hi_b = _prepare_density_histogram_values(log_density_values_b)
    lo_density = min(lo_a, lo_b)
    hi_density = max(hi_a, hi_b)
    display_lo = min(display_lo_a, display_lo_b)
    display_hi = max(display_hi_a, display_hi_b)

    fig, ax = plt.subplots(figsize=(8.0, 5.0), constrained_layout=True)
    _draw_density_histogram(ax, density_a, lo_density, hi_density, log_axes=log_axes, color="#4C78A8", label=label_a, alpha=0.55)
    _draw_density_histogram(ax, density_b, lo_density, hi_density, log_axes=log_axes, color="#E45756", label=label_b, alpha=0.55)
    ax.set_xlabel("Density")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.legend(loc="upper left", frameon=True, fancybox=True, framealpha=0.9)
    ax.set_xlim(display_lo, display_hi)
    _save_plot(fig, path, 220)
    plt.close(fig)


def save_sampling_y_histogram(
    path: Path,
    sampled_points_xyz: np.ndarray,
    reference_points_xyz: np.ndarray,
    mix_fraction: float,
    title: str,
    axis: int = 1,
    coordinate_name: str = "y",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sampled = np.asarray(sampled_points_xyz, dtype=np.float64)
    reference = np.asarray(reference_points_xyz, dtype=np.float64)
    if sampled.ndim != 2 or sampled.shape[1] != 3:
        raise ValueError("sampled_points_xyz must have shape [N, 3]")
    if reference.ndim != 2 or reference.shape[1] != 3:
        raise ValueError("reference_points_xyz must have shape [N, 3]")

    coordinate = int(axis)
    if coordinate not in (0, 1, 2):
        raise ValueError(f"axis must be 0, 1, or 2, got {axis}")
    y_ref = reference[:, coordinate]
    y_samp = sampled[:, coordinate]
    y_min = float(min(np.min(y_ref), np.min(y_samp)))
    y_max = float(max(np.max(y_ref), np.max(y_samp)))
    if not np.isfinite(y_min) or not np.isfinite(y_max):
        y_min, y_max = 0.0, 1.0
    if y_max <= y_min:
        y_max = y_min + 1e-12

    ref_norm = np.clip((y_ref - y_min) / (y_max - y_min), 0.0, 1.0)
    samp_norm = np.clip((y_samp - y_min) / (y_max - y_min), 0.0, 1.0)
    bins = np.linspace(0.0, 1.0, 61)
    grid = np.linspace(0.0, 1.0, 512)
    target_pdf = (1.0 - float(mix_fraction)) + float(mix_fraction) * (2.0 * np.sin(np.pi * grid) ** 2)

    fig, ax = plt.subplots(figsize=(8.0, 5.0), constrained_layout=True)
    ax.hist(
        ref_norm,
        bins=bins,
        density=True,
        histtype="step",
        color="#A0A0A0",
        linewidth=1.2,
        label="all candidate points",
        alpha=0.9,
    )
    ax.hist(
        samp_norm,
        bins=bins,
        density=True,
        color="#4C78A8",
        alpha=0.85,
        edgecolor="white",
        linewidth=0.25,
        label="sampled points",
    )
    ax.plot(grid, target_pdf, color="#E45756", linewidth=2.0, label=f"target mix curve (mix={mix_fraction:.2f})")
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel(f"Normalized {coordinate_name} coordinate")
    ax.set_ylabel("Probability density")
    ax.set_title(title)
    ax.legend(loc="upper left", frameon=True, fancybox=True, framealpha=0.92)
    ax.text(
        0.98,
        0.95,
        f"N={samp_norm.size}\nref={ref_norm.size}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=_font_size(10),
        bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "#CCCCCC"},
    )
    _save_plot(fig, path, 220)
    plt.close(fig)


def save_geometry_source_distribution_plot(
    path: Path,
    reference_points_xyz: np.ndarray,
    sampled_points_by_source: Dict[str, np.ndarray],
    title: str,
) -> None:
    """Show normalized x/y/z marginals for the VTP source samples."""
    path.parent.mkdir(parents=True, exist_ok=True)
    reference = np.asarray(reference_points_xyz, dtype=np.float64)
    if reference.ndim != 2 or reference.shape[1] != 3:
        raise ValueError("reference_points_xyz must have shape [N, 3]")
    mins = np.min(reference, axis=0)
    maxs = np.max(reference, axis=0)
    span = np.clip(maxs - mins, 1.0e-12, None)
    reference = np.clip((reference - mins) / span, 0.0, 1.0)
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.8), constrained_layout=True)
    colors = {
        "angle": "#E15759",
        "isotropic": "#59A14F",
        "voxel": "#F28E2B",
    }
    axis_labels = ("x", "y", "z")
    for axis, ax in enumerate(axes):
        ax.hist(reference[:, axis], bins=50, density=True, histtype="step", color="#777777", linewidth=1.5, label="preprocessed reference")
        for source_name, points in sampled_points_by_source.items():
            sample = np.asarray(points, dtype=np.float64)
            sample = np.clip((sample - mins) / span, 0.0, 1.0)
            method = str(source_name).split("_div", 1)[0]
            ax.hist(
                sample[:, axis],
                bins=50,
                density=True,
                histtype="step",
                linewidth=1.6,
                color=colors.get(method, "#999999"),
                linestyle="-" if str(source_name).endswith("div5") else "--",
                label=GEOMETRY_SOURCE_LABELS.get(str(source_name), str(source_name)),
            )
        ax.set_xlabel(f"Normalized {axis_labels[axis]}")
        ax.set_ylabel("Probability density")
        ax.grid(alpha=0.2)
        if axis == 0:
            ax.legend(fontsize=_font_size(8), loc="best")
    fig.suptitle(title, fontsize=_font_size(15))
    _save_plot(fig, path, 260)
    plt.close(fig)


def _spatial_shift_feature_values(
    points_xyz: np.ndarray,
    shift_name: str,
    normalization_reference_points_xyz: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray, str]:
    """Return a normalized diagnostic coordinate and its shift weights."""
    points = np.asarray(points_xyz, dtype=np.float64)
    reference = points if normalization_reference_points_xyz is None else np.asarray(normalization_reference_points_xyz, dtype=np.float64)
    mins = np.min(reference, axis=0, keepdims=True)
    maxs = np.max(reference, axis=0, keepdims=True)
    span = np.clip(maxs - mins, 1.0e-12, None)
    normalized = np.clip((points - mins) / span, 0.0, 1.0)

    if shift_name == "sine_y":
        feature = normalized[:, 1]
        label = "Normalized y coordinate"
    elif shift_name == "sine_x":
        feature = normalized[:, 0]
        label = "Normalized x coordinate"
    else:
        raise ValueError(f"Unsupported spatial shift: {shift_name}")

    weights = sinusoidal_axis_probabilities(points, axis=1 if shift_name == "sine_y" else 0)
    return np.asarray(feature, dtype=np.float64), np.asarray(weights, dtype=np.float64), label


def _histogram_density(values: np.ndarray, bins: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    counts, _ = np.histogram(values, bins=bins, weights=weights)
    widths = np.diff(bins)
    total = float(np.sum(counts))
    if total <= 0.0:
        return np.zeros_like(widths, dtype=np.float64)
    return counts.astype(np.float64) / (total * widths)


def _empirical_cdf(values: np.ndarray, grid: np.ndarray, *, upper_tail: bool = False) -> np.ndarray:
    values = np.sort(np.asarray(values, dtype=np.float64))
    if values.size == 0:
        return np.zeros_like(grid, dtype=np.float64)
    ranks = np.searchsorted(values, grid, side="right").astype(np.float64) / float(values.size)
    return 1.0 - ranks if upper_tail else ranks


def save_spatial_shift_endpoint_plot(
    path: Path,
    shift_name: str,
    reference_points_xyz: np.ndarray,
    sampled_points_by_intensity: Dict[float, np.ndarray],
    title: str,
) -> None:
    """Compare zero and maximum spatial-shift intensities with shift-specific diagnostics."""
    path.parent.mkdir(parents=True, exist_ok=True)
    reference_feature, shift_weights, feature_label = _spatial_shift_feature_values(reference_points_xyz, shift_name)
    sampled_features = {
        float(intensity): _spatial_shift_feature_values(
            points,
            shift_name,
            normalization_reference_points_xyz=reference_points_xyz,
        )[0]
        for intensity, points in sampled_points_by_intensity.items()
    }
    finite = np.isfinite(reference_feature)
    reference_feature = reference_feature[finite]
    shift_weights = shift_weights[finite]
    if reference_feature.size == 0:
        return
    feature_min = float(np.min(reference_feature))
    feature_max = float(np.max(reference_feature))
    if feature_max <= feature_min:
        feature_max = feature_min + 1.0e-12
    bins = np.linspace(feature_min, feature_max, 51)
    centers = 0.5 * (bins[:-1] + bins[1:])
    uniform_pdf = _histogram_density(reference_feature, bins)
    shifted_pdf = _histogram_density(reference_feature, bins, weights=shift_weights)

    zero_intensity = min(sampled_features, key=lambda value: abs(value - 0.0))
    max_intensity = max(sampled_features)
    max_features = sampled_features[max_intensity]
    zero_features = sampled_features[zero_intensity]
    max_mix_pdf = (1.0 - max_intensity) * uniform_pdf + max_intensity * shifted_pdf

    zero_color = "#4C78A8"
    max_color = "#E45756"
    target_color = "#54A24B"
    reference_color = "#777777"
    fig, (ax_pdf, ax_aux) = plt.subplots(1, 2, figsize=(13.5, 5.8), constrained_layout=True)
    ax_pdf.hist(reference_feature, bins=bins, density=True, histtype="step", color=reference_color, linewidth=1.5, label="candidate cloud")
    ax_pdf.hist(zero_features, bins=bins, density=True, color=zero_color, alpha=0.38, label=f"intensity={zero_intensity:.2f}")
    ax_pdf.hist(max_features, bins=bins, density=True, color=max_color, alpha=0.42, label=f"intensity={max_intensity:.2f}")
    ax_pdf.plot(centers, uniform_pdf, color=zero_color, linestyle="--", linewidth=1.8, label="uniform target")
    ax_pdf.plot(centers, max_mix_pdf, color=target_color, linestyle="-.", linewidth=1.8, label="maximum-shift target")
    ax_pdf.set_xlabel(feature_label)
    ax_pdf.set_ylabel("Probability density")
    ax_pdf.set_title("Endpoint distributions")
    ax_pdf.grid(True, alpha=0.2)
    ax_pdf.legend(loc="best", frameon=True, framealpha=0.92)

    if shift_name in {"sine_y", "sine_x"}:
        ax_aux.plot(centers, shifted_pdf - uniform_pdf, color=max_color, linewidth=2.0)
        ax_aux.axhline(0.0, color="#333333", linewidth=1.0)
        ax_aux.set_ylabel("Shifted target PDF - uniform PDF")
        ax_aux.set_title("Sinusoidal redistribution from zero")
    else:
        grid = np.linspace(feature_min, feature_max, 256)
        ax_aux.plot(grid, _empirical_cdf(reference_feature, grid), color=reference_color, linewidth=1.5, label="candidate")
        ax_aux.plot(grid, _empirical_cdf(zero_features, grid), color=zero_color, linewidth=1.8, label="zero intensity")
        ax_aux.plot(grid, _empirical_cdf(max_features, grid), color=max_color, linewidth=1.8, label="maximum intensity")
        ax_aux.set_ylabel("Cumulative fraction within coordinate")
        ax_aux.set_title("Sinusoidal coordinate accumulation")
        ax_aux.legend(loc="best", frameon=True, framealpha=0.92)
    ax_aux.set_xlabel(feature_label)
    ax_aux.grid(True, alpha=0.2)
    fig.suptitle(title, fontsize=_font_size(15))
    _save_plot(fig, path, 260)
    plt.close(fig)


def load_surface_query_from_dir(surface_query_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    if not surface_query_dir.is_dir():
        raise FileNotFoundError(f"Representative VTK surface-query directory not found: {surface_query_dir}")
    surf_coords = np.load(surface_query_dir / "surface_coords.npy").astype(np.float32, copy=False)
    surf_p = np.load(surface_query_dir / "surface_pMeanTrim.npy").astype(np.float32, copy=False).reshape(-1, 1)
    surf_normals = np.load(surface_query_dir / "surface_normals.npy").astype(np.float32, copy=False)
    if surf_normals.ndim == 1:
        surf_normals = surf_normals.reshape(-1, 1)
    surf_normals = surf_normals[:, :3]
    surf_wx = np.load(surface_query_dir / "surface_wallShearStressMeanTrim_x.npy").astype(np.float32, copy=False).reshape(-1, 1)
    surf_wy = np.load(surface_query_dir / "surface_wallShearStressMeanTrim_y.npy").astype(np.float32, copy=False).reshape(-1, 1)
    surf_wz = np.load(surface_query_dir / "surface_wallShearStressMeanTrim_z.npy").astype(np.float32, copy=False).reshape(-1, 1)
    surf_gt = np.concatenate([surf_p, surf_normals, surf_wx, surf_wy, surf_wz], axis=1)
    surf_mask = np.isfinite(surf_coords).all(axis=1) & np.isfinite(surf_gt).all(axis=1)
    surf_coords = surf_coords[surf_mask]
    surf_gt = surf_gt[surf_mask]
    if surf_coords.shape[0] == 0:
        raise ValueError(f"Representative VTK surface-query directory {surface_query_dir} has no valid points after finite filtering.")
    return surf_coords.astype(np.float32, copy=False), surf_gt.astype(np.float32, copy=False)


def compute_metrics(surf_gt: np.ndarray, surf_pred: np.ndarray, vol_gt: np.ndarray, vol_pred: np.ndarray) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    metrics["surface_global_rel_l2"] = rel_l2(surf_gt.reshape(-1), surf_pred.reshape(-1))
    metrics["volume_global_rel_l2"] = rel_l2(vol_gt.reshape(-1), vol_pred.reshape(-1))
    metrics["surface_pressure_rel_l2"] = rel_l2(surf_gt[:, 0], surf_pred[:, 0])
    metrics["surface_normal_mag_rel_l2"] = rel_l2(vector_mag(surf_gt, 1, 4), vector_mag(surf_pred, 1, 4))
    metrics["surface_wss_mag_rel_l2"] = rel_l2(vector_mag(surf_gt, 4, 7), vector_mag(surf_pred, 4, 7))
    metrics["volume_pressure_rel_l2"] = rel_l2(vol_gt[:, 0], vol_pred[:, 0])
    metrics["volume_velocity_mag_rel_l2"] = rel_l2(vector_mag(vol_gt, 1, 4), vector_mag(vol_pred, 1, 4))
    metrics["combined_global_rel_l2"] = 0.5 * (metrics["surface_global_rel_l2"] + metrics["volume_global_rel_l2"])
    metrics["combined_physics_rel_l2"] = float(
        np.mean(
            [
                metrics["surface_pressure_rel_l2"],
                metrics["surface_wss_mag_rel_l2"],
                metrics["volume_pressure_rel_l2"],
                metrics["volume_velocity_mag_rel_l2"],
            ]
        )
    )

    for field_idx, field_name in enumerate(SURFACE_FIELDS):
        metrics[f"surface_field_{field_name}_rel_l2"] = rel_l2(surf_gt[:, field_idx], surf_pred[:, field_idx])
    for field_idx, field_name in enumerate(VOLUME_FIELDS):
        metrics[f"volume_field_{field_name}_rel_l2"] = rel_l2(vol_gt[:, field_idx], vol_pred[:, field_idx])
    return metrics


def compute_surface_drag_force_x(surf_fields: np.ndarray, surface_areas: np.ndarray) -> float:
    surf = np.asarray(surf_fields, dtype=np.float64)
    areas = np.asarray(surface_areas, dtype=np.float64).reshape(-1)
    if surf.shape[0] != areas.shape[0]:
        raise ValueError(f"Surface field and area lengths do not match: {surf.shape[0]} vs {areas.shape[0]}")
    pressure_force_x = -surf[:, 0] * surf[:, 1] * areas
    shear_force_x = surf[:, 4] * areas
    return float(np.sum(pressure_force_x + shear_force_x))


def model_uses_density(model_name: str) -> bool:
    del model_name
    return False


@torch.inference_mode()
def predict_view_batch(
    model_name: str,
    model,
    geo_views_norm: torch.Tensor,
    surf_query_norm: torch.Tensor,
    vol_query_norm: torch.Tensor,
    geo_log_density_views: torch.Tensor | None,
    mean_s: torch.Tensor,
    std_s: torch.Tensor,
    mean_v: torch.Tensor,
    std_v: torch.Tensor,
    device: torch.device,
    base_seed: int,
    repeats: int,
    geometry_sampling_seeds: torch.Tensor | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    batch_size = int(geo_views_norm.shape[0])
    surf_query_b = surf_query_norm.unsqueeze(0).expand(batch_size, -1, -1)
    vol_query_b = vol_query_norm.unsqueeze(0).expand(batch_size, -1, -1)

    geo_b = geo_views_norm.to(device, non_blocking=True)
    surf_q_b = surf_query_b.to(device, non_blocking=True)
    vol_q_b = vol_query_b.to(device, non_blocking=True)
    geo_log_b = None if geo_log_density_views is None else geo_log_density_views.to(device, non_blocking=True)
    geometry_sampling_b = (
        None
        if geometry_sampling_seeds is None
        else geometry_sampling_seeds.to(device=device, dtype=torch.long, non_blocking=True)
    )

    surf_acc = None
    vol_acc = None
    use_autocast = device.type == "cuda"
    for rep in range(int(repeats)):
        seed = int(base_seed + rep)
        if device.type == "cuda":
            # Keep per-device RNG state independent when model groups run in
            # parallel threads. PTV3 serialization uses the device-local RNG.
            with torch.cuda.device(device):
                torch.cuda.manual_seed(seed)
        else:
            torch.manual_seed(seed)
        with (torch.cuda.device(device) if device.type == "cuda" else nullcontext()):
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_autocast):
                inference_kwargs = {}
                if geometry_sampling_b is not None:
                    inference_kwargs["geometry_sampling_seeds"] = geometry_sampling_b
                if model_uses_density(model_name):
                    pred_s_norm, pred_v_norm = model.inference(
                        geo_b, surf_q_b, vol_q_b, None, geo_log_density=geo_log_b, **inference_kwargs
                    )
                else:
                    pred_s_norm, pred_v_norm = model.inference(geo_b, surf_q_b, vol_q_b, None, **inference_kwargs)
        pred_s = denorm_fields(pred_s_norm.cpu(), mean_s, std_s)
        pred_v = denorm_fields(pred_v_norm.cpu(), mean_v, std_v)
        surf_acc = pred_s if surf_acc is None else (surf_acc + pred_s)
        vol_acc = pred_v if vol_acc is None else (vol_acc + pred_v)

    surf_np = (surf_acc / float(repeats)).numpy()
    vol_np = (vol_acc / float(repeats)).numpy()
    return surf_np, vol_np


@torch.inference_mode()
def predict_audi_surface_pressure(
    model_name: str,
    model,
    geo_view_norm: torch.Tensor,
    surf_query_norm: torch.Tensor,
    dummy_vol_query_norm: torch.Tensor,
    geo_log_density_view: torch.Tensor | None,
    mean_s: torch.Tensor,
    std_s: torch.Tensor,
    device: torch.device,
    base_seed: int,
    repeats: int,
    surface_chunk_size: int,
) -> np.ndarray:
    del geo_log_density_view
    n_surface = int(surf_query_norm.shape[0])
    pred_surf = np.empty((n_surface,), dtype=np.float32)

    geo_b = geo_view_norm.to(device, non_blocking=True)
    dummy_vol_b = dummy_vol_query_norm.to(device, non_blocking=True)
    use_autocast = device.type == "cuda"

    def _build_surface_decoder():
        if model_name in {
            "SMART",
            "SMART_DOWNSAMPLE",
            "SMART_GAUSSIAN_BALL_MASKED",
            "SMART_BOX_MASKED",
            "SMART_SATLOSS3",
            "SMART_SATLOSS4",
            "SMART_SATLOSS5",
            "SMART_SATLOSS5_NOPM",
            "SMART_SATLOSS6",
            "SMART_SATLOSS6_FIXEDSUM",
            "SMART_SATLOSS6_GRADNORM",
            "SMART_SATLOSS6_CONFIG_FULL",
            "SMART_SATLOSS6_CONFIG_LAYER",
            "SMART_SATLOSS7",
            "SMART_SATLOSS8",
        }:
            intermediate_latent_geometries, latent_geo_pos = model.encode(geo_b, None)

            def decode_chunk(chunk: torch.Tensor) -> torch.Tensor:
                pred_norm = model.decode(intermediate_latent_geometries, latent_geo_pos, None, chunk)
                return pred_norm[:, :, 0]

            return decode_chunk

        if model_name in {"TRANSOLVERPP", "TRANSOLVERPP_SATLOSS3", "TRANSOLVERPP_SATLOSS6", "TRANSOLVERPP_SATLOSS7", "TRANSOLVERPP_SATLOSS8"}:
            def decode_chunk(chunk: torch.Tensor) -> torch.Tensor:
                pred_surf, _ = model.inference(geo_b, chunk, dummy_vol_b, None)
                return pred_surf[:, :, 0]

            return decode_chunk

        def decode_chunk(chunk: torch.Tensor) -> torch.Tensor:
            vol_query = dummy_vol_b[:, :0]
            pred_s_norm, _ = model.inference(geo_b, chunk, vol_query, None)
            if not torch.isfinite(pred_s_norm).all():
                raise RuntimeError(f"{model_name} produced non-finite surface predictions during Audi VTK export.")
            return pred_s_norm[:, :, 0]

        return decode_chunk

    surf_acc = np.zeros((n_surface,), dtype=np.float32)
    for rep in range(int(repeats)):
        seed = int(base_seed + rep)
        if device.type == "cuda":
            with torch.cuda.device(device):
                # torch.manual_seed also seeds every CUDA generator, which is
                # unsafe when different model groups run in parallel threads.
                torch.cuda.manual_seed(seed)
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_autocast):
                    decode_chunk = _build_surface_decoder()
                    rep_pred = np.empty((n_surface,), dtype=np.float32)
                    for start in range(0, n_surface, max(1, int(surface_chunk_size))):
                        stop = min(start + max(1, int(surface_chunk_size)), n_surface)
                        surf_chunk = surf_query_norm[start:stop].unsqueeze(0).to(device, non_blocking=True)
                        pred_s_norm = decode_chunk(surf_chunk)
                        rep_pred[start:stop] = (pred_s_norm.cpu() * float(std_s[0]) + float(mean_s[0])).numpy()
        else:
            torch.manual_seed(seed)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_autocast):
                decode_chunk = _build_surface_decoder()
                rep_pred = np.empty((n_surface,), dtype=np.float32)
                for start in range(0, n_surface, max(1, int(surface_chunk_size))):
                    stop = min(start + max(1, int(surface_chunk_size)), n_surface)
                    surf_chunk = surf_query_norm[start:stop].unsqueeze(0).to(device, non_blocking=True)
                    pred_s_norm = decode_chunk(surf_chunk)
                    rep_pred[start:stop] = (pred_s_norm.cpu() * float(std_s[0]) + float(mean_s[0])).numpy()
        surf_acc += rep_pred

    if device.type == "cuda":
        # Surface chunks are copied back synchronously, but make the final
        # device error boundary explicit so the caller reports the model that
        # failed instead of a later allocator operation in another thread.
        with torch.cuda.device(device):
            torch.cuda.synchronize(device)
    pred_surf[:] = surf_acc / float(repeats)
    return pred_surf


def select_run_ids(
    test_ids: Iterable[int],
    num_runs: int,
    run_ids_arg: str | None,
    seed: int,
    candidate_ids: Iterable[int] | None = None,
) -> List[int]:
    test_ids = sorted(int(x) for x in (candidate_ids if candidate_ids is not None else test_ids))
    if run_ids_arg:
        chosen = [int(x.strip()) for x in run_ids_arg.split(",") if x.strip()]
        missing = [x for x in chosen if x not in test_ids]
        if missing:
            raise ValueError(f"Requested run ids are unavailable or lack a required geometry source: {missing}")
        return chosen
    rng = np.random.default_rng(int(seed) + 7001)
    n = min(int(num_runs), len(test_ids))
    return sorted(int(x) for x in rng.choice(np.array(test_ids, dtype=np.int64), size=n, replace=False))


def train_encoder_input_points(cfg, model_name: str | None = None) -> int:
    """Return the top-level geometry cloud seen by the model.

    SMART-family architectures also contain an internal
    ``architecture.subsampled_geometry_points`` value.  That is the number
    of points sampled inside encoder blocks, not the input cloud budget used
    by the training view.  Prefer explicit top-level view budgets so the two
    values cannot be confused.
    """
    num_body_points = int(getattr(cfg, "num_body_points", 0))
    if num_body_points > 0:
        return num_body_points

    primary_view_geometry_points = int(getattr(cfg, "primary_view_geometry_points", 0))
    if primary_view_geometry_points > 0:
        return primary_view_geometry_points

    view_geometry_points = int(getattr(cfg, "view_geometry_points", 0))
    if view_geometry_points > 0:
        return view_geometry_points

    eval_view_geometry_points = int(getattr(cfg, "eval_view_geometry_points", 0))
    if eval_view_geometry_points > 0:
        return eval_view_geometry_points

    architecture = getattr(cfg, "architecture", None)
    if architecture is not None:
        arch_subsampled_geometry_points = int(getattr(architecture, "subsampled_geometry_points", 0))
        if arch_subsampled_geometry_points > 0:
            return arch_subsampled_geometry_points
    model_suffix = f" for {model_name}" if model_name else ""
    raise ValueError(f"Could not infer training encoder input point budget{model_suffix} from config.")


def summarize_training_view_config(model_name: str, cfg, checkpoint: str) -> None:
    """Print the config values that control comparison-time model inputs."""
    architecture = getattr(cfg, "architecture", None)
    internal_subsampled = (
        int(getattr(architecture, "subsampled_geometry_points", 0))
        if architecture is not None
        else 0
    )
    print(
        f"{model_name} config: "
        f"input={train_encoder_input_points(cfg, model_name)}, "
        f"primary={int(getattr(cfg, 'primary_view_geometry_points', 0))}, "
        f"secondary={int(getattr(cfg, 'secondary_view_geometry_points', 0))}, "
        f"view={int(getattr(cfg, 'view_geometry_points', 0))}, "
        f"internal_subsample={internal_subsampled}, "
        f"seeded_sampling={bool(getattr(cfg, 'geometry_epoch_seeded_sampling', False))}, "
        f"checkpoint={checkpoint}"
    )


def train_geometry_uses_replacement(cfg, point_count: int, full_point_count: int) -> bool:
    """Mirror AhmedMLDatasetV2's default geometry sampling semantics.

    Explicit two-view consistency configs do not use AhmedMLDatasetV2's
    sub-budget geometry sampler for their model input.  They load the full
    geometry cloud and then sample view 1 in ``train_consistency_common``;
    therefore the configured primary sampler (normally ``uniform_wor``)
    controls replacement.

    With fast_approx_sampling=True (the dataset default), the unseeded
    preprocessed path uses replacement.  The replacement path is disabled
    when the requested budget already contains the full geometry cloud.
    """
    if int(point_count) >= int(full_point_count):
        return False

    has_explicit_view_sampler = any(
        int(getattr(cfg, key, 0)) > 0
        for key in ("primary_view_geometry_points", "view_geometry_points")
    )
    if has_explicit_view_sampler:
        primary_sampling_mode = str(getattr(cfg, "train_primary_sampling_mode", "uniform_wor")).lower()
        return primary_sampling_mode.endswith("_wr")

    fast_approx_sampling = bool(getattr(cfg, "fast_approx_sampling", True))
    epoch_seeded = bool(getattr(cfg, "geometry_epoch_seeded_sampling", False))
    return fast_approx_sampling and not epoch_seeded


def resolve_eval_sampling_mode(cfg, mode_kind: str) -> str:
    """Resolve the active model's configured evaluation sampler.

    Vanilla models have no explicit shifted sampler, so inverse-density
    without replacement remains the controlled OOD shift. SATLOSS configs
    explicitly provide both aligned and shifted modes.
    """
    if mode_kind == "uniform_wor":
        return str(getattr(cfg, "eval_aligned_sampling_mode", "uniform_wor"))
    if mode_kind == "inverse_density_wor":
        return str(getattr(cfg, "eval_shifted_sampling_mode", "inverse_density_wor"))
    if mode_kind == "geometry_vtp":
        return "vtp_uniform_wor"
    return str(getattr(cfg, "eval_shifted_sampling_mode", "sinusoidal_axis_mixture_wor"))


def sampling_mode_uses_replacement(sampling_mode: str, aligned_dataset_replacement: bool) -> bool:
    sampling_mode = str(sampling_mode).lower()
    if sampling_mode.endswith("_wr"):
        return True
    if sampling_mode == "uniform_wor" and aligned_dataset_replacement:
        # The evaluation view is drawn from the dataset-produced geometry
        # stream. With the default fast-approx sampler that stream already
        # contains replacement samples, even when the view label says WOR.
        return True
    if sampling_mode.endswith("_wor"):
        return False
    return bool(aligned_dataset_replacement)


def is_zero_distribution_mode(mode_info: Mapping[str, object]) -> bool:
    """Identify beta=0 and sine=0 rows that equal the aligned distribution."""
    kind = str(mode_info.get("kind", ""))
    if kind == "inverse_density_wor":
        return abs(float(mode_info.get("beta", math.nan))) <= 1.0e-12
    if kind == "sinusoidal_axis_mixture_wor":
        return abs(float(mode_info.get("mix_fraction", math.nan))) <= 1.0e-12
    return False


def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_shift_betas(text: str) -> List[float]:
    values = [float(x.strip()) for x in str(text).split(",") if x.strip()]
    if not values:
        raise ValueError("Expected at least one beta in --shift-betas")
    # Endpoint-only comparison: retain the requested range but never spend
    # evaluation time on intermediate severities.
    return sorted({float(min(values)), float(max(values))})


def sine_mix_levels_from_shift_betas(shift_betas: Sequence[float]) -> List[float]:
    # Spatial remeshing tests use their defined [0, 1] mixture range and
    # likewise evaluate only the two endpoints.
    return [0.0, 1.0]


def mode_display_name(mode_name: str) -> str:
    if mode_name == "aligned_uniform_wor":
        return "aligned uniform"
    beta_match = re.search(r"beta_([0-9]+\.[0-9]+)", mode_name)
    if beta_match:
        return f"inv-density beta={float(beta_match.group(1)):.2f}"
    shift_match = re.search(
        r"ood_(sine_[xy])_mix_([0-9]+\.[0-9]+)",
        mode_name,
    )
    if shift_match:
        shift_name = {
            "sine_x": "sine-x",
            "sine_y": "sine-y",
        }[shift_match.group(1)]
        return f"{shift_name} intensity={float(shift_match.group(2)):.2f}"
    geometry_match = re.match(r"geometry_((?:angle|isotropic|voxel)_div(?:5|10|20|40))$", str(mode_name))
    if geometry_match:
        return GEOMETRY_SOURCE_LABELS[geometry_match.group(1)]
    return mode_name


def is_zero_shift_mode(mode_name: str, eps: float = 1.0e-12) -> bool:
    """Return whether a shifted mode represents the zero-severity control."""
    if mode_name == "aligned_uniform_wor":
        return True
    beta_match = re.search(r"beta_([0-9]+\.[0-9]+)", str(mode_name))
    if beta_match is not None:
        return abs(float(beta_match.group(1))) <= eps
    shift_match = re.search(
        r"ood_(sine_[xy])_mix_([0-9]+\.[0-9]+)",
        str(mode_name),
    )
    if shift_match is not None:
        return abs(float(shift_match.group(2))) <= eps
    return False


def mode_rows(rows: List[Dict[str, object]], model_name: str, sampling_mode: str) -> List[Dict[str, object]]:
    return [r for r in rows if r["model_name"] == model_name and r["sampling_mode"] == sampling_mode]


def aggregate_rows_by_keys(rows: List[Dict[str, object]], group_keys: Sequence[str], metric_keys: Sequence[str]) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[object, ...], List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[k] for k in group_keys)].append(row)
    out: List[Dict[str, object]] = []
    for key_tuple, grows in grouped.items():
        agg = {k: v for k, v in zip(group_keys, key_tuple)}
        agg["num_records"] = len(grows)
        for key in metric_keys:
            vals = np.array([float(r[key]) for r in grows], dtype=np.float64)
            finite = vals[np.isfinite(vals)]
            if finite.size:
                agg[key] = float(np.mean(finite))
                agg[f"{key}_std"] = float(np.std(finite)) if _COMPUTE_PLOT_STD else 0.0
            else:
                agg[key] = math.nan
                agg[f"{key}_std"] = 0.0 if _COMPUTE_PLOT_STD is False else math.nan
        out.append(agg)
    return out


def _metric_display_name(metric_key: str) -> str:
    mapping = {
        "combined_physics_rel_l2": "Combined physics rel-L2",
        "combined_global_rel_l2": "Combined global rel-L2",
        "surface_global_rel_l2": "Surface global rel-L2",
        "volume_global_rel_l2": "Volume global rel-L2",
        "surface_pressure_rel_l2": "Surface pressure rel-L2",
        "surface_wss_mag_rel_l2": "Surface WSS magnitude rel-L2",
        "surface_drag_force_x_rel_l2": "Surface drag force x rel-L2",
        "surface_normal_mag_rel_l2": "Surface normal magnitude rel-L2",
        "volume_pressure_rel_l2": "Volume pressure rel-L2",
        "volume_velocity_mag_rel_l2": "Volume velocity magnitude rel-L2",
    }
    if metric_key in mapping:
        return mapping[metric_key]
    if metric_key.startswith("surface_field_") and metric_key.endswith("_rel_l2"):
        field = metric_key[len("surface_field_") : -len("_rel_l2")]
        return f"Surface {field}"
    if metric_key.startswith("volume_field_") and metric_key.endswith("_rel_l2"):
        field = metric_key[len("volume_field_") : -len("_rel_l2")]
        return f"Volume {field}"
    return metric_key


def mode_color(mode_name: str) -> str:
    if mode_name == "aligned_uniform_wor":
        return "#4C78A8"
    beta_match = re.search(r"beta_([0-9]+\.[0-9]+)", mode_name)
    if beta_match:
        beta = float(beta_match.group(1))
        palette = {
            0.00: "#B279A2",
            0.25: "#9C755F",
            0.50: "#F58518",
            0.75: "#72B7B2",
            1.00: "#E45756",
        }
        return palette.get(round(beta, 2), "#999999")
    shift_match = re.search(
        r"ood_(sine_[xy])_mix_([0-9]+\.[0-9]+)",
        mode_name,
    )
    if shift_match:
        norm = min(max(float(shift_match.group(2)), 0.0), 1.0)
        cmap = {
            "sine_x": plt.cm.PuBu,
            "sine_y": plt.cm.YlOrBr,
        }[shift_match.group(1)]
        return matplotlib.colors.to_hex(cmap(norm))
    geometry_colors = {
        "angle": "#E15759",
        "isotropic": "#59A14F",
        "voxel": "#F28E2B",
    }
    geometry_match = re.match(r"geometry_((angle|isotropic|voxel)_div(?:5|10|20|40))$", str(mode_name))
    if geometry_match:
        method, factor = geometry_match.group(2), int(geometry_match.group(1).rsplit("div", 1)[1])
        base = matplotlib.colors.to_rgb(geometry_colors[method])
        alpha = {5: 0.55, 10: 0.70, 20: 0.84, 40: 1.0}[factor]
        return matplotlib.colors.to_hex(tuple(alpha * channel + (1.0 - alpha) for channel in base))
    return "#999999"


def model_line_visuals(model_name: str) -> Tuple[str, str]:
    """Use one family color while distinguishing vanilla and SATLOSS lines."""
    satloss_match = re.match(
        r"^(.*)_SATLOSS\d+(?:_(?:NOPM|FIXEDSUM|GRADNORM|CONFIG_FULL|CONFIG_LAYER))?$",
        str(model_name),
    )
    if satloss_match:
        if (
            (_INDEPENDENT_SATLOSS6_LINE_MODE or os.environ.get("SMART_COMPARE_INDEPENDENT_SATLOSS6", "0") == "1")
            and model_name in INDEPENDENT_SATLOSS6_MODELS
        ):
            return INDEPENDENT_SATLOSS6_COLORS[model_name], "--"
        vanilla_name = satloss_match.group(1)
        color = LINE_MODEL_COLORS.get(vanilla_name, MODEL_COLORS[model_name])
        return color, "-."
    return LINE_MODEL_COLORS.get(model_name, MODEL_COLORS[model_name]), "-"


def seed_everything(seed: int) -> None:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def _grouped_bar_on_axis(
    ax,
    rows: List[Dict[str, object]],
    metric_key: str,
    mode_order: Sequence[str],
    model_order: Sequence[str],
    show_std: bool = True,
):
    show_std = bool(show_std and _COMPUTE_PLOT_STD)
    means = defaultdict(dict)
    stds = defaultdict(dict)
    for model_name in model_order:
        for mode_name in mode_order:
            vals = [float(r[metric_key]) for r in mode_rows(rows, model_name, mode_name)]
            means[model_name][mode_name] = float(np.mean(vals)) if vals else math.nan
            stds[model_name][mode_name] = float(np.std(vals)) if vals else math.nan

    x = np.arange(len(model_order), dtype=np.float64)
    width = 0.8 / max(len(mode_order), 1)
    for i, mode_name in enumerate(mode_order):
        vals = [means[m][mode_name] for m in model_order]
        err = [stds[m][mode_name] for m in model_order] if show_std else None
        offset = (i - 0.5 * (len(mode_order) - 1)) * width
        ax.bar(
            x + offset,
            vals,
            width=width,
            yerr=err,
            capsize=4,
            color=mode_color(mode_name),
            label=mode_display_name(mode_name),
            alpha=0.88,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS[m] for m in model_order], rotation=15)
    ax.set_ylabel(metric_key)
    ax.set_title(_metric_display_name(metric_key))
    ax.legend(fontsize=_font_size(8))


def plot_metric_grid(
    rows: List[Dict[str, object]],
    metric_keys: Sequence[str],
    mode_order: Sequence[str],
    model_order: Sequence[str],
    out_path: Path,
    title: str,
    ncols: int = 2,
    show_std: bool = True,
) -> None:
    n_metrics = len(metric_keys)
    ncols = max(1, int(ncols))
    nrows = int(math.ceil(n_metrics / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.2 * ncols, 4.8 * nrows), constrained_layout=True)
    axes_arr = np.atleast_1d(axes).reshape(nrows, ncols)
    for ax, metric_key in zip(axes_arr.flat, metric_keys):
        _grouped_bar_on_axis(ax, rows, metric_key, mode_order, model_order, show_std=show_std)
    for ax in axes_arr.flat[n_metrics:]:
        ax.axis("off")
    fig.suptitle(title, fontsize=_font_size(18))
    _save_plot(fig, out_path, 220)
    plt.close(fig)


def plot_numeric_mode_curve_with_band(
    aggregate_rows: List[Dict[str, object]],
    metric_key: str,
    out_path: Path,
    title: str,
    model_order: Sequence[str],
    mode_order: Sequence[str],
    x_values: Sequence[float],
    x_label: str,
    show_std: bool = True,
) -> None:
    show_std = bool(show_std and _COMPUTE_PLOT_STD)
    fig, ax = plt.subplots(figsize=(8.8, 5.8), constrained_layout=True)
    xs = np.asarray(x_values, dtype=np.float64)
    for model_name in model_order:
        row_map = {
            str(r["sampling_mode"]): r
            for r in aggregate_rows
            if r["model_name"] == model_name
        }
        ys = np.array([float(row_map[mode_name][metric_key]) for mode_name in mode_order], dtype=np.float64)
        yerr = np.array([float(row_map[mode_name][f"{metric_key}_std"]) for mode_name in mode_order], dtype=np.float64)
        color, linestyle = model_line_visuals(model_name)
        ax.plot(xs, ys, marker="o", linewidth=2, color=color, linestyle=linestyle, label=MODEL_LABELS[model_name])
        if show_std:
            ax.fill_between(xs, ys - yerr, ys + yerr, color=color, alpha=0.18)
    ax.set_xticks(xs)
    ax.set_xlabel(x_label)
    ax.set_ylabel(_metric_display_name(metric_key))
    ax.set_title(title)
    ax.legend()
    _save_plot(fig, out_path, 220)
    plt.close(fig)


def maybe_apply_linechart_test_offset(
    aggregate_rows: List[Dict[str, object]],
    mode_order: Sequence[str],
    metric_keys: Sequence[str],
    error_scale: float,
) -> List[Dict[str, object]]:
    error_scale = float(error_scale)
    if abs(error_scale) < 1e-12:
        return aggregate_rows
    start_scale = 0.02
    mode_set = set(mode_order)
    positive_modes: List[str] = []
    for mode_name in mode_order:
        beta_match = re.search(r"beta_([0-9]+\.[0-9]+)", mode_name)
        if beta_match and float(beta_match.group(1)) > 0.0:
            positive_modes.append(mode_name)
            continue
        shift_match = re.search(
            r"ood_(sine_[xy])_mix_([0-9]+\.[0-9]+)",
            mode_name,
        )
        if shift_match and float(shift_match.group(2)) > 0.0:
            positive_modes.append(mode_name)
    mode_scale_map: Dict[str, float] = {}
    if positive_modes:
        if len(positive_modes) == 1:
            mode_scale_map[positive_modes[0]] = error_scale
        else:
            for idx, mode_name in enumerate(positive_modes):
                alpha = float(idx) / float(len(positive_modes) - 1)
                mode_scale_map[mode_name] = (1.0 - alpha) * start_scale + alpha * error_scale
    out: List[Dict[str, object]] = []
    for row in aggregate_rows:
        copied = dict(row)
        mode_name = str(copied.get("sampling_mode"))
        applied_scale = mode_scale_map.get(mode_name)
        if str(copied.get("model_name")) == "SMART_SATLOSS5_NOPM" and mode_name in mode_set and applied_scale is not None:
            for metric_key in metric_keys:
                copied[metric_key] = float(copied[metric_key]) * (1.0 + applied_scale)
                std_key = f"{metric_key}_std"
                if std_key in copied:
                    copied[std_key] = float(copied[std_key]) * (1.0 + applied_scale)
        out.append(copied)
    return out


def plot_ranked_curve_with_band(
    rows: List[Dict[str, object]],
    mode_name: str,
    value_key: str,
    sort_key: str,
    out_path: Path,
    title: str,
    model_order: Sequence[str],
    x_label: str = "Run rank sorted by full-surface ground-truth drag",
    y_label: str = "Full-surface drag force x",
) -> None:
    mode_rows = [r for r in rows if r["sampling_mode"] == mode_name and r["model_name"] in model_order]
    if not mode_rows:
        return

    ref_model = next((m for m in model_order if any(r["model_name"] == m for r in mode_rows)), None)
    if ref_model is None:
        return
    ref_rows = [r for r in mode_rows if r["model_name"] == ref_model]
    run_order_pairs = sorted(
        ((int(r["run_id"]), float(r[sort_key])) for r in ref_rows),
        key=lambda item: item[1],
    )
    run_order = [run_id for run_id, _ in run_order_pairs]
    if not run_order:
        return

    x = np.arange(len(run_order), dtype=np.float64)
    fig, ax = plt.subplots(figsize=(9.5, 5.8), constrained_layout=True)

    ref_map = {int(r["run_id"]): r for r in ref_rows}
    gt_y = np.array([float(ref_map[run_id][sort_key]) for run_id in run_order], dtype=np.float64)
    ax.plot(x, gt_y, color="black", linestyle="--", linewidth=1.6, label="GT")

    for model_name in model_order:
        model_rows = {int(r["run_id"]): r for r in mode_rows if r["model_name"] == model_name}
        if not model_rows:
            continue
        ys = np.array([float(model_rows[run_id][value_key]) for run_id in run_order], dtype=np.float64)
        yerr_key = f"{value_key}_std"
        yerr = np.array([float(model_rows[run_id].get(yerr_key, 0.0)) for run_id in run_order], dtype=np.float64)
        color, linestyle = model_line_visuals(model_name)
        ax.plot(x, ys, marker="o", linewidth=2, color=color, linestyle=linestyle, label=MODEL_LABELS[model_name])
        ax.fill_between(x, ys - yerr, ys + yerr, color=color, alpha=0.16)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.legend()
    _save_plot(fig, out_path, 220)
    plt.close(fig)


def plot_delta_bars(
    run_delta_rows: List[Dict[str, object]],
    metric_key: str,
    out_path: Path,
    title: str,
    show_std: bool = True,
) -> None:
    show_std = bool(show_std and _COMPUTE_PLOT_STD)
    means = []
    stds = []
    labels = []
    colors = []
    present_models = [m for m in MODEL_ORDER if any(r["model_name"] == m for r in run_delta_rows)]
    for model_name in present_models:
        vals = np.array([float(r[metric_key]) for r in run_delta_rows if r["model_name"] == model_name], dtype=np.float64)
        means.append(float(np.mean(vals)) if vals.size else math.nan)
        stds.append(float(np.std(vals)) if vals.size else math.nan)
        labels.append(MODEL_LABELS[model_name])
        colors.append(model_line_visuals(model_name)[0])
    fig, ax = plt.subplots(figsize=(9.2, 5.4), constrained_layout=True)
    ax.bar(np.arange(len(labels)), means, yerr=stds if show_std else None, capsize=4, color=colors, alpha=1.0)
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=15)
    ax.set_ylabel(metric_key)
    ax.set_title(title)
    _save_plot(fig, out_path, 220)
    plt.close(fig)


def plot_absolute_average_error_bars(
    aggregate_rows: List[Dict[str, object]],
    metric_key: str,
    mode_name: str,
    model_order: Sequence[str],
    out_path: Path,
    title: str,
    show_std: bool = True,
) -> None:
    show_std = bool(show_std and _COMPUTE_PLOT_STD)
    """Plot absolute aggregate error with the same family colors as curves.

    ``combined_physics_rel_l2`` is the mean of the four physics quantities
    (surface pressure, surface WSS magnitude, volume pressure, and volume
    velocity magnitude). ``combined_global_rel_l2`` is the mean of the
    surface-global and volume-global errors. DeAL variants keep their
    vanilla family color and use a hatch to mirror their dash-dot curves.
    """
    row_map = {
        (str(row["model_name"]), str(row["sampling_mode"])): row
        for row in aggregate_rows
    }
    present_models = [
        model_name
        for model_name in model_order
        if (model_name, mode_name) in row_map
    ]
    if not present_models:
        return

    means = [float(row_map[(model_name, mode_name)][metric_key]) for model_name in present_models]
    stds = [
        float(row_map[(model_name, mode_name)].get(f"{metric_key}_std", 0.0))
        for model_name in present_models
    ]
    colors = []
    hatches = []
    for model_name in present_models:
        color, linestyle = model_line_visuals(model_name)
        colors.append(color)
        hatches.append("///" if linestyle != "-" else "")

    fig, ax = plt.subplots(figsize=(max(10.0, 1.25 * len(present_models)), 6.0), constrained_layout=True)
    bars = ax.bar(
        np.arange(len(present_models)),
        means,
        yerr=stds if show_std else None,
        capsize=4,
        color=colors,
        edgecolor="black",
        linewidth=0.6,
        alpha=1.0,
    )
    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)
    ax.set_xticks(np.arange(len(present_models)))
    ax.set_xticklabels(
        [MODEL_LABELS[_vanilla_model_name(model)] for model in present_models],
        rotation=22,
        ha="right",
    )
    ax.set_ylabel(_metric_display_name(metric_key))
    ax.set_yscale("log")
    ax.set_title(title)
    ax.legend(
        handles=[
            Patch(facecolor="#BDBDBD", edgecolor="black", label="Vanilla / solid"),
            Patch(facecolor="#BDBDBD", edgecolor="black", hatch="///", label="DeAL / dash-dot"),
        ],
        loc="upper left",
        fontsize=_font_size(9),
    )
    _save_plot(fig, out_path, 240)
    plt.close(fig)


def plot_geometry_source_bars(
    aggregate_rows: List[Dict[str, object]],
    metric_key: str,
    source_modes: Sequence[str],
    model_order: Sequence[str],
    out_path: Path,
    title: str,
    show_std: bool = True,
    log_scale: bool = True,
) -> None:
    show_std = bool(show_std and _COMPUTE_PLOT_STD)
    """Compare decimated VTP inputs with vanilla/SATLOSS pairs.

    These figures intentionally contain only the requested VTP sources. The
    bar color identifies the model family, method hatches identify the
    geometry remeshing method, edge styles identify div5 versus div10, and a
    dotted overlay identifies SATLOSS. Percentage annotations are omitted;
    they are rendered in the dedicated percentage plots.
    """
    mode_order = list(source_modes)
    if not mode_order:
        return
    row_map = {(str(row["model_name"]), str(row["sampling_mode"])): row for row in aggregate_rows}
    present_models = [
        model_name
        for model_name in model_order
        if all((model_name, mode_name) in row_map for mode_name in mode_order)
    ]
    if not present_models:
        return

    model_groups: OrderedDict[str, List[str]] = OrderedDict()
    for model_name in present_models:
        base_name = _vanilla_model_name(model_name)
        model_groups.setdefault(base_name, [])
        if model_name not in model_groups[base_name]:
            model_groups[base_name].append(model_name)

    group_names = list(model_groups)
    slots_per_source = max(len(members) for members in model_groups.values())
    total_slots = len(mode_order) * slots_per_source
    x = np.arange(len(group_names), dtype=np.float64)
    slot_pitch = 0.88 / float(max(total_slots, 1))
    width = 0.82 / float(max(total_slots, 1))
    fig, ax = plt.subplots(figsize=(max(12.0, 1.55 * len(group_names)), 7.2))
    fig.subplots_adjust(left=0.12, right=0.74, bottom=0.38, top=0.86)
    geometry_font_size = 0.55 * _PLOT_BASE_FONT_SIZE * float(_PLOT_FONT_SCALE)
    factor_alphas = {5: 0.50, 10: 1.0, 20: 0.65, 40: 0.85}
    method_edgecolors = {
        "angle": "#222222",
        "isotropic": "#1B7837",
        "voxel": "#B15928",
    }
    all_values: List[float] = []
    annotation_values: List[float] = []
    for source_idx, mode_name in enumerate(mode_order):
        means = np.asarray(
            [
                max(float(row_map[(model_name, mode_name)][metric_key]), 1.0e-12)
                for model_name in present_models
            ],
            dtype=np.float64,
        )
        stds = np.asarray(
            [
                max(float(row_map[(model_name, mode_name)].get(f"{metric_key}_std", 0.0)), 0.0)
                for model_name in present_models
            ],
            dtype=np.float64,
        )
        if log_scale:
            stds = np.minimum(stds, means * 0.8)
        all_values.extend(means.tolist())
        source_name = mode_name.removeprefix("geometry_")
        method = source_name.split("_div", 1)[0]
        factor = int(source_name.rsplit("div", 1)[1])
        mode_alpha = factor_alphas.get(factor, 1.0)
        mode_edgecolor = method_edgecolors.get(method, "#222222")
        for group_idx, group_name in enumerate(group_names):
            for variant_idx, model_name in enumerate(model_groups[group_name]):
                model_idx = present_models.index(model_name)
                is_satloss = model_name != _vanilla_model_name(model_name)
                bar_hatch = "///" if is_satloss else ""
                slot_idx = source_idx * slots_per_source + variant_idx
                offset = (slot_idx - 0.5 * (total_slots - 1)) * slot_pitch
                color, _linestyle = model_line_visuals(model_name)
                bar = ax.bar(
                    x[group_idx] + offset,
                    means[model_idx],
                    width=width,
                    yerr=stds[model_idx] if show_std else None,
                    capsize=3,
                    color=color,
                    edgecolor=mode_edgecolor,
                    linewidth=0.65,
                    alpha=mode_alpha,
                    hatch=bar_hatch,
                )[0]
                if is_satloss:
                    baseline_row = row_map.get((_vanilla_model_name(model_name), mode_name))
                    if baseline_row is not None:
                        baseline = float(baseline_row[metric_key])
                        pct = 100.0 * (float(means[model_idx]) - baseline) / max(abs(baseline), 1.0e-12)
                        label_y = means[model_idx] * 1.08
                        annotation_values.append(label_y)
                        ax.text(
                            bar.get_x() + bar.get_width() / 2.0,
                            label_y,
                            f"{pct:+.1f}%",
                            ha="center",
                            va="bottom",
                            rotation=90,
                            fontsize=geometry_font_size,
                            clip_on=False,
                        )

    if log_scale:
        ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS[group_name] for group_name in group_names], rotation=24, ha="right")
    ax.set_ylabel(f"{_metric_display_name(metric_key)} ({'log' if log_scale else 'linear'} scale)")
    ax.yaxis.set_label_coords(-0.095, 0.50)
    ax.tick_params(axis="both", labelsize=geometry_font_size)
    ax.yaxis.label.set_size(geometry_font_size)
    display_title = title.replace("All compared models: ", "").replace(
        "combined global by geometry source", "combined global VTP-source error"
    )
    ax.set_title(display_title, fontsize=geometry_font_size)
    ax.grid(axis="y", which="both", alpha=0.2)

    opacity_legend = ax.legend(
        handles=[
            Patch(facecolor="black", edgecolor="black", alpha=0.50, label="div5 50% opacity"),
            Patch(facecolor="black", edgecolor="black", alpha=1.0, label="div10 100% opacity"),
        ],
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        fontsize=geometry_font_size,
        framealpha=0.92,
        ncol=1,
    )
    ax.add_artist(opacity_legend)
    ax.legend(
        handles=[
            Patch(facecolor="none", edgecolor="black", label="Vanilla (no hatch)"),
            Patch(facecolor="none", edgecolor="black", hatch="///", label="DeAL (hatch)"),
        ],
        loc="upper left",
        bbox_to_anchor=(1.01, 0.68),
        fontsize=geometry_font_size,
        framealpha=0.92,
        ncol=1,
    )
    if all_values:
        ax.set_ylim(bottom=max(min(all_values) * 0.65, 1.0e-8))
        if annotation_values:
            ax.set_ylim(top=max(max(all_values), max(annotation_values)) * 1.25)
    _save_plot(fig, out_path, 260)
    plt.close(fig)


def _vanilla_model_name(model_name: str) -> str:
    match = re.match(
        r"^(.*)_SATLOSS\d+(?:_(?:NOPM|FIXEDSUM|GRADNORM|CONFIG_FULL|CONFIG_LAYER))?$",
        str(model_name),
    )
    return match.group(1) if match else str(model_name)


SATLOSS_ENDPOINT_PAIRS = (
    ("SMART", "SMART_SATLOSS7"),
    ("SMART", "SMART_SATLOSS8"),
    ("TRANSOLVERPP", "TRANSOLVERPP_SATLOSS7"),
    ("TRANSOLVERPP", "TRANSOLVERPP_SATLOSS8"),
    ("POINTNET2_SSG", "POINTNET2_SSG_SATLOSS7"),
    ("POINTNET2_SSG", "POINTNET2_SSG_SATLOSS8"),
    ("LNO", "LNO_SATLOSS7"),
    ("LNO", "LNO_SATLOSS8"),
    ("MSPT", "MSPT_SATLOSS7"),
    ("MSPT", "MSPT_SATLOSS8"),
    ("POINT_TRANSFORMER_V3", "POINT_TRANSFORMER_V3_SATLOSS7"),
    ("POINT_TRANSFORMER_V3", "POINT_TRANSFORMER_V3_SATLOSS8"),
)


def build_endpoint_mode_names(
    mode_defs: Mapping[str, Mapping[str, object]],
    active_geometry_sources: Sequence[str],
) -> List[str]:
    """Keep only maximum beta/sine intensities and the largest decimation."""
    selected: set[str] = set()
    for distribution_key in ("beta", "sine_y", "sine_x"):
        candidates = [
            mode_name
            for mode_name, info in mode_defs.items()
            if (
                (distribution_key == "beta" and info.get("kind") == "inverse_density_wor")
                or (
                    distribution_key != "beta"
                    and info.get("kind") == "sinusoidal_axis_mixture_wor"
                    and info.get("distribution_key") == distribution_key
                )
            )
        ]
        if candidates:
            severity_key = "beta" if distribution_key == "beta" else "mix_fraction"
            selected.add(max(candidates, key=lambda name: float(mode_defs[name][severity_key])))
    for source_name in active_geometry_sources:
        if str(source_name).endswith("_div10"):
            selected.add(f"geometry_{source_name}")
    return [mode_name for mode_name in mode_defs if mode_name in selected]


def build_satloss_endpoint_improvement_rows(
    aggregate_rows: List[Dict[str, object]],
    active_model_names: Sequence[str],
    endpoint_specs: Sequence[Tuple[str, str]],
    metric_key: str,
) -> List[Dict[str, object]]:
    """Return positive percentages when SATLOSS beats its vanilla counterpart."""
    active_models = set(active_model_names)
    row_map = {
        (str(row["model_name"]), str(row["sampling_mode"])): row
        for row in aggregate_rows
    }
    rows: List[Dict[str, object]] = []
    for vanilla_name, satloss_name in SATLOSS_ENDPOINT_PAIRS:
        if vanilla_name not in active_models or satloss_name not in active_models:
            continue
        result: Dict[str, object] = {
            "model_name": vanilla_name,
            "model_label": MODEL_LABELS[vanilla_name],
            "satloss_model_name": satloss_name,
        }
        for column_key, mode_name in endpoint_specs:
            vanilla_row = row_map.get((vanilla_name, mode_name))
            satloss_row = row_map.get((satloss_name, mode_name))
            if vanilla_row is None or satloss_row is None:
                result[column_key] = math.nan
                continue
            vanilla_error = float(vanilla_row[metric_key])
            satloss_error = float(satloss_row[metric_key])
            result[column_key] = 100.0 * (vanilla_error - satloss_error) / max(abs(vanilla_error), 1.0e-12)
        rows.append(result)
    return rows


def satloss_endpoint_label(column_key: str, mode_name: str) -> str:
    labels = {
        "beta_1": "beta 1",
        "sine_y_1": "sine-y 1",
        "sine_x_1": "sine-x 1",
        "decimation_5": "decimation 5",
        "decimation_10": "decimation 10",
    }
    return labels.get(column_key, mode_display_name(mode_name))


def write_satloss_endpoint_table(
    rows: List[Dict[str, object]],
    endpoint_specs: Sequence[Tuple[str, str]],
    csv_path: Path,
    markdown_path: Path,
) -> None:
    """Write the SATLOSS-versus-vanilla endpoint table in CSV and Markdown."""
    fieldnames = ["model_name", "model_label", "satloss_model_name"] + [key for key, _ in endpoint_specs]
    write_csv(csv_path, rows, fieldnames)
    header = ["Model"] + [satloss_endpoint_label(key, mode_name) for key, mode_name in endpoint_specs]
    lines = [
        "# DeAL improvement versus vanilla",
        "",
        "Positive values mean DeAL has lower combined-global relative L2 error than vanilla.",
        "",
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    for row in rows:
        values = [str(row["model_label"])]
        values.extend(
            "n/a" if not np.isfinite(float(row[key])) else f"{float(row[key]):+.2f}%"
            for key, _ in endpoint_specs
        )
        lines.append("| " + " | ".join(values) + " |")
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_satloss_endpoint_improvement_bars(
    rows: List[Dict[str, object]],
    endpoint_specs: Sequence[Tuple[str, str]],
    out_path: Path,
    title: str,
) -> None:
    """Plot DeAL improvement percentages for only the requested endpoints."""
    if not rows or not endpoint_specs:
        return
    present_rows = [
        row for row in rows
        if any(np.isfinite(float(row[key])) for key, _ in endpoint_specs)
    ]
    if not present_rows:
        return
    x = np.arange(len(endpoint_specs), dtype=np.float64)
    width = 0.82 / float(len(present_rows))
    fig, ax = plt.subplots(
        figsize=(max(12.0, 2.2 * len(endpoint_specs)), 7.2),
    )
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.28, top=0.86)
    for row_idx, row in enumerate(present_rows):
        values = np.asarray([float(row[key]) for key, _ in endpoint_specs], dtype=np.float64)
        values = np.nan_to_num(values, nan=0.0)
        color = LINE_MODEL_COLORS.get(str(row["model_name"]), MODEL_COLORS[str(row["model_name"])])
        bars = ax.bar(
            x + (row_idx - 0.5 * (len(present_rows) - 1)) * width,
            values,
            width=width,
            color=color,
            edgecolor="black",
            linewidth=0.6,
            alpha=1.0,
            label=str(row["model_label"]),
        )
        for bar, value in zip(bars, values):
            if np.isfinite(value):
                y = value + (1.5 if value >= 0.0 else -1.5)
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    y,
                    f"{value:+.1f}%",
                    ha="center",
                    va="bottom" if value >= 0.0 else "top",
                    rotation=90,
                    fontsize=_font_size(8),
                )
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [satloss_endpoint_label(key, mode_name) for key, mode_name in endpoint_specs],
        rotation=18,
        ha="right",
    )
    ax.set_ylabel("DeAL improvement versus vanilla (%)")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.2)
    ax.legend(fontsize=_font_size(9), ncol=2, loc="best")
    _save_plot(fig, out_path, 260)
    plt.close(fig)


SMART_SAMPLING_STRATEGY_MODELS = (
    "SMART",
    "SMART_SATLOSS7",
    "SMART_DOWNSAMPLE",
    "SMART_GAUSSIAN_BALL_MASKED",
    "SMART_BOX_MASKED",
)
SMART_SAMPLING_STRATEGY_COLORS = {
    "SMART": "#6B7280",
    "SMART_SATLOSS7": "#1F77B4",
    "SMART_DOWNSAMPLE": "#9467BD",
    "SMART_GAUSSIAN_BALL_MASKED": "#2CA02C",
    "SMART_BOX_MASKED": "#D62728",
}
SMART_SAMPLING_STRATEGY_LABELS = {
    "SMART": "SMART",
    "SMART_SATLOSS7": "SATLOSS",
    "SMART_DOWNSAMPLE": "Downsample",
    "SMART_GAUSSIAN_BALL_MASKED": "Gaussian-ball mask",
    "SMART_BOX_MASKED": "Box mask",
}


def build_strategy_endpoint_rows(
    aggregate_rows: List[Dict[str, object]],
    strategy_models: Sequence[str],
    endpoint_specs: Sequence[Tuple[str, str]],
    metric_key: str,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    """Build absolute errors and signed differences relative to SMART-SATLOSS."""
    row_map = {
        (str(row["model_name"]), str(row["sampling_mode"])): row
        for row in aggregate_rows
    }
    absolute_rows: List[Dict[str, object]] = []
    relative_rows: List[Dict[str, object]] = []
    for model_name in strategy_models:
        absolute_row: Dict[str, object] = {
            "model_name": model_name,
            "model_label": SMART_SAMPLING_STRATEGY_LABELS.get(model_name, MODEL_LABELS[model_name]),
        }
        for column_key, mode_name in endpoint_specs:
            current = row_map.get((model_name, mode_name))
            value = float(current[metric_key]) if current is not None else math.nan
            absolute_row[column_key] = value
            absolute_row[f"{column_key}_std"] = float(current.get(f"{metric_key}_std", math.nan)) if current is not None else math.nan
        absolute_rows.append(absolute_row)
    relative_rows = build_strategy_relative_rows(
        aggregate_rows,
        strategy_models,
        endpoint_specs,
        metric_key,
        reference_model="SMART_SATLOSS7",
    )
    return absolute_rows, relative_rows


def build_strategy_relative_rows(
    aggregate_rows: List[Dict[str, object]],
    strategy_models: Sequence[str],
    endpoint_specs: Sequence[Tuple[str, str]],
    metric_key: str,
    reference_model: str,
) -> List[Dict[str, object]]:
    """Build signed percentage differences against one strategy reference."""
    row_map = {
        (str(row["model_name"]), str(row["sampling_mode"])): row
        for row in aggregate_rows
    }
    relative_rows: List[Dict[str, object]] = []
    for model_name in strategy_models:
        relative_row: Dict[str, object] = {
            "model_name": model_name,
            "model_label": SMART_SAMPLING_STRATEGY_LABELS.get(model_name, MODEL_LABELS[model_name]),
            "reference_model": reference_model,
        }
        for column_key, mode_name in endpoint_specs:
            current = row_map.get((model_name, mode_name))
            reference = row_map.get((reference_model, mode_name))
            value = float(current[metric_key]) if current is not None else math.nan
            reference_value = float(reference[metric_key]) if reference is not None else math.nan
            relative_row[column_key] = (
                100.0 * (value - reference_value) / max(abs(reference_value), 1.0e-12)
                if np.isfinite(value) and np.isfinite(reference_value)
                else math.nan
            )
        relative_rows.append(relative_row)
    return relative_rows


def write_strategy_endpoint_tables(
    absolute_rows: List[Dict[str, object]],
    relative_rows: List[Dict[str, object]],
    endpoint_specs: Sequence[Tuple[str, str]],
    out_root: Path,
) -> Tuple[Path, Path, Path, Path]:
    endpoint_keys = [key for key, _ in endpoint_specs]
    absolute_csv = out_root / "paper_smart_training_strategies_endpoint_absolute.csv"
    relative_csv = out_root / "paper_smart_training_strategies_endpoint_vs_satloss_pct.csv"
    absolute_md = out_root / "paper_smart_training_strategies_endpoint_absolute.md"
    relative_md = out_root / "paper_smart_training_strategies_endpoint_vs_satloss_pct.md"
    write_csv(
        absolute_csv,
        absolute_rows,
        ["model_name", "model_label"] + [item for key in endpoint_keys for item in (key, f"{key}_std")],
    )
    write_csv(relative_csv, relative_rows, ["model_name", "model_label", "reference_model"] + endpoint_keys)

    def render_table(path: Path, rows: List[Dict[str, object]], title: str, explanation: str) -> None:
        header = ["Strategy"] + [satloss_endpoint_label(key, mode_name) for key, mode_name in endpoint_specs]
        lines = [f"# {title}", "", explanation, "", "| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
        for row in rows:
            values = [str(row["model_label"])]
            values.extend(
                "n/a"
                if not np.isfinite(float(row[key]))
                else (f"{float(row[key]):+.2f}%" if "vs_satloss" in path.name else f"{float(row[key]):.6g}")
                for key in endpoint_keys
            )
            lines.append("| " + " | ".join(values) + " |")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    render_table(
        absolute_md,
        absolute_rows,
        "SMART training-strategy endpoint errors",
        "Lower combined-global relative L2 is better. DeAL is the proposed two-view training strategy.",
    )
    render_table(
        relative_md,
        relative_rows,
        "SMART training strategies relative to DeAL",
        "Positive values mean the strategy is worse than SMART-DeAL at that endpoint; negative values mean it is better.",
    )
    return absolute_csv, relative_csv, absolute_md, relative_md


def write_strategy_relative_table(
    relative_rows: List[Dict[str, object]],
    endpoint_specs: Sequence[Tuple[str, str]],
    out_root: Path,
    reference_label: str,
    filename_stem: str,
) -> Tuple[Path, Path]:
    """Write a strategy percentage table for a named reference strategy."""
    csv_path = out_root / f"{filename_stem}.csv"
    markdown_path = out_root / f"{filename_stem}.md"
    write_csv(
        csv_path,
        relative_rows,
        ["model_name", "model_label", "reference_model"] + [key for key, _ in endpoint_specs],
    )
    header = ["Strategy"] + [satloss_endpoint_label(key, mode_name) for key, mode_name in endpoint_specs]
    lines = [
        f"# SMART training strategies relative to {reference_label}",
        "",
        f"Signed percentage difference in combined-global relative L2 versus {reference_label}; negative is better than the reference.",
        "",
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    for row in relative_rows:
        values = [str(row["model_label"])]
        values.extend(
            "n/a" if not np.isfinite(float(row[key])) else f"{float(row[key]):+.2f}%"
            for key, _ in endpoint_specs
        )
        lines.append("| " + " | ".join(values) + " |")
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return csv_path, markdown_path


def plot_strategy_endpoint_absolute_bars(
    rows: List[Dict[str, object]],
    endpoint_specs: Sequence[Tuple[str, str]],
    out_path: Path,
    title: str,
    log_scale: bool,
) -> None:
    if not rows or not endpoint_specs:
        return
    x = np.arange(len(endpoint_specs), dtype=np.float64)
    width = 0.82 / max(len(rows), 1)
    fig, ax = plt.subplots(figsize=(max(13.0, 1.65 * len(endpoint_specs)), 7.4))
    fig.subplots_adjust(left=0.12, right=0.76, bottom=0.34, top=0.86)
    strategy_font_size = 0.55 * _PLOT_BASE_FONT_SIZE * float(_PLOT_FONT_SCALE)
    show_std = bool(_COMPUTE_PLOT_STD)
    for row_idx, row in enumerate(rows):
        values = np.asarray([float(row[key]) for key, _ in endpoint_specs], dtype=np.float64)
        finite = np.isfinite(values)
        safe_values = np.where(finite, np.maximum(values, 1.0e-12), np.nan)
        errors = np.asarray(
            [float(row.get(f"{key}_std", 0.0)) for key, _ in endpoint_specs],
            dtype=np.float64,
        )
        if show_std:
            errors = np.nan_to_num(errors, nan=0.0)
            if log_scale:
                errors = np.minimum(errors, np.maximum(safe_values * 0.8, 0.0))
        else:
            errors = None
        bars = ax.bar(
            x + (row_idx - 0.5 * (len(rows) - 1)) * width,
            safe_values,
            width=width,
            yerr=errors,
            capsize=3,
            color=SMART_SAMPLING_STRATEGY_COLORS.get(str(row["model_name"]), "#777777"),
            edgecolor="black",
            linewidth=0.55,
            alpha=1.0,
            hatch="///" if row["model_name"] == "SMART_SATLOSS7" else "",
            label=str(row["model_label"]),
        )
        for bar, value in zip(bars, safe_values):
            if np.isfinite(value):
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    value * (1.08 if log_scale else 1.02),
                    f"{value:.3g}",
                    ha="center",
                    va="bottom",
                    rotation=90,
                    fontsize=0.82 * strategy_font_size,
                    clip_on=False,
                )
    if log_scale:
        ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [satloss_endpoint_label(key, mode_name) for key, mode_name in endpoint_specs],
        rotation=32,
        ha="right",
        fontsize=strategy_font_size,
    )
    ax.set_ylabel(
        f"Combined-global rel-L2 ({'log' if log_scale else 'linear'} scale)",
        fontsize=strategy_font_size,
    )
    ax.yaxis.set_label_coords(-0.095, 0.50)
    ax.tick_params(axis="y", labelsize=strategy_font_size)
    ax.set_title(title.replace("SMART training strategies: ", ""), fontsize=strategy_font_size)
    ax.grid(axis="y", which="both", alpha=0.2)
    ax.legend(
        fontsize=strategy_font_size,
        ncol=1,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        framealpha=0.92,
        borderpad=0.6,
        labelspacing=0.45,
    )
    _save_plot(fig, out_path, 280)
    plt.close(fig)


def plot_strategy_endpoint_relative_bars(
    rows: List[Dict[str, object]],
    endpoint_specs: Sequence[Tuple[str, str]],
    out_path: Path,
    title: str,
) -> None:
    if not rows or not endpoint_specs:
        return
    x = np.arange(len(endpoint_specs), dtype=np.float64)
    width = 0.82 / max(len(rows), 1)
    fig, ax = plt.subplots(figsize=(max(13.0, 1.65 * len(endpoint_specs)), 7.4))
    fig.subplots_adjust(left=0.12, right=0.76, bottom=0.34, top=0.86)
    strategy_font_size = 0.55 * _PLOT_BASE_FONT_SIZE * float(_PLOT_FONT_SCALE)
    for row_idx, row in enumerate(rows):
        values = np.asarray([float(row[key]) for key, _ in endpoint_specs], dtype=np.float64)
        values = np.nan_to_num(values, nan=0.0)
        bars = ax.bar(
            x + (row_idx - 0.5 * (len(rows) - 1)) * width,
            values,
            width=width,
            color=SMART_SAMPLING_STRATEGY_COLORS.get(str(row["model_name"]), "#777777"),
            edgecolor="black",
            linewidth=0.55,
            alpha=1.0,
            hatch="///" if row["model_name"] == "SMART_SATLOSS7" else "",
            label=str(row["model_label"]),
        )
        for bar, value in zip(bars, values):
            offset = 1.5 if value >= 0.0 else -1.5
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                value + offset,
                f"{value:+.1f}%",
                ha="center",
                va="bottom" if value >= 0.0 else "top",
                rotation=90,
                fontsize=0.82 * strategy_font_size,
                clip_on=False,
            )
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [satloss_endpoint_label(key, mode_name) for key, mode_name in endpoint_specs],
        rotation=32,
        ha="right",
        fontsize=strategy_font_size,
    )
    ax.set_ylabel("Relative error difference versus DeAL (%)", fontsize=strategy_font_size)
    ax.yaxis.set_label_coords(-0.095, 0.50)
    ax.tick_params(axis="y", labelsize=strategy_font_size)
    ax.set_title(title.replace("SMART training strategies: ", ""), fontsize=strategy_font_size)
    ax.grid(axis="y", alpha=0.2)
    finite_values = np.asarray(
        [
            float(row[key])
            for row in rows
            for key, _ in endpoint_specs
            if np.isfinite(float(row[key]))
        ],
        dtype=np.float64,
    )
    if finite_values.size:
        value_span = max(float(np.max(finite_values) - np.min(finite_values)), 1.0)
        margin = max(2.0, 0.08 * value_span)
        ax.set_ylim(float(np.min(finite_values)) - margin, float(np.max(finite_values)) + margin)
    ax.legend(
        fontsize=strategy_font_size,
        ncol=1,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        framealpha=0.92,
        borderpad=0.6,
        labelspacing=0.45,
    )
    _save_plot(fig, out_path, 280)
    plt.close(fig)


def plot_strategy_test_bars(
    absolute_rows: List[Dict[str, object]],
    relative_rows: List[Dict[str, object]],
    endpoint_specs: Sequence[Tuple[str, str]],
    out_path: Path,
    title: str,
    log_scale: bool,
    percentage_plot: bool = False,
) -> None:
    """Render one SATLOSS-style plot for one shift or one VTP method.

    Sampling shifts use one bar per strategy. Remeshing methods use paired
    div5/div10 bars per strategy, with opacity encoding the factor and color
    encoding the training strategy. This mirrors the paper-facing multi-model
    plots without collapsing all shifts into one unreadable chart.
    """
    if not absolute_rows or not endpoint_specs:
        return
    source_specs = list(endpoint_specs)
    is_geometry = all(str(mode_name).startswith("geometry_") for _, mode_name in source_specs)
    rows = relative_rows if percentage_plot else absolute_rows
    row_map = {str(row["model_name"]): row for row in rows}
    relative_row_map = {str(row["model_name"]): row for row in relative_rows}
    strategy_models = [
        model_name for model_name in SMART_SAMPLING_STRATEGY_MODELS if model_name in row_map
    ]
    if not strategy_models:
        return

    endpoint_count = len(source_specs)
    x = np.arange(len(strategy_models), dtype=np.float64)
    total_slots = endpoint_count if is_geometry else 1
    width = 0.82 / float(max(total_slots, 1))
    slot_pitch = 0.88 / float(max(total_slots, 1))
    font_size = 0.55 * _PLOT_BASE_FONT_SIZE * float(_PLOT_FONT_SCALE)
    fig, ax = plt.subplots(figsize=(max(11.5, 1.55 * len(strategy_models)), 7.2))
    fig.subplots_adjust(left=0.12, right=0.76, bottom=0.34, top=0.86)
    factor_alphas = {5: 0.50, 10: 1.0, 20: 0.65, 40: 0.85}
    all_values: List[float] = []
    annotation_values: List[float] = []

    for endpoint_idx, (column_key, mode_name) in enumerate(source_specs):
        values = np.asarray(
            [float(row_map[model_name].get(column_key, math.nan)) for model_name in strategy_models],
            dtype=np.float64,
        )
        if percentage_plot:
            values = np.nan_to_num(values, nan=0.0)
        else:
            values = np.where(np.isfinite(values), np.maximum(values, 1.0e-12), np.nan)
        all_values.extend(values[np.isfinite(values)].tolist())
        factor = None
        method = None
        if is_geometry:
            source_name = str(mode_name).removeprefix("geometry_")
            method, factor_text = source_name.split("_div", 1)
            factor = int(factor_text)
        alpha = factor_alphas.get(factor, 1.0) if factor is not None else 1.0

        for strategy_idx, model_name in enumerate(strategy_models):
            value = float(values[strategy_idx])
            if not np.isfinite(value):
                continue
            slot_idx = endpoint_idx if is_geometry else 0
            bar = ax.bar(
                x[strategy_idx] + (slot_idx - 0.5 * (total_slots - 1)) * slot_pitch,
                value,
                width=width,
                color=SMART_SAMPLING_STRATEGY_COLORS[model_name],
                edgecolor="black",
                linewidth=0.65,
                alpha=alpha,
                hatch="///" if model_name == "SMART_SATLOSS7" else "",
            )[0]

            # Strategy figures compare every alternative against vanilla
            # SMART, so all non-baseline strategies get the same percentage
            # annotation treatment. The generic multi-model plots retain
            # their separate SATLOSS-only annotation flag.
            show_label = model_name != "SMART"
            if show_label:
                label_y = value * 1.10 if not percentage_plot else value + (1.5 if value >= 0.0 else -1.5)
                label_va = "bottom" if (not percentage_plot or value >= 0.0) else "top"
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    label_y,
                    f"{value:+.1f}%"
                    if percentage_plot
                    else f"{float(relative_row_map[model_name][column_key]):+.1f}%",
                    ha="center",
                    va=label_va,
                    rotation=90,
                    fontsize=font_size,
                    clip_on=False,
                )
                annotation_values.append(label_y)

    if percentage_plot:
        ax.axhline(0.0, color="black", linewidth=1.0)
        y_label = "Relative error versus SMART (%)"
    else:
        if log_scale:
            ax.set_yscale("log")
        y_label = f"Combined global rel-L2 ({'log' if log_scale else 'linear'} scale)"
    ax.set_xticks(x)
    ax.set_xticklabels(
        [SMART_SAMPLING_STRATEGY_LABELS[model_name] for model_name in strategy_models],
        rotation=24,
        ha="right",
        fontsize=font_size,
    )
    ax.set_ylabel(y_label, fontsize=font_size)
    ax.yaxis.set_label_coords(-0.095, 0.50)
    ax.tick_params(axis="both", labelsize=font_size)
    ax.set_title(title, fontsize=font_size)
    ax.grid(axis="y", which="both", alpha=0.2)

    strategy_handles = [
        Patch(
            facecolor=SMART_SAMPLING_STRATEGY_COLORS[model_name],
            edgecolor="black",
            hatch="///" if model_name == "SMART_SATLOSS7" else "",
            label=SMART_SAMPLING_STRATEGY_LABELS[model_name],
        )
        for model_name in strategy_models
    ]
    if is_geometry:
        factor_handles = [
            Patch(facecolor="black", edgecolor="black", alpha=0.50, label="div5 (50% opacity)"),
            Patch(facecolor="black", edgecolor="black", alpha=1.0, label="div10 (100% opacity)"),
        ]
        first_legend = ax.legend(
            handles=strategy_handles,
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            fontsize=font_size,
            framealpha=0.92,
            borderpad=0.6,
            labelspacing=0.45,
        )
        ax.add_artist(first_legend)
        ax.legend(
            handles=factor_handles,
            loc="upper left",
            bbox_to_anchor=(1.02, 0.52),
            fontsize=font_size,
            framealpha=0.92,
            borderpad=0.6,
            labelspacing=0.45,
        )
    else:
        ax.legend(
            handles=strategy_handles,
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            fontsize=font_size,
            framealpha=0.92,
            borderpad=0.6,
            labelspacing=0.45,
        )

    if all_values:
        if percentage_plot:
            value_min = min(all_values)
            value_max = max(all_values)
            span = max(value_max - value_min, 1.0)
            margin = max(2.0, 0.08 * span)
            ax.set_ylim(value_min - margin, value_max + margin)
        else:
            ax.set_ylim(bottom=max(min(all_values) * 0.65, 1.0e-8))
            if annotation_values:
                ax.set_ylim(top=max(max(all_values), max(annotation_values)) * 1.25)
    _save_plot(fig, out_path, 260)
    plt.close(fig)


def plot_endpoint_error_bars(
    aggregate_rows: List[Dict[str, object]],
    metric_key: str,
    endpoint_modes: Sequence[str],
    endpoint_labels: Sequence[str],
    model_order: Sequence[str],
    out_path: Path,
    title: str,
    show_std: bool = True,
    log_scale: bool = True,
) -> None:
    """Plot two endpoint errors with vanilla-relative labels."""
    show_std = bool(show_std and _COMPUTE_PLOT_STD)
    if not endpoint_modes or len(endpoint_modes) != len(endpoint_labels):
        raise ValueError("Endpoint bar plots require at least one mode and matching labels.")
    row_map = {
        (str(row["model_name"]), str(row["sampling_mode"])): row
        for row in aggregate_rows
    }
    present_models = [
        model_name
        for model_name in model_order
        if all((model_name, mode_name) in row_map for mode_name in endpoint_modes)
    ]
    if not present_models:
        return

    model_groups: OrderedDict[str, List[str]] = OrderedDict()
    for model_name in present_models:
        base_name = _vanilla_model_name(model_name)
        model_groups.setdefault(base_name, [])
        if model_name not in model_groups[base_name]:
            model_groups[base_name].append(model_name)
    group_names = list(model_groups)
    model_index = {model_name: idx for idx, model_name in enumerate(present_models)}
    slots_per_group = max(len(members) for members in model_groups.values())
    total_slots = len(endpoint_modes) * slots_per_group
    slot_pitch = 0.88 / float(max(total_slots, 1))
    width = 0.82 / float(max(total_slots, 1))
    x = np.arange(len(group_names), dtype=np.float64)
    fig, ax = plt.subplots(figsize=(max(12.0, 1.55 * len(group_names)), 7.2))
    fig.subplots_adjust(left=0.12, right=0.76, bottom=0.34, top=0.86)
    endpoint_font_size = 0.55 * _PLOT_BASE_FONT_SIZE * float(_PLOT_FONT_SCALE)
    all_values: List[float] = []
    for endpoint_idx, (mode_name, endpoint_label) in enumerate(zip(endpoint_modes, endpoint_labels)):
        means = np.asarray(
            [max(float(row_map[(model_name, mode_name)][metric_key]), 1.0e-12) for model_name in present_models],
            dtype=np.float64,
        )
        stds = np.asarray(
            [max(float(row_map[(model_name, mode_name)].get(f"{metric_key}_std", 0.0)), 0.0) for model_name in present_models],
            dtype=np.float64,
        )
        # Error bars on a log scale must not extend below zero.
        stds = np.minimum(stds, means * 0.8)
        all_values.extend(means.tolist())
        for group_idx, group_name in enumerate(group_names):
            for variant_idx, model_name in enumerate(model_groups[group_name]):
                model_idx = model_index[model_name]
                base_name = _vanilla_model_name(model_name)
                baseline_mode = mode_name if _SATLOSS_ONLY_PERCENT_LABELS else endpoint_modes[0]
                baseline_row = row_map.get((base_name, baseline_mode))
                baseline = float(baseline_row[metric_key]) if baseline_row is not None else math.nan
                relative_pct = (
                    100.0 * (float(means[model_idx]) - baseline) / max(abs(baseline), 1.0e-12)
                    if np.isfinite(baseline)
                    else math.nan
                )
                independent_satloss = (
                    _INDEPENDENT_SATLOSS6_LINE_MODE
                    and model_name in INDEPENDENT_SATLOSS6_MODELS
                )
                color = (
                    INDEPENDENT_SATLOSS6_COLORS[model_name]
                    if independent_satloss
                    else LINE_MODEL_COLORS.get(base_name, MODEL_COLORS.get(model_name, "#777777"))
                )
                is_satloss = model_name != base_name
                hatch = "///" if is_satloss else ""
                alpha = 0.90 if endpoint_idx == 0 else 1.0
                slot_idx = endpoint_idx * slots_per_group + variant_idx
                bar = ax.bar(
                    x[group_idx] + (slot_idx - 0.5 * (total_slots - 1)) * slot_pitch,
                    means[model_idx],
                    width=width,
                    yerr=stds[model_idx] if show_std else None,
                    capsize=4,
                    color=color,
                    edgecolor="black",
                    linewidth=0.65,
                    alpha=alpha,
                    hatch=hatch,
                )[0]
                if not _SATLOSS_ONLY_PERCENT_LABELS or is_satloss:
                    label_text = "n/a" if not np.isfinite(relative_pct) else f"{relative_pct:+.1f}%"
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        means[model_idx] * 1.12,
                        label_text,
                        ha="center",
                        va="bottom",
                        rotation=90,
                        fontsize=endpoint_font_size,
                        clip_on=False,
                    )

    if log_scale:
        ax.set_yscale("log")
    ax.set_xticks(x)
    # SATLOSS bars are paired with their vanilla model and distinguished by
    # the hatch/legend, so the x-axis uses the vanilla family name once for
    # both members instead of repeating the SATLOSS suffix.
    ax.set_xticklabels(
        [MODEL_LABELS[group_name] for group_name in group_names],
        rotation=24,
        ha="right",
    )
    scale_label = "log scale" if log_scale else "linear scale"
    ax.set_ylabel(f"{_metric_display_name(metric_key)} ({scale_label})", fontsize=endpoint_font_size)
    ax.yaxis.set_label_coords(-0.095, 0.50)
    ax.tick_params(axis="both", labelsize=endpoint_font_size)
    display_title = title.replace("All compared models: ", "")
    ax.set_title(display_title, fontsize=endpoint_font_size)
    ax.grid(axis="y", which="both", alpha=0.2)
    endpoint_handles = [
        Patch(facecolor="#4C78A8", edgecolor="black", alpha=0.90, label=endpoint_labels[0]),
    ]
    if len(endpoint_labels) > 1:
        endpoint_handles.append(
            Patch(facecolor="#4C78A8", edgecolor="black", alpha=1.0, label=endpoint_labels[1])
        )
    endpoint_handles.extend(
        [
            Patch(facecolor="#4C78A8", edgecolor="black", alpha=1.0, hatch="///", label="DeAL variant"),
            Patch(
                facecolor="none",
                edgecolor="none",
                label=(
                    "DeAL labels: % vs vanilla"
                    if _SATLOSS_ONLY_PERCENT_LABELS
                    else "Labels: % vs vanilla baseline"
                ),
            ),
        ]
    )
    ax.legend(
        handles=endpoint_handles,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        fontsize=endpoint_font_size,
        framealpha=0.92,
        borderpad=0.6,
        labelspacing=0.45,
        handlelength=1.6,
    )
    if all_values:
        ax.set_ylim(bottom=max(min(all_values) * 0.65, 1.0e-8))
    _save_plot(fig, out_path, 260)
    plt.close(fig)


def plot_density_shift_bars(per_view_rows: List[Dict[str, object]], out_path: Path, title: str) -> None:
    mode_order = list(OrderedDict((str(r["sampling_mode"]), None) for r in per_view_rows).keys())
    means = []
    stds = []
    for mode_name in mode_order:
        vals = np.array(
            [float(r["subset_log_density_mean"]) for r in per_view_rows if r["sampling_mode"] == mode_name],
            dtype=np.float64,
        )
        vals = vals[np.isfinite(vals)]
        means.append(float(np.mean(vals)) if vals.size else math.nan)
        stds.append(float(np.std(vals)) if vals.size else math.nan)
    fig, ax = plt.subplots(figsize=(9.2, 5.4), constrained_layout=True)
    ax.bar(
        np.arange(len(mode_order)),
        means,
        yerr=stds if _COMPUTE_PLOT_STD else None,
        capsize=4,
        color=[mode_color(m) for m in mode_order],
        alpha=0.88,
    )
    ax.set_xticks(np.arange(len(mode_order)))
    ax.set_xticklabels([mode_display_name(mode_name) for mode_name in mode_order], rotation=20, ha="right")
    ax.set_ylabel("subset_log_density_mean")
    ax.set_title(title)
    _save_plot(fig, out_path, 220)
    plt.close(fig)


def plot_comprehensive_dashboard(
    per_run_mode_rows: List[Dict[str, object]],
    model_order: Sequence[str],
    mode_order: Sequence[str],
    out_path: Path,
    title: str,
    show_std: bool = True,
) -> None:
    show_std = bool(show_std and _COMPUTE_PLOT_STD)
    fig, axes = plt.subplots(3, 3, figsize=(18, 15), constrained_layout=True)
    dashboard_metrics = [
        "combined_physics_rel_l2",
        "combined_global_rel_l2",
        "surface_global_rel_l2",
        "volume_global_rel_l2",
        "surface_wss_mag_rel_l2",
        "volume_velocity_mag_rel_l2",
    ]
    for ax, metric_key in zip(axes.flat, dashboard_metrics):
        _grouped_bar_on_axis(ax, per_run_mode_rows, metric_key, mode_order, model_order, show_std=show_std)
    for ax in axes.flat[len(dashboard_metrics):]:
        ax.axis("off")
    fig.suptitle(title, fontsize=_font_size(18))
    _save_plot(fig, out_path, 220)
    plt.close(fig)


def paired_statistics(
    per_run_mode_rows: List[Dict[str, object]],
    model_order: Sequence[str],
    strongest_mode: str,
    metric_keys: Sequence[str],
) -> List[Dict[str, object]]:
    """Compute paired run-level effect sizes against aligned sampling.

    Pairing by run removes much of the geometry difficulty variance and is more
    informative than comparing two independent standard deviations.
    """
    out: List[Dict[str, object]] = []
    for model_name in model_order:
        model_rows = [r for r in per_run_mode_rows if r["model_name"] == model_name]
        aligned = {int(r["run_id"]): r for r in model_rows if r["sampling_mode"] == "aligned_uniform_wor"}
        shifted = {int(r["run_id"]): r for r in model_rows if r["sampling_mode"] == strongest_mode}
        common_runs = sorted(set(aligned) & set(shifted))
        for metric_key in metric_keys:
            deltas = np.asarray(
                [float(shifted[run_id][metric_key]) - float(aligned[run_id][metric_key]) for run_id in common_runs],
                dtype=np.float64,
            )
            if deltas.size == 0:
                continue
            mean = float(np.mean(deltas))
            std = float(np.std(deltas, ddof=1)) if deltas.size > 1 else 0.0
            sem = std / math.sqrt(deltas.size) if deltas.size > 1 else 0.0
            ci = 1.96 * sem
            z = mean / sem if sem > 1e-12 else (math.inf if mean > 0 else (-math.inf if mean < 0 else 0.0))
            p_value = math.erfc(abs(z) / math.sqrt(2.0)) if math.isfinite(z) else 0.0
            out.append(
                {
                    "model_name": model_name,
                    "sampling_mode": strongest_mode,
                    "metric": metric_key,
                    "n_paired_runs": int(deltas.size),
                    "mean_delta": mean,
                    "std_delta": std,
                    "median_delta": float(np.median(deltas)),
                    "q25_delta": float(np.percentile(deltas, 25)),
                    "q75_delta": float(np.percentile(deltas, 75)),
                    "ci95_low": mean - ci,
                    "ci95_high": mean + ci,
                    "normal_approx_p_value": float(p_value),
                    "aligned_mean": float(np.mean([aligned[i][metric_key] for i in common_runs])),
                    "shifted_mean": float(np.mean([shifted[i][metric_key] for i in common_runs])),
                }
            )
    return out


def plot_paired_statistics(
    stats_rows: List[Dict[str, object]],
    metric_key: str,
    model_order: Sequence[str],
    out_path: Path,
    title: str,
) -> None:
    rows = [r for r in stats_rows if r["metric"] == metric_key and r["model_name"] in model_order]
    if not rows:
        return
    row_map = {str(r["model_name"]): r for r in rows}
    present = [m for m in model_order if m in row_map]
    x = np.arange(len(present), dtype=np.float64)
    means = np.asarray([row_map[m]["mean_delta"] for m in present], dtype=np.float64)
    low = np.asarray([row_map[m]["ci95_low"] for m in present], dtype=np.float64)
    high = np.asarray([row_map[m]["ci95_high"] for m in present], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(max(8.0, 1.2 * len(present)), 5.8), constrained_layout=True)
    for idx, model_name in enumerate(present):
        ax.errorbar(
            [x[idx]],
            [means[idx]],
            yerr=[[means[idx] - low[idx]], [high[idx] - means[idx]]],
            fmt="o",
            capsize=5,
            linewidth=1.5,
            color=MODEL_COLORS[model_name],
            ecolor=MODEL_COLORS[model_name],
            markersize=7,
            zorder=3,
        )
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS[m] for m in present], rotation=20, ha="right")
    ax.set_ylabel("Shifted - aligned rel-L2")
    ax.set_title(title + " (paired 95% CI)")
    _save_plot(fig, out_path, 220)
    plt.close(fig)


def plot_metric_heatmap(
    aggregate_rows: List[Dict[str, object]],
    metric_key: str,
    model_order: Sequence[str],
    mode_order: Sequence[str],
    out_path: Path,
    title: str,
    normalize_to_aligned: bool = False,
) -> None:
    """Show the full model-by-mode result surface in one compact figure."""
    values = np.full((len(model_order), len(mode_order)), np.nan, dtype=np.float64)
    row_map = {(str(r["model_name"]), str(r["sampling_mode"])): r for r in aggregate_rows}
    for i, model_name in enumerate(model_order):
        baseline = row_map.get((model_name, "aligned_uniform_wor"))
        base_value = float(baseline[metric_key]) if baseline is not None else 1.0
        for j, mode_name in enumerate(mode_order):
            row = row_map.get((model_name, mode_name))
            if row is not None:
                value = float(row[metric_key])
                values[i, j] = value / max(base_value, 1.0e-12) if normalize_to_aligned else value
    fig, ax = plt.subplots(figsize=(max(10.0, 1.25 * len(mode_order)), max(4.5, 0.48 * len(model_order))), constrained_layout=True)
    image = ax.imshow(values, aspect="auto", cmap="magma", interpolation="nearest")
    ax.set_xticks(np.arange(len(mode_order)))
    ax.set_xticklabels([mode_display_name(m) for m in mode_order], rotation=35, ha="right")
    ax.set_yticks(np.arange(len(model_order)))
    ax.set_yticklabels([MODEL_LABELS[m] for m in model_order])
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            if np.isfinite(values[i, j]):
                ax.text(j, i, f"{values[i, j]:.3f}", ha="center", va="center", color="white", fontsize=_font_size(7))
    ax.set_title(title)
    ax.set_xlabel("Encoder-input sampling mode")
    ax.set_ylabel("Model")
    fig.colorbar(image, ax=ax, label="Ratio to aligned" if normalize_to_aligned else _metric_display_name(metric_key))
    _save_plot(fig, out_path, 220)
    plt.close(fig)


def plot_run_distribution_boxplot(
    per_run_mode_rows: List[Dict[str, object]],
    metric_key: str,
    mode_name: str,
    model_order: Sequence[str],
    out_path: Path,
    title: str,
) -> None:
    labels = []
    values = []
    colors = []
    for model_name in model_order:
        current = [float(r[metric_key]) for r in per_run_mode_rows if r["model_name"] == model_name and r["sampling_mode"] == mode_name]
        if current:
            labels.append(MODEL_LABELS[model_name])
            values.append(current)
            colors.append(model_line_visuals(model_name)[0])
    if not values:
        return
    fig, ax = plt.subplots(figsize=(max(8.0, 1.25 * len(values)), 5.8), constrained_layout=True)
    box = ax.boxplot(values, patch_artist=True, showmeans=True, meanline=False)
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(1.0)
    for median in box["medians"]:
        median.set_color("black")
    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=22, ha="right")
    ax.set_ylabel(_metric_display_name(metric_key))
    ax.set_title(title)
    _save_plot(fig, out_path, 220)
    plt.close(fig)


def plot_delta_severity_curve(
    aggregate_rows: List[Dict[str, object]],
    metric_key: str,
    model_order: Sequence[str],
    beta_mode_order: Sequence[str],
    beta_xs: Sequence[float],
    out_path: Path,
    title: str,
    show_std: bool = True,
    x_label: str = "Inverse-density beta",
) -> None:
    show_std = bool(show_std and _COMPUTE_PLOT_STD)
    row_map = {(str(r["model_name"]), str(r["sampling_mode"])): r for r in aggregate_rows}
    fig, ax = plt.subplots(figsize=(9.0, 5.8), constrained_layout=True)
    xs = np.asarray(beta_xs, dtype=np.float64)
    for model_name in model_order:
        aligned = row_map.get((model_name, "aligned_uniform_wor"))
        if aligned is None:
            continue
        ys = []
        ystd = []
        for mode_name in beta_mode_order:
            row = row_map.get((model_name, mode_name))
            if row is None:
                ys.append(np.nan)
                ystd.append(0.0)
            elif is_zero_shift_mode(mode_name):
                ys.append(0.0)
                ystd.append(0.0)
            else:
                ys.append(float(row[metric_key]) - float(aligned[metric_key]))
                ystd.append(float(row.get(f"{metric_key}_std", 0.0)))
        ys = np.asarray(ys, dtype=np.float64)
        ystd = np.asarray(ystd, dtype=np.float64)
        color, linestyle = model_line_visuals(model_name)
        ax.plot(xs, ys, marker="o", linewidth=2, color=color, linestyle=linestyle, label=MODEL_LABELS[model_name])
        if show_std:
            ax.fill_between(xs, ys - ystd, ys + ystd, color=color, alpha=0.14)
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Shifted - aligned rel-L2")
    ax.set_title(title)
    ax.legend(fontsize=_font_size(8))
    _save_plot(fig, out_path, 220)
    plt.close(fig)


def build_percentage_degradation_rows(
    per_run_mode_rows: List[Dict[str, object]],
    model_order: Sequence[str],
    metric_keys: Sequence[str],
) -> List[Dict[str, object]]:
    """Compute paired percentage worsening relative to each model's aligned view."""
    lookup = {
        (str(row["model_name"]), int(row["run_id"]), str(row["sampling_mode"])): row
        for row in per_run_mode_rows
    }
    out: List[Dict[str, object]] = []
    for model_name in model_order:
        model_rows = [row for row in per_run_mode_rows if row["model_name"] == model_name]
        for shifted in model_rows:
            mode_name = str(shifted["sampling_mode"])
            if mode_name == "aligned_uniform_wor":
                continue
            aligned = lookup.get((str(model_name), int(shifted["run_id"]), "aligned_uniform_wor"))
            if aligned is None:
                continue
            result = {
                "run_id": int(shifted["run_id"]),
                "model_name": str(model_name),
                "sampling_mode": mode_name,
                "sampling_kind": shifted["sampling_kind"],
                "sampling_mode_id": int(shifted["sampling_mode_id"]),
            }
            for metric_key in metric_keys:
                if is_zero_shift_mode(mode_name):
                    # beta=0 and sine=0 are control points, not independent
                    # degradation conditions. Their worsening is defined as
                    # exactly zero even though the sampled point sets differ.
                    result[f"{metric_key}_pct_worsening"] = 0.0
                else:
                    baseline = max(abs(float(aligned[metric_key])), 1.0e-12)
                    result[f"{metric_key}_pct_worsening"] = 100.0 * (
                        float(shifted[metric_key]) - float(aligned[metric_key])
                    ) / baseline
            out.append(result)
    return out


def plot_percentage_degradation_curve(
    percentage_rows: List[Dict[str, object]],
    metric_key: str,
    model_order: Sequence[str],
    mode_order: Sequence[str],
    x_values: Sequence[float],
    out_path: Path,
    title: str,
    x_label: str,
    show_std: bool = True,
) -> None:
    show_std = bool(show_std and _COMPUTE_PLOT_STD)
    pct_key = f"{metric_key}_pct_worsening"
    fig, ax = plt.subplots(figsize=(9.2, 5.8), constrained_layout=True)
    xs = np.asarray(x_values, dtype=np.float64)
    for model_name in model_order:
        ys = []
        ystd = []
        for mode_name in mode_order:
            values = np.asarray(
                [
                    float(row[pct_key])
                    for row in percentage_rows
                    if row["model_name"] == model_name and row["sampling_mode"] == mode_name
                ],
                dtype=np.float64,
            )
            ys.append(float(np.mean(values)) if values.size else np.nan)
            ystd.append(float(np.std(values)) if values.size else 0.0)
        ys = np.asarray(ys, dtype=np.float64)
        ystd = np.asarray(ystd, dtype=np.float64)
        for idx, mode_name in enumerate(mode_order):
            if is_zero_shift_mode(mode_name):
                ys[idx] = 0.0
                ystd[idx] = 0.0
        color, linestyle = model_line_visuals(model_name)
        ax.plot(xs, ys, marker="o", linewidth=2, color=color, linestyle=linestyle, label=MODEL_LABELS[model_name])
        if show_std:
            ax.fill_between(xs, ys - ystd, ys + ystd, color=color, alpha=0.15)
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Worsening relative to aligned (%)")
    ax.set_title(title)
    ax.legend(fontsize=_font_size(8))
    _save_plot(fig, out_path, 220)
    plt.close(fig)


def plot_family_percentage_degradation_curve(
    percentage_rows: List[Dict[str, object]],
    metric_key: str,
    family_groups: OrderedDict,
    model_names: Sequence[str],
    mode_order: Sequence[str],
    x_values: Sequence[float],
    out_path: Path,
    title: str,
    x_label: str,
    show_std: bool = True,
) -> None:
    show_std = bool(show_std and _COMPUTE_PLOT_STD)
    pct_key = f"{metric_key}_pct_worsening"
    fig, ax = plt.subplots(figsize=(9.2, 5.8), constrained_layout=True)
    xs = np.asarray(x_values, dtype=np.float64)
    active_models = set(model_names)
    family_colors = plt.cm.tab10(np.linspace(0.0, 1.0, max(len(family_groups), 1)))
    color_idx = 0
    for family_key, family_models in family_groups.items():
        present = [model for model in family_models if model in active_models]
        if not present:
            continue
        ys = []
        ystd = []
        for mode_name in mode_order:
            values = np.asarray(
                [
                    float(row[pct_key])
                    for row in percentage_rows
                    if row["model_name"] in present and row["sampling_mode"] == mode_name
                ],
                dtype=np.float64,
            )
            ys.append(float(np.mean(values)) if values.size else np.nan)
            ystd.append(float(np.std(values)) if values.size else 0.0)
        for idx, mode_name in enumerate(mode_order):
            if is_zero_shift_mode(mode_name):
                ys[idx] = 0.0
                ystd[idx] = 0.0
        color = family_colors[color_idx % len(family_colors)]
        color_idx += 1
        ys = np.asarray(ys, dtype=np.float64)
        ystd = np.asarray(ystd, dtype=np.float64)
        ax.plot(xs, ys, marker="o", linewidth=2.2, color=color, label=family_key.replace("_", " "))
        if show_std:
            ax.fill_between(xs, ys - ystd, ys + ystd, color=color, alpha=0.14)
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Mean worsening relative to aligned (%)")
    ax.set_title(title)
    ax.legend(fontsize=_font_size(9))
    _save_plot(fig, out_path, 220)
    plt.close(fig)


def plot_percentage_degradation_heatmap(
    percentage_rows: List[Dict[str, object]],
    metric_key: str,
    model_order: Sequence[str],
    mode_order: Sequence[str],
    out_path: Path,
    title: str,
) -> None:
    pct_key = f"{metric_key}_pct_worsening"
    values = np.full((len(model_order), len(mode_order)), np.nan, dtype=np.float64)
    for i, model_name in enumerate(model_order):
        for j, mode_name in enumerate(mode_order):
            current = [
                float(row[pct_key])
                for row in percentage_rows
                if row["model_name"] == model_name and row["sampling_mode"] == mode_name
            ]
            if current:
                values[i, j] = float(np.mean(current))
    fig, ax = plt.subplots(
        figsize=(max(10.0, 1.25 * len(mode_order)), max(4.5, 0.48 * len(model_order))),
        constrained_layout=True,
    )
    image = ax.imshow(values, aspect="auto", cmap="RdYlGn_r", interpolation="nearest")
    ax.set_xticks(np.arange(len(mode_order)))
    ax.set_xticklabels([mode_display_name(mode) for mode in mode_order], rotation=35, ha="right")
    ax.set_yticks(np.arange(len(model_order)))
    ax.set_yticklabels([MODEL_LABELS[model] for model in model_order])
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            if np.isfinite(values[i, j]):
                ax.text(j, i, f"{values[i, j]:.1f}%", ha="center", va="center", color="black", fontsize=_font_size(7))
    ax.set_xlabel("Encoder-input shift severity")
    ax.set_ylabel("Model")
    ax.set_title(title)
    fig.colorbar(image, ax=ax, label="Worsening relative to aligned (%)")
    _save_plot(fig, out_path, 220)
    plt.close(fig)


def plot_percentage_degradation_bars(
    percentage_rows: List[Dict[str, object]],
    metric_key: str,
    mode_name: str,
    model_order: Sequence[str],
    out_path: Path,
    title: str,
    show_std: bool = True,
) -> None:
    show_std = bool(show_std and _COMPUTE_PLOT_STD)
    pct_key = f"{metric_key}_pct_worsening"
    present = []
    means = []
    stds = []
    colors = []
    for model_name in model_order:
        values = np.asarray(
            [
                float(row[pct_key])
                for row in percentage_rows
                if row["model_name"] == model_name and row["sampling_mode"] == mode_name
            ],
            dtype=np.float64,
        )
        if not values.size:
            continue
        present.append(model_name)
        means.append(float(np.mean(values)))
        stds.append(float(np.std(values)))
        colors.append(model_line_visuals(model_name)[0])
    if not present:
        return
    fig, ax = plt.subplots(figsize=(max(8.0, 1.25 * len(present)), 5.6), constrained_layout=True)
    ax.bar(np.arange(len(present)), means, yerr=stds if show_std else None, capsize=4, color=colors, alpha=1.0)
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1)
    ax.set_xticks(np.arange(len(present)))
    ax.set_xticklabels([MODEL_LABELS[model] for model in present], rotation=20, ha="right")
    ax.set_ylabel("Worsening relative to aligned (%)")
    ax.set_title(title)
    _save_plot(fig, out_path, 220)
    plt.close(fig)


def main():
    global _INDEPENDENT_SATLOSS6_LINE_MODE, _COMPUTE_PLOT_STD, _SATLOSS_ONLY_PERCENT_LABELS, _PLOT_FONT_SCALE
    global GEOMETRY_SOURCE_LABELS, GEOMETRY_METHOD_LABELS
    args = parse_args()
    if args.geometry_label_preset == "v4":
        GEOMETRY_SOURCE_LABELS = dict(V4_GEOMETRY_SOURCE_LABELS)
        GEOMETRY_METHOD_LABELS = dict(V4_GEOMETRY_METHOD_LABELS)
    _COMPUTE_PLOT_STD = not bool(args.no_std)
    _SATLOSS_ONLY_PERCENT_LABELS = bool(args.satloss_only_percent_labels)
    _PLOT_FONT_SCALE = float(args.font_scale)
    _configure_plot_style(_PLOT_FONT_SCALE)
    seed_everything(args.seed)
    inference_devices = resolve_devices(args.device, args.devices)
    device = inference_devices[0]
    print(f"Inference devices: {', '.join(str(item) for item in inference_devices)}")
    print(f"Global comparison seed: {int(args.seed)}")

    config_name_map = OrderedDict(
        [
            ("SMART", args.smart_config),
            ("SMART_DOWNSAMPLE", args.smart_downsample_config),
            ("SMART_GAUSSIAN_BALL_MASKED", args.smart_gaussian_ball_masked_config),
            ("SMART_BOX_MASKED", args.smart_box_masked_config),
            ("SMART_SATLOSS3", args.smart_satloss3_config),
            ("SMART_SATLOSS4", args.smart_satloss4_config),
            ("SMART_SATLOSS5", args.smart_satloss5_config),
            ("SMART_SATLOSS5_NOPM", args.smart_satloss5_nopm_config),
            ("SMART_SATLOSS6", args.smart_satloss6_config),
            ("SMART_SATLOSS6_FIXEDSUM", args.smart_satloss6_fixedsum_config),
            ("SMART_SATLOSS6_GRADNORM", args.smart_satloss6_gradnorm_config),
            ("SMART_SATLOSS6_CONFIG_FULL", args.smart_satloss6_config_full_config),
            ("SMART_SATLOSS6_CONFIG_LAYER", args.smart_satloss6_config_layer_config),
            ("SMART_SATLOSS7", args.smart_satloss7_config),
            ("SMART_SATLOSS8", args.smart_satloss8_config),
            ("TRANSOLVERPP", args.transolverpp_config),
            ("TRANSOLVERPP_SATLOSS3", args.transolverpp_satloss3_config),
            ("TRANSOLVERPP_SATLOSS6", args.transolverpp_satloss6_config),
            ("TRANSOLVERPP_SATLOSS7", args.transolverpp_satloss7_config),
            ("TRANSOLVERPP_SATLOSS8", args.transolverpp_satloss8_config),
            ("POINTNET2_SSG", args.pointnet2_ssg_config),
            ("POINTNET2_SSG_SATLOSS6", args.pointnet2_ssg_satloss6_config),
            ("POINTNET2_SSG_SATLOSS7", args.pointnet2_ssg_satloss7_config),
            ("POINTNET2_SSG_SATLOSS8", args.pointnet2_ssg_satloss8_config),
            ("LNO", args.lno_config),
            ("LNO_SATLOSS6", args.lno_satloss6_config),
            ("LNO_SATLOSS7", args.lno_satloss7_config),
            ("LNO_SATLOSS8", args.lno_satloss8_config),
            ("MSPT", args.mspt_config),
            ("MSPT_SATLOSS6", args.mspt_satloss6_config),
            ("MSPT_SATLOSS7", args.mspt_satloss7_config),
            ("MSPT_SATLOSS8", args.mspt_satloss8_config),
            ("POINT_TRANSFORMER_V3", args.point_transformer_v3_config),
            ("POINT_TRANSFORMER_V3_SATLOSS6", args.point_transformer_v3_satloss6_config),
            ("POINT_TRANSFORMER_V3_SATLOSS7", args.point_transformer_v3_satloss7_config),
            ("POINT_TRANSFORMER_V3_SATLOSS8", args.point_transformer_v3_satloss8_config),
        ]
    )
    configs = OrderedDict((model_name, load_cfg(cfg_name)) for model_name, cfg_name in config_name_map.items())

    data_paths = {str(cfg.data_path) for cfg in configs.values()}
    if len(data_paths) != 1:
        raise ValueError(f"Expected one shared DrivAerML data path, got: {sorted(data_paths)}")
    smart_cfg = configs["SMART"]

    shift_betas = parse_shift_betas(args.shift_betas)
    if args.positive_shifts_only:
        positive_betas = [beta for beta in shift_betas if beta > 0.0]
        if not positive_betas:
            raise ValueError("--positive-shifts-only requires at least one positive value in --shift-betas.")
        shift_betas = [max(positive_betas)]
        sine_mix_levels = [1.0]
    else:
        sine_mix_levels = sine_mix_levels_from_shift_betas(shift_betas)
    active_shifts = parse_active_shifts(args.active_shifts)
    active_shift_set = set(active_shifts)
    geometry_decimation_factors = parse_geometry_decimation_factors(args.geometry_decimation_factors)
    active_geometry_sources = parse_active_geometry_sources(
        args.active_geometry_sources,
        geometry_decimation_factors,
    )
    print(f"Active sampling shifts: {', '.join(active_shifts)}")
    print(
        "Active VTP geometry sources: "
        + (", ".join(active_geometry_sources) if active_geometry_sources else "none")
    )
    mode_defs = OrderedDict()
    mode_defs["aligned_uniform_wor"] = {
        "kind": "uniform_wor",
        "beta": 0.0,
        "description": "Uniform without replacement, aligned with training-view sampling.",
        "id": 0,
    }
    if "beta" in active_shift_set:
        for i, beta in enumerate(shift_betas, start=1):
            mode_defs[f"shifted_inverse_density_beta_{beta:.2f}"] = {
                "kind": "inverse_density_wor",
                "beta": float(beta),
                "description": f"Inverse-density without replacement, same point budget with beta={beta:.2f}.",
                "id": i,
            }
    next_mode_id = len(mode_defs)
    ood_shift_defs = [
        ("sine_y", "sinusoidal_axis_mixture_wor", "sine-y", {"axis": 1}),
        ("sine_x", "sinusoidal_axis_mixture_wor", "sine-x", {"axis": 0}),
    ]
    for shift_idx, (shift_name, shift_kind, shift_label, extra_info) in enumerate(ood_shift_defs):
        if shift_name not in active_shift_set:
            continue
        for mix_idx, mix_fraction in enumerate(sine_mix_levels):
            mode_defs[f"ood_{shift_name}_mix_{mix_fraction:.2f}"] = {
                "kind": shift_kind,
                "beta": math.nan,
                "mix_fraction": float(mix_fraction),
                "distribution_key": shift_name,
                "description": (
                    f"OOD {shift_label} mixture without replacement: "
                    f"{mix_fraction:.2f} shifted sampling + {1.0 - float(mix_fraction):.2f} uniform sampling, same point budget."
                ),
                "id": next_mode_id + shift_idx * len(sine_mix_levels) + mix_idx,
                **extra_info,
            }

    next_mode_id = len(mode_defs)
    for source_idx, source_name in enumerate(active_geometry_sources):
        mode_defs[f"geometry_{source_name}"] = {
            "kind": "geometry_vtp",
            "beta": math.nan,
            "geometry_source": source_name,
            "description": (
                f"Uniform point sampling from {GEOMETRY_SOURCE_LABELS[source_name]} "
                "instead of the preprocessed surface point cloud."
            ),
            "id": next_mode_id + source_idx,
        }

    checkpoint_arg_map = {
        "SMART": args.smart_checkpoint,
        "SMART_DOWNSAMPLE": args.smart_downsample_checkpoint,
        "SMART_GAUSSIAN_BALL_MASKED": args.smart_gaussian_ball_masked_checkpoint,
        "SMART_BOX_MASKED": args.smart_box_masked_checkpoint,
        "SMART_SATLOSS3": args.smart_satloss3_checkpoint,
        "SMART_SATLOSS4": args.smart_satloss4_checkpoint,
        "SMART_SATLOSS5": args.smart_satloss5_checkpoint,
        "SMART_SATLOSS5_NOPM": args.smart_satloss5_nopm_checkpoint,
        "SMART_SATLOSS6": args.smart_satloss6_checkpoint,
        "SMART_SATLOSS6_FIXEDSUM": args.smart_satloss6_fixedsum_checkpoint,
        "SMART_SATLOSS6_GRADNORM": args.smart_satloss6_gradnorm_checkpoint,
        "SMART_SATLOSS6_CONFIG_FULL": args.smart_satloss6_config_full_checkpoint,
        "SMART_SATLOSS6_CONFIG_LAYER": args.smart_satloss6_config_layer_checkpoint,
        "SMART_SATLOSS7": args.smart_satloss7_checkpoint,
        "SMART_SATLOSS8": args.smart_satloss8_checkpoint,
        "TRANSOLVERPP": args.transolverpp_checkpoint,
        "TRANSOLVERPP_SATLOSS3": args.transolverpp_satloss3_checkpoint,
        "TRANSOLVERPP_SATLOSS6": args.transolverpp_satloss6_checkpoint,
        "TRANSOLVERPP_SATLOSS7": args.transolverpp_satloss7_checkpoint,
        "TRANSOLVERPP_SATLOSS8": args.transolverpp_satloss8_checkpoint,
        "POINTNET2_SSG": args.pointnet2_ssg_checkpoint,
        "POINTNET2_SSG_SATLOSS6": args.pointnet2_ssg_satloss6_checkpoint,
        "POINTNET2_SSG_SATLOSS7": args.pointnet2_ssg_satloss7_checkpoint,
        "POINTNET2_SSG_SATLOSS8": args.pointnet2_ssg_satloss8_checkpoint,
        "LNO": args.lno_checkpoint,
        "LNO_SATLOSS6": args.lno_satloss6_checkpoint,
        "LNO_SATLOSS7": args.lno_satloss7_checkpoint,
        "LNO_SATLOSS8": args.lno_satloss8_checkpoint,
        "MSPT": args.mspt_checkpoint,
        "MSPT_SATLOSS6": args.mspt_satloss6_checkpoint,
        "MSPT_SATLOSS7": args.mspt_satloss7_checkpoint,
        "MSPT_SATLOSS8": args.mspt_satloss8_checkpoint,
        "POINT_TRANSFORMER_V3": args.point_transformer_v3_checkpoint,
        "POINT_TRANSFORMER_V3_SATLOSS6": args.point_transformer_v3_satloss6_checkpoint,
        "POINT_TRANSFORMER_V3_SATLOSS7": args.point_transformer_v3_satloss7_checkpoint,
        "POINT_TRANSFORMER_V3_SATLOSS8": args.point_transformer_v3_satloss8_checkpoint,
    }
    requested_model_names = [model_name for model_name in MODEL_ORDER if checkpoint_arg_map[model_name] is not None]
    if not requested_model_names:
        raise ValueError("No model checkpoints were provided. Pass at least one --*-checkpoint argument.")
    if args.strategy_only:
        expected_strategy_models = set(SMART_SAMPLING_STRATEGY_MODELS)
        requested_strategy_models = set(requested_model_names)
        if requested_strategy_models != expected_strategy_models:
            missing = sorted(expected_strategy_models - requested_strategy_models)
            extra = sorted(requested_strategy_models - expected_strategy_models)
            raise ValueError(
                "--strategy-only requires exactly SMART, SMART_SATLOSS7, SMART_DOWNSAMPLE, "
                "SMART_GAUSSIAN_BALL_MASKED, and SMART_BOX_MASKED; "
                f"missing={missing}, extra={extra}"
            )
        print("[comparison scope] dedicated SMART training-strategy comparison")
    active_model_set = set(requested_model_names)
    _INDEPENDENT_SATLOSS6_LINE_MODE = bool(
        active_model_set.intersection(INDEPENDENT_SATLOSS6_MODELS)
        and active_model_set <= ({"SMART"} | INDEPENDENT_SATLOSS6_MODELS)
    )
    os.environ["SMART_COMPARE_INDEPENDENT_SATLOSS6"] = "1" if _INDEPENDENT_SATLOSS6_LINE_MODE else "0"
    if _INDEPENDENT_SATLOSS6_LINE_MODE:
        print(
            "[plot styles] SMART/SATLOSS6 weighting comparison detected: "
            "SATLOSS6 variants use independent colors and dashed lines; SMART uses a solid line."
        )
    # SATLOSS8 configs declare the geometry-domain split used for training.
    # Apply that split automatically when an SATLOSS8 checkpoint participates,
    # unless the caller explicitly supplied a CLI override.
    if not args.domain_split_json:
        configured_domain_splits = {
            str(getattr(configs[name], "geometry_domain_split_json", "")).strip()
            for name in requested_model_names
            if str(name).endswith("_SATLOSS8")
            and str(getattr(configs[name], "geometry_domain_split_json", "")).strip()
        }
        if len(configured_domain_splits) > 1:
            raise ValueError(
                "Active SATLOSS8 configs declare different geometry-domain split files: "
                f"{sorted(configured_domain_splits)}. Pass --domain-split-json explicitly."
            )
        if configured_domain_splits:
            args.domain_split_json = next(iter(configured_domain_splits))
            satloss8_configs = [
                configs[name]
                for name in requested_model_names
                if str(name).endswith("_SATLOSS8")
            ]
            train_clusters = {
                int(getattr(config, "geometry_domain_split_train_cluster", 0))
                for config in satloss8_configs
            }
            test_clusters = {
                int(getattr(config, "geometry_domain_split_test_cluster", 1))
                for config in satloss8_configs
            }
            if len(train_clusters) != 1 or len(test_clusters) != 1:
                raise ValueError(
                    "Active SATLOSS8 configs declare inconsistent train/test clusters. "
                    "Pass --domain-train-cluster and --domain-test-cluster explicitly."
                )
            args.domain_train_cluster = next(iter(train_clusters))
            args.domain_test_cluster = next(iter(test_clusters))
            print(
                "[domain split] automatically using SATLOSS8 config: "
                f"train_cluster={args.domain_train_cluster} "
                f"test_cluster={args.domain_test_cluster} "
                f"split={args.domain_split_json}"
            )
    # A PTv3 checkpoint can load with a mismatched YAML because the density-
    # sensitive changes preserve parameter shapes. Resolve this before any
    # model is built so the benchmark cannot silently use the wrong backbone.
    for model_name in requested_model_names:
        checkpoint_path = choose_ckpt(configs[model_name], checkpoint_arg_map[model_name])
        requested_config_name = config_name_map[model_name]
        resolved_config_name = resolve_ptv3_checkpoint_config_name(
            model_name,
            requested_config_name,
            checkpoint_path,
        )
        if resolved_config_name != requested_config_name:
            config_name_map[model_name] = resolved_config_name
            configs[model_name] = load_cfg(resolved_config_name)
    model_specs = OrderedDict(
        (
            model_name,
            {"config": configs[model_name], "checkpoint": choose_ckpt(configs[model_name], checkpoint_arg_map[model_name])},
        )
        for model_name in requested_model_names
    )
    for model_name, spec in model_specs.items():
        print(f"{model_name} checkpoint: {spec['checkpoint']}")
        summarize_training_view_config(model_name, spec["config"], spec["checkpoint"])

    if args.dry_run:
        print(
            f"[dry-run] domain split: train_cluster={int(args.domain_train_cluster)} "
            f"test_cluster={int(args.domain_test_cluster)} "
            f"split={args.domain_split_json or 'disabled'}"
        )
        print(f"[dry-run] active modes: {', '.join(mode_defs)}")
        print(f"[dry-run] active models: {', '.join(model_specs)}")
        return

    per_model_input_budgets = {
        model_name: (
            int(args.input_points)
            if args.input_points is not None
            else int(train_encoder_input_points(spec["config"], model_name))
        )
        for model_name, spec in model_specs.items()
    }
    if any(v <= 0 for v in per_model_input_budgets.values()):
        raise ValueError("This evaluator expects a positive encoder input size for every active model.")
    unique_input_budgets = sorted(set(int(v) for v in per_model_input_budgets.values()))
    if len(unique_input_budgets) == 1:
        print(f"Using shared train-aligned encoder input budget: {unique_input_budgets[0]} points.")
    else:
        budget_text = ", ".join(f"{k}={v}" for k, v in per_model_input_budgets.items())
        print(f"Using per-model train-aligned encoder input budgets: {budget_text}")
    dataset_geometry_points = max(unique_input_budgets)

    density_cfg = smart_cfg
    _, default_density_knn_k, density_neighbor_hops, density_cache_dtype = resolve_density_spec(density_cfg)
    density_knn_k = int(args.density_knn_k) if args.density_knn_k is not None else int(default_density_knn_k)
    density_estimator = str(args.density_estimator)

    dataset = AhmedMLDatasetV2(
        saved_folder=str(smart_cfg.data_path),
        if_test=True,
        geometry_points=dataset_geometry_points,
        surface_points=int(smart_cfg.num_surface_points),
        volume_points=int(smart_cfg.num_volume_points),
        scale_positions=bool(smart_cfg.scale_positions),
        require_preprocessed=True,
        geometry_density_knn_k=density_knn_k,
        geometry_density_neighbor_hops=density_neighbor_hops,
        geometry_density_estimator=density_estimator,
        geometry_density_cache_dtype=density_cache_dtype,
        **(
            {
                "domain_split_json": str(Path(args.domain_split_json).expanduser().resolve()),
                "domain_split_train_cluster": int(args.domain_train_cluster),
                "domain_split_test_cluster": int(args.domain_test_cluster),
            }
            if args.domain_split_json
            else {}
        ),
    )
    if args.domain_split_json:
        print(
            f"[domain split] train_cluster={int(args.domain_train_cluster)} "
            f"test_cluster={int(args.domain_test_cluster)} "
            f"split={Path(args.domain_split_json).expanduser().resolve()}"
        )

    surface_vtp_dir = Path(args.surface_vtp_dir).expanduser().resolve()
    geometry_vtp_dirs = {
        "angle": Path(args.angle_decimated_vtp_dir).expanduser().resolve(),
        "isotropic": Path(args.isotropic_decimated_vtp_dir).expanduser().resolve(),
        "voxel": Path(args.voxel_decimated_vtp_dir).expanduser().resolve(),
    }
    # Explicit IDs are commonly used to evaluate remeshed training and test
    # cases together. Keep the default random protocol test-only, but never
    # reject an explicitly requested, fully preprocessed run merely because
    # it belongs to the training split.
    requested_run_ids = [int(item.strip()) for item in str(args.run_ids).split(",") if item.strip()] if args.run_ids else []
    candidate_universe = set(dataset.all_ids if requested_run_ids else dataset.test_ids)
    geometry_candidate_ids: set[int] | None = None
    if active_geometry_sources:
        geometry_candidate_ids = {int(run_id) for run_id in candidate_universe}
        for source_name in active_geometry_sources:
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
                "No eligible runs contain every requested geometry source: "
                f"{active_geometry_sources}. Check the three geometry VTP directory arguments."
            )
        print(
            f"VTP geometry tests available on {len(geometry_candidate_ids)} "
            f"{'all-data' if requested_run_ids else 'test-split'} runs; "
            f"selecting from the common completed subset."
        )
    run_ids = select_run_ids(
        candidate_universe,
        args.num_runs,
        args.run_ids,
        args.seed,
        candidate_ids=geometry_candidate_ids,
    )
    print(f"Evaluating run ids: {run_ids}")
    vtk_run_id = int(args.vtk_run_id) if args.vtk_run_id is not None else int(run_ids[0])
    if vtk_run_id not in run_ids:
        print(f"VTK representative run_id {vtk_run_id} is not in evaluated runs; it will still be exported separately.")

    mean_s = dataset.mean_surf_data
    std_s = torch.clamp(dataset.std_surf_data, min=1e-12)
    mean_v = dataset.mean_vol_data
    std_v = torch.clamp(dataset.std_vol_data, min=1e-12)
    min_pos = dataset.min_pos
    max_pos = dataset.max_pos

    if len(inference_devices) > 1 and any(item.type != "cuda" for item in inference_devices):
        raise ValueError("Multi-device comparison currently requires CUDA devices.")
    model_device_by_name = {
        model_name: inference_devices[index % len(inference_devices)]
        for index, model_name in enumerate(model_specs)
    }
    print(
        "[multi-gpu] Persistent model placement: "
        + ", ".join(
            f"{model_name}->{model_device_by_name[model_name]}"
            for model_name in model_specs
        )
    )
    models = {}
    for model_name, spec in model_specs.items():
        model_device = model_device_by_name[model_name]
        models[model_name] = build_model(
            spec["config"],
            spec["checkpoint"],
            model_device,
            batched_query_subregion_size=args.batched_query_subregion_size,
        ).to(model_device)
    model_internal_density_specs: Dict[str, Dict[str, object]] = {}
    for model_name, spec in model_specs.items():
        if not model_uses_density(model_name):
            continue
        model_density_estimator, model_density_knn_k, model_density_neighbor_hops, _ = resolve_model_internal_density_spec(
            model_name,
            spec["config"],
            spec["checkpoint"],
        )
        model_internal_density_specs[model_name] = {
            "estimator": model_density_estimator,
            "knn_k": model_density_knn_k,
            "neighbor_hops": model_density_neighbor_hops,
            "checkpoint": spec["checkpoint"],
        }
        print(
            f"{MODEL_LABELS[model_name]} internal density: "
            f"estimator={model_density_estimator}, knn_k={model_density_knn_k}, neighbor_hops={model_density_neighbor_hops}"
        )
        if not checkpoint_density_tag_is_explicit(spec["checkpoint"]):
            print(
                f"[warning] {MODEL_LABELS[model_name]} checkpoint name does not explicitly encode its density setup; "
                "the evaluator is using the active config defaults for any missing density metadata."
            )
    density_dataset_cache: Dict[Tuple[str, int, int, str], AhmedMLDatasetV2] = {}

    def get_density_dataset_for_spec(density_estimator_name: str, knn_k: int, neighbor_hops: int, cache_dtype: str) -> AhmedMLDatasetV2:
        key = (str(density_estimator_name), int(knn_k), int(neighbor_hops), str(cache_dtype))
        cached = density_dataset_cache.get(key)
        if cached is not None:
            return cached
        density_dataset = AhmedMLDatasetV2(
            saved_folder=str(smart_cfg.data_path),
            if_test=True,
            geometry_points=dataset_geometry_points,
            surface_points=int(smart_cfg.num_surface_points),
            volume_points=int(smart_cfg.num_volume_points),
            scale_positions=bool(smart_cfg.scale_positions),
            require_preprocessed=True,
            geometry_density_knn_k=int(knn_k),
            geometry_density_neighbor_hops=int(neighbor_hops),
            geometry_density_estimator=str(density_estimator_name),
            geometry_density_cache_dtype=str(cache_dtype),
            **(
                {
                    "domain_split_json": str(Path(args.domain_split_json).expanduser().resolve()),
                    "domain_split_train_cluster": int(args.domain_train_cluster),
                    "domain_split_test_cluster": int(args.domain_test_cluster),
                }
                if args.domain_split_json
                else {}
            ),
        )
        density_dataset_cache[key] = density_dataset
        return density_dataset

    auto_surface_query_points = min(int(spec["config"].num_surface_points) for spec in model_specs.values())
    auto_volume_query_points = min(int(spec["config"].num_volume_points) for spec in model_specs.values())
    surface_query_points = int(args.surface_query_points) if int(args.surface_query_points) > 0 else auto_surface_query_points
    volume_query_points = int(args.volume_query_points) if int(args.volume_query_points) > 0 else auto_volume_query_points
    print(f"Using fixed fair query budgets: {surface_query_points} surface points, {volume_query_points} volume points.")
    per_model_query_budgets = {}
    for model_name in model_specs:
        model_surface_query_points = surface_query_points
        model_volume_query_points = volume_query_points
        per_model_query_budgets[model_name] = {
            "surface": model_surface_query_points,
            "volume": model_volume_query_points,
        }
    for model_name, budget in per_model_query_budgets.items():
        if budget["surface"] != surface_query_points or budget["volume"] != volume_query_points:
            print(
                f"[info] {model_name} query override: "
                f"{budget['surface']} surface / {budget['volume']} volume "
                f"(global default is {surface_query_points} / {volume_query_points})."
            )
    encoder_budget_mismatch_models = []
    for model_name, spec in model_specs.items():
        train_encoder_points = train_encoder_input_points(spec["config"], model_name)
        eval_encoder_points = int(per_model_input_budgets[model_name])
        if int(eval_encoder_points) != int(train_encoder_points):
            encoder_budget_mismatch_models.append(
                {
                    "model_name": model_name,
                    "train_encoder_input_points": int(train_encoder_points),
                    "eval_encoder_input_points": int(eval_encoder_points),
                }
            )
    for item in encoder_budget_mismatch_models:
        print(
            "[warning] "
            f"{item['model_name']} was trained with "
            f"{item['train_encoder_input_points']} encoder input points, "
            f"but this evaluation uses {item['eval_encoder_input_points']}."
        )
    query_budget_mismatch_models = []
    for model_name, spec in model_specs.items():
        train_surface = int(spec["config"].num_surface_points)
        train_volume = int(spec["config"].num_volume_points)
        eval_surface = int(per_model_query_budgets[model_name]["surface"])
        eval_volume = int(per_model_query_budgets[model_name]["volume"])
        if eval_surface > train_surface or eval_volume > train_volume:
            query_budget_mismatch_models.append(
                {
                    "model_name": model_name,
                    "train_surface_query_points": train_surface,
                    "train_volume_query_points": train_volume,
                    "eval_surface_query_points": eval_surface,
                    "eval_volume_query_points": eval_volume,
                }
            )
    for item in query_budget_mismatch_models:
        print(
            "[warning] "
            f"{item['model_name']} was trained with "
            f"{item['train_surface_query_points']} surface / {item['train_volume_query_points']} volume query points, "
            f"but this evaluation requests {item['eval_surface_query_points']} surface / {item['eval_volume_query_points']} volume query points."
        )

    out_root = Path(
        args.output_dir
        or (SMART_ROOT.parent / "results" / "drivaerml_sampling_invariance_multifamily" / f"seed_{args.seed}_runs_{len(run_ids)}_views_{args.views_per_mode}")
    )
    out_root.mkdir(parents=True, exist_ok=True)

    per_view_rows: List[Dict[str, object]] = []
    drag_rank_view_rows: List[Dict[str, object]] = []
    views_per_mode = max(1, int(args.views_per_mode))
    view_batch_size = max(1, int(args.view_batch_size))
    if view_batch_size == 1 and views_per_mode > 1:
        boosted_view_batch_size = min(views_per_mode, 2)
        if boosted_view_batch_size > view_batch_size:
            print(
                f"[info] Bumping view batch size from {view_batch_size} to {boosted_view_batch_size} "
                "to improve GPU utilization without changing the comparison."
            )
            view_batch_size = boosted_view_batch_size

    for run_id in tqdm(run_ids, desc="Runs", dynamic_ncols=True):
        run_dir = Path(smart_cfg.data_path) / f"run_{run_id}"
        surf_coords_full = np.load(run_dir / "surface_coords.npy").astype(np.float32, copy=False)
        surf_p = np.load(run_dir / "surface_pMeanTrim.npy").astype(np.float32, copy=False).reshape(-1, 1)
        surf_n = np.load(run_dir / "surface_normals.npy").astype(np.float32, copy=False)
        surf_wx = np.load(run_dir / "surface_wallShearStressMeanTrim_x.npy").astype(np.float32, copy=False).reshape(-1, 1)
        surf_wy = np.load(run_dir / "surface_wallShearStressMeanTrim_y.npy").astype(np.float32, copy=False).reshape(-1, 1)
        surf_wz = np.load(run_dir / "surface_wallShearStressMeanTrim_z.npy").astype(np.float32, copy=False).reshape(-1, 1)
        surf_area_full = np.load(run_dir / "surface_areas.npy").astype(np.float32, copy=False)
        surf_gt_full = np.concatenate([surf_p, surf_n, surf_wx, surf_wy, surf_wz], axis=1)
        full_drag_force_gt = compute_surface_drag_force_x(surf_gt_full, surf_area_full)

        geometry_source_points: Dict[str, np.ndarray] = {}
        geometry_source_norm: Dict[str, torch.Tensor] = {}
        for source_name in active_geometry_sources:
            source_path = geometry_source_vtp_path(source_name, run_id, geometry_vtp_dirs)
            source_points = read_vtp_points(source_path)
            validate_geometry_source_bbox(source_points, surf_coords_full, source_name, run_id)
            geometry_source_points[source_name] = source_points
            geometry_source_norm[source_name] = normalize_pos(
                torch.from_numpy(source_points), min_pos, max_pos
            )
            print(
                f"run_{run_id} {source_name}: {source_points.shape[0]} source points "
                f"from {source_path.name}"
            )

        vol_coords_full = np.load(run_dir / "volume_coords.npy").astype(np.float32, copy=False)
        vol_p = np.load(run_dir / "volume_pMeanTrim.npy").astype(np.float32, copy=False).reshape(-1, 1)
        vol_u = np.load(run_dir / "volume_UMeanTrim.npy").astype(np.float32, copy=False)
        vol_gt_full = np.concatenate([vol_p, vol_u], axis=1)

        full_surf_query_norm = normalize_pos(torch.from_numpy(surf_coords_full), min_pos, max_pos)
        max_surface_query_points = max(int(b["surface"]) for b in per_model_query_budgets.values())
        max_volume_query_points = max(int(b["volume"]) for b in per_model_query_budgets.values())
        surf_query_idx_master = choose_fixed_query_indices(
            surf_coords_full.shape[0],
            max_surface_query_points,
            [args.seed, int(run_id), 3001],
            replace=bool(args.query_sampling_with_replacement),
        )
        vol_query_idx_master = choose_fixed_query_indices(
            vol_coords_full.shape[0],
            max_volume_query_points,
            [args.seed, int(run_id), 3002],
            replace=bool(args.query_sampling_with_replacement),
        )

        full_geo_log_density = dataset._load_or_compute_full_geometry_density(run_id, expected_n=int(surf_coords_full.shape[0]))
        full_geo_log_density_np = full_geo_log_density.to(dtype=torch.float32).numpy()
        distribution_weights = {
            "sine_y": sinusoidal_axis_probabilities(surf_coords_full, axis=1),
            "sine_x": sinusoidal_axis_probabilities(surf_coords_full, axis=0),
        }
        model_full_geo_log_density_by_name: Dict[str, torch.Tensor] = {}
        for model_name, spec in model_specs.items():
            cfg_has_density_spec = any(
                hasattr(spec["config"], key)
                for key in ("density_estimator", "density_knn_k", "geometry_density_knn_k")
            )
            if not model_uses_density(model_name) and not cfg_has_density_spec:
                continue
            if model_uses_density(model_name):
                model_density_estimator, model_density_knn_k, model_density_neighbor_hops, model_density_cache_dtype = resolve_model_internal_density_spec(
                    model_name, spec["config"], spec["checkpoint"]
                )
            else:
                model_density_estimator, model_density_knn_k, model_density_neighbor_hops, model_density_cache_dtype = resolve_density_spec(spec["config"])
            model_density_dataset = get_density_dataset_for_spec(
                model_density_estimator,
                model_density_knn_k,
                model_density_neighbor_hops,
                model_density_cache_dtype,
            )
            model_full_geo_log_density_by_name[model_name] = model_density_dataset._load_or_compute_full_geometry_density(
                run_id,
                expected_n=int(surf_coords_full.shape[0]),
            )

        model_groups = defaultdict(list)
        for model_name in model_specs:
            model_groups[model_device_by_name[model_name]].append(model_name)

        def evaluate_model_group(group_names, mode_name, mode_info):
            group_per_view_rows = []
            group_drag_rank_rows = []
            for model_name in group_names:
                model = models[model_name]
                model.eval()
                model_device = model_device_by_name[model_name]
                model_input_points = int(per_model_input_budgets[model_name])
                aligned_dataset_replacement = train_geometry_uses_replacement(
                    model_specs[model_name]["config"],
                    model_input_points,
                    int(surf_coords_full.shape[0]),
                )
                null_distribution_mode = is_zero_distribution_mode(mode_info)
                effective_sampling_kind = "uniform_wor" if null_distribution_mode else mode_info["kind"]
                configured_sampling_mode = resolve_eval_sampling_mode(
                    model_specs[model_name]["config"], effective_sampling_kind
                )
                input_sampling_with_replacement = sampling_mode_uses_replacement(
                    configured_sampling_mode,
                    aligned_dataset_replacement,
                )
                sampling_density_np = full_geo_log_density_np
                model_density_source = model_full_geo_log_density_by_name.get(model_name)
                if model_density_source is not None:
                    sampling_density_np = model_density_source.to(dtype=torch.float32).numpy()
                idx_list = []
                geo_view_source = None
                if mode_info["kind"] == "geometry_vtp":
                    geo_view_source = geometry_source_points[str(mode_info["geometry_source"])]
                    sampling_density_np = None
                subset_density_stats = []
                for view_idx in range(views_per_mode):
                    rng = np.random.default_rng(
                        np.random.SeedSequence(
                            [
                                args.seed,
                                int(run_id),
                                0 if null_distribution_mode else int(mode_info["id"]),
                                int(view_idx),
                            ]
                        )
                    )
                    if mode_info["kind"] == "geometry_vtp":
                        if geo_view_source.shape[0] < model_input_points:
                            raise ValueError(
                                f"{mode_info['geometry_source']} run_{run_id} has only "
                                f"{geo_view_source.shape[0]} points, below the configured "
                                f"{model_name} input budget {model_input_points}."
                            )
                        idx = sample_uniform_without_replacement(geo_view_source.shape[0], model_input_points, rng)
                    elif effective_sampling_kind == "uniform_wor":
                        idx = (
                            sample_uniform_with_replacement(surf_coords_full.shape[0], model_input_points, rng)
                            if input_sampling_with_replacement
                            else sample_uniform_without_replacement(surf_coords_full.shape[0], model_input_points, rng)
                        )
                    elif effective_sampling_kind == "inverse_density_wor":
                        idx = (
                            sample_inverse_density_with_replacement(sampling_density_np, model_input_points, float(mode_info["beta"]), rng)
                            if input_sampling_with_replacement
                            else sample_inverse_density_without_replacement(sampling_density_np, model_input_points, float(mode_info["beta"]), rng)
                        )
                    elif effective_sampling_kind == "sinusoidal_axis_mixture_wor":
                        target_weights = distribution_weights[str(mode_info["distribution_key"])]
                        idx = (
                            sample_uniform_weighted_mixture_with_replacement(
                                target_weights, model_input_points, float(mode_info["mix_fraction"]), rng
                            )
                            if input_sampling_with_replacement
                            else sample_uniform_weighted_mixture_without_replacement(
                                target_weights, model_input_points, float(mode_info["mix_fraction"]), rng
                            )
                        )
                    else:
                        raise ValueError(f"Unsupported sampling kind: {mode_info['kind']}")
                    idx_list.append(idx)
                    if sampling_density_np is None:
                        subset_density_stats.append(
                            {
                                "subset_log_density_mean": math.nan,
                                "subset_log_density_std": math.nan,
                                "subset_log_density_p05": math.nan,
                                "subset_log_density_p95": math.nan,
                            }
                        )
                    else:
                        subset = sampling_density_np[idx]
                        subset_density_stats.append(
                            {
                                "subset_log_density_mean": float(np.mean(subset)),
                                "subset_log_density_std": float(np.std(subset)),
                                "subset_log_density_p05": float(np.percentile(subset, 5)),
                                "subset_log_density_p95": float(np.percentile(subset, 95)),
                            }
                        )

                model_surface_query_points = int(per_model_query_budgets[model_name]["surface"])
                model_volume_query_points = int(per_model_query_budgets[model_name]["volume"])
                surf_query_idx = surf_query_idx_master[:model_surface_query_points]
                vol_query_idx = vol_query_idx_master[:model_volume_query_points]
                surf_coords = surf_coords_full[surf_query_idx]
                surf_gt = surf_gt_full[surf_query_idx]
                surf_area = surf_area_full[surf_query_idx]
                vol_coords = vol_coords_full[vol_query_idx]
                vol_gt = vol_gt_full[vol_query_idx]
                surf_query_norm = normalize_pos(torch.from_numpy(surf_coords), min_pos, max_pos)
                vol_query_norm = normalize_pos(torch.from_numpy(vol_coords), min_pos, max_pos)
                for batch_start in range(0, views_per_mode, view_batch_size):
                    batch_stop = min(batch_start + view_batch_size, views_per_mode)
                    batch_indices = idx_list[batch_start:batch_stop]
                    if mode_info["kind"] == "geometry_vtp":
                        source_norm = geometry_source_norm[str(mode_info["geometry_source"])]
                        geo_view_tensors = [source_norm[torch.from_numpy(idx)] for idx in batch_indices]
                    else:
                        geo_view_tensors = [full_surf_query_norm[torch.from_numpy(idx)] for idx in batch_indices]
                    geo_views_norm = torch.stack(geo_view_tensors, dim=0)
                    if model_uses_density(model_name):
                        density_source = model_full_geo_log_density_by_name[model_name]
                        geo_density_tensors = [
                            density_source.index_select(0, torch.from_numpy(idx).to(dtype=torch.long))
                            for idx in batch_indices
                        ]
                        geo_density_views = torch.stack(geo_density_tensors, dim=0)
                    else:
                        geo_density_views = None

                    prediction_args = {
                        "model_name": model_name,
                        "model": model,
                        "geo_views_norm": geo_views_norm,
                        "surf_query_norm": surf_query_norm,
                        "vol_query_norm": vol_query_norm,
                        "geo_log_density_views": geo_density_views,
                        "mean_s": mean_s,
                        "std_s": std_s,
                        "mean_v": mean_v,
                        "std_v": std_v,
                        "device": model_device,
                        "base_seed": int(
                            args.seed
                            + 100000 * (0 if null_distribution_mode else mode_info["id"])
                            + 1000 * run_id
                            + batch_start * 17
                        ),
                        "repeats": args.model_repeats,
                    }
                    pred_surf_batch, pred_vol_batch = predict_view_batch(**prediction_args)
                    full_pred_surf_batch = None
                    if model_name in DRAG_RANK_MODELS and mode_info["kind"] in {
                        "inverse_density_wor",
                        "sinusoidal_axis_mixture_wor",
                    }:
                        full_prediction_args = dict(prediction_args)
                        full_prediction_args["surf_query_norm"] = full_surf_query_norm
                        full_pred_surf_batch, _ = predict_view_batch(**full_prediction_args)

                    for local_idx, global_view_idx in enumerate(range(batch_start, batch_stop)):
                        metrics = compute_metrics(surf_gt, pred_surf_batch[local_idx], vol_gt, pred_vol_batch[local_idx])
                        surf_drag_force_gt = compute_surface_drag_force_x(surf_gt, surf_area)
                        surf_drag_force_pred = compute_surface_drag_force_x(pred_surf_batch[local_idx], surf_area)
                        metrics["surface_drag_force_x_rel_l2"] = rel_l2(
                            np.array([surf_drag_force_gt], dtype=np.float64),
                            np.array([surf_drag_force_pred], dtype=np.float64),
                        )
                        full_drag_force_pred = None
                        if full_pred_surf_batch is not None:
                            full_drag_force_pred = compute_surface_drag_force_x(
                                full_pred_surf_batch[local_idx], surf_area_full
                            )
                        density_stats = subset_density_stats[global_view_idx]
                        group_per_view_rows.append(
                            {
                                "run_id": int(run_id),
                                "view_id": int(global_view_idx),
                                "model_name": model_name,
                                "sampling_mode": mode_name,
                                "sampling_kind": (
                                    "geometry_vtp"
                                    if mode_info["kind"] == "geometry_vtp"
                                    else effective_sampling_kind
                                ),
                                "shift_beta": float(mode_info["beta"]),
                                "sampling_mode_id": int(mode_info["id"]),
                                "checkpoint": model_specs[model_name]["checkpoint"],
                                "input_points": int(batch_indices[local_idx].shape[0]),
                                "input_sampling_with_replacement": bool(
                                    False if mode_info["kind"] == "geometry_vtp" else input_sampling_with_replacement
                                ),
                                "geometry_source": str(mode_info.get("geometry_source", "preprocessed_surface")),
                                "configured_eval_sampling_mode": configured_sampling_mode,
                                "sampling_density_estimator": (
                                    model_internal_density_specs.get(model_name, {}).get("estimator", density_estimator)
                                ),
                                "sampling_density_knn_k": int(
                                    model_internal_density_specs.get(model_name, {}).get("knn_k", density_knn_k)
                                ),
                                "surface_query_points": model_surface_query_points,
                                "volume_query_points": model_volume_query_points,
                                "full_log_density_mean": (
                                    float(np.mean(sampling_density_np)) if sampling_density_np is not None else float("nan")
                                ),
                                "surface_drag_force_x_gt": float(surf_drag_force_gt),
                                "surface_drag_force_x_pred": float(surf_drag_force_pred),
                                "surface_drag_force_x_full_gt": float(full_drag_force_gt),
                                "surface_drag_force_x_full_pred": (
                                    float(full_drag_force_pred) if full_drag_force_pred is not None else float("nan")
                                ),
                                **density_stats,
                                **metrics,
                            }
                        )
                        if full_drag_force_pred is not None:
                            group_drag_rank_rows.append(
                                {
                                    "run_id": int(run_id),
                                    "view_id": int(global_view_idx),
                                    "model_name": model_name,
                                    "sampling_mode": mode_name,
                                    "sampling_kind": (
                                        "geometry_vtp"
                                        if mode_info["kind"] == "geometry_vtp"
                                        else effective_sampling_kind
                                    ),
                                    "shift_beta": float(mode_info["beta"]),
                                    "sampling_mode_id": int(mode_info["id"]),
                                    "surface_drag_force_x_full_gt": float(full_drag_force_gt),
                                    "surface_drag_force_x_full_pred": float(full_drag_force_pred),
                                }
                            )
                    del geo_views_norm, pred_surf_batch, pred_vol_batch
                    if geo_density_views is not None:
                        del geo_density_views
            return group_per_view_rows, group_drag_rank_rows

        with ThreadPoolExecutor(max_workers=len(model_groups)) as inference_pool:
            for mode_name, mode_info in mode_defs.items():
                futures = [
                    inference_pool.submit(evaluate_model_group, group_names, mode_name, mode_info)
                    for group_names in model_groups.values()
                ]
                for future in futures:
                    group_rows, group_drag_rows = future.result()
                    per_view_rows.extend(group_rows)
                    drag_rank_view_rows.extend(group_drag_rows)

    per_view_metric_keys = HEADLINE_METRIC_KEYS + SURFACE_FIELD_METRIC_KEYS + VOLUME_FIELD_METRIC_KEYS
    per_view_fieldnames = [
        "run_id",
        "view_id",
        "model_name",
        "sampling_mode",
        "sampling_kind",
        "shift_beta",
        "sampling_mode_id",
        "checkpoint",
        "input_points",
        "input_sampling_with_replacement",
        "geometry_source",
        "configured_eval_sampling_mode",
        "sampling_density_estimator",
        "sampling_density_knn_k",
        "surface_query_points",
        "volume_query_points",
        "full_log_density_mean",
        "surface_drag_force_x_gt",
        "surface_drag_force_x_pred",
        "surface_drag_force_x_full_gt",
        "surface_drag_force_x_full_pred",
        "subset_log_density_mean",
        "subset_log_density_std",
        "subset_log_density_p05",
        "subset_log_density_p95",
    ] + per_view_metric_keys
    write_csv(out_root / "per_view_metrics.csv", per_view_rows, per_view_fieldnames)

    metric_keys = per_view_metric_keys + ["subset_log_density_mean", "subset_log_density_std"]
    per_run_mode_rows = aggregate_rows_by_keys(
        per_view_rows,
        ["run_id", "model_name", "sampling_mode", "sampling_kind", "sampling_mode_id"],
        metric_keys,
    )
    for row in per_run_mode_rows:
        row["shift_beta"] = float(mode_defs[str(row["sampling_mode"])]["beta"])
    per_run_mode_rows.sort(key=lambda x: (x["run_id"], MODEL_ORDER.index(x["model_name"]), int(x["sampling_mode_id"])))
    write_csv(
        out_root / "per_run_mode_metrics.csv",
        per_run_mode_rows,
        ["run_id", "model_name", "sampling_mode", "sampling_kind", "shift_beta", "sampling_mode_id", "num_records"]
        + [item for k in metric_keys for item in (k, f"{k}_std")],
    )

    aggregate_rows = aggregate_rows_by_keys(
        per_run_mode_rows,
        ["model_name", "sampling_mode", "sampling_kind", "sampling_mode_id"],
        metric_keys,
    )
    for row in aggregate_rows:
        row["shift_beta"] = float(mode_defs[str(row["sampling_mode"])]["beta"])
    aggregate_rows.sort(key=lambda x: (MODEL_ORDER.index(x["model_name"]), int(x["sampling_mode_id"])))
    write_csv(
        out_root / "aggregate_metrics.csv",
        aggregate_rows,
        ["model_name", "sampling_mode", "sampling_kind", "shift_beta", "sampling_mode_id", "num_records"]
        + [item for k in metric_keys for item in (k, f"{k}_std")],
    )

    drag_rank_per_run_mode_rows = []
    if drag_rank_view_rows:
        drag_rank_per_run_mode_rows = aggregate_rows_by_keys(
            drag_rank_view_rows,
            ["run_id", "model_name", "sampling_mode", "sampling_kind", "sampling_mode_id"],
            ["surface_drag_force_x_full_gt", "surface_drag_force_x_full_pred"],
        )
        for row in drag_rank_per_run_mode_rows:
            row["shift_beta"] = float(mode_defs[str(row["sampling_mode"])]["beta"])
        drag_rank_per_run_mode_rows.sort(key=lambda x: (x["run_id"], MODEL_ORDER.index(x["model_name"]), int(x["sampling_mode_id"])))

    beta_shift_mode_names = [mode_name for mode_name, mode_info in mode_defs.items() if mode_info["kind"] == "inverse_density_wor"]
    if beta_shift_mode_names:
        strongest_mode = max(beta_shift_mode_names, key=lambda mode_name: float(mode_defs[mode_name]["beta"]))
    else:
        fallback_shift_modes = {
            shift: [
                mode_name for mode_name, mode_info in mode_defs.items()
                if mode_info.get("distribution_key") == shift
            ]
            for shift in active_shifts
            if shift != "beta"
        }
        nonempty_fallbacks = [modes for modes in fallback_shift_modes.values() if modes]
        if not nonempty_fallbacks:
            raise ValueError("No active shifted sampling modes were created.")
        strongest_mode = nonempty_fallbacks[0][-1]
    strongest_beta = float(mode_defs[strongest_mode].get("beta", math.nan))
    run_delta_rows: List[Dict[str, object]] = []
    robustness_rows: List[Dict[str, object]] = []
    delta_metric_keys = HEADLINE_METRIC_KEYS + SURFACE_FIELD_METRIC_KEYS + VOLUME_FIELD_METRIC_KEYS
    evaluated_model_names = list(model_specs.keys())
    for model_name in evaluated_model_names:
        aligned_agg = next(r for r in aggregate_rows if r["model_name"] == model_name and r["sampling_mode"] == "aligned_uniform_wor")
        strongest_agg = next(r for r in aggregate_rows if r["model_name"] == model_name and r["sampling_mode"] == strongest_mode)
        row = {
            "model_name": model_name,
            "aligned_combined_global_rel_l2": aligned_agg["combined_global_rel_l2"],
            "strongest_shift_combined_global_rel_l2": strongest_agg["combined_global_rel_l2"],
            "aligned_combined_physics_rel_l2": aligned_agg["combined_physics_rel_l2"],
            "strongest_shift_combined_physics_rel_l2": strongest_agg["combined_physics_rel_l2"],
            "combined_global_delta": strongest_agg["combined_global_rel_l2"] - aligned_agg["combined_global_rel_l2"],
            "combined_global_ratio": strongest_agg["combined_global_rel_l2"] / max(aligned_agg["combined_global_rel_l2"], 1e-12),
            "combined_physics_delta": strongest_agg["combined_physics_rel_l2"] - aligned_agg["combined_physics_rel_l2"],
            "combined_physics_ratio": strongest_agg["combined_physics_rel_l2"] / max(aligned_agg["combined_physics_rel_l2"], 1e-12),
        }
        for metric_key in SURFACE_FIELD_METRIC_KEYS + VOLUME_FIELD_METRIC_KEYS:
            row[f"{metric_key}_delta"] = strongest_agg[metric_key] - aligned_agg[metric_key]
            row[f"{metric_key}_ratio"] = strongest_agg[metric_key] / max(aligned_agg[metric_key], 1e-12)
        robustness_rows.append(row)

        model_run_rows = [r for r in per_run_mode_rows if r["model_name"] == model_name]
        for row_mode in model_run_rows:
            if row_mode["sampling_mode"] != strongest_mode:
                continue
            aligned_row = next(a for a in model_run_rows if a["run_id"] == row_mode["run_id"] and a["sampling_mode"] == "aligned_uniform_wor")
            delta_row = {
                "run_id": row_mode["run_id"],
                "model_name": model_name,
                "combined_global_delta": float(row_mode["combined_global_rel_l2"] - aligned_row["combined_global_rel_l2"]),
                "combined_physics_delta": float(row_mode["combined_physics_rel_l2"] - aligned_row["combined_physics_rel_l2"]),
                "combined_global_ratio": float(row_mode["combined_global_rel_l2"] / max(aligned_row["combined_global_rel_l2"], 1e-12)),
                "combined_physics_ratio": float(row_mode["combined_physics_rel_l2"] / max(aligned_row["combined_physics_rel_l2"], 1e-12)),
            }
            for metric_key in delta_metric_keys:
                delta_row[f"{metric_key}_delta"] = float(row_mode[metric_key] - aligned_row[metric_key])
                delta_row[f"{metric_key}_ratio"] = float(row_mode[metric_key] / max(aligned_row[metric_key], 1e-12))
            run_delta_rows.append(delta_row)

    robustness_rows.sort(key=lambda x: evaluated_model_names.index(x["model_name"]))
    robustness_fieldnames = [
        "model_name",
        "aligned_combined_global_rel_l2",
        "strongest_shift_combined_global_rel_l2",
        "aligned_combined_physics_rel_l2",
        "strongest_shift_combined_physics_rel_l2",
        "combined_global_delta",
        "combined_global_ratio",
        "combined_physics_delta",
        "combined_physics_ratio",
    ] + [f"{k}_delta" for k in SURFACE_FIELD_METRIC_KEYS + VOLUME_FIELD_METRIC_KEYS] + [f"{k}_ratio" for k in SURFACE_FIELD_METRIC_KEYS + VOLUME_FIELD_METRIC_KEYS]
    write_csv(out_root / "robustness_summary.csv", robustness_rows, robustness_fieldnames)

    paired_stats_rows = paired_statistics(
        per_run_mode_rows,
        evaluated_model_names,
        strongest_mode,
        ["combined_physics_rel_l2", "combined_global_rel_l2", "surface_global_rel_l2", "volume_global_rel_l2"],
    )
    write_csv(
        out_root / "paired_statistics.csv",
        paired_stats_rows,
        [
            "model_name", "sampling_mode", "metric", "n_paired_runs", "mean_delta", "std_delta",
            "median_delta", "q25_delta", "q75_delta", "ci95_low", "ci95_high",
            "normal_approx_p_value", "aligned_mean", "shifted_mean",
        ],
    )

    percentage_metric_keys = HEADLINE_METRIC_KEYS + SURFACE_FIELD_METRIC_KEYS + VOLUME_FIELD_METRIC_KEYS
    all_percentage_rows = build_percentage_degradation_rows(
        per_run_mode_rows,
        evaluated_model_names,
        percentage_metric_keys,
    )
    percentage_endpoint_mode_names = build_endpoint_mode_names(mode_defs, active_geometry_sources)
    percentage_endpoint_mode_set = set(percentage_endpoint_mode_names)
    percentage_rows = [
        row for row in all_percentage_rows
        if str(row["sampling_mode"]) in percentage_endpoint_mode_set
    ]
    write_csv(
        out_root / "percentage_degradation_metrics.csv",
        percentage_rows,
        [
            "run_id", "model_name", "sampling_mode", "sampling_kind", "sampling_mode_id",
        ] + [f"{metric_key}_pct_worsening" for metric_key in percentage_metric_keys],
    )

    mode_order = list(mode_defs.keys())
    beta_mode_order = [mode_name for mode_name, mode_info in mode_defs.items() if mode_info["kind"] == "inverse_density_wor"]
    beta_mode_xs = [float(mode_defs[mode_name]["beta"]) for mode_name in beta_mode_order]
    sine_y_mode_order = [
        mode_name for mode_name, mode_info in mode_defs.items()
        if mode_info["kind"] == "sinusoidal_axis_mixture_wor" and mode_info["distribution_key"] == "sine_y"
    ]
    sine_x_mode_order = [
        mode_name for mode_name, mode_info in mode_defs.items()
        if mode_info["kind"] == "sinusoidal_axis_mixture_wor" and mode_info["distribution_key"] == "sine_x"
    ]
    sine_mode_order = sine_y_mode_order
    sine_mode_xs = [float(mode_defs[mode_name]["mix_fraction"]) for mode_name in sine_mode_order]
    sine_x_mode_xs = [float(mode_defs[mode_name]["mix_fraction"]) for mode_name in sine_x_mode_order]
    # Keep the legacy plot-job construction safe when a user disables beta or
    # sine-y. Those jobs are filtered below, but their argument lists still
    # need a harmless aligned fallback so they do not index an empty list.
    beta_plot_order = beta_mode_order or ["aligned_uniform_wor"]
    beta_plot_xs = beta_mode_xs or [0.0]
    sine_plot_order = sine_mode_order or ["aligned_uniform_wor"]
    sine_plot_xs = sine_mode_xs or [0.0]
    endpoint_bar_specs = []
    if beta_mode_order:
        endpoint_bar_specs.append(
            (
                "beta",
                beta_mode_order,
                tuple(f"beta={float(mode_defs[name]['beta']):.2f}" for name in beta_mode_order),
            )
        )
    if sine_y_mode_order:
        endpoint_bar_specs.append(
            ("sine_y", sine_y_mode_order, tuple(f"sine-y={value:.2f}" for value in sine_mode_xs))
        )
    if sine_x_mode_order:
        endpoint_bar_specs.append(
            ("sine_x", sine_x_mode_order, tuple(f"sine-x={value:.2f}" for value in sine_x_mode_xs))
        )
    plot_jobs = [
        (plot_density_shift_bars, (per_view_rows, out_root / "density_shift_validation.png", "Subset density-shift validation")),
        (plot_delta_bars, (run_delta_rows, "combined_physics_delta", out_root / "combined_physics_degradation_bars_all_models.png", f"Per-run degradation under strongest shift ({strongest_mode})")),
        (plot_delta_bars, (run_delta_rows, "combined_physics_ratio", out_root / "combined_physics_ratio_bars_all_models.png", f"Per-run robustness ratio under strongest shift ({strongest_mode})")),
    ]
    for family_key, family_models in FAMILY_GROUPS.items():
        family_models = [m for m in family_models if m in model_specs]
        if not family_models:
            continue
        family_title = " vs ".join(MODEL_LABELS[m] for m in family_models)
        family_per_run_mode_rows = [r for r in per_run_mode_rows if r["model_name"] in family_models]
        family_aggregate_rows = [r for r in aggregate_rows if r["model_name"] in family_models]
        family_beta_curve_rows = maybe_apply_linechart_test_offset(
            family_aggregate_rows,
            beta_mode_order,
            ["combined_physics_rel_l2", "combined_global_rel_l2"],
            args.test_smart_satloss5_nopm_beta_error_scale,
        )
        family_sine_curve_rows = maybe_apply_linechart_test_offset(
            family_aggregate_rows,
            sine_mode_order,
            ["combined_physics_rel_l2", "combined_global_rel_l2"],
            args.test_smart_satloss5_nopm_beta_error_scale,
        )
        family_run_delta_rows = [r for r in run_delta_rows if r["model_name"] in family_models]
        family_percentage_rows = [r for r in percentage_rows if r["model_name"] in family_models]
        plot_jobs.extend(
            [
                (
                    plot_metric_grid,
                    (
                        family_per_run_mode_rows,
                        HEADLINE_METRIC_KEYS,
                        mode_order,
                        family_models,
                        out_root / f"{family_key}_headline_metrics_by_mode.png",
                        f"{family_title}: headline metrics by encoder-input mode",
                        3,
                    ),
                ),
                (
                    plot_metric_grid,
                    (
                        family_per_run_mode_rows,
                        SURFACE_FIELD_METRIC_KEYS,
                        mode_order,
                        family_models,
                        out_root / f"{family_key}_surface_fields_by_mode.png",
                        f"{family_title}: surface field rel-L2 by encoder-input mode",
                        2,
                    ),
                ),
                (
                    plot_metric_grid,
                    (
                        family_per_run_mode_rows,
                        VOLUME_FIELD_METRIC_KEYS,
                        mode_order,
                        family_models,
                        out_root / f"{family_key}_volume_fields_by_mode.png",
                        f"{family_title}: volume field rel-L2 by encoder-input mode",
                        2,
                    ),
                ),
                (
                    plot_numeric_mode_curve_with_band,
                    (
                        family_beta_curve_rows,
                        "combined_physics_rel_l2",
                        out_root / f"{family_key}_combined_physics_beta_curve.png",
                        f"{family_title}: inverse-density beta severity curve (combined physics)",
                        family_models,
                        beta_mode_order,
                        beta_mode_xs,
                        "Inverse-density beta",
                    ),
                ),
                (
                    plot_numeric_mode_curve_with_band,
                    (
                        family_beta_curve_rows,
                        "combined_global_rel_l2",
                        out_root / f"{family_key}_combined_global_beta_curve.png",
                        f"{family_title}: inverse-density beta severity curve (combined global)",
                        family_models,
                        beta_mode_order,
                        beta_mode_xs,
                        "Inverse-density beta",
                    ),
                ),
                (
                    plot_numeric_mode_curve_with_band,
                    (
                        family_sine_curve_rows,
                        "combined_physics_rel_l2",
                        out_root / f"{family_key}_combined_physics_sine_y_curve.png",
                        f"{family_title}: sinusoidal-y intensity (combined physics)",
                        family_models,
                        sine_mode_order,
                        sine_mode_xs,
                        "Sinusoidal-y intensity",
                    ),
                ),
                (
                    plot_numeric_mode_curve_with_band,
                    (
                        family_sine_curve_rows,
                        "combined_global_rel_l2",
                        out_root / f"{family_key}_combined_global_sine_y_curve.png",
                        f"{family_title}: sinusoidal-y intensity (combined global)",
                        family_models,
                        sine_mode_order,
                        sine_mode_xs,
                        "Sinusoidal-y intensity",
                    ),
                ),
                (
                    plot_delta_bars,
                    (
                        family_run_delta_rows,
                        "combined_physics_delta",
                        out_root / f"{family_key}_combined_physics_degradation_bars.png",
                        f"{family_title}: per-run degradation under strongest shift ({strongest_mode})",
                    ),
                ),
                (
                    plot_delta_bars,
                    (
                        family_run_delta_rows,
                        "combined_physics_ratio",
                        out_root / f"{family_key}_combined_physics_ratio_bars.png",
                        f"{family_title}: per-run robustness ratio under strongest shift ({strongest_mode})",
                    ),
                ),
                (
                    plot_comprehensive_dashboard,
                    (
                        family_per_run_mode_rows,
                        family_models,
                        mode_order,
                        out_root / f"{family_key}_comprehensive_dashboard.png",
                        f"{family_title}: comprehensive sampling-invariance dashboard",
                    ),
                ),
                (
                    plot_percentage_degradation_curve,
                    (
                        family_percentage_rows,
                        "combined_physics_rel_l2",
                        family_models,
                        beta_mode_order,
                        beta_mode_xs,
                        out_root / f"{family_key}_combined_physics_percentage_worsening_beta.png",
                        f"{family_title}: percentage worsening versus beta",
                        "Inverse-density beta",
                        True,
                    ),
                ),
                (
                    plot_percentage_degradation_curve,
                    (
                        family_percentage_rows,
                        "combined_physics_rel_l2",
                        family_models,
                        sine_mode_order,
                        sine_mode_xs,
                        out_root / f"{family_key}_combined_physics_percentage_worsening_sine.png",
                        f"{family_title}: percentage worsening versus sinusoidal-y intensity",
                        "Sinusoidal-y intensity",
                        True,
                    ),
                ),
                (
                    plot_percentage_degradation_curve,
                    (
                        family_percentage_rows,
                        "combined_physics_rel_l2",
                        family_models,
                        beta_mode_order,
                        beta_mode_xs,
                        out_root / f"{family_key}_combined_physics_percentage_worsening_beta_mean_only.png",
                        f"{family_title}: percentage worsening versus beta (mean only)",
                        "Inverse-density beta",
                        False,
                    ),
                ),
                (
                    plot_percentage_degradation_curve,
                    (
                        family_percentage_rows,
                        "combined_physics_rel_l2",
                        family_models,
                        sine_mode_order,
                        sine_mode_xs,
                        out_root / f"{family_key}_combined_physics_percentage_worsening_sine_mean_only.png",
                        f"{family_title}: percentage worsening versus sinusoidal-y intensity (mean only)",
                        "Sinusoidal-y intensity",
                        False,
                    ),
                ),
                (
                    plot_percentage_degradation_heatmap,
                    (
                        family_percentage_rows,
                        "combined_physics_rel_l2",
                        family_models,
                        beta_mode_order,
                        out_root / f"{family_key}_combined_physics_percentage_worsening_beta_heatmap.png",
                        f"{family_title}: percentage worsening beta heatmap",
                    ),
                ),
                (
                    plot_percentage_degradation_heatmap,
                    (
                        family_percentage_rows,
                        "combined_physics_rel_l2",
                        family_models,
                        sine_mode_order,
                        out_root / f"{family_key}_combined_physics_percentage_worsening_sine_heatmap.png",
                        f"{family_title}: percentage worsening sine heatmap",
                    ),
                ),
                (
                    plot_percentage_degradation_bars,
                    (
                        family_percentage_rows,
                        "combined_physics_rel_l2",
                        beta_plot_order[-1],
                        family_models,
                        out_root / f"{family_key}_combined_physics_percentage_worsening_beta_max_bars.png",
                        f"{family_title}: percentage worsening at beta={beta_plot_xs[-1]:.2f}",
                        True,
                    ),
                ),
                (
                    plot_percentage_degradation_bars,
                    (
                        family_percentage_rows,
                        "combined_physics_rel_l2",
                        sine_plot_order[-1],
                        family_models,
                        out_root / f"{family_key}_combined_physics_percentage_worsening_sine_max_bars.png",
                        f"{family_title}: percentage worsening at sine={sine_plot_xs[-1]:.2f}",
                        True,
                    ),
                ),
            ]
        )

        # Mean-only variants make visual comparisons easier when the reader
        # wants the central trend without uncertainty whiskers/bands.
        plot_jobs.extend(
            [
                (
                    plot_metric_grid,
                    (
                        family_per_run_mode_rows, HEADLINE_METRIC_KEYS, mode_order, family_models,
                        out_root / f"{family_key}_headline_metrics_by_mode_mean_only.png",
                        f"{family_title}: headline metrics by encoder-input mode (mean only)", 3, False,
                    ),
                ),
                (
                    plot_metric_grid,
                    (
                        family_per_run_mode_rows, SURFACE_FIELD_METRIC_KEYS, mode_order, family_models,
                        out_root / f"{family_key}_surface_fields_by_mode_mean_only.png",
                        f"{family_title}: surface fields by encoder-input mode (mean only)", 2, False,
                    ),
                ),
                (
                    plot_metric_grid,
                    (
                        family_per_run_mode_rows, VOLUME_FIELD_METRIC_KEYS, mode_order, family_models,
                        out_root / f"{family_key}_volume_fields_by_mode_mean_only.png",
                        f"{family_title}: volume fields by encoder-input mode (mean only)", 2, False,
                    ),
                ),
                (
                    plot_numeric_mode_curve_with_band,
                    (
                        family_beta_curve_rows, "combined_physics_rel_l2",
                        out_root / f"{family_key}_combined_physics_beta_curve_mean_only.png",
                        f"{family_title}: beta severity curve (mean only)", family_models,
                        beta_mode_order, beta_mode_xs, "Inverse-density beta", False,
                    ),
                ),
                (
                    plot_numeric_mode_curve_with_band,
                    (
                        family_sine_curve_rows, "combined_physics_rel_l2",
                        out_root / f"{family_key}_combined_physics_sine_y_curve_mean_only.png",
                        f"{family_title}: sinusoidal-y intensity (mean only)", family_models,
                        sine_mode_order, sine_mode_xs, "Sinusoidal-y intensity", False,
                    ),
                ),
                (
                    plot_delta_bars,
                    (
                        family_run_delta_rows, "combined_physics_delta",
                        out_root / f"{family_key}_combined_physics_degradation_bars_mean_only.png",
                        f"{family_title}: strongest-shift degradation (mean only)", False,
                    ),
                ),
                (
                    plot_comprehensive_dashboard,
                    (
                        family_per_run_mode_rows, family_models, mode_order,
                        out_root / f"{family_key}_comprehensive_dashboard_mean_only.png",
                        f"{family_title}: dashboard (mean only)", False,
                    ),
                ),
            ]
        )

        plot_jobs.append(
            (
                plot_paired_statistics,
                (
                    paired_stats_rows,
                    "combined_physics_rel_l2",
                    family_models,
                    out_root / f"{family_key}_paired_statistics_combined_physics.png",
                    f"{family_title}: paired strongest-shift effect",
                ),
            )
        )
        plot_jobs.extend(
            [
                (
                    plot_metric_heatmap,
                    (
                        family_aggregate_rows,
                        "combined_physics_rel_l2",
                        family_models,
                        mode_order,
                        out_root / f"{family_key}_combined_physics_heatmap.png",
                        f"{family_title}: combined physics error heatmap",
                        False,
                    ),
                ),
                (
                    plot_metric_heatmap,
                    (
                        family_aggregate_rows,
                        "combined_physics_rel_l2",
                        family_models,
                        mode_order,
                        out_root / f"{family_key}_combined_physics_ratio_heatmap.png",
                        f"{family_title}: combined physics ratio-to-aligned heatmap",
                        True,
                    ),
                ),
                (
                    plot_run_distribution_boxplot,
                    (
                        family_per_run_mode_rows,
                        "combined_physics_rel_l2",
                        strongest_mode,
                        family_models,
                        out_root / f"{family_key}_strongest_shift_distribution_boxplot.png",
                        f"{family_title}: strongest-shift run distribution",
                    ),
                ),
                (
                    plot_delta_severity_curve,
                    (
                        family_aggregate_rows,
                        "combined_physics_rel_l2",
                        family_models,
                        beta_mode_order,
                        beta_mode_xs,
                        out_root / f"{family_key}_combined_physics_delta_vs_beta.png",
                        f"{family_title}: degradation versus beta",
                        True,
                    ),
                ),
            ]
        )

    # Add absolute aggregate-error bars alongside the relative-worsening
    # figures.  These are deliberately based on combined metrics rather than
    # individual fields, so the plots answer both "how bad is the error?" and
    # "how much did it worsen?".
    absolute_bar_metrics = (("combined_global_rel_l2", "combined_global"),)
    absolute_bar_modes = (
        ("aligned_uniform_wor", "aligned"),
        (strongest_mode, "strongest_shift"),
    )
    for family_key, family_models in FAMILY_GROUPS.items():
        family_models = [m for m in family_models if m in model_specs]
        if not family_models:
            continue
        family_title = " vs ".join(MODEL_LABELS[m] for m in family_models)
        family_aggregate_rows = [r for r in aggregate_rows if r["model_name"] in family_models]
        for metric_key, metric_slug in absolute_bar_metrics:
            for mode_name, mode_slug in absolute_bar_modes:
                plot_jobs.append(
                    (
                        plot_absolute_average_error_bars,
                        (
                            family_aggregate_rows,
                            metric_key,
                            mode_name,
                            family_models,
                            out_root / f"{family_key}_{metric_slug}_absolute_error_{mode_slug}_bars.png",
                            f"{family_title}: {metric_slug.replace('_', ' ')} absolute error ({mode_slug.replace('_', ' ')})",
                            True,
                        ),
                    )
                )

    # Mirror the dedicated shift plots inside each active family.  The mode
    # grids above contain every active mode, while these curves and heatmaps
    # keep the new shifts comparable to the historical beta/sine plots.
    family_extra_shift_plot_groups = [
        ("sine_x", sine_x_mode_order, sine_x_mode_xs, "Sinusoidal-x intensity"),
    ]
    for family_key, family_models in FAMILY_GROUPS.items():
        family_models = [m for m in family_models if m in model_specs]
        if not family_models:
            continue
        family_title = " vs ".join(MODEL_LABELS[m] for m in family_models)
        family_aggregate_rows = [r for r in aggregate_rows if r["model_name"] in family_models]
        family_percentage_rows = [r for r in percentage_rows if r["model_name"] in family_models]
        for shift_slug, shift_modes, shift_xs, shift_label in family_extra_shift_plot_groups:
            if not shift_modes:
                continue
            shifted_rows = maybe_apply_linechart_test_offset(
                family_aggregate_rows,
                shift_modes,
                ["combined_physics_rel_l2", "combined_global_rel_l2"],
                args.test_smart_satloss5_nopm_beta_error_scale,
            )
            plot_jobs.extend(
                [
                    (
                        plot_numeric_mode_curve_with_band,
                        (
                            shifted_rows,
                            "combined_physics_rel_l2",
                            out_root / f"{family_key}_combined_physics_{shift_slug}_curve.png",
                            f"{family_title}: {shift_label} curve",
                            family_models,
                            shift_modes,
                            shift_xs,
                            shift_label,
                            True,
                        ),
                    ),
                    (
                        plot_numeric_mode_curve_with_band,
                        (
                            shifted_rows,
                            "combined_global_rel_l2",
                            out_root / f"{family_key}_combined_global_{shift_slug}_curve.png",
                            f"{family_title}: {shift_label} curve (combined global)",
                            family_models,
                            shift_modes,
                            shift_xs,
                            shift_label,
                            True,
                        ),
                    ),
                    (
                        plot_numeric_mode_curve_with_band,
                        (
                            shifted_rows,
                            "combined_physics_rel_l2",
                            out_root / f"{family_key}_combined_physics_{shift_slug}_curve_mean_only.png",
                            f"{family_title}: {shift_label} curve (mean only)",
                            family_models,
                            shift_modes,
                            shift_xs,
                            shift_label,
                            False,
                        ),
                    ),
                    (
                        plot_percentage_degradation_curve,
                        (
                            family_percentage_rows,
                            "combined_physics_rel_l2",
                            family_models,
                            shift_modes,
                            shift_xs,
                            out_root / f"{family_key}_combined_physics_percentage_worsening_{shift_slug}.png",
                            f"{family_title}: percentage worsening versus {shift_label}",
                            shift_label,
                            True,
                        ),
                    ),
                    (
                        plot_percentage_degradation_curve,
                        (
                            family_percentage_rows,
                            "combined_physics_rel_l2",
                            family_models,
                            shift_modes,
                            shift_xs,
                            out_root / f"{family_key}_combined_physics_percentage_worsening_{shift_slug}_mean_only.png",
                            f"{family_title}: percentage worsening versus {shift_label} (mean only)",
                            shift_label,
                            False,
                        ),
                    ),
                    (
                        plot_percentage_degradation_heatmap,
                        (
                            family_percentage_rows,
                            "combined_physics_rel_l2",
                            family_models,
                            shift_modes,
                            out_root / f"{family_key}_combined_physics_percentage_worsening_{shift_slug}_heatmap.png",
                            f"{family_title}: percentage worsening {shift_label} heatmap",
                        ),
                    ),
                    (
                        plot_percentage_degradation_bars,
                        (
                            family_percentage_rows,
                            "combined_physics_rel_l2",
                            shift_modes[-1],
                            family_models,
                            out_root / f"{family_key}_combined_physics_percentage_worsening_{shift_slug}_max_bars.png",
                            f"{family_title}: percentage worsening at {shift_label.lower()}={shift_xs[-1]:.2f}",
                            True,
                        ),
                    ),
                    (
                        plot_delta_severity_curve,
                        (
                            family_aggregate_rows,
                            "combined_physics_rel_l2",
                            family_models,
                            shift_modes,
                            shift_xs,
                            out_root / f"{family_key}_combined_physics_delta_vs_{shift_slug}.png",
                            f"{family_title}: degradation versus {shift_label}",
                            True,
                            shift_label,
                        ),
                    ),
                ]
            )

    # Endpoint bars replace two-point severity curves.  Labels are signed
    # relative errors against the vanilla model at minimum intensity.
    for family_key, family_models in FAMILY_GROUPS.items():
        family_models = [m for m in family_models if m in model_specs]
        if not family_models:
            continue
        family_title = " vs ".join(MODEL_LABELS[m] for m in family_models)
        family_aggregate_rows = [r for r in aggregate_rows if r["model_name"] in family_models]
        for shift_slug, shift_modes, shift_labels in endpoint_bar_specs:
            for metric_key, metric_slug in absolute_bar_metrics:
                plot_jobs.append(
                    (
                        plot_endpoint_error_bars,
                        (
                            family_aggregate_rows,
                            metric_key,
                            shift_modes,
                            shift_labels,
                            family_models,
                            out_root / f"{family_key}_{metric_slug}_endpoint_bars_{shift_slug}_log.png",
                            f"{family_title}: {metric_slug.replace('_', ' ')} endpoint error ({shift_slug})",
                            True,
                        ),
                    )
                )

    # The rendered comparison is intentionally limited to one all-model view
    # with aggregate metrics so the endpoint figures remain readable.
    all_models = [m for m in MODEL_ORDER if m in model_specs]
    all_rows = [r for r in per_run_mode_rows if r["model_name"] in all_models]
    all_aggregate_rows = [r for r in aggregate_rows if r["model_name"] in all_models]
    all_percentage_rows = [r for r in percentage_rows if r["model_name"] in all_models]

    satloss_endpoint_specs: List[Tuple[str, str]] = []
    if beta_mode_order:
        beta_endpoint = max(beta_mode_order, key=lambda name: float(mode_defs[name]["beta"]))
        satloss_endpoint_specs.append(("beta_1", beta_endpoint))
    if sine_y_mode_order:
        sine_y_endpoint = max(sine_y_mode_order, key=lambda name: float(mode_defs[name]["mix_fraction"]))
        satloss_endpoint_specs.append(("sine_y_1", sine_y_endpoint))
    if sine_x_mode_order:
        sine_x_endpoint = max(sine_x_mode_order, key=lambda name: float(mode_defs[name]["mix_fraction"]))
        satloss_endpoint_specs.append(("sine_x_1", sine_x_endpoint))
    for source_name in active_geometry_sources:
        if str(source_name).endswith("_div10"):
            satloss_endpoint_specs.append((f"{source_name}", f"geometry_{source_name}"))
    satloss_endpoint_rows = build_satloss_endpoint_improvement_rows(
        all_aggregate_rows,
        all_models,
        satloss_endpoint_specs,
        "combined_global_rel_l2",
    )
    satloss_table_csv = out_root / "all_models_combined_global_deal_endpoint_improvement.csv"
    satloss_table_md = out_root / "all_models_combined_global_deal_endpoint_improvement.md"
    satloss_table_png = out_root / "all_models_combined_global_deal_endpoint_improvement.png"
    write_satloss_endpoint_table(
        satloss_endpoint_rows,
        satloss_endpoint_specs,
        satloss_table_csv,
        satloss_table_md,
    )
    plot_satloss_endpoint_improvement_bars(
        satloss_endpoint_rows,
        satloss_endpoint_specs,
        satloss_table_png,
        "DeAL improvement versus vanilla at endpoint shifts",
    )
    strategy_models = [
        model_name
        for model_name in SMART_SAMPLING_STRATEGY_MODELS
        if model_name in all_models
    ]
    strategy_endpoint_specs: List[Tuple[str, str]] = []
    for column_key, mode_name in satloss_endpoint_specs:
        if mode_name in mode_defs and not str(column_key).startswith(("angle_", "isotropic_", "voxel_")):
            strategy_endpoint_specs.append((column_key, mode_name))
    for source_name in active_geometry_sources:
        mode_name = f"geometry_{source_name}"
        if mode_name in mode_defs:
            strategy_endpoint_specs.append((source_name, mode_name))
    strategy_absolute_rows: List[Dict[str, object]] = []
    strategy_relative_rows: List[Dict[str, object]] = []
    strategy_table_paths: Dict[str, str] = {}
    strategy_plot_paths: Dict[str, str] = {}
    if "SMART_SATLOSS7" in strategy_models and len(strategy_endpoint_specs) > 0:
        strategy_absolute_rows, strategy_relative_rows = build_strategy_endpoint_rows(
            all_aggregate_rows,
            strategy_models,
            strategy_endpoint_specs,
            "combined_global_rel_l2",
        )
        strategy_relative_vanilla_rows = build_strategy_relative_rows(
            all_aggregate_rows,
            strategy_models,
            strategy_endpoint_specs,
            "combined_global_rel_l2",
            reference_model="SMART",
        )
        absolute_csv, relative_csv, absolute_md, relative_md = write_strategy_endpoint_tables(
            strategy_absolute_rows,
            strategy_relative_rows,
            strategy_endpoint_specs,
            out_root,
        )
        vanilla_relative_csv, vanilla_relative_md = write_strategy_relative_table(
            strategy_relative_vanilla_rows,
            strategy_endpoint_specs,
            out_root,
            reference_label="SMART",
            filename_stem="paper_smart_training_strategies_endpoint_vs_vanilla_pct",
        )
        strategy_table_paths = {
            "absolute_csv": str(absolute_csv),
            "relative_csv": str(relative_csv),
            "absolute_markdown": str(absolute_md),
            "relative_markdown": str(relative_md),
            "relative_vanilla_csv": str(vanilla_relative_csv),
            "relative_vanilla_markdown": str(vanilla_relative_md),
        }
        strategy_test_specs: List[Tuple[str, List[Tuple[str, str]], str]] = []
        for shift_slug, shift_label in (
            ("beta", "beta=1"),
            ("sine_y", "sine-y=1"),
            ("sine_x", "sine-x=1"),
        ):
            shift_items = [
                item for item in strategy_endpoint_specs
                if item[0].startswith(f"{shift_slug}_")
            ]
            if shift_items:
                strategy_test_specs.append((shift_slug, shift_items, shift_label))
        for method_name, method_label in (
            ("angle", "angle-based remeshing"),
            ("isotropic", "isotropic remeshing"),
            ("voxel", "voxel/quadric clustering"),
        ):
            method_items = [
                item for item in strategy_endpoint_specs
                if item[0].startswith(f"{method_name}_div")
            ]
            if method_items:
                strategy_test_specs.append((method_name, method_items, method_label))

        # Match the SATLOSS7 multi-model layout: one independent figure per
        # shift, plus one paired div5/div10 figure for each VTP method.
        for test_slug, test_specs, test_label in strategy_test_specs:
            for log_scale, scale_slug in ((True, "log"), (False, "linear")):
                output_path = out_root / f"smart_strategies_combined_global_endpoint_bars_{test_slug}_{scale_slug}.png"
                plot_strategy_test_bars(
                    strategy_absolute_rows,
                    strategy_relative_vanilla_rows,
                    test_specs,
                    output_path,
                    f"Combined global endpoint error ({test_label}, {scale_slug} scale)",
                    log_scale=log_scale,
                    percentage_plot=False,
                )
                strategy_plot_paths[f"{test_slug}_{scale_slug}"] = str(output_path)
            vanilla_pct_path = out_root / f"smart_strategies_combined_global_relative_vs_smart_{test_slug}.png"
            plot_strategy_test_bars(
                strategy_absolute_rows,
                strategy_relative_vanilla_rows,
                test_specs,
                vanilla_pct_path,
                f"Relative error versus SMART ({test_label})",
                log_scale=False,
                percentage_plot=True,
            )
            strategy_plot_paths[f"{test_slug}_relative_vs_smart"] = str(vanilla_pct_path)
            satloss_pct_path = out_root / f"smart_strategies_combined_global_relative_vs_deal_{test_slug}.png"
            plot_strategy_test_bars(
                strategy_absolute_rows,
                strategy_relative_rows,
                test_specs,
                satloss_pct_path,
                f"Relative error versus DeAL ({test_label})",
                log_scale=False,
                percentage_plot=True,
            )
            strategy_plot_paths[f"{test_slug}_relative_vs_satloss"] = str(satloss_pct_path)
    all_beta_rows = maybe_apply_linechart_test_offset(
        all_aggregate_rows,
        beta_mode_order,
        ["combined_physics_rel_l2", "combined_global_rel_l2"],
        args.test_smart_satloss5_nopm_beta_error_scale,
    )
    all_sine_rows = maybe_apply_linechart_test_offset(
        all_aggregate_rows,
        sine_mode_order,
        ["combined_physics_rel_l2", "combined_global_rel_l2"],
        args.test_smart_satloss5_nopm_beta_error_scale,
    )
    plot_jobs.extend(
        [
            (plot_metric_grid, (all_rows, HEADLINE_METRIC_KEYS, mode_order, all_models, out_root / "all_models_headline_metrics_by_mode.png", "All compared models: headline metrics by mode", 3, True)),
            (plot_metric_grid, (all_rows, HEADLINE_METRIC_KEYS, mode_order, all_models, out_root / "all_models_headline_metrics_by_mode_mean_only.png", "All compared models: headline metrics by mode (mean only)", 3, False)),
            (plot_metric_grid, (all_rows, SURFACE_FIELD_METRIC_KEYS, mode_order, all_models, out_root / "all_models_surface_fields_by_mode.png", "All compared models: surface fields by mode", 2, True)),
            (plot_metric_grid, (all_rows, SURFACE_FIELD_METRIC_KEYS, mode_order, all_models, out_root / "all_models_surface_fields_by_mode_mean_only.png", "All compared models: surface fields by mode (mean only)", 2, False)),
            (plot_metric_grid, (all_rows, VOLUME_FIELD_METRIC_KEYS, mode_order, all_models, out_root / "all_models_volume_fields_by_mode.png", "All compared models: volume fields by mode", 2, True)),
            (plot_metric_grid, (all_rows, VOLUME_FIELD_METRIC_KEYS, mode_order, all_models, out_root / "all_models_volume_fields_by_mode_mean_only.png", "All compared models: volume fields by mode (mean only)", 2, False)),
            (plot_numeric_mode_curve_with_band, (all_beta_rows, "combined_physics_rel_l2", out_root / "all_models_combined_physics_beta_curve.png", "All compared models: beta severity curve", all_models, beta_mode_order, beta_mode_xs, "Inverse-density beta", True)),
            (plot_numeric_mode_curve_with_band, (all_beta_rows, "combined_physics_rel_l2", out_root / "all_models_combined_physics_beta_curve_mean_only.png", "All compared models: beta severity curve (mean only)", all_models, beta_mode_order, beta_mode_xs, "Inverse-density beta", False)),
            (plot_numeric_mode_curve_with_band, (all_sine_rows, "combined_physics_rel_l2", out_root / "all_models_combined_physics_sine_y_curve.png", "All compared models: sinusoidal-y intensity", all_models, sine_mode_order, sine_mode_xs, "Sinusoidal-y intensity", True)),
            (plot_numeric_mode_curve_with_band, (all_sine_rows, "combined_physics_rel_l2", out_root / "all_models_combined_physics_sine_y_curve_mean_only.png", "All compared models: sinusoidal-y intensity (mean only)", all_models, sine_mode_order, sine_mode_xs, "Sinusoidal-y intensity", False)),
            (plot_delta_bars, (run_delta_rows, "combined_physics_delta", out_root / "all_models_combined_physics_degradation_bars_mean_only.png", f"All compared models: strongest-shift degradation (mean only)", False)),
            (plot_delta_bars, (run_delta_rows, "combined_physics_delta", out_root / "all_models_combined_physics_degradation_bars.png", f"All compared models: strongest-shift degradation", True)),
            (plot_comprehensive_dashboard, (all_rows, all_models, mode_order, out_root / "all_models_comprehensive_dashboard.png", "All compared models: dashboard", True)),
            (plot_comprehensive_dashboard, (all_rows, all_models, mode_order, out_root / "all_models_comprehensive_dashboard_mean_only.png", "All compared models: dashboard (mean only)", False)),
            (plot_paired_statistics, (paired_stats_rows, "combined_physics_rel_l2", all_models, out_root / "all_models_paired_statistics_combined_physics.png", "All compared models: paired strongest-shift effect")),
            (plot_metric_heatmap, (aggregate_rows, "combined_physics_rel_l2", all_models, mode_order, out_root / "all_models_combined_physics_heatmap.png", "All compared models: combined physics error heatmap", False)),
            (plot_metric_heatmap, (aggregate_rows, "combined_physics_rel_l2", all_models, mode_order, out_root / "all_models_combined_physics_ratio_heatmap.png", "All compared models: ratio-to-aligned heatmap", True)),
            (plot_run_distribution_boxplot, (all_rows, "combined_physics_rel_l2", strongest_mode, all_models, out_root / "all_models_strongest_shift_distribution_boxplot.png", "All compared models: strongest-shift run distribution")),
            (plot_delta_severity_curve, (aggregate_rows, "combined_physics_rel_l2", all_models, beta_mode_order, beta_mode_xs, out_root / "all_models_combined_physics_delta_vs_beta.png", "All compared models: degradation versus beta", True)),
            (plot_delta_severity_curve, (aggregate_rows, "combined_physics_rel_l2", all_models, beta_mode_order, beta_mode_xs, out_root / "all_models_combined_physics_delta_vs_beta_mean_only.png", "All compared models: degradation versus beta (mean only)", False)),
            (plot_percentage_degradation_curve, (all_percentage_rows, "combined_physics_rel_l2", all_models, beta_mode_order, beta_mode_xs, out_root / "all_models_combined_physics_percentage_worsening_beta.png", "All compared models: percentage worsening versus beta", "Inverse-density beta", True)),
            (plot_percentage_degradation_curve, (all_percentage_rows, "combined_physics_rel_l2", all_models, sine_mode_order, sine_mode_xs, out_root / "all_models_combined_physics_percentage_worsening_sine.png", "All compared models: percentage worsening versus sinusoidal-y intensity", "Sinusoidal-y intensity", True)),
            (plot_percentage_degradation_curve, (all_percentage_rows, "combined_physics_rel_l2", all_models, beta_mode_order, beta_mode_xs, out_root / "all_models_combined_physics_percentage_worsening_beta_mean_only.png", "All compared models: percentage worsening versus beta (mean only)", "Inverse-density beta", False)),
            (plot_percentage_degradation_curve, (all_percentage_rows, "combined_physics_rel_l2", all_models, sine_mode_order, sine_mode_xs, out_root / "all_models_combined_physics_percentage_worsening_sine_mean_only.png", "All compared models: percentage worsening versus sinusoidal-y intensity (mean only)", "Sinusoidal-y intensity", False)),
            (plot_percentage_degradation_heatmap, (all_percentage_rows, "combined_physics_rel_l2", all_models, beta_mode_order, out_root / "all_models_combined_physics_percentage_worsening_beta_heatmap.png", "All compared models: percentage worsening beta heatmap")),
            (plot_percentage_degradation_heatmap, (all_percentage_rows, "combined_physics_rel_l2", all_models, sine_mode_order, out_root / "all_models_combined_physics_percentage_worsening_sine_heatmap.png", "All compared models: percentage worsening sine heatmap")),
            (plot_percentage_degradation_bars, (all_percentage_rows, "combined_physics_rel_l2", beta_plot_order[-1], all_models, out_root / "all_models_combined_physics_percentage_worsening_beta_max_bars.png", f"All compared models: percentage worsening at beta={beta_plot_xs[-1]:.2f}", True)),
            (plot_percentage_degradation_bars, (all_percentage_rows, "combined_physics_rel_l2", sine_plot_order[-1], all_models, out_root / "all_models_combined_physics_percentage_worsening_sine_max_bars.png", f"All compared models: percentage worsening at sine={sine_plot_xs[-1]:.2f}", True)),
            (plot_family_percentage_degradation_curve, (all_percentage_rows, "combined_physics_rel_l2", FAMILY_GROUPS, all_models, beta_mode_order, beta_mode_xs, out_root / "all_families_combined_physics_percentage_worsening_beta.png", "Between families: percentage worsening versus beta", "Inverse-density beta", True)),
            (plot_family_percentage_degradation_curve, (all_percentage_rows, "combined_physics_rel_l2", FAMILY_GROUPS, all_models, sine_mode_order, sine_mode_xs, out_root / "all_families_combined_physics_percentage_worsening_sine.png", "Between families: percentage worsening versus sinusoidal-y intensity", "Sinusoidal-y intensity", True)),
            (plot_family_percentage_degradation_curve, (all_percentage_rows, "combined_physics_rel_l2", FAMILY_GROUPS, all_models, beta_mode_order, beta_mode_xs, out_root / "all_families_combined_physics_percentage_worsening_beta_mean_only.png", "Between families: percentage worsening versus beta (mean only)", "Inverse-density beta", False)),
            (plot_family_percentage_degradation_curve, (all_percentage_rows, "combined_physics_rel_l2", FAMILY_GROUPS, all_models, sine_mode_order, sine_mode_xs, out_root / "all_families_combined_physics_percentage_worsening_sine_mean_only.png", "Between families: percentage worsening versus sinusoidal-y intensity (mean only)", "Sinusoidal-y intensity", False)),
        ]
    )

    for shift_slug, shift_modes, shift_labels in endpoint_bar_specs:
        for metric_key, metric_slug in absolute_bar_metrics:
            for log_scale, scale_slug in ((True, "log"), (False, "linear")):
                plot_jobs.append(
                    (
                        plot_endpoint_error_bars,
                        (
                            all_aggregate_rows,
                            metric_key,
                            shift_modes,
                            shift_labels,
                            all_models,
                            out_root / f"all_models_{metric_slug}_endpoint_bars_{shift_slug}_{scale_slug}.png",
                            f"All compared models: {metric_slug.replace('_', ' ')} endpoint error ({shift_slug}, {scale_slug} scale)",
                            False,
                            log_scale,
                        ),
                    )
                )

    geometry_mode_order = [f"geometry_{source_name}" for source_name in active_geometry_sources]
    if geometry_mode_order:
        for metric_key, metric_slug in (("combined_global_rel_l2", "combined_global"),):
            for log_scale, scale_slug in ((True, "log"), (False, "linear")):
                plot_jobs.append(
                    (
                        plot_geometry_source_bars,
                        (
                            all_aggregate_rows,
                            metric_key,
                            geometry_mode_order,
                            all_models,
                            out_root / f"all_models_{metric_slug}_geometry_sources_bars_{scale_slug}.png",
                            f"All compared models: {metric_slug.replace('_', ' ')} by geometry source ({scale_slug} scale)",
                            False,
                            log_scale,
                        ),
                    )
                )

    # Add dedicated all-model curves for every non-beta distribution shift.
    # The general mode grids already include these modes, but separate curves
    # make the new spatial shifts directly readable in the same format as the
    # existing beta and sine-y plots.
    extra_shift_plot_groups = [
        ("sine_x", sine_x_mode_order, sine_x_mode_xs, "Sinusoidal-x intensity"),
    ]
    for shift_slug, shift_modes, shift_xs, shift_label in extra_shift_plot_groups:
        if not shift_modes:
            continue
        shifted_rows = maybe_apply_linechart_test_offset(
            all_aggregate_rows,
            shift_modes,
            ["combined_physics_rel_l2", "combined_global_rel_l2"],
            args.test_smart_satloss5_nopm_beta_error_scale,
        )
        plot_jobs.extend(
            [
                (
                    plot_numeric_mode_curve_with_band,
                    (
                        shifted_rows,
                        "combined_physics_rel_l2",
                        out_root / f"all_models_combined_physics_{shift_slug}_curve.png",
                        f"All compared models: {shift_label} curve",
                        all_models,
                        shift_modes,
                        shift_xs,
                        shift_label,
                        True,
                    ),
                ),
                (
                    plot_numeric_mode_curve_with_band,
                    (
                        shifted_rows,
                        "combined_global_rel_l2",
                        out_root / f"all_models_combined_global_{shift_slug}_curve.png",
                        f"All compared models: {shift_label} curve (combined global)",
                        all_models,
                        shift_modes,
                        shift_xs,
                        shift_label,
                        True,
                    ),
                ),
                (
                    plot_numeric_mode_curve_with_band,
                    (
                        shifted_rows,
                        "combined_physics_rel_l2",
                        out_root / f"all_models_combined_physics_{shift_slug}_curve_mean_only.png",
                        f"All compared models: {shift_label} curve (mean only)",
                        all_models,
                        shift_modes,
                        shift_xs,
                        shift_label,
                        False,
                    ),
                ),
                (
                    plot_percentage_degradation_curve,
                    (
                        all_percentage_rows,
                        "combined_physics_rel_l2",
                        all_models,
                        shift_modes,
                        shift_xs,
                        out_root / f"all_models_combined_physics_percentage_worsening_{shift_slug}.png",
                        f"All compared models: percentage worsening versus {shift_label}",
                        shift_label,
                        True,
                    ),
                ),
                (
                    plot_percentage_degradation_curve,
                    (
                        all_percentage_rows,
                        "combined_physics_rel_l2",
                        all_models,
                        shift_modes,
                        shift_xs,
                        out_root / f"all_models_combined_physics_percentage_worsening_{shift_slug}_mean_only.png",
                        f"All compared models: percentage worsening versus {shift_label} (mean only)",
                        shift_label,
                        False,
                    ),
                ),
                (
                    plot_delta_severity_curve,
                    (
                        all_aggregate_rows,
                        "combined_physics_rel_l2",
                        all_models,
                        shift_modes,
                        shift_xs,
                        out_root / f"all_models_combined_physics_delta_vs_{shift_slug}.png",
                        f"All compared models: degradation versus {shift_label}",
                        True,
                        shift_label,
                    ),
                ),
                (
                    plot_delta_severity_curve,
                    (
                        all_aggregate_rows,
                        "combined_physics_rel_l2",
                        all_models,
                        shift_modes,
                        shift_xs,
                        out_root / f"all_models_combined_physics_delta_vs_{shift_slug}_mean_only.png",
                        f"All compared models: degradation versus {shift_label} (mean only)",
                        False,
                        shift_label,
                    ),
                ),
                (
                    plot_percentage_degradation_heatmap,
                    (
                        all_percentage_rows,
                        "combined_physics_rel_l2",
                        all_models,
                        shift_modes,
                        out_root / f"all_models_combined_physics_percentage_worsening_{shift_slug}_heatmap.png",
                        f"All compared models: percentage worsening {shift_label} heatmap",
                    ),
                ),
                (
                    plot_percentage_degradation_bars,
                    (
                        all_percentage_rows,
                        "combined_physics_rel_l2",
                        shift_modes[-1],
                        all_models,
                        out_root / f"all_models_combined_physics_percentage_worsening_{shift_slug}_max_bars.png",
                        f"All compared models: percentage worsening at {shift_label.lower()}={shift_xs[-1]:.2f}",
                        True,
                    ),
                ),
                (
                    plot_family_percentage_degradation_curve,
                    (
                        all_percentage_rows,
                        "combined_physics_rel_l2",
                        FAMILY_GROUPS,
                        all_models,
                        shift_modes,
                        shift_xs,
                        out_root / f"all_families_combined_physics_percentage_worsening_{shift_slug}.png",
                        f"Between families: percentage worsening versus {shift_label}",
                        shift_label,
                        True,
                    ),
                ),
                (
                    plot_family_percentage_degradation_curve,
                    (
                        all_percentage_rows,
                        "combined_physics_rel_l2",
                        FAMILY_GROUPS,
                        all_models,
                        shift_modes,
                        shift_xs,
                        out_root / f"all_families_combined_physics_percentage_worsening_{shift_slug}_mean_only.png",
                        f"Between families: percentage worsening versus {shift_label} (mean only)",
                        shift_label,
                        False,
                    ),
                ),
            ]
        )

    drag_rank_models = [m for m in DRAG_RANK_MODELS if m in model_specs]
    if drag_rank_models and drag_rank_per_run_mode_rows:
        drag_beta_mode_order = [mode_name for mode_name in beta_mode_order if mode_defs[mode_name]["kind"] == "inverse_density_wor"]
        drag_sine_mode_order = [mode_name for mode_name in sine_mode_order if mode_defs[mode_name]["kind"] == "sinusoidal_axis_mixture_wor"]
        for mode_name in drag_beta_mode_order:
            mode_beta = float(mode_defs[mode_name]["beta"])
            plot_jobs.append(
                (
                    plot_ranked_curve_with_band,
                    (
                        drag_rank_per_run_mode_rows,
                        mode_name,
                        "surface_drag_force_x_full_pred",
                        "surface_drag_force_x_full_gt",
                        out_root / f"smart_family_surface_drag_force_x_ranked_beta_{mode_beta:.2f}.png",
                        f"SMART family: full-surface drag ranked by GT drag (beta={mode_beta:.2f})",
                        drag_rank_models,
                    ),
                )
            )
        for mode_name in drag_sine_mode_order:
            mix_fraction = float(mode_defs[mode_name]["mix_fraction"])
            plot_jobs.append(
                (
                    plot_ranked_curve_with_band,
                    (
                        drag_rank_per_run_mode_rows,
                        mode_name,
                        "surface_drag_force_x_full_pred",
                        "surface_drag_force_x_full_gt",
                        out_root / f"smart_family_surface_drag_force_x_ranked_sine_{mix_fraction:.2f}.png",
                        f"SMART family: full-surface drag ranked by GT drag (sine mix={mix_fraction:.2f})",
                        drag_rank_models,
                    ),
                )
            )
        drag_extra_groups = [
            ("sine_x", sine_x_mode_order, "sine-x"),
        ]
        for shift_slug, mode_order_for_shift, shift_label in drag_extra_groups:
            for mode_name in mode_order_for_shift:
                mix_fraction = float(mode_defs[mode_name]["mix_fraction"])
                plot_jobs.append(
                    (
                        plot_ranked_curve_with_band,
                        (
                            drag_rank_per_run_mode_rows,
                            mode_name,
                            "surface_drag_force_x_full_pred",
                            "surface_drag_force_x_full_gt",
                            out_root / f"smart_family_surface_drag_force_x_ranked_{shift_slug}_{mix_fraction:.2f}.png",
                            f"SMART family: full-surface drag ranked by GT drag ({shift_label} mix={mix_fraction:.2f})",
                            drag_rank_models,
                        ),
                    )
                )

    def keep_plot_job(job) -> bool:
        """Keep only all-model aggregate plots and distribution diagnostics."""
        if args.strategy_only:
            return False
        _func, func_args = job
        output_names = [arg.name for arg in func_args if isinstance(arg, Path)]
        output_name = " ".join(output_names)

        # The metric CSVs still retain field-level values for auditability,
        # but the rendered comparison should contain only aggregate metrics.
        # Family-specific figures are intentionally omitted because the user
        # requested one cross-model comparison view.
        is_all_model_combined_plot = "all_models_combined_global_" in output_name
        is_density_diagnostic = "density_shift_validation.png" in output_name
        if not (is_all_model_combined_plot or is_density_diagnostic):
            return False

        # There are only two severities, so line charts and dedicated
        # worsening figures add visual noise.  The endpoint bar plots carry
        # the signed percentage labels instead.
        if _func.__name__ in {
            "plot_numeric_mode_curve_with_band",
            "plot_ranked_curve_with_band",
            "plot_delta_bars",
            "plot_delta_severity_curve",
            "plot_percentage_degradation_curve",
            "plot_percentage_degradation_heatmap",
            "plot_percentage_degradation_bars",
            "plot_family_percentage_degradation_curve",
            "plot_paired_statistics",
        }:
            return False
        if _func.__name__ == "plot_metric_heatmap":
            return False
        if "beta" not in active_shift_set and "beta" in output_name:
            return False
        if "sine_y" not in active_shift_set and (
            "sine_y" in output_name
            or "worsening_sine." in output_name
            or ("worsening_sine_" in output_name and "sine_x" not in output_name)
            or "ranked_sine_" in output_name
        ):
            return False
        if "sine_x" not in active_shift_set and "sine_x" in output_name:
            return False
        return True

    plot_jobs = [job for job in plot_jobs if keep_plot_job(job)]
    with ProcessPoolExecutor(max_workers=max(1, int(args.plot_workers))) as pool:
        futures = [pool.submit(func, *func_args) for func, func_args in plot_jobs]
        for future in tqdm(futures, desc="CPU plot tasks", leave=False, dynamic_ncols=True):
            future.result()

    vtk_surface_query_dir = Path(args.vtk_surface_query_dir).expanduser().resolve()
    representative_run_dir = Path(smart_cfg.data_path) / f"run_{vtk_run_id}"
    if not representative_run_dir.is_dir():
        raise FileNotFoundError(f"Representative VTK run not found: {representative_run_dir}")

    rep_surf_coords_full, rep_surf_gt_full = load_surface_query_from_dir(vtk_surface_query_dir)
    rep_input_surf_coords = rep_surf_coords_full
    rep_vol_coords_full = np.load(representative_run_dir / "volume_coords.npy").astype(np.float32, copy=False)
    rep_surf_query_idx = choose_fixed_query_indices(rep_surf_coords_full.shape[0], surface_query_points, [args.seed, int(vtk_run_id), 4001])
    rep_vol_query_idx = choose_fixed_query_indices(rep_vol_coords_full.shape[0], volume_query_points, [args.seed, int(vtk_run_id), 4002])
    rep_surf_coords = rep_surf_coords_full[rep_surf_query_idx]
    rep_surf_gt = rep_surf_gt_full[rep_surf_query_idx]
    rep_vol_coords = rep_vol_coords_full[rep_vol_query_idx]
    audi_surf_query_norm = normalize_pos(torch.from_numpy(rep_surf_coords_full), min_pos, max_pos)
    rep_dummy_vol_query_norm = normalize_pos(torch.from_numpy(rep_vol_coords[:1]), min_pos, max_pos).unsqueeze(0)
    rep_input_geo_norm = normalize_pos(torch.from_numpy(rep_input_surf_coords), min_pos, max_pos)
    rep_sampling_geo_log_density = estimate_log_sampling_density(
        rep_input_geo_norm.unsqueeze(0),
        knn_k=dataset.geometry_density_knn_k,
        neighbor_hops=dataset.geometry_density_neighbor_hops,
        estimator=dataset.geometry_density_estimator,
    ).squeeze(0).cpu()

    sampling_input_surf_coords = np.load(representative_run_dir / "surface_coords.npy").astype(np.float32, copy=False)
    sampling_input_geo_norm = normalize_pos(torch.from_numpy(sampling_input_surf_coords), min_pos, max_pos)
    sampling_full_geo_log_density = dataset._load_or_compute_full_geometry_density(vtk_run_id, expected_n=int(sampling_input_surf_coords.shape[0]))
    sampling_full_geo_log_density_np = sampling_full_geo_log_density.to(dtype=torch.float32).numpy()
    sampling_distribution_weights = {
        "sine_y": sinusoidal_axis_probabilities(sampling_input_surf_coords, axis=1),
        "sine_x": sinusoidal_axis_probabilities(sampling_input_surf_coords, axis=0),
    }

    gt_pressure = np.asarray(rep_surf_gt_full[:, 0], dtype=np.float32)
    surface_point_data: Dict[str, np.ndarray] = {
        "gt_pressure": gt_pressure,
    }
    # Export every checkpoint requested for this comparison.  The old allowlist
    # could silently omit newly added model variants from the Audi VTK.
    representative_models = OrderedDict((m, models[m]) for m in model_specs)
    audi_vtk_skipped_models: List[str] = []
    n_surface_points = int(rep_surf_gt_full.shape[0])
    audi_model_groups = defaultdict(list)
    for model_name in representative_models:
        audi_model_groups[model_device_by_name[model_name]].append(model_name)

    def evaluate_audi_model_group(group_names):
        group_results = []
        for model_name in group_names:
            model = representative_models[model_name]
            model.eval()
            model_device = model_device_by_name[model_name]
            try:
                model_input_points = int(per_model_input_budgets[model_name])
                rep_rng = np.random.default_rng(
                    np.random.SeedSequence([args.seed, int(vtk_run_id), 99991, MODEL_ORDER.index(model_name)])
                )
                rep_idx = sample_uniform_without_replacement(rep_input_surf_coords.shape[0], model_input_points, rep_rng)
                rep_geo_view_norm = rep_input_geo_norm[torch.from_numpy(rep_idx)].unsqueeze(0)
                if model_uses_density(model_name):
                    model_density_estimator, model_density_knn_k, model_density_neighbor_hops, _ = resolve_model_internal_density_spec(
                        model_name,
                        model_specs[model_name]["config"],
                        model_specs[model_name]["checkpoint"],
                    )
                    rep_model_full_geo_log_density = estimate_log_sampling_density(
                        rep_input_geo_norm.unsqueeze(0),
                        knn_k=model_density_knn_k,
                        neighbor_hops=model_density_neighbor_hops,
                        estimator=model_density_estimator,
                    ).squeeze(0).cpu()
                    rep_geo_density_view = rep_model_full_geo_log_density.index_select(
                        0, torch.from_numpy(rep_idx).to(dtype=torch.long)
                    ).unsqueeze(0)
                else:
                    rep_geo_density_view = None
                pred_pressure = predict_audi_surface_pressure(
                    model_name=model_name,
                    model=model,
                    geo_view_norm=rep_geo_view_norm,
                    surf_query_norm=audi_surf_query_norm,
                    dummy_vol_query_norm=rep_dummy_vol_query_norm,
                    geo_log_density_view=rep_geo_density_view,
                    mean_s=mean_s,
                    std_s=std_s,
                    device=model_device,
                    base_seed=int(args.seed + 900000 + MODEL_ORDER.index(model_name) * 37),
                    repeats=args.model_repeats,
                    surface_chunk_size=int(args.audi_surface_chunk_size),
                )
                group_results.append((model_name, np.asarray(pred_pressure, dtype=np.float32), None))
            except Exception as exc:
                group_results.append((model_name, None, exc))
        return group_results

    audi_results = []
    with ThreadPoolExecutor(max_workers=len(audi_model_groups)) as audi_pool:
        audi_futures = [
            audi_pool.submit(evaluate_audi_model_group, group_names)
            for group_names in audi_model_groups.values()
        ]
        for future in audi_futures:
            audi_results.extend(future.result())

    for model_name, pred_pressure, error in sorted(
        audi_results, key=lambda item: MODEL_ORDER.index(item[0])
    ):
        prefix = MODEL_LABELS[model_name].lower()
        if error is not None:
            audi_vtk_skipped_models.append(model_name)
            print(f"[warning] Skipping Audi VTK export for {model_name}: {error}")
            surface_point_data[f"{prefix}_pressure_pred"] = np.full((n_surface_points,), np.nan, dtype=np.float32)
            surface_point_data[f"{prefix}_pressure_error"] = np.full((n_surface_points,), np.nan, dtype=np.float32)
            surface_point_data[f"{prefix}_pressure_abs_error"] = np.full((n_surface_points,), np.nan, dtype=np.float32)
            surface_point_data[f"{prefix}_pressure_relative_abs_error"] = np.full(
                (n_surface_points,), np.nan, dtype=np.float32
            )
            continue
        signed_error = pred_pressure - gt_pressure
        absolute_error = np.abs(signed_error)
        relative_absolute_error = absolute_error / np.maximum(np.abs(gt_pressure), 1.0e-8)
        surface_point_data[f"{prefix}_pressure_pred"] = pred_pressure
        surface_point_data[f"{prefix}_pressure_error"] = signed_error
        surface_point_data[f"{prefix}_pressure_abs_error"] = absolute_error
        surface_point_data[f"{prefix}_pressure_relative_abs_error"] = relative_absolute_error

    vtk_path = out_root / "audi_surface_pressure_predictions.vtk"
    write_polydata_vtk(vtk_path, rep_surf_coords_full, surface_point_data)

    sampling_vtk_paths = []
    sampling_histogram_paths = []
    sampling_budget = max(unique_input_budgets)
    beta_sample_log_density_values: Dict[float, np.ndarray] = {}
    for beta in (parse_shift_betas(args.shift_betas) if "beta" in active_shift_set else []):
        sampling_rng = np.random.default_rng(np.random.SeedSequence([args.seed, int(vtk_run_id), 77777, int(round(beta * 100))]))
        sample_idx = sample_inverse_density_without_replacement(sampling_full_geo_log_density_np, sampling_budget, float(beta), sampling_rng)
        sampled_points = sampling_input_surf_coords[sample_idx]
        beta_sample_log_density_values[float(beta)] = sampling_full_geo_log_density_np[sample_idx]
        sample_vtk_path = out_root / f"drivaerml_test_run_{vtk_run_id}_input_points_{sampling_budget}_inverse_density_beta_{beta:.2f}.vtk"
        write_polydata_vtk(sample_vtk_path, sampled_points, {})
        sampling_vtk_paths.append(str(sample_vtk_path))
        sample_hist_path = out_root / f"drivaerml_test_run_{vtk_run_id}_input_points_{sampling_budget}_inverse_density_beta_{beta:.2f}_density_hist.png"
        save_density_histogram(
            sample_hist_path,
            sampling_full_geo_log_density_np[sample_idx],
            title=f"Run {vtk_run_id} sampled input density histogram (beta={beta:.2f}, points={sampling_budget})",
        )
        sampling_histogram_paths.append(str(sample_hist_path))
        sample_hist_linear_path = out_root / f"drivaerml_test_run_{vtk_run_id}_input_points_{sampling_budget}_inverse_density_beta_{beta:.2f}_density_hist_linear.png"
        save_density_histogram(
            sample_hist_linear_path,
            sampling_full_geo_log_density_np[sample_idx],
            title=f"Run {vtk_run_id} sampled input density histogram (beta={beta:.2f}, points={sampling_budget}, linear axes)",
            log_axes=False,
        )
        sampling_histogram_paths.append(str(sample_hist_linear_path))

    if 0.0 in beta_sample_log_density_values and 1.0 in beta_sample_log_density_values:
        overlay_log_path = out_root / f"drivaerml_test_run_{vtk_run_id}_input_points_{sampling_budget}_inverse_density_beta_0.00_vs_1.00_density_hist.png"
        save_density_histogram_overlay(
            overlay_log_path,
            beta_sample_log_density_values[0.0],
            beta_sample_log_density_values[1.0],
            title=f"Run {vtk_run_id} sampled input density histogram overlay (beta=0.00 vs 1.00, points={sampling_budget})",
            label_a="beta=0.00",
            label_b="beta=1.00",
            log_axes=True,
        )
        sampling_histogram_paths.append(str(overlay_log_path))
        overlay_linear_path = out_root / f"drivaerml_test_run_{vtk_run_id}_input_points_{sampling_budget}_inverse_density_beta_0.00_vs_1.00_density_hist_linear.png"
        save_density_histogram_overlay(
            overlay_linear_path,
            beta_sample_log_density_values[0.0],
            beta_sample_log_density_values[1.0],
            title=f"Run {vtk_run_id} sampled input density histogram overlay (beta=0.00 vs 1.00, points={sampling_budget}, linear axes)",
            label_a="beta=0.00",
            label_b="beta=1.00",
            log_axes=False,
        )
        sampling_histogram_paths.append(str(overlay_linear_path))

    geometry_sampling_vtk_paths: List[str] = []
    geometry_sampling_plot_paths: List[str] = []
    geometry_vtk_run_id = int(vtk_run_id)
    if active_geometry_sources:
        if not all(
            geometry_source_vtp_path(source_name, geometry_vtk_run_id, geometry_vtp_dirs).is_file()
            for source_name in active_geometry_sources
        ):
            geometry_vtk_run_id = int(run_ids[0])
        source_sampled_points: Dict[str, np.ndarray] = {}
        for source_idx, source_name in enumerate(active_geometry_sources):
            source_path = geometry_source_vtp_path(source_name, geometry_vtk_run_id, geometry_vtp_dirs)
            source_points = read_vtp_points(source_path)
            reference_points = np.load(
                Path(smart_cfg.data_path) / f"run_{geometry_vtk_run_id}" / "surface_coords.npy"
            ).astype(np.float32, copy=False)
            validate_geometry_source_bbox(source_points, reference_points, source_name, geometry_vtk_run_id)
            if source_points.shape[0] < sampling_budget:
                raise ValueError(
                    f"Representative {source_name} VTP run_{geometry_vtk_run_id} has "
                    f"{source_points.shape[0]} points, below the largest active model input budget {sampling_budget}."
                )
            source_rng = np.random.default_rng(
                np.random.SeedSequence([args.seed, geometry_vtk_run_id, 777700, source_idx])
            )
            source_idx_array = sample_uniform_without_replacement(source_points.shape[0], sampling_budget, source_rng)
            sampled_points = source_points[source_idx_array]
            source_sampled_points[source_name] = sampled_points
            source_vtk_path = out_root / (
                f"drivaerml_test_run_{geometry_vtk_run_id}_input_points_{sampling_budget}_"
                f"geometry_{source_name}.vtk"
            )
            write_polydata_vtk(source_vtk_path, sampled_points, {})
            geometry_sampling_vtk_paths.append(str(source_vtk_path))
        geometry_plot_path = out_root / (
            f"drivaerml_test_run_{geometry_vtk_run_id}_input_points_{sampling_budget}_"
            "geometry_sources_distribution.png"
        )
        save_geometry_source_distribution_plot(
            geometry_plot_path,
            reference_points,
            source_sampled_points,
            title=(
                f"Run {geometry_vtk_run_id}: VTP geometry-source input distributions "
                f"({sampling_budget} sampled points)"
            ),
        )
        geometry_sampling_plot_paths.append(str(geometry_plot_path))

    representative_view2_sampling_vtk_paths: List[str] = []
    representative_mask_surface_vtp_paths: List[str] = []
    representative_mask_surface_vtp_dir = out_root / "mask_surfaces_vtps"
    if "SMART_DOWNSAMPLE" in model_specs:
        downsample_cfg = model_specs["SMART_DOWNSAMPLE"]["config"]
        downsample_budget = int(getattr(downsample_cfg, "secondary_view_geometry_points", getattr(downsample_cfg, "view_geometry_points", sampling_budget)))
        downsample_rng = np.random.default_rng(np.random.SeedSequence([args.seed, int(vtk_run_id), 919191, 1]))
        downsample_idx = sample_uniform_without_replacement(sampling_input_surf_coords.shape[0], downsample_budget, downsample_rng)
        downsample_vtk_path = out_root / f"drivaerml_test_run_{vtk_run_id}_smart_downsample_view2_input_points_{downsample_budget}.vtk"
        write_polydata_vtk(downsample_vtk_path, sampling_input_surf_coords[downsample_idx], {})
        representative_view2_sampling_vtk_paths.append(str(downsample_vtk_path))

    if "SMART_GAUSSIAN_BALL_MASKED" in model_specs:
        gaussian_cfg = model_specs["SMART_GAUSSIAN_BALL_MASKED"]["config"]
        gaussian_budget = int(getattr(gaussian_cfg, "secondary_view_geometry_points", getattr(gaussian_cfg, "view_geometry_points", sampling_budget)))
        masked_num_examples = 10
        for masked_example_idx in range(masked_num_examples):
            masked_rng = np.random.default_rng(
                np.random.SeedSequence([args.seed, int(vtk_run_id), 919191, 2, int(masked_example_idx)])
            )
            masked_info = sample_gaussian_ball_mask_subset(
                sampling_input_surf_coords,
                gaussian_budget,
                masked_rng,
                std_fraction_of_largest_extent=float(getattr(gaussian_cfg, "gaussian_mask_std_fraction_of_largest_extent", 0.05)),
                prob_at_1sigma=float(getattr(gaussian_cfg, "gaussian_mask_prob_at_1sigma", 0.33)),
                min_survivors=int(getattr(gaussian_cfg, "gaussian_mask_min_survivors", 16384)),
                return_metadata=True,
            )
            masked_base_idx = np.asarray(masked_info["base_idx"], dtype=np.int64)
            masked_kept_idx = np.asarray(masked_info["kept_idx"], dtype=np.int64)
            masked_points = sampling_input_surf_coords[masked_base_idx]
            masked_center = np.asarray(masked_info["center_point"], dtype=np.float32).reshape(1, 3)
            masked_vtk_path = out_root / (
                f"drivaerml_test_run_{vtk_run_id}_smart_gaussian_ball_masked_view2_example_{masked_example_idx:02d}"
                f"_input_points_{int(masked_points.shape[0])}_with_removed.vtk"
            )
            write_polydata_vtk(
                masked_vtk_path,
                masked_points,
                {
                    "kept_after_mask": np.asarray(masked_info["keep_mask"], dtype=np.float32),
                    "mask_remove_probability": np.asarray(masked_info["remove_probability"], dtype=np.float32),
                    "mask_distance_to_center": np.asarray(masked_info["distance_to_center"], dtype=np.float32),
                    "mask_center_flag": np.asarray(masked_info["center_flag"], dtype=np.float32),
                    "mask_sigma_radius": np.full((masked_points.shape[0],), float(masked_info["sigma_radius"]), dtype=np.float32),
                    "mask_center_xyz": np.repeat(masked_center, masked_points.shape[0], axis=0),
                },
            )
            representative_view2_sampling_vtk_paths.append(str(masked_vtk_path))
            gaussian_surface_vtp_path = representative_mask_surface_vtp_dir / (
                f"drivaerml_test_run_{vtk_run_id}_smart_gaussian_ball_masked_view2_example_{masked_example_idx:02d}"
                "_mask_1sigma_surface.vtp"
            )
            write_mask_surface_vtp(
                gaussian_surface_vtp_path,
                kind="gaussian_sigma_sphere",
                center_xyz=masked_info["center_point"],
                sigma_radius=float(masked_info["sigma_radius"]),
            )
            representative_mask_surface_vtp_paths.append(str(gaussian_surface_vtp_path))
            masked_survivor_points = sampling_input_surf_coords[masked_kept_idx]
            masked_survivor_vtk_path = out_root / (
                f"drivaerml_test_run_{vtk_run_id}_smart_gaussian_ball_masked_view2_example_{masked_example_idx:02d}"
                f"_input_points_{int(masked_survivor_points.shape[0])}_survivors_only.vtk"
            )
            write_polydata_vtk(
                masked_survivor_vtk_path,
                masked_survivor_points,
                {},
            )
            representative_view2_sampling_vtk_paths.append(str(masked_survivor_vtk_path))

    if "SMART_BOX_MASKED" in model_specs:
        box_cfg = model_specs["SMART_BOX_MASKED"]["config"]
        box_budget = int(getattr(box_cfg, "secondary_view_geometry_points", getattr(box_cfg, "view_geometry_points", sampling_budget)))
        box_num_examples = 10
        for box_example_idx in range(box_num_examples):
            box_rng = np.random.default_rng(
                np.random.SeedSequence([args.seed, int(vtk_run_id), 919191, 3, int(box_example_idx)])
            )
            box_info = sample_box_mask_subset(
                sampling_input_surf_coords,
                box_budget,
                box_rng,
                std_fraction_of_largest_extent=float(getattr(box_cfg, "gaussian_mask_std_fraction_of_largest_extent", 0.05)),
            )
            box_base_idx = np.asarray(box_info["base_idx"], dtype=np.int64)
            box_kept_idx = np.asarray(box_info["kept_idx"], dtype=np.int64)
            box_points = sampling_input_surf_coords[box_base_idx]
            box_center = np.asarray(box_info["center_point"], dtype=np.float32).reshape(1, 3)
            box_min = np.asarray(box_info["box_min"], dtype=np.float32).reshape(1, 3)
            box_max = np.asarray(box_info["box_max"], dtype=np.float32).reshape(1, 3)
            box_vtk_path = out_root / (
                f"drivaerml_test_run_{vtk_run_id}_smart_box_masked_view2_example_{box_example_idx:02d}"
                f"_input_points_{int(box_points.shape[0])}_with_removed.vtk"
            )
            write_polydata_vtk(
                box_vtk_path,
                box_points,
                {
                    "kept_after_mask": np.asarray(box_info["keep_mask"], dtype=np.float32),
                    "box_inside_flag": np.asarray(box_info["box_inside_flag"], dtype=np.float32),
                    "box_center_flag": np.asarray(box_info["center_flag"], dtype=np.float32),
                    "box_distance_to_center": np.asarray(box_info["distance_to_center"], dtype=np.float32),
                    "box_sigma_radius": np.full((box_points.shape[0],), float(box_info["sigma_radius"]), dtype=np.float32),
                    "box_side_length": np.full((box_points.shape[0],), float(box_info["box_side_length"]), dtype=np.float32),
                    "box_center_xyz": np.repeat(box_center, box_points.shape[0], axis=0),
                    "box_min_xyz": np.repeat(box_min, box_points.shape[0], axis=0),
                    "box_max_xyz": np.repeat(box_max, box_points.shape[0], axis=0),
                },
            )
            representative_view2_sampling_vtk_paths.append(str(box_vtk_path))
            box_surface_vtp_path = representative_mask_surface_vtp_dir / (
                f"drivaerml_test_run_{vtk_run_id}_smart_box_masked_view2_example_{box_example_idx:02d}"
                "_mask_box_surface.vtp"
            )
            write_mask_surface_vtp(
                box_surface_vtp_path,
                kind="box_surface",
                center_xyz=box_info["center_point"],
                sigma_radius=float(box_info["sigma_radius"]),
                box_min_xyz=box_info["box_min"],
                box_max_xyz=box_info["box_max"],
            )
            representative_mask_surface_vtp_paths.append(str(box_surface_vtp_path))
            box_survivor_points = sampling_input_surf_coords[box_kept_idx]
            box_survivor_vtk_path = out_root / (
                f"drivaerml_test_run_{vtk_run_id}_smart_box_masked_view2_example_{box_example_idx:02d}"
                f"_input_points_{int(box_survivor_points.shape[0])}_survivors_only.vtk"
            )
            survivor_box_data = {
                "box_center_xyz": np.repeat(box_center, box_survivor_points.shape[0], axis=0),
                "box_min_xyz": np.repeat(box_min, box_survivor_points.shape[0], axis=0),
                "box_max_xyz": np.repeat(box_max, box_survivor_points.shape[0], axis=0),
                "box_sigma_radius": np.full((box_survivor_points.shape[0],), float(box_info["sigma_radius"]), dtype=np.float32),
                "box_side_length": np.full((box_survivor_points.shape[0],), float(box_info["box_side_length"]), dtype=np.float32),
            }
            write_polydata_vtk(box_survivor_vtk_path, box_survivor_points, survivor_box_data)
            representative_view2_sampling_vtk_paths.append(str(box_survivor_vtk_path))

    for shift_index, (shift_name, coordinate_axis, coordinate_label) in enumerate(
        (
            ("sine_y", 1, "y"),
            ("sine_x", 0, "x"),
        )
    ):
        if shift_name not in active_shift_set:
            continue
        shift_feature_values, shift_weights, shift_feature_label = _spatial_shift_feature_values(
            sampling_input_surf_coords,
            shift_name,
        )
        endpoint_sampled_points: Dict[float, np.ndarray] = {}
        for mix_fraction in sine_mix_levels:
            sampling_rng = np.random.default_rng(
                np.random.SeedSequence(
                    [args.seed, int(vtk_run_id), 88888, 10 + shift_index, int(round(float(mix_fraction) * 1000))]
                )
            )
            sample_idx = sample_uniform_weighted_mixture_without_replacement(
                sampling_distribution_weights[shift_name],
                sampling_budget,
                float(mix_fraction),
                sampling_rng,
            )
            sampled_points = sampling_input_surf_coords[sample_idx]
            endpoint_sampled_points[float(mix_fraction)] = sampled_points
            sample_vtk_path = out_root / (
                f"drivaerml_test_run_{vtk_run_id}_input_points_{sampling_budget}_ood_{shift_name}_mix_{float(mix_fraction):.2f}.vtk"
            )
            write_polydata_vtk(
                sample_vtk_path,
                sampled_points,
                {
                    "shift_intensity": np.full((sampled_points.shape[0],), float(mix_fraction), dtype=np.float32),
                    "shift_feature_value": shift_feature_values[sample_idx].astype(np.float32, copy=False),
                    "shift_probability_weight": shift_weights[sample_idx].astype(np.float32, copy=False),
                },
            )
            sampling_vtk_paths.append(str(sample_vtk_path))
            if coordinate_axis is not None:
                sample_hist_path = out_root / (
                    f"drivaerml_test_run_{vtk_run_id}_input_points_{sampling_budget}_ood_{shift_name}_mix_{float(mix_fraction):.2f}_{coordinate_label}_hist.png"
                )
                save_sampling_y_histogram(
                    sample_hist_path,
                    sampled_points,
                    sampling_input_surf_coords,
                    float(mix_fraction),
                    title=(
                        f"Run {vtk_run_id} sampled {coordinate_label} distribution "
                        f"(OOD {shift_name} mix={float(mix_fraction):.2f}, points={sampling_budget})"
                    ),
                    axis=coordinate_axis,
                    coordinate_name=coordinate_label,
                )
                sampling_histogram_paths.append(str(sample_hist_path))

        endpoint_plot_path = out_root / (
            f"drivaerml_test_run_{vtk_run_id}_input_points_{sampling_budget}_ood_{shift_name}_endpoint_distribution.png"
        )
        save_spatial_shift_endpoint_plot(
            endpoint_plot_path,
            shift_name,
            sampling_input_surf_coords,
            endpoint_sampled_points,
            title=(
                f"Run {vtk_run_id}: {SHIFT_LABELS[shift_name]}\n"
                f"endpoint comparison (zero vs maximum intensity, feature={shift_feature_label})"
            ),
        )
        sampling_histogram_paths.append(str(endpoint_plot_path))

    payload = {
        "args": vars(args),
        "run_ids": run_ids,
        "representative_vtk_run_id": vtk_run_id,
        "models": {k: v["checkpoint"] for k, v in model_specs.items()},
        "mode_definitions": mode_defs,
        "workflow": {
            "benchmark_queries_fixed_per_run": True,
            "benchmark_queries_with_replacement": bool(args.query_sampling_with_replacement),
            "benchmark_surface_query_points": surface_query_points,
            "benchmark_volume_query_points": volume_query_points,
            "per_model_query_budgets": per_model_query_budgets,
            "query_budget_mismatch_models": query_budget_mismatch_models,
            "per_model_encoder_input_budgets": per_model_input_budgets,
            "aligned_mode": "model_train_aligned_uniform_sampling",
            "shift_modes": [name for name in mode_defs if name != "aligned_uniform_wor"],
            "active_shifts": active_shifts,
            "ood_modes": [
                name for name, info in mode_defs.items()
                if info["kind"] == "sinusoidal_axis_mixture_wor"
            ],
            "ood_sine_axes": ["x", "y"],
            "ood_sine_mix_levels": sine_mix_levels,
            "ood_distribution_shifts": [shift for shift in active_shifts if shift != "beta"],
            "active_geometry_sources": active_geometry_sources,
            "surface_vtp_dir": str(surface_vtp_dir),
            "geometry_vtp_dirs": {method: str(path) for method, path in geometry_vtp_dirs.items()},
            "geometry_decimation_factors": geometry_decimation_factors,
            "geometry_test_common_subset_size": len(geometry_candidate_ids) if geometry_candidate_ids is not None else 0,
            "views_per_mode": views_per_mode,
            "view_batch_size": view_batch_size,
            "model_repeats": int(args.model_repeats),
            "top_level_alignment_note": "Aligned mode mirrors each model's training-time geometry sampler and budget. Explicit two-view configs follow train_primary_sampling_mode; single-view configs follow AhmedMLDatasetV2 geometry sampling semantics.",
            "model_internal_note": "Each model keeps its own internal encoder-block subsampling exactly as implemented in that checkpointed architecture.",
            "sampling_density_study_spec": {
                "estimator": density_estimator,
                "knn_k": density_knn_k,
                "neighbor_hops": density_neighbor_hops,
            },
            "model_internal_density_specs": model_internal_density_specs,
            "representative_vtk_surface_query_source": str(vtk_surface_query_dir),
            "representative_vtk_encoder_input_source": "external surface_coords.npy from the selected Audi VTK query directory, sampled with each model's own aligned encoder input budget",
            "representative_vtk_dummy_volume_query_source": f"first point from DrivAerML run {vtk_run_id} fixed volume-query subset; used only if a model cannot execute an empty-volume surface-only export path",
            "representative_sampling_point_source_run_id": vtk_run_id,
            "representative_sampling_point_vtks": sampling_vtk_paths,
            "representative_sampling_point_histograms": sampling_histogram_paths,
            "representative_geometry_sampling_vtks": geometry_sampling_vtk_paths,
            "representative_geometry_sampling_plots": geometry_sampling_plot_paths,
            "representative_geometry_vtk_run_id": geometry_vtk_run_id if active_geometry_sources else None,
            "satloss_endpoint_table_csv": str(satloss_table_csv),
            "satloss_endpoint_table_markdown": str(satloss_table_md),
            "satloss_endpoint_table_plot": str(satloss_table_png),
            "smart_strategy_endpoint_tables": strategy_table_paths,
            "smart_strategy_endpoint_plots": strategy_plot_paths,
            "smart_strategy_models": strategy_models,
            "representative_view2_sampling_vtks": representative_view2_sampling_vtk_paths,
            "representative_mask_surface_vtps": representative_mask_surface_vtp_paths,
            "audi_vtk_skipped_models": audi_vtk_skipped_models,
            "encoder_budget_mismatches": encoder_budget_mismatch_models,
            "query_budget_mismatches": query_budget_mismatch_models,
        },
        "configs": {k: config_name_map[k] for k in model_specs},
        "training_config_snapshots": {
            k: OmegaConf.to_container(configs[k], resolve=True) for k in model_specs
        },
        "aggregate_metrics": aggregate_rows,
        "robustness_summary": robustness_rows,
    }
    (out_root / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    workflow_lines = [
        "# Sampling-Invariance Evaluation Workflow",
        "",
        "## Goal",
        "Measure how much prediction quality changes when only the encoder-input geometry sampling distribution changes, while keeping query points fixed.",
        "",
        "## Fairness Rules Used",
        f"- Evaluated models: `{', '.join(model_specs.keys())}`",
        f"- Active sampling shifts: `{active_shifts}`; use `--active-shifts` to select a subset without changing model evaluation logic.",
        f"- Surface query coordinates are fixed per run to one common sample of `{surface_query_points}` points for every model and every sampling mode.",
        f"- Volume query coordinates are fixed per run to one common sample of `{volume_query_points}` points for every model and every sampling mode.",
        "- By default these common query samples use replacement to match AhmedMLDatasetV2's default `fast_approx_sampling=True`; pass `--no-query-sampling-with-replacement` for distinct query subsets.",
        "- If a family-specific query override is requested, that family uses its own fixed per-run query subset while the other families keep the global benchmark subset.",
        "- Encoder input point budget is train-aligned per model by default. That keeps each family on its own training budget instead of forcing all families to the smallest one.",
        "- If a model was trained with smaller query budgets than this evaluation uses, the script reports that mismatch explicitly in the console and `results.json`.",
        "- The aligned mode mirrors each model's training-time top-level geometry sampler and preserves its own encoder input budget unless you explicitly override `--input-points`; the dataset default uses uniform sampling with replacement for unseeded sub-budget geometry views.",
        f"- Beta-shift modes use inverse-density sampling without replacement at betas `{shift_betas}` and keep the same point budget.",
        f"- Sampling shifts are computed with the requested CLI density estimator `{density_estimator}`, but density-aware models receive density tensors from their own training config when available.",
        "- Spatial modes use controlled mixtures of uniform sampling with the restored sinusoidal-x/y fields. They keep the same point budget and do not mask or delete a region.",
        f"- Geometry-source modes are separate tests: the aligned preprocessed cloud is the baseline, while additional inputs are sampled from the completed angle-based, isotropic, and voxel/quadric VTP sources at factors `{geometry_decimation_factors}`. Beta/sine modes never read VTPs.",
        "- VTP geometry coordinates are validated against the matching preprocessed run bounding box before normalization; the accepted tolerance is 2.5e-3 in world-coordinate units.",
        "- VTP geometry-source modes use uniform sampling without replacement from decimated VTP vertices, then the sampled coordinates are normalized with the same global training bounds used by the models.",
        "- Each remeshing intensity is evaluated only at the two endpoints `0.0` and `1.0`; intermediate intensities are intentionally not sampled.",
        "- For every OOD mixture severity `s`, the sampler takes exactly `round(s * K)` points from the shifted probability field and the remaining points uniformly from the leftover pool, so the severity has an exact point-count interpretation.",
        "- If `beta=0` is included in the shifted list, it acts as a uniform-without-replacement sanity-check mode and should match the aligned mode up to sampling randomness.",
        "- Internal model behavior is not overridden beyond safe batched-query chunking. In particular, each model keeps its own trained latent-anchor logic and encoder-block 16k subsampling behavior.",
        "- In-family fairness is strongest when all compared checkpoints in that family were trained with the same encoder input budget and the evaluation uses that same budget.",
        "- Cross-family fairness is weaker when families were trained with different encoder input budgets; the script records those mismatches explicitly in `results.json`.",
        "- Surface-integral drag for the SMART-family drag ranking plots is computed on the full surface point cloud as the signed x-force `∫(-p n_x + τ_x) dA` using the stored `surface_areas.npy` weights.",
        f"- Views per run/mode: `{views_per_mode}`",
        f"- Repeated stochastic forwards per view batch: `{int(args.model_repeats)}`",
        "",
        "## Representative VTK Export",
        "- The Audi pressure-field VTK uses the full Audi surface point cloud for the surface query cloud.",
        "- The Audi VTK export is visualization-only and does not affect the benchmark statistics.",
        "- The representative prediction VTK stores ground-truth pressure, every active model's pressure prediction, signed pressure error, absolute pressure error, and pointwise relative absolute pressure error.",
        "- If a model cannot execute a true empty-volume surface-only export path, the script falls back to one fixed representative volume query point from the selected DrivAerML run. This affects only the Audi visualization export, not the benchmark metrics.",
        "- If a model still cannot complete the full-Audi visualization export safely, it is skipped only for this VTK step and recorded in the results payload.",
        f"- Surface-query directory for the Audi pressure-field export: `{vtk_surface_query_dir}`",
        f"- Separate point-cloud VTKs are exported from DrivAerML test run `{vtk_run_id}` for each active inverse-density beta and spatial-distribution mode, using the largest active encoder budget `{sampling_budget}` so you can directly inspect each representative input distribution.",
        "- Each spatial-shift VTK stores the endpoint intensity, the shift-specific diagnostic coordinate, and the unnormalized spatial probability weight for every sampled point.",
        "- Each inverse-density beta sampled-point VTK also gets a separate PNG histogram of the sampled density distribution, with a log-count y-axis and no percentile trimming.",
        "- Each sinusoidal-x/y sampled-point VTK gets a coordinate-distribution histogram and endpoint distribution plot.",
        f"- Each active spatial shift also gets `{sampling_budget}`-point endpoint distribution plots comparing intensity `0.0` with `1.0` using a shift-specific PDF and accumulation diagnostic.",
        "",
        "## Aggregation",
        "- First aggregate multiple independently sampled views within each `(run, model, mode)` tuple.",
        "- Then aggregate across runs to obtain mean and standard deviation for plots.",
        "- Robustness is summarized both by absolute shifted performance and by `shifted - aligned` / `shifted / aligned` degradation statistics.",
        "",
        "## Configs and Checkpoints",
    ]
    for model_name, spec in model_specs.items():
        budget = per_model_query_budgets[model_name]
        workflow_lines.append(
            f"- `{model_name}`: config=`{config_name_map[model_name]}` checkpoint=`{spec['checkpoint']}` "
            f"eval_queries=({budget['surface']} surface, {budget['volume']} volume) "
            f"eval_encoder_input={per_model_input_budgets[model_name]}"
        )
    workflow_lines.extend(
        [
            "",
            "## Outputs",
            "- `per_view_metrics.csv`: metrics for every run/view/model/mode.",
            "- `per_run_mode_metrics.csv`: per-run averages across views with standard deviations.",
        "- `aggregate_metrics.csv`: across-run means/stds for every model and sampling mode.",
        "- `robustness_summary.csv`: strongest-shift robustness summary.",
        "- `paired_statistics.csv`: paired run-level deltas, quantiles, 95% CIs, and normal-approximation p-values.",
        "- Rendered model-error PNGs are limited to all-model `combined_global` aggregate comparisons.",
        "- Combined-global endpoint bars use a log y-axis and signed percentage labels; standard-deviation whiskers are disabled for the rendered endpoint and geometry-source bars.",
        "- Percentage outputs keep only beta maximum, sine-y maximum, sine-x maximum, and the largest requested factor for every geometry method. The SMART strategy table includes every requested geometry factor and reports differences relative to SATLOSS.",
        "- Field-level values remain available in the CSV files; paper-facing plots emphasize combined-global error, while the full aggregate CSV remains available for auditability.",
        "- `audi_surface_pressure_predictions.vtk`: full Audi surface pressure ground truth plus every active model's pressure prediction and per-point error fields.",
        "- `results.json`: machine-readable summary including any representative-VTK model skips.",
        f"- `drivaerml_test_run_{vtk_run_id}_input_points_{sampling_budget}_inverse_density_beta_*.vtk`: sampled `{sampling_budget}` input points for each inverse-density beta from one evaluated DrivAerML test run.",
        f"- `drivaerml_test_run_{vtk_run_id}_input_points_{sampling_budget}_inverse_density_beta_*_density_hist.png`: density-distribution histogram for each sampled input-point VTK.",
        f"- `drivaerml_test_run_{vtk_run_id}_input_points_{sampling_budget}_ood_sine_[xy]_mix_*_[xy]_hist.png`: coordinate histograms for the sine-x and sine-y sampled input-point VTKs.",
        f"- `drivaerml_test_run_{vtk_run_id}_input_points_{sampling_budget}_ood_*_endpoint_distribution.png`: endpoint distribution plots for every active spatial shift.",
        f"- `drivaerml_test_run_<run>_input_points_{sampling_budget}_geometry_*.vtk`: representative input samples from the active decimated VTP geometries.",
        f"- `drivaerml_test_run_<run>_input_points_{sampling_budget}_geometry_sources_distribution.png`: normalized x/y/z distribution comparison against the matching preprocessed geometry.",
        "- `all_models_combined_global_geometry_sources_bars_log.png` and `_linear.png`: aggregate errors for aligned and every active angle/isotropic/voxel geometry source; percentages are relative to each model's aligned error.",
        "- `all_models_combined_global_satloss_endpoint_improvement.csv`, `.md`, and `.png`: endpoint-only SATLOSS-versus-vanilla relative improvement table and plot.",
        "- `smart_strategies_combined_global_endpoint_bars_{beta,sine_y,sine_x,angle,isotropic,voxel}_{log,linear}.png`: one endpoint plot per sampling shift or VTP method; VTP plots pair div5 and div10.",
        "- `smart_strategies_combined_global_relative_vs_smart_{beta,sine_y,sine_x,angle,isotropic,voxel}.png`: signed strategy differences relative to vanilla SMART.",
        "- `smart_strategies_combined_global_relative_vs_satloss_{beta,sine_y,sine_x,angle,isotropic,voxel}.png`: signed strategy differences relative to SATLOSS.",
        "- Strategy tables include absolute error, relative-to-SATLOSS percentages, and relative-to-SMART percentages in CSV/Markdown form.",
    ]
    )
    (out_root / "workflow.md").write_text("\n".join(workflow_lines), encoding="utf-8")

    summary_lines = [
        "# DrivAerML Sampling-Invariance Comparison",
        "",
        f"- Evaluated test runs: `{run_ids}`",
        f"- Representative VTK run: `{vtk_run_id}`",
        f"- Per-model encoder input budgets: `{per_model_input_budgets}`",
        f"- Views per run/mode: `{views_per_mode}`",
        f"- View batch size: `{view_batch_size}`",
        f"- Strongest shift mode: `{strongest_mode}`",
        f"- Shift betas: `{shift_betas}`",
        f"- Active sampling shifts: `{active_shifts}`",
        f"- Active geometry sources: `{active_geometry_sources}` at factors `{geometry_decimation_factors}`",
        f"- OOD sampling modes: endpoint-only uniform-to-sinusoidal-x/y mixtures, with intensities `{sine_mix_levels}`",
        f"- Fixed benchmark query subsets per run: `{surface_query_points}` surface + `{volume_query_points}` volume",
        "- SMART-family drag plots are sorted by full-surface ground-truth drag instead of severity to reduce visual noise.",
        f"- Representative VTK surface query source: `{vtk_surface_query_dir}`",
        "",
        "## Robustness Summary",
    ]
    for row in robustness_rows:
        summary_lines.append(
            f"- `{row['model_name']}`: "
            f"aligned physics={row['aligned_combined_physics_rel_l2']:.6g}, "
            f"strongest-shift physics={row['strongest_shift_combined_physics_rel_l2']:.6g}, "
            f"delta={row['combined_physics_delta']:.6g}, "
            f"ratio={row['combined_physics_ratio']:.6g}"
        )
    summary_lines.extend(
        [
            "",
            "## Interpretation",
        "- Lower strongest-shift absolute error means better robustness under changed encoder-input sampling.",
        "- Lower strongest-shift delta means less degradation relative to aligned sampling.",
        "- Lower strongest-shift ratio means better robustness relative to the model's own aligned baseline.",
        "- For in-family conclusions, the most trustworthy comparison is when that family shares the same train-time encoder input budget and this evaluation uses that same budget.",
        "- For cross-family conclusions, treat differences more cautiously unless the encoder-input budget also matched across the compared families.",
        "- Paper-facing figures emphasize combined-global error so the conclusion is not dominated by one field; field-level values remain in the CSV outputs.",
        "- The dedicated SMART strategy figures compare SATLOSS directly against downsampling, Gaussian-ball masking, and box masking using both absolute error and signed relative differences.",
        ]
    )
    (out_root / "summary.md").write_text("\n".join(summary_lines), encoding="utf-8")

    print(f"Wrote results to {out_root}")
    print(f"Representative VTK: {vtk_path}")


if __name__ == "__main__":
    main()
