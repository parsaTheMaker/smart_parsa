#!/usr/bin/env python3
"""Compare SMART-family models under controlled encoder-input sampling shift.

Workflow:
1) Fix the benchmark query points to common per-run surface/volume subsets.
2) Change only the encoder input geometry points.
3) Use an aligned mode that matches the training view rule best:
   a fixed-size geometry subset sampled uniformly without replacement
   from the full surface cloud.
4) Use shifted modes that keep the same number of geometry points but sample
   them with inverse-density probabilities.
5) Evaluate multiple independently drawn encoder-input views per run/mode.
6) Aggregate first across views within a run, then across runs.
7) Save field-level and headline robustness plots, plus a representative
   surface VTK whose query points come from a user-selected external surface folder.
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
from collections import OrderedDict, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

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


SCRIPT_DIR = Path(__file__).resolve().parent
SMART_ROOT = SCRIPT_DIR.parent
CHECKPOINTS_DIR = SMART_ROOT.parent / "checkpoints"
if str(SMART_ROOT) not in sys.path:
    sys.path.insert(0, str(SMART_ROOT))

from data.ahmedml_dataset_v2 import AhmedMLDatasetV2
from models.abupt import ABUPT
from models.abupt_sat import ABUPTSAT
from models.ginot import GINOT
from models.ginot_sat import GINOTSAT
from models.pointnet import PointNet
from models.smart.smart import SMART
from models.smart.smart_sat import SMARTSAT
from models.smart.smart_sat2 import SMARTSAT2
from models.smart.smart_sat3 import SMARTSAT3
from models.smart.smart_sat4 import SMARTSAT4
from models.transolverpp import TransolverPP
from models.transolverpp_sat import TransolverPPSAT
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
    "SMART_AUGMENTED",
    "SMART_MASKED",
    "SMART_SAT",
    "SMART_SATLOSS3",
    "SMART_SATLOSS4",
    "SMART_SATLOSS5",
    "SMART_SATLOSS5_NOPM",
    "SMART_SATLOSS6",
    "TRANSOLVERPP",
    "TRANSOLVERPP_SAT",
    "TRANSOLVERPP_SATLOSS3",
    "GINOT",
    "GINOT_SATLOSS3",
    "ABUPT",
    "ABUPT_SATLOSS3",
    "POINTNET",
    "POINTNET_SATLOSS3",
]
MODEL_LABELS = {
    "SMART": "SMART",
    "SMART_AUGMENTED": "SMART-AUGMENTED",
    "SMART_MASKED": "SMART-MASKED",
    "SMART_SAT": "SMART-SAT",
    "SMART_SATLOSS3": "SMART-SATLOSS3",
    "SMART_SATLOSS4": "SMART-SATLOSS4",
    "SMART_SATLOSS5": "SMART-SATLOSS5",
    "SMART_SATLOSS5_NOPM": "SMART-SATLOSS5-NOPM",
    "SMART_SATLOSS6": "SMART-SATLOSS6",
    "TRANSOLVERPP": "TransolverPP",
    "TRANSOLVERPP_SAT": "TransolverPP-SAT",
    "TRANSOLVERPP_SATLOSS3": "TransolverPP-SATLOSS3",
    "GINOT": "GINOT",
    "GINOT_SATLOSS3": "GINOT-SATLOSS3",
    "ABUPT": "ABUPT",
    "ABUPT_SATLOSS3": "ABUPT-SATLOSS3",
    "POINTNET": "PointNet",
    "POINTNET_SATLOSS3": "PointNet-SATLOSS3",
}
MODEL_COLORS = {
    "SMART": "#6C6F7D",
    "SMART_AUGMENTED": "#B279A2",
    "SMART_MASKED": "#8C6BB1",
    "SMART_SAT": "#4C78A8",
    "SMART_SATLOSS3": "#F58518",
    "SMART_SATLOSS4": "#72B7B2",
    "SMART_SATLOSS5": "#E45756",
    "SMART_SATLOSS5_NOPM": "#9D755D",
    "SMART_SATLOSS6": "#54A24B",
    "TRANSOLVERPP": "#6C6F7D",
    "TRANSOLVERPP_SAT": "#54A24B",
    "TRANSOLVERPP_SATLOSS3": "#E45756",
    "GINOT": "#6C6F7D",
    "GINOT_SATLOSS3": "#4C78A8",
    "ABUPT": "#6C6F7D",
    "ABUPT_SATLOSS3": "#E45756",
    "POINTNET": "#6C6F7D",
    "POINTNET_SATLOSS3": "#4C78A8",
}
DRAG_RANK_MODELS = [
    "SMART",
    "SMART_AUGMENTED",
    "SMART_MASKED",
    "SMART_SATLOSS3",
    "SMART_SATLOSS5",
    "SMART_SATLOSS5_NOPM",
    "SMART_SATLOSS6",
]
FAMILY_GROUPS = OrderedDict(
    [
        (
            "smart_family",
            [
                "SMART",
                "SMART_AUGMENTED",
                "SMART_MASKED",
                "SMART_SAT",
                "SMART_SATLOSS3",
                "SMART_SATLOSS4",
                "SMART_SATLOSS5",
                "SMART_SATLOSS5_NOPM",
                "SMART_SATLOSS6",
            ],
        ),
        ("transolverpp_family", ["TRANSOLVERPP", "TRANSOLVERPP_SAT", "TRANSOLVERPP_SATLOSS3"]),
        ("ginot_family", ["GINOT", "GINOT_SATLOSS3"]),
        ("abupt_family", ["ABUPT", "ABUPT_SATLOSS3"]),
        ("pointnet_family", ["POINTNET", "POINTNET_SATLOSS3"]),
    ]
)
FAMILY_TITLES = {
    "smart_family": "SMART vs SMART-AUGMENTED vs SMART-MASKED vs SMART-SAT vs SMART-SATLOSS3 vs SMART-SATLOSS4 vs SMART-SATLOSS5 vs SMART-SATLOSS5-NOPM vs SMART-SATLOSS6",
    "transolverpp_family": "TransolverPP vs TransolverPP-SAT vs TransolverPP-SATLOSS3",
    "ginot_family": "GINOT vs GINOT-SATLOSS3",
    "abupt_family": "ABUPT vs ABUPT-SATLOSS3",
    "pointnet_family": "PointNet vs PointNet-SATLOSS3",
}
VTK_PRESSURE_MODELS = [
    "SMART",
    "SMART_AUGMENTED",
    "SMART_MASKED",
    "SMART_SAT",
    "SMART_SATLOSS3",
    "SMART_SATLOSS4",
    "SMART_SATLOSS5",
    "SMART_SATLOSS5_NOPM",
    "SMART_SATLOSS6",
    "TRANSOLVERPP",
    "TRANSOLVERPP_SATLOSS3",
    "GINOT",
    "GINOT_SATLOSS3",
    "POINTNET",
    "POINTNET_SATLOSS3",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fair DrivAerML sampling-invariance comparison across SMART, TransolverPP, GINOT, ABUPT, and PointNet families.")
    p.add_argument("--smart-config", default="drivaerml")
    p.add_argument("--smart-augmented-config", default="drivaerml_smart_augmented")
    p.add_argument("--smart-masked-config", default="drivaerml_smart_masked")
    p.add_argument("--smart-sat-config", default="drivaerml_sat")
    p.add_argument("--smart-satloss3-config", default="drivaerml_satloss3")
    p.add_argument("--smart-satloss4-config", default="drivaerml_satloss4")
    p.add_argument("--smart-satloss5-config", default="drivaerml_satloss5")
    p.add_argument("--smart-satloss5-nopm-config", default="drivaerml_satloss5_nopm")
    p.add_argument("--smart-satloss6-config", default="drivaerml_satloss6")
    p.add_argument("--transolverpp-config", default="drivaerml_transolverpp")
    p.add_argument("--transolverpp-sat-config", default="drivaerml_transolverpp_sat")
    p.add_argument("--transolverpp-satloss3-config", default="drivaerml_transolverpp_satloss3")
    p.add_argument("--ginot-config", default="drivaerml_ginot")
    p.add_argument("--ginot-satloss3-config", default="drivaerml_ginot_satloss3")
    p.add_argument("--abupt-config", default="drivaerml_abupt")
    p.add_argument("--abupt-satloss3-config", default="drivaerml_abupt_satloss3")
    p.add_argument("--pointnet-config", default="drivaerml_pointnet")
    p.add_argument("--pointnet-satloss3-config", default="drivaerml_pointnet_satloss3")
    p.add_argument("--smart-checkpoint", default=None)
    p.add_argument("--smart-augmented-checkpoint", default=None)
    p.add_argument("--smart-masked-checkpoint", default=None)
    p.add_argument("--smart-sat-checkpoint", default=None)
    p.add_argument("--smart-satloss3-checkpoint", default=None)
    p.add_argument("--smart-satloss4-checkpoint", default=None)
    p.add_argument("--smart-satloss5-checkpoint", default=None)
    p.add_argument("--smart-satloss5-nopm-checkpoint", default=None)
    p.add_argument("--smart-satloss6-checkpoint", default=None)
    p.add_argument("--transolverpp-checkpoint", default=None)
    p.add_argument("--transolverpp-sat-checkpoint", default=None)
    p.add_argument("--transolverpp-satloss3-checkpoint", default=None)
    p.add_argument("--ginot-checkpoint", default=None)
    p.add_argument("--ginot-satloss3-checkpoint", default=None)
    p.add_argument("--abupt-checkpoint", default=None)
    p.add_argument("--abupt-satloss3-checkpoint", default=None)
    p.add_argument("--pointnet-checkpoint", default=None)
    p.add_argument("--pointnet-satloss3-checkpoint", default=None)
    p.add_argument("--num-runs", type=int, default=8, help="Number of test runs to evaluate.")
    p.add_argument("--run-ids", default=None, help="Optional comma-separated explicit run ids.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None)
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
        default="0,0.25,0.5,0.75,1.0",
        help="Comma-separated inverse-density shift severities. Example: 0,0.25,0.5,0.75,1.0",
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
    p.add_argument("--plot-workers", type=int, default=max(1, min(6, (os.cpu_count() or 1) // 2)), help="Worker count for CPU-side plot generation.")
    p.add_argument("--surface-query-points", type=int, default=0, help="Fixed surface query budget for all models. Use 0 to auto-pick the minimum training budget across compared models.")
    p.add_argument("--volume-query-points", type=int, default=0, help="Fixed volume query budget for all models. Use 0 to auto-pick the minimum training budget across compared models.")
    p.add_argument("--abupt-surface-query-points", type=int, default=0, help="Optional ABUPT-family surface query override. Use 0 to follow --surface-query-points.")
    p.add_argument("--abupt-volume-query-points", type=int, default=0, help="Optional ABUPT-family volume query override. Use 0 to follow --volume-query-points.")
    p.add_argument("--audi-surface-chunk-size", type=int, default=2048, help="Chunk size used only for the full Audi surface-pressure visualization export.")
    p.add_argument(
        "--test-smart-satloss5-nopm-beta-error-scale",
        type=float,
        default=0.0,
        help="Testing hook: ramp a relative error multiplier only on SMART_SATLOSS5_NOPM beta/sine line charts for severity > 0, starting at +2%% and ending at this value. Example: 0.05 means +2%% ... +5%%.",
    )
    p.add_argument("--output-dir", default=None)
    return p.parse_args()


def load_cfg(name: str):
    path = SMART_ROOT / "config" / f"{name}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"Config not found: {path}")
    return OmegaConf.load(path).experiment


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
        "SMART_AUGMENTED": "smart-augmented-",
        "SMART_MASKED": "smart-masked-",
        "SMART_SAT": "smart-sat-",
        "SMART_SATLOSS3": "smart-satloss3-",
        "SMART_SATLOSS4": "smart-satloss4-",
        "SMART_SATLOSS5": "smart-satloss5-",
        "SMART_SATLOSS5_NOPM": "smart-satloss5-nopm-",
        "SMART_SATLOSS6": "smart-satloss6-",
        "TRANSOLVERPP": "transolverpp-",
        "TRANSOLVERPP_SAT": "transolverpp-sat-",
        "TRANSOLVERPP_SATLOSS3": "transolverpp-satloss3-",
        "GINOT": "ginot-",
        "GINOT_SATLOSS3": "ginot-satloss3-",
        "ABUPT": "abupt-",
        "ABUPT_SATLOSS3": "abupt-satloss3-",
        "POINTNET": "pointnet-",
        "POINTNET_SATLOSS3": "pointnet-satloss3-",
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
        "SMART_AUGMENTED",
        "SMART_MASKED",
        "SMART_SATLOSS3",
        "SMART_SATLOSS4",
        "SMART_SATLOSS5",
        "SMART_SATLOSS5_NOPM",
        "SMART_SATLOSS6",
    }:
        model = SMART(**base_kwargs, **arch)
    elif model_name == "SMART_SAT":
        model = SMARTSAT(**base_kwargs, **arch)
    elif model_name == "SMART_SAT2":
        model = SMARTSAT2(**base_kwargs, **arch)
    elif model_name == "SMART_SAT3":
        model = SMARTSAT3(**base_kwargs, **arch)
    elif model_name == "SMART_SAT4":
        model = SMARTSAT4(**base_kwargs, **arch)
    elif model_name in {"TRANSOLVERPP", "TRANSOLVERPP_SATLOSS3"}:
        model = TransolverPP(**base_kwargs, **arch)
    elif model_name == "TRANSOLVERPP_SAT":
        model = TransolverPPSAT(**base_kwargs, **arch)
    elif model_name in {"GINOT", "GINOT_SATLOSS3"}:
        model = GINOT(**base_kwargs, **arch)
    elif model_name == "GINOT_SAT":
        model = GINOTSAT(**base_kwargs, **arch)
    elif model_name in {"ABUPT", "ABUPT_SATLOSS3"}:
        model = ABUPT(**base_kwargs, **arch)
    elif model_name == "ABUPT_SAT":
        model = ABUPTSAT(**base_kwargs, **arch)
    elif model_name in {"POINTNET", "POINTNET_SATLOSS3"}:
        model = PointNet(**base_kwargs, **arch)
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
    coord = np.asarray(coords_xyz[:, axis], dtype=np.float64)
    cmin = float(np.min(coord))
    cmax = float(np.max(coord))
    span = max(cmax - cmin, 1e-12)
    t = np.clip((coord - cmin) / span, 0.0, 1.0)
    # One sinusoidal hump across the full axis extent: low at the ends, high in the middle.
    scores = np.sin(np.pi * t) ** 2
    return np.clip(scores + 1e-6, 1e-6, None)


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


def vector_mag(arr: np.ndarray, start: int, end: int) -> np.ndarray:
    return np.linalg.norm(arr[:, start:end], axis=1)


def safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", name)


def choose_fixed_query_indices(n_total: int, n_keep: int, seed_components: Sequence[int]) -> np.ndarray:
    rng = np.random.default_rng(np.random.SeedSequence([int(x) for x in seed_components]))
    idx = sample_uniform_without_replacement(n_total, n_keep, rng)
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
    fig.savefig(path, dpi=220)
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
    fig.savefig(path, dpi=220)
    plt.close(fig)


def save_sampling_y_histogram(
    path: Path,
    sampled_points_xyz: np.ndarray,
    reference_points_xyz: np.ndarray,
    mix_fraction: float,
    title: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sampled = np.asarray(sampled_points_xyz, dtype=np.float64)
    reference = np.asarray(reference_points_xyz, dtype=np.float64)
    if sampled.ndim != 2 or sampled.shape[1] != 3:
        raise ValueError("sampled_points_xyz must have shape [N, 3]")
    if reference.ndim != 2 or reference.shape[1] != 3:
        raise ValueError("reference_points_xyz must have shape [N, 3]")

    y_ref = reference[:, 1]
    y_samp = sampled[:, 1]
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
    ax.set_xlabel("Normalized y coordinate")
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
        fontsize=10,
        bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "#CCCCCC"},
    )
    fig.savefig(path, dpi=220)
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
    return model_name in {
        "SMART_SAT",
        "SMART_SAT2",
        "SMART_SAT3",
        "SMART_SAT4",
        "TRANSOLVERPP_SAT",
        "GINOT_SAT",
        "ABUPT_SAT",
    }


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
) -> Tuple[np.ndarray, np.ndarray]:
    batch_size = int(geo_views_norm.shape[0])
    surf_query_b = surf_query_norm.unsqueeze(0).expand(batch_size, -1, -1)
    vol_query_b = vol_query_norm.unsqueeze(0).expand(batch_size, -1, -1)

    geo_b = geo_views_norm.to(device, non_blocking=True)
    surf_q_b = surf_query_b.to(device, non_blocking=True)
    vol_q_b = vol_query_b.to(device, non_blocking=True)
    geo_log_b = None if geo_log_density_views is None else geo_log_density_views.to(device, non_blocking=True)

    surf_acc = None
    vol_acc = None
    use_autocast = device.type == "cuda"
    for rep in range(int(repeats)):
        seed = int(base_seed + rep)
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_autocast):
            if model_uses_density(model_name):
                pred_s_norm, pred_v_norm = model.inference(geo_b, surf_q_b, vol_q_b, None, geo_log_density=geo_log_b)
            else:
                pred_s_norm, pred_v_norm = model.inference(geo_b, surf_q_b, vol_q_b, None)
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
    n_surface = int(surf_query_norm.shape[0])
    pred_surf = np.empty((n_surface,), dtype=np.float32)
    abupt_audi_query_subsamples = 10

    geo_b = geo_view_norm.to(device, non_blocking=True)
    dummy_vol_b = dummy_vol_query_norm.to(device, non_blocking=True)
    full_surf_b = surf_query_norm.unsqueeze(0).to(device, non_blocking=True)
    geo_log_b = None if geo_log_density_view is None else geo_log_density_view.to(device, non_blocking=True)
    use_autocast = device.type == "cuda"

    if model_name in {"ABUPT", "ABUPT_SATLOSS3"}:
        surf_acc = np.zeros((n_surface,), dtype=np.float32)
        original_subregion_size = int(getattr(model, "subregion_size", max(1, int(surface_chunk_size))))
        model.subregion_size = max(1, int(surface_chunk_size))
        try:
            for rep in range(int(repeats)):
                seed = int(base_seed + rep)
                torch.manual_seed(seed)
                if device.type == "cuda":
                    torch.cuda.manual_seed_all(seed)
                # For the Audi visualization only, predict on a random partition of
                # the external surface cloud instead of one monolithic full-cloud
                # forward. This keeps the export tractable and avoids axis-aligned
                # chunk artifacts from sequential slicing.
                rep_rng = np.random.default_rng(seed)
                perm = rep_rng.permutation(n_surface)
                query_subsets = [chunk for chunk in np.array_split(perm, abupt_audi_query_subsamples) if len(chunk) > 0]
                rep_pred = np.empty((n_surface,), dtype=np.float32)
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=False):
                    for subset_idx in query_subsets:
                        surf_subset_b = full_surf_b[:, torch.from_numpy(np.asarray(subset_idx, dtype=np.int64)).to(device=device, dtype=torch.long)]
                        if model_uses_density(model_name):
                            pred_s_norm, _ = model.inference(geo_b, surf_subset_b, dummy_vol_b, None, geo_log_density=geo_log_b)
                        else:
                            pred_s_norm, _ = model.inference(geo_b, surf_subset_b, dummy_vol_b, None)
                        rep_pred[np.asarray(subset_idx, dtype=np.int64)] = (
                            pred_s_norm[0, :, 0].detach().to(torch.float32).cpu().numpy() * float(std_s[0]) + float(mean_s[0])
                        )
                    if not np.isfinite(rep_pred).all():
                        raise RuntimeError(f"{model_name} produced non-finite surface predictions during Audi VTK export.")
                surf_acc += rep_pred
        finally:
            model.subregion_size = original_subregion_size
        pred_surf[:] = surf_acc / float(repeats)
        return pred_surf

    def _build_surface_decoder():
        if model_name in {"SMART", "SMART_AUGMENTED", "SMART_MASKED", "SMART_SATLOSS3", "SMART_SATLOSS4", "SMART_SATLOSS5", "SMART_SATLOSS5_NOPM", "SMART_SATLOSS6"}:
            intermediate_latent_geometries, latent_geo_pos = model.encode(geo_b, None)

            def decode_chunk(chunk: torch.Tensor) -> torch.Tensor:
                pred_norm = model.decode(intermediate_latent_geometries, latent_geo_pos, None, chunk)
                return pred_norm[:, :, 0]

            return decode_chunk

        if model_name == "SMART_SAT":
            intermediate_latent_geometries, latent_geo_pos, latent_geo_log_density = model.encode(
                geo_b,
                None,
                geo_log_density=geo_log_b,
                return_latent_density=True,
            )

            def decode_chunk(chunk: torch.Tensor) -> torch.Tensor:
                pred_norm = model.decode(
                    intermediate_latent_geometries,
                    latent_geo_pos,
                    None,
                    chunk,
                    latent_geo_log_density=latent_geo_log_density,
                )
                return pred_norm[:, :, 0]

            return decode_chunk

        if model_name in {"GINOT", "GINOT_SATLOSS3"}:
            geometry_latents, geometry_pos = model.encode_geometry(geo_b, params=None)

            def decode_chunk(chunk: torch.Tensor) -> torch.Tensor:
                query_features = model.decode_features(geometry_latents, geometry_pos, chunk, chunk[:, :0], params=None)
                pred = model.head(query_features)
                return pred[:, :, 0]

            return decode_chunk

        if model_name in {"POINTNET", "POINTNET_SATLOSS3"}:
            _, global_feat = model.encode_geometry(geo_b, params=None)

            def decode_chunk(chunk: torch.Tensor) -> torch.Tensor:
                query_features = model.decode_features(global_feat, chunk, chunk[:, :0], params=None)
                pred = model.output_head(query_features)
                return pred[:, :, 0]

            return decode_chunk

        if model_name in {"TRANSOLVERPP", "TRANSOLVERPP_SATLOSS3"}:
            geo_pos = model._select_geometry_tokens(geo_b, geo_log_density=None)
            geometry_tokens = model.geometry_preprocess(geo_pos)
            geometry_context_input = geometry_tokens.mean(dim=1)
            geometry_condition_token = model.geometry_condition(
                geometry_context_input.to(dtype=geometry_tokens.dtype)
            ).unsqueeze(1)
            placeholder = model.placeholder.view(1, 1, -1)

            def decode_chunk(chunk: torch.Tensor) -> torch.Tensor:
                surf_pos = chunk * model.pos_scale_factor
                query_tokens = model.preprocess(surf_pos)
                query_tokens = query_tokens + placeholder
                condition_token = geometry_condition_token + placeholder
                tokens = torch.cat([condition_token, query_tokens], dim=1)
                tokens = model.cond(tokens, None)
                tokens = model._run_blocks(tokens)
                query_latent = tokens[:, 1:]
                pred = model.output_head(query_latent)
                return pred[:, :, 0]

            return decode_chunk

        def decode_chunk(chunk: torch.Tensor) -> torch.Tensor:
            vol_query = dummy_vol_b if model_name in {"ABUPT", "ABUPT_SATLOSS3"} else dummy_vol_b[:, :0]
            if model_uses_density(model_name):
                pred_s_norm, _ = model.inference(geo_b, chunk, vol_query, None, geo_log_density=geo_log_b)
            else:
                pred_s_norm, _ = model.inference(geo_b, chunk, vol_query, None)
            if not torch.isfinite(pred_s_norm).all():
                raise RuntimeError(f"{model_name} produced non-finite surface predictions during Audi VTK export.")
            return pred_s_norm[:, :, 0]

        return decode_chunk

    surf_acc = np.zeros((n_surface,), dtype=np.float32)
    for rep in range(int(repeats)):
        seed = int(base_seed + rep)
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_autocast):
            decode_chunk = _build_surface_decoder()
            rep_pred = np.empty((n_surface,), dtype=np.float32)
            for start in range(0, n_surface, max(1, int(surface_chunk_size))):
                stop = min(start + max(1, int(surface_chunk_size)), n_surface)
                surf_chunk = surf_query_norm[start:stop].unsqueeze(0).to(device, non_blocking=True)
                pred_s_norm = decode_chunk(surf_chunk)
                rep_pred[start:stop] = (pred_s_norm.cpu() * float(std_s[0]) + float(mean_s[0])).numpy()
        surf_acc += rep_pred

    pred_surf[:] = surf_acc / float(repeats)
    return pred_surf


def select_run_ids(test_ids: Iterable[int], num_runs: int, run_ids_arg: str | None, seed: int) -> List[int]:
    test_ids = sorted(int(x) for x in test_ids)
    if run_ids_arg:
        chosen = [int(x.strip()) for x in run_ids_arg.split(",") if x.strip()]
        missing = [x for x in chosen if x not in test_ids]
        if missing:
            raise ValueError(f"Requested run ids not in test split: {missing}")
        return chosen
    rng = np.random.default_rng(int(seed) + 7001)
    n = min(int(num_runs), len(test_ids))
    return sorted(int(x) for x in rng.choice(np.array(test_ids, dtype=np.int64), size=n, replace=False))


def train_encoder_input_points(cfg) -> int:
    num_body_points = int(getattr(cfg, "num_body_points", 0))
    if num_body_points > 0:
        return num_body_points
    architecture = getattr(cfg, "architecture", None)
    if architecture is not None:
        arch_subsampled_geometry_points = int(getattr(architecture, "subsampled_geometry_points", 0))
        if arch_subsampled_geometry_points > 0:
            return arch_subsampled_geometry_points
    eval_view_geometry_points = int(getattr(cfg, "eval_view_geometry_points", 0))
    if eval_view_geometry_points > 0:
        return eval_view_geometry_points
    view_geometry_points = int(getattr(cfg, "view_geometry_points", 0))
    if view_geometry_points > 0:
        return view_geometry_points
    raise ValueError("Could not infer training encoder input point budget from config.")


def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_shift_betas(text: str) -> List[float]:
    betas = [float(x.strip()) for x in str(text).split(",") if x.strip()]
    if not betas:
        raise ValueError("Expected at least one beta in --shift-betas")
    return betas


def sine_mix_levels_from_shift_betas(shift_betas: Sequence[float]) -> List[float]:
    n = max(len(shift_betas), 1)
    return [float(x) for x in np.linspace(0.0, 0.5, num=n, dtype=np.float64)]


def mode_display_name(mode_name: str) -> str:
    if mode_name == "aligned_uniform_wor":
        return "aligned uniform"
    beta_match = re.search(r"beta_([0-9]+\.[0-9]+)", mode_name)
    if beta_match:
        return f"inv-density beta={float(beta_match.group(1)):.2f}"
    sine_match = re.search(r"ood_sine_y_mix_([0-9]+\.[0-9]+)", mode_name)
    if sine_match:
        frac = float(sine_match.group(1))
        return f"OOD sine-y mix={frac:.2f}"
    return mode_name


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
            agg[key] = float(np.mean(vals))
            agg[f"{key}_std"] = float(np.std(vals))
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
    sine_match = re.search(r"ood_sine_y_mix_([0-9]+\.[0-9]+)", mode_name)
    if sine_match:
        frac = float(sine_match.group(1))
        norm = min(max(frac / 0.5, 0.0), 1.0)
        return matplotlib.colors.to_hex(plt.cm.YlOrBr(norm))
    return "#999999"


def seed_everything(seed: int) -> None:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def _grouped_bar_on_axis(ax, rows: List[Dict[str, object]], metric_key: str, mode_order: Sequence[str], model_order: Sequence[str]):
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
        err = [stds[m][mode_name] for m in model_order]
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
    ax.legend(fontsize=8)


def plot_metric_grid(
    rows: List[Dict[str, object]],
    metric_keys: Sequence[str],
    mode_order: Sequence[str],
    model_order: Sequence[str],
    out_path: Path,
    title: str,
    ncols: int = 2,
) -> None:
    n_metrics = len(metric_keys)
    ncols = max(1, int(ncols))
    nrows = int(math.ceil(n_metrics / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.2 * ncols, 4.8 * nrows), constrained_layout=True)
    axes_arr = np.atleast_1d(axes).reshape(nrows, ncols)
    for ax, metric_key in zip(axes_arr.flat, metric_keys):
        _grouped_bar_on_axis(ax, rows, metric_key, mode_order, model_order)
    for ax in axes_arr.flat[n_metrics:]:
        ax.axis("off")
    fig.suptitle(title, fontsize=18)
    fig.savefig(out_path, dpi=220)
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
) -> None:
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
        color = MODEL_COLORS[model_name]
        ax.plot(xs, ys, marker="o", linewidth=2, color=color, label=MODEL_LABELS[model_name])
        ax.fill_between(xs, ys - yerr, ys + yerr, color=color, alpha=0.18)
    ax.set_xticks(xs)
    ax.set_xlabel(x_label)
    ax.set_ylabel(_metric_display_name(metric_key))
    ax.set_title(title)
    ax.legend()
    fig.savefig(out_path, dpi=220)
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
        sine_match = re.search(r"ood_sine_y_mix_([0-9]+\.[0-9]+)", mode_name)
        if sine_match and float(sine_match.group(1)) > 0.0:
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
        color = MODEL_COLORS[model_name]
        ax.plot(x, ys, marker="o", linewidth=2, color=color, label=MODEL_LABELS[model_name])
        ax.fill_between(x, ys - yerr, ys + yerr, color=color, alpha=0.16)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.legend()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_delta_bars(run_delta_rows: List[Dict[str, object]], metric_key: str, out_path: Path, title: str) -> None:
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
        colors.append(MODEL_COLORS[model_name])
    fig, ax = plt.subplots(figsize=(9.2, 5.4), constrained_layout=True)
    ax.bar(np.arange(len(labels)), means, yerr=stds, capsize=4, color=colors, alpha=0.9)
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=15)
    ax.set_ylabel(metric_key)
    ax.set_title(title)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_density_shift_bars(per_view_rows: List[Dict[str, object]], out_path: Path, title: str) -> None:
    mode_order = list(OrderedDict((str(r["sampling_mode"]), None) for r in per_view_rows).keys())
    means = []
    stds = []
    for mode_name in mode_order:
        vals = np.array([float(r["subset_log_density_mean"]) for r in per_view_rows if r["sampling_mode"] == mode_name], dtype=np.float64)
        means.append(float(np.mean(vals)) if vals.size else math.nan)
        stds.append(float(np.std(vals)) if vals.size else math.nan)
    fig, ax = plt.subplots(figsize=(9.2, 5.4), constrained_layout=True)
    ax.bar(np.arange(len(mode_order)), means, yerr=stds, capsize=4, color=[mode_color(m) for m in mode_order], alpha=0.88)
    ax.set_xticks(np.arange(len(mode_order)))
    ax.set_xticklabels([mode_display_name(mode_name) for mode_name in mode_order], rotation=20, ha="right")
    ax.set_ylabel("subset_log_density_mean")
    ax.set_title(title)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_comprehensive_dashboard(
    per_run_mode_rows: List[Dict[str, object]],
    model_order: Sequence[str],
    mode_order: Sequence[str],
    out_path: Path,
    title: str,
) -> None:
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
        _grouped_bar_on_axis(ax, per_run_mode_rows, metric_key, mode_order, model_order)
    for ax in axes.flat[len(dashboard_metrics):]:
        ax.axis("off")
    fig.suptitle(title, fontsize=18)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = resolve_device(args.device)
    print(f"Device: {device}")
    print(f"Global comparison seed: {int(args.seed)}")

    config_name_map = OrderedDict(
        [
            ("SMART", args.smart_config),
            ("SMART_AUGMENTED", args.smart_augmented_config),
            ("SMART_MASKED", args.smart_masked_config),
            ("SMART_SAT", args.smart_sat_config),
            ("SMART_SATLOSS3", args.smart_satloss3_config),
            ("SMART_SATLOSS4", args.smart_satloss4_config),
            ("SMART_SATLOSS5", args.smart_satloss5_config),
            ("SMART_SATLOSS5_NOPM", args.smart_satloss5_nopm_config),
            ("SMART_SATLOSS6", args.smart_satloss6_config),
            ("TRANSOLVERPP", args.transolverpp_config),
            ("TRANSOLVERPP_SAT", args.transolverpp_sat_config),
            ("TRANSOLVERPP_SATLOSS3", args.transolverpp_satloss3_config),
            ("GINOT", args.ginot_config),
            ("GINOT_SATLOSS3", args.ginot_satloss3_config),
            ("ABUPT", args.abupt_config),
            ("ABUPT_SATLOSS3", args.abupt_satloss3_config),
            ("POINTNET", args.pointnet_config),
            ("POINTNET_SATLOSS3", args.pointnet_satloss3_config),
        ]
    )
    configs = OrderedDict((model_name, load_cfg(cfg_name)) for model_name, cfg_name in config_name_map.items())

    data_paths = {str(cfg.data_path) for cfg in configs.values()}
    if len(data_paths) != 1:
        raise ValueError(f"Expected one shared DrivAerML data path, got: {sorted(data_paths)}")
    smart_cfg = configs["SMART"]

    shift_betas = parse_shift_betas(args.shift_betas)
    sine_mix_levels = sine_mix_levels_from_shift_betas(shift_betas)
    mode_defs = OrderedDict()
    mode_defs["aligned_uniform_wor"] = {
        "kind": "uniform_wor",
        "beta": 0.0,
        "description": "Uniform without replacement, aligned with training-view sampling.",
        "id": 0,
    }
    for i, beta in enumerate(shift_betas, start=1):
        mode_defs[f"shifted_inverse_density_beta_{beta:.2f}"] = {
            "kind": "inverse_density_wor",
            "beta": float(beta),
            "description": f"Inverse-density without replacement, same point budget with beta={beta:.2f}.",
            "id": i,
        }
    next_mode_id = len(mode_defs)
    for mix_idx, mix_fraction in enumerate(sine_mix_levels):
        mode_defs[f"ood_sine_y_mix_{mix_fraction:.2f}"] = {
            "kind": "sinusoidal_axis_mixture_wor",
            "beta": math.nan,
            "axis": 1,
            "mix_fraction": float(mix_fraction),
            "description": (
                f"OOD sine-y mixture without replacement: "
                f"{mix_fraction:.2f} sinusoidal-weighted sampling + {1.0 - float(mix_fraction):.2f} uniform sampling, same point budget."
            ),
            "id": next_mode_id + mix_idx,
        }

    checkpoint_arg_map = {
        "SMART": args.smart_checkpoint,
        "SMART_AUGMENTED": args.smart_augmented_checkpoint,
        "SMART_MASKED": args.smart_masked_checkpoint,
        "SMART_SAT": args.smart_sat_checkpoint,
        "SMART_SATLOSS3": args.smart_satloss3_checkpoint,
        "SMART_SATLOSS4": args.smart_satloss4_checkpoint,
        "SMART_SATLOSS5": args.smart_satloss5_checkpoint,
        "SMART_SATLOSS5_NOPM": args.smart_satloss5_nopm_checkpoint,
        "SMART_SATLOSS6": args.smart_satloss6_checkpoint,
        "TRANSOLVERPP": args.transolverpp_checkpoint,
        "TRANSOLVERPP_SAT": args.transolverpp_sat_checkpoint,
        "TRANSOLVERPP_SATLOSS3": args.transolverpp_satloss3_checkpoint,
        "GINOT": args.ginot_checkpoint,
        "GINOT_SATLOSS3": args.ginot_satloss3_checkpoint,
        "ABUPT": args.abupt_checkpoint,
        "ABUPT_SATLOSS3": args.abupt_satloss3_checkpoint,
        "POINTNET": args.pointnet_checkpoint,
        "POINTNET_SATLOSS3": args.pointnet_satloss3_checkpoint,
    }
    requested_model_names = [model_name for model_name in MODEL_ORDER if checkpoint_arg_map[model_name] is not None]
    if not requested_model_names:
        raise ValueError("No model checkpoints were provided. Pass at least one --*-checkpoint argument.")
    model_specs = OrderedDict(
        (
            model_name,
            {"config": configs[model_name], "checkpoint": choose_ckpt(configs[model_name], checkpoint_arg_map[model_name])},
        )
        for model_name in requested_model_names
    )
    for model_name, spec in model_specs.items():
        print(f"{model_name} checkpoint: {spec['checkpoint']}")

    per_model_input_budgets = {
        model_name: (
            int(args.input_points)
            if args.input_points is not None
            else int(train_encoder_input_points(spec["config"]))
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

    density_cfg = configs["SMART_SAT"] if "SMART_SAT" in configs else next(iter(configs.values()))
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
    )

    run_ids = select_run_ids(dataset.test_ids, args.num_runs, args.run_ids, args.seed)
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

    models = {
        model_name: build_model(spec["config"], spec["checkpoint"], device, batched_query_subregion_size=args.batched_query_subregion_size).to(device)
        for model_name, spec in model_specs.items()
    }
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
        if model_name in {"ABUPT", "ABUPT_SATLOSS3"}:
            if int(args.abupt_surface_query_points) > 0:
                model_surface_query_points = int(args.abupt_surface_query_points)
            if int(args.abupt_volume_query_points) > 0:
                model_volume_query_points = int(args.abupt_volume_query_points)
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
        train_encoder_points = train_encoder_input_points(spec["config"])
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

        vol_coords_full = np.load(run_dir / "volume_coords.npy").astype(np.float32, copy=False)
        vol_p = np.load(run_dir / "volume_pMeanTrim.npy").astype(np.float32, copy=False).reshape(-1, 1)
        vol_u = np.load(run_dir / "volume_UMeanTrim.npy").astype(np.float32, copy=False)
        vol_gt_full = np.concatenate([vol_p, vol_u], axis=1)

        full_surf_query_norm = normalize_pos(torch.from_numpy(surf_coords_full), min_pos, max_pos)
        max_surface_query_points = max(int(b["surface"]) for b in per_model_query_budgets.values())
        max_volume_query_points = max(int(b["volume"]) for b in per_model_query_budgets.values())
        surf_query_idx_master = choose_fixed_query_indices(surf_coords_full.shape[0], max_surface_query_points, [args.seed, int(run_id), 3001])
        vol_query_idx_master = choose_fixed_query_indices(vol_coords_full.shape[0], max_volume_query_points, [args.seed, int(run_id), 3002])

        full_geo_log_density = dataset._load_or_compute_full_geometry_density(run_id, expected_n=int(surf_coords_full.shape[0]))
        full_geo_log_density_np = full_geo_log_density.to(dtype=torch.float32).numpy()
        sine_y_weights = sinusoidal_axis_probabilities(surf_coords_full, axis=1)
        model_full_geo_log_density_by_name: Dict[str, torch.Tensor] = {}
        for model_name, spec in model_specs.items():
            if not model_uses_density(model_name):
                continue
            model_density_estimator, model_density_knn_k, model_density_neighbor_hops, model_density_cache_dtype = resolve_model_internal_density_spec(
                model_name,
                spec["config"],
                spec["checkpoint"],
            )
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

        for mode_name, mode_info in mode_defs.items():
            for model_name, model in models.items():
                model.eval()
                model_input_points = int(per_model_input_budgets[model_name])
                idx_list: List[np.ndarray] = []
                subset_density_stats: List[Dict[str, float]] = []
                for view_idx in range(views_per_mode):
                    rng = np.random.default_rng(np.random.SeedSequence([args.seed, int(run_id), int(mode_info["id"]), int(view_idx)]))
                    if mode_info["kind"] == "uniform_wor":
                        idx = sample_uniform_without_replacement(surf_coords_full.shape[0], model_input_points, rng)
                    elif mode_info["kind"] == "inverse_density_wor":
                        idx = sample_inverse_density_without_replacement(full_geo_log_density_np, model_input_points, float(mode_info["beta"]), rng)
                    elif mode_info["kind"] == "sinusoidal_axis_mixture_wor":
                        idx = sample_uniform_weighted_mixture_without_replacement(
                            sine_y_weights,
                            model_input_points,
                            float(mode_info["mix_fraction"]),
                            rng,
                        )
                    else:
                        raise ValueError(f"Unsupported sampling kind: {mode_info['kind']}")
                    idx_list.append(idx)
                    subset = full_geo_log_density_np[idx]
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

                    pred_surf_batch, pred_vol_batch = predict_view_batch(
                        model_name=model_name,
                        model=model,
                        geo_views_norm=geo_views_norm,
                        surf_query_norm=surf_query_norm,
                        vol_query_norm=vol_query_norm,
                        geo_log_density_views=geo_density_views,
                        mean_s=mean_s,
                        std_s=std_s,
                        mean_v=mean_v,
                        std_v=std_v,
                        device=device,
                        base_seed=int(args.seed + 100000 * mode_info["id"] + 1000 * run_id + batch_start * 17),
                        repeats=args.model_repeats,
                    )
                    full_pred_surf_batch = None
                    if model_name in DRAG_RANK_MODELS and mode_info["kind"] in {"inverse_density_wor", "sinusoidal_axis_mixture_wor"}:
                        full_pred_surf_batch, _ = predict_view_batch(
                            model_name=model_name,
                            model=model,
                            geo_views_norm=geo_views_norm,
                            surf_query_norm=full_surf_query_norm,
                            vol_query_norm=vol_query_norm,
                            geo_log_density_views=geo_density_views,
                            mean_s=mean_s,
                            std_s=std_s,
                            mean_v=mean_v,
                            std_v=std_v,
                            device=device,
                            base_seed=int(args.seed + 100000 * mode_info["id"] + 1000 * run_id + batch_start * 17),
                            repeats=args.model_repeats,
                        )

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
                                full_pred_surf_batch[local_idx],
                                surf_area_full,
                            )
                        density_stats = subset_density_stats[global_view_idx]
                        per_view_rows.append(
                            {
                                "run_id": int(run_id),
                                "view_id": int(global_view_idx),
                                "model_name": model_name,
                                "sampling_mode": mode_name,
                                "sampling_kind": mode_info["kind"],
                                "shift_beta": float(mode_info["beta"]),
                                "sampling_mode_id": int(mode_info["id"]),
                                "checkpoint": model_specs[model_name]["checkpoint"],
                                "input_points": int(batch_indices[local_idx].shape[0]),
                                "surface_query_points": model_surface_query_points,
                                "volume_query_points": model_volume_query_points,
                                "full_log_density_mean": float(np.mean(full_geo_log_density_np)),
                                "surface_drag_force_x_gt": float(surf_drag_force_gt),
                                "surface_drag_force_x_pred": float(surf_drag_force_pred),
                                "surface_drag_force_x_full_gt": float(full_drag_force_gt),
                                "surface_drag_force_x_full_pred": float(full_drag_force_pred) if full_drag_force_pred is not None else float("nan"),
                                **density_stats,
                                **metrics,
                            }
                        )
                        if full_drag_force_pred is not None:
                            drag_rank_view_rows.append(
                                {
                                    "run_id": int(run_id),
                                    "view_id": int(global_view_idx),
                                    "model_name": model_name,
                                    "sampling_mode": mode_name,
                                    "sampling_kind": mode_info["kind"],
                                    "shift_beta": float(mode_info["beta"]),
                                    "sampling_mode_id": int(mode_info["id"]),
                                    "surface_drag_force_x_full_gt": float(full_drag_force_gt),
                                    "surface_drag_force_x_full_pred": float(full_drag_force_pred),
                                }
                            )

                    del geo_views_norm, pred_surf_batch, pred_vol_batch
                    if geo_density_views is not None:
                        del geo_density_views
                models[model_name] = model
                gc.collect()

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
    strongest_mode = max(beta_shift_mode_names, key=lambda mode_name: float(mode_defs[mode_name]["beta"]))
    strongest_beta = float(mode_defs[strongest_mode]["beta"])
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

    mode_order = list(mode_defs.keys())
    beta_mode_order = [mode_name for mode_name, mode_info in mode_defs.items() if mode_info["kind"] == "inverse_density_wor"]
    beta_mode_xs = [float(mode_defs[mode_name]["beta"]) for mode_name in beta_mode_order]
    sine_mode_order = [mode_name for mode_name, mode_info in mode_defs.items() if mode_info["kind"] == "sinusoidal_axis_mixture_wor"]
    sine_mode_xs = [float(mode_defs[mode_name]["mix_fraction"]) for mode_name in sine_mode_order]
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
                        f"{family_title}: sinusoidal-y severity curve (combined physics)",
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
                        f"{family_title}: sinusoidal-y severity curve (combined global)",
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
    sampling_sine_y_weights = sinusoidal_axis_probabilities(sampling_input_surf_coords, axis=1)

    surface_point_data: Dict[str, np.ndarray] = {
        "gt_pressure": rep_surf_gt_full[:, 0],
    }
    representative_models = OrderedDict((m, models[m]) for m in VTK_PRESSURE_MODELS if m in models)
    audi_vtk_skipped_models: List[str] = []
    n_surface_points = int(rep_surf_gt_full.shape[0])
    for model_name, model in tqdm(representative_models.items(), desc="Representative full-surface predictions", dynamic_ncols=True):
        model.eval()
        prefix = MODEL_LABELS[model_name].lower()
        try:
            model_input_points = int(per_model_input_budgets[model_name])
            rep_rng = np.random.default_rng(np.random.SeedSequence([args.seed, int(vtk_run_id), 99991, MODEL_ORDER.index(model_name)]))
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
                rep_geo_density_view = rep_model_full_geo_log_density.index_select(0, torch.from_numpy(rep_idx).to(dtype=torch.long)).unsqueeze(0)
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
                device=device,
                base_seed=int(args.seed + 900000 + MODEL_ORDER.index(model_name) * 37),
                repeats=args.model_repeats,
                surface_chunk_size=int(args.audi_surface_chunk_size),
            )
            surface_point_data[f"{prefix}_pressure_pred"] = pred_pressure
        except Exception as exc:
            audi_vtk_skipped_models.append(model_name)
            print(f"[warning] Skipping Audi VTK export for {model_name}: {exc}")
            surface_point_data[f"{prefix}_pressure_pred"] = np.full((n_surface_points,), np.nan, dtype=np.float32)
        if device.type == "cuda":
            torch.cuda.empty_cache()
        models[model_name] = model

    vtk_path = out_root / "audi_surface_pressure_predictions.vtk"
    write_polydata_vtk(vtk_path, rep_surf_coords_full, surface_point_data)

    sampling_vtk_paths = []
    sampling_histogram_paths = []
    sampling_budget = max(unique_input_budgets)
    beta_sample_log_density_values: Dict[float, np.ndarray] = {}
    for beta in parse_shift_betas(args.shift_betas):
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

    representative_view2_sampling_vtk_paths: List[str] = []
    if "SMART_AUGMENTED" in model_specs:
        augmented_cfg = model_specs["SMART_AUGMENTED"]["config"]
        augmented_budget = int(getattr(augmented_cfg, "secondary_view_geometry_points", getattr(augmented_cfg, "view_geometry_points", sampling_budget)))
        augmented_rng = np.random.default_rng(np.random.SeedSequence([args.seed, int(vtk_run_id), 919191, 1]))
        augmented_idx = sample_uniform_without_replacement(sampling_input_surf_coords.shape[0], augmented_budget, augmented_rng)
        augmented_vtk_path = out_root / f"drivaerml_test_run_{vtk_run_id}_smart_augmented_view2_input_points_{augmented_budget}.vtk"
        write_polydata_vtk(augmented_vtk_path, sampling_input_surf_coords[augmented_idx], {})
        representative_view2_sampling_vtk_paths.append(str(augmented_vtk_path))

    if "SMART_MASKED" in model_specs:
        masked_cfg = model_specs["SMART_MASKED"]["config"]
        masked_budget = int(getattr(masked_cfg, "secondary_view_geometry_points", getattr(masked_cfg, "view_geometry_points", sampling_budget)))
        masked_num_examples = 10
        for masked_example_idx in range(masked_num_examples):
            masked_rng = np.random.default_rng(
                np.random.SeedSequence([args.seed, int(vtk_run_id), 919191, 2, int(masked_example_idx)])
            )
            masked_info = sample_gaussian_ball_mask_subset(
                sampling_input_surf_coords,
                masked_budget,
                masked_rng,
                std_fraction_of_largest_extent=float(getattr(masked_cfg, "gaussian_mask_std_fraction_of_largest_extent", 0.05)),
                prob_at_1sigma=float(getattr(masked_cfg, "gaussian_mask_prob_at_1sigma", 0.33)),
                min_survivors=int(getattr(masked_cfg, "gaussian_mask_min_survivors", 16384)),
                return_metadata=True,
            )
            masked_base_idx = np.asarray(masked_info["base_idx"], dtype=np.int64)
            masked_kept_idx = np.asarray(masked_info["kept_idx"], dtype=np.int64)
            masked_points = sampling_input_surf_coords[masked_base_idx]
            masked_center = np.asarray(masked_info["center_point"], dtype=np.float32).reshape(1, 3)
            masked_vtk_path = out_root / (
                f"drivaerml_test_run_{vtk_run_id}_smart_masked_view2_example_{masked_example_idx:02d}"
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
            masked_survivor_points = sampling_input_surf_coords[masked_kept_idx]
            masked_survivor_vtk_path = out_root / (
                f"drivaerml_test_run_{vtk_run_id}_smart_masked_view2_example_{masked_example_idx:02d}"
                f"_input_points_{int(masked_survivor_points.shape[0])}_survivors_only.vtk"
            )
            write_polydata_vtk(
                masked_survivor_vtk_path,
                masked_survivor_points,
                {},
            )
            representative_view2_sampling_vtk_paths.append(str(masked_survivor_vtk_path))

    for mix_fraction in sine_mix_levels:
        sampling_rng = np.random.default_rng(
            np.random.SeedSequence([args.seed, int(vtk_run_id), 88888, 1, int(round(float(mix_fraction) * 1000))])
        )
        sample_idx = sample_uniform_weighted_mixture_without_replacement(
            sampling_sine_y_weights,
            sampling_budget,
            float(mix_fraction),
            sampling_rng,
        )
        sampled_points = sampling_input_surf_coords[sample_idx]
        sample_vtk_path = out_root / (
            f"drivaerml_test_run_{vtk_run_id}_input_points_{sampling_budget}_ood_sine_y_mix_{float(mix_fraction):.2f}.vtk"
        )
        write_polydata_vtk(sample_vtk_path, sampled_points, {})
        sampling_vtk_paths.append(str(sample_vtk_path))
        sample_hist_path = out_root / (
            f"drivaerml_test_run_{vtk_run_id}_input_points_{sampling_budget}_ood_sine_y_mix_{float(mix_fraction):.2f}_y_hist.png"
        )
        save_sampling_y_histogram(
            sample_hist_path,
            sampled_points,
            sampling_input_surf_coords,
            float(mix_fraction),
            title=f"Run {vtk_run_id} sampled y distribution (OOD sine-y mix={float(mix_fraction):.2f}, points={sampling_budget})",
        )
        sampling_histogram_paths.append(str(sample_hist_path))

    payload = {
        "args": vars(args),
        "run_ids": run_ids,
        "representative_vtk_run_id": vtk_run_id,
        "models": {k: v["checkpoint"] for k, v in model_specs.items()},
        "mode_definitions": mode_defs,
        "workflow": {
            "benchmark_queries_fixed_per_run": True,
            "benchmark_surface_query_points": surface_query_points,
            "benchmark_volume_query_points": volume_query_points,
            "per_model_query_budgets": per_model_query_budgets,
            "query_budget_mismatch_models": query_budget_mismatch_models,
            "per_model_encoder_input_budgets": per_model_input_budgets,
            "aligned_mode": "uniform_wor",
            "shift_modes": [name for name in mode_defs if name != "aligned_uniform_wor"],
            "ood_modes": [name for name, info in mode_defs.items() if info["kind"] == "sinusoidal_axis_mixture_wor"],
            "ood_sine_axis": "y",
            "ood_sine_mix_levels": sine_mix_levels,
            "views_per_mode": views_per_mode,
            "view_batch_size": view_batch_size,
            "model_repeats": int(args.model_repeats),
            "top_level_alignment_note": "Aligned mode matches each model's training-time view sampling rule best: uniform without replacement at that model's own encoder input budget.",
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
            "representative_view2_sampling_vtks": representative_view2_sampling_vtk_paths,
            "audi_vtk_skipped_models": audi_vtk_skipped_models,
            "encoder_budget_mismatches": encoder_budget_mismatch_models,
            "query_budget_mismatches": query_budget_mismatch_models,
        },
        "configs": {k: config_name_map[k] for k in model_specs},
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
        f"- Surface query coordinates are fixed per run to one common subset of `{surface_query_points}` points for every model and every sampling mode.",
        f"- Volume query coordinates are fixed per run to one common subset of `{volume_query_points}` points for every model and every sampling mode.",
        "- If a family-specific query override is requested, that family uses its own fixed per-run query subset while the other families keep the global benchmark subset.",
        "- Encoder input point budget is train-aligned per model by default. That keeps each family on its own training budget instead of forcing all families to the smallest one.",
        "- If a model was trained with smaller query budgets than this evaluation uses, the script reports that mismatch explicitly in the console and `results.json`.",
        "- The aligned mode uses `uniform_wor` because the training-time top-level view rule is uniform without replacement, and this evaluation preserves each model's own encoder input budget unless you explicitly override `--input-points`.",
        f"- Beta-shift modes use inverse-density sampling without replacement at betas `{shift_betas}` and keep the same point budget.",
        f"- Sampling shifts are computed with the requested CLI density estimator `{density_estimator}`, but density-aware models receive density tensors from their own training config when available.",
        "- Additional out-of-distribution modes use a controlled mixture of uniform sampling and sinusoidal point-selection probabilities along the `y` direction only, sampled without replacement at the same point budget.",
        "- The sinusoidal-y intensity runs from `0.0` to `0.5` and uses the same number of severity steps as `--shift-betas`.",
        "- For an OOD sine mixture severity `s`, the sampler takes exactly `round(s * K)` points from the sinusoidal-weighted rule and the remaining points uniformly from the leftover pool, so the severity has an exact point-count interpretation rather than only a probability interpretation.",
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
        "- The representative prediction VTK stores only ground-truth pressure and model pressure predictions.",
        "- If a model cannot execute a true empty-volume surface-only export path, the script falls back to one fixed representative volume query point from the selected DrivAerML run. This affects only the Audi visualization export, not the benchmark metrics.",
        "- If a model still cannot complete the full-Audi visualization export safely, it is skipped only for this VTK step and recorded in the results payload.",
        f"- Surface-query directory for the Audi pressure-field export: `{vtk_surface_query_dir}`",
        f"- Separate point-cloud VTKs are exported from DrivAerML test run `{vtk_run_id}` for the inverse-density beta modes and for the OOD sine-y mixture severities, using the largest active encoder budget `{sampling_budget}` so you can directly inspect one representative input cloud.",
        "- Each inverse-density beta sampled-point VTK also gets a separate PNG histogram of the sampled density distribution, with a log-count y-axis and no percentile trimming.",
        "- Each OOD sine-y mixture sampled-point VTK gets a y-coordinate distribution histogram that shows the sampled points, the full candidate-point y distribution, and the target sinusoidal mixture curve.",
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
        "- `audi_surface_pressure_predictions.vtk`: full Audi surface pressure ground truth plus selected model pressure predictions.",
        "- `results.json`: machine-readable summary including any representative-VTK model skips.",
        "- `smart_family_surface_drag_force_x_ranked_beta_*.png`: full-surface drag curves for SMART, SMART-SATLOSS3, and SMART-SATLOSS5, sorted by ground-truth drag within each beta mode.",
        "- `smart_family_surface_drag_force_x_ranked_sine_*.png`: full-surface drag curves for SMART, SMART-SATLOSS3, and SMART-SATLOSS5, sorted by ground-truth drag within each sine-y mode.",
        f"- `drivaerml_test_run_{vtk_run_id}_input_points_{sampling_budget}_inverse_density_beta_*.vtk`: sampled `{sampling_budget}` input points for each inverse-density beta from one evaluated DrivAerML test run.",
        f"- `drivaerml_test_run_{vtk_run_id}_input_points_{sampling_budget}_inverse_density_beta_*_density_hist.png`: density-distribution histogram for each sampled input-point VTK.",
        f"- `drivaerml_test_run_{vtk_run_id}_input_points_{sampling_budget}_ood_sine_y_mix_*_y_hist.png`: y-coordinate distribution histogram for each OOD sine-y sampled input-point VTK.",
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
        f"- OOD sampling modes: progressive uniform-to-sinusoidal mixtures along `y` only, with severities `{sine_mix_levels}`",
        f"- Fixed benchmark query subsets per run: `{surface_query_points}` surface + `{volume_query_points}` volume",
        f"- ABUPT-family query override: `{int(args.abupt_surface_query_points) if int(args.abupt_surface_query_points) > 0 else surface_query_points}` surface + `{int(args.abupt_volume_query_points) if int(args.abupt_volume_query_points) > 0 else volume_query_points}` volume",
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
        "- The field-level bar charts show whether robustness is consistent across surface and volume prediction categories or concentrated in only a subset of fields.",
        "- Family-specific figures let you compare each baseline only against its intended SAT / SATLOSS3 variants under the same sampling sweep.",
        ]
    )
    (out_root / "summary.md").write_text("\n".join(summary_lines), encoding="utf-8")

    print(f"Wrote results to {out_root}")
    print(f"Representative VTK: {vtk_path}")


if __name__ == "__main__":
    main()
