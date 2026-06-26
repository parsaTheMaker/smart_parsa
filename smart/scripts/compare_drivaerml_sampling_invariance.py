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
import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm.auto import tqdm


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
    "surface_normal_mag_rel_l2",
    "volume_pressure_rel_l2",
    "volume_velocity_mag_rel_l2",
    "combined_global_rel_l2",
    "combined_physics_rel_l2",
]

MODEL_ORDER = [
    "SMART",
    "SMART_SAT",
    "SMART_SATLOSS3",
    "SMART_SATLOSS4",
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
    "SMART_SAT": "SMART-SAT",
    "SMART_SATLOSS3": "SMART-SATLOSS3",
    "SMART_SATLOSS4": "SMART-SATLOSS4",
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
    "SMART_SAT": "#4C78A8",
    "SMART_SATLOSS3": "#F58518",
    "SMART_SATLOSS4": "#72B7B2",
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
FAMILY_GROUPS = OrderedDict(
    [
        ("smart_family", ["SMART", "SMART_SAT", "SMART_SATLOSS3", "SMART_SATLOSS4"]),
        ("transolverpp_family", ["TRANSOLVERPP", "TRANSOLVERPP_SAT", "TRANSOLVERPP_SATLOSS3"]),
        ("ginot_family", ["GINOT", "GINOT_SATLOSS3"]),
        ("abupt_family", ["ABUPT", "ABUPT_SATLOSS3"]),
        ("pointnet_family", ["POINTNET", "POINTNET_SATLOSS3"]),
    ]
)
FAMILY_TITLES = {
    "smart_family": "SMART vs SMART-SAT vs SMART-SATLOSS3 vs SMART-SATLOSS4",
    "transolverpp_family": "TransolverPP vs TransolverPP-SAT vs TransolverPP-SATLOSS3",
    "ginot_family": "GINOT vs GINOT-SATLOSS3",
    "abupt_family": "ABUPT vs ABUPT-SATLOSS3",
    "pointnet_family": "PointNet vs PointNet-SATLOSS3",
}
VTK_PRESSURE_MODELS = [
    "SMART",
    "SMART_SAT",
    "SMART_SATLOSS3",
    "SMART_SATLOSS4",
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
    p.add_argument("--smart-sat-config", default="drivaerml_sat")
    p.add_argument("--smart-satloss3-config", default="drivaerml_satloss3")
    p.add_argument("--smart-satloss4-config", default="drivaerml_satloss4")
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
    p.add_argument("--smart-sat-checkpoint", default=None)
    p.add_argument("--smart-satloss3-checkpoint", default=None)
    p.add_argument("--smart-satloss4-checkpoint", default=None)
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
    p.add_argument("--output-dir", default=None)
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
        "SMART_SAT": "smart-sat-",
        "SMART_SATLOSS3": "smart-satloss3-",
        "SMART_SATLOSS4": "smart-satloss4-",
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
    if model_name in {"SMART", "SMART_SATLOSS3", "SMART_SATLOSS4"}:
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
        model.subregion_size = min(int(model.subregion_size), int(batched_query_subregion_size))
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


def save_density_histogram(path: Path, log_density_values: np.ndarray, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    log_vals = np.asarray(log_density_values, dtype=np.float64)
    density_vals = np.exp(log_vals)
    finite_mask = np.isfinite(log_vals) & np.isfinite(density_vals)
    finite_density = density_vals[finite_mask]
    if finite_density.size == 0:
        finite_density = np.array([1.0], dtype=np.float64)
    lo_density = float(np.percentile(finite_density, 5.0))
    hi_density = float(np.percentile(finite_density, 90.0))
    clipped = finite_density[(finite_density >= lo_density) & (finite_density < hi_density)]
    if clipped.size == 0:
        clipped = finite_density[finite_density < hi_density]
    if clipped.size == 0:
        clipped = finite_density
    fig, ax = plt.subplots(figsize=(8.0, 5.0), constrained_layout=True)
    ax.hist(clipped, bins=60, range=(lo_density, hi_density), color="#4C78A8", alpha=0.9, edgecolor="white", linewidth=0.3)
    ax.set_xlabel("Density")
    ax.set_ylabel("Count")
    ax.set_yscale("log")
    ax.set_title(title)
    mean_val = float(np.mean(clipped))
    std_val = float(np.std(clipped))
    ax.axvline(mean_val, color="#E45756", linestyle="--", linewidth=1.5, label=f"mean={mean_val:.3g}")
    ax.legend()
    ax.text(
        0.98,
        0.95,
        f"std={std_val:.3g}\nN={clipped.size}\ntrim=[p5,p90 density]",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "#CCCCCC"},
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
        if model_name in {"SMART", "SMART_SATLOSS3", "SMART_SATLOSS4"}:
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
    return "#999999"


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
            label=mode_name,
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


def plot_shift_curve_with_band(aggregate_rows: List[Dict[str, object]], metric_key: str, out_path: Path, title: str, model_order: Sequence[str]) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 5.8), constrained_layout=True)
    for model_name in model_order:
        model_rows = [r for r in aggregate_rows if r["model_name"] == model_name]
        model_rows = sorted(model_rows, key=lambda r: float(r["shift_beta"]))
        xs = np.array([float(r["shift_beta"]) for r in model_rows], dtype=np.float64)
        ys = np.array([float(r[metric_key]) for r in model_rows], dtype=np.float64)
        yerr = np.array([float(r[f"{metric_key}_std"]) for r in model_rows], dtype=np.float64)
        color = MODEL_COLORS[model_name]
        ax.plot(xs, ys, marker="o", linewidth=2, color=color, label=MODEL_LABELS[model_name])
        ax.fill_between(xs, ys - yerr, ys + yerr, color=color, alpha=0.18)
    ax.set_xlabel("Shift severity beta")
    ax.set_ylabel(_metric_display_name(metric_key))
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
    ax.set_xticklabels(mode_order, rotation=15)
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
    fig, axes = plt.subplots(3, 2, figsize=(15, 15), constrained_layout=True)
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
    fig.suptitle(title, fontsize=18)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main():
    args = parse_args()
    device = resolve_device(args.device)
    print(f"Device: {device}")

    config_name_map = OrderedDict(
        [
            ("SMART", args.smart_config),
            ("SMART_SAT", args.smart_sat_config),
            ("SMART_SATLOSS3", args.smart_satloss3_config),
            ("SMART_SATLOSS4", args.smart_satloss4_config),
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

    checkpoint_arg_map = {
        "SMART": args.smart_checkpoint,
        "SMART_SAT": args.smart_sat_checkpoint,
        "SMART_SATLOSS3": args.smart_satloss3_checkpoint,
        "SMART_SATLOSS4": args.smart_satloss4_checkpoint,
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

    density_cfg = configs["SMART_SAT"]
    density_knn_k = int(getattr(density_cfg.architecture, "density_knn_k", 24))
    density_neighbor_hops = int(getattr(density_cfg.architecture, "density_neighbor_hops", 1))
    density_estimator = str(getattr(density_cfg.architecture, "density_estimator", "tangent_cov"))
    density_cache_dtype = str(getattr(density_cfg, "geometry_density_cache_dtype", "float16"))

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
        model_name: build_model(spec["config"], spec["checkpoint"], device, batched_query_subregion_size=args.batched_query_subregion_size)
        for model_name, spec in model_specs.items()
    }

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
    view_batch_size = max(1, int(args.view_batch_size))
    views_per_mode = max(1, int(args.views_per_mode))

    for run_id in tqdm(run_ids, desc="Runs", dynamic_ncols=True):
        run_dir = Path(smart_cfg.data_path) / f"run_{run_id}"
        surf_coords_full = np.load(run_dir / "surface_coords.npy").astype(np.float32, copy=False)
        surf_p = np.load(run_dir / "surface_pMeanTrim.npy").astype(np.float32, copy=False).reshape(-1, 1)
        surf_n = np.load(run_dir / "surface_normals.npy").astype(np.float32, copy=False)
        surf_wx = np.load(run_dir / "surface_wallShearStressMeanTrim_x.npy").astype(np.float32, copy=False).reshape(-1, 1)
        surf_wy = np.load(run_dir / "surface_wallShearStressMeanTrim_y.npy").astype(np.float32, copy=False).reshape(-1, 1)
        surf_wz = np.load(run_dir / "surface_wallShearStressMeanTrim_z.npy").astype(np.float32, copy=False).reshape(-1, 1)
        surf_gt_full = np.concatenate([surf_p, surf_n, surf_wx, surf_wy, surf_wz], axis=1)

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

        for mode_name, mode_info in mode_defs.items():
            for model_name, model in models.items():
                model = model.to(device)
                model.eval()
                model_input_points = int(per_model_input_budgets[model_name])
                idx_list: List[np.ndarray] = []
                subset_density_stats: List[Dict[str, float]] = []
                for view_idx in range(views_per_mode):
                    rng = np.random.default_rng(np.random.SeedSequence([args.seed, int(run_id), int(mode_info["id"]), int(view_idx)]))
                    if mode_info["kind"] == "uniform_wor":
                        idx = sample_uniform_without_replacement(surf_coords_full.shape[0], model_input_points, rng)
                    else:
                        idx = sample_inverse_density_without_replacement(full_geo_log_density_np, model_input_points, float(mode_info["beta"]), rng)
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
                vol_coords = vol_coords_full[vol_query_idx]
                vol_gt = vol_gt_full[vol_query_idx]
                surf_query_norm = normalize_pos(torch.from_numpy(surf_coords), min_pos, max_pos)
                vol_query_norm = normalize_pos(torch.from_numpy(vol_coords), min_pos, max_pos)
                for batch_start in range(0, views_per_mode, view_batch_size):
                    batch_stop = min(batch_start + view_batch_size, views_per_mode)
                    batch_indices = idx_list[batch_start:batch_stop]
                    geo_view_tensors = [full_surf_query_norm[torch.from_numpy(idx)] for idx in batch_indices]
                    geo_density_tensors = [
                        full_geo_log_density.index_select(0, torch.from_numpy(idx).to(dtype=torch.long))
                        for idx in batch_indices
                    ]
                    geo_views_norm = torch.stack(geo_view_tensors, dim=0)
                    geo_density_views = torch.stack(geo_density_tensors, dim=0)

                    pred_surf_batch, pred_vol_batch = predict_view_batch(
                        model_name=model_name,
                        model=model,
                        geo_views_norm=geo_views_norm,
                        surf_query_norm=surf_query_norm,
                        vol_query_norm=vol_query_norm,
                        geo_log_density_views=geo_density_views if model_uses_density(model_name) else None,
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
                        density_stats = subset_density_stats[global_view_idx]
                        per_view_rows.append(
                            {
                                "run_id": int(run_id),
                                "view_id": int(global_view_idx),
                                "model_name": model_name,
                                "sampling_mode": mode_name,
                                "sampling_kind": mode_info["kind"],
                                "shift_beta": float(mode_info["beta"]),
                                "checkpoint": model_specs[model_name]["checkpoint"],
                                "input_points": int(batch_indices[local_idx].shape[0]),
                                "surface_query_points": model_surface_query_points,
                                "volume_query_points": model_volume_query_points,
                                "full_log_density_mean": float(np.mean(full_geo_log_density_np)),
                                **density_stats,
                                **metrics,
                            }
                        )

                    del geo_views_norm, geo_density_views, pred_surf_batch, pred_vol_batch
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                model = model.to("cpu")
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
        "checkpoint",
        "input_points",
        "surface_query_points",
        "volume_query_points",
        "full_log_density_mean",
        "subset_log_density_mean",
        "subset_log_density_std",
        "subset_log_density_p05",
        "subset_log_density_p95",
    ] + per_view_metric_keys
    write_csv(out_root / "per_view_metrics.csv", per_view_rows, per_view_fieldnames)

    metric_keys = per_view_metric_keys + ["subset_log_density_mean", "subset_log_density_std"]
    per_run_mode_rows = aggregate_rows_by_keys(
        per_view_rows,
        ["run_id", "model_name", "sampling_mode", "sampling_kind", "shift_beta"],
        metric_keys,
    )
    per_run_mode_rows.sort(key=lambda x: (x["run_id"], MODEL_ORDER.index(x["model_name"]), x["shift_beta"]))
    write_csv(
        out_root / "per_run_mode_metrics.csv",
        per_run_mode_rows,
        ["run_id", "model_name", "sampling_mode", "sampling_kind", "shift_beta", "num_records"]
        + [item for k in metric_keys for item in (k, f"{k}_std")],
    )

    aggregate_rows = aggregate_rows_by_keys(
        per_run_mode_rows,
        ["model_name", "sampling_mode", "sampling_kind", "shift_beta"],
        metric_keys,
    )
    aggregate_rows.sort(key=lambda x: (MODEL_ORDER.index(x["model_name"]), float(x["shift_beta"])))
    write_csv(
        out_root / "aggregate_metrics.csv",
        aggregate_rows,
        ["model_name", "sampling_mode", "sampling_kind", "shift_beta", "num_records"]
        + [item for k in metric_keys for item in (k, f"{k}_std")],
    )

    strongest_mode = max(mode_defs.items(), key=lambda kv: float(kv[1]["beta"]))[0]
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
    plot_jobs = [
        (plot_density_shift_bars, (per_view_rows, out_root / "density_shift_validation.png", "Subset density-shift validation")),
        (plot_delta_bars, (run_delta_rows, "combined_physics_delta", out_root / "combined_physics_degradation_bars_all_models.png", f"Per-run degradation under strongest shift ({strongest_mode})")),
        (plot_delta_bars, (run_delta_rows, "combined_physics_ratio", out_root / "combined_physics_ratio_bars_all_models.png", f"Per-run robustness ratio under strongest shift ({strongest_mode})")),
    ]
    for family_key, family_models in FAMILY_GROUPS.items():
        family_models = [m for m in family_models if m in model_specs]
        if not family_models:
            continue
        family_title = FAMILY_TITLES[family_key]
        family_per_run_mode_rows = [r for r in per_run_mode_rows if r["model_name"] in family_models]
        family_aggregate_rows = [r for r in aggregate_rows if r["model_name"] in family_models]
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
                    plot_shift_curve_with_band,
                    (
                        family_aggregate_rows,
                        "combined_physics_rel_l2",
                        out_root / f"{family_key}_combined_physics_shift_curve.png",
                        f"{family_title}: sampling-shift severity curve (combined physics)",
                        family_models,
                    ),
                ),
                (
                    plot_shift_curve_with_band,
                    (
                        family_aggregate_rows,
                        "combined_global_rel_l2",
                        out_root / f"{family_key}_combined_global_shift_curve.png",
                        f"{family_title}: sampling-shift severity curve (combined global)",
                        family_models,
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
    rep_full_geo_log_density = estimate_log_sampling_density(
        rep_input_geo_norm.unsqueeze(0),
        knn_k=dataset.geometry_density_knn_k,
        neighbor_hops=dataset.geometry_density_neighbor_hops,
        estimator=dataset.geometry_density_estimator,
    ).squeeze(0).cpu()

    sampling_input_surf_coords = np.load(representative_run_dir / "surface_coords.npy").astype(np.float32, copy=False)
    sampling_input_geo_norm = normalize_pos(torch.from_numpy(sampling_input_surf_coords), min_pos, max_pos)
    sampling_full_geo_log_density = dataset._load_or_compute_full_geometry_density(vtk_run_id, expected_n=int(sampling_input_surf_coords.shape[0]))
    sampling_full_geo_log_density_np = sampling_full_geo_log_density.to(dtype=torch.float32).numpy()

    surface_point_data: Dict[str, np.ndarray] = {
        "gt_pressure": rep_surf_gt_full[:, 0],
    }
    representative_models = OrderedDict((m, models[m]) for m in VTK_PRESSURE_MODELS if m in models)
    audi_vtk_skipped_models: List[str] = []
    n_surface_points = int(rep_surf_gt_full.shape[0])
    for model_name, model in tqdm(representative_models.items(), desc="Representative full-surface predictions", dynamic_ncols=True):
        model = model.to(device)
        model.eval()
        prefix = MODEL_LABELS[model_name].lower()
        try:
            model_input_points = int(per_model_input_budgets[model_name])
            rep_rng = np.random.default_rng(np.random.SeedSequence([args.seed, int(vtk_run_id), 99991, MODEL_ORDER.index(model_name)]))
            rep_idx = sample_uniform_without_replacement(rep_input_surf_coords.shape[0], model_input_points, rep_rng)
            rep_geo_view_norm = rep_input_geo_norm[torch.from_numpy(rep_idx)].unsqueeze(0)
            rep_geo_density_view = rep_full_geo_log_density.index_select(0, torch.from_numpy(rep_idx).to(dtype=torch.long)).unsqueeze(0)
            pred_pressure = predict_audi_surface_pressure(
                model_name=model_name,
                model=model,
                geo_view_norm=rep_geo_view_norm,
                surf_query_norm=audi_surf_query_norm,
                dummy_vol_query_norm=rep_dummy_vol_query_norm,
                geo_log_density_view=rep_geo_density_view if model_uses_density(model_name) else None,
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
        model = model.to("cpu")
        models[model_name] = model

    vtk_path = out_root / "audi_surface_pressure_predictions.vtk"
    write_polydata_vtk(vtk_path, rep_surf_coords_full, surface_point_data)

    sampling_vtk_paths = []
    sampling_histogram_paths = []
    sampling_budget = max(unique_input_budgets)
    for beta in parse_shift_betas(args.shift_betas):
        sampling_rng = np.random.default_rng(np.random.SeedSequence([args.seed, int(vtk_run_id), 77777, int(round(beta * 100))]))
        sample_idx = sample_inverse_density_without_replacement(sampling_full_geo_log_density_np, sampling_budget, float(beta), sampling_rng)
        sampled_points = sampling_input_surf_coords[sample_idx]
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
            "views_per_mode": views_per_mode,
            "view_batch_size": view_batch_size,
            "model_repeats": int(args.model_repeats),
            "top_level_alignment_note": "Aligned mode matches each model's training-time view sampling rule best: uniform without replacement at that model's own encoder input budget.",
            "model_internal_note": "Each model keeps its own internal encoder-block subsampling exactly as implemented in that checkpointed architecture.",
            "representative_vtk_surface_query_source": str(vtk_surface_query_dir),
            "representative_vtk_encoder_input_source": "external surface_coords.npy from the selected Audi VTK query directory, sampled with each model's own aligned encoder input budget",
            "representative_vtk_dummy_volume_query_source": f"first point from DrivAerML run {vtk_run_id} fixed volume-query subset; used only if a model cannot execute an empty-volume surface-only export path",
            "representative_sampling_point_source_run_id": vtk_run_id,
            "representative_sampling_point_vtks": sampling_vtk_paths,
            "representative_sampling_point_histograms": sampling_histogram_paths,
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
        f"- Shifted modes use inverse-density sampling without replacement at betas `{shift_betas}` and keep the same point budget.",
        "- If `beta=0` is included in the shifted list, it acts as a uniform-without-replacement sanity-check mode and should match the aligned mode up to sampling randomness.",
        "- Internal model behavior is not overridden beyond safe batched-query chunking. In particular, each model keeps its own trained latent-anchor logic and encoder-block 16k subsampling behavior.",
        "- In-family fairness is strongest when all compared checkpoints in that family were trained with the same encoder input budget and the evaluation uses that same budget.",
        "- Cross-family fairness is weaker when families were trained with different encoder input budgets; the script records those mismatches explicitly in `results.json`.",
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
        f"- Separate point-cloud VTKs are exported from DrivAerML test run `{vtk_run_id}` for inverse-density sampling betas `0, 0.25, 0.5, 0.75, 1.0` using the largest active encoder budget `{sampling_budget}` so you can directly inspect one representative input cloud.",
        "- Each sampled-point VTK also gets a separate PNG histogram of the sampled density distribution, with a log-count y-axis.",
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
            f"- `drivaerml_test_run_{vtk_run_id}_input_points_{sampling_budget}_inverse_density_beta_*.vtk`: sampled `{sampling_budget}` input points for each inverse-density beta from one evaluated DrivAerML test run.",
            f"- `drivaerml_test_run_{vtk_run_id}_input_points_{sampling_budget}_inverse_density_beta_*_density_hist.png`: density-distribution histogram for each sampled input-point VTK.",
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
        f"- Fixed benchmark query subsets per run: `{surface_query_points}` surface + `{volume_query_points}` volume",
        f"- ABUPT-family query override: `{int(args.abupt_surface_query_points) if int(args.abupt_surface_query_points) > 0 else surface_query_points}` surface + `{int(args.abupt_volume_query_points) if int(args.abupt_volume_query_points) > 0 else volume_query_points}` volume",
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
