#!/usr/bin/env python3
"""Compare SMART-family models under controlled encoder-input sampling shift.

Workflow:
1) Fix the query points to the full preprocessed surface/volume coordinates.
2) Change only the encoder input geometry points.
3) Use an aligned mode that matches training best: 131072 geometry points
   sampled uniformly without replacement from the full surface cloud.
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
from models.smart.smart import SMART
from models.smart.smart_sat2 import SMARTSAT2
from models.smart.smart_sat3 import SMARTSAT3
from models.smart.smart_sat4 import SMARTSAT4
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

MODEL_ORDER = ["SMART", "SMART_SATLOSS2", "SMART_SAT2", "SMART_SAT3", "SMART_SAT4"]
MODEL_LABELS = {
    "SMART": "SMART",
    "SMART_SATLOSS2": "SATLOSS2",
    "SMART_SAT2": "SAT2",
    "SMART_SAT3": "SAT3",
    "SMART_SAT4": "SAT4",
}
MODEL_COLORS = {
    "SMART": "#6C6F7D",
    "SMART_SATLOSS2": "#F58518",
    "SMART_SAT2": "#54A24B",
    "SMART_SAT3": "#E45756",
    "SMART_SAT4": "#4C78A8",
}
MODE_COLORS = {
    "aligned_uniform_wor": "#4C78A8",
    "shifted_inverse_density_beta_0.50": "#F58518",
    "shifted_inverse_density_beta_1.00": "#E45756",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fair DrivAerML sampling-invariance comparison for SMART / SATLOSS2 / SAT2 / SAT3 / SAT4.")
    p.add_argument("--smart-config", default="drivaerml")
    p.add_argument("--satloss2-config", default="drivaerml_satloss2")
    p.add_argument("--sat2-config", default="drivaerml_sat2")
    p.add_argument("--sat3-config", default="drivaerml_sat3")
    p.add_argument("--sat4-config", default="drivaerml_sat4")
    p.add_argument("--smart-checkpoint", default=None)
    p.add_argument("--satloss2-checkpoint", default=None)
    p.add_argument("--sat2-checkpoint", default=None)
    p.add_argument("--sat3-checkpoint", default=None)
    p.add_argument("--sat4-checkpoint", default=None)
    p.add_argument("--num-runs", type=int, default=8, help="Number of test runs to evaluate.")
    p.add_argument("--run-ids", default=None, help="Optional comma-separated explicit run ids.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None)
    p.add_argument("--input-points", type=int, default=None, help="Encoder input size. Default: 131072.")
    p.add_argument(
        "--shift-betas",
        default="0.5,1.0",
        help="Comma-separated inverse-density shift severities. Example: 0.5,1.0",
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
        "SMART_SATLOSS2": "smart-satloss2-",
        "SMART_SAT2": "smart-sat2-",
        "SMART_SAT3": "smart-sat3-",
        "SMART_SAT4": "smart-sat4-",
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
    if model_name in {"SMART", "SMART_SATLOSS2"}:
        model = SMART(**base_kwargs, **arch).to(device)
    elif model_name == "SMART_SAT2":
        model = SMARTSAT2(**base_kwargs, **arch).to(device)
    elif model_name == "SMART_SAT3":
        model = SMARTSAT3(**base_kwargs, **arch).to(device)
    elif model_name == "SMART_SAT4":
        model = SMARTSAT4(**base_kwargs, **arch).to(device)
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
    return model_name in {"SMART_SAT2", "SMART_SAT3", "SMART_SAT4"}


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


def _grouped_bar_on_axis(ax, rows: List[Dict[str, object]], metric_key: str, mode_order: Sequence[str]):
    means = defaultdict(dict)
    stds = defaultdict(dict)
    for model_name in MODEL_ORDER:
        for mode_name in mode_order:
            vals = [float(r[metric_key]) for r in mode_rows(rows, model_name, mode_name)]
            means[model_name][mode_name] = float(np.mean(vals)) if vals else math.nan
            stds[model_name][mode_name] = float(np.std(vals)) if vals else math.nan

    x = np.arange(len(MODEL_ORDER), dtype=np.float64)
    width = 0.8 / max(len(mode_order), 1)
    for i, mode_name in enumerate(mode_order):
        vals = [means[m][mode_name] for m in MODEL_ORDER]
        err = [stds[m][mode_name] for m in MODEL_ORDER]
        offset = (i - 0.5 * (len(mode_order) - 1)) * width
        ax.bar(
            x + offset,
            vals,
            width=width,
            yerr=err,
            capsize=4,
            color=MODE_COLORS.get(mode_name, "#999999"),
            label=mode_name,
            alpha=0.88,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS[m] for m in MODEL_ORDER], rotation=15)
    ax.set_ylabel(metric_key)
    ax.set_title(_metric_display_name(metric_key))


def plot_metric_grid(
    rows: List[Dict[str, object]],
    metric_keys: Sequence[str],
    mode_order: Sequence[str],
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
        _grouped_bar_on_axis(ax, rows, metric_key, mode_order)
    handles, labels = axes_arr.flat[0].get_legend_handles_labels()
    for ax in axes_arr.flat[n_metrics:]:
        ax.axis("off")
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 4))
    fig.suptitle(title, fontsize=18)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_shift_curve_with_band(aggregate_rows: List[Dict[str, object]], metric_key: str, out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 5.8), constrained_layout=True)
    for model_name in MODEL_ORDER:
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
    for model_name in MODEL_ORDER:
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
    ax.bar(np.arange(len(mode_order)), means, yerr=stds, capsize=4, color=[MODE_COLORS.get(m, "#999999") for m in mode_order], alpha=0.88)
    ax.set_xticks(np.arange(len(mode_order)))
    ax.set_xticklabels(mode_order, rotation=15)
    ax.set_ylabel("subset_log_density_mean")
    ax.set_title(title)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_comprehensive_dashboard(
    per_run_mode_rows: List[Dict[str, object]],
    aggregate_rows: List[Dict[str, object]],
    run_delta_rows: List[Dict[str, object]],
    mode_order: Sequence[str],
    strongest_mode: str,
    out_path: Path,
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
        _grouped_bar_on_axis(ax, per_run_mode_rows, metric_key, mode_order)
    fig.suptitle(
        f"SMART vs SATLOSS2 / SAT2 / SAT3 / SAT4\n"
        f"Aligned mode and strongest shifted mode ({strongest_mode}) are included in every panel",
        fontsize=18,
    )
    handles, labels = axes.flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 4))
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main():
    args = parse_args()
    device = resolve_device(args.device)
    print(f"Device: {device}")

    smart_cfg = load_cfg(args.smart_config)
    satloss2_cfg = load_cfg(args.satloss2_config)
    sat2_cfg = load_cfg(args.sat2_config)
    sat3_cfg = load_cfg(args.sat3_config)
    sat4_cfg = load_cfg(args.sat4_config)

    data_paths = {
        str(smart_cfg.data_path),
        str(satloss2_cfg.data_path),
        str(sat2_cfg.data_path),
        str(sat3_cfg.data_path),
        str(sat4_cfg.data_path),
    }
    if len(data_paths) != 1:
        raise ValueError(f"Expected one shared DrivAerML data path, got: {sorted(data_paths)}")

    input_points = int(args.input_points) if args.input_points is not None else 131072
    if input_points <= 0:
        raise ValueError("This evaluator expects a positive encoder input size.")

    shift_betas = parse_shift_betas(args.shift_betas)
    mode_defs = OrderedDict()
    mode_defs["aligned_uniform_wor"] = {
        "kind": "uniform_wor",
        "beta": 0.0,
        "description": "Uniform without replacement, aligned with the 131072-point training view sampling.",
        "id": 0,
    }
    for i, beta in enumerate(shift_betas, start=1):
        mode_defs[f"shifted_inverse_density_beta_{beta:.2f}"] = {
            "kind": "inverse_density_wor",
            "beta": float(beta),
            "description": f"Inverse-density without replacement, same point budget with beta={beta:.2f}.",
            "id": i,
        }

    density_knn_k = int(getattr(sat4_cfg.architecture, "density_knn_k", 24))
    density_neighbor_hops = int(getattr(sat4_cfg.architecture, "density_neighbor_hops", 1))
    density_estimator = str(getattr(sat4_cfg.architecture, "density_estimator", "tangent_cov"))
    density_cache_dtype = str(getattr(sat4_cfg, "geometry_density_cache_dtype", "float16"))

    dataset = AhmedMLDatasetV2(
        saved_folder=str(smart_cfg.data_path),
        if_test=True,
        geometry_points=input_points,
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

    model_specs = OrderedDict(
        [
            ("SMART", {"config": smart_cfg, "checkpoint": choose_ckpt(smart_cfg, args.smart_checkpoint)}),
            ("SMART_SATLOSS2", {"config": satloss2_cfg, "checkpoint": choose_ckpt(satloss2_cfg, args.satloss2_checkpoint)}),
            ("SMART_SAT2", {"config": sat2_cfg, "checkpoint": choose_ckpt(sat2_cfg, args.sat2_checkpoint)}),
            ("SMART_SAT3", {"config": sat3_cfg, "checkpoint": choose_ckpt(sat3_cfg, args.sat3_checkpoint)}),
            ("SMART_SAT4", {"config": sat4_cfg, "checkpoint": choose_ckpt(sat4_cfg, args.sat4_checkpoint)}),
        ]
    )
    for model_name, spec in model_specs.items():
        print(f"{model_name} checkpoint: {spec['checkpoint']}")

    models = {
        model_name: build_model(spec["config"], spec["checkpoint"], device, batched_query_subregion_size=args.batched_query_subregion_size)
        for model_name, spec in model_specs.items()
    }

    out_root = Path(
        args.output_dir
        or (SMART_ROOT.parent / "results" / "drivaerml_sampling_invariance_5models" / f"seed_{args.seed}_runs_{len(run_ids)}_views_{args.views_per_mode}")
    )
    out_root.mkdir(parents=True, exist_ok=True)

    per_view_rows: List[Dict[str, object]] = []
    view_batch_size = max(1, int(args.view_batch_size))
    views_per_mode = max(1, int(args.views_per_mode))

    for run_id in tqdm(run_ids, desc="Runs", dynamic_ncols=True):
        run_dir = Path(smart_cfg.data_path) / f"run_{run_id}"
        surf_coords = np.load(run_dir / "surface_coords.npy").astype(np.float32, copy=False)
        surf_p = np.load(run_dir / "surface_pMeanTrim.npy").astype(np.float32, copy=False).reshape(-1, 1)
        surf_n = np.load(run_dir / "surface_normals.npy").astype(np.float32, copy=False)
        surf_wx = np.load(run_dir / "surface_wallShearStressMeanTrim_x.npy").astype(np.float32, copy=False).reshape(-1, 1)
        surf_wy = np.load(run_dir / "surface_wallShearStressMeanTrim_y.npy").astype(np.float32, copy=False).reshape(-1, 1)
        surf_wz = np.load(run_dir / "surface_wallShearStressMeanTrim_z.npy").astype(np.float32, copy=False).reshape(-1, 1)
        surf_gt = np.concatenate([surf_p, surf_n, surf_wx, surf_wy, surf_wz], axis=1)

        vol_coords = np.load(run_dir / "volume_coords.npy").astype(np.float32, copy=False)
        vol_p = np.load(run_dir / "volume_pMeanTrim.npy").astype(np.float32, copy=False).reshape(-1, 1)
        vol_u = np.load(run_dir / "volume_UMeanTrim.npy").astype(np.float32, copy=False)
        vol_gt = np.concatenate([vol_p, vol_u], axis=1)

        surf_query_norm = normalize_pos(torch.from_numpy(surf_coords), min_pos, max_pos)
        vol_query_norm = normalize_pos(torch.from_numpy(vol_coords), min_pos, max_pos)

        full_geo_log_density = dataset._load_or_compute_full_geometry_density(run_id, expected_n=int(surf_coords.shape[0]))
        full_geo_log_density_np = full_geo_log_density.to(dtype=torch.float32).numpy()

        for mode_name, mode_info in mode_defs.items():
            idx_list: List[np.ndarray] = []
            subset_density_stats: List[Dict[str, float]] = []
            for view_idx in range(views_per_mode):
                rng = np.random.default_rng(np.random.SeedSequence([args.seed, int(run_id), int(mode_info["id"]), int(view_idx)]))
                if mode_info["kind"] == "uniform_wor":
                    idx = sample_uniform_without_replacement(surf_coords.shape[0], input_points, rng)
                else:
                    idx = sample_inverse_density_without_replacement(full_geo_log_density_np, input_points, float(mode_info["beta"]), rng)
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

            for model_name, model in models.items():
                for batch_start in range(0, views_per_mode, view_batch_size):
                    batch_stop = min(batch_start + view_batch_size, views_per_mode)
                    batch_indices = idx_list[batch_start:batch_stop]
                    geo_view_tensors = [surf_query_norm[torch.from_numpy(idx)] for idx in batch_indices]
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
                                "full_log_density_mean": float(np.mean(full_geo_log_density_np)),
                                **density_stats,
                                **metrics,
                            }
                        )

                    del geo_views_norm, geo_density_views, pred_surf_batch, pred_vol_batch
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
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
    for model_name in MODEL_ORDER:
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

    robustness_rows.sort(key=lambda x: MODEL_ORDER.index(x["model_name"]))
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
        (plot_metric_grid, (per_run_mode_rows, HEADLINE_METRIC_KEYS, mode_order, out_root / "headline_metrics_by_mode.png", "Headline metrics by model and encoder-input mode", 3)),
        (plot_metric_grid, (per_run_mode_rows, SURFACE_FIELD_METRIC_KEYS, mode_order, out_root / "surface_fields_by_mode.png", "Surface field rel-L2 by model and encoder-input mode", 2)),
        (plot_metric_grid, (per_run_mode_rows, VOLUME_FIELD_METRIC_KEYS, mode_order, out_root / "volume_fields_by_mode.png", "Volume field rel-L2 by model and encoder-input mode", 2)),
        (plot_shift_curve_with_band, (aggregate_rows, "combined_physics_rel_l2", out_root / "combined_physics_shift_curve.png", "Sampling-shift severity curve (combined physics)")),
        (plot_shift_curve_with_band, (aggregate_rows, "combined_global_rel_l2", out_root / "combined_global_shift_curve.png", "Sampling-shift severity curve (combined global)")),
        (plot_delta_bars, (run_delta_rows, "combined_physics_delta", out_root / "combined_physics_degradation_bars.png", f"Per-run degradation under strongest shift ({strongest_mode})")),
        (plot_delta_bars, (run_delta_rows, "combined_physics_ratio", out_root / "combined_physics_ratio_bars.png", f"Per-run robustness ratio under strongest shift ({strongest_mode})")),
        (plot_density_shift_bars, (per_view_rows, out_root / "density_shift_validation.png", "Subset density-shift validation")),
        (plot_comprehensive_dashboard, (per_run_mode_rows, aggregate_rows, run_delta_rows, mode_order, strongest_mode, out_root / "comprehensive_dashboard.png")),
    ]
    with ProcessPoolExecutor(max_workers=max(1, int(args.plot_workers))) as pool:
        futures = [pool.submit(func, *func_args) for func, func_args in plot_jobs]
        for future in tqdm(futures, desc="CPU plot tasks", leave=False, dynamic_ncols=True):
            future.result()

    vtk_surface_query_dir = Path(args.vtk_surface_query_dir).expanduser().resolve()
    representative_run_dir = Path(smart_cfg.data_path) / f"run_{vtk_run_id}"
    if not representative_run_dir.is_dir():
        raise FileNotFoundError(f"Representative VTK run not found: {representative_run_dir}")

    rep_surf_coords, rep_surf_gt = load_surface_query_from_dir(vtk_surface_query_dir)
    rep_input_surf_coords = rep_surf_coords
    rep_vol_coords = np.load(representative_run_dir / "volume_coords.npy").astype(np.float32, copy=False)
    rep_surf_query_norm = normalize_pos(torch.from_numpy(rep_surf_coords), min_pos, max_pos)
    rep_vol_query_norm = normalize_pos(torch.from_numpy(rep_vol_coords), min_pos, max_pos)
    rep_input_geo_norm = normalize_pos(torch.from_numpy(rep_input_surf_coords), min_pos, max_pos)
    rep_full_geo_log_density = estimate_log_sampling_density(
        rep_input_geo_norm.unsqueeze(0),
        knn_k=dataset.geometry_density_knn_k,
        neighbor_hops=dataset.geometry_density_neighbor_hops,
        estimator=dataset.geometry_density_estimator,
    ).squeeze(0).cpu()
    rep_rng = np.random.default_rng(np.random.SeedSequence([args.seed, int(vtk_run_id), 99991]))
    rep_idx = sample_uniform_without_replacement(rep_input_surf_coords.shape[0], input_points, rep_rng)
    rep_geo_view_norm = rep_input_geo_norm[torch.from_numpy(rep_idx)].unsqueeze(0)
    rep_geo_density_view = rep_full_geo_log_density.index_select(0, torch.from_numpy(rep_idx).to(dtype=torch.long)).unsqueeze(0)

    surface_point_data: Dict[str, np.ndarray] = {
        "gt_pressure": rep_surf_gt[:, 0],
        "gt_normal": rep_surf_gt[:, 1:4],
        "gt_wss": rep_surf_gt[:, 4:7],
        "gt_wss_mag": vector_mag(rep_surf_gt, 4, 7),
    }
    for model_name, model in tqdm(models.items(), desc="Representative full-surface predictions", dynamic_ncols=True):
        pred_surf_batch, _ = predict_view_batch(
            model_name=model_name,
            model=model,
            geo_views_norm=rep_geo_view_norm,
            surf_query_norm=rep_surf_query_norm,
            vol_query_norm=rep_vol_query_norm,
            geo_log_density_views=rep_geo_density_view if model_uses_density(model_name) else None,
            mean_s=mean_s,
            std_s=std_s,
            mean_v=mean_v,
            std_v=std_v,
            device=device,
            base_seed=int(args.seed + 900000 + MODEL_ORDER.index(model_name) * 37),
            repeats=args.model_repeats,
        )
        pred_surf = pred_surf_batch[0]
        prefix = MODEL_LABELS[model_name].lower()
        surface_point_data[f"{prefix}_pressure_pred"] = pred_surf[:, 0]
        surface_point_data[f"{prefix}_pressure_abs_err"] = np.abs(pred_surf[:, 0] - rep_surf_gt[:, 0])
        surface_point_data[f"{prefix}_normal_pred"] = pred_surf[:, 1:4]
        surface_point_data[f"{prefix}_normal_abs_err"] = np.abs(pred_surf[:, 1:4] - rep_surf_gt[:, 1:4])
        surface_point_data[f"{prefix}_wss_pred"] = pred_surf[:, 4:7]
        surface_point_data[f"{prefix}_wss_abs_err"] = np.abs(pred_surf[:, 4:7] - rep_surf_gt[:, 4:7])
        surface_point_data[f"{prefix}_wss_mag_pred"] = vector_mag(pred_surf, 4, 7)
        surface_point_data[f"{prefix}_wss_mag_abs_err"] = np.abs(vector_mag(pred_surf, 4, 7) - vector_mag(rep_surf_gt, 4, 7))
        if device.type == "cuda":
            torch.cuda.empty_cache()

    vtk_path = out_root / f"representative_surface_predictions_run_{vtk_run_id}.vtk"
    write_polydata_vtk(vtk_path, rep_surf_coords, surface_point_data)

    payload = {
        "args": vars(args),
        "run_ids": run_ids,
        "representative_vtk_run_id": vtk_run_id,
        "models": {k: v["checkpoint"] for k, v in model_specs.items()},
        "mode_definitions": mode_defs,
        "workflow": {
            "queries_fixed_to_full_surface_and_volume": True,
            "encoder_input_points": input_points,
            "aligned_mode": "uniform_wor",
            "shift_modes": [name for name in mode_defs if name != "aligned_uniform_wor"],
            "views_per_mode": views_per_mode,
            "view_batch_size": view_batch_size,
            "model_repeats": int(args.model_repeats),
            "top_level_alignment_note": "Aligned mode matches the training-time 131072-point view sampling rule best: uniform without replacement.",
            "model_internal_note": "Each model keeps its own internal encoder-block subsampling exactly as implemented in that checkpointed architecture.",
            "representative_vtk_surface_query_source": str(vtk_surface_query_dir),
            "representative_vtk_encoder_input_source": "external surface_coords.npy from the selected VTK query directory, sampled with the same aligned 131072-point rule",
        },
        "configs": {
            "smart": args.smart_config,
            "satloss2": args.satloss2_config,
            "sat2": args.sat2_config,
            "sat3": args.sat3_config,
            "sat4": args.sat4_config,
        },
        "aggregate_metrics": aggregate_rows,
        "robustness_summary": robustness_rows,
    }
    (out_root / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    config_name_map = {
        "SMART": args.smart_config,
        "SMART_SATLOSS2": args.satloss2_config,
        "SMART_SAT2": args.sat2_config,
        "SMART_SAT3": args.sat3_config,
        "SMART_SAT4": args.sat4_config,
    }
    workflow_lines = [
        "# Sampling-Invariance Evaluation Workflow",
        "",
        "## Goal",
        "Measure how much prediction quality changes when only the encoder-input geometry sampling distribution changes, while keeping query points fixed.",
        "",
        "## Fairness Rules Used",
        f"- Evaluated models: `{', '.join(MODEL_ORDER)}`",
        "- Surface and volume query coordinates are fixed to the full preprocessed coordinates for every model, run, and mode.",
        f"- Encoder input point budget is fixed to `{input_points}` for every model and every sampling mode.",
        "- The aligned mode uses `uniform_wor` because SMART's top-level 131072-point training view is sampled uniformly without replacement by the dataset, and the consistency-trained models also use a uniform primary training view.",
        f"- Shifted modes use inverse-density sampling without replacement at betas `{shift_betas}` and keep the same point budget.",
        "- Internal model behavior is not overridden beyond safe batched-query chunking. In particular, each model keeps its own trained latent-anchor logic and encoder-block 16k subsampling behavior.",
        f"- Views per run/mode: `{views_per_mode}`",
        f"- Repeated stochastic forwards per view batch: `{int(args.model_repeats)}`",
        "",
        "## Representative VTK Export",
        "- The representative VTK keeps the encoder-input sampling rule unchanged: it still draws the 131072-point geometry view with uniform-without-replacement sampling.",
        "- The representative surface query cloud and the representative encoder-input cloud both come from the external surface folder you selected, so the exported predictions and points share the same coordinate frame.",
        "- Volume queries for that representative prediction remain the full preprocessed volume coordinates.",
        f"- Surface-query directory for that export: `{vtk_surface_query_dir}`",
        "",
        "## Aggregation",
        "- First aggregate multiple independently sampled views within each `(run, model, mode)` tuple.",
        "- Then aggregate across runs to obtain mean and standard deviation for plots.",
        "- Robustness is summarized both by absolute shifted performance and by `shifted - aligned` / `shifted / aligned` degradation statistics.",
        "",
        "## Configs and Checkpoints",
    ]
    for model_name, spec in model_specs.items():
        workflow_lines.append(f"- `{model_name}`: config=`{config_name_map[model_name]}` checkpoint=`{spec['checkpoint']}`")
    workflow_lines.extend(
        [
            "",
            "## Outputs",
            "- `per_view_metrics.csv`: metrics for every run/view/model/mode.",
            "- `per_run_mode_metrics.csv`: per-run averages across views with standard deviations.",
            "- `aggregate_metrics.csv`: across-run means/stds for every model and sampling mode.",
            "- `robustness_summary.csv`: strongest-shift robustness summary.",
            f"- `representative_surface_predictions_run_{vtk_run_id}.vtk`: full-surface ground truth plus all model predictions/errors.",
        ]
    )
    (out_root / "workflow.md").write_text("\n".join(workflow_lines), encoding="utf-8")

    summary_lines = [
        "# DrivAerML Sampling-Invariance Comparison",
        "",
        f"- Evaluated test runs: `{run_ids}`",
        f"- Representative VTK run: `{vtk_run_id}`",
        f"- Input points per encoder call: `{input_points}`",
        f"- Views per run/mode: `{views_per_mode}`",
        f"- View batch size: `{view_batch_size}`",
        f"- Strongest shift mode: `{strongest_mode}`",
        f"- Shift betas: `{shift_betas}`",
        "- Full queries fixed for every model and every mode: `True`",
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
            "- The field-level bar charts show whether robustness is consistent across surface and volume prediction categories or concentrated in only a subset of fields.",
        ]
    )
    (out_root / "summary.md").write_text("\n".join(summary_lines), encoding="utf-8")

    print(f"Wrote results to {out_root}")
    print(f"Representative VTK: {vtk_path}")


if __name__ == "__main__":
    main()
