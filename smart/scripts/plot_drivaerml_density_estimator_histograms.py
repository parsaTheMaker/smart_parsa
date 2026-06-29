#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
SMART_ROOT = SCRIPT_DIR.parent
if str(SMART_ROOT) not in sys.path:
    sys.path.insert(0, str(SMART_ROOT))

try:
    from data.ahmedml_dataset_v2 import DrivAerMLDataset
except ImportError:  # pragma: no cover
    from smart.data.ahmedml_dataset_v2 import DrivAerMLDataset
try:
    from utils.geometry_density import estimate_log_sampling_density
except ImportError:  # pragma: no cover
    from smart.utils.geometry_density import estimate_log_sampling_density


DEFAULT_DATA_PATH = "/mnt/ssdraid/parsa/drivaerml_preprocessed"
DEFAULT_OUTPUT_DIR = "/home/parsa/smart_parsa/results/density_estimator_histograms"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot full, untrimmed DrivAerML density histograms for KDE and tangent-cov estimators.")
    p.add_argument("--data-path", default=DEFAULT_DATA_PATH)
    p.add_argument("--run-id", type=int, default=None, help="Run id to analyze. Default: smallest run with an existing tangent-cov cache.")
    p.add_argument("--knn-k", type=int, default=24)
    p.add_argument("--neighbor-hops", type=int, default=1)
    p.add_argument("--cache-dtype", choices=["float16", "float32"], default="float16")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    return p.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def find_default_run_id(data_path: Path, knn_k: int, neighbor_hops: int, cache_dtype: str) -> int:
    pattern = f"run_*/geometry_log_density_v2_h5_noscale_tangent_cov_k{knn_k}_h{neighbor_hops}_{cache_dtype}.npy"
    matches = sorted(data_path.glob(pattern), key=lambda p: int(p.parent.name.split("_")[-1]))
    if not matches:
        raise FileNotFoundError(f"No tangent-cov cache files found under {data_path} matching {pattern}")
    return int(matches[0].parent.name.split("_")[-1])


def build_dataset(
    data_path: str,
    estimator: str,
    knn_k: int,
    neighbor_hops: int,
    cache_dtype: str,
) -> DrivAerMLDataset:
    return DrivAerMLDataset(
        saved_folder=data_path,
        if_test=False,
        geometry_points=0,
        surface_points=0,
        volume_points=0,
        scale_positions=False,
        require_preprocessed=True,
        return_geometry_density=True,
        geometry_density_knn_k=knn_k,
        geometry_density_neighbor_hops=neighbor_hops,
        geometry_density_estimator=estimator,
        geometry_density_cache_dtype=cache_dtype,
    )


def compute_or_load_kde_density(
    dataset: DrivAerMLDataset,
    run_id: int,
    device: torch.device,
) -> np.ndarray:
    cache_path = dataset._geometry_density_cache_path(run_id)
    if cache_path.is_file():
        arr = np.load(cache_path)
        return np.asarray(arr)

    full_geo_mesh = dataset._load_full_geometry_mesh(run_id)
    compute_device = device if device.type == "cuda" and torch.cuda.is_available() else torch.device("cpu")
    geo = full_geo_mesh.unsqueeze(0).to(device=compute_device, dtype=torch.float32, non_blocking=(compute_device.type == "cuda"))
    log_density = estimate_log_sampling_density(
        geo,
        knn_k=dataset.geometry_density_knn_k,
        neighbor_hops=dataset.geometry_density_neighbor_hops,
        estimator=dataset.geometry_density_estimator,
    ).squeeze(0).detach().cpu().numpy()

    if dataset.geometry_density_cache_dtype == "float16":
        cache_arr = log_density.astype(np.float16, copy=False)
    else:
        cache_arr = log_density.astype(np.float32, copy=False)
    dataset._atomic_save_npy(cache_path, cache_arr)
    return cache_arr


def positive_density_from_log(log_density: np.ndarray) -> np.ndarray:
    log_vals = np.asarray(log_density, dtype=np.float64)
    density = np.exp(log_vals)
    mask = np.isfinite(density) & (density > 0.0)
    out = density[mask]
    if out.size == 0:
        raise ValueError("No positive finite densities available for plotting.")
    return out


def save_full_density_histogram(path: Path, density: np.ndarray, title: str, color: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    density = np.asarray(density, dtype=np.float64)
    lo = float(np.min(density))
    hi = float(np.max(density))
    if not np.isfinite(lo) or not np.isfinite(hi):
        raise ValueError("Density histogram received non-finite min/max.")
    if hi <= lo:
        hi = np.nextafter(lo, float("inf"))
    bins = np.geomspace(lo, hi, num=120)

    fig, ax = plt.subplots(figsize=(8.4, 5.4), constrained_layout=True)
    ax.hist(density, bins=bins, color=color, alpha=0.9, edgecolor="white", linewidth=0.25)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Density")
    ax.set_ylabel("Count")
    ax.set_title(title)
    stats = (
        f"min={np.min(density):.4g}\n"
        f"p50={np.percentile(density, 50):.4g}\n"
        f"mean={np.mean(density):.4g}\n"
        f"p95={np.percentile(density, 95):.4g}\n"
        f"max={np.max(density):.4g}\n"
        f"N={density.size}"
    )
    ax.text(
        0.98,
        0.98,
        stats,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "#CCCCCC"},
    )
    fig.savefig(path, dpi=220)
    plt.close(fig)


def save_overlay_histogram(path: Path, tangent_density: np.ndarray, kde_density: np.ndarray, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lo = float(min(np.min(tangent_density), np.min(kde_density)))
    hi = float(max(np.max(tangent_density), np.max(kde_density)))
    if hi <= lo:
        hi = np.nextafter(lo, float("inf"))
    bins = np.geomspace(lo, hi, num=140)

    fig, ax = plt.subplots(figsize=(8.8, 5.6), constrained_layout=True)
    ax.hist(tangent_density, bins=bins, histtype="step", linewidth=1.8, color="#E45756", label="Tangent cov")
    ax.hist(kde_density, bins=bins, histtype="step", linewidth=1.8, color="#4C78A8", label="KDE")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Density")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.legend()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    data_path = Path(args.data_path).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    run_id = int(args.run_id) if args.run_id is not None else find_default_run_id(data_path, args.knn_k, args.neighbor_hops, args.cache_dtype)
    print(f"Using run_id={run_id}")
    print(f"Device preference: {device}")

    tangent_ds = build_dataset(
        str(data_path),
        estimator="tangent_cov",
        knn_k=args.knn_k,
        neighbor_hops=args.neighbor_hops,
        cache_dtype=args.cache_dtype,
    )
    kde_ds = build_dataset(
        str(data_path),
        estimator="kde",
        knn_k=args.knn_k,
        neighbor_hops=args.neighbor_hops,
        cache_dtype=args.cache_dtype,
    )

    tangent_log_density = tangent_ds._load_or_compute_full_geometry_density(run_id, expected_n=None).cpu().numpy()
    kde_log_density = compute_or_load_kde_density(kde_ds, run_id, device=device)

    tangent_density = positive_density_from_log(tangent_log_density)
    kde_density = positive_density_from_log(kde_log_density)

    tangent_path = output_dir / f"run_{run_id}_tangent_cov_k{args.knn_k}_density_hist_full.png"
    kde_path = output_dir / f"run_{run_id}_kde_k{args.knn_k}_density_hist_full.png"
    overlay_path = output_dir / f"run_{run_id}_tangent_cov_vs_kde_k{args.knn_k}_density_hist_full_overlay.png"

    save_full_density_histogram(
        tangent_path,
        tangent_density,
        title=f"Run {run_id} tangent-cov density histogram (full, untrimmed)",
        color="#E45756",
    )
    save_full_density_histogram(
        kde_path,
        kde_density,
        title=f"Run {run_id} KDE density histogram (full, untrimmed)",
        color="#4C78A8",
    )
    save_overlay_histogram(
        overlay_path,
        tangent_density=tangent_density,
        kde_density=kde_density,
        title=f"Run {run_id} tangent-cov vs KDE density histogram (full, untrimmed)",
    )

    print(f"Tangent histogram: {tangent_path}")
    print(f"KDE histogram: {kde_path}")
    print(f"Overlay histogram: {overlay_path}")


if __name__ == "__main__":
    main()
