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
    from utils.geometry_density import (
        NearestNeighbors,
        _tiny_f64,
        estimate_log_sampling_density,
        knn_edges_as_neighbor_center,
        torch_cluster_knn_graph,
    )
except ImportError:  # pragma: no cover
    from smart.utils.geometry_density import (
        NearestNeighbors,
        _tiny_f64,
        estimate_log_sampling_density,
        knn_edges_as_neighbor_center,
        torch_cluster_knn_graph,
    )


DEFAULT_DATA_PATH = "/mnt/ssdraid/parsa/drivaerml_preprocessed"
DEFAULT_OUTPUT_DIR = "/home/parsa/smart_parsa/results/kde_vs_vanilla_knn_histograms"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Compare full, untrimmed DrivAerML density histograms for KDE-k and "
            "a vanilla kNN baseline rho_i = mean(d_k) / d_k(i)."
        )
    )
    p.add_argument("--data-path", default=DEFAULT_DATA_PATH)
    p.add_argument("--run-id", type=int, default=100)
    p.add_argument("--knn-k", type=int, default=16)
    p.add_argument("--cache-dtype", choices=["float16", "float32"], default="float16")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    return p.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def build_dataset(data_path: str, estimator: str, knn_k: int, cache_dtype: str) -> DrivAerMLDataset:
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
        geometry_density_neighbor_hops=1,
        geometry_density_estimator=estimator,
        geometry_density_cache_dtype=cache_dtype,
    )


@torch.no_grad()
def estimate_log_sampling_density_vanilla_knn(points: torch.Tensor, knn_k: int = 16) -> torch.Tensor:
    """Return log(mean(d_k) / d_k(i)) using the same fp64/tiny discipline as KDE."""
    pts = points.clamp(0.0, 1.0 - 1e-6)
    bsz, _, _ = pts.shape
    k_eff = max(1, int(knn_k))
    tiny = _tiny_f64()

    if torch_cluster_knn_graph is not None:
        outputs = []
        for b in range(bsz):
            pts_b = pts[b]
            n = int(pts_b.shape[0])
            if n <= 1:
                outputs.append(torch.zeros((n,), device=pts.device, dtype=pts.dtype))
                continue

            k_cur = min(k_eff, n - 1)
            nbr, center = knn_edges_as_neighbor_center(pts_b, k_cur)
            diffs64 = pts_b[nbr].to(dtype=torch.float64) - pts_b[center].to(dtype=torch.float64)
            d2 = diffs64.square().sum(dim=-1)
            d = torch.sqrt(torch.clamp(d2, min=tiny))

            kth_distance = torch.zeros((n,), device=pts.device, dtype=torch.float64)
            kth_distance.scatter_reduce_(0, center, d, reduce="amax", include_self=False)
            kth_distance = torch.clamp(kth_distance, min=tiny)

            mean_kth_distance = torch.clamp(kth_distance.mean(), min=tiny)
            log_density = torch.log(mean_kth_distance) - torch.log(kth_distance)
            outputs.append(log_density.to(dtype=pts.dtype))
        return torch.stack(outputs, dim=0)

    if NearestNeighbors is not None:
        outputs = []
        for b in range(bsz):
            pts_b = pts[b]
            n = int(pts_b.shape[0])
            if n <= 1:
                outputs.append(torch.zeros((n,), device=pts.device, dtype=pts.dtype))
                continue

            k_cur = min(k_eff, n - 1)
            pts_np = pts_b.detach().cpu().numpy().astype(np.float64, copy=False)
            nbrs = NearestNeighbors(n_neighbors=k_cur + 1, algorithm="kd_tree", n_jobs=-1)
            distances, _ = nbrs.fit(pts_np).kneighbors(return_distance=True)
            kth_distance = np.clip(distances[:, -1].astype(np.float64, copy=False), np.finfo(np.float64).tiny, None)
            mean_kth_distance = max(float(kth_distance.mean()), np.finfo(np.float64).tiny)
            log_density = np.log(mean_kth_distance) - np.log(kth_distance)
            outputs.append(torch.from_numpy(log_density).to(device=pts.device, dtype=pts.dtype))
        return torch.stack(outputs, dim=0)

    raise RuntimeError(
        "Vanilla-kNN density estimation requires either torch_cluster.knn_graph "
        "or sklearn.neighbors.NearestNeighbors to be available."
    )


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
    bins = np.geomspace(1.0e-3, 1.0, num=121, dtype=np.float64)

    fig, ax = plt.subplots(figsize=(8.4, 5.4), constrained_layout=True)
    ax.hist(density, bins=bins, color=color, alpha=0.9, edgecolor="white", linewidth=0.25)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Density")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.set_xlim(1.0e-3, 1.0)
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


def save_overlay_histogram(path: Path, kde_density: np.ndarray, vanilla_density: np.ndarray, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    bins = np.geomspace(1.0e-3, 1.0, num=141, dtype=np.float64)

    fig, ax = plt.subplots(figsize=(8.8, 5.6), constrained_layout=True)
    ax.hist(vanilla_density, bins=bins, histtype="step", linewidth=1.8, color="#E45756", label="Vanilla kNN")
    ax.hist(kde_density, bins=bins, histtype="step", linewidth=1.8, color="#4C78A8", label="KDE")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Density")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.set_xlim(1.0e-3, 1.0)
    ax.legend()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    data_path = Path(args.data_path).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    run_id = int(args.run_id)
    print(f"Using run_id={run_id}")
    print(f"Device preference: {device}")

    kde_ds = build_dataset(
        str(data_path),
        estimator="kde",
        knn_k=args.knn_k,
        cache_dtype=args.cache_dtype,
    )

    full_geo_mesh = kde_ds._load_full_geometry_mesh(run_id)
    compute_device = device if device.type == "cuda" and torch.cuda.is_available() else torch.device("cpu")
    geo = full_geo_mesh.unsqueeze(0).to(device=compute_device, dtype=torch.float32, non_blocking=(compute_device.type == "cuda"))

    kde_log_density = estimate_log_sampling_density(
        geo,
        knn_k=args.knn_k,
        neighbor_hops=1,
        estimator="kde",
    ).squeeze(0).detach().cpu().numpy()
    vanilla_log_density = estimate_log_sampling_density_vanilla_knn(
        geo,
        knn_k=args.knn_k,
    ).squeeze(0).detach().cpu().numpy()

    kde_density = positive_density_from_log(kde_log_density)
    vanilla_density = positive_density_from_log(vanilla_log_density)

    kde_path = output_dir / f"run_{run_id}_kde_k{args.knn_k}_density_hist_full.png"
    vanilla_path = output_dir / f"run_{run_id}_vanilla_knn_k{args.knn_k}_density_hist_full.png"
    overlay_path = output_dir / f"run_{run_id}_kde_vs_vanilla_knn_k{args.knn_k}_density_hist_full_overlay.png"

    save_full_density_histogram(
        kde_path,
        kde_density,
        title=f"Run {run_id} KDE density histogram (k={args.knn_k}, full, untrimmed)",
        color="#4C78A8",
    )
    save_full_density_histogram(
        vanilla_path,
        vanilla_density,
        title=f"Run {run_id} vanilla kNN density histogram (k={args.knn_k}, full, untrimmed)",
        color="#E45756",
    )
    save_overlay_histogram(
        overlay_path,
        kde_density=kde_density,
        vanilla_density=vanilla_density,
        title=f"Run {run_id} KDE vs vanilla kNN density histogram (k={args.knn_k}, full, untrimmed)",
    )

    print(f"KDE histogram: {kde_path}")
    print(f"Vanilla kNN histogram: {vanilla_path}")
    print(f"Overlay histogram: {overlay_path}")


if __name__ == "__main__":
    main()
