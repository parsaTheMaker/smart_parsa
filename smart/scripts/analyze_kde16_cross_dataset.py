#!/usr/bin/env python3
"""Compare SMART KDE-16 sampling shifts across 3D simulation datasets.

The strict surface protocol extracts boundary points from each volume mesh
before KDE estimation. Heat3D is retained as an explicitly labeled
computational-support control because its public canonical subset has a shared
point support but no surface connectivity.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
SMART_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(SMART_ROOT) not in sys.path:
    sys.path.insert(0, str(SMART_ROOT))

try:
    from utils.geometry_density import estimate_log_sampling_density
except ImportError:  # pragma: no cover
    from smart.utils.geometry_density import estimate_log_sampling_density

try:
    from utils.kde_sampling_stats import (
        density_stats,
        effective_sample_size,
        ks_distance,
        normalize_geometry,
        sample_inverse_density,
        sample_uniform,
        wasserstein_1d,
    )
except ImportError:  # pragma: no cover
    from smart.utils.kde_sampling_stats import (
        density_stats,
        effective_sample_size,
        ks_distance,
        normalize_geometry,
        sample_inverse_density,
        sample_uniform,
        wasserstein_1d,
    )


DEFAULT_SHIFT_CRASH_DIR = "/mnt/ssdraid/parsa/shift_crash_preprocessed"
DEFAULT_SFEM_DIR = "/mnt/data/sfem_samples/val_cases"
DEFAULT_LPBF_DIR = "/mnt/data/lpbf_flare_samples"
DEFAULT_HEAT3D_DIR = "/mnt/data/heat3d_samples/subsets/heat3d_v6_p1h_shared_support1024_v0"
DEFAULT_NVIDIA_DIR = "/mnt/data/physicsnemo_datacenter_cfd_samples/datasets/test"
DEFAULT_OUTPUT_DIR = "/home/parsa/smart_parsa/results/kde16_density_cross_dataset_10cases"

DATASET_COLORS = {
    "SHIFT-Crash": "#54a24b",
    "SFEM": "#e45756",
    "LPBF-FLARE": "#72b7b2",
    "Heat3D control": "#b279a2",
    "NVIDIA Datacenter CFD": "#4c78a8",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shift-crash-dir", default=DEFAULT_SHIFT_CRASH_DIR)
    parser.add_argument("--sfem-dir", default=DEFAULT_SFEM_DIR)
    parser.add_argument("--lpbf-dir", default=DEFAULT_LPBF_DIR)
    parser.add_argument("--heat3d-dir", default=DEFAULT_HEAT3D_DIR)
    parser.add_argument("--nvidia-dir", default=DEFAULT_NVIDIA_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--num-cases", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--knn-k", type=int, default=16)
    parser.add_argument(
        "--reference-points",
        type=int,
        default=0,
        help=(
            "Optional legacy cap on the surface source cloud before KDE. "
            "Use 0 (the default) to estimate density on the complete "
            "extracted surface cloud, matching DrivAerML."
        ),
    )
    parser.add_argument("--sample-budget", type=int, default=32768)
    parser.add_argument(
        "--small-mesh-fraction",
        type=float,
        default=0.5,
        help="For meshes below --sample-budget, use this fraction of the boundary cloud.",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is unavailable.")
    return torch.device(value)


def choose_ids(candidates: list[str], count: int, seed: int, offset: int) -> list[str]:
    candidates = sorted(str(value) for value in candidates)
    if count <= 0 or count > len(candidates):
        raise ValueError(f"Requested {count} cases, but only {len(candidates)} are available.")
    rng = np.random.default_rng(int(seed) + int(offset))
    indices = np.sort(rng.choice(len(candidates), size=count, replace=False))
    return [candidates[int(index)] for index in indices]


def optionally_cap_surface(
    surface: np.ndarray,
    reference_points: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, int]:
    """Keep the complete source cloud unless an explicit cap is requested."""
    surface = np.asarray(surface, dtype=np.float32)
    original_count = int(surface.shape[0])
    cap = int(reference_points)
    if cap > 0 and original_count > cap:
        indices = np.sort(rng.choice(original_count, size=cap, replace=False))
        surface = surface[indices]
    return surface, original_count


def load_nvidia_geometry(root: Path, case_name: str, reference_points: int, rng: np.random.Generator) -> tuple[np.ndarray, int]:
    path = root / case_name
    # The public files are regular Cartesian point clouds rather than surface
    # meshes. Use only the near-zero wall-distance layer as a surface proxy;
    # never feed the full volume point cloud into the density estimator.
    grid = pv.read(path)
    points = np.asarray(grid.points, dtype=np.float32)
    wall_distance = np.asarray(grid.point_data["wallDistance"], dtype=np.float32)
    valid = np.asarray(grid.point_data.get("vtkValidPointMask", np.ones(points.shape[0], dtype=np.uint8))) > 0
    spacing = np.inf
    for axis in range(3):
        values = np.unique(points[:, axis])
        differences = np.diff(values)
        positive = differences[differences > 0.0]
        spacing = min(spacing, float(positive.min()))
    if not np.isfinite(spacing):
        raise ValueError(f"Could not infer a positive grid spacing from {path}")
    surface = points[valid & (np.abs(wall_distance) <= 0.5 * spacing)]
    if surface.shape[0] < 8:
        raise ValueError(f"Wall-distance surface proxy has too few points in {path}: {surface.shape[0]}")
    return optionally_cap_surface(surface, reference_points, rng)


def load_shift_crash_geometry(root: Path, case_id: str, reference_points: int, rng: np.random.Generator) -> tuple[np.ndarray, int]:
    points = np.load(root / "cases" / case_id / "geometry_and_terminal_displacement.npy", mmap_mode="r")[:, :3]
    return optionally_cap_surface(points, reference_points, rng)


def load_sfem_geometry(root: Path, case_name: str, reference_points: int, rng: np.random.Generator) -> tuple[np.ndarray, int]:
    path = root / "raw" / "val" / case_name
    with h5py.File(path, "r") as handle:
        vertices = np.asarray(handle["Vertices"][:], dtype=np.float32)
        facets = np.asarray(handle["Facets"][:], dtype=np.int64)
    facets = facets[(facets >= 0).all(axis=1)]
    boundary_indices = np.unique(facets.reshape(-1))
    surface = vertices[boundary_indices]
    return optionally_cap_surface(surface, reference_points, rng)


def load_lpbf_geometry(root: Path, case_name: str, reference_points: int, rng: np.random.Generator) -> tuple[np.ndarray, int]:
    path = root / "test" / case_name
    data = np.load(path, allow_pickle=False)
    points = np.asarray(data["pos"], dtype=np.float32)
    elements = np.asarray(data["elems"], dtype=np.int64)
    if elements.ndim != 2 or elements.shape[1] != 8:
        raise ValueError(f"Expected hexahedral LPBF elements [M,8], got {elements.shape} in {path}")
    cells = np.concatenate(
        [np.full((elements.shape[0], 1), 8, dtype=np.int64), elements], axis=1
    ).reshape(-1)
    cell_types = np.full(elements.shape[0], int(pv.CellType.HEXAHEDRON), dtype=np.uint8)
    surface = np.asarray(pv.UnstructuredGrid(cells, cell_types, points).extract_surface().points, dtype=np.float32)
    return optionally_cap_surface(surface, reference_points, rng)


def load_heat3d_geometry(root: Path, case_name: str, reference_points: int, rng: np.random.Generator) -> tuple[np.ndarray, int]:
    # Heat3D's canonical public subset exposes coordinates but not connectivity.
    # Use the points on the axis-aligned support boundary as a conservative
    # boundary-support control, and keep this dataset separate from strict mesh
    # surface results in the report.
    points = np.asarray(np.load(root / case_name / "coords.npy"), dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected Heat3D coords [N,3], got {points.shape}")
    lower = points.min(axis=0)
    upper = points.max(axis=0)
    scale = np.maximum(upper - lower, 1.0e-12)
    boundary = np.any(np.isclose(points, lower, atol=1.0e-6 * scale) | np.isclose(points, upper, atol=1.0e-6 * scale), axis=1)
    surface = points[boundary]
    if surface.shape[0] < 8:
        surface = points
    return optionally_cap_surface(surface, reference_points, rng)


def compute_case(
    dataset_name: str,
    case_id: str,
    points: np.ndarray,
    original_count: int,
    case_number: int,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[dict, dict[str, np.ndarray]]:
    normalized, lower, upper = normalize_geometry(points)
    n_points = int(normalized.shape[0])
    if not 0.0 < float(args.small_mesh_fraction) <= 1.0:
        raise ValueError("--small-mesh-fraction must be in (0, 1].")
    budget = min(int(args.sample_budget), n_points)
    if n_points <= int(args.sample_budget):
        budget = min(n_points, max(8, int(round(n_points * float(args.small_mesh_fraction)))))
    started = time.perf_counter()
    point_tensor = torch.from_numpy(normalized).unsqueeze(0).to(device=device, dtype=torch.float32, non_blocking=device.type == "cuda")
    log_density = estimate_log_sampling_density(point_tensor, knn_k=int(args.knn_k), estimator="kde").squeeze(0).detach().cpu().numpy().astype(np.float64)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    rng = np.random.default_rng(int(args.seed) + 104729 * (case_number + 1))
    beta0_idx = sample_uniform(n_points, budget, rng)
    beta1_idx = sample_inverse_density(log_density, budget, beta=1.0, rng=rng)
    full = density_stats(log_density)
    beta0 = density_stats(log_density[beta0_idx])
    beta1 = density_stats(log_density[beta1_idx])
    row = {
        "dataset": dataset_name,
        "case_id": case_id,
        "geometry_protocol": (
            "support_boundary_control"
            if dataset_name == "Heat3D control"
            else "surface_distance_proxy"
            if dataset_name == "NVIDIA Datacenter CFD"
            else "strict_surface_boundary"
        ),
        "original_geometry_points": original_count,
        "density_reference_points": n_points,
        "sample_budget": budget,
        "sample_budget_cap": int(args.sample_budget),
        "small_mesh_fraction": float(args.small_mesh_fraction),
        "bbox_x": float(upper[0] - lower[0]),
        "bbox_y": float(upper[1] - lower[1]),
        "bbox_z": float(upper[2] - lower[2]),
        "kde_seconds": float(elapsed),
        "full_mean_log_density": full["mean_log_density"],
        "full_std_log_density": full["std_log_density"],
        "full_kde_density_cv": full["kde_density_cv"],
        "beta0_mean_log_density": beta0["mean_log_density"],
        "beta1_mean_log_density": beta1["mean_log_density"],
        "beta1_minus_beta0_mean_log_density": beta1["mean_log_density"] - beta0["mean_log_density"],
        "beta0_beta1_ks": ks_distance(log_density[beta0_idx], log_density[beta1_idx]),
        "beta0_beta1_wasserstein": wasserstein_1d(log_density[beta0_idx], log_density[beta1_idx]),
        "beta0_low_density_quartile_share": float(np.mean(log_density[beta0_idx] <= full["p25_log_density"])),
        "beta1_low_density_quartile_share": float(np.mean(log_density[beta1_idx] <= full["p25_log_density"])),
        "inverse_density_weight_ess_ratio": effective_sample_size(log_density, 1.0) / float(n_points),
    }
    arrays = {
        "normalized_geometry": normalized,
        "log_density": log_density,
        "beta0_idx": beta0_idx,
        "beta1_idx": beta1_idx,
    }
    return row, arrays


def _histogram_bins(values: np.ndarray, xscale: str) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    lo = float(np.quantile(values, 0.001))
    hi = float(np.quantile(values, 0.999))
    if xscale == "log":
        lo = max(lo, 1.0e-12)
        hi = max(hi, np.nextafter(lo, np.inf))
        return np.geomspace(lo, hi, 80)
    if hi <= lo:
        hi = np.nextafter(lo, np.inf)
    return np.linspace(lo, hi, 80)


def _save_overlay_histograms(
    output_dir: Path,
    dataset_name: str,
    beta0: np.ndarray,
    beta1: np.ndarray,
    sample_budget: int,
    stem: str,
    xlabel: str,
    allow_log_x: bool,
) -> None:
    safe_name = dataset_name.lower().replace(" ", "_").replace("+", "p").replace("-", "_")
    combined = np.concatenate((beta0, beta1))
    xscales = ("linear", "log") if allow_log_x else ("linear",)
    for xscale in xscales:
        bins = _histogram_bins(combined, xscale)
        for yscale in ("linear", "log"):
            fig, axis = plt.subplots(figsize=(13.5, 8.2), constrained_layout=True)
            axis.hist(beta0, bins=bins, color="#3478ae", alpha=0.86, label="beta=0.00")
            axis.hist(beta1, bins=bins, color="#d94f57", alpha=0.78, label="beta=1.00")
            if xscale == "log":
                axis.set_xscale("log")
            if yscale == "log":
                axis.set_yscale("log")
            axis.set_xlabel(xlabel)
            axis.set_ylabel("Count")
            axis.set_title(
                f"{dataset_name} sampled {stem.replace('_', ' ')} overlay "
                f"(beta=0.00 vs 1.00, points={sample_budget}, x={xscale}, y={yscale})"
            )
            axis.grid(True, axis="y", alpha=0.18)
            axis.legend(loc="upper left", frameon=True)
            fig.savefig(output_dir / f"{safe_name}_kde16_beta0_vs_beta1_{stem}_x{xscale}_y{yscale}.png", dpi=240)
            plt.close(fig)


def save_histogram(output_dir: Path, dataset_name: str, arrays: list[dict[str, np.ndarray]], sample_budget: int) -> None:
    beta0_density = np.concatenate([np.exp(np.clip(item["log_density"][item["beta0_idx"]], -700, 700)) for item in arrays])
    beta1_density = np.concatenate([np.exp(np.clip(item["log_density"][item["beta1_idx"]], -700, 700)) for item in arrays])
    _save_overlay_histograms(
        output_dir,
        dataset_name,
        beta0_density,
        beta1_density,
        sample_budget,
        stem="density_hist",
        xlabel=f"SMART KDE-16 density (sample budget={sample_budget})",
        allow_log_x=True,
    )
    # Preserve the two original filenames as stable aliases for existing slide
    # links: linear-x/linear-y and linear-x/log-y respectively.
    safe_name = dataset_name.lower().replace(" ", "_").replace("+", "p").replace("-", "_")
    for old_name, new_name in (
        ("linear", "xlinear_ylinear"),
        ("log", "xlinear_ylog"),
    ):
        source = output_dir / f"{safe_name}_kde16_beta0_vs_beta1_density_hist_{new_name}.png"
        target = output_dir / f"{safe_name}_kde16_beta0_vs_beta1_density_hist_{old_name}.png"
        if source.is_file():
            target.write_bytes(source.read_bytes())


def save_log_histogram(output_dir: Path, dataset_name: str, arrays: list[dict[str, np.ndarray]], sample_budget: int) -> None:
    beta0 = np.concatenate([item["log_density"][item["beta0_idx"]] for item in arrays])
    beta1 = np.concatenate([item["log_density"][item["beta1_idx"]] for item in arrays])
    _save_overlay_histograms(
        output_dir,
        dataset_name,
        beta0,
        beta1,
        sample_budget,
        stem="log_density_hist",
        xlabel=f"SMART KDE-16 log density (sample budget={sample_budget})",
        allow_log_x=False,
    )
    safe_name = dataset_name.lower().replace(" ", "_").replace("+", "p").replace("-", "_")
    source = output_dir / f"{safe_name}_kde16_beta0_vs_beta1_log_density_hist_xlinear_ylinear.png"
    target = output_dir / f"{safe_name}_kde16_beta0_vs_beta1_log_density_hist_linear.png"
    if source.is_file():
        target.write_bytes(source.read_bytes())


def save_cross_dataset_plot(output_dir: Path, dataset_rows: dict[str, list[dict]]) -> None:
    names = list(dataset_rows)
    x = np.arange(len(names))
    means = [float(np.mean([row["beta1_minus_beta0_mean_log_density"] for row in dataset_rows[name]])) for name in names]
    ks = [float(np.mean([row["beta0_beta1_ks"] for row in dataset_rows[name]])) for name in names]
    w1 = [float(np.mean([row["beta0_beta1_wasserstein"] for row in dataset_rows[name]])) for name in names]
    low_density_gain = [
        float(np.mean([row["beta1_low_density_quartile_share"] - row["beta0_low_density_quartile_share"] for row in dataset_rows[name]]))
        for name in names
    ]
    fig, axes = plt.subplots(1, 3, figsize=(17.0, 6.4), constrained_layout=True)
    for axis, values, ylabel, title in (
        (axes[0], means, "Mean log-density delta", "beta=1 minus beta=0"),
        (axes[1], w1, "Wasserstein distance", "Sampled distributions"),
        (axes[2], low_density_gain, "Low-density quartile share delta", "Inverse-density shift"),
    ):
        axis.bar(x, values, width=0.62, color=[DATASET_COLORS[name] for name in names])
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.set_xticks(x)
        axis.set_xticklabels(names, rotation=15, ha="right")
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.20)
    fig.suptitle(
        "KDE-16 sampling shift across 3D simulation datasets (Heat3D control; NVIDIA wall-distance surface proxy)",
        fontsize=18,
    )
    fig.savefig(output_dir / "cross_dataset_kde16_sampling_shift_summary.png", dpi=240)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(12.5, 7.0), constrained_layout=True)
    width = 0.28
    for offset, name in enumerate(names):
        values = np.asarray([row["beta1_minus_beta0_mean_log_density"] for row in dataset_rows[name]])
        axis.bar(offset * (len(names) + 1) + np.arange(values.size) * 0.1, values, width=width, alpha=0.75, color=DATASET_COLORS[name], label=name)
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_xlabel("Selected geometry index within dataset")
    axis.set_ylabel("beta=1 minus beta=0 mean log density")
    axis.set_title("Per-geometry consistency of the KDE sampling shift")
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=0.20)
    fig.savefig(output_dir / "cross_dataset_kde16_per_geometry_mean_shift.png", dpi=240)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    shift_crash_dir = Path(args.shift_crash_dir).expanduser().resolve()
    sfem_dir = Path(args.sfem_dir).expanduser().resolve()
    lpbf_dir = Path(args.lpbf_dir).expanduser().resolve()
    heat3d_dir = Path(args.heat3d_dir).expanduser().resolve()
    nvidia_dir = Path(args.nvidia_dir).expanduser().resolve()
    split_path = shift_crash_dir / "splits.json"
    if not split_path.is_file():
        raise FileNotFoundError(f"Missing SHIFT-Crash split file: {split_path}")
    shifts = json.loads(split_path.read_text())
    sfem_files = sorted(sfem_dir.glob("raw/val/*.h5"))
    lpbf_files = sorted(lpbf_dir.glob("test/*.npz"))
    heat3d_files = sorted(heat3d_dir.glob("*/coords.npy"))
    nvidia_files = sorted(nvidia_dir.glob("internal_*.vtu"), key=lambda path: int(path.stem.rsplit("_", 1)[1]))
    for label, files in (("SFEM", sfem_files), ("LPBF-FLARE", lpbf_files), ("Heat3D", heat3d_files), ("NVIDIA Datacenter CFD", nvidia_files)):
        if len(files) < int(args.num_cases):
            raise FileNotFoundError(f"Need {args.num_cases} {label} samples, found {len(files)}")
    reference_label = "full surface source" if int(args.reference_points) <= 0 else str(int(args.reference_points))
    print(f"Cross-dataset KDE analysis on device={device}, k={args.knn_k}, reference={reference_label}, budget={args.sample_budget}")

    shift_ids = choose_ids(list(shifts["train"]), int(args.num_cases), int(args.seed), 23)
    sfem_ids = []
    seen_sfem_geometries = set()
    for path in sfem_files:
        geometry_id = path.name.split("_", 1)[0]
        if geometry_id in seen_sfem_geometries:
            continue
        seen_sfem_geometries.add(geometry_id)
        sfem_ids.append(path.name)
        if len(sfem_ids) == int(args.num_cases):
            break
    if len(sfem_ids) < int(args.num_cases):
        raise FileNotFoundError(
            f"Need {args.num_cases} distinct SFEM geometries in {sfem_dir}, found {len(sfem_ids)}"
        )
    lpbf_ids = [path.name for path in lpbf_files[: int(args.num_cases)]]
    heat3d_ids = [path.parent.name for path in heat3d_files[: int(args.num_cases)]]
    nvidia_ids = [path.name for path in nvidia_files[: int(args.num_cases)]]
    selections = {
        "SHIFT-Crash": shift_ids,
        "SFEM": sfem_ids,
        "LPBF-FLARE": lpbf_ids,
        "Heat3D control": heat3d_ids,
        "NVIDIA Datacenter CFD": nvidia_ids,
    }
    all_rows: dict[str, list[dict]] = {name: [] for name in selections}
    all_arrays: dict[str, list[dict[str, np.ndarray]]] = {name: [] for name in selections}
    for dataset_name, case_ids in selections.items():
        print(f"\n{dataset_name}: {len(case_ids)} cases")
        for case_number, case_id in enumerate(case_ids):
            source_rng = np.random.default_rng(int(args.seed) + 7919 * (case_number + 1))
            if dataset_name == "SHIFT-Crash":
                points, original_count = load_shift_crash_geometry(shift_crash_dir, case_id, int(args.reference_points), source_rng)
            elif dataset_name == "SFEM":
                points, original_count = load_sfem_geometry(sfem_dir, case_id, int(args.reference_points), source_rng)
            elif dataset_name == "LPBF-FLARE":
                points, original_count = load_lpbf_geometry(lpbf_dir, case_id, int(args.reference_points), source_rng)
            elif dataset_name == "Heat3D control":
                points, original_count = load_heat3d_geometry(heat3d_dir, case_id, int(args.reference_points), source_rng)
            elif dataset_name == "NVIDIA Datacenter CFD":
                points, original_count = load_nvidia_geometry(nvidia_dir, case_id, int(args.reference_points), source_rng)
            else:  # pragma: no cover
                raise KeyError(dataset_name)
            row, arrays = compute_case(dataset_name, case_id, points, original_count, case_number, args, device)
            all_rows[dataset_name].append(row)
            all_arrays[dataset_name].append(arrays)
            print(
                f"  [{case_number + 1}/{len(case_ids)}] {case_id}: source={original_count}, reference={row['density_reference_points']}, "
                f"delta={row['beta1_minus_beta0_mean_log_density']:.5f}, KS={row['beta0_beta1_ks']:.5f}, "
                f"W1={row['beta0_beta1_wasserstein']:.5f}, time={row['kde_seconds']:.2f}s",
                flush=True,
            )
        save_histogram(output_dir, dataset_name, all_arrays[dataset_name], int(args.sample_budget))
        save_log_histogram(output_dir, dataset_name, all_arrays[dataset_name], int(args.sample_budget))

    save_cross_dataset_plot(output_dir, all_rows)
    flat_rows = [row for rows in all_rows.values() for row in rows]
    with (output_dir / "cross_dataset_kde16_case_statistics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat_rows[0].keys()))
        writer.writeheader()
        writer.writerows(flat_rows)
    averages = {}
    for name, rows in all_rows.items():
        averages[name] = {
            "geometry_protocol": rows[0]["geometry_protocol"],
            "mean_full_std_log_density": float(np.mean([row["full_std_log_density"] for row in rows])),
            "mean_full_kde_density_cv": float(np.mean([row["full_kde_density_cv"] for row in rows])),
            "mean_beta1_minus_beta0_log_density": float(np.mean([row["beta1_minus_beta0_mean_log_density"] for row in rows])),
            "mean_beta0_beta1_ks": float(np.mean([row["beta0_beta1_ks"] for row in rows])),
            "mean_beta0_beta1_wasserstein": float(np.mean([row["beta0_beta1_wasserstein"] for row in rows])),
            "mean_low_density_quartile_share_gain": float(np.mean([row["beta1_low_density_quartile_share"] - row["beta0_low_density_quartile_share"] for row in rows])),
            "mean_inverse_density_ess_ratio": float(np.mean([row["inverse_density_weight_ess_ratio"] for row in rows])),
        }
    summary = {
        "datasets": {
            name: {
                "case_ids": ids,
                "case_count": len(ids),
                "geometry_protocol": averages[name]["geometry_protocol"],
            }
            for name, ids in selections.items()
        },
        "estimator": "SMART estimate_log_sampling_density estimator=kde",
        "knn_k": int(args.knn_k),
        "reference_points": int(args.reference_points),
        "sample_budget": int(args.sample_budget),
        "small_mesh_fraction": float(args.small_mesh_fraction),
        "seed": int(args.seed),
        "device": str(device),
        "normalization": "per-geometry axis-aligned bounding-box normalization to [0,1]",
        "sampling": "beta=0 uniform without replacement; beta=1 p_i proportional to exp(-log_density_i), without replacement",
        "averages": averages,
        "notes": [
            "NVIDIA Datacenter CFD uses only valid near-zero wall-distance points, within half a Cartesian grid spacing, as an explicitly labeled surface proxy before KDE.",
            "SHIFT-Crash uses the complete surface geometry stored by its preprocessing pipeline before KDE.",
            "By default, every dataset estimates KDE on its complete available surface source cloud; a positive --reference-points value enables an explicit legacy cap.",
            "Heat3D control uses points on the axis-aligned support boundary because the public canonical subset has shared coordinates and no explicit mesh connectivity; it is not a strict CAD-surface result.",
            "These are density-shift diagnostics, not prediction-accuracy results.",
        ],
    }
    (output_dir / "cross_dataset_kde16_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output_dir / "selected_cases.json").write_text(json.dumps(selections, indent=2) + "\n")
    report_lines = [
        "KDE-16 cross-dataset sampling-shift analysis",
        "=============================================",
        f"KDE k={args.knn_k}; reference source={'full surface cloud' if int(args.reference_points) <= 0 else f'cap {args.reference_points}'}; sample budget={args.sample_budget}; seed={args.seed}",
        "",
        "Average diagnostics:",
    ]
    for name, values in averages.items():
        report_lines.extend(
            [
                f"{name}:",
                f"  full log-density std = {values['mean_full_std_log_density']:.6f}",
                f"  beta=1 minus beta=0 mean log-density = {values['mean_beta1_minus_beta0_log_density']:.6f}",
                f"  beta=0/beta=1 KS = {values['mean_beta0_beta1_ks']:.6f}",
                f"  beta=0/beta=1 Wasserstein = {values['mean_beta0_beta1_wasserstein']:.6f}",
                f"  low-density-quartile share gain = {values['mean_low_density_quartile_share_gain']:.6f}",
                f"  inverse-density ESS ratio = {values['mean_inverse_density_ess_ratio']:.6e}",
            ]
        )
    report_lines.extend(
        [
            "",
            "Interpretation:",
            "A consistent negative beta=1 minus beta=0 value means inverse-density sampling moves the input cloud toward lower local KDE density.",
            "The size of the KS/Wasserstein shift indicates whether the SATLoss beta intervention is mild or strong for that dataset.",
            "NVIDIA Datacenter CFD uses only the near-zero wall-distance surface proxy and never the full volume point cloud.",
            "Heat3D is reported separately as a boundary-support control, not as a surface-mesh benchmark.",
            "When a boundary cloud is smaller than the sample budget, the effective beta sample budget is 50% of that case's cloud so the without-replacement beta=0 and beta=1 distributions remain distinguishable; the effective budget is recorded in the CSV.",
            "These are sampling diagnostics only; they do not establish a prediction-accuracy improvement without training and evaluation.",
        ]
    )
    (output_dir / "README.txt").write_text("\n".join(report_lines) + "\n")
    print(f"\nSaved cross-dataset statistics and plots to {output_dir}")


if __name__ == "__main__":
    main()
