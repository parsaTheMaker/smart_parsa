#!/usr/bin/env python3
"""Run the SMART surface-only KDE-16 density-shift diagnostic on aerospace data.

The protocol mirrors the DrivAerML diagnostic:

1. use the complete available surface source cloud for KDE estimation;
2. normalize each geometry independently to its axis-aligned unit box;
3. draw beta=0 uniformly and beta=1 with probability proportional to
   ``exp(-log_density)`` without replacement;
4. compare the sampled local-density distributions.

Datasets supported by this script are deliberately geometry-backed:

* SHIFT-Wing: the official approximately 200k-point surface STL;
* SuperWing: selected rows of the public structured surface geometry array;
* AASM NASA CRM: the public 454,404-point 3D surface-coordinate cases.

The AASM missile benchmark is not included because its public HDF5 files hold
force/moment coefficients and empty geometry arrays, not a usable surface
point cloud. This script is a density diagnostic, not a flow-prediction
benchmark.
"""

from __future__ import annotations

import argparse
import ast
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
import requests
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
SMART_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(SMART_ROOT) not in sys.path:
    sys.path.insert(0, str(SMART_ROOT))

try:
    from utils.geometry_density import estimate_log_sampling_density
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
    from smart.utils.geometry_density import estimate_log_sampling_density
    from smart.utils.kde_sampling_stats import (
        density_stats,
        effective_sample_size,
        ks_distance,
        normalize_geometry,
        sample_inverse_density,
        sample_uniform,
        wasserstein_1d,
    )


DEFAULT_SHIFT_WING_DIR = "/mnt/data/navarro/Wing-sample"
DEFAULT_SUPERWING_CACHE_DIR = "/mnt/data/superwing_kde16_samples"
DEFAULT_AASM_CRM_FILE = "/mnt/data/aasm_benchmarks/testData_NASA-CRM.h5"
DEFAULT_OUTPUT_DIR = "/home/parsa/smart_parsa/results/kde16_density_aerospace_10cases"
SUPERWING_URL = "https://huggingface.co/datasets/yunplus/SuperWing/resolve/main/origingeom.npy?download=true"
SOURCE_LINKS = {
    "SHIFT-Wing": "https://huggingface.co/datasets/luminary-shift/Wing-sample",
    "SuperWing": "https://huggingface.co/datasets/yunplus/SuperWing",
    "AASM NASA CRM": "https://www.aiaa-appliedsurrogate.org/real-hover-problem",
}

DATASET_COLORS = {
    "SHIFT-Wing": "#2f6f9f",
    "SuperWing": "#d96b27",
    "AASM NASA CRM": "#4b8b5a",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shift-wing-dir", default=DEFAULT_SHIFT_WING_DIR)
    parser.add_argument("--superwing-cache-dir", default=DEFAULT_SUPERWING_CACHE_DIR)
    parser.add_argument("--superwing-url", default=SUPERWING_URL)
    parser.add_argument("--aasm-crm-file", default=DEFAULT_AASM_CRM_FILE)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--datasets",
        default="shift_wing,superwing,aasm_crm",
        help="Comma-separated subset of shift_wing,superwing,aasm_crm.",
    )
    parser.add_argument("--num-cases", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--knn-k", type=int, default=16)
    parser.add_argument(
        "--reference-points",
        type=int,
        default=0,
        help="Optional explicit source cap. 0 (default) uses every point in each chosen source file.",
    )
    parser.add_argument("--sample-budget", type=int, default=32768)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--request-timeout", type=float, default=120.0)
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
    points: np.ndarray,
    reference_points: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, int]:
    points = np.asarray(points, dtype=np.float32)
    original_count = int(points.shape[0])
    cap = int(reference_points)
    if cap > 0 and original_count > cap:
        indices = np.sort(rng.choice(original_count, size=cap, replace=False))
        points = points[indices]
    return points, original_count


def shift_wing_sample_dirs(root: Path) -> list[Path]:
    candidates = sorted(root.glob("OnShape_luminary_crm_version001/sample_*/merged_surfaces_onshape_200k.stl"))
    if not candidates:
        candidates = sorted(root.glob("**/merged_surfaces_onshape_200k.stl"))
    return candidates


def load_shift_wing_geometry(
    root: Path,
    case_id: str,
    reference_points: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, int, str]:
    path = Path(case_id)
    if not path.is_absolute():
        path = root / path
    mesh = pv.read(path)
    points, original_count = optionally_cap_surface(np.asarray(mesh.points, dtype=np.float32), reference_points, rng)
    return points, original_count, "official_approximately_200k_surface_stl"


def superwing_header(url: str, timeout: float) -> tuple[int, np.dtype, tuple[int, ...], int]:
    response = requests.get(url, headers={"Range": "bytes=0-4095"}, timeout=timeout)
    response.raise_for_status()
    raw = response.content
    if not raw.startswith(b"\x93NUMPY"):
        raise RuntimeError("SuperWing response did not contain a NumPy header.")
    major, minor = raw[6], raw[7]
    if (major, minor) != (1, 0):
        raise RuntimeError(f"Unsupported SuperWing NumPy format {major}.{minor}.")
    header_length = int.from_bytes(raw[8:10], byteorder="little", signed=False)
    header_start = 10
    header_end = header_start + header_length
    header = raw[header_start:header_end].decode("latin1")
    metadata = ast.literal_eval(header)
    dtype = np.dtype(metadata["descr"])
    shape = tuple(int(value) for value in metadata["shape"])
    return header_end, dtype, shape, int(response.headers.get("content-range", "/0").split("/")[-1])


def fetch_superwing_row(
    url: str,
    row_index: int,
    header_end: int,
    dtype: np.dtype,
    shape: tuple[int, ...],
    timeout: float,
) -> np.ndarray:
    row_bytes = int(np.prod(shape[1:], dtype=np.int64)) * int(dtype.itemsize)
    start = int(header_end) + int(row_index) * row_bytes
    end = start + row_bytes - 1
    response = requests.get(url, headers={"Range": f"bytes={start}-{end}"}, timeout=timeout)
    response.raise_for_status()
    payload = response.content
    if len(payload) != row_bytes:
        raise RuntimeError(f"SuperWing range returned {len(payload)} bytes, expected {row_bytes}.")
    return np.frombuffer(payload, dtype=dtype).reshape(shape[1:]).transpose(1, 2, 0).reshape(-1, 3).copy()


def superwing_row_ids(cache_dir: Path, shape: tuple[int, ...], count: int, seed: int) -> list[int]:
    existing = sorted(cache_dir.glob("shape_*.npy"))
    if len(existing) >= count:
        return [int(path.stem.split("_")[-1]) for path in existing[:count]]
    rng = np.random.default_rng(int(seed) + 17011)
    return sorted(int(value) for value in rng.choice(shape[0], size=count, replace=False))


def prepare_superwing_cases(args: argparse.Namespace, count: int) -> tuple[list[str], dict[str, np.ndarray], dict]:
    cache_dir = Path(args.superwing_cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    header_end, dtype, shape, file_size = superwing_header(args.superwing_url, float(args.request_timeout))
    if len(shape) != 4 or shape[1:] != (3, 129, 257):
        raise RuntimeError(f"Unexpected SuperWing geometry array shape: {shape}")
    row_ids = superwing_row_ids(cache_dir, shape, count, int(args.seed))
    geometries = {}
    for row_id in row_ids:
        case_id = f"shape_{row_id:04d}"
        cache_path = cache_dir / f"{case_id}.npy"
        if cache_path.is_file():
            points = np.load(cache_path, allow_pickle=False)
        else:
            points = fetch_superwing_row(
                args.superwing_url,
                row_id,
                header_end,
                dtype,
                shape,
                float(args.request_timeout),
            ).astype(np.float32, copy=False)
            np.save(cache_path, points)
        geometries[case_id] = points
    metadata = {
        "url": args.superwing_url,
        "shape": list(shape),
        "dtype": dtype.str,
        "file_size": file_size,
        "header_end": header_end,
        "source_protocol": "structured origingeom.npy rows, flattened from [3,129,257] to 3D surface vertices",
    }
    return [f"shape_{row_id:04d}" for row_id in row_ids], geometries, metadata


def aasm_case_ids(path: Path) -> list[str]:
    with h5py.File(path, "r") as handle:
        return sorted(str(key) for key in handle.keys() if str(key).startswith("Sample"))


def load_aasm_crm_geometry(
    path: Path,
    case_id: str,
    reference_points: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, int, str]:
    with h5py.File(path, "r") as handle:
        group = handle[case_id]
        points = np.column_stack(
            [
                np.asarray(group["CoordinateX"][:], dtype=np.float32),
                np.asarray(group["CoordinateY"][:], dtype=np.float32),
                np.asarray(group["CoordinateZ"][:], dtype=np.float32),
            ]
        )
    points, original_count = optionally_cap_surface(points, reference_points, rng)
    return points, original_count, "public_454404_point_3d_surface_coordinates"


def compute_case(
    dataset_name: str,
    case_id: str,
    points: np.ndarray,
    original_count: int,
    source_protocol: str,
    case_number: int,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[dict, dict[str, np.ndarray]]:
    normalized, lower, upper = normalize_geometry(points)
    n_points = int(normalized.shape[0])
    budget = min(int(args.sample_budget), n_points)
    started = time.perf_counter()
    point_tensor = torch.from_numpy(normalized).unsqueeze(0).to(
        device=device,
        dtype=torch.float32,
        non_blocking=device.type == "cuda",
    )
    log_density = (
        estimate_log_sampling_density(point_tensor, knn_k=int(args.knn_k), estimator="kde")
        .squeeze(0)
        .detach()
        .cpu()
        .numpy()
        .astype(np.float64)
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    del point_tensor
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
        "geometry_protocol": "strict_surface_coordinates",
        "source_protocol": source_protocol,
        "original_geometry_points": int(original_count),
        "density_reference_points": n_points,
        "sample_budget": budget,
        "sample_budget_cap": int(args.sample_budget),
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
        "log_density": log_density,
        "beta0_idx": beta0_idx,
        "beta1_idx": beta1_idx,
    }
    return row, arrays


def safe_name(value: str) -> str:
    return value.lower().replace(" ", "_").replace("+", "p").replace("-", "_")


def histogram_bins(values: np.ndarray, xscale: str) -> np.ndarray:
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


def save_overlay_histograms(
    output_dir: Path,
    dataset_name: str,
    beta0: np.ndarray,
    beta1: np.ndarray,
    sample_budget: int,
    stem: str,
    xlabel: str,
    allow_log_x: bool,
) -> None:
    combined = np.concatenate((beta0, beta1))
    xscales = ("linear", "log") if allow_log_x else ("linear",)
    for xscale in xscales:
        bins = histogram_bins(combined, xscale)
        for yscale in ("linear", "log"):
            fig, axis = plt.subplots(figsize=(13.5, 8.2), constrained_layout=True)
            axis.hist(beta0, bins=bins, color="#3478ae", alpha=0.86, label="beta=0.00")
            axis.hist(beta1, bins=bins, color="#d94f57", alpha=0.80, label="beta=1.00")
            if xscale == "log":
                axis.set_xscale("log")
            if yscale == "log":
                axis.set_yscale("log")
            axis.set_xlabel(xlabel, fontsize=17)
            axis.set_ylabel("Count", fontsize=17)
            axis.tick_params(labelsize=14)
            axis.set_title(
                f"{dataset_name} KDE-16 surface density: beta=0 vs beta=1\n"
                f"sample budget={sample_budget}, x={xscale}, y={yscale}",
                fontsize=18,
            )
            axis.grid(True, axis="y", alpha=0.18)
            axis.legend(loc="upper left", frameon=True, fontsize=14)
            fig.savefig(
                output_dir / f"{safe_name(dataset_name)}_kde16_beta0_vs_beta1_{stem}_x{xscale}_y{yscale}.png",
                dpi=240,
            )
            plt.close(fig)


def save_dataset_histograms(output_dir: Path, dataset_name: str, arrays: list[dict], sample_budget: int) -> None:
    density0 = np.concatenate([np.exp(np.clip(item["log_density"][item["beta0_idx"]], -700, 700)) for item in arrays])
    density1 = np.concatenate([np.exp(np.clip(item["log_density"][item["beta1_idx"]], -700, 700)) for item in arrays])
    save_overlay_histograms(
        output_dir,
        dataset_name,
        density0,
        density1,
        sample_budget,
        "density_hist",
        f"SMART KDE-16 density (sample budget={sample_budget})",
        True,
    )
    log0 = np.concatenate([item["log_density"][item["beta0_idx"]] for item in arrays])
    log1 = np.concatenate([item["log_density"][item["beta1_idx"]] for item in arrays])
    save_overlay_histograms(
        output_dir,
        dataset_name,
        log0,
        log1,
        sample_budget,
        "log_density_hist",
        f"SMART KDE-16 log density (sample budget={sample_budget})",
        False,
    )


def save_summary_plot(output_dir: Path, rows_by_dataset: dict[str, list[dict]]) -> None:
    names = list(rows_by_dataset)
    x = np.arange(len(names))
    metrics = (
        ("beta1_minus_beta0_mean_log_density", "Mean log-density delta", "beta=1 minus beta=0"),
        ("beta0_beta1_ks", "KS distance", "Sampled distributions"),
        ("beta0_beta1_wasserstein", "Wasserstein distance", "Sampled distributions"),
        ("inverse_density_weight_ess_ratio", "Inverse-density ESS ratio", "Effective support"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(16.0, 11.0), constrained_layout=True)
    for axis, (key, ylabel, title) in zip(axes.ravel(), metrics):
        values = [float(np.mean([row[key] for row in rows_by_dataset[name]])) for name in names]
        axis.bar(x, values, width=0.62, color=[DATASET_COLORS[name] for name in names])
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.set_xticks(x)
        axis.set_xticklabels(names, rotation=15, ha="right", fontsize=13)
        axis.set_ylabel(ylabel, fontsize=15)
        axis.set_title(title, fontsize=16)
        axis.tick_params(labelsize=12)
        axis.grid(axis="y", alpha=0.20)
    fig.suptitle("Surface-only KDE-16 sampling shift across aerospace datasets", fontsize=21)
    fig.savefig(output_dir / "aerospace_kde16_sampling_shift_summary.png", dpi=240)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    requested = [value.strip() for value in str(args.datasets).split(",") if value.strip()]
    valid = {"shift_wing", "superwing", "aasm_crm"}
    unknown = sorted(set(requested) - valid)
    if unknown:
        raise ValueError(f"Unknown dataset(s): {unknown}. Choose from {sorted(valid)}.")
    if not requested:
        raise ValueError("At least one dataset must be selected.")
    if int(args.knn_k) < 1:
        raise ValueError("--knn-k must be positive.")
    if int(args.sample_budget) < 8:
        raise ValueError("--sample-budget must be at least 8.")

    shift_root = Path(args.shift_wing_dir)
    aasm_path = Path(args.aasm_crm_file)
    shift_paths = shift_wing_sample_dirs(shift_root) if "shift_wing" in requested else []
    if "shift_wing" in requested and len(shift_paths) < int(args.num_cases):
        raise FileNotFoundError(f"Need {args.num_cases} SHIFT-Wing official surface STLs, found {len(shift_paths)}.")
    if "aasm_crm" in requested and not aasm_path.is_file():
        raise FileNotFoundError(f"AASM NASA CRM file not found: {aasm_path}")

    selections: dict[str, list[str]] = {}
    source_metadata: dict[str, dict] = {}
    superwing_geometries: dict[str, np.ndarray] = {}
    if "shift_wing" in requested:
        selected = choose_ids([str(path) for path in shift_paths], int(args.num_cases), int(args.seed), 11)
        selections["SHIFT-Wing"] = selected
    if "superwing" in requested:
        selected, superwing_geometries, metadata = prepare_superwing_cases(args, int(args.num_cases))
        selections["SuperWing"] = selected
        source_metadata["SuperWing"] = metadata
    if "aasm_crm" in requested:
        selections["AASM NASA CRM"] = choose_ids(aasm_case_ids(aasm_path), int(args.num_cases), int(args.seed), 29)

    print(f"Device: {device}")
    print(f"Surface KDE protocol: complete source cloud unless --reference-points is explicitly positive")
    print(f"Datasets: {', '.join(requested)}; cases per dataset: {args.num_cases}; KDE k: {args.knn_k}")
    rows_by_dataset: dict[str, list[dict]] = {name: [] for name in selections}
    arrays_by_dataset: dict[str, list[dict]] = {name: [] for name in selections}

    for dataset_index, (dataset_name, case_ids) in enumerate(selections.items()):
        print(f"\n[{dataset_name}] selected {len(case_ids)} cases")
        for case_number, case_id in enumerate(case_ids):
            source_rng = np.random.default_rng(int(args.seed) + 7919 * (dataset_index + 1) + case_number)
            if dataset_name == "SHIFT-Wing":
                points, original_count, protocol = load_shift_wing_geometry(shift_root, case_id, int(args.reference_points), source_rng)
            elif dataset_name == "SuperWing":
                points, original_count = optionally_cap_surface(superwing_geometries[case_id], int(args.reference_points), source_rng)
                protocol = str(source_metadata["SuperWing"]["source_protocol"])
            elif dataset_name == "AASM NASA CRM":
                points, original_count, protocol = load_aasm_crm_geometry(aasm_path, case_id, int(args.reference_points), source_rng)
            else:  # pragma: no cover
                raise KeyError(dataset_name)
            row, arrays = compute_case(
                dataset_name,
                case_id,
                points,
                original_count,
                protocol,
                case_number,
                args,
                device,
            )
            rows_by_dataset[dataset_name].append(row)
            arrays_by_dataset[dataset_name].append(arrays)
            print(
                f"  [{case_number + 1}/{len(case_ids)}] {Path(case_id).name}: "
                f"source={original_count}, reference={row['density_reference_points']}, "
                f"delta={row['beta1_minus_beta0_mean_log_density']:.5f}, "
                f"KS={row['beta0_beta1_ks']:.5f}, W1={row['beta0_beta1_wasserstein']:.5f}, "
                f"time={row['kde_seconds']:.2f}s",
                flush=True,
            )
        save_dataset_histograms(output_dir, dataset_name, arrays_by_dataset[dataset_name], int(args.sample_budget))

    save_summary_plot(output_dir, rows_by_dataset)
    flat_rows = [row for rows in rows_by_dataset.values() for row in rows]
    with (output_dir / "aerospace_kde16_case_statistics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat_rows[0].keys()))
        writer.writeheader()
        writer.writerows(flat_rows)

    averages = {}
    for name, rows in rows_by_dataset.items():
        averages[name] = {
            "geometry_protocol": rows[0]["geometry_protocol"],
            "source_protocol": rows[0]["source_protocol"],
            "mean_source_points": float(np.mean([row["original_geometry_points"] for row in rows])),
            "mean_density_reference_points": float(np.mean([row["density_reference_points"] for row in rows])),
            "mean_full_std_log_density": float(np.mean([row["full_std_log_density"] for row in rows])),
            "mean_beta1_minus_beta0_log_density": float(np.mean([row["beta1_minus_beta0_mean_log_density"] for row in rows])),
            "mean_beta0_beta1_ks": float(np.mean([row["beta0_beta1_ks"] for row in rows])),
            "mean_beta0_beta1_wasserstein": float(np.mean([row["beta0_beta1_wasserstein"] for row in rows])),
            "mean_inverse_density_ess_ratio": float(np.mean([row["inverse_density_weight_ess_ratio"] for row in rows])),
        }

    summary = {
        "protocol": "SMART surface-only KDE-16 density diagnostic",
        "datasets": {name: {"case_ids": ids, "case_count": len(ids)} for name, ids in selections.items()},
        "estimator": "estimate_log_sampling_density(estimator='kde')",
        "knn_k": int(args.knn_k),
        "reference_points": int(args.reference_points),
        "sample_budget": int(args.sample_budget),
        "seed": int(args.seed),
        "device": str(device),
        "normalization": "per-geometry axis-aligned bounding-box normalization to [0,1]",
        "sampling": "beta=0 uniform without replacement; beta=1 p_i proportional to exp(-log_density_i), without replacement",
        "averages": averages,
        "source_metadata": source_metadata,
        "source_links": {name: SOURCE_LINKS[name] for name in selections},
        "notes": [
            "Every density_reference_points value equals the complete chosen source file unless --reference-points is positive.",
            "SHIFT-Wing uses the dataset-provided merged_surfaces_onshape_200k.stl, not the much larger solver cloud in data.h5.",
            "SuperWing rows are fetched by HTTP range requests from origingeom.npy and flattened from the structured 3D surface grid.",
            "AASM NASA CRM uses CoordinateX/Y/Z from the public testData_NASA-CRM.h5; the 454,404 coordinates are treated as its supplied surface cloud.",
            "The AASM missile coefficient files are intentionally excluded because they contain no usable surface coordinates.",
            "These are density-shift diagnostics, not prediction-accuracy results.",
        ],
    }
    (output_dir / "aerospace_kde16_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output_dir / "selected_cases.json").write_text(json.dumps(selections, indent=2) + "\n")
    report = [
        "Surface-only KDE-16 aerospace density-shift diagnostic",
        "=======================================================",
        f"k={args.knn_k}; source cap={'none' if int(args.reference_points) <= 0 else args.reference_points}; sample budget={args.sample_budget}; seed={args.seed}",
        "",
        "Mean diagnostics:",
    ]
    for name, values in averages.items():
        report.extend(
            [
                f"{name}:",
                f"  mean source points = {values['mean_source_points']:.1f}",
                f"  mean density-reference points = {values['mean_density_reference_points']:.1f}",
                f"  full log-density std = {values['mean_full_std_log_density']:.6f}",
                f"  beta=1 minus beta=0 mean log-density = {values['mean_beta1_minus_beta0_log_density']:.6f}",
                f"  beta=0/beta=1 KS = {values['mean_beta0_beta1_ks']:.6f}",
                f"  beta=0/beta=1 Wasserstein = {values['mean_beta0_beta1_wasserstein']:.6f}",
                f"  inverse-density ESS ratio = {values['mean_inverse_density_ess_ratio']:.6e}",
            ]
        )
    report.extend(
        [
            "",
            "Interpretation:",
            "A negative beta=1 minus beta=0 mean log-density means inverse-density sampling moves the input cloud toward lower local KDE density.",
            "KS and Wasserstein quantify the size of the sampling intervention independently of prediction accuracy.",
            "The density is estimated only on surface coordinates, never on interior CFD points.",
        ]
    )
    (output_dir / "README.txt").write_text("\n".join(report) + "\n")
    print(f"\nSaved aerospace KDE-16 statistics and plots to {output_dir}")


if __name__ == "__main__":
    main()
