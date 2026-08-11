#!/usr/bin/env python3
"""DrivAerML-aligned sampling-invariance study for SHIFT-Submarine.

The protocol mirrors the established comparison used for DrivAerML:

* ``aligned_uniform_wor`` is the fixed control;
* beta, sine-y, and sine-x are evaluated at zero and maximum intensity;
* remeshed VTPs are separate encoder-source tests and are never additionally
  beta/sine shifted;
* queries and targets remain fixed per run and use train-split statistics;
* absolute errors, percentage worsening, endpoint tables, distributions, and
  representative prediction VTKs are written.

Only the submarine-specific data/model channel layout is different.  This
file intentionally does not modify the large DrivAerML comparator.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
SMART_ROOT = SCRIPT_DIR.parent
if str(SMART_ROOT) not in sys.path:
    sys.path.insert(0, str(SMART_ROOT))

from data.shift_submarine_dataset import ShiftSubmarineDataset  # noqa: E402
from models.smart.smart import SMART  # noqa: E402


DEFAULT_DATA_ROOT = Path("/mnt/ssdraid/parsa/shift_submarine_sample_preprocessed")
DEFAULT_SUMMARY = Path(
    "/home/parsa/smart_parsa/results/shift_submarine_sampling_study/"
    "shift_submarine_sampling_study_summary.json"
)
DEFAULT_OUTPUT = Path("/home/parsa/smart_parsa/results/shift_submarine_sampling_invariance_15runs")
SHIFT_ORDER = ("beta", "sine_y", "sine_x")
SOURCE_ORDER = ("original", "angle_div5", "angle_div10", "isotropic_div5", "isotropic_div10", "voxel_div5", "voxel_div10")
SOURCE_LABELS = {
    "original": "Original uniform",
    "angle_div5": "Angle div5",
    "angle_div10": "Angle div10",
    "isotropic_div5": "Isotropic div5",
    "isotropic_div10": "Isotropic div10",
    "voxel_div5": "Voxel div5",
    "voxel_div10": "Voxel div10",
}
MODEL_LABELS = {"BASE": "SMART", "SATLOSS7": "SATLOSS"}
# Match the established SMART/DrivAerML comparison palette: neutral baseline
# and blue SATLOSS, with the hatch retaining the second-method distinction.
MODEL_COLORS = {"BASE": "#6B7280", "SATLOSS7": "#1F77B4"}
SURFACE_FIELDS = ("pressure", "wall_shear_x", "wall_shear_y", "wall_shear_z")
VOLUME_FIELDS = ("pressure", "velocity_x", "velocity_y", "velocity_z")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--base-checkpoint", required=True, type=Path)
    parser.add_argument("--satloss7-checkpoint", required=True, type=Path)
    parser.add_argument("--num-runs", type=int, default=15)
    parser.add_argument("--run-ids", default=None, help="Optional comma-separated preprocessed run IDs.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shift-betas", default="0,1")
    parser.add_argument("--active-shifts", default="beta,sine_y,sine_x")
    parser.add_argument("--views-per-mode", type=int, default=2)
    parser.add_argument("--model-repeats", type=int, default=1)
    parser.add_argument("--input-points", type=int, default=131072)
    parser.add_argument("--surface-query-points", type=int, default=65536)
    parser.add_argument("--volume-query-points", type=int, default=65536)
    parser.add_argument("--query-chunk-size", type=int, default=65536)
    parser.add_argument("--devices", default="cuda:0,cuda:1")
    parser.add_argument("--plot-scales", default="linear,log")
    parser.add_argument("--font-scale", type=float, default=1.0)
    parser.add_argument("--no-std", action="store_true")
    parser.add_argument("--vtk-run-id", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def parse_devices(value: str) -> list[torch.device]:
    devices = []
    for item in str(value).split(","):
        item = item.strip()
        if not item:
            continue
        device = torch.device(item)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA devices were requested but CUDA is unavailable.")
        devices.append(device)
    return devices or [torch.device("cpu")]


def parse_active_shifts(value: str) -> list[str]:
    aliases = {"all": SHIFT_ORDER, "sine": ("sine_y", "sine_x"), "sinusoidal": ("sine_y", "sine_x")}
    requested = []
    for item in str(value).lower().replace("-", "_").split(","):
        item = item.strip()
        if not item:
            continue
        requested.extend(aliases.get(item, (item,)))
    invalid = sorted(set(requested) - set(SHIFT_ORDER))
    if invalid:
        raise ValueError(f"Unknown shifts {invalid}; valid shifts are {SHIFT_ORDER}.")
    active = [shift for shift in SHIFT_ORDER if shift in set(requested)]
    if not active:
        raise ValueError("At least one sampling shift must be active.")
    return active


def endpoint_values(value: str) -> tuple[float, float]:
    values = sorted(float(item.strip()) for item in str(value).split(",") if item.strip())
    if not values:
        raise ValueError("--shift-betas must contain at least one value.")
    return float(values[0]), float(values[-1])


def load_checkpoint(model: torch.nn.Module, path: Path, device: torch.device) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location="cpu")
    state = checkpoint.get("model_state_dict", checkpoint)
    if any(key.startswith("module.") for key in state):
        state = {key.removeprefix("module."): value for key, value in state.items()}
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    print(f"Loaded {path} on {device} (epoch={checkpoint.get('epoch', 'unknown')})", flush=True)


def make_model(config_name: str, checkpoint: Path, device: torch.device, query_chunk_size: int) -> torch.nn.Module:
    from omegaconf import OmegaConf

    base = OmegaConf.load(str(SMART_ROOT / "config" / "shift_submarine.yaml"))
    variant = OmegaConf.load(str(SMART_ROOT / "config" / config_name))
    config = OmegaConf.merge(base, variant)
    architecture = OmegaConf.to_container(config.experiment.architecture, resolve=True)
    model = SMART(
        spatial_dim=3,
        surface_channels=4,
        volume_channels=4,
        parameter_channels=0,
        **architecture,
    )
    model.subregion_size = max(int(getattr(model, "subregion_size", 262144)), int(query_chunk_size))
    load_checkpoint(model, checkpoint, device)
    print(f"{config_name}: architecture={architecture}", flush=True)
    return model


def stable_tag(value: str) -> int:
    return sum((index + 1) * ord(char) for index, char in enumerate(str(value)))


def normalize_positions(points: np.ndarray, minimum: np.ndarray, span: np.ndarray) -> np.ndarray:
    return ((np.asarray(points, dtype=np.float32) - minimum[None, :]) / span[None, :]).astype(np.float32, copy=False)


def sample_uniform(count: int, budget: int, rng: np.random.Generator) -> np.ndarray:
    if budget <= 0 or budget >= count:
        return np.arange(count, dtype=np.int64)
    return rng.choice(count, size=budget, replace=False).astype(np.int64, copy=False)


def sample_inverse_density(log_density: np.ndarray, budget: int, beta: float, rng: np.random.Generator) -> np.ndarray:
    if budget <= 0 or budget >= log_density.shape[0]:
        return np.arange(log_density.shape[0], dtype=np.int64)
    log_weights = -float(beta) * np.asarray(log_density, dtype=np.float64)
    log_weights -= np.max(log_weights)
    weights = np.exp(log_weights)
    weights = np.clip(weights, 1.0e-24, None)
    probabilities = weights / weights.sum()
    return rng.choice(log_density.shape[0], size=budget, replace=False, p=probabilities).astype(np.int64, copy=False)


def sine_probabilities(points: np.ndarray, axis: int) -> np.ndarray:
    values = np.asarray(points[:, axis], dtype=np.float64)
    span = max(float(values.max() - values.min()), 1.0e-12)
    normalized = np.clip((values - values.min()) / span, 0.0, 1.0)
    return np.clip(np.sin(np.pi * normalized) ** 2 + 1.0e-6, 1.0e-6, None)


def sample_sine(points: np.ndarray, budget: int, intensity: float, axis: int, rng: np.random.Generator) -> np.ndarray:
    if budget <= 0 or budget >= points.shape[0]:
        return np.arange(points.shape[0], dtype=np.int64)
    intensity = float(np.clip(intensity, 0.0, 1.0))
    if intensity <= 1.0e-12:
        return sample_uniform(points.shape[0], budget, rng)
    weighted_count = min(budget, max(0, int(round(intensity * budget))))
    probabilities = sine_probabilities(points, axis)
    weighted = rng.choice(points.shape[0], size=weighted_count, replace=False, p=probabilities / probabilities.sum())
    selected = np.zeros(points.shape[0], dtype=bool)
    selected[weighted] = True
    remaining = np.flatnonzero(~selected)
    uniform_count = budget - weighted_count
    if uniform_count > 0:
        uniform = rng.choice(remaining, size=uniform_count, replace=False)
    else:
        # At intensity=1 the weighted branch owns the complete fixed-size
        # view. Returning all remaining source points here silently changed
        # the endpoint from a 131k sample into the full cloud.
        uniform = np.empty((0,), dtype=np.int64)
    return np.concatenate([weighted, np.asarray(uniform, dtype=np.int64)]).astype(np.int64, copy=False)


def load_vtp_points(path: Path) -> np.ndarray:
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy

    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(str(path))
    reader.Update()
    polydata = reader.GetOutput()
    if polydata is None or polydata.GetPoints() is None:
        raise RuntimeError(f"VTP contains no points: {path}")
    points = np.asarray(vtk_to_numpy(polydata.GetPoints().GetData()), dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError(f"Invalid VTP points: {path}")
    return np.ascontiguousarray(points)


def fixed_queries(run_id: int, root: Path, dataset: ShiftSubmarineDataset, surface_budget: int, volume_budget: int, seed: int):
    run_dir = root / f"run_{run_id}"
    surface_coords = np.load(run_dir / "surface_coords.npy", mmap_mode="r")
    surface_data = np.load(run_dir / "surface_data.npy", mmap_mode="r")
    volume_coords = np.load(run_dir / "volume_coords.npy", mmap_mode="r")
    volume_data = np.load(run_dir / "volume_data.npy", mmap_mode="r")
    rng = np.random.default_rng(np.random.SeedSequence([seed, run_id, 9001]))
    surface_idx = sample_uniform(surface_coords.shape[0], surface_budget, rng)
    volume_idx = sample_uniform(volume_coords.shape[0], volume_budget, rng)
    minimum = dataset.min_pos.numpy().astype(np.float32)
    span = dataset.position_span.numpy().astype(np.float32)
    surf_y = (np.asarray(surface_data[surface_idx], dtype=np.float32) - dataset.mean_surf_data.numpy()) / dataset.std_surf_data.numpy()
    vol_y = (np.asarray(volume_data[volume_idx], dtype=np.float32) - dataset.mean_vol_data.numpy()) / dataset.std_vol_data.numpy()
    return {
        "surface_q": normalize_positions(np.asarray(surface_coords[surface_idx]), minimum, span),
        "surface_y": np.ascontiguousarray(surf_y, dtype=np.float32),
        "volume_q": normalize_positions(np.asarray(volume_coords[volume_idx]), minimum, span),
        "volume_y": np.ascontiguousarray(vol_y, dtype=np.float32),
        "surface_q_physical": np.asarray(surface_coords[surface_idx], dtype=np.float32),
        "minimum": minimum,
        "span": span,
    }


def rel_l2(target: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(prediction) - np.asarray(target)) / max(np.linalg.norm(target), 1.0e-12))


def predict_once(model: torch.nn.Module, device: torch.device, geometry: np.ndarray, surface_q: np.ndarray, volume_q: np.ndarray):
    geo = torch.from_numpy(geometry).unsqueeze(0).to(device, non_blocking=True)
    surf = torch.from_numpy(surface_q).unsqueeze(0).to(device, non_blocking=True)
    vol = torch.from_numpy(volume_q).unsqueeze(0).to(device, non_blocking=True)
    with torch.inference_mode():
        if device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                outputs = model.inference(geo, surf, vol, None)
        else:
            outputs = model.inference(geo, surf, vol, None)
    return outputs[0].float().cpu().numpy()[0], outputs[1].float().cpu().numpy()[0]


def predict_repeated(model, device, geometry, queries, repeats):
    surface_predictions = []
    volume_predictions = []
    for _ in range(max(1, int(repeats))):
        surf, vol = predict_once(model, device, geometry, queries["surface_q"], queries["volume_q"])
        surface_predictions.append(surf)
        volume_predictions.append(vol)
    return np.mean(surface_predictions, axis=0), np.mean(volume_predictions, axis=0)


def source_records(summary: dict, run_id: int) -> list[tuple[str, Path | None]]:
    record = next(item for item in summary["records"] if int(item["run_id"]) == int(run_id))
    paths = {str(key): Path(value) for key, value in record.get("outputs", {}).items() if Path(value).is_file()}
    result = [("original", None)]
    for source in SOURCE_ORDER[1:]:
        if source in paths:
            result.append((source, paths[source]))
    return result


def mode_records(active_shifts: list[str], beta_endpoints: tuple[float, float]) -> OrderedDict:
    modes = OrderedDict([("aligned_uniform_wor", {"kind": "uniform", "shift": "aligned", "intensity": 0.0, "label": "aligned uniform"})])
    if "beta" in active_shifts:
        for beta in beta_endpoints:
            modes[f"beta_{beta:.2f}"] = {"kind": "beta", "shift": "beta", "intensity": beta, "beta": beta, "label": f"beta={beta:.2f}"}
    if "sine_y" in active_shifts:
        modes["sine_y_0.00"] = {"kind": "sine", "shift": "sine_y", "intensity": 0.0, "axis": 1, "label": "sine-y=0.00"}
        modes["sine_y_1.00"] = {"kind": "sine", "shift": "sine_y", "intensity": 1.0, "axis": 1, "label": "sine-y=1.00"}
    if "sine_x" in active_shifts:
        modes["sine_x_0.00"] = {"kind": "sine", "shift": "sine_x", "intensity": 0.0, "axis": 0, "label": "sine-x=0.00"}
        modes["sine_x_1.00"] = {"kind": "sine", "shift": "sine_x", "intensity": 1.0, "axis": 0, "label": "sine-x=1.00"}
    return modes


def sample_encoder_indices(points: np.ndarray, density: np.ndarray | None, mode: dict, budget: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(np.random.SeedSequence([seed, stable_tag(mode["kind"]), stable_tag(mode["shift"])]))
    if mode["kind"] == "uniform" or float(mode.get("intensity", 0.0)) <= 1.0e-12:
        indices = sample_uniform(points.shape[0], budget, rng)
    elif mode["kind"] == "beta":
        if density is None:
            raise RuntimeError("Inverse-density shift requested but no submarine density cache was found.")
        indices = sample_inverse_density(density, budget, float(mode["beta"]), rng)
    else:
        indices = sample_sine(points, budget, float(mode["intensity"]), int(mode["axis"]), rng)
    return np.ascontiguousarray(indices, dtype=np.int64)


def sample_encoder_source(points: np.ndarray, density: np.ndarray | None, mode: dict, budget: int, seed: int) -> np.ndarray:
    indices = sample_encoder_indices(points, density, mode, budget, seed)
    return np.ascontiguousarray(points[indices], dtype=np.float32)


def aggregate(rows: list[dict], key_name: str) -> OrderedDict:
    grouped = OrderedDict()
    for row in rows:
        key = str(row[key_name])
        grouped.setdefault(key, []).append(row)
    result = OrderedDict()
    for key, values in grouped.items():
        result[key] = {}
        for metric in ("base_surface_rel_l2", "satloss7_surface_rel_l2", "base_volume_rel_l2", "satloss7_volume_rel_l2", "base_combined_rel_l2", "satloss7_combined_rel_l2"):
            numbers = np.asarray([float(item[metric]) for item in values], dtype=np.float64)
            result[key][f"{metric}_mean"] = float(numbers.mean())
            result[key][f"{metric}_std"] = float(numbers.std(ddof=1)) if numbers.size > 1 else 0.0
        base = result[key]["base_combined_rel_l2_mean"]
        sat = result[key]["satloss7_combined_rel_l2_mean"]
        result[key]["satloss7_improvement_percent_vs_base"] = 100.0 * (base - sat) / max(abs(base), 1.0e-12)
    return result


def aggregate_worsening(rows: list[dict]) -> OrderedDict:
    grouped = OrderedDict()
    for row in rows:
        grouped.setdefault(str(row["mode"]), []).append(row)
    result = OrderedDict()
    for mode, values in grouped.items():
        result[mode] = {}
        for metric in ("surface", "volume", "combined"):
            for model in ("base", "satloss7"):
                numbers = np.asarray([float(item[f"{model}_{metric}_pct_worsening"]) for item in values], dtype=np.float64)
                result[mode][f"{model}_{metric}_pct_worsening_mean"] = float(numbers.mean())
                result[mode][f"{model}_{metric}_pct_worsening_std"] = float(numbers.std(ddof=1)) if numbers.size > 1 else 0.0
    return result


def configure_plot(font_scale: float) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    size = 15.0 * float(font_scale)
    plt.rcParams.update({
        "font.size": size,
        "axes.titlesize": size,
        "axes.labelsize": size,
        "xtick.labelsize": size,
        "ytick.labelsize": size,
        "legend.fontsize": size,
    })


def paired_bar_plot(groups: list[dict], path: Path, title: str, ylabel: str, scale: str, font_scale: float, no_std: bool, percentage: bool = False) -> None:
    import matplotlib.pyplot as plt

    if not groups:
        return
    configure_plot(font_scale)
    x = np.arange(len(groups), dtype=np.float64)
    width = 0.34
    fig, ax = plt.subplots(figsize=(max(12.0, len(groups) * 1.65), 7.2))
    for offset, model in ((-width / 2.0, "BASE"), (width / 2.0, "SATLOSS7")):
        means = np.asarray([group[f"{model.lower()}_mean"] for group in groups])
        stds = np.asarray([group[f"{model.lower()}_std"] for group in groups])
        bars = ax.bar(
            x + offset,
            means,
            width=width,
            yerr=None if no_std else stds,
            capsize=4 if not no_std else 0,
            color=MODEL_COLORS[model],
            edgecolor="#202124",
            linewidth=0.8,
            hatch="///" if model == "SATLOSS7" else None,
            label=MODEL_LABELS[model],
        )
        if percentage:
            for bar, value in zip(bars, means):
                label_y = value + (1.2 if value >= 0.0 else -1.2)
                ax.text(bar.get_x() + bar.get_width() / 2.0, label_y, f"{value:+.1f}%", ha="center", va="bottom" if value >= 0.0 else "top", fontsize=11.5 * font_scale, fontweight="bold")
    ax.set_xticks(x, [group["label"] for group in groups], rotation=25, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=16)
    ax.tick_params(axis="both", labelsize=13 * font_scale)
    ax.grid(axis="y", alpha=0.22)
    if percentage and scale == "log":
        ax.set_yscale("symlog", linthresh=1.0)
    elif scale == "log":
        positive = [value for group in groups for value in (group["base_mean"], group["satloss7_mean"]) if value > 0.0]
        if positive:
            ax.set_yscale("log")
    ax.legend(frameon=True, ncols=2)
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.30, top=0.87)
    fig.savefig(path, dpi=260, bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)


def absolute_groups(aggregate_values: OrderedDict, metric: str, labels: list[tuple[str, str]]) -> list[dict]:
    groups = []
    for key, label in labels:
        value = aggregate_values.get(key)
        if value is None:
            continue
        groups.append({
            "label": label,
            "base_mean": value[f"base_{metric}_rel_l2_mean"],
            "base_std": value[f"base_{metric}_rel_l2_std"],
            "satloss7_mean": value[f"satloss7_{metric}_rel_l2_mean"],
            "satloss7_std": value[f"satloss7_{metric}_rel_l2_std"],
        })
    return groups


def build_shift_endpoint_summary_groups(
    rows: list[dict],
    beta_endpoints: tuple[float, float] = (0.0, 1.0),
) -> list[dict]:
    """Build four point-cloud-shift groups without the original control.

    The remeshing group averages every available remeshing source equally
    (angle/isotropic/voxel, including whichever div5/div10 factors exist).
    This keeps the summary valid for both the full six-source study and a
    reduced study containing one factor per method.
    """
    original_rows = [row for row in rows if str(row["source"]) == "original"]
    mode_aggregate = aggregate(original_rows, "mode")
    endpoint_modes = (
        (f"beta_{beta_endpoints[1]:.2f}", "beta=1"),
        ("sine_x_1.00", "sine-x=1"),
        ("sine_y_1.00", "sine-y=1"),
    )
    groups: list[dict] = []
    for mode, label in endpoint_modes:
        values = mode_aggregate.get(mode)
        if values is None:
            continue
        groups.append(
            {
                "label": label,
                "base_mean": values["base_combined_rel_l2_mean"],
                "base_std": values["base_combined_rel_l2_std"],
                "satloss7_mean": values["satloss7_combined_rel_l2_mean"],
                "satloss7_std": values["satloss7_combined_rel_l2_std"],
            }
        )

    remesh_rows = [
        row
        for row in rows
        if str(row["source"]) != "original" and str(row["mode"]) == "aligned_uniform_wor"
    ]
    if remesh_rows:
        base_values = np.asarray([float(row["base_combined_rel_l2"]) for row in remesh_rows], dtype=np.float64)
        satloss_values = np.asarray([float(row["satloss7_combined_rel_l2"]) for row in remesh_rows], dtype=np.float64)
        groups.append(
            {
                "label": "remeshing mean",
                "base_mean": float(base_values.mean()),
                "base_std": float(base_values.std(ddof=1)) if base_values.size > 1 else 0.0,
                "satloss7_mean": float(satloss_values.mean()),
                "satloss7_std": float(satloss_values.std(ddof=1)) if satloss_values.size > 1 else 0.0,
            }
        )
    return groups


def write_shift_endpoint_summary_table(groups: list[dict], output_dir: Path) -> None:
    fields = ["category", "smart_combined_rel_l2", "satloss_combined_rel_l2", "satloss_reduction_vs_smart_percent"]
    with (output_dir / "shift_submarine_combined_global_shift_endpoint_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for group in groups:
            base = float(group["base_mean"])
            satloss = float(group["satloss7_mean"])
            writer.writerow(
                {
                    "category": group["label"],
                    "smart_combined_rel_l2": base,
                    "satloss_combined_rel_l2": satloss,
                    "satloss_reduction_vs_smart_percent": 100.0 * (base - satloss) / max(abs(base), 1.0e-12),
                }
            )
    lines = [
        "# SHIFT-Submarine Point-Cloud Shift Summary",
        "",
        "SATLOSS reduction is relative to the paired SMART combined global relative L2 error.",
        "",
        "| Shift | SMART | SATLOSS | SATLOSS reduction vs SMART |",
        "|---|---:|---:|---:|",
    ]
    for group in groups:
        base = float(group["base_mean"])
        satloss = float(group["satloss7_mean"])
        reduction = 100.0 * (base - satloss) / max(abs(base), 1.0e-12)
        lines.append(f"| {group['label']} | {base:.6g} | {satloss:.6g} | {reduction:+.2f}% |")
    (output_dir / "shift_submarine_combined_global_shift_endpoint_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def plot_shift_endpoint_summary(
    groups: list[dict],
    path: Path,
    scale: str,
    font_scale: float,
    no_std: bool,
) -> None:
    import matplotlib.pyplot as plt

    if not groups:
        return
    configure_plot(font_scale)
    x = np.arange(len(groups), dtype=np.float64)
    width = 0.34
    fig, ax = plt.subplots(figsize=(max(12.0, len(groups) * 2.4), 7.0))
    for offset, model in ((-width / 2.0, "BASE"), (width / 2.0, "SATLOSS7")):
        means = np.asarray([group[f"{model.lower()}_mean"] for group in groups], dtype=np.float64)
        stds = np.asarray([group[f"{model.lower()}_std"] for group in groups], dtype=np.float64)
        bars = ax.bar(
            x + offset,
            means,
            width=width,
            yerr=None if no_std else stds,
            capsize=4 if not no_std else 0,
            color=MODEL_COLORS[model],
            edgecolor="#202124",
            linewidth=0.8,
            hatch="///" if model == "SATLOSS7" else None,
            label=MODEL_LABELS[model],
        )
        if model == "SATLOSS7":
            for bar, group in zip(bars, groups):
                base = float(group["base_mean"])
                satloss = float(group["satloss7_mean"])
                reduction = 100.0 * (base - satloss) / max(abs(base), 1.0e-12)
                label_y = satloss * 1.08 if satloss > 0.0 else satloss + 0.01
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    label_y,
                    f"{reduction:+.1f}%",
                    ha="center",
                    va="bottom",
                    fontsize=12.0 * font_scale,
                    fontweight="bold",
                    clip_on=False,
                )
    if scale == "log":
        positive = [
            value
            for group in groups
            for value in (float(group["base_mean"]), float(group["satloss7_mean"]))
            if value > 0.0
        ]
        if positive:
            ax.set_yscale("log")
    ax.set_xticks(x, [group["label"] for group in groups])
    ax.set_ylabel("Combined global relative L2")
    ax.set_title("SHIFT-Submarine point-cloud shifts: SATLOSS reduction vs SMART", pad=16)
    ax.tick_params(axis="both", labelsize=13 * font_scale)
    ax.grid(axis="y", which="both", alpha=0.22)
    ax.legend(frameon=True, ncols=2)
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.18, top=0.87)
    fig.savefig(path, dpi=280, bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)


def worsening_rows(rows: list[dict], mode_defs: OrderedDict) -> list[dict]:
    aligned = {(int(row["run_id"]), str(row["source"])): row for row in rows if row["mode"] == "aligned_uniform_wor"}
    output = []
    for row in rows:
        mode = str(row["mode"])
        if mode == "aligned_uniform_wor":
            continue
        # Shifted beta/sine rows are only produced for the original source.
        # Geometry-source rows are not shifted and therefore do not enter this
        # degradation calculation.
        if row["source"] != "original":
            continue
        baseline = aligned.get((int(row["run_id"]), "original"))
        if baseline is None:
            continue
        info = mode_defs[mode]
        result = dict(row)
        for metric in ("surface", "volume", "combined"):
            for model in ("base", "satloss7"):
                base_value = float(baseline[f"{model}_{metric}_rel_l2"])
                current = float(row[f"{model}_{metric}_rel_l2"])
                result[f"{model}_{metric}_pct_worsening"] = (
                    0.0
                    if float(info.get("intensity", 0.0)) <= 1.0e-12
                    else 100.0 * (current - base_value) / max(abs(base_value), 1.0e-12)
                )
        output.append(result)
    return output


def write_prediction_vtk(path: Path, points: np.ndarray, fields: dict[str, np.ndarray]) -> None:
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray

    path.parent.mkdir(parents=True, exist_ok=True)
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"VTK points must have shape [N, 3], got {points.shape}")
    point_count = int(points.shape[0])
    vtk_points = vtk.vtkPoints()
    vtk_points.SetData(numpy_to_vtk(points, deep=True))
    poly = vtk.vtkPolyData()
    poly.SetPoints(vtk_points)
    offsets = numpy_to_vtkIdTypeArray(np.arange(point_count + 1, dtype=np.int64), deep=True)
    connectivity = numpy_to_vtkIdTypeArray(np.arange(point_count, dtype=np.int64), deep=True)
    vertices = vtk.vtkCellArray()
    vertices.SetData(offsets, connectivity)
    poly.SetVerts(vertices)
    for name, values in fields.items():
        array = np.asarray(values, dtype=np.float32)
        if array.ndim == 1:
            array = array.reshape(-1, 1)
        if array.ndim != 2 or array.shape[0] != point_count:
            raise ValueError(f"VTK field {name!r} has incompatible shape {array.shape}")
        if array.shape[1] != 1:
            raise ValueError(f"VTK field {name!r} must be scalar, got {array.shape[1]} components")
        vtk_array = numpy_to_vtk(array[:, 0], deep=True)
        vtk_array.SetName(name)
        poly.GetPointData().AddArray(vtk_array)
    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(str(path))
    writer.SetInputData(poly)
    writer.SetFileTypeToBinary()
    if writer.Write() != 1:
        raise RuntimeError(f"VTK failed to write {path}")


def write_distribution_plot(points: np.ndarray, zero_points: np.ndarray, max_points: np.ndarray, shift: str, path: Path, font_scale: float) -> None:
    import matplotlib.pyplot as plt

    axis = 1 if shift == "sine_y" else 0
    values = np.asarray(points[:, axis], dtype=np.float64)
    minimum, maximum = values.min(), values.max()
    bins = np.linspace(minimum, maximum, 51)
    fig, (ax_hist, ax_delta) = plt.subplots(1, 2, figsize=(14.0, 5.8))
    ax_hist.hist(values, bins=bins, density=True, histtype="step", linewidth=2.0, color="#555555", label="full original cloud")
    ax_hist.hist(zero_points[:, axis], bins=bins, density=True, alpha=0.40, color="#2166ac", label="intensity 0")
    ax_hist.hist(max_points[:, axis], bins=bins, density=True, alpha=0.45, color="#b2182b", label="intensity 1")
    ax_hist.set_xlabel("Physical coordinate")
    ax_hist.set_ylabel("Probability density")
    ax_hist.set_title(f"{shift.replace('_', '-')} endpoint distributions")
    ax_hist.legend()
    bins2 = np.linspace(0.0, 1.0, 51)
    norm = lambda p: np.clip((p[:, axis] - minimum) / max(maximum - minimum, 1.0e-12), 0.0, 1.0)
    ax_delta.hist(norm(max_points) , bins=bins2, density=True, color="#b2182b", alpha=0.65, label="intensity 1")
    ax_delta.hist(norm(zero_points), bins=bins2, density=True, histtype="step", linewidth=2.0, color="#2166ac", label="intensity 0")
    ax_delta.set_xlabel(f"Normalized {'y' if axis == 1 else 'x'} coordinate")
    ax_delta.set_ylabel("Probability density")
    ax_delta.set_title("Redistribution relative to zero")
    ax_delta.legend()
    for axis_obj in (ax_hist, ax_delta):
        axis_obj.grid(alpha=0.22)
        axis_obj.tick_params(labelsize=13 * font_scale)
    fig.suptitle(f"SHIFT-Submarine sampling shift: {shift.replace('_', '-')}", fontsize=17 * font_scale)
    fig.tight_layout()
    fig.savefig(path, dpi=260, bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)


def write_tables(rows: list[dict], mode_defs: OrderedDict, output_dir: Path) -> None:
    aggregate_modes = aggregate(rows, "mode")
    fields = ["mode", "label", "base_combined_rel_l2_mean", "satloss7_combined_rel_l2_mean", "satloss7_improvement_percent_vs_base"]
    with (output_dir / "shift_submarine_sampling_endpoint_table.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for mode, values in aggregate_modes.items():
            writer.writerow({"mode": mode, "label": mode_defs[mode]["label"], **{field: values.get(field, "") for field in fields[2:]}})
    lines = ["# SHIFT-Submarine Sampling Endpoints", "", "| Mode | SMART | SATLOSS | SATLOSS improvement vs SMART |", "|---|---:|---:|---:|"]
    for mode, values in aggregate_modes.items():
        lines.append(f"| {mode_defs[mode]['label']} | {values['base_combined_rel_l2_mean']:.6g} | {values['satloss7_combined_rel_l2_mean']:.6g} | {values['satloss7_improvement_percent_vs_base']:+.2f}% |")
    (output_dir / "shift_submarine_sampling_endpoint_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.num_runs <= 0 or args.views_per_mode <= 0 or args.model_repeats <= 0:
        raise ValueError("num-runs, views-per-mode, and model-repeats must be positive.")
    if args.input_points <= 0 or args.surface_query_points <= 0 or args.volume_query_points <= 0:
        raise ValueError("Point budgets must be positive.")
    summary = json.loads(args.study_summary.read_text(encoding="utf-8"))
    available = sorted(int(record["run_id"]) for record in summary["records"])
    if args.run_ids:
        run_ids = [int(item.strip()) for item in str(args.run_ids).split(",") if item.strip()]
        missing = sorted(set(run_ids) - set(available))
        if missing:
            raise ValueError(f"Requested runs are not in the study summary: {missing}")
    else:
        rng = np.random.default_rng(args.seed + 7001)
        count = min(int(args.num_runs), len(available))
        run_ids = sorted(int(value) for value in rng.choice(np.asarray(available), size=count, replace=False))
    active_shifts = parse_active_shifts(args.active_shifts)
    beta_endpoints = endpoint_values(args.shift_betas)
    mode_defs = mode_records(active_shifts, beta_endpoints)
    devices = parse_devices(args.devices)
    base_device = devices[0]
    sat_device = devices[1] if len(devices) > 1 else devices[0]
    print(f"Active sampling shifts: {', '.join(active_shifts)}", flush=True)
    print(f"Runs: {run_ids}", flush=True)
    print(f"Devices: {base_device}, {sat_device}", flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    base_model = make_model("shift_submarine.yaml", args.base_checkpoint, base_device, args.query_chunk_size)
    sat_model = make_model("shift_submarine_satloss7.yaml", args.satloss7_checkpoint, sat_device, args.query_chunk_size)
    dataset = ShiftSubmarineDataset(args.data_root, if_test=True, geometry_points=0, surface_points=args.surface_query_points, volume_points=args.volume_query_points)
    rows: list[dict] = []
    vtk_run_id = int(args.vtk_run_id) if args.vtk_run_id is not None else int(run_ids[0])
    distribution_done: set[str] = set()
    representative_sampling_sets: dict[str, np.ndarray] = {}
    sampling_diagnostics: list[dict] = []
    vtk_input_dir = args.output_dir / "input_vtks"
    vtk_prediction_dir = args.output_dir / "prediction_vtks"
    for run_index, run_id in enumerate(run_ids):
        run_dir = args.data_root / f"run_{run_id}"
        original_points = np.asarray(np.load(run_dir / "surface_coords.npy", mmap_mode="r"), dtype=np.float32)
        density = dataset._load_density(run_id, original_points).numpy().astype(np.float32, copy=False)
        queries = fixed_queries(run_id, args.data_root, dataset, args.surface_query_points, args.volume_query_points, args.seed)
        sources = source_records(summary, run_id)
        for source_name, vtp_path in sources:
            physical_source = original_points if vtp_path is None else load_vtp_points(vtp_path)
            source_bounds = np.stack([physical_source.min(axis=0), physical_source.max(axis=0)])
            original_bounds = np.stack([original_points.min(axis=0), original_points.max(axis=0)])
            tolerance = np.maximum(original_bounds[1] - original_bounds[0], 1.0e-6) * 0.025
            if np.any(source_bounds[0] < original_bounds[0] - tolerance) or np.any(source_bounds[1] > original_bounds[1] + tolerance):
                raise ValueError(f"{source_name} run_{run_id} has a bounding box inconsistent with the original surface.")
            source_modes = ["aligned_uniform_wor"] if source_name != "original" else list(mode_defs)
            for mode_name in source_modes:
                mode = mode_defs[mode_name]
                for view_index in range(args.views_per_mode):
                    seed = args.seed + 1000003 * run_id + 10007 * run_index + 101 * view_index + stable_tag(source_name)
                    density_for_source = density if source_name == "original" else None
                    sampling_mode = mode
                    if float(mode.get("intensity", 0.0)) <= 1.0e-12:
                        # beta=0 and sine=0 are exact aligned controls, not
                        # independently sampled noisy baselines.
                        sampling_mode = {"kind": "uniform", "shift": "aligned", "intensity": 0.0}
                    selected_indices = sample_encoder_indices(
                        physical_source,
                        density_for_source,
                        sampling_mode,
                        args.input_points,
                        seed,
                    )
                    selected = np.ascontiguousarray(physical_source[selected_indices], dtype=np.float32)
                    geometry = normalize_positions(selected, queries["minimum"], queries["span"])
                    with ThreadPoolExecutor(max_workers=2) as executor:
                        base_future = executor.submit(predict_repeated, base_model, base_device, geometry, queries, args.model_repeats)
                        sat_future = executor.submit(predict_repeated, sat_model, sat_device, geometry, queries, args.model_repeats)
                        base_surf, base_vol = base_future.result()
                        sat_surf, sat_vol = sat_future.result()
                    base_surface = rel_l2(queries["surface_y"], base_surf)
                    sat_surface = rel_l2(queries["surface_y"], sat_surf)
                    base_volume = rel_l2(queries["volume_y"], base_vol)
                    sat_volume = rel_l2(queries["volume_y"], sat_vol)
                    rows.append({
                        "run_id": run_id,
                        "source": source_name,
                        "mode": mode_name,
                        "shift": mode["shift"],
                        "intensity": float(mode["intensity"]),
                        "view": view_index,
                        "base_surface_rel_l2": base_surface,
                        "satloss7_surface_rel_l2": sat_surface,
                        "base_volume_rel_l2": base_volume,
                        "satloss7_volume_rel_l2": sat_volume,
                        "base_combined_rel_l2": 0.5 * (base_surface + base_volume),
                        "satloss7_combined_rel_l2": 0.5 * (sat_surface + sat_volume),
                    })
                    if run_id == vtk_run_id and view_index == 0:
                        representative_sampling_sets[f"{source_name}:{mode_name}"] = selected.copy()
                        input_fields = {
                            "sampling_intensity": np.full(
                                (selected.shape[0],), float(mode.get("intensity", 0.0)), dtype=np.float32
                            ),
                            "sample_index": selected_indices.astype(np.float32, copy=False),
                        }
                        if source_name == "original":
                            input_fields["source_log_density"] = density[selected_indices]
                        write_prediction_vtk(
                            vtk_input_dir / f"run_{run_id}_{source_name}_{mode_name}_input.vtk",
                            selected,
                            input_fields,
                        )
                        sampling_diagnostics.append(
                            {
                                "run_id": int(run_id),
                                "source": source_name,
                                "mode": mode_name,
                                "shift": mode["shift"],
                                "intensity": float(mode.get("intensity", 0.0)),
                                "point_count": int(selected.shape[0]),
                                "source_point_count": int(physical_source.shape[0]),
                                "x_mean": float(selected[:, 0].mean()),
                                "y_mean": float(selected[:, 1].mean()),
                                "z_mean": float(selected[:, 2].mean()),
                                "x_std": float(selected[:, 0].std()),
                                "y_std": float(selected[:, 1].std()),
                                "z_std": float(selected[:, 2].std()),
                                "sampled_log_density_mean": (
                                    float(density[selected_indices].mean()) if source_name == "original" else None
                                ),
                                "sampled_log_density_std": (
                                    float(density[selected_indices].std()) if source_name == "original" else None
                                ),
                            }
                        )
                        mean_s = dataset.mean_surf_data.numpy()
                        std_s = dataset.std_surf_data.numpy()
                        gt = queries["surface_y"] * std_s + mean_s
                        base_physical = base_surf * std_s + mean_s
                        sat_physical = sat_surf * std_s + mean_s
                        fields = {}
                        for channel, field_name in enumerate(SURFACE_FIELDS):
                            fields[f"gt_{field_name}"] = gt[:, channel]
                            fields[f"base_{field_name}"] = base_physical[:, channel]
                            fields[f"satloss7_{field_name}"] = sat_physical[:, channel]
                            fields[f"base_error_{field_name}"] = np.abs(
                                base_physical[:, channel] - gt[:, channel]
                            )
                            fields[f"satloss7_error_{field_name}"] = np.abs(
                                sat_physical[:, channel] - gt[:, channel]
                            )
                        write_prediction_vtk(
                            vtk_prediction_dir / f"run_{run_id}_{source_name}_{mode_name}_surface.vtk",
                            queries["surface_q_physical"],
                            fields,
                        )
        print(f"[{run_index + 1}/{len(run_ids)}] evaluated run_{run_id}", flush=True)

        if run_id == vtk_run_id:
            for shift in active_shifts:
                zero_mode = "beta_0.00" if shift == "beta" else f"{shift}_0.00"
                max_mode = f"beta_{beta_endpoints[1]:.2f}" if shift == "beta" else f"{shift}_1.00"
                if zero_mode not in mode_defs or max_mode not in mode_defs:
                    continue
                zero_points = representative_sampling_sets.get(f"original:{zero_mode}")
                max_points = representative_sampling_sets.get(f"original:{max_mode}")
                if zero_points is None or max_points is None:
                    raise RuntimeError(
                        f"Representative samples for {shift} were not recorded; "
                        f"expected {zero_mode!r} and {max_mode!r}."
                    )
                write_distribution_plot(original_points, zero_points, max_points, shift, args.output_dir / f"shift_submarine_{shift}_endpoint_distribution.png", args.font_scale)
                distribution_done.add(shift)

    if not rows:
        raise RuntimeError("No comparison rows were produced.")
    fieldnames = list(rows[0].keys())
    with (args.output_dir / "shift_submarine_sampling_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    sampling_rows = [row for row in rows if row["source"] == "original"]
    all_aggregate = aggregate(rows, "mode")
    sampling_aggregate = aggregate(sampling_rows, "mode")
    source_aggregate = aggregate([row for row in rows if row["mode"] == "aligned_uniform_wor"], "source")
    shift_endpoint_groups = build_shift_endpoint_summary_groups(rows, beta_endpoints)
    write_shift_endpoint_summary_table(shift_endpoint_groups, args.output_dir)
    for scale in ("linear", "log"):
        plot_shift_endpoint_summary(
            shift_endpoint_groups,
            args.output_dir / f"shift_submarine_combined_global_shift_endpoint_summary_bars_{scale}.png",
            scale,
            args.font_scale,
            args.no_std,
        )
    payload = {
        "dataset": "SHIFT-Submarine-sample",
        "protocol": "DrivAerML-aligned endpoint sampling and geometry-source comparison",
        "run_ids": run_ids,
        "active_shifts": active_shifts,
        "shift_betas": beta_endpoints,
        "input_points": args.input_points,
        "surface_query_points": args.surface_query_points,
        "volume_query_points": args.volume_query_points,
        "base_checkpoint": str(args.base_checkpoint.resolve()),
        "satloss7_checkpoint": str(args.satloss7_checkpoint.resolve()),
        "metric": "surface/volume normalized global relative L2; combined is their mean",
        "note": "Beta/sine shifts apply only to original preprocessed surface inputs. VTP sources are uniform encoder-source tests.",
        "aggregate_by_mode": all_aggregate,
        "aggregate_by_sampling_mode_original_source": sampling_aggregate,
        "aggregate_by_source_aligned": source_aggregate,
    }
    (args.output_dir / "shift_submarine_sampling_metrics.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "shift_submarine_sampling_diagnostics.json").write_text(
        json.dumps(sampling_diagnostics, indent=2) + "\n", encoding="utf-8"
    )

    plot_scales = [item.strip() for item in str(args.plot_scales).split(",") if item.strip()]
    for shift in active_shifts:
        labels = [("beta_0.00", "beta=0"), (f"beta_{beta_endpoints[1]:.2f}", f"beta={beta_endpoints[1]:.2f}")] if shift == "beta" else [(f"{shift}_0.00", f"{shift.replace('_', '-')}=0"), (f"{shift}_1.00", f"{shift.replace('_', '-')}=1")]
        for scale in plot_scales:
            for metric in ("combined", "surface", "volume"):
                groups = absolute_groups(sampling_aggregate, metric, labels)
                paired_bar_plot(groups, args.output_dir / f"shift_submarine_{metric}_global_endpoint_bars_{shift}_{scale}.png", f"SHIFT-Submarine {shift.replace('_', '-')} endpoints", f"{metric.capitalize()} normalized relative L2", scale, args.font_scale, args.no_std)
    geometry_labels = [(source, SOURCE_LABELS[source]) for source in SOURCE_ORDER if source in source_aggregate]
    for scale in plot_scales:
        for metric in ("combined", "surface", "volume"):
            groups = absolute_groups(source_aggregate, metric, geometry_labels)
            paired_bar_plot(groups, args.output_dir / f"shift_submarine_{metric}_global_geometry_sources_bars_{scale}.png", "SHIFT-Submarine remeshed encoder sources", f"{metric.capitalize()} normalized relative L2", scale, args.font_scale, args.no_std)

    degradation = worsening_rows(rows, mode_defs)
    degradation_aggregate = aggregate_worsening(degradation) if degradation else OrderedDict()
    with (args.output_dir / "shift_submarine_sampling_worsening_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        if degradation:
            writer = csv.DictWriter(handle, fieldnames=list(degradation[0].keys()))
            writer.writeheader()
            writer.writerows(degradation)
    for shift in active_shifts:
        labels = [("beta_0.00", "beta=0"), (f"beta_{beta_endpoints[1]:.2f}", f"beta={beta_endpoints[1]:.2f}")] if shift == "beta" else [(f"{shift}_0.00", f"{shift.replace('_', '-')}=0"), (f"{shift}_1.00", f"{shift.replace('_', '-')}=1")]
        for scale in plot_scales:
            for metric in ("combined", "surface", "volume"):
                groups = []
                for key, label in labels:
                    value = degradation_aggregate.get(key)
                    if value is None:
                        continue
                    groups.append({
                        "label": label,
                        "base_mean": value.get(f"base_{metric}_pct_worsening_mean", 0.0),
                        "base_std": value.get(f"base_{metric}_pct_worsening_std", 0.0),
                        "satloss7_mean": value.get(f"satloss7_{metric}_pct_worsening_mean", 0.0),
                        "satloss7_std": value.get(f"satloss7_{metric}_pct_worsening_std", 0.0),
                    })
                paired_bar_plot(groups, args.output_dir / f"shift_submarine_{metric}_global_endpoint_percentage_worsening_{shift}_{scale}.png", f"SHIFT-Submarine {shift.replace('_', '-')} worsening vs aligned", "Worsening relative to aligned (%)", scale, args.font_scale, args.no_std, percentage=True)
    write_tables(sampling_rows, mode_defs, args.output_dir)
    print(f"Metrics: {args.output_dir / 'shift_submarine_sampling_metrics.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
