#!/usr/bin/env python3
"""Compare SMART and SATLOSS7 on the preprocessed SHIFT-Pump cases.

The comparison follows the established DrivAerML protocol without importing
its aerodynamic assumptions: queries and targets are fixed per case, only the
surface encoder cloud is resampled, beta uses the cached surface KDE, and
sine-x/sine-y use deterministic coordinate redistribution.  Pump parameters
are passed to both models from ``params.json`` after train-split
standardization.
"""

from __future__ import annotations

import argparse
import csv
import json
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

from data.pump_dataset import PumpDataset  # noqa: E402
from models.smart.smart import SMART  # noqa: E402
from utils.strategy_sampling import sample_strategy  # noqa: E402
from scripts.compare_shift_submarine_sampling_invariance import (  # noqa: E402
    endpoint_values,
    load_vtp_points,
    mode_records,
    normalize_positions,
    parse_active_shifts,
    sample_encoder_indices,
    sample_uniform,
    stable_tag,
    write_prediction_vtk,
)


DEFAULT_DATA_ROOT = Path("/mnt/ssdraid/parsa/shift_pump_preprocessed")
DEFAULT_OUTPUT = Path("/home/parsa/smart_parsa/results/pump_sampling_invariance")
MODEL_COLORS = {"BASE": "#6B7280", "SATLOSS": "#1F77B4"}
MODEL_LABELS = {"BASE": "SMART", "SATLOSS": "SATLOSS"}
SURFACE_FIELDS = PumpDataset.SURFACE_FIELDS
VOLUME_FIELDS = PumpDataset.VOLUME_FIELDS
SOURCE_ORDER = ("original", "angle_div5", "angle_div10", "isotropic_div5", "isotropic_div10", "voxel_div5", "voxel_div10")
SOURCE_LABELS = {
    "original": "Original surface",
    "angle_div5": "Angle div5",
    "angle_div10": "Angle div10",
    "isotropic_div5": "Isotropic div5",
    "isotropic_div10": "Isotropic div10",
    "voxel_div5": "Voxel div5",
    "voxel_div10": "Voxel div10",
}
STRATEGY_ORDER = ("downsample", "gaussian_ball_masked", "box_masked")
STRATEGY_LABELS = {
    "downsample": "Downsample",
    "gaussian_ball_masked": "Gaussian-ball mask",
    "box_masked": "Box mask",
}
STRATEGY_COLORS = {
    "base": "#6B7280",
    "satloss": "#1F77B4",
    "downsample": "#9467BD",
    "gaussian_ball_masked": "#2CA02C",
    "box_masked": "#D62728",
}
MODEL_LABELS_ALL = {
    "base": "SMART",
    "satloss": "SATLOSS",
    "downsample": "Downsample",
    "gaussian_ball_masked": "Gaussian-ball mask",
    "box_masked": "Box mask",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--study-summary", type=Path, default=None, help="Remeshing study summary JSON; omit for beta/sine-only comparison.")
    parser.add_argument("--active-geometry-sources", default="original", help="Comma-separated sources: original, angle, isotropic, voxel, or all.")
    parser.add_argument("--geometry-decimation-factors", default="5,10")
    parser.add_argument("--active-strategies", default="downsample,gaussian_ball_masked,box_masked")
    parser.add_argument("--downsample-config", default="pump_satloss7_downsample")
    parser.add_argument("--gaussian-ball-masked-config", default="pump_satloss7_gaussian_ball_masked")
    parser.add_argument("--box-masked-config", default="pump_satloss7_box_masked")
    parser.add_argument("--base-checkpoint", required=True, type=Path)
    parser.add_argument("--satloss-checkpoint", "--satloss7-checkpoint", dest="satloss_checkpoint", required=True, type=Path)
    parser.add_argument("--downsample-checkpoint", type=Path, default=None)
    parser.add_argument("--gaussian-ball-masked-checkpoint", type=Path, default=None)
    parser.add_argument("--box-masked-checkpoint", type=Path, default=None)
    parser.add_argument("--num-runs", type=int, default=10)
    parser.add_argument("--run-ids", default=None)
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
    devices = [torch.device(item.strip()) for item in str(value).split(",") if item.strip()]
    if not devices:
        return [torch.device("cpu")]
    if any(device.type == "cuda" for device in devices) and not torch.cuda.is_available():
        raise RuntimeError("CUDA devices were requested but CUDA is unavailable.")
    return devices


def parse_active_sources(value: str) -> list[str]:
    aliases = {"all": SOURCE_ORDER, "angle": ("angle_div5", "angle_div10"), "isotropic": ("isotropic_div5", "isotropic_div10"), "voxel": ("voxel_div5", "voxel_div10")}
    requested = []
    for item in str(value).lower().replace("-", "_").split(","):
        item = item.strip()
        if item:
            requested.extend(aliases.get(item, (item,)))
    invalid = sorted(set(requested) - set(SOURCE_ORDER))
    if invalid:
        raise ValueError(f"Unknown geometry sources {invalid}; valid sources are {SOURCE_ORDER}.")
    active = [source for source in SOURCE_ORDER if source in set(requested)]
    if not active:
        raise ValueError("At least one geometry source must be active.")
    return active


def parse_active_strategies(value: str) -> list[str]:
    aliases = {"all": STRATEGY_ORDER, "gaussian": ("gaussian_ball_masked",), "box": ("box_masked",), "subsample": ("downsample",)}
    requested = []
    for item in str(value).lower().replace("-", "_").split(","):
        item = item.strip()
        if item:
            requested.extend(aliases.get(item, (item,)))
    invalid = sorted(set(requested) - set(STRATEGY_ORDER))
    if invalid:
        raise ValueError(f"Unknown strategies {invalid}; valid strategies are {STRATEGY_ORDER}.")
    return [strategy for strategy in STRATEGY_ORDER if strategy in set(requested)]


def source_records(summary: dict | None, run_id: int, active_sources: list[str]) -> list[tuple[str, Path | None]]:
    result = [("original", None)] if "original" in active_sources else []
    if summary is None:
        if any(source != "original" for source in active_sources):
            raise ValueError("Remeshed geometry sources require --study-summary.")
        return result
    record = next((item for item in summary.get("records", []) if int(item.get("run_id", -1)) == int(run_id)), None)
    if record is None:
        raise ValueError(f"Run {run_id} is missing from the remeshing study summary.")
    outputs = {str(key): Path(value) for key, value in record.get("outputs", {}).items()}
    for source in active_sources:
        if source == "original":
            continue
        path = outputs.get(source)
        if path is None or not path.is_file():
            raise FileNotFoundError(f"Missing remeshed source {source} for run {run_id}: {path}")
        result.append((source, path))
    return result


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

    base = OmegaConf.load(str(SMART_ROOT / "config" / "pump.yaml"))
    variant = OmegaConf.load(str(SMART_ROOT / "config" / config_name))
    config = OmegaConf.merge(base, variant)
    architecture = OmegaConf.to_container(config.experiment.architecture, resolve=True)
    model = SMART(spatial_dim=3, surface_channels=7, volume_channels=4, parameter_channels=13, **architecture)
    model.subregion_size = max(int(getattr(model, "subregion_size", 262144)), int(query_chunk_size))
    load_checkpoint(model, checkpoint, device)
    print(f"{config_name}: architecture={architecture}", flush=True)
    return model


def fixed_queries(run_id: int, root: Path, dataset: PumpDataset, surface_budget: int, volume_budget: int, seed: int) -> dict:
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
        "params": dataset.get_case_params(run_id),
    }


def predict_once(model: torch.nn.Module, device: torch.device, geometry: np.ndarray, queries: dict) -> tuple[np.ndarray, np.ndarray]:
    geo = torch.from_numpy(geometry).unsqueeze(0).to(device, non_blocking=True)
    surf = torch.from_numpy(queries["surface_q"]).unsqueeze(0).to(device, non_blocking=True)
    vol = torch.from_numpy(queries["volume_q"]).unsqueeze(0).to(device, non_blocking=True)
    params = torch.from_numpy(queries["params"]).unsqueeze(0).to(device, non_blocking=True)
    with torch.inference_mode():
        if device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                surface, volume = model.inference(geo, surf, vol, params)
        else:
            surface, volume = model.inference(geo, surf, vol, params)
    return surface.float().cpu().numpy()[0], volume.float().cpu().numpy()[0]


def predict_repeated(model, device, geometry, queries, repeats):
    surfaces, volumes = [], []
    for _ in range(max(1, int(repeats))):
        surface, volume = predict_once(model, device, geometry, queries)
        surfaces.append(surface); volumes.append(volume)
    return np.mean(surfaces, axis=0), np.mean(volumes, axis=0)


def rel_l2(target: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(prediction) - np.asarray(target)) / max(np.linalg.norm(target), 1.0e-12))


def aggregate(rows: list[dict], key_name: str, model_keys: tuple[str, ...] = ("base", "satloss")) -> OrderedDict:
    grouped = OrderedDict()
    for row in rows:
        grouped.setdefault(str(row[key_name]), []).append(row)
    result = OrderedDict()
    for key, values in grouped.items():
        result[key] = {}
        for metric in ("surface", "volume", "combined"):
            for model in model_keys:
                if not values or f"{model}_{metric}_rel_l2" not in values[0]:
                    continue
                numbers = np.asarray([float(item[f"{model}_{metric}_rel_l2"]) for item in values], dtype=np.float64)
                result[key][f"{model}_{metric}_rel_l2_mean"] = float(numbers.mean())
                result[key][f"{model}_{metric}_rel_l2_std"] = float(numbers.std(ddof=1)) if numbers.size > 1 else 0.0
        if "base_combined_rel_l2_mean" in result[key] and "satloss_combined_rel_l2_mean" in result[key]:
            base = result[key]["base_combined_rel_l2_mean"]
            satloss = result[key]["satloss_combined_rel_l2_mean"]
            result[key]["satloss_improvement_percent_vs_base"] = 100.0 * (base - satloss) / max(abs(base), 1.0e-12)
    return result


def configure_plot(font_scale: float) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    size = 15.0 * float(font_scale)
    plt.rcParams.update({"font.size": size, "axes.titlesize": size, "axes.labelsize": size, "xtick.labelsize": size, "ytick.labelsize": size, "legend.fontsize": size})


def make_groups(values: OrderedDict, metric: str, labels: list[tuple[str, str]]) -> list[dict]:
    groups = []
    for key, label in labels:
        item = values.get(key)
        if item is None:
            continue
        groups.append({"label": label, "base_mean": item[f"base_{metric}_rel_l2_mean"], "base_std": item[f"base_{metric}_rel_l2_std"], "satloss_mean": item[f"satloss_{metric}_rel_l2_mean"], "satloss_std": item[f"satloss_{metric}_rel_l2_std"]})
    return groups


def paired_bar_plot(groups: list[dict], path: Path, title: str, ylabel: str, scale: str, font_scale: float, no_std: bool, percentage: bool = False) -> None:
    if not groups:
        return
    import matplotlib.pyplot as plt
    configure_plot(font_scale)
    x = np.arange(len(groups), dtype=np.float64); width = 0.34
    fig, ax = plt.subplots(figsize=(max(12.0, len(groups) * 1.9), 7.2))
    for offset, model in ((-width / 2.0, "BASE"), (width / 2.0, "SATLOSS")):
        means = np.asarray([group[f"{model.lower()}_mean"] for group in groups], dtype=np.float64)
        stds = np.asarray([group[f"{model.lower()}_std"] for group in groups], dtype=np.float64)
        ax.bar(x + offset, means, width=width, yerr=None if no_std else stds, capsize=4 if not no_std else 0, color=MODEL_COLORS[model], edgecolor="#202124", linewidth=0.8, hatch="///" if model == "SATLOSS" else None, label=MODEL_LABELS[model])
        if percentage and model == "SATLOSS":
            for bar, group in zip(ax.containers[-1], groups):
                base = float(group["base_mean"]); satloss = float(group["satloss_mean"])
                improvement = 100.0 * (base - satloss) / max(abs(base), 1.0e-12)
                y = satloss * 1.08 if satloss > 0 else satloss + 0.01
                ax.text(bar.get_x() + bar.get_width() / 2, y, f"{improvement:+.1f}%", ha="center", va="bottom", fontsize=12 * font_scale, fontweight="bold", clip_on=False)
    if scale == "log" and not percentage:
        ax.set_yscale("log")
    elif scale == "log" and percentage:
        ax.set_yscale("symlog", linthresh=1.0)
    ax.set_xticks(x, [group["label"] for group in groups])
    ax.set_ylabel(ylabel); ax.set_title(title, pad=16); ax.grid(axis="y", which="both", alpha=0.22); ax.legend(frameon=True, ncols=2)
    ax.tick_params(axis="both", labelsize=13 * font_scale)
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.18, top=0.87)
    fig.savefig(path, dpi=280, bbox_inches="tight", pad_inches=0.18); plt.close(fig)


def strategy_bar_plot(
    values: OrderedDict,
    metric: str,
    labels: list[tuple[str, str]],
    model_keys: tuple[str, ...],
    path: Path,
    title: str,
    scale: str,
    font_scale: float,
    no_std: bool,
) -> None:
    groups = [(key, label, values.get(key)) for key, label in labels if values.get(key) is not None]
    if not groups:
        return
    import matplotlib.pyplot as plt

    configure_plot(font_scale)
    x = np.arange(len(groups), dtype=np.float64)
    width = min(0.82 / max(len(model_keys), 1), 0.22)
    fig, ax = plt.subplots(figsize=(max(12.0, len(groups) * 2.0), 7.2))
    center = (len(model_keys) - 1) / 2.0
    for index, model in enumerate(model_keys):
        means = np.asarray([item[2].get(f"{model}_{metric}_rel_l2_mean", np.nan) for item in groups], dtype=np.float64)
        stds = np.asarray([item[2].get(f"{model}_{metric}_rel_l2_std", 0.0) for item in groups], dtype=np.float64)
        positions = x + (index - center) * width
        bars = ax.bar(
            positions,
            means,
            width=width * 0.92,
            yerr=None if no_std else stds,
            capsize=3 if not no_std else 0,
            color=STRATEGY_COLORS[model],
            edgecolor="#202124",
            linewidth=0.7,
            hatch="///" if model == "satloss" else None,
            label=MODEL_LABELS_ALL[model],
        )
        for bar, mean in zip(bars, means):
            if np.isfinite(mean):
                bar.set_label(MODEL_LABELS_ALL[model])
    if scale == "log":
        positive = [value for item in groups for model in model_keys for value in [item[2].get(f"{model}_{metric}_rel_l2_mean", np.nan)] if np.isfinite(value) and value > 0.0]
        if positive:
            ax.set_yscale("log")
    ax.set_xticks(x, [item[1] for item in groups], rotation=18, ha="right")
    ax.set_ylabel(f"{metric.capitalize()} normalized relative L2")
    ax.set_title(title, pad=16)
    ax.grid(axis="y", which="both", alpha=0.22)
    handles, labels_out = ax.get_legend_handles_labels()
    unique = OrderedDict((label, handle) for handle, label in zip(handles, labels_out))
    ax.legend(unique.values(), unique.keys(), frameon=True, ncols=min(3, len(unique)))
    ax.tick_params(axis="both", labelsize=13 * font_scale)
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.23, top=0.87)
    fig.savefig(path, dpi=280, bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)


def write_distribution_plot(points: np.ndarray, zero_points: np.ndarray, max_points: np.ndarray, shift: str, path: Path, font_scale: float) -> None:
    import matplotlib.pyplot as plt
    axis = 1 if shift == "sine_y" else 0
    values = np.asarray(points[:, axis], dtype=np.float64); lower, upper = values.min(), values.max(); bins = np.linspace(lower, upper, 51)
    fig, (left, right) = plt.subplots(1, 2, figsize=(14.0, 5.8))
    left.hist(values, bins=bins, density=True, histtype="step", linewidth=2.0, color="#555555", label="full surface")
    left.hist(zero_points[:, axis], bins=bins, density=True, alpha=0.40, color="#2166ac", label="intensity 0")
    left.hist(max_points[:, axis], bins=bins, density=True, alpha=0.45, color="#b2182b", label="intensity 1")
    left.set_xlabel("Physical coordinate"); left.set_ylabel("Probability density"); left.set_title(f"{shift.replace('_', '-')} endpoint distributions"); left.legend()
    normalize = lambda array: np.clip((array[:, axis] - lower) / max(upper - lower, 1.0e-12), 0.0, 1.0)
    right.hist(normalize(max_points), bins=np.linspace(0, 1, 51), density=True, color="#b2182b", alpha=0.65, label="intensity 1")
    right.hist(normalize(zero_points), bins=np.linspace(0, 1, 51), density=True, histtype="step", linewidth=2.0, color="#2166ac", label="intensity 0")
    right.set_xlabel(f"Normalized {'y' if axis == 1 else 'x'} coordinate"); right.set_ylabel("Probability density"); right.set_title("Redistribution relative to zero"); right.legend()
    for axis_obj in (left, right): axis_obj.grid(alpha=0.22); axis_obj.tick_params(labelsize=13 * font_scale)
    fig.suptitle(f"SHIFT-Pump sampling shift: {shift.replace('_', '-')}", fontsize=17 * font_scale); fig.tight_layout(); fig.savefig(path, dpi=280, bbox_inches="tight", pad_inches=0.18); plt.close(fig)


def summary_groups(rows: list[dict], beta_endpoints: tuple[float, float]) -> list[dict]:
    aggregate_values = aggregate([row for row in rows if row["source"] == "original"], "mode")
    requested = [(f"beta_{beta_endpoints[1]:.2f}", "beta=1"), ("sine_x_1.00", "sine-x=1"), ("sine_y_1.00", "sine-y=1")]
    groups = []
    for key, label in requested:
        value = aggregate_values.get(key)
        if value is not None:
            groups.append({"label": label, "base_mean": value["base_combined_rel_l2_mean"], "base_std": value["base_combined_rel_l2_std"], "satloss_mean": value["satloss_combined_rel_l2_mean"], "satloss_std": value["satloss_combined_rel_l2_std"]})
    return groups


def write_summary(groups: list[dict], output_dir: Path) -> None:
    fields = ["shift", "smart_combined_rel_l2", "satloss_combined_rel_l2", "satloss_reduction_vs_smart_percent"]
    with (output_dir / "pump_combined_global_shift_endpoint_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for group in groups:
            base = float(group["base_mean"]); satloss = float(group["satloss_mean"]); writer.writerow({"shift": group["label"], "smart_combined_rel_l2": base, "satloss_combined_rel_l2": satloss, "satloss_reduction_vs_smart_percent": 100 * (base - satloss) / max(abs(base), 1.0e-12)})
    lines = ["# SHIFT-Pump point-cloud shift summary", "", "| Shift | SMART | SATLOSS | SATLOSS reduction vs SMART |", "|---|---:|---:|---:|"]
    for group in groups:
        base = float(group["base_mean"]); satloss = float(group["satloss_mean"]); reduction = 100 * (base - satloss) / max(abs(base), 1.0e-12); lines.append(f"| {group['label']} | {base:.6g} | {satloss:.6g} | {reduction:+.2f}% |")
    (output_dir / "pump_combined_global_shift_endpoint_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.num_runs <= 0 or args.views_per_mode <= 0 or args.model_repeats <= 0:
        raise ValueError("num-runs, views-per-mode, and model-repeats must be positive.")
    if args.input_points <= 0 or args.surface_query_points <= 0 or args.volume_query_points <= 0:
        raise ValueError("Point budgets must be positive.")

    active_shifts = parse_active_shifts(args.active_shifts)
    beta_endpoints = endpoint_values(args.shift_betas)
    modes = mode_records(active_shifts, beta_endpoints)
    active_sources = parse_active_sources(args.active_geometry_sources)
    factors = {int(item.strip()) for item in str(args.geometry_decimation_factors).split(",") if item.strip()}
    if any(factor <= 1 for factor in factors):
        raise ValueError("Geometry decimation factors must be greater than one.")
    active_sources = [source for source in active_sources if source == "original" or int(source.rsplit("div", 1)[1]) in factors]
    study = None
    if args.study_summary is not None:
        study = json.loads(args.study_summary.read_text(encoding="utf-8"))
    if any(source != "original" for source in active_sources) and study is None:
        raise ValueError("Remeshed geometry sources require --study-summary.")

    devices = parse_devices(args.devices)
    base_device = devices[0]
    satloss_device = devices[1] if len(devices) > 1 else devices[0]
    dataset = PumpDataset(
        args.data_root,
        if_test=True,
        geometry_points=0,
        surface_points=args.surface_query_points,
        volume_points=args.volume_query_points,
    )
    available = list(dataset.data)
    if args.run_ids:
        run_ids = [int(item.strip()) for item in str(args.run_ids).split(",") if item.strip()]
        missing = sorted(set(run_ids) - set(available))
        if missing:
            raise ValueError(f"Requested run IDs are not in the Pump test split: {missing}")
    else:
        count = min(args.num_runs, len(available))
        rng = np.random.default_rng(args.seed + 7001)
        run_ids = sorted(int(value) for value in rng.choice(np.asarray(available), size=count, replace=False))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"SHIFT-Pump runs={run_ids}", flush=True)
    print(f"Active sampling shifts={active_shifts}", flush=True)
    print(f"Active geometry sources={active_sources}", flush=True)
    print(f"Devices={base_device},{satloss_device}", flush=True)
    base_model = make_model("pump.yaml", args.base_checkpoint, base_device, args.query_chunk_size)
    satloss_model = make_model("pump_satloss7.yaml", args.satloss_checkpoint, satloss_device, args.query_chunk_size)

    rows: list[dict] = []
    representative: dict[str, np.ndarray] = {}
    representative_source: np.ndarray | None = None
    vtk_run_id = int(args.vtk_run_id) if args.vtk_run_id is not None else int(run_ids[0])
    input_dir = args.output_dir / "input_vtks"
    prediction_dir = args.output_dir / "prediction_vtks"
    input_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir.mkdir(parents=True, exist_ok=True)

    for run_index, run_id in enumerate(run_ids):
        run_dir = args.data_root / f"run_{run_id}"
        original_source = np.ascontiguousarray(np.asarray(np.load(run_dir / "surface_coords.npy", mmap_mode="r"), dtype=np.float32))
        density = dataset._load_density(run_id, original_source).numpy().astype(np.float32, copy=False)
        queries = fixed_queries(run_id, args.data_root, dataset, args.surface_query_points, args.volume_query_points, args.seed)
        original_bounds = np.stack([original_source.min(axis=0), original_source.max(axis=0)])
        for source_name, vtp_path in source_records(study, run_id, active_sources):
            physical_source = original_source if vtp_path is None else load_vtp_points(vtp_path)
            source_bounds = np.stack([physical_source.min(axis=0), physical_source.max(axis=0)])
            tolerance = np.maximum(original_bounds[1] - original_bounds[0], 1.0e-6) * 0.025
            if np.any(source_bounds[0] < original_bounds[0] - tolerance) or np.any(source_bounds[1] > original_bounds[1] + tolerance):
                raise ValueError(f"{source_name} run_{run_id} has an inconsistent source bounding box.")
            source_modes = modes if source_name == "original" else OrderedDict([
                ("aligned_uniform_wor", {"kind": "uniform", "shift": "aligned", "intensity": 0.0, "label": "aligned uniform"})
            ])
            for mode_name, mode in source_modes.items():
                for view_index in range(args.views_per_mode):
                    seed = args.seed + 1000003 * run_id + 10007 * run_index + 101 * view_index + stable_tag(source_name) + stable_tag(mode_name)
                    density_for_source = density if source_name == "original" else None
                    sampling_mode = mode if float(mode.get("intensity", 0.0)) > 1.0e-12 else {"kind": "uniform", "shift": "aligned", "intensity": 0.0}
                    selected_indices = sample_encoder_indices(physical_source, density_for_source, sampling_mode, args.input_points, seed)
                    selected = np.ascontiguousarray(physical_source[selected_indices], dtype=np.float32)
                    geometry = normalize_positions(selected, queries["minimum"], queries["span"])
                    with ThreadPoolExecutor(max_workers=2) as executor:
                        base_future = executor.submit(predict_repeated, base_model, base_device, geometry, queries, args.model_repeats)
                        satloss_future = executor.submit(predict_repeated, satloss_model, satloss_device, geometry, queries, args.model_repeats)
                        base_surface, base_volume = base_future.result()
                        satloss_surface, satloss_volume = satloss_future.result()
                    base_surface_error = rel_l2(queries["surface_y"], base_surface)
                    satloss_surface_error = rel_l2(queries["surface_y"], satloss_surface)
                    base_volume_error = rel_l2(queries["volume_y"], base_volume)
                    satloss_volume_error = rel_l2(queries["volume_y"], satloss_volume)
                    rows.append({
                        "run_id": run_id, "source": source_name, "mode": mode_name, "shift": mode["shift"],
                        "intensity": float(mode["intensity"]), "view": view_index,
                        "base_surface_rel_l2": base_surface_error, "satloss_surface_rel_l2": satloss_surface_error,
                        "base_volume_rel_l2": base_volume_error, "satloss_volume_rel_l2": satloss_volume_error,
                        "base_combined_rel_l2": 0.5 * (base_surface_error + base_volume_error),
                        "satloss_combined_rel_l2": 0.5 * (satloss_surface_error + satloss_volume_error),
                    })
                    if run_id == vtk_run_id and view_index == 0:
                        representative[f"{source_name}:{mode_name}"] = selected.copy()
                        if source_name == "original":
                            representative_source = original_source
                        input_fields = {
                            "sampling_intensity": np.full(selected.shape[0], float(mode["intensity"]), dtype=np.float32),
                            "sample_index": selected_indices.astype(np.float32),
                        }
                        if source_name == "original":
                            input_fields["source_log_density"] = density[selected_indices]
                        write_prediction_vtk(input_dir / f"run_{run_id}_{source_name}_{mode_name}_input.vtk", selected, input_fields)
                        mean_surface = dataset.mean_surf_data.numpy()
                        std_surface = dataset.std_surf_data.numpy()
                        ground_truth = queries["surface_y"] * std_surface + mean_surface
                        base_physical = base_surface * std_surface + mean_surface
                        satloss_physical = satloss_surface * std_surface + mean_surface
                        fields = {}
                        for channel, field_name in enumerate(SURFACE_FIELDS):
                            fields[f"gt_{field_name}"] = ground_truth[:, channel]
                            fields[f"base_{field_name}"] = base_physical[:, channel]
                            fields[f"satloss_{field_name}"] = satloss_physical[:, channel]
                            fields[f"base_error_{field_name}"] = np.abs(base_physical[:, channel] - ground_truth[:, channel])
                            fields[f"satloss_error_{field_name}"] = np.abs(satloss_physical[:, channel] - ground_truth[:, channel])
                        write_prediction_vtk(prediction_dir / f"run_{run_id}_{source_name}_{mode_name}_surface.vtk", queries["surface_q_physical"], fields)
        print(f"[{run_index + 1}/{len(run_ids)}] evaluated run_{run_id}", flush=True)

    if not rows:
        raise RuntimeError("No Pump comparison rows were produced.")
    fields = list(rows[0].keys())
    with (args.output_dir / "pump_sampling_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    sampling_rows = [row for row in rows if row["source"] == "original"]
    source_rows = [row for row in rows if row["mode"] == "aligned_uniform_wor"]
    sampling_aggregate = aggregate(sampling_rows, "mode")
    source_aggregate = aggregate(source_rows, "source")
    write_summary(summary_groups(sampling_rows, beta_endpoints), args.output_dir)
    plot_scales = [item.strip() for item in str(args.plot_scales).split(",") if item.strip()]
    for scale in plot_scales:
        for metric in ("combined", "surface", "volume"):
            if "beta" in active_shifts:
                labels = [(f"beta_{beta_endpoints[0]:.2f}", f"beta={beta_endpoints[0]:.2f}"), (f"beta_{beta_endpoints[1]:.2f}", f"beta={beta_endpoints[1]:.2f}")]
                paired_bar_plot(make_groups(sampling_aggregate, metric, labels), args.output_dir / f"pump_{metric}_global_endpoint_bars_beta_{scale}.png", "SHIFT-Pump beta endpoints", f"{metric.capitalize()} normalized relative L2", scale, args.font_scale, args.no_std)
            for shift in ("sine_x", "sine_y"):
                if shift not in active_shifts:
                    continue
                labels = [(f"{shift}_0.00", f"{shift.replace('_', '-')}=0"), (f"{shift}_1.00", f"{shift.replace('_', '-')}=1")]
                paired_bar_plot(make_groups(sampling_aggregate, metric, labels), args.output_dir / f"pump_{metric}_global_endpoint_bars_{shift}_{scale}.png", f"SHIFT-Pump {shift.replace('_', '-')} endpoints", f"{metric.capitalize()} normalized relative L2", scale, args.font_scale, args.no_std)
            geometry_labels = [(source, SOURCE_LABELS[source]) for source in SOURCE_ORDER if source in source_aggregate]
            paired_bar_plot(make_groups(source_aggregate, metric, geometry_labels), args.output_dir / f"pump_{metric}_global_geometry_sources_bars_{scale}.png", "SHIFT-Pump remeshed encoder sources", f"{metric.capitalize()} normalized relative L2", scale, args.font_scale, args.no_std)

    baseline_rows = {(int(row["run_id"]), int(row["view"])): row for row in sampling_rows if row["mode"] == "aligned_uniform_wor"}
    degradation = []
    for row in sampling_rows:
        if row["mode"] == "aligned_uniform_wor":
            continue
        baseline = baseline_rows.get((int(row["run_id"]), int(row["view"])))
        if baseline is None:
            continue
        item = dict(row)
        for metric in ("surface", "volume", "combined"):
            for model in ("base", "satloss"):
                current = float(row[f"{model}_{metric}_rel_l2"])
                reference = float(baseline[f"{model}_{metric}_rel_l2"])
                item[f"{model}_{metric}_pct_worsening"] = 100.0 * (current - reference) / max(abs(reference), 1.0e-12)
        degradation.append(item)
    for scale in plot_scales:
        for shift in active_shifts:
            mode_name = f"beta_{beta_endpoints[1]:.2f}" if shift == "beta" else f"{shift}_1.00"
            matching = [row for row in degradation if row["mode"] == mode_name]
            if not matching:
                continue
            for metric in ("combined", "surface", "volume"):
                groups = [{
                    "label": mode_name.replace("_", "-"),
                    "base_mean": float(np.mean([row[f"base_{metric}_pct_worsening"] for row in matching])),
                    "base_std": 0.0,
                    "satloss_mean": float(np.mean([row[f"satloss_{metric}_pct_worsening"] for row in matching])),
                    "satloss_std": 0.0,
                }]
                paired_bar_plot(groups, args.output_dir / f"pump_{metric}_global_endpoint_percentage_worsening_{shift}_{scale}.png", f"SHIFT-Pump {shift.replace('_', '-')} worsening vs aligned", "Worsening relative to aligned (%)", scale, args.font_scale, True, percentage=True)

    for shift in active_shifts:
        zero = "beta_0.00" if shift == "beta" else f"{shift}_0.00"
        maximum = f"beta_{beta_endpoints[1]:.2f}" if shift == "beta" else f"{shift}_1.00"
        if representative_source is not None and f"original:{zero}" in representative and f"original:{maximum}" in representative:
            write_distribution_plot(representative_source, representative[f"original:{zero}"], representative[f"original:{maximum}"], shift, args.output_dir / f"pump_{shift}_endpoint_distribution.png", args.font_scale)

    payload = {
        "dataset": "SHIFT-Pump-sample",
        "protocol": "DrivAerML-aligned surface encoder density-shift and separate remeshed geometry-source study",
        "run_ids": run_ids,
        "active_shifts": active_shifts,
        "active_geometry_sources": active_sources,
        "shift_betas": beta_endpoints,
        "input_points": args.input_points,
        "surface_query_points": args.surface_query_points,
        "volume_query_points": args.volume_query_points,
        "base_checkpoint": str(args.base_checkpoint.resolve()),
        "satloss_checkpoint": str(args.satloss_checkpoint.resolve()),
        "metric": "mean of surface and volume normalized global relative L2",
        "note": "Beta/sine shifts apply only to the original preprocessed surface. Remeshed VTPs are aligned uniform encoder-source tests.",
        "aggregate_by_mode_original": sampling_aggregate,
        "aggregate_by_source_aligned": source_aggregate,
    }
    (args.output_dir / "pump_sampling_metrics.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Metrics: {args.output_dir / 'pump_sampling_metrics.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
