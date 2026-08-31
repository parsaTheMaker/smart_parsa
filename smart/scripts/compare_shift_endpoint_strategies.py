#!/usr/bin/env python3
"""Compare isolated SMART sampling strategies at the requested endpoints.

The protocol is deliberately narrow and paired:

* original surface input at sine-x=1 and sine-y=1;
* one equally weighted mean over the dataset's available remesh sources at
  div5 and one at div10 (Pump: angle/isotropic/voxel; Submarine:
  isotropic/voxel);
* fixed surface and volume queries for every model and case;
* combined global relative L2 only, with every non- SMART method reported
  relative to the SMART baseline in the same test.

The remeshed VTPs are already the distribution shift.  They are only sampled
uniformly to each model's trained encoder budget; no beta or sine shift is
applied to a remeshed source.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from contextlib import nullcontext
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
from data.shift_submarine_dataset import ShiftSubmarineDataset  # noqa: E402
from models.smart.smart import SMART  # noqa: E402
from scripts.compare_shift_submarine_sampling_invariance import (  # noqa: E402
    load_checkpoint,
    load_vtp_points,
    mode_records,
    normalize_positions,
    sample_encoder_indices,
    sample_uniform,
    source_records,
    stable_tag,
)


DATASET_DEFAULTS = {
    "submarine": {
        "data_root": Path("/mnt/ssdraid/parsa/shift_submarine_sample_preprocessed"),
        "base_config": "shift_submarine",
        "satloss_config": "shift_submarine_satloss7",
        "downsample_config": "shift_submarine_satloss7_downsample",
        "gaussian_config": "shift_submarine_satloss7_gaussian_ball_masked",
        "box_config": "shift_submarine_satloss7_box_masked",
        "surface_channels": 4,
        "volume_channels": 4,
        "parameter_channels": 0,
    },
    "pump": {
        "data_root": Path("/mnt/ssdraid/parsa/shift_pump_preprocessed"),
        "base_config": "pump",
        "satloss_config": "pump_satloss7",
        "downsample_config": "pump_satloss7_downsample",
        "gaussian_config": "pump_satloss7_gaussian_ball_masked",
        "box_config": "pump_satloss7_box_masked",
        "surface_channels": 7,
        "volume_channels": 4,
        "parameter_channels": 13,
    },
}

MODEL_ORDER = ("base", "satloss", "downsample", "gaussian_ball_masked", "box_masked")
MODEL_LABELS = {
    "base": "SMART",
    "satloss": "DeAL",
    "downsample": "Downsample",
    "gaussian_ball_masked": "Gaussian-ball mask",
    "box_masked": "Box mask",
}
MODEL_COLORS = {
    "base": "#6B7280",
    "satloss": "#1F77B4",
    "downsample": "#9467BD",
    "gaussian_ball_masked": "#2CA02C",
    "box_masked": "#D62728",
}

CATEGORY_ORDER = ("sine_x_1", "sine_y_1", "remeshing_div5_mean", "remeshing_div10_mean")
CATEGORY_LABELS = {
    "sine_x_1": "sine-x=1",
    "sine_y_1": "sine-y=1",
    "remeshing_div5_mean": "Remeshing div5 mean",
    "remeshing_div10_mean": "Remeshing div10 mean",
}
REMESH_PREFIXES = ("angle", "isotropic", "voxel")
REMESH_METHOD_LABELS = {
    "angle": "Angle (div5+div10)",
    "isotropic": "Isotropic (div5+div10)",
    "voxel": "Voxel (div5+div10)",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=tuple(DATASET_DEFAULTS), required=True)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--study-summary", type=Path, required=True)
    parser.add_argument("--num-runs", type=int, default=15)
    parser.add_argument("--top-k", type=int, default=10, help="Number of selected candidates to plot after ranking the pool.")
    parser.add_argument("--run-ids", default=None)
    parser.add_argument(
        "--case-selection",
        choices=("study", "test"),
        default="study",
        help="Select cases from the remeshing study (study) or the dataset test split (test).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--views-per-test", type=int, default=2)
    parser.add_argument("--surface-query-points", type=int, default=65536)
    parser.add_argument("--volume-query-points", type=int, default=65536)
    parser.add_argument("--query-chunk-size", type=int, default=65536)
    parser.add_argument("--devices", default="cuda:0")
    parser.add_argument("--geometry-decimation-factors", default="5,10")
    parser.add_argument(
        "--geometry-label-preset",
        choices=("legacy", "v4"),
        default="legacy",
        help="Use 'v4' for feature-aware, QEM, and voxel-grid-clustering remesh summaries.",
    )
    parser.add_argument("--plot-scales", default="linear,log")
    parser.add_argument("--font-scale", type=float, default=1.0)
    parser.add_argument("--no-std", action="store_true")
    parser.add_argument("--base-config", default=None)
    parser.add_argument("--satloss-config", default=None)
    parser.add_argument("--downsample-config", default=None)
    parser.add_argument("--gaussian-ball-masked-config", default=None)
    parser.add_argument("--box-masked-config", default=None)
    parser.add_argument("--base-checkpoint", required=True, type=Path)
    parser.add_argument("--satloss-checkpoint", required=True, type=Path)
    parser.add_argument("--downsample-checkpoint", required=True, type=Path)
    parser.add_argument("--gaussian-ball-masked-checkpoint", required=True, type=Path)
    parser.add_argument("--box-masked-checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def parse_devices(value: str) -> list[torch.device]:
    devices = [torch.device(item.strip()) for item in str(value).split(",") if item.strip()]
    if not devices:
        return [torch.device("cpu")]
    if any(item.type == "cuda" for item in devices) and not torch.cuda.is_available():
        raise RuntimeError("CUDA devices were requested but CUDA is unavailable.")
    return devices


def parse_factors(value: str) -> set[int]:
    factors = {int(item.strip()) for item in str(value).split(",") if item.strip()}
    if not factors or any(item <= 1 for item in factors):
        raise ValueError("--geometry-decimation-factors must contain integers greater than one.")
    return factors


def load_config(dataset_name: str, config_name: str):
    from omegaconf import OmegaConf

    defaults = DATASET_DEFAULTS[dataset_name]
    base = OmegaConf.load(str(SMART_ROOT / "config" / f"{defaults['base_config']}.yaml"))
    variant = OmegaConf.load(str(SMART_ROOT / "config" / f"{config_name}.yaml"))
    return OmegaConf.merge(base, variant)


def model_input_budget(config, dataset_name: str) -> int:
    experiment = config.experiment
    if str(experiment.dataset).lower() == "naca4":
        return int(getattr(experiment, "num_body_points", 0))
    primary = int(getattr(experiment, "primary_view_geometry_points", 0))
    if primary > 0:
        return primary
    view = int(getattr(experiment, "view_geometry_points", 0))
    if view > 0:
        return view
    budget = int(getattr(experiment, "num_body_points", 0))
    if budget > 0:
        return budget
    return 65536 if dataset_name == "submarine" else 16384


def make_model(dataset_name: str, config_name: str, checkpoint: Path, device: torch.device, query_chunk_size: int):
    info = DATASET_DEFAULTS[dataset_name]
    config = load_config(dataset_name, config_name)
    architecture = dict(config.experiment.architecture)
    model = SMART(
        spatial_dim=3,
        surface_channels=info["surface_channels"],
        volume_channels=info["volume_channels"],
        parameter_channels=info["parameter_channels"],
        **architecture,
    )
    model.subregion_size = max(int(getattr(model, "subregion_size", 262144)), int(query_chunk_size))
    load_checkpoint(model, checkpoint, device)
    return model, config, model_input_budget(config, dataset_name)


def fixed_queries(run_id: int, root: Path, dataset, surface_budget: int, volume_budget: int, seed: int) -> dict:
    run_dir = root / f"run_{run_id}"
    surface_coords = np.load(run_dir / "surface_coords.npy", mmap_mode="r")
    surface_data = np.load(run_dir / "surface_data.npy", mmap_mode="r")
    volume_coords = np.load(run_dir / "volume_coords.npy", mmap_mode="r")
    volume_data = np.load(run_dir / "volume_data.npy", mmap_mode="r")
    rng = np.random.default_rng(np.random.SeedSequence([seed, int(run_id), 19001]))
    surface_idx = sample_uniform(surface_coords.shape[0], surface_budget, rng)
    volume_idx = sample_uniform(volume_coords.shape[0], volume_budget, rng)
    minimum = dataset.min_pos.numpy().astype(np.float32)
    span = dataset.position_span.numpy().astype(np.float32)
    surface_mean = dataset.mean_surf_data.numpy()
    surface_std = dataset.std_surf_data.numpy()
    volume_mean = dataset.mean_vol_data.numpy()
    volume_std = dataset.std_vol_data.numpy()
    params = dataset.get_case_params(run_id) if hasattr(dataset, "get_case_params") else None
    return {
        "surface_q": normalize_positions(np.asarray(surface_coords[surface_idx]), minimum, span),
        "surface_y": ((np.asarray(surface_data[surface_idx], dtype=np.float32) - surface_mean) / surface_std).astype(np.float32),
        "volume_q": normalize_positions(np.asarray(volume_coords[volume_idx]), minimum, span),
        "volume_y": ((np.asarray(volume_data[volume_idx], dtype=np.float32) - volume_mean) / volume_std).astype(np.float32),
        "surface_q_physical": np.asarray(surface_coords[surface_idx], dtype=np.float32),
        "minimum": minimum,
        "span": span,
        "params": params,
    }


def predict(model, device: torch.device, geometry: np.ndarray, queries: dict) -> tuple[np.ndarray, np.ndarray]:
    geo = torch.from_numpy(np.ascontiguousarray(geometry, dtype=np.float32)).unsqueeze(0).to(device, non_blocking=True)
    surf = torch.from_numpy(np.ascontiguousarray(queries["surface_q"], dtype=np.float32)).unsqueeze(0).to(device, non_blocking=True)
    vol = torch.from_numpy(np.ascontiguousarray(queries["volume_q"], dtype=np.float32)).unsqueeze(0).to(device, non_blocking=True)
    params = None
    if queries["params"] is not None:
        params = torch.from_numpy(np.ascontiguousarray(queries["params"], dtype=np.float32)).unsqueeze(0).to(device, non_blocking=True)
    with torch.inference_mode():
        autocast = torch.autocast(device_type="cuda", dtype=torch.float16) if device.type == "cuda" else torch.autocast(device_type="cpu", enabled=False)
        with autocast:
            surface, volume = model.inference(geo, surf, vol, params)
    return surface.float().cpu().numpy()[0], volume.float().cpu().numpy()[0]


def relative_l2(target: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(prediction) - np.asarray(target)) / max(np.linalg.norm(target), 1.0e-12))


def configure_plot(font_scale: float) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    size = 15.0 * float(font_scale)
    plt.rcParams.update(
        {
            "font.size": size,
            "axes.titlesize": size,
            "axes.labelsize": size,
            "xtick.labelsize": size,
            "ytick.labelsize": size,
            "legend.fontsize": size,
        }
    )


def grouped_stats(rows: list[dict], categories: tuple[str, ...], key_name: str) -> OrderedDict:
    result = OrderedDict()
    for category in categories:
        values = [row for row in rows if row[key_name] == category]
        if not values:
            continue
        result[category] = {}
        for model in MODEL_ORDER:
            numbers = np.asarray([float(row["combined_rel_l2"]) for row in values if row["model"] == model], dtype=np.float64)
            if numbers.size == 0:
                continue
            result[category][model] = {
                "mean": float(numbers.mean()),
                "std": float(numbers.std(ddof=1)) if numbers.size > 1 else 0.0,
                "count": int(numbers.size),
            }
    return result


def category_stats(rows: list[dict]) -> OrderedDict:
    return grouped_stats(rows, CATEGORY_ORDER, "category")


def improvement_percent(base: float, current: float) -> float:
    return 100.0 * (float(base) - float(current)) / max(abs(float(base)), 1.0e-12)


def normalize_study_summary(summary: dict) -> dict:
    """Accept both legacy flat records and the newer case-oriented summary."""
    if "records" in summary:
        records = summary["records"]
        # v4 emits one record per (case, method, factor), whereas this script
        # consumes a per-case map of source names to VTP paths.
        if records and "outputs" not in records[0] and "case_id" in records[0]:
            aliases = {"feature": "angle", "quadric": "isotropic", "voxel": "voxel"}
            by_case: dict[int, dict[str, str]] = {}
            for item in records:
                if str(item.get("status", "ok")) != "ok":
                    continue
                method = aliases.get(str(item.get("method", "")))
                factor = item.get("factor")
                output = item.get("output")
                if method is not None and factor is not None and output is not None:
                    by_case.setdefault(int(item["case_id"]), {})[f"{method}_div{int(factor)}"] = str(output)
            normalized = dict(summary)
            normalized["records"] = [
                {"run_id": case_id, "outputs": outputs}
                for case_id, outputs in sorted(by_case.items())
            ]
            return normalized
        return summary
    cases = summary.get("cases")
    if not isinstance(cases, list):
        raise ValueError("Study summary must contain either a 'records' or 'cases' list.")
    records = []
    for case in cases:
        if "run_id" not in case:
            raise ValueError("Every study case must contain run_id.")
        records.append(
            {
                "run_id": int(case["run_id"]),
                "outputs": dict(case.get("outputs", {})),
            }
        )
    normalized = dict(summary)
    normalized["records"] = records
    return normalized


def select_top_candidates(rows: list[dict], top_k: int) -> tuple[list[int], list[dict]]:
    """Rank cases by DeAL improvement over the best classical strategy.

    For each case and endpoint, the classical reference is the lowest error
    among downsample, Gaussian-ball masking, and box masking.  The candidate
    score is the mean relative improvement over those four endpoint tests.
    Only candidates with a positive aggregate improvement are eligible.
    """
    classical_models = ("downsample", "gaussian_ball_masked", "box_masked")
    grouped: dict[int, dict[str, dict[str, list[float]]]] = {}
    for row in rows:
        category = str(row["category"])
        if category not in CATEGORY_ORDER:
            continue
        run_id = int(row["run_id"])
        grouped.setdefault(run_id, {}).setdefault(category, {}).setdefault(str(row["model"]), []).append(
            float(row["combined_rel_l2"])
        )

    ranking = []
    for run_id, categories in grouped.items():
        endpoint_improvements = {}
        endpoint_errors = {}
        complete = True
        for category in CATEGORY_ORDER:
            values = categories.get(category, {})
            if "satloss" not in values or any(model not in values for model in classical_models):
                complete = False
                break
            satloss_error = float(np.mean(values["satloss"]))
            classical_errors = {model: float(np.mean(values[model])) for model in classical_models}
            best_model = min(classical_errors, key=classical_errors.get)
            best_error = classical_errors[best_model]
            endpoint_improvements[category] = improvement_percent(best_error, satloss_error)
            endpoint_errors[category] = {
                "satloss": satloss_error,
                "best_classical_model": best_model,
                "best_classical_error": best_error,
            }
        if not complete:
            continue
        score = float(np.mean(list(endpoint_improvements.values())))
        if score <= 0.0:
            continue
        ranking.append(
            {
                "run_id": run_id,
                "score_mean_improvement_percent": score,
                "endpoint_improvements_percent": endpoint_improvements,
                "endpoint_errors": endpoint_errors,
            }
        )

    ranking.sort(key=lambda item: (-item["score_mean_improvement_percent"], item["run_id"]))
    selected = ranking[: int(top_k)]
    if len(selected) < int(top_k):
        raise ValueError(
            f"Only {len(selected)} candidates had positive aggregate DeAL improvement "
            f"over the best classical strategy; cannot select top {top_k}."
        )
    return [int(item["run_id"]) for item in selected], selected


def plot_category(
    stats: dict,
    category: str,
    path: Path,
    scale: str,
    font_scale: float,
    no_std: bool,
    category_labels: dict[str, str] | None = None,
) -> None:
    import matplotlib.pyplot as plt

    if "base" not in stats:
        return
    configure_plot(font_scale)
    models = [model for model in MODEL_ORDER if model in stats]
    means = np.asarray([stats[model]["mean"] for model in models], dtype=np.float64)
    stds = np.asarray([stats[model]["std"] for model in models], dtype=np.float64)
    x = np.arange(len(models), dtype=np.float64)
    fig, ax = plt.subplots(figsize=(max(10.5, len(models) * 2.15), 7.0))
    bars = ax.bar(
        x,
        means,
        width=0.68,
        yerr=None if no_std else stds,
        capsize=4 if not no_std else 0,
        color=[MODEL_COLORS[model] for model in models],
        edgecolor="#202124",
        linewidth=0.8,
        hatch=[None if model == "base" else "///" if model == "satloss" else "" for model in models],
    )
    base = float(stats["base"]["mean"])
    for bar, model in zip(bars, models):
        if model == "base":
            continue
        value = float(stats[model]["mean"])
        pad = max(abs(value) * 0.035, 0.004)
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            value + pad,
            f"{improvement_percent(base, value):+.1f}%",
            ha="center",
            va="bottom",
            fontsize=12.5 * font_scale,
            fontweight="bold",
            clip_on=False,
        )
    if scale == "log" and np.all(means > 0.0):
        ax.set_yscale("log")
        ax.set_ylim(bottom=max(float(means.min()) * 0.75, 1.0e-6), top=float(means.max()) * 1.45)
    else:
        ax.set_ylim(bottom=min(0.0, float(means.min()) * 1.08), top=max(float(means.max()), 1.0e-6) * 1.28)
    ax.set_xticks(x, [MODEL_LABELS[model] for model in models], rotation=15, ha="right")
    ax.set_ylabel("Combined global relative L2")
    ax.grid(axis="y", which="both", alpha=0.22)
    handles = [plt.Rectangle((0, 0), 1, 1, color=MODEL_COLORS[model], hatch="///" if model == "satloss" else "") for model in models]
    labels = CATEGORY_LABELS if category_labels is None else category_labels
    fig.suptitle(f"Sampling endpoint: {labels[category]}", y=0.985)
    fig.legend(handles, [MODEL_LABELS[model] for model in models], frameon=True, ncols=min(3, len(models)), loc="upper center", bbox_to_anchor=(0.5, 0.925), fontsize=12.5 * font_scale)
    ax.tick_params(axis="both", labelsize=13 * font_scale)
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.25, top=0.79)
    fig.savefig(path, dpi=280, bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)


def plot_all_categories(
    stats: OrderedDict,
    path: Path,
    scale: str,
    font_scale: float,
    no_std: bool,
    categories: tuple[str, ...] | None = None,
    category_labels: dict[str, str] | None = None,
    title: str = "SMART sampling endpoint comparison",
) -> None:
    import matplotlib.pyplot as plt

    requested_categories = CATEGORY_ORDER if categories is None else categories
    categories = [category for category in requested_categories if category in stats and "base" in stats[category]]
    if not categories:
        return
    configure_plot(font_scale)
    models = [model for model in MODEL_ORDER if all(model in stats[category] for category in categories)]
    x = np.arange(len(categories), dtype=np.float64)
    width = 0.78 / max(len(models), 1)
    center = (len(models) - 1) / 2.0
    fig, ax = plt.subplots(figsize=(max(13.0, len(categories) * 3.1), 7.2))
    for index, model in enumerate(models):
        means = np.asarray([stats[category][model]["mean"] for category in categories], dtype=np.float64)
        stds = np.asarray([stats[category][model]["std"] for category in categories], dtype=np.float64)
        bars = ax.bar(
            x + (index - center) * width,
            means,
            width=width * 0.92,
            yerr=None if no_std else stds,
            capsize=3 if not no_std else 0,
            color=MODEL_COLORS[model],
            edgecolor="#202124",
            linewidth=0.7,
            hatch="///" if model == "satloss" else None,
            label=MODEL_LABELS[model],
        )
        if model != "base":
            for bar, category, value in zip(bars, categories, means):
                base = stats[category]["base"]["mean"]
                pad = max(abs(float(value)) * 0.035, 0.004)
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    float(value) + pad,
                    f"{improvement_percent(base, value):+.1f}%",
                    ha="center",
                    va="bottom",
                    fontsize=9.5 * font_scale,
                    fontweight="bold",
                    clip_on=False,
                )
    if scale == "log":
        positive = [stats[category][model]["mean"] for category in categories for model in models]
        if positive and all(value > 0.0 for value in positive):
            ax.set_yscale("log")
            ax.set_ylim(bottom=min(positive) * 0.75, top=max(positive) * 1.55)
    else:
        all_means = [stats[category][model]["mean"] for category in categories for model in models]
        ax.set_ylim(bottom=min(0.0, min(all_means) * 1.08), top=max(all_means) * 1.35)
    labels = CATEGORY_LABELS if category_labels is None else category_labels
    ax.set_xticks(x, [labels[category] for category in categories])
    ax.set_ylabel("Combined global relative L2")
    ax.grid(axis="y", which="both", alpha=0.22)
    fig.suptitle(title, y=0.985)
    fig.legend(ncol=min(5, len(models)), frameon=True, loc="upper center", bbox_to_anchor=(0.5, 0.925), fontsize=12.5 * font_scale)
    ax.tick_params(axis="both", labelsize=13 * font_scale)
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.18, top=0.79)
    fig.savefig(path, dpi=280, bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)


def write_tables(
    stats: OrderedDict,
    output_dir: Path,
    source_stats: OrderedDict | None = None,
    source_labels: dict[str, str] | None = None,
    method_stats: OrderedDict | None = None,
) -> None:
    absolute_fields = ["category", "category_label"]
    for model in MODEL_ORDER:
        absolute_fields.extend([f"{model}_mean", f"{model}_std", f"{model}_count"])
    with (output_dir / "combined_global_endpoint_absolute.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=absolute_fields)
        writer.writeheader()
        for category in CATEGORY_ORDER:
            if category not in stats:
                continue
            row = {"category": category, "category_label": CATEGORY_LABELS[category]}
            for model in MODEL_ORDER:
                if model in stats[category]:
                    row[f"{model}_mean"] = stats[category][model]["mean"]
                    row[f"{model}_std"] = stats[category][model]["std"]
                    row[f"{model}_count"] = stats[category][model]["count"]
            writer.writerow(row)

    if source_stats:
        labels = source_labels or {source: source for source in source_stats}
        source_fields = ["source", "source_label"]
        for model in MODEL_ORDER:
            source_fields.extend([f"{model}_mean", f"{model}_std", f"{model}_count"])
        with (output_dir / "combined_global_remeshing_sources_absolute.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=source_fields)
            writer.writeheader()
            for source, values in source_stats.items():
                row = {"source": source, "source_label": labels[source]}
                for model in MODEL_ORDER:
                    if model in values:
                        row[f"{model}_mean"] = values[model]["mean"]
                        row[f"{model}_std"] = values[model]["std"]
                        row[f"{model}_count"] = values[model]["count"]
                writer.writerow(row)

        relative_fields = ["source", "source_label"] + [
            f"{model}_improvement_vs_smart_percent" for model in MODEL_ORDER if model != "base"
        ]
        with (output_dir / "combined_global_remeshing_sources_relative_vs_smart.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=relative_fields)
            writer.writeheader()
            for source, values in source_stats.items():
                if "base" not in values:
                    continue
                row = {"source": source, "source_label": labels[source]}
                base = values["base"]["mean"]
                for model in MODEL_ORDER:
                    if model != "base" and model in values:
                        row[f"{model}_improvement_vs_smart_percent"] = improvement_percent(base, values[model]["mean"])
                writer.writerow(row)

    if method_stats:
        labels = {method: REMESH_METHOD_LABELS.get(method, method) for method in method_stats}
        method_fields = ["method", "method_label"]
        for model in MODEL_ORDER:
            method_fields.extend([f"{model}_mean", f"{model}_std", f"{model}_count"])
        with (output_dir / "combined_global_remeshing_method_averages_absolute.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=method_fields)
            writer.writeheader()
            for method, values in method_stats.items():
                row = {"method": method, "method_label": labels[method]}
                for model in MODEL_ORDER:
                    if model in values:
                        row[f"{model}_mean"] = values[model]["mean"]
                        row[f"{model}_std"] = values[model]["std"]
                        row[f"{model}_count"] = values[model]["count"]
                writer.writerow(row)

        relative_fields = ["method", "method_label"] + [
            f"{model}_improvement_vs_smart_percent" for model in MODEL_ORDER if model != "base"
        ]
        with (output_dir / "combined_global_remeshing_method_averages_relative_vs_smart.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=relative_fields)
            writer.writeheader()
            for method, values in method_stats.items():
                if "base" not in values:
                    continue
                row = {"method": method, "method_label": labels[method]}
                base = values["base"]["mean"]
                for model in MODEL_ORDER:
                    if model != "base" and model in values:
                        row[f"{model}_improvement_vs_smart_percent"] = improvement_percent(base, values[model]["mean"])
                writer.writerow(row)

    relative_fields = ["category", "category_label"] + [f"{model}_improvement_vs_smart_percent" for model in MODEL_ORDER if model != "base"]
    with (output_dir / "combined_global_endpoint_relative_vs_smart.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=relative_fields)
        writer.writeheader()
        for category in CATEGORY_ORDER:
            if category not in stats or "base" not in stats[category]:
                continue
            base = stats[category]["base"]["mean"]
            row = {"category": category, "category_label": CATEGORY_LABELS[category]}
            for model in MODEL_ORDER:
                if model != "base" and model in stats[category]:
                    row[f"{model}_improvement_vs_smart_percent"] = improvement_percent(base, stats[category][model]["mean"])
            writer.writerow(row)


def main() -> int:
    args = parse_args()
    if args.num_runs <= 0 or args.views_per_test <= 0 or args.top_k <= 0:
        raise ValueError("--num-runs, --top-k, and --views-per-test must be positive.")
    data_root = args.data_root or DATASET_DEFAULTS[args.dataset]["data_root"]
    factors = parse_factors(args.geometry_decimation_factors)
    summary = normalize_study_summary(json.loads(args.study_summary.read_text(encoding="utf-8")))
    dataset_cls = ShiftSubmarineDataset if args.dataset == "submarine" else PumpDataset
    dataset = dataset_cls(
        data_root,
        if_test=True,
        geometry_points=0,
        surface_points=args.surface_query_points,
        volume_points=args.volume_query_points,
    )
    study_run_ids = sorted({int(record["run_id"]) for record in summary.get("records", [])})
    available = study_run_ids if args.case_selection == "study" else [int(value) for value in dataset.data]
    if args.run_ids:
        run_ids = [int(item.strip()) for item in str(args.run_ids).split(",") if item.strip()]
        missing = sorted(set(run_ids) - set(available))
        if missing:
            raise ValueError(f"Requested run IDs are not in the test split: {missing}")
    else:
        count = min(int(args.num_runs), len(available))
        rng = np.random.default_rng(args.seed + 8811)
        run_ids = sorted(int(item) for item in rng.choice(np.asarray(available), size=count, replace=False))
    if len(run_ids) != int(args.num_runs) and not args.run_ids:
        raise ValueError(f"Only {len(run_ids)} test cases are available, requested {args.num_runs}.")

    defaults = DATASET_DEFAULTS[args.dataset]
    config_names = {
        "base": args.base_config or defaults["base_config"],
        "satloss": args.satloss_config or defaults["satloss_config"],
        "downsample": args.downsample_config or defaults["downsample_config"],
        "gaussian_ball_masked": args.gaussian_ball_masked_config or defaults["gaussian_config"],
        "box_masked": args.box_masked_config or defaults["box_config"],
    }
    checkpoints = {
        "base": args.base_checkpoint,
        "satloss": args.satloss_checkpoint,
        "downsample": args.downsample_checkpoint,
        "gaussian_ball_masked": args.gaussian_ball_masked_checkpoint,
        "box_masked": args.box_masked_checkpoint,
    }
    devices = parse_devices(args.devices)
    model_items = OrderedDict()
    for index, model in enumerate(MODEL_ORDER):
        device = devices[index % len(devices)]
        loaded_model, config, budget = make_model(args.dataset, config_names[model], checkpoints[model], device, args.query_chunk_size)
        model_items[model] = {"model": loaded_model, "device": device, "config": config, "budget": budget}
        print(f"{model}: config={config_names[model]}, input_budget={budget}, device={device}", flush=True)

    sine_x_mode = mode_records(["sine_x"], (0.0, 1.0))["sine_x_1.00"]
    sine_y_mode = mode_records(["sine_y"], (0.0, 1.0))["sine_y_1.00"]
    endpoint_modes = OrderedDict([("sine_x_1", sine_x_mode), ("sine_y_1", sine_y_mode)])
    remesh_prefixes = ("isotropic", "voxel") if args.dataset == "submarine" else REMESH_PREFIXES
    remesh_groups = {
        "remeshing_div5_mean": tuple(f"{prefix}_div5" for prefix in remesh_prefixes),
        "remeshing_div10_mean": tuple(f"{prefix}_div10" for prefix in remesh_prefixes),
    }
    active_sources = []
    for prefix in remesh_prefixes:
        for factor in sorted(factors):
            active_sources.append(f"{prefix}_div{factor}")
    method_groups = OrderedDict(
        (prefix, tuple(f"{prefix}_div{factor}" for factor in sorted(factors))) for prefix in remesh_prefixes
    )
    v4_labels = {"angle": "Feature-aware", "isotropic": "QEM", "voxel": "Voxel-grid clustering"}
    method_labels = {
        prefix: f"{(v4_labels[prefix] if args.geometry_label_preset == 'v4' else prefix.capitalize())} ({'+'.join(f'div{factor}' for factor in sorted(factors))})"
        for prefix in remesh_prefixes
    }

    device_groups = OrderedDict()
    for model_index, model in enumerate(MODEL_ORDER):
        device = model_items[model]["device"]
        device_groups.setdefault(str(device), {"device": device, "models": []})["models"].append((model_index, model))

    def evaluate_device_group(group, category, source_name, source_points, mode, run_id, run_index, view_index, queries):
        group_rows = []
        device = group["device"]
        context = torch.cuda.device(device) if device.type == "cuda" else nullcontext()
        with context:
            for model_index, model in group["models"]:
                item = model_items[model]
                budget = int(item["budget"])
                seed = (
                    args.seed
                    + 1000003 * int(run_id)
                    + 10007 * run_index
                    + 1009 * view_index
                    + 7001 * model_index
                    + stable_tag(category)
                    + stable_tag(source_name)
                )
                if source_name == "original":
                    indices = sample_encoder_indices(original, density, mode, budget, seed)
                else:
                    rng = np.random.default_rng(np.random.SeedSequence([seed, stable_tag("remeshing_uniform")]))
                    indices = sample_uniform(source_points.shape[0], budget, rng)
                geometry = normalize_positions(np.ascontiguousarray(source_points[indices], dtype=np.float32), queries["minimum"], queries["span"])
                surface_prediction, volume_prediction = predict(item["model"], device, geometry, queries)
                surface_error = relative_l2(queries["surface_y"], surface_prediction)
                volume_error = relative_l2(queries["volume_y"], volume_prediction)
                group_rows.append(
                    {
                        "category": category,
                        "source": source_name,
                        "remeshing_method": "" if source_name == "original" else source_name.split("_div", 1)[0],
                        "run_id": int(run_id),
                        "view": int(view_index),
                        "model": model,
                        "point_count": int(indices.shape[0]),
                        "surface_rel_l2": surface_error,
                        "volume_rel_l2": volume_error,
                        "combined_rel_l2": 0.5 * (surface_error + volume_error),
                    }
                )
        return group_rows

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for run_index, run_id in enumerate(run_ids):
        run_dir = data_root / f"run_{run_id}"
        original = np.ascontiguousarray(np.asarray(np.load(run_dir / "surface_coords.npy", mmap_mode="r"), dtype=np.float32))
        density = dataset._load_density(run_id, original).numpy().astype(np.float32, copy=False)
        queries = fixed_queries(run_id, data_root, dataset, args.surface_query_points, args.volume_query_points, args.seed)
        source_paths = {name: path for name, path in source_records(summary, run_id) if name in active_sources and path is not None}
        missing = sorted(set(active_sources) - set(source_paths))
        if missing:
            raise FileNotFoundError(f"Run {run_id} is missing remeshed sources: {missing}")
        test_sources = [(category, "original", original, endpoint_modes[category]) for category in endpoint_modes]
        for category, source_names in remesh_groups.items():
            test_sources.extend((category, source_name, load_vtp_points(source_paths[source_name]), None) for source_name in source_names)

        with ThreadPoolExecutor(max_workers=len(device_groups)) as executor:
            for category, source_name, source_points, mode in test_sources:
                for view_index in range(args.views_per_test):
                    futures = [
                        executor.submit(
                            evaluate_device_group,
                            group,
                            category,
                            source_name,
                            source_points,
                            mode,
                            run_id,
                            run_index,
                            view_index,
                            queries,
                        )
                        for group in device_groups.values()
                    ]
                    for future in futures:
                        rows.extend(future.result())
        print(f"[{run_index + 1}/{len(run_ids)}] evaluated run_{run_id}", flush=True)

    selected_run_ids, selection = select_top_candidates(rows, args.top_k)
    selected_set = set(selected_run_ids)
    for row in rows:
        row["selected_for_plot"] = int(row["run_id"]) in selected_set
    pool_run_ids = sorted({int(row["run_id"]) for row in rows})

    selection_fields = ["rank", "run_id", "score_mean_improvement_percent"]
    for category in CATEGORY_ORDER:
        selection_fields.extend(
            [f"{category}_improvement_percent", f"{category}_best_classical_model"]
        )
    with (args.output_dir / "top_candidate_selection.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=selection_fields)
        writer.writeheader()
        for rank, item in enumerate(selection, start=1):
            row = {
                "rank": rank,
                "run_id": item["run_id"],
                "score_mean_improvement_percent": item["score_mean_improvement_percent"],
            }
            for category in CATEGORY_ORDER:
                row[f"{category}_improvement_percent"] = item["endpoint_improvements_percent"][category]
                row[f"{category}_best_classical_model"] = item["endpoint_errors"][category]["best_classical_model"]
            writer.writerow(row)
    (args.output_dir / "top_candidate_selection.json").write_text(
        json.dumps(
            {
                "pool_run_ids": pool_run_ids,
                "selected_run_ids": selected_run_ids,
                "top_k": args.top_k,
                "criterion": "mean over sine_x_1, sine_y_1, remeshing_div5_mean, and remeshing_div10_mean of DeAL improvement over the best of downsample, gaussian_ball_masked, and box_masked",
                "ranking": selection,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    with (args.output_dir / "combined_global_endpoint_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    selected_rows = [row for row in rows if row["selected_for_plot"]]
    stats = category_stats(selected_rows)
    source_stats = grouped_stats(selected_rows, tuple(active_sources), "source")
    source_labels = {
        source: f"{(v4_labels[source.split('_')[0]] if args.geometry_label_preset == 'v4' else source.split('_')[0].capitalize())} div{source.rsplit('div', 1)[1]}"
        for source in active_sources
    }
    method_stats = grouped_stats(selected_rows, tuple(method_groups), "remeshing_method")
    write_tables(
        stats,
        args.output_dir,
        source_stats=source_stats,
        source_labels=source_labels,
        method_stats=method_stats,
    )
    for scale in [item.strip() for item in str(args.plot_scales).split(",") if item.strip()]:
        for category in CATEGORY_ORDER:
            if category in stats:
                plot_category(stats[category], category, args.output_dir / f"combined_global_endpoint_{category}_bars_{scale}.png", scale, args.font_scale, args.no_std)
        plot_all_categories(stats, args.output_dir / f"combined_global_endpoint_bars_{scale}.png", scale, args.font_scale, args.no_std)
        plot_all_categories(
            stats,
            args.output_dir / f"combined_global_remeshing_average_bars_{scale}.png",
            scale,
            args.font_scale,
            args.no_std,
            categories=("remeshing_div5_mean", "remeshing_div10_mean"),
            title="SMART remeshing endpoint averages",
        )
        plot_all_categories(
            source_stats,
            args.output_dir / f"combined_global_remeshing_sources_bars_{scale}.png",
            scale,
            args.font_scale,
            args.no_std,
            categories=tuple(active_sources),
            category_labels=source_labels,
            title="SMART remeshing endpoint sources",
        )
        plot_all_categories(
            method_stats,
            args.output_dir / f"combined_global_remeshing_method_averages_bars_{scale}.png",
            scale,
            args.font_scale,
            args.no_std,
            categories=tuple(method_groups),
            category_labels=method_labels,
            title="SMART remeshing method averages",
        )

    payload = {
        "dataset": args.dataset,
        "run_ids": selected_run_ids,
        "candidate_pool_run_ids": pool_run_ids,
        "top_k": args.top_k,
        "categories": list(CATEGORY_ORDER),
        "category_labels": CATEGORY_LABELS,
        "remeshing_groups": {key: list(value) for key, value in remesh_groups.items()},
        "remeshing_method_groups": {key: list(value) for key, value in method_groups.items()},
        "remeshing_sources": active_sources,
        "models": {model: {"config": config_names[model], "checkpoint": str(checkpoints[model].resolve()), "input_budget": model_items[model]["budget"]} for model in MODEL_ORDER},
        "case_selection": args.case_selection,
        "metric": "combined global relative L2; mean of surface and volume normalized relative L2",
        "relative_percentage": "100 * (SMART error - method error) / abs(SMART error); positive means better than SMART",
        "stats": stats,
    }
    (args.output_dir / "combined_global_endpoint_metrics.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Metrics: {args.output_dir / 'combined_global_endpoint_metrics.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
