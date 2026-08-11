#!/usr/bin/env python3
"""Compare SMART SATLOSS strategy checkpoints on isolated 3-D datasets.

The target queries are fixed per case. Only the surface encoder cloud changes:
the aligned uniform source, a one-quarter uniform downsample, a Gaussian-ball
mask, or a box mask. Every checkpoint is evaluated on every source so the
strategy comparison is paired rather than giving each model an easier input.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import OrderedDict
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
    normalize_positions,
    sample_uniform,
    stable_tag,
    write_prediction_vtk,
)
from utils.strategy_sampling import sample_strategy  # noqa: E402


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
        "surface_fields": ("pressure", "wall_shear_x", "wall_shear_y", "wall_shear_z"),
        "volume_fields": ("pressure", "velocity_x", "velocity_y", "velocity_z"),
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
        "surface_fields": PumpDataset.SURFACE_FIELDS,
        "volume_fields": PumpDataset.VOLUME_FIELDS,
    },
}
MODEL_ORDER = ("base", "satloss", "downsample", "gaussian_ball_masked", "box_masked")
MODEL_LABELS = {
    "base": "SMART",
    "satloss": "SATLOSS",
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
STRATEGY_ORDER = ("aligned_uniform", "downsample", "gaussian_ball_masked", "box_masked")
STRATEGY_LABELS = {
    "aligned_uniform": "Aligned uniform",
    "downsample": "Downsample",
    "gaussian_ball_masked": "Gaussian-ball mask",
    "box_masked": "Box mask",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=tuple(DATASET_DEFAULTS), required=True)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--num-runs", type=int, default=20)
    parser.add_argument("--run-ids", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--views-per-strategy", type=int, default=2)
    parser.add_argument("--input-points", type=int, default=None)
    parser.add_argument("--surface-query-points", type=int, default=65536)
    parser.add_argument("--volume-query-points", type=int, default=65536)
    parser.add_argument("--query-chunk-size", type=int, default=65536)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--plot-scales", default="linear,log")
    parser.add_argument("--font-scale", type=float, default=1.0)
    parser.add_argument("--no-std", action="store_true")
    parser.add_argument("--vtk-run-id", type=int, default=None)
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
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_config(dataset_name: str, config_name: str):
    from omegaconf import OmegaConf

    defaults = DATASET_DEFAULTS[dataset_name]
    base = OmegaConf.load(str(SMART_ROOT / "config" / f"{defaults['base_config']}.yaml"))
    variant = OmegaConf.load(str(SMART_ROOT / "config" / f"{config_name}.yaml"))
    return OmegaConf.merge(base, variant)


def make_model(dataset_name: str, config_name: str, checkpoint: Path, device: torch.device, query_chunk_size: int):
    cfg = load_config(dataset_name, config_name)
    info = DATASET_DEFAULTS[dataset_name]
    architecture = dict(cfg.experiment.architecture)
    model = SMART(
        spatial_dim=3,
        surface_channels=info["surface_channels"],
        volume_channels=info["volume_channels"],
        parameter_channels=info["parameter_channels"],
        **architecture,
    )
    model.subregion_size = max(int(getattr(model, "subregion_size", 262144)), int(query_chunk_size))
    load_checkpoint(model, checkpoint, device)
    return model, cfg


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
    queries = {
        "surface_q": normalize_positions(np.asarray(surface_coords[surface_idx]), minimum, span),
        "surface_y": ((np.asarray(surface_data[surface_idx], dtype=np.float32) - surface_mean) / surface_std).astype(np.float32),
        "volume_q": normalize_positions(np.asarray(volume_coords[volume_idx]), minimum, span),
        "volume_y": ((np.asarray(volume_data[volume_idx], dtype=np.float32) - volume_mean) / volume_std).astype(np.float32),
        "surface_q_physical": np.asarray(surface_coords[surface_idx], dtype=np.float32),
        "minimum": minimum,
        "span": span,
    }
    if hasattr(dataset, "get_case_params"):
        queries["params"] = dataset.get_case_params(run_id)
    else:
        queries["params"] = None
    return queries


def predict(model, device: torch.device, geometry: np.ndarray, queries: dict) -> tuple[np.ndarray, np.ndarray]:
    geo = torch.from_numpy(np.ascontiguousarray(geometry)).unsqueeze(0).to(device, non_blocking=True)
    surf = torch.from_numpy(queries["surface_q"]).unsqueeze(0).to(device, non_blocking=True)
    vol = torch.from_numpy(queries["volume_q"]).unsqueeze(0).to(device, non_blocking=True)
    params = None
    if queries["params"] is not None:
        params = torch.from_numpy(queries["params"]).unsqueeze(0).to(device, non_blocking=True)
    with torch.inference_mode():
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            out_surface, out_volume = model.inference(geo, surf, vol, params)
    return out_surface.float().cpu().numpy()[0], out_volume.float().cpu().numpy()[0]


def relative_l2(target: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(prediction) - np.asarray(target)) / max(np.linalg.norm(target), 1.0e-12))


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


def aggregate(rows: list[dict], model_keys: tuple[str, ...], metric: str) -> OrderedDict:
    grouped = OrderedDict()
    for row in rows:
        grouped.setdefault(str(row["strategy"]), []).append(row)
    result = OrderedDict()
    for strategy, values in grouped.items():
        result[strategy] = {}
        for model in model_keys:
            numbers = np.asarray([float(row[f"{model}_{metric}_rel_l2"]) for row in values], dtype=np.float64)
            result[strategy][f"{model}_mean"] = float(numbers.mean())
            result[strategy][f"{model}_std"] = float(numbers.std(ddof=1)) if numbers.size > 1 else 0.0
    return result


def plot_bars(values: OrderedDict, model_keys: tuple[str, ...], path: Path, title: str, ylabel: str, scale: str, font_scale: float, no_std: bool) -> None:
    import matplotlib.pyplot as plt

    configure_plot(font_scale)
    groups = [(strategy, values[strategy]) for strategy in STRATEGY_ORDER if strategy in values]
    if not groups:
        return
    x = np.arange(len(groups), dtype=np.float64)
    width = min(0.82 / len(model_keys), 0.20)
    fig, ax = plt.subplots(figsize=(max(12.0, len(groups) * 2.0), 7.2))
    center = (len(model_keys) - 1) / 2.0
    for index, model in enumerate(model_keys):
        means = np.asarray([item[1][f"{model}_mean"] for item in groups], dtype=np.float64)
        stds = np.asarray([item[1][f"{model}_std"] for item in groups], dtype=np.float64)
        ax.bar(
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
    if scale == "log":
        ax.set_yscale("log")
    ax.set_xticks(x, [STRATEGY_LABELS[item[0]] for item in groups], rotation=18, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=16)
    ax.grid(axis="y", which="both", alpha=0.22)
    ax.legend(frameon=True, ncols=min(3, len(model_keys)))
    ax.tick_params(axis="both", labelsize=13 * font_scale)
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.23, top=0.87)
    fig.savefig(path, dpi=280, bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)


def write_relative_table(rows: list[dict], model_keys: tuple[str, ...], output_dir: Path) -> None:
    fields = ["strategy"] + [f"{model}_improvement_vs_smart_percent" for model in model_keys if model != "base"]
    with (output_dir / "strategy_relative_improvement_vs_smart.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for strategy in STRATEGY_ORDER:
            subset = [row for row in rows if row["strategy"] == strategy]
            if not subset:
                continue
            base = float(np.mean([row["base_combined_rel_l2"] for row in subset]))
            values = {"strategy": strategy}
            for model in model_keys:
                if model == "base":
                    continue
                current = float(np.mean([row[f"{model}_combined_rel_l2"] for row in subset]))
                values[f"{model}_improvement_vs_smart_percent"] = 100.0 * (base - current) / max(abs(base), 1.0e-12)
            writer.writerow(values)


def main() -> int:
    args = parse_args()
    if args.num_runs <= 0 or args.views_per_strategy <= 0:
        raise ValueError("num-runs and views-per-strategy must be positive.")
    info = DATASET_DEFAULTS[args.dataset]
    data_root = args.data_root or info["data_root"]
    input_points = args.input_points or (65536 if args.dataset == "submarine" else 16384)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")

    config_names = {
        "base": args.base_config or info["base_config"],
        "satloss": args.satloss_config or info["satloss_config"],
        "downsample": args.downsample_config or info["downsample_config"],
        "gaussian_ball_masked": args.gaussian_ball_masked_config or info["gaussian_config"],
        "box_masked": args.box_masked_config or info["box_config"],
    }
    checkpoints = {
        "base": args.base_checkpoint,
        "satloss": args.satloss_checkpoint,
        "downsample": args.downsample_checkpoint,
        "gaussian_ball_masked": args.gaussian_ball_masked_checkpoint,
        "box_masked": args.box_masked_checkpoint,
    }
    model_keys = MODEL_ORDER
    model_items = {}
    for model_key in model_keys:
        model_items[model_key] = make_model(args.dataset, config_names[model_key], checkpoints[model_key], device, args.query_chunk_size)

    dataset_cls = ShiftSubmarineDataset if args.dataset == "submarine" else PumpDataset
    dataset = dataset_cls(data_root, if_test=True, geometry_points=0, surface_points=args.surface_query_points, volume_points=args.volume_query_points)
    available = list(dataset.data)
    if args.run_ids:
        run_ids = [int(item.strip()) for item in str(args.run_ids).split(",") if item.strip()]
        missing = sorted(set(run_ids) - set(available))
        if missing:
            raise ValueError(f"Requested run IDs are not in the test split: {missing}")
    else:
        count = min(args.num_runs, len(available))
        rng = np.random.default_rng(args.seed + 8811)
        run_ids = sorted(int(item) for item in rng.choice(np.asarray(available), size=count, replace=False))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    input_dir = args.output_dir / "input_vtks"
    prediction_dir = args.output_dir / "prediction_vtks"
    input_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir.mkdir(parents=True, exist_ok=True)
    vtk_run_id = int(args.vtk_run_id) if args.vtk_run_id is not None else int(run_ids[0])
    rows = []

    for run_index, run_id in enumerate(run_ids):
        source = np.ascontiguousarray(np.asarray(np.load(data_root / f"run_{run_id}" / "surface_coords.npy", mmap_mode="r"), dtype=np.float32))
        queries = fixed_queries(run_id, data_root, dataset, args.surface_query_points, args.volume_query_points, args.seed)
        for strategy in STRATEGY_ORDER:
            for view_index in range(args.views_per_strategy):
                seed = args.seed + 1000003 * run_id + 10007 * run_index + 1009 * view_index + stable_tag(strategy)
                rng = np.random.default_rng(np.random.SeedSequence([seed]))
                if strategy == "aligned_uniform":
                    info_sample = {"strategy": strategy, "kept_idx": sample_uniform(source.shape[0], input_points, rng), "base_idx": None, "removed_mask": None, "center_point": None}
                    selected = source[info_sample["kept_idx"]]
                else:
                    downsample_budget = max(1, input_points // 4) if strategy == "downsample" else input_points
                    info_sample = sample_strategy(
                        source,
                        strategy,
                        input_points,
                        rng,
                        downsample_budget=downsample_budget,
                        gaussian_std_fraction=0.05,
                        gaussian_probability_at_1sigma=0.33,
                        gaussian_min_survivors=max(1, input_points // 8),
                    )
                    selected = source[info_sample["kept_idx"]]
                selected = np.ascontiguousarray(selected, dtype=np.float32)
                geometry = normalize_positions(selected, queries["minimum"], queries["span"])
                predictions = {}
                for model_key in model_keys:
                    predictions[model_key] = predict(model_items[model_key][0], device, geometry, queries)
                row = {"run_id": run_id, "strategy": strategy, "view": view_index, "point_count": int(selected.shape[0])}
                for model_key, (surface_prediction, volume_prediction) in predictions.items():
                    surface_error = relative_l2(queries["surface_y"], surface_prediction)
                    volume_error = relative_l2(queries["volume_y"], volume_prediction)
                    row[f"{model_key}_surface_rel_l2"] = surface_error
                    row[f"{model_key}_volume_rel_l2"] = volume_error
                    row[f"{model_key}_combined_rel_l2"] = 0.5 * (surface_error + volume_error)
                rows.append(row)

                if run_id == vtk_run_id and view_index == 0:
                    input_fields = {"kept_flag": np.ones(selected.shape[0], dtype=np.float32)}
                    write_prediction_vtk(input_dir / f"run_{run_id}_{strategy}_kept.vtk", selected, input_fields)
                    base_idx = info_sample.get("base_idx")
                    if base_idx is not None:
                        candidate = source[np.asarray(base_idx, dtype=np.int64)]
                        removed = np.asarray(info_sample["removed_mask"], dtype=bool)
                        candidate_fields = {"removed_flag": removed.astype(np.float32), "kept_flag": (~removed).astype(np.float32)}
                        distance = info_sample.get("distance_to_center")
                        if distance is not None:
                            candidate_fields["distance_to_center"] = np.asarray(distance, dtype=np.float32)
                        if info_sample.get("remove_probability") is not None:
                            candidate_fields["remove_probability"] = np.asarray(info_sample["remove_probability"], dtype=np.float32)
                        center_rel = info_sample.get("center_rel")
                        if center_rel is not None:
                            center_flag = np.zeros(candidate.shape[0], dtype=np.float32)
                            center_flag[int(center_rel)] = 1.0
                            candidate_fields["center_flag"] = center_flag
                        write_prediction_vtk(input_dir / f"run_{run_id}_{strategy}_with_removed.vtk", candidate, candidate_fields)

                    mean_surface = dataset.mean_surf_data.numpy()
                    std_surface = dataset.std_surf_data.numpy()
                    ground_truth = queries["surface_y"] * std_surface + mean_surface
                    fields = {}
                    for model_key, (surface_prediction, _volume_prediction) in predictions.items():
                        physical_prediction = surface_prediction * std_surface + mean_surface
                        for channel, field_name in enumerate(info["surface_fields"]):
                            fields[f"{model_key}_{field_name}"] = physical_prediction[:, channel]
                            fields[f"{model_key}_error_{field_name}"] = np.abs(physical_prediction[:, channel] - ground_truth[:, channel])
                    for channel, field_name in enumerate(info["surface_fields"]):
                        fields[f"gt_{field_name}"] = ground_truth[:, channel]
                    write_prediction_vtk(prediction_dir / f"run_{run_id}_{strategy}_surface.vtk", queries["surface_q_physical"], fields)
        print(f"[{run_index + 1}/{len(run_ids)}] evaluated run_{run_id}", flush=True)

    fieldnames = list(rows[0].keys())
    with (args.output_dir / "strategy_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    for metric in ("combined", "surface", "volume"):
        values = aggregate(rows, model_keys, metric)
        for scale in [item.strip() for item in str(args.plot_scales).split(",") if item.strip()]:
            plot_bars(values, model_keys, args.output_dir / f"{metric}_strategy_bars_{scale}.png", f"SHIFT-{args.dataset.capitalize()} SATLOSS strategy comparison", f"{metric.capitalize()} normalized relative L2", scale, args.font_scale, args.no_std)
    write_relative_table(rows, model_keys, args.output_dir)
    payload = {
        "dataset": args.dataset,
        "run_ids": run_ids,
        "input_points": input_points,
        "surface_query_points": args.surface_query_points,
        "volume_query_points": args.volume_query_points,
        "models": {key: {"config": config_names[key], "checkpoint": str(checkpoints[key].resolve())} for key in model_keys},
        "metric": "global relative L2; combined is the mean of surface and volume errors",
        "rows": rows,
    }
    (args.output_dir / "strategy_metrics.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Metrics: {args.output_dir / 'strategy_metrics.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
