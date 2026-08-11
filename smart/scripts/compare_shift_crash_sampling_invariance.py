#!/usr/bin/env python3
"""Compare SHIFT-Crash SMART and SATLOSS7 under remeshed encoder inputs.

The SHIFT-Crash task is displacement-only: the reference surface/node cloud is
both the geometry input and the query domain.  This comparator therefore keeps
the original query nodes and terminal displacements fixed and changes only the
encoder point cloud.  Remeshed points receive static features, part IDs, and
rail masks by nearest-neighbor transfer from the original reference nodes.

This is intentionally a companion script, not a modification of the
DrivAerML comparator.  It uses the rebuilt SHIFT-Crash configs and preserves
their 65,536-point training budget for both the base and SATLOSS7 checkpoints.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import zlib
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf

SCRIPT_DIR = Path(__file__).resolve().parent
SMART_ROOT = SCRIPT_DIR.parent
if str(SMART_ROOT) not in sys.path:
    sys.path.insert(0, str(SMART_ROOT))

from data.shift_crash_dataset import ShiftCrashDataset  # noqa: E402
from models.shift_crash_smart import ShiftCrashSMART  # noqa: E402


SOURCE_ORDER = (
    "aligned_uniform",
    "angle_div5",
    "angle_div10",
    "voxel_div5",
    "voxel_div10",
    "isotropic_div5",
    "isotropic_div10",
)
SOURCE_LABELS = {
    "aligned_uniform": "Original uniform",
    "angle_div5": "Angle div5",
    "angle_div10": "Angle div10",
    "voxel_div5": "Voxel div5",
    "voxel_div10": "Voxel div10",
    "isotropic_div5": "Isotropic div5",
    "isotropic_div10": "Isotropic div10",
}
METHOD_COLORS = {
    "aligned_uniform": "#4C78A8",
    "angle_div5": "#F58518",
    "angle_div10": "#E45756",
    "voxel_div5": "#54A24B",
    "voxel_div10": "#2CA02C",
    "isotropic_div5": "#B279A2",
    "isotropic_div10": "#9467BD",
}
MODEL_LABELS = {"BASE": "Base", "SATLOSS7": "SATLOSS7"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--study-summary",
        type=Path,
        default=Path("/home/parsa/smart_parsa/results/shift_crash_sampling_study/shift_crash_sampling_study_summary.json"),
    )
    parser.add_argument("--data-root", type=Path, default=Path("/mnt/ssdraid/parsa/shift_crash_preprocessed"))
    parser.add_argument("--split", choices=("train", "validation"), default="validation")
    parser.add_argument("--base-config", default="shift_crash_rebuilt")
    parser.add_argument("--satloss7-config", default="shift_crash_rebuilt_satloss7")
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--satloss7-checkpoint", required=True)
    parser.add_argument("--num-runs", type=int, default=15)
    parser.add_argument("--case-ids", default=None, help="Optional comma-separated case IDs overriding the study summary.")
    parser.add_argument("--query-points", type=int, default=65536)
    parser.add_argument("--input-points", type=int, default=65536)
    parser.add_argument("--model-repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--devices", default="cuda:0,cuda:1")
    parser.add_argument("--query-chunk-size", type=int, default=65536)
    parser.add_argument("--mapping-workers", type=int, default=-1)
    parser.add_argument("--plot-font-scale", type=float, default=1.0)
    parser.add_argument("--output-dir", type=Path, default=Path("/home/parsa/smart_parsa/results/shift_crash_sampling_invariance"))
    parser.add_argument("--overwrite-mapping-cache", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_cfg(name: str, stack: tuple[str, ...] = ()):
    if name in stack:
        raise ValueError(f"Circular config defaults: {' -> '.join((*stack, name))}")
    path = SMART_ROOT / "config" / f"{name}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"Config not found: {path}")
    root = OmegaConf.load(path)
    merged = OmegaConf.create()
    for default in root.get("defaults", ()):
        if not isinstance(default, str) or default == "_self_" or default.startswith("override "):
            continue
        merged = OmegaConf.merge(merged, load_cfg(default.rsplit("/", 1)[-1], (*stack, name)))
    return OmegaConf.merge(merged, root.get("experiment", OmegaConf.create()))


def parse_devices(text: str) -> list[torch.device]:
    devices = []
    for item in str(text).split(","):
        item = item.strip()
        if not item:
            continue
        if item.isdigit():
            item = f"cuda:{item}"
        devices.append(torch.device(item))
    if not devices:
        raise ValueError("--devices must contain at least one device.")
    if any(device.type == "cuda" for device in devices) and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    for device in devices:
        if device.type == "cuda" and device.index is not None and device.index >= torch.cuda.device_count():
            raise ValueError(f"Requested {device}, but only {torch.cuda.device_count()} visible CUDA devices exist.")
    return devices


def load_state_dict(path: Path, device: torch.device) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(path, map_location=device)
    state = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
    if not isinstance(state, dict):
        raise TypeError(f"Unsupported checkpoint format: {path}")
    if state and all(str(key).startswith("module.") for key in state):
        state = {str(key)[7:]: value for key, value in state.items()}
    return state


def build_model(config, checkpoint: Path, device: torch.device, query_chunk_size: int) -> ShiftCrashSMART:
    architecture = OmegaConf.to_container(config.architecture, resolve=True)
    indices = tuple(int(index) for index in getattr(config, "conditioning_parameter_indices", (0, 1, 2, 3, 4, 5)))
    model_kwargs = {
        "spatial_dim": 3,
        "surface_channels": 3,
        "volume_channels": 1,
        "parameter_channels": len(indices),
        "conditioning_input_channels": int(getattr(config, "conditioning_input_channels", 6)),
        "conditioning_parameter_indices": indices,
        "geometry_feature_channels": int(getattr(config, "geometry_feature_channels", 8)),
        "query_feature_channels": int(getattr(config, "query_feature_channels", 8)),
        "part_embedding_size": int(getattr(config, "part_embedding_size", 906)),
        "part_embedding_dim": int(getattr(config, "part_embedding_dim", 16)),
        **architecture,
    }
    model = ShiftCrashSMART(**model_kwargs).to(device)
    if hasattr(model, "configure_conditioning"):
        model.configure_conditioning(
            mode=str(getattr(config, "conditioning_mode", "token_only")),
            residual_scale=float(getattr(config, "conditioning_residual_scale", 0.0)),
            shift_scale=float(getattr(config, "conditioning_shift_scale", 0.0)),
        )
    model.load_state_dict(load_state_dict(checkpoint, device), strict=True)
    model.eval()
    if hasattr(model, "subregion_size"):
        model.subregion_size = max(int(model.subregion_size), int(query_chunk_size))
    return model


def read_vtp_points(path: Path) -> np.ndarray:
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
        raise ValueError(f"Invalid point array in {path}: {points.shape}")
    return np.ascontiguousarray(points)


def normalize_positions(points: np.ndarray, min_position: np.ndarray, position_span: np.ndarray) -> np.ndarray:
    return ((points.astype(np.float32, copy=False) - min_position) / position_span).astype(np.float32, copy=False)


def sample_indices(num_points: int, budget: int, rng: np.random.Generator) -> np.ndarray:
    if budget <= 0 or num_points <= budget:
        return np.arange(num_points, dtype=np.int64)
    return rng.choice(num_points, size=budget, replace=False).astype(np.int64, copy=False)


def case_payload(dataset: ShiftCrashDataset, case_id: str, query_points: int, seed: int) -> dict[str, np.ndarray]:
    data = dataset._case_array(case_id)
    raw_coordinates = np.asarray(data[:, :3], dtype=np.float32)
    normalized = normalize_positions(raw_coordinates, dataset.min_position, dataset.position_span)
    static_features, part_ids, rail_mask = dataset._case_static_inputs(case_id, int(data.shape[0]))
    static_features = np.asarray(static_features, dtype=np.float32)
    part_ids = np.asarray(part_ids, dtype=np.int64)
    rail_mask = np.asarray(rail_mask, dtype=np.float32)
    standardized = (static_features - dataset.static_feature_mean) / dataset.static_feature_std
    all_features = np.concatenate([standardized, rail_mask[:, None]], axis=-1).astype(np.float32, copy=False)
    case_seed = int(zlib.crc32(case_id.encode("utf-8")))
    rng = np.random.default_rng(np.random.SeedSequence([seed, case_seed, 1701]))
    query_idx = sample_indices(int(data.shape[0]), int(query_points), rng)
    target = ((np.asarray(data[query_idx, 3:6], dtype=np.float32) - dataset.displacement_mean) / dataset.displacement_std).astype(np.float32, copy=False)
    params = ((dataset._case_params(case_id) - dataset.parameter_mean) / dataset.parameter_std).astype(np.float32, copy=False)
    return {
        "raw_coordinates": raw_coordinates,
        "normalized_coordinates": normalized,
        "all_features": all_features,
        "part_ids": part_ids,
        "query_coordinates": normalized[query_idx],
        "query_features": all_features[query_idx],
        "query_part_ids": part_ids[query_idx],
        "target": target,
        "params": params,
        "query_indices": query_idx,
    }


def map_remeshed_features(
    dataset: ShiftCrashDataset,
    case: dict[str, np.ndarray],
    points: np.ndarray,
    cache_path: Path,
    mapping_workers: int,
    overwrite: bool,
) -> dict[str, np.ndarray]:
    if cache_path.is_file() and not overwrite:
        mapped = np.load(cache_path)
        nearest = np.asarray(mapped["nearest"], dtype=np.int64)
    else:
        from scipy.spatial import cKDTree

        tree = cKDTree(case["raw_coordinates"].astype(np.float64, copy=False))
        _distance, nearest = tree.query(points.astype(np.float64, copy=False), k=1, workers=int(mapping_workers))
        nearest = np.asarray(nearest, dtype=np.int64)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache_path.with_suffix(cache_path.suffix + ".partial")
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, nearest=nearest)
        temporary.replace(cache_path)
    if nearest.shape != (points.shape[0],):
        raise ValueError(f"Invalid nearest-node map in {cache_path}: {nearest.shape}")
    return {
        "normalized_coordinates": normalize_positions(points, dataset.min_position, dataset.position_span),
        "all_features": case["all_features"][nearest],
        "part_ids": case["part_ids"][nearest],
    }


def infer_one(
    model: ShiftCrashSMART,
    device: torch.device,
    source: dict[str, np.ndarray],
    case: dict[str, np.ndarray],
    query_chunk_size: int,
    sampling_seed: int,
) -> tuple[np.ndarray, float]:
    geo = torch.from_numpy(source["normalized_coordinates"]).unsqueeze(0).to(device)
    geo_features = torch.from_numpy(source["all_features"]).unsqueeze(0).to(device)
    geo_part_ids = torch.from_numpy(source["part_ids"]).unsqueeze(0).to(device)
    query = torch.from_numpy(case["query_coordinates"]).unsqueeze(0).to(device)
    query_features = torch.from_numpy(case["query_features"]).unsqueeze(0).to(device)
    query_part_ids = torch.from_numpy(case["query_part_ids"]).unsqueeze(0).to(device)
    params = torch.from_numpy(case["params"]).unsqueeze(0).to(device)
    sampling_seeds = torch.tensor([int(sampling_seed)], device=device, dtype=torch.long)
    empty_volume = query.new_empty((1, 0, 3))
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
        prediction, _empty = model.inference(
            geo,
            query,
            empty_volume,
            params,
            geometry_features=geo_features,
            query_features=query_features,
            geometry_part_ids=geo_part_ids,
            query_part_ids=query_part_ids,
            volume_query_features=None,
            volume_query_part_ids=None,
            sampling_seeds=sampling_seeds,
        )
    return prediction.squeeze(0).float().cpu().numpy(), float(geo.shape[1])


def relative_l2(target: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.linalg.norm(prediction - target) / max(np.linalg.norm(target), 1.0e-12))


def save_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def configure_plot(font_scale: float) -> None:
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


def save_figure(fig, path: Path) -> None:
    fig.savefig(path, dpi=260, bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)


def source_bar_plot(rows: list[dict[str, object]], output_dir: Path, log_scale: bool, source_subset: tuple[str, ...], title_prefix: str) -> None:
    means = OrderedDict()
    for source in source_subset:
        means[source] = {}
        for model in ("BASE", "SATLOSS7"):
            values = [float(row["rel_l2"]) for row in rows if row["source"] == source and row["model"] == model]
            if values:
                means[source][model] = float(np.mean(values))
    present = [source for source in source_subset if means[source]]
    if not present:
        return
    fig, ax = plt.subplots(figsize=(max(12.0, 1.7 * len(present)), 7.0))
    x = np.arange(len(present), dtype=np.float64)
    width = 0.34
    for offset, model in ((-width / 2.0, "BASE"), (width / 2.0, "SATLOSS7")):
        values = [means[source].get(model, np.nan) for source in present]
        bars = ax.bar(
            x + offset,
            values,
            width=width,
            color=[METHOD_COLORS[source] for source in present],
            edgecolor="#202124",
            linewidth=0.8,
            hatch="///" if model == "SATLOSS7" else None,
            label=MODEL_LABELS[model],
        )
        for bar, value in zip(bars, values):
            if not np.isfinite(value):
                continue
            ax.text(bar.get_x() + bar.get_width() / 2.0, value * (1.04 if not log_scale else 1.08), f"{value:.3g}", ha="center", va="bottom", fontsize=11.5)
    ax.set_xticks(x, [SOURCE_LABELS[source] for source in present], rotation=25, ha="right")
    ax.set_ylabel("Relative L2 displacement error")
    ax.set_title(f"{title_prefix}: SHIFT-Crash encoder-source comparison")
    ax.grid(axis="y", alpha=0.22)
    if log_scale:
        ax.set_yscale("log")
    ax.legend(frameon=True, loc="upper left")
    save_figure(fig, output_dir / f"shift_crash_combined_global_geometry_sources_bars_{'log' if log_scale else 'linear'}.png")


def relative_plot(rows: list[dict[str, object]], output_dir: Path, source_subset: tuple[str, ...], title_prefix: str) -> None:
    values = []
    labels = []
    present_sources = []
    for source in source_subset:
        base = [float(row["rel_l2"]) for row in rows if row["source"] == source and row["model"] == "BASE"]
        sat = [float(row["rel_l2"]) for row in rows if row["source"] == source and row["model"] == "SATLOSS7"]
        if not base or not sat:
            continue
        base_mean = float(np.mean(base))
        sat_mean = float(np.mean(sat))
        values.append(100.0 * (base_mean - sat_mean) / max(base_mean, 1.0e-12))
        labels.append(SOURCE_LABELS[source])
        present_sources.append(source)
    if not values:
        return
    fig, ax = plt.subplots(figsize=(max(10.5, 1.65 * len(values)), 6.6))
    x = np.arange(len(values))
    bars = ax.bar(x, values, color=[METHOD_COLORS[source] for source in present_sources], edgecolor="#202124")
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2.0, value + (1.0 if value >= 0 else -1.0), f"{value:+.1f}%", ha="center", va="bottom" if value >= 0 else "top", fontsize=12)
    ax.axhline(0.0, color="#202124", linewidth=1.0)
    ax.set_xticks(x, labels, rotation=25, ha="right")
    ax.set_ylabel("SATLOSS7 improvement vs base (%)")
    ax.set_title(f"{title_prefix}: SATLOSS7 relative improvement")
    ax.grid(axis="y", alpha=0.22)
    save_figure(fig, output_dir / "shift_crash_combined_global_geometry_sources_relative_vs_base.png")


def write_prediction_vtk(path: Path, points: np.ndarray, fields: dict[str, np.ndarray]) -> None:
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk

    path.parent.mkdir(parents=True, exist_ok=True)
    vtk_points = vtk.vtkPoints()
    vtk_points.SetData(numpy_to_vtk(np.asarray(points, dtype=np.float32), deep=True))
    poly = vtk.vtkPolyData()
    poly.SetPoints(vtk_points)
    vertices = vtk.vtkCellArray()
    # Use one vertex cell per point; this keeps the export readable in ParaView
    # without fabricating surface connectivity for the sampled query nodes.
    for index in range(int(points.shape[0])):
        vertices.InsertNextCell(1)
        vertices.InsertCellPoint(index)
    poly.SetVerts(vertices)
    point_data = poly.GetPointData()
    for name, values in fields.items():
        array = np.asarray(values)
        vtk_array = numpy_to_vtk(array, deep=True)
        vtk_array.SetName(name)
        point_data.AddArray(vtk_array)
    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(str(path))
    writer.SetInputData(poly)
    writer.SetFileTypeToBinary()
    if writer.Write() != 1:
        raise RuntimeError(f"Failed to write {path}")


def main() -> int:
    args = parse_args()
    if args.num_runs <= 0 or args.model_repeats <= 0:
        raise ValueError("--num-runs and --model-repeats must be positive.")
    if args.input_points <= 0 or args.query_points <= 0:
        raise ValueError("--input-points and --query-points must be positive.")
    summary = json.loads(args.study_summary.read_text(encoding="utf-8"))
    case_ids = [str(case_id) for case_id in summary["cases"]]
    if args.case_ids:
        requested = [item.strip() for item in str(args.case_ids).split(",") if item.strip()]
        case_ids = [case_id for case_id in case_ids if case_id in requested]
        if not case_ids:
            raise ValueError("None of --case-ids are present in the preparation summary.")
    case_ids = case_ids[: int(args.num_runs)]
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    devices = parse_devices(args.devices)
    base_cfg = load_cfg(args.base_config)
    sat_cfg = load_cfg(args.satloss7_config)
    base_checkpoint = Path(args.base_checkpoint).expanduser().resolve()
    sat_checkpoint = Path(args.satloss7_checkpoint).expanduser().resolve()
    for checkpoint in (base_checkpoint, sat_checkpoint):
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    print(f"Cases: {', '.join(case_ids)}")
    print(f"Devices: {', '.join(str(device) for device in devices)}")
    print("Protocol: fixed original queries/targets; only encoder source changes; no beta/sine shifts on VTPs.")
    if args.dry_run:
        print(f"Base config: {args.base_config}; SATLOSS7 config: {args.satloss7_config}")
        print(f"Study sources: {', '.join(SOURCE_ORDER)}")
        return 0

    dataset = ShiftCrashDataset(
        args.data_root,
        split=args.split,
        geometry_points=0,
        query_points=int(args.query_points),
        seed=int(args.seed),
        epoch_seeded_sampling=False,
        deterministic_geometry_sampling=True,
        deterministic_query_sampling=True,
        return_log_density=False,
        coordinate_normalization=str(getattr(base_cfg, "coordinate_normalization", "global_bounds")),
    )
    model_specs = OrderedDict(
        [
            ("BASE", (base_cfg, base_checkpoint, devices[0])),
            ("SATLOSS7", (sat_cfg, sat_checkpoint, devices[1 % len(devices)])),
        ]
    )
    models = {
        name: build_model(config, checkpoint, device, args.query_chunk_size)
        for name, (config, checkpoint, device) in model_specs.items()
    }
    summary_cases = {str(item["case_id"]): item for item in summary["records"]}
    rows: list[dict[str, object]] = []
    cache_root = output_dir / "nearest_feature_maps"
    vtk_root = output_dir / "prediction_vtks"
    for case_index, case_id in enumerate(case_ids):
        if case_id not in summary_cases:
            raise KeyError(f"Case {case_id} is missing from preparation summary records.")
        case = case_payload(dataset, case_id, args.query_points, args.seed)
        source_paths = {"aligned_uniform": None}
        source_paths.update(summary_cases[case_id]["outputs"])
        sources: dict[str, dict[str, np.ndarray]] = {}
        for source_name in SOURCE_ORDER:
            vtp_path = source_paths.get(source_name)
            if source_name == "aligned_uniform":
                sources[source_name] = {
                    "normalized_coordinates": case["normalized_coordinates"],
                    "all_features": case["all_features"],
                    "part_ids": case["part_ids"],
                }
            else:
                points = read_vtp_points(Path(vtp_path))
                mapping_path = cache_root / case_id / f"{source_name}.npz"
                sources[source_name] = map_remeshed_features(
                    dataset,
                    case,
                    points,
                    mapping_path,
                    args.mapping_workers,
                    args.overwrite_mapping_cache,
                )
        for repeat in range(int(args.model_repeats)):
            for source_name, source in sources.items():
                source_seed = int(zlib.crc32(source_name.encode("utf-8")))
                source_rng = np.random.default_rng(np.random.SeedSequence([args.seed, case_index, repeat, source_seed]))
                input_idx = sample_indices(int(source["normalized_coordinates"].shape[0]), args.input_points, source_rng)
                view = {key: value[input_idx] for key, value in source.items()}
                with ThreadPoolExecutor(max_workers=len(model_specs)) as inference_pool:
                    futures = {
                        model_name: inference_pool.submit(
                            infer_one,
                            models[model_name],
                            device,
                            view,
                            case,
                            args.query_chunk_size,
                            int(args.seed + case_index * 100003 + repeat * 1009 + source_seed),
                        )
                        for model_name, (_config, _checkpoint, device) in model_specs.items()
                    }
                    inference_results = {
                        model_name: future.result()
                        for model_name, future in futures.items()
                    }
                for model_name, (_config, _checkpoint, device) in model_specs.items():
                    prediction, actual_input_points = inference_results[model_name]
                    target = case["target"]
                    error_norm = relative_l2(target, prediction)
                    prediction_physical = prediction * dataset.displacement_std + dataset.displacement_mean
                    target_physical = target * dataset.displacement_std + dataset.displacement_mean
                    error_physical = relative_l2(target_physical, prediction_physical)
                    rows.append(
                        {
                            "case_id": case_id,
                            "source": source_name,
                            "model": model_name,
                            "repeat": repeat,
                            "source_points": int(source["normalized_coordinates"].shape[0]),
                            "input_points": int(actual_input_points),
                            "rel_l2": error_norm,
                            "physical_rel_l2": error_physical,
                        }
                    )
                    if case_index == 0 and repeat == 0:
                        fields = {
                            "target_displacement_x": target_physical[:, 0],
                            "target_displacement_y": target_physical[:, 1],
                            "target_displacement_z": target_physical[:, 2],
                            "predicted_displacement_x": prediction_physical[:, 0],
                            "predicted_displacement_y": prediction_physical[:, 1],
                            "predicted_displacement_z": prediction_physical[:, 2],
                            "absolute_error": np.linalg.norm(prediction_physical - target_physical, axis=1),
                        }
                        write_prediction_vtk(
                            vtk_root / f"{case_id}_{source_name}_{model_name.lower()}_query.vtk",
                            case["raw_coordinates"][case["query_indices"]],
                            fields,
                        )
        print(f"[{case_index + 1}/{len(case_ids)}] completed {case_id}", flush=True)

    save_csv(output_dir / "shift_crash_sampling_metrics.csv", rows)
    aggregate_rows = []
    for source in SOURCE_ORDER:
        for model in ("BASE", "SATLOSS7"):
            values = [float(row["rel_l2"]) for row in rows if row["source"] == source and row["model"] == model]
            physical = [float(row["physical_rel_l2"]) for row in rows if row["source"] == source and row["model"] == model]
            if values:
                aggregate_rows.append(
                    {
                        "source": source,
                        "model": model,
                        "mean_rel_l2": float(np.mean(values)),
                        "std_rel_l2": float(np.std(values)),
                        "mean_physical_rel_l2": float(np.mean(physical)),
                        "std_physical_rel_l2": float(np.std(physical)),
                        "count": len(values),
                    }
                )
    (output_dir / "shift_crash_sampling_aggregate.json").write_text(json.dumps(aggregate_rows, indent=2) + "\n", encoding="utf-8")
    configure_plot(args.plot_font_scale)
    title_prefix = f"SHIFT-Crash ({len(case_ids)} cases)"
    source_bar_plot(rows, output_dir, False, SOURCE_ORDER, title_prefix)
    source_bar_plot(rows, output_dir, True, SOURCE_ORDER, title_prefix)
    relative_plot(rows, output_dir, SOURCE_ORDER, title_prefix)
    for method in ("angle", "voxel", "isotropic"):
        subset = tuple(source for source in SOURCE_ORDER if source.startswith(method))
        source_bar_plot(rows, output_dir, False, subset, f"{title_prefix}, {method}")
        source_bar_plot(rows, output_dir, True, subset, f"{title_prefix}, {method}")
    metadata = {
        "data_root": str(args.data_root.resolve()),
        "study_summary": str(args.study_summary.resolve()),
        "base_config": args.base_config,
        "satloss7_config": args.satloss7_config,
        "base_checkpoint": str(base_checkpoint),
        "satloss7_checkpoint": str(sat_checkpoint),
        "cases": case_ids,
        "query_points": int(args.query_points),
        "input_points": int(args.input_points),
        "protocol": "fixed original query nodes and terminal displacement; remeshed encoder points only; no beta/sine VTP shifts",
    }
    (output_dir / "comparison_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Metrics: {output_dir / 'shift_crash_sampling_metrics.csv'}")
    print(f"Aggregate: {output_dir / 'shift_crash_sampling_aggregate.json'}")
    print(f"Plots: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
