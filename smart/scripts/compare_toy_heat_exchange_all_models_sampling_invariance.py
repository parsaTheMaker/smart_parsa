#!/usr/bin/env python3
"""Paper-style all-model DeAL comparison on held-out heat exchangers.

The candidate ranking is deliberately internal: every base/DeAL pair is
evaluated on the same positive sampling and remeshing conditions, then only
the selected cases are written and visualised.  This prevents a selected-case
plot from accidentally including a different condition or a different query
cloud for one of the model families.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from tqdm import tqdm

from data.toy_heat_exchange_dataset import ToyHeatExchangeDataset
from models.lno import LNO
from models.mspt import MSPT
from models.point_transformer_v3 import PointTransformerV3
from models.pointnet2_ssg import PointNet2SSG
from models.smart.smart import SMART
from models.transolverpp import TransolverPP
from train_consistency_common import sample_geometry_view


MODEL_SPECS: "OrderedDict[str, dict[str, Any]]" = OrderedDict(
    [
        ("SMART", {"label": "SMART", "ctor": SMART, "config": "toy_heat_exchange", "pair": "SMART"}),
        ("SMART_SATLOSS7", {"label": "SMART-DeAL", "ctor": SMART, "config": "toy_heat_exchange_satloss7", "pair": "SMART"}),
        ("MSPT", {"label": "MSPT", "ctor": MSPT, "config": "toy_heat_exchange_mspt", "pair": "MSPT"}),
        ("MSPT_SATLOSS7", {"label": "MSPT-DeAL", "ctor": MSPT, "config": "toy_heat_exchange_mspt_satloss7", "pair": "MSPT"}),
        ("LNO", {"label": "LNO", "ctor": LNO, "config": "toy_heat_exchange_lno", "pair": "LNO"}),
        ("LNO_SATLOSS7", {"label": "LNO-DeAL", "ctor": LNO, "config": "toy_heat_exchange_lno_satloss7", "pair": "LNO"}),
        ("POINTNET2_SSG", {"label": "PointNet++-SSG", "ctor": PointNet2SSG, "config": "toy_heat_exchange_pointnet2_ssg", "pair": "POINTNET2_SSG"}),
        ("POINTNET2_SSG_SATLOSS7", {"label": "PointNet++-SSG-DeAL", "ctor": PointNet2SSG, "config": "toy_heat_exchange_pointnet2_ssg_satloss7", "pair": "POINTNET2_SSG"}),
        ("TRANSOLVERPP", {"label": "TransolverPP", "ctor": TransolverPP, "config": "toy_heat_exchange_transolverpp", "pair": "TRANSOLVERPP"}),
        ("TRANSOLVERPP_SATLOSS7", {"label": "TransolverPP-DeAL", "ctor": TransolverPP, "config": "toy_heat_exchange_transolverpp_satloss7", "pair": "TRANSOLVERPP"}),
        ("POINT_TRANSFORMER_V3", {"label": "PointTransformerV3", "ctor": PointTransformerV3, "config": "toy_heat_exchange_point_transformer_v3", "pair": "POINT_TRANSFORMER_V3"}),
        ("POINT_TRANSFORMER_V3_SATLOSS7", {"label": "PointTransformerV3-DeAL", "ctor": PointTransformerV3, "config": "toy_heat_exchange_point_transformer_v3_satloss7", "pair": "POINT_TRANSFORMER_V3"}),
    ]
)

# These are the paired family colours from the established DrivAerML plots.
FAMILY_COLORS = {
    "SMART": "#1F77B4",
    "TRANSOLVERPP": "#FF7F0E",
    "POINTNET2_SSG": "#17BECF",
    "LNO": "#D62728",
    "MSPT": "#2CA02C",
    "POINT_TRANSFORMER_V3": "#7F3C8D",
}
FAMILY_ORDER = ("SMART", "TRANSOLVERPP", "POINTNET2_SSG", "LNO", "MSPT", "POINT_TRANSFORMER_V3")
SOURCE_LABELS = {
    "angle_div5": "Angle div5",
    "angle_div10": "Angle div10",
    "isotropic_div5": "Isotropic div5",
    "isotropic_div10": "Isotropic div10",
    "voxel_div5": "Voxel div5",
    "voxel_div10": "Voxel div10",
}
V4_SOURCE_LABELS = {
    "angle_div5": "Feature-aware div5",
    "angle_div10": "Feature-aware div10",
    "isotropic_div5": "QEM div5",
    "isotropic_div10": "QEM div10",
    "voxel_div5": "Voxel-grid clustering div5",
    "voxel_div10": "Voxel-grid clustering div10",
}
METHOD_EDGES = {"angle": "#222222", "isotropic": "#1B7837", "voxel": "#B15928"}
FACTOR_ALPHAS = {5: 0.50, 10: 1.00}


def parse_csv(value: str, cast=str) -> list[Any]:
    return [cast(item.strip()) for item in str(value).split(",") if item.strip()]


def source_stem(case_id: int) -> str:
    return f"heat_exchange_case_{case_id:05d}_surface"


def make_config(config_name: str):
    config_dir = str(Path(__file__).resolve().parents[1] / "config")
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        return compose(config_name=config_name)


def model_input_budget(cfg) -> int:
    exp = cfg.experiment
    for key in ("eval_view_geometry_points", "view_geometry_points", "num_body_points"):
        value = int(exp.get(key, 0))
        if value > 0:
            return value
    raise ValueError("Could not resolve a positive train-aligned geometry budget.")


def load_model(model_name: str, checkpoint_path: Path, device: torch.device):
    spec = MODEL_SPECS[model_name]
    cfg = make_config(str(spec["config"]))
    exp = cfg.experiment
    architecture = OmegaConf.to_container(exp.architecture, resolve=True)
    if not isinstance(architecture, dict):
        raise TypeError(f"{model_name} architecture must be a mapping.")
    kwargs = {
        "spatial_dim": 3,
        "surface_channels": 1,
        "volume_channels": 1,
        "parameter_channels": 0,
        **architecture,
    }
    model = spec["ctor"](**kwargs).to(device).eval()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint.get("model_state_dict", checkpoint)
    if not isinstance(state, dict):
        raise TypeError(f"{checkpoint_path} does not contain a model state dict.")
    state = {str(key).removeprefix("module."): value for key, value in state.items()}
    target = model.state_dict()
    missing = [key for key in target if key not in state]
    unexpected = [key for key in state if key not in target]
    mismatched = [key for key in target if key in state and tuple(target[key].shape) != tuple(state[key].shape)]
    if missing or unexpected or mismatched:
        raise RuntimeError(
            f"Checkpoint/config mismatch for {model_name}: missing={len(missing)}, "
            f"unexpected={len(unexpected)}, shape_mismatch={len(mismatched)}. "
            "Use the checkpoint produced by the listed toy configuration."
        )
    model.load_state_dict(state, strict=True)
    return model, cfg


def select_indices(geo: torch.Tensor, density: torch.Tensor, budget: int, shift: str, seed: int) -> torch.Tensor:
    if shift == "beta":
        mode, beta, axis, mix = "inverse_density_wor", 1.0, None, 0.0
    elif shift == "sine_x":
        mode, beta, axis, mix = "sinusoidal_axis_mixture_wor", 0.0, 0, 1.0
    elif shift == "sine_y":
        mode, beta, axis, mix = "sinusoidal_axis_mixture_wor", 0.0, 1, 1.0
    else:
        raise ValueError(f"Unsupported shift {shift!r}")
    _, _, _, indices = sample_geometry_view(
        geo.unsqueeze(0), density.unsqueeze(0), budget, mode, beta, 1.0, seed,
        sinusoidal_axis=axis, sinusoidal_mix_fraction=mix, return_indices=True,
    )
    return indices[0]


def load_vtp_mesh(path: Path) -> tuple[np.ndarray, np.ndarray]:
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy

    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(str(path))
    reader.Update()
    poly = reader.GetOutput()
    if poly is None or poly.GetPoints() is None or poly.GetNumberOfPoints() == 0:
        raise RuntimeError(f"Invalid or empty VTP: {path}")
    points = np.asarray(vtk_to_numpy(poly.GetPoints().GetData()), dtype=np.float32)
    raw = np.asarray(vtk_to_numpy(poly.GetPolys().GetData()), dtype=np.int64)
    if raw.size == 0 or raw.size % 4 or not np.all(raw[::4] == 3):
        raise RuntimeError(f"VTP is not a triangle mesh: {path}")
    faces = raw.reshape(-1, 4)[:, 1:]
    if not np.isfinite(points).all() or faces.max(initial=-1) >= points.shape[0]:
        raise RuntimeError(f"Invalid mesh coordinates/connectivity in {path}")
    return np.ascontiguousarray(points), np.ascontiguousarray(faces)


def sample_remeshed_encoder_points(path: Path, budget: int, seed: int) -> np.ndarray:
    """Uniformly sample triangles; no extra beta/sine shift is applied to VTPs."""
    points, faces = load_vtp_mesh(path)
    rng = np.random.default_rng(seed)
    chosen_faces = rng.integers(0, faces.shape[0], size=int(budget), endpoint=False)
    vertices = points[faces[chosen_faces]]
    weights = -np.log(np.maximum(rng.random((int(budget), 3)), 1.0e-12))
    weights /= weights.sum(axis=1, keepdims=True)
    return np.ascontiguousarray(np.einsum("ni,nij->nj", weights, vertices), dtype=np.float32)


def vtp_paths(case_id: int, args: argparse.Namespace) -> dict[str, Path]:
    roots = {
        "angle": Path(args.angle_decimated_vtp_dir),
        "isotropic": Path(args.isotropic_decimated_vtp_dir),
        "voxel": Path(args.voxel_decimated_vtp_dir),
    }
    paths: dict[str, Path] = {}
    for method in parse_csv(args.active_geometry_sources):
        if method not in roots:
            raise ValueError(f"Unknown remeshing source {method!r}")
        for factor in parse_csv(args.geometry_decimation_factors, int):
            path = roots[method] / f"case_{case_id:05d}" / f"{source_stem(case_id)}_faces_div{factor}.vtp"
            if path.is_file():
                paths[f"{method}_div{factor}"] = path
    return paths


def physical_to_normalized(points: np.ndarray, dataset: ToyHeatExchangeDataset) -> torch.Tensor:
    normalized = (points - dataset.min_pos.numpy()[None, :]) / dataset.position_span.numpy()[None, :]
    if not np.isfinite(normalized).all():
        raise RuntimeError("Remeshed coordinates became non-finite after the training normalization.")
    return torch.from_numpy(np.ascontiguousarray(normalized, dtype=np.float32))


def rel_l2(pred: torch.Tensor, target: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(pred - target) / torch.linalg.vector_norm(target).clamp_min(1.0e-12))


@torch.inference_mode()
def predict_errors(model, device: torch.device, geo: torch.Tensor, surf_q: torch.Tensor, surf_y: torch.Tensor, vol_q: torch.Tensor, vol_y: torch.Tensor, dataset: ToyHeatExchangeDataset) -> tuple[float, float, float]:
    with torch.autocast(device_type=device.type, enabled=device.type == "cuda", dtype=torch.float16):
        pred_s, pred_v = model.inference(
            geo.unsqueeze(0).to(device, non_blocking=True),
            surf_q.unsqueeze(0).to(device, non_blocking=True),
            vol_q.unsqueeze(0).to(device, non_blocking=True),
            None,
        )
    pred_s = pred_s[0].float().cpu() * dataset.std_surf_data + dataset.mean_surf_data
    pred_v = pred_v[0].float().cpu() * dataset.std_vol_data + dataset.mean_vol_data
    true_s = surf_y * dataset.std_surf_data + dataset.mean_surf_data
    true_v = vol_y * dataset.std_vol_data + dataset.mean_vol_data
    surface = rel_l2(pred_s, true_s)
    volume = rel_l2(pred_v, true_v)
    return surface, volume, 0.5 * (surface + volume)


def conditions_for_case(case_index: int, case_id: int, geo: torch.Tensor, density: torch.Tensor, dataset: ToyHeatExchangeDataset, args: argparse.Namespace, input_points: int) -> dict[str, torch.Tensor]:
    views = {
        shift: geo.index_select(0, select_indices(geo, density, input_points, shift, args.seed + 100_003 * case_index + 10_007 * index))
        for index, shift in enumerate(("beta", "sine_x", "sine_y"))
    }
    expected = [f"{method}_div{factor}" for method in parse_csv(args.active_geometry_sources) for factor in parse_csv(args.geometry_decimation_factors, int)]
    found = vtp_paths(case_id, args)
    missing = [name for name in expected if name not in found]
    if missing:
        raise FileNotFoundError(f"case {case_id}: missing required remeshing VTPs: {', '.join(missing)}")
    for source, path in found.items():
        sampled = sample_remeshed_encoder_points(path, input_points, args.seed + case_id + sum(map(ord, source)))
        views[source] = physical_to_normalized(sampled, dataset)
    return views


def evaluate_case(case_index: int, dataset: ToyHeatExchangeDataset, models: dict[str, Any], model_devices: dict[str, torch.device], args: argparse.Namespace, input_points: int) -> list[dict[str, Any]]:
    geo, surf_q, surf_y, vol_q, vol_y, density = dataset[case_index]
    case_id = int(dataset.data[case_index])
    conditions = conditions_for_case(case_index, case_id, geo, density, dataset, args, input_points)
    device_models: OrderedDict[str, list[str]] = OrderedDict()
    for name in MODEL_SPECS:
        device_models.setdefault(str(model_devices[name]), []).append(name)

    def evaluate_device_group(names: list[str]) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for model_name in names:
            for condition, view in conditions.items():
                surface, volume, global_error = predict_errors(
                    models[model_name], model_devices[model_name], view, surf_q, surf_y, vol_q, vol_y, dataset
                )
                records.append({
                    "case_id": case_id,
                    "model_name": model_name,
                    "sampling_mode": condition,
                    "surface_global_rel_l2": surface,
                    "volume_global_rel_l2": volume,
                    "combined_global_rel_l2": global_error,
                })
        return records

    with ThreadPoolExecutor(max_workers=len(device_models)) as executor:
        futures = [executor.submit(evaluate_device_group, names) for names in device_models.values()]
        rows: list[dict[str, Any]] = []
        for future in futures:
            rows.extend(future.result())
    return rows


def candidate_score(
    rows: list[dict[str, Any]],
    ranking_families: list[str],
    ranking_modes: list[str],
) -> float:
    """Mean paired DeAL improvement used only for internal case selection."""
    values: list[float] = []
    row_map = {(row["model_name"], row["sampling_mode"]): row for row in rows}
    for family in ranking_families:
        for mode in ranking_modes:
            base = row_map.get((family, mode))
            sat = row_map.get((f"{family}_SATLOSS7", mode))
            if base is None or sat is None:
                raise RuntimeError(f"Missing paired candidate result for {family}/{mode}.")
            base_error = float(base["combined_global_rel_l2"])
            sat_error = float(sat["combined_global_rel_l2"])
            values.append(100.0 * (base_error - sat_error) / max(abs(base_error), 1.0e-12))
    return float(np.mean(values))


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for model_name in MODEL_SPECS:
        modes = sorted({str(row["sampling_mode"]) for row in rows if row["model_name"] == model_name})
        for mode in modes:
            subset = [row for row in rows if row["model_name"] == model_name and row["sampling_mode"] == mode]
            record = {"model_name": model_name, "sampling_mode": mode}
            for metric in ("surface_global_rel_l2", "volume_global_rel_l2", "combined_global_rel_l2"):
                values = np.asarray([float(row[metric]) for row in subset], dtype=np.float64)
                record[metric] = float(np.mean(values))
            result.append(record)
    return result


def family_models() -> OrderedDict[str, list[str]]:
    return OrderedDict((family, [family, f"{family}_SATLOSS7"]) for family in FAMILY_ORDER)


def add_percent_label(ax, bar, value: float, baseline: float, scale: str, font_size: float) -> None:
    percent = 100.0 * (value - baseline) / max(abs(baseline), 1.0e-12)
    if scale == "log":
        y = value * 1.12
    else:
        y = value
    ax.text(bar.get_x() + bar.get_width() / 2.0, y, f"{percent:+.1f}%", ha="center", va="bottom", rotation=90, fontsize=font_size, clip_on=False)


def plot_endpoint_bars(aggregate: list[dict[str, Any]], mode: str, out_path: Path, title: str, scale: str, font_scale: float) -> None:
    row_map = {(row["model_name"], row["sampling_mode"]): row for row in aggregate}
    groups = family_models()
    x = np.arange(len(groups), dtype=np.float64)
    width = 0.30
    fig, ax = plt.subplots(figsize=(12.2, 5.7))
    fig.subplots_adjust(left=0.12, right=0.80, bottom=0.27, top=0.85)
    values: list[float] = []
    font = 8.2 * font_scale
    for group_index, (family, names) in enumerate(groups.items()):
        base = float(row_map[(names[0], mode)]["combined_global_rel_l2"])
        sat = float(row_map[(names[1], mode)]["combined_global_rel_l2"])
        values.extend((base, sat))
        color = FAMILY_COLORS[family]
        ax.bar(x[group_index] - width / 2, base, width, color=color, edgecolor="black", linewidth=0.65)
        sat_bar = ax.bar(x[group_index] + width / 2, sat, width, color=color, edgecolor="black", linewidth=0.65, hatch="///")[0]
        add_percent_label(ax, sat_bar, sat, base, scale, font)
    if scale == "log":
        ax.set_yscale("log")
        ax.set_ylim(bottom=max(min(values) * 0.65, 1.0e-10), top=max(values) * 1.38)
    else:
        lo, hi = min(values), max(values)
        ax.set_ylim(bottom=max(0.0, lo - 0.10 * max(hi - lo, hi * 0.1)), top=hi * 1.22)
    ax.set_xticks(x, [MODEL_SPECS[family]["label"] for family in groups], rotation=22, ha="right")
    ax.set_ylabel(f"Combined global relative L2 ({scale} scale)", fontsize=font + 1)
    ax.yaxis.set_label_coords(-0.11, 0.5)
    ax.tick_params(axis="both", labelsize=font)
    ax.set_title(title, fontsize=font + 1.5)
    ax.grid(axis="y", which="both", alpha=0.20)
    ax.legend(handles=[
        Patch(facecolor="none", edgecolor="black", label="Base (no hatch)"),
        Patch(facecolor="none", edgecolor="black", hatch="///", label="DeAL (hatch)"),
        Patch(facecolor="none", edgecolor="none", label="DeAL labels: % vs base"),
    ], loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=font, framealpha=0.92)
    fig.savefig(out_path, dpi=280, bbox_inches="tight", pad_inches=0.10)
    plt.close(fig)


def plot_geometry_factor_average_bars(aggregate: list[dict[str, Any]], modes: list[str], out_path: Path, title: str, scale: str, font_scale: float) -> None:
    """Average remeshing methods, retaining paired div5/div10 comparisons.

    A div5 bar is the mean error over angle, isotropic, and voxel div5 inputs
    for the same model and selected cases. The matching DeAL bar uses the
    same three sources. This keeps the plot readable without privileging one
    remesher or silently changing the paired numerator/denominator.
    """
    row_map = {(row["model_name"], row["sampling_mode"]): row for row in aggregate}
    groups = family_models()
    factors = sorted({int(mode.rsplit("div", 1)[1]) for mode in modes})
    factor_modes = {factor: [mode for mode in modes if mode.endswith(f"div{factor}")] for factor in factors}
    if any(not source_modes for source_modes in factor_modes.values()):
        raise ValueError("Every displayed decimation factor must include at least one source.")
    x = np.arange(len(groups), dtype=np.float64)
    slots = len(factors) * 2
    width = 0.78 / slots
    pitch = 0.84 / slots
    fig, ax = plt.subplots(figsize=(14.0, 6.2))
    fig.subplots_adjust(left=0.11, right=0.74, bottom=0.30, top=0.85)
    values: list[float] = []
    labels: list[float] = []
    font = 7.8 * font_scale
    for factor_index, factor in enumerate(factors):
        source_modes = factor_modes[factor]
        for group_index, (family, names) in enumerate(groups.items()):
            color = FAMILY_COLORS[family]
            base = float(np.mean([float(row_map[(names[0], mode)]["combined_global_rel_l2"]) for mode in source_modes]))
            sat = float(np.mean([float(row_map[(names[1], mode)]["combined_global_rel_l2"]) for mode in source_modes]))
            values.extend((base, sat))
            for variant_index, (name, value) in enumerate(((names[0], base), (names[1], sat))):
                slot = factor_index * 2 + variant_index
                pos = x[group_index] + (slot - 0.5 * (slots - 1)) * pitch
                bar = ax.bar(pos, value, width, color=color, edgecolor="black", linewidth=0.7, alpha=FACTOR_ALPHAS.get(factor, 1.0), hatch="///" if variant_index else "")[0]
                if variant_index:
                    labels.append(value)
                    add_percent_label(ax, bar, value, base, scale, font)
    if scale == "log":
        ax.set_yscale("log")
        ax.set_ylim(bottom=max(min(values) * 0.65, 1.0e-10), top=max(max(values), max(labels, default=0.0)) * 1.42)
    else:
        lo, hi = min(values), max(values)
        ax.set_ylim(bottom=max(0.0, lo - 0.10 * max(hi - lo, hi * 0.1)), top=hi * 1.26)
    ax.set_xticks(x, [MODEL_SPECS[family]["label"] for family in groups], rotation=22, ha="right")
    ax.set_ylabel(f"Combined global relative L2 ({scale} scale)", fontsize=font + 1)
    ax.yaxis.set_label_coords(-0.12, 0.5)
    ax.tick_params(axis="both", labelsize=font)
    ax.set_title(title, fontsize=font + 1.5)
    ax.grid(axis="y", which="both", alpha=0.20)
    opacity = ax.legend(handles=[
        Patch(facecolor="black", edgecolor="black", alpha=0.50, label="div5 50% opacity"),
        Patch(facecolor="black", edgecolor="black", alpha=1.00, label="div10 100% opacity"),
    ], loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=font, framealpha=0.92)
    ax.add_artist(opacity)
    ax.legend(handles=[
        Patch(facecolor="none", edgecolor="black", label="Base (no hatch)"),
        Patch(facecolor="none", edgecolor="black", hatch="///", label="DeAL (hatch)"),
        Patch(facecolor="none", edgecolor="none", label="Each div bar: mean of three remeshers"),
        Patch(facecolor="none", edgecolor="none", label="DeAL labels: % vs base"),
    ], loc="upper left", bbox_to_anchor=(1.01, 0.67), fontsize=font, framealpha=0.92)
    fig.savefig(out_path, dpi=280, bbox_inches="tight", pad_inches=0.10)
    plt.close(fig)


def write_summary_table(aggregate: list[dict[str, Any]], modes: list[str], path: Path) -> None:
    row_map = {(row["model_name"], row["sampling_mode"]): row for row in aggregate}
    fields = ["model", *[f"{mode}_deal_improvement_percent" for mode in modes]]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for family in FAMILY_ORDER:
            record: dict[str, Any] = {"model": MODEL_SPECS[family]["label"]}
            for mode in modes:
                base = float(row_map[(family, mode)]["combined_global_rel_l2"])
                sat = float(row_map[(f"{family}_SATLOSS7", mode)]["combined_global_rel_l2"])
                record[f"{mode}_deal_improvement_percent"] = 100.0 * (base - sat) / max(abs(base), 1.0e-12)
            writer.writerow(record)


def write_improvement_markdown(aggregate: list[dict[str, Any]], modes: list[str], path: Path) -> None:
    row_map = {(row["model_name"], row["sampling_mode"]): row for row in aggregate}
    labels = [SOURCE_LABELS.get(mode, mode.replace("_", " ").title()) for mode in modes]
    lines = ["| Model | " + " | ".join(labels) + " |", "|" + "---|" * (len(labels) + 1)]
    for family in FAMILY_ORDER:
        values = []
        for mode in modes:
            base = float(row_map[(family, mode)]["combined_global_rel_l2"])
            sat = float(row_map[(f"{family}_SATLOSS7", mode)]["combined_global_rel_l2"])
            values.append(f"{100.0 * (base - sat) / max(abs(base), 1.0e-12):+.2f}%")
        lines.append("| " + MODEL_SPECS[family]["label"] + " | " + " | ".join(values) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_satloss_improvements(aggregate: list[dict[str, Any]], modes: list[str], out_path: Path, font_scale: float) -> None:
    """Mirror the DrivAerML endpoint-improvement figure for every toy pair."""
    row_map = {(row["model_name"], row["sampling_mode"]): row for row in aggregate}
    x = np.arange(len(modes), dtype=np.float64)
    width = 0.80 / len(FAMILY_ORDER)
    fig, ax = plt.subplots(figsize=(max(12.0, 1.85 * len(modes)), 6.1))
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.27, top=0.85)
    all_values: list[float] = []
    font = 8.0 * font_scale
    for index, family in enumerate(FAMILY_ORDER):
        values = []
        for mode in modes:
            base = float(row_map[(family, mode)]["combined_global_rel_l2"])
            sat = float(row_map[(f"{family}_SATLOSS7", mode)]["combined_global_rel_l2"])
            values.append(100.0 * (base - sat) / max(abs(base), 1.0e-12))
        all_values.extend(values)
        bars = ax.bar(
            x + (index - 0.5 * (len(FAMILY_ORDER) - 1)) * width,
            values,
            width=width,
            color=FAMILY_COLORS[family],
            edgecolor="black",
            linewidth=0.55,
            label=MODEL_SPECS[family]["label"],
        )
        for bar, value in zip(bars, values):
            vertical = 0.8 if value >= 0.0 else -0.8
            ax.text(bar.get_x() + bar.get_width() / 2.0, value + vertical, f"{value:+.1f}%", ha="center", va="bottom" if value >= 0 else "top", rotation=90, fontsize=font)
    margin = max(2.0, 0.12 * max(max(all_values) - min(all_values), 1.0))
    ax.set_ylim(min(all_values) - margin, max(all_values) + margin)
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.set_xticks(x, [SOURCE_LABELS.get(mode, mode.replace("_", " ").title()) for mode in modes], rotation=18, ha="right")
    ax.set_ylabel("DeAL improvement versus base (%)", fontsize=font + 1)
    ax.set_title("Heat Exchanger: DeAL improvement by shift", fontsize=font + 1.5)
    ax.tick_params(axis="both", labelsize=font)
    ax.grid(axis="y", alpha=0.2)
    ax.legend(fontsize=font, ncol=3, loc="best", framealpha=0.92)
    fig.savefig(out_path, dpi=280, bbox_inches="tight", pad_inches=0.10)
    plt.close(fig)


def plot_density_validation(dataset: ToyHeatExchangeDataset, case_index: int, input_points: int, args: argparse.Namespace, out_path: Path) -> None:
    geo, _, _, _, _, density = dataset[case_index]
    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    for offset, (shift, color) in enumerate((("beta", "#1F77B4"), ("sine_x", "#D62728"), ("sine_y", "#2CA02C"))):
        indices = select_indices(geo, density, input_points, shift, args.seed + 17_000 + offset)
        ax.hist(density[indices].numpy(), bins=48, density=True, histtype="step", linewidth=2.1, color=color, label=shift.replace("_", " "))
    ax.set_xlabel("KDE-16 log density")
    ax.set_ylabel("Probability density")
    ax.set_title("Heat Exchanger: encoder density under positive shifts")
    ax.grid(alpha=0.2)
    ax.legend(framealpha=0.92)
    fig.savefig(out_path, dpi=260, bbox_inches="tight", pad_inches=0.10)
    plt.close(fig)


def write_point_vtp(path: Path, points: np.ndarray, arrays: dict[str, np.ndarray]) -> None:
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray

    path.parent.mkdir(parents=True, exist_ok=True)
    poly = vtk.vtkPolyData()
    vtk_points = vtk.vtkPoints()
    vtk_points.SetData(numpy_to_vtk(np.ascontiguousarray(points, dtype=np.float32), deep=True))
    poly.SetPoints(vtk_points)
    ids = np.arange(points.shape[0], dtype=np.int64)
    cells = vtk.vtkCellArray()
    cells.SetData(numpy_to_vtkIdTypeArray(np.arange(points.shape[0] + 1, dtype=np.int64), deep=True), numpy_to_vtkIdTypeArray(ids, deep=True))
    poly.SetVerts(cells)
    for name, values in arrays.items():
        array = numpy_to_vtk(np.ascontiguousarray(values, dtype=np.float32), deep=True)
        array.SetName(name)
        poly.GetPointData().AddArray(array)
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(str(path))
    writer.SetInputData(poly)
    writer.SetDataModeToBinary()
    writer.SetCompressor(None)
    if writer.Write() != 1:
        raise RuntimeError(f"Could not write {path}")


def export_analysis(dataset: ToyHeatExchangeDataset, selected: list[int], models: dict[str, Any], model_devices: dict[str, torch.device], args: argparse.Namespace, input_points: int, output_dir: Path) -> None:
    if args.analysis_case_count <= 0:
        return
    root = output_dir / "analysis_vtps"
    original_root = Path(args.original_vtp_dir)
    for case_index in selected[:args.analysis_case_count]:
        geo, surf_q, surf_y, vol_q, vol_y, density = dataset[case_index]
        case_id = int(dataset.data[case_index])
        case_dir = root / f"case_{case_id:05d}"
        case_dir.mkdir(parents=True, exist_ok=True)
        original = original_root / f"case_{case_id:05d}" / f"{source_stem(case_id)}.vtp"
        if original.is_file():
            shutil.copy2(original, case_dir / "original_surface_mesh.vtp")
        conditions = conditions_for_case(case_index, case_id, geo, density, dataset, args, input_points)
        true_surface = np.asarray((surf_y * dataset.std_surf_data + dataset.mean_surf_data).numpy(), dtype=np.float32)
        physical_queries = surf_q.numpy() * dataset.position_span.numpy()[None, :] + dataset.min_pos.numpy()[None, :]
        for condition, view in conditions.items():
            physical_encoder = view.numpy() * dataset.position_span.numpy()[None, :] + dataset.min_pos.numpy()[None, :]
            write_point_vtp(case_dir / f"{condition}_encoder_input_points.vtp", physical_encoder, {"encoder_point": np.ones((view.shape[0], 1), dtype=np.float32)})
            for model_name, model in models.items():
                device = model_devices[model_name]
                with torch.inference_mode(), torch.autocast(device_type=device.type, enabled=device.type == "cuda", dtype=torch.float16):
                    pred_s, _ = model.inference(view.unsqueeze(0).to(device), surf_q.unsqueeze(0).to(device), vol_q.unsqueeze(0).to(device), None)
                prediction = np.asarray((pred_s[0].float().cpu() * dataset.std_surf_data + dataset.mean_surf_data).numpy(), dtype=np.float32)
                write_point_vtp(case_dir / f"{condition}_{model_name.lower()}_surface_prediction.vtp", physical_queries, {"ground_truth": true_surface, "prediction": prediction, "absolute_error": np.abs(prediction - true_surface)})


def main() -> None:
    global SOURCE_LABELS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="/mnt/ssdraid/parsa/toy_heat_exchange_fem_v1")
    parser.add_argument("--candidate-pool-size", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--candidate-split",
        choices=("validation", "train", "all"),
        default="validation",
        help="Cases eligible for ranking. 'all' intentionally mixes training and validation cases; use only for exploratory visual analysis.",
    )
    parser.add_argument(
        "--ranking-models",
        default="all",
        help="Comma-separated base family names used only to select cases, or 'all'.",
    )
    parser.add_argument(
        "--ranking-modes",
        default="all",
        help="Comma-separated evaluated modes used only to select cases, or 'all'.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--devices", default="cuda:0,cuda:1,cuda:2,cuda:3,cuda:4,cuda:5")
    parser.add_argument("--surface-query-points", type=int, default=32768)
    parser.add_argument("--volume-query-points", type=int, default=32768)
    parser.add_argument(
        "--active-geometry-sources",
        default="angle,isotropic",
        help="Comma-separated remeshing methods. Voxel clustering is excluded by default because its achieved toy-mesh reductions are not calibrated to the requested factors.",
    )
    parser.add_argument("--geometry-decimation-factors", default="5,10")
    parser.add_argument("--angle-decimated-vtp-dir", default="/mnt/ssdraid/parsa/toy_heat_exchange_surface_vtp_angle")
    parser.add_argument("--isotropic-decimated-vtp-dir", default="/mnt/ssdraid/parsa/toy_heat_exchange_surface_vtp_isotropic")
    parser.add_argument("--voxel-decimated-vtp-dir", default="/mnt/ssdraid/parsa/toy_heat_exchange_surface_vtp_voxel")
    parser.add_argument("--original-vtp-dir", default="/mnt/ssdraid/parsa/toy_heat_exchange_surface_vtp")
    parser.add_argument(
        "--geometry-label-preset",
        choices=("legacy", "v4"),
        default="legacy",
        help="Use 'v4' for feature-aware, QEM, and voxel-grid-clustering remesh inputs.",
    )
    parser.add_argument("--font-scale", type=float, default=1.2)
    parser.add_argument("--analysis-case-count", type=int, default=1)
    parser.add_argument("--output-dir", required=True)
    for key in MODEL_SPECS:
        parser.add_argument(f"--{key.lower().replace('_', '-')}-checkpoint", type=Path, required=True)
    args = parser.parse_args()
    if args.geometry_label_preset == "v4":
        SOURCE_LABELS = dict(V4_SOURCE_LABELS)

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_args = {name: getattr(args, f"{name.lower().replace('_', '_')}_checkpoint") for name in MODEL_SPECS}
    missing_checkpoints = [str(path) for path in checkpoint_args.values() if not path.is_file()]
    if missing_checkpoints:
        raise FileNotFoundError("Missing checkpoints:\n" + "\n".join(missing_checkpoints))
    devices = [torch.device(item) for item in parse_csv(args.devices)]
    if not devices:
        raise ValueError("At least one --devices entry is required.")
    manifest = json.loads((Path(args.data_root).expanduser().resolve() / "preprocessed_manifest.json").read_text(encoding="utf-8"))
    if args.candidate_split == "validation":
        candidate_case_ids = [int(case_id) for case_id in manifest["validation_ids"]]
    elif args.candidate_split == "train":
        candidate_case_ids = [int(case_id) for case_id in manifest["train_ids"]]
    else:
        candidate_case_ids = sorted({
            *(int(case_id) for case_id in manifest["train_ids"]),
            *(int(case_id) for case_id in manifest["validation_ids"]),
        })
    dataset = ToyHeatExchangeDataset(
        args.data_root,
        if_test=True,
        geometry_points=0,
        surface_points=args.surface_query_points,
        volume_points=args.volume_query_points,
        return_geometry_density=True,
        case_ids=candidate_case_ids,
    )
    common_indices: list[int] = []
    expected_sources = len(parse_csv(args.active_geometry_sources)) * len(parse_csv(args.geometry_decimation_factors, int))
    for index in range(len(dataset)):
        if len(vtp_paths(int(dataset.data[index]), args)) == expected_sources:
            common_indices.append(index)
    if len(common_indices) < args.candidate_pool_size:
        raise RuntimeError(
            f"Requested a pool of {args.candidate_pool_size} complete remeshed cases from split={args.candidate_split}, "
            f"but only {len(common_indices)} have every requested VTP source."
        )
    rng = np.random.default_rng(args.seed)
    pool_indices = sorted(rng.choice(np.asarray(common_indices), size=args.candidate_pool_size, replace=False).tolist())
    if args.top_k <= 0 or args.top_k > len(pool_indices):
        raise ValueError("--top-k must be between one and --candidate-pool-size.")
    available_modes = [
        "beta", "sine_x", "sine_y",
        *[f"{method}_div{factor}" for method in parse_csv(args.active_geometry_sources) for factor in parse_csv(args.geometry_decimation_factors, int)],
    ]
    if str(args.ranking_models).strip().lower() == "all":
        ranking_families = list(FAMILY_ORDER)
    else:
        ranking_families = parse_csv(args.ranking_models)
        invalid_families = sorted(set(ranking_families) - set(FAMILY_ORDER))
        if invalid_families:
            raise ValueError(f"Unknown --ranking-models family names: {invalid_families}. Available: {list(FAMILY_ORDER)}")
    if str(args.ranking_modes).strip().lower() == "all":
        ranking_modes = available_modes
    else:
        ranking_modes = parse_csv(args.ranking_modes)
        invalid_modes = sorted(set(ranking_modes) - set(available_modes))
        if invalid_modes:
            raise ValueError(f"Unknown --ranking-modes: {invalid_modes}. Available: {available_modes}")

    models: dict[str, Any] = {}
    model_devices: dict[str, torch.device] = {}
    model_configs = {}
    for index, (name, checkpoint) in enumerate(checkpoint_args.items()):
        device = devices[(index // 2) % len(devices)]
        models[name], model_configs[name] = load_model(name, checkpoint.resolve(), device)
        model_devices[name] = device
    budgets = {name: model_input_budget(cfg) for name, cfg in model_configs.items()}
    if len(set(budgets.values())) != 1:
        raise RuntimeError(f"All compared models must have an identical train-aligned encoder budget; got {budgets}.")
    input_points = next(iter(budgets.values()))
    if args.surface_query_points != 32768 or args.volume_query_points != 32768:
        raise ValueError("This comparison is fixed to the common training query budgets: 32768 surface and 32768 volume points.")

    print(f"Inference devices: {', '.join(map(str, devices))}")
    print(f"Candidate pool: {len(pool_indices)} complete remeshing cases from split={args.candidate_split}; selecting top {args.top_k} internally.")
    print(f"Shared train-aligned encoder budget: {input_points} points.")
    candidate_rows: dict[int, list[dict[str, Any]]] = {}
    for index in tqdm(pool_indices, desc="Ranking remeshed candidates"):
        candidate_rows[index] = evaluate_case(index, dataset, models, model_devices, args, input_points)
    ranked = sorted(
        pool_indices,
        key=lambda index: candidate_score(candidate_rows[index], ranking_families, ranking_modes),
        reverse=True,
    )
    selected_indices = ranked[:args.top_k]
    rows = [record for index in selected_indices for record in candidate_rows[index]]
    aggregate = aggregate_rows(rows)
    fields = ["case_id", "model_name", "sampling_mode", "surface_global_rel_l2", "volume_global_rel_l2", "combined_global_rel_l2"]
    with (output_dir / "per_run_mode_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / "aggregate_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model_name", "sampling_mode", "surface_global_rel_l2", "volume_global_rel_l2", "combined_global_rel_l2"])
        writer.writeheader()
        writer.writerows(aggregate)

    for mode, title_name in (("beta", "beta"), ("sine_x", "sine-x"), ("sine_y", "sine-y")):
        for scale in ("linear", "log"):
            plot_endpoint_bars(
                aggregate,
                mode,
                output_dir / f"all_models_combined_global_endpoint_bars_{mode}_{scale}.png",
                f"Heat Exchanger: {title_name} endpoint",
                scale,
                args.font_scale,
            )
    geometry_modes = [f"{method}_div{factor}" for method in parse_csv(args.active_geometry_sources) for factor in parse_csv(args.geometry_decimation_factors, int)]
    for scale in ("linear", "log"):
        plot_geometry_factor_average_bars(aggregate, geometry_modes, output_dir / f"all_models_combined_global_geometry_sources_bars_{scale}.png", "Heat Exchanger: mean remeshing by decimation factor", scale, args.font_scale)
    summary_modes = ["beta", "sine_x", "sine_y", *geometry_modes]
    write_summary_table(aggregate, summary_modes, output_dir / "all_models_combined_global_deal_endpoint_improvement.csv")
    write_improvement_markdown(aggregate, summary_modes, output_dir / "all_models_combined_global_deal_endpoint_improvement.md")
    plot_satloss_improvements(aggregate, summary_modes, output_dir / "all_models_combined_global_deal_endpoint_improvement.png", args.font_scale)
    plot_density_validation(dataset, selected_indices[0], input_points, args, output_dir / "density_shift_validation.png")
    compact_summary = {
        "benchmark": "heat_exchanger",
        "metric": "combined_global_rel_l2",
        "models": [MODEL_SPECS[name]["label"] for name in MODEL_SPECS],
        "conditions": summary_modes,
        "encoder_input_points": input_points,
        "surface_query_points": args.surface_query_points,
        "volume_query_points": args.volume_query_points,
    }
    (output_dir / "results.json").write_text(json.dumps(compact_summary, indent=2) + "\n", encoding="utf-8")
    (output_dir / "summary.md").write_text(
        "# Heat-Exchanger Sampling-Invariance Comparison\n\n"
        "Each base/DeAL pair uses its checkpoint-matched architecture, its common 65,536-point encoder budget, and identical 32,768-point surface and volume query clouds. "
        "Only the encoder cloud changes under beta, sine, or remeshed-input conditions.\n",
        encoding="utf-8",
    )
    (output_dir / "workflow.md").write_text(
        "Positive beta and sine conditions are sampled from the native surface cloud using KDE-16. "
        "VTP conditions are sampled uniformly from remeshed triangles and receive no secondary density shift.\n",
        encoding="utf-8",
    )
    export_analysis(dataset, selected_indices, models, model_devices, args, input_points, output_dir)
    print(f"Saved the selected-case comparison to {output_dir}")


if __name__ == "__main__":
    main()
