#!/usr/bin/env python3
"""Evaluate the analytic toy benchmark under matched point-density shifts.

This script evaluates identical unbiased query clouds for SMART Base and
SMART SATLOSS.  Only the encoder input distribution changes.  It writes tidy
per-case data, endpoint/curve figures, and a compact protocol record suitable
for supplementary material.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from data.toy_satloss_dataset import ToySATLossDataset
from data.toy_perforated_fin_dataset import ToyPerforatedFinDataset
from models.smart.smart import SMART
from train_consistency_common import sample_geometry_view


SOURCE_LABELS = {
    "angle_div5": "Angle 5x",
    "angle_div10": "Angle 10x",
    "isotropic_div5": "Isotropic 5x",
    "isotropic_div10": "Isotropic 10x",
    "voxel_div5": "Voxel 5x",
    "voxel_div10": "Voxel 10x",
}


def parse_csv(value: str, cast=float):
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def parse_ids(value: str) -> list[int]:
    return [int(item.strip()) for item in str(value).split(",") if item.strip()]


def device_for(index: int, devices: list[str]) -> torch.device:
    value = devices[index % len(devices)]
    return torch.device(value if torch.cuda.is_available() else "cpu")


def build_model(config_path: Path, checkpoint_path: Path, device: torch.device) -> SMART:
    cfg = OmegaConf.load(config_path)
    # SATLOSS config files inherit the base toy configuration through Hydra.
    # Reconstruct that small composition here without starting a Hydra job.
    if "architecture" not in cfg.experiment:
        base_path = config_path.with_name(
            "toy_perforated_fin.yaml" if "perforated_fin" in config_path.name else "toy_satloss.yaml"
        )
        cfg = OmegaConf.merge(OmegaConf.load(base_path), cfg)
    exp = cfg.experiment
    arch = exp.architecture
    model = SMART(
        spatial_dim=3, surface_channels=1, volume_channels=1, parameter_channels=0,
        latent_dim=int(arch.latent_dim), latent_geometry_points=int(arch.latent_geometry_points),
        subsampled_geometry_points=int(arch.subsampled_geometry_points),
        subsampled_geometry_with_replacement=bool(arch.subsampled_geometry_with_replacement),
        num_encoder_decoder_blocks=int(arch.num_encoder_decoder_blocks),
        pos_scale_factor=float(arch.pos_scale_factor),
    ).to(device).eval()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state = checkpoint.get("model_state_dict", checkpoint)
    target = model.state_dict()
    compatible = {key: value for key, value in state.items() if key in target and target[key].shape == value.shape}
    if len(compatible) != len(target):
        missing = len(target) - len(compatible)
        raise RuntimeError(f"{checkpoint_path} is incompatible with the toy architecture: {missing} tensors missing.")
    model.load_state_dict(compatible, strict=True)
    return model


def select_indices(geo: torch.Tensor, density: torch.Tensor, budget: int, shift: str, level: float, seed: int) -> torch.Tensor:
    if shift == "beta":
        mode, beta, axis, mix = "inverse_density_wor", level, None, 0.0
    elif shift == "sine_x":
        mode, beta, axis, mix = "sinusoidal_axis_mixture_wor", 0.0, 0, level
    elif shift == "sine_y":
        mode, beta, axis, mix = "sinusoidal_axis_mixture_wor", 0.0, 1, level
    else:
        raise ValueError(f"Unsupported shift: {shift}")
    _, _, _, indices = sample_geometry_view(
        geo.unsqueeze(0), density.unsqueeze(0), budget, mode, beta, 1.0, seed,
        sinusoidal_axis=axis, sinusoidal_mix_fraction=mix, return_indices=True,
    )
    return indices[0]


def rel_l2(pred: torch.Tensor, target: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(pred - target) / torch.linalg.vector_norm(target).clamp_min(1.0e-12))


def load_vtp_mesh(path: Path) -> tuple[np.ndarray, np.ndarray]:
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy

    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(str(path))
    reader.Update()
    polydata = reader.GetOutput()
    if polydata is None or polydata.GetPoints() is None or polydata.GetNumberOfPoints() == 0:
        raise RuntimeError(f"Invalid or empty remeshed VTP: {path}")
    points = np.asarray(vtk_to_numpy(polydata.GetPoints().GetData()), dtype=np.float32)
    raw = np.asarray(vtk_to_numpy(polydata.GetPolys().GetData()), dtype=np.int64)
    if raw.size == 0 or raw.size % 4 != 0 or not np.all(raw[::4] == 3):
        raise RuntimeError(f"Remeshed VTP is not a triangle mesh: {path}")
    faces = raw.reshape(-1, 4)[:, 1:]
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all() or faces.max(initial=-1) >= points.shape[0]:
        raise RuntimeError(f"Invalid point coordinates in {path}")
    return np.ascontiguousarray(points), np.ascontiguousarray(faces)


def sample_remeshed_encoder_points(path: Path, budget: int, seed: int, mode: str) -> np.ndarray:
    """Create a fixed-size encoder cloud from a remeshed triangular surface."""
    points, faces = load_vtp_mesh(path)
    rng = np.random.default_rng(seed)
    if mode == "vertices":
        chosen = rng.choice(points.shape[0], size=min(int(budget), points.shape[0]), replace=False)
        return np.ascontiguousarray(points[chosen], dtype=np.float32)
    if mode != "triangle_uniform":
        raise ValueError(f"Unknown --remesh-input-sampling mode {mode!r}.")
    # This is exactly the native-cloud convention used by the fin generator:
    # triangles receive equal probability, then points are uniform inside them.
    chosen_faces = rng.integers(0, faces.shape[0], size=int(budget), endpoint=False)
    vertices = points[faces[chosen_faces]]
    weights = -np.log(np.maximum(rng.random((int(budget), 3)), 1.0e-12))
    weights /= weights.sum(axis=1, keepdims=True)
    return np.ascontiguousarray(np.einsum("ni,nij->nj", weights, vertices), dtype=np.float32)


def vtp_stem(case_id: int, benchmark: str) -> str:
    if benchmark == "perforated_fin":
        return f"perforated_fin_case_{case_id:05d}_surface"
    return f"toy_case_{case_id:05d}_surface"


def vtp_source_paths(case_id: int, args: argparse.Namespace) -> dict[str, Path]:
    sources: dict[str, Path] = {}
    root_map = {
        "angle": Path(args.angle_decimated_vtp_dir),
        "isotropic": Path(args.isotropic_decimated_vtp_dir),
        "voxel": Path(args.voxel_decimated_vtp_dir),
    }
    for method in parse_csv(args.active_geometry_sources, str):
        if method not in root_map:
            raise ValueError(f"Unknown geometry source {method!r}; expected angle,isotropic,voxel.")
        for factor in parse_csv(args.geometry_decimation_factors, int):
            name = f"{vtp_stem(case_id, args.benchmark)}_faces_div{factor}.vtp"
            path = root_map[method].expanduser().resolve() / f"case_{case_id:05d}" / name
            if path.is_file():
                sources[f"{method}_div{factor}"] = path
    return sources


def predict_errors(model: SMART, device: torch.device, geo: torch.Tensor, surf_q: torch.Tensor, surf_y: torch.Tensor, vol_q: torch.Tensor, vol_y: torch.Tensor, dataset: ToySATLossDataset) -> tuple[float, float, float]:
    with torch.inference_mode(), torch.autocast(device_type=device.type, enabled=device.type == "cuda", dtype=torch.float16):
        pred_s, pred_v = model.inference(geo.unsqueeze(0).to(device), surf_q.unsqueeze(0).to(device), vol_q.unsqueeze(0).to(device), None)
    pred_s = pred_s[0].float().cpu() * dataset.std_surf_data + dataset.mean_surf_data
    pred_v = pred_v[0].float().cpu() * dataset.std_vol_data + dataset.mean_vol_data
    true_s = surf_y * dataset.std_surf_data + dataset.mean_surf_data
    true_v = vol_y * dataset.std_vol_data + dataset.mean_vol_data
    surf_error, vol_error = rel_l2(pred_s, true_s), rel_l2(pred_v, true_v)
    return surf_error, vol_error, 0.5 * (surf_error + vol_error)


def predict_surface_field(model: SMART, device: torch.device, geo: torch.Tensor, surf_q: torch.Tensor, vol_q: torch.Tensor, dataset: ToySATLossDataset) -> np.ndarray:
    with torch.inference_mode(), torch.autocast(device_type=device.type, enabled=device.type == "cuda", dtype=torch.float16):
        pred_s, _ = model.inference(geo.unsqueeze(0).to(device), surf_q.unsqueeze(0).to(device), vol_q.unsqueeze(0).to(device), None)
    return np.asarray((pred_s[0].float().cpu() * dataset.std_surf_data + dataset.mean_surf_data).numpy(), dtype=np.float32)


def write_point_vtp(path: Path, points: np.ndarray, arrays: dict[str, np.ndarray]) -> None:
    """Write a compact vertex VTP for direct ParaView field comparison."""
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
        raise RuntimeError(f"Failed to write analysis VTP: {path}")


def export_analysis_vtps(dataset: ToySATLossDataset, selected: list[int], models: dict[str, SMART], model_devices: dict[str, torch.device], args: argparse.Namespace, output_dir: Path) -> None:
    if not args.save_analysis_vtps:
        return
    if args.analysis_case_ids:
        requested_ids = parse_ids(args.analysis_case_ids)
        index_by_id = {int(dataset.data[index]): index for index in selected}
        missing = sorted(set(requested_ids) - set(index_by_id))
        if missing:
            raise ValueError(f"--analysis-case-ids must be among evaluated case IDs; invalid: {missing}")
        requested = [index_by_id[case_id] for case_id in requested_ids]
    else:
        requested = selected[: max(0, args.analysis_case_count)]
    if not requested:
        return
    analysis_root = output_dir / "analysis_vtps"
    original_root = Path(args.original_vtp_dir).expanduser().resolve()
    for case_index in requested:
        case_id = int(dataset.data[case_index])
        case_dir = analysis_root / f"case_{case_id:05d}"
        case_dir.mkdir(parents=True, exist_ok=True)
        original_path = original_root / f"case_{case_id:05d}" / f"{vtp_stem(case_id, args.benchmark)}.vtp"
        if not original_path.is_file():
            raise FileNotFoundError(f"Missing original triangular VTP for analysis export: {original_path}")
        shutil.copy2(original_path, case_dir / "original_surface_mesh.vtp")

        geo, surf_q, surf_y, vol_q, vol_y, density = dataset[case_index]
        true_surface = np.asarray((surf_y * dataset.std_surf_data + dataset.mean_surf_data).numpy(), dtype=np.float32)
        conditions: list[tuple[str, torch.Tensor, Path | None]] = []
        for shift in ("sine_x", "sine_y"):
            indices = select_indices(geo, density, args.input_points, shift, 1.0, args.seed + 100_003 * case_index + 10_000)
            conditions.append((f"{shift}_1", geo.index_select(0, indices), None))
        for source, path in vtp_source_paths(case_id, args).items():
            shutil.copy2(path, case_dir / f"{source}_input_mesh.vtp")
            physical = sample_remeshed_encoder_points(
                path, args.input_points, args.seed + case_id + sum(map(ord, source)), args.remesh_input_sampling,
            )
            normalized = (physical - dataset.min_pos.numpy()[None, :]) / dataset.position_span.numpy()[None, :]
            conditions.append((source, torch.from_numpy(np.ascontiguousarray(normalized, dtype=np.float32)), path))
        for name, geo_view, _ in conditions:
            physical_encoder = geo_view.numpy() * dataset.position_span.numpy()[None, :] + dataset.min_pos.numpy()[None, :]
            physical_queries = surf_q.numpy() * dataset.position_span.numpy()[None, :] + dataset.min_pos.numpy()[None, :]
            write_point_vtp(case_dir / f"{name}_encoder_input_points.vtp", physical_encoder, {"encoder_point": np.ones((geo_view.shape[0], 1), dtype=np.float32)})
            smart = predict_surface_field(models["SMART Base"], model_devices["SMART Base"], geo_view, surf_q, vol_q, dataset)
            satloss = predict_surface_field(models["SMART SATLOSS"], model_devices["SMART SATLOSS"], geo_view, surf_q, vol_q, dataset)
            write_point_vtp(
                case_dir / f"{name}_surface_predictions.vtp",
                physical_queries,
                {
                    "ground_truth": true_surface,
                    "smart_prediction": smart,
                    "smart_absolute_error": np.abs(smart - true_surface),
                    "satloss_prediction": satloss,
                    "satloss_absolute_error": np.abs(satloss - true_surface),
                },
            )
        (case_dir / "README.txt").write_text(
            "Original/remeshed meshes are triangular VTPs. Prediction VTPs are surface-query point clouds with physical-unit ground truth, SMART/SATLOSS predictions, and absolute errors.\n",
            encoding="utf-8",
        )


def plot_curves(rows: list[dict], output_dir: Path, font_scale: float) -> None:
    colors = {"SMART Base": "#6B7280", "SMART SATLOSS": "#1F77B4"}
    for shift in ("beta", "sine_x", "sine_y"):
        fig, ax = plt.subplots(figsize=(9.5, 5.4), constrained_layout=True)
        for model in colors:
            subset = [row for row in rows if row["model"] == model and row["shift"] == shift and row["source"] == "native"]
            levels = sorted(set(float(row["level"]) for row in subset))
            means = [np.mean([float(row["global_rel_l2"]) for row in subset if float(row["level"]) == level]) for level in levels]
            ax.plot(levels, means, marker="o", linewidth=2.8, markersize=7, color=colors[model], label=model)
        ax.set_xlabel("Shift intensity", fontsize=12 * font_scale)
        ax.set_ylabel("Global relative L2 error", fontsize=12 * font_scale)
        ax.set_title(f"Toy benchmark: {shift.replace('_', ' ')} sampling shift", fontsize=14 * font_scale, weight="bold")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=11 * font_scale)
        ax.tick_params(labelsize=10 * font_scale)
        fig.savefig(output_dir / f"toy_sampling_curve_{shift}.png", dpi=260)
        fig.savefig(output_dir / f"toy_sampling_curve_{shift}.pdf")
        plt.close(fig)


def plot_endpoint_bars(rows: list[dict], output_dir: Path, font_scale: float) -> None:
    shifts = [("beta", "Beta 1"), ("sine_x", "Sine x 1"), ("sine_y", "Sine y 1")]
    labels = [label for _, label in shifts]
    base = []
    sat = []
    for shift, _ in shifts:
        base.append(np.mean([float(row["global_rel_l2"]) for row in rows if row["model"] == "SMART Base" and row["shift"] == shift and float(row["level"]) == 1.0]))
        sat.append(np.mean([float(row["global_rel_l2"]) for row in rows if row["model"] == "SMART SATLOSS" and row["shift"] == shift and float(row["level"]) == 1.0]))
    x = np.arange(len(labels)); width = 0.34
    for logarithmic in (False, True):
        fig, ax = plt.subplots(figsize=(10, 5.6), constrained_layout=True)
        bars_base = ax.bar(x - width / 2, base, width, label="SMART Base", color="#6B7280", edgecolor="#202124")
        bars_sat = ax.bar(x + width / 2, sat, width, label="SMART SATLOSS", color="#1F77B4", edgecolor="#202124", hatch="///")
        for index, (base_value, sat_value) in enumerate(zip(base, sat)):
            improvement = 100.0 * (base_value - sat_value) / max(base_value, 1.0e-12)
            ax.annotate(f"{improvement:+.1f}%", (bars_sat[index].get_x() + width / 2, sat_value), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=10 * font_scale, weight="bold")
        ax.set_xticks(x, labels, fontsize=11 * font_scale)
        ax.set_ylabel("Global relative L2 error", fontsize=12 * font_scale)
        ax.set_title("Toy benchmark point-cloud-shift robustness", fontsize=14 * font_scale, weight="bold")
        if logarithmic:
            ax.set_yscale("log")
        ax.legend(fontsize=11 * font_scale)
        ax.tick_params(axis="y", labelsize=10 * font_scale)
        fig.savefig(output_dir / f"toy_sampling_endpoint_bars_{'log' if logarithmic else 'linear'}.png", dpi=260)
        fig.savefig(output_dir / f"toy_sampling_endpoint_bars_{'log' if logarithmic else 'linear'}.pdf")
        plt.close(fig)


def paired_groups(rows: list[dict], groups: list[tuple[str, list[str]]]) -> list[tuple[str, float, float]]:
    """Aggregate paired SMART/SATLOSS global errors for paper-facing endpoint bars."""
    output = []
    for label, selectors in groups:
        subset = [row for row in rows if row["source"] in selectors]
        base = [float(row["global_rel_l2"]) for row in subset if row["model"] == "SMART Base"]
        sat = [float(row["global_rel_l2"]) for row in subset if row["model"] == "SMART SATLOSS"]
        if not base or not sat:
            raise RuntimeError(f"Missing paired results for plot group {label!r}.")
        output.append((label, float(np.mean(base)), float(np.mean(sat))))
    return output


def plot_paired_endpoint_bars(groups: list[tuple[str, float, float]], title: str, stem: str, output_dir: Path, font_scale: float) -> None:
    """Match the established endpoint-bar grammar: paired bars and SATLOSS-only labels."""
    x = np.arange(len(groups), dtype=np.float64)
    width = 0.34
    base_values = [group[1] for group in groups]
    sat_values = [group[2] for group in groups]
    for logarithmic in (False, True):
        fig, ax = plt.subplots(figsize=(11.8, 5.8))
        fig.subplots_adjust(left=0.13, right=0.97, bottom=0.24, top=0.86)
        ax.bar(x - width / 2, base_values, width=width, color="#6B7280", edgecolor="#222222", linewidth=0.7, label="SMART")
        bars_sat = ax.bar(x + width / 2, sat_values, width=width, color="#1F77B4", edgecolor="#222222", linewidth=0.7, hatch="///", label="SATLOSS")
        finite = np.asarray(base_values + sat_values, dtype=np.float64)
        if logarithmic:
            ax.set_yscale("log")
            ax.set_ylim(max(float(finite.min()) * 0.90, 1.0e-12), float(finite.max()) * 1.10)
        else:
            span = max(float(finite.max() - finite.min()), float(finite.max()), 1.0e-12)
            ax.set_ylim(max(0.0, float(finite.min()) - 0.10 * span), float(finite.max()) + 0.10 * span)
        for bar, (_, base, sat) in zip(bars_sat, groups):
            reduction = 100.0 * (base - sat) / max(abs(base), 1.0e-12)
            y = sat * 1.012 if logarithmic else sat + max(0.005 * span, 1.0e-6)
            ax.text(bar.get_x() + bar.get_width() / 2.0, y, f"{reduction:+.1f}%", ha="center", va="bottom", fontsize=11 * font_scale, fontweight="bold", clip_on=False)
        ax.set_xticks(x, [group[0] for group in groups])
        ax.set_ylabel(f"Combined global relative L2 ({'log' if logarithmic else 'linear'} scale)", fontsize=13 * font_scale)
        ax.yaxis.set_label_coords(-0.10, 0.5)
        ax.set_title(title, fontsize=14 * font_scale, pad=12)
        ax.tick_params(axis="both", labelsize=12 * font_scale)
        ax.grid(axis="y", which="both", alpha=0.20)
        ax.set_axisbelow(True)
        ax.legend(loc="upper right", fontsize=11 * font_scale, framealpha=0.92)
        fig.savefig(output_dir / f"{stem}_{'log' if logarithmic else 'linear'}.png", dpi=280, bbox_inches="tight", pad_inches=0.16)
        fig.savefig(output_dir / f"{stem}_{'log' if logarithmic else 'linear'}.pdf", bbox_inches="tight", pad_inches=0.16)
        plt.close(fig)


def write_endpoint_table(rows: list[dict], output_dir: Path) -> None:
    groups = []
    for shift, label in (("beta", "Beta 1"), ("sine_x", "Sine x 1"), ("sine_y", "Sine y 1")):
        groups.append((label, ["native:" + shift]))
    for source in SOURCE_LABELS:
        if any(row["source"] == source for row in rows):
            groups.append((SOURCE_LABELS[source], [source]))
    fields = ["condition", "smart_global_rel_l2", "satloss_global_rel_l2", "satloss_reduction_vs_smart_percent"]
    with (output_dir / "toy_combined_global_endpoint_table.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for label, selectors in groups:
            if selectors[0].startswith("native:"):
                shift = selectors[0].split(":", 1)[1]
                subset = [row for row in rows if row["source"] == "native" and row["shift"] == shift and abs(float(row["level"]) - 1.0) < 1.0e-8]
                base = float(np.mean([float(row["global_rel_l2"]) for row in subset if row["model"] == "SMART Base"]))
                sat = float(np.mean([float(row["global_rel_l2"]) for row in subset if row["model"] == "SMART SATLOSS"]))
            else:
                label, base, sat = paired_groups(rows, [(label, selectors)])[0]
            writer.writerow({"condition": label, "smart_global_rel_l2": base, "satloss_global_rel_l2": sat, "satloss_reduction_vs_smart_percent": 100.0 * (base - sat) / max(abs(base), 1.0e-12)})


def native_endpoint_group(rows: list[dict], shift: str, endpoint: float, label: str) -> tuple[str, float, float]:
    subset = [row for row in rows if row["source"] == "native" and row["shift"] == shift and abs(float(row["level"]) - endpoint) < 1.0e-8]
    base = [float(row["global_rel_l2"]) for row in subset if row["model"] == "SMART Base"]
    sat = [float(row["global_rel_l2"]) for row in subset if row["model"] == "SMART SATLOSS"]
    if not base or not sat:
        raise RuntimeError(f"Missing native endpoint results for {shift}={endpoint:g}.")
    return label, float(np.mean(base)), float(np.mean(sat))


def render_endpoint_plots(rows: list[dict], output_dir: Path, levels: list[float], methods: list[str], factors: list[int], font_scale: float) -> None:
    endpoint = max(levels)
    for shift, label in (("beta", "Beta shift"), ("sine_x", "Sine-x shift"), ("sine_y", "Sine-y shift")):
        groups = [native_endpoint_group(rows, shift, endpoint, f"{label} {endpoint:g}")]
        plot_paired_endpoint_bars(groups, f"Toy benchmark: {label} endpoint", f"{shift}_combined_global_rel_l2_endpoint_absolute", output_dir, font_scale)

    for method in methods:
        method_groups = [(f"div{factor}", [f"{method}_div{factor}"]) for factor in factors]
        plot_paired_endpoint_bars(
            paired_groups(rows, method_groups),
            f"Toy benchmark: {method.title()} remeshing",
            f"geometry_{method}_combined_global_rel_l2_endpoint_absolute",
            output_dir,
            font_scale,
        )
    average_groups = [(f"Average div{factor}", [f"{method}_div{factor}" for method in methods]) for factor in factors]
    plot_paired_endpoint_bars(
        paired_groups(rows, average_groups),
        "Toy benchmark: average remeshing",
        "remeshing_average_combined_global_rel_l2_endpoint_absolute",
        output_dir,
        font_scale,
    )
    combined_groups = [
        native_endpoint_group(rows, "sine_x", endpoint, "Sine-x 1"),
        native_endpoint_group(rows, "sine_y", endpoint, "Sine-y 1"),
        *paired_groups(rows, [(f"Mean div{factor}", [f"{method}_div{factor}" for method in methods]) for factor in factors]),
    ]
    plot_paired_endpoint_bars(
        combined_groups,
        "Toy benchmark: sampling and remeshing shifts",
        "combined_global_rel_l2_endpoint_absolute",
        output_dir,
        font_scale,
    )
    write_endpoint_table(rows, output_dir)


def plot_density(case: tuple, output_dir: Path, budget: int) -> None:
    geo, _, _, _, _, density = case
    fig, ax = plt.subplots(figsize=(9, 5.2), constrained_layout=True)
    for shift, level, color in (("beta", 0.0, "#777777"), ("beta", 1.0, "#1f77b4"), ("sine_x", 1.0, "#d62728"), ("sine_y", 1.0, "#2ca02c")):
        idx = select_indices(geo, density, budget, shift, level, 42 + int(level * 100) + len(shift))
        ax.hist(density[idx].numpy(), bins=48, density=True, histtype="step", linewidth=2.2, color=color, label=f"{shift}, {level:g}")
    ax.set_xlabel("KDE-16 log density", fontsize=12)
    ax.set_ylabel("Probability density", fontsize=12)
    ax.set_title("Encoder-cloud density under each toy shift", fontsize=14, weight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.2)
    fig.savefig(output_dir / "toy_input_density_shift_histogram.png", dpi=260)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=("poisson", "perforated_fin"), default="poisson")
    parser.add_argument("--data-root", default="/mnt/ssdraid/parsa/toy_satloss_poisson_benchmark_v2")
    parser.add_argument("--base-checkpoint", default=None)
    parser.add_argument("--satloss-checkpoint", default=None)
    parser.add_argument("--base-config", default="/home/parsa/smart_parsa/smart/config/toy_satloss.yaml")
    parser.add_argument("--satloss-config", default="/home/parsa/smart_parsa/smart/config/toy_satloss7.yaml")
    parser.add_argument("--output-dir", default="/home/parsa/smart_parsa/results/toy_satloss_sampling_invariance")
    parser.add_argument("--levels", default="0,0.25,0.5,0.75,1")
    parser.add_argument("--input-points", type=int, default=16384)
    parser.add_argument("--surface-query-points", type=int, default=16384)
    parser.add_argument("--volume-query-points", type=int, default=16384)
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--devices", default="cuda:0,cuda:1")
    parser.add_argument("--font-scale", type=float, default=1.2)
    parser.add_argument("--plot-only-results-csv", type=Path, default=None, help="Regenerate endpoint plots from an existing CSV without inference.")
    parser.add_argument("--active-geometry-sources", default="angle,isotropic,voxel")
    parser.add_argument("--geometry-decimation-factors", default="5,10")
    parser.add_argument("--angle-decimated-vtp-dir", default="/mnt/ssdraid/parsa/toy_satloss_surface_vtp_angle_decimated")
    parser.add_argument("--isotropic-decimated-vtp-dir", default="/mnt/ssdraid/parsa/toy_satloss_surface_vtp_isotropic_remeshed")
    parser.add_argument("--voxel-decimated-vtp-dir", default="/mnt/ssdraid/parsa/toy_satloss_surface_vtp_voxel_quadric_clustered")
    parser.add_argument("--original-vtp-dir", default="/mnt/ssdraid/parsa/toy_satloss_surface_vtp")
    parser.add_argument("--remesh-input-sampling", choices=("vertices", "triangle_uniform"), default="vertices")
    parser.add_argument("--save-analysis-vtps", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--analysis-only", action="store_true", help="Export analysis_vtps without recomputing CSV metrics or plots.")
    parser.add_argument("--analysis-case-count", type=int, default=1, help="Number of evaluated cases exported to analysis_vtps.")
    parser.add_argument("--analysis-case-ids", default="", help="Optional comma-separated evaluated case IDs for analysis VTP export.")
    args = parser.parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve(); output_dir.mkdir(parents=True, exist_ok=True)
    levels = parse_csv(args.levels)
    requested_methods = parse_csv(args.active_geometry_sources, str)
    requested_factors = parse_csv(args.geometry_decimation_factors, int)
    if args.plot_only_results_csv is not None:
        with args.plot_only_results_csv.expanduser().resolve().open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            raise ValueError(f"No rows found in {args.plot_only_results_csv}")
        render_endpoint_plots(rows, output_dir, levels, requested_methods, requested_factors, args.font_scale)
        print(f"Regenerated endpoint plots from {args.plot_only_results_csv}")
        return
    if not args.base_checkpoint or not args.satloss_checkpoint:
        parser.error("--base-checkpoint and --satloss-checkpoint are required unless --plot-only-results-csv is used.")
    devices = [value.strip() for value in args.devices.split(",") if value.strip()]
    dataset_class = ToyPerforatedFinDataset if args.benchmark == "perforated_fin" else ToySATLossDataset
    dataset = dataset_class(args.data_root, if_test=True, geometry_points=0, surface_points=args.surface_query_points, volume_points=args.volume_query_points, return_geometry_density=True)
    selected = list(range(len(dataset))) if args.max_cases <= 0 else list(range(min(args.max_cases, len(dataset))))
    if requested_methods:
        missing_vtps = []
        for case_index in selected:
            case_id = int(dataset.data[case_index])
            present = vtp_source_paths(case_id, args)
            for method in requested_methods:
                for factor in requested_factors:
                    source = f"{method}_div{factor}"
                    if source not in present:
                        missing_vtps.append(source + ":" + str(case_id))
        if missing_vtps:
            preview = ", ".join(missing_vtps[:8])
            raise FileNotFoundError(
                "Requested remeshing VTP sources are missing. Generate and decimate every selected "
                f"validation case first. Examples: {preview}"
            )
    models = {
        "SMART Base": build_model(Path(args.base_config), Path(args.base_checkpoint), device_for(0, devices)),
        "SMART SATLOSS": build_model(Path(args.satloss_config), Path(args.satloss_checkpoint), device_for(1, devices)),
    }
    model_devices = {name: next(model.parameters()).device for name, model in models.items()}
    if args.analysis_only:
        export_analysis_vtps(dataset, selected, models, model_devices, args, output_dir)
        print(f"Saved analysis VTPs to {output_dir / 'analysis_vtps'}")
        return
    rows: list[dict] = []
    first_case = None
    for case_index in tqdm(selected, desc="Toy cases"):
        geo, surf_q, surf_y, vol_q, vol_y, density = dataset[case_index]
        if first_case is None:
            first_case = (geo, surf_q, surf_y, vol_q, vol_y, density)
        for shift in ("beta", "sine_x", "sine_y"):
            for level in levels:
                indices = select_indices(geo, density, args.input_points, shift, level, args.seed + 100_003 * case_index + int(level * 10_000))
                geo_view = geo.index_select(0, indices)
                for name, model in models.items():
                    device = model_devices[name]
                    surf_error, vol_error, global_error = predict_errors(model, device, geo_view, surf_q, surf_y, vol_q, vol_y, dataset)
                    rows.append({"case_id": int(dataset.data[case_index]), "model": name, "source": "native", "shift": shift, "level": level, "surface_rel_l2": surf_error, "volume_rel_l2": vol_error, "global_rel_l2": global_error})

        # VTP remeshes are already the density-shifted encoder sources. Do not
        # apply beta or sine weighting to them, matching the production studies.
        case_id = int(dataset.data[case_index])
        for source, path in vtp_source_paths(case_id, args).items():
            physical = sample_remeshed_encoder_points(
                path, args.input_points, args.seed + case_id + sum(map(ord, source)), args.remesh_input_sampling,
            )
            normalized = (physical - dataset.min_pos.numpy()[None, :]) / dataset.position_span.numpy()[None, :]
            if not np.isfinite(normalized).all():
                raise RuntimeError(f"Non-finite normalized VTP coordinates for {path}")
            geo_view = torch.from_numpy(np.ascontiguousarray(normalized, dtype=np.float32))
            for name, model in models.items():
                device = model_devices[name]
                surf_error, vol_error, global_error = predict_errors(model, device, geo_view, surf_q, surf_y, vol_q, vol_y, dataset)
                rows.append({"case_id": case_id, "model": name, "source": source, "shift": "remesh", "level": float(source.rsplit("div", 1)[-1]), "surface_rel_l2": surf_error, "volume_rel_l2": vol_error, "global_rel_l2": global_error})
    with (output_dir / "toy_sampling_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    protocol = {
        "benchmark": args.benchmark,
        "purpose": "Isolate encoder point-density sensitivity with fixed FEM/analytic ground truth and fixed evaluation quadrature.",
        "input_source": "Case-specific nonuniform virtual meshing density.",
        "queries": "Independent reference surface and volume clouds, shared exactly by all models and shifts.",
        "shifts": {"beta": "KDE-16 inverse-density reweighting", "sine_x": "sinusoidal x-axis mixture", "sine_y": "sinusoidal y-axis mixture", "remesh": "VTP remeshed encoder source, resampled uniformly per triangle to the train-aligned encoder budget without an additional density shift"},
        "objective": "SATLOSS: 0.2 primary supervised + 0.2 secondary supervised + 0.6 symmetric prediction consistency.",
        "cases": [int(dataset.data[index]) for index in selected], "levels": levels,
    }
    (output_dir / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    render_endpoint_plots(rows, output_dir, levels, requested_methods, requested_factors, args.font_scale)
    plot_density(first_case, output_dir, args.input_points)
    export_analysis_vtps(dataset, selected, models, model_devices, args, output_dir)
    print(f"Saved {len(rows)} matched evaluations to {output_dir}")


if __name__ == "__main__":
    main()
