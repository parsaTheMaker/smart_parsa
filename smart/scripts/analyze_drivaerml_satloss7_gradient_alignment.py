#!/usr/bin/env python3
"""Build exact, two-dimensional SATLOSS7 gradient diagnostics on a fixed probe.

Every diagnostic is derived from the three losses used during SATLOSS7
training: supervised view 1, supervised view 2, and prediction consistency.
There is no external or canonical reference gradient.  The report asks whether
the weighted consistency gradient supports the ground-truth supervised update
from the two views, and records exact full-parameter metrics at every stored
checkpoint.  It never modifies an optimizer or checkpoint.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data._utils.collate import default_collate

SCRIPT_DIR = Path(__file__).resolve().parent
SMART_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(SMART_ROOT) not in sys.path:
    sys.path.insert(0, str(SMART_ROOT))

from export_drivaerml_smart_anchor_attention import build_model, load_experiment_config  # noqa: E402
from data.datasets import get_dataset  # noqa: E402
from loss.losses import CombinedLoss, RelL2Loss  # noqa: E402
from train_consistency_common import (  # noqa: E402
    _cpu_generator,
    forward_model_view,
    prediction_consistency_smooth_l1_loss,
    sample_geometry_view,
    sample_uniform_beta,
    unpack_batch,
)


COMPONENTS = (
    "weighted_view_1",
    "weighted_view_2",
    "weighted_consistency",
    "two_view_supervised",
    "satloss_total",
)
COLORS = {
    "weighted_view_1": "#2878b5",
    "weighted_view_2": "#ef8a35",
    "weighted_consistency": "#7a5195",
    "two_view_supervised": "#6f6f6f",
    "satloss_total": "#2a9d62",
}
LABELS = {
    "weighted_view_1": "0.2 x view 1 supervised",
    "weighted_view_2": "0.2 x view 2 supervised",
    "weighted_consistency": "0.6 x consistency",
    "two_view_supervised": "Two-view ground-truth supervised",
    "satloss_total": "SATLOSS update",
}


def parse_checkpoint_spec(value: str) -> tuple[int | None, Path]:
    value = str(value)
    match = re.fullmatch(r"\s*(\d+)\s*:(.+)", value)
    if match:
        return int(match.group(1)), Path(match.group(2)).expanduser().resolve()
    return None, Path(value).expanduser().resolve()


def resolve_checkpoints(specs: list[str], patterns: list[str]) -> list[tuple[int, Path]]:
    found: dict[Path, int | None] = {}
    for spec in specs:
        epoch, path = parse_checkpoint_spec(spec)
        found[path] = epoch
    for pattern in patterns:
        for text in glob.glob(pattern):
            found[Path(text).expanduser().resolve()] = None
    if not found:
        raise ValueError("Provide at least one --checkpoint or --checkpoint-glob.")
    resolved = []
    for path, explicit_epoch in found.items():
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        epoch = explicit_epoch if explicit_epoch is not None else int(checkpoint.get("epoch", -1)) + 1
        resolved.append((epoch, path))
    return sorted(resolved, key=lambda item: (item[0], str(item[1])))


def select_parameters(model: torch.nn.Module, scope: str) -> tuple[list[str], list[torch.nn.Parameter]]:
    named = list(model.named_parameters())
    if scope == "all":
        selected = [(name, parameter) for name, parameter in named if parameter.requires_grad]
    elif scope == "output_and_final_decoder":
        final_index = len(model.decoder_blocks) - 1
        prefix = f"decoder_blocks.{final_index}."
        selected = [(name, parameter) for name, parameter in named if parameter.requires_grad and (name.startswith("mlp.") or name.startswith(prefix))]
    else:
        raise ValueError(f"Unsupported --parameter-scope: {scope}")
    if not selected:
        raise RuntimeError("No trainable parameters selected for gradient analysis.")
    return [name for name, _ in selected], [parameter for _, parameter in selected]


def zero_like(parameters: Iterable[torch.nn.Parameter]) -> list[torch.Tensor]:
    return [torch.zeros_like(parameter, dtype=torch.float32) for parameter in parameters]


def add_gradients(total: list[torch.Tensor], gradients: Iterable[torch.Tensor | None], scale: float) -> None:
    for target, gradient in zip(total, gradients):
        if gradient is not None:
            target.add_(gradient.detach().float(), alpha=float(scale))


def weighted_sum(left: list[torch.Tensor], left_weight: float, right: list[torch.Tensor], right_weight: float) -> list[torch.Tensor]:
    return [left_weight * first + right_weight * second for first, second in zip(left, right)]


def weighted_sum_three(
    first: list[torch.Tensor], first_weight: float,
    second: list[torch.Tensor], second_weight: float,
    third: list[torch.Tensor], third_weight: float,
) -> list[torch.Tensor]:
    return [first_weight * a + second_weight * b + third_weight * c for a, b, c in zip(first, second, third)]


def scaled(value: list[torch.Tensor], weight: float) -> list[torch.Tensor]:
    return [float(weight) * part for part in value]


def dot(left: list[torch.Tensor], right: list[torch.Tensor]) -> float:
    return float(sum(torch.sum(a.double() * b.double()) for a, b in zip(left, right)).item())


def norm(value: list[torch.Tensor]) -> float:
    return math.sqrt(max(dot(value, value), 0.0))


def cosine(left: list[torch.Tensor], right: list[torch.Tensor]) -> float:
    denominator = norm(left) * norm(right)
    return dot(left, right) / denominator if denominator > 1.0e-24 else float("nan")


class CountSketchProjector:
    """Reusable linear CountSketch for qualitative two-dimensional PCA.

    The sketch maps are determined only by the selected parameter topology.
    Building them once per persistent GPU worker avoids five full index/hash
    passes per checkpoint while preserving linearity of the projected updates.
    """

    def __init__(self, parameters: list[torch.nn.Parameter], dimension: int, seed: int):
        self.dimension = int(dimension)
        self.maps: list[tuple[torch.Tensor, torch.Tensor]] = []
        offset = 0
        for parameter in parameters:
            index = torch.arange(parameter.numel(), device=parameter.device, dtype=torch.int64) + int(offset)
            mixed = index * 1103515245 + int(seed) * 12345 + 1013904223
            bucket = torch.remainder(mixed, self.dimension).long()
            sign = (1.0 - 2.0 * torch.remainder(mixed // 2654435761, 2).double()).to(torch.float32)
            self.maps.append((bucket, sign))
            offset += parameter.numel()

    def project(self, parts: list[torch.Tensor]) -> np.ndarray:
        if len(parts) != len(self.maps):
            raise RuntimeError("Gradient topology changed while projecting CountSketch.")
        output = torch.zeros(self.dimension, device=parts[0].device, dtype=torch.float64)
        for part, (bucket, sign) in zip(parts, self.maps):
            values = part.detach().reshape(-1).double()
            output.scatter_add_(0, bucket, values * sign.double())
        return output.cpu().numpy().astype(np.float64)


def model_forward_pair(model, view_1, view_2, surf_mesh, vol_mesh):
    fused_geo = torch.cat([view_1, view_2], dim=0)
    fused_surf = torch.cat([surf_mesh, surf_mesh], dim=0)
    fused_vol = torch.cat([vol_mesh, vol_mesh], dim=0)
    output_surf, output_vol = forward_model_view(
        model, fused_geo, fused_surf, fused_vol, None, model_requires_density=False
    )
    return output_surf.chunk(2, dim=0), output_vol.chunk(2, dim=0)


def fixed_views(geometry, log_density, config, seed: int, case_offset: int):
    """Match SATLOSS7: one shared family, independently sampled view levels."""
    family = int(case_offset % 3)
    view_seed = int(seed + case_offset * 100003)
    generator_1 = _cpu_generator(view_seed + 17)
    generator_2 = _cpu_generator(view_seed + 23)
    budget = int(getattr(config, "view_geometry_points", 131072))
    if family == 0:
        mode, axis = "inverse_density_wor", None
        level_1 = sample_uniform_beta(getattr(config, "shared_shift_beta_min", 0.0), getattr(config, "shared_shift_beta_max", 1.0), generator_1)
        level_2 = sample_uniform_beta(getattr(config, "shared_shift_beta_min", 0.0), getattr(config, "shared_shift_beta_max", 1.0), generator_2)
        beta_1, beta_2, sine_1, sine_2 = level_1, level_2, 0.0, 0.0
    else:
        mode, axis = "sinusoidal_axis_mixture_wor", (1 if family == 1 else 0)
        level_1 = sample_uniform_beta(getattr(config, "shared_shift_sine_min", 0.0), getattr(config, "shared_shift_sine_max", 1.0), generator_1)
        level_2 = sample_uniform_beta(getattr(config, "shared_shift_sine_min", 0.0), getattr(config, "shared_shift_sine_max", 1.0), generator_2)
        beta_1, beta_2, sine_1, sine_2 = 0.0, 0.0, level_1, level_2
    view_1, _density_1, _ = sample_geometry_view(geometry, log_density, budget, mode, beta_1, 1.0, view_seed + 11, sinusoidal_axis=axis, sinusoidal_mix_fraction=sine_1)
    view_2, _density_2, _ = sample_geometry_view(geometry, log_density, budget, mode, beta_2, 1.0, view_seed + 29, sinusoidal_axis=axis, sinusoidal_mix_fraction=sine_2)
    return view_1, view_2, ("beta", "sine_y", "sine_x")[family]


def make_probe_batches(config, cases: int, seed: int):
    probe_config = config.copy()
    probe_config.num_surface_points = int(getattr(config, "gradient_probe_surface_points", 8192))
    probe_config.num_volume_points = int(getattr(config, "gradient_probe_volume_points", 8192))
    # This controls all dataset streams, not geometry alone.  Keeping it on
    # makes surface and volume supervision queries identical across every
    # checkpoint and spawned GPU worker.  SATLOSS view shifts are constructed
    # separately below from the complete native geometry cloud.
    probe_config.geometry_epoch_seeded_sampling = True
    train_data, _test_data, _stats, _spatial_dim, _surf_channels, _vol_channels, params_dim, fields = get_dataset(probe_config)
    if params_dim != 0 or config.dataset != "DrivAerML":
        raise ValueError("This initial dashboard implementation is intentionally restricted to parameter-free DrivAerML SMART SATLOSS7.")
    if not hasattr(train_data, "set_epoch"):
        raise RuntimeError("The gradient probe requires a dataset with deterministic epoch control.")
    train_data.set_epoch(0)
    rng = np.random.default_rng(int(seed))
    indices = np.sort(rng.choice(len(train_data), size=min(int(cases), len(train_data)), replace=False))
    batches = [default_collate([train_data[int(index)]]) for index in indices]
    run_ids = [int(train_data.data[int(index)]) for index in indices]
    return batches, run_ids, fields


def probe_checkpoint(model, config, batches, parameters, device, seed: int):
    loss_fn = RelL2Loss()
    combined_loss = CombinedLoss(loss_fn, {"surface": [], "volume": []})
    weights = list(getattr(config, "config_task_base_weights", [0.2, 0.2, 0.6]))
    if len(weights) != 3:
        raise ValueError("SATLOSS7 gradient analysis requires exactly three fixed task weights.")
    w1, w2, wc = (float(weight) for weight in weights)
    totals = {name: zero_like(parameters) for name in ("view_1", "view_2", "consistency")}
    loss_values = {name: [] for name in ("view_1", "view_2", "consistency")}
    model.eval()
    precision = str(getattr(config, "precision", "float16"))
    amp_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}.get(precision, torch.float16)
    amp_enabled = bool(getattr(config, "amp", False)) and device.type == "cuda" and precision != "float32"
    for case_index, batch in enumerate(batches):
        geo, surf, surf_target, vol, vol_target, params, log_density = unpack_batch(batch, params_dim=0)
        if params is not None:
            raise ValueError("This gradient probe expects parameter-free DrivAerML batches.")
        view_1, view_2, _family = fixed_views(geo, log_density, config, seed, case_index)
        view_1 = view_1.to(device=device, dtype=torch.float32, non_blocking=True)
        view_2 = view_2.to(device=device, dtype=torch.float32, non_blocking=True)
        surf = surf.to(device=device, dtype=torch.float32, non_blocking=True)
        vol = vol.to(device=device, dtype=torch.float32, non_blocking=True)
        surf_target = surf_target.to(device=device, dtype=torch.float32, non_blocking=True)
        vol_target = vol_target.to(device=device, dtype=torch.float32, non_blocking=True)
        with torch.random.fork_rng(devices=[device] if device.type == "cuda" else []):
            torch.manual_seed(int(seed + case_index * 7001 + 1))
            if device.type == "cuda":
                torch.cuda.manual_seed_all(int(seed + case_index * 7001 + 1))
            # Match the training forward precision, while losses remain FP32.
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                (pred_1, pred_2), (vol_1, vol_2) = model_forward_pair(model, view_1, view_2, surf, vol)
            loss_1 = combined_loss(pred_1.float(), vol_1.float(), surf_target, vol_target)
            loss_2 = combined_loss(pred_2.float(), vol_2.float(), surf_target, vol_target)
            loss_c = prediction_consistency_smooth_l1_loss(
                pred_1.float(), vol_1.float(), pred_2.float(), vol_2.float(),
                beta=float(getattr(config, "prediction_consistency_smooth_l1_beta", 0.1)),
                symmetric_detached=bool(getattr(config, "prediction_consistency_symmetric_detached", True)),
                average_groups=bool(getattr(config, "prediction_consistency_average_groups", True)),
            )
            grad_1 = torch.autograd.grad(loss_1, parameters, retain_graph=True, allow_unused=True)
            grad_2 = torch.autograd.grad(loss_2, parameters, retain_graph=True, allow_unused=True)
            grad_c = torch.autograd.grad(loss_c, parameters, retain_graph=False, allow_unused=True)
        scale = 1.0 / len(batches)
        add_gradients(totals["view_1"], grad_1, scale)
        add_gradients(totals["view_2"], grad_2, scale)
        add_gradients(totals["consistency"], grad_c, scale)
        for name, value in (("view_1", loss_1), ("view_2", loss_2), ("consistency", loss_c)):
            loss_values[name].append(float(value.detach().item()))
        del grad_1, grad_2, grad_c, loss_1, loss_2, loss_c, pred_1, pred_2, vol_1, vol_2
    totals["weighted_view_1"] = scaled(totals["view_1"], w1)
    totals["weighted_view_2"] = scaled(totals["view_2"], w2)
    totals["weighted_consistency"] = scaled(totals["consistency"], wc)
    totals["two_view_supervised"] = weighted_sum(totals["view_1"], w1, totals["view_2"], w2)
    totals["satloss_total"] = weighted_sum_three(totals["view_1"], w1, totals["view_2"], w2, totals["consistency"], wc)
    return totals, {name: float(np.mean(values)) for name, values in loss_values.items()}, (w1, w2, wc)


def ema(values: np.ndarray, decay: float) -> np.ndarray:
    """Causal EMA along the checkpoint axis, retaining the first observation."""
    output = np.empty_like(values, dtype=np.float64)
    output[0] = values[0]
    for index in range(1, len(values)):
        output[index] = float(decay) * output[index - 1] + (1.0 - float(decay)) * values[index]
    return output


def pca_2d_direction_coordinates(sketches: np.ndarray, decay: float) -> tuple[np.ndarray, np.ndarray]:
    """Return raw/EMA PC1-PC2 coordinates from one shared basis.

    The PCA input is a signed CountSketch of unit full-gradient directions.
    It is explicitly qualitative; all reported numerical claims remain based
    on exact full-parameter dot products and norms.
    """
    unit = sketches / np.maximum(np.linalg.norm(sketches, axis=-1, keepdims=True), 1.0e-12)
    ema_unit = ema(unit, decay)
    flat = unit.reshape(-1, unit.shape[-1])
    center = flat.mean(axis=0, keepdims=True)
    _u, _s, vt = np.linalg.svd(flat - center, full_matrices=False)
    basis = vt[:2].T
    raw = ((unit.reshape(-1, unit.shape[-1]) - center) @ basis).reshape(*unit.shape[:2], 2)
    smooth = ((ema_unit.reshape(-1, ema_unit.shape[-1]) - center) @ basis).reshape(*unit.shape[:2], 2)
    return raw, smooth


def record_from_totals(epoch, checkpoint_path, run_ids, weights, losses, totals) -> dict:
    supervised = totals["two_view_supervised"]
    satloss = totals["satloss_total"]
    consistency = totals["weighted_consistency"]
    supervised_norm_sq = max(dot(supervised, supervised), 1.0e-24)
    row = {
        "epoch": int(epoch), "checkpoint": str(checkpoint_path), "probe_runs": run_ids,
        "weight_view_1": weights[0], "weight_view_2": weights[1], "weight_consistency": weights[2],
        **{f"loss_{key}": value for key, value in losses.items()},
        "loss_supervised_mean": 0.5 * (losses["view_1"] + losses["view_2"]),
        "norm_view_1": norm(totals["view_1"]), "norm_view_2": norm(totals["view_2"]), "norm_consistency": norm(totals["consistency"]),
        **{f"norm_{name}": norm(totals[name]) for name in COMPONENTS},
        "cos_view1_view2": cosine(totals["view_1"], totals["view_2"]),
        "cos_consistency_two_view_supervised": cosine(consistency, supervised),
        "cos_satloss_two_view_supervised": cosine(satloss, supervised),
        "supervised_descent_multiplier": dot(supervised, satloss) / supervised_norm_sq,
        "consistency_descent_increment": dot(supervised, consistency) / supervised_norm_sq,
        "weighted_consistency_norm_fraction": norm(consistency) / max(norm(satloss), 1.0e-24),
    }
    for left_index, left_name in enumerate(COMPONENTS):
        for right_name in COMPONENTS[left_index + 1:]:
            row[f"cos_{left_name}__{right_name}"] = cosine(totals[left_name], totals[right_name])
    return row


def evaluate_checkpoint_shard(tasks: list[dict]) -> list[tuple[dict, np.ndarray, list[str]]]:
    """Evaluate a dedicated sequential checkpoint shard on one persistent GPU."""
    if not tasks:
        return []
    device = torch.device(tasks[0]["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device {device} was requested but CUDA is unavailable.")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    task = tasks[0]
    config = load_experiment_config(task["config"])
    config.gradient_probe_surface_points = int(task["probe_surface_points"])
    config.gradient_probe_volume_points = int(task["probe_volume_points"])
    batches, run_ids, _fields = make_probe_batches(config, int(task["probe_cases"]), int(task["seed"]))
    projector = None
    parameter_names_reference = None
    results = []
    for task in tasks:
        epoch = int(task["epoch"])
        checkpoint_path = Path(task["checkpoint"])
        model, _cfg = build_model(task["config"], checkpoint_path, device)
        parameter_names, parameters = select_parameters(model, task["parameter_scope"])
        if parameter_names_reference is None:
            parameter_names_reference = parameter_names
            projector = CountSketchProjector(parameters, int(task["pca_sketch_dim"]), int(task["seed"]))
        elif parameter_names != parameter_names_reference:
            raise RuntimeError("Parameter topology differs between checkpoints in one GPU shard.")
        totals, losses, weights = probe_checkpoint(model, config, batches, parameters, device, int(task["seed"]))
        # CountSketch is linear, so three projections reconstruct all five
        # displayed components exactly in sketch space.
        sketch_1 = projector.project(totals["view_1"])
        sketch_2 = projector.project(totals["view_2"])
        sketch_c = projector.project(totals["consistency"])
        component_sketches = np.stack((
            weights[0] * sketch_1,
            weights[1] * sketch_2,
            weights[2] * sketch_c,
            weights[0] * sketch_1 + weights[1] * sketch_2,
            weights[0] * sketch_1 + weights[1] * sketch_2 + weights[2] * sketch_c,
        ))
        row = record_from_totals(epoch, checkpoint_path, run_ids, weights, losses, totals)
        print(
            f"[{task['device']}] epoch={epoch} view_cos={row['cos_view1_view2']:.5f} "
            f"consistency_to_supervised={row['cos_consistency_two_view_supervised']:+.5f} "
            f"descent_multiplier={row['supervised_descent_multiplier']:.5f}",
            flush=True,
        )
        results.append((row, component_sketches, parameter_names))
        del model, parameters, totals, losses
        if device.type == "cuda":
            torch.cuda.empty_cache()
    del batches, projector
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return results


def apply_ema_metrics(records: list[dict], decay: float) -> list[str]:
    """Add causal EMA values while leaving exact raw metrics untouched."""
    keys = (
        "loss_view_1", "loss_view_2", "loss_consistency", "loss_supervised_mean",
        "cos_view1_view2", "cos_consistency_two_view_supervised",
        "cos_satloss_two_view_supervised", "supervised_descent_multiplier",
        "consistency_descent_increment", "weighted_consistency_norm_fraction",
        *(f"norm_{name}" for name in COMPONENTS),
    )
    for key in keys:
        values = np.asarray([record[key] for record in records], dtype=np.float64)
        for record, value in zip(records, ema(values, decay)):
            record[f"ema_{key}"] = float(value)
    return list(keys)


def _plot_style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 320,
        "font.family": "DejaVu Sans",
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "axes.titleweight": "bold",
        "axes.edgecolor": "#495057",
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "grid.color": "#d9dee3",
        "grid.alpha": 0.8,
        "grid.linewidth": 0.7,
        "legend.frameon": False,
        "figure.facecolor": "#ffffff",
        "axes.facecolor": "#fbfcfd",
    })
    return plt


def _line_with_ema(axis, epochs, records, key, label, color):
    raw = [record[key] for record in records]
    smooth = [record[f"ema_{key}"] for record in records]
    axis.plot(epochs, raw, color=color, alpha=0.26, linewidth=1.2, label=f"{label}: raw")
    axis.plot(epochs, smooth, color=color, linewidth=2.8, marker="o", markersize=3.0, label=f"{label}: EMA")


def pairwise_cosine_matrix(record: dict) -> np.ndarray:
    matrix = np.eye(len(COMPONENTS), dtype=np.float64)
    for left_index, left_name in enumerate(COMPONENTS):
        for right_index, right_name in enumerate(COMPONENTS[left_index + 1:], start=left_index + 1):
            value = float(record[f"cos_{left_name}__{right_name}"])
            matrix[left_index, right_index] = matrix[right_index, left_index] = value
    return matrix


def save_static_figures(
    records: list[dict],
    raw_pca: np.ndarray,
    ema_pca: np.ndarray,
    output_dir: Path,
    ema_decay: float,
) -> list[str]:
    """Save publication-quality 2-D diagnostics without projected gradients."""
    plt = _plot_style()
    epochs = np.asarray([record["epoch"] for record in records], dtype=np.int64)
    paths: list[str] = []

    fig, axes = plt.subplots(2, 2, figsize=(15.2, 10.2), constrained_layout=True)
    for key, label, color in (
        ("loss_view_1", "View 1 supervised", COLORS["weighted_view_1"]),
        ("loss_view_2", "View 2 supervised", COLORS["weighted_view_2"]),
        ("loss_consistency", "Consistency", COLORS["weighted_consistency"]),
    ):
        _line_with_ema(axes[0, 0], epochs, records, key, label, color)
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_title("Fixed-Probe Losses")
    axes[0, 0].set_xlabel("Checkpoint epoch")
    axes[0, 0].set_ylabel("Loss (log scale)")
    axes[0, 0].legend(ncol=2, fontsize=9, loc="best")

    for key, label, color in (
        ("cos_view1_view2", "View 1 vs view 2", "#2f6690"),
        ("cos_consistency_two_view_supervised", "Consistency vs two-view supervised", COLORS["weighted_consistency"]),
        ("cos_satloss_two_view_supervised", "SATLOSS update vs two-view supervised", COLORS["satloss_total"]),
    ):
        _line_with_ema(axes[0, 1], epochs, records, key, label, color)
    axes[0, 1].axhline(0.0, color="#343a40", linewidth=0.9)
    axes[0, 1].set_ylim(-1.05, 1.05)
    axes[0, 1].set_title("Exact Full-Parameter Direction Cosines")
    axes[0, 1].set_xlabel("Checkpoint epoch")
    axes[0, 1].set_ylabel("Cosine similarity")
    axes[0, 1].legend(fontsize=8.8, loc="best")

    _line_with_ema(axes[1, 0], epochs, records, "supervised_descent_multiplier", "SATLOSS descent multiplier", COLORS["satloss_total"])
    _line_with_ema(axes[1, 0], epochs, records, "consistency_descent_increment", "Consistency increment", COLORS["weighted_consistency"])
    axes[1, 0].axhline(1.0, color="#343a40", linewidth=0.9, linestyle="--", label="Supervised-only update")
    axes[1, 0].axhline(0.0, color="#343a40", linewidth=0.7)
    axes[1, 0].set_title("First-Order Effect on Two-View GT Supervision")
    axes[1, 0].set_xlabel("Checkpoint epoch")
    axes[1, 0].set_ylabel("Multiplier / increment")
    axes[1, 0].legend(fontsize=8.8, loc="best")

    for name in COMPONENTS:
        _line_with_ema(axes[1, 1], epochs, records, f"norm_{name}", LABELS[name], COLORS[name])
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_title("Weighted Gradient Norms")
    axes[1, 1].set_xlabel("Checkpoint epoch")
    axes[1, 1].set_ylabel("Full-parameter L2 norm (log scale)")
    axes[1, 1].legend(fontsize=8.0, loc="best")
    path = output_dir / "gradient_diagnostics_trajectory.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    paths.append(path.name)

    last = records[-1]
    matrix = pairwise_cosine_matrix(last)
    fig, axis = plt.subplots(figsize=(10.2, 8.2), constrained_layout=True)
    image = axis.imshow(matrix, cmap="RdBu_r", vmin=-1.0, vmax=1.0)
    labels = [LABELS[name] for name in COMPONENTS]
    axis.set_xticks(range(len(labels)), labels, rotation=28, ha="right")
    axis.set_yticks(range(len(labels)), labels)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            color = "#ffffff" if abs(matrix[row, column]) > 0.55 else "#111111"
            axis.text(column, row, f"{matrix[row, column]:+.2f}", ha="center", va="center", color=color, fontsize=11, fontweight="bold")
    fig.colorbar(image, ax=axis, fraction=0.048, pad=0.04, label="Exact cosine similarity")
    axis.set_title(f"Last Checkpoint (Epoch {last['epoch']}): Gradient Cosine Matrix")
    path = output_dir / "last_epoch_gradient_cosine_matrix.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    paths.append(path.name)

    fig, axes = plt.subplots(1, 2, figsize=(14.2, 5.7), constrained_layout=True)
    component_norms = [last[f"norm_{name}"] for name in COMPONENTS]
    bars = axes[0].bar(range(len(COMPONENTS)), component_norms, color=[COLORS[name] for name in COMPONENTS], width=0.72)
    axes[0].set_yscale("log")
    axes[0].set_xticks(range(len(COMPONENTS)), ["V1", "V2", "Consistency", "V1+V2", "SATLOSS"], rotation=0)
    axes[0].set_ylabel("Full-parameter L2 norm (log scale)")
    axes[0].set_title("Last-Epoch Weighted Gradient Magnitudes")
    for bar, value in zip(bars, component_norms):
        axes[0].text(bar.get_x() + bar.get_width() / 2, value, f"{value:.2e}", ha="center", va="bottom", fontsize=9, rotation=90)

    cosine_values = [
        last["cos_view1_view2"],
        last["cos_consistency_two_view_supervised"],
        last["cos_satloss_two_view_supervised"],
    ]
    cosine_labels = ["V1 vs V2", "Consistency vs V1+V2", "SATLOSS vs V1+V2"]
    bars = axes[1].bar(range(3), cosine_values, color=["#2f6690", COLORS["weighted_consistency"], COLORS["satloss_total"]], width=0.65)
    axes[1].axhline(0.0, color="#343a40", linewidth=0.9)
    axes[1].set_ylim(-1.05, 1.05)
    axes[1].set_xticks(range(3), cosine_labels, rotation=13, ha="right")
    axes[1].set_ylabel("Exact cosine similarity")
    axes[1].set_title("Last-Epoch Directional Relationships")
    for bar, value in zip(bars, cosine_values):
        axes[1].text(bar.get_x() + bar.get_width() / 2, value + (0.04 if value >= 0 else -0.06), f"{value:+.3f}", ha="center", va="bottom" if value >= 0 else "top", fontweight="bold")
    path = output_dir / "last_epoch_gradient_summary.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    paths.append(path.name)
    return paths


def _pca_vector_limits(coordinates: np.ndarray) -> tuple[float, float]:
    extent = float(np.max(np.abs(coordinates)))
    extent = max(extent * 1.16, 0.05)
    return -extent, extent


def _animation_frame_data(coordinates: np.ndarray):
    import plotly.graph_objects as go

    traces = []
    for component_index, name in enumerate(COMPONENTS):
        x, y = coordinates[component_index]
        traces.append(go.Scatter(
            x=[float(x)], y=[float(y)],
            mode="markers",
            marker={"color": COLORS[name], "size": 10, "line": {"color": "white", "width": 1.2}},
            hovertemplate=(
                f"<b>{LABELS[name]}</b><br>PC1: %{{x:.4f}}<br>PC2: %{{y:.4f}}<extra></extra>"
            ),
            name=LABELS[name],
            showlegend=False,
        ))
    return traces


def _epoch_evidence_annotations(record: dict, coordinates: np.ndarray) -> list[dict]:
    arrows = []
    for component_index, name in enumerate(COMPONENTS):
        x, y = coordinates[component_index]
        arrows.append({
            "xref": "x", "yref": "y", "axref": "x", "ayref": "y",
            "x": float(x), "y": float(y), "ax": 0.0, "ay": 0.0,
            "text": "", "showarrow": True, "arrowhead": 3, "arrowsize": 1.15,
            "arrowwidth": 3.3, "arrowcolor": COLORS[name],
        })
    values = (
        ("Epoch", f"{record['epoch']}", "#17212b"),
        ("View 1 / View 2", f"{record['cos_view1_view2']:+.3f}", COLORS["weighted_view_1"]),
        ("Consistency / two-view GT", f"{record['cos_consistency_two_view_supervised']:+.3f}", COLORS["weighted_consistency"]),
        ("SATLOSS descent multiplier", f"{record['supervised_descent_multiplier']:.3f}", COLORS["satloss_total"]),
    )
    annotations = [{
        "xref": "paper", "yref": "paper", "x": 0.855, "y": 1.04,
        "text": "Exact full-gradient evidence", "showarrow": False,
        "font": {"size": 17, "color": "#17212b"}, "xanchor": "center",
    }]
    for index, (label, value, color) in enumerate(values):
        y = 0.81 - 0.22 * index
        annotations.extend((
            {"xref": "paper", "yref": "paper", "x": 0.855, "y": y,
             "text": label, "showarrow": False, "font": {"size": 12, "color": "#59636f"}, "xanchor": "center"},
            {"xref": "paper", "yref": "paper", "x": 0.855, "y": y - 0.08,
             "text": f"<b>{value}</b>", "showarrow": False, "font": {"size": 25, "color": color}, "xanchor": "center"},
        ))
    annotations.extend((
        {"xref": "paper", "yref": "paper", "x": 0.855, "y": 0.00,
         "text": "Positive consistency / GT cosine means the<br>consistency update locally supports supervised learning.",
         "showarrow": False, "font": {"size": 11, "color": "#59636f"}, "xanchor": "center", "align": "center"},
        {"xref": "paper", "yref": "paper", "x": 0.36, "y": -0.14,
         "text": "Each arrow is one weighted loss gradient projected to PC1-PC2. Arrow length and angle are qualitative; the panel at right reports exact full-space values.",
         "showarrow": False, "font": {"size": 12, "color": "#59636f"}, "xanchor": "center", "align": "center"},
    ))
    return arrows + annotations


def save_interactive_figures(
    records: list[dict],
    ema_pca: np.ndarray,
    output_dir: Path,
    ema_decay: float,
) -> list[str]:
    """Create clean standalone Plotly reports; PCA is intentionally 2-D only."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    epochs = [int(record["epoch"]) for record in records]
    lower, upper = _pca_vector_limits(ema_pca)
    first_data = _animation_frame_data(ema_pca[0])
    frames = [go.Frame(
        name=str(record["epoch"]), data=_animation_frame_data(ema_pca[index]),
        layout=go.Layout(annotations=_epoch_evidence_annotations(record, ema_pca[index])),
    ) for index, record in enumerate(records)]
    animation = go.Figure(data=first_data, frames=frames)
    animation.update_layout(
        title={"text": (
            f"SATLOSS7 Gradient Directions: 2-D PCA Animation (EMA decay={ema_decay:.2f})<br>"
            f"<span style='font-size:13px'><span style='color:{COLORS['weighted_view_1']}'>● View 1</span> &nbsp; "
            f"<span style='color:{COLORS['weighted_view_2']}'>● View 2</span> &nbsp; "
            f"<span style='color:{COLORS['weighted_consistency']}'>● Consistency</span> &nbsp; "
            f"<span style='color:{COLORS['two_view_supervised']}'>● Two-view GT</span> &nbsp; "
            f"<span style='color:{COLORS['satloss_total']}'>● SATLOSS</span></span>"
        ), "x": 0.42, "xanchor": "center", "font": {"size": 23}},
        template="plotly_white", width=1240, height=790,
        margin={"l": 80, "r": 55, "t": 92, "b": 116},
        paper_bgcolor="#ffffff", plot_bgcolor="#fbfcfd",
        xaxis={"domain": [0.0, 0.71], "range": [lower, upper], "zeroline": True, "zerolinecolor": "#aeb8c2", "title": "PC 1", "gridcolor": "#e3e8ed", "constrain": "domain"},
        yaxis={"range": [lower, upper], "zeroline": True, "zerolinecolor": "#aeb8c2", "title": "PC 2", "gridcolor": "#e3e8ed", "scaleanchor": "x", "scaleratio": 1},
        shapes=[{"type": "rect", "xref": "paper", "yref": "paper", "x0": 0.745, "x1": 0.965, "y0": 0.08, "y1": 0.95,
                 "line": {"color": "#d8e0e7", "width": 1}, "fillcolor": "#f7f9fb", "layer": "below"}],
        annotations=_epoch_evidence_annotations(records[0], ema_pca[0]),
        updatemenus=[{
            "type": "buttons", "direction": "left", "x": 0.0, "y": -0.005, "xanchor": "left", "yanchor": "top",
            "buttons": [
                {"label": "Play", "method": "animate", "args": [None, {"frame": {"duration": 180, "redraw": True}, "transition": {"duration": 0}, "fromcurrent": True}]},
                {"label": "Pause", "method": "animate", "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate", "transition": {"duration": 0}}]},
            ],
        }],
        sliders=[{
            "active": 0, "x": 0.13, "y": -0.02, "len": 0.58,
            "currentvalue": {"prefix": "Epoch: ", "font": {"size": 14}},
            "steps": [{"label": str(epoch), "method": "animate", "args": [[str(epoch)], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate", "transition": {"duration": 0}}]} for epoch in epochs],
        }],
    )
    animation_path = output_dir / "gradient_direction_pca_2d_animation.html"
    animation.write_html(animation_path, include_plotlyjs=True, full_html=True, auto_play=False, config={"displaylogo": False, "responsive": True})

    metrics = (
        ("cos_view1_view2", "View 1 vs view 2", "#2f6690"),
        ("cos_consistency_two_view_supervised", "Consistency vs two-view GT", COLORS["weighted_consistency"]),
        ("cos_satloss_two_view_supervised", "SATLOSS vs two-view GT", COLORS["satloss_total"]),
        ("supervised_descent_multiplier", "SATLOSS descent multiplier", "#1f7a4d"),
    )
    trajectory = make_subplots(rows=2, cols=1, vertical_spacing=0.18, subplot_titles=("Exact directional agreement", "Exact first-order effect on two-view supervised loss"))
    for key, label, color in metrics[:3]:
        trajectory.add_trace(go.Scatter(x=epochs, y=[record[key] for record in records], mode="lines", line={"color": color, "width": 1}, opacity=0.22, hovertemplate=f"<b>{label}, raw</b><br>Epoch %{{x}}<br>%{{y:+.4f}}<extra></extra>", showlegend=False), row=1, col=1)
        trajectory.add_trace(go.Scatter(x=epochs, y=[record[f"ema_{key}"] for record in records], mode="lines", line={"color": color, "width": 3}, name=label, hovertemplate=f"<b>{label}, EMA</b><br>Epoch %{{x}}<br>%{{y:+.4f}}<extra></extra>"), row=1, col=1)
    key, label, color = metrics[3]
    trajectory.add_trace(go.Scatter(x=epochs, y=[record[key] for record in records], mode="lines", line={"color": color, "width": 1}, opacity=0.22, showlegend=False, hovertemplate="<b>Raw multiplier</b><br>Epoch %{x}<br>%{y:.4f}<extra></extra>"), row=2, col=1)
    trajectory.add_trace(go.Scatter(x=epochs, y=[record[f"ema_{key}"] for record in records], mode="lines", line={"color": color, "width": 3}, name="SATLOSS descent multiplier", hovertemplate="<b>EMA multiplier</b><br>Epoch %{x}<br>%{y:.4f}<extra></extra>"), row=2, col=1)
    trajectory.add_hline(y=0.0, line_color="#65717c", line_width=1, row=1, col=1)
    trajectory.add_hline(y=1.0, line_color="#65717c", line_width=1, line_dash="dash", annotation_text="supervised-only baseline", row=2, col=1)
    trajectory.update_layout(template="plotly_white", height=760, width=1110, title={"text": "Exact Full-Gradient Learning Evidence", "x": 0.5, "font": {"size": 23}}, legend={"orientation": "h", "y": 1.08}, margin={"l": 78, "r": 35, "t": 100, "b": 65}, paper_bgcolor="#ffffff", plot_bgcolor="#fbfcfd")
    trajectory.update_xaxes(title_text="Checkpoint epoch", gridcolor="#e3e8ed")
    trajectory.update_yaxes(range=[-1.05, 1.05], title_text="Cosine similarity", gridcolor="#e3e8ed", row=1, col=1)
    trajectory.update_yaxes(title_text="Multiplier", gridcolor="#e3e8ed", row=2, col=1)
    trajectory_path = output_dir / "gradient_learning_evidence_interactive.html"
    trajectory.write_html(trajectory_path, include_plotlyjs=True, full_html=True, config={"displaylogo": False, "responsive": True})
    return [animation_path.name, trajectory_path.name]


def write_last_epoch_interpretation(records: list[dict], output_dir: Path, ema_decay: float) -> None:
    last = records[-1]
    support = last["cos_consistency_two_view_supervised"]
    multiplier = last["supervised_descent_multiplier"]
    if support > 0.05:
        conclusion = "The weighted consistency gradient supports the two-view ground-truth supervised direction at this checkpoint."
    elif support < -0.05:
        conclusion = "The weighted consistency gradient locally conflicts with the two-view ground-truth supervised direction at this checkpoint."
    else:
        conclusion = "The weighted consistency gradient is nearly orthogonal to the two-view ground-truth supervised direction at this checkpoint."
    text = f"""# Last-Epoch SATLOSS7 Gradient Interpretation

Checkpoint epoch: **{last['epoch']}**  
EMA decay used in trajectory figures: **{ema_decay:.2f}**

## Exact quantities

All values below are computed before plotting, in the selected full parameter space. No dimensionality reduction or reference gradient is used.

| Quantity | Value | Interpretation |
|---|---:|---|
| View-1 vs view-2 gradient cosine | {last['cos_view1_view2']:+.6f} | Agreement/conflict between the two independently shifted, ground-truth supervised views. |
| Consistency vs two-view supervised cosine | {support:+.6f} | Whether the weighted consistency gradient supports the weighted sum of the two ground-truth supervised gradients. |
| SATLOSS vs two-view supervised cosine | {last['cos_satloss_two_view_supervised']:+.6f} | Whether the complete update remains aligned with ground-truth supervision. |
| First-order supervised descent multiplier | {multiplier:.6f} | Dot(weighted two-view supervised gradient, SATLOSS gradient) divided by squared norm of the weighted two-view supervised gradient. A value above 1 means the consistency term increases local first-order descent of the two-view GT loss; between 0 and 1 means it still descends but less strongly; below 0 means conflict. |
| Consistency descent increment | {last['consistency_descent_increment']:+.6f} | Multiplier minus 1; the direct local contribution of the weighted consistency term. |
| Weighted consistency norm / SATLOSS norm | {last['weighted_consistency_norm_fraction']:.6f} | Relative magnitude of the consistency contribution; it is not a percentage of total squared norm because gradients can cancel or reinforce. |

## Reading the evidence

{conclusion}

The strongest evidence should combine this local gradient diagnostic with the held-out sampling and remeshing error results. A positive consistency-to-supervised cosine and a multiplier above one indicate direct local support for learning the supervised physical fields. If the consistency gradient is orthogonal or mildly conflicting at isolated epochs, it can still improve representation invariance; the trajectory and final robustness errors should be reported together rather than relying on one scalar.
"""
    (output_dir / "last_epoch_gradient_interpretation.md").write_text(text, encoding="utf-8")


def write_dashboard(
    records: list[dict], output_dir: Path, ema_decay: float,
    interactive_figures: list[str], static_figures: list[str],
) -> None:
    last = records[-1]
    cards = (
        ("View 1 vs view 2", last["cos_view1_view2"], "The two ground-truth supervised views agree when this is near +1."),
        ("Consistency vs GT supervision", last["cos_consistency_two_view_supervised"], "Positive means consistency supports the two-view supervised direction."),
        ("SATLOSS vs GT supervision", last["cos_satloss_two_view_supervised"], "Positive means the complete update remains a supervised descent direction."),
        ("Supervised descent multiplier", last["supervised_descent_multiplier"], "1 is the supervised-only baseline; higher is stronger local descent."),
    )
    card_html = "".join(
        f"<section class='card'><h3>{title}</h3><div class='value'>{value:+.3f}</div><p>{description}</p></section>"
        for title, value, description in cards
    )
    animation_name, evidence_name = interactive_figures
    static_links = "".join(f"<li><a href='{name}'>{name}</a></li>" for name in static_figures)
    html = f"""<!doctype html>
<html><head><meta charset='utf-8'><title>SATLOSS7 gradient diagnostics</title>
<style>
body {{ margin: 0; background: #f4f6f8; color: #17212b; font: 16px/1.55 'Aptos','Segoe UI',sans-serif; }}
main {{ max-width: 1360px; margin: auto; padding: 36px 42px 68px; }} h1 {{ font-size: 32px; margin: 0 0 4px; }} h2 {{ margin: 42px 0 12px; font-size: 22px; }}
.sub {{ color: #53606d; margin: 0; }} .cards {{ display: grid; grid-template-columns: repeat(4, minmax(180px, 1fr)); gap: 14px; margin-top: 22px; }}
.card {{ background: #fff; border-top: 4px solid #2878b5; border-radius: 10px; padding: 15px 17px; box-shadow: 0 2px 8px #1d2b3a12; }} .card:nth-child(2) {{ border-color: #7a5195; }} .card:nth-child(3),.card:nth-child(4) {{ border-color: #2a9d62; }}
.card h3 {{ margin: 0; font-size: 14px; }} .value {{ font-size: 30px; font-weight: 730; margin: 4px 0; }} .card p {{ font-size: 13px; color: #53606d; margin: 0; }}
.panel {{ background: #fff; border-radius: 12px; padding: 12px; box-shadow: 0 2px 10px #1d2b3a12; }} iframe {{ width: 100%; border: 0; display: block; border-radius: 8px; }} .animation {{ height: 850px; }} .evidence {{ height: 800px; }}
.note {{ background: #fff; border-left: 5px solid #ef8a35; padding: 14px 18px; border-radius: 6px; }} code {{ background: #e9eef3; padding: 2px 5px; border-radius: 4px; }} a {{ color: #1e689e; }} details {{ margin-top: 24px; color: #53606d; }}
@media(max-width: 900px) {{ main {{ padding: 22px; }} .cards {{ grid-template-columns: 1fr; }} .animation,.evidence {{ height: 720px; }} }}
</style></head><body><main>
<h1>SATLOSS7 Gradient Diagnostics</h1>
<p class='sub'>Fixed probe, checkpoint epoch {last['epoch']}, exact full-parameter gradients. The animated directions use a causal EMA with decay {ema_decay:.2f}; numerical evidence is never PCA-reduced.</p>
<div class='cards'>{card_html}</div>
<h2>How to Interpret the Animation</h2>
<div class='note'><strong>Each frame shows five weighted loss-gradient directions in the PC1-PC2 plane.</strong> Use the slider to inspect a specific epoch or play to see their evolution. The vectors are an interpretable qualitative map, not a numerical test: PC sign and orientation are arbitrary. Read the exact full-parameter cosine and descent-multiplier values in the panel at the right of each frame and in the evidence chart below.</div>
<h2>2-D Gradient-Direction Animation</h2><section class='panel'><iframe class='animation' src='{animation_name}' title='Animated PC1-PC2 gradient directions'></iframe></section>
<h2>Exact Quantitative Evidence</h2><section class='panel'><iframe class='evidence' src='{evidence_name}' title='Exact gradient learning evidence'></iframe></section>
<details><summary>Static publication figures and raw data</summary><ul>{static_links}</ul><p><code>gradient_alignment_metrics.csv</code> and <code>gradient_alignment_metrics.json</code> contain exact raw and EMA values. <code>last_epoch_gradient_interpretation.md</code> gives a report-ready interpretation of the final checkpoint.</p></details>
</main></body></html>"""
    (output_dir / "gradient_alignment_dashboard.html").write_text(html, encoding="utf-8")


def render_existing_dashboard(output_dir: Path, ema_decay_override: float | None = None) -> None:
    """Rebuild the interactive report from an already completed gradient run."""
    metrics_path = output_dir / "gradient_alignment_metrics.json"
    pca_path = output_dir / "gradient_direction_pca_2d.npz"
    if not metrics_path.is_file() or not pca_path.is_file():
        raise FileNotFoundError(
            "--render-existing requires gradient_alignment_metrics.json and gradient_direction_pca_2d.npz "
            f"in {output_dir}."
        )
    records = json.loads(metrics_path.read_text(encoding="utf-8"))
    if not records:
        raise RuntimeError("Cannot render an empty gradient analysis.")
    archive = np.load(pca_path)
    ema_pca = np.asarray(archive["ema_coordinates"], dtype=np.float64)
    ema_decay = float(archive["ema_decay"]) if ema_decay_override is None else float(ema_decay_override)
    if ema_pca.shape[:2] != (len(records), len(COMPONENTS)) or ema_pca.shape[-1] != 2:
        raise RuntimeError("Stored PCA coordinates do not match the current report component layout.")
    static_figures = [
        name for name in (
            "gradient_diagnostics_trajectory.png",
            "last_epoch_gradient_cosine_matrix.png",
            "last_epoch_gradient_summary.png",
        ) if (output_dir / name).is_file()
    ]
    interactive_figures = save_interactive_figures(records, ema_pca, output_dir, ema_decay)
    write_dashboard(records, output_dir, ema_decay, interactive_figures, static_figures)
    print(f"Rebuilt interactive dashboard: {output_dir / 'gradient_alignment_dashboard.html'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="drivaerml_satloss7_range100")
    parser.add_argument("--checkpoint", action="append", default=[], help="Repeatable EPOCH:/absolute/path or /absolute/path.")
    parser.add_argument("--checkpoint-glob", action="append", default=[])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--devices", default="", help="Comma-separated evaluation devices. Overrides --device when set.")
    parser.add_argument("--probe-cases", type=int, default=8)
    parser.add_argument("--probe-surface-points", type=int, default=8192)
    parser.add_argument("--probe-volume-points", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=314159)
    parser.add_argument("--parameter-scope", choices=("all", "output_and_final_decoder"), default="all")
    parser.add_argument("--pca-sketch-dim", type=int, default=4096, help="Signed-sketch dimension for qualitative PC1-PC2 visualization.")
    parser.add_argument("--ema-decay", type=float, default=0.90, help="Causal EMA decay for gradient-direction visualization.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--render-existing", action="store_true", help="Regenerate the interactive HTML report from saved metrics/PCA data without evaluating checkpoints.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    if args.render_existing:
        render_existing_dashboard(output_dir)
        return
    if args.probe_cases <= 0 or args.probe_surface_points <= 0 or args.probe_volume_points <= 0 or args.pca_sketch_dim < 2:
        raise ValueError("Probe budgets must be positive and --pca-sketch-dim must be at least 2.")
    if not 0.0 <= args.ema_decay < 1.0:
        raise ValueError("--ema-decay must be in [0, 1).")
    checkpoints = resolve_checkpoints(args.checkpoint, args.checkpoint_glob)
    devices = [value.strip() for value in args.devices.split(",") if value.strip()] or [args.device]
    if any(torch.device(value).type == "cuda" for value in devices) and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    output_dir.mkdir(parents=True, exist_ok=True)
    records, all_sketches = [], []
    parameter_names_reference = None
    tasks = [
        {
            "epoch": epoch, "checkpoint": str(checkpoint_path), "device": devices[index % len(devices)],
            "config": args.config, "probe_cases": args.probe_cases,
            "probe_surface_points": args.probe_surface_points, "probe_volume_points": args.probe_volume_points,
            "seed": args.seed, "parameter_scope": args.parameter_scope, "pca_sketch_dim": args.pca_sketch_dim,
        }
        for index, (epoch, checkpoint_path) in enumerate(checkpoints)
    ]
    shards = [tasks[index::len(devices)] for index in range(len(devices))]
    shards = [shard for shard in shards if shard]
    print(
        f"Checkpoints={len(checkpoints)}; devices={devices}; parameter_scope={args.parameter_scope}; "
        f"persistent_gpu_shards={[len(shard) for shard in shards]}",
        flush=True,
    )
    partial_path = output_dir / "gradient_alignment_partial.jsonl"
    with partial_path.open("w", encoding="utf-8") as partial_handle:
        if len(shards) == 1:
            results = evaluate_checkpoint_shard(shards[0])
            for row, _sketches, _names in results:
                partial_handle.write(json.dumps(row) + "\n")
        else:
            results = []
            # One process owns one CUDA device for its entire shard.  A generic
            # task pool can otherwise run two model probes on the same device.
            with ProcessPoolExecutor(max_workers=len(shards), mp_context=get_context("spawn")) as executor:
                futures = [executor.submit(evaluate_checkpoint_shard, shard) for shard in shards]
                for future in as_completed(futures):
                    shard_results = future.result()
                    results.extend(shard_results)
                    for row, _sketches, _names in shard_results:
                        partial_handle.write(json.dumps(row) + "\n")
                    partial_handle.flush()
    for row, component_sketches, parameter_names in sorted(results, key=lambda result: (result[0]["epoch"], result[0]["checkpoint"])):
        if parameter_names_reference is None:
            parameter_names_reference = parameter_names
        elif parameter_names != parameter_names_reference:
            raise RuntimeError("Parameter topology differs between checkpoints.")
        records.append(row)
        all_sketches.append(component_sketches)
    apply_ema_metrics(records, args.ema_decay)
    raw_pca, ema_pca = pca_2d_direction_coordinates(np.stack(all_sketches), args.ema_decay)
    serializable_records = [{key: value for key, value in record.items() if key != "probe_runs"} | {"probe_runs": record["probe_runs"]} for record in records]
    (output_dir / "gradient_alignment_metrics.json").write_text(json.dumps(serializable_records, indent=2) + "\n", encoding="utf-8")
    with (output_dir / "gradient_alignment_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        fields_csv = [key for key in records[0] if key != "probe_runs"]
        writer = csv.DictWriter(handle, fieldnames=fields_csv)
        writer.writeheader()
        writer.writerows([{key: value for key, value in record.items() if key != "probe_runs"} for record in records])
    np.savez_compressed(
        output_dir / "gradient_alignment_metadata.npz",
        epochs=np.asarray([record["epoch"] for record in records]),
        component_names=np.asarray(COMPONENTS),
        ema_decay=np.asarray(args.ema_decay),
        pca_sketch_dim=np.asarray(args.pca_sketch_dim),
        parameter_names=np.asarray(parameter_names_reference),
    )
    # The earlier dashboard stored projected reference-gradient sketches. They
    # are intentionally obsolete after the exact two-view redesign.
    legacy_sketches = output_dir / "gradient_alignment_sketches.npz"
    if legacy_sketches.exists():
        legacy_sketches.unlink()
    np.savez_compressed(
        output_dir / "gradient_direction_pca_2d.npz",
        epochs=np.asarray([record["epoch"] for record in records]),
        component_names=np.asarray(COMPONENTS),
        raw_coordinates=raw_pca,
        ema_coordinates=ema_pca,
        ema_decay=np.asarray(args.ema_decay),
        sketch_dimension=np.asarray(args.pca_sketch_dim),
    )
    static_figures = save_static_figures(records, raw_pca, ema_pca, output_dir, args.ema_decay)
    interactive_figures = save_interactive_figures(records, ema_pca, output_dir, args.ema_decay)
    write_last_epoch_interpretation(records, output_dir, args.ema_decay)
    write_dashboard(records, output_dir, args.ema_decay, interactive_figures, static_figures)
    print(f"Dashboard: {output_dir / 'gradient_alignment_dashboard.html'}")


if __name__ == "__main__":
    main()
