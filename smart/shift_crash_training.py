"""Isolated SMART and SATLoss7 training for SHIFT-Crash terminal displacement."""

from __future__ import annotations

import os
import inspect
import zlib
from collections import OrderedDict
from timeit import default_timer

import numpy as np
import torch
import wandb
from torch.nn import DataParallel
from tqdm.auto import tqdm

from data.shift_crash_dataset import ShiftCrashDataset
from models.smart.smart import Modulator, SMART
from train_consistency_common import sample_geometry_view, sample_shared_shift_family, sample_uniform_beta
from utils.utils import (
    count_model_params,
    exclude_params_from_weight_decay,
    get_model_checkpoint_name,
    initialize_gpu,
    initialize_wandb,
    make_grad_scaler,
)


def _unwrap(model):
    return model.module if hasattr(model, "module") else model


def _move(value, device):
    return value.to(device, non_blocking=True) if torch.is_tensor(value) else value


def _record_stream(value, stream):
    if torch.is_tensor(value):
        if value.is_cuda:
            value.record_stream(stream)
    elif isinstance(value, dict):
        for item in value.values():
            _record_stream(item, stream)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _record_stream(item, stream)


def _move_batch(batch, device, keep_cpu_keys):
    return {
        key: value if key in keep_cpu_keys else _move_batch_value(value, device)
        for key, value in batch.items()
    }


def _move_batch_value(value, device):
    if torch.is_tensor(value):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {key: _move_batch_value(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_move_batch_value(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_batch_value(item, device) for item in value)
    return value


class _CudaPrefetchLoader:
    """Prefetch dict batches while optionally retaining CPU sampling tensors."""

    def __init__(self, loader, device, keep_cpu_keys=()):
        self.loader = loader
        self.device = device
        self.keep_cpu_keys = frozenset(keep_cpu_keys)
        self.enabled = device.type == "cuda" and torch.cuda.is_available()
        self.stream = torch.cuda.Stream(device=device) if self.enabled else None

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        if not self.enabled:
            yield from self.loader
            return
        loader_iter = iter(self.loader)
        next_batch = None

        def preload():
            nonlocal next_batch
            try:
                batch = next(loader_iter)
            except StopIteration:
                next_batch = None
                return
            with torch.cuda.stream(self.stream):
                next_batch = _move_batch(batch, self.device, self.keep_cpu_keys)

        preload()
        while next_batch is not None:
            torch.cuda.current_stream(device=self.device).wait_stream(self.stream)
            batch = next_batch
            _record_stream(batch, torch.cuda.current_stream(device=self.device))
            preload()
            yield batch


def _empty_volume(query, channels=3):
    del channels
    return query.new_empty((query.shape[0], 0, query.shape[-1]))


def _forward(
    model,
    geometry,
    query,
    params,
    geometry_features=None,
    query_features=None,
    geometry_part_ids=None,
    query_part_ids=None,
    sampling_seeds=None,
):
    # SMART's FiLM modulators broadcast over point tokens.  DrivAerML uses no
    # conditioning parameters, so this path is normally not exercised there;
    # SHIFT-Crash has six case parameters and needs the singleton token axis.
    if params is not None and params.ndim == 2:
        params = params.unsqueeze(1)
    prediction, _empty = model(
        geometry,
        query,
        _empty_volume(query),
        params,
        geometry_features=geometry_features,
        query_features=query_features,
        geometry_part_ids=geometry_part_ids,
        query_part_ids=query_part_ids,
        sampling_seeds=sampling_seeds,
    )
    return prediction


def _gather_batch_points(values, indices):
    """Gather [B,N] or [B,N,C] CPU node attributes using [B,K] indices."""
    if values.ndim == 2:
        return torch.gather(values, 1, indices)
    return torch.gather(values, 1, indices.unsqueeze(-1).expand(-1, -1, values.shape[-1]))


def _conditioning_gradient_diagnostics(model):
    """Summarize only SMART FiLM MLPs, without changing their gradients."""
    base_model = _unwrap(model)
    parameter_sq = None
    gradient_sq = None
    gradient_max = 0.0
    nonfinite = 0
    module_count = 0
    for module in base_model.modules():
        if not isinstance(module, Modulator):
            continue
        module_count += 1
        for parameter in module.mlp.parameters():
            value = parameter.detach().float()
            value_sq = value.square().sum()
            parameter_sq = value_sq if parameter_sq is None else parameter_sq + value_sq
            if parameter.grad is None:
                continue
            gradient = parameter.grad.detach().float()
            nonfinite += int((~torch.isfinite(gradient)).any().item())
            finite_gradient = torch.where(torch.isfinite(gradient), gradient, torch.zeros_like(gradient))
            gradient_value = finite_gradient.square().sum()
            gradient_sq = gradient_value if gradient_sq is None else gradient_sq + gradient_value
            if gradient.numel():
                gradient_max = max(gradient_max, float(finite_gradient.abs().max().item()))
    if parameter_sq is None:
        reference = next(base_model.parameters()).detach()
        parameter_norm = reference.new_zeros((), dtype=torch.float32)
    else:
        parameter_norm = torch.sqrt(parameter_sq.clamp_min(0.0))
    if gradient_sq is None:
        gradient_norm = parameter_norm.new_zeros(())
    else:
        gradient_norm = torch.sqrt(gradient_sq.clamp_min(0.0))
    return {
        "conditioning_module_count": float(module_count),
        "conditioning_parameter_norm": float(parameter_norm.item()),
        "conditioning_gradient_norm": float(gradient_norm.item()),
        "conditioning_gradient_max_abs": float(gradient_max),
        "conditioning_gradient_nonfinite": float(nonfinite),
    }


def _relative_l2(prediction, target):
    # Same relative L2 definition used by the DrivAerML SMART trainer, with
    # explicit float32 arithmetic for stable mixed-precision optimization.
    prediction = prediction.float()
    target = target.float()
    target_norm = target.square().sum(dim=1)
    target_norm = torch.where(target_norm < 1.0e-5, torch.full_like(target_norm, 1.0e-5), target_norm)
    error_norm = (prediction - target).square().sum(dim=1)
    # This is RelL2Loss(dim=1, reduction="sum", reduce_all=True), the same
    # channel-wise relative error used by the DrivAerML SMART trainer.
    return (error_norm / target_norm).sqrt().mean()


def _relative_prediction_consistency(prediction_1, prediction_2, target, symmetric_detached=True):
    """Prediction agreement in the same dimensionless units as RelL2.

    A raw SmoothL1 error is measured in normalized displacement units, whereas
    the supervised losses are relative norms.  Combining them with fixed
    weights made the nominal 0.6 consistency term much smaller than either
    nominal 0.2 supervised term.  This is the direct relative-L2 analogue of
    prediction matching: its denominator is the known target response norm,
    detached so it cannot alter target scaling or gradients.
    """
    target_norm = target.float().square().sum(dim=1).clamp_min(1.0e-5).sqrt()

    def relative_difference(left, right):
        return ((left.float() - right.float()).square().sum(dim=1).sqrt() / target_norm).mean()

    if symmetric_detached:
        return 0.5 * (
            relative_difference(prediction_1, prediction_2.detach())
            + relative_difference(prediction_2, prediction_1.detach())
        )
    return relative_difference(prediction_1, prediction_2)


def _case_sampling_seeds(case_ids, offset, device):
    """Stable per-case seeds for evaluation-only internal SMART sampling."""
    values = [
        (zlib.crc32(str(case_id).encode("utf-8")) + int(offset)) % (2**31 - 1)
        for case_id in case_ids
    ]
    return torch.tensor(values, device=device, dtype=torch.long)


def _make_optimizer(model, config, steps_per_epoch):
    parameters = exclude_params_from_weight_decay(
        model,
        exclude=["bias", "norm", "query_pos", "modulation_weight", "B", "hash", "table"],
    )
    optimizer_name = str(config.optimizer).lower()
    fused_kwargs = {}
    if torch.cuda.is_available():
        try:
            if "fused" in inspect.signature(torch.optim.AdamW).parameters:
                fused_kwargs = {"fused": True}
        except (TypeError, ValueError):
            pass
    if optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(
            parameters,
            lr=float(config.learning_rate),
            weight_decay=float(getattr(config, "weight_decay", 1.0e-4)),
            **fused_kwargs,
        )
    elif optimizer_name == "adam":
        adam_kwargs = {}
        if torch.cuda.is_available():
            try:
                if "fused" in inspect.signature(torch.optim.Adam).parameters:
                    adam_kwargs = {"fused": True}
            except (TypeError, ValueError):
                pass
        optimizer = torch.optim.Adam(parameters, lr=float(config.learning_rate), weight_decay=1.0e-5, **adam_kwargs)
    else:
        raise ValueError(f"Unsupported SHIFT-Crash optimizer: {config.optimizer}")

    total_steps = max(1, int(config.epochs) * max(1, int(steps_per_epoch)))
    if str(config.scheduler).lower() == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_steps)
    else:
        raise ValueError(f"Unsupported SHIFT-Crash scheduler: {config.scheduler}")
    return optimizer, scheduler


def _load_partial(model, path, device):
    if not path:
        return
    checkpoint = torch.load(path, map_location=device)
    _restore_checkpoint_conditioning(model, checkpoint)
    source = checkpoint.get("model_state_dict", checkpoint)
    # Checkpoints saved from an externally wrapped DataParallel model may
    # retain the wrapper prefix even though this trainer saves the unwrapped
    # model.  Normalize it before matching shapes so weight-only warm starts
    # work in either direction.
    if source and all(key.startswith("module.") for key in source):
        source = {key[len("module."):]: value for key, value in source.items()}
    target_model = _unwrap(model)
    target = target_model.state_dict()
    filtered = {key: value for key, value in source.items() if key in target and target[key].shape == value.shape}
    target.update(filtered)
    target_model.load_state_dict(target, strict=False)
    print(f"[init] Loaded {len(filtered)} compatible tensors from {path}.")


def _load_full(model, optimizer, scheduler, scaler, path, device):
    checkpoint = torch.load(path, map_location=device)
    _restore_checkpoint_conditioning(model, checkpoint)
    source = checkpoint["model_state_dict"]
    if source and all(key.startswith("module.") for key in source):
        source = {key[len("module."):]: value for key, value in source.items()}
    _unwrap(model).load_state_dict(source, strict=True)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    if checkpoint.get("scaler_state_dict"):
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
    return int(checkpoint.get("epoch", -1)) + 1, int(checkpoint.get("global_step", 0)), float(checkpoint.get("best_val_rel_l2", np.inf))


def _conditioning_checkpoint_state(model):
    base_model = _unwrap(model)
    metadata = {
        "input_channels": int(getattr(base_model, "conditioning_input_channels", 0)),
        "parameter_channels": int(getattr(base_model, "conditioning_channels", 0)),
        "parameter_indices": [int(index) for index in getattr(base_model, "conditioning_parameter_indices", ())],
        "conditioning_hidden_dim": int(getattr(base_model, "conditioning_hidden_dim", 0)),
        "residual_update_scale": float(getattr(base_model, "residual_update_scale", 1.0)),
        "normalize_residuals": bool(getattr(base_model, "normalize_residuals", False)),
    }
    explicit_mode = getattr(base_model, "shift_crash_conditioning_mode", None)
    if explicit_mode is not None:
        metadata.update({
            "mode": str(explicit_mode),
            "residual_scale": 0.0 if explicit_mode == "token_only" else 1.0,
            "shift_scale": 0.0 if explicit_mode == "token_only" else 1.0,
        })
        return metadata
    for module in base_model.modules():
        if isinstance(module, Modulator):
            metadata.update({
                "mode": str(getattr(module, "conditioning_mode", "direct")),
                "residual_scale": float(getattr(module, "conditioning_residual_scale", 1.0)),
                "shift_scale": float(getattr(module, "conditioning_shift_scale", 1.0)),
            })
            return metadata
    return metadata


def _restore_checkpoint_conditioning(model, checkpoint):
    conditioning = checkpoint.get("conditioning")
    base_model = _unwrap(model)
    if not conditioning or not hasattr(base_model, "configure_conditioning"):
        return
    expected_indices = tuple(int(index) for index in getattr(base_model, "conditioning_parameter_indices", ()))
    saved_indices = conditioning.get("parameter_indices")
    if saved_indices is not None and tuple(int(index) for index in saved_indices) != expected_indices:
        raise ValueError(
            "Checkpoint conditioning indices do not match the current SHIFT-Crash model: "
            f"checkpoint={tuple(saved_indices)}, current={expected_indices}."
        )
    saved_input_channels = conditioning.get("input_channels")
    current_input_channels = int(getattr(base_model, "conditioning_input_channels", 0))
    if saved_input_channels is not None and int(saved_input_channels) != current_input_channels:
        raise ValueError(
            "Checkpoint conditioning input width does not match the current SHIFT-Crash model: "
            f"checkpoint={int(saved_input_channels)}, current={current_input_channels}."
        )
    saved_mode = conditioning.get("mode")
    current_mode = getattr(base_model, "shift_crash_conditioning_mode", None)
    if current_mode is not None and saved_mode is not None and str(saved_mode) != str(current_mode):
        raise ValueError(
            "SHIFT-Crash checkpoint conditioning schema does not match the current model: "
            f"checkpoint={saved_mode!r}, current={current_mode!r}. "
            "Start this corrected model from scratch; loading an older FiLM-conditioned checkpoint would mix "
            "incompatible conditioning mechanisms."
        )
    base_model.configure_conditioning(
        mode=saved_mode or "bounded_residual",
        residual_scale=float(conditioning.get("residual_scale", 0.25)),
        shift_scale=float(conditioning.get("shift_scale", 0.25)),
    )
    print(
        f"[checkpoint] Restored conditioning: mode={conditioning.get('mode')}, "
        f"residual_scale={float(conditioning.get('residual_scale', 0.25)):.4g}, "
        f"shift_scale={float(conditioning.get('shift_scale', 0.25)):.4g}, "
        f"indices={expected_indices}"
    )


def _save_checkpoint(
    path,
    epoch,
    model,
    optimizer,
    scheduler,
    scaler,
    global_step,
    best_val_rel_l2,
    metrics,
    satloss7,
    coordinate_normalization,
):
    base_model = _unwrap(model)
    torch.save(
        {
            "epoch": int(epoch),
            "global_step": int(global_step),
            "best_val_rel_l2": float(best_val_rel_l2),
            "model_state_dict": _unwrap(model).state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "loss": float(metrics.get("loss", np.nan)),
            "rel_l2_loss": float(metrics.get("rel_l2", np.nan)),
            "metric_values": {key: float(value) for key, value in metrics.items()},
            "dataset": "SHIFT_CRASH",
            "satloss7": bool(satloss7),
            "conditioning": _conditioning_checkpoint_state(model),
            "static_inputs": {
                "geometry_feature_channels": int(getattr(base_model, "geometry_feature_channels", 0)),
                "query_feature_channels": int(getattr(base_model, "query_feature_channels", 0)),
                "part_embedding_size": int(getattr(base_model, "part_embedding_size", 0)),
                "part_embedding_dim": int(getattr(base_model, "part_embedding_dim", 0)),
                "continuous_file": "static_geometry_features.npy",
                "categorical_file": "part_id_embedding_indices.npy",
                "binary_file": "front_rail_mask.npy",
                "feature_encoding": "channelwise_train_standardized_v3",
                "geometry_anchor_sampling": "rail_aware_within_supplied_view_v5",
                "case_conditioning": "all_six_design_variables_single_token_with_global_residual_v6",
                "validation_protocol": "deterministic_all_satloss7_families_v6",
                "coordinate_normalization": str(coordinate_normalization),
                "local_query_geometry": {
                    "enabled": bool(getattr(base_model, "use_local_query_geometry", False)),
                    "support_points": int(getattr(base_model, "local_query_support_points", 0)),
                    "neighbors": int(getattr(base_model, "local_query_neighbors", 0)),
                    "point_order_independent": True,
                },
            },
        },
        path,
    )


def _prepare_sat_views(batch, config, epoch, batch_index):
    geometry = batch["geometry"]
    density = batch["geometry_log_density"]
    geometry_features = batch["geometry_features"]
    geometry_part_ids = batch["geometry_part_ids"]
    seed_base = int(config.random_seed) + 1000003 * int(epoch) + 10007 * int(batch_index)
    family_generator = torch.Generator(device="cpu")
    family_generator.manual_seed(seed_base + 5)
    family = sample_shared_shift_family(
        getattr(config, "shared_shift_family_probabilities", [1 / 3] * 3), family_generator
    )

    view_points = int(config.view_geometry_points)
    if family == 0:
        mode = "inverse_density_wor"
        primary_generator = torch.Generator(device="cpu")
        primary_generator.manual_seed(seed_base + 17)
        secondary_generator = torch.Generator(device="cpu")
        secondary_generator.manual_seed(seed_base + 23)
        beta_1 = sample_uniform_beta(config.shared_shift_beta_min, config.shared_shift_beta_max, primary_generator)
        beta_2 = sample_uniform_beta(config.shared_shift_beta_min, config.shared_shift_beta_max, secondary_generator)
        geo_1, _density_1, _, idx_1 = sample_geometry_view(
            geometry, density, view_points, mode, beta_1, 1.0, seed_base + 11, return_indices=True
        )
        geo_2, _density_2, _, idx_2 = sample_geometry_view(
            geometry, density, view_points, mode, beta_2, 1.0, seed_base + 29, return_indices=True
        )
        label = "beta"
        values = (beta_1, beta_2)
    else:
        axis = 1 if family == 1 else 0
        mode = "sinusoidal_axis_mixture_wor"
        primary_generator = torch.Generator(device="cpu")
        primary_generator.manual_seed(seed_base + 17)
        secondary_generator = torch.Generator(device="cpu")
        secondary_generator.manual_seed(seed_base + 23)
        sine_1 = sample_uniform_beta(config.shared_shift_sine_min, config.shared_shift_sine_max, primary_generator)
        sine_2 = sample_uniform_beta(config.shared_shift_sine_min, config.shared_shift_sine_max, secondary_generator)
        geo_1, _density_1, _, idx_1 = sample_geometry_view(
            geometry,
            None,
            view_points,
            mode,
            0.0,
            1.0,
            seed_base + 11,
            sinusoidal_axis=axis,
            sinusoidal_mix_fraction=sine_1,
            return_indices=True,
        )
        geo_2, _density_2, _, idx_2 = sample_geometry_view(
            geometry,
            None,
            view_points,
            mode,
            0.0,
            1.0,
            seed_base + 29,
            sinusoidal_axis=axis,
            sinusoidal_mix_fraction=sine_2,
            return_indices=True,
        )
        label = "sine_y" if axis == 1 else "sine_x"
        values = (sine_1, sine_2)
    features_1 = _gather_batch_points(geometry_features, idx_1)
    features_2 = _gather_batch_points(geometry_features, idx_2)
    part_ids_1 = _gather_batch_points(geometry_part_ids, idx_1)
    part_ids_2 = _gather_batch_points(geometry_part_ids, idx_2)
    return geo_1, geo_2, features_1, features_2, part_ids_1, part_ids_2, label, values


def _satloss7_weights(config):
    weights = [float(value) for value in getattr(config, "config_task_base_weights", [0.2, 0.2, 0.6])]
    if len(weights) != 3 or any(value < 0.0 for value in weights) or sum(weights) <= 0.0:
        raise ValueError("SHIFT-Crash SATLoss7 requires three non-negative task weights with a positive sum.")
    total = sum(weights)
    return [value / total for value in weights]


def _train_batch(model, batch, optimizer, scheduler, scaler, config, device, dtype, amp, satloss7, epoch, batch_index):
    query = _move(batch["query"], device)
    query_features = _move(batch["query_features"], device)
    query_part_ids = _move(batch["query_part_ids"], device)
    target = _move(batch["target"], device)
    params = _move(batch["params"], device)

    if satloss7:
        geo_1, geo_2, features_1, features_2, part_ids_1, part_ids_2, family, shift_values = _prepare_sat_views(
            batch, config, epoch, batch_index
        )
        geometry = torch.cat([geo_1, geo_2], dim=0).to(device, non_blocking=True)
        geometry_features = torch.cat([features_1, features_2], dim=0).to(device, non_blocking=True)
        geometry_part_ids = torch.cat([part_ids_1, part_ids_2], dim=0).to(device, non_blocking=True)
        query_pair = torch.cat([query, query], dim=0)
        query_features_pair = torch.cat([query_features, query_features], dim=0)
        query_part_ids_pair = torch.cat([query_part_ids, query_part_ids], dim=0)
        target_pair = torch.cat([target, target], dim=0)
        params_pair = torch.cat([params, params], dim=0)
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=amp):
            prediction_pair = _forward(
                model,
                geometry,
                query_pair,
                params_pair,
                geometry_features=geometry_features,
                query_features=query_features_pair,
                geometry_part_ids=geometry_part_ids,
                query_part_ids=query_part_ids_pair,
            )
        prediction_1, prediction_2 = prediction_pair.chunk(2, dim=0)
        supervised_1 = _relative_l2(prediction_1, target)
        supervised_2 = _relative_l2(prediction_2, target)
        consistency = _relative_prediction_consistency(
            prediction_1,
            prediction_2,
            target,
            symmetric_detached=bool(config.prediction_consistency_symmetric_detached),
        )
        losses = OrderedDict(supervised_primary=supervised_1, supervised_secondary=supervised_2, prediction_consistency=consistency)
        weights = _satloss7_weights(config)
        warmup_epochs = int(getattr(config, "consistency_warmup_epochs", 0))
        warmup = 1.0 if warmup_epochs <= 0 else min(1.0, float(epoch + 1) / float(warmup_epochs))
        consistency_weight = warmup * float(getattr(config, "prediction_consistency_weight", 1.0))
        loss = (
            weights[0] * supervised_1
            + weights[1] * supervised_2
            + weights[2] * consistency_weight * consistency
        )
    else:
        geometry = _move(batch["geometry"], device)
        geometry_features = _move(batch["geometry_features"], device)
        geometry_part_ids = _move(batch["geometry_part_ids"], device)
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=amp):
            prediction = _forward(
                model,
                geometry,
                query,
                params,
                geometry_features=geometry_features,
                query_features=query_features,
                geometry_part_ids=geometry_part_ids,
                query_part_ids=query_part_ids,
            )
        supervised_1 = _relative_l2(prediction, target)
        losses = OrderedDict(supervised=supervised_1)
        loss = supervised_1
        family, shift_values = "uniform", (0.0, 0.0)

    if not torch.isfinite(loss):
        optimizer.zero_grad(set_to_none=True)
        return None
    optimizer.zero_grad(set_to_none=True)
    previous_scale = scaler.get_scale() if scaler.is_enabled() else None
    scaler.scale(loss).backward()
    collect_conditioning_diagnostics = bool(getattr(config, "conditioning_diagnostics", False))
    if config.gradient_norm is not None or collect_conditioning_diagnostics:
        scaler.unscale_(optimizer)
        if config.gradient_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.gradient_norm))
    diagnostics = _conditioning_gradient_diagnostics(model) if collect_conditioning_diagnostics else {}
    scaler.step(optimizer)
    scaler.update()
    # Do not advance the cosine schedule when AMP rejects an overflowing step.
    if previous_scale is None or scaler.get_scale() >= previous_scale:
        scheduler.step()
    return (
        float(loss.detach().float().item()),
        {key: float(value.detach().float().item()) for key, value in losses.items()},
        family,
        shift_values,
        diagnostics,
    )


@torch.inference_mode()
def _evaluate(model, loader, config, device, dtype, amp, satloss7):
    model.eval()
    sums = {"loss": 0.0, "rel_l2": 0.0}
    if satloss7:
        sums.update({
            "aligned_rel_l2": 0.0,
            "beta_rel_l2": 0.0,
            "sine_y_rel_l2": 0.0,
            "sine_x_rel_l2": 0.0,
            "shifted_rel_l2": 0.0,
            "robust_rel_l2": 0.0,
        })
    count = 0
    for batch in tqdm(loader, desc="Eval", leave=False, dynamic_ncols=True):
        query = _move(batch["query"], device)
        query_features = _move(batch["query_features"], device)
        query_part_ids = _move(batch["query_part_ids"], device)
        target = _move(batch["target"], device)
        params = _move(batch["params"], device)
        batch_size = int(query.shape[0])
        if satloss7:
            geometry = batch["geometry"]
            density = batch["geometry_log_density"]
            geometry_features = batch["geometry_features"]
            geometry_part_ids = batch["geometry_part_ids"]
            # Validation follows every family used by SATLoss7 training.  Each
            # model call gets explicit per-case support seeds, so a checkpoint
            # is selected by model quality rather than random encoder samples.
            view_specs = (
                ("aligned", "uniform_wor", None, 0.0, None, 0.0, 101),
                ("beta", "inverse_density_wor", density, 1.0, None, 0.0, 200003),
                ("sine_y", "sinusoidal_axis_mixture_wor", None, 0.0, 1, 1.0, 400009),
                ("sine_x", "sinusoidal_axis_mixture_wor", None, 0.0, 0, 1.0, 600011),
            )
            losses = {}
            for name, mode, view_density, beta, axis, sine_mix, offset in view_specs:
                view_geo, _d, _m, view_idx = sample_geometry_view(
                    geometry,
                    view_density,
                    int(config.eval_view_geometry_points),
                    mode,
                    beta,
                    1.0,
                    int(config.random_seed) + offset + count,
                    sinusoidal_axis=axis,
                    sinusoidal_mix_fraction=sine_mix,
                    return_indices=True,
                )
                view_features = _gather_batch_points(geometry_features, view_idx).to(device, non_blocking=True)
                view_part_ids = _gather_batch_points(geometry_part_ids, view_idx).to(device, non_blocking=True)
                with torch.autocast(device_type=device.type, dtype=dtype, enabled=amp):
                    prediction = _forward(
                        model,
                        view_geo.to(device, non_blocking=True),
                        query,
                        params,
                        geometry_features=view_features,
                        query_features=query_features,
                        geometry_part_ids=view_part_ids,
                        query_part_ids=query_part_ids,
                        sampling_seeds=_case_sampling_seeds(batch["case_id"], offset, device),
                    )
                losses[name] = _relative_l2(prediction, target)
            shifted = (losses["beta"] + losses["sine_y"] + losses["sine_x"]) / 3.0
            robust = (losses["aligned"] + losses["beta"] + losses["sine_y"] + losses["sine_x"]) / 4.0
            sums["aligned_rel_l2"] += float(losses["aligned"].item()) * batch_size
            sums["beta_rel_l2"] += float(losses["beta"].item()) * batch_size
            sums["sine_y_rel_l2"] += float(losses["sine_y"].item()) * batch_size
            sums["sine_x_rel_l2"] += float(losses["sine_x"].item()) * batch_size
            sums["shifted_rel_l2"] += float(shifted.item()) * batch_size
            sums["robust_rel_l2"] += float(robust.item()) * batch_size
            sums["loss"] += float(robust.item()) * batch_size
            sums["rel_l2"] += float(losses["aligned"].item()) * batch_size
        else:
            geometry = _move(batch["geometry"], device)
            geometry_features = _move(batch["geometry_features"], device)
            geometry_part_ids = _move(batch["geometry_part_ids"], device)
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=amp):
                prediction = _forward(
                    model,
                    geometry,
                    query,
                    params,
                    geometry_features=geometry_features,
                    query_features=query_features,
                    geometry_part_ids=geometry_part_ids,
                    query_part_ids=query_part_ids,
                    sampling_seeds=_case_sampling_seeds(batch["case_id"], 101, device),
                )
            value = _relative_l2(prediction, target)
            sums["loss"] += float(value.item()) * batch_size
            sums["rel_l2"] += float(value.item()) * batch_size
        count += batch_size
    return {key: value / max(count, 1) for key, value in sums.items()}


def run_shift_crash_training(cfg, satloss7=False, model_cls=SMART):
    config = cfg.experiment
    if str(config.loss_fn).lower() != "rel_l2":
        raise ValueError("SHIFT-Crash SMART is aligned with the repository RelL2 objective; set experiment.loss_fn=rel_l2.")
    if satloss7:
        if str(getattr(config, "task_weighting_method", "fixed_sum")).lower() not in {"fixed_sum", "fixed-sum", "fixedsum"}:
            raise ValueError("SHIFT-Crash SATLoss7 requires task_weighting_method=fixed_sum.")
        if not bool(getattr(config, "use_prediction_consistency", True)):
            raise ValueError("SHIFT-Crash SATLoss7 requires use_prediction_consistency=True.")
        if not bool(getattr(config, "fuse_consistency_views", True)):
            raise ValueError("SHIFT-Crash SATLoss7 requires fuse_consistency_views=True.")
        if not bool(getattr(config, "fixed_sum_use_view_losses", True)):
            raise ValueError("SHIFT-Crash SATLoss7 requires fixed_sum_use_view_losses=True.")
        if str(getattr(config, "train_shared_shift_sampling_mode", "random_beta_sine")).lower() not in {
            "random_beta_sine",
            "random_beta_sine_axis",
            "shared_random",
        }:
            raise ValueError("SHIFT-Crash SATLoss7 requires the shared random beta/sine shift family.")
    run = initialize_wandb(config, cfg.wandb)
    device = initialize_gpu(int(config.random_seed), high_precision=False)
    precisions = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    dtype = precisions.get(str(config.precision), torch.float16)
    amp = bool(config.amp) and device.type == "cuda"

    if satloss7:
        train_geometry_points = 0
        train_density = True
    else:
        train_geometry_points = int(config.num_body_points)
        train_density = False
    train_data = ShiftCrashDataset(
        config.data_path,
        split="train",
        geometry_points=train_geometry_points,
        query_points=int(config.num_query_points),
        seed=int(config.random_seed),
        epoch_seeded_sampling=bool(config.geometry_epoch_seeded_sampling),
        return_log_density=train_density,
        density_voxel_resolution=int(config.density_voxel_resolution),
        coordinate_normalization=str(getattr(config, "coordinate_normalization", "global_bounds")),
    )
    val_data = ShiftCrashDataset(
        config.data_path,
        split=str(config.evaluation_split),
        geometry_points=train_geometry_points,
        query_points=int(config.eval_query_points),
        seed=int(config.random_seed),
        epoch_seeded_sampling=False,
        deterministic_geometry_sampling=True,
        deterministic_query_sampling=True,
        return_log_density=train_density,
        density_voxel_resolution=int(config.density_voxel_resolution),
        coordinate_normalization=str(getattr(config, "coordinate_normalization", "global_bounds")),
    )
    loader_kwargs = {
        "batch_size": int(config.batch_size),
        "num_workers": int(config.num_workers),
        "pin_memory": bool(config.pin_memory),
    }
    if int(config.num_workers) > 0:
        loader_kwargs.update(prefetch_factor=int(config.prefetch_factor), persistent_workers=True)
    train_loader = torch.utils.data.DataLoader(train_data, shuffle=True, **loader_kwargs)
    val_loader = torch.utils.data.DataLoader(val_data, shuffle=False, **loader_kwargs)
    cuda_batch_prefetch = bool(getattr(config, "cuda_batch_prefetch", True)) and device.type == "cuda"
    keep_cpu_keys = (
        {"geometry", "geometry_log_density", "geometry_features", "geometry_part_ids"}
        if satloss7
        else set()
    )
    train_batches = _CudaPrefetchLoader(train_loader, device, keep_cpu_keys) if cuda_batch_prefetch else train_loader
    val_batches = _CudaPrefetchLoader(val_loader, device, keep_cpu_keys) if cuda_batch_prefetch else val_loader
    print(f"[dataloader] workers={config.num_workers}, prefetch_factor={config.prefetch_factor}, cuda_batch_prefetch={cuda_batch_prefetch}, keep_cpu_keys={sorted(keep_cpu_keys)}")

    conditioning_indices = tuple(
        int(index) for index in getattr(config, "conditioning_parameter_indices", (0, 1, 2, 3, 4, 5))
    )
    conditioning_input_channels = int(getattr(config, "conditioning_input_channels", 6))
    architecture_kwargs = dict(config.architecture)
    configured_parameter_channels = architecture_kwargs.pop("parameter_channels", None)
    if configured_parameter_channels is not None and int(configured_parameter_channels) != len(conditioning_indices):
        raise ValueError(
            "SHIFT-Crash architecture.parameter_channels must equal the number of selected conditioning parameters; "
            f"received {configured_parameter_channels} and {len(conditioning_indices)}."
        )
    model_kwargs = {
        "spatial_dim": 3,
        "surface_channels": 3,
        # SMART currently requires a positive volume head.  The empty volume
        # query means this head is never evaluated or supervised.
        "volume_channels": 1,
        "parameter_channels": len(conditioning_indices),
        "conditioning_input_channels": conditioning_input_channels,
        "conditioning_parameter_indices": conditioning_indices,
        "geometry_feature_channels": int(getattr(config, "geometry_feature_channels", 8)),
        "query_feature_channels": int(getattr(config, "query_feature_channels", 8)),
        "part_embedding_size": int(getattr(config, "part_embedding_size", 906)),
        "part_embedding_dim": int(getattr(config, "part_embedding_dim", 16)),
        **architecture_kwargs,
    }
    print(f"Model kwargs: {model_kwargs}")
    print(
        "[static inputs] continuous=normal/thickness/area/rail-material(7), "
        "binary=front_rail_mask(1), categorical=part_id_embedding(906 entries), "
        "query features use the same node indices as query coordinates."
    )
    model = model_cls(**model_kwargs).to(device)
    base_model = _unwrap(model)
    if hasattr(base_model, "configure_conditioning"):
        base_model.configure_conditioning(
            mode=getattr(config, "conditioning_mode", "bounded_residual"),
            residual_scale=float(getattr(config, "conditioning_residual_scale", 0.25)),
            shift_scale=float(getattr(config, "conditioning_shift_scale", 0.25)),
        )
        print(
            f"[conditioning] mode={getattr(config, 'conditioning_mode', 'bounded_residual')}, "
            f"residual_scale={float(getattr(config, 'conditioning_residual_scale', 0.25)):.4g}, "
            f"shift_scale={float(getattr(config, 'conditioning_shift_scale', 0.25)):.4g}"
        )
    if bool(getattr(config, "initialize_model_weights", False)) and hasattr(base_model, "initialize_shift_crash_weights"):
        base_model.initialize_shift_crash_weights()
        print("[init] Applied SMART initialization for a fresh SHIFT-Crash model.")
    if str(config.multi_gpu_strategy).lower() == "data_parallel" and torch.cuda.device_count() > 1:
        model = DataParallel(model)
        print(f"[SHIFT-Crash] DataParallel devices={list(range(torch.cuda.device_count()))}")
    print(f"Total parameters: {count_model_params(model)}")
    if hasattr(_unwrap(model), "conditioning_channels"):
        print(
            f"[conditioning] SMART FiLM parameters: channels={_unwrap(model).conditioning_channels}, "
            f"input_channels={_unwrap(model).conditioning_input_channels}, "
            f"selected_indices={_unwrap(model).conditioning_parameter_indices}, "
            f"hidden_dim={_unwrap(model).conditioning_hidden_dim}, "
            f"residual_update_scale={_unwrap(model).residual_update_scale:.4g}, "
            f"normalize_residuals={_unwrap(model).normalize_residuals}, "
            "shape=[batch,1,selected_channels], normalized_by_dataset_stats=True"
        )
    checkpoint_name = get_model_checkpoint_name(config)
    print(f"Checkpoint name: {checkpoint_name}")

    optimizer, scheduler = _make_optimizer(model, config, len(train_loader))
    scaler = make_grad_scaler(config)
    start_epoch = 0
    global_step = 0
    best_val_rel_l2 = np.inf
    epochs_without_improvement = 0
    early_stopping_patience = int(getattr(config, "early_stopping_patience", 0))
    early_stopping_min_delta = float(getattr(config, "early_stopping_min_delta", 0.0))
    resume_ckpt = str(getattr(config, "resume_ckpt", "")).strip()
    init_ckpt = str(getattr(config, "init_ckpt", "")).strip()
    if bool(getattr(config, "resume_full_state", False)):
        if not resume_ckpt:
            raise ValueError("resume_full_state=True requires experiment.resume_ckpt.")
        start_epoch, global_step, best_val_rel_l2 = _load_full(model, optimizer, scheduler, scaler, resume_ckpt, device)
    elif resume_ckpt:
        _load_partial(model, resume_ckpt, device)
    elif init_ckpt:
        _load_partial(model, init_ckpt, device)

    print(
        f"[SHIFT-Crash] protocol={'SATLoss7' if satloss7 else 'vanilla SMART'}, "
        f"train_cases={len(train_data)}, val_cases={len(val_data)}, "
        f"epochs={config.epochs}, geometry_budget={train_geometry_points or 'full source'}, "
        f"query_budget={config.num_query_points}"
    )

    try:
        for epoch in tqdm(range(start_epoch, int(config.epochs)), desc="Epochs", dynamic_ncols=True):
            train_data.set_epoch(epoch)
            model.train()
            running = {
                "loss": 0.0,
                "batches": 0.0,
                "supervised": 0.0,
                "supervised_primary": 0.0,
                "supervised_secondary": 0.0,
                "prediction_consistency": 0.0,
            }
            t0 = default_timer()
            train_pbar = tqdm(train_batches, desc=f"Train {epoch + 1}/{config.epochs}", leave=False, dynamic_ncols=True)
            for batch_index, batch in enumerate(train_pbar):
                result = _train_batch(model, batch, optimizer, scheduler, scaler, config, device, dtype, amp, satloss7, epoch, batch_index)
                if result is None:
                    print(f"[warn] Non-finite SHIFT-Crash loss at epoch={epoch} batch={batch_index}; skipped.")
                    continue
                loss_value, terms, family, shifts, conditioning_diagnostics = result
                running["loss"] += loss_value
                running["batches"] += 1.0
                for key, value in terms.items():
                    if key not in running:
                        running[key] = 0.0
                    running[key] += value
                for key, value in conditioning_diagnostics.items():
                    if key not in running:
                        running[key] = 0.0
                    running[key] += value
                global_step += 1
                if run is not None and (batch_index % int(config.log_every_n_steps) == 0 or batch_index == len(train_loader) - 1):
                    batch_log = {
                        "train/batch_loss": loss_value,
                        "train/batch_supervised_primary": terms.get("supervised_primary", terms.get("supervised", loss_value)),
                        "train/batch_supervised_secondary": terms.get("supervised_secondary", 0.0),
                        "train/batch_prediction_consistency": terms.get("prediction_consistency", 0.0),
                        "train/shift_family": {"uniform": 0, "beta": 1, "sine_y": 2, "sine_x": 3}.get(family, -1),
                        "train/shift_value_1": float(shifts[0]),
                        "train/shift_value_2": float(shifts[1]),
                        "lr": scheduler.get_last_lr()[0],
                        "epoch": epoch,
                    }
                    batch_log.update({f"train/batch_{key}": value for key, value in conditioning_diagnostics.items()})
                    wandb.log(batch_log, step=global_step)
                train_pbar.set_postfix(loss=f"{loss_value:.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

            val_metrics = _evaluate(model, val_batches, config, device, dtype, amp, satloss7)
            denominator = max(running["batches"], 1.0)
            train_metrics = {key: value / denominator for key, value in running.items() if key != "batches"}
            criterion = float(val_metrics["robust_rel_l2"] if satloss7 else val_metrics["rel_l2"])
            if criterion < best_val_rel_l2 - early_stopping_min_delta:
                best_val_rel_l2 = criterion
                epochs_without_improvement = 0
                _save_checkpoint(
                    os.path.join("checkpoints", checkpoint_name + "_best.pt"),
                    epoch, model, optimizer, scheduler, scaler, global_step, best_val_rel_l2,
                    val_metrics, satloss7, config.coordinate_normalization,
                )
            else:
                epochs_without_improvement += 1
            _save_checkpoint(
                os.path.join("checkpoints", checkpoint_name + "_last.pt"),
                epoch, model, optimizer, scheduler, scaler, global_step, best_val_rel_l2,
                val_metrics, satloss7, config.coordinate_normalization,
            )
            print(
                f"epoch: {epoch}, time: {default_timer() - t0:.2f}s, "
                f"train loss: {train_metrics['loss']:.5f}, "
                f"val rel_l2: {val_metrics['rel_l2']:.5f}"
                + (f", shifted rel_l2: {val_metrics['shifted_rel_l2']:.5f}, robust rel_l2: {val_metrics['robust_rel_l2']:.5f}" if satloss7 else "")
            )
            if run is not None:
                wandb.log(
                    {"epoch": epoch, "lr": scheduler.get_last_lr()[0], **{f"train/{key}": value for key, value in train_metrics.items()}, **{f"val/{key}": value for key, value in val_metrics.items()}},
                    step=global_step,
                )
            if early_stopping_patience > 0 and epochs_without_improvement >= early_stopping_patience:
                print(
                    f"[early-stop] No validation improvement larger than {early_stopping_min_delta:.3g} "
                    f"for {epochs_without_improvement} epochs; retained best checkpoint with "
                    f"rel_l2={best_val_rel_l2:.6f}."
                )
                break
    finally:
        if run is not None:
            run.finish()
