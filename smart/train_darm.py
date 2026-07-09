import gc
import inspect
import os
from timeit import default_timer

import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from omegaconf import DictConfig
from torch.nn import DataParallel
from tqdm.auto import tqdm

from data.datasets import get_dataset
from loss.losses import CombinedLoss, RelL2Loss
from models.smart.darm import DARM
from utils.surface_volume_trainer import (
    accumulate_channel_metrics,
    add_all_field_metrics,
    add_canonical_field_metrics,
    init_metric_dict,
    load_partial_state_dict,
)
from utils.utils import (
    apply_naca4_auto_point_budget,
    count_model_params,
    get_model_checkpoint_name,
    initialize_gpu,
    initialize_wandb,
    print_point_budget,
)


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def set_dataset_epoch(dataset, epoch):
    if hasattr(dataset, "set_epoch"):
        dataset.set_epoch(epoch)


def parse_batch(batch, params_dim):
    sample_info = None
    if params_dim > 0:
        if len(batch) == 7:
            geo_mesh, surf_mesh, surf_data, vol_mesh, vol_data, params, sample_info = batch
        elif len(batch) == 6:
            geo_mesh, surf_mesh, surf_data, vol_mesh, vol_data, params = batch
        else:
            raise ValueError(f"Unexpected parameterized batch size {len(batch)}")
    else:
        if len(batch) == 6:
            geo_mesh, surf_mesh, surf_data, vol_mesh, vol_data, sample_info = batch
            params = None
        elif len(batch) == 5:
            geo_mesh, surf_mesh, surf_data, vol_mesh, vol_data = batch
            params = None
        else:
            raise ValueError(f"Unexpected batch size {len(batch)}")
    return geo_mesh, surf_mesh, surf_data, vol_mesh, vol_data, params, sample_info


def move_to_device(value, device):
    if torch.is_tensor(value):
        if value.device == device:
            return value
        return value.to(device, non_blocking=True)
    if isinstance(value, tuple):
        return tuple(move_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [move_to_device(item, device) for item in value]
    if isinstance(value, dict):
        return {key: move_to_device(item, device) for key, item in value.items()}
    return value


def record_batch_stream(value, stream):
    if torch.is_tensor(value):
        if value.is_cuda:
            value.record_stream(stream)
        return
    if isinstance(value, dict):
        for item in value.values():
            record_batch_stream(item, stream)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            record_batch_stream(item, stream)


class CudaPrefetchLoader:
    def __init__(self, loader, device):
        self.loader = loader
        self.device = device
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
                next_batch = move_to_device(batch, self.device)

        preload()
        while next_batch is not None:
            torch.cuda.current_stream(device=self.device).wait_stream(self.stream)
            batch = next_batch
            record_batch_stream(batch, torch.cuda.current_stream(device=self.device))
            preload()
            yield batch


def load_full_training_state(model, optimizer, scheduler, scaler, checkpoint_path, device, amp_enabled):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    unwrap_model(model).load_state_dict(checkpoint["model_state_dict"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    if amp_enabled and "scaler_state_dict" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
    start_epoch = int(checkpoint.get("epoch", -1)) + 1
    global_step = int(checkpoint.get("global_step", 0))
    best_rel_l2 = float(checkpoint.get("best_rel_l2", np.inf))
    print(
        f"[resume] Restored full training state from {checkpoint_path}: "
        f"next_epoch={start_epoch}, global_step={global_step}, best_rel_l2={best_rel_l2:.6g}"
    )
    return start_epoch, global_step, best_rel_l2


def get_checkpoint_compatible_parameter_names(model, checkpoint_path, device):
    if not checkpoint_path:
        return set()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    source = checkpoint.get("model_state_dict", checkpoint)
    if any(key.startswith("module.") for key in source.keys()):
        source = {key.removeprefix("module."): value for key, value in source.items()}

    model_params = dict(unwrap_model(model).named_parameters())
    compatible_names = set()
    for key, value in source.items():
        if key in model_params and model_params[key].shape == value.shape:
            compatible_names.add(key)
    return compatible_names


def freeze_named_parameters(model, parameter_names):
    frozen = 0
    for name, param in unwrap_model(model).named_parameters():
        if name in parameter_names:
            param.requires_grad_(False)
            frozen += 1
    return frozen


def gather_points(points, idx):
    return torch.gather(points, 1, idx.unsqueeze(-1).expand(-1, -1, points.shape[-1]))


def expand_weights(weight, target):
    if weight is None:
        return None
    while weight.ndim < target.ndim:
        weight = weight.unsqueeze(-1)
    return weight.to(device=target.device, dtype=target.dtype)


def masked_mean(values, weight, eps=1.0e-6):
    if weight is None:
        return values.mean()
    weight = expand_weights(weight, values)
    denom = weight.sum().clamp_min(eps)
    return (values * weight).sum() / denom


def masked_response_norm(pred, target, weight, eps=1.0e-8):
    delta = pred.float() - target.float()
    response = torch.sqrt(delta.pow(2).mean(dim=-1, keepdim=True) + eps)
    return masked_mean(response, weight)


def pointwise_response_norm(pred, target, eps=1.0e-8):
    delta = pred.float() - target.float()
    return torch.sqrt(delta.pow(2).mean(dim=-1, keepdim=True) + eps)


def pointwise_feature_norm(values, eps=1.0e-8):
    return torch.sqrt(values.float().pow(2).mean(dim=-1, keepdim=True) + eps)


def build_displacement_targets(original_points, edited_points, eps=1.0e-6):
    displacement = (edited_points - original_points).float().norm(dim=-1, keepdim=True)
    max_disp = displacement.amax(dim=1, keepdim=True).clamp_min(eps)
    changed = (displacement / max_disp).clamp_(0.0, 1.0)
    unchanged = 1.0 - changed
    return (
        changed.to(dtype=original_points.dtype),
        unchanged.to(dtype=original_points.dtype),
        displacement.to(dtype=original_points.dtype),
    )


def locality_focus_loss(values, changed_weight, eps=1.0e-6):
    values = values.float().clamp_min(0.0)
    changed_weight = changed_weight.float().clamp_min(0.0)
    inside_mass = (values * changed_weight).sum(dim=1)
    total_mass = values.sum(dim=1).clamp_min(eps)
    focus = inside_mass / total_mass
    return (-torch.log(focus.clamp_min(eps))).mean()


def locality_separation_loss(values, changed_weight, unchanged_weight):
    changed_mean = masked_mean(values, changed_weight)
    unchanged_mean = masked_mean(values, unchanged_weight)
    return F.softplus(unchanged_mean - changed_mean)


def balanced_soft_bce(pred, target, eps=1.0e-6):
    with torch.autocast(device_type=pred.device.type, enabled=False):
        pred = pred.float().clamp(eps, 1.0 - eps)
        target = target.float().clamp(0.0, 1.0)
        pos_weight = target / target.sum(dim=1, keepdim=True).clamp_min(eps)
        neg_target = 1.0 - target
        neg_weight = neg_target / neg_target.sum(dim=1, keepdim=True).clamp_min(eps)
        weight = pos_weight + neg_weight
        loss = F.binary_cross_entropy(pred, target, reduction="none")
        return (loss * weight).sum() / weight.sum().clamp_min(eps)


def reliability_map_loss(pred_map, changed_weight, unchanged_weight):
    calibration = balanced_soft_bce(pred_map, changed_weight)
    separation = locality_separation_loss(pred_map, changed_weight, unchanged_weight)
    return 0.5 * (calibration + separation)


def ensure_scalar_tensor(value):
    if not torch.is_tensor(value):
        return value
    if value.ndim == 0:
        return value
    return value.mean()


def apply_gaussian_deformation(points, center, normal, sigma, edit_scale, ramp_power):
    delta = points - center.unsqueeze(1)
    dist2 = delta.float().pow(2).sum(dim=-1, keepdim=True)
    sigma2 = sigma.float().unsqueeze(1).unsqueeze(-1).pow(2).clamp_min(1.0e-6)
    weight = torch.exp(-0.5 * dist2 / sigma2).to(dtype=points.dtype)
    support_mask = dist2 <= (9.0 * sigma2)
    weight = weight * support_mask.to(dtype=weight.dtype)
    weight = weight.pow(float(ramp_power))
    if torch.is_tensor(edit_scale):
        scale = edit_scale.to(device=points.device, dtype=points.dtype).unsqueeze(1).unsqueeze(-1)
    else:
        scale = points.new_tensor(float(edit_scale))
    displacement = scale * weight * normal.unsqueeze(1)
    return points + displacement


def apply_deformation_field(points, centers, directions, sigmas, scales, ramp_power):
    edited = points
    num_lobes = int(centers.shape[1])
    for lobe_idx in range(num_lobes):
        edited = apply_gaussian_deformation(
            edited,
            centers[:, lobe_idx],
            directions[:, lobe_idx],
            sigmas[:, lobe_idx],
            edit_scale=scales[:, lobe_idx],
            ramp_power=ramp_power,
        )
    return edited


def build_local_edit(
    geo_mesh,
    min_points,
    max_points,
    edit_strength,
    ramp_power,
    candidate_points=8192,
):
    batch_size, num_points, _spatial_dim = geo_mesh.shape
    device = geo_mesh.device
    min_points = max(8, min(int(min_points), num_points))
    max_points = max(min_points, min(int(max_points), num_points))
    candidate_points = max(max_points, min(int(candidate_points), num_points))

    center_idx = torch.randint(0, num_points, (batch_size,), device=device, dtype=torch.long)
    center = geo_mesh[torch.arange(batch_size, device=device), center_idx]
    candidate_idx = torch.randint(0, num_points, (batch_size, candidate_points), device=device, dtype=torch.long)
    candidate_idx[:, 0] = center_idx
    candidate_points_geo = gather_points(geo_mesh, candidate_idx)
    dist2 = (candidate_points_geo - center.unsqueeze(1)).pow(2).sum(dim=-1)

    if min_points == max_points:
        patch_count = torch.full((batch_size,), min_points, device=device, dtype=torch.long)
    else:
        patch_count = torch.randint(min_points, max_points + 1, (batch_size,), device=device, dtype=torch.long)

    patch_rel_idx = torch.topk(dist2, k=max_points, dim=1, largest=False).indices
    patch_idx = candidate_idx.gather(1, patch_rel_idx)
    patch_points = gather_points(geo_mesh, patch_idx)
    patch_dist = torch.sqrt(dist2.gather(1, patch_rel_idx).clamp_min(1.0e-12))

    rank = torch.arange(max_points, device=device, dtype=torch.long).view(1, -1)
    valid = rank < patch_count.unsqueeze(1)
    patch_radius = patch_dist.gather(1, (patch_count - 1).unsqueeze(1)).squeeze(1).clamp_min(1.0e-6)
    radial = (1.0 - patch_dist / patch_radius.unsqueeze(1)).clamp_min(0.0)
    patch_weight = radial.pow(float(ramp_power)) * valid.to(dtype=geo_mesh.dtype)

    weight_sum = patch_weight.sum(dim=1, keepdim=True).clamp_min(1.0e-6)
    centroid = (patch_weight.unsqueeze(-1) * patch_points).sum(dim=1) / weight_sum
    centered = patch_points - centroid.unsqueeze(1)
    cov = torch.einsum("bki,bkj,bk->bij", centered.float(), centered.float(), patch_weight.float() / weight_sum.float())
    _eigvals, eigvecs = torch.linalg.eigh(cov)
    normal = eigvecs[..., 0].to(dtype=geo_mesh.dtype)
    tangent_a = eigvecs[..., 2].to(dtype=geo_mesh.dtype)
    tangent_b = eigvecs[..., 1].to(dtype=geo_mesh.dtype)

    edit_sigma = patch_radius.clamp_min(1.0e-6)
    primary_mix_a = torch.empty(batch_size, device=device, dtype=geo_mesh.dtype).uniform_(-0.35, 0.35)
    primary_mix_b = torch.empty(batch_size, device=device, dtype=geo_mesh.dtype).uniform_(-0.20, 0.20)
    primary_direction = F.normalize(
        normal + primary_mix_a.unsqueeze(-1) * tangent_a + primary_mix_b.unsqueeze(-1) * tangent_b,
        dim=-1,
    )
    secondary_offset = torch.empty(batch_size, device=device, dtype=geo_mesh.dtype).uniform_(-0.60, 0.60)
    secondary_center = center + secondary_offset.unsqueeze(-1) * patch_radius.unsqueeze(-1) * tangent_a
    secondary_mix_n = torch.empty(batch_size, device=device, dtype=geo_mesh.dtype).uniform_(-0.75, 0.75)
    secondary_mix_t = torch.empty(batch_size, device=device, dtype=geo_mesh.dtype).uniform_(-0.50, 0.50)
    secondary_direction = F.normalize(
        secondary_mix_n.unsqueeze(-1) * normal + tangent_b + secondary_mix_t.unsqueeze(-1) * tangent_a,
        dim=-1,
    )
    primary_scale = torch.empty(batch_size, device=device, dtype=geo_mesh.dtype).uniform_(-1.0, 1.0)
    primary_scale = primary_scale * float(edit_strength) * patch_radius
    secondary_scale = torch.empty(batch_size, device=device, dtype=geo_mesh.dtype).uniform_(-0.5, 0.5)
    secondary_scale = secondary_scale * float(edit_strength) * patch_radius

    centers = torch.stack([center, secondary_center], dim=1)
    directions = torch.stack([primary_direction, secondary_direction], dim=1)
    sigmas = torch.stack([edit_sigma, 0.65 * edit_sigma], dim=1)
    scales = torch.stack([primary_scale, secondary_scale], dim=1)
    edited_geo = apply_deformation_field(
        geo_mesh,
        centers,
        directions,
        sigmas,
        scales,
        ramp_power=ramp_power,
    )

    deformation_field = {
        "centers": centers,
        "directions": directions,
        "sigmas": sigmas,
        "scales": scales,
    }
    return edited_geo, deformation_field


def duplicate_batch(x):
    if x is None:
        return None
    if x.shape[0] == 1:
        return x.expand(2, *x.shape[1:])
    return torch.cat([x, x], dim=0)


def split_aux_batch(aux, split_index):
    left = {}
    right = {}
    for key, value in aux.items():
        if torch.is_tensor(value):
            if value.ndim == 0:
                left[key] = value
                right[key] = value
            elif value.shape[0] == split_index * 2:
                left[key] = value[:split_index]
                right[key] = value[split_index:]
            elif value.ndim == 1:
                shared = value.float().mean().to(dtype=value.dtype)
                left[key] = shared
                right[key] = shared
            else:
                left[key] = value[:split_index]
                right[key] = value[split_index:]
        elif isinstance(value, dict):
            left[key], right[key] = split_aux_batch(value, split_index)
        else:
            left[key] = value
            right[key] = value
    return left, right


def aux_query_mean(aux, *keys):
    total = None
    count = 0
    ref = None
    for key in keys:
        value = aux.get(key)
        if not torch.is_tensor(value):
            continue
        ref = value if ref is None else ref
        if value.numel() == 0:
            continue
        value_f = value.float()
        component = value_f.sum()
        total = component if total is None else (total + component)
        count += int(value_f.numel())
    if total is not None and count > 0:
        return ensure_scalar_tensor(total / float(count))
    if ref is not None:
        return ref.new_zeros((), dtype=torch.float32)
    return torch.tensor(0.0)


def normalized_loss_scales(
    loss_values,
    base_weights,
    reference_name="loss_supervised",
    min_scale=0.25,
    max_scale=4.0,
    eps=1.0e-8,
    target_multipliers=None,
):
    active_names = [name for name, weight in base_weights.items() if float(weight) > 0.0]
    if not active_names:
        return {name: 1.0 for name in loss_values}

    if reference_name not in active_names:
        reference_name = active_names[0]

    target_multipliers = dict(target_multipliers or {})
    reference_target = float(target_multipliers.get(reference_name, 1.0))
    reference_target = max(reference_target, eps)

    reference_value = abs(float(loss_values.get(reference_name, 0.0)))
    reference_contribution = abs(float(base_weights.get(reference_name, 0.0))) * reference_value
    if not np.isfinite(reference_contribution) or reference_contribution <= eps:
        return {name: 1.0 for name in loss_values}

    scales = {name: 1.0 for name in loss_values}
    for name in active_names:
        loss_value = abs(float(loss_values.get(name, 0.0)))
        weighted_contribution = abs(float(base_weights.get(name, 0.0))) * loss_value
        if not np.isfinite(weighted_contribution) or weighted_contribution <= eps:
            scales[name] = 1.0
            continue
        target_multiplier = float(target_multipliers.get(name, 1.0))
        target_contribution = reference_contribution * (target_multiplier / reference_target)
        scale = target_contribution / weighted_contribution
        scales[name] = float(np.clip(scale, float(min_scale), float(max_scale)))

    scales[reference_name] = 1.0
    return scales


def should_reweight_epoch(epoch, explicit_epochs, interval_epochs):
    if int(epoch) in explicit_epochs:
        return True
    if int(interval_epochs) <= 0:
        return False
    return int(epoch) > 0 and (int(epoch) % int(interval_epochs) == 0)


def _cuda_optimizer_impl_kwargs(optimizer_cls):
    if not torch.cuda.is_available():
        return {}
    try:
        parameters = inspect.signature(optimizer_cls).parameters
    except (TypeError, ValueError):
        return {}
    if "fused" in parameters:
        return {"fused": True}
    if "foreach" in parameters:
        return {"foreach": True}
    return {}


def build_named_param_groups(named_parameters, lr, weight_decay, exclude=None):
    exclude = tuple(exclude or ("bias", "norm", "query_pos", "lora_A", "lora_B"))
    decay = []
    no_decay = []
    for name, param in named_parameters:
        if not param.requires_grad:
            continue
        if any(token in name for token in exclude):
            no_decay.append(param)
        else:
            decay.append(param)

    groups = []
    if decay:
        groups.append({"params": decay, "lr": lr, "weight_decay": weight_decay})
    if no_decay:
        groups.append({"params": no_decay, "lr": lr, "weight_decay": 0.0})
    return groups


def create_darm_optimizer_scheduler_loss(model, config, train_loader, loss_dim=-2):
    named_parameters = [
        (name, param) for name, param in unwrap_model(model).named_parameters() if param.requires_grad
    ]
    base_lr = float(config.learning_rate)
    grouped_parameters = build_named_param_groups(named_parameters, lr=base_lr, weight_decay=1.0e-4)

    if config.optimizer == "adam":
        optimizer = torch.optim.Adam(
            grouped_parameters,
            lr=base_lr,
            weight_decay=1.0e-5,
            **_cuda_optimizer_impl_kwargs(torch.optim.Adam),
        )
    elif config.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            grouped_parameters,
            lr=base_lr,
            weight_decay=1.0e-4,
            **_cuda_optimizer_impl_kwargs(torch.optim.AdamW),
        )
    else:
        raise ValueError(f"Unsupported optimizer for DARM training: {config.optimizer}")

    scheduler_warmup_fraction = float(
        getattr(config, "scheduler_warmup_fraction", getattr(config, "scheduler_warumup_fraction", 0.2))
    )
    if config.scheduler == "one-cycle":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=base_lr,
            pct_start=scheduler_warmup_fraction,
            div_factor=1e2,
            final_div_factor=1e3,
            total_steps=config.epochs * len(train_loader),
        )
    elif config.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config.epochs * len(train_loader))
    elif config.scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.scheduler_step, gamma=config.scheduler_gamma)
    elif config.scheduler == "exponential":
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.9, last_epoch=-1)
    else:
        raise ValueError(f"Unsupported scheduler for DARM training: {config.scheduler}")

    if config.loss_fn == "mse":
        loss_fn = nn.MSELoss(reduction="mean")
    elif config.loss_fn == "l1":
        loss_fn = nn.L1Loss(reduction="mean")
    elif config.loss_fn == "rel_l2":
        loss_fn = RelL2Loss(dim=loss_dim, reduction="sum")
    else:
        raise ValueError(f"Unsupported loss for DARM training: {config.loss_fn}")

    rel_l2_loss_fn = RelL2Loss(dim=loss_dim, reduction="sum")
    lora_params = sum(param.numel() for name, param in named_parameters if ".lora_" in name)
    trainable_params = sum(param.numel() for _name, param in named_parameters)
    group_info = {
        "trainable_params": trainable_params,
        "lora_params": lora_params,
        "non_lora_params": trainable_params - lora_params,
        "lr": base_lr,
    }
    return optimizer, scheduler, loss_fn, rel_l2_loss_fn, group_info


def cleanup_after_oom(optimizer):
    optimizer.zero_grad(set_to_none=True)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def cleanup_after_invalid_step(optimizer, clear_cache):
    optimizer.zero_grad(set_to_none=True)
    if clear_cache:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def collect_non_finite_problems(value, prefix, problems, max_entries=8):
    if len(problems) >= max_entries:
        return
    if torch.is_tensor(value):
        if value.numel() == 0:
            return
        mask = ~torch.isfinite(value)
        if bool(mask.any().item()):
            bad_count = int(mask.sum().item())
            total_count = int(value.numel())
            problems.append(f"{prefix}: {bad_count}/{total_count} non-finite, shape={tuple(value.shape)}")
        return
    if isinstance(value, dict):
        for key, sub_value in value.items():
            collect_non_finite_problems(sub_value, f"{prefix}.{key}" if prefix else str(key), problems, max_entries=max_entries)
            if len(problems) >= max_entries:
                return
        return
    if isinstance(value, (list, tuple)):
        for idx, sub_value in enumerate(value):
            collect_non_finite_problems(sub_value, f"{prefix}[{idx}]", problems, max_entries=max_entries)
            if len(problems) >= max_entries:
                return


def find_non_finite_problems(named_values, max_entries=8):
    problems = []
    collect_non_finite_problems(named_values, "", problems, max_entries=max_entries)
    return problems


def summarize_non_finite_problems(problems):
    if not problems:
        return ""
    return "; ".join(problems)


def find_non_finite_gradients(model, max_entries=8):
    problems = []
    for name, param in unwrap_model(model).named_parameters():
        grad = param.grad
        if grad is None:
            continue
        mask = ~torch.isfinite(grad)
        if bool(mask.any().item()):
            bad_count = int(mask.sum().item())
            total_count = int(grad.numel())
            problems.append(f"grad.{name}: {bad_count}/{total_count} non-finite, shape={tuple(grad.shape)}")
            if len(problems) >= max_entries:
                break
    return problems


def should_run_periodic_check(step_index, interval):
    interval = max(1, int(interval))
    return step_index == 0 or ((step_index + 1) % interval == 0)


def build_checkpoint(
    epoch,
    model,
    optimizer,
    scheduler,
    scaler,
    test_loss,
    test_rel_l2,
    fields,
    global_step,
    best_rel_l2,
    metric_values,
):
    return {
        "epoch": int(epoch),
        "global_step": int(global_step),
        "best_rel_l2": float(best_rel_l2),
        "model_state_dict": unwrap_model(model).state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "loss": float(test_loss),
        "rel_l2_loss": float(test_rel_l2),
        "surface_fields": fields["surface"],
        "volume_fields": fields["volume"],
        "metric_values": metric_values,
    }


def evaluate_loader(
    model,
    loader,
    config,
    device,
    dtype,
    amp,
    params_dim,
    mean_surf,
    std_surf,
    mean_vol,
    std_vol,
    fields,
    rel_l2_loss_fn,
    combined_loss_fn,
    loss_fn,
    use_surface_supervision,
    cuda_batch_prefetch,
    eval_nonfinite_check_interval,
):
    metrics = init_metric_dict(fields["surface"], fields["volume"])
    model.eval()
    eval_sample_count = 0
    skipped_nonfinite_batches = 0

    eval_loader = CudaPrefetchLoader(loader, device) if cuda_batch_prefetch else loader

    with torch.inference_mode():
        for batch_idx, batch in enumerate(tqdm(eval_loader, desc="Eval", leave=False, dynamic_ncols=True)):
            geo_mesh, surf_mesh, surf_data, vol_mesh, vol_data, params, _sample_info = parse_batch(batch, params_dim)

            geo_mesh = move_to_device(geo_mesh, device)
            surf_mesh = move_to_device(surf_mesh, device)
            surf_data = move_to_device(surf_data, device)
            vol_mesh = move_to_device(vol_mesh, device)
            vol_data = move_to_device(vol_data, device)
            if params is not None:
                params = move_to_device(params, device)

            if config.dataset == "NACA4":
                surf_data = surf_data[..., :1]
                vol_data = torch.cat([vol_data[..., :1], vol_data[..., 2:4]], dim=-1)

            if should_run_periodic_check(batch_idx, eval_nonfinite_check_interval):
                input_problems = find_non_finite_problems(
                    {
                        "geo_mesh": geo_mesh,
                        "surf_mesh": surf_mesh,
                        "surf_data": surf_data,
                        "vol_mesh": vol_mesh,
                        "vol_data": vol_data,
                        "params": params,
                    },
                    max_entries=4,
                )
                if input_problems:
                    skipped_nonfinite_batches += 1
                    print(f"[eval/nonfinite] Skipping batch with invalid inputs: {summarize_non_finite_problems(input_problems)}")
                    continue

            with torch.autocast(device_type=str(device).split(":")[0], dtype=dtype, enabled=amp):
                y_hat_surf, y_hat_vol = model(geo_mesh, surf_mesh, vol_mesh, params)

            if use_surface_supervision:
                pred_surf = y_hat_surf * std_surf + mean_surf
                gt_surf = surf_data * std_surf + mean_surf
                batch_loss = ensure_scalar_tensor(combined_loss_fn(y_hat_surf.float(), y_hat_vol.float(), surf_data, vol_data))
                surface_rel_l2 = ensure_scalar_tensor(rel_l2_loss_fn(y_hat_surf.float(), surf_data))
            else:
                pred_surf = None
                gt_surf = None
                batch_loss = ensure_scalar_tensor(loss_fn(y_hat_vol.float(), vol_data))
                surface_rel_l2 = torch.tensor(0.0, device=device)

            pred_vol = y_hat_vol * std_vol + mean_vol
            gt_vol = vol_data * std_vol + mean_vol
            volume_rel_l2 = ensure_scalar_tensor(rel_l2_loss_fn(y_hat_vol.float(), vol_data))

            if should_run_periodic_check(batch_idx, eval_nonfinite_check_interval):
                eval_problems = find_non_finite_problems(
                    {
                        "y_hat_surf": y_hat_surf if use_surface_supervision else None,
                        "y_hat_vol": y_hat_vol,
                        "pred_surf": pred_surf if use_surface_supervision else None,
                        "pred_vol": pred_vol,
                        "batch_loss": batch_loss,
                        "surface_rel_l2": surface_rel_l2,
                        "volume_rel_l2": volume_rel_l2,
                    },
                    max_entries=4,
                )
                if eval_problems:
                    skipped_nonfinite_batches += 1
                    print(f"[eval/nonfinite] Skipping batch with invalid outputs: {summarize_non_finite_problems(eval_problems)}")
                    continue

            batch_size = surf_data.size(0)
            eval_sample_count += batch_size
            metrics["loss"] += batch_loss.item() * batch_size
            metrics["rel_l2_surf"] += surface_rel_l2.item() * batch_size
            metrics["rel_l2_vol"] += volume_rel_l2.item() * batch_size
            metrics["rel_l2"] += (surface_rel_l2 + volume_rel_l2).item() * batch_size
            if use_surface_supervision:
                accumulate_channel_metrics(metrics, "rel_l2_surf", pred_surf, gt_surf, fields["surface"], rel_l2_loss_fn, batch_size)
            accumulate_channel_metrics(metrics, "rel_l2_vol", pred_vol, gt_vol, fields["volume"], rel_l2_loss_fn, batch_size)

    denom = max(int(eval_sample_count), 1)
    for key in metrics.keys():
        metrics[key] /= denom
    if skipped_nonfinite_batches > 0:
        print(f"[eval/nonfinite] skipped_batches={skipped_nonfinite_batches}")
    return metrics


@hydra.main(version_base="1.2", config_path="config", config_name="drivaerml_darm")
def main(cfg: DictConfig):
    config = cfg.experiment
    wandb_config = cfg.wandb
    multi_gpu_strategy = str(getattr(config, "multi_gpu_strategy", "data_parallel")).lower()
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        raise RuntimeError("DARM uses plain python execution. Do not launch it with torchrun.")
    if multi_gpu_strategy not in {"data_parallel", "single"}:
        print(f"[DARM] Unsupported multi_gpu_strategy={multi_gpu_strategy!r}. Falling back to data_parallel.")
        multi_gpu_strategy = "data_parallel"

    run = initialize_wandb(config, wandb_config)
    initialize_gpu(config.random_seed, high_precision=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    gradient_norm = config.gradient_norm
    precisions = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    dtype = precisions.get(config.precision, torch.float16)
    if dtype == torch.bfloat16 and torch.cuda.is_available() and not torch.cuda.is_bf16_supported():
        print("[DARM] CUDA device does not report bfloat16 support. Falling back to float16.")
        dtype = torch.float16
    amp = bool(config.amp)
    print(f"Model {config.model_name}, random seed: {config.random_seed}, epochs: {config.epochs}, learning rate: {config.learning_rate}")
    print(gradient_norm, amp, dtype)

    train_data, test_data, stats, spatial_dim, surf_channels, vol_channels, params_dim, fields = get_dataset(config)

    def apply_vanilla_field_subset():
        nonlocal fields, surf_channels, vol_channels
        if config.dataset == "NACA4":
            fields = {"surface": ["pressure"], "volume": ["pressure", "velocity_x", "velocity_y"]}
            surf_channels = 1
            vol_channels = 3

    apply_vanilla_field_subset()
    print(f"[{config.model_name}] training signals -> surface: {fields['surface']} | volume: {fields['volume']}")

    point_info = apply_naca4_auto_point_budget(config, train_data, for_cat=False)
    if point_info is not None:
        print_point_budget(config.model_name, point_info)
        train_data, test_data, stats, spatial_dim, surf_channels, vol_channels, params_dim, fields = get_dataset(config)
        apply_vanilla_field_subset()
        print(f"[{config.model_name}] training signals -> surface: {fields['surface']} | volume: {fields['volume']}")

    use_surface_supervision = len(fields["surface"]) > 0
    set_dataset_epoch(train_data, 0)
    set_dataset_epoch(test_data, 0)

    prefetch_factor = int(getattr(config, "prefetch_factor", 2))
    pin_memory = bool(getattr(config, "pin_memory", True))
    num_workers = int(getattr(config, "num_workers", 0))
    cuda_batch_prefetch = bool(getattr(config, "cuda_batch_prefetch", device.type == "cuda")) and device.type == "cuda"
    dl_common = dict(batch_size=config.batch_size, num_workers=num_workers, pin_memory=pin_memory)
    if num_workers > 0:
        dl_common["prefetch_factor"] = prefetch_factor
        dl_common["persistent_workers"] = True

    train_loader = torch.utils.data.DataLoader(train_data, shuffle=True, **dl_common)
    test_loader = torch.utils.data.DataLoader(test_data, shuffle=False, **dl_common)

    mean_surf = stats[0][:surf_channels].to(device)
    std_surf = stats[1][:surf_channels].to(device)
    if config.dataset == "NACA4" and vol_channels == 2:
        mean_vol = stats[2][2:4].to(device)
        std_vol = stats[3][2:4].to(device)
    elif config.dataset == "NACA4" and vol_channels == 3:
        mean_vol = torch.stack([stats[2][0], stats[2][2], stats[2][3]]).to(device)
        std_vol = torch.stack([stats[3][0], stats[3][2], stats[3][3]]).to(device)
    else:
        mean_vol = stats[2][:vol_channels].to(device)
        std_vol = stats[3][:vol_channels].to(device)

    model_kwargs = {
        "spatial_dim": spatial_dim,
        "surface_channels": surf_channels,
        "volume_channels": vol_channels,
        "parameter_channels": params_dim,
    }
    merged_kwargs = {**model_kwargs, **config.architecture} if "architecture" in config else model_kwargs
    print(f"Model kwargs: {merged_kwargs}")
    model = DARM(**merged_kwargs).to(device)
    if getattr(model, "lora_linear_module_count", 0) > 0:
        print(
            f"[DARM] Enabled LoRA on {model.lora_linear_module_count} pretrained linear layers "
            f"(rank={model.lora_rank}, alpha={model.lora_alpha:g}, dropout={model.lora_dropout:g})."
        )

    resume_ckpt = str(getattr(config, "resume_ckpt", "")).strip()
    init_ckpt = str(getattr(config, "init_ckpt", "")).strip()

    print(f"Total parameters: {count_model_params(model)}")
    model_checkpoint_name = get_model_checkpoint_name(config)
    print(f"Checkpoint name: {model_checkpoint_name}")

    if run is not None and bool(getattr(config, "wandb_watch_model", False)):
        run.watch(model, log="all")

    start_epoch = 0
    global_step = 0
    best_rel_l2 = np.inf
    checkpoint_compatible_param_names = set()
    resume_full_state = bool(getattr(config, "resume_full_state", False))
    if not resume_full_state and resume_ckpt:
        checkpoint_compatible_param_names = get_checkpoint_compatible_parameter_names(model, resume_ckpt, device)
        load_partial_state_dict(model, resume_ckpt, device)
    elif not resume_full_state and init_ckpt:
        checkpoint_compatible_param_names = get_checkpoint_compatible_parameter_names(model, init_ckpt, device)
        load_partial_state_dict(model, init_ckpt, device)
    if checkpoint_compatible_param_names:
        print(
            "[DARM] SMART checkpoint loading only restores the shared pretrained backbone/base predictor "
            "tensors. DARM-specific residual/router/reliability/LoRA tensors still start from DARM init."
        )

    freeze_loaded_pretrained_params = bool(getattr(config, "freeze_loaded_pretrained_params", False))
    if not resume_full_state and checkpoint_compatible_param_names and freeze_loaded_pretrained_params:
        frozen_count = freeze_named_parameters(model, checkpoint_compatible_param_names)
        print(
            f"[DARM] Froze {frozen_count} checkpoint-compatible pretrained parameters. "
            "Training DARM residual/editor parameters plus LoRA adapters."
        )
        print(f"[DARM] Trainable parameters after freezing: {count_model_params(model)}")

    scaler = torch.amp.GradScaler("cuda", enabled=amp and torch.cuda.is_available())
    optimizer, scheduler, loss_fn, rel_l2_loss_fn, optimizer_group_info = create_darm_optimizer_scheduler_loss(
        model,
        config,
        train_loader,
        loss_dim=1,
    )
    combined_loss_fn = CombinedLoss(loss_fn, fields) if use_surface_supervision else None
    print(
        f"[DARM] optimizer setup -> trainable_params={optimizer_group_info['trainable_params']}, "
        f"lora_params={optimizer_group_info['lora_params']}, "
        f"non_lora_params={optimizer_group_info['non_lora_params']}, "
        f"lr={optimizer_group_info['lr']:.3e}"
    )

    if resume_full_state:
        if not resume_ckpt:
            raise ValueError("resume_full_state=True requires experiment.resume_ckpt to be set.")
        start_epoch, global_step, best_rel_l2 = load_full_training_state(
            model, optimizer, scheduler, scaler, resume_ckpt, device, amp
        )

    train_model = model
    if multi_gpu_strategy == "data_parallel" and torch.cuda.is_available() and torch.cuda.device_count() > 1:
        device_ids = list(range(torch.cuda.device_count()))
        if int(config.batch_size) >= len(device_ids):
            train_model = DataParallel(model, device_ids=device_ids, output_device=device_ids[0], dim=0)
            print(f"[DARM] Enabled DataParallel on device ids {device_ids}.")
        else:
            print(
                f"[DARM] DataParallel disabled because batch_size={config.batch_size} "
                f"is smaller than num_gpus={len(device_ids)}. Using a single process on cuda:0."
            )

    extra_metric_keys = [
        "loss_supervised",
        "loss_ghost",
        "loss_reliability",
        "loss_ghost_surface",
        "loss_ghost_volume",
        "loss_reliability_surface",
        "loss_reliability_volume",
        "mean_support_mass",
        "mean_route_confidence",
        "mean_route_entropy",
        "mean_evidence_gate",
        "mean_raw_evidence_gate",
        "mean_route_spread",
        "changed_support_mass_surface",
        "changed_support_mass_volume",
        "unchanged_support_mass_surface",
        "unchanged_support_mass_volume",
        "changed_evidence_mass_surface",
        "changed_evidence_mass_volume",
        "unchanged_evidence_mass_surface",
        "unchanged_evidence_mass_volume",
        "changed_response_surface",
        "changed_response_volume",
        "unchanged_response_surface",
        "unchanged_response_volume",
        "changed_residual_surface",
        "changed_residual_volume",
        "unchanged_residual_surface",
        "unchanged_residual_volume",
        "changed_base_response_surface",
        "changed_base_response_volume",
        "unchanged_base_response_surface",
        "unchanged_base_response_volume",
        "changed_edit_displacement_surface",
        "changed_edit_displacement_volume",
    ]
    log_every_n_steps = int(getattr(config, "log_every_n_steps", 0))
    console_log_every_n_steps = int(getattr(config, "console_log_every_n_steps", 0))
    show_first_batch_timing = bool(getattr(config, "show_first_batch_timing", False))
    track_train_channel_metrics = bool(getattr(config, "track_train_channel_metrics", False))
    train_input_check_interval = max(1, int(getattr(config, "train_input_check_interval", 16)))
    train_edit_check_interval = max(1, int(getattr(config, "train_edit_check_interval", 16)))
    train_forward_check_interval = max(1, int(getattr(config, "train_forward_check_interval", 16)))
    train_grad_check_interval = max(1, int(getattr(config, "train_grad_check_interval", 8)))
    eval_nonfinite_check_interval = max(1, int(getattr(config, "eval_nonfinite_check_interval", 8)))

    base_forward_batch_size = int(config.batch_size)
    local_edit_forward_batch_size = 2 * int(config.batch_size)
    print(
        f"[DARM] steps_per_epoch={len(train_loader)}, "
        f"visible_gpus={torch.cuda.device_count() if torch.cuda.is_available() else 0}, "
        f"batch_size={int(config.batch_size)}, "
        f"base_forward_batch_size={base_forward_batch_size}, "
        f"local_edit_forward_batch_size={local_edit_forward_batch_size}, "
        f"num_workers={num_workers}, "
        f"prefetch_factor={prefetch_factor}, "
        f"cuda_batch_prefetch={cuda_batch_prefetch}, "
        f"train_checks=({train_input_check_interval},{train_edit_check_interval},{train_forward_check_interval},{train_grad_check_interval})"
    )
    print(
        "[DARM] Local-edit consistency uses one concatenated forward pass over "
        "full surface/volume query sets while preserving the current "
        "methodology and improving DataParallel utilization."
    )
    if run is not None:
        wandb.log(
            {
                "setup/trainable_params": count_model_params(model),
                "setup/batch_size": int(config.batch_size),
                "setup/local_edit_forward_batch_size": local_edit_forward_batch_size,
                "setup/num_workers": num_workers,
                "setup/prefetch_factor": prefetch_factor,
                "setup/trainable_params_total": optimizer_group_info["trainable_params"],
                "setup/lora_params": optimizer_group_info["lora_params"],
                "setup/non_lora_params": optimizer_group_info["non_lora_params"],
                "setup/lr": optimizer_group_info["lr"],
            },
            step=global_step,
        )
    oom_skip_batches = bool(getattr(config, "oom_skip_batches", True))
    oom_clear_cache = bool(getattr(config, "oom_clear_cache", True))
    total_oom_batches = 0
    nonfinite_skip_batches = bool(getattr(config, "nonfinite_skip_batches", True))
    nonfinite_clear_cache = bool(getattr(config, "nonfinite_clear_cache", True))
    total_nonfinite_batches = 0
    loss_reweight_epochs = {int(epoch_idx) for epoch_idx in getattr(config, "loss_reweight_epochs", [0, 3])}
    loss_reweight_interval_epochs = max(0, int(getattr(config, "loss_reweight_interval_epochs", 0)))
    loss_reweight_batches = max(1, int(getattr(config, "loss_reweight_batches", 8)))
    loss_reweight_reference = str(getattr(config, "loss_reweight_reference", "loss_supervised"))
    loss_reweight_supervised_ratio = max(float(getattr(config, "loss_reweight_supervised_ratio", 4.0)), 1.0e-6)
    loss_reweight_min_scale = float(getattr(config, "loss_reweight_min_scale", 0.25))
    loss_reweight_max_scale = float(getattr(config, "loss_reweight_max_scale", 4.0))
    reliability_weight = float(getattr(config, "reliability_weight", 0.10))
    target_loss_multipliers = {
        "loss_supervised": 1.0,
        "loss_ghost": 1.0 / loss_reweight_supervised_ratio,
        "loss_reliability": 1.0 / loss_reweight_supervised_ratio,
    }
    loss_weight_scales = {
        "loss_supervised": 1.0,
        "loss_ghost": 1.0,
        "loss_reliability": 1.0,
    }
    loss_reweight_done_epochs = set()

    try:
        for ep in tqdm(range(start_epoch, config.epochs), desc="Epochs", dynamic_ncols=True):
            t1 = default_timer()
            set_dataset_epoch(train_data, ep)
            set_dataset_epoch(test_data, 0)

            train_losses = {
                key: torch.zeros((), device=device, dtype=torch.float32)
                for key in (["loss", "rel_l2", "rel_l2_surf", "rel_l2_vol"] + extra_metric_keys)
            }
            for field_name in fields["surface"]:
                train_losses[f"rel_l2_surf_{field_name}"] = torch.zeros((), device=device, dtype=torch.float32)
            for field_name in fields["volume"]:
                train_losses[f"rel_l2_vol_{field_name}"] = torch.zeros((), device=device, dtype=torch.float32)
            train_sample_count = 0
            epoch_oom_batches = 0
            epoch_nonfinite_batches = 0

            train_model.train()
            train_batch_source = CudaPrefetchLoader(train_loader, device) if cuda_batch_prefetch else train_loader
            train_pbar = tqdm(
                train_batch_source,
                desc=f"Train {ep + 1}/{config.epochs}",
                leave=False,
                dynamic_ncols=True,
            )

            ghost_weight = float(getattr(config, "ghost_suppression_weight", 0.25))
            reliability_weight_epoch = reliability_weight
            supervised_weight = float(getattr(config, "supervised_weight", 1.0))
            base_loss_weights = {
                "loss_supervised": supervised_weight,
                "loss_ghost": ghost_weight,
                "loss_reliability": reliability_weight_epoch,
            }
            reweight_accum = {name: 0.0 for name in loss_weight_scales}
            reweight_steps = 0

            for batch_idx, batch in enumerate(train_pbar):
                try:
                    batch_t0 = default_timer()
                    geo_mesh, surf_mesh, surf_data, vol_mesh, vol_data, params, _sample_info = parse_batch(batch, params_dim)
                    geo_mesh = move_to_device(geo_mesh, device)
                    surf_mesh = move_to_device(surf_mesh, device)
                    surf_data = move_to_device(surf_data, device)
                    vol_mesh = move_to_device(vol_mesh, device)
                    vol_data = move_to_device(vol_data, device)
                    if params is not None:
                        params = move_to_device(params, device)

                    if config.dataset == "NACA4":
                        surf_data = surf_data[..., :1]
                        vol_data = torch.cat([vol_data[..., :1], vol_data[..., 2:4]], dim=-1)

                    if should_run_periodic_check(batch_idx, train_input_check_interval):
                        input_problems = find_non_finite_problems(
                            {
                                "geo_mesh": geo_mesh,
                                "surf_mesh": surf_mesh,
                                "surf_data": surf_data,
                                "vol_mesh": vol_mesh,
                                "vol_data": vol_data,
                                "params": params,
                            },
                            max_entries=4,
                        )
                        if input_problems:
                            epoch_nonfinite_batches += 1
                            total_nonfinite_batches += 1
                            print(
                                f"[train/nonfinite] epoch={ep + 1}/{config.epochs} "
                                f"batch={batch_idx + 1}/{len(train_loader)} invalid inputs: "
                                f"{summarize_non_finite_problems(input_problems)}"
                            )
                            if not nonfinite_skip_batches:
                                raise RuntimeError("Encountered non-finite inputs during training.")
                            cleanup_after_invalid_step(optimizer, clear_cache=nonfinite_clear_cache)
                            continue

                    optimizer.zero_grad(set_to_none=True)

                    with torch.no_grad():
                        edit_ramp_power = float(getattr(config, "edit_ramp_power", 1.5))
                        edited_geo, deformation_field = build_local_edit(
                            geo_mesh,
                            min_points=int(getattr(config, "edit_patch_min_points", 256)),
                            max_points=int(getattr(config, "edit_patch_max_points", 1024)),
                            edit_strength=float(getattr(config, "edit_strength", 0.75)),
                            ramp_power=edit_ramp_power,
                            candidate_points=int(getattr(config, "edit_candidate_points", 2048)),
                        )
                        edited_surf_mesh = apply_deformation_field(
                            surf_mesh,
                            deformation_field["centers"],
                            deformation_field["directions"],
                            deformation_field["sigmas"],
                            deformation_field["scales"],
                            ramp_power=edit_ramp_power,
                        )
                        edited_vol_mesh = apply_deformation_field(
                            vol_mesh,
                            deformation_field["centers"],
                            deformation_field["directions"],
                            deformation_field["sigmas"],
                            deformation_field["scales"],
                            ramp_power=edit_ramp_power,
                        )
                        surf_changed_w, surf_unchanged_w, surf_edit_disp_map = build_displacement_targets(
                            surf_mesh,
                            edited_surf_mesh,
                        )
                        vol_changed_w, vol_unchanged_w, vol_edit_disp_map = build_displacement_targets(
                            vol_mesh,
                            edited_vol_mesh,
                        )
                    prep_t1 = default_timer()

                    if should_run_periodic_check(batch_idx, train_edit_check_interval):
                        edit_problems = find_non_finite_problems(
                            {
                                "edited_geo": edited_geo,
                                "edited_surf_mesh": edited_surf_mesh,
                                "edited_vol_mesh": edited_vol_mesh,
                                "surf_changed_w": surf_changed_w,
                                "surf_unchanged_w": surf_unchanged_w,
                                "vol_changed_w": vol_changed_w,
                                "vol_unchanged_w": vol_unchanged_w,
                            },
                            max_entries=4,
                        )
                        if edit_problems:
                            epoch_nonfinite_batches += 1
                            total_nonfinite_batches += 1
                            print(
                                f"[train/nonfinite] epoch={ep + 1}/{config.epochs} "
                                f"batch={batch_idx + 1}/{len(train_loader)} invalid edit tensors: "
                                f"{summarize_non_finite_problems(edit_problems)}"
                            )
                            if not nonfinite_skip_batches:
                                raise RuntimeError("Encountered non-finite edit tensors during training.")
                            cleanup_after_invalid_step(optimizer, clear_cache=nonfinite_clear_cache)
                            continue

                    with torch.autocast(device_type=str(device).split(":")[0], dtype=dtype, enabled=amp):
                        geo_pair = torch.cat([geo_mesh, edited_geo], dim=0)
                        surf_pair = torch.cat([surf_mesh, edited_surf_mesh], dim=0)
                        vol_pair = torch.cat([vol_mesh, edited_vol_mesh], dim=0)
                        params_pair = duplicate_batch(params)

                        y_pair_surf, y_pair_vol, aux_pair = train_model(
                            geo_pair,
                            surf_pair,
                            vol_pair,
                            params_pair,
                            return_aux=True,
                        )

                        pair_batch = int(geo_mesh.shape[0])
                        y_orig_surf, y_edit_surf = y_pair_surf.split(pair_batch, dim=0)
                        y_orig_vol, y_edit_vol = y_pair_vol.split(pair_batch, dim=0)
                        aux_orig, aux_edit = split_aux_batch(aux_pair, pair_batch)

                        y_orig_surf_f = y_orig_surf.float()
                        y_orig_vol_f = y_orig_vol.float()
                        y_edit_surf_f = y_edit_surf.float()
                        y_edit_vol_f = y_edit_vol.float()

                        loss_supervised_main = (
                            combined_loss_fn(y_orig_surf_f, y_orig_vol_f, surf_data, vol_data)
                            if use_surface_supervision
                            else loss_fn(y_orig_vol_f, vol_data)
                        )
                        loss_supervised_main = ensure_scalar_tensor(loss_supervised_main)
                        loss_supervised = ensure_scalar_tensor(loss_supervised_main)

                        surf_response_map = (
                            pointwise_response_norm(y_edit_surf_f, y_orig_surf_f.detach())
                            if use_surface_supervision else surf_mesh.new_zeros(surf_mesh.shape[0], surf_mesh.shape[1], 1)
                        )
                        vol_response_map = pointwise_response_norm(y_edit_vol_f, y_orig_vol_f.detach())
                        surf_base_response_map = (
                            pointwise_response_norm(
                                aux_edit["surface_base"].detach().float(),
                                aux_orig["surface_base"].detach().float(),
                            )
                            if use_surface_supervision else surf_mesh.new_zeros(surf_mesh.shape[0], surf_mesh.shape[1], 1)
                        )
                        vol_base_response_map = pointwise_response_norm(
                            aux_edit["volume_base"].detach().float(),
                            aux_orig["volume_base"].detach().float(),
                        )
                        surf_adapter_response_map = (
                            pointwise_response_norm(
                                aux_edit["surface_residual"].float(),
                                aux_orig["surface_residual"].detach().float(),
                            )
                            if use_surface_supervision else surf_mesh.new_zeros(surf_mesh.shape[0], surf_mesh.shape[1], 1)
                        )
                        vol_adapter_response_map = pointwise_response_norm(
                            aux_edit["volume_residual"].float(),
                            aux_orig["volume_residual"].detach().float(),
                        )
                        surf_residual_norm_map = (
                            pointwise_feature_norm(aux_edit["surface_residual"].float())
                            if use_surface_supervision else surf_mesh.new_zeros(surf_mesh.shape[0], surf_mesh.shape[1], 1)
                        )
                        vol_residual_norm_map = pointwise_feature_norm(aux_edit["volume_residual"].float())

                        changed_support_mass_surface = (
                            masked_mean(aux_edit["surface_support_mass"].float(), surf_changed_w)
                            if use_surface_supervision else y_orig_vol_f.new_zeros(())
                        )
                        changed_support_mass_surface = ensure_scalar_tensor(changed_support_mass_surface)
                        changed_support_mass_volume = masked_mean(aux_edit["volume_support_mass"].float(), vol_changed_w)
                        changed_support_mass_volume = ensure_scalar_tensor(changed_support_mass_volume)
                        unchanged_support_mass_surface = (
                            masked_mean(aux_edit["surface_support_mass"].float(), surf_unchanged_w)
                            if use_surface_supervision else y_orig_vol_f.new_zeros(())
                        )
                        unchanged_support_mass_surface = ensure_scalar_tensor(unchanged_support_mass_surface)
                        unchanged_support_mass_volume = masked_mean(aux_edit["volume_support_mass"].float(), vol_unchanged_w)
                        unchanged_support_mass_volume = ensure_scalar_tensor(unchanged_support_mass_volume)
                        changed_response_surface = (
                            masked_mean(surf_response_map, surf_changed_w)
                            if use_surface_supervision else y_orig_vol_f.new_zeros(())
                        )
                        changed_response_surface = ensure_scalar_tensor(changed_response_surface)
                        changed_response_volume = masked_mean(vol_response_map, vol_changed_w)
                        changed_response_volume = ensure_scalar_tensor(changed_response_volume)
                        unchanged_response_surface = (
                            masked_mean(surf_response_map, surf_unchanged_w)
                            if use_surface_supervision else y_orig_vol_f.new_zeros(())
                        )
                        unchanged_response_surface = ensure_scalar_tensor(unchanged_response_surface)
                        unchanged_response_volume = masked_mean(vol_response_map, vol_unchanged_w)
                        unchanged_response_volume = ensure_scalar_tensor(unchanged_response_volume)
                        changed_residual_surface = (
                            masked_mean(surf_residual_norm_map, surf_changed_w)
                            if use_surface_supervision else y_orig_vol_f.new_zeros(())
                        )
                        changed_residual_surface = ensure_scalar_tensor(changed_residual_surface)
                        changed_residual_volume = ensure_scalar_tensor(masked_mean(vol_residual_norm_map, vol_changed_w))
                        unchanged_residual_surface = (
                            masked_mean(surf_residual_norm_map, surf_unchanged_w)
                            if use_surface_supervision else y_orig_vol_f.new_zeros(())
                        )
                        unchanged_residual_surface = ensure_scalar_tensor(unchanged_residual_surface)
                        unchanged_residual_volume = ensure_scalar_tensor(masked_mean(vol_residual_norm_map, vol_unchanged_w))
                        changed_base_response_surface = (
                            masked_mean(surf_base_response_map, surf_changed_w)
                            if use_surface_supervision else y_orig_vol_f.new_zeros(())
                        )
                        changed_base_response_surface = ensure_scalar_tensor(changed_base_response_surface)
                        changed_base_response_volume = masked_mean(vol_base_response_map, vol_changed_w)
                        changed_base_response_volume = ensure_scalar_tensor(changed_base_response_volume)
                        unchanged_base_response_surface = (
                            masked_mean(surf_base_response_map, surf_unchanged_w)
                            if use_surface_supervision else y_orig_vol_f.new_zeros(())
                        )
                        unchanged_base_response_surface = ensure_scalar_tensor(unchanged_base_response_surface)
                        unchanged_base_response_volume = masked_mean(vol_base_response_map, vol_unchanged_w)
                        unchanged_base_response_volume = ensure_scalar_tensor(unchanged_base_response_volume)
                        ghost_loss = y_orig_vol_f.new_zeros(())
                        ghost_surface = y_orig_vol_f.new_zeros(())
                        ghost_volume = y_orig_vol_f.new_zeros(())
                        if ghost_weight > 0.0:
                            if use_surface_supervision:
                                surface_response_focus = locality_focus_loss(
                                    surf_adapter_response_map,
                                    surf_changed_w,
                                )
                                surface_response_separation = locality_separation_loss(
                                    surf_adapter_response_map,
                                    surf_changed_w,
                                    surf_unchanged_w,
                                )
                                ghost_surface = 0.5 * (surface_response_focus + surface_response_separation)
                                ghost_surface = ensure_scalar_tensor(ghost_surface)
                            volume_response_focus = locality_focus_loss(
                                vol_adapter_response_map,
                                vol_changed_w,
                            )
                            volume_response_separation = locality_separation_loss(
                                vol_adapter_response_map,
                                vol_changed_w,
                                vol_unchanged_w,
                            )
                            ghost_volume = 0.5 * (volume_response_focus + volume_response_separation)
                            ghost_volume = ensure_scalar_tensor(ghost_volume)
                            ghost_terms = [ghost_volume]
                            if use_surface_supervision:
                                ghost_terms.insert(0, ghost_surface)
                            ghost_loss = sum(ghost_terms) / float(len(ghost_terms))
                        ghost_loss = ensure_scalar_tensor(ghost_loss)

                        reliability_loss = y_orig_vol_f.new_zeros(())
                        reliability_surface = y_orig_vol_f.new_zeros(())
                        reliability_volume = y_orig_vol_f.new_zeros(())
                        if reliability_weight_epoch > 0.0:
                            if use_surface_supervision:
                                reliability_surface = 0.5 * (
                                    reliability_map_loss(
                                        aux_edit["surface_support_mass"].float(),
                                        surf_changed_w,
                                        surf_unchanged_w,
                                    )
                                    + reliability_map_loss(
                                        aux_edit["surface_evidence_mass"].float(),
                                        surf_changed_w,
                                        surf_unchanged_w,
                                    )
                                )
                                reliability_surface = ensure_scalar_tensor(reliability_surface)
                            reliability_volume = 0.5 * (
                                reliability_map_loss(
                                    aux_edit["volume_support_mass"].float(),
                                    vol_changed_w,
                                    vol_unchanged_w,
                                )
                                + reliability_map_loss(
                                    aux_edit["volume_evidence_mass"].float(),
                                    vol_changed_w,
                                    vol_unchanged_w,
                                )
                            )
                            reliability_volume = ensure_scalar_tensor(reliability_volume)
                            reliability_terms = [reliability_volume]
                            if use_surface_supervision:
                                reliability_terms.insert(0, reliability_surface)
                            reliability_loss = sum(reliability_terms) / float(len(reliability_terms))
                        reliability_loss = ensure_scalar_tensor(reliability_loss)

                        changed_edit_displacement_surface = (
                            masked_mean(
                                surf_edit_disp_map,
                                surf_changed_w,
                            )
                            if use_surface_supervision else y_orig_vol_f.new_zeros(())
                        )
                        changed_edit_displacement_surface = ensure_scalar_tensor(changed_edit_displacement_surface)
                        changed_edit_displacement_volume = masked_mean(
                            vol_edit_disp_map,
                            vol_changed_w,
                        )
                        changed_edit_displacement_volume = ensure_scalar_tensor(changed_edit_displacement_volume)

                        mean_support_mass_value = aux_query_mean(
                            aux_pair,
                            "surface_support_mass",
                            "volume_support_mass",
                        )
                        mean_route_confidence_value = aux_query_mean(
                            aux_pair,
                            "surface_route_confidence",
                            "volume_route_confidence",
                        )
                        mean_route_entropy_value = aux_query_mean(
                            aux_pair,
                            "surface_route_entropy",
                            "volume_route_entropy",
                        )
                        mean_evidence_gate_value = aux_query_mean(
                            aux_pair,
                            "surface_evidence_mass",
                            "volume_evidence_mass",
                        )
                        mean_raw_evidence_gate_value = aux_query_mean(
                            aux_pair,
                            "surface_raw_evidence_gate",
                            "volume_raw_evidence_gate",
                        )
                        mean_route_spread_value = ensure_scalar_tensor(
                            aux_pair.get("mean_route_spread", y_orig_vol_f.new_zeros(()))
                        )

                        with torch.no_grad():
                            changed_evidence_mass_surface = (
                                masked_mean(aux_edit["surface_evidence_mass"].float(), surf_changed_w)
                                if use_surface_supervision else y_orig_vol_f.new_zeros(())
                            )
                            changed_evidence_mass_surface = ensure_scalar_tensor(changed_evidence_mass_surface)
                            changed_evidence_mass_volume = masked_mean(aux_edit["volume_evidence_mass"].float(), vol_changed_w)
                            changed_evidence_mass_volume = ensure_scalar_tensor(changed_evidence_mass_volume)
                            unchanged_evidence_mass_surface = (
                                masked_mean(aux_edit["surface_evidence_mass"].float(), surf_unchanged_w)
                                if use_surface_supervision else y_orig_vol_f.new_zeros(())
                            )
                            unchanged_evidence_mass_surface = ensure_scalar_tensor(unchanged_evidence_mass_surface)
                            unchanged_evidence_mass_volume = masked_mean(aux_edit["volume_evidence_mass"].float(), vol_unchanged_w)
                            unchanged_evidence_mass_volume = ensure_scalar_tensor(unchanged_evidence_mass_volume)

                        forward_t1 = default_timer()

                    if should_run_periodic_check(batch_idx, train_forward_check_interval):
                        forward_problems = find_non_finite_problems(
                            {
                                "y_orig_surf": y_orig_surf if use_surface_supervision else None,
                                "y_orig_vol": y_orig_vol,
                                "y_edit_surf": y_edit_surf if use_surface_supervision else None,
                                "y_edit_vol": y_edit_vol,
                                "aux_pair": aux_pair,
                                "loss_supervised": loss_supervised,
                                "ghost_loss": ghost_loss,
                                "reliability_loss": reliability_loss,
                            },
                            max_entries=6,
                        )
                        if forward_problems:
                            epoch_nonfinite_batches += 1
                            total_nonfinite_batches += 1
                            print(
                                f"[train/nonfinite] epoch={ep + 1}/{config.epochs} "
                                f"batch={batch_idx + 1}/{len(train_loader)} invalid forward tensors: "
                                f"{summarize_non_finite_problems(forward_problems)}"
                            )
                            if not nonfinite_skip_batches:
                                raise RuntimeError("Encountered non-finite forward outputs during training.")
                            cleanup_after_invalid_step(optimizer, clear_cache=nonfinite_clear_cache)
                            continue

                    loss_terms = {
                        "loss_supervised": loss_supervised,
                        "loss_ghost": ghost_loss,
                        "loss_reliability": reliability_loss,
                    }
                    loss_term_problems = find_non_finite_problems(loss_terms, max_entries=6)
                    if loss_term_problems:
                        epoch_nonfinite_batches += 1
                        total_nonfinite_batches += 1
                        print(
                            f"[train/nonfinite] epoch={ep + 1}/{config.epochs} "
                            f"batch={batch_idx + 1}/{len(train_loader)} invalid loss terms: "
                            f"{summarize_non_finite_problems(loss_term_problems)}"
                        )
                        if not nonfinite_skip_batches:
                            raise RuntimeError("Encountered non-finite loss terms during training.")
                        cleanup_after_invalid_step(optimizer, clear_cache=nonfinite_clear_cache)
                        continue

                    if should_reweight_epoch(ep, loss_reweight_epochs, loss_reweight_interval_epochs) and ep not in loss_reweight_done_epochs:
                        current_loss_values = {
                            name: float(term.detach().float().abs().item())
                            for name, term in loss_terms.items()
                        }
                        for name, value in current_loss_values.items():
                            reweight_accum[name] += value
                        reweight_steps += 1
                        averaged_loss_values = {
                            name: reweight_accum[name] / float(max(reweight_steps, 1))
                            for name in current_loss_values
                        }
                        loss_weight_scales = normalized_loss_scales(
                            averaged_loss_values,
                            base_loss_weights,
                            reference_name=loss_reweight_reference,
                            min_scale=loss_reweight_min_scale,
                            max_scale=loss_reweight_max_scale,
                            target_multipliers=target_loss_multipliers,
                        )
                        if reweight_steps >= loss_reweight_batches or batch_idx == len(train_loader) - 1:
                            loss_reweight_done_epochs.add(ep)
                            print(
                                f"[DARM] epoch={ep + 1}: normalized loss scales over "
                                f"{reweight_steps} calibration batches -> {loss_weight_scales}"
                            )
                            if run is not None:
                                wandb.log(
                                    {f"loss_scale/{name}": scale for name, scale in loss_weight_scales.items()},
                                    step=global_step,
                                )

                    loss = (
                        (supervised_weight * loss_weight_scales["loss_supervised"]) * loss_supervised
                        + (ghost_weight * loss_weight_scales["loss_ghost"]) * ghost_loss
                        + (reliability_weight_epoch * loss_weight_scales["loss_reliability"]) * reliability_loss
                    )
                    loss = ensure_scalar_tensor(loss)

                    if not torch.isfinite(loss):
                        epoch_nonfinite_batches += 1
                        total_nonfinite_batches += 1
                        print(f"[warn] Non-finite loss at epoch {ep} batch {batch_idx}; skipping optimizer step.")
                        cleanup_after_invalid_step(optimizer, clear_cache=nonfinite_clear_cache)
                        continue

                    if amp and torch.cuda.is_available():
                        prev_scale = scaler.get_scale()
                        scaler.scale(loss).backward()
                        scaler.unscale_(optimizer)
                        if should_run_periodic_check(batch_idx, train_grad_check_interval):
                            grad_problems = find_non_finite_gradients(model, max_entries=6)
                            if grad_problems:
                                epoch_nonfinite_batches += 1
                                total_nonfinite_batches += 1
                                print(
                                    f"[train/nonfinite] epoch={ep + 1}/{config.epochs} "
                                    f"batch={batch_idx + 1}/{len(train_loader)} invalid gradients: "
                                    f"{summarize_non_finite_problems(grad_problems)}"
                                )
                                if not nonfinite_skip_batches:
                                    raise RuntimeError("Encountered non-finite gradients during training.")
                                cleanup_after_invalid_step(optimizer, clear_cache=nonfinite_clear_cache)
                                scaler.update()
                                continue
                        if gradient_norm is not None:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_norm)
                        scaler.step(optimizer)
                        scaler.update()
                        if scaler.get_scale() >= prev_scale:
                            scheduler.step()
                    else:
                        loss.backward()
                        if should_run_periodic_check(batch_idx, train_grad_check_interval):
                            grad_problems = find_non_finite_gradients(model, max_entries=6)
                            if grad_problems:
                                epoch_nonfinite_batches += 1
                                total_nonfinite_batches += 1
                                print(
                                    f"[train/nonfinite] epoch={ep + 1}/{config.epochs} "
                                    f"batch={batch_idx + 1}/{len(train_loader)} invalid gradients: "
                                    f"{summarize_non_finite_problems(grad_problems)}"
                                )
                                if not nonfinite_skip_batches:
                                    raise RuntimeError("Encountered non-finite gradients during training.")
                                cleanup_after_invalid_step(optimizer, clear_cache=nonfinite_clear_cache)
                                continue
                        if gradient_norm is not None:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_norm)
                        optimizer.step()
                        scheduler.step()
                    backward_t1 = default_timer()

                    batch_size = surf_data.size(0)
                    batch_size_float = float(batch_size)
                    train_sample_count += batch_size
                    train_losses["loss"] += loss.detach().float() * batch_size_float
                    train_losses["loss_supervised"] += loss_supervised_main.detach().float() * batch_size_float
                    train_losses["loss_ghost"] += ghost_loss.detach().float() * batch_size_float
                    train_losses["loss_reliability"] += reliability_loss.detach().float() * batch_size_float
                    train_losses["loss_ghost_surface"] += ghost_surface.detach().float() * batch_size_float
                    train_losses["loss_ghost_volume"] += ghost_volume.detach().float() * batch_size_float
                    train_losses["loss_reliability_surface"] += reliability_surface.detach().float() * batch_size_float
                    train_losses["loss_reliability_volume"] += reliability_volume.detach().float() * batch_size_float
                    mean_support_mass = mean_support_mass_value.detach().float()
                    mean_route_confidence = mean_route_confidence_value.detach().float()
                    mean_route_entropy = mean_route_entropy_value.detach().float()
                    mean_evidence_gate = mean_evidence_gate_value.detach().float()
                    mean_raw_evidence_gate = mean_raw_evidence_gate_value.detach().float()
                    mean_route_spread = mean_route_spread_value.detach().float()
                    train_losses["mean_support_mass"] += mean_support_mass * batch_size_float
                    train_losses["mean_route_confidence"] += mean_route_confidence * batch_size_float
                    train_losses["mean_route_entropy"] += mean_route_entropy * batch_size_float
                    train_losses["mean_evidence_gate"] += mean_evidence_gate * batch_size_float
                    train_losses["mean_raw_evidence_gate"] += mean_raw_evidence_gate * batch_size_float
                    train_losses["mean_route_spread"] += mean_route_spread * batch_size_float
                    train_losses["changed_support_mass_surface"] += changed_support_mass_surface.detach().float() * batch_size_float
                    train_losses["changed_support_mass_volume"] += changed_support_mass_volume.detach().float() * batch_size_float
                    train_losses["unchanged_support_mass_surface"] += unchanged_support_mass_surface.detach().float() * batch_size_float
                    train_losses["unchanged_support_mass_volume"] += unchanged_support_mass_volume.detach().float() * batch_size_float
                    train_losses["changed_response_surface"] += changed_response_surface.detach().float() * batch_size_float
                    train_losses["changed_response_volume"] += changed_response_volume.detach().float() * batch_size_float
                    train_losses["unchanged_response_surface"] += unchanged_response_surface.detach().float() * batch_size_float
                    train_losses["unchanged_response_volume"] += unchanged_response_volume.detach().float() * batch_size_float
                    train_losses["changed_residual_surface"] += changed_residual_surface.detach().float() * batch_size_float
                    train_losses["changed_residual_volume"] += changed_residual_volume.detach().float() * batch_size_float
                    train_losses["unchanged_residual_surface"] += unchanged_residual_surface.detach().float() * batch_size_float
                    train_losses["unchanged_residual_volume"] += unchanged_residual_volume.detach().float() * batch_size_float
                    train_losses["changed_base_response_surface"] += changed_base_response_surface.detach().float() * batch_size_float
                    train_losses["changed_base_response_volume"] += changed_base_response_volume.detach().float() * batch_size_float
                    train_losses["unchanged_base_response_surface"] += unchanged_base_response_surface.detach().float() * batch_size_float
                    train_losses["unchanged_base_response_volume"] += unchanged_base_response_volume.detach().float() * batch_size_float
                    train_losses["changed_evidence_mass_surface"] += changed_evidence_mass_surface.detach().float() * batch_size_float
                    train_losses["changed_evidence_mass_volume"] += changed_evidence_mass_volume.detach().float() * batch_size_float
                    train_losses["unchanged_evidence_mass_surface"] += unchanged_evidence_mass_surface.detach().float() * batch_size_float
                    train_losses["unchanged_evidence_mass_volume"] += unchanged_evidence_mass_volume.detach().float() * batch_size_float
                    train_losses["changed_edit_displacement_surface"] += changed_edit_displacement_surface.detach().float() * batch_size_float
                    train_losses["changed_edit_displacement_volume"] += changed_edit_displacement_volume.detach().float() * batch_size_float

                    with torch.no_grad():
                        surface_loss = (
                            rel_l2_loss_fn(y_orig_surf_f, surf_data)
                            if use_surface_supervision
                            else torch.tensor(0.0, device=device)
                        )
                        volume_loss = rel_l2_loss_fn(y_orig_vol_f, vol_data)
                        surface_loss = ensure_scalar_tensor(surface_loss)
                        volume_loss = ensure_scalar_tensor(volume_loss)
                        train_losses["rel_l2_surf"] += surface_loss.detach().float() * batch_size_float
                        train_losses["rel_l2_vol"] += volume_loss.detach().float() * batch_size_float
                        train_losses["rel_l2"] += (surface_loss + volume_loss).detach().float() * batch_size_float
                        if track_train_channel_metrics:
                            if use_surface_supervision:
                                pred_surf_train = y_orig_surf_f * std_surf + mean_surf
                                gt_surf_train = surf_data * std_surf + mean_surf
                                for channel_idx, field_name in enumerate(fields["surface"]):
                                    channel_loss = rel_l2_loss_fn(
                                        pred_surf_train[..., channel_idx:channel_idx + 1],
                                        gt_surf_train[..., channel_idx:channel_idx + 1],
                                    )
                                    channel_loss = ensure_scalar_tensor(channel_loss)
                                    train_losses[f"rel_l2_surf_{field_name}"] += (
                                        channel_loss.detach().float() * batch_size_float
                                    )
                            pred_vol_train = y_orig_vol_f * std_vol + mean_vol
                            gt_vol_train = vol_data * std_vol + mean_vol
                            for channel_idx, field_name in enumerate(fields["volume"]):
                                channel_loss = rel_l2_loss_fn(
                                    pred_vol_train[..., channel_idx:channel_idx + 1],
                                    gt_vol_train[..., channel_idx:channel_idx + 1],
                                )
                                channel_loss = ensure_scalar_tensor(channel_loss)
                                train_losses[f"rel_l2_vol_{field_name}"] += (
                                    channel_loss.detach().float() * batch_size_float
                                )

                    global_step += 1
                    if run is not None and log_every_n_steps > 0 and (
                        (batch_idx + 1) % log_every_n_steps == 0 or batch_idx == len(train_loader) - 1
                    ):
                        wandb.log(
                            {
                                "train/batch_loss": float(loss.item()),
                                "train/batch_supervised": float(loss_supervised_main.item()),
                                "train/batch_ghost": float(ghost_loss.item()),
                                "train/batch_reliability": float(reliability_loss.item()),
                                "train/batch_ghost_surface": float(ghost_surface.item()),
                                "train/batch_ghost_volume": float(ghost_volume.item()),
                                "train/batch_reliability_surface": float(reliability_surface.item()),
                                "train/batch_reliability_volume": float(reliability_volume.item()),
                                "train/batch_mean_support_mass": float(mean_support_mass.item()),
                                "train/batch_mean_route_confidence": float(mean_route_confidence.item()),
                                "train/batch_mean_route_entropy": float(mean_route_entropy.item()),
                                "train/batch_mean_evidence_gate": float(mean_evidence_gate.item()),
                                "train/batch_mean_raw_evidence_gate": float(mean_raw_evidence_gate.item()),
                                "train/batch_mean_route_spread": float(mean_route_spread.item()),
                                "train/batch_changed_support_mass_surface": float(changed_support_mass_surface.item()),
                                "train/batch_changed_support_mass_volume": float(changed_support_mass_volume.item()),
                                "train/batch_unchanged_support_mass_surface": float(unchanged_support_mass_surface.item()),
                                "train/batch_unchanged_support_mass_volume": float(unchanged_support_mass_volume.item()),
                                "train/batch_changed_response_surface": float(changed_response_surface.item()),
                                "train/batch_changed_response_volume": float(changed_response_volume.item()),
                                "train/batch_unchanged_response_surface": float(unchanged_response_surface.item()),
                                "train/batch_unchanged_response_volume": float(unchanged_response_volume.item()),
                                "train/batch_changed_residual_surface": float(changed_residual_surface.item()),
                                "train/batch_changed_residual_volume": float(changed_residual_volume.item()),
                                "train/batch_unchanged_residual_surface": float(unchanged_residual_surface.item()),
                                "train/batch_unchanged_residual_volume": float(unchanged_residual_volume.item()),
                                "train/batch_changed_base_response_surface": float(changed_base_response_surface.item()),
                                "train/batch_changed_base_response_volume": float(changed_base_response_volume.item()),
                                "train/batch_unchanged_base_response_surface": float(unchanged_base_response_surface.item()),
                                "train/batch_unchanged_base_response_volume": float(unchanged_base_response_volume.item()),
                                "train/batch_changed_evidence_mass_surface": float(changed_evidence_mass_surface.item()),
                                "train/batch_changed_evidence_mass_volume": float(changed_evidence_mass_volume.item()),
                                "train/batch_unchanged_evidence_mass_surface": float(unchanged_evidence_mass_surface.item()),
                                "train/batch_unchanged_evidence_mass_volume": float(unchanged_evidence_mass_volume.item()),
                                "train/batch_changed_edit_displacement_surface": float(changed_edit_displacement_surface.item()),
                                "train/batch_changed_edit_displacement_volume": float(changed_edit_displacement_volume.item()),
                                "train/ghost_weight": ghost_weight,
                                "train/reliability_weight": reliability_weight_epoch,
                                "train/effective_supervised_weight": supervised_weight * loss_weight_scales["loss_supervised"],
                                "train/effective_ghost_weight": ghost_weight * loss_weight_scales["loss_ghost"],
                                "train/effective_reliability_weight": reliability_weight_epoch * loss_weight_scales["loss_reliability"],
                                "lr": scheduler.get_last_lr()[0],
                                "epoch": ep,
                            },
                            step=global_step,
                        )

                    log_t1 = default_timer()
                    if show_first_batch_timing and batch_idx == 0:
                        print(
                            f"[DARM] first_batch_timing: prep={prep_t1 - batch_t0:.2f}s, "
                            f"forward+loss={forward_t1 - prep_t1:.2f}s, "
                            f"backward+step={backward_t1 - forward_t1:.2f}s, "
                            f"logging={log_t1 - backward_t1:.2f}s, "
                            f"total={log_t1 - batch_t0:.2f}s"
                        )
                    if console_log_every_n_steps > 0 and (
                        (batch_idx + 1) % console_log_every_n_steps == 0 or batch_idx == len(train_loader) - 1
                    ):
                        print(
                            f"[train] epoch={ep + 1}/{config.epochs} "
                            f"step={batch_idx + 1}/{len(train_loader)} "
                            f"loss={loss.item():.4f} step_time={log_t1 - batch_t0:.2f}s"
                        )
                except torch.OutOfMemoryError as exc:
                    epoch_oom_batches += 1
                    total_oom_batches += 1
                    optimizer.zero_grad(set_to_none=True)
                    if oom_clear_cache:
                        cleanup_after_oom(optimizer)
                    free_gib = torch.cuda.mem_get_info(device=device)[0] / (1024 ** 3) if torch.cuda.is_available() else float("nan")
                    print(
                        f"[OOM] epoch={ep + 1}/{config.epochs} batch={batch_idx + 1}/{len(train_loader)} "
                        f"effective_forward_batch_size={local_edit_forward_batch_size} "
                        f"free_cuda_gib={free_gib:.2f} "
                        f"message={exc}"
                    )
                    if not oom_skip_batches:
                        raise
                    print(
                        "[OOM] Cleared gradients/cache and skipped this batch. "
                        "If this repeats, lower experiment.batch_size or architecture.subsampled_geometry_points."
                    )
                    continue

            denom = max(int(train_sample_count), 1)
            train_losses_log = {key: float((value / float(denom)).detach().cpu().item()) for key, value in train_losses.items()}

            test_losses = evaluate_loader(
                train_model,
                test_loader,
                config,
                device,
                dtype,
                amp,
                params_dim,
                mean_surf,
                std_surf,
                mean_vol,
                std_vol,
                fields,
                rel_l2_loss_fn,
                combined_loss_fn,
                loss_fn,
                use_surface_supervision,
                cuda_batch_prefetch,
                eval_nonfinite_check_interval,
            )

            if test_losses["rel_l2"] < best_rel_l2:
                best_rel_l2 = test_losses["rel_l2"]
                torch.save(
                    build_checkpoint(
                        ep,
                        model,
                        optimizer,
                        scheduler,
                        scaler,
                        test_losses["loss"],
                        test_losses["rel_l2"],
                        fields,
                        global_step,
                        best_rel_l2,
                        {k: v for k, v in test_losses.items() if k.startswith("rel_l2")},
                    ),
                    "checkpoints/" + model_checkpoint_name + "_best.pt",
                )

            torch.save(
                build_checkpoint(
                    ep,
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    test_losses["loss"],
                    test_losses["rel_l2"],
                    fields,
                    global_step,
                    best_rel_l2,
                    {k: v for k, v in test_losses.items() if k.startswith("rel_l2")},
                ),
                "checkpoints/" + model_checkpoint_name + "_last.pt",
            )

            t2 = default_timer()
            print(
                f"epoch: {ep}, t2-t1 (epoch time): {t2 - t1:.5f}, "
                f"train total loss: {train_losses_log['loss']:.5f}, "
                f"train supervised loss: {train_losses_log['loss_supervised']:.5f}, "
                f"test supervised loss: {test_losses['loss']:.5f}, "
                f"train rel_l2: {train_losses_log['rel_l2']:.5f}, "
                f"test rel_l2: {test_losses['rel_l2']:.5f}, "
                f"oom_batches: {epoch_oom_batches}, nonfinite_batches: {epoch_nonfinite_batches}"
            )

            if run is not None:
                wandb_dict = {
                    "lr": scheduler.get_last_lr()[0],
                    "train/total_loss": train_losses_log["loss"],
                    "train/supervised_loss": train_losses_log["loss_supervised"],
                    "test/supervised_loss": test_losses["loss"],
                    "train/ghost_weight": ghost_weight,
                    "train/reliability_weight": reliability_weight_epoch,
                    "train/effective_supervised_weight": supervised_weight * loss_weight_scales["loss_supervised"],
                    "train/effective_ghost_weight": ghost_weight * loss_weight_scales["loss_ghost"],
                    "train/effective_reliability_weight": reliability_weight_epoch * loss_weight_scales["loss_reliability"],
                    "train/oom_batches_epoch": epoch_oom_batches,
                    "train/oom_batches_total": total_oom_batches,
                    "train/nonfinite_batches_epoch": epoch_nonfinite_batches,
                    "train/nonfinite_batches_total": total_nonfinite_batches,
                }
                train_log_values = train_losses_log if track_train_channel_metrics else {
                    key: value
                    for key, value in train_losses_log.items()
                    if key in {
                        "loss",
                        "rel_l2",
                        "rel_l2_surf",
                        "rel_l2_vol",
                        "loss_supervised",
                        "loss_ghost",
                        "loss_reliability",
                        "loss_ghost_surface",
                        "loss_ghost_volume",
                        "loss_reliability_surface",
                        "loss_reliability_volume",
                        "mean_support_mass",
                        "mean_route_confidence",
                        "mean_route_entropy",
                        "mean_evidence_gate",
                        "mean_raw_evidence_gate",
                        "mean_route_spread",
                        "changed_support_mass_surface",
                        "changed_support_mass_volume",
                        "unchanged_support_mass_surface",
                        "unchanged_support_mass_volume",
                        "changed_response_surface",
                        "changed_response_volume",
                        "unchanged_response_surface",
                        "unchanged_response_volume",
                        "changed_residual_surface",
                        "changed_residual_volume",
                        "unchanged_residual_surface",
                        "unchanged_residual_volume",
                        "changed_base_response_surface",
                        "changed_base_response_volume",
                        "unchanged_base_response_surface",
                        "unchanged_base_response_volume",
                        "changed_evidence_mass_surface",
                        "changed_evidence_mass_volume",
                        "unchanged_evidence_mass_surface",
                        "unchanged_evidence_mass_volume",
                        "changed_edit_displacement_surface",
                        "changed_edit_displacement_volume",
                    }
                }
                wandb_dict.update({f"train/{key}": value for key, value in train_log_values.items()})
                wandb_dict.update({f"test/{key}": value for key, value in test_losses.items()})
                if track_train_channel_metrics:
                    add_all_field_metrics(wandb_dict, "train", fields["surface"], fields["volume"], metric_values=train_losses_log)
                    add_canonical_field_metrics(wandb_dict, "train", fields["surface"], fields["volume"], metric_values=train_losses_log)
                add_all_field_metrics(wandb_dict, "test", fields["surface"], fields["volume"], metric_values=test_losses)
                add_canonical_field_metrics(wandb_dict, "test", fields["surface"], fields["volume"], metric_values=test_losses)
                wandb_dict["meta/training_surface_signals"] = ",".join(fields["surface"])
                wandb_dict["meta/training_volume_signals"] = ",".join(fields["volume"])
                wandb.log(wandb_dict, step=global_step)
    finally:
        if run is not None:
            best_ckpt = os.path.join("checkpoints", model_checkpoint_name + "_best.pt")
            last_ckpt = os.path.join("checkpoints", model_checkpoint_name + "_last.pt")
            if os.path.isfile(best_ckpt) or os.path.isfile(last_ckpt):
                artifact = wandb.Artifact("model", type="model")
                if os.path.isfile(best_ckpt):
                    artifact.add_file(best_ckpt)
                if os.path.isfile(last_ckpt):
                    artifact.add_file(last_ckpt)
                run.log_artifact(artifact)
            run.finish()


if __name__ == "__main__":
    main()
    print("Training done.")
