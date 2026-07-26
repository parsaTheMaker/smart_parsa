from __future__ import annotations

import math
import os
from contextlib import contextmanager
from timeit import default_timer

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.nn as nn
import wandb
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.parallel import DataParallel
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm

from data.datasets import get_dataset
from loss.losses import CombinedLoss
from utils.utils import (
    apply_naca4_auto_point_budget,
    count_model_params,
    get_model_checkpoint_name,
    get_optimizer_scheduler_loss,
    initialize_gpu,
    initialize_wandb,
    make_grad_scaler,
    print_point_budget,
    reset_scheduler_for_extension,
)

CANON_SURF_FIELDS = ["pressure", "normal_x", "normal_y"]
CANON_VOL_FIELDS = ["pressure", "sdf", "velocity_x", "velocity_y"]


def is_dist_enabled():
    return dist.is_available() and dist.is_initialized()


def get_rank():
    return dist.get_rank() if is_dist_enabled() else 0


def get_world_size():
    return dist.get_world_size() if is_dist_enabled() else 1


def is_main_process():
    return get_rank() == 0


def unwrap_model(model):
    return model.module if isinstance(model, (DDP, DataParallel)) else model


def gradient_diagnostics(model):
    """Return finite gradient and parameter norms for training observability."""
    grad_sq = None
    param_sq = None
    max_grad = None
    finite = True
    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue
        parameter_sq = parameter.detach().float().pow(2).sum()
        param_sq = parameter_sq if param_sq is None else param_sq + parameter_sq
        if parameter.grad is None:
            continue
        gradient = parameter.grad.detach().float()
        if not torch.isfinite(gradient).all():
            finite = False
        gradient_sq = gradient.pow(2).sum()
        grad_sq = gradient_sq if grad_sq is None else grad_sq + gradient_sq
        gradient_max = gradient.abs().max()
        max_grad = gradient_max if max_grad is None else torch.maximum(max_grad, gradient_max)
    reference = next(model.parameters()).detach()
    zero = reference.new_zeros((), dtype=torch.float32)
    return {
        "grad_norm": torch.sqrt(grad_sq.clamp_min(0.0)) if grad_sq is not None else zero,
        "parameter_norm": torch.sqrt(param_sq.clamp_min(0.0)) if param_sq is not None else zero,
        "max_grad": max_grad if max_grad is not None else zero,
        "finite": finite,
    }


def snapshot_model_buffers(model):
    return {name: buffer.detach().clone() for name, buffer in unwrap_model(model).named_buffers()}


def restore_model_buffers(model, snapshot):
    if snapshot is None:
        return
    for name, value in unwrap_model(model).named_buffers():
        if name in snapshot:
            value.copy_(snapshot[name])


def model_buffers_are_finite(model):
    return all(bool(torch.isfinite(buffer).all().item()) for buffer in unwrap_model(model).buffers())


def _last_linear_params(module):
    last_linear = None
    for submodule in module.modules():
        if isinstance(submodule, nn.Linear):
            last_linear = submodule
    if last_linear is None:
        return []
    params = [last_linear.weight]
    if last_linear.bias is not None:
        params.append(last_linear.bias)
    return params


def resolve_gradnorm_reference_params(model):
    base_model = unwrap_model(model)
    for attr_name in ("mlp", "output_head", "head", "surface_decoder", "volume_decoder"):
        if hasattr(base_model, attr_name):
            params = _last_linear_params(getattr(base_model, attr_name))
            if params:
                # Use the last linear weight as the GradNorm reference layer.
                return (params[0],)

    trainable_params = [param for param in base_model.parameters() if param.requires_grad]
    if not trainable_params:
        raise ValueError("Could not resolve GradNorm reference parameters for the model.")
    return (trainable_params[-1],)


def resolve_balance_reference_parameter(model):
    """Return a shared backbone parameter used by layer-local balancers.

    The prediction head is intentionally avoided here. It is not shared by
    the task branches in many operator models and its gradients can make a
    task balancer overreact directly at the output.
    """
    base_model = unwrap_model(model)
    for attr_name in ("encoder_blocks", "geometry_encoder_blocks", "geometry_encoder"):
        blocks = getattr(base_model, attr_name, None)
        if blocks is None:
            continue
        for block in reversed(list(blocks)):
            params = _last_linear_params(block)
            if params:
                return params[0]

    reference_params = resolve_gradnorm_reference_params(model)
    if len(reference_params) != 1:
        raise ValueError("Layer-local gradient balancing requires exactly one reference parameter.")
    return reference_params[0]


def config_layer_gradient(task_losses, reference_parameter, return_diagnostics=False):
    """Compute the official ConFIG direction for one reference parameter.

    The surrounding model gradient is still produced by the ordinary weighted
    objective. Only this selected layer is replaced by the conflict-free
    direction, which keeps the experiment layer-local rather than full-network.
    """
    from conflictfree.grad_operator import ConFIG_update

    gradients = []
    for loss in task_losses:
        gradient = torch.autograd.grad(
            loss,
            reference_parameter,
            retain_graph=True,
            create_graph=False,
            allow_unused=True,
        )[0]
        if gradient is None:
            gradient = torch.zeros_like(reference_parameter)
        gradients.append(gradient.float().reshape(-1))

    gradient_matrix = torch.stack(gradients, dim=0)
    finite = bool(torch.isfinite(gradient_matrix).all().item())
    nonzero = bool((gradient_matrix.norm(dim=1) > 1.0e-12).all().item())
    ordinary_direction = gradient_matrix.sum(dim=0)
    ordinary_norm = ordinary_direction.norm()
    used_fallback = not (finite and nonzero)
    if finite and nonzero:
        direction = ConFIG_update(gradient_matrix, use_least_square=True)
        if not bool(torch.isfinite(direction).all().item()):
            direction = gradient_matrix.sum(dim=0)
            used_fallback = True
    else:
        direction = ordinary_direction

    # ConFIG's projection-length rule can produce a very different magnitude
    # from the configured weighted sum. Keep its conflict-free direction, but
    # preserve the optimizer step scale of the fixed-sum baseline for this
    # layer.
    direction_norm = direction.norm()
    if bool(torch.isfinite(direction_norm).item()) and float(direction_norm.item()) > 1.0e-12:
        if bool(torch.isfinite(ordinary_norm).item()) and float(ordinary_norm.item()) > 1.0e-12:
            direction = direction * (ordinary_norm / direction_norm)
    else:
        direction = ordinary_direction
        used_fallback = True

    direction = direction.to(dtype=reference_parameter.dtype).view_as(reference_parameter)
    if not return_diagnostics:
        return direction
    final_norm = direction.float().norm()
    cosine = torch.zeros((), device=direction.device, dtype=torch.float32)
    if float(final_norm.item()) > 1.0e-12 and float(ordinary_norm.item()) > 1.0e-12:
        cosine = F.cosine_similarity(direction.float().reshape(1, -1), ordinary_direction.reshape(1, -1), dim=1)[0]
    scale = torch.ones((), device=direction.device, dtype=torch.float32)
    if float(direction_norm.item()) > 1.0e-12:
        scale = (ordinary_norm / direction_norm).float()
    diagnostics = {
        "reference_grad_norm": ordinary_norm.detach().float(),
        "direction_norm": final_norm.detach().float(),
        "direction_scale": scale.detach().float(),
        "direction_cosine": cosine.detach().float(),
        "used_fallback": torch.tensor(float(used_fallback), device=direction.device),
    }
    return direction, diagnostics


def _config_full_update(gradient_matrix):
    """Apply the standard non-momentum ConFIG operator to full gradients.

    For m loss gradients and P parameters, the paper's least-squares solve can
    be reduced from an m-by-P solve to the m-by-m Gram system
    (U U^T) a = 1, followed by U^T a. This is the same minimum-norm
    least-squares solution while keeping the expensive linear algebra in loss
    space rather than parameter space.
    """
    with torch.no_grad():
        had_nonfinite = not bool(torch.isfinite(gradient_matrix).all().item())
        if had_nonfinite:
            # Keep AMP enabled and skip only invalid gradient coordinates.
            # This is intentionally a loose fallback: finite task-gradient
            # directions still contribute, while one overflowed coordinate
            # cannot terminate a long run.
            torch.nan_to_num(gradient_matrix, nan=0.0, posinf=0.0, neginf=0.0, out=gradient_matrix)

        gradient_norms = gradient_matrix.float().norm(dim=1)
        unit_gradients = gradient_matrix.float()
        unit_gradients.div_(gradient_norms.unsqueeze(1).clamp_min(1.0e-12))
        zero_rows = gradient_norms <= 1.0e-12
        unit_gradients.masked_fill_(zero_rows.unsqueeze(1), 0.0)

        gram = unit_gradients @ unit_gradients.transpose(0, 1)
        equal_weights = torch.ones(gradient_matrix.shape[0], device=gradient_matrix.device, dtype=gradient_matrix.dtype)
        coefficients = torch.linalg.lstsq(gram, equal_weights).solution
        best_direction = unit_gradients.transpose(0, 1) @ coefficients
        best_norm = best_direction.norm()
        unit_direction = best_direction / best_norm.clamp_min(1.0e-12)
        unit_direction = torch.nan_to_num(unit_direction, nan=0.0, posinf=0.0, neginf=0.0)

        # ProjectionLength from the reference implementation:
        # |g_c| = sum_i <g_i, g_c / |g_c|>.
        projection_lengths = (unit_gradients @ unit_direction) * gradient_norms
        config_gradient = unit_direction * projection_lengths.sum()
        min_cosine = (unit_gradients @ unit_direction).min()

        diagnostics = {
            "mean_grad_norm": gradient_norms.mean().detach().float(),
            "direction_norm": config_gradient.norm().detach().float(),
            "min_cosine": min_cosine.detach().float(),
            "used_fallback": torch.tensor(float(had_nonfinite), device=gradient_matrix.device),
            "nonfinite_gradients": torch.tensor(float(had_nonfinite), device=gradient_matrix.device),
        }
        return config_gradient, diagnostics


def config_full_backward(
    task_losses,
    parameters,
    scaler=None,
    amp_enabled=False,
    vectorized=True,
    allow_sequential_fallback=True,
):
    """Compute full-network ConFIG gradients and assign them to parameters.

    Each task is differentiated separately through one retained forward graph,
    matching full ConFIG rather than the paper's alternating M-ConFIG variant.
    Only flattened detached gradient rows are retained; parameter gradients are
    assigned from the final ConFIG vector in place.
    """
    parameters = [parameter for parameter in parameters if parameter.requires_grad]
    if not parameters:
        raise ValueError("ConFIG_full requires at least one trainable parameter.")

    task_count = len(task_losses)
    stacked_losses = torch.stack(list(task_losses))
    amp_scale = 1.0
    if amp_enabled and scaler is not None and scaler.is_enabled():
        stacked_losses = scaler.scale(stacked_losses)
        amp_scale = float(scaler.get_scale())

    gradient_matrix = None
    if vectorized:
        try:
            gradients = torch.autograd.grad(
                stacked_losses,
                parameters,
                grad_outputs=torch.eye(
                    task_count,
                    device=stacked_losses.device,
                    dtype=stacked_losses.dtype,
                ),
                retain_graph=False,
                create_graph=False,
                allow_unused=True,
                is_grads_batched=True,
            )
            gradient_parts = []
            for parameter, gradient in zip(parameters, gradients):
                if gradient is None:
                    gradient_parts.append(
                        torch.zeros((task_count, parameter.numel()), device=parameter.device, dtype=torch.float32)
                    )
                else:
                    gradient_parts.append(gradient.detach().float().reshape(task_count, -1))
            gradient_matrix = torch.cat(gradient_parts, dim=1)
            del gradients, gradient_parts
        except RuntimeError as exc:
            if not allow_sequential_fallback:
                raise RuntimeError(
                    "Vectorized ConFIG_full gradients failed. Set "
                    "experiment.config_full_vectorized_gradients=False to use the sequential path."
                ) from exc
            print(f"[ConFIG_full] Vectorized gradients unavailable; using sequential fallback: {exc}")

    if gradient_matrix is None:
        gradient_rows = []
        for task_index, task_loss in enumerate(task_losses):
            scaled_loss = task_loss
            if amp_enabled and scaler is not None and scaler.is_enabled():
                scaled_loss = scaler.scale(task_loss)
            gradients = torch.autograd.grad(
                scaled_loss,
                parameters,
                retain_graph=task_index < len(task_losses) - 1,
                create_graph=False,
                allow_unused=True,
            )
            row_parts = []
            for parameter, gradient in zip(parameters, gradients):
                if gradient is None:
                    row_parts.append(torch.zeros(parameter.numel(), device=parameter.device, dtype=torch.float32))
                else:
                    row_parts.append(gradient.detach().float().reshape(-1))
            gradient_rows.append(torch.cat(row_parts, dim=0))
            del gradients, row_parts, scaled_loss
        gradient_matrix = torch.stack(gradient_rows, dim=0)
        del gradient_rows

    del stacked_losses
    config_gradient, diagnostics = _config_full_update(gradient_matrix)
    if amp_scale != 1.0:
        # The assigned parameter gradients stay AMP-scaled for the normal
        # scaler.unscale_/step path; expose raw norms in W&B diagnostics.
        diagnostics["mean_grad_norm"] = diagnostics["mean_grad_norm"] / amp_scale
        diagnostics["direction_norm"] = diagnostics["direction_norm"] / amp_scale
    diagnostics["amp_scale"] = torch.tensor(amp_scale, device=parameters[0].device, dtype=torch.float32)

    offset = 0
    for parameter in parameters:
        parameter_size = parameter.numel()
        parameter.grad = config_gradient[offset:offset + parameter_size].view_as(parameter).to(dtype=parameter.dtype)
        offset += parameter_size

    return diagnostics


def external_gradnorm_backward(
    external_gradnorm,
    losses,
    reference_parameter,
    weighted_total_loss,
    min_loss_weights=None,
    scaler=None,
    amp_enabled=False,
):
    """Apply GradNorm weight updates and the weighted model backward separately.

    The upstream helper backpropagates its auxiliary GradNorm loss through the
    model as well as through its temporary loss-weight parameters. That adds a
    second-order model update to the task objective. We retain its state and
    update rule, but restrict the GradNorm derivative to the temporary weights;
    the model receives only the configured weighted task gradient.
    """
    losses = torch.stack(list(losses))
    if external_gradnorm.initted.device != losses.device:
        external_gradnorm.to(losses.device)

    step = external_gradnorm.step.item()
    external_gradnorm.step.add_(int(external_gradnorm.training))
    weighted_total_loss = weighted_total_loss.float()

    should_update = (
        external_gradnorm.training
        and not external_gradnorm.frozen
        and step >= external_gradnorm.update_after_step
        and (step % external_gradnorm.update_every) == 0
        and bool(external_gradnorm.loss_mask.any().item())
    )
    if not should_update:
        model_loss = weighted_total_loss
        if amp_enabled and scaler is not None and scaler.is_enabled():
            model_loss = scaler.scale(model_loss)
        model_loss.backward()
        return torch.zeros((), device=losses.device, dtype=torch.float32)

    loss_mask = external_gradnorm.loss_mask.to(device=losses.device, dtype=torch.bool)
    if external_gradnorm.has_restoring_force:
        if not bool(external_gradnorm.initted.item()):
            initial_losses = losses.detach().clone()
            if is_dist_enabled():
                dist.all_reduce(initial_losses)
                initial_losses.div_(get_world_size())
            external_gradnorm.initial_losses.copy_(initial_losses)
            external_gradnorm.initted.fill_(True)
        elif external_gradnorm.initial_losses_decay < 1.0:
            meaned_losses = losses.detach().clone()
            if is_dist_enabled():
                dist.all_reduce(meaned_losses)
                meaned_losses.div_(get_world_size())
            external_gradnorm.initial_losses.lerp_(meaned_losses, 1.0 - external_gradnorm.initial_losses_decay)

    selected_losses = losses[loss_mask]
    selected_weights = nn.Parameter(external_gradnorm.loss_weights[loss_mask].detach().clone())
    grad_norms = []
    for weight, loss in zip(selected_weights, selected_losses):
        gradient = torch.autograd.grad(
            weight * loss,
            reference_parameter,
            create_graph=True,
            retain_graph=True,
            allow_unused=True,
        )[0]
        if gradient is None:
            gradient = torch.zeros_like(reference_parameter)
        grad_norms.append(gradient.float().norm(p=2))
    grad_norms = torch.stack(grad_norms)
    grad_norm_average = grad_norms.mean()
    if is_dist_enabled():
        dist.all_reduce(grad_norm_average)
        grad_norm_average.div_(get_world_size())

    if external_gradnorm.has_restoring_force:
        loss_ratio = selected_losses.detach() / external_gradnorm.initial_losses[loss_mask].clamp_min(1.0e-8)
        relative_training_rate = F.normalize(loss_ratio, p=1, dim=0) * selected_losses.numel()
        gradient_target = (grad_norm_average * relative_training_rate.pow(-external_gradnorm.alpha)).detach()
    else:
        gradient_target = grad_norm_average.expand_as(grad_norms).detach()

    gradnorm_loss = F.l1_loss(grad_norms, gradient_target)
    weight_gradient = torch.autograd.grad(
        gradnorm_loss,
        selected_weights,
        retain_graph=True,
        allow_unused=True,
    )[0]
    if weight_gradient is None:
        weight_gradient = torch.zeros_like(selected_weights)

    # The weighted loss keeps a detached view of the current weight buffer.
    # Backpropagate it before updating that buffer in place.
    model_loss = weighted_total_loss
    if amp_enabled and scaler is not None and scaler.is_enabled():
        model_loss = scaler.scale(model_loss)
    model_loss.backward()

    with torch.no_grad():
        updated_weights = selected_weights.detach() - weight_gradient.detach() * float(external_gradnorm.learning_rate)
        total_weight = external_gradnorm.init_loss_weights_for_sum[loss_mask].sum()
        if min_loss_weights is None:
            renormalized_weights = F.normalize(updated_weights, p=1, dim=0) * total_weight
        else:
            floors = torch.as_tensor(
                min_loss_weights,
                device=updated_weights.device,
                dtype=updated_weights.dtype,
            )[loss_mask]
            floors = floors.clamp_min(0.0)
            if bool((floors.sum() >= total_weight).item()):
                raise ValueError("external_gradnorm_min_weights must sum to less than the GradNorm weight total.")
            remaining_weight = total_weight - floors.sum()
            free_weights = (updated_weights.clamp_min(0.0) - floors).clamp_min(0.0)
            free_sum = free_weights.sum()
            if bool((free_sum > 1.0e-12).item()):
                renormalized_weights = floors + remaining_weight * free_weights / free_sum
            else:
                renormalized_weights = floors + remaining_weight / float(floors.numel())
        external_gradnorm.loss_weights[loss_mask] = renormalized_weights
        external_gradnorm.loss_weights_grad[loss_mask] = 0.0

    return gradnorm_loss.detach().float()


def setup_distributed():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return {
            "enabled": False,
            "rank": 0,
            "local_rank": 0,
            "world_size": 1,
        }

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    dist.init_process_group(backend=backend)
    return {
        "enabled": True,
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
    }


def cleanup_distributed():
    if is_dist_enabled():
        dist.destroy_process_group()


def distributed_average_scalars(values):
    if not is_dist_enabled():
        return values
    tensor = torch.tensor(values, device="cuda" if torch.cuda.is_available() else "cpu", dtype=torch.float64)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= float(get_world_size())
    return tensor.cpu().tolist()


def init_metric_dict(surface_fields, volume_fields, extra_keys=None):
    metrics = {
        "loss": 0.0,
        "rel_l2": 0.0,
        "rel_l2_surf": 0.0,
        "rel_l2_vol": 0.0,
    }
    for field_name in surface_fields:
        metrics[f"rel_l2_surf_{field_name}"] = 0.0
    for field_name in volume_fields:
        metrics[f"rel_l2_vol_{field_name}"] = 0.0
    for key in extra_keys or []:
        metrics[key] = 0.0
    return metrics


def init_metric_tensor_dict(surface_fields, volume_fields, device, extra_keys=None):
    metrics = {
        "loss": torch.zeros((), device=device, dtype=torch.float32),
        "rel_l2": torch.zeros((), device=device, dtype=torch.float32),
        "rel_l2_surf": torch.zeros((), device=device, dtype=torch.float32),
        "rel_l2_vol": torch.zeros((), device=device, dtype=torch.float32),
    }
    for field_name in surface_fields:
        metrics[f"rel_l2_surf_{field_name}"] = torch.zeros((), device=device, dtype=torch.float32)
    for field_name in volume_fields:
        metrics[f"rel_l2_vol_{field_name}"] = torch.zeros((), device=device, dtype=torch.float32)
    for key in extra_keys or []:
        metrics[key] = torch.zeros((), device=device, dtype=torch.float32)
    return metrics


def accumulate_channel_metrics(metrics, prefix, pred, gt, field_names, rel_l2_loss_fn, batch_size, metric_weight=1.0):
    for channel_idx, field_name in enumerate(field_names):
        channel_loss = rel_l2_loss_fn(pred[..., channel_idx:channel_idx + 1], gt[..., channel_idx:channel_idx + 1])
        metrics[f"{prefix}_{field_name}"] += channel_loss.item() * batch_size * metric_weight


def accumulate_channel_metrics_tensor(metrics, prefix, pred, gt, field_names, rel_l2_loss_fn, batch_size, metric_weight=1.0):
    for channel_idx, field_name in enumerate(field_names):
        channel_loss = rel_l2_loss_fn(pred[..., channel_idx:channel_idx + 1], gt[..., channel_idx:channel_idx + 1]).detach()
        metrics[f"{prefix}_{field_name}"] += channel_loss * float(batch_size) * float(metric_weight)


def add_canonical_field_metrics(wandb_dict, split, surface_fields, volume_fields, metric_values=None):
    metric_values = metric_values or {}
    for f in CANON_SURF_FIELDS:
        key = f"{split}/rel_l2_surf_{f}"
        src_key = f"rel_l2_surf_{f}"
        wandb_dict[key] = metric_values.get(src_key, np.nan) if f in surface_fields else np.nan
    for f in CANON_VOL_FIELDS:
        key = f"{split}/rel_l2_vol_{f}"
        src_key = f"rel_l2_vol_{f}"
        wandb_dict[key] = metric_values.get(src_key, np.nan) if f in volume_fields else np.nan


def add_all_field_metrics(wandb_dict, split, surface_fields, volume_fields, metric_values=None):
    metric_values = metric_values or {}
    for f in surface_fields:
        src_key = f"rel_l2_surf_{f}"
        wandb_dict[f"{split}/rel_l2_surf_{f}"] = metric_values.get(src_key, np.nan)
    for f in volume_fields:
        src_key = f"rel_l2_vol_{f}"
        wandb_dict[f"{split}/rel_l2_vol_{f}"] = metric_values.get(src_key, np.nan)


def set_dataset_epoch(dataset, epoch):
    if hasattr(dataset, "set_epoch"):
        dataset.set_epoch(epoch)


def load_partial_state_dict(model, checkpoint_path, device):
    if not checkpoint_path:
        return 0, 0
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    source = checkpoint.get("model_state_dict", checkpoint)
    target = model.state_dict()
    filtered = {}
    matched = 0
    skipped = 0
    for key, value in source.items():
        if key in target and target[key].shape == value.shape:
            filtered[key] = value
            matched += 1
        else:
            skipped += 1
    target.update(filtered)
    model.load_state_dict(target, strict=False)
    print(f"[resume] Loaded {matched} tensors from {checkpoint_path}; skipped {skipped} incompatible tensors.")
    return matched, skipped


def load_full_training_state(
    model,
    optimizer,
    scheduler,
    scaler,
    checkpoint_path,
    device,
    load_scaler=True,
    uncertainty_balancer=None,
    external_gradnorm=None,
    reset_scheduler=False,
    steps_per_epoch=None,
    target_epochs=None,
):
    if not checkpoint_path:
        raise ValueError("checkpoint_path must be provided for full-state resume.")
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    source = checkpoint.get("model_state_dict")
    if source is None:
        raise KeyError(f"Checkpoint {checkpoint_path} does not contain model_state_dict for full-state resume.")

    model.load_state_dict(source, strict=True)

    optimizer_state = checkpoint.get("optimizer_state_dict")
    scheduler_state = checkpoint.get("scheduler_state_dict")
    if optimizer_state is None or scheduler_state is None:
        raise KeyError(
            f"Checkpoint {checkpoint_path} is missing optimizer/scheduler state required for full-state resume."
        )
    optimizer.load_state_dict(optimizer_state)
    scheduler.load_state_dict(scheduler_state)

    scaler_state = checkpoint.get("scaler_state_dict")
    if load_scaler and scaler_state is not None:
        scaler.load_state_dict(scaler_state)

    if uncertainty_balancer is not None:
        uncertainty_state = checkpoint.get("task_weighting_state_dict")
        if uncertainty_state is None:
            uncertainty_state = checkpoint.get("uncertainty_balancer_state_dict")
        if uncertainty_state is None:
            raise KeyError(
                f"Checkpoint {checkpoint_path} is missing task_weighting_state_dict required for full-state resume."
            )
        uncertainty_balancer.load_state_dict(uncertainty_state, strict=True)
    if external_gradnorm is not None:
        external_gradnorm_state = checkpoint.get("external_gradnorm_state_dict")
        if external_gradnorm_state is None:
            raise KeyError(
                f"Checkpoint {checkpoint_path} is missing external_gradnorm_state_dict required for full-state resume."
            )
        external_gradnorm.load_state_dict(external_gradnorm_state, strict=True)

    resumed_epoch = int(checkpoint.get("epoch", -1))
    start_epoch = resumed_epoch + 1
    if reset_scheduler:
        if steps_per_epoch is None or target_epochs is None:
            raise ValueError("steps_per_epoch and target_epochs are required to reset a resumed scheduler.")
        extension_epochs = max(int(target_epochs) - start_epoch, 1)
        reset_scheduler_for_extension(scheduler, optimizer, extension_epochs * int(steps_per_epoch))
        if is_main_process():
            print(f"[resume] Reset cosine scheduler for {extension_epochs} extension epochs.")
    global_step = int(checkpoint.get("global_step", 0))
    best_robust_rel_l2 = float(checkpoint.get("best_robust_rel_l2", np.inf))
    print(
        f"[resume] Restored full training state from {checkpoint_path}: "
        f"epoch={resumed_epoch}, next_epoch={start_epoch}, global_step={global_step}, "
        f"best_robust_rel_l2={best_robust_rel_l2:.6g}"
    )
    return start_epoch, global_step, best_robust_rel_l2


def build_training_checkpoint(
    *,
    epoch,
    model,
    optimizer,
    scheduler,
    scaler,
    loss,
    rel_l2_loss,
    surface_fields,
    volume_fields,
    global_step,
    best_robust_rel_l2,
    extra_metrics=None,
):
    checkpoint = {
        "epoch": int(epoch),
        "global_step": int(global_step),
        "best_robust_rel_l2": float(best_robust_rel_l2),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "loss": float(loss),
        "rel_l2_loss": float(rel_l2_loss),
        "surface_fields": surface_fields,
        "volume_fields": volume_fields,
    }
    if extra_metrics:
        checkpoint.update(extra_metrics)
    return checkpoint


def unpack_batch(batch, params_dim):
    geo_log_density = None
    if params_dim > 0:
        if len(batch) == 7:
            geo_mesh, surf_mesh, surf_data, vol_mesh, vol_data, params, geo_log_density = batch
        elif len(batch) == 6:
            geo_mesh, surf_mesh, surf_data, vol_mesh, vol_data, params = batch
        else:
            raise ValueError(f"Unexpected parameterized batch size: {len(batch)}")
    else:
        params = None
        if len(batch) == 6:
            geo_mesh, surf_mesh, surf_data, vol_mesh, vol_data, geo_log_density = batch
        elif len(batch) == 5:
            geo_mesh, surf_mesh, surf_data, vol_mesh, vol_data = batch
        else:
            raise ValueError(f"Unexpected batch size: {len(batch)}")
    return geo_mesh, surf_mesh, surf_data, vol_mesh, vol_data, params, geo_log_density


def gather_points(points, idx):
    return torch.gather(points, 1, idx.unsqueeze(-1).expand(-1, -1, points.shape[-1]))


def gather_scalar(points, idx):
    return torch.gather(points, 1, idx)


def move_to_device(value, device):
    if torch.is_tensor(value):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {key: move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [move_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(move_to_device(item, device) for item in value)
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


def _cpu_generator(seed):
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    return gen


def _resolve_sampling_mode(mode, mixed_inverse_density_prob, generator):
    mode = str(mode)
    if mode != "mixed":
        return mode
    draw = torch.rand((), generator=generator).item()
    return "inverse_density_wor" if draw < float(mixed_inverse_density_prob) else "uniform_wor"


def sample_uniform_beta(beta_min, beta_max, generator):
    beta_min = float(beta_min)
    beta_max = float(beta_max)
    if beta_max < beta_min:
        beta_min, beta_max = beta_max, beta_min
    if abs(beta_max - beta_min) < 1e-12:
        return beta_min
    return float(torch.empty((), dtype=torch.float32).uniform_(beta_min, beta_max, generator=generator).item())


def _sample_gaussian_ball_mask_indices(
    geo_row,
    n_points,
    num_points,
    generator,
    std_fraction,
    prob_at_1sigma,
    min_survivors,
):
    if geo_row is None:
        raise RuntimeError("gaussian_ball_mask sampling requires geometry coordinates.")

    if num_points <= 0 or num_points >= n_points:
        base_idx = torch.arange(n_points, dtype=torch.long)
    else:
        base_idx = torch.randperm(n_points, generator=generator)[:num_points].to(dtype=torch.long)

    base_idx_for_geo = base_idx.to(device=geo_row.device, dtype=torch.long)
    geo_subset = geo_row.index_select(0, base_idx_for_geo).detach().float().cpu()
    if geo_subset.shape[0] == 0:
        raise RuntimeError("gaussian_ball_mask sampling produced an empty base subset.")

    center_rel = int(torch.randint(0, int(geo_subset.shape[0]), (1,), generator=generator, dtype=torch.long).item())
    center = geo_subset[center_rel]
    extent = (geo_subset.max(dim=0).values - geo_subset.min(dim=0).values).amax().item()
    sigma = max(float(std_fraction) * float(extent), 1.0e-8)
    prob_at_1sigma = min(max(float(prob_at_1sigma), 1.0e-8), 0.999999)
    gaussian_coeff = -math.log(prob_at_1sigma)
    dist = torch.linalg.vector_norm(geo_subset - center.unsqueeze(0), dim=1)
    remove_prob = torch.exp(-gaussian_coeff * (dist / sigma).pow(2)).clamp_(0.0, 1.0)
    keep_mask = torch.rand((geo_subset.shape[0],), generator=generator, dtype=torch.float32) >= remove_prob

    min_survivors = max(1, min(int(min_survivors), int(base_idx.shape[0])))
    if int(keep_mask.sum().item()) < min_survivors:
        keep_scores = 1.0 - remove_prob
        keep_rel = torch.topk(keep_scores, k=min_survivors, largest=True).indices
        keep_mask = torch.zeros_like(keep_mask, dtype=torch.bool)
        keep_mask[keep_rel] = True

    return base_idx[keep_mask].to(dtype=torch.long), "gaussian_ball_mask"


def _sample_box_mask_indices(
    geo_row,
    n_points,
    num_points,
    generator,
    std_fraction_of_largest_extent,
):
    """Sample a base view and remove a 2-sigma box around a random center."""
    if geo_row is None:
        raise RuntimeError("box_mask sampling requires geometry coordinates.")

    if num_points <= 0 or num_points >= n_points:
        base_idx = torch.arange(n_points, dtype=torch.long)
    else:
        base_idx = torch.randperm(n_points, generator=generator)[:num_points].to(dtype=torch.long)

    base_idx_for_geo = base_idx.to(device=geo_row.device, dtype=torch.long)
    geo_subset = geo_row.index_select(0, base_idx_for_geo).detach().float().cpu()
    if geo_subset.shape[0] == 0:
        raise RuntimeError("box_mask sampling produced an empty base subset.")

    center_rel = int(torch.randint(0, int(geo_subset.shape[0]), (1,), generator=generator, dtype=torch.long).item())
    center = geo_subset[center_rel]
    extent = (geo_subset.max(dim=0).values - geo_subset.min(dim=0).values).amax().item()
    sigma = max(float(std_fraction_of_largest_extent) * float(extent), 1.0e-8)
    side_length = 2.0 * sigma
    half_side = 0.5 * side_length
    remove_mask = (torch.abs(geo_subset - center.unsqueeze(0)) <= half_side).all(dim=1)
    keep_mask = ~remove_mask

    if not bool(keep_mask.any()):
        raise RuntimeError("box_mask sampling removed every point from the secondary view.")

    return base_idx[keep_mask].to(dtype=torch.long), "box_mask"


def _sample_single_view_indices(
    geo_row,
    log_density_row,
    n_points,
    num_points,
    mode,
    inverse_density_beta,
    mixed_inverse_density_prob,
    generator,
    gaussian_mask_std_fraction=0.05,
    gaussian_mask_prob_at_1sigma=0.33,
    gaussian_mask_min_survivors=16384,
):
    resolved_mode = _resolve_sampling_mode(mode, mixed_inverse_density_prob, generator)

    if num_points <= 0:
        return torch.arange(n_points, dtype=torch.long), resolved_mode

    if resolved_mode == "uniform_wor":
        if num_points >= n_points:
            return torch.arange(n_points, dtype=torch.long), resolved_mode
        return torch.randperm(n_points, generator=generator)[:num_points].to(dtype=torch.long), resolved_mode

    if resolved_mode == "uniform_wr":
        return torch.randint(0, n_points, (num_points,), generator=generator, dtype=torch.long), resolved_mode

    if resolved_mode in {"inverse_density_wor", "inverse_density_wr"}:
        if log_density_row is None:
            raise RuntimeError(
                f"Sampling mode {resolved_mode!r} requires geometry log density, but none was provided."
            )
        replacement = resolved_mode.endswith("_wr")
        if not replacement and num_points >= n_points:
            return torch.arange(n_points, dtype=torch.long), resolved_mode

        log_weights = (-float(inverse_density_beta) * log_density_row.float()).cpu()
        log_weights = log_weights - torch.max(log_weights)
        weights = torch.exp(log_weights).clamp_min(1e-12)
        if not torch.isfinite(weights).all() or float(weights.sum()) <= 0.0:
            weights = torch.ones_like(weights)
        idx = torch.multinomial(weights, num_samples=num_points, replacement=replacement, generator=generator)
        return idx.to(dtype=torch.long), resolved_mode

    if resolved_mode == "gaussian_ball_mask":
        return _sample_gaussian_ball_mask_indices(
            geo_row,
            n_points=n_points,
            num_points=num_points,
            generator=generator,
            std_fraction=gaussian_mask_std_fraction,
            prob_at_1sigma=gaussian_mask_prob_at_1sigma,
            min_survivors=gaussian_mask_min_survivors,
        )

    if resolved_mode == "box_mask":
        return _sample_box_mask_indices(
            geo_row,
            n_points=n_points,
            num_points=num_points,
            generator=generator,
            std_fraction_of_largest_extent=gaussian_mask_std_fraction,
        )

    raise ValueError(f"Unsupported sampling mode: {resolved_mode}")


def sample_geometry_view(
    geo_mesh,
    geo_log_density,
    num_points,
    mode,
    inverse_density_beta,
    mixed_inverse_density_prob,
    seed,
    gaussian_mask_std_fraction=0.05,
    gaussian_mask_prob_at_1sigma=0.33,
    gaussian_mask_min_survivors=16384,
):
    if geo_log_density is None and _sampling_mode_requires_density(mode):
        raise RuntimeError(f"Sampling mode {mode!r} requires geometry log density for view sampling.")

    idx_rows = []
    resolved_modes = []
    batch_size = int(geo_mesh.shape[0])
    n_points = int(geo_mesh.shape[1])
    for batch_idx in range(batch_size):
        generator = _cpu_generator(seed + 1009 * batch_idx)
        geo_row = geo_mesh[batch_idx]
        log_density_row = None if geo_log_density is None else geo_log_density[batch_idx]
        idx_row, resolved_mode = _sample_single_view_indices(
            geo_row,
            log_density_row,
            n_points=n_points,
            num_points=num_points,
            mode=mode,
            inverse_density_beta=inverse_density_beta,
            mixed_inverse_density_prob=mixed_inverse_density_prob,
            generator=generator,
            gaussian_mask_std_fraction=gaussian_mask_std_fraction,
            gaussian_mask_prob_at_1sigma=gaussian_mask_prob_at_1sigma,
            gaussian_mask_min_survivors=gaussian_mask_min_survivors,
        )
        idx_rows.append(idx_row)
        resolved_modes.append(resolved_mode)

    row_lengths = [int(idx_row.numel()) for idx_row in idx_rows]
    if len(set(row_lengths)) > 1:
        common_num_points = max(1, min(row_lengths))
        trimmed_rows = []
        for batch_idx, idx_row in enumerate(idx_rows):
            if int(idx_row.numel()) == common_num_points:
                trimmed_rows.append(idx_row)
                continue
            trim_generator = _cpu_generator(seed + 200003 + 1009 * batch_idx)
            keep_rel = torch.randperm(int(idx_row.numel()), generator=trim_generator)[:common_num_points]
            trimmed_rows.append(idx_row[keep_rel].to(dtype=torch.long))
        idx_rows = trimmed_rows

    idx = torch.stack(idx_rows, dim=0)
    if geo_mesh.device.type != idx.device.type or geo_mesh.device != idx.device:
        idx = idx.to(device=geo_mesh.device, non_blocking=(geo_mesh.device.type == "cuda"))
    sampled_density = None if geo_log_density is None else gather_scalar(geo_log_density, idx)
    return gather_points(geo_mesh, idx), sampled_density, resolved_modes


def consistency_warmup_factor(epoch, warmup_epochs):
    warmup_epochs = int(warmup_epochs)
    if warmup_epochs <= 0:
        return 1.0
    return min(1.0, float(epoch + 1) / float(warmup_epochs))


def prediction_consistency_smooth_l1_loss(y1_surf, y1_vol, y2_surf, y2_vol, beta=0.05):
    surf_target = (0.5 * (y1_surf.detach() + y2_surf.detach())).to(dtype=y1_surf.dtype)
    vol_target = (0.5 * (y1_vol.detach() + y2_vol.detach())).to(dtype=y1_vol.dtype)
    surf_loss = 0.5 * (
        F.smooth_l1_loss(y1_surf, surf_target, beta=beta) + F.smooth_l1_loss(y2_surf, surf_target, beta=beta)
    )
    vol_loss = 0.5 * (
        F.smooth_l1_loss(y1_vol, vol_target, beta=beta) + F.smooth_l1_loss(y2_vol, vol_target, beta=beta)
    )
    return surf_loss + vol_loss


def soft_worst_case_loss(loss_a, loss_b, tau=0.1):
    tau = max(float(tau), 1e-6)
    pair = torch.stack([loss_a.float(), loss_b.float()], dim=0)
    return torch.logsumexp(pair / tau, dim=0) * tau


def latent_consistency_loss(latent_teacher, latent_student):
    latent_teacher = F.layer_norm(latent_teacher.detach().float(), (latent_teacher.shape[-1],))
    latent_student = F.layer_norm(latent_student.float(), (latent_student.shape[-1],))
    return F.mse_loss(latent_student, latent_teacher)


class LearnedTaskWeighting(nn.Module):
    def __init__(
        self,
        task_names,
        init_logits=None,
        min_logit=-4.0,
        max_logit=4.0,
        min_weights=None,
        base_weights=None,
        warmup_epochs=0,
    ):
        super().__init__()
        self.task_names = tuple(task_names)
        if init_logits is None:
            init_logits = [0.0] * len(self.task_names)
        if len(init_logits) != len(self.task_names):
            raise ValueError(f"Expected {len(self.task_names)} init logits, got {len(init_logits)}.")
        self.logits = nn.Parameter(torch.tensor(init_logits, dtype=torch.float32))
        self.min_logit = float(min_logit)
        self.max_logit = float(max_logit)
        if min_weights is None:
            min_weights = [0.0] * len(self.task_names)
        if base_weights is None:
            base_weights = [1.0 / len(self.task_names)] * len(self.task_names)
        if len(min_weights) != len(self.task_names):
            raise ValueError(f"Expected {len(self.task_names)} min weights, got {len(min_weights)}.")
        if len(base_weights) != len(self.task_names):
            raise ValueError(f"Expected {len(self.task_names)} base weights, got {len(base_weights)}.")
        min_weights_t = torch.tensor(min_weights, dtype=torch.float32)
        base_weights_t = torch.tensor(base_weights, dtype=torch.float32)
        if torch.any(min_weights_t < 0):
            raise ValueError("min_weights must be non-negative.")
        if float(min_weights_t.sum().item()) >= 1.0:
            raise ValueError("Sum of min_weights must be < 1.")
        if torch.any(base_weights_t < 0):
            raise ValueError("base_weights must be non-negative.")
        if not torch.isclose(base_weights_t.sum(), torch.tensor(1.0, dtype=torch.float32), atol=1e-5):
            raise ValueError("base_weights must sum to 1.")
        self.register_buffer("min_weights", min_weights_t)
        self.register_buffer("base_weights", base_weights_t)
        self.warmup_epochs = int(warmup_epochs)

    def combine(self, losses, epoch_idx=None):
        if len(losses) != len(self.task_names):
            raise ValueError(f"Expected {len(self.task_names)} task losses, got {len(losses)}.")
        stacked_losses = torch.stack([loss.float() for loss in losses])
        clamped_logits = torch.clamp(self.logits, min=self.min_logit, max=self.max_logit)
        raw_weights = torch.softmax(clamped_logits, dim=0)
        free_mass = 1.0 - self.min_weights.sum()
        learned_weights = self.min_weights + free_mass * raw_weights
        if self.warmup_epochs > 0 and epoch_idx is not None:
            mix = min(1.0, float(epoch_idx + 1) / float(self.warmup_epochs))
            weights = (1.0 - mix) * self.base_weights + mix * learned_weights
        else:
            weights = learned_weights
        weighted_terms = weights * stacked_losses
        total_loss = weighted_terms.sum()
        return {
            "total_loss": total_loss,
            "weights": weights.detach(),
            "raw_weights": raw_weights.detach(),
            "logits": clamped_logits.detach(),
            "per_task_terms": weighted_terms.detach(),
        }


def move_optional_tensor(x, device):
    if x is None:
        return None
    return x.to(device, non_blocking=True)


def duplicate_batch_tensor(x):
    if x is None:
        return None
    return torch.cat([x, x], dim=0)


def _optional_positive_int(value):
    if value is None:
        return None
    value = int(value)
    return value if value > 0 else None


def _sampling_mode_requires_density(mode):
    mode = str(mode)
    return mode == "mixed" or mode.startswith("inverse_density")


@contextmanager
def temporary_model_attr(model, attr_name, value):
    value = _optional_positive_int(value)
    if value is None:
        yield
        return

    base_model = unwrap_model(model)
    if not hasattr(base_model, attr_name):
        yield
        return

    old_value = getattr(base_model, attr_name)
    setattr(base_model, attr_name, value)
    try:
        yield
    finally:
        setattr(base_model, attr_name, old_value)


def forward_model_view(
    model,
    geo,
    surf_mesh,
    vol_mesh,
    params,
    *,
    model_requires_density,
    geo_log_density=None,
    return_latent=False,
    subsampled_geometry_points=None,
):
    with temporary_model_attr(model, "subsampled_geometry_points", subsampled_geometry_points):
        if model_requires_density:
            if return_latent:
                return model(
                    geo,
                    surf_mesh,
                    vol_mesh,
                    params,
                    geo_log_density=geo_log_density,
                    return_latent=True,
                )
            return model(
                geo,
                surf_mesh,
                vol_mesh,
                params,
                geo_log_density=geo_log_density,
            )
        if return_latent:
            return model(
                geo,
                surf_mesh,
                vol_mesh,
                params,
                return_latent=True,
            )
        return model(
            geo,
            surf_mesh,
            vol_mesh,
            params,
        )


def inference_model_view(
    model,
    geo,
    surf_mesh,
    vol_mesh,
    params,
    *,
    model_requires_density,
    geo_log_density=None,
    subsampled_geometry_points=None,
):
    with temporary_model_attr(model, "subsampled_geometry_points", subsampled_geometry_points):
        if model_requires_density:
            return model.inference(geo, surf_mesh, vol_mesh, params, geo_log_density=geo_log_density)
        return model.inference(geo, surf_mesh, vol_mesh, params)


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
    mode_name,
    num_view_points,
    eval_subsampled_geometry_points,
    fixed_seed_offset,
    model_requires_density,
    cuda_batch_prefetch,
):
    metrics = init_metric_dict(fields["surface"], fields["volume"])
    model.eval()
    sample_count = 0

    eval_loader = CudaPrefetchLoader(loader, device) if cuda_batch_prefetch else loader
    pbar = tqdm(eval_loader, desc=f"Eval {mode_name}", leave=False, dynamic_ncols=True, disable=not is_main_process())
    with torch.inference_mode():
        for batch_idx, batch in enumerate(pbar):
            geo_mesh, surf_mesh, surf_data, vol_mesh, vol_data, params, geo_log_density = unpack_batch(batch, params_dim)

            view_geo, view_log_density, _ = sample_geometry_view(
                geo_mesh,
                geo_log_density,
                num_points=num_view_points,
                mode=mode_name,
                inverse_density_beta=float(getattr(config, "inverse_density_beta", 1.0)),
                mixed_inverse_density_prob=float(getattr(config, "mixed_inverse_density_prob", 0.5)),
                seed=int(config.random_seed + fixed_seed_offset + batch_idx * 10007),
            )

            params = move_optional_tensor(params, device)
            surf_mesh = surf_mesh.to(device, non_blocking=True)
            surf_data = surf_data.to(device, non_blocking=True)
            vol_mesh = vol_mesh.to(device, non_blocking=True)
            vol_data = vol_data.to(device, non_blocking=True)
            view_geo = view_geo.to(device, non_blocking=True)
            if model_requires_density:
                view_log_density = view_log_density.to(device, non_blocking=True)

            with torch.autocast(device_type=str(device).split(":")[0], dtype=dtype, enabled=amp):
                y_hat_surf, y_hat_vol = inference_model_view(
                    model,
                    view_geo,
                    surf_mesh,
                    vol_mesh,
                    params,
                    model_requires_density=model_requires_density,
                    geo_log_density=view_log_density if model_requires_density else None,
                    subsampled_geometry_points=eval_subsampled_geometry_points,
                )

            if use_surface_supervision:
                pred_surf = y_hat_surf * std_surf + mean_surf
                gt_surf = surf_data * std_surf + mean_surf
                batch_loss = combined_loss_fn(y_hat_surf, y_hat_vol, surf_data, vol_data)
                surface_rel_l2 = rel_l2_loss_fn(y_hat_surf, surf_data)
            else:
                pred_surf = None
                gt_surf = None
                batch_loss = loss_fn(y_hat_vol, vol_data)
                surface_rel_l2 = torch.tensor(0.0, device=device)

            pred_vol = y_hat_vol * std_vol + mean_vol
            gt_vol = vol_data * std_vol + mean_vol
            volume_rel_l2 = rel_l2_loss_fn(y_hat_vol, vol_data)

            batch_size = surf_data.size(0)
            sample_count += batch_size
            metrics["loss"] += batch_loss.item() * batch_size
            metrics["rel_l2_surf"] += surface_rel_l2.item() * batch_size
            metrics["rel_l2_vol"] += volume_rel_l2.item() * batch_size
            metrics["rel_l2"] += (surface_rel_l2 + volume_rel_l2).item() * batch_size

            if use_surface_supervision:
                accumulate_channel_metrics(metrics, "rel_l2_surf", pred_surf, gt_surf, fields["surface"], rel_l2_loss_fn, batch_size)
            accumulate_channel_metrics(metrics, "rel_l2_vol", pred_vol, gt_vol, fields["volume"], rel_l2_loss_fn, batch_size)

            pbar.set_postfix(loss=f"{batch_loss.item():.4f}")

    if is_dist_enabled():
        key_list = list(metrics.keys())
        metric_tensor = torch.tensor(
            [metrics[key] for key in key_list] + [float(sample_count)],
            device=device,
            dtype=torch.float64,
        )
        dist.all_reduce(metric_tensor, op=dist.ReduceOp.SUM)
        sample_count = max(int(metric_tensor[-1].item()), 1)
        for idx, key in enumerate(key_list):
            metrics[key] = float(metric_tensor[idx].item())

    denom = max(int(sample_count), 1)
    for key in metrics.keys():
        metrics[key] /= denom
    return metrics


def run_consistency_training(cfg, model_ctor, model_requires_density):
    config = cfg.experiment
    wandb_config = cfg.wandb
    multi_gpu_strategy = str(getattr(config, "multi_gpu_strategy", "auto")).lower()
    if multi_gpu_strategy == "data_parallel" and int(os.environ.get("WORLD_SIZE", "1")) > 1:
        raise RuntimeError(
            "multi_gpu_strategy=data_parallel must be launched with plain python, not torchrun/DDP. "
            "Use CUDA_VISIBLE_DEVICES=1,2 python smart/train_satloss4.py ..."
        )

    dist_info = setup_distributed() if multi_gpu_strategy != "data_parallel" else {
        "enabled": False,
        "rank": 0,
        "local_rank": 0,
        "world_size": 1,
    }
    run = initialize_wandb(config, wandb_config) if is_main_process() else None

    device = initialize_gpu(config.random_seed, high_precision=False)

    gradient_norm = config.gradient_norm
    track_gradient_diagnostics = bool(getattr(config, "track_gradient_diagnostics", False))
    precisions = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    dtype = precisions.get(config.precision, torch.float16)
    amp = config.amp
    if is_main_process():
        print(gradient_norm, amp, dtype)

    train_data, test_data, stats, spatial_dim, surf_channels, vol_channels, params_dim, fields = get_dataset(config)

    def apply_vanilla_smart_field_subset():
        nonlocal fields, surf_channels, vol_channels
        if config.dataset == "NACA4":
            fields = {"surface": ["pressure"], "volume": ["pressure", "velocity_x", "velocity_y"]}
            surf_channels = 1
            vol_channels = 3

    apply_vanilla_smart_field_subset()
    if is_main_process():
        print(f"[{config.model_name}] training signals -> surface: {fields['surface']} | volume: {fields['volume']}")

    point_info = apply_naca4_auto_point_budget(config, train_data, for_cat=False)
    if point_info is not None:
        if is_main_process():
            print_point_budget(config.model_name, point_info)
        train_data, test_data, stats, spatial_dim, surf_channels, vol_channels, params_dim, fields = get_dataset(config)
        apply_vanilla_smart_field_subset()
        if is_main_process():
            print(f"[{config.model_name}] training signals -> surface: {fields['surface']} | volume: {fields['volume']}")

    use_surface_supervision = len(fields["surface"]) > 0
    set_dataset_epoch(train_data, 0)
    set_dataset_epoch(test_data, 0)

    prefetch_factor = int(getattr(config, "prefetch_factor", 2))
    pin_memory = bool(getattr(config, "pin_memory", True))
    cuda_batch_prefetch = bool(getattr(config, "cuda_batch_prefetch", device.type == "cuda")) and device.type == "cuda"
    effective_num_workers = int(getattr(config, "num_workers", 0))
    if dist_info["enabled"]:
        effective_num_workers = 0
        prefetch_factor = 2
        pin_memory = False
    dl_common = dict(batch_size=config.batch_size, num_workers=effective_num_workers, pin_memory=pin_memory)
    if effective_num_workers > 0:
        dl_common["prefetch_factor"] = prefetch_factor
        dl_common["persistent_workers"] = not dist_info["enabled"]
        if dist_info["enabled"]:
            dl_common["multiprocessing_context"] = "spawn"
    if is_main_process():
        print(
            f"[dataloader] world_size={dist_info['world_size']}, "
            f"num_workers_per_rank={effective_num_workers}, "
            f"prefetch_factor={dl_common.get('prefetch_factor', 'n/a')}, "
            f"persistent_workers={dl_common.get('persistent_workers', False)}, "
            f"cuda_batch_prefetch={cuda_batch_prefetch}"
        )

    train_sampler = DistributedSampler(train_data, shuffle=True) if dist_info["enabled"] else None
    train_loader = torch.utils.data.DataLoader(
        train_data,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        **dl_common,
    )
    test_loader = torch.utils.data.DataLoader(
        test_data,
        shuffle=False,
        **dl_common,
    )

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
    if is_main_process():
        print(f"Model kwargs: {merged_kwargs}")
    model = model_ctor(**merged_kwargs).to(device)
    rollback_buffers = bool(getattr(model, "rollback_buffers_on_nonfinite", False))

    if is_main_process():
        print(f"Total parameters: {count_model_params(model)}")
    model_checkpoint_name = get_model_checkpoint_name(config)
    if is_main_process():
        print(f"Checkpoint name: {model_checkpoint_name}")
    if run is not None and bool(getattr(config, "wandb_watch_model", False)):
        run.watch(model, log="all")

    use_prediction_consistency = bool(getattr(config, "use_prediction_consistency", True))
    task_weighting_method = str(getattr(config, "task_weighting_method", "legacy")).strip().lower()
    use_config_layer = task_weighting_method in {"config", "config_layer", "config-layer"}
    use_config_full = task_weighting_method in {"config_full", "config-full", "configfull"}
    use_external_gradnorm = task_weighting_method in {"gradnorm_external", "external_gradnorm", "gradnorm"}
    use_fixed_sum = task_weighting_method in {"fixed_sum", "fixed-sum", "fixedsum"}
    use_learned_task_weighting = bool(
        getattr(config, "use_learned_task_weighting", getattr(config, "use_uncertainty_weighting", False))
    )
    if sum((use_config_layer, use_config_full, use_external_gradnorm, use_fixed_sum, use_learned_task_weighting)) > 1:
        raise ValueError(
            "Only one task weighting backend can be enabled: task_weighting_method=CONFIG, "
            "task_weighting_method=config_full, "
            "task_weighting_method=gradnorm_external, task_weighting_method=fixed_sum, "
            "or use_learned_task_weighting."
        )
    scaler = make_grad_scaler(config)
    uncertainty_balancer = None
    config_reference_parameter = None
    config_full_parameters = None
    config_task_weights = None
    external_gradnorm = None
    external_gradnorm_min_weights = None
    extra_optimizer_param_groups = []
    if use_learned_task_weighting:
        learned_task_names = ["supervised_mean", "supervised_worst"]
        if use_prediction_consistency:
            learned_task_names.append("prediction_consistency")
        uncertainty_balancer = LearnedTaskWeighting(
            task_names=tuple(learned_task_names),
            init_logits=list(getattr(config, "task_weight_init_logits", [0.0] * len(learned_task_names))),
            min_logit=float(getattr(config, "task_weight_logit_min", -4.0)),
            max_logit=float(getattr(config, "task_weight_logit_max", 4.0)),
            min_weights=list(getattr(config, "task_weight_min_weights", [0.4] * len(learned_task_names))),
            base_weights=list(getattr(config, "task_weight_base_weights", [1.0 / len(learned_task_names)] * len(learned_task_names))),
            warmup_epochs=int(getattr(config, "task_weight_warmup_epochs", 25)),
        ).to(device)
        extra_optimizer_param_groups.append(
            {
                "params": list(uncertainty_balancer.parameters()),
                "lr": float(getattr(config, "task_weight_lr", 1.0e-3)),
                "weight_decay": 0.0,
            }
        )
    if use_config_layer or use_config_full or use_external_gradnorm or use_fixed_sum:
        task_count = 3 if use_prediction_consistency else 2
        configured_task_weights = list(
            getattr(config, "config_task_base_weights", getattr(config, "task_weight_base_weights", []))
        )
        if not configured_task_weights:
            configured_task_weights = [1.0 / task_count] * task_count
        if len(configured_task_weights) != task_count:
            raise ValueError(
                f"Expected {task_count} task weights for {task_weighting_method}, "
                f"got {len(configured_task_weights)}."
            )
        weight_sum = float(sum(float(weight) for weight in configured_task_weights))
        if weight_sum <= 0.0:
            raise ValueError(f"Task weights for {task_weighting_method} must have a positive sum.")
        config_task_weights = [float(weight) / weight_sum for weight in configured_task_weights]
        if use_config_layer or use_external_gradnorm:
            config_reference_parameter = resolve_balance_reference_parameter(model)
        if use_config_full:
            config_full_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        if use_external_gradnorm:
            configured_min_weights = getattr(config, "external_gradnorm_min_weights", None)
            if configured_min_weights is not None:
                external_gradnorm_min_weights = [float(weight) for weight in configured_min_weights]
                if len(external_gradnorm_min_weights) != task_count:
                    raise ValueError(
                        f"Expected {task_count} external_gradnorm_min_weights, "
                        f"got {len(external_gradnorm_min_weights)}."
                    )
                if any(weight < 0.0 for weight in external_gradnorm_min_weights):
                    raise ValueError("external_gradnorm_min_weights must be non-negative.")
                if sum(external_gradnorm_min_weights) >= 1.0:
                    raise ValueError("external_gradnorm_min_weights must sum to less than 1.0.")
            try:
                from gradnorm_pytorch import GradNormLossWeighter
            except ImportError as exc:
                raise RuntimeError(
                    "task_weighting_method=gradnorm_external requires gradnorm-pytorch. "
                    "Install it with: python -m pip install gradnorm-pytorch"
                ) from exc
            external_gradnorm = GradNormLossWeighter(
                num_losses=task_count,
                loss_weights=config_task_weights,
                learning_rate=float(getattr(config, "external_gradnorm_lr", 1.0e-4)),
                restoring_force_alpha=float(getattr(config, "external_gradnorm_alpha", 0.0)),
                grad_norm_parameters=config_reference_parameter,
                initial_losses_decay=float(getattr(config, "external_gradnorm_initial_losses_decay", 1.0)),
                update_after_step=int(getattr(config, "external_gradnorm_update_after_step", 0)),
                update_every=int(getattr(config, "external_gradnorm_update_every", 1)),
            ).to(device)
    if is_main_process():
        active_backend = (
            task_weighting_method
            if use_config_layer or use_config_full or use_external_gradnorm
            else "uncertainty"
            if use_learned_task_weighting
            else "fixed_sum"
            if use_fixed_sum
            else "fixed"
        )
        print(f"[loss balancing] backend={active_backend}")
    optimizer, scheduler, loss_fn, rel_l2_loss_fn = get_optimizer_scheduler_loss(
        model,
        config,
        train_loader,
        loss_dim=1,
        extra_param_groups=extra_optimizer_param_groups,
    )
    combined_loss_fn = CombinedLoss(loss_fn, fields) if use_surface_supervision else None

    best_robust_rel_l2 = np.inf
    global_step = 0
    start_epoch = 0
    log_every_n_steps = getattr(config, "log_every_n_steps", 10)

    primary_view_points = int(getattr(config, "primary_view_geometry_points", getattr(config, "view_geometry_points", 0)))
    secondary_view_points = int(getattr(config, "secondary_view_geometry_points", getattr(config, "view_geometry_points", 0)))
    eval_aligned_view_points = int(
        getattr(config, "eval_aligned_view_geometry_points", getattr(config, "eval_view_geometry_points", primary_view_points))
    )
    eval_shifted_view_points = int(
        getattr(config, "eval_shifted_view_geometry_points", getattr(config, "eval_view_geometry_points", secondary_view_points))
    )
    primary_view_subsampled_geometry_points = _optional_positive_int(
        getattr(config, "primary_view_subsampled_geometry_points", 0)
    )
    secondary_view_subsampled_geometry_points = _optional_positive_int(
        getattr(config, "secondary_view_subsampled_geometry_points", 0)
    )
    eval_aligned_subsampled_geometry_points = _optional_positive_int(
        getattr(config, "eval_aligned_subsampled_geometry_points", primary_view_subsampled_geometry_points or 0)
    )
    eval_shifted_subsampled_geometry_points = _optional_positive_int(
        getattr(config, "eval_shifted_subsampled_geometry_points", secondary_view_subsampled_geometry_points or 0)
    )

    init_ckpt = str(getattr(config, "init_ckpt", "")).strip()
    resume_ckpt = str(getattr(config, "resume_ckpt", "")).strip()
    resume_full_state = bool(getattr(config, "resume_full_state", False))
    if resume_full_state:
        if not resume_ckpt:
            raise ValueError("resume_full_state=True requires experiment.resume_ckpt to be set.")
        start_epoch, global_step, best_robust_rel_l2 = load_full_training_state(
            model,
            optimizer,
            scheduler,
            scaler,
            resume_ckpt,
            device,
            load_scaler=bool(amp) and not bool(getattr(config, "amp_scaler_reset_on_resume", False)),
            uncertainty_balancer=uncertainty_balancer,
            external_gradnorm=external_gradnorm,
            reset_scheduler=bool(getattr(config, "scheduler_reset_on_resume", False)),
            steps_per_epoch=len(train_loader),
            target_epochs=int(config.epochs),
        )
    elif resume_ckpt:
        if is_main_process():
            print(f"[init] experiment.resume_ckpt is being used for partial model initialization from {resume_ckpt}")
        load_partial_state_dict(model, resume_ckpt, device)
    elif init_ckpt:
        if is_main_process():
            print(f"[init] Loading model weights from {init_ckpt}")
        load_partial_state_dict(model, init_ckpt, device)

    train_model = model
    if multi_gpu_strategy == "data_parallel" and torch.cuda.is_available() and torch.cuda.device_count() > 1:
        visible_gpus = torch.cuda.device_count()
        if int(config.batch_size) < visible_gpus and is_main_process():
            print(
                f"[multi-gpu warning] batch_size={int(config.batch_size)} is smaller than the number of visible GPUs "
                f"({visible_gpus}); DataParallel will underutilize devices."
            )
        # Keep DataParallel as lightweight as possible:
        # - make device placement explicit
        # - avoid buffer broadcasts because SMART-family models do not use BN/SyncBN
        device_ids = list(range(visible_gpus))
        train_model = DataParallel(
            model,
            device_ids=device_ids,
            output_device=device_ids[0],
            dim=0,
        )
        if is_main_process():
            print(f"[multi-gpu] Using DataParallel on {visible_gpus} GPUs.")
    elif dist_info["enabled"]:
        ddp_kwargs = {
            "device_ids": [dist_info["local_rank"]] if device.type == "cuda" else None,
            "output_device": dist_info["local_rank"] if device.type == "cuda" else None,
            "broadcast_buffers": False,
            "find_unused_parameters": False,
            "gradient_as_bucket_view": True,
            "static_graph": True,
        }
        train_model = DDP(model, **ddp_kwargs)

    train_extra_keys = [
        "loss_supervised_primary",
        "loss_supervised_secondary",
        "loss_supervised_mean",
        "loss_supervised_worst",
        "loss_supervised_worst_soft",
        "loss_prediction_consistency",
        "primary_inverse_density_fraction",
        "primary_inverse_density_beta",
        "secondary_inverse_density_fraction",
        "secondary_inverse_density_beta",
        "learned_weight_supervised_mean",
        "learned_weight_supervised_worst",
        "learned_weight_prediction_consistency",
        "learned_logit_supervised_mean",
        "learned_logit_supervised_worst",
        "learned_logit_prediction_consistency",
        "external_gradnorm_weight_primary",
        "external_gradnorm_weight_secondary",
        "external_gradnorm_weight_prediction_consistency",
        "external_gradnorm_loss",
        "config_reference_grad_norm",
        "config_direction_norm",
        "config_direction_scale",
        "config_direction_cosine",
        "config_direction_fallback",
        "config_full_mean_grad_norm",
        "config_full_direction_norm",
        "config_full_min_cosine",
        "config_full_fallback",
        "gradient_norm_raw",
        "parameter_norm",
        "gradient_max_abs",
        "gradient_nonfinite",
    ]

    fuse_consistency_views = bool(getattr(config, "fuse_consistency_views", False))
    if (
        primary_view_points != secondary_view_points
        or primary_view_subsampled_geometry_points != secondary_view_subsampled_geometry_points
    ):
        if fuse_consistency_views and is_main_process():
            print(
                "[consistency] Disabling fused consistency views because primary and secondary "
                "geometry budgets differ."
            )
        fuse_consistency_views = False

    configured_source_points = int(getattr(config, "num_body_points", 0))
    if (
        use_prediction_consistency
        and configured_source_points > 0
        and primary_view_points >= configured_source_points
        and secondary_view_points >= configured_source_points
    ):
        raise ValueError(
            f"{config.model_name} has no effective two-view geometry augmentation: "
            f"num_body_points={configured_source_points}, "
            f"primary_view_geometry_points={primary_view_points}, "
            f"secondary_view_geometry_points={secondary_view_points}. "
            "Set experiment.num_body_points=0 (full cloud) or make both view budgets smaller "
            "than the dataset geometry budget."
        )
    if is_main_process() and use_prediction_consistency:
        source_text = "full geometry cloud" if configured_source_points <= 0 else f"{configured_source_points} dataset geometry points"
        print(
            f"[consistency] source={source_text}, "
            f"views=({primary_view_points}, {secondary_view_points}), "
            f"modes=({getattr(config, 'train_primary_sampling_mode', 'uniform_wor')}, "
            f"{getattr(config, 'train_secondary_sampling_mode', 'uniform_wor')})"
        )

    try:
        for ep in tqdm(range(start_epoch, config.epochs), desc="Epochs", dynamic_ncols=True):
            t1 = default_timer()
            set_dataset_epoch(train_data, ep)
            set_dataset_epoch(test_data, 0)
            if train_sampler is not None:
                train_sampler.set_epoch(ep)

            train_losses = init_metric_tensor_dict(fields["surface"], fields["volume"], device, extra_keys=train_extra_keys)
            train_model.train()
            train_sample_count = 0
            train_batch_source = CudaPrefetchLoader(train_loader, device) if cuda_batch_prefetch else train_loader
            train_pbar = tqdm(
                train_batch_source,
                desc=f"Train {ep + 1}/{config.epochs}",
                leave=False,
                dynamic_ncols=True,
                miniters=max(1, int(log_every_n_steps)),
                mininterval=1.0,
                smoothing=0.0,
                disable=not is_main_process(),
            )

            warmup = consistency_warmup_factor(ep, getattr(config, "consistency_warmup_epochs", 0))
            pred_consistency_weight = warmup * float(getattr(config, "prediction_consistency_weight", 1.0))
            if not use_prediction_consistency:
                pred_consistency_weight = 0.0
            use_latent_consistency = bool(getattr(config, "use_latent_consistency", False))

            for batch_idx, batch in enumerate(train_pbar):
                geo_mesh, surf_mesh, surf_data, vol_mesh, vol_data, params, geo_log_density = unpack_batch(batch, params_dim)
                primary_sampling_mode = str(getattr(config, "train_primary_sampling_mode", "uniform_wor"))
                secondary_sampling_mode = str(getattr(config, "train_secondary_sampling_mode", "mixed"))
                if geo_log_density is None and (
                    _sampling_mode_requires_density(primary_sampling_mode)
                    or _sampling_mode_requires_density(secondary_sampling_mode)
                ):
                    raise RuntimeError(
                        f"{config.model_name} requested density-based geometry-view resampling "
                        f"(primary={primary_sampling_mode!r}, secondary={secondary_sampling_mode!r}) "
                        "but the dataset did not return geometry log density."
                    )

                primary_beta_generator = _cpu_generator(int(config.random_seed + ep * 1000003 + batch_idx * 10007 + 17))
                if bool(getattr(config, "randomize_primary_inverse_density_beta", False)):
                    primary_inverse_density_beta = sample_uniform_beta(
                        getattr(config, "primary_inverse_density_beta_min", 0.0),
                        getattr(config, "primary_inverse_density_beta_max", 0.5),
                        primary_beta_generator,
                    )
                else:
                    primary_inverse_density_beta = float(getattr(config, "inverse_density_beta", 1.0))
                primary_view_geo, primary_view_density, primary_modes = sample_geometry_view(
                    geo_mesh,
                    geo_log_density,
                    num_points=primary_view_points,
                    mode=primary_sampling_mode,
                    inverse_density_beta=primary_inverse_density_beta,
                    mixed_inverse_density_prob=float(getattr(config, "mixed_inverse_density_prob", 0.5)),
                    seed=int(config.random_seed + ep * 1000003 + batch_idx * 10007 + 11),
                    gaussian_mask_std_fraction=float(getattr(config, "gaussian_mask_std_fraction_of_largest_extent", 0.05)),
                    gaussian_mask_prob_at_1sigma=float(getattr(config, "gaussian_mask_prob_at_1sigma", 0.33)),
                    gaussian_mask_min_survivors=int(getattr(config, "gaussian_mask_min_survivors", 16384)),
                )
                secondary_beta_generator = _cpu_generator(int(config.random_seed + ep * 1000003 + batch_idx * 10007 + 23))
                if bool(getattr(config, "randomize_secondary_inverse_density_beta", False)):
                    secondary_inverse_density_beta = sample_uniform_beta(
                        getattr(config, "secondary_inverse_density_beta_min", 0.1),
                        getattr(config, "secondary_inverse_density_beta_max", 0.5),
                        secondary_beta_generator,
                    )
                else:
                    secondary_inverse_density_beta = float(getattr(config, "inverse_density_beta", 1.0))
                secondary_view_geo, secondary_view_density, secondary_modes = sample_geometry_view(
                    geo_mesh,
                    geo_log_density,
                    num_points=secondary_view_points,
                    mode=secondary_sampling_mode,
                    inverse_density_beta=secondary_inverse_density_beta,
                    mixed_inverse_density_prob=float(getattr(config, "mixed_inverse_density_prob", 0.5)),
                    seed=int(config.random_seed + ep * 1000003 + batch_idx * 10007 + 29),
                    gaussian_mask_std_fraction=float(getattr(config, "gaussian_mask_std_fraction_of_largest_extent", 0.05)),
                    gaussian_mask_prob_at_1sigma=float(getattr(config, "gaussian_mask_prob_at_1sigma", 0.33)),
                    gaussian_mask_min_survivors=int(getattr(config, "gaussian_mask_min_survivors", 16384)),
                )

                params = move_optional_tensor(params, device)
                surf_mesh = surf_mesh.to(device, non_blocking=True)
                surf_data = surf_data.to(device, non_blocking=True)
                vol_mesh = vol_mesh.to(device, non_blocking=True)
                vol_data = vol_data.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                buffer_snapshot = snapshot_model_buffers(model) if rollback_buffers else None

                primary_view_geo = primary_view_geo.to(device, non_blocking=True)
                if model_requires_density:
                    primary_view_density = primary_view_density.to(device, non_blocking=True)
                secondary_view_geo = secondary_view_geo.to(device, non_blocking=True)
                if model_requires_density:
                    secondary_view_density = secondary_view_density.to(device, non_blocking=True)

                with torch.autocast(device_type=str(device).split(":")[0], dtype=dtype, enabled=amp):
                    if fuse_consistency_views:
                        fused_geo = torch.cat([primary_view_geo, secondary_view_geo], dim=0)
                        fused_surf_mesh = duplicate_batch_tensor(surf_mesh)
                        fused_vol_mesh = duplicate_batch_tensor(vol_mesh)
                        fused_params = duplicate_batch_tensor(params)
                        fused_density = (
                            torch.cat([primary_view_density, secondary_view_density], dim=0)
                            if model_requires_density else None
                        )
                        fused_out = forward_model_view(
                            train_model,
                            fused_geo,
                            fused_surf_mesh,
                            fused_vol_mesh,
                            fused_params,
                            model_requires_density=model_requires_density,
                            geo_log_density=fused_density,
                            return_latent=use_latent_consistency,
                            subsampled_geometry_points=primary_view_subsampled_geometry_points,
                        )
                        if use_latent_consistency:
                            fused_surf, fused_vol, fused_latent = fused_out
                            y1_surf, y2_surf = fused_surf.chunk(2, dim=0)
                            y1_vol, y2_vol = fused_vol.chunk(2, dim=0)
                            latent1, latent2 = fused_latent.chunk(2, dim=0)
                        else:
                            fused_surf, fused_vol = fused_out
                            y1_surf, y2_surf = fused_surf.chunk(2, dim=0)
                            y1_vol, y2_vol = fused_vol.chunk(2, dim=0)
                            latent1 = None
                            latent2 = None
                    else:
                        primary_out = forward_model_view(
                            train_model,
                            primary_view_geo,
                            surf_mesh,
                            vol_mesh,
                            params,
                            model_requires_density=model_requires_density,
                            geo_log_density=primary_view_density if model_requires_density else None,
                            return_latent=use_latent_consistency,
                            subsampled_geometry_points=primary_view_subsampled_geometry_points,
                        )
                        if use_latent_consistency:
                            y1_surf, y1_vol, latent1 = primary_out
                        else:
                            y1_surf, y1_vol = primary_out
                            latent1 = None
                        secondary_out = forward_model_view(
                            train_model,
                            secondary_view_geo,
                            surf_mesh,
                            vol_mesh,
                            params,
                            model_requires_density=model_requires_density,
                            geo_log_density=secondary_view_density if model_requires_density else None,
                            return_latent=use_latent_consistency,
                            subsampled_geometry_points=secondary_view_subsampled_geometry_points,
                        )
                        if use_latent_consistency:
                            y2_surf, y2_vol, latent2 = secondary_out
                        else:
                            y2_surf, y2_vol = secondary_out
                            latent2 = None

                    y1_surf_f = y1_surf.float()
                    y1_vol_f = y1_vol.float()
                    y2_surf_f = y2_surf.float()
                    y2_vol_f = y2_vol.float()

                supervised_primary = combined_loss_fn(y1_surf_f, y1_vol_f, surf_data, vol_data) if use_surface_supervision else loss_fn(y1_vol_f, vol_data)
                supervised_secondary = combined_loss_fn(y2_surf_f, y2_vol_f, surf_data, vol_data) if use_surface_supervision else loss_fn(y2_vol_f, vol_data)
                y1_surf_teacher = y1_surf_f.detach()
                y1_vol_teacher = y1_vol_f.detach()
                if use_prediction_consistency:
                    pred_consistency = prediction_consistency_smooth_l1_loss(
                        y1_surf_teacher,
                        y1_vol_teacher,
                        y2_surf_f,
                        y2_vol_f,
                        beta=float(getattr(config, "prediction_consistency_smooth_l1_beta", 0.05)),
                    )
                else:
                    pred_consistency = y1_vol_f.new_zeros(())
                supervised_mean = 0.5 * (supervised_primary + supervised_secondary)
                supervised_worst = torch.maximum(supervised_primary, supervised_secondary)
                supervised_worst_soft = soft_worst_case_loss(
                    supervised_primary,
                    supervised_secondary,
                    tau=float(getattr(config, "soft_worst_case_tau", 0.1)),
                )
                task_losses = [
                    supervised_primary.float(),
                    supervised_secondary.float(),
                    supervised_mean.float(),
                    supervised_worst_soft.float(),
                    pred_consistency.float(),
                ]
                external_gradnorm_tasks = None
                config_full_info = None
                if use_learned_task_weighting:
                    learned_tasks = [
                        task_losses[2],
                        task_losses[3],
                    ]
                    if use_prediction_consistency:
                        learned_tasks.append((pred_consistency_weight * task_losses[4]).float())
                    uncertainty_info = uncertainty_balancer.combine(
                        learned_tasks,
                        epoch_idx=ep,
                    )
                    current_task_weights = uncertainty_info["weights"].float()
                    weighted_total_loss = uncertainty_info["total_loss"].float()
                    config_info = None
                    external_gradnorm_info = None
                elif use_config_layer:
                    config_tasks = [
                        task_losses[2],
                        task_losses[3],
                    ]
                    if use_prediction_consistency:
                        config_tasks.append((pred_consistency_weight * task_losses[4]).float())
                    external_gradnorm_tasks = config_tasks
                    current_task_weights = torch.tensor(config_task_weights, device=device, dtype=torch.float32)
                    weighted_total_loss = sum(
                        weight * task_loss for weight, task_loss in zip(config_task_weights, config_tasks)
                    ).float()
                    uncertainty_info = None
                    config_info = {"task_losses": config_tasks}
                    external_gradnorm_info = None
                elif use_config_full:
                    config_full_tasks = [
                        config_task_weights[0] * task_losses[2],
                        config_task_weights[1] * task_losses[3],
                    ]
                    if use_prediction_consistency:
                        config_full_tasks.append(
                            (config_task_weights[2] * pred_consistency_weight * task_losses[4]).float()
                        )
                    current_task_weights = torch.tensor(config_task_weights, device=device, dtype=torch.float32)
                    weighted_total_loss = sum(config_full_tasks).float()
                    uncertainty_info = None
                    config_info = None
                    config_full_info = {"task_losses": config_full_tasks}
                    external_gradnorm_info = None
                elif use_external_gradnorm:
                    external_gradnorm_tasks = [
                        task_losses[2],
                        task_losses[3],
                    ]
                    if use_prediction_consistency:
                        external_gradnorm_tasks.append((pred_consistency_weight * task_losses[4]).float())
                    current_task_weights = external_gradnorm.loss_weights.detach().float()
                    weighted_total_loss = sum(
                        weight * task_loss
                        for weight, task_loss in zip(current_task_weights, external_gradnorm_tasks)
                    ).float()
                    uncertainty_info = None
                    config_info = None
                    external_gradnorm_info = {"task_losses": external_gradnorm_tasks}
                elif use_fixed_sum:
                    fixed_sum_tasks = [
                        task_losses[2],
                        task_losses[3],
                    ]
                    if use_prediction_consistency:
                        fixed_sum_tasks.append((pred_consistency_weight * task_losses[4]).float())
                    current_task_weights = torch.tensor(config_task_weights, device=device, dtype=torch.float32)
                    weighted_total_loss = sum(
                        weight * task_loss for weight, task_loss in zip(config_task_weights, fixed_sum_tasks)
                    ).float()
                    uncertainty_info = None
                    config_info = None
                    external_gradnorm_info = None
                else:
                    uncertainty_info = None
                    config_info = None
                    external_gradnorm_info = None
                    current_task_weights = None
                    weighted_total_loss = (
                        supervised_mean
                        + (pred_consistency_weight * pred_consistency if use_prediction_consistency else 0.0)
                    ).float()

                config_direction = None
                config_diagnostics = None
                config_full_diagnostics = None
                if config_info is not None:
                    config_direction, config_diagnostics = config_layer_gradient(
                        [
                            weight * task_loss
                            for weight, task_loss in zip(config_task_weights, config_info["task_losses"])
                        ],
                        config_reference_parameter,
                        return_diagnostics=True,
                    )

                if not torch.isfinite(weighted_total_loss):
                    restore_model_buffers(model, buffer_snapshot)
                    raise FloatingPointError(
                        f"Non-finite consistency loss detected at epoch={ep} batch={batch_idx}: "
                        f"supervised_primary={float(supervised_primary.detach().item()):.6g}, "
                        f"supervised_secondary={float(supervised_secondary.detach().item()):.6g}, "
                        f"supervised_mean={float(supervised_mean.detach().item()):.6g}, "
                        f"supervised_worst_soft={float(supervised_worst_soft.detach().item()):.6g}, "
                        f"pred_consistency={float(pred_consistency.detach().item()):.6g}"
                    )

                external_gradnorm_loss = weighted_total_loss.new_zeros((), dtype=torch.float32)
                if use_config_full:
                    config_full_diagnostics = config_full_backward(
                        config_full_info["task_losses"],
                        config_full_parameters,
                        scaler=scaler,
                        amp_enabled=amp,
                        vectorized=bool(getattr(config, "config_full_vectorized_gradients", False)),
                        allow_sequential_fallback=bool(
                            getattr(config, "config_full_allow_sequential_fallback", True)
                        ),
                    )
                elif use_external_gradnorm:
                    external_gradnorm_loss = external_gradnorm_backward(
                        external_gradnorm,
                        external_gradnorm_tasks,
                        config_reference_parameter,
                        weighted_total_loss,
                        min_loss_weights=external_gradnorm_min_weights,
                        scaler=scaler,
                        amp_enabled=amp,
                    )
                elif amp:
                    scaler.scale(weighted_total_loss).backward()
                else:
                    weighted_total_loss.backward()

                if amp:
                    scaler.unscale_(optimizer)
                if config_direction is not None:
                    if config_reference_parameter.grad is None:
                        config_reference_parameter.grad = config_direction.clone()
                    else:
                        config_reference_parameter.grad.copy_(config_direction)
                gradient_stats = gradient_diagnostics(train_model) if track_gradient_diagnostics else None
                if gradient_stats is not None and not gradient_stats["finite"]:
                    print(
                        f"[warn] Non-finite gradients at epoch={ep} batch={batch_idx}; "
                        "skipping optimizer step and reducing AMP scale."
                    )
                    restore_model_buffers(model, buffer_snapshot)
                    optimizer.zero_grad(set_to_none=True)
                    if amp:
                        scaler.update(new_scale=max(1.0, 0.5 * scaler.get_scale()))
                    continue
                if gradient_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_norm)
                if amp:
                    scaler.step(optimizer)
                    if config_full_diagnostics is not None and float(config_full_diagnostics["used_fallback"].item()) > 0.0:
                        scaler.update(new_scale=max(1.0, 0.5 * scaler.get_scale()))
                    else:
                        scaler.update()
                else:
                    optimizer.step()
                if rollback_buffers and not model_buffers_are_finite(model):
                    restore_model_buffers(model, buffer_snapshot)
                    raise FloatingPointError(
                        f"Non-finite PTv3 state after optimizer step at epoch={ep} batch={batch_idx}."
                    )
                scheduler.step()

                total_loss = weighted_total_loss.detach()
                batch_size = surf_data.size(0)
                train_sample_count += batch_size
                secondary_inverse_density_fraction = float(
                    sum(mode.startswith("inverse_density") for mode in secondary_modes) / max(len(secondary_modes), 1)
                )
                primary_inverse_density_fraction = float(
                    sum(mode.startswith("inverse_density") for mode in primary_modes) / max(len(primary_modes), 1)
                )

                with torch.inference_mode():
                    surf_rel_primary = rel_l2_loss_fn(y1_surf_teacher, surf_data) if use_surface_supervision else torch.tensor(0.0, device=device)
                    vol_rel_primary = rel_l2_loss_fn(y1_vol_teacher, vol_data)
                    surf_rel_secondary = rel_l2_loss_fn(y2_surf, surf_data) if use_surface_supervision else torch.tensor(0.0, device=device)
                    vol_rel_secondary = rel_l2_loss_fn(y2_vol, vol_data)

                    surface_loss = 0.5 * (surf_rel_primary + surf_rel_secondary)
                    volume_loss = 0.5 * (vol_rel_primary + vol_rel_secondary)

                    batch_size_float = float(batch_size)
                    train_losses["loss"] += total_loss.detach().float() * batch_size_float
                    train_losses["rel_l2_surf"] += surface_loss.detach().float() * batch_size_float
                    train_losses["rel_l2_vol"] += volume_loss.detach().float() * batch_size_float
                    train_losses["rel_l2"] += (surface_loss + volume_loss).detach().float() * batch_size_float
                    train_losses["loss_supervised_primary"] += supervised_primary.detach().float() * batch_size_float
                    train_losses["loss_supervised_secondary"] += supervised_secondary.detach().float() * batch_size_float
                    train_losses["loss_supervised_mean"] += supervised_mean.detach().float() * batch_size_float
                    train_losses["loss_supervised_worst"] += supervised_worst.detach().float() * batch_size_float
                    train_losses["loss_supervised_worst_soft"] += supervised_worst_soft.detach().float() * batch_size_float
                    train_losses["loss_prediction_consistency"] += pred_consistency.detach().float() * batch_size_float
                    if gradient_stats is not None:
                        train_losses["gradient_norm_raw"] += gradient_stats["grad_norm"] * batch_size_float
                        train_losses["parameter_norm"] += gradient_stats["parameter_norm"] * batch_size_float
                        train_losses["gradient_max_abs"] += gradient_stats["max_grad"] * batch_size_float
                        train_losses["gradient_nonfinite"] += float(not gradient_stats["finite"]) * batch_size_float
                    train_losses["primary_inverse_density_fraction"] += torch.tensor(
                        primary_inverse_density_fraction, device=device, dtype=torch.float32
                    ) * batch_size_float
                    train_losses["primary_inverse_density_beta"] += torch.tensor(
                        float(primary_inverse_density_beta), device=device, dtype=torch.float32
                    ) * batch_size_float
                    train_losses["secondary_inverse_density_fraction"] += torch.tensor(
                        secondary_inverse_density_fraction, device=device, dtype=torch.float32
                    ) * batch_size_float
                    train_losses["secondary_inverse_density_beta"] += torch.tensor(
                        float(secondary_inverse_density_beta), device=device, dtype=torch.float32
                    ) * batch_size_float
                    if uncertainty_info is not None:
                        train_losses["learned_weight_supervised_mean"] += current_task_weights[0].detach().float() * batch_size_float
                        train_losses["learned_weight_supervised_worst"] += current_task_weights[1].detach().float() * batch_size_float
                        train_losses["learned_logit_supervised_mean"] += uncertainty_info["logits"][0].detach().float() * batch_size_float
                        train_losses["learned_logit_supervised_worst"] += uncertainty_info["logits"][1].detach().float() * batch_size_float
                        if use_prediction_consistency:
                            train_losses["learned_weight_prediction_consistency"] += current_task_weights[2].detach().float() * batch_size_float
                            train_losses["learned_logit_prediction_consistency"] += uncertainty_info["logits"][2].detach().float() * batch_size_float
                    if external_gradnorm_info is not None:
                        train_losses["external_gradnorm_weight_primary"] += current_task_weights[0].detach().float() * batch_size_float
                        train_losses["external_gradnorm_weight_secondary"] += current_task_weights[1].detach().float() * batch_size_float
                        if use_prediction_consistency:
                            train_losses["external_gradnorm_weight_prediction_consistency"] += current_task_weights[2].detach().float() * batch_size_float
                        train_losses["external_gradnorm_loss"] += external_gradnorm_loss.detach().float() * batch_size_float
                    if config_diagnostics is not None:
                        train_losses["config_reference_grad_norm"] += config_diagnostics["reference_grad_norm"] * batch_size_float
                        train_losses["config_direction_norm"] += config_diagnostics["direction_norm"] * batch_size_float
                        train_losses["config_direction_scale"] += config_diagnostics["direction_scale"] * batch_size_float
                        train_losses["config_direction_cosine"] += config_diagnostics["direction_cosine"] * batch_size_float
                        train_losses["config_direction_fallback"] += config_diagnostics["used_fallback"] * batch_size_float
                    if config_full_diagnostics is not None:
                        train_losses["config_full_mean_grad_norm"] += config_full_diagnostics["mean_grad_norm"] * batch_size_float
                        train_losses["config_full_direction_norm"] += config_full_diagnostics["direction_norm"] * batch_size_float
                        train_losses["config_full_min_cosine"] += config_full_diagnostics["min_cosine"] * batch_size_float
                        train_losses["config_full_fallback"] += config_full_diagnostics["used_fallback"] * batch_size_float

                    if use_surface_supervision:
                        pred_surf_primary = y1_surf_teacher * std_surf + mean_surf
                        pred_surf_secondary = y2_surf.detach() * std_surf + mean_surf
                        gt_surf = surf_data * std_surf + mean_surf
                        accumulate_channel_metrics_tensor(train_losses, "rel_l2_surf", pred_surf_primary, gt_surf, fields["surface"], rel_l2_loss_fn, batch_size, metric_weight=0.5)
                        accumulate_channel_metrics_tensor(train_losses, "rel_l2_surf", pred_surf_secondary, gt_surf, fields["surface"], rel_l2_loss_fn, batch_size, metric_weight=0.5)

                    pred_vol_primary = y1_vol_teacher * std_vol + mean_vol
                    pred_vol_secondary = y2_vol.detach() * std_vol + mean_vol
                    gt_vol = vol_data * std_vol + mean_vol
                    accumulate_channel_metrics_tensor(train_losses, "rel_l2_vol", pred_vol_primary, gt_vol, fields["volume"], rel_l2_loss_fn, batch_size, metric_weight=0.5)
                    accumulate_channel_metrics_tensor(train_losses, "rel_l2_vol", pred_vol_secondary, gt_vol, fields["volume"], rel_l2_loss_fn, batch_size, metric_weight=0.5)

                global_step += 1
                should_log = batch_idx % log_every_n_steps == 0 or batch_idx == len(train_loader) - 1
                if should_log and run is not None:
                    log_scalars = distributed_average_scalars(
                        [
                            total_loss.item(),
                            (surface_loss + volume_loss).item(),
                            surface_loss.item(),
                            volume_loss.item(),
                            supervised_primary.item(),
                            supervised_secondary.item(),
                            supervised_mean.item(),
                            supervised_worst_soft.item(),
                            pred_consistency.item(),
                            primary_inverse_density_fraction,
                            float(primary_inverse_density_beta),
                            secondary_inverse_density_fraction,
                            float(secondary_inverse_density_beta),
                        ]
                    )
                    wandb.log(
                        {
                            "train/batch_loss": log_scalars[0],
                            "train/batch_rel_l2": log_scalars[1],
                            "train/batch_rel_l2_surf": log_scalars[2],
                            "train/batch_rel_l2_vol": log_scalars[3],
                            "train/batch_supervised_primary": log_scalars[4],
                            "train/batch_supervised_secondary": log_scalars[5],
                            "train/batch_supervised_mean": log_scalars[6],
                            "train/batch_supervised_worst_soft": log_scalars[7],
                            "train/batch_prediction_consistency": log_scalars[8],
                            "train/batch_primary_inverse_density_fraction": log_scalars[9],
                            "train/batch_primary_inverse_density_beta": log_scalars[10],
                            "train/batch_secondary_inverse_density_fraction": log_scalars[11],
                            "train/batch_secondary_inverse_density_beta": log_scalars[12],
                            "train/prediction_consistency_weight": pred_consistency_weight,
                            "train/batch_amp_scale": float(scaler.get_scale()) if amp else 1.0,
                            "lr": scheduler.get_last_lr()[0],
                            "epoch": ep,
                        },
                        step=global_step,
                    )
                    if gradient_stats is not None:
                        wandb.log(
                            {
                                "train/batch_gradient_norm_raw": float(gradient_stats["grad_norm"].item()),
                                "train/batch_parameter_norm": float(gradient_stats["parameter_norm"].item()),
                                "train/batch_gradient_max_abs": float(gradient_stats["max_grad"].item()),
                                "train/batch_gradient_nonfinite": float(not gradient_stats["finite"]),
                            },
                            step=global_step,
                        )
                    if uncertainty_info is not None:
                        uncertainty_values = [
                            float(current_task_weights[0].item()),
                            float(current_task_weights[1].item()),
                            float(uncertainty_info["logits"][0].item()),
                            float(uncertainty_info["logits"][1].item()),
                        ]
                        if use_prediction_consistency:
                            uncertainty_values.extend(
                                [
                                    float(current_task_weights[2].item()),
                                    float(uncertainty_info["logits"][2].item()),
                                ]
                            )
                        uncertainty_log_scalars = distributed_average_scalars(uncertainty_values)
                        uncertainty_log_dict = {
                            "train/batch_learned_weight_supervised_mean": uncertainty_log_scalars[0],
                            "train/batch_learned_weight_supervised_worst": uncertainty_log_scalars[1],
                            "train/batch_learned_logit_supervised_mean": uncertainty_log_scalars[2],
                            "train/batch_learned_logit_supervised_worst": uncertainty_log_scalars[3],
                        }
                        if use_prediction_consistency:
                            uncertainty_log_dict["train/batch_learned_weight_prediction_consistency"] = uncertainty_log_scalars[4]
                            uncertainty_log_dict["train/batch_learned_logit_prediction_consistency"] = uncertainty_log_scalars[5]
                        wandb.log(uncertainty_log_dict, step=global_step)
                    if external_gradnorm_info is not None:
                        wandb.log(
                            {
                                "train/batch_external_gradnorm_loss": float(external_gradnorm_loss.item()),
                                "train/batch_external_gradnorm_weight_primary": float(current_task_weights[0].item()),
                                "train/batch_external_gradnorm_weight_secondary": float(current_task_weights[1].item()),
                                "train/batch_external_gradnorm_weight_prediction_consistency": float(current_task_weights[2].item()),
                            },
                            step=global_step,
                        )
                    if config_diagnostics is not None:
                        wandb.log(
                            {
                                "train/batch_config_reference_grad_norm": float(config_diagnostics["reference_grad_norm"].item()),
                                "train/batch_config_direction_norm": float(config_diagnostics["direction_norm"].item()),
                                "train/batch_config_direction_scale": float(config_diagnostics["direction_scale"].item()),
                                "train/batch_config_direction_cosine": float(config_diagnostics["direction_cosine"].item()),
                                "train/batch_config_direction_fallback": float(config_diagnostics["used_fallback"].item()),
                            },
                            step=global_step,
                        )
                    if config_full_diagnostics is not None:
                        wandb.log(
                            {
                                "train/batch_config_full_mean_grad_norm": float(config_full_diagnostics["mean_grad_norm"].item()),
                                "train/batch_config_full_direction_norm": float(config_full_diagnostics["direction_norm"].item()),
                                "train/batch_config_full_min_cosine": float(config_full_diagnostics["min_cosine"].item()),
                                "train/batch_config_full_fallback": float(config_full_diagnostics["used_fallback"].item()),
                                "train/batch_config_full_amp_scale": float(config_full_diagnostics["amp_scale"].item()),
                            },
                            step=global_step,
                        )
                if should_log:
                    train_pbar.set_postfix(loss=f"{total_loss.item():.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")
                    train_pbar.refresh()

            if is_dist_enabled():
                dist.barrier()
            if is_main_process():
                aligned_metrics = evaluate_loader(
                    model=model,
                    loader=test_loader,
                    config=config,
                    device=device,
                    dtype=dtype,
                    amp=amp,
                    params_dim=params_dim,
                    mean_surf=mean_surf,
                    std_surf=std_surf,
                    mean_vol=mean_vol,
                    std_vol=std_vol,
                    fields=fields,
                    rel_l2_loss_fn=rel_l2_loss_fn,
                    combined_loss_fn=combined_loss_fn,
                    loss_fn=loss_fn,
                    use_surface_supervision=use_surface_supervision,
                    mode_name=str(getattr(config, "eval_aligned_sampling_mode", "uniform_wor")),
                    num_view_points=eval_aligned_view_points,
                    eval_subsampled_geometry_points=eval_aligned_subsampled_geometry_points,
                    fixed_seed_offset=50000011,
                    model_requires_density=model_requires_density,
                    cuda_batch_prefetch=cuda_batch_prefetch,
                )
                shifted_metrics = evaluate_loader(
                    model=model,
                    loader=test_loader,
                    config=config,
                    device=device,
                    dtype=dtype,
                    amp=amp,
                    params_dim=params_dim,
                    mean_surf=mean_surf,
                    std_surf=std_surf,
                    mean_vol=mean_vol,
                    std_vol=std_vol,
                    fields=fields,
                    rel_l2_loss_fn=rel_l2_loss_fn,
                    combined_loss_fn=combined_loss_fn,
                    loss_fn=loss_fn,
                    use_surface_supervision=use_surface_supervision,
                    mode_name=str(getattr(config, "eval_shifted_sampling_mode", "inverse_density_wor")),
                    num_view_points=eval_shifted_view_points,
                    eval_subsampled_geometry_points=eval_shifted_subsampled_geometry_points,
                    fixed_seed_offset=70000029,
                    model_requires_density=model_requires_density,
                    cuda_batch_prefetch=cuda_batch_prefetch,
                )
            else:
                aligned_metrics = init_metric_dict(fields["surface"], fields["volume"])
                shifted_metrics = init_metric_dict(fields["surface"], fields["volume"])
            if is_dist_enabled():
                dist.barrier()

            if is_dist_enabled():
                key_list = list(train_losses.keys())
                metric_tensor = torch.tensor(
                    [float(train_losses[key].item()) for key in key_list] + [float(train_sample_count)],
                    device=device,
                    dtype=torch.float64,
                )
                dist.all_reduce(metric_tensor, op=dist.ReduceOp.SUM)
                train_sample_count = max(int(metric_tensor[-1].item()), 1)
                for idx, key in enumerate(key_list):
                    train_losses[key] = metric_tensor[idx].to(device=device, dtype=torch.float32)

            denom = max(int(train_sample_count), 1)
            for loss_name in train_losses.keys():
                train_losses[loss_name] = train_losses[loss_name] / float(denom)

            train_losses_log = {key: float(value.detach().cpu().item()) for key, value in train_losses.items()}

            robust_rel_l2 = 0.5 * (aligned_metrics["rel_l2"] + shifted_metrics["rel_l2"])
            robust_loss = 0.5 * (aligned_metrics["loss"] + shifted_metrics["loss"])
            checkpoint_extra_metrics = {
                "test_aligned_metrics": aligned_metrics,
                "test_shifted_metrics": shifted_metrics,
                "test_robust_rel_l2": robust_rel_l2,
            }
            if uncertainty_balancer is not None:
                checkpoint_extra_metrics["task_weighting_state_dict"] = uncertainty_balancer.state_dict()
            if external_gradnorm is not None:
                checkpoint_extra_metrics["external_gradnorm_state_dict"] = external_gradnorm.state_dict()

            if robust_rel_l2 < best_robust_rel_l2 and is_main_process():
                best_robust_rel_l2 = robust_rel_l2
                torch.save(
                    build_training_checkpoint(
                        epoch=ep,
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scaler,
                        loss=robust_loss,
                        rel_l2_loss=robust_rel_l2,
                        surface_fields=fields["surface"],
                        volume_fields=fields["volume"],
                        global_step=global_step,
                        best_robust_rel_l2=best_robust_rel_l2,
                        extra_metrics=checkpoint_extra_metrics,
                    ),
                    "checkpoints/" + model_checkpoint_name + "_best.pt",
                )

            if is_main_process():
                torch.save(
                    build_training_checkpoint(
                        epoch=ep,
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scaler,
                        loss=robust_loss,
                        rel_l2_loss=robust_rel_l2,
                        surface_fields=fields["surface"],
                        volume_fields=fields["volume"],
                        global_step=global_step,
                        best_robust_rel_l2=best_robust_rel_l2,
                        extra_metrics=checkpoint_extra_metrics,
                    ),
                    "checkpoints/" + model_checkpoint_name + "_last.pt",
                )

            t2 = default_timer()
            if is_main_process():
                print(
                    f"epoch: {ep}, t2-t1 (epoch time): {t2 - t1:.5f}, "
                    f"train loss: {train_losses_log['loss']:.5f}, "
                    f"aligned rel_l2: {aligned_metrics['rel_l2']:.5f}, "
                    f"shifted rel_l2: {shifted_metrics['rel_l2']:.5f}, "
                    f"robust rel_l2: {robust_rel_l2:.5f}"
                )

            if run is not None:
                wandb_dict = {
                    "lr": scheduler.get_last_lr()[0],
                    "test/robust_rel_l2": robust_rel_l2,
                    "test/robust_loss": robust_loss,
                    "train/prediction_consistency_weight": pred_consistency_weight,
                }
                wandb_dict.update({f"train/{key}": value for key, value in train_losses_log.items()})
                wandb_dict.update({f"test_aligned/{key}": value for key, value in aligned_metrics.items()})
                wandb_dict.update({f"test_shifted/{key}": value for key, value in shifted_metrics.items()})
                add_all_field_metrics(wandb_dict, "train", fields["surface"], fields["volume"], metric_values=train_losses_log)
                add_all_field_metrics(wandb_dict, "test_aligned", fields["surface"], fields["volume"], metric_values=aligned_metrics)
                add_all_field_metrics(wandb_dict, "test_shifted", fields["surface"], fields["volume"], metric_values=shifted_metrics)
                add_canonical_field_metrics(wandb_dict, "train", fields["surface"], fields["volume"], metric_values=train_losses_log)
                add_canonical_field_metrics(wandb_dict, "test_aligned", fields["surface"], fields["volume"], metric_values=aligned_metrics)
                add_canonical_field_metrics(wandb_dict, "test_shifted", fields["surface"], fields["volume"], metric_values=shifted_metrics)
                wandb_dict["meta/training_surface_signals"] = ",".join(fields["surface"])
                wandb_dict["meta/training_volume_signals"] = ",".join(fields["volume"])
                wandb.log(wandb_dict, step=global_step)

    except KeyboardInterrupt:
        if is_main_process():
            print("\nTraining interrupted by user (Ctrl+C). Saving current state and exiting cleanly...")
        try:
            emergency_state = {
                "epoch": locals().get("ep", -1),
                "global_step": global_step,
                "best_robust_rel_l2": best_robust_rel_l2,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
            }
            if uncertainty_balancer is not None:
                emergency_state["task_weighting_state_dict"] = uncertainty_balancer.state_dict()
            if external_gradnorm is not None:
                emergency_state["external_gradnorm_state_dict"] = external_gradnorm.state_dict()
            if is_main_process():
                torch.save(emergency_state, "checkpoints/" + model_checkpoint_name + "_last.pt")
                print("Saved the latest checkpoint before exiting.")
        except Exception as exc:
            if is_main_process():
                print(f"Could not save an emergency checkpoint: {exc}")
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
        cleanup_distributed()
