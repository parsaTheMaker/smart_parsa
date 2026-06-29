from __future__ import annotations

import os
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
    print_point_budget,
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
    gradnorm_balancer=None,
    gradnorm_optimizer=None,
    uncertainty_balancer=None,
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

    if gradnorm_balancer is not None:
        gradnorm_state = checkpoint.get("gradnorm_balancer_state_dict")
        if gradnorm_state is None:
            raise KeyError(f"Checkpoint {checkpoint_path} is missing gradnorm_balancer_state_dict required for full-state resume.")
        gradnorm_incompat = gradnorm_balancer.load_state_dict(gradnorm_state, strict=False)
        missing_keys = getattr(gradnorm_incompat, "missing_keys", [])
        unexpected_keys = getattr(gradnorm_incompat, "unexpected_keys", [])
        if missing_keys or unexpected_keys:
            print(
                f"[resume] Loaded gradnorm balancer from {checkpoint_path} with "
                f"missing_keys={missing_keys} unexpected_keys={unexpected_keys}"
            )
    if gradnorm_optimizer is not None:
        gradnorm_opt_state = checkpoint.get("gradnorm_optimizer_state_dict")
        if gradnorm_opt_state is None:
            raise KeyError(f"Checkpoint {checkpoint_path} is missing gradnorm_optimizer_state_dict required for full-state resume.")
        gradnorm_optimizer.load_state_dict(gradnorm_opt_state)
    if uncertainty_balancer is not None:
        uncertainty_state = checkpoint.get("task_weighting_state_dict")
        if uncertainty_state is None:
            uncertainty_state = checkpoint.get("uncertainty_balancer_state_dict")
        if uncertainty_state is None:
            raise KeyError(
                f"Checkpoint {checkpoint_path} is missing task_weighting_state_dict required for full-state resume."
            )
        uncertainty_balancer.load_state_dict(uncertainty_state, strict=True)

    resumed_epoch = int(checkpoint.get("epoch", -1))
    start_epoch = resumed_epoch + 1
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


def _sample_single_view_indices(log_density_row, num_points, mode, inverse_density_beta, mixed_inverse_density_prob, generator):
    n_points = int(log_density_row.shape[0])
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

    raise ValueError(f"Unsupported sampling mode: {resolved_mode}")


def sample_geometry_view(geo_mesh, geo_log_density, num_points, mode, inverse_density_beta, mixed_inverse_density_prob, seed):
    if geo_log_density is None:
        raise RuntimeError("Consistency training requires geometry log density for view sampling.")

    idx_rows = []
    resolved_modes = []
    batch_size = int(geo_mesh.shape[0])
    for batch_idx in range(batch_size):
        generator = _cpu_generator(seed + 1009 * batch_idx)
        idx_row, resolved_mode = _sample_single_view_indices(
            geo_log_density[batch_idx],
            num_points=num_points,
            mode=mode,
            inverse_density_beta=inverse_density_beta,
            mixed_inverse_density_prob=mixed_inverse_density_prob,
            generator=generator,
        )
        idx_rows.append(idx_row)
        resolved_modes.append(resolved_mode)

    idx = torch.stack(idx_rows, dim=0)
    return gather_points(geo_mesh, idx), gather_scalar(geo_log_density, idx), resolved_modes


class GradNormBalancer(torch.nn.Module):
    def __init__(
        self,
        num_tasks,
        alpha=1.5,
        update_interval=1,
        clamp_start=0.2,
        clamp_end=2.0,
        clamp_warmup_epochs=50,
    ):
        super().__init__()
        self.num_tasks = int(num_tasks)
        self.alpha = float(alpha)
        self.update_interval = max(1, int(update_interval))
        self.clamp_start = float(clamp_start)
        self.clamp_end = float(clamp_end)
        self.clamp_warmup_epochs = max(0, int(clamp_warmup_epochs))
        self.log_task_weights = torch.nn.Parameter(torch.zeros(self.num_tasks, dtype=torch.float32))
        self.register_buffer("initial_losses", torch.zeros(self.num_tasks, dtype=torch.float32))
        self.register_buffer("initialized", torch.tensor(False, dtype=torch.bool))
        self.register_buffer("last_update_step", torch.tensor(-1, dtype=torch.long))

    def clamp_for_epoch(self, epoch_idx=None):
        start = max(0.0, float(self.clamp_start))
        end = max(start, float(self.clamp_end))
        if epoch_idx is None or self.clamp_warmup_epochs <= 0:
            return end
        if self.clamp_warmup_epochs == 1:
            return end
        progress = min(max(int(epoch_idx), 0), self.clamp_warmup_epochs - 1) / float(self.clamp_warmup_epochs - 1)
        return start + (end - start) * progress

    def normalized_weights(self, epoch_idx=None):
        clamp_mag = self.clamp_for_epoch(epoch_idx)
        logits = torch.clamp(self.log_task_weights, min=-clamp_mag, max=clamp_mag)
        weights = torch.nan_to_num(
            torch.softmax(logits, dim=0),
            nan=1.0 / max(self.num_tasks, 1),
            posinf=1.0,
            neginf=1.0,
        )
        return weights / torch.clamp(weights.sum(), min=1e-8)

    def maybe_initialize(self, losses):
        if bool(self.initialized.item()):
            return
        init_losses = torch.stack([loss.detach().float().clamp_min(1e-8) for loss in losses])
        self.initial_losses.copy_(init_losses)
        self.initialized.fill_(True)

    def compute(
        self,
        losses,
        reference_params,
        epoch_idx=None,
        step_idx=None,
    ):
        if len(losses) != self.num_tasks:
            raise ValueError(f"GradNorm expected {self.num_tasks} losses, got {len(losses)}.")
        if len(reference_params) == 0:
            raise ValueError("GradNorm requires at least one reference parameter.")

        self.maybe_initialize(losses)
        weights = self.normalized_weights(epoch_idx=epoch_idx)
        should_update = True
        if step_idx is not None:
            should_update = (int(step_idx) % self.update_interval) == 0

        loss_ratios = torch.stack([loss.detach().float().clamp_min(1e-8) for loss in losses]) / torch.clamp(self.initial_losses, min=1e-8)
        loss_ratios = torch.nan_to_num(loss_ratios, nan=1.0, posinf=1.0, neginf=1.0)

        if should_update:
            base_grad_norms = []
            for loss in losses:
                grads = torch.autograd.grad(
                    loss,
                    reference_params,
                    retain_graph=True,
                    create_graph=False,
                    allow_unused=True,
                )
                grad_sq_sum = None
                for grad in grads:
                    if grad is None:
                        continue
                    contrib = grad.float().pow(2).sum()
                    grad_sq_sum = contrib if grad_sq_sum is None else (grad_sq_sum + contrib)
                if grad_sq_sum is None:
                    base_grad_norms.append(loss.new_tensor(0.0, dtype=torch.float32))
                else:
                    base_grad_norms.append(torch.sqrt(torch.clamp(grad_sq_sum, min=1e-12)).detach())
            base_grad_norms = torch.stack(base_grad_norms)
            base_grad_norms = torch.nan_to_num(base_grad_norms, nan=0.0, posinf=1e6, neginf=0.0)
            self.last_update_step.fill_(int(step_idx) if step_idx is not None else -1)
            inverse_train_rates = loss_ratios / torch.clamp(loss_ratios.mean(), min=1e-8)
            grad_norms = weights * base_grad_norms
            mean_grad_norm = grad_norms.detach().mean()
            targets = mean_grad_norm * inverse_train_rates.pow(self.alpha)
            gradnorm_loss = torch.abs(grad_norms - targets).sum()
            gradnorm_loss = torch.nan_to_num(gradnorm_loss, nan=0.0, posinf=1e6, neginf=0.0)
        else:
            base_grad_norms = torch.zeros_like(loss_ratios)
            inverse_train_rates = (loss_ratios / torch.clamp(loss_ratios.mean(), min=1e-8)).detach()
            grad_norms = torch.zeros_like(loss_ratios)
            gradnorm_loss = losses[0].detach().new_zeros(())

        return {
            "weights": weights,
            "weights_detached": weights.detach(),
            "gradnorm_loss": gradnorm_loss,
            "clamp_mag": torch.tensor(float(self.clamp_for_epoch(epoch_idx)), device=weights.device),
            "base_grad_norms": base_grad_norms,
            "grad_norms": grad_norms.detach(),
            "inverse_train_rates": inverse_train_rates.detach(),
            "should_update": should_update,
        }


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
    fixed_seed_offset,
    model_requires_density,
):
    metrics = init_metric_dict(fields["surface"], fields["volume"])
    model.eval()
    sample_count = 0

    pbar = tqdm(loader, desc=f"Eval {mode_name}", leave=False, dynamic_ncols=True, disable=not is_main_process())
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
            view_log_density = view_log_density.to(device, non_blocking=True)

            with torch.autocast(device_type=str(device).split(":")[0], dtype=dtype, enabled=amp):
                if model_requires_density:
                    y_hat_surf, y_hat_vol = model.inference(view_geo, surf_mesh, vol_mesh, params, geo_log_density=view_log_density)
                else:
                    y_hat_surf, y_hat_vol = model.inference(view_geo, surf_mesh, vol_mesh, params)

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
            f"persistent_workers={dl_common.get('persistent_workers', False)}"
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

    if is_main_process():
        print(f"Total parameters: {count_model_params(model)}")
    model_checkpoint_name = get_model_checkpoint_name(config)
    if is_main_process():
        print(f"Checkpoint name: {model_checkpoint_name}")
    if run is not None and bool(getattr(config, "wandb_watch_model", False)):
        run.watch(model, log="all")

    use_gradnorm = bool(getattr(config, "use_gradnorm", False))
    use_learned_task_weighting = bool(
        getattr(config, "use_learned_task_weighting", getattr(config, "use_uncertainty_weighting", False))
    )
    if use_gradnorm and use_learned_task_weighting:
        raise ValueError("use_gradnorm and use_learned_task_weighting cannot both be enabled.")
    scaler = torch.amp.GradScaler("cuda")
    gradnorm_balancer = None
    gradnorm_optimizer = None
    gradnorm_reference_params = ()
    uncertainty_balancer = None
    extra_optimizer_param_groups = []
    if use_gradnorm:
        gradnorm_balancer = GradNormBalancer(
            num_tasks=3,
            alpha=float(getattr(config, "gradnorm_alpha", 1.5)),
            update_interval=int(getattr(config, "gradnorm_update_interval", 1)),
            clamp_start=float(getattr(config, "gradnorm_clamp_start", 0.2)),
            clamp_end=float(getattr(config, "gradnorm_clamp_end", 2.0)),
            clamp_warmup_epochs=int(getattr(config, "gradnorm_clamp_warmup_epochs", 50)),
        ).to(device)
        gradnorm_optimizer = torch.optim.Adam(
            gradnorm_balancer.parameters(),
            lr=float(getattr(config, "gradnorm_lr", 2.5e-2)),
        )
        gradnorm_reference_params = resolve_gradnorm_reference_params(model)
    if use_learned_task_weighting:
        uncertainty_balancer = LearnedTaskWeighting(
            task_names=("supervised_mean", "supervised_worst", "prediction_consistency"),
            init_logits=list(getattr(config, "task_weight_init_logits", [0.0, 0.0, 0.0])),
            min_logit=float(getattr(config, "task_weight_logit_min", -4.0)),
            max_logit=float(getattr(config, "task_weight_logit_max", 4.0)),
            min_weights=list(getattr(config, "task_weight_min_weights", [0.4, 0.4, 0.0])),
            base_weights=list(getattr(config, "task_weight_base_weights", [0.45, 0.45, 0.10])),
            warmup_epochs=int(getattr(config, "task_weight_warmup_epochs", 25)),
        ).to(device)
        extra_optimizer_param_groups.append(
            {
                "params": list(uncertainty_balancer.parameters()),
                "lr": float(getattr(config, "task_weight_lr", 1.0e-3)),
                "weight_decay": 0.0,
            }
        )
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
            load_scaler=bool(amp),
            gradnorm_balancer=gradnorm_balancer,
            gradnorm_optimizer=gradnorm_optimizer,
            uncertainty_balancer=uncertainty_balancer,
        )
    elif init_ckpt:
        if is_main_process():
            print(f"[init] Loading model weights from {init_ckpt}")
        load_partial_state_dict(model, init_ckpt, device)
    elif resume_ckpt:
        if is_main_process():
            print(f"[init] experiment.resume_ckpt is being used for partial model initialization from {resume_ckpt}")
        load_partial_state_dict(model, resume_ckpt, device)

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
        "secondary_inverse_density_fraction",
        "secondary_inverse_density_beta",
        "gradnorm_weight_primary",
        "gradnorm_weight_secondary",
        "gradnorm_weight_prediction_consistency",
        "gradnorm_loss",
        "learned_weight_supervised_mean",
        "learned_weight_supervised_worst",
        "learned_weight_prediction_consistency",
        "learned_logit_supervised_mean",
        "learned_logit_supervised_worst",
        "learned_logit_prediction_consistency",
    ]

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
            train_pbar = tqdm(
                train_loader,
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
            use_latent_consistency = bool(getattr(config, "use_latent_consistency", False))

            for batch_idx, batch in enumerate(train_pbar):
                geo_mesh, surf_mesh, surf_data, vol_mesh, vol_data, params, geo_log_density = unpack_batch(batch, params_dim)
                if geo_log_density is None:
                    raise RuntimeError(
                        f"{config.model_name} consistency training requires geometry log density from the dataset "
                        "for geometry-view resampling."
                    )

                primary_view_geo, primary_view_density, _ = sample_geometry_view(
                    geo_mesh,
                    geo_log_density,
                    num_points=int(getattr(config, "view_geometry_points", 0)),
                    mode=str(getattr(config, "train_primary_sampling_mode", "uniform_wor")),
                    inverse_density_beta=float(getattr(config, "inverse_density_beta", 1.0)),
                    mixed_inverse_density_prob=float(getattr(config, "mixed_inverse_density_prob", 0.5)),
                    seed=int(config.random_seed + ep * 1000003 + batch_idx * 10007 + 11),
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
                    num_points=int(getattr(config, "view_geometry_points", 0)),
                    mode=str(getattr(config, "train_secondary_sampling_mode", "mixed")),
                    inverse_density_beta=secondary_inverse_density_beta,
                    mixed_inverse_density_prob=float(getattr(config, "mixed_inverse_density_prob", 0.5)),
                    seed=int(config.random_seed + ep * 1000003 + batch_idx * 10007 + 29),
                )

                params = move_optional_tensor(params, device)
                surf_mesh = surf_mesh.to(device, non_blocking=True)
                surf_data = surf_data.to(device, non_blocking=True)
                vol_mesh = vol_mesh.to(device, non_blocking=True)
                vol_data = vol_data.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                if gradnorm_optimizer is not None:
                    gradnorm_optimizer.zero_grad(set_to_none=True)

                primary_view_geo = primary_view_geo.to(device, non_blocking=True)
                if model_requires_density:
                    primary_view_density = primary_view_density.to(device, non_blocking=True)
                secondary_view_geo = secondary_view_geo.to(device, non_blocking=True)
                if model_requires_density:
                    secondary_view_density = secondary_view_density.to(device, non_blocking=True)

                fuse_consistency_views = bool(getattr(config, "fuse_consistency_views", False))
                with torch.autocast(device_type=str(device).split(":")[0], dtype=dtype, enabled=amp):
                    if fuse_consistency_views:
                        fused_geo = torch.cat([primary_view_geo, secondary_view_geo], dim=0)
                        fused_surf_mesh = duplicate_batch_tensor(surf_mesh)
                        fused_vol_mesh = duplicate_batch_tensor(vol_mesh)
                        fused_params = duplicate_batch_tensor(params)
                        if model_requires_density:
                            fused_density = torch.cat([primary_view_density, secondary_view_density], dim=0)
                            if use_latent_consistency:
                                fused_out = train_model(
                                    fused_geo,
                                    fused_surf_mesh,
                                    fused_vol_mesh,
                                    fused_params,
                                    geo_log_density=fused_density,
                                    return_latent=True,
                                )
                            else:
                                fused_out = train_model(
                                    fused_geo,
                                    fused_surf_mesh,
                                    fused_vol_mesh,
                                    fused_params,
                                    geo_log_density=fused_density,
                                )
                        else:
                            if use_latent_consistency:
                                fused_out = train_model(
                                    fused_geo,
                                    fused_surf_mesh,
                                    fused_vol_mesh,
                                    fused_params,
                                    return_latent=True,
                                )
                            else:
                                fused_out = train_model(
                                    fused_geo,
                                    fused_surf_mesh,
                                    fused_vol_mesh,
                                    fused_params,
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
                        if model_requires_density:
                            if use_latent_consistency:
                                primary_out = train_model(
                                    primary_view_geo,
                                    surf_mesh,
                                    vol_mesh,
                                    params,
                                    geo_log_density=primary_view_density,
                                    return_latent=True,
                                )
                            else:
                                primary_out = train_model(
                                    primary_view_geo,
                                    surf_mesh,
                                    vol_mesh,
                                    params,
                                    geo_log_density=primary_view_density,
                                )
                        else:
                            if use_latent_consistency:
                                primary_out = train_model(
                                    primary_view_geo,
                                    surf_mesh,
                                    vol_mesh,
                                    params,
                                    return_latent=True,
                                )
                            else:
                                primary_out = train_model(
                                    primary_view_geo,
                                    surf_mesh,
                                    vol_mesh,
                                    params,
                                )
                        if use_latent_consistency:
                            y1_surf, y1_vol, latent1 = primary_out
                        else:
                            y1_surf, y1_vol = primary_out
                            latent1 = None
                        if model_requires_density:
                            if use_latent_consistency:
                                secondary_out = train_model(
                                    secondary_view_geo,
                                    surf_mesh,
                                    vol_mesh,
                                    params,
                                    geo_log_density=secondary_view_density,
                                    return_latent=True,
                                )
                            else:
                                secondary_out = train_model(
                                    secondary_view_geo,
                                    surf_mesh,
                                    vol_mesh,
                                    params,
                                    geo_log_density=secondary_view_density,
                                )
                        else:
                            if use_latent_consistency:
                                secondary_out = train_model(
                                    secondary_view_geo,
                                    surf_mesh,
                                    vol_mesh,
                                    params,
                                    return_latent=True,
                                )
                            else:
                                secondary_out = train_model(
                                    secondary_view_geo,
                                    surf_mesh,
                                    vol_mesh,
                                    params,
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
                pred_consistency = prediction_consistency_smooth_l1_loss(
                    y1_surf_teacher,
                    y1_vol_teacher,
                    y2_surf_f,
                    y2_vol_f,
                    beta=float(getattr(config, "prediction_consistency_smooth_l1_beta", 0.05)),
                )
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
                if use_gradnorm:
                    gradnorm_info = gradnorm_balancer.compute(
                        [
                            task_losses[0],
                            task_losses[1],
                            (pred_consistency_weight * task_losses[4]).float(),
                        ],
                        gradnorm_reference_params,
                        epoch_idx=ep,
                        step_idx=global_step,
                    )
                    current_task_weights = gradnorm_info["weights_detached"].float()
                    gradnorm_should_update = bool(gradnorm_info.get("should_update", True))
                    weighted_total_loss = sum(
                        current_task_weights[i] * task_loss
                        for i, task_loss in enumerate(
                            [
                                task_losses[0],
                                task_losses[1],
                                (pred_consistency_weight * task_losses[4]).float(),
                            ]
                        )
                    )
                    uncertainty_info = None
                elif use_learned_task_weighting:
                    uncertainty_info = uncertainty_balancer.combine(
                        [
                            task_losses[2],
                            task_losses[3],
                            (pred_consistency_weight * task_losses[4]).float(),
                        ],
                        epoch_idx=ep,
                    )
                    current_task_weights = uncertainty_info["weights"].float()
                    gradnorm_should_update = False
                    gradnorm_info = None
                    weighted_total_loss = uncertainty_info["total_loss"].float()
                else:
                    gradnorm_info = None
                    uncertainty_info = None
                    current_task_weights = None
                    gradnorm_should_update = False
                    weighted_total_loss = (
                        supervised_mean
                        + pred_consistency_weight * pred_consistency
                    ).float()

                if gradnorm_should_update:
                    gradnorm_grads = torch.autograd.grad(
                        gradnorm_info["gradnorm_loss"],
                        tuple(gradnorm_balancer.parameters()),
                        retain_graph=True,
                        allow_unused=False,
                    )
                    for param, grad in zip(gradnorm_balancer.parameters(), gradnorm_grads):
                        param.grad = grad.detach()

                if not torch.isfinite(weighted_total_loss):
                    raise FloatingPointError(
                        f"Non-finite SATLOSS5 loss detected at epoch={ep} batch={batch_idx}: "
                        f"supervised_primary={float(supervised_primary.detach().item()):.6g}, "
                        f"supervised_secondary={float(supervised_secondary.detach().item()):.6g}, "
                        f"supervised_mean={float(supervised_mean.detach().item()):.6g}, "
                        f"supervised_worst_soft={float(supervised_worst_soft.detach().item()):.6g}, "
                        f"pred_consistency={float(pred_consistency.detach().item()):.6g}"
                    )

                if amp:
                    scaler.scale(weighted_total_loss).backward()
                    if gradient_norm is not None:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    weighted_total_loss.backward()
                    if gradient_norm is not None:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_norm)
                    optimizer.step()
                if gradnorm_optimizer is not None and gradnorm_should_update:
                    gradnorm_optimizer.step()
                    with torch.no_grad():
                        clamp_mag = float(gradnorm_balancer.clamp_for_epoch(ep))
                        gradnorm_balancer.log_task_weights.clamp_(-clamp_mag, clamp_mag)
                scheduler.step()

                total_loss = weighted_total_loss.detach()
                batch_size = surf_data.size(0)
                train_sample_count += batch_size
                secondary_inverse_density_fraction = float(
                    sum(mode.startswith("inverse_density") for mode in secondary_modes) / max(len(secondary_modes), 1)
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
                    train_losses["secondary_inverse_density_fraction"] += torch.tensor(
                        secondary_inverse_density_fraction, device=device, dtype=torch.float32
                    ) * batch_size_float
                    train_losses["secondary_inverse_density_beta"] += torch.tensor(
                        float(secondary_inverse_density_beta), device=device, dtype=torch.float32
                    ) * batch_size_float
                    if gradnorm_info is not None:
                        train_losses["gradnorm_weight_primary"] += current_task_weights[0].detach().float() * batch_size_float
                        train_losses["gradnorm_weight_secondary"] += current_task_weights[1].detach().float() * batch_size_float
                        train_losses["gradnorm_weight_prediction_consistency"] += current_task_weights[2].detach().float() * batch_size_float
                    if gradnorm_info is not None:
                        train_losses["gradnorm_loss"] += gradnorm_info["gradnorm_loss"].detach().float() * batch_size_float
                    if uncertainty_info is not None:
                        train_losses["learned_weight_supervised_mean"] += current_task_weights[0].detach().float() * batch_size_float
                        train_losses["learned_weight_supervised_worst"] += current_task_weights[1].detach().float() * batch_size_float
                        train_losses["learned_weight_prediction_consistency"] += current_task_weights[2].detach().float() * batch_size_float
                        train_losses["learned_logit_supervised_mean"] += uncertainty_info["logits"][0].detach().float() * batch_size_float
                        train_losses["learned_logit_supervised_worst"] += uncertainty_info["logits"][1].detach().float() * batch_size_float
                        train_losses["learned_logit_prediction_consistency"] += uncertainty_info["logits"][2].detach().float() * batch_size_float

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
                            "train/batch_secondary_inverse_density_fraction": log_scalars[9],
                            "train/batch_secondary_inverse_density_beta": log_scalars[10],
                            "train/prediction_consistency_weight": pred_consistency_weight,
                            "lr": scheduler.get_last_lr()[0],
                            "epoch": ep,
                        },
                        step=global_step,
                    )
                    if gradnorm_info is not None:
                        gradnorm_weight_log_scalars = distributed_average_scalars(
                            [
                                float(current_task_weights[0].item()),
                                float(current_task_weights[1].item()),
                                float(current_task_weights[2].item()),
                            ]
                        )
                        wandb.log(
                            {
                                "train/batch_gradnorm_weight_primary": gradnorm_weight_log_scalars[0],
                                "train/batch_gradnorm_weight_secondary": gradnorm_weight_log_scalars[1],
                                "train/batch_gradnorm_weight_prediction_consistency": gradnorm_weight_log_scalars[2],
                            },
                            step=global_step,
                        )
                    if gradnorm_info is not None:
                        gradnorm_log_scalars = distributed_average_scalars(
                            [
                                float(gradnorm_info["gradnorm_loss"].detach().item()),
                            ]
                        )
                        wandb.log(
                            {
                                "train/batch_gradnorm_loss": gradnorm_log_scalars[0],
                            },
                            step=global_step,
                        )
                    if uncertainty_info is not None:
                        uncertainty_log_scalars = distributed_average_scalars(
                            [
                                float(current_task_weights[0].item()),
                                float(current_task_weights[1].item()),
                                float(current_task_weights[2].item()),
                                float(uncertainty_info["logits"][0].item()),
                                float(uncertainty_info["logits"][1].item()),
                                float(uncertainty_info["logits"][2].item()),
                            ]
                        )
                        wandb.log(
                            {
                                "train/batch_learned_weight_supervised_mean": uncertainty_log_scalars[0],
                                "train/batch_learned_weight_supervised_worst": uncertainty_log_scalars[1],
                                "train/batch_learned_weight_prediction_consistency": uncertainty_log_scalars[2],
                                "train/batch_learned_logit_supervised_mean": uncertainty_log_scalars[3],
                                "train/batch_learned_logit_supervised_worst": uncertainty_log_scalars[4],
                                "train/batch_learned_logit_prediction_consistency": uncertainty_log_scalars[5],
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
                    num_view_points=int(getattr(config, "eval_view_geometry_points", getattr(config, "view_geometry_points", 0))),
                    fixed_seed_offset=50000011,
                    model_requires_density=model_requires_density,
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
                    num_view_points=int(getattr(config, "eval_view_geometry_points", getattr(config, "view_geometry_points", 0))),
                    fixed_seed_offset=70000029,
                    model_requires_density=model_requires_density,
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
            if gradnorm_balancer is not None and gradnorm_optimizer is not None:
                checkpoint_extra_metrics.update(
                    {
                        "gradnorm_balancer_state_dict": gradnorm_balancer.state_dict(),
                        "gradnorm_optimizer_state_dict": gradnorm_optimizer.state_dict(),
                    }
                )
            if uncertainty_balancer is not None:
                checkpoint_extra_metrics["task_weighting_state_dict"] = uncertainty_balancer.state_dict()

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
            if gradnorm_balancer is not None and gradnorm_optimizer is not None:
                emergency_state["gradnorm_balancer_state_dict"] = gradnorm_balancer.state_dict()
                emergency_state["gradnorm_optimizer_state_dict"] = gradnorm_optimizer.state_dict()
            if uncertainty_balancer is not None:
                emergency_state["task_weighting_state_dict"] = uncertainty_balancer.state_dict()
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
