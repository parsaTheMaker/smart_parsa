import os
from collections import OrderedDict
from timeit import default_timer

import hydra
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from omegaconf import DictConfig
from torch.nn import DataParallel
from tqdm.auto import tqdm

from data.datasets import get_dataset
from loss.losses import CombinedLoss
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
    get_optimizer_scheduler_loss,
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


def consistency_warmup_factor(epoch, warmup_epochs):
    warmup_epochs = int(warmup_epochs)
    if warmup_epochs <= 0:
        return 1.0
    return min(1.0, float(epoch + 1) / float(warmup_epochs))


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


def masked_smooth_l1(pred, target, weight, beta=0.05):
    loss = F.smooth_l1_loss(pred, target, beta=beta, reduction="none")
    return masked_mean(loss, weight)


def masked_group_sparsity(x, weight, eps=1.0e-8):
    norms = torch.sqrt(x.float().pow(2).sum(dim=-1) + eps)
    return masked_mean(norms, weight)


def relative_locality_penalty(spill_value, anchor_value, target_ratio, eps=1.0e-6):
    target = anchor_value.detach() * float(target_ratio)
    return F.relu(spill_value - target) / (anchor_value.detach() + eps)


def masked_response_norm(pred, target, weight, eps=1.0e-8):
    delta = pred.float() - target.float()
    response = torch.sqrt(delta.pow(2).mean(dim=-1, keepdim=True) + eps)
    return masked_mean(response, weight)


def apply_gaussian_deformation(points, center, normal, sigma, edit_scale, ramp_power):
    delta = points - center.unsqueeze(1)
    dist2 = delta.float().pow(2).sum(dim=-1, keepdim=True)
    sigma2 = sigma.float().unsqueeze(1).unsqueeze(-1).pow(2).clamp_min(1.0e-6)
    weight = torch.exp(-0.5 * dist2 / sigma2).to(dtype=points.dtype)
    weight = weight.pow(float(ramp_power))
    if torch.is_tensor(edit_scale):
        scale = edit_scale.to(device=points.device, dtype=points.dtype).unsqueeze(1).unsqueeze(-1)
    else:
        scale = points.new_tensor(float(edit_scale))
    displacement = scale * weight * normal.unsqueeze(1)
    return points + displacement


def build_local_edit(
    geo_mesh,
    min_points,
    max_points,
    edit_strength,
    ramp_power,
    changed_scale,
    unchanged_scale,
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

    signed = (centered * normal.unsqueeze(1)).sum(dim=-1)
    signed_rms = torch.sqrt(
        ((signed.float().pow(2) * patch_weight.float()).sum(dim=1) / weight_sum.squeeze(1).float()).clamp_min(1.0e-8)
    ).to(dtype=geo_mesh.dtype)
    edit_sigma = patch_radius.clamp_min(1.0e-6)
    # Use a small signed normal bump/dent so the edit stays local and remains
    # meaningful even on nearly planar patches where signed_rms is tiny.
    amplitude_floor = 0.05 * patch_radius
    amplitude_cap = 0.20 * patch_radius
    amplitude = signed_rms.clamp_min(amplitude_floor)
    amplitude = torch.minimum(amplitude, amplitude_cap)
    direction = torch.where(
        torch.rand(batch_size, device=device) < 0.5,
        -torch.ones(batch_size, device=device, dtype=geo_mesh.dtype),
        torch.ones(batch_size, device=device, dtype=geo_mesh.dtype),
    )
    edit_scale = direction * float(edit_strength) * amplitude
    edited_geo = apply_gaussian_deformation(
        geo_mesh,
        center,
        normal,
        edit_sigma,
        edit_scale=edit_scale,
        ramp_power=ramp_power,
    )

    changed_radius = edit_sigma * float(changed_scale)
    unchanged_radius = edit_sigma * float(max(unchanged_scale, changed_scale + 1.0e-3))
    return edited_geo, center, normal, edit_sigma, edit_scale, changed_radius, unchanged_radius


def build_query_locality_weights(query_pos, center, changed_radius, unchanged_radius):
    delta = query_pos - center.unsqueeze(1)
    dist2 = delta.float().pow(2).sum(dim=-1)
    changed_sigma2 = changed_radius.float().unsqueeze(1).pow(2).clamp_min(1.0e-6)
    unchanged_sigma2 = unchanged_radius.float().unsqueeze(1).pow(2).clamp_min(1.0e-6)
    changed = torch.exp(-0.5 * dist2 / changed_sigma2)
    outer = torch.exp(-0.5 * dist2 / unchanged_sigma2)
    unchanged = (outer - changed).clamp_min(0.0)
    return changed.to(dtype=query_pos.dtype), unchanged.to(dtype=query_pos.dtype)


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
    ref = None
    total = None
    count = 0
    for key in keys:
        value = aux.get(key)
        if not torch.is_tensor(value):
            continue
        ref = value if ref is None else ref
        if value.ndim < 2 or value.shape[1] == 0:
            continue
        per_query = value.float().mean(dim=-1)
        component = per_query.sum()
        total = component if total is None else (total + component)
        count += int(per_query.numel())
    if ref is None:
        return torch.tensor(0.0)
    if count == 0:
        return ref.new_zeros((), dtype=torch.float32)
    return total / float(count)


def update_ema_dict(ema_dict, value_dict, momentum):
    if ema_dict is None:
        return {key: float(value) for key, value in value_dict.items()}
    out = {}
    momentum = float(momentum)
    for key, value in value_dict.items():
        out[key] = momentum * float(ema_dict.get(key, value)) + (1.0 - momentum) * float(value)
    return out


def calibrate_loss_scales(loss_terms, base_weights, reference_name, min_scale, max_scale, eps=1.0e-8):
    active_names = [name for name, weight in base_weights.items() if float(weight) > 0.0]
    if not active_names:
        return {name: 1.0 for name in loss_terms}
    if reference_name not in loss_terms or float(base_weights.get(reference_name, 0.0)) <= 0.0:
        reference_name = active_names[0]

    ref_value = loss_terms[reference_name].detach().float().abs().clamp_min(eps)
    scales = {}
    for name, term in loss_terms.items():
        if float(base_weights.get(name, 0.0)) <= 0.0:
            scales[name] = 1.0
            continue
        value = term.detach().float().abs()
        if not bool(torch.isfinite(value).item()) or float(value.item()) <= eps:
            scales[name] = 1.0
            continue
        scales[name] = float((ref_value / value).clamp(float(min_scale), float(max_scale)).item())
    scales[reference_name] = 1.0
    return scales


def weighted_loss_sum(loss_terms, base_weights, loss_scales):
    total = None
    for name, term in loss_terms.items():
        weight = float(base_weights.get(name, 0.0)) * float(loss_scales.get(name, 1.0))
        if weight == 0.0:
            continue
        contribution = weight * term
        total = contribution if total is None else (total + contribution)
    if total is None:
        return next(iter(loss_terms.values())).new_zeros(())
    return total


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
):
    metrics = init_metric_dict(fields["surface"], fields["volume"])
    model.eval()

    with torch.inference_mode():
        for batch in tqdm(loader, desc="Eval", leave=False, dynamic_ncols=True):
            geo_mesh, surf_mesh, surf_data, vol_mesh, vol_data, params, _sample_info = parse_batch(batch, params_dim)

            geo_mesh = geo_mesh.to(device, non_blocking=True)
            surf_mesh = surf_mesh.to(device, non_blocking=True)
            surf_data = surf_data.to(device, non_blocking=True)
            vol_mesh = vol_mesh.to(device, non_blocking=True)
            vol_data = vol_data.to(device, non_blocking=True)
            if params is not None:
                params = params.to(device, non_blocking=True)

            if config.dataset == "NACA4":
                surf_data = surf_data[..., :1]
                vol_data = torch.cat([vol_data[..., :1], vol_data[..., 2:4]], dim=-1)

            with torch.autocast(device_type=str(device).split(":")[0], dtype=dtype, enabled=amp):
                y_hat_surf, y_hat_vol = model(geo_mesh, surf_mesh, vol_mesh, params)

            if use_surface_supervision:
                pred_surf = y_hat_surf * std_surf + mean_surf
                gt_surf = surf_data * std_surf + mean_surf
                batch_loss = combined_loss_fn(y_hat_surf.float(), y_hat_vol.float(), surf_data, vol_data)
                surface_rel_l2 = rel_l2_loss_fn(y_hat_surf.float(), surf_data)
            else:
                pred_surf = None
                gt_surf = None
                batch_loss = loss_fn(y_hat_vol.float(), vol_data)
                surface_rel_l2 = torch.tensor(0.0, device=device)

            pred_vol = y_hat_vol * std_vol + mean_vol
            gt_vol = vol_data * std_vol + mean_vol
            volume_rel_l2 = rel_l2_loss_fn(y_hat_vol.float(), vol_data)

            batch_size = surf_data.size(0)
            metrics["loss"] += batch_loss.item() * batch_size
            metrics["rel_l2_surf"] += surface_rel_l2.item() * batch_size
            metrics["rel_l2_vol"] += volume_rel_l2.item() * batch_size
            metrics["rel_l2"] += (surface_rel_l2 + volume_rel_l2).item() * batch_size
            if use_surface_supervision:
                accumulate_channel_metrics(metrics, "rel_l2_surf", pred_surf, gt_surf, fields["surface"], rel_l2_loss_fn, batch_size)
            accumulate_channel_metrics(metrics, "rel_l2_vol", pred_vol, gt_vol, fields["volume"], rel_l2_loss_fn, batch_size)

    for key in metrics.keys():
        metrics[key] /= len(loader.dataset)
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

    gradient_norm = config.gradient_norm
    precisions = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    dtype = precisions.get(config.precision, torch.float16)
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

    resume_ckpt = str(getattr(config, "resume_ckpt", "")).strip()
    init_ckpt = str(getattr(config, "init_ckpt", "")).strip()

    print(f"Total parameters: {count_model_params(model)}")
    model_checkpoint_name = get_model_checkpoint_name(config)
    print(f"Checkpoint name: {model_checkpoint_name}")

    if run is not None and bool(getattr(config, "wandb_watch_model", False)):
        run.watch(model, log="all")

    scaler = torch.amp.GradScaler("cuda", enabled=amp and torch.cuda.is_available())
    optimizer, scheduler, loss_fn, rel_l2_loss_fn = get_optimizer_scheduler_loss(model, config, train_loader, loss_dim=1)
    combined_loss_fn = CombinedLoss(loss_fn, fields) if use_surface_supervision else None

    start_epoch = 0
    global_step = 0
    best_rel_l2 = np.inf
    if bool(getattr(config, "resume_full_state", False)):
        if not resume_ckpt:
            raise ValueError("resume_full_state=True requires experiment.resume_ckpt to be set.")
        start_epoch, global_step, best_rel_l2 = load_full_training_state(
            model, optimizer, scheduler, scaler, resume_ckpt, device, amp
        )
    elif resume_ckpt:
        load_partial_state_dict(model, resume_ckpt, device)
    elif init_ckpt:
        load_partial_state_dict(model, init_ckpt, device)

    train_model = model
    if multi_gpu_strategy == "data_parallel" and torch.cuda.is_available() and torch.cuda.device_count() > 1:
        device_ids = list(range(torch.cuda.device_count()))
        train_model = DataParallel(model, device_ids=device_ids, output_device=device_ids[0], dim=0)
        print(f"[DARM] Enabled DataParallel on device ids {device_ids}.")
        if int(config.batch_size) < len(device_ids):
            print(f"[DARM] Warning: batch_size={config.batch_size} is smaller than num_gpus={len(device_ids)}.")

    extra_metric_keys = [
        "loss_supervised",
        "loss_stability",
        "loss_ghost",
        "loss_high_sparsity",
        "loss_stability_surface",
        "loss_stability_volume",
        "loss_ghost_surface",
        "loss_ghost_volume",
        "mean_support_mass_orig",
        "mean_support_mass_edit",
        "mean_support_mass",
        "mean_route_confidence_orig",
        "mean_route_confidence_edit",
        "mean_route_confidence",
        "mean_route_entropy_orig",
        "mean_route_entropy_edit",
        "mean_route_entropy",
        "mean_evidence_gate_orig",
        "mean_evidence_gate_edit",
        "mean_evidence_gate",
        "mean_raw_evidence_gate_orig",
        "mean_raw_evidence_gate_edit",
        "mean_raw_evidence_gate",
        "changed_support_mass_surface",
        "changed_support_mass_volume",
        "unchanged_support_mass_surface",
        "unchanged_support_mass_volume",
        "changed_evidence_mass_surface",
        "changed_evidence_mass_volume",
        "unchanged_evidence_mass_surface",
        "unchanged_evidence_mass_volume",
        "changed_route_confidence_surface",
        "changed_route_confidence_volume",
        "unchanged_route_confidence_surface",
        "unchanged_route_confidence_volume",
        "changed_response_surface",
        "changed_response_volume",
        "unchanged_response_surface",
        "unchanged_response_volume",
        "changed_route_entropy_surface",
        "changed_route_entropy_volume",
        "unchanged_route_entropy_surface",
        "unchanged_route_entropy_volume",
    ]
    log_every_n_steps = int(getattr(config, "log_every_n_steps", 0))
    console_log_every_n_steps = int(getattr(config, "console_log_every_n_steps", 0))
    show_first_batch_timing = bool(getattr(config, "show_first_batch_timing", False))
    track_train_channel_metrics = bool(getattr(config, "track_train_channel_metrics", False))

    print(
        f"[DARM] steps_per_epoch={len(train_loader)}, "
        f"visible_gpus={torch.cuda.device_count() if torch.cuda.is_available() else 0}, "
        f"batch_size={int(config.batch_size)}"
    )

    loss_balance_enabled = bool(getattr(config, "loss_balance_enabled", getattr(config, "loss_calibration_enabled", True)))
    loss_balance_reference = str(getattr(config, "loss_balance_reference", getattr(config, "loss_calibration_reference", "supervised")))
    loss_balance_min_scale = float(getattr(config, "loss_balance_min_scale", getattr(config, "loss_calibration_min_scale", 0.1)))
    loss_balance_max_scale = float(getattr(config, "loss_balance_max_scale", getattr(config, "loss_calibration_max_scale", 10.0)))
    loss_balance_scales = None
    loss_balance_steps = max(1, int(getattr(config, "loss_balance_steps", 16)))
    loss_balance_ema_momentum = float(getattr(config, "loss_balance_ema_momentum", 0.8))
    loss_balance_ema = None

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

            train_model.train()
            train_pbar = tqdm(
                train_loader,
                desc=f"Train {ep + 1}/{config.epochs}",
                leave=False,
                dynamic_ncols=True,
            )

            edit_warmup = consistency_warmup_factor(ep, getattr(config, "edit_warmup_epochs", getattr(config, "consistency_warmup_epochs", 10)))
            stability_weight = edit_warmup * float(getattr(config, "stability_weight", getattr(config, "edit_consistency_weight", 0.5)))
            ghost_weight = edit_warmup * float(getattr(config, "ghost_suppression_weight", 0.25))
            high_sparsity_weight = float(getattr(config, "high_sparsity_weight", 0.0))
            supervised_weight = float(getattr(config, "supervised_weight", 1.0))
            loss_beta = float(getattr(config, "stability_smooth_l1_beta", getattr(config, "prediction_consistency_smooth_l1_beta", 0.05)))
            ghost_target_ratio = float(getattr(config, "ghost_target_ratio", 0.35))
            ghost_absolute_weight = float(getattr(config, "ghost_absolute_weight", 0.1))

            for batch_idx, batch in enumerate(train_pbar):
                batch_t0 = default_timer()
                geo_mesh, surf_mesh, surf_data, vol_mesh, vol_data, params, _sample_info = parse_batch(batch, params_dim)
                geo_mesh = geo_mesh.to(device, non_blocking=True)
                surf_mesh = surf_mesh.to(device, non_blocking=True)
                surf_data = surf_data.to(device, non_blocking=True)
                vol_mesh = vol_mesh.to(device, non_blocking=True)
                vol_data = vol_data.to(device, non_blocking=True)
                if params is not None:
                    params = params.to(device, non_blocking=True)

                if config.dataset == "NACA4":
                    surf_data = surf_data[..., :1]
                    vol_data = torch.cat([vol_data[..., :1], vol_data[..., 2:4]], dim=-1)

                optimizer.zero_grad(set_to_none=True)

                with torch.no_grad():
                    edit_ramp_power = float(getattr(config, "edit_ramp_power", 1.5))
                    edited_geo, edit_center, edit_normal, edit_sigma, edit_scale, changed_radius, unchanged_radius = build_local_edit(
                        geo_mesh,
                        min_points=int(getattr(config, "edit_patch_min_points", 256)),
                        max_points=int(getattr(config, "edit_patch_max_points", 1024)),
                        edit_strength=float(getattr(config, "edit_strength", 0.75)),
                        ramp_power=edit_ramp_power,
                        changed_scale=float(getattr(config, "edit_changed_radius_scale", 1.2)),
                        unchanged_scale=float(getattr(config, "edit_unchanged_radius_scale", 2.5)),
                        candidate_points=int(getattr(config, "edit_candidate_points", 2048)),
                    )
                    edited_surf_mesh = apply_gaussian_deformation(
                        surf_mesh,
                        edit_center,
                        edit_normal,
                        edit_sigma,
                        edit_scale=edit_scale,
                        ramp_power=edit_ramp_power,
                    )
                    edited_vol_mesh = apply_gaussian_deformation(
                        vol_mesh,
                        edit_center,
                        edit_normal,
                        edit_sigma,
                        edit_scale=edit_scale,
                        ramp_power=edit_ramp_power,
                    )
                    surf_changed_w, surf_unchanged_w = build_query_locality_weights(
                        surf_mesh, edit_center, changed_radius, unchanged_radius
                    )
                    vol_changed_w, vol_unchanged_w = build_query_locality_weights(
                        vol_mesh, edit_center, changed_radius, unchanged_radius
                    )
                prep_t1 = default_timer()

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

                    loss_supervised = (
                        combined_loss_fn(y_orig_surf_f, y_orig_vol_f, surf_data, vol_data)
                        if use_surface_supervision
                        else loss_fn(y_orig_vol_f, vol_data)
                    )

                    stability_loss = y_orig_vol_f.new_zeros(())
                    stability_surface = y_orig_vol_f.new_zeros(())
                    stability_volume = y_orig_vol_f.new_zeros(())
                    if use_surface_supervision:
                        stability_surface = masked_smooth_l1(
                            y_edit_surf_f,
                            y_orig_surf_f.detach(),
                            surf_unchanged_w,
                            beta=loss_beta,
                        )
                        stability_loss = stability_loss + stability_surface
                    stability_volume = masked_smooth_l1(
                        y_edit_vol_f,
                        y_orig_vol_f.detach(),
                        vol_unchanged_w,
                        beta=loss_beta,
                    )
                    stability_loss = stability_loss + stability_volume

                    changed_support_mass_surface = (
                        masked_mean(aux_edit["surface_support_mass"].float(), surf_changed_w)
                        if use_surface_supervision else y_orig_vol_f.new_zeros(())
                    )
                    changed_support_mass_volume = masked_mean(aux_edit["volume_support_mass"].float(), vol_changed_w)
                    unchanged_support_mass_surface = (
                        masked_mean(aux_edit["surface_support_mass"].float(), surf_unchanged_w)
                        if use_surface_supervision else y_orig_vol_f.new_zeros(())
                    )
                    unchanged_support_mass_volume = masked_mean(aux_edit["volume_support_mass"].float(), vol_unchanged_w)
                    changed_response_surface = (
                        masked_response_norm(y_edit_surf_f, y_orig_surf_f.detach(), surf_changed_w)
                        if use_surface_supervision else y_orig_vol_f.new_zeros(())
                    )
                    changed_response_volume = masked_response_norm(y_edit_vol_f, y_orig_vol_f.detach(), vol_changed_w)
                    unchanged_response_surface = (
                        masked_response_norm(y_edit_surf_f, y_orig_surf_f.detach(), surf_unchanged_w)
                        if use_surface_supervision else y_orig_vol_f.new_zeros(())
                    )
                    unchanged_response_volume = masked_response_norm(y_edit_vol_f, y_orig_vol_f.detach(), vol_unchanged_w)

                    ghost_loss = y_orig_vol_f.new_zeros(())
                    ghost_surface = y_orig_vol_f.new_zeros(())
                    ghost_volume = y_orig_vol_f.new_zeros(())
                    if ghost_weight > 0.0:
                        # Penalize prediction-change spillover outside the edited locality.
                        if use_surface_supervision:
                            ghost_surface = relative_locality_penalty(
                                unchanged_response_surface,
                                changed_response_surface,
                                target_ratio=ghost_target_ratio,
                            )
                            if ghost_absolute_weight > 0.0:
                                ghost_surface = ghost_surface + ghost_absolute_weight * unchanged_response_surface
                            ghost_loss = ghost_loss + ghost_surface
                        ghost_volume = relative_locality_penalty(
                            unchanged_response_volume,
                            changed_response_volume,
                            target_ratio=ghost_target_ratio,
                        )
                        if ghost_absolute_weight > 0.0:
                            ghost_volume = ghost_volume + ghost_absolute_weight * unchanged_response_volume
                        ghost_loss = ghost_loss + ghost_volume

                    high_sparsity_loss = y_orig_vol_f.new_zeros(())
                    if high_sparsity_weight > 0.0:
                        if use_surface_supervision:
                            high_sparsity_loss = high_sparsity_loss + 0.5 * (
                                masked_group_sparsity(aux_orig["surface_high"], None)
                                + masked_group_sparsity(aux_edit["surface_high"], None)
                            )
                        high_sparsity_loss = high_sparsity_loss + 0.5 * (
                            masked_group_sparsity(aux_orig["volume_high"], None)
                            + masked_group_sparsity(aux_edit["volume_high"], None)
                        )

                    with torch.no_grad():
                        changed_evidence_mass_surface = (
                            masked_mean(aux_edit["surface_evidence_mass"].float(), surf_changed_w)
                            if use_surface_supervision else y_orig_vol_f.new_zeros(())
                        )
                        changed_evidence_mass_volume = masked_mean(aux_edit["volume_evidence_mass"].float(), vol_changed_w)
                        unchanged_evidence_mass_surface = (
                            masked_mean(aux_edit["surface_evidence_mass"].float(), surf_unchanged_w)
                            if use_surface_supervision else y_orig_vol_f.new_zeros(())
                        )
                        unchanged_evidence_mass_volume = masked_mean(aux_edit["volume_evidence_mass"].float(), vol_unchanged_w)

                        changed_route_confidence_surface = (
                            masked_mean(aux_edit["surface_route_confidence"].float(), surf_changed_w)
                            if use_surface_supervision else y_orig_vol_f.new_zeros(())
                        )
                        changed_route_confidence_volume = masked_mean(aux_edit["volume_route_confidence"].float(), vol_changed_w)
                        unchanged_route_confidence_surface = (
                            masked_mean(aux_edit["surface_route_confidence"].float(), surf_unchanged_w)
                            if use_surface_supervision else y_orig_vol_f.new_zeros(())
                        )
                        unchanged_route_confidence_volume = masked_mean(aux_edit["volume_route_confidence"].float(), vol_unchanged_w)

                        changed_route_entropy_surface = (
                            masked_mean(aux_edit["surface_route_entropy"].float(), surf_changed_w)
                            if use_surface_supervision else y_orig_vol_f.new_zeros(())
                        )
                        changed_route_entropy_volume = masked_mean(aux_edit["volume_route_entropy"].float(), vol_changed_w)
                        unchanged_route_entropy_surface = (
                            masked_mean(aux_edit["surface_route_entropy"].float(), surf_unchanged_w)
                            if use_surface_supervision else y_orig_vol_f.new_zeros(())
                        )
                        unchanged_route_entropy_volume = masked_mean(aux_edit["volume_route_entropy"].float(), vol_unchanged_w)

                    forward_t1 = default_timer()

                loss_terms = OrderedDict(
                    [
                        ("supervised", loss_supervised),
                        ("stability", stability_loss),
                        ("ghost", ghost_loss),
                        ("high_sparsity", high_sparsity_loss),
                    ]
                )
                loss_base_weights = OrderedDict(
                    [
                        ("supervised", supervised_weight),
                        ("stability", stability_weight),
                        ("ghost", ghost_weight),
                        ("high_sparsity", high_sparsity_weight),
                    ]
                )
                if loss_balance_enabled and global_step < loss_balance_steps:
                    current_loss_values = {
                        name: float(term.detach().float().abs().item())
                        for name, term in loss_terms.items()
                    }
                    loss_balance_ema = update_ema_dict(loss_balance_ema, current_loss_values, loss_balance_ema_momentum)
                    ema_terms = {
                        name: torch.tensor(value, device=device, dtype=torch.float32)
                        for name, value in loss_balance_ema.items()
                    }
                    loss_balance_scales = calibrate_loss_scales(
                        ema_terms,
                        loss_base_weights,
                        reference_name=loss_balance_reference,
                        min_scale=loss_balance_min_scale,
                        max_scale=loss_balance_max_scale,
                    )
                    if run is not None and (global_step == loss_balance_steps - 1 or global_step == 0):
                        print(f"[DARM] calibrated loss scales: {loss_balance_scales}")
                        wandb.log({f"loss_scale/{name}": scale for name, scale in loss_balance_scales.items()}, step=global_step)
                elif loss_balance_scales is None:
                    loss_balance_scales = {name: 1.0 for name in loss_terms}

                loss = weighted_loss_sum(loss_terms, loss_base_weights, loss_balance_scales)

                if not torch.isfinite(loss):
                    print(f"[warn] Non-finite loss at epoch {ep} batch {batch_idx}; skipping optimizer step.")
                    optimizer.zero_grad(set_to_none=True)
                    continue

                if amp and torch.cuda.is_available():
                    prev_scale = scaler.get_scale()
                    scaler.scale(loss).backward()
                    if gradient_norm is not None:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    if scaler.get_scale() >= prev_scale:
                        scheduler.step()
                else:
                    loss.backward()
                    if gradient_norm is not None:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_norm)
                    optimizer.step()
                    scheduler.step()
                backward_t1 = default_timer()

                batch_size = surf_data.size(0)
                batch_size_float = float(batch_size)
                train_sample_count += batch_size
                train_losses["loss"] += loss.detach().float() * batch_size_float
                train_losses["loss_supervised"] += loss_supervised.detach().float() * batch_size_float
                train_losses["loss_stability"] += stability_loss.detach().float() * batch_size_float
                train_losses["loss_ghost"] += ghost_loss.detach().float() * batch_size_float
                train_losses["loss_high_sparsity"] += high_sparsity_loss.detach().float() * batch_size_float
                train_losses["loss_stability_surface"] += stability_surface.detach().float() * batch_size_float
                train_losses["loss_stability_volume"] += stability_volume.detach().float() * batch_size_float
                train_losses["loss_ghost_surface"] += ghost_surface.detach().float() * batch_size_float
                train_losses["loss_ghost_volume"] += ghost_volume.detach().float() * batch_size_float
                mean_support_mass = aux_query_mean(aux_pair, "surface_support_mass", "volume_support_mass").detach().float()
                mean_support_mass_orig = aux_query_mean(aux_orig, "surface_support_mass", "volume_support_mass").detach().float()
                mean_support_mass_edit = aux_query_mean(aux_edit, "surface_support_mass", "volume_support_mass").detach().float()
                mean_route_confidence = aux_query_mean(aux_pair, "surface_route_confidence", "volume_route_confidence").detach().float()
                mean_route_confidence_orig = aux_query_mean(aux_orig, "surface_route_confidence", "volume_route_confidence").detach().float()
                mean_route_confidence_edit = aux_query_mean(aux_edit, "surface_route_confidence", "volume_route_confidence").detach().float()
                mean_route_entropy = aux_query_mean(aux_pair, "surface_route_entropy", "volume_route_entropy").detach().float()
                mean_route_entropy_orig = aux_query_mean(aux_orig, "surface_route_entropy", "volume_route_entropy").detach().float()
                mean_route_entropy_edit = aux_query_mean(aux_edit, "surface_route_entropy", "volume_route_entropy").detach().float()
                mean_evidence_gate = aux_query_mean(aux_pair, "surface_evidence_mass", "volume_evidence_mass").detach().float()
                mean_evidence_gate_orig = aux_query_mean(aux_orig, "surface_evidence_mass", "volume_evidence_mass").detach().float()
                mean_evidence_gate_edit = aux_query_mean(aux_edit, "surface_evidence_mass", "volume_evidence_mass").detach().float()
                mean_raw_evidence_gate = aux_query_mean(aux_pair, "surface_raw_evidence_gate", "volume_raw_evidence_gate").detach().float()
                mean_raw_evidence_gate_orig = aux_query_mean(aux_orig, "surface_raw_evidence_gate", "volume_raw_evidence_gate").detach().float()
                mean_raw_evidence_gate_edit = aux_query_mean(aux_edit, "surface_raw_evidence_gate", "volume_raw_evidence_gate").detach().float()
                train_losses["mean_support_mass_orig"] += mean_support_mass_orig * batch_size_float
                train_losses["mean_support_mass_edit"] += mean_support_mass_edit * batch_size_float
                train_losses["mean_support_mass"] += mean_support_mass * batch_size_float
                train_losses["mean_route_confidence_orig"] += mean_route_confidence_orig * batch_size_float
                train_losses["mean_route_confidence_edit"] += mean_route_confidence_edit * batch_size_float
                train_losses["mean_route_confidence"] += mean_route_confidence * batch_size_float
                train_losses["mean_route_entropy_orig"] += mean_route_entropy_orig * batch_size_float
                train_losses["mean_route_entropy_edit"] += mean_route_entropy_edit * batch_size_float
                train_losses["mean_route_entropy"] += mean_route_entropy * batch_size_float
                train_losses["mean_evidence_gate_orig"] += mean_evidence_gate_orig * batch_size_float
                train_losses["mean_evidence_gate_edit"] += mean_evidence_gate_edit * batch_size_float
                train_losses["mean_evidence_gate"] += mean_evidence_gate * batch_size_float
                train_losses["mean_raw_evidence_gate_orig"] += mean_raw_evidence_gate_orig * batch_size_float
                train_losses["mean_raw_evidence_gate_edit"] += mean_raw_evidence_gate_edit * batch_size_float
                train_losses["mean_raw_evidence_gate"] += mean_raw_evidence_gate * batch_size_float
                train_losses["changed_support_mass_surface"] += changed_support_mass_surface.detach().float() * batch_size_float
                train_losses["changed_support_mass_volume"] += changed_support_mass_volume.detach().float() * batch_size_float
                train_losses["unchanged_support_mass_surface"] += unchanged_support_mass_surface.detach().float() * batch_size_float
                train_losses["unchanged_support_mass_volume"] += unchanged_support_mass_volume.detach().float() * batch_size_float
                train_losses["changed_response_surface"] += changed_response_surface.detach().float() * batch_size_float
                train_losses["changed_response_volume"] += changed_response_volume.detach().float() * batch_size_float
                train_losses["unchanged_response_surface"] += unchanged_response_surface.detach().float() * batch_size_float
                train_losses["unchanged_response_volume"] += unchanged_response_volume.detach().float() * batch_size_float
                train_losses["changed_evidence_mass_surface"] += changed_evidence_mass_surface.detach().float() * batch_size_float
                train_losses["changed_evidence_mass_volume"] += changed_evidence_mass_volume.detach().float() * batch_size_float
                train_losses["unchanged_evidence_mass_surface"] += unchanged_evidence_mass_surface.detach().float() * batch_size_float
                train_losses["unchanged_evidence_mass_volume"] += unchanged_evidence_mass_volume.detach().float() * batch_size_float
                train_losses["changed_route_confidence_surface"] += changed_route_confidence_surface.detach().float() * batch_size_float
                train_losses["changed_route_confidence_volume"] += changed_route_confidence_volume.detach().float() * batch_size_float
                train_losses["unchanged_route_confidence_surface"] += unchanged_route_confidence_surface.detach().float() * batch_size_float
                train_losses["unchanged_route_confidence_volume"] += unchanged_route_confidence_volume.detach().float() * batch_size_float
                train_losses["changed_route_entropy_surface"] += changed_route_entropy_surface.detach().float() * batch_size_float
                train_losses["changed_route_entropy_volume"] += changed_route_entropy_volume.detach().float() * batch_size_float
                train_losses["unchanged_route_entropy_surface"] += unchanged_route_entropy_surface.detach().float() * batch_size_float
                train_losses["unchanged_route_entropy_volume"] += unchanged_route_entropy_volume.detach().float() * batch_size_float

                with torch.no_grad():
                    surface_loss = rel_l2_loss_fn(y_orig_surf_f, surf_data) if use_surface_supervision else torch.tensor(0.0, device=device)
                    volume_loss = rel_l2_loss_fn(y_orig_vol_f, vol_data)
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
                                train_losses[f"rel_l2_surf_{field_name}"] += channel_loss.detach().float() * batch_size_float
                        pred_vol_train = y_orig_vol_f * std_vol + mean_vol
                        gt_vol_train = vol_data * std_vol + mean_vol
                        for channel_idx, field_name in enumerate(fields["volume"]):
                            channel_loss = rel_l2_loss_fn(
                                pred_vol_train[..., channel_idx:channel_idx + 1],
                                gt_vol_train[..., channel_idx:channel_idx + 1],
                            )
                            train_losses[f"rel_l2_vol_{field_name}"] += channel_loss.detach().float() * batch_size_float

                global_step += 1
                if run is not None and log_every_n_steps > 0 and (
                    (batch_idx + 1) % log_every_n_steps == 0 or batch_idx == len(train_loader) - 1
                ):
                    wandb.log(
                        {
                            "train/batch_loss": float(loss.item()),
                            "train/batch_supervised": float(loss_supervised.item()),
                            "train/batch_stability": float(stability_loss.item()),
                            "train/batch_ghost": float(ghost_loss.item()),
                            "train/batch_high_sparsity": float(high_sparsity_loss.item()),
                            "train/batch_stability_surface": float(stability_surface.item()),
                            "train/batch_stability_volume": float(stability_volume.item()),
                            "train/batch_ghost_surface": float(ghost_surface.item()),
                            "train/batch_ghost_volume": float(ghost_volume.item()),
                            "train/batch_mean_support_mass_orig": float(mean_support_mass_orig.item()),
                            "train/batch_mean_support_mass_edit": float(mean_support_mass_edit.item()),
                            "train/batch_mean_support_mass": float(mean_support_mass.item()),
                            "train/batch_mean_route_confidence_orig": float(mean_route_confidence_orig.item()),
                            "train/batch_mean_route_confidence_edit": float(mean_route_confidence_edit.item()),
                            "train/batch_mean_route_confidence": float(mean_route_confidence.item()),
                            "train/batch_mean_route_entropy_orig": float(mean_route_entropy_orig.item()),
                            "train/batch_mean_route_entropy_edit": float(mean_route_entropy_edit.item()),
                            "train/batch_mean_route_entropy": float(mean_route_entropy.item()),
                            "train/batch_mean_evidence_gate_orig": float(mean_evidence_gate_orig.item()),
                            "train/batch_mean_evidence_gate_edit": float(mean_evidence_gate_edit.item()),
                            "train/batch_mean_evidence_gate": float(mean_evidence_gate.item()),
                            "train/batch_mean_raw_evidence_gate_orig": float(mean_raw_evidence_gate_orig.item()),
                            "train/batch_mean_raw_evidence_gate_edit": float(mean_raw_evidence_gate_edit.item()),
                            "train/batch_mean_raw_evidence_gate": float(mean_raw_evidence_gate.item()),
                            "train/batch_changed_support_mass_surface": float(changed_support_mass_surface.item()),
                            "train/batch_changed_support_mass_volume": float(changed_support_mass_volume.item()),
                            "train/batch_unchanged_support_mass_surface": float(unchanged_support_mass_surface.item()),
                            "train/batch_unchanged_support_mass_volume": float(unchanged_support_mass_volume.item()),
                            "train/batch_changed_response_surface": float(changed_response_surface.item()),
                            "train/batch_changed_response_volume": float(changed_response_volume.item()),
                            "train/batch_unchanged_response_surface": float(unchanged_response_surface.item()),
                            "train/batch_unchanged_response_volume": float(unchanged_response_volume.item()),
                            "train/batch_changed_evidence_mass_surface": float(changed_evidence_mass_surface.item()),
                            "train/batch_changed_evidence_mass_volume": float(changed_evidence_mass_volume.item()),
                            "train/batch_unchanged_evidence_mass_surface": float(unchanged_evidence_mass_surface.item()),
                            "train/batch_unchanged_evidence_mass_volume": float(unchanged_evidence_mass_volume.item()),
                            "train/batch_changed_route_confidence_surface": float(changed_route_confidence_surface.item()),
                            "train/batch_changed_route_confidence_volume": float(changed_route_confidence_volume.item()),
                            "train/batch_unchanged_route_confidence_surface": float(unchanged_route_confidence_surface.item()),
                            "train/batch_unchanged_route_confidence_volume": float(unchanged_route_confidence_volume.item()),
                            "train/batch_changed_route_entropy_surface": float(changed_route_entropy_surface.item()),
                            "train/batch_changed_route_entropy_volume": float(changed_route_entropy_volume.item()),
                            "train/batch_unchanged_route_entropy_surface": float(unchanged_route_entropy_surface.item()),
                            "train/batch_unchanged_route_entropy_volume": float(unchanged_route_entropy_volume.item()),
                            "train/stability_weight": stability_weight,
                            "train/ghost_weight": ghost_weight,
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
                f"train loss: {train_losses_log['loss']:.5f}, test loss: {test_losses['loss']:.5f}"
            )

            if run is not None:
                wandb_dict = {
                    "lr": scheduler.get_last_lr()[0],
                    "train/stability_weight": stability_weight,
                    "train/ghost_weight": ghost_weight,
                    "train/ghost_target_ratio": ghost_target_ratio,
                    "train/ghost_absolute_weight": ghost_absolute_weight,
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
                        "loss_stability",
                        "loss_ghost",
                        "loss_high_sparsity",
                        "loss_stability_surface",
                        "loss_stability_volume",
                        "loss_ghost_surface",
                        "loss_ghost_volume",
                        "mean_support_mass_orig",
                        "mean_support_mass_edit",
                        "mean_support_mass",
                        "mean_route_confidence_orig",
                        "mean_route_confidence_edit",
                        "mean_route_confidence",
                        "mean_route_entropy_orig",
                        "mean_route_entropy_edit",
                        "mean_route_entropy",
                        "mean_evidence_gate_orig",
                        "mean_evidence_gate_edit",
                        "mean_evidence_gate",
                        "mean_raw_evidence_gate_orig",
                        "mean_raw_evidence_gate_edit",
                        "mean_raw_evidence_gate",
                        "changed_support_mass_surface",
                        "changed_support_mass_volume",
                        "unchanged_support_mass_surface",
                        "unchanged_support_mass_volume",
                        "changed_response_surface",
                        "changed_response_volume",
                        "unchanged_response_surface",
                        "unchanged_response_volume",
                        "changed_evidence_mass_surface",
                        "changed_evidence_mass_volume",
                        "unchanged_evidence_mass_surface",
                        "unchanged_evidence_mass_volume",
                        "changed_route_confidence_surface",
                        "changed_route_confidence_volume",
                        "unchanged_route_confidence_surface",
                        "unchanged_route_confidence_volume",
                        "changed_route_entropy_surface",
                        "changed_route_entropy_volume",
                        "unchanged_route_entropy_surface",
                        "unchanged_route_entropy_volume",
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
