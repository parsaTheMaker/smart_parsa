from __future__ import annotations

import os
from timeit import default_timer

import hydra
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from omegaconf import DictConfig
from tqdm.auto import tqdm

from data.datasets import get_dataset
from loss.losses import CombinedLoss
from models.smart.smart_sat3 import SMARTSAT3
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


def accumulate_channel_metrics(metrics, prefix, pred, gt, field_names, rel_l2_loss_fn, batch_size, metric_weight=1.0):
    for channel_idx, field_name in enumerate(field_names):
        channel_loss = rel_l2_loss_fn(pred[..., channel_idx:channel_idx + 1], gt[..., channel_idx:channel_idx + 1])
        metrics[f"{prefix}_{field_name}"] += channel_loss.item() * batch_size * metric_weight


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


def load_full_training_state(model, optimizer, scheduler, scaler, checkpoint_path, device, load_scaler=True):
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


def consistency_warmup_factor(epoch, warmup_epochs):
    warmup_epochs = int(warmup_epochs)
    if warmup_epochs <= 0:
        return 1.0
    return min(1.0, float(epoch + 1) / float(warmup_epochs))


def prediction_consistency_loss(y1_surf, y1_vol, y2_surf, y2_vol):
    return F.mse_loss(y2_surf, y1_surf.detach()) + F.mse_loss(y2_vol, y1_vol.detach())


def latent_consistency_loss(latent_teacher, latent_student):
    latent_teacher = F.layer_norm(latent_teacher.detach().float(), (latent_teacher.shape[-1],))
    latent_student = F.layer_norm(latent_student.float(), (latent_student.shape[-1],))
    return F.mse_loss(latent_student, latent_teacher)


def move_optional_tensor(x, device):
    if x is None:
        return None
    return x.to(device, non_blocking=True)


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

    pbar = tqdm(loader, desc=f"Eval {mode_name}", leave=False, dynamic_ncols=True)
    with torch.no_grad():
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
            metrics["loss"] += batch_loss.item() * batch_size
            metrics["rel_l2_surf"] += surface_rel_l2.item() * batch_size
            metrics["rel_l2_vol"] += volume_rel_l2.item() * batch_size
            metrics["rel_l2"] += (surface_rel_l2 + volume_rel_l2).item() * batch_size

            if use_surface_supervision:
                accumulate_channel_metrics(metrics, "rel_l2_surf", pred_surf, gt_surf, fields["surface"], rel_l2_loss_fn, batch_size)
            accumulate_channel_metrics(metrics, "rel_l2_vol", pred_vol, gt_vol, fields["volume"], rel_l2_loss_fn, batch_size)

            pbar.set_postfix(loss=f"{batch_loss.item():.4f}")

    for key in metrics.keys():
        metrics[key] /= len(loader.dataset)
    return metrics


def run_consistency_training(cfg, model_ctor, model_requires_density):
    config = cfg.experiment
    wandb_config = cfg.wandb
    run = initialize_wandb(config, wandb_config)

    device = initialize_gpu(config.random_seed, high_precision=False)

    gradient_norm = config.gradient_norm
    precisions = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    dtype = precisions.get(config.precision, torch.float16)
    amp = config.amp
    print(gradient_norm, amp, dtype)

    train_data, test_data, stats, spatial_dim, surf_channels, vol_channels, params_dim, fields = get_dataset(config)

    def apply_vanilla_smart_field_subset():
        nonlocal fields, surf_channels, vol_channels
        if config.dataset == "NACA4":
            fields = {"surface": ["pressure"], "volume": ["pressure", "velocity_x", "velocity_y"]}
            surf_channels = 1
            vol_channels = 3

    apply_vanilla_smart_field_subset()
    print(f"[{config.model_name}] training signals -> surface: {fields['surface']} | volume: {fields['volume']}")

    point_info = apply_naca4_auto_point_budget(config, train_data, for_cat=False)
    if point_info is not None:
        print_point_budget(config.model_name, point_info)
        train_data, test_data, stats, spatial_dim, surf_channels, vol_channels, params_dim, fields = get_dataset(config)
        apply_vanilla_smart_field_subset()
        print(f"[{config.model_name}] training signals -> surface: {fields['surface']} | volume: {fields['volume']}")

    use_surface_supervision = len(fields["surface"]) > 0
    set_dataset_epoch(train_data, 0)
    set_dataset_epoch(test_data, 0)

    prefetch_factor = int(getattr(config, "prefetch_factor", 2))
    pin_memory = bool(getattr(config, "pin_memory", True))
    dl_common = dict(batch_size=config.batch_size, num_workers=config.num_workers, pin_memory=pin_memory)
    if config.num_workers > 0:
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
    model = model_ctor(**merged_kwargs).to(device)

    print(f"Total parameters: {count_model_params(model)}")
    model_checkpoint_name = get_model_checkpoint_name(config)
    print(f"Checkpoint name: {model_checkpoint_name}")
    if bool(getattr(config, "wandb_watch_model", False)):
        run.watch(model, log="all")

    scaler = torch.amp.GradScaler("cuda")
    optimizer, scheduler, loss_fn, rel_l2_loss_fn = get_optimizer_scheduler_loss(model, config, train_loader, loss_dim=1)
    combined_loss_fn = CombinedLoss(loss_fn, fields) if use_surface_supervision else None

    best_robust_rel_l2 = np.inf
    global_step = 0
    start_epoch = 0
    log_every_n_steps = getattr(config, "log_every_n_steps", 10)

    resume_ckpt = str(getattr(config, "resume_ckpt", "")).strip()
    resume_full_state = bool(getattr(config, "resume_full_state", False))
    if resume_ckpt:
        if resume_full_state:
            start_epoch, global_step, best_robust_rel_l2 = load_full_training_state(
                model,
                optimizer,
                scheduler,
                scaler,
                resume_ckpt,
                device,
                load_scaler=bool(amp),
            )
        else:
            load_partial_state_dict(model, resume_ckpt, device)

    train_extra_keys = [
        "loss_supervised_primary",
        "loss_supervised_secondary",
        "loss_prediction_consistency",
        "loss_latent_consistency",
        "secondary_inverse_density_fraction",
    ]

    try:
        for ep in tqdm(range(start_epoch, config.epochs), desc="Epochs", dynamic_ncols=True):
            t1 = default_timer()
            set_dataset_epoch(train_data, ep)
            set_dataset_epoch(test_data, 0)

            train_losses = init_metric_dict(fields["surface"], fields["volume"], extra_keys=train_extra_keys)
            model.train()
            train_pbar = tqdm(train_loader, desc=f"Train {ep + 1}/{config.epochs}", leave=False, dynamic_ncols=True)

            warmup = consistency_warmup_factor(ep, getattr(config, "consistency_warmup_epochs", 0))
            pred_consistency_weight = warmup * float(getattr(config, "prediction_consistency_weight", 1.0))
            latent_consistency_weight = warmup * float(getattr(config, "latent_consistency_weight", 0.0))

            for batch_idx, batch in enumerate(train_pbar):
                geo_mesh, surf_mesh, surf_data, vol_mesh, vol_data, params, geo_log_density = unpack_batch(batch, params_dim)
                if geo_log_density is None:
                    raise RuntimeError(f"{config.model_name} requires geometry log density from the dataset.")

                primary_view_geo, primary_view_density, _ = sample_geometry_view(
                    geo_mesh,
                    geo_log_density,
                    num_points=int(getattr(config, "view_geometry_points", 0)),
                    mode=str(getattr(config, "train_primary_sampling_mode", "uniform_wor")),
                    inverse_density_beta=float(getattr(config, "inverse_density_beta", 1.0)),
                    mixed_inverse_density_prob=float(getattr(config, "mixed_inverse_density_prob", 0.5)),
                    seed=int(config.random_seed + ep * 1000003 + batch_idx * 10007 + 11),
                )
                secondary_view_geo, secondary_view_density, secondary_modes = sample_geometry_view(
                    geo_mesh,
                    geo_log_density,
                    num_points=int(getattr(config, "view_geometry_points", 0)),
                    mode=str(getattr(config, "train_secondary_sampling_mode", "mixed")),
                    inverse_density_beta=float(getattr(config, "inverse_density_beta", 1.0)),
                    mixed_inverse_density_prob=float(getattr(config, "mixed_inverse_density_prob", 0.5)),
                    seed=int(config.random_seed + ep * 1000003 + batch_idx * 10007 + 29),
                )

                params = move_optional_tensor(params, device)
                surf_mesh = surf_mesh.to(device, non_blocking=True)
                surf_data = surf_data.to(device, non_blocking=True)
                vol_mesh = vol_mesh.to(device, non_blocking=True)
                vol_data = vol_data.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                primary_view_geo = primary_view_geo.to(device, non_blocking=True)
                primary_view_density = primary_view_density.to(device, non_blocking=True)

                with torch.autocast(device_type=str(device).split(":")[0], dtype=dtype, enabled=amp):
                    if model_requires_density:
                        y1_surf, y1_vol, latent1 = model(
                            primary_view_geo,
                            surf_mesh,
                            vol_mesh,
                            params,
                            geo_log_density=primary_view_density,
                            return_latent=True,
                        )
                    else:
                        y1_surf, y1_vol, latent1 = model(
                            primary_view_geo,
                            surf_mesh,
                            vol_mesh,
                            params,
                            return_latent=True,
                        )
                    supervised_primary = combined_loss_fn(y1_surf, y1_vol, surf_data, vol_data) if use_surface_supervision else loss_fn(y1_vol, vol_data)
                    primary_loss = 0.5 * supervised_primary

                if amp:
                    scaler.scale(primary_loss).backward()
                else:
                    primary_loss.backward()

                y1_surf_teacher = y1_surf.detach()
                y1_vol_teacher = y1_vol.detach()
                latent1_teacher = latent1.detach()

                del primary_view_geo
                del primary_view_density
                del latent1

                secondary_view_geo = secondary_view_geo.to(device, non_blocking=True)
                secondary_view_density = secondary_view_density.to(device, non_blocking=True)

                with torch.autocast(device_type=str(device).split(":")[0], dtype=dtype, enabled=amp):
                    if model_requires_density:
                        y2_surf, y2_vol, latent2 = model(
                            secondary_view_geo,
                            surf_mesh,
                            vol_mesh,
                            params,
                            geo_log_density=secondary_view_density,
                            return_latent=True,
                        )
                    else:
                        y2_surf, y2_vol, latent2 = model(
                            secondary_view_geo,
                            surf_mesh,
                            vol_mesh,
                            params,
                            return_latent=True,
                        )
                    supervised_secondary = combined_loss_fn(y2_surf, y2_vol, surf_data, vol_data) if use_surface_supervision else loss_fn(y2_vol, vol_data)
                    pred_consistency = prediction_consistency_loss(y1_surf_teacher, y1_vol_teacher, y2_surf, y2_vol)
                    lat_consistency = latent_consistency_loss(latent1_teacher, latent2)
                    secondary_loss = (
                        0.5 * supervised_secondary
                        + pred_consistency_weight * pred_consistency
                        + latent_consistency_weight * lat_consistency
                    )

                if amp:
                    scaler.scale(secondary_loss).backward()
                    if gradient_norm is not None:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    secondary_loss.backward()
                    if gradient_norm is not None:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_norm)
                    optimizer.step()
                scheduler.step()

                total_loss = primary_loss.detach() + secondary_loss.detach()
                batch_size = surf_data.size(0)
                secondary_inverse_density_fraction = float(
                    sum(mode.startswith("inverse_density") for mode in secondary_modes) / max(len(secondary_modes), 1)
                )

                with torch.no_grad():
                    surf_rel_primary = rel_l2_loss_fn(y1_surf_teacher, surf_data) if use_surface_supervision else torch.tensor(0.0, device=device)
                    vol_rel_primary = rel_l2_loss_fn(y1_vol_teacher, vol_data)
                    surf_rel_secondary = rel_l2_loss_fn(y2_surf, surf_data) if use_surface_supervision else torch.tensor(0.0, device=device)
                    vol_rel_secondary = rel_l2_loss_fn(y2_vol, vol_data)

                    surface_loss = 0.5 * (surf_rel_primary + surf_rel_secondary)
                    volume_loss = 0.5 * (vol_rel_primary + vol_rel_secondary)

                    train_losses["loss"] += total_loss.item() * batch_size
                    train_losses["rel_l2_surf"] += surface_loss.item() * batch_size
                    train_losses["rel_l2_vol"] += volume_loss.item() * batch_size
                    train_losses["rel_l2"] += (surface_loss + volume_loss).item() * batch_size
                    train_losses["loss_supervised_primary"] += supervised_primary.item() * batch_size
                    train_losses["loss_supervised_secondary"] += supervised_secondary.item() * batch_size
                    train_losses["loss_prediction_consistency"] += pred_consistency.item() * batch_size
                    train_losses["loss_latent_consistency"] += lat_consistency.item() * batch_size
                    train_losses["secondary_inverse_density_fraction"] += secondary_inverse_density_fraction * batch_size

                    if use_surface_supervision:
                        pred_surf_primary = y1_surf_teacher * std_surf + mean_surf
                        pred_surf_secondary = y2_surf.detach() * std_surf + mean_surf
                        gt_surf = surf_data * std_surf + mean_surf
                        accumulate_channel_metrics(train_losses, "rel_l2_surf", pred_surf_primary, gt_surf, fields["surface"], rel_l2_loss_fn, batch_size, metric_weight=0.5)
                        accumulate_channel_metrics(train_losses, "rel_l2_surf", pred_surf_secondary, gt_surf, fields["surface"], rel_l2_loss_fn, batch_size, metric_weight=0.5)

                    pred_vol_primary = y1_vol_teacher * std_vol + mean_vol
                    pred_vol_secondary = y2_vol.detach() * std_vol + mean_vol
                    gt_vol = vol_data * std_vol + mean_vol
                    accumulate_channel_metrics(train_losses, "rel_l2_vol", pred_vol_primary, gt_vol, fields["volume"], rel_l2_loss_fn, batch_size, metric_weight=0.5)
                    accumulate_channel_metrics(train_losses, "rel_l2_vol", pred_vol_secondary, gt_vol, fields["volume"], rel_l2_loss_fn, batch_size, metric_weight=0.5)

                global_step += 1
                if batch_idx % log_every_n_steps == 0 or batch_idx == len(train_loader) - 1:
                    wandb.log(
                        {
                            "train/batch_loss": total_loss.item(),
                            "train/batch_rel_l2": (surface_loss + volume_loss).item(),
                            "train/batch_rel_l2_surf": surface_loss.item(),
                            "train/batch_rel_l2_vol": volume_loss.item(),
                            "train/batch_supervised_primary": supervised_primary.item(),
                            "train/batch_supervised_secondary": supervised_secondary.item(),
                            "train/batch_prediction_consistency": pred_consistency.item(),
                            "train/batch_latent_consistency": lat_consistency.item(),
                            "train/batch_secondary_inverse_density_fraction": secondary_inverse_density_fraction,
                            "train/prediction_consistency_weight": pred_consistency_weight,
                            "train/latent_consistency_weight": latent_consistency_weight,
                            "lr": scheduler.get_last_lr()[0],
                            "epoch": ep,
                        },
                        step=global_step,
                    )
                    train_pbar.set_postfix(loss=f"{total_loss.item():.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

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

            for loss_name in train_losses.keys():
                train_losses[loss_name] /= len(train_loader.dataset)

            robust_rel_l2 = 0.5 * (aligned_metrics["rel_l2"] + shifted_metrics["rel_l2"])
            robust_loss = 0.5 * (aligned_metrics["loss"] + shifted_metrics["loss"])

            if robust_rel_l2 < best_robust_rel_l2:
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
                        extra_metrics={
                            "test_aligned_metrics": aligned_metrics,
                            "test_shifted_metrics": shifted_metrics,
                            "test_robust_rel_l2": robust_rel_l2,
                        },
                    ),
                    "checkpoints/" + model_checkpoint_name + "_best.pt",
                )

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
                    extra_metrics={
                        "test_aligned_metrics": aligned_metrics,
                        "test_shifted_metrics": shifted_metrics,
                        "test_robust_rel_l2": robust_rel_l2,
                    },
                ),
                "checkpoints/" + model_checkpoint_name + "_last.pt",
            )

            t2 = default_timer()
            print(
                f"epoch: {ep}, t2-t1 (epoch time): {t2 - t1:.5f}, "
                f"train loss: {train_losses['loss']:.5f}, "
                f"aligned rel_l2: {aligned_metrics['rel_l2']:.5f}, "
                f"shifted rel_l2: {shifted_metrics['rel_l2']:.5f}, "
                f"robust rel_l2: {robust_rel_l2:.5f}"
            )

            wandb_dict = {
                "lr": scheduler.get_last_lr()[0],
                "test/robust_rel_l2": robust_rel_l2,
                "test/robust_loss": robust_loss,
                "train/prediction_consistency_weight": pred_consistency_weight,
                "train/latent_consistency_weight": latent_consistency_weight,
            }
            wandb_dict.update({f"train/{key}": value for key, value in train_losses.items()})
            wandb_dict.update({f"test_aligned/{key}": value for key, value in aligned_metrics.items()})
            wandb_dict.update({f"test_shifted/{key}": value for key, value in shifted_metrics.items()})
            add_all_field_metrics(wandb_dict, "train", fields["surface"], fields["volume"], metric_values=train_losses)
            add_all_field_metrics(wandb_dict, "test_aligned", fields["surface"], fields["volume"], metric_values=aligned_metrics)
            add_all_field_metrics(wandb_dict, "test_shifted", fields["surface"], fields["volume"], metric_values=shifted_metrics)
            add_canonical_field_metrics(wandb_dict, "train", fields["surface"], fields["volume"], metric_values=train_losses)
            add_canonical_field_metrics(wandb_dict, "test_aligned", fields["surface"], fields["volume"], metric_values=aligned_metrics)
            add_canonical_field_metrics(wandb_dict, "test_shifted", fields["surface"], fields["volume"], metric_values=shifted_metrics)
            wandb_dict["meta/training_surface_signals"] = ",".join(fields["surface"])
            wandb_dict["meta/training_volume_signals"] = ",".join(fields["volume"])
            wandb.log(wandb_dict, step=global_step)

    except KeyboardInterrupt:
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
            torch.save(emergency_state, "checkpoints/" + model_checkpoint_name + "_last.pt")
            print("Saved the latest checkpoint before exiting.")
        except Exception as exc:
            print(f"Could not save an emergency checkpoint: {exc}")
    finally:
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


@hydra.main(version_base="1.2", config_path="config", config_name="drivaerml_sat3")
def main(cfg: DictConfig):
    run_consistency_training(cfg, model_ctor=SMARTSAT3, model_requires_density=True)


if __name__ == "__main__":
    main()
    print("Training done.")
