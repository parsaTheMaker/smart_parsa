from __future__ import annotations

import os
from timeit import default_timer

import hydra
import numpy as np
import torch
import wandb
from omegaconf import DictConfig
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


def init_metric_dict(surface_fields, volume_fields):
    metrics = {"loss": 0.0, "rel_l2": 0.0, "rel_l2_surf": 0.0, "rel_l2_vol": 0.0}
    for field_name in surface_fields:
        metrics[f"rel_l2_surf_{field_name}"] = 0.0
    for field_name in volume_fields:
        metrics[f"rel_l2_vol_{field_name}"] = 0.0
    return metrics


def accumulate_channel_metrics(metrics, prefix, pred, gt, field_names, rel_l2_loss_fn, batch_size):
    for channel_idx, field_name in enumerate(field_names):
        channel_loss = rel_l2_loss_fn(pred[..., channel_idx:channel_idx + 1], gt[..., channel_idx:channel_idx + 1])
        metrics[f"{prefix}_{field_name}"] += channel_loss.item() * batch_size


def add_canonical_field_metrics(wandb_dict, split, surface_fields, volume_fields, metric_values=None):
    metric_values = metric_values or {}
    for f in CANON_SURF_FIELDS:
        if f not in surface_fields:
            continue
        src_key = f"rel_l2_surf_{f}"
        wandb_dict[f"{split}/rel_l2_surf_{f}"] = metric_values.get(src_key, np.nan)
    for f in CANON_VOL_FIELDS:
        if f not in volume_fields:
            continue
        src_key = f"rel_l2_vol_{f}"
        wandb_dict[f"{split}/rel_l2_vol_{f}"] = metric_values.get(src_key, np.nan)


def add_all_field_metrics(wandb_dict, split, surface_fields, volume_fields, metric_values=None):
    metric_values = metric_values or {}
    for f in surface_fields:
        wandb_dict[f"{split}/rel_l2_surf_{f}"] = metric_values.get(f"rel_l2_surf_{f}", np.nan)
    for f in volume_fields:
        wandb_dict[f"{split}/rel_l2_vol_{f}"] = metric_values.get(f"rel_l2_vol_{f}", np.nan)


def load_partial_state_dict(model, checkpoint_path, device):
    if not checkpoint_path:
        return 0, 0
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    source = checkpoint.get("model_state_dict", checkpoint)
    target_model = model.module if hasattr(model, "module") else model
    target = target_model.state_dict()
    if any(key.startswith("module.") for key in source.keys()):
        source = {key.removeprefix("module."): value for key, value in source.items()}
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
    target_model.load_state_dict(target, strict=False)
    print(f"[resume] Loaded {matched} tensors from {checkpoint_path}; skipped {skipped} incompatible tensors.")
    return matched, skipped


def load_full_training_state(model, optimizer, scheduler, scaler, checkpoint_path, device, steps_per_epoch=None):
    """Restore a vanilla trainer checkpoint without replaying completed epochs."""
    if not checkpoint_path:
        raise ValueError("checkpoint_path must be provided for full-state resume.")
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    source = checkpoint.get("model_state_dict")
    if source is None:
        raise KeyError(f"Checkpoint {checkpoint_path} does not contain model_state_dict.")

    target_model = model.module if hasattr(model, "module") else model
    if any(key.startswith("module.") for key in source):
        source = {key.removeprefix("module."): value for key, value in source.items()}
    target_model.load_state_dict(source, strict=True)

    optimizer_state = checkpoint.get("optimizer_state_dict")
    scheduler_state = checkpoint.get("scheduler_state_dict")
    if optimizer_state is None or scheduler_state is None:
        raise KeyError(
            f"Checkpoint {checkpoint_path} is missing optimizer_state_dict or scheduler_state_dict."
        )
    optimizer.load_state_dict(optimizer_state)
    scheduler.load_state_dict(scheduler_state)

    scaler_state = checkpoint.get("scaler_state_dict")
    if scaler_state is not None:
        scaler.load_state_dict(scaler_state)

    resumed_epoch = int(checkpoint.get("epoch", -1))
    start_epoch = resumed_epoch + 1
    global_step = checkpoint.get("global_step")
    if global_step is None:
        global_step = (resumed_epoch + 1) * int(steps_per_epoch or 0)
    global_step = int(global_step)
    best_rel_l2 = float(checkpoint.get("best_rel_l2", checkpoint.get("rel_l2_loss", np.inf)))
    print(
        f"[resume] Restored full training state from {checkpoint_path}: "
        f"epoch={resumed_epoch}, next_epoch={start_epoch}, global_step={global_step}, "
        f"best_rel_l2={best_rel_l2:.6g}"
    )
    return start_epoch, global_step, best_rel_l2


def _parse_batch(batch, params_dim):
    geo_log_density = None
    if params_dim > 0:
        if len(batch) == 7:
            geo_mesh, surf_mesh, surf_data, vol_mesh, vol_data, params, geo_log_density = batch
        elif len(batch) == 6:
            geo_mesh, surf_mesh, surf_data, vol_mesh, vol_data, params = batch
        else:
            raise ValueError(f"Unexpected batch size {len(batch)}")
    else:
        if len(batch) == 6:
            geo_mesh, surf_mesh, surf_data, vol_mesh, vol_data, geo_log_density = batch
            params = None
        elif len(batch) == 5:
            geo_mesh, surf_mesh, surf_data, vol_mesh, vol_data = batch
            params = None
        else:
            raise ValueError(f"Unexpected batch size {len(batch)}")
    return geo_mesh, surf_mesh, surf_data, vol_mesh, vol_data, params, geo_log_density


def _move_to_device(value, device):
    if torch.is_tensor(value):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {key: _move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_move_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_to_device(item, device) for item in value)
    return value


def _record_stream(value, stream):
    if torch.is_tensor(value):
        if value.is_cuda:
            value.record_stream(stream)
        return
    if isinstance(value, dict):
        for item in value.values():
            _record_stream(item, stream)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _record_stream(item, stream)


class CudaPrefetchLoader:
    """Overlap host-to-device copies with the current training step."""

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
                next_batch = _move_to_device(batch, self.device)

        preload()
        while next_batch is not None:
            torch.cuda.current_stream(device=self.device).wait_stream(self.stream)
            batch = next_batch
            _record_stream(batch, torch.cuda.current_stream(device=self.device))
            preload()
            yield batch


def run_surface_volume_training(cfg: DictConfig, model_cls, accepts_geo_log_density=False):
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

    prefetch_factor = int(getattr(config, "prefetch_factor", 2))
    pin_memory = bool(getattr(config, "pin_memory", True))
    dl_common = dict(batch_size=config.batch_size, num_workers=config.num_workers, pin_memory=pin_memory)
    if config.num_workers > 0:
        dl_common["prefetch_factor"] = prefetch_factor
        dl_common["persistent_workers"] = True

    train_loader = torch.utils.data.DataLoader(train_data, shuffle=True, **dl_common)
    test_loader = torch.utils.data.DataLoader(test_data, shuffle=False, **dl_common)
    cuda_batch_prefetch = bool(getattr(config, "cuda_batch_prefetch", False)) and device.type == "cuda"
    train_batch_source = CudaPrefetchLoader(train_loader, device) if cuda_batch_prefetch else train_loader
    test_batch_source = CudaPrefetchLoader(test_loader, device) if cuda_batch_prefetch else test_loader
    print(f"[dataloader] cuda_batch_prefetch={cuda_batch_prefetch}")

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
    model = model_cls(**merged_kwargs).to(device)

    resume_ckpt = str(getattr(config, "resume_ckpt", "")).strip()
    init_ckpt = str(getattr(config, "init_ckpt", "")).strip()
    resume_full_state = bool(getattr(config, "resume_full_state", False))
    if resume_full_state and not resume_ckpt:
        raise ValueError("resume_full_state=True requires experiment.resume_ckpt to be set.")
    if resume_full_state:
        pass
    elif resume_ckpt:
        print(f"[init] Loading model weights from experiment.resume_ckpt={resume_ckpt}")
        load_partial_state_dict(model, resume_ckpt, device)
    elif init_ckpt:
        print(f"[init] Loading model weights from experiment.init_ckpt={init_ckpt}")
        load_partial_state_dict(model, init_ckpt, device)

    print(f"Total parameters: {count_model_params(model)}")
    model_checkpoint_name = get_model_checkpoint_name(config)
    print(f"Checkpoint name: {model_checkpoint_name}")
    if bool(getattr(config, "wandb_watch_model", False)):
        run.watch(model, log="all")

    scaler = torch.amp.GradScaler("cuda")
    optimizer, scheduler, loss_fn, rel_l2_loss_fn = get_optimizer_scheduler_loss(model, config, train_loader, loss_dim=1)
    combined_loss_fn = CombinedLoss(loss_fn, fields) if use_surface_supervision else None

    loss_test_min = np.inf
    global_step = 0
    start_epoch = 0
    if resume_full_state:
        start_epoch, global_step, loss_test_min = load_full_training_state(
            model,
            optimizer,
            scheduler,
            scaler,
            resume_ckpt,
            device,
            steps_per_epoch=len(train_loader),
        )
    log_every_n_steps = getattr(config, "log_every_n_steps", 10)

    try:
        for ep in tqdm(range(start_epoch, config.epochs), desc="Epochs", dynamic_ncols=True):
            t1 = default_timer()
            # Propagate the epoch to datasets that use epoch-seeded point sampling.
            if hasattr(train_data, "set_epoch"):
                train_data.set_epoch(ep)
            if hasattr(test_data, "set_epoch"):
                test_data.set_epoch(0)
            train_losses = init_metric_dict(fields["surface"], fields["volume"])
            test_losses = init_metric_dict(fields["surface"], fields["volume"])

            model.train()
            train_pbar = tqdm(train_batch_source, desc=f"Train {ep + 1}/{config.epochs}", leave=False, dynamic_ncols=True)
            for batch_idx, batch in enumerate(train_pbar):
                geo_mesh, surf_mesh, surf_data, vol_mesh, vol_data, params, geo_log_density = _parse_batch(batch, params_dim)
                geo_mesh = geo_mesh.to(device)
                surf_mesh = surf_mesh.to(device)
                surf_data = surf_data.to(device)
                vol_mesh = vol_mesh.to(device)
                vol_data = vol_data.to(device)
                if params is not None:
                    params = params.to(device)
                if geo_log_density is not None:
                    geo_log_density = geo_log_density.to(device)

                if config.dataset == "NACA4":
                    surf_data = surf_data[..., :1]
                    vol_data = torch.cat([vol_data[..., :1], vol_data[..., 2:4]], dim=-1)

                optimizer.zero_grad(set_to_none=True)
                if amp:
                    with torch.autocast(device_type=str(device).split(":")[0], dtype=dtype, enabled=True):
                        if accepts_geo_log_density:
                            y_hat_surf, y_hat_vol = model(geo_mesh, surf_mesh, vol_mesh, params, geo_log_density=geo_log_density)
                        else:
                            y_hat_surf, y_hat_vol = model(geo_mesh, surf_mesh, vol_mesh, params)
                        # Keep the large pointwise reductions in float32. A float16
                        # sum over 65k query points can overflow and produce NaN
                        # gradients even when the forward loss is finite.
                        loss = (
                            combined_loss_fn(y_hat_surf.float(), y_hat_vol.float(), surf_data.float(), vol_data.float())
                            if use_surface_supervision
                            else loss_fn(y_hat_vol.float(), vol_data.float())
                        )
                    if not torch.isfinite(loss):
                        print(f"[warn] Non-finite training loss at epoch {ep} batch {batch_idx}; skipping optimizer step.")
                        optimizer.zero_grad(set_to_none=True)
                        continue
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
                    if accepts_geo_log_density:
                        y_hat_surf, y_hat_vol = model(geo_mesh, surf_mesh, vol_mesh, params, geo_log_density=geo_log_density)
                    else:
                        y_hat_surf, y_hat_vol = model(geo_mesh, surf_mesh, vol_mesh, params)
                    loss = (
                        combined_loss_fn(y_hat_surf.float(), y_hat_vol.float(), surf_data.float(), vol_data.float())
                        if use_surface_supervision
                        else loss_fn(y_hat_vol.float(), vol_data.float())
                    )
                    if not torch.isfinite(loss):
                        print(f"[warn] Non-finite training loss at epoch {ep} batch {batch_idx}; skipping optimizer step.")
                        optimizer.zero_grad(set_to_none=True)
                        continue
                    loss.backward()
                    if gradient_norm is not None:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_norm)
                    optimizer.step()
                    scheduler.step()

                batch_size = surf_data.size(0)
                train_losses["loss"] += loss.item() * batch_size
                with torch.no_grad():
                    surface_loss = rel_l2_loss_fn(y_hat_surf.float(), surf_data.float()) if use_surface_supervision else torch.tensor(0.0, device=device)
                    volume_loss = rel_l2_loss_fn(y_hat_vol.float(), vol_data.float())
                    train_losses["rel_l2_surf"] += surface_loss.item() * batch_size
                    train_losses["rel_l2_vol"] += volume_loss.item() * batch_size
                    train_losses["rel_l2"] += (surface_loss + volume_loss).item() * batch_size
                    if use_surface_supervision:
                        pred_surf_train = y_hat_surf[..., :] * std_surf + mean_surf
                        gt_surf_train = surf_data * std_surf + mean_surf
                        accumulate_channel_metrics(train_losses, "rel_l2_surf", pred_surf_train, gt_surf_train, fields["surface"], rel_l2_loss_fn, batch_size)
                    pred_vol_train = y_hat_vol[..., :] * std_vol + mean_vol
                    gt_vol_train = vol_data * std_vol + mean_vol
                    accumulate_channel_metrics(train_losses, "rel_l2_vol", pred_vol_train, gt_vol_train, fields["volume"], rel_l2_loss_fn, batch_size)

                global_step += 1
                if batch_idx % log_every_n_steps == 0 or batch_idx == len(train_loader) - 1:
                    wandb.log({
                        "train/batch_loss": loss.item(),
                        "train/batch_rel_l2": (surface_loss + volume_loss).item(),
                        "train/batch_rel_l2_surf": surface_loss.item(),
                        "train/batch_rel_l2_vol": volume_loss.item(),
                        "lr": scheduler.get_last_lr()[0],
                        "epoch": ep,
                    }, step=global_step)
                    train_pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

            model.eval()
            test_pbar = tqdm(test_batch_source, desc=f"Eval  {ep + 1}/{config.epochs}", leave=False, dynamic_ncols=True)
            with torch.no_grad():
                for batch in test_pbar:
                    geo_mesh, surf_mesh, surf_data, vol_mesh, vol_data, params, geo_log_density = _parse_batch(batch, params_dim)
                    geo_mesh = geo_mesh.to(device)
                    surf_mesh = surf_mesh.to(device)
                    surf_data = surf_data.to(device)
                    vol_mesh = vol_mesh.to(device)
                    vol_data = vol_data.to(device)
                    if params is not None:
                        params = params.to(device)
                    if geo_log_density is not None:
                        geo_log_density = geo_log_density.to(device)

                    if config.dataset == "NACA4":
                        surf_data = surf_data[..., :1]
                        vol_data = torch.cat([vol_data[..., :1], vol_data[..., 2:4]], dim=-1)

                    if amp:
                        with torch.autocast(device_type=str(device).split(":")[0], dtype=dtype, enabled=True):
                            if accepts_geo_log_density:
                                y_hat_surf, y_hat_vol = model(geo_mesh, surf_mesh, vol_mesh, params, geo_log_density=geo_log_density)
                            else:
                                y_hat_surf, y_hat_vol = model(geo_mesh, surf_mesh, vol_mesh, params)
                    else:
                        if accepts_geo_log_density:
                            y_hat_surf, y_hat_vol = model(geo_mesh, surf_mesh, vol_mesh, params, geo_log_density=geo_log_density)
                        else:
                            y_hat_surf, y_hat_vol = model(geo_mesh, surf_mesh, vol_mesh, params)

                    if use_surface_supervision:
                        pred_surf = y_hat_surf[..., :] * std_surf + mean_surf
                        gt_surf = surf_data * std_surf + mean_surf
                    pred_vol = y_hat_vol[..., :] * std_vol + mean_vol
                    gt_vol = vol_data * std_vol + mean_vol

                    batch_size = surf_data.size(0)
                    if use_surface_supervision:
                        batch_loss = combined_loss_fn(y_hat_surf.float(), y_hat_vol.float(), surf_data.float(), vol_data.float())
                        surface_rel_l2 = rel_l2_loss_fn(y_hat_surf.float(), surf_data.float())
                    else:
                        batch_loss = loss_fn(y_hat_vol.float(), vol_data.float())
                        surface_rel_l2 = torch.tensor(0.0, device=device)
                    test_losses["loss"] += batch_loss.item() * batch_size

                    volume_rel_l2 = rel_l2_loss_fn(y_hat_vol.float(), vol_data.float())
                    test_losses["rel_l2_surf"] += surface_rel_l2.item() * batch_size
                    test_losses["rel_l2_vol"] += volume_rel_l2.item() * batch_size
                    test_losses["rel_l2"] += (surface_rel_l2 + volume_rel_l2).item() * batch_size
                    if use_surface_supervision:
                        accumulate_channel_metrics(test_losses, "rel_l2_surf", pred_surf, gt_surf, fields["surface"], rel_l2_loss_fn, batch_size)
                    accumulate_channel_metrics(test_losses, "rel_l2_vol", pred_vol, gt_vol, fields["volume"], rel_l2_loss_fn, batch_size)
                    test_pbar.set_postfix(loss=f"{batch_loss.item():.4f}")

            for loss_name in train_losses.keys():
                train_losses[loss_name] /= len(train_loader.dataset)
            for loss_name in test_losses.keys():
                test_losses[loss_name] /= len(test_loader.dataset)

            if test_losses["rel_l2"] < loss_test_min:
                loss_test_min = test_losses["rel_l2"]
                torch.save({
                    "epoch": ep,
                    "global_step": global_step,
                    "best_rel_l2": loss_test_min,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                    "loss": test_losses["loss"],
                    "rel_l2_loss": test_losses["rel_l2"],
                    "surface_fields": fields["surface"],
                    "volume_fields": fields["volume"],
                    "metric_values": {k: v for k, v in test_losses.items() if k.startswith("rel_l2")},
                }, "checkpoints/" + model_checkpoint_name + "_best.pt")

            torch.save({
                "epoch": ep,
                "global_step": global_step,
                "best_rel_l2": loss_test_min,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "loss": test_losses["loss"],
                "rel_l2_loss": test_losses["rel_l2"],
                "surface_fields": fields["surface"],
                "volume_fields": fields["volume"],
                "metric_values": {k: v for k, v in test_losses.items() if k.startswith("rel_l2")},
            }, "checkpoints/" + model_checkpoint_name + "_last.pt")

            t2 = default_timer()
            print(f"epoch: {ep}, t2-t1 (epoch time): {t2-t1:.5f}, train loss: {train_losses['loss']:.5f}, test loss: {test_losses['loss']:.5f}")
            wandb_dict = {"lr": scheduler.get_last_lr()[0]}
            wandb_dict.update({f"train/{key}": value for key, value in train_losses.items()})
            wandb_dict.update({f"test/{key}": value for key, value in test_losses.items()})
            add_all_field_metrics(wandb_dict, "train", fields["surface"], fields["volume"], metric_values=train_losses)
            add_all_field_metrics(wandb_dict, "test", fields["surface"], fields["volume"], metric_values=test_losses)
            add_canonical_field_metrics(wandb_dict, "train", fields["surface"], fields["volume"], metric_values=train_losses)
            add_canonical_field_metrics(wandb_dict, "test", fields["surface"], fields["volume"], metric_values=test_losses)
            wandb_dict["meta/training_surface_signals"] = ",".join(fields["surface"])
            wandb_dict["meta/training_volume_signals"] = ",".join(fields["volume"])
            wandb.log(wandb_dict, step=global_step)
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
