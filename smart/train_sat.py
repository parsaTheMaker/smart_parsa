import hydra
from omegaconf import DictConfig

import os
import torch
import numpy as np
import wandb
from timeit import default_timer
from tqdm.auto import tqdm

from data.datasets import get_dataset
from utils.utils import initialize_gpu, initialize_wandb, get_model_checkpoint_name, count_model_params, get_optimizer_scheduler_loss, apply_naca4_auto_point_budget, print_point_budget
from loss.losses import CombinedLoss

from models.smart.smart import SMART
from models.smart.smart_sat import SMARTSAT

CANON_SURF_FIELDS = ["pressure", "normal_x", "normal_y"]
CANON_VOL_FIELDS = ["pressure", "sdf", "velocity_x", "velocity_y"]


def init_metric_dict(surface_fields, volume_fields):
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
    return metrics


def accumulate_channel_metrics(metrics, prefix, pred, gt, field_names, rel_l2_loss_fn, batch_size):
    for channel_idx, field_name in enumerate(field_names):
        channel_loss = rel_l2_loss_fn(pred[..., channel_idx:channel_idx + 1], gt[..., channel_idx:channel_idx + 1])
        metrics[f"{prefix}_{field_name}"] += channel_loss.item() * batch_size


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


@hydra.main(version_base="1.2", config_path="config", config_name="drivaerml_sat")
def main(cfg: DictConfig):
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
        if config.model_name in {"SMART", "SMART_SAT"} and config.dataset == "NACA4":
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
    if config.model_name in {"SMART", "SMART_SAT"} and config.dataset == "NACA4" and vol_channels == 2:
        mean_vol = stats[2][2:4].to(device)
        std_vol = stats[3][2:4].to(device)
    elif config.model_name in {"SMART", "SMART_SAT"} and config.dataset == "NACA4" and vol_channels == 3:
        mean_vol = torch.stack([stats[2][0], stats[2][2], stats[2][3]]).to(device)
        std_vol = torch.stack([stats[3][0], stats[3][2], stats[3][3]]).to(device)
    else:
        mean_vol = stats[2][:vol_channels].to(device)
        std_vol = stats[3][:vol_channels].to(device)

    models = {
        "SMART": (SMART, {"spatial_dim": spatial_dim, "surface_channels": surf_channels, "volume_channels": vol_channels, "parameter_channels": params_dim}),
        "SMART_SAT": (SMARTSAT, {"spatial_dim": spatial_dim, "surface_channels": surf_channels, "volume_channels": vol_channels, "parameter_channels": params_dim}),
    }

    if config.model_name in models:
        merged_kwargs = {**models[config.model_name][1], **config.architecture} if "architecture" in config else models[config.model_name][1]
        print(f"Model kwargs: {merged_kwargs}")
        model = models[config.model_name][0](**merged_kwargs).to(device)
    else:
        raise ValueError("Unknown model class name!")
    model = model.to(device)

    resume_ckpt = str(getattr(config, "resume_ckpt", "")).strip()
    if resume_ckpt:
        load_partial_state_dict(model, resume_ckpt, device)

    print(f"Total parameters: {count_model_params(model)}")
    model_checkpoint_name = get_model_checkpoint_name(config)
    print(f"Checkpoint name: {model_checkpoint_name}")
    run.watch(model, log="all")

    scaler = torch.amp.GradScaler("cuda")
    optimizer, scheduler, loss_fn, rel_l2_loss_fn = get_optimizer_scheduler_loss(model, config, train_loader, loss_dim=1)
    combined_loss_fn = CombinedLoss(loss_fn, fields) if use_surface_supervision else None

    loss_test_min = np.inf
    global_step = 0
    log_every_n_steps = getattr(config, "log_every_n_steps", 10)

    try:
        for ep in tqdm(range(config.epochs), desc="Epochs", dynamic_ncols=True):
            t1 = default_timer()
            train_losses = init_metric_dict(fields["surface"], fields["volume"])
            test_losses = init_metric_dict(fields["surface"], fields["volume"])

            model.train()
            train_pbar = tqdm(train_loader, desc=f"Train {ep + 1}/{config.epochs}", leave=False, dynamic_ncols=True)
            for batch_idx, batch in enumerate(train_pbar):
                geo_log_density = None
                if params_dim > 0:
                    if len(batch) == 7:
                        geo_mesh, surf_mesh, surf_data, vol_mesh, vol_data, params, geo_log_density = batch
                        geo_log_density = geo_log_density.to(device)
                    else:
                        geo_mesh, surf_mesh, surf_data, vol_mesh, vol_data, params = batch
                    params = params.to(device)
                else:
                    if len(batch) == 6:
                        geo_mesh, surf_mesh, surf_data, vol_mesh, vol_data, geo_log_density = batch
                        geo_log_density = geo_log_density.to(device)
                    else:
                        geo_mesh, surf_mesh, surf_data, vol_mesh, vol_data = batch
                    params = None

                geo_mesh = geo_mesh.to(device)
                surf_mesh = surf_mesh.to(device)
                surf_data = surf_data.to(device)
                vol_mesh = vol_mesh.to(device)
                vol_data = vol_data.to(device)

                if config.model_name in {"SMART", "SMART_SAT"} and config.dataset == "NACA4":
                    surf_data = surf_data[..., :1]
                    vol_data = torch.cat([vol_data[..., :1], vol_data[..., 2:4]], dim=-1)

                optimizer.zero_grad()

                if amp:
                    with torch.autocast(device_type=str(device).split(":")[0], dtype=dtype, enabled=True):
                        if config.model_name == "SMART_SAT":
                            y_hat_surf, y_hat_vol = model(geo_mesh, surf_mesh, vol_mesh, params, geo_log_density=geo_log_density)
                        else:
                            y_hat_surf, y_hat_vol = model(geo_mesh, surf_mesh, vol_mesh, params)
                        loss = combined_loss_fn(y_hat_surf, y_hat_vol, surf_data, vol_data) if use_surface_supervision else loss_fn(y_hat_vol, vol_data)
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                        scheduler.step()
                else:
                    if config.model_name == "SMART_SAT":
                        y_hat_surf, y_hat_vol = model(geo_mesh, surf_mesh, vol_mesh, params, geo_log_density=geo_log_density)
                    else:
                        y_hat_surf, y_hat_vol = model(geo_mesh, surf_mesh, vol_mesh, params)
                    loss = combined_loss_fn(y_hat_surf, y_hat_vol, surf_data, vol_data) if use_surface_supervision else loss_fn(y_hat_vol, vol_data)
                    loss.backward()
                    if gradient_norm is not None:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_norm)
                    optimizer.step()
                    scheduler.step()

                batch_size = surf_data.size(0)
                train_losses["loss"] += loss.item() * batch_size
                with torch.no_grad():
                    surface_loss = rel_l2_loss_fn(y_hat_surf, surf_data) if use_surface_supervision else torch.tensor(0.0, device=device)
                    volume_loss = rel_l2_loss_fn(y_hat_vol, vol_data)
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
            test_pbar = tqdm(test_loader, desc=f"Eval  {ep + 1}/{config.epochs}", leave=False, dynamic_ncols=True)
            with torch.no_grad():
                for batch in test_pbar:
                    geo_log_density = None
                    if params_dim > 0:
                        if len(batch) == 7:
                            geo_mesh, surf_mesh, surf_data, vol_mesh, vol_data, params, geo_log_density = batch
                            geo_log_density = geo_log_density.to(device)
                        else:
                            geo_mesh, surf_mesh, surf_data, vol_mesh, vol_data, params = batch
                        params = params.to(device)
                    else:
                        if len(batch) == 6:
                            geo_mesh, surf_mesh, surf_data, vol_mesh, vol_data, geo_log_density = batch
                            geo_log_density = geo_log_density.to(device)
                        else:
                            geo_mesh, surf_mesh, surf_data, vol_mesh, vol_data = batch
                        params = None

                    geo_mesh = geo_mesh.to(device)
                    surf_mesh = surf_mesh.to(device)
                    surf_data = surf_data.to(device)
                    vol_mesh = vol_mesh.to(device)
                    vol_data = vol_data.to(device)

                    if config.model_name in {"SMART", "SMART_SAT"} and config.dataset == "NACA4":
                        surf_data = surf_data[..., :1]
                        vol_data = torch.cat([vol_data[..., :1], vol_data[..., 2:4]], dim=-1)

                    if amp:
                        with torch.autocast(device_type=str(device).split(":")[0], dtype=dtype, enabled=True):
                            if config.model_name == "SMART_SAT":
                                y_hat_surf, y_hat_vol = model(geo_mesh, surf_mesh, vol_mesh, params, geo_log_density=geo_log_density)
                            else:
                                y_hat_surf, y_hat_vol = model(geo_mesh, surf_mesh, vol_mesh, params)
                    else:
                        if config.model_name == "SMART_SAT":
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
                        batch_loss = combined_loss_fn(y_hat_surf, y_hat_vol, surf_data, vol_data)
                        surface_rel_l2 = rel_l2_loss_fn(y_hat_surf, surf_data)
                    else:
                        batch_loss = loss_fn(y_hat_vol, vol_data)
                        surface_rel_l2 = torch.tensor(0.0, device=device)
                    test_losses["loss"] += batch_loss.item() * batch_size

                    volume_rel_l2 = rel_l2_loss_fn(y_hat_vol, vol_data)
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
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "loss": test_losses["loss"],
                    "rel_l2_loss": test_losses["rel_l2"],
                    "surface_fields": fields["surface"],
                    "volume_fields": fields["volume"],
                    "metric_values": {k: v for k, v in test_losses.items() if k.startswith("rel_l2")},
                }, "checkpoints/" + model_checkpoint_name + "_best.pt")

            torch.save({
                "epoch": ep,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
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

    except KeyboardInterrupt:
        print("\nTraining interrupted by user (Ctrl+C). Saving current state and exiting cleanly...")
        try:
            emergency_state = {
                "epoch": locals().get("ep", -1),
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
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


if __name__ == "__main__":
    main()
    print("Training done.")
