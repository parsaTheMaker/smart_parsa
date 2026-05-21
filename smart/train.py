import hydra
from omegaconf import DictConfig

import os
import torch
import numpy as np
import wandb
from timeit import default_timer
from tqdm.auto import tqdm

# Dataset and loss functions
from data.datasets import get_dataset
from utils.utils import initialize_gpu, initialize_wandb, get_model_checkpoint_name, count_model_params, get_optimizer_scheduler_loss
from loss.losses import CombinedLoss

# SMART Model
from models.smart.smart import SMART


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


@hydra.main(version_base="1.2", config_path="config", config_name="car")
def main(cfg: DictConfig):
    # Extract config
    config = cfg.experiment
    wandb_config = cfg.wandb
    
    # Initialize WandB
    run = initialize_wandb(config, wandb_config)
    
    # Set seed and GPU settings
    device = initialize_gpu(config.random_seed, high_precision=False)
    
    # Set gradient norm clipping, precision and amp
    gradient_norm = config.gradient_norm
    precisions = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    dtype = precisions.get(config.precision, torch.float16)
    amp = config.amp
    print(gradient_norm, amp, dtype)

    # Load data
    train_data, test_data, stats, spatial_dim, surf_channels, vol_channels, params_dim, fields = get_dataset(config)
    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=config.batch_size,
                                               num_workers=config.num_workers,
                                               shuffle=True,
                                               prefetch_factor=56)
    test_loader = torch.utils.data.DataLoader(test_data,
                                              batch_size=config.batch_size,
                                              num_workers=config.num_workers,
                                              shuffle=False,
                                              prefetch_factor=56)
    # Move stats to device
    mean_surf = stats[0].to(device)
    std_surf = stats[1].to(device)
    mean_vol = stats[2].to(device)
    std_vol = stats[3].to(device)
    
    # Extract one training sample for inspection
    sample = train_data[0]
    if params_dim > 0:
        sample_geo_mesh, sample_surf_mesh, sample_surf_data, sample_vol_mesh, sample_vol_data, params = sample
    else:
        sample_geo_mesh, sample_surf_mesh, sample_surf_data, sample_vol_mesh, sample_vol_data = sample
        params = None
    print("Sample geo_mesh shape:", sample_geo_mesh.shape)
    print("Sample surf_mesh shape:", sample_surf_mesh.shape)
    print("Sample surface fields shape:", sample_surf_data.shape, "fields:", fields["surface"])
    print("Sample vol_mesh shape:", sample_vol_mesh.shape)
    print("Sample volume fields shape:", sample_vol_data.shape, "fields:", fields["volume"])
    if params is not None:
        print("Sample params shape:", params.shape)
    
    # Create model
    models = {"SMART": (SMART, {"spatial_dim": spatial_dim, "surface_channels": surf_channels, "volume_channels": vol_channels, "parameter_channels": params_dim})}
    
    if config.model_name in models:
        merged_kwargs = {**models[config.model_name][1], **config.architecture} if "architecture" in config else models[config.model_name][1]
        print(f"Model kwargs: {merged_kwargs}")
        model = models[config.model_name][0](**merged_kwargs).to(device)
    else:
        raise ValueError("Unknown model class name!")
    model = model.to(device)

    print(f"Total parameters: {count_model_params(model)}")
    model_checkpoint_name = get_model_checkpoint_name(config)
    print(f"Checkpoint name: {model_checkpoint_name}")
    
    # Monitor gradients and parameters with wandb
    run.watch(model, log="all")

    # Training and evaluation
    scaler = torch.amp.GradScaler("cuda")
    optimizer, scheduler, loss_fn, rel_l2_loss_fn = get_optimizer_scheduler_loss(model, config, train_loader, loss_dim=1)
    combined_loss_fn = CombinedLoss(loss_fn, fields)
        
    # Training loop
    loss_test_min = np.inf
    global_step = 0
    log_every_n_steps = getattr(config, "log_every_n_steps", 10)
    interrupted = False

    try:
        for ep in tqdm(range(config.epochs), desc="Epochs", dynamic_ncols=True):
            t1 = default_timer()
            train_losses = init_metric_dict(fields["surface"], fields["volume"])
            test_losses = init_metric_dict(fields["surface"], fields["volume"])

            model.train()
            train_pbar = tqdm(train_loader, desc=f"Train {ep + 1}/{config.epochs}", leave=False, dynamic_ncols=True)
            for batch_idx, batch in enumerate(train_pbar):
                # b, n, c
                if params_dim > 0:
                    geo_mesh, surf_mesh, surf_data, vol_mesh, vol_data, params = batch
                    params = params.to(device)
                else:
                    geo_mesh, surf_mesh, surf_data, vol_mesh, vol_data = batch
                    params = None

                # Move to device
                geo_mesh = geo_mesh.to(device)
                surf_mesh = surf_mesh.to(device)
                surf_data = surf_data.to(device)
                vol_mesh = vol_mesh.to(device)
                vol_data = vol_data.to(device)

                # Forward pass
                optimizer.zero_grad()

                if amp:
                    with torch.autocast(device_type=str(device).split(":")[0], dtype=dtype, enabled=True):
                        y_hat_surf, y_hat_vol = model(geo_mesh, surf_mesh, vol_mesh, params)

                        # Rel l2 loss
                        loss = combined_loss_fn(y_hat_surf, y_hat_vol, surf_data, vol_data)

                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                        scheduler.step()
                else:
                    y_hat_surf, y_hat_vol = model(geo_mesh, surf_mesh, vol_mesh, params)

                    # Rel l2 loss
                    loss = combined_loss_fn(y_hat_surf, y_hat_vol, surf_data, vol_data)
                    loss.backward()

                    # Gradient clipping
                    if gradient_norm is not None:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_norm)

                    optimizer.step()
                    scheduler.step()

                # Metrics
                batch_size = surf_data.size(0)
                train_losses["loss"] += loss.item() * batch_size
                with torch.no_grad():
                    surface_loss = rel_l2_loss_fn(y_hat_surf, surf_data)
                    volume_loss = rel_l2_loss_fn(y_hat_vol, vol_data)
                    train_losses["rel_l2_surf"] += surface_loss.item() * batch_size
                    train_losses["rel_l2_vol"] += volume_loss.item() * batch_size
                    train_losses["rel_l2"] += (surface_loss + volume_loss).item() * batch_size

                global_step += 1
                if batch_idx % log_every_n_steps == 0 or batch_idx == len(train_loader) - 1:
                    wandb.log({
                        "train/batch_loss": loss.item(),
                        "train/batch_rel_l2": (surface_loss + volume_loss).item(),
                        "lr": scheduler.get_last_lr()[0],
                        "epoch": ep,
                    }, step=global_step)
                    train_pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

            # Evaluation
            model.eval()
            test_pbar = tqdm(test_loader, desc=f"Eval  {ep + 1}/{config.epochs}", leave=False, dynamic_ncols=True)
            with torch.no_grad():
                for batch in test_pbar:
                    # b, n, c
                    if params_dim > 0:
                        geo_mesh, surf_mesh, surf_data, vol_mesh, vol_data, params = batch
                        params = params.to(device)
                    else:
                        geo_mesh, surf_mesh, surf_data, vol_mesh, vol_data = batch
                        params = None

                    # Move to device
                    geo_mesh = geo_mesh.to(device)
                    surf_mesh = surf_mesh.to(device)
                    surf_data = surf_data.to(device)
                    vol_mesh = vol_mesh.to(device)
                    vol_data = vol_data.to(device)

                    # Forward pass
                    if amp:
                        with torch.autocast(device_type=str(device).split(":")[0], dtype=dtype, enabled=True):
                            y_hat_surf, y_hat_vol = model(geo_mesh, surf_mesh, vol_mesh, params)
                    else:
                        y_hat_surf, y_hat_vol = model(geo_mesh, surf_mesh, vol_mesh, params)

                    # Denormalize
                    pred_surf = y_hat_surf[..., :] * std_surf + mean_surf
                    gt_surf = surf_data * std_surf + mean_surf
                    pred_vol = y_hat_vol[..., :] * std_vol + mean_vol
                    gt_vol = vol_data * std_vol + mean_vol

                    # Metrics
                    batch_size = surf_data.size(0)

                    # Combine loss
                    batch_loss = combined_loss_fn(y_hat_surf, y_hat_vol, surf_data, vol_data)
                    test_losses["loss"] += batch_loss.item() * batch_size

                    surface_rel_l2 = rel_l2_loss_fn(y_hat_surf, surf_data)
                    volume_rel_l2 = rel_l2_loss_fn(y_hat_vol, vol_data)
                    test_losses["rel_l2_surf"] += surface_rel_l2.item() * batch_size
                    test_losses["rel_l2_vol"] += volume_rel_l2.item() * batch_size
                    test_losses["rel_l2"] += (surface_rel_l2 + volume_rel_l2).item() * batch_size

                    accumulate_channel_metrics(test_losses, "rel_l2_surf", pred_surf, gt_surf, fields["surface"], rel_l2_loss_fn, batch_size)
                    accumulate_channel_metrics(test_losses, "rel_l2_vol", pred_vol, gt_vol, fields["volume"], rel_l2_loss_fn, batch_size)

                    test_pbar.set_postfix(loss=f"{batch_loss.item():.4f}")

            # Divide by total number of samples to get mean
            for loss_name in train_losses.keys():
                train_losses[loss_name] /= len(train_loader.dataset)
            for loss_name in test_losses.keys():
                test_losses[loss_name] /= len(test_loader.dataset)

            # Store best run
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
            # Store last run
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

            wandb_dict.update({f"train/{key}": value for key, value in train_losses.items() if value != 0.0})
            wandb_dict.update({f"test/{key}": value for key, value in test_losses.items() if value != 0.0})
            wandb.log(wandb_dict, step=global_step)

    except KeyboardInterrupt:
        interrupted = True
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
