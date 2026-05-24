import os
from timeit import default_timer

import hydra
import numpy as np
import torch
import wandb
from omegaconf import DictConfig
from tqdm.auto import tqdm

from data.datasets import get_dataset
from models.smart.cat import CAT
from utils.utils import (
    initialize_gpu,
    initialize_wandb,
    get_model_checkpoint_name,
    count_model_params,
    get_optimizer_scheduler_loss,
    apply_naca4_auto_point_budget,
    print_point_budget,
)

CANON_SURF_FIELDS = ["pressure", "normal_x", "normal_y"]
CANON_VOL_FIELDS = ["pressure", "sdf", "velocity_x", "velocity_y"]


def sample_indices(n: int, k: int, device: torch.device, disjoint_from: torch.Tensor | None = None) -> torch.Tensor:
    if k <= 0:
        return torch.empty((0,), dtype=torch.long, device=device)
    if disjoint_from is not None:
        mask = torch.ones((n,), dtype=torch.bool, device=device)
        mask[disjoint_from] = False
        candidate = torch.where(mask)[0]
        if candidate.numel() == 0:
            return torch.randint(0, n, (k,), device=device)
        if k <= candidate.numel():
            perm = torch.randperm(candidate.numel(), device=device)[:k]
            return candidate[perm]
        extra = candidate[torch.randint(0, candidate.numel(), (k - candidate.numel(),), device=device)]
        return torch.cat([candidate, extra], dim=0)
    if k <= n:
        return torch.randperm(n, device=device)[:k]
    extra = torch.randint(0, n, (k - n,), device=device)
    return torch.cat([torch.arange(n, device=device), extra], dim=0)


def gather_per_batch(x: torch.Tensor, idx_list: list[torch.Tensor]) -> torch.Tensor:
    return torch.stack([x[b, idx_list[b], :] for b in range(x.shape[0])], dim=0)


def init_metric_dict(surface_fields, volume_fields):
    metrics = {"loss": 0.0, "rel_l2": 0.0, "rel_l2_surf": 0.0, "rel_l2_vol": 0.0}
    for field_name in surface_fields:
        metrics[f"rel_l2_surf_{field_name}"] = 0.0
    for field_name in volume_fields:
        metrics[f"rel_l2_vol_{field_name}"] = 0.0
    return metrics


def add_canonical_field_metrics(wandb_dict, split, surface_fields, volume_fields, metric_values=None):
    metric_values = metric_values or {}
    for f in CANON_SURF_FIELDS:
        src_key = f"rel_l2_surf_{f}"
        wandb_dict[f"{split}/rel_l2_surf_{f}"] = metric_values.get(src_key, np.nan) if f in surface_fields else np.nan
    for f in CANON_VOL_FIELDS:
        src_key = f"rel_l2_vol_{f}"
        wandb_dict[f"{split}/rel_l2_vol_{f}"] = metric_values.get(src_key, np.nan) if f in volume_fields else np.nan


def resolve_targets(fields: dict, mean_vol: torch.Tensor, std_vol: torch.Tensor):
    surface_fields = list(fields.get("surface", []))
    if len(surface_fields) == 0:
        raise ValueError("CAT requires at least one surface field.")
    pressure_idx_s = surface_fields.index("pressure") if "pressure" in surface_fields else 0
    surface_target_indices = [pressure_idx_s]
    surface_target_fields = [surface_fields[pressure_idx_s]]

    volume_fields = list(fields.get("volume", []))
    if len(volume_fields) == 0:
        raise ValueError("CAT requires at least one volume field.")
    velocity_idx = [i for i, name in enumerate(volume_fields) if str(name).startswith("velocity_")]
    pressure_idx_v = volume_fields.index("pressure") if "pressure" in volume_fields else None

    use_pressure = False
    if pressure_idx_v is not None:
        pressure_std = float(std_vol[pressure_idx_v].item()) if pressure_idx_v < len(std_vol) else 0.0
        pressure_mean = float(mean_vol[pressure_idx_v].item()) if pressure_idx_v < len(mean_vol) else 0.0
        use_pressure = not (abs(pressure_mean) < 1e-8 and pressure_std < 1e-8)

    volume_target_indices = ([pressure_idx_v] if (pressure_idx_v is not None and use_pressure) else []) + velocity_idx
    if len(volume_target_indices) == 0:
        volume_target_indices = list(range(len(volume_fields)))

    volume_target_fields = [volume_fields[i] for i in volume_target_indices]
    return surface_target_indices, surface_target_fields, volume_target_indices, volume_target_fields


def prepare_stage_batch(stage: int, batch, config, device, surface_target_indices, volume_target_indices):
    geo_mesh, surf_mesh, surf_data, vol_mesh, vol_data = batch
    del geo_mesh
    surf_mesh = surf_mesh.to(device)
    surf_data = surf_data.to(device)
    vol_mesh = vol_mesh.to(device)
    vol_data = vol_data.to(device)

    bsz, ns, _ = surf_mesh.shape
    nv = vol_mesh.shape[1]

    s_in = int(getattr(config, "single_surface_input_points", getattr(config, "num_body_points", ns)))
    s_q = int(getattr(config, "single_surface_query_points", getattr(config, "num_surface_points", ns)))
    v_q = int(getattr(config, "single_volume_query_points", getattr(config, "num_volume_points", nv)))

    if s_in <= 0:
        s_in = ns
    if s_q <= 0:
        s_q = ns
    if v_q <= 0:
        v_q = nv

    enc_idx, surf_q_idx = [], []
    vol_q_idx = []
    for _b in range(bsz):
        e = sample_indices(ns, s_in, device)
        sq = sample_indices(ns, s_q, device, disjoint_from=e)
        enc_idx.append(e)
        surf_q_idx.append(sq)
        if stage == 2:
            vol_q_idx.append(sample_indices(nv, v_q, device))

    surface_input_tokens = gather_per_batch(surf_mesh, enc_idx)
    surface_query_tokens = gather_per_batch(surf_mesh, surf_q_idx)

    s_idx = torch.tensor(surface_target_indices, dtype=torch.long, device=surf_data.device)
    surface_target = gather_per_batch(surf_data.index_select(dim=2, index=s_idx), surf_q_idx)

    out = {
        "surface_input_tokens": surface_input_tokens,
        "surface_query_tokens": surface_query_tokens,
        "surface_target": surface_target,
    }

    if stage == 2:
        volume_query_tokens = gather_per_batch(vol_mesh, vol_q_idx)
        v_idx = torch.tensor(volume_target_indices, dtype=torch.long, device=vol_data.device)
        volume_target = gather_per_batch(vol_data.index_select(dim=2, index=v_idx), vol_q_idx)
        out["volume_query_tokens"] = volume_query_tokens
        out["volume_target"] = volume_target

    return out


def accumulate_channel_rel(metrics_dict, prefix, pred, gt, field_names, rel_l2_fn, batch_size):
    for channel_idx, field_name in enumerate(field_names):
        v = rel_l2_fn(pred[..., channel_idx:channel_idx + 1], gt[..., channel_idx:channel_idx + 1])
        metrics_dict[f"{prefix}_{field_name}"] = metrics_dict.get(f"{prefix}_{field_name}", 0.0) + v.item() * batch_size


def compute_loss(stage: int, model: CAT, stage_batch, loss_fn, rel_l2, surf_fields, vol_fields):
    zero = torch.tensor(0.0, device=stage_batch["surface_input_tokens"].device)
    if stage == 1:
        pred_s = model.forward_stage1_only(stage_batch["surface_input_tokens"], stage_batch["surface_query_tokens"], return_aux=False)
        loss_s = loss_fn(pred_s, stage_batch["surface_target"])
        rel_s = rel_l2(pred_s, stage_batch["surface_target"])
        channel_specs = [("rel_l2_surf", pred_s, stage_batch["surface_target"], surf_fields[:pred_s.shape[-1]])]
        return loss_s, loss_s, zero, rel_s, rel_s, zero, channel_specs, None

    pred_v, aux = model.forward_stage2_only(
        stage_batch["surface_input_tokens"],
        stage_batch["surface_query_tokens"],
        stage_batch["volume_query_tokens"],
        return_aux=True,
    )
    loss_v = loss_fn(pred_v, stage_batch["volume_target"])
    rel_v = rel_l2(pred_v, stage_batch["volume_target"])
    channel_specs = [("rel_l2_vol", pred_v, stage_batch["volume_target"], vol_fields[:pred_v.shape[-1]])]
    return loss_v, zero, loss_v, rel_v, zero, rel_v, channel_specs, aux


def build_model_payload(model: CAT):
    return {
        "geometry_encoder_state_dict": model.geometry_encoder.state_dict(),
        "surface_decoder_state_dict": model.surface_decoder.state_dict(),
        "stage2_head_state_dict": model.stage2_head.state_dict(),
        "surface_physics_encoder_state_dict": model.surface_physics_encoder.state_dict(),
        "volume_decoder_state_dict": model.volume_decoder.state_dict(),
        "stage3_head_state_dict": model.stage3_head.state_dict(),
        "stage3_decoder_state_dict": model.stage3_decoder.state_dict(),
    }


def load_stage1_weights(model: CAT, ckpt_path: str, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("model_state_dict", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"Loaded stage1 checkpoint from {ckpt_path}")
    if missing:
        print(f"Missing keys (allowed): {len(missing)}")
    if unexpected:
        print(f"Unexpected keys (allowed): {len(unexpected)}")


@hydra.main(version_base="1.2", config_path="config", config_name="naca4_cat")
def main(cfg: DictConfig):
    config = cfg.experiment
    wandb_config = cfg.wandb

    stage = int(getattr(config, "cat_stage", 1))
    if stage not in (1, 2):
        raise ValueError("cat_stage must be 1 or 2")

    run = initialize_wandb(config, wandb_config)
    device = initialize_gpu(config.random_seed, high_precision=False)

    gradient_norm = config.gradient_norm
    precisions = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    dtype = precisions.get(config.precision, torch.float16)
    amp = config.amp

    train_data, test_data, stats, spatial_dim, surf_channels, vol_channels, params_dim, fields = get_dataset(config)
    point_info = apply_naca4_auto_point_budget(config, train_data, for_cat=True)
    if point_info is not None:
        print_point_budget("CAT", point_info)
        train_data, test_data, stats, spatial_dim, surf_channels, vol_channels, params_dim, fields = get_dataset(config)

    s_idx, s_fields, v_idx, v_fields = resolve_targets(fields, train_data.mean_vol_data, train_data.std_vol_data)
    if stage == 1:
        vol_channels = len(v_idx)
        vol_signals = []
    else:
        vol_channels = len(v_idx)
        vol_signals = v_fields

    if params_dim > 0:
        raise NotImplementedError("CAT train script currently supports params_dim=0 datasets.")

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=config.batch_size, num_workers=config.num_workers, shuffle=True, prefetch_factor=56)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=config.batch_size, num_workers=config.num_workers, shuffle=False, prefetch_factor=56)

    model = CAT(
        spatial_dim=spatial_dim,
        surface_channels=surf_channels,
        volume_channels=vol_channels,
        parameter_channels=params_dim,
        **(dict(config.architecture) if "architecture" in config else {}),
    ).to(device)

    if stage == 2:
        stage1_ckpt = str(getattr(config, "stage2_stage1_ckpt", "")).strip()
        if not stage1_ckpt:
            raise ValueError("Stage 2 requires experiment.stage2_stage1_ckpt")
        load_stage1_weights(model, stage1_ckpt, device)
        model.freeze_stage1()
        print("Stage 1 modules frozen for stage 2 training.")

    model_checkpoint_name = f"{get_model_checkpoint_name(config)}-cat-stage{stage}"
    print(f"Total parameters: {count_model_params(model)}")
    print(f"Checkpoint name: {model_checkpoint_name}")

    run.watch(model, log="all")

    optimizer, scheduler, loss_fn, rel_l2 = get_optimizer_scheduler_loss(model, config, train_loader, loss_dim=1)
    scaler = torch.amp.GradScaler("cuda")

    loss_test_min = np.inf
    global_step = 0
    log_every_n_steps = int(getattr(config, "log_every_n_steps", 10))

    try:
        for ep in tqdm(range(config.epochs), desc="Epochs", dynamic_ncols=True):
            t1 = default_timer()
            model.train()
            train_metrics = init_metric_dict(s_fields, vol_signals)
            test_metrics = init_metric_dict(s_fields, vol_signals)

            train_pbar = tqdm(train_loader, desc=f"Train S{stage} {ep + 1}/{config.epochs}", leave=False, dynamic_ncols=True)
            for batch_idx, batch in enumerate(train_pbar):
                stage_batch = prepare_stage_batch(stage, batch, config, device, s_idx, v_idx)
                optimizer.zero_grad()

                if amp:
                    with torch.autocast(device_type=str(device).split(":")[0], dtype=dtype, enabled=True):
                        loss, loss_s, loss_v, rel, rel_s, rel_v, channel_specs, aux = compute_loss(stage, model, stage_batch, loss_fn, rel_l2, s_fields, vol_signals)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                else:
                    loss, loss_s, loss_v, rel, rel_s, rel_v, channel_specs, aux = compute_loss(stage, model, stage_batch, loss_fn, rel_l2, s_fields, vol_signals)
                    loss.backward()
                    if gradient_norm is not None:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_norm)
                    optimizer.step()
                    scheduler.step()

                bsz = stage_batch["surface_input_tokens"].shape[0]
                train_metrics["loss"] += loss.item() * bsz
                train_metrics["rel_l2"] += rel.item() * bsz
                train_metrics["rel_l2_surf"] += rel_s.item() * bsz
                train_metrics["rel_l2_vol"] += rel_v.item() * bsz

                for prefix, pred_ch, gt_ch, names in channel_specs:
                    accumulate_channel_rel(train_metrics, prefix, pred_ch, gt_ch, names, rel_l2, bsz)

                global_step += 1
                if batch_idx % log_every_n_steps == 0 or batch_idx == len(train_loader) - 1:
                    batch_log = {
                        "train/batch_loss": loss.item(),
                        "train/batch_rel_l2": rel.item(),
                        "train/batch_rel_l2_surf": rel_s.item(),
                        "train/batch_rel_l2_vol": rel_v.item(),
                        "train/batch_surface_loss": loss_s.item(),
                        "train/batch_volume_loss": loss_v.item(),
                        "lr": scheduler.get_last_lr()[0],
                        "epoch": ep,
                        "stage": stage,
                    }
                    if aux is not None:
                        batch_log["model/surface_to_volume_skip_weight_mean"] = float(aux["skip_weight_mean"].detach().item())
                        batch_log["model/surface_to_volume_skip_weight_std"] = float(aux["skip_weight_std"].detach().item())
                    wandb.log(batch_log, step=global_step)
                    train_pbar.set_postfix(loss=f"{loss.item():.4f}", rel=f"{rel.item():.4f}")

            model.eval()
            with torch.no_grad():
                test_pbar = tqdm(test_loader, desc=f"Eval  S{stage} {ep + 1}/{config.epochs}", leave=False, dynamic_ncols=True)
                for batch in test_pbar:
                    stage_batch = prepare_stage_batch(stage, batch, config, device, s_idx, v_idx)
                    if amp:
                        with torch.autocast(device_type=str(device).split(":")[0], dtype=dtype, enabled=True):
                            loss, _ls, _lv, rel, rel_s, rel_v, channel_specs, _aux = compute_loss(stage, model, stage_batch, loss_fn, rel_l2, s_fields, vol_signals)
                    else:
                        loss, _ls, _lv, rel, rel_s, rel_v, channel_specs, _aux = compute_loss(stage, model, stage_batch, loss_fn, rel_l2, s_fields, vol_signals)

                    bsz = stage_batch["surface_input_tokens"].shape[0]
                    test_metrics["loss"] += loss.item() * bsz
                    test_metrics["rel_l2"] += rel.item() * bsz
                    test_metrics["rel_l2_surf"] += rel_s.item() * bsz
                    test_metrics["rel_l2_vol"] += rel_v.item() * bsz
                    for prefix, pred_ch, gt_ch, names in channel_specs:
                        accumulate_channel_rel(test_metrics, prefix, pred_ch, gt_ch, names, rel_l2, bsz)

            for k in train_metrics:
                train_metrics[k] /= len(train_loader.dataset)
            for k in test_metrics:
                test_metrics[k] /= len(test_loader.dataset)

            if test_metrics["rel_l2"] < loss_test_min:
                loss_test_min = test_metrics["rel_l2"]
                best_payload = {
                    "epoch": ep,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "loss": test_metrics["loss"],
                    "rel_l2_loss": test_metrics["rel_l2"],
                    "stage": stage,
                }
                best_payload.update(build_model_payload(model))
                torch.save(best_payload, f"checkpoints/{model_checkpoint_name}_best.pt")

            last_payload = {
                "epoch": ep,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "loss": test_metrics["loss"],
                "rel_l2_loss": test_metrics["rel_l2"],
                "stage": stage,
            }
            last_payload.update(build_model_payload(model))
            torch.save(last_payload, f"checkpoints/{model_checkpoint_name}_last.pt")

            t2 = default_timer()
            print(f"stage: {stage}, epoch: {ep}, epoch_time: {t2 - t1:.4f}, train_rel: {train_metrics['rel_l2']:.5f}, test_rel: {test_metrics['rel_l2']:.5f}")

            epoch_log = {
                "train/loss": train_metrics["loss"],
                "train/rel_l2": train_metrics["rel_l2"],
                "train/rel_l2_surf": train_metrics["rel_l2_surf"],
                "train/rel_l2_vol": train_metrics["rel_l2_vol"],
                "test/loss": test_metrics["loss"],
                "test/rel_l2": test_metrics["rel_l2"],
                "test/rel_l2_surf": test_metrics["rel_l2_surf"],
                "test/rel_l2_vol": test_metrics["rel_l2_vol"],
                "lr": scheduler.get_last_lr()[0],
                "epoch": ep,
                "stage": stage,
            }
            if stage == 2:
                w = torch.clamp(model.surface_to_volume_skip_weights.detach(), min=0.0, max=1.0)
                epoch_log["model/surface_to_volume_skip_weight_mean"] = float(w.mean().item())
                epoch_log["model/surface_to_volume_skip_weight_std"] = float(w.std(unbiased=False).item())
            add_canonical_field_metrics(epoch_log, "train", s_fields, vol_signals, metric_values=train_metrics)
            add_canonical_field_metrics(epoch_log, "test", s_fields, vol_signals, metric_values=test_metrics)
            wandb.log(epoch_log, step=global_step)

    finally:
        best_ckpt = f"checkpoints/{model_checkpoint_name}_best.pt"
        last_ckpt = f"checkpoints/{model_checkpoint_name}_last.pt"
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
    print("CAT training done.")
