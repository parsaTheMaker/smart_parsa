import os
from timeit import default_timer

import hydra
import numpy as np
import torch
import wandb
from omegaconf import DictConfig
from tqdm.auto import tqdm

from data.datasets import get_dataset
from loss.losses import RelL2Loss
from models.smart.cat import CAT
from utils.utils import initialize_gpu, initialize_wandb, get_model_checkpoint_name, count_model_params, get_optimizer_scheduler_loss, apply_naca4_auto_point_budget, print_point_budget

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
        # Fallback only when k > candidate size.
        extra = candidate[torch.randint(0, candidate.numel(), (k - candidate.numel(),), device=device)]
        return torch.cat([candidate, extra], dim=0)

    if k <= n:
        return torch.randperm(n, device=device)[:k]
    # Fallback only when k > n.
    extra = torch.randint(0, n, (k - n,), device=device)
    return torch.cat([torch.arange(n, device=device), extra], dim=0)


def gather_per_batch(x: torch.Tensor, idx_list: list[torch.Tensor]) -> torch.Tensor:
    return torch.stack([x[b, idx_list[b], :] for b in range(x.shape[0])], dim=0)





def add_canonical_field_metrics(wandb_dict, split, surface_fields, volume_fields, metric_values=None):
    metric_values = metric_values or {}
    for f in CANON_SURF_FIELDS:
        src_key = f"rel_l2_surf_{f}"
        wandb_dict[f"{split}/rel_l2_surf_{f}"] = metric_values.get(src_key, np.nan) if f in surface_fields else np.nan
    for f in CANON_VOL_FIELDS:
        src_key = f"rel_l2_vol_{f}"
        wandb_dict[f"{split}/rel_l2_vol_{f}"] = metric_values.get(src_key, np.nan) if f in volume_fields else np.nan


def resolve_stage3_volume_targets(fields: dict, mean_vol: torch.Tensor, std_vol: torch.Tensor):
    vol_fields = list(fields.get("volume", []))
    if len(vol_fields) == 0:
        raise ValueError("No volume fields available for CAT stage3.")

    velocity_idx = [i for i, name in enumerate(vol_fields) if str(name).startswith("velocity_")]
    if len(velocity_idx) == 0:
        raise ValueError(f"CAT stage3 requires velocity channels in volume fields, got: {vol_fields}")

    pressure_idx = vol_fields.index("pressure") if "pressure" in vol_fields else None
    use_pressure = False
    if pressure_idx is not None:
        pressure_std = float(std_vol[pressure_idx].item()) if pressure_idx < len(std_vol) else 0.0
        pressure_mean = float(mean_vol[pressure_idx].item()) if pressure_idx < len(mean_vol) else 0.0
        use_pressure = not (abs(pressure_mean) < 1e-8 and pressure_std < 1e-8)

    target_indices = ([pressure_idx] if use_pressure else []) + velocity_idx
    target_fields = [vol_fields[i] for i in target_indices]
    return target_indices, target_fields, use_pressure

def prepare_stage_batch(stage: int, batch, config, device: torch.device, stage3_target_indices=None):
    geo_mesh, surf_mesh, surf_data, vol_mesh, vol_data = batch
    geo_mesh = geo_mesh.to(device)
    surf_mesh = surf_mesh.to(device)
    surf_data = surf_data.to(device)
    vol_mesh = vol_mesh.to(device)
    vol_data = vol_data.to(device)

    bsz, ns, _ = surf_mesh.shape
    nv = vol_mesh.shape[1]

    if stage == 1:
        s_in = int(getattr(config, "stage1_surface_input_points", config.num_body_points))
        s_q = int(getattr(config, "stage1_surface_query_points", config.num_surface_points))
        v_q = int(getattr(config, "stage1_volume_query_points", min(config.num_volume_points, 4 * config.num_surface_points)))
        s_attr = int(getattr(config, "stage1_surface_attr_channels", 2))
        v_attr = int(getattr(config, "stage1_volume_attr_channels", 1))

        enc_idx, surf_q_idx, vol_q_idx = [], [], []
        for b in range(bsz):
            e = sample_indices(ns, s_in, device)
            sq = sample_indices(ns, s_q, device, disjoint_from=e)
            vq = sample_indices(nv, v_q, device)
            enc_idx.append(e)
            surf_q_idx.append(sq)
            vol_q_idx.append(vq)

        surface_input = gather_per_batch(surf_mesh, enc_idx)
        surf_query = gather_per_batch(surf_mesh, surf_q_idx)
        vol_query = gather_per_batch(vol_mesh, vol_q_idx)

        # stage1 surface attribute defaults to normals: surf_data[:, :, 1:3]
        surf_target = gather_per_batch(surf_data[:, :, 1 : 1 + s_attr], surf_q_idx)
        # stage1 volume attribute defaults to sdf: vol_data[:, :, 1:2]
        vol_target = gather_per_batch(vol_data[:, :, 1 : 1 + v_attr], vol_q_idx)
        query_points = torch.cat([surf_query, vol_query], dim=1)

        return {
            "surface_input": surface_input,
            "query_points": query_points,
            "surf_target": surf_target,
            "vol_target": vol_target,
            "surf_query_count": surf_query.shape[1],
            "surf_attr_channels": s_attr,
            "vol_attr_channels": v_attr,
        }

    if stage == 2:
        s_in = int(getattr(config, "stage2_surface_input_points", config.num_body_points))
        s_q = int(getattr(config, "stage2_surface_query_points", config.num_surface_points))
        s_field = int(getattr(config, "stage2_surface_channels", 1))

        enc_idx, surf_q_idx = [], []
        for b in range(bsz):
            e = sample_indices(ns, s_in, device)
            sq = sample_indices(ns, s_q, device, disjoint_from=e)
            enc_idx.append(e)
            surf_q_idx.append(sq)

        surface_input = gather_per_batch(surf_mesh, enc_idx)
        surf_query = gather_per_batch(surf_mesh, surf_q_idx)
        # stage2 defaults to pressure: surf_data[:, :, 0:1]
        surf_target = gather_per_batch(surf_data[:, :, 0:s_field], surf_q_idx)

        return {
            "surface_input": surface_input,
            "surf_query": surf_query,
            "surf_target": surf_target,
        }

    # stage 3
    s_in = int(getattr(config, "stage3_surface_input_points", config.num_body_points))
    v_q = int(getattr(config, "stage3_volume_query_points", config.num_volume_points))

    enc_idx, vol_q_idx = [], []
    for b in range(bsz):
        e = sample_indices(ns, s_in, device)
        vq = sample_indices(nv, v_q, device)
        enc_idx.append(e)
        vol_q_idx.append(vq)

    surface_input = gather_per_batch(surf_mesh, enc_idx)
    vol_query = gather_per_batch(vol_mesh, vol_q_idx)
    if stage3_target_indices is not None:
        idx = torch.tensor(stage3_target_indices, dtype=torch.long, device=vol_data.device)
        vol_target_data = vol_data.index_select(dim=2, index=idx)
    else:
        vol_target_data = vol_data

    vol_target = gather_per_batch(vol_target_data, vol_q_idx)

    return {
        "surface_input": surface_input,
        "vol_query": vol_query,
        "vol_target": vol_target,
    }


def accumulate_channel_rel(metrics_dict, prefix, pred, gt, field_names, rel_l2_fn, batch_size):
    if pred is None or gt is None:
        return
    for channel_idx, field_name in enumerate(field_names):
        v = rel_l2_fn(pred[..., channel_idx:channel_idx + 1], gt[..., channel_idx:channel_idx + 1])
        metrics_dict[f"{prefix}_{field_name}"] = metrics_dict.get(f"{prefix}_{field_name}", 0.0) + v.item() * batch_size


def compute_stage_loss(model: CAT, stage_batch: dict, stage: int, loss_fn, rel_l2, surf_signals, vol_signals):
    zero = torch.tensor(0.0, device=stage_batch["surface_input"].device)

    if stage == 1:
        pred = model.forward_stage1(stage_batch["surface_input"], stage_batch["query_points"])
        qs = stage_batch["surf_query_count"]
        s_attr = stage_batch["surf_attr_channels"]
        v_attr = stage_batch["vol_attr_channels"]

        pred_surf = pred[:, :qs, :s_attr]
        pred_vol = pred[:, qs:, s_attr : s_attr + v_attr]
        loss_surf = loss_fn(pred_surf, stage_batch["surf_target"])
        loss_vol = loss_fn(pred_vol, stage_batch["vol_target"])
        loss = loss_surf + loss_vol
        rel_surf = rel_l2(pred_surf, stage_batch["surf_target"])
        rel_vol = rel_l2(pred_vol, stage_batch["vol_target"])
        rel = rel_surf + rel_vol
        channel_specs = [
            ("rel_l2_surf", pred_surf, stage_batch["surf_target"], surf_signals[:pred_surf.shape[-1]]),
            ("rel_l2_vol", pred_vol, stage_batch["vol_target"], vol_signals[:pred_vol.shape[-1]]),
        ]
        return loss, rel, rel_surf, rel_vol, channel_specs

    if stage == 2:
        pred = model.forward_stage2(stage_batch["surface_input"], stage_batch["surf_query"])
        loss = loss_fn(pred, stage_batch["surf_target"])
        rel_surf = rel_l2(pred, stage_batch["surf_target"])
        rel = rel_surf
        channel_specs = [
            ("rel_l2_surf", pred, stage_batch["surf_target"], surf_signals[:pred.shape[-1]]),
        ]
        return loss, rel, rel_surf, zero, channel_specs

    pred = model.forward_stage3(stage_batch["surface_input"], stage_batch["vol_query"])
    loss = loss_fn(pred, stage_batch["vol_target"])
    rel_vol = rel_l2(pred, stage_batch["vol_target"])
    rel = rel_vol
    channel_specs = [
        ("rel_l2_vol", pred, stage_batch["vol_target"], vol_signals[:pred.shape[-1]]),
    ]
    return loss, rel, zero, rel_vol, channel_specs


def load_encoder_from_checkpoint(model: CAT, ckpt_path: str, which: str):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if which == "geometry":
        if "geometry_encoder_state_dict" in ckpt:
            model.geometry_encoder.load_state_dict(ckpt["geometry_encoder_state_dict"], strict=True)
            return
        if "model_state_dict" in ckpt:
            sub = {k.replace("geometry_encoder.", "", 1): v for k, v in ckpt["model_state_dict"].items() if k.startswith("geometry_encoder.")}
            if sub:
                model.geometry_encoder.load_state_dict(sub, strict=True)
                return
    if which == "surface":
        if "surface_encoder_state_dict" in ckpt:
            model.surface_encoder.load_state_dict(ckpt["surface_encoder_state_dict"], strict=True)
            return
        if "model_state_dict" in ckpt:
            sub = {k.replace("surface_encoder.", "", 1): v for k, v in ckpt["model_state_dict"].items() if k.startswith("surface_encoder.")}
            if sub:
                model.surface_encoder.load_state_dict(sub, strict=True)
                return

    raise ValueError(f"Could not load {which} encoder from checkpoint: {ckpt_path}")


def build_stage_payload(model: CAT, stage: int):
    payload = {}
    if stage == 1:
        payload["geometry_encoder_state_dict"] = model.geometry_encoder.state_dict()
        payload["stage1_decoder_state_dict"] = model.stage1_decoder.state_dict()
        payload["stage1_head_state_dict"] = model.stage1_head.state_dict()
    elif stage == 2:
        payload["surface_encoder_state_dict"] = model.surface_encoder.state_dict()
        payload["stage2_decoder_state_dict"] = model.stage2_decoder.state_dict()
        payload["stage2_head_state_dict"] = model.stage2_head.state_dict()
    else:
        payload["geometry_encoder_state_dict"] = model.geometry_encoder.state_dict()
        payload["surface_encoder_state_dict"] = model.surface_encoder.state_dict()
        if hasattr(model, "fusion"):
            payload["fusion_state_dict"] = model.fusion.state_dict()
        if hasattr(model, "stage3_decoder"):
            payload["stage3_decoder_state_dict"] = model.stage3_decoder.state_dict()
        if hasattr(model, "stage3_decoder_geom"):
            payload["stage3_decoder_geom_state_dict"] = model.stage3_decoder_geom.state_dict()
            # Backward-compatible alias for tooling expecting single key.
            payload["stage3_decoder_state_dict"] = model.stage3_decoder_geom.state_dict()
        if hasattr(model, "stage3_decoder_surf"):
            payload["stage3_decoder_surf_state_dict"] = model.stage3_decoder_surf.state_dict()
        payload["stage3_head_state_dict"] = model.stage3_head.state_dict()
    return payload




def load_stage3_resume_state(model: CAT, optimizer, scheduler, ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)
    if "model_state_dict" not in ckpt:
        raise ValueError(f"Resume checkpoint has no model_state_dict: {ckpt_path}")

    model.load_state_dict(ckpt["model_state_dict"], strict=True)

    if "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    else:
        print("Warning: resume checkpoint missing optimizer_state_dict; optimizer is freshly initialized.")

    if "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    else:
        print("Warning: resume checkpoint missing scheduler_state_dict; scheduler is freshly initialized.")

    start_epoch = int(ckpt.get("epoch", -1)) + 1
    best_rel = float(ckpt.get("rel_l2_loss", np.inf)) if ckpt.get("rel_l2_loss", None) is not None else np.inf
    print(f"Resumed from checkpoint: {ckpt_path}")
    print(f"Resume start epoch: {start_epoch}")
    return start_epoch, best_rel

@hydra.main(version_base="1.2", config_path="config", config_name="naca4_cat")
def main(cfg: DictConfig):
    config = cfg.experiment
    wandb_config = cfg.wandb

    stage = int(getattr(config, "cat_stage", 1))
    if stage not in (1, 2, 3):
        raise ValueError("cat_stage must be 1, 2, or 3")

    resume_stage3_ckpt = str(getattr(config, "resume_stage3_ckpt", "")).strip()

    if getattr(config, "model_tag", "") in (None, ""):
        config.model_tag = f"stage{stage}"

    run = initialize_wandb(config, wandb_config)
    device = initialize_gpu(config.random_seed, high_precision=False)

    gradient_norm = config.gradient_norm
    precisions = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    dtype = precisions.get(config.precision, torch.float16)
    amp = config.amp

    train_data, test_data, stats, spatial_dim, surf_channels, vol_channels, params_dim, fields = get_dataset(config)
    stage3_target_indices = None
    stage3_target_fields = []
    stage3_use_pressure = False

    point_info = apply_naca4_auto_point_budget(config, train_data, for_cat=True)
    if point_info is not None:
        print_point_budget("CAT", point_info)
        # Rebuild datasets with resolved point counts.
        train_data, test_data, stats, spatial_dim, surf_channels, vol_channels, params_dim, fields = get_dataset(config)

    if stage == 3:
        stage3_target_indices, stage3_target_fields, stage3_use_pressure = resolve_stage3_volume_targets(
            fields,
            train_data.mean_vol_data,
            train_data.std_vol_data,
        )
        vol_channels = len(stage3_target_indices)
        print(f"CAT stage3 target fields: {stage3_target_fields}")

    if stage == 1:
        surf_signals = ["normal_x", "normal_y"]
        vol_signals = ["sdf"]
    elif stage == 2:
        surf_signals = ["pressure"]
        vol_signals = []
    else:
        surf_signals = []
        vol_signals = stage3_target_fields
    print(f"[CAT] stage {stage} training signals -> surface: {surf_signals} | volume: {vol_signals}")

    if params_dim > 0:
        raise NotImplementedError("CAT train script currently supports params_dim=0 datasets.")

    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=True,
        prefetch_factor=56,
    )
    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=False,
        prefetch_factor=56,
    )

    models = {
        "CAT": (
            CAT,
            {
                "spatial_dim": spatial_dim,
                "surface_channels": surf_channels,
                "volume_channels": vol_channels,
                "parameter_channels": params_dim,
            },
        )
    }

    if config.model_name not in models:
        raise ValueError("Unknown model class name for CAT train script")

    merged_kwargs = {**models[config.model_name][1], **config.architecture} if "architecture" in config else models[config.model_name][1]
    print(f"CAT stage {stage} model kwargs: {merged_kwargs}")
    model = models[config.model_name][0](**merged_kwargs).to(device)

    if stage == 3:
        geom_ckpt = str(getattr(config, "stage3_geometry_ckpt", ""))
        surf_ckpt = str(getattr(config, "stage3_surface_ckpt", ""))
        if not geom_ckpt or not surf_ckpt:
            raise ValueError("Stage 3 requires stage3_geometry_ckpt and stage3_surface_ckpt in config.")
        load_encoder_from_checkpoint(model, geom_ckpt, which="geometry")
        load_encoder_from_checkpoint(model, surf_ckpt, which="surface")
        for p in model.geometry_encoder.parameters():
            p.requires_grad = True
        for p in model.surface_encoder.parameters():
            p.requires_grad = True
        print(f"Loaded encoders from: {geom_ckpt} and {surf_ckpt}")
        print("Stage3 encoders are fully trainable (LoRA disabled).")

    model_checkpoint_name = get_model_checkpoint_name(config)
    model_checkpoint_name = f"{model_checkpoint_name}-cat-stage{stage}"

    print(f"Total parameters: {count_model_params(model)}")
    print(f"Checkpoint name: {model_checkpoint_name}")

    run.watch(model, log="all")

    optimizer, scheduler, loss_fn, rel_l2 = get_optimizer_scheduler_loss(model, config, train_loader, loss_dim=1)
    scaler = torch.amp.GradScaler("cuda")

    loss_test_min = np.inf
    start_epoch = 0
    if stage == 3 and resume_stage3_ckpt:
        start_epoch, loss_test_min = load_stage3_resume_state(model, optimizer, scheduler, resume_stage3_ckpt, device)

    global_step = start_epoch * len(train_loader)
    log_every_n_steps = int(getattr(config, "log_every_n_steps", 10))

    try:
        for ep in tqdm(range(start_epoch, config.epochs), desc="Epochs", dynamic_ncols=True):
            t1 = default_timer()
            model.train()
            train_loss_sum = 0.0
            train_rel_sum = 0.0
            train_rel_surf_sum = 0.0
            train_rel_vol_sum = 0.0
            train_channel_metrics = {}
            test_channel_metrics = {}

            train_pbar = tqdm(train_loader, desc=f"Train S{stage} {ep + 1}/{config.epochs}", leave=False, dynamic_ncols=True)
            for batch_idx, batch in enumerate(train_pbar):
                stage_batch = prepare_stage_batch(stage, batch, config, device, stage3_target_indices=stage3_target_indices)
                optimizer.zero_grad()

                if amp:
                    with torch.autocast(device_type=str(device).split(":")[0], dtype=dtype, enabled=True):
                        loss, rel, rel_surf, rel_vol, channel_specs = compute_stage_loss(model, stage_batch, stage, loss_fn, rel_l2, surf_signals, vol_signals)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                else:
                    loss, rel, rel_surf, rel_vol, channel_specs = compute_stage_loss(model, stage_batch, stage, loss_fn, rel_l2, surf_signals, vol_signals)
                    loss.backward()
                    if gradient_norm is not None:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_norm)
                    optimizer.step()
                    scheduler.step()

                bsz = next(iter(stage_batch.values())).shape[0]
                train_loss_sum += loss.item() * bsz
                train_rel_sum += rel.item() * bsz
                train_rel_surf_sum += rel_surf.item() * bsz
                train_rel_vol_sum += rel_vol.item() * bsz

                for prefix, pred_ch, gt_ch, names in channel_specs:
                    accumulate_channel_rel(train_channel_metrics, prefix, pred_ch, gt_ch, names, rel_l2, bsz)

                global_step += 1
                if batch_idx % log_every_n_steps == 0 or batch_idx == len(train_loader) - 1:
                    batch_log = {
                        "train/batch_loss": loss.item(),
                        "train/batch_rel_l2": rel.item(),
                        "train/batch_rel_l2_surf": rel_surf.item(),
                        "train/batch_rel_l2_vol": rel_vol.item(),
                        "lr": scheduler.get_last_lr()[0],
                        "epoch": ep,
                        "stage": stage,
                    }
                    batch_metric_values = {}
                    for prefix, pred_ch, gt_ch, names in channel_specs:
                        for channel_idx, field_name in enumerate(names):
                            v = rel_l2(pred_ch[..., channel_idx:channel_idx + 1], gt_ch[..., channel_idx:channel_idx + 1])
                            batch_metric_values[f"{prefix}_{field_name}"] = v.item()
                    add_canonical_field_metrics(batch_log, "train", surf_signals, vol_signals, metric_values=batch_metric_values)
                    wandb.log(batch_log, step=global_step)
                    train_pbar.set_postfix(loss=f"{loss.item():.4f}", rel=f"{rel.item():.4f}")

            model.eval()
            test_loss_sum = 0.0
            test_rel_sum = 0.0
            test_rel_surf_sum = 0.0
            test_rel_vol_sum = 0.0
            with torch.no_grad():
                test_pbar = tqdm(test_loader, desc=f"Eval  S{stage} {ep + 1}/{config.epochs}", leave=False, dynamic_ncols=True)
                for batch in test_pbar:
                    stage_batch = prepare_stage_batch(stage, batch, config, device, stage3_target_indices=stage3_target_indices)
                    if amp:
                        with torch.autocast(device_type=str(device).split(":")[0], dtype=dtype, enabled=True):
                            loss, rel, rel_surf, rel_vol, channel_specs = compute_stage_loss(model, stage_batch, stage, loss_fn, rel_l2, surf_signals, vol_signals)
                    else:
                        loss, rel, rel_surf, rel_vol, channel_specs = compute_stage_loss(model, stage_batch, stage, loss_fn, rel_l2, surf_signals, vol_signals)

                    bsz = next(iter(stage_batch.values())).shape[0]
                    test_loss_sum += loss.item() * bsz
                    test_rel_sum += rel.item() * bsz
                    test_rel_surf_sum += rel_surf.item() * bsz
                    test_rel_vol_sum += rel_vol.item() * bsz

                    for prefix, pred_ch, gt_ch, names in channel_specs:
                        accumulate_channel_rel(test_channel_metrics, prefix, pred_ch, gt_ch, names, rel_l2, bsz)

                    test_pbar.set_postfix(loss=f"{loss.item():.4f}", rel=f"{rel.item():.4f}")

            train_loss = train_loss_sum / len(train_loader.dataset)
            train_rel = train_rel_sum / len(train_loader.dataset)
            train_rel_surf = train_rel_surf_sum / len(train_loader.dataset)
            train_rel_vol = train_rel_vol_sum / len(train_loader.dataset)
            test_loss = test_loss_sum / len(test_loader.dataset)
            test_rel = test_rel_sum / len(test_loader.dataset)
            test_rel_surf = test_rel_surf_sum / len(test_loader.dataset)
            test_rel_vol = test_rel_vol_sum / len(test_loader.dataset)

            for k in list(train_channel_metrics.keys()):
                train_channel_metrics[k] /= len(train_loader.dataset)
            for k in list(test_channel_metrics.keys()):
                test_channel_metrics[k] /= len(test_loader.dataset)

            if test_rel < loss_test_min:
                loss_test_min = test_rel
                best_payload = {
                    "epoch": ep,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "loss": test_loss,
                    "rel_l2_loss": test_rel,
                    "stage": stage,
                    "stage3_target_fields": stage3_target_fields,
                    "stage3_use_pressure": stage3_use_pressure,
                }
                best_payload.update(build_stage_payload(model, stage))
                torch.save(best_payload, f"checkpoints/{model_checkpoint_name}_best.pt")

            last_payload = {
                "epoch": ep,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "loss": test_loss,
                "rel_l2_loss": test_rel,
                "stage": stage,
                "stage3_target_fields": stage3_target_fields,
                "stage3_use_pressure": stage3_use_pressure,
            }
            last_payload.update(build_stage_payload(model, stage))
            torch.save(last_payload, f"checkpoints/{model_checkpoint_name}_last.pt")

            t2 = default_timer()
            print(
                f"stage: {stage}, epoch: {ep}, epoch_time: {t2 - t1:.4f}, "
                f"train_loss: {train_loss:.5f}, train_rel: {train_rel:.5f}, "
                f"test_loss: {test_loss:.5f}, test_rel: {test_rel:.5f}"
            )

            epoch_log = {
                "train/loss": train_loss,
                "train/rel_l2": train_rel,
                "train/rel_l2_surf": train_rel_surf,
                "train/rel_l2_vol": train_rel_vol,
                "test/loss": test_loss,
                "test/rel_l2": test_rel,
                "test/rel_l2_surf": test_rel_surf,
                "test/rel_l2_vol": test_rel_vol,
                "lr": scheduler.get_last_lr()[0],
                "epoch": ep,
                "stage": stage,
                "meta/training_surface_signals": ",".join(surf_signals),
                "meta/training_volume_signals": ",".join(vol_signals),
            }
            add_canonical_field_metrics(epoch_log, "train", surf_signals, vol_signals, metric_values=train_channel_metrics)
            add_canonical_field_metrics(epoch_log, "test", surf_signals, vol_signals, metric_values=test_channel_metrics)
            wandb.log(epoch_log, step=global_step)

    except KeyboardInterrupt:
        print("\nTraining interrupted by user (Ctrl+C). Saving last checkpoint...")
        emergency = {
            "epoch": locals().get("ep", -1),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "stage": stage,
            "stage3_target_fields": stage3_target_fields,
            "stage3_use_pressure": stage3_use_pressure,
        }
        emergency.update(build_stage_payload(model, stage))
        torch.save(emergency, f"checkpoints/{model_checkpoint_name}_last.pt")

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
