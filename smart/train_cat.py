import os
from timeit import default_timer

import hydra
import numpy as np
import torch
import wandb
from omegaconf import DictConfig
from omegaconf import open_dict
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
        if k == candidate.numel():
            return candidate
        if k <= candidate.numel():
            perm = torch.randperm(candidate.numel(), device=device)[:k]
            return candidate[perm]
        extra = candidate[torch.randint(0, candidate.numel(), (k - candidate.numel(),), device=device)]
        return torch.cat([candidate, extra], dim=0)
    if k == n:
        return torch.arange(n, device=device)
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


def add_all_field_metrics(wandb_dict, split, surface_fields, volume_fields, metric_values=None):
    metric_values = metric_values or {}
    for f in surface_fields:
        src_key = f"rel_l2_surf_{f}"
        wandb_dict[f"{split}/rel_l2_surf_{f}"] = metric_values.get(src_key, np.nan)
    for f in volume_fields:
        src_key = f"rel_l2_vol_{f}"
        wandb_dict[f"{split}/rel_l2_vol_{f}"] = metric_values.get(src_key, np.nan)


def resolve_targets(fields: dict, mean_vol: torch.Tensor, std_vol: torch.Tensor):
    surface_fields = list(fields.get("surface", []))
    if len(surface_fields) == 0:
        raise ValueError("CAT requires at least one surface field.")
    # Stage-1 should learn surface pressure + wall-shear channels when available.
    preferred_surface = ["pressure", "normal_x", "normal_y", "normal_z", "wall_shear_x", "wall_shear_y", "wall_shear_z"]
    surface_target_indices = [surface_fields.index(name) for name in preferred_surface if name in surface_fields]
    if len(surface_target_indices) == 0:
        surface_target_indices = list(range(len(surface_fields)))
    surface_target_fields = [surface_fields[i] for i in surface_target_indices]

    volume_fields = list(fields.get("volume", []))
    if len(volume_fields) == 0:
        raise ValueError("CAT requires at least one volume field.")
    velocity_idx = [i for i, name in enumerate(volume_fields) if str(name).startswith("velocity_")]
    pressure_idx_v = volume_fields.index("pressure") if "pressure" in volume_fields else None
    # Stage-2 should learn volume pressure + velocity when available.
    volume_target_indices = ([pressure_idx_v] if pressure_idx_v is not None else []) + velocity_idx
    if len(volume_target_indices) == 0:
        volume_target_indices = list(range(len(volume_fields)))

    volume_target_fields = [volume_fields[i] for i in volume_target_indices]
    return surface_target_indices, surface_target_fields, volume_target_indices, volume_target_fields


def prepare_stage_batch(stage: int, batch, config, device, surface_target_indices, volume_target_indices):
    geo_mesh, surf_mesh, surf_data, vol_mesh, vol_data = batch
    geo_mesh = geo_mesh.to(device)
    surf_mesh = surf_mesh.to(device)
    surf_data = surf_data.to(device)
    vol_mesh = vol_mesh.to(device)
    vol_data = vol_data.to(device)

    bsz, ng, _ = geo_mesh.shape
    ns = surf_mesh.shape[1]
    nv = vol_mesh.shape[1]

    num_body_points = int(getattr(config, "num_body_points", ng))
    s_in = int(getattr(config, "single_surface_input_points", num_body_points))
    # Keep CAT query handling as close to SMART as possible: use the dataset-provided
    # surface/volume query tensors directly instead of introducing a second query cap.
    # The dataset already applied num_surface_points / num_volume_points.
    s_q = ns
    v_q = nv

    # Match SMART semantics: when the dataset is asked for full geometry, CAT should
    # also consume the full geometry pool for its encoder input, regardless of any
    # stale single-surface-input cap left in the config.
    if num_body_points <= 0:
        s_in = ng
    if s_in <= 0:
        s_in = ng
    enc_idx, surf_q_idx = [], []
    vol_q_idx = []
    for _b in range(bsz):
        e = sample_indices(ng, s_in, device)
        # Geometry inputs and surface queries may come from different pools/sizes,
        # so disjoint sampling only makes sense when they share the same index space.
        disjoint = e if ng == ns else None
        sq = sample_indices(ns, s_q, device, disjoint_from=disjoint)
        enc_idx.append(e)
        surf_q_idx.append(sq)
        if stage == 2:
            vol_q_idx.append(sample_indices(nv, v_q, device))

    surface_input_tokens = gather_per_batch(geo_mesh, enc_idx)
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
        "volume_head_state_dict": model.volume_head.state_dict(),
        "stage3_head_state_dict": model.volume_head.state_dict(),
        "volume_decoder_state_dict": model.volume_decoder.state_dict(),
        "stage3_decoder_state_dict": model.volume_decoder.state_dict(),
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
    # Keep CAT head dimensions aligned with selected targets.
    with open_dict(config.architecture):
        config.architecture.stage2_surface_channels = len(s_idx)
    if stage == 1:
        vol_channels = len(v_idx)
        vol_signals = []
    else:
        vol_channels = len(v_idx)
        vol_signals = v_fields

    if params_dim > 0:
        raise NotImplementedError("CAT train script currently supports params_dim=0 datasets.")

    prefetch_factor = int(getattr(config, "prefetch_factor", 2))
    pin_memory = bool(getattr(config, "pin_memory", True))
    dl_common = dict(
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
    )
    if config.num_workers > 0:
        dl_common["prefetch_factor"] = prefetch_factor
        dl_common["persistent_workers"] = True

    train_loader = torch.utils.data.DataLoader(
        train_data,
        shuffle=True,
        **dl_common,
    )
    test_loader = torch.utils.data.DataLoader(
        test_data,
        shuffle=False,
        **dl_common,
    )

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
        with torch.no_grad():
            model.surface_to_volume_skip_couple_weights.fill_(0.05)
            model.surface_to_volume_skip_fuse_weights.fill_(0.05)
        model.freeze_stage1()
        print("Stage 1 modules frozen for stage 2 training.")
        print("Reset surface_to_volume_skip_couple_weights and surface_to_volume_skip_fuse_weights to 0.05 after stage-1 checkpoint load.")

    model_checkpoint_name = f"{get_model_checkpoint_name(config)}-cat-stage{stage}"
    print(f"Total parameters: {count_model_params(model)}")
    print(f"Checkpoint name: {model_checkpoint_name}")

    run.watch(model, log="all")

    optimizer, scheduler, loss_fn, rel_l2 = get_optimizer_scheduler_loss(model, config, train_loader, loss_dim=1)
    scaler = torch.amp.GradScaler("cuda")

    loss_test_min = np.inf
    global_step = 0
    start_epoch = 0
    log_every_n_steps = int(getattr(config, "log_every_n_steps", 10))

    resume_ckpt = str(getattr(config, "resume_ckpt", "")).strip()
    resume_reset_scheduler = bool(getattr(config, "resume_reset_scheduler", False))
    resume_reset_optimizer = bool(getattr(config, "resume_reset_optimizer", False))
    if resume_ckpt:
        if not os.path.isfile(resume_ckpt):
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_ckpt}")
        resume_payload = torch.load(resume_ckpt, map_location=device)
        state = resume_payload.get("model_state_dict", resume_payload)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            print(f"[resume] Missing keys: {len(missing)}")
        if unexpected:
            print(f"[resume] Unexpected keys: {len(unexpected)}")
        if (not resume_reset_optimizer) and "optimizer_state_dict" in resume_payload:
            optimizer.load_state_dict(resume_payload["optimizer_state_dict"])
        if (not resume_reset_scheduler) and "scheduler_state_dict" in resume_payload:
            scheduler.load_state_dict(resume_payload["scheduler_state_dict"])
        if "scaler_state_dict" in resume_payload:
            scaler.load_state_dict(resume_payload["scaler_state_dict"])
        start_epoch = int(resume_payload.get("epoch", -1)) + 1
        loss_test_min = float(resume_payload.get("rel_l2_loss", np.inf))
        global_step = start_epoch * len(train_loader)
        print(f"Resumed stage {stage} from {resume_ckpt}")
        print(f"Resume state -> start_epoch={start_epoch}, global_step={global_step}, best_rel_l2={loss_test_min}")
        if resume_reset_scheduler:
            # Start a fresh LR schedule window for this continuation run.
            start_epoch = 0
            global_step = 0
            base_lr = float(config.learning_rate)
            for pg in optimizer.param_groups:
                pg["lr"] = base_lr
            print("[resume] Scheduler reset requested: starting LR schedule from initial value.")
            print(f"[resume] Optimizer param-group lr reset to {base_lr:.6g}.")
        if resume_reset_optimizer:
            print("[resume] Optimizer reset requested: using freshly initialized optimizer state.")

    try:
        for ep in tqdm(range(start_epoch, config.epochs), desc="Epochs", dynamic_ncols=True):
            t1 = default_timer()
            model.train()
            train_metrics = init_metric_dict(s_fields, vol_signals)
            test_metrics = init_metric_dict(s_fields, vol_signals)
            train_skip_sum = 0.0
            train_skip_sq_sum = 0.0
            train_skip_count = 0
            train_skip_min = float("inf")
            train_skip_max = float("-inf")
            test_skip_sum = 0.0
            test_skip_sq_sum = 0.0
            test_skip_count = 0
            test_skip_min = float("inf")
            test_skip_max = float("-inf")

            train_pbar = tqdm(train_loader, desc=f"Train S{stage} {ep + 1}/{config.epochs}", leave=False, dynamic_ncols=True)
            for batch_idx, batch in enumerate(train_pbar):
                stage_batch = prepare_stage_batch(stage, batch, config, device, s_idx, v_idx)
                optimizer.zero_grad()

                if amp:
                    with torch.autocast(device_type=str(device).split(":")[0], dtype=dtype, enabled=True):
                        loss, loss_s, loss_v, rel, rel_s, rel_v, channel_specs, aux = compute_loss(stage, model, stage_batch, loss_fn, rel_l2, s_fields, vol_signals)
                    scaler.scale(loss).backward()
                    if gradient_norm is not None:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_norm)
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

                if stage == 2 and aux is not None:
                    skip_vals = aux["skip_weights"].detach()
                    train_skip_sum += float(skip_vals.sum().item())
                    train_skip_sq_sum += float((skip_vals * skip_vals).sum().item())
                    train_skip_count += int(skip_vals.numel())
                    train_skip_min = min(train_skip_min, float(skip_vals.min().item()))
                    train_skip_max = max(train_skip_max, float(skip_vals.max().item()))

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
                        batch_log["model/surface_to_volume_skip_weight_min"] = float(aux["skip_weights"].detach().min().item())
                        batch_log["model/surface_to_volume_skip_weight_max"] = float(aux["skip_weights"].detach().max().item())
                        if "skip_weights_couple" in aux:
                            batch_log["model/surface_to_volume_skip_weight_couple_mean"] = float(aux["skip_weight_couple_mean"].detach().item())
                            batch_log["model/surface_to_volume_skip_weight_couple_std"] = float(aux["skip_weight_couple_std"].detach().item())
                            batch_log["model/surface_to_volume_skip_weight_couple_min"] = float(aux["skip_weights_couple"].detach().min().item())
                            batch_log["model/surface_to_volume_skip_weight_couple_max"] = float(aux["skip_weights_couple"].detach().max().item())
                            batch_log["model/surface_to_volume_skip_weight_fuse_mean"] = float(aux["skip_weight_fuse_mean"].detach().item())
                            batch_log["model/surface_to_volume_skip_weight_fuse_std"] = float(aux["skip_weight_fuse_std"].detach().item())
                            batch_log["model/surface_to_volume_skip_weight_fuse_min"] = float(aux["skip_weights_fuse"].detach().min().item())
                            batch_log["model/surface_to_volume_skip_weight_fuse_max"] = float(aux["skip_weights_fuse"].detach().max().item())
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

                    if stage == 2 and _aux is not None:
                        skip_vals = _aux["skip_weights"].detach()
                        test_skip_sum += float(skip_vals.sum().item())
                        test_skip_sq_sum += float((skip_vals * skip_vals).sum().item())
                        test_skip_count += int(skip_vals.numel())
                        test_skip_min = min(test_skip_min, float(skip_vals.min().item()))
                        test_skip_max = max(test_skip_max, float(skip_vals.max().item()))

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
                    "scaler_state_dict": scaler.state_dict(),
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
                "scaler_state_dict": scaler.state_dict(),
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
            if stage == 2 and test_skip_count > 0:
                train_skip_mean = train_skip_sum / max(train_skip_count, 1)
                train_skip_var = max(train_skip_sq_sum / max(train_skip_count, 1) - train_skip_mean * train_skip_mean, 0.0)
                test_skip_mean = test_skip_sum / test_skip_count
                test_skip_var = max(test_skip_sq_sum / test_skip_count - test_skip_mean * test_skip_mean, 0.0)

                epoch_log["model/surface_to_volume_skip_weight_mean"] = float(test_skip_mean)
                epoch_log["model/surface_to_volume_skip_weight_std"] = float(test_skip_var ** 0.5)
                epoch_log["model/surface_to_volume_skip_weight_min"] = float(test_skip_min)
                epoch_log["model/surface_to_volume_skip_weight_max"] = float(test_skip_max)
                epoch_log["model/surface_to_volume_skip_weight_mean_train"] = float(train_skip_mean)
                epoch_log["model/surface_to_volume_skip_weight_std_train"] = float(train_skip_var ** 0.5)
                epoch_log["model/surface_to_volume_skip_weight_min_train"] = float(train_skip_min)
                epoch_log["model/surface_to_volume_skip_weight_max_train"] = float(train_skip_max)
            add_all_field_metrics(epoch_log, "train", s_fields, vol_signals, metric_values=train_metrics)
            add_all_field_metrics(epoch_log, "test", s_fields, vol_signals, metric_values=test_metrics)
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
