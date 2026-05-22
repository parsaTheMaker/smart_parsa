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
from utils.utils import initialize_gpu, initialize_wandb, get_model_checkpoint_name, count_model_params, get_optimizer_scheduler_loss


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
        repl = candidate[torch.randint(0, candidate.numel(), (k,), device=device)]
        return repl

    if k <= n:
        return torch.randperm(n, device=device)[:k]
    return torch.randint(0, n, (k,), device=device)


def gather_per_batch(x: torch.Tensor, idx_list: list[torch.Tensor]) -> torch.Tensor:
    return torch.stack([x[b, idx_list[b], :] for b in range(x.shape[0])], dim=0)


def prepare_stage_batch(stage: int, batch, config, device: torch.device):
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
    vol_target = gather_per_batch(vol_data, vol_q_idx)

    return {
        "surface_input": surface_input,
        "vol_query": vol_query,
        "vol_target": vol_target,
    }


def compute_stage_loss(model: CAT, stage_batch: dict, stage: int, loss_fn, rel_l2):
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
        rel = rel_l2(pred_surf, stage_batch["surf_target"]) + rel_l2(pred_vol, stage_batch["vol_target"])
        return loss, rel

    if stage == 2:
        pred = model.forward_stage2(stage_batch["surface_input"], stage_batch["surf_query"])
        loss = loss_fn(pred, stage_batch["surf_target"])
        rel = rel_l2(pred, stage_batch["surf_target"])
        return loss, rel

    pred = model.forward_stage3(stage_batch["surface_input"], stage_batch["vol_query"])
    loss = loss_fn(pred, stage_batch["vol_target"])
    rel = rel_l2(pred, stage_batch["vol_target"])
    return loss, rel


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
        payload["fusion_state_dict"] = model.fusion.state_dict()
        payload["stage3_decoder_state_dict"] = model.stage3_decoder.state_dict()
        payload["stage3_head_state_dict"] = model.stage3_head.state_dict()
    return payload


@hydra.main(version_base="1.2", config_path="config", config_name="naca4_cat")
def main(cfg: DictConfig):
    config = cfg.experiment
    wandb_config = cfg.wandb

    stage = int(getattr(config, "cat_stage", 1))
    if stage not in (1, 2, 3):
        raise ValueError("cat_stage must be 1, 2, or 3")

    if getattr(config, "model_tag", "") in (None, ""):
        config.model_tag = f"stage{stage}"

    run = initialize_wandb(config, wandb_config)
    device = initialize_gpu(config.random_seed, high_precision=False)

    gradient_norm = config.gradient_norm
    precisions = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    dtype = precisions.get(config.precision, torch.float16)
    amp = config.amp

    train_data, test_data, stats, spatial_dim, surf_channels, vol_channels, params_dim, fields = get_dataset(config)
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
        model.freeze_stage3_encoders()
        print(f"Loaded and froze encoders from: {geom_ckpt} and {surf_ckpt}")

    model_checkpoint_name = get_model_checkpoint_name(config)
    model_checkpoint_name = f"{model_checkpoint_name}-cat-stage{stage}"

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
            train_loss_sum = 0.0
            train_rel_sum = 0.0

            train_pbar = tqdm(train_loader, desc=f"Train S{stage} {ep + 1}/{config.epochs}", leave=False, dynamic_ncols=True)
            for batch_idx, batch in enumerate(train_pbar):
                stage_batch = prepare_stage_batch(stage, batch, config, device)
                optimizer.zero_grad()

                if amp:
                    with torch.autocast(device_type=str(device).split(":")[0], dtype=dtype, enabled=True):
                        loss, rel = compute_stage_loss(model, stage_batch, stage, loss_fn, rel_l2)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                else:
                    loss, rel = compute_stage_loss(model, stage_batch, stage, loss_fn, rel_l2)
                    loss.backward()
                    if gradient_norm is not None:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_norm)
                    optimizer.step()
                    scheduler.step()

                bsz = next(iter(stage_batch.values())).shape[0]
                train_loss_sum += loss.item() * bsz
                train_rel_sum += rel.item() * bsz

                global_step += 1
                if batch_idx % log_every_n_steps == 0 or batch_idx == len(train_loader) - 1:
                    wandb.log(
                        {
                            "train/batch_loss": loss.item(),
                            "train/batch_rel_l2": rel.item(),
                            "lr": scheduler.get_last_lr()[0],
                            "epoch": ep,
                            "stage": stage,
                        },
                        step=global_step,
                    )
                    train_pbar.set_postfix(loss=f"{loss.item():.4f}", rel=f"{rel.item():.4f}")

            model.eval()
            test_loss_sum = 0.0
            test_rel_sum = 0.0
            with torch.no_grad():
                test_pbar = tqdm(test_loader, desc=f"Eval  S{stage} {ep + 1}/{config.epochs}", leave=False, dynamic_ncols=True)
                for batch in test_pbar:
                    stage_batch = prepare_stage_batch(stage, batch, config, device)
                    if amp:
                        with torch.autocast(device_type=str(device).split(":")[0], dtype=dtype, enabled=True):
                            loss, rel = compute_stage_loss(model, stage_batch, stage, loss_fn, rel_l2)
                    else:
                        loss, rel = compute_stage_loss(model, stage_batch, stage, loss_fn, rel_l2)

                    bsz = next(iter(stage_batch.values())).shape[0]
                    test_loss_sum += loss.item() * bsz
                    test_rel_sum += rel.item() * bsz
                    test_pbar.set_postfix(loss=f"{loss.item():.4f}", rel=f"{rel.item():.4f}")

            train_loss = train_loss_sum / len(train_loader.dataset)
            train_rel = train_rel_sum / len(train_loader.dataset)
            test_loss = test_loss_sum / len(test_loader.dataset)
            test_rel = test_rel_sum / len(test_loader.dataset)

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
            }
            last_payload.update(build_stage_payload(model, stage))
            torch.save(last_payload, f"checkpoints/{model_checkpoint_name}_last.pt")

            t2 = default_timer()
            print(
                f"stage: {stage}, epoch: {ep}, epoch_time: {t2 - t1:.4f}, "
                f"train_loss: {train_loss:.5f}, train_rel: {train_rel:.5f}, "
                f"test_loss: {test_loss:.5f}, test_rel: {test_rel:.5f}"
            )

            wandb.log(
                {
                    "train/loss": train_loss,
                    "train/rel_l2": train_rel,
                    "test/loss": test_loss,
                    "test/rel_l2": test_rel,
                    "lr": scheduler.get_last_lr()[0],
                    "epoch": ep,
                    "stage": stage,
                },
                step=global_step,
            )

    except KeyboardInterrupt:
        print("\nTraining interrupted by user (Ctrl+C). Saving last checkpoint...")
        emergency = {
            "epoch": locals().get("ep", -1),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "stage": stage,
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
