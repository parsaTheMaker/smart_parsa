import wandb
import torch
import torch.nn as nn
import numpy as np
import random
import inspect
from loss.losses import RelL2Loss
from lion_pytorch import Lion
from omegaconf import OmegaConf, open_dict
import os
import json
import re
import hashlib


def _slugify(text):
    text = str(text).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = re.sub(r"-+", "-", text).strip("-")
    return text or "run"


def _join_nonempty(parts):
    return "-".join(_slugify(part) for part in parts if part not in (None, "", []))


def infer_fields_from_config(config):
    dataset = getattr(config, "dataset", None)
    if dataset == "NACA4":
        return {"surface": ["pressure", "normal_x", "normal_y"], "volume": ["pressure", "sdf", "velocity_x", "velocity_y"]}
    if dataset in {"AhmedMLV2", "DrivAerML"}:
        return {"surface": ["pressure", "normal_x", "normal_y", "normal_z", "wall_shear_x", "wall_shear_y", "wall_shear_z"], "volume": ["pressure", "velocity_x", "velocity_y", "velocity_z"]}
    if dataset in {"ShapeNetCar", "AhmedML", "ShiftSUV"}:
        return {"surface": ["pressure"], "volume": ["velocity_x", "velocity_y", "velocity_z"]}
    if dataset == "ShiftWing":
        return {"surface": ["pressure"], "volume": ["velocity_x", "velocity_y", "velocity_z"]}
    return None


def get_field_tag(fields):
    if not fields:
        return None
    surface = fields.get("surface", [])
    volume = fields.get("volume", [])
    surface_tag = "+".join([
        field.replace("velocity_", "v").replace("normal_", "n").replace("pressure", "p").replace("sdf", "sdf")
        for field in surface
    ])
    volume_tag = "+".join([
        field.replace("velocity_", "v").replace("normal_", "n").replace("pressure", "p").replace("sdf", "sdf")
        for field in volume
    ])
    return f"s-{surface_tag}-v-{volume_tag}"


def _safe_wandb_tag(tag, max_len=64):
    """W&B tags must be <=64 chars; compact long tags deterministically."""
    tag = str(tag)
    if len(tag) <= max_len:
        return tag
    digest = hashlib.sha1(tag.encode("utf-8")).hexdigest()[:8]
    # Keep a readable prefix and append stable hash.
    keep = max_len - len("-") - len(digest)
    return f"{tag[:keep]}-{digest}"


def get_run_name(config, fields=None):
    variant = getattr(config, "manifest_variant", None)
    parts = [
        config.model_name,
        getattr(config, "model_tag", None) if getattr(config, "model_tag", "") else None,
        getattr(config, "dataset", None),
        variant if variant and variant != "full" else None,
        f"s{getattr(config, 'random_seed', 'na')}",
    ]
    return _join_nonempty(parts)


def get_output_run_name(config, fields=None):
    return get_run_name(config, fields)



def apply_naca4_auto_point_budget(config, dataset_obj, for_cat=False):
    """Auto-resolve point budgets from the minimum non-zero surface count across the dataset."""
    if getattr(config, "dataset", None) != "NACA4":
        return None

    if not hasattr(dataset_obj, "get_min_surface_points_nonzero"):
        return None

    min_surface = int(dataset_obj.get_min_surface_points_nonzero())
    if min_surface <= 0:
        raise ValueError("Could not infer a positive minimum surface-point count from NACA4 dataset.")

    num_blocks = int(getattr(getattr(config, "architecture", {}), "num_encoder_decoder_blocks", 1))
    num_blocks = max(num_blocks, 1)

    effective_surface_points = min_surface
    anchor_points = max(1, effective_surface_points // num_blocks)

    config.num_body_points = effective_surface_points
    config.num_surface_points = effective_surface_points

    if hasattr(config, "architecture"):
        with open_dict(config.architecture):
            config.architecture.subsampled_geometry_points = effective_surface_points
            config.architecture.latent_geometry_points = anchor_points

    info = {
        "min_surface_points_nonzero": min_surface,
        "effective_surface_points": effective_surface_points,
        "num_blocks": num_blocks,
        "anchor_points": anchor_points,
    }

    if for_cat:
        with open_dict(config):
            config.stage1_surface_input_points = effective_surface_points
            config.stage1_surface_query_points = effective_surface_points
            config.stage2_surface_input_points = effective_surface_points
            config.stage2_surface_query_points = effective_surface_points
            config.stage3_surface_input_points = effective_surface_points

        stage3_vq = int(getattr(config, "stage3_volume_query_points", 0))
        if stage3_vq <= 0:
            stage3_vq = int(getattr(config, "num_volume_points", 0))
        if stage3_vq <= 0 and hasattr(dataset_obj, "get_min_volume_points_nonzero"):
            stage3_vq = int(dataset_obj.get_min_volume_points_nonzero())
        if stage3_vq <= 0:
            raise ValueError("Could not infer a positive stage3 volume query count.")

        with open_dict(config):
            config.stage3_volume_query_points = stage3_vq
            # Request: stage1 volume query should be 4x old value and equal to stage3 volume query.
            config.stage1_volume_query_points = stage3_vq

        info.update({
            "stage1_surface_input_points": int(config.stage1_surface_input_points),
            "stage1_surface_query_points": int(config.stage1_surface_query_points),
            "stage1_volume_query_points": int(config.stage1_volume_query_points),
            "stage2_surface_input_points": int(config.stage2_surface_input_points),
            "stage2_surface_query_points": int(config.stage2_surface_query_points),
            "stage3_surface_input_points": int(config.stage3_surface_input_points),
            "stage3_volume_query_points": int(config.stage3_volume_query_points),
        })

    return info


def print_point_budget(prefix, info):
    if not info:
        return
    print(f"[{prefix}] min surface points (non-zero across dataset): {info['min_surface_points_nonzero']}")
    print(f"[{prefix}] effective surface points: {info['effective_surface_points']}")
    print(f"[{prefix}] encoder/decoder blocks (M): {info['num_blocks']}")
    print(f"[{prefix}] anchor points (effective_surface/M): {info['anchor_points']}")
    for key in [
        "stage1_surface_input_points",
        "stage1_surface_query_points",
        "stage1_volume_query_points",
        "stage2_surface_input_points",
        "stage2_surface_query_points",
        "stage3_surface_input_points",
        "stage3_volume_query_points",
    ]:
        if key in info:
            print(f"[{prefix}] {key}: {info[key]}")

def initialize_gpu(random_seed, high_precision=True):
    """Initializes the GPU settings and sets the random seed."""
    
    # Device settings
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        if hasattr(torch.backends.cuda, "enable_flash_sdp"):
            torch.backends.cuda.enable_flash_sdp(True)
        if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
            torch.backends.cuda.enable_mem_efficient_sdp(True)
        if hasattr(torch.backends.cuda, "enable_math_sdp"):
            torch.backends.cuda.enable_math_sdp(True)
    if high_precision:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
    else:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    # Set random seed
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    
    return device


def initialize_wandb(config, wandb_config, model_files=[]):
    """Initializes wandb with the given config."""

    fields = getattr(config, "fields", None) or infer_fields_from_config(config)
    run_name = get_run_name(config, fields)
    tags = [
        f"seed_{config.random_seed}",
        getattr(config, "dataset", "unknown"),
        getattr(config, "model_name", "unknown"),
    ]
    if getattr(config, "manifest_variant", None) and getattr(config, "manifest_variant", "full") != "full":
        tags.append(f"variant_{config.manifest_variant}")
    if fields:
        tags.append(_safe_wandb_tag(f"fields_{get_field_tag(fields)}"))
    tags = [_safe_wandb_tag(t) for t in tags]

    run = wandb.init(
        name=run_name,
        project=wandb_config.project,
        entity=wandb_config.entity,
        tags=tags,
    )
    
    # Add config to wandb
    wandb.config.update(OmegaConf.to_container(config, resolve=True, throw_on_missing=True))
    
    # Add model files to wandb
    if model_files:
        artifact = wandb.Artifact("model-code", type="code")
        for file in model_files:
            artifact.add_file(file)
        wandb.log_artifact(artifact)
    
    print(f"Model {config.model_name}, "
          f"random seed: {config.random_seed}, "
          f"epochs: {config.epochs}, "
          f"learning rate: {config.learning_rate}")
    
    return run


def get_model_checkpoint_name(config):
    """Returns the model checkpoint name based on the config."""

    if not os.path.exists("checkpoints"):
        os.makedirs("checkpoints")
    variant = getattr(config, "manifest_variant", None)
    parts = [
        config.model_name,
        getattr(config, "model_tag", None) if getattr(config, "model_tag", "") else None,
        getattr(config, "dataset", None),
        variant if variant and variant != "full" else None,
        f"s{getattr(config, 'random_seed', 'na')}",
    ]
    return _join_nonempty(parts)


def count_model_params(model):
    """Calculates number of parameters of the given model. Complex-valued weights count as two weights (for imaginary
    and real part)."""
    
    params = []
    for p in model.parameters():
        if p.requires_grad:
            if torch.is_complex(p):
                params.append(2 * p.numel())
            else:
                params.append(p.numel())
    return sum(params)


def exclude_params_from_weight_decay(model,
                                     exclude=["bias", "filter_bias", "norm", "query_pos", "modulation_weight", "B", "hash", "table"],
                                     verbose=False):
    """Excludes the given parameters from the weight decay."""
    
    named_parameters = model.named_parameters()
    decay_parameters = []
    decay_parameters_names = []
    no_decay_parameters = []
    no_decay_parameters_names = []

    for name, param in named_parameters:
        if not any(ex in name for ex in exclude):
            decay_parameters_names.append(name)
            decay_parameters.append(param)
        else:
            no_decay_parameters_names.append(name)
            no_decay_parameters.append(param)

    if verbose:
        print("Exclude from weight decay:", no_decay_parameters_names)
        print("Weight decay for:", decay_parameters_names)

    grouped_parameters = [
        {"params": decay_parameters},
        {"params": no_decay_parameters, "weight_decay": 0.0}
    ]
    return grouped_parameters


def get_optimizer_scheduler_loss(model, config, train_loader, loss_dim=-2, extra_param_groups=None):
    """Returns the optimizer, scheduler, and loss for the given model and config.
    
    Args:
        model (torch.nn.Module): The model whose parameters will be optimized.
        config (object): Configuration object containing optimizer, scheduler, and loss function settings.
        train_loader (torch.utils.data.DataLoader): DataLoader for the training data, used to determine the number of steps per epoch.
        loss_dim (int, optional): Dimension over which to compute the relative L2 loss. Defaults to -2.
    
    Returns:
        tuple: A tuple containing:
            - optimizer (torch.optim.Optimizer): Configured optimizer.
            - scheduler (torch.optim.lr_scheduler._LRScheduler): Configured learning rate scheduler.
            - loss_fn (torch.nn.Module): Primary loss function.
            - rel_l2_loss_fn (torch.nn.Module): Relative L2 loss function (always returned for evaluation).
    
    Raises:
        ValueError: If an unsupported optimizer, scheduler, or loss function is specified in the config.
    """
    
    def _cuda_optimizer_impl_kwargs(optimizer_cls):
        if not torch.cuda.is_available():
            return {}
        try:
            parameters = inspect.signature(optimizer_cls).parameters
        except (TypeError, ValueError):
            return {}
        if "fused" in parameters:
            return {"fused": True}
        if "foreach" in parameters:
            return {"foreach": True}
        return {}

    extra_param_groups = list(extra_param_groups or [])

    # Get optimizer
    if config.optimizer == "adam":
        # we have to exclude the bias and weights from norms
        grouped_parameters = exclude_params_from_weight_decay(model)
        grouped_parameters = grouped_parameters + extra_param_groups
        optimizer = torch.optim.Adam(
            grouped_parameters,
            lr=config.learning_rate,
            weight_decay=1e-5,
            **_cuda_optimizer_impl_kwargs(torch.optim.Adam),
        )
    elif config.optimizer == "adamw":
        # we have to exclude the bias and weights from norms
        grouped_parameters = exclude_params_from_weight_decay(model, exclude=["bias", "norm", "query_pos", "B"])
        grouped_parameters = grouped_parameters + extra_param_groups
        optimizer = torch.optim.AdamW(
            grouped_parameters,
            lr=config.learning_rate,
            weight_decay=1e-4,
            **_cuda_optimizer_impl_kwargs(torch.optim.AdamW),
        )
    elif config.optimizer == "lion":
        grouped_parameters = exclude_params_from_weight_decay(model, exclude=["bias", "norm", "query_pos", "B"])
        grouped_parameters = grouped_parameters + extra_param_groups
        optimizer = Lion(grouped_parameters, lr=config.learning_rate, weight_decay=1e-4)
    else:
        raise ValueError("Optimizer not supported!")
    
    # Get scheduler
    if config.scheduler == "one-cycle":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                        max_lr=config.learning_rate,
                                                        pct_start=config.scheduler_warmup_fraction,
                                                        div_factor=1e2, final_div_factor=1e3,
                                                        total_steps=config.epochs * len(train_loader))
    elif config.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config.epochs * len(train_loader))
    elif config.scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.scheduler_step, gamma=config.scheduler_gamma)
    elif config.scheduler == "exponential":
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.9, last_epoch=-1)
    else:
        raise ValueError("Scheduler not supported!")
    
    # Get loss functions
    if config.loss_fn == "mse":
        loss_fn = nn.MSELoss(reduction="mean")
    elif config.loss_fn == "l1":
        loss_fn = nn.L1Loss(reduction="mean")
    elif config.loss_fn == "rel_l2":
        loss_fn = RelL2Loss(dim=loss_dim, reduction="sum")
    else:
        raise ValueError("Loss function not supported!")
    
    # Get rel. L2 loss functions
    rel_l2_loss_fn = RelL2Loss(dim=loss_dim, reduction="sum")

    return optimizer, scheduler, loss_fn, rel_l2_loss_fn


def store_inference_results(dir, model_checkpoint_name, test_losses):
    """Stores inference results in a JSON file."""
    
    if not os.path.exists(dir):
        os.makedirs(dir)
        
    with open(dir + "/" + model_checkpoint_name + "_full_inference.json", 'w') as f: 
        json.dump(test_losses, f)
