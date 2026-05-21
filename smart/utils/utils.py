import wandb
import torch
import torch.nn as nn
import numpy as np
import random
from loss.losses import RelL2Loss
from lion_pytorch import Lion
from omegaconf import OmegaConf
import os
import json
import re


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


def get_run_name(config, fields=None):
    variant = getattr(config, "manifest_variant", None)
    parts = [
        config.model_name,
        getattr(config, "dataset", None),
        variant if variant and variant != "full" else None,
        f"s{getattr(config, 'random_seed', 'na')}",
    ]
    return _join_nonempty(parts)


def get_output_run_name(config, fields=None):
    return get_run_name(config, fields)

def initialize_gpu(random_seed, high_precision=True):
    """Initializes the GPU settings and sets the random seed."""
    
    # Device settings
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if high_precision:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")

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
        tags.append(f"fields_{get_field_tag(fields)}")

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


def get_optimizer_scheduler_loss(model, config, train_loader, loss_dim=-2):
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
    
    # Get optimizer
    if config.optimizer == "adam":
        # we have to exclude the bias and weights from norms
        grouped_parameters = exclude_params_from_weight_decay(model)
        optimizer = torch.optim.Adam(grouped_parameters, lr=config.learning_rate, weight_decay=1e-5)
    elif config.optimizer == "adamw":
        # we have to exclude the bias and weights from norms
        grouped_parameters = exclude_params_from_weight_decay(model, exclude=["bias", "norm", "query_pos", "B"])
        optimizer = torch.optim.AdamW(grouped_parameters, lr=config.learning_rate, weight_decay=1e-4)
    elif config.optimizer == "lion":
        grouped_parameters = exclude_params_from_weight_decay(model, exclude=["bias", "norm", "query_pos", "B"])
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
