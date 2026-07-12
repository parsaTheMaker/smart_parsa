from dataclasses import fields, is_dataclass

from omegaconf import DictConfig, OmegaConf


def shallow_asdict(value):
    if is_dataclass(value):
        return {field.name: getattr(value, field.name) for field in fields(value)}
    if isinstance(value, DictConfig):
        return dict(value)
    raise TypeError(f"Unsupported config type: {type(value)}")


def safe_replace(value, **kwargs):
    if is_dataclass(value):
        field_names = {field.name for field in fields(value)}
        for key, item in kwargs.items():
            if key in field_names:
                setattr(value, key, item)
        return value
    if isinstance(value, DictConfig):
        return OmegaConf.merge(value, OmegaConf.create(kwargs))
    raise TypeError(f"Unsupported config type: {type(value)}")
