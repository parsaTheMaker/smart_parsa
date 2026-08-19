"""Vanilla PointTransformerV3 training entry point for toy heat exchange."""

import hydra
from omegaconf import DictConfig

from models.point_transformer_v3 import PointTransformerV3
from utils.surface_volume_trainer import run_surface_volume_training


@hydra.main(version_base="1.2", config_path="config", config_name="toy_heat_exchange_point_transformer_v3")
def main(cfg: DictConfig):
    run_surface_volume_training(cfg, PointTransformerV3, accepts_geo_log_density=False)


if __name__ == "__main__":
    main()
