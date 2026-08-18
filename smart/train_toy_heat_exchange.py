"""Vanilla SMART entry point for the nonlinear toy heat-exchanger benchmark."""

import hydra
from omegaconf import DictConfig

from models.smart.smart import SMART
from utils.surface_volume_trainer import run_surface_volume_training


@hydra.main(version_base="1.2", config_path="config", config_name="toy_heat_exchange")
def main(cfg: DictConfig):
    run_surface_volume_training(cfg, SMART, accepts_geo_log_density=False)


if __name__ == "__main__":
    main()
