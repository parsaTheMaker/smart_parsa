"""Geo-FNO base training entry point for DrivAerML."""

import hydra
from omegaconf import DictConfig

from models.geo_fno import GeoFNO
from utils.surface_volume_trainer import run_surface_volume_training


@hydra.main(version_base="1.2", config_path="config", config_name="drivaerml_geo_fno")
def main(cfg: DictConfig):
    run_surface_volume_training(cfg, GeoFNO, accepts_geo_log_density=False)


if __name__ == "__main__":
    main()
