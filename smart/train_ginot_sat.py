import hydra
from omegaconf import DictConfig

from models.ginot_sat import GINOTSAT
from utils.surface_volume_trainer import run_surface_volume_training


@hydra.main(version_base="1.2", config_path="config", config_name="drivaerml_ginot_sat")
def main(cfg: DictConfig):
    run_surface_volume_training(cfg, GINOTSAT, accepts_geo_log_density=True)


if __name__ == "__main__":
    main()
    print("Training done.")
