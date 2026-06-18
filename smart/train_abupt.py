import hydra
from omegaconf import DictConfig

from models.abupt import ABUPT
from utils.surface_volume_trainer import run_surface_volume_training


@hydra.main(version_base="1.2", config_path="config", config_name="drivaerml_abupt")
def main(cfg: DictConfig):
    run_surface_volume_training(cfg, ABUPT, accepts_geo_log_density=False)


if __name__ == "__main__":
    main()
    print("Training done.")
