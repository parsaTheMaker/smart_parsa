import hydra
from omegaconf import DictConfig

from models.lno2 import LNO2
from utils.surface_volume_trainer import run_surface_volume_training


@hydra.main(version_base="1.2", config_path="config", config_name="drivaerml_lno2")
def main(cfg: DictConfig):
    run_surface_volume_training(cfg, LNO2, accepts_geo_log_density=False)


if __name__ == "__main__":
    main()
    print("Training done.")

