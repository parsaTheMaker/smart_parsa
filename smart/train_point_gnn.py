import hydra
from omegaconf import DictConfig

from models.point_gnn import PointGNN
from utils.surface_volume_trainer import run_surface_volume_training


@hydra.main(version_base="1.2", config_path="config", config_name="drivaerml_point_gnn")
def main(cfg: DictConfig):
    run_surface_volume_training(cfg, PointGNN, accepts_geo_log_density=False)


if __name__ == "__main__":
    main()
    print("Training done.")
