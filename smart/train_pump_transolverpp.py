import hydra
from omegaconf import DictConfig

from models.transolverpp import TransolverPP
from utils.surface_volume_trainer import run_surface_volume_training


@hydra.main(version_base="1.2", config_path="config", config_name="pump_transolverpp")
def main(cfg: DictConfig):
    run_surface_volume_training(cfg, TransolverPP, accepts_geo_log_density=False)


if __name__ == "__main__":
    main()
