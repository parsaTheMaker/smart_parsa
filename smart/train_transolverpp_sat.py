import hydra
from omegaconf import DictConfig

from models.transolverpp_sat import TransolverPPSAT
from utils.surface_volume_trainer import run_surface_volume_training


@hydra.main(version_base="1.2", config_path="config", config_name="drivaerml_transolverpp_sat")
def main(cfg: DictConfig):
    run_surface_volume_training(cfg, TransolverPPSAT, accepts_geo_log_density=True)


if __name__ == "__main__":
    main()
    print("Training done.")
