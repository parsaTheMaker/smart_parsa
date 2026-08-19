"""SATLOSS7 PointNet++ SSG training entry point for toy heat exchange."""

import hydra
from omegaconf import DictConfig

from models.pointnet2_ssg import PointNet2SSGWithLatent
from train_consistency_common import run_consistency_training


@hydra.main(version_base="1.2", config_path="config", config_name="toy_heat_exchange_pointnet2_ssg_satloss7")
def main(cfg: DictConfig):
    run_consistency_training(cfg, model_ctor=PointNet2SSGWithLatent, model_requires_density=False)


if __name__ == "__main__":
    main()
