"""DeAL/SATLOSS7 continuation for PointNet++ SSG on Pump."""

import hydra
from omegaconf import DictConfig

from models.pointnet2_ssg import PointNet2SSGWithLatent
from train_consistency_common import run_consistency_training


@hydra.main(version_base="1.2", config_path="config", config_name="pump_pointnet2_ssg_deal_from_base")
def main(cfg: DictConfig):
    run_consistency_training(cfg, model_ctor=PointNet2SSGWithLatent, model_requires_density=False)


if __name__ == "__main__":
    main()
