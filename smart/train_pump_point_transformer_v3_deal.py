"""DeAL/SATLOSS7 continuation for Point Transformer v3 on Pump."""

import hydra
from omegaconf import DictConfig

from models.point_transformer_v3 import PointTransformerV3WithLatent
from train_consistency_common import run_consistency_training


@hydra.main(version_base="1.2", config_path="config", config_name="pump_point_transformer_v3_deal_from_base")
def main(cfg: DictConfig):
    run_consistency_training(cfg, model_ctor=PointTransformerV3WithLatent, model_requires_density=False)


if __name__ == "__main__":
    main()
