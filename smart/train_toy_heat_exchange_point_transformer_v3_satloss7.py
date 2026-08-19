"""SATLOSS7 PointTransformerV3 training entry point for toy heat exchange."""

import hydra
from omegaconf import DictConfig

from models.point_transformer_v3 import PointTransformerV3WithLatent
from train_consistency_common import run_consistency_training


@hydra.main(version_base="1.2", config_path="config", config_name="toy_heat_exchange_point_transformer_v3_satloss7")
def main(cfg: DictConfig):
    run_consistency_training(cfg, model_ctor=PointTransformerV3WithLatent, model_requires_density=False)


if __name__ == "__main__":
    main()
