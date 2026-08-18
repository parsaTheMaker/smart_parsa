"""SATLOSS7 entry point for the nonlinear toy heat-exchanger benchmark."""

import hydra
from omegaconf import DictConfig

from models.smart.smart import SMART
from train_consistency_common import run_consistency_training


@hydra.main(version_base="1.2", config_path="config", config_name="toy_heat_exchange_satloss7")
def main(cfg: DictConfig):
    run_consistency_training(cfg, model_ctor=SMART, model_requires_density=False)


if __name__ == "__main__":
    main()
