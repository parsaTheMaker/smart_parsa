"""DeAL/SATLOSS7 continuation for AB-UPT on Heat Exchanger."""

import hydra
from omegaconf import DictConfig

from models.ab_upt import ABUPT
from train_consistency_common import run_consistency_training


@hydra.main(version_base="1.2", config_path="config", config_name="toy_heat_exchange_ab_upt_deal_from_base")
def main(cfg: DictConfig):
    run_consistency_training(cfg, model_ctor=ABUPT, model_requires_density=False)


if __name__ == "__main__":
    main()
