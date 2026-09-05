"""DeAL/SATLOSS7 continuation for LNO on Pump."""

import hydra
from omegaconf import DictConfig

from models.lno import LNOWithLatent
from train_consistency_common import run_consistency_training


@hydra.main(version_base="1.2", config_path="config", config_name="pump_lno_deal_from_base")
def main(cfg: DictConfig):
    run_consistency_training(cfg, model_ctor=LNOWithLatent, model_requires_density=False)


if __name__ == "__main__":
    main()
