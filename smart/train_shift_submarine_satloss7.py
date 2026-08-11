"""SATLOSS7 SMART training entrypoint for SHIFT-Submarine."""

import hydra
from omegaconf import DictConfig

from models.smart.smart import SMART
from train_consistency_common import run_consistency_training


@hydra.main(version_base="1.2", config_path="config", config_name="shift_submarine_satloss7")
def main(cfg: DictConfig):
    run_consistency_training(cfg, model_ctor=SMART, model_requires_density=False)


if __name__ == "__main__":
    main()
    print("Training done.")
