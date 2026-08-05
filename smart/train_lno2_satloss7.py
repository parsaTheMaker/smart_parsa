import hydra
from omegaconf import DictConfig

from models.lno2 import LNO2WithLatent
from train_consistency_common import run_consistency_training


@hydra.main(version_base="1.2", config_path="config", config_name="drivaerml_lno2_satloss7")
def main(cfg: DictConfig):
    run_consistency_training(cfg, model_ctor=LNO2WithLatent, model_requires_density=False)


if __name__ == "__main__":
    main()
    print("Training done.")

