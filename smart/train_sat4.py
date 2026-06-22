import hydra
from omegaconf import DictConfig

from models.smart.smart_sat4 import SMARTSAT4
from train_consistency_common import run_consistency_training


@hydra.main(version_base="1.2", config_path="config", config_name="drivaerml_sat4")
def main(cfg: DictConfig):
    run_consistency_training(cfg, model_ctor=SMARTSAT4, model_requires_density=True)


if __name__ == "__main__":
    main()
    print("Training done.")
