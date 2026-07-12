import hydra
from omegaconf import DictConfig

from models.gaot import GAOT
from train_consistency_common import run_consistency_training


class GAOTWithLatent(GAOT):
    pass


@hydra.main(version_base="1.2", config_path="config", config_name="drivaerml_gaot_satloss6")
def main(cfg: DictConfig):
    run_consistency_training(cfg, model_ctor=GAOTWithLatent, model_requires_density=False)


if __name__ == "__main__":
    main()
    print("Training done.")
