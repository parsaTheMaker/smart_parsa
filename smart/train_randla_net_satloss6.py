import hydra
from omegaconf import DictConfig

from models.randla_net import RandLANetWithLatent
from train_consistency_common import run_consistency_training


@hydra.main(version_base="1.2", config_path="config", config_name="drivaerml_randla_net_satloss6")
def main(cfg: DictConfig):
    run_consistency_training(cfg, model_ctor=RandLANetWithLatent, model_requires_density=False)


if __name__ == "__main__":
    main()
    print("Training done.")
