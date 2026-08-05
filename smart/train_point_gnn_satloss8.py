import hydra
from omegaconf import DictConfig

from models.point_gnn import PointGNNWithLatent
from train_consistency_common import run_consistency_training


@hydra.main(version_base="1.2", config_path="config", config_name="drivaerml_point_gnn_satloss8")
def main(cfg: DictConfig):
    run_consistency_training(cfg, model_ctor=PointGNNWithLatent, model_requires_density=False)


if __name__ == "__main__":
    main()
    print("Training done.")
