"""DeAL/SATLOSS7 continuation for Geo-FNO on DrivAerML."""

import hydra
from omegaconf import DictConfig

from models.geo_fno import GeoFNO
from train_consistency_common import run_consistency_training


@hydra.main(version_base="1.2", config_path="config", config_name="drivaerml_geo_fno_deal_from_base")
def main(cfg: DictConfig):
    run_consistency_training(cfg, model_ctor=GeoFNO, model_requires_density=False)


if __name__ == "__main__":
    main()
