"""DeAL/SATLOSS7 continuation for MSPT on Pump."""

import hydra
from omegaconf import DictConfig

from models.mspt import MSPT
from train_consistency_common import run_consistency_training


class MSPTWithLatent(MSPT):
    def forward(self, geo, surf_query_pos, vol_query_pos, params=None, geo_log_density=None, return_latent=False):
        return super().forward(
            geo,
            surf_query_pos,
            vol_query_pos,
            params=params,
            geo_log_density=geo_log_density,
            return_latent=return_latent,
        )


@hydra.main(version_base="1.2", config_path="config", config_name="pump_mspt_deal_from_base")
def main(cfg: DictConfig):
    run_consistency_training(cfg, model_ctor=MSPTWithLatent, model_requires_density=False)


if __name__ == "__main__":
    main()
