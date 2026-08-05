import hydra
from omegaconf import DictConfig

from models.mspt import MSPT
from train_consistency_common import run_consistency_training


class MSPTSATLOSS8(MSPT):
    def forward(self, geo, surf_query_pos, vol_query_pos, params=None, geo_log_density=None, return_latent=False):
        return super().forward(
            geo,
            surf_query_pos,
            vol_query_pos,
            params=params,
            geo_log_density=geo_log_density,
            return_latent=return_latent,
        )


@hydra.main(version_base="1.2", config_path="config", config_name="drivaerml_mspt_satloss8")
def main(cfg: DictConfig):
    run_consistency_training(cfg, model_ctor=MSPTSATLOSS8, model_requires_density=False)


if __name__ == "__main__":
    main()
    print("Training done.")
