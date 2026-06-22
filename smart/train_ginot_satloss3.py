import hydra
from omegaconf import DictConfig

from models.ginot import GINOT
from train_consistency_common import run_consistency_training


class GINOTWithLatent(GINOT):
    def forward(self, geo, surf_query_pos, vol_query_pos, params=None, geo_log_density=None, return_latent=False):
        geometry_latents, geometry_pos = self.encode_geometry(geo, params=params, geo_log_density=geo_log_density)
        query_features = self.decode_features(
            geometry_latents,
            geometry_pos,
            surf_query_pos,
            vol_query_pos,
            params=params,
        )
        pred = self.head(query_features)
        pred_surf, pred_vol = pred[:, : surf_query_pos.shape[1], : self.surface_channels], pred[:, surf_query_pos.shape[1] :, self.surface_channels :]
        if return_latent:
            return pred_surf, pred_vol, geometry_latents
        return pred_surf, pred_vol


@hydra.main(version_base="1.2", config_path="config", config_name="drivaerml_ginot_satloss3")
def main(cfg: DictConfig):
    run_consistency_training(cfg, model_ctor=GINOTWithLatent, model_requires_density=False)


if __name__ == "__main__":
    main()
    print("Training done.")
