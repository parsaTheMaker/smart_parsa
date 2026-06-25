import hydra
import torch
from omegaconf import DictConfig

from models.smart.smart import SMART
from train_consistency_common import run_consistency_training


class SMARTWithLatent(SMART):
    def forward(self, geo, surf_query_pos, vol_query_pos, params, return_latent=False):
        encoded = self.encode(geo, params, return_final=return_latent)
        if return_latent:
            intermediate_latent_geometries, latent_geo_pos, final_latent = encoded
        else:
            intermediate_latent_geometries, latent_geo_pos = encoded
        query_pos = torch.cat([surf_query_pos, vol_query_pos], dim=1)
        pred = self.decode(intermediate_latent_geometries, latent_geo_pos, params, query_pos)
        pred_surf = pred[:, :surf_query_pos.shape[1], : self.surface_channels]
        pred_vol = pred[:, surf_query_pos.shape[1] :, self.surface_channels :]
        if return_latent:
            return pred_surf, pred_vol, final_latent
        return pred_surf, pred_vol


@hydra.main(version_base="1.2", config_path="config", config_name="drivaerml_satloss4")
def main(cfg: DictConfig):
    run_consistency_training(cfg, model_ctor=SMARTWithLatent, model_requires_density=False)


if __name__ == "__main__":
    main()
    print("Training done.")
