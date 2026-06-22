import hydra
import torch
from omegaconf import DictConfig

from models.transolverpp import TransolverPP
from train_consistency_common import run_consistency_training


class TransolverPPWithLatent(TransolverPP):
    def forward(self, geo, surf_query_pos, vol_query_pos, params=None, geo_log_density=None, return_latent=False):
        geo_pos = self._select_geometry_tokens(geo, geo_log_density=geo_log_density)
        surf_pos = surf_query_pos * self.pos_scale_factor
        vol_pos = vol_query_pos * self.pos_scale_factor

        geometry_tokens = self.geometry_preprocess(geo_pos)
        geometry_context_input = geometry_tokens.mean(dim=1)
        geometry_condition_token = self.geometry_condition(geometry_context_input.to(dtype=geometry_tokens.dtype)).unsqueeze(1)

        query_pos = torch.cat([surf_pos, vol_pos], dim=1)
        query_tokens = self.preprocess(query_pos)
        query_tokens = query_tokens + self.placeholder.view(1, 1, -1)
        condition_token = geometry_condition_token + self.placeholder.view(1, 1, -1)
        tokens = torch.cat([condition_token, query_tokens], dim=1)
        tokens = self.cond(tokens, params)
        tokens = self._run_blocks(tokens)
        query_latent = tokens[:, 1:]

        surf_count = surf_query_pos.shape[1]
        pred_all = self.output_head(query_latent)
        pred_surf = pred_all[:, :surf_count, : self.surface_channels]
        pred_vol = pred_all[:, surf_count:, self.surface_channels :]
        if return_latent:
            return pred_surf, pred_vol, query_latent
        return pred_surf, pred_vol


@hydra.main(version_base="1.2", config_path="config", config_name="drivaerml_transolverpp_satloss3")
def main(cfg: DictConfig):
    run_consistency_training(cfg, model_ctor=TransolverPPWithLatent, model_requires_density=False)


if __name__ == "__main__":
    main()
    print("Training done.")
