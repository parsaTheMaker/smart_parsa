import hydra
from omegaconf import DictConfig

from models.abupt import ABUPT
from train_consistency_common import run_consistency_training


class ABUPTWithLatent(ABUPT):
    def forward(self, geo, surf_query_pos, vol_query_pos, params=None, geo_log_density=None, return_latent=False):
        prepared = self.prepare_contract_inputs(geo, surf_query_pos, vol_query_pos, geo_log_density=geo_log_density)
        geometry_encoding, geometry_pos = self.encode_geometry(
            prepared["geometry_position"],
            geometry_supernode_position=prepared["geometry_supernode_position"],
            geometry_supernode_idx=prepared["geometry_supernode_idx"],
            params=params,
            geo_log_density=geo_log_density,
        )

        x_surface, x_volume, surface_position_all, volume_position_all = self._encode_surface_volume_tokens(
            prepared["surface_anchor_position"],
            prepared["volume_anchor_position"],
            prepared["surface_query_position"],
            prepared["volume_query_position"],
            params=params,
        )
        x_surface, x_volume = self._shared_forward(
            x_surface,
            x_volume,
            surface_position_all,
            volume_position_all,
            geometry_encoding,
            geometry_pos,
            params=params,
        )

        num_surface_anchor = prepared["surface_anchor_position"].shape[1]
        num_volume_anchor = prepared["volume_anchor_position"].shape[1]
        surface_anchor_tokens = x_surface[:, :num_surface_anchor]
        volume_anchor_tokens = x_volume[:, :num_volume_anchor]
        for block in self.surface_blocks:
            x_surface = block(x_surface, surface_anchor_tokens, params=params, x_pos=surface_position_all, anchor_pos=prepared["surface_anchor_position"])
        for block in self.volume_blocks:
            x_volume = block(x_volume, volume_anchor_tokens, params=params, x_pos=volume_position_all, anchor_pos=prepared["volume_anchor_position"])

        pred_surface_all = self.surface_decoder(x_surface)
        pred_volume_all = self.volume_decoder(x_volume)
        pred_surf = pred_surface_all[:, num_surface_anchor:]
        pred_vol = pred_volume_all[:, num_volume_anchor:]
        if return_latent:
            return pred_surf, pred_vol, geometry_encoding
        return pred_surf, pred_vol


@hydra.main(version_base="1.2", config_path="config", config_name="drivaerml_abupt_satloss3")
def main(cfg: DictConfig):
    run_consistency_training(cfg, model_ctor=ABUPTWithLatent, model_requires_density=False)


if __name__ == "__main__":
    main()
    print("Training done.")
