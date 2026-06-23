from __future__ import annotations

import torch
import torch.nn as nn

from .family_common import CondInjection, QueryTypeEmbedding, init_linear_layer_weights, sample_tokens, split_surface_volume_predictions
from .smart.smart import PlainMLP


class PointNetBase(nn.Module):
    expects_geo_log_density = False

    def __init__(
        self,
        spatial_dim=3,
        surface_channels=1,
        volume_channels=3,
        parameter_channels=0,
        latent_dim=256,
        latent_geometry_points=4096,  # kept for config compatibility
        subsampled_geometry_points=32768,
        num_encoder_decoder_blocks=6,
        pos_scale_factor=100,
        dropout=0.0,
        subregion_size=65536,
    ):
        super().__init__()
        self.surface_channels = int(surface_channels)
        self.volume_channels = int(volume_channels)
        self.parameter_channels = int(parameter_channels)
        self.latent_dim = int(latent_dim)
        self.subsampled_geometry_points = int(subsampled_geometry_points)
        self.pos_scale_factor = float(pos_scale_factor)
        self.subregion_size = int(subregion_size)

        self.geometry_stem = nn.Sequential(
            nn.Linear(spatial_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
        )
        self.geometry_refine = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
        )
        self.geometry_global = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim),
        )
        self.geometry_cond = CondInjection(latent_dim, parameter_channels)

        self.query_embed = nn.Sequential(
            nn.Linear(spatial_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim),
        )
        self.query_type = QueryTypeEmbedding(latent_dim)
        self.query_blocks = nn.ModuleList(
            [PlainMLP(dim=latent_dim, hidden_dim=latent_dim * 4, dropout=dropout) for _ in range(num_encoder_decoder_blocks)]
        )
        self.query_cond = CondInjection(latent_dim, parameter_channels)
        self.global_to_query = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim),
        )
        self.output_head = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, surface_channels + volume_channels),
        )
        self.apply(init_linear_layer_weights)

    def _select_geometry_tokens(self, geo):
        geo_pos = geo * self.pos_scale_factor
        geo_tokens, _ = sample_tokens(geo_pos, self.subsampled_geometry_points)
        return geo_tokens

    def encode_geometry(self, geo, params=None):
        geo_tokens = self._select_geometry_tokens(geo)
        local_feat = self.geometry_stem(geo_tokens)
        pooled_local = torch.amax(local_feat, dim=1, keepdim=True)
        refined_feat = self.geometry_refine(torch.cat([local_feat, pooled_local.expand_as(local_feat)], dim=-1))
        refined_feat = self.geometry_cond(refined_feat, params)
        global_feat = torch.amax(refined_feat, dim=1)
        global_feat = self.geometry_global(global_feat)
        return refined_feat, global_feat

    def decode_features(self, global_feat, surf_query_pos, vol_query_pos, params=None):
        surf_pos = surf_query_pos * self.pos_scale_factor
        vol_pos = vol_query_pos * self.pos_scale_factor
        surf_tokens = self.query_embed(surf_pos)
        vol_tokens = self.query_embed(vol_pos)
        surf_tokens, vol_tokens = self.query_type(surf_tokens, vol_tokens)
        query_tokens = torch.cat([surf_tokens, vol_tokens], dim=1)
        query_tokens = query_tokens + self.global_to_query(global_feat).unsqueeze(1)
        query_tokens = self.query_cond(query_tokens, params)
        for block in self.query_blocks:
            query_tokens = query_tokens + block(query_tokens, params)
        return query_tokens

    def forward(self, geo, surf_query_pos, vol_query_pos, params=None, geo_log_density=None, return_latent=False):
        point_latents, global_feat = self.encode_geometry(geo, params=params)
        query_features = self.decode_features(global_feat, surf_query_pos, vol_query_pos, params=params)
        pred = self.output_head(query_features)
        pred_surf, pred_vol = split_surface_volume_predictions(pred, surf_query_pos, self.surface_channels)
        if return_latent:
            return pred_surf, pred_vol, point_latents
        return pred_surf, pred_vol

    @torch.inference_mode()
    def inference(self, geo, surf_query_pos, vol_query_pos, params=None, geo_log_density=None):
        point_latents, global_feat = self.encode_geometry(geo, params=params)
        query_count = int(surf_query_pos.shape[1] + vol_query_pos.shape[1])
        if query_count <= self.subregion_size:
            query_features = self.decode_features(global_feat, surf_query_pos, vol_query_pos, params=params)
            pred = self.output_head(query_features)
            return split_surface_volume_predictions(pred, surf_query_pos, self.surface_channels)

        outputs = []
        surf_count = int(surf_query_pos.shape[1])
        full_query = torch.cat([surf_query_pos, vol_query_pos], dim=1)
        for start in range(0, query_count, self.subregion_size):
            stop = min(start + self.subregion_size, query_count)
            query_chunk = full_query[:, start:stop]
            surf_chunk_len = max(0, min(stop, surf_count) - start)
            surf_chunk = query_chunk[:, :surf_chunk_len]
            vol_chunk = query_chunk[:, surf_chunk_len:]
            chunk_features = self.decode_features(global_feat, surf_chunk, vol_chunk, params=params)
            outputs.append(self.output_head(chunk_features))
        pred = torch.cat(outputs, dim=1)
        return split_surface_volume_predictions(pred, surf_query_pos, self.surface_channels)


class PointNet(PointNetBase):
    pass
