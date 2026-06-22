"""SMART-SAT4: SMART with density-corrected geometry attention and paired-view training.

SAT4 is an ablation-style model:
- latent anchors are sampled exactly like SMART;
- latent reuse and decoder attention stay SMART-like;
- only the geometry-to-latent aggregation is density corrected.

This isolates the effect of encoder input density correction from anchor
sampling or latent reuse corrections.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .smart import (
    CrossAttention,
    DecoderBlock,
    ModulatedPositionalEmbedding,
    PlainMLP,
    SimulationParamModulatedMLP,
)
from .smart_sat import DensityCompensatedCrossAttention, sample_geometry

try:
    from utils.geometry_density import estimate_log_sampling_density
except ImportError:  # pragma: no cover - package-style imports
    from smart.utils.geometry_density import estimate_log_sampling_density


class DensityCorrectedInputEncoderBlock(nn.Module):
    """SMART encoder block with density correction only on input aggregation."""

    def __init__(self, dim, num_heads=8, dropout=0.1, spatial_dim=3, cond_dim=2):
        super().__init__()
        self.geo_attn = DensityCompensatedCrossAttention(dim=dim, num_heads=num_heads, spatial_dim=spatial_dim)
        self.cross_attn = CrossAttention(dim=dim, num_heads=num_heads, dropout=dropout, spatial_dim=spatial_dim)
        self.attn_dropout = nn.Dropout(dropout)

        if cond_dim > 0:
            self.mlp = SimulationParamModulatedMLP(dim=dim, hidden_dim=dim * 4, cond_dim=cond_dim, dropout=dropout)
        else:
            self.mlp = PlainMLP(dim=dim, hidden_dim=dim * 4, dropout=dropout)

    def forward(
        self,
        latent_geometry,
        geometry_tokens,
        params,
        latent_geometry_pos=None,
        geometry_pos=None,
        geometry_log_density=None,
    ):
        latent_geometry_cross = latent_geometry + self.attn_dropout(
            self.geo_attn(
                q=latent_geometry,
                kv=geometry_tokens,
                q_pos=latent_geometry_pos,
                kv_pos=geometry_pos,
                kv_log_density=geometry_log_density,
            )
        )
        latent_geometry_self = latent_geometry + self.attn_dropout(
            self.cross_attn(
                q=latent_geometry,
                kv=latent_geometry_cross,
                q_pos=latent_geometry_pos,
                kv_pos=latent_geometry_pos,
            )
        )
        latent_geometry_mlp = latent_geometry_self + self.mlp(latent_geometry_self, params)
        return latent_geometry_mlp, latent_geometry_cross


class SMARTSAT4(nn.Module):
    """SMART variant with density-corrected input attention only."""

    def __init__(
        self,
        spatial_dim=3,
        surface_channels=1,
        volume_channels=3,
        parameter_channels=2,
        latent_dim=256,
        latent_geometry_points=4096,
        subsampled_geometry_points=16384,
        num_encoder_decoder_blocks=8,
        num_heads=8,
        pos_scale_factor=1000,
        dropout=0.0,
        subregion_size=262144,
        density_knn_k=24,
        density_neighbor_hops=1,
        density_estimator="tangent_cov",
        subsampled_geometry_with_replacement=False,
    ):
        super().__init__()
        assert surface_channels > 0 and volume_channels > 0, "surface_channels and volume_channels must be positive integers."

        self.surface_channels = surface_channels
        self.volume_channels = volume_channels
        self.num_geo = latent_geometry_points
        self.subsampled_geometry_points = subsampled_geometry_points
        self.subsampled_geometry_with_replacement = bool(subsampled_geometry_with_replacement)
        self.pos_scale_factor = pos_scale_factor
        self.density_knn_k = int(density_knn_k)
        self.density_neighbor_hops = int(density_neighbor_hops)
        self.density_estimator = str(density_estimator)
        self.pos_encoder = ModulatedPositionalEmbedding(latent_dim, spatial_dim)

        self.encoder_blocks = nn.ModuleList(
            [
                DensityCorrectedInputEncoderBlock(
                    dim=latent_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    spatial_dim=spatial_dim,
                    cond_dim=parameter_channels,
                )
                for _ in range(num_encoder_decoder_blocks)
            ]
        )
        self.decoder_blocks = nn.ModuleList(
            [
                DecoderBlock(
                    dim=latent_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    spatial_dim=spatial_dim,
                    cond_dim=parameter_channels,
                    shared_attn=self.encoder_blocks[i].cross_attn,
                    shared_mlp=self.encoder_blocks[i].mlp,
                )
                for i in range(num_encoder_decoder_blocks)
            ]
        )
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, surface_channels + volume_channels),
        )
        self.subregion_size = subregion_size

    def initialize_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def encode(self, geo, params, return_final=False, geo_log_density=None):
        if geo_log_density is None:
            geo_log_density = estimate_log_sampling_density(
                geo,
                knn_k=self.density_knn_k,
                neighbor_hops=self.density_neighbor_hops,
                estimator=self.density_estimator,
            )
        geo_log_density = geo_log_density.to(device=geo.device, dtype=geo.dtype)
        geo = geo * self.pos_scale_factor

        latent_geo_pos, _ = sample_geometry(geo, self.num_geo)
        latent_geo_emb = self.pos_encoder(latent_geo_pos)

        intermediate_latent_geometries = []
        for block in self.encoder_blocks:
            sub_geo_pos, sub_idx = sample_geometry(
                geo,
                self.subsampled_geometry_points,
                with_replacement=self.subsampled_geometry_with_replacement,
            )
            sub_geo_emb = self.pos_encoder(sub_geo_pos)
            sub_geo_log_density = torch.gather(geo_log_density, 1, sub_idx)
            latent_geo_emb, e_ca = block(
                latent_geo_emb,
                sub_geo_emb,
                params,
                latent_geometry_pos=latent_geo_pos,
                geometry_pos=sub_geo_pos,
                geometry_log_density=sub_geo_log_density,
            )
            intermediate_latent_geometries.append(e_ca)

        if return_final:
            return intermediate_latent_geometries, latent_geo_pos, latent_geo_emb
        return intermediate_latent_geometries, latent_geo_pos

    def decode_features(self, intermediate_latent_geometries, latent_geo_pos, params, query_pos):
        query_pos = query_pos * self.pos_scale_factor
        query_emb = self.pos_encoder(query_pos)
        for e_ca, block in zip(intermediate_latent_geometries, self.decoder_blocks):
            query_emb = block(query_emb, e_ca, params, queries_pos=query_pos, latent_geometry_pos=latent_geo_pos)
        return query_emb

    def decode(self, intermediate_latent_geometries, latent_geo_pos, params, query_pos):
        return self.mlp(self.decode_features(intermediate_latent_geometries, latent_geo_pos, params, query_pos))

    def forward(self, geo, surf_query_pos, vol_query_pos, params, geo_log_density=None, return_latent=False):
        intermediate_latent_geometries, latent_geo_pos, final_latent = self.encode(
            geo,
            params,
            return_final=True,
            geo_log_density=geo_log_density,
        )
        query_pos = torch.cat([surf_query_pos, vol_query_pos], dim=1)
        pred = self.decode(intermediate_latent_geometries, latent_geo_pos, params, query_pos)
        pred_surf = pred[:, :surf_query_pos.shape[1], : self.surface_channels]
        pred_vol = pred[:, surf_query_pos.shape[1] :, self.surface_channels :]
        if return_latent:
            return pred_surf, pred_vol, final_latent
        return pred_surf, pred_vol

    @torch.inference_mode()
    def inference(self, geo, surf_query_pos, vol_query_pos, params, geo_log_density=None):
        intermediate_latent_geometries, latent_geo_pos = self.encode(
            geo,
            params,
            return_final=False,
            geo_log_density=geo_log_density,
        )
        n_surf = surf_query_pos.shape[1]
        y_hat_surf_subregions = []
        for i in range(0, n_surf, self.subregion_size):
            surf_subregion = surf_query_pos[:, i:i + self.subregion_size, :]
            y_surf_subregion = self.decode(intermediate_latent_geometries, latent_geo_pos, params, surf_subregion)
            y_hat_surf_subregions.append(y_surf_subregion)
        y_hat_surf = torch.cat(y_hat_surf_subregions, dim=1)

        n_vol = vol_query_pos.shape[1]
        y_hat_vol_subregions = []
        for i in range(0, n_vol, self.subregion_size):
            vol_subregion = vol_query_pos[:, i:i + self.subregion_size, :]
            y_vol_subregion = self.decode(intermediate_latent_geometries, latent_geo_pos, params, vol_subregion)
            y_hat_vol_subregions.append(y_vol_subregion)
        y_hat_vol = torch.cat(y_hat_vol_subregions, dim=1)

        pred_surf = y_hat_surf[:, :, : self.surface_channels]
        pred_vol = y_hat_vol[:, :, self.surface_channels :]
        return pred_surf, pred_vol
