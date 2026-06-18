from __future__ import annotations

import torch
import torch.nn as nn

from .family_common import (
    CondInjection,
    GeometryCrossBlock,
    ModulatedPositionalEmbedding,
    QueryTypeEmbedding,
    SelfAttentionBlock,
    init_linear_layer_weights,
    knn_group,
    resolve_geo_log_density,
    sample_tokens_fps,
    sample_tokens_density_compensated_fps,
    split_surface_volume_predictions,
)
from .smart.smart import CrossAttention, PlainMLP, SimulationParamModulatedMLP


class LocalGroupingEncoder(nn.Module):
    """Sampling-and-grouping front-end in the spirit of the public GINOT design."""

    def __init__(self, dim, spatial_dim=3, cond_dim=0):
        super().__init__()
        self.neighbor_mlp = nn.Sequential(
            nn.Linear(spatial_dim * 2, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.GELU(),
        )
        self.post = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.cond = CondInjection(dim, cond_dim)

    def forward(self, full_geo_pos, local_pos, params=None, group_k=32, full_geo_log_density=None):
        neighbors, neighbor_idx = knn_group(full_geo_pos, local_pos, k=group_k)
        rel = neighbors - local_pos.unsqueeze(2)
        feat = self.neighbor_mlp(torch.cat([rel, neighbors], dim=-1))

        if full_geo_log_density is not None:
            neighbor_log_density = torch.gather(
                full_geo_log_density.unsqueeze(1).expand(-1, local_pos.shape[1], -1),
                2,
                neighbor_idx,
            )
            weights = torch.exp(-neighbor_log_density.float())
            weights = weights / torch.clamp(weights.sum(dim=2, keepdim=True), min=1e-6)
            weights = weights.to(dtype=feat.dtype)
        else:
            weights = torch.full(
                feat.shape[:3],
                1.0 / float(max(feat.shape[2], 1)),
                device=feat.device,
                dtype=feat.dtype,
            )

        feat_mean = (feat * weights.unsqueeze(-1)).sum(dim=2)
        feat_second = (feat.pow(2) * weights.unsqueeze(-1)).sum(dim=2)
        feat_scale = torch.sqrt(torch.clamp(feat_second - feat_mean.pow(2), min=1e-6))
        local_feat = self.post(torch.cat([feat_mean, feat_scale], dim=-1))
        return self.cond(local_feat, params)


class GINOTGeometryEncoder(nn.Module):
    def __init__(
        self,
        dim,
        local_geometry_points,
        latent_geometry_points,
        local_group_k,
        local_depth,
        global_depth,
        num_heads,
        dropout,
        spatial_dim,
        cond_dim,
        pos_scale_factor,
        density_compensated=False,
        density_knn_k=8,
        density_neighbor_hops=1,
        density_estimator="rk2",
    ):
        super().__init__()
        self.local_geometry_points = local_geometry_points
        self.latent_geometry_points = latent_geometry_points
        self.local_group_k = int(local_group_k)
        self.pos_scale_factor = pos_scale_factor
        self.density_compensated = bool(density_compensated)
        self.density_knn_k = int(density_knn_k)
        self.density_neighbor_hops = int(density_neighbor_hops)
        self.density_estimator = str(density_estimator)

        self.pos_encoder = ModulatedPositionalEmbedding(dim, spatial_dim)
        self.local_group_encoder = LocalGroupingEncoder(dim=dim, spatial_dim=spatial_dim, cond_dim=cond_dim)
        self.local_blocks = nn.ModuleList(
            [
                SelfAttentionBlock(
                    dim=dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    spatial_dim=spatial_dim,
                    cond_dim=cond_dim,
                )
                for _ in range(local_depth)
            ]
        )
        self.global_cross = GeometryCrossBlock(
            dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            spatial_dim=spatial_dim,
            cond_dim=cond_dim,
            density_compensated=False,
        )
        self.global_self = nn.ModuleList(
            [
                SelfAttentionBlock(
                    dim=dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    spatial_dim=spatial_dim,
                    cond_dim=cond_dim,
                )
                for _ in range(global_depth)
            ]
        )

    def forward(self, geo, params=None, geo_log_density=None):
        full_geo_log_density = None
        if self.density_compensated:
            full_geo_log_density = resolve_geo_log_density(
                geo,
                geo_log_density,
                knn_k=self.density_knn_k,
                neighbor_hops=self.density_neighbor_hops,
                estimator=self.density_estimator,
            )

        geo_pos = geo * self.pos_scale_factor
        if self.density_compensated and full_geo_log_density is not None:
            local_pos, _ = sample_tokens_density_compensated_fps(
                geo_pos,
                self.local_geometry_points,
                full_geo_log_density,
            )
        else:
            local_pos, _ = sample_tokens_fps(geo_pos, self.local_geometry_points, random_start=False)

        local_tokens = self.pos_encoder(local_pos)
        local_tokens = local_tokens + self.local_group_encoder(
            geo_pos,
            local_pos,
            params=params,
            group_k=self.local_group_k,
            full_geo_log_density=full_geo_log_density,
        )
        for block in self.local_blocks:
            local_tokens = block(local_tokens, params=params, pos=local_pos)

        # Avoid the previous SAT bug: once local centers are density-corrected,
        # the global support is chosen geometrically from that corrected set.
        global_pos, _ = sample_tokens_fps(local_pos, self.latent_geometry_points, random_start=False)
        global_tokens = self.pos_encoder(global_pos)
        global_tokens = self.global_cross(global_tokens, local_tokens, params=params, q_pos=global_pos, kv_pos=local_pos)
        for block in self.global_self:
            global_tokens = block(global_tokens, params=params, pos=global_pos)
        return global_tokens, global_pos


class GINOTDecoderBlock(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.0, spatial_dim=3, cond_dim=0):
        super().__init__()
        self.cross = CrossAttention(dim=dim, num_heads=num_heads, dropout=dropout, spatial_dim=spatial_dim)
        self.dropout = nn.Dropout(dropout)
        self.mlp = (
            SimulationParamModulatedMLP(dim=dim, hidden_dim=dim * 4, cond_dim=cond_dim, dropout=dropout)
            if cond_dim > 0
            else PlainMLP(dim=dim, hidden_dim=dim * 4, dropout=dropout)
        )

    def forward(self, queries, geometry_latents, params=None, query_pos=None, geometry_pos=None):
        queries = queries + self.dropout(self.cross(q=queries, kv=geometry_latents, q_pos=query_pos, kv_pos=geometry_pos))
        queries = queries + self.mlp(queries, params)
        return queries


class GINOTBase(nn.Module):
    expects_geo_log_density = False

    def __init__(
        self,
        spatial_dim=3,
        surface_channels=1,
        volume_channels=3,
        parameter_channels=0,
        latent_dim=256,
        latent_geometry_points=4096,
        subsampled_geometry_points=65536,  # kept for config compatibility
        local_geometry_points=4096,
        local_group_k=32,
        num_encoder_decoder_blocks=8,
        num_heads=8,
        pos_scale_factor=100,
        dropout=0.0,
        subregion_size=262144,
        density_compensated=False,
        density_knn_k=8,
        density_neighbor_hops=1,
        density_estimator="rk2",
    ):
        super().__init__()
        self.surface_channels = surface_channels
        self.volume_channels = volume_channels
        self.subregion_size = subregion_size
        self.pos_scale_factor = pos_scale_factor
        self.expects_geo_log_density = bool(density_compensated)

        local_depth = max(1, num_encoder_decoder_blocks // 3)
        global_depth = max(1, num_encoder_decoder_blocks // 3)
        decoder_depth = max(1, num_encoder_decoder_blocks)

        self.query_pos_encoder = ModulatedPositionalEmbedding(latent_dim, spatial_dim)
        self.query_types = QueryTypeEmbedding(latent_dim)
        self.query_cond = CondInjection(latent_dim, parameter_channels)
        self.geometry_encoder = GINOTGeometryEncoder(
            dim=latent_dim,
            local_geometry_points=local_geometry_points,
            latent_geometry_points=latent_geometry_points,
            local_group_k=local_group_k,
            local_depth=local_depth,
            global_depth=global_depth,
            num_heads=num_heads,
            dropout=dropout,
            spatial_dim=spatial_dim,
            cond_dim=parameter_channels,
            pos_scale_factor=pos_scale_factor,
            density_compensated=density_compensated,
            density_knn_k=density_knn_k,
            density_neighbor_hops=density_neighbor_hops,
            density_estimator=density_estimator,
        )
        self.decoder_blocks = nn.ModuleList(
            [
                GINOTDecoderBlock(
                    dim=latent_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    spatial_dim=spatial_dim,
                    cond_dim=parameter_channels,
                )
                for _ in range(decoder_depth)
            ]
        )
        self.head = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, surface_channels + volume_channels),
        )
        self.apply(init_linear_layer_weights)

    def encode_geometry(self, geo, params=None, geo_log_density=None):
        return self.geometry_encoder(geo, params=params, geo_log_density=geo_log_density)

    def decode_features(self, geometry_latents, geometry_pos, surf_query_pos, vol_query_pos, params=None):
        surf_pos = surf_query_pos * self.pos_scale_factor
        vol_pos = vol_query_pos * self.pos_scale_factor
        surf_q = self.query_pos_encoder(surf_pos)
        vol_q = self.query_pos_encoder(vol_pos)
        surf_q, vol_q = self.query_types(surf_q, vol_q)
        surf_q = self.query_cond(surf_q, params)
        vol_q = self.query_cond(vol_q, params)
        queries = torch.cat([surf_q, vol_q], dim=1)
        query_pos = torch.cat([surf_pos, vol_pos], dim=1)
        for block in self.decoder_blocks:
            queries = block(queries, geometry_latents, params=params, query_pos=query_pos, geometry_pos=geometry_pos)
        return queries

    def forward(self, geo, surf_query_pos, vol_query_pos, params=None, geo_log_density=None):
        geometry_latents, geometry_pos = self.encode_geometry(geo, params=params, geo_log_density=geo_log_density)
        query_features = self.decode_features(geometry_latents, geometry_pos, surf_query_pos, vol_query_pos, params=params)
        pred = self.head(query_features)
        return split_surface_volume_predictions(pred, surf_query_pos, self.surface_channels)

    @torch.inference_mode()
    def inference(self, geo, surf_query_pos, vol_query_pos, params=None, geo_log_density=None):
        geometry_latents, geometry_pos = self.encode_geometry(geo, params=params, geo_log_density=geo_log_density)

        surf_parts = []
        for i in range(0, surf_query_pos.shape[1], self.subregion_size):
            surf_chunk = surf_query_pos[:, i : i + self.subregion_size]
            pred = self.head(self.decode_features(geometry_latents, geometry_pos, surf_chunk, vol_query_pos[:, :0], params=params))
            surf_parts.append(pred[:, :, : self.surface_channels])
        pred_surf = torch.cat(surf_parts, dim=1)

        vol_parts = []
        for i in range(0, vol_query_pos.shape[1], self.subregion_size):
            vol_chunk = vol_query_pos[:, i : i + self.subregion_size]
            pred = self.head(self.decode_features(geometry_latents, geometry_pos, surf_query_pos[:, :0], vol_chunk, params=params))
            vol_parts.append(pred[:, :, self.surface_channels :])
        pred_vol = torch.cat(vol_parts, dim=1)
        return pred_surf, pred_vol


class GINOT(GINOTBase):
    def __init__(self, **kwargs):
        super().__init__(density_compensated=False, **kwargs)
