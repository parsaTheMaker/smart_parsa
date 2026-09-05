"""Lightweight anchored branched UPT for surface/volume field prediction.

This adapter independently implements the AB-UPT design: geometry supernode
pooling, surface/volume anchor branches, cross-domain attention, and arbitrary
query decoding. It uses the repository's operator interface and does not copy
the Noether implementation.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .family_common import (
    CondInjection,
    PlainMLP,
    QueryTypeEmbedding,
    SimulationParamModulatedMLP,
    SelfAttentionBlock,
    SharedCrossModalBlock,
    SharedPerceiverBlock,
    SharedSplitBlock,
    init_linear_layer_weights,
)
from .geometry_operator_common import (
    chunked_knn_indices,
    gather_neighbors,
    gather_points,
    sample_point_indices,
)
from .smart.smart import CrossAttention


class _SupernodePooling(nn.Module):
    def __init__(self, spatial_dim: int, hidden_dim: int, neighbors: int, knn_chunk_size: int):
        super().__init__()
        self.neighbors = int(neighbors)
        self.knn_chunk_size = int(knn_chunk_size)
        self.edge_mlp = nn.Sequential(
            nn.Linear(spatial_dim * 2 + 1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, max(16, hidden_dim // 4)),
            nn.GELU(),
            nn.Linear(max(16, hidden_dim // 4), 1),
        )

    def forward(self, geometry: torch.Tensor, supernodes: torch.Tensor) -> torch.Tensor:
        indices = chunked_knn_indices(
            geometry,
            supernodes,
            self.neighbors,
            query_chunk_size=self.knn_chunk_size,
        )
        neighbors = gather_neighbors(geometry, indices)
        centers = supernodes.unsqueeze(2).expand_as(neighbors)
        relative = neighbors - centers
        distance = torch.linalg.vector_norm(relative.float(), dim=-1, keepdim=True).to(relative.dtype)
        edge = self.edge_mlp(torch.cat([neighbors, relative, distance], dim=-1))
        weights = torch.softmax(self.attention(edge).float(), dim=2).to(edge.dtype)
        return torch.sum(edge * weights, dim=2)


class _AnchorDecoder(nn.Module):
    def __init__(
        self,
        spatial_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_heads: int,
        depth: int,
        dropout: float,
        parameter_channels: int,
        query_chunk_size: int,
    ):
        super().__init__()
        self.query_chunk_size = max(1, int(query_chunk_size))
        self.query_embed = nn.Sequential(
            nn.Linear(spatial_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.query_cond = CondInjection(hidden_dim, parameter_channels)
        self.cross = CrossAttention(hidden_dim, num_heads=num_heads, spatial_dim=spatial_dim, dropout=dropout)
        self.blocks = nn.ModuleList(
            [
                SimulationParamModulatedMLP(
                    dim=hidden_dim,
                    hidden_dim=hidden_dim * 4,
                    cond_dim=parameter_channels,
                    dropout=dropout,
                )
                if parameter_channels > 0
                else PlainMLP(dim=hidden_dim, hidden_dim=hidden_dim * 4, dropout=dropout)
                for _ in range(int(depth))
            ]
        )
        self.output = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, output_dim))

    def forward(
        self,
        queries: torch.Tensor,
        anchors: torch.Tensor,
        anchor_positions: torch.Tensor,
        params: torch.Tensor | None,
        position_scale: float,
    ) -> torch.Tensor:
        chunks = []
        scaled_anchor_positions = anchor_positions * position_scale
        prepared_anchors = self.cross.prepare_kv(anchors, kv_pos=scaled_anchor_positions)
        for start in range(0, queries.shape[1], self.query_chunk_size):
            positions = queries[:, start : start + self.query_chunk_size]
            scaled_positions = positions * position_scale
            tokens = self.query_embed(scaled_positions)
            tokens = self.query_cond(tokens, params)
            tokens = tokens + self.cross.forward_prepared(
                tokens,
                prepared_anchors,
                q_pos=scaled_positions,
            )
            for block in self.blocks:
                tokens = tokens + block(tokens, params)
            chunks.append(self.output(tokens))
        return torch.cat(chunks, dim=1)


class ABUPT(nn.Module):
    """Memory-efficient AB-UPT adapter with separate surface/volume anchors."""

    expects_geo_log_density = False

    def __init__(
        self,
        spatial_dim=3,
        surface_channels=1,
        volume_channels=3,
        parameter_channels=0,
        hidden_dim=128,
        num_heads=4,
        geometry_supernodes=1024,
        supernode_neighbors=32,
        surface_anchors=512,
        volume_anchors=512,
        geometry_depth=1,
        physics_depth=3,
        interleaved_depth=None,
        final_self_depth=0,
        domain_decoder_depth=1,
        dropout=0.0,
        pos_scale_factor=100.0,
        knn_chunk_size=64,
        query_chunk_size=8192,
        **_unused,
    ):
        super().__init__()
        if int(hidden_dim) % int(num_heads) != 0:
            raise ValueError("AB-UPT hidden_dim must be divisible by num_heads.")
        self.spatial_dim = int(spatial_dim)
        self.surface_channels = int(surface_channels)
        self.volume_channels = int(volume_channels)
        self.parameter_channels = int(parameter_channels)
        self.geometry_supernodes = int(geometry_supernodes)
        self.surface_anchors = int(surface_anchors)
        self.volume_anchors = int(volume_anchors)
        self.pos_scale_factor = float(pos_scale_factor)

        self.geometry_pool = _SupernodePooling(
            self.spatial_dim, int(hidden_dim), int(supernode_neighbors), int(knn_chunk_size)
        )
        self.geometry_pos_embed = nn.Sequential(
            nn.Linear(self.spatial_dim, int(hidden_dim)), nn.GELU(), nn.Linear(int(hidden_dim), int(hidden_dim))
        )
        self.geometry_cond = CondInjection(int(hidden_dim), int(parameter_channels))
        self.geometry_blocks = nn.ModuleList(
            [
                SelfAttentionBlock(
                    int(hidden_dim), int(num_heads), float(dropout), self.spatial_dim, int(parameter_channels)
                )
                for _ in range(int(geometry_depth))
            ]
        )

        self.anchor_pos_embed = nn.Sequential(
            nn.Linear(self.spatial_dim, int(hidden_dim)), nn.GELU(), nn.Linear(int(hidden_dim), int(hidden_dim))
        )
        self.anchor_type = QueryTypeEmbedding(int(hidden_dim))
        self.anchor_cond = CondInjection(int(hidden_dim), int(parameter_channels))
        self.geometry_injection = SharedPerceiverBlock(
            int(hidden_dim), int(num_heads), float(dropout), self.spatial_dim, int(parameter_channels)
        )
        # The coupled blocks communicate between surface and volume anchors.
        # Subsequent domain-specific blocks retain the branch capacity required
        # for surface-only and volume-only physics after this exchange.
        interleaved_depth = int(physics_depth if interleaved_depth is None else interleaved_depth)
        self.self_blocks = nn.ModuleList(
            [
                SharedSplitBlock(
                    int(hidden_dim), int(num_heads), float(dropout), self.spatial_dim, int(parameter_channels)
                )
                for _ in range(interleaved_depth)
            ]
        )
        self.cross_blocks = nn.ModuleList(
            [
                SharedCrossModalBlock(
                    int(hidden_dim), int(num_heads), float(dropout), self.spatial_dim, int(parameter_channels)
                )
                for _ in range(interleaved_depth)
            ]
        )
        self.surface_final_blocks = nn.ModuleList(
            [
                SelfAttentionBlock(
                    int(hidden_dim), int(num_heads), float(dropout), self.spatial_dim, int(parameter_channels)
                )
                for _ in range(int(final_self_depth))
            ]
        )
        self.volume_final_blocks = nn.ModuleList(
            [
                SelfAttentionBlock(
                    int(hidden_dim), int(num_heads), float(dropout), self.spatial_dim, int(parameter_channels)
                )
                for _ in range(int(final_self_depth))
            ]
        )
        self.surface_decoder = _AnchorDecoder(
            self.spatial_dim,
            int(hidden_dim),
            self.surface_channels,
            int(num_heads),
            int(domain_decoder_depth),
            float(dropout),
            int(parameter_channels),
            int(query_chunk_size),
        )
        self.volume_decoder = _AnchorDecoder(
            self.spatial_dim,
            int(hidden_dim),
            self.volume_channels,
            int(num_heads),
            int(domain_decoder_depth),
            float(dropout),
            int(parameter_channels),
            int(query_chunk_size),
        )
        self.apply(init_linear_layer_weights)

    def _sample_positions(self, positions: torch.Tensor, count: int) -> torch.Tensor:
        indices = sample_point_indices(positions, count, self.training)
        return gather_points(positions, indices)

    def encode_geometry(self, geo: torch.Tensor, params: torch.Tensor | None):
        supernode_pos = self._sample_positions(geo, self.geometry_supernodes)
        scaled_pos = supernode_pos * self.pos_scale_factor
        tokens = self.geometry_pool(geo, supernode_pos) + self.geometry_pos_embed(scaled_pos)
        tokens = self.geometry_cond(tokens, params)
        for block in self.geometry_blocks:
            tokens = block(tokens, params=params, pos=scaled_pos)
        return tokens, supernode_pos

    def encode_anchors(
        self,
        geometry_tokens: torch.Tensor,
        geometry_positions: torch.Tensor,
        surf_query_pos: torch.Tensor,
        vol_query_pos: torch.Tensor,
        params: torch.Tensor | None,
    ):
        surf_pos = self._sample_positions(surf_query_pos, self.surface_anchors)
        vol_pos = self._sample_positions(vol_query_pos, self.volume_anchors)
        surf = self.anchor_pos_embed(surf_pos * self.pos_scale_factor)
        vol = self.anchor_pos_embed(vol_pos * self.pos_scale_factor)
        surf, vol = self.anchor_type(surf, vol)
        surf = self.anchor_cond(surf, params)
        vol = self.anchor_cond(vol, params)
        geometry_scaled = geometry_positions * self.pos_scale_factor
        surf = self.geometry_injection(
            surf,
            geometry_tokens,
            params=params,
            x_pos=surf_pos * self.pos_scale_factor,
            geometry_pos=geometry_scaled,
        )
        vol = self.geometry_injection(
            vol,
            geometry_tokens,
            params=params,
            x_pos=vol_pos * self.pos_scale_factor,
            geometry_pos=geometry_scaled,
        )
        for self_block, cross_block in zip(self.self_blocks, self.cross_blocks):
            surf, vol = self_block(
                surf,
                vol,
                params=params,
                surf_pos=surf_pos * self.pos_scale_factor,
                vol_pos=vol_pos * self.pos_scale_factor,
            )
            surf, vol = cross_block(
                surf,
                vol,
                params=params,
                surf_pos=surf_pos * self.pos_scale_factor,
                vol_pos=vol_pos * self.pos_scale_factor,
            )
        for block in self.surface_final_blocks:
            surf = block(surf, params=params, pos=surf_pos * self.pos_scale_factor)
        for block in self.volume_final_blocks:
            vol = block(vol, params=params, pos=vol_pos * self.pos_scale_factor)
        return surf, vol, surf_pos, vol_pos

    def forward(
        self,
        geo,
        surf_query_pos,
        vol_query_pos,
        params=None,
        geo_log_density=None,
        return_latent=False,
    ):
        del geo_log_density
        if self.parameter_channels > 0 and params is None:
            raise ValueError("AB-UPT was configured with parameter channels but received no parameters.")
        geometry, geometry_pos = self.encode_geometry(geo, params)
        surf_anchor, vol_anchor, surf_anchor_pos, vol_anchor_pos = self.encode_anchors(
            geometry, geometry_pos, surf_query_pos, vol_query_pos, params
        )
        pred_surf = self.surface_decoder(
            surf_query_pos, surf_anchor, surf_anchor_pos, params, self.pos_scale_factor
        )
        pred_vol = self.volume_decoder(vol_query_pos, vol_anchor, vol_anchor_pos, params, self.pos_scale_factor)
        if return_latent:
            return pred_surf, pred_vol, torch.cat([surf_anchor, vol_anchor], dim=1)
        return pred_surf, pred_vol

    @torch.inference_mode()
    def inference(self, geo, surf_query_pos, vol_query_pos, params=None, geo_log_density=None):
        return self.forward(geo, surf_query_pos, vol_query_pos, params, geo_log_density)
