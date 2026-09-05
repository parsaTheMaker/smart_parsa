"""Lightweight Geometry-Aware Operator Transformer adapter.

The model follows GAOT's encode-process-decode structure: multiscale
attentional graph operators map geometry to regional tokens, a transformer
processes those tokens globally, and another multiscale graph operator maps
them to arbitrary surface and volume queries.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .family_common import CondInjection, QueryTypeEmbedding, SelfAttentionBlock, init_linear_layer_weights
from .geometry_operator_common import (
    chunked_knn_indices,
    gather_neighbors,
    gather_points,
    sample_point_indices,
)


class _MultiscaleAttentionalGraphOperator(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        spatial_dim: int,
        neighbors: tuple[int, ...],
        geometry_groups: int = 1,
    ):
        super().__init__()
        if not neighbors or any(int(value) <= 0 for value in neighbors):
            raise ValueError("GAOT neighbor scales must contain positive integers.")
        self.neighbors = tuple(sorted({int(value) for value in neighbors}))
        self.geometry_groups = int(geometry_groups)
        if self.geometry_groups <= 0 or int(hidden_dim) % self.geometry_groups != 0:
            raise ValueError("GAOT geometry_groups must divide hidden_dim.")
        geometry_dim = 2 * int(spatial_dim) + 1
        kernel_dim = max(16, int(hidden_dim) // 4)
        self.value_projections = nn.ModuleList(
            [nn.Linear(int(hidden_dim), int(hidden_dim)) for _ in self.neighbors]
        )
        self.kernel_mlps = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(geometry_dim, kernel_dim),
                    nn.GELU(),
                    # One attention logit plus independent gates for feature
                    # groups. This preserves geometry-conditioned messages
                    # without materializing an expensive hidden vector per edge.
                    nn.Linear(kernel_dim, 1 + self.geometry_groups),
                )
                for _ in self.neighbors
            ]
        )
        self.scale_logits = nn.Parameter(torch.zeros(len(self.neighbors)))
        self.output = nn.Sequential(nn.LayerNorm(int(hidden_dim)), nn.Linear(int(hidden_dim), int(hidden_dim)))

    def project_source_features(self, source_features: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Project regional features once before decoding all query chunks."""
        return tuple(projection(source_features) for projection in self.value_projections)

    def forward(
        self,
        source_positions: torch.Tensor,
        source_features: torch.Tensor,
        query_positions: torch.Tensor,
        knn_chunk_size: int,
        projected_source_features: tuple[torch.Tensor, ...] | None = None,
    ) -> torch.Tensor:
        indices = chunked_knn_indices(
            source_positions,
            query_positions,
            max(self.neighbors),
            query_chunk_size=knn_chunk_size,
        )
        neighbor_positions = gather_neighbors(source_positions, indices)
        query = query_positions.unsqueeze(2).expand(-1, -1, indices.shape[2], -1)
        relative = neighbor_positions - query
        distance = torch.linalg.vector_norm(relative.float(), dim=-1, keepdim=True).to(relative.dtype)
        geometry = torch.cat([neighbor_positions, relative, distance], dim=-1)
        scale_outputs = []
        if projected_source_features is None:
            projected_source_features = self.project_source_features(source_features)
        for neighbor_count, projected_source, kernel_mlp in zip(
            self.neighbors, projected_source_features, self.kernel_mlps
        ):
            # Project each source token once; only the inexpensive geometric
            # kernel is evaluated per edge. This is substantially leaner than
            # repeating a hidden-to-hidden MLP for every query-neighbor pair.
            values = gather_neighbors(projected_source, indices[:, :, :neighbor_count])
            kernel = kernel_mlp(geometry[:, :, :neighbor_count])
            attention = torch.softmax(kernel[..., :1].float(), dim=2).to(values.dtype)
            gates = 1.0 + torch.tanh(kernel[..., 1:]).to(values.dtype)
            values = values.reshape(*values.shape[:-1], self.geometry_groups, -1)
            values = values * gates.unsqueeze(-1)
            values = values.reshape(*values.shape[:-2], -1)
            scale_outputs.append(torch.sum(values * attention, dim=2))
        scale_weights = torch.softmax(self.scale_logits.float(), dim=0).to(scale_outputs[0].dtype)
        mixed = sum(weight * value for weight, value in zip(scale_weights, scale_outputs))
        return self.output(mixed)


class GAOT(nn.Module):
    """Memory-bounded GAOT adapter for variable 3D point distributions."""

    expects_geo_log_density = False

    def __init__(
        self,
        spatial_dim=3,
        surface_channels=1,
        volume_channels=3,
        parameter_channels=0,
        hidden_dim=128,
        regional_tokens=512,
        encoder_neighbors=(16, 32),
        decoder_neighbors=(8, 16),
        processor_depth=4,
        num_heads=4,
        geometry_groups=1,
        dropout=0.0,
        pos_scale_factor=100.0,
        encoder_knn_chunk_size=64,
        query_chunk_size=4096,
        **_unused,
    ):
        super().__init__()
        if int(hidden_dim) % int(num_heads) != 0:
            raise ValueError("GAOT hidden_dim must be divisible by num_heads.")
        self.spatial_dim = int(spatial_dim)
        self.surface_channels = int(surface_channels)
        self.volume_channels = int(volume_channels)
        self.parameter_channels = int(parameter_channels)
        self.regional_tokens = int(regional_tokens)
        self.pos_scale_factor = float(pos_scale_factor)
        self.encoder_knn_chunk_size = max(1, int(encoder_knn_chunk_size))
        self.query_chunk_size = max(1, int(query_chunk_size))
        self.geometry_groups = int(geometry_groups)

        self.geometry_lift = nn.Sequential(
            nn.Linear(self.spatial_dim, int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
        )
        self.region_pos_embed = nn.Sequential(
            nn.Linear(self.spatial_dim, int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
        )
        self.encoder = _MultiscaleAttentionalGraphOperator(
            int(hidden_dim),
            self.spatial_dim,
            tuple(int(value) for value in encoder_neighbors),
            geometry_groups=self.geometry_groups,
        )
        self.region_cond = CondInjection(int(hidden_dim), int(parameter_channels))
        self.processor = nn.ModuleList(
            [
                SelfAttentionBlock(
                    int(hidden_dim), int(num_heads), float(dropout), self.spatial_dim, int(parameter_channels)
                )
                for _ in range(int(processor_depth))
            ]
        )
        self.decoder = _MultiscaleAttentionalGraphOperator(
            int(hidden_dim),
            self.spatial_dim,
            tuple(int(value) for value in decoder_neighbors),
            geometry_groups=self.geometry_groups,
        )
        self.query_embed = nn.Sequential(
            nn.Linear(self.spatial_dim, int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
        )
        self.query_type = QueryTypeEmbedding(int(hidden_dim))
        self.query_cond = CondInjection(int(hidden_dim), int(parameter_channels))
        self.surface_head = nn.Sequential(
            nn.LayerNorm(int(hidden_dim)), nn.Linear(int(hidden_dim), self.surface_channels)
        )
        self.volume_head = nn.Sequential(
            nn.LayerNorm(int(hidden_dim)), nn.Linear(int(hidden_dim), self.volume_channels)
        )
        self.apply(init_linear_layer_weights)

    def encode_geometry(self, geo: torch.Tensor, params: torch.Tensor | None):
        indices = sample_point_indices(geo, self.regional_tokens, self.training)
        region_positions = gather_points(geo, indices)
        geometry_features = self.geometry_lift(geo * self.pos_scale_factor)
        regions = self.encoder(
            geo,
            geometry_features,
            region_positions,
            knn_chunk_size=self.encoder_knn_chunk_size,
        )
        regions = regions + self.region_pos_embed(region_positions * self.pos_scale_factor)
        regions = self.region_cond(regions, params)
        scaled_positions = region_positions * self.pos_scale_factor
        for block in self.processor:
            regions = block(regions, params=params, pos=scaled_positions)
        return regions, region_positions

    def _decode_domain(
        self,
        regions: torch.Tensor,
        region_positions: torch.Tensor,
        queries: torch.Tensor,
        params: torch.Tensor | None,
        is_surface: bool,
    ) -> torch.Tensor:
        output_chunks = []
        projected_regions = self.decoder.project_source_features(regions)
        for start in range(0, queries.shape[1], self.query_chunk_size):
            positions = queries[:, start : start + self.query_chunk_size]
            tokens = self.decoder(
                region_positions,
                regions,
                positions,
                knn_chunk_size=self.query_chunk_size,
                projected_source_features=projected_regions,
            )
            query_embedding = self.query_embed(positions * self.pos_scale_factor)
            if is_surface:
                query_embedding = query_embedding + self.query_type.surface
            else:
                query_embedding = query_embedding + self.query_type.volume
            tokens = self.query_cond(tokens + query_embedding, params)
            output_chunks.append(self.surface_head(tokens) if is_surface else self.volume_head(tokens))
        return torch.cat(output_chunks, dim=1)

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
            raise ValueError("GAOT was configured with parameter channels but received no parameters.")
        regions, region_positions = self.encode_geometry(geo, params)
        pred_surf = self._decode_domain(regions, region_positions, surf_query_pos, params, True)
        pred_vol = self._decode_domain(regions, region_positions, vol_query_pos, params, False)
        if return_latent:
            return pred_surf, pred_vol, regions
        return pred_surf, pred_vol

    @torch.inference_mode()
    def inference(self, geo, surf_query_pos, vol_query_pos, params=None, geo_log_density=None):
        return self.forward(geo, surf_query_pos, vol_query_pos, params, geo_log_density)
