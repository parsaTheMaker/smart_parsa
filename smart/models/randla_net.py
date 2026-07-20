"""RandLA-Net encoder adapted for mesh-free DrivAerML field queries.

The point encoder follows RandLA-Net's defining operations: random
decimation, local spatial encoding, attentive neighbour pooling, residual
local-feature aggregation, and nearest-neighbour feature restoration.  The
restored point features are then interpolated at arbitrary surface/volume
queries by the SMART operator adapter.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .family_common import CondInjection, QueryTypeEmbedding, split_surface_volume_predictions

try:
    from torch_cluster import knn as torch_cluster_knn
except ImportError:  # pragma: no cover - the project environment installs torch-cluster
    torch_cluster_knn = None


def _knn_indices(points: torch.Tensor, centers: torch.Tensor, k: int) -> torch.Tensor:
    """GPU KNN with torch_cluster's query/source edge ordering handled correctly."""
    batch_size, num_points, _ = points.shape
    num_centers = centers.shape[1]
    k = min(max(1, int(k)), num_points)
    if torch_cluster_knn is not None:
        flat_points = points.reshape(batch_size * num_points, -1).float().contiguous()
        flat_centers = centers.reshape(batch_size * num_centers, -1).float().contiguous()
        batch_points = torch.arange(batch_size, device=points.device).repeat_interleave(num_points)
        batch_centers = torch.arange(batch_size, device=points.device).repeat_interleave(num_centers)
        edges = torch_cluster_knn(
            flat_points,
            flat_centers,
            k=k,
            batch_x=batch_points,
            batch_y=batch_centers,
        )
        query_index, source_index = edges[0], edges[1]
        order = torch.argsort(query_index, stable=True)
        return (source_index[order] % num_points).view(batch_size, num_centers, k).to(dtype=torch.long)

    # Bounded fallback for CPU smoke tests and environments without extensions.
    chunks = []
    chunk_size = 2048
    for start in range(0, num_centers, chunk_size):
        distances = torch.cdist(centers[:, start:start + chunk_size].float(), points.float()).square()
        chunks.append(torch.topk(distances, k=k, dim=-1, largest=False).indices)
    return torch.cat(chunks, dim=1)


def _gather_features(features: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    return torch.gather(
        features.unsqueeze(1).expand(-1, indices.shape[1], -1, -1),
        2,
        indices.unsqueeze(-1).expand(-1, -1, -1, features.shape[-1]),
    )


def _index_select_features(features, indices):
    """Gather batched K-neighbor features without expanding source points."""
    batch_size, source_points, channels = features.shape
    offsets = torch.arange(batch_size, device=features.device, dtype=indices.dtype).view(batch_size, 1, 1)
    flat_indices = (indices + offsets * source_points).reshape(-1)
    flat_features = features.reshape(batch_size * source_points, channels)
    return flat_features.index_select(0, flat_indices).view(
        batch_size, indices.shape[1], indices.shape[2], channels
    )


def _weighted_interpolate(source_coords, source_features, target_coords, indices):
    """Interpolate local features while retaining sensitivity to point spacing."""
    neighbor_coords = _index_select_features(source_coords, indices)
    neighbor_features = _index_select_features(source_features, indices)
    distances = torch.linalg.vector_norm(
        neighbor_coords.float() - target_coords.float().unsqueeze(2), dim=-1
    )
    weights = torch.reciprocal(distances.clamp_min(1.0e-5))
    weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1.0e-6)
    interpolated = (weights.to(dtype=neighbor_features.dtype).unsqueeze(-1) * neighbor_features).sum(dim=2)
    effective_distance = (weights * distances).sum(dim=-1, keepdim=True)
    return interpolated, effective_distance


def _spacing_modulate(features, distances):
    """Keep nearest-feature decoding sensitive to the relative local spacing."""
    relative_spacing = distances / distances.detach().mean(dim=1, keepdim=True).clamp_min(1.0e-6)
    modulation = (1.0 + 0.35 * relative_spacing.clamp(0.0, 3.0)).to(dtype=features.dtype)
    return features * modulation


class _SharedMLP(nn.Module):
    """RandLA-Net's shared 1x1 Conv2d MLP for channel-last point tensors."""

    def __init__(self, input_dim, output_dim, activation=True, batch_norm=False):
        super().__init__()
        self.conv = nn.Conv2d(int(input_dim), int(output_dim), kernel_size=1, bias=True)
        self.batch_norm = nn.BatchNorm2d(int(output_dim), eps=1.0e-6, momentum=0.99) if batch_norm else None
        self.activation = activation if isinstance(activation, nn.Module) else (nn.ReLU() if activation else None)

    def forward(self, x):
        if x.ndim == 3:
            channel_first = x.transpose(1, 2).unsqueeze(-1)
            restore = lambda value: value.squeeze(-1).transpose(1, 2)
        elif x.ndim == 4:
            channel_first = x.permute(0, 3, 1, 2)
            restore = lambda value: value.permute(0, 2, 3, 1)
        else:
            raise ValueError(f"RandLA shared MLP expects rank-3 or rank-4 input, got {x.ndim}")
        output = self.conv(channel_first)
        if self.batch_norm is not None:
            # The real DrivAerML hierarchy has many spatial elements at every
            # level.  This fallback keeps tiny CPU smoke tests well-defined.
            normalization_count = output.shape[0] * output.shape[2] * output.shape[3]
            if self.training and normalization_count <= 1:
                output = F.batch_norm(
                    output,
                    self.batch_norm.running_mean,
                    self.batch_norm.running_var,
                    self.batch_norm.weight,
                    self.batch_norm.bias,
                    training=False,
                    eps=self.batch_norm.eps,
                )
            else:
                output = self.batch_norm(output)
        if self.activation is not None:
            output = self.activation(output)
        return restore(output)


class _LocalSpatialEncoding(nn.Module):
    def __init__(self, output_dim, num_neighbors):
        super().__init__()
        self.num_neighbors = int(num_neighbors)
        self.position_mlp = _SharedMLP(10, output_dim, activation=nn.ReLU(), batch_norm=True)

    def forward(self, coords, features, neighbor_indices):
        neighbor_coords = _gather_features(coords, neighbor_indices)
        center_coords = coords.unsqueeze(2).expand_as(neighbor_coords)
        relative = center_coords - neighbor_coords
        distance = torch.linalg.vector_norm(relative.float(), dim=-1, keepdim=True).to(dtype=coords.dtype)
        positional = torch.cat([center_coords, neighbor_coords, relative, distance], dim=-1)
        encoded_position = self.position_mlp(positional)
        center_features = features.unsqueeze(2).expand(-1, -1, neighbor_indices.shape[-1], -1)
        return torch.cat([encoded_position, center_features], dim=-1), distance


class _AttentivePooling(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.score = nn.Linear(int(input_dim), int(input_dim), bias=False)
        # A scalar spacing bias avoids allocating another full K-by-channel
        # attention tensor while keeping the pooling arrangement-sensitive.
        self.distance_score = nn.Linear(1, 1, bias=False)
        self.distance_context = nn.Linear(2, int(input_dim))
        self.output = _SharedMLP(input_dim, output_dim, activation=nn.ReLU(), batch_norm=True)

    def forward(self, features, distances):
        normalized_distances = distances / distances.mean(dim=2, keepdim=True).clamp_min(1.0e-6)
        logits = self.score(features) + 0.25 * self.distance_score(normalized_distances)
        weights = torch.softmax(logits, dim=2)
        pooled = (weights * features).sum(dim=2)
        distance_weights = weights.mean(dim=-1, keepdim=True)
        distance_mean = (distance_weights * distances).sum(dim=2)
        distance_variance = (distance_weights * (distances - distance_mean.unsqueeze(2)).square()).sum(dim=2)
        distance_stats = torch.cat([distance_mean, torch.sqrt(distance_variance.clamp_min(1.0e-8))], dim=-1)
        pooled = pooled + 0.1 * self.distance_context(distance_stats)
        return self.output(pooled)


class _LocalFeatureAggregation(nn.Module):
    def __init__(self, input_dim, output_dim, num_neighbors):
        super().__init__()
        half = int(output_dim) // 2
        self.mlp1 = _SharedMLP(input_dim, half, activation=nn.LeakyReLU(0.2), batch_norm=False)
        self.lse1 = _LocalSpatialEncoding(half, num_neighbors)
        self.pool1 = _AttentivePooling(output_dim, half)
        self.lse2 = _LocalSpatialEncoding(half, num_neighbors)
        self.pool2 = _AttentivePooling(output_dim, output_dim)
        self.mlp2 = _SharedMLP(output_dim, 2 * output_dim, activation=False, batch_norm=False)
        self.shortcut = _SharedMLP(input_dim, 2 * output_dim, activation=False, batch_norm=True)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, coords, features):
        neighbor_indices = _knn_indices(coords, coords, self.lse1.num_neighbors)
        x = self.mlp1(features)
        x, distances = self.lse1(coords, x, neighbor_indices)
        x = self.pool1(x, distances)
        x, distances = self.lse2(coords, x, neighbor_indices)
        x = self.pool2(x, distances)
        return self.activation(self.mlp2(x) + self.shortcut(features))


class RandLANet(nn.Module):
    expects_geo_log_density = False

    def __init__(
        self,
        spatial_dim=3,
        surface_channels=1,
        volume_channels=3,
        parameter_channels=0,
        num_neighbors=16,
        decimation=4,
        query_chunk_size=32768,
        latent_dim=256,
        pos_scale_factor=1.0,
        dropout=0.0,
        query_interpolation_neighbors=2,
        decoder_interpolation_neighbors=2,
        **_unused,
    ):
        super().__init__()
        if int(spatial_dim) != 3:
            raise ValueError("RandLANet currently expects 3D coordinates.")
        self.surface_channels = int(surface_channels)
        self.volume_channels = int(volume_channels)
        self.parameter_channels = int(parameter_channels)
        self.num_neighbors = int(num_neighbors)
        self.decimation = max(2, int(decimation))
        self.query_chunk_size = max(1, int(query_chunk_size))
        self.latent_dim = int(latent_dim)
        self.pos_scale_factor = float(pos_scale_factor)
        self.query_interpolation_neighbors = max(1, int(query_interpolation_neighbors))
        self.decoder_interpolation_neighbors = max(1, int(decoder_interpolation_neighbors))

        self.fc_start = _SharedMLP(3, 8, activation=nn.LeakyReLU(0.2), batch_norm=True)
        encoder_dims = (16, 64, 128, 256)
        input_dims = (8, 32, 128, 256)
        self.encoder = nn.ModuleList(
            [_LocalFeatureAggregation(input_dim, output_dim, self.num_neighbors) for input_dim, output_dim in zip(input_dims, encoder_dims)]
        )
        self.bottleneck = nn.Sequential(
            _SharedMLP(512, 512, activation=nn.ReLU(), batch_norm=False),
            nn.Dropout(float(dropout)),
        )

        # Four encoder blocks are followed by four restorations in the
        # reference RandLA-Net.  The final encoder decimation is important:
        # it creates the N/256 bottleneck before the first decoder block.
        decoder_inputs = (512 + 512, 256 + 256, 128 + 128, 32 + 32)
        decoder_outputs = (256, 128, 32, 8)
        self.decoder = nn.ModuleList(
            [
                _SharedMLP(input_dim, output_dim, activation=nn.ReLU(), batch_norm=True)
                for input_dim, output_dim in zip(decoder_inputs, decoder_outputs)
            ]
        )
        self.geometry_to_latent = nn.Sequential(
            nn.Linear(16, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
        )
        self.geometry_cond = CondInjection(latent_dim, parameter_channels)
        self.query_type = QueryTypeEmbedding(latent_dim)
        self.query_projection = nn.Sequential(
            nn.Linear(8 + 3 + latent_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
        )
        self.query_cond = CondInjection(latent_dim, parameter_channels)
        self.query_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(latent_dim),
                    nn.Linear(latent_dim, 4 * latent_dim),
                    nn.GELU(),
                    nn.Linear(4 * latent_dim, latent_dim),
                )
                for _ in range(3)
            ]
        )
        self.output_head = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, self.surface_channels + self.volume_channels),
        )
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def encode_geometry(self, geo, params=None):
        coords = geo * self.pos_scale_factor
        features = self.fc_start(coords)
        skip_features = []
        skip_coords = []
        # The reference implementation shuffles the input once, then keeps
        # prefixes of that same permutation for each decimation level.
        permutation = torch.randperm(coords.shape[1], device=coords.device)
        coords = coords[:, permutation]
        features = features[:, permutation]
        for level, block in enumerate(self.encoder):
            features = block(coords, features)
            skip_features.append(features)
            skip_coords.append(coords)
            count = max(1, coords.shape[1] // self.decimation)
            coords = coords[:, :count]
            features = features[:, :count]

        features = self.bottleneck(features)
        current_coords = coords
        for decoder_index in range(len(self.decoder)):
            target_level = len(skip_features) - 1 - decoder_index
            target_coords = skip_coords[target_level]
            target_skip = skip_features[target_level]
            interpolation_indices = _knn_indices(
                current_coords,
                target_coords,
                self.decoder_interpolation_neighbors,
            )
            upsampled, interpolation_distances = _weighted_interpolate(
                current_coords,
                features,
                target_coords,
                interpolation_indices,
            )
            upsampled = _spacing_modulate(upsampled, interpolation_distances)
            features = self.decoder[decoder_index](torch.cat([upsampled, target_skip], dim=-1))
            current_coords = target_coords

        pooled = torch.cat([features.amax(dim=1), features.mean(dim=1)], dim=-1)
        latent = self.geometry_to_latent(pooled)
        latent = self.geometry_cond(latent, params)
        return current_coords, features, latent

    def _decode_query_chunk(self, geometry_coords, geometry_features, latent, query, query_type, params=None):
        query_coords = query * self.pos_scale_factor
        query_indices = _knn_indices(
            geometry_coords,
            query_coords,
            self.query_interpolation_neighbors,
        )
        local, query_distances = _weighted_interpolate(
            geometry_coords,
            geometry_features,
            query_coords,
            query_indices,
        )
        local = _spacing_modulate(local, query_distances)
        hidden = self.query_projection(torch.cat([local, query_coords, latent.unsqueeze(1).expand(-1, query.shape[1], -1)], dim=-1))
        hidden = hidden + query_type
        hidden = self.query_cond(hidden, params)
        for block in self.query_blocks:
            hidden = hidden + block(hidden)
        return self.output_head(hidden)

    def _decode_features(self, geometry_coords, geometry_features, latent, surf_query_pos, vol_query_pos, params=None):
        surf_type = self.query_type.surface
        vol_type = self.query_type.volume
        full_query = torch.cat([surf_query_pos, vol_query_pos], dim=1)
        type_tokens = torch.cat(
            [surf_type.expand(surf_query_pos.shape[0], surf_query_pos.shape[1], -1), vol_type.expand(vol_query_pos.shape[0], vol_query_pos.shape[1], -1)],
            dim=1,
        )
        outputs = []
        for start in range(0, full_query.shape[1], self.query_chunk_size):
            stop = min(start + self.query_chunk_size, full_query.shape[1])
            outputs.append(
                self._decode_query_chunk(
                    geometry_coords,
                    geometry_features,
                    latent,
                    full_query[:, start:stop],
                    type_tokens[:, start:stop],
                    params=params,
                )
            )
        pred = (
            torch.cat(outputs, dim=1)
            if outputs
            else full_query.new_empty(
                (full_query.shape[0], 0, self.surface_channels + self.volume_channels)
            )
        )
        return split_surface_volume_predictions(pred, surf_query_pos, self.surface_channels)

    def forward(self, geo, surf_query_pos, vol_query_pos, params=None, geo_log_density=None, return_latent=False):
        geometry_coords, geometry_features, latent = self.encode_geometry(geo, params=params)
        pred_surf, pred_vol = self._decode_features(
            geometry_coords, geometry_features, latent, surf_query_pos, vol_query_pos, params=params
        )
        if return_latent:
            return pred_surf, pred_vol, latent.unsqueeze(1)
        return pred_surf, pred_vol

    @torch.inference_mode()
    def inference(self, geo, surf_query_pos, vol_query_pos, params=None, geo_log_density=None):
        geometry_coords, geometry_features, latent = self.encode_geometry(geo, params=params)
        return self._decode_features(
            geometry_coords, geometry_features, latent, surf_query_pos, vol_query_pos, params=params
        )


class RandLANetWithLatent(RandLANet):
    pass
