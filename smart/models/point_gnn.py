"""Point-GNN-style mesh-free operator for DrivAerML.

The graph encoder keeps the local-graph, relative-coordinate, residual-update,
and auto-registration structure of Point-GNN.  This experiment intentionally
uses learned transformed *unnormalized sums* instead of max/mean aggregation so
the encoder retains information about local point count and sampling density.
The
original implementation targets 3D object detection and uses
TensorFlow/Open3D.  This PyTorch adapter uses ``torch_cluster.radius`` and
native scatter-add for the graph path, then decodes arbitrary surface and
volume query locations for operator training.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .family_common import CondInjection, QueryTypeEmbedding, split_surface_volume_predictions

try:
    from torch_cluster import knn as torch_cluster_knn
    from torch_cluster import radius as torch_cluster_radius
except ImportError:  # pragma: no cover - production uses the CUDA extension
    torch_cluster_knn = None
    torch_cluster_radius = None


def _batched_radius_edges(source, centers, source_batch, center_batch, radius, max_neighbors):
    """Return ``(source_index, center_index)`` for a bounded radius graph."""
    if torch_cluster_radius is not None:
        # torch_cluster returns (center/query index, source index) for
        # radius(x=source, y=centers), so swap the two rows here.
        edges = torch_cluster_radius(
            source.float().contiguous(),
            centers.float().contiguous(),
            r=float(radius),
            batch_x=source_batch,
            batch_y=center_batch,
            max_num_neighbors=max(1, int(max_neighbors)),
        )
        return edges[1].to(dtype=torch.long), edges[0].to(dtype=torch.long)

    # Small CPU fallback. It is intentionally chunked and is not the
    # production path; the CUDA torch_cluster implementation is required for
    # full DrivAerML runs.
    source_indices = []
    center_indices = []
    for batch_id in torch.unique(center_batch, sorted=True).tolist():
        source_local = torch.nonzero(source_batch == batch_id, as_tuple=False).squeeze(-1)
        center_local = torch.nonzero(center_batch == batch_id, as_tuple=False).squeeze(-1)
        source_points = source[source_local].float()
        center_points = centers[center_local].float()
        for start in range(0, center_points.shape[0], 1024):
            stop = min(start + 1024, center_points.shape[0])
            distances = torch.cdist(center_points[start:stop], source_points)
            local = torch.where(distances <= float(radius))[1]
            row = torch.where(distances <= float(radius))[0]
            if local.numel() == 0:
                nearest = distances.argmin(dim=1)
                row = torch.arange(stop - start, device=source.device)
                local = nearest
            if int(max_neighbors) > 0:
                keep = []
                for row_id in range(stop - start):
                    row_points = torch.nonzero(row == row_id, as_tuple=False).squeeze(-1)
                    if row_points.numel() > int(max_neighbors):
                        row_points = row_points[: int(max_neighbors)]
                    keep.append(row_points)
                keep = torch.cat(keep) if keep else row.new_empty((0,))
                row = row[keep]
                local = local[keep]
            source_indices.append(source_local[local])
            center_indices.append(center_local[row])
    return torch.cat(source_indices), torch.cat(center_indices)


def _batched_knn_indices(source, centers, source_batch, center_batch, k):
    """Return source indices with shape ``[num_centers, k]`` in packed form."""
    k = min(max(1, int(k)), int(source.shape[0]))
    if torch_cluster_knn is not None:
        edges = torch_cluster_knn(
            source.float().contiguous(),
            centers.float().contiguous(),
            k=k,
            batch_x=source_batch,
            batch_y=center_batch,
        )
        center_index, source_index = edges[0], edges[1]
        order = torch.argsort(center_index, stable=True)
        return source_index[order].view(centers.shape[0], k).to(dtype=torch.long)

    chunks = []
    for batch_id in torch.unique(center_batch, sorted=True).tolist():
        source_local = torch.nonzero(source_batch == batch_id, as_tuple=False).squeeze(-1)
        center_local = torch.nonzero(center_batch == batch_id, as_tuple=False).squeeze(-1)
        distances = torch.cdist(centers[center_local].float(), source[source_local].float()).square()
        local = distances.topk(k=k, dim=-1, largest=False).indices
        chunks.append(source_local[local])
    return torch.cat(chunks, dim=0)


def _scatter_sum(values, indices, size, scale=1.0):
    """Add contributions without normalizing by neighborhood cardinality."""
    output = values.new_zeros((int(size), values.shape[-1]))
    output.index_add_(0, indices, values)
    return output * float(scale)


def _chunked_edge_sum(
    source_features,
    source_coords,
    center_coords,
    source_index,
    center_index,
    edge_mlp,
    output_channels,
    center_count,
    scale,
    chunk_size,
):
    """Transform and add radius edges in bounded chunks."""
    output = source_features.new_zeros((int(center_count), int(output_channels)))
    chunk_size = max(1, int(chunk_size))
    for start in range(0, int(source_index.numel()), chunk_size):
        stop = min(start + chunk_size, int(source_index.numel()))
        source_chunk = source_index[start:stop]
        center_chunk = center_index[start:stop]
        edge_input = torch.cat(
            [
                source_features[source_chunk],
                source_coords[source_chunk] - center_coords[center_chunk],
            ],
            dim=-1,
        )
        output.index_add_(0, center_chunk, edge_mlp(edge_input))
    return output * float(scale)


def _voxel_downsample(coords, features, batch, voxel_size, random_offset=True):
    """Point-GNN voxel downsampling with one randomly selected point/cell."""
    if float(voxel_size) <= 0.0:
        return coords, features, batch
    output_coords = []
    output_features = []
    output_batch = []
    for batch_id in torch.unique(batch, sorted=True).tolist():
        point_indices = torch.nonzero(batch == batch_id, as_tuple=False).squeeze(-1)
        current_coords = coords[point_indices]
        current_features = features[point_indices]
        origin = current_coords.amin(dim=0, keepdim=True)
        offset = (
            torch.rand((1, 3), device=coords.device, dtype=coords.dtype) * float(voxel_size)
            if random_offset
            else 0.0
        )
        voxel = torch.floor((current_coords - origin + offset) / float(voxel_size)).to(dtype=torch.long)
        _, inverse = torch.unique(voxel, dim=0, return_inverse=True, sorted=True)
        point_count = int(current_coords.shape[0])
        random_rank = torch.rand(point_count, device=coords.device, dtype=torch.float32)
        sort_key = inverse.to(dtype=torch.float32) * float(point_count + 1) + random_rank
        order = torch.argsort(sort_key)
        sorted_groups = inverse[order]
        first = torch.ones_like(sorted_groups, dtype=torch.bool)
        if first.numel() > 1:
            first[1:] = sorted_groups[1:] != sorted_groups[:-1]
        selected = order[first]
        output_coords.append(current_coords[selected])
        output_features.append(current_features[selected])
        output_batch.append(torch.full((selected.numel(),), int(batch_id), device=batch.device, dtype=batch.dtype))
    return torch.cat(output_coords), torch.cat(output_features), torch.cat(output_batch)


class _PointMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, activation=True, final_activation=None):
        super().__init__()
        layers = [nn.Linear(int(input_dim), int(hidden_dim))]
        if activation:
            layers.append(nn.ReLU())
        layers.extend([nn.Linear(int(hidden_dim), int(output_dim))])
        if final_activation is None:
            final_activation = activation
        if final_activation:
            layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class PointGNNPointSetPooling(nn.Module):
    """Density-sensitive unnormalized sum pooling to voxel keypoints."""

    def __init__(self, channels, edge_hidden_dim, graph_radius, max_neighbors, sum_scale, edge_chunk_size):
        super().__init__()
        self.graph_radius = float(graph_radius)
        self.max_neighbors = int(max_neighbors)
        self.sum_scale = float(sum_scale)
        self.edge_chunk_size = int(edge_chunk_size)
        self.point_feature_mlp = _PointMLP(channels + 3, edge_hidden_dim, channels, activation=True)
        self.output_mlp = _PointMLP(channels, edge_hidden_dim, channels, activation=True)

    def forward(self, source_coords, source_features, source_batch, center_coords, center_batch):
        source_index, center_index = _batched_radius_edges(
            source_coords,
            center_coords,
            source_batch,
            center_batch,
            self.graph_radius,
            self.max_neighbors,
        )
        pooled = _chunked_edge_sum(
            source_features,
            source_coords,
            center_coords,
            source_index,
            center_index,
            self.point_feature_mlp,
            source_features.shape[-1],
            center_coords.shape[0],
            self.sum_scale,
            self.edge_chunk_size,
        )
        return self.output_mlp(pooled)


class PointGNNGraphBlock(nn.Module):
    """GraphNetAutoCenter block with learned, density-sensitive edge sums."""

    def __init__(
        self,
        channels,
        edge_hidden_dim,
        graph_radius,
        max_neighbors,
        auto_center=True,
        offset_scale=0.05,
        sum_scale=1.0,
        edge_chunk_size=262144,
    ):
        super().__init__()
        self.graph_radius = float(graph_radius)
        self.max_neighbors = int(max_neighbors)
        self.auto_center = bool(auto_center)
        self.offset_scale = float(offset_scale)
        self.sum_scale = float(sum_scale)
        self.edge_chunk_size = int(edge_chunk_size)
        self.edge_mlp = _PointMLP(int(channels) + 3, edge_hidden_dim, channels, activation=True)
        self.update_mlp = _PointMLP(channels, edge_hidden_dim, channels, activation=True, final_activation=False)
        self.auto_offset = (
            _PointMLP(channels, edge_hidden_dim, 3, activation=True, final_activation=False)
            if self.auto_center
            else None
        )

    def forward(self, coords, features, batch):
        graph_coords = coords
        if self.auto_offset:
            offset = torch.tanh(self.auto_offset(features)) * self.offset_scale
            graph_coords = coords + offset
        source_index, center_index = _batched_radius_edges(
            graph_coords,
            graph_coords,
            batch,
            batch,
            self.graph_radius,
            self.max_neighbors,
        )
        aggregated = _chunked_edge_sum(
            features,
            graph_coords,
            graph_coords,
            source_index,
            center_index,
            self.edge_mlp,
            features.shape[-1],
            features.shape[0],
            self.sum_scale,
            self.edge_chunk_size,
        )
        return features + self.update_mlp(aggregated)


class PointGNN(nn.Module):
    """Point-GNN encoder plus arbitrary surface/volume query decoder."""

    expects_geo_log_density = False

    def __init__(
        self,
        spatial_dim=3,
        surface_channels=1,
        volume_channels=3,
        parameter_channels=0,
        latent_dim=256,
        node_dim=64,
        edge_hidden_dim=128,
        graph_radii=(0.035, 0.08, 0.16, 0.24),
        graph_max_neighbors=16,
        pool_voxel_sizes=(0.025, 0.06, 0.12),
        auto_center_layers=(True, True, True, True),
        auto_center_offset_scale=0.05,
        query_neighbors=8,
        query_chunk_size=32768,
        num_query_blocks=3,
        pos_scale_factor=100.0,
        local_sum_scale=None,
        global_sum_scale=0.015625,
        edge_chunk_size=262144,
        dropout=0.0,
        **_unused,
    ):
        super().__init__()
        if int(spatial_dim) != 3:
            raise ValueError("PointGNN currently expects 3D coordinates.")
        self.surface_channels = int(surface_channels)
        self.volume_channels = int(volume_channels)
        self.parameter_channels = int(parameter_channels)
        self.latent_dim = int(latent_dim)
        self.node_dim = int(node_dim)
        self.query_neighbors = max(1, int(query_neighbors))
        self.query_chunk_size = max(1, int(query_chunk_size))
        self.pos_scale_factor = float(pos_scale_factor)
        if local_sum_scale is None:
            local_sum_scale = 1.0 / (max(1, int(graph_max_neighbors)) ** 0.5)
        self.local_sum_scale = float(local_sum_scale)
        self.global_sum_scale = float(global_sum_scale)
        self.edge_chunk_size = max(1, int(edge_chunk_size))
        self.graph_radii = tuple(float(x) for x in graph_radii)
        self.pool_voxel_sizes = tuple(float(x) for x in pool_voxel_sizes)
        if len(self.graph_radii) != len(self.pool_voxel_sizes) + 1:
            raise ValueError("graph_radii must contain one more value than pool_voxel_sizes.")
        auto_center_layers = tuple(bool(x) for x in auto_center_layers)
        if len(auto_center_layers) != len(self.graph_radii):
            raise ValueError("auto_center_layers must match graph_radii length.")

        self.input_projection = _PointMLP(3, node_dim, node_dim, activation=True)
        self.graph_blocks = nn.ModuleList(
            [
                PointGNNGraphBlock(
                    node_dim,
                    edge_hidden_dim,
                    graph_radius,
                    graph_max_neighbors,
                    auto_center=auto_center,
                    offset_scale=auto_center_offset_scale,
                    sum_scale=self.local_sum_scale,
                    edge_chunk_size=self.edge_chunk_size,
                )
                for graph_radius, auto_center in zip(self.graph_radii, auto_center_layers)
            ]
        )
        self.pool_blocks = nn.ModuleList(
            [
                PointGNNPointSetPooling(
                    node_dim,
                    edge_hidden_dim,
                    graph_radii[index + 1],
                    graph_max_neighbors,
                    sum_scale=self.local_sum_scale,
                    edge_chunk_size=self.edge_chunk_size,
                )
                for index in range(len(self.pool_voxel_sizes))
            ]
        )
        self.global_projection = nn.Sequential(
            nn.Linear(node_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
        )
        self.geometry_cond = CondInjection(latent_dim, parameter_channels)
        self.query_type = QueryTypeEmbedding(latent_dim)
        self.query_projection = nn.Sequential(
            nn.Linear(node_dim + 3 + latent_dim + 1, latent_dim),
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
                    nn.Dropout(float(dropout)),
                    nn.Linear(4 * latent_dim, latent_dim),
                )
                for _ in range(int(num_query_blocks))
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
        batch_size, point_count, _ = geo.shape
        coords = geo.reshape(batch_size * point_count, 3)
        batch = torch.arange(batch_size, device=geo.device, dtype=torch.long).repeat_interleave(point_count)
        features = self.input_projection(coords)
        full_coords = coords
        full_features = None
        for block_index, block in enumerate(self.graph_blocks):
            features = block(coords, features, batch)
            if block_index == 0:
                full_coords = coords
                full_features = features
            if block_index < len(self.pool_voxel_sizes):
                next_coords, _, next_batch = _voxel_downsample(
                    coords,
                    features,
                    batch,
                    self.pool_voxel_sizes[block_index],
                    random_offset=True,
                )
                features = self.pool_blocks[block_index](
                    coords,
                    features,
                    batch,
                    next_coords,
                    next_batch,
                )
                coords, batch = next_coords, next_batch
        global_sum = _scatter_sum(
            features,
            batch,
            batch_size,
            scale=self.global_sum_scale,
        )
        latent = self.global_projection(global_sum)
        latent = self.geometry_cond(latent, params)
        return full_coords, full_features, latent

    def _decode_query_chunk(self, geometry_coords, geometry_features, geometry_batch, latent, query, query_type, params=None):
        query_batch = torch.arange(query.shape[0], device=query.device, dtype=torch.long).repeat_interleave(query.shape[1])
        query_flat = query.reshape(-1, 3)
        source_index = _batched_knn_indices(
            geometry_coords,
            query_flat,
            geometry_batch,
            query_batch,
            self.query_neighbors,
        )
        neighbor_features = geometry_features[source_index]
        neighbor_coords = geometry_coords[source_index]
        query_expanded = query_flat.unsqueeze(1)
        distances = torch.linalg.vector_norm(neighbor_coords - query_expanded, dim=-1)
        weights = torch.reciprocal(distances.float().clamp_min(1.0e-5))
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1.0e-6)
        local = (neighbor_features * weights.to(dtype=neighbor_features.dtype).unsqueeze(-1)).sum(dim=1)
        effective_distance = (weights * distances.float()).sum(dim=-1, keepdim=True).to(dtype=local.dtype)
        latent_flat = latent.unsqueeze(1).expand(-1, query.shape[1], -1).reshape(-1, self.latent_dim)
        hidden = self.query_projection(
            torch.cat(
                [
                    local,
                    query_flat * self.pos_scale_factor,
                    latent_flat,
                    effective_distance,
                ],
                dim=-1,
            )
        )
        hidden = hidden.reshape(query.shape[0], query.shape[1], self.latent_dim) + query_type
        hidden = self.query_cond(hidden, params)
        for block in self.query_blocks:
            hidden = hidden + block(hidden)
        return self.output_head(hidden)

    def _decode_features(self, geometry_coords, geometry_features, geometry_batch, latent, surf_query_pos, vol_query_pos, params=None):
        full_query = torch.cat([surf_query_pos, vol_query_pos], dim=1)
        surface_type = self.query_type.surface.expand(surf_query_pos.shape[0], surf_query_pos.shape[1], -1)
        volume_type = self.query_type.volume.expand(vol_query_pos.shape[0], vol_query_pos.shape[1], -1)
        query_type = torch.cat([surface_type, volume_type], dim=1)
        outputs = []
        for start in range(0, full_query.shape[1], self.query_chunk_size):
            stop = min(start + self.query_chunk_size, full_query.shape[1])
            outputs.append(
                self._decode_query_chunk(
                    geometry_coords,
                    geometry_features,
                    geometry_batch,
                    latent,
                    full_query[:, start:stop],
                    query_type[:, start:stop],
                    params=params,
                )
            )
        pred = torch.cat(outputs, dim=1) if outputs else full_query.new_empty((full_query.shape[0], 0, self.surface_channels + self.volume_channels))
        return split_surface_volume_predictions(pred, surf_query_pos, self.surface_channels)

    def forward(self, geo, surf_query_pos, vol_query_pos, params=None, geo_log_density=None, return_latent=False):
        geometry_coords, geometry_features, latent = self.encode_geometry(geo, params=params)
        geometry_batch = torch.arange(geo.shape[0], device=geo.device, dtype=torch.long).repeat_interleave(geo.shape[1])
        pred_surf, pred_vol = self._decode_features(
            geometry_coords,
            geometry_features,
            geometry_batch,
            latent,
            surf_query_pos,
            vol_query_pos,
            params=params,
        )
        if return_latent:
            return pred_surf, pred_vol, latent.unsqueeze(1)
        return pred_surf, pred_vol

    @torch.inference_mode()
    def inference(self, geo, surf_query_pos, vol_query_pos, params=None, geo_log_density=None):
        geometry_coords, geometry_features, latent = self.encode_geometry(geo, params=params)
        geometry_batch = torch.arange(geo.shape[0], device=geo.device, dtype=torch.long).repeat_interleave(geo.shape[1])
        return self._decode_features(
            geometry_coords,
            geometry_features,
            geometry_batch,
            latent,
            surf_query_pos,
            vol_query_pos,
            params=params,
        )


class PointGNNWithLatent(PointGNN):
    pass
