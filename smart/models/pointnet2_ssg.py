"""PointNet++ single-scale grouping adapter for SMART's operator interface.

The encoder follows the PointNet++ SSG recipe: farthest-point sampled
centroids, one metric neighbourhood per abstraction level, shared point MLPs,
and max aggregation.  The final global feature is decoded at arbitrary
surface and volume query locations, which is the only adapter-specific part
needed for DrivAerML field prediction.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .family_common import CondInjection, QueryTypeEmbedding, split_surface_volume_predictions
from .family_common import sample_tokens_fps

try:
    from torch_cluster import knn as torch_cluster_knn
except ImportError:  # pragma: no cover - the project environment installs torch-cluster
    torch_cluster_knn = None


def _batched_knn_indices(points: torch.Tensor, centers: torch.Tensor, k: int) -> torch.Tensor:
    """Return source indices with shape ``[B, M, K]`` using torch_cluster."""
    batch_size, num_points, _ = points.shape
    num_centers = centers.shape[1]
    k = min(max(1, int(k)), num_points)
    if torch_cluster_knn is not None:
        # Flattening with batch labels keeps the KNN operation in one
        # extension call instead of launching one call per sample.
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

    # Small-test fallback.  Production runs use torch_cluster above.
    distances = torch.cdist(centers.float(), points.float()).square()
    return torch.topk(distances, k=k, dim=-1, largest=False).indices


class _SharedPointMLP(nn.Module):
    def __init__(self, input_dim: int, widths: tuple[int, ...]):
        super().__init__()
        layers = []
        current = int(input_dim)
        for width in widths:
            width = int(width)
            layers.extend([
                nn.Conv2d(current, width, kernel_size=1),
                nn.BatchNorm2d(width),
                nn.ReLU(),
            ])
            current = width
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        if x.ndim == 3:
            channel_first = x.transpose(1, 2).unsqueeze(-1)
            output = self.layers(channel_first)
            return output.squeeze(-1).transpose(1, 2)
        if x.ndim == 4:
            return self.layers(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        raise ValueError(f"PointNet++ shared MLP expects rank-3 or rank-4 input, got {x.ndim}")


class SetAbstractionSSG(nn.Module):
    """One PointNet++ single-scale set-abstraction layer."""

    def __init__(self, input_channels, output_channels, npoint, radius, nsample):
        super().__init__()
        self.npoint = int(npoint)
        self.radius = float(radius)
        self.nsample = int(nsample)
        self.mlp = _SharedPointMLP(int(input_channels) + 3, tuple(output_channels))

    def forward(self, xyz, features):
        # FPS is the defining SSG centroid sampler.  It is accelerated by
        # torch_cluster and remains deterministic under the caller's seed.
        centers, center_idx = sample_tokens_fps(xyz.float(), self.npoint, random_start=True)
        center_idx = center_idx.to(dtype=torch.long)
        neighbor_idx = _batched_knn_indices(xyz, centers, self.nsample)
        grouped_xyz = torch.gather(
            xyz.unsqueeze(1).expand(-1, centers.shape[1], -1, -1),
            2,
            neighbor_idx.unsqueeze(-1).expand(-1, -1, -1, xyz.shape[-1]),
        )
        relative_xyz = grouped_xyz - centers.unsqueeze(2)
        distance2 = relative_xyz.float().square().sum(dim=-1, keepdim=True)
        valid = distance2 <= self.radius * self.radius
        # PointNet++ keeps a fixed-size group by replacing out-of-radius
        # neighbors with the closest valid neighbor.  KNN always includes
        # the sampled center itself, so every group has at least one valid
        # point even when the configured radius is small.
        nearest_rel = distance2.argmin(dim=2, keepdim=True)
        nearest_xyz = grouped_xyz.gather(
            2,
            nearest_rel.expand(-1, -1, -1, grouped_xyz.shape[-1]),
        )
        grouped_xyz = torch.where(valid, grouped_xyz, nearest_xyz)
        relative_xyz = grouped_xyz - centers.unsqueeze(2)

        grouped_features = torch.gather(
            features.unsqueeze(1).expand(-1, centers.shape[1], -1, -1),
            2,
            neighbor_idx.unsqueeze(-1).expand(-1, -1, -1, features.shape[-1]),
        )
        nearest_features = grouped_features.gather(
            2,
            nearest_rel.expand(-1, -1, -1, grouped_features.shape[-1]),
        )
        grouped_features = torch.where(valid, grouped_features, nearest_features)
        grouped = torch.cat([relative_xyz, grouped_features], dim=-1)
        encoded = self.mlp(grouped)
        next_features = encoded.amax(dim=2)
        return centers, next_features, center_idx


class PointNet2SSG(nn.Module):
    expects_geo_log_density = False

    def __init__(
        self,
        spatial_dim=3,
        surface_channels=1,
        volume_channels=3,
        parameter_channels=0,
        latent_dim=256,
        sa1_npoint=1024,
        sa1_radius=0.05,
        sa1_nsample=32,
        sa2_npoint=256,
        sa2_radius=0.12,
        sa2_nsample=64,
        pos_scale_factor=1.0,
        query_chunk_size=65536,
        dropout=0.0,
        **_unused,
    ):
        super().__init__()
        if int(spatial_dim) != 3:
            raise ValueError("PointNet2SSG currently expects 3D coordinates.")
        self.surface_channels = int(surface_channels)
        self.volume_channels = int(volume_channels)
        self.parameter_channels = int(parameter_channels)
        self.latent_dim = int(latent_dim)
        self.pos_scale_factor = float(pos_scale_factor)
        self.query_chunk_size = max(1, int(query_chunk_size))

        self.sa1 = SetAbstractionSSG(0, (64, 64, 128), sa1_npoint, sa1_radius, sa1_nsample)
        self.sa2 = SetAbstractionSSG(128, (128, 128, 256), sa2_npoint, sa2_radius, sa2_nsample)
        self.global_mlp = _SharedPointMLP(259, (256, 512, latent_dim))
        self.geometry_cond = CondInjection(latent_dim, parameter_channels)

        self.query_embed = nn.Sequential(
            nn.Linear(3, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
        )
        self.query_type = QueryTypeEmbedding(latent_dim)
        self.global_to_query = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
        )
        self.query_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(latent_dim),
                    nn.Linear(latent_dim, 4 * latent_dim),
                    nn.GELU(),
                    nn.Dropout(float(dropout)),
                    nn.Linear(4 * latent_dim, latent_dim),
                )
                for _ in range(4)
            ]
        )
        self.query_cond = CondInjection(latent_dim, parameter_channels)
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
        # PointNet++ radii and query coordinates stay in the dataset's
        # normalized coordinate system, as in the reference implementation.
        xyz = geo
        features = xyz.new_zeros((*xyz.shape[:-1], 0))
        xyz1, feat1, _ = self.sa1(xyz, features)
        xyz2, feat2, _ = self.sa2(xyz1, feat1)
        global_group = torch.cat([xyz2, feat2], dim=-1)
        global_feat = self.global_mlp(global_group).amax(dim=1)
        global_feat = self.geometry_cond(global_feat, params)
        return global_feat

    def decode_features(self, global_feat, surf_query_pos, vol_query_pos, params=None):
        surf_tokens = self.query_embed(surf_query_pos * self.pos_scale_factor)
        vol_tokens = self.query_embed(vol_query_pos * self.pos_scale_factor)
        surf_tokens, vol_tokens = self.query_type(surf_tokens, vol_tokens)
        tokens = torch.cat([surf_tokens, vol_tokens], dim=1)
        tokens = tokens + self.global_to_query(global_feat).unsqueeze(1)
        tokens = self.query_cond(tokens, params)
        for block in self.query_blocks:
            tokens = tokens + block(tokens)
        pred = self.output_head(tokens)
        return split_surface_volume_predictions(pred, surf_query_pos, self.surface_channels)

    def forward(self, geo, surf_query_pos, vol_query_pos, params=None, geo_log_density=None, return_latent=False):
        global_feat = self.encode_geometry(geo, params=params)
        pred_surf, pred_vol = self.decode_features(global_feat, surf_query_pos, vol_query_pos, params=params)
        if return_latent:
            return pred_surf, pred_vol, global_feat.unsqueeze(1)
        return pred_surf, pred_vol

    @torch.inference_mode()
    def inference(self, geo, surf_query_pos, vol_query_pos, params=None, geo_log_density=None):
        global_feat = self.encode_geometry(geo, params=params)
        query_count = int(surf_query_pos.shape[1] + vol_query_pos.shape[1])
        if query_count <= self.query_chunk_size:
            return self.decode_features(global_feat, surf_query_pos, vol_query_pos, params=params)

        surf_count = int(surf_query_pos.shape[1])
        surf_chunks = []
        vol_chunks = []
        full_query = torch.cat([surf_query_pos, vol_query_pos], dim=1)
        for start in range(0, query_count, self.query_chunk_size):
            stop = min(start + self.query_chunk_size, query_count)
            query = full_query[:, start:stop]
            surf_len = max(0, min(stop, surf_count) - start)
            surf_chunk = query[:, :surf_len]
            vol_chunk = query[:, surf_len:]
            if surf_chunk.shape[1] > 0:
                surf_pred, _ = self.decode_features(global_feat, surf_chunk, surf_chunk[:, :0], params=params)
                surf_chunks.append(surf_pred)
            if vol_chunk.shape[1] > 0:
                _, vol_pred = self.decode_features(global_feat, vol_chunk[:, :0], vol_chunk, params=params)
                vol_chunks.append(vol_pred)
        surf_pred = (
            torch.cat(surf_chunks, dim=1)
            if surf_chunks
            else surf_query_pos.new_empty((surf_query_pos.shape[0], 0, self.surface_channels))
        )
        vol_pred = (
            torch.cat(vol_chunks, dim=1)
            if vol_chunks
            else vol_query_pos.new_empty((vol_query_pos.shape[0], 0, self.volume_channels))
        )
        return surf_pred, vol_pred


class PointNet2SSGWithLatent(PointNet2SSG):
    pass
