"""PointNet++ single-scale grouping adapter for SMART's operator interface."""

from __future__ import annotations

import torch
import torch.nn as nn

from .family_common import CondInjection, QueryTypeEmbedding, split_surface_volume_predictions
from .family_common import sample_tokens, sample_tokens_fps

try:
    from torch_cluster import knn as torch_cluster_knn
    from torch_cluster import radius as torch_cluster_radius
except ImportError:  # pragma: no cover - the project environment installs torch-cluster
    torch_cluster_knn = None
    torch_cluster_radius = None


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


def _gather_batched(points: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Gather ``[B, N, C]`` points using ``[B, ...]`` indices safely.

    Flattened ``index_select`` avoids the huge expanded autograd views created
    by gathering from ``points.unsqueeze(1).expand(...)``.
    """
    batch_size, num_points, channels = points.shape
    offsets = torch.arange(batch_size, device=points.device, dtype=torch.long).view(
        batch_size, *([1] * (indices.ndim - 1))
    ) * num_points
    flat_indices = (indices.to(dtype=torch.long) + offsets).reshape(-1)
    return points.reshape(batch_size * num_points, channels).index_select(0, flat_indices).reshape(
        *indices.shape, channels
    )


def _batched_radius_indices(
    points: torch.Tensor,
    centers: torch.Tensor,
    center_indices: torch.Tensor,
    radius: float,
    max_neighbors: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return true-ball-query indices and validity with shape ``[B, M, K]``.

    PointNet++ ball query keeps only points inside the configured radius and
    pads short groups with the center.  The previous implementation performed
    KNN first and then replaced all rejected points, which made the radius
    effectively meaningless for these normalized DrivAerML coordinates.
    """
    batch_size, num_points, _ = points.shape
    num_centers = centers.shape[1]
    max_neighbors = max(1, min(int(max_neighbors), num_points))

    if torch_cluster_radius is None:
        distances = torch.cdist(centers.float(), points.float()).square()
        distances = distances.masked_fill(distances > float(radius) ** 2, float("inf"))
        nearest = distances.argmin(dim=-1, keepdim=True)
        distances = distances.scatter(2, nearest, distances.gather(2, nearest))
        # Sort the in-radius points first; padded entries are replaced by the
        # center index below.  This fallback is intended for small tests only.
        sorted_dist, sorted_idx = torch.sort(distances, dim=-1)
        sorted_idx = sorted_idx[..., :max_neighbors]
        valid = torch.isfinite(sorted_dist[..., :max_neighbors])
        center_idx = center_indices.unsqueeze(-1).expand(-1, -1, max_neighbors)
        return torch.where(valid, sorted_idx, center_idx).to(dtype=torch.long), valid

    flat_points = points.reshape(batch_size * num_points, -1).float().contiguous()
    flat_centers = centers.reshape(batch_size * num_centers, -1).float().contiguous()
    batch_points = torch.arange(batch_size, device=points.device).repeat_interleave(num_points)
    batch_centers = torch.arange(batch_size, device=points.device).repeat_interleave(num_centers)
    edges = torch_cluster_radius(
        flat_points,
        flat_centers,
        r=float(radius),
        batch_x=batch_points,
        batch_y=batch_centers,
        max_num_neighbors=max_neighbors,
    )

    # torch_cluster returns [query/center index, source-point index].
    query_index, source_index = edges[0], edges[1]
    order = torch.argsort(query_index, stable=True)
    query_index = query_index[order]
    source_index = source_index[order]
    counts = torch.bincount(query_index, minlength=batch_size * num_centers)
    rank = torch.arange(query_index.numel(), device=points.device) - torch.repeat_interleave(
        torch.cumsum(counts, dim=0) - counts,
        counts,
    )
    keep = rank < max_neighbors

    center_global = (
        torch.arange(batch_size, device=points.device).unsqueeze(1) * num_points
        + center_indices.to(device=points.device, dtype=torch.long)
    ).reshape(-1)
    grouped = center_global.unsqueeze(1).expand(-1, max_neighbors).clone()
    valid = torch.zeros(
        (batch_size * num_centers, max_neighbors),
        device=points.device,
        dtype=torch.bool,
    )
    grouped[query_index[keep], rank[keep]] = source_index[keep]
    valid[query_index[keep], rank[keep]] = True
    grouped = (grouped.reshape(batch_size, num_centers, max_neighbors) % num_points).to(dtype=torch.long)
    valid = valid.reshape(batch_size, num_centers, max_neighbors)
    return grouped, valid


def _batched_inverse_distance_interpolation(
    source_xyz: torch.Tensor,
    source_features: torch.Tensor,
    query_xyz: torch.Tensor,
    k: int,
) -> torch.Tensor:
    """PointNet++ feature-propagation interpolation for arbitrary queries."""
    neighbor_idx = _batched_knn_indices(source_xyz, query_xyz, k)
    grouped_xyz = _gather_batched(source_xyz, neighbor_idx)
    grouped_features = _gather_batched(source_features, neighbor_idx)
    distance2 = (grouped_xyz - query_xyz.unsqueeze(2)).float().square().sum(dim=-1)
    weights = torch.reciprocal(distance2.clamp_min(1.0e-8))
    weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)
    return (grouped_features * weights.to(dtype=grouped_features.dtype).unsqueeze(-1)).sum(dim=2)


class _SharedPointMLP(nn.Module):
    def __init__(self, input_dim: int, widths: tuple[int, ...]):
        super().__init__()
        layers = []
        current = int(input_dim)
        for width in widths:
            width = int(width)
            layers.extend([
                nn.Linear(current, width),
                # Group/set-local normalization avoids running-statistic drift
                # when SATLOSS6 processes differently sampled views together.
                nn.LayerNorm(width),
                nn.ReLU(),
            ])
            current = width
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        if x.ndim not in (3, 4):
            raise ValueError(f"PointNet++ shared MLP expects rank-3 or rank-4 input, got {x.ndim}")
        return self.layers(x)


class SetAbstractionSSG(nn.Module):
    """One PointNet++ single-scale set-abstraction layer."""

    def __init__(
        self,
        input_channels,
        output_channels,
        npoint,
        radius,
        nsample,
        centroid_sampling="fps",
        deterministic_eval_centroids=False,
        eval_centroid_seed=0,
    ):
        super().__init__()
        self.npoint = int(npoint)
        self.radius = float(radius)
        self.nsample = int(nsample)
        self.centroid_sampling = str(centroid_sampling).lower()
        self.deterministic_eval_centroids = bool(deterministic_eval_centroids)
        self.eval_centroid_seed = int(eval_centroid_seed)
        if self.centroid_sampling not in {"fps", "random"}:
            raise ValueError(f"Unsupported PointNet++ centroid sampler: {centroid_sampling!r}")
        output_channels = tuple(output_channels)
        self.output_channels = int(output_channels[-1])
        # Support occupancy is intentionally part of the local descriptor:
        # fixed-radius remeshing changes how many points support a center.
        self.mlp = _SharedPointMLP(int(input_channels) + 5, output_channels)
        attention_width = max(16, self.output_channels // 4)
        self.pool_logits = nn.Sequential(
            nn.Linear(self.output_channels, attention_width),
            nn.GELU(),
            nn.Linear(attention_width, 1),
        )

    def forward(self, xyz, features):
        # Random centroids preserve the input-density shift for this
        # sensitivity benchmark; ``fps`` remains available for canonical
        # PointNet++ behavior.
        if self.centroid_sampling == "random":
            if self.deterministic_eval_centroids and not self.training:
                # Keep random-centroid density sensitivity during training,
                # but make every evaluation reproducible.  Otherwise a base
                # and SATLOSS model receive different hidden centroid draws
                # for the same controlled encoder cloud.
                indices = []
                for batch_index in range(xyz.shape[0]):
                    generator = torch.Generator(device=xyz.device)
                    generator.manual_seed(self.eval_centroid_seed + batch_index)
                    indices.append(
                        torch.randperm(xyz.shape[1], device=xyz.device, generator=generator)[: self.npoint]
                    )
                center_idx = torch.stack(indices, dim=0)
                centers = _gather_batched(xyz, center_idx)
            else:
                centers, center_idx = sample_tokens(xyz, self.npoint)
        else:
            centers, center_idx = sample_tokens_fps(xyz.float(), self.npoint, random_start=True)
        center_idx = center_idx.to(dtype=torch.long)
        neighbor_idx, neighbor_valid = _batched_radius_indices(
            xyz,
            centers,
            center_idx,
            radius=self.radius,
            max_neighbors=self.nsample,
        )
        grouped_xyz = _gather_batched(xyz, neighbor_idx)
        relative_xyz = grouped_xyz - centers.unsqueeze(2)
        grouped_features = _gather_batched(features, neighbor_idx)
        valid_float = neighbor_valid.to(dtype=xyz.dtype).unsqueeze(-1)
        support_count = neighbor_valid.sum(dim=2, keepdim=True).to(dtype=xyz.dtype)
        support_fraction = support_count / float(max(self.nsample, 1))
        support_fraction = support_fraction.expand(-1, -1, self.nsample).unsqueeze(-1)
        grouped = torch.cat([relative_xyz, grouped_features, valid_float, support_fraction], dim=-1)
        encoded = self.mlp(grouped)
        logits = self.pool_logits(encoded).squeeze(-1)
        logits = logits.masked_fill(~neighbor_valid, torch.finfo(logits.dtype).min)
        weights = torch.softmax(logits.float(), dim=2).to(dtype=encoded.dtype).unsqueeze(-1)
        next_features = (encoded * weights).sum(dim=2)
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
        sa1_npoint=2048,
        sa1_radius=0.02,
        sa1_nsample=64,
        sa2_npoint=512,
        sa2_radius=0.05,
        sa2_nsample=128,
        query_neighbors=3,
        centroid_sampling="random",
        deterministic_eval_centroids=False,
        density_histogram_bins=8,
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
        self.query_neighbors = max(1, int(query_neighbors))
        self.density_histogram_bins = max(2, int(density_histogram_bins))

        self.centroid_sampling = str(centroid_sampling).lower()
        self.sa1 = SetAbstractionSSG(
            0, (64, 64, 128), sa1_npoint, sa1_radius, sa1_nsample,
            self.centroid_sampling, deterministic_eval_centroids, eval_centroid_seed=104729,
        )
        self.sa2 = SetAbstractionSSG(
            128, (128, 128, 256), sa2_npoint, sa2_radius, sa2_nsample,
            self.centroid_sampling, deterministic_eval_centroids, eval_centroid_seed=130363,
        )
        self.global_mlp = _SharedPointMLP(259, (256, 512, latent_dim))
        self.global_pool_logits = nn.Sequential(
            nn.Linear(259, max(32, latent_dim // 4)),
            nn.GELU(),
            nn.Linear(max(32, latent_dim // 4), 1),
        )
        density_width = max(32, latent_dim // 4)
        self.density_encoder = nn.Sequential(
            nn.Linear(3 * self.density_histogram_bins, density_width),
            nn.LayerNorm(density_width),
            nn.GELU(),
            nn.Linear(density_width, latent_dim),
        )
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
        self.local_sa1_to_query = nn.Sequential(
            nn.LayerNorm(128),
            nn.Linear(128, latent_dim),
            nn.GELU(),
        )
        self.local_sa2_to_query = nn.Sequential(
            nn.LayerNorm(256),
            nn.Linear(256, latent_dim),
            nn.GELU(),
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
        global_tokens = self.global_mlp(global_group)
        global_logits = self.global_pool_logits(global_group).squeeze(-1)
        global_weights = torch.softmax(global_logits.float(), dim=1).to(dtype=global_tokens.dtype).unsqueeze(-1)
        global_feat = (global_tokens * global_weights).sum(dim=1)
        # Keep a direct record of where the sampled points lie.  The fixed
        # normalized-coordinate histogram is distribution-sensitive while
        # avoiding a max/mean collapse of the input cloud.  The input budget
        # is fixed for each experiment, so normalizing counts by N changes
        # neither the relative occupancy pattern nor the benchmark signal.
        coordinates = xyz.float().clamp(0.0, 1.0)
        bins = self.density_histogram_bins
        bin_indices = (coordinates * float(bins)).to(dtype=torch.long).clamp_(0, bins - 1)
        axis_offsets = torch.arange(3, device=xyz.device, dtype=torch.long).view(1, 1, 3) * bins
        flat_indices = (bin_indices + axis_offsets).reshape(xyz.shape[0], -1)
        histogram = torch.zeros(
            (xyz.shape[0], 3 * bins),
            device=xyz.device,
            dtype=torch.float32,
        )
        histogram.scatter_add_(1, flat_indices, torch.ones_like(flat_indices, dtype=histogram.dtype))
        histogram = histogram / float(max(xyz.shape[1], 1))
        global_feat = global_feat + self.density_encoder(histogram.to(dtype=global_feat.dtype))
        global_feat = self.geometry_cond(global_feat, params)
        return global_feat, xyz1, feat1, xyz2, feat2

    def decode_features(self, encoded, surf_query_pos, vol_query_pos, params=None):
        global_feat, xyz1, feat1, xyz2, feat2 = encoded
        surf_count = int(surf_query_pos.shape[1])
        full_query = torch.cat([surf_query_pos, vol_query_pos], dim=1)
        query_chunks = []
        global_token = self.global_to_query(global_feat).unsqueeze(1)
        for start in range(0, full_query.shape[1], self.query_chunk_size):
            stop = min(start + self.query_chunk_size, full_query.shape[1])
            query = full_query[:, start:stop]
            local1 = _batched_inverse_distance_interpolation(xyz1, feat1, query, self.query_neighbors)
            local2 = _batched_inverse_distance_interpolation(xyz2, feat2, query, self.query_neighbors)
            tokens = self.query_embed(query * self.pos_scale_factor)
            tokens = tokens + global_token
            tokens = tokens + self.local_sa1_to_query(local1) + self.local_sa2_to_query(local2)

            surface_end = max(0, min(stop, surf_count) - start)
            type_embedding = torch.cat(
                [
                    self.query_type.surface.expand(query.shape[0], surface_end, -1),
                    self.query_type.volume.expand(query.shape[0], query.shape[1] - surface_end, -1),
                ],
                dim=1,
            )
            tokens = tokens + type_embedding
            tokens = self.query_cond(tokens, params)
            for block in self.query_blocks:
                tokens = tokens + block(tokens)
            query_chunks.append(self.output_head(tokens))

        pred = torch.cat(query_chunks, dim=1)
        return split_surface_volume_predictions(pred, surf_query_pos, self.surface_channels)

    def forward(self, geo, surf_query_pos, vol_query_pos, params=None, geo_log_density=None, return_latent=False):
        encoded = self.encode_geometry(geo, params=params)
        pred_surf, pred_vol = self.decode_features(encoded, surf_query_pos, vol_query_pos, params=params)
        if return_latent:
            return pred_surf, pred_vol, encoded[0].unsqueeze(1)
        return pred_surf, pred_vol

    @torch.inference_mode()
    def inference(self, geo, surf_query_pos, vol_query_pos, params=None, geo_log_density=None):
        encoded = self.encode_geometry(geo, params=params)
        query_count = int(surf_query_pos.shape[1] + vol_query_pos.shape[1])
        if query_count <= self.query_chunk_size:
            return self.decode_features(encoded, surf_query_pos, vol_query_pos, params=params)

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
            surf_pred, vol_pred = self.decode_features(encoded, surf_chunk, vol_chunk, params=params)
            if surf_chunk.shape[1] > 0:
                surf_chunks.append(surf_pred)
            if vol_chunk.shape[1] > 0:
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
