"""Memory-bounded geometric primitives shared by operator adapters."""

from __future__ import annotations

import torch

try:
    from torch_cluster import knn as torch_cluster_knn
except ImportError:  # pragma: no cover - optional CUDA acceleration
    torch_cluster_knn = None


def sample_point_indices(points: torch.Tensor, count: int, training: bool) -> torch.Tensor:
    """Sample a fixed number of indices without constructing an NxN matrix."""
    batch_size, num_points, _ = points.shape
    count = min(max(1, int(count)), int(num_points))
    if count == num_points:
        return torch.arange(num_points, device=points.device).unsqueeze(0).expand(batch_size, -1)
    if training:
        return torch.stack(
            [torch.randperm(num_points, device=points.device)[:count] for _ in range(batch_size)],
            dim=0,
        )
    # A deterministic, coverage-preserving subset makes repeated validation exact.
    base = torch.linspace(0, num_points - 1, count, device=points.device).round().long()
    return base.unsqueeze(0).expand(batch_size, -1)


def gather_points(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Gather [B, M] indices from [B, N, C] values."""
    batch_size, num_points, channels = values.shape
    offsets = torch.arange(batch_size, device=values.device).view(-1, 1) * num_points
    flat_indices = (indices + offsets).reshape(-1)
    return values.reshape(batch_size * num_points, channels)[flat_indices].reshape(
        batch_size, indices.shape[1], channels
    )


@torch.no_grad()
def chunked_knn_indices(
    source_positions: torch.Tensor,
    query_positions: torch.Tensor,
    k: int,
    query_chunk_size: int = 256,
) -> torch.Tensor:
    """Return exact KNN indices while bounding the temporary distance matrix.

    Positions are data, not trainable parameters, so graph construction is kept
    outside autograd. Computation is performed in float32 for stable ordering.
    """
    if source_positions.ndim != 3 or query_positions.ndim != 3:
        raise ValueError("source_positions and query_positions must have shape [B, N, C].")
    if source_positions.shape[0] != query_positions.shape[0]:
        raise ValueError("source_positions and query_positions must share a batch dimension.")
    k = min(max(1, int(k)), int(source_positions.shape[1]))
    query_chunk_size = max(1, int(query_chunk_size))
    if torch_cluster_knn is not None and source_positions.is_cuda:
        batch_size, num_source, channels = source_positions.shape
        num_query = int(query_positions.shape[1])
        flat_source = source_positions.reshape(batch_size * num_source, channels).float().contiguous()
        flat_query = query_positions.reshape(batch_size * num_query, channels).float().contiguous()
        source_batch = torch.arange(batch_size, device=source_positions.device).repeat_interleave(num_source)
        query_batch = torch.arange(batch_size, device=query_positions.device).repeat_interleave(num_query)
        edge_index = torch_cluster_knn(
            flat_source,
            flat_query,
            k=k,
            batch_x=source_batch,
            batch_y=query_batch,
        )
        query_index, source_index = edge_index[0], edge_index[1]
        order = torch.argsort(query_index, stable=True)
        return (source_index[order] % num_source).reshape(batch_size, num_query, k).long()

    per_batch = []
    for batch_index in range(source_positions.shape[0]):
        source = source_positions[batch_index].float()
        query = query_positions[batch_index].float()
        chunks = []
        for start in range(0, query.shape[0], query_chunk_size):
            distance = torch.cdist(query[start : start + query_chunk_size], source)
            chunks.append(torch.topk(distance, k=k, dim=-1, largest=False, sorted=True).indices)
        per_batch.append(torch.cat(chunks, dim=0))
    return torch.stack(per_batch, dim=0)


def gather_neighbors(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Gather [B, M, K] indices from [B, N, C] values."""
    batch_size, num_points, channels = values.shape
    offsets = torch.arange(batch_size, device=values.device).view(-1, 1, 1) * num_points
    flat_indices = (indices + offsets).reshape(-1)
    return values.reshape(batch_size * num_points, channels)[flat_indices].reshape(
        batch_size, indices.shape[1], indices.shape[2], channels
    )
