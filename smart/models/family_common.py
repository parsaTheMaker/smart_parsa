from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

try:
    from torch_cluster import fps as torch_cluster_fps
    from torch_cluster import knn as torch_cluster_knn
except ImportError:  # pragma: no cover - optional acceleration backend
    torch_cluster_fps = None
    torch_cluster_knn = None

from .smart.smart import (
    CrossAttention,
    PlainMLP,
    SimulationParamModulatedMLP,
)


def gather_tokens(tokens, idx):
    return torch.gather(tokens, 1, idx.unsqueeze(-1).expand(-1, -1, tokens.shape[-1]))


def sample_indices(n_points, num_samples, device, batch_size):
    if num_samples <= 0 or num_samples >= n_points:
        return torch.arange(n_points, device=device, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)
    return torch.stack(
        [torch.randperm(n_points, device=device)[:num_samples] for _ in range(batch_size)],
        dim=0,
    )


def _fps_indices_single(points, num_samples, random_start=False):
    n_points = int(points.shape[0])
    if num_samples <= 0 or num_samples >= n_points:
        return torch.arange(n_points, device=points.device, dtype=torch.long)

    if torch_cluster_fps is not None:
        ratio = float(num_samples) / float(n_points)
        idx = torch_cluster_fps(points.float().contiguous(), ratio=ratio, random_start=random_start)
        if idx.numel() < num_samples:
            remaining_mask = torch.ones(n_points, device=points.device, dtype=torch.bool)
            remaining_mask[idx] = False
            remaining = torch.nonzero(remaining_mask, as_tuple=False).squeeze(-1)
            extra = remaining[: num_samples - idx.numel()]
            idx = torch.cat([idx, extra], dim=0)
        return idx[:num_samples].to(dtype=torch.long)

    # Fallback: approximate FPS on a reduced candidate pool, then exact greedy on that pool.
    candidate_budget = min(n_points, max(num_samples * 8, num_samples + 1))
    if candidate_budget < n_points:
        candidate_idx = torch.randperm(n_points, device=points.device)[:candidate_budget]
        candidate_points = points[candidate_idx]
    else:
        candidate_idx = torch.arange(n_points, device=points.device, dtype=torch.long)
        candidate_points = points

    selected_local = torch.empty((num_samples,), device=points.device, dtype=torch.long)
    if random_start:
        current = torch.randint(candidate_points.shape[0], (1,), device=points.device).item()
    else:
        current = 0
    selected_local[0] = current
    dist2 = torch.cdist(candidate_points[current : current + 1].float(), candidate_points.float()).squeeze(0).pow_(2)
    for i in range(1, num_samples):
        current = int(torch.argmax(dist2).item())
        selected_local[i] = current
        new_dist2 = torch.cdist(candidate_points[current : current + 1].float(), candidate_points.float()).squeeze(0).pow_(2)
        dist2 = torch.minimum(dist2, new_dist2)
    return candidate_idx[selected_local]


def sample_tokens_fps(tokens, num_samples, random_start=False):
    n_points = int(tokens.shape[1])
    batch_size = int(tokens.shape[0])
    if num_samples <= 0 or num_samples >= n_points:
        idx = torch.arange(n_points, device=tokens.device, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)
        return tokens, idx
    idx = torch.stack(
        [_fps_indices_single(tokens[b], num_samples=num_samples, random_start=random_start) for b in range(batch_size)],
        dim=0,
    )
    return gather_tokens(tokens, idx), idx


def knn_group(points, centers, k):
    """Group k nearest points for each center.

    Returns gathered points of shape [B, M, k, C] and indices [B, M, k].
    """
    batch_size, n_points, channels = points.shape
    num_centers = centers.shape[1]
    k_eff = min(max(1, int(k)), n_points)

    if torch_cluster_knn is not None:
        gathered_points = []
        gathered_idx = []
        for b in range(batch_size):
            pts_b = points[b]
            ctr_b = centers[b]
            edge_index = torch_cluster_knn(
                pts_b.float().contiguous(),
                ctr_b.float().contiguous(),
                k=k_eff,
            )
            # torch_cluster.knn returns [query_index, source_index].
            center, nbr = edge_index[0], edge_index[1]
            nbr = nbr.view(num_centers, k_eff)
            center = center.view(num_centers, k_eff)
            if not torch.all(center[:, 0] == torch.arange(num_centers, device=points.device, dtype=center.dtype)):
                order = torch.argsort(center * k_eff + torch.arange(center.numel(), device=center.device).view_as(center))
                nbr = nbr.reshape(-1)[order].view(num_centers, k_eff)
            gathered_idx.append(nbr.to(dtype=torch.long))
            gathered_points.append(pts_b[nbr.to(dtype=torch.long)])
        return torch.stack(gathered_points, dim=0), torch.stack(gathered_idx, dim=0)

    d2 = torch.cdist(centers.float(), points.float()).pow_(2)
    idx = torch.topk(d2, k=k_eff, dim=-1, largest=False).indices
    gathered = torch.gather(
        points.unsqueeze(1).expand(-1, num_centers, -1, -1),
        2,
        idx.unsqueeze(-1).expand(-1, -1, -1, channels),
    )
    return gathered, idx


def sample_tokens(tokens, num_samples):
    idx = sample_indices(tokens.shape[1], num_samples, tokens.device, tokens.shape[0])
    return gather_tokens(tokens, idx), idx


def split_surface_volume_predictions(pred, surf_query_pos, surface_channels):
    pred_surf = pred[:, :surf_query_pos.shape[1], :surface_channels]
    pred_vol = pred[:, surf_query_pos.shape[1]:, surface_channels:]
    return pred_surf, pred_vol


class QueryTypeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.surface = nn.Parameter(torch.zeros(1, 1, dim))
        self.volume = nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, surf_tokens, vol_tokens):
        return surf_tokens + self.surface, vol_tokens + self.volume


class CondInjection(nn.Module):
    def __init__(self, dim, cond_dim):
        super().__init__()
        self.cond_dim = int(cond_dim)
        self.proj = nn.Linear(cond_dim, dim) if cond_dim > 0 else None

    def forward(self, x, params):
        if self.proj is None or params is None:
            return x
        return x + self.proj(params).unsqueeze(1)


class SelfAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.0, spatial_dim=3, cond_dim=0):
        super().__init__()
        self.attn = CrossAttention(dim=dim, num_heads=num_heads, dropout=dropout, spatial_dim=spatial_dim)
        self.dropout = nn.Dropout(dropout)
        self.mlp = (
            SimulationParamModulatedMLP(dim=dim, hidden_dim=dim * 4, cond_dim=cond_dim, dropout=dropout)
            if cond_dim > 0
            else PlainMLP(dim=dim, hidden_dim=dim * 4, dropout=dropout)
        )

    def forward(self, x, params=None, pos=None):
        x = x + self.dropout(self.attn(q=x, kv=x, q_pos=pos, kv_pos=pos))
        x = x + self.mlp(x, params)
        return x


def gumbel_softmax(logits, tau):
    u = torch.rand_like(logits)
    g = -torch.log(-torch.log(u + 1e-8) + 1e-8)
    return torch.softmax((logits + g) / tau, dim=-1)


class SliceAttention(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.0, slice_num=64):
        super().__init__()
        assert dim % heads == 0, "dim must be divisible by heads"
        self.dim = dim
        self.heads = heads
        self.dim_head = dim // heads
        self.slice_num = slice_num

        self.in_project_x = nn.Linear(dim, dim)
        self.in_project_slice = nn.Linear(self.dim_head, slice_num)
        torch.nn.init.orthogonal_(self.in_project_slice.weight)
        if self.in_project_slice.bias is not None:
            nn.init.zeros_(self.in_project_slice.bias)
        self.to_q = nn.Linear(self.dim_head, self.dim_head, bias=False)
        self.to_k = nn.Linear(self.dim_head, self.dim_head, bias=False)
        self.to_v = nn.Linear(self.dim_head, self.dim_head, bias=False)
        self.to_out = nn.Sequential(nn.Linear(dim, dim), nn.Dropout(dropout))
        self.bias = nn.Parameter(torch.ones(1, heads, 1, 1) * 0.5)
        self.proj_temperature = nn.Sequential(
            nn.Linear(self.dim_head, slice_num),
            nn.GELU(),
            nn.Linear(slice_num, 1),
            nn.GELU(),
        )

    def forward(self, x, token_weights=None):
        bsz, num_tokens, _ = x.shape
        x_mid = self.in_project_x(x).reshape(bsz, num_tokens, self.heads, self.dim_head).permute(0, 2, 1, 3).contiguous()
        temperature = torch.clamp(self.proj_temperature(x_mid) + self.bias, min=0.01)
        logits = self.in_project_slice(x_mid)
        if self.training:
            slice_weights = gumbel_softmax(logits, temperature)
        else:
            slice_weights = torch.softmax(logits / temperature, dim=-1)

        if token_weights is None:
            point_weights = torch.ones((bsz, 1, num_tokens, 1), device=x.device, dtype=x.dtype)
        else:
            point_weights = token_weights[:, None, :, None].to(device=x.device, dtype=x.dtype)

        weighted_assign = slice_weights * point_weights
        slice_norm = torch.clamp(weighted_assign.sum(dim=2), min=1e-6)
        slice_token = torch.einsum("bhnc,bhng->bhgc", x_mid * point_weights, slice_weights)
        slice_token = slice_token / slice_norm[..., None]

        q = self.to_q(slice_token)
        k = self.to_k(slice_token)
        v = self.to_v(slice_token)
        out_slice = F.scaled_dot_product_attention(q, k, v)
        out_x = torch.einsum("bhgc,bhng->bhnc", out_slice, slice_weights)
        out_x = rearrange(out_x, "b h n d -> b n (h d)")
        return self.to_out(out_x)


class SliceBlock(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.0, slice_num=64, cond_dim=0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = SliceAttention(dim=dim, heads=num_heads, dropout=dropout, slice_num=slice_num)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = (
            SimulationParamModulatedMLP(dim=dim, hidden_dim=dim * 4, cond_dim=cond_dim, dropout=dropout)
            if cond_dim > 0
            else PlainMLP(dim=dim, hidden_dim=dim * 4, dropout=dropout)
        )

    def forward(self, x, params=None, token_weights=None):
        x = x + self.attn(self.norm1(x), token_weights=token_weights)
        x = x + self.mlp(self.norm2(x), params)
        return x


class SharedSplitBlock(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.0, spatial_dim=3, cond_dim=0):
        super().__init__()
        self.block = SelfAttentionBlock(dim=dim, num_heads=num_heads, dropout=dropout, spatial_dim=spatial_dim, cond_dim=cond_dim)

    def forward(self, surf, vol, params=None, surf_pos=None, vol_pos=None):
        return self.block(surf, params=params, pos=surf_pos), self.block(vol, params=params, pos=vol_pos)


class SharedCrossModalBlock(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.0, spatial_dim=3, cond_dim=0):
        super().__init__()
        self.cross = CrossAttention(dim=dim, num_heads=num_heads, dropout=dropout, spatial_dim=spatial_dim)
        self.dropout = nn.Dropout(dropout)
        self.surf_mlp = (
            SimulationParamModulatedMLP(dim=dim, hidden_dim=dim * 4, cond_dim=cond_dim, dropout=dropout)
            if cond_dim > 0
            else PlainMLP(dim=dim, hidden_dim=dim * 4, dropout=dropout)
        )
        self.vol_mlp = (
            SimulationParamModulatedMLP(dim=dim, hidden_dim=dim * 4, cond_dim=cond_dim, dropout=dropout)
            if cond_dim > 0
            else PlainMLP(dim=dim, hidden_dim=dim * 4, dropout=dropout)
        )

    def forward(self, surf, vol, params=None, surf_pos=None, vol_pos=None):
        surf = surf + self.dropout(self.cross(q=surf, kv=vol, q_pos=surf_pos, kv_pos=vol_pos))
        vol = vol + self.dropout(self.cross(q=vol, kv=surf, q_pos=vol_pos, kv_pos=surf_pos))
        surf = surf + self.surf_mlp(surf, params)
        vol = vol + self.vol_mlp(vol, params)
        return surf, vol


class SharedPerceiverBlock(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.0, spatial_dim=3, cond_dim=0):
        super().__init__()
        self.cross = CrossAttention(dim=dim, num_heads=num_heads, dropout=dropout, spatial_dim=spatial_dim)
        self.dropout = nn.Dropout(dropout)
        self.mlp = (
            SimulationParamModulatedMLP(dim=dim, hidden_dim=dim * 4, cond_dim=cond_dim, dropout=dropout)
            if cond_dim > 0
            else PlainMLP(dim=dim, hidden_dim=dim * 4, dropout=dropout)
        )

    def forward(self, x, geometry, params=None, x_pos=None, geometry_pos=None):
        x = x + self.dropout(self.cross(q=x, kv=geometry, q_pos=x_pos, kv_pos=geometry_pos))
        x = x + self.mlp(x, params)
        return x


def init_linear_layer_weights(module):
    if isinstance(module, nn.Linear):
        torch.nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)
    elif isinstance(module, nn.LayerNorm):
        nn.init.constant_(module.bias, 0.0)
        nn.init.constant_(module.weight, 1.0)
