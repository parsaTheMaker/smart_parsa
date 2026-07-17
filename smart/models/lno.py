"""Latent Neural Operator (LNO) for point-cloud field queries.

This is a PyTorch port of the public LNO design: geometry/source points are
projected to a small set of learned latent modes, latent self-attention is
applied with Galerkin linear attention, and arbitrary query points are decoded
from the modes.  The DrivAerML adapter uses coordinates as the source branch
because the benchmark provides geometry, not an input physical field.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .family_common import CondInjection, QueryTypeEmbedding, split_surface_volume_predictions


class _LNOBackboneMLP(nn.Module):
    """The residual MLP used by the reference LNO implementation."""

    def __init__(self, input_dim, hidden_dim, output_dim, n_layer, activation):
        super().__init__()
        self.input = nn.Linear(int(input_dim), int(hidden_dim))
        self.hidden = nn.ModuleList([nn.Linear(int(hidden_dim), int(hidden_dim)) for _ in range(int(n_layer))])
        self.output = nn.Linear(int(hidden_dim), int(output_dim))
        self.activation = activation

    def forward(self, x):
        x = self.activation(self.input(x))
        for layer in self.hidden:
            x = x + self.activation(layer(x))
        return self.output(x)


class _GalerkinAttention(nn.Module):
    """Linear Galerkin attention from the public LNO implementation."""

    def __init__(self, dim, heads):
        super().__init__()
        dim = int(dim)
        heads = int(heads)
        if dim % heads != 0:
            raise ValueError(f"LNO dimension {dim} must be divisible by heads {heads}.")
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.to_qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.key_norm = nn.LayerNorm(self.head_dim)
        self.value_norm = nn.LayerNorm(self.head_dim)
        self.to_out = nn.Linear(dim, dim) if heads > 1 else nn.Identity()

    def forward(self, x):
        batch_size, sequence_length, _ = x.shape
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q = q.view(batch_size, sequence_length, self.heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, sequence_length, self.heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, sequence_length, self.heads, self.head_dim).transpose(1, 2)
        k = self.key_norm(k)
        v = self.value_norm(v)
        kv = torch.matmul(k.transpose(-1, -2), v)
        out = torch.matmul(q, kv) / float(max(1, sequence_length))
        out = out.transpose(1, 2).reshape(batch_size, sequence_length, self.dim)
        return self.to_out(out)


class _LNOAttentionBlock(nn.Module):
    def __init__(self, dim, heads, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(int(dim))
        self.attention = _GalerkinAttention(dim, heads)
        self.dropout = nn.Dropout(float(dropout))
        self.norm2 = nn.LayerNorm(int(dim))
        self.mlp = nn.Sequential(
            nn.Linear(int(dim), 2 * int(dim)),
            nn.GELU(),
            nn.Linear(2 * int(dim), int(dim)),
        )

    def forward(self, x):
        x = x + self.dropout(self.attention(self.norm1(x)))
        return x + self.mlp(self.norm2(x))


class LNO(nn.Module):
    expects_geo_log_density = False

    def __init__(
        self,
        spatial_dim=3,
        surface_channels=1,
        volume_channels=3,
        parameter_channels=0,
        n_block=6,
        n_mode=64,
        n_dim=128,
        n_head=8,
        n_layer=2,
        attn="Galerkin",
        activation="GELU",
        pos_scale_factor=1.0,
        query_chunk_size=65536,
        dropout=0.0,
        normalize_scores=True,
        encode_score_temperature=2.0,
        decode_score_temperature=1.0,
        **_unused,
    ):
        super().__init__()
        if int(spatial_dim) != 3:
            raise ValueError("LNO currently expects 3D coordinates.")
        if str(attn).lower() not in {"galerkin", "linear"}:
            raise ValueError("This large-point-cloud LNO adapter supports the reference Galerkin attention path.")
        if str(activation).lower() not in {"gelu"}:
            raise ValueError("LNO currently uses the reference GELU activation.")
        self.surface_channels = int(surface_channels)
        self.volume_channels = int(volume_channels)
        self.parameter_channels = int(parameter_channels)
        self.n_mode = int(n_mode)
        self.n_dim = int(n_dim)
        self.pos_scale_factor = float(pos_scale_factor)
        self.query_chunk_size = max(1, int(query_chunk_size))
        self.normalize_scores = bool(normalize_scores)
        self.encode_score_temperature = float(encode_score_temperature)
        self.decode_score_temperature = float(decode_score_temperature)
        if self.encode_score_temperature <= 0.0 or self.decode_score_temperature <= 0.0:
            raise ValueError("LNO score temperatures must be positive.")
        act = nn.GELU()

        self.trunk_projector = _LNOBackboneMLP(3, n_dim, n_dim, n_layer, act)
        self.branch_projector = _LNOBackboneMLP(3, n_dim, n_dim, n_layer, act)
        self.attention_projector = _LNOBackboneMLP(n_dim, n_dim, n_mode, n_layer, act)
        self.branch_cond = CondInjection(n_dim, parameter_channels)
        self.query_cond = CondInjection(n_dim, parameter_channels)
        self.query_type = QueryTypeEmbedding(n_dim)
        self.attention_blocks = nn.ModuleList(
            [_LNOAttentionBlock(n_dim, n_head, dropout=dropout) for _ in range(int(n_block))]
        )
        self.latent_norm = nn.LayerNorm(n_dim)
        self.output_mlp = _LNOBackboneMLP(n_dim, n_dim, self.surface_channels + self.volume_channels, n_layer, act)

    def _score_logits(self, logits, dim, temperature):
        logits = logits.float()
        if self.normalize_scores:
            mean = logits.mean(dim=dim, keepdim=True)
            std = logits.std(dim=dim, keepdim=True, unbiased=False).clamp_min(1.0e-4)
            logits = (logits - mean) / std
        return logits * float(temperature)

    def encode_geometry(self, geo, params=None):
        source = geo * self.pos_scale_factor
        source_trunk = self.trunk_projector(source)
        branch = self.branch_cond(self.branch_projector(source), params)
        encode_logits = self.attention_projector(source_trunk)
        encode_logits = self._score_logits(encode_logits, dim=1, temperature=self.encode_score_temperature)
        encode_weights = torch.softmax(encode_logits, dim=1)
        # Accumulate over large point clouds in fp32. Casting the normalized
        # weights to fp16 before this reduction loses useful geometry signal.
        latent = torch.einsum("bnm,bnd->bmd", encode_weights, branch.float()).to(dtype=branch.dtype)
        for block in self.attention_blocks:
            latent = block(latent)
        return self.latent_norm(latent)

    def decode_features(self, latent, surf_query_pos, vol_query_pos, params=None):
        surf = self.trunk_projector(surf_query_pos * self.pos_scale_factor)
        vol = self.trunk_projector(vol_query_pos * self.pos_scale_factor)
        surf, vol = self.query_type(surf, vol)
        query = torch.cat([surf, vol], dim=1)
        query = self.query_cond(query, params)
        decode_logits = self.attention_projector(query)
        decode_logits = self._score_logits(decode_logits, dim=-1, temperature=self.decode_score_temperature)
        decode_weights = torch.softmax(decode_logits, dim=-1)
        decoded = torch.einsum("bnm,bmd->bnd", decode_weights, latent.float()).to(dtype=query.dtype)
        pred = self.output_mlp(decoded)
        return split_surface_volume_predictions(pred, surf_query_pos, self.surface_channels)

    def forward(self, geo, surf_query_pos, vol_query_pos, params=None, geo_log_density=None, return_latent=False):
        latent = self.encode_geometry(geo, params=params)
        pred_surf, pred_vol = self.decode_features(latent, surf_query_pos, vol_query_pos, params=params)
        if return_latent:
            return pred_surf, pred_vol, latent
        return pred_surf, pred_vol

    @torch.inference_mode()
    def inference(self, geo, surf_query_pos, vol_query_pos, params=None, geo_log_density=None):
        latent = self.encode_geometry(geo, params=params)
        query_count = int(surf_query_pos.shape[1] + vol_query_pos.shape[1])
        if query_count <= self.query_chunk_size:
            return self.decode_features(latent, surf_query_pos, vol_query_pos, params=params)

        surf_count = int(surf_query_pos.shape[1])
        full_query = torch.cat([surf_query_pos, vol_query_pos], dim=1)
        surf_chunks = []
        vol_chunks = []
        for start in range(0, query_count, self.query_chunk_size):
            stop = min(start + self.query_chunk_size, query_count)
            query = full_query[:, start:stop]
            surf_len = max(0, min(stop, surf_count) - start)
            surf_chunk = query[:, :surf_len]
            vol_chunk = query[:, surf_len:]
            if surf_chunk.shape[1] > 0:
                surf_pred, _ = self.decode_features(latent, surf_chunk, surf_chunk[:, :0], params=params)
                surf_chunks.append(surf_pred)
            if vol_chunk.shape[1] > 0:
                _, vol_pred = self.decode_features(latent, vol_chunk[:, :0], vol_chunk, params=params)
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


class LNOWithLatent(LNO):
    pass
