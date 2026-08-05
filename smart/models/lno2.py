"""LNO2: stable LNO adapter for large unstructured point clouds.

The model keeps the LNO Physics-Cross-Attention structure: geometry is mapped
to a fixed latent basis, the operator is learned in latent space, and queries
are decoded at arbitrary coordinates.  Score standardization follows the
reference LNO path, while the encoder uses a two-pass streaming reduction so
large source clouds do not require a point-by-mode weight tensor.  The
optional Fourier residual supplies bounded local detail for CFD fields without
replacing the latent operator.
"""

from __future__ import annotations

from contextlib import nullcontext

import torch
import torch.nn as nn

from .family_common import CondInjection, QueryTypeEmbedding
from .lno import _LNOBackboneMLP


class _LNO2GalerkinAttention(nn.Module):
    """Galerkin attention with fp32-only reduction for AMP stability."""

    def __init__(self, dim, heads):
        super().__init__()
        dim = int(dim)
        heads = int(heads)
        if dim % heads != 0:
            raise ValueError(f"LNO2 dimension {dim} must be divisible by heads {heads}.")
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

        # The Galerkin contractions sum over the latent sequence and can
        # overflow in fp16 even when q/k/v are individually finite.  Compute
        # only this reduction in fp32; the surrounding network remains AMP.
        with torch.autocast(device_type=x.device.type, enabled=False):
            q = q.float()
            k = self.key_norm(k.float())
            v = self.value_norm(v.float())
            kv = torch.matmul(k.transpose(-1, -2), v)
            out = torch.matmul(q, kv) / float(max(1, sequence_length))
            out = out.transpose(1, 2).reshape(batch_size, sequence_length, self.dim)
            out = self.to_out(out)
        return out.to(dtype=x.dtype)


class _LNO2AttentionBlock(nn.Module):
    def __init__(self, dim, heads, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(int(dim))
        self.attention = _LNO2GalerkinAttention(dim, heads)
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


class LNO2(nn.Module):
    """LNO with stable reference-style scores and a lightweight query path."""

    expects_geo_log_density = False

    def __init__(
        self,
        spatial_dim=3,
        surface_channels=1,
        volume_channels=3,
        parameter_channels=0,
        n_block=8,
        n_mode=256,
        n_dim=256,
        n_head=8,
        n_layer=2,
        attn="Galerkin",
        activation="GELU",
        pos_scale_factor=1.0,
        query_chunk_size=65536,
        encode_chunk_size=32768,
        query_residual_scale=0.1,
        score_standardization=True,
        score_std_floor=1.0e-4,
        encode_score_temperature=2.0,
        decode_score_temperature=2.0,
        use_fourier_geometry_features=True,
        use_fourier_query_features=False,
        fourier_num_bands=6,
        fourier_min_frequency=1.0,
        fourier_max_frequency=32.0,
        fourier_residual_scale=0.1,
        dropout=0.0,
        **_unused,
    ):
        super().__init__()
        if int(spatial_dim) != 3:
            raise ValueError("LNO2 currently expects 3D coordinates.")
        if str(attn).lower() not in {"galerkin", "linear"}:
            raise ValueError("LNO2 supports the reference Galerkin attention path.")
        if str(activation).lower() != "gelu":
            raise ValueError("LNO2 currently uses the reference GELU activation.")

        self.surface_channels = int(surface_channels)
        self.volume_channels = int(volume_channels)
        self.parameter_channels = int(parameter_channels)
        self.n_mode = int(n_mode)
        self.n_dim = int(n_dim)
        self.pos_scale_factor = float(pos_scale_factor)
        self.query_chunk_size = max(1, int(query_chunk_size))
        self.encode_chunk_size = max(1, int(encode_chunk_size))
        self.query_residual_scale = nn.Parameter(torch.tensor(float(query_residual_scale)))
        self.score_standardization = bool(score_standardization)
        self.score_std_floor = float(score_std_floor)
        self.encode_score_temperature = float(encode_score_temperature)
        self.decode_score_temperature = float(decode_score_temperature)
        self.use_fourier_geometry_features = bool(use_fourier_geometry_features)
        self.use_fourier_query_features = bool(use_fourier_query_features)
        if self.score_std_floor <= 0.0:
            raise ValueError("LNO2 score_std_floor must be positive.")
        if self.encode_score_temperature <= 0.0 or self.decode_score_temperature <= 0.0:
            raise ValueError("LNO2 score temperatures must be positive.")
        act = nn.GELU()

        fourier_num_bands = max(1, int(fourier_num_bands))
        fourier_min_frequency = float(fourier_min_frequency)
        fourier_max_frequency = float(fourier_max_frequency)
        needs_fourier_buffer = self.use_fourier_geometry_features or self.use_fourier_query_features
        if needs_fourier_buffer:
            if fourier_min_frequency <= 0.0 or fourier_max_frequency < fourier_min_frequency:
                raise ValueError("LNO2 Fourier frequencies must be positive and ordered.")
            frequencies = torch.logspace(
                torch.log10(torch.tensor(fourier_min_frequency)),
                torch.log10(torch.tensor(fourier_max_frequency)),
                steps=fourier_num_bands,
            )
            self.register_buffer("fourier_frequencies", frequencies, persistent=False)
        else:
            self.fourier_frequencies = None
        fourier_dim = 2 * int(spatial_dim) * fourier_num_bands
        coordinate_input_dim = int(spatial_dim) + (
            fourier_dim if self.use_fourier_geometry_features else 0
        )
        self.trunk_projector = _LNOBackboneMLP(coordinate_input_dim, n_dim, n_dim, n_layer, act)
        self.branch_projector = _LNOBackboneMLP(coordinate_input_dim, n_dim, n_dim, n_layer, act)
        self.attention_projector = _LNOBackboneMLP(n_dim, n_dim, n_mode, n_layer, act)
        self.branch_cond = CondInjection(n_dim, parameter_channels)
        self.query_cond = CondInjection(n_dim, parameter_channels)
        self.query_type = QueryTypeEmbedding(n_dim)

        self.attention_blocks = nn.ModuleList(
            [_LNO2AttentionBlock(n_dim, n_head, dropout=dropout) for _ in range(int(n_block))]
        )
        self.latent_norm = nn.LayerNorm(n_dim)

        # Share the decoder trunk, but keep the surface and volume projections
        # independent because they represent different physical quantities.
        self.output_trunk = _LNOBackboneMLP(n_dim, n_dim, n_dim, n_layer, act)
        self.surface_head = nn.Linear(n_dim, self.surface_channels)
        self.volume_head = nn.Linear(n_dim, self.volume_channels)
        self.query_residual = nn.Linear(n_dim, n_dim)

        if self.use_fourier_query_features:
            self.fourier_query_projector = _LNOBackboneMLP(
                fourier_dim,
                n_dim,
                n_dim,
                1,
                act,
            )
            self.fourier_query_norm = nn.LayerNorm(n_dim)
            self.fourier_context = nn.Linear(n_dim, n_dim)
            self.fourier_gate = nn.Linear(n_dim, 1)
            self.fourier_residual_scale = nn.Parameter(torch.tensor(float(fourier_residual_scale)))
        else:
            self.fourier_query_projector = None
            self.fourier_query_norm = None
            self.fourier_context = None
            self.fourier_gate = None
            self.fourier_residual_scale = None

    @staticmethod
    def _fp32_autocast_disabled(device):
        if device.type in {"cuda", "cpu"}:
            return torch.autocast(device_type=device.type, enabled=False)
        return nullcontext()

    def _score_logits_fp32(self, features):
        """Project and retain score logits in fp32 without disabling global AMP."""
        with self._fp32_autocast_disabled(features.device):
            return self.attention_projector(features.float()).float()

    def _standardize_scores(self, logits, dim):
        logits = logits.float()
        if not self.score_standardization:
            return logits
        mean = logits.mean(dim=dim, keepdim=True)
        centered = logits - mean
        variance = centered.square().mean(dim=dim, keepdim=True)
        std = variance.sqrt().clamp_min(self.score_std_floor)
        return centered / std

    def _encode_source_chunk(self, geo_chunk, params):
        source = self._coordinate_features(geo_chunk)
        trunk = self.trunk_projector(source)
        branch = self.branch_cond(self.branch_projector(source), params)
        logits = self._score_logits_fp32(trunk)
        return logits, branch

    def _encoder_score_statistics(self, geo, params):
        """Get global per-mode score statistics without retaining an AMP graph."""
        score_sum = None
        score_square_sum = None
        count = 0
        with torch.no_grad():
            for start in range(0, int(geo.shape[1]), self.encode_chunk_size):
                stop = min(start + self.encode_chunk_size, int(geo.shape[1]))
                logits, _branch = self._encode_source_chunk(geo[:, start:stop], params)
                logits = logits.float()
                chunk_sum = logits.sum(dim=1)
                chunk_square_sum = logits.square().sum(dim=1)
                score_sum = chunk_sum if score_sum is None else score_sum + chunk_sum
                score_square_sum = (
                    chunk_square_sum
                    if score_square_sum is None
                    else score_square_sum + chunk_square_sum
                )
                count += stop - start
        if score_sum is None or count <= 0:
            raise ValueError("LNO2 cannot standardize an empty geometry cloud.")
        mean = score_sum / float(count)
        variance = (score_square_sum / float(count) - mean.square()).clamp_min(0.0)
        return mean, variance.sqrt().clamp_min(self.score_std_floor)

    def _fourier_features(self, query_pos):
        if not self.use_fourier_query_features or self.fourier_frequencies is None:
            return None
        angles = 2.0 * torch.pi * query_pos.float().unsqueeze(-1) * self.fourier_frequencies
        features = torch.cat([angles.sin(), angles.cos()], dim=-1)
        return features.flatten(start_dim=-2).to(dtype=query_pos.dtype)

    def _coordinate_features(self, position):
        """Build the raw-plus-Fourier coordinates used by source and query trunks."""
        scaled_position = position * self.pos_scale_factor
        if not self.use_fourier_geometry_features:
            return scaled_position
        angles = 2.0 * torch.pi * position.float().unsqueeze(-1) * self.fourier_frequencies
        fourier = torch.cat([angles.sin(), angles.cos()], dim=-1)
        fourier = fourier.flatten(start_dim=-2).to(dtype=position.dtype)
        return torch.cat([scaled_position, fourier], dim=-1)

    def encode_geometry(self, geo, params=None):
        """Encode with exact streaming softmax after global score standardization."""
        num_points = int(geo.shape[1])
        score_mean = score_std = None
        if self.score_standardization:
            score_mean, score_std = self._encoder_score_statistics(geo, params)
        running_max = None
        running_denom = None
        running_numerator = None

        for start in range(0, num_points, self.encode_chunk_size):
            stop = min(start + self.encode_chunk_size, num_points)
            logits, branch = self._encode_source_chunk(geo[:, start:stop], params)
            if score_mean is not None:
                logits = (logits - score_mean.unsqueeze(1)) / score_std.unsqueeze(1)
                logits = logits * self.encode_score_temperature
            chunk_max = logits.amax(dim=1).detach()

            if running_max is None:
                running_max = chunk_max
                old_scale = None
            else:
                new_max = torch.maximum(running_max, chunk_max).detach()
                old_scale = torch.exp(running_max - new_max)
                running_max = new_max

            exp_logits = torch.exp(logits - running_max.unsqueeze(1))
            chunk_denom = exp_logits.sum(dim=1)
            chunk_numerator = torch.einsum("bnm,bnd->bmd", exp_logits, branch.float())

            if running_denom is None:
                running_denom = chunk_denom
                running_numerator = chunk_numerator
            else:
                running_denom = running_denom * old_scale + chunk_denom
                running_numerator = running_numerator * old_scale.unsqueeze(-1) + chunk_numerator

        if running_denom is None:
            raise ValueError("LNO2 cannot encode an empty geometry cloud.")
        latent = running_numerator / running_denom.unsqueeze(-1).clamp_min(1.0e-12)
        # Keep only the score/reduction in fp32.  The latent attention blocks
        # should still follow the surrounding AMP activation dtype.
        latent = latent.to(dtype=branch.dtype)
        for block in self.attention_blocks:
            latent = block(latent)
        return self.latent_norm(latent)

    def _build_query_features(self, query_pos, query_type, params):
        query = self.trunk_projector(self._coordinate_features(query_pos))
        if query_type == "surface":
            query, _ = self.query_type(query, query[:, :0])
        else:
            _, query = self.query_type(query[:, :0], query)
        return self.query_cond(query, params)

    def _decode_query_chunk(self, latent, query_pos, query_type, params):
        query = self._build_query_features(query_pos, query_type, params)
        decode_logits = self._score_logits_fp32(query)
        decode_logits = self._standardize_scores(decode_logits, dim=-1)
        decode_logits = decode_logits * self.decode_score_temperature
        decode_weights = torch.softmax(decode_logits, dim=-1)
        decoded = torch.einsum("bnm,bmd->bnd", decode_weights, latent.float())
        local = self.query_residual(query)
        decoded = decoded + self.query_residual_scale.tanh().float() * local.float()
        if self.use_fourier_query_features:
            fourier = self._fourier_features(query_pos)
            detail = self.fourier_query_projector(fourier)
            detail = self.fourier_query_norm(detail)
            context = self.fourier_context(latent.float().mean(dim=1, keepdim=True))
            detail = detail.float() + context
            gate = torch.sigmoid(self.fourier_gate(detail)).float()
            decoded = decoded + self.fourier_residual_scale.tanh().float() * gate * detail
        decoded = self.output_trunk(decoded.to(dtype=query.dtype))
        if query_type == "surface":
            return self.surface_head(decoded)
        return self.volume_head(decoded)

    def _decode_queries(self, latent, query_pos, query_type, params):
        if query_pos.shape[1] == 0:
            channels = self.surface_channels if query_type == "surface" else self.volume_channels
            return query_pos.new_empty((query_pos.shape[0], 0, channels))
        outputs = []
        for start in range(0, int(query_pos.shape[1]), self.query_chunk_size):
            stop = min(start + self.query_chunk_size, int(query_pos.shape[1]))
            outputs.append(self._decode_query_chunk(latent, query_pos[:, start:stop], query_type, params))
        return torch.cat(outputs, dim=1)

    def decode_features(self, latent, surf_query_pos, vol_query_pos, params=None):
        surf = self._decode_queries(latent, surf_query_pos, "surface", params)
        vol = self._decode_queries(latent, vol_query_pos, "volume", params)
        return surf, vol

    def forward(self, geo, surf_query_pos, vol_query_pos, params=None, geo_log_density=None, return_latent=False):
        del geo_log_density
        latent = self.encode_geometry(geo, params=params)
        pred_surf, pred_vol = self.decode_features(latent, surf_query_pos, vol_query_pos, params=params)
        if return_latent:
            return pred_surf, pred_vol, latent
        return pred_surf, pred_vol

    @torch.inference_mode()
    def inference(self, geo, surf_query_pos, vol_query_pos, params=None, geo_log_density=None):
        del geo_log_density
        latent = self.encode_geometry(geo, params=params)
        return self.decode_features(latent, surf_query_pos, vol_query_pos, params=params)


class LNO2WithLatent(LNO2):
    pass
