"""SMART-SAT: density-compensated SMART with tempered anchor correction.

This is the canonical SAT implementation. It keeps SMART's progressive block
refinement and shared decoder attention, while correcting the main places where
non-uniform surface sampling leaks into the representation:

1. geometry-to-latent aggregation uses a density-compensated attention rule;
2. latent anchors are sampled with a tempered inverse-density law;
3. latent-token reuse paths in the encoder and decoder use a proposal-matched
   density correction for the resulting latent anchor distribution.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .smart import ModulatedPositionalEmbedding, PlainMLP, RotaryPositionalEmbedding, SimulationParamModulatedMLP

try:
    from utils.geometry_density import estimate_log_sampling_density
except ImportError:  # pragma: no cover - package-style imports
    from smart.utils.geometry_density import estimate_log_sampling_density


class DensityCompensatedCrossAttention(nn.Module):
    """Standard attention with an explicit density correction term."""

    def __init__(self, dim, num_heads=8, spatial_dim=3):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.norm_q = nn.LayerNorm(dim, eps=1e-6)
        self.norm_kv = nn.LayerNorm(dim, eps=1e-6)

        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.out_proj = nn.Linear(dim, dim)
        self.rope = RotaryPositionalEmbedding(dim=dim // num_heads, spatial_dim=spatial_dim)

    def forward(self, q, kv, q_pos=None, kv_pos=None, kv_log_density=None):
        q_in = self.norm_q(q)
        kv_in = self.norm_kv(kv)

        q_proj = self.q(q_in)
        kv_proj = self.kv(kv_in)

        q_heads = rearrange(q_proj, "b q (h d) -> b h q d", h=self.num_heads, d=self.head_dim)
        k_heads, v_heads = torch.tensor_split(
            rearrange(kv_proj, "b kv (h d) -> b h kv d", h=2 * self.num_heads, d=self.head_dim),
            2,
            dim=1,
        )

        if q_pos is not None and kv_pos is not None:
            q_heads = self.rope(q_heads, q_pos)
            k_heads = self.rope(k_heads, kv_pos)

        attn_bias = None
        if kv_log_density is not None:
            attn_bias = -kv_log_density.float()[:, None, None, :]

        out = F.scaled_dot_product_attention(
            q_heads.float(),
            k_heads.float(),
            v_heads.float(),
            attn_mask=attn_bias,
            dropout_p=0.0,
        )

        out = rearrange(out, "b h q d -> b q (h d)").to(dtype=q_proj.dtype)
        return self.out_proj(out)


class DensityCompensatedEncoderBlock(nn.Module):
    """SMART encoder block with density-compensated geometry attention."""

    def __init__(self, dim, num_heads=8, dropout=0.1, spatial_dim=3, cond_dim=2):
        super().__init__()
        self.geo_attn = DensityCompensatedCrossAttention(dim=dim, num_heads=num_heads, spatial_dim=spatial_dim)
        self.cross_attn = DensityCompensatedCrossAttention(dim=dim, num_heads=num_heads, spatial_dim=spatial_dim)
        self.attn_dropout = nn.Dropout(dropout)

        if cond_dim > 0:
            self.mlp = SimulationParamModulatedMLP(dim=dim, hidden_dim=dim * 4, cond_dim=cond_dim, dropout=dropout)
        else:
            self.mlp = PlainMLP(dim=dim, hidden_dim=dim * 4, dropout=dropout)

    def forward(
        self,
        latent_geometry,
        geometry_tokens,
        params,
        latent_geometry_pos=None,
        geometry_pos=None,
        geometry_log_density=None,
        latent_geometry_log_density=None,
    ):
        latent_geometry_cross = latent_geometry + self.attn_dropout(
            self.geo_attn(
                q=latent_geometry,
                kv=geometry_tokens,
                q_pos=latent_geometry_pos,
                kv_pos=geometry_pos,
                kv_log_density=geometry_log_density,
            )
        )

        latent_geometry_self = latent_geometry + self.attn_dropout(
            self.cross_attn(
                q=latent_geometry,
                kv=latent_geometry_cross,
                q_pos=latent_geometry_pos,
                kv_pos=latent_geometry_pos,
                kv_log_density=latent_geometry_log_density,
            )
        )

        latent_geometry_mlp = latent_geometry_self + self.mlp(latent_geometry_self, params)
        return latent_geometry_mlp, latent_geometry_cross


def gather_geometry(geometry, idx):
    """Gather geometry points using per-batch integer indices."""
    return torch.gather(geometry, 1, idx.unsqueeze(-1).expand(-1, -1, geometry.shape[-1]))


def sample_geometry(geometry, num_samples, with_replacement=False):
    """Sample geometry points and return the chosen indices."""
    n_points = int(geometry.shape[1])
    batch_size = int(geometry.shape[0])
    if num_samples <= 0:
        idx = torch.arange(n_points, device=geometry.device, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)
        return geometry, idx
    if with_replacement:
        idx = torch.randint(0, n_points, (batch_size, num_samples), device=geometry.device, dtype=torch.long)
    else:
        if num_samples >= n_points:
            idx = torch.arange(n_points, device=geometry.device, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)
            return geometry, idx
        idx = torch.stack(
            [torch.randperm(n_points, device=geometry.device)[:num_samples] for _ in range(batch_size)],
            dim=0,
        )
    sampled_geometry = gather_geometry(geometry, idx)
    return sampled_geometry, idx


def sample_geometry_density_tempered(geometry, num_samples, log_density, alpha=0.25):
    """Sample geometry with probabilities proportional to rho(x)^(-alpha)."""
    n_points = int(geometry.shape[1])
    batch_size = int(geometry.shape[0])
    if num_samples <= 0 or num_samples >= n_points:
        idx = torch.arange(n_points, device=geometry.device, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)
        return geometry, idx

    alpha = float(alpha)
    if alpha <= 0.0:
        return sample_geometry(geometry, num_samples)

    weights = torch.exp(-alpha * log_density.float())
    weights = torch.clamp(weights, min=1e-12)
    probs = weights / torch.clamp(weights.sum(dim=1, keepdim=True), min=1e-12)
    idx = torch.multinomial(probs, num_samples=num_samples, replacement=False)
    sampled_geometry = gather_geometry(geometry, idx)
    return sampled_geometry, idx


class DensityCompensatedDecoderBlock(nn.Module):
    """Decoder block that reuses latent tokens with density-aware key weighting."""

    def __init__(self, dim, num_heads=8, dropout=0.1, spatial_dim=3, cond_dim=2, shared_attn=None, shared_mlp=None):
        super().__init__()
        self.attn = DensityCompensatedCrossAttention(dim=dim, num_heads=num_heads, spatial_dim=spatial_dim) if shared_attn is None else shared_attn
        self.attn_dropout = nn.Dropout(dropout)

        if cond_dim > 0:
            self.mlp = SimulationParamModulatedMLP(dim=dim, hidden_dim=dim * 4, cond_dim=cond_dim, dropout=dropout) if shared_mlp is None else shared_mlp
        else:
            self.mlp = PlainMLP(dim=dim, hidden_dim=dim * 4, dropout=dropout) if shared_mlp is None else shared_mlp

    def forward(self, queries, latent_geometry, params, queries_pos=None, latent_geometry_pos=None, latent_geometry_log_density=None):
        queries = queries + self.attn_dropout(
            self.attn(
                q=queries,
                kv=latent_geometry,
                q_pos=queries_pos,
                kv_pos=latent_geometry_pos,
                kv_log_density=latent_geometry_log_density,
            )
        )
        queries = queries + self.mlp(queries, params)
        return queries


class SMARTSAT(nn.Module):
    """SMART variant with density-compensated stochastic progressive refinement."""

    def __init__(
        self,
        spatial_dim=3,
        surface_channels=1,
        volume_channels=3,
        parameter_channels=2,
        latent_dim=256,
        latent_geometry_points=4096,
        subsampled_geometry_points=16384,
        num_encoder_decoder_blocks=8,
        num_heads=8,
        pos_scale_factor=1000,
        dropout=0.0,
        subregion_size=262144,
        density_knn_k=8,
        density_neighbor_hops=1,
        density_estimator="rk2",
        latent_density_alpha=0.25,
        subsampled_geometry_with_replacement=False,
    ):
        super().__init__()
        assert surface_channels > 0 and volume_channels > 0, "surface_channels and volume_channels must be positive integers."

        self.surface_channels = surface_channels
        self.volume_channels = volume_channels
        self.num_geo = latent_geometry_points
        self.subsampled_geometry_points = subsampled_geometry_points
        self.subsampled_geometry_with_replacement = bool(subsampled_geometry_with_replacement)
        self.pos_scale_factor = pos_scale_factor
        self.density_knn_k = int(density_knn_k)
        self.density_neighbor_hops = int(density_neighbor_hops)
        self.density_estimator = str(density_estimator)
        self.latent_density_alpha = float(latent_density_alpha)

        self.pos_encoder = ModulatedPositionalEmbedding(latent_dim, spatial_dim)

        self.encoder_blocks = nn.ModuleList(
            [
                DensityCompensatedEncoderBlock(
                    dim=latent_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    spatial_dim=spatial_dim,
                    cond_dim=parameter_channels,
                )
                for _ in range(num_encoder_decoder_blocks)
            ]
        )
        self.decoder_blocks = nn.ModuleList(
            [
                DensityCompensatedDecoderBlock(
                    dim=latent_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    spatial_dim=spatial_dim,
                    cond_dim=parameter_channels,
                    shared_attn=self.encoder_blocks[i].cross_attn,
                    shared_mlp=self.encoder_blocks[i].mlp,
                )
                for i in range(num_encoder_decoder_blocks)
            ]
        )

        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, surface_channels + volume_channels),
        )

        self.subregion_size = subregion_size

    def initialize_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def encode(self, geo, params, return_final=False, geo_log_density=None, return_latent_density=False):
        if geo_log_density is None:
            full_geo_log_density = estimate_log_sampling_density(
                geo,
                knn_k=self.density_knn_k,
                neighbor_hops=self.density_neighbor_hops,
                estimator=self.density_estimator,
            )
        else:
            full_geo_log_density = geo_log_density.to(device=geo.device, dtype=geo.dtype)
        geo_scaled = geo * self.pos_scale_factor

        latent_geo_pos, latent_geo_idx = sample_geometry_density_tempered(
            geo_scaled,
            self.num_geo,
            log_density=full_geo_log_density,
            alpha=self.latent_density_alpha,
        )
        latent_geo_log_density = torch.gather(full_geo_log_density, 1, latent_geo_idx)
        # If anchors are sampled with p(x) propto rho(x)^(-alpha) from an
        # original cloud with density rho(x), the retained latent-anchor
        # density scales like rho(x)^(1 - alpha). Reuse paths should therefore
        # correct by (1 - alpha) * log rho rather than the full log rho.
        latent_reuse_log_density = latent_geo_log_density * max(0.0, 1.0 - self.latent_density_alpha)
        latent_geo_emb = self.pos_encoder(latent_geo_pos)

        intermediate_latent_geometries = []
        for block in self.encoder_blocks:
            sub_geo_raw, sub_idx = sample_geometry(
                geo,
                self.subsampled_geometry_points,
                with_replacement=self.subsampled_geometry_with_replacement,
            )
            sub_geo_pos = sub_geo_raw * self.pos_scale_factor
            sub_geo_emb = self.pos_encoder(sub_geo_pos)
            sub_geo_log_density = torch.gather(full_geo_log_density, 1, sub_idx)

            latent_geo_emb, e_ca = block(
                latent_geo_emb,
                sub_geo_emb,
                params,
                latent_geometry_pos=latent_geo_pos,
                geometry_pos=sub_geo_pos,
                geometry_log_density=sub_geo_log_density,
                latent_geometry_log_density=latent_reuse_log_density,
            )
            intermediate_latent_geometries.append(e_ca)

        if return_final:
            if return_latent_density:
                return intermediate_latent_geometries, latent_geo_pos, latent_geo_emb, latent_reuse_log_density
            return intermediate_latent_geometries, latent_geo_pos, latent_geo_emb
        if return_latent_density:
            return intermediate_latent_geometries, latent_geo_pos, latent_reuse_log_density
        return intermediate_latent_geometries, latent_geo_pos

    def decode_features(self, intermediate_latent_geometries, latent_geo_pos, params, query_pos, latent_geo_log_density=None):
        query_pos = query_pos * self.pos_scale_factor
        query_emb = self.pos_encoder(query_pos)

        for e_ca, block in zip(intermediate_latent_geometries, self.decoder_blocks):
            query_emb = block(
                query_emb,
                e_ca,
                params,
                queries_pos=query_pos,
                latent_geometry_pos=latent_geo_pos,
                latent_geometry_log_density=latent_geo_log_density,
            )
        return query_emb

    def decode(self, intermediate_latent_geometries, latent_geo_pos, params, query_pos, latent_geo_log_density=None):
        query_emb = self.decode_features(
            intermediate_latent_geometries,
            latent_geo_pos,
            params,
            query_pos,
            latent_geo_log_density=latent_geo_log_density,
        )
        return self.mlp(query_emb)

    def forward(self, geo, surf_query_pos, vol_query_pos, params, geo_log_density=None):
        intermediate_latent_geometries, latent_geo_pos, latent_geo_log_density = self.encode(
            geo,
            params,
            geo_log_density=geo_log_density,
            return_latent_density=True,
        )
        query_pos = torch.cat([surf_query_pos, vol_query_pos], dim=1)
        pred = self.decode(
            intermediate_latent_geometries,
            latent_geo_pos,
            params,
            query_pos,
            latent_geo_log_density=latent_geo_log_density,
        )
        pred_surf = pred[:, :surf_query_pos.shape[1], 0:self.surface_channels]
        pred_vol = pred[:, surf_query_pos.shape[1]:, self.surface_channels:]
        return pred_surf, pred_vol

    @torch.inference_mode()
    def inference(self, geo, surf_query_pos, vol_query_pos, params, geo_log_density=None):
        intermediate_latent_geometries, latent_geo_pos, latent_geo_log_density = self.encode(
            geo,
            params,
            geo_log_density=geo_log_density,
            return_latent_density=True,
        )

        n_surf = surf_query_pos.shape[1]
        y_hat_surf_subregions = []
        for i in range(0, n_surf, self.subregion_size):
            surf_subregion = surf_query_pos[:, i:i + self.subregion_size, :]
            y_surf_subregion = self.decode(
                intermediate_latent_geometries,
                latent_geo_pos,
                params,
                surf_subregion,
                latent_geo_log_density=latent_geo_log_density,
            )
            y_hat_surf_subregions.append(y_surf_subregion)
        y_hat_surf = torch.cat(y_hat_surf_subregions, dim=1)

        n_vol = vol_query_pos.shape[1]
        y_hat_vol_subregions = []
        for i in range(0, n_vol, self.subregion_size):
            vol_subregion = vol_query_pos[:, i:i + self.subregion_size, :]
            y_vol_subregion = self.decode(
                intermediate_latent_geometries,
                latent_geo_pos,
                params,
                vol_subregion,
                latent_geo_log_density=latent_geo_log_density,
            )
            y_hat_vol_subregions.append(y_vol_subregion)
        y_hat_vol = torch.cat(y_hat_vol_subregions, dim=1)

        pred_surf = y_hat_surf[:, :, 0:self.surface_channels]
        pred_vol = y_hat_vol[:, :, self.surface_channels:]
        return pred_surf, pred_vol
