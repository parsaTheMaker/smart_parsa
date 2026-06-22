"""SMART-SAT3: sampled-anchor variant of SAT2 with exact inverse-density anchors.

SAT3 keeps SAT2's measure-corrected point-to-latent aggregation and paired-view
training setup, but replaces fixed canonical anchors with anchors sampled from
the input geometry using inverse-density probabilities.

Mathematically:
1. geometry-to-latent cross-attention uses inverse-density quadrature weights;
2. latent anchors are sampled with proposal q(x) propto 1 / rho(x);
3. since q(x) cancels the sampling density, the retained anchor set is
   approximately uniform with respect to the underlying surface measure, so no
   further latent-density reuse correction is needed.

Because sampled anchors form an unordered set, SAT3 uses a shared learned anchor
token rather than slot-specific latent tokens.
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


class MeasureCorrectedCrossAttention(nn.Module):
    """Cross-attention with density-aware quadrature correction in softmax space."""

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
            attn_bias = (-kv_log_density)[:, None, None, :].to(dtype=q_heads.dtype)

        out = F.scaled_dot_product_attention(
            q_heads,
            k_heads,
            v_heads,
            attn_mask=attn_bias,
            dropout_p=0.0,
        )
        out = rearrange(out, "b h q d -> b q (h d)")
        return self.out_proj(out)


class SampledLatentEncoderBlock(nn.Module):
    """Encoder block with density-aware input aggregation and latent reuse."""

    def __init__(self, dim, num_heads=8, dropout=0.1, spatial_dim=3, cond_dim=2):
        super().__init__()
        self.input_attn = MeasureCorrectedCrossAttention(dim=dim, num_heads=num_heads, spatial_dim=spatial_dim)
        self.latent_attn = MeasureCorrectedCrossAttention(dim=dim, num_heads=num_heads, spatial_dim=spatial_dim)
        self.attn_dropout = nn.Dropout(dropout)

        if cond_dim > 0:
            self.mlp = SimulationParamModulatedMLP(dim=dim, hidden_dim=dim * 4, cond_dim=cond_dim, dropout=dropout)
        else:
            self.mlp = PlainMLP(dim=dim, hidden_dim=dim * 4, dropout=dropout)

    def forward(
        self,
        latent,
        geometry_tokens,
        params,
        latent_pos=None,
        geometry_pos=None,
        geometry_log_density=None,
        latent_log_density=None,
    ):
        latent = latent + self.attn_dropout(
            self.input_attn(
                q=latent,
                kv=geometry_tokens,
                q_pos=latent_pos,
                kv_pos=geometry_pos,
                kv_log_density=geometry_log_density,
            )
        )
        latent = latent + self.attn_dropout(
            self.latent_attn(
                q=latent,
                kv=latent,
                q_pos=latent_pos,
                kv_pos=latent_pos,
                kv_log_density=latent_log_density,
            )
        )
        latent = latent + self.mlp(latent, params)
        return latent


class DensityAwareDecoderBlock(nn.Module):
    """Decoder block with proposal-matched latent density correction."""

    def __init__(self, dim, num_heads=8, dropout=0.1, spatial_dim=3, cond_dim=2):
        super().__init__()
        self.attn = MeasureCorrectedCrossAttention(dim=dim, num_heads=num_heads, spatial_dim=spatial_dim)
        self.attn_dropout = nn.Dropout(dropout)

        if cond_dim > 0:
            self.mlp = SimulationParamModulatedMLP(dim=dim, hidden_dim=dim * 4, cond_dim=cond_dim, dropout=dropout)
        else:
            self.mlp = PlainMLP(dim=dim, hidden_dim=dim * 4, dropout=dropout)

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


def gather_geometry(geometry, idx):
    return torch.gather(geometry, 1, idx.unsqueeze(-1).expand(-1, -1, geometry.shape[-1]))


def sample_geometry_subset(geometry, num_samples, with_replacement=False):
    n_points = int(geometry.shape[1])
    batch_size = int(geometry.shape[0])
    if num_samples <= 0 or num_samples >= n_points:
        idx = torch.arange(n_points, device=geometry.device, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)
        return geometry, idx
    if with_replacement:
        idx = torch.randint(0, n_points, (batch_size, num_samples), device=geometry.device, dtype=torch.long)
    else:
        idx = torch.stack([torch.randperm(n_points, device=geometry.device)[:num_samples] for _ in range(batch_size)], dim=0)
    return gather_geometry(geometry, idx), idx


def sample_geometry_inverse_density(geometry, num_samples, log_density):
    """Sample anchors with probabilities proportional to 1 / rho(x)."""

    n_points = int(geometry.shape[1])
    batch_size = int(geometry.shape[0])
    if num_samples <= 0 or num_samples >= n_points:
        idx = torch.arange(n_points, device=geometry.device, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)
        return geometry, idx

    weights = torch.exp(-log_density.float())
    weights = torch.clamp(weights, min=1e-12)
    probs = weights / torch.clamp(weights.sum(dim=1, keepdim=True), min=1e-12)
    idx = torch.multinomial(probs, num_samples=num_samples, replacement=False)
    return gather_geometry(geometry, idx), idx


class SMARTSAT3(nn.Module):
    """SAT2-style model with sampled inverse-density anchors."""

    def __init__(
        self,
        spatial_dim=3,
        surface_channels=1,
        volume_channels=3,
        parameter_channels=2,
        latent_dim=256,
        latent_geometry_points=4096,
        subsampled_geometry_points=16384,
        subsampled_geometry_with_replacement=False,
        num_encoder_decoder_blocks=8,
        num_heads=8,
        pos_scale_factor=1000,
        dropout=0.0,
        subregion_size=262144,
        density_knn_k=24,
        density_neighbor_hops=1,
        density_estimator="tangent_cov",
    ):
        super().__init__()
        assert surface_channels > 0 and volume_channels > 0, "surface_channels and volume_channels must be positive integers."

        self.surface_channels = surface_channels
        self.volume_channels = volume_channels
        self.spatial_dim = spatial_dim
        self.num_geo = latent_geometry_points
        self.subsampled_geometry_points = subsampled_geometry_points
        self.subsampled_geometry_with_replacement = bool(subsampled_geometry_with_replacement)
        self.pos_scale_factor = pos_scale_factor
        self.subregion_size = subregion_size
        self.density_knn_k = int(density_knn_k)
        self.density_neighbor_hops = int(density_neighbor_hops)
        self.density_estimator = str(density_estimator)

        self.pos_encoder = ModulatedPositionalEmbedding(latent_dim, spatial_dim)
        # Shared anchor token keeps the sampled-anchor set permutation equivariant.
        self.anchor_token = nn.Parameter(torch.zeros(1, 1, latent_dim))

        self.encoder_blocks = nn.ModuleList(
            [
                SampledLatentEncoderBlock(
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
                DensityAwareDecoderBlock(
                    dim=latent_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    spatial_dim=spatial_dim,
                    cond_dim=parameter_channels,
                )
                for _ in range(num_encoder_decoder_blocks)
            ]
        )
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, surface_channels + volume_channels),
        )

    def initialize_weights(self):
        self.apply(self._init_weights)
        nn.init.trunc_normal_(self.anchor_token, std=0.02)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def encode(self, geo, params, geo_log_density=None, return_final=False, return_latent_density=False):
        if geo_log_density is None:
            geo_log_density = estimate_log_sampling_density(
                geo,
                knn_k=self.density_knn_k,
                neighbor_hops=self.density_neighbor_hops,
                estimator=self.density_estimator,
            )
        geo_log_density = geo_log_density.to(device=geo.device, dtype=geo.dtype)

        geo_pos = geo * self.pos_scale_factor
        latent_pos, latent_idx = sample_geometry_inverse_density(
            geo_pos,
            self.num_geo,
            log_density=geo_log_density,
        )
        latent_reuse_log_density = None

        latent = self.anchor_token.expand(geo.shape[0], latent_pos.shape[1], -1) + self.pos_encoder(latent_pos)

        intermediate_latents = []
        for block in self.encoder_blocks:
            geo_subset_pos, geo_subset_idx = sample_geometry_subset(
                geo_pos,
                self.subsampled_geometry_points,
                with_replacement=self.subsampled_geometry_with_replacement,
            )
            geo_subset_emb = self.pos_encoder(geo_subset_pos)
            geo_subset_log_density = torch.gather(geo_log_density, 1, geo_subset_idx)
            latent = block(
                latent,
                geo_subset_emb,
                params,
                latent_pos=latent_pos,
                geometry_pos=geo_subset_pos,
                geometry_log_density=geo_subset_log_density,
                latent_log_density=latent_reuse_log_density,
            )
            intermediate_latents.append(latent)

        if return_final:
            if return_latent_density:
                return intermediate_latents, latent_pos, latent, latent_reuse_log_density
            return intermediate_latents, latent_pos, latent
        if return_latent_density:
            return intermediate_latents, latent_pos, latent_reuse_log_density
        return intermediate_latents, latent_pos

    def decode_features(self, intermediate_latents, latent_pos, params, query_pos, latent_log_density=None):
        query_pos = query_pos * self.pos_scale_factor
        query_emb = self.pos_encoder(query_pos)
        for latent, block in zip(intermediate_latents, self.decoder_blocks):
            query_emb = block(
                query_emb,
                latent,
                params,
                queries_pos=query_pos,
                latent_geometry_pos=latent_pos,
                latent_geometry_log_density=latent_log_density,
            )
        return query_emb

    def decode(self, intermediate_latents, latent_pos, params, query_pos, latent_log_density=None):
        return self.mlp(self.decode_features(intermediate_latents, latent_pos, params, query_pos, latent_log_density=latent_log_density))

    def forward(self, geo, surf_query_pos, vol_query_pos, params, geo_log_density=None, return_latent=False):
        intermediate_latents, latent_pos, final_latent, latent_log_density = self.encode(
            geo,
            params,
            geo_log_density=geo_log_density,
            return_final=True,
            return_latent_density=True,
        )
        query_pos = torch.cat([surf_query_pos, vol_query_pos], dim=1)
        pred = self.decode(
            intermediate_latents,
            latent_pos,
            params,
            query_pos,
            latent_log_density=latent_log_density,
        )
        pred_surf = pred[:, :surf_query_pos.shape[1], : self.surface_channels]
        pred_vol = pred[:, surf_query_pos.shape[1] :, self.surface_channels :]
        if return_latent:
            return pred_surf, pred_vol, final_latent
        return pred_surf, pred_vol

    @torch.inference_mode()
    def inference(self, geo, surf_query_pos, vol_query_pos, params, geo_log_density=None):
        intermediate_latents, latent_pos, latent_log_density = self.encode(
            geo,
            params,
            geo_log_density=geo_log_density,
            return_final=False,
            return_latent_density=True,
        )

        n_surf = surf_query_pos.shape[1]
        y_hat_surf_subregions = []
        for i in range(0, n_surf, self.subregion_size):
            surf_subregion = surf_query_pos[:, i : i + self.subregion_size, :]
            y_surf_subregion = self.decode(
                intermediate_latents,
                latent_pos,
                params,
                surf_subregion,
                latent_log_density=latent_log_density,
            )
            y_hat_surf_subregions.append(y_surf_subregion)
        y_hat_surf = torch.cat(y_hat_surf_subregions, dim=1)

        n_vol = vol_query_pos.shape[1]
        y_hat_vol_subregions = []
        for i in range(0, n_vol, self.subregion_size):
            vol_subregion = vol_query_pos[:, i : i + self.subregion_size, :]
            y_vol_subregion = self.decode(
                intermediate_latents,
                latent_pos,
                params,
                vol_subregion,
                latent_log_density=latent_log_density,
            )
            y_hat_vol_subregions.append(y_vol_subregion)
        y_hat_vol = torch.cat(y_hat_vol_subregions, dim=1)

        pred_surf = y_hat_surf[:, :, : self.surface_channels]
        pred_vol = y_hat_vol[:, :, self.surface_channels :]
        return pred_surf, pred_vol
