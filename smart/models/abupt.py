from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .family_common import (
    CondInjection,
    ModulatedPositionalEmbedding,
    QueryTypeEmbedding,
    SelfAttentionBlock,
    init_linear_layer_weights,
    knn_group,
    resolve_geo_log_density,
    sample_tokens_fps,
    sample_tokens_density_compensated_fps,
)
from .smart.smart import CrossAttention, PlainMLP, RotaryPositionalEmbedding, SimulationParamModulatedMLP


class SupernodePoolingPosOnly(nn.Module):
    """Approximate version of AB-UPT's position-only supernode pooling."""

    def __init__(self, dim, spatial_dim=3, cond_dim=0):
        super().__init__()
        self.rel_mlp = nn.Sequential(
            nn.Linear(spatial_dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.GELU(),
        )
        self.post = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.cond = CondInjection(dim, cond_dim)

    def forward(self, geometry_pos, supernode_pos, params=None, group_k=32, geo_log_density=None):
        neighbors, neighbor_idx = knn_group(geometry_pos, supernode_pos, k=group_k)
        rel = neighbors - supernode_pos.unsqueeze(2)
        feat = self.rel_mlp(rel)
        if geo_log_density is not None:
            neighbor_log_density = torch.gather(
                geo_log_density.unsqueeze(1).expand(-1, supernode_pos.shape[1], -1),
                2,
                neighbor_idx,
            )
            weights = torch.exp(-neighbor_log_density.float())
            weights = weights / torch.clamp(weights.sum(dim=2, keepdim=True), min=1e-6)
            weights = weights.to(dtype=feat.dtype)
        else:
            weights = torch.full(
                feat.shape[:3],
                1.0 / float(max(feat.shape[2], 1)),
                device=feat.device,
                dtype=feat.dtype,
            )
        feat_mean = (feat * weights.unsqueeze(-1)).sum(dim=2)
        feat_second = (feat.pow(2) * weights.unsqueeze(-1)).sum(dim=2)
        feat_scale = torch.sqrt(torch.clamp(feat_second - feat_mean.pow(2), min=1e-6))
        pooled = self.post(torch.cat([feat_mean, feat_scale], dim=-1))
        return self.cond(pooled, params)


class AnchorDecoderBlock(nn.Module):
    """Anchor-centric decoder update: all tokens attend to anchors only."""

    def __init__(self, dim, num_heads=8, dropout=0.0, spatial_dim=3, cond_dim=0):
        super().__init__()
        self.cross = CrossAttention(dim=dim, num_heads=num_heads, dropout=dropout, spatial_dim=spatial_dim)
        self.dropout = nn.Dropout(dropout)
        self.mlp = (
            SimulationParamModulatedMLP(dim=dim, hidden_dim=dim * 4, cond_dim=cond_dim, dropout=dropout)
            if cond_dim > 0
            else PlainMLP(dim=dim, hidden_dim=dim * 4, dropout=dropout)
        )

    def forward(self, x, anchor_tokens, params=None, x_pos=None, anchor_pos=None):
        x = x + self.dropout(self.cross(q=x, kv=anchor_tokens, q_pos=x_pos, kv_pos=anchor_pos))
        x = x + self.mlp(x, params)
        return x


class SplitAwareSharedAttention(nn.Module):
    """Shared self-attention over a concatenated surface+volume sequence."""

    def __init__(self, dim, num_heads=8, dropout=0.0, spatial_dim=3):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.qkv = nn.Linear(dim, dim * 3)
        self.out_proj = nn.Linear(dim, dim)
        self.rope = RotaryPositionalEmbedding(dim=dim // num_heads, spatial_dim=spatial_dim)
        self.dropout = float(dropout)

    def forward(self, x, pos=None, attn_bias=None):
        x_in = self.norm(x)
        qkv = self.qkv(x_in)
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        q = q.view(x.shape[0], x.shape[1], self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(x.shape[0], x.shape[1], self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(x.shape[0], x.shape[1], self.num_heads, self.head_dim).transpose(1, 2)
        if pos is not None:
            q = self.rope(q, pos)
            k = self.rope(k, pos)
        out = F.scaled_dot_product_attention(
            q.float(),
            k.float(),
            v.float(),
            attn_mask=attn_bias,
            dropout_p=(self.dropout if self.training else 0.0),
        )
        out = out.transpose(1, 2).contiguous().view(x.shape[0], x.shape[1], -1).to(dtype=x.dtype)
        return self.out_proj(out)


class ABUPTSharedTransformerBlock(nn.Module):
    """AB-UPT-style split-aware shared transformer block on a single sequence."""

    def __init__(self, dim, num_heads=8, dropout=0.0, spatial_dim=3, cond_dim=0, mode="within"):
        super().__init__()
        self.mode = str(mode)
        self.attn = SplitAwareSharedAttention(dim=dim, num_heads=num_heads, dropout=dropout, spatial_dim=spatial_dim)
        self.dropout = nn.Dropout(dropout)
        self.mlp = (
            SimulationParamModulatedMLP(dim=dim, hidden_dim=dim * 4, cond_dim=cond_dim, dropout=dropout)
            if cond_dim > 0
            else PlainMLP(dim=dim, hidden_dim=dim * 4, dropout=dropout)
        )

    def _build_attn_bias(self, surface_len, volume_len, device):
        total_len = int(surface_len + volume_len)
        neg_inf = torch.finfo(torch.float32).min
        bias = torch.full((1, 1, total_len, total_len), neg_inf, device=device, dtype=torch.float32)
        if self.mode == "within":
            bias[:, :, :surface_len, :surface_len] = 0.0
            bias[:, :, surface_len:, surface_len:] = 0.0
        elif self.mode == "cross":
            bias[:, :, :surface_len, surface_len:] = 0.0
            bias[:, :, surface_len:, :surface_len] = 0.0
        else:
            raise ValueError(f"Unknown AB-UPT shared block mode '{self.mode}'")
        return bias

    def forward(self, x, params=None, pos=None, split_sizes=None):
        if split_sizes is None or len(split_sizes) != 2:
            raise ValueError("ABUPTSharedTransformerBlock expects split_sizes=[surface_len, volume_len].")
        surface_len, volume_len = int(split_sizes[0]), int(split_sizes[1])
        attn_bias = self._build_attn_bias(surface_len, volume_len, x.device)
        x = x + self.dropout(self.attn(x, pos=pos, attn_bias=attn_bias))
        x = x + self.mlp(x, params)
        return x


class ABUPTSharedPerceiverBlock(nn.Module):
    """Shared geometry cross-attention on the concatenated surface+volume sequence."""

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


class ABUPTGeometryEncoder(nn.Module):
    def __init__(
        self,
        dim,
        num_supernodes,
        supernode_group_k,
        depth,
        num_heads,
        dropout,
        spatial_dim,
        cond_dim,
        pos_scale_factor,
        density_compensated=False,
        density_knn_k=8,
        density_neighbor_hops=1,
        density_estimator="rk2",
    ):
        super().__init__()
        self.num_supernodes = num_supernodes
        self.supernode_group_k = int(supernode_group_k)
        self.pos_scale_factor = pos_scale_factor
        self.density_compensated = bool(density_compensated)
        self.density_knn_k = int(density_knn_k)
        self.density_neighbor_hops = int(density_neighbor_hops)
        self.density_estimator = str(density_estimator)

        self.pos_encoder = ModulatedPositionalEmbedding(dim, spatial_dim)
        self.pool = SupernodePoolingPosOnly(dim=dim, spatial_dim=spatial_dim, cond_dim=cond_dim)
        self.blocks = nn.ModuleList(
            [
                SelfAttentionBlock(
                    dim=dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    spatial_dim=spatial_dim,
                    cond_dim=cond_dim,
                )
                for _ in range(depth)
            ]
        )

    def forward(self, geometry_position, geometry_supernode_position=None, params=None, geo_log_density=None):
        full_geo_log_density = None
        if self.density_compensated:
            full_geo_log_density = resolve_geo_log_density(
                geometry_position / self.pos_scale_factor,
                geo_log_density,
                knn_k=self.density_knn_k,
                neighbor_hops=self.density_neighbor_hops,
                estimator=self.density_estimator,
            )

        geo_pos = geometry_position
        if geometry_supernode_position is not None:
            super_pos = geometry_supernode_position
        elif self.density_compensated and full_geo_log_density is not None:
            super_pos, _ = sample_tokens_density_compensated_fps(geo_pos, self.num_supernodes, full_geo_log_density)
        else:
            super_pos, _ = sample_tokens_fps(geo_pos, self.num_supernodes, random_start=False)

        super_tokens = self.pos_encoder(super_pos)
        super_tokens = super_tokens + self.pool(
            geo_pos,
            super_pos,
            params=params,
            group_k=self.supernode_group_k,
            geo_log_density=full_geo_log_density,
        )
        for block in self.blocks:
            super_tokens = block(super_tokens, params=params, pos=super_pos)
        return super_tokens, super_pos


class ABUPTBase(nn.Module):
    expects_geo_log_density = False

    def __init__(
        self,
        spatial_dim=3,
        surface_channels=1,
        volume_channels=3,
        parameter_channels=0,
        latent_dim=256,
        latent_geometry_points=4096,
        subsampled_geometry_points=65536,  # kept for config compatibility
        anchor_points=2048,
        num_encoder_decoder_blocks=8,
        num_heads=8,
        pos_scale_factor=100,
        dropout=0.0,
        subregion_size=262144,
        block_pattern="pscscs",
        supernode_group_k=32,
        density_compensated=False,
        density_knn_k=8,
        density_neighbor_hops=1,
        density_estimator="rk2",
    ):
        super().__init__()
        self.surface_channels = surface_channels
        self.volume_channels = volume_channels
        self.anchor_points = anchor_points
        self.subregion_size = subregion_size
        self.pos_scale_factor = pos_scale_factor
        self.expects_geo_log_density = bool(density_compensated)

        geom_depth = max(1, num_encoder_decoder_blocks // 2)
        self.geometry_encoder = ABUPTGeometryEncoder(
            dim=latent_dim,
            num_supernodes=latent_geometry_points,
            supernode_group_k=supernode_group_k,
            depth=geom_depth,
            num_heads=num_heads,
            dropout=dropout,
            spatial_dim=spatial_dim,
            cond_dim=parameter_channels,
            pos_scale_factor=pos_scale_factor,
            density_compensated=density_compensated,
            density_knn_k=density_knn_k,
            density_neighbor_hops=density_neighbor_hops,
            density_estimator=density_estimator,
        )
        self.pos_embed = ModulatedPositionalEmbedding(latent_dim, spatial_dim)
        self.surface_bias = nn.Sequential(nn.Linear(latent_dim, latent_dim), nn.GELU(), nn.Linear(latent_dim, latent_dim))
        self.volume_bias = nn.Sequential(nn.Linear(latent_dim, latent_dim), nn.GELU(), nn.Linear(latent_dim, latent_dim))
        self.query_types = QueryTypeEmbedding(latent_dim)
        self.query_cond = CondInjection(latent_dim, parameter_channels)

        self.blocks = nn.ModuleList()
        for symbol in block_pattern:
            if symbol == "p":
                self.blocks.append(
                    ABUPTSharedPerceiverBlock(
                        dim=latent_dim,
                        num_heads=num_heads,
                        dropout=dropout,
                        spatial_dim=spatial_dim,
                        cond_dim=parameter_channels,
                    )
                )
            elif symbol == "s":
                self.blocks.append(
                    ABUPTSharedTransformerBlock(
                        dim=latent_dim,
                        num_heads=num_heads,
                        dropout=dropout,
                        spatial_dim=spatial_dim,
                        cond_dim=parameter_channels,
                        mode="within",
                    )
                )
            elif symbol == "c":
                self.blocks.append(
                    ABUPTSharedTransformerBlock(
                        dim=latent_dim,
                        num_heads=num_heads,
                        dropout=dropout,
                        spatial_dim=spatial_dim,
                        cond_dim=parameter_channels,
                        mode="cross",
                    )
                )
            else:
                raise ValueError(f"Unknown AB-UPT block symbol '{symbol}'")

        self.surface_blocks = nn.ModuleList(
            [AnchorDecoderBlock(dim=latent_dim, num_heads=num_heads, dropout=dropout, spatial_dim=spatial_dim, cond_dim=parameter_channels) for _ in range(2)]
        )
        self.volume_blocks = nn.ModuleList(
            [AnchorDecoderBlock(dim=latent_dim, num_heads=num_heads, dropout=dropout, spatial_dim=spatial_dim, cond_dim=parameter_channels) for _ in range(2)]
        )
        self.surface_decoder = nn.Linear(latent_dim, surface_channels)
        self.volume_decoder = nn.Linear(latent_dim, volume_channels)
        self.apply(init_linear_layer_weights)

    def prepare_contract_inputs(self, geo, surf_query_pos, vol_query_pos, geo_log_density=None):
        geo_pos = geo * self.pos_scale_factor
        if self.expects_geo_log_density and geo_log_density is not None:
            geometry_supernode_pos, geometry_supernode_idx = sample_tokens_density_compensated_fps(
                geo_pos,
                self.geometry_encoder.num_supernodes,
                geo_log_density,
            )
        else:
            geometry_supernode_pos, geometry_supernode_idx = sample_tokens_fps(
                geo_pos,
                self.geometry_encoder.num_supernodes,
                random_start=False,
            )

        surface_anchor_position, _ = sample_tokens_fps(surf_query_pos * self.pos_scale_factor, self.anchor_points, random_start=False)
        volume_anchor_position, _ = sample_tokens_fps(vol_query_pos * self.pos_scale_factor, self.anchor_points, random_start=False)
        return {
            "geometry_position": geo_pos,
            "geometry_supernode_idx": geometry_supernode_idx,
            "geometry_supernode_position": geometry_supernode_pos,
            "surface_anchor_position": surface_anchor_position,
            "volume_anchor_position": volume_anchor_position,
            "surface_query_position": surf_query_pos * self.pos_scale_factor,
            "volume_query_position": vol_query_pos * self.pos_scale_factor,
        }

    def encode_geometry(self, geometry_position, geometry_supernode_position=None, params=None, geo_log_density=None):
        return self.geometry_encoder(
            geometry_position,
            geometry_supernode_position=geometry_supernode_position,
            params=params,
            geo_log_density=geo_log_density,
        )

    def _encode_surface_volume_tokens(self, surface_anchor_position, volume_anchor_position, surface_query_position, volume_query_position, params=None):
        surface_position_all = torch.cat([surface_anchor_position, surface_query_position], dim=1)
        volume_position_all = torch.cat([volume_anchor_position, volume_query_position], dim=1)

        x_surface = self.surface_bias(self.pos_embed(surface_position_all))
        x_volume = self.volume_bias(self.pos_embed(volume_position_all))
        x_surface, x_volume = self.query_types(x_surface, x_volume)
        x_surface = self.query_cond(x_surface, params)
        x_volume = self.query_cond(x_volume, params)
        return x_surface, x_volume, surface_position_all, volume_position_all

    def _shared_forward(self, x_surface, x_volume, surface_position_all, volume_position_all, geometry_encoding, geometry_pos, params=None):
        surface_len = x_surface.shape[1]
        volume_len = x_volume.shape[1]
        x = torch.cat([x_surface, x_volume], dim=1)
        pos = torch.cat([surface_position_all, volume_position_all], dim=1)
        for block in self.blocks:
            if isinstance(block, ABUPTSharedPerceiverBlock):
                x = block(x, geometry_encoding, params=params, x_pos=pos, geometry_pos=geometry_pos)
            else:
                x = block(x, params=params, pos=pos, split_sizes=[surface_len, volume_len])
        x_surface = x[:, :surface_len]
        x_volume = x[:, surface_len:]
        return x_surface, x_volume

    def forward(self, geo, surf_query_pos, vol_query_pos, params=None, geo_log_density=None):
        prepared = self.prepare_contract_inputs(geo, surf_query_pos, vol_query_pos, geo_log_density=geo_log_density)
        geometry_encoding, geometry_pos = self.encode_geometry(
            prepared["geometry_position"],
            geometry_supernode_position=prepared["geometry_supernode_position"],
            params=params,
            geo_log_density=geo_log_density,
        )

        x_surface, x_volume, surface_position_all, volume_position_all = self._encode_surface_volume_tokens(
            prepared["surface_anchor_position"],
            prepared["volume_anchor_position"],
            prepared["surface_query_position"],
            prepared["volume_query_position"],
            params=params,
        )
        x_surface, x_volume = self._shared_forward(
            x_surface,
            x_volume,
            surface_position_all,
            volume_position_all,
            geometry_encoding,
            geometry_pos,
            params=params,
        )

        num_surface_anchor = prepared["surface_anchor_position"].shape[1]
        num_volume_anchor = prepared["volume_anchor_position"].shape[1]
        surface_anchor_tokens = x_surface[:, :num_surface_anchor]
        volume_anchor_tokens = x_volume[:, :num_volume_anchor]
        for block in self.surface_blocks:
            x_surface = block(x_surface, surface_anchor_tokens, params=params, x_pos=surface_position_all, anchor_pos=prepared["surface_anchor_position"])
        for block in self.volume_blocks:
            x_volume = block(x_volume, volume_anchor_tokens, params=params, x_pos=volume_position_all, anchor_pos=prepared["volume_anchor_position"])

        pred_surface_all = self.surface_decoder(x_surface)
        pred_volume_all = self.volume_decoder(x_volume)
        pred_surf = pred_surface_all[:, num_surface_anchor:]
        pred_vol = pred_volume_all[:, num_volume_anchor:]
        return pred_surf, pred_vol

    @torch.inference_mode()
    def inference(self, geo, surf_query_pos, vol_query_pos, params=None, geo_log_density=None):
        # Keep the exact same anchor/query interaction path as training.
        return self.forward(geo, surf_query_pos, vol_query_pos, params=params, geo_log_density=geo_log_density)


class ABUPT(ABUPTBase):
    def __init__(self, **kwargs):
        super().__init__(density_compensated=False, **kwargs)
