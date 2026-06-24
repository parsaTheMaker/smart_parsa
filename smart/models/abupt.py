from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .family_common import (
    CondInjection,
    ModulatedPositionalEmbedding,
    SelfAttentionBlock,
    gather_tokens,
    init_linear_layer_weights,
    knn_group,
    resolve_geo_log_density,
    sample_indices,
    sample_tokens,
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

    def _attend(self, q, k, v, pos_q=None, pos_k=None, attn_bias=None):
        if pos_q is not None:
            q = self.rope(q, pos_q)
        if pos_k is not None:
            k = self.rope(k, pos_k)
        out = F.scaled_dot_product_attention(
            q.float(),
            k.float(),
            v.float(),
            attn_mask=attn_bias,
            dropout_p=(self.dropout if self.training else 0.0),
        )
        return out

    def forward(self, x, pos=None, attn_bias=None, split_sizes=None, mode=None, query_chunk_size=None):
        x_in = self.norm(x)
        qkv = self.qkv(x_in)
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        q = q.view(x.shape[0], x.shape[1], self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(x.shape[0], x.shape[1], self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(x.shape[0], x.shape[1], self.num_heads, self.head_dim).transpose(1, 2)
        if split_sizes is not None and len(split_sizes) == 2 and mode in {"within", "cross"}:
            surface_len = int(split_sizes[0])
            volume_len = int(split_sizes[1])
            q_surface, q_volume = q[:, :, :surface_len], q[:, :, surface_len:]
            k_surface, k_volume = k[:, :, :surface_len], k[:, :, surface_len:]
            v_surface, v_volume = v[:, :, :surface_len], v[:, :, surface_len:]
            if pos is None:
                pos_surface = None
                pos_volume = None
            else:
                pos_surface = pos[:, :surface_len]
                pos_volume = pos[:, surface_len:]
            chunk_size = int(query_chunk_size) if query_chunk_size is not None else 0

            def chunked_attend(q_group, k_group, v_group, pos_q_group, pos_k_group):
                if chunk_size <= 0 or q_group.shape[2] <= chunk_size:
                    return self._attend(q_group, k_group, v_group, pos_q=pos_q_group, pos_k=pos_k_group)
                pieces = []
                for start in range(0, q_group.shape[2], chunk_size):
                    stop = min(start + chunk_size, q_group.shape[2])
                    q_chunk = q_group[:, :, start:stop]
                    pos_q_chunk = None if pos_q_group is None else pos_q_group[:, start:stop]
                    pieces.append(
                        self._attend(q_chunk, k_group, v_group, pos_q=pos_q_chunk, pos_k=pos_k_group)
                    )
                return torch.cat(pieces, dim=2)

            if mode == "within":
                out_surface = chunked_attend(q_surface, k_surface, v_surface, pos_surface, pos_surface)
                out_volume = chunked_attend(q_volume, k_volume, v_volume, pos_volume, pos_volume)
            else:
                out_surface = chunked_attend(q_surface, k_volume, v_volume, pos_surface, pos_volume)
                out_volume = chunked_attend(q_volume, k_surface, v_surface, pos_volume, pos_surface)
            out = torch.cat([out_surface, out_volume], dim=2)
        else:
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

    def forward(self, x, params=None, pos=None, split_sizes=None, query_chunk_size=None):
        if split_sizes is None or len(split_sizes) != 2:
            raise ValueError("ABUPTSharedTransformerBlock expects split_sizes=[surface_len, volume_len].")
        surface_len, volume_len = int(split_sizes[0]), int(split_sizes[1])
        x = x + self.dropout(
            self.attn(
                x,
                pos=pos,
                split_sizes=[surface_len, volume_len],
                mode=self.mode,
                query_chunk_size=query_chunk_size,
            )
        )
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

    def forward(
        self,
        geometry_position,
        geometry_supernode_position=None,
        geometry_supernode_idx=None,
        params=None,
        geo_log_density=None,
    ):
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
        if geometry_supernode_idx is not None:
            super_pos = torch.gather(geo_pos, 1, geometry_supernode_idx.unsqueeze(-1).expand(-1, -1, geo_pos.shape[-1]))
        elif geometry_supernode_position is not None:
            super_pos = geometry_supernode_position
        else:
            super_pos, _ = sample_tokens(geo_pos, self.num_supernodes)

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
        geometry_depth=1,
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

        self.geometry_encoder = ABUPTGeometryEncoder(
            dim=latent_dim,
            num_supernodes=latent_geometry_points,
            supernode_group_k=supernode_group_k,
            depth=max(1, int(geometry_depth)),
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
            [
                AnchorDecoderBlock(dim=latent_dim, num_heads=num_heads, dropout=dropout, spatial_dim=spatial_dim, cond_dim=parameter_channels)
                for _ in range(int(num_encoder_decoder_blocks))
            ]
        )
        self.volume_blocks = nn.ModuleList(
            [
                AnchorDecoderBlock(dim=latent_dim, num_heads=num_heads, dropout=dropout, spatial_dim=spatial_dim, cond_dim=parameter_channels)
                for _ in range(int(num_encoder_decoder_blocks))
            ]
        )
        self.surface_decoder = nn.Linear(latent_dim, surface_channels)
        self.volume_decoder = nn.Linear(latent_dim, volume_channels)
        self.apply(init_linear_layer_weights)

    def _split_anchor_and_query(self, positions, num_anchor):
        batch_size, n_points, channels = positions.shape
        if n_points == 0:
            empty_idx = torch.empty((batch_size, 0), device=positions.device, dtype=torch.long)
            empty_pos = positions[:, :0, :]
            return empty_pos, empty_pos, empty_idx, empty_idx

        anchor_idx = sample_indices(n_points, min(max(int(num_anchor), 0), n_points), positions.device, batch_size)
        if anchor_idx.shape[1] == n_points:
            empty_idx = torch.empty((batch_size, 0), device=positions.device, dtype=torch.long)
            return gather_tokens(positions, anchor_idx), positions[:, :0, :], anchor_idx, empty_idx

        all_idx = torch.arange(n_points, device=positions.device, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)
        query_idx_rows = []
        for b in range(batch_size):
            mask = torch.ones(n_points, device=positions.device, dtype=torch.bool)
            mask[anchor_idx[b]] = False
            query_idx_rows.append(all_idx[b][mask])
        query_idx = torch.stack(query_idx_rows, dim=0)
        return gather_tokens(positions, anchor_idx), gather_tokens(positions, query_idx), anchor_idx, query_idx

    def prepare_contract_inputs(self, geo, surf_query_pos, vol_query_pos, geo_log_density=None):
        geo_pos = geo * self.pos_scale_factor
        geometry_supernode_pos, geometry_supernode_idx = sample_tokens(geo_pos, self.geometry_encoder.num_supernodes)
        surface_all_position = surf_query_pos * self.pos_scale_factor
        volume_all_position = vol_query_pos * self.pos_scale_factor
        surface_anchor_position, surface_query_position, surface_anchor_idx, surface_query_idx = self._split_anchor_and_query(
            surface_all_position,
            self.anchor_points,
        )
        volume_anchor_position, volume_query_position, volume_anchor_idx, volume_query_idx = self._split_anchor_and_query(
            volume_all_position,
            self.anchor_points,
        )
        return {
            "geometry_position": geo_pos,
            "geometry_supernode_idx": geometry_supernode_idx,
            "geometry_supernode_position": geometry_supernode_pos,
            "surface_anchor_position": surface_anchor_position,
            "volume_anchor_position": volume_anchor_position,
            "surface_query_position": surface_query_position,
            "volume_query_position": volume_query_position,
            "surface_anchor_idx": surface_anchor_idx,
            "surface_query_idx": surface_query_idx,
            "surface_total_points": surf_query_pos.shape[1],
            "volume_anchor_idx": volume_anchor_idx,
            "volume_query_idx": volume_query_idx,
            "volume_total_points": vol_query_pos.shape[1],
        }

    def encode_geometry(
        self,
        geometry_position,
        geometry_supernode_position=None,
        geometry_supernode_idx=None,
        params=None,
        geo_log_density=None,
    ):
        return self.geometry_encoder(
            geometry_position,
            geometry_supernode_position=geometry_supernode_position,
            geometry_supernode_idx=geometry_supernode_idx,
            params=params,
            geo_log_density=geo_log_density,
        )

    def _encode_surface_volume_tokens(self, surface_anchor_position, volume_anchor_position, surface_query_position, volume_query_position, params=None):
        surface_position_all = torch.cat([surface_anchor_position, surface_query_position], dim=1)
        volume_position_all = torch.cat([volume_anchor_position, volume_query_position], dim=1)

        x_surface = self.surface_bias(self.pos_embed(surface_position_all))
        x_volume = self.volume_bias(self.pos_embed(volume_position_all))
        x_surface = self.query_cond(x_surface, params)
        x_volume = self.query_cond(x_volume, params)
        return x_surface, x_volume, surface_position_all, volume_position_all

    def _restore_full_predictions(self, pred_all, anchor_idx, query_idx, total_points):
        batch_size = pred_all.shape[0]
        channels = pred_all.shape[-1]
        restored = pred_all.new_empty((batch_size, int(total_points), channels))
        num_anchor = anchor_idx.shape[1]
        if num_anchor > 0:
            restored.scatter_(
                1,
                anchor_idx.unsqueeze(-1).expand(-1, -1, channels),
                pred_all[:, :num_anchor],
            )
        if query_idx.shape[1] > 0:
            restored.scatter_(
                1,
                query_idx.unsqueeze(-1).expand(-1, -1, channels),
                pred_all[:, num_anchor:],
            )
        return restored

    def _shared_forward(self, x_surface, x_volume, surface_position_all, volume_position_all, geometry_encoding, geometry_pos, params=None):
        surface_len = x_surface.shape[1]
        volume_len = x_volume.shape[1]
        x = torch.cat([x_surface, x_volume], dim=1)
        pos = torch.cat([surface_position_all, volume_position_all], dim=1)
        for block in self.blocks:
            if isinstance(block, ABUPTSharedPerceiverBlock):
                x = block(x, geometry_encoding, params=params, x_pos=pos, geometry_pos=geometry_pos)
            else:
                x = block(
                    x,
                    params=params,
                    pos=pos,
                    split_sizes=[surface_len, volume_len],
                    query_chunk_size=self.subregion_size,
                )
        x_surface = x[:, :surface_len]
        x_volume = x[:, surface_len:]
        return x_surface, x_volume

    def forward(self, geo, surf_query_pos, vol_query_pos, params=None, geo_log_density=None):
        prepared = self.prepare_contract_inputs(geo, surf_query_pos, vol_query_pos, geo_log_density=geo_log_density)
        geometry_encoding, geometry_pos = self.encode_geometry(
            prepared["geometry_position"],
            geometry_supernode_position=prepared["geometry_supernode_position"],
            geometry_supernode_idx=prepared["geometry_supernode_idx"],
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
        for block in self.surface_blocks:
            surface_anchor_tokens = x_surface[:, :num_surface_anchor]
            x_surface = block(x_surface, surface_anchor_tokens, params=params, x_pos=surface_position_all, anchor_pos=prepared["surface_anchor_position"])
        for block in self.volume_blocks:
            volume_anchor_tokens = x_volume[:, :num_volume_anchor]
            x_volume = block(x_volume, volume_anchor_tokens, params=params, x_pos=volume_position_all, anchor_pos=prepared["volume_anchor_position"])

        pred_surface_all = self.surface_decoder(x_surface)
        pred_volume_all = self.volume_decoder(x_volume)
        pred_surf = self._restore_full_predictions(
            pred_surface_all,
            prepared["surface_anchor_idx"],
            prepared["surface_query_idx"],
            prepared["surface_total_points"],
        )
        pred_vol = self._restore_full_predictions(
            pred_volume_all,
            prepared["volume_anchor_idx"],
            prepared["volume_query_idx"],
            prepared["volume_total_points"],
        )
        return pred_surf, pred_vol

    @torch.inference_mode()
    def inference(self, geo, surf_query_pos, vol_query_pos, params=None, geo_log_density=None):
        # Keep the exact same anchor/query interaction path as training.
        return self.forward(geo, surf_query_pos, vol_query_pos, params=params, geo_log_density=geo_log_density)


class ABUPT(ABUPTBase):
    def __init__(self, **kwargs):
        super().__init__(density_compensated=False, **kwargs)
