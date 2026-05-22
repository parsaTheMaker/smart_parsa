"""Cascaded Aero Transformer (CAT) built as a SMART spin-off.

Stages:
1) Geometry pretraining: surface/volume attribute proxy prediction
2) Surface-field pretraining: surface pressure prediction
3) Volume training: frozen geometry+surface encoders, train fusion+volume decoder
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .smart import ModulatedPositionalEmbedding, CrossAttention, PlainMLP


class LoopEncoder(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        loops: int,
        anchors: int,
        num_heads: int,
        spatial_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.loops = loops
        self.anchors = anchors

        self.pos_encoder = ModulatedPositionalEmbedding(latent_dim, spatial_dim=spatial_dim)
        self.input_attn = nn.ModuleList(
            [CrossAttention(dim=latent_dim, num_heads=num_heads, spatial_dim=spatial_dim, dropout=dropout) for _ in range(loops)]
        )
        self.self_attn = nn.ModuleList(
            [CrossAttention(dim=latent_dim, num_heads=num_heads, spatial_dim=spatial_dim, dropout=dropout) for _ in range(loops)]
        )
        self.ffn = nn.ModuleList([PlainMLP(dim=latent_dim, hidden_dim=latent_dim * 4, dropout=dropout) for _ in range(loops)])

    @staticmethod
    def _chunk_indices(n: int, loops: int, device: torch.device):
        perm = torch.randperm(n, device=device)
        return torch.chunk(perm, loops)

    @staticmethod
    def _sample_anchor_idx(n: int, anchors: int, device: torch.device):
        if anchors <= n:
            return torch.randperm(n, device=device)[:anchors]
        return torch.randint(0, n, (anchors,), device=device)

    def forward(self, surface_points_scaled: torch.Tensor, anchor_idx: torch.Tensor | None = None):
        # surface_points_scaled: [B, N, D]
        bsz, n_pts, _ = surface_points_scaled.shape
        device = surface_points_scaled.device

        if anchor_idx is None:
            anchor_idx = self._sample_anchor_idx(n_pts, self.anchors, device)

        anchor_pos = surface_points_scaled[:, anchor_idx, :]
        latent = self.pos_encoder(anchor_pos)

        chunks = self._chunk_indices(n_pts, self.loops, device)
        intermediate_latents = []
        for m, idx in enumerate(chunks):
            if idx.numel() == 0:
                intermediate_latents.append(latent)
                continue
            chunk_pos = surface_points_scaled[:, idx, :]
            chunk_emb = self.pos_encoder(chunk_pos)

            latent = latent + self.input_attn[m](
                q=latent,
                kv=chunk_emb,
                q_pos=anchor_pos,
                kv_pos=chunk_pos,
            )
            latent = latent + self.self_attn[m](
                q=latent,
                kv=latent,
                q_pos=anchor_pos,
                kv_pos=anchor_pos,
            )
            latent = latent + self.ffn[m](latent, None)
            intermediate_latents.append(latent)

        return intermediate_latents, anchor_pos, anchor_idx


class LoopDecoder(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        loops: int,
        num_heads: int,
        spatial_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.loops = loops
        self.pos_encoder = ModulatedPositionalEmbedding(latent_dim, spatial_dim=spatial_dim)
        self.cross_attn = nn.ModuleList(
            [CrossAttention(dim=latent_dim, num_heads=num_heads, spatial_dim=spatial_dim, dropout=dropout) for _ in range(loops)]
        )
        self.ffn = nn.ModuleList([PlainMLP(dim=latent_dim, hidden_dim=latent_dim * 4, dropout=dropout) for _ in range(loops)])

    def forward(self, query_pos_scaled: torch.Tensor, latent_list: list[torch.Tensor], anchor_pos_scaled: torch.Tensor):
        query = self.pos_encoder(query_pos_scaled)
        for m in range(self.loops):
            lat = latent_list[m if m < len(latent_list) else -1]
            query = query + self.cross_attn[m](
                q=query,
                kv=lat,
                q_pos=query_pos_scaled,
                kv_pos=anchor_pos_scaled,
            )
            query = query + self.ffn[m](query, None)
        return query


class SharedFusion(nn.Module):
    """Shared self-attention fusion block used at each CAT loop in Stage 3."""

    def __init__(self, latent_dim: int, num_heads: int, spatial_dim: int, dropout: float = 0.0):
        super().__init__()
        self.fuse_proj = nn.Linear(2 * latent_dim, latent_dim)
        self.shared_attn = CrossAttention(dim=latent_dim, num_heads=num_heads, spatial_dim=spatial_dim, dropout=dropout)
        self.shared_ffn = PlainMLP(dim=latent_dim, hidden_dim=latent_dim * 4, dropout=dropout)

    def forward(self, geom_latents: list[torch.Tensor], surf_latents: list[torch.Tensor], anchor_pos_scaled: torch.Tensor):
        fused = []
        loops = min(len(geom_latents), len(surf_latents))
        for m in range(loops):
            z = self.fuse_proj(torch.cat([geom_latents[m], surf_latents[m]], dim=-1))
            z = z + self.shared_attn(q=z, kv=z, q_pos=anchor_pos_scaled, kv_pos=anchor_pos_scaled)
            z = z + self.shared_ffn(z, None)
            fused.append(z)
        return fused


class CAT(nn.Module):
    def __init__(
        self,
        spatial_dim=2,
        surface_channels=3,
        volume_channels=4,
        parameter_channels=0,
        latent_dim=128,
        latent_geometry_points=2048,
        subsampled_geometry_points=4096,
        num_encoder_decoder_blocks=6,
        num_heads=8,
        pos_scale_factor=100,
        dropout=0.0,
        subregion_size=262144,
        stage1_surface_attr_channels=2,
        stage1_volume_attr_channels=1,
        stage2_surface_channels=1,
    ):
        super().__init__()
        del parameter_channels, subsampled_geometry_points
        self.surface_channels = surface_channels
        self.volume_channels = volume_channels
        self.stage1_surface_attr_channels = stage1_surface_attr_channels
        self.stage1_volume_attr_channels = stage1_volume_attr_channels
        self.stage2_surface_channels = stage2_surface_channels
        self.loops = num_encoder_decoder_blocks
        self.pos_scale_factor = pos_scale_factor
        self.subregion_size = subregion_size

        self.geometry_encoder = LoopEncoder(
            latent_dim=latent_dim,
            loops=self.loops,
            anchors=latent_geometry_points,
            num_heads=num_heads,
            spatial_dim=spatial_dim,
            dropout=dropout,
        )
        self.surface_encoder = LoopEncoder(
            latent_dim=latent_dim,
            loops=self.loops,
            anchors=latent_geometry_points,
            num_heads=num_heads,
            spatial_dim=spatial_dim,
            dropout=dropout,
        )

        self.stage1_decoder = LoopDecoder(latent_dim=latent_dim, loops=self.loops, num_heads=num_heads, spatial_dim=spatial_dim, dropout=dropout)
        self.stage2_decoder = LoopDecoder(latent_dim=latent_dim, loops=self.loops, num_heads=num_heads, spatial_dim=spatial_dim, dropout=dropout)
        self.stage3_decoder = LoopDecoder(latent_dim=latent_dim, loops=self.loops, num_heads=num_heads, spatial_dim=spatial_dim, dropout=dropout)

        self.fusion = SharedFusion(latent_dim=latent_dim, num_heads=num_heads, spatial_dim=spatial_dim, dropout=dropout)

        stage1_out = stage1_surface_attr_channels + stage1_volume_attr_channels
        self.stage1_head = nn.Sequential(nn.Linear(latent_dim, 128), nn.GELU(), nn.Linear(128, 64), nn.GELU(), nn.Linear(64, stage1_out))
        self.stage2_head = nn.Sequential(nn.Linear(latent_dim, 128), nn.GELU(), nn.Linear(128, 64), nn.GELU(), nn.Linear(64, stage2_surface_channels))
        self.stage3_head = nn.Sequential(nn.Linear(latent_dim, 128), nn.GELU(), nn.Linear(128, 64), nn.GELU(), nn.Linear(64, volume_channels))

    def _scale(self, pos: torch.Tensor):
        return pos * self.pos_scale_factor

    def encode_geometry(self, surface_points: torch.Tensor, anchor_idx: torch.Tensor | None = None):
        return self.geometry_encoder(self._scale(surface_points), anchor_idx=anchor_idx)

    def encode_surface(self, surface_points: torch.Tensor, anchor_idx: torch.Tensor | None = None):
        return self.surface_encoder(self._scale(surface_points), anchor_idx=anchor_idx)

    def forward_stage1(self, surface_points: torch.Tensor, query_points: torch.Tensor):
        geom_latents, anchor_pos, _ = self.encode_geometry(surface_points)
        q = self.stage1_decoder(self._scale(query_points), geom_latents, anchor_pos)
        return self.stage1_head(q)

    def forward_stage2(self, surface_points: torch.Tensor, surface_query_points: torch.Tensor):
        surf_latents, anchor_pos, _ = self.encode_surface(surface_points)
        q = self.stage2_decoder(self._scale(surface_query_points), surf_latents, anchor_pos)
        return self.stage2_head(q)

    def forward_stage3(self, surface_points: torch.Tensor, volume_query_points: torch.Tensor):
        # Same anchor index so latent_g_m and latent_s_m are aligned point-wise.
        n_pts = surface_points.shape[1]
        anchor_idx = LoopEncoder._sample_anchor_idx(n_pts, self.geometry_encoder.anchors, surface_points.device)

        geom_latents, anchor_pos, _ = self.encode_geometry(surface_points, anchor_idx=anchor_idx)
        surf_latents, _, _ = self.encode_surface(surface_points, anchor_idx=anchor_idx)

        fused_latents = self.fusion(geom_latents, surf_latents, anchor_pos)
        q = self.stage3_decoder(self._scale(volume_query_points), fused_latents, anchor_pos)
        return self.stage3_head(q)

    def freeze_stage3_encoders(self):
        for p in self.geometry_encoder.parameters():
            p.requires_grad = False
        for p in self.surface_encoder.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def inference_stage3(self, surface_points: torch.Tensor, volume_query_points: torch.Tensor):
        n_vol = volume_query_points.shape[1]
        preds = []
        for i in range(0, n_vol, self.subregion_size):
            q = volume_query_points[:, i : i + self.subregion_size, :]
            preds.append(self.forward_stage3(surface_points, q))
        return torch.cat(preds, dim=1)
