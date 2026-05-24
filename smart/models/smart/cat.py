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


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int = 16, alpha: float = 16.0):
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be > 0")
        self.base = base
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank

        in_features = base.in_features
        out_features = base.out_features

        device = base.weight.device
        dtype = base.weight.dtype
        self.lora_a = nn.Parameter(torch.zeros(self.rank, in_features, device=device, dtype=dtype))
        self.lora_b = nn.Parameter(torch.zeros(out_features, self.rank, device=device, dtype=dtype))

        # Freeze base linear; train only LoRA adapters.
        for p in self.base.parameters():
            p.requires_grad = False

        nn.init.kaiming_uniform_(self.lora_a, a=5 ** 0.5)
        nn.init.zeros_(self.lora_b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        # delta = (x @ A^T) @ B^T
        delta = (x @ self.lora_a.t()) @ self.lora_b.t()
        return base_out + self.scaling * delta



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

    def forward(
        self,
        surface_points_scaled: torch.Tensor,
        anchor_idx: torch.Tensor | None = None,
        shared_chunks: list[torch.Tensor] | None = None,
    ):
        # surface_points_scaled: [B, N, D]
        _, n_pts, _ = surface_points_scaled.shape
        device = surface_points_scaled.device

        if anchor_idx is None:
            anchor_idx = self._sample_anchor_idx(n_pts, self.anchors, device)

        anchor_pos = surface_points_scaled[:, anchor_idx, :]
        latent = self.pos_encoder(anchor_pos)

        chunks = shared_chunks if shared_chunks is not None else self._chunk_indices(n_pts, self.loops, device)
        intermediate_latents = []
        for m in range(self.loops):
            idx = chunks[m] if m < len(chunks) else torch.empty((0,), dtype=torch.long, device=device)
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
    """Cross-gated shared fusion block used at each CAT loop in Stage 3.

    Improvements over the prior variant:
    - Pre-norm each encoder stream before cross-gating.
    - Residual-safe fusion with learnable scales initialized to near-identity.
    - Stabilized gates initialized to weak coupling at startup.
    """

    def __init__(self, latent_dim: int, num_heads: int, spatial_dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm_g = nn.LayerNorm(latent_dim, eps=1e-6)
        self.norm_s = nn.LayerNorm(latent_dim, eps=1e-6)
        self.norm_out = nn.LayerNorm(latent_dim, eps=1e-6)

        # Bidirectional projections used by cross-gates.
        self.geom_from_surf = nn.Linear(latent_dim, latent_dim)
        self.surf_from_geom = nn.Linear(latent_dim, latent_dim)

        # Token-wise/channel-wise gates computed from concatenated normalized latents.
        self.gate_geom = nn.Sequential(
            nn.Linear(2 * latent_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim),
            nn.Sigmoid(),
        )
        self.gate_surf = nn.Sequential(
            nn.Linear(2 * latent_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim),
            nn.Sigmoid(),
        )

        self.fuse_proj = nn.Linear(2 * latent_dim, latent_dim)

        # Residual-safe scaling: start near identity and let training grow fusion contribution.
        self.cross_delta_scale = nn.Parameter(torch.tensor(1e-2))
        self.post_delta_scale = nn.Parameter(torch.tensor(1e-2))

        self.shared_attn = CrossAttention(dim=latent_dim, num_heads=num_heads, spatial_dim=spatial_dim, dropout=dropout)
        self.shared_ffn = PlainMLP(dim=latent_dim, hidden_dim=latent_dim * 4, dropout=dropout)

        self._init_stable()

    def _init_stable(self):
        # Start with weak cross-stream gates to avoid noisy fusion early in training.
        for gate in (self.gate_geom, self.gate_surf):
            last = gate[-2]
            if isinstance(last, nn.Linear):
                nn.init.zeros_(last.weight)
                nn.init.constant_(last.bias, -2.0)

    def forward(self, geom_latents: list[torch.Tensor], surf_latents: list[torch.Tensor], anchor_pos_scaled: torch.Tensor):
        fused = []
        loops = min(len(geom_latents), len(surf_latents))
        for m in range(loops):
            g = geom_latents[m]
            s = surf_latents[m]

            g_n = self.norm_g(g)
            s_n = self.norm_s(s)
            cat = torch.cat([g_n, s_n], dim=-1)

            # Cross-gated bidirectional feature injection.
            gate_g = self.gate_geom(cat)
            gate_s = self.gate_surf(cat)
            g_tilde = g_n + gate_g * self.geom_from_surf(s_n)
            s_tilde = s_n + gate_s * self.surf_from_geom(g_n)

            base = 0.5 * (g + s)
            cross_delta = self.fuse_proj(torch.cat([g_tilde, s_tilde], dim=-1))
            z = base + torch.tanh(self.cross_delta_scale) * cross_delta

            z_norm = self.norm_out(z)
            post_delta = self.shared_attn(q=z_norm, kv=z_norm, q_pos=anchor_pos_scaled, kv_pos=anchor_pos_scaled)
            post_delta = post_delta + self.shared_ffn(z_norm, None)
            z = z + torch.tanh(self.post_delta_scale) * post_delta
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

    def encode_geometry(
        self,
        surface_points: torch.Tensor,
        anchor_idx: torch.Tensor | None = None,
        shared_chunks: list[torch.Tensor] | None = None,
    ):
        return self.geometry_encoder(self._scale(surface_points), anchor_idx=anchor_idx, shared_chunks=shared_chunks)

    def encode_surface(
        self,
        surface_points: torch.Tensor,
        anchor_idx: torch.Tensor | None = None,
        shared_chunks: list[torch.Tensor] | None = None,
    ):
        return self.surface_encoder(self._scale(surface_points), anchor_idx=anchor_idx, shared_chunks=shared_chunks)

    def forward_stage1(self, surface_points: torch.Tensor, query_points: torch.Tensor):
        geom_latents, anchor_pos, _ = self.encode_geometry(surface_points)
        q = self.stage1_decoder(self._scale(query_points), geom_latents, anchor_pos)
        return self.stage1_head(q)

    def forward_stage2(self, surface_points: torch.Tensor, surface_query_points: torch.Tensor):
        surf_latents, anchor_pos, _ = self.encode_surface(surface_points)
        q = self.stage2_decoder(self._scale(surface_query_points), surf_latents, anchor_pos)
        return self.stage2_head(q)

    def forward_stage3(self, surface_points: torch.Tensor, volume_query_points: torch.Tensor):
        # Same anchor index and same per-loop surface chunks so latent_g_m and latent_s_m are aligned.
        n_pts = surface_points.shape[1]
        device = surface_points.device
        anchor_idx = LoopEncoder._sample_anchor_idx(n_pts, self.geometry_encoder.anchors, device)
        shared_chunks = LoopEncoder._chunk_indices(n_pts, self.loops, device)

        geom_latents, anchor_pos, _ = self.encode_geometry(surface_points, anchor_idx=anchor_idx, shared_chunks=shared_chunks)
        surf_latents, _, _ = self.encode_surface(surface_points, anchor_idx=anchor_idx, shared_chunks=shared_chunks)

        fused_latents = self.fusion(geom_latents, surf_latents, anchor_pos)
        q = self.stage3_decoder(self._scale(volume_query_points), fused_latents, anchor_pos)
        return self.stage3_head(q)

    @staticmethod
    def _replace_linear_with_lora(module: nn.Module, rank: int, alpha: float):
        for name, child in list(module.named_children()):
            if isinstance(child, nn.Linear):
                setattr(module, name, LoRALinear(child, rank=rank, alpha=alpha))
            else:
                CAT._replace_linear_with_lora(child, rank=rank, alpha=alpha)

    def freeze_stage3_encoders(self):
        for p in self.geometry_encoder.parameters():
            p.requires_grad = False
        for p in self.surface_encoder.parameters():
            p.requires_grad = False

    def enable_stage3_encoder_lora(self, rank: int = 16, alpha: float = 16.0):
        # Freeze encoder backbones first, then attach trainable LoRA to all encoder Linear layers.
        self.freeze_stage3_encoders()
        self._replace_linear_with_lora(self.geometry_encoder, rank=rank, alpha=alpha)
        self._replace_linear_with_lora(self.surface_encoder, rank=rank, alpha=alpha)

    @torch.no_grad()
    def inference_stage3(self, surface_points: torch.Tensor, volume_query_points: torch.Tensor):
        n_vol = volume_query_points.shape[1]
        preds = []
        for i in range(0, n_vol, self.subregion_size):
            q = volume_query_points[:, i : i + self.subregion_size, :]
            preds.append(self.forward_stage3(surface_points, q))
        return torch.cat(preds, dim=1)
