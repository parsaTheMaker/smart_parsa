"""Two-stage CAT rebuilt directly from SMART blocks.

Design:
- Stage 1: geometry encoder + surface decoder -> surface pressure
- Stage 2: frozen stage-1 branch, surface-physics encoder + volume decoder
           where stage-1 pressure is injected into stage-2 encoder inputs
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .smart import ModulatedPositionalEmbedding, EncoderBlock, DecoderBlock


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

        for p in self.base.parameters():
            p.requires_grad = False

        nn.init.kaiming_uniform_(self.lora_a, a=5 ** 0.5)
        nn.init.zeros_(self.lora_b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        delta = (x @ self.lora_a.t()) @ self.lora_b.t()
        return base_out + self.scaling * delta


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
        volume_query_feature_channels=1,
    ):
        super().__init__()
        del parameter_channels, stage1_surface_attr_channels, stage1_volume_attr_channels, volume_query_feature_channels

        self.spatial_dim = spatial_dim
        self.surface_channels = surface_channels
        self.volume_channels = volume_channels
        self.stage2_surface_channels = stage2_surface_channels
        self.loops = num_encoder_decoder_blocks
        self.num_geo = latent_geometry_points
        self.subsampled_geometry_points = subsampled_geometry_points
        self.pos_scale_factor = pos_scale_factor
        self.subregion_size = subregion_size

        self.pos_encoder = ModulatedPositionalEmbedding(latent_dim, spatial_dim)
        # Pressure projection used only in stage-2 encoder inputs.
        self.stage2_pressure_proj = nn.Sequential(
            nn.Linear(self.stage2_surface_channels, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim),
        )

        # Stage 1 blocks (SMART-style tied decoder weights).
        self.geometry_encoder_blocks = nn.ModuleList([
            EncoderBlock(dim=latent_dim, num_heads=num_heads, dropout=dropout, spatial_dim=spatial_dim, cond_dim=0)
            for _ in range(self.loops)
        ])
        self.surface_decoder_blocks = nn.ModuleList([
            DecoderBlock(
                dim=latent_dim,
                num_heads=num_heads,
                dropout=dropout,
                spatial_dim=spatial_dim,
                cond_dim=0,
                shared_attn=self.geometry_encoder_blocks[i].cross_attn,
                shared_mlp=self.geometry_encoder_blocks[i].mlp,
            )
            for i in range(self.loops)
        ])

        # Stage 2 blocks (SMART-style tied decoder weights).
        self.surface_physics_encoder_blocks = nn.ModuleList([
            EncoderBlock(dim=latent_dim, num_heads=num_heads, dropout=dropout, spatial_dim=spatial_dim, cond_dim=0)
            for _ in range(self.loops)
        ])
        self.volume_decoder_blocks = nn.ModuleList([
            DecoderBlock(
                dim=latent_dim,
                num_heads=num_heads,
                dropout=dropout,
                spatial_dim=spatial_dim,
                cond_dim=0,
                shared_attn=self.surface_physics_encoder_blocks[i].cross_attn,
                shared_mlp=self.surface_physics_encoder_blocks[i].mlp,
            )
            for i in range(self.loops)
        ])

        self.stage2_head = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.GELU(), nn.Linear(128, 64), nn.GELU(), nn.Linear(64, stage2_surface_channels)
        )
        self.stage3_head = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.GELU(), nn.Linear(128, 64), nn.GELU(), nn.Linear(64, volume_channels)
        )

        # Per-block coupling for stage-1->stage-2 latent injection.
        self.surface_to_volume_skip_weights = nn.Parameter(torch.full((self.loops,), 1e-2))
        self.surface_to_volume_skip_weight = self.surface_to_volume_skip_weights

        # Compatibility aliases expected by tooling.
        self.geometry_encoder = self.geometry_encoder_blocks
        self.surface_encoder = self.surface_physics_encoder_blocks
        self.surface_physics_encoder = self.surface_physics_encoder_blocks
        self.surface_decoder = self.surface_decoder_blocks
        self.stage1_decoder = self.surface_decoder_blocks
        self.stage2_decoder = self.surface_decoder_blocks
        self.volume_decoder = self.volume_decoder_blocks
        self.stage3_decoder = self.volume_decoder_blocks

    def _scale_pos(self, pos: torch.Tensor):
        return pos * self.pos_scale_factor

    def _sample_positions(self, pos: torch.Tensor, num_samples: int):
        idx = torch.randperm(pos.shape[1], device=pos.device)[: min(num_samples, pos.shape[1])]
        return pos[:, idx, :]

    def _encode_stage1(self, surface_pos: torch.Tensor):
        geo = self._scale_pos(surface_pos)
        latent_pos = self._sample_positions(geo, self.num_geo)
        latent_emb = self.pos_encoder(latent_pos)

        inter = []
        for block in self.geometry_encoder_blocks:
            sub_pos = self._sample_positions(geo, self.subsampled_geometry_points)
            sub_emb = self.pos_encoder(sub_pos)
            latent_emb, e_ca = block(latent_emb, sub_emb, None, latent_geometry_pos=latent_pos, subsampled_geometry_pos=sub_pos)
            inter.append(e_ca)
        return inter, latent_pos, latent_emb

    def _decode(self, query_pos: torch.Tensor, inter_latents: list[torch.Tensor], latent_pos: torch.Tensor, decoder_blocks: nn.ModuleList):
        query_pos = self._scale_pos(query_pos)
        query_emb = self.pos_encoder(query_pos)
        for e_ca, block in zip(inter_latents, decoder_blocks):
            query_emb = block(query_emb, e_ca, None, queries_pos=query_pos, latent_geometry_pos=latent_pos)
        return query_emb

    def _encode_stage2(self, surface_query_pos: torch.Tensor, surface_pred: torch.Tensor, latent_pos: torch.Tensor, initial_latent: torch.Tensor | None):
        # Pressure is injected as an additive feature embedding on top of positional embeddings.
        surf_pos = self._scale_pos(surface_query_pos)
        pressure_emb_full = self.stage2_pressure_proj(surface_pred)

        if initial_latent is None:
            latent_emb = self.pos_encoder(latent_pos)
        else:
            latent_emb = initial_latent

        inter = []
        for block in self.surface_physics_encoder_blocks:
            idx = torch.randperm(surf_pos.shape[1], device=surf_pos.device)[: min(self.subsampled_geometry_points, surf_pos.shape[1])]
            sub_pos = surf_pos[:, idx, :]
            sub_emb = self.pos_encoder(sub_pos) + pressure_emb_full[:, idx, :]
            latent_emb, e_ca = block(latent_emb, sub_emb, None, latent_geometry_pos=latent_pos, subsampled_geometry_pos=sub_pos)
            inter.append(e_ca)
        return inter, latent_emb

    def forward_stage1_only(self, surface_input_tokens: torch.Tensor, surface_query_tokens: torch.Tensor, return_aux: bool = False):
        surface_input_pos = surface_input_tokens[..., : self.spatial_dim]
        surface_query_pos = surface_query_tokens[..., : self.spatial_dim]

        geom_latents, anchor_pos, geom_final = self._encode_stage1(surface_input_pos)
        q = self._decode(surface_query_pos, geom_latents, anchor_pos, self.surface_decoder_blocks)
        surface_pred = self.stage2_head(q)

        if return_aux:
            return surface_pred, {"anchor_pos": anchor_pos, "geom_latents": geom_latents, "geom_final": geom_final}
        return surface_pred

    def forward_stage2_only(self, surface_input_tokens: torch.Tensor, surface_query_tokens: torch.Tensor, volume_query_tokens: torch.Tensor, return_aux: bool = False):
        surface_input_pos = surface_input_tokens[..., : self.spatial_dim]
        surface_query_pos = surface_query_tokens[..., : self.spatial_dim]
        volume_query_pos = volume_query_tokens[..., : self.spatial_dim]

        with torch.no_grad():
            surface_pred, aux_s1 = self.forward_stage1_only(surface_input_pos, surface_query_pos, return_aux=True)
            geom_latents = aux_s1["geom_latents"]
            anchor_pos = aux_s1["anchor_pos"]
            geom_final = aux_s1["geom_final"]

        # Stage-2 prev/new streams + per-layer coupling.
        prev_latents, _ = self._encode_stage2(surface_query_pos, surface_pred, anchor_pos, initial_latent=geom_final)
        new_latents, _ = self._encode_stage2(surface_query_pos, surface_pred, anchor_pos, initial_latent=None)

        w = torch.clamp(self.surface_to_volume_skip_weights, min=0.0, max=1.0)
        fused_latents = []
        for m in range(self.loops):
            wm = w[m].view(1, 1, 1)
            coupled = prev_latents[m] + wm * geom_latents[m]
            fused = (1.0 - wm) * new_latents[m] + wm * coupled
            fused_latents.append(fused)

        qv = self._decode(volume_query_pos, fused_latents, anchor_pos, self.volume_decoder_blocks)
        volume_pred = self.stage3_head(qv)

        if return_aux:
            return volume_pred, {
                "surface_pred": surface_pred,
                "geom_latents": geom_latents,
                "vol_latents": fused_latents,
                "anchor_pos": anchor_pos,
                "skip_weights": w,
                "skip_weight_mean": w.mean(),
                "skip_weight_std": w.std(unbiased=False),
            }
        return volume_pred

    def forward_single_stage(self, surface_input_tokens: torch.Tensor, surface_query_tokens: torch.Tensor, volume_query_tokens: torch.Tensor, return_aux: bool = False):
        surface_pred, aux_s1 = self.forward_stage1_only(surface_input_tokens, surface_query_tokens, return_aux=True)

        surface_query_pos = surface_query_tokens[..., : self.spatial_dim]
        volume_query_pos = volume_query_tokens[..., : self.spatial_dim]
        geom_latents = aux_s1["geom_latents"]
        anchor_pos = aux_s1["anchor_pos"]
        geom_final = aux_s1["geom_final"]

        prev_latents, _ = self._encode_stage2(surface_query_pos, surface_pred, anchor_pos, initial_latent=geom_final)
        new_latents, _ = self._encode_stage2(surface_query_pos, surface_pred, anchor_pos, initial_latent=None)

        w = torch.clamp(self.surface_to_volume_skip_weights, min=0.0, max=1.0)
        fused_latents = []
        for m in range(self.loops):
            wm = w[m].view(1, 1, 1)
            coupled = prev_latents[m] + wm * geom_latents[m]
            fused = (1.0 - wm) * new_latents[m] + wm * coupled
            fused_latents.append(fused)

        qv = self._decode(volume_query_pos, fused_latents, anchor_pos, self.volume_decoder_blocks)
        volume_pred = self.stage3_head(qv)

        if return_aux:
            return surface_pred, volume_pred, {
                "skip_weights": w,
                "skip_weight_mean": w.mean(),
                "skip_weight_std": w.std(unbiased=False),
                "geom_latents": geom_latents,
                "vol_latents": fused_latents,
                "anchor_pos": anchor_pos,
            }
        return surface_pred, volume_pred

    def freeze_stage1(self):
        for module in (self.pos_encoder, self.geometry_encoder_blocks, self.surface_decoder_blocks, self.stage2_head):
            for p in module.parameters():
                p.requires_grad = False

    @staticmethod
    def _replace_linear_with_lora(module: nn.Module, rank: int, alpha: float):
        for name, child in list(module.named_children()):
            if isinstance(child, nn.Linear):
                setattr(module, name, LoRALinear(child, rank=rank, alpha=alpha))
            else:
                CAT._replace_linear_with_lora(child, rank=rank, alpha=alpha)

    def freeze_stage3_encoders(self):
        for p in self.geometry_encoder_blocks.parameters():
            p.requires_grad = False
        for p in self.surface_physics_encoder_blocks.parameters():
            p.requires_grad = False

    def enable_stage3_encoder_lora(self, rank: int = 16, alpha: float = 16.0):
        self.freeze_stage3_encoders()
        self._replace_linear_with_lora(self.geometry_encoder_blocks, rank=rank, alpha=alpha)
        self._replace_linear_with_lora(self.surface_physics_encoder_blocks, rank=rank, alpha=alpha)

    # Legacy wrappers
    def forward_stage1(self, surface_points: torch.Tensor, query_points: torch.Tensor):
        return self.forward_stage1_only(surface_points, query_points, return_aux=False)

    def forward_stage2(self, surface_points: torch.Tensor, surface_query_points: torch.Tensor):
        return self.forward_stage1_only(surface_points, surface_query_points, return_aux=False)

    def forward_stage3(self, surface_points: torch.Tensor, volume_query_points: torch.Tensor):
        return self.forward_stage2_only(surface_points, surface_points, volume_query_points, return_aux=False)

    @torch.no_grad()
    def inference_stage3(self, surface_points: torch.Tensor, volume_query_points: torch.Tensor):
        n_vol = volume_query_points.shape[1]
        preds = []
        for i in range(0, n_vol, self.subregion_size):
            q = volume_query_points[:, i : i + self.subregion_size, :]
            preds.append(self.forward_stage3(surface_points, q))
        return torch.cat(preds, dim=1)
