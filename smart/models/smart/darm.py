import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from einops import rearrange

from .smart import SMART


def _batched_index_select(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    if x.ndim not in (2, 3):
        raise ValueError(f"Expected rank-2 or rank-3 tensor, got shape {tuple(x.shape)}")

    batch_size, num_items = x.shape[:2]
    flat_offset = torch.arange(batch_size, device=x.device, dtype=idx.dtype).view(batch_size, 1, 1) * num_items
    flat_idx = (idx + flat_offset).reshape(-1)

    if x.ndim == 2:
        flat_x = x.reshape(batch_size * num_items)
        gathered = flat_x.index_select(0, flat_idx)
        return gathered.reshape(batch_size, *idx.shape[1:])

    channels = x.shape[-1]
    flat_x = x.reshape(batch_size * num_items, channels)
    gathered = flat_x.index_select(0, flat_idx)
    return gathered.reshape(batch_size, *idx.shape[1:], channels)


def _get_parent_module(root: nn.Module, module_name: str):
    if "." not in module_name:
        return root, module_name
    parent_name, child_name = module_name.rsplit(".", 1)
    parent = root.get_submodule(parent_name)
    return parent, child_name


class LoRALinear(nn.Module):
    def __init__(self, linear: nn.Linear, rank: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        if rank <= 0:
            raise ValueError(f"LoRA rank must be positive, got {rank}.")
        self.in_features = int(linear.in_features)
        self.out_features = int(linear.out_features)
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / float(self.rank)
        self.weight = nn.Parameter(linear.weight.detach().clone())
        if linear.bias is None:
            self.register_parameter("bias", None)
        else:
            self.bias = nn.Parameter(linear.bias.detach().clone())
        self.weight.requires_grad_(False)
        if self.bias is not None:
            self.bias.requires_grad_(False)
        self.lora_dropout = nn.Dropout(float(dropout)) if float(dropout) > 0.0 else nn.Identity()
        self.lora_A = nn.Parameter(self.weight.new_zeros(self.rank, self.in_features))
        self.lora_B = nn.Parameter(self.weight.new_zeros(self.out_features, self.rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5.0))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = F.linear(x, self.weight, self.bias)
        lora_hidden = F.linear(self.lora_dropout(x), self.lora_A, bias=None)
        lora_delta = F.linear(lora_hidden, self.lora_B, bias=None)
        return base + self.scaling * lora_delta


class FeedForwardBlock(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(self.norm(x))


class DARMModulatedPositionalEmbedding(nn.Module):
    def __init__(self, dim, spatial_dim=3, max_seq_length=10000):
        super().__init__()
        self.dim = dim
        self.spatial_dim = spatial_dim

        max_dim_per_spatial_dim = dim // spatial_dim
        dim_per_spatial_dim = max_dim_per_spatial_dim & ~1
        self.dim_per_spatial_dim = dim_per_spatial_dim

        self.total_padding = dim - (dim_per_spatial_dim * spatial_dim)
        self.register_buffer("padding", torch.zeros(1, 1, self.total_padding))

        div_term = torch.exp(torch.arange(0, dim_per_spatial_dim, 2) * (-math.log(max_seq_length) / dim_per_spatial_dim))
        self.register_buffer("div_term", div_term)

        self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim_per_spatial_dim * spatial_dim * 2))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def compute_embedding(self, pos, shift_sin=None, scale_sin=None, shift_cos=None, scale_cos=None):
        with torch.autocast(device_type=str(pos.device).split(":")[0], enabled=False):
            pos = pos.float()
            sin_cos_arg = pos[..., None] @ self.div_term[None, ...]

            embedding = torch.zeros((*sin_cos_arg.shape[:-1], self.dim_per_spatial_dim), device=sin_cos_arg.device, dtype=sin_cos_arg.dtype)
            if shift_sin is not None and scale_sin is not None and shift_cos is not None and scale_cos is not None:
                embedding[..., 0::2] = scale_sin * torch.sin(sin_cos_arg + shift_sin)
                embedding[..., 1::2] = scale_cos * torch.cos(sin_cos_arg + shift_cos)
            else:
                embedding[..., 0::2] = torch.sin(sin_cos_arg)
                embedding[..., 1::2] = torch.cos(sin_cos_arg)

        embedding = rearrange(embedding, "b n spatial_dim d -> b n (spatial_dim d)")
        if self.total_padding > 0:
            embedding = torch.concat([embedding, self.padding.expand(*embedding.shape[:-1], -1)], dim=-1)
        return embedding

    def forward(self, pos):
        initial_embedding = self.compute_embedding(pos)
        with torch.autocast(device_type=str(pos.device).split(":")[0], enabled=False):
            shift_scale = self.mlp(initial_embedding.float())
        shift_sin, scale_sin, shift_cos, scale_cos = torch.unbind(
            rearrange(
                shift_scale,
                "b n (d shift_scale spatial_dim) -> b n spatial_dim d shift_scale",
                shift_scale=4,
                spatial_dim=self.spatial_dim,
            ),
            -1,
        )
        scale_sin = 1.0 + scale_sin
        scale_cos = 1.0 + scale_cos
        return self.compute_embedding(pos, shift_sin=shift_sin, scale_sin=scale_sin, shift_cos=shift_cos, scale_cos=scale_cos)


class DARM(SMART):
    """SMART with a light dynamic anchor-routed residual readout."""

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
        subsampled_geometry_with_replacement=False,
        route_top_r=8,
        route_dim=64,
        route_temperature=1.0,
        route_chunk_size=2048,
        route_anchor_chunk_size=1024,
        high_compute_chunk_size=512,
        anchor_shift_scale=0.5,
        anchor_metric_eps=1.0e-4,
        anchor_scale_eps=1.0e-4,
        anchor_dropout=0.0,
        use_intermediate_layer_mix=False,
        type_embedding_dim=8,
        low_pos_dim=32,
        low_hidden_dim=64,
        low_pos_scale_factor=25.0,
        low_branch_num_coarse_layers=1,
        high_hidden_dim=64,
        prediction_query_chunk_size=32768,
        decoder_checkpointing=True,
        use_high_value_block=False,
        route_margin_norm_eps=1.0e-4,
        use_low_residual_branch=False,
        low_residual_init_gain=0.05,
        high_residual_init_gain=0.10,
        lora_rank=32,
        lora_alpha=32.0,
        lora_dropout=0.0,
        lora_last_n_blocks=2,
    ):
        super().__init__(
            spatial_dim=spatial_dim,
            surface_channels=surface_channels,
            volume_channels=volume_channels,
            parameter_channels=parameter_channels,
            latent_dim=latent_dim,
            latent_geometry_points=latent_geometry_points,
            subsampled_geometry_points=subsampled_geometry_points,
            num_encoder_decoder_blocks=num_encoder_decoder_blocks,
            num_heads=num_heads,
            pos_scale_factor=pos_scale_factor,
            dropout=dropout,
            subregion_size=subregion_size,
            subsampled_geometry_with_replacement=subsampled_geometry_with_replacement,
        )
        self._smart_linear_module_names = tuple(
            name for name, module in self.named_modules() if isinstance(module, nn.Linear)
        )

        self.spatial_dim = int(spatial_dim)
        self.route_top_r = max(1, int(route_top_r))
        self.route_dim = int(route_dim)
        self.route_temperature = max(float(route_temperature), 1.0e-4)
        self.route_chunk_size = max(1, int(route_chunk_size))
        self.route_anchor_chunk_size = max(self.route_top_r, int(route_anchor_chunk_size))
        self.high_compute_chunk_size = max(1, int(high_compute_chunk_size))
        self.anchor_shift_scale = float(anchor_shift_scale)
        self.anchor_metric_eps = float(anchor_metric_eps)
        self.anchor_scale_eps = float(anchor_scale_eps)
        self.anchor_dropout = float(anchor_dropout)
        self.use_intermediate_layer_mix = bool(use_intermediate_layer_mix)
        self.low_branch_num_coarse_layers = max(1, int(low_branch_num_coarse_layers))
        self.prediction_query_chunk_size = max(1, int(prediction_query_chunk_size))
        self.decoder_checkpointing = bool(decoder_checkpointing)
        self.use_high_value_block = bool(use_high_value_block)
        self.route_margin_norm_eps = float(route_margin_norm_eps)
        self.use_low_residual_branch = bool(use_low_residual_branch)
        self.low_residual_init_gain = float(low_residual_init_gain)
        self.high_residual_init_gain = float(high_residual_init_gain)
        self.lora_rank = max(0, int(lora_rank))
        self.lora_alpha = float(lora_alpha)
        self.lora_dropout = float(lora_dropout)
        self.lora_last_n_blocks = max(0, int(lora_last_n_blocks))
        self.lora_linear_module_count = 0
        self.type_embedding = nn.Embedding(2, int(type_embedding_dim))

        if self.use_intermediate_layer_mix:
            self.layer_router = nn.Sequential(
                nn.LayerNorm(latent_dim, eps=1e-6),
                nn.Linear(latent_dim, len(self.encoder_blocks)),
            )
        else:
            self.layer_router = None

        self.anchor_context_norm = nn.LayerNorm(latent_dim, eps=1e-6)
        self.route_query_norm = nn.LayerNorm(latent_dim, eps=1e-6)
        self.route_anchor_norm = nn.LayerNorm(latent_dim, eps=1e-6)
        self.route_query_proj = nn.Linear(latent_dim, self.route_dim)
        self.route_anchor_proj = nn.Linear(latent_dim, self.route_dim)

        self.anchor_shift_proj = nn.Linear(latent_dim, spatial_dim)
        self.anchor_metric_proj = nn.Linear(latent_dim, spatial_dim)
        self.anchor_scale_proj = nn.Linear(latent_dim, 1)
        self.anchor_bias_proj = nn.Linear(latent_dim, 1)

        self.anchor_value_proj = nn.Linear(latent_dim, high_hidden_dim)
        self.delta_value_proj = nn.Sequential(
            nn.Linear(spatial_dim, high_hidden_dim),
            nn.GELU(),
        )
        self.type_value_proj = nn.Linear(type_embedding_dim, high_hidden_dim)
        self.query_gate_proj = nn.Linear(latent_dim, 1)
        self.anchor_gate_proj = nn.Linear(high_hidden_dim, 1)
        self.delta_gate_proj = nn.Linear(spatial_dim, 1)
        self.query_stat_gate_proj = nn.Linear(3, 1)
        self.route_temperature_proj = nn.Linear(latent_dim, 1)
        self.score_gate_scale = nn.Parameter(torch.tensor(1.0))
        self.distance_gate_scale = nn.Parameter(torch.tensor(0.1))
        self.gate_bias = nn.Parameter(torch.tensor(0.5))
        self.margin_confidence_scale = nn.Parameter(torch.tensor(1.0))
        self.support_gain_proj = nn.Sequential(
            nn.LayerNorm(latent_dim, eps=1e-6),
            nn.Linear(latent_dim, high_hidden_dim),
            nn.Sigmoid(),
        )
        self.high_context_norm = nn.LayerNorm(high_hidden_dim, eps=1e-6)
        self.high_value_block = FeedForwardBlock(high_hidden_dim, high_hidden_dim * 2) if self.use_high_value_block else nn.Identity()

        self.low_pos_scale_factor = float(low_pos_scale_factor)
        self.low_pos_encoder = DARMModulatedPositionalEmbedding(int(low_pos_dim), spatial_dim=spatial_dim)
        self.low_query_proj = nn.Sequential(
            nn.LayerNorm(latent_dim, eps=1e-6),
            nn.Linear(latent_dim, low_hidden_dim),
            nn.GELU(),
        )
        self.low_pos_proj = nn.Linear(int(low_pos_dim), low_hidden_dim)
        self.low_type_proj = nn.Linear(type_embedding_dim, low_hidden_dim)
        self.low_block = FeedForwardBlock(low_hidden_dim, low_hidden_dim * 2)

        self.surface_low_head = nn.Linear(low_hidden_dim, surface_channels)
        self.volume_low_head = nn.Linear(low_hidden_dim, volume_channels)
        self.surface_high_head = nn.Linear(high_hidden_dim, surface_channels)
        self.volume_high_head = nn.Linear(high_hidden_dim, volume_channels)
        self.surface_low_gain = nn.Parameter(torch.tensor(self.low_residual_init_gain))
        self.volume_low_gain = nn.Parameter(torch.tensor(self.low_residual_init_gain))
        self.surface_high_gain = nn.Parameter(torch.tensor(self.high_residual_init_gain))
        self.volume_high_gain = nn.Parameter(torch.tensor(self.high_residual_init_gain))
        self._initialize_residual_heads()
        self.lora_linear_module_count = self._inject_lora_into_pretrained_linears()

    def _initialize_residual_heads(self):
        # DARM is trained as a residual editor on top of the pretrained SMART
        # predictor. Use a tiny non-zero initialization so the initial
        # prediction remains essentially the pretrained SMART solution while
        # still allowing gradients to flow through the residual pathways from
        # the first optimization step.
        for head in (
            self.surface_low_head,
            self.volume_low_head,
            self.surface_high_head,
            self.volume_high_head,
        ):
            nn.init.trunc_normal_(head.weight, std=1.0e-4)
            nn.init.zeros_(head.bias)

    def _inject_lora_into_pretrained_linears(self):
        if self.lora_rank <= 0:
            return 0

        replaced = 0
        skip_prefixes = ("pos_encoder.", "mlp.")
        allowed_prefixes = ()
        if self.lora_last_n_blocks > 0:
            start_idx = max(0, len(self.encoder_blocks) - self.lora_last_n_blocks)
            allowed_prefixes = tuple(
                f"encoder_blocks.{block_idx}."
                for block_idx in range(start_idx, len(self.encoder_blocks))
            )
        for module_name in self._smart_linear_module_names:
            if module_name.startswith(skip_prefixes):
                continue
            if allowed_prefixes and not module_name.startswith(allowed_prefixes):
                continue
            parent, child_name = _get_parent_module(self, module_name)
            module = getattr(parent, child_name)
            if not isinstance(module, nn.Linear):
                continue
            setattr(
                parent,
                child_name,
                LoRALinear(
                    module,
                    rank=self.lora_rank,
                    alpha=self.lora_alpha,
                    dropout=self.lora_dropout,
                ),
            )
            replaced += 1
        return replaced

    def _decode_feature_summary(self, intermediate_latent_geometries, latent_geo_pos, params, query_pos):
        query_pos_scaled = query_pos * self.pos_scale_factor
        query_emb = self.pos_encoder(query_pos_scaled)
        low_query_sum = None
        coarse_remaining = min(self.low_branch_num_coarse_layers, len(self.decoder_blocks))
        for e_ca, block in zip(intermediate_latent_geometries, self.decoder_blocks):
            if self.training and self.decoder_checkpointing and query_emb.requires_grad:
                def block_forward(query_tensor, latent_tensor, query_pos_tensor, latent_pos_tensor):
                    return block(
                        query_tensor,
                        latent_tensor,
                        params,
                        queries_pos=query_pos_tensor,
                        latent_geometry_pos=latent_pos_tensor,
                    )

                query_emb = checkpoint(
                    block_forward,
                    query_emb,
                    e_ca,
                    query_pos_scaled,
                    latent_geo_pos,
                    use_reentrant=False,
                )
            else:
                query_emb = block(query_emb, e_ca, params, queries_pos=query_pos_scaled, latent_geometry_pos=latent_geo_pos)
            if coarse_remaining > 0:
                low_query_sum = query_emb if low_query_sum is None else (low_query_sum + query_emb)
                coarse_remaining -= 1
        if low_query_sum is None:
            raise RuntimeError("DARM requires at least one decoder block.")
        num_coarse = self.low_branch_num_coarse_layers - coarse_remaining
        return low_query_sum / float(num_coarse), query_emb

    def _prepare_readout_context(self, intermediate_latent_geometries, latent_geo_pos, final_latent_geo):
        anchor_context = self.anchor_context_norm(final_latent_geo)
        anchor_centers = latent_geo_pos + self.anchor_shift_scale * torch.tanh(self.anchor_shift_proj(anchor_context))
        anchor_metric = F.softplus(self.anchor_metric_proj(anchor_context)) + self.anchor_metric_eps
        anchor_scale = F.softplus(self.anchor_scale_proj(anchor_context)).squeeze(-1) + self.anchor_scale_eps
        anchor_bias = self.anchor_bias_proj(anchor_context).squeeze(-1)

        source_latents = intermediate_latent_geometries if self.use_intermediate_layer_mix else [final_latent_geo]
        route_anchor_keys = []
        route_anchor_values = []
        for latent in source_latents:
            latent_norm = self.route_anchor_norm(latent)
            route_anchor_keys.append(self.route_anchor_proj(latent_norm))
            route_anchor_values.append(self.anchor_value_proj(latent_norm))

        return {
            "anchor_centers": anchor_centers,
            "anchor_metric": anchor_metric,
            "anchor_scale": anchor_scale,
            "anchor_bias": anchor_bias,
            "route_anchor_keys": route_anchor_keys,
            "route_anchor_values": route_anchor_values,
            "num_layers": len(route_anchor_keys),
        }

    def _low_branch(self, query_emb, query_pos_raw, query_type):
        if query_emb.shape[1] == 0 or not self.use_low_residual_branch:
            out_dim = self.surface_channels if query_type == 0 else self.volume_channels
            return query_emb.new_zeros(query_emb.shape[0], query_emb.shape[1], out_dim)

        with torch.autocast(device_type=str(query_emb.device).split(":")[0], enabled=False):
            query_emb = query_emb.float()
            low_pos = self.low_pos_encoder(query_pos_raw * self.low_pos_scale_factor).float()
            type_embed = self.type_embedding.weight[query_type].view(1, 1, -1).float()
            low_hidden = (
                self.low_query_proj(query_emb)
                + self.low_pos_proj(low_pos)
                + self.low_type_proj(type_embed).expand(query_emb.shape[0], query_emb.shape[1], -1)
            )
            low_hidden = self.low_block(low_hidden)
            return self.surface_low_head(low_hidden) if query_type == 0 else self.volume_low_head(low_hidden)

    def _apply_anchor_dropout(self, top_scores: torch.Tensor) -> torch.Tensor:
        if not self.training or self.anchor_dropout <= 0.0 or top_scores.shape[-1] <= 1:
            return top_scores
        keep_mask = torch.rand_like(top_scores) > self.anchor_dropout
        keep_mask[..., 0] = True
        return top_scores.masked_fill(~keep_mask, torch.finfo(top_scores.dtype).min)

    def _compute_route_scores_block(
        self,
        query_key,
        layer_weights,
        query_pos_scaled,
        readout_context,
        anchor_start,
        anchor_end,
    ):
        batch_size, num_queries = query_key.shape[:2]
        work_dtype = torch.float32
        query_key_w = query_key.to(dtype=work_dtype)
        layer_weights_w = None if layer_weights is None else layer_weights.to(dtype=work_dtype)
        block_size = anchor_end - anchor_start
        scores = query_key_w.new_zeros(batch_size, num_queries, block_size)

        route_anchor_keys = readout_context["route_anchor_keys"]
        if len(route_anchor_keys) == 1:
            anchor_keys_block = route_anchor_keys[0][:, anchor_start:anchor_end].to(dtype=work_dtype)
            scores = torch.matmul(query_key_w, anchor_keys_block.transpose(1, 2))
        else:
            for layer_idx, anchor_keys in enumerate(route_anchor_keys):
                anchor_keys_block = anchor_keys[:, anchor_start:anchor_end].to(dtype=work_dtype)
                layer_scores = torch.matmul(query_key_w, anchor_keys_block.transpose(1, 2))
                scores = scores + layer_weights_w[..., layer_idx:layer_idx + 1] * layer_scores

        scores = scores / math.sqrt(float(self.route_dim))

        anchor_centers = readout_context["anchor_centers"][:, anchor_start:anchor_end].to(dtype=work_dtype)
        anchor_metric = readout_context["anchor_metric"][:, anchor_start:anchor_end].to(dtype=work_dtype)
        anchor_scale = readout_context["anchor_scale"][:, anchor_start:anchor_end].to(dtype=work_dtype)
        anchor_bias = readout_context["anchor_bias"][:, anchor_start:anchor_end].to(dtype=work_dtype)

        query_pos_w = query_pos_scaled.to(dtype=work_dtype)
        query_pos_sq = query_pos_w.square()
        metric_center = anchor_metric * anchor_centers
        metric_center_sq = (metric_center * anchor_centers).sum(dim=-1)

        dist = torch.matmul(query_pos_sq, anchor_metric.transpose(1, 2))
        dist = dist - 2.0 * torch.matmul(query_pos_w, metric_center.transpose(1, 2))
        dist = dist + metric_center_sq.unsqueeze(1)
        dist = dist / (float(self.pos_scale_factor) ** 2)
        return scores - dist / anchor_scale.unsqueeze(1) + anchor_bias.unsqueeze(1)

    def _select_topk_route_scores(self, query_key, layer_weights, query_pos_scaled, readout_context):
        num_anchors = readout_context["anchor_centers"].shape[1]
        top_r = min(self.route_top_r, num_anchors)

        best_scores = None
        best_indices = None
        for anchor_start in range(0, num_anchors, self.route_anchor_chunk_size):
            anchor_end = min(anchor_start + self.route_anchor_chunk_size, num_anchors)
            block_scores = self._compute_route_scores_block(
                query_key,
                layer_weights,
                query_pos_scaled,
                readout_context,
                anchor_start,
                anchor_end,
            )
            block_top_scores, block_top_idx = torch.topk(block_scores, k=min(top_r, anchor_end - anchor_start), dim=-1)
            block_top_idx = block_top_idx + anchor_start

            if best_scores is None:
                best_scores = block_top_scores
                best_indices = block_top_idx
                continue

            merged_scores = torch.cat([best_scores, block_top_scores], dim=-1)
            merged_idx = torch.cat([best_indices, block_top_idx], dim=-1)
            keep_idx = torch.topk(merged_scores, k=top_r, dim=-1).indices
            best_scores = torch.gather(merged_scores, -1, keep_idx)
            best_indices = torch.gather(merged_idx, -1, keep_idx)

        return best_scores, best_indices

    def _mix_selected_anchor_values(self, top_idx, layer_weights, readout_context):
        route_anchor_values = readout_context["route_anchor_values"]
        if len(route_anchor_values) == 1:
            return _batched_index_select(route_anchor_values[0], top_idx)

        mixed_values = None
        for layer_idx, anchor_values in enumerate(route_anchor_values):
            selected = _batched_index_select(anchor_values, top_idx)
            weight = layer_weights[..., layer_idx:layer_idx + 1].unsqueeze(-1)
            contrib = weight * selected
            mixed_values = contrib if mixed_values is None else (mixed_values + contrib)
        return mixed_values

    def _high_branch_chunk(
        self,
        query_emb_chunk,
        query_pos_chunk_raw,
        query_type,
        readout_context,
        return_route_debug=False,
        route_debug_max_queries=0,
    ):
        num_queries = int(query_emb_chunk.shape[1])
        if num_queries == 0:
            out_dim = self.surface_channels if query_type == 0 else self.volume_channels
            empty = query_emb_chunk.new_zeros(query_emb_chunk.shape[0], 0, out_dim)
            zero = query_emb_chunk.new_zeros(())
            empty_aux = query_emb_chunk.new_zeros(query_emb_chunk.shape[0], 0, 1)
            return empty, {
                "gate_sum": zero,
                "raw_gate_sum": zero,
                "support_mass_sum": zero,
                "route_confidence_sum": zero,
                "route_entropy_sum": zero,
                "route_spread_sum": zero,
                "query_count": 0,
                "support_mass": empty_aux,
                "route_confidence": empty_aux,
                "route_entropy": empty_aux,
                "evidence_mass": empty_aux,
                "raw_gate": empty_aux,
            }

        with torch.autocast(device_type=str(query_emb_chunk.device).split(":")[0], enabled=False):
            query_type_embed = self.type_embedding.weight[query_type].view(1, 1, -1).float()
            type_value = self.type_value_proj(query_type_embed).unsqueeze(2)
            query_pos_scaled = (query_pos_chunk_raw * self.pos_scale_factor).float()
            query_emb_chunk = query_emb_chunk.float()

            if readout_context["num_layers"] == 1 or self.layer_router is None:
                layer_weights = None
            else:
                layer_weights = torch.softmax(self.layer_router(query_emb_chunk), dim=-1)

            query_key = self.route_query_proj(self.route_query_norm(query_emb_chunk))
            top_scores, top_idx = self._select_topk_route_scores(query_key, layer_weights, query_pos_scaled, readout_context)
            top_scores = self._apply_anchor_dropout(top_scores.float())
            query_temperature = self.route_temperature * (
                0.5 + torch.sigmoid(self.route_temperature_proj(query_emb_chunk))
            )
            if top_scores.shape[-1] > 1:
                route_margin = top_scores[..., :1] - top_scores[..., 1:2]
                route_scale = top_scores.std(dim=-1, keepdim=True, unbiased=False)
                route_scale = route_scale + query_temperature.clamp_min(self.route_margin_norm_eps)
                probe_scores = top_scores - top_scores.mean(dim=-1, keepdim=True)
                probe_scores = probe_scores / route_scale.clamp_min(self.route_margin_norm_eps)
                probe_rho = torch.softmax(probe_scores, dim=-1)
                route_entropy = -(probe_rho.clamp_min(1.0e-8) * probe_rho.clamp_min(1.0e-8).log()).sum(dim=-1, keepdim=True)
                route_entropy = route_entropy / math.log(float(top_scores.shape[-1]))
            else:
                route_margin = torch.ones_like(top_scores[..., :1])
                route_scale = torch.ones_like(route_margin)
                route_entropy = torch.zeros_like(route_margin)
            route_margin_norm = route_margin / route_scale.clamp_min(self.route_margin_norm_eps)
            route_confidence = torch.sigmoid(
                self.margin_confidence_scale.to(dtype=route_margin.dtype) * route_margin_norm
            )
            rho_logits = top_scores / query_temperature.clamp_min(1.0e-4)
            rho_logits = rho_logits - rho_logits.max(dim=-1, keepdim=True).values
            rho = torch.softmax(rho_logits, dim=-1)

        gate_sum = query_emb_chunk.new_zeros(())
        raw_gate_sum = query_emb_chunk.new_zeros(())
        support_mass_sum = query_emb_chunk.new_zeros(())
        route_confidence_sum = query_emb_chunk.new_zeros(())
        route_entropy_sum = query_emb_chunk.new_zeros(())
        route_spread_sum = query_emb_chunk.new_zeros(())
        high_out_chunks = []
        support_mass_chunks = []
        route_confidence_chunks = []
        route_entropy_chunks = []
        evidence_mass_chunks = []
        raw_gate_chunks = []
        debug = None
        debug_queries_remaining = int(route_debug_max_queries)

        for start in range(0, num_queries, self.high_compute_chunk_size):
            end = min(start + self.high_compute_chunk_size, num_queries)
            top_idx_sub = top_idx[:, start:end]
            top_scores_sub = top_scores[:, start:end]
            rho_sub = rho[:, start:end]
            query_emb_sub = query_emb_chunk[:, start:end]
            query_pos_sub = query_pos_scaled[:, start:end]
            layer_weights_sub = None if layer_weights is None else layer_weights[:, start:end]

            selected_centers = _batched_index_select(readout_context["anchor_centers"], top_idx_sub)
            selected_metric = _batched_index_select(readout_context["anchor_metric"], top_idx_sub)
            delta_scaled = query_pos_sub.unsqueeze(2) - selected_centers
            delta = delta_scaled / float(self.pos_scale_factor)
            selected_dist = (selected_metric * delta_scaled.square()).sum(dim=-1) / (float(self.pos_scale_factor) ** 2)
            selected_dist_feature = torch.sqrt(selected_dist.clamp_min(1.0e-8))
            selected_dist_feature = selected_dist_feature / selected_dist_feature.detach().mean(dim=-1, keepdim=True).clamp_min(1.0e-6)
            mixed_anchor_values = self._mix_selected_anchor_values(top_idx_sub, layer_weights_sub, readout_context)
            anchor_tokens = mixed_anchor_values + self.delta_value_proj(delta) + type_value

            gate_logits = self.query_gate_proj(query_emb_sub).squeeze(-1).unsqueeze(-1)
            gate_logits = gate_logits + self.anchor_gate_proj(mixed_anchor_values).squeeze(-1)
            gate_logits = gate_logits + self.delta_gate_proj(delta).squeeze(-1)
            gate_logits = gate_logits + self.score_gate_scale.to(dtype=top_scores_sub.dtype) * top_scores_sub
            gate_logits = gate_logits - self.distance_gate_scale.to(dtype=selected_dist_feature.dtype) * selected_dist_feature
            route_confidence_sub = route_confidence[:, start:end]
            route_entropy_sub = route_entropy[:, start:end]
            route_spread_hint = torch.sqrt(selected_dist.mean(dim=-1, keepdim=True).clamp_min(1.0e-8))
            query_gate_stats = torch.cat(
                [
                    route_confidence_sub.to(dtype=query_emb_sub.dtype),
                    route_entropy_sub.to(dtype=query_emb_sub.dtype),
                    route_spread_hint.to(dtype=query_emb_sub.dtype),
                ],
                dim=-1,
            )
            gate_logits = gate_logits + self.query_stat_gate_proj(query_gate_stats).to(dtype=gate_logits.dtype)
            gate_logits = gate_logits + self.gate_bias.to(dtype=gate_logits.dtype)
            evidence = torch.sigmoid(gate_logits)
            raw_gate = evidence.mean(dim=-1, keepdim=True)
            support = rho_sub * evidence.to(dtype=rho_sub.dtype)
            support_mass = support.sum(dim=-1, keepdim=True).clamp_min(1.0e-6)
            evidence_mass = raw_gate * route_confidence_sub.to(dtype=raw_gate.dtype)
            support_norm = support / support_mass
            route_spread = torch.sqrt(
                (support_norm * selected_dist.to(dtype=support_norm.dtype)).sum(dim=-1, keepdim=True).clamp_min(1.0e-8)
            )

            flat_support = support_norm.to(dtype=anchor_tokens.dtype).reshape(-1, 1, support_norm.shape[-1])
            flat_tokens = anchor_tokens.reshape(-1, anchor_tokens.shape[-2], anchor_tokens.shape[-1])
            anchor_context = torch.bmm(flat_support, flat_tokens).reshape(
                anchor_tokens.shape[0],
                anchor_tokens.shape[1],
                anchor_tokens.shape[-1],
            )
            high_hidden_base = self.high_context_norm(anchor_context)
            high_hidden_base = high_hidden_base * self.support_gain_proj(query_emb_sub)
            high_hidden_base = self.high_value_block(high_hidden_base)
            high_hidden = high_hidden_base * raw_gate.to(dtype=high_hidden_base.dtype)
            high_hidden = high_hidden * route_confidence_sub.to(dtype=high_hidden_base.dtype)

            if query_type == 0:
                high_out_chunks.append(self.surface_high_head(high_hidden))
            else:
                high_out_chunks.append(self.volume_high_head(high_hidden))

            gate_sum = gate_sum + evidence_mass.sum()
            raw_gate_sum = raw_gate_sum + raw_gate.sum()
            support_mass_sum = support_mass_sum + support_mass.sum()
            route_confidence_sum = route_confidence_sum + route_confidence_sub.sum()
            route_entropy_sum = route_entropy_sum + route_entropy_sub.sum()
            route_spread_sum = route_spread_sum + route_spread.sum()
            support_mass_chunks.append(support_mass)
            route_confidence_chunks.append(route_confidence_sub)
            route_entropy_chunks.append(route_entropy_sub)
            evidence_mass_chunks.append(evidence_mass)
            raw_gate_chunks.append(raw_gate)

            if return_route_debug and debug_queries_remaining > 0:
                keep = min(debug_queries_remaining, end - start)
                debug_piece = {
                    "top_indices": top_idx_sub[:, :keep].detach(),
                    "routing_weights": rho_sub[:, :keep].detach(),
                    "evidence_gates": evidence[:, :keep].detach(),
                    "evidence_mass": evidence_mass[:, :keep].detach(),
                    "support_mass": support_mass[:, :keep].detach(),
                    "route_confidence": route_confidence_sub[:, :keep].detach(),
                    "route_entropy": route_entropy_sub[:, :keep].detach(),
                    "selected_distance": selected_dist[:, :keep].detach(),
                }
                if debug is None:
                    debug = debug_piece
                else:
                    debug = {key: torch.cat([debug[key], value], dim=1) for key, value in debug_piece.items()}
                debug_queries_remaining -= keep

        high_out = torch.cat(high_out_chunks, dim=1) if high_out_chunks else query_emb_chunk.new_zeros(
            query_emb_chunk.shape[0],
            0,
            self.surface_channels if query_type == 0 else self.volume_channels,
        )

        aux = {
            "gate_sum": gate_sum,
            "raw_gate_sum": raw_gate_sum,
            "support_mass_sum": support_mass_sum,
            "route_confidence_sum": route_confidence_sum,
            "route_entropy_sum": route_entropy_sum,
            "route_spread_sum": route_spread_sum,
            "query_count": num_queries * query_emb_chunk.shape[0],
            "support_mass": torch.cat(support_mass_chunks, dim=1) if support_mass_chunks else query_emb_chunk.new_zeros(query_emb_chunk.shape[0], 0, 1),
            "route_confidence": torch.cat(route_confidence_chunks, dim=1) if route_confidence_chunks else query_emb_chunk.new_zeros(query_emb_chunk.shape[0], 0, 1),
            "route_entropy": torch.cat(route_entropy_chunks, dim=1) if route_entropy_chunks else query_emb_chunk.new_zeros(query_emb_chunk.shape[0], 0, 1),
            "evidence_mass": torch.cat(evidence_mass_chunks, dim=1) if evidence_mass_chunks else query_emb_chunk.new_zeros(query_emb_chunk.shape[0], 0, 1),
            "raw_gate": torch.cat(raw_gate_chunks, dim=1) if raw_gate_chunks else query_emb_chunk.new_zeros(query_emb_chunk.shape[0], 0, 1),
        }
        if debug is not None:
            aux["debug"] = debug
        return high_out, aux

    def _readout_branch(
        self,
        low_query_emb,
        high_query_emb,
        query_pos_raw,
        query_type,
        readout_context,
        only_low=False,
        return_route_debug=False,
        route_debug_max_queries=0,
    ):
        low_out = self._low_branch(low_query_emb, query_pos_raw, query_type)
        if only_low:
            high_out = torch.zeros_like(low_out)
            return low_out, high_out, {
                "mean_support_mass": low_out.new_zeros(()),
                "mean_evidence_gate": low_out.new_zeros(()),
                "mean_raw_evidence_gate": low_out.new_zeros(()),
                "mean_route_confidence": low_out.new_zeros(()),
                "mean_route_entropy": low_out.new_zeros(()),
                "mean_route_spread": low_out.new_zeros(()),
                "support_mass": low_out.new_zeros(low_out.shape[0], low_out.shape[1], 1),
                "route_confidence": low_out.new_zeros(low_out.shape[0], low_out.shape[1], 1),
                "route_entropy": low_out.new_zeros(low_out.shape[0], low_out.shape[1], 1),
                "evidence_mass": low_out.new_zeros(low_out.shape[0], low_out.shape[1], 1),
                "raw_gate": low_out.new_zeros(low_out.shape[0], low_out.shape[1], 1),
            }

        high_out = torch.zeros_like(low_out)
        gate_sum = low_out.new_zeros(())
        raw_gate_sum = low_out.new_zeros(())
        support_mass_sum = low_out.new_zeros(())
        route_confidence_sum = low_out.new_zeros(())
        route_entropy_sum = low_out.new_zeros(())
        route_spread_sum = low_out.new_zeros(())
        query_count = 0
        route_debug = None
        support_mass = low_out.new_zeros(low_out.shape[0], low_out.shape[1], 1)
        route_confidence = low_out.new_zeros(low_out.shape[0], low_out.shape[1], 1)
        route_entropy = low_out.new_zeros(low_out.shape[0], low_out.shape[1], 1)
        evidence_mass = low_out.new_zeros(low_out.shape[0], low_out.shape[1], 1)
        raw_gate = low_out.new_zeros(low_out.shape[0], low_out.shape[1], 1)

        for start in range(0, high_query_emb.shape[1], self.route_chunk_size):
            end = min(start + self.route_chunk_size, high_query_emb.shape[1])
            high_chunk, chunk_aux = self._high_branch_chunk(
                high_query_emb[:, start:end],
                query_pos_raw[:, start:end],
                query_type,
                readout_context,
                return_route_debug=return_route_debug and route_debug is None and route_debug_max_queries > 0,
                route_debug_max_queries=route_debug_max_queries,
            )
            high_out[:, start:end] = high_chunk
            gate_sum = gate_sum + chunk_aux["gate_sum"]
            raw_gate_sum = raw_gate_sum + chunk_aux["raw_gate_sum"]
            support_mass_sum = support_mass_sum + chunk_aux["support_mass_sum"]
            route_confidence_sum = route_confidence_sum + chunk_aux["route_confidence_sum"]
            route_entropy_sum = route_entropy_sum + chunk_aux["route_entropy_sum"]
            route_spread_sum = route_spread_sum + chunk_aux["route_spread_sum"]
            query_count += int(chunk_aux["query_count"])
            support_mass[:, start:end] = chunk_aux["support_mass"]
            route_confidence[:, start:end] = chunk_aux["route_confidence"]
            route_entropy[:, start:end] = chunk_aux["route_entropy"]
            evidence_mass[:, start:end] = chunk_aux["evidence_mass"]
            raw_gate[:, start:end] = chunk_aux["raw_gate"]

            if return_route_debug and route_debug is None and route_debug_max_queries > 0 and "debug" in chunk_aux:
                keep = min(route_debug_max_queries, high_chunk.shape[1])
                route_debug = {key: value[:, :keep].cpu() for key, value in chunk_aux["debug"].items()}

        denom = float(max(query_count, 1))
        aux = {
            "mean_support_mass": support_mass_sum / denom,
            "mean_evidence_gate": gate_sum / denom,
            "mean_raw_evidence_gate": raw_gate_sum / denom,
            "mean_route_confidence": route_confidence_sum / denom,
            "mean_route_entropy": route_entropy_sum / denom,
            "mean_route_spread": route_spread_sum / denom,
            "support_mass": support_mass,
            "route_confidence": route_confidence,
            "route_entropy": route_entropy,
            "evidence_mass": evidence_mass,
            "raw_gate": raw_gate,
        }
        if route_debug is not None:
            aux["route_debug"] = route_debug
        return low_out, high_out, aux

    def predict_from_encoded(
        self,
        intermediate_latent_geometries,
        latent_geo_pos,
        final_latent_geo,
        surf_query_pos,
        vol_query_pos,
        params,
        return_aux=False,
        return_route_debug=False,
        route_debug_max_queries=0,
        only_low=False,
    ):
        readout_context = self._prepare_readout_context(intermediate_latent_geometries, latent_geo_pos, final_latent_geo)
        surf_count = int(surf_query_pos.shape[1])
        vol_count = int(vol_query_pos.shape[1])
        total_count = surf_count + vol_count

        if total_count == 0:
            pred_surf = surf_query_pos.new_zeros(surf_query_pos.shape[0], 0, self.surface_channels)
            pred_vol = vol_query_pos.new_zeros(vol_query_pos.shape[0], 0, self.volume_channels)
            if not return_aux:
                return pred_surf, pred_vol
            aux = {
                "surface_base": pred_surf.detach(),
                "volume_base": pred_vol.detach(),
                "surface_residual": pred_surf.new_zeros(pred_surf.shape[0], 0, self.surface_channels),
                "volume_residual": pred_vol.new_zeros(pred_vol.shape[0], 0, self.volume_channels),
                "surface_support_mass": pred_surf.new_zeros(pred_surf.shape[0], 0, 1),
                "volume_support_mass": pred_vol.new_zeros(pred_vol.shape[0], 0, 1),
                "surface_route_confidence": pred_surf.new_zeros(pred_surf.shape[0], 0, 1),
                "volume_route_confidence": pred_vol.new_zeros(pred_vol.shape[0], 0, 1),
                "surface_route_entropy": pred_surf.new_zeros(pred_surf.shape[0], 0, 1),
                "volume_route_entropy": pred_vol.new_zeros(pred_vol.shape[0], 0, 1),
                "surface_evidence_mass": pred_surf.new_zeros(pred_surf.shape[0], 0, 1),
                "volume_evidence_mass": pred_vol.new_zeros(pred_vol.shape[0], 0, 1),
                "surface_raw_evidence_gate": pred_surf.new_zeros(pred_surf.shape[0], 0, 1),
                "volume_raw_evidence_gate": pred_vol.new_zeros(pred_vol.shape[0], 0, 1),
                "mean_support_mass": pred_surf.new_zeros(()),
                "mean_evidence_gate": pred_surf.new_zeros(()),
                "mean_raw_evidence_gate": pred_surf.new_zeros(()),
                "mean_route_confidence": pred_surf.new_zeros(()),
                "mean_route_entropy": pred_surf.new_zeros(()),
                "mean_route_spread": pred_surf.new_zeros(()),
            }
            return pred_surf, pred_vol, aux

        query_pos = torch.cat([surf_query_pos, vol_query_pos], dim=1)

        surf_low = surf_query_pos.new_zeros(surf_query_pos.shape[0], surf_count, self.surface_channels)
        surf_high = surf_query_pos.new_zeros(surf_query_pos.shape[0], surf_count, self.surface_channels)
        vol_low = vol_query_pos.new_zeros(vol_query_pos.shape[0], vol_count, self.volume_channels)
        vol_high = vol_query_pos.new_zeros(vol_query_pos.shape[0], vol_count, self.volume_channels)
        surf_base = surf_query_pos.new_zeros(surf_query_pos.shape[0], surf_count, self.surface_channels)
        vol_base = vol_query_pos.new_zeros(vol_query_pos.shape[0], vol_count, self.volume_channels)
        surf_support_sum = query_pos.new_zeros(())
        surf_gate_sum = query_pos.new_zeros(())
        surf_raw_gate_sum = query_pos.new_zeros(())
        surf_conf_sum = query_pos.new_zeros(())
        surf_entropy_sum = query_pos.new_zeros(())
        surf_spread_sum = query_pos.new_zeros(())
        vol_support_sum = query_pos.new_zeros(())
        vol_gate_sum = query_pos.new_zeros(())
        vol_raw_gate_sum = query_pos.new_zeros(())
        vol_conf_sum = query_pos.new_zeros(())
        vol_entropy_sum = query_pos.new_zeros(())
        vol_spread_sum = query_pos.new_zeros(())
        surf_query_total = 0
        vol_query_total = 0
        surf_debug = None
        vol_debug = None
        if return_aux:
            surf_support = surf_query_pos.new_zeros(surf_query_pos.shape[0], surf_count, 1)
            surf_conf = surf_query_pos.new_zeros(surf_query_pos.shape[0], surf_count, 1)
            surf_entropy = surf_query_pos.new_zeros(surf_query_pos.shape[0], surf_count, 1)
            surf_evidence = surf_query_pos.new_zeros(surf_query_pos.shape[0], surf_count, 1)
            surf_raw_gate = surf_query_pos.new_zeros(surf_query_pos.shape[0], surf_count, 1)
            vol_support = vol_query_pos.new_zeros(vol_query_pos.shape[0], vol_count, 1)
            vol_conf = vol_query_pos.new_zeros(vol_query_pos.shape[0], vol_count, 1)
            vol_entropy = vol_query_pos.new_zeros(vol_query_pos.shape[0], vol_count, 1)
            vol_evidence = vol_query_pos.new_zeros(vol_query_pos.shape[0], vol_count, 1)
            vol_raw_gate = vol_query_pos.new_zeros(vol_query_pos.shape[0], vol_count, 1)

        for start in range(0, total_count, self.prediction_query_chunk_size):
            end = min(start + self.prediction_query_chunk_size, total_count)
            query_chunk = query_pos[:, start:end]
            low_query_emb, high_query_emb = self._decode_feature_summary(
                intermediate_latent_geometries,
                latent_geo_pos,
                params,
                query_chunk,
            )
            base_chunk = self.mlp(high_query_emb)

            surf_local_end = max(0, min(end, surf_count) - start)
            if surf_local_end > 0:
                surf_low_chunk, surf_high_chunk, surf_aux = self._readout_branch(
                    low_query_emb[:, :surf_local_end],
                    high_query_emb[:, :surf_local_end],
                    query_chunk[:, :surf_local_end],
                    query_type=0,
                    readout_context=readout_context,
                    only_low=only_low,
                    return_route_debug=return_route_debug and surf_debug is None,
                    route_debug_max_queries=route_debug_max_queries,
                )
                surf_slice = slice(start, start + surf_local_end)
                surf_base[:, surf_slice] = base_chunk[:, :surf_local_end, 0:self.surface_channels]
                surf_low[:, surf_slice] = surf_low_chunk
                surf_high[:, surf_slice] = surf_high_chunk
                chunk_weight = float(max(int(query_chunk.shape[0] * surf_local_end), 1))
                surf_support_sum = surf_support_sum + surf_aux["mean_support_mass"] * chunk_weight
                surf_gate_sum = surf_gate_sum + surf_aux["mean_evidence_gate"] * chunk_weight
                surf_raw_gate_sum = surf_raw_gate_sum + surf_aux["mean_raw_evidence_gate"] * chunk_weight
                surf_conf_sum = surf_conf_sum + surf_aux["mean_route_confidence"] * chunk_weight
                surf_entropy_sum = surf_entropy_sum + surf_aux["mean_route_entropy"] * chunk_weight
                surf_spread_sum = surf_spread_sum + surf_aux["mean_route_spread"] * chunk_weight
                surf_query_total += int(query_chunk.shape[0] * surf_local_end)
                if return_aux:
                    surf_support[:, surf_slice] = surf_aux["support_mass"]
                    surf_conf[:, surf_slice] = surf_aux["route_confidence"]
                    surf_entropy[:, surf_slice] = surf_aux["route_entropy"]
                    surf_evidence[:, surf_slice] = surf_aux["evidence_mass"]
                    surf_raw_gate[:, surf_slice] = surf_aux["raw_gate"]
                if "route_debug" in surf_aux and surf_debug is None:
                    surf_debug = surf_aux["route_debug"]

            vol_local_start = max(0, surf_count - start)
            if vol_local_start < (end - start):
                vol_low_chunk, vol_high_chunk, vol_aux = self._readout_branch(
                    low_query_emb[:, vol_local_start:],
                    high_query_emb[:, vol_local_start:],
                    query_chunk[:, vol_local_start:],
                    query_type=1,
                    readout_context=readout_context,
                    only_low=only_low,
                    return_route_debug=return_route_debug and vol_debug is None,
                    route_debug_max_queries=route_debug_max_queries,
                )
                vol_count_chunk = end - start - vol_local_start
                vol_global_start = max(start - surf_count, 0)
                vol_slice = slice(vol_global_start, vol_global_start + vol_count_chunk)
                vol_base[:, vol_slice] = base_chunk[:, vol_local_start:, self.surface_channels:]
                vol_low[:, vol_slice] = vol_low_chunk
                vol_high[:, vol_slice] = vol_high_chunk
                chunk_weight = float(max(int(query_chunk.shape[0] * vol_count_chunk), 1))
                vol_support_sum = vol_support_sum + vol_aux["mean_support_mass"] * chunk_weight
                vol_gate_sum = vol_gate_sum + vol_aux["mean_evidence_gate"] * chunk_weight
                vol_raw_gate_sum = vol_raw_gate_sum + vol_aux["mean_raw_evidence_gate"] * chunk_weight
                vol_conf_sum = vol_conf_sum + vol_aux["mean_route_confidence"] * chunk_weight
                vol_entropy_sum = vol_entropy_sum + vol_aux["mean_route_entropy"] * chunk_weight
                vol_spread_sum = vol_spread_sum + vol_aux["mean_route_spread"] * chunk_weight
                vol_query_total += int(query_chunk.shape[0] * vol_count_chunk)
                if return_aux:
                    vol_support[:, vol_slice] = vol_aux["support_mass"]
                    vol_conf[:, vol_slice] = vol_aux["route_confidence"]
                    vol_entropy[:, vol_slice] = vol_aux["route_entropy"]
                    vol_evidence[:, vol_slice] = vol_aux["evidence_mass"]
                    vol_raw_gate[:, vol_slice] = vol_aux["raw_gate"]
                if "route_debug" in vol_aux and vol_debug is None:
                    vol_debug = vol_aux["route_debug"]

        surf_residual = self.surface_high_gain.to(dtype=surf_high.dtype) * surf_high
        vol_residual = self.volume_high_gain.to(dtype=vol_high.dtype) * vol_high
        if self.use_low_residual_branch:
            surf_residual = surf_residual + self.surface_low_gain.to(dtype=surf_low.dtype) * surf_low
            vol_residual = vol_residual + self.volume_low_gain.to(dtype=vol_low.dtype) * vol_low

        pred_surf = surf_base + surf_residual
        pred_vol = vol_base + vol_residual

        if not return_aux:
            return pred_surf, pred_vol

        surf_weight = max(surf_query_total, 0)
        vol_weight = max(vol_query_total, 0)
        total_weight = float(max(surf_weight + vol_weight, 1))
        aux = {
            "surface_base": surf_base,
            "volume_base": vol_base,
            "surface_residual": surf_residual,
            "volume_residual": vol_residual,
            "surface_support_mass": surf_support,
            "volume_support_mass": vol_support,
            "surface_route_confidence": surf_conf,
            "volume_route_confidence": vol_conf,
            "surface_route_entropy": surf_entropy,
            "volume_route_entropy": vol_entropy,
            "surface_evidence_mass": surf_evidence,
            "volume_evidence_mass": vol_evidence,
            "surface_raw_evidence_gate": surf_raw_gate,
            "volume_raw_evidence_gate": vol_raw_gate,
            "mean_support_mass": (surf_support_sum + vol_support_sum) / total_weight,
            "mean_evidence_gate": (surf_gate_sum + vol_gate_sum) / total_weight,
            "mean_raw_evidence_gate": (surf_raw_gate_sum + vol_raw_gate_sum) / total_weight,
            "mean_route_confidence": (surf_conf_sum + vol_conf_sum) / total_weight,
            "mean_route_entropy": (surf_entropy_sum + vol_entropy_sum) / total_weight,
            "mean_route_spread": (surf_spread_sum + vol_spread_sum) / total_weight,
        }
        if surf_debug is not None:
            aux["surface_route_debug"] = surf_debug
        if vol_debug is not None:
            aux["volume_route_debug"] = vol_debug
        return pred_surf, pred_vol, aux

    def forward(
        self,
        geo,
        surf_query_pos,
        vol_query_pos,
        params,
        return_aux=False,
        return_latent=False,
        return_route_debug=False,
        route_debug_max_queries=0,
        only_low=False,
        **_unused_kwargs,
    ):
        intermediate_latent_geometries, latent_geo_pos, final_latent_geo = self.encode(geo, params, return_final=True)
        outputs = self.predict_from_encoded(
            intermediate_latent_geometries,
            latent_geo_pos,
            final_latent_geo,
            surf_query_pos,
            vol_query_pos,
            params,
            return_aux=return_aux,
            return_route_debug=return_route_debug,
            route_debug_max_queries=route_debug_max_queries,
            only_low=only_low,
        )

        if return_aux and return_latent:
            pred_surf, pred_vol, aux = outputs
            return pred_surf, pred_vol, aux, final_latent_geo
        if return_latent:
            pred_surf, pred_vol = outputs
            return pred_surf, pred_vol, final_latent_geo
        return outputs

    @torch.inference_mode()
    def inference(
        self,
        geo,
        surf_query_pos,
        vol_query_pos,
        params,
        return_aux=False,
        return_route_debug=False,
        route_debug_max_queries=0,
        only_low=False,
        **_unused_kwargs,
    ):
        intermediate_latent_geometries, latent_geo_pos, final_latent_geo = self.encode(geo, params, return_final=True)

        y_hat_surf_subregions = []
        y_hat_vol_subregions = []
        surf_support_subregions = []
        surf_conf_subregions = []
        surf_entropy_subregions = []
        surf_evidence_subregions = []
        surf_raw_gate_subregions = []
        vol_support_subregions = []
        vol_conf_subregions = []
        vol_entropy_subregions = []
        vol_evidence_subregions = []
        vol_raw_gate_subregions = []
        support_sum = 0.0
        gate_sum = 0.0
        raw_gate_sum = 0.0
        route_conf_sum = 0.0
        route_entropy_sum = 0.0
        route_spread_sum = 0.0
        route_weight_sum = 0
        debug_out = {}

        for i in range(0, surf_query_pos.shape[1], self.subregion_size):
            surf_subregion = surf_query_pos[:, i:i + self.subregion_size]
            if return_aux:
                y_surf, _, aux = self.predict_from_encoded(
                    intermediate_latent_geometries,
                    latent_geo_pos,
                    final_latent_geo,
                    surf_subregion,
                    vol_query_pos[:, :0],
                    params,
                    return_aux=True,
                    return_route_debug=return_route_debug and "surface_route_debug" not in debug_out,
                    route_debug_max_queries=route_debug_max_queries,
                    only_low=only_low,
                )
                y_hat_surf_subregions.append(y_surf)
                surf_support_subregions.append(aux["surface_support_mass"])
                surf_conf_subregions.append(aux["surface_route_confidence"])
                surf_entropy_subregions.append(aux["surface_route_entropy"])
                surf_evidence_subregions.append(aux["surface_evidence_mass"])
                surf_raw_gate_subregions.append(aux["surface_raw_evidence_gate"])
                weight = max(int(surf_subregion.shape[1]), 1)
                support_sum += float(aux["mean_support_mass"].item()) * weight
                gate_sum += float(aux["mean_evidence_gate"].item()) * weight
                raw_gate_sum += float(aux["mean_raw_evidence_gate"].item()) * weight
                route_conf_sum += float(aux["mean_route_confidence"].item()) * weight
                route_entropy_sum += float(aux["mean_route_entropy"].item()) * weight
                route_spread_sum += float(aux["mean_route_spread"].item()) * weight
                route_weight_sum += weight
                if "surface_route_debug" in aux:
                    debug_out["surface_route_debug"] = aux["surface_route_debug"]
            else:
                y_surf, _ = self.predict_from_encoded(
                    intermediate_latent_geometries,
                    latent_geo_pos,
                    final_latent_geo,
                    surf_subregion,
                    vol_query_pos[:, :0],
                    params,
                    return_aux=False,
                    only_low=only_low,
                )
                y_hat_surf_subregions.append(y_surf)

        for i in range(0, vol_query_pos.shape[1], self.subregion_size):
            vol_subregion = vol_query_pos[:, i:i + self.subregion_size]
            if return_aux:
                _, y_vol, aux = self.predict_from_encoded(
                    intermediate_latent_geometries,
                    latent_geo_pos,
                    final_latent_geo,
                    surf_query_pos[:, :0],
                    vol_subregion,
                    params,
                    return_aux=True,
                    return_route_debug=return_route_debug and "volume_route_debug" not in debug_out,
                    route_debug_max_queries=route_debug_max_queries,
                    only_low=only_low,
                )
                y_hat_vol_subregions.append(y_vol)
                vol_support_subregions.append(aux["volume_support_mass"])
                vol_conf_subregions.append(aux["volume_route_confidence"])
                vol_entropy_subregions.append(aux["volume_route_entropy"])
                vol_evidence_subregions.append(aux["volume_evidence_mass"])
                vol_raw_gate_subregions.append(aux["volume_raw_evidence_gate"])
                weight = max(int(vol_subregion.shape[1]), 1)
                support_sum += float(aux["mean_support_mass"].item()) * weight
                gate_sum += float(aux["mean_evidence_gate"].item()) * weight
                raw_gate_sum += float(aux["mean_raw_evidence_gate"].item()) * weight
                route_conf_sum += float(aux["mean_route_confidence"].item()) * weight
                route_entropy_sum += float(aux["mean_route_entropy"].item()) * weight
                route_spread_sum += float(aux["mean_route_spread"].item()) * weight
                route_weight_sum += weight
                if "volume_route_debug" in aux:
                    debug_out["volume_route_debug"] = aux["volume_route_debug"]
            else:
                _, y_vol = self.predict_from_encoded(
                    intermediate_latent_geometries,
                    latent_geo_pos,
                    final_latent_geo,
                    surf_query_pos[:, :0],
                    vol_subregion,
                    params,
                    return_aux=False,
                    only_low=only_low,
                )
                y_hat_vol_subregions.append(y_vol)

        pred_surf = torch.cat(y_hat_surf_subregions, dim=1) if y_hat_surf_subregions else geo.new_zeros(geo.shape[0], 0, self.surface_channels)
        pred_vol = torch.cat(y_hat_vol_subregions, dim=1) if y_hat_vol_subregions else geo.new_zeros(geo.shape[0], 0, self.volume_channels)

        if not return_aux:
            return pred_surf, pred_vol

        aux = {
            "surface_support_mass": torch.cat(surf_support_subregions, dim=1) if surf_support_subregions else pred_surf.new_zeros(pred_surf.shape[0], pred_surf.shape[1], 1),
            "volume_support_mass": torch.cat(vol_support_subregions, dim=1) if vol_support_subregions else pred_vol.new_zeros(pred_vol.shape[0], pred_vol.shape[1], 1),
            "surface_route_confidence": torch.cat(surf_conf_subregions, dim=1) if surf_conf_subregions else pred_surf.new_zeros(pred_surf.shape[0], pred_surf.shape[1], 1),
            "volume_route_confidence": torch.cat(vol_conf_subregions, dim=1) if vol_conf_subregions else pred_vol.new_zeros(pred_vol.shape[0], pred_vol.shape[1], 1),
            "surface_route_entropy": torch.cat(surf_entropy_subregions, dim=1) if surf_entropy_subregions else pred_surf.new_zeros(pred_surf.shape[0], pred_surf.shape[1], 1),
            "volume_route_entropy": torch.cat(vol_entropy_subregions, dim=1) if vol_entropy_subregions else pred_vol.new_zeros(pred_vol.shape[0], pred_vol.shape[1], 1),
            "surface_evidence_mass": torch.cat(surf_evidence_subregions, dim=1) if surf_evidence_subregions else pred_surf.new_zeros(pred_surf.shape[0], pred_surf.shape[1], 1),
            "volume_evidence_mass": torch.cat(vol_evidence_subregions, dim=1) if vol_evidence_subregions else pred_vol.new_zeros(pred_vol.shape[0], pred_vol.shape[1], 1),
            "surface_raw_evidence_gate": torch.cat(surf_raw_gate_subregions, dim=1) if surf_raw_gate_subregions else pred_surf.new_zeros(pred_surf.shape[0], pred_surf.shape[1], 1),
            "volume_raw_evidence_gate": torch.cat(vol_raw_gate_subregions, dim=1) if vol_raw_gate_subregions else pred_vol.new_zeros(pred_vol.shape[0], pred_vol.shape[1], 1),
            "mean_support_mass": pred_surf.new_tensor(support_sum / max(route_weight_sum, 1)),
            "mean_evidence_gate": pred_surf.new_tensor(gate_sum / max(route_weight_sum, 1)),
            "mean_raw_evidence_gate": pred_surf.new_tensor(raw_gate_sum / max(route_weight_sum, 1)),
            "mean_route_confidence": pred_surf.new_tensor(route_conf_sum / max(route_weight_sum, 1)),
            "mean_route_entropy": pred_surf.new_tensor(route_entropy_sum / max(route_weight_sum, 1)),
            "mean_route_spread": pred_surf.new_tensor(route_spread_sum / max(route_weight_sum, 1)),
        }
        aux.update(debug_out)
        return pred_surf, pred_vol, aux
