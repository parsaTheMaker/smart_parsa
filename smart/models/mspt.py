"""MSPT adapter for the SMART DrivAerML surface/volume interface.

The attention, pooled-supernode update, block ordering, and point restoration
follow the official unstructured MSPT implementation.  The only adapter logic
is concatenating geometry support points with surface/volume query points and
splitting the final pointwise head back into SMART's two outputs.
"""

from __future__ import annotations

import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

try:
    from flash_attn.flash_attn_interface import flash_attn_func
except Exception:
    flash_attn_func = None


ACTIVATIONS = {
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "relu": nn.ReLU,
    "leaky_relu": lambda: nn.LeakyReLU(0.1),
    "softplus": nn.Softplus,
    "ELU": nn.ELU,
    "silu": nn.SiLU,
}

_BALLTREE_FALLBACK_WARNED = False


def _rotate_half(x):
    first = x[..., ::2]
    second = x[..., 1::2]
    return torch.stack((-second, first), dim=-1).reshape_as(x)


def _rope_cache(seq_len, head_dim, device, dtype, base=10000.0):
    if head_dim % 2 != 0:
        raise ValueError(f"MSPT RoPE requires an even head dimension, got {head_dim}")
    half_dim = head_dim // 2
    frequency_index = torch.arange(half_dim, device=device, dtype=torch.float32)
    inverse_frequency = base ** (-frequency_index / half_dim)
    positions = torch.arange(seq_len, device=device, dtype=torch.float32)
    frequencies = torch.einsum("i,j->ij", positions, inverse_frequency)
    cos = torch.stack((frequencies.cos(), frequencies.cos()), dim=-1).reshape(seq_len, head_dim)
    sin = torch.stack((frequencies.sin(), frequencies.sin()), dim=-1).reshape(seq_len, head_dim)
    return cos.to(dtype=dtype)[None, None], sin.to(dtype=dtype)[None, None]


def _apply_rope(q, k, cos, sin):
    return (q * cos) + (_rotate_half(q) * sin), (k * cos) + (_rotate_half(k) * sin)


def _partition_spatial_tree(points, num_chunks):
    """Fallback spatial partition with the same contiguous-patch contract."""
    if num_chunks <= 0 or num_chunks & (num_chunks - 1):
        raise ValueError("MSPT spatial partition requires a power-of-two patch count.")
    groups = [torch.arange(points.shape[0], device=points.device)]
    while len(groups) < num_chunks:
        next_groups = []
        for group in groups:
            if group.numel() <= 1:
                raise ValueError("Not enough points to build the requested MSPT patches.")
            group_points = points[group]
            spread = group_points.max(dim=0).values - group_points.min(dim=0).values
            split_dim = int(torch.argmax(spread).item())
            order = torch.argsort(group_points[:, split_dim], stable=True)
            midpoint = group.numel() // 2
            next_groups.extend([group[order[:midpoint]], group[order[midpoint:]]])
        groups = next_groups
    return torch.cat(groups, dim=0)


def _partition_balltree(points, num_chunks):
    """Use the official balltree-erwin partitioner when installed.

    The repository's dependency is optional in this project.  The deterministic
    spatial-tree fallback preserves MSPT's patch locality and shape contract
    when balltree-erwin is unavailable.
    """
    global _BALLTREE_FALLBACK_WARNED
    try:
        from balltree import partition_balltree

        batch_index = torch.zeros(points.shape[0], dtype=torch.long, device=points.device)
        target_level = max(0, math.ceil(math.log2(num_chunks)))
        partition = partition_balltree(points, batch_index, target_level).long()
        if partition.numel() >= points.shape[0]:
            return partition[: points.shape[0]].to(device=points.device)
    except ImportError:
        pass

    if not _BALLTREE_FALLBACK_WARNED:
        warnings.warn(
            "balltree-erwin is unavailable; MSPT is using its deterministic "
            "spatial-tree patch fallback.",
            RuntimeWarning,
            stacklevel=2,
        )
        _BALLTREE_FALLBACK_WARNED = True
    return _partition_spatial_tree(points, num_chunks)


class MSPTMLP(nn.Module):
    def __init__(self, n_input, n_hidden, n_output, n_layers=1, act="gelu", residual=True):
        super().__init__()
        if act not in ACTIVATIONS:
            raise ValueError(f"Unsupported MSPT activation: {act}")
        activation = ACTIVATIONS[act]
        self.residual = bool(residual)
        self.linear_pre = nn.Sequential(nn.Linear(n_input, n_hidden), activation())
        self.linears = nn.ModuleList(
            [nn.Sequential(nn.Linear(n_hidden, n_hidden), activation()) for _ in range(int(n_layers))]
        )
        self.linear_post = nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = self.linear_pre(x)
        for layer in self.linears:
            update = layer(x)
            x = x + update if self.residual else update
        return self.linear_post(x)


class ChunkedGlobalPoolAttention(nn.Module):
    """Official MSPT parallelized multi-scale attention block."""

    def __init__(
        self,
        dim,
        heads=8,
        V=16,
        Q=1,
        dropout=0.1,
        pool="mean",
        use_rope=False,
        rope_base=10000.0,
        use_flash_attn=False,
    ):
        super().__init__()
        self.dim = int(dim)
        self.heads = int(heads)
        self.V = int(V)
        self.Q = int(Q)
        self.pool = str(pool)
        self.use_rope = bool(use_rope)
        self.rope_base = float(rope_base)
        self.use_flash_attn = bool(use_flash_attn and flash_attn_func is not None)
        if self.pool == "linear":
            self.pool_proj = nn.Linear(self.dim, self.Q * self.dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=self.dim,
            num_heads=self.heads,
            dropout=dropout,
            batch_first=True,
        )
        if self.use_rope or self.use_flash_attn:
            if self.dim % self.heads != 0:
                raise ValueError(f"MSPT attention dimension {self.dim} must divide heads {self.heads}")
            self.head_dim = self.dim // self.heads
        self.norm = nn.LayerNorm(self.dim)
        self.ff = nn.Sequential(
            nn.Linear(self.dim, 4 * self.dim),
            nn.GELU(),
            nn.Linear(4 * self.dim, self.dim),
            nn.Dropout(dropout),
        )

    @staticmethod
    def _pad_to_multiple(x, multiple, dim=1):
        length = x.size(dim)
        pad_len = (multiple - (length % multiple)) % multiple
        if pad_len == 0:
            return x, 0
        pad_shape = list(x.shape)
        pad_shape[dim] = pad_len
        return torch.cat([x, x.new_zeros(pad_shape)], dim=dim), pad_len

    def _pool(self, chunks):
        batch_size, _num_chunks, seq_len, _dim = chunks.shape
        if self.pool == "mean":
            if self.Q == 1:
                pooled = chunks.mean(dim=2, keepdim=True)
            else:
                k = min(self.Q, seq_len)
                norms = chunks.norm(dim=-1)
                order = torch.argsort(norms, dim=2, descending=True)
                running_sum = chunks.sum(dim=2)
                counts = torch.full(
                    (batch_size, self.V),
                    seq_len,
                    device=chunks.device,
                    dtype=chunks.dtype,
                )
                means = []
                for q_index in range(k):
                    means.append((running_sum / counts.unsqueeze(-1)).unsqueeze(2))
                    if q_index == k - 1:
                        break
                    selected_index = order[:, :, q_index].unsqueeze(-1).unsqueeze(-1).expand(
                        -1, -1, 1, chunks.shape[-1]
                    )
                    selected = torch.gather(chunks, 2, selected_index).squeeze(2)
                    running_sum = running_sum - selected
                    counts = counts - 1
                pooled = torch.cat(means, dim=2)
                if k < self.Q:
                    pooled = torch.cat(
                        [
                            pooled,
                            chunks.new_zeros(batch_size, self.V, self.Q - k, self.dim),
                        ],
                        dim=2,
                    )
        elif self.pool == "max":
            k = min(self.Q, seq_len)
            pooled, _ = chunks.topk(k=k, dim=2)
            if k < self.Q:
                pooled = torch.cat(
                    [pooled, chunks.new_zeros(batch_size, self.V, self.Q - k, self.dim)], dim=2
                )
        elif self.pool == "linear":
            pooled = self.pool_proj(chunks.mean(dim=2)).view(batch_size, self.V, self.Q, self.dim)
        else:
            raise ValueError(f"Unsupported MSPT pooling mode: {self.pool}")
        if pooled.size(2) == 1:
            pooled = pooled.expand(batch_size, self.V, self.Q, self.dim)
        return pooled

    def forward(self, features, prev_supernodes=None):
        batch_size, num_points, _ = features.shape
        x, pad_len = self._pad_to_multiple(features, self.V, dim=1)
        padded_points = x.size(1)
        seq_len = padded_points // self.V
        chunks = x.view(batch_size, self.V, seq_len, self.dim)

        pooled = self._pool(chunks)
        global_tokens = pooled.reshape(batch_size, self.V * self.Q, self.dim)
        if prev_supernodes is not None:
            if prev_supernodes.shape != global_tokens.shape:
                raise ValueError(
                    f"MSPT supernode shape {tuple(prev_supernodes.shape)} does not match "
                    f"{tuple(global_tokens.shape)}"
                )
            global_tokens = global_tokens + prev_supernodes.to(
                device=global_tokens.device, dtype=global_tokens.dtype
            )

        expanded_global = global_tokens.unsqueeze(1).expand(-1, self.V, -1, -1)
        sequence = torch.cat([chunks, expanded_global], dim=2)
        sequence = sequence.view(batch_size * self.V, seq_len + self.V * self.Q, self.dim)
        residual = sequence
        sequence = self.norm(sequence)
        attention_out = self._self_attention(sequence)
        sequence = residual + attention_out
        sequence = sequence + self.ff(self.norm(sequence))
        sequence = sequence.view(batch_size, self.V, seq_len + self.V * self.Q, self.dim)

        point_features = sequence[:, :, :seq_len, :].reshape(batch_size, padded_points, self.dim)
        if pad_len > 0:
            point_features = point_features[:, :-pad_len, :]
        supernodes = sequence[:, :, -self.V * self.Q :, :].mean(dim=1)
        return point_features, supernodes

    def _self_attention(self, sequence):
        if not (self.use_rope or self.use_flash_attn):
            attention_out, _ = self.attn(sequence, sequence, sequence, need_weights=False)
            return attention_out

        batch_size, sequence_length, _ = sequence.shape
        qkv = F.linear(sequence, self.attn.in_proj_weight, self.attn.in_proj_bias)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(batch_size, sequence_length, self.heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, sequence_length, self.heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, sequence_length, self.heads, self.head_dim).transpose(1, 2)
        if self.use_rope:
            cos, sin = _rope_cache(
                sequence_length,
                self.head_dim,
                sequence.device,
                sequence.dtype,
                base=self.rope_base,
            )
            q, k = _apply_rope(q, k, cos, sin)
        dropout_probability = self.attn.dropout if self.training else 0.0
        if self.use_flash_attn and sequence.is_cuda:
            output = flash_attn_func(
                q,
                k,
                v,
                dropout_p=dropout_probability,
                softmax_scale=None,
                causal=False,
            )
        else:
            output = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=dropout_probability,
            )
        output = output.transpose(1, 2).reshape(batch_size, sequence_length, self.dim)
        return F.linear(output, self.attn.out_proj.weight, self.attn.out_proj.bias)


class MSPTBlock(nn.Module):
    def __init__(
        self,
        num_heads,
        hidden_dim,
        dropout,
        act="gelu",
        mlp_ratio=1,
        last_layer=False,
        out_dim=1,
        V=32,
        Q=1,
        attn_pool="mean",
        use_checkpoint=True,
        use_rope=False,
        rope_base=10000.0,
        use_flash_attn=False,
    ):
        super().__init__()
        self.last_layer = bool(last_layer)
        self.use_checkpoint = bool(use_checkpoint)
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.Attn = ChunkedGlobalPoolAttention(
            dim=hidden_dim,
            heads=num_heads,
            V=V,
            Q=Q,
            dropout=dropout,
            pool=attn_pool,
            use_rope=use_rope,
            rope_base=rope_base,
            use_flash_attn=use_flash_attn,
        )
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.mlp = MSPTMLP(
            hidden_dim,
            hidden_dim * int(mlp_ratio),
            hidden_dim,
            n_layers=0,
            act=act,
            residual=False,
        )
        if self.last_layer:
            self.ln_3 = nn.LayerNorm(hidden_dim)
            self.mlp2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, features, supernodes=None):
        if supernodes is None:
            supernodes = features.new_zeros(
                features.shape[0], self.Attn.V * self.Attn.Q, self.Attn.dim
            )
        attention_input = self.ln_1(features)

        def attention_forward(inputs, previous_supernodes):
            return self.Attn(inputs, previous_supernodes)

        if self.training and self.use_checkpoint:
            attention_out, supernodes = checkpoint(
                attention_forward,
                attention_input,
                supernodes,
                use_reentrant=False,
            )
            features = features + attention_out
            features = features + checkpoint(
                self.mlp,
                self.ln_2(features),
                use_reentrant=False,
            )
        else:
            attention_out, supernodes = self.Attn(attention_input, supernodes)
            features = features + attention_out
            features = features + self.mlp(self.ln_2(features))

        if self.last_layer:
            return self.mlp2(self.ln_3(features)), supernodes
        return features, supernodes


class MSPT(nn.Module):
    """Unstructured MSPT adapted for SMART's surface and volume queries."""

    expects_geo_log_density = False

    def __init__(
        self,
        spatial_dim=3,
        surface_channels=1,
        volume_channels=3,
        parameter_channels=0,
        num_blocks=6,
        n_hidden=256,
        num_heads=8,
        dropout=0.1,
        activation="gelu",
        mlp_ratio=1,
        V=32,
        Q=1,
        attn_pool="mean",
        chunking_mode="balltree",
        use_checkpoint=True,
        use_rope=False,
        rope_base=10000.0,
        use_flash_attn=False,
        use_token_type_embeddings=False,
    ):
        super().__init__()
        if parameter_channels:
            raise ValueError("MSPT DrivAerML adapter does not use parameter channels.")
        if n_hidden % num_heads != 0:
            raise ValueError(f"n_hidden={n_hidden} must be divisible by num_heads={num_heads}")
        if V <= 0 or V & (V - 1):
            raise ValueError("MSPT V must be a positive power of two.")
        self.spatial_dim = int(spatial_dim)
        self.surface_channels = int(surface_channels)
        self.volume_channels = int(volume_channels)
        self.V = int(V)
        self.Q = int(Q)
        self.chunking_mode = str(chunking_mode).lower()
        self.use_token_type_embeddings = bool(use_token_type_embeddings)
        if self.chunking_mode not in {"linear", "balltree"}:
            raise ValueError("MSPT chunking_mode must be 'linear' or 'balltree'.")
        self.subsampled_geometry_points = 0

        self.preprocess = MSPTMLP(
            self.spatial_dim,
            int(n_hidden) * 2,
            int(n_hidden),
            n_layers=0,
            act=activation,
            residual=False,
        )
        self.blocks = nn.ModuleList(
            [
                MSPTBlock(
                    num_heads=num_heads,
                    hidden_dim=n_hidden,
                    dropout=dropout,
                    act=activation,
                    mlp_ratio=mlp_ratio,
                    last_layer=index == int(num_blocks) - 1,
                    out_dim=self.surface_channels + self.volume_channels,
                    V=V,
                    Q=Q,
                    attn_pool=attn_pool,
                    use_checkpoint=use_checkpoint,
                    use_rope=use_rope,
                    rope_base=rope_base,
                    use_flash_attn=use_flash_attn,
                )
                for index in range(int(num_blocks))
            ]
        )
        # The original MSPT operates on one homogeneous point set. Our
        # geometry-to-field adapter instead joins geometry support, surface
        # queries, and volume queries. Segment embeddings preserve that role
        # information through the spatial reordering required by MSPT.
        self.token_type_embedding = (
            nn.Embedding(3, int(n_hidden)) if self.use_token_type_embeddings else None
        )
        self.initialize_weights()
        if self.token_type_embedding is not None:
            nn.init.normal_(self.token_type_embedding.weight, std=0.02)
        self.placeholder = nn.Parameter((1.0 / n_hidden) * torch.rand(n_hidden))

    def initialize_weights(self):
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, (nn.LayerNorm, nn.BatchNorm1d)):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    def _partition(self, positions):
        batch_size, num_points, _ = positions.shape
        if self.chunking_mode == "linear":
            # The official linear path leaves preprocessing unpadded; the
            # attention block pads hidden features only after this MLP.
            identity = torch.arange(num_points, device=positions.device)
            permutation = identity.unsqueeze(0).expand(batch_size, -1)
            return positions, permutation, num_points

        padded_points = math.ceil(num_points / self.V) * self.V
        pad_len = padded_points - num_points
        if pad_len:
            positions = torch.cat(
                [positions, positions.new_zeros(batch_size, pad_len, positions.shape[-1])], dim=1
            )

        permutations = []
        for batch_index in range(batch_size):
            permutations.append(_partition_balltree(positions[batch_index], self.V))
        permutation = torch.stack(permutations, dim=0)
        inverse = torch.empty_like(permutation)
        inverse.scatter_(1, permutation, torch.arange(padded_points, device=positions.device).expand(batch_size, -1))
        chunked = torch.gather(positions, 1, permutation.unsqueeze(-1).expand(-1, -1, positions.shape[-1]))
        return chunked, inverse, num_points

    @staticmethod
    def _restore(features, inverse, original_points):
        restored = torch.gather(features, 1, inverse.unsqueeze(-1).expand(-1, -1, features.shape[-1]))
        return restored[:, :original_points]

    def forward(
        self,
        geo,
        surf_query_pos,
        vol_query_pos,
        params=None,
        geo_log_density=None,
        return_latent=False,
    ):
        del geo_log_density
        if params is not None:
            raise ValueError("MSPT DrivAerML adapter does not use parameter channels.")
        geometry_count = int(geo.shape[1])
        surface_count = int(surf_query_pos.shape[1])
        all_positions = torch.cat([geo, surf_query_pos, vol_query_pos], dim=1)
        chunked_positions, inverse, original_points = self._partition(all_positions)
        features = self.preprocess(chunked_positions)
        if self.token_type_embedding is not None:
            batch_size = int(geo.shape[0])
            role_ids = torch.cat(
                [
                    torch.zeros((batch_size, geometry_count), device=geo.device, dtype=torch.long),
                    torch.ones((batch_size, surface_count), device=geo.device, dtype=torch.long),
                    torch.full(
                        (batch_size, int(vol_query_pos.shape[1])),
                        2,
                        device=geo.device,
                        dtype=torch.long,
                    ),
                ],
                dim=1,
            )
            # ``inverse`` maps original-token indices to the spatially
            # reordered MSPT sequence. Pad tokens, when present, stay geometry
            # type zero and are discarded by ``_restore`` later.
            if inverse.shape[1] > role_ids.shape[1]:
                role_ids = F.pad(role_ids, (0, inverse.shape[1] - role_ids.shape[1]))
            chunked_role_ids = torch.zeros_like(inverse)
            chunked_role_ids.scatter_(1, inverse, role_ids)
            features = features + self.token_type_embedding(chunked_role_ids).to(dtype=features.dtype)
        features = features + self.placeholder.view(1, 1, -1)

        supernodes = None
        for block in self.blocks:
            features, supernodes = block(features, supernodes)
        outputs = self._restore(features, inverse, original_points)
        query_outputs = outputs[:, geometry_count:]
        pred_surf = query_outputs[:, :surface_count, : self.surface_channels]
        pred_vol = query_outputs[:, surface_count:, self.surface_channels :]
        if return_latent:
            return pred_surf, pred_vol, query_outputs
        return pred_surf, pred_vol

    @torch.inference_mode()
    def inference(self, geo, surf_query_pos, vol_query_pos, params=None, geo_log_density=None):
        return self.forward(
            geo,
            surf_query_pos,
            vol_query_pos,
            params=params,
            geo_log_density=geo_log_density,
        )
