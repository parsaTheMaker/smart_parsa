from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .family_common import (
    CondInjection,
    init_linear_layer_weights,
    resolve_geo_log_density,
    sample_tokens,
    sample_tokens_density_compensated,
)


def gumbel_softmax(logits, tau):
    u = torch.rand_like(logits)
    g = -torch.log(-torch.log(u + 1e-8) + 1e-8)
    return torch.softmax((logits + g) / tau, dim=-1)


class PhysicsAttention1DEidetic(nn.Module):
    """Local adaptation of the official Transolver++ slice attention."""

    def __init__(self, dim, heads=8, dropout=0.0, slice_num=64, token_chunk_size=None, use_gumbel_routing=False):
        super().__init__()
        assert dim % heads == 0, "dim must be divisible by heads"
        self.dim = dim
        self.heads = heads
        self.dim_head = dim // heads
        self.slice_num = int(slice_num)
        self.token_chunk_size = None if token_chunk_size is None else int(token_chunk_size)
        self.use_gumbel_routing = bool(use_gumbel_routing)

        self.in_project_x = nn.Linear(dim, dim)
        self.in_project_slice = nn.Linear(self.dim_head, self.slice_num)
        torch.nn.init.orthogonal_(self.in_project_slice.weight)
        nn.init.zeros_(self.in_project_slice.bias)

        self.to_q = nn.Linear(self.dim_head, self.dim_head, bias=False)
        self.to_k = nn.Linear(self.dim_head, self.dim_head, bias=False)
        self.to_v = nn.Linear(self.dim_head, self.dim_head, bias=False)
        self.to_out = nn.Sequential(nn.Linear(dim, dim), nn.Dropout(dropout))

        self.bias = nn.Parameter(torch.ones(1, heads, 1, 1) * 0.5)
        self.proj_temperature = nn.Sequential(
            nn.Linear(self.dim_head, self.slice_num),
            nn.GELU(),
            nn.Linear(self.slice_num, 1),
            nn.GELU(),
        )

    def _compute_slice_weights(self, x_mid):
        x_mid = x_mid.float()
        temperature = torch.clamp(self.proj_temperature(x_mid) + self.bias.float(), min=0.01, max=100.0)
        logits = self.in_project_slice(x_mid)
        if self.training and self.use_gumbel_routing:
            return gumbel_softmax(logits, temperature)
        return torch.softmax(logits / temperature, dim=-1)

    def _forward_full(self, x_mid):
        slice_weights = self._compute_slice_weights(x_mid)
        slice_norm = torch.clamp(slice_weights.sum(dim=2), min=1e-6)
        slice_token = torch.einsum("bhnc,bhng->bhgc", x_mid.float(), slice_weights)
        slice_token = slice_token / slice_norm[..., None]

        q = self.to_q(slice_token)
        k = self.to_k(slice_token)
        v = self.to_v(slice_token)
        out_slice = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        out_x = torch.einsum("bhgc,bhng->bhnc", out_slice, slice_weights)
        return out_x

    def _forward_chunked(self, x_mid):
        bsz, _, num_tokens, _ = x_mid.shape
        chunk = int(self.token_chunk_size)
        slice_num = self.slice_num

        slice_norm = torch.zeros((bsz, self.heads, slice_num), device=x_mid.device, dtype=torch.float32)
        slice_token_num = torch.zeros((bsz, self.heads, slice_num, self.dim_head), device=x_mid.device, dtype=torch.float32)

        for start in range(0, num_tokens, chunk):
            stop = min(start + chunk, num_tokens)
            x_chunk = x_mid[:, :, start:stop, :]
            slice_weights = self._compute_slice_weights(x_chunk)
            slice_norm = slice_norm + slice_weights.sum(dim=2)
            slice_token_num = slice_token_num + torch.einsum("bhnc,bhng->bhgc", x_chunk.float(), slice_weights)

        slice_token = slice_token_num / torch.clamp(slice_norm[..., None], min=1e-6)
        q = self.to_q(slice_token)
        k = self.to_k(slice_token)
        v = self.to_v(slice_token)
        out_slice = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)

        out_chunks = []
        for start in range(0, num_tokens, chunk):
            stop = min(start + chunk, num_tokens)
            x_chunk = x_mid[:, :, start:stop, :]
            slice_weights = self._compute_slice_weights(x_chunk)
            out_chunks.append(torch.einsum("bhgc,bhng->bhnc", out_slice, slice_weights))

        return torch.cat(out_chunks, dim=2)

    def forward(self, x):
        bsz, num_tokens, _ = x.shape
        x_mid = self.in_project_x(x).reshape(bsz, num_tokens, self.heads, self.dim_head).permute(0, 2, 1, 3).contiguous()

        if (
            self.token_chunk_size is not None
            and num_tokens > self.token_chunk_size
            and not (self.training and self.use_gumbel_routing)
        ):
            out_x = self._forward_chunked(x_mid)
        else:
            out_x = self._forward_full(x_mid)

        out_x = out_x.permute(0, 2, 1, 3).reshape(bsz, num_tokens, self.dim).to(dtype=x.dtype)
        return self.to_out(out_x)


class TransolverPlusMLP(nn.Module):
    def __init__(self, n_input, n_hidden, n_output, n_layers=0, activation=nn.GELU, residual=False):
        super().__init__()
        self.residual = bool(residual)
        self.linear_pre = nn.Sequential(nn.Linear(n_input, n_hidden), activation())
        self.linears = nn.ModuleList(
            [nn.Sequential(nn.Linear(n_hidden, n_hidden), activation()) for _ in range(n_layers)]
        )
        self.linear_post = nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = self.linear_pre(x)
        for block in self.linears:
            if self.residual:
                x = x + block(x)
            else:
                x = block(x)
        return self.linear_post(x)


class TransolverPlusBlock(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.0, mlp_ratio=2, slice_num=64, use_checkpoint=False, token_chunk_size=None, use_gumbel_routing=False):
        super().__init__()
        self.ln_1 = nn.LayerNorm(dim)
        self.attn = PhysicsAttention1DEidetic(
            dim=dim,
            heads=num_heads,
            dropout=dropout,
            slice_num=slice_num,
            token_chunk_size=token_chunk_size,
            use_gumbel_routing=use_gumbel_routing,
        )
        self.ln_2 = nn.LayerNorm(dim)
        self.mlp = TransolverPlusMLP(dim, dim * mlp_ratio, dim, n_layers=0, residual=False)
        self.use_checkpoint = bool(use_checkpoint)

    def forward(self, x):
        if self.training and self.use_checkpoint:
            x = x + checkpoint(self.attn, self.ln_1(x), use_reentrant=True)
            x = x + checkpoint(self.mlp, self.ln_2(x), use_reentrant=True)
        else:
            x = x + self.attn(self.ln_1(x))
            x = x + self.mlp(self.ln_2(x))
        return x


class TransolverPPBase(nn.Module):
    expects_geo_log_density = False

    def __init__(
        self,
        spatial_dim=3,
        surface_channels=1,
        volume_channels=3,
        parameter_channels=0,
        latent_dim=256,
        latent_geometry_points=4096,
        subsampled_geometry_points=65536,
        num_encoder_decoder_blocks=8,
        num_heads=8,
        pos_scale_factor=100,
        dropout=0.0,
        subregion_size=262144,
        slice_num=64,
        mlp_ratio=2,
        use_checkpoint=False,
        token_chunk_size=8192,
        use_gumbel_routing=False,
        density_compensated=False,
        density_knn_k=8,
        density_neighbor_hops=1,
        density_estimator="rk2",
    ):
        super().__init__()
        self.surface_channels = surface_channels
        self.volume_channels = volume_channels
        self.geometry_token_points = latent_geometry_points if latent_geometry_points > 0 else subsampled_geometry_points
        self.pos_scale_factor = pos_scale_factor
        self.subregion_size = subregion_size
        self.expects_geo_log_density = bool(density_compensated)
        self.density_compensated = bool(density_compensated)
        self.density_knn_k = int(density_knn_k)
        self.density_neighbor_hops = int(density_neighbor_hops)
        self.density_estimator = str(density_estimator)

        # Stay close to the public model structure: preprocess a token tensor x,
        # run the Transolver++ blocks directly on that sequence, and inject
        # external conditioning separately rather than by hard-coded token types.
        self.input_dim = spatial_dim
        self.preprocess = TransolverPlusMLP(self.input_dim, latent_dim * 2, latent_dim, n_layers=0, residual=False)
        self.geometry_preprocess = TransolverPlusMLP(spatial_dim, latent_dim * 2, latent_dim, n_layers=0, residual=False)
        self.placeholder = nn.Parameter((1.0 / latent_dim) * torch.rand(latent_dim, dtype=torch.float))
        self.cond = CondInjection(latent_dim, parameter_channels)
        self.geometry_condition = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim),
        )
        self.blocks = nn.ModuleList(
            [
                TransolverPlusBlock(
                    dim=latent_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    mlp_ratio=mlp_ratio,
                    slice_num=slice_num,
                    use_checkpoint=use_checkpoint,
                    token_chunk_size=token_chunk_size,
                    use_gumbel_routing=use_gumbel_routing,
                )
                for _ in range(num_encoder_decoder_blocks)
            ]
        )
        self.output_head = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, surface_channels + volume_channels),
        )
        self.apply(init_linear_layer_weights)

    def _sample_tokens_strided(self, tokens, num_samples):
        n_points = int(tokens.shape[1])
        if num_samples <= 0 or num_samples >= n_points:
            return tokens
        idx_1d = torch.linspace(
            0,
            n_points - 1,
            steps=num_samples,
            device=tokens.device,
        ).round().to(dtype=torch.long)
        idx = idx_1d.unsqueeze(0).expand(tokens.shape[0], -1)
        return torch.gather(tokens, 1, idx.unsqueeze(-1).expand(-1, -1, tokens.shape[-1]))

    def _select_geometry_tokens(self, geo, geo_log_density=None):
        geo_scaled = geo * self.pos_scale_factor
        if self.density_compensated:
            full_geo_log_density = resolve_geo_log_density(
                geo,
                geo_log_density,
                knn_k=self.density_knn_k,
                neighbor_hops=self.density_neighbor_hops,
                estimator=self.density_estimator,
            )
            geo_tokens, _ = sample_tokens_density_compensated(
                geo_scaled,
                self.geometry_token_points,
                full_geo_log_density,
            )
        else:
            geo_tokens = self._sample_tokens_strided(geo_scaled, self.geometry_token_points)
        return geo_tokens

    def _run_blocks(self, tokens):
        for block in self.blocks:
            tokens = block(tokens)
        return tokens

    def forward(self, geo, surf_query_pos, vol_query_pos, params=None, geo_log_density=None):
        geo_pos = self._select_geometry_tokens(geo, geo_log_density=geo_log_density)
        surf_pos = surf_query_pos * self.pos_scale_factor
        vol_pos = vol_query_pos * self.pos_scale_factor

        geometry_tokens = self.geometry_preprocess(geo_pos)
        geometry_context_input = geometry_tokens.mean(dim=1)
        geometry_condition_token = self.geometry_condition(geometry_context_input.to(dtype=geometry_tokens.dtype)).unsqueeze(1)

        query_pos = torch.cat([surf_pos, vol_pos], dim=1)
        query_tokens = self.preprocess(query_pos)
        query_tokens = query_tokens + self.placeholder.view(1, 1, -1)
        condition_token = geometry_condition_token + self.placeholder.view(1, 1, -1)
        tokens = torch.cat([condition_token, query_tokens], dim=1)
        tokens = self.cond(tokens, params)
        tokens = self._run_blocks(tokens)
        query_latent = tokens[:, 1:]

        surf_count = surf_query_pos.shape[1]
        pred_all = self.output_head(query_latent)
        pred_surf = pred_all[:, :surf_count, : self.surface_channels]
        pred_vol = pred_all[:, surf_count:, self.surface_channels :]
        return pred_surf, pred_vol

    @torch.inference_mode()
    def inference(self, geo, surf_query_pos, vol_query_pos, params=None, geo_log_density=None):
        # Exact inference path: keep the same mixed-token operator used in training.
        return self.forward(geo, surf_query_pos, vol_query_pos, params=params, geo_log_density=geo_log_density)


class TransolverPP(TransolverPPBase):
    def __init__(self, **kwargs):
        super().__init__(density_compensated=False, **kwargs)
