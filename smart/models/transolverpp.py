from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

def gumbel_softmax(logits, tau=1.0):
    u = torch.rand_like(logits)
    noise = -torch.log(-torch.log(u + 1.0e-8) + 1.0e-8)
    return torch.softmax((logits + noise) / tau, dim=-1)


def _distributed_sum(value):
    # The upstream implementation reduces across distributed workers. A
    # single-process run must skip that collective.
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
    return value


class PhysicsAttention1DEidetic(nn.Module):
    """Transolver++ Physics_Attention_1D_Eidetic."""

    def __init__(self, dim, heads=8, dropout=0.0, slice_num=32):
        super().__init__()
        if dim % heads != 0:
            raise ValueError(f"hidden dimension {dim} must be divisible by heads {heads}")
        self.dim_head = dim // heads
        self.heads = int(heads)
        self.slice_num = int(slice_num)
        self.bias = nn.Parameter(torch.ones(1, heads, 1, 1) * 0.5)
        self.proj_temperature = nn.Sequential(
            nn.Linear(self.dim_head, self.slice_num),
            nn.GELU(),
            nn.Linear(self.slice_num, 1),
            nn.GELU(),
        )
        self.in_project_x = nn.Linear(dim, dim)
        self.in_project_slice = nn.Linear(self.dim_head, self.slice_num)
        nn.init.orthogonal_(self.in_project_slice.weight)
        self.to_q = nn.Linear(self.dim_head, self.dim_head, bias=False)
        self.to_k = nn.Linear(self.dim_head, self.dim_head, bias=False)
        self.to_v = nn.Linear(self.dim_head, self.dim_head, bias=False)
        self.to_out = nn.Sequential(nn.Linear(dim, dim), nn.Dropout(dropout))

    def forward(self, x):
        batch_size, num_tokens, _ = x.shape
        # The slice assignment and normalization reduce over tens of thousands
        # of points. Keeping this path in fp32 avoids fp16 underflow/overflow
        # in the backward pass while the surrounding model remains AMP-enabled.
        with torch.autocast(device_type=x.device.type, enabled=False):
            x_float = x.float()
            x_mid = self.in_project_x(x_float).reshape(
                batch_size, num_tokens, self.heads, self.dim_head
            ).permute(0, 2, 1, 3).contiguous()

            temperature = self.proj_temperature(x_mid) + self.bias
            temperature = torch.clamp(temperature, min=0.01)
            slice_weights = gumbel_softmax(self.in_project_slice(x_mid), temperature)
            slice_norm = _distributed_sum(slice_weights.sum(dim=2))
            slice_token = torch.einsum("bhnc,bhng->bhgc", x_mid, slice_weights).contiguous()
            slice_token = _distributed_sum(slice_token)
            slice_token = slice_token / (slice_norm[..., None] + 1.0e-5)

            q = self.to_q(slice_token)
            k = self.to_k(slice_token)
            v = self.to_v(slice_token)
            out_slice = F.scaled_dot_product_attention(q, k, v)
            out_x = torch.einsum("bhgc,bhng->bhnc", out_slice, slice_weights)
            out_x = out_x.permute(0, 2, 1, 3).reshape(batch_size, num_tokens, -1)
            output = self.to_out(out_x)
        # Keep the residual stream in fp32 after attention. Casting it back to
        # fp16 here lets GradScaler overflow the backward path at large scales.
        return output


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
        for layer in self.linears:
            update = layer(x)
            x = x + update if self.residual else update
        return self.linear_post(x)


class TransolverPlusBlock(nn.Module):
    def __init__(
        self,
        num_heads,
        hidden_dim,
        dropout,
        mlp_ratio=2,
        last_layer=False,
        out_dim=1,
        slice_num=32,
    ):
        super().__init__()
        self.last_layer = bool(last_layer)
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.attn = PhysicsAttention1DEidetic(
            hidden_dim, heads=num_heads, dropout=dropout, slice_num=slice_num
        )
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.mlp = TransolverPlusMLP(
            hidden_dim, hidden_dim * mlp_ratio, hidden_dim, n_layers=0, residual=False
        )
        if self.last_layer:
            self.ln_3 = nn.LayerNorm(hidden_dim)
            self.output = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, return_hidden=False):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        if not self.last_layer:
            return x
        # Keep the supervised output head in fp32. Relative field losses can
        # produce large gradients for low-energy channels, which can overflow
        # when the prediction is first materialized as fp16 under AMP.
        with torch.autocast(device_type=x.device.type, enabled=False):
            prediction = self.output(self.ln_3(x.float()))
        return (x, prediction) if return_hidden else prediction


class TransolverPPBase(nn.Module):
    """Official Transolver++ blocks adapted to arbitrary query locations.

    The upstream solver predicts at the same locations supplied to its token
    sequence. For DrivAerML, geometry support tokens are prepended to surface
    and volume query tokens, and only the query-token outputs are returned.
    """

    expects_geo_log_density = False

    def __init__(
        self,
        spatial_dim=3,
        surface_channels=1,
        volume_channels=3,
        parameter_channels=0,
        n_layers=4,
        n_hidden=256,
        n_heads=8,
        dropout=0.1,
        mlp_ratio=2,
        slice_num=32,
        geometry_points=0,
    ):
        super().__init__()
        if parameter_channels:
            raise ValueError("The DrivAerML Transolver++ adapter does not use parameter channels.")
        self.surface_channels = int(surface_channels)
        self.volume_channels = int(volume_channels)
        self.spatial_dim = int(spatial_dim)
        self.geometry_points = int(geometry_points)
        self.preprocess = TransolverPlusMLP(
            self.spatial_dim, n_hidden * 2, n_hidden, n_layers=0, residual=False
        )
        self.placeholder = nn.Parameter((1.0 / n_hidden) * torch.rand(n_hidden))
        self.blocks = nn.ModuleList(
            [
                TransolverPlusBlock(
                    num_heads=n_heads,
                    hidden_dim=n_hidden,
                    dropout=dropout,
                    mlp_ratio=mlp_ratio,
                    last_layer=index == n_layers - 1,
                    out_dim=self.surface_channels + self.volume_channels,
                    slice_num=slice_num,
                )
                for index in range(int(n_layers))
            ]
        )
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)

    def _select_geometry(self, geo, geo_log_density=None):
        del geo_log_density
        if self.geometry_points <= 0 or self.geometry_points >= geo.shape[1]:
            return geo
        indices = torch.stack(
            [torch.randperm(geo.shape[1], device=geo.device)[: self.geometry_points] for _ in range(geo.shape[0])],
            dim=0,
        )
        return torch.gather(geo, 1, indices.unsqueeze(-1).expand(-1, -1, geo.shape[-1]))

    def forward(
        self,
        geo,
        surf_query_pos,
        vol_query_pos,
        params=None,
        geo_log_density=None,
        return_latent=False,
    ):
        if params is not None:
            raise ValueError("The DrivAerML Transolver++ adapter does not use parameter channels.")
        geometry_pos = self._select_geometry(geo, geo_log_density=geo_log_density)
        query_pos = torch.cat([surf_query_pos, vol_query_pos], dim=1)
        # Start the residual stream in fp32 so AMP does not introduce a
        # low-precision cast before the first physics-attention block.
        with torch.autocast(device_type=geo.device.type, enabled=False):
            tokens = self.preprocess(torch.cat([geometry_pos, query_pos], dim=1).float())
            tokens = tokens + self.placeholder.view(1, 1, -1)

        hidden = None
        prediction = None
        for block_index, block in enumerate(self.blocks):
            if block_index == len(self.blocks) - 1:
                hidden, prediction = block(tokens, return_hidden=True)
            else:
                tokens = block(tokens)
        query_prediction = prediction[:, geometry_pos.shape[1] :]
        surf_count = surf_query_pos.shape[1]
        pred_surf = query_prediction[:, :surf_count, : self.surface_channels]
        pred_vol = query_prediction[:, surf_count:, self.surface_channels :]
        if return_latent:
            return pred_surf, pred_vol, hidden[:, geometry_pos.shape[1] :]
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


class TransolverPP(TransolverPPBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
