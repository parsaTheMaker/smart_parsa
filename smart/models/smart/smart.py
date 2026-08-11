"""SMART: Scalable Mesh-free Aerodynamic Simulations from Raw Geometries using a Transformer-based Surrogate Model

This module contains the implementation of the encoder and decoder blocks used in SMART, as well as the complete SMART model.
Designed for simulating time-independent PDEs over complex 3D geometries, SMART leverages a Transformer-based architecture
to perform simulations using solely inexpensive geometry meshes, eliminating the need for costly surface and volumetric
meshing during inference.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class ModulatedPositionalEmbedding(nn.Module):
    """Embedding layer that applies a modulated sine-cosine positional embedding to the spatial positions. This means 
    that the sine-cosine functions are shifted and scaled based on learned parameters from an MLP for each position.
    This allows the model to adaptively adjust the positional embeddings based on the data to emphasize or suppress
    high-frequency variations.

    The implementation follows the original Transformer positional embedding (https://arxiv.org/abs/1706.03762) and
    the class is based on the PositionalEncoding class from the PyTorch tutorial 'Language Modeling with nn.Transformer and torchtext'.
    
    Args:
        dim: Dimensionality of the embedded positions.
        spatial_dim: The spatial dimensionality of the positions (e.g., 2 for 2D positions, 3 for 3D positions). Defaults to 3.
        max_seq_length: Max sequence length. Defaults to 10000 as suggested in the original Transformer paper.
    """

    def __init__(self, dim, spatial_dim=3, max_seq_length=10000):
        super().__init__()
        self.dim = dim
        self.spatial_dim = spatial_dim
        
        # Compute dimensions per spatial dimension
        max_dim_per_spatial_dim = dim // spatial_dim
        dim_per_spatial_dim = max_dim_per_spatial_dim & ~1 # This is equal to (max_dim_per_spatial_dim // 2) * 2
        self.dim_per_spatial_dim = dim_per_spatial_dim
        
        # Compute the total padding
        self.total_padding = dim - (dim_per_spatial_dim * spatial_dim)
        self.register_buffer("padding", torch.zeros(1, 1, self.total_padding))
        
        # Compute the div_term for sine-cosine embedding
        div_term = torch.exp(torch.arange(0, dim_per_spatial_dim, 2) * (-math.log(max_seq_length) / dim_per_spatial_dim))
        self.register_buffer("div_term", div_term)
        
        # Modulation MLP
        self.mlp = nn.Sequential(nn.Linear(dim, dim*4), nn.GELU(), nn.Linear(dim*4, dim_per_spatial_dim * spatial_dim * 2))

    def compute_embedding(self, pos, shift_sin=None, scale_sin=None, shift_cos=None, scale_cos=None):
        # Following UPT (https://arxiv.org/abs/2402.12365) and compute positional embeddings in float32 to avoid numerical instabilities
        with torch.autocast(device_type=str(pos.device).split(":")[0], enabled=False):
            pos = pos.float()
            sin_cos_arg = pos[..., None] @ self.div_term[None, ...]
            
            embedding = torch.zeros((*sin_cos_arg.shape[:-1], self.dim_per_spatial_dim), device=sin_cos_arg.device, dtype=sin_cos_arg.dtype)
            # Apply shift and scale to embedding if provided
            if shift_sin is not None and scale_sin is not None and shift_cos is not None and scale_cos is not None:
                embedding[..., 0::2] = scale_sin * torch.sin(sin_cos_arg + shift_sin)
                embedding[..., 1::2] = scale_cos * torch.cos(sin_cos_arg + shift_cos)
            else:
                embedding[..., 0::2] = torch.sin(sin_cos_arg)
                embedding[..., 1::2] = torch.cos(sin_cos_arg)
            
        # Rearrange spatial dimensions
        embedding = rearrange(embedding, "b n spatial_dim d -> b n (spatial_dim d)")
        
        # Apply padding if necessary
        if self.total_padding > 0: embedding = torch.concat([embedding, self.padding.expand(*embedding.shape[:-1], -1)], dim=-1)
        
        return embedding
        
    def forward(self, pos):
        """Embeds the positions, normalized to [0, max_seq_length], using modulated sine-cosine positional embeddings.

        Args:
            pos: Normalized positions with shape (batch size, number points, spatial_dim).

        Returns:
            Embedded positions with shape (batch size, number points, dim).
        """
        initial_embedding = self.compute_embedding(pos)
        
        # Apply modulation MLP for shift and scaling
        shift_scale = self.mlp(initial_embedding)
        shift_sin, scale_sin, shift_cos, scale_cos = torch.unbind(rearrange(shift_scale, "b n (d shift_scale spatial_dim) -> b n spatial_dim d shift_scale", shift_scale=4, spatial_dim=self.spatial_dim), -1)
        
        embedding = self.compute_embedding(pos, shift_sin=shift_sin, scale_sin=scale_sin, shift_cos=shift_cos, scale_cos=scale_cos)
        
        return embedding


class RotaryPositionalEmbedding(nn.Module):
    """Rotary Positional Embedding (RoPE; https://arxiv.org/abs/2104.09864) for spatial positions.
    
    Args:
        dim: Dimensionality of the features to be embedded.
        spatial_dim: The spatial dimensionality of the positions (e.g., 2 for 2D positions, 3 for 3D positions). Defaults to 3.
        max_seq_length: Max sequence length. Defaults to 10000 as suggested in the original RoPE paper.
    """
    
    def __init__(self, dim, spatial_dim, max_seq_length=10000.0):
        super().__init__()
        assert dim % 2 == 0, "dim must be even for rotary embeddings"
        
        self.dim = dim
        self.spatial_dim = spatial_dim
        
        # Compute dimensions per spatial dimension
        max_dim_per_spatial_dim = dim // spatial_dim
        dim_per_spatial_dim = max_dim_per_spatial_dim & ~1 # This is equal to (max_dim_per_spatial_dim // 2) * 2
        
        # Compute the padding
        self.total_padding = dim - (dim_per_spatial_dim * spatial_dim)
        self.register_buffer("padding", torch.zeros(1, 1, self.total_padding // 2))
        
        # Compute the div_term for sine-cosine embedding
        div_term = torch.exp(torch.arange(0, dim_per_spatial_dim, 2) * (-math.log(max_seq_length) / dim_per_spatial_dim))
        self.register_buffer("div_term", div_term)

    def forward(self, x, pos):
        """Applies RoPE to the features x based on the positions pos.
        
        Args:
            x: Features to apply RoPE to with shape (batch size, number points, dim).
            pos: Normalized positions with shape (batch size, number points, spatial_dim).
            
        Returns:
            Features with RoPE applied to with shape (batch size, number points, dim).
        """
        # Following UPT (https://arxiv.org/abs/2402.12365) and compute positional embeddings in float32 to avoid numerical instabilities
        with torch.autocast(device_type=str(pos.device).split(":")[0], enabled=False):
            pos = pos.float()
            theta = pos[..., None] @ self.div_term[None, ...]
        
        theta = rearrange(theta, "b n spatial_dim d -> b n (spatial_dim d)")
        
        # Add padding
        theta = torch.concat([theta, self.padding.expand(*theta.shape[:-1], -1)], dim=-1)
        
        # Apply rotation matrix in complex space following Llama 3 implementation
        # (https://github.com/meta-llama/llama3/blob/a0940f9cf7065d45bb6675660f80d305c041a754/llama/model.py#L65)
        rotation = torch.polar(torch.ones_like(theta), theta)
        x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        embedded = torch.view_as_real(x_complex * rotation[:, None, ...]).flatten(3)
        
        return embedded.type_as(x)
    

class CrossAttention(nn.Module):
    """Computes multi-head cross-attention (https://arxiv.org/abs/1706.03762) between the query and key/value sequences. It
    optionally applies Rotary Positional Embedding (RoPE; https://arxiv.org/abs/2104.09864) to both the query and key features.

    Args:
        dim: Dimensionality of the query and key/value features.
        num_heads: Number of attention heads. Defaults to 8.
        spatial_dim: Number of spatial dimensions for RoPE. Defaults to 3.
        dropout: Dropout rate. Defaults to 0.1.
    """

    def __init__(self, dim, num_heads=8, spatial_dim=3, dropout=0.1):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        # Pre-layer normalization
        self.norm_q = nn.LayerNorm(dim, eps=1e-6)
        self.norm_kv = nn.LayerNorm(dim, eps=1e-6)
        
        # Projections
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.out_proj = nn.Linear(dim, dim)
        
        # RoPE
        self.rope = RotaryPositionalEmbedding(dim=dim // num_heads, spatial_dim=spatial_dim)
        
        self.dropout = dropout

    def forward(self, q, kv, q_pos=None, kv_pos=None):
        """Applies pre-norm and computes cross-attention between q and kv.

        Args:
            q: Queries with shape (batch size, number query tokens, dim).
            kv: Key/value with shape (batch size, number key/value tokens, dim).
            q_pos (optional): Positions for the queries with shape (batch size, number query tokens, spatial_dim).
            kv_pos (optional): Positions for the key/value with shape (batch size, num key/value tokens, spatial_dim).

        Returns:
            Updated queries that attend to kv with shape (batch size, number query tokens, dim).
        """
        # Apply layer normalization
        q = self.norm_q(q)
        kv = self.norm_kv(kv)
        
        # Linear projections
        q = self.q(q)
        kv = self.kv(kv)
            
        # Split heads and keys/values
        q_heads = rearrange(q, "b q (h d) -> b h q d", h=self.num_heads, d=self.head_dim)
        k_heads, v_heads = torch.tensor_split(rearrange(kv, "b kv (h d) -> b h kv d", h=2*self.num_heads, d=self.head_dim), 2, dim=1)
        
        # Apply RoPE if positions are provided
        if q_pos is not None and kv_pos is not None:
            q_heads = self.rope(q_heads, q_pos)
            k_heads = self.rope(k_heads, kv_pos)
            
        # Compute attention using PyTorch's scaled_dot_product_attention
        x = F.scaled_dot_product_attention(q_heads, k_heads, v_heads, dropout_p=(self.dropout if self.training else 0.0))
        
        # Merge heads and output projection
        x = rearrange(x, "b h q d -> b q (h d)")
        x = self.out_proj(x)
        
        return x


class Modulator(nn.Module):
    """Modulator module for FiLM-like modulation (https://github.com/ethanjperez/film) of features based on conditioning parameters.
    
    Args:
        dim: Dimensionality of the features to be modulated.
        cond_dim: Dimensionality of the conditioning parameters.
        hidden_dim: Dimensionality of the hidden layer in the modulation MLP. Default is 128.
        use_residual: If true, use residual connection in modulation. Default is False.
    """
    
    def __init__(self, dim, cond_dim, hidden_dim=128, use_residual=False):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(cond_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, dim * 2))
        self.use_residual = use_residual
        # Keep the original direct FiLM behavior as the default.  SHIFT-Crash
        # opts into a bounded residual mode through its isolated adapter.
        self.conditioning_mode = "direct"
        self.conditioning_residual_scale = 1.0
        self.conditioning_shift_scale = 1.0

    def configure_conditioning(self, mode="direct", residual_scale=1.0, shift_scale=1.0):
        mode = str(mode).lower().strip()
        if mode not in {"direct", "residual", "bounded_residual"}:
            raise ValueError(f"Unsupported conditioning mode: {mode!r}")
        residual_scale = float(residual_scale)
        shift_scale = float(shift_scale)
        if residual_scale < 0.0 or shift_scale < 0.0:
            raise ValueError("Conditioning scales must be non-negative.")
        self.conditioning_mode = mode
        self.conditioning_residual_scale = residual_scale
        self.conditioning_shift_scale = shift_scale

    def forward(self, x, params):
        """Modulates the features x based on the conditioning parameters params.
        
        Args:
            x: Features to be modulated with shape (batch size, number points, dim).
            params: Conditioning parameters with shape (batch size, cond_dim).
        
        Returns:
            Modulated features with shape (batch size, number points, dim).
        """
        scale, shift = torch.tensor_split(self.mlp(params), 2, dim=-1)
        # The conditioner produces one feature-wise vector per sample, while
        # encoder/decoder activations may contain an additional point/latent
        # axis.  Align that vector with the feature axis before broadcasting.
        # Without this, a batch of two and 512 latent points compares the
        # batch dimension against the latent-point dimension.
        while scale.ndim < x.ndim:
            scale = scale.unsqueeze(-2)
            shift = shift.unsqueeze(-2)
        if self.conditioning_mode == "bounded_residual":
            scale = torch.tanh(scale) * self.conditioning_residual_scale
            shift = torch.tanh(shift) * self.conditioning_shift_scale
            return x + scale * x + shift
        if self.conditioning_mode == "residual" or self.use_residual:
            return x + scale * x + shift
        return scale * x + shift


class SimulationParamModulatedMLP(nn.Module):
    """MLP with FiLM-like modulation (https://github.com/ethanjperez/film) based on simulation parameters.
    
    Args:
        dim: Dimensionality of the input and output features.
        hidden_dim: Dimensionality of the hidden layer of the MLP.
        cond_dim: Dimensionality of the conditioning parameters.
        cond_hidden_dim: Width of the conditioning bottleneck.
        dropout: Dropout rate. Default is 0.1.
        use_residual: If true, use residual connection in modulation. Default is False.
    """

    def __init__(self, dim, hidden_dim, cond_dim, cond_hidden_dim=128, dropout=0.1, use_residual=False):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.linear1 = nn.Linear(dim, hidden_dim)
        self.non_linearity = nn.GELU()
        self.linear2 = nn.Linear(hidden_dim, dim)
        self.modulator = Modulator(hidden_dim, cond_dim, hidden_dim=cond_hidden_dim, use_residual=use_residual)
        

    def forward(self, x, params):        
        """Processes the features x with an MLP, modulated based on the conditioning parameters params.
        
        Args:
            x: Features with shape (batch size, number points, dim).
            params: Conditioning parameters with shape (batch size, cond_dim).
        
        Returns:
            Processed features with shape (batch size, number points, dim).
        """
        x = self.modulator(self.non_linearity(self.linear1(self.norm(x))), params)
        x = self.linear2(x)
        
        return x


class PlainMLP(nn.Module):
    """Plain multi-layer perceptron (MLP) **without** modulation
    
    Args:
        dim: Dimensionality of the input and output features.
        hidden_dim: Dimensionality of the hidden layer of the MLP.
        dropout: Dropout rate. Default is 0.1.
    """

    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.linear1 = nn.Linear(dim, hidden_dim)
        self.non_linearity = nn.GELU()
        self.linear2 = nn.Linear(hidden_dim, dim)
        

    def forward(self, x, params):
        """Processes the features x with an MLP without modulation.
            
        Args:
            x: Features with shape (batch size, number points, dim).
            params: Conditioning parameters will be ignored.
        
        Returns:
            Processed features with shape (batch size, number points, dim).
        """
        x = self.linear2(self.non_linearity(self.linear1(self.norm(x))))
        
        return x


class EncoderBlock(nn.Module):
    """The encoder block updates the latent geometry by attending to a cross-attended version of itself. This means that
    first, the latent geometry attends to a subsampled version of the input geometry to integrate geometric information,
    and then it attends to this cross-attended version of itself to refine the latent representation.

    Args:
        dim: Dimensionality of the features.
        num_heads: Number of attention heads. Defaults to 8.
        dropout: Dropout rate. Defaults to 0.1.
        spatial_dim: Number of spatial dimensions for RoPE. Defaults to 3.
        cond_dim: Dimensionality of the conditioning parameters for the MLP. Defaults to 2.
        conditioning_hidden_dim: Width of the conditioning bottleneck. Defaults to 128.
        residual_update_scale: Multiplier applied to each residual update. Defaults to 1.
        normalize_residuals: If true, normalize block outputs after residual updates.
    """

    def __init__(
        self,
        dim,
        num_heads=8,
        dropout=0.1,
        spatial_dim=3,
        cond_dim=2,
        conditioning_hidden_dim=128,
        residual_update_scale=1.0,
        normalize_residuals=False,
    ):
        super().__init__()
        residual_update_scale = float(residual_update_scale)
        if not math.isfinite(residual_update_scale) or residual_update_scale <= 0.0:
            raise ValueError("residual_update_scale must be a finite positive number.")
        self.residual_update_scale = residual_update_scale
        self.normalize_residuals = bool(normalize_residuals)
        self.geo_attn = CrossAttention(dim=dim, num_heads=num_heads, dropout=dropout, spatial_dim=spatial_dim)
        self.cross_attn = CrossAttention(dim=dim, num_heads=num_heads, dropout=dropout, spatial_dim=spatial_dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.cross_output_norm = nn.LayerNorm(dim, eps=1e-6) if self.normalize_residuals else nn.Identity()
        self.output_norm = nn.LayerNorm(dim, eps=1e-6) if self.normalize_residuals else nn.Identity()
        
        # Pointwise MLP
        if cond_dim > 0:
            self.mlp = SimulationParamModulatedMLP(
                dim=dim,
                hidden_dim=dim * 4,
                cond_dim=cond_dim,
                cond_hidden_dim=conditioning_hidden_dim,
                dropout=dropout,
            )
        else:
            self.mlp = PlainMLP(dim=dim, hidden_dim=dim * 4, dropout=dropout)
       
    def forward(self, latent_geometry, subsampled_geometry, params, latent_geometry_pos=None, subsampled_geometry_pos=None):
        """Updates the latent geometry by attending to a cross-attended version of itself that first attends to a subsampled
        version of the input geometry.

        Args:
            latent_geometry: Latent geometry with shape (batch size, number latent points, dim).
            subsampled_geometry: Subsampled input geometry with shape (batch size, number subsampled points, dim).
            params: Conditioning parameters with shape (batch size, cond_dim).
            latent_geometry_pos (optional): Positions of the latent geometry for the positional embeddings with shape (batch size, number latent points, spatial_dim). Defaults to None.
            subsampled_geometry_pos (optional): Positions of the subsampled input geometry for the positional embeddings with shape (batch size, number subsampled points, spatial_dim) . Defaults to None.

        Returns:
            tuple: A tuple containing:
                - Updated latent geometry with shape (batch size, number latent points, dim).
                - Latent geometry after geometry cross-attention and before cross-attention and MLP with shape (batch size, number latent points, dim).
        """
        # First cross-attention with the subsampled geometry
        latent_geometry_cross = latent_geometry + self.residual_update_scale * self.attn_dropout(
            self.geo_attn(q=latent_geometry, kv=subsampled_geometry, q_pos=latent_geometry_pos, kv_pos=subsampled_geometry_pos)
        )
        latent_geometry_cross = self.cross_output_norm(latent_geometry_cross)

        # Update the initial latent geometry by attending to the cross-attended version of itself
        latent_geometry_self = latent_geometry + self.residual_update_scale * self.attn_dropout(
            self.cross_attn(q=latent_geometry, kv=latent_geometry_cross, q_pos=latent_geometry_pos, kv_pos=latent_geometry_pos)
        )
        
        # Pointwise MLP
        latent_geometry_mlp = latent_geometry_self + self.residual_update_scale * self.mlp(latent_geometry_self, params)
        
        return self.output_norm(latent_geometry_mlp), latent_geometry_cross


class DecoderBlock(nn.Module):
    """The decoder block attends to the latent geometry of the corresponding encoder block to produce predictions
    of physical quantities at query positions.

    Args:
        dim: Dimensionality of the features.
        num_heads: Number of attention heads. Defaults to 8.
        dropout: Dropout rate. Defaults to 0.1.
        spatial_dim: Number of spatial dimensions for RoPE. Defaults to 3.
        cond_dim: Dimensionality of the conditioning parameters for the MLP. Defaults to 2.
        conditioning_hidden_dim: Width of the conditioning bottleneck. Defaults to 128.
        residual_update_scale: Multiplier applied to each residual update. Defaults to 1.
        normalize_residuals: If true, normalize block outputs after residual updates.
        shared_attn: Shared cross-attention module from the corresponding encoder block. Defaults to None.
        shared_mlp: Shared MLP module from the corresponding encoder block. Defaults to None.
    """

    def __init__(
        self,
        dim,
        num_heads=8,
        dropout=0.1,
        spatial_dim=3,
        cond_dim=2,
        conditioning_hidden_dim=128,
        residual_update_scale=1.0,
        normalize_residuals=False,
        shared_attn=None,
        shared_mlp=None,
    ):
        super().__init__()
        residual_update_scale = float(residual_update_scale)
        if not math.isfinite(residual_update_scale) or residual_update_scale <= 0.0:
            raise ValueError("residual_update_scale must be a finite positive number.")
        self.residual_update_scale = residual_update_scale
        self.normalize_residuals = bool(normalize_residuals)
        self.attn = CrossAttention(dim=dim, num_heads=num_heads, dropout=dropout, spatial_dim=spatial_dim) if shared_attn is None else shared_attn
        self.attn_dropout = nn.Dropout(dropout)
        self.output_norm = nn.LayerNorm(dim, eps=1e-6) if self.normalize_residuals else nn.Identity()
        
        # Pointwise MLP
        if cond_dim > 0:
            self.mlp = (
                SimulationParamModulatedMLP(
                    dim=dim,
                    hidden_dim=dim * 4,
                    cond_dim=cond_dim,
                    cond_hidden_dim=conditioning_hidden_dim,
                    dropout=dropout,
                )
                if shared_mlp is None
                else shared_mlp
            )
        else:
            self.mlp = PlainMLP(dim=dim, hidden_dim=dim * 4, dropout=dropout) if shared_mlp is None else shared_mlp
       
    def forward(self, queries, latent_geometry, params, queries_pos=None, latent_geometry_pos=None):
        """Updates the queries by attending to the latent geometry of the corresponding encoder block.

        Args:
            queries: Features of the query positions with shape (batch size, number query points, dim).
            latent_geometry: Latent geometry with shape (batch size, number latent points, dim).
            params: Conditioning parameters with shape (batch size, cond_dim).
            queries_pos (optional): Positions of the query positions for the positional embeddings with shape (batch size, number query points, spatial_dim). Defaults to None.
            latent_geometry_pos (optional): Positions of the latent geometry for the positional embeddings with shape (batch size, number latent points, spatial_dim). Defaults to None.
        
        Returns:
            Updated queries with shape (batch size, number query points, dim).
        """
        # Cross-attention with the latent geometry
        queries = queries + self.residual_update_scale * self.attn_dropout(
            self.attn(q=queries, kv=latent_geometry, q_pos=queries_pos, kv_pos=latent_geometry_pos)
        )
        
        # Pointwise MLP
        queries = queries + self.residual_update_scale * self.mlp(queries, params)
        
        return self.output_norm(queries)


def sample_geometry(geometry, num_samples, with_replacement=False):
    """Samples points from the input geometry.

    Args:
        geometry: Input geometry with shape (batch size, number points, spatial_dim).
        num_samples: Number of points to sample.
    
    Returns:
        Sampled input geometry with shape (batch size, num_samples, spatial_dim).
    """
    n_points = int(geometry.shape[1])
    batch_size = int(geometry.shape[0])
    if num_samples <= 0:
        return geometry
    if with_replacement:
        idx = torch.randint(0, n_points, (batch_size, num_samples), device=geometry.device, dtype=torch.long)
    else:
        if num_samples >= n_points:
            return geometry
        idx = torch.stack(
            [torch.randperm(n_points, device=geometry.device)[:num_samples] for _ in range(batch_size)],
            dim=0,
        )
    sampled_geometry = torch.gather(geometry, 1, idx.unsqueeze(-1).expand(-1, -1, geometry.shape[-1]))
    return sampled_geometry


def sample_geometry_indices(geometry, num_samples, with_replacement=False):
    """Return the point indices used by ``sample_geometry``."""
    n_points = int(geometry.shape[1])
    batch_size = int(geometry.shape[0])
    if num_samples <= 0 or num_samples >= n_points:
        return torch.arange(n_points, device=geometry.device, dtype=torch.long).view(1, -1).expand(batch_size, -1)
    if with_replacement:
        return torch.randint(0, n_points, (batch_size, num_samples), device=geometry.device, dtype=torch.long)
    return torch.stack(
        [torch.randperm(n_points, device=geometry.device)[:num_samples] for _ in range(batch_size)],
        dim=0,
    )


def gather_point_values(values, indices):
    """Gather [B,N] or [B,N,C] node attributes using [B,K] indices."""
    if values is None:
        return None
    if values.ndim == 2:
        return torch.gather(values, 1, indices)
    return torch.gather(values, 1, indices.unsqueeze(-1).expand(-1, -1, values.shape[-1]))

   
class SMART(nn.Module):
    """SMART model for simulating time-independent PDEs over complex 3D geometries.
    
    Args:
        spatial_dim: Number of spatial dimensions. Default is 3.
        surface_channels: Number of output channels for surface predictions. Default is 1.
        volume_channels: Number of output channels for volume predictions. Default is 3.
        parameter_channels: Number of conditioning parameter channels. Default is 2.
        latent_dim: Dimensionality of the latent representations. Default is 256.
        latent_geometry_points: Number of points of the latent geometry. Default is 4096.
        subsampled_geometry_points: Number of points in the subsampled geometry for geometry cross-attention. Default is 16384.
        num_encoder_decoder_blocks: Number of encoder-decoder blocks. Default is 8.
        num_heads: Number of attention heads. Default is 8.
        pos_scale_factor: Scaling factor for the positions to use more/less of the dynamic range of the positional embedding. Default is 1000.
        dropout: Dropout rate. Default is 0.0.
        subregion_size: Number of query points to process in each subregion during sequential inference. Default is 262144.
        conditioning_hidden_dim: Width of each FiLM conditioning bottleneck. Default is 128.
        residual_update_scale: Multiplier applied to each residual update. Default is 1.
        normalize_residuals: If true, normalize encoder and decoder block outputs.
    """

    def __init__(self, spatial_dim=3,
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
                 conditioning_hidden_dim=128,
                 residual_update_scale=1.0,
                 normalize_residuals=False,
                 geometry_feature_channels=0,
                 query_feature_channels=0,
                 part_embedding_size=0,
                 part_embedding_dim=16):
        super(SMART, self).__init__()
        assert surface_channels > 0 and volume_channels > 0, "surface_channels and volume_channels must be positive integers."
        
        self.surface_channels = surface_channels
        self.volume_channels = volume_channels
        self.num_geo = latent_geometry_points
        self.subsampled_geometry_points = subsampled_geometry_points
        self.subsampled_geometry_with_replacement = bool(subsampled_geometry_with_replacement)
        self.pos_scale_factor = pos_scale_factor
        self.conditioning_hidden_dim = int(conditioning_hidden_dim)
        if self.conditioning_hidden_dim <= 0:
            raise ValueError("conditioning_hidden_dim must be positive.")
        self.residual_update_scale = float(residual_update_scale)
        self.normalize_residuals = bool(normalize_residuals)
        self.geometry_feature_channels = int(geometry_feature_channels)
        self.query_feature_channels = int(query_feature_channels)
        self.part_embedding_size = int(part_embedding_size)
        self.part_embedding_dim = int(part_embedding_dim)
        if self.geometry_feature_channels < 0 or self.query_feature_channels < 0:
            raise ValueError("Feature channel counts must be non-negative.")
        if self.geometry_feature_channels != self.query_feature_channels:
            raise ValueError("Geometry and query feature widths must match for shared SMART feature encoding.")
        if self.part_embedding_size < 0 or self.part_embedding_dim <= 0:
            raise ValueError("Invalid part embedding configuration.")
        self.point_feature_encoder = None
        if self.geometry_feature_channels > 0 or self.part_embedding_size > 0:
            if self.geometry_feature_channels <= 0 or self.part_embedding_size <= 0:
                raise ValueError("Both continuous feature channels and part embedding size are required together.")
            self.point_feature_norm = nn.LayerNorm(self.geometry_feature_channels, eps=1.0e-6)
            self.point_feature_projection = nn.Sequential(
                nn.Linear(self.geometry_feature_channels, latent_dim),
                nn.GELU(),
                nn.Linear(latent_dim, latent_dim),
            )
            self.part_embedding = nn.Embedding(self.part_embedding_size, self.part_embedding_dim, padding_idx=0)
            self.part_projection = nn.Linear(self.part_embedding_dim, latent_dim, bias=False)
            self.point_feature_scale = nn.Parameter(torch.tensor(0.1))
            self.part_feature_scale = nn.Parameter(torch.tensor(0.1))
            self.point_feature_encoder = nn.Identity()
        self.pos_encoder = ModulatedPositionalEmbedding(latent_dim, spatial_dim)
        
        # Encoder and decoder blocks
        self.encoder_blocks = nn.ModuleList(
            [
                EncoderBlock(
                    dim=latent_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    spatial_dim=spatial_dim,
                    cond_dim=parameter_channels,
                    conditioning_hidden_dim=self.conditioning_hidden_dim,
                    residual_update_scale=self.residual_update_scale,
                    normalize_residuals=self.normalize_residuals,
                )
                for _ in range(num_encoder_decoder_blocks)
            ]
        )
        self.decoder_blocks = nn.ModuleList(
            [
                DecoderBlock(
                    dim=latent_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    spatial_dim=spatial_dim,
                    cond_dim=parameter_channels,
                    conditioning_hidden_dim=self.conditioning_hidden_dim,
                    residual_update_scale=self.residual_update_scale,
                    normalize_residuals=self.normalize_residuals,
                    shared_attn=self.encoder_blocks[i].cross_attn,
                    shared_mlp=self.encoder_blocks[i].mlp,
                )
                for i in range(num_encoder_decoder_blocks)
            ]
        )
        
        # Final MLP
        self.mlp = nn.Sequential(nn.Linear(latent_dim, 128), nn.GELU(),
                                 nn.Linear(128, 64), nn.GELU(),
                                 nn.Linear(64, surface_channels+volume_channels))
        
        # Subregion size for inference
        self.subregion_size = subregion_size
    
    def initialize_weights(self):
        self.apply(self._init_weights)

    # Weight initialization from Transolver
    # (https://github.com/thuml/Transolver/blob/a11be9c4f7db1885e4b08c68432bc31799492ec9/Car-Design-ShapeNetCar/models/Transolver.py#L168)
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def _encode_point_features(self, features, part_ids):
        if self.point_feature_encoder is None:
            return None
        if features is None or part_ids is None:
            raise ValueError("Configured SMART point features require both continuous features and part IDs.")
        if features.shape[-1] != self.geometry_feature_channels:
            raise ValueError(
                f"Expected {self.geometry_feature_channels} continuous point features, got {features.shape[-1]}."
            )
        part_ids = part_ids.long().clamp(0, self.part_embedding_size - 1)
        continuous = self.point_feature_projection(self.point_feature_norm(features.float()))
        categorical = self.part_projection(self.part_embedding(part_ids))
        return self.point_feature_scale * continuous + self.part_feature_scale * categorical

    def encode(self, geo, params, geometry_features=None, geometry_part_ids=None, return_final=False):
        # Prepare positions by scaling
        geo = geo * self.pos_scale_factor
        
        # Sample the initial latent geometry
        latent_idx = sample_geometry_indices(geo, self.num_geo)
        latent_geo_pos = gather_point_values(geo, latent_idx)
        latent_geo_emb = self.pos_encoder(latent_geo_pos)
        if self.point_feature_encoder is not None:
            latent_geo_emb = latent_geo_emb + self._encode_point_features(
                gather_point_values(geometry_features, latent_idx),
                gather_point_values(geometry_part_ids, latent_idx),
            )
        
        # Apply encoder blocks
        intermediate_latent_geometries = []
        for block in self.encoder_blocks:
            # Subsample the geometry for geometry cross-attention
            sub_idx = sample_geometry_indices(
                geo, self.subsampled_geometry_points, with_replacement=self.subsampled_geometry_with_replacement
            )
            sub_geo_pos = gather_point_values(geo, sub_idx)
            sub_geo_emb = self.pos_encoder(sub_geo_pos)
            if self.point_feature_encoder is not None:
                sub_geo_emb = sub_geo_emb + self._encode_point_features(
                    gather_point_values(geometry_features, sub_idx),
                    gather_point_values(geometry_part_ids, sub_idx),
                )
            
            # Apply encoder block
            latent_geo_emb, e_ca = block(latent_geo_emb, sub_geo_emb, params, latent_geometry_pos=latent_geo_pos, subsampled_geometry_pos=sub_geo_pos)
            
            # Store for decoder
            intermediate_latent_geometries.append(e_ca)
        
        if return_final:
            return intermediate_latent_geometries, latent_geo_pos, latent_geo_emb
        return intermediate_latent_geometries, latent_geo_pos
    
    def decode_features(
        self,
        intermediate_latent_geometries,
        latent_geo_pos,
        params,
        query_pos,
        query_features=None,
        query_part_ids=None,
    ):
        # Prepare positions by scaling
        query_pos = query_pos * self.pos_scale_factor
        query_emb = self.pos_encoder(query_pos)
        if self.point_feature_encoder is not None:
            query_emb = query_emb + self._encode_point_features(query_features, query_part_ids)
        
        # Apply decoder blocks
        for e_ca, block in zip(intermediate_latent_geometries, self.decoder_blocks):
            query_emb = block(query_emb, e_ca, params, queries_pos=query_pos, latent_geometry_pos=latent_geo_pos)
        
        return query_emb

    def decode(self, intermediate_latent_geometries, latent_geo_pos, params, query_pos, query_features=None, query_part_ids=None):
        query_emb = self.decode_features(
            intermediate_latent_geometries,
            latent_geo_pos,
            params,
            query_pos,
            query_features=query_features,
            query_part_ids=query_part_ids,
        )
        pred = self.mlp(query_emb)
        return pred

    def forward(
        self,
        geo,
        surf_query_pos,
        vol_query_pos,
        params,
        geometry_features=None,
        query_features=None,
        geometry_part_ids=None,
        query_part_ids=None,
    ):
        """Forward method for SMART model.
        
        Args:
            geo: Input geometry with shape (batch size, number points, spatial_dim).
            surf_query_pos: Surface query positions with shape (batch size, number surface query points, spatial_dim).
            vol_query_pos: Volume query positions with shape (batch size, number volume query points, spatial_dim).
            params: Conditioning parameters with shape (batch size, cond_dim). If not used, pass None.
        
        Returns:
            tuple: A tuple containing:
                - Surface predictions with shape (batch size, number surface query points, surface_channels).
                - Volume predictions with shape (batch size, number volume query points, volume_channels).
        """
        # Encode
        intermediate_latent_geometries, latent_geo_pos = self.encode(
            geo, params, geometry_features=geometry_features, geometry_part_ids=geometry_part_ids
        )
        
        # Prepare query positions by concatenating surface and volume query positions
        query_pos = torch.cat([surf_query_pos, vol_query_pos], dim=1)
        
        # Decode
        pred = self.decode(
            intermediate_latent_geometries,
            latent_geo_pos,
            params,
            query_pos,
            query_features=query_features,
            query_part_ids=query_part_ids,
        )
        
        # Split surface and volume predictions
        pred_surf = pred[:, :surf_query_pos.shape[1], 0:self.surface_channels]
        pred_vol = pred[:, surf_query_pos.shape[1]:, self.surface_channels:]
        
        return pred_surf, pred_vol
    
    @torch.inference_mode()
    def inference(
        self,
        geo,
        surf_query_pos,
        vol_query_pos,
        params,
        geometry_features=None,
        query_features=None,
        geometry_part_ids=None,
        query_part_ids=None,
        volume_query_features=None,
        volume_query_part_ids=None,
    ):
        """Sequential inference method to handle large number of query points that may not fit into GPU memory.
        
        Args:
            geo: Input geometry with shape (batch size, number points, spatial_dim).
            surf_query_pos: Surface query positions with shape (batch size, number surface query points, spatial_dim).
            vol_query_pos: Volume query positions with shape (batch size, number volume query points, spatial_dim).
            params: Conditioning parameters with shape (batch size, cond_dim). If not used, pass None.
        
        Returns:
            tuple: A tuple containing:
                - Surface predictions with shape (batch size, number surface query points, surface_channels).
                - Volume predictions with shape (batch size, number volume query points, volume_channels).
        """
        # Encode
        intermediate_latent_geometries, latent_geo_pos = self.encode(
            geo, params, geometry_features=geometry_features, geometry_part_ids=geometry_part_ids
        )
        
        # Surface predictions sequentially
        N_surf = surf_query_pos.shape[1]
        y_hat_surf_subregions = []
        for i in range(0, N_surf, self.subregion_size):
            surf_subregion = surf_query_pos[:, i:i+self.subregion_size, :]
            surf_features = None if query_features is None else query_features[:, i:i+self.subregion_size, :]
            surf_part_ids = None if query_part_ids is None else query_part_ids[:, i:i+self.subregion_size]
            y_surf_subregion = self.decode(
                intermediate_latent_geometries,
                latent_geo_pos,
                params,
                surf_subregion,
                query_features=surf_features,
                query_part_ids=surf_part_ids,
            )
            y_hat_surf_subregions.append(y_surf_subregion)
        y_hat_surf = torch.cat(y_hat_surf_subregions, dim=1)

        # Volume predictions sequentially
        N_vol = vol_query_pos.shape[1]
        y_hat_vol_subregions = []
        for i in range(0, N_vol, self.subregion_size):
            vol_subregion = vol_query_pos[:, i:i+self.subregion_size, :]
            vol_features = None if volume_query_features is None else volume_query_features[:, i:i+self.subregion_size, :]
            vol_part_ids = None if volume_query_part_ids is None else volume_query_part_ids[:, i:i+self.subregion_size]
            y_vol_subregion = self.decode(
                intermediate_latent_geometries,
                latent_geo_pos,
                params,
                vol_subregion,
                query_features=vol_features,
                query_part_ids=vol_part_ids,
            )
            y_hat_vol_subregions.append(y_vol_subregion)
        y_hat_vol = torch.cat(y_hat_vol_subregions, dim=1)
        
        # Split surface and volume predictions
        pred_surf = y_hat_surf[:, :, 0:self.surface_channels]
        pred_vol = y_hat_vol[:, :, self.surface_channels:]
        
        return pred_surf, pred_vol
