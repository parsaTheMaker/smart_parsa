"""GAOT-3D adapter for SMART's surface/volume point-cloud interface."""

from __future__ import annotations

import torch
from torch import nn
from torch_geometric.data import Batch, Data

from .gaot_layers.gaot_3d_reference import GAOT3D
from .gaot_layers.attn import AttentionConfig, TransformerConfig
from .gaot_layers.magno import MAGNOConfig


class GAOT(nn.Module):
    """Geometry-aware latent-grid operator following the GAOT-3D repository."""

    expects_geo_log_density = False

    def __init__(
        self,
        spatial_dim=3,
        surface_channels=1,
        volume_channels=3,
        parameter_channels=0,
        latent_tokens=(32, 32, 16),
        lifting_channels=64,
        gno_radius=0.033,
        gno_neighbor_strategy="bidirectional",
        gno_k_neighbors=1,
        gno_scales=(1.0,),
        gno_encoder_scales=None,
        gno_decoder_scales=None,
        gno_embedding_method="statistical",
        gno_mlp_type="linear",
        projection_channels=384,
        in_gno_channel_mlp_hidden_layers=(128, 128, 128),
        out_gno_channel_mlp_hidden_layers=(128, 128),
        transformer_patch_size=2,
        transformer_hidden_size=384,
        transformer_num_layers=12,
        transformer_num_heads=8,
        transformer_dropout=0.0,
        transformer_positional_embedding="absolute",
        use_geoembed=True,
        use_query_residual_head=True,
        query_residual_hidden=256,
        query_residual_scale=0.25,
        global_geometry_head=True,
        dataset_position_min=(0.0, 0.0, 0.0),
        dataset_position_max=(1.0, 1.0, 1.0),
    ):
        super().__init__()
        if int(spatial_dim) != 3:
            raise ValueError("GAOT-3D requires spatial_dim=3")
        if parameter_channels:
            raise ValueError("The GAOT-3D DrivAerML adapter does not use parameter channels.")

        latent_tokens = tuple(int(value) for value in latent_tokens)
        if len(latent_tokens) != 3:
            raise ValueError("latent_tokens must contain three dimensions")
        if any(value % int(transformer_patch_size) for value in latent_tokens):
            raise ValueError("Every latent-token dimension must be divisible by transformer_patch_size")

        self.surface_channels = int(surface_channels)
        self.volume_channels = int(volume_channels)
        self.dataset_position_min = tuple(float(value) for value in dataset_position_min)
        self.dataset_position_max = tuple(float(value) for value in dataset_position_max)
        if len(self.dataset_position_min) != 3 or len(self.dataset_position_max) != 3:
            raise ValueError("dataset_position_min/max must contain three values")
        if any(upper <= lower for lower, upper in zip(self.dataset_position_min, self.dataset_position_max)):
            raise ValueError("dataset_position_max must be greater than dataset_position_min")

        # AhmedMLDatasetV2 stores each coordinate axis in [0, 1]. GAOT's
        # reference preprocessing instead uses one shared physical scale, so
        # retain the dataset aspect ratio before mapping into GAOT coordinates.
        physical_min = min(self.dataset_position_min)
        physical_max = max(self.dataset_position_max)
        physical_span = max(physical_max - physical_min, 1.0e-6)
        self.latent_domain_min = tuple(
            2.0 * (value - physical_min) / physical_span - 1.0
            for value in self.dataset_position_min
        )
        self.latent_domain_max = tuple(
            2.0 * (value - physical_min) / physical_span - 1.0
            for value in self.dataset_position_max
        )
        self.output_channels = self.surface_channels + self.volume_channels

        magno_config = MAGNOConfig(
            gno_coord_dim=3,
            gno_radius=float(gno_radius),
            lifting_channels=int(lifting_channels),
            projection_channels=int(projection_channels),
            neighbor_strategy=str(gno_neighbor_strategy),
            k_neighbors=int(gno_k_neighbors),
            scales=list(gno_scales),
            encoder_scales=None if gno_encoder_scales is None else list(gno_encoder_scales),
            decoder_scales=None if gno_decoder_scales is None else list(gno_decoder_scales),
            use_geoembed=[bool(use_geoembed), False],
            embedding_method=str(gno_embedding_method),
            encoder_feature_attr="pos",
            mlp_type=str(gno_mlp_type),
            precompute_edges=False,
            gno_use_torch_cluster=True,
            in_gno_channel_mlp_hidden_layers=list(in_gno_channel_mlp_hidden_layers),
            out_gno_channel_mlp_hidden_layers=list(out_gno_channel_mlp_hidden_layers),
            out_gno_transform_type="linear",
        )
        attention_config = AttentionConfig(
            hidden_size=int(transformer_hidden_size),
            num_heads=int(transformer_num_heads),
            num_kv_heads=int(transformer_num_heads),
            atten_dropout=float(transformer_dropout),
            positional_embedding=str(transformer_positional_embedding),
        )
        transformer_config = TransformerConfig(
            patch_size=int(transformer_patch_size),
            hidden_size=int(transformer_hidden_size),
            num_layers=int(transformer_num_layers),
            positional_embedding=str(transformer_positional_embedding),
            use_long_range_skip=True,
            attn_config=attention_config,
        )
        self.core = GAOT3D(
            input_size=3,
            output_size=self.output_channels,
            magno_config=magno_config,
            attn_config=transformer_config,
            latent_tokens=latent_tokens,
            norm_domin=(self.latent_domain_min, self.latent_domain_max),
        )
        self.query_residual_scale = float(query_residual_scale)
        if self.query_residual_scale < 0.0:
            raise ValueError("query_residual_scale must be non-negative")
        self.global_geometry_head = bool(global_geometry_head)
        residual_input = self.output_channels + 3 + (12 if self.global_geometry_head else 0)
        if bool(use_query_residual_head):
            self.query_residual_head = nn.Sequential(
                nn.Linear(residual_input, int(query_residual_hidden)),
                nn.GELU(),
                nn.LayerNorm(int(query_residual_hidden)),
                nn.Linear(int(query_residual_hidden), self.output_channels),
            )
            # Start with a small, nonzero correction so this path is active
            # from the first optimizer step without disturbing GAOT strongly.
            nn.init.normal_(self.query_residual_head[-1].weight, mean=0.0, std=1.0e-3)
            nn.init.constant_(self.query_residual_head[-1].bias, 1.0e-3)
        else:
            self.query_residual_head = None

    def _normalize_coordinates(self, points):
        points = points.float()
        lower = points.new_tensor(self.dataset_position_min).view(1, 1, -1)
        upper = points.new_tensor(self.dataset_position_max).view(1, 1, -1)
        physical = points * (upper - lower) + lower
        physical_min = min(self.dataset_position_min)
        physical_max = max(self.dataset_position_max)
        return 2.0 * (physical - physical_min) / max(physical_max - physical_min, 1.0e-6) - 1.0

    @staticmethod
    def _geometry_summary(geo_norm):
        """Return sample-dependent global geometry statistics for the residual path."""
        mean = geo_norm.mean(dim=1)
        std = geo_norm.float().std(dim=1, unbiased=False).to(dtype=geo_norm.dtype)
        minimum = geo_norm.amin(dim=1)
        maximum = geo_norm.amax(dim=1)
        return torch.cat([mean, std, minimum, maximum], dim=-1)

    @staticmethod
    def _make_batch(geo):
        data = [Data(pos=geo[index]) for index in range(geo.shape[0])]
        return Batch.from_data_list(data)

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
            raise ValueError("The GAOT-3D DrivAerML adapter does not use parameter channels.")

        geo_norm = self._normalize_coordinates(geo)
        surf_norm = self._normalize_coordinates(surf_query_pos)
        vol_norm = self._normalize_coordinates(vol_query_pos)
        query_norm = torch.cat([surf_norm, vol_norm], dim=1)
        batch = self._make_batch(geo_norm)
        query_batch = torch.arange(geo.shape[0], device=geo.device).repeat_interleave(query_norm.shape[1])
        output = self.core(
            batch=batch,
            query_coord_pos=query_norm.reshape(-1, query_norm.shape[-1]),
            query_coord_batch_idx=query_batch,
        )
        output = output.view(geo.shape[0], query_norm.shape[1], self.output_channels)
        if self.query_residual_head is not None:
            residual_inputs = [output, query_norm]
            if self.global_geometry_head:
                geometry_summary = self._geometry_summary(geo_norm)
                geometry_summary = geometry_summary.unsqueeze(1).expand(-1, query_norm.shape[1], -1)
                residual_inputs.append(geometry_summary)
            residual_input = torch.cat(residual_inputs, dim=-1)
            output = output + self.query_residual_scale * self.query_residual_head(residual_input)
        surface_count = surf_query_pos.shape[1]
        pred_surface = output[:, :surface_count, : self.surface_channels]
        pred_volume = output[:, surface_count:, self.surface_channels :]
        if return_latent:
            return pred_surface, pred_volume, None
        return pred_surface, pred_volume

    def inference(self, geo, surf_query_pos, vol_query_pos, params=None, geo_log_density=None):
        """Evaluation entry point required by the shared two-view trainer."""
        return self.forward(
            geo,
            surf_query_pos,
            vol_query_pos,
            params=params,
            geo_log_density=geo_log_density,
        )
