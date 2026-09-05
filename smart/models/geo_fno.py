"""3D Geo-FNO adapter for geometry-to-field prediction.

This follows the Geo-FNO construction of Li et al.: a learned coordinate map
transfers an irregular geometry cloud to a canonical Fourier grid, standard
FNO blocks operate on that grid, and a direct inverse Fourier evaluation
returns values at arbitrary physical queries.  The original repository ships
2D examples; this module is the corresponding 3D implementation required by
the surface/volume datasets in this project.

The base model intentionally uses the raw empirical geometry cloud.  It does
not receive density estimates, area weights, or a canonical resampling step.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _signed_frequencies(modes: int) -> torch.Tensor:
    """Return the symmetric low-frequency stencil used by the direct NDFT."""
    if int(modes) < 1:
        raise ValueError("Geo-FNO modes must be positive.")
    return torch.arange(-(int(modes) - 1), int(modes), dtype=torch.float32)


class _IPHI3D(nn.Module):
    """Learned inverse coordinate map, generalized from Geo-FNO's IPHI."""

    def __init__(self, parameter_channels: int, width: int, fourier_bands: int, max_deformation: float):
        super().__init__()
        self.parameter_channels = int(parameter_channels)
        self.max_deformation = float(max_deformation)
        if self.max_deformation <= 0.0:
            raise ValueError("Geo-FNO max_deformation must be positive.")
        bands = math.pi * torch.pow(2.0, torch.arange(int(fourier_bands), dtype=torch.float32))
        self.register_buffer("bands", bands, persistent=False)
        # Coordinates, radius, azimuth, and elevation are the 3D analogue of
        # the original IPHI coordinate/radius/angle feature construction.
        feature_dim = 6 + 12 * int(fourier_bands)
        self.input = nn.Linear(feature_dim, int(width))
        self.code = nn.Linear(self.parameter_channels, int(width)) if self.parameter_channels > 0 else None
        self.hidden = nn.Sequential(
            nn.Linear(int(width) * 2, int(width) * 2),
            nn.Tanh(),
            nn.Linear(int(width) * 2, int(width) * 2),
            nn.Tanh(),
            nn.Linear(int(width) * 2, int(width)),
            nn.Tanh(),
        )
        self.output = nn.Linear(int(width), 3)

    def forward(self, x: torch.Tensor, params: torch.Tensor | None = None) -> torch.Tensor:
        x32 = x.float()
        centered = x32 - 0.5
        radius = torch.linalg.vector_norm(centered, dim=-1, keepdim=True)
        azimuth = torch.atan2(centered[..., 1:2], centered[..., 0:1]) / math.pi
        elevation = torch.atan2(centered[..., 2:3], radius.clamp_min(1.0e-6)) / math.pi
        raw = torch.cat([x32, radius, azimuth, elevation], dim=-1)
        phase = raw.unsqueeze(-1) * self.bands.view(1, 1, 1, -1)
        harmonic = torch.cat([torch.sin(phase), torch.cos(phase)], dim=-1).flatten(start_dim=2)
        features = torch.cat([raw, harmonic], dim=-1)
        hidden = self.input(features)
        if self.code is not None and params is not None:
            code = self.code(params.float()).unsqueeze(1).expand(-1, x.shape[1], -1)
        else:
            code = torch.zeros_like(hidden)
        hidden = self.hidden(torch.cat([hidden, code], dim=-1))
        delta = torch.tanh(self.output(hidden)) * self.max_deformation
        # Preserve Geo-FNO's residual, coordinate-proportional deformation.
        return x32 + x32 * delta


class _SpectralConv3d(nn.Module):
    """Standard Fourier-grid convolution used between Geo-FNO NDFT stages."""

    def __init__(self, channels: int, modes: int):
        super().__init__()
        self.channels = int(channels)
        self.modes = int(modes)
        scale = 1.0 / float(self.channels * self.channels)
        shape = (self.channels, self.channels, self.modes, self.modes, self.modes, 2)
        self.weights = nn.ParameterList(
            [nn.Parameter(scale * torch.randn(shape, dtype=torch.float32)) for _ in range(4)]
        )

    @staticmethod
    def _mul(x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bcxyz,coxyz->boxyz", x, weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size_x, size_y, size_z = x.shape[-3:]
        modes_x = min(self.modes, size_x // 2)
        modes_y = min(self.modes, size_y // 2)
        modes_z = min(self.modes, size_z // 2 + 1)
        if min(modes_x, modes_y, modes_z) < 1:
            raise ValueError(f"Geo-FNO latent grid {x.shape[-3:]} is too small for modes={self.modes}.")
        with torch.autocast(device_type=x.device.type, enabled=False):
            x_ft = torch.fft.rfftn(x.float(), dim=(-3, -2, -1))
            out_ft = torch.zeros(
                x.shape[0], self.channels, size_x, size_y, size_z // 2 + 1,
                dtype=torch.cfloat,
                device=x.device,
            )
            # Keep trainable weights as real/imaginary pairs: GradScaler
            # cannot unscale gradients stored in ComplexFloat parameters.
            weights = [
                torch.view_as_complex(weight[:, :, :modes_x, :modes_y, :modes_z].contiguous())
                for weight in self.weights
            ]
            out_ft[:, :, :modes_x, :modes_y, :modes_z] = self._mul(
                x_ft[:, :, :modes_x, :modes_y, :modes_z], weights[0]
            )
            out_ft[:, :, -modes_x:, :modes_y, :modes_z] = self._mul(
                x_ft[:, :, -modes_x:, :modes_y, :modes_z], weights[1]
            )
            out_ft[:, :, :modes_x, -modes_y:, :modes_z] = self._mul(
                x_ft[:, :, :modes_x, -modes_y:, :modes_z], weights[2]
            )
            out_ft[:, :, -modes_x:, -modes_y:, :modes_z] = self._mul(
                x_ft[:, :, -modes_x:, -modes_y:, :modes_z], weights[3]
            )
            return torch.fft.irfftn(out_ft, s=(size_x, size_y, size_z), dim=(-3, -2, -1))


class _DirectSpectralMix(nn.Module):
    """Learned Fourier-channel map for Geo-FNO's NDFT boundary stages."""

    def __init__(self, channels: int, frequency_count: int):
        super().__init__()
        self.channels = int(channels)
        self.frequency_count = int(frequency_count)
        scale = 1.0 / float(self.channels * self.channels)
        # Real/imaginary pairs preserve the complex Geo-FNO calculation while
        # remaining compatible with CUDA AMP gradient scaling.
        self.weight = nn.Parameter(
            scale * torch.randn(self.channels, self.channels, self.frequency_count, 2, dtype=torch.float32)
        )

    def forward(self, coefficients: torch.Tensor) -> torch.Tensor:
        if coefficients.shape[-1] != self.frequency_count:
            raise ValueError(
                f"Expected {self.frequency_count} direct Fourier modes, got {coefficients.shape[-1]}."
            )
        weight = torch.view_as_complex(self.weight.contiguous())
        return torch.einsum("bck,cok->bok", coefficients, weight)


class GeoFNO(nn.Module):
    """Medium-capacity, raw-point-cloud Geo-FNO for 3D field surrogates."""

    expects_geo_log_density = False

    def __init__(
        self,
        spatial_dim: int = 3,
        surface_channels: int = 1,
        volume_channels: int = 3,
        parameter_channels: int = 0,
        width: int = 64,
        modes: int = 4,
        latent_resolution: int = 20,
        num_fourier_blocks: int = 4,
        geometry_points: int = 0,
        source_chunk_size: int = 4096,
        query_chunk_size: int = 8192,
        iphi_width: int = 96,
        iphi_fourier_bands: int = 8,
        max_deformation: float = 0.25,
        normalize_source_transform: bool = True,
        enable_tf32: bool = True,
        **_unused,
    ):
        super().__init__()
        if int(spatial_dim) != 3:
            raise ValueError("This Geo-FNO adapter is implemented for 3D geometry-to-field tasks.")
        if int(num_fourier_blocks) < 2:
            raise ValueError("Geo-FNO requires an input transform and at least one Fourier-grid block.")
        if int(latent_resolution) < 2 * int(modes) + 1:
            raise ValueError("latent_resolution must be at least 2*modes+1.")
        self.surface_channels = int(surface_channels)
        self.volume_channels = int(volume_channels)
        self.width = int(width)
        self.modes = int(modes)
        self.latent_resolution = int(latent_resolution)
        self.geometry_points = max(0, int(geometry_points))
        self.source_chunk_size = max(1, int(source_chunk_size))
        self.query_chunk_size = max(1, int(query_chunk_size))
        self.normalize_source_transform = bool(normalize_source_transform)
        if bool(enable_tf32) and torch.cuda.is_available():
            # The NDFT contractions are fp32 matrix products. TF32 preserves
            # their stable accumulation path while using tensor cores on the
            # A5000 and L40S GPUs used for these experiments.
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision("high")

        frequencies_1d = _signed_frequencies(self.modes)
        frequency_grid = torch.stack(torch.meshgrid(frequencies_1d, frequencies_1d, frequencies_1d, indexing="ij"), dim=-1)
        self.register_buffer("frequencies", frequency_grid.reshape(-1, 3), persistent=False)
        grid_1d = torch.linspace(0.0, 1.0, self.latent_resolution, dtype=torch.float32)
        latent_grid = torch.stack(torch.meshgrid(grid_1d, grid_1d, grid_1d, indexing="ij"), dim=-1)
        self.register_buffer("latent_grid", latent_grid.reshape(-1, 3), persistent=False)
        self.register_buffer("latent_grid_volume", torch.tensor(float(self.latent_resolution ** 3)), persistent=False)

        self.iphi = _IPHI3D(parameter_channels, iphi_width, iphi_fourier_bands, max_deformation)
        self.source_lift = nn.Sequential(nn.Linear(3, self.width), nn.GELU(), nn.Linear(self.width, self.width))
        # These are the direct-domain counterparts of the official Geo-FNO
        # conv0 and conv4 spectral maps. They were missing in the initial 3D
        # adapter, which made the NDFT boundary stages only fixed projections.
        self.input_spectral_mix = _DirectSpectralMix(self.width, self.frequencies.shape[0])
        self.output_spectral_mix = _DirectSpectralMix(self.width, self.frequencies.shape[0])
        self.grid_bias = nn.ModuleList([nn.Linear(3, self.width) for _ in range(int(num_fourier_blocks))])
        self.spectral_blocks = nn.ModuleList(
            [_SpectralConv3d(self.width, self.modes) for _ in range(int(num_fourier_blocks) - 1)]
        )
        self.pointwise_blocks = nn.ModuleList(
            [nn.Conv3d(self.width, self.width, kernel_size=1) for _ in range(int(num_fourier_blocks) - 1)]
        )
        self.grid_condition = nn.Linear(parameter_channels, self.width) if int(parameter_channels) > 0 else None
        self.query_condition = nn.Linear(parameter_channels, self.width) if int(parameter_channels) > 0 else None
        self.surface_head = self._make_head(self.surface_channels)
        self.volume_head = self._make_head(self.volume_channels)

    def _make_head(self, output_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(self.width, self.width * 2),
            nn.GELU(),
            nn.Linear(self.width * 2, int(output_channels)),
        )

    def _select_geometry(self, geometry: torch.Tensor) -> torch.Tensor:
        if self.geometry_points <= 0 or geometry.shape[1] <= self.geometry_points:
            return geometry
        # The dataset already gives an epoch-seeded raw sample.  Striding it
        # only bounds the exact NDFT cost; it neither estimates nor corrects
        # sampling density.
        index = torch.linspace(0, geometry.shape[1] - 1, self.geometry_points, device=geometry.device).round().long()
        return geometry.index_select(1, index)

    def _basis(self, positions: torch.Tensor, sign: float) -> torch.Tensor:
        phase = torch.matmul(positions.float(), self.frequencies.t()) * (float(sign) * (2.0 * math.pi))
        return torch.complex(torch.cos(phase), torch.sin(phase))

    def _direct_transform(self, values: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Compute unweighted non-uniform Fourier coefficients in chunks."""
        coefficients = torch.zeros(
            values.shape[0], values.shape[-1], self.frequencies.shape[0], dtype=torch.cfloat, device=values.device
        )
        for start in range(0, positions.shape[1], self.source_chunk_size):
            stop = min(start + self.source_chunk_size, positions.shape[1])
            basis = self._basis(positions[:, start:stop], sign=-1.0)
            complex_values = torch.complex(values[:, start:stop].float(), torch.zeros_like(values[:, start:stop].float()))
            coefficients += torch.einsum("bnc,bnk->bck", complex_values, basis)
        if self.normalize_source_transform:
            # Global count normalization stabilizes the numerical scale across
            # fixed source budgets.  Every source point still has equal raw
            # weight, so spatial density remains visible to the base model.
            coefficients = coefficients / float(max(1, positions.shape[1]))
        return coefficients

    def _evaluate(self, coefficients: torch.Tensor, positions: torch.Tensor, normalizer: float) -> torch.Tensor:
        outputs = []
        for start in range(0, positions.shape[1], self.query_chunk_size):
            stop = min(start + self.query_chunk_size, positions.shape[1])
            basis = self._basis(positions[:, start:stop], sign=1.0)
            value = torch.einsum("bck,bnk->bnc", coefficients, basis).real / float(normalizer)
            outputs.append(value)
        return torch.cat(outputs, dim=1)

    def _encode(self, geometry: torch.Tensor, params: torch.Tensor | None) -> torch.Tensor:
        geometry = self._select_geometry(geometry)
        canonical = self.iphi(geometry, params)
        source = self.source_lift(geometry)
        coefficients = self.input_spectral_mix(self._direct_transform(source, canonical))
        grid_positions = self.latent_grid.unsqueeze(0).expand(geometry.shape[0], -1, -1)
        latent = self._evaluate(coefficients, grid_positions, normalizer=1.0)
        size = self.latent_resolution
        latent = latent.transpose(1, 2).reshape(geometry.shape[0], self.width, size, size, size)
        grid = self.latent_grid.reshape(size, size, size, 3).permute(3, 0, 1, 2).unsqueeze(0)
        latent = F.gelu(latent + self.grid_bias[0](grid.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3))
        if self.grid_condition is not None and params is not None:
            latent = latent + self.grid_condition(params.float()).view(params.shape[0], self.width, 1, 1, 1)
        for block_idx, (spectral, pointwise) in enumerate(zip(self.spectral_blocks, self.pointwise_blocks), start=1):
            bias = self.grid_bias[block_idx](grid.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3)
            latent = F.gelu(spectral(latent) + pointwise(latent) + bias)
        return latent

    def _decode(self, latent: torch.Tensor, queries: torch.Tensor, params: torch.Tensor | None, head: nn.Module) -> torch.Tensor:
        full_coefficients = torch.fft.fftn(latent.float(), dim=(-3, -2, -1))
        frequency_index = torch.remainder(self.frequencies.long(), self.latent_resolution)
        selected = full_coefficients[:, :, frequency_index[:, 0], frequency_index[:, 1], frequency_index[:, 2]]
        selected = self.output_spectral_mix(selected)
        canonical = self.iphi(queries, params)
        decoded = self._evaluate(selected, canonical, normalizer=float(self.latent_grid_volume.item()))
        if self.query_condition is not None and params is not None:
            decoded = decoded + self.query_condition(params.float()).unsqueeze(1)
        return head(decoded.to(dtype=latent.dtype))

    def forward(
        self,
        geo: torch.Tensor,
        surf_query_pos: torch.Tensor,
        vol_query_pos: torch.Tensor,
        params: torch.Tensor | None = None,
        geo_log_density: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del geo_log_density
        # NDFTs and FFTs stay fp32/complex even when the outer trainer uses
        # AMP. This avoids unstable half-precision Fourier accumulations.
        with torch.autocast(device_type=geo.device.type, enabled=False):
            latent = self._encode(geo.float(), params)
            surface = self._decode(latent, surf_query_pos.float(), params, self.surface_head)
            volume = self._decode(latent, vol_query_pos.float(), params, self.volume_head)
        return surface, volume

    @torch.inference_mode()
    def inference(
        self,
        geo: torch.Tensor,
        surf_query_pos: torch.Tensor,
        vol_query_pos: torch.Tensor,
        params: torch.Tensor | None = None,
        geo_log_density: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Consistency-evaluator entry point; Geo-FNO does not use density."""
        return self(geo, surf_query_pos, vol_query_pos, params, geo_log_density)
