"""SHIFT-Crash-only SMART adapter with an explicit case-conditioning contract.

SMART conditions every modulated MLP with one global parameter vector per
sample. This adapter instead normalizes the documented SHIFT design vector to
``[B, 1, C]`` once and injects one global token at architecture boundaries.

The adapter is intentionally separate from ``models.smart.smart.SMART`` so
the existing DrivAerML models and checkpoints are not changed.
"""

from __future__ import annotations

import torch

from models.smart.smart import Modulator, SMART, gather_point_values

try:
    from torch_cluster import knn as torch_cluster_knn
except ImportError:  # pragma: no cover - SHIFT training environment includes torch-cluster
    torch_cluster_knn = None


def _interpolate_local_support(query_pos, support_pos, support_features, neighbors, chunk_size):
    """Interpolate local geometry tokens at arbitrary query positions.

    SMART's native decoder sees only global latent tokens.  Terminal crash
    deformation additionally depends on the nearby undeformed surface shape,
    so this bounded KNN path supplies a local geometric residual without
    relying on mesh connectivity or point ordering.
    """
    batch_size = int(query_pos.shape[0])
    output = []
    neighbors = max(1, int(neighbors))
    chunk_size = max(1, int(chunk_size))
    for batch_index in range(batch_size):
        source_pos = support_pos[batch_index].float().contiguous()
        source_features = support_features[batch_index]
        queries = query_pos[batch_index].float().contiguous()
        if int(queries.shape[0]) == 0:
            output.append(source_features.new_empty((0, source_features.shape[-1] + 4)))
            continue
        k = min(neighbors, int(source_pos.shape[0]))
        chunks = []
        for start in range(0, int(queries.shape[0]), chunk_size):
            query_chunk = queries[start:start + chunk_size]
            if torch_cluster_knn is None:
                distance2 = torch.cdist(query_chunk, source_pos).square()
                neighbor_index = torch.topk(distance2, k=k, dim=-1, largest=False).indices
            else:
                edge_index = torch_cluster_knn(source_pos, query_chunk, k=k)
                query_index, neighbor_index = edge_index[0], edge_index[1]
                order = torch.argsort(query_index, stable=True)
                query_index = query_index[order]
                neighbor_index = neighbor_index[order]
                expected = torch.arange(query_chunk.shape[0], device=query_chunk.device).repeat_interleave(k)
                if not torch.equal(query_index, expected):
                    raise RuntimeError("torch_cluster.knn did not return ordered local SHIFT neighbors.")
                neighbor_index = neighbor_index.view(query_chunk.shape[0], k)
                neighbor_pos = source_pos[neighbor_index]
                distance2 = (query_chunk.unsqueeze(1) - neighbor_pos).square().sum(dim=-1)
            if torch_cluster_knn is None:
                neighbor_pos = source_pos[neighbor_index]
            offsets = neighbor_pos - query_chunk.unsqueeze(1)
            # Per-query temperature avoids a global length-scale assumption
            # across rails, cabin panels, and highly nonuniform mesh regions.
            temperature = distance2.detach().mean(dim=-1, keepdim=True).clamp_min(1.0e-10)
            weights = torch.softmax(-distance2 / temperature, dim=-1).to(dtype=source_features.dtype)
            interpolated = (weights.unsqueeze(-1) * source_features[neighbor_index]).sum(dim=1)
            mean_offset = (weights.unsqueeze(-1) * offsets).sum(dim=1).to(dtype=source_features.dtype)
            rms_radius = torch.sqrt((weights * distance2).sum(dim=-1, keepdim=True).clamp_min(1.0e-12)).to(
                dtype=source_features.dtype
            )
            chunks.append(torch.cat([interpolated, mean_offset, rms_radius], dim=-1))
        output.append(torch.cat(chunks, dim=0))
    return torch.stack(output, dim=0)


class ShiftCrashSMART(SMART):
    """Paper-faithful SMART with strict SHIFT-Crash parameter handling."""

    def __init__(self, *args, **kwargs):
        self.shift_crash_latent_dim = int(kwargs.get("latent_dim", 256))
        self.conditioning_input_channels = int(kwargs.pop("conditioning_input_channels", 6))
        raw_indices = kwargs.pop("conditioning_parameter_indices", (0, 1, 2, 3, 4, 5))
        self.conditioning_parameter_indices = tuple(int(index) for index in raw_indices)
        self.conditioning_channels = int(kwargs.get("parameter_channels", 0))
        self.local_query_support_points = int(kwargs.pop("local_query_support_points", 8192))
        self.local_query_neighbors = max(1, int(kwargs.pop("local_query_neighbors", 8)))
        self.local_query_chunk_size = max(1, int(kwargs.pop("local_query_chunk_size", 8192)))
        self.use_local_query_geometry = bool(kwargs.pop("use_local_query_geometry", True))
        if self.conditioning_channels <= 0:
            raise ValueError("ShiftCrashSMART requires positive parameter_channels.")
        if self.conditioning_input_channels <= 0:
            raise ValueError("SHIFT-Crash conditioning_input_channels must be positive.")
        if len(self.conditioning_parameter_indices) != self.conditioning_channels:
            raise ValueError(
                "conditioning_parameter_indices must contain exactly parameter_channels entries; "
                f"received {self.conditioning_parameter_indices} for {self.conditioning_channels} channels."
            )
        if any(index < 0 or index >= self.conditioning_input_channels for index in self.conditioning_parameter_indices):
            raise ValueError(
                "conditioning_parameter_indices must be valid indices into the raw conditioning vector; "
                f"received {self.conditioning_parameter_indices} for {self.conditioning_input_channels} inputs."
            )
        super().__init__(*args, **kwargs)
        # SHIFT-Crash continuous features are already standardized per channel
        # by the train-set statistics saved during preprocessing.  Applying a
        # LayerNorm across unrelated physical channels at each node destroys
        # their absolute information: in particular, rail thickness and yield
        # strength become almost identical whenever they are the largest two
        # entries at a rail node.  Keep SMART's projection, but feed it the
        # correctly channel-standardized values directly.
        if self.point_feature_encoder is not None:
            self.point_feature_norm = torch.nn.Identity()
        # A single global design token is a lower-capacity, explicit route
        # from case-level design inputs to the geometry encoder and query
        # decoder. Repeating a FiLM MLP in every block gave the model a strong
        # case-memorization path, while sparse random supports made global
        # geometry offsets unreliable to recover from coordinates alone.
        hidden_dim = max(16, min(128, int(getattr(self, "conditioning_hidden_dim", 32))))
        self.material_token = torch.nn.Sequential(
            torch.nn.Linear(self.conditioning_channels, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, self.shift_crash_latent_dim),
        )
        self.material_token_scale = torch.nn.Parameter(torch.tensor(1.0))
        # Terminal crash displacement has a substantial case-level component
        # that is shared by every query node. A linear head is deliberately
        # low capacity: it can represent the documented design-response trend
        # without forcing the point operator to rediscover it from sparse
        # supports, while the SMART decoder remains responsible for the local
        # deformation residual.
        self.global_response_head = torch.nn.Linear(
            self.conditioning_channels,
            self.surface_channels + self.volume_channels,
        )
        # A compact local residual restores query-to-surface context that the
        # original global-latent SMART decoder cannot retain at 65K queries.
        self.local_query_projection = torch.nn.Sequential(
            torch.nn.Linear(self.shift_crash_latent_dim + 4, self.shift_crash_latent_dim),
            torch.nn.LayerNorm(self.shift_crash_latent_dim, eps=1.0e-6),
            torch.nn.GELU(),
            torch.nn.Linear(self.shift_crash_latent_dim, self.shift_crash_latent_dim),
        )
        # Projection weights use SMART's small truncated-normal initialization,
        # so a unit residual is stable while still giving the local path useful
        # gradients from the first optimizer step.
        self.local_query_scale = torch.nn.Parameter(torch.tensor(1.0))
        # Make standalone checkpoint loading safe.  The trainer still applies
        # the YAML configuration explicitly, but a model reconstructed only
        # from a SHIFT-Crash checkpoint must not silently use direct FiLM.
        self.configure_conditioning("token_only")

    def configure_conditioning(self, mode="bounded_residual", residual_scale=0.25, shift_scale=0.25):
        """Configure stable conditioning without changing DrivAerML SMART."""
        mode = str(mode).lower().strip()
        if mode == "token_only":
            # Retain the original SMART blocks and checkpoint structure, but
            # deliberately remove their repeated parameter modulation.  The
            # dedicated case token is the sole global-condition route.
            for module in self.modules():
                if isinstance(module, Modulator):
                    module.configure_conditioning("bounded_residual", 0.0, 0.0)
            self.shift_crash_conditioning_mode = "token_only"
            return
        if mode not in {"direct", "residual", "bounded_residual"}:
            raise ValueError(f"Unsupported SHIFT-Crash conditioning mode: {mode!r}")
        for module in self.modules():
            if isinstance(module, Modulator):
                module.configure_conditioning(mode, residual_scale, shift_scale)
        self.shift_crash_conditioning_mode = mode

    def initialize_shift_crash_weights(self):
        """Use SMART's documented initialization for fresh SHIFT-Crash runs."""
        self.initialize_weights()

    def prepare_conditioning(self, params, batch_size=None):
        """Select documented design variables and make them broadcastable.

        The dataset returns six standardized pre-impact design variables.  The
        SHIFT terminal response is strongly controlled by the four geometry
        offsets as well as rail material; the configured token therefore uses
        all six values while retaining one shared, low-capacity pathway.
        """
        if params is None:
            raise ValueError("SHIFT-Crash SMART requires six standardized parameters per case.")
        if params.ndim == 2:
            params = params.unsqueeze(1)
        if params.ndim != 3 or params.shape[1] != 1 or params.shape[-1] != self.conditioning_input_channels:
            raise ValueError(
                f"SHIFT-Crash conditioning must have shape [B, {self.conditioning_input_channels}] or "
                f"[B, 1, {self.conditioning_input_channels}]; "
                f"received {tuple(params.shape)}."
            )
        if batch_size is not None and int(params.shape[0]) != int(batch_size):
            raise ValueError(
                f"Conditioning batch size {params.shape[0]} does not match geometry batch size {batch_size}."
            )
        if not bool(torch.isfinite(params).all().item()):
            raise ValueError("SHIFT-Crash conditioning contains NaN or Inf values.")
        index = torch.as_tensor(self.conditioning_parameter_indices, device=params.device, dtype=torch.long)
        return params.index_select(-1, index)

    def _sample_structure_aware_indices(self, geometry_features, budget, rare_budget, sampling_seeds=None, salt=0):
        """Sample anchors while retaining sparse front-rail evidence.

        The full SHIFT mesh has fewer than 1.2% rail points.  Purely random
        latent anchors therefore retain only a few rail points, despite those
        points carrying the material-dependent crush response.  This sampler
        operates only on the supplied view (never the hidden full cloud), so
        SATLoss7 still sees the intended sampling-density shifts.  The rest of
        each set is sampled uniformly without replacement.
        """
        batch_size, num_points = geometry_features.shape[:2]
        budget = min(max(int(budget), 1), num_points)
        rare_budget = min(max(int(rare_budget), 0), budget)
        if budget == num_points:
            return torch.arange(num_points, device=geometry_features.device, dtype=torch.long).view(1, -1).expand(batch_size, -1)

        rail_mask = geometry_features[..., -1] > 0.5
        if sampling_seeds is not None:
            if sampling_seeds.ndim != 1 or int(sampling_seeds.shape[0]) != batch_size:
                raise ValueError(
                    f"sampling_seeds must have shape [{batch_size}], got {tuple(sampling_seeds.shape)}."
                )
            seed_values = sampling_seeds.detach().to(device="cpu", dtype=torch.long).tolist()
        else:
            seed_values = [None] * batch_size

        indices = []
        for sample_index in range(batch_size):
            generator = None
            if seed_values[sample_index] is not None:
                generator = torch.Generator(device=geometry_features.device)
                generator.manual_seed(int(seed_values[sample_index]) + int(salt))
            rail_indices = torch.nonzero(rail_mask[sample_index], as_tuple=False).flatten()
            if rail_indices.numel() > rare_budget:
                rail_indices = rail_indices[
                    torch.randperm(rail_indices.numel(), device=rail_indices.device, generator=generator)[:rare_budget]
                ]
            chosen_count = int(rail_indices.numel())
            remaining_count = budget - chosen_count
            if remaining_count > 0:
                available = torch.ones(num_points, device=geometry_features.device, dtype=torch.bool)
                available[rail_indices] = False
                candidates = torch.nonzero(available, as_tuple=False).flatten()
                fill = candidates[
                    torch.randperm(candidates.numel(), device=candidates.device, generator=generator)[:remaining_count]
                ]
                sample_indices = torch.cat([rail_indices, fill], dim=0)
            else:
                sample_indices = rail_indices
            indices.append(
                sample_indices[
                    torch.randperm(sample_indices.numel(), device=sample_indices.device, generator=generator)
                ]
            )
        return torch.stack(indices, dim=0)

    @staticmethod
    def _sample_uniform_indices(geometry, budget, sampling_seeds=None, salt=0):
        """Uniform local-support samples from the supplied, never hidden, view."""
        batch_size, num_points = geometry.shape[:2]
        budget = min(max(int(budget), 1), num_points)
        if budget == num_points:
            return torch.arange(num_points, device=geometry.device, dtype=torch.long).view(1, -1).expand(batch_size, -1)
        if sampling_seeds is not None:
            seed_values = sampling_seeds.detach().to(device="cpu", dtype=torch.long).tolist()
        else:
            seed_values = [None] * batch_size
        indices = []
        for batch_index, seed in enumerate(seed_values):
            generator = None
            if seed is not None:
                generator = torch.Generator(device=geometry.device)
                generator.manual_seed(int(seed) + int(salt))
            indices.append(torch.randperm(num_points, device=geometry.device, generator=generator)[:budget])
        return torch.stack(indices, dim=0)

    def _case_context(self, params, dtype=None):
        if params.ndim != 3 or params.shape[1] != 1:
            raise ValueError(f"Expected prepared parameters [B,1,C], got {tuple(params.shape)}.")
        context = self.material_token(params[:, 0, :].float()) * self.material_token_scale
        # Parameters remain float32 for their normalization contract, while
        # the attention stream may be autocast to float16/bfloat16.  Returning
        # the context in the latent dtype avoids silently promoting every
        # encoder/decoder activation back to float32.
        return context if dtype is None else context.to(dtype=dtype)

    def encode(
        self,
        geo,
        params,
        geometry_features=None,
        geometry_part_ids=None,
        return_final=False,
        sampling_seeds=None,
    ):
        """Encode one supplied SHIFT-Crash point-cloud view.

        This is SMART's encoder with structure-aware index selection.  The
        attention and decoder topology are unchanged.
        """
        if geometry_features is None:
            return super().encode(
                geo, params, geometry_features=geometry_features,
                geometry_part_ids=geometry_part_ids, return_final=return_final,
            )
        input_geo = geo
        geo = geo * self.pos_scale_factor
        latent_idx = self._sample_structure_aware_indices(
            geometry_features,
            self.num_geo,
            rare_budget=min(128, max(32, self.num_geo // 8)),
            sampling_seeds=sampling_seeds,
            salt=1009,
        )
        latent_geo_pos = gather_point_values(geo, latent_idx)
        latent_geo_emb = self.pos_encoder(latent_geo_pos)
        latent_geo_emb = latent_geo_emb + self._encode_point_features(
            gather_point_values(geometry_features, latent_idx),
            gather_point_values(geometry_part_ids, latent_idx),
        )
        latent_geo_emb = latent_geo_emb + self._case_context(params, dtype=latent_geo_emb.dtype).unsqueeze(1)

        intermediate_latent_geometries = []
        for block in self.encoder_blocks:
            sub_idx = self._sample_structure_aware_indices(
                geometry_features,
                self.subsampled_geometry_points,
                rare_budget=min(512, max(128, self.subsampled_geometry_points // 8)),
                sampling_seeds=sampling_seeds,
                salt=10007 + 7919 * len(intermediate_latent_geometries),
            )
            sub_geo_pos = gather_point_values(geo, sub_idx)
            sub_geo_emb = self.pos_encoder(sub_geo_pos)
            sub_geo_emb = sub_geo_emb + self._encode_point_features(
                gather_point_values(geometry_features, sub_idx),
                gather_point_values(geometry_part_ids, sub_idx),
            )
            latent_geo_emb, encoder_cross_attention = block(
                latent_geo_emb,
                sub_geo_emb,
                params,
                latent_geometry_pos=latent_geo_pos,
                subsampled_geometry_pos=sub_geo_pos,
            )
            intermediate_latent_geometries.append(encoder_cross_attention)
        if return_final:
            local_idx = self._sample_uniform_indices(
                input_geo,
                self.local_query_support_points,
                sampling_seeds=sampling_seeds,
                salt=900001,
            )
            local_support_pos = gather_point_values(input_geo, local_idx)
            local_support_emb = self.pos_encoder(gather_point_values(geo, local_idx))
            local_support_emb = local_support_emb + self._encode_point_features(
                gather_point_values(geometry_features, local_idx),
                gather_point_values(geometry_part_ids, local_idx),
            )
            return intermediate_latent_geometries, latent_geo_pos, latent_geo_emb, local_support_pos, local_support_emb
        return intermediate_latent_geometries, latent_geo_pos

    def decode_features(
        self,
        intermediate_latent_geometries,
        latent_geo_pos,
        params,
        query_pos,
        query_features=None,
        query_part_ids=None,
        local_support_pos=None,
        local_support_features=None,
    ):
        raw_query_pos = query_pos
        query_pos = query_pos * self.pos_scale_factor
        query_emb = self.pos_encoder(query_pos)
        if self.point_feature_encoder is not None:
            query_emb = query_emb + self._encode_point_features(query_features, query_part_ids)
        query_emb = query_emb + self._case_context(params, dtype=query_emb.dtype).unsqueeze(1)
        if (
            self.use_local_query_geometry
            and local_support_pos is not None
            and local_support_features is not None
        ):
            local = _interpolate_local_support(
                raw_query_pos,
                local_support_pos,
                local_support_features,
                neighbors=self.local_query_neighbors,
                chunk_size=self.local_query_chunk_size,
            )
            query_emb = query_emb + self.local_query_scale.to(dtype=query_emb.dtype) * self.local_query_projection(local)
        for encoder_cross_attention, block in zip(intermediate_latent_geometries, self.decoder_blocks):
            query_emb = block(
                query_emb,
                encoder_cross_attention,
                params,
                queries_pos=query_pos,
                latent_geometry_pos=latent_geo_pos,
            )
        return query_emb

    def decode(
        self,
        intermediate_latent_geometries,
        latent_geo_pos,
        params,
        query_pos,
        query_features=None,
        query_part_ids=None,
        local_support_pos=None,
        local_support_features=None,
    ):
        query_emb = self.decode_features(
            intermediate_latent_geometries,
            latent_geo_pos,
            params,
            query_pos,
            query_features=query_features,
            query_part_ids=query_part_ids,
            local_support_pos=local_support_pos,
            local_support_features=local_support_features,
        )
        # Preserve the global design signal across pre-normalized decoder
        # blocks.  This is the same single token, not another conditioning
        # network or a per-layer FiLM path.
        local_response = self.mlp(
            query_emb + self._case_context(params, dtype=query_emb.dtype).unsqueeze(1)
        )
        global_response = self.global_response_head(params[:, 0, :].float()).to(dtype=local_response.dtype)
        return local_response + global_response.unsqueeze(1)

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
        sampling_seeds=None,
    ):
        params = self.prepare_conditioning(params, batch_size=geo.shape[0])
        intermediate, latent_pos, _latent_final, local_support_pos, local_support_features = self.encode(
            geo,
            params,
            geometry_features=geometry_features,
            geometry_part_ids=geometry_part_ids,
            sampling_seeds=sampling_seeds,
            return_final=True,
        )
        query_pos = torch.cat([surf_query_pos, vol_query_pos], dim=1)
        prediction = self.decode(
            intermediate,
            latent_pos,
            params,
            query_pos,
            query_features=query_features,
            query_part_ids=query_part_ids,
            local_support_pos=local_support_pos,
            local_support_features=local_support_features,
        )
        return (
            prediction[:, :surf_query_pos.shape[1], :self.surface_channels],
            prediction[:, surf_query_pos.shape[1]:, self.surface_channels:],
        )

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
        sampling_seeds=None,
    ):
        params = self.prepare_conditioning(params, batch_size=geo.shape[0])
        intermediate, latent_pos, _latent_final, local_support_pos, local_support_features = self.encode(
            geo,
            params,
            geometry_features=geometry_features,
            geometry_part_ids=geometry_part_ids,
            sampling_seeds=sampling_seeds,
            return_final=True,
        )

        def decode_in_chunks(points, features, part_ids):
            chunks = []
            for start in range(0, int(points.shape[1]), self.subregion_size):
                chunks.append(
                    self.decode(
                        intermediate,
                        latent_pos,
                        params,
                        points[:, start:start + self.subregion_size],
                        query_features=None if features is None else features[:, start:start + self.subregion_size],
                        query_part_ids=None if part_ids is None else part_ids[:, start:start + self.subregion_size],
                        local_support_pos=local_support_pos,
                        local_support_features=local_support_features,
                    )
                )
            return torch.cat(chunks, dim=1) if chunks else points.new_empty((points.shape[0], 0, self.surface_channels + self.volume_channels))

        surface = decode_in_chunks(surf_query_pos, query_features, query_part_ids)
        volume = decode_in_chunks(vol_query_pos, volume_query_features, volume_query_part_ids)
        return surface[..., :self.surface_channels], volume[..., self.surface_channels:]
