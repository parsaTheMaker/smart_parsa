"""Dataset adapter for the preprocessed SHIFT-Crash terminal-state data.

The adapter deliberately keeps the SHIFT representation separate from the
DrivAerML surface/volume dataset.  A case contains reference coordinates and
terminal displacement at the same nodes, so the reference cloud is both the
geometry input and the query domain.
"""

from __future__ import annotations

import json
import os
from collections import OrderedDict
from multiprocessing import Value
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class ShiftCrashDataset(Dataset):
    """Read one deterministic split of the preprocessed SHIFT-Crash dataset.

    ``geometry_points=0`` means that the complete reference cloud is returned.
    This is used only as the source cloud for SATLoss7 view sampling.  The
    default finite budgets are scaled from the complete DrivAerML source size.
    """

    CACHE_VERSION = "shift_crash_voxel_density_v3"

    def __init__(
        self,
        root,
        split="train",
        geometry_points=32768,
        query_points=16384,
        seed=42,
        epoch_seeded_sampling=True,
        deterministic_geometry_sampling=False,
        deterministic_query_sampling=False,
        return_log_density=False,
        density_voxel_resolution=96,
        coordinate_normalization="global_bounds",
    ):
        self.root = Path(root).expanduser().resolve()
        self.split = str(split)
        self.geometry_points = int(geometry_points)
        self.query_points = int(query_points)
        self.seed = int(seed)
        self.epoch_seeded_sampling = bool(epoch_seeded_sampling)
        self.deterministic_geometry_sampling = bool(deterministic_geometry_sampling)
        self.deterministic_query_sampling = bool(deterministic_query_sampling)
        self.return_log_density = bool(return_log_density)
        self.density_voxel_resolution = max(4, int(density_voxel_resolution))
        self.coordinate_normalization = str(coordinate_normalization).lower().strip()
        if self.coordinate_normalization not in {"global_bounds", "per_case_centered_extent"}:
            raise ValueError(
                "coordinate_normalization must be 'global_bounds' or 'per_case_centered_extent', got "
                f"{coordinate_normalization!r}."
            )

        if not self.root.is_dir():
            raise FileNotFoundError(f"SHIFT-Crash data root does not exist: {self.root}")
        if self.geometry_points < 0 or self.query_points <= 0:
            raise ValueError("geometry_points must be non-negative and query_points must be positive.")

        with (self.root / "splits.json").open("r", encoding="utf-8") as handle:
            splits = json.load(handle)
        if self.split not in splits:
            raise ValueError(f"Unknown SHIFT-Crash split {self.split!r}; available: {sorted(splits)}")
        self.case_ids = tuple(str(case_id) for case_id in splits[self.split])
        if not self.case_ids:
            raise ValueError(f"SHIFT-Crash split {self.split!r} is empty.")
        self._validate_normalization_provenance(splits)

        self.cases_root = self.root / "cases"
        self.min_position, self.max_position = self._load_array("reference_position_minmax.npy")
        self.displacement_mean, self.displacement_std = self._load_array("terminal_displacement_stats.npy")
        self.parameter_mean, self.parameter_std = self._load_array("parameter_stats.npy")
        self.static_feature_mean, self.static_feature_std = self._load_array("static_geometry_feature_stats.npy")
        if self.static_feature_mean.shape != (7,) or self.static_feature_std.shape != (7,):
            raise ValueError(
                "SHIFT-Crash static_geometry_feature_stats.npy must contain seven continuous feature statistics."
            )
        self.position_span = np.maximum(self.max_position - self.min_position, 1.0e-6)
        self.displacement_std = np.maximum(self.displacement_std, 1.0e-6)
        self.parameter_std = np.maximum(self.parameter_std, 1.0e-6)
        self.static_feature_std = np.maximum(self.static_feature_std, 1.0e-6)

        self._shared_epoch = Value("i", 0, lock=False)
        # Persistent DataLoader workers can visit every case over a long run.
        # Keep these caches bounded so workers do not retain thousands of open
        # memmaps or density arrays.
        self._memmap_cache = OrderedDict()
        self._memmap_cache_max_entries = 32
        self._static_feature_cache = OrderedDict()
        self._part_id_cache = OrderedDict()
        self._rail_mask_cache = OrderedDict()
        self._feature_cache_max_entries = 16
        self._params_cache = {}
        self._density_ram_cache = OrderedDict()
        self._density_ram_cache_max_entries = 16
        # These RNGs are initialized lazily in each process.  That avoids
        # paying OS-entropy setup costs for every query and remains independent
        # after DataLoader workers fork.
        self._query_rng = None
        self._unseeded_geometry_rng = None

        self.density_cache_root = self.root / ".shift_crash_cache" / self.CACHE_VERSION
        if self.return_log_density:
            self.density_cache_root.mkdir(parents=True, exist_ok=True)

        print(
            f"[SHIFT-Crash] split={self.split}, cases={len(self.case_ids)}, "
            f"geometry_points={'full' if self.geometry_points == 0 else self.geometry_points}, "
            f"query_points={self.query_points}, return_log_density={self.return_log_density}, "
            f"coordinate_normalization={self.coordinate_normalization}"
        )

    def _validate_normalization_provenance(self, splits):
        """Reject train statistics built for a different active split.

        Target and continuous-feature normalization must be estimated from
        the current training cases only.  The optional provenance is written
        by ``recompute_shift_crash_train_stats.py``; older preprocessing roots
        remain readable, while rebuilt roots fail loudly instead of leaking
        validation cases through stale statistics.
        """
        provenance_path = self.root / "normalization_provenance.json"
        if not provenance_path.is_file():
            return
        with provenance_path.open("r", encoding="utf-8") as handle:
            provenance = json.load(handle)
        recorded_ids = tuple(str(case_id) for case_id in provenance.get("case_ids", ()))
        active_train_ids = tuple(str(case_id) for case_id in splits.get("train", ()))
        if not recorded_ids:
            raise ValueError(f"Normalization provenance is missing train case IDs: {provenance_path}")
        if set(recorded_ids) != set(active_train_ids) or len(recorded_ids) != len(active_train_ids):
            raise ValueError(
                "SHIFT-Crash normalization statistics do not match the active train split. "
                "Run smart/scripts/recompute_shift_crash_train_stats.py with --overwrite before training."
            )

    def _load_array(self, name):
        path = self.root / name
        if not path.is_file():
            raise FileNotFoundError(f"Missing SHIFT-Crash normalization file: {path}")
        return np.asarray(np.load(path), dtype=np.float32)

    def __len__(self):
        return len(self.case_ids)

    def set_epoch(self, epoch):
        self._shared_epoch.value = int(epoch)

    def _case_array(self, case_id):
        data = self._memmap_cache.get(case_id)
        if data is None:
            path = self.cases_root / case_id / "geometry_and_terminal_displacement.npy"
            if not path.is_file():
                raise FileNotFoundError(f"Missing SHIFT-Crash case array: {path}")
            data = np.load(path, mmap_mode="r")
            if data.ndim != 2 or data.shape[1] != 6:
                raise ValueError(f"Expected [N,6] case array at {path}, got {data.shape}")
            self._memmap_cache[case_id] = data
            while len(self._memmap_cache) > self._memmap_cache_max_entries:
                self._memmap_cache.popitem(last=False)
        else:
            self._memmap_cache.move_to_end(case_id)
        return data

    def _case_params(self, case_id):
        params = self._params_cache.get(case_id)
        if params is None:
            path = self.cases_root / case_id / "params.npy"
            if not path.is_file():
                raise FileNotFoundError(f"Missing SHIFT-Crash parameter file: {path}")
            params = np.asarray(np.load(path), dtype=np.float32)
            if params.shape != self.parameter_mean.shape:
                raise ValueError(f"Expected parameter shape {self.parameter_mean.shape}, got {params.shape} at {path}")
            self._params_cache[case_id] = params
        return params

    def _case_static_inputs(self, case_id, num_nodes):
        """Load static, pre-impact node attributes without copying full cases."""
        static_features = self._static_feature_cache.get(case_id)
        part_ids = self._part_id_cache.get(case_id)
        rail_mask = self._rail_mask_cache.get(case_id)
        if static_features is None:
            static_path = self.cases_root / case_id / "static_geometry_features.npy"
            static_features = np.load(static_path, mmap_mode="r")
            if static_features.shape != (num_nodes, 7):
                raise ValueError(f"Expected static feature shape {(num_nodes, 7)} at {static_path}, got {static_features.shape}")
            self._static_feature_cache[case_id] = static_features
        if part_ids is None:
            part_path = self.cases_root / case_id / "part_id_embedding_indices.npy"
            part_ids = np.load(part_path, mmap_mode="r")
            if part_ids.shape != (num_nodes,):
                raise ValueError(f"Expected part-id shape {(num_nodes,)} at {part_path}, got {part_ids.shape}")
            self._part_id_cache[case_id] = part_ids
        if rail_mask is None:
            mask_path = self.cases_root / case_id / "front_rail_mask.npy"
            rail_mask = np.load(mask_path, mmap_mode="r")
            if rail_mask.shape != (num_nodes,):
                raise ValueError(f"Expected rail-mask shape {(num_nodes,)} at {mask_path}, got {rail_mask.shape}")
            self._rail_mask_cache[case_id] = rail_mask
        for cache in (self._static_feature_cache, self._part_id_cache, self._rail_mask_cache):
            cache.move_to_end(case_id)
            while len(cache) > self._feature_cache_max_entries:
                cache.popitem(last=False)
        return static_features, part_ids, rail_mask

    def _geometry_rng(self, index):
        if self.deterministic_geometry_sampling:
            return np.random.default_rng(np.random.SeedSequence([self.seed, int(index), 0]))
        epoch = int(self._shared_epoch.value)
        # Match DrivAerML: geometry sampling is deterministic within an epoch
        # when geometry_epoch_seeded_sampling is enabled.
        seed_sequence = np.random.SeedSequence([self.seed, epoch, int(index), 0])
        return np.random.default_rng(seed_sequence)

    @staticmethod
    def _sample_indices(num_available, budget, rng):
        budget = int(budget)
        if budget <= 0 or budget >= num_available:
            return np.arange(num_available, dtype=np.int64)
        return rng.choice(num_available, size=budget, replace=False).astype(np.int64, copy=False)

    def _normalize_positions(self, coordinates):
        if self.coordinate_normalization == "per_case_centered_extent":
            coordinate_min = coordinates.min(axis=0, keepdims=True)
            coordinate_max = coordinates.max(axis=0, keepdims=True)
            center = 0.5 * (coordinate_min + coordinate_max)
            # One isotropic scale preserves aspect ratio while removing
            # arbitrary global placement and overall scale.
            half_extent = max(0.5 * float((coordinate_max - coordinate_min).max()), 1.0e-6)
            return ((coordinates.astype(np.float32, copy=False) - center) / half_extent).astype(np.float32, copy=False)
        # Keep one train-defined physical coordinate chart for every case.
        # Validation designs may legitimately extend a little beyond the train
        # extrema; clipping those locations aliases distinct vehicle nodes at
        # 0 or 1 and destroys the correspondence signal the operator needs.
        return ((coordinates.astype(np.float32, copy=False) - self.min_position) / self.position_span).astype(
            np.float32, copy=False
        )

    def _density_path(self, case_id):
        return self.density_cache_root / (
            f"{case_id}_r{self.density_voxel_resolution}_{self.coordinate_normalization}.npy"
        )

    def _load_or_build_log_density(self, case_id, data):
        cached = self._density_ram_cache.get(case_id)
        if cached is not None and cached.shape == (data.shape[0],):
            self._density_ram_cache.move_to_end(case_id)
            return cached

        path = self._density_path(case_id)
        if path.is_file():
            density = np.asarray(np.load(path), dtype=np.float32)
            if density.shape == (data.shape[0],):
                self._density_ram_cache[case_id] = density
                self._density_ram_cache.move_to_end(case_id)
                while len(self._density_ram_cache) > self._density_ram_cache_max_entries:
                    self._density_ram_cache.popitem(last=False)
                return density

        coordinates = self._normalize_positions(np.asarray(data[:, :3]))
        if self.coordinate_normalization == "per_case_centered_extent":
            coordinates = 0.5 * (coordinates + 1.0)
        resolution = self.density_voxel_resolution
        cell = np.floor(coordinates * resolution).astype(np.int32)
        cell = np.clip(cell, 0, resolution - 1)
        linear = (cell[:, 0].astype(np.int64) * resolution + cell[:, 1]) * resolution + cell[:, 2]
        _, inverse, counts = np.unique(linear, return_inverse=True, return_counts=True)
        density = np.log(np.maximum(counts[inverse], 1)).astype(np.float32)

        # Atomic replacement prevents an interrupted worker from leaving a
        # partially written cache file behind.
        temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        with temporary.open("wb") as handle:
            np.save(handle, density)
        try:
            os.replace(temporary, path)
        except FileNotFoundError:
            # Another worker completed the same case between our computation
            # and replace; the valid file can be used by this worker as well.
            pass
        self._density_ram_cache[case_id] = density
        self._density_ram_cache.move_to_end(case_id)
        while len(self._density_ram_cache) > self._density_ram_cache_max_entries:
            self._density_ram_cache.popitem(last=False)
        return density

    def __getitem__(self, index):
        case_id = self.case_ids[int(index)]
        data = self._case_array(case_id)
        num_nodes = int(data.shape[0])
        if self.deterministic_geometry_sampling or self.epoch_seeded_sampling:
            geometry_rng = self._geometry_rng(index)
        else:
            if self._unseeded_geometry_rng is None:
                self._unseeded_geometry_rng = np.random.default_rng()
            geometry_rng = self._unseeded_geometry_rng
        # Training queries remain stochastic, but validation must use the same
        # geometry/query subsets every epoch.  Otherwise validation noise is
        # accidentally treated as generalization and drives checkpoint choice.
        if self.deterministic_query_sampling:
            query_rng = np.random.default_rng(np.random.SeedSequence([self.seed, int(index), 1]))
        else:
            if self._query_rng is None:
                self._query_rng = np.random.default_rng()
            query_rng = self._query_rng
        geometry_indices = self._sample_indices(num_nodes, self.geometry_points, geometry_rng)
        query_indices = self._sample_indices(num_nodes, self.query_points, query_rng)

        coordinates = np.asarray(data[:, :3])
        normalized_coordinates = self._normalize_positions(coordinates)
        displacement = np.asarray(data[:, 3:6])
        static_features, part_ids, rail_mask = self._case_static_inputs(case_id, num_nodes)
        # Geometry and query points must use the same case-level coordinate
        # frame.  Canonicalizing each independently sampled subset would make
        # the training pair internally inconsistent.
        geometry = normalized_coordinates[geometry_indices]
        query = normalized_coordinates[query_indices]
        geometry_static = (
            np.asarray(static_features[geometry_indices], dtype=np.float32) - self.static_feature_mean
        ) / self.static_feature_std
        query_static = (
            np.asarray(static_features[query_indices], dtype=np.float32) - self.static_feature_mean
        ) / self.static_feature_std
        geometry_features = np.concatenate(
            [geometry_static, np.asarray(rail_mask[geometry_indices, None], dtype=np.float32)],
            axis=-1,
        ).astype(np.float32, copy=False)
        query_features = np.concatenate(
            [query_static, np.asarray(rail_mask[query_indices, None], dtype=np.float32)],
            axis=-1,
        ).astype(np.float32, copy=False)
        target = ((displacement[query_indices].astype(np.float32, copy=False) - self.displacement_mean) / self.displacement_std).astype(np.float32, copy=False)
        parameters = ((self._case_params(case_id) - self.parameter_mean) / self.parameter_std).astype(np.float32, copy=False)

        sample = {
            "geometry": torch.from_numpy(np.ascontiguousarray(geometry)),
            "query": torch.from_numpy(np.ascontiguousarray(query)),
            "geometry_features": torch.from_numpy(np.ascontiguousarray(geometry_features)),
            "query_features": torch.from_numpy(np.ascontiguousarray(query_features)),
            "geometry_part_ids": torch.from_numpy(np.ascontiguousarray(part_ids[geometry_indices].astype(np.int64, copy=False))),
            "query_part_ids": torch.from_numpy(np.ascontiguousarray(part_ids[query_indices].astype(np.int64, copy=False))),
            "target": torch.from_numpy(np.ascontiguousarray(target)),
            "params": torch.from_numpy(np.ascontiguousarray(parameters)),
            "case_id": case_id,
        }
        if self.return_log_density:
            if self.geometry_points != 0:
                raise ValueError("return_log_density=True requires geometry_points=0 so density aligns with the full source cloud.")
            density = self._load_or_build_log_density(case_id, data)
            sample["geometry_log_density"] = torch.from_numpy(np.ascontiguousarray(density))
        return sample
