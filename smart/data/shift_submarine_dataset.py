"""Dataset reader for the preprocessed SHIFT-Submarine sample dataset.

This adapter intentionally does not reuse the DrivAerML reader.  Submarine
cases use the generic ``surface_data.npy``/``volume_data.npy`` layout emitted
by ``stream_preprocess_shift_submarine.py`` and contain four target channels
on each domain:

* surface: pressure and the three wall-shear components;
* volume: pressure and the three velocity components.

Coordinates are normalized with global training-split bounds.  Targets use
training-split statistics, so neither coordinate nor target normalization
leaks validation cases.  SATLOSS7 can request the cached surface KDE values;
the density vector always follows the same point indexing as the geometry.
"""

from __future__ import annotations

import json
from collections import OrderedDict
from multiprocessing import Value
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    from utils.geometry_density import estimate_log_sampling_density
except ImportError:  # pragma: no cover
    from smart.utils.geometry_density import estimate_log_sampling_density


class ShiftSubmarineDataset(Dataset):
    """Read one train/test split from a streamed SHIFT-Submarine root."""

    CACHE_VERSION = "shift_submarine_train_stats_v1"
    SURFACE_FIELDS = ("pressure", "wall_shear_x", "wall_shear_y", "wall_shear_z")
    VOLUME_FIELDS = ("pressure", "velocity_x", "velocity_y", "velocity_z")

    def __init__(
        self,
        saved_folder,
        if_test=False,
        geometry_points=131072,
        surface_points=65536,
        volume_points=65536,
        scale_positions=False,
        coordinate_normalization="global_train_bounds",
        split_seed=42,
        test_fraction=0.2,
        geometry_epoch_seeded_sampling=False,
        return_geometry_density=False,
        geometry_density_knn_k=16,
        geometry_density_neighbor_hops=1,
        geometry_density_estimator="kde",
        geometry_density_cache_dtype="float16",
        **_unused,
    ):
        del scale_positions  # Kept for compatibility with the shared dataset registry.
        self.coordinate_normalization = str(coordinate_normalization).strip().lower()
        if self.coordinate_normalization not in {"global_train_bounds", "global_bounds"}:
            raise ValueError(
                "SHIFT-Submarine supports only global_train_bounds coordinate normalization; "
                f"got {coordinate_normalization!r}."
            )
        self.root = Path(saved_folder).expanduser().resolve()
        self.if_test = bool(if_test)
        self.geometry_points = int(geometry_points)
        self.surface_points = int(surface_points)
        self.volume_points = int(volume_points)
        self.geometry_epoch_seeded_sampling = bool(geometry_epoch_seeded_sampling)
        self.return_geometry_density = bool(return_geometry_density)
        self.geometry_density_knn_k = max(1, int(geometry_density_knn_k))
        self.geometry_density_neighbor_hops = max(0, int(geometry_density_neighbor_hops))
        self.geometry_density_estimator = str(geometry_density_estimator)
        self.geometry_density_cache_dtype = str(geometry_density_cache_dtype)
        self.split_seed = int(split_seed)
        self.test_fraction = float(test_fraction)
        self._shared_epoch = Value("i", 0, lock=False)
        self._memmap_cache = OrderedDict()
        self._density_cache = OrderedDict()
        self._memmap_cache_max_entries = 8
        self._density_cache_max_entries = 16

        if not self.root.is_dir():
            raise FileNotFoundError(f"SHIFT-Submarine data root does not exist: {self.root}")
        if self.geometry_points < 0 or self.surface_points <= 0 or self.volume_points <= 0:
            raise ValueError("geometry_points must be non-negative; surface_points and volume_points must be positive.")

        self.all_ids = self._discover_ids()
        train_ids, test_ids = self._load_split_ids(self.all_ids)
        self.training_ids = tuple(train_ids)
        self.test_ids = tuple(test_ids)
        self.data = self.test_ids if self.if_test else self.training_ids
        if not self.data:
            raise ValueError("The selected SHIFT-Submarine split is empty.")

        self.surface_field_names = list(self.SURFACE_FIELDS)
        self.volume_field_names = list(self.VOLUME_FIELDS)
        self._load_train_statistics()
        density_tag = "enabled" if self.return_geometry_density else "disabled"
        print(
            f"[SHIFT-Submarine] split={'test' if self.if_test else 'train'}, cases={len(self.data)}, "
            f"geometry_points={'full' if self.geometry_points == 0 else self.geometry_points}, "
            f"surface_queries={self.surface_points}, volume_queries={self.volume_points}, "
            f"geometry_density={density_tag}"
        )

    def _discover_ids(self):
        ids = []
        for entry in self.root.glob("run_*"):
            if not entry.is_dir():
                continue
            try:
                run_id = int(entry.name.split("_", 1)[1])
            except (IndexError, ValueError):
                continue
            required = (
                entry / "_COMPLETE.json",
                entry / "surface_coords.npy",
                entry / "surface_data.npy",
                entry / "volume_coords.npy",
                entry / "volume_data.npy",
            )
            if all(path.is_file() for path in required):
                ids.append(run_id)
        ids = sorted(set(ids))
        if not ids:
            raise FileNotFoundError(f"No complete SHIFT-Submarine run_* folders found in {self.root}")
        return ids

    def _load_split_ids(self, available):
        available = set(int(value) for value in available)
        manifest_path = self.root / "preprocessed_manifest.json"
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            train_ids = [int(value) for value in manifest.get("train_ids", []) if int(value) in available]
            test_ids = [int(value) for value in manifest.get("test_ids", []) if int(value) in available]
            if train_ids and test_ids:
                return sorted(set(train_ids)), sorted(set(test_ids))

        ids = np.asarray(sorted(available), dtype=np.int64)
        rng = np.random.default_rng(self.split_seed)
        ids = ids[rng.permutation(ids.shape[0])]
        if ids.size == 1:
            return [int(ids[0])], [int(ids[0])]
        test_count = max(1, min(ids.size - 1, int(round(ids.size * self.test_fraction))))
        return sorted(ids[test_count:].tolist()), sorted(ids[:test_count].tolist())

    def _stats_paths(self):
        return (
            self.root / f"surface_stats_{self.CACHE_VERSION}.npy",
            self.root / f"volume_stats_{self.CACHE_VERSION}.npy",
            self.root / f"position_stats_{self.CACHE_VERSION}.npy",
            self.root / f"stats_provenance_{self.CACHE_VERSION}.json",
        )

    def _stats_are_current(self, provenance_path):
        if not provenance_path.is_file():
            return False
        try:
            provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
            return provenance.get("train_ids") == list(self.training_ids)
        except (OSError, ValueError, TypeError):
            return False

    @staticmethod
    def _safe_std(sum_value, square_sum, count):
        if count <= 1:
            return np.ones_like(sum_value, dtype=np.float64)
        variance = (square_sum - (sum_value * sum_value) / float(count)) / float(count - 1)
        return np.sqrt(np.maximum(variance, 1.0e-12))

    def _compute_train_statistics(self):
        surface_sum = np.zeros(4, dtype=np.float64)
        surface_sq_sum = np.zeros(4, dtype=np.float64)
        volume_sum = np.zeros(4, dtype=np.float64)
        volume_sq_sum = np.zeros(4, dtype=np.float64)
        surface_count = 0
        volume_count = 0
        position_min = np.full(3, np.inf, dtype=np.float64)
        position_max = np.full(3, -np.inf, dtype=np.float64)
        for run_id in self.training_ids:
            metadata = json.loads((self._run_dir(run_id) / "case_metadata.json").read_text(encoding="utf-8"))
            surface_sum += np.asarray(metadata["surface_sum"], dtype=np.float64)
            surface_sq_sum += np.asarray(metadata["surface_sq_sum"], dtype=np.float64)
            volume_sum += np.asarray(metadata["volume_sum"], dtype=np.float64)
            volume_sq_sum += np.asarray(metadata["volume_sq_sum"], dtype=np.float64)
            surface_count += int(metadata["surface_count"])
            volume_count += int(metadata["volume_count"])
            position_min = np.minimum(position_min, np.asarray(metadata["position_min"], dtype=np.float64))
            position_max = np.maximum(position_max, np.asarray(metadata["position_max"], dtype=np.float64))
        if not np.isfinite(position_min).all() or not np.isfinite(position_max).all():
            raise ValueError("Training positions contain no finite bounds.")
        surface_stats = np.stack(
            [surface_sum / float(surface_count), self._safe_std(surface_sum, surface_sq_sum, surface_count)]
        ).astype(np.float32)
        volume_stats = np.stack(
            [volume_sum / float(volume_count), self._safe_std(volume_sum, volume_sq_sum, volume_count)]
        ).astype(np.float32)
        position_stats = np.stack([position_min, position_max]).astype(np.float32)
        return surface_stats, volume_stats, position_stats

    @staticmethod
    def _atomic_save(path, array):
        temporary = path.with_suffix(path.suffix + ".partial")
        with temporary.open("wb") as handle:
            np.save(handle, np.asarray(array), allow_pickle=False)
        temporary.replace(path)

    def _load_train_statistics(self):
        surface_path, volume_path, position_path, provenance_path = self._stats_paths()
        if not (
            surface_path.is_file()
            and volume_path.is_file()
            and position_path.is_file()
            and self._stats_are_current(provenance_path)
        ):
            surface_stats, volume_stats, position_stats = self._compute_train_statistics()
            self._atomic_save(surface_path, surface_stats)
            self._atomic_save(volume_path, volume_stats)
            self._atomic_save(position_path, position_stats)
            temporary = provenance_path.with_suffix(provenance_path.suffix + ".partial")
            temporary.write_text(
                json.dumps({"train_ids": list(self.training_ids), "version": self.CACHE_VERSION}, indent=2) + "\n",
                encoding="utf-8",
            )
            temporary.replace(provenance_path)
        surface_stats = np.load(surface_path)
        volume_stats = np.load(volume_path)
        position_stats = np.load(position_path)
        if surface_stats.shape != (2, 4) or volume_stats.shape != (2, 4) or position_stats.shape != (2, 3):
            raise ValueError(
                f"Invalid SHIFT-Submarine statistics shapes: surface={surface_stats.shape}, "
                f"volume={volume_stats.shape}, positions={position_stats.shape}."
            )
        self.mean_surf_data = torch.from_numpy(np.asarray(surface_stats[0], dtype=np.float32))
        self.std_surf_data = torch.from_numpy(np.maximum(surface_stats[1], 1.0e-12).astype(np.float32))
        self.mean_vol_data = torch.from_numpy(np.asarray(volume_stats[0], dtype=np.float32))
        self.std_vol_data = torch.from_numpy(np.maximum(volume_stats[1], 1.0e-12).astype(np.float32))
        self.min_pos = torch.from_numpy(np.asarray(position_stats[0], dtype=np.float32))
        self.max_pos = torch.from_numpy(np.asarray(position_stats[1], dtype=np.float32))
        self.position_span = torch.clamp(self.max_pos - self.min_pos, min=1.0e-12)

    def _run_dir(self, run_id):
        return self.root / f"run_{int(run_id)}"

    def _get_arrays(self, run_id):
        key = int(run_id)
        arrays = self._memmap_cache.get(key)
        if arrays is None:
            run_dir = self._run_dir(key)
            arrays = {
                "surface_coords": np.load(run_dir / "surface_coords.npy", mmap_mode="r"),
                "surface_data": np.load(run_dir / "surface_data.npy", mmap_mode="r"),
                "volume_coords": np.load(run_dir / "volume_coords.npy", mmap_mode="r"),
                "volume_data": np.load(run_dir / "volume_data.npy", mmap_mode="r"),
            }
            self._memmap_cache[key] = arrays
        self._memmap_cache.move_to_end(key)
        while len(self._memmap_cache) > self._memmap_cache_max_entries:
            self._memmap_cache.popitem(last=False)
        return arrays

    def _density_path(self, run_id):
        run_dir = self._run_dir(run_id)
        pattern = f"geometry_log_density_*_k{self.geometry_density_knn_k}_h{self.geometry_density_neighbor_hops}_*.npy"
        candidates = sorted(run_dir.glob(pattern))
        return candidates[0] if candidates else None

    def _load_density(self, run_id, coords):
        key = int(run_id)
        cached = self._density_cache.get(key)
        if cached is not None and int(cached.shape[0]) == int(coords.shape[0]):
            return cached
        path = self._density_path(key)
        if path is not None:
            density = torch.from_numpy(np.asarray(np.load(path, mmap_mode="r"), dtype=np.float32))
            if int(density.shape[0]) != int(coords.shape[0]):
                raise ValueError(f"Density/geometry size mismatch for run_{key}: {density.shape} vs {coords.shape}")
        else:
            raw = np.asarray(coords, dtype=np.float32)
            lower = raw.min(axis=0)
            upper = raw.max(axis=0)
            normalized = (raw - lower) / np.maximum(upper - lower, 1.0e-12)
            density = estimate_log_sampling_density(
                torch.from_numpy(normalized).unsqueeze(0),
                knn_k=self.geometry_density_knn_k,
                neighbor_hops=self.geometry_density_neighbor_hops,
                estimator=self.geometry_density_estimator,
            ).squeeze(0).cpu().float()
        self._density_cache[key] = density
        self._density_cache.move_to_end(key)
        while len(self._density_cache) > self._density_cache_max_entries:
            self._density_cache.popitem(last=False)
        return density

    def set_epoch(self, epoch):
        self._shared_epoch.value = int(epoch)

    def get_epoch(self):
        return int(self._shared_epoch.value)

    def _sample_indices(self, count, target, rng, replace=False):
        if target <= 0 or target >= count:
            return np.arange(count, dtype=np.int64)
        return rng.choice(count, size=target, replace=replace).astype(np.int64, copy=False)

    def _geometry_rng(self, run_id):
        if not self.geometry_epoch_seeded_sampling:
            return np.random.default_rng()
        return np.random.default_rng(np.random.SeedSequence([self.split_seed, self.get_epoch(), int(run_id), 0]))

    @staticmethod
    def _normalize_positions(coords, min_pos, position_span):
        return (np.asarray(coords, dtype=np.float32) - min_pos[None, :]) / position_span[None, :]

    @staticmethod
    def _owned_float_tensor(values):
        # Full-geometry SATLOSS samples can be read-only np.memmap views.
        # Give DataLoader collation owned storage instead of that view.
        return torch.from_numpy(np.array(values, dtype=np.float32, order="C", copy=True))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        run_id = int(self.data[index])
        arrays = self._get_arrays(run_id)
        surface_coords = arrays["surface_coords"]
        surface_data = arrays["surface_data"]
        volume_coords = arrays["volume_coords"]
        volume_data = arrays["volume_data"]
        ns = int(surface_coords.shape[0])
        nv = int(volume_coords.shape[0])
        if surface_data.shape != (ns, 4) or volume_data.shape != (nv, 4):
            raise ValueError(f"Unexpected channel shape in run_{run_id}: surface={surface_data.shape}, volume={volume_data.shape}")

        geometry_rng = self._geometry_rng(run_id)
        query_rng = np.random.default_rng()
        geo_idx = self._sample_indices(ns, self.geometry_points, geometry_rng, replace=False)
        surf_idx = self._sample_indices(ns, self.surface_points, query_rng, replace=False)
        vol_idx = self._sample_indices(nv, self.volume_points, query_rng, replace=False)

        min_pos = self.min_pos.numpy()
        position_span = self.position_span.numpy()
        geo = self._normalize_positions(surface_coords[geo_idx], min_pos, position_span)
        surf = self._normalize_positions(surface_coords[surf_idx], min_pos, position_span)
        vol = self._normalize_positions(volume_coords[vol_idx], min_pos, position_span)
        surf_target = (np.asarray(surface_data[surf_idx], dtype=np.float32) - self.mean_surf_data.numpy()) / self.std_surf_data.numpy()
        vol_target = (np.asarray(volume_data[vol_idx], dtype=np.float32) - self.mean_vol_data.numpy()) / self.std_vol_data.numpy()

        output = (
            self._owned_float_tensor(geo),
            self._owned_float_tensor(surf),
            self._owned_float_tensor(surf_target),
            self._owned_float_tensor(vol),
            self._owned_float_tensor(vol_target),
        )
        if not self.return_geometry_density:
            return output

        full_density = self._load_density(run_id, surface_coords)
        density = full_density.index_select(0, torch.from_numpy(np.asarray(geo_idx, dtype=np.int64)))
        return output + (density.contiguous(),)
