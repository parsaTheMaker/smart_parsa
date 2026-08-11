"""Dataset adapter for the streamed SHIFT-Pump preprocessing format."""

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


class PumpDataset(Dataset):
    CACHE_VERSION = "pump_v1"
    SURFACE_FIELDS = ("pressure", "velocity_x", "velocity_y", "velocity_z", "wall_shear_x", "wall_shear_y", "wall_shear_z")
    VOLUME_FIELDS = ("pressure", "velocity_x", "velocity_y", "velocity_z")
    PARAMETER_KEYS = ("flow_rate", "flow_rate_op_condition", "head", "outer_diameter_factor", "outlet_width_factor", "shroud_diameter_factor", "compactness", "leAngleDelta", "leRake", "leShapevar", "leWidthVar", "teAngleDelta", "teRake")

    def __init__(self, saved_folder, if_test=False, geometry_points=131072, surface_points=65536, volume_points=65536, scale_positions=False, coordinate_normalization="global_train_bounds", split_seed=42, test_fraction=0.2, geometry_epoch_seeded_sampling=False, return_geometry_density=False, geometry_density_knn_k=16, geometry_density_neighbor_hops=1, geometry_density_estimator="kde", geometry_density_cache_dtype="float16", **_unused):
        del scale_positions
        if str(coordinate_normalization).lower() not in {"global_train_bounds", "global_bounds"}:
            raise ValueError("Pump supports global_train_bounds coordinate normalization only.")
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
            raise FileNotFoundError(f"Pump data root does not exist: {self.root}")
        self.all_ids = self._discover_ids()
        self.training_ids, self.test_ids = self._load_split_ids(self.all_ids)
        self.data = tuple(self.test_ids if self.if_test else self.training_ids)
        if not self.data:
            raise ValueError("Selected Pump split is empty.")
        self.surface_field_names = list(self.SURFACE_FIELDS)
        self.volume_field_names = list(self.VOLUME_FIELDS)
        self._load_train_statistics()
        print(f"[SHIFT-Pump] split={'test' if self.if_test else 'train'}, cases={len(self.data)}, geometry_points={'full' if self.geometry_points == 0 else self.geometry_points}, surface_queries={self.surface_points}, volume_queries={self.volume_points}, geometry_density={'enabled' if self.return_geometry_density else 'disabled'}")

    def _discover_ids(self):
        ids = []
        for path in self.root.glob("run_*"):
            if path.is_dir() and path.name.split("_", 1)[-1].isdigit() and all((path / name).is_file() for name in ("_COMPLETE.json", "surface_coords.npy", "surface_data.npy", "volume_coords.npy", "volume_data.npy", "case_metadata.json")):
                ids.append(int(path.name.split("_", 1)[1]))
        if not ids:
            raise FileNotFoundError(f"No complete Pump run folders found in {self.root}")
        return sorted(set(ids))

    def _load_split_ids(self, available):
        available = set(int(value) for value in available)
        manifest_path = self.root / "preprocessed_manifest.json"
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            train = [int(value) for value in manifest.get("train_ids", []) if int(value) in available]
            test = [int(value) for value in manifest.get("test_ids", []) if int(value) in available]
            if train and test:
                return sorted(set(train)), sorted(set(test))
        ids = np.asarray(sorted(available), dtype=np.int64)
        rng = np.random.default_rng(self.split_seed)
        rng.shuffle(ids)
        if ids.size == 1:
            return [int(ids[0])], [int(ids[0])]
        count = max(1, min(ids.size - 1, int(round(ids.size * self.test_fraction))))
        return sorted(ids[count:].tolist()), sorted(ids[:count].tolist())

    def _run_dir(self, run_id):
        return self.root / f"run_{int(run_id)}"

    def _stats_paths(self):
        return tuple(self.root / f"{name}_stats_{self.CACHE_VERSION}.npy" for name in ("surface", "volume", "parameter", "position"))

    def _compute_stats(self):
        ss = np.zeros(7, dtype=np.float64); ssq = np.zeros(7, dtype=np.float64)
        vs = np.zeros(4, dtype=np.float64); vsq = np.zeros(4, dtype=np.float64)
        ps = np.zeros(13, dtype=np.float64); psq = np.zeros(13, dtype=np.float64)
        sc = vc = pc = 0
        lo = np.full(3, np.inf); hi = np.full(3, -np.inf)
        for run_id in self.training_ids:
            row = json.loads((self._run_dir(run_id) / "case_metadata.json").read_text(encoding="utf-8"))
            ss += np.asarray(row["surface_sum"], dtype=np.float64); ssq += np.asarray(row["surface_sq_sum"], dtype=np.float64); sc += int(row["surface_count"])
            vs += np.asarray(row["volume_sum"], dtype=np.float64); vsq += np.asarray(row["volume_sq_sum"], dtype=np.float64); vc += int(row["volume_count"])
            ps += np.asarray(row["parameter_sum"], dtype=np.float64); psq += np.asarray(row["parameter_sq_sum"], dtype=np.float64); pc += int(row["parameter_count"])
            lo = np.minimum(lo, np.asarray(row["position_min"], dtype=np.float64)); hi = np.maximum(hi, np.asarray(row["position_max"], dtype=np.float64))

        def stats(total, square, count):
            mean = total / float(count)
            var = np.maximum((square - total * total / float(count)) / max(count - 1, 1), 1.0e-12)
            return np.stack([mean, np.sqrt(var)]).astype(np.float32)
        return stats(ss, ssq, sc), stats(vs, vsq, vc), stats(ps, psq, pc), np.stack([lo, hi]).astype(np.float32)

    @staticmethod
    def _atomic_save(path, array):
        temporary = path.with_suffix(path.suffix + ".partial")
        with temporary.open("wb") as handle:
            np.save(handle, np.asarray(array), allow_pickle=False)
        temporary.replace(path)

    def _load_train_statistics(self):
        paths = self._stats_paths()
        provenance = self.root / f"stats_provenance_{self.CACHE_VERSION}.json"
        try:
            current = json.loads(provenance.read_text(encoding="utf-8")).get("train_ids") == list(self.training_ids)
        except (OSError, ValueError, TypeError):
            current = False
        if not current or not all(path.is_file() for path in paths):
            values = self._compute_stats()
            for path, value in zip(paths, values):
                self._atomic_save(path, value)
            temporary = provenance.with_suffix(".partial")
            temporary.write_text(json.dumps({"train_ids": list(self.training_ids), "version": self.CACHE_VERSION}, indent=2) + "\n", encoding="utf-8")
            temporary.replace(provenance)
        surface, volume, parameter, position = (np.load(path) for path in paths)
        if surface.shape != (2, 7) or volume.shape != (2, 4) or parameter.shape != (2, 13) or position.shape != (2, 3):
            raise ValueError(f"Invalid Pump statistics shapes: surface={surface.shape}, volume={volume.shape}, parameter={parameter.shape}, position={position.shape}")
        self.mean_surf_data = torch.from_numpy(np.asarray(surface[0], dtype=np.float32)); self.std_surf_data = torch.from_numpy(np.maximum(surface[1], 1.0e-12).astype(np.float32))
        self.mean_vol_data = torch.from_numpy(np.asarray(volume[0], dtype=np.float32)); self.std_vol_data = torch.from_numpy(np.maximum(volume[1], 1.0e-12).astype(np.float32))
        self.mean_params = torch.from_numpy(np.asarray(parameter[0], dtype=np.float32)); self.std_params = torch.from_numpy(np.maximum(parameter[1], 1.0e-12).astype(np.float32))
        self.min_pos = torch.from_numpy(np.asarray(position[0], dtype=np.float32)); self.max_pos = torch.from_numpy(np.asarray(position[1], dtype=np.float32)); self.position_span = torch.clamp(self.max_pos - self.min_pos, min=1.0e-12)

    def _get_arrays(self, run_id):
        key = int(run_id)
        arrays = self._memmap_cache.get(key)
        if arrays is None:
            directory = self._run_dir(key)
            arrays = {name: np.load(directory / f"{name}.npy", mmap_mode="r") for name in ("surface_coords", "surface_data", "volume_coords", "volume_data")}
            self._memmap_cache[key] = arrays
        self._memmap_cache.move_to_end(key)
        while len(self._memmap_cache) > self._memmap_cache_max_entries:
            self._memmap_cache.popitem(last=False)
        return arrays

    def get_case_params(self, run_id):
        row = json.loads((self._run_dir(run_id) / "case_metadata.json").read_text(encoding="utf-8"))
        values = (np.asarray(row["parameters"], dtype=np.float32) - self.mean_params.numpy()) / self.std_params.numpy()
        return np.ascontiguousarray(values, dtype=np.float32)

    def _density_path(self, run_id):
        candidates = sorted(self._run_dir(run_id).glob(f"geometry_log_density_{self.CACHE_VERSION}_casebbox_k{self.geometry_density_knn_k}_h{self.geometry_density_neighbor_hops}_*.npy"))
        return candidates[0] if candidates else None

    def _load_density(self, run_id, coords):
        key = int(run_id)
        cached = self._density_cache.get(key)
        if cached is not None and cached.shape[0] == coords.shape[0]:
            return cached
        path = self._density_path(key)
        if path is None:
            points = np.asarray(coords, dtype=np.float32); lo = points.min(0); span = np.maximum(points.max(0) - lo, 1.0e-12); normalized = (points - lo) / span
            density = estimate_log_sampling_density(torch.from_numpy(normalized).unsqueeze(0), knn_k=self.geometry_density_knn_k, neighbor_hops=self.geometry_density_neighbor_hops, estimator=self.geometry_density_estimator).squeeze(0).float()
        else:
            density = torch.from_numpy(np.asarray(np.load(path, mmap_mode="r"), dtype=np.float32))
        if density.shape[0] != coords.shape[0]:
            raise ValueError(f"Pump density/geometry mismatch for run_{key}: {density.shape} vs {coords.shape}")
        self._density_cache[key] = density
        return density

    def set_epoch(self, epoch):
        self._shared_epoch.value = int(epoch)

    def _sample(self, count, target, rng):
        if target <= 0 or target >= count:
            return np.arange(count, dtype=np.int64)
        return rng.choice(count, size=target, replace=False).astype(np.int64, copy=False)

    @staticmethod
    def _owned_float_tensor(values):
        # Full-geometry SATLOSS samples can be read-only np.memmap views.
        # Give DataLoader collation owned storage instead of that view.
        return torch.from_numpy(np.array(values, dtype=np.float32, order="C", copy=True))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        run_id = int(self.data[index]); arrays = self._get_arrays(run_id)
        surface_coords = arrays["surface_coords"]; surface_data = arrays["surface_data"]; volume_coords = arrays["volume_coords"]; volume_data = arrays["volume_data"]
        if surface_data.shape[1] != 7 or volume_data.shape[1] != 4:
            raise ValueError(f"Unexpected Pump channel shapes in run_{run_id}: {surface_data.shape}, {volume_data.shape}")
        geometry_rng = np.random.default_rng(np.random.SeedSequence([self.split_seed, int(self._shared_epoch.value), run_id, 0])) if self.geometry_epoch_seeded_sampling else np.random.default_rng()
        query_rng = np.random.default_rng()
        geo_idx = self._sample(surface_coords.shape[0], self.geometry_points, geometry_rng)
        surf_idx = self._sample(surface_coords.shape[0], self.surface_points, query_rng)
        vol_idx = self._sample(volume_coords.shape[0], self.volume_points, query_rng)
        lo = self.min_pos.numpy(); span = self.position_span.numpy()
        normalize = lambda values: (np.asarray(values, dtype=np.float32) - lo[None, :]) / span[None, :]
        geo = normalize(surface_coords[geo_idx]); surf = normalize(surface_coords[surf_idx]); vol = normalize(volume_coords[vol_idx])
        surf_target = (np.asarray(surface_data[surf_idx], dtype=np.float32) - self.mean_surf_data.numpy()) / self.std_surf_data.numpy()
        vol_target = (np.asarray(volume_data[vol_idx], dtype=np.float32) - self.mean_vol_data.numpy()) / self.std_vol_data.numpy()
        params = self.get_case_params(run_id)
        output = (
            self._owned_float_tensor(geo),
            self._owned_float_tensor(surf),
            self._owned_float_tensor(surf_target),
            self._owned_float_tensor(vol),
            self._owned_float_tensor(vol_target),
            self._owned_float_tensor(params),
        )
        if not self.return_geometry_density:
            return output
        density = self._load_density(run_id, surface_coords).index_select(0, torch.from_numpy(np.asarray(geo_idx, dtype=np.int64)))
        return output + (density.contiguous(),)
