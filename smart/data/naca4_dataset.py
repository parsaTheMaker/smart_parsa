import json
import os
import shutil
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class NACA4Dataset(Dataset):
    """Dataset for NACA4-to-AirfRANS-like data stored as per-case NPZ files.

    Surface targets:
        - pressure
        - normal_x
        - normal_y

    Volume targets:
        - pressure
        - sdf
        - velocity_x
        - velocity_y

    Notes:
        - The raw dataset already separates surface and volume points through the
          boolean `surface` mask.
        - Volume points are kept outside the airfoil by construction; we never
          sample interior points from the aerofoil body.
    """

    CACHE_VERSION = "v2"

    def __init__(
        self,
        saved_folder="../data/",
        if_test=False,
        geometry_points=2048,
        surface_points=2048,
        volume_points=32768,
        copy_to_node=False,
        prepare_data=False,
        fast_approx_sampling=True,
        scale_positions=False,
        manifest_variant="full",
    ):
        print(f"Using {geometry_points} geometry points, {surface_points} surface points, and {volume_points} volume points.")

        self.geometry_points = geometry_points
        self.surface_points = surface_points
        self.volume_points = volume_points
        self.fast_approx_sampling = fast_approx_sampling
        self.file_path = os.path.abspath(saved_folder)
        self.manifest_variant = str(manifest_variant).lower()
        self.if_test = if_test
        self.scale_positions = scale_positions

        # Documented domain bounds for the converted NACA4 data.
        self.min_pos = torch.tensor([-5.0, -5.0], dtype=torch.float32)
        self.max_pos = torch.tensor([5.0, 5.0], dtype=torch.float32)

        self.surface_field_names = ["pressure", "normal_x", "normal_y"]
        self.volume_field_names = ["pressure", "sdf", "velocity_x", "velocity_y"]

        self.manifest = self._load_manifest()
        self.training_ids = self._resolve_split("train")
        self.test_ids = self._resolve_split("test")
        self.all_ids = list(dict.fromkeys(self.training_ids + self.test_ids))

        if if_test:
            self.data = self.test_ids
        else:
            self.data = self.training_ids

        if prepare_data:
            print("Precompute numpy arrays...")
            self.precompute_numpy_arrays()
            print("Computing stats...")
            self.compute_stats()

        if copy_to_node:
            user = os.getenv("USER", "user")
            self.copy_data_to_node(f"/data/scratch/{user}/data/naca4")

        self.load_stats()

    def _load_manifest(self):
        manifest_file = Path(os.path.join(self.file_path, "manifest.json"))
        if not manifest_file.is_file():
            raise FileNotFoundError(f"manifest.json not found in {self.file_path}")

        with open(manifest_file, "r", encoding="utf-8") as f:
            return json.load(f)

    def _flatten_split(self, value):
        if isinstance(value, dict):
            for key in ("cases", "ids", "samples", "items", "entries", "data"):
                if key in value:
                    value = value[key]
                    break
            else:
                value = list(value.keys())

        if isinstance(value, str):
            return [value]

        if isinstance(value, (list, tuple)):
            return [str(item) for item in value]

        raise TypeError(f"Unsupported manifest split type: {type(value)!r}")

    def _resolve_split(self, split_name):
        candidate_keys = [
            f"{self.manifest_variant}_{split_name}",
            f"{split_name}_{self.manifest_variant}",
            split_name,
        ]

        for key in candidate_keys:
            if key in self.manifest:
                return self._filter_existing_cases(self._flatten_split(self.manifest[key]))

        for outer_key in (self.manifest_variant, "splits", "split"):
            container = self.manifest.get(outer_key)
            if isinstance(container, dict):
                for key in candidate_keys + [split_name]:
                    if key in container:
                        return self._filter_existing_cases(self._flatten_split(container[key]))

        raise KeyError(f"Could not resolve split '{split_name}' for variant '{self.manifest_variant}' in manifest.json")

    def _filter_existing_cases(self, cases):
        filtered = []
        missing = []
        for case_id in cases:
            if self._npz_path(case_id).is_file():
                filtered.append(case_id)
            else:
                missing.append(str(case_id))

        if missing:
            print(f"Warning: skipped {len(missing)} missing NACA4 cases from the manifest")

        return filtered

    def _case_dir(self, case_id):
        case_id = str(case_id)
        case_dir = Path(self.file_path) / case_id
        if case_dir.is_dir():
            return case_dir
        return case_dir

    def _npz_path(self, case_id):
        case_dir = self._case_dir(case_id)
        if case_dir.suffix == ".npz":
            return case_dir
        return case_dir / f"{case_dir.name}.npz"

    def _cache_paths(self, case_id):
        case_dir = self._case_dir(case_id)
        return {
            "geometry": case_dir / f"geometry_{self.CACHE_VERSION}.npy",
            "surface": case_dir / f"surface_{self.CACHE_VERSION}.npy",
            "surface_targets": case_dir / f"surface_targets_{self.CACHE_VERSION}.npy",
            "volume": case_dir / f"volume_{self.CACHE_VERSION}.npy",
            "volume_targets": case_dir / f"volume_targets_{self.CACHE_VERSION}.npy",
        }

    def _load_case_arrays(self, case_id, write_cache=True):
        case_dir = self._case_dir(case_id)
        npz_path = self._npz_path(case_id)
        cache_paths = self._cache_paths(case_id)

        cache_ready = all(path.is_file() for path in cache_paths.values())
        if cache_ready:
            geo_mesh = torch.tensor(np.load(cache_paths["geometry"]), dtype=torch.float32)
            surf_mesh = torch.tensor(np.load(cache_paths["surface"]), dtype=torch.float32)
            surf_data = torch.tensor(np.load(cache_paths["surface_targets"]), dtype=torch.float32)
            vol_mesh = torch.tensor(np.load(cache_paths["volume"]), dtype=torch.float32)
            vol_data = torch.tensor(np.load(cache_paths["volume_targets"]), dtype=torch.float32)
            return geo_mesh, surf_mesh, surf_data, vol_mesh, vol_data

        with np.load(npz_path) as raw:
            positions = torch.tensor(raw["position"], dtype=torch.float32)
            velocity = torch.tensor(raw["velocity"], dtype=torch.float32)
            pressure = torch.tensor(raw["pressure"], dtype=torch.float32)
            sdf = torch.tensor(raw["sdf"], dtype=torch.float32)
            normals = torch.tensor(raw["normals"], dtype=torch.float32)
            surface = torch.tensor(raw["surface"], dtype=torch.bool).reshape(-1)

        if positions.ndim != 2 or positions.shape[-1] != 2:
            raise ValueError(f"Expected 2D positions in {npz_path}, got shape {tuple(positions.shape)}")

        if pressure.ndim == 1:
            pressure = pressure[:, None]
        else:
            pressure = pressure[..., :1]

        if sdf.ndim == 1:
            sdf = sdf[:, None]
        else:
            sdf = sdf[..., :1]

        if normals.ndim == 1:
            normals = normals[:, None]
        if normals.shape[-1] < 2:
            raise ValueError(f"Expected at least 2 normal channels in {npz_path}")
        normals = normals[..., :2]

        if velocity.ndim == 1:
            velocity = velocity[:, None]
        if velocity.shape[-1] < 2:
            raise ValueError(f"Expected at least 2 velocity channels in {npz_path}")
        velocity = velocity[..., :2]

        surf_mask = surface.reshape(-1)
        vol_mask = ~surf_mask

        geo_mesh = positions[surf_mask]
        surf_mesh = positions[surf_mask]
        surf_data = torch.cat([pressure[surf_mask], normals[surf_mask]], dim=-1)
        vol_mesh = positions[vol_mask]
        vol_data = torch.cat([pressure[vol_mask], sdf[vol_mask], velocity[vol_mask]], dim=-1)

        if geo_mesh.numel() == 0 or surf_mesh.numel() == 0 or vol_mesh.numel() == 0:
            raise ValueError(f"Empty surface or volume subset in {npz_path}")

        if torch.any(sdf[vol_mask] < -1e-6):
            raise ValueError(f"Found negative signed-distance values in the volume subset for {npz_path}")

        if write_cache:
            case_dir.mkdir(parents=True, exist_ok=True)
            np.save(cache_paths["geometry"], geo_mesh.numpy())
            np.save(cache_paths["surface"], surf_mesh.numpy())
            np.save(cache_paths["surface_targets"], surf_data.numpy())
            np.save(cache_paths["volume"], vol_mesh.numpy())
            np.save(cache_paths["volume_targets"], vol_data.numpy())

        return geo_mesh, surf_mesh, surf_data, vol_mesh, vol_data

    def copy_data_to_node(self, path, force_copy=False):
        """Copy the data to the node where the training is running to make loading faster."""

        if not os.path.exists(path) or force_copy:
            print(f"Creating directory {path}")
            os.makedirs(path, exist_ok=True)

            for entry in os.scandir(self.file_path):
                src = entry.path
                dst = os.path.join(path, entry.name)
                if entry.is_dir():
                    shutil.copytree(src, dst, dirs_exist_ok=True)
                elif not os.path.exists(dst):
                    shutil.copy2(src, dst)
        else:
            print(f"Data already copied to {path}, skipping copy step.")

        self.file_path = path

    def precompute_numpy_arrays(self):
        """Load the data to precompute the numpy arrays for faster loading later."""

        for case_id in self.all_ids:
            print(f"Precompute numpy array for sample {case_id}")
            _ = self._load_case_arrays(case_id, write_cache=True)

    def load_stats(self):
        """Load the precomputed mean and std of the dataset for normalization."""
        vol_stats_file = Path(os.path.join(self.file_path, f"volume_stats_{self.CACHE_VERSION}.npy"))
        surf_stats_file = Path(os.path.join(self.file_path, f"surface_stats_{self.CACHE_VERSION}.npy"))
        pos_stats_file = Path(os.path.join(self.file_path, f"position_stats_{self.CACHE_VERSION}.npy"))

        if not (vol_stats_file.is_file() and surf_stats_file.is_file() and pos_stats_file.is_file()):
            if self.if_test:
                raise FileNotFoundError(
                    "Stats files not found for the NACA4 test split. Run the training split once or execute `python3 smart/prepare.py --config-name=naca4` first."
                )
            print("Stats files not found, computing NACA4 statistics from the training split...")
            self.compute_stats()

        print("Loading stats")
        data = np.load(vol_stats_file)
        self.mean_vol_data = torch.tensor(data[0], dtype=torch.float32)
        self.std_vol_data = torch.tensor(data[1], dtype=torch.float32)

        data = np.load(surf_stats_file)
        self.mean_surf_data = torch.tensor(data[0], dtype=torch.float32)
        self.std_surf_data = torch.tensor(data[1], dtype=torch.float32)

        data = np.load(pos_stats_file)
        self.min_pos = torch.tensor(data[0], dtype=torch.float32)
        self.max_pos = torch.tensor(data[1], dtype=torch.float32)

        print(f"Average surface: {self.mean_surf_data}")
        print(f"Average volume: {self.mean_vol_data}")
        print(f"Std surface: {self.std_surf_data}")
        print(f"Std volume: {self.std_vol_data}")
        print(f"Min position: {self.min_pos}")
        print(f"Max position: {self.max_pos}")

    @staticmethod
    def _safe_std(sum_, squared_sum, count):
        if count <= 1:
            return torch.zeros_like(sum_)
        variance = (squared_sum - ((sum_ ** 2) / count)) / (count - 1)
        return torch.sqrt(torch.clamp(variance, min=1e-12))

    def compute_stats(self):
        """Iteratively compute the mean and std of the dataset for normalization."""

        vol_stats_file = Path(os.path.join(self.file_path, f"volume_stats_{self.CACHE_VERSION}.npy"))
        surf_stats_file = Path(os.path.join(self.file_path, f"surface_stats_{self.CACHE_VERSION}.npy"))
        pos_stats_file = Path(os.path.join(self.file_path, f"position_stats_{self.CACHE_VERSION}.npy"))

        min_pos = torch.full((2,), np.inf, dtype=torch.float32)
        max_pos = torch.full((2,), -np.inf, dtype=torch.float32)

        surf_data_sum = torch.zeros((len(self.surface_field_names),), dtype=torch.float32)
        surf_data_squared_sum = torch.zeros((len(self.surface_field_names),), dtype=torch.float32)
        surf_data_count = 0

        vol_data_sum = torch.zeros((len(self.volume_field_names),), dtype=torch.float32)
        vol_data_squared_sum = torch.zeros((len(self.volume_field_names),), dtype=torch.float32)
        vol_data_count = 0

        for case_id in self.training_ids:
            geo_mesh, surf_mesh, surf_data, vol_mesh, vol_data = self._load_case_arrays(case_id, write_cache=True)

            for dim in range(2):
                max_pos[dim] = max(
                    max_pos[dim],
                    geo_mesh[:, dim].max().item(),
                    surf_mesh[:, dim].max().item(),
                    vol_mesh[:, dim].max().item(),
                )
                min_pos[dim] = min(
                    min_pos[dim],
                    geo_mesh[:, dim].min().item(),
                    surf_mesh[:, dim].min().item(),
                    vol_mesh[:, dim].min().item(),
                )

            surf_data_sum += surf_data.sum(dim=0)
            surf_data_squared_sum += (surf_data ** 2).sum(dim=0)
            surf_data_count += surf_data.shape[0]

            vol_data_sum += vol_data.sum(dim=0)
            vol_data_squared_sum += (vol_data ** 2).sum(dim=0)
            vol_data_count += vol_data.shape[0]

        self.mean_surf_data = surf_data_sum / max(surf_data_count, 1)
        self.std_surf_data = self._safe_std(surf_data_sum, surf_data_squared_sum, surf_data_count)

        self.mean_vol_data = vol_data_sum / max(vol_data_count, 1)
        self.std_vol_data = self._safe_std(vol_data_sum, vol_data_squared_sum, vol_data_count)

        self.min_pos = min_pos
        self.max_pos = max_pos

        np.save(surf_stats_file, np.stack([self.mean_surf_data.cpu().numpy(), self.std_surf_data.cpu().numpy()]))
        np.save(vol_stats_file, np.stack([self.mean_vol_data.cpu().numpy(), self.std_vol_data.cpu().numpy()]))
        np.save(pos_stats_file, np.stack([self.min_pos.cpu().numpy(), self.max_pos.cpu().numpy()]))

        print(f"Average surface: {self.mean_surf_data}")
        print(f"Average volume: {self.mean_vol_data}")
        print(f"Std surface: {self.std_surf_data}")
        print(f"Std volume: {self.std_vol_data}")
        print(f"Min position: {self.min_pos}")
        print(f"Max position: {self.max_pos}")

    def get_surface_mesh(self, case_id):
        geo_mesh, _, _, _, _ = self._load_case_arrays(case_id, write_cache=True)
        return geo_mesh

    def get_surface_data(self, case_id):
        _, surf_mesh, surf_data, _, _ = self._load_case_arrays(case_id, write_cache=True)
        return surf_mesh, surf_data

    def get_volume_data(self, case_id):
        _, _, _, vol_mesh, vol_data = self._load_case_arrays(case_id, write_cache=True)
        return vol_mesh, vol_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """Retrieves a sample for a given index with geometry, surface mesh and data, and volume mesh and data."""
        case_id = self.data[idx]
        geo_mesh, surf_mesh, surf_data, vol_mesh, vol_data = self._load_case_arrays(case_id, write_cache=True)

        if self.geometry_points > 0:
            if not self.fast_approx_sampling and self.geometry_points <= geo_mesh.shape[0]:
                geo_points = torch.randperm(geo_mesh.shape[0])[:self.geometry_points]
            else:
                geo_points = torch.randint(0, geo_mesh.shape[0], (self.geometry_points,))
        else:
            geo_points = torch.arange(geo_mesh.shape[0])
        geo_mesh = (geo_mesh[geo_points, :] - self.min_pos) / (self.max_pos - self.min_pos)

        if self.surface_points > 0:
            if not self.fast_approx_sampling and self.surface_points <= surf_mesh.shape[0]:
                surface_points = torch.randperm(surf_mesh.shape[0])[:self.surface_points]
            else:
                surface_points = torch.randint(0, surf_mesh.shape[0], (self.surface_points,))
        else:
            surface_points = torch.arange(surf_mesh.shape[0])
        surf_mesh = (surf_mesh[surface_points, :] - self.min_pos) / (self.max_pos - self.min_pos)
        surf_data = (surf_data[surface_points, :] - self.mean_surf_data) / self.std_surf_data

        if self.volume_points > 0:
            if not self.fast_approx_sampling and self.volume_points <= vol_mesh.shape[0]:
                vol_points = torch.randperm(vol_mesh.shape[0])[:self.volume_points]
            else:
                vol_points = torch.randint(0, vol_mesh.shape[0], (self.volume_points,))
        else:
            vol_points = torch.arange(vol_mesh.shape[0])
        vol_mesh = (vol_mesh[vol_points, :] - self.min_pos) / (self.max_pos - self.min_pos)
        vol_data = (vol_data[vol_points, :] - self.mean_vol_data) / self.std_vol_data

        return geo_mesh, surf_mesh, surf_data, vol_mesh, vol_data
