import os
import json
import shutil
import tempfile
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class AhmedMLDatasetV2(Dataset):
    """DrivAerML/AhmedML-v2 dataset reader for run_*/boundary_*.h5 + volume_*_filtered.h5."""

    CACHE_VERSION = "v2_h5"

    def __init__(
        self,
        saved_folder="../data/",
        if_test=False,
        geometry_points=65536,
        surface_points=65536,
        volume_points=65536,
        copy_to_node=False,
        prepare_data=False,
        fast_approx_sampling=True,
        scale_positions=False,
        split_seed=42,
        test_fraction=0.2,
        stats_stride=20,
        stats_max_runs=32,
        io_oversample_factor=4,
        cache_root=None,
        require_preprocessed=False,
    ):
        print(f"Using {geometry_points} geometry points, {surface_points} surface points, and {volume_points} volume points.")

        self.geometry_points = int(geometry_points)
        self.surface_points = int(surface_points)
        self.volume_points = int(volume_points)
        self.fast_approx_sampling = bool(fast_approx_sampling)
        self.if_test = bool(if_test)
        self.scale_positions = bool(scale_positions)
        self.stats_stride = max(1, int(stats_stride))
        self.stats_max_runs = max(1, int(stats_max_runs))
        self.io_oversample_factor = max(1, int(io_oversample_factor))

        self.file_path = os.path.abspath(saved_folder)
        self.cache_root = self._resolve_cache_root(cache_root)
        print(f"AhmedMLDatasetV2 cache root: {self.cache_root}")
        self.preprocessed_mode = (Path(self.file_path) / "preprocessed_manifest.json").is_file()
        self.require_preprocessed = bool(require_preprocessed)
        if self.require_preprocessed and not self.preprocessed_mode:
            raise FileNotFoundError(
                "DrivAerML is configured to use preprocessed-only mode, but "
                f"`preprocessed_manifest.json` was not found in {self.file_path}."
            )
        if self.preprocessed_mode:
            print("AhmedMLDatasetV2: detected preprocessed DrivAerML layout.")

        # Conservative defaults; overwritten by stats when available.
        if scale_positions:
            self.min_pos = torch.tensor([-4.0, -4.0, -4.0], dtype=torch.float32)
            self.max_pos = torch.tensor([6.0, 6.0, 6.0], dtype=torch.float32)
        else:
            self.min_pos = torch.tensor([-4.0, -2.0, -2.0], dtype=torch.float32)
            self.max_pos = torch.tensor([6.0, 2.0, 2.0], dtype=torch.float32)

        self.surface_field_names = ["pressure", "normal_x", "normal_y", "normal_z", "wall_shear_x", "wall_shear_y", "wall_shear_z"]
        self.volume_field_names = ["pressure", "velocity_x", "velocity_y", "velocity_z"]

        self.all_ids = self._discover_ids()
        self._point_counts = self._load_point_counts_cache()
        self.training_ids, self.test_ids = self._resolve_split_ids(split_seed, test_fraction)
        self.data = self.test_ids if self.if_test else self.training_ids

        if prepare_data:
            print("Precompute numpy arrays...")
            self.precompute_numpy_arrays()
            print("Computing stats...")
            self.compute_stats()

        if copy_to_node:
            user = os.getenv("USER", "user")
            self.copy_data_to_node(f"/data/scratch/{user}/data/ahmedml_v2")

        self.load_stats()

    def _resolve_cache_root(self, cache_root):
        # Prefer user-provided cache root; otherwise use dataset dir only if writable.
        if cache_root:
            root = Path(cache_root).expanduser().resolve()
            root.mkdir(parents=True, exist_ok=True)
            return root

        data_root = Path(self.file_path)
        if os.access(data_root, os.W_OK):
            return data_root

        fallback = Path.home() / "smart_parsa" / ".cache" / "ahmedml_v2"
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback

    def _discover_ids(self):
        if self.preprocessed_mode:
            ids = []
            for entry in os.scandir(self.file_path):
                if not entry.is_dir() or not entry.name.startswith("run_"):
                    continue
                try:
                    rid = int(entry.name.split("_")[1])
                except (IndexError, ValueError):
                    continue
                run_dir = Path(entry.path)
                required = [
                    run_dir / "surface_coords.npy",
                    run_dir / "surface_pMeanTrim.npy",
                    run_dir / "volume_coords.npy",
                    run_dir / "volume_UMeanTrim.npy",
                ]
                if all(p.is_file() for p in required):
                    ids.append(rid)
            ids = sorted(set(ids))
            if not ids:
                raise FileNotFoundError(f"No preprocessed run_* folders found in {self.file_path}")
            print(f"Found {len(ids)} valid preprocessed run folders.")
            return ids

        ids = []
        for entry in os.scandir(self.file_path):
            if not entry.is_dir() or not entry.name.startswith("run_"):
                continue
            try:
                rid = int(entry.name.split("_")[1])
            except (IndexError, ValueError):
                continue
            if self._boundary_h5_path(rid).is_file() and self._volume_h5_path(rid).is_file():
                ids.append(rid)
        ids = sorted(set(ids))
        if not ids:
            raise FileNotFoundError(f"No valid run_* folders with boundary_*.h5 and volume_*_filtered.h5 found in {self.file_path}")
        print(f"Found {len(ids)} valid run folders.")
        return ids

    def _point_counts_cache_path(self):
        return self.cache_root / f"point_counts_{self.CACHE_VERSION}.json"

    def _load_point_counts_cache(self):
        path = self._point_counts_cache_path()
        if not path.is_file():
            return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            out = {}
            for k, v in raw.items():
                rid = int(k)
                if isinstance(v, dict) and "surface" in v and "volume" in v:
                    out[rid] = {"surface": int(v["surface"]), "volume": int(v["volume"])}
            return out
        except Exception:
            return {}

    def _save_point_counts_cache(self):
        path = self._point_counts_cache_path()
        tmp = path.with_suffix(path.suffix + ".tmp")
        serializable = {str(k): v for k, v in self._point_counts.items()}
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2)
        os.replace(tmp, path)

    def _get_point_counts(self, run_id):
        if self.preprocessed_mode:
            run_dir = self._run_dir(run_id)
            ns = int(np.load(run_dir / "surface_coords.npy", mmap_mode="r").shape[0])
            nv = int(np.load(run_dir / "volume_coords.npy", mmap_mode="r").shape[0])
            return ns, nv
        if run_id in self._point_counts:
            return self._point_counts[run_id]["surface"], self._point_counts[run_id]["volume"]
        with h5py.File(self._boundary_h5_path(run_id), "r") as hb:
            ns = int(hb["coords"].shape[0])
        with h5py.File(self._volume_h5_path(run_id), "r") as hv:
            nv = int(hv["coords"].shape[0])
        self._point_counts[run_id] = {"surface": ns, "volume": nv}
        # Persist incrementally so train/test dataset instances share this quickly.
        try:
            self._save_point_counts_cache()
        except Exception:
            pass
        return ns, nv

    @staticmethod
    def _split_ids(ids, seed=42, test_fraction=0.2):
        ids = list(ids)
        rng = np.random.default_rng(int(seed))
        perm = rng.permutation(len(ids))
        n_test = max(1, int(round(len(ids) * float(test_fraction))))
        test_idx = set(perm[:n_test].tolist())
        test_ids = [ids[i] for i in range(len(ids)) if i in test_idx]
        train_ids = [ids[i] for i in range(len(ids)) if i not in test_idx]
        return train_ids, test_ids

    def _resolve_split_ids(self, split_seed, test_fraction):
        if self.preprocessed_mode:
            manifest_file = Path(self.file_path) / "preprocessed_manifest.json"
            try:
                with open(manifest_file, "r", encoding="utf-8") as f:
                    m = json.load(f)
                train_ids = [int(x) for x in m.get("train_ids", [])]
                test_ids = [int(x) for x in m.get("test_ids", [])]
                if train_ids and test_ids:
                    have = set(self.all_ids)
                    train_ids = [x for x in train_ids if x in have]
                    test_ids = [x for x in test_ids if x in have]
                    if train_ids and test_ids:
                        return train_ids, test_ids
            except Exception:
                pass
        return self._split_ids(self.all_ids, split_seed, test_fraction)

    def _run_dir(self, run_id):
        return Path(self.file_path) / f"run_{run_id}"

    def _boundary_h5_path(self, run_id):
        return self._run_dir(run_id) / f"boundary_{run_id}.h5"

    def _volume_h5_path(self, run_id):
        return self._run_dir(run_id) / f"volume_{run_id}_filtered.h5"

    def _cache_paths(self, run_id):
        run_dir = self.cache_root / f"run_{run_id}"
        run_dir.mkdir(parents=True, exist_ok=True)
        return {
            "geometry": run_dir / f"geometry_{self.CACHE_VERSION}.npy",
            "surface": run_dir / f"surface_{self.CACHE_VERSION}.npy",
            "surface_targets": run_dir / f"surface_targets_{self.CACHE_VERSION}.npy",
            "volume": run_dir / f"volume_{self.CACHE_VERSION}.npy",
            "volume_targets": run_dir / f"volume_targets_{self.CACHE_VERSION}.npy",
        }

    @staticmethod
    def _finite_mask(*arrays):
        mask = None
        for arr in arrays:
            cur = torch.isfinite(arr).all(dim=-1) if arr.ndim > 1 else torch.isfinite(arr)
            mask = cur if mask is None else (mask & cur)
        return mask

    @staticmethod
    def _atomic_save_npy(path: Path, array: np.ndarray):
        """Write npy atomically to avoid half-written files seen by other workers."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + ".tmp.", suffix=".npy", delete=False) as tf:
            tmp_name = tf.name
        try:
            np.save(tmp_name, array)
            os.replace(tmp_name, path)
        finally:
            if os.path.exists(tmp_name):
                try:
                    os.remove(tmp_name)
                except OSError:
                    pass

    def _load_case_arrays(self, run_id, write_cache=True):
        cache = self._cache_paths(run_id)
        if all(p.is_file() for p in cache.values()):
            try:
                geo_mesh = torch.tensor(np.load(cache["geometry"]), dtype=torch.float32)
                surf_mesh = torch.tensor(np.load(cache["surface"]), dtype=torch.float32)
                surf_data = torch.tensor(np.load(cache["surface_targets"]), dtype=torch.float32)
                vol_mesh = torch.tensor(np.load(cache["volume"]), dtype=torch.float32)
                vol_data = torch.tensor(np.load(cache["volume_targets"]), dtype=torch.float32)
                return geo_mesh, surf_mesh, surf_data, vol_mesh, vol_data
            except Exception as exc:
                # Corrupted/truncated cache from interrupted or concurrent writes. Rebuild from H5.
                print(f"Warning: cache for run_{run_id} is invalid ({exc}). Rebuilding from H5.")
                for p in cache.values():
                    try:
                        if p.exists():
                            p.unlink()
                    except OSError:
                        pass

        with h5py.File(self._boundary_h5_path(run_id), "r") as hb:
            bcoords = np.asarray(hb["coords"], dtype=np.float32)
            p_surf = np.asarray(hb["pMeanTrim"], dtype=np.float32).reshape(-1, 1)
            normals = np.asarray(hb["normals"], dtype=np.float32)
            if normals.ndim == 1:
                normals = normals.reshape(-1, 1)
            normals = normals[:, :3]
            wsx = np.asarray(hb["wallShearStressMeanTrim_x"], dtype=np.float32).reshape(-1, 1)
            wsy = np.asarray(hb["wallShearStressMeanTrim_y"], dtype=np.float32).reshape(-1, 1)
            wsz = np.asarray(hb["wallShearStressMeanTrim_z"], dtype=np.float32).reshape(-1, 1)

        with h5py.File(self._volume_h5_path(run_id), "r") as hv:
            vcoords = np.asarray(hv["coords"], dtype=np.float32)
            p_vol = np.asarray(hv["pMeanTrim"], dtype=np.float32).reshape(-1, 1)
            u = np.asarray(hv["UMeanTrim"], dtype=np.float32)
            if u.ndim == 1:
                u = u.reshape(-1, 1)
            u = u[:, :3]

        geo_mesh = torch.tensor(bcoords, dtype=torch.float32)
        surf_mesh = torch.tensor(bcoords, dtype=torch.float32)
        surf_data = torch.tensor(np.concatenate([p_surf, normals, wsx, wsy, wsz], axis=1), dtype=torch.float32)
        vol_mesh = torch.tensor(vcoords, dtype=torch.float32)
        vol_data = torch.tensor(np.concatenate([p_vol, u], axis=1), dtype=torch.float32)

        surf_mask = self._finite_mask(geo_mesh, surf_mesh, surf_data)
        vol_mask = self._finite_mask(vol_mesh, vol_data)
        geo_mesh, surf_mesh, surf_data = geo_mesh[surf_mask], surf_mesh[surf_mask], surf_data[surf_mask]
        vol_mesh, vol_data = vol_mesh[vol_mask], vol_data[vol_mask]
        if geo_mesh.shape[0] == 0 or surf_mesh.shape[0] == 0 or vol_mesh.shape[0] == 0:
            raise ValueError(f"Run {run_id} has empty arrays after finite filtering.")

        if write_cache:
            self._atomic_save_npy(cache["geometry"], geo_mesh.numpy())
            self._atomic_save_npy(cache["surface"], surf_mesh.numpy())
            self._atomic_save_npy(cache["surface_targets"], surf_data.numpy())
            self._atomic_save_npy(cache["volume"], vol_mesh.numpy())
            self._atomic_save_npy(cache["volume_targets"], vol_data.numpy())

        return geo_mesh, surf_mesh, surf_data, vol_mesh, vol_data

    def copy_data_to_node(self, path, force_copy=False):
        if not os.path.exists(path) or force_copy:
            print(f"Creating directory {path}")
            os.makedirs(path, exist_ok=True)
            for run_id in self.all_ids:
                src_dir = self._run_dir(run_id)
                dst_dir = Path(path) / f"run_{run_id}"
                dst_dir.mkdir(parents=True, exist_ok=True)
                for file in src_dir.glob("*.npy"):
                    dst = dst_dir / file.name
                    if not dst.exists():
                        shutil.copy2(file, dst)
            for name in [
                f"volume_stats_{self.CACHE_VERSION}.npy",
                f"surface_stats_{self.CACHE_VERSION}.npy",
                f"position_stats_{self.CACHE_VERSION}.npy",
                f"split_{self.CACHE_VERSION}.json",
            ]:
                src = self.cache_root / name
                dst = Path(path) / name
                if src.is_file() and not dst.exists():
                    shutil.copy2(src, dst)
        else:
            print(f"Data already copied to {path}, skipping copy step.")
        self.file_path = path

    def precompute_numpy_arrays(self):
        for run_id in self.all_ids:
            print(f"Precompute numpy arrays for run_{run_id}")
            self._load_case_arrays(run_id, write_cache=True)

    def _stats_paths(self):
        return (
            self.cache_root / f"volume_stats_{self.CACHE_VERSION}.npy",
            self.cache_root / f"surface_stats_{self.CACHE_VERSION}.npy",
            self.cache_root / f"position_stats_{self.CACHE_VERSION}.npy",
        )

    def load_stats(self):
        vol_file, surf_file, pos_file = self._stats_paths()
        if not (vol_file.is_file() and surf_file.is_file() and pos_file.is_file()):
            if self.if_test:
                raise FileNotFoundError(
                    f"Stats files missing ({self.CACHE_VERSION}). Run training split once or `python3 smart/prepare.py --config-name=drivaerml`."
                )
            print(
                "Stats files not found. Computing FAST sampled statistics from training split "
                f"(max_runs={self.stats_max_runs}, stride={self.stats_stride})..."
            )
            if self.preprocessed_mode:
                self.compute_stats_from_preprocessed()
            else:
                self.compute_stats_fast()

        print("Loading stats")
        surf = np.load(surf_file)
        vol = np.load(vol_file)
        pos = np.load(pos_file)
        if surf.shape[-1] != len(self.surface_field_names) or vol.shape[-1] != len(self.volume_field_names):
            if self.if_test:
                raise ValueError(
                    f"Stats channel mismatch. Found surface={surf.shape[-1]}, volume={vol.shape[-1]} "
                    f"but expected surface={len(self.surface_field_names)}, volume={len(self.volume_field_names)}. "
                    "Recompute stats with the training split."
                )
            print("Stats shape mismatch for current target channels, recomputing fast stats...")
            if self.preprocessed_mode:
                self.compute_stats_from_preprocessed()
            else:
                self.compute_stats_fast()
            surf = np.load(surf_file)
            vol = np.load(vol_file)
            pos = np.load(pos_file)
        self.mean_surf_data = torch.tensor(surf[0], dtype=torch.float32)
        self.std_surf_data = torch.tensor(surf[1], dtype=torch.float32)
        self.mean_vol_data = torch.tensor(vol[0], dtype=torch.float32)
        self.std_vol_data = torch.tensor(vol[1], dtype=torch.float32)
        self.min_pos = torch.tensor(pos[0], dtype=torch.float32)
        self.max_pos = torch.tensor(pos[1], dtype=torch.float32)

    @staticmethod
    def _safe_std(sum_, sq_sum, count):
        if count <= 1:
            return torch.ones_like(sum_)
        var = (sq_sum - (sum_ ** 2) / count) / (count - 1)
        return torch.sqrt(torch.clamp(var, min=1e-12))

    def compute_stats(self):
        min_pos = torch.full((3,), np.inf, dtype=torch.float32)
        max_pos = torch.full((3,), -np.inf, dtype=torch.float32)

        surf_c = len(self.surface_field_names)
        vol_c = len(self.volume_field_names)

        surf_sum = torch.zeros((surf_c,), dtype=torch.float32)
        surf_sq_sum = torch.zeros((surf_c,), dtype=torch.float32)
        surf_count = 0

        vol_sum = torch.zeros((vol_c,), dtype=torch.float32)
        vol_sq_sum = torch.zeros((vol_c,), dtype=torch.float32)
        vol_count = 0

        stride = self.stats_stride
        for run_id in self.training_ids:
            geo_mesh, surf_mesh, surf_data, vol_mesh, vol_data = self._load_case_arrays(run_id, write_cache=True)

            geo_s = geo_mesh[::stride]
            surf_s, surf_d = surf_mesh[::stride], surf_data[::stride]
            vol_s, vol_d = vol_mesh[::stride], vol_data[::stride]

            for d in range(3):
                max_pos[d] = max(max_pos[d], geo_s[:, d].max().item(), surf_s[:, d].max().item(), vol_s[:, d].max().item())
                min_pos[d] = min(min_pos[d], geo_s[:, d].min().item(), surf_s[:, d].min().item(), vol_s[:, d].min().item())

            surf_sum += surf_d.sum(dim=0)
            surf_sq_sum += (surf_d ** 2).sum(dim=0)
            surf_count += int(surf_d.shape[0])

            vol_sum += vol_d.sum(dim=0)
            vol_sq_sum += (vol_d ** 2).sum(dim=0)
            vol_count += int(vol_d.shape[0])

        self.mean_surf_data = surf_sum / max(surf_count, 1)
        self.std_surf_data = self._safe_std(surf_sum, surf_sq_sum, surf_count)
        self.mean_vol_data = vol_sum / max(vol_count, 1)
        self.std_vol_data = self._safe_std(vol_sum, vol_sq_sum, vol_count)
        self.min_pos = min_pos
        self.max_pos = max_pos

        vol_file, surf_file, pos_file = self._stats_paths()
        np.save(surf_file, np.stack([self.mean_surf_data.numpy(), self.std_surf_data.numpy()]))
        np.save(vol_file, np.stack([self.mean_vol_data.numpy(), self.std_vol_data.numpy()]))
        np.save(pos_file, np.stack([self.min_pos.numpy(), self.max_pos.numpy()]))

        split_file = self.cache_root / f"split_{self.CACHE_VERSION}.json"
        with open(split_file, "w", encoding="utf-8") as f:
            json.dump({"train_ids": self.training_ids, "test_ids": self.test_ids}, f, indent=2)

    def _load_h5_sample_for_stats(self, run_id, stride):
        with h5py.File(self._boundary_h5_path(run_id), "r") as hb:
            bcoords = np.asarray(hb["coords"][::stride], dtype=np.float32)
            p_surf = np.asarray(hb["pMeanTrim"][::stride], dtype=np.float32).reshape(-1, 1)
            normals = np.asarray(hb["normals"][::stride], dtype=np.float32)
            if normals.ndim == 1:
                normals = normals.reshape(-1, 1)
            normals = normals[:, :3]
            wsx = np.asarray(hb["wallShearStressMeanTrim_x"][::stride], dtype=np.float32).reshape(-1, 1)
            wsy = np.asarray(hb["wallShearStressMeanTrim_y"][::stride], dtype=np.float32).reshape(-1, 1)
            wsz = np.asarray(hb["wallShearStressMeanTrim_z"][::stride], dtype=np.float32).reshape(-1, 1)
        with h5py.File(self._volume_h5_path(run_id), "r") as hv:
            vcoords = np.asarray(hv["coords"][::stride], dtype=np.float32)
            p_vol = np.asarray(hv["pMeanTrim"][::stride], dtype=np.float32).reshape(-1, 1)
            u = np.asarray(hv["UMeanTrim"][::stride], dtype=np.float32)
            if u.ndim == 1:
                u = u.reshape(-1, 1)
            u = u[:, :3]

        geo_mesh = torch.tensor(bcoords, dtype=torch.float32)
        surf_mesh = torch.tensor(bcoords, dtype=torch.float32)
        surf_data = torch.tensor(np.concatenate([p_surf, normals, wsx, wsy, wsz], axis=1), dtype=torch.float32)
        vol_mesh = torch.tensor(vcoords, dtype=torch.float32)
        vol_data = torch.tensor(np.concatenate([p_vol, u], axis=1), dtype=torch.float32)

        surf_mask = self._finite_mask(geo_mesh, surf_mesh, surf_data)
        vol_mask = self._finite_mask(vol_mesh, vol_data)
        return geo_mesh[surf_mask], surf_mesh[surf_mask], surf_data[surf_mask], vol_mesh[vol_mask], vol_data[vol_mask]

    def compute_stats_fast(self):
        """Fast startup stats: sample directly from H5 with coarse stride and limited runs."""
        min_pos = torch.full((3,), np.inf, dtype=torch.float32)
        max_pos = torch.full((3,), -np.inf, dtype=torch.float32)

        surf_c = len(self.surface_field_names)
        vol_c = len(self.volume_field_names)

        surf_sum = torch.zeros((surf_c,), dtype=torch.float32)
        surf_sq_sum = torch.zeros((surf_c,), dtype=torch.float32)
        surf_count = 0

        vol_sum = torch.zeros((vol_c,), dtype=torch.float32)
        vol_sq_sum = torch.zeros((vol_c,), dtype=torch.float32)
        vol_count = 0

        run_ids = self.training_ids[: self.stats_max_runs]
        stride = self.stats_stride
        print(f"Fast stats over {len(run_ids)} runs...")

        for run_id in run_ids:
            geo_mesh, surf_mesh, surf_data, vol_mesh, vol_data = self._load_h5_sample_for_stats(run_id, stride)
            if geo_mesh.shape[0] == 0 or surf_mesh.shape[0] == 0 or vol_mesh.shape[0] == 0:
                continue

            for d in range(3):
                max_pos[d] = max(max_pos[d], geo_mesh[:, d].max().item(), surf_mesh[:, d].max().item(), vol_mesh[:, d].max().item())
                min_pos[d] = min(min_pos[d], geo_mesh[:, d].min().item(), surf_mesh[:, d].min().item(), vol_mesh[:, d].min().item())

            surf_sum += surf_data.sum(dim=0)
            surf_sq_sum += (surf_data ** 2).sum(dim=0)
            surf_count += int(surf_data.shape[0])

            vol_sum += vol_data.sum(dim=0)
            vol_sq_sum += (vol_data ** 2).sum(dim=0)
            vol_count += int(vol_data.shape[0])

        self.mean_surf_data = surf_sum / max(surf_count, 1)
        self.std_surf_data = self._safe_std(surf_sum, surf_sq_sum, surf_count)
        self.mean_vol_data = vol_sum / max(vol_count, 1)
        self.std_vol_data = self._safe_std(vol_sum, vol_sq_sum, vol_count)
        self.min_pos = min_pos
        self.max_pos = max_pos

        vol_file, surf_file, pos_file = self._stats_paths()
        np.save(surf_file, np.stack([self.mean_surf_data.numpy(), self.std_surf_data.numpy()]))
        np.save(vol_file, np.stack([self.mean_vol_data.numpy(), self.std_vol_data.numpy()]))
        np.save(pos_file, np.stack([self.min_pos.numpy(), self.max_pos.numpy()]))

        split_file = self.cache_root / f"split_{self.CACHE_VERSION}.json"
        with open(split_file, "w", encoding="utf-8") as f:
            json.dump({"train_ids": self.training_ids, "test_ids": self.test_ids}, f, indent=2)

    def compute_stats_from_preprocessed(self):
        """Compute stats directly from preprocessed NPY files (no H5 access)."""
        surf_sum = np.zeros((len(self.surface_field_names),), dtype=np.float64)
        surf_sq = np.zeros((len(self.surface_field_names),), dtype=np.float64)
        surf_n = 0
        vol_sum = np.zeros((len(self.volume_field_names),), dtype=np.float64)
        vol_sq = np.zeros((len(self.volume_field_names),), dtype=np.float64)
        vol_n = 0
        min_pos = np.array([np.inf, np.inf, np.inf], dtype=np.float64)
        max_pos = np.array([-np.inf, -np.inf, -np.inf], dtype=np.float64)

        print(f"Computing stats from preprocessed files over {len(self.training_ids)} runs...")
        for rid in self.training_ids:
            run_dir = self._run_dir(rid)
            sc = np.load(run_dir / "surface_coords.npy", mmap_mode="r")
            sp = np.load(run_dir / "surface_pMeanTrim.npy", mmap_mode="r")
            sn = np.load(run_dir / "surface_normals.npy", mmap_mode="r")
            swx = np.load(run_dir / "surface_wallShearStressMeanTrim_x.npy", mmap_mode="r")
            swy = np.load(run_dir / "surface_wallShearStressMeanTrim_y.npy", mmap_mode="r")
            swz = np.load(run_dir / "surface_wallShearStressMeanTrim_z.npy", mmap_mode="r")
            vc = np.load(run_dir / "volume_coords.npy", mmap_mode="r")
            vp = np.load(run_dir / "volume_pMeanTrim.npy", mmap_mode="r")
            vu = np.load(run_dir / "volume_UMeanTrim.npy", mmap_mode="r")

            surf = np.concatenate(
                [
                    np.asarray(sp, dtype=np.float32).reshape(-1, 1),
                    np.asarray(sn, dtype=np.float32),
                    np.asarray(swx, dtype=np.float32).reshape(-1, 1),
                    np.asarray(swy, dtype=np.float32).reshape(-1, 1),
                    np.asarray(swz, dtype=np.float32).reshape(-1, 1),
                ],
                axis=1,
            ).astype(np.float64, copy=False)
            vol = np.concatenate(
                [
                    np.asarray(vp, dtype=np.float32).reshape(-1, 1),
                    np.asarray(vu, dtype=np.float32),
                ],
                axis=1,
            ).astype(np.float64, copy=False)

            surf_sum += surf.sum(axis=0)
            surf_sq += (surf ** 2).sum(axis=0)
            surf_n += int(surf.shape[0])

            vol_sum += vol.sum(axis=0)
            vol_sq += (vol ** 2).sum(axis=0)
            vol_n += int(vol.shape[0])

            min_pos = np.minimum(min_pos, np.minimum(sc.min(axis=0), vc.min(axis=0)))
            max_pos = np.maximum(max_pos, np.maximum(sc.max(axis=0), vc.max(axis=0)))

        self.mean_surf_data = torch.tensor(surf_sum / max(surf_n, 1), dtype=torch.float32)
        self.std_surf_data = torch.tensor(
            np.sqrt(np.clip((surf_sq - (surf_sum ** 2) / max(surf_n, 1)) / max(surf_n - 1, 1), 1e-12, None)),
            dtype=torch.float32,
        )
        self.mean_vol_data = torch.tensor(vol_sum / max(vol_n, 1), dtype=torch.float32)
        self.std_vol_data = torch.tensor(
            np.sqrt(np.clip((vol_sq - (vol_sum ** 2) / max(vol_n, 1)) / max(vol_n - 1, 1), 1e-12, None)),
            dtype=torch.float32,
        )
        self.min_pos = torch.tensor(min_pos, dtype=torch.float32)
        self.max_pos = torch.tensor(max_pos, dtype=torch.float32)

        vol_file, surf_file, pos_file = self._stats_paths()
        np.save(surf_file, np.stack([self.mean_surf_data.numpy(), self.std_surf_data.numpy()]))
        np.save(vol_file, np.stack([self.mean_vol_data.numpy(), self.std_vol_data.numpy()]))
        np.save(pos_file, np.stack([self.min_pos.numpy(), self.max_pos.numpy()]))

    def _sample_idx(self, n, k):
        if k <= 0 or k >= n:
            return torch.arange(n, dtype=torch.long)
        if self.fast_approx_sampling:
            return torch.randint(0, n, (k,), dtype=torch.long)
        # Avoid torch.randperm(n) for huge n.
        idx = np.random.choice(n, size=k, replace=False)
        return torch.from_numpy(idx.astype(np.int64))

    @staticmethod
    def _normalize_pos(pos, min_pos, max_pos):
        denom = torch.clamp(max_pos - min_pos, min=1e-12)
        return (pos - min_pos) / denom

    def __len__(self):
        return len(self.data)

    @staticmethod
    def _read_rows_h5(ds, idx_np):
        # h5py fancy indexing is fastest/most reliable with sorted indices.
        if idx_np.size == 0:
            return np.empty((0,) + ds.shape[1:], dtype=ds.dtype)
        # h5py requires strictly increasing indices (no duplicates).
        # We gather unique sorted rows, then expand back to original order.
        unique_sorted, inverse = np.unique(idx_np, return_inverse=True)
        arr_unique = ds[unique_sorted]
        return arr_unique[inverse]

    def _read_strided_pool(self, ds, n_total, k_target):
        """Read a near-sequential pool from H5 and sample locally.

        This avoids heavy random disk seeks on huge contiguous H5 datasets.
        """
        if k_target <= 0 or k_target >= n_total:
            return np.asarray(ds[:], dtype=np.float32)

        # Read a moderately larger pool, then subsample in memory.
        pool_target = min(n_total, max(k_target, k_target * self.io_oversample_factor))
        stride = max(1, n_total // pool_target)
        offset = int(np.random.randint(0, stride)) if stride > 1 else 0
        pool = ds[offset::stride]
        return np.asarray(pool, dtype=np.float32)

    def _strided_slice_params(self, n_total, k_target):
        if k_target <= 0 or k_target >= n_total:
            return 1, 0
        pool_target = min(n_total, max(k_target, k_target * self.io_oversample_factor))
        stride = max(1, n_total // pool_target)
        offset = int(np.random.randint(0, stride)) if stride > 1 else 0
        return stride, offset

    @staticmethod
    def _sample_local(arr, k):
        n = arr.shape[0]
        if k <= 0 or k >= n:
            return arr
        idx = np.random.choice(n, size=k, replace=False)
        return arr[idx]

    def _load_case_sampled_from_h5(self, run_id):
        ns, nv = self._get_point_counts(run_id)

        with h5py.File(self._boundary_h5_path(run_id), "r") as hb:
            # Two independent pools for geometry/surface to keep stochasticity.
            stride_g, offset_g = self._strided_slice_params(ns, self.geometry_points)
            geo_pool = np.asarray(hb["coords"][offset_g::stride_g], dtype=np.float32)

            stride_s, offset_s = self._strided_slice_params(ns, self.surface_points)
            surf_pool = np.asarray(hb["coords"][offset_s::stride_s], dtype=np.float32)
            p_pool = np.asarray(hb["pMeanTrim"][offset_s::stride_s], dtype=np.float32).reshape(-1, 1)
            bcoords_geo = self._sample_local(geo_pool, self.geometry_points)
            # Keep surface coords and pressure aligned using shared local index.
            surf_n = min(surf_pool.shape[0], p_pool.shape[0])
            surf_pool = surf_pool[:surf_n]
            p_pool = p_pool[:surf_n]
            if self.surface_points > 0 and self.surface_points < surf_n:
                sidx = np.random.choice(surf_n, size=self.surface_points, replace=False)
                bcoords_surf = surf_pool[sidx]
                ps = p_pool[sidx]
            else:
                bcoords_surf = surf_pool
                ps = p_pool

        with h5py.File(self._volume_h5_path(run_id), "r") as hv:
            stride_v, offset_v = self._strided_slice_params(nv, self.volume_points)
            vcoords_pool = np.asarray(hv["coords"][offset_v::stride_v], dtype=np.float32)
            u_pool = np.asarray(hv["UMeanTrim"][offset_v::stride_v], dtype=np.float32)
            vol_n = min(vcoords_pool.shape[0], u_pool.shape[0])
            vcoords_pool = vcoords_pool[:vol_n]
            u_pool = u_pool[:vol_n]
            if self.volume_points > 0 and self.volume_points < vol_n:
                vidx = np.random.choice(vol_n, size=self.volume_points, replace=False)
                vcoords = vcoords_pool[vidx]
                u = u_pool[vidx]
            else:
                vcoords = vcoords_pool
                u = u_pool
            if u.ndim == 1:
                u = u.reshape(-1, 1)
            u = u[:, :3]

        geo_mesh = torch.from_numpy(bcoords_geo)
        surf_mesh = torch.from_numpy(bcoords_surf)
        surf_data = torch.from_numpy(ps)
        vol_mesh = torch.from_numpy(vcoords)
        vol_data = torch.from_numpy(u)

        return geo_mesh, surf_mesh, surf_data, vol_mesh, vol_data

    def _load_case_from_preprocessed(self, run_id):
        run_dir = self._run_dir(run_id)
        surf_coords = np.load(run_dir / "surface_coords.npy", mmap_mode="r")
        surf_p = np.load(run_dir / "surface_pMeanTrim.npy", mmap_mode="r")
        surf_n = np.load(run_dir / "surface_normals.npy", mmap_mode="r")
        surf_wx = np.load(run_dir / "surface_wallShearStressMeanTrim_x.npy", mmap_mode="r")
        surf_wy = np.load(run_dir / "surface_wallShearStressMeanTrim_y.npy", mmap_mode="r")
        surf_wz = np.load(run_dir / "surface_wallShearStressMeanTrim_z.npy", mmap_mode="r")
        vol_coords = np.load(run_dir / "volume_coords.npy", mmap_mode="r")
        vol_u = np.load(run_dir / "volume_UMeanTrim.npy", mmap_mode="r")
        vol_p = np.load(run_dir / "volume_pMeanTrim.npy", mmap_mode="r")

        ns = int(surf_coords.shape[0])
        nv = int(vol_coords.shape[0])
        geo_idx = self._sample_idx(ns, self.geometry_points).numpy().astype(np.int64, copy=False)
        surf_idx = self._sample_idx(ns, self.surface_points).numpy().astype(np.int64, copy=False)
        vol_idx = self._sample_idx(nv, self.volume_points).numpy().astype(np.int64, copy=False)

        geo_mesh = torch.from_numpy(np.asarray(surf_coords[geo_idx], dtype=np.float32))
        surf_mesh = torch.from_numpy(np.asarray(surf_coords[surf_idx], dtype=np.float32))
        surf_data = torch.from_numpy(
            np.concatenate(
                [
                    np.asarray(surf_p[surf_idx], dtype=np.float32).reshape(-1, 1),
                    np.asarray(surf_n[surf_idx], dtype=np.float32),
                    np.asarray(surf_wx[surf_idx], dtype=np.float32).reshape(-1, 1),
                    np.asarray(surf_wy[surf_idx], dtype=np.float32).reshape(-1, 1),
                    np.asarray(surf_wz[surf_idx], dtype=np.float32).reshape(-1, 1),
                ],
                axis=1,
            )
        )
        vol_mesh = torch.from_numpy(np.asarray(vol_coords[vol_idx], dtype=np.float32))
        vol_data = torch.from_numpy(
            np.concatenate(
                [
                    np.asarray(vol_p[vol_idx], dtype=np.float32).reshape(-1, 1),
                    np.asarray(vol_u[vol_idx], dtype=np.float32),
                ],
                axis=1,
            )
        )

        return geo_mesh, surf_mesh, surf_data, vol_mesh, vol_data

    def __getitem__(self, idx):
        run_id = self.data[idx]
        if self.preprocessed_mode:
            geo_mesh, surf_mesh, surf_data, vol_mesh, vol_data = self._load_case_from_preprocessed(run_id)
        else:
            # Fast path: sample directly from H5 so we avoid full-array materialization.
            geo_mesh, surf_mesh, surf_data, vol_mesh, vol_data = self._load_case_sampled_from_h5(run_id)

        geo_mesh = self._normalize_pos(geo_mesh, self.min_pos, self.max_pos)
        surf_mesh = self._normalize_pos(surf_mesh, self.min_pos, self.max_pos)
        vol_mesh = self._normalize_pos(vol_mesh, self.min_pos, self.max_pos)

        surf_data = (surf_data - self.mean_surf_data) / torch.clamp(self.std_surf_data, min=1e-12)
        vol_data = (vol_data - self.mean_vol_data) / torch.clamp(self.std_vol_data, min=1e-12)

        return geo_mesh, surf_mesh, surf_data, vol_mesh, vol_data


# Convenient alias.
DrivAerMLDataset = AhmedMLDatasetV2
