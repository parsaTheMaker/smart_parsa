"""Dataset adapter for the deterministic C-core magnetic benchmark."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from data.toy_heat_exchange_dataset import ToyHeatExchangeDataset


class CCoreMagneticDataset(ToyHeatExchangeDataset):
    """Read full native C-core surface and solid-volume point fields."""

    CACHE_VERSION = "c_core_magnetic_train_stats_v1"
    SURFACE_FIELDS = ("B_x", "B_y", "B_z")
    VOLUME_FIELDS = ("B_x", "B_y", "B_z")
    DATASET_LABEL = "CCoreMagnetic"

    def _load_train_statistics(self):
        surface_path, volume_path, position_path = self._stats_paths()
        if not (surface_path.is_file() and volume_path.is_file() and position_path.is_file()):
            surface_sum = np.zeros(3, dtype=np.float64)
            surface_sq_sum = np.zeros(3, dtype=np.float64)
            volume_sum = np.zeros(3, dtype=np.float64)
            volume_sq_sum = np.zeros(3, dtype=np.float64)
            surface_count = volume_count = 0
            pos_min = np.full(3, np.inf, dtype=np.float64)
            pos_max = np.full(3, -np.inf, dtype=np.float64)
            for case_id in self.training_ids:
                metadata = json.loads(
                    (self._run_dir(case_id) / "case_metadata.json").read_text(encoding="utf-8")
                )
                surface_sum += np.asarray(metadata["surface_sum"], dtype=np.float64)
                surface_sq_sum += np.asarray(metadata["surface_sq_sum"], dtype=np.float64)
                volume_sum += np.asarray(metadata["volume_sum"], dtype=np.float64)
                volume_sq_sum += np.asarray(metadata["volume_sq_sum"], dtype=np.float64)
                surface_count += int(metadata["surface_count"])
                volume_count += int(metadata["volume_count"])
                pos_min = np.minimum(pos_min, np.asarray(metadata["position_min"], dtype=np.float64))
                pos_max = np.maximum(pos_max, np.asarray(metadata["position_max"], dtype=np.float64))

            def stats(total, total_sq, count):
                mean = total / max(count, 1)
                variance = np.maximum(
                    (total_sq - total * total / max(count, 1)) / max(count - 1, 1),
                    1.0e-12,
                )
                return np.stack((mean, np.sqrt(variance))).astype(np.float32)

            np.save(surface_path, stats(surface_sum, surface_sq_sum, surface_count), allow_pickle=False)
            np.save(volume_path, stats(volume_sum, volume_sq_sum, volume_count), allow_pickle=False)
            np.save(position_path, np.stack((pos_min, pos_max)).astype(np.float32), allow_pickle=False)
        surface_stats = np.asarray(np.load(surface_path), dtype=np.float32)
        volume_stats = np.asarray(np.load(volume_path), dtype=np.float32)
        self.mean_surf_data = torch.from_numpy(surface_stats[0])
        self.std_surf_data = torch.from_numpy(np.maximum(surface_stats[1], 1.0e-12))
        self.mean_vol_data = torch.from_numpy(volume_stats[0])
        self.std_vol_data = torch.from_numpy(np.maximum(volume_stats[1], 1.0e-12))
        bounds = np.asarray(np.load(position_path), dtype=np.float32)
        self.min_pos = torch.from_numpy(bounds[0])
        self.position_span = torch.from_numpy(np.maximum(bounds[1] - bounds[0], 1.0e-12))
