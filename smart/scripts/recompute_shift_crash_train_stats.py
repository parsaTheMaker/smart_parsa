#!/usr/bin/env python3
"""Recompute SHIFT-Crash normalization statistics for the active train split.

The preprocessing root keeps global ``*.npy`` normalization files.  Whenever
``splits.json`` is rebuilt, these files must be rebuilt as well: using the old
split silently leaks validation cases into feature/target normalization and
makes the experiment provenance ambiguous.  This script streams memmaps one
case at a time, so it has bounded RAM use even for the complete dataset.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


STAT_FILES = (
    "reference_position_minmax.npy",
    "reference_position_stats.npy",
    "terminal_displacement_stats.npy",
    "parameter_stats.npy",
    "static_geometry_feature_stats.npy",
)


class RunningMoments:
    """Float64 sum/sumsq accumulator for population mean and standard deviation."""

    def __init__(self, channels: int):
        self.count = 0
        self.total = np.zeros(channels, dtype=np.float64)
        self.total_sq = np.zeros(channels, dtype=np.float64)

    def update(self, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=np.float64)
        if values.ndim == 1:
            values = values[None, :]
        if values.ndim != 2 or values.shape[1] != self.total.shape[0]:
            raise ValueError(f"Expected [N,{self.total.shape[0]}], got {values.shape}")
        if not np.isfinite(values).all():
            raise FloatingPointError("Encountered non-finite values while rebuilding statistics.")
        self.count += int(values.shape[0])
        self.total += values.sum(axis=0, dtype=np.float64)
        self.total_sq += np.square(values, dtype=np.float64).sum(axis=0, dtype=np.float64)

    def finalize(self) -> np.ndarray:
        if self.count <= 0:
            raise RuntimeError("No values were accumulated.")
        mean = self.total / self.count
        variance = np.maximum(self.total_sq / self.count - np.square(mean), 0.0)
        return np.stack([mean, np.sqrt(variance)], axis=0).astype(np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--split", default="train")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def atomic_save(path: Path, value: np.ndarray) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.save(handle, np.asarray(value, dtype=np.float32))
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    root = args.data_root.expanduser().resolve()
    split_path = root / "splits.json"
    with split_path.open("r", encoding="utf-8") as handle:
        splits = json.load(handle)
    case_ids = [str(case_id) for case_id in splits.get(args.split, [])]
    if not case_ids:
        raise ValueError(f"Split {args.split!r} is missing or empty in {split_path}.")

    cases_root = root / "cases"
    coordinate_moments = RunningMoments(3)
    target_moments = RunningMoments(3)
    feature_moments = RunningMoments(7)
    parameter_moments = RunningMoments(6)
    coordinate_min = np.full(3, np.inf, dtype=np.float64)
    coordinate_max = np.full(3, -np.inf, dtype=np.float64)
    nodes_per_case = None

    for position, case_id in enumerate(case_ids, start=1):
        case_root = cases_root / case_id
        state = np.load(case_root / "geometry_and_terminal_displacement.npy", mmap_mode="r")
        features = np.load(case_root / "static_geometry_features.npy", mmap_mode="r")
        params = np.load(case_root / "params.npy")
        if state.ndim != 2 or state.shape[1] != 6:
            raise ValueError(f"Invalid state array for {case_id}: {state.shape}")
        if features.shape != (state.shape[0], 7):
            raise ValueError(f"Invalid static feature array for {case_id}: {features.shape}")
        if params.shape != (6,):
            raise ValueError(f"Invalid parameter array for {case_id}: {params.shape}")
        if nodes_per_case is None:
            nodes_per_case = int(state.shape[0])
        elif int(state.shape[0]) != nodes_per_case:
            raise ValueError(
                "The current trainer assumes fixed-size point clouds; found "
                f"{state.shape[0]} nodes for {case_id}, expected {nodes_per_case}."
            )

        coordinates = np.asarray(state[:, :3])
        coordinate_moments.update(coordinates)
        target_moments.update(np.asarray(state[:, 3:6]))
        feature_moments.update(np.asarray(features))
        parameter_moments.update(params)
        coordinate_min = np.minimum(coordinate_min, coordinates.min(axis=0))
        coordinate_max = np.maximum(coordinate_max, coordinates.max(axis=0))
        if position == 1 or position % 100 == 0 or position == len(case_ids):
            print(f"[stats] {position}/{len(case_ids)} cases", flush=True)

    outputs = {
        "reference_position_minmax.npy": np.stack([coordinate_min, coordinate_max], axis=0),
        "reference_position_stats.npy": coordinate_moments.finalize(),
        "terminal_displacement_stats.npy": target_moments.finalize(),
        "parameter_stats.npy": parameter_moments.finalize(),
        "static_geometry_feature_stats.npy": feature_moments.finalize(),
    }
    print(f"[stats] train_cases={len(case_ids)}, nodes_per_case={nodes_per_case}")
    for name, value in outputs.items():
        print(f"[stats] {name}: shape={value.shape}, finite={bool(np.isfinite(value).all())}")

    if args.dry_run:
        print("[stats] Dry run: no files written.")
        return
    if not args.overwrite:
        raise ValueError("Refusing to overwrite active stats without --overwrite.")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = root / f"stats_before_{args.split}_split_rebuild_{timestamp}"
    backup.mkdir(parents=False, exist_ok=False)
    for name in STAT_FILES:
        source = root / name
        if not source.is_file():
            raise FileNotFoundError(f"Missing existing stats file: {source}")
        shutil.copy2(source, backup / name)
    for name, value in outputs.items():
        atomic_save(root / name, value)
    provenance = {
        "split": args.split,
        "case_count": len(case_ids),
        "nodes_per_case": nodes_per_case,
        "case_ids": case_ids,
        "generated_at_utc": timestamp,
        "backup_dir": str(backup),
        "population_standard_deviation": True,
    }
    with (root / "normalization_provenance.json").open("w", encoding="utf-8") as handle:
        json.dump(provenance, handle, indent=2)
        handle.write("\n")
    metadata_path = root / "dataset_metadata.json"
    if metadata_path.is_file():
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        metadata.setdefault("normalization", {})["statistics_split"] = args.split
        metadata["normalization"]["statistics_recomputed_at_utc"] = timestamp
        metadata["normalization"]["statistics_case_count"] = len(case_ids)
        temporary = metadata_path.with_name(f".{metadata_path.name}.{os.getpid()}.tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, metadata_path)
    print(f"[stats] Wrote active train-only stats; previous values backed up at {backup}")


if __name__ == "__main__":
    main()
