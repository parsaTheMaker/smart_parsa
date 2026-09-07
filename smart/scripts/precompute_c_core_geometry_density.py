"""Materialize C-core geometry-density caches before concurrent DeAL training."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from utils.geometry_density import estimate_log_sampling_density


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--knn-k", type=int, default=16)
    parser.add_argument("--estimator", choices=("kde", "rk2", "tangent_cov"), default="kde")
    parser.add_argument("--cache-dtype", choices=("float16", "float32"), default="float16")
    return parser.parse_args()


def atomic_save(path: Path, value: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f"{path.name}.{os.getpid()}.", suffix=".npy", delete=False) as handle:
        temporary = Path(handle.name)
        np.save(handle, value, allow_pickle=False)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def compute_one(task):
    case_dir, knn_k, estimator, cache_dtype = task
    case_dir = Path(case_dir)
    destination = case_dir / f"geometry_log_density_k{knn_k}_{estimator}.npy"
    coords = np.load(case_dir / "geometry_coords.npy", mmap_mode="r")
    expected = int(coords.shape[0])
    if destination.is_file():
        try:
            cached = np.load(destination, mmap_mode="r")
            if cached.shape == (expected,) and np.isfinite(cached).all():
                return case_dir.name, "cached"
        except (OSError, ValueError):
            pass
    points = np.asarray(coords, dtype=np.float32)
    lower = points.min(axis=0)
    normalized = (points - lower) / np.maximum(points.max(axis=0) - lower, 1.0e-12)
    normalized = np.clip(normalized, 0.0, 1.0 - 1.0e-6)
    values = estimate_log_sampling_density(
        torch.from_numpy(normalized).unsqueeze(0),
        knn_k=knn_k,
        neighbor_hops=1,
        estimator=estimator,
    ).squeeze(0).numpy()
    dtype = np.float16 if cache_dtype == "float16" else np.float32
    values = np.asarray(values, dtype=dtype)
    if values.shape != (expected,) or not np.isfinite(values).all():
        raise ValueError(f"Invalid density result for {case_dir}: {values.shape}")
    atomic_save(destination, values)
    return case_dir.name, "written"


def main():
    args = parse_args()
    root = args.data_root.expanduser().resolve()
    manifest = json.loads((root / "preprocessed_manifest.json").read_text(encoding="utf-8"))
    case_ids = sorted(set(map(int, manifest["train_ids"])) | set(map(int, manifest["validation_ids"])))
    tasks = [(str(root / f"case_{case_id:05d}"), args.knn_k, args.estimator, args.cache_dtype) for case_id in case_ids]
    counts = {"cached": 0, "written": 0}
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as executor:
        for _, status in tqdm(executor.map(compute_one, tasks), total=len(tasks), desc="C-core KDE caches"):
            counts[status] += 1
    print(f"Validated {len(tasks)} density caches: {counts['written']} written, {counts['cached']} reused.")


if __name__ == "__main__":
    main()

