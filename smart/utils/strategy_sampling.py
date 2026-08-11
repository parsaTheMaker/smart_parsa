"""Numpy sampling helpers for the isolated SATLOSS strategy studies.

The implementations intentionally mirror the established DrivAerML strategy
definitions: downsampled views are uniform subsets, Gaussian masks remove
points probabilistically around a random center, and box masks remove an
axis-aligned box with side length ``2 * sigma``.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def sample_uniform_without_replacement(n: int, k: int, rng: np.random.Generator) -> np.ndarray:
    n = int(n)
    k = int(k)
    if k <= 0 or k >= n:
        return np.arange(n, dtype=np.int64)
    return rng.choice(n, size=k, replace=False).astype(np.int64, copy=False)


def sample_gaussian_ball_mask(
    coords_xyz: np.ndarray,
    base_budget: int,
    rng: np.random.Generator,
    *,
    std_fraction: float = 0.05,
    probability_at_1sigma: float = 0.33,
    min_survivors: int = 1,
) -> dict[str, Any]:
    coords = np.asarray(coords_xyz, dtype=np.float32)
    base_idx = sample_uniform_without_replacement(coords.shape[0], base_budget, rng)
    subset = np.asarray(coords[base_idx], dtype=np.float64)
    if subset.size == 0:
        raise RuntimeError("Gaussian-ball masking received an empty point cloud.")

    center_rel = int(rng.integers(0, subset.shape[0]))
    center = subset[center_rel]
    extent = float(np.max(np.max(subset, axis=0) - np.min(subset, axis=0)))
    sigma = max(float(std_fraction) * extent, 1.0e-12)
    probability_at_1sigma = float(np.clip(probability_at_1sigma, 1.0e-8, 0.999999))
    coefficient = -np.log(probability_at_1sigma)
    distance = np.linalg.norm(subset - center[None, :], axis=1)
    remove_probability = np.exp(-coefficient * (distance / sigma) ** 2)
    keep_mask = rng.random(subset.shape[0]) >= remove_probability

    min_survivors = max(1, min(int(min_survivors), subset.shape[0]))
    if int(np.count_nonzero(keep_mask)) < min_survivors:
        keep_scores = 1.0 - remove_probability
        keep_rel = np.argsort(keep_scores)[-min_survivors:]
        keep_mask = np.zeros(subset.shape[0], dtype=bool)
        keep_mask[keep_rel] = True

    return {
        "base_idx": np.asarray(base_idx, dtype=np.int64),
        "kept_idx": np.asarray(base_idx[keep_mask], dtype=np.int64),
        "keep_mask": keep_mask,
        "removed_mask": ~keep_mask,
        "remove_probability": remove_probability.astype(np.float32),
        "distance_to_center": distance.astype(np.float32),
        "center_rel": center_rel,
        "center_point": center.astype(np.float32),
        "sigma_radius": sigma,
    }


def sample_box_mask(
    coords_xyz: np.ndarray,
    base_budget: int,
    rng: np.random.Generator,
    *,
    std_fraction: float = 0.05,
) -> dict[str, Any]:
    coords = np.asarray(coords_xyz, dtype=np.float32)
    base_idx = sample_uniform_without_replacement(coords.shape[0], base_budget, rng)
    subset = np.asarray(coords[base_idx], dtype=np.float64)
    if subset.size == 0:
        raise RuntimeError("Box masking received an empty point cloud.")

    center_rel = int(rng.integers(0, subset.shape[0]))
    center = subset[center_rel]
    extent = float(np.max(np.max(subset, axis=0) - np.min(subset, axis=0)))
    sigma = max(float(std_fraction) * extent, 1.0e-12)
    removed_mask = np.all(np.abs(subset - center[None, :]) <= sigma, axis=1)
    keep_mask = ~removed_mask
    if not bool(np.any(keep_mask)):
        raise RuntimeError("Box masking removed every candidate point.")

    return {
        "base_idx": np.asarray(base_idx, dtype=np.int64),
        "kept_idx": np.asarray(base_idx[keep_mask], dtype=np.int64),
        "keep_mask": keep_mask,
        "removed_mask": removed_mask,
        "distance_to_center": np.linalg.norm(subset - center[None, :], axis=1).astype(np.float32),
        "center_rel": center_rel,
        "center_point": center.astype(np.float32),
        "sigma_radius": sigma,
        "box_side_length": 2.0 * sigma,
        "box_min": (center - sigma).astype(np.float32),
        "box_max": (center + sigma).astype(np.float32),
    }


def sample_strategy(
    coords_xyz: np.ndarray,
    strategy: str,
    primary_budget: int,
    rng: np.random.Generator,
    *,
    downsample_budget: int | None = None,
    gaussian_std_fraction: float = 0.05,
    gaussian_probability_at_1sigma: float = 0.33,
    gaussian_min_survivors: int = 1,
) -> dict[str, Any]:
    strategy = str(strategy).strip().lower().replace("-", "_")
    if strategy in {"downsample", "subsample", "uniform_subsample"}:
        budget = int(primary_budget if downsample_budget is None else downsample_budget)
        indices = sample_uniform_without_replacement(int(np.asarray(coords_xyz).shape[0]), budget, rng)
        return {
            "strategy": "downsample",
            "base_idx": indices,
            "kept_idx": indices,
            "removed_mask": np.zeros(indices.shape[0], dtype=bool),
            "center_point": None,
        }
    if strategy in {"gaussian_ball", "gaussian_ball_masked", "gaussian_mask"}:
        info = sample_gaussian_ball_mask(
            coords_xyz,
            primary_budget,
            rng,
            std_fraction=gaussian_std_fraction,
            probability_at_1sigma=gaussian_probability_at_1sigma,
            min_survivors=gaussian_min_survivors,
        )
        info["strategy"] = "gaussian_ball_masked"
        return info
    if strategy in {"box", "box_mask", "box_masked"}:
        info = sample_box_mask(
            coords_xyz,
            primary_budget,
            rng,
            std_fraction=gaussian_std_fraction,
        )
        info["strategy"] = "box_masked"
        return info
    raise ValueError(f"Unknown isolated sampling strategy: {strategy!r}")
