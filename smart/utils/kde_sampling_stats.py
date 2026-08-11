"""Shared statistics for SMART KDE sampling diagnostics."""

from __future__ import annotations

import math

import numpy as np


def finite_float(value: float) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"Expected a finite value, got {value!r}.")
    return value


def normalize_geometry(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected [N,3] geometry coordinates, got {points.shape}.")
    if not np.isfinite(points).all():
        raise ValueError("Geometry contains non-finite coordinates.")
    lower = points.min(axis=0)
    upper = points.max(axis=0)
    span = np.maximum(upper - lower, 1.0e-12)
    normalized = (points - lower) / span
    return normalized.astype(np.float32, copy=False), lower, upper


def sample_uniform(n: int, k: int, rng: np.random.Generator) -> np.ndarray:
    if k <= 0 or k >= n:
        return np.arange(n, dtype=np.int64)
    return rng.choice(n, size=k, replace=False).astype(np.int64, copy=False)


def sample_inverse_density(
    log_density: np.ndarray,
    k: int,
    beta: float,
    rng: np.random.Generator,
) -> np.ndarray:
    n = int(log_density.shape[0])
    if k <= 0 or k >= n:
        return np.arange(n, dtype=np.int64)
    log_weights = -float(beta) * np.asarray(log_density, dtype=np.float64)
    log_weights -= np.max(log_weights)
    weights = np.exp(log_weights)
    probabilities = weights / np.clip(weights.sum(), 1.0e-300, None)
    return rng.choice(n, size=k, replace=False, p=probabilities).astype(np.int64, copy=False)


def quantile(values: np.ndarray, q: float) -> float:
    return finite_float(np.quantile(np.asarray(values, dtype=np.float64), q))


def ks_distance(values_a: np.ndarray, values_b: np.ndarray) -> float:
    a = np.sort(np.asarray(values_a, dtype=np.float64))
    b = np.sort(np.asarray(values_b, dtype=np.float64))
    combined = np.sort(np.concatenate((a, b)))
    cdf_a = np.searchsorted(a, combined, side="right") / float(a.size)
    cdf_b = np.searchsorted(b, combined, side="right") / float(b.size)
    return finite_float(np.max(np.abs(cdf_a - cdf_b)))


def wasserstein_1d(values_a: np.ndarray, values_b: np.ndarray, points: int = 2048) -> float:
    q = np.linspace(0.0, 1.0, min(points, max(values_a.size, values_b.size)), dtype=np.float64)
    qa = np.quantile(np.asarray(values_a, dtype=np.float64), q)
    qb = np.quantile(np.asarray(values_b, dtype=np.float64), q)
    return finite_float(np.trapezoid(np.abs(qa - qb), q))


def density_stats(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    positive_density = np.exp(np.clip(values, -700.0, 700.0))
    return {
        "count": int(values.size),
        "mean_log_density": finite_float(values.mean()),
        "std_log_density": finite_float(values.std()),
        "min_log_density": finite_float(values.min()),
        "p01_log_density": quantile(values, 0.01),
        "p05_log_density": quantile(values, 0.05),
        "p25_log_density": quantile(values, 0.25),
        "median_log_density": quantile(values, 0.50),
        "p75_log_density": quantile(values, 0.75),
        "p95_log_density": quantile(values, 0.95),
        "p99_log_density": quantile(values, 0.99),
        "max_log_density": finite_float(values.max()),
        "mean_kde_density": finite_float(positive_density.mean()),
        "std_kde_density": finite_float(positive_density.std()),
        "kde_density_cv": finite_float(positive_density.std() / max(positive_density.mean(), 1.0e-300)),
    }


def effective_sample_size(log_density: np.ndarray, beta: float) -> float:
    log_weights = -float(beta) * np.asarray(log_density, dtype=np.float64)
    log_weights -= np.max(log_weights)
    weights = np.exp(log_weights)
    return finite_float(weights.sum() ** 2 / max(np.square(weights).sum(), 1.0e-300))
