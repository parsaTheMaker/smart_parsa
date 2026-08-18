#!/usr/bin/env python3
"""Generate a deterministic analytic 3D benchmark for point-sampling robustness.

Each case is a smooth star-shaped solid.  Encoder points are sampled from a
case-specific, nonuniform virtual-meshing density, while surface and volume
queries are sampled independently from the reference distribution.  The two
analytic fields are deterministic functions of the geometry, so no numerical
solver, labels, or external data are required.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from utils.geometry_density import estimate_log_sampling_density


def unit_directions(rng: np.random.Generator, count: int) -> np.ndarray:
    points = rng.normal(size=(count, 3)).astype(np.float32)
    return points / np.maximum(np.linalg.norm(points, axis=1, keepdims=True), 1.0e-12)


def case_parameters(seed: int) -> dict[str, np.ndarray | float]:
    rng = np.random.default_rng(seed)
    rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(rotation) < 0:
        rotation[:, 0] *= -1.0
    bump = unit_directions(rng, 1)[0]
    density_axis = unit_directions(rng, 1)[0]
    density_center = unit_directions(rng, 1)[0]
    return {
        "rotation": rotation.astype(np.float32),
        "axes": rng.uniform(0.72, 1.28, size=3).astype(np.float32),
        "harmonics": rng.uniform(-0.22, 0.22, size=3).astype(np.float32),
        "bump_direction": bump.astype(np.float32),
        "bump_amplitude": float(rng.uniform(-0.12, 0.18)),
        "bump_sharpness": float(rng.uniform(7.0, 14.0)),
        "density_axis": density_axis.astype(np.float32),
        "density_center": density_center.astype(np.float32),
        "density_phase": float(rng.uniform(0.0, 1.0)),
        # Deliberately realistic-but-strong density variation: this mimics
        # curvature/feature-adaptive meshing without making the cloud singular.
        "density_amplitude": float(rng.uniform(1.3, 2.1)),
        "density_focus": float(rng.uniform(1.0, 2.0)),
    }


def radial_geometry(directions: np.ndarray, params: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    local = directions @ params["rotation"]
    axes = params["axes"]
    ellipsoid = 1.0 / np.sqrt(np.sum(np.square(local / axes[None, :]), axis=1))
    h = params["harmonics"]
    angular = h[0] * (local[:, 0] ** 2 - local[:, 1] ** 2)
    angular += h[1] * (2.0 * local[:, 1] * local[:, 2])
    angular += h[2] * (local[:, 0] ** 3 - 3.0 * local[:, 0] * local[:, 1] ** 2)
    bump_cos = np.sum(local * params["bump_direction"][None, :], axis=1)
    bump = params["bump_amplitude"] * np.exp(params["bump_sharpness"] * (bump_cos - 1.0))
    radius = ellipsoid * np.clip(1.0 + angular + bump, 0.55, 1.45)
    return (directions * radius[:, None]).astype(np.float32), radius.astype(np.float32), local.astype(np.float32)


def physical_loading(coords: np.ndarray) -> np.ndarray:
    """Positive, low-frequency load factor in physical coordinates.

    Using world coordinates avoids a hidden local-frame target dependency: it
    is directly recoverable from each query location and the visible geometry.
    """
    return (1.0 + 0.25 * coords[:, 0] - 0.18 * coords[:, 1] + 0.12 * coords[:, 2]).astype(np.float32)


def surface_field(local: np.ndarray, radius: np.ndarray, params: dict) -> np.ndarray:
    """Physical normal flux for the exact isotropic Poisson problem."""
    u = np.asarray(local, dtype=np.float32)
    axes = np.asarray(params["axes"], dtype=np.float32)
    h = np.asarray(params["harmonics"], dtype=np.float32)
    ellipsoid = 1.0 / np.sqrt(np.sum(np.square(u / axes[None, :]), axis=1))
    angular = h[0] * (u[:, 0] ** 2 - u[:, 1] ** 2)
    angular += h[1] * (2.0 * u[:, 1] * u[:, 2])
    angular += h[2] * (u[:, 0] ** 3 - 3.0 * u[:, 0] * u[:, 1] ** 2)
    bump_cos = np.sum(u * params["bump_direction"][None, :], axis=1)
    bump = params["bump_amplitude"] * np.exp(params["bump_sharpness"] * (bump_cos - 1.0))
    unclipped_scale = 1.0 + angular + bump
    radial_scale = np.clip(unclipped_scale, 0.55, 1.45)

    gradient_ellipsoid = -np.square(ellipsoid)[:, None] * ellipsoid[:, None] * u / np.square(axes)[None, :]
    gradient_angular = np.stack(
        [
            2.0 * h[0] * u[:, 0] + 3.0 * h[2] * (u[:, 0] ** 2 - u[:, 1] ** 2),
            -2.0 * h[0] * u[:, 1] + 2.0 * h[1] * u[:, 2] - 6.0 * h[2] * u[:, 0] * u[:, 1],
            2.0 * h[1] * u[:, 1],
        ],
        axis=1,
    )
    gradient_bump = (params["bump_sharpness"] * bump)[:, None] * params["bump_direction"][None, :]
    active = ((unclipped_scale > 0.55) & (unclipped_scale < 1.45)).astype(np.float32)[:, None]
    gradient_radius = gradient_ellipsoid * radial_scale[:, None] + ellipsoid[:, None] * active * (gradient_angular + gradient_bump)
    # R depends on direction only.  Project its ambient derivative onto S^2.
    gradient_radius -= u * np.sum(u * gradient_radius, axis=1, keepdims=True)

    identity = np.eye(3, dtype=np.float32)[None, :, :]
    jacobian = radius[:, None, None] * identity + u[:, :, None] * gradient_radius[:, None, :]
    determinant = np.linalg.det(jacobian)
    if float(determinant.min()) <= 1.0e-6:
        raise RuntimeError("Toy geometry map lost orientation; adjust the shape parameter bounds.")
    del determinant  # Orientation was checked above; K is the identity tensor.
    inverse_transpose_direction = np.linalg.solve(np.swapaxes(jacobian, 1, 2), u[..., None]).squeeze(-1)
    global_direction = u @ params["rotation"].T
    boundary_coords = global_direction * radius[:, None]
    flux = -2.0 * physical_loading(boundary_coords) * np.linalg.norm(inverse_transpose_direction, axis=1)
    return flux.astype(np.float32)[:, None]


def volume_field(coords: np.ndarray, params: dict) -> np.ndarray:
    radii = np.linalg.norm(coords, axis=1)
    directions = coords / np.maximum(radii[:, None], 1.0e-12)
    _, boundary_radius, local = radial_geometry(directions.astype(np.float32), params)
    xi = local * (radii / np.maximum(boundary_radius, 1.0e-12))[:, None]
    q2 = np.sum(np.square(xi), axis=1)
    value = (1.0 - q2) * physical_loading(coords)
    return value.astype(np.float32)[:, None]


def sample_nonuniform_surface(rng: np.random.Generator, params: dict, count: int) -> np.ndarray:
    candidates = unit_directions(rng, max(count * 5, 20_000))
    coords, _, _ = radial_geometry(candidates, params)
    axis_term = np.sin(2.0 * np.pi * (candidates @ params["density_axis"] + params["density_phase"]))
    focus_term = np.exp(10.0 * (candidates @ params["density_center"] - 1.0))
    log_weight = params["density_amplitude"] * axis_term + params["density_focus"] * focus_term
    weights = np.exp(log_weight - log_weight.max())
    weights /= weights.sum()
    chosen = rng.choice(candidates.shape[0], size=count, replace=False, p=weights)
    return coords[chosen].astype(np.float32)


def sample_reference_surface(rng: np.random.Generator, params: dict, count: int) -> tuple[np.ndarray, np.ndarray]:
    directions = unit_directions(rng, count)
    coords, radius, local = radial_geometry(directions, params)
    return coords, surface_field(local, radius, params)


def sample_reference_volume(rng: np.random.Generator, params: dict, count: int) -> tuple[np.ndarray, np.ndarray]:
    candidates = unit_directions(rng, max(count * 4, 20_000))
    _, boundary_radius, _ = radial_geometry(candidates, params)
    weights = np.power(boundary_radius, 3.0)
    weights /= weights.sum()
    directions = candidates[rng.choice(candidates.shape[0], size=count, replace=False, p=weights)]
    _, radius, _ = radial_geometry(directions, params)
    coords = directions * (rng.random(count, dtype=np.float32) ** (1.0 / 3.0))[:, None] * radius[:, None]
    return coords.astype(np.float32), volume_field(coords, params)


def save_array(path: Path, array: np.ndarray, dtype=np.float32) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("wb") as handle:
        np.save(handle, np.asarray(array, dtype=dtype), allow_pickle=False)
    temporary.replace(path)


def generate_case(case_id: int, split: str, args_dict: dict) -> dict:
    root = Path(args_dict["output_dir"])
    case_dir = root / f"case_{case_id:05d}"
    marker = case_dir / "_COMPLETE.json"
    if marker.is_file() and not args_dict["overwrite"]:
        return {"case_id": case_id, "skipped": True}
    case_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(np.random.SeedSequence([args_dict["seed"], case_id, 1701]))
    params = case_parameters(int(rng.integers(0, 2**31 - 1)))
    geometry = sample_nonuniform_surface(rng, params, args_dict["geometry_points"])
    surface, surface_data = sample_reference_surface(rng, params, args_dict["surface_points"])
    volume, volume_data = sample_reference_volume(rng, params, args_dict["volume_points"])
    save_array(case_dir / "geometry_coords.npy", geometry)
    save_array(case_dir / "surface_coords.npy", surface)
    save_array(case_dir / "surface_data.npy", surface_data)
    save_array(case_dir / "volume_coords.npy", volume)
    save_array(case_dir / "volume_data.npy", volume_data)
    all_coords = np.concatenate([geometry, surface, volume], axis=0)
    metadata = {
        "case_id": case_id,
        "split": split,
        "surface_sum": float(surface_data.sum()), "surface_sq_sum": float(np.square(surface_data).sum()),
        "surface_count": int(surface_data.shape[0]),
        "volume_sum": float(volume_data.sum()), "volume_sq_sum": float(np.square(volume_data).sum()),
        "volume_count": int(volume_data.shape[0]),
        "position_min": all_coords.min(axis=0).tolist(), "position_max": all_coords.max(axis=0).tolist(),
        "generator": "toy_satloss_poisson_v2", "parameters": {key: np.asarray(value).tolist() if isinstance(value, np.ndarray) else value for key, value in params.items()},
    }
    (case_dir / "case_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    marker.write_text(json.dumps({"case_id": case_id, "split": split}) + "\n", encoding="utf-8")
    return {"case_id": case_id, "skipped": False}


def cache_density(case_id: int, root: str, bounds: np.ndarray, knn_k: int) -> None:
    case_dir = Path(root) / f"case_{case_id:05d}"
    coords = np.asarray(np.load(case_dir / "geometry_coords.npy", mmap_mode="r"), dtype=np.float32)
    normalized = (coords - bounds[0]) / np.maximum(bounds[1] - bounds[0], 1.0e-12)
    normalized = np.clip(normalized, 0.0, 1.0 - 1.0e-6)
    density = estimate_log_sampling_density(
        torch.from_numpy(normalized).unsqueeze(0), knn_k=knn_k, estimator="kde"
    ).squeeze(0).cpu().numpy().astype(np.float16)
    save_array(case_dir / f"geometry_log_density_k{knn_k}_kde.npy", density, dtype=np.float16)


def save_preview(root: Path, case_id: int, results_dir: Path) -> None:
    case = root / f"case_{case_id:05d}"
    geometry = np.asarray(np.load(case / "geometry_coords.npy", mmap_mode="r"))
    surface = np.asarray(np.load(case / "surface_coords.npy", mmap_mode="r"))
    values = np.asarray(np.load(case / "surface_data.npy", mmap_mode="r"))[:, 0]
    rng = np.random.default_rng(42)
    fig = plt.figure(figsize=(15, 6), constrained_layout=True)
    for panel, (points, color, title) in enumerate(((geometry, None, "Native nonuniform encoder cloud"), (surface, values, "Reference surface target")), 1):
        ax = fig.add_subplot(1, 2, panel, projection="3d")
        idx = rng.choice(points.shape[0], min(12_000, points.shape[0]), replace=False)
        scatter_kwargs = {"s": 1.2, "rasterized": True}
        if color is not None:
            scatter_kwargs.update({"c": color[idx], "cmap": "coolwarm"})
        draw = ax.scatter(points[idx, 0], points[idx, 1], points[idx, 2], **scatter_kwargs)
        if color is not None:
            fig.colorbar(draw, ax=ax, shrink=0.72, label="Manufactured surface field")
        ax.set_title(title, fontsize=15, weight="bold")
        ax.set_box_aspect((1, 1, 1))
        ax.set_axis_off()
    results_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(results_dir / f"toy_satloss_case_{case_id:05d}_preview.png", dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="/mnt/ssdraid/parsa/toy_satloss_poisson_benchmark_v2")
    parser.add_argument("--results-dir", default="/home/parsa/smart_parsa/results/toy_satloss_benchmark")
    parser.add_argument("--train-cases", type=int, default=256)
    parser.add_argument("--validation-cases", type=int, default=64)
    parser.add_argument("--geometry-points", type=int, default=131072)
    parser.add_argument("--surface-points", type=int, default=65536)
    parser.add_argument("--volume-points", type=int, default=65536)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--density-knn-k", type=int, default=16)
    parser.add_argument("--skip-density-cache", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if min(args.train_cases, args.validation_cases, args.geometry_points, args.surface_points, args.volume_points) <= 0:
        raise ValueError("Case and point counts must be positive.")
    root = Path(args.output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    records = [(case_id, "train") for case_id in range(args.train_cases)]
    records += [(args.train_cases + index, "validation") for index in range(args.validation_cases)]
    args_dict = vars(args).copy()
    print(f"Generating {len(records)} analytic cases with {args.workers} CPU workers in {root}")
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = [executor.submit(generate_case, case_id, split, args_dict) for case_id, split in records]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Generating cases"):
            future.result()
    manifest = {
        "version": "toy_satloss_poisson_v2", "seed": args.seed,
        "train_ids": list(range(args.train_cases)),
        "validation_ids": list(range(args.train_cases, args.train_cases + args.validation_cases)),
        "geometry_points": args.geometry_points, "surface_points": args.surface_points, "volume_points": args.volume_points,
        "protocol": "nonuniform encoder cloud; independent reference surface and volume queries",
    }
    (root / "preprocessed_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    bounds_min = np.full(3, np.inf, dtype=np.float32)
    bounds_max = np.full(3, -np.inf, dtype=np.float32)
    surface_sum = surface_sq_sum = volume_sum = volume_sq_sum = 0.0
    surface_count = volume_count = 0
    for case_id in manifest["train_ids"]:
        meta = json.loads((root / f"case_{case_id:05d}" / "case_metadata.json").read_text(encoding="utf-8"))
        bounds_min = np.minimum(bounds_min, np.asarray(meta["position_min"], dtype=np.float32))
        bounds_max = np.maximum(bounds_max, np.asarray(meta["position_max"], dtype=np.float32))
        surface_sum += float(meta["surface_sum"]); surface_sq_sum += float(meta["surface_sq_sum"]); surface_count += int(meta["surface_count"])
        volume_sum += float(meta["volume_sum"]); volume_sq_sum += float(meta["volume_sq_sum"]); volume_count += int(meta["volume_count"])
    bounds = np.stack([bounds_min, bounds_max])
    def scalar_stats(total, total_sq, count):
        mean = total / float(count)
        variance = max((total_sq - total * total / float(count)) / float(max(count - 1, 1)), 1.0e-12)
        return np.asarray([mean, np.sqrt(variance)], dtype=np.float32)
    # Write train-only normalization before DDP workers are launched.  This
    # avoids a first-epoch race while preserving the dataset's exact contract.
    np.save(root / "surface_stats_toy_satloss_poisson_train_stats_v2.npy", scalar_stats(surface_sum, surface_sq_sum, surface_count), allow_pickle=False)
    np.save(root / "volume_stats_toy_satloss_poisson_train_stats_v2.npy", scalar_stats(volume_sum, volume_sq_sum, volume_count), allow_pickle=False)
    np.save(root / "position_stats_toy_satloss_poisson_train_stats_v2.npy", bounds.astype(np.float32), allow_pickle=False)
    if not args.skip_density_cache:
        print(f"Caching KDE-{args.density_knn_k} geometry densities with {args.workers} CPU workers")
        with ProcessPoolExecutor(max_workers=max(1, args.workers)) as executor:
            futures = [executor.submit(cache_density, case_id, str(root), bounds, args.density_knn_k) for case_id, _ in records]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Caching density"):
                future.result()
    save_preview(root, 0, Path(args.results_dir).expanduser().resolve())
    print(f"Complete. Preview: {Path(args.results_dir).resolve() / 'toy_satloss_case_00000_preview.png'}")


if __name__ == "__main__":
    main()
