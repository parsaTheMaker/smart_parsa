#!/usr/bin/env python3
"""Export deterministic DrivAerML inverse-density point-cloud views as VTKs."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from data.ahmedml_dataset_v2 import AhmedMLDatasetV2
from compare_drivaerml_sampling_invariance import (
    sample_inverse_density_without_replacement,
    write_polydata_vtk,
)


def parse_betas(value: str) -> list[float]:
    betas = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not betas:
        raise ValueError("At least one beta value is required.")
    if any(beta < 0.0 or beta > 1.0 for beta in betas):
        raise ValueError("This export is restricted to beta values in [0, 1].")
    return betas


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="/mnt/ssdraid/parsa/drivaerml_preprocessed")
    parser.add_argument("--run-id", type=int, default=29)
    parser.add_argument("--input-points", type=int, default=131072)
    parser.add_argument("--betas", default="0,0.2,0.4,0.6,0.8,1.0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--density-estimator", choices=("kde", "rk2", "tangent_cov"), default="kde")
    parser.add_argument("--density-knn-k", type=int, default=16)
    parser.add_argument("--density-neighbor-hops", type=int, default=1)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    betas = parse_betas(args.betas)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    run_dir = Path(args.data_root).expanduser().resolve() / f"run_{int(args.run_id)}"
    points_path = run_dir / "surface_coords.npy"
    if not points_path.is_file():
        raise FileNotFoundError(f"Surface point cloud not found: {points_path}")

    # This loads the same full-cloud KDE cache and normalized geometry convention
    # used by the DrivAerML sampling-invariance comparisons.
    dataset = AhmedMLDatasetV2(
        saved_folder=str(Path(args.data_root).expanduser().resolve()),
        if_test=True,
        geometry_points=int(args.input_points),
        surface_points=65536,
        volume_points=65536,
        require_preprocessed=True,
        geometry_density_knn_k=int(args.density_knn_k),
        geometry_density_neighbor_hops=int(args.density_neighbor_hops),
        geometry_density_estimator=str(args.density_estimator),
        geometry_density_cache_dtype="float16",
    )
    points = np.load(points_path).astype(np.float32, copy=False)
    if int(args.input_points) > points.shape[0]:
        raise ValueError(f"Requested {args.input_points} points, but run {args.run_id} only has {points.shape[0]}.")
    log_density = dataset._load_or_compute_full_geometry_density(
        int(args.run_id), expected_n=int(points.shape[0])
    ).cpu().numpy().astype(np.float32, copy=False)

    for beta in betas:
        rng = np.random.default_rng(
            np.random.SeedSequence([int(args.seed), int(args.run_id), 77777, int(round(beta * 100.0))])
        )
        indices = sample_inverse_density_without_replacement(
            log_density,
            int(args.input_points),
            beta,
            rng,
        )
        sampled_log_density = log_density[indices]
        unnormalized_weight = np.exp(-float(beta) * sampled_log_density.astype(np.float64)).astype(np.float32)
        path = output_dir / (
            f"drivaerml_test_run_{int(args.run_id)}_input_points_{int(args.input_points)}_"
            f"inverse_density_beta_{beta:.2f}.vtk"
        )
        write_polydata_vtk(
            path,
            points[indices],
            {
                "beta": np.full((indices.shape[0],), float(beta), dtype=np.float32),
                "kde16_log_density": sampled_log_density,
                "inverse_density_sampling_weight": unnormalized_weight,
            },
        )
        print(f"beta={beta:.2f}: wrote {indices.shape[0]} points to {path}")


if __name__ == "__main__":
    main()
