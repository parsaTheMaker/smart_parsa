#!/usr/bin/env python3
"""Audit Heat Exchanger discretization error against the persisted native mesh.

The production field is retained on the native Gmsh/SKFEM tetrahedral mesh.
For each selected case this script regenerates a geometrically identical, coarser
mesh, resolves the nonlinear problem, and compares temperature at interior
locations that lie in both meshes.  It also compares the integrated exterior
heat flow, which is determined by the surface-flux target.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from generate_toy_heat_exchange_benchmark import (
    boundary_triangles,
    classify_boundary_faces,
    make_tetra_mesh,
    solve_heat,
    tetra_gradients,
)


DEFAULT_ROOT = Path("/mnt/ssdraid/parsa/toy_heat_exchange_fem_v1")


def parse_case_ids(value: str) -> list[int]:
    ids = [int(token) for token in value.split(",") if token.strip()]
    if not ids:
        raise argparse.ArgumentTypeError("Provide at least one comma-separated case id.")
    return ids


def as_grid(points: np.ndarray, tetra: np.ndarray, temperature: np.ndarray):
    import pyvista as pv

    cells = np.column_stack((np.full(tetra.shape[0], 4, dtype=np.int64), tetra)).ravel()
    grid = pv.UnstructuredGrid(cells, np.full(tetra.shape[0], pv.CellType.TETRA, dtype=np.uint8), points)
    grid.point_data["temperature"] = np.asarray(temperature, dtype=np.float64)
    return grid


def exterior_heat_flow(
    points: np.ndarray,
    tetra: np.ndarray,
    temperature: np.ndarray,
    params: dict,
    args: dict,
    mesh_h_max: float,
) -> float:
    faces, owners, areas, normals = boundary_triangles(points, tetra)
    _inner, outer, _ = classify_boundary_faces(points, faces, params, mesh_h_max)
    face_temperature = temperature[faces].mean(axis=1)
    flux = np.empty(faces.shape[0], dtype=np.float64)
    flux[outer] = (
        float(args["exterior_biot"]) * face_temperature[outer]
        + float(args["radiation"])
        * (
            np.power(face_temperature[outer] + float(args["ambient_temperature_ratio"]), 4.0)
            - float(args["ambient_temperature_ratio"]) ** 4
        )
    )
    # The outer Robin flux is the exported surface target on this boundary.
    return float(np.sum(flux[outer] * areas[outer]))


def audit_case(
    case_id: int,
    root: Path,
    mesh_scale: float,
    query_count: int,
    gmsh_threads: int,
) -> dict[str, float | int]:
    import pyvista as pv

    case_dir = root / f"case_{case_id:05d}"
    metadata = json.loads((case_dir / "case_metadata.json").read_text(encoding="utf-8"))
    params = metadata["parameters"]
    physics = metadata["physics"]
    fine_points = np.load(case_dir / "surface_mesh_points.npy").astype(np.float64)
    fine_tetra = np.load(case_dir / "volume_mesh_tetra.npy").astype(np.int64)
    fine_temperature = np.load(case_dir / "fem_nodal_temperature.npy").astype(np.float64)

    solve_args = {
        "mesh_h_max": 0.028 * mesh_scale,
        "exterior_biot": float(physics["exterior_biot"]),
        "radiation": float(physics["radiation"]),
        "ambient_temperature_ratio": float(physics["tau"]),
        "nonlinear_conductivity": float(physics["nonlinear_conductivity"]),
        "nonlinear_iterations": 60,
        "nonlinear_tolerance": 2.0e-7,
    }
    coarse_points, coarse_tetra = make_tetra_mesh(
        params,
        h_min=0.0025 * mesh_scale,
        h_max=solve_args["mesh_h_max"],
        gmsh_threads=gmsh_threads,
    )
    coarse_temperature, residual, update, iterations, wall_error = solve_heat(
        coarse_points, coarse_tetra, params, solve_args
    )

    # Interior centroids avoid comparing interpolation outside the coarser domain.
    rng = np.random.default_rng(np.random.SeedSequence([42, case_id, 9182]))
    selected = rng.choice(fine_tetra.shape[0], size=min(query_count, fine_tetra.shape[0]), replace=False)
    query_points = fine_points[fine_tetra[selected]].mean(axis=1)
    fine_samples = pv.PolyData(query_points).sample(as_grid(fine_points, fine_tetra, fine_temperature))
    coarse_samples = pv.PolyData(query_points).sample(as_grid(coarse_points, coarse_tetra, coarse_temperature))
    valid = np.asarray(fine_samples["vtkValidPointMask"], dtype=bool) & np.asarray(
        coarse_samples["vtkValidPointMask"], dtype=bool
    )
    if valid.mean() < 0.995:
        raise RuntimeError(f"case_{case_id:05d}: only {valid.mean():.2%} of interior queries are shared.")
    fine_values = np.asarray(fine_samples["temperature"], dtype=np.float64)[valid]
    coarse_values = np.asarray(coarse_samples["temperature"], dtype=np.float64)[valid]
    temperature_rel_l2 = float(
        np.linalg.norm(coarse_values - fine_values) / max(np.linalg.norm(fine_values), 1.0e-12)
    )
    fine_heat = exterior_heat_flow(
        fine_points, fine_tetra, fine_temperature, params, solve_args, mesh_h_max=0.028
    )
    coarse_heat = exterior_heat_flow(
        coarse_points,
        coarse_tetra,
        coarse_temperature,
        params,
        solve_args,
        mesh_h_max=solve_args["mesh_h_max"],
    )
    heat_flow_relative_difference = float(abs(coarse_heat - fine_heat) / max(abs(fine_heat), 1.0e-12))
    return {
        "case_id": case_id,
        "native_nodes": int(fine_points.shape[0]),
        "native_tetrahedra": int(fine_tetra.shape[0]),
        "coarse_nodes": int(coarse_points.shape[0]),
        "coarse_tetrahedra": int(coarse_tetra.shape[0]),
        "shared_query_fraction": float(valid.mean()),
        "temperature_relative_l2": temperature_rel_l2,
        "exterior_heat_flow_relative_difference": heat_flow_relative_difference,
        "coarse_linear_residual": float(residual),
        "coarse_nonlinear_relative_change": float(update),
        "coarse_nonlinear_iterations": int(iterations),
        "coarse_channel_temperature_max_error": float(wall_error),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case-ids", type=parse_case_ids, default=[256, 270, 287])
    parser.add_argument("--mesh-scale", type=float, default=1.35)
    parser.add_argument("--query-count", type=int, default=50_000)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--gmsh-threads", type=int, default=1)
    args = parser.parse_args()
    if args.mesh_scale <= 1.0:
        raise ValueError("--mesh-scale must be greater than one for a coarser comparison mesh.")
    if min(args.query_count, args.workers, args.gmsh_threads) <= 0:
        raise ValueError("Query count, workers, and Gmsh threads must be positive.")

    root = args.data_root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    context = mp.get_context("spawn")
    rows: list[dict[str, float | int]] = []
    with ProcessPoolExecutor(max_workers=min(args.workers, len(args.case_ids)), mp_context=context) as executor:
        futures = {
            executor.submit(audit_case, case_id, root, args.mesh_scale, args.query_count, args.gmsh_threads): case_id
            for case_id in args.case_ids
        }
        for future in as_completed(futures):
            rows.append(future.result())
    rows.sort(key=lambda row: int(row["case_id"]))
    summary = {
        "description": "Coarser-to-native Heat Exchanger mesh audit at shared interior queries.",
        "native_mesh": "production first-order tetrahedral Gmsh mesh",
        "coarse_mesh_scale": float(args.mesh_scale),
        "case_ids": args.case_ids,
        "per_case": rows,
        "temperature_relative_l2": {
            "mean": float(np.mean([float(row["temperature_relative_l2"]) for row in rows])),
            "max": float(np.max([float(row["temperature_relative_l2"]) for row in rows])),
        },
        "exterior_heat_flow_relative_difference": {
            "mean": float(np.mean([float(row["exterior_heat_flow_relative_difference"]) for row in rows])),
            "max": float(np.max([float(row["exterior_heat_flow_relative_difference"]) for row in rows])),
        },
    }
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
