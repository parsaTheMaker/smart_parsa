#!/usr/bin/env python3
"""Export JDecke/ubend-cfd sample fields to inspectable VTP files.

The source tensors store two structured 2D cell-center grids: a fluid region
and a solid region.  This exporter preserves their native grid connectivity as
quad faces and adds a region label so both domains can be inspected together
in ParaView.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pyvista as pv
from safetensors.numpy import load_file


def structured_quad_faces(rows: int, columns: int, offset: int = 0) -> np.ndarray:
    """Build VTK quad connectivity for a row-major structured point grid."""
    indices = np.arange(rows * columns, dtype=np.int64).reshape(rows, columns)
    quads = np.column_stack(
        (
            np.full((rows - 1) * (columns - 1), 4, dtype=np.int64),
            indices[:-1, :-1].ravel() + offset,
            indices[:-1, 1:].ravel() + offset,
            indices[1:, 1:].ravel() + offset,
            indices[1:, :-1].ravel() + offset,
        )
    )
    return quads.ravel()


def points_from_coords(coords: np.ndarray) -> np.ndarray:
    if coords.ndim != 3 or coords.shape[0] != 3:
        raise ValueError(f"Expected coordinates shaped [3, rows, columns], got {coords.shape}.")
    return np.moveaxis(coords, 0, -1).reshape(-1, 3)


def scalar_from_grid(grid: np.ndarray, name: str) -> np.ndarray:
    if grid.ndim != 2:
        raise ValueError(f"Expected scalar grid {name!r} shaped [rows, columns], got {grid.shape}.")
    return grid.reshape(-1).astype(np.float32, copy=False)


def vector_from_grid(grid: np.ndarray, name: str) -> np.ndarray:
    if grid.ndim != 3 or grid.shape[0] != 3:
        raise ValueError(f"Expected vector grid {name!r} shaped [3, rows, columns], got {grid.shape}.")
    return np.moveaxis(grid, 0, -1).reshape(-1, 3).astype(np.float32, copy=False)


def export_sample(source: Path, destination: Path) -> dict[str, int]:
    fields = load_file(str(source))
    required = {"coords", "U", "p", "T", "k", "nut", "solid_coords", "solid_T"}
    missing = sorted(required.difference(fields))
    if missing:
        raise KeyError(f"{source} is missing fields: {missing}")

    fluid_points = points_from_coords(fields["coords"])
    solid_points = points_from_coords(fields["solid_coords"])
    fluid_rows, fluid_columns = fields["coords"].shape[1:]
    solid_rows, solid_columns = fields["solid_coords"].shape[1:]
    points = np.vstack((fluid_points, solid_points))
    faces = np.concatenate(
        (
            structured_quad_faces(fluid_rows, fluid_columns),
            structured_quad_faces(solid_rows, solid_columns, offset=len(fluid_points)),
        )
    )
    mesh = pv.PolyData(points, faces)
    fluid_count, solid_count = len(fluid_points), len(solid_points)
    zeros_solid = np.zeros(solid_count, dtype=np.float32)
    zeros_fluid = np.zeros(fluid_count, dtype=np.float32)

    # Temperature is meaningful in both coupled regions. Other flow quantities
    # are zero in the solid and can be filtered using region_id in ParaView.
    mesh.point_data["temperature"] = np.concatenate(
        (scalar_from_grid(fields["T"], "T"), scalar_from_grid(fields["solid_T"], "solid_T"))
    )
    mesh.point_data["fluid_temperature"] = np.concatenate(
        (scalar_from_grid(fields["T"], "T"), zeros_solid)
    )
    mesh.point_data["solid_temperature"] = np.concatenate(
        (zeros_fluid, scalar_from_grid(fields["solid_T"], "solid_T"))
    )
    mesh.point_data["pressure"] = np.concatenate((scalar_from_grid(fields["p"], "p"), zeros_solid))
    velocity = np.vstack((vector_from_grid(fields["U"], "U"), np.zeros((solid_count, 3), dtype=np.float32)))
    mesh.point_data["velocity"] = velocity
    mesh.point_data["velocity_magnitude"] = np.linalg.norm(velocity, axis=1).astype(np.float32)
    mesh.point_data["turbulent_kinetic_energy"] = np.concatenate(
        (scalar_from_grid(fields["k"], "k"), zeros_solid)
    )
    mesh.point_data["turbulent_viscosity"] = np.concatenate(
        (scalar_from_grid(fields["nut"], "nut"), zeros_solid)
    )
    mesh.point_data["region_id"] = np.concatenate(
        (np.zeros(fluid_count, dtype=np.uint8), np.ones(solid_count, dtype=np.uint8))
    )

    destination.parent.mkdir(parents=True, exist_ok=True)
    mesh.save(destination, binary=True)
    return {"points": mesh.n_points, "cells": mesh.n_cells, "fluid_points": fluid_count, "solid_points": solid_count}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("/mnt/data/parsa/dataset_preview_sources/ubend/fields"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/home/parsa/smart_parsa/results/dataset_previews/ubend"),
    )
    parser.add_argument("--samples", default="0,1", help="Comma-separated sample indices.")
    args = parser.parse_args()

    for raw_index in args.samples.split(","):
        index = int(raw_index.strip())
        source = args.source_dir / f"sample_{index}.safetensors"
        destination = args.output_dir / f"ubend_sample_{index:04d}_fields.vtp"
        if not source.is_file():
            raise FileNotFoundError(source)
        summary = export_sample(source, destination)
        print(f"Wrote {destination}: {summary['points']:,} points, {summary['cells']:,} quads.")


if __name__ == "__main__":
    main()
