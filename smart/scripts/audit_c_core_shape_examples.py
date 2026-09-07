#!/usr/bin/env python3
"""Audit C-core geometry, mesh, fields, and optional two-grid convergence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pyvista as pv


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def convergence(fine: pv.UnstructuredGrid, coarse: pv.UnstructuredGrid) -> dict[str, float]:
    queries = fine.cell_centers()
    fine_values = queries.sample(
        fine, pass_point_data=False, pass_cell_data=False, tolerance=1.0e-7
    )
    coarse_values = queries.sample(
        coarse, pass_point_data=False, pass_cell_data=False, tolerance=1.0e-7
    )
    valid = np.asarray(fine_values["vtkValidPointMask"], dtype=bool) & np.asarray(
        coarse_values["vtkValidPointMask"], dtype=bool
    )
    if not np.all(valid):
        raise RuntimeError(f"Only {valid.mean():.2%} of interior convergence queries are valid.")
    fine_b = np.asarray(fine_values["B_T"], dtype=np.float64)
    coarse_b = np.asarray(coarse_values["B_T"], dtype=np.float64)
    vector_error = np.linalg.norm(coarse_b - fine_b) / np.linalg.norm(fine_b)
    fine_magnitude = np.linalg.norm(fine_b, axis=1)
    coarse_magnitude = np.linalg.norm(coarse_b, axis=1)
    magnitude_error = np.linalg.norm(coarse_magnitude - fine_magnitude) / np.linalg.norm(fine_magnitude)
    return {
        "valid_query_fraction": float(valid.mean()),
        "vector_relative_l2": float(vector_error),
        "magnitude_relative_l2": float(magnitude_error),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--coarse-dir", type=Path)
    args = parser.parse_args()

    summary_path = args.input_dir / "c_core_shape_examples_summary.json"
    summary = json.loads(summary_path.read_text())
    reports = []
    convergence_reports = []
    for record in summary["problems"]:
        name = record["name"]
        surface_path = args.input_dir / f"{name}_solid_surface.vtp"
        volume_path = args.input_dir / f"{name}_solid_volume.vtu"
        surface = pv.read(surface_path)
        volume = pv.read(volume_path)
        if surface.point_data.keys() != ["B_T"] or volume.point_data.keys() != ["B_T"]:
            raise RuntimeError(f"{name}: final point fields must contain only B_T.")
        if surface.cell_data.keys() or volume.cell_data.keys():
            raise RuntimeError(f"{name}: final meshes contain unexpected cell metadata.")
        if not surface.is_all_triangles or set(volume.cells_dict) != {pv.CellType.TETRA}:
            raise RuntimeError(f"{name}: unexpected element type in final mesh.")
        for mesh in (surface, volume):
            values = np.asarray(mesh.point_data["B_T"])
            if values.ndim != 2 or values.shape[1] != 3 or not np.isfinite(values).all():
                raise RuntimeError(f"{name}: B_T must be a finite three-component vector.")
        volume_b = np.asarray(volume.point_data["B_T"], dtype=np.float64)
        rms_components = np.sqrt(np.mean(volume_b**2, axis=0))
        rms_magnitude = np.sqrt(np.mean(np.sum(volume_b**2, axis=1)))
        bz_rms_fraction = float(rms_components[2] / max(rms_magnitude, 1.0e-14))
        if not 0.02 <= bz_rms_fraction <= 0.35:
            raise RuntimeError(f"{name}: RMS Bz fraction {bz_rms_fraction:.3%} is outside [2%, 35%].")
        reports.append(
            {
                "name": name,
                "surface_points": int(surface.n_points),
                "surface_triangles": int(surface.n_cells),
                "volume_points": int(volume.n_points),
                "volume_tetrahedra": int(volume.n_cells),
                "B_rms_components_T": rms_components.tolist(),
                "Bz_rms_fraction": bz_rms_fraction,
                "surface_sha256": file_digest(surface_path),
                "volume_sha256": file_digest(volume_path),
            }
        )
        if args.coarse_dir is not None:
            coarse_path = args.coarse_dir / volume_path.name
            if coarse_path.exists():
                convergence_reports.append(
                    {"name": name, **convergence(volume, pv.read(coarse_path))}
                )

    report = {
        "status": "passed",
        "case_count": len(reports),
        "field_contract": "B_T only; three Cartesian components in tesla",
        "files": reports,
        "two_grid_convergence": convergence_reports,
    }
    output = args.input_dir / "c_core_validation_report.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Validated {len(reports)} cases; wrote {output}")


if __name__ == "__main__":
    main()
