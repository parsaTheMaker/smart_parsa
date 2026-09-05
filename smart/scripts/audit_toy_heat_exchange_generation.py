#!/usr/bin/env python3
"""Summarize persisted FEM, boundary-condition, and mesh-quality diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def describe(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {"min": float(array.min()), "mean": float(array.mean()), "p95": float(np.quantile(array, 0.95)), "max": float(array.max())}


def main() -> int:
    args = parse_args()
    records = []
    for path in sorted(args.data_root.glob("case_*/case_metadata.json")):
        metadata = json.loads(path.read_text(encoding="utf-8"))
        mesh, physics = metadata["mesh"], metadata["physics"]
        records.append({
            "case_id": int(metadata["case_id"]), "split": metadata["split"], "channels": len(metadata["parameters"]["channels"]),
            "nodes": int(mesh["nodes"]), "tetrahedra": int(mesh["tetrahedra"]), "surface_triangles": int(mesh["surface_triangles"]),
            "minimum_tetra_volume": float(mesh["minimum_tetra_volume"]), "surface_area_p95_over_p05": float(mesh["surface_area_p95_over_p05"]),
            "channel_temperature_max_error": float(mesh["channel_temperature_max_error"]), "linear_residual": float(physics["linear_residual"]),
            "nonlinear_relative_change": float(physics["nonlinear_relative_change"]), "nonlinear_iterations": int(physics["nonlinear_iterations"]),
            "inner_faces": int(mesh["inner_faces"]), "outer_faces": int(mesh["outer_faces"]),
        })
    if not records:
        raise FileNotFoundError(f"No case metadata found under {args.data_root}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "heat_exchange_generation_per_case.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader(); writer.writerows(records)
    first = json.loads((args.data_root / f"case_{records[0]['case_id']:05d}" / "case_metadata.json").read_text(encoding="utf-8"))
    summary = {
        "case_count": len(records),
        "splits": {split: sum(row["split"] == split for row in records) for split in sorted({row["split"] for row in records})},
        "governing_equation": first["physics"]["equation"],
        "channel_boundary_condition": first["physics"]["channel_bc"],
        "exterior_boundary_condition": first["physics"]["exterior_bc"],
        "global_physics_parameters": {key: first["physics"][key] for key in ("exterior_biot", "radiation", "tau", "nonlinear_conductivity")},
        "channels_per_case": describe([float(row["channels"]) for row in records]),
        "mesh_nodes": describe([float(row["nodes"]) for row in records]),
        "mesh_tetrahedra": describe([float(row["tetrahedra"]) for row in records]),
        "surface_triangles": describe([float(row["surface_triangles"]) for row in records]),
        "minimum_tetra_volume": describe([row["minimum_tetra_volume"] for row in records]),
        "surface_area_p95_over_p05": describe([row["surface_area_p95_over_p05"] for row in records]),
        "linear_residual": describe([row["linear_residual"] for row in records]),
        "nonlinear_relative_change": describe([row["nonlinear_relative_change"] for row in records]),
        "channel_temperature_max_error": describe([row["channel_temperature_max_error"] for row in records]),
        "nonlinear_iterations": describe([float(row["nonlinear_iterations"]) for row in records]),
    }
    (args.output_dir / "heat_exchange_generation_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Audited {len(records)} persisted heat-exchanger cases in {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
