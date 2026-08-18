#!/usr/bin/env python3
"""Export analytic Toy-SATLOSS validation surfaces as triangular VTP meshes.

The VTPs are geometry-only sources for the same angle, isotropic, and voxel
remeshing protocols used in the sampling-invariance studies.  They are not
used as supervision; surface and volume query targets remain the independent
reference clouds stored by the toy dataset.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from visualize_toy_satloss_examples import (
    latitude_longitude_mesh,
    load_params,
    make_polydata,
    save_surface_mesh_png,
    write_vtp,
)


def parse_ids(value: str) -> list[int]:
    return [int(item.strip()) for item in str(value).split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("/mnt/ssdraid/parsa/toy_satloss_poisson_benchmark_v2"))
    parser.add_argument("--output-dir", type=Path, default=Path("/mnt/ssdraid/parsa/toy_satloss_surface_vtp"))
    parser.add_argument("--results-dir", type=Path, default=Path("/home/parsa/smart_parsa/results/toy_satloss_remeshing"))
    parser.add_argument("--case-ids", default="", help="Optional comma-separated validation case IDs.")
    parser.add_argument("--max-cases", type=int, default=0, help="Export the first this-many validation cases; 0 means all.")
    parser.add_argument("--theta-count", type=int, default=256)
    parser.add_argument("--phi-count", type=int, default=512)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.theta_count < 8 or args.phi_count < 16:
        raise ValueError("--theta-count must be >=8 and --phi-count must be >=16.")

    root = args.data_root.expanduser().resolve()
    manifest = json.loads((root / "preprocessed_manifest.json").read_text(encoding="utf-8"))
    validation_ids = [int(case_id) for case_id in manifest["validation_ids"]]
    case_ids = parse_ids(args.case_ids) if args.case_ids else validation_ids
    if args.max_cases > 0 and not args.case_ids:
        if args.max_cases > len(validation_ids):
            raise ValueError(f"Requested {args.max_cases} cases, but only {len(validation_ids)} validation cases exist.")
        case_ids = validation_ids[:args.max_cases]
    invalid = sorted(set(case_ids) - set(validation_ids))
    if invalid:
        raise ValueError(f"Only validation cases may be exported; invalid IDs: {invalid[:8]}")

    output_dir = args.output_dir.expanduser().resolve()
    results_dir = args.results_dir.expanduser().resolve()
    records = []
    for case_id in tqdm(case_ids, desc="Exporting toy surface VTPs"):
        output_path = output_dir / f"case_{case_id:05d}" / f"toy_case_{case_id:05d}_surface.vtp"
        if output_path.is_file() and not args.overwrite:
            records.append({"case_id": case_id, "path": str(output_path), "status": "existing"})
            continue
        params = load_params(root / f"case_{case_id:05d}")
        points, faces, field, _ = latitude_longitude_mesh(params, args.theta_count, args.phi_count)
        write_vtp(output_path, make_polydata(points, faces, {"manufactured_surface": field}))
        records.append({"case_id": case_id, "path": str(output_path), "points": int(points.shape[0]), "triangles": int(faces.shape[0]), "status": "written"})

        # One mesh rendering is sufficient to verify topology and analytic field.
        if case_id == case_ids[0]:
            results_dir.mkdir(parents=True, exist_ok=True)
            save_surface_mesh_png(case_id, points, faces, field, results_dir / f"toy_case_{case_id:05d}_original_surface.png")
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "toy_surface_vtp_manifest.json").write_text(
        json.dumps({"case_ids": case_ids, "theta_count": args.theta_count, "phi_count": args.phi_count, "records": records}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Exported {len(case_ids)} VTP meshes to {output_dir}")


if __name__ == "__main__":
    main()
