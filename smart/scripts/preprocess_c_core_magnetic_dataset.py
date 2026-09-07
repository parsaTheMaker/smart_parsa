#!/usr/bin/env python3
"""Convert native C-core VTK meshes into memory-mappable training arrays."""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pyvista as pv
from tqdm.auto import tqdm


def preprocess_case(case_id: int, source_root: Path, output_root: Path) -> dict[str, object]:
    source_dir = source_root / f"case_{case_id:05d}"
    native = json.loads((source_dir / "native_case_metadata.json").read_text(encoding="utf-8"))
    target_dir = output_root / f"case_{case_id:05d}"
    metadata_path = target_dir / "case_metadata.json"
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if all((target_dir / f"{key}.npy").is_file() for key in (
            "geometry_coords", "surface_coords", "surface_data", "volume_coords", "volume_data"
        )):
            return metadata
    surface = pv.read(source_dir / native["surface_vtp"])
    volume = pv.read(source_dir / native["solid_volume_vtu"])
    surface_coords = np.ascontiguousarray(surface.points, dtype=np.float32)
    volume_coords = np.ascontiguousarray(volume.points, dtype=np.float32)
    surface_data = np.ascontiguousarray(surface.point_data["B_T"], dtype=np.float32)
    volume_data = np.ascontiguousarray(volume.point_data["B_T"], dtype=np.float32)
    if surface_data.shape != surface_coords.shape or volume_data.shape != volume_coords.shape:
        raise RuntimeError(f"case_{case_id:05d}: B_T must be a three-component point field.")
    if not all(np.isfinite(array).all() for array in (surface_coords, volume_coords, surface_data, volume_data)):
        raise RuntimeError(f"case_{case_id:05d}: non-finite coordinate or field value.")
    target_dir.mkdir(parents=True, exist_ok=True)
    np.save(target_dir / "surface_coords.npy", surface_coords, allow_pickle=False)
    geometry_path = target_dir / "geometry_coords.npy"
    if geometry_path.exists():
        geometry_path.unlink()
    os.link(target_dir / "surface_coords.npy", geometry_path)
    np.save(target_dir / "surface_data.npy", surface_data, allow_pickle=False)
    np.save(target_dir / "volume_coords.npy", volume_coords, allow_pickle=False)
    np.save(target_dir / "volume_data.npy", volume_data, allow_pickle=False)
    all_min = np.minimum(surface_coords.min(0), volume_coords.min(0))
    all_max = np.maximum(surface_coords.max(0), volume_coords.max(0))
    metadata = {
        "case_id": case_id,
        "split": native["split"],
        "surface_count": int(len(surface_coords)),
        "volume_count": int(len(volume_coords)),
        "surface_sum": surface_data.astype(np.float64).sum(0).tolist(),
        "surface_sq_sum": np.square(surface_data.astype(np.float64)).sum(0).tolist(),
        "volume_sum": volume_data.astype(np.float64).sum(0).tolist(),
        "volume_sq_sum": np.square(volume_data.astype(np.float64)).sum(0).tolist(),
        "position_min": all_min.astype(float).tolist(),
        "position_max": all_max.astype(float).tolist(),
        "native_surface_vtp": str(source_dir / native["surface_vtp"]),
        "native_volume_vtu": str(source_dir / native["solid_volume_vtu"]),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()
    source_root = args.source_dir.expanduser().resolve()
    output_root = args.output_dir.expanduser().resolve()
    raw_manifest = json.loads((source_root / "raw_manifest.json").read_text(encoding="utf-8"))
    case_ids = [int(value) for value in raw_manifest["train_ids"] + raw_manifest["validation_ids"]]
    output_root.mkdir(parents=True, exist_ok=True)
    records = []
    with ProcessPoolExecutor(max_workers=min(args.workers, len(case_ids))) as executor:
        futures = {
            executor.submit(preprocess_case, case_id, source_root, output_root): case_id
            for case_id in case_ids
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Preprocessing C-core"):
            records.append(future.result())
    records.sort(key=lambda row: int(row["case_id"]))
    manifest = {
        "dataset": "CCoreMagnetic",
        "source_root": str(source_root),
        "train_ids": raw_manifest["train_ids"],
        "validation_ids": raw_manifest["validation_ids"],
        "surface_fields": raw_manifest["surface_fields"],
        "volume_fields": raw_manifest["volume_fields"],
        "cases": records,
    }
    (output_root / "preprocessed_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote model-ready arrays for {len(case_ids)} cases to {output_root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
