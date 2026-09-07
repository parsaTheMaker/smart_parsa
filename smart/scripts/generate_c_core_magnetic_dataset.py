#!/usr/bin/env python3
"""Generate a deterministic C-core geometry-to-magnetic-field dataset."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

try:
    from generate_c_core_shape_examples import (
        VARIANTS,
        CoreVariant,
        generate_variant,
        validate_variant,
    )
except ImportError:  # Support importing as smart.scripts.* in tests.
    from scripts.generate_c_core_shape_examples import (
        VARIANTS,
        CoreVariant,
        generate_variant,
        validate_variant,
    )


def sampled_variant(case_id: int, seed: int) -> CoreVariant:
    """Perturb a validated design family without changing its topology."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, case_id]))
    base = VARIANTS[case_id % len(VARIANTS)]
    for _ in range(256):
        scale_x = float(rng.uniform(0.96, 1.04))
        scale_y = scale_x if base.profile == "circle" else float(rng.uniform(0.96, 1.04))
        window_x = float(np.clip(base.window_ratio_x + rng.uniform(-0.015, 0.015), 0.43, 0.59))
        window_y = float(np.clip(base.window_ratio_y + rng.uniform(-0.015, 0.015), 0.43, 0.65))
        if base.pole_style == "parallel":
            gap = float(np.clip(base.inner_gap * rng.uniform(0.78, 1.22), 0.0045, 0.026))
            inner_gap = tip_gap = gap
        else:
            inner_gap = float(np.clip(base.inner_gap * rng.uniform(0.88, 1.12), 0.020, 0.035))
            tip_gap = float(np.clip(base.tip_gap * rng.uniform(0.80, 1.20), 0.0035, 0.016))
            tip_gap = min(tip_gap, 0.72 * inner_gap)
        candidate = replace(
            base,
            name=f"case_{case_id:05d}",
            center_x=float(base.center_x + rng.uniform(-0.002, 0.002)),
            width=float(base.width * scale_x),
            height=float(base.height * scale_y),
            window_ratio_x=window_x,
            window_ratio_y=window_y,
            inner_gap=inner_gap,
            tip_gap=tip_gap,
            transition_angle_deg=(
                float(rng.uniform(58.0, 74.0)) if base.window_topology == "unified" else 0.0
            ),
        )
        try:
            validate_variant(candidate)
        except ValueError:
            continue
        return candidate
    raise RuntimeError(f"Could not sample a valid variant for case {case_id}.")


def generate_case(
    case_id: int,
    split: str,
    root: Path,
    seed: int,
    gmsh_threads: int,
    max_cells: int,
    mesh_size_scale: float,
) -> dict[str, object]:
    case_dir = root / f"case_{case_id:05d}"
    metadata_path = case_dir / "native_case_metadata.json"
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        expected = [case_dir / metadata["surface_vtp"], case_dir / metadata["solid_volume_vtu"]]
        if all(path.is_file() and path.stat().st_size > 0 for path in expected):
            return metadata
    variant = sampled_variant(case_id, seed)
    record = generate_variant(variant, case_dir, gmsh_threads, max_cells, mesh_size_scale)
    record.update({"case_id": case_id, "split": split, "seed": seed, "variant": asdict(variant)})
    metadata_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


def load_completed_case(case_id: int, root: Path) -> dict[str, object] | None:
    """Return a complete existing case without touching expensive native files."""
    case_dir = root / f"case_{case_id:05d}"
    metadata_path = case_dir / "native_case_metadata.json"
    if not metadata_path.is_file():
        return None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        expected = [case_dir / metadata["surface_vtp"], case_dir / metadata["solid_volume_vtu"]]
    except (json.JSONDecodeError, KeyError):
        return None
    return metadata if all(path.is_file() and path.stat().st_size > 0 for path in expected) else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-cases", type=int, default=256)
    parser.add_argument("--validation-cases", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--gmsh-threads", type=int, default=1)
    parser.add_argument("--max-cells", type=int, default=1_000_000)
    parser.add_argument("--mesh-size-scale", type=float, default=1.0)
    parser.add_argument(
        "--allow-failures",
        action="store_true",
        help="Keep successful cases and write a manifest when isolated CAD cases fail.",
    )
    parser.add_argument(
        "--existing-only",
        action="store_true",
        help="Finalize a manifest from complete existing cases without generating missing cases.",
    )
    args = parser.parse_args()
    if args.train_cases <= 0 or args.validation_cases <= 0:
        raise ValueError("Both train and validation splits must be nonempty.")
    if args.workers <= 0 or args.gmsh_threads <= 0:
        raise ValueError("Worker and Gmsh thread counts must be positive.")
    if args.mesh_size_scale <= 0.0:
        raise ValueError("--mesh-size-scale must be positive.")
    root = args.output_dir.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    total = args.train_cases + args.validation_cases
    jobs = [
        (case_id, "train" if case_id < args.train_cases else "validation")
        for case_id in range(total)
    ]
    records: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    if args.existing_only:
        for case_id, _split in jobs:
            record = load_completed_case(case_id, root)
            if record is not None:
                records.append(record)
            else:
                failures.append({"case_id": case_id, "error": "missing or incomplete"})
    else:
        context = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=min(args.workers, total), mp_context=context) as executor:
            futures = {
                executor.submit(
                    generate_case,
                    case_id,
                    split,
                    root,
                    args.seed,
                    args.gmsh_threads,
                    args.max_cells,
                    args.mesh_size_scale,
                ): case_id
                for case_id, split in jobs
            }
            for future in tqdm(as_completed(futures), total=len(futures), desc="C-core FEM cases"):
                case_id = futures[future]
                try:
                    records.append(future.result())
                except Exception as exc:
                    if not args.allow_failures:
                        raise
                    failures.append({"case_id": case_id, "error": f"{type(exc).__name__}: {exc}"})
                    case_dir = root / f"case_{case_id:05d}"
                    if load_completed_case(case_id, root) is None:
                        shutil.rmtree(case_dir, ignore_errors=True)
    records.sort(key=lambda row: int(row["case_id"]))
    successful_ids = {int(row["case_id"]) for row in records}
    train_ids = [case_id for case_id in range(args.train_cases) if case_id in successful_ids]
    validation_ids = [case_id for case_id in range(args.train_cases, total) if case_id in successful_ids]
    if not train_ids or not validation_ids:
        raise RuntimeError(
            f"Completed cases do not provide nonempty splits: train={len(train_ids)}, "
            f"validation={len(validation_ids)}."
        )
    manifest = {
        "dataset": "CCoreMagnetic",
        "seed": args.seed,
        "requested_train_cases": args.train_cases,
        "requested_validation_cases": args.validation_cases,
        "train_ids": train_ids,
        "validation_ids": validation_ids,
        "failed_cases": sorted(failures, key=lambda row: int(row["case_id"])),
        "surface_fields": ["B_x", "B_y", "B_z"],
        "volume_fields": ["B_x", "B_y", "B_z"],
        "cases": records,
    }
    (root / "raw_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(
        f"Wrote manifest for {len(records)} native FEM cases "
        f"(train={len(train_ids)}, validation={len(validation_ids)}, failed={len(failures)}) to {root}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
