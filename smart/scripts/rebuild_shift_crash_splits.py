"""Rebuild SHIFT-Crash as one reproducible random train/validation split.

The source cases are first merged from every existing partition, then shuffled
once with a recorded seed.  A backup is written before replacing ``splits.json``
so this operation is reversible and never alters case arrays.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument("--train-fraction", type=float, default=0.9)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def atomic_json_dump(path: Path, value):
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def main():
    args = parse_args()
    root = Path(args.data_root).expanduser().resolve()
    split_path = root / "splits.json"
    if not split_path.is_file():
        raise FileNotFoundError(split_path)
    if not 0.5 < args.train_fraction < 1.0:
        raise ValueError("--train-fraction must be strictly between 0.5 and 1.0.")

    previous = json.loads(split_path.read_text(encoding="utf-8"))
    merged = [str(case_id) for case_ids in previous.values() for case_id in case_ids]
    if len(merged) != len(set(merged)):
        raise ValueError("Existing splits are not disjoint; refusing to rebuild from duplicate case IDs.")
    missing = [case_id for case_id in merged if not (root / "cases" / case_id).is_dir()]
    if missing:
        raise FileNotFoundError(f"Missing case directories, first examples: {missing[:5]}")

    case_ids = np.asarray(sorted(merged), dtype=object)
    np.random.default_rng(args.seed).shuffle(case_ids)
    train_count = int(round(float(args.train_fraction) * len(case_ids)))
    train_count = min(max(train_count, 1), len(case_ids) - 1)
    rebuilt = {
        "train": case_ids[:train_count].tolist(),
        "validation": case_ids[train_count:].tolist(),
    }
    print(
        f"Merged {len(case_ids)} cases -> train={len(rebuilt['train'])}, "
        f"validation={len(rebuilt['validation'])}, seed={args.seed}."
    )
    if args.dry_run:
        return

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = root / f"splits_before_random_90_10_{stamp}.json"
    atomic_json_dump(backup, previous)
    atomic_json_dump(split_path, rebuilt)

    # Existing normalization files describe the preceding train split.  Leave
    # the arrays intact for recovery, but remove their provenance so the
    # dataset cannot silently treat them as valid for the new partition.
    provenance_path = root / "normalization_provenance.json"
    if provenance_path.is_file():
        provenance_path.unlink()

    metadata_path = root / "dataset_metadata.json"
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["split_counts"] = {name: len(ids) for name, ids in rebuilt.items()}
        metadata["split_fractions"] = {
            "train": len(rebuilt["train"]) / len(case_ids),
            "validation": len(rebuilt["validation"]) / len(case_ids),
        }
        metadata["split_seed"] = int(args.seed)
        metadata["split_strategy"] = "reproducible merged case-level random 90/10 train/validation split"
        metadata["split_rebuilt_at_utc"] = datetime.now(timezone.utc).isoformat()
        metadata["normalization"]["statistics_split"] = "stale_after_split_rebuild"
        atomic_json_dump(metadata_path, metadata)
    print(f"Wrote {split_path}; backup={backup}")


if __name__ == "__main__":
    main()
