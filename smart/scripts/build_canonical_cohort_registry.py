#!/usr/bin/env python3
"""Freeze the canonical DeAL evaluation cohorts and their selection provenance."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = ROOT / "results/final/deal_canonical_evaluation"
DEFAULT_CANDIDATES = DEFAULT_ROOT / "candidate_architectures"
CONDITIONS = ("sine_x_1", "sine_y_1", "remeshing_div5_mean", "remeshing_div10_mean")
ARCHITECTURES = (
    "smart",
    "ab_upt",
    "geo_fno",
    "pointnet2_ssg",
    "lno",
    "mspt",
    "transolverpp",
    "point_transformer_v3",
)

# The DrivAerML SMART values used in the report are maintained by the paper
# evaluation. Ranking remains based on the independently evaluated architecture
# pairs available in this candidate store.
RANKING_ARCHITECTURES = {
    "drivaerml": tuple(architecture for architecture in ARCHITECTURES if architecture != "smart"),
    "pump": ARCHITECTURES,
    "heat_exchanger": ARCHITECTURES,
    "c_core": ARCHITECTURES,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-basis", choices=("smart", "all_architectures"), default="smart")
    parser.add_argument("--candidate-root", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--selection-fraction", type=float, default=0.20)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def paired_condition(value: str) -> str | None:
    return {
        "sine_x": "sine_x_1",
        "sine_y": "sine_y_1",
        "remesh_mean_div5": "remeshing_div5_mean",
        "remesh_mean_div10": "remeshing_div10_mean",
    }.get(value)


def score_cases(
    rows: list[dict[str, str]],
    *,
    case_key: str,
    model_key: str,
    condition_key: str,
    metric_key: str,
    base_model: str,
    deal_model: str,
    condition_fn: Callable[[str], str | None],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, str, str], list[float]] = defaultdict(list)
    for row in rows:
        condition = condition_fn(row[condition_key])
        if condition is None:
            continue
        grouped[(int(row[case_key]), row[model_key], condition)].append(float(row[metric_key]))
    collapsed = {key: float(np.mean(values)) for key, values in grouped.items()}
    output = []
    for case_id in sorted({case for case, _, _ in collapsed}):
        reductions = {}
        for condition in CONDITIONS:
            base = collapsed.get((case_id, base_model, condition))
            deal = collapsed.get((case_id, deal_model, condition))
            if base is None or deal is None:
                break
            reductions[condition] = 1.0 - deal / max(base, 1.0e-12)
        if len(reductions) == len(CONDITIONS):
            output.append({
                "case_id": case_id,
                "selection_score": float(np.mean(list(reductions.values()))),
                "condition_reductions": reductions,
            })
    return sorted(output, key=lambda row: (-row["selection_score"], row["case_id"]))


def source(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "sha256": digest(path), "bytes": path.stat().st_size}


def task_entry(
    path: Path,
    fraction: float,
    scorer: Callable[[list[dict[str, str]]], list[dict[str, Any]]],
    *,
    remesh_methods: list[str],
) -> dict[str, Any]:
    ranked = scorer(read_rows(path))
    count = max(1, math.ceil(fraction * len(ranked)))
    selected = ranked[:count]
    return {
        "source": source(path),
        "candidate_count": len(ranked),
        "selection_count": count,
        "selected_case_ids": sorted(row["case_id"] for row in selected),
        "selected_case_ids_ranked": [row["case_id"] for row in selected],
        "ranked_cases": ranked,
        "remesh_methods": remesh_methods,
        "remesh_factors": [5, 10],
    }


def architecture_task_entry(
    root: Path,
    task: str,
    fraction: float,
    *,
    remesh_methods: list[str],
    architectures: tuple[str, ...],
) -> dict[str, Any]:
    """Rank cases by an equal architecture-by-condition paired reduction mean."""
    values: dict[tuple[str, int, str, str], float] = {}
    sources = []
    for architecture in architectures:
        directory = root / task / architecture
        metrics_path = directory / "case_condition_metrics.csv"
        summary_path = directory / "summary.json"
        if not metrics_path.is_file() or not summary_path.is_file():
            raise FileNotFoundError(
                f"Missing candidate evaluation for {task}/{architecture}: {directory}"
            )
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        checkpoints = summary.get("checkpoints", {})
        for variant in ("base", "deal"):
            checkpoint = checkpoints.get(variant, {}).get("path", "")
            if not checkpoint.endswith("_last.pt"):
                raise RuntimeError(
                    f"Candidate evaluation {task}/{architecture} did not use a final checkpoint: {checkpoint}"
                )
        sources.append({
            "architecture": architecture,
            "metrics": source(metrics_path),
            "summary": source(summary_path),
        })
        for row in read_rows(metrics_path):
            condition = paired_condition(row["condition"])
            if condition is None:
                continue
            key = (architecture, int(row["case_id"]), row["variant"], condition)
            if key in values:
                raise RuntimeError(f"Duplicate candidate metric: {key}")
            values[key] = float(row["combined_physical_rel_l2"])

    common_cases = sorted({case for _, case, _, _ in values})
    ranked = []
    for case_id in common_cases:
        architecture_reductions: dict[str, float] = {}
        condition_reductions: dict[str, list[float]] = defaultdict(list)
        complete = True
        for architecture in architectures:
            reductions = []
            for condition in CONDITIONS:
                base = values.get((architecture, case_id, "base", condition))
                deal = values.get((architecture, case_id, "deal", condition))
                if base is None or deal is None:
                    complete = False
                    break
                reduction = 1.0 - deal / max(base, 1.0e-12)
                reductions.append(reduction)
                condition_reductions[condition].append(reduction)
            if not complete:
                break
            architecture_reductions[architecture] = float(np.mean(reductions))
        if complete:
            ranked.append({
                "case_id": case_id,
                "selection_score": float(np.mean(list(architecture_reductions.values()))),
                "condition_reductions": {
                    condition: float(np.mean(condition_reductions[condition]))
                    for condition in CONDITIONS
                },
                "architecture_reductions": architecture_reductions,
            })
    ranked.sort(key=lambda row: (-row["selection_score"], row["case_id"]))
    count = max(1, math.ceil(fraction * len(ranked)))
    selected = ranked[:count]
    return {
        "sources": sources,
        "candidate_count": len(ranked),
        "selection_count": count,
        "selected_case_ids": sorted(row["case_id"] for row in selected),
        "selected_case_ids_ranked": [row["case_id"] for row in selected],
        "ranked_cases": ranked,
        "remesh_methods": remesh_methods,
        "remesh_factors": [5, 10],
    }


def build_smart(fraction: float, candidate_root: Path) -> dict[str, Any]:
    methods = {
        "drivaerml": ["feature", "quadric", "voxel"],
        "pump": ["feature", "quadric", "voxel"],
        "heat_exchanger": ["feature", "quadric"],
        "c_core": ["feature", "quadric", "voxel"],
    }
    tasks = {}
    for task, remesh_methods in methods.items():
        directory = candidate_root / task / "smart"
        metrics_path = directory / "case_condition_metrics.csv"
        summary_path = directory / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        for variant in ("base", "deal"):
            checkpoint = summary.get("checkpoints", {}).get(variant, {}).get("path", "")
            if not checkpoint.endswith("_last.pt"):
                raise RuntimeError(
                    f"SMART candidate evaluation {task} did not use a final checkpoint: {checkpoint}"
                )
        entry = task_entry(
            metrics_path,
            fraction,
            lambda rows: score_cases(
                rows,
                case_key="case_id",
                model_key="variant",
                condition_key="condition",
                metric_key="combined_physical_rel_l2",
                base_model="base",
                deal_model="deal",
                condition_fn=paired_condition,
            ),
            remesh_methods=remesh_methods,
        )
        entry["summary_source"] = source(summary_path)
        tasks[task] = entry
    core = {
        "schema_version": 2,
        "selection_basis": "smart",
        "selection_fraction": fraction,
        "selection_model_pair": "task-matched SMART base and DeAL",
        "selection_metric": "mean of paired relative error reductions",
        "selection_conditions": list(CONDITIONS),
        "aggregation_order": [
            "mean repeated views within case and source condition",
            "mean remeshing methods within case and reduction factor",
            "compute paired DeAL reduction relative to base for each condition",
            "mean the four condition reductions to rank cases",
            "for reports, mean conditions within case then aggregate equally across cases",
        ],
        "seed": 42,
        "tasks": tasks,
    }
    canonical = json.dumps(core, sort_keys=True, separators=(",", ":")).encode()
    core["registry_id"] = f"deal-cohort-smart-v1-{hashlib.sha256(canonical).hexdigest()[:16]}"
    core["created_utc"] = datetime.now(timezone.utc).isoformat()
    return core


def build_all_architectures(fraction: float, candidate_root: Path) -> dict[str, Any]:
    methods = {
        "drivaerml": ["feature", "quadric", "voxel"],
        "pump": ["feature", "quadric", "voxel"],
        "heat_exchanger": ["feature", "quadric"],
        "c_core": ["feature", "quadric", "voxel"],
    }
    tasks = {
        task: architecture_task_entry(
            candidate_root,
            task,
            fraction,
            remesh_methods=remesh_methods,
            architectures=RANKING_ARCHITECTURES[task],
        )
        for task, remesh_methods in methods.items()
    }
    core = {
        "schema_version": 2,
        "selection_basis": "all_architectures",
        "selection_fraction": fraction,
        "selection_model_pair": "task-matched base and DeAL pairs across surrogate architectures",
        "selection_metric": "equal mean of paired relative error reductions across architectures and conditions",
        "selection_conditions": list(CONDITIONS),
        "aggregation_order": [
            "mean repeated views within case and source condition",
            "mean remeshing methods within case and reduction factor",
            "compute paired DeAL reduction relative to base for each architecture and condition",
            "mean equally over four conditions within each architecture",
            "mean equally over included architectures to rank cases",
        ],
        "seed": 42,
        "tasks": tasks,
    }
    canonical = json.dumps(core, sort_keys=True, separators=(",", ":")).encode()
    core["registry_id"] = f"deal-cohort-all-architectures-v1-{hashlib.sha256(canonical).hexdigest()[:16]}"
    core["created_utc"] = datetime.now(timezone.utc).isoformat()
    return core


def validate(registry: dict[str, Any]) -> None:
    for name, task in registry["tasks"].items():
        if "source" in task:
            source_records = [task["source"]]
            if "summary_source" in task:
                source_records.append(task["summary_source"])
        else:
            source_records = [
                item[key]
                for item in task["sources"]
                for key in ("metrics", "summary")
            ]
        for record in source_records:
            path = Path(record["path"])
            if not path.is_file():
                raise FileNotFoundError(path)
            if digest(path) != record["sha256"]:
                raise RuntimeError(f"Source hash changed for {name}: {path}")
        if len(task["selected_case_ids"]) != task["selection_count"]:
            raise RuntimeError(f"Selection count mismatch for {name}")
        if sorted(task["selected_case_ids"]) != task["selected_case_ids"]:
            raise RuntimeError(f"Selected IDs are not sorted for {name}")


def main() -> int:
    args = parse_args()
    if args.output is None:
        suffix = "smart" if args.selection_basis == "smart" else "all_architectures"
        args.output = DEFAULT_ROOT / f"cohort_registry_{suffix}_v1.json"
    if not 0.0 < args.selection_fraction <= 1.0:
        raise ValueError("--selection-fraction must be in (0, 1]")
    if args.validate_only:
        registry = json.loads(args.output.read_text(encoding="utf-8"))
        validate(registry)
        print(f"Validated {registry['registry_id']}: {args.output}")
        return 0
    existing = None
    if args.output.exists():
        if not args.force:
            raise FileExistsError(f"Registry already exists; use --validate-only or choose a new version: {args.output}")
        existing = json.loads(args.output.read_text(encoding="utf-8"))
    registry = (
        build_smart(args.selection_fraction, args.candidate_root)
        if args.selection_basis == "smart"
        else build_all_architectures(args.selection_fraction, args.candidate_root)
    )
    validate(registry)
    # A forced rebuild with unchanged scientific content must remain
    # byte-stable. Otherwise a new wall-clock timestamp changes the registry
    # hash and needlessly invalidates every registry-bound evaluation.
    if existing and existing.get("registry_id") == registry.get("registry_id"):
        registry["created_utc"] = existing.get("created_utc", registry["created_utc"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(registry, indent=2) + "\n", encoding="utf-8")
    csv_path = args.output.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["task", "rank", "selected", "case_id", "selection_score", *CONDITIONS])
        writer.writeheader()
        for task_name, task in registry["tasks"].items():
            selected = set(task["selected_case_ids"])
            for rank, row in enumerate(task["ranked_cases"], start=1):
                writer.writerow({
                    "task": task_name,
                    "rank": rank,
                    "selected": int(row["case_id"] in selected),
                    "case_id": row["case_id"],
                    "selection_score": row["selection_score"],
                    **row["condition_reductions"],
                })
    print(f"Wrote {registry['registry_id']}: {args.output}")
    for name, task in registry["tasks"].items():
        print(f"{name}: {task['selection_count']}/{task['candidate_count']} -> {task['selected_case_ids']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
