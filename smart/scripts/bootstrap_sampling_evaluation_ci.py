#!/usr/bin/env python3
"""Compute paired, case-clustered bootstrap intervals for sampling evaluations.

Rows from the comparison scripts contain repeated stochastic views of the same
case.  This utility first averages those views within each case, then
bootstraps cases (not individual views).  The resulting intervals therefore do
not spuriously treat correlated views as independent test examples.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Comparison CSV (per-view or endpoint metrics).")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-model", required=True, help="Baseline model label in the input CSV.")
    parser.add_argument("--comparison-model", required=True, help="DeAL model label in the input CSV.")
    parser.add_argument("--metric", default="", help="Metric column; inferred when omitted.")
    parser.add_argument("--conditions", default="", help="Optional comma-separated canonical conditions to retain.")
    parser.add_argument(
        "--reference-only-conditions",
        default="",
        help=(
            "Comma-separated conditions reported only for the reference model. "
            "Use this for a nominal Base-only control without evaluating or reporting DeAL."
        ),
    )
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def first_present(fieldnames: set[str], candidates: tuple[str, ...], label: str) -> str:
    for candidate in candidates:
        if candidate in fieldnames:
            return candidate
    raise ValueError(f"Could not infer {label}; expected one of {candidates}, found {sorted(fieldnames)}")


def canonical_condition(row: dict[str, str], condition_column: str) -> str:
    """Map the two comparison schemas onto stable paper-facing condition names."""
    raw = str(row.get(condition_column, "")).strip()
    source = str(row.get("source", "")).strip()
    if raw == "aligned_uniform_wor":
        return "original_uniform"
    if raw == "ood_sine_x_mix_1.00":
        return "sine_x_1"
    if raw == "ood_sine_y_mix_1.00":
        return "sine_y_1"
    if raw.startswith("geometry_"):
        raw = raw.removeprefix("geometry_")
    # Endpoint comparisons store all individual remesh methods under a mean
    # category.  Keeping only the factor here lets us average methods per case.
    if raw.startswith("remeshing_div") or raw.endswith("_div5") or raw.endswith("_div10") or source.endswith("_div5") or source.endswith("_div10"):
        candidate = source or raw
        if "div5" in candidate:
            return "remeshing_div5_mean"
        if "div10" in candidate:
            return "remeshing_div10_mean"
    return raw


def percentile_interval(samples: np.ndarray) -> tuple[float, float]:
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def main() -> int:
    args = parse_args()
    if args.bootstrap_samples < 100:
        raise ValueError("--bootstrap-samples must be at least 100.")
    with args.input.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
        fieldnames = set(handle.seek(0) or next(csv.reader(handle)))
    if not rows:
        raise ValueError(f"No rows in {args.input}")
    model_column = first_present(set(rows[0]), ("model_name", "model"), "model column")
    case_column = first_present(set(rows[0]), ("run_id", "case_id"), "case column")
    view_column = first_present(set(rows[0]), ("view_id", "view"), "view column")
    condition_column = first_present(set(rows[0]), ("sampling_mode", "category"), "condition column")
    metric_column = args.metric or first_present(
        set(rows[0]), ("combined_global_rel_l2", "combined_rel_l2"), "combined error metric"
    )
    wanted = {item.strip() for item in args.conditions.split(",") if item.strip()}
    reference_only = {item.strip() for item in args.reference_only_conditions.split(",") if item.strip()}
    if reference_only - wanted and wanted:
        raise ValueError("--reference-only-conditions must be a subset of --conditions.")

    # First aggregate all source methods and random views per (case, model,
    # condition). This gives one independent observational unit per case.
    grouped: dict[tuple[int, str, str], list[float]] = defaultdict(list)
    for row in rows:
        try:
            value = float(row[metric_column])
            case_id = int(row[case_column])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid metric row: {row}") from exc
        if not np.isfinite(value):
            continue
        condition = canonical_condition(row, condition_column)
        if wanted and condition not in wanted:
            continue
        grouped[(case_id, str(row[model_column]), condition)].append(value)

    per_case: dict[tuple[str, int, str], float] = {
        (model, case_id, condition): float(np.mean(values))
        for (case_id, model, condition), values in grouped.items()
    }
    conditions = sorted({condition for _, _, condition in per_case})
    generator = np.random.default_rng(args.seed)
    output_rows: list[dict[str, object]] = []
    for condition in conditions:
        reference_case_ids = {
            case_id for model, case_id, item in per_case if model == args.reference_model and item == condition
        }
        comparison_case_ids = {
            case_id for model, case_id, item in per_case if model == args.comparison_model and item == condition
        }
        case_ids = sorted(reference_case_ids if condition in reference_only else reference_case_ids & comparison_case_ids)
        if len(case_ids) < 2:
            qualifier = "reference" if condition in reference_only else "paired"
            raise ValueError(f"{condition}: only {len(case_ids)} {qualifier} cases for {args.reference_model} vs {args.comparison_model}.")
        reference = np.asarray([per_case[(args.reference_model, case_id, condition)] for case_id in case_ids], dtype=np.float64)
        draws = generator.integers(0, len(case_ids), size=(args.bootstrap_samples, len(case_ids)))
        reference_draws = reference[draws].mean(axis=1)
        reference_ci = percentile_interval(reference_draws)
        result = {
            "condition": condition,
            "metric": metric_column,
            "reference_model": args.reference_model,
            "comparison_model": args.comparison_model,
            "comparison_available": condition not in reference_only,
            "paired_cases": len(case_ids),
            "reference_mean": float(reference.mean()),
            "reference_std": float(reference.std(ddof=1)),
            "reference_ci95_low": reference_ci[0],
            "reference_ci95_high": reference_ci[1],
            "comparison_mean": float("nan"),
            "comparison_std": float("nan"),
            "comparison_ci95_low": float("nan"),
            "comparison_ci95_high": float("nan"),
            "paired_error_reduction": float("nan"),
            "paired_error_reduction_ci95_low": float("nan"),
            "paired_error_reduction_ci95_high": float("nan"),
            "relative_improvement_percent": float("nan"),
            "relative_improvement_ci95_low": float("nan"),
            "relative_improvement_ci95_high": float("nan"),
            "bootstrap_probability_comparison_better": float("nan"),
            "bootstrap_samples": args.bootstrap_samples,
            "seed": args.seed,
        }
        if condition not in reference_only:
            comparison = np.asarray([per_case[(args.comparison_model, case_id, condition)] for case_id in case_ids], dtype=np.float64)
            comparison_draws = comparison[draws].mean(axis=1)
            difference_draws = reference_draws - comparison_draws
            improvement_draws = 100.0 * difference_draws / np.maximum(np.abs(reference_draws), 1.0e-12)
            comparison_ci = percentile_interval(comparison_draws)
            difference_ci = percentile_interval(difference_draws)
            improvement_ci = percentile_interval(improvement_draws)
            result.update(
                {
                    "comparison_mean": float(comparison.mean()),
                    "comparison_std": float(comparison.std(ddof=1)),
                    "comparison_ci95_low": comparison_ci[0],
                    "comparison_ci95_high": comparison_ci[1],
                    "paired_error_reduction": float((reference - comparison).mean()),
                    "paired_error_reduction_ci95_low": difference_ci[0],
                    "paired_error_reduction_ci95_high": difference_ci[1],
                    "relative_improvement_percent": float(100.0 * (reference.mean() - comparison.mean()) / max(abs(reference.mean()), 1.0e-12)),
                    "relative_improvement_ci95_low": improvement_ci[0],
                    "relative_improvement_ci95_high": improvement_ci[1],
                    "bootstrap_probability_comparison_better": float(np.mean(difference_draws > 0.0)),
                }
            )
        output_rows.append(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"Wrote clustered bootstrap intervals for {len(output_rows)} conditions to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
