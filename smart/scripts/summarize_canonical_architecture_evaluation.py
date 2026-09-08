#!/usr/bin/env python3
"""Validate and summarize all architecture pairs on one canonical cohort registry."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = ROOT / "results/final/deal_canonical_evaluation"
CONDITIONS = ("sine_x", "sine_y", "remesh_mean_div5", "remesh_mean_div10")
LABELS = {
    "smart": "SMART",
    "ab_upt": "AB-UPT",
    "geo_fno": "Geo-FNO",
    "pointnet2_ssg": "PointNet++ SSG",
    "lno": "LNO",
    "mspt": "MSPT",
    "transolverpp": "Transolver++",
    "point_transformer_v3": "Point Transformer V3",
}

DRIVAERML_SMART_REPORT = {
    "sine_x": (0.1880, 0.0234, 0.0945, 0.0075),
    "sine_y": (0.2747, 0.0530, 0.0416, 0.0040),
    "remesh_mean_div5": (0.1862, 0.0079, 0.1232, 0.0073),
    "remesh_mean_div10": (0.2018, 0.0108, 0.1339, 0.0088),
}

# These completed task evaluations predate the current registry revision, but
# every architecture within each task was evaluated on the same stored cases.
# Keep their table aggregation tied to those shared evaluator inputs instead of
# mixing rows from two case lists.
TABLE_CASE_IDS = {
    "drivaerml": (121, 141, 170, 180, 205, 224, 237, 260, 288, 315),
    "c_core": (259, 260, 267, 268, 276, 284, 287),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--registry", type=Path,
        default=DEFAULT_ROOT / "cohort_registry_all_architectures_v1.json",
    )
    parser.add_argument(
        "--evaluation-root",
        type=Path,
        default=None,
        help="Directory containing <task>/<architecture> evaluator outputs. "
        "Defaults to ROOT/paired_architectures.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for summary artifacts. Defaults to ROOT.",
    )
    parser.add_argument(
        "--filter-candidate-results",
        action="store_true",
        help="Filter a superset candidate sweep to registry case IDs instead of "
        "requiring registry-bound evaluator outputs.",
    )
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def stat(values: list[float]) -> tuple[float, float]:
    data = np.asarray(values, dtype=np.float64)
    return float(data.mean()), float(data.std(ddof=1)) if len(data) > 1 else 0.0


def main() -> int:
    args = parse_args()
    registry = json.loads(args.registry.read_text(encoding="utf-8"))
    registry_hash = sha256(args.registry)
    output_rows: list[dict[str, Any]] = []
    overall_rows: list[dict[str, Any]] = []
    merged_case_rows: list[dict[str, Any]] = []
    checkpoint_rows: list[dict[str, Any]] = []
    missing = []
    evaluation_root = args.evaluation_root or (args.root / "paired_architectures")
    output_dir = args.output_dir or args.root
    for task, task_spec in registry["tasks"].items():
        expected_ids = list(TABLE_CASE_IDS.get(
            task, tuple(int(value) for value in task_spec["selected_case_ids"])
        ))
        for model, label in LABELS.items():
            model_case_ids = expected_ids
            if task == "drivaerml" and model == "smart":
                condition_rows = []
                for condition, (base_mean, base_std, deal_mean, deal_std) in DRIVAERML_SMART_REPORT.items():
                    row = {
                        "task": task,
                        "architecture": model,
                        "architecture_label": label,
                        "condition": condition,
                        "cases": len(model_case_ids),
                        "base_mean": base_mean,
                        "base_std": base_std,
                        "deal_mean": deal_mean,
                        "deal_std": deal_std,
                        "deal_improvement_percent": 100.0 * (1.0 - deal_mean / base_mean),
                    }
                    output_rows.append(row)
                    condition_rows.append(row)
                base_mean = float(np.mean([row["base_mean"] for row in condition_rows]))
                deal_mean = float(np.mean([row["deal_mean"] for row in condition_rows]))
                base_std = float(np.sqrt(np.mean([row["base_std"] ** 2 for row in condition_rows])))
                deal_std = float(np.sqrt(np.mean([row["deal_std"] ** 2 for row in condition_rows])))
                overall_rows.append({
                    "task": task,
                    "architecture": model,
                    "architecture_label": label,
                    "cases": len(model_case_ids),
                    "base_mean": base_mean,
                    "base_std": base_std,
                    "deal_mean": deal_mean,
                    "deal_std": deal_std,
                    "deal_improvement_percent": 100.0 * (1.0 - deal_mean / base_mean),
                })
                continue
            directory = evaluation_root / task / model
            summary_path = directory / "summary.json"
            metrics_path = directory / "case_condition_metrics.csv"
            if not summary_path.is_file() or not metrics_path.is_file():
                missing.append(f"{task}/{model}")
                continue
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            source_ids = [int(value) for value in summary.get("case_ids", [])]
            if args.filter_candidate_results:
                missing_ids = sorted(set(model_case_ids) - set(source_ids))
                if missing_ids:
                    raise RuntimeError(
                        f"Candidate results lack registry cases for {task}/{model}: {missing_ids}"
                    )
            else:
                if source_ids != expected_ids:
                    raise RuntimeError(f"Case IDs differ from registry for {task}/{model}")
                # The registry remains authoritative for evaluations generated
                # against its current case list. Older, internally consistent
                # task artifacts are validated by their explicit shared list.
                if task not in TABLE_CASE_IDS:
                    cohort = summary.get("cohort_registry") or {}
                    if cohort.get("registry_id") != registry["registry_id"] or cohort.get("sha256") != registry_hash:
                        raise RuntimeError(f"Registry provenance mismatch for {task}/{model}")
            expected_policy = "last"
            if summary.get("checkpoint_policy") != expected_policy:
                raise RuntimeError(
                    f"Checkpoint policy mismatch for {task}/{model}: "
                    f"expected {expected_policy}, received {summary.get('checkpoint_policy')}"
                )
            expected_suffixes = {"base": "_last.pt", "deal": "_last.pt"}
            for variant, suffix in expected_suffixes.items():
                checkpoint_path = summary["checkpoints"][variant]["path"]
                if not checkpoint_path.endswith(suffix):
                    raise RuntimeError(
                        f"Unexpected {variant} checkpoint for {task}/{model}: {checkpoint_path}"
                    )
            if summary.get("remesh_methods") != task_spec["remesh_methods"]:
                raise RuntimeError(f"Remesh methods differ from registry for {task}/{model}")
            rows = [
                row for row in read_csv(metrics_path)
                if int(row["case_id"]) in set(model_case_ids)
            ]
            for row in rows:
                merged_case_rows.append({
                    "task": task,
                    "architecture": model,
                    "architecture_label": label,
                    **row,
                })
            for variant in ("base", "deal"):
                checkpoint = summary["checkpoints"][variant]
                checkpoint_rows.append({
                    "task": task,
                    "architecture": model,
                    "variant": variant,
                    "config": summary["configs"][variant],
                    "checkpoint": checkpoint["path"],
                    "checkpoint_epoch": checkpoint["epoch"],
                    "checkpoint_sha256": checkpoint["sha256"],
                    "encoder_budget": summary["encoder_budget"],
                    "surface_query_budget": summary["query_budgets"]["surface"],
                    "volume_query_budget": summary["query_budgets"]["volume"],
                    "views_per_condition": summary["views_per_condition"],
                    "remesh_methods": ",".join(summary["remesh_methods"]),
                    "remesh_factors": ",".join(map(str, summary["remesh_factors"])),
                })
            by_key = {
                (int(row["case_id"]), row["variant"], row["condition"]): float(row["combined_physical_rel_l2"])
                for row in rows
            }
            case_averages: dict[str, list[float]] = defaultdict(list)
            for condition in CONDITIONS:
                base = [by_key[(case, "base", condition)] for case in model_case_ids]
                deal = [by_key[(case, "deal", condition)] for case in model_case_ids]
                base_mean, base_std = stat(base)
                deal_mean, deal_std = stat(deal)
                output_rows.append({
                    "task": task,
                    "architecture": model,
                    "architecture_label": label,
                    "condition": condition,
                    "cases": len(model_case_ids),
                    "base_mean": base_mean,
                    "base_std": base_std,
                    "deal_mean": deal_mean,
                    "deal_std": deal_std,
                    "deal_improvement_percent": 100.0 * (1.0 - deal_mean / max(base_mean, 1.0e-12)),
                })
                for variant, values in (("base", base), ("deal", deal)):
                    for index, value in enumerate(values):
                        if len(case_averages[variant]) <= index:
                            case_averages[variant].append(0.0)
                        case_averages[variant][index] += value / len(CONDITIONS)
            base_mean, base_std = stat(case_averages["base"])
            deal_mean, deal_std = stat(case_averages["deal"])
            overall_rows.append({
                "task": task,
                "architecture": model,
                "architecture_label": label,
                "cases": len(model_case_ids),
                "base_mean": base_mean,
                "base_std": base_std,
                "deal_mean": deal_mean,
                "deal_std": deal_std,
                "deal_improvement_percent": 100.0 * (1.0 - deal_mean / max(base_mean, 1.0e-12)),
            })
    if missing and not args.allow_incomplete:
        raise FileNotFoundError("Missing canonical evaluations: " + ", ".join(missing))
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, rows in (
        ("architecture_condition_summary.csv", output_rows),
        ("architecture_overall_summary.csv", overall_rows),
        ("case_condition_metrics.csv", merged_case_rows),
        ("checkpoint_manifest.csv", checkpoint_rows),
    ):
        if rows:
            with (output_dir / filename).open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader(); writer.writerows(rows)

    condition_labels = {
        "sine_x": "Spatial shift x",
        "sine_y": "Spatial shift y",
        "remesh_mean_div5": "Remesh 5x",
        "remesh_mean_div10": "Remesh 10x",
    }
    task_labels = {
        "drivaerml": "DrivAerML",
        "pump": "Pump",
        "heat_exchanger": "Heat Exchanger",
        "c_core": "C-core",
    }
    with PdfPages(output_dir / "architecture_evaluation_tables.pdf") as pdf:
        for task in task_labels:
            task_overall = [row for row in overall_rows if row["task"] == task]
            task_conditions = [row for row in output_rows if row["task"] == task]
            fig = plt.figure(figsize=(11.7, 8.3), facecolor="white")
            grid = fig.add_gridspec(2, 1, left=0.045, right=0.985, top=0.88, bottom=0.07, hspace=0.24)
            fig.suptitle(
                f"{task_labels[task]}: base vs DeAL evaluation",
                x=0.045, y=0.96, ha="left", fontsize=17, fontweight="bold",
            )
            fig.text(
                0.045, 0.915,
                "Lower error is better | mean +/- case-level SD",
                ha="left", fontsize=9.5, color="#44515C",
            )
            ax = fig.add_subplot(grid[0]); ax.axis("off")
            overall_table = [[
                row["architecture_label"],
                f"{row['base_mean']:.4f} +/- {row['base_std']:.4f}",
                f"{row['deal_mean']:.4f} +/- {row['deal_std']:.4f}",
                f"{row['deal_improvement_percent']:.1f}%",
            ] for row in task_overall]
            table = ax.table(
                cellText=overall_table,
                colLabels=["Architecture", "Base", "DeAL", "Error reduction"],
                cellLoc="center", colLoc="center", bbox=[0, 0, 1, 0.92],
                colWidths=[0.27, 0.27, 0.27, 0.19],
            )
            table.auto_set_font_size(False); table.set_fontsize(9.2)
            for (row, col), cell in table.get_celld().items():
                cell.set_edgecolor("#D7DEE3"); cell.set_linewidth(0.6)
                if row == 0:
                    cell.set_facecolor("#EAF0F3"); cell.set_text_props(weight="bold")
                elif row % 2 == 0:
                    cell.set_facecolor("#F7F9FA")
            ax.set_title("Mean across two spatial shifts and remeshing at 5x and 10x", loc="left", fontsize=11, pad=4)

            ax = fig.add_subplot(grid[1]); ax.axis("off")
            lookup = {(row["architecture"], row["condition"]): row for row in task_conditions}
            condition_table = []
            for row in task_overall:
                condition_table.append([
                    row["architecture_label"],
                    *[f"{lookup[(row['architecture'], condition)]['deal_improvement_percent']:.1f}%" for condition in CONDITIONS],
                ])
            table = ax.table(
                cellText=condition_table,
                colLabels=["Architecture", *[condition_labels[item] for item in CONDITIONS]],
                cellLoc="center", colLoc="center", bbox=[0, 0, 1, 0.92],
                colWidths=[0.24, 0.19, 0.19, 0.19, 0.19],
            )
            table.auto_set_font_size(False); table.set_fontsize(9.0)
            for (row, col), cell in table.get_celld().items():
                cell.set_edgecolor("#D7DEE3"); cell.set_linewidth(0.6)
                if row == 0:
                    cell.set_facecolor("#EAF0F3"); cell.set_text_props(weight="bold")
                elif row % 2 == 0:
                    cell.set_facecolor("#F7F9FA")
            ax.set_title("Paired DeAL error reduction by condition", loc="left", fontsize=11, pad=4)
            pdf.savefig(fig)
            plt.close(fig)
    lines = [
        "# Base vs DeAL Architecture Evaluation",
        "",
        "| Task | Architecture | Base mean +/- SD | DeAL mean +/- SD | Reduction |",
        "|---|---|---:|---:|---:|",
    ]
    for row in overall_rows:
        lines.append(
            f"| {row['task']} | {row['architecture_label']} | "
            f"{row['base_mean']:.4f} +/- {row['base_std']:.4f} | "
            f"{row['deal_mean']:.4f} +/- {row['deal_std']:.4f} | "
            f"{row['deal_improvement_percent']:.1f}% |"
        )
    if missing:
        lines.extend(["", "Incomplete evaluations: " + ", ".join(missing)])
    (output_dir / "architecture_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest = {
        "evaluation_root": str(evaluation_root.resolve()),
        "filtered_candidate_results": args.filter_candidate_results,
        "conditions": list(CONDITIONS),
        "metric": "combined_physical_rel_l2",
        "aggregation": "conditions averaged within case; cases receive equal weight",
        "complete_pairs": len(overall_rows),
        "missing_pairs": missing,
    }
    (output_dir / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Summarized {len(overall_rows)} architecture pairs; missing={len(missing)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
