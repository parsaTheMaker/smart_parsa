#!/usr/bin/env python3
"""Compare SMART-ranked and cross-architecture-ranked DeAL cohorts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = ROOT / "results/final/deal_canonical_evaluation"
TASK_LABELS = {
    "drivaerml": "DrivAerML",
    "pump": "Pump",
    "heat_exchanger": "Heat Exchanger",
    "c_core": "C-core",
}
ARCHITECTURES = {
    "smart": "SMART",
    "ab_upt": "AB-UPT",
    "geo_fno": "Geo-FNO",
    "pointnet2_ssg": "PointNet++ SSG",
    "lno": "LNO",
    "mspt": "MSPT",
    "transolverpp": "Transolver++",
    "point_transformer_v3": "Point Transformer V3",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smart-registry", type=Path, default=DEFAULT_ROOT / "cohort_registry_smart_v1.json")
    parser.add_argument(
        "--all-architectures-registry",
        type=Path,
        default=DEFAULT_ROOT / "cohort_registry_all_architectures_v1.json",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_ROOT / "cohort_comparison")
    parser.add_argument(
        "--smart-study",
        type=Path,
        default=DEFAULT_ROOT / "cohort_studies/smart_ranked/architecture_overall_summary.csv",
    )
    parser.add_argument(
        "--all-architectures-study",
        type=Path,
        default=DEFAULT_ROOT / "cohort_studies/all_architectures_ranked/architecture_overall_summary.csv",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    args = parse_args()
    smart = json.loads(args.smart_registry.read_text(encoding="utf-8"))
    all_arch = json.loads(args.all_architectures_registry.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    summary = []
    for task in TASK_LABELS:
        smart_task = smart["tasks"][task]
        all_task = all_arch["tasks"][task]
        smart_rank = {int(row["case_id"]): rank for rank, row in enumerate(smart_task["ranked_cases"], 1)}
        all_rank = {int(row["case_id"]): rank for rank, row in enumerate(all_task["ranked_cases"], 1)}
        smart_score = {int(row["case_id"]): float(row["selection_score"]) for row in smart_task["ranked_cases"]}
        all_score = {int(row["case_id"]): float(row["selection_score"]) for row in all_task["ranked_cases"]}
        smart_selected = set(map(int, smart_task["selected_case_ids"]))
        all_selected = set(map(int, all_task["selected_case_ids"]))
        union = smart_selected | all_selected
        overlap = smart_selected & all_selected
        summary.append({
            "task": task,
            "smart_selected": len(smart_selected),
            "all_architectures_selected": len(all_selected),
            "overlap": len(overlap),
            "jaccard": len(overlap) / max(len(union), 1),
        })
        for case_id in sorted(set(smart_rank) & set(all_rank)):
            rows.append({
                "task": task,
                "case_id": case_id,
                "smart_rank": smart_rank[case_id],
                "all_architectures_rank": all_rank[case_id],
                "smart_score": smart_score[case_id],
                "all_architectures_score": all_score[case_id],
                "selected_by_smart": int(case_id in smart_selected),
                "selected_by_all_architectures": int(case_id in all_selected),
                "selected_by_both": int(case_id in overlap),
            })

    for name, records in (("cohort_rank_comparison.csv", rows), ("cohort_overlap_summary.csv", summary)):
        with (args.output_dir / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.5))
    for ax, (task, label) in zip(axes.flat, TASK_LABELS.items()):
        selected_rows = [row for row in rows if row["task"] == task]
        colors = []
        sizes = []
        for row in selected_rows:
            if row["selected_by_both"]:
                colors.append("#D1495B")
                sizes.append(34)
            elif row["selected_by_smart"]:
                colors.append("#4C78A8")
                sizes.append(30)
            elif row["selected_by_all_architectures"]:
                colors.append("#E07A3F")
                sizes.append(30)
            else:
                colors.append("#B8C2C9")
                sizes.append(18)
        ax.scatter(
            [row["smart_rank"] for row in selected_rows],
            [row["all_architectures_rank"] for row in selected_rows],
            c=colors, s=sizes, alpha=0.88, linewidths=0,
        )
        limit = max(max(row["smart_rank"] for row in selected_rows), max(row["all_architectures_rank"] for row in selected_rows))
        ax.plot([1, limit], [1, limit], color="#64727D", linewidth=1, linestyle="--")
        ax.set_title(label, fontweight="bold")
        ax.set_xlabel("Rank from SMART pair")
        ax.set_ylabel("Rank across all architectures")
        ax.grid(color="#DDE3E7", linewidth=0.6, alpha=0.7)
    fig.legend(
        handles=[
            Line2D([], [], marker="o", linestyle="", color="#D1495B", label="Selected by both"),
            Line2D([], [], marker="o", linestyle="", color="#4C78A8", label="SMART only"),
            Line2D([], [], marker="o", linestyle="", color="#E07A3F", label="All architectures only"),
            Line2D([], [], marker="o", linestyle="", color="#B8C2C9", label="Neither"),
        ],
        loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.992),
    )
    fig.subplots_adjust(left=0.09, right=0.985, bottom=0.09, top=0.90, wspace=0.20, hspace=0.34)
    fig.savefig(args.output_dir / "cohort_rank_comparison.pdf", bbox_inches="tight")
    fig.savefig(args.output_dir / "cohort_rank_comparison.png", dpi=240, bbox_inches="tight")
    plt.close(fig)

    outcome_rows = []
    if args.smart_study.is_file() and args.all_architectures_study.is_file():
        smart_study = {
            (row["task"], row["architecture"]): row for row in read_csv(args.smart_study)
        }
        all_study = {
            (row["task"], row["architecture"]): row
            for row in read_csv(args.all_architectures_study)
        }
        if set(smart_study) != set(all_study):
            raise RuntimeError("The two cohort studies do not contain the same task/architecture pairs.")
        ordered_keys = [
            (task, architecture)
            for task in TASK_LABELS
            for architecture in ARCHITECTURES
        ]
        for task, architecture in ordered_keys:
            key = (task, architecture)
            if key not in smart_study:
                continue
            smart_row, all_row = smart_study[key], all_study[key]
            outcome_rows.append({
                "task": task,
                "architecture": architecture,
                "architecture_label": smart_row["architecture_label"],
                "smart_ranked_base_mean": smart_row["base_mean"],
                "smart_ranked_deal_mean": smart_row["deal_mean"],
                "smart_ranked_reduction_percent": smart_row["deal_improvement_percent"],
                "all_architectures_ranked_base_mean": all_row["base_mean"],
                "all_architectures_ranked_deal_mean": all_row["deal_mean"],
                "all_architectures_ranked_reduction_percent": all_row["deal_improvement_percent"],
                "reduction_difference_percentage_points": (
                    float(all_row["deal_improvement_percent"])
                    - float(smart_row["deal_improvement_percent"])
                ),
            })
        with (args.output_dir / "cohort_outcome_comparison.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(outcome_rows[0]))
            writer.writeheader()
            writer.writerows(outcome_rows)

        fig, axes = plt.subplots(2, 2, figsize=(12.2, 7.7))
        for ax, (task, label) in zip(axes.flat, TASK_LABELS.items()):
            task_rows = [row for row in outcome_rows if row["task"] == task]
            x = np.arange(len(task_rows), dtype=np.float64)
            width = 0.34
            ax.bar(
                x - width / 2,
                [float(row["smart_ranked_reduction_percent"]) for row in task_rows],
                width,
                label="SMART-ranked cohort",
                color="#4C78A8",
            )
            ax.bar(
                x + width / 2,
                [float(row["all_architectures_ranked_reduction_percent"]) for row in task_rows],
                width,
                label="All-architecture-ranked cohort",
                color="#E07A3F",
                hatch="///",
                edgecolor="#8B4726",
                linewidth=0.7,
            )
            ax.axhline(0.0, color="#39434A", linewidth=0.8)
            ax.set_title(label, fontweight="bold")
            ax.set_ylabel("DeAL error reduction (%)")
            ax.set_xticks(x)
            ax.set_xticklabels(
                [row["architecture_label"] for row in task_rows],
                rotation=24,
                ha="right",
                fontsize=8,
            )
            ax.grid(axis="y", color="#DDE3E7", linewidth=0.6, alpha=0.8)
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.995))
        fig.subplots_adjust(left=0.07, right=0.99, bottom=0.11, top=0.91, wspace=0.16, hspace=0.42)
        fig.savefig(args.output_dir / "cohort_outcome_comparison.pdf", bbox_inches="tight")
        fig.savefig(args.output_dir / "cohort_outcome_comparison.png", dpi=240, bbox_inches="tight")
        plt.close(fig)

    manifest = {
        "smart_registry": str(args.smart_registry.resolve()),
        "smart_registry_id": smart["registry_id"],
        "all_architectures_registry": str(args.all_architectures_registry.resolve()),
        "all_architectures_registry_id": all_arch["registry_id"],
        "smart_study": str(args.smart_study.resolve()),
        "all_architectures_study": str(args.all_architectures_study.resolve()),
        "summary": summary,
    }
    (args.output_dir / "comparison_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    for row in summary:
        print(
            f"{row['task']}: overlap {row['overlap']}/{row['smart_selected']} "
            f"(Jaccard={row['jaccard']:.3f})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
