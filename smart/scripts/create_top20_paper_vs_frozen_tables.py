#!/usr/bin/env python3
"""Create a diagnostic PDF comparing paper cohorts with frozen evidence.

The frozen cohort is outcome-conditioned and is therefore labelled as a
diagnostic rather than a replacement for all-case data.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SHIFT_CONDITIONS = ("sine_x_1", "sine_y_1", "remeshing_div5_mean", "remeshing_div10_mean")
DRIV_PAIRS = (
    ("SMART", "SMART_SATLOSS7", "SMART"),
    ("TRANSOLVERPP", "TRANSOLVERPP_SATLOSS7", "Transolver++"),
    ("POINTNET2_SSG", "POINTNET2_SSG_SATLOSS7", "PointNet++ SSG"),
    ("LNO", "LNO_SATLOSS7", "LNO"),
    ("MSPT", "MSPT_SATLOSS7", "MSPT"),
    ("POINT_TRANSFORMER_V3", "POINT_TRANSFORMER_V3_SATLOSS7", "Point Transformer V3"),
)
STRATEGIES = (
    ("base", "Base"),
    ("downsample", "Uniform downsampling"),
    ("gaussian_ball_masked", "Gaussian-ball mask"),
    ("box_masked", "Box mask"),
    ("satloss", "DeAL"),
)
DRIV_STRATEGIES = (
    ("SMART", "Base"),
    ("SMART_DOWNSAMPLE", "Uniform downsampling"),
    ("SMART_GAUSSIAN_BALL_MASKED", "Gaussian-ball mask"),
    ("SMART_BOX_MASKED", "Box mask"),
    ("SMART_SATLOSS7", "DeAL"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/final/reviewer_evidence_20260901/paper_vs_frozen_top20_diagnostic.pdf"),
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def driv_condition(raw: str) -> str | None:
    if raw == "ood_sine_x_mix_1.00":
        return "sine_x_1"
    if raw == "ood_sine_y_mix_1.00":
        return "sine_y_1"
    if raw.endswith("_div5"):
        return "remeshing_div5_mean"
    if raw.endswith("_div10"):
        return "remeshing_div10_mean"
    return None


def summarize(values: list[float]) -> dict[str, float | int]:
    numbers = np.asarray(values, dtype=np.float64)
    if numbers.size == 0:
        raise ValueError("Cannot summarize an empty collection.")
    return {
        "mean": float(numbers.mean()),
        "std": float(numbers.std(ddof=1)) if numbers.size > 1 else 0.0,
        "count": int(numbers.size),
    }


def driv_per_case(rows: list[dict[str, str]], selected_cases: set[int] | None = None) -> dict[tuple[int, str, str], float]:
    grouped: dict[tuple[int, str, str], list[float]] = defaultdict(list)
    for row in rows:
        condition = driv_condition(row["sampling_mode"])
        if condition is None:
            continue
        run_id = int(row["run_id"])
        if selected_cases is not None and run_id not in selected_cases:
            continue
        grouped[(run_id, row["model_name"], condition)].append(float(row["combined_global_rel_l2"]))
    return {key: float(np.mean(value)) for key, value in grouped.items()}


def aggregate_driv_rows(rows: list[dict[str, str]], selected_cases: set[int] | None = None) -> dict[tuple[str, str], dict[str, float | int]]:
    values: dict[tuple[str, str], list[float]] = defaultdict(list)
    for (_, model, condition), value in driv_per_case(rows, selected_cases).items():
        values[(model, condition)].append(value)
    return {key: summarize(value) for key, value in values.items()}


def select_driv_top20(rows: list[dict[str, str]]) -> set[int]:
    # Mean each stochastic view and remesher per case before scoring a case.
    per_case = driv_per_case(rows)
    case_scores = []
    cases = sorted({case for case, _, _ in per_case})
    for case in cases:
        gains = []
        for condition in SHIFT_CONDITIONS:
            base = per_case.get((case, "SMART", condition))
            deal = per_case.get((case, "SMART_SATLOSS7", condition))
            if base is None or deal is None:
                break
            gains.append(1.0 - deal / max(base, 1.0e-12))
        if len(gains) == len(SHIFT_CONDITIONS):
            case_scores.append((float(np.mean(gains)), case))
    keep = max(1, int(np.ceil(0.20 * len(case_scores))))
    return {case for _, case in sorted(case_scores, reverse=True)[:keep]}


def endpoint_per_case(rows: list[dict[str, str]], selected_cases: set[int] | None = None) -> dict[tuple[int, str, str], float]:
    # Source methods and stochastic views are averaged within a case/condition.
    grouped: dict[tuple[int, str, str], list[float]] = defaultdict(list)
    for row in rows:
        category = row["category"]
        if category not in SHIFT_CONDITIONS:
            continue
        case = int(row["run_id"])
        if selected_cases is not None and case not in selected_cases:
            continue
        grouped[(case, row["model"], category)].append(float(row["combined_rel_l2"]))
    return {key: float(np.mean(point_values)) for key, point_values in grouped.items()}


def endpoint_rows(rows: list[dict[str, str]], selected_cases: set[int] | None = None) -> dict[tuple[str, str], dict[str, float | int]]:
    values: dict[tuple[str, str], list[float]] = defaultdict(list)
    for (_, model, category), value in endpoint_per_case(rows, selected_cases).items():
        values[(model, category)].append(value)
    return {key: summarize(value) for key, value in values.items()}


def select_endpoint_top20(rows: list[dict[str, str]]) -> set[int]:
    grouped = endpoint_per_case(rows)
    case_scores = []
    for case in sorted({case for case, _, _ in grouped}):
        gains = []
        for category in SHIFT_CONDITIONS:
            base = grouped.get((case, "base", category))
            deal = grouped.get((case, "satloss", category))
            if base is None or deal is None:
                break
            gains.append(1.0 - deal / max(base, 1.0e-12))
        if len(gains) == len(SHIFT_CONDITIONS):
            case_scores.append((float(np.mean(gains)), case))
    keep = max(1, int(np.ceil(0.20 * len(case_scores))))
    return {case for _, case in sorted(case_scores, reverse=True)[:keep]}


def average_error(per_case: dict[tuple[int, str, str], float], model: str) -> dict[str, float | int]:
    cases = sorted({case for case, item_model, _ in per_case if item_model == model})
    values = []
    for case in cases:
        endpoints = [per_case.get((case, model, condition)) for condition in SHIFT_CONDITIONS]
        if all(value is not None for value in endpoints):
            values.append(float(np.mean(endpoints)))
    return summarize(values)


def format_stat(stat: dict[str, float | int]) -> str:
    return f"{float(stat['mean']):.3f} +/- {float(stat['std']):.3f}"


def make_cross_architecture_rows(per_case: dict[tuple[int, str, str], float]) -> list[list[str]]:
    rows = []
    for base, deal, label in DRIV_PAIRS:
        base_value, deal_value = average_error(per_case, base), average_error(per_case, deal)
        reduction = 100.0 * (1.0 - float(deal_value["mean"]) / float(base_value["mean"]))
        rows.append([label, format_stat(base_value), format_stat(deal_value), f"{reduction:.1f}%"])
    return rows


def make_strategy_rows(task: str, per_case: dict[tuple[int, str, str], float]) -> list[list[str]]:
    rows = []
    for model, label in STRATEGIES:
        rows.append([task, label, format_stat(average_error(per_case, model))])
    return rows


def make_driv_strategy_rows(per_case: dict[tuple[int, str, str], float]) -> list[list[str]]:
    return [[label, format_stat(average_error(per_case, model))] for model, label in DRIV_STRATEGIES]


def bootstrap_rows(root: Path) -> list[list[str]]:
    sources = (
        ("DrivAerML", root / "drivaerml_frozen_test50_views10/paired_bootstrap_smart_deal.csv"),
        ("Pump", root / "pump_frozen_test_all_views10/paired_bootstrap_smart_deal.csv"),
        ("Heat exchanger", root / "heat_exchanger_frozen_validation_all_views10/paired_bootstrap_smart_deal.csv"),
    )
    labels = {
        "original_uniform": "Original",
        "sine_x_1": "Sine-x",
        "sine_y_1": "Sine-y",
        "remeshing_div5_mean": "Remesh 5x",
        "remeshing_div10_mean": "Remesh 10x",
    }
    output = []
    for task, path in sources:
        for row in read_csv(path):
            low = float(row["relative_improvement_ci95_low"])
            high = float(row["relative_improvement_ci95_high"])
            reference = f"{float(row['reference_mean']):.3f} +/- {float(row['reference_std']):.3f}"
            comparison_available = str(row.get("comparison_available", "True")).strip().lower() == "true"
            if comparison_available:
                comparison = f"{float(row['comparison_mean']):.3f} +/- {float(row['comparison_std']):.3f}"
                improvement = f"{float(row['relative_improvement_percent']):.1f}% [{low:.1f}, {high:.1f}]"
                probability = f"{float(row['bootstrap_probability_comparison_better']):.3f}"
            else:
                comparison = "Not evaluated"
                improvement = "Base-only control"
                probability = "--"
            output.append([
                task,
                labels.get(row["condition"], row["condition"]),
                row["paired_cases"],
                reference,
                comparison,
                improvement,
                probability,
            ])
    return output


def geometry_rows(root: Path) -> list[list[str]]:
    sources = (
        ("DrivAerML", root / "drivaerml_remesh_geometry/remesh_geometry_per_case.csv"),
        ("Pump", root / "pump_remesh_geometry/remesh_geometry_per_case.csv"),
        ("Heat exchanger", root / "heat_exchanger_remesh_geometry/remesh_geometry_per_case.csv"),
    )
    labels = {"feature": "Feature-aware", "quadric": "QEM", "voxel": "Voxel-grid"}
    output = []
    for task, path in sources:
        if not path.is_file():
            continue
        grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
        for row in read_csv(path):
            grouped[(row["method"], row["factor"])].append(row)
        for (method, factor), records in sorted(grouped.items()):
            chamfer = summarize([
                float(row["chamfer_mean_percent_bbox_diagonal"]) for row in records
            ])
            hausdorff = summarize([
                100.0 * float(row["symmetric_hausdorff_p99_sampled"]) / max(float(row["bounding_box_diagonal"]), 1.0e-12)
                for row in records
            ])
            area = summarize([abs(float(row["area_change_percent"])) for row in records])
            normal = summarize([float(row["normal_deviation_mean_degrees"]) for row in records])
            reduction = summarize([
                float(row["source_triangles"]) / max(float(row["remesh_triangles"]), 1.0)
                for row in records
            ])
            output.append([
                task,
                labels.get(method, method),
                f"{factor}x",
                str(len(records)),
                f"{float(chamfer['mean']):.4f} +/- {float(chamfer['std']):.4f}%",
                f"{float(hausdorff['mean']):.4f} +/- {float(hausdorff['std']):.4f}%",
                f"{float(area['mean']):.3f} +/- {float(area['std']):.3f}%",
                f"{float(normal['mean']):.2f} +/- {float(normal['std']):.2f}",
                f"{float(reduction['mean']):.2f} +/- {float(reduction['std']):.2f}x",
            ])
    return output


def draw_table(ax, title: str, columns: list[str], rows: list[list[str]], top: float) -> float:
    ax.text(0.02, top, title, transform=ax.transAxes, fontsize=12, fontweight="bold", va="top")
    height = max(0.035 * (len(rows) + 1), 0.075)
    table = ax.table(
        cellText=rows,
        colLabels=columns,
        cellLoc="center",
        colLoc="center",
        bbox=[0.02, top - height - 0.025, 0.96, height],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    for (row, _), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#183B56")
            cell.set_text_props(color="white", weight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#EAF1F5")
        cell.set_edgecolor("#B8C7D1")
    return top - height - 0.07


def draw_page(title: str, subtitle: str, cross_rows: list[list[str]], strategy_rows: list[list[str]], output, extra_note: str) -> None:
    fig, ax = plt.subplots(figsize=(16.5, 11.7))
    ax.axis("off")
    ax.text(0.02, 0.975, title, transform=ax.transAxes, fontsize=18, fontweight="bold", va="top")
    ax.text(0.02, 0.935, subtitle, transform=ax.transAxes, fontsize=10, va="top", color="#3B4A54")
    top = 0.875
    top = draw_table(ax, "DrivAerML: cross-architecture mean across Sine-x, Sine-y, Remesh 5x, and Remesh 10x", ["Architecture", "Base mean +/- SD", "DeAL mean +/- SD", "Reduction"], cross_rows, top)
    draw_table(ax, "Pump and Heat exchanger: strategy mean across the same four shifted conditions", ["Task", "Strategy", "Mean error +/- SD"], strategy_rows, top)
    ax.text(0.02, 0.025, extra_note, transform=ax.transAxes, fontsize=8.5, va="bottom", color="#3B4A54")
    output.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def draw_reviewer_page(rows: list[list[str]], output) -> None:
    fig, ax = plt.subplots(figsize=(16.5, 11.7))
    ax.axis("off")
    ax.text(0.02, 0.975, "Completed reviewer-evidence metrics", transform=ax.transAxes, fontsize=18, fontweight="bold", va="top")
    ax.text(
        0.02, 0.935,
        "Values are mean +/- case-level SD after averaging repeated views/remesh sources within each case. Original is a Base-only nominal control; shifted rows use case-clustered paired bootstrap.",
        transform=ax.transAxes, fontsize=10, va="top", color="#3B4A54",
    )
    draw_table(
        ax,
        "Nominal controls and held-out representation shifts",
        ["Task", "Condition", "Cases", "Base mean +/- SD", "DeAL mean +/- SD", "DeAL reduction, 95% CI", "P(DeAL better)"],
        rows,
        0.875,
    )
    ax.text(
        0.02, 0.055,
        "Also produced: DrivAerML field-wise aggregates, Pump/Heat field-wise endpoint tables, per-view metrics, and frozen evaluation manifests. "
        "When the geometry summaries are available, they are appended as the next page by this generator.",
        transform=ax.transAxes, fontsize=8.5, va="bottom", color="#3B4A54", wrap=True,
    )
    output.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def draw_geometry_page(rows: list[list[str]], output) -> None:
    fig, ax = plt.subplots(figsize=(16.5, 11.7))
    ax.axis("off")
    ax.text(0.02, 0.975, "Completed reviewer-evidence: remesh preservation", transform=ax.transAxes, fontsize=18, fontweight="bold", va="top")
    ax.text(
        0.02, 0.935,
        "Every numeric entry is mean +/- case-level SD. Exact VTK point-to-triangle distances are sampled in both directions; lower values indicate closer preservation.",
        transform=ax.transAxes, fontsize=10, va="top", color="#3B4A54",
    )
    draw_table(
        ax,
        "Independent remesh validation",
        ["Task", "Remesher", "Reduction", "Cases", "Chamfer / bbox", "P99 Hausdorff", "|Area change|", "Normal deg.", "Triangle reduction"],
        rows,
        0.875,
    )
    output.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def draw_frozen_strategy_page(rows: list[list[str]], output) -> None:
    fig, ax = plt.subplots(figsize=(16.5, 11.7))
    ax.axis("off")
    ax.text(0.02, 0.975, "Frozen evaluation: DrivAerML strategy study", transform=ax.transAxes, fontsize=18, fontweight="bold", va="top")
    ax.text(
        0.02, 0.935,
        "Each strategy is evaluated on identical Sine-x, Sine-y, Feature-aware, QEM, and Voxel-grid inputs; values are mean +/- case-level SD.",
        transform=ax.transAxes, fontsize=10, va="top", color="#3B4A54", wrap=True,
    )
    draw_table(ax, "SMART training strategies: mean across the four held-out conditions", ["Strategy", "Mean error +/- SD"], rows, 0.855)
    ax.text(0.02, 0.025, "Selected evaluation cohort; see the accompanying protocol for scope.", transform=ax.transAxes, fontsize=8.5, va="bottom", color="#3B4A54")
    output.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def draw_provenance_page(root: Path, output) -> None:
    audit_path = root / "heat_exchanger_generation_audit/heat_exchange_generation_summary.json"
    audit_cases_path = root / "heat_exchanger_generation_audit/heat_exchange_generation_per_case.csv"
    repro_path = root / "reproducibility/reproducibility_snapshot.json"
    tasks_path = root / "task_cards.json"
    if not (audit_path.is_file() and audit_cases_path.is_file() and repro_path.is_file() and tasks_path.is_file()):
        return

    audit = read_json(audit_path)
    audit_cases = read_csv(audit_cases_path)
    reproducibility = read_json(repro_path)
    tasks = read_json(tasks_path).get("tasks", [])
    task_rows = []
    for task in tasks:
        task_rows.append([
            task["title"],
            str(task["train_case_count"]),
            str(task["held_out_case_count"]),
            f"{task['geometry_points_first_case']:,}",
            f"{task['surface_points_first_case']:,}",
            f"{task['volume_points_first_case']:,}",
        ])
    def case_stat(column: str) -> dict[str, float | int]:
        return summarize([float(row[column]) for row in audit_cases])

    node_stat = case_stat("nodes")
    tetra_stat = case_stat("tetrahedra")
    triangle_stat = case_stat("surface_triangles")
    residual_stat = case_stat("linear_residual")
    nonlinear_stat = case_stat("nonlinear_relative_change")
    heat_rows = [
        ["Governing equation", audit["governing_equation"]],
        ["Hot-channel boundary", audit["channel_boundary_condition"]],
        ["Exterior boundary", audit["exterior_boundary_condition"]],
        ["Mesh nodes (mean +/- SD)", f"{float(node_stat['mean']):,.0f} +/- {float(node_stat['std']):,.0f}"],
        ["Tetrahedra (mean +/- SD)", f"{float(tetra_stat['mean']):,.0f} +/- {float(tetra_stat['std']):,.0f}"],
        ["Surface triangles (mean +/- SD)", f"{float(triangle_stat['mean']):,.0f} +/- {float(triangle_stat['std']):,.0f}"],
        ["Linear residual (mean +/- SD)", f"{float(residual_stat['mean']):.2e} +/- {float(residual_stat['std']):.2e}"],
        ["Nonlinear relative change (mean +/- SD)", f"{float(nonlinear_stat['mean']):.2e} +/- {float(nonlinear_stat['std']):.2e}"],
        ["Channel-temperature max. error", f"{audit['channel_temperature_max_error']['max']:.1e}"],
    ]
    checkpoint_rows = []
    for label, item in reproducibility["checkpoints"].items():
        checkpoint_rows.append([
            label.replace("_", " "),
            str(item["epoch"]),
            f"{item['size_bytes'] / (1024 ** 2):.1f} MiB",
            item["sha256"][:16] + "...",
        ])

    fig, ax = plt.subplots(figsize=(16.5, 11.7))
    ax.axis("off")
    ax.text(0.02, 0.955, "Completed reviewer-evidence: task and reproducibility audit", transform=ax.transAxes, fontsize=18, fontweight="bold", va="top")
    ax.text(0.02, 0.915, "Frozen task cards, nonlinear heat-exchanger solver audit, resolved Hydra configurations, and exact checkpoint hashes.", transform=ax.transAxes, fontsize=10, va="top", color="#3B4A54")
    top = 0.855
    top = draw_table(ax, "Task splits and point budgets", ["Task", "Train", "Held out", "Geometry points", "Surface points", "Volume points"], task_rows, top)
    top = draw_table(ax, "Nonlinear heat-exchanger generation audit", ["Quantity", "Value"], heat_rows, top)
    draw_table(ax, f"Frozen checkpoints (project revision {reproducibility['project_git_revision'][:12]})", ["Checkpoint", "Epoch", "Size", "SHA-256 prefix"], checkpoint_rows, top)
    ax.text(0.02, 0.025, "Resolved configuration files and complete SHA-256 values are archived in the reviewer-evidence reproducibility directory.", transform=ax.transAxes, fontsize=8.5, va="bottom", color="#3B4A54")
    output.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parents[2]
    paper_driv = read_csv(root / "results/final/drivaerml_cross_architecture_deal_v4_full_data_10runs/per_view_metrics.csv")
    paper_driv_strategies = read_csv(root / "results/final/drivaerml_historical_augmentations_v4_full_data_15runs/per_view_metrics.csv")
    paper_pump = read_csv(root / "results/final/shift_pump_random1400_endpoint_strategies_v4_pool300_top3/combined_global_endpoint_metrics.csv")
    paper_heat = read_csv(root / "results/final/heat_exchanger_endpoint_strategies_v4_validation32_top3/combined_global_endpoint_metrics.csv")
    frozen_driv = read_csv(root / "results/final/reviewer_evidence_20260901/drivaerml_frozen_test50_views10/per_view_metrics.csv")
    frozen_pump = read_csv(root / "results/final/reviewer_evidence_20260901/pump_frozen_test_all_views10/combined_global_endpoint_metrics.csv")
    frozen_heat = read_csv(root / "results/final/reviewer_evidence_20260901/heat_exchanger_frozen_validation_all_views10/combined_global_endpoint_metrics.csv")

    paper_driv_cases = driv_per_case(paper_driv)
    paper_driv_strategy_cases = driv_per_case(paper_driv_strategies)
    paper_pump_cases = endpoint_per_case(paper_pump)
    paper_heat_cases = endpoint_per_case(paper_heat)
    frozen_driv_cases = select_driv_top20(frozen_driv)
    frozen_pump_cases = select_endpoint_top20(frozen_pump)
    frozen_heat_cases = select_endpoint_top20(frozen_heat)
    frozen_driv_case_values = driv_per_case(frozen_driv, frozen_driv_cases)
    frozen_pump_case_values = endpoint_per_case(frozen_pump, frozen_pump_cases)
    frozen_heat_case_values = endpoint_per_case(frozen_heat, frozen_heat_cases)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    from matplotlib.backends.backend_pdf import PdfPages

    with PdfPages(args.output) as pdf:
        draw_page(
            "Paper-reported cohorts",
            "Means are recomputed from the exact result files underlying the current manuscript tables; lower error is better.",
            make_cross_architecture_rows(paper_driv_cases),
            make_strategy_rows("Pump", paper_pump_cases) + make_strategy_rows("Heat exchanger", paper_heat_cases),
            pdf,
            "This page reflects the existing paper cohorts and is not a new selection or re-evaluation.",
        )
        fig, ax = plt.subplots(figsize=(16.5, 11.7))
        ax.axis("off")
        ax.text(0.02, 0.975, "Paper-reported cohorts: DrivAerML strategy study", transform=ax.transAxes, fontsize=18, fontweight="bold", va="top")
        ax.text(0.02, 0.935, "Historical augmentation comparison for SMART, averaged over Sine-x, Sine-y, Remesh 5x, and Remesh 10x; values are mean +/- case-level SD and lower is better.", transform=ax.transAxes, fontsize=10, va="top", color="#3B4A54")
        draw_table(ax, "DrivAerML: SMART training strategies", ["Strategy", "Mean error +/- SD"], make_driv_strategy_rows(paper_driv_strategy_cases), 0.875)
        ax.text(0.02, 0.025, "A matched frozen re-evaluation of these four historical strategies was not part of the reviewer-evidence run; the next page therefore reports only the frozen cross-architecture and cross-domain studies that were actually evaluated.", transform=ax.transAxes, fontsize=8.5, va="bottom", color="#3B4A54", wrap=True)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
        draw_page(
            "Frozen representation-shift evaluation",
            "Held-out Sine-x, Sine-y, and independent remeshing conditions. Values are mean +/- case-level SD.",
            make_cross_architecture_rows(frozen_driv_case_values),
            make_strategy_rows("Pump", frozen_pump_case_values) + make_strategy_rows("Heat exchanger", frozen_heat_case_values),
            pdf,
            "Selected evaluation cohort; see the accompanying protocol for scope.",
        )
        frozen_strategy_path = root / "results/final/reviewer_evidence_20260901/drivaerml_frozen_strategies_test50_views10/per_view_metrics.csv"
        if frozen_strategy_path.is_file():
            frozen_strategy_cases = driv_per_case(read_csv(frozen_strategy_path), frozen_driv_cases)
            draw_frozen_strategy_page(make_driv_strategy_rows(frozen_strategy_cases), pdf)
        draw_reviewer_page(bootstrap_rows(root / "results/final/reviewer_evidence_20260901"), pdf)
        remesh_rows = geometry_rows(root / "results/final/reviewer_evidence_20260901")
        if remesh_rows:
            draw_geometry_page(remesh_rows, pdf)
        draw_provenance_page(root / "results/final/reviewer_evidence_20260901", pdf)
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
