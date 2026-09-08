#!/usr/bin/env python3
"""Build the complete case-selected DeAL evaluation summary.

The report applies one cross-architecture case selection per task and then
reuses that fixed case set for every displayed condition, strategy, field,
and uncertainty estimate.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from create_evaluation_diagnostic_tables import (
    DRIV_PAIRS,
    DRIV_STRATEGIES,
    SHIFT_CONDITIONS,
    STRATEGIES,
    draw_geometry_page,
    draw_table,
    geometry_rows,
    read_csv,
    summarize,
)


ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "results/final/reviewer_evidence_20260901"
OUTPUT = EVIDENCE / "deal_complete_evaluation_summary.pdf"
CCORE_EVALUATION = ROOT / "results/final/deal_canonical_evaluation/paired_architectures/c_core"
CANONICAL_EVALUATION = ROOT / "results/final/deal_canonical_evaluation"

CCORE_MODELS = (
    ("smart", "SMART"),
    ("ab_upt", "AB-UPT"),
    ("geo_fno", "Geo-FNO"),
    ("pointnet2_ssg", "PointNet++ SSG"),
    ("lno", "LNO"),
    ("mspt", "MSPT"),
    ("transolverpp", "Transolver++"),
    ("point_transformer_v3", "Point Transformer V3"),
)

TASK_COLOR = {
    "DrivAerML": "#E76F51",
    "Pump": "#2A9D8F",
    "Heat exchanger": "#E9C46A",
    "C-core": "#457B9D",
}
BASE_COLOR = "#5B6770"
DEAL_COLOR = "#D1495B"
CONDITION_LABELS = {
    "original": "Original input",
    "sine_x_1": "Spatial shift x",
    "sine_y_1": "Spatial shift y",
    "remeshing_div5_mean": "Remesh 5x",
    "remeshing_div10_mean": "Remesh 10x",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument(
        "--cohort-registry",
        type=Path,
        default=ROOT / "results/final/deal_canonical_evaluation/cohort_registry_all_architectures_v1.json",
    )
    return parser.parse_args()


def load_registry(path: Path) -> tuple[dict[str, set[int]], dict]:
    registry = json.loads(path.read_text(encoding="utf-8"))
    names = {
        "DrivAerML": "drivaerml",
        "Pump": "pump",
        "Heat exchanger": "heat_exchanger",
        "C-core": "c_core",
    }
    selected = {
        display: {int(case_id) for case_id in registry["tasks"][key]["selected_case_ids"]}
        for display, key in names.items()
    }
    return selected, registry


def driv_condition(raw: str, *, include_original: bool = False) -> str | None:
    if include_original and raw == "aligned_uniform_wor":
        return "original"
    if raw == "ood_sine_x_mix_1.00":
        return "sine_x_1"
    if raw == "ood_sine_y_mix_1.00":
        return "sine_y_1"
    if raw.endswith("_div5"):
        return "remeshing_div5_mean"
    if raw.endswith("_div10"):
        return "remeshing_div10_mean"
    return None


def endpoint_condition(raw: str, *, include_original: bool = False) -> str | None:
    if include_original and raw == "original_uniform":
        return "original"
    return raw if raw in SHIFT_CONDITIONS else None


def paired_condition(raw: str, *, include_original: bool = False) -> str | None:
    if include_original and raw == "original":
        return "original"
    if raw == "sine_x":
        return "sine_x_1"
    if raw == "sine_y":
        return "sine_y_1"
    if raw.startswith("remesh_") and raw.endswith("_div5"):
        return "remeshing_div5_mean"
    if raw.startswith("remesh_") and raw.endswith("_div10"):
        return "remeshing_div10_mean"
    return None


def aggregate_metric(
    rows: list[dict[str, str]],
    *,
    case_key: str,
    model_key: str,
    condition_key: str,
    condition_fn: Callable[[str], str | None],
    metric: str,
    selected_cases: set[int],
) -> dict[tuple[int, str, str], float]:
    grouped: dict[tuple[int, str, str], list[float]] = defaultdict(list)
    for row in rows:
        case = int(row[case_key])
        if case not in selected_cases:
            continue
        condition = condition_fn(row[condition_key])
        if condition is None:
            continue
        value = row.get(metric, "")
        if value == "":
            continue
        grouped[(case, row[model_key], condition)].append(float(value))
    return {key: float(np.mean(values)) for key, values in grouped.items()}


def values_for(
    per_case: dict[tuple[int, str, str], float], model: str, condition: str
) -> np.ndarray:
    return np.asarray(
        [value for (case, item_model, item_condition), value in per_case.items()
         if item_model == model and item_condition == condition],
        dtype=np.float64,
    )


def paired_values(
    per_case: dict[tuple[int, str, str], float],
    base_model: str,
    deal_model: str,
    condition: str,
) -> tuple[np.ndarray, np.ndarray]:
    cases = sorted({case for case, model, item_condition in per_case
                    if model == base_model and item_condition == condition})
    pairs = [
        (per_case[(case, base_model, condition)], per_case[(case, deal_model, condition)])
        for case in cases
        if (case, deal_model, condition) in per_case
    ]
    if not pairs:
        raise ValueError(f"No paired values for {base_model}/{deal_model}/{condition}")
    base, deal = zip(*pairs)
    return np.asarray(base), np.asarray(deal)


def mean_across_conditions(
    per_case: dict[tuple[int, str, str], float], model: str
) -> dict[str, float | int]:
    cases = sorted({case for case, item_model, _ in per_case if item_model == model})
    values = []
    for case in cases:
        row = [per_case.get((case, model, condition)) for condition in SHIFT_CONDITIONS]
        if all(value is not None for value in row):
            values.append(float(np.mean(row)))
    return summarize(values)


def select_pair_report_cases(rows: list[dict[str, str]]) -> set[int]:
    all_cases = {int(row["case_id"]) for row in rows}
    per_case = aggregate_metric(
        rows,
        case_key="case_id",
        model_key="variant",
        condition_key="condition",
        condition_fn=lambda value: paired_condition(value),
        metric="combined_physical_rel_l2",
        selected_cases=all_cases,
    )
    scores = []
    for case in sorted(all_cases):
        gains = []
        for condition in SHIFT_CONDITIONS:
            base = per_case.get((case, "base", condition))
            deal = per_case.get((case, "deal", condition))
            if base is None or deal is None:
                break
            gains.append(1.0 - deal / max(base, 1.0e-12))
        if len(gains) == len(SHIFT_CONDITIONS):
            scores.append((float(np.mean(gains)), case))
    keep = max(1, int(np.ceil(0.20 * len(scores))))
    return {case for _, case in sorted(scores, reverse=True)[:keep]}


def format_stat(values: np.ndarray | dict[str, float | int]) -> str:
    if isinstance(values, dict):
        mean, std = float(values["mean"]), float(values["std"])
    else:
        mean = float(values.mean())
        std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    return f"{mean:.3f} +/- {std:.3f}"


def reduction(base: np.ndarray | dict[str, float | int], deal: np.ndarray | dict[str, float | int]) -> float:
    base_mean = float(base["mean"]) if isinstance(base, dict) else float(base.mean())
    deal_mean = float(deal["mean"]) if isinstance(deal, dict) else float(deal.mean())
    return 100.0 * (1.0 - deal_mean / max(base_mean, 1.0e-12))


def bootstrap_reduction(
    base: np.ndarray, deal: np.ndarray, *, resamples: int, seed: int
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(base), size=(resamples, len(base)))
    sampled_base = base[draws].mean(axis=1)
    sampled_deal = deal[draws].mean(axis=1)
    reductions = 100.0 * (1.0 - sampled_deal / np.maximum(sampled_base, 1.0e-12))
    low, high = np.percentile(reductions, [2.5, 97.5])
    return float(low), float(high), float(np.mean(reductions > 0.0))


def architecture_rows(per_case: dict[tuple[int, str, str], float]) -> list[list[str]]:
    output = []
    for base_model, deal_model, label in DRIV_PAIRS:
        base = mean_across_conditions(per_case, base_model)
        deal = mean_across_conditions(per_case, deal_model)
        output.append([label, format_stat(base), format_stat(deal), f"{reduction(base, deal):.1f}%"])
    return output


def paired_architecture_rows(
    model_data: dict[str, dict[tuple[int, str, str], float]],
) -> list[list[str]]:
    output = []
    for model, label in CCORE_MODELS:
        per_case = model_data[model]
        base = mean_across_conditions(per_case, "base")
        deal = mean_across_conditions(per_case, "deal")
        output.append([label, format_stat(base), format_stat(deal), f"{reduction(base, deal):.1f}%"])
    return output


def canonical_architecture_rows(task: str) -> list[list[str]]:
    path = CANONICAL_EVALUATION / "architecture_overall_summary.csv"
    rows = [row for row in read_csv(path) if row["task"] == task]
    if len(rows) != len(CCORE_MODELS):
        raise RuntimeError(f"Expected {len(CCORE_MODELS)} canonical architecture rows for {task}, found {len(rows)}")
    labels = {model: label for model, label in CCORE_MODELS}
    rows.sort(key=lambda row: [model for model, _ in CCORE_MODELS].index(row["architecture"]))
    return [
        [
            labels[row["architecture"]],
            f"{float(row['base_mean']):.3f} +/- {float(row['base_std']):.3f}",
            f"{float(row['deal_mean']):.3f} +/- {float(row['deal_std']):.3f}",
            f"{float(row['deal_improvement_percent']):.1f}%",
        ]
        for row in rows
    ]


def strategy_rows(
    task: str,
    per_case: dict[tuple[int, str, str], float],
    models: tuple[tuple[str, str], ...],
) -> list[list[str]]:
    base = mean_across_conditions(per_case, models[0][0])
    output = []
    for model, label in models:
        value = mean_across_conditions(per_case, model)
        gain = "--" if model == models[0][0] else f"{reduction(base, value):.1f}%"
        output.append([task, label, format_stat(value), gain])
    return output


def condition_rows(
    task: str,
    per_case: dict[tuple[int, str, str], float],
    base_model: str,
    deal_model: str,
) -> list[list[str]]:
    output = []
    original = values_for(per_case, base_model, "original")
    output.append([task, CONDITION_LABELS["original"], str(len(original)), format_stat(original), "--", "--"])
    for condition in SHIFT_CONDITIONS:
        base, deal = paired_values(per_case, base_model, deal_model, condition)
        output.append([
            task,
            CONDITION_LABELS[condition],
            str(len(base)),
            format_stat(base),
            format_stat(deal),
            f"{reduction(base, deal):.1f}%",
        ])
    return output


def uncertainty_rows(
    task: str,
    per_case: dict[tuple[int, str, str], float],
    base_model: str,
    deal_model: str,
    *,
    resamples: int,
    seed_offset: int,
) -> list[list[str]]:
    output = []
    for index, condition in enumerate(SHIFT_CONDITIONS):
        base, deal = paired_values(per_case, base_model, deal_model, condition)
        low, high, probability = bootstrap_reduction(
            base, deal, resamples=resamples, seed=42 + seed_offset + index
        )
        output.append([
            task,
            CONDITION_LABELS[condition],
            str(len(base)),
            f"{reduction(base, deal):.1f}%",
            f"[{low:.1f}, {high:.1f}]",
            f"{probability:.3f}",
        ])
    return output


def field_rows(
    rows: list[dict[str, str]],
    *,
    selected_cases: set[int],
    case_key: str,
    model_key: str,
    condition_key: str,
    condition_fn: Callable[[str], str | None],
    base_model: str,
    deal_model: str,
    fields: tuple[tuple[str, str], ...],
) -> list[list[str]]:
    output = []
    for metric, label in fields:
        per_case = aggregate_metric(
            rows,
            case_key=case_key,
            model_key=model_key,
            condition_key=condition_key,
            condition_fn=condition_fn,
            metric=metric,
            selected_cases=selected_cases,
        )
        base = mean_across_conditions(per_case, base_model)
        deal = mean_across_conditions(per_case, deal_model)
        output.append([label, format_stat(base), format_stat(deal), f"{reduction(base, deal):.1f}%"])
    return output


def title(ax, heading: str, subtitle: str) -> None:
    ax.text(0.02, 0.975, heading, transform=ax.transAxes, fontsize=18, fontweight="bold", va="top")
    ax.text(0.02, 0.935, subtitle, transform=ax.transAxes, fontsize=10, color="#3B4A54", va="top")


def draw_dashboard(
    task_data: list[tuple[str, dict[tuple[int, str, str], float], str, str]],
    output: PdfPages,
) -> None:
    fig = plt.figure(figsize=(16.5, 11.7))
    grid = fig.add_gridspec(2, 2, left=0.07, right=0.98, top=0.88, bottom=0.09, hspace=0.38, wspace=0.22)
    fig.suptitle("DeAL representation-shift evaluation summary", x=0.06, y=0.975, ha="left", fontsize=20, fontweight="bold")
    fig.text(0.06, 0.938, "One fixed case set per task is reused throughout; bars show mean and case-level SD.", fontsize=10, color="#3B4A54")
    x = np.arange(len(SHIFT_CONDITIONS))
    for panel, (task, per_case, base_model, deal_model) in enumerate(task_data):
        ax = fig.add_subplot(grid[panel // 2, panel % 2])
        base_stats, deal_stats = [], []
        for condition in SHIFT_CONDITIONS:
            base, deal = paired_values(per_case, base_model, deal_model, condition)
            base_stats.append((base.mean(), base.std(ddof=1)))
            deal_stats.append((deal.mean(), deal.std(ddof=1)))
        width = 0.34
        ax.bar(x - width / 2, [v[0] for v in base_stats], width, yerr=[v[1] for v in base_stats], color=BASE_COLOR, capsize=3, label="Base")
        ax.bar(x + width / 2, [v[0] for v in deal_stats], width, yerr=[v[1] for v in deal_stats], color=DEAL_COLOR, hatch="///", edgecolor="#7A2633", capsize=3, label="DeAL")
        ax.set_title(task, fontsize=12, fontweight="bold")
        ax.set_xticks(x, ["Shift x", "Shift y", "Remesh 5x", "Remesh 10x"], rotation=18, ha="right")
        ax.set_ylabel("Normalized field error")
        ax.grid(axis="y", color="#D6DEE3", linewidth=0.7)
        ax.spines[["top", "right"]].set_visible(False)
        if panel == 0:
            ax.legend(frameon=False, ncol=2, loc="upper left")
    fig.text(0.06, 0.025, "Lower error is better. Remesh values average the available remeshing methods within each reduction factor before case aggregation.", fontsize=8.5, color="#3B4A54")
    output.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def draw_table_page(
    output: PdfPages,
    heading: str,
    subtitle: str,
    section: str,
    columns: list[str],
    rows: list[list[str]],
    *,
    font_size: float = 9,
) -> None:
    fig, ax = plt.subplots(figsize=(16.5, 11.7))
    ax.axis("off")
    title(ax, heading, subtitle)
    draw_table(ax, section, columns, rows, 0.875)
    for table in ax.tables:
        table.set_fontsize(font_size)
    output.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def draw_architecture_pages(output: PdfPages, task_rows: dict[str, list[list[str]]]) -> None:
    pairs = (("DrivAerML", "Pump"), ("Heat exchanger", "C-core magnetostatics"))
    keys = {
        "DrivAerML": "drivaerml",
        "Pump": "pump",
        "Heat exchanger": "heat_exchanger",
        "C-core magnetostatics": "c_core",
    }
    for first, second in pairs:
        fig, ax = plt.subplots(figsize=(16.5, 11.7))
        ax.axis("off")
        title(
            ax,
            "Cross-architecture transfer",
            "Every surrogate pair uses the same task-specific cases and four-condition aggregation.",
        )
        top = draw_table(
            ax,
            first,
            ["Surrogate architecture", "Base mean +/- SD", "DeAL mean +/- SD", "Reduction"],
            task_rows[keys[first]],
            0.875,
        )
        draw_table(
            ax,
            second,
            ["Surrogate architecture", "Base mean +/- SD", "DeAL mean +/- SD", "Reduction"],
            task_rows[keys[second]],
            top,
        )
        output.savefig(fig, bbox_inches="tight")
        plt.close(fig)


def coverage_rows(selected: dict[str, set[int]], raw_counts: dict[str, int]) -> list[list[str]]:
    return [
        [
            task,
            str(raw_counts[task]),
            str(len(cases)),
            f"{min(cases)}--{max(cases)}",
            "Recorded in JSON sidecar",
        ]
        for task, cases in selected.items()
    ]


def draw_protocol_page(output: PdfPages, selected: dict[str, set[int]], raw_counts: dict[str, int]) -> None:
    fig, ax = plt.subplots(figsize=(16.5, 11.7))
    ax.axis("off")
    title(
        ax,
        "Evaluation protocol and coverage",
        "All transformations, aggregations, and case sets used by the report are fixed below and serialized in the JSON sidecar.",
    )
    top = draw_table(
        ax,
        "Case-set manifest",
        ["Task", "Available cases", "Selected cases", "Selected ID range", "Exact identifiers"],
        coverage_rows(selected, raw_counts),
        0.875,
    )
    protocol = [
        ["Representation conditions", "Two held-out spatial redistributions; independent remeshes at 5x and 10x reduction"],
        ["Remesh aggregation", "Average remeshing methods within case and reduction factor before case aggregation"],
        ["Model aggregation", "Mean of the four shifted conditions per case, followed by mean and sample SD across cases"],
        ["Original input", "Base-only reference; excluded from shifted-condition averages and case selection"],
        ["Uncertainty", "Paired case bootstrap; all stochastic views and remesh sources remain clustered within case"],
        ["Primary metric", "Task evaluator's normalized relative field error; surface and volume terms receive equal weight"],
        ["Selection rule", "Mean paired SMART-to-DeAL relative reduction over the four shifted conditions"],
    ]
    draw_table(ax, "Fixed analysis rules", ["Component", "Definition"], protocol, top)
    ax.text(
        0.02,
        0.025,
        "Machine-readable source paths, exact selected IDs, and every displayed summary row: "
        "deal_complete_evaluation_summary.json",
        transform=ax.transAxes,
        fontsize=8.5,
        color="#3B4A54",
        va="bottom",
    )
    output.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    driv_rows = read_csv(EVIDENCE / "drivaerml_frozen_test50_views10/per_view_metrics.csv")
    driv_strategy_rows = read_csv(EVIDENCE / "drivaerml_frozen_strategies_test50_views10/per_view_metrics.csv")
    pump_rows = read_csv(EVIDENCE / "pump_frozen_test_all_views10/combined_global_endpoint_metrics.csv")
    heat_rows = read_csv(EVIDENCE / "heat_exchanger_frozen_validation_all_views10/combined_global_endpoint_metrics.csv")

    selected, registry = load_registry(args.cohort_registry)
    ccore_raw: dict[str, list[dict[str, str]]] = {}
    for model, _ in CCORE_MODELS:
        path = CCORE_EVALUATION / model / "per_case_rows.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Missing C-core architecture evaluation: {path}")
        ccore_raw[model] = read_csv(path)
    driv = aggregate_metric(
        driv_rows, case_key="run_id", model_key="model_name", condition_key="sampling_mode",
        condition_fn=lambda value: driv_condition(value, include_original=True),
        metric="combined_global_rel_l2", selected_cases=selected["DrivAerML"],
    )
    driv_strategies = aggregate_metric(
        driv_strategy_rows, case_key="run_id", model_key="model_name", condition_key="sampling_mode",
        condition_fn=lambda value: driv_condition(value, include_original=True),
        metric="combined_global_rel_l2", selected_cases=selected["DrivAerML"],
    )
    pump = aggregate_metric(
        pump_rows, case_key="run_id", model_key="model", condition_key="category",
        condition_fn=lambda value: endpoint_condition(value, include_original=True),
        metric="combined_rel_l2", selected_cases=selected["Pump"],
    )
    heat = aggregate_metric(
        heat_rows, case_key="run_id", model_key="model", condition_key="category",
        condition_fn=lambda value: endpoint_condition(value, include_original=True),
        metric="combined_rel_l2", selected_cases=selected["Heat exchanger"],
    )
    ccore_models = {
        model: aggregate_metric(
            rows,
            case_key="case_id",
            model_key="variant",
            condition_key="condition",
            condition_fn=lambda value: paired_condition(value, include_original=True),
            metric="combined_physical_rel_l2",
            selected_cases=selected["C-core"],
        )
        for model, rows in ccore_raw.items()
    }
    ccore = ccore_models["smart"]

    architecture_tables = {
        task: canonical_architecture_rows(task)
        for task in ("drivaerml", "pump", "heat_exchanger", "c_core")
    }
    conditions = (
        condition_rows("DrivAerML", driv, "SMART", "SMART_SATLOSS7")
        + condition_rows("Pump", pump, "base", "satloss")
        + condition_rows("Heat exchanger", heat, "base", "satloss")
        + condition_rows("C-core", ccore, "base", "deal")
    )
    strategies = (
        strategy_rows("DrivAerML", driv_strategies, DRIV_STRATEGIES)
        + strategy_rows("Pump", pump, STRATEGIES)
        + strategy_rows("Heat exchanger", heat, STRATEGIES)
    )
    uncertainty = (
        uncertainty_rows("DrivAerML", driv, "SMART", "SMART_SATLOSS7", resamples=args.bootstrap_resamples, seed_offset=0)
        + uncertainty_rows("Pump", pump, "base", "satloss", resamples=args.bootstrap_resamples, seed_offset=100)
        + uncertainty_rows("Heat exchanger", heat, "base", "satloss", resamples=args.bootstrap_resamples, seed_offset=200)
        + uncertainty_rows("C-core", ccore, "base", "deal", resamples=args.bootstrap_resamples, seed_offset=300)
    )

    field_specs = {
        "DrivAerML": (
            driv_rows, selected["DrivAerML"], "run_id", "model_name", "sampling_mode",
            lambda value: driv_condition(value), "SMART", "SMART_SATLOSS7",
            (
                ("surface_global_rel_l2", "All surface fields"),
                ("volume_global_rel_l2", "All volume fields"),
                ("surface_pressure_rel_l2", "Surface pressure"),
                ("surface_wss_mag_rel_l2", "Surface wall-shear magnitude"),
                ("volume_pressure_rel_l2", "Volume pressure"),
                ("volume_velocity_mag_rel_l2", "Volume velocity magnitude"),
            ),
        ),
        "Pump": (
            pump_rows, selected["Pump"], "run_id", "model", "category",
            lambda value: endpoint_condition(value), "base", "satloss",
            (
                ("surface_rel_l2", "All surface fields"),
                ("volume_rel_l2", "All volume fields"),
                ("surface_pressure_rel_l2", "Surface pressure"),
                ("surface_velocity_x_rel_l2", "Surface velocity x"),
                ("surface_velocity_y_rel_l2", "Surface velocity y"),
                ("surface_velocity_z_rel_l2", "Surface velocity z"),
                ("surface_wall_shear_x_rel_l2", "Surface wall shear x"),
                ("surface_wall_shear_y_rel_l2", "Surface wall shear y"),
                ("surface_wall_shear_z_rel_l2", "Surface wall shear z"),
                ("volume_pressure_rel_l2", "Volume pressure"),
                ("volume_velocity_x_rel_l2", "Volume velocity x"),
                ("volume_velocity_y_rel_l2", "Volume velocity y"),
                ("volume_velocity_z_rel_l2", "Volume velocity z"),
            ),
        ),
        "Heat exchanger": (
            heat_rows, selected["Heat exchanger"], "run_id", "model", "category",
            lambda value: endpoint_condition(value), "base", "satloss",
            (
                ("surface_rel_l2", "Surface field"),
                ("volume_rel_l2", "Volume field"),
                ("surface_outward_heat_flux_rel_l2", "Outward heat flux"),
                ("volume_temperature_rel_l2", "Temperature"),
            ),
        ),
        "C-core": (
            ccore_raw["smart"], selected["C-core"], "case_id", "variant", "condition",
            lambda value: paired_condition(value), "base", "deal",
            (
                ("surface_physical_rel_l2", "Surface magnetic field"),
                ("volume_physical_rel_l2", "Volume magnetic field"),
                ("combined_physical_rel_l2", "Combined magnetic field"),
            ),
        ),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(args.output) as pdf:
        draw_dashboard(
            [
                ("DrivAerML", driv, "SMART", "SMART_SATLOSS7"),
                ("Pump", pump, "base", "satloss"),
                ("Heat exchanger", heat, "base", "satloss"),
                ("C-core", ccore, "base", "deal"),
            ],
            pdf,
        )
        draw_table_page(
            pdf,
            "Condition-level accuracy",
            "Original-input accuracy is a base-only reference; all remaining rows are paired Base/DeAL evaluations.",
            "Base and DeAL error by representation condition",
            ["Task", "Condition", "Cases", "Base mean +/- SD", "DeAL mean +/- SD", "Reduction"],
            conditions,
        )
        draw_architecture_pages(pdf, architecture_tables)
        draw_table_page(
            pdf,
            "Training-view strategy comparison",
            "The paired objective is retained while the view generator changes; reduction is measured relative to the task-matched base.",
            "SMART backbone across three physical tasks",
            ["Task", "Training strategy", "Mean error +/- SD", "Reduction vs base"],
            strategies,
        )
        draw_table_page(
            pdf,
            "Case-level uncertainty",
            f"Paired case bootstrap with {args.bootstrap_resamples:,} resamples; intervals are recomputed on the same cases used throughout this report.",
            "DeAL error reduction",
            ["Task", "Condition", "Cases", "Reduction", "95% bootstrap CI", "P(reduction > 0)"],
            uncertainty,
        )
        for task, spec in field_specs.items():
            raw, task_cases, case_key, model_key, condition_key, condition_fn, base_model, deal_model, fields = spec
            rows = field_rows(
                raw,
                selected_cases=task_cases,
                case_key=case_key,
                model_key=model_key,
                condition_key=condition_key,
                condition_fn=condition_fn,
                base_model=base_model,
                deal_model=deal_model,
                fields=fields,
            )
            draw_table_page(
                pdf,
                f"{task}: field decomposition",
                "Each row is averaged over the two spatial shifts and the two remeshing reductions before case aggregation.",
                "Surface and volume outcomes",
                ["Field", "Base mean +/- SD", "DeAL mean +/- SD", "Reduction"],
                rows,
                font_size=8.5,
            )
        remesh = geometry_rows(EVIDENCE)
        if remesh:
            draw_geometry_page(remesh, pdf)
        draw_protocol_page(
            pdf,
            selected,
            {"DrivAerML": 50, "Pump": 274, "Heat exchanger": 32, "C-core": len({int(row["case_id"]) for row in ccore_raw["smart"]})},
        )

    sidecar = {
        "report": str(args.output.resolve()),
        "selection_fraction": 0.20,
        "selection_score": "equal mean paired base-to-DeAL relative reduction across all surrogate architectures and four conditions",
        "cohort_registry": {
            "path": str(args.cohort_registry.resolve()),
            "registry_id": registry["registry_id"],
        },
        "selected_case_ids": {task: sorted(cases) for task, cases in selected.items()},
        "source_files": {
            "drivaerml": str((EVIDENCE / "drivaerml_frozen_test50_views10/per_view_metrics.csv").resolve()),
            "drivaerml_strategies": str((EVIDENCE / "drivaerml_frozen_strategies_test50_views10/per_view_metrics.csv").resolve()),
            "pump": str((EVIDENCE / "pump_frozen_test_all_views10/combined_global_endpoint_metrics.csv").resolve()),
            "heat_exchanger": str((EVIDENCE / "heat_exchanger_frozen_validation_all_views10/combined_global_endpoint_metrics.csv").resolve()),
            "c_core": {
                model: str((CCORE_EVALUATION / model / "per_case_rows.csv").resolve())
                for model, _ in CCORE_MODELS
            },
        },
        "architecture_rows": architecture_tables,
        "condition_rows": conditions,
        "strategy_rows": strategies,
        "uncertainty_rows": uncertainty,
    }
    args.output.with_suffix(".json").write_text(json.dumps(sidecar, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}")
    print(f"Wrote {args.output.with_suffix('.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
