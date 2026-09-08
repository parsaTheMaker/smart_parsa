#!/usr/bin/env python3
"""Append table-only DeAL evidence pages to the architecture evaluation PDF."""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages


ROOT = Path(__file__).resolve().parents[2]
EVALUATION_ROOT = ROOT / "results/final/deal_canonical_evaluation"
PAIR_ROOT = EVALUATION_ROOT / "paired_architectures"
DEFAULT_MAIN_PDF = EVALUATION_ROOT / "architecture_evaluation_tables.pdf"
DEFAULT_APPENDIX_PDF = EVALUATION_ROOT / "architecture_evaluation_statistics_appendix.pdf"

TASKS = (
    ("drivaerml", "DrivAerML"),
    ("pump", "Pump"),
    ("heat_exchanger", "Heat Exchanger"),
    ("c_core", "C-core"),
)
MODELS = (
    ("smart", "SMART"),
    ("ab_upt", "AB-UPT"),
    ("geo_fno", "Geo-FNO"),
    ("pointnet2_ssg", "PointNet++ SSG"),
    ("lno", "LNO"),
    ("mspt", "MSPT"),
    ("transolverpp", "Transolver++"),
    ("point_transformer_v3", "Point Transformer V3"),
)
CONDITIONS = ("sine_x", "sine_y", "remesh_mean_div5", "remesh_mean_div10")
CONDITION_LABELS = {
    "sine_x": "Spatial shift x",
    "sine_y": "Spatial shift y",
    "remesh_mean_div5": "Remesh 5x",
    "remesh_mean_div10": "Remesh 10x",
}

# These tables are the paper-reference DrivAerML SMART ablations.  Keep the
# compact evaluation report aligned with the manuscript's fixed reference rows
# instead of mixing in older archive exports produced under a different sweep.
PAPER_REFERENCE_ABLATIONS = {
    "density_range_ablation": (
        ("SMART baseline", (0.1880, 0.2747, 0.1862, 0.2018)),
        ("DeAL [0, 0.25]", (0.1734, 0.1408, 0.1710, 0.1902)),
        ("DeAL [0, 0.50]", (0.1629, 0.0867, 0.1594, 0.1767)),
        ("DeAL [0, 0.75]", (0.1523, 0.0627, 0.1437, 0.1581)),
        ("DeAL [0, 1.00]", (0.0945, 0.0416, 0.1232, 0.1339)),
        ("DeAL [0, 2.00]", (0.0953, 0.0417, 0.1146, 0.1252)),
        ("DeAL [0, 3.00]", (0.0994, 0.0427, 0.1175, 0.1285)),
    ),
    "kde_neighborhood_ablation": (
        ("SMART baseline", (0.1880, 0.2747, 0.1862, 0.2018)),
        ("DeAL KDE k=4", (0.0939, 0.0409, 0.1398, 0.1512)),
        ("DeAL KDE k=8", (0.0937, 0.0409, 0.1285, 0.1396)),
        ("DeAL KDE k=16 (reference)", (0.0945, 0.0416, 0.1232, 0.1339)),
        ("DeAL KDE k=32", (0.0951, 0.0407, 0.1209, 0.1310)),
        ("DeAL KDE k=64", (0.0923, 0.0407, 0.1218, 0.1323)),
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-pdf", type=Path, default=DEFAULT_MAIN_PDF)
    parser.add_argument("--appendix-pdf", type=Path, default=DEFAULT_APPENDIX_PDF)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def configure_style() -> None:
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "figure.facecolor": "white",
    })


def make_page(title: str, subtitle: str) -> tuple[plt.Figure, plt.Axes]:
    fig = plt.figure(figsize=(11.7, 8.3), facecolor="white")
    fig.text(0.045, 0.955, title, fontsize=17, fontweight="bold", ha="left", va="top", color="#152A38")
    fig.text(0.045, 0.915, subtitle, fontsize=9.5, ha="left", va="top", color="#49606D")
    ax = fig.add_axes([0.035, 0.07, 0.93, 0.79])
    ax.axis("off")
    return fig, ax


def draw_table(ax: plt.Axes, rows: list[list[str]], columns: list[str], font_size: float = 7.7) -> None:
    if not rows:
        ax.text(0.5, 0.5, "No compatible source table is available.", ha="center", va="center")
        return
    table = ax.table(cellText=rows, colLabels=columns, cellLoc="center", colLoc="center", bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)
    for (row, _), cell in table.get_celld().items():
        cell.set_edgecolor("#C7D3DA")
        cell.set_linewidth(0.55)
        if row == 0:
            cell.set_facecolor("#173F57")
            cell.set_text_props(color="white", fontweight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#EEF3F5")


def source_condition(value: str) -> str | None:
    if value in {"sine_x", "sine_y"}:
        return value
    match = re.fullmatch(r"remesh_.+_div(5|10)", value)
    if match:
        return f"remesh_mean_div{match.group(1)}"
    return None


def read_case_rows(task: str, model: str) -> list[dict[str, str]]:
    path = PAIR_ROOT / task / model / "per_case_rows.csv"
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def aggregate_rows(rows: list[dict[str, str]], metric: str) -> dict[tuple[str, str, str], float]:
    grouped: dict[tuple[str, str, str, str], list[float]] = defaultdict(list)
    for row in rows:
        condition = source_condition(row["condition"])
        if condition is None:
            continue
        value = row.get(metric, "")
        if not value:
            continue
        grouped[(row["case_id"], row["variant"], condition, row.get("method", ""))].append(float(value))
    per_method = {key: float(np.mean(values)) for key, values in grouped.items()}
    merged: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for (case_id, variant, condition, _), value in per_method.items():
        merged[(case_id, variant, condition)].append(value)
    return {key: float(np.mean(values)) for key, values in merged.items()}


def paired_case_means(rows: list[dict[str, str]], metric: str) -> tuple[np.ndarray, np.ndarray]:
    values = aggregate_rows(rows, metric)
    base_case: dict[str, list[float]] = defaultdict(list)
    deal_case: dict[str, list[float]] = defaultdict(list)
    for condition in CONDITIONS:
        for (case_id, variant, current), value in values.items():
            if current != condition:
                continue
            if variant == "base":
                base_case[case_id].append(value)
            elif variant == "deal":
                deal_case[case_id].append(value)
    pairs = [
        (float(np.mean(base_case[case_id])), float(np.mean(deal_case[case_id])))
        for case_id in sorted(set(base_case) & set(deal_case))
        if len(base_case[case_id]) == len(CONDITIONS) and len(deal_case[case_id]) == len(CONDITIONS)
    ]
    if not pairs:
        return np.empty(0), np.empty(0)
    base, deal = zip(*pairs)
    return np.asarray(base), np.asarray(deal)


def bootstrap_interval(base: np.ndarray, deal: np.ndarray, seed: int, resamples: int) -> tuple[float, float]:
    if len(base) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(base), size=(resamples, len(base)))
    estimates = 100.0 * (1.0 - deal[indices].mean(axis=1) / np.maximum(base[indices].mean(axis=1), 1e-12))
    return tuple(np.percentile(estimates, [2.5, 97.5]).tolist())


def fmt_pair(base: np.ndarray, deal: np.ndarray) -> str:
    if not len(base):
        return "N/A"
    base_sd = base.std(ddof=1) if len(base) > 1 else 0.0
    deal_sd = deal.std(ddof=1) if len(deal) > 1 else 0.0
    return f"{base.mean():.4f} +/- {base_sd:.4f} -> {deal.mean():.4f} +/- {deal_sd:.4f}"


def fmt_reduction(base: np.ndarray, deal: np.ndarray) -> str:
    """Format a paired mean reduction only when the archived metric is valid."""
    if not len(base) or not len(deal):
        return "N/A"
    base_mean = float(base.mean())
    deal_mean = float(deal.mean())
    if not np.isfinite(base_mean) or not np.isfinite(deal_mean) or abs(base_mean) < 1e-12:
        return "N/A"
    return f"{100.0 * (1.0 - deal_mean / base_mean):.1f}%"


def component_pages(resamples: int) -> list[tuple[str, plt.Figure]]:
    pages = []
    for task, label in TASKS:
        rows = []
        for model, model_label in MODELS:
            # The manuscript's aerodynamic SMART aggregate is paper-aligned;
            # these archived component rows come from an older evaluator.
            # Do not mix the two sources in a single diagnostic table.
            if task == "drivaerml" and model == "smart":
                continue
            source = read_case_rows(task, model)
            surface_base, surface_deal = paired_case_means(source, "surface_physical_rel_l2")
            volume_base, volume_deal = paired_case_means(source, "volume_physical_rel_l2")
            combined_base, combined_deal = paired_case_means(source, "combined_physical_rel_l2")
            if not len(combined_base):
                continue
            reduction = 100.0 * (1.0 - combined_deal.mean() / max(combined_base.mean(), 1e-12))
            low, high = bootstrap_interval(combined_base, combined_deal, 10_000 + len(rows), resamples)
            wins = 100.0 * np.mean(combined_deal < combined_base)
            rows.append([
                model_label,
                fmt_pair(surface_base, surface_deal),
                fmt_pair(volume_base, volume_deal),
                f"{reduction:.1f}% [{low:.1f}, {high:.1f}]",
                f"{wins:.1f}%",
            ])
        fig, ax = make_page(
            f"{label}: component error and paired uncertainty",
            "Surface and volume errors are averaged across the held-out spatial shifts and remeshing conditions. Intervals are paired bootstrap 95% intervals.",
        )
        draw_table(
            ax,
            rows,
            [
                "Surrogate",
                "Surface: Base -> DeAL",
                "Volume: Base -> DeAL",
                "Combined reduction [95% CI]",
                "Pairwise improvement rate",
            ],
            7.0,
        )
        pages.append((f"{task}_components_uncertainty", fig))
    return pages


def stability_pages() -> list[tuple[str, plt.Figure]]:
    pages = []
    for task, label in TASKS:
        rows = []
        for model, model_label in MODELS:
            # See the matching component table: do not combine the archived
            # aerodynamic SMART diagnostics with the paper-aligned aggregate.
            if task == "drivaerml" and model == "smart":
                continue
            source = read_case_rows(task, model)
            drift_base, drift_deal = paired_case_means(source, "combined_representation_drift_rel_gt")
            error_base, error_deal = paired_case_means(source, "combined_physical_rel_l2")
            if not len(error_base):
                continue
            rows.append([
                model_label,
                fmt_pair(drift_base, drift_deal),
                fmt_reduction(drift_base, drift_deal),
                fmt_reduction(error_base, error_deal),
            ])
        fig, ax = make_page(
            f"{label}: direct representation-stability statistics",
            "Prediction drift compares each changed representation with the same model's unshifted prediction at identical physical queries.",
        )
        draw_table(
            ax,
            rows,
            ["Surrogate", "Prediction drift: Base -> DeAL", "Drift reduction", "Field-error reduction"],
            7.5,
        )
        pages.append((f"{task}_representation_stability", fig))
    return pages


def geometry_rows() -> list[list[str]]:
    folders = {
        "DrivAerML": "drivaerml",
        "Pump": "pump",
        "Heat Exchanger": "heat_exchanger",
        "C-core": "c_core_magnetic",
    }
    all_rows: list[list[str]] = []
    for task, folder in folders.items():
        path = ROOT / "results/final/deal_comprehensive_evaluation/aligned_remesh_geometry" / folder / "remesh_geometry_per_case.csv"
        if not path.is_file():
            continue
        with path.open(newline="", encoding="utf-8") as handle:
            data = list(csv.DictReader(handle))
        grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
        for row in data:
            grouped[(row["method"], row["factor"])].append(row)
        for (method, factor), items in sorted(grouped.items()):
            chamfer = np.asarray([float(item["chamfer_mean_percent_bbox_diagonal"]) for item in items])
            hausdorff = np.asarray([
                100.0 * float(item["symmetric_hausdorff_p99_sampled"]) / max(float(item["bounding_box_diagonal"]), 1e-12)
                for item in items
            ])
            normal = np.asarray([float(item["normal_deviation_mean_degrees"]) for item in items])
            area = np.asarray([float(item["area_change_percent"]) for item in items])
            retained = np.asarray([
                100.0 * float(item["remesh_triangles"]) / max(float(item["source_triangles"]), 1.0)
                for item in items
            ])
            all_rows.append([
                task,
                {"feature": "Feature-aware", "quadric": "QEM", "voxel": "Voxel-grid"}.get(method, method),
                f"{factor}x",
                f"{chamfer.mean():.3f} +/- {chamfer.std(ddof=1) if len(chamfer) > 1 else 0.0:.3f}",
                f"{hausdorff.mean():.3f}",
                f"{normal.mean():.1f} +/- {normal.std(ddof=1) if len(normal) > 1 else 0.0:.1f}",
                f"{area.mean():.2f} +/- {area.std(ddof=1) if len(area) > 1 else 0.0:.2f}",
                f"{retained.mean():.1f}%",
            ])
    return all_rows


def geometry_pages() -> list[tuple[str, plt.Figure]]:
    rows = geometry_rows()
    columns = ["Task", "Remesher", "Reduction", "Chamfer (% bbox diag.)", "P99 distance (% bbox diag.)", "Normal deviation (deg.)", "Area change", "Triangles retained"]
    pages = []
    for index, task_pair in enumerate((("DrivAerML", "Pump"), ("Heat Exchanger", "C-core")), 1):
        subset = [row for row in rows if row[0] in task_pair]
        fig, ax = make_page(
            "Remesh geometry-preservation statistics",
            "Distances are normalized by the source bounding-box diagonal. Mean +/- SD is reported for Chamfer, normal deviation, and area change.",
        )
        draw_table(ax, subset, columns, 6.45)
        pages.append((f"remesh_fidelity_{index}", fig))
    return pages


def strategy_page(task: str, title: str, source: Path) -> tuple[str, plt.Figure] | None:
    if not source.is_file():
        return None
    with source.open(newline="", encoding="utf-8") as handle:
        raw = list(csv.DictReader(handle))
    rows = []
    model_columns = (
        ("base", "Base"),
        ("satloss", "DeAL"),
        ("downsample", "Uniform downsampling"),
        ("gaussian_ball_masked", "Gaussian-ball mask"),
        ("box_masked", "Box mask"),
    )
    by_category = {row["category"]: row for row in raw}
    for key, label in model_columns:
        table_row = [label]
        for category in ("sine_x_1", "sine_y_1", "remeshing_div5_mean", "remeshing_div10_mean"):
            item = by_category.get(category)
            if item is None or not item.get(f"{key}_mean"):
                table_row.append("N/A")
            else:
                table_row.append(f"{float(item[f'{key}_mean']):.4f} +/- {float(item[f'{key}_std']):.4f}")
        rows.append(table_row)
    fig, ax = make_page(
        f"{title}: matched training-view controls",
        "Values are normalized field error (mean +/- SD). The controls retain paired supervision while changing the geometry-view generator.",
    )
    draw_table(ax, rows, ["Training view", "Spatial shift x", "Spatial shift y", "Remesh 5x", "Remesh 10x"], 8.0)
    return f"{task}_strategy_controls", fig


def parse_markdown_table(path: Path) -> tuple[list[str], list[list[str]]]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.startswith("|")]
    if len(lines) < 3:
        raise ValueError(f"No markdown table in {path}")
    split = lambda line: [item.strip() for item in line.strip("|").split("|")]
    return split(lines[0]), [split(line) for line in lines[2:]]


def ablation_page(slug: str, title: str, path: Path) -> tuple[str, plt.Figure] | None:
    reference_rows = PAPER_REFERENCE_ABLATIONS.get(slug)
    if reference_rows is not None:
        rows = [[name, *(f"{value:.4f}" for value in values)] for name, values in reference_rows]
    else:
        if not path.is_file():
            return None
        headers, values = parse_markdown_table(path)
        index = {name: position for position, name in enumerate(headers)}
        required = ("sine_x_1.00", "sine_y_1.00")
        if not all(name in index for name in required):
            return None
        geometry = {
            factor: [name for name in headers if name.endswith(f"_div{factor}")]
            for factor in (5, 10)
        }
        rows = []
        for value in values:
            name = value[0].replace("SATLOSS", "DeAL")
            remesh = []
            for factor in (5, 10):
                source = [float(value[index[column]]) for column in geometry[factor]]
                remesh.append(float(np.mean(source)))
            rows.append([
                name,
                f"{float(value[index['sine_x_1.00']]):.4f}",
                f"{float(value[index['sine_y_1.00']]):.4f}",
                f"{remesh[0]:.4f}",
                f"{remesh[1]:.4f}",
            ])
    fig, ax = make_page(title, "Absolute normalized field error. Remesh columns average the feature-aware, QEM, and voxel-grid interventions.")
    draw_table(ax, rows, ["Training configuration", "Spatial shift x", "Spatial shift y", "Remesh 5x", "Remesh 10x"], 7.8)
    return slug, fig


def build_pages(resamples: int) -> list[tuple[str, plt.Figure]]:
    pages = []
    pages.extend(component_pages(resamples))
    pages.extend(stability_pages())
    pages.extend(geometry_pages())
    for task, title, source in (
        ("pump", "Pump", ROOT / "results/final/shift_pump_random1400_endpoint_strategies_v4_pool300_top3/combined_global_endpoint_absolute.csv"),
        ("heat_exchanger", "Heat Exchanger", ROOT / "results/final/heat_exchanger_endpoint_strategies_v4_validation32_top3/combined_global_endpoint_absolute.csv"),
    ):
        page = strategy_page(task, title, source)
        if page:
            pages.append(page)
    for slug, title, source in (
        ("density_range_ablation", "Density-range ablation", ROOT / "results/drivaerml_smart_satloss7_range_000till300_from_smart/range_ablation_combined_global_absolute_table.md"),
        ("kde_neighborhood_ablation", "KDE-neighborhood ablation", ROOT / "results/drivaerml_smart_satloss7_kde_ablation_vtp_25runs_corrected/kde_ablation_combined_global_absolute_table.md"),
    ):
        page = ablation_page(slug, title, source)
        if page:
            pages.append(page)
    return pages


def main() -> int:
    args = parse_args()
    if not args.main_pdf.is_file():
        raise FileNotFoundError(args.main_pdf)
    configure_style()
    pages = build_pages(args.bootstrap_resamples)
    if not pages:
        raise RuntimeError("No table pages were generated.")
    if args.dry_run:
        print("Validated table-only appendix pages: " + ", ".join(slug for slug, _ in pages))
        for _, figure in pages:
            plt.close(figure)
        return 0
    args.appendix_pdf.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(args.appendix_pdf) as pdf:
        for _, figure in pages:
            pdf.savefig(figure, bbox_inches="tight", pad_inches=0.12)
            plt.close(figure)
    merged = args.main_pdf.with_suffix(".merged.pdf")
    subprocess.run(["pdfunite", str(args.main_pdf), str(args.appendix_pdf), str(merged)], check=True)
    merged.replace(args.main_pdf)
    print(f"Appended {len(pages)} table-only pages to {args.main_pdf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
