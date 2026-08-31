#!/usr/bin/env python3
"""Create a compact SMART-versus-DeAL comparison for Pump and Heat Exchanger.

The script consumes completed evaluation summaries only: it never reruns model
inference.  Pump remeshing entries use its feature-aware source at each
decimation factor, while the Heat Exchanger study contains QEM remeshes only.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


DEFAULT_PUMP_DIR = Path("/home/parsa/smart_parsa/results/final/shift_pump_endpoint_strategies_v4_pool100_top2")
DEFAULT_HEAT_DIR = Path(
    "/home/parsa/smart_parsa/results/final/heat_exchanger_all_models_deal_qem_pool100_top3_pointnet2_remesh_ranking"
)
DEFAULT_OUTPUT_DIR = Path("/home/parsa/smart_parsa/results/final/pump_heat_exchanger_smart_deal_summary")

TESTS = (
    ("sine_x", "Sine-x"),
    ("sine_y", "Sine-y"),
    ("remeshing_div5_mean", "Mean remesh div5"),
    ("remeshing_div10_mean", "Mean remesh div10"),
)
PUMP_BLUE = "#1598E6"
HEAT_GREEN = "#20C875"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pump-dir", type=Path, default=DEFAULT_PUMP_DIR)
    parser.add_argument("--heat-exchanger-dir", type=Path, default=DEFAULT_HEAT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--font-scale", type=float, default=1.2)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Expected completed-study table is missing: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_pump_values(study_dir: Path) -> dict[str, dict[str, float]]:
    rows = read_csv(study_dir / "combined_global_endpoint_absolute.csv")
    by_category = {row["category"]: row for row in rows}
    remesh_rows = read_csv(study_dir / "combined_global_remeshing_sources_absolute.csv")
    by_source = {row["source"]: row for row in remesh_rows}
    category_map = {
        "sine_x": "sine_x_1",
        "sine_y": "sine_y_1",
        "remeshing_div5_mean": "remeshing_div5_mean",
        "remeshing_div10_mean": "remeshing_div10_mean",
    }
    values: dict[str, dict[str, float]] = {}
    for key, _ in TESTS:
        if key.startswith("remeshing_div"):
            factor = key.removeprefix("remeshing_div").removesuffix("_mean")
            row = by_source.get(f"angle_div{factor}")
        else:
            row = by_category.get(category_map[key])
        if row is None:
            raise KeyError(f"Pump summary is missing category '{key}'.")
        values[key] = {"base": float(row["base_mean"]), "deal": float(row["satloss_mean"])}
    return values


def load_heat_values(study_dir: Path) -> dict[str, dict[str, float]]:
    rows = read_csv(study_dir / "aggregate_metrics.csv")
    by_key = {(row["model_name"], row["sampling_mode"]): float(row["combined_global_rel_l2"]) for row in rows}
    mode_map = {
        "sine_x": "sine_x",
        "sine_y": "sine_y",
        # This study evaluated the QEM source only, so each factor mean is its
        # QEM aggregate rather than an undocumented cross-remesher average.
        "remeshing_div5_mean": "isotropic_div5",
        "remeshing_div10_mean": "isotropic_div10",
    }
    values: dict[str, dict[str, float]] = {}
    for key, _ in TESTS:
        mode = mode_map[key]
        try:
            values[key] = {"base": by_key[("SMART", mode)], "deal": by_key[("SMART_SATLOSS7", mode)]}
        except KeyError as exc:
            raise KeyError(f"Heat Exchanger summary is missing SMART/DeAL metric for '{mode}'.") from exc
    return values


def improvement_percent(base: float, deal: float) -> float:
    return 100.0 * (base - deal) / max(abs(base), 1.0e-12)


def add_improvement_label(ax, bar, base: float, deal: float, font_size: float) -> None:
    value = improvement_percent(base, deal)
    # The y-padding reserves headroom, so labels are anchored to the actual
    # DeAL bar rather than an invisible standard-deviation offset.
    label = f"{value:+.1f}%"
    ax.annotate(
        label,
        (bar.get_x() + 0.5 * bar.get_width(), deal),
        xytext=(0, 4),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=font_size,
        fontweight="semibold",
        rotation=90,
        clip_on=False,
    )


def plot(values: dict[str, dict[str, dict[str, float]]], output_path: Path, font_scale: float) -> None:
    font = 10.0 * font_scale
    x = np.arange(len(TESTS), dtype=np.float64)
    width = 0.17
    offsets = {"pump_base": -1.5 * width, "pump_deal": -0.5 * width, "heat_base": 0.5 * width, "heat_deal": 1.5 * width}
    fig, ax = plt.subplots(figsize=(12.6, 6.35))
    fig.subplots_adjust(left=0.12, right=0.985, bottom=0.20, top=0.87)

    all_values: list[float] = []
    for index, (test_key, _) in enumerate(TESTS):
        pump = values["pump"][test_key]
        heat = values["heat_exchanger"][test_key]
        all_values.extend((pump["base"], pump["deal"], heat["base"], heat["deal"]))
        ax.bar(
            x[index] + offsets["pump_base"],
            pump["base"],
            width,
            color=PUMP_BLUE,
            edgecolor="black",
            linewidth=0.75,
            zorder=3,
        )
        pump_deal = ax.bar(
            x[index] + offsets["pump_deal"],
            pump["deal"],
            width,
            color=PUMP_BLUE,
            edgecolor="black",
            linewidth=0.75,
            hatch="///",
            zorder=3,
        )[0]
        ax.bar(
            x[index] + offsets["heat_base"],
            heat["base"],
            width,
            color=HEAT_GREEN,
            edgecolor="black",
            linewidth=0.75,
            zorder=3,
        )
        heat_deal = ax.bar(
            x[index] + offsets["heat_deal"],
            heat["deal"],
            width,
            color=HEAT_GREEN,
            edgecolor="black",
            linewidth=0.75,
            hatch="///",
            zorder=3,
        )[0]
        add_improvement_label(ax, pump_deal, pump["base"], pump["deal"], font * 0.88)
        add_improvement_label(ax, heat_deal, heat["base"], heat["deal"], font * 0.88)

    ymax = max(all_values)
    ax.set_ylim(0.0, ymax * 1.17)
    ax.set_xticks(x, [label for _, label in TESTS])
    ax.set_ylabel("Combined global relative L2", fontsize=font + 1)
    ax.yaxis.set_label_coords(-0.085, 0.5)
    ax.tick_params(axis="both", labelsize=font)
    ax.grid(axis="y", alpha=0.23, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.set_title("SMART versus DeAL under sampling and remeshing shifts", fontsize=font + 2, pad=11)
    ax.legend(
        handles=[
            Patch(facecolor=PUMP_BLUE, edgecolor="black", label="Pump"),
            Patch(facecolor=HEAT_GREEN, edgecolor="black", label="Heat Exchanger"),
            Patch(facecolor="white", edgecolor="black", label="Base (no hatch)"),
            Patch(facecolor="white", edgecolor="black", hatch="///", label="DeAL (hatch)"),
            Patch(facecolor="none", edgecolor="none", label="DeAL labels: improvement vs base"),
        ],
        ncol=5,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        frameon=False,
        fontsize=font * 0.85,
        handlelength=1.45,
        columnspacing=1.3,
    )
    fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def write_summary(values: dict[str, dict[str, dict[str, float]]], output_path: Path) -> None:
    rows: list[dict[str, str | float]] = []
    for test_key, label in TESTS:
        for dataset, dataset_label in (("pump", "Pump"), ("heat_exchanger", "Heat Exchanger")):
            entry = values[dataset][test_key]
            rows.append(
                {
                    "dataset": dataset_label,
                    "test": label,
                    "base_combined_global_rel_l2": entry["base"],
                    "deal_combined_global_rel_l2": entry["deal"],
                    "deal_improvement_vs_base_percent": improvement_percent(entry["base"], entry["deal"]),
                }
            )
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.font_scale <= 0:
        raise ValueError("--font-scale must be positive.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    values = {
        "pump": load_pump_values(args.pump_dir),
        "heat_exchanger": load_heat_values(args.heat_exchanger_dir),
    }
    summary_path = args.output_dir / "pump_heat_exchanger_smart_deal_endpoint_summary.csv"
    plot_path = args.output_dir / "pump_heat_exchanger_smart_deal_endpoint_summary_linear.png"
    write_summary(values, summary_path)
    plot(values, plot_path, args.font_scale)
    metadata = {
        "pump_source": str(args.pump_dir),
        "heat_exchanger_source": str(args.heat_exchanger_dir),
        "remeshing_definition": {
            "pump": "feature-aware remesh metric for each factor; plot labels are retained by request",
            "heat_exchanger": "QEM remesh aggregate for each factor; it is the sole completed remesher in this study",
        },
        "plot": str(plot_path),
    }
    (args.output_dir / "pump_heat_exchanger_smart_deal_endpoint_summary.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {plot_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
