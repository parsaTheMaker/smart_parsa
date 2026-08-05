#!/usr/bin/env python3
"""Create a clearly labeled illustrative, non-measured comparison plot."""

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


FAMILIES = [
    ("SMART", "SMART", "#1F77B4"),
    ("TransolverPP", "TransolverPP", "#FF7F0E"),
    ("PointNet++-SSG", "PointNet++-SSG", "#17BECF"),
    ("LNO", "LNO", "#D62728"),
    ("MSPT", "MSPT", "#2CA02C"),
    ("PointTransformerV3", "PointTransformerV3", "#7F3C8D"),
]
BASE_FONT_SIZE = 15.0
FONT_SCALE = 1.2


def font_size(multiplier: float = 1.0) -> float:
    return 0.55 * BASE_FONT_SIZE * FONT_SCALE * float(multiplier)


def set_reference_limits(ax, values) -> None:
    finite = np.asarray([float(value) for value in values if float(value) > 0.0], dtype=float)
    low = float(finite.min())
    high = float(finite.max())
    span = max(high - low, high, 1.0e-12)
    ax.set_ylim(max(0.0, low - 0.10 * span), high + 0.10 * span)


def relative_improvement(base: float, satloss: float) -> float:
    return 100.0 * (base - satloss) / max(abs(base), 1.0e-12)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-summary", required=True)
    parser.add_argument("--output-plot", required=True)
    parser.add_argument("--output-data", required=True)
    args = parser.parse_args()

    measured = {}
    with Path(args.input_summary).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["scope"] != "positive_first_consensus_top10":
                continue
            measured[row["model"]] = {
                "base": float(row["vanilla8_combined_global_rel_l2"]),
                "satloss": float(row["satloss8_combined_global_rel_l2"]),
            }

    required = {label for _, label, _ in FAMILIES}
    missing = sorted(required.difference(measured))
    if missing:
        raise ValueError(f"Missing measured summary rows: {missing}")

    data = {label: dict(measured[label]) for _, label, _ in FAMILIES}
    # Synthetic targets requested by the user. All other bars remain measured.
    data["SMART"]["satloss"] = data["SMART"]["base"] * (1.0 - 0.002)
    data["PointTransformerV3"]["base"] = data["PointTransformerV3"]["satloss"] / (1.0 - 0.013)
    for label in data:
        data[label]["relative_improvement_percent"] = relative_improvement(
            data[label]["base"], data[label]["satloss"]
        )
        data[label]["source"] = "synthetic target" if label in {"SMART", "PointTransformerV3"} else "measured"

    output_data = Path(args.output_data)
    output_data.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")

    labels = [label for _, label, _ in FAMILIES]
    colors = [color for _, _, color in FAMILIES]
    x = np.arange(len(labels), dtype=float)
    width = 0.22
    offsets = {"base": -0.13, "satloss": 0.13}
    fig, ax = plt.subplots(figsize=(13.4, 7.4))
    fig.subplots_adjust(left=0.12, right=0.78, bottom=0.29, top=0.84)
    base_values = [data[label]["base"] for label in labels]
    satloss_values = [data[label]["satloss"] for label in labels]
    ax.bar(
        x + offsets["base"],
        base_values,
        width=width,
        color=colors,
        edgecolor="#222222",
        linewidth=0.8,
        alpha=0.96,
        label="base",
    )
    satloss_bars = ax.bar(
        x + offsets["satloss"],
        satloss_values,
        width=width,
        color=colors,
        edgecolor="#222222",
        linewidth=0.8,
        hatch="///",
        alpha=0.96,
        label="satloss",
    )
    all_values = base_values + satloss_values
    span = max(max(all_values) - min(all_values), max(all_values), 1.0e-6)
    for bar, label in zip(satloss_bars, labels):
        value = data[label]["relative_improvement_percent"]
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.025 * span,
            f"{value:+.1f}%",
            ha="center",
            va="bottom",
            fontsize=font_size(0.78),
            rotation=90 if abs(value) >= 100.0 else 0,
            clip_on=False,
        )

    ax.set_title(
        "Illustrative synthetic scenario generated on request; not measured results\n"
        "Requested targets: SMART satloss +0.2%, PointTransformerV3 satloss +1.3%",
        fontsize=font_size(),
        pad=12,
    )
    ax.set_ylabel("Combined-global relative L2 error", fontsize=font_size())
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=24, ha="right", fontsize=font_size())
    ax.tick_params(axis="both", labelsize=font_size())
    ax.grid(axis="y", alpha=0.20, linewidth=0.8)
    ax.set_axisbelow(True)
    set_reference_limits(ax, all_values)
    fig.legend(
        handles=[
            Patch(facecolor="#777777", edgecolor="#222222", label="base"),
            Patch(facecolor="#777777", edgecolor="#222222", hatch="///", label="satloss"),
        ],
        loc="upper left",
        bbox_to_anchor=(0.81, 0.77),
        framealpha=0.94,
        fontsize=font_size(),
    )
    fig.legend(
        handles=[Patch(facecolor=color, edgecolor="#222222", label=label) for _, label, color in FAMILIES],
        loc="upper left",
        bbox_to_anchor=(0.81, 0.49),
        framealpha=0.94,
        fontsize=font_size(),
    )
    fig.savefig(args.output_plot, dpi=280, bbox_inches="tight", pad_inches=0.16)
    plt.close(fig)
    print(f"Wrote illustrative plot: {args.output_plot}")
    print(f"Wrote synthetic values: {args.output_data}")


if __name__ == "__main__":
    main()
