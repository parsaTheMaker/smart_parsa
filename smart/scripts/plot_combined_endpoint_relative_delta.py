#!/usr/bin/env python3
"""Plot endpoint-only relative degradation from completed comparison results."""

from __future__ import annotations

import argparse
import csv
import textwrap
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from compare_drivaerml_sampling_invariance import MODEL_COLORS, MODEL_LABELS, MODEL_ORDER


DEFAULT_MODE_SPECS = {
    "beta": (
        "shifted_inverse_density_beta_0.00",
        "shifted_inverse_density_beta_1.00",
        "beta=1.00 relative to beta=0.00",
        "Inverse-density beta shift",
    ),
    "sine": (
        "ood_sine_y_mix_0.00",
        "ood_sine_y_mix_0.50",
        "sine=0.50 relative to sine=0.00",
        "Sinusoidal-y shift",
    ),
}

MODEL_PAIRS = (
    ("SMART", "SMART_SATLOSS6"),
    ("TRANSOLVERPP", "TRANSOLVERPP_SATLOSS6"),
    ("POINTNET2_SSG", "POINTNET2_SSG_SATLOSS6"),
    ("POINT_GNN", "POINT_GNN_SATLOSS6"),
    ("LNO", "LNO_SATLOSS6"),
    ("MSPT", "MSPT_SATLOSS6"),
)

CLASSICAL_SMART_COLORS = {
    "SMART": "#1F77B4",
    "SMART_SATLOSS6": "#9467BD",
    "SMART_DOWNSAMPLE": "#FF7F0E",
    "SMART_GAUSSIAN_BALL_MASKED": "#2CA02C",
    "SMART_BOX_MASKED": "#D62728",
}

MODEL_PAIR_COLORS = {
    "SMART": "#1F77B4",
    "TRANSOLVERPP": "#FF7F0E",
    "POINTNET2_SSG": "#17BECF",
    "POINT_GNN": "#8C564B",
    "LNO": "#D62728",
    "MSPT": "#2CA02C",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument(
        "--metric",
        default="combined_global_rel_l2",
        help="Metric column in per_run_mode_metrics.csv. Default: combined_global_rel_l2.",
    )
    parser.add_argument(
        "--extra-results-dir",
        action="append",
        default=[],
        type=Path,
        help="Completed comparison folder supplying an additional model's per-run metrics.",
    )
    parser.add_argument(
        "--extra-model",
        action="append",
        default=[],
        help="Model name to import from the corresponding --extra-results-dir.",
    )
    return parser.parse_args()


def load_rows(path: Path, metric: str):
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"run_id", "model_name", "sampling_mode", metric}
    missing = required.difference(rows[0].keys() if rows else set())
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    return rows


def endpoint_deltas(rows, metric: str, baseline_mode: str, endpoint_mode: str):
    values = defaultdict(dict)
    for row in rows:
        key = (row["model_name"], int(row["run_id"]))
        values[key][row["sampling_mode"]] = float(row[metric])

    by_model = defaultdict(list)
    for (model_name, _run_id), modes in values.items():
        if baseline_mode not in modes or endpoint_mode not in modes:
            continue
        baseline = modes[baseline_mode]
        endpoint = modes[endpoint_mode]
        denominator = max(abs(baseline), 1.0e-12)
        by_model[model_name].append(100.0 * (endpoint - baseline) / denominator)

    summaries = {}
    for model_name, model_values in by_model.items():
        values_array = np.asarray(model_values, dtype=np.float64)
        summaries[model_name] = {
            "values": values_array,
            "mean": float(values_array.mean()),
            "std": float(values_array.std(ddof=1)) if values_array.size > 1 else 0.0,
            "count": int(values_array.size),
        }
    return summaries


def wrap_label(label: str, width: int = 24) -> str:
    return "\n".join(textwrap.wrap(label, width=width, break_long_words=False, break_on_hyphens=False))


def display_label(model_name: str) -> str:
    return MODEL_LABELS.get(model_name, model_name).replace("SATLOSS6", "SATLOSS")


def model_pair_color(model_name: str, classical_smart: bool) -> str:
    if classical_smart and model_name in CLASSICAL_SMART_COLORS:
        return CLASSICAL_SMART_COLORS[model_name]
    for vanilla_name, satloss_name in MODEL_PAIRS:
        if model_name == satloss_name:
            return MODEL_PAIR_COLORS.get(vanilla_name, MODEL_COLORS.get(vanilla_name, "#4C78A8"))
        if model_name == vanilla_name:
            return MODEL_PAIR_COLORS.get(vanilla_name, MODEL_COLORS.get(vanilla_name, "#4C78A8"))
    return MODEL_COLORS.get(model_name, "#4C78A8")


def is_satloss(model_name: str) -> bool:
    return model_name.endswith("_SATLOSS6")


def ordered_models(summaries, paired: bool):
    if not paired:
        ordered = [model_name for model_name in MODEL_ORDER if model_name in summaries]
        ordered.sort(key=lambda name: summaries[name]["mean"])
        return ordered

    ordered = []
    for vanilla_name, satloss_name in MODEL_PAIRS:
        for model_name in (vanilla_name, satloss_name):
            if model_name in summaries:
                ordered.append(model_name)
    extras = [model_name for model_name in MODEL_ORDER if model_name in summaries and model_name not in ordered]
    return ordered + extras


def plot_endpoint_delta(
    summaries,
    output_path: Path,
    endpoint_description: str,
    show_std: bool,
    paired: bool,
    classical_smart: bool,
):
    ordered = ordered_models(summaries, paired=paired)
    if not ordered:
        raise ValueError("No complete baseline/endpoint pairs were found in the results.")

    means = np.asarray([summaries[name]["mean"] for name in ordered], dtype=np.float64)
    stds = np.asarray([summaries[name]["std"] for name in ordered], dtype=np.float64)
    labels = [wrap_label(display_label(name)) for name in ordered]
    colors = [model_pair_color(name, classical_smart=classical_smart) for name in ordered]
    hatches = ["///" if is_satloss(name) else "" for name in ordered]

    # Horizontal bars keep the model labels readable even with many variants.
    height = max(8.5, 0.68 * len(ordered) + 2.0)
    fig, ax = plt.subplots(figsize=(12.5, height), constrained_layout=True)
    positions = np.arange(len(ordered))
    errors = stds if show_std else None
    bars = ax.barh(
        positions,
        means,
        xerr=errors,
        color=colors,
        alpha=0.9,
        hatch=hatches,
        edgecolor="#333333",
        linewidth=0.5,
        capsize=4 if show_std else 0,
        error_kw={"elinewidth": 1.2, "ecolor": "#333333"},
    )
    ax.axvline(0.0, color="#222222", linewidth=1.1)
    ax.set_yticks(positions)
    ax.set_yticklabels(labels, fontsize=12)
    ax.tick_params(axis="x", labelsize=11)
    ax.tick_params(axis="y", length=0, pad=10)
    ax.set_xlabel("Relative change from baseline (%)", fontsize=14, labelpad=10)
    ax.set_title(f"Combined global error: {endpoint_description}", fontsize=19, pad=16)
    ax.grid(axis="x", linestyle=":", linewidth=0.8, alpha=0.55)
    ax.set_axisbelow(True)
    has_satloss = any(is_satloss(model_name) for model_name in summaries)
    if paired and (not classical_smart or has_satloss):
        ax.legend(
            handles=[
                Patch(facecolor="#888888", edgecolor="#333333", label="Vanilla"),
                Patch(facecolor="#888888", edgecolor="#333333", hatch="///", label="SATLOSS"),
            ],
            loc="lower right",
            fontsize=11,
            framealpha=0.9,
        )

    span = max(float(np.max(np.abs(means + stds))), 1.0)
    label_pad = max(0.8, 0.018 * span)
    for bar, mean, std, summary in zip(bars, means, stds, (summaries[name] for name in ordered)):
        x = float(bar.get_width())
        if x >= 0.0:
            text_x = x + label_pad
            ha = "left"
        else:
            text_x = x - label_pad
            ha = "right"
        text = f"{mean:+.1f}%"
        if show_std:
            text += f" +/- {std:.1f}"
        ax.text(text_x, bar.get_y() + bar.get_height() / 2.0, text, va="center", ha=ha, fontsize=10)

    lower, upper = ax.get_xlim()
    extra = max(label_pad * 5.0, 1.0)
    ax.set_xlim(lower - extra if lower < 0.0 else lower, upper + extra)
    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def write_summary(path: Path, summaries_by_shift):
    fieldnames = ["shift", "model_name", "model_label", "baseline_mode", "endpoint_mode", "mean_relative_delta_pct", "std_relative_delta_pct", "num_runs"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for shift, (summaries, baseline_mode, endpoint_mode) in summaries_by_shift.items():
            for model_name in MODEL_ORDER:
                summary = summaries.get(model_name)
                if summary is None:
                    continue
                writer.writerow(
                    {
                        "shift": shift,
                        "model_name": model_name,
                        "model_label": display_label(model_name),
                        "baseline_mode": baseline_mode,
                        "endpoint_mode": endpoint_mode,
                        "mean_relative_delta_pct": summary["mean"],
                        "std_relative_delta_pct": summary["std"],
                        "num_runs": summary["count"],
                    }
                )


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir.resolve()
    input_path = results_dir / "per_run_mode_metrics.csv"
    if not input_path.is_file():
        raise FileNotFoundError(f"Could not find completed per-run metrics: {input_path}")

    rows = load_rows(input_path, args.metric)
    if len(args.extra_results_dir) != len(args.extra_model):
        raise ValueError("Each --extra-results-dir must have one matching --extra-model.")
    for extra_dir, model_name in zip(args.extra_results_dir, args.extra_model):
        extra_path = extra_dir.resolve() / "per_run_mode_metrics.csv"
        if not extra_path.is_file():
            raise FileNotFoundError(f"Could not find extra per-run metrics: {extra_path}")
        extra_rows = load_rows(extra_path, args.metric)
        selected_rows = [row for row in extra_rows if row["model_name"] == model_name]
        if not selected_rows:
            raise ValueError(f"Model {model_name!r} was not found in {extra_path}")
        if any(row["model_name"] == model_name for row in rows):
            raise ValueError(f"Model {model_name!r} already exists in the primary results.")
        rows.extend(selected_rows)
        print(f"Imported {len(selected_rows)} rows for {model_name} from {extra_dir}")
    summaries_by_shift = {}
    for shift, (baseline_mode, endpoint_mode, endpoint_description, axis_description) in DEFAULT_MODE_SPECS.items():
        summaries = endpoint_deltas(rows, args.metric, baseline_mode, endpoint_mode)
        classical_smart = set(summaries).issubset(set(CLASSICAL_SMART_COLORS)) and len(summaries) > 1
        summaries_by_shift[shift] = (summaries, baseline_mode, endpoint_mode)
        plot_endpoint_delta(
            summaries,
            results_dir / f"all_models_combined_global_relative_delta_{shift}_mean_only.png",
            endpoint_description,
            show_std=False,
            paired=False,
            classical_smart=classical_smart,
        )
        plot_endpoint_delta(
            summaries,
            results_dir / f"all_models_combined_global_relative_delta_{shift}_with_std.png",
            endpoint_description,
            show_std=True,
            paired=False,
            classical_smart=classical_smart,
        )
        plot_endpoint_delta(
            summaries,
            results_dir / f"all_models_combined_global_relative_delta_{shift}_paired_mean_only.png",
            endpoint_description,
            show_std=False,
            paired=True,
            classical_smart=classical_smart,
        )
        plot_endpoint_delta(
            summaries,
            results_dir / f"all_models_combined_global_relative_delta_{shift}_paired_with_std.png",
            endpoint_description,
            show_std=True,
            paired=True,
            classical_smart=classical_smart,
        )
        print(
            f"{shift}: baseline={baseline_mode}, endpoint={endpoint_mode}, "
            f"models={len(summaries)}, axis={axis_description}"
        )

    write_summary(results_dir / "all_models_combined_global_endpoint_relative_delta.csv", summaries_by_shift)
    print(f"Saved endpoint plots and summary to {results_dir}")


if __name__ == "__main__":
    main()
