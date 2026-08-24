#!/usr/bin/env python3
"""Regenerate consistency-ablation figures from saved aggregate CSV files only."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path
from typing import Dict, List


SCRIPT_DIR = Path(__file__).resolve().parent
ENGINE_PATH = SCRIPT_DIR / "compare_drivaerml_satloss7_range_ablation.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, help="Existing comparison result directory.")
    parser.add_argument(
        "--font-scale",
        type=float,
        required=True,
        help="Replacement plot font scale. No inference or metric aggregation is performed.",
    )
    parser.add_argument(
        "--with-std",
        action="store_true",
        help="Also write a separate compact endpoint summary with across-run standard-deviation error bars.",
    )
    return parser.parse_args()


def load_engine():
    spec = importlib.util.spec_from_file_location("consistency_ablation_replot_engine", ENGINE_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load comparison engine from {ENGINE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_csv_rows(path: Path) -> List[Dict[str, object]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = []
        for raw_row in csv.DictReader(handle):
            row: Dict[str, object] = {}
            for key, value in raw_row.items():
                if value is None:
                    row[key] = value
                    continue
                try:
                    row[key] = float(value)
                except ValueError:
                    row[key] = value
            rows.append(row)
    return rows


def main() -> None:
    args = parse_args()
    if args.font_scale <= 0.0:
        raise ValueError("--font-scale must be positive.")
    output_dir = Path(args.output_dir).expanduser().resolve()
    metadata_path = output_dir / "comparison_metadata.json"
    aggregate_path = output_dir / "aggregate_metrics.csv"
    percentage_path = output_dir / "aggregate_percentage_worsening.csv"
    for path in (metadata_path, aggregate_path, percentage_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required saved comparison artifact is missing: {path}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    aggregate = load_csv_rows(aggregate_path)
    percentage_aggregate = load_csv_rows(percentage_path)
    engine = load_engine()
    engine.MODEL_ORDER = tuple(str(name) for name in metadata["models"])
    engine.MODEL_LABELS = {str(key): str(value) for key, value in metadata["labels"].items()}
    engine.MODEL_COLORS = {
        "SMART": "#4C78A8",
        "SMART_SATLOSS7_RANGE025": "#7A5195",
        "SMART_SATLOSS7_RANGE050": "#F28E2B",
    }
    engine.REFERENCE_MODEL = "SMART"
    engine.REFERENCE_MODEL_LABEL = "SMART baseline"
    engine.ABLATION_PREFIX = "consistency_ablation"
    engine.ABLATION_TABLE_TITLE = "SMART consistency-loss ablation"
    engine._COMPUTE_PLOT_STD = False
    engine.configure_plot_style(float(args.font_scale))

    levels = {
        str(shift): [float(value) for value in values]
        for shift, values in metadata["shift_levels"].items()
    }
    active_shifts = [str(shift) for shift in metadata["active_shifts"]]
    geometry_source_modes = [f"geometry_{source}" for source in metadata["active_geometry_sources"]]
    y_pad_fraction = float(metadata["y_pad_fraction"])
    plot_scales = [str(scale) for scale in metadata["plot_scales"]]

    for shift in active_shifts:
        endpoint = float(max(levels[shift]))
        for metric in engine.METRIC_KEYS:
            for plot_scale in plot_scales:
                engine.plot_endpoint_bars(
                    aggregate,
                    metric,
                    shift,
                    endpoint,
                    output_dir / f"{shift}_{metric}_endpoint_absolute_{plot_scale}.png",
                    percentage=False,
                    baseline_aggregate=aggregate,
                    percentage_reference_aggregate=percentage_aggregate,
                    log_scale=plot_scale == "log",
                    show_std=False,
                    y_pad_fraction=y_pad_fraction,
                )
            engine.plot_endpoint_bars(
                percentage_aggregate,
                metric,
                shift,
                endpoint,
                output_dir / f"{shift}_{metric}_endpoint_percentage_worsening.png",
                percentage=True,
                baseline_aggregate=aggregate,
                show_std=False,
                y_pad_fraction=y_pad_fraction,
            )

    for method in ("angle", "isotropic", "voxel"):
        method_modes = [
            source for source in geometry_source_modes
            if source.removeprefix("geometry_").startswith(f"{method}_")
        ]
        if not method_modes:
            continue
        for log_scale, scale_slug in ((True, "log"), (False, "linear")):
            engine.plot_combined_geometry_source_bars(
                aggregate,
                percentage_aggregate,
                method_modes,
                output_dir / f"consistency_ablation_combined_global_endpoint_bars_{method}_{scale_slug}.png",
                f"Combined global error ({method.title()} remeshing, {scale_slug} scale)",
                log_scale=log_scale,
                percentage=False,
                show_std=False,
                y_pad_fraction=y_pad_fraction,
            )
        engine.plot_combined_geometry_source_bars(
            aggregate,
            percentage_aggregate,
            method_modes,
            output_dir / f"consistency_ablation_combined_global_relative_vs_smart_{method}.png",
            f"Combined global relative difference from SMART baseline ({method.title()} remeshing)",
            log_scale=False,
            percentage=True,
            show_std=False,
            y_pad_fraction=y_pad_fraction,
        )

    average_absolute = engine.average_remeshing_rows(aggregate, geometry_source_modes, percentage=False)
    average_percentage = engine.average_remeshing_rows(percentage_aggregate, geometry_source_modes, percentage=True)
    average_modes = sorted({str(row["shift_name"]) for row in average_absolute})
    if average_modes:
        engine.plot_combined_geometry_source_bars(
            average_absolute,
            average_percentage,
            average_modes,
            output_dir / "consistency_ablation_combined_global_endpoint_bars_remeshing_average_linear.png",
            "Combined global error (average of angle/isotropic/voxel remeshing, linear scale)",
            log_scale=False,
            percentage=False,
            show_std=False,
            y_pad_fraction=y_pad_fraction,
        )

    if bool(metadata.get("compact_endpoint_summary", False)):
        engine.plot_compact_endpoint_summary(
            aggregate,
            percentage_aggregate,
            levels,
            geometry_source_modes,
            output_dir / "consistency_ablation_combined_global_endpoint_summary_linear.png",
            y_pad_fraction,
        )
        if args.with_std:
            engine.plot_compact_endpoint_summary(
                aggregate,
                percentage_aggregate,
                levels,
                geometry_source_modes,
                output_dir / "consistency_ablation_combined_global_endpoint_summary_linear_with_std.png",
                y_pad_fraction,
                show_std=True,
            )

    metadata["font_scale"] = float(args.font_scale)
    metadata["plots_regenerated_from_saved_metrics"] = True
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Regenerated plots from saved metrics with font_scale={args.font_scale:.2f}: {output_dir}")


if __name__ == "__main__":
    main()
