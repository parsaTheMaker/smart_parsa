#!/usr/bin/env python3
"""Replot the compact SHIFT-Submarine point-cloud shift summary.

This reads an existing comparison CSV, so the summary can be regenerated
without loading checkpoints or repeating model inference.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from compare_shift_submarine_sampling_invariance import (
    build_shift_endpoint_summary_groups,
    plot_shift_endpoint_summary,
    write_shift_endpoint_summary_table,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-csv", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--beta-endpoint", type=float, default=1.0)
    parser.add_argument("--font-scale", type=float, default=1.0)
    parser.add_argument("--plot-scales", default="linear,log")
    parser.add_argument("--with-std", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = (args.output_dir or args.metrics_csv.parent).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    with args.metrics_csv.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No rows found in {args.metrics_csv}")
    groups = build_shift_endpoint_summary_groups(rows, (0.0, float(args.beta_endpoint)))
    if not groups:
        raise ValueError("No requested endpoint or remeshing rows were found in the metrics CSV.")
    write_shift_endpoint_summary_table(groups, output_dir)
    for scale in (item.strip() for item in str(args.plot_scales).split(",")):
        if not scale:
            continue
        plot_shift_endpoint_summary(
            groups,
            output_dir / f"shift_submarine_combined_global_shift_endpoint_summary_bars_{scale}.png",
            scale,
            args.font_scale,
            no_std=not args.with_std,
        )
    print(f"Wrote SHIFT-Submarine endpoint summary to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
