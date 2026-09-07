#!/usr/bin/env python3
"""Collect the limited-five paired audits into one complete table."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


EXPECTED = {
    "drivaerml": ["ab_upt", "geo_fno"],
    "heat_exchanger": ["ab_upt", "geo_fno"],
    "pump": ["ab_upt", "geo_fno", "lno", "mspt", "pointnet2_ssg", "point_transformer_v3", "transolverpp"],
    "c_core": ["smart", "ab_upt", "geo_fno", "lno", "mspt", "pointnet2_ssg", "point_transformer_v3", "transolverpp"],
}
CONDITIONS = ("sine_x", "sine_y", "remesh_mean_div5", "remesh_mean_div10")
LABELS = {
    "drivaerml": "DrivAerML", "heat_exchanger": "Heat Exchanger", "pump": "Pump", "c_core": "C-core",
    "smart": "SMART", "ab_upt": "AB-UPT", "geo_fno": "Geo-FNO", "lno": "LNO", "mspt": "MSPT",
    "pointnet2_ssg": "PointNet++", "point_transformer_v3": "Point Transformer V3", "transolverpp": "Transolver++",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for dataset, models in EXPECTED.items():
        for model in models:
            path = args.root / dataset / model / "summary.json"
            values = {}
            case_ids = "N/A"
            if path.is_file():
                payload = json.loads(path.read_text(encoding="utf-8"))
                case_ids = ",".join(map(str, payload["case_ids"]))
                values = {
                    item["condition"]: item for item in payload["summary"]
                    if item["metric"] == "combined_physical_rel_l2"
                }
            for condition in CONDITIONS:
                item = values.get(condition)
                rows.append({
                    "dataset": LABELS[dataset], "model": LABELS[model], "condition": condition,
                    "case_ids": case_ids,
                    "base_mean": "N/A" if item is None else f"{item['base_mean']:.8f}",
                    "base_std": "N/A" if item is None else f"{item['base_std']:.8f}",
                    "deal_mean": "N/A" if item is None else f"{item['deal_mean']:.8f}",
                    "deal_std": "N/A" if item is None else f"{item['deal_std']:.8f}",
                    "deal_improvement_percent": "N/A" if item is None else f"{item['deal_improvement_percent']:.3f}",
                })
    args.root.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with (args.root / "base_vs_deal_limited5_table.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    lines = ["| Dataset | Model | Condition | Base | DeAL | Improvement |", "|---|---|---|---:|---:|---:|"]
    for row in rows:
        improvement = row["deal_improvement_percent"]
        improvement = improvement if improvement == "N/A" else f"{float(improvement):+.1f}%"
        base = row["base_mean"] if row["base_mean"] == "N/A" else f"{float(row['base_mean']):.4f}"
        deal = row["deal_mean"] if row["deal_mean"] == "N/A" else f"{float(row['deal_mean']):.4f}"
        lines.append(f"| {row['dataset']} | {row['model']} | {row['condition']} | {base} | {deal} | {improvement} |")
    (args.root / "base_vs_deal_limited5_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
