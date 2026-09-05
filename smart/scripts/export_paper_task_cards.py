#!/usr/bin/env python3
"""Create concise, source-grounded task cards for the supplementary material."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


TASKS = {
    "pump": {
        "title": "Pump flow",
        "root": "/mnt/data/parsa/shift_pump_random1400_preprocessed",
        "manifest": "preprocessed_manifest.json",
        "geometry": "surface_coords.npy",
        "surface_fields": ["pressure", "velocity_x", "velocity_y", "velocity_z", "wall_shear_x", "wall_shear_y", "wall_shear_z"],
        "volume_fields": ["pressure", "velocity_x", "velocity_y", "velocity_z"],
        "description": "Steady three-dimensional pump-flow surrogate. Geometry and operating-condition parameters are inputs; surface and volume flow fields are targets.",
    },
    "heat_exchanger": {
        "title": "Nonlinear heat exchanger",
        "root": "/mnt/ssdraid/parsa/toy_heat_exchange_fem_v1",
        "manifest": "preprocessed_manifest.json",
        "geometry": "geometry_coords.npy",
        "surface_fields": ["outward_heat_flux"],
        "volume_fields": ["temperature"],
        "description": "Deterministic nonlinear steady conduction with fixed hot interior-channel boundaries and nonlinear convection-radiation on the exterior boundary.",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    parser.add_argument("--output-latex", type=Path, required=True)
    return parser.parse_args()


def case_dir(root: Path, case_id: int) -> Path:
    candidates = (root / f"case_{case_id:05d}", root / f"run_{case_id}")
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"No case folder for id={case_id} under {root}")


def escape_latex(value: str) -> str:
    return value.replace("_", "\\_")


def main() -> int:
    args = parse_args()
    cards = []
    for key, definition in TASKS.items():
        root = Path(definition["root"])
        manifest_path = root / definition["manifest"]
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Missing {key} manifest: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        train_ids = [int(value) for value in manifest.get("train_ids", [])]
        test_ids = [int(value) for value in manifest.get("test_ids", manifest.get("validation_ids", []))]
        if not train_ids or not test_ids:
            raise ValueError(f"{key} manifest has no non-empty train/test or validation split.")
        first = case_dir(root, train_ids[0])
        geometry = np.load(first / definition["geometry"], mmap_mode="r")
        surface = np.load(first / "surface_data.npy", mmap_mode="r")
        volume = np.load(first / "volume_data.npy", mmap_mode="r")
        cards.append(
            {
                "id": key,
                "title": definition["title"],
                "root": str(root.resolve()),
                "description": definition["description"],
                "train_case_count": len(train_ids),
                "held_out_case_count": len(test_ids),
                "geometry_points_first_case": int(geometry.shape[0]),
                "surface_channels": list(definition["surface_fields"]),
                "volume_channels": list(definition["volume_fields"]),
                "surface_points_first_case": int(surface.shape[0]),
                "volume_points_first_case": int(volume.shape[0]),
            }
        )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps({"tasks": cards}, indent=2) + "\n", encoding="utf-8")
    markdown = ["# Secondary Task Cards", ""]
    latex = ["\\begin{tabular}{p{0.18\\linewidth}p{0.18\\linewidth}p{0.24\\linewidth}p{0.30\\linewidth}}", "\\toprule", "Task & Split & Inputs & Targets \\\\ ", "\\midrule"]
    for card in cards:
        markdown.extend(
            [
                f"## {card['title']}",
                card["description"],
                f"- Split: {card['train_case_count']} train / {card['held_out_case_count']} held out.",
                f"- Geometry cloud: {card['geometry_points_first_case']:,} points in the first stored case.",
                f"- Surface targets: {', '.join(card['surface_channels'])}.",
                f"- Volume targets: {', '.join(card['volume_channels'])}.",
                "",
            ]
        )
        latex.append(
            f"{escape_latex(card['title'])} & {card['train_case_count']} train / {card['held_out_case_count']} held out & "
            f"surface geometry ({card['geometry_points_first_case']:,} points in example case) & "
            f"surface: {escape_latex(', '.join(card['surface_channels']))}; volume: {escape_latex(', '.join(card['volume_channels']))} \\\\ "
        )
    latex.extend(["\\bottomrule", "\\end{tabular}"])
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.write_text("\n".join(markdown), encoding="utf-8")
    args.output_latex.parent.mkdir(parents=True, exist_ok=True)
    args.output_latex.write_text("\n".join(latex) + "\n", encoding="utf-8")
    print(f"Wrote task cards: {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
