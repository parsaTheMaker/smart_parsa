#!/usr/bin/env python3
"""Export resolved configurations and checkpoint provenance for the paper.

This utility writes machine-readable evidence plus a compact LaTeX table.  It
never changes checkpoints or source configurations; it records exactly the
artifacts supplied on the command line.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import torch
from omegaconf import OmegaConf

from compare_shift_endpoint_strategies import compose_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--latex-output", type=Path, required=True)
    parser.add_argument(
        "--config",
        action="append",
        default=[],
        metavar="LABEL=CONFIG_NAME",
        help="Repeatable resolved Hydra configuration to archive.",
    )
    parser.add_argument(
        "--checkpoint",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="Repeatable checkpoint to checksum and inspect.",
    )
    return parser.parse_args()


def parse_mapping(values: list[str], flag: str) -> dict[str, str]:
    parsed = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"{flag} must use LABEL=VALUE, got {value!r}")
        label, item = value.split("=", 1)
        if not label or not item or label in parsed:
            raise ValueError(f"Invalid or duplicate {flag}: {value!r}")
        parsed[label] = item
    return parsed


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_revision(root: Path) -> str | None:
    try:
        return subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def latex_escape(value: str) -> str:
    return value.replace("_", "\\_")


def main() -> int:
    args = parse_args()
    configs = parse_mapping(args.config, "--config")
    checkpoints = parse_mapping(args.checkpoint, "--checkpoint")
    if not configs or not checkpoints:
        raise ValueError("Provide at least one --config and one --checkpoint.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    resolved_dir = args.output_dir / "resolved_configs"
    resolved_dir.mkdir(parents=True, exist_ok=True)

    config_records = {}
    for label, name in configs.items():
        cfg = compose_config(name)
        path = resolved_dir / f"{label}.yaml"
        OmegaConf.save(cfg, path, resolve=True)
        config_records[label] = {"config_name": name, "resolved_config": str(path.resolve())}

    checkpoint_records = {}
    for label, raw_path in checkpoints.items():
        path = Path(raw_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Missing checkpoint for {label}: {path}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        checkpoint_records[label] = {
            "path": str(path),
            "sha256": sha256(path),
            "size_bytes": path.stat().st_size,
            "epoch": payload.get("epoch") if isinstance(payload, dict) else None,
            "model_name": payload.get("model_name") if isinstance(payload, dict) else None,
        }

    project_root = Path(__file__).resolve().parents[2]
    snapshot = {
        "project_git_revision": git_revision(project_root),
        "configs": config_records,
        "checkpoints": checkpoint_records,
        "notes": [
            "Resolved configuration files are archival copies of the effective Hydra settings.",
            "Checkpoint SHA-256 identifies the exact model weights used by the frozen evaluations.",
        ],
    }
    (args.output_dir / "reproducibility_snapshot.json").write_text(json.dumps(snapshot, indent=2) + "\n", encoding="utf-8")

    args.latex_output.parent.mkdir(parents=True, exist_ok=True)
    rows = ["\\begin{tabular}{lll}", "\\toprule", "Artifact & Configuration / checkpoint & Epoch \\\\ ", "\\midrule"]
    for label in sorted(set(config_records) | set(checkpoint_records)):
        config_name = config_records.get(label, {}).get("config_name", "--")
        epoch = checkpoint_records.get(label, {}).get("epoch", "--")
        rows.append(f"{latex_escape(label)} & {latex_escape(str(config_name))} & {epoch if epoch is not None else '--'} \\\\ ")
    rows.extend(["\\bottomrule", "\\end{tabular}"])
    args.latex_output.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"Wrote reproducibility snapshot: {args.output_dir / 'reproducibility_snapshot.json'}")
    print(f"Wrote LaTeX provenance table: {args.latex_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
