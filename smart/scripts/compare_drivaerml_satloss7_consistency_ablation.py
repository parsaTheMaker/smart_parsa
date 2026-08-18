#!/usr/bin/env python3
"""SMART consistency-loss ablation with the established KDE/range plots.

The plotting and aggregation engine is the same one used by the KDE
ablation. Its two SATLOSS slots are mapped here to the consistency-enabled
and no-consistency range-100 checkpoints, so this experiment gets the same
endpoint bars, remeshing bars, tables, colors, fonts, and percentage labels.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


SCRIPT_PATH = Path(__file__).with_name("compare_drivaerml_satloss7_range_ablation.py")
SPEC = importlib.util.spec_from_file_location("smart_drivaerml_range_ablation_comparison", SCRIPT_PATH)
if SPEC is None or SPEC.loader is None:
    raise ImportError(f"Could not load comparison engine from {SCRIPT_PATH}")
comparison = importlib.util.module_from_spec(SPEC)
# Register the dynamically loaded engine before execution. The range engine
# uses threads, but its imported model helpers also need a stable module name.
sys.modules[SPEC.name] = comparison
SPEC.loader.exec_module(comparison)


CONSISTENCY_MODEL_ORDER = (
    "SMART",
    "SMART_SATLOSS7_RANGE025",
    "SMART_SATLOSS7_RANGE050",
)
CONSISTENCY_MODEL_LABELS = {
    "SMART": "SMART baseline",
    "SMART_SATLOSS7_RANGE025": "w consistency loss",
    "SMART_SATLOSS7_RANGE050": "wo consistency loss",
}
CONSISTENCY_MODEL_COLORS = {
    "SMART": "#4C78A8",
    "SMART_SATLOSS7_RANGE025": "#7A5195",
    "SMART_SATLOSS7_RANGE050": "#F28E2B",
}


def _consistency_checkpoint_map(args):
    checkpoints = comparison.OrderedDict(
        [
            ("SMART", args.smart_checkpoint),
            ("SMART_SATLOSS7_RANGE025", args.kde4_checkpoint),
            ("SMART_SATLOSS7_RANGE050", args.kde8_checkpoint),
        ]
    )
    missing = [name for name, path in checkpoints.items() if not path]
    if missing:
        raise ValueError("Consistency ablation requires checkpoints for: " + ", ".join(missing))
    return checkpoints


def _consistency_config_map(args):
    return comparison.OrderedDict(
        [
            ("SMART", args.smart_config),
            ("SMART_SATLOSS7_RANGE025", args.kde4_config),
            ("SMART_SATLOSS7_RANGE050", args.kde8_config),
        ]
    )


def _translate_cli_aliases() -> None:
    """Expose descriptive options while reusing the range engine parser."""
    aliases = {
        "--w-consistency-config": "--kde4-config",
        "--w-consistency-checkpoint": "--kde4-checkpoint",
        "--wo-consistency-config": "--kde8-config",
        "--wo-consistency-checkpoint": "--kde8-checkpoint",
    }
    translated = []
    for token in sys.argv[1:]:
        replacement = token
        for source, target in aliases.items():
            if token == source:
                replacement = target
                break
            if token.startswith(source + "="):
                replacement = target + token[len(source):]
                break
        translated.append(replacement)

    if "--experiment-preset" not in translated:
        translated.extend(["--experiment-preset", "kde_ablation_vtp"])
    else:
        for index, token in enumerate(translated[:-1]):
            if token == "--experiment-preset" and translated[index + 1] == "consistency_ablation_vtp":
                translated[index + 1] = "kde_ablation_vtp"
    sys.argv = [sys.argv[0], *translated]


def _rename_outputs(output_dir: Path) -> None:
    """Use consistency-specific filenames while preserving KDE plot geometry."""
    for path in sorted(output_dir.iterdir()):
        if path.is_file() and "kde_ablation" in path.name:
            target = path.with_name(path.name.replace("kde_ablation", "consistency_ablation"))
            path.replace(target)
    for path in output_dir.iterdir():
        if not path.is_file() or path.suffix not in {".json", ".md", ".csv"}:
            continue
        text = path.read_text(encoding="utf-8")
        text = text.replace("kde_ablation", "consistency_ablation")
        text = text.replace("kde_ablation_vtp", "consistency_ablation_vtp")
        text = text.replace("SATLOSS KDE-neighborhood ablation", "SMART consistency-loss ablation")
        path.write_text(text, encoding="utf-8")


def main() -> None:
    _translate_cli_aliases()
    comparison.KDE_MODEL_ORDER = CONSISTENCY_MODEL_ORDER
    comparison.KDE_MODEL_LABELS = dict(CONSISTENCY_MODEL_LABELS)
    comparison.KDE_MODEL_COLORS = dict(CONSISTENCY_MODEL_COLORS)
    comparison.checkpoint_map = _consistency_checkpoint_map
    comparison.config_map = _consistency_config_map
    comparison.main()

    output_dir = Path(comparison.parse_args().output_dir).expanduser().resolve()
    _rename_outputs(output_dir)


if __name__ == "__main__":
    main()
