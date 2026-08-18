#!/usr/bin/env python3
"""Remesh Toy Heat Exchange validation surfaces with the shared three-method tool."""

from __future__ import annotations

import sys

from scripts.prepare_toy_perforated_fin_remeshing import main


if __name__ == "__main__":
    # Keep the established implementation and validation behavior while giving
    # this benchmark an explicit, self-documenting entry point.
    if "--case-stem" not in sys.argv:
        sys.argv.extend(["--case-stem", "heat_exchange_case"])
    raise SystemExit(main())
