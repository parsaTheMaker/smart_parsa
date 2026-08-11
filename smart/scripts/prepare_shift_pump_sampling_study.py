#!/usr/bin/env python3
"""Pump-named entrypoint for the shared geometry-only remeshing study.

All dataset-specific paths and the Hugging Face repository are supplied by
the command line.  The implementation is shared with the validated
SHIFT-Submarine study so the three remeshing methods remain identical.
"""

from scripts.prepare_shift_submarine_sampling_study import main


if __name__ == "__main__":
    raise SystemExit(main())
