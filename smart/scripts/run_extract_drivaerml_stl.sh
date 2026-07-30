#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/parsa/smart_parsa"
PYTHON="/home/parsa/miniconda3/envs/smart/bin/python"

export PYTHONPATH="${ROOT}/smart${PYTHONPATH:+:${PYTHONPATH}}"
exec "${PYTHON}" "${ROOT}/smart/scripts/extract_drivaerml_stl_from_surface_vtp.py" "$@"
