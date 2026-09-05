#!/usr/bin/env bash
# Complete only the reviewer artifacts that were not produced before geometry
# validation was interrupted. Each distance remains an exact VTK point-to-
# triangle distance; 2,000 samples per direction across many cases is used to
# keep the audit tractable on the large DrivAerML meshes.
set -euo pipefail

ROOT=/home/parsa/smart_parsa
PYTHON=/home/parsa/miniconda3/envs/smart/bin/python
OUT="$ROOT/results/final/reviewer_evidence_20260901"
export PYTHONPATH="$ROOT/smart"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export VTK_SMP_MAX_THREADS=1

run_geometry() {
  "$PYTHON" "$ROOT/smart/scripts/validate_remesh_geometry.py" "$@"
}

# The 16 + 8 + 8 worker allocation fills 32 CPU cores without letting VTK's
# internal threading multiply the process count.
run_geometry \
  --dataset drivaerml \
  --source-dir /mnt/ssdraid/parsa/drivaerml_surface_vtp \
  --remesh-dir /mnt/ssdraid/parsa/drivaerml_surface_vtp_remesh_v4 \
  --output-dir "$OUT/drivaerml_remesh_geometry" \
  --methods voxel,quadric,feature --factors 5,10 \
  --max-cases 50 --distance-samples 2000 --normal-samples 1000 --workers 16 --seed 42 &
pid_drivaerml=$!

run_geometry \
  --dataset pump \
  --source-dir /mnt/data/parsa/shift_pump_raw_random1400 \
  --remesh-dir /mnt/data/parsa/shift_pump_random1400_surface_vtp_remesh_v4 \
  --output-dir "$OUT/pump_remesh_geometry" \
  --methods voxel,quadric,feature --factors 5,10 \
  --max-cases 50 --distance-samples 2000 --normal-samples 1000 --workers 8 --seed 42 &
pid_pump=$!

run_geometry \
  --dataset heat_exchanger \
  --source-dir /mnt/ssdraid/parsa/toy_heat_exchange_surface_vtp \
  --remesh-dir /mnt/ssdraid/parsa/toy_heat_exchange_surface_vtp_remesh_v4 \
  --output-dir "$OUT/heat_exchanger_remesh_geometry" \
  --methods quadric,feature --factors 5,10 \
  --max-cases 0 --distance-samples 2000 --normal-samples 1000 --workers 8 --seed 42 &
pid_heat=$!

status=0
for pid in "$pid_drivaerml" "$pid_pump" "$pid_heat"; do
  wait "$pid" || status=1
done
if (( status != 0 )); then
  echo "At least one geometry audit failed; documentation was not started." >&2
  exit "$status"
fi

"$PYTHON" "$ROOT/smart/scripts/audit_toy_heat_exchange_generation.py" \
  --data-root /mnt/ssdraid/parsa/toy_heat_exchange_fem_v1 \
  --output-dir "$OUT/heat_exchanger_generation_audit"
"$PYTHON" "$ROOT/smart/scripts/export_paper_task_cards.py" \
  --output-json "$OUT/task_cards.json" \
  --output-markdown "$OUT/task_cards.md" \
  --output-latex "$OUT/task_cards.tex"
"$PYTHON" "$ROOT/smart/scripts/export_paper_reproducibility_snapshot.py" \
  --output-dir "$OUT/reproducibility" \
  --latex-output "$OUT/reproducibility/reproducibility_snapshot.tex" \
  --config pump_base=pump --config pump_deal=pump_deal_from_smart_full \
  --config heat_base=toy_heat_exchange --config heat_deal=toy_heat_exchange_satloss7 \
  --checkpoint pump_base="$ROOT/checkpoints/smart-pump-random1400-base-16k-pump-s42_best.pt" \
  --checkpoint pump_deal="$ROOT/checkpoints/smart-pump-deal-random1400-from-smart-150ep-pump-s42_best.pt" \
  --checkpoint heat_base="$ROOT/checkpoints/smart-toy-heat-exchange-heat-exchange-base-ratio-aligned-toyheatexchange-s42_best.pt" \
  --checkpoint heat_deal="$ROOT/checkpoints/smart-toy-heat-exchange-satloss7-heat-exchange-satloss-ratio-aligned-toyheatexchange-s42_best.pt"
"$PYTHON" "$ROOT/smart/scripts/create_top20_paper_vs_frozen_tables.py" \
  --output "$OUT/paper_vs_frozen_top20_diagnostic.pdf"

echo "Reviewer geometry and documentation evidence complete: $OUT"
