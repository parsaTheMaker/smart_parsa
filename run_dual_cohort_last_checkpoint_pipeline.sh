#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/parsa/smart_parsa}
PYTHON=${PYTHON:-/home/parsa/miniconda3/envs/smart/bin/python}
REMOTE=${REMOTE:-parsa@servus06.ge.in.tum.de}
REMOTE_ROOT=${REMOTE_ROOT:-/mnt/data5/parsa/smart_parsa}
CANONICAL="$ROOT/results/final/deal_canonical_evaluation"
export PYTHONPATH="$ROOT/smart"
export PYTHONUNBUFFERED=1

run_sweeps() {
  local scope=$1
  local directory
  if [[ "$scope" == candidates ]]; then
    directory=candidate_architectures
  else
    directory=paired_architectures
  fi
  bash "$ROOT/run_canonical_architecture_evaluations.sh" "$scope" & local_pid=$!
  ssh "$REMOTE" "cd '$REMOTE_ROOT' && bash run_canonical_pump_architectures_servus06.sh '$scope'" & remote_pid=$!
  wait "$local_pid"
  wait "$remote_pid"
  mkdir -p "$CANONICAL/$directory/pump"
  rsync -a --checksum "$REMOTE:$REMOTE_ROOT/results/final/deal_canonical_evaluation/$directory/pump/" \
    "$CANONICAL/$directory/pump/"
  local pump_summaries
  pump_summaries=$(find "$CANONICAL/$directory/pump" -name summary.json | wc -l)
  if [[ "$pump_summaries" -ne 8 ]]; then
    printf 'Expected 8 synchronized Pump summaries for %s, found %s.\n' \
      "$scope" "$pump_summaries" >&2
    return 1
  fi
}

run_sweeps candidates

"$PYTHON" "$ROOT/smart/scripts/build_canonical_cohort_registry.py" \
  --selection-basis smart --force \
  --output "$CANONICAL/cohort_registry_smart_v1.json"
"$PYTHON" "$ROOT/smart/scripts/build_canonical_cohort_registry.py" \
  --selection-basis all_architectures --force \
  --output "$CANONICAL/cohort_registry_all_architectures_v1.json"
"$PYTHON" "$ROOT/smart/scripts/summarize_canonical_architecture_evaluation.py" \
  --registry "$CANONICAL/cohort_registry_smart_v1.json" \
  --evaluation-root "$CANONICAL/candidate_architectures" \
  --output-dir "$CANONICAL/cohort_studies/smart_ranked" \
  --filter-candidate-results
"$PYTHON" "$ROOT/smart/scripts/summarize_canonical_architecture_evaluation.py" \
  --registry "$CANONICAL/cohort_registry_all_architectures_v1.json" \
  --evaluation-root "$CANONICAL/candidate_architectures" \
  --output-dir "$CANONICAL/cohort_studies/all_architectures_ranked" \
  --filter-candidate-results
"$PYTHON" "$ROOT/smart/scripts/compare_canonical_cohort_registries.py"

scp -q "$CANONICAL/cohort_registry_all_architectures_v1.json" \
  "$REMOTE:$REMOTE_ROOT/results/final/deal_canonical_evaluation/"
run_sweeps selected

"$PYTHON" "$ROOT/smart/scripts/summarize_canonical_architecture_evaluation.py"
