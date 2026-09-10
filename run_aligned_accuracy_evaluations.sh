#!/usr/bin/env bash
set -u

ROOT=${ROOT:-/home/parsa/smart_parsa}
PYTHON=${PYTHON:-/home/parsa/miniconda3/envs/smart/bin/python}
EVALUATOR="$ROOT/smart/scripts/audit_paired_architecture_sampling.py"
SOURCE_ROOT="$ROOT/results/final/deal_canonical_evaluation/paired_architectures"
OUTPUT_ROOT="$ROOT/results/final/deal_canonical_evaluation/aligned_accuracy_20260910"
PUMP_CHECKPOINT_CACHE=${PUMP_CHECKPOINT_CACHE:-/mnt/data/parsa/deal_aligned_checkpoint_cache_20260910}

export PYTHONPATH="$ROOT/smart"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

run_pair() {
  local gpu=$1 task=$2 model=$3
  local source="$SOURCE_ROOT/$task/$model/summary.json"
  local output="$OUTPUT_ROOT/$task/$model"
  local metadata=()
  local runtime_args=()
  if [[ ! -s "$source" ]]; then
    printf 'Missing canonical source summary: %s\n' "$source" >&2
    return 1
  fi
  mapfile -t metadata < <("$PYTHON" - "$source" <<'PY'
import json
import sys

summary = json.load(open(sys.argv[1], encoding="utf-8"))
print(summary["dataset"])
print(summary["model"])
print(summary["configs"]["base"])
print(summary["configs"]["deal"])
print(summary["checkpoints"]["base"]["path"])
print(summary["checkpoints"]["deal"]["path"])
print(summary["data_root"])
print(summary["remesh_root"])
print(",".join(map(str, summary["case_ids"])))
print(summary["views_per_condition"])
print(summary["seed"])
print(",".join(summary["remesh_methods"]))
print(",".join(map(str, summary["remesh_factors"])))
print(summary["checkpoint_policy"])
PY
  )
  if [[ "$task" == pump && ! -d "${metadata[6]}" ]]; then
    metadata[4]="$PUMP_CHECKPOINT_CACHE/$(basename "${metadata[4]}")"
    metadata[5]="$PUMP_CHECKPOINT_CACHE/$(basename "${metadata[5]}")"
    metadata[6]="/mnt/data/parsa/shift_pump_random1400_preprocessed"
    metadata[7]="/mnt/data/parsa/shift_pump_random1400_surface_vtp_remesh_v4"
  fi
  if [[ "$model" == transolverpp && ( "$task" == drivaerml || "$task" == heat_exchanger ) ]]; then
    runtime_args=(--transolver-slice-assignment-mode deterministic_softmax)
  fi

  mkdir -p "$output"
  printf '[%s] aligned audit %s/%s on physical GPU %s\n' \
    "$(date --iso-8601=seconds)" "$task" "$model" "$gpu" > "$output/run.log"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" "$EVALUATOR" \
    --dataset "${metadata[0]}" --model "${metadata[1]}" \
    --base-config "${metadata[2]}" --deal-config "${metadata[3]}" \
    --base-checkpoint "${metadata[4]}" --deal-checkpoint "${metadata[5]}" \
    --data-root "${metadata[6]}" --remesh-root "${metadata[7]}" \
    --case-ids "${metadata[8]}" --views-per-condition "${metadata[9]}" \
    --seed "${metadata[10]}" --methods "${metadata[11]}" --factors "${metadata[12]}" \
    --checkpoint-policy "${metadata[13]}" --conditions original --device cuda:0 \
    "${runtime_args[@]}" \
    --output-dir "$output" >> "$output/run.log" 2>&1
  local status=$?
  printf '%s\n' "$status" > "$output/exit_code"
  printf '[%s] finished %s/%s status=%s\n' \
    "$(date --iso-8601=seconds)" "$task" "$model" "$status" >> "$output/run.log"
}

run_sequence() {
  local gpu=$1 spec task model status=0
  shift
  for spec in "$@"; do
    IFS=/ read -r task model <<< "$spec"
    run_pair "$gpu" "$task" "$model" || status=1
  done
  return "$status"
}

scope=${1:-all}
if [[ "$scope" == pump ]]; then
  run_sequence 2 pump/geo_fno pump/mspt & p2=$!
  run_sequence 5 pump/ab_upt pump/transolverpp & p5=$!
  run_sequence 6 pump/pointnet2_ssg pump/lno & p6=$!
  run_sequence 7 pump/smart pump/point_transformer_v3 & p7=$!
elif [[ "$scope" == all ]]; then
  run_sequence 2 \
    drivaerml/geo_fno drivaerml/mspt pump/geo_fno pump/mspt \
    heat_exchanger/geo_fno heat_exchanger/mspt c_core/geo_fno c_core/mspt & p2=$!
  run_sequence 5 \
    drivaerml/smart drivaerml/ab_upt drivaerml/transolverpp drivaerml/point_transformer_v3 \
    pump/ab_upt pump/transolverpp heat_exchanger/ab_upt heat_exchanger/transolverpp \
    c_core/ab_upt c_core/transolverpp & p5=$!
  run_sequence 6 \
    drivaerml/pointnet2_ssg drivaerml/lno pump/pointnet2_ssg pump/lno \
    heat_exchanger/pointnet2_ssg heat_exchanger/lno c_core/pointnet2_ssg c_core/lno & p6=$!
  run_sequence 7 \
    pump/smart pump/point_transformer_v3 heat_exchanger/smart \
    heat_exchanger/point_transformer_v3 c_core/smart c_core/point_transformer_v3 & p7=$!
else
  printf 'Usage: %s [all|pump]\n' "$0" >&2
  exit 2
fi

status=0
for pid in "$p2" "$p5" "$p6" "$p7"; do
  wait "$pid" || status=1
done
exit "$status"
