#!/usr/bin/env bash

set -euo pipefail

ROOT="/home/parsa/smart_parsa"
PYTHON="/home/parsa/miniconda3/envs/smart/bin/python"
TARGET="${1:-all}"
GPU_IDS="${GPU_IDS:-0,3}"
LOG_DIR="${LOG_DIR:-${ROOT}/results/base_architecture_training_logs}"

ALL_TARGETS=(
  drivaerml_ab_upt
  drivaerml_gaot
  heat_exchanger_ab_upt
  heat_exchanger_gaot
  pump_transolverpp
  pump_pointnet2_ssg
  pump_lno
  pump_mspt
  pump_point_transformer_v3
  pump_ab_upt
  pump_gaot
)

DRIVAERML_TARGETS=(drivaerml_ab_upt drivaerml_gaot)
HEAT_EXCHANGER_TARGETS=(heat_exchanger_ab_upt heat_exchanger_gaot)
PUMP_TARGETS=(
  pump_transolverpp
  pump_pointnet2_ssg
  pump_lno
  pump_mspt
  pump_point_transformer_v3
  pump_ab_upt
  pump_gaot
)

usage() {
  printf 'Usage: GPU_IDS=0,3 bash %s {all|drivaerml|heat_exchanger|pump|MODEL}\n' "$0"
  printf 'Models: %s\n' "${ALL_TARGETS[*]}"
}

entrypoint_for() {
  case "$1" in
    drivaerml_ab_upt) printf '%s\n' "smart/train_ab_upt.py" ;;
    drivaerml_gaot) printf '%s\n' "smart/train_gaot.py" ;;
    heat_exchanger_ab_upt) printf '%s\n' "smart/train_toy_heat_exchange_ab_upt.py" ;;
    heat_exchanger_gaot) printf '%s\n' "smart/train_toy_heat_exchange_gaot.py" ;;
    pump_transolverpp) printf '%s\n' "smart/train_pump_transolverpp.py" ;;
    pump_pointnet2_ssg) printf '%s\n' "smart/train_pump_pointnet2_ssg.py" ;;
    pump_lno) printf '%s\n' "smart/train_pump_lno.py" ;;
    pump_mspt) printf '%s\n' "smart/train_pump_mspt.py" ;;
    pump_point_transformer_v3) printf '%s\n' "smart/train_pump_point_transformer_v3.py" ;;
    pump_ab_upt) printf '%s\n' "smart/train_pump_ab_upt.py" ;;
    pump_gaot) printf '%s\n' "smart/train_pump_gaot.py" ;;
    *) return 1 ;;
  esac
}

case "$TARGET" in
  all) TARGETS=("${ALL_TARGETS[@]}") ;;
  drivaerml) TARGETS=("${DRIVAERML_TARGETS[@]}") ;;
  heat_exchanger) TARGETS=("${HEAT_EXCHANGER_TARGETS[@]}") ;;
  pump) TARGETS=("${PUMP_TARGETS[@]}") ;;
  -h|--help)
    usage
    exit 0
    ;;
  *)
    if ! entrypoint_for "$TARGET" >/dev/null; then
      usage >&2
      exit 2
    fi
    TARGETS=("$TARGET")
    ;;
esac

IFS=',' read -r -a GPUS <<< "$GPU_IDS"
if (( ${#GPUS[@]} == 0 )); then
  printf 'GPU_IDS must contain at least one physical GPU ID.\n' >&2
  exit 2
fi
if (( ${#GPUS[@]} > ${#TARGETS[@]} )); then
  GPUS=("${GPUS[@]:0:${#TARGETS[@]}}")
fi

mkdir -p "$LOG_DIR"
cd "$ROOT"

run_target() {
  local target="$1"
  local gpu="$2"
  local entrypoint
  entrypoint="$(entrypoint_for "$target")"
  printf '[%s] starting on physical GPU %s (%s)\n' "$target" "$gpu" "$entrypoint"
  CUDA_VISIBLE_DEVICES="$gpu" \
  PYTHONPATH="${ROOT}/smart" \
  PYTHONUNBUFFERED=1 \
  OMP_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 \
  SMART_KNN_N_JOBS=1 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "$PYTHON" "$ROOT/$entrypoint" 2>&1 | tee "$LOG_DIR/${target}.log"
  printf '[%s] completed on physical GPU %s\n' "$target" "$gpu"
}

worker() {
  local worker_index="$1"
  local gpu="$2"
  local target_index
  for ((target_index=worker_index; target_index<${#TARGETS[@]}; target_index+=${#GPUS[@]})); do
    run_target "${TARGETS[target_index]}" "$gpu"
  done
}

pids=()
for ((worker_index=0; worker_index<${#GPUS[@]}; worker_index++)); do
  worker "$worker_index" "${GPUS[worker_index]}" &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done
exit "$status"
