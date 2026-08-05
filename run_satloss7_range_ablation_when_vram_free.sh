#!/usr/bin/env bash
set -Eeuo pipefail

# Schedule the three SATLOSS7 range ablations independently. Before each run,
# scan all eight GPUs and use the first unreserved one below the VRAM limit.
# Other pending jobs continue waiting while already-launched jobs run.

GPUS=(0 1 2 3 4 5 6 7)
VRAM_LIMIT_PERCENT="${VRAM_LIMIT_PERCENT:-15}"
POLL_SECONDS="${POLL_SECONDS:-30}"
DRY_RUN="${DRY_RUN:-0}"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi is required." >&2
  exit 1
fi

if ! [[ "$VRAM_LIMIT_PERCENT" =~ ^[0-9]+$ ]] || (( VRAM_LIMIT_PERCENT < 1 || VRAM_LIMIT_PERCENT > 100 )); then
  echo "ERROR: VRAM_LIMIT_PERCENT must be an integer from 1 to 100." >&2
  exit 1
fi

if ! [[ "$POLL_SECONDS" =~ ^[0-9]+$ ]] || (( POLL_SECONDS < 1 )); then
  echo "ERROR: POLL_SECONDS must be a positive integer." >&2
  exit 1
fi

gpu_usage() {
  local gpu="$1"
  local used total
  IFS=',' read -r used total < <(
    nvidia-smi -i "$gpu" \
      --query-gpu=memory.used,memory.total \
      --format=csv,noheader,nounits | tr -d ' '
  )
  if [[ -z "${used:-}" || -z "${total:-}" ]]; then
    echo "ERROR: could not read VRAM usage for GPU $gpu." >&2
    return 1
  fi
  printf '%s %s\n' "$used" "$total"
}

print_gpu_status() {
  local gpu used total percent marker
  for gpu in "${GPUS[@]}"; do
    read -r used total < <(gpu_usage "$gpu")
    percent=$((used * 100 / total))
    marker=""
    if [[ "${RESERVED_GPU[$gpu]:-0}" == "1" ]]; then
      marker=" [reserved]"
    fi
    printf '[wait] GPU %s: %s MiB / %s MiB (%s%% used)%s\n' \
      "$gpu" "$used" "$total" "$percent" "$marker"
  done
}

find_free_gpu() {
  local gpu used total
  for gpu in "${GPUS[@]}"; do
    if [[ "${RESERVED_GPU[$gpu]:-0}" == "1" ]]; then
      continue
    fi
    read -r used total < <(gpu_usage "$gpu")
    if (( used * 100 < total * VRAM_LIMIT_PERCENT )); then
      printf '%s\n' "$gpu"
      return 0
    fi
  done
  return 1
}

cleanup() {
  local status=$?
  trap - EXIT INT TERM
  if ((${#PIDS[@]} > 0)); then
    echo "[stop] stopping training processes..." >&2
    kill "${PIDS[@]}" 2>/dev/null || true
    wait "${PIDS[@]}" 2>/dev/null || true
  fi
  exit "$status"
}

PIDS=()
declare -A RESERVED_GPU=()
declare -A PID_GPU=()
trap cleanup EXIT
trap 'exit 130' INT TERM

commands=(
  "drivaerml_satloss7_range025|smart-satloss7-range025-drivaerml|DrivAerML_SMART_SATLOSS7_RANGE_025"
  "drivaerml_satloss7_range050|smart-satloss7-range050-drivaerml|DrivAerML_SMART_SATLOSS7_RANGE_050"
  "drivaerml_satloss7_range075|smart-satloss7-range075-drivaerml|DrivAerML_SMART_SATLOSS7_RANGE_075"
)

if [[ "$DRY_RUN" == "1" ]]; then
  printf '[dry-run] concurrent jobs; each selects an unreserved GPU from: %s\n' "${GPUS[*]}"
  printf '[dry-run] VRAM threshold: %s%%\n' "$VRAM_LIMIT_PERCENT"
  printf '[dry-run] %s\n' "${commands[@]}"
  exit 0
fi

launch_job() {
  local job_index="$1"
  local gpu="$2"
  local config_name model_tag experiment_name command pid
  IFS='|' read -r config_name model_tag experiment_name <<< "${commands[$job_index]}"
  command="PYTHONPATH=/home/parsa/smart_parsa/smart CUDA_VISIBLE_DEVICES=${gpu} /home/parsa/miniconda3/envs/smart/bin/python smart/train_satloss7.py --config-name=${config_name} experiment.model_tag=${model_tag} experiment.name=${experiment_name} wandb.project=smart_drivaerml wandb.entity=parsa-vatani99-technical-university-of-munich"
  echo "[run] GPU ${gpu}: ${command}"
  bash -c "$command" &
  pid="$!"
  PIDS+=("$pid")
  PID_GPU["$pid"]="$gpu"
  RESERVED_GPU["$gpu"]="1"
  echo "[run] started PID ${pid} for ${model_tag}; other pending jobs will continue scanning for free GPUs."
}

reap_finished_jobs() {
  local pid status gpu model_tag
  local active_pids=()
  for pid in "${PIDS[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      active_pids+=("$pid")
      continue
    fi
    if wait "$pid"; then
      status=0
    else
      status=$?
    fi
    gpu="${PID_GPU[$pid]}"
    unset "RESERVED_GPU[$gpu]"
    unset "PID_GPU[$pid]"
    IFS='|' read -r _ model_tag _ <<< "${commands[$JOB_INDEX_BY_PID[$pid]]}"
    if (( status != 0 )); then
      echo "[error] ${model_tag} failed with exit code ${status}." >&2
      return "$status"
    fi
    COMPLETED=$((COMPLETED + 1))
    echo "[run] completed ${model_tag} on GPU ${gpu}; that GPU is available for future jobs."
  done
  PIDS=("${active_pids[@]}")
  return 0
}

declare -A JOB_INDEX_BY_PID=()
cd /home/parsa/smart_parsa
NEXT_JOB=0
COMPLETED=0
TOTAL_JOBS=${#commands[@]}

while (( COMPLETED < TOTAL_JOBS )); do
  while (( NEXT_JOB < TOTAL_JOBS )); do
    if gpu="$(find_free_gpu)"; then
      launch_job "$NEXT_JOB" "$gpu"
      JOB_INDEX_BY_PID["${PIDS[-1]}"]="$NEXT_JOB"
      NEXT_JOB=$((NEXT_JOB + 1))
    else
      break
    fi
  done

  if reap_finished_jobs; then
    :
  else
    status=$?
    exit "$status"
  fi

  if (( COMPLETED < TOTAL_JOBS )); then
    if (( NEXT_JOB < TOTAL_JOBS )); then
      print_gpu_status
      echo "[wait] pending jobs are waiting for any unreserved GPU below ${VRAM_LIMIT_PERCENT}%; rescanning in ${POLL_SECONDS}s..."
    fi
    sleep "$POLL_SECONDS"
  fi
done

echo "[run] all three SATLOSS7 range ablations completed."
