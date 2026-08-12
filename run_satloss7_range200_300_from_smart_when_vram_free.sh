#!/usr/bin/env bash
set -Eeuo pipefail

# Launch the range-200 and range-300 SATLOSS7 jobs independently. Each job
# waits for a separate pair of GPUs with at least 90% free VRAM.

ROOT=/home/parsa/smart_parsa
PYTHON=/home/parsa/miniconda3/envs/smart/bin/python
GPUS=(0 1 2 3 4 5 6 7)
GPUS_PER_JOB=2
VRAM_LIMIT_PERCENT="${VRAM_LIMIT_PERCENT:-12}"
POLL_SECONDS="${POLL_SECONDS:-30}"
DRY_RUN="${DRY_RUN:-0}"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi is required." >&2
  exit 1
fi
if ! command -v setsid >/dev/null 2>&1; then
  echo "ERROR: setsid is required." >&2
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
  local gpu="$1" used total
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

find_free_pair() {
  local gpu used total
  local free_gpus=()
  for gpu in "${GPUS[@]}"; do
    [[ "${RESERVED_GPU[$gpu]:-0}" == "1" ]] && continue
    read -r used total < <(gpu_usage "$gpu")
    if (( used * 100 >= total * VRAM_LIMIT_PERCENT )); then
      continue
    fi
    free_gpus+=("$gpu")
    if (( ${#free_gpus[@]} == GPUS_PER_JOB )); then
      printf '%s %s\n' "${free_gpus[0]}" "${free_gpus[1]}"
      return 0
    fi
  done
  return 1
}

commands=(
  "drivaerml_satloss7_range200|smart-satloss7-range200-from-smart-150ep|DrivAerML_SMART_SATLOSS7_RANGE_200_FROM_SMART_150EP|29632"
  "drivaerml_satloss7_range300|smart-satloss7-range300-from-smart-150ep|DrivAerML_SMART_SATLOSS7_RANGE_300_FROM_SMART_150EP|29633"
)

PIDS=()
declare -A RESERVED_GPU=()
declare -A PID_JOB=()
declare -A PID_GPU_PAIR=()

cleanup() {
  local status=$? pid
  trap - EXIT INT TERM
  for pid in "${PIDS[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      echo "[stop] stopping process group for PID $pid..." >&2
      kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
    fi
  done
  for pid in "${PIDS[@]}"; do
    wait "$pid" 2>/dev/null || true
  done
  exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT TERM

if [[ "$DRY_RUN" == "1" ]]; then
  printf '[dry-run] jobs: %s; GPU pool: %s\n' "${#commands[@]}" "${GPUS[*]}"
  printf '[dry-run] each job requires %s GPUs below %s%% VRAM used; poll=%ss\n' \
    "$GPUS_PER_JOB" "$VRAM_LIMIT_PERCENT" "$POLL_SECONDS"
  printf '[dry-run] %s\n' "${commands[@]}"
  exit 0
fi

launch_job() {
  local job_index="$1" gpu_a="$2" gpu_b="$3"
  local config_name model_tag experiment_name master_port pid
  IFS='|' read -r config_name model_tag experiment_name master_port <<< "${commands[$job_index]}"
  echo "[run] launching $model_tag on GPUs $gpu_a,$gpu_b"
  setsid env \
    PYTHONPATH="$ROOT/smart" \
    CUDA_VISIBLE_DEVICES="$gpu_a,$gpu_b" \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$PYTHON" -m torch.distributed.run \
      --standalone --nnodes=1 --nproc_per_node=2 --master_port="$master_port" \
      smart/train_satloss7.py \
      --config-name="$config_name" \
      experiment.model_tag="$model_tag" \
      experiment.name="$experiment_name" \
      experiment.epochs=150 \
      experiment.multi_gpu_strategy=ddp \
      experiment.batch_size=1 \
      experiment.init_ckpt="$ROOT/checkpoints/smart-smart-drivaerml-131k16kwr-drivaerml-s42_best.pt" \
      experiment.resume_ckpt= \
      experiment.resume_full_state=False \
      wandb.project=smart_drivaerml \
      wandb.entity=parsa-vatani99-technical-university-of-munich &
  pid=$!
  PIDS+=("$pid")
  PID_JOB["$pid"]="$job_index"
  PID_GPU_PAIR["$pid"]="$gpu_a,$gpu_b"
  RESERVED_GPU["$gpu_a"]=1
  RESERVED_GPU["$gpu_b"]=1
  echo "[run] started $model_tag with launcher PID $pid."
}

reap_finished_jobs() {
  local pid status gpu_pair job_index config_name model_tag experiment_name master_port
  local active_pids=()
  for pid in "${PIDS[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      active_pids+=("$pid")
      continue
    fi
    if wait "$pid"; then status=0; else status=$?; fi
    gpu_pair="${PID_GPU_PAIR[$pid]}"
    job_index="${PID_JOB[$pid]}"
    IFS='|' read -r config_name model_tag experiment_name master_port <<< "${commands[$job_index]}"
    IFS=',' read -r gpu_a gpu_b <<< "$gpu_pair"
    unset "RESERVED_GPU[$gpu_a]" "RESERVED_GPU[$gpu_b]"
    unset "PID_JOB[$pid]" "PID_GPU_PAIR[$pid]"
    if (( status != 0 )); then
      echo "[error] $model_tag failed with exit code $status." >&2
      return "$status"
    fi
    COMPLETED=$((COMPLETED + 1))
    echo "[run] completed $model_tag on GPUs $gpu_pair."
  done
  PIDS=("${active_pids[@]}")
  return 0
}

cd "$ROOT"
NEXT_JOB=0
COMPLETED=0
TOTAL_JOBS=${#commands[@]}
while (( COMPLETED < TOTAL_JOBS )); do
  while (( NEXT_JOB < TOTAL_JOBS )); do
    if pair="$(find_free_pair)"; then
      read -r gpu_a gpu_b <<< "$pair"
      launch_job "$NEXT_JOB" "$gpu_a" "$gpu_b"
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
    print_gpu_status
    echo "[wait] rescanning for another free pair in ${POLL_SECONDS}s..."
    sleep "$POLL_SECONDS"
  fi
done

trap - EXIT INT TERM
echo "[run] both range-200 and range-300 jobs completed."
