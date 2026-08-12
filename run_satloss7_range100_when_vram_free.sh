#!/usr/bin/env bash
set -Eeuo pipefail

# Wait until any four GPUs in the 0-7 pool are available, then launch one
# four-process SATLOSS7 RANGE100 DDP run on the selected GPUs.

ROOT=/home/parsa/smart_parsa
PYTHON=/home/parsa/miniconda3/envs/smart/bin/python
GPUS=(0 1 2 3 4 5 6 7)
REQUIRED_GPUS=4
VRAM_LIMIT_PERCENT="${VRAM_LIMIT_PERCENT:-10}"
POLL_SECONDS="${POLL_SECONDS:-30}"
DRY_RUN="${DRY_RUN:-0}"
MASTER_PORT="${MASTER_PORT:-29630}"

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
  local gpu used total percent
  for gpu in "${GPUS[@]}"; do
    read -r used total < <(gpu_usage "$gpu")
    percent=$((used * 100 / total))
    printf '[wait] GPU %s: %s MiB / %s MiB (%s%% used)\n' \
      "$gpu" "$used" "$total" "$percent"
  done
}

find_free_gpus() {
  local gpu used total
  local free_gpus=()
  for gpu in "${GPUS[@]}"; do
    read -r used total < <(gpu_usage "$gpu")
    if (( used * 100 >= total * VRAM_LIMIT_PERCENT )); then
      continue
    fi
    free_gpus+=("$gpu")
    if (( ${#free_gpus[@]} == REQUIRED_GPUS )); then
      printf '%s\n' "${free_gpus[*]}"
      return 0
    fi
  done
  return 1
}

if [[ "$DRY_RUN" == "1" ]]; then
  printf '[dry-run] waiting for any %s GPUs from pool: %s\n' "$REQUIRED_GPUS" "${GPUS[*]}"
  printf '[dry-run] threshold: below %s%% VRAM used on each selected GPU; poll=%ss\n' "$VRAM_LIMIT_PERCENT" "$POLL_SECONDS"
  exit 0
fi

PID=""
cleanup() {
  local status=$?
  trap - EXIT INT TERM
  if [[ -n "$PID" ]] && kill -0 "$PID" 2>/dev/null; then
    echo "[stop] stopping the DDP process group..." >&2
    kill -TERM -- "-$PID" 2>/dev/null || kill -TERM "$PID" 2>/dev/null || true
    wait "$PID" 2>/dev/null || true
  fi
  exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT TERM

cd "$ROOT"
SELECTED_GPUS=""
while ! SELECTED_GPUS="$(find_free_gpus)"; do
  print_gpu_status
  echo "[wait] need ${REQUIRED_GPUS} GPUs below ${VRAM_LIMIT_PERCENT}% used from pool ${GPUS[*]}; rescanning in ${POLL_SECONDS}s..."
  sleep "$POLL_SECONDS"
done

read -r -a SELECTED_GPU_ARRAY <<< "$SELECTED_GPUS"
CUDA_DEVICES=$(IFS=,; printf '%s' "${SELECTED_GPU_ARRAY[*]}")
echo "[run] selected GPUs: ${SELECTED_GPU_ARRAY[*]} (CUDA_VISIBLE_DEVICES=$CUDA_DEVICES)."
echo "[run] launching SATLOSS7 RANGE100."
setsid env \
  PYTHONPATH="$ROOT/smart" \
  CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" \
  PYTHONUNBUFFERED=1 \
  OMP_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "$PYTHON" -m torch.distributed.run \
    --standalone --nnodes=1 --nproc_per_node=4 --master_port="$MASTER_PORT" \
    smart/train_satloss7.py \
    --config-name=drivaerml_satloss7_range100 \
    experiment.model_tag=smart-satloss7-range100 \
    experiment.name=DrivAerML_SMART_SATLOSS7_RANGE_100_250EP \
    experiment.epochs=250 \
    experiment.multi_gpu_strategy=ddp \
    experiment.batch_size=1 \
    experiment.init_ckpt= \
    experiment.resume_ckpt= \
    experiment.resume_full_state=False \
    wandb.project=smart_drivaerml \
    wandb.entity=parsa-vatani99-technical-university-of-munich &
PID=$!
echo "[run] started DDP launcher PID $PID; press Ctrl+C to stop the full process group."
wait "$PID"
