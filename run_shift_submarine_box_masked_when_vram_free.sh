#!/usr/bin/env bash
set -Eeuo pipefail

# Launch the submarine box-masked SATLOSS run on the first GPU with
# strictly more than 95% free VRAM.

ROOT=/home/parsa/smart_parsa
PYTHON=/home/parsa/miniconda3/envs/smart/bin/python
POLL_SECONDS="${POLL_SECONDS:-30}"
GPUS=(0 1 2 3 4 5 6 7)

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi is required." >&2
  exit 1
fi
if ! [[ "$POLL_SECONDS" =~ ^[0-9]+$ ]] || (( POLL_SECONDS < 1 )); then
  echo "ERROR: POLL_SECONDS must be a positive integer." >&2
  exit 1
fi

gpu_memory() {
  local gpu="$1" used total
  IFS=',' read -r used total < <(
    nvidia-smi -i "$gpu" --query-gpu=memory.used,memory.total \
      --format=csv,noheader,nounits | tr -d ' '
  )
  if [[ -z "${used:-}" || -z "${total:-}" ]]; then
    echo "ERROR: could not read VRAM for GPU $gpu." >&2
    return 1
  fi
  printf '%s %s\n' "$used" "$total"
}

find_free_gpu() {
  local gpu used total free_percent
  for gpu in "${GPUS[@]}"; do
    read -r used total < <(gpu_memory "$gpu")
    free_percent=$(( (total - used) * 100 / total ))
    printf '[wait] GPU %s: %s MiB free of %s MiB (%s%% free)\n' \
      "$gpu" "$((total - used))" "$total" "$free_percent" >&2
    # Cross multiplication avoids rounding a borderline 95% value upward.
    if (( (total - used) * 100 > total * 95 )); then
      printf '%s\n' "$gpu"
      return 0
    fi
  done
  return 1
}

trap 'echo; echo "[wait] interrupted."; exit 130' INT TERM

cd "$ROOT"
while true; do
  if gpu="$(find_free_gpu)"; then
    echo "[run] launching on physical GPU $gpu"
    exec env \
      PYTHONPATH="$ROOT/smart" \
      CUDA_VISIBLE_DEVICES="$gpu" \
      PYTHONUNBUFFERED=1 \
      PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      "$PYTHON" \
      "$ROOT/smart/train_shift_submarine_satloss7.py" \
      --config-name=shift_submarine_satloss7_box_masked \
      experiment.model_tag=satloss7-box-masked-65k \
      experiment.name=SHIFT_SUBMARINE_SMART_SATLOSS7_BOX_MASKED_65K \
      experiment.epochs=150 \
      experiment.init_ckpt= \
      experiment.resume_ckpt= \
      experiment.resume_full_state=False \
      wandb.project=smart_shift_submarine \
      wandb.entity=parsa-vatani99-technical-university-of-munich
  fi
  echo "[wait] no GPU has more than 95% free VRAM; rescanning in ${POLL_SECONDS}s."
  sleep "$POLL_SECONDS"
done
