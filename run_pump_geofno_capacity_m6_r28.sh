#!/usr/bin/env bash
# Isolated Pump Geo-FNO capacity study: six modes on a 28^3 latent grid.
set -uo pipefail

ROOT="${ROOT:-/mnt/data5/parsa/smart_parsa}"
PYTHON="${PYTHON:-/mnt/data5/parsa/conda_envs/smart-deal/bin/python}"
DATA="${DATA:-/mnt/data5/parsa/shift_pump_random1400_preprocessed}"
GPU_ID="${GPU_ID:-1}"
CHECKPOINT="$ROOT/checkpoints/geofno-pump-capacity-study-m6-r28-raw16k-pump-s42_last.pt"
FINAL_EPOCH=299

export PYTHONPATH="$ROOT/smart"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

checkpoint_epoch() {
  if [[ ! -f "$CHECKPOINT" ]]; then
    echo -1
    return
  fi
  "$PYTHON" - "$CHECKPOINT" <<'PY'
import sys
import torch

checkpoint = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
print(int(checkpoint.get("epoch", -1)))
PY
}

cd "$ROOT" || exit 1
while (( $(checkpoint_epoch) < FINAL_EPOCH )); do
  epoch=$(checkpoint_epoch)
  if (( epoch >= 0 )); then
    resume_args=(
      "experiment.init_ckpt="
      "experiment.resume_ckpt=$CHECKPOINT"
      "experiment.resume_full_state=True"
    )
    echo "[resume] Geo-FNO capacity study from durable epoch $epoch on physical GPU $GPU_ID."
  else
    resume_args=(
      "experiment.init_ckpt="
      "experiment.resume_ckpt="
      "experiment.resume_full_state=False"
    )
    echo "[start] Geo-FNO capacity study from scratch on physical GPU $GPU_ID."
  fi

  CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON" smart/train_pump_geo_fno.py \
    --config-name=pump_geo_fno_capacity_m6_r28 \
    experiment.data_path="$DATA" \
    experiment.epochs=300 \
    experiment.batch_size=6 \
    experiment.multi_gpu_strategy=single \
    "${resume_args[0]}" "${resume_args[1]}" "${resume_args[2]}" \
    wandb.mode=disabled
  status=$?
  (( $(checkpoint_epoch) >= FINAL_EPOCH )) && break
  echo "[retry] Training exited with status $status; retrying from the latest durable checkpoint in 60 seconds." >&2
  sleep 60
done

echo "Pump Geo-FNO capacity study complete at epoch $(checkpoint_epoch)."
