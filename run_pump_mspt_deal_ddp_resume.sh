#!/usr/bin/env bash
# Resume Pump MSPT DeAL with two DDP ranks until the final epoch is durable.
set -uo pipefail

ROOT="${ROOT:-/mnt/data5/parsa/smart_parsa}"
PYTHON="${PYTHON:-/mnt/data5/parsa/conda_envs/smart-deal/bin/python}"
DATA="${DATA:-/mnt/data5/parsa/shift_pump_random1400_preprocessed}"
CHECKPOINT="$ROOT/checkpoints/mspt-pump-deal-from-base-150ep-pump-s42_last.pt"
GPU_IDS="${GPU_IDS:-0,2}"
MASTER_PORT="${MASTER_PORT:-29942}"
FINAL_EPOCH=149

export PYTHONPATH="$ROOT/smart"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

checkpoint_epoch() {
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
  echo "[resume] Pump MSPT DeAL from durable epoch $epoch on physical GPUs $GPU_IDS."
  CUDA_VISIBLE_DEVICES="$GPU_IDS" "$PYTHON" -m torch.distributed.run \
    --standalone --nnodes=1 --nproc_per_node=2 --master_port="$MASTER_PORT" \
    smart/train_pump_mspt_deal.py \
    --config-name=pump_mspt_deal_from_base \
    experiment.data_path="$DATA" \
    experiment.epochs=150 \
    experiment.batch_size=1 \
    experiment.multi_gpu_strategy=ddp \
    experiment.init_ckpt= \
    experiment.resume_ckpt="$CHECKPOINT" \
    experiment.resume_full_state=True \
    wandb.mode=disabled
  status=$?
  (( $(checkpoint_epoch) >= FINAL_EPOCH )) && break
  echo "[retry] torchrun exited with status $status; retrying from the latest durable checkpoint in 60 seconds." >&2
  sleep 60
done

echo "Pump MSPT DeAL complete at epoch $(checkpoint_epoch)."
