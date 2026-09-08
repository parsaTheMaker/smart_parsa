#!/usr/bin/env bash
# Fresh C-core Transolver++ DeAL continuation from the completed base weights.
set -euo pipefail

TASK_ROOT="${TASK_ROOT:-/home/parsa/smart_parsa}"
TASK_PYTHON="${TASK_PYTHON:-/home/parsa/miniconda3/envs/smart/bin/python}"
TASK_DATA="${TASK_DATA:-/mnt/data/parsa/c_core_magnetic_fem_v1}"
TASK_LOG_DIR="$TASK_ROOT/results/c_core_magnetic/training_logs"
TASK_LOG="$TASK_LOG_DIR/c_core_magnetic_transolverpp_deal_ddp.log"
TASK_BASE="$TASK_ROOT/checkpoints/transolverpp-c-core-magnetic-ccoremagnetic-s42_last.pt"

[[ -f "$TASK_BASE" ]] || { echo "Missing base checkpoint: $TASK_BASE" >&2; exit 1; }
mkdir -p "$TASK_LOG_DIR"
cd "$TASK_ROOT"

export PYTHONPATH="$TASK_ROOT/smart"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export SMART_KNN_N_JOBS=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

exec "$TASK_PYTHON" -m torch.distributed.run \
  --standalone --nnodes=1 --nproc_per_node=2 --master_port="${MASTER_PORT:-29683}" \
  smart/train_c_core_magnetic_deal.py \
  --config-name=c_core_magnetic_transolverpp_deal_from_base \
  experiment.data_path="$TASK_DATA" \
  experiment.epochs=150 \
  experiment.batch_size=1 \
  experiment.multi_gpu_strategy=ddp \
  experiment.init_ckpt="$TASK_BASE" \
  experiment.resume_ckpt= \
  experiment.resume_full_state=False \
  wandb.entity=parsa-vatani99-technical-university-of-munich \
  "$@" 2>&1 | tee -a "$TASK_LOG"
