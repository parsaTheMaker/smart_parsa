#!/usr/bin/env bash
# Recover the failed C-core jobs sequentially on an otherwise empty GPU 0.
set -uo pipefail

ROOT="${ROOT:-/home/parsa/smart_parsa}"
PYTHON="${PYTHON:-/home/parsa/miniconda3/envs/smart/bin/python}"
DATA="${DATA:-/mnt/data/parsa/c_core_magnetic_fem_v1}"
STATE_DIR="$ROOT/results/c_core_magnetic/deal_scheduler"
LOG_DIR="$ROOT/results/c_core_magnetic/recovery_logs"
GPU_ID=0

mkdir -p "$STATE_DIR" "$LOG_DIR"
cd "$ROOT" || exit 1

export PYTHONPATH="$ROOT/smart"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export SMART_KNN_N_JOBS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cleanup_reservations() {
  local job owner
  for job in lno transolverpp; do
    owner=$(cat "$STATE_DIR/$job.pid" 2>/dev/null || true)
    if [[ "$owner" == "$$" ]]; then
      rm -f "$STATE_DIR/$job.pid" "$STATE_DIR/$job.gpu"
    fi
  done
}
trap cleanup_reservations EXIT INT TERM

# Reserve both scheduler slots on GPU 0. This prevents an independently ready
# DeAL continuation from being co-located before Transolver++ starts.
for job in lno transolverpp; do
  printf '%s\n' "$$" >"$STATE_DIR/$job.pid"
  printf '%s\n' "$GPU_ID" >"$STATE_DIR/$job.gpu"
done

echo "[$(date '+%F %T')] Resuming LNO DeAL on GPU $GPU_ID from its full checkpoint."
CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON" smart/train_c_core_magnetic_deal.py \
  --config-name=c_core_magnetic_lno_deal_from_base \
  experiment.data_path="$DATA" \
  experiment.epochs=150 \
  experiment.multi_gpu_strategy=single \
  experiment.init_ckpt= \
  experiment.resume_ckpt="$ROOT/checkpoints/lno-c-core-magnetic-deal-from-base-150ep-ccoremagnetic-s42_last.pt" \
  experiment.resume_full_state=True \
  wandb.entity=parsa-vatani99-technical-university-of-munich \
  >>"$LOG_DIR/lno_deal_gpu0_recovery.log" 2>&1
lno_status=$?
echo "[$(date '+%F %T')] LNO DeAL exited with status $lno_status."

# Transolver++ requires the GPU by itself. Wait if an unrelated process claims
# GPU 0 despite the scheduler reservation.
while true; do
  read -r total free < <(nvidia-smi -i "$GPU_ID" \
    --query-gpu=memory.total,memory.free --format=csv,noheader,nounits | tr ',' ' ')
  free_percent=$(awk -v f="$free" -v t="$total" 'BEGIN {printf "%d", 100*f/t}')
  (( free_percent >= 80 )) && break
  echo "[$(date '+%F %T')] GPU $GPU_ID is ${free_percent}% free; waiting for 80% before Transolver++."
  sleep 30
done

echo "[$(date '+%F %T')] Resuming Transolver++ base on GPU $GPU_ID from its full checkpoint."
CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON" smart/train_c_core_magnetic.py \
  --config-name=c_core_magnetic_transolverpp \
  experiment.data_path="$DATA" \
  experiment.epochs=300 \
  experiment.multi_gpu_strategy=single \
  experiment.init_ckpt= \
  experiment.resume_ckpt="$ROOT/checkpoints/transolverpp-c-core-magnetic-ccoremagnetic-s42_last.pt" \
  experiment.resume_full_state=True \
  wandb.entity=parsa-vatani99-technical-university-of-munich \
  >>"$LOG_DIR/transolverpp_base_gpu0_recovery.log" 2>&1
transolver_status=$?
echo "[$(date '+%F %T')] Transolver++ base exited with status $transolver_status."

(( lno_status == 0 && transolver_status == 0 ))
