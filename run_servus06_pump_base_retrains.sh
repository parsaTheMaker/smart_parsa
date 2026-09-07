#!/usr/bin/env bash
# Reliably retrain the four Pump baselines whose earlier runs wrote checkpoints
# outside the repository because their working directory was /home/parsa.
set -uo pipefail

ROOT="${ROOT:-/mnt/data5/parsa/smart_parsa}"
PYTHON="${PYTHON:-/mnt/data5/parsa/conda_envs/smart-deal/bin/python}"
POLL_SECONDS="${POLL_SECONDS:-60}"
FREE_GPU_PERCENT="${FREE_GPU_PERCENT:-85}"
DRY_RUN=0
[[ "${1:-}" == "--once" ]] && ONCE=1 || ONCE=0
[[ "${1:-}" == "--dry-run" || "${2:-}" == "--dry-run" ]] && DRY_RUN=1

cd "$ROOT" || exit 1
mkdir -p logs/pump_base_retrains

export PYTHONPATH="$ROOT/smart"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Preferred assignments avoid sharing a GPU between base retrains. If a
# preferred GPU remains busy, a pending job can use any other safely idle GPU.
declare -A GPU TRAINER CONFIG BATCH EXTRA STEM ATTEMPTS
GPU[mspt]=0
TRAINER[mspt]="smart/train_pump_mspt.py"
CONFIG[mspt]="pump_mspt"
BATCH[mspt]=2
EXTRA[mspt]=""
STEM[mspt]="mspt-pump-servus06-base-v1-pump-s42"

GPU[pointnet2]=1
TRAINER[pointnet2]="smart/train_pump_pointnet2_ssg.py"
CONFIG[pointnet2]="pump_pointnet2_ssg"
BATCH[pointnet2]=4
EXTRA[pointnet2]=""
STEM[pointnet2]="pointnet2-ssg-pump-servus06-base-v1-pump-s42"

GPU[transolverpp]=2
TRAINER[transolverpp]="smart/train_pump_transolverpp.py"
CONFIG[transolverpp]="pump_transolverpp"
BATCH[transolverpp]=2
EXTRA[transolverpp]=""
STEM[transolverpp]="transolverpp-pump-servus06-base-v1-pump-s42"

GPU[lno]=3
TRAINER[lno]="smart/train_pump_lno.py"
CONFIG[lno]="pump_lno"
BATCH[lno]=4
# LNO previously produced FP16 non-finite losses. BF16 plus clipping avoids
# that narrow-exponent failure mode while preserving the architecture/task.
EXTRA[lno]="experiment.gradient_norm=1.0"
STEM[lno]="lno-pump-servus06-base-v1-pump-s42"

JOBS=(mspt pointnet2 transolverpp lno)

checkpoint_epoch() {
  local checkpoint="$1"
  [[ -f "$checkpoint" ]] || { echo -1; return; }
  "$PYTHON" - "$checkpoint" <<'PY'
import sys
import torch
try:
    print(int(torch.load(sys.argv[1], map_location="cpu", weights_only=False).get("epoch", -1)))
except Exception:
    print(-1)
PY
}

process_exists() { pgrep -f -- "$1" >/dev/null 2>&1; }

gpu_is_idle() {
  local gpu="$1" total free pct processes
  IFS=, read -r total free < <(nvidia-smi -i "$gpu" --query-gpu=memory.total,memory.free --format=csv,noheader,nounits)
  total="${total// /}"; free="${free// /}"
  pct=$((100 * free / total))
  processes=$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c '[0-9]' || true)
  [[ "$processes" -eq 0 && "$pct" -ge "$FREE_GPU_PERCENT" ]]
}

choose_idle_gpu() {
  local preferred="$1" candidate
  if gpu_is_idle "$preferred"; then
    printf '%s\n' "$preferred"
    return
  fi
  while read -r candidate; do
    [[ "$candidate" == "$preferred" ]] && continue
    if gpu_is_idle "$candidate"; then
      printf '%s\n' "$candidate"
      return
    fi
  done < <(nvidia-smi --query-gpu=index --format=csv,noheader,nounits | tr -d ' ')
}

retrain_finished() { [[ "$(checkpoint_epoch "checkpoints/${STEM[$1]}_last.pt")" -ge 299 ]]; }

launch() {
  local key="$1" gpu="$2" log="logs/pump_base_retrains/${key}.log"
  local -a overrides=(
    "--config-name=${CONFIG[$key]}"
    "experiment.name=SHIFT_PUMP_${key^^}_BASE_RETRAIN"
    "experiment.model_tag=servus06-base-v1"
    "experiment.data_path=/mnt/data5/parsa/shift_pump_random1400_preprocessed"
    "experiment.epochs=300"
    "experiment.batch_size=${BATCH[$key]}"
    "experiment.multi_gpu_strategy=single"
    "experiment.num_workers=2"
    "experiment.prefetch_factor=1"
    "experiment.cuda_batch_prefetch=False"
    "experiment.amp=True"
    "experiment.precision=bfloat16"
    "experiment.init_ckpt="
    "experiment.resume_ckpt="
    "experiment.resume_full_state=False"
    "+wandb.mode=disabled"
  )
  [[ -z "${EXTRA[$key]}" ]] || overrides+=("${EXTRA[$key]}")
  echo "[launch] $key -> physical GPU $gpu, checkpoint stem=${STEM[$key]}" | tee -a "$log"
  if [[ "$DRY_RUN" -eq 0 ]]; then
    # The subshell's explicit cd is the checkpoint-path fix.
    nohup bash -lc "cd '$ROOT' && exec env CUDA_VISIBLE_DEVICES='$gpu' PYTHONPATH='$ROOT/smart' PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True '$PYTHON' '${TRAINER[$key]}' ${overrides[*]}" >>"$log" 2>&1 &
  fi
  ATTEMPTS[$key]=$(( ${ATTEMPTS[$key]:-0} + 1 ))
}

while true; do
  pending=0
  printf '\n[%(%F %T)T] Pump base-retrain scheduler\n' -1
  for key in "${JOBS[@]}"; do
    if retrain_finished "$key"; then
      printf '  %-14s complete\n' "$key"
      continue
    fi
    pending=$((pending + 1))
    if process_exists "${TRAINER[$key]}"; then
      printf '  %-14s running\n' "$key"
      continue
    fi
    if [[ ${ATTEMPTS[$key]:-0} -ge 1 ]]; then
      printf '  %-14s stopped or failed; inspect logs/pump_base_retrains/%s.log\n' "$key" "$key"
      continue
    fi
    gpu="$(choose_idle_gpu "${GPU[$key]}" || true)"
    if [[ -n "$gpu" ]]; then
      printf '  %-14s ready -> GPU %s\n' "$key" "$gpu"
      launch "$key" "$gpu"
    else
      printf '  %-14s waiting for a fully idle GPU (preferred %s)\n' "$key" "${GPU[$key]}"
    fi
  done
  [[ "$pending" -eq 0 || "$ONCE" -eq 1 ]] && exit 0
  sleep "$POLL_SECONDS"
done
