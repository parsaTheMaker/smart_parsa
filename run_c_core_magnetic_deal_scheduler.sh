#!/usr/bin/env bash
# Launch C-core DeAL continuations as their matched 300-epoch bases complete.
set -uo pipefail

ROOT="${ROOT:-/home/parsa/smart_parsa}"
PYTHON="${PYTHON:-/home/parsa/miniconda3/envs/smart/bin/python}"
DATA="${DATA:-/mnt/data/parsa/c_core_magnetic_fem_v1}"
GPU_IDS=(0 1 3)
PTV3_DDP_GPU_IDS=(0 3)
PTV3_DDP_VISIBLE="0,3"
PTV3_DDP_MASTER_PORT="${PTV3_DDP_MASTER_PORT:-29953}"
POLL_SECONDS="${POLL_SECONDS:-30}"
BASE_DONE_EPOCH="${BASE_DONE_EPOCH:-299}"
DEAL_DONE_EPOCH="${DEAL_DONE_EPOCH:-149}"
MAX_PER_GPU="${MAX_PER_GPU:-2}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-4}"
WANDB_ENTITY="${WANDB_ENTITY:-parsa-vatani99-technical-university-of-munich}"
STATE_DIR="$ROOT/results/c_core_magnetic/deal_scheduler"
LOG_DIR="$STATE_DIR/logs"
TRAINER="$ROOT/smart/train_c_core_magnetic_deal.py"
THRESHOLDS=(35 40 45 50)

cd "$ROOT" || exit 1
mkdir -p "$LOG_DIR"
export PYTHONPATH="$ROOT/smart"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export SMART_KNN_N_JOBS=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

JOBS=(smart ab_upt geo_fno pointnet2_ssg lno mspt transolverpp point_transformer_v3)
declare -A CONFIG BASE_STEM DEAL_STEM
CONFIG[smart]=c_core_magnetic_smart_deal_from_base
BASE_STEM[smart]=smart-c-core-magnetic-ccoremagnetic-s42
DEAL_STEM[smart]=smart-c-core-magnetic-deal-from-base-150ep-ccoremagnetic-s42
CONFIG[ab_upt]=c_core_magnetic_ab_upt_deal_from_base
BASE_STEM[ab_upt]=ab-upt-c-core-magnetic-ccoremagnetic-s42
DEAL_STEM[ab_upt]=ab-upt-c-core-magnetic-deal-from-base-150ep-ccoremagnetic-s42
CONFIG[geo_fno]=c_core_magnetic_geo_fno_deal_from_base
BASE_STEM[geo_fno]=geofno-c-core-magnetic-ccoremagnetic-s42
DEAL_STEM[geo_fno]=geofno-c-core-magnetic-deal-from-base-150ep-ccoremagnetic-s42
CONFIG[pointnet2_ssg]=c_core_magnetic_pointnet2_ssg_deal_from_base
BASE_STEM[pointnet2_ssg]=pointnet2-ssg-c-core-magnetic-ccoremagnetic-s42
DEAL_STEM[pointnet2_ssg]=pointnet2-ssg-c-core-magnetic-deal-from-base-150ep-ccoremagnetic-s42
CONFIG[lno]=c_core_magnetic_lno_deal_from_base
BASE_STEM[lno]=lno-c-core-magnetic-ccoremagnetic-s42
DEAL_STEM[lno]=lno-c-core-magnetic-deal-from-base-150ep-ccoremagnetic-s42
CONFIG[mspt]=c_core_magnetic_mspt_deal_from_base
BASE_STEM[mspt]=mspt-c-core-magnetic-ccoremagnetic-s42
DEAL_STEM[mspt]=mspt-c-core-magnetic-deal-from-base-150ep-ccoremagnetic-s42
CONFIG[transolverpp]=c_core_magnetic_transolverpp_deal_from_base
BASE_STEM[transolverpp]=transolverpp-c-core-magnetic-ccoremagnetic-s42
DEAL_STEM[transolverpp]=transolverpp-c-core-magnetic-deal-from-base-150ep-ccoremagnetic-s42
CONFIG[point_transformer_v3]=c_core_magnetic_point_transformer_v3_deal_from_base
BASE_STEM[point_transformer_v3]=point-transformer-v3-c-core-magnetic-ccoremagnetic-s42
DEAL_STEM[point_transformer_v3]=point-transformer-v3-c-core-magnetic-deal-from-base-150ep-ccoremagnetic-s42

checkpoint_epoch() {
  local path="$1"
  [[ -f "$path" ]] || { echo -1; return; }
  "$PYTHON" - "$path" <<'PY'
import sys
import torch
try:
    value = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
    print(int(value.get("epoch", -1)))
except Exception:
    print(-1)
PY
}

attempts_for() { cat "$STATE_DIR/$1.attempts" 2>/dev/null || echo 0; }

is_running() {
  local job="$1" pid
  pid=$(cat "$STATE_DIR/$job.pid" 2>/dev/null || true)
  [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null
}

active_on_gpu() {
  local target="$1" job count=0 gpu
  for job in "${JOBS[@]}"; do
    if is_running "$job"; then
      gpu=$(cat "$STATE_DIR/$job.gpu" 2>/dev/null || true)
      [[ ",$gpu," == *",$target,"* ]] && count=$((count + 1))
    fi
  done
  echo "$count"
}

select_ptv3_ddp_gpus() {
  local threshold="$1" gpu free active min_free=100
  for gpu in "${PTV3_DDP_GPU_IDS[@]}"; do
    active=$(active_on_gpu "$gpu")
    (( active < MAX_PER_GPU )) || return 1
    free=$(gpu_free_percent "$gpu")
    (( free > threshold )) || return 1
    (( free < min_free )) && min_free=$free
  done
  printf '%s %s\n' "$PTV3_DDP_VISIBLE" "$min_free"
}

gpu_free_percent() {
  local gpu="$1" total free
  read -r total free < <(nvidia-smi -i "$gpu" --query-gpu=memory.total,memory.free --format=csv,noheader,nounits | tr ',' ' ')
  awk -v free="$free" -v total="$total" 'BEGIN {printf "%d", (100 * free) / total}'
}

select_gpu() {
  local threshold="$1" gpu free active best_gpu="" best_free=-1
  for gpu in "${GPU_IDS[@]}"; do
    active=$(active_on_gpu "$gpu")
    (( active < MAX_PER_GPU )) || continue
    free=$(gpu_free_percent "$gpu")
    if (( free > threshold && free > best_free )); then
      best_gpu="$gpu"
      best_free="$free"
    fi
  done
  [[ -n "$best_gpu" ]] && printf '%s %s\n' "$best_gpu" "$best_free"
}

deal_complete() {
  [[ $(checkpoint_epoch "$ROOT/checkpoints/${DEAL_STEM[$1]}_last.pt") -ge "$DEAL_DONE_EPOCH" ]]
}

base_complete() {
  [[ $(checkpoint_epoch "$ROOT/checkpoints/${BASE_STEM[$1]}_last.pt") -ge "$BASE_DONE_EPOCH" \
     && -f "$ROOT/checkpoints/${BASE_STEM[$1]}_best.pt" ]]
}

ptv3_base_running() {
  pgrep -f '[t]rain_c_core_magnetic.py --config-name=c_core_magnetic_point_transformer_v3([[:space:]]|$)' \
    >/dev/null 2>&1
}

record_finished_processes() {
  local job pid attempts exit_code
  for job in "${JOBS[@]}"; do
    pid=$(cat "$STATE_DIR/$job.pid" 2>/dev/null || true)
    [[ "$pid" =~ ^[0-9]+$ ]] || continue
    if ! kill -0 "$pid" 2>/dev/null; then
      exit_code=$(cat "$STATE_DIR/$job.exit" 2>/dev/null || echo 1)
      rm -f "$STATE_DIR/$job.pid" "$STATE_DIR/$job.gpu"
      if deal_complete "$job"; then
        echo "[complete] $job reached DeAL epoch $(checkpoint_epoch "$ROOT/checkpoints/${DEAL_STEM[$job]}_last.pt")."
      else
        attempts=$(attempts_for "$job")
        echo "[retry] $job exited with status $exit_code after attempt $attempts; next launch requires >${THRESHOLDS[$attempts]:-50}% free VRAM." >&2
      fi
    fi
  done
}

launch_job() {
  local job="$1" gpu="$2" free="$3" attempts threshold init resume log
  local strategy="single"
  local -a launcher resume_args
  attempts=$(attempts_for "$job")
  threshold=${THRESHOLDS[$attempts]:-50}
  attempts=$((attempts + 1))
  echo "$attempts" >"$STATE_DIR/$job.attempts"
  init="$ROOT/checkpoints/${BASE_STEM[$job]}_best.pt"
  resume="$ROOT/checkpoints/${DEAL_STEM[$job]}_last.pt"
  log="$LOG_DIR/${job}_attempt${attempts}.log"
  if [[ -f "$resume" && $(checkpoint_epoch "$resume") -ge 0 ]]; then
    init=""
    resume_args=("experiment.init_ckpt=" "experiment.resume_ckpt=$resume" "experiment.resume_full_state=True")
    echo "[launch] $job resumes on physical GPU $gpu ($free% free, threshold >$threshold%)."
  else
    resume_args=("experiment.init_ckpt=$init" "experiment.resume_ckpt=" "experiment.resume_full_state=False")
    echo "[launch] $job starts from matched base best checkpoint on physical GPU $gpu ($free% free, threshold >$threshold%)."
  fi
  if [[ "$job" == "point_transformer_v3" ]]; then
    strategy="ddp"
    launcher=(
      "$PYTHON" -m torch.distributed.run --standalone --nnodes=1
      --nproc_per_node=2 --master_port="$PTV3_DDP_MASTER_PORT" "$TRAINER"
    )
  else
    launcher=("$PYTHON" "$TRAINER")
  fi
  (
    CUDA_VISIBLE_DEVICES="$gpu" "${launcher[@]}" \
      --config-name="${CONFIG[$job]}" \
      experiment.data_path="$DATA" \
      experiment.epochs=150 \
      experiment.multi_gpu_strategy="$strategy" \
      "${resume_args[0]}" "${resume_args[1]}" "${resume_args[2]}" \
      wandb.entity="$WANDB_ENTITY" \
      >>"$log" 2>&1
    echo "$?" >"$STATE_DIR/$job.exit"
  ) &
  echo "$!" >"$STATE_DIR/$job.pid"
  echo "$gpu" >"$STATE_DIR/$job.gpu"
}

echo "[setup] Materializing reusable C-core KDE-16 caches before concurrent DeAL training."
"$PYTHON" "$ROOT/smart/scripts/precompute_c_core_geometry_density.py" \
  --data-root "$DATA" --workers "${DENSITY_WORKERS:-16}" --knn-k 16 --estimator kde --cache-dtype float16

while true; do
  record_finished_processes
  pending=0
  launched=0
  printf '\n[%(%F %T)T] C-core DeAL scheduler\n' -1
  for job in "${JOBS[@]}"; do
    if deal_complete "$job"; then
      printf '  %-22s complete (epoch %s)\n' "$job" "$(checkpoint_epoch "$ROOT/checkpoints/${DEAL_STEM[$job]}_last.pt")"
      continue
    fi
    pending=$((pending + 1))
    if is_running "$job"; then
      printf '  %-22s running on GPU %s\n' "$job" "$(cat "$STATE_DIR/$job.gpu")"
      continue
    fi
    attempts=$(attempts_for "$job")
    if (( attempts >= MAX_ATTEMPTS )); then
      printf '  %-22s failed after %s attempts; inspect %s\n' "$job" "$attempts" "$LOG_DIR" >&2
      continue
    fi
    if ! base_complete "$job"; then
      printf '  %-22s waiting for base (last epoch %s/%s)\n' "$job" \
        "$(checkpoint_epoch "$ROOT/checkpoints/${BASE_STEM[$job]}_last.pt")" "$BASE_DONE_EPOCH"
      continue
    fi
    if [[ "$job" == "point_transformer_v3" ]] && ptv3_base_running; then
      printf '  %-22s base checkpoint complete; waiting for base process to release GPU 3\n' "$job"
      continue
    fi
    threshold=${THRESHOLDS[$attempts]:-50}
    if [[ "$job" == "point_transformer_v3" ]]; then
      candidate=$(select_ptv3_ddp_gpus "$threshold" || true)
    else
      candidate=$(select_gpu "$threshold" || true)
    fi
    if [[ -z "$candidate" ]]; then
      printf '  %-22s ready; waiting for >%s%% free VRAM\n' "$job" "$threshold"
      continue
    fi
    read -r gpu free <<<"$candidate"
    launch_job "$job" "$gpu" "$free"
    launched=$((launched + 1))
    # Let CUDA allocation settle before considering another process.
    sleep "${LAUNCH_SETTLE_SECONDS:-45}"
  done
  if (( pending == 0 )); then
    echo "All eight C-core DeAL continuations completed."
    exit 0
  fi
  sleep "$POLL_SECONDS"
done
