#!/usr/bin/env bash
# Queue the seven non-SMART Pump DeAL continuations once their base checkpoint
# reaches epoch 298. Run with nohup; it keeps polling until every DeAL run ends.
set -uo pipefail

ROOT="${ROOT:-/mnt/data5/parsa/smart_parsa}"
PYTHON="${PYTHON:-/mnt/data5/parsa/conda_envs/smart-deal/bin/python}"
POLL_SECONDS="${POLL_SECONDS:-60}"
MIN_FREE_GPU_PERCENT="${MIN_FREE_GPU_PERCENT:-60}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-1}"
DRY_RUN=0
[[ "${1:-}" == "--once" ]] && ONCE=1 || ONCE=0
[[ "${1:-}" == "--dry-run" || "${2:-}" == "--dry-run" ]] && DRY_RUN=1

cd "$ROOT" || exit 1
mkdir -p logs/pump_deal_scheduler

export PYTHONPATH="$ROOT/smart"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

declare -A TRAINER CONFIG BASE_BEST BASE_LAST DEAL_STEM ATTEMPTS RESERVED_GPU
TRAINER[ab_upt]="smart/train_pump_ab_upt_deal.py"
CONFIG[ab_upt]="pump_ab_upt_deal_from_base"
BASE_BEST[ab_upt]="checkpoints/ab-upt-pump-servus06-base-v1-pump-s42_best.pt"
BASE_LAST[ab_upt]="checkpoints/ab-upt-pump-servus06-base-v1-pump-s42_last.pt"
DEAL_STEM[ab_upt]="ab-upt-pump-deal-from-base-150ep-pump-s42"

TRAINER[geo_fno]="smart/train_pump_geo_fno_deal.py"
CONFIG[geo_fno]="pump_geo_fno_deal_from_base"
BASE_BEST[geo_fno]="checkpoints/geofno-pump-medium-v2-raw16k-pump-s42_best.pt"
BASE_LAST[geo_fno]="checkpoints/geofno-pump-medium-v2-raw16k-pump-s42_last.pt"
DEAL_STEM[geo_fno]="geofno-pump-deal-medium-v2-from-base-150ep-pump-s42"

TRAINER[lno]="smart/train_pump_lno_deal.py"
CONFIG[lno]="pump_lno_deal_from_base"
BASE_BEST[lno]="checkpoints/lno-pump-servus06-base-v1-pump-s42_best.pt"
BASE_LAST[lno]="checkpoints/lno-pump-servus06-base-v1-pump-s42_last.pt"
DEAL_STEM[lno]="lno-pump-deal-from-base-150ep-pump-s42"

TRAINER[mspt]="smart/train_pump_mspt_deal.py"
CONFIG[mspt]="pump_mspt_deal_from_base"
BASE_BEST[mspt]="checkpoints/mspt-pump-servus06-base-v1-pump-s42_best.pt"
BASE_LAST[mspt]="checkpoints/mspt-pump-servus06-base-v1-pump-s42_last.pt"
DEAL_STEM[mspt]="mspt-pump-deal-from-base-150ep-pump-s42"

TRAINER[pointnet2]="smart/train_pump_pointnet2_ssg_deal.py"
CONFIG[pointnet2]="pump_pointnet2_ssg_deal_from_base"
BASE_BEST[pointnet2]="checkpoints/pointnet2-ssg-pump-servus06-base-v1-pump-s42_best.pt"
BASE_LAST[pointnet2]="checkpoints/pointnet2-ssg-pump-servus06-base-v1-pump-s42_last.pt"
DEAL_STEM[pointnet2]="pointnet2-ssg-pump-deal-from-base-150ep-pump-s42"

TRAINER[ptv3]="smart/train_pump_point_transformer_v3_deal.py"
CONFIG[ptv3]="pump_point_transformer_v3_deal_from_base"
BASE_BEST[ptv3]="checkpoints/point-transformer-v3-pump-servus06-base-v1-pump-s42_best.pt"
BASE_LAST[ptv3]="checkpoints/point-transformer-v3-pump-servus06-base-v1-pump-s42_last.pt"
DEAL_STEM[ptv3]="point-transformer-v3-pump-deal-from-base-150ep-pump-s42"

TRAINER[transolverpp]="smart/train_pump_transolverpp_deal.py"
CONFIG[transolverpp]="pump_transolverpp_deal_from_base"
BASE_BEST[transolverpp]="checkpoints/transolverpp-pump-servus06-base-v1-pump-s42_best.pt"
BASE_LAST[transolverpp]="checkpoints/transolverpp-pump-servus06-base-v1-pump-s42_last.pt"
DEAL_STEM[transolverpp]="transolverpp-pump-deal-from-base-150ep-pump-s42"

JOBS=(ab_upt geo_fno lno mspt pointnet2 ptv3 transolverpp)

checkpoint_epoch() {
  local checkpoint="$1"
  [[ -f "$checkpoint" ]] || { echo -1; return; }
  "$PYTHON" - "$checkpoint" <<'PY'
import sys
import torch
try:
    state = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
    print(int(state.get("epoch", -1)))
except Exception:
    print(-1)
PY
}

process_exists() {
  pgrep -f -- "$1" >/dev/null 2>&1
}

base_ready_for_deal() {
  local key="$1"
  # The base can finish its final epoch while DeAL starts from the selected
  # best checkpoint. The last checkpoint is only an eligibility witness.
  [[ -f "${BASE_BEST[$key]}" ]] || return 1
  [[ "$(checkpoint_epoch "${BASE_LAST[$key]}")" -ge 298 ]]
}

deal_finished() {
  [[ "$(checkpoint_epoch "checkpoints/${DEAL_STEM[$1]}_last.pt")" -ge 149 ]]
}

gpu_candidate() {
  local best="" best_free=-1 index total free pct
  while IFS=, read -r index total free; do
    index="${index// /}"; total="${total// /}"; free="${free// /}"
    [[ "$total" -gt 0 ]] || continue
    pct=$(( 100 * free / total ))
    # Existing training is permitted. Reserve each GPU after one launch in a
    # polling pass so concurrent launches cannot all select the same device.
    [[ "$pct" -ge "$MIN_FREE_GPU_PERCENT" ]] || continue
    [[ "${RESERVED_GPU[$index]:-0}" -eq 0 ]] || continue
    if [[ "$pct" -gt "$best_free" ]]; then best="$index"; best_free="$pct"; fi
  done < <(nvidia-smi --query-gpu=index,memory.total,memory.free --format=csv,noheader,nounits)
  [[ -n "$best" ]] && printf '%s %s\n' "$best" "$best_free"
}

launch_deal() {
  local key="$1" gpu="$2" base="${BASE_BEST[$key]}" resume="checkpoints/${DEAL_STEM[$key]}_last.pt"
  local log="logs/pump_deal_scheduler/${key}.log"
  local -a overrides=(
    "--config-name=${CONFIG[$key]}"
    "experiment.data_path=/mnt/data5/parsa/shift_pump_random1400_preprocessed"
    "experiment.multi_gpu_strategy=single"
    "wandb.mode=disabled"
  )
  if [[ -f "$resume" && "$(checkpoint_epoch "$resume")" -ge 0 ]]; then
    overrides+=("experiment.init_ckpt=" "experiment.resume_ckpt=$resume" "experiment.resume_full_state=True")
    echo "[launch] $key resumes DeAL on physical GPU $gpu" | tee -a "$log"
  else
    overrides+=("experiment.init_ckpt=$base" "experiment.resume_ckpt=" "experiment.resume_full_state=False")
    echo "[launch] $key starts DeAL from $base on physical GPU $gpu" | tee -a "$log"
  fi
  if [[ "$DRY_RUN" -eq 0 ]]; then
    nohup env CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" "${TRAINER[$key]}" "${overrides[@]}" >>"$log" 2>&1 &
  fi
  RESERVED_GPU[$gpu]=$(( ${RESERVED_GPU[$gpu]:-0} + 1 ))
  ATTEMPTS[$key]=$(( ${ATTEMPTS[$key]:-0} + 1 ))
}

while true; do
  pending=0
  RESERVED_GPU=()
  printf '\n[%(%F %T)T] Pump DeAL scheduler\n' -1
  for key in "${JOBS[@]}"; do
    if deal_finished "$key"; then
      printf '  %-14s complete\n' "$key"
      continue
    fi
    pending=$((pending + 1))
    if process_exists "${TRAINER[$key]}"; then
      printf '  %-14s running\n' "$key"
      continue
    fi
    if ! base_ready_for_deal "$key"; then
      printf '  %-14s waiting for base checkpoint epoch 298\n' "$key"
      continue
    fi
    if [[ ${ATTEMPTS[$key]:-0} -ge "$MAX_ATTEMPTS" ]]; then
      printf '  %-14s launch failed or exited; see logs/pump_deal_scheduler/%s.log\n' "$key" "$key"
      continue
    fi
    candidate="$(gpu_candidate || true)"
    if [[ -z "$candidate" ]]; then
      printf '  %-14s ready, waiting for an eligible GPU\n' "$key"
      continue
    fi
    read -r gpu free_pct <<<"$candidate"
    printf '  %-14s ready -> GPU %s (%s%% free)\n' "$key" "$gpu" "$free_pct"
    launch_deal "$key" "$gpu"
  done
  [[ "$pending" -eq 0 || "$ONCE" -eq 1 ]] && exit 0
  sleep "$POLL_SECONDS"
done
