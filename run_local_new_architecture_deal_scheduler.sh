#!/usr/bin/env bash
# Queue local DrivAerML and Heat Exchanger DeAL continuations for AB-UPT/GeoFNO.
# One training process maximum per GPU; only fully idle GPUs with >=80% free VRAM qualify.
set -uo pipefail

ROOT="${ROOT:-/home/parsa/smart_parsa}"
PYTHON="${PYTHON:-/home/parsa/miniconda3/envs/smart/bin/python}"
POLL_SECONDS="${POLL_SECONDS:-60}"
FREE_GPU_PERCENT="${FREE_GPU_PERCENT:-80}"
WANDB_ENTITY="${WANDB_ENTITY:-parsa-vatani99-technical-university-of-munich}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-1}"
DRY_RUN=0
[[ "${1:-}" == "--once" ]] && ONCE=1 || ONCE=0
[[ "${1:-}" == "--dry-run" || "${2:-}" == "--dry-run" ]] && DRY_RUN=1

cd "$ROOT" || exit 1
mkdir -p logs/new_architecture_deal_scheduler

export PYTHONPATH="$ROOT/smart"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

declare -A TRAINER CONFIG BASE_BEST DEAL_STEM ATTEMPTS RESERVED_GPU
TRAINER[drivaerml_ab_upt]="smart/train_ab_upt_deal.py"
CONFIG[drivaerml_ab_upt]="drivaerml_ab_upt_deal_from_base"
BASE_BEST[drivaerml_ab_upt]="checkpoints/ab-upt-expanded-v3-drivaerml-s42_best.pt"
DEAL_STEM[drivaerml_ab_upt]="ab-upt-deal-from-base-150ep-drivaerml-s42"

TRAINER[drivaerml_geo_fno]="smart/train_geo_fno_deal.py"
CONFIG[drivaerml_geo_fno]="drivaerml_geo_fno_deal_from_base"
BASE_BEST[drivaerml_geo_fno]="checkpoints/geofno-medium-v2-raw65k-drivaerml-s42_best.pt"
DEAL_STEM[drivaerml_geo_fno]="geofno-deal-medium-v2-from-base-150ep-drivaerml-s42"

TRAINER[heat_ab_upt]="smart/train_toy_heat_exchange_ab_upt_deal.py"
CONFIG[heat_ab_upt]="toy_heat_exchange_ab_upt_deal_from_base"
BASE_BEST[heat_ab_upt]="checkpoints/ab-upt-toy-heat-exchange-expanded-v3-toyheatexchange-s42_best.pt"
DEAL_STEM[heat_ab_upt]="ab-upt-heat-exchange-deal-from-base-150ep-toyheatexchange-s42"

TRAINER[heat_geo_fno]="smart/train_toy_heat_exchange_geo_fno_deal.py"
CONFIG[heat_geo_fno]="toy_heat_exchange_geo_fno_deal_from_base"
BASE_BEST[heat_geo_fno]="checkpoints/geofno-heat-exchanger-medium-v2-raw65k-toyheatexchange-s42_best.pt"
DEAL_STEM[heat_geo_fno]="geofno-heat-exchanger-deal-medium-v2-from-base-150ep-toyheatexchange-s42"

JOBS=(drivaerml_ab_upt drivaerml_geo_fno heat_ab_upt heat_geo_fno)

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

process_exists() { pgrep -f -- "$1" >/dev/null 2>&1; }
deal_finished() { [[ "$(checkpoint_epoch "checkpoints/${DEAL_STEM[$1]}_last.pt")" -ge 149 ]]; }

gpu_candidate() {
  local best="" best_free=-1 index total free pct app_count
  while IFS=, read -r index total free; do
    index="${index// /}"; total="${total// /}"; free="${free// /}"
    [[ -z "${RESERVED_GPU[$index]:-}" ]] || continue
    [[ "$total" -gt 0 ]] || continue
    pct=$(( 100 * free / total ))
    app_count=$(nvidia-smi -i "$index" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c '[0-9]' || true)
    if [[ "$app_count" -eq 0 && "$pct" -ge "$FREE_GPU_PERCENT" && "$pct" -gt "$best_free" ]]; then
      best="$index"; best_free="$pct"
    fi
  done < <(nvidia-smi --query-gpu=index,memory.total,memory.free --format=csv,noheader,nounits)
  [[ -n "$best" ]] && printf '%s %s\n' "$best" "$best_free"
}

launch_deal() {
  local key="$1" gpu="$2" base="${BASE_BEST[$key]}" resume="checkpoints/${DEAL_STEM[$key]}_last.pt"
  local log="logs/new_architecture_deal_scheduler/${key}.log"
  local -a overrides=(
    "--config-name=${CONFIG[$key]}"
    "experiment.multi_gpu_strategy=single"
    "wandb.entity=$WANDB_ENTITY"
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
  RESERVED_GPU[$gpu]=1
  ATTEMPTS[$key]=$(( ${ATTEMPTS[$key]:-0} + 1 ))
}

while true; do
  pending=0
  RESERVED_GPU=()
  printf '\n[%(%F %T)T] Local new-architecture DeAL scheduler\n' -1
  for key in "${JOBS[@]}"; do
    if deal_finished "$key"; then
      printf '  %-20s complete\n' "$key"
      continue
    fi
    pending=$((pending + 1))
    if process_exists "${TRAINER[$key]}"; then
      printf '  %-20s running\n' "$key"
      continue
    fi
    if [[ ! -f "${BASE_BEST[$key]}" ]]; then
      printf '  %-20s waiting for base checkpoint\n' "$key"
      continue
    fi
    if [[ ${ATTEMPTS[$key]:-0} -ge "$MAX_ATTEMPTS" ]]; then
      printf '  %-20s launch failed or exited; see logs/new_architecture_deal_scheduler/%s.log\n' "$key" "$key"
      continue
    fi
    candidate="$(gpu_candidate || true)"
    if [[ -z "$candidate" ]]; then
      printf '  %-20s ready, waiting for a fully idle GPU\n' "$key"
      continue
    fi
    read -r gpu free_pct <<<"$candidate"
    printf '  %-20s ready -> GPU %s (%s%% free)\n' "$key" "$gpu" "$free_pct"
    launch_deal "$key" "$gpu"
  done
  [[ "$pending" -eq 0 || "$ONCE" -eq 1 ]] && exit 0
  sleep "$POLL_SECONDS"
done
