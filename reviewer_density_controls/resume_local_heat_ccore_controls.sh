#!/usr/bin/env bash
# Resume Heat and C-core density controls migrated from servus06.
set -uo pipefail

ROOT="${ROOT:-/home/parsa/smart_parsa}"
PYTHON="${PYTHON:-/home/parsa/miniconda3/envs/smart/bin/python}"
LOG_DIR="${LOG_DIR:-${ROOT}/logs/reviewer_density_distribution_controls/local_heat_ccore}"
POLL_SECONDS="${POLL_SECONDS:-60}"
EPOCHS=150

JOBS=(
  ccore_beta0_pair
  ccore_beta1_pair
  ccore_beta01_endpoints
  heat_beta0_pair
  heat_beta1_pair
  heat_beta01_endpoints
)

declare -A JOB_STATE JOB_PID JOB_GPU

task_for() {
  printf '%s' "${1%%_*}"
}

control_for() {
  printf '%s' "${1#*_}"
}

entrypoint_for() {
  case "$1" in
    heat) printf '%s' smart/train_toy_heat_exchange_satloss7.py ;;
    ccore) printf '%s' smart/train_c_core_magnetic_deal.py ;;
  esac
}

config_for() {
  case "$1" in
    heat) printf '%s' toy_heat_exchange_density_distribution_control ;;
    ccore) printf '%s' c_core_magnetic_density_distribution_control ;;
  esac
}

dataset_for() {
  case "$1" in
    heat) printf '%s' ToyHeatExchange ;;
    ccore) printf '%s' CCoreMagnetic ;;
  esac
}

data_for() {
  case "$1" in
    heat) printf '%s' /mnt/ssdraid/parsa/toy_heat_exchange_fem_v1 ;;
    ccore) printf '%s' /mnt/data/parsa/c_core_magnetic_fem_v1 ;;
  esac
}

checkpoint_for() {
  local task="$1" control="$2" normalized
  normalized="${control//_/-}"
  case "$task" in
    heat)
      printf '%s/checkpoints/smart-heat-exchange-deal-density-control-density-control-%s-v1-toyheatexchange-s42_last.pt' \
        "$ROOT" "$normalized"
      ;;
    ccore)
      printf '%s/checkpoints/smart-c-core-magnetic-deal-density-control-density-control-%s-v1-ccoremagnetic-s42_last.pt' \
        "$ROOT" "$normalized"
      ;;
  esac
}

beta_for() {
  local control="$1" view="$2"
  case "$control:$view" in
    beta0_pair:primary|beta0_pair:secondary|beta01_endpoints:primary) printf '0' ;;
    beta1_pair:primary|beta1_pair:secondary|beta01_endpoints:secondary) printf '1' ;;
  esac
}

gpu_is_reserved() {
  local gpu="$1" job
  for job in "${JOBS[@]}"; do
    if [[ "${JOB_STATE[$job]:-}" == running && "${JOB_GPU[$job]:-}" == "$gpu" ]]; then
      return 0
    fi
  done
  return 1
}

choose_gpu() {
  local task="$1" gpu free threshold
  local candidates
  case "$task" in
    ccore) candidates="5 7 4 6 2 0 1 3"; threshold=19500 ;;
    heat) candidates="7 5 4 6 2 0 1 3"; threshold=12500 ;;
  esac
  for gpu in $candidates; do
    gpu_is_reserved "$gpu" && continue
    free="$(nvidia-smi -i "$gpu" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
    (( free >= threshold )) && { printf '%s' "$gpu"; return 0; }
  done
  return 1
}

checkpoint_epoch() {
  "$PYTHON" - "$1" <<'PY'
import sys
import torch

checkpoint = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
print(int(checkpoint.get("epoch", -1)))
PY
}

validate_inputs() {
  local job task control checkpoint data failed=0
  [[ -x "$PYTHON" ]] || { printf 'Missing Python: %s\n' "$PYTHON" >&2; return 1; }
  command -v nvidia-smi >/dev/null || { printf 'nvidia-smi is required.\n' >&2; return 1; }
  for job in "${JOBS[@]}"; do
    task="$(task_for "$job")"
    control="$(control_for "$job")"
    checkpoint="$(checkpoint_for "$task" "$control")"
    data="$(data_for "$task")"
    [[ -f "$ROOT/$(entrypoint_for "$task")" ]] || { printf 'Missing entrypoint for %s\n' "$task" >&2; failed=1; }
    [[ -f "$ROOT/smart/config/$(config_for "$task").yaml" ]] || { printf 'Missing config for %s\n' "$task" >&2; failed=1; }
    [[ -d "$data" ]] || { printf 'Missing data: %s\n' "$data" >&2; failed=1; }
    [[ -f "$checkpoint" ]] || { printf 'Missing migrated checkpoint: %s\n' "$checkpoint" >&2; failed=1; }
  done
  return "$failed"
}

run_job() {
  local job="$1" gpu="$2" task control checkpoint status=0
  local entrypoint config dataset data tag log
  task="$(task_for "$job")"
  control="$(control_for "$job")"
  checkpoint="$(checkpoint_for "$task" "$control")"
  entrypoint="$(entrypoint_for "$task")"
  config="$(config_for "$task")"
  dataset="$(dataset_for "$task")"
  data="$(data_for "$task")"
  tag="density-control-${control}-v1"
  log="$LOG_DIR/${job}.gpu${gpu}.log"

  printf '[%s] resuming %s from epoch %s on physical GPU %s\n' \
    "$(date --iso-8601=seconds)" "$job" "$(checkpoint_epoch "$checkpoint")" "$gpu" >"$log"
  (
    cd "$ROOT" || exit 1
    env \
      CUDA_VISIBLE_DEVICES="$gpu" \
      PYTHONPATH="$ROOT/smart" \
      PYTHONUNBUFFERED=1 \
      OMP_NUM_THREADS=1 \
      OPENBLAS_NUM_THREADS=1 \
      MKL_NUM_THREADS=1 \
      NUMEXPR_NUM_THREADS=1 \
      SMART_KNN_N_JOBS=1 \
      PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      "$PYTHON" "$entrypoint" \
        --config-name="$config" \
        "experiment.name=${dataset}_${control}" \
        "experiment.model_tag=${tag}" \
        "experiment.data_path=${data}" \
        "experiment.epochs=${EPOCHS}" \
        "experiment.random_seed=42" \
        "experiment.multi_gpu_strategy=single" \
        "experiment.train_shared_shift_sampling_mode=" \
        "experiment.randomize_primary_inverse_density_beta=True" \
        "experiment.primary_inverse_density_beta_min=$(beta_for "$control" primary)" \
        "experiment.primary_inverse_density_beta_max=$(beta_for "$control" primary)" \
        "experiment.randomize_secondary_inverse_density_beta=True" \
        "experiment.secondary_inverse_density_beta_min=$(beta_for "$control" secondary)" \
        "experiment.secondary_inverse_density_beta_max=$(beta_for "$control" secondary)" \
        "experiment.init_ckpt=" \
        "experiment.resume_ckpt=${checkpoint}" \
        "experiment.resume_full_state=True"
  ) >>"$log" 2>&1 || status=$?

  printf '%s\n' "$status" >"$LOG_DIR/${job}.exit_code"
  printf '[%s] finished %s on GPU %s with status %s\n' \
    "$(date --iso-8601=seconds)" "$job" "$gpu" "$status" >>"$log"
  return "$status"
}

validate_inputs || exit $?
mkdir -p "$LOG_DIR"

for job in "${JOBS[@]}"; do
  checkpoint="$(checkpoint_for "$(task_for "$job")" "$(control_for "$job")")"
  if (( $(checkpoint_epoch "$checkpoint") >= EPOCHS - 1 )); then
    JOB_STATE[$job]=done
  else
    JOB_STATE[$job]=pending
  fi
done

while :; do
  for job in "${JOBS[@]}"; do
    [[ "${JOB_STATE[$job]}" == running ]] || continue
    kill -0 "${JOB_PID[$job]}" 2>/dev/null && continue
    status=0
    wait "${JOB_PID[$job]}" || status=$?
    if (( status == 0 )); then JOB_STATE[$job]=done; else JOB_STATE[$job]=failed; fi
    printf '[%s] reaped %s status=%s\n' "$(date --iso-8601=seconds)" "$job" "$status"
  done

  for job in "${JOBS[@]}"; do
    [[ "${JOB_STATE[$job]}" == pending ]] || continue
    task="$(task_for "$job")"
    gpu="$(choose_gpu "$task" || true)"
    [[ -n "$gpu" ]] || continue
    run_job "$job" "$gpu" &
    JOB_PID[$job]=$!
    JOB_GPU[$job]="$gpu"
    JOB_STATE[$job]=running
    printf '[%s] allocated %s to GPU %s (pid %s)\n' \
      "$(date --iso-8601=seconds)" "$job" "$gpu" "${JOB_PID[$job]}"
    sleep 5
  done

  pending=0; running=0; done_count=0; failed=0
  for job in "${JOBS[@]}"; do
    case "${JOB_STATE[$job]}" in
      pending) pending=$((pending + 1)) ;;
      running) running=$((running + 1)) ;;
      done) done_count=$((done_count + 1)) ;;
      failed) failed=$((failed + 1)) ;;
    esac
  done
  printf '[%s] state pending=%s running=%s done=%s failed=%s\n' \
    "$(date --iso-8601=seconds)" "$pending" "$running" "$done_count" "$failed"
  (( pending == 0 && running == 0 )) && break
  sleep "$POLL_SECONDS"
done

(( failed == 0 ))
