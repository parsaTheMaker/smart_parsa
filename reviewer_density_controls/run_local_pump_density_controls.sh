#!/usr/bin/env bash
# Queue two Pump density controls on otherwise-idle local GPUs, preferring GPU 5.
set -uo pipefail

ROOT="${ROOT:-/home/parsa/smart_parsa}"
PYTHON="${PYTHON:-/home/parsa/miniconda3/envs/smart/bin/python}"
DATA="${DATA:-/mnt/data/parsa/shift_pump_random1400_preprocessed}"
CHECKPOINT="${CHECKPOINT:-${ROOT}/checkpoints/smart-pump-random1400-base-16k-pump-s42_last.pt}"
LOG_DIR="${LOG_DIR:-${ROOT}/logs/reviewer_density_distribution_controls/local_pump}"
POLL_SECONDS="${POLL_SECONDS:-60}"
MIN_FREE_MIB="${MIN_FREE_MIB:-23000}"
MAX_IDLE_UTIL="${MAX_IDLE_UTIL:-15}"
EPOCHS=150
GPU_IDS=(5 7 4 6 2 0 1 3)
CONTROLS=(beta0_pair beta1_pair)

declare -A PRIMARY_BETA=(
  [beta0_pair]=0
  [beta1_pair]=1
)
declare -A SECONDARY_BETA=(
  [beta0_pair]=0
  [beta1_pair]=1
)
declare -A JOB_STATE JOB_PID JOB_GPU

validate_inputs() {
  [[ -x "$PYTHON" ]] || { printf 'Missing Python: %s\n' "$PYTHON" >&2; return 1; }
  [[ -f "$ROOT/smart/train_pump_satloss7.py" ]] || { printf 'Missing Pump entrypoint.\n' >&2; return 1; }
  [[ -f "$ROOT/smart/config/pump_density_distribution_control.yaml" ]] || { printf 'Missing Pump control config.\n' >&2; return 1; }
  [[ -d "$DATA" ]] || { printf 'Missing Pump data: %s\n' "$DATA" >&2; return 1; }
  [[ -f "$CHECKPOINT" ]] || { printf 'Missing Pump checkpoint: %s\n' "$CHECKPOINT" >&2; return 1; }
  command -v nvidia-smi >/dev/null || { printf 'nvidia-smi is required.\n' >&2; return 1; }
}

gpu_is_available() {
  local gpu="$1" free util process_count control
  for control in "${CONTROLS[@]}"; do
    if [[ "${JOB_STATE[$control]:-}" == running && "${JOB_GPU[$control]:-}" == "$gpu" ]]; then
      return 1
    fi
  done
  read -r free util < <(
    nvidia-smi -i "$gpu" --query-gpu=memory.free,utilization.gpu \
      --format=csv,noheader,nounits | tr -d ','
  )
  process_count="$(nvidia-smi -i "$gpu" --query-compute-apps=pid \
    --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d' | wc -l)"
  (( free >= MIN_FREE_MIB && util <= MAX_IDLE_UTIL && process_count == 0 ))
}

choose_gpu() {
  local gpu
  for gpu in "${GPU_IDS[@]}"; do
    gpu_is_available "$gpu" && { printf '%s' "$gpu"; return 0; }
  done
  return 1
}

run_control() {
  local control="$1" gpu="$2" status=0
  local tag="density-control-${control//_/-}-v1"
  local log="$LOG_DIR/pump_${control}.gpu${gpu}.log"

  printf '[%s] starting pump_%s on physical GPU %s\n' \
    "$(date --iso-8601=seconds)" "$control" "$gpu" >"$log"
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
      "$PYTHON" smart/train_pump_satloss7.py \
        --config-name=pump_density_distribution_control \
        "experiment.name=Pump_${control}" \
        "experiment.model_tag=${tag}" \
        "experiment.data_path=${DATA}" \
        "experiment.epochs=${EPOCHS}" \
        "experiment.random_seed=42" \
        "experiment.multi_gpu_strategy=single" \
        "experiment.train_shared_shift_sampling_mode=" \
        "experiment.randomize_primary_inverse_density_beta=True" \
        "experiment.primary_inverse_density_beta_min=${PRIMARY_BETA[$control]}" \
        "experiment.primary_inverse_density_beta_max=${PRIMARY_BETA[$control]}" \
        "experiment.randomize_secondary_inverse_density_beta=True" \
        "experiment.secondary_inverse_density_beta_min=${SECONDARY_BETA[$control]}" \
        "experiment.secondary_inverse_density_beta_max=${SECONDARY_BETA[$control]}" \
        "experiment.init_ckpt=${CHECKPOINT}" \
        "experiment.resume_ckpt=" \
        "experiment.resume_full_state=False"
  ) >>"$log" 2>&1 || status=$?

  printf '%s\n' "$status" >"$LOG_DIR/pump_${control}.exit_code"
  printf '[%s] finished pump_%s on GPU %s with status %s\n' \
    "$(date --iso-8601=seconds)" "$control" "$gpu" "$status" >>"$log"
  return "$status"
}

validate_inputs || exit $?
mkdir -p "$LOG_DIR"

for control in "${CONTROLS[@]}"; do
  JOB_STATE[$control]=pending
done

while :; do
  for control in "${CONTROLS[@]}"; do
    [[ "${JOB_STATE[$control]}" == running ]] || continue
    kill -0 "${JOB_PID[$control]}" 2>/dev/null && continue
    status=0
    wait "${JOB_PID[$control]}" || status=$?
    if (( status == 0 )); then
      JOB_STATE[$control]=done
    else
      JOB_STATE[$control]=failed
    fi
    printf '[%s] pump_%s finished with status %s\n' \
      "$(date --iso-8601=seconds)" "$control" "$status"
  done

  for control in "${CONTROLS[@]}"; do
    [[ "${JOB_STATE[$control]}" == pending ]] || continue
    gpu="$(choose_gpu || true)"
    [[ -n "$gpu" ]] || break
    run_control "$control" "$gpu" &
    JOB_PID[$control]=$!
    JOB_GPU[$control]="$gpu"
    JOB_STATE[$control]=running
    printf '[%s] allocated pump_%s to GPU %s (pid %s)\n' \
      "$(date --iso-8601=seconds)" "$control" "$gpu" "${JOB_PID[$control]}"
    sleep 5
  done

  pending=0
  running=0
  failed=0
  for control in "${CONTROLS[@]}"; do
    case "${JOB_STATE[$control]}" in
      pending) pending=$((pending + 1)) ;;
      running) running=$((running + 1)) ;;
      failed) failed=$((failed + 1)) ;;
    esac
  done
  printf '[%s] state pending=%s running=%s failed=%s\n' \
    "$(date --iso-8601=seconds)" "$pending" "$running" "$failed"
  (( pending == 0 && running == 0 )) && break
  sleep "$POLL_SECONDS"
done

(( failed == 0 ))
