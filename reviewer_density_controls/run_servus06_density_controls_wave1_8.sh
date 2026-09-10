#!/usr/bin/env bash
# Launch eight independent weight-initialized controls, two processes per GPU.
set -uo pipefail

ROOT="${ROOT:-/mnt/data5/parsa/smart_parsa}"
PYTHON="${PYTHON:-/mnt/data5/parsa/conda_envs/smart-deal/bin/python}"
LOG_DIR="${LOG_DIR:-${ROOT}/logs/reviewer_density_distribution_controls/wave1_8}"
POLL_SECONDS="${POLL_SECONDS:-60}"
RESERVE_MIB="${RESERVE_MIB:-4096}"
EPOCHS=150

declare -A PEAK_MIB=(
  [car]=19124
  [pump]=20580
  [heat]=9986
  [ccore]=17054
)
declare -A PRIMARY_BETA=(
  [beta0_pair]=0
  [beta01_endpoints]=0
)
declare -A SECONDARY_BETA=(
  [beta0_pair]=0
  [beta01_endpoints]=1
)

task_entrypoint() {
  case "$1" in
    car) printf '%s' smart/train_satloss7.py ;;
    pump) printf '%s' smart/train_pump_satloss7.py ;;
    heat) printf '%s' smart/train_toy_heat_exchange_satloss7.py ;;
    ccore) printf '%s' smart/train_c_core_magnetic_deal.py ;;
  esac
}

task_config() {
  case "$1" in
    car) printf '%s' drivaerml_density_distribution_control ;;
    pump) printf '%s' pump_density_distribution_control ;;
    heat) printf '%s' toy_heat_exchange_density_distribution_control ;;
    ccore) printf '%s' c_core_magnetic_density_distribution_control ;;
  esac
}

task_dataset() {
  case "$1" in
    car) printf '%s' DrivAerML ;;
    pump) printf '%s' Pump ;;
    heat) printf '%s' ToyHeatExchange ;;
    ccore) printf '%s' CCoreMagnetic ;;
  esac
}

task_data() {
  case "$1" in
    car) printf '%s' /mnt/data5/parsa/drivaerml_preprocessed ;;
    pump) printf '%s' /mnt/data5/parsa/shift_pump_random1400_preprocessed ;;
    heat) printf '%s' /mnt/data5/parsa/toy_heat_exchange_fem_v1 ;;
    ccore) printf '%s' /mnt/data5/parsa/c_core_magnetic_fem_v1 ;;
  esac
}

task_checkpoint() {
  case "$1" in
    car) printf '%s' "$ROOT/checkpoints/smart-smart-drivaerml-131k16kwr-drivaerml-s42_last.pt" ;;
    pump) printf '%s' "$ROOT/checkpoints/smart-pump-random1400-base-16k-pump-s42_last.pt" ;;
    heat) printf '%s' "$ROOT/checkpoints/smart-toy-heat-exchange-heat-exchange-base-ratio-aligned-toyheatexchange-s42_last.pt" ;;
    ccore) printf '%s' "$ROOT/checkpoints/smart-c-core-magnetic-ccoremagnetic-s42_last.pt" ;;
  esac
}

validate_job() {
  local task="$1" control="$2" config data checkpoint dataset_key
  config="$(task_config "$task")"
  data="$(task_data "$task")"
  checkpoint="$(task_checkpoint "$task")"
  dataset_key="$(task_dataset "$task")"
  dataset_key="${dataset_key,,}"
  [[ -x "$PYTHON" ]] || { printf 'Missing Python: %s\n' "$PYTHON" >&2; return 1; }
  [[ -f "$ROOT/smart/config/$config.yaml" ]] || { printf 'Missing config: %s\n' "$config" >&2; return 1; }
  [[ -f "$data/.reviewer_density_controls_ready" ]] || { printf 'Dataset not verified: %s\n' "$data" >&2; return 1; }
  [[ -f "$checkpoint" ]] || { printf 'Missing checkpoint: %s\n' "$checkpoint" >&2; return 1; }
  if compgen -G "$ROOT/checkpoints/*density-control-${control}-v1*${dataset_key}*s42*.pt" >/dev/null; then
    printf 'Output checkpoint already exists for %s/%s; refusing to overwrite it.\n' "$task" "$control" >&2
    return 1
  fi
}

gpu_free_mib() {
  nvidia-smi -i "$1" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' '
}

wait_for_pair_capacity() {
  local gpu="$1" first_task="$2" second_task="$3" free
  local required=$(( PEAK_MIB[$first_task] + PEAK_MIB[$second_task] + RESERVE_MIB ))
  while :; do
    free="$(gpu_free_mib "$gpu")"
    if [[ "$free" =~ ^[0-9]+$ ]] && (( free >= required )); then
      printf '[%s] GPU %s ready: %s MiB free, %s MiB required.\n' \
        "$(date --iso-8601=seconds)" "$gpu" "$free" "$required"
      return 0
    fi
    printf '[%s] GPU %s waiting: %s MiB free, %s MiB required.\n' \
      "$(date --iso-8601=seconds)" "$gpu" "${free:-unknown}" "$required"
    sleep "$POLL_SECONDS"
  done
}

run_job() {
  local gpu="$1" task="$2" control="$3"
  local entrypoint config dataset data checkpoint tag label log status=0
  entrypoint="$(task_entrypoint "$task")"
  config="$(task_config "$task")"
  dataset="$(task_dataset "$task")"
  data="$(task_data "$task")"
  checkpoint="$(task_checkpoint "$task")"
  tag="density-control-${control}-v1"
  label="${task}_${control}"
  log="$LOG_DIR/${label}.gpu${gpu}.log"

  printf '[%s] starting %s on physical GPU %s\n' "$(date --iso-8601=seconds)" "$label" "$gpu" >"$log"
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
        "experiment.primary_inverse_density_beta_min=${PRIMARY_BETA[$control]}" \
        "experiment.primary_inverse_density_beta_max=${PRIMARY_BETA[$control]}" \
        "experiment.randomize_secondary_inverse_density_beta=True" \
        "experiment.secondary_inverse_density_beta_min=${SECONDARY_BETA[$control]}" \
        "experiment.secondary_inverse_density_beta_max=${SECONDARY_BETA[$control]}" \
        "experiment.init_ckpt=${checkpoint}" \
        "experiment.resume_ckpt=" \
        "experiment.resume_full_state=False"
  ) >>"$log" 2>&1 || status=$?
  printf '%s\n' "$status" > "$LOG_DIR/${label}.exit_code"
  printf '[%s] finished %s on GPU %s with status %s\n' \
    "$(date --iso-8601=seconds)" "$label" "$gpu" "$status" >>"$log"
  return "$status"
}

run_pair() {
  local gpu="$1" first_task="$2" first_control="$3" second_task="$4" second_control="$5"
  local first_pid second_pid first_status=0 second_status=0
  validate_job "$first_task" "$first_control" || return $?
  validate_job "$second_task" "$second_control" || return $?
  wait_for_pair_capacity "$gpu" "$first_task" "$second_task"

  run_job "$gpu" "$first_task" "$first_control" & first_pid=$!
  run_job "$gpu" "$second_task" "$second_control" & second_pid=$!
  printf '%s %s/%s gpu=%s\n' "$first_pid" "$first_task" "$first_control" "$gpu" >>"$LOG_DIR/pids.txt"
  printf '%s %s/%s gpu=%s\n' "$second_pid" "$second_task" "$second_control" "$gpu" >>"$LOG_DIR/pids.txt"
  wait "$first_pid" || first_status=$?
  wait "$second_pid" || second_status=$?
  (( first_status == 0 && second_status == 0 ))
}

mkdir -p "$LOG_DIR"
: >"$LOG_DIR/pids.txt"

# Pair heavy and light tasks according to the measured one-step peak VRAM.
run_pair 0 pump beta0_pair heat beta01_endpoints & queue0=$!
run_pair 1 car beta0_pair ccore beta01_endpoints & queue1=$!
run_pair 2 pump beta01_endpoints heat beta0_pair & queue2=$!
run_pair 3 car beta01_endpoints ccore beta0_pair & queue3=$!

status0=0; status1=0; status2=0; status3=0
wait "$queue0" || status0=$?
wait "$queue1" || status1=$?
wait "$queue2" || status2=$?
wait "$queue3" || status3=$?
printf 'Pair exit codes: gpu0=%s gpu1=%s gpu2=%s gpu3=%s\n' "$status0" "$status1" "$status2" "$status3"
(( status0 == 0 && status1 == 0 && status2 == 0 && status3 == 0 ))
