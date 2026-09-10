#!/usr/bin/env bash
# Dynamically schedule all 12 controls with at most two training jobs per GPU.
set -uo pipefail

ROOT="${ROOT:-/mnt/data5/parsa/smart_parsa}"
PYTHON="${PYTHON:-/mnt/data5/parsa/conda_envs/smart-deal/bin/python}"
LOG_DIR="${LOG_DIR:-${ROOT}/logs/reviewer_density_distribution_controls/dynamic_12}"
POLL_SECONDS="${POLL_SECONDS:-30}"
RESERVE_MIB="${RESERVE_MIB:-4096}"
EPOCHS=150
GPU_IDS=(0 1 2 3)

declare -A PEAK_MIB=(
  [car]=19124
  [pump]=20580
  [heat]=9986
  [ccore]=17054
)
declare -A PRIMARY_BETA=(
  [beta0_pair]=0
  [beta1_pair]=1
  [beta01_endpoints]=0
)
declare -A SECONDARY_BETA=(
  [beta0_pair]=0
  [beta1_pair]=1
  [beta01_endpoints]=1
)

JOB_TASK=(
  car car car
  pump pump pump
  heat heat heat
  ccore ccore ccore
)
JOB_CONTROL=(
  beta0_pair beta1_pair beta01_endpoints
  beta0_pair beta1_pair beta01_endpoints
  beta0_pair beta1_pair beta01_endpoints
  beta0_pair beta1_pair beta01_endpoints
)
declare -a JOB_STATE JOB_PID JOB_GPU

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

job_label() {
  printf '%s_%s' "${JOB_TASK[$1]}" "${JOB_CONTROL[$1]}"
}

job_experiment_name() {
  printf '%s_%s' "$(task_dataset "${JOB_TASK[$1]}")" "${JOB_CONTROL[$1]}"
}

job_output_last() {
  local task="${JOB_TASK[$1]}" control="${JOB_CONTROL[$1]}" dataset_key
  dataset_key="$(task_dataset "$task")"
  dataset_key="${dataset_key,,}"
  control="${control//_/-}"
  compgen -G "$ROOT/checkpoints/*density-control-${control}-v1*${dataset_key}*s42_last.pt" | head -n 1
}

checkpoint_epoch() {
  "$PYTHON" - "$1" <<'PY'
import sys
import torch

checkpoint = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
print(int(checkpoint.get("epoch", -1)))
PY
}

job_running_gpu() {
  local index="$1" gpu pid command expected
  expected="experiment.name=$(job_experiment_name "$index")"
  for gpu in "${GPU_IDS[@]}"; do
    while read -r pid; do
      [[ "$pid" =~ ^[0-9]+$ && -r "/proc/$pid/cmdline" ]] || continue
      command="$(tr '\0' ' ' <"/proc/$pid/cmdline")"
      if [[ "$command" == *"$expected"* ]]; then
        printf '%s' "$gpu"
        return 0
      fi
    done < <(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null)
  done
  return 1
}

validate_inputs() {
  local index task config data checkpoint failed=0
  [[ -x "$PYTHON" ]] || { printf 'Missing Python: %s\n' "$PYTHON" >&2; return 1; }
  command -v nvidia-smi >/dev/null || { printf 'nvidia-smi is required.\n' >&2; return 1; }
  for index in "${!JOB_TASK[@]}"; do
    task="${JOB_TASK[$index]}"
    config="$(task_config "$task")"
    data="$(task_data "$task")"
    checkpoint="$(task_checkpoint "$task")"
    [[ -f "$ROOT/$(task_entrypoint "$task")" ]] || { printf 'Missing entrypoint for %s\n' "$task" >&2; failed=1; }
    [[ -f "$ROOT/smart/config/$config.yaml" ]] || { printf 'Missing config: %s\n' "$config" >&2; failed=1; }
    [[ -f "$data/.reviewer_density_controls_ready" ]] || { printf 'Dataset not verified: %s\n' "$data" >&2; failed=1; }
    [[ -f "$checkpoint" ]] || { printf 'Missing checkpoint: %s\n' "$checkpoint" >&2; failed=1; }
  done
  return "$failed"
}

gpu_free_mib() {
  nvidia-smi -i "$1" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' '
}

run_job() {
  local index="$1" gpu="$2" task="${JOB_TASK[$1]}" control="${JOB_CONTROL[$1]}"
  local entrypoint config dataset data checkpoint tag label log status=0
  entrypoint="$(task_entrypoint "$task")"
  config="$(task_config "$task")"
  dataset="$(task_dataset "$task")"
  data="$(task_data "$task")"
  checkpoint="$(task_checkpoint "$task")"
  tag="density-control-${control}-v1"
  label="$(job_label "$index")"
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
  printf '%s\n' "$status" >"$LOG_DIR/${label}.exit_code"
  printf '[%s] finished %s on GPU %s with status %s\n' \
    "$(date --iso-8601=seconds)" "$label" "$gpu" "$status" >>"$log"
  return "$status"
}

reap_owned_jobs() {
  local index status checkpoint epoch
  for index in "${!JOB_TASK[@]}"; do
    [[ "${JOB_STATE[$index]}" == owned ]] || continue
    kill -0 "${JOB_PID[$index]}" 2>/dev/null && continue
    status=0
    wait "${JOB_PID[$index]}" || status=$?
    if (( status == 0 )); then
      JOB_STATE[$index]=done
    else
      JOB_STATE[$index]=failed
    fi
    printf '[%s] reaped %s status=%s\n' "$(date --iso-8601=seconds)" "$(job_label "$index")" "$status"
  done

  for index in "${!JOB_TASK[@]}"; do
    [[ "${JOB_STATE[$index]}" == external ]] || continue
    if job_running_gpu "$index" >/dev/null; then
      continue
    fi
    checkpoint="$(job_output_last "$index")"
    epoch=-1
    [[ -n "$checkpoint" ]] && epoch="$(checkpoint_epoch "$checkpoint")"
    if (( epoch == EPOCHS - 1 )); then
      JOB_STATE[$index]=done
      printf '[%s] observed completion of %s\n' "$(date --iso-8601=seconds)" "$(job_label "$index")"
    else
      JOB_STATE[$index]=failed
      printf '[%s] external job %s stopped before epoch %s\n' \
        "$(date --iso-8601=seconds)" "$(job_label "$index")" "$((EPOCHS - 1))" >&2
    fi
  done
}

active_count_on_gpu() {
  local gpu="$1" index count=0 running_gpu
  for index in "${!JOB_TASK[@]}"; do
    case "${JOB_STATE[$index]}" in
      owned)
        [[ "${JOB_GPU[$index]}" == "$gpu" ]] && count=$((count + 1))
        ;;
      external)
        running_gpu="$(job_running_gpu "$index" || true)"
        [[ "$running_gpu" == "$gpu" ]] && count=$((count + 1))
        ;;
    esac
  done
  printf '%s' "$count"
}

choose_heaviest_single() {
  local capacity="$1" index memory best=-1 best_memory=-1
  for index in "${!JOB_TASK[@]}"; do
    [[ "${JOB_STATE[$index]}" == pending ]] || continue
    memory="${PEAK_MIB[${JOB_TASK[$index]}]}"
    if (( memory <= capacity && memory > best_memory )); then
      best="$index"
      best_memory="$memory"
    fi
  done
  printf '%s' "$best"
}

choose_heaviest_pair() {
  local capacity="$1" first second total best_first=-1 best_second=-1 best_total=-1
  for first in "${!JOB_TASK[@]}"; do
    [[ "${JOB_STATE[$first]}" == pending ]] || continue
    for second in "${!JOB_TASK[@]}"; do
      (( second > first )) || continue
      [[ "${JOB_STATE[$second]}" == pending ]] || continue
      total=$(( PEAK_MIB[${JOB_TASK[$first]}] + PEAK_MIB[${JOB_TASK[$second]}] ))
      if (( total <= capacity && total > best_total )); then
        best_first="$first"
        best_second="$second"
        best_total="$total"
      fi
    done
  done
  printf '%s %s' "$best_first" "$best_second"
}

launch_index() {
  local index="$1" gpu="$2" label
  label="$(job_label "$index")"
  run_job "$index" "$gpu" &
  JOB_PID[$index]=$!
  JOB_GPU[$index]="$gpu"
  JOB_STATE[$index]=owned
  printf '%s %s gpu=%s peak_mib=%s\n' \
    "${JOB_PID[$index]}" "$label" "$gpu" "${PEAK_MIB[${JOB_TASK[$index]}]}" >>"$LOG_DIR/pids.tsv"
  printf '[%s] allocated %s to GPU %s\n' "$(date --iso-8601=seconds)" "$label" "$gpu"
}

schedule_available_capacity() {
  local gpu free active slots capacity first second index
  while read -r free gpu; do
    active="$(active_count_on_gpu "$gpu")"
    slots=$((2 - active))
    (( slots > 0 )) || continue
    free="$(gpu_free_mib "$gpu")"
    capacity=$((free - RESERVE_MIB))
    (( capacity > 0 )) || continue

    if (( slots >= 2 )); then
      read -r first second <<<"$(choose_heaviest_pair "$capacity")"
      if (( first >= 0 && second >= 0 )); then
        launch_index "$first" "$gpu"
        launch_index "$second" "$gpu"
        continue
      fi
    fi

    index="$(choose_heaviest_single "$capacity")"
    if (( index >= 0 )); then
      launch_index "$index" "$gpu"
    fi
  done < <(
    for gpu in "${GPU_IDS[@]}"; do
      printf '%s %s\n' "$(gpu_free_mib "$gpu")" "$gpu"
    done | sort -nr
  )
}

state_counts() {
  local index pending=0 running=0 done=0 failed=0
  for index in "${!JOB_TASK[@]}"; do
    case "${JOB_STATE[$index]}" in
      pending) pending=$((pending + 1)) ;;
      owned|external) running=$((running + 1)) ;;
      done) done=$((done + 1)) ;;
      failed) failed=$((failed + 1)) ;;
    esac
  done
  printf '%s %s %s %s' "$pending" "$running" "$done" "$failed"
}

validate_inputs || exit $?
mkdir -p "$LOG_DIR"
: >"$LOG_DIR/pids.tsv"

for index in "${!JOB_TASK[@]}"; do
  if running_gpu="$(job_running_gpu "$index" || true)"; [[ -n "$running_gpu" ]]; then
    JOB_STATE[$index]=external
    JOB_GPU[$index]="$running_gpu"
    printf '[%s] adopted running job %s on GPU %s\n' \
      "$(date --iso-8601=seconds)" "$(job_label "$index")" "$running_gpu"
  elif output_last="$(job_output_last "$index")"; [[ -n "$output_last" ]]; then
    output_epoch="$(checkpoint_epoch "$output_last")"
    if (( output_epoch == EPOCHS - 1 )); then
      JOB_STATE[$index]=done
    else
      printf 'Incomplete output already exists for %s at epoch %s; refusing a silent restart.\n' \
        "$(job_label "$index")" "$output_epoch" >&2
      exit 3
    fi
  else
    JOB_STATE[$index]=pending
  fi
done

while :; do
  reap_owned_jobs
  schedule_available_capacity
  read -r pending running done failed <<<"$(state_counts)"
  printf '[%s] state pending=%s running=%s done=%s failed=%s\n' \
    "$(date --iso-8601=seconds)" "$pending" "$running" "$done" "$failed"
  (( pending == 0 && running == 0 )) && break
  sleep "$POLL_SECONDS"
done

(( failed == 0 ))
