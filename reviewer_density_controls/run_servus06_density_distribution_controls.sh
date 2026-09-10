#!/usr/bin/env bash
# Prepare or run the cross-system controls requested for continuous beta variation.
set -uo pipefail

MODE="${1:-plan}"
ROOT="${ROOT:-/mnt/data5/parsa/smart_parsa}"
PYTHON="${PYTHON:-/mnt/data5/parsa/conda_envs/smart-deal/bin/python}"
LOG_DIR="${LOG_DIR:-${ROOT}/logs/reviewer_density_distribution_controls}"
MIN_FREE_MIB="${MIN_FREE_MIB:-38000}"
POLL_SECONDS="${POLL_SECONDS:-60}"
EPOCHS=150
REQUIRE_DATA_READY_MARKER="${REQUIRE_DATA_READY_MARKER:-1}"
DATA_READY_MARKER=".reviewer_density_controls_ready"

CAR_DATA="${CAR_DATA:-/mnt/data5/parsa/drivaerml_preprocessed}"
PUMP_DATA="${PUMP_DATA:-/mnt/data5/parsa/shift_pump_random1400_preprocessed}"
HEAT_DATA="${HEAT_DATA:-/mnt/data5/parsa/toy_heat_exchange_fem_v1}"
CCORE_DATA="${CCORE_DATA:-/mnt/data5/parsa/c_core_magnetic_fem_v1}"

declare -A GPU_BY_CONTROL=(
  [beta0_pair]=0
  [beta1_pair]=1
  [beta01_endpoints]=2
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

TASKS=(car pump heat ccore)
CONTROLS=(beta0_pair beta1_pair beta01_endpoints)

usage() {
  printf 'Usage: %s [plan|run]\n' "$0"
  printf '  plan  Validate and print the 12-job matrix without launching (default).\n'
  printf '  run   Queue four tasks per control on physical GPUs 0, 1, and 2.\n'
}

case "$MODE" in
  plan|run) ;;
  -h|--help) usage; exit 0 ;;
  *) usage >&2; exit 2 ;;
esac

task_config() {
  case "$1" in
    car) printf '%s' drivaerml_density_distribution_control ;;
    pump) printf '%s' pump_density_distribution_control ;;
    heat) printf '%s' toy_heat_exchange_density_distribution_control ;;
    ccore) printf '%s' c_core_magnetic_density_distribution_control ;;
  esac
}

task_entrypoint() {
  case "$1" in
    car) printf '%s' smart/train_satloss7.py ;;
    pump) printf '%s' smart/train_pump_satloss7.py ;;
    heat) printf '%s' smart/train_toy_heat_exchange_satloss7.py ;;
    ccore) printf '%s' smart/train_c_core_magnetic_deal.py ;;
  esac
}

task_data() {
  case "$1" in
    car) printf '%s' "$CAR_DATA" ;;
    pump) printf '%s' "$PUMP_DATA" ;;
    heat) printf '%s' "$HEAT_DATA" ;;
    ccore) printf '%s' "$CCORE_DATA" ;;
  esac
}

task_dataset_name() {
  case "$1" in
    car) printf '%s' DrivAerML ;;
    pump) printf '%s' Pump ;;
    heat) printf '%s' ToyHeatExchange ;;
    ccore) printf '%s' CCoreMagnetic ;;
  esac
}

validate_static_inputs() {
  local failed=0 task config entrypoint data
  if [[ ! -x "$PYTHON" ]]; then
    printf 'MISSING python: %s\n' "$PYTHON" >&2
    failed=1
  fi
  for task in "${TASKS[@]}"; do
    config="$(task_config "$task")"
    entrypoint="$(task_entrypoint "$task")"
    data="$(task_data "$task")"
    [[ -f "$ROOT/$entrypoint" ]] || { printf 'MISSING entrypoint: %s\n' "$ROOT/$entrypoint" >&2; failed=1; }
    [[ -f "$ROOT/smart/config/$config.yaml" ]] || { printf 'MISSING config: %s\n' "$ROOT/smart/config/$config.yaml" >&2; failed=1; }
    if [[ -f "$data/preprocessed_manifest.json" ]] && \
       { [[ "$REQUIRE_DATA_READY_MARKER" == 0 ]] || [[ -f "$data/$DATA_READY_MARKER" ]]; }; then
      printf 'READY   %-6s data=%s\n' "$task" "$data"
    else
      printf 'MISSING %-6s data=%s\n' "$task" "$data"
      failed=1
    fi
  done
  return "$failed"
}

validate_resolved_configs() {
  ROOT="$ROOT" PYTHONPATH="$ROOT/smart" "$PYTHON" - <<'PY'
import os
from hydra import compose, initialize_config_dir

root = os.environ["ROOT"]
configs = {
    "car": ("drivaerml_density_distribution_control", 131072),
    "pump": ("pump_density_distribution_control", 16384),
    "heat": ("toy_heat_exchange_density_distribution_control", 65536),
    "ccore": ("c_core_magnetic_density_distribution_control", 32768),
}
with initialize_config_dir(config_dir=os.path.join(root, "smart", "config"), version_base="1.2"):
    for task, (name, budget) in configs.items():
        cfg = compose(config_name=name).experiment
        checks = {
            "epochs": int(cfg.epochs) == 150,
            "from_scratch": not str(cfg.init_ckpt) and not str(cfg.resume_ckpt) and not bool(cfg.resume_full_state),
            "single_gpu": str(cfg.multi_gpu_strategy) == "single",
            "primary_budget": int(cfg.primary_view_geometry_points) == budget,
            "secondary_budget": int(cfg.secondary_view_geometry_points) == budget,
            "beta_only": str(cfg.train_shared_shift_sampling_mode) == "",
            "primary_inverse_density": str(cfg.train_primary_sampling_mode) == "inverse_density_wor",
            "secondary_inverse_density": str(cfg.train_secondary_sampling_mode) == "inverse_density_wor",
            "dual_supervision": bool(cfg.fixed_sum_use_view_losses),
            "consistency": bool(cfg.use_prediction_consistency),
        }
        failed = [key for key, passed in checks.items() if not passed]
        if failed:
            raise SystemExit(f"{task}/{name} failed invariants: {', '.join(failed)}")
        print(f"VALID   {task:<6} config={name} budget={budget} epochs={cfg.epochs}")
PY
}

print_plan() {
  local control task
  printf '\n%-18s %-4s %-6s %-6s %s\n' CONTROL GPU BETA_A BETA_B TASKS
  for control in "${CONTROLS[@]}"; do
    printf '%-18s %-4s %-6s %-6s %s\n' \
      "$control" "${GPU_BY_CONTROL[$control]}" "${PRIMARY_BETA[$control]}" \
      "${SECONDARY_BETA[$control]}" "${TASKS[*]}"
  done
  printf '\nAll views use independent sampling seeds and the task-specific DeAL point budget.\n'
}

static_status=0
validate_static_inputs || static_status=$?
if [[ -x "$PYTHON" ]]; then
  validate_resolved_configs || static_status=$?
fi
print_plan

if [[ "$MODE" == plan ]]; then
  if (( static_status == 0 )); then
    printf '\nPlan validation passed. No training was launched.\n'
  else
    printf '\nPlan is staged, but missing inputs above must be supplied before launch. No training was launched.\n'
  fi
  exit "$static_status"
fi
if (( static_status != 0 )); then
  printf 'Refusing to launch because preflight validation failed.\n' >&2
  exit "$static_status"
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
  printf 'nvidia-smi is required for run mode.\n' >&2
  exit 2
fi

mkdir -p "$LOG_DIR" "$ROOT/checkpoints"

gpu_free_mib() {
  nvidia-smi -i "$1" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' '
}

wait_for_gpu() {
  local gpu="$1" free
  while :; do
    free="$(gpu_free_mib "$gpu")"
    if [[ "$free" =~ ^[0-9]+$ ]] && (( free >= MIN_FREE_MIB )); then
      return 0
    fi
    printf '[%s] GPU %s has %s MiB free; waiting for %s MiB.\n' \
      "$(date --iso-8601=seconds)" "$gpu" "${free:-unknown}" "$MIN_FREE_MIB"
    sleep "$POLL_SECONDS"
  done
}

run_job() {
  local control="$1" task="$2" gpu="$3"
  local beta_a="${PRIMARY_BETA[$control]}" beta_b="${SECONDARY_BETA[$control]}"
  local config entrypoint data dataset tag label log status_file
  config="$(task_config "$task")"
  entrypoint="$(task_entrypoint "$task")"
  data="$(task_data "$task")"
  dataset="$(task_dataset_name "$task")"
  tag="density-control-${control}-v1"
  label="${task}_${control}"
  log="$LOG_DIR/${label}.gpu${gpu}.log"
  status_file="$LOG_DIR/${label}.exit_code"

  if compgen -G "$ROOT/checkpoints/*${tag}*${dataset}*s42*.pt" >/dev/null; then
    printf 'Refusing to overwrite an existing checkpoint for %s.\n' "$label" >&2
    return 3
  fi

  wait_for_gpu "$gpu"
  printf '[%s] starting %s on physical GPU %s\n' "$(date --iso-8601=seconds)" "$label" "$gpu" | tee "$log"
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
        "experiment.primary_inverse_density_beta_min=${beta_a}" \
        "experiment.primary_inverse_density_beta_max=${beta_a}" \
        "experiment.randomize_secondary_inverse_density_beta=True" \
        "experiment.secondary_inverse_density_beta_min=${beta_b}" \
        "experiment.secondary_inverse_density_beta_max=${beta_b}" \
        "experiment.init_ckpt=" \
        "experiment.resume_ckpt=" \
        "experiment.resume_full_state=False"
  ) >>"$log" 2>&1
  local status=$?
  printf '%s\n' "$status" > "$status_file"
  printf '[%s] finished %s on GPU %s with status %s\n' \
    "$(date --iso-8601=seconds)" "$label" "$gpu" "$status" | tee -a "$log"
  return "$status"
}

run_control_queue() {
  local control="$1" gpu="${GPU_BY_CONTROL[$1]}" task
  for task in "${TASKS[@]}"; do
    run_job "$control" "$task" "$gpu" || return $?
  done
}

printf '\nLaunching three isolated queues. Each queue runs its four systems sequentially.\n'
run_control_queue beta0_pair & pid0=$!
run_control_queue beta1_pair & pid1=$!
run_control_queue beta01_endpoints & pid2=$!

status0=0; status1=0; status2=0
wait "$pid0" || status0=$?
wait "$pid1" || status1=$?
wait "$pid2" || status2=$?
printf 'Queue exit codes: beta0_pair=%s beta1_pair=%s beta01_endpoints=%s\n' "$status0" "$status1" "$status2"
(( status0 == 0 && status1 == 0 && status2 == 0 ))
