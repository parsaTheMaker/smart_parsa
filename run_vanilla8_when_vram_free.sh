#!/usr/bin/env bash
set -Eeuo pipefail

# Launch each vanilla8 experiment independently as soon as an unreserved GPU
# has less than 10% VRAM in use. Jobs run concurrently; pending jobs do not
# wait for earlier jobs to finish.

GPUS=(0 1 2 3 4 5 6 7)
VRAM_LIMIT_PERCENT="${VRAM_LIMIT_PERCENT:-10}"
POLL_SECONDS="${POLL_SECONDS:-30}"
DRY_RUN="${DRY_RUN:-0}"
ROOT=/home/parsa/smart_parsa
PYTHON=/home/parsa/miniconda3/envs/smart/bin/python
SPLIT="$ROOT/results/drivaerml_geometry_statistical_split/geometry_domain_split.json"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi is required." >&2
  exit 1
fi
if ! command -v setsid >/dev/null 2>&1; then
  echo "ERROR: setsid is required." >&2
  exit 1
fi
if ! [[ "$VRAM_LIMIT_PERCENT" =~ ^[0-9]+$ ]] || (( VRAM_LIMIT_PERCENT < 1 || VRAM_LIMIT_PERCENT > 100 )); then
  echo "ERROR: VRAM_LIMIT_PERCENT must be an integer from 1 to 100." >&2
  exit 1
fi
if ! [[ "$POLL_SECONDS" =~ ^[0-9]+$ ]] || (( POLL_SECONDS < 1 )); then
  echo "ERROR: POLL_SECONDS must be a positive integer." >&2
  exit 1
fi
if [[ ! -f "$SPLIT" ]]; then
  echo "ERROR: split JSON not found: $SPLIT" >&2
  exit 1
fi

gpu_usage() {
  local gpu="$1" used total
  IFS=',' read -r used total < <(
    nvidia-smi -i "$gpu" --query-gpu=memory.used,memory.total \
      --format=csv,noheader,nounits | tr -d ' '
  )
  if [[ -z "${used:-}" || -z "${total:-}" ]]; then
    echo "ERROR: could not read VRAM usage for GPU $gpu." >&2
    return 1
  fi
  printf '%s %s\n' "$used" "$total"
}

print_gpu_status() {
  local gpu used total percent marker
  for gpu in "${GPUS[@]}"; do
    read -r used total < <(gpu_usage "$gpu")
    percent=$((used * 100 / total))
    marker=""
    if [[ "${RESERVED_GPU[$gpu]:-0}" == "1" ]]; then
      marker=" [reserved]"
    fi
    printf '[wait] GPU %s: %s MiB / %s MiB (%s%% used)%s\n' \
      "$gpu" "$used" "$total" "$percent" "$marker"
  done
}

find_free_gpu() {
  local gpu used total
  for gpu in "${GPUS[@]}"; do
    [[ "${RESERVED_GPU[$gpu]:-0}" == "1" ]] && continue
    read -r used total < <(gpu_usage "$gpu")
    if (( used * 100 < total * VRAM_LIMIT_PERCENT )); then
      printf '%s\n' "$gpu"
      return 0
    fi
  done
  return 1
}

commands=(
  "smart/train.py|drivaerml|smart-vanilla8-domain-cluster0|DrivAerML_SMART_VANILLA8_DOMAIN"
  "smart/train_transolverpp.py|drivaerml_transolverpp|transolverpp-vanilla8-domain-cluster0|DrivAerML_TRANSOLVERPP_VANILLA8_DOMAIN"
  "smart/train_pointnet2_ssg.py|drivaerml_pointnet2_ssg|pointnet2-ssg-vanilla8-domain-cluster0|DrivAerML_POINTNET2_SSG_VANILLA8_DOMAIN"
  "smart/train_point_gnn.py|drivaerml_point_gnn|point-gnn-vanilla8-domain-cluster0|DrivAerML_POINT_GNN_VANILLA8_DOMAIN"
  "smart/train_lno.py|drivaerml_lno|lno-vanilla8-domain-cluster0|DrivAerML_LNO_VANILLA8_DOMAIN"
  "smart/train_mspt.py|drivaerml_mspt|mspt-vanilla8-domain-cluster0|DrivAerML_MSPT_VANILLA8_DOMAIN"
  "smart/train_point_transformer_v3.py|drivaerml_point_transformer_v3|ptv3-vanilla8-domain-cluster0|DrivAerML_POINT_TRANSFORMER_V3_VANILLA8_DOMAIN"
  "smart/train_lno2.py|drivaerml_lno2|lno2-vanilla8-domain-cluster0|DrivAerML_LNO2_VANILLA8_DOMAIN"
)

PIDS=()
declare -A RESERVED_GPU=()
declare -A PID_GPU=()
declare -A PID_JOB=()
NEXT_JOB=0
COMPLETED=0
TOTAL_JOBS=${#commands[@]}

stop_all() {
  local status=$? pid
  trap - EXIT INT TERM
  for pid in "${PIDS[@]}"; do
    kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
  done
  for pid in "${PIDS[@]}"; do
    wait "$pid" 2>/dev/null || true
  done
  exit "$status"
}

trap stop_all EXIT
trap 'exit 130' INT TERM

launch_job() {
  local job_index="$1" gpu="$2"
  local script config tag name command pid
  IFS='|' read -r script config tag name <<< "${commands[$job_index]}"
  command=(
    env
    PYTHONPATH="$ROOT/smart"
    CUDA_VISIBLE_DEVICES="$gpu"
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    "$PYTHON" "$ROOT/$script"
    "--config-name=$config"
    "experiment.model_tag=$tag"
    "experiment.name=$name"
    "experiment.epochs=250"
    "experiment.geometry_epoch_seeded_sampling=True"
    "+experiment.geometry_domain_split_json=$SPLIT"
    "+experiment.geometry_domain_split_train_cluster=0"
    "+experiment.geometry_domain_split_test_cluster=1"
    "wandb.project=smart_drivaerml"
    "wandb.entity=parsa-vatani99-technical-university-of-munich"
  )
  printf '[run] GPU %s: ' "$gpu"
  printf '%q ' setsid "${command[@]}"
  printf '\n'
  cd "$ROOT"
  setsid "${command[@]}" &
  pid="$!"
  PIDS+=("$pid")
  PID_GPU["$pid"]="$gpu"
  PID_JOB["$pid"]="$job_index"
  RESERVED_GPU["$gpu"]=1
  echo "[run] started PID $pid for $tag on GPU $gpu; pending jobs keep scanning."
}

reap_finished_jobs() {
  local pid status gpu job_index script config tag name
  local active_pids=()
  for pid in "${PIDS[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      active_pids+=("$pid")
      continue
    fi
    if wait "$pid"; then status=0; else status=$?; fi
    gpu="${PID_GPU[$pid]}"
    job_index="${PID_JOB[$pid]}"
    unset "RESERVED_GPU[$gpu]" "PID_GPU[$pid]" "PID_JOB[$pid]"
    IFS='|' read -r script config tag name <<< "${commands[$job_index]}"
    if (( status != 0 )); then
      echo "[error] $tag failed with exit code $status." >&2
      return "$status"
    fi
    COMPLETED=$((COMPLETED + 1))
    echo "[run] completed $tag on GPU $gpu; that GPU is available again."
  done
  PIDS=("${active_pids[@]}")
  return 0
}

if [[ "$DRY_RUN" == "1" ]]; then
  printf '[dry-run] %s jobs, GPUs: %s, VRAM limit: %s%% used, poll: %ss\n' \
    "$TOTAL_JOBS" "${GPUS[*]}" "$VRAM_LIMIT_PERCENT" "$POLL_SECONDS"
  printf '[dry-run] split: %s\n' "$SPLIT"
  printf '[dry-run] %s\n' "${commands[@]}"
  exit 0
fi

cd "$ROOT"
while (( COMPLETED < TOTAL_JOBS )); do
  while (( NEXT_JOB < TOTAL_JOBS )); do
    if gpu="$(find_free_gpu)"; then
      launch_job "$NEXT_JOB" "$gpu"
      NEXT_JOB=$((NEXT_JOB + 1))
    else
      break
    fi
  done

  reap_finished_jobs

  if (( COMPLETED < TOTAL_JOBS )); then
    if (( NEXT_JOB < TOTAL_JOBS )); then
      print_gpu_status
      echo "[wait] rescanning in ${POLL_SECONDS}s; any GPU below ${VRAM_LIMIT_PERCENT}% used can receive the next job."
    fi
    sleep "$POLL_SECONDS"
  fi
done

trap - EXIT INT TERM
echo "[run] all vanilla8 jobs completed."
