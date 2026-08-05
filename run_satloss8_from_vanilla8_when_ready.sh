#!/usr/bin/env bash
set -Eeuo pipefail

# Launch SATLOSS8 jobs independently once their matching vanilla8 checkpoint
# has reached the final configured vanilla8 epoch and a GPU has enough free VRAM.

ROOT=/home/parsa/smart_parsa
PYTHON=/home/parsa/miniconda3/envs/smart/bin/python
SPLIT="$ROOT/results/drivaerml_geometry_statistical_split/geometry_domain_split.json"
GPUS=(0 1 2 3 4 5 6 7)
VRAM_LIMIT_PERCENT="${VRAM_LIMIT_PERCENT:-10}"
POLL_SECONDS="${POLL_SECONDS:-30}"
VANILLA8_LAUNCHER="$ROOT/run_vanilla8_when_vram_free.sh"
if [[ -z "${VANILLA8_TOTAL_EPOCHS+x}" ]]; then
  # Read the effective experiment.epochs override used by the vanilla8 launcher.
  VANILLA8_TOTAL_EPOCHS="$(sed -n 's/.*experiment\.epochs=\([0-9][0-9]*\).*/\1/p' "$VANILLA8_LAUNCHER" | head -n 1)"
fi
DRY_RUN="${DRY_RUN:-0}"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi is required." >&2
  exit 1
fi
if ! command -v setsid >/dev/null 2>&1; then
  echo "ERROR: setsid is required." >&2
  exit 1
fi
if [[ ! -f "$SPLIT" ]]; then
  echo "ERROR: split JSON not found: $SPLIT" >&2
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
if ! [[ "$VANILLA8_TOTAL_EPOCHS" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: could not resolve a positive vanilla8 epoch count from $VANILLA8_LAUNCHER; set VANILLA8_TOTAL_EPOCHS explicitly." >&2
  exit 1
fi
# script|config|tag|name|vanilla8-last-checkpoint|vanilla8-config
commands=(
  "smart/train_satloss8.py|drivaerml_satloss8|smart-satloss8-domain-cluster0-from-vanilla8-100ep|DrivAerML_SMART_SATLOSS8_FROM_VANILLA8_100EP|smart-smart-vanilla8-domain-cluster0-drivaerml-s42_last.pt|drivaerml"
  "smart/train_transolverpp_satloss8.py|drivaerml_transolverpp_satloss8|transolverpp-satloss8-domain-cluster0-from-vanilla8-100ep|DrivAerML_TRANSOLVERPP_SATLOSS8_FROM_VANILLA8_100EP|transolverpp-transolverpp-vanilla8-domain-cluster0-drivaerml-s42_last.pt|drivaerml_transolverpp"
  "smart/train_pointnet2_ssg_satloss8.py|drivaerml_pointnet2_ssg_satloss8|pointnet2-ssg-satloss8-domain-cluster0-from-vanilla8-100ep|DrivAerML_POINTNET2_SSG_SATLOSS8_FROM_VANILLA8_100EP|pointnet2-ssg-pointnet2-ssg-vanilla8-domain-cluster0-drivaerml-s42_last.pt|drivaerml_pointnet2_ssg"
  "smart/train_point_gnn_satloss8.py|drivaerml_point_gnn_satloss8|point-gnn-satloss8-domain-cluster0-from-vanilla8-100ep|DrivAerML_POINT_GNN_SATLOSS8_FROM_VANILLA8_100EP|point-gnn-point-gnn-vanilla8-domain-cluster0-drivaerml-s42_last.pt|drivaerml_point_gnn"
  "smart/train_lno_satloss8.py|drivaerml_lno_satloss8|lno-satloss8-domain-cluster0-from-vanilla8-100ep|DrivAerML_LNO_SATLOSS8_FROM_VANILLA8_100EP|lno-lno-vanilla8-domain-cluster0-drivaerml-s42_last.pt|drivaerml_lno"
  "smart/train_mspt_satloss8.py|drivaerml_mspt_satloss8|mspt-satloss8-domain-cluster0-from-vanilla8-100ep|DrivAerML_MSPT_SATLOSS8_FROM_VANILLA8_100EP|mspt-mspt-vanilla8-domain-cluster0-drivaerml-s42_last.pt|drivaerml_mspt"
  "smart/train_point_transformer_v3_satloss8.py|drivaerml_point_transformer_v3_satloss8|ptv3-satloss8-domain-cluster0-from-vanilla8-100ep|DrivAerML_POINT_TRANSFORMER_V3_SATLOSS8_FROM_VANILLA8_100EP|point-transformer-v3-ptv3-vanilla8-domain-cluster0-drivaerml-s42_last.pt|drivaerml_point_transformer_v3"
  "smart/train_lno2_satloss8.py|drivaerml_lno2_satloss8|lno2-satloss8-domain-cluster0-from-vanilla8-100ep|DrivAerML_LNO2_SATLOSS8_FROM_VANILLA8_100EP|lno2-lno2-vanilla8-domain-cluster0-drivaerml-s42_last.pt|drivaerml_lno2"
)

declare -A RESERVED_GPU=()
declare -A PID_GPU=()
declare -A PID_JOB=()
declare -A STARTED_JOB=()
PIDS=()
COMPLETED=0
TOTAL_JOBS=${#commands[@]}

gpu_usage() {
  local gpu="$1" used total
  IFS=',' read -r used total < <(
    nvidia-smi -i "$gpu" --query-gpu=memory.used,memory.total \
      --format=csv,noheader,nounits | tr -d ' '
  )
  [[ -n "${used:-}" && -n "${total:-}" ]] || return 1
  printf '%s %s\n' "$used" "$total"
}

print_gpu_status() {
  local gpu used total percent marker
  for gpu in "${GPUS[@]}"; do
    read -r used total < <(gpu_usage "$gpu")
    percent=$((used * 100 / total))
    marker=""
    [[ "${RESERVED_GPU[$gpu]:-0}" == "1" ]] && marker=" [reserved]"
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

vanilla_epoch() {
  local checkpoint="$1"
  [[ -f "$checkpoint" ]] || return 1
  "$PYTHON" - "$checkpoint" <<'PY'
import sys
import torch

checkpoint = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
epoch = checkpoint.get("epoch", -1)
try:
    print(int(epoch))
except (TypeError, ValueError):
    print(-1)
PY
}

vanilla_ready() {
  local checkpoint="$1" config_name="$2" epoch required_epoch
  checkpoint="$ROOT/checkpoints/$checkpoint"
  required_epoch=$((VANILLA8_TOTAL_EPOCHS - 1))
  epoch="$(vanilla_epoch "$checkpoint" 2>/dev/null || printf '%s' -1)"
  (( epoch >= required_epoch ))
}

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
  local script config tag name vanilla_checkpoint vanilla_config init_checkpoint pid
  local -a command
  IFS='|' read -r script config tag name vanilla_checkpoint vanilla_config <<< "${commands[$job_index]}"
  init_checkpoint="$ROOT/checkpoints/$vanilla_checkpoint"
  command=(
    env
    PYTHONPATH="$ROOT/smart"
    CUDA_VISIBLE_DEVICES="$gpu"
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    "$PYTHON" "$ROOT/$script"
    "--config-name=$config"
    "experiment.model_tag=$tag"
    "experiment.name=$name"
    "experiment.epochs=100"
    "experiment.init_ckpt=$init_checkpoint"
    "experiment.resume_ckpt="
    "experiment.resume_full_state=False"
    "experiment.geometry_domain_split_json=$SPLIT"
    "experiment.geometry_domain_split_train_cluster=0"
    "experiment.geometry_domain_split_test_cluster=1"
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
  echo "[run] started PID $pid for $tag on GPU $gpu"
}

reap_finished_jobs() {
  local pid status gpu job_index script config tag name vanilla_checkpoint vanilla_config
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
    IFS='|' read -r script config tag name vanilla_checkpoint vanilla_config <<< "${commands[$job_index]}"
    if (( status != 0 )); then
      echo "[error] $tag failed with exit code $status." >&2
      return "$status"
    fi
    COMPLETED=$((COMPLETED + 1))
    echo "[run] completed $tag on GPU $gpu"
  done
  PIDS=("${active_pids[@]}")
}

if [[ "$DRY_RUN" == "1" ]]; then
  printf '[dry-run] %s SATLOSS8 jobs, GPUs: %s, VRAM limit: %s%% used, poll: %ss\n' \
    "$TOTAL_JOBS" "${GPUS[*]}" "$VRAM_LIMIT_PERCENT" "$POLL_SECONDS"
  printf '[dry-run] require each vanilla8 last checkpoint epoch >= %s (vanilla8 total epochs=%s)\n' \
    "$((VANILLA8_TOTAL_EPOCHS - 1))" "$VANILLA8_TOTAL_EPOCHS"
  printf '[dry-run] %s\n' "${commands[@]}"
  exit 0
fi

cd "$ROOT"
while (( COMPLETED < TOTAL_JOBS )); do
  for (( job_index = 0; job_index < TOTAL_JOBS; job_index++ )); do
    [[ "${STARTED_JOB[$job_index]:-0}" == "1" ]] && continue
    IFS='|' read -r _script _config _tag _name vanilla_checkpoint vanilla_config <<< "${commands[$job_index]}"
    required_epoch=$((VANILLA8_TOTAL_EPOCHS - 1))
    if ! vanilla_ready "$vanilla_checkpoint" "$vanilla_config"; then
      echo "[wait] ${_tag}: waiting for $vanilla_checkpoint to reach epoch >= $required_epoch (config: $vanilla_config)"
      continue
    fi
    if gpu="$(find_free_gpu)"; then
      launch_job "$job_index" "$gpu"
      STARTED_JOB["$job_index"]=1
    else
      break
    fi
  done

  reap_finished_jobs

  if (( COMPLETED < TOTAL_JOBS )); then
    print_gpu_status
    echo "[wait] rescanning in ${POLL_SECONDS}s; jobs launch independently when both prerequisites are ready."
    sleep "$POLL_SECONDS"
  fi
done

trap - EXIT INT TERM
echo '[run] all SATLOSS8-from-vanilla8 jobs completed.'
