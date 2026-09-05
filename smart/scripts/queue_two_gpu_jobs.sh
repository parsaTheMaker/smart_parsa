#!/usr/bin/env bash
# Queue two foreground training commands on the first eligible GPU in a small pool.
# Commands must not detach themselves with nohup or '&'; this launcher owns and
# monitors their process sessions.
set -uo pipefail

GPU_IDS="0,3"
MIN_FREE_PERCENT="80"
POLL_SECONDS="30"
LOG_DIR="$(pwd)/logs/gpu_queue"
DRY_RUN=0
WAIT_FOR_COMPLETION=1
declare -a LABELS=()
declare -a COMMANDS=()

usage() {
  cat <<'EOF'
Usage:
  queue_two_gpu_jobs.sh --label NAME --command 'CUDA_VISIBLE_DEVICES={gpu} python train.py ...' \
                        --label NAME --command 'CUDA_VISIBLE_DEVICES={gpu} python train.py ...' [options]

Options:
  --gpu-ids 0,3            Physical GPU IDs to monitor (default: 0,3)
  --min-free-percent 80    Require strictly more free VRAM than this percentage
  --poll-seconds 30        Polling interval while jobs remain pending
  --log-dir PATH           Per-job stdout/stderr destination
  --no-wait                Exit after both jobs are launched (not recommended)
  --dry-run                Validate scheduling without starting commands

Use {gpu} anywhere in a command; it is replaced with the allocated physical
GPU ID. Commands are launched in isolated sessions and must stay in the
foreground so their session can be monitored safely.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --label)
      LABELS+=("${2:?--label requires a value}")
      shift 2
      ;;
    --command)
      COMMANDS+=("${2:?--command requires a value}")
      shift 2
      ;;
    --gpu-ids)
      GPU_IDS="${2:?--gpu-ids requires a value}"
      shift 2
      ;;
    --min-free-percent)
      MIN_FREE_PERCENT="${2:?--min-free-percent requires a value}"
      shift 2
      ;;
    --poll-seconds)
      POLL_SECONDS="${2:?--poll-seconds requires a value}"
      shift 2
      ;;
    --log-dir)
      LOG_DIR="${2:?--log-dir requires a value}"
      shift 2
      ;;
    --no-wait)
      WAIT_FOR_COMPLETION=0
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      printf 'Unknown option: %s\n' "$1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ ${#LABELS[@]} -ne 2 || ${#COMMANDS[@]} -ne 2 ]]; then
  printf 'Exactly two --label/--command pairs are required.\n' >&2
  usage >&2
  exit 2
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
  printf 'nvidia-smi is required.\n' >&2
  exit 2
fi
if ! [[ "$MIN_FREE_PERCENT" =~ ^([0-9]+([.][0-9]+)?|[.][0-9]+)$ ]] || ! awk -v value="$MIN_FREE_PERCENT" 'BEGIN { exit !(value >= 0 && value < 100) }'; then
  printf '--min-free-percent must be in [0,100).\n' >&2
  exit 2
fi
if ! [[ "$POLL_SECONDS" =~ ^[1-9][0-9]*$ ]]; then
  printf '--poll-seconds must be a positive integer.\n' >&2
  exit 2
fi

IFS=',' read -r -a GPUS <<< "$GPU_IDS"
if [[ ${#GPUS[@]} -eq 0 ]]; then
  printf 'At least one GPU ID is required.\n' >&2
  exit 2
fi
mkdir -p "$LOG_DIR"

gpu_is_free_enough() {
  local gpu="$1" total free
  read -r total free < <(nvidia-smi -i "$gpu" --query-gpu=memory.total,memory.free --format=csv,noheader,nounits | tr ',' ' ')
  [[ -n "$total" && -n "$free" ]] || return 1
  awk -v free="$free" -v total="$total" -v minimum="$MIN_FREE_PERCENT" 'BEGIN { exit !(100.0 * free / total > minimum) }'
}

gpu_memory_status() {
  local gpu="$1" total free
  read -r total free < <(nvidia-smi -i "$gpu" --query-gpu=memory.total,memory.free --format=csv,noheader,nounits | tr ',' ' ')
  awk -v free="$free" -v total="$total" 'BEGIN { printf "%.1f%% free", 100.0 * free / total }'
}

declare -A GPU_OWNER=()
declare -A PID_BY_JOB=()
declare -A GPU_BY_JOB=()
declare -A STARTED=()
declare -A FINISHED=()

reap_finished_jobs() {
  local index pid status gpu
  for index in 0 1; do
    [[ -v STARTED[$index] && ! -v FINISHED[$index] ]] || continue
    pid="${PID_BY_JOB[$index]}"
    # Each command runs under setsid, so this checks the entire owned session.
    if kill -0 -- "-$pid" 2>/dev/null; then
      continue
    fi
    status=0
    wait "$pid" || status=$?
    FINISHED[$index]=1
    gpu="${GPU_BY_JOB[$index]}"
    unset 'GPU_OWNER[$gpu]'
    printf '[%s] %s finished on GPU %s (exit=%s).\n' "$(date '+%F %T')" "${LABELS[$index]}" "$gpu" "$status"
  done
}

launch_job() {
  local index="$1" gpu="$2" label command log_file pid
  label="${LABELS[$index]}"
  command="${COMMANDS[$index]//\{gpu\}/$gpu}"
  log_file="$LOG_DIR/${label}.gpu${gpu}.log"
  printf '[%s] allocating %s to GPU %s (%s); log=%s\n' "$(date '+%F %T')" "$label" "$gpu" "$(gpu_memory_status "$gpu")" "$log_file"
  STARTED[$index]=1
  GPU_BY_JOB[$index]="$gpu"
  GPU_OWNER[$gpu]="$index"
  if [[ "$DRY_RUN" -eq 1 ]]; then
    FINISHED[$index]=1
    return
  fi
  setsid bash -lc "$command" >"$log_file" 2>&1 < /dev/null &
  pid=$!
  PID_BY_JOB[$index]="$pid"
  printf '%s %s gpu=%s command=%q\n' "$pid" "$label" "$gpu" "$command" >> "$LOG_DIR/pids.txt"
}

pending_jobs() {
  local index count=0
  for index in 0 1; do
    [[ -v STARTED[$index] ]] || count=$((count + 1))
  done
  printf '%s' "$count"
}

running_jobs() {
  local index count=0
  for index in 0 1; do
    [[ -v STARTED[$index] && ! -v FINISHED[$index] ]] && count=$((count + 1))
  done
  printf '%s' "$count"
}

printf 'Queueing %s and %s on GPUs [%s]; require >%s%% free VRAM.\n' "${LABELS[0]}" "${LABELS[1]}" "$GPU_IDS" "$MIN_FREE_PERCENT"
while :; do
  reap_finished_jobs
  for index in 0 1; do
    [[ -v STARTED[$index] ]] && continue
    for gpu in "${GPUS[@]}"; do
      [[ -v GPU_OWNER[$gpu] ]] && continue
      if gpu_is_free_enough "$gpu"; then
        launch_job "$index" "$gpu"
        break
      fi
    done
  done

  pending="$(pending_jobs)"
  running="$(running_jobs)"
  if [[ "$pending" -eq 0 && ( "$WAIT_FOR_COMPLETION" -eq 0 || "$running" -eq 0 ) ]]; then
    break
  fi
  if [[ "$pending" -gt 0 ]]; then
    printf '[%s] pending=%s; GPU status:' "$(date '+%F %T')" "$pending"
    for gpu in "${GPUS[@]}"; do
      printf ' gpu%s=%s' "$gpu" "$(gpu_memory_status "$gpu")"
    done
    printf '\n'
  fi
  sleep "$POLL_SECONDS"
done

printf 'Queue finished: both jobs were %s.\n' "$([[ "$WAIT_FOR_COMPLETION" -eq 1 ]] && printf 'completed' || printf 'launched')"
