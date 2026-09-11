#!/usr/bin/env bash
# Resume Pump beta1_pair on GPUs 0,1 once the two named Car controls finish.
set -euo pipefail

ROOT="${ROOT:-/mnt/data5/parsa/smart_parsa}"
PYTHON="${PYTHON:-/mnt/data5/parsa/conda_envs/smart-deal/bin/python}"
DATA="${DATA:-/mnt/data5/parsa/shift_pump_random1400_preprocessed}"
CHECKPOINT="${CHECKPOINT:-${ROOT}/checkpoints/smart-pump-deal-density-control-density-control-beta1-pair-v1-pump-s42_last.pt}"
LOG_DIR="${LOG_DIR:-${ROOT}/logs/reviewer_density_distribution_controls/migrated}"
RUN_LOG="${RUN_LOG:-${LOG_DIR}/pump_beta1_pair.gpu0-1.dataparallel.log}"
POLL_SECONDS="${POLL_SECONDS:-60}"
MIN_FREE_MIB="${MIN_FREE_MIB:-20000}"

CAR_BETA0_PATTERN='smart/train_satloss7.py.*experiment.name=DrivAerML_beta0_pair'
CAR_BETA1_PATTERN='smart/train_satloss7.py.*experiment.name=DrivAerML_beta1_pair'

timestamp() {
  date --iso-8601=seconds
}

car_controls_running() {
  pgrep -f "$CAR_BETA0_PATTERN" >/dev/null || pgrep -f "$CAR_BETA1_PATTERN" >/dev/null
}

gpu_free_mib() {
  nvidia-smi -i "$1" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' '
}

mkdir -p "$LOG_DIR"
[[ -x "$PYTHON" ]] || { printf 'Missing Python: %s\n' "$PYTHON" >&2; exit 1; }
[[ -f "$ROOT/smart/train_pump_satloss7.py" ]] || { printf 'Missing Pump entrypoint.\n' >&2; exit 1; }
[[ -f "$ROOT/smart/config/pump_density_distribution_control.yaml" ]] || { printf 'Missing Pump config.\n' >&2; exit 1; }
[[ -d "$DATA" ]] || { printf 'Missing Pump data: %s\n' "$DATA" >&2; exit 1; }
[[ -f "$CHECKPOINT" ]] || { printf 'Missing Pump checkpoint: %s\n' "$CHECKPOINT" >&2; exit 1; }

while car_controls_running; do
  printf '[%s] waiting for Car beta0_pair and beta1_pair to finish\n' "$(timestamp)"
  sleep "$POLL_SECONDS"
done

# GPU 1 may still host another experiment. Start only when both replicas have
# enough headroom instead of assuming that process completion implies capacity.
while true; do
  free0="$(gpu_free_mib 0)"
  free1="$(gpu_free_mib 1)"
  if (( free0 >= MIN_FREE_MIB && free1 >= MIN_FREE_MIB )); then
    break
  fi
  printf '[%s] waiting for memory: GPU0=%s MiB, GPU1=%s MiB (need %s MiB each)\n' \
    "$(timestamp)" "$free0" "$free1" "$MIN_FREE_MIB"
  sleep "$POLL_SECONDS"
done

if pgrep -f 'smart/train_pump_satloss7.py.*experiment.name=Pump_beta1_pair' >/dev/null; then
  printf '[%s] refusing duplicate launch: Pump beta1_pair is already running\n' "$(timestamp)" >&2
  exit 1
fi

checkpoint_epoch="$($PYTHON - "$CHECKPOINT" <<'PY'
import sys
import torch

state = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
print(int(state.get("epoch", -1)))
PY
)"
printf '[%s] resuming Pump beta1_pair after checkpoint epoch %s on GPUs 0,1\n' \
  "$(timestamp)" "$checkpoint_epoch"

cd "$ROOT"
exec env \
  CUDA_VISIBLE_DEVICES=0,1 \
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
    experiment.name=Pump_beta1_pair \
    experiment.model_tag=density-control-beta1_pair-v1 \
    "experiment.data_path=$DATA" \
    experiment.epochs=150 \
    experiment.batch_size=1 \
    experiment.random_seed=42 \
    experiment.multi_gpu_strategy=data_parallel \
    experiment.fuse_consistency_views=True \
    experiment.train_shared_shift_sampling_mode= \
    experiment.randomize_primary_inverse_density_beta=True \
    experiment.primary_inverse_density_beta_min=1 \
    experiment.primary_inverse_density_beta_max=1 \
    experiment.randomize_secondary_inverse_density_beta=True \
    experiment.secondary_inverse_density_beta_min=1 \
    experiment.secondary_inverse_density_beta_max=1 \
    experiment.init_ckpt= \
    "experiment.resume_ckpt=$CHECKPOINT" \
    experiment.resume_full_state=True >>"$RUN_LOG" 2>&1
