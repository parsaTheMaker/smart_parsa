#!/usr/bin/env bash
# Sequentially resume the three C-core density controls on one servus06 GPU.
set -euo pipefail

ROOT="${ROOT:-/mnt/data5/parsa/smart_parsa}"
PYTHON="${PYTHON:-/mnt/data5/parsa/conda_envs/smart-deal/bin/python}"
DATA="${DATA:-/mnt/data5/parsa/c_core_magnetic_fem_v1}"
GPU="${GPU:-2}"
MIN_FREE_MIB="${MIN_FREE_MIB:-19000}"
POLL_SECONDS="${POLL_SECONDS:-60}"
LOG_DIR="${LOG_DIR:-${ROOT}/logs/reviewer_density_distribution_controls/ccore_resumed}"
EPOCHS=150

CONTROLS=(beta01_endpoints beta0_pair beta1_pair)

checkpoint_for() {
  local normalized="${1//_/-}"
  printf '%s/checkpoints/smart-c-core-magnetic-deal-density-control-density-control-%s-v1-ccoremagnetic-s42_last.pt' \
    "$ROOT" "$normalized"
}

checkpoint_epoch() {
  "$PYTHON" - "$1" <<'PY'
import sys
import torch

state = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
print(int(state.get("epoch", -1)))
PY
}

beta_for() {
  case "$1:$2" in
    beta0_pair:primary|beta0_pair:secondary|beta01_endpoints:primary) printf '0' ;;
    beta1_pair:primary|beta1_pair:secondary|beta01_endpoints:secondary) printf '1' ;;
  esac
}

gpu_free_mib() {
  nvidia-smi -i "$GPU" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' '
}

mkdir -p "$LOG_DIR"
[[ -x "$PYTHON" ]] || { printf 'Missing Python: %s\n' "$PYTHON" >&2; exit 1; }
[[ -f "$ROOT/smart/train_c_core_magnetic_deal.py" ]] || { printf 'Missing C-core entrypoint.\n' >&2; exit 1; }
[[ -f "$ROOT/smart/config/c_core_magnetic_density_distribution_control.yaml" ]] || { printf 'Missing C-core config.\n' >&2; exit 1; }
[[ -f "$DATA/.reviewer_density_controls_ready" ]] || { printf 'C-core dataset is not verified: %s\n' "$DATA" >&2; exit 1; }

for control in "${CONTROLS[@]}"; do
  checkpoint="$(checkpoint_for "$control")"
  [[ -f "$checkpoint" ]] || { printf 'Missing checkpoint: %s\n' "$checkpoint" >&2; exit 1; }
  epoch="$(checkpoint_epoch "$checkpoint")"
  if (( epoch >= EPOCHS - 1 )); then
    printf '[%s] skipping completed CCoreMagnetic_%s checkpoint\n' "$(date --iso-8601=seconds)" "$control"
    continue
  fi

  while (( $(gpu_free_mib) < MIN_FREE_MIB )); do
    printf '[%s] waiting for GPU %s memory: %s MiB free, need %s MiB\n' \
      "$(date --iso-8601=seconds)" "$GPU" "$(gpu_free_mib)" "$MIN_FREE_MIB"
    sleep "$POLL_SECONDS"
  done

  if pgrep -f "smart/train_c_core_magnetic_deal.py.*experiment.name=CCoreMagnetic_${control}" >/dev/null; then
    printf '[%s] refusing duplicate CCoreMagnetic_%s launch\n' "$(date --iso-8601=seconds)" "$control" >&2
    exit 1
  fi

  primary_beta="$(beta_for "$control" primary)"
  secondary_beta="$(beta_for "$control" secondary)"
  log="$LOG_DIR/ccore_${control}.gpu${GPU}.log"
  printf '[%s] resuming CCoreMagnetic_%s after checkpoint epoch %s on GPU %s\n' \
    "$(date --iso-8601=seconds)" "$control" "$epoch" "$GPU" | tee -a "$log"

  cd "$ROOT"
  env \
    CUDA_VISIBLE_DEVICES="$GPU" \
    PYTHONPATH="$ROOT/smart" \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    SMART_KNN_N_JOBS=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$PYTHON" smart/train_c_core_magnetic_deal.py \
      --config-name=c_core_magnetic_density_distribution_control \
      "experiment.name=CCoreMagnetic_${control}" \
      "experiment.model_tag=density-control-${control}-v1" \
      "experiment.data_path=$DATA" \
      "experiment.epochs=$EPOCHS" \
      experiment.random_seed=42 \
      experiment.multi_gpu_strategy=single \
      experiment.train_shared_shift_sampling_mode= \
      experiment.randomize_primary_inverse_density_beta=True \
      "experiment.primary_inverse_density_beta_min=$primary_beta" \
      "experiment.primary_inverse_density_beta_max=$primary_beta" \
      experiment.randomize_secondary_inverse_density_beta=True \
      "experiment.secondary_inverse_density_beta_min=$secondary_beta" \
      "experiment.secondary_inverse_density_beta_max=$secondary_beta" \
      experiment.init_ckpt= \
      "experiment.resume_ckpt=$checkpoint" \
      experiment.resume_full_state=True >>"$log" 2>&1

  printf '[%s] completed CCoreMagnetic_%s on GPU %s\n' \
    "$(date --iso-8601=seconds)" "$control" "$GPU" | tee -a "$log"
done
