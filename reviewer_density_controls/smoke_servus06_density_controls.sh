#!/usr/bin/env bash
# Run one optimizer step and bounded evaluation per task; never save checkpoints.
set -uo pipefail

ROOT="${ROOT:-/mnt/data5/parsa/smart_parsa}"
PYTHON="${PYTHON:-/mnt/data5/parsa/conda_envs/smart-deal/bin/python}"
GPU_ID="${GPU_ID:-1}"
LOG_DIR="${LOG_DIR:-${ROOT}/logs/reviewer_density_distribution_controls/smoke}"

mkdir -p "$LOG_DIR"

run_smoke() {
  local task="$1" entrypoint="$2" config="$3" data="$4" checkpoint="$5"
  local log="$LOG_DIR/${task}.log" pid peak=0 used status=0

  printf 'SMOKE START %s\n' "$task"
  (
    cd "$ROOT" || exit 1
    env \
      CUDA_VISIBLE_DEVICES="$GPU_ID" \
      PYTHONPATH="$ROOT/smart" \
      PYTHONUNBUFFERED=1 \
      WANDB_MODE=disabled \
      WANDB_DISABLED=true \
      OMP_NUM_THREADS=1 \
      OPENBLAS_NUM_THREADS=1 \
      MKL_NUM_THREADS=1 \
      NUMEXPR_NUM_THREADS=1 \
      SMART_KNN_N_JOBS=1 \
      PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      "$PYTHON" "$entrypoint" \
        --config-name="$config" \
        "experiment.data_path=$data" \
        "experiment.epochs=1" \
        "experiment.batch_size=1" \
        "experiment.multi_gpu_strategy=single" \
        "experiment.num_workers=0" \
        "experiment.cuda_batch_prefetch=False" \
        "++experiment.max_train_batches=1" \
        "++experiment.max_eval_batches=1" \
        "++experiment.save_checkpoints=False" \
        "experiment.train_shared_shift_sampling_mode=" \
        "experiment.primary_inverse_density_beta_min=0" \
        "experiment.primary_inverse_density_beta_max=0" \
        "experiment.secondary_inverse_density_beta_min=1" \
        "experiment.secondary_inverse_density_beta_max=1" \
        "experiment.init_ckpt=$checkpoint" \
        "experiment.resume_ckpt=" \
        "experiment.resume_full_state=False" \
        "++wandb.mode=disabled"
  ) >"$log" 2>&1 &
  pid=$!

  while kill -0 "$pid" 2>/dev/null; do
    used="$(
      nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits |
        awk -F, -v target="$pid" '$1 + 0 == target {gsub(/ /, "", $2); print $2}' |
        head -n 1
    )"
    if [[ "$used" =~ ^[0-9]+$ ]] && (( used > peak )); then
      peak="$used"
    fi
    sleep 0.2
  done

  wait "$pid" || status=$?
  printf 'SMOKE END %s status=%s peak_mib=%s log=%s\n' "$task" "$status" "$peak" "$log"
  return "$status"
}

run_smoke \
  car \
  smart/train_satloss7.py \
  drivaerml_density_distribution_control \
  /mnt/data5/parsa/drivaerml_preprocessed \
  "$ROOT/checkpoints/smart-smart-drivaerml-131k16kwr-drivaerml-s42_last.pt" || exit $?

run_smoke \
  pump \
  smart/train_pump_satloss7.py \
  pump_density_distribution_control \
  /mnt/data5/parsa/shift_pump_random1400_preprocessed \
  "$ROOT/checkpoints/smart-pump-random1400-base-16k-pump-s42_last.pt" || exit $?

run_smoke \
  heat \
  smart/train_toy_heat_exchange_satloss7.py \
  toy_heat_exchange_density_distribution_control \
  /mnt/data5/parsa/toy_heat_exchange_fem_v1 \
  "$ROOT/checkpoints/smart-toy-heat-exchange-heat-exchange-base-ratio-aligned-toyheatexchange-s42_last.pt" || exit $?

run_smoke \
  ccore \
  smart/train_c_core_magnetic_deal.py \
  c_core_magnetic_density_distribution_control \
  /mnt/data5/parsa/c_core_magnetic_fem_v1 \
  "$ROOT/checkpoints/smart-c-core-magnetic-ccoremagnetic-s42_last.pt"
