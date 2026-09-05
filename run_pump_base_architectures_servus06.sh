#!/usr/bin/env bash
# Launch the eight Pump base-surrogate trainings as four measured pairs on servus06.
# Each process sees exactly one physical GPU; the script refuses to launch a pair on
# a GPU that is already too occupied for safe coexistence.
set -euo pipefail

ROOT="${ROOT:-/mnt/data5/parsa/smart_parsa}"
ENV_ROOT="${ENV_ROOT:-/mnt/data5/parsa/conda_envs/smart-deal}"
DATA_ROOT="${DATA_ROOT:-/mnt/data5/parsa/shift_pump_random1400_preprocessed}"
PYTHON="${PYTHON:-${ENV_ROOT}/bin/python}"
LOG_DIR="${LOG_DIR:-${ROOT}/logs/pump_base_architectures_servus06}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
MIN_FREE_MIB="${MIN_FREE_MIB:-40000}"
EPOCHS="${EPOCHS:-300}"
GPU_WAIT_INTERVAL_SECONDS="${GPU_WAIT_INTERVAL_SECONDS:-60}"
MAX_GPU_WAIT_SECONDS="${MAX_GPU_WAIT_SECONDS:-0}"

if [[ ! -x "${PYTHON}" ]]; then
  echo "Missing packed smart environment: ${PYTHON}" >&2
  exit 1
fi
if [[ ! -d "${DATA_ROOT}" ]]; then
  echo "Missing Pump preprocessing: ${DATA_ROOT}" >&2
  exit 1
fi

IFS=',' read -r -a GPUS <<< "${GPU_IDS}"
if [[ "${#GPUS[@]}" -ne 4 ]]; then
  echo "GPU_IDS must contain exactly four physical GPU IDs, got: ${GPU_IDS}" >&2
  exit 1
fi

mkdir -p "${LOG_DIR}" "${ROOT}/checkpoints"
cd "${ROOT}"

common_env=(
  "PYTHONPATH=${ROOT}/smart"
  "PYTHONUNBUFFERED=1"
  "WANDB_MODE=disabled"
  "WANDB_DISABLED=true"
  "OMP_NUM_THREADS=1"
  "OPENBLAS_NUM_THREADS=1"
  "MKL_NUM_THREADS=1"
  "NUMEXPR_NUM_THREADS=1"
  "SMART_KNN_N_JOBS=1"
  "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
)

launch() {
  local gpu="$1"
  local label="$2"
  local entrypoint="$3"
  local batch_size="$4"
  local model_tag="servus06-base-v1"
  local log_file="${LOG_DIR}/${label}.log"

  echo "Launching ${label} on physical GPU ${gpu} (batch=${batch_size}); log: ${log_file}"
  nohup env "${common_env[@]}" CUDA_VISIBLE_DEVICES="${gpu}" \
    "${PYTHON}" "${ROOT}/smart/${entrypoint}" \
      "experiment.name=SHIFT_PUMP_${label}_BASE" \
      "experiment.model_tag=${model_tag}" \
      "experiment.data_path=${DATA_ROOT}" \
      "experiment.epochs=${EPOCHS}" \
      "experiment.batch_size=${batch_size}" \
      "experiment.multi_gpu_strategy=single" \
      "experiment.num_workers=2" \
      "experiment.prefetch_factor=1" \
      "experiment.cuda_batch_prefetch=False" \
      "experiment.init_ckpt=" \
      "experiment.resume_ckpt=" \
      "experiment.resume_full_state=False" \
      "++wandb.mode=disabled" \
    >"${log_file}" 2>&1 &
  echo "$! ${label} gpu=${gpu} log=${log_file}" >> "${LOG_DIR}/pids.txt"
}

wait_for_gpu() {
  local gpu="$1"
  local waited=0
  local free_mib
  while true; do
    free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "${gpu}" | tr -d ' ')"
    if (( free_mib >= MIN_FREE_MIB )); then
      echo "GPU ${gpu} is ready for a pair (${free_mib} MiB free)."
      return 0
    fi
    if (( MAX_GPU_WAIT_SECONDS > 0 && waited >= MAX_GPU_WAIT_SECONDS )); then
      echo "GPU ${gpu} remained below ${MIN_FREE_MIB} MiB after ${waited}s; its pair was not launched." >&2
      return 1
    fi
    echo "GPU ${gpu} has ${free_mib} MiB free; waiting ${GPU_WAIT_INTERVAL_SECONDS}s before launching its pair."
    sleep "${GPU_WAIT_INTERVAL_SECONDS}"
    waited=$((waited + GPU_WAIT_INTERVAL_SECONDS))
  done
}

launch_pair() {
  local gpu="$1"
  local first_label="$2"
  local first_entrypoint="$3"
  local first_batch="$4"
  local second_label="$5"
  local second_entrypoint="$6"
  local second_batch="$7"
  wait_for_gpu "${gpu}"
  launch "${gpu}" "${first_label}" "${first_entrypoint}" "${first_batch}"
  launch "${gpu}" "${second_label}" "${second_entrypoint}" "${second_batch}"
}

: > "${LOG_DIR}/pids.txt"
launch_pair "${GPUS[0]}" "SMART" "train_pump.py" 2 "POINTNET2_SSG" "train_pump_pointnet2_ssg.py" 4
launch_pair "${GPUS[1]}" "TRANSOLVERPP" "train_pump_transolverpp.py" 2 "LNO" "train_pump_lno.py" 4
launch_pair "${GPUS[2]}" "MSPT" "train_pump_mspt.py" 2 "POINT_TRANSFORMER_V3" "train_pump_point_transformer_v3.py" 4
launch_pair "${GPUS[3]}" "AB_UPT" "train_pump_ab_upt.py" 4 "GAOT" "train_pump_gaot.py" 4

echo
echo "All eight jobs launched. Follow logs with:"
echo "  tail -f ${LOG_DIR}/*.log"
echo "PIDs are recorded in: ${LOG_DIR}/pids.txt"
