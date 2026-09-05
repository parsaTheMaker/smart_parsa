#!/usr/bin/env bash
# Safely queue the two Heat Exchanger base architectures on GPUs 0 and 3.
set -euo pipefail

ROOT="/home/parsa/smart_parsa"
PYTHON="/home/parsa/miniconda3/envs/smart/bin/python"
QUEUE="${ROOT}/smart/scripts/queue_two_gpu_jobs.sh"
GPU_IDS="${GPU_IDS:-0,3}"
MIN_FREE_PERCENT="${MIN_FREE_PERCENT:-80}"
POLL_SECONDS="${POLL_SECONDS:-30}"
LOG_DIR="${LOG_DIR:-${ROOT}/results/heat_exchanger_base_training_logs}"

for path in \
  "${QUEUE}" \
  "${ROOT}/smart/train_toy_heat_exchange_ab_upt.py" \
  "${ROOT}/smart/train_toy_heat_exchange_geom_deeponet.py" \
  "${ROOT}/smart/config/toy_heat_exchange_ab_upt.yaml" \
  "${ROOT}/smart/config/toy_heat_exchange_geom_deeponet.yaml" \
  "/mnt/ssdraid/parsa/toy_heat_exchange_fem_v1/preprocessed_manifest.json"; do
  [[ -f "${path}" ]] || { printf 'Required file is missing: %s\n' "${path}" >&2; exit 1; }
done

mkdir -p "${LOG_DIR}"

# Commands stay in the foreground; the queue owns their process sessions and
# will not allocate a GPU until it has the requested free-memory fraction.
AB_UPT_COMMAND="cd ${ROOT} && \
CUDA_VISIBLE_DEVICES={gpu} \
PYTHONPATH=${ROOT}/smart \
PYTHONUNBUFFERED=1 \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 SMART_KNN_N_JOBS=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
${PYTHON} smart/train_toy_heat_exchange_ab_upt.py \
experiment.epochs=300 \
wandb.project=smart_toy_heat_exchange \
wandb.entity=parsa-vatani99-technical-university-of-munich \
wandb.mode=online"

GEOM_DEEPONET_COMMAND="cd ${ROOT} && \
CUDA_VISIBLE_DEVICES={gpu} \
PYTHONPATH=${ROOT}/smart \
PYTHONUNBUFFERED=1 \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 SMART_KNN_N_JOBS=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
${PYTHON} smart/train_toy_heat_exchange_geom_deeponet.py \
experiment.epochs=300 \
wandb.project=smart_toy_heat_exchange \
wandb.entity=parsa-vatani99-technical-university-of-munich \
wandb.mode=online"

exec bash "${QUEUE}" \
  --gpu-ids "${GPU_IDS}" \
  --min-free-percent "${MIN_FREE_PERCENT}" \
  --poll-seconds "${POLL_SECONDS}" \
  --log-dir "${LOG_DIR}" \
  --label heat_exchanger_ab_upt \
  --command "${AB_UPT_COMMAND}" \
  --label heat_exchanger_geom_deeponet \
  --command "${GEOM_DEEPONET_COMMAND}" \
  "$@"
