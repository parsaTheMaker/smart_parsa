#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/home/parsa/smart_parsa
PYTHON=/home/parsa/miniconda3/envs/smart/bin/python
RAW=/mnt/data/parsa/c_core_magnetic_fem_v1_raw
DATA=/mnt/data/parsa/c_core_magnetic_fem_v1
REMESH=/mnt/data/parsa/c_core_magnetic_surface_vtp_remesh_v4
RESULTS="$ROOT/results/c_core_magnetic"
SMOKE="$RESULTS/pipeline_smoke"
LOG_DIR="$RESULTS/training_logs"
GPU_IDS=(0 3)
START_STAGE=${PIPELINE_START_STAGE:-1}
CONFIGS=(
  c_core_magnetic_smart
  c_core_magnetic_ab_upt
  c_core_magnetic_geo_fno
  c_core_magnetic_pointnet2_ssg
  c_core_magnetic_lno
  c_core_magnetic_mspt
  c_core_magnetic_transolverpp
  c_core_magnetic_point_transformer_v3
)

export PYTHONPATH="$ROOT/smart"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export VTK_SMP_MAX_THREADS=1 SMART_KNN_N_JOBS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$RESULTS" "$LOG_DIR"

available_gib=$(df --output=avail -B1G /mnt/data | tail -n1 | tr -d ' ')
if (( available_gib < 250 )); then
  echo "Need at least 250 GiB free on /mnt/data; found ${available_gib} GiB." >&2
  exit 1
fi

cleanup_smoke() {
  "$PYTHON" - "$SMOKE" <<'PY'
import shutil, sys
shutil.rmtree(sys.argv[1], ignore_errors=True)
PY
}
on_exit() {
  status=$?
  if (( status == 0 )); then
    cleanup_smoke
  else
    echo "Smoke artifacts preserved for diagnosis at $SMOKE" >&2
  fi
}
trap on_exit EXIT

if (( START_STAGE <= 1 )); then
  echo "[1/7] Smoke-testing native FEM generation with two concurrent cases."
  cleanup_smoke
  "$PYTHON" "$ROOT/smart/scripts/generate_c_core_magnetic_dataset.py" \
    --output-dir "$SMOKE/raw" --train-cases 1 --validation-cases 1 \
    --seed 1042 --workers 2 --gmsh-threads 1 --max-cells 1000000 --mesh-size-scale 1.4
fi

if (( START_STAGE <= 2 )); then
  echo "[2/7] Smoke-testing full-vertex preprocessing and dataset loading."
  "$PYTHON" "$ROOT/smart/scripts/preprocess_c_core_magnetic_dataset.py" \
    --source-dir "$SMOKE/raw" --output-dir "$SMOKE/preprocessed" --workers 2
  "$PYTHON" - "$SMOKE/preprocessed" <<'PY'
import sys, torch
from data.c_core_magnetic_dataset import CCoreMagneticDataset
for is_validation in (False, True):
    dataset = CCoreMagneticDataset(
        sys.argv[1], if_test=is_validation, geometry_points=16384,
        surface_points=32768, volume_points=32768,
    )
    sample = dataset[0]
    assert tuple(sample[0].shape) == (16384, 3)
    assert tuple(sample[1].shape) == tuple(sample[2].shape) == (32768, 3)
    assert tuple(sample[3].shape) == tuple(sample[4].shape) == (32768, 3)
    assert all(torch.isfinite(tensor).all() for tensor in sample)
print("dataset adapter smoke test passed")
PY
fi

if (( START_STAGE <= 3 )); then
  echo "[3/7] Smoke-testing all three remeshers at div5 and div10."
  "$PYTHON" "$ROOT/smart/scripts/remesh_surface_meshes_v2.py" \
    --dataset c_core_magnetic --source-dir "$SMOKE/raw" --output-dir "$SMOKE/remesh" \
    --results-dir "$SMOKE/remesh_results" --methods voxel,quadric,feature \
    --factors 5,10 --workers 2 --max-cases 2 --fail-fast
fi

wait_for_gpu() {
  while true; do
    for gpu in "${GPU_IDS[@]}"; do
      read -r total free < <(nvidia-smi -i "$gpu" --query-gpu=memory.total,memory.free --format=csv,noheader,nounits | tr ',' ' ')
      if awk -v f="$free" -v t="$total" 'BEGIN {exit !(f/t >= 0.40)}'; then
        echo "$gpu"
        return
      fi
    done
    sleep 30
  done
}

if (( START_STAGE <= 4 )); then
  echo "[4/7] Smoke-testing one forward/backward/evaluation batch for every architecture."
  for config in "${CONFIGS[@]}"; do
    gpu=$(wait_for_gpu)
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" "$ROOT/smart/train_c_core_magnetic.py" \
      --config-name="$config" \
      experiment.data_path="$SMOKE/preprocessed" \
      experiment.epochs=1 experiment.batch_size=1 experiment.num_workers=0 \
      experiment.num_body_points=16384 experiment.num_surface_points=32768 \
      experiment.num_volume_points=32768 \
      +experiment.max_train_batches=1 +experiment.max_eval_batches=1 \
      +experiment.save_checkpoints=false wandb.mode=disabled \
      >"$SMOKE/${config}.log" 2>&1
    grep -q "epoch: 0" "$SMOKE/${config}.log" || {
      cat "$SMOKE/${config}.log" >&2
      exit 1
    }
    echo "  passed: $config on physical GPU $gpu"
  done
fi

if [[ "${PIPELINE_STOP_AFTER_SMOKE:-0}" == "1" ]]; then
  echo "All C-core pipeline smoke tests passed; stopping before production as requested."
  exit 0
fi

if (( START_STAGE <= 5 )); then
  echo "[5/7] Generating 256 train + 32 validation FEM cases with 32 workers."
  "$PYTHON" "$ROOT/smart/scripts/generate_c_core_magnetic_dataset.py" \
    --output-dir "$RAW" --train-cases 256 --validation-cases 32 \
    --seed 42 --workers 32 --gmsh-threads 1 --max-cells 1000000 --allow-failures
fi

echo "[6/7] Preprocessing all native vertices and generating six remesh representations."
"$PYTHON" "$ROOT/smart/scripts/preprocess_c_core_magnetic_dataset.py" \
  --source-dir "$RAW" --output-dir "$DATA" --workers 32
"$PYTHON" - "$DATA" <<'PY'
import sys
from data.c_core_magnetic_dataset import CCoreMagneticDataset
# Materialize train-only normalization statistics before concurrent trainers
# can race to create the same cache files.
CCoreMagneticDataset(
    sys.argv[1], if_test=False, geometry_points=1,
    surface_points=1, volume_points=1,
)
print("training statistics cache ready")
PY
"$PYTHON" "$ROOT/smart/scripts/remesh_surface_meshes_v2.py" \
  --dataset c_core_magnetic --source-dir "$RAW" --output-dir "$REMESH" \
  --results-dir "$RESULTS/remeshing" --methods voxel,quadric,feature \
  --factors 5,10 --workers 32 --fail-fast

echo "[7/7] Scheduling eight 300-epoch base trainings on GPUs 0 and 3."
declare -A PID_GPU=()
declare -A PID_NAME=()
failures=0

gpu_process_count() {
  local gpu="$1" uuid
  uuid=$(nvidia-smi -i "$gpu" --query-gpu=uuid --format=csv,noheader)
  nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader 2>/dev/null \
    | awk -F, -v u="$uuid" '$1 == u {count++} END {print count+0}'
}

reap_finished() {
  local pid
  for pid in "${!PID_GPU[@]}"; do
    if ! kill -0 "$pid" 2>/dev/null; then
      if wait "$pid"; then
        echo "[training complete] ${PID_NAME[$pid]}"
      else
        echo "[training failed] ${PID_NAME[$pid]} (see $LOG_DIR/${PID_NAME[$pid]}.log)" >&2
        failures=$((failures + 1))
      fi
      unset 'PID_GPU[$pid]' 'PID_NAME[$pid]'
    fi
  done
}

find_training_gpu() {
  local desired gpu total free count
  # Fill empty GPUs before allowing a second process on either device.
  for desired in 0 1; do
    for gpu in "${GPU_IDS[@]}"; do
      read -r total free < <(nvidia-smi -i "$gpu" --query-gpu=memory.total,memory.free --format=csv,noheader,nounits | tr ',' ' ')
      count=$(gpu_process_count "$gpu")
      if (( count == desired )) && awk -v f="$free" -v t="$total" 'BEGIN {exit !(f/t > 0.40)}'; then
        echo "$gpu"
        return 0
      fi
    done
  done
  return 1
}

for config in "${CONFIGS[@]}"; do
  while true; do
    reap_finished
    if gpu=$(find_training_gpu); then
      echo "[training launch] $config on physical GPU $gpu"
      CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" "$ROOT/smart/train_c_core_magnetic.py" \
        --config-name="$config" \
        experiment.data_path="$DATA" experiment.epochs=300 \
        experiment.multi_gpu_strategy=single \
        wandb.entity=parsa-vatani99-technical-university-of-munich \
        >"$LOG_DIR/${config}.log" 2>&1 &
      pid=$!
      PID_GPU[$pid]="$gpu"
      PID_NAME[$pid]="$config"
      sleep 60
      if ! kill -0 "$pid" 2>/dev/null; then
        reap_finished
        (( failures == 0 )) || exit 1
      fi
      break
    fi
    sleep 30
  done
done

while ((${#PID_GPU[@]})); do
  reap_finished
  sleep 30
done
(( failures == 0 )) || exit 1
echo "C-core magnetic pipeline and all eight base trainings completed successfully."
