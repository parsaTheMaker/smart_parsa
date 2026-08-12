#!/usr/bin/env bash
set -Eeuo pipefail

# Generate two otherwise identical range-ablation comparisons. The only
# difference is whether RANGE100/200/300 came from scratch or from SMART.

ROOT=/home/parsa/smart_parsa
PYTHON=/home/parsa/miniconda3/envs/smart/bin/python
SCRIPT="$ROOT/smart/scripts/compare_drivaerml_satloss7_range_ablation.py"
export PYTHONPATH="$ROOT/smart"
export CUDA_VISIBLE_DEVICES=3,4,6,7
RUN_MODE="${RUN_MODE:-both}"

if [[ "$RUN_MODE" != "both" && "$RUN_MODE" != "from_scratch" && "$RUN_MODE" != "from_smart" ]]; then
  echo "ERROR: RUN_MODE must be both, from_scratch, or from_smart." >&2
  exit 1
fi

COMMON_ARGS=(
  --experiment-preset range_ablation_vtp
  --num-runs 10
  --run-ids 139,287,275,288,186,475,45,250,231,486
  --seed 42
  --beta-levels 0,0.25,0.5,0.75,1
  --sine-levels 0,0.25,0.5,0.75,1
  --active-shifts beta,sine_y,sine_x
  --active-geometry-sources angle,isotropic,voxel
  --geometry-decimation-factors 5,10
  --angle-decimated-vtp-dir /mnt/ssdraid/parsa/drivaerml_surface_vtp_decimated
  --isotropic-decimated-vtp-dir /mnt/ssdraid/parsa/drivaerml_surface_vtp_isotropic_gpu
  --voxel-decimated-vtp-dir /mnt/ssdraid/parsa/drivaerml_surface_vtp_voxel_quadric_clustered
  --views-per-mode 2
  --view-batch-size 2
  --model-repeats 1
  --surface-query-points 65536
  --volume-query-points 65536
  --batched-query-subregion-size 65536
  --density-estimator kde
  --density-knn-k 16
  --plot-scales linear,log
  --font-scale 1.2
  --y-pad-fraction 0.10
  --no-std
  --exclude-range500
  --devices cuda:0,cuda:1,cuda:2,cuda:3
  --smart-checkpoint "$ROOT/checkpoints/smart-smart-drivaerml-131k16kwr-drivaerml-s42_best.pt"
  --range025-checkpoint "$ROOT/checkpoints/smart-satloss7-range025-smart-satloss7-range025-extension100-drivaerml-s42_best.pt"
  --range050-checkpoint "$ROOT/checkpoints/smart-satloss7-range050-smart-satloss7-range050-extension100-drivaerml-s42_best.pt"
  --range075-checkpoint "$ROOT/checkpoints/smart-satloss7-range075-smart-satloss7-range075-extension100-drivaerml-s42_best.pt"
  --satloss7-config drivaerml_satloss7_range100
  --range200-config drivaerml_satloss7_range200
  --range300-config drivaerml_satloss7_range300
)

run_comparison() {
  local checkpoint="$1"
  local range200_checkpoint="$2"
  local range300_checkpoint="$3"
  local output_dir="$4"
  echo "[run] output: $output_dir"
  "$PYTHON" "$SCRIPT" \
    "${COMMON_ARGS[@]}" \
    --satloss7-checkpoint "$checkpoint" \
    --range200-checkpoint "$range200_checkpoint" \
    --range300-checkpoint "$range300_checkpoint" \
    --output-dir "$output_dir"
}

cd "$ROOT"

if [[ "$RUN_MODE" == "both" || "$RUN_MODE" == "from_scratch" ]]; then
  run_comparison \
    "$ROOT/checkpoints/smart-satloss7-range100-smart-satloss7-range100-drivaerml-s42_best.pt" \
    "$ROOT/checkpoints/smart-satloss7-range200-smart-satloss7-range200-drivaerml-s42_best.pt" \
    "$ROOT/checkpoints/smart-satloss7-range300-smart-satloss7-range300-drivaerml-s42_best.pt" \
    "$ROOT/results/drivaerml_smart_satloss7_range_000till300_from_scratch"
fi

if [[ "$RUN_MODE" == "both" || "$RUN_MODE" == "from_smart" ]]; then
  run_comparison \
    "$ROOT/checkpoints/smart-satloss7-range100-smart-satloss7-range100-from-smart-150ep-drivaerml-s42_best.pt" \
    "$ROOT/checkpoints/smart-satloss7-range200-smart-satloss7-range200-from-smart-150ep-drivaerml-s42_best.pt" \
    "$ROOT/checkpoints/smart-satloss7-range300-smart-satloss7-range300-from-smart-150ep-drivaerml-s42_best.pt" \
    "$ROOT/results/drivaerml_smart_satloss7_range_000till300_from_smart"
fi
