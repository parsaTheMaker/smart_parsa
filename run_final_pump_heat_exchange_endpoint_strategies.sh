#!/usr/bin/env bash
# Reproduce matched SMART/DeAL/historical-strategy endpoint studies.
set -euo pipefail

ROOT=/home/parsa/smart_parsa
PYTHON=/home/parsa/miniconda3/envs/smart/bin/python
FINAL="$ROOT/results/final"
export PYTHONPATH="$ROOT/smart"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PUMP_DATA=/mnt/data/parsa/shift_pump_random1400_preprocessed
PUMP_RAW=/mnt/data/parsa/shift_pump_raw_random1400
PUMP_REMESH=/mnt/data/parsa/shift_pump_random1400_surface_vtp_remesh_v4
PUMP_REMESH_RESULTS="$FINAL/shift_pump_random1400_remeshing_full1400"
HEAT_DATA=/mnt/ssdraid/parsa/toy_heat_exchange_fem_v1
HEAT_REMESH=/mnt/ssdraid/parsa/toy_heat_exchange_surface_vtp_remesh_v4

case "${1:-}" in
  pump_remesh)
    "$PYTHON" "$ROOT/smart/scripts/remesh_surface_meshes_v2.py" \
      --dataset pump \
      --source-dir "$PUMP_RAW" \
      --output-dir "$PUMP_REMESH" \
      --results-dir "$PUMP_REMESH_RESULTS" \
      --pump-count 0 --factors 5,10 \
      --methods voxel,quadric,feature --workers 16
    ;;

  pump)
    CUDA_VISIBLE_DEVICES=0,1,2,3,4 "$PYTHON" "$ROOT/smart/scripts/compare_shift_endpoint_strategies.py" \
      --dataset pump --data-root "$PUMP_DATA" \
      --study-summary "$PUMP_REMESH/remeshing_v2_summary.json" \
      --case-selection study --num-runs 300 --top-k 3 --seed 42 --views-per-test 2 \
      --geometry-decimation-factors 5,10 --geometry-label-preset v4 \
      --surface-query-points 65536 --volume-query-points 65536 --query-chunk-size 65536 \
      --plot-scales linear,log --font-scale 1.2 --no-std \
      --devices cuda:0,cuda:1,cuda:2,cuda:3,cuda:4 \
      --base-config pump --satloss-config pump_deal_from_smart_full \
      --downsample-config pump_downsample --gaussian-ball-masked-config pump_gaussian_ball_masked --box-masked-config pump_box_masked \
      --base-checkpoint "$ROOT/checkpoints/smart-pump-random1400-base-16k-pump-s42_best.pt" \
      --satloss-checkpoint "$ROOT/checkpoints/smart-pump-deal-random1400-from-smart-150ep-pump-s42_best.pt" \
      --downsample-checkpoint "$ROOT/checkpoints/smart-pump-downsample-random1400-downsample-200ep-pump-s42_best.pt" \
      --gaussian-ball-masked-checkpoint "$ROOT/checkpoints/smart-pump-gaussian-ball-masked-random1400-gaussian-ball-masked-200ep-pump-s42_best.pt" \
      --box-masked-checkpoint "$ROOT/checkpoints/smart-pump-box-masked-random1400-box-masked-200ep-pump-s42_best.pt" \
      --output-dir "$FINAL/shift_pump_random1400_endpoint_strategies_v4_pool300_top3"
    ;;

  heat_exchanger)
    HEAT_BOX="$ROOT/checkpoints/smart-heat-exchange-box-masked-heat-exchange-box-masked-200ep-toyheatexchange-s42_best.pt"
    if [[ ! -f "$HEAT_BOX" ]]; then
      echo "Heat Exchanger Box-mask checkpoint is not available yet: $HEAT_BOX" >&2
      exit 1
    fi
    CUDA_VISIBLE_DEVICES=0,1,2,3,4 "$PYTHON" "$ROOT/smart/scripts/compare_shift_endpoint_strategies.py" \
      --dataset heat_exchanger --data-root "$HEAT_DATA" \
      --study-summary "$HEAT_REMESH/remeshing_v2_summary.json" \
      --case-selection study --num-runs 288 --top-k 3 --seed 42 --views-per-test 2 \
      --geometry-decimation-factors 5,10 --geometry-label-preset v4 \
      --surface-query-points 32768 --volume-query-points 32768 --query-chunk-size 32768 \
      --plot-scales linear,log --font-scale 1.2 --no-std \
      --devices cuda:0,cuda:1,cuda:2,cuda:3,cuda:4 \
      --base-config toy_heat_exchange --satloss-config toy_heat_exchange_satloss7 \
      --downsample-config toy_heat_exchange_downsample --gaussian-ball-masked-config toy_heat_exchange_gaussian_ball_masked --box-masked-config toy_heat_exchange_box_masked \
      --base-checkpoint "$ROOT/checkpoints/smart-toy-heat-exchange-heat-exchange-base-ratio-aligned-toyheatexchange-s42_best.pt" \
      --satloss-checkpoint "$ROOT/checkpoints/smart-toy-heat-exchange-satloss7-heat-exchange-satloss-ratio-aligned-toyheatexchange-s42_best.pt" \
      --downsample-checkpoint "$ROOT/checkpoints/smart-heat-exchange-downsample-heat-exchange-downsample-200ep-toyheatexchange-s42_best.pt" \
      --gaussian-ball-masked-checkpoint "$ROOT/checkpoints/smart-heat-exchange-gaussian-ball-masked-heat-exchange-gaussian-ball-masked-200ep-toyheatexchange-s42_best.pt" \
      --box-masked-checkpoint "$HEAT_BOX" \
      --output-dir "$FINAL/heat_exchanger_endpoint_strategies_v4_pool288_top3"
    ;;

  *)
    echo "Usage: $0 {pump_remesh|pump|heat_exchanger}" >&2
    exit 2
    ;;
esac
