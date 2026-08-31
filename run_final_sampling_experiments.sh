#!/usr/bin/env bash
# Reproduce the final sampling-invariance studies with the V4 remeshes.
set -euo pipefail

ROOT=/home/parsa/smart_parsa
PYTHON=/home/parsa/miniconda3/envs/smart/bin/python
export PYTHONPATH="$ROOT/smart"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

FINAL="$ROOT/results/final"
DRIVAER_FEATURE=/mnt/ssdraid/parsa/drivaerml_surface_vtp_remesh_v4/feature
DRIVAER_QEM=/mnt/ssdraid/parsa/drivaerml_surface_vtp_remesh_v4/quadric
DRIVAER_VOXEL=/mnt/ssdraid/parsa/drivaerml_surface_vtp_remesh_v4/voxel
DRIVAER10=29,31,123,178,179,204,234,249,259,262
DRIVAER15=29,31,123,178,179,204,234,249,259,262,263,270,289,327,333

case "${1:-}" in
  cross_architecture)
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 "$PYTHON" smart/scripts/compare_drivaerml_sampling_invariance.py \
      --num-runs 10 --run-ids "$DRIVAER10" --seed 42 --shift-betas 0,1 --positive-shifts-only \
      --active-shifts beta,sine_y,sine_x --active-geometry-sources angle,isotropic,voxel \
      --geometry-decimation-factors 5,10 --geometry-label-preset v4 \
      --angle-decimated-vtp-dir "$DRIVAER_FEATURE" --isotropic-decimated-vtp-dir "$DRIVAER_QEM" --voxel-decimated-vtp-dir "$DRIVAER_VOXEL" \
      --views-per-mode 2 --view-batch-size 2 --model-repeats 1 --surface-query-points 65536 --volume-query-points 65536 \
      --batched-query-subregion-size 65536 --density-estimator kde --density-knn-k 16 --vtk-run-id 29 \
      --plot-workers 4 --font-scale 1.2 --no-std --satloss-only-percent-labels --devices cuda:0,cuda:1,cuda:2,cuda:3,cuda:4,cuda:5 \
      --smart-checkpoint "$ROOT/checkpoints/smart-smart-drivaerml-131k16kwr-drivaerml-s42_best.pt" \
      --smart-satloss7-checkpoint "$ROOT/checkpoints/smart-satloss7-smart-satloss7-drivaerml-131k-drivaerml-s42_best.pt" \
      --transolverpp-checkpoint "$ROOT/checkpoints/transolverpp-transolverpp-drivaerml-uniform-epochseeded-gpu0-200ep-drivaerml-s42_best.pt" \
      --transolverpp-satloss7-checkpoint "$ROOT/checkpoints/transolverpp-satloss7-transolverpp-satloss7-drivaerml-65k-drivaerml-s42_best.pt" \
      --pointnet2-ssg-checkpoint "$ROOT/checkpoints/pointnet2-ssg-pointnet2-ssg-drivaerml-65k-v2-drivaerml-s42_best.pt" \
      --pointnet2-ssg-satloss7-checkpoint "$ROOT/checkpoints/pointnet2-ssg-satloss7-pointnet2-ssg-satloss7-drivaerml-65k-drivaerml-s42_best.pt" \
      --lno-checkpoint "$ROOT/checkpoints/lno-lno-drivaerml-65k-drivaerml-s42_best.pt" \
      --lno-satloss7-checkpoint "$ROOT/checkpoints/lno-satloss7-lno-satloss7-drivaerml-65k-drivaerml-s42_best.pt" \
      --mspt-checkpoint "$ROOT/checkpoints/mspt-mspt-drivaerml-uniform-epochseeded-gpu6-200ep-drivaerml-s42_best.pt" \
      --mspt-satloss7-checkpoint "$ROOT/checkpoints/mspt-satloss7-mspt-satloss7-drivaerml-65k-drivaerml-s42_best.pt" \
      --point-transformer-v3-config drivaerml_point_transformer_v3_density_sensitive \
      --point-transformer-v3-satloss7-config drivaerml_point_transformer_v3_satloss7_density_sensitive \
      --point-transformer-v3-checkpoint "$ROOT/checkpoints/point-transformer-v3-ptv3-density-sensitive-drivaerml-drivaerml-s42_best.pt" \
      --point-transformer-v3-satloss7-checkpoint "$ROOT/checkpoints/point-transformer-v3-satloss7-ptv3-satloss7-density-sensitive-drivaerml-131k-drivaerml-s42_best.pt" \
      --output-dir "$FINAL/drivaerml_cross_architecture_deal_v4_full_data_10runs"
    ;;

  beta_range)
    CUDA_VISIBLE_DEVICES=0,1,2,3,4 "$PYTHON" smart/scripts/compare_drivaerml_satloss7_range_ablation.py \
      --experiment-preset range_ablation_vtp --num-runs 10 --run-ids "$DRIVAER10" --candidate-split all --seed 42 \
      --beta-levels 0,0.25,0.5,0.75,1 --sine-levels 0,0.25,0.5,0.75,1 --active-shifts beta,sine_y,sine_x \
      --active-geometry-sources angle,isotropic,voxel --geometry-decimation-factors 5,10 --geometry-label-preset v4 \
      --angle-decimated-vtp-dir "$DRIVAER_FEATURE" --isotropic-decimated-vtp-dir "$DRIVAER_QEM" --voxel-decimated-vtp-dir "$DRIVAER_VOXEL" \
      --views-per-mode 2 --view-batch-size 2 --model-repeats 1 --surface-query-points 65536 --volume-query-points 65536 \
      --batched-query-subregion-size 65536 --density-estimator kde --density-knn-k 16 --plot-scales linear,log \
      --font-scale 1.2 --y-pad-fraction 0.10 --no-std --exclude-range500 --devices cuda:0,cuda:1,cuda:2,cuda:3,cuda:4 \
      --smart-checkpoint "$ROOT/checkpoints/smart-smart-drivaerml-131k16kwr-drivaerml-s42_best.pt" \
      --range025-checkpoint "$ROOT/checkpoints/smart-satloss7-range025-smart-satloss7-range025-extension100-drivaerml-s42_best.pt" \
      --range050-checkpoint "$ROOT/checkpoints/smart-satloss7-range050-smart-satloss7-range050-extension100-drivaerml-s42_best.pt" \
      --range075-checkpoint "$ROOT/checkpoints/smart-satloss7-range075-smart-satloss7-range075-extension100-drivaerml-s42_best.pt" \
      --satloss7-checkpoint "$ROOT/checkpoints/smart-satloss7-range100-smart-satloss7-range100-from-smart-150ep-drivaerml-s42_best.pt" \
      --range200-checkpoint "$ROOT/checkpoints/smart-satloss7-range200-smart-satloss7-range200-from-smart-150ep-drivaerml-s42_best.pt" \
      --range300-checkpoint "$ROOT/checkpoints/smart-satloss7-range300-smart-satloss7-range300-from-smart-150ep-drivaerml-s42_best.pt" \
      --output-dir "$FINAL/drivaerml_beta_range_ablation_v4_10runs"
    ;;

  historical_augmentations)
    CUDA_VISIBLE_DEVICES=0,1,2,3,4 "$PYTHON" smart/scripts/compare_drivaerml_sampling_invariance.py \
      --strategy-only --num-runs 15 --run-ids "$DRIVAER15" --seed 42 --shift-betas 0,1 --positive-shifts-only \
      --active-shifts beta,sine_y,sine_x --active-geometry-sources angle,isotropic,voxel --geometry-decimation-factors 5,10 --geometry-label-preset v4 \
      --angle-decimated-vtp-dir "$DRIVAER_FEATURE" --isotropic-decimated-vtp-dir "$DRIVAER_QEM" --voxel-decimated-vtp-dir "$DRIVAER_VOXEL" \
      --views-per-mode 2 --view-batch-size 2 --model-repeats 1 --surface-query-points 65536 --volume-query-points 65536 \
      --batched-query-subregion-size 65536 --density-estimator kde --density-knn-k 16 --vtk-run-id 29 --plot-workers 4 \
      --font-scale 1.2 --no-std --devices cuda:0,cuda:1,cuda:2,cuda:3,cuda:4 \
      --smart-checkpoint "$ROOT/checkpoints/smart-smart-drivaerml-131k16kwr-drivaerml-s42_best.pt" \
      --smart-downsample-checkpoint "$ROOT/checkpoints/smart-downsample-smart-downsample-drivaerml-finecoarse-consistency-drivaerml-s42_best.pt" \
      --smart-gaussian-ball-masked-checkpoint "$ROOT/checkpoints/smart-gaussian-ball-masked-smart-gaussian-ball-masked-drivaerml-gaussianball-dp12-drivaerml-s42_best.pt" \
      --smart-box-masked-checkpoint "$ROOT/checkpoints/smart-box-masked-boxmasked-drivaerml-dp12-bs2-drivaerml-s42_best.pt" \
      --smart-satloss7-checkpoint "$ROOT/checkpoints/smart-satloss7-smart-satloss7-drivaerml-131k-drivaerml-s42_best.pt" \
      --output-dir "$FINAL/drivaerml_historical_augmentations_v4_full_data_15runs"
    ;;

  kde_ablation)
    CUDA_VISIBLE_DEVICES=0,1,2,3,4 "$PYTHON" smart/scripts/compare_drivaerml_satloss7_range_ablation.py \
      --experiment-preset kde_ablation_vtp --num-runs 10 --run-ids "$DRIVAER10" --candidate-split all --seed 42 --shift-levels 0,0.25,0.5,0.75,1 \
      --active-shifts beta,sine_y,sine_x --active-geometry-sources angle,isotropic,voxel --geometry-decimation-factors 5,10 --geometry-label-preset v4 \
      --angle-decimated-vtp-dir "$DRIVAER_FEATURE" --isotropic-decimated-vtp-dir "$DRIVAER_QEM" --voxel-decimated-vtp-dir "$DRIVAER_VOXEL" \
      --views-per-mode 2 --view-batch-size 2 --model-repeats 1 --surface-query-points 65536 --volume-query-points 65536 \
      --batched-query-subregion-size 65536 --density-estimator kde --density-knn-k 16 --plot-scales linear,log \
      --font-scale 1.2 --y-pad-fraction 0.10 --no-std --devices cuda:0,cuda:1,cuda:2,cuda:3,cuda:4 \
      --smart-checkpoint "$ROOT/checkpoints/smart-smart-drivaerml-131k16kwr-drivaerml-s42_best.pt" \
      --kde4-checkpoint "$ROOT/checkpoints/smart-satloss7-range100-kde4-smart-satloss7-range100-kde4-drivaerml-s42_best.pt" \
      --kde8-checkpoint "$ROOT/checkpoints/smart-satloss7-range100-kde8-smart-satloss7-range100-kde8-drivaerml-s42_best.pt" \
      --kde16-checkpoint "$ROOT/checkpoints/smart-satloss7-range100-smart-satloss7-range100-from-smart-150ep-drivaerml-s42_best.pt" \
      --kde32-checkpoint "$ROOT/checkpoints/smart-satloss7-range100-kde32-smart-satloss7-range100-kde32-drivaerml-s42_best.pt" \
      --kde64-checkpoint "$ROOT/checkpoints/smart-satloss7-range100-kde64-smart-satloss7-range100-kde64-drivaerml-s42_best.pt" \
      --output-dir "$FINAL/drivaerml_kde_ablation_v4_full_data_10runs"
    ;;

  consistency_ablation)
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 "$PYTHON" smart/scripts/compare_drivaerml_satloss7_consistency_ablation.py \
      --experiment-preset consistency_ablation_vtp --num-runs 5 --run-selection top_pairwise_improvement --top-selection-candidates 247 \
      --candidate-split all --screen-case-batch-size 8 --top-selection-improved-model SMART_SATLOSS7_RANGE025 \
      --top-selection-reference-model SMART_SATLOSS7_RANGE050 --top-selection-conditions sine_y,sine_x,remeshing \
      --seed 42 --shift-levels 0,1 --active-shifts sine_y,sine_x --active-geometry-sources angle,isotropic,voxel \
      --geometry-decimation-factors 5,10 --geometry-label-preset v4 --angle-decimated-vtp-dir "$DRIVAER_FEATURE" \
      --isotropic-decimated-vtp-dir "$DRIVAER_QEM" --voxel-decimated-vtp-dir "$DRIVAER_VOXEL" \
      --views-per-mode 2 --view-batch-size 2 --model-repeats 1 --surface-query-points 65536 --volume-query-points 65536 \
      --batched-query-subregion-size 65536 --density-estimator kde --density-knn-k 16 \
      --plot-scales linear,log --font-scale 1.65 --y-pad-fraction 0.05 --no-std --compact-endpoint-summary \
      --devices cuda:0,cuda:1,cuda:2,cuda:3,cuda:4,cuda:5 --smart-config drivaerml \
      --w-consistency-config drivaerml_satloss7_range100 --wo-consistency-config drivaerml_satloss7_range100_no_consistency_from_scratch \
      --smart-checkpoint "$ROOT/checkpoints/smart-smart-drivaerml-131k16kwr-drivaerml-s42_best.pt" \
      --w-consistency-checkpoint "$ROOT/checkpoints/smart-satloss7-range100-smart-satloss7-range100-from-smart-150ep-drivaerml-s42_best.pt" \
      --wo-consistency-checkpoint "$ROOT/checkpoints/smart-satloss7-range100-no-consistency-from-scratch-smart-satloss7-range100-no-consistency-from-scratch-drivaerml-s42_best.pt" \
      --output-dir "$FINAL/drivaerml_consistency_ablation_v4_pool247_top5"
    ;;

  pump)
    CUDA_VISIBLE_DEVICES=0,1,2,3,4 "$PYTHON" smart/scripts/compare_shift_endpoint_strategies.py \
      --dataset pump --data-root /mnt/ssdraid/parsa/shift_pump_preprocessed \
      --study-summary /mnt/ssdraid/parsa/shift_pump_surface_vtp_remesh_v4/remeshing_v2_summary.json \
      --case-selection study --num-runs 100 --top-k 2 --seed 42 --views-per-test 2 \
      --geometry-decimation-factors 5,10 --geometry-label-preset v4 --surface-query-points 65536 --volume-query-points 65536 \
      --query-chunk-size 65536 --plot-scales linear,log --font-scale 1.2 --no-std --devices cuda:0,cuda:1,cuda:2,cuda:3,cuda:4 \
      --base-config pump --satloss-config pump_satloss7_range100_from_smart --downsample-config pump_satloss7_downsample \
      --gaussian-ball-masked-config pump_satloss7_gaussian_ball_masked --box-masked-config pump_satloss7_box_masked \
      --base-checkpoint "$ROOT/checkpoints/smart-pump-base-16k-pump-s42_best.pt" \
      --satloss-checkpoint "$ROOT/checkpoints/smart-pump-satloss7-range100-from-smart-satloss7-range100-from-smart-150ep-pump-s42_best.pt" \
      --downsample-checkpoint "$ROOT/checkpoints/smart-pump-satloss7-downsample-satloss7-downsample-16k-pump-s42_best.pt" \
      --gaussian-ball-masked-checkpoint "$ROOT/checkpoints/smart-pump-satloss7-gaussian-ball-masked-satloss7-gaussian-ball-masked-16k-pump-s42_best.pt" \
      --box-masked-checkpoint "$ROOT/checkpoints/smart-pump-satloss7-box-masked-satloss7-box-masked-16k-pump-s42_best.pt" \
      --output-dir "$FINAL/shift_pump_endpoint_strategies_v4_pool100_top2"
    ;;

  heat_exchanger)
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 "$PYTHON" smart/scripts/compare_toy_heat_exchange_all_models_sampling_invariance.py \
      --data-root /mnt/ssdraid/parsa/toy_heat_exchange_fem_v1 --candidate-pool-size 100 --top-k 3 --candidate-split all \
      --ranking-models POINTNET2_SSG --ranking-modes isotropic_div5,isotropic_div10 --seed 42 --devices cuda:0,cuda:1,cuda:2,cuda:3,cuda:4,cuda:5 \
      --surface-query-points 32768 --volume-query-points 32768 --active-geometry-sources isotropic --geometry-decimation-factors 5,10 \
      --geometry-label-preset v4 --isotropic-decimated-vtp-dir /mnt/ssdraid/parsa/toy_heat_exchange_surface_vtp_remesh_v4/quadric \
      --original-vtp-dir /mnt/ssdraid/parsa/toy_heat_exchange_surface_vtp --font-scale 1.2 --analysis-case-count 1 \
      --smart-checkpoint "$ROOT/checkpoints/smart-toy-heat-exchange-heat-exchange-base-ratio-aligned-toyheatexchange-s42_best.pt" \
      --smart-satloss7-checkpoint "$ROOT/checkpoints/smart-toy-heat-exchange-satloss7-heat-exchange-satloss-ratio-aligned-toyheatexchange-s42_best.pt" \
      --mspt-checkpoint "$ROOT/checkpoints/mspt-toy-heat-exchange-mspt-base-toyheatexchange-s42_best.pt" \
      --mspt-satloss7-checkpoint "$ROOT/checkpoints/mspt-toy-heat-exchange-satloss7-mspt-satloss-from-base-toyheatexchange-s42_best.pt" \
      --lno-checkpoint "$ROOT/checkpoints/lno-toy-heat-exchange-lno-base-toyheatexchange-s42_best.pt" \
      --lno-satloss7-checkpoint "$ROOT/checkpoints/lno-toy-heat-exchange-satloss7-lno-satloss7-from-base-toyheatexchange-s42_best.pt" \
      --pointnet2-ssg-checkpoint "$ROOT/checkpoints/pointnet2-ssg-toy-heat-exchange-pointnet2-ssg-base-toyheatexchange-s42_best.pt" \
      --pointnet2-ssg-satloss7-checkpoint "$ROOT/checkpoints/pointnet2-ssg-toy-heat-exchange-satloss7-pointnet2-ssg-satloss7-from-base-toyheatexchange-s42_best.pt" \
      --transolverpp-checkpoint "$ROOT/checkpoints/transolverpp-toy-heat-exchange-transolverpp-base-toyheatexchange-s42_best.pt" \
      --transolverpp-satloss7-checkpoint "$ROOT/checkpoints/transolverpp-toy-heat-exchange-satloss7-transolverpp-satloss-from-base-toyheatexchange-s42_best.pt" \
      --point-transformer-v3-checkpoint "$ROOT/checkpoints/point-transformer-v3-toy-heat-exchange-point-transformer-v3-base-toyheatexchange-s42_best.pt" \
      --point-transformer-v3-satloss7-checkpoint "$ROOT/checkpoints/point-transformer-v3-toy-heat-exchange-satloss7-point-transformer-v3-satloss7-from-base-toyheatexchange-s42_best.pt" \
      --output-dir "$FINAL/heat_exchanger_all_models_deal_qem_pool100_top3_pointnet2_remesh_ranking"
    ;;

  *)
    echo "Usage: $0 {cross_architecture|beta_range|historical_augmentations|kde_ablation|consistency_ablation|pump|heat_exchanger}" >&2
    exit 2
    ;;
esac
