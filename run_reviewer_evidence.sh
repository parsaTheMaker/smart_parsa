#!/usr/bin/env bash
# Frozen reviewer-evidence protocol. This script never edits the manuscript.
set -euo pipefail

ROOT=/home/parsa/smart_parsa
PYTHON=/home/parsa/miniconda3/envs/smart/bin/python
# Override only for isolated smoke runs; the default is the final evidence
# location used by the paper workflow.
OUT="${REVIEWER_EVIDENCE_OUTPUT_DIR:-$ROOT/results/final/reviewer_evidence_20260901}"
export PYTHONPATH="$ROOT/smart"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Select every GPU with at least 85% of its memory free. CUDA_VISIBLE_DEVICES
# remaps these physical IDs to consecutive local ordinals, which is exactly the
# device notation expected by the Python comparison scripts.
select_free_gpus() {
  local minimum_count=${1:-4}
  local free_fraction=${FREE_GPU_FRACTION:-0.85}
  local selected=()
  while IFS=, read -r index free total; do
    [[ -n "$index" && "$total" -gt 0 ]] || continue
    if "$PYTHON" - "$free" "$total" "$free_fraction" <<'PY'
import sys
free, total, threshold = map(float, sys.argv[1:])
raise SystemExit(0 if free / total >= threshold else 1)
PY
    then
      selected+=("$index")
    fi
  done < <(nvidia-smi --query-gpu=index,memory.free,memory.total --format=csv,noheader,nounits)
  if (( ${#selected[@]} < minimum_count )); then
    echo "Need at least ${minimum_count} GPUs with >=${free_fraction} free VRAM; found ${#selected[@]} (${selected[*]:-none})." >&2
    exit 1
  fi
  GPU_IDS=$(IFS=,; echo "${selected[*]}")
  GPU_DEVICES=$("$PYTHON" - "${#selected[@]}" <<'PY'
import sys
print(','.join(f'cuda:{i}' for i in range(int(sys.argv[1]))))
PY
)
  echo "Using ${#selected[@]} free GPUs (physical IDs: $GPU_IDS; local devices: $GPU_DEVICES)." >&2
}

drivaerml() {
  local output="$OUT/drivaerml_frozen_test50_views10"
  local num_runs="${DRIVAER_NUM_RUNS:-50}"
  local views_per_mode="${DRIVAER_VIEWS_PER_MODE:-10}"
  local expected_rows=$((num_runs * views_per_mode * 12 * 9))
  local metrics_file="$output/per_view_metrics.csv"

  # The evaluator writes all scientific tables and plots before optional VTK
  # export. Reuse a complete frozen evaluation if that optional final export
  # failed, rather than repeating an expensive 50-case experiment.
  if [[ -s "$metrics_file" && -f "$output/aggregate_metrics.csv" && -f "$output/robustness_summary.csv" ]]; then
    local observed_rows
    observed_rows=$(( $(wc -l < "$metrics_file") - 1 ))
    if (( observed_rows == expected_rows )); then
      echo "[DrivAerML] Reusing completed frozen metrics: $observed_rows rows in $output."
      return
    fi
  fi

  select_free_gpus 4
  CUDA_VISIBLE_DEVICES="$GPU_IDS" "$PYTHON" "$ROOT/smart/scripts/compare_drivaerml_sampling_invariance.py" \
    --num-runs "${DRIVAER_NUM_RUNS:-50}" --seed 42 --positive-shifts-only --active-shifts sine_y,sine_x \
    --active-geometry-sources angle,isotropic,voxel --geometry-decimation-factors 5,10 --geometry-label-preset v4 \
    --angle-decimated-vtp-dir /mnt/ssdraid/parsa/drivaerml_surface_vtp_remesh_v4/feature \
    --isotropic-decimated-vtp-dir /mnt/ssdraid/parsa/drivaerml_surface_vtp_remesh_v4/quadric \
    --voxel-decimated-vtp-dir /mnt/ssdraid/parsa/drivaerml_surface_vtp_remesh_v4/voxel \
    --views-per-mode "${DRIVAER_VIEWS_PER_MODE:-10}" --view-batch-size "${DRIVAER_VIEW_BATCH_SIZE:-2}" --model-repeats 1 \
    --surface-query-points "${DRIVAER_SURFACE_QUERY_POINTS:-65536}" --volume-query-points "${DRIVAER_VOLUME_QUERY_POINTS:-65536}" \
    --batched-query-subregion-size "${DRIVAER_QUERY_CHUNK_SIZE:-65536}" --density-estimator kde --density-knn-k 16 --vtk-run-id 29 --plot-workers 4 \
    --font-scale 1.2 --satloss-only-percent-labels --devices "$GPU_DEVICES" \
    --smart-checkpoint "$ROOT/checkpoints/smart-smart-drivaerml-131k16kwr-drivaerml-s42_best.pt" \
    --smart-satloss7-config drivaerml_satloss7_range100 \
    --smart-satloss7-checkpoint "$ROOT/checkpoints/smart-satloss7-range100-smart-satloss7-range100-from-smart-150ep-drivaerml-s42_best.pt" \
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
    --output-dir "$output"
}

drivaerml_strategies() {
  local output="$OUT/drivaerml_frozen_strategies_test50_views10"
  local run_ids="1,21,101,108,116,121,132,133,136,139,141,146,152,153,160,170,173,178,180,182,185,186,187,199,200,205,206,214,222,224,230,231,235,237,239,249,250,260,262,275,279,287,288,303,304,313,314,315,331,334"
  local expected_rows=$((50 * 10 * 5 * 9))
  local metrics_file="$output/per_view_metrics.csv"

  if [[ -s "$metrics_file" && -f "$output/aggregate_metrics.csv" && -f "$output/robustness_summary.csv" ]]; then
    local observed_rows
    observed_rows=$(( $(wc -l < "$metrics_file") - 1 ))
    if (( observed_rows == expected_rows )); then
      echo "[DrivAerML strategies] Reusing completed frozen metrics: $observed_rows rows in $output."
      "$PYTHON" "$ROOT/smart/scripts/create_top20_paper_vs_frozen_tables.py" \
        --output "$OUT/paper_vs_frozen_top20_diagnostic.pdf"
      return
    fi
  fi

  select_free_gpus 4
  CUDA_VISIBLE_DEVICES="$GPU_IDS" "$PYTHON" "$ROOT/smart/scripts/compare_drivaerml_sampling_invariance.py" \
    --strategy-only --num-runs 50 --run-ids "$run_ids" --seed 42 --positive-shifts-only --active-shifts sine_y,sine_x \
    --active-geometry-sources angle,isotropic,voxel --geometry-decimation-factors 5,10 --geometry-label-preset v4 \
    --angle-decimated-vtp-dir /mnt/ssdraid/parsa/drivaerml_surface_vtp_remesh_v4/feature \
    --isotropic-decimated-vtp-dir /mnt/ssdraid/parsa/drivaerml_surface_vtp_remesh_v4/quadric \
    --voxel-decimated-vtp-dir /mnt/ssdraid/parsa/drivaerml_surface_vtp_remesh_v4/voxel \
    --views-per-mode 10 --view-batch-size 2 --model-repeats 1 \
    --surface-query-points 65536 --volume-query-points 65536 --batched-query-subregion-size 65536 \
    --density-estimator kde --density-knn-k 16 --skip-representative-exports --plot-workers 8 --font-scale 1.2 \
    --satloss-only-percent-labels --devices "$GPU_DEVICES" \
    --smart-config drivaerml \
    --smart-downsample-config drivaerml_smart_downsample \
    --smart-gaussian-ball-masked-config drivaerml_smart_gaussian_ball_masked \
    --smart-box-masked-config drivaerml_smart_box_masked \
    --smart-satloss7-config drivaerml_satloss7_range100 \
    --smart-checkpoint "$ROOT/checkpoints/smart-smart-drivaerml-131k16kwr-drivaerml-s42_best.pt" \
    --smart-downsample-checkpoint "$ROOT/checkpoints/smart-downsample-smart-downsample-drivaerml-finecoarse-consistency-drivaerml-s42_best.pt" \
    --smart-gaussian-ball-masked-checkpoint "$ROOT/checkpoints/smart-gaussian-ball-masked-smart-gaussian-ball-masked-drivaerml-gaussianball-dp12-drivaerml-s42_best.pt" \
    --smart-box-masked-checkpoint "$ROOT/checkpoints/smart-box-masked-boxmasked-drivaerml-dp12-bs2-drivaerml-s42_best.pt" \
    --smart-satloss7-checkpoint "$ROOT/checkpoints/smart-satloss7-range100-smart-satloss7-range100-from-smart-150ep-drivaerml-s42_best.pt" \
    --output-dir "$output"
  "$PYTHON" "$ROOT/smart/scripts/create_top20_paper_vs_frozen_tables.py" \
    --output "$OUT/paper_vs_frozen_top20_diagnostic.pdf"
}

pump() {
  select_free_gpus 4
  CUDA_VISIBLE_DEVICES="$GPU_IDS" "$PYTHON" "$ROOT/smart/scripts/compare_shift_endpoint_strategies.py" \
    --dataset pump --data-root /mnt/data/parsa/shift_pump_random1400_preprocessed \
    --study-summary /mnt/data/parsa/shift_pump_random1400_surface_vtp_remesh_v4/remeshing_v2_summary.json \
    --case-selection test --selection-policy all_cases --include-original --original-base-only --num-runs 0 --seed 42 --views-per-test 10 --inference-batch-size 4 \
    --active-geometry-sources feature,quadric,voxel --geometry-decimation-factors 5,10 --geometry-label-preset v4 \
    --surface-query-points 65536 --volume-query-points 65536 --query-chunk-size 65536 --plot-scales linear,log \
    --font-scale 1.2 --min-free-gib 8 --devices "$GPU_DEVICES" \
    --base-config pump --satloss-config pump_deal_from_smart_full --downsample-config pump_downsample \
    --gaussian-ball-masked-config pump_gaussian_ball_masked --box-masked-config pump_box_masked \
    --base-checkpoint "$ROOT/checkpoints/smart-pump-random1400-base-16k-pump-s42_best.pt" \
    --satloss-checkpoint "$ROOT/checkpoints/smart-pump-deal-random1400-from-smart-150ep-pump-s42_best.pt" \
    --downsample-checkpoint "$ROOT/checkpoints/smart-pump-downsample-random1400-downsample-200ep-pump-s42_best.pt" \
    --gaussian-ball-masked-checkpoint "$ROOT/checkpoints/smart-pump-gaussian-ball-masked-random1400-gaussian-ball-masked-200ep-pump-s42_best.pt" \
    --box-masked-checkpoint "$ROOT/checkpoints/smart-pump-box-masked-random1400-box-masked-200ep-pump-s42_best.pt" \
    --output-dir "$OUT/pump_frozen_test_all_views10"
}

heat_exchanger() {
  select_free_gpus 4
  CUDA_VISIBLE_DEVICES="$GPU_IDS" "$PYTHON" "$ROOT/smart/scripts/compare_shift_endpoint_strategies.py" \
    --dataset heat_exchanger --data-root /mnt/ssdraid/parsa/toy_heat_exchange_fem_v1 \
    --study-summary /mnt/ssdraid/parsa/toy_heat_exchange_surface_vtp_remesh_v4/remeshing_v2_summary.json \
    --case-selection test --selection-policy all_cases --include-original --original-base-only --num-runs 0 --seed 42 --views-per-test 10 --inference-batch-size 4 \
    --active-geometry-sources feature,quadric --geometry-decimation-factors 5,10 --geometry-label-preset v4 \
    --surface-query-points 32768 --volume-query-points 32768 --query-chunk-size 32768 --plot-scales linear,log \
    --font-scale 1.2 --min-free-gib 8 --devices "$GPU_DEVICES" \
    --base-config toy_heat_exchange --satloss-config toy_heat_exchange_satloss7 --downsample-config toy_heat_exchange_downsample \
    --gaussian-ball-masked-config toy_heat_exchange_gaussian_ball_masked --box-masked-config toy_heat_exchange_box_masked \
    --base-checkpoint "$ROOT/checkpoints/smart-toy-heat-exchange-heat-exchange-base-ratio-aligned-toyheatexchange-s42_best.pt" \
    --satloss-checkpoint "$ROOT/checkpoints/smart-toy-heat-exchange-satloss7-heat-exchange-satloss-ratio-aligned-toyheatexchange-s42_best.pt" \
    --downsample-checkpoint "$ROOT/checkpoints/smart-heat-exchange-downsample-heat-exchange-downsample-200ep-toyheatexchange-s42_best.pt" \
    --gaussian-ball-masked-checkpoint "$ROOT/checkpoints/smart-heat-exchange-gaussian-ball-masked-heat-exchange-gaussian-ball-masked-200ep-toyheatexchange-s42_best.pt" \
    --box-masked-checkpoint "$ROOT/checkpoints/smart-heat-exchange-box-masked-heat-exchange-box-masked-200ep-toyheatexchange-s42_best.pt" \
    --output-dir "$OUT/heat_exchanger_frozen_validation_all_views10"
}

bootstrap() {
  "$PYTHON" "$ROOT/smart/scripts/bootstrap_sampling_evaluation_ci.py" --input "$OUT/drivaerml_frozen_test50_views10/per_view_metrics.csv" --output "$OUT/drivaerml_frozen_test50_views10/paired_bootstrap_smart_deal.csv" --reference-model SMART --comparison-model SMART_SATLOSS7 --conditions original_uniform,sine_x_1,sine_y_1,remeshing_div5_mean,remeshing_div10_mean --reference-only-conditions original_uniform --bootstrap-samples 10000 --seed 42
  "$PYTHON" "$ROOT/smart/scripts/bootstrap_sampling_evaluation_ci.py" --input "$OUT/pump_frozen_test_all_views10/combined_global_endpoint_metrics.csv" --output "$OUT/pump_frozen_test_all_views10/paired_bootstrap_smart_deal.csv" --reference-model base --comparison-model satloss --conditions original_uniform,sine_x_1,sine_y_1,remeshing_div5_mean,remeshing_div10_mean --reference-only-conditions original_uniform --bootstrap-samples 10000 --seed 42
  "$PYTHON" "$ROOT/smart/scripts/bootstrap_sampling_evaluation_ci.py" --input "$OUT/heat_exchanger_frozen_validation_all_views10/combined_global_endpoint_metrics.csv" --output "$OUT/heat_exchanger_frozen_validation_all_views10/paired_bootstrap_smart_deal.csv" --reference-model base --comparison-model satloss --conditions original_uniform,sine_x_1,sine_y_1,remeshing_div5_mean,remeshing_div10_mean --reference-only-conditions original_uniform --bootstrap-samples 10000 --seed 42
}

geometry() {
  "$PYTHON" "$ROOT/smart/scripts/validate_remesh_geometry.py" --dataset drivaerml --source-dir /mnt/ssdraid/parsa/drivaerml_surface_vtp --remesh-dir /mnt/ssdraid/parsa/drivaerml_surface_vtp_remesh_v4 --output-dir "$OUT/drivaerml_remesh_geometry" --methods voxel,quadric,feature --factors 5,10 --max-cases 100 --distance-samples 50000 --normal-samples 5000 --workers 8 --seed 42
  "$PYTHON" "$ROOT/smart/scripts/validate_remesh_geometry.py" --dataset pump --source-dir /mnt/data/parsa/shift_pump_raw_random1400 --remesh-dir /mnt/data/parsa/shift_pump_random1400_surface_vtp_remesh_v4 --output-dir "$OUT/pump_remesh_geometry" --methods voxel,quadric,feature --factors 5,10 --max-cases 100 --distance-samples 50000 --normal-samples 5000 --workers 8 --seed 42
  "$PYTHON" "$ROOT/smart/scripts/validate_remesh_geometry.py" --dataset heat_exchanger --source-dir /mnt/ssdraid/parsa/toy_heat_exchange_surface_vtp --remesh-dir /mnt/ssdraid/parsa/toy_heat_exchange_surface_vtp_remesh_v4 --output-dir "$OUT/heat_exchanger_remesh_geometry" --methods voxel,quadric,feature --factors 5,10 --max-cases 100 --distance-samples 50000 --normal-samples 5000 --workers 8 --seed 42
}

documentation() {
  "$PYTHON" "$ROOT/smart/scripts/audit_toy_heat_exchange_generation.py" --data-root /mnt/ssdraid/parsa/toy_heat_exchange_fem_v1 --output-dir "$OUT/heat_exchanger_generation_audit"
  "$PYTHON" "$ROOT/smart/scripts/export_paper_task_cards.py" --output-json "$OUT/task_cards.json" --output-markdown "$OUT/task_cards.md" --output-latex "$OUT/task_cards.tex"
  "$PYTHON" "$ROOT/smart/scripts/export_paper_reproducibility_snapshot.py" --output-dir "$OUT/reproducibility" --latex-output "$OUT/reproducibility/reproducibility_snapshot.tex" --config pump_base=pump --config pump_deal=pump_deal_from_smart_full --config heat_base=toy_heat_exchange --config heat_deal=toy_heat_exchange_satloss7 --checkpoint pump_base="$ROOT/checkpoints/smart-pump-random1400-base-16k-pump-s42_best.pt" --checkpoint pump_deal="$ROOT/checkpoints/smart-pump-deal-random1400-from-smart-150ep-pump-s42_best.pt" --checkpoint heat_base="$ROOT/checkpoints/smart-toy-heat-exchange-heat-exchange-base-ratio-aligned-toyheatexchange-s42_best.pt" --checkpoint heat_deal="$ROOT/checkpoints/smart-toy-heat-exchange-satloss7-heat-exchange-satloss-ratio-aligned-toyheatexchange-s42_best.pt"
}

case "${1:-}" in
  drivaerml|drivaerml_strategies|pump|heat_exchanger|bootstrap|geometry|documentation) "$1" ;;
  all) drivaerml; pump; heat_exchanger; bootstrap; geometry; documentation ;;
  *) echo "Usage: bash $0 {drivaerml|drivaerml_strategies|pump|heat_exchanger|bootstrap|geometry|documentation|all}" >&2; exit 2 ;;
esac
