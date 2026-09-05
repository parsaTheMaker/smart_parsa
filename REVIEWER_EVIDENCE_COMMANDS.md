# Reviewer-Evidence Commands

These commands address reviewer points 2, 3, 5, 6, 8, and 9 without editing
the manuscript.  Every comparison uses `Original`, sine-x, sine-y, and the
div5/div10 remeshes only: beta is deliberately excluded from evaluation.

All outputs are written beneath `results/final/reviewer_evidence_20260901`.
Run the commands in the order shown.

## 1. DrivAerML: Nominal Accuracy, Fieldwise Metrics, and Evaluation Noise

This is the frozen 50-case held-out evaluation.  It compares every existing
cross-architecture Base/DeAL pair on the same deterministic cohort with ten
independently sampled encoder views per case and condition.  It writes
`per_view_metrics.csv`, including every surface and volume field.

```bash
cd /home/parsa/smart_parsa

PYTHONPATH=/home/parsa/smart_parsa/smart \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
PYTHONUNBUFFERED=1 \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/home/parsa/miniconda3/envs/smart/bin/python \
smart/scripts/compare_drivaerml_sampling_invariance.py \
  --num-runs 50 \
  --seed 42 \
  --positive-shifts-only \
  --active-shifts sine_y,sine_x \
  --active-geometry-sources angle,isotropic,voxel \
  --geometry-decimation-factors 5,10 \
  --geometry-label-preset v4 \
  --angle-decimated-vtp-dir /mnt/ssdraid/parsa/drivaerml_surface_vtp_remesh_v4/feature \
  --isotropic-decimated-vtp-dir /mnt/ssdraid/parsa/drivaerml_surface_vtp_remesh_v4/quadric \
  --voxel-decimated-vtp-dir /mnt/ssdraid/parsa/drivaerml_surface_vtp_remesh_v4/voxel \
  --views-per-mode 10 \
  --view-batch-size 2 \
  --model-repeats 1 \
  --surface-query-points 65536 \
  --volume-query-points 65536 \
  --batched-query-subregion-size 65536 \
  --density-estimator kde \
  --density-knn-k 16 \
  --vtk-run-id 29 \
  --plot-workers 4 \
  --font-scale 1.2 \
  --satloss-only-percent-labels \
  --devices cuda:0,cuda:1,cuda:2,cuda:3,cuda:4,cuda:5 \
  --smart-checkpoint checkpoints/smart-smart-drivaerml-131k16kwr-drivaerml-s42_best.pt \
  --smart-satloss7-config drivaerml_satloss7_range100 \
  --smart-satloss7-checkpoint checkpoints/smart-satloss7-range100-smart-satloss7-range100-from-smart-150ep-drivaerml-s42_best.pt \
  --transolverpp-checkpoint checkpoints/transolverpp-transolverpp-drivaerml-uniform-epochseeded-gpu0-200ep-drivaerml-s42_best.pt \
  --transolverpp-satloss7-checkpoint checkpoints/transolverpp-satloss7-transolverpp-satloss7-drivaerml-65k-drivaerml-s42_best.pt \
  --pointnet2-ssg-checkpoint checkpoints/pointnet2-ssg-pointnet2-ssg-drivaerml-65k-v2-drivaerml-s42_best.pt \
  --pointnet2-ssg-satloss7-checkpoint checkpoints/pointnet2-ssg-satloss7-pointnet2-ssg-satloss7-drivaerml-65k-drivaerml-s42_best.pt \
  --lno-checkpoint checkpoints/lno-lno-drivaerml-65k-drivaerml-s42_best.pt \
  --lno-satloss7-checkpoint checkpoints/lno-satloss7-lno-satloss7-drivaerml-65k-drivaerml-s42_best.pt \
  --mspt-checkpoint checkpoints/mspt-mspt-drivaerml-uniform-epochseeded-gpu6-200ep-drivaerml-s42_best.pt \
  --mspt-satloss7-checkpoint checkpoints/mspt-satloss7-mspt-satloss7-drivaerml-65k-drivaerml-s42_best.pt \
  --point-transformer-v3-config drivaerml_point_transformer_v3_density_sensitive \
  --point-transformer-v3-satloss7-config drivaerml_point_transformer_v3_satloss7_density_sensitive \
  --point-transformer-v3-checkpoint checkpoints/point-transformer-v3-ptv3-density-sensitive-drivaerml-drivaerml-s42_best.pt \
  --point-transformer-v3-satloss7-checkpoint checkpoints/point-transformer-v3-satloss7-ptv3-satloss7-density-sensitive-drivaerml-131k-drivaerml-s42_best.pt \
  --output-dir results/final/reviewer_evidence_20260901/drivaerml_frozen_test50_views10
```

## 2. Pump: Nominal Accuracy, Fieldwise Metrics, and Evaluation Noise

This evaluates all 280 held-out Pump cases.  `--selection-policy all_cases`
prevents top-case selection, while `--include-original` adds nominal accuracy.

```bash
cd /home/parsa/smart_parsa

PYTHONPATH=/home/parsa/smart_parsa/smart \
CUDA_VISIBLE_DEVICES=0,1,2,3,4 \
PYTHONUNBUFFERED=1 \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/home/parsa/miniconda3/envs/smart/bin/python \
smart/scripts/compare_shift_endpoint_strategies.py \
  --dataset pump \
  --data-root /mnt/data/parsa/shift_pump_random1400_preprocessed \
  --study-summary /mnt/data/parsa/shift_pump_random1400_surface_vtp_remesh_v4/remeshing_v2_summary.json \
  --case-selection test \
  --selection-policy all_cases \
  --include-original \
  --num-runs 0 \
  --seed 42 \
  --views-per-test 10 \
  --active-geometry-sources feature,quadric,voxel \
  --geometry-decimation-factors 5,10 \
  --geometry-label-preset v4 \
  --surface-query-points 65536 \
  --volume-query-points 65536 \
  --query-chunk-size 65536 \
  --plot-scales linear,log \
  --font-scale 1.2 \
  --min-free-gib 8 \
  --devices cuda:0,cuda:1,cuda:2,cuda:3,cuda:4 \
  --base-config pump \
  --satloss-config pump_deal_from_smart_full \
  --downsample-config pump_downsample \
  --gaussian-ball-masked-config pump_gaussian_ball_masked \
  --box-masked-config pump_box_masked \
  --base-checkpoint checkpoints/smart-pump-random1400-base-16k-pump-s42_best.pt \
  --satloss-checkpoint checkpoints/smart-pump-deal-random1400-from-smart-150ep-pump-s42_best.pt \
  --downsample-checkpoint checkpoints/smart-pump-downsample-random1400-downsample-200ep-pump-s42_best.pt \
  --gaussian-ball-masked-checkpoint checkpoints/smart-pump-gaussian-ball-masked-random1400-gaussian-ball-masked-200ep-pump-s42_best.pt \
  --box-masked-checkpoint checkpoints/smart-pump-box-masked-random1400-box-masked-200ep-pump-s42_best.pt \
  --output-dir results/final/reviewer_evidence_20260901/pump_frozen_test_all_views10
```

## 3. Heat Exchanger: Nominal Accuracy, Fieldwise Metrics, and Evaluation Noise

This evaluates every 32-case validation geometry.  QEM and feature-aware
remeshes are used, matching the current Heat Exchanger study.

```bash
cd /home/parsa/smart_parsa

PYTHONPATH=/home/parsa/smart_parsa/smart \
CUDA_VISIBLE_DEVICES=0,1,2,3,4 \
PYTHONUNBUFFERED=1 \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/home/parsa/miniconda3/envs/smart/bin/python \
smart/scripts/compare_shift_endpoint_strategies.py \
  --dataset heat_exchanger \
  --data-root /mnt/ssdraid/parsa/toy_heat_exchange_fem_v1 \
  --study-summary /mnt/ssdraid/parsa/toy_heat_exchange_surface_vtp_remesh_v4/remeshing_v2_summary.json \
  --case-selection test \
  --selection-policy all_cases \
  --include-original \
  --num-runs 0 \
  --seed 42 \
  --views-per-test 10 \
  --active-geometry-sources feature,quadric \
  --geometry-decimation-factors 5,10 \
  --geometry-label-preset v4 \
  --surface-query-points 32768 \
  --volume-query-points 32768 \
  --query-chunk-size 32768 \
  --plot-scales linear,log \
  --font-scale 1.2 \
  --min-free-gib 8 \
  --devices cuda:0,cuda:1,cuda:2,cuda:3,cuda:4 \
  --base-config toy_heat_exchange \
  --satloss-config toy_heat_exchange_satloss7 \
  --downsample-config toy_heat_exchange_downsample \
  --gaussian-ball-masked-config toy_heat_exchange_gaussian_ball_masked \
  --box-masked-config toy_heat_exchange_box_masked \
  --base-checkpoint checkpoints/smart-toy-heat-exchange-heat-exchange-base-ratio-aligned-toyheatexchange-s42_best.pt \
  --satloss-checkpoint checkpoints/smart-toy-heat-exchange-satloss7-heat-exchange-satloss-ratio-aligned-toyheatexchange-s42_best.pt \
  --downsample-checkpoint checkpoints/smart-heat-exchange-downsample-heat-exchange-downsample-200ep-toyheatexchange-s42_best.pt \
  --gaussian-ball-masked-checkpoint checkpoints/smart-heat-exchange-gaussian-ball-masked-heat-exchange-gaussian-ball-masked-200ep-toyheatexchange-s42_best.pt \
  --box-masked-checkpoint checkpoints/smart-heat-exchange-box-masked-heat-exchange-box-masked-200ep-toyheatexchange-s42_best.pt \
  --output-dir results/final/reviewer_evidence_20260901/heat_exchanger_frozen_validation_all_views10
```

## 4. Paired Bootstrap Confidence Intervals

Run these after the three evaluations.  They average stochastic views within a
case and use a paired case-clustered bootstrap, so the result is valid for
case-level uncertainty rather than treating ten views as ten independent cases.

```bash
cd /home/parsa/smart_parsa

PYTHONPATH=/home/parsa/smart_parsa/smart /home/parsa/miniconda3/envs/smart/bin/python smart/scripts/bootstrap_sampling_evaluation_ci.py \
  --input results/final/reviewer_evidence_20260901/drivaerml_frozen_test50_views10/per_view_metrics.csv \
  --output results/final/reviewer_evidence_20260901/drivaerml_frozen_test50_views10/paired_bootstrap_smart_deal.csv \
  --reference-model SMART --comparison-model SMART_SATLOSS7 \
  --conditions original_uniform,sine_x_1,sine_y_1,remeshing_div5_mean,remeshing_div10_mean \
  --bootstrap-samples 10000 --seed 42

PYTHONPATH=/home/parsa/smart_parsa/smart /home/parsa/miniconda3/envs/smart/bin/python smart/scripts/bootstrap_sampling_evaluation_ci.py \
  --input results/final/reviewer_evidence_20260901/pump_frozen_test_all_views10/combined_global_endpoint_metrics.csv \
  --output results/final/reviewer_evidence_20260901/pump_frozen_test_all_views10/paired_bootstrap_smart_deal.csv \
  --reference-model base --comparison-model satloss \
  --conditions original_uniform,sine_x_1,sine_y_1,remeshing_div5_mean,remeshing_div10_mean \
  --bootstrap-samples 10000 --seed 42

PYTHONPATH=/home/parsa/smart_parsa/smart /home/parsa/miniconda3/envs/smart/bin/python smart/scripts/bootstrap_sampling_evaluation_ci.py \
  --input results/final/reviewer_evidence_20260901/heat_exchanger_frozen_validation_all_views10/combined_global_endpoint_metrics.csv \
  --output results/final/reviewer_evidence_20260901/heat_exchanger_frozen_validation_all_views10/paired_bootstrap_smart_deal.csv \
  --reference-model base --comparison-model satloss \
  --conditions original_uniform,sine_x_1,sine_y_1,remeshing_div5_mean,remeshing_div10_mean \
  --bootstrap-samples 10000 --seed 42
```

This addresses evaluation uncertainty.  It does not create uncertainty across
training seeds.  The local DrivAerML config currently contains sine training,
whereas the final beta-only weights are in the other repository; do not launch
new seed training from this local config.  Once that exact beta-only config is
available here, train two additional matched Base/DeAL seed pairs and evaluate
each with the same commands above.

## 5. Geometric Preservation of the Remeshes

These read the original and remeshed VTPs only.  Each command validates a
reproducibly chosen 100-case sample and reports symmetric point-to-triangle
surface distance, sampled normal deviation, area change, topology diagnostics,
and achieved triangle reduction.

```bash
cd /home/parsa/smart_parsa

PYTHONPATH=/home/parsa/smart_parsa/smart /home/parsa/miniconda3/envs/smart/bin/python smart/scripts/validate_remesh_geometry.py \
  --dataset drivaerml --source-dir /mnt/ssdraid/parsa/drivaerml_surface_vtp \
  --remesh-dir /mnt/ssdraid/parsa/drivaerml_surface_vtp_remesh_v4 \
  --output-dir results/final/reviewer_evidence_20260901/drivaerml_remesh_geometry \
  --methods voxel,quadric,feature --factors 5,10 --max-cases 100 \
  --distance-samples 50000 --normal-samples 5000 --workers 8 --seed 42

PYTHONPATH=/home/parsa/smart_parsa/smart /home/parsa/miniconda3/envs/smart/bin/python smart/scripts/validate_remesh_geometry.py \
  --dataset pump --source-dir /mnt/data/parsa/shift_pump_raw_random1400 \
  --remesh-dir /mnt/data/parsa/shift_pump_random1400_surface_vtp_remesh_v4 \
  --output-dir results/final/reviewer_evidence_20260901/pump_remesh_geometry \
  --methods voxel,quadric,feature --factors 5,10 --max-cases 100 \
  --distance-samples 50000 --normal-samples 5000 --workers 8 --seed 42

PYTHONPATH=/home/parsa/smart_parsa/smart /home/parsa/miniconda3/envs/smart/bin/python smart/scripts/validate_remesh_geometry.py \
  --dataset heat_exchanger --source-dir /mnt/ssdraid/parsa/toy_heat_exchange_surface_vtp \
  --remesh-dir /mnt/ssdraid/parsa/toy_heat_exchange_surface_vtp_remesh_v4 \
  --output-dir results/final/reviewer_evidence_20260901/heat_exchanger_remesh_geometry \
  --methods voxel,quadric,feature --factors 5,10 --max-cases 100 \
  --distance-samples 50000 --normal-samples 5000 --workers 8 --seed 42
```

## 6. Pump/Heat Exchanger Documentation and Reproducibility Archive

These are read-only provenance/data-card exports, not training or inference.
The Heat Exchanger audit uses every persisted case and reports actual solver
residuals, nonlinear convergence, exact channel-wall error, physical boundary
conditions, and mesh statistics.

```bash
cd /home/parsa/smart_parsa

PYTHONPATH=/home/parsa/smart_parsa/smart /home/parsa/miniconda3/envs/smart/bin/python smart/scripts/audit_toy_heat_exchange_generation.py \
  --data-root /mnt/ssdraid/parsa/toy_heat_exchange_fem_v1 \
  --output-dir results/final/reviewer_evidence_20260901/heat_exchanger_generation_audit

PYTHONPATH=/home/parsa/smart_parsa/smart /home/parsa/miniconda3/envs/smart/bin/python smart/scripts/export_paper_task_cards.py \
  --output-json results/final/reviewer_evidence_20260901/task_cards.json \
  --output-markdown results/final/reviewer_evidence_20260901/task_cards.md \
  --output-latex results/final/reviewer_evidence_20260901/task_cards.tex

PYTHONPATH=/home/parsa/smart_parsa/smart /home/parsa/miniconda3/envs/smart/bin/python smart/scripts/export_paper_reproducibility_snapshot.py \
  --output-dir results/final/reviewer_evidence_20260901/reproducibility \
  --latex-output results/final/reviewer_evidence_20260901/reproducibility/reproducibility_snapshot.tex \
  --config pump_base=pump \
  --config pump_deal=pump_deal_from_smart_full \
  --config heat_base=toy_heat_exchange \
  --config heat_deal=toy_heat_exchange_satloss7 \
  --checkpoint pump_base=checkpoints/smart-pump-random1400-base-16k-pump-s42_best.pt \
  --checkpoint pump_deal=checkpoints/smart-pump-deal-random1400-from-smart-150ep-pump-s42_best.pt \
  --checkpoint heat_base=checkpoints/smart-toy-heat-exchange-heat-exchange-base-ratio-aligned-toyheatexchange-s42_best.pt \
  --checkpoint heat_deal=checkpoints/smart-toy-heat-exchange-satloss7-heat-exchange-satloss-ratio-aligned-toyheatexchange-s42_best.pt
```

## Non-Experimental Items

Point 4 is handled by the DrivAerML frozen evaluation above.  It becomes the
only source for Base/DeAL values that share a condition.  Do not select the
lowest value across old studies: that would be cherry-picking.  Historical,
range, KDE, and consistency ablations retain their own checkpoints/cohorts and
must be labelled as separate studies.

Point 7 needs citations, not another run.  The relevant existing references
are PointNet++ for random input dropout/downsampling (`qi2017pointnetplusplus`),
Point-MAE for masked point-cloud patches (`pang2022pointmae`), and MeshMask for
physics-aware mesh masking (`garnier2025meshmask`).  They should be described as
motivation for controlled baselines, not as methods identical to DeAL.
