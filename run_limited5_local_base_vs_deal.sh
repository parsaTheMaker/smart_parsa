#!/usr/bin/env bash
set -u

ROOT=/home/parsa/smart_parsa
PYTHON=/home/parsa/miniconda3/envs/smart/bin/python
EVALUATOR="$ROOT/smart/scripts/audit_paired_architecture_sampling.py"
OUT="$ROOT/results/final/limited5_base_vs_deal_20260907"
export PYTHONPATH="$ROOT/smart"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

run_pair() {
  local gpu=$1 dataset=$2 model=$3 base_config=$4 deal_config=$5
  local base_checkpoint=$6 deal_checkpoint=$7 data_root=$8 remesh_root=$9
  local methods=${10} case_ids=${11}
  local job_dir="$OUT/$dataset/$model"
  mkdir -p "$job_dir"
  echo "[$(date --iso-8601=seconds)] starting $dataset/$model on physical GPU $gpu" > "$job_dir/run.log"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" "$EVALUATOR" \
    --dataset "$dataset" --model "$model" \
    --base-config "$base_config" --deal-config "$deal_config" \
    --base-checkpoint "$ROOT/$base_checkpoint" --deal-checkpoint "$ROOT/$deal_checkpoint" \
    --data-root "$data_root" --remesh-root "$remesh_root" \
    --methods "$methods" --factors 5,10 --num-cases 5 --case-ids "$case_ids" \
    --seed 42 --device cuda:0 --output-dir "$job_dir" >> "$job_dir/run.log" 2>&1
  local status=$?
  printf '%s\n' "$status" > "$job_dir/exit_code"
  echo "[$(date --iso-8601=seconds)] finished $dataset/$model status=$status" >> "$job_dir/run.log"
}

# Selected at launch by descending free VRAM: physical GPUs 3, 5, 6, and 4.
queue_gpu3() {
  run_pair 3 drivaerml ab_upt drivaerml_ab_upt drivaerml_ab_upt_deal_from_base \
    checkpoints/ab-upt-expanded-v3-drivaerml-s42_best.pt \
    checkpoints/ab-upt-deal-from-base-150ep-drivaerml-s42_best.pt \
    /mnt/ssdraid/parsa/drivaerml_preprocessed /mnt/ssdraid/parsa/drivaerml_surface_vtp_remesh_v4 \
    feature,quadric,voxel 1,21,101,108,116
  run_pair 3 c_core ab_upt c_core_magnetic_ab_upt c_core_magnetic_ab_upt_deal_from_base \
    checkpoints/ab-upt-c-core-magnetic-ccoremagnetic-s42_best.pt \
    checkpoints/ab-upt-c-core-magnetic-deal-from-base-150ep-ccoremagnetic-s42_best.pt \
    /mnt/data/parsa/c_core_magnetic_fem_v1 /mnt/data/parsa/c_core_magnetic_surface_vtp_remesh_v4 \
    feature,quadric,voxel 256,257,258,259,260
  run_pair 3 c_core pointnet2_ssg c_core_magnetic_pointnet2_ssg c_core_magnetic_pointnet2_ssg_deal_from_base \
    checkpoints/pointnet2-ssg-c-core-magnetic-ccoremagnetic-s42_best.pt \
    checkpoints/pointnet2-ssg-c-core-magnetic-deal-from-base-150ep-ccoremagnetic-s42_best.pt \
    /mnt/data/parsa/c_core_magnetic_fem_v1 /mnt/data/parsa/c_core_magnetic_surface_vtp_remesh_v4 \
    feature,quadric,voxel 256,257,258,259,260
}

queue_gpu5() {
  run_pair 5 drivaerml geo_fno drivaerml_geo_fno drivaerml_geo_fno_deal_from_base \
    checkpoints/geofno-medium-v2-raw65k-drivaerml-s42_best.pt \
    checkpoints/geofno-deal-medium-v2-from-base-150ep-drivaerml-s42_best.pt \
    /mnt/ssdraid/parsa/drivaerml_preprocessed /mnt/ssdraid/parsa/drivaerml_surface_vtp_remesh_v4 \
    feature,quadric,voxel 1,21,101,108,116
  run_pair 5 c_core geo_fno c_core_magnetic_geo_fno c_core_magnetic_geo_fno_deal_from_base \
    checkpoints/geofno-c-core-magnetic-ccoremagnetic-s42_best.pt \
    checkpoints/geofno-c-core-magnetic-deal-from-base-150ep-ccoremagnetic-s42_best.pt \
    /mnt/data/parsa/c_core_magnetic_fem_v1 /mnt/data/parsa/c_core_magnetic_surface_vtp_remesh_v4 \
    feature,quadric,voxel 256,257,258,259,260
}

queue_gpu6() {
  run_pair 6 heat_exchanger ab_upt toy_heat_exchange_ab_upt toy_heat_exchange_ab_upt_deal_from_base \
    checkpoints/ab-upt-toy-heat-exchange-expanded-v3-toyheatexchange-s42_best.pt \
    checkpoints/ab-upt-heat-exchange-deal-from-base-150ep-toyheatexchange-s42_best.pt \
    /mnt/ssdraid/parsa/toy_heat_exchange_fem_v1 /mnt/ssdraid/parsa/toy_heat_exchange_surface_vtp_remesh_v4 \
    feature,quadric 256,257,258,259,260
  run_pair 6 c_core lno c_core_magnetic_lno c_core_magnetic_lno_deal_from_base \
    checkpoints/lno-c-core-magnetic-ccoremagnetic-s42_best.pt \
    checkpoints/lno-c-core-magnetic-deal-from-base-150ep-ccoremagnetic-s42_best.pt \
    /mnt/data/parsa/c_core_magnetic_fem_v1 /mnt/data/parsa/c_core_magnetic_surface_vtp_remesh_v4 \
    feature,quadric,voxel 256,257,258,259,260
}

queue_gpu4() {
  run_pair 4 heat_exchanger geo_fno toy_heat_exchange_geo_fno toy_heat_exchange_geo_fno_deal_from_base \
    checkpoints/geofno-heat-exchanger-medium-v2-raw65k-toyheatexchange-s42_best.pt \
    checkpoints/geofno-heat-exchanger-deal-medium-v2-from-base-150ep-toyheatexchange-s42_best.pt \
    /mnt/ssdraid/parsa/toy_heat_exchange_fem_v1 /mnt/ssdraid/parsa/toy_heat_exchange_surface_vtp_remesh_v4 \
    feature,quadric 256,257,258,259,260
  run_pair 4 c_core mspt c_core_magnetic_mspt c_core_magnetic_mspt_deal_from_base \
    checkpoints/mspt-c-core-magnetic-ccoremagnetic-s42_best.pt \
    checkpoints/mspt-c-core-magnetic-deal-from-base-150ep-ccoremagnetic-s42_best.pt \
    /mnt/data/parsa/c_core_magnetic_fem_v1 /mnt/data/parsa/c_core_magnetic_surface_vtp_remesh_v4 \
    feature,quadric,voxel 256,257,258,259,260
}

queue_gpu3 & p1=$!
queue_gpu5 & p2=$!
queue_gpu6 & p3=$!
queue_gpu4 & p4=$!
wait "$p1" "$p2" "$p3" "$p4"

