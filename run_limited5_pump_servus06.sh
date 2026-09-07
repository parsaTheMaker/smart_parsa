#!/usr/bin/env bash
set -u

ROOT=/mnt/data5/parsa/smart_parsa
PYTHON=/mnt/data5/parsa/conda_envs/smart-deal/bin/python
EVALUATOR="$ROOT/smart/scripts/audit_paired_architecture_sampling.py"
OUT="$ROOT/results/final/limited5_base_vs_deal_20260907/pump"
DATA=/mnt/data5/parsa/shift_pump_random1400_preprocessed
REMESH=/mnt/data5/parsa/shift_pump_random1400_surface_vtp_remesh_v4
export PYTHONPATH="$ROOT/smart"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

run_pair() {
  local gpu=$1 model=$2 base_config=$3 deal_config=$4 base_checkpoint=$5 deal_checkpoint=$6
  local job_dir="$OUT/$model"
  mkdir -p "$job_dir"
  echo "[$(date --iso-8601=seconds)] starting pump/$model on physical GPU $gpu" > "$job_dir/run.log"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" "$EVALUATOR" \
    --dataset pump --model "$model" --base-config "$base_config" --deal-config "$deal_config" \
    --base-checkpoint "$ROOT/$base_checkpoint" --deal-checkpoint "$ROOT/$deal_checkpoint" \
    --data-root "$DATA" --remesh-root "$REMESH" --methods feature,quadric,voxel --factors 5,10 \
    --num-cases 5 --case-ids 33,62,78,85,136 --seed 42 --device cuda:0 \
    --output-dir "$job_dir" >> "$job_dir/run.log" 2>&1
  local status=$?
  printf '%s\n' "$status" > "$job_dir/exit_code"
  echo "[$(date --iso-8601=seconds)] finished pump/$model status=$status" >> "$job_dir/run.log"
}

# servus06 has four GPUs; ordered by free VRAM at launch: 0, 2, 3, then 1.
queue_gpu0() {
  run_pair 0 ab_upt pump_ab_upt pump_ab_upt_deal_from_base \
    checkpoints/ab-upt-pump-servus06-base-v1-pump-s42_best.pt \
    checkpoints/ab-upt-pump-deal-from-base-150ep-pump-s42_best.pt
  run_pair 0 point_transformer_v3 pump_point_transformer_v3 pump_point_transformer_v3_deal_from_base \
    checkpoints/point-transformer-v3-pump-servus06-base-v1-pump-s42_best.pt \
    checkpoints/point-transformer-v3-pump-deal-from-base-150ep-pump-s42_best.pt
}

queue_gpu2() {
  run_pair 2 geo_fno pump_geo_fno pump_geo_fno_deal_from_base \
    checkpoints/geofno-pump-medium-v2-raw16k-pump-s42_best.pt \
    checkpoints/geofno-pump-deal-medium-v2-from-base-150ep-pump-s42_best.pt
}

queue_gpu3() {
  run_pair 3 lno pump_lno pump_lno_deal_from_base \
    checkpoints/lno-pump-servus06-base-v1-pump-s42_best.pt \
    checkpoints/lno-pump-deal-from-base-150ep-pump-s42_best.pt
}

queue_gpu1() {
  run_pair 1 pointnet2_ssg pump_pointnet2_ssg pump_pointnet2_ssg_deal_from_base \
    checkpoints/pointnet2-ssg-pump-servus06-base-v1-pump-s42_best.pt \
    checkpoints/pointnet2-ssg-pump-deal-from-base-150ep-pump-s42_best.pt
}

queue_gpu0 & p1=$!
queue_gpu2 & p2=$!
queue_gpu3 & p3=$!
queue_gpu1 & p4=$!
wait "$p1" "$p2" "$p3" "$p4"

