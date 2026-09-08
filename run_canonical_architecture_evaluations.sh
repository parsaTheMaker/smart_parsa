#!/usr/bin/env bash
set -u

ROOT=${ROOT:-/home/parsa/smart_parsa}
PYTHON=${PYTHON:-/home/parsa/miniconda3/envs/smart/bin/python}
EVALUATOR="$ROOT/smart/scripts/audit_paired_architecture_sampling.py"
SCOPE=${1:-selected}
REGISTRY="$ROOT/results/final/deal_canonical_evaluation/cohort_registry_all_architectures_v1.json"
# Archived DrivAerML PTV3 DeAL weights require standard PTv3 runtime sampling.
# Do not replace this with the density-sensitive DeAL config: it loads but is wrong.
DRIVAERML_PTV3_DEAL_CONFIG=drivaerml_point_transformer_v3_satloss7
case "$SCOPE" in
  candidates) OUT="$ROOT/results/final/deal_canonical_evaluation/candidate_architectures" ;;
  selected) OUT="$ROOT/results/final/deal_canonical_evaluation/paired_architectures" ;;
  *) printf 'Usage: %s [candidates|selected]\n' "$0" >&2; exit 2 ;;
esac
export PYTHONPATH="$ROOT/smart"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

run_pair() {
  local gpu=$1 dataset=$2 model=$3 base_config=$4 deal_config=$5
  local base_checkpoint=$6 deal_checkpoint=$7 data_root=$8 remesh_root=$9 methods=${10}
  base_checkpoint="${base_checkpoint%_best.pt}_last.pt"
  deal_checkpoint="${deal_checkpoint%_best.pt}_last.pt"
  local job_dir="$OUT/$dataset/$model"
  local selection_args=()
  if [[ "$SCOPE" == candidates ]]; then
    case "$dataset" in
      drivaerml) selection_args=(--num-cases 50) ;;
      pump) selection_args=(--num-cases 274) ;;
      heat_exchanger) selection_args=(--num-cases 32) ;;
      c_core) selection_args=(--num-cases 31) ;;
    esac
  else
    selection_args=(--cohort-registry "$REGISTRY")
  fi
  local reusable=false
  if [[ -f "$job_dir/exit_code" && "$(<"$job_dir/exit_code")" == 0 \
        && -s "$job_dir/case_condition_metrics.csv" && -s "$job_dir/summary.json" ]]; then
    reusable=true
    if [[ "$SCOPE" == selected ]]; then
      "$PYTHON" - "$job_dir/summary.json" "$REGISTRY" <<'PY' || reusable=false
import hashlib, json, sys
summary = json.load(open(sys.argv[1], encoding="utf-8"))
registry_path = sys.argv[2]
registry = json.load(open(registry_path, encoding="utf-8"))
provenance = summary.get("cohort_registry") or {}
expected_hash = hashlib.sha256(open(registry_path, "rb").read()).hexdigest()
raise SystemExit(0 if provenance.get("registry_id") == registry.get("registry_id") and provenance.get("sha256") == expected_hash else 1)
PY
    fi
  fi
  if [[ "$reusable" == true ]]; then
    printf '[%s] reusing complete %s/%s candidate=%s\n' \
      "$(date --iso-8601=seconds)" "$dataset" "$model" "$SCOPE"
    return 0
  fi
  for checkpoint in "$base_checkpoint" "$deal_checkpoint"; do
    if [[ ! -f "$ROOT/$checkpoint" ]]; then
      printf 'Missing required last checkpoint: %s\n' "$ROOT/$checkpoint" >&2
      return 1
    fi
  done
  mkdir -p "$job_dir"
  printf '[%s] starting %s/%s on physical GPU %s scope=%s\n' \
    "$(date --iso-8601=seconds)" "$dataset" "$model" "$gpu" "$SCOPE" > "$job_dir/run.log"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" "$EVALUATOR" \
    --dataset "$dataset" --model "$model" \
    --base-config "$base_config" --deal-config "$deal_config" \
    --base-checkpoint "$ROOT/$base_checkpoint" --deal-checkpoint "$ROOT/$deal_checkpoint" \
    --data-root "$data_root" --remesh-root "$remesh_root" --methods "$methods" --factors 5,10 \
    "${selection_args[@]}" --views-per-condition 1 --seed 42 --device cuda:0 \
    --output-dir "$job_dir" >> "$job_dir/run.log" 2>&1
  local status=$?
  printf '%s\n' "$status" > "$job_dir/exit_code"
  printf '[%s] finished %s/%s status=%s\n' "$(date --iso-8601=seconds)" "$dataset" "$model" "$status" >> "$job_dir/run.log"
  return 0
}

driv() {
  local gpu=$1 model=$2 base_config=$3 deal_config=$4 base=$5 deal=$6
  run_pair "$gpu" drivaerml "$model" "$base_config" "$deal_config" "$base" "$deal" \
    /mnt/ssdraid/parsa/drivaerml_preprocessed /mnt/ssdraid/parsa/drivaerml_surface_vtp_remesh_v4 \
    feature,quadric,voxel
}

heat() {
  local gpu=$1 model=$2 base_config=$3 deal_config=$4 base=$5 deal=$6
  run_pair "$gpu" heat_exchanger "$model" "$base_config" "$deal_config" "$base" "$deal" \
    /mnt/ssdraid/parsa/toy_heat_exchange_fem_v1 /mnt/ssdraid/parsa/toy_heat_exchange_surface_vtp_remesh_v4 \
    feature,quadric
}

ccore() {
  local gpu=$1 model=$2 base_config=$3 deal_config=$4 base=$5 deal=$6
  run_pair "$gpu" c_core "$model" "$base_config" "$deal_config" "$base" "$deal" \
    /mnt/data/parsa/c_core_magnetic_fem_v1 /mnt/data/parsa/c_core_magnetic_surface_vtp_remesh_v4 \
    feature,quadric,voxel
}

queue_gpu0() {
  driv 0 smart drivaerml drivaerml_satloss7_range100 \
    checkpoints/smart-smart-drivaerml-131k16kwr-drivaerml-s42_best.pt \
    checkpoints/smart-satloss7-range100-smart-satloss7-range100-from-smart-150ep-drivaerml-s42_best.pt
  driv 0 pointnet2_ssg drivaerml_pointnet2_ssg drivaerml_pointnet2_ssg_satloss7 \
    checkpoints/pointnet2-ssg-pointnet2-ssg-drivaerml-65k-v2-drivaerml-s42_best.pt \
    checkpoints/pointnet2-ssg-satloss7-pointnet2-ssg-satloss7-drivaerml-65k-drivaerml-s42_best.pt
  driv 0 mspt drivaerml_mspt drivaerml_mspt_satloss7 \
    checkpoints/mspt-mspt-drivaerml-uniform-epochseeded-gpu6-200ep-drivaerml-s42_best.pt \
    checkpoints/mspt-satloss7-mspt-satloss7-drivaerml-65k-drivaerml-s42_best.pt
  driv 0 ab_upt drivaerml_ab_upt drivaerml_ab_upt_deal_from_base \
    checkpoints/ab-upt-expanded-v3-drivaerml-s42_best.pt \
    checkpoints/ab-upt-deal-from-base-150ep-drivaerml-s42_best.pt

  heat 0 smart toy_heat_exchange toy_heat_exchange_satloss7 \
    checkpoints/smart-toy-heat-exchange-heat-exchange-base-ratio-aligned-toyheatexchange-s42_best.pt \
    checkpoints/smart-toy-heat-exchange-satloss7-heat-exchange-satloss-ratio-aligned-toyheatexchange-s42_best.pt
  heat 0 pointnet2_ssg toy_heat_exchange_pointnet2_ssg toy_heat_exchange_pointnet2_ssg_satloss7 \
    checkpoints/pointnet2-ssg-toy-heat-exchange-pointnet2-ssg-base-toyheatexchange-s42_best.pt \
    checkpoints/pointnet2-ssg-toy-heat-exchange-satloss7-pointnet2-ssg-satloss7-from-base-toyheatexchange-s42_best.pt
  heat 0 mspt toy_heat_exchange_mspt toy_heat_exchange_mspt_satloss7 \
    checkpoints/mspt-toy-heat-exchange-mspt-base-toyheatexchange-s42_best.pt \
    checkpoints/mspt-toy-heat-exchange-satloss7-mspt-satloss-from-base-toyheatexchange-s42_best.pt
  heat 0 ab_upt toy_heat_exchange_ab_upt toy_heat_exchange_ab_upt_deal_from_base \
    checkpoints/ab-upt-toy-heat-exchange-expanded-v3-toyheatexchange-s42_best.pt \
    checkpoints/ab-upt-heat-exchange-deal-from-base-150ep-toyheatexchange-s42_best.pt

  ccore 0 smart c_core_magnetic_smart c_core_magnetic_smart_deal_from_base \
    checkpoints/smart-c-core-magnetic-ccoremagnetic-s42_best.pt \
    checkpoints/smart-c-core-magnetic-deal-from-base-150ep-ccoremagnetic-s42_best.pt
  ccore 0 pointnet2_ssg c_core_magnetic_pointnet2_ssg c_core_magnetic_pointnet2_ssg_deal_from_base \
    checkpoints/pointnet2-ssg-c-core-magnetic-ccoremagnetic-s42_best.pt \
    checkpoints/pointnet2-ssg-c-core-magnetic-deal-from-base-150ep-ccoremagnetic-s42_best.pt
  ccore 0 mspt c_core_magnetic_mspt c_core_magnetic_mspt_deal_from_base \
    checkpoints/mspt-c-core-magnetic-ccoremagnetic-s42_best.pt \
    checkpoints/mspt-c-core-magnetic-deal-from-base-150ep-ccoremagnetic-s42_best.pt
  ccore 0 ab_upt c_core_magnetic_ab_upt c_core_magnetic_ab_upt_deal_from_base \
    checkpoints/ab-upt-c-core-magnetic-ccoremagnetic-s42_best.pt \
    checkpoints/ab-upt-c-core-magnetic-deal-from-base-150ep-ccoremagnetic-s42_best.pt

}

queue_gpu3() {
  driv 3 lno drivaerml_lno drivaerml_lno_satloss7 \
    checkpoints/lno-lno-drivaerml-65k-drivaerml-s42_best.pt \
    checkpoints/lno-satloss7-lno-satloss7-drivaerml-65k-drivaerml-s42_best.pt
  driv 3 transolverpp drivaerml_transolverpp drivaerml_transolverpp_satloss7 \
    checkpoints/transolverpp-transolverpp-drivaerml-uniform-epochseeded-gpu0-200ep-drivaerml-s42_best.pt \
    checkpoints/transolverpp-satloss7-transolverpp-satloss7-drivaerml-65k-drivaerml-s42_best.pt
  # The archived DeAL checkpoint was trained with the standard PTv3 ordering;
  # tensor shapes alone cannot detect this runtime-configuration distinction.
  driv 3 point_transformer_v3 drivaerml_point_transformer_v3_density_sensitive "$DRIVAERML_PTV3_DEAL_CONFIG" \
    checkpoints/point-transformer-v3-ptv3-density-sensitive-drivaerml-drivaerml-s42_best.pt \
    checkpoints/point-transformer-v3-satloss7-ptv3-satloss7-density-sensitive-drivaerml-131k-drivaerml-s42_best.pt
  driv 3 geo_fno drivaerml_geo_fno drivaerml_geo_fno_deal_from_base \
    checkpoints/geofno-medium-v2-raw65k-drivaerml-s42_best.pt \
    checkpoints/geofno-deal-medium-v2-from-base-150ep-drivaerml-s42_best.pt

  heat 3 lno toy_heat_exchange_lno toy_heat_exchange_lno_satloss7 \
    checkpoints/lno-toy-heat-exchange-lno-base-toyheatexchange-s42_best.pt \
    checkpoints/lno-toy-heat-exchange-satloss7-lno-satloss7-from-base-toyheatexchange-s42_best.pt
  heat 3 transolverpp toy_heat_exchange_transolverpp toy_heat_exchange_transolverpp_satloss7 \
    checkpoints/transolverpp-toy-heat-exchange-transolverpp-base-toyheatexchange-s42_best.pt \
    checkpoints/transolverpp-toy-heat-exchange-satloss7-transolverpp-satloss-from-base-toyheatexchange-s42_best.pt
  heat 3 point_transformer_v3 toy_heat_exchange_point_transformer_v3 toy_heat_exchange_point_transformer_v3_satloss7 \
    checkpoints/point-transformer-v3-toy-heat-exchange-point-transformer-v3-base-toyheatexchange-s42_best.pt \
    checkpoints/point-transformer-v3-toy-heat-exchange-satloss7-point-transformer-v3-satloss7-from-base-toyheatexchange-s42_best.pt
  heat 3 geo_fno toy_heat_exchange_geo_fno toy_heat_exchange_geo_fno_deal_from_base \
    checkpoints/geofno-heat-exchanger-medium-v2-raw65k-toyheatexchange-s42_best.pt \
    checkpoints/geofno-heat-exchanger-deal-medium-v2-from-base-150ep-toyheatexchange-s42_best.pt

  ccore 3 lno c_core_magnetic_lno c_core_magnetic_lno_deal_from_base \
    checkpoints/lno-c-core-magnetic-ccoremagnetic-s42_best.pt \
    checkpoints/lno-c-core-magnetic-deal-from-base-150ep-ccoremagnetic-s42_best.pt
  ccore 3 transolverpp c_core_magnetic_transolverpp c_core_magnetic_transolverpp_deal_from_base \
    checkpoints/transolverpp-c-core-magnetic-ccoremagnetic-s42_best.pt \
    checkpoints/transolverpp-c-core-magnetic-deal-from-base-150ep-ccoremagnetic-s42_best.pt
  ccore 3 point_transformer_v3 c_core_magnetic_point_transformer_v3 c_core_magnetic_point_transformer_v3_deal_from_base \
    checkpoints/point-transformer-v3-c-core-magnetic-ccoremagnetic-s42_best.pt \
    checkpoints/point-transformer-v3-c-core-magnetic-deal-from-base-150ep-ccoremagnetic-s42_best.pt
  ccore 3 geo_fno c_core_magnetic_geo_fno c_core_magnetic_geo_fno_deal_from_base \
    checkpoints/geofno-c-core-magnetic-ccoremagnetic-s42_best.pt \
    checkpoints/geofno-c-core-magnetic-deal-from-base-150ep-ccoremagnetic-s42_best.pt
}

queue_gpu0 & p0=$!
queue_gpu3 & p3=$!
wait "$p0" "$p3"
if [[ "$SCOPE" == selected ]]; then
  "$PYTHON" "$ROOT/smart/scripts/summarize_canonical_architecture_evaluation.py" \
    --registry "$REGISTRY" --allow-incomplete
fi
