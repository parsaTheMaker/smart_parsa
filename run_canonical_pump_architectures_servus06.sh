#!/usr/bin/env bash
set -u

ROOT=/mnt/data5/parsa/smart_parsa
PYTHON=/mnt/data5/parsa/conda_envs/smart-deal/bin/python
EVALUATOR="$ROOT/smart/scripts/audit_paired_architecture_sampling.py"
SCOPE=${1:-selected}
FORCE=${FORCE:-0}
REGISTRY="$ROOT/results/final/deal_canonical_evaluation/cohort_registry_all_architectures_v1.json"
case "$SCOPE" in
  candidates) OUT="$ROOT/results/final/deal_canonical_evaluation/candidate_architectures/pump" ;;
  selected) OUT="$ROOT/results/final/deal_canonical_evaluation/paired_architectures/pump" ;;
  *) printf 'Usage: %s [candidates|selected]\n' "$0" >&2; exit 2 ;;
esac
DATA=/mnt/data5/parsa/shift_pump_random1400_preprocessed
REMESH=/mnt/data5/parsa/shift_pump_random1400_surface_vtp_remesh_v4
export PYTHONPATH="$ROOT/smart"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

run_pair() {
  local gpu=$1 model=$2 base_config=$3 deal_config=$4 base=$5 deal=$6
  base="${base%_best.pt}_last.pt"
  deal="${deal%_best.pt}_last.pt"
  local job_dir="$OUT/$model"
  local selection_args=()
  if [[ "$SCOPE" == candidates ]]; then
    selection_args=(--num-cases 274)
  else
    selection_args=(--cohort-registry "$REGISTRY")
  fi
  local reusable=false
  if [[ "$FORCE" != 1 && -f "$job_dir/exit_code" && "$(<"$job_dir/exit_code")" == 0 \
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
    printf '[%s] reusing complete pump/%s scope=%s\n' \
      "$(date --iso-8601=seconds)" "$model" "$SCOPE"
    return 0
  fi
  for checkpoint in "$base" "$deal"; do
    if [[ ! -f "$ROOT/$checkpoint" ]]; then
      printf 'Missing required last checkpoint: %s\n' "$ROOT/$checkpoint" >&2
      return 1
    fi
  done
  mkdir -p "$job_dir"
  printf '[%s] starting pump/%s on physical GPU %s scope=%s\n' \
    "$(date --iso-8601=seconds)" "$model" "$gpu" "$SCOPE" > "$job_dir/run.log"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" "$EVALUATOR" \
    --dataset pump --model "$model" --base-config "$base_config" --deal-config "$deal_config" \
    --base-checkpoint "$ROOT/$base" --deal-checkpoint "$ROOT/$deal" \
    --data-root "$DATA" --remesh-root "$REMESH" --methods feature,quadric,voxel --factors 5,10 \
    "${selection_args[@]}" --views-per-condition 1 --seed 42 --device cuda:0 \
    --output-dir "$job_dir" >> "$job_dir/run.log" 2>&1
  local status=$?
  printf '%s\n' "$status" > "$job_dir/exit_code"
  printf '[%s] finished pump/%s status=%s\n' "$(date --iso-8601=seconds)" "$model" "$status" >> "$job_dir/run.log"
  return 0
}

TASKS=(
  "ab_upt|pump_ab_upt|pump_ab_upt_deal_from_base|checkpoints/ab-upt-pump-servus06-base-v1-pump-s42_best.pt|checkpoints/ab-upt-pump-deal-from-base-150ep-pump-s42_best.pt"
  "geo_fno|pump_geo_fno|pump_geo_fno_deal_from_base|checkpoints/geofno-pump-medium-v2-raw16k-pump-s42_best.pt|checkpoints/geofno-pump-deal-medium-v2-from-base-150ep-pump-s42_best.pt"
  "pointnet2_ssg|pump_pointnet2_ssg|pump_pointnet2_ssg_deal_from_base|checkpoints/pointnet2-ssg-pump-servus06-base-v1-pump-s42_best.pt|checkpoints/pointnet2-ssg-pump-deal-from-base-150ep-pump-s42_best.pt"
  "lno|pump_lno|pump_lno_deal_from_base|checkpoints/lno-pump-servus06-base-v1-pump-s42_best.pt|checkpoints/lno-pump-deal-from-base-150ep-pump-s42_best.pt"
  "mspt|pump_mspt|pump_mspt_deal_from_base|checkpoints/mspt-pump-servus06-base-v1-pump-s42_best.pt|checkpoints/mspt-pump-deal-from-base-150ep-pump-s42_best.pt"
  "transolverpp|pump_transolverpp|pump_transolverpp_deal_from_base|checkpoints/transolverpp-pump-servus06-base-v1-pump-s42_best.pt|checkpoints/transolverpp-pump-deal-from-base-150ep-pump-s42_best.pt"
  "point_transformer_v3|pump_point_transformer_v3|pump_point_transformer_v3_deal_from_base|checkpoints/point-transformer-v3-pump-servus06-base-v1-pump-s42_best.pt|checkpoints/point-transformer-v3-pump-deal-from-base-150ep-pump-s42_best.pt"
  "smart|pump|pump_deal_from_smart_full|checkpoints/smart-pump-random1400-base-16k-pump-s42_best.pt|checkpoints/smart-pump-deal-random1400-from-smart-150ep-pump-s42_best.pt"
)

QUEUE_FILE=$(mktemp)
LOCK_FILE=$(mktemp)
trap 'rm -f "$QUEUE_FILE" "$LOCK_FILE"' EXIT
printf '%s\n' "${TASKS[@]}" > "$QUEUE_FILE"

worker() {
  local gpu=$1 task model base_config deal_config base deal
  while true; do
    {
      flock -x 9
      if [[ ! -s "$QUEUE_FILE" ]]; then
        return 0
      fi
      IFS= read -r task < "$QUEUE_FILE"
      sed -i '1d' "$QUEUE_FILE"
    } 9>"$LOCK_FILE"
    IFS='|' read -r model base_config deal_config base deal <<< "$task"
    run_pair "$gpu" "$model" "$base_config" "$deal_config" "$base" "$deal"
  done
}

worker 0 & p0=$!
worker 1 & p1=$!
worker 2 & p2=$!
wait "$p0" "$p1" "$p2"
