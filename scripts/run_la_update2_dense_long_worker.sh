#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/mnt/pool/sqy/stdloc_la_update2_dense_long_v1}
DATA_ROOT=${DATA_ROOT:-/mnt/pool/sqy/Cambridge_stdloc}
BASELINE_ROOT=${BASELINE_ROOT:-/mnt/pool/sqy/stdloc_la_full_runs}
V03_ROOT=${V03_ROOT:-/mnt/pool/sqy/stdloc_la_v03_full_length}
SCENES=${SCENES:-${1:-ShopFacade}}
TRAIN_SEEDS=${TRAIN_SEEDS:-${SEEDS:-0 1 2}}
QUERY_SPLIT_SEEDS=${QUERY_SPLIT_SEEDS:-${SEEDS:-2025 2026 2027}}
V03_ITERATION=${V03_ITERATION:-30500}
DENSEKL_STEPS=${DENSEKL_STEPS:-500}
DENSEKL_SAVE_STEPS=${DENSEKL_SAVE_STEPS:-100 500}
DENSEKL_EVAL_STEPS=${DENSEKL_EVAL_STEPS:-$DENSEKL_SAVE_STEPS}
DENSEKL_WEIGHT=${DENSEKL_WEIGHT:-0.02}
DENSEKL_TOPK=${DENSEKL_TOPK:-32}
LOC_ANCHORS=${LOC_ANCHORS:-256}
DENSEKL_POSE_GATE=${DENSEKL_POSE_GATE:-1}
DENSEKL_POSE_GATE_MIN_TE=${DENSEKL_POSE_GATE_MIN_TE:-0.0}
DENSEKL_POSE_GATE_MIN_AE=${DENSEKL_POSE_GATE_MIN_AE:-0.0}
DENSEKL_ATTR_COSINE_THRESHOLD=${DENSEKL_ATTR_COSINE_THRESHOLD:--1.0}
DENSEKL_ATTR_ENTROPY_THRESHOLD=${DENSEKL_ATTR_ENTROPY_THRESHOLD:--1.0}
DENSEKL_MIN_POSITIVE_PROB=${DENSEKL_MIN_POSITIVE_PROB:--1.0}
DENSEKL_MAX_REPROJ_ERROR=${DENSEKL_MAX_REPROJ_ERROR:--1.0}
DENSEKL_MIN_ELIGIBLE_ANCHORS=${DENSEKL_MIN_ELIGIBLE_ANCHORS:-1}
RUN_DIAGNOSTICS=${RUN_DIAGNOSTICS:-0}
RUN_EVAL=${RUN_EVAL:-1}
FORCE_DENSEKL_TRAIN=${FORCE_DENSEKL_TRAIN:-0}
REPROJECTION_ERROR=${REPROJECTION_ERROR:-12}
CFG=${CFG:-configs/stdloc_cambridge.yaml}

if [[ -z "${PYTHON:-}" ]]; then
  if [[ -x /root/miniconda3/envs/ulfloc_repro/bin/python ]]; then
    PYTHON=/root/miniconda3/envs/ulfloc_repro/bin/python
  else
    PYTHON=python
  fi
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$ROOT/logs"
CACHE_DIR="$ROOT/cache"
mkdir -p "$LOG_DIR" "$CACHE_DIR"

run_one() {
  local scene=$1
  local train_seed=$2
  local query_split_seed=$3
  local tag="pose_gate_${DENSEKL_STEPS}"
  local source_model="$V03_ROOT/$scene/train_seed_${train_seed}/query_split_${query_split_seed}/${scene}_v03"
  local legacy_source_model="$V03_ROOT/$scene/seed_${query_split_seed}/${scene}_v03"
  local dense_model="$ROOT/models/$tag/$scene/train_seed_${train_seed}/query_split_${query_split_seed}/${scene}_densekl_from_${V03_ITERATION}"
  local dense_pose_cache="$CACHE_DIR/${scene}_train${train_seed}_query${query_split_seed}_dense_pose_cache_${V03_ITERATION}.pt"
  local log="$LOG_DIR/${scene}_train${train_seed}_query${query_split_seed}_${tag}.log"

  if [[ ! -f "$source_model/point_cloud/iteration_${V03_ITERATION}/point_cloud.ply" && -f "$legacy_source_model/point_cloud/iteration_${V03_ITERATION}/point_cloud.ply" ]]; then
    source_model="$legacy_source_model"
  fi

  if [[ ! -f "$source_model/point_cloud/iteration_${V03_ITERATION}/point_cloud.ply" ]]; then
    echo "[LA-update2-dense] skip missing source: $source_model iteration $V03_ITERATION"
    return 0
  fi

  echo "[LA-update2-dense] start scene=$scene train_seed=$train_seed query_split_seed=$query_split_seed steps=$DENSEKL_STEPS tag=$tag gpu=${CUDA_VISIBLE_DEVICES:-all}"
  if env \
      PYTHON="$PYTHON" \
      DATA_ROOT="$DATA_ROOT" \
      BASELINE_MODEL="$BASELINE_ROOT/${scene}_baseline" \
      SOURCE_MODEL="$source_model" \
      MODEL_ROOT="$ROOT/run_state" \
      DENSEKL_MODEL="$dense_model" \
      DENSE_POSE_CACHE="$dense_pose_cache" \
      SCENE="$scene" \
      CFG="$CFG" \
      LOAD_ITERATION="$V03_ITERATION" \
      DENSEKL_STEPS="$DENSEKL_STEPS" \
      DENSEKL_SAVE_STEPS="$DENSEKL_SAVE_STEPS" \
      DENSEKL_EVAL_STEPS="$DENSEKL_EVAL_STEPS" \
      DENSEKL_WEIGHT="$DENSEKL_WEIGHT" \
      DENSEKL_TOPK="$DENSEKL_TOPK" \
      LOC_ANCHORS="$LOC_ANCHORS" \
      DENSEKL_POSE_GATE="$DENSEKL_POSE_GATE" \
      DENSEKL_POSE_GATE_MIN_TE="$DENSEKL_POSE_GATE_MIN_TE" \
      DENSEKL_POSE_GATE_MIN_AE="$DENSEKL_POSE_GATE_MIN_AE" \
      DENSEKL_ATTR_COSINE_THRESHOLD="$DENSEKL_ATTR_COSINE_THRESHOLD" \
      DENSEKL_ATTR_ENTROPY_THRESHOLD="$DENSEKL_ATTR_ENTROPY_THRESHOLD" \
      DENSEKL_MIN_POSITIVE_PROB="$DENSEKL_MIN_POSITIVE_PROB" \
      DENSEKL_MAX_REPROJ_ERROR="$DENSEKL_MAX_REPROJ_ERROR" \
      DENSEKL_MIN_ELIGIBLE_ANCHORS="$DENSEKL_MIN_ELIGIBLE_ANCHORS" \
      DENSEKL_TRAIN_SEED="$train_seed" \
      RUN_DIAGNOSTICS="$RUN_DIAGNOSTICS" \
      RUN_EVAL="$RUN_EVAL" \
      FORCE_DENSEKL_TRAIN="$FORCE_DENSEKL_TRAIN" \
      REPROJECTION_ERROR="$REPROJECTION_ERROR" \
      "$SCRIPT_DIR/run_densekl_v03_cambridge.sh" >"$log" 2>&1; then
    echo "[LA-update2-dense] done scene=$scene train_seed=$train_seed query_split_seed=$query_split_seed tag=$tag log=$log"
  else
    echo "[LA-update2-dense] failed scene=$scene train_seed=$train_seed query_split_seed=$query_split_seed tag=$tag log=$log" >&2
    tail -n 80 "$log" >&2 || true
    return 1
  fi
}

for scene in $SCENES; do
  for train_seed in $TRAIN_SEEDS; do
    for query_split_seed in $QUERY_SPLIT_SEEDS; do
      run_one "$scene" "$train_seed" "$query_split_seed"
    done
  done
done
