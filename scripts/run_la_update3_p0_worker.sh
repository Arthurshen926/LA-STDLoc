#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/mnt/pool/sqy/stdloc_la_update3_p0_splits_v1}
DATA_ROOT=${DATA_ROOT:-/mnt/pool/sqy/Cambridge_stdloc}
BASELINE_ROOT=${BASELINE_ROOT:-/mnt/pool/sqy/stdloc_la_full_runs}
V03_ROOT=${V03_ROOT:-/mnt/pool/sqy/stdloc_la_v03_full_length}
SCENES=${SCENES:-${1:-ShopFacade}}
TRAIN_SEEDS=${TRAIN_SEEDS:-0}
QUERY_SPLIT_SEEDS=${QUERY_SPLIT_SEEDS:-2025}
STEPS=${STEPS:-500}
SPECS=${SPECS:-S0:no_mutation S1:one_shot_split S2:split_only S3:one_shot_split_freeze}
V03_ITERATION=${V03_ITERATION:-30500}
LABEL_MAX_IMAGES=${LABEL_MAX_IMAGES:-64}
TOPOLOGY_UPDATE_INTERVAL=${TOPOLOGY_UPDATE_INTERVAL:-25}
TOPOLOGY_FULL_BANK_IGNORE_3D_RADIUS=${TOPOLOGY_FULL_BANK_IGNORE_3D_RADIUS:-0.1}
TOPOLOGY_FULL_BANK_IGNORE_UV_RADIUS=${TOPOLOGY_FULL_BANK_IGNORE_UV_RADIUS:-2.0}
TOPOLOGY_FULL_BANK_NEARBY_AS_POSITIVE_UNTIL=${TOPOLOGY_FULL_BANK_NEARBY_AS_POSITIVE_UNTIL:-0}
S3_CHILD_FEATURE_FREEZE_STEPS=${S3_CHILD_FEATURE_FREEZE_STEPS:-${TOPOLOGY_CHILD_FEATURE_FREEZE_STEPS:-100}}
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
mkdir -p "$LOG_DIR"

run_one() {
  local scene=$1
  local train_seed=$2
  local query_split_seed=$3
  local steps=$4
  local label=$5
  local mode=$6

  local source_model="$V03_ROOT/$scene/train_seed_${train_seed}/query_split_${query_split_seed}/${scene}_v03"
  local legacy_source_model="$V03_ROOT/$scene/seed_${query_split_seed}/${scene}_v03"
  local tag="p0_${label}_${steps}"
  local topology_model="$ROOT/models/${tag}/${scene}/train_seed_${train_seed}/query_split_${query_split_seed}/${scene}_v03_topology_from_${V03_ITERATION}"
  local log="$LOG_DIR/${scene}_train${train_seed}_query${query_split_seed}_${tag}.log"
  local freeze_steps=0
  if [[ "$mode" == "one_shot_split_freeze" ]]; then
    freeze_steps="$S3_CHILD_FEATURE_FREEZE_STEPS"
  fi

  if [[ ! -f "$source_model/point_cloud/iteration_${V03_ITERATION}/point_cloud.ply" && -f "$legacy_source_model/point_cloud/iteration_${V03_ITERATION}/point_cloud.ply" ]]; then
    source_model="$legacy_source_model"
  fi

  if [[ ! -f "$source_model/point_cloud/iteration_${V03_ITERATION}/point_cloud.ply" ]]; then
    echo "[LA-update3-P0] skip missing source: $source_model iteration $V03_ITERATION"
    return 0
  fi

  echo "[LA-update3-P0] start scene=$scene train_seed=$train_seed query_split_seed=$query_split_seed label=$label mode=$mode steps=$steps gpu=${CUDA_VISIBLE_DEVICES:-all}"
  if env \
      PYTHON="$PYTHON" \
      DATA_ROOT="$DATA_ROOT" \
      BASELINE_ROOT="$BASELINE_ROOT" \
      V03_ROOT="$V03_ROOT" \
      SCENE="$scene" \
      TRAIN_SEED="$train_seed" \
      QUERY_SPLIT_SEED="$query_split_seed" \
      CFG="$CFG" \
      SOURCE_MODEL="$source_model" \
      TOPOLOGY_MODEL="$topology_model" \
      MODEL_ROOT="$ROOT/run_state" \
      V03_ITERATION="$V03_ITERATION" \
      TOPOLOGY_STEPS="$steps" \
      TOPOLOGY_MUTATION_MODE="$mode" \
      TOPOLOGY_UPDATE_INTERVAL="$TOPOLOGY_UPDATE_INTERVAL" \
      TOPOLOGY_FULL_BANK_IGNORE_3D_RADIUS="$TOPOLOGY_FULL_BANK_IGNORE_3D_RADIUS" \
      TOPOLOGY_FULL_BANK_IGNORE_UV_RADIUS="$TOPOLOGY_FULL_BANK_IGNORE_UV_RADIUS" \
      TOPOLOGY_FULL_BANK_NEARBY_AS_POSITIVE_UNTIL="$TOPOLOGY_FULL_BANK_NEARBY_AS_POSITIVE_UNTIL" \
      TOPOLOGY_CHILD_FEATURE_FREEZE_STEPS="$freeze_steps" \
      LABEL_MAX_IMAGES="$LABEL_MAX_IMAGES" \
      FORCE_TOPOLOGY_COPY=0 \
      FORCE_TOPOLOGY_TRAIN=1 \
      FORCE_LABEL_STATE=0 \
      "$SCRIPT_DIR/run_locaware_v03_topology_full.sh" >"$log" 2>&1; then
    echo "[LA-update3-P0] done scene=$scene train_seed=$train_seed query_split_seed=$query_split_seed tag=$tag log=$log"
  else
    echo "[LA-update3-P0] failed scene=$scene train_seed=$train_seed query_split_seed=$query_split_seed tag=$tag log=$log" >&2
    tail -n 80 "$log" >&2 || true
    return 1
  fi
}

for scene in $SCENES; do
  for train_seed in $TRAIN_SEEDS; do
    for query_split_seed in $QUERY_SPLIT_SEEDS; do
      for steps in $STEPS; do
        for spec in $SPECS; do
          IFS=: read -r label mode <<<"$spec"
          run_one "$scene" "$train_seed" "$query_split_seed" "$steps" "$label" "$mode"
        done
      done
    done
  done
done
