#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/mnt/pool/sqy/stdloc_la_update2_long_closure_v2}
DATA_ROOT=${DATA_ROOT:-/mnt/pool/sqy/Cambridge_stdloc}
BASELINE_ROOT=${BASELINE_ROOT:-/mnt/pool/sqy/stdloc_la_full_runs}
V03_ROOT=${V03_ROOT:-/mnt/pool/sqy/stdloc_la_v03_full_length}
SCENES=${SCENES:-${1:-ShopFacade}}
TRAIN_SEEDS=${TRAIN_SEEDS:-${SEEDS:-0 1 2}}
QUERY_SPLIT_SEEDS=${QUERY_SPLIT_SEEDS:-${SEEDS:-2025 2026 2027}}
CORE_MODES=${CORE_MODES:-no_mutation split_only}
CORE_STEPS=${CORE_STEPS:-100 500}
PRUNE_TRAIN_SEEDS=${PRUNE_TRAIN_SEEDS:-${PRUNE_SEEDS:-$TRAIN_SEEDS}}
PRUNE_QUERY_SPLIT_SEEDS=${PRUNE_QUERY_SPLIT_SEEDS:-${PRUNE_SEEDS:-$QUERY_SPLIT_SEEDS}}
PRUNE_STEPS=${PRUNE_STEPS:-100}
PRUNE_SWEEP=${PRUNE_SWEEP:-mild:0.003:0.20:0.10 balanced:0.004:0.20:0.10 active:0.005:0.20:0.10}
RUN_CORE=${RUN_CORE:-1}
RUN_PRUNE_SWEEP=${RUN_PRUNE_SWEEP:-1}
V03_ITERATION=${V03_ITERATION:-30500}
LABEL_MAX_IMAGES=${LABEL_MAX_IMAGES:-64}
TOPOLOGY_UPDATE_INTERVAL=${TOPOLOGY_UPDATE_INTERVAL:-25}
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
  local mode=$4
  local steps=$5
  local tag=$6
  shift 6

  local source_model="$V03_ROOT/$scene/train_seed_${train_seed}/query_split_${query_split_seed}/${scene}_v03"
  local legacy_source_model="$V03_ROOT/$scene/seed_${query_split_seed}/${scene}_v03"
  local topology_model="$ROOT/models/${tag}/${scene}/train_seed_${train_seed}/query_split_${query_split_seed}/${scene}_v03_topology_from_${V03_ITERATION}"
  local log="$LOG_DIR/${scene}_train${train_seed}_query${query_split_seed}_${tag}.log"

  if [[ ! -f "$source_model/point_cloud/iteration_${V03_ITERATION}/point_cloud.ply" && -f "$legacy_source_model/point_cloud/iteration_${V03_ITERATION}/point_cloud.ply" ]]; then
    source_model="$legacy_source_model"
  fi

  if [[ ! -f "$source_model/point_cloud/iteration_${V03_ITERATION}/point_cloud.ply" ]]; then
    echo "[LA-update2-long] skip missing source: $source_model iteration $V03_ITERATION"
    return 0
  fi

  echo "[LA-update2-long] start scene=$scene train_seed=$train_seed query_split_seed=$query_split_seed mode=$mode steps=$steps tag=$tag gpu=${CUDA_VISIBLE_DEVICES:-all}"
  if env \
      PYTHON="$PYTHON" \
      DATA_ROOT="$DATA_ROOT" \
      BASELINE_ROOT="$BASELINE_ROOT" \
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
      LABEL_MAX_IMAGES="$LABEL_MAX_IMAGES" \
      FORCE_TOPOLOGY_COPY=0 \
      FORCE_TOPOLOGY_TRAIN=1 \
      FORCE_LABEL_STATE=0 \
      "$@" \
      "$SCRIPT_DIR/run_locaware_v03_topology_full.sh" >"$log" 2>&1; then
    echo "[LA-update2-long] done scene=$scene train_seed=$train_seed query_split_seed=$query_split_seed tag=$tag log=$log"
  else
    echo "[LA-update2-long] failed scene=$scene train_seed=$train_seed query_split_seed=$query_split_seed tag=$tag log=$log" >&2
    tail -n 80 "$log" >&2 || true
    return 1
  fi
}

if [[ "$RUN_CORE" == "1" ]]; then
  for scene in $SCENES; do
    for train_seed in $TRAIN_SEEDS; do
      for query_split_seed in $QUERY_SPLIT_SEEDS; do
        for steps in $CORE_STEPS; do
          for mode in $CORE_MODES; do
            tag="core_${mode}_${steps}"
            run_one "$scene" "$train_seed" "$query_split_seed" "$mode" "$steps" "$tag"
          done
        done
      done
    done
  done
fi

if [[ "$RUN_PRUNE_SWEEP" == "1" ]]; then
  for scene in $SCENES; do
    for train_seed in $PRUNE_TRAIN_SEEDS; do
      for query_split_seed in $PRUNE_QUERY_SPLIT_SEEDS; do
        for steps in $PRUNE_STEPS; do
          for spec in $PRUNE_SWEEP; do
            IFS=: read -r name rgb loc utility <<<"$spec"
            tag="prune_${name}_${steps}"
            run_one \
              "$scene" "$train_seed" "$query_split_seed" physical_prune_only "$steps" "$tag" \
              TOPOLOGY_USE_LOC_OPACITY=1 \
              TOPOLOGY_LOC_OPACITY_WEIGHT=0.01 \
              TOPOLOGY_PROTECT_LANDMARKS=1 \
              TOPOLOGY_PHYSICAL_RGB_THRESHOLD="$rgb" \
              TOPOLOGY_PHYSICAL_LOC_THRESHOLD="$loc" \
              TOPOLOGY_PHYSICAL_UTILITY_THRESHOLD="$utility"
          done
        done
      done
    done
  done
fi
