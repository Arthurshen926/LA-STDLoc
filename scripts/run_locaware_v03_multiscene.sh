#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT=${DATA_ROOT:-/mnt/pool/sqy/Cambridge_stdloc}
BASELINE_ROOT=${BASELINE_ROOT:-/mnt/pool/sqy/stdloc_la_full_runs}
MODEL_ROOT=${MODEL_ROOT:-/mnt/pool/sqy/stdloc_la_v03_multiscene}
SCENES=${SCENES:-ShopFacade KingsCollege OldHospital}
TRAIN_SEEDS=${TRAIN_SEEDS:-0 1 2}
QUERY_SPLIT_SEEDS=${QUERY_SPLIT_SEEDS:-2025 2026 2027}
V03_QUERY_SPLIT_MODE=${V03_QUERY_SPLIT_MODE:-random}
RUN_SWEEP=${RUN_SWEEP:-1}

if [[ -z "${PYTHON:-}" ]]; then
  if [[ -x /root/miniconda3/envs/ulfloc_repro/bin/python ]]; then
    PYTHON=/root/miniconda3/envs/ulfloc_repro/bin/python
  else
    PYTHON=python
  fi
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for scene in $SCENES; do
  BASELINE_MODEL="$BASELINE_ROOT/${scene}_baseline"
  if [[ ! -d "$BASELINE_MODEL" ]]; then
    echo "[LA-STDLoc v0.3 multiscene] Skip $scene: missing baseline model $BASELINE_MODEL."
    continue
  fi
  if [[ ! -f "$BASELINE_MODEL/detector/30000_detector.pth" || ! -f "$BASELINE_MODEL/detector/sampled_idx.pkl" ]]; then
    echo "[LA-STDLoc v0.3 multiscene] Skip $scene: missing baseline detector artifacts under $BASELINE_MODEL/detector."
    echo "[LA-STDLoc v0.3 multiscene] Run scripts/prepare_cambridge_baseline_artifacts.sh for this scene first."
    continue
  fi
  if [[ ! -d "$DATA_ROOT/$scene" ]]; then
    echo "[LA-STDLoc v0.3 multiscene] Skip $scene: missing data directory $DATA_ROOT/$scene."
    continue
  fi
  for train_seed in $TRAIN_SEEDS; do
    for query_split_seed in $QUERY_SPLIT_SEEDS; do
      echo "[LA-STDLoc v0.3 multiscene] scene=$scene train_seed=$train_seed query_split_seed=$query_split_seed split_mode=$V03_QUERY_SPLIT_MODE"
      env \
        PYTHON="$PYTHON" \
        DATA_ROOT="$DATA_ROOT" \
        SCENE="$scene" \
        BASELINE_MODEL="$BASELINE_MODEL" \
        MODEL_ROOT="$MODEL_ROOT/${scene}/train_seed_${train_seed}/query_split_${query_split_seed}" \
        V03_TRAIN_SEED="$train_seed" \
        V03_QUERY_SPLIT_SEED="$query_split_seed" \
        V03_QUERY_SPLIT_MODE="$V03_QUERY_SPLIT_MODE" \
        RUN_SWEEP="$RUN_SWEEP" \
        "$SCRIPT_DIR/run_locaware_v03_shopfacade.sh"
    done
  done
done
