#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT=${DATA_ROOT:-/mnt/pool/sqy/Cambridge_stdloc}
BASELINE_ROOT=${BASELINE_ROOT:-/mnt/pool/sqy/stdloc_la_full_runs}
MODEL_ROOT=${MODEL_ROOT:-/mnt/pool/sqy/stdloc_la_v03_multiscene}
SCENES=${SCENES:-ShopFacade KingsCollege OldHospital}
SEEDS=${SEEDS:-2025 2026 2027}
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
  for seed in $SEEDS; do
    echo "[LA-STDLoc v0.3 multiscene] scene=$scene seed=$seed"
    env \
      PYTHON="$PYTHON" \
      DATA_ROOT="$DATA_ROOT" \
      SCENE="$scene" \
      BASELINE_MODEL="$BASELINE_MODEL" \
      MODEL_ROOT="$MODEL_ROOT/${scene}/seed_${seed}" \
      V03_QUERY_SPLIT_SEED="$seed" \
      RUN_SWEEP="$RUN_SWEEP" \
      "$SCRIPT_DIR/run_locaware_v03_shopfacade.sh"
  done
done
