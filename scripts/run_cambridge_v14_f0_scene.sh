#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <scene> <gpu> <train|eval|all> [test|validation]" >&2
  exit 2
fi

SCENE="$1"
GPU="$2"
MODE="$3"
EVAL_SUBSET="${4:-test}"

case "$SCENE" in
  GreatCourt|KingsCollege|OldHospital|ShopFacade|StMarysChurch) ;;
  *) echo "Unsupported Cambridge scene: $SCENE" >&2; exit 2 ;;
esac
case "$MODE" in
  train|eval|all) ;;
  *) echo "Mode must be train, eval, or all" >&2; exit 2 ;;
esac
case "$EVAL_SUBSET" in
  test|validation) ;;
  *) echo "Evaluation subset must be test or validation" >&2; exit 2 ;;
esac

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXPERIMENT_ROOT="${CAMBRIDGE_STRICT_2DGS_ROOT:-/mnt/pool/sqy/stdloc_lafgs_cambridge_matcha2dgs_strict_20260711}"
MODEL_ROOT="$EXPERIMENT_ROOT/lafgs_from_sfm/$SCENE"
TAG="${CAMBRIDGE_V14_F0_TAG:-dust025_bias075_v14}"
FOLDER="detector_mapfim_fieldonly_F0_2000_${TAG}"

export SHOP_MAPFIM_SCENE="$SCENE"
export SHOP_MAPFIM_MODEL_ROOT="$MODEL_ROOT"
export SHOP_MAPFIM_MODEL_ITERATION=30000
export SHOP_MAPFIM_BASE_DETECTOR_FOLDER=detector_covsoft_fixlineage_30000
export SHOP_MAPFIM_OUTPUT_TAG="$TAG"
export SHOP_MAPFIM_EVAL_FOLDER="$FOLDER"
export SHOP_MAPFIM_CONFIG_ROOT="$EXPERIMENT_ROOT/eval_configs/v14_f0/$SCENE"
export SHOP_MAPFIM_RESULT_PREFIX="strict2dgs-lafgs-v14f0-$SCENE"
export SHOP_MAPFIM_USE_DUSTBIN=1

# Frozen-detector v14 F0: candidate assignment plus cleanliness/bias map risks.
export SHOP_MAPFIM_DUSTBIN_WEIGHT=0.25
export SHOP_MAPFIM_MAP_CLEANLINESS_WEIGHT=0.5
export SHOP_MAPFIM_MAP_BIAS_WEIGHT=0.75

if [[ "$MODE" == "train" || "$MODE" == "all" ]]; then
  "$REPO_ROOT/scripts/run_shopfacade_mapfim_field_ablation.sh" F0 "$GPU" 2000
fi
if [[ "$MODE" == "eval" || "$MODE" == "all" ]]; then
  "$REPO_ROOT/scripts/evaluate_shopfacade_mapfim_field_ablation.sh" \
    F0 "$GPU" 2000 2000 "$EVAL_SUBSET"
fi
