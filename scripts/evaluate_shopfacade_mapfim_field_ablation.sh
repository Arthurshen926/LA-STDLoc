#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <F0|F1|F2|F3|F4|F5|baseline> <gpu> [train_iterations] [checkpoint] [test|validation]" >&2
  exit 2
fi

MODE="${1^^}"
GPU="$2"
TRAIN_ITERATIONS="${3:-2000}"
CHECKPOINT="${4:-$TRAIN_ITERATIONS}"
EVAL_SUBSET="${5:-test}"
if [[ "$EVAL_SUBSET" != "test" && "$EVAL_SUBSET" != "validation" ]]; then
  echo "Unknown evaluation subset: $EVAL_SUBSET" >&2
  exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/root/miniconda3/envs/cybersim_agent/bin/python}"
MODEL_ROOT="${SHOP_MAPFIM_MODEL_ROOT:-/mnt/pool/sqy/stdloc_lafgs_shop_strict_fresh_20260712/lafgs_from_sfm/ShopFacade}"
MODEL_ITERATION="${SHOP_MAPFIM_MODEL_ITERATION:-30000}"
BASE_DETECTOR_FOLDER="${SHOP_MAPFIM_BASE_DETECTOR_FOLDER:-detector_covsoft_fixlineage_30000}"
OUTPUT_TAG="${SHOP_MAPFIM_OUTPUT_TAG:-holdout_v2}"
SAFE_OUTPUT_TAG="${OUTPUT_TAG//\//_}"
EVAL_FOLDER_OVERRIDE="${SHOP_MAPFIM_EVAL_FOLDER:-}"
CONFIG_ROOT="${SHOP_MAPFIM_CONFIG_ROOT:-/mnt/pool/sqy/stdloc_lafgs_shop_strict_fresh_20260712/eval_configs/mapfim/ShopFacade}"
MATERIALIZED="${SHOP_MAPFIM_EMBEDDINGS_MATERIALIZED:-0}"
USE_DUSTBIN="${SHOP_MAPFIM_USE_DUSTBIN:-0}"
USE_DETECTOR_MATCHABILITY="${SHOP_MAPFIM_USE_DETECTOR_MATCHABILITY:-0}"
DETECTOR_MATCHABILITY_MODE="${SHOP_MAPFIM_DETECTOR_MATCHABILITY_MODE:-combined_nms}"
USE_DETECTOR_OFFSET="${SHOP_MAPFIM_USE_DETECTOR_OFFSET:-0}"
DUMP_CORRESPONDENCES="${SHOP_MAPFIM_DUMP_CORRESPONDENCES:-0}"
DUMP_ALL_CORRESPONDENCES="${SHOP_MAPFIM_DUMP_ALL_CORRESPONDENCES:-0}"
VALIDATION_RATIO="${SHOP_MAPFIM_VALIDATION_RATIO:-0.2}"
SPLIT_MODE="${SHOP_MAPFIM_SPLIT_MODE:-temporal_block}"
SPLIT_SEED="${SHOP_MAPFIM_SPLIT_SEED:-2026}"

case "$MODE" in
  BASELINE)
    DETECTOR_FOLDER="$BASE_DETECTOR_FOLDER"
    DETECTOR_ITERATION=30000
    STATE_PATH=""
    PREFIX="mapfim-baseline-ShopFacade"
    ;;
  F0|F1|F2|F3|F4|F5)
    DETECTOR_FOLDER="detector_mapfim_fieldonly_${MODE}_${TRAIN_ITERATIONS}_${OUTPUT_TAG}"
    DETECTOR_ITERATION="$CHECKPOINT"
    STATE_PATH="$DETECTOR_FOLDER/${CHECKPOINT}_candidate_teacher_state.pt"
    PREFIX="mapfim-${MODE}-${OUTPUT_TAG}-${TRAIN_ITERATIONS}-ckpt${CHECKPOINT}-ShopFacade"
    ;;
  *)
    echo "Unknown mode: $MODE" >&2
    exit 2
    ;;
esac

if [[ -n "$EVAL_FOLDER_OVERRIDE" ]]; then
  if [[ "$MODE" == "BASELINE" ]]; then
    echo "SHOP_MAPFIM_EVAL_FOLDER is only valid for candidate-field modes" >&2
    exit 2
  fi
  DETECTOR_FOLDER="$EVAL_FOLDER_OVERRIDE"
  STATE_PATH="$DETECTOR_FOLDER/${CHECKPOINT}_candidate_teacher_state.pt"
fi

if [[ "$USE_DUSTBIN" == "1" ]]; then
  SAFE_OUTPUT_TAG="${SAFE_OUTPUT_TAG}_dustbin"
fi
if [[ "$USE_DETECTOR_MATCHABILITY" == "1" ]]; then
  SAFE_OUTPUT_TAG="${SAFE_OUTPUT_TAG}_matchability_${DETECTOR_MATCHABILITY_MODE}"
elif [[ "$USE_DETECTOR_MATCHABILITY" != "0" ]]; then
  echo "SHOP_MAPFIM_USE_DETECTOR_MATCHABILITY must be 0 or 1, got: $USE_DETECTOR_MATCHABILITY" >&2
  exit 2
fi
if [[ "$USE_DETECTOR_OFFSET" == "1" ]]; then
  SAFE_OUTPUT_TAG="${SAFE_OUTPUT_TAG}_offset"
elif [[ "$USE_DETECTOR_OFFSET" != "0" ]]; then
  echo "SHOP_MAPFIM_USE_DETECTOR_OFFSET must be 0 or 1, got: $USE_DETECTOR_OFFSET" >&2
  exit 2
fi
if [[ "$DUMP_CORRESPONDENCES" == "1" ]]; then
  SAFE_OUTPUT_TAG="${SAFE_OUTPUT_TAG}_corrdump"
elif [[ "$DUMP_CORRESPONDENCES" != "0" ]]; then
  echo "SHOP_MAPFIM_DUMP_CORRESPONDENCES must be 0 or 1, got: $DUMP_CORRESPONDENCES" >&2
  exit 2
fi
if [[ "$DUMP_ALL_CORRESPONDENCES" == "1" ]]; then
  if [[ "$DUMP_CORRESPONDENCES" != "1" ]]; then
    echo "SHOP_MAPFIM_DUMP_ALL_CORRESPONDENCES requires SHOP_MAPFIM_DUMP_CORRESPONDENCES=1" >&2
    exit 2
  fi
  SAFE_OUTPUT_TAG="${SAFE_OUTPUT_TAG}_all"
elif [[ "$DUMP_ALL_CORRESPONDENCES" != "0" ]]; then
  echo "SHOP_MAPFIM_DUMP_ALL_CORRESPONDENCES must be 0 or 1, got: $DUMP_ALL_CORRESPONDENCES" >&2
  exit 2
fi

if [[ ! -f "$MODEL_ROOT/$DETECTOR_FOLDER/${DETECTOR_ITERATION}_detector.pth" ]]; then
  echo "Missing detector checkpoint: $MODEL_ROOT/$DETECTOR_FOLDER/${DETECTOR_ITERATION}_detector.pth" >&2
  exit 1
fi
if [[ -n "$STATE_PATH" && ! -f "$MODEL_ROOT/$STATE_PATH" ]]; then
  echo "Missing feature checkpoint: $MODEL_ROOT/$STATE_PATH" >&2
  exit 1
fi

mkdir -p "$CONFIG_ROOT"
CFG_PATH="$CONFIG_ROOT/${MODE,,}_${SAFE_OUTPUT_TAG}_${TRAIN_ITERATIONS}_ckpt${CHECKPOINT}_${EVAL_SUBSET}.yaml"
CFG_SUMMARY="$CONFIG_ROOT/${MODE,,}_${SAFE_OUTPUT_TAG}_${TRAIN_ITERATIONS}_ckpt${CHECKPOINT}_${EVAL_SUBSET}.json"
EXTRA_ARGS=()
if [[ -n "$STATE_PATH" ]]; then
  EXTRA_ARGS+=(--candidate_teacher_state_path "$STATE_PATH")
  if [[ "$MATERIALIZED" == "1" ]]; then
    PREFIX="${PREFIX}-materialized"
  else
    EXTRA_ARGS+=(
      --landmark_feature_override_path "$STATE_PATH"
      --override_landmark_features
    )
  fi
fi
if [[ "$USE_DUSTBIN" == "1" ]]; then
  if [[ -z "$STATE_PATH" ]]; then
    echo "Candidate dustbin evaluation requires a candidate teacher state" >&2
    exit 2
  fi
  EXTRA_ARGS+=(--pair_scorer_state_path "$STATE_PATH")
  EXTRA_ARGS+=(--use_candidate_dustbin)
  PREFIX="${PREFIX}-dustbin"
fi
if [[ "$USE_DETECTOR_MATCHABILITY" == "1" ]]; then
  EXTRA_ARGS+=(
    --use_detector_matchability
    --detector_matchability_mode "$DETECTOR_MATCHABILITY_MODE"
  )
  PREFIX="${PREFIX}-matchability-${DETECTOR_MATCHABILITY_MODE}"
fi
if [[ "$USE_DETECTOR_OFFSET" == "1" ]]; then
  EXTRA_ARGS+=(--use_detector_offset)
  PREFIX="${PREFIX}-offset"
fi
if [[ "$DUMP_CORRESPONDENCES" == "1" ]]; then
  EXTRA_ARGS+=(--diagnostics_dump_correspondences)
  PREFIX="${PREFIX}-corrdump"
fi
if [[ "$DUMP_ALL_CORRESPONDENCES" == "1" ]]; then
  EXTRA_ARGS+=(--diagnostics_dump_all)
  PREFIX="${PREFIX}-all"
fi
SUBSET_ARGS=()
if [[ "$EVAL_SUBSET" == "validation" ]]; then
  SUBSET_ARGS=(
    --evaluation_camera_subset candidate_validation
    --candidate_validation_ratio "$VALIDATION_RATIO"
    --candidate_split_mode "$SPLIT_MODE"
    --candidate_split_seed "$SPLIT_SEED"
    --candidate_direct_validation_holdout
  )
fi

export CUDA_VISIBLE_DEVICES="$GPU"
export CUDA_HOME=/usr/local/cuda-11.8
export PATH="/root/miniconda3/envs/cybersim_agent/bin:/usr/local/cuda-11.8/bin:$PATH"
export LD_LIBRARY_PATH="/root/miniconda3/envs/cybersim_agent/lib:/usr/local/cuda-11.8/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$REPO_ROOT"
export PYTHONHASHSEED=0

cd "$REPO_ROOT"
"$PYTHON" scripts/make_stdloc_eval_cfg.py \
  --base_cfg configs/stdloc_cambridge.yaml \
  --output "$CFG_PATH" \
  --artifact_model_path "$MODEL_ROOT" \
  --detector_folder "$DETECTOR_FOLDER" \
  --detector_iters "$DETECTOR_ITERATION" \
  --detect_num 4096 \
  --reprojection_error 12 \
  --nms 2 \
  --match_threshold 0 \
  --match_topk 1 \
  --max_matches_per_landmark 2 \
  --candidate_frontend_match_policy error \
  --summary_json "$CFG_SUMMARY" \
  "${EXTRA_ARGS[@]}"

"$PYTHON" stdloc.py \
  --model_path "$MODEL_ROOT" \
  --iteration "$MODEL_ITERATION" \
  --cfg "$CFG_PATH" \
  --prefix "${PREFIX}-${EVAL_SUBSET}" \
  --sparse_only \
  "${SUBSET_ARGS[@]}"
