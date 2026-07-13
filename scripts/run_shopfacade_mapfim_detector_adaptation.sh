#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <gpu> [iterations]" >&2
  exit 2
fi

GPU="$1"
ITERATIONS="${2:-2000}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/root/miniconda3/envs/cybersim_agent/bin/python}"
MODEL_ROOT="${SHOP_MAPFIM_MODEL_ROOT:-/mnt/pool/sqy/stdloc_lafgs_shop_strict_fresh_20260712/lafgs_from_sfm/ShopFacade}"
MODEL_ITERATION="${SHOP_MAPFIM_MODEL_ITERATION:-30000}"
FIELD_MODE="${SHOP_MAPFIM_FIELD_MODE:-F0}"
FIELD_ITERATIONS="${SHOP_MAPFIM_FIELD_ITERATIONS:-2000}"
FIELD_TAG="${SHOP_MAPFIM_FIELD_TAG:-all231_dust025_bias075_v15}"
OUTPUT_TAG="${SHOP_MAPFIM_OUTPUT_TAG:-${FIELD_TAG}_predicted_correct}"
TARGET_SOURCE="${SHOP_MAPFIM_DETECTOR_TARGET_SOURCE:-predicted_correct}"
MATCHABILITY_ONLY="${SHOP_MAPFIM_MATCHABILITY_ONLY:-1}"
OFFSET_ONLY="${SHOP_MAPFIM_OFFSET_ONLY:-0}"
OFFSET_TARGET_SOURCE="${SHOP_MAPFIM_OFFSET_TARGET_SOURCE:-matched_top1}"
DETECTOR_MATCH_WEIGHT="${SHOP_MAPFIM_DETECTOR_MATCH_WEIGHT:-1}"
DETECTOR_OFFSET_WEIGHT="${SHOP_MAPFIM_DETECTOR_OFFSET_WEIGHT:-0}"
BASE_DETECTOR_WEIGHT="${SHOP_MAPFIM_BASE_DETECTOR_WEIGHT:-0}"
VALIDATION_RATIO="${SHOP_MAPFIM_VALIDATION_RATIO:-0}"
SPLIT_MODE="${SHOP_MAPFIM_SPLIT_MODE:-temporal_block}"
SPLIT_SEED="${SHOP_MAPFIM_SPLIT_SEED:-2026}"

FIELD_FOLDER="detector_mapfim_fieldonly_${FIELD_MODE}_${FIELD_ITERATIONS}_${FIELD_TAG}"
OUTPUT_FOLDER="detector_mapfim_adapter_${ITERATIONS}_${OUTPUT_TAG}"
FIELD_STATE="$MODEL_ROOT/$FIELD_FOLDER/${FIELD_ITERATIONS}_candidate_teacher_state.pt"
FIELD_DETECTOR="$MODEL_ROOT/$FIELD_FOLDER/${FIELD_ITERATIONS}_detector.pth"
LANDMARK_PATH="$MODEL_ROOT/$FIELD_FOLDER/sampled_idx.pkl"
FINAL_STATE="$MODEL_ROOT/$OUTPUT_FOLDER/${ITERATIONS}_candidate_teacher_state.pt"

for path in "$FIELD_STATE" "$FIELD_DETECTOR" "$LANDMARK_PATH"; do
  if [[ ! -f "$path" ]]; then
    echo "Missing field artifact: $path" >&2
    exit 1
  fi
done
if [[ -f "$FINAL_STATE" ]]; then
  echo "Completed artifact already exists: $FINAL_STATE"
  exit 0
fi

SAVE_ITERATIONS=()
if (( ITERATIONS >= 500 )); then
  SAVE_ITERATIONS+=(500)
fi

MATCHABILITY_ARGS=()
if [[ "$MATCHABILITY_ONLY" == "1" && "$OFFSET_ONLY" == "1" ]]; then
  echo "SHOP_MAPFIM_MATCHABILITY_ONLY and SHOP_MAPFIM_OFFSET_ONLY are mutually exclusive" >&2
  exit 2
fi
if [[ "$MATCHABILITY_ONLY" == "1" ]]; then
  MATCHABILITY_ARGS+=(
    --candidate_teacher_matchability_head
    --candidate_teacher_matchability_only
  )
elif [[ "$MATCHABILITY_ONLY" != "0" ]]; then
  echo "SHOP_MAPFIM_MATCHABILITY_ONLY must be 0 or 1, got: $MATCHABILITY_ONLY" >&2
  exit 2
fi
if [[ "$OFFSET_ONLY" == "1" ]]; then
  MATCHABILITY_ARGS+=(
    --candidate_teacher_offset_head
    --candidate_teacher_offset_only
    --candidate_teacher_offset_target_source "$OFFSET_TARGET_SOURCE"
  )
  DETECTOR_MATCH_WEIGHT="${SHOP_MAPFIM_DETECTOR_MATCH_WEIGHT:-0}"
  DETECTOR_OFFSET_WEIGHT="${SHOP_MAPFIM_DETECTOR_OFFSET_WEIGHT:-1}"
elif [[ "$OFFSET_ONLY" != "0" ]]; then
  echo "SHOP_MAPFIM_OFFSET_ONLY must be 0 or 1, got: $OFFSET_ONLY" >&2
  exit 2
fi
if (( ITERATIONS >= 1000 )); then
  SAVE_ITERATIONS+=(1000)
fi
if (( ${#SAVE_ITERATIONS[@]} == 0 )) || [[ "${SAVE_ITERATIONS[-1]}" != "$ITERATIONS" ]]; then
  SAVE_ITERATIONS+=("$ITERATIONS")
fi

export CUDA_VISIBLE_DEVICES="$GPU"
export CUDA_HOME=/usr/local/cuda-11.8
export PATH="/root/miniconda3/envs/cybersim_agent/bin:/usr/local/cuda-11.8/bin:$PATH"
export LD_LIBRARY_PATH="/root/miniconda3/envs/cybersim_agent/lib:/usr/local/cuda-11.8/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$REPO_ROOT"
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export PYTHONHASHSEED=0

cd "$REPO_ROOT"
"$PYTHON" train_detector.py \
  --model_path "$MODEL_ROOT" \
  --iteration "$MODEL_ITERATION" \
  --iterations "$ITERATIONS" \
  --test_iterations "$((ITERATIONS + 1))" \
  --save_iterations "${SAVE_ITERATIONS[@]}" \
  --detector_folder "$OUTPUT_FOLDER" \
  --landmark_num 16384 \
  --precomputed_landmark_path "$LANDMARK_PATH" \
  --sparse_candidate_teacher \
  --candidate_teacher_detector_init_path "$FIELD_DETECTOR" \
  --candidate_teacher_state_init_path "$FIELD_STATE" \
  --candidate_teacher_detector_lr 0.0001 \
  --candidate_teacher_detect_num 4096 \
  --candidate_teacher_nms_radius 2 \
  --candidate_teacher_match_topk 1 \
  --candidate_teacher_hard_negatives 0 \
  --candidate_teacher_pair_weight 0 \
  --candidate_teacher_hard_negative_weight 0 \
  --candidate_teacher_assignment_weight 0 \
  --candidate_teacher_dustbin_weight 0 \
  --candidate_teacher_matcher_assignment_weight 0 \
  --candidate_teacher_matcher_reprojection_weight 0 \
  --candidate_teacher_pair_scorer_weight 0 \
  --candidate_teacher_pair_scorer_assignment_weight 0 \
  --candidate_teacher_pair_measurement_inlier_weight 0 \
  --candidate_teacher_pair_measurement_nll_weight 0 \
  "${MATCHABILITY_ARGS[@]}" \
  --candidate_teacher_detector_target_source "$TARGET_SOURCE" \
  --candidate_teacher_detector_match_weight "$DETECTOR_MATCH_WEIGHT" \
  --candidate_teacher_detector_offset_weight "$DETECTOR_OFFSET_WEIGHT" \
  --candidate_teacher_geometry_weight 0 \
  --candidate_teacher_coverage_weight 0 \
  --candidate_teacher_base_detector_weight "$BASE_DETECTOR_WEIGHT" \
  --candidate_teacher_feature_anchor_weight 0 \
  --candidate_teacher_validation_ratio "$VALIDATION_RATIO" \
  --candidate_teacher_split_mode "$SPLIT_MODE" \
  --candidate_teacher_split_seed "$SPLIT_SEED"
