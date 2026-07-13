#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <F0|F1|F2|F3|F4|F5> <gpu> [iterations]" >&2
  exit 2
fi

MODE="${1^^}"
GPU="$2"
ITERATIONS="${3:-2000}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/root/miniconda3/envs/cybersim_agent/bin/python}"
MODEL_ROOT="${SHOP_MAPFIM_MODEL_ROOT:-/mnt/pool/sqy/stdloc_lafgs_shop_strict_fresh_20260712/lafgs_from_sfm/ShopFacade}"
MODEL_ITERATION="${SHOP_MAPFIM_MODEL_ITERATION:-30000}"
BASE_DETECTOR_FOLDER="${SHOP_MAPFIM_BASE_DETECTOR_FOLDER:-detector_covsoft_fixlineage_30000}"
OUTPUT_TAG="${SHOP_MAPFIM_OUTPUT_TAG:-holdout_v2}"
OUTPUT_FOLDER="detector_mapfim_fieldonly_${MODE}_${ITERATIONS}_${OUTPUT_TAG}"
VALIDATION_RATIO="${SHOP_MAPFIM_VALIDATION_RATIO:-0.2}"
SPLIT_MODE="${SHOP_MAPFIM_SPLIT_MODE:-temporal_block}"
SPLIT_SEED="${SHOP_MAPFIM_SPLIT_SEED:-2026}"

POSE_MODE="none"
POSE_WEIGHT="0"
NORMALIZATION="quantile"
MATCHABILITY_ARGS=()
case "$MODE" in
  F0)
    ;;
  F1)
    POSE_MODE="point_jacobian"
    POSE_WEIGHT="0.5"
    NORMALIZATION="max"
    ;;
  F2)
    POSE_MODE="full_set_leverage"
    POSE_WEIGHT="0.5"
    ;;
  F3)
    POSE_MODE="conditional_full"
    POSE_WEIGHT="0.5"
    ;;
  F4)
    POSE_MODE="conditional_translation"
    POSE_WEIGHT="0.5"
    ;;
  F5)
    POSE_MODE="conditional_translation"
    POSE_WEIGHT="0.5"
    MATCHABILITY_ARGS=(
      --candidate_teacher_assignment_fisher_use_matchability
      --candidate_teacher_assignment_fisher_matchability_floor "${SHOP_MAPFIM_MATCHABILITY_FLOOR:-0.05}"
      --candidate_teacher_assignment_fisher_matchability_power "${SHOP_MAPFIM_MATCHABILITY_POWER:-1.0}"
      --candidate_teacher_assignment_fisher_uncertainty_entropy_scale "${SHOP_MAPFIM_ENTROPY_SCALE:-2.0}"
    )
    ;;
  *)
    echo "Unknown mode: $MODE" >&2
    exit 2
    ;;
esac

POSE_WEIGHT="${SHOP_MAPFIM_POSE_WEIGHT:-$POSE_WEIGHT}"
POSE_FLOOR="${SHOP_MAPFIM_POSE_FLOOR:-0.2}"
NORMALIZATION="${SHOP_MAPFIM_NORMALIZATION:-$NORMALIZATION}"
TRANSLATION_SCALE="${SHOP_MAPFIM_TRANSLATION_SCALE:-0.02}"
ROTATION_SCALE_DEGREES="${SHOP_MAPFIM_ROTATION_SCALE_DEGREES:-2.0}"
MEASUREMENT_SIGMA="${SHOP_MAPFIM_MEASUREMENT_SIGMA:-1.0}"

MAP_CLEANLINESS_WEIGHT="${SHOP_MAPFIM_MAP_CLEANLINESS_WEIGHT:-0}"
MAP_FULL_INFORMATION_WEIGHT="${SHOP_MAPFIM_MAP_FULL_INFORMATION_WEIGHT:-0}"
MAP_TRANSLATION_INFORMATION_WEIGHT="${SHOP_MAPFIM_MAP_TRANSLATION_INFORMATION_WEIGHT:-0}"
MAP_TRANSLATION_TRACE_WEIGHT="${SHOP_MAPFIM_MAP_TRANSLATION_TRACE_WEIGHT:-0}"
MAP_TRANSLATION_CONDITION_WEIGHT="${SHOP_MAPFIM_MAP_TRANSLATION_CONDITION_WEIGHT:-0}"
MAP_BIAS_WEIGHT="${SHOP_MAPFIM_MAP_BIAS_WEIGHT:-0}"
MAP_DIRECTIONAL_BIAS_WEIGHT="${SHOP_MAPFIM_MAP_DIRECTIONAL_BIAS_WEIGHT:-0}"
MAP_CAPACITY_WEIGHT="${SHOP_MAPFIM_MAP_CAPACITY_WEIGHT:-0}"
MAP_TRANSLATION_SCALE="${SHOP_MAPFIM_MAP_TRANSLATION_SCALE:-$TRANSLATION_SCALE}"
MAP_ROTATION_SCALE_DEGREES="${SHOP_MAPFIM_MAP_ROTATION_SCALE_DEGREES:-$ROTATION_SCALE_DEGREES}"
MAP_MEASUREMENT_SIGMA="${SHOP_MAPFIM_MAP_MEASUREMENT_SIGMA:-$MEASUREMENT_SIGMA}"
MAP_RESIDUAL_CLIP_PX="${SHOP_MAPFIM_MAP_RESIDUAL_CLIP_PX:-12.0}"
MAP_INLIER_SIGMA_PX="${SHOP_MAPFIM_MAP_INLIER_SIGMA_PX:-4.0}"
MAP_CONDITION_TARGET="${SHOP_MAPFIM_MAP_CONDITION_TARGET:-100.0}"
MAP_MAX_MATCHES_PER_LANDMARK="${SHOP_MAPFIM_MAP_MAX_MATCHES_PER_LANDMARK:-2}"
MAP_DIRECTIONAL_TOPK="${SHOP_MAPFIM_MAP_DIRECTIONAL_TOPK:-0}"
MAP_DIRECTIONAL_TEMPERATURE="${SHOP_MAPFIM_MAP_DIRECTIONAL_TEMPERATURE:-0.05}"
MAP_DIRECTIONAL_RESIDUAL_CLIP_PX="${SHOP_MAPFIM_MAP_DIRECTIONAL_RESIDUAL_CLIP_PX:-24.0}"
MAP_DIRECTIONAL_ROBUST_SCALE_PX="${SHOP_MAPFIM_MAP_DIRECTIONAL_ROBUST_SCALE_PX:-12.0}"
MAP_DIRECTIONAL_ROBUST_QUALITY_FLOOR="${SHOP_MAPFIM_MAP_DIRECTIONAL_ROBUST_QUALITY_FLOOR:-0.01}"
DUSTBIN_WEIGHT="${SHOP_MAPFIM_DUSTBIN_WEIGHT:-0}"
ASSIGNMENT_MODE="${SHOP_MAPFIM_ASSIGNMENT_MODE:-single_nearest}"
COUNTERFACTUAL_ASSIGNMENT_WEIGHT="${SHOP_MAPFIM_COUNTERFACTUAL_ASSIGNMENT_WEIGHT:-0}"
COUNTERFACTUAL_BIAS_UTILITY_WEIGHT="${SHOP_MAPFIM_COUNTERFACTUAL_BIAS_UTILITY_WEIGHT:-1.0}"
COUNTERFACTUAL_TRANSLATION_UTILITY_WEIGHT="${SHOP_MAPFIM_COUNTERFACTUAL_TRANSLATION_UTILITY_WEIGHT:-0.0}"
COUNTERFACTUAL_UTILITY_FLOOR="${SHOP_MAPFIM_COUNTERFACTUAL_UTILITY_FLOOR:-0.1}"
COUNTERFACTUAL_TARGET_MODE="${SHOP_MAPFIM_COUNTERFACTUAL_TARGET_MODE:-all_false}"
COUNTERFACTUAL_ARGS=(
  --candidate_teacher_counterfactual_target_mode "$COUNTERFACTUAL_TARGET_MODE"
)
if [[ "${SHOP_MAPFIM_COUNTERFACTUAL_REQUIRE_CURRENT_RETAINED:-0}" == "1" ]]; then
  COUNTERFACTUAL_ARGS+=(--candidate_teacher_counterfactual_require_current_retained)
fi
if [[ "${SHOP_MAPFIM_COUNTERFACTUAL_REQUIRE_POSITIVE_BIAS_GAIN:-0}" == "1" ]]; then
  COUNTERFACTUAL_ARGS+=(--candidate_teacher_counterfactual_require_positive_bias_gain)
fi
if [[ "${SHOP_MAPFIM_COUNTERFACTUAL_REQUIRE_NONNEGATIVE_TRANSLATION_GAIN:-0}" == "1" ]]; then
  COUNTERFACTUAL_ARGS+=(--candidate_teacher_counterfactual_require_nonnegative_translation_gain)
fi

SAVE_ITERATIONS=()
if (( ITERATIONS >= 500 )); then
  SAVE_ITERATIONS+=(500)
fi
if (( ITERATIONS >= 1000 )); then
  SAVE_ITERATIONS+=(1000)
fi
if (( ${#SAVE_ITERATIONS[@]} == 0 )) || [[ "${SAVE_ITERATIONS[-1]}" != "$ITERATIONS" ]]; then
  SAVE_ITERATIONS+=("$ITERATIONS")
fi

LANDMARK_PATH="$MODEL_ROOT/$BASE_DETECTOR_FOLDER/sampled_idx.pkl"
DETECTOR_PATH="$MODEL_ROOT/$BASE_DETECTOR_FOLDER/30000_detector.pth"
FINAL_STATE="$MODEL_ROOT/$OUTPUT_FOLDER/${ITERATIONS}_candidate_teacher_state.pt"
if [[ -f "$FINAL_STATE" ]]; then
  echo "Completed artifact already exists: $FINAL_STATE"
  exit 0
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
  --candidate_teacher_detector_init_path "$DETECTOR_PATH" \
  --candidate_teacher_optimize_features \
  --candidate_teacher_freeze_detector \
  --candidate_teacher_feature_lr 0.0002 \
  --candidate_teacher_detect_num 4096 \
  --candidate_teacher_nms_radius 2 \
  --candidate_teacher_match_topk 1 \
  --candidate_teacher_hard_negatives 8 \
  --candidate_teacher_pair_weight 0 \
  --candidate_teacher_hard_negative_weight 0 \
  --candidate_teacher_assignment_weight 1 \
  --candidate_teacher_assignment_mode "$ASSIGNMENT_MODE" \
  --candidate_teacher_counterfactual_assignment_weight "$COUNTERFACTUAL_ASSIGNMENT_WEIGHT" \
  --candidate_teacher_dustbin_weight "$DUSTBIN_WEIGHT" \
  --candidate_teacher_assignment_temperature 0.05 \
  --candidate_teacher_assignment_margin 0.05 \
  --candidate_teacher_detector_match_weight 0 \
  --candidate_teacher_geometry_weight 0 \
  --candidate_teacher_coverage_weight 0 \
  --candidate_teacher_base_detector_weight 0 \
  --candidate_teacher_feature_anchor_weight 0.01 \
  --candidate_teacher_validation_ratio "$VALIDATION_RATIO" \
  --candidate_teacher_split_mode "$SPLIT_MODE" \
  --candidate_teacher_split_seed "$SPLIT_SEED" \
  --candidate_teacher_assignment_pose_information_mode "$POSE_MODE" \
  --candidate_teacher_assignment_pose_information_weight "$POSE_WEIGHT" \
  --candidate_teacher_assignment_pose_information_floor "$POSE_FLOOR" \
  --candidate_teacher_assignment_pose_information_normalization "$NORMALIZATION" \
  --candidate_teacher_assignment_fisher_translation_scale "$TRANSLATION_SCALE" \
  --candidate_teacher_assignment_fisher_rotation_scale_degrees "$ROTATION_SCALE_DEGREES" \
  --candidate_teacher_assignment_fisher_measurement_sigma "$MEASUREMENT_SIGMA" \
  --candidate_teacher_map_cleanliness_weight "$MAP_CLEANLINESS_WEIGHT" \
  --candidate_teacher_map_full_information_weight "$MAP_FULL_INFORMATION_WEIGHT" \
  --candidate_teacher_map_translation_information_weight "$MAP_TRANSLATION_INFORMATION_WEIGHT" \
  --candidate_teacher_map_translation_trace_weight "$MAP_TRANSLATION_TRACE_WEIGHT" \
  --candidate_teacher_map_translation_condition_weight "$MAP_TRANSLATION_CONDITION_WEIGHT" \
  --candidate_teacher_map_bias_weight "$MAP_BIAS_WEIGHT" \
  --candidate_teacher_map_directional_bias_weight "$MAP_DIRECTIONAL_BIAS_WEIGHT" \
  --candidate_teacher_map_capacity_weight "$MAP_CAPACITY_WEIGHT" \
  --candidate_teacher_map_fisher_translation_scale "$MAP_TRANSLATION_SCALE" \
  --candidate_teacher_map_fisher_rotation_scale_degrees "$MAP_ROTATION_SCALE_DEGREES" \
  --candidate_teacher_map_fisher_measurement_sigma_px "$MAP_MEASUREMENT_SIGMA" \
  --candidate_teacher_map_fisher_residual_clip_px "$MAP_RESIDUAL_CLIP_PX" \
  --candidate_teacher_map_fisher_inlier_sigma_px "$MAP_INLIER_SIGMA_PX" \
  --candidate_teacher_map_fisher_condition_target "$MAP_CONDITION_TARGET" \
  --candidate_teacher_map_max_matches_per_landmark "$MAP_MAX_MATCHES_PER_LANDMARK" \
  --candidate_teacher_map_directional_topk "$MAP_DIRECTIONAL_TOPK" \
  --candidate_teacher_map_directional_temperature "$MAP_DIRECTIONAL_TEMPERATURE" \
  --candidate_teacher_map_directional_residual_clip_px "$MAP_DIRECTIONAL_RESIDUAL_CLIP_PX" \
  --candidate_teacher_map_directional_robust_scale_px "$MAP_DIRECTIONAL_ROBUST_SCALE_PX" \
  --candidate_teacher_map_directional_robust_quality_floor "$MAP_DIRECTIONAL_ROBUST_QUALITY_FLOOR" \
  --candidate_teacher_counterfactual_bias_utility_weight "$COUNTERFACTUAL_BIAS_UTILITY_WEIGHT" \
  --candidate_teacher_counterfactual_translation_utility_weight "$COUNTERFACTUAL_TRANSLATION_UTILITY_WEIGHT" \
  --candidate_teacher_counterfactual_utility_floor "$COUNTERFACTUAL_UTILITY_FLOOR" \
  "${COUNTERFACTUAL_ARGS[@]}" \
  "${MATCHABILITY_ARGS[@]}"
