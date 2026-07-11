#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <scene> <gpu> <r2|pair|configs|eval> [baseline|field|pair|best|all]" >&2
  exit 2
fi

SCENE="$1"
GPU="$2"
MODE="$3"
VARIANT="${4:-all}"

case "$SCENE" in
  GreatCourt|KingsCollege|OldHospital|StMarysChurch) ;;
  *)
    echo "Unsupported Cambridge scene: $SCENE" >&2
    exit 2
    ;;
esac

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_ROOT="/mnt/pool/sqy/stdloc_la_full_runs/${SCENE}_baseline"
EXPERIMENT_ROOT="${CAMBRIDGE_CROSSSCENE_ROOT:-/mnt/pool/sqy/stdloc_lafgs_cambridge_best_crossscene_20260711}"
CONFIG_ROOT="$EXPERIMENT_ROOT/eval_configs/$SCENE"

R2_FOLDER="detector_crossscene_best_R2_2000"
JOINT_FOLDER="detector_crossscene_pair_joint500"
SET_BIAS_FOLDER="detector_crossscene_pair_setbias500"
SET_CONTEXT_FOLDER="detector_crossscene_pair_setcontext500"
GEOMETRY_CONTEXT_FOLDER="detector_crossscene_pair_geometrycontext500"

export CUDA_HOME=/usr/local/cuda-11.8
export PATH="/root/miniconda3/envs/cybersim_agent/bin:/usr/local/cuda-11.8/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-11.8/lib64:/root/miniconda3/envs/cybersim_agent/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$REPO_ROOT"
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export CUDA_VISIBLE_DEVICES="$GPU"

mkdir -p "$EXPERIMENT_ROOT/logs" "$CONFIG_ROOT"
cd "$REPO_ROOT"

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "Required artifact is missing: $1" >&2
    exit 1
  fi
}

run_r2() {
  require_file "$MODEL_ROOT/detector/30000_detector.pth"
  require_file "$MODEL_ROOT/detector/sampled_idx.pkl"

  python train_detector.py \
    --model_path "$MODEL_ROOT" \
    --iteration 30000 \
    --iterations 2000 \
    --test_iterations 1000 2000 \
    --save_iterations 1000 2000 \
    --detector_folder "$R2_FOLDER" \
    --landmark_num 16384 \
    --precomputed_landmark_path "$MODEL_ROOT/detector/sampled_idx.pkl" \
    --sparse_candidate_teacher \
    --candidate_teacher_detector_init_path "$MODEL_ROOT/detector/30000_detector.pth" \
    --candidate_teacher_optimize_features \
    --candidate_teacher_detect_num 4096 \
    --candidate_teacher_nms_radius 2 \
    --candidate_teacher_match_topk 1 \
    --candidate_teacher_pair_weight 0 \
    --candidate_teacher_hard_negative_weight 0 \
    --candidate_teacher_assignment_weight 1 \
    --candidate_teacher_detector_match_weight 1 \
    --candidate_teacher_geometry_weight 0 \
    --candidate_teacher_coverage_weight 0 \
    --candidate_teacher_base_detector_weight 0.05 \
    --candidate_teacher_feature_anchor_weight 0.01
}

pair_common() {
  local folder="$1"
  local init_path="$2"
  local lr="$3"
  shift 3

  require_file "$MODEL_ROOT/$R2_FOLDER/2000_detector.pth"
  require_file "$MODEL_ROOT/$R2_FOLDER/2000_candidate_teacher_state.pt"
  require_file "$MODEL_ROOT/$R2_FOLDER/sampled_idx.pkl"
  if [[ -n "$init_path" ]]; then
    require_file "$MODEL_ROOT/$init_path"
  fi

  local init_args=()
  if [[ -n "$init_path" ]]; then
    init_args=(--candidate_teacher_pair_measurement_init_path "$init_path")
  fi

  python train_detector.py \
    --model_path "$MODEL_ROOT" \
    --iteration 30000 \
    --iterations 500 \
    --test_iterations 500 \
    --save_iterations 100 250 500 \
    --detector_folder "$folder" \
    --landmark_num 16384 \
    --sampling_mode baseline \
    --precomputed_landmark_path "$MODEL_ROOT/$R2_FOLDER/sampled_idx.pkl" \
    --sparse_candidate_teacher \
    --candidate_teacher_detector_init_path "$MODEL_ROOT/$R2_FOLDER/2000_detector.pth" \
    --candidate_teacher_state_init_path "$R2_FOLDER/2000_candidate_teacher_state.pt" \
    "${init_args[@]}" \
    --candidate_teacher_freeze_detector \
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
    --candidate_teacher_pair_measurement_inlier_weight 1 \
    --candidate_teacher_pair_measurement_nll_weight 1 \
    --candidate_teacher_pair_measurement_lr "$lr" \
    --candidate_teacher_detector_match_weight 0 \
    --candidate_teacher_detector_offset_weight 0 \
    --candidate_teacher_geometry_weight 0 \
    --candidate_teacher_coverage_weight 0 \
    --candidate_teacher_base_detector_weight 0 \
    --candidate_teacher_feature_anchor_weight 0 \
    --candidate_teacher_dustbin_lr 5e-5 \
    --candidate_teacher_support_query_split \
    --candidate_teacher_query_ratio 0.2 \
    --candidate_teacher_validation_ratio 0.25 \
    --candidate_teacher_split_mode temporal_block \
    --candidate_teacher_split_seed 2026 \
    "$@"
}

run_pair() {
  pair_common "$JOINT_FOLDER" "" 0.001
  pair_common \
    "$SET_BIAS_FOLDER" \
    "$JOINT_FOLDER/500_candidate_teacher_state.pt" \
    0.0003 \
    --candidate_teacher_pair_measurement_bias_weight 0.0001 \
    --candidate_teacher_pair_measurement_covariance_weight 0 \
    --candidate_teacher_pair_measurement_residual_clip_px 32 \
    --candidate_teacher_pair_measurement_reference_translation_m 0.01
  pair_common \
    "$SET_CONTEXT_FOLDER" \
    "$SET_BIAS_FOLDER/500_candidate_teacher_state.pt" \
    0.0003 \
    --candidate_teacher_pair_measurement_bias_weight 0.0001 \
    --candidate_teacher_pair_measurement_covariance_weight 0 \
    --candidate_teacher_pair_measurement_residual_clip_px 32 \
    --candidate_teacher_pair_measurement_reference_translation_m 0.01 \
    --candidate_teacher_pair_measurement_set_context
  pair_common \
    "$GEOMETRY_CONTEXT_FOLDER" \
    "$SET_CONTEXT_FOLDER/500_candidate_teacher_state.pt" \
    0.0003 \
    --candidate_teacher_pair_measurement_bias_weight 0.0001 \
    --candidate_teacher_pair_measurement_covariance_weight 0 \
    --candidate_teacher_pair_measurement_residual_clip_px 32 \
    --candidate_teacher_pair_measurement_reference_translation_m 0.01 \
    --candidate_teacher_pair_measurement_geometry_context
}

make_config() {
  local variant="$1"
  local detector_folder="detector"
  local detector_iters=30000
  local extra=()

  case "$variant" in
    baseline) ;;
    field)
      detector_folder="$R2_FOLDER"
      detector_iters=2000
      extra=(
        --candidate_teacher_state_path "$R2_FOLDER/2000_candidate_teacher_state.pt"
        --landmark_feature_override_path "$R2_FOLDER/2000_candidate_teacher_state.pt"
        --override_landmark_features
      )
      ;;
    pair|best)
      detector_folder="$R2_FOLDER"
      detector_iters=2000
      extra=(
        --candidate_teacher_state_path "$R2_FOLDER/2000_candidate_teacher_state.pt"
        --landmark_feature_override_path "$R2_FOLDER/2000_candidate_teacher_state.pt"
        --override_landmark_features
        --pair_measurement_state_path "$GEOMETRY_CONTEXT_FOLDER/500_candidate_teacher_state.pt"
        --use_pair_measurement
        --use_pair_measurement_calibrated_threshold
      )
      if [[ "$variant" == "best" ]]; then
        extra+=(
          --min_candidate_matches 1024
          --candidate_refill_trigger_count 1024
          --pair_measurement_refill_mode score
        )
      fi
      ;;
    *)
      echo "Unknown eval variant: $variant" >&2
      exit 2
      ;;
  esac

  python scripts/make_stdloc_eval_cfg.py \
    --base_cfg configs/stdloc_cambridge.yaml \
    --output "$CONFIG_ROOT/${variant}.yaml" \
    --artifact_model_path "$MODEL_ROOT" \
    --detector_folder "$detector_folder" \
    --detector_iters "$detector_iters" \
    --detect_num 4096 \
    --reprojection_error 12 \
    --nms 2 \
    --match_threshold 0 \
    --match_topk 1 \
    --max_matches_per_landmark 2 \
    --candidate_frontend_match_policy error \
    --summary_json "$CONFIG_ROOT/${variant}.json" \
    "${extra[@]}"
}

run_configs() {
  make_config baseline
  make_config field
  make_config pair
  make_config best
}

run_eval_variant() {
  local variant="$1"
  require_file "$CONFIG_ROOT/${variant}.yaml"
  python stdloc.py \
    --model_path "$MODEL_ROOT" \
    --iteration 30000 \
    --cfg "$CONFIG_ROOT/${variant}.yaml" \
    --prefix "crossscene-${variant}-${SCENE}" \
    --sparse_only
}

run_eval() {
  if [[ "$VARIANT" == "all" ]]; then
    run_eval_variant baseline
    run_eval_variant field
    run_eval_variant pair
    run_eval_variant best
  else
    run_eval_variant "$VARIANT"
  fi
}

case "$MODE" in
  r2) run_r2 ;;
  pair) run_pair ;;
  configs) run_configs ;;
  eval) run_eval ;;
  *)
    echo "Unknown mode: $MODE" >&2
    exit 2
    ;;
esac
