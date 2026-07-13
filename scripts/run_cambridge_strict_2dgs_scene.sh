#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 4 ]]; then
  echo "Usage: $0 <scene> <gpu> <lafgs|matcha> <map|covsoft|r2|pair|configs|eval|all> [baseline|field|pair|best|all]" >&2
  exit 2
fi

SCENE="$1"
GPU="$2"
MAP_KIND="$3"
MODE="$4"
VARIANT="${5:-all}"

case "$SCENE" in
  GreatCourt|KingsCollege|OldHospital|ShopFacade|StMarysChurch) ;;
  *) echo "Unsupported Cambridge scene: $SCENE" >&2; exit 2 ;;
esac
case "$MAP_KIND" in
  lafgs|matcha) ;;
  *) echo "Map kind must be lafgs or matcha" >&2; exit 2 ;;
esac

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/root/miniconda3/envs/cybersim_agent/bin/python}"
DATA_ROOT="${CAMBRIDGE_DATA_ROOT:-/mnt/pool/sqy/Cambridge_stdloc}"
EXPERIMENT_ROOT="${CAMBRIDGE_STRICT_2DGS_ROOT:-/mnt/pool/sqy/stdloc_lafgs_cambridge_matcha2dgs_strict_20260711}"

if [[ "$MAP_KIND" == "lafgs" ]]; then
  MODEL_ROOT="$EXPERIMENT_ROOT/lafgs_from_sfm/$SCENE"
  MODEL_ITERATION=30000
else
  MODEL_ROOT="$EXPERIMENT_ROOT/matcha_feature_baseline/$SCENE"
  MODEL_ITERATION=60000
fi

CONFIG_ROOT="$EXPERIMENT_ROOT/eval_configs/$MAP_KIND/$SCENE"
R2_FOLDER="detector_strict2dgs_R2_flr2e4_2000"
JOINT_FOLDER="detector_strict2dgs_pair_flr2e4_joint500"
SET_BIAS_FOLDER="detector_strict2dgs_pair_flr2e4_setbias500"
SET_CONTEXT_FOLDER="detector_strict2dgs_pair_flr2e4_setcontext500"
GEOMETRY_CONTEXT_FOLDER="detector_strict2dgs_pair_flr2e4_geometrycontext500"
COVSOFT_FOLDER="detector_covsoft_fixlineage_30000"

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

run_lafgs_map() {
  local final_ply="$MODEL_ROOT/point_cloud/iteration_30000/point_cloud.ply"
  if [[ -f "$final_ply" ]]; then
    echo "[strict-2dgs] Skip completed LaFGS map: $final_ply"
    return
  fi
  mkdir -p "$MODEL_ROOT"
  "$PYTHON" train_lafgs.py \
    -s "$DATA_ROOT/$SCENE" -m "$MODEL_ROOT" \
    -r 1 -f sp -g 2dgs --images processed --data_device cpu \
    --densify_grad_threshold 0.0004 \
    --position_lr_init 0.000016 \
    --scaling_lr 0.001 \
    --iterations 30000 \
    --train_seed 0 \
    --train_phase full \
    --loc_interval 1 \
    --synthetic_view_ratio 0 \
    --synthetic_view_desc_weight 0 \
    --synthetic_view_reproj_weight 0 \
    --lafgs_stage_schedule sfm_from_zero \
    --lafgs_stage_bootstrap_until 3000 \
    --lafgs_stage_joint_until 15000 \
    --lafgs_rgb_densify \
    --lafgs_rgb_densify_until_iter 15000 \
    --landmark_path __all__ \
    --lafgs_mvinit_enabled \
    --lafgs_mvinit_max_views 64 \
    --lafgs_mvinit_view_selection uniform \
    --lafgs_mvinit_chunk_size 32768 \
    --lafgs_mvinit_feature_scale 0.5 \
    --loc_full_bank_pose_information_weight 0.5 \
    --loc_full_bank_pose_information_floor 0.2 \
    --loc_full_bank_nearby_as_positive \
    --loc_full_bank_nearby_as_positive_until 15000 \
    --lafgs_curriculum \
    --lafgs_diff_pnp_start_iter 3000 \
    --lafgs_diff_pnp_weight 0.05 \
    --lafgs_diff_pnp_max_correspondences 64 \
    --lafgs_diff_pnp_spatial_grid_size 4 \
    --lafgs_diff_pnp_point_weight_floor 0.05 \
    --lafgs_diff_pnp_local_window_radius 1.25 \
    --lafgs_diff_pnp_geometry_local_window_radius 1.5 \
    --lafgs_diff_pnp_max_condition_number 100000 \
    --lafgs_diff_pnp_geometry_pose_guard_max_loss_increase -1 \
    --lafgs_diff_pnp_geometry_pose_guard_max_loss 5 \
    --lafgs_diff_pnp_geometry_pose_guard_softness 10 \
    --lafgs_diff_pnp_geometry_pose_guard_min_scale 0.05 \
    --lafgs_diff_pnp_feedback_pose_guard_max_loss_increase 30 \
    --lafgs_diff_pnp_feedback_pose_guard_max_loss 5 \
    --lafgs_diff_pnp_feedback_pose_guard_softness 10 \
    --lafgs_diff_pnp_feedback_pose_guard_min_scale 0.05 \
    --lafgs_diff_pnp_allow_geometry_grad \
    --lafgs_diff_pnp_isolate_geometry_grad \
    --disallow_raw_xyz_geometry_grad \
    --lafgs_diff_pnp_geometry_xyz_lr 0 \
    --lafgs_diff_pnp_geometry_reproj_weight 0.01 \
    --lafgs_diff_pnp_geometry_depth_anchor_weight 0.1 \
    --loc_anchor_lr 0.00005 \
    --surfel_loc_tangent_bound 0.03 \
    --surfel_loc_normal_bound 0.005 \
    --surfel_loc_radius_floor 1 \
    --surfel_loc_anchor_reg_weight 0.1 \
    --lafgs_diff_pnp_geometry_match_reproj_weight 0.5 \
    --lafgs_diff_pnp_geometry_match_max_reproj_error 2 \
    --lafgs_diff_pnp_geometry_max_reproj_error 4 \
    --loc_full_checkpoint_mode none \
    --save_iterations 5000 10000 20000 30000 \
    --test_iterations 5000 10000 20000 30000
}

run_matcha_feature_map() {
  local source_ply="$MODEL_ROOT/point_cloud/iteration_30000/point_cloud.ply"
  local final_ply="$MODEL_ROOT/point_cloud/iteration_60000/point_cloud.ply"
  require_file "$source_ply"
  if [[ -f "$final_ply" ]]; then
    echo "[strict-2dgs] Skip completed MAtCha feature baseline: $final_ply"
    return
  fi
  "$PYTHON" train_locaware.py \
    -s "$DATA_ROOT/$SCENE" -m "$MODEL_ROOT" \
    -r 1 -f sp -g 2dgs --images processed --data_device cpu \
    --load_iteration 30000 \
    --iterations 60000 \
    --train_seed 0 \
    --feature_only \
    --train_phase feature \
    --base_loss_weight 1 \
    --base_feature_weight 1 \
    --loc_loss_weight 0 \
    --loc_start_iter 1000000 \
    --loc_interval 1 \
    --use_loc_opacity \
    --lafgs_mvinit_enabled \
    --lafgs_mvinit_max_views 64 \
    --lafgs_mvinit_view_selection uniform \
    --lafgs_mvinit_chunk_size 32768 \
    --lafgs_mvinit_feature_scale 0.5 \
    --loc_full_checkpoint_mode none \
    --save_iterations 60000 \
    --test_iterations 60000
}

run_map() {
  if [[ "$MAP_KIND" == "lafgs" ]]; then
    run_lafgs_map
  else
    run_matcha_feature_map
  fi
}

run_covsoft() {
  require_file "$MODEL_ROOT/point_cloud/iteration_${MODEL_ITERATION}/point_cloud.ply"
  if [[ -f "$MODEL_ROOT/$COVSOFT_FOLDER/30000_detector.pth" ]]; then
    echo "[strict-2dgs] Skip completed covsoft detector: $MODEL_ROOT/$COVSOFT_FOLDER"
    return
  fi
  "$PYTHON" train_detector.py \
    -s "$DATA_ROOT/$SCENE" -m "$MODEL_ROOT" \
    -r 1 -f sp -g 2dgs --images processed --data_device cpu \
    --iteration "$MODEL_ITERATION" \
    --iterations 30000 \
    --test_iterations 30000 \
    --save_iterations 30000 \
    --detector_folder "$COVSOFT_FOLDER" \
    --landmark_num 16384 \
    --landmark_k 32 \
    --sampling_mode coverage_preserving \
    --detector_target_mode soft \
    --soft_sigma 1.5 \
    --min_loc_observations 4 \
    --utility_weight 1 \
    --pnp_voxel_size 0.25 \
    --pnp_max_per_voxel 8 \
    --pnp_preserve_ratio 0.5 \
    --coverage_preserve_ratio 0.75 \
    --coverage_utility_ratio 0.10 \
    --coverage_high_confidence_ratio 0.10 \
    --coverage_grid_size 4 \
    --coverage_max_per_grid 1536 \
    --coverage_depth_bins 4 \
    --coverage_max_per_depth_bin 6144 \
    --coverage_allow_unbalanced_fallback
}

run_r2() {
  require_file "$MODEL_ROOT/$COVSOFT_FOLDER/30000_detector.pth"
  require_file "$MODEL_ROOT/$COVSOFT_FOLDER/sampled_idx.pkl"
  if [[ -f "$MODEL_ROOT/$R2_FOLDER/2000_candidate_teacher_state.pt" ]]; then
    echo "[strict-2dgs] Skip completed R2: $MODEL_ROOT/$R2_FOLDER"
    return
  fi
  "$PYTHON" train_detector.py \
    --model_path "$MODEL_ROOT" \
    --iteration "$MODEL_ITERATION" \
    --iterations 2000 \
    --test_iterations 1000 2000 \
    --save_iterations 1000 2000 \
    --detector_folder "$R2_FOLDER" \
    --landmark_num 16384 \
    --precomputed_landmark_path "$MODEL_ROOT/$COVSOFT_FOLDER/sampled_idx.pkl" \
    --sparse_candidate_teacher \
    --candidate_teacher_detector_init_path "$MODEL_ROOT/$COVSOFT_FOLDER/30000_detector.pth" \
    --candidate_teacher_optimize_features \
    --candidate_teacher_feature_lr 0.0002 \
    --candidate_teacher_detector_lr 0.0001 \
    --candidate_teacher_detect_num 4096 \
    --candidate_teacher_nms_radius 2 \
    --candidate_teacher_match_topk 1 \
    --candidate_teacher_hard_negatives 8 \
    --candidate_teacher_pair_weight 0 \
    --candidate_teacher_hard_negative_weight 0 \
    --candidate_teacher_assignment_weight 1 \
    --candidate_teacher_assignment_temperature 0.05 \
    --candidate_teacher_assignment_margin 0.05 \
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
  if [[ -f "$MODEL_ROOT/$folder/500_candidate_teacher_state.pt" ]]; then
    echo "[strict-2dgs] Skip completed pair stage: $MODEL_ROOT/$folder"
    return
  fi
  require_file "$MODEL_ROOT/$R2_FOLDER/2000_detector.pth"
  require_file "$MODEL_ROOT/$R2_FOLDER/2000_candidate_teacher_state.pt"
  require_file "$MODEL_ROOT/$R2_FOLDER/sampled_idx.pkl"
  local init_args=()
  if [[ -n "$init_path" ]]; then
    require_file "$MODEL_ROOT/$init_path"
    init_args=(--candidate_teacher_pair_measurement_init_path "$init_path")
  fi
  "$PYTHON" train_detector.py \
    --model_path "$MODEL_ROOT" \
    --iteration "$MODEL_ITERATION" \
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
  pair_common "$SET_BIAS_FOLDER" "$JOINT_FOLDER/500_candidate_teacher_state.pt" 0.0003 \
    --candidate_teacher_pair_measurement_bias_weight 0.0001 \
    --candidate_teacher_pair_measurement_covariance_weight 0 \
    --candidate_teacher_pair_measurement_residual_clip_px 32 \
    --candidate_teacher_pair_measurement_reference_translation_m 0.01
  pair_common "$SET_CONTEXT_FOLDER" "$SET_BIAS_FOLDER/500_candidate_teacher_state.pt" 0.0003 \
    --candidate_teacher_pair_measurement_bias_weight 0.0001 \
    --candidate_teacher_pair_measurement_covariance_weight 0 \
    --candidate_teacher_pair_measurement_residual_clip_px 32 \
    --candidate_teacher_pair_measurement_reference_translation_m 0.01 \
    --candidate_teacher_pair_measurement_set_context
  pair_common "$GEOMETRY_CONTEXT_FOLDER" "$SET_CONTEXT_FOLDER/500_candidate_teacher_state.pt" 0.0003 \
    --candidate_teacher_pair_measurement_bias_weight 0.0001 \
    --candidate_teacher_pair_measurement_covariance_weight 0 \
    --candidate_teacher_pair_measurement_residual_clip_px 32 \
    --candidate_teacher_pair_measurement_reference_translation_m 0.01 \
    --candidate_teacher_pair_measurement_geometry_context
}

make_config() {
  local variant="$1"
  local detector_folder="$COVSOFT_FOLDER"
  local detector_iters=30000
  local extra=()
  case "$variant" in
    baseline) ;;
    field)
      detector_folder="$R2_FOLDER"; detector_iters=2000
      extra=(--candidate_teacher_state_path "$R2_FOLDER/2000_candidate_teacher_state.pt" --landmark_feature_override_path "$R2_FOLDER/2000_candidate_teacher_state.pt" --override_landmark_features)
      ;;
    pair|best)
      detector_folder="$R2_FOLDER"; detector_iters=2000
      extra=(--candidate_teacher_state_path "$R2_FOLDER/2000_candidate_teacher_state.pt" --landmark_feature_override_path "$R2_FOLDER/2000_candidate_teacher_state.pt" --override_landmark_features --pair_measurement_state_path "$GEOMETRY_CONTEXT_FOLDER/500_candidate_teacher_state.pt" --use_pair_measurement --use_pair_measurement_calibrated_threshold)
      if [[ "$variant" == "best" ]]; then
        extra+=(--min_candidate_matches 1024 --candidate_refill_trigger_count 1024 --pair_measurement_refill_mode score)
      fi
      ;;
    *) echo "Unknown eval variant: $variant" >&2; exit 2 ;;
  esac
  "$PYTHON" scripts/make_stdloc_eval_cfg.py \
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
  "$PYTHON" stdloc.py \
    --model_path "$MODEL_ROOT" \
    --iteration "$MODEL_ITERATION" \
    --cfg "$CONFIG_ROOT/${variant}.yaml" \
    --prefix "strict2dgs-${MAP_KIND}-${variant}-${SCENE}" \
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
  map) run_map ;;
  covsoft) run_covsoft ;;
  r2) run_r2 ;;
  pair) run_pair ;;
  configs) run_configs ;;
  eval) run_eval ;;
  all) run_map; run_covsoft; run_r2; run_pair; run_configs; run_eval ;;
  *) echo "Unknown mode: $MODE" >&2; exit 2 ;;
esac
