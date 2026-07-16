#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <scene> <gpu> <prepare|smoke|field|detector|calibrate|candidate|eval|select|all> [test|validation] [detector|candidate] [iteration]" >&2
  exit 2
fi

SCENE="$1"
GPU="$2"
MODE="$3"
EVAL_SUBSET="${4:-test}"
EVAL_VARIANT="${5:-candidate}"
EVAL_ITERATION="${6:-}"

case "$SCENE" in
  GreatCourt|KingsCollege|OldHospital|ShopFacade|StMarysChurch) ;;
  *) echo "Unsupported Cambridge scene: $SCENE" >&2; exit 2 ;;
esac
case "$GPU" in
  1|2) ;;
  *) echo "This experiment is restricted to GPU 1 or GPU 2; got GPU $GPU" >&2; exit 2 ;;
esac
case "$MODE" in
  prepare|smoke|field|detector|calibrate|candidate|eval|select|all) ;;
  *) echo "Unsupported mode: $MODE" >&2; exit 2 ;;
esac
case "$EVAL_SUBSET" in
  test|validation) ;;
  *) echo "Evaluation subset must be test or validation" >&2; exit 2 ;;
esac
case "$EVAL_VARIANT" in
  detector|candidate) ;;
  *) echo "Evaluation variant must be detector or candidate" >&2; exit 2 ;;
esac

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/root/miniconda3/envs/cybersim_agent/bin/python}"
DATA_ROOT="${CAMBRIDGE_DATA_ROOT:-/mnt/pool/sqy/Cambridge_stdloc}"
MATCHA_ROOT="${CAMBRIDGE_MATCHA_2DGS_ROOT:-/root/MAtCha/output_cambridge_full_retained_v2}"
EXPERIMENT_ROOT="${CAMBRIDGE_MATCHA_MAINLINE_ROOT:-/mnt/pool/sqy/stdloc_matcha_pretrained_mainline_v1_20260714}"
FIELD_STEPS="${CAMBRIDGE_MATCHA_FIELD_STEPS:-30000}"
SMOKE_STEPS="${CAMBRIDGE_MATCHA_SMOKE_STEPS:-1}"
LOC_INTERVAL="${CAMBRIDGE_MATCHA_LOC_INTERVAL:-2}"
SMOKE_PNP_START="${CAMBRIDGE_MATCHA_SMOKE_PNP_START:-2}"
SMOKE_GEOMETRY_START="${CAMBRIDGE_MATCHA_SMOKE_GEOMETRY_START:-4}"
SMOKE_TOPOLOGY="${CAMBRIDGE_MATCHA_SMOKE_TOPOLOGY:-0}"
SOURCE_ITERATION=30000
MODEL_ITERATION=$((SOURCE_ITERATION + FIELD_STEPS))
MODEL_ROOT="$EXPERIMENT_ROOT/lafgs_external_matcha/$SCENE"
NORMALIZATION_JSON="$EXPERIMENT_ROOT/normalization/$SCENE.json"
CALIBRATION_JSON="$EXPERIMENT_ROOT/calibration/$SCENE.json"
CONFIG_ROOT="$EXPERIMENT_ROOT/eval_configs/$SCENE"
EVALUATION_ROOT="$EXPERIMENT_ROOT/evaluations/$SCENE"
LOG_ROOT="$EXPERIMENT_ROOT/logs/$SCENE"
AUDIT_ROOT="$EXPERIMENT_ROOT/audit/$SCENE"
CANDIDATE_OBJECTIVE="${CAMBRIDGE_MATCHA_CANDIDATE_OBJECTIVE:-f0}"
CANDIDATE_PAIR="${CAMBRIDGE_MATCHA_CANDIDATE_PAIR:-0}"
PAIR_FIXED_CANDIDATE_COUNT="${CAMBRIDGE_MATCHA_PAIR_FIXED_CANDIDATE_COUNT:-0}"
PAIR_REFILL_MODE="${CAMBRIDGE_MATCHA_PAIR_REFILL_MODE:-score}"
PAIR_COVARIANCE_REFINEMENT="${CAMBRIDGE_MATCHA_PAIR_COVARIANCE_REFINEMENT:-0}"
PAIR_PROGRESSIVE_SAMPLING="${CAMBRIDGE_MATCHA_PAIR_PROGRESSIVE_SAMPLING:-0}"
PAIR_USE_OFFSET="${CAMBRIDGE_MATCHA_PAIR_USE_OFFSET:-1}"
CANDIDATE_ONLINE_RENDER="${CAMBRIDGE_MATCHA_CANDIDATE_ONLINE_RENDER:-0}"
ONLINE_RENDER_RATIO_START="${CAMBRIDGE_MATCHA_ONLINE_RENDER_RATIO_START:-0.10}"
ONLINE_RENDER_RATIO_END="${CAMBRIDGE_MATCHA_ONLINE_RENDER_RATIO_END:-0.30}"
ONLINE_RENDER_PROVENANCE="${CAMBRIDGE_MATCHA_ONLINE_RENDER_PROVENANCE:-none}"
ONLINE_RENDER_PROVENANCE_WEIGHT="${CAMBRIDGE_MATCHA_ONLINE_RENDER_PROVENANCE_WEIGHT:-0.25}"
ONLINE_RENDER_SAMPLING_MODE="${CAMBRIDGE_MATCHA_ONLINE_RENDER_SAMPLING_MODE:-uniform}"
USE_CALIBRATED_FINAL_FRONTEND="${CAMBRIDGE_MATCHA_USE_CALIBRATED_FINAL_FRONTEND:-1}"
CANDIDATE_EVAL_TAG="${CAMBRIDGE_MATCHA_CANDIDATE_EVAL_TAG:-candidate_${CANDIDATE_OBJECTIVE}}"

case "$CANDIDATE_OBJECTIVE" in
  f0|exact) ;;
  *) echo "Candidate objective must be f0 or exact; got $CANDIDATE_OBJECTIVE" >&2; exit 2 ;;
esac
case "$CANDIDATE_PAIR" in
  0|1) ;;
  *) echo "CAMBRIDGE_MATCHA_CANDIDATE_PAIR must be 0 or 1" >&2; exit 2 ;;
esac
case "$PAIR_REFILL_MODE" in
  score|geometry) ;;
  *) echo "CAMBRIDGE_MATCHA_PAIR_REFILL_MODE must be score or geometry" >&2; exit 2 ;;
esac
case "$PAIR_COVARIANCE_REFINEMENT:$PAIR_PROGRESSIVE_SAMPLING" in
  0:0|0:1|1:0|1:1) ;;
  *) echo "Pair covariance/progressive flags must be 0 or 1" >&2; exit 2 ;;
esac
case "$PAIR_USE_OFFSET" in
  0|1) ;;
  *) echo "CAMBRIDGE_MATCHA_PAIR_USE_OFFSET must be 0 or 1" >&2; exit 2 ;;
esac
case "$CANDIDATE_ONLINE_RENDER" in
  0|1) ;;
  *) echo "CAMBRIDGE_MATCHA_CANDIDATE_ONLINE_RENDER must be 0 or 1" >&2; exit 2 ;;
esac
case "$ONLINE_RENDER_PROVENANCE" in
  none|hard|soft) ;;
  *) echo "CAMBRIDGE_MATCHA_ONLINE_RENDER_PROVENANCE must be none, hard, or soft" >&2; exit 2 ;;
esac
case "$ONLINE_RENDER_SAMPLING_MODE" in
  uniform|failure_guided) ;;
  *) echo "CAMBRIDGE_MATCHA_ONLINE_RENDER_SAMPLING_MODE must be uniform or failure_guided" >&2; exit 2 ;;
esac
case "$USE_CALIBRATED_FINAL_FRONTEND" in
  0|1) ;;
  *) echo "CAMBRIDGE_MATCHA_USE_CALIBRATED_FINAL_FRONTEND must be 0 or 1" >&2; exit 2 ;;
esac
case "$SMOKE_TOPOLOGY" in
  0|1) ;;
  *) echo "CAMBRIDGE_MATCHA_SMOKE_TOPOLOGY must be 0 or 1" >&2; exit 2 ;;
esac

case "$SCENE" in
  GreatCourt) MATCHA_RUN="GreatCourt_n20_long_masked_retrain_retry" ;;
  KingsCollege) MATCHA_RUN="KingsCollege_n20_long_masked_retrain" ;;
  OldHospital) MATCHA_RUN="OldHospital_n20_long_masked_retrain_retry" ;;
  ShopFacade) MATCHA_RUN="ShopFacade_n20_long_masked_retrain" ;;
  StMarysChurch) MATCHA_RUN="StMarysChurch_n20_long_masked_retrain" ;;
esac
SOURCE_PLY="$MATCHA_ROOT/$MATCHA_RUN/free_gaussians/point_cloud/iteration_${SOURCE_ITERATION}/point_cloud.ply"

export CUDA_VISIBLE_DEVICES="$GPU"
export CUDA_HOME=/usr/local/cuda-11.8
export PATH="/root/miniconda3/envs/cybersim_agent/bin:$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:/root/miniconda3/envs/cybersim_agent/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$REPO_ROOT"
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export PYTHONHASHSEED=0

mkdir -p "$LOG_ROOT" "$CONFIG_ROOT" "$EVALUATION_ROOT" "$AUDIT_ROOT"
cd "$REPO_ROOT"

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "Required artifact is missing: $1" >&2
    exit 1
  fi
}

record_command() {
  local stage="$1"
  shift
  printf '%q ' "$@" > "$LOG_ROOT/${stage}.command.sh"
  printf '\n' >> "$LOG_ROOT/${stage}.command.sh"
}

run_logged() {
  local stage="$1"
  shift
  record_command "$stage" "$@"
  "$@" 2>&1 | tee "$LOG_ROOT/${stage}.log"
}

prepare_scene() {
  require_file "$SOURCE_PLY"
  if [[ ! -L "$MODEL_ROOT/point_cloud/iteration_${SOURCE_ITERATION}" ]]; then
    run_logged audit \
      "$PYTHON" scripts/audit_cambridge_matcha_2dgs_protocol.py \
      --runs_root "$MATCHA_ROOT" \
      --data_root "$DATA_ROOT" \
      --scenes "$SCENE" \
      --output_json "$AUDIT_ROOT/protocol.json" \
      --output_markdown "$AUDIT_ROOT/protocol.md" \
      --prepare_wrapper_root "$EXPERIMENT_ROOT/lafgs_external_matcha"
  fi
  "$PYTHON" scripts/compute_scene_normalization.py \
    --dataset_root "$DATA_ROOT/$SCENE" \
    --point_cloud "$SOURCE_PLY" \
    --images processed \
    --field_steps "$FIELD_STEPS" \
    --output_json "$NORMALIZATION_JSON" > "$LOG_ROOT/normalization.log"
  require_file "$MODEL_ROOT/artifact_provenance.json"
  require_file "$NORMALIZATION_JSON"
}

load_normalization() {
  prepare_scene
  eval "$("$PYTHON" scripts/compute_scene_normalization.py \
    --dataset_root "$DATA_ROOT/$SCENE" \
    --point_cloud "$SOURCE_PLY" \
    --images processed \
    --field_steps "$FIELD_STEPS" \
    --output_json "$NORMALIZATION_JSON" \
    --shell)"
  CANDIDATE_STEPS="${CAMBRIDGE_MATCHA_CANDIDATE_STEPS_OVERRIDE:-$CANDIDATE_STEPS}"
  GEOMETRY_XYZ_LR="${CAMBRIDGE_MATCHA_GEOMETRY_XYZ_LR_OVERRIDE:-$GEOMETRY_XYZ_LR}"
  GEOMETRY_MAX_STEP_M="${CAMBRIDGE_MATCHA_GEOMETRY_MAX_STEP_M:-$((0))}"
  EVAL_DETECT_NUM="${CAMBRIDGE_MATCHA_EVAL_DETECT_NUM_OVERRIDE:-$EVAL_DETECT_NUM}"
  DETECTOR_FOLDER="detector_covsoft_normalized_${DETECTOR_STEPS}"
  local pair_suffix=""
  if [[ "$CANDIDATE_PAIR" == "1" ]]; then
    pair_suffix="_pair"
  fi
  if [[ "$CANDIDATE_ONLINE_RENDER" == "1" ]]; then
    pair_suffix="${pair_suffix}_render_${ONLINE_RENDER_PROVENANCE}"
  fi
  CANDIDATE_FOLDER="${CAMBRIDGE_MATCHA_CANDIDATE_FOLDER:-detector_queryjoint_${CANDIDATE_OBJECTIVE}_normalized${pair_suffix}_${CANDIDATE_STEPS}}"
  COVERAGE_MAX_PER_GRID=$(((LANDMARK_COUNT * 3 + 31) / 32))
  COVERAGE_MAX_PER_DEPTH=$(((LANDMARK_COUNT * 3 + 7) / 8))
  CANDIDATE_Q1=$(((CANDIDATE_STEPS + 3) / 4))
  CANDIDATE_Q2=$(((CANDIDATE_STEPS + 1) / 2))
}

field_command() {
  local output_model="$1"
  local final_iteration="$2"
  local mv_views="$3"
  local pnp_start="$4"
  local geometry_start="$5"
  FIELD_CMD=(
    "$PYTHON" train_lafgs.py
    -s "$DATA_ROOT/$SCENE" -m "$output_model"
    -r 1 -f sp -g 2dgs --images processed --data_device cpu
    --load_iteration "$SOURCE_ITERATION"
    --iterations "$final_iteration"
    --train_seed 2026
    --train_phase full
    --loc_interval "$LOC_INTERVAL"
    --synthetic_view_ratio 0
    --synthetic_view_desc_weight 0
    --synthetic_view_reproj_weight 0
    --lafgs_stage_schedule pretrained_2dgs
    --lafgs_stage_bootstrap_until "$BOOTSTRAP_STEPS"
    --lafgs_stage_joint_until "$JOINT_STEPS"
    --lafgs_stage_bootstrap_base_weight 1.0
    --lafgs_stage_bootstrap_loc_weight 0.15
    --lafgs_stage_bootstrap_geometry_anchor_weight 0.10
    --lafgs_stage_joint_base_weight 0.50
    --lafgs_stage_joint_loc_weight 1.0
    --lafgs_stage_joint_geometry_anchor_weight 0.10
    --lafgs_stage_refine_base_weight 0.20
    --lafgs_stage_refine_loc_weight 1.5
    --lafgs_stage_refine_geometry_anchor_weight 0.05
    --landmark_path __all__
    --lafgs_mvinit_enabled
    --lafgs_mvinit_max_views "$mv_views"
    --lafgs_mvinit_min_observations "$MVINIT_MIN_OBSERVATIONS"
    --lafgs_mvinit_view_selection uniform
    --lafgs_mvinit_chunk_size 32768
    --lafgs_mvinit_feature_scale 0.5
    --no-lafgs_dense_feature_render
    --loc_direct_weight 1.0
    --loc_multiview_weight 0.1
    --loc_multiview_slots 2
    --loc_multiview_memory_device cpu
    --loc_full_bank_weight 0.1
    --loc_full_bank_hard_negatives 64
    --loc_full_bank_stats_chunk_size 256
    --loc_full_bank_max_landmarks "$FULL_BANK_LANDMARK_COUNT"
    --loc_full_bank_ignore_3d_radius "$SURFEL_RADIUS_P90_M"
    --loc_full_bank_ignore_uv_radius "$POSITIVE_RADIUS_PX"
    --loc_full_bank_nearby_as_positive
    --loc_full_bank_nearby_as_positive_until "$JOINT_STEPS"
    --loc_full_bank_pose_information_weight 0.5
    --loc_full_bank_pose_information_floor 0.2
    --loc_full_bank_pose_information_mode point_jacobian
    --loc_full_bank_pose_information_normalization quantile
    --loc_full_bank_fisher_translation_scale "$TRANSLATION_SCALE_M"
    --loc_full_bank_fisher_rotation_scale_degrees 2.0
    --loc_full_bank_fisher_measurement_sigma 1.0
    --loc_full_bank_balance_weight 0.25
    --loc_full_bank_balance_grid_size 4
    --loc_full_bank_balance_depth_bins 4
    --loc_clean_hard_negative_weight 0.5
    --loc_full_bank_clean_reproj_radius "$INLIER_SIGMA_PX"
    --loc_full_bank_clean_hard_negatives 16
    --loc_clean_field_start_iter "$JOINT_STEPS"
    --loc_clean_field_full_bank_weight_scale 0.5
    --loc_clean_field_clean_hn_weight_scale 1.5
    --loc_clean_field_balance_weight 0.5
    --loc_clean_field_pose_information_weight 0.5
    --lafgs_curriculum
    --lafgs_diff_pnp_start_iter "$pnp_start"
    --lafgs_geometry_start_iter "$geometry_start"
    --lafgs_topology_start_iter "$((FIELD_STEPS + 1))"
    --lafgs_diff_pnp_weight 0.05
    --lafgs_diff_pnp_reprojection_loss_type smooth_l1
    --lafgs_diff_pnp_reprojection_loss_delta "$REPROJECTION_SIGMA_PX"
    --lafgs_diff_pnp_max_correspondences 64
    --lafgs_diff_pnp_spatial_grid_size 4
    --lafgs_diff_pnp_point_weight_floor 0.05
    --lafgs_diff_pnp_translation_scale_m "$TRANSLATION_SCALE_M"
    --lafgs_diff_pnp_rotation_scale_degrees 2.0
    --lafgs_diff_pnp_utility_reprojection_error_scale "$INLIER_SIGMA_PX"
    --lafgs_diff_pnp_local_window_radius "$POSITIVE_RADIUS_PX"
    --lafgs_diff_pnp_geometry_local_window_radius "$POSITIVE_RADIUS_PX"
    --lafgs_diff_pnp_max_condition_number 100000
    --lafgs_diff_pnp_geometry_pose_guard_max_loss_increase -1
    --lafgs_diff_pnp_geometry_pose_guard_max_loss 5
    --lafgs_diff_pnp_geometry_pose_guard_softness 10
    --lafgs_diff_pnp_geometry_pose_guard_min_scale 0.05
    --lafgs_diff_pnp_feedback_pose_guard_max_loss_increase 30
    --lafgs_diff_pnp_feedback_pose_guard_max_loss 5
    --lafgs_diff_pnp_feedback_pose_guard_softness 10
    --lafgs_diff_pnp_feedback_pose_guard_min_scale 0.05
    --lafgs_diff_pnp_allow_geometry_grad
    --lafgs_diff_pnp_isolate_geometry_grad
    --allow_raw_xyz_geometry_grad
    --lafgs_diff_pnp_geometry_xyz_lr "$GEOMETRY_XYZ_LR"
    --lafgs_diff_pnp_geometry_max_step_m "$GEOMETRY_MAX_STEP_M"
    --lafgs_diff_pnp_geometry_reproj_weight 0.01
    --lafgs_diff_pnp_geometry_depth_anchor_weight 0.1
    --lafgs_diff_pnp_geometry_match_reproj_weight 0.5
    --lafgs_diff_pnp_geometry_match_max_reproj_error "$POSITIVE_RADIUS_PX"
    --lafgs_diff_pnp_geometry_max_reproj_error "$INLIER_SIGMA_PX"
    --loc_anchor_lr "$LOC_ANCHOR_LR"
    --surfel_loc_tangent_bound "$SURFEL_TANGENT_BOUND_M"
    --surfel_loc_normal_bound "$SURFEL_NORMAL_BOUND_M"
    --surfel_loc_radius_floor 1
    --surfel_loc_anchor_reg_weight 0.1
    --geometry_anchor_weight 0.1
    --lafgs_geometry_grad_clip_abs 10
    --loc_full_checkpoint_mode none
    --save_iterations "$final_iteration"
    --test_iterations "$final_iteration"
  )
}

run_smoke() {
  load_normalization
  local smoke_root="$EXPERIMENT_ROOT/smoke/$SCENE"
  local smoke_model="$smoke_root/model"
  local smoke_final=$((SOURCE_ITERATION + SMOKE_STEPS))
  if [[ ! -L "$smoke_model/point_cloud/iteration_${SOURCE_ITERATION}" ]]; then
    mkdir -p "$smoke_model/point_cloud"
    ln -s "$MODEL_ROOT/point_cloud/iteration_${SOURCE_ITERATION}" \
      "$smoke_model/point_cloud/iteration_${SOURCE_ITERATION}"
    cp "$MODEL_ROOT/cfg_args" "$smoke_model/cfg_args"
    cp "$MODEL_ROOT/artifact_provenance.json" "$smoke_model/artifact_provenance.json"
  fi
  field_command "$smoke_model" "$smoke_final" 2 "$SMOKE_PNP_START" "$SMOKE_GEOMETRY_START"
  if [[ "$SMOKE_TOPOLOGY" == "1" ]]; then
    FIELD_CMD+=(
      --enable_topology
      --lafgs_topology_start_iter "$SMOKE_GEOMETRY_START"
      --topology_stats_warmup 1
      --topology_update_interval 2
      --topology_min_observations 1
      --topology_split_quantile 0.5
      --topology_ambiguity_quantile 0.5
      --topology_min_repeatability 0
      --topology_min_radius 0
      --topology_growth_cap_per_event 0.00001
      --topology_total_point_budget_ratio 1.001
      --topology_cooldown_iterations 1
      --topology_max_mutation_events 1
      --topology_risk_commit_policy heldout_descriptor
      --topology_risk_holdout_size 2
      --topology_risk_epsilon 0
    )
  fi
  run_logged smoke "${FIELD_CMD[@]}"
  require_file "$smoke_model/point_cloud/iteration_${smoke_final}/point_cloud.ply"
  rm -rf "$smoke_model/point_cloud/iteration_${smoke_final}"
  touch "$smoke_root/smoke_passed"
}

run_field() {
  load_normalization
  local final_ply="$MODEL_ROOT/point_cloud/iteration_${MODEL_ITERATION}/point_cloud.ply"
  if [[ -f "$final_ply" ]]; then
    echo "[mainline] Skip completed field: $final_ply"
    return
  fi
  field_command "$MODEL_ROOT" "$MODEL_ITERATION" "$MVINIT_VIEWS" \
    "$PNP_START_STEPS" "$GEOMETRY_START_STEPS"
  run_logged field "${FIELD_CMD[@]}"
  require_file "$final_ply"
  require_file "$MODEL_ROOT/point_cloud/iteration_${MODEL_ITERATION}/loc_state.pt"
}

run_detector() {
  load_normalization
  require_file "$MODEL_ROOT/point_cloud/iteration_${MODEL_ITERATION}/point_cloud.ply"
  local final_detector="$MODEL_ROOT/$DETECTOR_FOLDER/${DETECTOR_STEPS}_detector.pth"
  if [[ -f "$final_detector" ]]; then
    echo "[mainline] Skip completed detector: $final_detector"
    return
  fi
  local cmd=(
    "$PYTHON" train_detector.py
    -s "$DATA_ROOT/$SCENE" -m "$MODEL_ROOT"
    -r 1 -f sp -g 2dgs --images processed --data_device cpu
    --iteration "$MODEL_ITERATION"
    --iterations "$DETECTOR_STEPS"
    --test_iterations "$DETECTOR_STEPS"
    --save_iterations "$DETECTOR_STEPS"
    --detector_folder "$DETECTOR_FOLDER"
    --landmark_num "$LANDMARK_COUNT"
    --landmark_k 32
    --sampling_mode coverage_preserving
    --detector_target_mode soft
    --soft_sigma "$REPROJECTION_SIGMA_PX"
    --min_loc_observations "$MIN_LOC_OBSERVATIONS"
    --utility_weight 1
    --candidate_reprojection_error_scale "$INLIER_SIGMA_PX"
    --candidate_cleanliness_weight 1
    --candidate_pose_info_weight 1
    --candidate_balance_weight 1
    --candidate_reliability_weight 0.25
    --candidate_utility_weight 0
    --pnp_voxel_size "$PNP_VOXEL_SIZE_M"
    --pnp_max_per_voxel 8
    --pnp_preserve_ratio 0.5
    --coverage_preserve_ratio 0.75
    --coverage_utility_ratio 0.10
    --coverage_high_confidence_ratio 0.10
    --coverage_grid_size 4
    --coverage_max_per_grid "$COVERAGE_MAX_PER_GRID"
    --coverage_depth_bins 4
    --coverage_max_per_depth_bin "$COVERAGE_MAX_PER_DEPTH"
    --coverage_allow_unbalanced_fallback
  )
  run_logged detector "${cmd[@]}"
  require_file "$final_detector"
  require_file "$MODEL_ROOT/$DETECTOR_FOLDER/sampled_idx.pkl"
  require_file "$MODEL_ROOT/$DETECTOR_FOLDER/landmark_meta.pt"
}

make_eval_config() {
  local variant="$1"
  local detector_iteration="$2"
  local output_cfg="$3"
  local use_calibration="${4:-0}"
  local detector_folder="$DETECTOR_FOLDER"
  local diagnostics_translation_scale_m="$TRANSLATION_SCALE_M"
  local reprojection_error_px="$RESIDUAL_CLIP_PX"
  local extra=()
  if [[ "$use_calibration" == "1" ]]; then
    load_calibration
    diagnostics_translation_scale_m="$CALIBRATED_TRANSLATION_SCALE_M"
    reprojection_error_px="$CALIBRATED_RESIDUAL_CLIP_PX"
  fi
  if [[ "$variant" == "candidate" ]]; then
    detector_folder="$CANDIDATE_FOLDER"
    extra=(
      --candidate_teacher_state_path "$CANDIDATE_FOLDER/${detector_iteration}_candidate_teacher_state.pt"
      --landmark_feature_override_path "$CANDIDATE_FOLDER/${detector_iteration}_candidate_teacher_state.pt"
      --override_landmark_features
      --use_candidate_dustbin
    )
    if [[ "$CANDIDATE_PAIR" == "1" ]]; then
      extra+=(
        --pair_measurement_state_path "$CANDIDATE_FOLDER/${detector_iteration}_candidate_teacher_state.pt"
        --use_pair_measurement
        --pair_measurement_fixed_candidate_count "$PAIR_FIXED_CANDIDATE_COUNT"
        --pair_measurement_refill_mode "$PAIR_REFILL_MODE"
        --pair_measurement_refill_voxel_size "$PNP_VOXEL_SIZE_M"
      )
      if [[ "$PAIR_COVARIANCE_REFINEMENT" == "1" ]]; then
        extra+=(--use_pair_measurement_covariance_refinement)
      fi
      if [[ "$PAIR_PROGRESSIVE_SAMPLING" == "1" ]]; then
        extra+=(--use_pair_measurement_progressive_sampling)
      fi
      if [[ "$PAIR_USE_OFFSET" == "0" ]]; then
        extra+=(--disable_pair_measurement_offset)
      fi
    fi
  fi
  "$PYTHON" scripts/make_stdloc_eval_cfg.py \
    --base_cfg configs/stdloc_cambridge.yaml \
    --output "$output_cfg" \
    --artifact_model_path "$MODEL_ROOT" \
    --detector_folder "$detector_folder" \
    --detector_iters "$detector_iteration" \
    --detect_num "$EVAL_DETECT_NUM" \
    --reprojection_error "$reprojection_error_px" \
    --nms "$NMS_RADIUS_PX" \
    --match_threshold 0 \
    --match_topk 1 \
    --max_matches_per_landmark 2 \
    --candidate_frontend_match_policy error \
    --diagnostics_grid_rows 4 \
    --diagnostics_grid_cols 4 \
    --diagnostics_voxel_size "$PNP_VOXEL_SIZE_M" \
    --diagnostics_task_translation_scale_m "$diagnostics_translation_scale_m" \
    --diagnostics_task_rotation_scale_degrees 2.0 \
    --summary_json "${output_cfg%.yaml}.json" \
    "${extra[@]}"
}

run_eval() {
  local subset="$1"
  local variant="$2"
  local detector_iteration="${3:-}"
  load_normalization
  if [[ -z "$detector_iteration" ]]; then
    if [[ "$variant" == "candidate" ]]; then
      detector_iteration="$CANDIDATE_STEPS"
    else
      detector_iteration="$DETECTOR_STEPS"
    fi
  fi
  local detector_folder="$DETECTOR_FOLDER"
  local result_variant="$variant"
  local use_calibration=0
  if [[ "$USE_CALIBRATED_FINAL_FRONTEND" == "1" ]] && {
    [[ "$variant" == "candidate" ]] || [[ "$subset" == "test" ]]
  }; then
    use_calibration=1
  fi
  if [[ "$variant" == "candidate" ]]; then
    detector_folder="$CANDIDATE_FOLDER"
    result_variant="$CANDIDATE_EVAL_TAG"
    require_file "$MODEL_ROOT/$detector_folder/${detector_iteration}_candidate_teacher_state.pt"
  fi
  require_file "$MODEL_ROOT/$detector_folder/${detector_iteration}_detector.pth"
  local calibration_tag=""
  if [[ "$use_calibration" == "1" ]]; then
    calibration_tag="_calibrated"
  fi
  local stable_dir="$EVALUATION_ROOT/${result_variant}_${detector_iteration}${calibration_tag}_${subset}"
  if [[ -f "$stable_dir/results_summary.json" ]]; then
    echo "[mainline] Skip completed evaluation: $stable_dir"
    return
  fi
  local cfg="$CONFIG_ROOT/${result_variant}_${detector_iteration}${calibration_tag}.yaml"
  make_eval_config "$variant" "$detector_iteration" "$cfg" "$use_calibration"
  local prefix="matcha-mainline-v2-${SCENE}-${result_variant}-${detector_iteration}${calibration_tag}-${subset}"
  local eval_args=()
  if [[ "$subset" == "validation" ]]; then
    eval_args=(
      --evaluation_camera_subset candidate_validation
      --candidate_validation_ratio 0.2
      --candidate_split_mode temporal_block
      --candidate_split_seed 2026
      --candidate_direct_validation_holdout
    )
  fi
  local cmd=(
    "$PYTHON" stdloc.py
    --model_path "$MODEL_ROOT"
    --iteration "$MODEL_ITERATION"
    --cfg "$cfg"
    --prefix "$prefix"
    --sparse_only
    "${eval_args[@]}"
  )
  run_logged "eval_${result_variant}_${detector_iteration}_${subset}" "${cmd[@]}"
  local result_dir
  result_dir="$(find "$REPO_ROOT/results" -maxdepth 1 -type d -name "${prefix}-*" \
    -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)"
  if [[ -z "$result_dir" || ! -f "$result_dir/results_summary.json" ]]; then
    echo "Could not locate completed STDLoc result for prefix $prefix" >&2
    exit 1
  fi
  mv "$result_dir" "$stable_dir"
}

run_calibration() {
  load_normalization
  run_eval validation detector "$DETECTOR_STEPS"
  if [[ -f "$CALIBRATION_JSON" ]]; then
    echo "[mainline] Reusing calibration: $CALIBRATION_JSON"
    return
  fi
  "$PYTHON" scripts/calibrate_candidate_scales.py \
    --results_json "$EVALUATION_ROOT/detector_${DETECTOR_STEPS}_validation/results.json" \
    --fallback_translation_scale_m "$TRANSLATION_SCALE_M" \
    --fallback_inlier_sigma_px "$INLIER_SIGMA_PX" \
    --output_json "$CALIBRATION_JSON" > "$LOG_ROOT/calibration.log"
  require_file "$CALIBRATION_JSON"
}

load_calibration() {
  run_calibration
  eval "$("$PYTHON" scripts/calibrate_candidate_scales.py \
    --results_json "$EVALUATION_ROOT/detector_${DETECTOR_STEPS}_validation/results.json" \
    --fallback_translation_scale_m "$TRANSLATION_SCALE_M" \
    --fallback_inlier_sigma_px "$INLIER_SIGMA_PX" \
    --output_json "$CALIBRATION_JSON" \
    --shell)"
}

run_candidate() {
  load_normalization
  load_calibration
  local final_state="$MODEL_ROOT/$CANDIDATE_FOLDER/${CANDIDATE_STEPS}_candidate_teacher_state.pt"
  if [[ -f "$final_state" ]]; then
    echo "[mainline] Skip completed candidate joint training: $final_state"
    return
  fi
  local base_detector="$MODEL_ROOT/$DETECTOR_FOLDER/${DETECTOR_STEPS}_detector.pth"
  local landmark_path="$MODEL_ROOT/$DETECTOR_FOLDER/sampled_idx.pkl"
  require_file "$base_detector"
  require_file "$landmark_path"
  local objective_args=()
  if [[ "$CANDIDATE_OBJECTIVE" == "exact" ]]; then
    objective_args=(
      --candidate_teacher_counterfactual_assignment_weight 0.5
      --candidate_teacher_counterfactual_bias_utility_weight 1.0
      --candidate_teacher_counterfactual_translation_utility_weight 0.25
      --candidate_teacher_counterfactual_utility_floor 0.0
      --candidate_teacher_counterfactual_target_mode all_false
      --candidate_teacher_counterfactual_exact_decision_set
      --candidate_teacher_counterfactual_require_positive_bias_gain
      --candidate_teacher_counterfactual_require_nonnegative_translation_gain
    )
  fi
  if [[ "$CANDIDATE_PAIR" == "1" ]]; then
    objective_args+=(
      --candidate_teacher_pair_measurement_inlier_weight 0.25
      --candidate_teacher_pair_measurement_nll_weight 0.10
      --candidate_teacher_pair_measurement_bias_weight 0.05
      --candidate_teacher_pair_measurement_covariance_weight 0
      --candidate_teacher_pair_measurement_residual_clip_px "$CALIBRATED_RESIDUAL_CLIP_PX"
      --candidate_teacher_pair_measurement_reference_translation_m "$CALIBRATED_TRANSLATION_SCALE_M"
      --candidate_teacher_pair_measurement_set_context
      --candidate_teacher_pair_measurement_geometry_context
    )
  fi
  if [[ "$CANDIDATE_ONLINE_RENDER" == "1" ]]; then
    objective_args+=(
      --candidate_teacher_online_render_ratio_start "$ONLINE_RENDER_RATIO_START"
      --candidate_teacher_online_render_ratio_end "$ONLINE_RENDER_RATIO_END"
      --candidate_teacher_online_render_ramp_start 0.0
      --candidate_teacher_online_render_ramp_end 1.0
      --candidate_teacher_online_render_alpha_min 0.35
      --candidate_teacher_online_render_alpha_max 0.65
      --candidate_teacher_online_render_sampling_mode "$ONLINE_RENDER_SAMPLING_MODE"
      --candidate_teacher_online_render_failure_ema 0.9
      --candidate_teacher_online_render_failure_temperature 1.0
      --candidate_teacher_online_render_uniform_floor 0.1
    )
    if [[ "$ONLINE_RENDER_PROVENANCE" != "none" ]]; then
      objective_args+=(
        --candidate_teacher_online_render_provenance_mode "$ONLINE_RENDER_PROVENANCE"
        --candidate_teacher_online_render_provenance_weight "$ONLINE_RENDER_PROVENANCE_WEIGHT"
        --candidate_teacher_online_render_provenance_topk 4
        --candidate_teacher_online_render_provenance_temperature 0.05
      )
    fi
  fi
  local cmd=(
    "$PYTHON" train_detector.py
    --model_path "$MODEL_ROOT"
    --iteration "$MODEL_ITERATION"
    --iterations "$CANDIDATE_STEPS"
    --test_iterations "$CANDIDATE_Q1" "$CANDIDATE_Q2" "$CANDIDATE_STEPS"
    --save_iterations "$CANDIDATE_Q1" "$CANDIDATE_Q2" "$CANDIDATE_STEPS"
    --detector_folder "$CANDIDATE_FOLDER"
    --landmark_num "$LANDMARK_COUNT"
    --precomputed_landmark_path "$landmark_path"
    --sparse_candidate_teacher
    --candidate_teacher_detector_init_path "$base_detector"
    --candidate_teacher_optimize_features
    --candidate_teacher_detector_lr 0.0001
    --candidate_teacher_feature_lr 0.0002
    --candidate_teacher_dustbin_lr 0.0002
    --candidate_teacher_detect_num "$TRAIN_DETECT_NUM"
    --candidate_teacher_nms_radius "$NMS_RADIUS_PX"
    --candidate_teacher_match_mode topk
    --candidate_teacher_match_topk 1
    --candidate_teacher_positive_radius_px "$POSITIVE_RADIUS_PX"
    --candidate_teacher_negative_radius_px "$NEGATIVE_RADIUS_PX"
    --candidate_teacher_max_positives 4
    --candidate_teacher_hard_negatives 16
    --candidate_teacher_match_temperature 0.1
    --candidate_teacher_match_margin 0.5
    --candidate_teacher_pair_weight 0
    --candidate_teacher_hard_negative_weight 0.5
    --candidate_teacher_assignment_weight 1
    "${objective_args[@]}"
    --candidate_teacher_assignment_mode multi_positive
    --candidate_teacher_assignment_temperature 0.05
    --candidate_teacher_assignment_margin 0.05
    --candidate_teacher_dustbin_weight 0.25
    --candidate_teacher_reprojection_sigma_px "$REPROJECTION_SIGMA_PX"
    --candidate_teacher_detector_match_weight 1
    --candidate_teacher_detector_target_source final_or_geometric
    --candidate_teacher_geometry_weight 0.1
    --candidate_teacher_coverage_weight 0.1
    --candidate_teacher_base_detector_weight 0.05
    --candidate_teacher_feature_anchor_weight 0.01
    --candidate_teacher_validation_ratio 0.2
    --candidate_teacher_split_mode temporal_block
    --candidate_teacher_split_seed 2026
    --candidate_teacher_assignment_pose_information_mode conditional_translation
    --candidate_teacher_assignment_pose_information_weight 0.10
    --candidate_teacher_map_cleanliness_weight 0.5
    --candidate_teacher_map_translation_information_weight 0.05
    --candidate_teacher_map_translation_trace_weight 0.02
    --candidate_teacher_map_translation_condition_weight 0.01
    --candidate_teacher_map_bias_weight 0.75
    --candidate_teacher_map_capacity_weight 0.10
    --candidate_teacher_map_fisher_translation_scale "$CALIBRATED_TRANSLATION_SCALE_M"
    --candidate_teacher_map_fisher_rotation_scale_degrees 2.0
    --candidate_teacher_map_fisher_measurement_sigma_px "$REPROJECTION_SIGMA_PX"
    --candidate_teacher_map_fisher_residual_clip_px "$CALIBRATED_RESIDUAL_CLIP_PX"
    --candidate_teacher_map_fisher_inlier_sigma_px "$CALIBRATED_INLIER_SIGMA_PX"
    --candidate_teacher_map_fisher_condition_target 100
    --candidate_teacher_map_bias_huber_delta "$CALIBRATED_BIAS_HUBER_DELTA"
    --candidate_teacher_map_bias_clip "$CALIBRATED_BIAS_CLIP"
    --candidate_teacher_map_max_matches_per_landmark 2
    --candidate_teacher_grid_rows 4
    --candidate_teacher_grid_cols 4
    --candidate_teacher_depth_bins 4
  )
  run_logged "candidate_${CANDIDATE_FOLDER}" "${cmd[@]}"
  require_file "$final_state"
  require_file "$MODEL_ROOT/$CANDIDATE_FOLDER/${CANDIDATE_STEPS}_detector.pth"
}

select_candidate_checkpoint() {
  local selection_json="$EVALUATION_ROOT/candidate_selection.json"
  "$PYTHON" scripts/select_candidate_checkpoint.py \
    --evaluation_root "$EVALUATION_ROOT" \
    --candidate_tag "$CANDIDATE_EVAL_TAG" \
    --iterations "$CANDIDATE_Q1" "$CANDIDATE_Q2" "$CANDIDATE_STEPS" \
    --output_json "$selection_json" \
    --shell
}

case "$MODE" in
  prepare) prepare_scene ;;
  smoke) run_smoke ;;
  field) run_field ;;
  detector) run_detector ;;
  calibrate) run_calibration ;;
  candidate) run_candidate ;;
  eval) run_eval "$EVAL_SUBSET" "$EVAL_VARIANT" "$EVAL_ITERATION" ;;
  select)
    load_normalization
    select_candidate_checkpoint
    ;;
  all)
    run_field
    run_detector
    run_calibration
    run_eval test detector "$DETECTOR_STEPS"
    run_candidate
    run_eval validation candidate "$CANDIDATE_Q1"
    run_eval validation candidate "$CANDIDATE_Q2"
    run_eval validation candidate "$CANDIDATE_STEPS"
    eval "$(select_candidate_checkpoint)"
    run_eval test candidate "$SELECTED_CANDIDATE_ITERATION"
    ;;
esac
