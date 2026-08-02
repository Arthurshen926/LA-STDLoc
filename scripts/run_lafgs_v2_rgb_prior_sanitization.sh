#!/usr/bin/env bash
set -euo pipefail

# Frozen RGB Gaussian prior -> robust KCS/GWFF -> query-specific Stage-A.
# All mapping frames are training data and all Cambridge test frames are the
# development evaluation set. Deployment is exactly one native SuperPoint
# pass, cosine top-1 retrieval, and one RANSAC/PnP solve.

if [[ $# -ne 3 ]]; then
  echo "Usage: bash $0 <rgb_2dgs|rgb_nosky|rgb_sky_dirty|vanilla_2dgs|vanilla_3dgs|feature_stripped> <gpu> <bootstrap|stage_a|statistics|sanitize|eval|all>" >&2
  exit 2
fi

VARIANT="$1"
GPU="$2"
MODE="$3"
case "$VARIANT" in
  rgb_2dgs)
    MODEL_NAME="rgb_only_2dgs_stdloc"
    GAUSSIAN_TYPE="2dgs"
    SH_DEGREE=3
    PRIOR_KIND="rgb_only"
    ;;
  rgb_nosky)
    MODEL_NAME="rgb_only_3dgs_nosky"
    GAUSSIAN_TYPE="3dgs"
    SH_DEGREE=0
    PRIOR_KIND="rgb_only"
    ;;
  rgb_sky_dirty)
    MODEL_NAME="rgb_only_3dgs_sky_dirty"
    GAUSSIAN_TYPE="3dgs"
    SH_DEGREE=0
    PRIOR_KIND="rgb_only"
    ;;
  vanilla_2dgs)
    MODEL_NAME="vanilla_2dgs"
    GAUSSIAN_TYPE="2dgs"
    SH_DEGREE=3
    PRIOR_KIND="rgb_only"
    ;;
  vanilla_3dgs)
    MODEL_NAME="vanilla_3dgs"
    GAUSSIAN_TYPE="3dgs"
    SH_DEGREE=3
    PRIOR_KIND="rgb_only"
    ;;
  feature_stripped)
    MODEL_NAME="stdloc_feature_stripped_3dgs"
    GAUSSIAN_TYPE="3dgs"
    SH_DEGREE=3
    PRIOR_KIND="feature_stripped"
    ;;
  *) echo "Unknown prior variant: $VARIANT" >&2; exit 2 ;;
esac
case "$GPU" in 0|1|2) ;; *) echo "GPU must be 0, 1, or 2" >&2; exit 2 ;; esac
case "$MODE" in bootstrap|stage_a|statistics|sanitize|eval|all) ;; *) echo "Unknown mode: $MODE" >&2; exit 2 ;; esac

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/root/miniconda3/envs/cybersim_agent/bin/python}"
DATA_ROOT="${CAMBRIDGE_DATA_ROOT:-/mnt/pool/sqy/Cambridge_stdloc}"
EXPERIMENT_ROOT="${LAFGS_SANITIZATION_ROOT:-/mnt/pool/sqy/stdloc_lafgs_rgb_prior_sanitization_20260725}"
SCENE="${LAFGS_CAMBRIDGE_SCENE:-OldHospital}"
case "$SCENE" in
  GreatCourt|KingsCollege|OldHospital|ShopFacade|StMarysChurch) ;;
  *) echo "Unsupported Cambridge scene: $SCENE" >&2; exit 2 ;;
esac
MODEL_ROOT="${LAFGS_SANITIZATION_MODEL_ROOT:-$EXPERIMENT_ROOT/$SCENE/$MODEL_NAME}"
RUN_TAG="${LAFGS_SANITIZATION_RUN_TAG:-$VARIANT}"
SCAFFOLD_BUDGET="${LAFGS_SANITIZATION_SCAFFOLD_BUDGET:-48000}"
SANITIZED_BUDGET="${LAFGS_SANITIZATION_FINAL_BUDGET:-32000}"
MAX_TRAIN_VIEWS="${LAFGS_MAX_TRAIN_VIEWS:-0}"
STAGE_A_PROFILE="${LAFGS_STAGE_A_PROFILE:-combined}"
STAGE_A_STEPS="${LAFGS_STAGE_A_STEPS:-2500}"
SANITIZATION_SOURCE_STEP="${LAFGS_SANITIZATION_SOURCE_STEP:-$STAGE_A_STEPS}"
STATISTICS_CHECKPOINT_STEP="${LAFGS_STATISTICS_CHECKPOINT_STEP:-$SANITIZATION_SOURCE_STEP}"
if [[ -n "${LAFGS_GEOMETRY_TEACHER_IDENTITY_MODE:-}" ]]; then
  GEOMETRY_TEACHER_IDENTITY_MODE="$LAFGS_GEOMETRY_TEACHER_IDENTITY_MODE"
elif [[ "$GAUSSIAN_TYPE" == "2dgs" ]]; then
  GEOMETRY_TEACHER_IDENTITY_MODE="track_first_provenance"
else
  GEOMETRY_TEACHER_IDENTITY_MODE="track_first"
fi
if [[ -n "${LAFGS_GEOMETRY_TEACHER_TAG:-}" ]]; then
  GEOMETRY_TEACHER_TAG="$LAFGS_GEOMETRY_TEACHER_TAG"
else
  case "$GEOMETRY_TEACHER_IDENTITY_MODE" in
    track_first_provenance) GEOMETRY_TEACHER_TAG="g3_track_provenance_v1" ;;
    track_first) GEOMETRY_TEACHER_TAG="g2_track_first_v1" ;;
    gt_clean_map_top1) GEOMETRY_TEACHER_TAG="g1_gt_clean_v1" ;;
    map_top1) GEOMETRY_TEACHER_TAG="g0_map_top1_v1" ;;
  esac
fi
SANITIZATION_MODES="${LAFGS_SANITIZATION_MODES:-loc hard_geo_reject_only hard_geo_loc loc_query_coverage hard_geo_loc_query_coverage}"
KCS_EXTENT_QUANTILE="${LAFGS_SANITIZATION_KCS_EXTENT_QUANTILE:-0.01}"
SUPPORT_MASK_POLICY="${LAFGS_SANITIZATION_SUPPORT_MASK_POLICY:-support_rgb_only}"
if ! [[ "$SCAFFOLD_BUDGET" =~ ^[1-9][0-9]*$ ]]; then
  echo "LAFGS_SANITIZATION_SCAFFOLD_BUDGET must be a positive integer" >&2
  exit 2
fi
if ! [[ "$SANITIZED_BUDGET" =~ ^[1-9][0-9]*$ ]]; then
  echo "LAFGS_SANITIZATION_FINAL_BUDGET must be a positive integer" >&2
  exit 2
fi
if ! [[ "$MAX_TRAIN_VIEWS" =~ ^[0-9]+$ ]]; then
  echo "LAFGS_MAX_TRAIN_VIEWS must be a non-negative integer" >&2
  exit 2
fi
if (( SANITIZED_BUDGET > SCAFFOLD_BUDGET )); then
  echo "Sanitized budget cannot exceed scaffold budget" >&2
  exit 2
fi
case "$SUPPORT_MASK_POLICY" in
  support_rgb_only|deployment_post_filter) ;;
  *) echo "Unknown support mask policy: $SUPPORT_MASK_POLICY" >&2; exit 2 ;;
esac
case "$STAGE_A_PROFILE" in
  pure|semidense|protected|combined) ;;
  *) echo "Unknown Stage-A profile: $STAGE_A_PROFILE" >&2; exit 2 ;;
esac
case "$GEOMETRY_TEACHER_IDENTITY_MODE" in
  map_top1|gt_clean_map_top1|track_first|track_first_provenance) ;;
  *) echo "Unknown geometry teacher identity mode: $GEOMETRY_TEACHER_IDENTITY_MODE" >&2; exit 2 ;;
esac
if [[ "${LAFGS_ALLOW_LEGACY_GEOMETRY_TEACHER:-0}" != "1" && "$GEOMETRY_TEACHER_IDENTITY_MODE" == "map_top1" ]]; then
  echo "Canonical sanitization refuses map_top1; set LAFGS_ALLOW_LEGACY_GEOMETRY_TEACHER=1 only for a labeled G0 control" >&2
  exit 2
fi
for sanitization_mode in $SANITIZATION_MODES; do
  case "$sanitization_mode" in
    loc|loc_query_coverage|hard_geo_reject_only|hard_geo_loc|hard_geo_loc_query_coverage) ;;
    loc_geo|loc_geo_coverage)
      if [[ "${LAFGS_ALLOW_LEGACY_LOC_GEO:-0}" != "1" ]]; then
        echo "Canonical sanitization refuses legacy mode: $sanitization_mode" >&2
        exit 2
      fi
      ;;
    *) echo "Unknown sanitization mode: $sanitization_mode" >&2; exit 2 ;;
  esac
done
geometry_teacher_track_lgcv_args=()
if [[ "${LAFGS_GEOMETRY_TEACHER_TRACK_LGCV:-0}" == "1" ]]; then
  geometry_teacher_track_lgcv_args+=(--geometry_teacher_track_lgcv)
fi
geometry_teacher_track_chain_args=()
if [[ "${LAFGS_GEOMETRY_TEACHER_TRACK_ALLOW_CHAIN_TRACKS:-0}" == "1" ]]; then
  geometry_teacher_track_chain_args+=(--geometry_teacher_track_allow_chain_tracks)
fi
for value in "$STAGE_A_STEPS" "$SANITIZATION_SOURCE_STEP" "$STATISTICS_CHECKPOINT_STEP"; do
  if ! [[ "$value" =~ ^[1-9][0-9]*$ ]]; then
    echo "Stage-A and sanitization checkpoint steps must be positive integers" >&2
    exit 2
  fi
done
if (( SANITIZATION_SOURCE_STEP > STAGE_A_STEPS || STATISTICS_CHECKPOINT_STEP > STAGE_A_STEPS )); then
  echo "Requested source/statistics checkpoint exceeds Stage-A training steps" >&2
  exit 2
fi
RUN_ROOT="$EXPERIMENT_ROOT/$SCENE/runs/$RUN_TAG"
BOOTSTRAP_DIR="$RUN_ROOT/bootstrap"
if [[ "$STAGE_A_PROFILE" == "combined" && "$STAGE_A_STEPS" == "2500" ]]; then
  STAGE_A_DIR="$RUN_ROOT/stage_a_2500"
else
  STAGE_A_DIR="$RUN_ROOT/stage_a_${STAGE_A_PROFILE}_${STAGE_A_STEPS}"
fi
STATISTICS_DIR="$RUN_ROOT/statistics_${STAGE_A_PROFILE}_${STATISTICS_CHECKPOINT_STEP}_frozen_${GEOMETRY_TEACHER_TAG}"
CONFIG_ROOT="$RUN_ROOT/configs"
LOG_ROOT="$RUN_ROOT/logs"
RESULT_ROOT="$RUN_ROOT/results"
STDLOC_RESULTS_ROOT="$RUN_ROOT/stdloc_results"
QUERY_CACHE="${LAFGS_QUERY_CACHE_PATH:-$RUN_ROOT/query_cache_native_fullres_k2048.pt}"
if (( MAX_TRAIN_VIEWS > 0 )); then
  VISIBILITY_CACHE="$RUN_ROOT/visibility_${SCAFFOLD_BUDGET}_native_${MAX_TRAIN_VIEWS}views.pt"
else
  VISIBILITY_CACHE="$RUN_ROOT/visibility_${SCAFFOLD_BUDGET}_native.pt"
fi
PRIOR_MANIFEST="$MODEL_ROOT/rgb_prior_manifest.json"
LANDMARK_IDS="$BOOTSTRAP_DIR/sampled_idx.pkl"
LANDMARK_META="$BOOTSTRAP_DIR/landmark_meta.pt"
BOOTSTRAP_STATE="$BOOTSTRAP_DIR/0_lafgs_map_state.pt"
STAGE_A_FINAL_STATE="$STAGE_A_DIR/${STAGE_A_STEPS}_lafgs_map_state.pt"
SANITIZATION_SOURCE_STATE="$STAGE_A_DIR/${SANITIZATION_SOURCE_STEP}_lafgs_map_state.pt"
STATISTICS_SOURCE_STATE="$STAGE_A_DIR/${STATISTICS_CHECKPOINT_STEP}_lafgs_map_state.pt"
STATISTICS_PATH="$STATISTICS_DIR/landmark_statistics_full.pt"

export CUDA_VISIBLE_DEVICES="$GPU"
export CUDA_HOME=/usr/local/cuda-11.8
export PATH="/root/miniconda3/envs/cybersim_agent/bin:$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="/root/miniconda3/envs/cybersim_agent/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$REPO_ROOT"
export PYTHONHASHSEED=2026
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export STDLOC_CAMERA_LOADER_WORKERS=0
export STDLOC_RESULTS_ROOT

mkdir -p "$BOOTSTRAP_DIR" "$STAGE_A_DIR" "$STATISTICS_DIR" "$CONFIG_ROOT" "$LOG_ROOT" "$RESULT_ROOT" "$STDLOC_RESULTS_ROOT"
cd "$REPO_ROOT"

PRIOR_ARGS=(--require_rgb_prior_manifest --rgb_prior_manifest_path "$PRIOR_MANIFEST")
if [[ "$PRIOR_KIND" == "feature_stripped" ]]; then
  PRIOR_ARGS+=(--allow_feature_stripped_prior)
fi

run_logged() {
  local name="$1"
  shift
  printf '%q ' "$@" > "$LOG_ROOT/${name}.command.sh"
  printf '\n' >> "$LOG_ROOT/${name}.command.sh"
  "$@" 2>&1 | tee "$LOG_ROOT/${name}.log"
}

common_train_args=(
  --model_path "$MODEL_ROOT" --source_path "$DATA_ROOT/$SCENE"
  --images processed --data_device cpu --gaussian_type "$GAUSSIAN_TYPE" --sh_degree "$SH_DEGREE"
  --feature_type sp --resolution 1 --longest_edge 0 --norm_before_render
  --load_iteration 30000 "${PRIOR_ARGS[@]}"
  --query_feature_contract native_resized_input
  --query_cache_path "$QUERY_CACHE"
  --visibility_cache_path "$VISIBILITY_CACHE" --visibility_mode rasterizer
  --objective hard --observation_source native --native_keypoint_count 2048
  --max_observations 2048 --validation_observations 2048
  --native_sampling_mode detector_grid --native_association_radius_px 2
  --native_anchor_aux_weight 0 --generic_proposal_count 0 --distill_budget 0
  --validation_ratio 0 --split_mode stratified_temporal_block
  --split_seed 2026 --train_seed 2026
  --max_train_views "$MAX_TRAIN_VIEWS"
  --mv_weight 0 --local_weight 0 --dustbin_weight 0
  --geometry_weight 0 --pose_weight 0 --pose_gradient_mode off
)

bootstrap() {
  [[ -f "$PRIOR_MANIFEST" ]] || { echo "Missing prior manifest: $PRIOR_MANIFEST" >&2; exit 1; }
  if [[ -f "$BOOTSTRAP_STATE" && -f "$LANDMARK_IDS" && -f "$LANDMARK_META" ]]; then
    echo "Reusing bootstrap: $BOOTSTRAP_STATE"
    return
  fi
  run_logged bootstrap \
    "$PYTHON" train_lafgs_map.py \
    "${common_train_args[@]}" --query_cache_policy reuse_or_build \
    --output_dir "$BOOTSTRAP_DIR" \
    --scaffold_mode ulf_robust_consensus \
    --generated_landmark_path "$BOOTSTRAP_DIR/robust_ids.pkl" \
    --regenerate_scaffold --scaffold_budget "$SCAFFOLD_BUDGET" --scaffold_seed 2026 \
    --scaffold_min_opacity 0 --scaffold_opacity_keep_quantile 0.1 \
    --initialization_mode ulf_robust_geometry \
    --ulf_consensus_keypoints 2048 --ulf_consensus_radius_px 1 \
    --ulf_consensus_min_visible_views 4 --ulf_consensus_min_votes 2 \
    --ulf_consensus_min_rate 0.01 --ulf_consensus_view_bins 4 \
    --ulf_consensus_min_distinct_view_bins 2 \
    --ulf_consensus_trajectory_bins 4 \
    --ulf_consensus_min_distinct_trajectory_bins 2 \
    --ulf_consensus_independent_bin_scoring \
    --ulf_consensus_allow_nonconsensus_fallback \
    --ulf_consensus_extent_quantile "$KCS_EXTENT_QUANTILE" \
    --ulf_support_view_sampling uniform --ulf_support_mask_policy "$SUPPORT_MASK_POLICY" \
    --ulf_consensus_max_views 0 --ulf_fusion_max_views 0 \
    --ulf_fusion_min_cosine 0 --ulf_fusion_view_bins 4 \
    --ulf_fusion_descriptor_trim_fraction 0.1 \
    --ulf_fusion_descriptor_min_cosine -1 \
    --ulf_fusion_trim_histogram_bins 64 --ulf_fusion_reference_mode mean \
    --no-ulf_fusion_exact_bin_balance \
    --no-native_outcome_mode --retrieval_weight 0 --trust_weight 0 \
    --steps 0 --save_steps 0
}

stage_a() {
  bootstrap
  if [[ -f "$STAGE_A_FINAL_STATE" ]]; then
    echo "Reusing Stage-A: $STAGE_A_FINAL_STATE"
    return
  fi
  local profile_args=()
  if [[ "$STAGE_A_PROFILE" == "semidense" || "$STAGE_A_PROFILE" == "combined" ]]; then
    profile_args+=(
      --native_semidense_weight 0.01 --native_semidense_start_step 500
      --native_semidense_interval 5 --native_semidense_max_anchors 64
      --native_semidense_neighbors 8 --native_semidense_local_radius_px 8
      --native_semidense_target_sigma_px 2 --native_semidense_temperature 0.07
      --native_semidense_lgcv_weight 0 --native_semidense_protected_v2
      --native_semidense_measurement_min_reprojection_px 2
      --native_semidense_measurement_max_reprojection_px 8
      --native_semidense_surface_point_plane_m 0.03
      --native_semidense_surface_max_distance_m 0.15
      --native_semidense_surface_normal_cosine 0.95
      --native_semidense_projected_neighbor_radius_px 64
      --native_semidense_local_identity_weight 0.25
      --native_semidense_margin_preservation_weight 4
      --native_semidense_reference_refresh_steps 500
      --native_semidense_alternate_global
      --native_semidense_max_gradient_ratio 0.25
    )
  else
    profile_args+=(--native_semidense_weight 0)
  fi
  if [[ "$STAGE_A_PROFILE" == "protected" || "$STAGE_A_PROFILE" == "combined" ]]; then
    profile_args+=(
      --native_protected_set_weight 0.02
      --native_protected_set_start_step 500
      --native_protected_set_interval 5
      --native_protected_set_refresh_visits 1
      --native_protected_set_ransac_seed 0
      --native_protected_set_ransac_reprojection_px 8
      --native_protected_set_ransac_max_iterations 5000
      --native_protected_set_ransac_min_iterations 100
      --native_protected_set_max_pose_error_cm
      "${LAFGS_PROTECTED_MAX_POSE_ERROR_CM:-100}"
      --native_protected_set_max_useful 96
      --native_protected_set_max_harmful 96
      --native_protected_set_grid_rows 4 --native_protected_set_grid_cols 4
      --native_protected_set_depth_bins 4
      --native_protected_set_surface_voxel_m 0.25
      --native_protected_set_max_per_surface_group 2
    )
  else
    profile_args+=(--native_protected_set_weight 0)
  fi
  local save_steps=("$STAGE_A_STEPS")
  if (( STAGE_A_STEPS > 1000 )); then
    save_steps=(1000 "$STAGE_A_STEPS")
  fi
  run_logged stage_a \
    "$PYTHON" train_lafgs_map.py \
    "${common_train_args[@]}" --query_cache_policy readonly \
    --output_dir "$STAGE_A_DIR" --scaffold_mode file \
    --landmark_path "$LANDMARK_IDS" \
    --initial_state_path "$BOOTSTRAP_STATE" \
    --initial_state_blend 1 --initial_state_alignment exact \
    --initialization_mode ulf_robust_geometry \
    --native_outcome_mode --native_nce_weight 0 \
    --native_keep_weight 1 --native_keep_margin 0.05 \
    --native_swap_weight 1 --native_swap_margin 0.05 \
    --native_miss_weight 1 --native_miss_margin 0.05 \
    --native_reject_weight 0 --native_reject_threshold 0 \
    --native_global_attractor_weight 0.25 \
    --native_global_attractor_min_incoming 4 \
    --native_global_attractor_support_power 0.5 \
    --native_global_attractor_max_score 4 \
    "${profile_args[@]}" \
    --retrieval_weight 1 --trust_weight 0.02 \
    --feature_lr 5e-5 --weight_decay 1e-4 \
    --hypothesis_topk 32 --positive_radius_px 2 --negative_radius_px 8 \
    --steps "$STAGE_A_STEPS" --save_steps "${save_steps[@]}" --log_interval 100
  if [[ ! -f "$STAGE_A_FINAL_STATE" ]]; then
    echo "Stage-A completed without final checkpoint: $STAGE_A_FINAL_STATE" >&2
    exit 1
  fi
}

offline_statistics() {
  stage_a
  [[ -f "$STATISTICS_SOURCE_STATE" ]] || {
    echo "Missing frozen statistics checkpoint: $STATISTICS_SOURCE_STATE" >&2
    exit 1
  }
  if [[ -f "$STATISTICS_PATH" ]]; then
    echo "Reusing frozen independent statistics: $STATISTICS_PATH"
    "$PYTHON" scripts/verify_geometry_teacher_statistics.py \
      --statistics "$STATISTICS_PATH" \
      --expected_identity "$GEOMETRY_TEACHER_IDENTITY_MODE"
    return
  fi
  run_logged "statistics_${STAGE_A_PROFILE}_${STATISTICS_CHECKPOINT_STEP}" \
    "$PYTHON" train_lafgs_map.py \
    "${common_train_args[@]}" --query_cache_policy reuse_or_build \
    --output_dir "$STATISTICS_DIR" --scaffold_mode file \
    --landmark_path "$LANDMARK_IDS" \
    --initial_state_path "$STATISTICS_SOURCE_STATE" \
    --initial_state_blend 1 --initial_state_alignment exact \
    --initialization_mode ulf_robust_geometry \
    --native_outcome_mode --retrieval_weight 0 --trust_weight 0 \
    --positive_radius_px 2 --negative_radius_px 8 \
    --save_landmark_statistics --save_independent_geometry_teacher \
    --statistics_observations 2048 \
    --geometry_teacher_identity_mode "$GEOMETRY_TEACHER_IDENTITY_MODE" \
    --geometry_teacher_min_similarity "${LAFGS_GEOMETRY_TEACHER_MIN_SIMILARITY:-0.7}" \
    --geometry_teacher_min_margin "${LAFGS_GEOMETRY_TEACHER_MIN_MARGIN:-0.03}" \
    --geometry_teacher_min_views "${LAFGS_GEOMETRY_TEACHER_MIN_VIEWS:-3}" \
    --geometry_teacher_min_view_bins "${LAFGS_GEOMETRY_TEACHER_MIN_VIEW_BINS:-2}" \
    --geometry_teacher_min_parallax_deg "${LAFGS_GEOMETRY_TEACHER_MIN_PARALLAX_DEG:-1}" \
    --geometry_teacher_parallax_quantile "${LAFGS_GEOMETRY_TEACHER_PARALLAX_QUANTILE:-0.75}" \
    --geometry_teacher_max_reprojection_px "${LAFGS_GEOMETRY_TEACHER_MAX_REPROJECTION_PX:-2}" \
    --geometry_teacher_max_covariance_trace_m2 "${LAFGS_GEOMETRY_TEACHER_MAX_COVARIANCE_TRACE_M2:-0.01}" \
    --geometry_teacher_max_rendered_depth_residual_m "${LAFGS_GEOMETRY_TEACHER_MAX_DEPTH_RESIDUAL_M:-0.15}" \
    --geometry_teacher_min_rendered_depth_observations "${LAFGS_GEOMETRY_TEACHER_MIN_DEPTH_OBSERVATIONS:-2}" \
    --geometry_teacher_track_pair_neighbors "${LAFGS_GEOMETRY_TEACHER_TRACK_PAIR_NEIGHBORS:-6}" \
    --geometry_teacher_track_min_similarity "${LAFGS_GEOMETRY_TEACHER_TRACK_MIN_SIMILARITY:-0.65}" \
    --geometry_teacher_track_min_margin "${LAFGS_GEOMETRY_TEACHER_TRACK_MIN_MARGIN:-0.01}" \
    --geometry_teacher_track_max_epipolar_error_px "${LAFGS_GEOMETRY_TEACHER_TRACK_MAX_EPIPOLAR_ERROR_PX:-2}" \
    --geometry_teacher_track_epipolar_candidate_topk "${LAFGS_GEOMETRY_TEACHER_TRACK_EPIPOLAR_CANDIDATE_TOPK:-1}" \
    --geometry_teacher_track_epipolar_recovered_min_similarity "${LAFGS_GEOMETRY_TEACHER_TRACK_EPIPOLAR_RECOVERED_MIN_SIMILARITY:--1}" \
    --geometry_teacher_track_epipolar_recovered_min_margin "${LAFGS_GEOMETRY_TEACHER_TRACK_EPIPOLAR_RECOVERED_MIN_MARGIN:--1}" \
    "${geometry_teacher_track_chain_args[@]}" \
    "${geometry_teacher_track_lgcv_args[@]}" \
    --geometry_teacher_track_lgcv_neighbors "${LAFGS_GEOMETRY_TEACHER_TRACK_LGCV_NEIGHBORS:-8}" \
    --geometry_teacher_track_lgcv_support_threshold "${LAFGS_GEOMETRY_TEACHER_TRACK_LGCV_SUPPORT_THRESHOLD:-4}" \
    --geometry_teacher_track_lgcv_angle_cosine "${LAFGS_GEOMETRY_TEACHER_TRACK_LGCV_ANGLE_COSINE:-0.9659}" \
    --geometry_teacher_track_lgcv_scale_threshold "${LAFGS_GEOMETRY_TEACHER_TRACK_LGCV_SCALE_THRESHOLD:-0.1}" \
    --geometry_teacher_track_lgcv_scale_limit "${LAFGS_GEOMETRY_TEACHER_TRACK_LGCV_SCALE_LIMIT:-3}" \
    --geometry_teacher_track_lgcv_maximum_edge_px "${LAFGS_GEOMETRY_TEACHER_TRACK_LGCV_MAXIMUM_EDGE_PX:-50}" \
    --geometry_teacher_track_lgcv_minimum_matches "${LAFGS_GEOMETRY_TEACHER_TRACK_LGCV_MINIMUM_MATCHES:-8}" \
    --geometry_teacher_track_lgcv_mode "${LAFGS_GEOMETRY_TEACHER_TRACK_LGCV_MODE:-hard}" \
    --geometry_teacher_track_lgcv_confidence_floor "${LAFGS_GEOMETRY_TEACHER_TRACK_LGCV_CONFIDENCE_FLOOR:-0.25}" \
    --geometry_teacher_track_assignment_max_distance_m "${LAFGS_GEOMETRY_TEACHER_TRACK_ASSIGNMENT_MAX_DISTANCE_M:-0.2}" \
    --geometry_teacher_track_assignment_min_margin_m "${LAFGS_GEOMETRY_TEACHER_TRACK_ASSIGNMENT_MIN_MARGIN_M:-0}" \
    --geometry_teacher_provenance_topk "${LAFGS_GEOMETRY_TEACHER_PROVENANCE_TOPK:-4}" \
    --geometry_teacher_provenance_min_consensus_rate "${LAFGS_GEOMETRY_TEACHER_PROVENANCE_MIN_CONSENSUS_RATE:-0.35}" \
    --geometry_teacher_provenance_min_views "${LAFGS_GEOMETRY_TEACHER_PROVENANCE_MIN_VIEWS:-2}" \
    --geometry_teacher_provenance_group_max_landmarks "${LAFGS_GEOMETRY_TEACHER_PROVENANCE_GROUP_MAX_LANDMARKS:-1}" \
    --geometry_teacher_provenance_group_min_relative_mass "${LAFGS_GEOMETRY_TEACHER_PROVENANCE_GROUP_MIN_RELATIVE_MASS:-0.25}" \
    --geometry_teacher_provenance_group_min_consensus_rate "${LAFGS_GEOMETRY_TEACHER_PROVENANCE_GROUP_MIN_CONSENSUS_RATE:-0.10}" \
    --save_track_micro_anchor_payload \
    --steps 0 --save_steps 0
  [[ -f "$STATISTICS_PATH" ]] || {
    echo "Offline statistics sweep did not produce: $STATISTICS_PATH" >&2
    exit 1
  }
  [[ -f "$STATISTICS_DIR/track_micro_anchor_payload.pt" ]] || {
    echo "Offline statistics sweep did not produce Track payload" >&2
    exit 1
  }
  "$PYTHON" scripts/verify_geometry_teacher_statistics.py \
    --statistics "$STATISTICS_PATH" \
    --expected_identity "$GEOMETRY_TEACHER_IDENTITY_MODE"
}

sanitize() {
  offline_statistics
  [[ -f "$SANITIZATION_SOURCE_STATE" ]] || {
    echo "Missing sanitization descriptor checkpoint: $SANITIZATION_SOURCE_STATE" >&2
    exit 1
  }
  [[ -f "$STATISTICS_PATH" ]] || {
    echo "Missing offline landmark statistics: $STATISTICS_PATH" >&2
    exit 1
  }
  local mode
  for mode in $SANITIZATION_MODES; do
    local label="src${SANITIZATION_SOURCE_STEP}_stats${STATISTICS_CHECKPOINT_STEP}_${mode}_${SANITIZED_BUDGET}"
    local output_dir="$RUN_ROOT/sanitize_${label}"
    if [[ -f "$output_dir/sanitized_lafgs_map_state.pt" ]]; then
      continue
    fi
    run_logged "sanitize_${label}" \
      "$PYTHON" scripts/sanitize_lafgs_landmarks.py \
      --source_state "$SANITIZATION_SOURCE_STATE" --statistics "$STATISTICS_PATH" \
      --output_dir "$output_dir" --mode "$mode" --budget "$SANITIZED_BUDGET"
  done
}

eval_state() {
  local label="$1"
  local state="$2"
  local bank_dir="${3:-$BOOTSTRAP_DIR}"
  [[ -f "$state" ]] || return 0
  local pointer="$RESULT_ROOT/${label}.path"
  if [[ -f "$pointer" ]] && [[ -f "$(<"$pointer")/results_summary.json" ]]; then
    echo "Reusing evaluation: $(<"$pointer")"
    return
  fi
  local cfg="$CONFIG_ROOT/${label}.yaml"
  "$PYTHON" scripts/make_stdloc_eval_cfg.py \
    --base_cfg configs/stdloc_cambridge.yaml --output "$cfg" \
    --artifact_model_path "$MODEL_ROOT" \
    --detector_folder ulfloc_native_no_detector --detector_iters 0 \
    --landmark_path "$bank_dir/sampled_idx.pkl" \
    --landmark_meta_path "$bank_dir/landmark_meta.pt" \
    --landmark_feature_override_path "$state" --override_landmark_features \
    --detect_num 2048 --nms 2 \
    --sparse_query_feature_contract native_resized_input \
    --sparse_frontend ulfloc_native \
    --reprojection_error 12 --match_threshold 0 --match_topk 1 \
    --max_matches_per_landmark 0 --candidate_frontend_match_policy error \
    --diagnostics --diagnostics_grid_rows 4 --diagnostics_grid_cols 4 \
    --diagnostics_voxel_size 1 \
    --diagnostics_task_translation_scale_m 0.07160573943725686 \
    --diagnostics_task_rotation_scale_degrees 2 \
    > "$LOG_ROOT/eval_${label}_config.json"
  run_logged "eval_${label}" \
    "$PYTHON" stdloc.py \
    --model_path "$MODEL_ROOT" --source_path "$DATA_ROOT/$SCENE" \
    --images processed --data_device cpu --gaussian_type "$GAUSSIAN_TYPE" \
    --sh_degree "$SH_DEGREE" --feature_type sp --resolution 1 \
    --longest_edge 0 --norm_before_render --iteration 30000 \
    --cfg "$cfg" --prefix "lafgs-v2-sanitization-${RUN_TAG}-${label}" \
    --sparse_only --evaluation_camera_subset test
  local output_path
  output_path="$(sed -n 's/^Output path: //p' "$LOG_ROOT/eval_${label}.log" | tail -n 1)"
  [[ -n "$output_path" && -f "$output_path/results_summary.json" ]] || {
    echo "Missing evaluation output for $label" >&2
    exit 1
  }
  printf '%s\n' "$output_path" > "$pointer"
}

eval_sanitized_states() {
  local mode
  for mode in $SANITIZATION_MODES; do
    local label="src${SANITIZATION_SOURCE_STEP}_stats${STATISTICS_CHECKPOINT_STEP}_${mode}_${SANITIZED_BUDGET}"
    local bank_dir="$RUN_ROOT/sanitize_${label}"
    eval_state "sanitize_${label}" \
      "$bank_dir/sanitized_lafgs_map_state.pt" "$bank_dir"
  done
}

case "$MODE" in
  bootstrap) bootstrap ;;
  stage_a) stage_a ;;
  statistics) offline_statistics ;;
  sanitize) sanitize ;;
  eval)
    eval_state bootstrap "$BOOTSTRAP_STATE"
    eval_state "stage_a_${STAGE_A_PROFILE}_${SANITIZATION_SOURCE_STEP}" \
      "$SANITIZATION_SOURCE_STATE"
    eval_sanitized_states
    ;;
  all)
    sanitize
    eval_state bootstrap "$BOOTSTRAP_STATE"
    eval_state "stage_a_${STAGE_A_PROFILE}_${SANITIZATION_SOURCE_STEP}" \
      "$SANITIZATION_SOURCE_STATE"
    eval_sanitized_states
    ;;
esac
