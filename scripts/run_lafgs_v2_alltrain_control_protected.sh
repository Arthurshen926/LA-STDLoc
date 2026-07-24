#!/usr/bin/env bash
set -euo pipefail

# Full-map development protocol:
#   895 Cambridge mapping images are used for optimization.
#   The 182 official-test images are treated as a development evaluation set.
# Both residual branches share one immutable all-train KCS/GWFF bootstrap.

if [[ $# -ne 3 ]]; then
  echo "Usage: bash $0 <OldHospital> <gpu> <control|protected|semantic_control|semantic_full|semantic_full_lgcv>" >&2
  exit 2
fi

SCENE="$1"
GPU="$2"
VARIANT="$3"
[[ "$SCENE" == "OldHospital" ]] || {
  echo "The initial all-train study is locked to OldHospital" >&2
  exit 2
}
case "$GPU" in
  1|2) ;;
  *) echo "GPU must be 1 or 2" >&2; exit 2 ;;
esac
case "$VARIANT" in
  control) SEMIDENSE_WEIGHT=0; PROTECTED_SET_WEIGHT=0; LGCV_WEIGHT=0; NEGATIVE_RADIUS=6; REJECT_WEIGHT=0.05 ;;
  protected) SEMIDENSE_WEIGHT=0.01; PROTECTED_SET_WEIGHT=0; LGCV_WEIGHT=0; NEGATIVE_RADIUS=6; REJECT_WEIGHT=0.05 ;;
  semantic_control) SEMIDENSE_WEIGHT=0; PROTECTED_SET_WEIGHT=0; LGCV_WEIGHT=0; NEGATIVE_RADIUS=8; REJECT_WEIGHT=0 ;;
  semantic_full) SEMIDENSE_WEIGHT=0.01; PROTECTED_SET_WEIGHT=0.02; LGCV_WEIGHT=0; NEGATIVE_RADIUS=8; REJECT_WEIGHT=0 ;;
  semantic_full_lgcv) SEMIDENSE_WEIGHT=0.01; PROTECTED_SET_WEIGHT=0.02; LGCV_WEIGHT=0.1; NEGATIVE_RADIUS=8; REJECT_WEIGHT=0 ;;
  *) echo "Unknown variant: $VARIANT" >&2; exit 2 ;;
esac

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/root/miniconda3/envs/cybersim_agent/bin/python}"
DATA_ROOT="${CAMBRIDGE_DATA_ROOT:-/mnt/pool/sqy/Cambridge_stdloc}"
MODEL_ROOT="${LAFGS_ALLTRAIN_MODEL_ROOT:-/mnt/pool/sqy/stdloc_lafgs_v2_ulfparity_alternating_20260721/matcha_wrappers/OldHospital}"
EXPERIMENT_ROOT="${LAFGS_ALLTRAIN_ROOT:-/mnt/pool/sqy/stdloc_lafgs_v2_alltrain_20260724}"
BOOTSTRAP_ROOT="${LAFGS_ALLTRAIN_BOOTSTRAP_ROOT:-$EXPERIMENT_ROOT/OldHospital/robust32k_all895_bootstrap/bootstrap}"
QUERY_CACHE="${LAFGS_ALLTRAIN_QUERY_CACHE:-/mnt/pool/sqy/stdloc_lafgs_v2_ulfparity_alternating_20260721/OldHospital/ulfparity_native20k_s128_k2048_v2/query_cache_native_fullres_k2048.pt}"
VISIBILITY_CACHE="${LAFGS_ALLTRAIN_VISIBILITY_CACHE:-$EXPERIMENT_ROOT/OldHospital/robust32k_all895_bootstrap/visibility_32000_native.pt}"
STEPS="${LAFGS_ALLTRAIN_STEPS:-5000}"

BOOTSTRAP_STATE="$BOOTSTRAP_ROOT/0_lafgs_map_state.pt"
LANDMARK_IDS="$BOOTSTRAP_ROOT/sampled_idx.pkl"
LANDMARK_META="$BOOTSTRAP_ROOT/landmark_meta.pt"
RUN_ROOT="$EXPERIMENT_ROOT/$SCENE/${VARIANT}_native2048_steps${STEPS}"
STATE_ROOT="$RUN_ROOT/map"
CONFIG_ROOT="$RUN_ROOT/configs"
LOG_ROOT="$RUN_ROOT/logs"
RESULT_ROOT="$RUN_ROOT/results"
STDLOC_RESULTS_ROOT="$RUN_ROOT/stdloc_results"

for required in \
  "$BOOTSTRAP_STATE" "$LANDMARK_IDS" "$LANDMARK_META" \
  "$QUERY_CACHE" "$VISIBILITY_CACHE"; do
  [[ -f "$required" ]] || {
    echo "Missing required all-train input: $required" >&2
    exit 1
  }
done

export CUDA_VISIBLE_DEVICES="$GPU"
export CUDA_HOME=/usr/local/cuda-11.8
export PATH="/root/miniconda3/envs/cybersim_agent/bin:$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="/root/miniconda3/envs/cybersim_agent/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$REPO_ROOT"
export PYTHONHASHSEED=2026
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export STDLOC_CAMERA_LOADER_WORKERS=0
export STDLOC_RESULTS_ROOT

mkdir -p "$STATE_ROOT" "$CONFIG_ROOT" "$LOG_ROOT" "$RESULT_ROOT" "$STDLOC_RESULTS_ROOT"
cd "$REPO_ROOT"

run_logged() {
  local name="$1"
  shift
  printf '%q ' "$@" > "$LOG_ROOT/${name}.command.sh"
  printf '\n' >> "$LOG_ROOT/${name}.command.sh"
  "$@" 2>&1 | tee "$LOG_ROOT/${name}.log"
}

verify_alltrain_state() {
  local state="$1"
  "$PYTHON" - "$state" "$VARIANT" <<'PY'
import math
import sys

import torch

state_path, variant = sys.argv[1:]
state = torch.load(state_path, map_location="cpu")
config = dict(state.get("config", {}))
errors = []

def exact(key, expected):
    if config.get(key) != expected:
        errors.append(f"{key}={config.get(key)!r}, expected {expected!r}")

def numeric(key, expected):
    try:
        valid = math.isclose(
            float(config.get(key)), float(expected), rel_tol=0.0, abs_tol=1e-12
        )
    except (TypeError, ValueError):
        valid = False
    if not valid:
        errors.append(f"{key}={config.get(key)!r}, expected {expected!r}")

numeric("validation_ratio", 0.0)
numeric("train_camera_count", 895)
numeric("validation_camera_count", 0)
numeric("native_sparse_keypoint_count", 2048)
numeric("max_observations", 2048)
numeric("native_global_attractor_weight", 0.25)
numeric(
    "native_semidense_weight",
    0.01 if variant in {"protected", "semantic_full", "semantic_full_lgcv"} else 0.0,
)
numeric(
    "native_protected_set_weight",
    0.02 if variant in {"semantic_full", "semantic_full_lgcv"} else 0.0,
)
numeric("native_semidense_lgcv_weight", 0.1 if variant == "semantic_full_lgcv" else 0.0)
numeric(
    "negative_radius_px",
    8.0 if variant.startswith("semantic_") else 6.0,
)
numeric(
    "native_reject_weight",
    0.0 if variant.startswith("semantic_") else 0.05,
)
exact("observation_source", "native")
exact("native_outcome_mode", True)
exact("geometry_frozen", True)
exact("online_rendering", False)
if errors:
    raise SystemExit("Invalid all-train state: " + "; ".join(errors))
print(f"Verified all-train {variant} state: {state_path}")
PY
}

FINAL_STATE="$STATE_ROOT/${STEPS}_lafgs_map_state.pt"
if [[ ! -f "$FINAL_STATE" ]]; then
  SEMIDENSE_ARGS=()
  if [[ "$VARIANT" == "protected" || "$VARIANT" == semantic_full* ]]; then
    SEMIDENSE_ARGS=(
      --native_semidense_weight "$SEMIDENSE_WEIGHT"
      --native_semidense_start_step 1000
      --native_semidense_interval 5
      --native_semidense_max_anchors 64
      --native_semidense_neighbors 8
      --native_semidense_local_radius_px 8
      --native_semidense_target_sigma_px 2
      --native_semidense_temperature 0.07
      --native_semidense_pose_safe_max_delete_gain_m -1
      --no-native_semidense_pose_safe_teacher_pairs
      --native_semidense_lgcv_weight "$LGCV_WEIGHT"
      --native_semidense_lgcv_minimum_edge_px 1
      --native_semidense_protected_v2
      --native_semidense_measurement_min_reprojection_px 2
      --native_semidense_measurement_max_reprojection_px 8
      --native_semidense_surface_point_plane_m 0.03
      --native_semidense_surface_max_distance_m 0.15
      --native_semidense_surface_normal_cosine 0.95
      --native_semidense_projected_neighbor_radius_px 64
      --native_semidense_local_identity_weight 0.25
      --native_semidense_margin_preservation_weight 4
      --native_semidense_gradient_audit
    )
    if [[ "$VARIANT" == semantic_full* ]]; then
      SEMIDENSE_ARGS+=(
        --native_semidense_reference_refresh_steps 500
        --native_semidense_alternate_global
        --native_semidense_max_gradient_ratio 0.25
        --native_protected_set_weight "$PROTECTED_SET_WEIGHT"
        --native_protected_set_start_step 1000
        --native_protected_set_interval 5
        --native_protected_set_refresh_visits 1
        --native_protected_set_ransac_seed 0
        --native_protected_set_ransac_reprojection_px 8
        --native_protected_set_ransac_max_iterations 5000
        --native_protected_set_ransac_min_iterations 100
        --native_protected_set_max_useful 96
        --native_protected_set_max_harmful 96
        --native_protected_set_grid_rows 4
        --native_protected_set_grid_cols 4
        --native_protected_set_depth_bins 4
        --native_protected_set_surface_voxel_m 0.25
        --native_protected_set_max_per_surface_group 2
      )
    fi
  fi
  run_logged train \
    "$PYTHON" train_lafgs_map.py \
    --model_path "$MODEL_ROOT" --source_path "$DATA_ROOT/$SCENE" \
    --images processed --data_device cpu --gaussian_type 2dgs --feature_type sp \
    --resolution 1 --longest_edge 0 --norm_before_render --load_iteration 30000 \
    --query_feature_contract native_resized_input \
    --query_cache_path "$QUERY_CACHE" --query_cache_policy readonly \
    --visibility_cache_path "$VISIBILITY_CACHE" --visibility_mode rasterizer \
    --output_dir "$STATE_ROOT" --scaffold_mode file \
    --landmark_path "$LANDMARK_IDS" --initial_state_path "$BOOTSTRAP_STATE" \
    --initial_state_blend 1 --initial_state_alignment exact \
    --initialization_mode ulf_robust_geometry \
    --objective hard --observation_source native --native_keypoint_count 2048 \
    --max_observations 2048 --validation_observations 2048 \
    --native_sampling_mode detector_grid --native_association_radius_px 2 \
    --native_anchor_aux_weight 0 --native_outcome_mode --native_nce_weight 0 \
    --native_keep_weight 1 --native_keep_margin 0.05 \
    --native_swap_weight 1 --native_swap_margin 0.05 \
    --native_miss_weight 1 --native_miss_margin 0.05 \
    --native_reject_weight "$REJECT_WEIGHT" --native_reject_threshold 0 \
    --native_global_attractor_weight 0.25 \
    --native_global_attractor_min_incoming 4 \
    --native_global_attractor_support_power 0.5 \
    --native_global_attractor_max_score 4 \
    "${SEMIDENSE_ARGS[@]}" \
    --mv_weight 0 --retrieval_weight 1 --trust_weight 0.02 --local_weight 0 \
    --dustbin_weight 0 --generic_proposal_count 0 --distill_budget 0 \
    --geometry_weight 0 --pose_weight 0 --pose_gradient_mode off \
    --feature_lr 5e-5 --weight_decay 1e-4 \
    --hypothesis_topk 32 --positive_radius_px 2 --negative_radius_px "$NEGATIVE_RADIUS" \
    --validation_ratio 0 --split_mode stratified_temporal_block \
    --split_seed 2026 --train_seed 2026 \
    --steps "$STEPS" --save_steps 1000 2500 "$STEPS" --log_interval 100
fi
verify_alltrain_state "$FINAL_STATE"

for STEP in 1000 2500 "$STEPS"; do
  STATE_PATH="$STATE_ROOT/${STEP}_lafgs_map_state.pt"
  [[ -f "$STATE_PATH" ]] || continue
  POINTER="$RESULT_ROOT/${STEP}_test.path"
  if [[ -f "$POINTER" ]] && [[ -f "$(<"$POINTER")/results_summary.json" ]]; then
    continue
  fi
  CONFIG_PATH="$CONFIG_ROOT/${STEP}.yaml"
  "$PYTHON" scripts/make_stdloc_eval_cfg.py \
    --base_cfg configs/stdloc_cambridge.yaml --output "$CONFIG_PATH" \
    --artifact_model_path "$MODEL_ROOT" \
    --detector_folder ulfloc_native_no_detector --detector_iters 0 \
    --landmark_path "$LANDMARK_IDS" --landmark_meta_path "$LANDMARK_META" \
    --landmark_feature_override_path "$STATE_PATH" --override_landmark_features \
    --detect_num 2048 --nms 2 \
    --sparse_query_feature_contract native_resized_input \
    --sparse_frontend ulfloc_native \
    --reprojection_error 12 --match_threshold 0 --match_topk 1 \
    --max_matches_per_landmark 0 --candidate_frontend_match_policy error \
    --diagnostics --diagnostics_grid_rows 4 --diagnostics_grid_cols 4 \
    --diagnostics_voxel_size 1.0 \
    --diagnostics_task_translation_scale_m 0.07160573943725686 \
    --diagnostics_task_rotation_scale_degrees 2.0 \
    > "$LOG_ROOT/eval_${STEP}_config.json"
  run_logged "eval_${STEP}_test" \
    "$PYTHON" stdloc.py \
    --model_path "$MODEL_ROOT" --source_path "$DATA_ROOT/$SCENE" \
    --images processed --data_device cpu --gaussian_type 2dgs --feature_type sp \
    --resolution 1 --longest_edge 0 --norm_before_render --iteration 30000 \
    --cfg "$CONFIG_PATH" --prefix "lafgs-v2-alltrain-${VARIANT}-${SCENE}-${STEP}" \
    --sparse_only --evaluation_camera_subset test
  OUTPUT_PATH="$(sed -n 's/^Output path: //p' "$LOG_ROOT/eval_${STEP}_test.log" | tail -n 1)"
  [[ -n "$OUTPUT_PATH" && -f "$OUTPUT_PATH/results_summary.json" ]] || {
    echo "Missing test output for step $STEP" >&2
    exit 1
  }
  printf '%s\n' "$OUTPUT_PATH" > "$POINTER"
done
