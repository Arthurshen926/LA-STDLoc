#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT=${DATA_ROOT:-/mnt/pool/sqy/Cambridge_stdloc}
BASELINE_ROOT=${BASELINE_ROOT:-/mnt/pool/sqy/stdloc_la_full_runs}
V03_ROOT=${V03_ROOT:-/mnt/pool/sqy/stdloc_la_v03_full_length}
MODEL_ROOT=${MODEL_ROOT:-/mnt/pool/sqy/stdloc_la_v03_topology_full}
SCENE=${SCENE:-ShopFacade}
TRAIN_SEED=${TRAIN_SEED:-0}
QUERY_SPLIT_SEED=${QUERY_SPLIT_SEED:-2025}
TOPOLOGY_SUPPORT_QUERY_SPLIT=${TOPOLOGY_SUPPORT_QUERY_SPLIT:-1}
TOPOLOGY_QUERY_HOLDOUT_RATIO=${TOPOLOGY_QUERY_HOLDOUT_RATIO:-0.2}
TOPOLOGY_QUERY_SPLIT_MODE=${TOPOLOGY_QUERY_SPLIT_MODE:-sequence_block}
CFG=${CFG:-configs/stdloc_cambridge.yaml}
BASELINE_ITERS=${BASELINE_ITERS:-30000}
V03_ITERATION=${V03_ITERATION:-32000}
TOPOLOGY_STEPS=${TOPOLOGY_STEPS:-100}
TRAIN_PHASE_WAS_SET=${TRAIN_PHASE+x}
LOC_TEACHER_WAS_SET=${LOC_TEACHER+x}
TOPOLOGY_GEOMETRY_ANCHOR_WEIGHT_WAS_SET=${TOPOLOGY_GEOMETRY_ANCHOR_WEIGHT+x}
TOPOLOGY_DENSE_DESC_WEIGHT_WAS_SET=${TOPOLOGY_DENSE_DESC_WEIGHT+x}
TOPOLOGY_DENSE_REPROJ_WEIGHT_WAS_SET=${TOPOLOGY_DENSE_REPROJ_WEIGHT+x}
TRAIN_PHASE=${TRAIN_PHASE:-feature}
LOC_TEACHER=${LOC_TEACHER:-direct}
TOPOLOGY_MUTATION_MODE=${TOPOLOGY_MUTATION_MODE:-split_only}
TOPOLOGY_LOC_INTERVAL=${TOPOLOGY_LOC_INTERVAL:-1}
TOPOLOGY_LOC_ANCHORS=${TOPOLOGY_LOC_ANCHORS:-2048}
TOPOLOGY_DIRECT_WEIGHT=${TOPOLOGY_DIRECT_WEIGHT:-0.05}
TOPOLOGY_MULTIVIEW_WEIGHT=${TOPOLOGY_MULTIVIEW_WEIGHT:-0.03}
TOPOLOGY_FULL_BANK_WEIGHT=${TOPOLOGY_FULL_BANK_WEIGHT:-0.05}
TOPOLOGY_FULL_BANK_HARD_NEGATIVES=${TOPOLOGY_FULL_BANK_HARD_NEGATIVES:-32}
TOPOLOGY_FULL_BANK_MARGIN=${TOPOLOGY_FULL_BANK_MARGIN:-0.2}
TOPOLOGY_FULL_BANK_IGNORE_3D_RADIUS=${TOPOLOGY_FULL_BANK_IGNORE_3D_RADIUS:-0.0}
TOPOLOGY_FULL_BANK_IGNORE_UV_RADIUS=${TOPOLOGY_FULL_BANK_IGNORE_UV_RADIUS:-0.0}
TOPOLOGY_FULL_BANK_SOURCE_MODE=${TOPOLOGY_FULL_BANK_SOURCE_MODE:-ignore}
TOPOLOGY_FULL_BANK_NEARBY_AS_POSITIVE_WAS_SET=${TOPOLOGY_FULL_BANK_NEARBY_AS_POSITIVE+x}
TOPOLOGY_FULL_BANK_NEARBY_AS_POSITIVE=${TOPOLOGY_FULL_BANK_NEARBY_AS_POSITIVE:-0}
TOPOLOGY_FULL_BANK_NEARBY_AS_POSITIVE_UNTIL=${TOPOLOGY_FULL_BANK_NEARBY_AS_POSITIVE_UNTIL:-0}
TOPOLOGY_ANCHOR_WEIGHT=${TOPOLOGY_ANCHOR_WEIGHT:-0.01}
TOPOLOGY_LOC_OVERLAY_MODE=${TOPOLOGY_LOC_OVERLAY_MODE:-none}
TOPOLOGY_LOC_OVERLAY_LR=${TOPOLOGY_LOC_OVERLAY_LR:-0.0}
TOPOLOGY_LOC_OVERLAY_ACTIVE_LOGIT=${TOPOLOGY_LOC_OVERLAY_ACTIVE_LOGIT:-0.0}
TOPOLOGY_LOC_OVERLAY_MAX_RESIDUAL_NORM=${TOPOLOGY_LOC_OVERLAY_MAX_RESIDUAL_NORM:-0.0}
TOPOLOGY_LOC_OVERLAY_NORMALIZE=${TOPOLOGY_LOC_OVERLAY_NORMALIZE:-0}
TOPOLOGY_LOC_OVERLAY_REG_WEIGHT=${TOPOLOGY_LOC_OVERLAY_REG_WEIGHT:-0.0}
TOPOLOGY_CHILD_FEATURE_FREEZE_STEPS=${TOPOLOGY_CHILD_FEATURE_FREEZE_STEPS:-0}
TOPOLOGY_CHILD_RESPONSIBILITY_MODE=${TOPOLOGY_CHILD_RESPONSIBILITY_MODE:-none}
TOPOLOGY_CHILD_RESPONSIBILITY_START_ITER=${TOPOLOGY_CHILD_RESPONSIBILITY_START_ITER:-0}
TOPOLOGY_DENSE_DESC_WEIGHT=${TOPOLOGY_DENSE_DESC_WEIGHT:-0.0}
TOPOLOGY_DENSE_REPROJ_WEIGHT=${TOPOLOGY_DENSE_REPROJ_WEIGHT:-0.0}
TOPOLOGY_USE_LOC_OPACITY=${TOPOLOGY_USE_LOC_OPACITY:-0}
TOPOLOGY_LOC_OPACITY_WEIGHT=${TOPOLOGY_LOC_OPACITY_WEIGHT:-0.0}
TOPOLOGY_ALLOW_UNTRAINED_LOC_OPACITY_PRUNE=${TOPOLOGY_ALLOW_UNTRAINED_LOC_OPACITY_PRUNE:-0}
TOPOLOGY_GEOMETRY_ANCHOR_WEIGHT=${TOPOLOGY_GEOMETRY_ANCHOR_WEIGHT:-0.0}
TOPOLOGY_UPDATE_INTERVAL=${TOPOLOGY_UPDATE_INTERVAL:-25}
TOPOLOGY_MAX_MUTATION_EVENTS=${TOPOLOGY_MAX_MUTATION_EVENTS:-0}
TOPOLOGY_RISK_COMMIT_POLICY=${TOPOLOGY_RISK_COMMIT_POLICY:-off}
TOPOLOGY_RISK_HOLDOUT_SIZE=${TOPOLOGY_RISK_HOLDOUT_SIZE:-4}
TOPOLOGY_RISK_HOLDOUT_SELECTION=${TOPOLOGY_RISK_HOLDOUT_SELECTION:-prefix}
TOPOLOGY_RISK_EPSILON=${TOPOLOGY_RISK_EPSILON:-0.0}
TOPOLOGY_RISK_CI_Z=${TOPOLOGY_RISK_CI_Z:-0.0}
TOPOLOGY_RISK_MIN_CI_SAMPLES=${TOPOLOGY_RISK_MIN_CI_SAMPLES:-2}
TOPOLOGY_RISK_DESC_WEIGHT=${TOPOLOGY_RISK_DESC_WEIGHT:-1.0}
TOPOLOGY_RISK_FULL_BANK_WEIGHT=${TOPOLOGY_RISK_FULL_BANK_WEIGHT:-1.0}
TOPOLOGY_RISK_REPROJ_WEIGHT=${TOPOLOGY_RISK_REPROJ_WEIGHT:-0.0}
TOPOLOGY_RISK_ANCHORS=${TOPOLOGY_RISK_ANCHORS:-256}
TOPOLOGY_RISK_POSE_AE_WEIGHT=${TOPOLOGY_RISK_POSE_AE_WEIGHT:-1.0}
TOPOLOGY_RISK_POSE_TE_WEIGHT=${TOPOLOGY_RISK_POSE_TE_WEIGHT:-1.0}
TOPOLOGY_RISK_POSE_INLIER_WEIGHT=${TOPOLOGY_RISK_POSE_INLIER_WEIGHT:-0.0}
TOPOLOGY_RISK_POSE_AE_SCALE=${TOPOLOGY_RISK_POSE_AE_SCALE:-5.0}
TOPOLOGY_RISK_POSE_TE_SCALE=${TOPOLOGY_RISK_POSE_TE_SCALE:-200.0}
TOPOLOGY_RISK_POSE_INLIER_SCALE=${TOPOLOGY_RISK_POSE_INLIER_SCALE:-100.0}
TOPOLOGY_RISK_POSE_R5_MISS_WEIGHT=${TOPOLOGY_RISK_POSE_R5_MISS_WEIGHT:-0.0}
TOPOLOGY_RISK_POSE_R2_MISS_WEIGHT=${TOPOLOGY_RISK_POSE_R2_MISS_WEIGHT:-0.0}
TOPOLOGY_RISK_POSE_TAIL_FAIL_WEIGHT=${TOPOLOGY_RISK_POSE_TAIL_FAIL_WEIGHT:-0.0}
TOPOLOGY_RISK_POSE_CVAR_WEIGHT=${TOPOLOGY_RISK_POSE_CVAR_WEIGHT:-0.0}
TOPOLOGY_RISK_POSE_CVAR_FRACTION=${TOPOLOGY_RISK_POSE_CVAR_FRACTION:-0.25}
TOPOLOGY_RISK_POSE_VETO_MODE=${TOPOLOGY_RISK_POSE_VETO_MODE:-off}
TOPOLOGY_RISK_POSE_R2_TOLERANCE=${TOPOLOGY_RISK_POSE_R2_TOLERANCE:-0.0}
TOPOLOGY_RISK_POSE_TAIL_TOLERANCE=${TOPOLOGY_RISK_POSE_TAIL_TOLERANCE:-0.0}
QUERY_ARTIFACT_FILTER_PATH=${QUERY_ARTIFACT_FILTER_PATH:-}
QUERY_ARTIFACT_FILTER_SEVERITIES=${QUERY_ARTIFACT_FILTER_SEVERITIES:-mild,severe}
QUERY_ARTIFACT_FILTER_SPLITS=${QUERY_ARTIFACT_FILTER_SPLITS:-heldout_query_sample}
RENDER_ARTIFACT_WEIGHT_PATH=${RENDER_ARTIFACT_WEIGHT_PATH:-}
RENDER_ARTIFACT_WEIGHT_SPLITS=${RENDER_ARTIFACT_WEIGHT_SPLITS:-heldout_query_sample}
RENDER_ARTIFACT_WEIGHT_MODE=${RENDER_ARTIFACT_WEIGHT_MODE:-severity}
RENDER_ARTIFACT_WEIGHT_SEVERITIES=${RENDER_ARTIFACT_WEIGHT_SEVERITIES:-severe}
RENDER_ARTIFACT_WEIGHT_TARGETS=${RENDER_ARTIFACT_WEIGHT_TARGETS:-teacher}
RENDER_ARTIFACT_WEIGHT_DEFAULT=${RENDER_ARTIFACT_WEIGHT_DEFAULT:-1.0}
RENDER_ARTIFACT_WEIGHT_MILD=${RENDER_ARTIFACT_WEIGHT_MILD:-1.0}
RENDER_ARTIFACT_WEIGHT_SEVERE=${RENDER_ARTIFACT_WEIGHT_SEVERE:-0.70}
RENDER_ARTIFACT_WEIGHT_CONTINUOUS_MIN=${RENDER_ARTIFACT_WEIGHT_CONTINUOUS_MIN:-0.70}
RENDER_ARTIFACT_WEIGHT_CONTINUOUS_POWER=${RENDER_ARTIFACT_WEIGHT_CONTINUOUS_POWER:-1.0}
RENDER_ARTIFACT_DIRECT_WEIGHT_COMBINE_MODE=${RENDER_ARTIFACT_DIRECT_WEIGHT_COMBINE_MODE:-product}
RENDER_ARTIFACT_DIRECT_LOSS_SCALE_MODE=${RENDER_ARTIFACT_DIRECT_LOSS_SCALE_MODE:-none}
RENDER_ARTIFACT_REGION_WEIGHT_PATH=${RENDER_ARTIFACT_REGION_WEIGHT_PATH:-}
RENDER_ARTIFACT_REGION_WEIGHT_ROOT=${RENDER_ARTIFACT_REGION_WEIGHT_ROOT:-}
RENDER_ARTIFACT_REGION_WEIGHT_SPLITS=${RENDER_ARTIFACT_REGION_WEIGHT_SPLITS:-heldout_query_sample}
RENDER_ARTIFACT_REGION_WEIGHT_SEVERITIES=${RENDER_ARTIFACT_REGION_WEIGHT_SEVERITIES:-severe}
RENDER_ARTIFACT_REGION_WEIGHT_TARGETS=${RENDER_ARTIFACT_REGION_WEIGHT_TARGETS:-direct}
RENDER_ARTIFACT_REGION_WEIGHT_DEFAULT=${RENDER_ARTIFACT_REGION_WEIGHT_DEFAULT:-1.0}
TOPOLOGY_SUPPORT_QUERY_SORT_BY_NAME=${TOPOLOGY_SUPPORT_QUERY_SORT_BY_NAME:-0}
TOPOLOGY_MIN_OBSERVATIONS=${TOPOLOGY_MIN_OBSERVATIONS:-8}
TOPOLOGY_SPLIT_QUANTILE=${TOPOLOGY_SPLIT_QUANTILE:-0.95}
TOPOLOGY_AMBIGUITY_QUANTILE=${TOPOLOGY_AMBIGUITY_QUANTILE:-0.90}
TOPOLOGY_GROWTH_CAP_PER_EVENT=${TOPOLOGY_GROWTH_CAP_PER_EVENT:-0.0001}
TOPOLOGY_TOTAL_POINT_BUDGET_RATIO=${TOPOLOGY_TOTAL_POINT_BUDGET_RATIO:-1.02}
TOPOLOGY_COOLDOWN_ITERATIONS=${TOPOLOGY_COOLDOWN_ITERATIONS:-300}
TOPOLOGY_DISABLE_SPLIT=${TOPOLOGY_DISABLE_SPLIT:-0}
TOPOLOGY_MIN_REPEATABILITY=${TOPOLOGY_MIN_REPEATABILITY:-0.05}
TOPOLOGY_MIN_RADIUS=${TOPOLOGY_MIN_RADIUS:-0.5}
TOPOLOGY_PHYSICAL_RGB_THRESHOLD=${TOPOLOGY_PHYSICAL_RGB_THRESHOLD:-0.005}
TOPOLOGY_PHYSICAL_LOC_THRESHOLD=${TOPOLOGY_PHYSICAL_LOC_THRESHOLD:-0.005}
TOPOLOGY_PHYSICAL_UTILITY_THRESHOLD=${TOPOLOGY_PHYSICAL_UTILITY_THRESHOLD:--3.0}
TOPOLOGY_PROTECT_LANDMARKS=${TOPOLOGY_PROTECT_LANDMARKS:-0}
TOPOLOGY_REMAP_MODE=${TOPOLOGY_REMAP_MODE:-source_distance}
TOPOLOGY_REMAP_MAX_SOURCE_DISTANCE=${TOPOLOGY_REMAP_MAX_SOURCE_DISTANCE:-}
LABEL_SPLIT=${LABEL_SPLIT:-train}
LABEL_MAX_IMAGES=${LABEL_MAX_IMAGES:-64}
LABEL_REPROJECTION_ERROR=${LABEL_REPROJECTION_ERROR:-12}
FORCE_TOPOLOGY_COPY=${FORCE_TOPOLOGY_COPY:-0}
FORCE_TOPOLOGY_TRAIN=${FORCE_TOPOLOGY_TRAIN:-0}
FORCE_LABEL_STATE=${FORCE_LABEL_STATE:-0}

if [[ -z "${PYTHON:-}" ]]; then
  if [[ -x /root/miniconda3/envs/ulfloc_repro/bin/python ]]; then
    PYTHON=/root/miniconda3/envs/ulfloc_repro/bin/python
  else
    PYTHON=python
  fi
fi

CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-11.8}
export CUDA_HOME
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH=${PYTHONPATH:-/root/STDLoc}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BASELINE_MODEL=${BASELINE_MODEL:-$BASELINE_ROOT/${SCENE}_baseline}
SOURCE_MODEL=${SOURCE_MODEL:-$V03_ROOT/${SCENE}/train_seed_${TRAIN_SEED}/query_split_${QUERY_SPLIT_SEED}/${SCENE}_v03}
TOPOLOGY_MODEL=${TOPOLOGY_MODEL:-$MODEL_ROOT/${SCENE}/train_seed_${TRAIN_SEED}/query_split_${QUERY_SPLIT_SEED}/${SCENE}_v03_topology_from_${V03_ITERATION}}
TOPOLOGY_END=$((V03_ITERATION + TOPOLOGY_STEPS))
RUN_DIR="$MODEL_ROOT/${SCENE}/train_seed_${TRAIN_SEED}/query_split_${QUERY_SPLIT_SEED}"
BASELINE_CFG="$RUN_DIR/${SCENE}_stdloc_baseline_artifacts.yaml"
TOPOLOGY_CFG="$RUN_DIR/${SCENE}_stdloc_topology_${TOPOLOGY_END}_artifacts.yaml"
LABEL_DIR="$RUN_DIR/labels"
LABEL_STATE="$LABEL_DIR/${SCENE}_v03_${V03_ITERATION}_${LABEL_SPLIT}${LABEL_MAX_IMAGES}_sparse_label_state.pt"
LABEL_SUMMARY="$LABEL_DIR/${SCENE}_v03_${V03_ITERATION}_${LABEL_SPLIT}${LABEL_MAX_IMAGES}_sparse_labels.json"
TOPOLOGY_LANDMARK_REL="detector_topology/sampled_idx.pkl"
TOPOLOGY_LANDMARK_PATH="$TOPOLOGY_MODEL/$TOPOLOGY_LANDMARK_REL"
REMAP_SUMMARY="$TOPOLOGY_MODEL/detector_topology/remap_summary.json"

DATA_ARGS=(
  -s "$DATA_ROOT/$SCENE"
  -r 1
  -f sp
  -g 3dgs
  --images processed
  --data_device cpu
)

TRAIN_ARGS=(
  --densify_grad_threshold 0.0004
  --position_lr_init 0.000016
  --scaling_lr 0.001
)

FULL_BANK_ARGS=()
LOC_OVERLAY_ARGS=()
SUPPORT_QUERY_ARGS=()
QUERY_ARTIFACT_FILTER_ARGS=()
RENDER_ARTIFACT_WEIGHT_ARGS=()
RENDER_ARTIFACT_REGION_WEIGHT_ARGS=()
if [[ "$TOPOLOGY_LOC_OVERLAY_NORMALIZE" == "1" ]]; then
  LOC_OVERLAY_ARGS+=(--loc_overlay_normalize)
fi
if [[ "$TOPOLOGY_SUPPORT_QUERY_SPLIT" == "1" ]]; then
  SUPPORT_QUERY_ARGS+=(
    --support_query_split
    --query_holdout_ratio "$TOPOLOGY_QUERY_HOLDOUT_RATIO"
    --query_split_seed "$QUERY_SPLIT_SEED"
    --query_split_mode "$TOPOLOGY_QUERY_SPLIT_MODE"
  )
  if [[ "$TOPOLOGY_SUPPORT_QUERY_SORT_BY_NAME" == "1" ]]; then
    SUPPORT_QUERY_ARGS+=(--support_query_sort_by_name)
  fi
fi
if [[ -n "$QUERY_ARTIFACT_FILTER_PATH" ]]; then
  QUERY_ARTIFACT_FILTER_ARGS+=(
    --query_artifact_filter_path "$QUERY_ARTIFACT_FILTER_PATH"
    --query_artifact_filter_severities "$QUERY_ARTIFACT_FILTER_SEVERITIES"
    --query_artifact_filter_splits "$QUERY_ARTIFACT_FILTER_SPLITS"
  )
fi
if [[ -n "$RENDER_ARTIFACT_WEIGHT_PATH" ]]; then
  RENDER_ARTIFACT_WEIGHT_ARGS+=(
    --render_artifact_weight_path "$RENDER_ARTIFACT_WEIGHT_PATH"
    --render_artifact_weight_splits "$RENDER_ARTIFACT_WEIGHT_SPLITS"
    --render_artifact_weight_mode "$RENDER_ARTIFACT_WEIGHT_MODE"
    --render_artifact_weight_severities "$RENDER_ARTIFACT_WEIGHT_SEVERITIES"
    --render_artifact_weight_targets "$RENDER_ARTIFACT_WEIGHT_TARGETS"
    --render_artifact_weight_default "$RENDER_ARTIFACT_WEIGHT_DEFAULT"
    --render_artifact_weight_mild "$RENDER_ARTIFACT_WEIGHT_MILD"
    --render_artifact_weight_severe "$RENDER_ARTIFACT_WEIGHT_SEVERE"
    --render_artifact_weight_continuous_min "$RENDER_ARTIFACT_WEIGHT_CONTINUOUS_MIN"
    --render_artifact_weight_continuous_power "$RENDER_ARTIFACT_WEIGHT_CONTINUOUS_POWER"
    --render_artifact_direct_weight_combine_mode "$RENDER_ARTIFACT_DIRECT_WEIGHT_COMBINE_MODE"
    --render_artifact_direct_loss_scale_mode "$RENDER_ARTIFACT_DIRECT_LOSS_SCALE_MODE"
  )
fi
if [[ -n "$RENDER_ARTIFACT_REGION_WEIGHT_PATH" ]]; then
  RENDER_ARTIFACT_REGION_WEIGHT_ARGS+=(
    --render_artifact_region_weight_path "$RENDER_ARTIFACT_REGION_WEIGHT_PATH"
    --render_artifact_region_weight_root "$RENDER_ARTIFACT_REGION_WEIGHT_ROOT"
    --render_artifact_region_weight_splits "$RENDER_ARTIFACT_REGION_WEIGHT_SPLITS"
    --render_artifact_region_weight_severities "$RENDER_ARTIFACT_REGION_WEIGHT_SEVERITIES"
    --render_artifact_region_weight_targets "$RENDER_ARTIFACT_REGION_WEIGHT_TARGETS"
    --render_artifact_region_weight_default "$RENDER_ARTIFACT_REGION_WEIGHT_DEFAULT"
  )
fi

point_cloud_exists() {
  local model_path=$1
  local iteration=$2
  [[ -f "$model_path/point_cloud/iteration_${iteration}/point_cloud.ply" ]]
}

strip_future_point_clouds() {
  local iteration_dir
  local iteration
  for iteration_dir in "$TOPOLOGY_MODEL"/point_cloud/iteration_*; do
    [[ -d "$iteration_dir" ]] || continue
    iteration=${iteration_dir##*/iteration_}
    [[ "$iteration" =~ ^[0-9]+$ ]] || continue
    if (( iteration > V03_ITERATION )); then
      rm -rf "$iteration_dir"
    fi
  done
}

if [[ ! -d "$BASELINE_MODEL" ]]; then
  echo "[LA-STDLoc v0.3 topology] Missing baseline model: $BASELINE_MODEL" >&2
  exit 1
fi
if [[ ! -f "$BASELINE_MODEL/detector/${BASELINE_ITERS}_detector.pth" || ! -f "$BASELINE_MODEL/detector/sampled_idx.pkl" ]]; then
  echo "[LA-STDLoc v0.3 topology] Missing baseline detector artifacts under $BASELINE_MODEL/detector." >&2
  exit 1
fi
if ! point_cloud_exists "$SOURCE_MODEL" "$V03_ITERATION"; then
  echo "[LA-STDLoc v0.3 topology] Missing source checkpoint iteration $V03_ITERATION under $SOURCE_MODEL." >&2
  exit 1
fi
if [[ "$(realpath -m "$SOURCE_MODEL")" == "$(realpath -m "$TOPOLOGY_MODEL")" ]]; then
  echo "[LA-STDLoc v0.3 topology] TOPOLOGY_MODEL must differ from SOURCE_MODEL." >&2
  exit 1
fi

mkdir -p "$RUN_DIR" "$LABEL_DIR"

"$PYTHON" - "$CFG" "$BASELINE_CFG" "$BASELINE_MODEL" <<'PY'
import sys
import yaml

src, dst, baseline_model = sys.argv[1], sys.argv[2], sys.argv[3]
with open(src) as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)
sparse = cfg.setdefault("sparse", {})
sparse["detector_path"] = "detector/30000_detector.pth"
sparse["landmark_path"] = "detector/sampled_idx.pkl"
sparse["detector_model_path"] = baseline_model
sparse["landmark_model_path"] = baseline_model
sparse["landmark_meta_model_path"] = baseline_model
sparse["use_landmark_prior"] = False
with open(dst, "w") as f:
    yaml.dump(cfg, f)
PY

if [[ ! -d "$TOPOLOGY_MODEL" || "$FORCE_TOPOLOGY_COPY" == "1" ]]; then
  rm -rf "$TOPOLOGY_MODEL"
  mkdir -p "$(dirname "$TOPOLOGY_MODEL")"
  cp -a "$SOURCE_MODEL" "$TOPOLOGY_MODEL"
  strip_future_point_clouds
elif [[ "$FORCE_TOPOLOGY_TRAIN" == "1" ]]; then
  strip_future_point_clouds
fi

if [[ ! -f "$LABEL_STATE" || "$FORCE_LABEL_STATE" == "1" ]]; then
  "$PYTHON" "$REPO_ROOT/scripts/diagnose_sparse_inliers.py" \
    "${DATA_ARGS[@]}" \
    -m "$TOPOLOGY_MODEL" \
    --iteration "$V03_ITERATION" \
    --cfg "$BASELINE_CFG" \
    --split "$LABEL_SPLIT" \
    --max_images "$LABEL_MAX_IMAGES" \
    --reprojection_error "$LABEL_REPROJECTION_ERROR" \
    --depth_check \
    --label_state_output "$LABEL_STATE" \
    --label_state_reset \
    --output "$LABEL_SUMMARY"
else
  echo "[LA-STDLoc v0.3 topology] Skip label-state generation: found $LABEL_STATE."
fi

TOPOLOGY_ARGS=()
LOC_OPACITY_ARGS=(--no-use_loc_opacity --loc_opacity_weight 0.0)
if [[ "$TOPOLOGY_USE_LOC_OPACITY" == "1" ]]; then
  LOC_OPACITY_ARGS=(--use_loc_opacity --loc_opacity_weight "$TOPOLOGY_LOC_OPACITY_WEIGHT")
fi
REMAP_ARGS=(--remap_mode "$TOPOLOGY_REMAP_MODE")
if [[ -n "$TOPOLOGY_REMAP_MAX_SOURCE_DISTANCE" ]]; then
  REMAP_ARGS+=(--max_source_distance "$TOPOLOGY_REMAP_MAX_SOURCE_DISTANCE")
fi
case "$TOPOLOGY_MUTATION_MODE" in
  no_mutation)
    ;;
  split_only)
    TOPOLOGY_ARGS+=(--enable_topology)
    ;;
  one_shot_split)
    TOPOLOGY_ARGS+=(--enable_topology)
    TOPOLOGY_MAX_MUTATION_EVENTS=1
    ;;
  one_shot_split_freeze)
    TOPOLOGY_ARGS+=(--enable_topology)
    TOPOLOGY_MAX_MUTATION_EVENTS=1
    if [[ "$TOPOLOGY_CHILD_FEATURE_FREEZE_STEPS" == "0" ]]; then
      TOPOLOGY_CHILD_FEATURE_FREEZE_STEPS=100
    fi
    if [[ -z "$TOPOLOGY_FULL_BANK_NEARBY_AS_POSITIVE_WAS_SET" ]]; then
      TOPOLOGY_FULL_BANK_NEARBY_AS_POSITIVE=1
    fi
    if [[ "$TOPOLOGY_FULL_BANK_NEARBY_AS_POSITIVE_UNTIL" == "0" ]]; then
      TOPOLOGY_FULL_BANK_NEARBY_AS_POSITIVE_UNTIL=$((V03_ITERATION + TOPOLOGY_UPDATE_INTERVAL + TOPOLOGY_CHILD_FEATURE_FREEZE_STEPS))
    fi
    ;;
  soft_prune)
    TOPOLOGY_ARGS+=(--enable_topology --topology_enable_soft_prune)
    ;;
  soft_prune_only)
    TOPOLOGY_ARGS+=(--enable_topology --topology_enable_soft_prune)
    TOPOLOGY_DISABLE_SPLIT=1
    ;;
  physical_prune)
    TOPOLOGY_ARGS+=(--enable_topology)
    TOPOLOGY_ARGS+=(--topology_enable_physical_prune)
    if [[ "$TOPOLOGY_PROTECT_LANDMARKS" == "1" ]]; then
      TOPOLOGY_ARGS+=(--topology_protect_landmarks)
    fi
    ;;
  physical_prune_only)
    TOPOLOGY_ARGS+=(--enable_topology)
    TOPOLOGY_ARGS+=(--topology_enable_physical_prune)
    TOPOLOGY_DISABLE_SPLIT=1
    if [[ "$TOPOLOGY_PROTECT_LANDMARKS" == "1" ]]; then
      TOPOLOGY_ARGS+=(--topology_protect_landmarks)
    fi
    ;;
  current_full)
    if [[ -z "$TRAIN_PHASE_WAS_SET" ]]; then
      TRAIN_PHASE=topology
    fi
    if [[ -z "$LOC_TEACHER_WAS_SET" ]]; then
      LOC_TEACHER=dense
    fi
    if [[ -z "$TOPOLOGY_GEOMETRY_ANCHOR_WEIGHT_WAS_SET" ]]; then
      TOPOLOGY_GEOMETRY_ANCHOR_WEIGHT=0.01
    fi
    if [[ -z "$TOPOLOGY_DENSE_DESC_WEIGHT_WAS_SET" ]]; then
      TOPOLOGY_DENSE_DESC_WEIGHT=1.0
    fi
    if [[ -z "$TOPOLOGY_DENSE_REPROJ_WEIGHT_WAS_SET" ]]; then
      TOPOLOGY_DENSE_REPROJ_WEIGHT=0.1
    fi
    TOPOLOGY_ARGS+=(--enable_topology)
    TOPOLOGY_ARGS+=(--topology_enable_physical_prune)
    TOPOLOGY_ARGS+=(--topology_protect_landmarks)
    TOPOLOGY_ALLOW_UNTRAINED_LOC_OPACITY_PRUNE=1
    ;;
  *)
    echo "[LA-STDLoc v0.3 topology] Unknown TOPOLOGY_MUTATION_MODE=$TOPOLOGY_MUTATION_MODE" >&2
    exit 1
    ;;
esac
if [[ "$TOPOLOGY_DISABLE_SPLIT" == "1" ]]; then
  TOPOLOGY_ARGS+=(--topology_disable_split)
fi
if [[ "$TOPOLOGY_ALLOW_UNTRAINED_LOC_OPACITY_PRUNE" == "1" ]]; then
  TOPOLOGY_ARGS+=(--topology_allow_untrained_loc_opacity_prune)
fi
if [[ "$TOPOLOGY_FULL_BANK_NEARBY_AS_POSITIVE" == "1" ]]; then
  FULL_BANK_ARGS+=(--loc_full_bank_nearby_as_positive)
fi
if (( TOPOLOGY_FULL_BANK_NEARBY_AS_POSITIVE_UNTIL > 0 )); then
  FULL_BANK_ARGS+=(--loc_full_bank_nearby_as_positive_until "$TOPOLOGY_FULL_BANK_NEARBY_AS_POSITIVE_UNTIL")
fi

if [[ "$FORCE_TOPOLOGY_TRAIN" == "1" ]]; then
  rm -rf "$TOPOLOGY_MODEL/point_cloud/iteration_${TOPOLOGY_END}"
fi

if [[ "$FORCE_TOPOLOGY_TRAIN" == "1" ]] || ! point_cloud_exists "$TOPOLOGY_MODEL" "$TOPOLOGY_END"; then
  "$PYTHON" train_locaware.py \
    "${DATA_ARGS[@]}" \
    "${TRAIN_ARGS[@]}" \
    -m "$TOPOLOGY_MODEL" \
    --load_iteration "$V03_ITERATION" \
    --iterations "$TOPOLOGY_END" \
    --train_phase "$TRAIN_PHASE" \
    --loc_teacher "$LOC_TEACHER" \
    --landmark_path "$BASELINE_MODEL/detector/sampled_idx.pkl" \
    --loc_direct_weight "$TOPOLOGY_DIRECT_WEIGHT" \
    --loc_multiview_weight "$TOPOLOGY_MULTIVIEW_WEIGHT" \
    --loc_multiview_temperature 0.07 \
    --loc_multiview_slots 4 \
    --loc_full_bank_weight "$TOPOLOGY_FULL_BANK_WEIGHT" \
    --loc_full_bank_temperature 0.07 \
    --loc_full_bank_hard_negatives "$TOPOLOGY_FULL_BANK_HARD_NEGATIVES" \
    --loc_full_bank_margin "$TOPOLOGY_FULL_BANK_MARGIN" \
    --loc_full_bank_ignore_3d_radius "$TOPOLOGY_FULL_BANK_IGNORE_3D_RADIUS" \
    --loc_full_bank_ignore_uv_radius "$TOPOLOGY_FULL_BANK_IGNORE_UV_RADIUS" \
    --loc_full_bank_source_mode "$TOPOLOGY_FULL_BANK_SOURCE_MODE" \
    "${FULL_BANK_ARGS[@]}" \
    --loc_overlay_mode "$TOPOLOGY_LOC_OVERLAY_MODE" \
    --loc_overlay_lr "$TOPOLOGY_LOC_OVERLAY_LR" \
    --loc_overlay_active_logit "$TOPOLOGY_LOC_OVERLAY_ACTIVE_LOGIT" \
    --loc_overlay_max_residual_norm "$TOPOLOGY_LOC_OVERLAY_MAX_RESIDUAL_NORM" \
    "${LOC_OVERLAY_ARGS[@]}" \
    --loc_overlay_reg_weight "$TOPOLOGY_LOC_OVERLAY_REG_WEIGHT" \
    --loc_child_feature_freeze_steps "$TOPOLOGY_CHILD_FEATURE_FREEZE_STEPS" \
    --loc_child_responsibility_mode "$TOPOLOGY_CHILD_RESPONSIBILITY_MODE" \
    --loc_child_responsibility_start_iter "$TOPOLOGY_CHILD_RESPONSIBILITY_START_ITER" \
    --loc_anchor_weight "$TOPOLOGY_ANCHOR_WEIGHT" \
    --loc_desc_weight "$TOPOLOGY_DENSE_DESC_WEIGHT" \
    --loc_reproj_weight "$TOPOLOGY_DENSE_REPROJ_WEIGHT" \
    --loc_proto_weight 0.0 \
    --loc_rank_weight 0.0 \
    "${LOC_OPACITY_ARGS[@]}" \
    --localization_state_path "$LABEL_STATE" \
    --geometry_anchor_weight "$TOPOLOGY_GEOMETRY_ANCHOR_WEIGHT" \
    --loc_interval "$TOPOLOGY_LOC_INTERVAL" \
    --loc_anchors "$TOPOLOGY_LOC_ANCHORS" \
    --train_seed "$TRAIN_SEED" \
    "${SUPPORT_QUERY_ARGS[@]}" \
    "${QUERY_ARTIFACT_FILTER_ARGS[@]}" \
    "${RENDER_ARTIFACT_WEIGHT_ARGS[@]}" \
    "${RENDER_ARTIFACT_REGION_WEIGHT_ARGS[@]}" \
    --direct_depth_check \
    --direct_depth_abs_tolerance 0.001 \
    --direct_depth_rel_tolerance 0.01 \
    "${TOPOLOGY_ARGS[@]}" \
    --topology_stats_warmup "$((V03_ITERATION + TOPOLOGY_UPDATE_INTERVAL))" \
    --topology_update_interval "$TOPOLOGY_UPDATE_INTERVAL" \
    --topology_max_mutation_events "$TOPOLOGY_MAX_MUTATION_EVENTS" \
    --topology_risk_commit_policy "$TOPOLOGY_RISK_COMMIT_POLICY" \
    --topology_risk_holdout_size "$TOPOLOGY_RISK_HOLDOUT_SIZE" \
    --topology_risk_holdout_selection "$TOPOLOGY_RISK_HOLDOUT_SELECTION" \
    --topology_risk_epsilon "$TOPOLOGY_RISK_EPSILON" \
    --topology_risk_ci_z "$TOPOLOGY_RISK_CI_Z" \
    --topology_risk_min_ci_samples "$TOPOLOGY_RISK_MIN_CI_SAMPLES" \
    --topology_risk_desc_weight "$TOPOLOGY_RISK_DESC_WEIGHT" \
    --topology_risk_full_bank_weight "$TOPOLOGY_RISK_FULL_BANK_WEIGHT" \
    --topology_risk_reproj_weight "$TOPOLOGY_RISK_REPROJ_WEIGHT" \
    --topology_risk_anchors "$TOPOLOGY_RISK_ANCHORS" \
    --topology_risk_pose_cfg "$BASELINE_CFG" \
    --topology_risk_pose_ae_weight "$TOPOLOGY_RISK_POSE_AE_WEIGHT" \
    --topology_risk_pose_te_weight "$TOPOLOGY_RISK_POSE_TE_WEIGHT" \
    --topology_risk_pose_inlier_weight "$TOPOLOGY_RISK_POSE_INLIER_WEIGHT" \
    --topology_risk_pose_ae_scale "$TOPOLOGY_RISK_POSE_AE_SCALE" \
    --topology_risk_pose_te_scale "$TOPOLOGY_RISK_POSE_TE_SCALE" \
    --topology_risk_pose_inlier_scale "$TOPOLOGY_RISK_POSE_INLIER_SCALE" \
    --topology_risk_pose_r5_miss_weight "$TOPOLOGY_RISK_POSE_R5_MISS_WEIGHT" \
    --topology_risk_pose_r2_miss_weight "$TOPOLOGY_RISK_POSE_R2_MISS_WEIGHT" \
    --topology_risk_pose_tail_fail_weight "$TOPOLOGY_RISK_POSE_TAIL_FAIL_WEIGHT" \
    --topology_risk_pose_cvar_weight "$TOPOLOGY_RISK_POSE_CVAR_WEIGHT" \
    --topology_risk_pose_cvar_fraction "$TOPOLOGY_RISK_POSE_CVAR_FRACTION" \
    --topology_risk_pose_veto_mode "$TOPOLOGY_RISK_POSE_VETO_MODE" \
    --topology_risk_pose_r2_tolerance "$TOPOLOGY_RISK_POSE_R2_TOLERANCE" \
    --topology_risk_pose_tail_tolerance "$TOPOLOGY_RISK_POSE_TAIL_TOLERANCE" \
    --topology_min_observations "$TOPOLOGY_MIN_OBSERVATIONS" \
    --topology_split_quantile "$TOPOLOGY_SPLIT_QUANTILE" \
    --topology_ambiguity_quantile "$TOPOLOGY_AMBIGUITY_QUANTILE" \
    --topology_growth_cap_per_event "$TOPOLOGY_GROWTH_CAP_PER_EVENT" \
    --topology_total_point_budget_ratio "$TOPOLOGY_TOTAL_POINT_BUDGET_RATIO" \
    --topology_cooldown_iterations "$TOPOLOGY_COOLDOWN_ITERATIONS" \
    --topology_min_repeatability "$TOPOLOGY_MIN_REPEATABILITY" \
    --topology_min_radius "$TOPOLOGY_MIN_RADIUS" \
    --topology_physical_rgb_threshold "$TOPOLOGY_PHYSICAL_RGB_THRESHOLD" \
    --topology_physical_loc_threshold "$TOPOLOGY_PHYSICAL_LOC_THRESHOLD" \
    --topology_physical_utility_threshold "$TOPOLOGY_PHYSICAL_UTILITY_THRESHOLD" \
    --save_iterations "$TOPOLOGY_END" \
    --test_iterations "$TOPOLOGY_END"
else
  echo "[LA-STDLoc v0.3 topology] Skip topology phase: found iteration ${TOPOLOGY_END}."
fi

"$PYTHON" "$REPO_ROOT/scripts/remap_topology_landmarks.py" \
  --source_sampled_idx "$BASELINE_MODEL/detector/sampled_idx.pkl" \
  --topology_loc_state "$TOPOLOGY_MODEL/point_cloud/iteration_${TOPOLOGY_END}/loc_state.pt" \
  --output_sampled_idx "$TOPOLOGY_LANDMARK_PATH" \
  --summary_output "$REMAP_SUMMARY" \
  --remap_score_source label_quality \
  "${REMAP_ARGS[@]}"

"$PYTHON" - "$BASELINE_CFG" "$TOPOLOGY_CFG" "$BASELINE_MODEL" "$TOPOLOGY_MODEL" "$TOPOLOGY_LANDMARK_REL" <<'PY'
import sys
import yaml

src, dst, baseline_model, topology_model, topology_landmark_path = sys.argv[1:6]
with open(src) as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)
sparse = cfg.setdefault("sparse", {})
sparse["detector_model_path"] = baseline_model
sparse["landmark_path"] = topology_landmark_path
sparse["landmark_model_path"] = topology_model
sparse["landmark_meta_model_path"] = topology_model
sparse["use_landmark_prior"] = False
with open(dst, "w") as f:
    yaml.dump(cfg, f)
PY

"$PYTHON" stdloc.py \
  "${DATA_ARGS[@]}" \
  -m "$TOPOLOGY_MODEL" \
  --iteration "$TOPOLOGY_END" \
  --cfg "$TOPOLOGY_CFG" \
  --prefix "phase-v03-topology-${TOPOLOGY_END}" \
  --sparse_only
