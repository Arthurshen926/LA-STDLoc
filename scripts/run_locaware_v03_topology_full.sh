#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT=${DATA_ROOT:-/mnt/pool/sqy/Cambridge_stdloc}
BASELINE_ROOT=${BASELINE_ROOT:-/mnt/pool/sqy/stdloc_la_full_runs}
V03_ROOT=${V03_ROOT:-/mnt/pool/sqy/stdloc_la_v03_full_length}
MODEL_ROOT=${MODEL_ROOT:-/mnt/pool/sqy/stdloc_la_v03_topology_full}
SCENE=${SCENE:-ShopFacade}
SEED=${SEED:-2025}
CFG=${CFG:-configs/stdloc_cambridge.yaml}
BASELINE_ITERS=${BASELINE_ITERS:-30000}
V03_ITERATION=${V03_ITERATION:-32000}
TOPOLOGY_STEPS=${TOPOLOGY_STEPS:-100}
TOPOLOGY_UPDATE_INTERVAL=${TOPOLOGY_UPDATE_INTERVAL:-25}
TOPOLOGY_MIN_OBSERVATIONS=${TOPOLOGY_MIN_OBSERVATIONS:-8}
TOPOLOGY_SPLIT_QUANTILE=${TOPOLOGY_SPLIT_QUANTILE:-0.95}
TOPOLOGY_AMBIGUITY_QUANTILE=${TOPOLOGY_AMBIGUITY_QUANTILE:-0.90}
TOPOLOGY_GROWTH_CAP_PER_EVENT=${TOPOLOGY_GROWTH_CAP_PER_EVENT:-0.0001}
TOPOLOGY_TOTAL_POINT_BUDGET_RATIO=${TOPOLOGY_TOTAL_POINT_BUDGET_RATIO:-1.02}
TOPOLOGY_COOLDOWN_ITERATIONS=${TOPOLOGY_COOLDOWN_ITERATIONS:-300}
TOPOLOGY_MIN_REPEATABILITY=${TOPOLOGY_MIN_REPEATABILITY:-0.05}
TOPOLOGY_MIN_RADIUS=${TOPOLOGY_MIN_RADIUS:-0.5}
TOPOLOGY_PHYSICAL_RGB_THRESHOLD=${TOPOLOGY_PHYSICAL_RGB_THRESHOLD:-0.005}
TOPOLOGY_PHYSICAL_LOC_THRESHOLD=${TOPOLOGY_PHYSICAL_LOC_THRESHOLD:-0.005}
TOPOLOGY_PHYSICAL_UTILITY_THRESHOLD=${TOPOLOGY_PHYSICAL_UTILITY_THRESHOLD:--3.0}
LABEL_SPLIT=${LABEL_SPLIT:-train}
LABEL_MAX_IMAGES=${LABEL_MAX_IMAGES:-64}
LABEL_REPROJECTION_ERROR=${LABEL_REPROJECTION_ERROR:-12}
FORCE_TOPOLOGY_COPY=${FORCE_TOPOLOGY_COPY:-0}
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
SOURCE_MODEL=${SOURCE_MODEL:-$V03_ROOT/${SCENE}/seed_${SEED}/${SCENE}_v03}
TOPOLOGY_MODEL=${TOPOLOGY_MODEL:-$MODEL_ROOT/${SCENE}/seed_${SEED}/${SCENE}_v03_topology_from_${V03_ITERATION}}
TOPOLOGY_END=$((V03_ITERATION + TOPOLOGY_STEPS))
RUN_DIR="$MODEL_ROOT/${SCENE}/seed_${SEED}"
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

point_cloud_exists() {
  local model_path=$1
  local iteration=$2
  [[ -f "$model_path/point_cloud/iteration_${iteration}/point_cloud.ply" ]]
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

if ! point_cloud_exists "$TOPOLOGY_MODEL" "$TOPOLOGY_END"; then
  "$PYTHON" train_locaware.py \
    "${DATA_ARGS[@]}" \
    "${TRAIN_ARGS[@]}" \
    -m "$TOPOLOGY_MODEL" \
    --load_iteration "$V03_ITERATION" \
    --iterations "$TOPOLOGY_END" \
    --train_phase topology \
    --enable_topology \
    --localization_state_path "$LABEL_STATE" \
    --geometry_anchor_weight 0.01 \
    --loc_interval 8 \
    --loc_anchors 1024 \
    --topology_stats_warmup "$((V03_ITERATION + TOPOLOGY_UPDATE_INTERVAL))" \
    --topology_update_interval "$TOPOLOGY_UPDATE_INTERVAL" \
    --topology_min_observations "$TOPOLOGY_MIN_OBSERVATIONS" \
    --topology_split_quantile "$TOPOLOGY_SPLIT_QUANTILE" \
    --topology_ambiguity_quantile "$TOPOLOGY_AMBIGUITY_QUANTILE" \
    --topology_growth_cap_per_event "$TOPOLOGY_GROWTH_CAP_PER_EVENT" \
    --topology_total_point_budget_ratio "$TOPOLOGY_TOTAL_POINT_BUDGET_RATIO" \
    --topology_cooldown_iterations "$TOPOLOGY_COOLDOWN_ITERATIONS" \
    --topology_min_repeatability "$TOPOLOGY_MIN_REPEATABILITY" \
    --topology_min_radius "$TOPOLOGY_MIN_RADIUS" \
    --topology_enable_physical_prune \
    --topology_protect_landmarks \
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
  --remap_score_source label_quality

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
