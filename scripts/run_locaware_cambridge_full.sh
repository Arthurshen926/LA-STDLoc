#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT=${DATA_ROOT:-/mnt/pool/sqy/Cambridge_stdloc}
SCENE=${SCENE:-ShopFacade}
MODEL_ROOT=${MODEL_ROOT:-map_cambridge_la_full}
BASELINE_MODEL="$MODEL_ROOT/${SCENE}_baseline"
LA_MODEL="$MODEL_ROOT/${SCENE}_la"
CFG=${CFG:-configs/stdloc_cambridge.yaml}

BASELINE_ITERS=${BASELINE_ITERS:-30000}
FEATURE_ITERS=${FEATURE_ITERS:-3000}
GEOMETRY_ITERS=${GEOMETRY_ITERS:-3000}
TOPOLOGY_ITERS=${TOPOLOGY_ITERS:-3000}
CLOSED_LOOP_ITERS=${CLOSED_LOOP_ITERS:-1000}
DETECTOR_ITERS=${DETECTOR_ITERS:-30000}
RUN_CFG="$MODEL_ROOT/${SCENE}_stdloc_detector_${DETECTOR_ITERS}.yaml"

FEATURE_END=$((BASELINE_ITERS + FEATURE_ITERS))
GEOMETRY_END=$((FEATURE_END + GEOMETRY_ITERS))
TOPOLOGY_END=$((GEOMETRY_END + TOPOLOGY_ITERS))
CLOSED_LOOP_END=$((TOPOLOGY_END + CLOSED_LOOP_ITERS))

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

mkdir -p "$MODEL_ROOT"
"$PYTHON" - "$CFG" "$RUN_CFG" "$DETECTOR_ITERS" <<'PY'
import sys
import yaml

src, dst, detector_iters = sys.argv[1], sys.argv[2], sys.argv[3]
with open(src) as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)
cfg.setdefault("sparse", {})["detector_path"] = f"detector/{detector_iters}_detector.pth"
with open(dst, "w") as f:
    yaml.dump(cfg, f)
PY

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

if [[ ! -f "$BASELINE_MODEL/point_cloud/iteration_${BASELINE_ITERS}/point_cloud.ply" || ! -f "$BASELINE_MODEL/detector/${DETECTOR_ITERS}_detector.pth" ]]; then
  "$PYTHON" train.py \
    "${DATA_ARGS[@]}" \
    "${TRAIN_ARGS[@]}" \
    -m "$BASELINE_MODEL" \
    --iterations "$BASELINE_ITERS" \
    --train_detector \
    --test_iterations "$BASELINE_ITERS" \
    --save_iterations "$BASELINE_ITERS" \
    --test_detector_iterations "$DETECTOR_ITERS" \
    --save_detector_iterations "$DETECTOR_ITERS"
fi

"$PYTHON" stdloc.py \
  "${DATA_ARGS[@]}" \
  -m "$BASELINE_MODEL" \
  --iteration "$BASELINE_ITERS" \
  --cfg "$RUN_CFG" \
  --prefix "phase0-baseline"

if [[ ! -d "$LA_MODEL" || "${FORCE_LA_COPY:-0}" == "1" ]]; then
  rm -rf "$LA_MODEL"
  mkdir -p "$(dirname "$LA_MODEL")"
  cp -a "$BASELINE_MODEL" "$LA_MODEL"
fi

if ! point_cloud_exists "$LA_MODEL" "$FEATURE_END"; then
  "$PYTHON" train_locaware.py \
    "${DATA_ARGS[@]}" \
    "${TRAIN_ARGS[@]}" \
    -m "$LA_MODEL" \
    --load_iteration "$BASELINE_ITERS" \
    --iterations "$FEATURE_END" \
    --train_phase feature \
    --loc_interval 8 \
    --loc_anchors 1024 \
    --save_iterations "$FEATURE_END" \
    --test_iterations "$FEATURE_END"
else
  echo "[LA-STDLoc] Skip feature phase: found iteration ${FEATURE_END}."
fi

if ! point_cloud_exists "$LA_MODEL" "$GEOMETRY_END"; then
  "$PYTHON" train_locaware.py \
    "${DATA_ARGS[@]}" \
    "${TRAIN_ARGS[@]}" \
    -m "$LA_MODEL" \
    --load_iteration "$FEATURE_END" \
    --iterations "$GEOMETRY_END" \
    --train_phase geometry \
    --geometry_anchor_weight 0.01 \
    --loc_interval 8 \
    --loc_anchors 1024 \
    --save_iterations "$GEOMETRY_END" \
    --test_iterations "$GEOMETRY_END"
else
  echo "[LA-STDLoc] Skip geometry phase: found iteration ${GEOMETRY_END}."
fi

if ! point_cloud_exists "$LA_MODEL" "$TOPOLOGY_END"; then
  "$PYTHON" train_locaware.py \
    "${DATA_ARGS[@]}" \
    "${TRAIN_ARGS[@]}" \
    -m "$LA_MODEL" \
    --load_iteration "$GEOMETRY_END" \
    --iterations "$TOPOLOGY_END" \
    --train_phase topology \
    --enable_topology \
    --geometry_anchor_weight 0.01 \
    --topology_stats_warmup "$((GEOMETRY_END + 1000))" \
    --loc_interval 8 \
    --loc_anchors 1024 \
    --save_iterations "$TOPOLOGY_END" \
    --test_iterations "$TOPOLOGY_END"
else
  echo "[LA-STDLoc] Skip topology phase: found iteration ${TOPOLOGY_END}."
fi

"$PYTHON" train_detector.py \
  "${DATA_ARGS[@]}" \
  "${TRAIN_ARGS[@]}" \
  -m "$LA_MODEL" \
  --iteration "$TOPOLOGY_END" \
  --iterations "$DETECTOR_ITERS" \
  --sampling_mode localization_aware \
  --detector_target_mode soft \
  --min_loc_observations 8 \
  --test_iterations "$DETECTOR_ITERS" \
  --save_iterations "$DETECTOR_ITERS"

"$PYTHON" cache_sparse_poses.py \
  "${DATA_ARGS[@]}" \
  -m "$LA_MODEL" \
  --iteration "$TOPOLOGY_END" \
  --cfg "$RUN_CFG" \
  --output "$LA_MODEL/sparse_pose_cache/train_sparse_cache.pt" \
  --split train

if ! point_cloud_exists "$LA_MODEL" "$CLOSED_LOOP_END"; then
  "$PYTHON" train_locaware.py \
    "${DATA_ARGS[@]}" \
    "${TRAIN_ARGS[@]}" \
    -m "$LA_MODEL" \
    --load_iteration "$TOPOLOGY_END" \
    --iterations "$CLOSED_LOOP_END" \
    --train_phase closed_loop \
    --query_mode mixed \
    --sparse_pose_cache "$LA_MODEL/sparse_pose_cache/train_sparse_cache.pt" \
    --geometry_anchor_weight 0.01 \
    --loc_interval 8 \
    --loc_anchors 1024 \
    --save_iterations "$CLOSED_LOOP_END" \
    --test_iterations "$CLOSED_LOOP_END"
else
  echo "[LA-STDLoc] Skip closed-loop phase: found iteration ${CLOSED_LOOP_END}."
fi

"$PYTHON" train_detector.py \
  "${DATA_ARGS[@]}" \
  "${TRAIN_ARGS[@]}" \
  -m "$LA_MODEL" \
  --iteration "$CLOSED_LOOP_END" \
  --iterations "$DETECTOR_ITERS" \
  --sampling_mode localization_aware \
  --detector_target_mode soft \
  --min_loc_observations 8 \
  --test_iterations "$DETECTOR_ITERS" \
  --save_iterations "$DETECTOR_ITERS"

"$PYTHON" stdloc.py \
  "${DATA_ARGS[@]}" \
  -m "$LA_MODEL" \
  --iteration "$CLOSED_LOOP_END" \
  --cfg "$RUN_CFG" \
  --prefix "phase6-la-sparse" \
  --sparse_only

"$PYTHON" stdloc.py \
  "${DATA_ARGS[@]}" \
  -m "$LA_MODEL" \
  --iteration "$CLOSED_LOOP_END" \
  --cfg "$RUN_CFG" \
  --prefix "phase6-la-sparse-dense"
