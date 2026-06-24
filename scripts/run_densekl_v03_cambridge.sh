#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT=${DATA_ROOT:-/mnt/pool/sqy/Cambridge_stdloc}
SCENE=${SCENE:-ShopFacade}
BASELINE_MODEL=${BASELINE_MODEL:-/mnt/pool/sqy/stdloc_la_full_runs/${SCENE}_baseline}
SOURCE_MODEL=${SOURCE_MODEL:-/mnt/pool/sqy/stdloc_la_v03_runs/${SCENE}_v03_100_20260623_114535}
MODEL_ROOT=${MODEL_ROOT:-/mnt/pool/sqy/stdloc_la_densekl_runs}
DENSEKL_MODEL=${DENSEKL_MODEL:-$MODEL_ROOT/${SCENE}_densekl}
CFG=${CFG:-configs/stdloc_cambridge.yaml}
LOAD_ITERATION=${LOAD_ITERATION:-30100}
DENSEKL_STEPS=${DENSEKL_STEPS:-100}
DENSEKL_SAVE_STEPS=${DENSEKL_SAVE_STEPS:-$DENSEKL_STEPS}
DENSEKL_EVAL_STEPS=${DENSEKL_EVAL_STEPS:-$DENSEKL_SAVE_STEPS}
DENSEKL_WEIGHT=${DENSEKL_WEIGHT:-0.02}
DENSEKL_TOPK=${DENSEKL_TOPK:-32}
DENSEKL_OPACITY_WEIGHT=${DENSEKL_OPACITY_WEIGHT:-0.0}
DENSEKL_DEPTH_WEIGHT=${DENSEKL_DEPTH_WEIGHT:-0.0}
DENSEKL_QUERY_MODE_WAS_SET=${DENSEKL_QUERY_MODE+x}
DENSEKL_QUERY_MODE=${DENSEKL_QUERY_MODE:-noise}
DENSEKL_POSE_GATE=${DENSEKL_POSE_GATE:-0}
DENSEKL_POSE_GATE_MIN_TE=${DENSEKL_POSE_GATE_MIN_TE:-0.0}
DENSEKL_POSE_GATE_MIN_AE=${DENSEKL_POSE_GATE_MIN_AE:-0.0}
DENSEKL_ATTR_COSINE_THRESHOLD=${DENSEKL_ATTR_COSINE_THRESHOLD:--1.0}
DENSEKL_ATTR_ENTROPY_THRESHOLD=${DENSEKL_ATTR_ENTROPY_THRESHOLD:--1.0}
DENSEKL_MIN_POSITIVE_PROB=${DENSEKL_MIN_POSITIVE_PROB:--1.0}
DENSEKL_MAX_REPROJ_ERROR=${DENSEKL_MAX_REPROJ_ERROR:--1.0}
DENSEKL_MIN_ELIGIBLE_ANCHORS=${DENSEKL_MIN_ELIGIBLE_ANCHORS:-1}
DENSEKL_TRAIN_SEED=${DENSEKL_TRAIN_SEED:-0}
RUN_DENSE_POSE_CACHE=${RUN_DENSE_POSE_CACHE:-0}
LOC_INTERVAL=${LOC_INTERVAL:-1}
LOC_ANCHORS=${LOC_ANCHORS:-512}
REPROJECTION_ERROR=${REPROJECTION_ERROR:-12}
RUN_DIAGNOSTICS=${RUN_DIAGNOSTICS:-1}
RUN_EVAL=${RUN_EVAL:-1}
FORCE_DENSEKL_TRAIN=${FORCE_DENSEKL_TRAIN:-0}

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
RUN_CFG="$MODEL_ROOT/${SCENE}_densekl_baseline_artifacts_reproj${REPROJECTION_ERROR}.yaml"
DENSEKL_END=$((LOAD_ITERATION + DENSEKL_STEPS))
DENSE_POSE_CACHE=${DENSE_POSE_CACHE:-$MODEL_ROOT/${SCENE}_dense_pose_cache_${LOAD_ITERATION}.pt}

steps_to_iterations() {
  local steps
  local out=()
  for steps in "$@"; do
    out+=("$((LOAD_ITERATION + steps))")
  done
  printf '%s\n' "${out[@]}"
}

read -r -a DENSEKL_SAVE_STEP_ARRAY <<<"$DENSEKL_SAVE_STEPS"
read -r -a DENSEKL_EVAL_STEP_ARRAY <<<"$DENSEKL_EVAL_STEPS"
mapfile -t DENSEKL_SAVE_ITERATIONS < <(steps_to_iterations "${DENSEKL_SAVE_STEP_ARRAY[@]}")
mapfile -t DENSEKL_EVAL_ITERATIONS < <(steps_to_iterations "${DENSEKL_EVAL_STEP_ARRAY[@]}")

"$PYTHON" - "$CFG" "$RUN_CFG" "$BASELINE_MODEL" "$REPROJECTION_ERROR" <<'PY'
import sys
import yaml

src, dst, baseline_model, reprojection_error = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4])
with open(src) as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)
sparse = cfg.setdefault("sparse", {})
sparse["detector_path"] = "detector/30000_detector.pth"
sparse["landmark_path"] = "detector/sampled_idx.pkl"
sparse["detector_model_path"] = baseline_model
sparse["landmark_model_path"] = baseline_model
sparse["landmark_meta_model_path"] = baseline_model
sparse["use_landmark_prior"] = False
sparse["reprojection_error"] = reprojection_error
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

strip_future_point_clouds() {
  local iteration_dir
  local iteration
  for iteration_dir in "$DENSEKL_MODEL"/point_cloud/iteration_*; do
    [[ -d "$iteration_dir" ]] || continue
    iteration=${iteration_dir##*/iteration_}
    [[ "$iteration" =~ ^[0-9]+$ ]] || continue
    if (( iteration > LOAD_ITERATION )); then
      rm -rf "$iteration_dir"
    fi
  done
}

if [[ ! -d "$DENSEKL_MODEL" || "${FORCE_DENSEKL_COPY:-0}" == "1" ]]; then
  rm -rf "$DENSEKL_MODEL"
  mkdir -p "$(dirname "$DENSEKL_MODEL")"
  cp -a "$SOURCE_MODEL" "$DENSEKL_MODEL"
  strip_future_point_clouds
elif [[ "$FORCE_DENSEKL_TRAIN" == "1" ]]; then
  strip_future_point_clouds
fi

if [[ "$RUN_DIAGNOSTICS" == "1" ]]; then
  "$PYTHON" scripts/diagnose_dense_responsibility.py \
    "${DATA_ARGS[@]}" \
    -m "$DENSEKL_MODEL" \
    --iteration "$LOAD_ITERATION" \
    --split train \
    --max_images 8 \
    --anchor_count "$LOC_ANCHORS" \
    --dense_kl_weight "$DENSEKL_WEIGHT" \
    --responsibility_topk "$DENSEKL_TOPK" \
    --responsibility_opacity_weight "$DENSEKL_OPACITY_WEIGHT" \
    --responsibility_depth_weight "$DENSEKL_DEPTH_WEIGHT" \
    --output "$MODEL_ROOT/${SCENE}_densekl_responsibility_diag.json"
fi

POSE_GATE_ARGS=()
if [[ "$DENSEKL_POSE_GATE" == "1" ]]; then
  POSE_GATE_ARGS+=(
    --loc_dense_pose_gate
    --loc_dense_pose_gate_min_te "$DENSEKL_POSE_GATE_MIN_TE"
    --loc_dense_pose_gate_min_ae "$DENSEKL_POSE_GATE_MIN_AE"
  )
  RUN_DENSE_POSE_CACHE=1
  if [[ -z "$DENSEKL_QUERY_MODE_WAS_SET" ]]; then
    DENSEKL_QUERY_MODE=sparse
  fi
fi

SELECTIVE_KL_ARGS=(
  --loc_dense_attr_cosine_threshold "$DENSEKL_ATTR_COSINE_THRESHOLD"
  --loc_dense_attr_entropy_threshold "$DENSEKL_ATTR_ENTROPY_THRESHOLD"
  --loc_dense_min_positive_prob "$DENSEKL_MIN_POSITIVE_PROB"
  --loc_dense_max_reproj_error "$DENSEKL_MAX_REPROJ_ERROR"
  --loc_dense_min_eligible_anchors "$DENSEKL_MIN_ELIGIBLE_ANCHORS"
)

SPARSE_POSE_CACHE_ARGS=()
if [[ "$RUN_DENSE_POSE_CACHE" == "1" ]]; then
  SPARSE_POSE_CACHE_ARGS=(--sparse_pose_cache "$DENSE_POSE_CACHE")
fi

if [[ "$RUN_DENSE_POSE_CACHE" == "1" && ! -f "$DENSE_POSE_CACHE" ]]; then
  "$PYTHON" cache_sparse_poses.py \
    "${DATA_ARGS[@]}" \
    -m "$DENSEKL_MODEL" \
    --iteration "$LOAD_ITERATION" \
    --cfg "$RUN_CFG" \
    --output "$DENSE_POSE_CACHE" \
    --split train \
    --include_dense
fi

NEED_DENSEKL_TRAIN=0
for save_iteration in "${DENSEKL_SAVE_ITERATIONS[@]}"; do
  if ! point_cloud_exists "$DENSEKL_MODEL" "$save_iteration"; then
    NEED_DENSEKL_TRAIN=1
    break
  fi
done
if ! point_cloud_exists "$DENSEKL_MODEL" "$DENSEKL_END"; then
  NEED_DENSEKL_TRAIN=1
fi

if [[ "$NEED_DENSEKL_TRAIN" == "1" ]]; then
  "$PYTHON" train_locaware.py \
    "${DATA_ARGS[@]}" \
    "${TRAIN_ARGS[@]}" \
    -m "$DENSEKL_MODEL" \
    --load_iteration "$LOAD_ITERATION" \
    --iterations "$DENSEKL_END" \
    --train_phase feature \
    --loc_teacher dense \
    --loc_interval "$LOC_INTERVAL" \
    --loc_anchors "$LOC_ANCHORS" \
    --train_seed "$DENSEKL_TRAIN_SEED" \
    --loc_desc_weight 0.0 \
    --loc_reproj_weight 0.0 \
    --loc_dense_kl_weight "$DENSEKL_WEIGHT" \
    --loc_dense_kl_temperature 0.07 \
    --loc_responsibility_topk "$DENSEKL_TOPK" \
    --loc_responsibility_opacity_weight "$DENSEKL_OPACITY_WEIGHT" \
    --loc_responsibility_depth_weight "$DENSEKL_DEPTH_WEIGHT" \
    --query_mode "$DENSEKL_QUERY_MODE" \
    "${POSE_GATE_ARGS[@]}" \
    "${SELECTIVE_KL_ARGS[@]}" \
    "${SPARSE_POSE_CACHE_ARGS[@]}" \
    --loc_proto_weight 0.0 \
    --loc_rank_weight 0.0 \
    --loc_opacity_weight 0.0 \
    --no-use_loc_opacity \
    --save_iterations "${DENSEKL_SAVE_ITERATIONS[@]}" \
    --test_iterations "${DENSEKL_SAVE_ITERATIONS[@]}"
else
  echo "[LA-STDLoc dense-KL] Skip training: found iteration ${DENSEKL_END}."
fi

if [[ "$RUN_EVAL" == "1" ]]; then
  for eval_iteration in "${DENSEKL_EVAL_ITERATIONS[@]}"; do
    "$PYTHON" stdloc.py \
      "${DATA_ARGS[@]}" \
      -m "$DENSEKL_MODEL" \
      --iteration "$eval_iteration" \
      --cfg "$RUN_CFG" \
      --prefix "phase-densekl-${SCENE}-${eval_iteration}-reproj${REPROJECTION_ERROR}" \
      --sparse_only
  done
fi
