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
DENSEKL_WEIGHT=${DENSEKL_WEIGHT:-0.02}
DENSEKL_TOPK=${DENSEKL_TOPK:-32}
DENSEKL_OPACITY_WEIGHT=${DENSEKL_OPACITY_WEIGHT:-0.0}
DENSEKL_DEPTH_WEIGHT=${DENSEKL_DEPTH_WEIGHT:-0.0}
LOC_INTERVAL=${LOC_INTERVAL:-1}
LOC_ANCHORS=${LOC_ANCHORS:-512}
REPROJECTION_ERROR=${REPROJECTION_ERROR:-12}
RUN_DIAGNOSTICS=${RUN_DIAGNOSTICS:-1}
RUN_EVAL=${RUN_EVAL:-1}

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

if [[ ! -d "$DENSEKL_MODEL" || "${FORCE_DENSEKL_COPY:-0}" == "1" ]]; then
  rm -rf "$DENSEKL_MODEL"
  mkdir -p "$(dirname "$DENSEKL_MODEL")"
  cp -a "$SOURCE_MODEL" "$DENSEKL_MODEL"
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

if ! point_cloud_exists "$DENSEKL_MODEL" "$DENSEKL_END"; then
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
    --loc_desc_weight 0.0 \
    --loc_reproj_weight 0.0 \
    --loc_dense_kl_weight "$DENSEKL_WEIGHT" \
    --loc_dense_kl_temperature 0.07 \
    --loc_responsibility_topk "$DENSEKL_TOPK" \
    --loc_responsibility_opacity_weight "$DENSEKL_OPACITY_WEIGHT" \
    --loc_responsibility_depth_weight "$DENSEKL_DEPTH_WEIGHT" \
    --loc_proto_weight 0.0 \
    --loc_rank_weight 0.0 \
    --loc_opacity_weight 0.0 \
    --no-use_loc_opacity \
    --save_iterations "$DENSEKL_END" \
    --test_iterations "$DENSEKL_END"
else
  echo "[LA-STDLoc dense-KL] Skip training: found iteration ${DENSEKL_END}."
fi

if [[ "$RUN_EVAL" == "1" ]]; then
  "$PYTHON" stdloc.py \
    "${DATA_ARGS[@]}" \
    -m "$DENSEKL_MODEL" \
    --iteration "$DENSEKL_END" \
    --cfg "$RUN_CFG" \
    --prefix "phase-densekl-${SCENE}-${DENSEKL_END}-reproj${REPROJECTION_ERROR}" \
    --sparse_only
fi
