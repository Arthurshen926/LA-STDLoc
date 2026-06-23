#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT=${DATA_ROOT:-/mnt/pool/sqy/Cambridge_stdloc}
SCENE=${SCENE:-ShopFacade}
BASELINE_MODEL=${BASELINE_MODEL:-/mnt/pool/sqy/stdloc_la_full_runs/${SCENE}_baseline}
MODEL_ROOT=${MODEL_ROOT:-/mnt/pool/sqy/stdloc_la_v03_runs}
V03_MODEL=${V03_MODEL:-$MODEL_ROOT/${SCENE}_v03}
CFG=${CFG:-configs/stdloc_cambridge.yaml}
BASELINE_ITERS=${BASELINE_ITERS:-30000}
V03_STEPS=${V03_STEPS:-500 1000 2000}
V03_MAX_STEP=${V03_MAX_STEP:-2000}
V03_LOC_ANCHORS=${V03_LOC_ANCHORS:-2048}
V03_DIRECT_WEIGHT=${V03_DIRECT_WEIGHT:-0.05}
V03_MULTIVIEW_WEIGHT=${V03_MULTIVIEW_WEIGHT:-0.03}
V03_FULL_BANK_WEIGHT=${V03_FULL_BANK_WEIGHT:-0.05}
V03_FULL_BANK_HARD_NEGATIVES=${V03_FULL_BANK_HARD_NEGATIVES:-32}
V03_FULL_BANK_MARGIN=${V03_FULL_BANK_MARGIN:-0.2}
V03_ANCHOR_WEIGHT=${V03_ANCHOR_WEIGHT:-0.01}
V03_TRAIN_SEED=${V03_TRAIN_SEED:-0}
V03_QUERY_SPLIT_SEED=${V03_QUERY_SPLIT_SEED:-2025}
V03_QUERY_SPLIT_MODE=${V03_QUERY_SPLIT_MODE:-random}
RUN_SWEEP=${RUN_SWEEP:-1}
SWEEP_THRESHOLDS=${SWEEP_THRESHOLDS:-2 4 6 8 10 12 16}
RUN_CFG="$MODEL_ROOT/${SCENE}_stdloc_baseline_artifacts.yaml"

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
"$PYTHON" - "$CFG" "$RUN_CFG" "$BASELINE_MODEL" <<'PY'
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

if [[ ! -d "$V03_MODEL" || "${FORCE_V03_COPY:-0}" == "1" ]]; then
  rm -rf "$V03_MODEL"
  mkdir -p "$(dirname "$V03_MODEL")"
  cp -a "$BASELINE_MODEL" "$V03_MODEL"
fi

V03_END=$((BASELINE_ITERS + V03_MAX_STEP))
SAVE_ITERS=()
for step in $V03_STEPS; do
  SAVE_ITERS+=("$((BASELINE_ITERS + step))")
done

if ! point_cloud_exists "$V03_MODEL" "$V03_END"; then
  "$PYTHON" train_locaware.py \
    "${DATA_ARGS[@]}" \
    "${TRAIN_ARGS[@]}" \
    -m "$V03_MODEL" \
    --load_iteration "$BASELINE_ITERS" \
    --iterations "$V03_END" \
    --train_phase feature \
    --loc_teacher direct \
    --landmark_path "$BASELINE_MODEL/detector/sampled_idx.pkl" \
    --loc_interval 1 \
    --loc_anchors "$V03_LOC_ANCHORS" \
    --loc_direct_weight "$V03_DIRECT_WEIGHT" \
    --loc_multiview_weight "$V03_MULTIVIEW_WEIGHT" \
    --loc_multiview_temperature 0.07 \
    --loc_multiview_slots 4 \
    --loc_full_bank_weight "$V03_FULL_BANK_WEIGHT" \
    --loc_full_bank_temperature 0.07 \
    --loc_full_bank_hard_negatives "$V03_FULL_BANK_HARD_NEGATIVES" \
    --loc_full_bank_margin "$V03_FULL_BANK_MARGIN" \
    --loc_anchor_weight "$V03_ANCHOR_WEIGHT" \
    --loc_desc_weight 0.0 \
    --loc_reproj_weight 0.0 \
    --loc_proto_weight 0.0 \
    --loc_rank_weight 0.0 \
    --loc_opacity_weight 0.0 \
    --no-use_loc_opacity \
    --support_query_split \
    --query_holdout_ratio 0.2 \
    --train_seed "$V03_TRAIN_SEED" \
    --query_split_seed "$V03_QUERY_SPLIT_SEED" \
    --query_split_mode "$V03_QUERY_SPLIT_MODE" \
    --direct_depth_check \
    --direct_depth_abs_tolerance 0.001 \
    --direct_depth_rel_tolerance 0.01 \
    --save_iterations "${SAVE_ITERS[@]}" \
    --test_iterations "${SAVE_ITERS[@]}"
else
  echo "[LA-STDLoc v0.3] Skip feature distillation: found iteration ${V03_END}."
fi

for checkpoint in "${SAVE_ITERS[@]}"; do
  "$PYTHON" stdloc.py \
    "${DATA_ARGS[@]}" \
    -m "$V03_MODEL" \
    --iteration "$checkpoint" \
    --cfg "$RUN_CFG" \
    --prefix "phase-v03-${checkpoint}" \
    --sparse_only
done

if [[ "$RUN_SWEEP" == "1" ]]; then
  for reproj in $SWEEP_THRESHOLDS; do
    SWEEP_CFG="$MODEL_ROOT/${SCENE}_stdloc_baseline_artifacts_reproj${reproj}.yaml"
    "$PYTHON" - "$RUN_CFG" "$SWEEP_CFG" "$reproj" <<'PY'
import sys
import yaml

src, dst, reproj = sys.argv[1], sys.argv[2], float(sys.argv[3])
with open(src) as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)
cfg.setdefault("sparse", {})["reprojection_error"] = reproj
with open(dst, "w") as f:
    yaml.dump(cfg, f)
PY
    "$PYTHON" stdloc.py \
      "${DATA_ARGS[@]}" \
      -m "$BASELINE_MODEL" \
      --iteration "$BASELINE_ITERS" \
      --cfg "$SWEEP_CFG" \
      --prefix "phase0-baseline-reproj${reproj}" \
      --sparse_only
    "$PYTHON" stdloc.py \
      "${DATA_ARGS[@]}" \
      -m "$V03_MODEL" \
      --iteration "$V03_END" \
      --cfg "$SWEEP_CFG" \
      --prefix "phase-v03-${V03_END}-reproj${reproj}" \
      --sparse_only
  done
fi
