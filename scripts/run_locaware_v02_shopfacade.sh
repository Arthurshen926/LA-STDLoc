#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT=${DATA_ROOT:-/mnt/pool/sqy/Cambridge_stdloc}
SCENE=${SCENE:-ShopFacade}
BASELINE_MODEL=${BASELINE_MODEL:-/mnt/pool/sqy/stdloc_la_full_runs/${SCENE}_baseline}
LA_FULL_MODEL=${LA_FULL_MODEL:-/mnt/pool/sqy/stdloc_la_full_runs/${SCENE}_la}
MODEL_ROOT=${MODEL_ROOT:-/mnt/pool/sqy/stdloc_la_v02_runs}
V02_MODEL="$MODEL_ROOT/${SCENE}_v02"
CFG=${CFG:-configs/stdloc_cambridge.yaml}
BASELINE_ITERS=${BASELINE_ITERS:-30000}
V02_STEPS=${V02_STEPS:-500 1000 2000}
V02_MAX_STEP=${V02_MAX_STEP:-2000}
V02_LOC_ANCHORS=${V02_LOC_ANCHORS:-2048}
RUN_BASELINE_DENSE=${RUN_BASELINE_DENSE:-1}
RUN_E1=${RUN_E1:-1}
RUN_E3=${RUN_E3:-0}
E3_ITERATION=${E3_ITERATION:-40000}
E3_DETECTOR_ITERS=${E3_DETECTOR_ITERS:-30000}
E3_DETECTOR_FOLDER=${E3_DETECTOR_FOLDER:-detector_e3_baseline_hard}
RUN_CFG="$MODEL_ROOT/${SCENE}_stdloc_baseline_artifacts.yaml"
E3_CFG="$MODEL_ROOT/${SCENE}_stdloc_e3_baseline_hard.yaml"

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

if [[ ! -d "$V02_MODEL" || "${FORCE_V02_COPY:-0}" == "1" ]]; then
  rm -rf "$V02_MODEL"
  mkdir -p "$(dirname "$V02_MODEL")"
  cp -a "$BASELINE_MODEL" "$V02_MODEL"
fi

if [[ "$RUN_BASELINE_DENSE" == "1" ]]; then
  "$PYTHON" stdloc.py \
    "${DATA_ARGS[@]}" \
    -m "$BASELINE_MODEL" \
    --iteration "$BASELINE_ITERS" \
    --cfg "$CFG" \
    --prefix "phase0-baseline-dense"
else
  echo "[LA-STDLoc v0.2] Skip baseline dense control: RUN_BASELINE_DENSE=${RUN_BASELINE_DENSE}."
fi

if [[ "$RUN_E1" == "1" ]]; then
  "$PYTHON" stdloc.py \
    "${DATA_ARGS[@]}" \
    -m "$LA_FULL_MODEL" \
    --iteration 33000 \
    --cfg "$RUN_CFG" \
    --prefix "phase-e1-33k-fixed-baseline-sparse" \
    --sparse_only
else
  echo "[LA-STDLoc v0.2] Skip E1 control: RUN_E1=${RUN_E1}."
fi

if [[ "$RUN_E3" == "1" ]]; then
  E3_DETECTOR_PATH="$LA_FULL_MODEL/$E3_DETECTOR_FOLDER/${E3_DETECTOR_ITERS}_detector.pth"
  if [[ ! -f "$E3_DETECTOR_PATH" || "${FORCE_E3_DETECTOR:-0}" == "1" ]]; then
    if [[ "${FORCE_E3_DETECTOR:-0}" == "1" ]]; then
      rm -rf "$LA_FULL_MODEL/$E3_DETECTOR_FOLDER"
    fi
    "$PYTHON" train_detector.py \
      "${DATA_ARGS[@]}" \
      "${TRAIN_ARGS[@]}" \
      -m "$LA_FULL_MODEL" \
      --iteration "$E3_ITERATION" \
      --iterations "$E3_DETECTOR_ITERS" \
      --detector_folder "$E3_DETECTOR_FOLDER" \
      --sampling_mode baseline \
      --detector_target_mode hard \
      --test_iterations "$E3_DETECTOR_ITERS" \
      --save_iterations "$E3_DETECTOR_ITERS"
  else
    echo "[LA-STDLoc v0.2] Skip E3 detector training: found $E3_DETECTOR_PATH."
  fi
  "$PYTHON" - "$CFG" "$E3_CFG" "$E3_DETECTOR_FOLDER" "$E3_DETECTOR_ITERS" <<'PY'
import sys
import yaml

src, dst, detector_folder, detector_iters = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
with open(src) as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)
sparse = cfg.setdefault("sparse", {})
sparse["detector_path"] = f"{detector_folder}/{detector_iters}_detector.pth"
sparse["landmark_path"] = f"{detector_folder}/sampled_idx.pkl"
sparse.pop("detector_model_path", None)
sparse.pop("landmark_model_path", None)
sparse.pop("landmark_meta_model_path", None)
sparse["use_landmark_prior"] = False
with open(dst, "w") as f:
    yaml.dump(cfg, f)
PY
  "$PYTHON" stdloc.py \
    "${DATA_ARGS[@]}" \
    -m "$LA_FULL_MODEL" \
    --iteration "$E3_ITERATION" \
    --cfg "$E3_CFG" \
    --prefix "phase-e3-40k-baseline-hard-sparse" \
    --sparse_only
else
  echo "[LA-STDLoc v0.2] Skip E3 control by default: final-map topology does not preserve baseline landmark indices. Set RUN_E3=1 to train a final-map baseline-spatial hard detector."
fi

V02_END=$((BASELINE_ITERS + V02_MAX_STEP))
SAVE_ITERS=()
for step in $V02_STEPS; do
  SAVE_ITERS+=("$((BASELINE_ITERS + step))")
done

if ! point_cloud_exists "$V02_MODEL" "$V02_END"; then
  "$PYTHON" train_locaware.py \
    "${DATA_ARGS[@]}" \
    "${TRAIN_ARGS[@]}" \
    -m "$V02_MODEL" \
    --load_iteration "$BASELINE_ITERS" \
    --iterations "$V02_END" \
    --train_phase feature \
    --loc_teacher direct \
    --landmark_path "$BASELINE_MODEL/detector/sampled_idx.pkl" \
    --loc_interval 1 \
    --loc_anchors "$V02_LOC_ANCHORS" \
    --loc_direct_weight 0.1 \
    --loc_multiview_weight 0.05 \
    --loc_multiview_temperature 0.07 \
    --loc_multiview_slots 4 \
    --loc_desc_weight 0.0 \
    --loc_reproj_weight 0.0 \
    --loc_proto_weight 0.0 \
    --loc_rank_weight 0.0 \
    --loc_opacity_weight 0.0 \
    --no-use_loc_opacity \
    --support_query_split \
    --query_holdout_ratio 0.2 \
    --query_split_seed 2025 \
    --direct_depth_check \
    --direct_depth_abs_tolerance 0.001 \
    --direct_depth_rel_tolerance 0.01 \
    --save_iterations "${SAVE_ITERS[@]}" \
    --test_iterations "${SAVE_ITERS[@]}"
else
  echo "[LA-STDLoc v0.2] Skip feature distillation: found iteration ${V02_END}."
fi

for checkpoint in "${SAVE_ITERS[@]}"; do
  "$PYTHON" stdloc.py \
    "${DATA_ARGS[@]}" \
    -m "$V02_MODEL" \
    --iteration "$checkpoint" \
    --cfg "$RUN_CFG" \
    --prefix "phase-v02-${checkpoint}" \
    --sparse_only
done
