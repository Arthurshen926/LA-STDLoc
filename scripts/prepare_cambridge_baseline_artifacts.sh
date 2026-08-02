#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT=${DATA_ROOT:-/mnt/pool/sqy/Cambridge_stdloc}
SOURCE_ROOT=${SOURCE_ROOT:-}
TARGET_ROOT=${TARGET_ROOT:-/mnt/pool/sqy/stdloc_la_full_runs}
SCENES=${SCENES:-ShopFacade KingsCollege OldHospital}
BASELINE_ITERS=${BASELINE_ITERS:-30000}
DETECTOR_ITERS=${DETECTOR_ITERS:-30000}
SKIP_DETECTOR=${SKIP_DETECTOR:-0}
REQUIRE_LOC_FEATURE=${REQUIRE_LOC_FEATURE:-1}
TRAIN_MISSING_BASELINE=${TRAIN_MISSING_BASELINE:-0}
FORCE_BASELINE_TRAIN=${FORCE_BASELINE_TRAIN:-0}
CFG=${CFG:-configs/stdloc_cambridge.yaml}

if [[ -z "$SOURCE_ROOT" ]]; then
  echo "SOURCE_ROOT is required and must contain one subdirectory per Cambridge scene." >&2
  exit 2
fi

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

ply_has_loc_feature() {
  local ply_path=$1
  "$PYTHON" - "$ply_path" <<'PY'
from pathlib import Path
import sys

data = Path(sys.argv[1]).read_bytes()[:65536]
header = data.split(b"end_header", 1)[0].decode("latin1", errors="ignore")
has_loc = any(line.startswith("property") and " loc_0" in line for line in header.splitlines())
raise SystemExit(0 if has_loc else 1)
PY
}

train_full_baseline() {
  local scene=$1
  local target_model=$2
  if [[ -d "$target_model" && "$FORCE_BASELINE_TRAIN" != "1" ]]; then
    echo "[prepare baseline] Skip full baseline training for $scene: target exists at $target_model."
    echo "[prepare baseline] Set FORCE_BASELINE_TRAIN=1 to replace it."
    return 0
  fi
  if [[ "$FORCE_BASELINE_TRAIN" == "1" ]]; then
    rm -rf "$target_model"
  fi
  echo "[prepare baseline] Train full STDLoc baseline for $scene -> $target_model"
  "$PYTHON" train.py \
    -s "$DATA_ROOT/$scene" \
    -m "$target_model" \
    -r 1 -f sp -g 3dgs --images processed --data_device cpu \
    --densify_grad_threshold 0.0004 \
    --position_lr_init 0.000016 \
    --scaling_lr 0.001 \
    --iterations "$BASELINE_ITERS" \
    --train_detector \
    --test_iterations "$BASELINE_ITERS" \
    --save_iterations "$BASELINE_ITERS" \
    --test_detector_iterations "$DETECTOR_ITERS" \
    --save_detector_iterations "$DETECTOR_ITERS" \
    --detector_folder detector
}

mkdir -p "$TARGET_ROOT"

for scene in $SCENES; do
  SOURCE_MODEL="$SOURCE_ROOT/$scene"
  TARGET_MODEL="$TARGET_ROOT/${scene}_baseline"
  SOURCE_PLY="$SOURCE_MODEL/point_cloud/iteration_${BASELINE_ITERS}/point_cloud.ply"
  TARGET_PLY="$TARGET_MODEL/point_cloud/iteration_${BASELINE_ITERS}/point_cloud.ply"
  if [[ ! -d "$DATA_ROOT/$scene" ]]; then
    echo "[prepare baseline] Skip $scene: missing data directory $DATA_ROOT/$scene."
    continue
  fi
  if [[ -f "$TARGET_MODEL/detector/${DETECTOR_ITERS}_detector.pth" && -f "$TARGET_MODEL/detector/sampled_idx.pkl" ]]; then
    echo "[prepare baseline] Skip detector for $scene: artifacts already exist."
    continue
  fi
  if [[ ! -f "$SOURCE_PLY" ]]; then
    if [[ "$TRAIN_MISSING_BASELINE" == "1" ]]; then
      train_full_baseline "$scene" "$TARGET_MODEL"
      continue
    fi
    echo "[prepare baseline] Skip $scene: missing source point cloud $SOURCE_PLY."
    continue
  fi
  if [[ "$REQUIRE_LOC_FEATURE" == "1" ]] && ! ply_has_loc_feature "$SOURCE_PLY"; then
    if [[ "$TRAIN_MISSING_BASELINE" == "1" ]]; then
      train_full_baseline "$scene" "$TARGET_MODEL"
      continue
    fi
    echo "[prepare baseline] Skip $scene: source point cloud lacks loc_* feature fields: $SOURCE_PLY."
    echo "[prepare baseline] Run train.py --train_detector for a full STDLoc baseline, or set SOURCE_ROOT to an existing feature-3DGS baseline."
    continue
  fi
  if [[ ! -d "$TARGET_MODEL" || "${FORCE_BASELINE_COPY:-0}" == "1" ]]; then
    rm -rf "$TARGET_MODEL"
    mkdir -p "$(dirname "$TARGET_MODEL")"
    cp -a "$SOURCE_MODEL" "$TARGET_MODEL"
  fi
  if [[ ! -f "$TARGET_PLY" ]]; then
    echo "[prepare baseline] Skip $scene: target point cloud copy failed at $TARGET_MODEL."
    continue
  fi
  if [[ "$REQUIRE_LOC_FEATURE" == "1" ]] && ! ply_has_loc_feature "$TARGET_PLY"; then
    echo "[prepare baseline] Skip $scene: target point cloud lacks loc_* feature fields after copy: $TARGET_PLY."
    continue
  fi
  if [[ "$SKIP_DETECTOR" == "1" ]]; then
    echo "[prepare baseline] Skip detector for $scene: SKIP_DETECTOR=1."
    continue
  fi
  echo "[prepare baseline] Train baseline detector for $scene -> $TARGET_MODEL/detector"
  "$PYTHON" train_detector.py \
    -s "$DATA_ROOT/$scene" \
    -m "$TARGET_MODEL" \
    -r 1 -f sp -g 3dgs --images processed --data_device cpu \
    --iteration "$BASELINE_ITERS" \
    --iterations "$DETECTOR_ITERS" \
    --detector_folder detector \
    --sampling_mode baseline \
    --detector_target_mode hard \
    --test_iterations "$DETECTOR_ITERS" \
    --save_iterations "$DETECTOR_ITERS"
done
