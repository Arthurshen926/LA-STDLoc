#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT=${DATA_ROOT:-/mnt/pool/sqy/Cambridge_stdloc}
SCENE=${SCENE:-ShopFacade}
BASELINE_MODEL=${BASELINE_MODEL:-/mnt/pool/sqy/stdloc_la_full_runs/${SCENE}_baseline}
LA_MODEL=${LA_MODEL:-/mnt/pool/sqy/stdloc_la_v03_runs/${SCENE}_v03}
MODEL_ROOT=${MODEL_ROOT:-/mnt/pool/sqy/stdloc_la_geometry_2x2}
CFG=${CFG:-configs/stdloc_cambridge.yaml}
BASELINE_ITERATION=${BASELINE_ITERATION:-30000}
LA_ITERATION=${LA_ITERATION:-32000}
REPROJECTION_ERROR=${REPROJECTION_ERROR:-8}
BALANCED_GRID_ROWS=${BALANCED_GRID_ROWS:-4}
BALANCED_GRID_COLS=${BALANCED_GRID_COLS:-4}
BALANCED_MAX_PER_CELL=${BALANCED_MAX_PER_CELL:-48}
BALANCED_VOXEL_SIZE=${BALANCED_VOXEL_SIZE:-0.25}
BALANCED_MAX_PER_VOXEL=${BALANCED_MAX_PER_VOXEL:-48}
BALANCED_MAX_MATCHES=${BALANCED_MAX_MATCHES:-1024}
BALANCED_POST_MAX_MATCHES=${BALANCED_POST_MAX_MATCHES:-256}
BALANCED_POST_CANDIDATE_POOL=${BALANCED_POST_CANDIDATE_POOL:-1024}
BALANCED_POST_REGULARIZATION=${BALANCED_POST_REGULARIZATION:-1e-4}
BALANCED_POST_SCORE_WEIGHT=${BALANCED_POST_SCORE_WEIGHT:-1e-3}

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

DATA_ARGS=(
  -s "$DATA_ROOT/$SCENE"
  -r 1
  -f sp
  -g 3dgs
  --images processed
  --data_device cpu
)

write_cfg() {
  local dst=$1
  local balanced=$2
  "$PYTHON" - "$CFG" "$dst" "$BASELINE_MODEL" "$balanced" "$REPROJECTION_ERROR" \
    "$BALANCED_GRID_ROWS" "$BALANCED_GRID_COLS" "$BALANCED_MAX_PER_CELL" \
    "$BALANCED_VOXEL_SIZE" "$BALANCED_MAX_PER_VOXEL" "$BALANCED_MAX_MATCHES" \
    "$BALANCED_POST_MAX_MATCHES" "$BALANCED_POST_CANDIDATE_POOL" \
    "$BALANCED_POST_REGULARIZATION" "$BALANCED_POST_SCORE_WEIGHT" <<'PY'
import sys
import yaml

(
    src,
    dst,
    baseline_model,
    balanced,
    reprojection_error,
    grid_rows,
    grid_cols,
    max_per_cell,
    voxel_size,
    max_per_voxel,
    max_matches,
    post_max_matches,
    post_candidate_pool,
    post_regularization,
    post_score_weight,
) = sys.argv[1:]
with open(src) as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)
sparse = cfg.setdefault("sparse", {})
sparse["detector_path"] = "detector/30000_detector.pth"
sparse["landmark_path"] = "detector/sampled_idx.pkl"
sparse["detector_model_path"] = baseline_model
sparse["landmark_model_path"] = baseline_model
sparse["landmark_meta_model_path"] = baseline_model
sparse["use_landmark_prior"] = False
sparse["reprojection_error"] = float(reprojection_error)
if balanced == "1":
    sparse["geometry_balance"] = {
        "enabled": True,
        "grid_rows": int(grid_rows),
        "grid_cols": int(grid_cols),
        "max_per_cell": int(max_per_cell),
        "voxel_size": float(voxel_size),
        "max_per_voxel": int(max_per_voxel),
        "max_matches": int(max_matches),
        "post": {
            "enabled": True,
            "max_matches": int(post_max_matches),
            "candidate_pool": int(post_candidate_pool),
            "regularization": float(post_regularization),
            "score_weight": float(post_score_weight),
        },
    }
else:
    sparse["geometry_balance"] = {"enabled": False}
with open(dst, "w") as f:
    yaml.dump(cfg, f)
PY
}

ORIGINAL_CFG="$MODEL_ROOT/${SCENE}_original_reproj${REPROJECTION_ERROR}.yaml"
BALANCED_CFG="$MODEL_ROOT/${SCENE}_balanced_reproj${REPROJECTION_ERROR}.yaml"
write_cfg "$ORIGINAL_CFG" 0
write_cfg "$BALANCED_CFG" 1

"$PYTHON" stdloc.py \
  "${DATA_ARGS[@]}" \
  -m "$BASELINE_MODEL" \
  --iteration "$BASELINE_ITERATION" \
  --cfg "$ORIGINAL_CFG" \
  --prefix "phase-2x2-baseline-original" \
  --sparse_only

"$PYTHON" stdloc.py \
  "${DATA_ARGS[@]}" \
  -m "$BASELINE_MODEL" \
  --iteration "$BASELINE_ITERATION" \
  --cfg "$BALANCED_CFG" \
  --prefix "phase-2x2-baseline-balanced" \
  --sparse_only

"$PYTHON" stdloc.py \
  "${DATA_ARGS[@]}" \
  -m "$LA_MODEL" \
  --iteration "$LA_ITERATION" \
  --cfg "$ORIGINAL_CFG" \
  --prefix "phase-2x2-la-original" \
  --sparse_only

"$PYTHON" stdloc.py \
  "${DATA_ARGS[@]}" \
  -m "$LA_MODEL" \
  --iteration "$LA_ITERATION" \
  --cfg "$BALANCED_CFG" \
  --prefix "phase-2x2-la-balanced" \
  --sparse_only
