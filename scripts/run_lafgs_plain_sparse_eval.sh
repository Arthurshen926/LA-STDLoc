#!/usr/bin/env bash
set -euo pipefail

# Evaluate an arbitrary localization map under the frozen plain sparse protocol.
if [[ $# -lt 6 || $# -gt 7 ]]; then
  echo "Usage: bash $0 <scene> <gpu> <label> <map.pt> <metric.pt> <output-root> [seed]" >&2
  exit 2
fi

SCENE="$1"
GPU="$2"
LABEL="$3"
MAP_PATH="$4"
METRIC_PATH="$5"
OUTPUT_ROOT="$6"
SEED="${7:-2026}"
BASE_ROOT="${LAFGS_V1_MULTISCENE_ROOT:-/mnt/pool/sqy/stdloc_lafgs_v1_frozen_multiscene_20260731}"
DATA_ROOT="${CAMBRIDGE_DATA_ROOT:-/mnt/pool/sqy/Cambridge_stdloc}"
PYTHON="${PYTHON:-/root/miniconda3/envs/cybersim_agent/bin/python}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCENE_ROOT="$BASE_ROOT/$SCENE"
MODEL_ROOT="${LAFGS_EVAL_MODEL_ROOT_OVERRIDE:-$SCENE_ROOT/prior/rgb_matcha_2dgs}"
GAUSSIAN_TYPE="${LAFGS_EVAL_GAUSSIAN_TYPE_OVERRIDE:-2dgs}"
SH_DEGREE="${LAFGS_EVAL_SH_DEGREE_OVERRIDE:-3}"
SPARSE_FRONTEND="${LAFGS_EVAL_SPARSE_FRONTEND_OVERRIDE:-ulfloc_native_metric}"
MAP_INPUT_MODE="${LAFGS_EVAL_MAP_INPUT_MODE_OVERRIDE:-materialized}"
SOURCE_ROOT="$DATA_ROOT/$SCENE"
BOOTSTRAP="$SCENE_ROOT/runs/frozen_v1/bootstrap"
OUTPUT="$OUTPUT_ROOT/$SCENE/$LABEL/seed$SEED"
CFG="$OUTPUT/config.yaml"

for path in "$MAP_PATH" "$METRIC_PATH" "$MODEL_ROOT" \
  "$BOOTSTRAP/sampled_idx.pkl" "$BOOTSTRAP/landmark_meta.pt"; do
  [[ -e "$path" ]] || { echo "Missing required artifact: $path" >&2; exit 1; }
done
if [[ -s "$OUTPUT/evaluation_summary.json" ]]; then
  cat "$OUTPUT/evaluation_summary.json"
  exit 0
fi

export CUDA_HOME=/usr/local/cuda-11.8
export PATH="/root/miniconda3/envs/cybersim_agent/bin:$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="/root/miniconda3/envs/cybersim_agent/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$REPO_ROOT"
mkdir -p "$OUTPUT/results"
cd "$REPO_ROOT"

case "$MAP_INPUT_MODE" in
  materialized)
    MAP_ARGS=(--materialized_anchor_map_path "$MAP_PATH")
    ;;
  feature_override)
    MAP_ARGS=(
      --landmark_feature_override_path "$MAP_PATH"
      --override_landmark_features
    )
    ;;
  *)
    echo "Unsupported map input mode: $MAP_INPUT_MODE" >&2
    exit 2
    ;;
esac

"$PYTHON" scripts/make_stdloc_eval_cfg.py \
  --base_cfg configs/stdloc_cambridge.yaml --output "$CFG" \
  --artifact_model_path "$MODEL_ROOT" \
  --detector_folder ulfloc_native_no_detector --detector_iters 0 \
  --landmark_path "$BOOTSTRAP/sampled_idx.pkl" \
  --landmark_meta_path "$BOOTSTRAP/landmark_meta.pt" \
  --detect_num 2048 --nms 2 --sparse_ransac_seed "$SEED" \
  --sparse_query_feature_contract native_resized_input \
  --reprojection_error 12 --match_threshold 0 --match_topk 1 \
  --max_matches_per_keypoint 0 --max_matches_per_landmark 0 \
  --candidate_frontend_match_policy error \
  --diagnostics --diagnostics_grid_rows 4 --diagnostics_grid_cols 4 \
  --diagnostics_voxel_size 1 \
  --diagnostics_task_translation_scale_m 0.1 \
  --diagnostics_task_rotation_scale_degrees 2 \
  --sparse_frontend "$SPARSE_FRONTEND" \
  "${MAP_ARGS[@]}" \
  --metric_state_path "$METRIC_PATH" > "$OUTPUT/config_build.json"

(
  export CUDA_VISIBLE_DEVICES="$GPU"
  export STDLOC_RESULTS_ROOT="$OUTPUT/results"
  "$PYTHON" stdloc.py \
    --model_path "$MODEL_ROOT" --source_path "$SOURCE_ROOT" \
    --images processed --data_device cpu --gaussian_type "$GAUSSIAN_TYPE" \
    --sh_degree "$SH_DEGREE" --feature_type sp --resolution 1 --longest_edge 0 \
    --norm_before_render --iteration 30000 --cfg "$CFG" \
    --prefix "lafgs-$SCENE-$LABEL-seed$SEED" \
    --sparse_only --evaluation_camera_subset test \
    2>&1 | tee "$OUTPUT/eval.log"
)
RESULT="$(sed -n 's/^Output path: //p' "$OUTPUT/eval.log" | tail -n 1)"
[[ -n "$RESULT" && -f "$RESULT/results_summary.json" ]] || {
  echo "Evaluation failed for $SCENE/$LABEL" >&2
  exit 1
}
printf '%s\n' "$RESULT" > "$OUTPUT/result.path"

"$PYTHON" - "$RESULT" "$MAP_PATH" "$METRIC_PATH" "$LABEL" "$SPARSE_FRONTEND" > "$OUTPUT/evaluation_summary.json" <<'PY'
import json
import sys
from pathlib import Path

import numpy as np
import torch

result = Path(sys.argv[1])
rows = json.loads((result / "results.json").read_text())
report = json.loads((result / "results_summary.json").read_text())
te = np.asarray([float(row["sparse_TE"]) for row in rows])
ae = np.asarray([float(row["sparse_AE"]) for row in rows])
diag = report.get("sparse_diagnostics", {})
state = torch.load(sys.argv[2], map_location="cpu", weights_only=False)
anchor_tensor = state.get(
    "anchor_xyz",
    state.get("landmark_xyz", state.get("landmark_features")),
)
if anchor_tensor is None:
    raise ValueError("evaluation map does not expose an anchor registry")
output = {
    "schema": "lafgs_plain_sparse_evaluation",
    "label": sys.argv[4],
    "query_count": len(rows),
    "anchor_count": int(torch.as_tensor(anchor_tensor).shape[0]),
    "median_te_cm": float(np.median(te)),
    "mean_te_cm": float(te.mean()),
    "p90_te_cm": float(np.percentile(te, 90)),
    "median_ae_deg": float(np.median(ae)),
    "mean_ae_deg": float(ae.mean()),
    "recall_5cm_5deg_percent": 100.0 * float(report["sparse"]["recall_5cm_5d"]),
    "raw_gt_precision_2px_percent": 100.0 * float(
        diag.get("sparse_diag_all_gt_precision_2px_mean", 0.0)
    ),
    "inlier_gt_precision_2px_percent": 100.0 * float(
        diag.get("sparse_diag_inlier_gt_precision_2px_mean", 0.0)
    ),
    "solver_inlier_ratio_percent": 100.0 * float(
        diag.get("sparse_diag_ransac_inlier_ratio_solver_mean", 0.0)
    ),
    "mean_hypotheses": diag.get("sparse_diag_ransac_actual_hypotheses_mean"),
    "matching_ms": diag.get("sparse_diag_runtime_matching_ms_mean"),
    "ransac_ms": diag.get("sparse_diag_runtime_ransac_ms_mean"),
    "total_ms": diag.get("sparse_diag_runtime_total_ms_mean"),
    "map": str(Path(sys.argv[2]).resolve()),
    "metric_state": str(Path(sys.argv[3]).resolve()),
    "sparse_frontend": sys.argv[5],
    "metric_applied": sys.argv[5] == "ulfloc_native_metric",
    "result_path": str(result.resolve()),
    "deployment": (
        "native SuperPoint, uncapped global top-1, one PoseLib solve"
        + (", learned shared metric" if sys.argv[5] == "ulfloc_native_metric" else "")
    ),
}
print(json.dumps(output, indent=2))
PY
cat "$OUTPUT/evaluation_summary.json"
