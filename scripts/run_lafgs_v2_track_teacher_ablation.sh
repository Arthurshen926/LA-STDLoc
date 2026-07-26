#!/usr/bin/env bash
set -euo pipefail

# G0/G1/G2 geometry-teacher controls on the frozen OldHospital 48K A4 field.
# This runner intentionally uses only physical GPU 1.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/root/miniconda3/envs/cybersim_agent/bin/python}"
RUN_ROOT="${LAFGS_TRACK_TEACHER_ROOT:-/mnt/pool/sqy/stdloc_lafgs_track_teacher_20260726/OldHospital}"
CANONICAL_ROOT="/mnt/pool/sqy/stdloc_lafgs_rgb_prior_sanitization_20260725/OldHospital/runs/rgb_2dgs_robustq01_wide48_to32"
STATE="$CANONICAL_ROOT/stage_a_2500/1000_lafgs_map_state.pt"
BASE_STATISTICS="$CANONICAL_ROOT/statistics_combined_1000_frozen_independent/landmark_statistics_full.pt"
QUERY_CACHE="$CANONICAL_ROOT/query_cache_native_fullres_k2048.pt"
C1_ROOT="$CANONICAL_ROOT/controlled_anchor_f0p1_m0p1_b32000_independent_v2"
C1_STATE="$C1_ROOT/corrupted/corrupted_lafgs_map_state.pt"
C1_LABELS="$C1_ROOT/corrupted/corruption_labels.pt"

export CUDA_VISIBLE_DEVICES=1
export CUDA_HOME=/usr/local/cuda-11.8
export PATH="/root/miniconda3/envs/cybersim_agent/bin:$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="/root/miniconda3/envs/cybersim_agent/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$REPO_ROOT"
export PYTHONHASHSEED=2026

mkdir -p "$RUN_ROOT"
cd "$REPO_ROOT"

track_lgcv_args=()
if [[ "${LAFGS_GEOMETRY_TEACHER_TRACK_LGCV:-0}" == "1" ]]; then
  track_lgcv_args+=(--track_lgcv)
fi

for required in "$STATE" "$BASE_STATISTICS" "$QUERY_CACHE" "$C1_STATE" "$C1_LABELS"; do
  [[ -f "$required" ]] || { echo "Missing required input: $required" >&2; exit 1; }
done

for mode in map_top1 gt_clean_map_top1 track_first; do
  case "$mode" in
    map_top1) label="g0_map_top1" ;;
    gt_clean_map_top1) label="g1_gt_clean_top1" ;;
    track_first) label="g2_track_first" ;;
  esac
  output_dir="$RUN_ROOT/$label"
  statistics="$output_dir/landmark_statistics_full.pt"
  if [[ ! -f "$statistics" ]]; then
    mkdir -p "$output_dir"
    "$PYTHON" scripts/build_lafgs_geometry_teacher.py \
      --query_cache "$QUERY_CACHE" \
      --state "$STATE" \
      --base_statistics "$BASE_STATISTICS" \
      --output "$statistics" \
      --mode "$mode" \
      --view_direction_weight 0.5 \
      --parallax_quantile 0.75 \
      --max_covariance_trace_m2 0.01 \
      --max_rendered_depth_residual_m 0.15 \
      --min_rendered_depth_observations 2 \
      --track_pair_neighbors 6 \
      --track_min_similarity 0.65 \
      --track_min_margin 0.01 \
      --track_max_epipolar_error_px 2 \
      "${track_lgcv_args[@]}" \
      --track_lgcv_neighbors "${LAFGS_GEOMETRY_TEACHER_TRACK_LGCV_NEIGHBORS:-8}" \
      --track_lgcv_support_threshold "${LAFGS_GEOMETRY_TEACHER_TRACK_LGCV_SUPPORT_THRESHOLD:-4}" \
      --track_lgcv_angle_cosine "${LAFGS_GEOMETRY_TEACHER_TRACK_LGCV_ANGLE_COSINE:-0.9659}" \
      --track_lgcv_scale_threshold "${LAFGS_GEOMETRY_TEACHER_TRACK_LGCV_SCALE_THRESHOLD:-0.1}" \
      --track_lgcv_scale_limit "${LAFGS_GEOMETRY_TEACHER_TRACK_LGCV_SCALE_LIMIT:-3}" \
      --track_lgcv_maximum_edge_px "${LAFGS_GEOMETRY_TEACHER_TRACK_LGCV_MAXIMUM_EDGE_PX:-50}" \
      --track_lgcv_minimum_matches "${LAFGS_GEOMETRY_TEACHER_TRACK_LGCV_MINIMUM_MATCHES:-8}" \
      --track_lgcv_mode "${LAFGS_GEOMETRY_TEACHER_TRACK_LGCV_MODE:-hard}" \
      --track_lgcv_confidence_floor "${LAFGS_GEOMETRY_TEACHER_TRACK_LGCV_CONFIDENCE_FLOOR:-0.25}" \
      --track_assignment_max_distance_m 0.2 \
      2>&1 | tee "$output_dir/build.log"
  fi

  rebound="$output_dir/c1_rebound_statistics.pt"
  if [[ ! -f "$rebound" ]]; then
    "$PYTHON" scripts/rebind_geometry_teacher_to_state.py \
      --statistics "$statistics" \
      --state "$C1_STATE" \
      --output "$rebound"
  fi
  sanitized="$output_dir/c1_hard_geo_loc"
  if [[ ! -f "$sanitized/sanitization_report.json" ]]; then
    "$PYTHON" scripts/sanitize_lafgs_landmarks.py \
      --source_state "$C1_STATE" \
      --statistics "$rebound" \
      --output_dir "$sanitized" \
      --mode hard_geo_loc \
      --budget 32000 \
      --outlier_labels "$C1_LABELS"
  fi
done

"$PYTHON" - "$RUN_ROOT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
summary = {}
for directory in sorted(path for path in root.iterdir() if path.is_dir()):
    report_path = directory / "c1_hard_geo_loc" / "sanitization_report.json"
    if not report_path.is_file():
        continue
    report = json.loads(report_path.read_text())
    summary[directory.name] = {
        "state_counts": report.get("state_counts", {}),
        "controlled_outlier_evaluation": report.get(
            "controlled_outlier_evaluation", {}
        ),
    }
print(json.dumps(summary, indent=2, sort_keys=True))
PY
