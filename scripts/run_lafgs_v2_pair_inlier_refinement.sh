#!/usr/bin/env bash
set -euo pipefail

# Validation-only true Pair/LGCV sidecar.  The dense stage is seeded from the
# exact sparse RANSAC inliers emitted by the same frozen LaFGS map/configuration;
# it is never an official-test or mainline sparse-localization result.

if [[ $# -ne 3 ]]; then
  echo "Usage: $0 <scene> <gpu> <lafgs_field|prior_rgb>" >&2
  exit 2
fi

SCENE="$1"
GPU="$2"
REPRESENTATION="$3"

case "$SCENE" in
  OldHospital|ShopFacade) ;;
  *) echo "True-Pair runner currently supports OldHospital or ShopFacade" >&2; exit 2 ;;
esac
case "$GPU" in
  0|1|2) ;;
  *) echo "GPU must be 0, 1, or 2" >&2; exit 2 ;;
esac
case "$REPRESENTATION" in
  lafgs_field|prior_rgb) ;;
  *) echo "Representation must be lafgs_field or prior_rgb" >&2; exit 2 ;;
esac

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/root/miniconda3/envs/cybersim_agent/bin/python}"
DATA_ROOT="${CAMBRIDGE_DATA_ROOT:-/mnt/pool/sqy/Cambridge_stdloc}"
REFERENCE_ROOT="${LAFGS_V2_ULFPARITY_REFERENCE_ROOT:-/mnt/pool/sqy/stdloc_lafgs_v2_ulfparity_alternating_20260721}"
EXPERIMENT_ROOT="${LAFGS_V2_PAIR_INLIER_ROOT:-/mnt/pool/sqy/stdloc_lafgs_v2_pair_inlier_20260722}"
MODEL_ROOT="$REFERENCE_ROOT/matcha_wrappers/$SCENE"

# Locked sparse deployment contract.  The only new element is the optional
# validation-only local sidecar after the sparse result has already been made.
LONGEST_EDGE=0
NATIVE_KEYPOINTS=2048
MATCH_THRESHOLD=0
MAX_MATCHES_PER_LANDMARK=0
EVAL_REPROJECTION_PX=12
VALIDATION_RATIO="${LAFGS_PAIR_VALIDATION_RATIO:-0.2}"
# This sidecar used to silently retain the legacy temporal split, which made
# it unsafe to attach to the formal robust native-residual path.  Keep the
# current frozen protocol as the default and require an explicit override for
# historical temporal-block artifacts.
SPLIT_MODE="${LAFGS_PAIR_SPLIT_MODE:-stratified_temporal_block}"
SPLIT_SEED="${LAFGS_PAIR_SPLIT_SEED:-2026}"
CAMERA_LOADER_WORKERS="${LAFGS_PAIR_CAMERA_LOADER_WORKERS:-0}"

SOURCE_RUN_ROOT="${LAFGS_PAIR_SOURCE_RUN_ROOT:-$REFERENCE_ROOT/$SCENE/ulfparity_native16000_s128_k2048_ulfloc_parity_tau0_cap0_v8_fullres_native_uncapped_pure_native}"
FIELD_STATE="${LAFGS_PAIR_FIELD_STATE:-$SOURCE_RUN_ROOT/residual_5000/5000_lafgs_map_state.pt}"
LANDMARK_IDS="${LAFGS_PAIR_LANDMARK_IDS:-$SOURCE_RUN_ROOT/bootstrap/sampled_idx.pkl}"
LANDMARK_META="${LAFGS_PAIR_LANDMARK_META:-$SOURCE_RUN_ROOT/bootstrap/landmark_meta.pt}"
# Keep the cache tied to the supplied field run by default.  Old historical
# runs can still select their legacy cache through LAFGS_PAIR_QUERY_CACHE_PATH.
QUERY_CACHE="${LAFGS_PAIR_QUERY_CACHE_PATH:-$SOURCE_RUN_ROOT/query_cache_native_fullres_k2048.pt}"
RUN_LABEL="${LAFGS_PAIR_LABEL:-u0_residual5k_native}"
PAIR_POSE_SOLVER="${LAFGS_PAIR_POSE_SOLVER:-prior_gn}"
TASK_TRANSLATION_SCALE_M="${LAFGS_PAIR_DIAGNOSTICS_TRANSLATION_SCALE_M:-}"
PRIOR_GN_TRANSLATION_SCALE_M="${LAFGS_PAIR_PRIOR_GN_TRANSLATION_SCALE_M:-}"
PRIOR_GN_MAX_TRANSLATION_M="${LAFGS_PAIR_PRIOR_GN_MAX_TRANSLATION_M:-}"

case "$PAIR_POSE_SOLVER" in
  prior_gn|ransac_pnp) ;;
  *) echo "LAFGS_PAIR_POSE_SOLVER must be prior_gn or ransac_pnp" >&2; exit 2 ;;
esac
case "$SPLIT_MODE" in
  temporal_block|stratified_temporal_block) ;;
  *) echo "LAFGS_PAIR_SPLIT_MODE must be temporal_block or stratified_temporal_block" >&2; exit 2 ;;
esac
if ! [[ "$SPLIT_SEED" =~ ^[0-9]+$ ]]; then
  echo "LAFGS_PAIR_SPLIT_SEED must be a non-negative integer" >&2
  exit 2
fi
if ! [[ "$CAMERA_LOADER_WORKERS" =~ ^[0-9]+$ ]]; then
  echo "LAFGS_PAIR_CAMERA_LOADER_WORKERS must be a non-negative integer" >&2
  exit 2
fi

RUN_ROOT="$EXPERIMENT_ROOT/$SCENE/${RUN_LABEL}_${REPRESENTATION}_pair_inlier_local_v1"
CONFIG_ROOT="$RUN_ROOT/configs"
LOG_ROOT="$RUN_ROOT/logs"
RESULT_ROOT="$RUN_ROOT/results"
STDLOC_RESULTS_ROOT="$RUN_ROOT/stdloc_results"
SPARSE_REF="$RESULT_ROOT/sparse_dump.results_path"
MANIFEST="$RUN_ROOT/protocol_manifest.json"

export CUDA_VISIBLE_DEVICES="$GPU"
export CUDA_HOME=/usr/local/cuda-11.8
export PATH="/root/miniconda3/envs/cybersim_agent/bin:$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="/root/miniconda3/envs/cybersim_agent/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$REPO_ROOT"
export PYTHONHASHSEED=2026
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export STDLOC_CAMERA_LOADER_WORKERS="$CAMERA_LOADER_WORKERS"
export STDLOC_RESULTS_ROOT

mkdir -p "$CONFIG_ROOT" "$LOG_ROOT" "$RESULT_ROOT" "$STDLOC_RESULTS_ROOT"
cd "$REPO_ROOT"

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "Required artifact is missing: $1" >&2
    exit 1
  fi
}

run_logged() {
  local stage="$1"
  shift
  printf '%q ' "$@" > "$LOG_ROOT/${stage}.command.sh"
  printf '\n' >> "$LOG_ROOT/${stage}.command.sh"
  "$@" 2>&1 | tee "$LOG_ROOT/${stage}.log"
}

resolve_scene_scales() {
  # The local optimizer uses a normalized translation parameter.  Preserve the
  # historical OldHospital scale (0.05m / 0.10m) as 0.7x / 1.4x of the scene
  # normalization, instead of silently reusing those metric values everywhere.
  if [[ -z "$TASK_TRANSLATION_SCALE_M" ]]; then
    TASK_TRANSLATION_SCALE_M="$(
      "$PYTHON" scripts/compute_scene_normalization.py \
        --dataset_root "$DATA_ROOT/$SCENE" \
        --point_cloud "$MODEL_ROOT/point_cloud/iteration_30000/point_cloud.ply" \
        --images processed --target_longest_edge 640 --field_steps 30000 --shell \
        | sed -n 's/^TRANSLATION_SCALE_M=//p'
    )"
  fi
  if [[ -z "$TASK_TRANSLATION_SCALE_M" ]] || ! "$PYTHON" - "$TASK_TRANSLATION_SCALE_M" <<'PY'
import math
import sys

value = float(sys.argv[1])
if not math.isfinite(value) or value <= 0.0:
    raise SystemExit(1)
PY
  then
    echo "Could not resolve a positive Pair/LGCV scene translation scale" >&2
    exit 1
  fi
  if [[ -z "$PRIOR_GN_TRANSLATION_SCALE_M" ]]; then
    PRIOR_GN_TRANSLATION_SCALE_M="$(awk -v scale="$TASK_TRANSLATION_SCALE_M" 'BEGIN { printf "%.12g", 0.7 * scale }')"
  fi
  if [[ -z "$PRIOR_GN_MAX_TRANSLATION_M" ]]; then
    PRIOR_GN_MAX_TRANSLATION_M="$(awk -v scale="$TASK_TRANSLATION_SCALE_M" 'BEGIN { printf "%.12g", 1.4 * scale }')"
  fi
}

write_manifest() {
  "$PYTHON" - "$MANIFEST" <<PY
import json
from pathlib import Path

payload = {
    "schema_version": 1,
    "purpose": "validation_only_true_sparse_ransac_inlier_pair_lgcv_sidecar",
    "test_evaluation_forbidden": True,
    "scene": "${SCENE}",
    "representation": "${REPRESENTATION}",
    "sparse_contract": {
        "longest_edge": ${LONGEST_EDGE},
        "frontend": "ulfloc_native",
        "native_keypoints": ${NATIVE_KEYPOINTS},
        "topk": 1,
        "cosine_threshold": ${MATCH_THRESHOLD},
        "max_matches_per_landmark": ${MAX_MATCHES_PER_LANDMARK},
        "candidate_split_mode": "${SPLIT_MODE}",
        "candidate_split_seed": ${SPLIT_SEED},
        "candidate_validation_ratio": ${VALIDATION_RATIO},
    },
    "pair": {
        "seed": "same_run_sparse_ransac_inlier_p2d_p3d",
        "gt_used_for_refinement": False,
        "lgcv": True,
        "feature_grid": "native",
        "pose_solver": "${PAIR_POSE_SOLVER}",
        "expanded_anchor_radius_feature_px": 4,
        "expanded_anchor_stride_feature_px": 2,
        "max_expanded_anchors": 2048,
        "task_translation_scale_m": ${TASK_TRANSLATION_SCALE_M},
        "prior_gn_translation_scale_m": ${PRIOR_GN_TRANSLATION_SCALE_M},
        "prior_gn_max_translation_m": ${PRIOR_GN_MAX_TRANSLATION_M},
    },
    "inputs": {
        "model_root": str(Path("${MODEL_ROOT}").resolve()),
        "field_state": str(Path("${FIELD_STATE}").resolve()),
        "landmark_ids": str(Path("${LANDMARK_IDS}").resolve()),
        "landmark_meta": str(Path("${LANDMARK_META}").resolve()),
        "query_cache": str(Path("${QUERY_CACHE}").resolve()),
    },
}
Path("${MANIFEST}").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY
}

verify_field_binding() {
  "$PYTHON" - "$FIELD_STATE" "$LANDMARK_IDS" "$SPLIT_MODE" "$SPLIT_SEED" <<'PY'
import json
import pickle
import sys
from pathlib import Path

import torch

state_path, ids_path, expected_mode, expected_seed = sys.argv[1:]
state_path = Path(state_path)
ids_path = Path(ids_path)
state = torch.load(state_path, map_location="cpu")
state_indices = torch.as_tensor(state.get("landmark_indices"), dtype=torch.long)
try:
    with ids_path.open("rb") as handle:
        landmark_indices = pickle.load(handle)
except Exception:
    landmark_indices = torch.load(ids_path, map_location="cpu")
landmark_indices = torch.as_tensor(landmark_indices, dtype=torch.long)
if state_indices.ndim != 1 or not torch.equal(state_indices, landmark_indices):
    raise SystemExit(
        "Pair/LGCV field state does not use the supplied landmark IDs; "
        "refuse a mismatched sparse map."
    )
config = state.get("config", {})
if str(config.get("observation_source")) != "native":
    raise SystemExit("Pair/LGCV sidecar requires a native sparse LaFGS state")
if str(config.get("query_feature_contract")) != "native_resized_input":
    raise SystemExit("Pair/LGCV sidecar requires the native resized-input contract")
manifest_path = state_path.parent / "reproducibility_manifest.json"
if not manifest_path.is_file():
    raise SystemExit(f"Pair/LGCV field state is missing manifest: {manifest_path}")
arguments = json.loads(manifest_path.read_text()).get("arguments", {})
if str(arguments.get("split_mode")) != expected_mode:
    raise SystemExit(
        "Pair/LGCV split mismatch between field state and requested validation: "
        f"state={arguments.get('split_mode')!r} requested={expected_mode!r}"
    )
if int(arguments.get("split_seed", -1)) != int(expected_seed):
    raise SystemExit(
        "Pair/LGCV split seed mismatch between field state and requested validation: "
        f"state={arguments.get('split_seed')!r} requested={expected_seed!r}"
    )
print("Verified Pair/LGCV field-state and landmark binding")
PY
}

verify_sparse_dump_binding() {
  local sparse_root="$1"
  "$PYTHON" - "$sparse_root/results_summary.json" "$FIELD_STATE" "$SPLIT_MODE" "$SPLIT_SEED" <<'PY'
import json
import sys
from pathlib import Path

summary_path, field_state, expected_mode, expected_seed = sys.argv[1:]
payload = json.loads(Path(summary_path).read_text())
provenance = payload.get("artifact_provenance", {})
active_state = provenance.get("landmark_feature_override_path")
if not active_state or Path(active_state).resolve() != Path(field_state).resolve():
    raise SystemExit("Pair/LGCV sparse dump does not bind to the requested field state")
if payload.get("evaluation_camera_subset") != "candidate_validation":
    raise SystemExit("Pair/LGCV sparse dump is not candidate-validation only")
candidate = payload.get("evaluation_protocol", {}).get("candidate_split", {})
if candidate.get("mode") != expected_mode or int(candidate.get("seed", -1)) != int(expected_seed):
    raise SystemExit(
        "Pair/LGCV sparse dump candidate split does not match the field protocol"
    )
if not bool(candidate.get("direct_holdout")):
    raise SystemExit("Pair/LGCV sparse dump must use direct candidate holdout")
print("Verified Pair/LGCV sparse-dump binding")
PY
}

make_sparse_config() {
  local cfg="$CONFIG_ROOT/sparse_dump_validation.yaml"
  "$PYTHON" scripts/make_stdloc_eval_cfg.py \
    --base_cfg configs/stdloc_cambridge.yaml --output "$cfg" \
    --artifact_model_path "$MODEL_ROOT" \
    --detector_folder ulfloc_native_no_detector --detector_iters 0 \
    --landmark_path "$LANDMARK_IDS" --landmark_meta_path "$LANDMARK_META" \
    --landmark_feature_override_path "$FIELD_STATE" --override_landmark_features \
    --detect_num "$NATIVE_KEYPOINTS" --nms 2 \
    --sparse_query_feature_contract native_resized_input --sparse_frontend ulfloc_native \
    --reprojection_error "$EVAL_REPROJECTION_PX" --match_threshold "$MATCH_THRESHOLD" --match_topk 1 \
    --max_matches_per_landmark "$MAX_MATCHES_PER_LANDMARK" \
    --candidate_frontend_match_policy error --diagnostics --diagnostics_dump_correspondences \
    --diagnostics_grid_rows 4 --diagnostics_grid_cols 4 --diagnostics_voxel_size 1.0 \
    --diagnostics_task_translation_scale_m "$TASK_TRANSLATION_SCALE_M" \
    --diagnostics_task_rotation_scale_degrees 2.0 \
    > "$LOG_ROOT/sparse_dump_validation_config.json"
  printf '%s\n' "$cfg"
}

run_sparse_dump() {
  if [[ -f "$SPARSE_REF" ]]; then
    local existing
    existing="$(<"$SPARSE_REF")"
    if [[ -f "$existing/results.json" && -f "$existing/sparse_correspondences.jsonl" ]]; then
      verify_sparse_dump_binding "$existing"
      echo "[Pair] Reusing sparse correspondence dump: $existing"
      return
    fi
  fi
  local cfg
  cfg="$(make_sparse_config)"
  run_logged sparse_dump_validation \
    "$PYTHON" stdloc.py \
    --model_path "$MODEL_ROOT" --source_path "$DATA_ROOT/$SCENE" \
    --images processed --data_device cpu --gaussian_type 2dgs --feature_type sp \
    --resolution 1 --longest_edge "$LONGEST_EDGE" --norm_before_render --iteration 30000 \
    --cfg "$cfg" --prefix "lafgs-v2-pair-${SCENE}-${RUN_LABEL}-validation" --sparse_only \
    --evaluation_camera_subset candidate_validation --candidate_direct_validation_holdout \
    --candidate_validation_ratio "$VALIDATION_RATIO" \
    --candidate_split_mode "$SPLIT_MODE" --candidate_split_seed "$SPLIT_SEED"
  local output_path
  output_path="$(sed -n 's/^Output path: //p' "$LOG_ROOT/sparse_dump_validation.log" | tail -n 1)"
  if [[ -z "$output_path" || ! -f "$output_path/results.json" || ! -f "$output_path/sparse_correspondences.jsonl" ]]; then
    echo "Sparse validation did not produce pair seed correspondences" >&2
    exit 1
  fi
  printf '%s\n' "$output_path" > "$SPARSE_REF"
  verify_sparse_dump_binding "$output_path"
}

run_pair_dense() {
  run_sparse_dump
  local sparse_root
  sparse_root="$(<"$SPARSE_REF")"
  local output="$RUN_ROOT/pair_dense_validation"
  if [[ -f "$output/summary.json" ]]; then
    echo "[Pair] Reusing dense validation: $output"
    return
  fi
  local mode_args=()
  if [[ "$REPRESENTATION" == "lafgs_field" ]]; then
    mode_args+=(--field_state "$FIELD_STATE")
  fi
  run_logged pair_inlier_local_validation \
    "$PYTHON" scripts/eval_lafgs_dense_refinement.py \
    -s "$DATA_ROOT/$SCENE" -m "$MODEL_ROOT" --images processed --data_device cpu \
    --gaussian_type 2dgs --feature_type sp --resolution 1 --longest_edge "$LONGEST_EDGE" \
    --norm_before_render --iteration 30000 --cfg "$CONFIG_ROOT/sparse_dump_validation.yaml" \
    --input_results "$sparse_root/results.json" \
    --sparse_correspondences "$sparse_root/sparse_correspondences.jsonl" \
    --output_dir "$output" --mode "$REPRESENTATION" \
    --query_cache "$QUERY_CACHE" --dense_iterations 1 --feature_grid native \
    --matching_mode pair_inlier_local --local_radius_px 3 --local_anchor_stride 1 \
    --local_temperature 0.07 --local_correspondence_mode soft \
    --pair_seed_expansion_radius_px 4 --pair_seed_expansion_stride_px 2 --pair_seed_max_anchors 2048 \
    --max_dense_matches 2048 --dense_pose_solver "$PAIR_POSE_SOLVER" \
    --prior_gn_iterations 1 --prior_gn_damping 100 --prior_gn_translation_scale_m "$PRIOR_GN_TRANSLATION_SCALE_M" \
    --prior_gn_rotation_scale_deg 0.5 --prior_gn_robust_delta_px 0.75 \
    --prior_gn_max_translation_m "$PRIOR_GN_MAX_TRANSLATION_M" --prior_gn_max_rotation_deg 0.75 --prior_gn_min_matches 32 \
    --dense_gt_diagnostics --checkpoint_every 10 \
    "${mode_args[@]}"
  require_file "$output/summary.json"
}

require_file "$MODEL_ROOT/artifact_provenance.json"
require_file "$MODEL_ROOT/point_cloud/iteration_30000/point_cloud.ply"
require_file "$FIELD_STATE"
require_file "$LANDMARK_IDS"
require_file "$LANDMARK_META"
require_file "$QUERY_CACHE"
resolve_scene_scales
verify_field_binding
write_manifest
run_pair_dense
echo "True sparse-inlier Pair/LGCV validation complete: $RUN_ROOT"
