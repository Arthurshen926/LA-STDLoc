#!/usr/bin/env bash
set -euo pipefail

# Controlled localization-anchor corruption on a frozen RGB Gaussian prior.
# This experiment changes only the PnP landmark coordinates. RGB geometry,
# rendering, descriptors, and the 895/182 protocol remain fixed.

if [[ $# -ne 4 ]]; then
  echo "Usage: bash $0 <rgb_2dgs|rgb_nosky|rgb_sky_dirty> <gpu> <fraction> <magnitude_m>" >&2
  exit 2
fi

VARIANT="$1"
GPU="$2"
FRACTION="$3"
MAGNITUDE_M="$4"
case "$VARIANT" in
  rgb_2dgs)
    MODEL_NAME="rgb_only_2dgs_stdloc"
    GAUSSIAN_TYPE="2dgs"
    SH_DEGREE=3
    DEFAULT_SCAFFOLD_BUDGET=48000
    DEFAULT_FINAL_BUDGET=32000
    ;;
  rgb_nosky)
    MODEL_NAME="rgb_only_3dgs_nosky"
    GAUSSIAN_TYPE="3dgs"
    SH_DEGREE=0
    DEFAULT_SCAFFOLD_BUDGET=32000
    DEFAULT_FINAL_BUDGET=24000
    ;;
  rgb_sky_dirty)
    MODEL_NAME="rgb_only_3dgs_sky_dirty"
    GAUSSIAN_TYPE="3dgs"
    SH_DEGREE=0
    DEFAULT_SCAFFOLD_BUDGET=32000
    DEFAULT_FINAL_BUDGET=24000
    ;;
  *) echo "Unknown RGB prior variant" >&2; exit 2 ;;
esac
case "$GPU" in 0|1|2) ;; *) echo "GPU must be 0, 1, or 2" >&2; exit 2 ;; esac

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/root/miniconda3/envs/cybersim_agent/bin/python}"
DATA_ROOT="${CAMBRIDGE_DATA_ROOT:-/mnt/pool/sqy/Cambridge_stdloc}"
EXPERIMENT_ROOT="${LAFGS_SANITIZATION_ROOT:-/mnt/pool/sqy/stdloc_lafgs_rgb_prior_sanitization_20260725}"
SCENE="OldHospital"
MODEL_ROOT="$EXPERIMENT_ROOT/$SCENE/$MODEL_NAME"
BASE_RUN_TAG="${LAFGS_CONTROLLED_BASE_RUN_TAG:-$VARIANT}"
RUN_ROOT="$EXPERIMENT_ROOT/$SCENE/runs/$BASE_RUN_TAG"
SCAFFOLD_BUDGET="${LAFGS_SANITIZATION_SCAFFOLD_BUDGET:-$DEFAULT_SCAFFOLD_BUDGET}"
BOOTSTRAP_DIR="$RUN_ROOT/bootstrap"
STAGE_A_DIR="$RUN_ROOT/stage_a_2500"
STAGE_A_STATE="$STAGE_A_DIR/2500_lafgs_map_state.pt"
LANDMARK_IDS="$BOOTSTRAP_DIR/sampled_idx.pkl"
LANDMARK_META="$BOOTSTRAP_DIR/landmark_meta.pt"
QUERY_CACHE="${LAFGS_QUERY_CACHE_PATH:-$RUN_ROOT/query_cache_native_fullres_k2048.pt}"
VISIBILITY_CACHE="$RUN_ROOT/visibility_${SCAFFOLD_BUDGET}_native.pt"
PRIOR_MANIFEST="$MODEL_ROOT/rgb_prior_manifest.json"
TAG_F="${FRACTION//./p}"
TAG_M="${MAGNITUDE_M//./p}"
PROTOCOL_TAG="${LAFGS_CONTROLLED_PROTOCOL_TAG:-independent_v2}"
SANITIZED_BUDGET="${LAFGS_CONTROLLED_FINAL_BUDGET:-$DEFAULT_FINAL_BUDGET}"
if ! [[ "$SANITIZED_BUDGET" =~ ^[1-9][0-9]*$ ]] || (( SANITIZED_BUDGET > SCAFFOLD_BUDGET )); then
  echo "LAFGS_CONTROLLED_FINAL_BUDGET must be in [1, $SCAFFOLD_BUDGET]" >&2
  exit 2
fi
CONTROL_ROOT="$RUN_ROOT/controlled_anchor_f${TAG_F}_m${TAG_M}_b${SANITIZED_BUDGET}_${PROTOCOL_TAG}"
CORRUPT_DIR="$CONTROL_ROOT/corrupted"
STATS_DIR="$CONTROL_ROOT/statistics"
CONFIG_ROOT="$CONTROL_ROOT/configs"
LOG_ROOT="$CONTROL_ROOT/logs"
RESULT_ROOT="$CONTROL_ROOT/results"
STDLOC_RESULTS_ROOT="$CONTROL_ROOT/stdloc_results"
CORRUPT_STATE="$CORRUPT_DIR/corrupted_lafgs_map_state.pt"
LABELS="$CORRUPT_DIR/corruption_labels.pt"
STATISTICS="$STATS_DIR/landmark_statistics_full.pt"
BASE_STATISTICS="${LAFGS_CONTROLLED_BASE_STATISTICS:-}"

export CUDA_VISIBLE_DEVICES="$GPU"
export CUDA_HOME=/usr/local/cuda-11.8
export PATH="/root/miniconda3/envs/cybersim_agent/bin:$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="/root/miniconda3/envs/cybersim_agent/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$REPO_ROOT"
export PYTHONHASHSEED=2026
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export STDLOC_CAMERA_LOADER_WORKERS=0
export STDLOC_RESULTS_ROOT

mkdir -p "$CONTROL_ROOT" "$CONFIG_ROOT" "$LOG_ROOT" "$RESULT_ROOT" "$STDLOC_RESULTS_ROOT"
cd "$REPO_ROOT"

run_logged() {
  local name="$1"
  shift
  printf '%q ' "$@" > "$LOG_ROOT/${name}.command.sh"
  printf '\n' >> "$LOG_ROOT/${name}.command.sh"
  "$@" 2>&1 | tee "$LOG_ROOT/${name}.log"
}

if [[ ! -f "$STAGE_A_STATE" ]]; then
  LAFGS_SANITIZATION_RUN_TAG="$BASE_RUN_TAG" \
    LAFGS_SANITIZATION_SCAFFOLD_BUDGET="$SCAFFOLD_BUDGET" \
    LAFGS_QUERY_CACHE_PATH="$QUERY_CACHE" \
    bash scripts/run_lafgs_v2_rgb_prior_sanitization.sh "$VARIANT" "$GPU" stage_a
fi

if [[ ! -f "$CORRUPT_STATE" ]]; then
  run_logged corrupt \
    "$PYTHON" scripts/corrupt_localization_anchors.py \
    --source_state "$STAGE_A_STATE" --output_dir "$CORRUPT_DIR" \
    --fraction "$FRACTION" --magnitude_m "$MAGNITUDE_M" --seed 2026
fi

if [[ ! -f "$STATISTICS" ]]; then
  if [[ -n "$BASE_STATISTICS" ]]; then
    run_logged statistics_rebind \
      "$PYTHON" scripts/rebind_geometry_teacher_to_state.py \
      --statistics "$BASE_STATISTICS" --state "$CORRUPT_STATE" \
      --output "$STATISTICS"
  else
    run_logged statistics \
      "$PYTHON" train_lafgs_map.py \
    --model_path "$MODEL_ROOT" --source_path "$DATA_ROOT/$SCENE" \
    --images processed --data_device cpu --gaussian_type "$GAUSSIAN_TYPE" --sh_degree "$SH_DEGREE" \
    --feature_type sp --resolution 1 --longest_edge 0 --norm_before_render \
    --load_iteration 30000 --require_rgb_prior_manifest \
    --rgb_prior_manifest_path "$PRIOR_MANIFEST" \
    --query_feature_contract native_resized_input \
    --query_cache_path "$QUERY_CACHE" --query_cache_policy reuse_or_build \
    --visibility_cache_path "$VISIBILITY_CACHE" --visibility_mode rasterizer \
    --objective hard --observation_source native --native_keypoint_count 2048 \
    --max_observations 2048 --validation_observations 2048 \
    --native_sampling_mode detector_grid --native_association_radius_px 2 \
    --native_anchor_aux_weight 0 --generic_proposal_count 0 --distill_budget 0 \
    --validation_ratio 0 --split_mode stratified_temporal_block \
    --split_seed 2026 --train_seed 2026 \
    --output_dir "$STATS_DIR" --scaffold_mode file \
    --landmark_path "$LANDMARK_IDS" \
    --initial_state_path "$CORRUPT_STATE" --initial_state_blend 1 \
    --initial_state_alignment exact --initial_state_geometry_as_base \
    --initialization_mode ulf_robust_geometry \
    --no-native_outcome_mode --retrieval_weight 0 --trust_weight 0 \
    --mv_weight 0 --local_weight 0 --dustbin_weight 0 \
    --geometry_weight 0 --pose_weight 0 --pose_gradient_mode off \
    --positive_radius_px 2 --negative_radius_px 8 \
    --save_landmark_statistics --save_independent_geometry_teacher \
    --statistics_observations 2048 \
      --steps 0 --save_steps 0
  fi
fi

for mode in loc hard_geo_loc loc_query_coverage hard_geo_loc_query_coverage loc_geo; do
  output_dir="$CONTROL_ROOT/sanitize_${mode}_${SANITIZED_BUDGET}"
  if [[ ! -f "$output_dir/sanitized_lafgs_map_state.pt" ]]; then
    run_logged "sanitize_${mode}" \
      "$PYTHON" scripts/sanitize_lafgs_landmarks.py \
      --source_state "$CORRUPT_STATE" --statistics "$STATISTICS" \
      --output_dir "$output_dir" --mode "$mode" --budget "$SANITIZED_BUDGET" \
      --outlier_labels "$LABELS"
  fi
done

eval_state() {
  local label="$1"
  local state="$2"
  local bank_dir="$3"
  local pointer="$RESULT_ROOT/${label}.path"
  if [[ -f "$pointer" ]] && [[ -f "$(<"$pointer")/results_summary.json" ]]; then
    return
  fi
  local cfg="$CONFIG_ROOT/${label}.yaml"
  "$PYTHON" scripts/make_stdloc_eval_cfg.py \
    --base_cfg configs/stdloc_cambridge.yaml --output "$cfg" \
    --artifact_model_path "$MODEL_ROOT" \
    --detector_folder ulfloc_native_no_detector --detector_iters 0 \
    --landmark_path "$bank_dir/sampled_idx.pkl" \
    --landmark_meta_path "$bank_dir/landmark_meta.pt" \
    --landmark_feature_override_path "$state" --override_landmark_features \
    --detect_num 2048 --nms 2 \
    --sparse_query_feature_contract native_resized_input \
    --sparse_frontend ulfloc_native \
    --reprojection_error 12 --match_threshold 0 --match_topk 1 \
    --max_matches_per_landmark 0 --candidate_frontend_match_policy error \
    --diagnostics --diagnostics_grid_rows 4 --diagnostics_grid_cols 4 \
    --diagnostics_voxel_size 1 \
    --diagnostics_task_translation_scale_m 0.07160573943725686 \
    --diagnostics_task_rotation_scale_degrees 2 \
    > "$LOG_ROOT/eval_${label}_config.json"
  run_logged "eval_${label}" \
    "$PYTHON" stdloc.py \
    --model_path "$MODEL_ROOT" --source_path "$DATA_ROOT/$SCENE" \
    --images processed --data_device cpu --gaussian_type "$GAUSSIAN_TYPE" \
    --sh_degree "$SH_DEGREE" --feature_type sp --resolution 1 --longest_edge 0 \
    --norm_before_render --iteration 30000 --cfg "$cfg" \
    --prefix "lafgs-v2-controlled-${VARIANT}-${TAG_F}-${TAG_M}-${label}" \
    --sparse_only --evaluation_camera_subset test
  local output_path
  output_path="$(sed -n 's/^Output path: //p' "$LOG_ROOT/eval_${label}.log" | tail -n 1)"
  [[ -n "$output_path" && -f "$output_path/results_summary.json" ]]
  printf '%s\n' "$output_path" > "$pointer"
}

eval_state corrupted "$CORRUPT_STATE" "$CORRUPT_DIR"
for mode in loc hard_geo_loc loc_query_coverage hard_geo_loc_query_coverage loc_geo; do
  eval_state "sanitize_${mode}" \
    "$CONTROL_ROOT/sanitize_${mode}_${SANITIZED_BUDGET}/sanitized_lafgs_map_state.pt" \
    "$CONTROL_ROOT/sanitize_${mode}_${SANITIZED_BUDGET}"
done

"$PYTHON" - "$CONTROL_ROOT" "$FRACTION" "$MAGNITUDE_M" "$SANITIZED_BUDGET" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
payload = {
    "schema_version": 1,
    "fraction": float(sys.argv[2]),
    "magnitude_m": float(sys.argv[3]),
    "sanitized_budget": int(sys.argv[4]),
    "rgb_map_modified": False,
    "evaluations": {},
    "sanitization": {},
}
for pointer in sorted((root / "results").glob("*.path")):
    payload["evaluations"][pointer.stem] = pointer.read_text().strip()
for report in sorted(root.glob("sanitize_*/sanitization_report.json")):
    payload["sanitization"][report.parent.name] = json.loads(report.read_text())
(root / "controlled_summary.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n"
)
print(json.dumps(payload, indent=2, sort_keys=True))
PY
