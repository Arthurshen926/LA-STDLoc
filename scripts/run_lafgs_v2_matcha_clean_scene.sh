#!/usr/bin/env bash
set -euo pipefail

# Causal clean-chain LaFGS V2 experiment on an external RGB-only MAtCha 2DGS.
#
# The script intentionally keeps map reconstruction and detector fitting apart:
#   1. build a wide geometry pool and select a fixed matchability-aware bank;
#   2. reconstruct descriptors while geometry remains frozen;
#   3. select only on a direct held-out training-camera split with a frozen
#      bootstrap detector;
#   4. fit a new detector only after the map state has been selected;
#   5. evaluate test cameras once, after all choices are fixed.

if [[ $# -lt 3 || $# -gt 4 ]]; then
  echo "Usage: $0 <scene> <gpu:1|2> <prepare|bootstrap|baseline_detector|descriptor|validate|select|final_refit|final_detector|test|all> [mean|medoid]" >&2
  exit 2
fi

SCENE="$1"
GPU="$2"
MODE="$3"
MVINIT_MODE="${4:-medoid}"

case "$SCENE" in
  GreatCourt|KingsCollege|OldHospital|ShopFacade|StMarysChurch) ;;
  *) echo "Unsupported Cambridge scene: $SCENE" >&2; exit 2 ;;
esac
case "$GPU" in
  1|2) ;;
  *) echo "This experiment is restricted to GPU 1 or GPU 2; got $GPU" >&2; exit 2 ;;
esac
case "$MODE" in
  prepare|bootstrap|baseline_detector|descriptor|validate|select|final_refit|final_detector|test|all) ;;
  *) echo "Unsupported mode: $MODE" >&2; exit 2 ;;
esac
case "$MVINIT_MODE" in
  mean|medoid) ;;
  *) echo "MVInit mode must be mean or medoid; got $MVINIT_MODE" >&2; exit 2 ;;
esac

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/root/miniconda3/envs/cybersim_agent/bin/python}"
DATA_ROOT="${CAMBRIDGE_DATA_ROOT:-/mnt/pool/sqy/Cambridge_stdloc}"
MATCHA_ROOT="${CAMBRIDGE_MATCHA_2DGS_ROOT:-/root/MAtCha/output_cambridge_full_retained_v2}"
EXPERIMENT_ROOT="${LAFGS_V2_CLEAN_MATCHA_ROOT:-/mnt/pool/sqy/stdloc_lafgs_v2_clean_matcha_20260719}"
WRAPPER_ROOT="$EXPERIMENT_ROOT/matcha_wrappers"
SCENE_ROOT="$EXPERIMENT_ROOT/$SCENE"
MODEL_ROOT="$WRAPPER_ROOT/$SCENE"
DESCRIPTOR_STEPS="${LAFGS_V2_DESCRIPTOR_STEPS:-3000}"
DETECTOR_STEPS="${LAFGS_V2_DETECTOR_STEPS:-2000}"
POOL_BUDGET="${LAFGS_V2_POOL_BUDGET:-98304}"
LANDMARK_BUDGET="${LAFGS_V2_LANDMARK_BUDGET:-16384}"
STATS_OBSERVATIONS="${LAFGS_V2_STATS_OBSERVATIONS:-512}"
MVINIT_OBSERVATIONS="${LAFGS_V2_MVINIT_OBSERVATIONS:-512}"
DISTILL_RANK_POOL_MULTIPLIER="${LAFGS_V2_DISTILL_RANK_POOL_MULTIPLIER:-2}"
VALIDATION_RATIO="${LAFGS_V2_VALIDATION_RATIO:-0.2}"
SPLIT_MODE="${LAFGS_V2_SPLIT_MODE:-temporal_block}"
SPLIT_SEED="${LAFGS_V2_SPLIT_SEED:-2026}"
SELECTION_MODE="${LAFGS_V2_SELECTION_MODE:-performance}"
case "$SELECTION_MODE" in
  safety|performance) ;;
  *) echo "Selection mode must be safety or performance; got $SELECTION_MODE" >&2; exit 2 ;;
esac
EXPERIMENT_TAG="${MVINIT_MODE}_p${POOL_BUDGET}_s${STATS_OBSERVATIONS}_rp${DISTILL_RANK_POOL_MULTIPLIER}"
BOOTSTRAP_DIR="$SCENE_ROOT/bootstrap_${EXPERIMENT_TAG}"
DESCRIPTOR_DIR="$SCENE_ROOT/descriptor_${EXPERIMENT_TAG}_${DESCRIPTOR_STEPS}"
QUERY_CACHE="$SCENE_ROOT/query_cache_v6.pt"
NORMALIZATION_JSON="$SCENE_ROOT/scene_normalization.json"
CONFIG_ROOT="$SCENE_ROOT/configs/$EXPERIMENT_TAG"
LOG_ROOT="$SCENE_ROOT/logs/$EXPERIMENT_TAG"
RESULT_ROOT="$SCENE_ROOT/results/$EXPERIMENT_TAG"
STDLOC_RESULTS_ROOT="$SCENE_ROOT/stdloc_results/$EXPERIMENT_TAG"
BASELINE_DETECTOR_FOLDER="detector_bootstrap_${EXPERIMENT_TAG}_val${VALIDATION_RATIO}_${DETECTOR_STEPS}_canonical-v1"
BOOTSTRAP_STATE="$BOOTSTRAP_DIR/distilled_lafgs_map_state.pt"
BOOTSTRAP_IDS="$BOOTSTRAP_DIR/distilled_sampled_idx.pkl"
BOOTSTRAP_META="$BOOTSTRAP_DIR/landmark_meta.pt"
SELECTION_REPORT="$RESULT_ROOT/selection_${SELECTION_MODE}.json"
FINAL_REFIT_DIR="$SCENE_ROOT/final_refit_${EXPERIMENT_TAG}"
FULL_BOOTSTRAP_DIR="$FINAL_REFIT_DIR/bootstrap"
FULL_BOOTSTRAP_STATE="$FULL_BOOTSTRAP_DIR/distilled_lafgs_map_state.pt"
FULL_BOOTSTRAP_IDS="$FULL_BOOTSTRAP_DIR/distilled_sampled_idx.pkl"
FULL_BOOTSTRAP_META="$FULL_BOOTSTRAP_DIR/landmark_meta.pt"
FULL_RESULT_ROOT="$FINAL_REFIT_DIR/results"
FINAL_REFIT_MANIFEST="$FINAL_REFIT_DIR/final_refit_manifest.json"
FULL_BOOTSTRAP_DETECTOR_FOLDER="detector_bootstrap_${EXPERIMENT_TAG}_full_${DETECTOR_STEPS}"
FULL_FINAL_DETECTOR_FOLDER="detector_final_${EXPERIMENT_TAG}_full_${DETECTOR_STEPS}"
INITIALIZATION_SELECTION="${LAFGS_V2_INITIALIZATION_SELECTION:-}"

case "$SCENE" in
  GreatCourt) MATCHA_RUN="GreatCourt_n20_long_masked_retrain_retry" ;;
  KingsCollege) MATCHA_RUN="KingsCollege_n20_long_masked_retrain" ;;
  OldHospital) MATCHA_RUN="OldHospital_n20_long_masked_retrain_retry" ;;
  ShopFacade) MATCHA_RUN="ShopFacade_n20_long_masked_retrain" ;;
  StMarysChurch) MATCHA_RUN="StMarysChurch_n20_long_masked_retrain" ;;
esac
SOURCE_PLY="$MATCHA_ROOT/$MATCHA_RUN/free_gaussians/point_cloud/iteration_30000/point_cloud.ply"

export CUDA_VISIBLE_DEVICES="$GPU"
export CUDA_HOME=/usr/local/cuda-11.8
export PATH="/root/miniconda3/envs/cybersim_agent/bin:$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="/root/miniconda3/envs/cybersim_agent/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$REPO_ROOT"
export PYTHONHASHSEED=2026
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export STDLOC_RESULTS_ROOT

mkdir -p "$SCENE_ROOT" "$CONFIG_ROOT" "$LOG_ROOT" "$RESULT_ROOT" "$STDLOC_RESULTS_ROOT"
cd "$REPO_ROOT"

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "Required artifact is missing: $1" >&2
    exit 1
  fi
}

require_exact_bank_size() {
  local artifact="$1"
  local count
  count="$("$PYTHON" -c 'import pickle, sys; value = pickle.load(open(sys.argv[1], "rb")); print(int(value.numel() if hasattr(value, "numel") else len(value)))' "$artifact")"
  if [[ "$count" != "$LANDMARK_BUDGET" ]]; then
    echo "Expected exactly $LANDMARK_BUDGET distilled landmarks, found $count in $artifact" >&2
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

prepare_scene() {
  require_file "$SOURCE_PLY"
  if [[ ! -f "$MODEL_ROOT/artifact_provenance.json" ]]; then
    run_logged prepare \
      "$PYTHON" scripts/audit_cambridge_matcha_2dgs_protocol.py \
      --runs_root "$MATCHA_ROOT" \
      --data_root "$DATA_ROOT" \
      --scenes "$SCENE" \
      --output_json "$SCENE_ROOT/matcha_protocol.json" \
      --output_markdown "$SCENE_ROOT/matcha_protocol.md" \
      --prepare_wrapper_root "$WRAPPER_ROOT"
  fi
  require_file "$MODEL_ROOT/artifact_provenance.json"
  require_file "$MODEL_ROOT/point_cloud/iteration_30000/point_cloud.ply"
  if [[ ! -f "$NORMALIZATION_JSON" ]]; then
    run_logged normalization \
      "$PYTHON" scripts/compute_scene_normalization.py \
      --dataset_root "$DATA_ROOT/$SCENE" \
      --point_cloud "$SOURCE_PLY" \
      --images processed \
      --target_longest_edge 640 \
      --field_steps "$DESCRIPTOR_STEPS" \
      --output_json "$NORMALIZATION_JSON"
  fi
}

load_normalization() {
  prepare_scene
  # Import only the runtime normalization fields.  The normalization helper
  # also reports training recommendations such as DETECTOR_STEPS; importing
  # those wholesale would overwrite this experiment's fixed schedule.
  local normalization_shell
  normalization_shell="$("$PYTHON" scripts/compute_scene_normalization.py \
    --dataset_root "$DATA_ROOT/$SCENE" \
    --point_cloud "$SOURCE_PLY" \
    --images processed \
    --target_longest_edge 640 \
    --field_steps "$DESCRIPTOR_STEPS" \
    --output_json "$NORMALIZATION_JSON" \
    --shell)"
  local key value
  while IFS='=' read -r key value; do
    case "$key" in
      EVAL_DETECT_NUM|NMS_RADIUS_PX|RESIDUAL_CLIP_PX|PNP_VOXEL_SIZE_M|TRANSLATION_SCALE_M)
        printf -v "$key" '%s' "$value"
        ;;
    esac
  done <<< "$normalization_shell"
}

bootstrap_to_dir() {
  local stage="$1"
  local output_dir="$2"
  local validation_ratio="$3"
  local requested_cache_policy="$4"
  local state="$output_dir/distilled_lafgs_map_state.pt"
  local ids="$output_dir/distilled_sampled_idx.pkl"
  local meta="$output_dir/landmark_meta.pt"
  prepare_scene
  if [[ -f "$state" && -f "$ids" && -f "$meta" ]]; then
    require_exact_bank_size "$ids"
    echo "[LaFGS V2 clean] Reusing bootstrap: $output_dir"
    return
  fi
  local cache_policy="$requested_cache_policy"
  if [[ "$cache_policy" == "auto" ]]; then
    cache_policy="reuse_or_build"
    if [[ -f "$QUERY_CACHE" ]]; then
      cache_policy="readonly"
    fi
  fi
  run_logged "$stage" \
    "$PYTHON" train_lafgs_map.py \
    --model_path "$MODEL_ROOT" \
    --source_path "$DATA_ROOT/$SCENE" \
    --images processed --data_device cpu \
    --gaussian_type 2dgs --feature_type sp \
    --resolution 1 --longest_edge 640 --norm_before_render \
    --load_iteration 30000 --output_dir "$output_dir" \
    --scaffold_mode pure_geometry \
    --generated_landmark_path "$output_dir/wide_geometry_pool.pkl" \
    --regenerate_scaffold \
    --scaffold_budget "$POOL_BUDGET" \
    --scaffold_min_opacity 0.05 --scaffold_min_visible_views 0 \
    --scaffold_normal_bins 6 --scaffold_seed 2026 \
    --query_cache_path "$QUERY_CACHE" --query_cache_policy "$cache_policy" \
    --visibility_mode depth --objective hard \
    --steps 0 --save_steps 0 \
    --mvinit_mode "$MVINIT_MODE" --mvinit_max_observations "$MVINIT_OBSERVATIONS" \
    --mv_weight 0 --retrieval_weight 0 --trust_weight 0 --local_weight 0 \
    --proposal_jitter_std 0 --proposal_jitter_max 0 \
    --generic_proposal_count 512 --generic_proposal_weight 0 \
    --generic_proposal_nms_radius 2 --generic_proposal_positive_radius 2 \
    --unmatched_rejection_weight 0 --dustbin_weight 0 \
    --geometry_weight 0 --pose_weight 0 --pose_gradient_mode off \
    --distill_budget "$LANDMARK_BUDGET" \
    --distill_require_exact_budget \
    --distill_rank_pool_multiplier "$DISTILL_RANK_POOL_MULTIPLIER" \
    --statistics_observations "$STATS_OBSERVATIONS" \
    --statistics_hypothesis_topk 32 \
    --distill_min_observations 2 \
    --distill_matchability_threshold 0.5 --distill_false_top1_max 0.5 \
    --distill_proposal_weight 1 \
    --distill_matchability_preserve_ratio 0.30 \
    --distill_utility_preserve_ratio 0.35 \
    --distill_grid_size 8 --distill_max_per_grid 512 \
    --distill_depth_bins 8 --distill_max_per_depth_bin 4096 \
    --validation_ratio "$validation_ratio" --split_mode "$SPLIT_MODE" \
    --split_seed "$SPLIT_SEED" --train_seed 2026 \
    --max_observations 512 --validation_observations 512 \
    --log_interval 50
  require_file "$state"
  require_file "$ids"
  require_file "$meta"
  require_exact_bank_size "$ids"
}

bootstrap() {
  bootstrap_to_dir bootstrap "$BOOTSTRAP_DIR" "$VALIDATION_RATIO" auto
}

full_bootstrap() {
  # The cache was built under the exact same frozen RGB/frontend signature
  # during validation.  Requiring it here prevents final refit from silently
  # changing the image-side protocol.
  bootstrap_to_dir final_bootstrap "$FULL_BOOTSTRAP_DIR" 0 readonly
}

train_detector_from_state() {
  local stage="$1"
  local folder="$2"
  local state="$3"
  local use_validation_holdout="$4"
  local landmark_ids="${5:-$BOOTSTRAP_IDS}"
  local target="$MODEL_ROOT/$folder/${DETECTOR_STEPS}_detector.pth"
  load_normalization
  require_file "$state"
  require_file "$landmark_ids"
  require_exact_bank_size "$landmark_ids"
  if [[ ! -f "$target" ]]; then
    local split_args=()
    if [[ "$use_validation_holdout" == "true" ]]; then
      split_args=(
        --candidate_teacher_validation_ratio "$VALIDATION_RATIO"
        --candidate_teacher_split_mode "$SPLIT_MODE"
        --candidate_teacher_split_seed "$SPLIT_SEED"
      )
    fi
    run_logged "$stage" \
      "$PYTHON" train_detector.py \
      --model_path "$MODEL_ROOT" --iteration 30000 \
      --iterations "$DETECTOR_STEPS" --save_iterations "$DETECTOR_STEPS" \
      --detector_folder "$folder" --landmark_num "$LANDMARK_BUDGET" \
      --precomputed_landmark_path "$landmark_ids" \
      --sparse_candidate_teacher --candidate_teacher_detector_only \
      --candidate_teacher_state_init_path "$state" \
      --candidate_teacher_detector_lr 1e-4 \
      --candidate_teacher_detect_num "$EVAL_DETECT_NUM" \
      --candidate_teacher_nms_radius "$NMS_RADIUS_PX" \
      --candidate_teacher_match_topk 1 \
      --candidate_teacher_hard_negatives 0 \
      --candidate_teacher_pair_weight 0 \
      --candidate_teacher_hard_negative_weight 0 \
      --candidate_teacher_assignment_weight 0 \
      --candidate_teacher_detector_match_weight 1 \
      --candidate_teacher_detector_offset_weight 0 \
      --candidate_teacher_geometry_weight 0 \
      --candidate_teacher_coverage_weight 0 \
      --candidate_teacher_base_detector_weight 0.1 \
      --candidate_teacher_feature_anchor_weight 0 \
      --candidate_teacher_dustbin_weight 0 \
      --candidate_teacher_pair_scorer_weight 0 \
      --candidate_teacher_pair_scorer_assignment_weight 0 \
      --candidate_teacher_pair_measurement_inlier_weight 0 \
      --candidate_teacher_pair_measurement_nll_weight 0 \
      --candidate_teacher_pair_measurement_bias_weight 0 \
      --candidate_teacher_pair_measurement_covariance_weight 0 \
      "${split_args[@]}"
  else
    echo "[LaFGS V2 clean] Reusing detector: $folder"
  fi
  require_file "$target"
  require_file "$MODEL_ROOT/$folder/sampled_idx.pkl"
  if [[ "$use_validation_holdout" == "true" ]]; then
    run_logged "${stage}_holdout_alignment" \
      "$PYTHON" scripts/verify_lafgs_direct_holdout.py \
      --map-state "$state" \
      --detector-summary "$MODEL_ROOT/$folder/candidate_teacher_training_summary.json" \
      --source-path "$DATA_ROOT/$SCENE" \
      --output "$MODEL_ROOT/$folder/holdout_alignment.json"
    require_file "$MODEL_ROOT/$folder/holdout_alignment.json"
  fi
}

baseline_detector() {
  bootstrap
  train_detector_from_state baseline_detector "$BASELINE_DETECTOR_FOLDER" "$BOOTSTRAP_STATE" true "$BOOTSTRAP_IDS"
}

descriptor() {
  bootstrap
  if [[ -f "$DESCRIPTOR_DIR/${DESCRIPTOR_STEPS}_lafgs_map_state.pt" ]]; then
    echo "[LaFGS V2 clean] Reusing descriptor phase: $DESCRIPTOR_DIR"
    return
  fi
  require_file "$QUERY_CACHE"
  run_logged descriptor \
    "$PYTHON" train_lafgs_map.py \
    --model_path "$MODEL_ROOT" \
    --source_path "$DATA_ROOT/$SCENE" \
    --images processed --data_device cpu \
    --gaussian_type 2dgs --feature_type sp \
    --resolution 1 --longest_edge 640 --norm_before_render \
    --load_iteration 30000 --output_dir "$DESCRIPTOR_DIR" \
    --scaffold_mode file --landmark_path "$BOOTSTRAP_IDS" \
    --initial_state_path "$BOOTSTRAP_STATE" --initial_state_blend 1 \
    --initial_state_alignment exact \
    --query_cache_path "$QUERY_CACHE" --query_cache_policy readonly \
    --visibility_mode depth --objective hard \
    --steps "$DESCRIPTOR_STEPS" \
    --save_steps 500 1000 1500 2000 2500 "$DESCRIPTOR_STEPS" \
    --feature_lr 5e-5 --weight_decay 1e-4 \
    --mvinit_mode "$MVINIT_MODE" --mvinit_max_observations "$MVINIT_OBSERVATIONS" \
    --mv_weight 0.5 --retrieval_weight 0.5 --trust_weight 0.1 --local_weight 0.05 \
    --hypothesis_topk 32 --positive_radius_px 2 --negative_radius_px 6 \
    --retrieval_margin 0.05 --missed_positive_weight 1 --missed_positive_margin 0.05 \
    --proposal_jitter_std 0.75 --proposal_jitter_max 2 \
    --generic_proposal_count 512 --generic_proposal_weight 0.25 \
    --generic_proposal_nms_radius 2 --generic_proposal_positive_radius 2 \
    --unmatched_rejection_weight 0.1 --unmatched_max_similarity 0.5 \
    --dustbin_weight 0 --geometry_weight 0 --pose_weight 0 --pose_gradient_mode off \
    --validation_ratio "$VALIDATION_RATIO" --split_mode "$SPLIT_MODE" \
    --split_seed "$SPLIT_SEED" --train_seed 2026 \
    --max_observations 512 --validation_observations 512 --log_interval 100
  require_file "$DESCRIPTOR_DIR/${DESCRIPTOR_STEPS}_lafgs_map_state.pt"
}

selected_tag() {
  require_file "$SELECTION_REPORT"
  "$PYTHON" - "$SELECTION_REPORT" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], "r", encoding="utf-8"))
if payload.get("selection_protocol", {}).get("test_metrics_used") is not False:
    raise SystemExit("selection report is not validation-only")
print(payload["selected_tag"])
PY
}

selected_descriptor_step() {
  local tag
  tag="$(selected_tag)"
  if [[ "$tag" == "control_strong" ]]; then
    printf '0\n'
    return
  fi
  if [[ "$tag" =~ ^descriptor_([0-9]+)$ ]]; then
    printf '%s\n' "${BASH_REMATCH[1]}"
    return
  fi
  echo "Unsupported selected validation tag: $tag" >&2
  return 1
}

verify_initialization_selection() {
  if [[ -z "$INITIALIZATION_SELECTION" ]]; then
    echo "Final refit requires LAFGS_V2_INITIALIZATION_SELECTION from scripts/select_lafgs_clean_initialization.py" >&2
    return 1
  fi
  require_file "$INITIALIZATION_SELECTION"
  "$PYTHON" - "$INITIALIZATION_SELECTION" "$MVINIT_MODE" "$SELECTION_REPORT" <<'PY'
import json
import sys
from pathlib import Path

selection_path, expected_mode, expected_report = map(Path, sys.argv[1:])
payload = json.loads(selection_path.read_text())
protocol = payload.get("selection_protocol", {})
if protocol.get("test_metrics_used") is not False:
    raise SystemExit("initialization selection is not validation-only")
if payload.get("selected_initialization") != str(expected_mode):
    raise SystemExit(
        "attempted final refit does not match selected initialization: "
        f"selected={payload.get('selected_initialization')!r} requested={expected_mode!r}"
    )
expected_report = expected_report.resolve()
selected_reports = {
    Path(candidate["selection_report"]).resolve()
    for candidate in payload.get("candidates", [])
}
if expected_report not in selected_reports:
    raise SystemExit("per-initialization selection report is absent from global selection")
print(
    "Validated global initialization selection: "
    f"{selection_path.resolve()} -> {expected_mode}"
)
PY
}

full_descriptor_dir() {
  local step="$1"
  printf '%s/descriptor_%s\n' "$FINAL_REFIT_DIR" "$step"
}

full_descriptor_state_path() {
  local step="$1"
  if [[ "$step" == "0" ]]; then
    printf '%s\n' "$FULL_BOOTSTRAP_STATE"
  else
    local output_dir
    output_dir="$(full_descriptor_dir "$step")"
    printf '%s/%s_lafgs_map_state.pt\n' "$output_dir" "$step"
  fi
}

full_descriptor() {
  local step
  step="$(selected_descriptor_step)"
  if [[ "$step" == "0" ]]; then
    return
  fi
  local output_dir
  local target
  output_dir="$(full_descriptor_dir "$step")"
  target="$(full_descriptor_state_path "$step")"
  if [[ -f "$target" ]]; then
    echo "[LaFGS V2 clean] Reusing full-data descriptor refit: $output_dir"
    return
  fi
  full_bootstrap
  require_file "$QUERY_CACHE"
  run_logged "final_descriptor_${step}" \
    "$PYTHON" train_lafgs_map.py \
    --model_path "$MODEL_ROOT" \
    --source_path "$DATA_ROOT/$SCENE" \
    --images processed --data_device cpu \
    --gaussian_type 2dgs --feature_type sp \
    --resolution 1 --longest_edge 640 --norm_before_render \
    --load_iteration 30000 --output_dir "$output_dir" \
    --scaffold_mode file --landmark_path "$FULL_BOOTSTRAP_IDS" \
    --initial_state_path "$FULL_BOOTSTRAP_STATE" --initial_state_blend 1 \
    --initial_state_alignment exact \
    --query_cache_path "$QUERY_CACHE" --query_cache_policy readonly \
    --visibility_mode depth --objective hard \
    --steps "$step" --save_steps "$step" \
    --feature_lr 5e-5 --weight_decay 1e-4 \
    --mvinit_mode "$MVINIT_MODE" --mvinit_max_observations "$MVINIT_OBSERVATIONS" \
    --mv_weight 0.5 --retrieval_weight 0.5 --trust_weight 0.1 --local_weight 0.05 \
    --hypothesis_topk 32 --positive_radius_px 2 --negative_radius_px 6 \
    --retrieval_margin 0.05 --missed_positive_weight 1 --missed_positive_margin 0.05 \
    --proposal_jitter_std 0.75 --proposal_jitter_max 2 \
    --generic_proposal_count 512 --generic_proposal_weight 0.25 \
    --generic_proposal_nms_radius 2 --generic_proposal_positive_radius 2 \
    --unmatched_rejection_weight 0.1 --unmatched_max_similarity 0.5 \
    --dustbin_weight 0 --geometry_weight 0 --pose_weight 0 --pose_gradient_mode off \
    --validation_ratio 0 --split_mode "$SPLIT_MODE" \
    --split_seed "$SPLIT_SEED" --train_seed 2026 \
    --max_observations 512 --validation_observations 512 --log_interval 100
  require_file "$target"
}

write_final_refit_manifest() {
  local state="$1"
  local step="$2"
  "$PYTHON" - \
    "$FINAL_REFIT_MANIFEST" "$SELECTION_REPORT" "$FULL_BOOTSTRAP_STATE" \
    "$FULL_BOOTSTRAP_IDS" "$FULL_BOOTSTRAP_META" "$state" "$step" \
    "$MODEL_ROOT" "$SOURCE_PLY" "$QUERY_CACHE" "$INITIALIZATION_SELECTION" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

(
    output,
    selection_report,
    bootstrap_state,
    landmark_ids,
    landmark_meta,
    final_state,
    selected_step,
    model_root,
    source_ply,
    query_cache,
    initialization_selection,
) = map(Path, sys.argv[1:])
payload = json.loads(selection_report.read_text())
protocol = payload.get("selection_protocol", {})
if protocol.get("test_metrics_used") is not False:
    raise SystemExit("final refit requires a validation-only selection report")
selected_tag = payload["selected_tag"]
if selected_tag == "control_strong" and int(str(selected_step)) != 0:
    raise SystemExit("control selection must not run descriptor refit")
if selected_tag != "control_strong" and selected_tag != f"descriptor_{selected_step}":
    raise SystemExit("selected checkpoint does not match final refit schedule")
initialization_payload = json.loads(initialization_selection.read_text())
initialization_protocol = initialization_payload.get("selection_protocol", {})
if initialization_protocol.get("test_metrics_used") is not False:
    raise SystemExit("initialization selection must be validation-only")
sha256 = hashlib.sha256(selection_report.read_bytes()).hexdigest()
record = {
    "schema_version": 1,
    "selection_report": str(selection_report.resolve()),
    "selection_report_sha256": sha256,
    "selection_was_validation_only": True,
    "initialization_selection": str(initialization_selection.resolve()),
    "selected_initialization": initialization_payload.get("selected_initialization"),
    "selected_validation_tag": selected_tag,
    "selected_validation_step": int(str(selected_step)),
    "full_training_validation_ratio": 0.0,
    "full_training_uses_all_train_cameras": True,
    "full_bootstrap_state": str(bootstrap_state.resolve()),
    "full_landmark_ids": str(landmark_ids.resolve()),
    "full_landmark_meta": str(landmark_meta.resolve()),
    "full_refit_state": str(final_state.resolve()),
    "frozen_2dgs_model_path": str(model_root.resolve()),
    "source_matcha_ply": str(source_ply.resolve()),
    "query_cache": str(query_cache.resolve()),
}
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
print(json.dumps(record, indent=2, sort_keys=True))
PY
}

final_refit() {
  select_checkpoint
  verify_initialization_selection
  full_bootstrap
  full_descriptor
  local step
  local state
  step="$(selected_descriptor_step)"
  state="$(full_descriptor_state_path "$step")"
  require_file "$state"
  require_file "$FULL_BOOTSTRAP_IDS"
  require_file "$FULL_BOOTSTRAP_META"
  require_exact_bank_size "$FULL_BOOTSTRAP_IDS"
  write_final_refit_manifest "$state" "$step"
}

make_eval_config() {
  local label="$1"
  local detector_folder="$2"
  local state="$3"
  local landmark_ids="${4:-$BOOTSTRAP_IDS}"
  local landmark_meta="${5:-$BOOTSTRAP_META}"
  local cfg="$CONFIG_ROOT/${label}.yaml"
  load_normalization
  require_file "$landmark_ids"
  require_file "$landmark_meta"
  require_exact_bank_size "$landmark_ids"
  "$PYTHON" scripts/make_stdloc_eval_cfg.py \
    --base_cfg configs/stdloc_cambridge.yaml --output "$cfg" \
    --artifact_model_path "$MODEL_ROOT" \
    --detector_folder "$detector_folder" --detector_iters "$DETECTOR_STEPS" \
    --landmark_path "$landmark_ids" --landmark_meta_path "$landmark_meta" \
    --landmark_feature_override_path "$state" --override_landmark_features \
    --detect_num "$EVAL_DETECT_NUM" --nms "$NMS_RADIUS_PX" \
    --reprojection_error "$RESIDUAL_CLIP_PX" \
    --match_threshold 0 --match_topk 1 --max_matches_per_landmark 2 \
    --candidate_frontend_match_policy error --diagnostics \
    --diagnostics_grid_rows 4 --diagnostics_grid_cols 4 \
    --diagnostics_voxel_size "$PNP_VOXEL_SIZE_M" \
    --diagnostics_task_translation_scale_m "$TRANSLATION_SCALE_M" \
    --diagnostics_task_rotation_scale_degrees 2.0 \
    > "$LOG_ROOT/${label}_config.json"
  printf '%s\n' "$cfg"
}

run_eval() {
  local label="$1"
  local subset="$2"
  local detector_folder="$3"
  local state="$4"
  local landmark_ids="${5:-$BOOTSTRAP_IDS}"
  local landmark_meta="${6:-$BOOTSTRAP_META}"
  local result_root="${7:-$RESULT_ROOT}"
  local result_ref="$result_root/${label}_${subset}.results_path"
  if [[ -f "$result_ref" ]] && [[ -f "$(<"$result_ref")/results_summary.json" ]]; then
    echo "[LaFGS V2 clean] Reusing evaluation: $label/$subset"
    return
  fi
  local cfg
  mkdir -p "$result_root"
  cfg="$(make_eval_config "${label}_${subset}" "$detector_folder" "$state" "$landmark_ids" "$landmark_meta")"
  local log="$LOG_ROOT/${label}_${subset}.log"
  local eval_args=(
    "$PYTHON" stdloc.py
    --model_path "$MODEL_ROOT" --source_path "$DATA_ROOT/$SCENE"
    --images processed --data_device cpu
    --gaussian_type 2dgs --feature_type sp
    --resolution 1 --longest_edge 640 --norm_before_render
    --iteration 30000 --cfg "$cfg"
    --prefix "lafgs-v2-clean-${SCENE}-${MVINIT_MODE}-${label}-${subset}"
    --sparse_only
  )
  if [[ "$subset" == "validation" ]]; then
    eval_args+=(
      --evaluation_camera_subset candidate_validation
      --candidate_direct_validation_holdout
      --candidate_validation_ratio "$VALIDATION_RATIO"
      --candidate_split_mode "$SPLIT_MODE"
      --candidate_split_seed "$SPLIT_SEED"
    )
  fi
  printf '%q ' "${eval_args[@]}" > "$LOG_ROOT/${label}_${subset}.command.sh"
  printf '\n' >> "$LOG_ROOT/${label}_${subset}.command.sh"
  "${eval_args[@]}" 2>&1 | tee "$log"
  local output_path
  output_path="$(sed -n 's/^Output path: //p' "$log" | tail -n 1)"
  if [[ -z "$output_path" ]] || [[ ! -f "$output_path/results_summary.json" ]]; then
    echo "Evaluation did not expose a valid results_summary.json in $log" >&2
    exit 1
  fi
  printf '%s\n' "$output_path" > "$result_ref"
}

validate() {
  baseline_detector
  descriptor
  run_eval control validation "$BASELINE_DETECTOR_FOLDER" "$BOOTSTRAP_STATE"
  local checkpoint
  for checkpoint in 500 1000 1500 2000 2500 "$DESCRIPTOR_STEPS"; do
    local state="$DESCRIPTOR_DIR/${checkpoint}_lafgs_map_state.pt"
    if [[ -f "$state" ]]; then
      run_eval "descriptor_${checkpoint}" validation "$BASELINE_DETECTOR_FOLDER" "$state"
    fi
  done
}

select_checkpoint() {
  validate
  if [[ -f "$SELECTION_REPORT" ]]; then
    echo "[LaFGS V2 clean] Reusing selection: $SELECTION_REPORT"
    return
  fi
  local command=(
    "$PYTHON" scripts/select_lafgs_map_checkpoint.py
    --control_results "$(<"$RESULT_ROOT/control_validation.results_path")/results_summary.json"
    --control_state "$BOOTSTRAP_STATE"
    --selection_mode "$SELECTION_MODE"
    --min_te_gain_cm 0.02 --metric_tolerance 1e-9
    --mean_te_weight 0.05 --max_recall_2m_drop 0.01 --max_recall_5cm_drop 0.01
    --output "$SELECTION_REPORT"
  )
  local checkpoint
  for checkpoint in 500 1000 1500 2000 2500 "$DESCRIPTOR_STEPS"; do
    local state="$DESCRIPTOR_DIR/${checkpoint}_lafgs_map_state.pt"
    local result_ref="$RESULT_ROOT/descriptor_${checkpoint}_validation.results_path"
    if [[ -f "$state" && -f "$result_ref" ]]; then
      command+=(--candidate "descriptor_${checkpoint}" "$(<"$result_ref")/results_summary.json" "$state")
    fi
  done
  run_logged select "${command[@]}"
  require_file "$SELECTION_REPORT"
}

final_detector() {
  final_refit
  local step
  local state
  step="$(selected_descriptor_step)"
  state="$(full_descriptor_state_path "$step")"
  if [[ -z "$state" ]]; then
    echo "Could not resolve selected state from $SELECTION_REPORT" >&2
    exit 1
  fi
  # Report a full-data bootstrap control beside the selected map.  Neither is
  # used for model selection; both are evaluated only once on the test split.
  train_detector_from_state final_bootstrap_detector \
    "$FULL_BOOTSTRAP_DETECTOR_FOLDER" "$FULL_BOOTSTRAP_STATE" false "$FULL_BOOTSTRAP_IDS"
  if [[ "$state" != "$FULL_BOOTSTRAP_STATE" ]]; then
    train_detector_from_state final_detector \
      "$FULL_FINAL_DETECTOR_FOLDER" "$state" false "$FULL_BOOTSTRAP_IDS"
  fi
}

test_selected() {
  final_detector
  local step
  local state
  local selected_detector
  step="$(selected_descriptor_step)"
  state="$(full_descriptor_state_path "$step")"
  selected_detector="$FULL_FINAL_DETECTOR_FOLDER"
  if [[ "$state" == "$FULL_BOOTSTRAP_STATE" ]]; then
    selected_detector="$FULL_BOOTSTRAP_DETECTOR_FOLDER"
  fi
  run_eval bootstrap_full test "$FULL_BOOTSTRAP_DETECTOR_FOLDER" "$FULL_BOOTSTRAP_STATE" \
    "$FULL_BOOTSTRAP_IDS" "$FULL_BOOTSTRAP_META" "$FULL_RESULT_ROOT"
  if [[ "$state" == "$FULL_BOOTSTRAP_STATE" ]]; then
    # The validation gate selected the bootstrap control.  Do not rerun an
    # identical model on the test set; preserve an explicit alias instead.
    printf '%s\n' "$(<"$FULL_RESULT_ROOT/bootstrap_full_test.results_path")" \
      > "$FULL_RESULT_ROOT/selected_full_test.results_path"
  else
    run_eval selected_full test "$selected_detector" "$state" \
      "$FULL_BOOTSTRAP_IDS" "$FULL_BOOTSTRAP_META" "$FULL_RESULT_ROOT"
  fi
}

case "$MODE" in
  prepare) prepare_scene ;;
  bootstrap) bootstrap ;;
  baseline_detector) baseline_detector ;;
  descriptor) descriptor ;;
  validate) validate ;;
  select) select_checkpoint ;;
  final_refit) final_refit ;;
  final_detector) final_detector ;;
  test) test_selected ;;
  all)
    prepare_scene
    bootstrap
    baseline_detector
    descriptor
    validate
    select_checkpoint
    final_refit
    final_detector
    test_selected
    ;;
esac
