#!/usr/bin/env bash
set -euo pipefail

# Validation-only alternating landmark pruning:
# fixed 32K field -> train-only protected selection -> fixed 24K field ->
# short native refresh. Deployment remains one sparse top-1 pass plus PnP.

if [[ $# -ne 3 ]]; then
  echo "Usage: bash $0 <OldHospital> <gpu:1|2> <distill|all>" >&2
  exit 2
fi
SCENE="$1"
GPU="$2"
MODE="$3"
[[ "$SCENE" == "OldHospital" ]] || { echo "OldHospital only in this pilot" >&2; exit 2; }
[[ "$GPU" == "1" || "$GPU" == "2" ]] || { echo "GPU must be 1 or 2" >&2; exit 2; }
[[ "$MODE" == "distill" || "$MODE" == "all" ]] || { echo "Mode must be distill or all" >&2; exit 2; }

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/root/miniconda3/envs/cybersim_agent/bin/python}"
DATA_ROOT="${CAMBRIDGE_DATA_ROOT:-/mnt/pool/sqy/Cambridge_stdloc}"
MODEL_ROOT="${LAFGS_PROTECTED_MODEL_ROOT:-/mnt/pool/sqy/stdloc_lafgs_v2_ulfparity_alternating_20260721/matcha_wrappers/OldHospital}"
SOURCE_ROOT="${LAFGS_PROTECTED_SOURCE_ROOT:-/mnt/pool/sqy/stdloc_lafgs_v2_selflocal_semidense_20260724/OldHospital/semidense_neighborhood_w001_steps5000_native2048}"
SOURCE_STEP="${LAFGS_PROTECTED_SOURCE_STEP:-5000}"
SOURCE_STATE="$SOURCE_ROOT/map/${SOURCE_STEP}_lafgs_map_state.pt"
SOURCE_IDS="$SOURCE_ROOT/map/sampled_idx.pkl"
SOURCE_META="$SOURCE_ROOT/map/landmark_meta.pt"
SOURCE_RESULT_POINTER="$SOURCE_ROOT/results/${SOURCE_STEP}.path"
QUERY_CACHE="${LAFGS_PROTECTED_QUERY_CACHE:-/mnt/pool/sqy/stdloc_lafgs_v2_ulfparity_alternating_20260721/OldHospital/ulfparity_native20k_s128_k2048_v2/query_cache_native_fullres_k2048.pt}"
SOURCE_VISIBILITY="${LAFGS_PROTECTED_SOURCE_VISIBILITY:-/mnt/pool/sqy/stdloc_lafgs_v2_robust_initializer_20260722/OldHospital/robustkcs_gwff32000_s0_uniform_mv4_v2_r0p01_vb4m2_tb4m2_t0p1_dcm1p0_h64_support_rgb_only_v2_splitstratified_temporal_block_seed2026_fullres_native_uncapped/visibility_32000_native.pt}"
FINAL_BUDGET="${LAFGS_PROTECTED_FINAL_BUDGET:-24000}"
REFRESH_STEPS="${LAFGS_PROTECTED_REFRESH_STEPS:-1000}"
ROOT="${LAFGS_PROTECTED_ROOT:-/mnt/pool/sqy/stdloc_lafgs_v2_protected_prune_20260724/OldHospital/source${SOURCE_STEP}_to${FINAL_BUDGET}}"
DISTILL_DIR="$ROOT/distill"
REFRESH_DIR="$ROOT/refresh"
CONFIG_DIR="$ROOT/configs"
LOG_DIR="$ROOT/logs"
RESULT_DIR="$ROOT/results"
DISTILL_STATE="$DISTILL_DIR/distilled_lafgs_map_state.pt"
DISTILL_IDS="$DISTILL_DIR/distilled_sampled_idx.pkl"
DISTILL_META="$DISTILL_DIR/landmark_meta.pt"
REFRESH_VISIBILITY="$REFRESH_DIR/visibility_${FINAL_BUDGET}_native.pt"

for path in "$SOURCE_STATE" "$SOURCE_IDS" "$SOURCE_META" "$SOURCE_RESULT_POINTER" "$QUERY_CACHE" "$SOURCE_VISIBILITY"; do
  [[ -f "$path" ]] || { echo "Missing required input: $path" >&2; exit 1; }
done

export CUDA_VISIBLE_DEVICES="$GPU"
export CUDA_HOME=/usr/local/cuda-11.8
export PATH="/root/miniconda3/envs/cybersim_agent/bin:$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="/root/miniconda3/envs/cybersim_agent/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$REPO_ROOT"
export PYTHONHASHSEED=2026
export STDLOC_CAMERA_LOADER_WORKERS=0
export STDLOC_RESULTS_ROOT="$ROOT/stdloc_results"
mkdir -p "$DISTILL_DIR" "$REFRESH_DIR" "$CONFIG_DIR" "$LOG_DIR" "$RESULT_DIR" "$STDLOC_RESULTS_ROOT"
cd "$REPO_ROOT"

run_logged() {
  local label="$1"
  shift
  printf '%q ' "$@" > "$LOG_DIR/${label}.command.sh"
  printf '\n' >> "$LOG_DIR/${label}.command.sh"
  "$@" 2>&1 | tee "$LOG_DIR/${label}.log"
}

verify_source() {
  "$PYTHON" - "$SOURCE_STATE" "$SOURCE_IDS" "$SOURCE_RESULT_POINTER" <<'PY'
import json, pickle, sys, torch
from pathlib import Path
state_path, ids_path, result_pointer = sys.argv[1:]
state = torch.load(state_path, map_location="cpu")
with open(ids_path, "rb") as handle:
    ids = torch.as_tensor(pickle.load(handle), dtype=torch.long).reshape(-1)
state_ids = torch.as_tensor(state["landmark_indices"], dtype=torch.long).reshape(-1)
cfg = state["config"]
errors = []
if ids.numel() != 32000 or not torch.equal(ids, state_ids):
    errors.append("source is not the exact 32K bank")
for key, expected in {
    "query_feature_contract": "native_resized_input",
    "observation_source": "native",
    "split_mode": "stratified_temporal_block",
}.items():
    if cfg.get(key) != expected:
        errors.append(f"{key}={cfg.get(key)!r}")
if int(cfg.get("native_sparse_keypoint_count", -1)) != 2048:
    errors.append("source did not train on all 2048 native proposals")
summary = Path(Path(result_pointer).read_text().strip()) / "results_summary.json"
if not summary.is_file():
    errors.append("source candidate-validation summary is missing")
else:
    protocol = json.loads(summary.read_text()).get("evaluation_protocol", {})
    candidate = protocol.get("candidate_split", {})
    if protocol.get("evaluation_camera_subset") != "candidate_validation":
        errors.append("source evaluation is not candidate-validation")
    if candidate.get("mode") != "stratified_temporal_block" or candidate.get("seed") != 2026:
        errors.append("source validation split differs")
    if int(protocol.get("longest_edge", -1)) != 0:
        errors.append("source evaluation is not full resolution")
if errors:
    raise SystemExit("Invalid protected-pruning source: " + "; ".join(errors))
print("Verified immutable full-resolution 32K source and candidate-validation protocol")
PY
}

base_args() {
  printf '%s\0' \
    "$PYTHON" train_lafgs_map.py \
    --model_path "$MODEL_ROOT" --source_path "$DATA_ROOT/$SCENE" \
    --images processed --data_device cpu --gaussian_type 2dgs --feature_type sp \
    --resolution 1 --longest_edge 0 --norm_before_render --load_iteration 30000 \
    --query_feature_contract native_resized_input \
    --query_cache_path "$QUERY_CACHE" --query_cache_policy readonly \
    --visibility_mode rasterizer --objective hard \
    --native_keypoint_count 2048 --max_observations 2048 --validation_observations 2048 \
    --native_association_radius_px 2 --native_sampling_mode detector_grid \
    --generic_proposal_count 0 --generic_proposal_weight 0 \
    --validation_ratio 0.2 --split_mode stratified_temporal_block --split_seed 2026 \
    --train_seed 2026 --tangent_bound_m 0.005 --normal_bound_m 0.002
}

append_base_args() {
  local -n output="$1"
  local item
  while IFS= read -r -d '' item; do output+=("$item"); done < <(base_args)
}

make_eval_config() {
  local label="$1" ids="$2" meta="$3" state="$4"
  local cfg="$CONFIG_DIR/${label}.yaml"
  "$PYTHON" scripts/make_stdloc_eval_cfg.py \
    --base_cfg configs/stdloc_cambridge.yaml --output "$cfg" \
    --artifact_model_path "$MODEL_ROOT" \
    --detector_folder ulfloc_native_no_detector --detector_iters 0 \
    --landmark_path "$ids" --landmark_meta_path "$meta" \
    --landmark_feature_override_path "$state" --override_landmark_features \
    --detect_num 2048 --nms 2 --sparse_query_feature_contract native_resized_input \
    --sparse_frontend ulfloc_native --reprojection_error 12 \
    --match_threshold 0 --match_topk 1 --max_matches_per_landmark 0 \
    --candidate_frontend_match_policy error --diagnostics \
    --diagnostics_grid_rows 4 --diagnostics_grid_cols 4 \
    --diagnostics_voxel_size 1.0 \
    --diagnostics_task_translation_scale_m 0.07160573943725686 \
    --diagnostics_task_rotation_scale_degrees 2.0 >/dev/null
  printf '%s\n' "$cfg"
}

eval_state() {
  local label="$1" ids="$2" meta="$3" state="$4"
  local pointer="$RESULT_DIR/${label}.path"
  if [[ -f "$pointer" && -f "$(<"$pointer")/results_summary.json" ]]; then return; fi
  local cfg
  cfg="$(make_eval_config "$label" "$ids" "$meta" "$state")"
  run_logged "eval_${label}" "$PYTHON" stdloc.py \
    --model_path "$MODEL_ROOT" --source_path "$DATA_ROOT/$SCENE" \
    --images processed --data_device cpu --gaussian_type 2dgs --feature_type sp \
    --resolution 1 --longest_edge 0 --norm_before_render --iteration 30000 \
    --cfg "$cfg" --prefix "lafgs-v2-protected-${label}" --sparse_only \
    --evaluation_camera_subset candidate_validation --candidate_direct_validation_holdout \
    --candidate_validation_ratio 0.2 --candidate_split_mode stratified_temporal_block \
    --candidate_split_seed 2026
  local output
  output="$(sed -n 's/^Output path: //p' "$LOG_DIR/eval_${label}.log" | tail -n 1)"
  [[ -f "$output/results_summary.json" ]] || { echo "Missing evaluation for $label" >&2; exit 1; }
  printf '%s\n' "$output" > "$pointer"
}

distill() {
  verify_source
  if [[ ! -f "$DISTILL_STATE" ]]; then
    local command=()
    append_base_args command
    command+=(
      --output_dir "$DISTILL_DIR" --scaffold_mode file --landmark_path "$SOURCE_IDS"
      --initial_state_path "$SOURCE_STATE" --initial_state_blend 1 --initial_state_alignment exact
      --initialization_mode ulf_robust_geometry --observation_source native
      --native_anchor_aux_weight 0
      --no-native_outcome_mode --steps 0 --save_steps 0
      --mv_weight 0 --retrieval_weight 0 --trust_weight 0 --local_weight 0
      --dustbin_weight 0 --geometry_weight 0 --pose_weight 0 --pose_gradient_mode off
      --visibility_cache_path "$SOURCE_VISIBILITY"
      --distill_budget "$FINAL_BUDGET" --distill_require_exact_budget --distill_allow_coverage_fill
      --statistics_observations 2048 --statistics_hypothesis_topk 32
      --distill_min_observations 2 --distill_matchability_threshold 0.5
      --distill_false_top1_max 0.5 --distill_proposal_weight 0
      --distill_global_attractor_weight 0.25 --distill_rank_pool_multiplier 1.5
      --distill_quality_reservoir_multiplier 1.25
      --distill_quality_reservoir_score wilson_lower
      --distill_hard_matchability_core_ratio 0
      --distill_protected_core_ratio 0.25 --distill_protected_min_correct 3
      --distill_protected_matchability 0.75
      --distill_protected_identity_switch_max 0.25
      --distill_matchability_preserve_ratio 0.25 --distill_utility_preserve_ratio 0.25
      --distill_grid_size 8 --distill_max_per_grid 512
      --distill_depth_bins 8 --distill_max_per_depth_bin 4096
    )
    run_logged distill "${command[@]}"
  fi
  eval_state distill "$DISTILL_IDS" "$DISTILL_META" "$DISTILL_STATE"
}

refresh() {
  distill
  local state="$REFRESH_DIR/${REFRESH_STEPS}_lafgs_map_state.pt"
  if [[ ! -f "$state" ]]; then
    local command=()
    append_base_args command
    command+=(
      --output_dir "$REFRESH_DIR" --scaffold_mode file --landmark_path "$DISTILL_IDS"
      --initial_state_path "$DISTILL_STATE" --initial_state_blend 1 --initial_state_alignment exact
      --initialization_mode ulf_robust_geometry --visibility_cache_path "$REFRESH_VISIBILITY"
      --observation_source native --native_anchor_aux_weight 0 --native_outcome_mode
      --native_nce_weight 0 --native_keep_weight 1 --native_keep_margin 0.05
      --native_swap_weight 1 --native_swap_margin 0.05
      --native_miss_weight 1 --native_miss_margin 0.05
      --native_reject_weight 0.05 --native_reject_threshold 0
      --native_global_attractor_weight 0.25
      --mv_weight 0 --retrieval_weight 1 --trust_weight 0.02 --local_weight 0
      --dustbin_weight 0 --geometry_weight 0 --pose_weight 0 --pose_gradient_mode off
      --steps "$REFRESH_STEPS" --save_steps 500 "$REFRESH_STEPS"
      --feature_lr 2.5e-5 --weight_decay 1e-4 --hypothesis_topk 32
      --positive_radius_px 2 --negative_radius_px 6 --log_interval 100
    )
    run_logged refresh "${command[@]}"
  fi
  eval_state refresh "$DISTILL_IDS" "$DISTILL_META" "$state"
}

write_summary() {
  "$PYTHON" - "$ROOT/summary.json" "$SOURCE_RESULT_POINTER" \
    "$RESULT_DIR/distill.path" "$RESULT_DIR/refresh.path" <<'PY'
import json, sys
from pathlib import Path
output, source_pointer, distill_pointer, refresh_pointer = sys.argv[1:]
def load_pointer(path):
    path = Path(path)
    if not path.is_file(): return None
    return json.loads((Path(path.read_text().strip()) / "results_summary.json").read_text())
def metrics(payload):
    if payload is None: return None
    sparse, diag = payload["sparse"], payload.get("sparse_diagnostics", {})
    return {
        "median_te_cm": float(sparse["median_te"]),
        "mean_te_cm": (
            float(sparse["mean_te"]) if sparse.get("mean_te") is not None else None
        ),
        "recall_5cm_5deg": float(sparse["recall_5cm_5d"]),
        "raw_gt_precision_2px": float(diag["sparse_diag_all_gt_precision_2px_mean"]),
        "inlier_gt_precision_2px": float(diag["sparse_diag_inlier_gt_precision_2px_mean"]),
        "pose_info_translation_logdet": float(diag["sparse_diag_inlier_pose_info_translation_logdet_mean"]),
        "ransac_hypotheses_median": float(diag["sparse_diag_ransac_actual_hypotheses_median"]),
        "runtime_total_ms_median": float(diag["sparse_diag_runtime_total_ms_median"]),
    }
source = metrics(load_pointer(source_pointer))
stages = {"source": source, "distill": metrics(load_pointer(distill_pointer)), "refresh": metrics(load_pointer(refresh_pointer))}
def safe(candidate):
    if candidate is None: return False
    return (
        candidate["median_te_cm"] <= source["median_te_cm"] + 0.1
        and candidate["raw_gt_precision_2px"] >= source["raw_gt_precision_2px"]
        and candidate["inlier_gt_precision_2px"] >= source["inlier_gt_precision_2px"]
        and candidate["recall_5cm_5deg"] >= source["recall_5cm_5deg"] - 0.01
    )
accepted = [name for name in ("distill", "refresh") if safe(stages[name])]
selected = min(accepted, key=lambda name: stages[name]["median_te_cm"]) if accepted else "source"
payload = {
    "schema_version": 1,
    "test_evaluation_forbidden": True,
    "inference_contract": "native_sp_top1_once_then_ransac_pnp_once",
    "selection_gate": "median<=source+0.1cm, raw/inlier_P2>=source, recall5>=source-0.01",
    "stages": stages,
    "accepted_pruned_stages": accepted,
    "selected_stage": selected,
}
Path(output).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
print(json.dumps(payload, indent=2, sort_keys=True))
PY
}

case "$MODE" in
  distill) distill ;;
  all) refresh; write_summary ;;
esac
