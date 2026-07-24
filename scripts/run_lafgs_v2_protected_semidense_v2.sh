#!/usr/bin/env bash
set -euo pipefail

# P1 validation: training-only failure-conditioned semidense supervision.
# Deployment remains one native SuperPoint top-1 pass plus one RANSAC/PnP.

if [[ $# -ne 3 ]]; then
  echo "Usage: bash $0 <OldHospital> <gpu> <protected_v2|protected_v2_lgcv>" >&2
  exit 2
fi

SCENE="$1"
GPU="$2"
VARIANT="$3"
[[ "$SCENE" == "OldHospital" ]] || {
  echo "P1 validation is locked to OldHospital" >&2
  exit 2
}
case "$GPU" in
  1|2) ;;
  *) echo "This runner may use only GPU 1 or GPU 2" >&2; exit 2 ;;
esac
case "$VARIANT" in
  protected_v2) LGCV_WEIGHT=0 ;;
  protected_v2_lgcv) LGCV_WEIGHT="${LAFGS_PSV2_LGCV_WEIGHT:-0.1}" ;;
  *) echo "Unknown variant: $VARIANT" >&2; exit 2 ;;
esac

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/root/miniconda3/envs/cybersim_agent/bin/python}"
DATA_ROOT="${CAMBRIDGE_DATA_ROOT:-/mnt/pool/sqy/Cambridge_stdloc}"
MODEL_ROOT="${LAFGS_PSV2_MODEL_ROOT:-/mnt/pool/sqy/stdloc_lafgs_v2_ulfparity_alternating_20260721/matcha_wrappers/OldHospital}"
BOOTSTRAP_ROOT="${LAFGS_PSV2_BOOTSTRAP_ROOT:-/mnt/pool/sqy/stdloc_lafgs_v2_robust_initializer_20260722/OldHospital/robustkcs_gwff32000_s0_uniform_mv4_v2_r0p01_vb4m2_tb4m2_t0p1_dcm1p0_h64_support_rgb_only_v2_splitstratified_temporal_block_seed2026_fullres_native_uncapped/bootstrap}"
CONTROL_ROOT="${LAFGS_PSV2_CONTROL_ROOT:-/mnt/pool/sqy/stdloc_lafgs_v2_distinctiveness_20260723/OldHospital/global_attractor025_retry_workers0}"
QUERY_CACHE="${LAFGS_PSV2_QUERY_CACHE:-/mnt/pool/sqy/stdloc_lafgs_v2_ulfparity_alternating_20260721/OldHospital/ulfparity_native20k_s128_k2048_v2/query_cache_native_fullres_k2048.pt}"
VISIBILITY_CACHE="${LAFGS_PSV2_VISIBILITY_CACHE:-/mnt/pool/sqy/stdloc_lafgs_v2_robust_initializer_20260722/OldHospital/robustkcs_gwff32000_s0_uniform_mv4_v2_r0p01_vb4m2_tb4m2_t0p1_dcm1p0_h64_support_rgb_only_v2_splitstratified_temporal_block_seed2026_fullres_native_uncapped/visibility_32000_native.pt}"
EXPERIMENT_ROOT="${LAFGS_PSV2_ROOT:-/mnt/pool/sqy/stdloc_lafgs_v2_protected_semidense_v2_20260724}"
STEPS="${LAFGS_PSV2_STEPS:-2500}"

BOOTSTRAP_STATE="$BOOTSTRAP_ROOT/0_lafgs_map_state.pt"
LANDMARK_IDS="$BOOTSTRAP_ROOT/sampled_idx.pkl"
LANDMARK_META="$BOOTSTRAP_ROOT/landmark_meta.pt"
CONTROL_RESULTS="$CONTROL_ROOT/stdloc_results/lafgs-v2-global-attractor025-OldHospital-residual_5000-validation-_mnt_pool_sqy_stdloc_lafgs_v2_ulfparity_alternating_20260721_matcha_wrappers_OldHospital-20260723_134414/results_summary.json"
RUN_ROOT="$EXPERIMENT_ROOT/$SCENE/${VARIANT}_steps${STEPS}_native2048"
STATE_ROOT="$RUN_ROOT/map"
CONFIG_ROOT="$RUN_ROOT/configs"
LOG_ROOT="$RUN_ROOT/logs"
RESULT_ROOT="$RUN_ROOT/results"
SUMMARY_PATH="$RUN_ROOT/validation_summary.json"

for required in \
  "$BOOTSTRAP_STATE" "$LANDMARK_IDS" "$LANDMARK_META" \
  "$QUERY_CACHE" "$VISIBILITY_CACHE" "$CONTROL_RESULTS"; do
  [[ -f "$required" ]] || {
    echo "Missing required input: $required" >&2
    exit 1
  }
done

export CUDA_VISIBLE_DEVICES="$GPU"
export CUDA_HOME=/usr/local/cuda-11.8
export PATH="/root/miniconda3/envs/cybersim_agent/bin:$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="/root/miniconda3/envs/cybersim_agent/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$REPO_ROOT"
export PYTHONHASHSEED=2026
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export STDLOC_CAMERA_LOADER_WORKERS=0
export STDLOC_RESULTS_ROOT="$RUN_ROOT/stdloc_results"

mkdir -p "$STATE_ROOT" "$CONFIG_ROOT" "$LOG_ROOT" "$RESULT_ROOT" "$STDLOC_RESULTS_ROOT"
cd "$REPO_ROOT"

SAVE_STEPS=("$STEPS")
if (( STEPS >= 500 )); then
  SAVE_STEPS+=(500)
fi
if (( STEPS >= 1000 )); then
  SAVE_STEPS+=(1000)
fi

run_logged() {
  local name="$1"
  shift
  "$@" 2>&1 | tee "$LOG_ROOT/$name.log"
}

if [[ ! -f "$STATE_ROOT/${STEPS}_lafgs_map_state.pt" ]]; then
  run_logged train \
    "$PYTHON" train_lafgs_map.py \
    --model_path "$MODEL_ROOT" --source_path "$DATA_ROOT/$SCENE" \
    --images processed --data_device cpu --gaussian_type 2dgs --feature_type sp \
    --resolution 1 --longest_edge 0 --norm_before_render --load_iteration 30000 \
    --query_feature_contract native_resized_input \
    --query_cache_path "$QUERY_CACHE" --query_cache_policy readonly \
    --visibility_cache_path "$VISIBILITY_CACHE" --visibility_mode rasterizer \
    --output_dir "$STATE_ROOT" --scaffold_mode file \
    --landmark_path "$LANDMARK_IDS" --initial_state_path "$BOOTSTRAP_STATE" \
    --initial_state_blend 1 --initial_state_alignment exact \
    --initialization_mode ulf_robust_geometry \
    --objective hard --observation_source native --native_keypoint_count 2048 \
    --max_observations 2048 --validation_observations 2048 \
    --native_sampling_mode detector_grid --native_association_radius_px 2 \
    --native_anchor_aux_weight 0 --native_outcome_mode --native_nce_weight 0 \
    --native_keep_weight 1 --native_keep_margin 0.05 \
    --native_swap_weight 1 --native_swap_margin 0.05 \
    --native_miss_weight 1 --native_miss_margin 0.05 \
    --native_reject_weight 0.05 --native_reject_threshold 0 \
    --native_global_attractor_weight 0.25 \
    --native_global_attractor_min_incoming 4 \
    --native_global_attractor_support_power 0.5 \
    --native_global_attractor_max_score 4 \
    --native_semidense_weight "${LAFGS_PSV2_WEIGHT:-0.01}" \
    --native_semidense_start_step "${LAFGS_PSV2_START_STEP:-1000}" \
    --native_semidense_interval 5 --native_semidense_max_anchors 64 \
    --native_semidense_neighbors 8 \
    --native_semidense_local_radius_px 8 \
    --native_semidense_target_sigma_px 2 \
    --native_semidense_temperature 0.07 \
    --native_semidense_pose_safe_max_delete_gain_m -1 \
    --no-native_semidense_pose_safe_teacher_pairs \
    --native_semidense_lgcv_weight "$LGCV_WEIGHT" \
    --native_semidense_lgcv_minimum_edge_px 1 \
    --native_semidense_protected_v2 \
    --native_semidense_measurement_min_reprojection_px 2 \
    --native_semidense_measurement_max_reprojection_px 8 \
    --native_semidense_surface_point_plane_m 0.03 \
    --native_semidense_surface_max_distance_m 0.15 \
    --native_semidense_surface_normal_cosine 0.95 \
    --native_semidense_projected_neighbor_radius_px 64 \
    --native_semidense_local_identity_weight "${LAFGS_PSV2_LOCAL_ID_WEIGHT:-0.25}" \
    --native_semidense_margin_preservation_weight "${LAFGS_PSV2_MARGIN_WEIGHT:-0.25}" \
    --native_semidense_gradient_audit \
    --mv_weight 0 --retrieval_weight 1 --trust_weight 0.02 --local_weight 0 \
    --dustbin_weight 0 --generic_proposal_count 0 --distill_budget 0 \
    --geometry_weight 0 --pose_weight 0 --pose_gradient_mode off \
    --feature_lr 5e-5 --weight_decay 1e-4 \
    --hypothesis_topk 32 --positive_radius_px 2 --negative_radius_px 6 \
    --validation_ratio 0.2 --split_mode stratified_temporal_block \
    --split_seed 2026 --train_seed 2026 \
    --steps "$STEPS" --save_steps "${SAVE_STEPS[@]}" --log_interval 100
fi

EVAL_STEPS=("$STEPS")
for STEP in "${EVAL_STEPS[@]}"; do
  STATE_PATH="$STATE_ROOT/${STEP}_lafgs_map_state.pt"
  RESULT_POINTER="$RESULT_ROOT/${STEP}.path"
  if [[ -f "$RESULT_POINTER" ]] && [[ -f "$(<"$RESULT_POINTER")/results_summary.json" ]]; then
    continue
  fi
  CONFIG_PATH="$CONFIG_ROOT/${STEP}.yaml"
  "$PYTHON" scripts/make_stdloc_eval_cfg.py \
    --base_cfg configs/stdloc_cambridge.yaml --output "$CONFIG_PATH" \
    --artifact_model_path "$MODEL_ROOT" \
    --detector_folder ulfloc_native_no_detector --detector_iters 0 \
    --landmark_path "$LANDMARK_IDS" --landmark_meta_path "$LANDMARK_META" \
    --landmark_feature_override_path "$STATE_PATH" --override_landmark_features \
    --detect_num 2048 --nms 2 \
    --sparse_query_feature_contract native_resized_input \
    --sparse_frontend ulfloc_native \
    --reprojection_error 12 --match_threshold 0 --match_topk 1 \
    --max_matches_per_landmark 0 --candidate_frontend_match_policy error \
    --diagnostics --diagnostics_grid_rows 4 --diagnostics_grid_cols 4 \
    --diagnostics_voxel_size 1.0 \
    --diagnostics_task_translation_scale_m 0.07160573943725686 \
    --diagnostics_task_rotation_scale_degrees 2.0 \
    > "$LOG_ROOT/eval_${STEP}_config.json"
  run_logged "eval_${STEP}" \
    "$PYTHON" stdloc.py \
    --model_path "$MODEL_ROOT" --source_path "$DATA_ROOT/$SCENE" \
    --images processed --data_device cpu --gaussian_type 2dgs --feature_type sp \
    --resolution 1 --longest_edge 0 --norm_before_render --iteration 30000 \
    --cfg "$CONFIG_PATH" \
    --prefix "lafgs-v2-${VARIANT}-${SCENE}-${STEP}" --sparse_only \
    --evaluation_camera_subset candidate_validation \
    --candidate_direct_validation_holdout --candidate_validation_ratio 0.2 \
    --candidate_split_mode stratified_temporal_block --candidate_split_seed 2026
  OUTPUT_PATH="$(sed -n 's/^Output path: //p' "$LOG_ROOT/eval_${STEP}.log" | tail -n 1)"
  [[ -n "$OUTPUT_PATH" && -f "$OUTPUT_PATH/results_summary.json" ]] || {
    echo "Missing validation output for step $STEP" >&2
    exit 1
  }
  printf '%s\n' "$OUTPUT_PATH" > "$RESULT_POINTER"
done

"$PYTHON" - "$SUMMARY_PATH" "$CONTROL_RESULTS" "$RESULT_ROOT" "$VARIANT" "$STEPS" <<'PY'
import json
import sys
from pathlib import Path

output, control_path, result_root, variant, final_step = sys.argv[1:]

def metrics(path):
    payload = json.load(open(path, "r", encoding="utf-8"))
    sparse = payload["sparse"]
    diagnostics = payload.get("sparse_diagnostics", {})
    return {
        "median_te_cm": float(sparse["median_te"]),
        "median_ae_deg": float(sparse["median_ae"]),
        "recall_5cm_5deg": float(sparse["recall_5cm_5d"]),
        "avg_candidates": diagnostics.get("sparse_diag_match_count_mean"),
        "avg_inliers": float(sparse["avg_inliers"]),
        "solver_inlier_ratio": (
            float(sparse["avg_inliers"])
            / max(float(diagnostics.get("sparse_diag_match_count_mean", 0.0)), 1.0)
        ),
        "raw_gt_precision_2px": diagnostics.get("sparse_diag_all_gt_precision_2px_mean"),
        "inlier_gt_precision_2px": diagnostics.get("sparse_diag_inlier_gt_precision_2px_mean"),
        "inlier_gt_precision_12px": diagnostics.get("sparse_diag_inlier_gt_precision_12px_mean"),
        "clean2_group_effective_ratio": diagnostics.get(
            "sparse_diag_inlier_gt_clean2_effective_ratio_mean"
        ),
        "clean2_group_ess": diagnostics.get(
            "sparse_diag_inlier_gt_clean2_group_ess_mean"
        ),
        "translation_bias_m": diagnostics.get(
            "sparse_diag_inlier_gt_pose_bias_translation_norm_m_mean"
        ),
        "ransac_hypotheses": diagnostics.get(
            "sparse_diag_ransac_actual_hypotheses_mean"
        ),
    }

result_root = Path(result_root)
payload = {
    "schema_version": 2,
    "test_evaluation_forbidden": True,
    "inference_contract": "native_sp_top1_once_then_ransac_pnp_once",
    "variant": variant,
    "control": metrics(control_path),
    "checkpoints": {},
}
for pointer in sorted(result_root.glob("*.path"), key=lambda path: int(path.stem)):
    result_dir = Path(pointer.read_text().strip())
    payload["checkpoints"][pointer.stem] = metrics(
        result_dir / "results_summary.json"
    )
Path(output).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
print(json.dumps(payload, indent=2, sort_keys=True))
PY
