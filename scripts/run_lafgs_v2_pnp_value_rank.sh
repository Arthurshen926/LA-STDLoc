#!/usr/bin/env bash
set -euo pipefail

# Refine the strongest pure-cosine Stage-A field with a 4:1 Stage-A/exact-rank
# schedule and optional frozen functional replay. Evaluation stays native
# SuperPoint, uncapped cosine top-1, and one PnP over all 182 development images.

GPU="${1:-2}"
[[ "$GPU" =~ ^[012]$ ]] || { echo "GPU must be 0, 1, or 2" >&2; exit 2; }

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/root/miniconda3/envs/cybersim_agent/bin/python}"
DATA_ROOT="/mnt/pool/sqy/Cambridge_stdloc/OldHospital"
MODEL_ROOT="/mnt/pool/sqy/stdloc_lafgs_v2_ulfparity_alternating_20260721/matcha_wrappers/OldHospital"
BASE_ROOT="/mnt/pool/sqy/stdloc_lafgs_v2_exact_semantic_20260724/OldHospital/semantic_full_native2048_steps2500"
BOOTSTRAP="$BASE_ROOT/map/2500_lafgs_map_state.pt"
LANDMARK_IDS="$BASE_ROOT/map/sampled_idx.pkl"
LANDMARK_META="$BASE_ROOT/map/landmark_meta.pt"
QUERY_CACHE="/mnt/pool/sqy/stdloc_lafgs_v2_ulfparity_alternating_20260721/OldHospital/ulfparity_native20k_s128_k2048_v2/query_cache_native_fullres_k2048.pt"
VISIBILITY="/mnt/pool/sqy/stdloc_lafgs_v2_independent_bins_20260724/OldHospital/robustkcs_gwff32000_s0_uniform_mv2_v2_r0p01_vb8m2_tb4m2_ib1_fvb8_t0p1_dcm1p0_h64_refmean_adapt0_deployment_post_filter_v3_independent_bins_splitstratified_temporal_block_seed2026_fullres_native_uncapped/visibility_32000_native.pt"
RUN_NAME="${LAFGS_PNP_VALUE_RANK_RUN_NAME:-r2_stagea4_rank1_s2026_steps1000}"
RUN_ROOT="${LAFGS_PNP_VALUE_RANK_ROOT:-/mnt/pool/sqy/stdloc_lafgs_v2_pnp_value_rank_20260725}/OldHospital/$RUN_NAME"
MAP_ROOT="$RUN_ROOT/map"
RESULT_ROOT="$RUN_ROOT/results"
LOG_ROOT="$RUN_ROOT/logs"
CFG_ROOT="$RUN_ROOT/configs"
STDLOC_RESULTS_ROOT="$RUN_ROOT/stdloc_results"
FINAL_STEP="${LAFGS_RANK_FINAL_STEP:-1000}"
read -r -a EVAL_STEPS <<< "${LAFGS_RANK_EVAL_STEPS:-0 500 $FINAL_STEP}"
FUNCTIONAL_REPLAY_WEIGHT="${LAFGS_FUNCTIONAL_REPLAY_WEIGHT:-0}"
FUNCTIONAL_REPLAY_CACHE="${LAFGS_FUNCTIONAL_REPLAY_CACHE:-$RUN_ROOT/functional_replay_bank.pt}"
REPLAY_ARGS=()
if [[ "$FUNCTIONAL_REPLAY_WEIGHT" != "0" ]]; then
  REPLAY_ARGS=(
    --functional_replay_weight "$FUNCTIONAL_REPLAY_WEIGHT"
    --functional_replay_cache_path "$FUNCTIONAL_REPLAY_CACHE"
    --functional_replay_rows_per_query 64
    --functional_replay_core_rows_per_query 16
    --functional_replay_batch_size 256
    --functional_replay_topm 64
    --functional_replay_temperature 0.05
    --functional_replay_margin_slack 0.005
    --functional_replay_distribution_weight 1
    --functional_replay_pnp_core_weight "${LAFGS_FUNCTIONAL_REPLAY_CORE_WEIGHT:-0}"
    --functional_replay_ransac_max_iterations 20000
    --functional_replay_ransac_min_iterations 1000
  )
  if [[ "${LAFGS_FUNCTIONAL_REPLAY_BUILD_PNP_CORE:-0}" == "1" ]]; then
    REPLAY_ARGS+=(--functional_replay_build_pnp_core)
  fi
  if [[ "${LAFGS_FUNCTIONAL_REPLAY_GRADIENT_PROJECTION:-0}" == "1" ]]; then
    REPLAY_ARGS+=(--functional_replay_gradient_projection)
  fi
fi

export CUDA_VISIBLE_DEVICES="$GPU"
export CUDA_HOME=/usr/local/cuda-11.8
export PATH="/root/miniconda3/envs/cybersim_agent/bin:$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="/root/miniconda3/envs/cybersim_agent/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$REPO_ROOT"
export PYTHONHASHSEED=2026
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export STDLOC_CAMERA_LOADER_WORKERS=0
export STDLOC_RESULTS_ROOT

mkdir -p "$MAP_ROOT" "$RESULT_ROOT" "$LOG_ROOT" "$CFG_ROOT" "$STDLOC_RESULTS_ROOT"
cd "$REPO_ROOT"

run_logged() {
  local name="$1"
  shift
  printf '%q ' "$@" > "$LOG_ROOT/$name.command.sh"
  printf '\n' >> "$LOG_ROOT/$name.command.sh"
  "$@" 2>&1 | tee "$LOG_ROOT/$name.log"
}

if [[ ! -f "$MAP_ROOT/${FINAL_STEP}_lafgs_map_state.pt" ]]; then
  run_logged train "$PYTHON" train_lafgs_map.py \
    --model_path "$MODEL_ROOT" --source_path "$DATA_ROOT" \
    --images processed --data_device cpu --gaussian_type 2dgs --feature_type sp \
    --resolution 1 --longest_edge 0 --norm_before_render --load_iteration 30000 \
    --query_feature_contract native_resized_input \
    --query_cache_path "$QUERY_CACHE" --query_cache_policy readonly \
    --visibility_cache_path "$VISIBILITY" --visibility_mode rasterizer \
    --output_dir "$MAP_ROOT" --scaffold_mode file \
    --landmark_path "$LANDMARK_IDS" --initial_state_path "$BOOTSTRAP" \
    --initial_state_blend 1 --initial_state_alignment exact \
    --initialization_mode ulf_robust_geometry \
    --objective hard --observation_source native --native_keypoint_count 2048 \
    --max_observations 2048 --validation_observations 2048 \
    --native_sampling_mode detector_grid --native_association_radius_px 2 \
    --native_anchor_aux_weight 0 \
    --native_outcome_mode --native_nce_weight 0 \
    --native_keep_weight 1 --native_keep_margin 0.05 \
    --native_swap_weight 1 --native_swap_margin 0.05 \
    --native_miss_weight 1 --native_miss_margin 0.05 \
    --native_reject_weight 0 --native_reject_threshold 0 \
    --native_global_attractor_weight 0.25 \
    --native_global_attractor_min_incoming 4 \
    --native_global_attractor_support_power 0.5 \
    --native_global_attractor_max_score 4 \
    --native_rank_budget_mode --native_rank_stage_a_steps 4 \
    --native_rank_steps 1 --native_rank_temperature 0.03 \
    --native_rank_margin_at1 0.02 --native_rank_margin_at4 0.02 \
    --native_rank_margin_at8 0.02 --native_rank_margin_at32 0.02 \
    --native_rank_top1_weight 0.25 --native_rank_keep_weight 1 \
    --native_rank_band_rank1 0.25 --native_rank_band_rank2_4 0.25 \
    --native_rank_band_rank5_32 0.30 --native_rank_band_rank33_plus 0.20 \
    --native_rank_landmark_balance \
    --native_rank_reference_clean_weight 0 \
    --mv_weight 0 --retrieval_weight 1 --local_weight 0 \
    --dustbin_weight 0 --generic_proposal_count 0 --distill_budget 0 \
    --geometry_weight 0 --pose_weight 0 --pose_gradient_mode off \
    --feature_lr 0.0001 --trust_weight 0.02 --weight_decay 0.0001 \
    --max_residual_norm 0.05 \
    --hypothesis_topk 32 --positive_radius_px 2 --negative_radius_px 8 \
    --validation_ratio 0 --split_mode stratified_temporal_block \
    --split_seed 2026 --train_seed 2026 --steps "$FINAL_STEP" \
    --save_steps 500 "$FINAL_STEP" --log_interval 50 \
    "${REPLAY_ARGS[@]}"
fi

for STEP in "${EVAL_STEPS[@]}"; do
  STATE="$MAP_ROOT/${STEP}_lafgs_map_state.pt"
  CFG="$CFG_ROOT/${STEP}.yaml"
  POINTER="$RESULT_ROOT/${STEP}_test.path"
  "$PYTHON" scripts/make_stdloc_eval_cfg.py \
    --base_cfg configs/stdloc_cambridge.yaml --output "$CFG" \
    --artifact_model_path "$MODEL_ROOT" \
    --detector_folder ulfloc_native_no_detector --detector_iters 0 \
    --landmark_path "$LANDMARK_IDS" --landmark_meta_path "$LANDMARK_META" \
    --landmark_feature_override_path "$STATE" --override_landmark_features \
    --detect_num 2048 --nms 2 \
    --sparse_query_feature_contract native_resized_input \
    --sparse_frontend ulfloc_native \
    --reprojection_error 12 --match_threshold 0 --match_topk 1 \
    --max_matches_per_landmark 0 --candidate_frontend_match_policy error \
    --diagnostics --diagnostics_grid_rows 4 --diagnostics_grid_cols 4 \
    --diagnostics_voxel_size 1.0 \
    --diagnostics_task_translation_scale_m 0.07160573943725686 \
    --diagnostics_task_rotation_scale_degrees 2 \
    --diagnostics_dump_discrete_oracle --diagnostics_oracle_topk 32 \
    > "$LOG_ROOT/eval_${STEP}_config.json"
  if [[ ! -f "$POINTER" ]] || [[ ! -f "$(<"$POINTER")/results_summary.json" ]]; then
    run_logged "eval_${STEP}" "$PYTHON" stdloc.py \
      --model_path "$MODEL_ROOT" --source_path "$DATA_ROOT" \
      --images processed --data_device cpu --gaussian_type 2dgs --feature_type sp \
      --resolution 1 --longest_edge 0 --norm_before_render --iteration 30000 \
      --cfg "$CFG" --prefix "lafgs-v2-pnp-value-${RUN_NAME}-${STEP}" \
      --sparse_only --evaluation_camera_subset test
    OUTPUT="$(sed -n 's/^Output path: //p' "$LOG_ROOT/eval_${STEP}.log" | tail -n 1)"
    [[ -f "$OUTPUT/results_summary.json" ]] || { echo "Missing eval output" >&2; exit 1; }
    printf '%s\n' "$OUTPUT" > "$POINTER"
  fi
done

FINAL_RESULT="$(<"$RESULT_ROOT/${FINAL_STEP}_test.path")"
if [[ -f "$RESULT_ROOT/0_test.path" ]]; then
  BASE_RESULT="$(<"$RESULT_ROOT/0_test.path")"
  REFERENCE_DUMP_DIR="$BASE_RESULT/discrete_oracle_dump"
else
  REFERENCE_DUMP_DIR="${LAFGS_RANK_REFERENCE_DUMP_DIR:-}"
fi
if [[ -n "$REFERENCE_DUMP_DIR" ]]; then
  "$PYTHON" scripts/summarize_rank_oracle_dump.py \
    --dump_dir "$FINAL_RESULT/discrete_oracle_dump" \
    --reference_dump_dir "$REFERENCE_DUMP_DIR" \
    --output "$RUN_ROOT/rank_summary.json"
else
  echo "Skipping rank transition summary: no step-0 or explicit reference dump."
fi
