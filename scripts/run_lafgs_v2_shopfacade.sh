#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-test_strong}"
case "$MODE" in
  descriptor|test_strong|test_best|verify|all|legacy_c0|legacy_c1|legacy_eval_c0|legacy_eval_c1) ;;
  *)
    echo "Usage: $0 <descriptor|test_strong|test_best|verify|all|legacy_c0|legacy_c1|legacy_eval_c0|legacy_eval_c1>" >&2
    exit 2
    ;;
esac

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/root/miniconda3/envs/cybersim_agent/bin/python}"
GPU="${LAFGS_V2_GPU:-2}"
DATA_ROOT="${CAMBRIDGE_DATA_ROOT:-/mnt/pool/sqy/Cambridge_stdloc}"
MODEL_ROOT="${LAFGS_V2_MODEL_ROOT:-/mnt/pool/sqy/stdloc_lafgs_v2_20260717/ShopFacade}"
EXPERIMENT_ROOT="${LAFGS_V2_EXPERIMENT_ROOT:-/mnt/pool/sqy/stdloc_lafgs_v2_phased}"
ACTIVE_ROOT="$MODEL_ROOT/activefield_p1b_exactset_fim_1500"
LANDMARK_IDS="$ACTIVE_ROOT/sampled_idx.pkl"
LANDMARK_META="$ACTIVE_ROOT/landmark_meta.pt"
STRONG_STATE="$ACTIVE_ROOT/750_candidate_teacher_state.pt"
STRONG_DETECTOR_FOLDER="strongmap_detectoronly_2000_exact_control"
STRONG_DETECTOR_STEPS=2000
BEST_STATE="$MODEL_ROOT/interpolation/p1b_to_exactreplay750_strictval_w130.pt"
BEST_DETECTOR_FOLDER="activefield_p1b_exactset_fim_1500"
BEST_DETECTOR_STEPS=750
QUERY_CACHE="$MODEL_ROOT/lafgs_map_frontendexact_query_cache_v6.pt"
PHASE1_DIR="$EXPERIMENT_ROOT/ShopFacade/phase1_descriptor"
CONFIG_ROOT="$EXPERIMENT_ROOT/ShopFacade/configs"
LOG_ROOT="$EXPERIMENT_ROOT/ShopFacade/logs"

export CUDA_VISIBLE_DEVICES="$GPU"
export CUDA_HOME=/usr/local/cuda-11.8
export PATH="/root/miniconda3/envs/cybersim_agent/bin:$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="/root/miniconda3/envs/cybersim_agent/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$REPO_ROOT"
export PYTHONHASHSEED=0

mkdir -p "$PHASE1_DIR" "$CONFIG_ROOT" "$LOG_ROOT"
cd "$REPO_ROOT"

check_map_inputs() {
  test -f "$MODEL_ROOT/point_cloud/iteration_30000/point_cloud.ply"
  test -f "$LANDMARK_IDS"
  test -f "$LANDMARK_META"
}

check_descriptor_inputs() {
  check_map_inputs
  test -f "$STRONG_STATE"
  test -f "$QUERY_CACHE"
}

run_descriptor() {
  check_descriptor_inputs
  if [[ -f "$PHASE1_DIR/1000_lafgs_map_state.pt" ]]; then
    echo "[LaFGS V2] Reusing completed descriptor phase: $PHASE1_DIR"
    return
  fi
  "$PYTHON" train_lafgs_map.py \
    --model_path "$MODEL_ROOT" \
    --source_path "$DATA_ROOT/ShopFacade" \
    --images processed --data_device cpu \
    --gaussian_type 2dgs --feature_type sp \
    --resolution 1 --longest_edge 640 --norm_before_render \
    --load_iteration 30000 --output_dir "$PHASE1_DIR" \
    --scaffold_mode file --landmark_path "$LANDMARK_IDS" \
    --initial_state_path "$STRONG_STATE" --initial_state_blend 1 \
    --initial_state_alignment exact \
    --query_cache_path "$QUERY_CACHE" --query_cache_policy readonly \
    --visibility_mode depth --objective hard \
    --steps 1000 --save_steps 250 500 750 1000 \
    --feature_lr 5e-5 --weight_decay 1e-4 \
    --mvinit_mode medoid --mv_weight 0.5 \
    --retrieval_weight 0.5 --trust_weight 0.1 --local_weight 0.05 \
    --hypothesis_topk 32 --positive_radius_px 2 --negative_radius_px 6 \
    --retrieval_margin 0.05 \
    --missed_positive_weight 1 --missed_positive_margin 0.05 \
    --proposal_jitter_std 0.75 --proposal_jitter_max 2 \
    --generic_proposal_count 512 --generic_proposal_weight 0.25 \
    --generic_proposal_nms_radius 2 --generic_proposal_positive_radius 2 \
    --unmatched_rejection_weight 0.1 --unmatched_max_similarity 0.5 \
    --dustbin_weight 0 --geometry_weight 0 \
    --pose_weight 0 --pose_gradient_mode off \
    --validation_ratio 0.2 --split_mode temporal_block \
    --split_seed 2026 --train_seed 2026 \
    --max_observations 512 --validation_observations 512 \
    --log_interval 100 \
    2>&1 | tee "$LOG_ROOT/phase1_descriptor.log"
  echo "[LaFGS V2] Evaluate every saved state with the fixed strong detector on the held-out split."
  echo "[LaFGS V2] Use scripts/select_lafgs_map_checkpoint.py before any detector retraining or later phase."
}

make_eval_config() {
  local label="$1"
  local state="$2"
  local detector_folder="$3"
  local detector_steps="$4"
  local cfg="$CONFIG_ROOT/${label}.yaml"
  "$PYTHON" scripts/make_stdloc_eval_cfg.py \
    --base_cfg configs/stdloc_cambridge.yaml --output "$cfg" \
    --artifact_model_path "$MODEL_ROOT" \
    --detector_folder "$detector_folder" --detector_iters "$detector_steps" \
    --landmark_path "$LANDMARK_IDS" --landmark_meta_path "$LANDMARK_META" \
    --landmark_feature_override_path "$state" --override_landmark_features \
    --detect_num 4096 --nms 2 \
    --reprojection_error 8.306069762524674 \
    --match_threshold 0 --match_topk 1 \
    --max_matches_per_landmark 2 \
    --candidate_frontend_match_policy ignore --diagnostics \
    > "$LOG_ROOT/${label}_config.json"
  printf '%s\n' "$cfg"
}

run_test() {
  local label="$1"
  local state="$2"
  local detector_folder="$3"
  local detector_steps="$4"
  check_map_inputs
  test -f "$state"
  test -f "$MODEL_ROOT/$detector_folder/${detector_steps}_detector.pth"
  local cfg
  cfg="$(make_eval_config "$label" "$state" "$detector_folder" "$detector_steps")"
  "$PYTHON" stdloc.py \
    --model_path "$MODEL_ROOT" --source_path "$DATA_ROOT/ShopFacade" \
    --images processed --data_device cpu \
    --gaussian_type 2dgs --feature_type sp \
    --resolution 1 --longest_edge 640 --norm_before_render \
    --iteration 30000 --cfg "$cfg" \
    --prefix "lafgs-v2-${label}-r1-test" --sparse_only \
    2>&1 | tee "$LOG_ROOT/${label}_test.log"
}

run_verify() {
  "$PYTHON" -m pytest -q \
    tests/test_detector_free_map.py \
    tests/test_make_stdloc_eval_cfg.py \
    tests/test_surface_anchor.py \
    tests/test_stdloc_config_paths.py \
    tests/test_select_lafgs_map_checkpoint.py
}

LEGACY="$REPO_ROOT/scripts/run_lafgs_v2_progressive_legacy_shopfacade.sh"
case "$MODE" in
  descriptor) run_descriptor ;;
  test_strong)
    run_test strong_fallback "$STRONG_STATE" "$STRONG_DETECTOR_FOLDER" "$STRONG_DETECTOR_STEPS"
    ;;
  test_best)
    run_test numerical_best "$BEST_STATE" "$BEST_DETECTOR_FOLDER" "$BEST_DETECTOR_STEPS"
    ;;
  verify) run_verify ;;
  all)
    run_descriptor
    run_test strong_fallback "$STRONG_STATE" "$STRONG_DETECTOR_FOLDER" "$STRONG_DETECTOR_STEPS"
    ;;
  legacy_c0) "$LEGACY" c0 ;;
  legacy_c1) "$LEGACY" c1 ;;
  legacy_eval_c0) "$LEGACY" eval_c0 ;;
  legacy_eval_c1) "$LEGACY" eval_c1 ;;
esac
