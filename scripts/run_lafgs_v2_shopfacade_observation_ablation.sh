#!/usr/bin/env bash
set -euo pipefail

# Controlled detector-free observation curriculum ablation.  All variants use
# the same proven strong bank, descriptor state, fixed detector, held-out split,
# query cache, and PnP configuration.  Only the observation distribution changes.

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 <gpu:1|2> <q0|q1|q2|q3|control|select|all>" >&2
  exit 2
fi

GPU="$1"
MODE="$2"
case "$GPU" in 1|2) ;; *) echo "GPU must be 1 or 2" >&2; exit 2 ;; esac
case "$MODE" in q0|q1|q2|q3|control|select|all) ;; *) echo "Unknown mode: $MODE" >&2; exit 2 ;; esac

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/root/miniconda3/envs/cybersim_agent/bin/python}"
DATA_ROOT="${CAMBRIDGE_DATA_ROOT:-/mnt/pool/sqy/Cambridge_stdloc}"
MODEL_ROOT="${LAFGS_V2_MODEL_ROOT:-/mnt/pool/sqy/stdloc_lafgs_v2_20260717/ShopFacade}"
ACTIVE_ROOT="$MODEL_ROOT/activefield_p1b_exactset_fim_1500"
LANDMARK_IDS="$ACTIVE_ROOT/sampled_idx.pkl"
LANDMARK_META="$ACTIVE_ROOT/landmark_meta.pt"
STRONG_STATE="$ACTIVE_ROOT/750_candidate_teacher_state.pt"
DETECTOR_FOLDER="strongmap_detectoronly_2000_exact_control"
DETECTOR_STEPS=2000
QUERY_CACHE="$MODEL_ROOT/lafgs_map_frontendexact_query_cache_v6.pt"
ROOT="${LAFGS_V2_OBSERVATION_ABLATION_ROOT:-/mnt/pool/sqy/stdloc_lafgs_v2_clean_matcha_20260719/ShopFacade/observation_ablation}"
CONFIG_ROOT="$ROOT/configs"
LOG_ROOT="$ROOT/logs"
RESULT_ROOT="$ROOT/results"
STDLOC_RESULTS_ROOT="$ROOT/stdloc_results"
STEPS="${LAFGS_V2_OBSERVATION_STEPS:-1000}"

export CUDA_VISIBLE_DEVICES="$GPU"
export CUDA_HOME=/usr/local/cuda-11.8
export PATH="/root/miniconda3/envs/cybersim_agent/bin:$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="/root/miniconda3/envs/cybersim_agent/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$REPO_ROOT"
export PYTHONHASHSEED=2026
export STDLOC_RESULTS_ROOT

mkdir -p "$CONFIG_ROOT" "$LOG_ROOT" "$RESULT_ROOT" "$STDLOC_RESULTS_ROOT"
cd "$REPO_ROOT"

require_file() {
  [[ -f "$1" ]] || { echo "Required artifact is missing: $1" >&2; exit 1; }
}

run_logged() {
  local stage="$1"
  shift
  printf '%q ' "$@" > "$LOG_ROOT/${stage}.command.sh"
  printf '\n' >> "$LOG_ROOT/${stage}.command.sh"
  "$@" 2>&1 | tee "$LOG_ROOT/${stage}.log"
}

variant_dir() {
  printf '%s/q%s_%s\n' "$ROOT" "$1" "$STEPS"
}

train_variant() {
  local variant="$1"
  local output_dir
  output_dir="$(variant_dir "$variant")"
  if [[ -f "$output_dir/${STEPS}_lafgs_map_state.pt" ]]; then
    echo "[observation ablation] Reusing q${variant}: $output_dir"
    return
  fi
  require_file "$LANDMARK_IDS"
  require_file "$STRONG_STATE"
  require_file "$QUERY_CACHE"
  local variant_args=()
  case "$variant" in
    0)
      variant_args=(
        --proposal_jitter_std 0 --proposal_jitter_max 0
        --generic_proposal_count 0 --generic_proposal_weight 0
        --unmatched_rejection_weight 0
      )
      ;;
    1)
      variant_args=(
        --proposal_jitter_std 0.75 --proposal_jitter_max 2
        --generic_proposal_count 0 --generic_proposal_weight 0
        --unmatched_rejection_weight 0
      )
      ;;
    2)
      variant_args=(
        --proposal_jitter_std 0.75 --proposal_jitter_max 2
        --generic_proposal_count 512 --generic_proposal_weight 0.25
        --generic_proposal_nms_radius 2 --generic_proposal_positive_radius 2
        --no-generic_proposal_include_unmatched
        --unmatched_rejection_weight 0
      )
      ;;
    3)
      variant_args=(
        --proposal_jitter_std 0.75 --proposal_jitter_max 2
        --generic_proposal_count 512 --generic_proposal_weight 0.25
        --generic_proposal_nms_radius 2 --generic_proposal_positive_radius 2
        --generic_proposal_include_unmatched
        --unmatched_rejection_weight 0.1 --unmatched_max_similarity 0.5
      )
      ;;
  esac
  run_logged "q${variant}_train" \
    "$PYTHON" train_lafgs_map.py \
    --model_path "$MODEL_ROOT" --source_path "$DATA_ROOT/ShopFacade" \
    --images processed --data_device cpu --gaussian_type 2dgs --feature_type sp \
    --resolution 1 --longest_edge 640 --norm_before_render --load_iteration 30000 \
    --output_dir "$output_dir" --scaffold_mode file --landmark_path "$LANDMARK_IDS" \
    --initial_state_path "$STRONG_STATE" --initial_state_blend 1 \
    --initial_state_alignment exact --query_cache_path "$QUERY_CACHE" \
    --query_cache_policy readonly --visibility_mode depth --objective hard \
    --steps "$STEPS" --save_steps 250 500 750 "$STEPS" \
    --feature_lr 5e-5 --weight_decay 1e-4 --mvinit_mode medoid \
    --mv_weight 0.5 --retrieval_weight 0.5 --trust_weight 0.1 --local_weight 0.05 \
    --hypothesis_topk 32 --positive_radius_px 2 --negative_radius_px 6 \
    --retrieval_margin 0.05 --missed_positive_weight 1 --missed_positive_margin 0.05 \
    --dustbin_weight 0 --geometry_weight 0 --pose_weight 0 --pose_gradient_mode off \
    --validation_ratio 0.2 --split_mode temporal_block --split_seed 2026 --train_seed 2026 \
    --max_observations 512 --validation_observations 512 --log_interval 100 \
    "${variant_args[@]}"
}

make_config() {
  local label="$1"
  local state="$2"
  local cfg="$CONFIG_ROOT/${label}.yaml"
  "$PYTHON" scripts/make_stdloc_eval_cfg.py \
    --base_cfg configs/stdloc_cambridge.yaml --output "$cfg" \
    --artifact_model_path "$MODEL_ROOT" \
    --detector_folder "$DETECTOR_FOLDER" --detector_iters "$DETECTOR_STEPS" \
    --landmark_path "$LANDMARK_IDS" --landmark_meta_path "$LANDMARK_META" \
    --landmark_feature_override_path "$state" --override_landmark_features \
    --detect_num 4096 --nms 2 --reprojection_error 8.306069762524674 \
    --match_threshold 0 --match_topk 1 --max_matches_per_landmark 2 \
    --candidate_frontend_match_policy ignore --diagnostics \
    > "$LOG_ROOT/${label}_config.json"
  printf '%s\n' "$cfg"
}

evaluate_state() {
  local label="$1"
  local state="$2"
  local ref="$RESULT_ROOT/${label}.results_path"
  if [[ -f "$ref" ]] && [[ -f "$(<"$ref")/results_summary.json" ]]; then
    echo "[observation ablation] Reusing validation: $label"
    return
  fi
  local cfg
  cfg="$(make_config "$label" "$state")"
  local log="$LOG_ROOT/${label}_validation.log"
  local cmd=(
    "$PYTHON" stdloc.py --model_path "$MODEL_ROOT" --source_path "$DATA_ROOT/ShopFacade"
    --images processed --data_device cpu --gaussian_type 2dgs --feature_type sp
    --resolution 1 --longest_edge 640 --norm_before_render --iteration 30000
    --cfg "$cfg" --prefix "lafgs-v2-observation-${label}" --sparse_only
    --evaluation_camera_subset candidate_validation --candidate_direct_validation_holdout
    --candidate_validation_ratio 0.2 --candidate_split_mode temporal_block --candidate_split_seed 2026
  )
  printf '%q ' "${cmd[@]}" > "$LOG_ROOT/${label}_validation.command.sh"
  printf '\n' >> "$LOG_ROOT/${label}_validation.command.sh"
  "${cmd[@]}" 2>&1 | tee "$log"
  local output_path
  output_path="$(sed -n 's/^Output path: //p' "$log" | tail -n 1)"
  [[ -n "$output_path" && -f "$output_path/results_summary.json" ]] || {
    echo "Validation result missing for $label" >&2
    exit 1
  }
  printf '%s\n' "$output_path" > "$ref"
}

run_control() {
  require_file "$STRONG_STATE"
  require_file "$MODEL_ROOT/$DETECTOR_FOLDER/${DETECTOR_STEPS}_detector.pth"
  evaluate_state control "$STRONG_STATE"
}

evaluate_variant() {
  local variant="$1"
  train_variant "$variant"
  local output_dir
  output_dir="$(variant_dir "$variant")"
  local step
  for step in 250 500 750 "$STEPS"; do
    local state="$output_dir/${step}_lafgs_map_state.pt"
    [[ -f "$state" ]] && evaluate_state "q${variant}_${step}" "$state"
  done
}

select_variants() {
  run_control
  local command=(
    "$PYTHON" scripts/select_lafgs_map_checkpoint.py
    --control_results "$(<"$RESULT_ROOT/control.results_path")/results_summary.json"
    --control_state "$STRONG_STATE"
    --min_te_gain_cm 0.02 --metric_tolerance 1e-9
    --output "$RESULT_ROOT/selection_report.json"
  )
  local variant step
  for variant in 0 1 2 3; do
    local output_dir
    output_dir="$(variant_dir "$variant")"
    for step in 250 500 750 "$STEPS"; do
      local state="$output_dir/${step}_lafgs_map_state.pt"
      local ref="$RESULT_ROOT/q${variant}_${step}.results_path"
      if [[ -f "$state" && -f "$ref" ]]; then
        command+=(--candidate "q${variant}_${step}" "$(<"$ref")/results_summary.json" "$state")
      fi
    done
  done
  run_logged select "${command[@]}"
}

case "$MODE" in
  q0) evaluate_variant 0 ;;
  q1) evaluate_variant 1 ;;
  q2) evaluate_variant 2 ;;
  q3) evaluate_variant 3 ;;
  control) run_control ;;
  select) select_variants ;;
  all)
    run_control
    evaluate_variant 0
    evaluate_variant 1
    evaluate_variant 2
    evaluate_variant 3
    select_variants
    ;;
esac
