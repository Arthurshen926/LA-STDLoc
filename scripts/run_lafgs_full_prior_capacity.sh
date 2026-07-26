#!/usr/bin/env bash
set -euo pipefail

# Full-RGB-prior micro-anchor capacity sweep. GPU 1 only.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/root/miniconda3/envs/cybersim_agent/bin/python}"
DATA_ROOT="${CAMBRIDGE_DATA_ROOT:-/mnt/pool/sqy/Cambridge_stdloc}"
ROOT="${LAFGS_MICRO_ANCHOR_ROOT:-/mnt/pool/sqy/stdloc_lafgs_micro_anchors_20260726/OldHospital}"
MODEL_ROOT="/mnt/pool/sqy/stdloc_lafgs_rgb_prior_sanitization_20260725/OldHospital/rgb_only_2dgs_stdloc"
CANONICAL_ROOT="/mnt/pool/sqy/stdloc_lafgs_rgb_prior_sanitization_20260725/OldHospital/runs/rgb_2dgs_robustq01_wide48_to32"
BANK_ROOT="$ROOT/micro_anchor_v4_full_prior_v1"
CONFIG_ROOT="$ROOT/configs"
LOG_ROOT="$ROOT/logs"
RESULT_ROOT="$ROOT/results"
STDLOC_RESULTS_ROOT="$ROOT/stdloc_results"
BUDGETS="${LAFGS_FULL_PRIOR_BUDGETS:-32 64 96 128 256 405}"

export CUDA_VISIBLE_DEVICES=1
export CUDA_HOME=/usr/local/cuda-11.8
export PATH="/root/miniconda3/envs/cybersim_agent/bin:$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="/root/miniconda3/envs/cybersim_agent/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$REPO_ROOT"
export STDLOC_RESULTS_ROOT
export PYTHONHASHSEED=2026

mkdir -p "$CONFIG_ROOT" "$LOG_ROOT" "$RESULT_ROOT" "$STDLOC_RESULTS_ROOT"
cd "$REPO_ROOT"

for budget in $BUDGETS; do
  label="$(printf 'full_prior_%04d' "$budget")"
  state="$BANK_ROOT/${label}.pt"
  if [[ "$budget" == 405 && ! -f "$state" ]]; then
    state="$BANK_ROOT/full_prior_micro_anchor_0512.pt"
  else
    state="$BANK_ROOT/full_prior_micro_anchor_$(printf '%04d' "$budget").pt"
  fi
  pointer="$RESULT_ROOT/${label}.path"
  [[ -f "$state" ]] || { echo "Missing anchor map: $state" >&2; exit 1; }
  if [[ -f "$pointer" && -f "$(<"$pointer")/results_summary.json" ]]; then
    echo "Reusing $label: $(<"$pointer")"
    continue
  fi
  cfg="$CONFIG_ROOT/${label}.yaml"
  "$PYTHON" scripts/make_stdloc_eval_cfg.py \
    --base_cfg configs/stdloc_cambridge.yaml --output "$cfg" \
    --artifact_model_path "$MODEL_ROOT" \
    --detector_folder ulfloc_native_no_detector --detector_iters 0 \
    --landmark_path "$CANONICAL_ROOT/bootstrap/sampled_idx.pkl" \
    --landmark_meta_path "$CANONICAL_ROOT/bootstrap/landmark_meta.pt" \
    --materialized_anchor_map_path "$state" \
    --detect_num 2048 --nms 2 \
    --sparse_query_feature_contract native_resized_input \
    --sparse_frontend ulfloc_native \
    --reprojection_error 12 --match_threshold 0 --match_topk 1 \
    --max_matches_per_landmark 0 --candidate_frontend_match_policy error \
    --diagnostics --diagnostics_grid_rows 4 --diagnostics_grid_cols 4 \
    --diagnostics_voxel_size 1 \
    --diagnostics_task_translation_scale_m 0.07160573943725686 \
    --diagnostics_task_rotation_scale_degrees 2 \
    > "$LOG_ROOT/${label}_config.json"
  "$PYTHON" stdloc.py \
    --model_path "$MODEL_ROOT" --source_path "$DATA_ROOT/OldHospital" \
    --images processed --data_device cpu --gaussian_type 2dgs \
    --sh_degree 3 --feature_type sp --resolution 1 --longest_edge 0 \
    --norm_before_render --iteration 30000 --cfg "$cfg" \
    --prefix "lafgs-v2-${label}" \
    --sparse_only --evaluation_camera_subset test \
    2>&1 | tee "$LOG_ROOT/${label}.log"
  output_path="$(sed -n 's/^Output path: //p' "$LOG_ROOT/${label}.log" | tail -n 1)"
  [[ -n "$output_path" && -f "$output_path/results_summary.json" ]] || {
    echo "Missing evaluation output for $label" >&2
    exit 1
  }
  printf '%s\n' "$output_path" > "$pointer"
done
