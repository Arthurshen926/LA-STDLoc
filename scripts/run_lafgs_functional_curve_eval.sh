#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/root/miniconda3/envs/cybersim_agent/bin/python}"
GPU="${LAFGS_FUNCTIONAL_GPU:?Set LAFGS_FUNCTIONAL_GPU}"
RUNS="${LAFGS_FUNCTIONAL_RUNS:?Set profile:budget entries}"
MAP_ROOT="${LAFGS_FUNCTIONAL_MAP_ROOT:?Set functional map root}"
OUTPUT_ROOT="${LAFGS_FUNCTIONAL_OUTPUT_ROOT:?Set evaluation output root}"
DATA_ROOT="${CAMBRIDGE_DATA_ROOT:-/mnt/pool/sqy/Cambridge_stdloc}"
MODEL_ROOT="/mnt/pool/sqy/stdloc_lafgs_rgb_prior_sanitization_20260725/OldHospital/rgb_only_2dgs_stdloc"
CANONICAL_ROOT="/mnt/pool/sqy/stdloc_lafgs_rgb_prior_sanitization_20260725/OldHospital/runs/rgb_2dgs_robustq01_wide48_to32"

export CUDA_VISIBLE_DEVICES="$GPU"
export CUDA_HOME=/usr/local/cuda-11.8
export PATH="/root/miniconda3/envs/cybersim_agent/bin:$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="/root/miniconda3/envs/cybersim_agent/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$REPO_ROOT"
export STDLOC_RESULTS_ROOT="$OUTPUT_ROOT/stdloc_results"
export PYTHONHASHSEED=2026

mkdir -p "$OUTPUT_ROOT/configs" "$OUTPUT_ROOT/logs" \
  "$OUTPUT_ROOT/results" "$STDLOC_RESULTS_ROOT"
cd "$REPO_ROOT"

for run in $RUNS; do
  profile="${run%%:*}"
  budget="${run##*:}"
  label="$(printf 'functional_%s_%05d' "$profile" "$budget")"
  state="$MAP_ROOT/$profile/functional_$(printf '%05d' "$budget").pt"
  pointer="$OUTPUT_ROOT/results/${label}.path"
  [[ -f "$state" ]] || { echo "Missing map: $state" >&2; exit 1; }
  if [[ -f "$pointer" && -f "$(<"$pointer")/results_summary.json" ]]; then
    echo "Reusing $label: $(<"$pointer")"
    continue
  fi
  cfg="$OUTPUT_ROOT/configs/${label}.yaml"
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
    --max_matches_per_landmark 0 \
    --candidate_frontend_match_policy error \
    --diagnostics --diagnostics_grid_rows 4 --diagnostics_grid_cols 4 \
    --diagnostics_voxel_size 1 \
    --diagnostics_task_translation_scale_m 0.07160573943725686 \
    --diagnostics_task_rotation_scale_degrees 2 \
    > "$OUTPUT_ROOT/logs/${label}_config.json"
  "$PYTHON" stdloc.py \
    --model_path "$MODEL_ROOT" \
    --source_path "$DATA_ROOT/OldHospital" \
    --images processed --data_device cpu --gaussian_type 2dgs \
    --sh_degree 3 --feature_type sp --resolution 1 --longest_edge 0 \
    --norm_before_render --iteration 30000 --cfg "$cfg" \
    --prefix "lafgs-v3-${label}" \
    --sparse_only --evaluation_camera_subset test \
    2>&1 | tee "$OUTPUT_ROOT/logs/${label}.log"
  output_path="$(
    sed -n 's/^Output path: //p' "$OUTPUT_ROOT/logs/${label}.log" |
      tail -n 1
  )"
  [[ -n "$output_path" && -f "$output_path/results_summary.json" ]] || {
    echo "Missing evaluation output for $label" >&2
    exit 1
  }
  printf '%s\n' "$output_path" > "$pointer"
done
