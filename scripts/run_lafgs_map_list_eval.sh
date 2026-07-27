#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/root/miniconda3/envs/cybersim_agent/bin/python}"
GPU="${LAFGS_MAP_LIST_GPU:?Set LAFGS_MAP_LIST_GPU}"
RUNS="${LAFGS_MAP_LIST_RUNS:?Set label=absolute_map entries}"
RANSAC_SEED="${LAFGS_MAP_LIST_RANSAC_SEED:-0}"
ORACLE_DUMP="${LAFGS_MAP_LIST_ORACLE_DUMP:-0}"
ORACLE_TOPK="${LAFGS_MAP_LIST_ORACLE_TOPK:-32}"
METRIC_STATE="${LAFGS_MAP_LIST_METRIC_STATE:-}"
OUTPUT_ROOT="${LAFGS_MAP_LIST_OUTPUT_ROOT:?Set output root}"
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

ORACLE_ARGS=()
if [[ "$ORACLE_DUMP" == "1" ]]; then
  ORACLE_ARGS=(
    --diagnostics_dump_discrete_oracle
    --diagnostics_oracle_topk "$ORACLE_TOPK"
  )
fi

for run in $RUNS; do
  label="${run%%=*}"
  state="${run#*=}"
  pointer="$OUTPUT_ROOT/results/${label}.path"
  [[ -f "$state" ]] || { echo "Missing map: $state" >&2; exit 1; }
  if [[ -f "$pointer" && -f "$(<"$pointer")/results_summary.json" ]]; then
    echo "Reusing $label: $(<"$pointer")"
    continue
  fi
  cfg="$OUTPUT_ROOT/configs/${label}.yaml"
  FRONTEND_ARGS=(--sparse_frontend ulfloc_native)
  if [[ -n "$METRIC_STATE" ]]; then
    [[ -f "$METRIC_STATE" ]] || {
      echo "Missing metric state: $METRIC_STATE" >&2
      exit 1
    }
    FRONTEND_ARGS=(
      --sparse_frontend ulfloc_native_metric
      --metric_state_path "$METRIC_STATE"
    )
  fi
  "$PYTHON" scripts/make_stdloc_eval_cfg.py \
    --base_cfg configs/stdloc_cambridge.yaml --output "$cfg" \
    --artifact_model_path "$MODEL_ROOT" \
    --detector_folder ulfloc_native_no_detector --detector_iters 0 \
    --landmark_path "$CANONICAL_ROOT/bootstrap/sampled_idx.pkl" \
    --landmark_meta_path "$CANONICAL_ROOT/bootstrap/landmark_meta.pt" \
    --materialized_anchor_map_path "$state" \
    --detect_num 2048 --nms 2 \
    --sparse_query_feature_contract native_resized_input \
    "${FRONTEND_ARGS[@]}" \
    --reprojection_error 12 --match_threshold 0 --match_topk 1 \
    --max_matches_per_landmark 0 \
    --candidate_frontend_match_policy error \
    --diagnostics --diagnostics_grid_rows 4 --diagnostics_grid_cols 4 \
    --diagnostics_voxel_size 1 \
    --diagnostics_task_translation_scale_m 0.07160573943725686 \
    --diagnostics_task_rotation_scale_degrees 2 \
    "${ORACLE_ARGS[@]}" \
    > "$OUTPUT_ROOT/logs/${label}_config.json"
  "$PYTHON" - "$cfg" "$RANSAC_SEED" <<'PY'
import sys
from pathlib import Path

path = Path(sys.argv[1])
seed = int(sys.argv[2])
lines = path.read_text(encoding="utf-8").splitlines()
inside_sparse = False
updated = False
for index, line in enumerate(lines):
    if line and not line.startswith((" ", "\t")):
        inside_sparse = line.rstrip() == "sparse:"
    elif inside_sparse and line.lstrip().startswith("ransac_seed:"):
        indent = line[: len(line) - len(line.lstrip())]
        lines[index] = f"{indent}ransac_seed: {seed}"
        updated = True
        break
if not updated:
    raise RuntimeError("sparse.ransac_seed was not found in generated config")
path.write_text("\n".join(lines) + "\n", encoding="utf-8")
PY
  "$PYTHON" stdloc.py \
    --model_path "$MODEL_ROOT" \
    --source_path "$DATA_ROOT/OldHospital" \
    --images processed --data_device cpu --gaussian_type 2dgs \
    --sh_degree 3 --feature_type sp --resolution 1 --longest_edge 0 \
    --norm_before_render --iteration 30000 --cfg "$cfg" \
    --prefix "lafgs-v4-${label}" \
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
