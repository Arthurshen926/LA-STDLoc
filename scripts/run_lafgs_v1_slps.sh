#!/usr/bin/env bash
set -euo pipefail

# Scene-specific Self-Localization-Guided Pose-Sufficient Set Learning.
# The script consumes the frozen A1 single-descriptor map and never enables
# family descriptors, dense refinement, a learned sampler, or a second PnP.

if [[ $# -ne 3 ]]; then
  echo "Usage: bash $0 <scene> <gpu> <topk|corpus|train|apply|mapping|test|all>" >&2
  exit 2
fi

SCENE="$1"
GPU="$2"
MODE="$3"
case "$SCENE" in
  GreatCourt|KingsCollege|OldHospital|ShopFacade|StMarysChurch) ;;
  *) echo "Unsupported Cambridge scene: $SCENE" >&2; exit 2 ;;
esac
case "$GPU" in
  0|1) ;;
  *) echo "SLPS experiments are restricted to GPU 0 or 1" >&2; exit 2 ;;
esac
case "$MODE" in
  topk|corpus|train|apply|mapping|test|all) ;;
  *) echo "Unsupported mode: $MODE" >&2; exit 2 ;;
esac

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/root/miniconda3/envs/cybersim_agent/bin/python}"
FROZEN_ROOT="${LAFGS_V1_MULTISCENE_ROOT:-/mnt/pool/sqy/stdloc_lafgs_v1_frozen_multiscene_20260731}"
SLPS_ROOT="${LAFGS_SLPS_ROOT:-/mnt/pool/sqy/stdloc_lafgs_slps_20260731}"
DATA_ROOT="${CAMBRIDGE_DATA_ROOT:-/mnt/pool/sqy/Cambridge_stdloc}"
ROOT="$SLPS_ROOT/$SCENE"
BASE="$FROZEN_ROOT/$SCENE"
MAP="$BASE/self_localization_reconstruction/anchor_map_step_0175.pt"
METRIC="$BASE/self_localization_reconstruction/metric_state_step_0175.pt"
QUERY_CACHE="$BASE/runs/frozen_v1/query_cache_native_sparse_teacher.pt"
GRAPH="$BASE/function_graph/function_graph_v3.pt"
DYNAMIC="$BASE/family_refinement/dynamic_base.pt"
TOPK="$ROOT/topk16_single.pt"
CORPUS="$ROOT/slps_set_outcomes.pt"
SELECTOR_TAG="${LAFGS_SLPS_SELECTOR_TAG:-base}"
[[ "$SELECTOR_TAG" =~ ^[A-Za-z0-9_-]+$ ]] || {
  echo "Invalid LAFGS_SLPS_SELECTOR_TAG: $SELECTOR_TAG" >&2
  exit 2
}
if [[ "$SELECTOR_TAG" == "base" ]]; then
  SELECTOR="${LAFGS_SLPS_SELECTOR_PATH:-$ROOT/selector_slps.pt}"
  SELECTED="$ROOT/selected_adaptive.pt"
  SELECTED_PREFIX="$ROOT/selected"
  MAPPING_PREFIX="$ROOT/mapping"
  DEFAULT_TEST_VARIANT="adaptive"
else
  SELECTOR="${LAFGS_SLPS_SELECTOR_PATH:-$ROOT/selector_${SELECTOR_TAG}.pt}"
  SELECTED="$ROOT/selected_${SELECTOR_TAG}_adaptive.pt"
  SELECTED_PREFIX="$ROOT/selected_${SELECTOR_TAG}"
  MAPPING_PREFIX="$ROOT/mapping_${SELECTOR_TAG}"
  DEFAULT_TEST_VARIANT="${SELECTOR_TAG}_adaptive"
fi
TEST_VARIANT="${LAFGS_SLPS_TEST_VARIANT:-$DEFAULT_TEST_VARIANT}"
TEST_SELECTOR="${LAFGS_SLPS_TEST_SELECTOR:-$SELECTOR}"
TEST_SEEDS="${LAFGS_SLPS_TEST_SEEDS:-2026 2027 2028}"
MODEL_ROOT="$BASE/prior/rgb_matcha_2dgs"
SOURCE_ROOT="$DATA_ROOT/$SCENE"
BOOTSTRAP="$BASE/runs/frozen_v1/bootstrap"
LOGS="$ROOT/logs"

export CUDA_HOME=/usr/local/cuda-11.8
export PATH="/root/miniconda3/envs/cybersim_agent/bin:$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="/root/miniconda3/envs/cybersim_agent/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$REPO_ROOT"
export PYTHONHASHSEED=2026
mkdir -p "$ROOT" "$LOGS"
cd "$REPO_ROOT"

require_file() {
  [[ -f "$1" ]] || {
    echo "Required artifact is missing: $1" >&2
    exit 1
  }
}

run_logged() {
  local name="$1"
  shift
  printf '%q ' "$@" > "$LOGS/${name}.command.sh"
  printf '\n' >> "$LOGS/${name}.command.sh"
  "$@" 2>&1 | tee "$LOGS/${name}.log"
}

build_topk() {
  for path in "$MAP" "$METRIC" "$QUERY_CACHE" "$GRAPH"; do
    require_file "$path"
  done
  if [[ ! -f "$TOPK" ]]; then
    run_logged topk_single \
      env CUDA_VISIBLE_DEVICES="$GPU" \
      "$PYTHON" scripts/build_lafgs_topk_outcomes.py \
      --map "$MAP" --metric-state "$METRIC" \
      --query-cache "$QUERY_CACHE" --function-graph "$GRAPH" \
      --output "$TOPK" --topk 16 --device cuda
  fi
}

build_corpus() {
  build_topk
  require_file "$DYNAMIC"
  if [[ ! -f "$CORPUS" ]]; then
    local maximum_queries="${LAFGS_SLPS_MAXIMUM_QUERIES:-384}"
    if [[ -z "${LAFGS_SLPS_MAXIMUM_QUERIES:-}" ]] && \
       [[ "$SCENE" == "ShopFacade" ]]; then
      maximum_queries=0
    fi
    run_logged set_outcomes \
      env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
      "$PYTHON" scripts/build_lafgs_slps_set_outcomes.py \
      --map "$MAP" --query-cache "$QUERY_CACHE" \
      --topk-outcomes "$TOPK" --dynamic-outcomes "$DYNAMIC" \
      --output "$CORPUS" --maximum-queries "$maximum_queries" \
      --workers "${LAFGS_SLPS_WORKERS:-8}" \
      --budgets 256,384,512,768 --seed 2026 --secondary-seed 2027
  fi
}

train_selector() {
  build_corpus
  if [[ "$SELECTOR_TAG" != "base" ]]; then
    require_file "$SELECTOR"
    return
  fi
  if [[ ! -f "$SELECTOR" ]]; then
    run_logged selector_train \
      env CUDA_VISIBLE_DEVICES="$GPU" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
      "$PYTHON" scripts/train_lafgs_slps_selector.py \
      --corpus "$CORPUS" --output "$SELECTOR" --device cuda \
      --steps 3000 --learning-rate 0.0002 \
      --maximum-sets-per-query 16 --validation-interval 100 \
      --patience 8 --budgets 256,384,512,768 --seed 2026
  fi
}

apply_selector() {
  train_selector
  if [[ ! -f "$SELECTED" ]]; then
    run_logged "selector_apply_${SELECTOR_TAG}" \
      env CUDA_VISIBLE_DEVICES="$GPU" OMP_NUM_THREADS=1 \
      MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
      "$PYTHON" scripts/apply_lafgs_slps_selector.py \
      --map "$MAP" --query-cache "$QUERY_CACHE" \
      --topk-outcomes "$TOPK" --selector "$SELECTOR" \
      --output "$SELECTED" --device cuda
  fi
  local budget
  for budget in 256 384 512 768; do
    local output="${SELECTED_PREFIX}_fixed${budget}.pt"
    if [[ ! -f "$output" ]]; then
      run_logged "selector_apply_${SELECTOR_TAG}_fixed${budget}" \
        env CUDA_VISIBLE_DEVICES="$GPU" OMP_NUM_THREADS=1 \
        MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
        "$PYTHON" scripts/apply_lafgs_slps_selector.py \
        --map "$MAP" --query-cache "$QUERY_CACHE" \
        --topk-outcomes "$TOPK" --selector "$SELECTOR" \
        --output "$output" --device cuda --fixed-budget "$budget"
    fi
  done
}

mapping_gate() {
  apply_selector
  local variant
  for variant in adaptive fixed256 fixed384 fixed512 fixed768; do
    local selected="${SELECTED_PREFIX}_${variant}.pt"
    local output="${MAPPING_PREFIX}_${variant}_seed2026.json"
    [[ "$variant" == "adaptive" ]] && selected="$SELECTED"
    if [[ ! -f "$output" ]]; then
      run_logged "mapping_${SELECTOR_TAG}_${variant}" \
        env CUDA_VISIBLE_DEVICES="$GPU" \
        "$PYTHON" scripts/evaluate_lafgs_map_on_query_cache.py \
        --map "$MAP" --metric-state "$METRIC" \
        --query-cache "$QUERY_CACHE" --function-graph "$GRAPH" \
        --precomputed-topk "$selected" --output "$output" \
        --reprojection-error 12 --seed 2026 --split crossfold_mapping
    fi
  done
}

test_selector() {
  mapping_gate
  require_file "$TEST_SELECTOR"
  local seed
  for seed in $TEST_SEEDS; do
    [[ "$seed" =~ ^[0-9]+$ ]] || {
      echo "Invalid LAFGS_SLPS_TEST_SEEDS entry: $seed" >&2
      exit 2
    }
    local output="$ROOT/test/$TEST_VARIANT/seed${seed}"
    local cfg="$output/config.yaml"
    if [[ -f "$output/result.path" ]] && \
       [[ -f "$(<"$output/result.path")/results_summary.json" ]]; then
      continue
    fi
    mkdir -p "$output/results"
    "$PYTHON" scripts/make_stdloc_eval_cfg.py \
      --base_cfg configs/stdloc_cambridge.yaml --output "$cfg" \
      --artifact_model_path "$MODEL_ROOT" \
      --detector_folder ulfloc_native_no_detector --detector_iters 0 \
      --landmark_path "$BOOTSTRAP/sampled_idx.pkl" \
      --landmark_meta_path "$BOOTSTRAP/landmark_meta.pt" \
      --detect_num 2048 --nms 2 --sparse_ransac_seed "$seed" \
      --sparse_query_feature_contract native_resized_input \
      --reprojection_error 12 --match_threshold 0 --match_topk 1 \
      --max_matches_per_keypoint 0 --max_matches_per_landmark 0 \
      --candidate_frontend_match_policy error \
      --diagnostics --diagnostics_grid_rows 4 --diagnostics_grid_cols 4 \
      --diagnostics_voxel_size 1 \
      --diagnostics_task_translation_scale_m 0.1 \
      --diagnostics_task_rotation_scale_degrees 2 \
      --sparse_frontend ulfloc_native_metric \
      --materialized_anchor_map_path "$MAP" \
      --metric_state_path "$METRIC" \
      --pose_sufficient_selector_state_path "$TEST_SELECTOR" \
      --pose_sufficient_budget 768 > "$output/config_build.json"
    (
      export CUDA_VISIBLE_DEVICES="$GPU"
      export STDLOC_RESULTS_ROOT="$output/results"
      "$PYTHON" stdloc.py \
        --model_path "$MODEL_ROOT" --source_path "$SOURCE_ROOT" \
        --images processed --data_device cpu --gaussian_type 2dgs \
        --sh_degree 3 --feature_type sp --resolution 1 --longest_edge 0 \
        --norm_before_render --iteration 30000 --cfg "$cfg" \
        --prefix "lafgs-slps-$SCENE-$TEST_VARIANT-seed$seed" \
        --sparse_only --evaluation_camera_subset test \
        2>&1 | tee "$output/eval.log"
    )
    local result
    result="$(sed -n 's/^Output path: //p' "$output/eval.log" | tail -n 1)"
    [[ -n "$result" && -f "$result/results_summary.json" ]] || {
      echo "SLPS test failed for $SCENE seed $seed" >&2
      exit 1
    }
    printf '%s\n' "$result" > "$output/result.path"
  done
}

case "$MODE" in
  topk) build_topk ;;
  corpus) build_corpus ;;
  train) train_selector ;;
  apply) apply_selector ;;
  mapping) mapping_gate ;;
  test|all) test_selector ;;
esac
