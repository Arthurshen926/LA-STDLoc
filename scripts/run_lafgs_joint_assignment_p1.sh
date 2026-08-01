#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 || $# -gt 4 ]]; then
  echo "Usage: bash $0 <scene> <gpu> <graph|train|eval|all> [A1-All|S512-PoseSufficient|S1024-Block8]" >&2
  exit 2
fi

SCENE="$1"
GPU="$2"
MODE="$3"
SELECTOR="${4:-A1-All}"
SCENES=(GreatCourt KingsCollege OldHospital ShopFacade StMarysChurch)
[[ " ${SCENES[*]} " == *" $SCENE "* ]] || { echo "Unsupported scene: $SCENE" >&2; exit 2; }
[[ "$GPU" == 0 || "$GPU" == 1 ]] || { echo "GPU must be 0 or 1" >&2; exit 2; }
case "$MODE" in graph|train|eval|all) ;; *) echo "Unsupported mode: $MODE" >&2; exit 2 ;; esac
case "$SELECTOR" in A1-All|S512-PoseSufficient|S1024-Block8) ;; *) echo "Unsupported selector: $SELECTOR" >&2; exit 2 ;; esac

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/root/miniconda3/envs/cybersim_agent/bin/python}"
DATA_ROOT="${CAMBRIDGE_DATA_ROOT:-/mnt/pool/sqy/Cambridge_stdloc}"
FROZEN_ROOT="${LAFGS_V1_MULTISCENE_ROOT:-/mnt/pool/sqy/stdloc_lafgs_v1_frozen_multiscene_20260731}"
OUTPUT_ROOT="${LAFGS_JOINT_ASSIGNMENT_P1_ROOT:-/mnt/pool/sqy/stdloc_lafgs_joint_assignment_p1_20260731}"
ARTIFACT_VERSION="${LAFGS_JOINT_ASSIGNMENT_ARTIFACT_VERSION:-5}"
case "$ARTIFACT_VERSION" in 4|5) ;; *) echo "LAFGS_JOINT_ASSIGNMENT_ARTIFACT_VERSION must be 4 or 5" >&2; exit 2 ;; esac
EVAL_TAG="${LAFGS_JOINT_ASSIGNMENT_EVAL_TAG:-evaluation_v${ARTIFACT_VERSION}}"
USE_LEARNED_NULL="${LAFGS_JOINT_ASSIGNMENT_USE_LEARNED_NULL:-1}"
case "$USE_LEARNED_NULL" in 0|1) ;; *) echo "LAFGS_JOINT_ASSIGNMENT_USE_LEARNED_NULL must be 0 or 1" >&2; exit 2 ;; esac
SCENE_ROOT="$FROZEN_ROOT/$SCENE"
MODEL_ROOT="$SCENE_ROOT/prior/rgb_matcha_2dgs"
SOURCE_ROOT="$DATA_ROOT/$SCENE"
BOOTSTRAP="$SCENE_ROOT/runs/frozen_v1/bootstrap"
MAP="$SCENE_ROOT/self_localization_reconstruction/anchor_map_step_0175.pt"
METRIC="$SCENE_ROOT/self_localization_reconstruction/metric_state_step_0175.pt"
QUERY_CACHE="$SCENE_ROOT/runs/frozen_v1/query_cache_native_fullres_k2048.pt"
POSITIVE="$SCENE_ROOT/self_localization_reconstruction/complete_positive_teacher.pt"
DYNAMIC="$SCENE_ROOT/family_refinement/dynamic_base.pt"
SELECTOR_STATE="$SCENE_ROOT/pose_sufficient_selector/selector_model_fixed0512.pt"
GRAPH_DIR="$OUTPUT_ROOT/graphs"
GRAPH="$GRAPH_DIR/${SCENE}_K8_v${ARTIFACT_VERSION}.pt"
MODEL_DIR="$OUTPUT_ROOT/models/heldout_$SCENE"
ASSIGNMENT="$MODEL_DIR/joint_assignment_K8_v${ARTIFACT_VERSION}.pt"

export CUDA_HOME=/usr/local/cuda-11.8
export PATH="/root/miniconda3/envs/cybersim_agent/bin:$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="/root/miniconda3/envs/cybersim_agent/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$REPO_ROOT"
export PYTHONHASHSEED=2026
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export STDLOC_CAMERA_LOADER_WORKERS=0

require_file() { [[ -f "$1" ]] || { echo "Missing required artifact: $1" >&2; exit 1; }; }

build_graph() {
  for path in "$MAP" "$METRIC" "$QUERY_CACHE" "$POSITIVE" "$DYNAMIC" "$SELECTOR_STATE"; do require_file "$path"; done
  mkdir -p "$GRAPH_DIR"
  [[ -f "$GRAPH" ]] && return
  CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" scripts/build_joint_assignment_scene_graph.py \
    --scene "$SCENE" --map "$MAP" --metric-state "$METRIC" \
    --query-cache "$QUERY_CACHE" --positive-teacher "$POSITIVE" \
    --dynamic-outcomes "$DYNAMIC" --selector-state "$SELECTOR_STATE" \
    --output "$GRAPH" --topk 8 --patch-radius 2 --patch-step-px 8 \
    --maximum-rows 1024 --null-to-positive-ratio 2 \
    --exact-query-budget 20 --exact-rows-per-query 2 \
    --exact-candidates-per-row 2 --seed 2026 \
    2>&1 | tee "$GRAPH_DIR/${SCENE}_K8_v${ARTIFACT_VERSION}.log"
}

train_head() {
  local graph_args=()
  for value in "${SCENES[@]}"; do
    require_file "$GRAPH_DIR/${value}_K8_v${ARTIFACT_VERSION}.pt"
    graph_args+=(--graph "$GRAPH_DIR/${value}_K8_v${ARTIFACT_VERSION}.pt")
  done
  mkdir -p "$MODEL_DIR"
  [[ -f "$ASSIGNMENT" ]] && return
  CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" scripts/train_joint_assignment_loso.py \
    "${graph_args[@]}" --heldout-scene "$SCENE" --output "$ASSIGNMENT" \
    --epochs 8 --learning-rate 0.0005 --weight-decay 0.0001 \
    --hidden-dim 48 --bounded-residual-max 0.05 \
    --null-loss-weight 0.5 --null-minimum-precision 0.95 --seed 2026 \
    2>&1 | tee "$MODEL_DIR/train_v${ARTIFACT_VERSION}.log"
}

evaluate_head() {
  require_file "$ASSIGNMENT"
  local run_root="$OUTPUT_ROOT/$EVAL_TAG/$SCENE/$SELECTOR/seed2026"
  local cfg="$run_root/config.yaml"
  local pointer="$run_root/result.path"
  local null_args=()
  if [[ "$USE_LEARNED_NULL" == 1 ]]; then
    null_args+=(--rerank_use_learned_null)
  else
    null_args+=(--no-rerank_use_learned_null)
  fi
  mkdir -p "$run_root/results"
  if [[ -f "$pointer" && -f "$(<"$pointer")/results_summary.json" ]]; then return; fi
  "$PYTHON" scripts/make_stdloc_eval_cfg.py \
    --base_cfg configs/stdloc_cambridge.yaml --output "$cfg" \
    --artifact_model_path "$MODEL_ROOT" \
    --detector_folder ulfloc_native_no_detector --detector_iters 0 \
    --landmark_path "$BOOTSTRAP/sampled_idx.pkl" \
    --landmark_meta_path "$BOOTSTRAP/landmark_meta.pt" \
    --detect_num 2048 --nms 2 --sparse_ransac_seed 2026 \
    --sparse_query_feature_contract native_resized_input \
    --reprojection_error 12 --match_threshold 0 --match_topk 1 \
    --max_matches_per_keypoint 0 --max_matches_per_landmark 0 \
    --candidate_frontend_match_policy error \
    --diagnostics --diagnostics_grid_rows 4 --diagnostics_grid_cols 4 \
    --diagnostics_retrieval_topk 8 \
    --diagnostics_voxel_size 1 \
    --diagnostics_task_translation_scale_m 0.1 \
    --diagnostics_task_rotation_scale_degrees 2 \
    --sparse_frontend ulfloc_native_rerank \
    --materialized_anchor_map_path "$MAP" --metric_state_path "$METRIC" \
    --rerank_state_path "$ASSIGNMENT" --rerank_topk 8 \
    --rerank_patch_radius 2 --rerank_patch_step_px 8 \
    "${null_args[@]}" \
    --joint_assignment_fixed_selector "$SELECTOR" > "$run_root/config_build.json"
  (
    export CUDA_VISIBLE_DEVICES="$GPU"
    export STDLOC_RESULTS_ROOT="$run_root/results"
    "$PYTHON" stdloc.py \
      --model_path "$MODEL_ROOT" --source_path "$SOURCE_ROOT" \
      --images processed --data_device cpu --gaussian_type 2dgs \
      --sh_degree 3 --feature_type sp --resolution 1 --longest_edge 0 \
      --norm_before_render --iteration 30000 --cfg "$cfg" \
      --prefix "lafgs-joint-p1-$SCENE-$SELECTOR-seed2026" \
      --sparse_only --evaluation_camera_subset test \
      2>&1 | tee "$run_root/eval.log"
  )
  local result
  result="$(sed -n 's/^Output path: //p' "$run_root/eval.log" | tail -n 1)"
  [[ -n "$result" && -f "$result/results_summary.json" ]] || { echo "Evaluation failed" >&2; exit 1; }
  printf '%s\n' "$result" > "$pointer"
}

cd "$REPO_ROOT"
case "$MODE" in
  graph) build_graph ;;
  train) train_head ;;
  eval) evaluate_head ;;
  all) build_graph; train_head; evaluate_head ;;
esac
