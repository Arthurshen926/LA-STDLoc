#!/usr/bin/env bash
set -euo pipefail

# Frozen five-scene P0 gate for joint top-K identity assignment and fixed set
# selection.  Every dump uses the same single-descriptor A1 map and native
# full-resolution sparse frontend; no family prototype or learned selector is
# permitted in this protocol.

if [[ $# -ne 3 ]]; then
  echo "Usage: bash $0 <scene> <gpu> <dump|oracle|all>" >&2
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
  *) echo "Joint-assignment experiments are restricted to GPU 0 or 1" >&2; exit 2 ;;
esac
case "$MODE" in
  dump|oracle|all) ;;
  *) echo "Unsupported mode: $MODE" >&2; exit 2 ;;
esac

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/root/miniconda3/envs/cybersim_agent/bin/python}"
DATA_ROOT="${CAMBRIDGE_DATA_ROOT:-/mnt/pool/sqy/Cambridge_stdloc}"
FROZEN_ROOT="${LAFGS_V1_MULTISCENE_ROOT:-/mnt/pool/sqy/stdloc_lafgs_v1_frozen_multiscene_20260731}"
OUTPUT_ROOT="${LAFGS_JOINT_ASSIGNMENT_ROOT:-/mnt/pool/sqy/stdloc_lafgs_joint_assignment_20260731}"

SCENE_ROOT="$FROZEN_ROOT/$SCENE"
MODEL_ROOT="$SCENE_ROOT/prior/rgb_matcha_2dgs"
SOURCE_ROOT="$DATA_ROOT/$SCENE"
BOOTSTRAP="$SCENE_ROOT/runs/frozen_v1/bootstrap"
MAP="$SCENE_ROOT/self_localization_reconstruction/anchor_map_step_0175.pt"
METRIC="$SCENE_ROOT/self_localization_reconstruction/metric_state_step_0175.pt"
RUN_ROOT="$OUTPUT_ROOT/$SCENE/A1_single_descriptor_protected_topk16/seed2026"
CFG="$RUN_ROOT/config.yaml"
RESULTS="$RUN_ROOT/results"
RESULT_POINTER="$RUN_ROOT/result.path"
REPORT="$RUN_ROOT/joint_assignment_p0.json"

export CUDA_HOME=/usr/local/cuda-11.8
export PATH="/root/miniconda3/envs/cybersim_agent/bin:$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="/root/miniconda3/envs/cybersim_agent/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$REPO_ROOT"
export PYTHONHASHSEED=2026
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export STDLOC_CAMERA_LOADER_WORKERS=0

require_file() {
  [[ -f "$1" ]] || {
    echo "Required frozen A1 artifact is missing: $1" >&2
    exit 1
  }
}

dump_graph() {
  require_file "$MAP"
  require_file "$METRIC"
  require_file "$BOOTSTRAP/sampled_idx.pkl"
  require_file "$BOOTSTRAP/landmark_meta.pt"
  mkdir -p "$RUN_ROOT" "$RESULTS"
  if [[ -f "$RESULT_POINTER" ]]; then
    local existing
    existing="$(<"$RESULT_POINTER")"
    if [[ -f "$existing/discrete_oracle_dump/manifest.json" ]]; then
      return
    fi
  fi

  "$PYTHON" scripts/make_stdloc_eval_cfg.py \
    --base_cfg configs/stdloc_cambridge.yaml --output "$CFG" \
    --artifact_model_path "$MODEL_ROOT" \
    --detector_folder ulfloc_native_no_detector --detector_iters 0 \
    --landmark_path "$BOOTSTRAP/sampled_idx.pkl" \
    --landmark_meta_path "$BOOTSTRAP/landmark_meta.pt" \
    --detect_num 2048 --nms 2 --sparse_ransac_seed 2026 \
    --sparse_query_feature_contract native_resized_input \
    --reprojection_error 12 --match_threshold 0 --match_topk 1 \
    --max_matches_per_keypoint 0 --max_matches_per_landmark 0 \
    --candidate_frontend_match_policy error \
    --diagnostics --diagnostics_dump_discrete_oracle \
    --diagnostics_oracle_topk 16 \
    --diagnostics_grid_rows 4 --diagnostics_grid_cols 4 \
    --diagnostics_voxel_size 1 \
    --diagnostics_task_translation_scale_m 0.1 \
    --diagnostics_task_rotation_scale_degrees 2 \
    --sparse_frontend ulfloc_native_metric \
    --materialized_anchor_map_path "$MAP" \
    --metric_state_path "$METRIC" > "$RUN_ROOT/config_build.json"

  (
    export CUDA_VISIBLE_DEVICES="$GPU"
    export STDLOC_RESULTS_ROOT="$RESULTS"
    "$PYTHON" stdloc.py \
      --model_path "$MODEL_ROOT" --source_path "$SOURCE_ROOT" \
      --images processed --data_device cpu --gaussian_type 2dgs \
      --sh_degree 3 --feature_type sp --resolution 1 --longest_edge 0 \
      --norm_before_render --iteration 30000 --cfg "$CFG" \
      --prefix "lafgs-joint-p0-$SCENE-A1-protected-topk-seed2026" \
      --sparse_only --evaluation_camera_subset test \
      2>&1 | tee "$RUN_ROOT/dump.log"
  )
  local result
  result="$(sed -n 's/^Output path: //p' "$RUN_ROOT/dump.log" | tail -n 1)"
  [[ -n "$result" && -f "$result/discrete_oracle_dump/manifest.json" ]] || {
    echo "Failed to produce the $SCENE A1 discrete dump" >&2
    exit 1
  }
  printf '%s\n' "$result" > "$RESULT_POINTER"
}

evaluate_oracle() {
  dump_graph
  local result
  result="$(<"$RESULT_POINTER")"
  "$PYTHON" scripts/evaluate_joint_assignment_p0.py \
    --dump-dir "$result/discrete_oracle_dump" \
    --output "$REPORT" --radius 2 --seed 2026 \
    --bootstrap-samples 2000 \
    2>&1 | tee "$RUN_ROOT/oracle.log"
}

cd "$REPO_ROOT"
case "$MODE" in
  dump) dump_graph ;;
  oracle) evaluate_oracle ;;
  all) evaluate_oracle ;;
esac
