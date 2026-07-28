#!/usr/bin/env bash
set -euo pipefail

# Full-chain LaFGS rebuild. The only reusable learned inputs are the frozen
# RGB-only 2DGS and frozen SuperPoint weights. Every localization artifact is
# written below CLEAN_ROOT.

if [[ $# -ne 1 ]]; then
  echo "Usage: bash $0 <base|graph|maps|metric|eval|v10|v10eval|all_core|all_pose_reserve|all>" >&2
  exit 2
fi
MODE="$1"
case "$MODE" in
  base|graph|maps|metric|eval|v10|v10eval|all_core|all_pose_reserve|all) ;;
  *) exit 2 ;;
esac

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/root/miniconda3/envs/cybersim_agent/bin/python}"
CLEAN_ROOT="${LAFGS_CLEAN_ROOT:-/mnt/pool/sqy/stdloc_lafgs_clean_rebuild_20260728_v2}"
SCENE="OldHospital"
ROOT="$CLEAN_ROOT/$SCENE"
RUN_TAG="clean_rgb2dgs_trackfirst_stagea1000_v2"
RUN_ROOT="$ROOT/runs/$RUN_TAG"
MODEL_ROOT="${LAFGS_FROZEN_RGB_PRIOR:-/mnt/pool/sqy/stdloc_lafgs_rgb_prior_sanitization_20260725/OldHospital/rgb_only_2dgs_stdloc}"
DATA_ROOT="${CAMBRIDGE_DATA_ROOT:-/mnt/pool/sqy/Cambridge_stdloc}"
SOURCE_ROOT="$DATA_ROOT/$SCENE"
PLY="$MODEL_ROOT/point_cloud/iteration_30000/point_cloud.ply"
PRIOR_MANIFEST="$MODEL_ROOT/rgb_prior_manifest.json"
SP_WEIGHTS="$REPO_ROOT/encoders/sp_encoder/weights/superpoint_v1.pth"
MASKS="$SOURCE_ROOT/processed/masks.pkl"

BOOTSTRAP="$RUN_ROOT/bootstrap"
STAGE_A="$RUN_ROOT/stage_a_combined_1000"
STATISTICS="$RUN_ROOT/statistics_combined_1000_frozen_g3_track_provenance_v1"
QUERY_CACHE="$RUN_ROOT/query_cache_native_fullres_k2048.pt"
SPARSE_QUERY_CACHE="$RUN_ROOT/query_cache_native_sparse_teacher.pt"
VISIBILITY="$RUN_ROOT/visibility_48000_native.pt"
BASE_STATE="$STAGE_A/1000_lafgs_map_state.pt"
TRACK_PAYLOAD="$STATISTICS/track_micro_anchor_payload.pt"
CANONICAL="$ROOT/canonical/canonical_48000.pt"
GRAPH_DIR="$ROOT/function_graph"
GRAPH_V2="$GRAPH_DIR/function_graph_v2.pt"
PROVENANCE="$GRAPH_DIR/raster_provenance.pt"
GRAPH_V3="$GRAPH_DIR/function_graph_v3.pt"
CANONICAL_TEACHER="$GRAPH_DIR/complete_positive_teacher_48000.pt"
V7_DIR="$ROOT/v7_maps"
V7_16K="$V7_DIR/track_centric_b16000_t06000_strict_global.pt"
V9_DIR="$ROOT/v9_maps"
V10_DIR="$ROOT/v10_pose_maps"
V10_REPORT="$V10_DIR/pose_sufficient_build.json"
METRIC_DIR="$ROOT/metric_16k"
V7_PROVENANCE="$METRIC_DIR/raster_provenance_16k.pt"
V7_TEACHER="$METRIC_DIR/complete_positive_teacher_16k.pt"
METRIC_MAP="$METRIC_DIR/anchor_map_step_0250.pt"
METRIC_STATE="$METRIC_DIR/metric_state_step_0250.pt"
EVAL_ROOT="$ROOT/evaluation"
CONTRACTS="$ROOT/contracts"
LOGS="$ROOT/logs"
MARKER="$ROOT/clean_input_boundary.json"
EVIDENCE_CONTRACT="$CONTRACTS/localization_evidence_graph.json"

export CUDA_HOME=/usr/local/cuda-11.8
export PATH="/root/miniconda3/envs/cybersim_agent/bin:$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="/root/miniconda3/envs/cybersim_agent/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$REPO_ROOT"
export PYTHONHASHSEED=2026
export STDLOC_CAMERA_LOADER_WORKERS=0
mkdir -p "$ROOT" "$CONTRACTS" "$LOGS"
cd "$REPO_ROOT"

for input in "$PLY" "$PRIOR_MANIFEST" "$SP_WEIGHTS" "$MASKS"; do
  [[ -f "$input" ]] || { echo "Missing allowed clean input: $input" >&2; exit 1; }
done

if [[ ! -f "$MARKER" ]]; then
  "$PYTHON" - "$MARKER" "$MODEL_ROOT" "$SOURCE_ROOT" "$SP_WEIGHTS" <<'PY'
import json, sys
from pathlib import Path
marker, model, source, weights = map(Path, sys.argv[1:])
marker.write_text(json.dumps({
    "schema": "lafgs_clean_input_boundary",
    "version": 1,
    "allowed_learned_inputs": {
        "frozen_rgb_only_2dgs": str(model.resolve()),
        "frozen_superpoint": str(weights.resolve()),
    },
    "allowed_raw_inputs": str(source.resolve()),
    "forbidden": "all pre-existing LaFGS localization artifacts",
    "run_type": "full_chain_rebuild",
}, indent=2) + "\n")
PY
fi

contract() {
  "$PYTHON" scripts/lafgs_artifact_contract.py register "$@"
}

base() {
  export LAFGS_SANITIZATION_ROOT="$CLEAN_ROOT"
  export LAFGS_SANITIZATION_MODEL_ROOT="$MODEL_ROOT"
  export LAFGS_SANITIZATION_RUN_TAG="$RUN_TAG"
  export LAFGS_STAGE_A_STEPS=1000
  export LAFGS_SANITIZATION_SOURCE_STEP=1000
  export LAFGS_STATISTICS_CHECKPOINT_STEP=1000
  export LAFGS_GEOMETRY_TEACHER_TRACK_EPIPOLAR_CANDIDATE_TOPK=4
  export LAFGS_GEOMETRY_TEACHER_TRACK_ALLOW_CHAIN_TRACKS=1
  export LAFGS_GEOMETRY_TEACHER_PROVENANCE_GROUP_MAX_LANDMARKS=4
  bash scripts/run_lafgs_v2_rgb_prior_sanitization.sh rgb_2dgs 0 statistics
  for output in "$QUERY_CACHE" "$BASE_STATE" "$TRACK_PAYLOAD"; do
    [[ -f "$output" ]] || { echo "Missing clean base output: $output" >&2; exit 1; }
  done
  if [[ ! -f "$SPARSE_QUERY_CACHE" ]]; then
    "$PYTHON" scripts/slim_lafgs_query_cache.py \
      --input "$QUERY_CACHE" --output "$SPARSE_QUERY_CACHE"
  fi
  if [[ ! -f "$CONTRACTS/query_cache.json" ]]; then
    contract --artifact "$QUERY_CACHE" \
      --manifest "$CONTRACTS/query_cache.json" \
      --kind query_cache --run-type full_chain_rebuild \
      --repo-root "$REPO_ROOT" \
      --parent "rgb_prior=$PLY" --parent "superpoint=$SP_WEIGHTS" \
      --parent "deployment_masks=$MASKS" \
      --config-json '{"resolution":"native","keypoints":2048,"mapping_queries":895}' \
      --query-registry-from "$QUERY_CACHE"
  fi
  if [[ ! -f "$CONTRACTS/track_payload.json" ]]; then
    contract --artifact "$TRACK_PAYLOAD" \
      --manifest "$CONTRACTS/track_payload.json" \
      --kind track_first_payload --run-type full_chain_rebuild \
      --repo-root "$REPO_ROOT" \
      --parent "query_contract=$CONTRACTS/query_cache.json" \
      --parent "stage_a=$BASE_STATE" \
      --config-json '{"identity":"track_first_provenance","group_max":4,"checkpoint":1000}'
  fi
}

graph() {
  base
  mkdir -p "$ROOT/canonical" "$GRAPH_DIR"
  if [[ ! -f "$CANONICAL" ]]; then
    "$PYTHON" scripts/build_lafgs_micro_anchor_bank.py \
      --base_state "$BASE_STATE" --track_payload "$TRACK_PAYLOAD" \
      --query_cache "$SPARSE_QUERY_CACHE" --output "$CANONICAL" --budget 0
  fi
  if [[ ! -f "$GRAPH_V2" ]]; then
    CUDA_VISIBLE_DEVICES=0 "$PYTHON" scripts/build_lafgs_keypoint_function_graph.py \
      --anchor-map "$CANONICAL" --query-cache "$QUERY_CACHE" \
      --deployment-mask-cache "$MASKS" --raster-visibility-cache "$VISIBILITY" \
      --output "$GRAPH_DIR/graph_shard0.pt" --topk 64 --num-shards 2 --shard-index 0 \
      > "$LOGS/graph_shard0.log" 2>&1 &
    pid0=$!
    CUDA_VISIBLE_DEVICES=1 "$PYTHON" scripts/build_lafgs_keypoint_function_graph.py \
      --anchor-map "$CANONICAL" --query-cache "$QUERY_CACHE" \
      --deployment-mask-cache "$MASKS" --raster-visibility-cache "$VISIBILITY" \
      --output "$GRAPH_DIR/graph_shard1.pt" --topk 64 --num-shards 2 --shard-index 1 \
      > "$LOGS/graph_shard1.log" 2>&1 &
    pid1=$!
    wait "$pid0"; wait "$pid1"
    "$PYTHON" scripts/merge_lafgs_keypoint_function_graph.py \
      --inputs "$GRAPH_DIR/graph_shard0.pt" "$GRAPH_DIR/graph_shard1.pt" \
      --output "$GRAPH_V2"
  fi
  if [[ ! -f "$PROVENANCE" ]]; then
    CUDA_VISIBLE_DEVICES=0 "$PYTHON" scripts/build_lafgs_raster_provenance_cache.py \
      --anchor-map "$CANONICAL" --query-cache "$QUERY_CACHE" \
      --gaussian-ply "$PLY" --function-graph "$GRAPH_V2" \
      --track-payload "$TRACK_PAYLOAD" --deployment-mask-cache "$MASKS" \
      --output "$GRAPH_DIR/provenance_shard0.pt" \
      --num-shards 2 --shard-index 0 > "$LOGS/provenance_shard0.log" 2>&1 &
    pid0=$!
    CUDA_VISIBLE_DEVICES=1 "$PYTHON" scripts/build_lafgs_raster_provenance_cache.py \
      --anchor-map "$CANONICAL" --query-cache "$QUERY_CACHE" \
      --gaussian-ply "$PLY" --function-graph "$GRAPH_V2" \
      --track-payload "$TRACK_PAYLOAD" --deployment-mask-cache "$MASKS" \
      --output "$GRAPH_DIR/provenance_shard1.pt" \
      --num-shards 2 --shard-index 1 > "$LOGS/provenance_shard1.log" 2>&1 &
    pid1=$!
    wait "$pid0"; wait "$pid1"
    "$PYTHON" scripts/merge_lafgs_raster_provenance_cache.py \
      --inputs "$GRAPH_DIR/provenance_shard0.pt" "$GRAPH_DIR/provenance_shard1.pt" \
      --output "$PROVENANCE"
  fi
  if [[ ! -f "$GRAPH_V3" ]]; then
    CUDA_VISIBLE_DEVICES=0 "$PYTHON" scripts/build_lafgs_function_graph_v3.py \
      --function-graph-v2 "$GRAPH_V2" --raster-provenance "$PROVENANCE" \
      --output "$GRAPH_V3"
  fi
  if [[ ! -f "$CANONICAL_TEACHER" ]]; then
    CUDA_VISIBLE_DEVICES=1 "$PYTHON" scripts/build_lafgs_v9_complete_positive_teacher.py \
      --anchor-map "$CANONICAL" --query-cache "$QUERY_CACHE" \
      --raster-provenance "$PROVENANCE" --track-payload "$TRACK_PAYLOAD" \
      --output "$CANONICAL_TEACHER"
  fi
  if [[ ! -f "$EVIDENCE_CONTRACT" ]]; then
    "$PYTHON" scripts/build_lafgs_evidence_graph_contract.py \
      --query-cache "$QUERY_CACHE" --track-payload "$TRACK_PAYLOAD" \
      --primitive-prior "$PLY" --anchor-map "$CANONICAL" \
      --function-graph "$GRAPH_V3" --raster-provenance "$PROVENANCE" \
      --positive-teacher "$CANONICAL_TEACHER" \
      --output "$EVIDENCE_CONTRACT"
  else
    "$PYTHON" scripts/build_lafgs_evidence_graph_contract.py \
      --output "$EVIDENCE_CONTRACT" --verify
  fi
  if [[ ! -f "$CONTRACTS/function_graph.json" ]]; then
    contract --artifact "$GRAPH_V3" \
      --manifest "$CONTRACTS/function_graph.json" \
      --kind function_graph_v3 --run-type full_chain_rebuild \
      --repo-root "$REPO_ROOT" \
      --parent "track_contract=$CONTRACTS/track_payload.json" \
      --parent "canonical_map=$CANONICAL" --parent "raster_provenance=$PROVENANCE" \
      --config-json '{"topk":64,"mapping_queries":895}' \
      --anchor-registry-from "$CANONICAL"
  fi
}

maps() {
  graph
  mkdir -p "$V7_DIR" "$V9_DIR"
  if [[ ! -f "$V7_16K" ]]; then
    "$PYTHON" scripts/build_lafgs_v7_track_centric_maps.py \
      --canonical-map "$CANONICAL" --function-graph "$GRAPH_V3" \
      --track-payload "$TRACK_PAYLOAD" --query-cache "$SPARSE_QUERY_CACHE" \
      --output-dir "$V7_DIR" \
      --specs "16000:6000:strict,20000:8400:medium,24000:9900:broad"
  fi
  if [[ ! -f "$V9_DIR/minimum_sufficient_build.json" ]]; then
    "$PYTHON" scripts/build_lafgs_v9_minimum_sufficient_maps.py \
      --canonical-map "$CANONICAL" --function-graph "$GRAPH_V3" \
      --complete-positive-teacher "$CANONICAL_TEACHER" \
      --track-payload "$TRACK_PAYLOAD" --query-cache "$SPARSE_QUERY_CACHE" \
      --output-dir "$V9_DIR" --track-cores "6000:strict,8000:medium" \
      --minimum-rows-per-query 96 --maximum-reserve 16000
  fi
  contract --artifact "$V7_16K" \
    --manifest "$CONTRACTS/v7_16k.json" \
    --kind active_map --run-type full_chain_rebuild \
    --repo-root "$REPO_ROOT" \
    --parent "graph_contract=$CONTRACTS/function_graph.json" \
    --parent "track_contract=$CONTRACTS/track_payload.json" \
    --config-json '{"anchors":16000,"track_core":6000,"tier":"strict"}' \
      --anchor-registry-from "$V7_16K"
}

metric() {
  maps
  mkdir -p "$METRIC_DIR"
  if [[ ! -f "$V7_PROVENANCE" ]]; then
    CUDA_VISIBLE_DEVICES=0 "$PYTHON" scripts/build_lafgs_raster_provenance_cache.py \
      --anchor-map "$V7_16K" --query-cache "$QUERY_CACHE" \
      --gaussian-ply "$PLY" --track-payload "$TRACK_PAYLOAD" \
      --deployment-mask-cache "$MASKS" \
      --output "$METRIC_DIR/provenance_shard0.pt" \
      --num-shards 2 --shard-index 0 > "$LOGS/v7_provenance_shard0.log" 2>&1 &
    pid0=$!
    CUDA_VISIBLE_DEVICES=1 "$PYTHON" scripts/build_lafgs_raster_provenance_cache.py \
      --anchor-map "$V7_16K" --query-cache "$QUERY_CACHE" \
      --gaussian-ply "$PLY" --track-payload "$TRACK_PAYLOAD" \
      --deployment-mask-cache "$MASKS" \
      --output "$METRIC_DIR/provenance_shard1.pt" \
      --num-shards 2 --shard-index 1 > "$LOGS/v7_provenance_shard1.log" 2>&1 &
    pid1=$!
    wait "$pid0"; wait "$pid1"
    "$PYTHON" scripts/merge_lafgs_raster_provenance_cache.py \
      --inputs "$METRIC_DIR/provenance_shard0.pt" "$METRIC_DIR/provenance_shard1.pt" \
      --output "$V7_PROVENANCE"
  fi
  if [[ ! -f "$V7_TEACHER" ]]; then
    CUDA_VISIBLE_DEVICES=0 "$PYTHON" scripts/build_lafgs_v9_complete_positive_teacher.py \
      --anchor-map "$V7_16K" --query-cache "$QUERY_CACHE" \
      --raster-provenance "$V7_PROVENANCE" --track-payload "$TRACK_PAYLOAD" \
      --output "$V7_TEACHER"
  fi
  if [[ ! -f "$METRIC_MAP" || ! -f "$METRIC_STATE" ]]; then
    CUDA_VISIBLE_DEVICES=1 "$PYTHON" scripts/train_lafgs_v7_online_metric.py \
      --map "$V7_16K" --function-graph "$GRAPH_V3" \
      --track-payload "$TRACK_PAYLOAD" --query-cache "$SPARSE_QUERY_CACHE" \
      --complete-positive-teacher "$V7_TEACHER" \
      --output-dir "$METRIC_DIR" --steps 250 \
      --checkpoint-steps "100,175,250" --batch-size 512 --topk 64 \
      --max-positives 8 --rank 16 --metric-residual 0.05 \
      --anchor-residual 0.02 --learning-rate 0.0002 \
      --temperature 0.04 --harmful-weight 0.1 --trust-weight 1 \
      --group-dro-eta 0.03 --refresh-interval 50 \
      --refresh-query-limit 128 --refresh-shards 7 \
      --null-weight 0 --null-threshold 0 --null-minimum-total 0 \
      --training-mode metric_only --metric-only-steps 250 --seed 2026 \
      | tee "$LOGS/metric_16k.log"
  fi
  for output in "$METRIC_MAP" "$METRIC_STATE"; do
    [[ -f "$output" ]] || {
      echo "Missing clean metric output: $output" >&2
      exit 1
    }
  done
  contract --artifact "$METRIC_STATE" \
    --manifest "$CONTRACTS/metric_16k.json" \
    --kind metric_checkpoint --run-type full_chain_rebuild \
    --repo-root "$REPO_ROOT" \
    --parent "v7_contract=$CONTRACTS/v7_16k.json" \
    --parent "positive_teacher=$V7_TEACHER" \
    --config-json '{"mode":"metric_only","steps":250,"seed":2026,"null_weight":0}'
}

eval_one() {
  local label="$1"
  local gpu="$2"
  local map="$3"
  local metric_state="${4:-}"
  local output="$EVAL_ROOT/$label"
  local cfg="$output/config.yaml"
  local frontend="ulfloc_native"
  local metric_args=()
  [[ -n "$metric_state" ]] && {
    frontend="ulfloc_native_metric"
    metric_args=(--metric_state_path "$metric_state")
  }
  if [[ -f "$output/result.path" ]] && [[ -f "$(<"$output/result.path")/results_summary.json" ]]; then
    return
  fi
  mkdir -p "$output/stdloc_results"
  "$PYTHON" scripts/make_stdloc_eval_cfg.py \
    --base_cfg configs/stdloc_cambridge.yaml --output "$cfg" \
    --artifact_model_path "$MODEL_ROOT" \
    --detector_folder ulfloc_native_no_detector --detector_iters 0 \
    --landmark_path "$BOOTSTRAP/sampled_idx.pkl" \
    --landmark_meta_path "$BOOTSTRAP/landmark_meta.pt" \
    --materialized_anchor_map_path "$map" \
    --detect_num 2048 --nms 2 --sparse_ransac_seed 2026 \
    --sparse_query_feature_contract native_resized_input \
    --sparse_frontend "$frontend" "${metric_args[@]}" \
    --reprojection_error 12 --match_threshold 0 --match_topk 1 \
    --max_matches_per_landmark 0 --candidate_frontend_match_policy error \
    --diagnostics --diagnostics_grid_rows 4 --diagnostics_grid_cols 4 \
    --diagnostics_voxel_size 1 \
    --diagnostics_task_translation_scale_m 0.07160573943725686 \
    --diagnostics_task_rotation_scale_degrees 2 > "$output/config_build.json"
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    export STDLOC_RESULTS_ROOT="$output/stdloc_results"
    "$PYTHON" stdloc.py \
      --model_path "$MODEL_ROOT" --source_path "$SOURCE_ROOT" \
      --images processed --data_device cpu --gaussian_type 2dgs \
      --sh_degree 3 --feature_type sp --resolution 1 --longest_edge 0 \
      --norm_before_render --iteration 30000 --cfg "$cfg" \
      --prefix "lafgs-clean-$label" --sparse_only \
      --evaluation_camera_subset test 2>&1 | tee "$output/eval.log"
  )
  local result
  result="$(sed -n 's/^Output path: //p' "$output/eval.log" | tail -n 1)"
  [[ -f "$result/results_summary.json" ]] || {
    echo "Evaluation failed for $label" >&2
    return 1
  }
  printf '%s\n' "$result" > "$output/result.path"
}

eval_all() {
  metric
  mapfile -t compact_maps < <(
    "$PYTHON" - "$V9_DIR/minimum_sufficient_build.json" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1]))
for value in payload["maps"].values():
    print(value["path"])
PY
  )
  eval_one v7_16k 0 "$V7_16K" > "$LOGS/eval_v7_16k.log" 2>&1 &
  pid0=$!
  eval_one metric_16k 1 "$METRIC_MAP" "$METRIC_STATE" \
    > "$LOGS/eval_metric_16k.log" 2>&1 &
  pid1=$!
  wait "$pid0"; wait "$pid1"
  local index=0
  local jobs=()
  for map in "${compact_maps[@]}"; do
    local label
    label="$(basename "${map%.pt}")"
    eval_one "$label" "$((index % 2))" "$map" \
      > "$LOGS/eval_${label}.log" 2>&1 &
    jobs+=("$!")
    index=$((index + 1))
    if (( ${#jobs[@]} == 2 )); then
      wait "${jobs[0]}"; wait "${jobs[1]}"
      jobs=()
    fi
  done
  for pid in "${jobs[@]}"; do wait "$pid"; done
  summarize_results
}

summarize_results() {
  "$PYTHON" - "$EVAL_ROOT" "$ROOT/clean_rebuild_results.json" <<'PY'
import json, sys
from pathlib import Path
import numpy as np
root, output = map(Path, sys.argv[1:])
summary = {"schema": "lafgs_clean_rebuild_results", "run_type": "full_chain_rebuild", "results": {}}
for pointer in sorted(root.glob("*/result.path")):
    result_path = Path(pointer.read_text().strip())
    result = json.loads((result_path / "results_summary.json").read_text())
    per_query = json.loads((result_path / "results.json").read_text())
    translation_errors = np.asarray(
        [float(item["sparse_TE"]) for item in per_query], dtype=np.float64
    )
    sparse = result["sparse"]
    diagnostics = result.get("sparse_diagnostics", {})
    summary["results"][pointer.parent.name] = {
        "result_path": str(result_path),
        "query_count": result["evaluation_camera_count"],
        "median_te_cm": sparse["median_te"],
        "mean_te_cm": float(translation_errors.mean()),
        "p90_te_cm": float(np.percentile(translation_errors, 90)),
        "recall_5cm_percent": 100.0 * sparse["recall_5cm_5d"],
        "raw_gt_precision_2px_percent": 100.0 * diagnostics.get("sparse_diag_all_gt_precision_2px_mean", 0.0),
        "inlier_gt_precision_2px_percent": 100.0 * diagnostics.get("sparse_diag_inlier_gt_precision_2px_mean", 0.0),
        "matching_ms": diagnostics.get("sparse_diag_runtime_matching_ms_mean"),
        "ransac_ms": diagnostics.get("sparse_diag_runtime_ransac_ms_mean"),
        "mean_hypotheses": diagnostics.get("sparse_diag_ransac_actual_hypotheses_mean"),
    }
output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(json.dumps(summary, indent=2, sort_keys=True))
PY
}

eval_v10() {
  build_v10
  mapfile -t pose_maps < <(
    "$PYTHON" - "$V10_REPORT" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1]))
for value in payload["maps"].values():
    print(value["path"])
PY
  )
  local index=0
  local jobs=()
  for map in "${pose_maps[@]}"; do
    local label
    label="v10_$(basename "${map%.pt}")"
    eval_one "$label" "$((index % 2))" "$map" \
      > "$LOGS/eval_${label}.log" 2>&1 &
    jobs+=("$!")
    index=$((index + 1))
    if (( ${#jobs[@]} == 2 )); then
      wait "${jobs[0]}"; wait "${jobs[1]}"
      jobs=()
    fi
  done
  for pid in "${jobs[@]}"; do wait "$pid"; done
  summarize_results
}

build_v10() {
  maps
  if [[ ! -f "$V10_REPORT" ]]; then
    mkdir -p "$V10_DIR"
    "$PYTHON" scripts/build_lafgs_v10_pose_sufficient_maps.py \
      --core-map "$V9_DIR/minimum_sufficient_core06000_strict_qrows096_total09110.pt" \
      --canonical-map "$CANONICAL" --function-graph "$GRAPH_V3" \
      --complete-positive-teacher "$CANONICAL_TEACHER" \
      --track-payload "$TRACK_PAYLOAD" --query-cache "$SPARSE_QUERY_CACHE" \
      --output-dir "$V10_DIR" --reserve-additions "1000,2000,3000"
  fi
}

case "$MODE" in
  base) base ;;
  graph) graph ;;
  maps) maps ;;
  metric) metric ;;
  eval) eval_all ;;
  v10) eval_v10 ;;
  v10eval) eval_v10 ;;
  all_core) eval_all ;;
  all_pose_reserve) eval_v10 ;;
  all)
    eval_all
    eval_v10
    ;;
esac
