#!/usr/bin/env bash
set -euo pipefail

# Frozen LaFGS-v1.0 Cambridge protocol. Each scene is reconstructed from its
# external RGB-only MAtCha 2DGS and is evaluated with the same hyperparameters.

if [[ $# -ne 3 ]]; then
  echo "Usage: bash $0 <scene> <gpu> <prepare|base|graph|map|reconstruct|family|selector|eval|all>" >&2
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
  0|1|2) ;;
  *) echo "GPU must be 0, 1, or 2" >&2; exit 2 ;;
esac
case "$MODE" in
  prepare|base|graph|map|reconstruct|family|selector|eval|all) ;;
  *) echo "Unsupported mode: $MODE" >&2; exit 2 ;;
esac

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/root/miniconda3/envs/cybersim_agent/bin/python}"
DATA_ROOT="${CAMBRIDGE_DATA_ROOT:-/mnt/pool/sqy/Cambridge_stdloc}"
MATCHA_ROOT="${CAMBRIDGE_MATCHA_2DGS_ROOT:-/root/MAtCha/output_cambridge_full_retained_v2}"
EXPERIMENT_ROOT="${LAFGS_V1_MULTISCENE_ROOT:-/mnt/pool/sqy/stdloc_lafgs_v1_frozen_multiscene_20260731}"
PRIOR_PROFILE="${LAFGS_V1_PRIOR_PROFILE_OVERRIDE:-rgb_2dgs}"
case "$PRIOR_PROFILE" in
  rgb_2dgs) GAUSSIAN_TYPE="2dgs"; SH_DEGREE=3 ;;
  rgb_nosky|rgb_sky_dirty) GAUSSIAN_TYPE="3dgs"; SH_DEGREE=0 ;;
  vanilla_2dgs) GAUSSIAN_TYPE="2dgs"; SH_DEGREE=3 ;;
  vanilla_3dgs) GAUSSIAN_TYPE="3dgs"; SH_DEGREE=3 ;;
  anysplat_ff) GAUSSIAN_TYPE="3dgs"; SH_DEGREE=0 ;;
  *) echo "Unsupported frozen prior profile: $PRIOR_PROFILE" >&2; exit 2 ;;
esac
if [[ "$GAUSSIAN_TYPE" == "2dgs" ]]; then
  GEOMETRY_TEACHER_IDENTITY_MODE="track_first_provenance"
  GEOMETRY_TEACHER_TAG="g3_track_provenance_v1"
else
  # The in-training composition teacher is 2DGS-specific. 3DGS source
  # lineage is attached later by the generic raster-provenance stage.
  GEOMETRY_TEACHER_IDENTITY_MODE="track_first"
  GEOMETRY_TEACHER_TAG="g2_track_first_v1"
fi
CONFIG="${LAFGS_V1_CONFIG_OVERRIDE:-$REPO_ROOT/configs/lafgs_v1_frozen_cambridge.yaml}"
ROOT="$EXPERIMENT_ROOT/$SCENE"
MODEL_ROOT="${LAFGS_V1_MODEL_ROOT_OVERRIDE:-$ROOT/prior/rgb_matcha_2dgs}"
EXTERNAL_PRIOR=0
if [[ -n "${LAFGS_V1_MODEL_ROOT_OVERRIDE:-}" ]]; then
  EXTERNAL_PRIOR=1
fi
SOURCE_ROOT="$DATA_ROOT/$SCENE"
RUN_TAG="frozen_v1"
RUN_ROOT="$ROOT/runs/$RUN_TAG"
LOGS="$ROOT/logs"
CONTRACTS="$ROOT/contracts"
EVAL_ROOT="$ROOT/evaluation"

case "$SCENE" in
  GreatCourt) MATCHA_RUN="GreatCourt_n20_long_masked_retrain_retry" ;;
  KingsCollege) MATCHA_RUN="KingsCollege_n20_long_masked_retrain" ;;
  OldHospital) MATCHA_RUN="OldHospital_n20_long_masked_retrain_retry" ;;
  ShopFacade) MATCHA_RUN="ShopFacade_n20_long_masked_retrain" ;;
  StMarysChurch) MATCHA_RUN="StMarysChurch_n20_long_masked_retrain" ;;
esac
SOURCE_PLY="$MATCHA_ROOT/$MATCHA_RUN/free_gaussians/point_cloud/iteration_30000/point_cloud.ply"
PLY="$MODEL_ROOT/point_cloud/iteration_30000/point_cloud.ply"
PRIOR_MANIFEST="$MODEL_ROOT/rgb_prior_manifest.json"
MASKS="$SOURCE_ROOT/processed/masks.pkl"
SP_WEIGHTS="$REPO_ROOT/encoders/sp_encoder/weights/superpoint_v1.pth"

eval "$("$PYTHON" - "$CONFIG" <<'PY'
import shlex
import sys
import yaml

cfg = yaml.safe_load(open(sys.argv[1]))
values = {
    "SCAFFOLD_BUDGET": cfg["initialization"]["scaffold_budget"],
    "STAGE_A_STEPS": cfg["reconstruction"]["stage_a_steps"],
    "TRACK_CORE": cfg["reconstruction"]["track_core"],
    "MINIMUM_ROWS": cfg["reconstruction"]["minimum_rows_per_mapping_query"],
    "MAXIMUM_RESERVE": cfg["reconstruction"]["maximum_reserve"],
    "POSE_RESERVE": cfg["reconstruction"]["pose_reserve_additions"],
    "METRIC_STEPS": cfg["reconstruction"]["metric_steps"],
    "FAMILY_MAX_MODES": cfg["family"]["maximum_modes_per_anchor"],
    "FAMILY_MIN_OBS": cfg["family"]["minimum_observations"],
    "FAMILY_MIN_TRAJECTORIES": cfg["family"]["minimum_trajectories"],
    "FAMILY_MIN_VIEW_BINS": cfg["family"]["minimum_view_bins"],
    "FAMILY_MIN_SEPARATION": cfg["family"]["minimum_separation"],
    "FAMILY_MAX_PRIMARY_SIMILARITY": cfg["family"]["maximum_primary_similarity"],
    "FAMILY_TRIM": cfg["family"]["trim_fraction"],
    "FAMILY_MIN_LEGAL": cfg["family"]["minimum_legal_activations"],
    "FAMILY_MIN_PRECISION": cfg["family"]["minimum_activation_precision"],
    "FAMILY_FALSE_COST": cfg["family"]["false_activation_cost"],
    "FAMILY_MAX_SELECTED": cfg["family"]["maximum_selected"],
    "FAMILY_BIASES": cfg["family"]["biases"],
    "SELECTOR_TOPK": cfg["selector"]["retrieval_topk"],
    "SELECTOR_CORE": cfg["selector"]["core_budget"],
    "SELECTOR_BUDGET": cfg["selector"]["fixed_budget"],
    "SELECTOR_OOF_FOLDS": cfg["selector"]["single_trajectory_oof_folds"],
}
for key, value in values.items():
    print(f"{key}={shlex.quote(str(value))}")
print("FROZEN_SEEDS=" + shlex.quote(" ".join(map(str, cfg["seeds"]))))
PY
)"

if [[ -n "${LAFGS_FROZEN_SEEDS_OVERRIDE:-}" ]]; then
  FROZEN_SEEDS="$LAFGS_FROZEN_SEEDS_OVERRIDE"
fi
EVAL_VARIANTS="${LAFGS_EVAL_VARIANTS_OVERRIDE:-A0_bootstrap A1_reconstructed}"
EVAL_SKIP_SUMMARY="${LAFGS_EVAL_SKIP_SUMMARY:-0}"

BOOTSTRAP="$RUN_ROOT/bootstrap"
STAGE_A="$RUN_ROOT/stage_a_combined_${STAGE_A_STEPS}"
STATISTICS="$RUN_ROOT/statistics_combined_${STAGE_A_STEPS}_frozen_${GEOMETRY_TEACHER_TAG}"
QUERY_CACHE="${LAFGS_V1_QUERY_CACHE_OVERRIDE:-$RUN_ROOT/query_cache_native_fullres_k2048.pt}"
SPARSE_QUERY_CACHE="${LAFGS_V1_SPARSE_QUERY_CACHE_OVERRIDE:-$RUN_ROOT/query_cache_native_sparse_teacher.pt}"
VISIBILITY="$RUN_ROOT/visibility_${SCAFFOLD_BUDGET}_native.pt"
BASE_STATE="$STAGE_A/${STAGE_A_STEPS}_lafgs_map_state.pt"
BOOTSTRAP_STATE="$BOOTSTRAP/0_lafgs_map_state.pt"
TRACK_PAYLOAD="$STATISTICS/track_micro_anchor_payload.pt"

CANONICAL_DIR="$ROOT/canonical"
CANONICAL="$CANONICAL_DIR/canonical_${SCAFFOLD_BUDGET}.pt"
GRAPH_DIR="$ROOT/function_graph"
GRAPH_V2="$GRAPH_DIR/function_graph_v2.pt"
RASTER_PROVENANCE="$GRAPH_DIR/raster_provenance.pt"
GRAPH_V3="$GRAPH_DIR/function_graph_v3.pt"
CANONICAL_TEACHER="$GRAPH_DIR/complete_positive_teacher_${SCAFFOLD_BUDGET}.pt"
EVIDENCE_CONTRACT="$CONTRACTS/localization_evidence_graph.json"

MAP_DIR="$ROOT/compact_map"
MAP_REPORT="$MAP_DIR/minimum_sufficient_build.json"
RESERVE_DIR="$ROOT/pose_reserve"
RESERVE_REPORT="$RESERVE_DIR/pose_sufficient_build.json"
ADD_MAP="$RESERVE_DIR/pose_sufficient_add$(printf '%04d' "$POSE_RESERVE").pt"

RECON_DIR="$ROOT/self_localization_reconstruction"
RECON_PROVENANCE="$RECON_DIR/raster_provenance.pt"
RECON_TEACHER="$RECON_DIR/complete_positive_teacher.pt"
RECON_MAP="$RECON_DIR/anchor_map_step_$(printf '%04d' "$METRIC_STEPS").pt"
METRIC_STATE="$RECON_DIR/metric_state_step_$(printf '%04d' "$METRIC_STEPS").pt"

FAMILY_DIR="$ROOT/family_refinement"
BASE_REPLAY="$FAMILY_DIR/replay_base.json"
BASE_DYNAMIC="$FAMILY_DIR/dynamic_base.pt"
FAMILY_POOL="$FAMILY_DIR/appearance_pool.pt"
FAMILY_STATE="$FAMILY_DIR/calibrated_family.pt"
FAMILY_REPLAY="$FAMILY_DIR/replay_family.json"
FAMILY_DYNAMIC="$FAMILY_DIR/dynamic_family.pt"

SELECTOR_DIR="$ROOT/pose_sufficient_selector"
TOPK_OUTCOMES="$SELECTOR_DIR/topk${SELECTOR_TOPK}_family.pt"
SELECTOR_STATE="$SELECTOR_DIR/selector_model_fixed0512.pt"

export CUDA_HOME=/usr/local/cuda-11.8
export PATH="/root/miniconda3/envs/cybersim_agent/bin:$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="/root/miniconda3/envs/cybersim_agent/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$REPO_ROOT"
export PYTHONHASHSEED=2026
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export STDLOC_CAMERA_LOADER_WORKERS=0
mkdir -p "$ROOT" "$LOGS" "$CONTRACTS" "$EVAL_ROOT"
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

json_map_path() {
  local report="$1"
  local prefix="$2"
  "$PYTHON" - "$report" "$prefix" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1]))
matches = [
    value["path"]
    for key, value in payload["maps"].items()
    if key.startswith(sys.argv[2])
]
if len(matches) != 1:
    raise SystemExit(f"expected one {sys.argv[2]!r} map, found {matches}")
print(matches[0])
PY
}

register_contract() {
  local artifact="$1"
  local manifest="$2"
  local kind="$3"
  shift 3
  if [[ -f "$manifest" ]]; then
    "$PYTHON" scripts/lafgs_artifact_contract.py verify \
      --manifest "$manifest"
    return
  fi
  "$PYTHON" scripts/lafgs_artifact_contract.py register \
    --artifact "$artifact" --manifest "$manifest" \
    --kind "$kind" --run-type frozen_multiscene \
    --repo-root "$REPO_ROOT" "$@"
}

prepare() {
  require_file "$CONFIG"
  require_file "$MASKS"
  require_file "$SP_WEIGHTS"
  mkdir -p "$ROOT/prior" "$ROOT/audit"
  if [[ "$EXTERNAL_PRIOR" == "0" ]]; then
    require_file "$SOURCE_PLY"
  fi
  if [[ "$EXTERNAL_PRIOR" == "0" && ! -f "$ROOT/audit/matcha_protocol.json" ]]; then
    run_logged matcha_protocol_audit \
      "$PYTHON" scripts/audit_cambridge_matcha_2dgs_protocol.py \
      --runs_root "$MATCHA_ROOT" --data_root "$DATA_ROOT" \
      --scenes "$SCENE" \
      --output_json "$ROOT/audit/matcha_protocol.json" \
      --output_markdown "$ROOT/audit/matcha_protocol.md"
  fi
  if [[ "$EXTERNAL_PRIOR" == "0" && ! -f "$PRIOR_MANIFEST" ]]; then
    run_logged export_rgb_prior \
      "$PYTHON" scripts/export_rgb_gaussian_prior.py \
      --input_ply "$SOURCE_PLY" --output_model "$MODEL_ROOT" \
      --gaussian_type "$GAUSSIAN_TYPE" --sh_degree "$SH_DEGREE" \
      --source_path "$SOURCE_ROOT" --images processed \
      --longest_edge 0 --iteration 30000 --prior_kind rgb_only
  fi
  require_file "$PLY"
  require_file "$PRIOR_MANIFEST"
  register_contract "$PLY" "$CONTRACTS/rgb_prior.json" "rgb_only_$GAUSSIAN_TYPE" \
    --parent "source_prior_manifest=$PRIOR_MANIFEST" \
    --config-json "{\"scene\":\"$SCENE\",\"iteration\":30000,\"profile\":\"$PRIOR_PROFILE\"}"
}

base() {
  prepare
  if [[ ! -f "$TRACK_PAYLOAD" ]]; then
    export LAFGS_CAMBRIDGE_SCENE="$SCENE"
    export LAFGS_SANITIZATION_ROOT="$EXPERIMENT_ROOT"
    export LAFGS_SANITIZATION_MODEL_ROOT="$MODEL_ROOT"
    export LAFGS_SANITIZATION_RUN_TAG="$RUN_TAG"
    export LAFGS_QUERY_CACHE_PATH="$QUERY_CACHE"
    export LAFGS_SANITIZATION_SCAFFOLD_BUDGET="$SCAFFOLD_BUDGET"
    export LAFGS_STAGE_A_STEPS="$STAGE_A_STEPS"
    export LAFGS_SANITIZATION_SOURCE_STEP="$STAGE_A_STEPS"
    export LAFGS_STATISTICS_CHECKPOINT_STEP="$STAGE_A_STEPS"
    export LAFGS_GEOMETRY_TEACHER_IDENTITY_MODE="$GEOMETRY_TEACHER_IDENTITY_MODE"
    export LAFGS_GEOMETRY_TEACHER_TAG="$GEOMETRY_TEACHER_TAG"
    export LAFGS_GEOMETRY_TEACHER_TRACK_EPIPOLAR_CANDIDATE_TOPK=4
    export LAFGS_GEOMETRY_TEACHER_TRACK_ALLOW_CHAIN_TRACKS=1
    export LAFGS_GEOMETRY_TEACHER_PROVENANCE_GROUP_MAX_LANDMARKS=4
    bash scripts/run_lafgs_v2_rgb_prior_sanitization.sh \
      "$PRIOR_PROFILE" "$GPU" statistics
  fi
  if [[ ! -f "$SPARSE_QUERY_CACHE" ]]; then
    run_logged slim_query_cache \
      "$PYTHON" scripts/slim_lafgs_query_cache.py \
      --input "$QUERY_CACHE" --output "$SPARSE_QUERY_CACHE"
  fi
  for path in \
    "$QUERY_CACHE" "$SPARSE_QUERY_CACHE" "$BOOTSTRAP_STATE" \
    "$BASE_STATE" "$TRACK_PAYLOAD"; do
    require_file "$path"
  done
  # Statistics are materialized before their fail-closed semantic audit. Keep
  # the audit outside the creation branch so an interrupted run cannot bypass
  # it merely because the payload already exists.
  "$PYTHON" scripts/verify_geometry_teacher_statistics.py \
    --statistics "$TRACK_PAYLOAD" \
    --expected_identity "$GEOMETRY_TEACHER_IDENTITY_MODE"
  register_contract "$QUERY_CACHE" "$CONTRACTS/query_cache.json" query_cache \
    --parent "rgb_prior=$CONTRACTS/rgb_prior.json" \
    --parent "superpoint=$SP_WEIGHTS" \
    --parent "deployment_masks=$MASKS" \
    --config-json '{"resolution":"native","keypoints":2048,"split":"all_mapping"}' \
    --query-registry-from "$QUERY_CACHE"
  register_contract "$TRACK_PAYLOAD" "$CONTRACTS/track_payload.json" \
    track_first_payload \
    --parent "query_cache=$CONTRACTS/query_cache.json" \
    --parent "stage_a=$BASE_STATE" \
    --config-json "{\"checkpoint\":$STAGE_A_STEPS,\"identity\":\"$GEOMETRY_TEACHER_IDENTITY_MODE\"}"
}

graph() {
  base
  mkdir -p "$CANONICAL_DIR" "$GRAPH_DIR"
  if [[ ! -f "$CANONICAL" ]]; then
    run_logged canonical_map \
      "$PYTHON" scripts/build_lafgs_micro_anchor_bank.py \
      --base_state "$BASE_STATE" --track_payload "$TRACK_PAYLOAD" \
      --query_cache "$SPARSE_QUERY_CACHE" --output "$CANONICAL" --budget 0
  fi
  if [[ ! -f "$GRAPH_V2" ]]; then
    run_logged function_graph_v2 \
      env CUDA_VISIBLE_DEVICES="$GPU" \
      "$PYTHON" scripts/build_lafgs_keypoint_function_graph.py \
      --anchor-map "$CANONICAL" --query-cache "$QUERY_CACHE" \
      --deployment-mask-cache "$MASKS" \
      --raster-visibility-cache "$VISIBILITY" \
      --output "$GRAPH_V2" --topk 64
  fi
  if [[ ! -f "$RASTER_PROVENANCE" ]]; then
    run_logged raster_provenance \
      env CUDA_VISIBLE_DEVICES="$GPU" \
      "$PYTHON" scripts/build_lafgs_raster_provenance_cache.py \
      --anchor-map "$CANONICAL" --query-cache "$QUERY_CACHE" \
      --gaussian-ply "$PLY" --gaussian-type "$GAUSSIAN_TYPE" \
      --sh-degree "$SH_DEGREE" --function-graph "$GRAPH_V2" \
      --track-payload "$TRACK_PAYLOAD" \
      --deployment-mask-cache "$MASKS" \
      --output "$RASTER_PROVENANCE"
  fi
  if [[ ! -f "$GRAPH_V3" ]]; then
    run_logged function_graph_v3 \
      env CUDA_VISIBLE_DEVICES="$GPU" \
      "$PYTHON" scripts/build_lafgs_function_graph_v3.py \
      --function-graph-v2 "$GRAPH_V2" \
      --raster-provenance "$RASTER_PROVENANCE" --output "$GRAPH_V3"
  fi
  if [[ ! -f "$CANONICAL_TEACHER" ]]; then
    run_logged canonical_positive_teacher \
      env CUDA_VISIBLE_DEVICES="$GPU" \
      "$PYTHON" scripts/build_lafgs_v9_complete_positive_teacher.py \
      --anchor-map "$CANONICAL" --query-cache "$QUERY_CACHE" \
      --raster-provenance "$RASTER_PROVENANCE" \
      --track-payload "$TRACK_PAYLOAD" --output "$CANONICAL_TEACHER"
  fi
  if [[ ! -f "$EVIDENCE_CONTRACT" ]]; then
    run_logged evidence_contract \
      "$PYTHON" scripts/build_lafgs_evidence_graph_contract.py \
      --query-cache "$QUERY_CACHE" --track-payload "$TRACK_PAYLOAD" \
      --primitive-prior "$PLY" --anchor-map "$CANONICAL" \
      --function-graph "$GRAPH_V3" \
      --raster-provenance "$RASTER_PROVENANCE" \
      --positive-teacher "$CANONICAL_TEACHER" \
      --output "$EVIDENCE_CONTRACT"
  else
    "$PYTHON" scripts/build_lafgs_evidence_graph_contract.py \
      --output "$EVIDENCE_CONTRACT" --verify
  fi
}

compact_map() {
  graph
  mkdir -p "$MAP_DIR" "$RESERVE_DIR"
  if [[ ! -f "$MAP_REPORT" ]]; then
    run_logged compact_map \
      "$PYTHON" scripts/build_lafgs_v9_minimum_sufficient_maps.py \
      --canonical-map "$CANONICAL" --function-graph "$GRAPH_V3" \
      --complete-positive-teacher "$CANONICAL_TEACHER" \
      --track-payload "$TRACK_PAYLOAD" \
      --query-cache "$SPARSE_QUERY_CACHE" \
      --output-dir "$MAP_DIR" --track-cores "${TRACK_CORE}:medium" \
      --minimum-rows-per-query "$MINIMUM_ROWS" \
      --maximum-reserve "$MAXIMUM_RESERVE"
  fi
  local core_map
  core_map="$(json_map_path "$MAP_REPORT" "core$(printf '%05d' "$TRACK_CORE")_medium")"
  require_file "$core_map"
  if [[ ! -f "$RESERVE_REPORT" ]]; then
    run_logged pose_reserve \
      "$PYTHON" scripts/build_lafgs_v10_pose_sufficient_maps.py \
      --core-map "$core_map" --canonical-map "$CANONICAL" \
      --function-graph "$GRAPH_V3" \
      --complete-positive-teacher "$CANONICAL_TEACHER" \
      --track-payload "$TRACK_PAYLOAD" \
      --query-cache "$SPARSE_QUERY_CACHE" \
      --output-dir "$RESERVE_DIR" \
      --reserve-additions "$POSE_RESERVE"
  fi
  require_file "$ADD_MAP"
}

self_localization_reconstruction() {
  compact_map
  mkdir -p "$RECON_DIR"
  if [[ ! -f "$RECON_PROVENANCE" ]]; then
    run_logged reconstruction_provenance \
      env CUDA_VISIBLE_DEVICES="$GPU" \
      "$PYTHON" scripts/build_lafgs_raster_provenance_cache.py \
      --anchor-map "$ADD_MAP" --query-cache "$QUERY_CACHE" \
      --gaussian-ply "$PLY" --gaussian-type "$GAUSSIAN_TYPE" \
      --sh-degree "$SH_DEGREE" --track-payload "$TRACK_PAYLOAD" \
      --deployment-mask-cache "$MASKS" --output "$RECON_PROVENANCE"
  fi
  if [[ ! -f "$RECON_TEACHER" ]]; then
    run_logged reconstruction_teacher \
      env CUDA_VISIBLE_DEVICES="$GPU" \
      "$PYTHON" scripts/build_lafgs_v9_complete_positive_teacher.py \
      --anchor-map "$ADD_MAP" --query-cache "$QUERY_CACHE" \
      --raster-provenance "$RECON_PROVENANCE" \
      --track-payload "$TRACK_PAYLOAD" --output "$RECON_TEACHER"
  fi
  if [[ ! -f "$RECON_MAP" || ! -f "$METRIC_STATE" ]]; then
    run_logged self_localization_reconstruction \
      env CUDA_VISIBLE_DEVICES="$GPU" \
      "$PYTHON" scripts/train_lafgs_v7_online_metric.py \
      --map "$ADD_MAP" --function-graph "$GRAPH_V3" \
      --track-payload "$TRACK_PAYLOAD" \
      --query-cache "$SPARSE_QUERY_CACHE" \
      --complete-positive-teacher "$RECON_TEACHER" \
      --output-dir "$RECON_DIR" --steps "$METRIC_STEPS" \
      --checkpoint-steps "$METRIC_STEPS" \
      --batch-size 512 --topk 64 --max-positives 8 \
      --rank 16 --metric-residual 0.05 --anchor-residual 0.02 \
      --learning-rate 0.0002 --temperature 0.04 \
      --harmful-weight 0.1 --trust-weight 1 \
      --group-dro-eta 0.03 --refresh-interval 0 \
      --refresh-query-limit 128 --refresh-shards 7 \
      --null-weight 0 --null-threshold 0 --null-minimum-total 0 \
      --training-mode metric_only --metric-only-steps "$METRIC_STEPS" \
      --seed 2026
  fi
  require_file "$RECON_MAP"
  require_file "$METRIC_STATE"
  register_contract "$RECON_MAP" "$CONTRACTS/reconstructed_map.json" \
    localization_anchor_map \
    --parent "track_payload=$CONTRACTS/track_payload.json" \
    --parent "positive_teacher=$RECON_TEACHER" \
    --config-json "{\"track_core\":$TRACK_CORE,\"reserve\":$POSE_RESERVE,\"steps\":$METRIC_STEPS}" \
    --anchor-registry-from "$RECON_MAP"
}

family_refinement() {
  self_localization_reconstruction
  mkdir -p "$FAMILY_DIR"
  if [[ ! -f "$BASE_DYNAMIC" ]]; then
    run_logged family_base_replay \
      env CUDA_VISIBLE_DEVICES="$GPU" \
      "$PYTHON" scripts/evaluate_lafgs_map_on_query_cache.py \
      --map "$RECON_MAP" --metric-state "$METRIC_STATE" \
      --query-cache "$SPARSE_QUERY_CACHE" \
      --function-graph "$GRAPH_V3" --output "$BASE_REPLAY" \
      --dynamic-outcomes-output "$BASE_DYNAMIC" \
      --reprojection-error 12 --seed 2026 --split mapping_replay
  fi
  if [[ ! -f "$FAMILY_POOL" ]]; then
    run_logged family_pool \
      env CUDA_VISIBLE_DEVICES="$GPU" OMP_NUM_THREADS=4 \
      "$PYTHON" scripts/build_lafgs_basin_family_prototypes.py \
      --map "$RECON_MAP" --output "$FAMILY_POOL" \
      --mode appearance_pool \
      --complete-positive-teacher "$RECON_TEACHER" \
      --dynamic-outcomes "$BASE_DYNAMIC" \
      --query-cache "$SPARSE_QUERY_CACHE" \
      --track-payload "$TRACK_PAYLOAD" \
      --metric-state "$METRIC_STATE" --device cuda \
      --maximum-modes-per-anchor "$FAMILY_MAX_MODES" \
      --minimum-observations "$FAMILY_MIN_OBS" \
      --minimum-trajectories "$FAMILY_MIN_TRAJECTORIES" \
      --minimum-view-bins "$FAMILY_MIN_VIEW_BINS" \
      --minimum-separation "$FAMILY_MIN_SEPARATION" \
      --maximum-primary-similarity "$FAMILY_MAX_PRIMARY_SIMILARITY" \
      --trim-fraction "$FAMILY_TRIM" --initial-bias -0.01
  fi
  if [[ ! -f "$FAMILY_STATE" ]]; then
    run_logged family_calibration \
      env CUDA_VISIBLE_DEVICES="$GPU" OMP_NUM_THREADS=4 \
      "$PYTHON" scripts/calibrate_lafgs_appearance_modes.py \
      --mode-pool "$FAMILY_POOL" --metric-state "$METRIC_STATE" \
      --query-cache "$SPARSE_QUERY_CACHE" \
      --complete-positive-teacher "$RECON_TEACHER" \
      --dynamic-outcomes "$BASE_DYNAMIC" --output "$FAMILY_STATE" \
      --device cuda --biases "$FAMILY_BIASES" \
      --minimum-legal-activations "$FAMILY_MIN_LEGAL" \
      --minimum-activation-precision "$FAMILY_MIN_PRECISION" \
      --false-activation-cost "$FAMILY_FALSE_COST" \
      --maximum-selected "$FAMILY_MAX_SELECTED"
  fi
  if [[ ! -f "$FAMILY_DYNAMIC" ]]; then
    run_logged family_replay \
      env CUDA_VISIBLE_DEVICES="$GPU" \
      "$PYTHON" scripts/evaluate_lafgs_map_on_query_cache.py \
      --map "$RECON_MAP" --metric-state "$METRIC_STATE" \
      --family-prototype-state "$FAMILY_STATE" \
      --query-cache "$SPARSE_QUERY_CACHE" \
      --function-graph "$GRAPH_V3" --output "$FAMILY_REPLAY" \
      --dynamic-outcomes-output "$FAMILY_DYNAMIC" \
      --reprojection-error 12 --seed 2026 --split mapping_replay
  fi
  register_contract "$FAMILY_STATE" "$CONTRACTS/family_state.json" \
    family_descriptor_modes \
    --parent "map=$CONTRACTS/reconstructed_map.json" \
    --parent "dynamic_outcomes=$BASE_DYNAMIC" \
    --config-json "{\"minimum_precision\":$FAMILY_MIN_PRECISION,\"maximum_selected\":$FAMILY_MAX_SELECTED}"
}

selector() {
  family_refinement
  mkdir -p "$SELECTOR_DIR"
  if [[ ! -f "$TOPK_OUTCOMES" ]]; then
    run_logged selector_topk \
      env CUDA_VISIBLE_DEVICES="$GPU" \
      "$PYTHON" scripts/build_lafgs_topk_outcomes.py \
      --map "$RECON_MAP" --metric-state "$METRIC_STATE" \
      --family-prototype-state "$FAMILY_STATE" \
      --query-cache "$SPARSE_QUERY_CACHE" \
      --function-graph "$GRAPH_V3" --output "$TOPK_OUTCOMES" \
      --topk "$SELECTOR_TOPK" --device cuda
  fi
  if [[ ! -f "$SELECTOR_STATE" ]]; then
    run_logged selector_train \
      "$PYTHON" scripts/train_lafgs_basis_core_reserve_selector.py \
      --map "$RECON_MAP" --query-cache "$SPARSE_QUERY_CACHE" \
      --topk-outcomes "$TOPK_OUTCOMES" \
      --dynamic-outcomes "$FAMILY_DYNAMIC" \
      --output-dir "$SELECTOR_DIR" \
      --maximum-fit-rows 500000 --core-budget "$SELECTOR_CORE" \
      --minimum-budget "$SELECTOR_BUDGET" \
      --maximum-budget "$SELECTOR_BUDGET" \
      --single-trajectory-fold-count "$SELECTOR_OOF_FOLDS" \
      --minimum-strict-lcb 80 --minimum-dependency-groups 96 \
      --minimum-image-cells 16 --minimum-log-expected-basis 11 \
      --representative-count 64 --pair-count 256 --seed 2026
  fi
  require_file "$SELECTOR_STATE"
  register_contract "$SELECTOR_STATE" "$CONTRACTS/selector_state.json" \
    pose_sufficient_selector \
    --parent "map=$CONTRACTS/reconstructed_map.json" \
    --parent "family=$CONTRACTS/family_state.json" \
    --parent "outcomes=$FAMILY_DYNAMIC" \
    --config-json "{\"retrieval_topk\":$SELECTOR_TOPK,\"budget\":$SELECTOR_BUDGET,\"single_trajectory_oof_folds\":$SELECTOR_OOF_FOLDS,\"seed\":2026}"
}

deployment_eval_prerequisites() {
  # A0/A1 inference consumes only the frozen RGB prior and materialized
  # localization maps.  Do not make deployment evaluation depend on mutable
  # offline query caches that are not read by stdloc.py.
  prepare
  for path in \
    "$BOOTSTRAP_STATE" "$BOOTSTRAP/sampled_idx.pkl" \
    "$BOOTSTRAP/landmark_meta.pt" "$RECON_MAP" "$METRIC_STATE" \
    "$CONTRACTS/rgb_prior.json" "$CONTRACTS/reconstructed_map.json"; do
    require_file "$path"
  done
  "$PYTHON" scripts/lafgs_artifact_contract.py verify \
    --manifest "$CONTRACTS/rgb_prior.json"
  "$PYTHON" scripts/lafgs_artifact_contract.py verify \
    --manifest "$CONTRACTS/reconstructed_map.json"
}

eval_one() {
  local label="$1"
  local seed="$2"
  local output="$EVAL_ROOT/$label/seed${seed}"
  local cfg="$output/config.yaml"
  shift 2
  if [[ -f "$output/result.path" ]] && \
     [[ -f "$(<"$output/result.path")/results_summary.json" ]]; then
    return
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
    "$@" > "$output/config_build.json"
  (
    export CUDA_VISIBLE_DEVICES="$GPU"
    export STDLOC_RESULTS_ROOT="$output/results"
    "$PYTHON" stdloc.py \
      --model_path "$MODEL_ROOT" --source_path "$SOURCE_ROOT" \
      --images processed --data_device cpu --gaussian_type "$GAUSSIAN_TYPE" \
      --sh_degree "$SH_DEGREE" --feature_type sp --resolution 1 --longest_edge 0 \
      --norm_before_render --iteration 30000 --cfg "$cfg" \
      --prefix "lafgs-v1-$SCENE-$label-seed$seed" \
      --sparse_only --evaluation_camera_subset test \
      2>&1 | tee "$output/eval.log"
  )
  local result
  result="$(sed -n 's/^Output path: //p' "$output/eval.log" | tail -n 1)"
  [[ -n "$result" && -f "$result/results_summary.json" ]] || {
    echo "Evaluation failed for $SCENE/$label/seed$seed" >&2
    exit 1
  }
  printf '%s\n' "$result" > "$output/result.path"
}

summarize() {
  "$PYTHON" - "$ROOT" "$CONFIG" "$METRIC_STEPS" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

root = Path(sys.argv[1])
config_path = Path(sys.argv[2])
metric_steps = int(sys.argv[3])
config = yaml.safe_load(config_path.read_text())
summary = {
    "schema": "lafgs_v1_frozen_multiscene_result",
    "scene": root.name,
    "canonical_config": str(config_path.resolve()),
    "canonical_config_sha256": hashlib.sha256(
        config_path.read_bytes()
    ).hexdigest(),
    "results": {},
}
for pointer in sorted((root / "evaluation").glob("*/seed*/result.path")):
    result_dir = Path(pointer.read_text().strip())
    report = json.loads((result_dir / "results_summary.json").read_text())
    per_query = json.loads((result_dir / "results.json").read_text())
    te = np.asarray([float(row["sparse_TE"]) for row in per_query])
    ae = np.asarray([float(row["sparse_AE"]) for row in per_query])
    sparse = report["sparse"]
    diagnostics = report.get("sparse_diagnostics", {})
    label = pointer.parents[1].name
    seed = int(pointer.parent.name.removeprefix("seed"))
    summary["results"].setdefault(label, {})[str(seed)] = {
        "query_count": int(report["evaluation_camera_count"]),
        "median_te_cm": float(np.median(te)),
        "mean_te_cm": float(te.mean()),
        "p90_te_cm": float(np.percentile(te, 90)),
        "median_ae_deg": float(np.median(ae)),
        "mean_ae_deg": float(ae.mean()),
        "recall_2cm_2deg_percent": 100.0 * float(
            sparse["recall_2cm_2d"]
        ),
        "recall_5cm_5deg_percent": 100.0 * float(
            sparse["recall_5cm_5d"]
        ),
        "raw_gt_precision_2px_percent": 100.0 * float(
            diagnostics.get(
                "sparse_diag_all_gt_precision_2px_mean", 0.0
            )
        ),
        "raw_gt_precision_4px_percent": 100.0 * float(
            diagnostics.get(
                "sparse_diag_all_gt_precision_4px_mean", 0.0
            )
        ),
        "inlier_gt_precision_2px_percent": 100.0 * float(
            diagnostics.get(
                "sparse_diag_inlier_gt_precision_2px_mean", 0.0
            )
        ),
        "inlier_gt_precision_4px_percent": 100.0 * float(
            diagnostics.get(
                "sparse_diag_inlier_gt_precision_4px_mean", 0.0
            )
        ),
        "matcher_raw_gt_precision_2px_percent": 100.0 * float(
            diagnostics.get(
                "sparse_diag_matcher_raw_all_gt_precision_2px_mean",
                diagnostics.get(
                    "sparse_diag_all_gt_precision_2px_mean", 0.0
                ),
            )
        ),
        "matcher_raw_gt_precision_4px_percent": 100.0 * float(
            diagnostics.get(
                "sparse_diag_matcher_raw_all_gt_precision_4px_mean",
                diagnostics.get(
                    "sparse_diag_all_gt_precision_4px_mean", 0.0
                ),
            )
        ),
        "solver_inlier_ratio_percent": 100.0 * float(
            diagnostics.get(
                "sparse_diag_ransac_inlier_ratio_solver_mean",
                diagnostics.get("sparse_diag_inlier_ratio_mean", 0.0),
            )
        ),
        "mean_hypotheses": diagnostics.get(
            "sparse_diag_ransac_actual_hypotheses_mean"
        ),
        "frontend_ms": diagnostics.get(
            "sparse_diag_runtime_frontend_ms_mean"
        ),
        "matching_ms": diagnostics.get(
            "sparse_diag_runtime_matching_ms_mean"
        ),
        "selection_ms": diagnostics.get(
            "sparse_diag_pose_sufficient_runtime_ms_mean", 0.0
        ),
        "ransac_ms": diagnostics.get(
            "sparse_diag_runtime_ransac_ms_mean"
        ),
        "total_ms": diagnostics.get(
            "sparse_diag_runtime_total_ms_mean"
        ),
        "result_path": str(result_dir),
    }
for label, seeds in summary["results"].items():
    metric_keys = [
        "median_te_cm",
        "mean_te_cm",
        "p90_te_cm",
        "median_ae_deg",
        "mean_ae_deg",
        "recall_2cm_2deg_percent",
        "recall_5cm_5deg_percent",
        "raw_gt_precision_2px_percent",
        "raw_gt_precision_4px_percent",
        "inlier_gt_precision_2px_percent",
        "inlier_gt_precision_4px_percent",
        "matcher_raw_gt_precision_2px_percent",
        "matcher_raw_gt_precision_4px_percent",
        "solver_inlier_ratio_percent",
        "mean_hypotheses",
        "frontend_ms",
        "matching_ms",
        "selection_ms",
        "ransac_ms",
        "total_ms",
    ]
    summary["results"][label]["seed_aggregate"] = {
        key: {
            "mean": float(np.mean([row[key] for row in seeds.values()])),
            "std": float(np.std([row[key] for row in seeds.values()])),
        }
        for key in metric_keys
        if all(row.get(key) is not None for row in seeds.values())
    }
map_path = root / "self_localization_reconstruction"
maps = sorted(map_path.glob("anchor_map_step_*.pt"))
if maps:
    payload = torch.load(maps[-1], map_location="cpu", weights_only=False)
    summary["anchor_count"] = int(
        torch.as_tensor(payload["anchor_xyz"]).shape[0]
    )
    summary["map_bytes"] = maps[-1].stat().st_size
    deployment_artifacts = {
        "anchor_map": maps[-1],
        "metric_state": (
            root / "self_localization_reconstruction"
            / f"metric_state_step_{metric_steps:04d}.pt"
        ),
        "family_state": (
            root / "family_refinement" / "calibrated_family.pt"
        ),
        "selector_state": (
            root / "pose_sufficient_selector"
            / "selector_model_fixed0512.pt"
        ),
    }
    summary["deployment_artifact_bytes"] = {
        name: path.stat().st_size
        for name, path in deployment_artifacts.items()
        if path.is_file()
    }
    summary["deployment_total_bytes"] = int(
        sum(summary["deployment_artifact_bytes"].values())
    )
(root / "frozen_results.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n"
)
print(json.dumps(summary, indent=2, sort_keys=True))
PY
}

evaluate() {
  case " $EVAL_VARIANTS " in
    *" A3_p1_fixed512 "*) selector ;;
    *" A2_family_all "*) family_refinement ;;
    *) deployment_eval_prerequisites ;;
  esac
  local seed
  for seed in $FROZEN_SEEDS; do
    if [[ " $EVAL_VARIANTS " == *" A0_bootstrap "* ]]; then
      eval_one A0_bootstrap "$seed" \
        --sparse_frontend ulfloc_native \
        --landmark_feature_override_path "$BOOTSTRAP_STATE" \
        --override_landmark_features
    fi
    if [[ " $EVAL_VARIANTS " == *" A1_reconstructed "* ]]; then
      eval_one A1_reconstructed "$seed" \
        --sparse_frontend ulfloc_native_metric \
        --materialized_anchor_map_path "$RECON_MAP" \
        --metric_state_path "$METRIC_STATE"
    fi
    if [[ " $EVAL_VARIANTS " == *" A2_family_all "* ]]; then
      eval_one A2_family_all "$seed" \
        --sparse_frontend ulfloc_native_metric \
        --materialized_anchor_map_path "$RECON_MAP" \
        --metric_state_path "$METRIC_STATE" \
        --family_prototype_state_path "$FAMILY_STATE"
    fi
    if [[ " $EVAL_VARIANTS " == *" A3_p1_fixed512 "* ]]; then
      eval_one A3_p1_fixed512 "$seed" \
        --sparse_frontend ulfloc_native_metric \
        --materialized_anchor_map_path "$RECON_MAP" \
        --metric_state_path "$METRIC_STATE" \
        --family_prototype_state_path "$FAMILY_STATE" \
        --pose_sufficient_selector_state_path "$SELECTOR_STATE" \
        --pose_sufficient_budget "$SELECTOR_BUDGET"
    fi
  done
  if [[ "$EVAL_SKIP_SUMMARY" != "1" ]]; then
    summarize
  fi
}

case "$MODE" in
  prepare) prepare ;;
  base) base ;;
  graph) graph ;;
  map) compact_map ;;
  reconstruct) self_localization_reconstruction ;;
  family) family_refinement ;;
  selector) selector ;;
  eval) evaluate ;;
  all) evaluate ;;
esac
