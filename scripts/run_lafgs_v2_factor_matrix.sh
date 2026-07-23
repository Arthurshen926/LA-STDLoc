#!/usr/bin/env bash
set -euo pipefail

# Validation-only A--E factor matrix for the narrowed LaFGS V2 main line.
#
# A: strong landmark IDs + inherited strong descriptor, frozen base geometry
# B: same strong IDs + fresh strict GWFF descriptor
# C: B + fixed 5K pure-native residual
# D: KCS IDs + strict GWFF descriptor (the canonical U0 bootstrap)
# E: D + fixed 5K pure-native residual
#
# No invocation in this script evaluates an official test split.  The A state
# is explicitly geometry-sanitized: historical anchor offsets are not allowed
# to masquerade as a descriptor advantage in this factorization.

if [[ $# -ne 3 ]]; then
  echo "Usage: $0 <scene> <gpu> <a|b|c|d|e|all|summary>" >&2
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
  *) echo "GPU must be 0, 1, or 2; got $GPU" >&2; exit 2 ;;
esac
case "$MODE" in
  a|b|c|d|e|all|summary) ;;
  *) echo "Unsupported factor mode: $MODE" >&2; exit 2 ;;
esac

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/root/miniconda3/envs/cybersim_agent/bin/python}"
DATA_ROOT="${CAMBRIDGE_DATA_ROOT:-/mnt/pool/sqy/Cambridge_stdloc}"
MATCHA_ROOT="${CAMBRIDGE_MATCHA_2DGS_ROOT:-/root/MAtCha/output_cambridge_full_retained_v2}"
REFERENCE_ROOT="${LAFGS_V2_ULFPARITY_REFERENCE_ROOT:-/mnt/pool/sqy/stdloc_lafgs_v2_ulfparity_alternating_20260721}"
EXPERIMENT_ROOT="${LAFGS_V2_FACTOR_MATRIX_ROOT:-/mnt/pool/sqy/stdloc_lafgs_v2_factor_matrix_20260722}"
MODEL_ROOT="$REFERENCE_ROOT/matcha_wrappers/$SCENE"

# This is the immutable sparse deployment contract.  Capacity and initializer
# choices are the only variables in the matrix.
LONGEST_EDGE=0
NATIVE_KEYPOINTS=2048
MATCH_THRESHOLD=0
MAX_MATCHES_PER_LANDMARK=0
EVAL_PROFILE="ulfloc_parity"
EVAL_REPROJECTION_PX=12
VALIDATION_RATIO=0.2
# The factor matrix must use the same sequence-stratified holdout as the
# frozen formal protocol; otherwise bank/descriptor effects are confounded by
# a different validation split.
SPLIT_MODE="stratified_temporal_block"
SPLIT_SEED=2026
SUPPORT_VIEWS=128
TANGENT_BOUND_M=0.005
NORMAL_BOUND_M=0.002
RESIDUAL_STEPS=5000
CAMERA_LOADER_WORKERS="${LAFGS_FACTOR_CAMERA_LOADER_WORKERS:-0}"

case "$SCENE" in
  GreatCourt) MATCHA_RUN="GreatCourt_n20_long_masked_retrain_retry" ;;
  KingsCollege) MATCHA_RUN="KingsCollege_n20_long_masked_retrain" ;;
  OldHospital) MATCHA_RUN="OldHospital_n20_long_masked_retrain_retry" ;;
  ShopFacade) MATCHA_RUN="ShopFacade_n20_long_masked_retrain" ;;
  StMarysChurch) MATCHA_RUN="StMarysChurch_n20_long_masked_retrain" ;;
esac
SOURCE_PLY="$MATCHA_ROOT/$MATCHA_RUN/free_gaussians/point_cloud/iteration_30000/point_cloud.ply"

if [[ "$SCENE" == "OldHospital" ]]; then
  DEFAULT_STRONG_ROOT="/mnt/pool/sqy/stdloc_lafgs_v2_alternating_native_20260720/OldHospital/final_refit/descriptor_refresh_1000"
  DEFAULT_STRONG_PLY="/mnt/pool/sqy/stdloc_lafgs_v2_ulf_v8_full128_20260720/matcha_wrappers/OldHospital/point_cloud/iteration_30000/point_cloud.ply"
else
  DEFAULT_STRONG_ROOT=""
  DEFAULT_STRONG_PLY=""
fi
STRONG_IDS="${LAFGS_FACTOR_STRONG_IDS:-${DEFAULT_STRONG_ROOT:+$DEFAULT_STRONG_ROOT/sampled_idx.pkl}}"
STRONG_STATE="${LAFGS_FACTOR_STRONG_STATE:-${DEFAULT_STRONG_ROOT:+$DEFAULT_STRONG_ROOT/1000_lafgs_map_state.pt}}"
STRONG_SOURCE_PLY="${LAFGS_FACTOR_STRONG_SOURCE_PLY:-$DEFAULT_STRONG_PLY}"
if [[ -z "$STRONG_IDS" || -z "$STRONG_STATE" || -z "$STRONG_SOURCE_PLY" ]]; then
  echo "Set LAFGS_FACTOR_STRONG_IDS, LAFGS_FACTOR_STRONG_STATE, and LAFGS_FACTOR_STRONG_SOURCE_PLY for $SCENE" >&2
  exit 2
fi
if ! [[ "$CAMERA_LOADER_WORKERS" =~ ^[0-9]+$ ]]; then
  echo "LAFGS_FACTOR_CAMERA_LOADER_WORKERS must be a non-negative integer" >&2
  exit 2
fi

# Keep factor artifacts in a protocol-specific namespace.  The original v1
# directory was produced with temporal_block before the formal protocol moved
# to stratified_temporal_block, so it must never be implicitly reused.
FACTOR_PROTOCOL_VERSION="v2_split${SPLIT_MODE}_fullres_native_uncapped"
RUN_ROOT="${LAFGS_FACTOR_RUN_ROOT:-$EXPERIMENT_ROOT/$SCENE/strong16k_vs_kcs16k_${FACTOR_PROTOCOL_VERSION}}"
A_DIR="$RUN_ROOT/a_strong_ids_strong_descriptor_base_geometry"
B_DIR="$RUN_ROOT/b_strong_ids_gwff"
C_DIR="$RUN_ROOT/c_strong_ids_gwff_native_residual_${RESIDUAL_STEPS}"
CONTROL_STATE="$RUN_ROOT/inputs/strong_descriptor_base_geometry.pt"
CONFIG_ROOT="$RUN_ROOT/configs"
LOG_ROOT="$RUN_ROOT/logs"
RESULT_ROOT="$RUN_ROOT/results"
STDLOC_RESULTS_ROOT="$RUN_ROOT/stdloc_results"
MANIFEST="$RUN_ROOT/factor_matrix_manifest.json"
QUERY_CACHE="${LAFGS_FACTOR_QUERY_CACHE_PATH:-$REFERENCE_ROOT/OldHospital/ulfparity_native20k_s128_k2048_v2/query_cache_native_fullres_k2048.pt}"
if [[ "$SCENE" != "OldHospital" && -z "${LAFGS_FACTOR_QUERY_CACHE_PATH:-}" ]]; then
  QUERY_CACHE="$RUN_ROOT/query_cache_native_fullres_k${NATIVE_KEYPOINTS}.pt"
fi

# D/E are deliberately the same U0 root used by the capacity matrix.  Build
# the name from this runner's locked split so a legacy temporal artifact can
# never be silently reused after the formal default changed to stratified.
KCS_TAG="ulfparity_native16000_s128_k2048_ulfloc_parity_tau0_cap0_v8_fullres_native_uncapped_pure_native"
if [[ "$SPLIT_MODE" != "temporal_block" ]]; then
  KCS_TAG="${KCS_TAG}_split${SPLIT_MODE}"
fi
KCS_RUN_ROOT="${LAFGS_FACTOR_KCS_RUN_ROOT:-$REFERENCE_ROOT/$SCENE/$KCS_TAG}"
KCS_BOOTSTRAP_DIR="$KCS_RUN_ROOT/bootstrap"
KCS_BOOTSTRAP_STATE="$KCS_BOOTSTRAP_DIR/0_lafgs_map_state.pt"
KCS_BOOTSTRAP_IDS="$KCS_BOOTSTRAP_DIR/sampled_idx.pkl"
KCS_BOOTSTRAP_META="$KCS_BOOTSTRAP_DIR/landmark_meta.pt"

export CUDA_VISIBLE_DEVICES="$GPU"
export CUDA_HOME=/usr/local/cuda-11.8
export PATH="/root/miniconda3/envs/cybersim_agent/bin:$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="/root/miniconda3/envs/cybersim_agent/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$REPO_ROOT"
export PYTHONHASHSEED=2026
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export STDLOC_CAMERA_LOADER_WORKERS="$CAMERA_LOADER_WORKERS"
export STDLOC_RESULTS_ROOT

mkdir -p "$RUN_ROOT/inputs" "$CONFIG_ROOT" "$LOG_ROOT" "$RESULT_ROOT" "$STDLOC_RESULTS_ROOT"
cd "$REPO_ROOT"

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "Required artifact is missing: $1" >&2
    exit 1
  fi
}

run_logged() {
  local stage="$1"
  shift
  printf '%q ' "$@" > "$LOG_ROOT/${stage}.command.sh"
  printf '\n' >> "$LOG_ROOT/${stage}.command.sh"
  "$@" 2>&1 | tee "$LOG_ROOT/${stage}.log"
}

verify_control_inputs() {
  require_file "$SOURCE_PLY"
  require_file "$MODEL_ROOT/artifact_provenance.json"
  require_file "$STRONG_IDS"
  require_file "$STRONG_STATE"
  require_file "$STRONG_SOURCE_PLY"
  "$PYTHON" - "$SOURCE_PLY" "$STRONG_SOURCE_PLY" "$STRONG_IDS" "$STRONG_STATE" <<'PY'
import hashlib
import pickle
import sys

import torch

def digest(path):
    value = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()

source_ply, strong_ply, ids_path, state_path = sys.argv[1:]
if digest(source_ply) != digest(strong_ply):
    raise SystemExit('Strong descriptor source PLY does not match the frozen 2DGS input')
with open(ids_path, 'rb') as handle:
    ids = torch.as_tensor(pickle.load(handle), dtype=torch.long).reshape(-1)
state = torch.load(state_path, map_location='cpu')
state_ids = torch.as_tensor(state.get('landmark_indices'), dtype=torch.long).reshape(-1)
features = torch.as_tensor(state.get('landmark_features'), dtype=torch.float32)
if ids.numel() != 16384 or not torch.equal(ids, state_ids):
    raise SystemExit('Strong IDs and state must be an exactly aligned 16K bank')
if features.shape != (ids.numel(), 256):
    raise SystemExit(f'Unexpected strong descriptor tensor shape: {tuple(features.shape)}')
print('Verified factor control inputs: 16K IDs, 256D descriptors, matching frozen PLY')
PY
}

write_manifest() {
  "$PYTHON" - "$MANIFEST" "$STRONG_STATE" <<PY
import json
from pathlib import Path
import sys

import torch

strong_state_path = Path(sys.argv[2])
strong_state = torch.load(strong_state_path, map_location="cpu")
strong_config = dict(strong_state.get("config", {}))
raw_anchor_offset = torch.as_tensor(
    strong_state.get("raw_anchor_offset", torch.zeros(1)), dtype=torch.float32
)

payload = {
    "schema_version": 2,
    "purpose": "validation_only_lafgs_ids_gwff_residual_factor_matrix",
    "test_evaluation_forbidden": True,
    "scene": "${SCENE}",
    "formal_deployment_protocol": {
        "longest_edge": ${LONGEST_EDGE},
        "native_keypoints": ${NATIVE_KEYPOINTS},
        "frontend": "ulfloc_native",
        "topk": 1,
        "cosine_threshold": ${MATCH_THRESHOLD},
        "max_matches_per_landmark": ${MAX_MATCHES_PER_LANDMARK},
        "candidate_split_mode": "${SPLIT_MODE}",
        "candidate_split_seed": ${SPLIT_SEED},
        "factor_protocol_version": "${FACTOR_PROTOCOL_VERSION}",
    },
    "factors": {
        "A": "strong_ids + inherited_strong_descriptor + frozen_base_geometry",
        "B": "strong_ids + strict_gwff",
        "C": "B + fixed_5k_pure_native_residual",
        "D": "KCS_ids + strict_gwff",
        "E": "D + fixed_5k_pure_native_residual",
    },
    "inputs": {
        "source_ply": str(Path("${SOURCE_PLY}").resolve()),
        "strong_ids": str(Path("${STRONG_IDS}").resolve()),
        "strong_state": str(Path("${STRONG_STATE}").resolve()),
        "strong_source_ply": str(Path("${STRONG_SOURCE_PLY}").resolve()),
        "query_cache": str(Path("${QUERY_CACHE}").resolve()),
        "kcs_u0_root": str(Path("${KCS_RUN_ROOT}").resolve()),
    },
    "strong_control_provenance": {
        "role": "historical_descriptor_control_evaluated_on_current_validation_split",
        "source_training_split_mode": strong_config.get("split_mode"),
        "source_query_feature_contract": strong_config.get("query_feature_contract"),
        "source_observation_source": strong_config.get("observation_source"),
        "source_native_outcome_mode": strong_config.get("native_outcome_mode"),
        "source_raw_anchor_offset_absmax_m": float(raw_anchor_offset.abs().max().item()),
        "factor_geometry_policy": "reset_to_frozen_source_2dgs_xyz_before_A",
    },
    "runtime": {"camera_loader_workers": ${CAMERA_LOADER_WORKERS}},
}
Path("${MANIFEST}").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY
}

base_map_args() {
  printf '%s\0' \
    "$PYTHON" train_lafgs_map.py \
    --model_path "$MODEL_ROOT" --source_path "$DATA_ROOT/$SCENE" \
    --images processed --data_device cpu --gaussian_type 2dgs --feature_type sp \
    --resolution 1 --longest_edge "$LONGEST_EDGE" --norm_before_render --load_iteration 30000 \
    --query_feature_contract native_resized_input \
    --query_cache_path "$QUERY_CACHE" --query_cache_policy readonly \
    --visibility_mode rasterizer --objective hard \
    --native_keypoint_count "$NATIVE_KEYPOINTS" \
    --native_association_radius_px 2 --native_sampling_mode detector_grid \
    --generic_proposal_count 0 --generic_proposal_weight 0 \
    --distill_budget 0 --validation_ratio "$VALIDATION_RATIO" \
    --split_mode "$SPLIT_MODE" --split_seed "$SPLIT_SEED" --train_seed 2026 \
    --max_observations 512 --validation_observations 512 \
    --tangent_bound_m "$TANGENT_BOUND_M" --normal_bound_m "$NORMAL_BOUND_M"
}

append_base_map_args() {
  local -n target="$1"
  local item
  while IFS= read -r -d '' item; do
    target+=("$item")
  done < <(base_map_args)
}

validate_state_ids() {
  local state="$1"
  local ids="$2"
  require_file "$state"
  require_file "$ids"
  "$PYTHON" - "$state" "$ids" <<'PY'
import pickle
import sys
import torch

state = torch.load(sys.argv[1], map_location='cpu')
with open(sys.argv[2], 'rb') as handle:
    ids = torch.as_tensor(pickle.load(handle), dtype=torch.long).reshape(-1)
state_ids = torch.as_tensor(state.get('landmark_indices'), dtype=torch.long).reshape(-1)
if not torch.equal(state_ids, ids):
    raise SystemExit('State and landmark IDs are not exactly aligned')
features = torch.as_tensor(state.get('landmark_features'), dtype=torch.float32)
if features.shape != (ids.numel(), 256):
    raise SystemExit(f'Unexpected state feature shape: {tuple(features.shape)}')
print(f'Validated exact bank alignment: {ids.numel()} landmarks')
PY
}

# A file's presence is not evidence that it was created under the frozen
# protocol.  In particular, the former factor root used temporal_block and
# would otherwise silently contaminate the A--E comparison.  Verify the
# checkpoint configuration before any reuse and fail closed on a mismatch.
verify_factor_state_protocol() {
  local label="$1"
  local state="$2"
  local expected_native_outcome="$3"
  local expected_retrieval_weight="$4"
  local expected_trust_weight="$5"
  require_file "$state"
  "$PYTHON" - "$label" "$state" "$SPLIT_MODE" "$SPLIT_SEED" \
    "$expected_native_outcome" "$expected_retrieval_weight" "$expected_trust_weight" \
    "$NATIVE_KEYPOINTS" <<'PY'
import math
import sys

import torch

(
    label,
    state_path,
    expected_split_mode,
    expected_split_seed,
    expected_native_outcome,
    expected_retrieval_weight,
    expected_trust_weight,
    expected_keypoints,
) = sys.argv[1:]

state = torch.load(state_path, map_location="cpu")
config = dict(state.get("config", {}))
errors = []

def exact(key, expected):
    actual = config.get(key)
    if actual != expected:
        errors.append(f"{key}={actual!r}, expected {expected!r}")

def numeric(key, expected):
    actual = config.get(key)
    try:
        valid = math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=1e-12)
    except (TypeError, ValueError):
        valid = False
    if not valid:
        errors.append(f"{key}={actual!r}, expected {expected!r}")

exact("split_mode", expected_split_mode)
numeric("split_seed", int(expected_split_seed))
exact("query_feature_contract", "native_resized_input")
exact("observation_source", "native")
numeric("native_sparse_keypoint_count", int(expected_keypoints))
numeric("native_anchor_aux_weight", 0.0)
if bool(config.get("native_outcome_mode")) != bool(int(expected_native_outcome)):
    errors.append(
        "native_outcome_mode="
        f"{config.get('native_outcome_mode')!r}, expected {bool(int(expected_native_outcome))!r}"
    )
for key in ("mv_weight", "local_weight", "dustbin_weight", "geometry_weight", "pose_weight"):
    numeric(key, 0.0)
numeric("retrieval_weight", float(expected_retrieval_weight))
numeric("trust_weight", float(expected_trust_weight))

if errors:
    raise SystemExit(
        f"Refusing to reuse factor state {label} at {state_path}: " + "; ".join(errors)
    )
print(f"Verified frozen factor state protocol for {label}")
PY
}

# Results are equally protocol-sensitive.  Reusing a temporal-block candidate
# validation under a stratified selection rule would make the factor table look
# complete while comparing different query populations.
verify_factor_validation_protocol() {
  local label="$1"
  local ref="$2"
  require_file "$ref"
  "$PYTHON" - "$label" "$ref" "$SPLIT_MODE" "$SPLIT_SEED" "$LONGEST_EDGE" <<'PY'
import json
import sys
from pathlib import Path

label, ref_path, expected_split_mode, expected_split_seed, expected_longest_edge = sys.argv[1:]
result_root = Path(ref_path).read_text().strip()
summary_path = Path(result_root) / "results_summary.json"
if not summary_path.is_file():
    raise SystemExit(f"Missing results_summary.json for factor {label}: {summary_path}")
payload = json.loads(summary_path.read_text())
protocol = dict(payload.get("evaluation_protocol", {}))
candidate = dict(protocol.get("candidate_split", {}))
errors = []
if protocol.get("evaluation_camera_subset") != "candidate_validation":
    errors.append(
        "evaluation_camera_subset="
        f"{protocol.get('evaluation_camera_subset')!r}, expected 'candidate_validation'"
    )
if candidate.get("mode") != expected_split_mode:
    errors.append(f"candidate_split.mode={candidate.get('mode')!r}, expected {expected_split_mode!r}")
if candidate.get("seed") != int(expected_split_seed):
    errors.append(f"candidate_split.seed={candidate.get('seed')!r}, expected {expected_split_seed!r}")
if candidate.get("direct_holdout") is not True:
    errors.append(f"candidate_split.direct_holdout={candidate.get('direct_holdout')!r}, expected True")
if protocol.get("longest_edge") != int(expected_longest_edge):
    errors.append(f"longest_edge={protocol.get('longest_edge')!r}, expected {expected_longest_edge!r}")
if errors:
    raise SystemExit(
        f"Refusing to reuse factor validation {label} at {summary_path}: " + "; ".join(errors)
    )
print(f"Verified frozen factor validation protocol for {label}")
PY
}

make_eval_config() {
  local label="$1"
  local ids="$2"
  local meta="$3"
  local state="$4"
  local cfg="$CONFIG_ROOT/${label}_validation.yaml"
  "$PYTHON" scripts/make_stdloc_eval_cfg.py \
    --base_cfg configs/stdloc_cambridge.yaml --output "$cfg" \
    --artifact_model_path "$MODEL_ROOT" \
    --detector_folder ulfloc_native_no_detector --detector_iters 0 \
    --landmark_path "$ids" --landmark_meta_path "$meta" \
    --landmark_feature_override_path "$state" --override_landmark_features \
    --detect_num "$NATIVE_KEYPOINTS" --nms 2 \
    --sparse_query_feature_contract native_resized_input --sparse_frontend ulfloc_native \
    --reprojection_error "$EVAL_REPROJECTION_PX" --match_threshold "$MATCH_THRESHOLD" --match_topk 1 \
    --max_matches_per_landmark "$MAX_MATCHES_PER_LANDMARK" \
    --candidate_frontend_match_policy error --diagnostics \
    --diagnostics_grid_rows 4 --diagnostics_grid_cols 4 --diagnostics_voxel_size 1.0 \
    --diagnostics_task_translation_scale_m 0.07160573943725686 \
    --diagnostics_task_rotation_scale_degrees 2.0 \
    > "$LOG_ROOT/${label}_validation_config.json"
  printf '%s\n' "$cfg"
}

run_eval() {
  local label="$1"
  local ids="$2"
  local meta="$3"
  local state="$4"
  local ref="$RESULT_ROOT/${label}_validation.results_path"
  if [[ -f "$ref" ]]; then
    verify_factor_validation_protocol "$label" "$ref"
    echo "[factor matrix] Reusing protocol-matched validation: $label"
    return
  fi
  local cfg
  cfg="$(make_eval_config "$label" "$ids" "$meta" "$state")"
  run_logged "${label}_validation" \
    "$PYTHON" stdloc.py \
    --model_path "$MODEL_ROOT" --source_path "$DATA_ROOT/$SCENE" \
    --images processed --data_device cpu --gaussian_type 2dgs --feature_type sp \
    --resolution 1 --longest_edge "$LONGEST_EDGE" --norm_before_render --iteration 30000 \
    --cfg "$cfg" --prefix "lafgs-v2-factor-${SCENE}-${label}-validation" --sparse_only \
    --evaluation_camera_subset candidate_validation --candidate_direct_validation_holdout \
    --candidate_validation_ratio "$VALIDATION_RATIO" \
    --candidate_split_mode "$SPLIT_MODE" --candidate_split_seed "$SPLIT_SEED"
  local output_path
  output_path="$(sed -n 's/^Output path: //p' "$LOG_ROOT/${label}_validation.log" | tail -n 1)"
  if [[ -z "$output_path" || ! -f "$output_path/results_summary.json" ]]; then
    echo "Validation did not create results_summary.json for $label" >&2
    exit 1
  fi
  printf '%s\n' "$output_path" > "$ref"
  verify_factor_validation_protocol "$label" "$ref"
}

prepare_strong_descriptor_control() {
  verify_control_inputs
  if [[ -f "$CONTROL_STATE" ]]; then
    return
  fi
  run_logged prepare_strong_descriptor_control \
    "$PYTHON" scripts/prepare_lafgs_factor_state.py \
    --source-state "$STRONG_STATE" --source-ply "$SOURCE_PLY" \
    --output-state "$CONTROL_STATE" \
    --tangent-bound-m "$TANGENT_BOUND_M" --normal-bound-m "$NORMAL_BOUND_M"
}

factor_a() {
  write_manifest
  prepare_strong_descriptor_control
  local state="$A_DIR/0_lafgs_map_state.pt"
  local ids="$A_DIR/sampled_idx.pkl"
  local meta="$A_DIR/landmark_meta.pt"
  if [[ ! -f "$state" ]]; then
    local command=()
    append_base_map_args command
    command+=(
      --output_dir "$A_DIR" --scaffold_mode file --landmark_path "$STRONG_IDS"
      --initial_state_path "$CONTROL_STATE" --initial_state_blend 1 --initial_state_alignment exact
      --initialization_mode ulf_parity --observation_source native --native_anchor_aux_weight 0
      --no-native_outcome_mode --steps 0 --save_steps 0
      --mv_weight 0 --retrieval_weight 0 --trust_weight 0 --local_weight 0 --dustbin_weight 0
      --geometry_weight 0 --pose_weight 0 --pose_gradient_mode off
    )
    run_logged factor_a "${command[@]}"
  fi
  validate_state_ids "$state" "$ids"
  verify_factor_state_protocol A "$state" 0 0 0
  run_eval A_strong_descriptor "$ids" "$meta" "$state"
}

factor_b() {
  write_manifest
  verify_control_inputs
  local state="$B_DIR/0_lafgs_map_state.pt"
  local ids="$B_DIR/sampled_idx.pkl"
  local meta="$B_DIR/landmark_meta.pt"
  if [[ ! -f "$state" ]]; then
    local command=()
    append_base_map_args command
    command+=(
      --output_dir "$B_DIR" --scaffold_mode file --landmark_path "$STRONG_IDS"
      --initialization_mode ulf_parity --ulf_consensus_keypoints "$NATIVE_KEYPOINTS"
      --ulf_consensus_radius_px 1.0 --ulf_consensus_knn 32
      --ulf_consensus_max_views "$SUPPORT_VIEWS" --ulf_fusion_max_views "$SUPPORT_VIEWS"
      --ulf_fusion_min_cosine 0 --ulf_support_view_sampling uniform
      --ulf_parity_kcs_mask_policy rgb_only
      --observation_source native --native_anchor_aux_weight 0 --no-native_outcome_mode
      --steps 0 --save_steps 0
      --mv_weight 0 --retrieval_weight 0 --trust_weight 0 --local_weight 0 --dustbin_weight 0
      --geometry_weight 0 --pose_weight 0 --pose_gradient_mode off
    )
    run_logged factor_b "${command[@]}"
  fi
  validate_state_ids "$state" "$ids"
  verify_factor_state_protocol B "$state" 0 0 0
  run_eval B_strong_gwff "$ids" "$meta" "$state"
}

factor_c() {
  factor_b
  local state="$C_DIR/${RESIDUAL_STEPS}_lafgs_map_state.pt"
  local ids="$B_DIR/sampled_idx.pkl"
  local meta="$B_DIR/landmark_meta.pt"
  if [[ ! -f "$state" ]]; then
    local command=()
    append_base_map_args command
    command+=(
      --output_dir "$C_DIR" --scaffold_mode file --landmark_path "$ids"
      --initial_state_path "$B_DIR/0_lafgs_map_state.pt" --initial_state_blend 1 --initial_state_alignment exact
      --initialization_mode ulf_parity --observation_source native --native_anchor_aux_weight 0
      --native_outcome_mode --native_nce_weight 0
      --native_keep_weight 1 --native_keep_margin 0.05
      --native_swap_weight 1 --native_swap_margin 0.05
      --native_miss_weight 1 --native_miss_margin 0.05
      --native_reject_weight 0.05 --native_reject_threshold "$MATCH_THRESHOLD"
      --steps "$RESIDUAL_STEPS" --save_steps 500 1000 2500 "$RESIDUAL_STEPS"
      --feature_lr 5e-5 --weight_decay 1e-4 --hypothesis_topk 32
      --positive_radius_px 2 --negative_radius_px 6
      --mv_weight 0 --retrieval_weight 1 --trust_weight 0.02 --local_weight 0 --dustbin_weight 0
      --geometry_weight 0 --pose_weight 0 --pose_gradient_mode off --log_interval 100
    )
    run_logged factor_c "${command[@]}"
  fi
  verify_factor_state_protocol C "$state" 1 1 0.02
  local step
  for step in 500 1000 2500 "$RESIDUAL_STEPS"; do
    local checkpoint="$C_DIR/${step}_lafgs_map_state.pt"
    if [[ -f "$checkpoint" ]]; then
      run_eval "C_strong_gwff_residual_${step}" "$ids" "$meta" "$checkpoint"
    fi
  done
}

run_kcs_u0() {
  local stage="$1"
  env -u LAFGS_ULF_BOOTSTRAP_SOURCE_DIR \
    -u LAFGS_ULF_LANDMARK_BUDGET -u LAFGS_ULF_SUPPORT_VIEWS \
    -u LAFGS_ULF_SUPPORT_VIEW_SAMPLING -u LAFGS_ULF_RESIDUAL_STEPS \
    -u LAFGS_ULF_EVAL_PROFILE -u LAFGS_ULF_NATIVE_MATCH_THRESHOLD \
    -u LAFGS_ULF_MAX_MATCHES_PER_LANDMARK -u LAFGS_ULF_PARITY_KCS_MASK_POLICY \
    -u LAFGS_ULF_SPLIT_SEED -u LAFGS_ULF_TRAIN_SEED \
    LAFGS_ULF_LANDMARK_BUDGET=16000 \
    LAFGS_ULF_SUPPORT_VIEWS=128 \
    LAFGS_ULF_SUPPORT_VIEW_SAMPLING=uniform \
    LAFGS_ULF_PARITY_KCS_MASK_POLICY=rgb_only \
    LAFGS_ULF_SPLIT_MODE="$SPLIT_MODE" \
    LAFGS_ULF_SPLIT_SEED="$SPLIT_SEED" \
    LAFGS_ULF_TRAIN_SEED=2026 \
    LAFGS_ULF_CAMERA_LOADER_WORKERS="$CAMERA_LOADER_WORKERS" \
    LAFGS_ULF_RESIDUAL_STEPS="$RESIDUAL_STEPS" \
    LAFGS_ULF_QUERY_CACHE_PATH="$QUERY_CACHE" \
    bash scripts/run_lafgs_v2_ulfparity_alternating.sh "$SCENE" "$GPU" "$stage"
}

factor_d() {
  if [[ ! -f "$KCS_BOOTSTRAP_STATE" || ! -f "$KCS_RUN_ROOT/validation/results/bootstrap_validation.results_path" ]]; then
    run_kcs_u0 bootstrap_validate
  fi
  validate_state_ids "$KCS_BOOTSTRAP_STATE" "$KCS_BOOTSTRAP_IDS"
  verify_factor_state_protocol D "$KCS_BOOTSTRAP_STATE" 0 0 0
  verify_factor_validation_protocol D "$KCS_RUN_ROOT/validation/results/bootstrap_validation.results_path"
}

factor_e() {
  factor_d
  local state="$KCS_RUN_ROOT/residual_${RESIDUAL_STEPS}/${RESIDUAL_STEPS}_lafgs_map_state.pt"
  if [[ ! -f "$state" || ! -f "$KCS_RUN_ROOT/validation/results/residual_${RESIDUAL_STEPS}_validation.results_path" ]]; then
    run_kcs_u0 residual_validate
  fi
  validate_state_ids "$state" "$KCS_BOOTSTRAP_IDS"
  verify_factor_state_protocol E "$state" 1 1 0.02
  verify_factor_validation_protocol E "$KCS_RUN_ROOT/validation/results/residual_${RESIDUAL_STEPS}_validation.results_path"
}

write_summary() {
  "$PYTHON" - "$RUN_ROOT/factor_matrix_summary.json" "$RUN_ROOT/factor_matrix_summary.md" \
    "$RESULT_ROOT/A_strong_descriptor_validation.results_path" \
    "$RESULT_ROOT/B_strong_gwff_validation.results_path" \
    "$RESULT_ROOT/C_strong_gwff_residual_${RESIDUAL_STEPS}_validation.results_path" \
    "$KCS_RUN_ROOT/validation/results/bootstrap_validation.results_path" \
    "$KCS_RUN_ROOT/validation/results/residual_${RESIDUAL_STEPS}_validation.results_path" <<'PY'
import json
import sys
from pathlib import Path

output_json = Path(sys.argv[1])
output_markdown = Path(sys.argv[2])
labels = [
    ("A", "strong IDs + inherited descriptor + base geometry"),
    ("B", "strong IDs + strict GWFF"),
    ("C", "B + 5K native residual"),
    ("D", "KCS IDs + strict GWFF"),
    ("E", "D + 5K native residual"),
]
refs = [Path(value) for value in sys.argv[3:]]

keys = {
    "median_te_cm": ("sparse", "median_te"),
    "median_ae_deg": ("sparse", "median_ae"),
    "recall_5cm": ("sparse", "recall_5cm_5d"),
    "avg_inliers": ("sparse", "avg_inliers"),
    "raw_gt_p2": ("sparse_diagnostics", "sparse_diag_all_gt_precision_2px_mean"),
    "inlier_gt_p2": ("sparse_diagnostics", "sparse_diag_inlier_gt_precision_2px_mean"),
    "translation_pose_info_logdet": ("sparse_diagnostics", "sparse_diag_inlier_pose_info_translation_logdet_mean"),
    "frontend_ms": ("sparse_diagnostics", "sparse_diag_runtime_frontend_ms_mean"),
    "matching_ms": ("sparse_diagnostics", "sparse_diag_runtime_matching_ms_mean"),
    "ransac_ms": ("sparse_diagnostics", "sparse_diag_runtime_ransac_ms_mean"),
    "total_ms": ("sparse_diagnostics", "sparse_diag_runtime_total_ms_mean"),
    "ransac_hypotheses": ("sparse_diagnostics", "sparse_diag_ransac_actual_hypotheses_mean"),
}

rows = []
for (name, definition), ref in zip(labels, refs):
    row = {"factor": name, "definition": definition, "available": False}
    if ref.is_file():
        result_path = Path(ref.read_text().strip()) / "results_summary.json"
        if result_path.is_file():
            payload = json.loads(result_path.read_text())
            row["available"] = True
            row["results_summary"] = str(result_path)
            for key, (section, source_key) in keys.items():
                row[key] = payload.get(section, {}).get(source_key)
    rows.append(row)

payload = {
    "schema_version": 2,
    "purpose": "validation_only_lafgs_factor_matrix",
    "test_metrics_used": False,
    "rows": rows,
}
output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
lines = [
    "# LaFGS A-E Factor Matrix",
    "",
    "Validation only. All rows use full-resolution native SuperPoint, top-1 cosine retrieval, threshold 0, and no per-landmark match cap.",
    "",
    "| Factor | Definition | Median TE (cm) | Recall@5cm | Raw P@2 | Inlier P@2 | Pose-info | Total ms | RANSAC hypotheses |",
    "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
]
for row in rows:
    if not row["available"]:
        lines.append(f"| {row['factor']} | {row['definition']} | pending | pending | pending | pending | pending | pending | pending |")
        continue
    def value(key):
        value = row.get(key)
        return "n/a" if value is None else f"{float(value):.4f}"
    lines.append(
        f"| {row['factor']} | {row['definition']} | {value('median_te_cm')} | "
        f"{value('recall_5cm')} | {value('raw_gt_p2')} | {value('inlier_gt_p2')} | "
        f"{value('translation_pose_info_logdet')} | {value('total_ms')} | {value('ransac_hypotheses')} |"
    )
output_markdown.write_text("\n".join(lines) + "\n")
print(output_json)
print(output_markdown)
PY
}

case "$MODE" in
  a) factor_a ;;
  b) factor_b ;;
  c) factor_c ;;
  d) factor_d ;;
  e) factor_e ;;
  all)
    factor_a
    factor_b
    factor_c
    factor_d
    factor_e
    write_summary
    ;;
  summary) write_summary ;;
esac
