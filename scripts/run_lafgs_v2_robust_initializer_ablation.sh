#!/usr/bin/env bash
set -euo pipefail

# Validation-only robust KCS/GWFF study. This script is intentionally separate
# from run_lafgs_v2_ulfparity_alternating.sh: strict ULF parity is a control,
# whereas this path adds explicit consensus gates and descriptor outlier trim.

if [[ $# -ne 4 ]]; then
  echo "Usage: $0 <scene> <gpu> <support_rgb_only|deployment_post_filter> <bootstrap|validate|residual|residual_validate|select_residual|all>" >&2
  exit 2
fi

SCENE="$1"
GPU="$2"
MASK_POLICY="$3"
MODE="$4"

case "$SCENE" in
  GreatCourt|KingsCollege|OldHospital|ShopFacade|StMarysChurch) ;;
  *) echo "Unsupported Cambridge scene: $SCENE" >&2; exit 2 ;;
esac
case "$GPU" in
  0|1|2) ;;
  *) echo "GPU must be 0, 1, or 2; got $GPU" >&2; exit 2 ;;
esac
case "$MASK_POLICY" in
  support_rgb_only|deployment_post_filter) ;;
  *) echo "Unsupported support mask policy: $MASK_POLICY" >&2; exit 2 ;;
esac
case "$MODE" in
  bootstrap|validate|residual|residual_validate|select_residual|all) ;;
  *) echo "Unsupported mode: $MODE" >&2; exit 2 ;;
esac

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/root/miniconda3/envs/cybersim_agent/bin/python}"
DATA_ROOT="${CAMBRIDGE_DATA_ROOT:-/mnt/pool/sqy/Cambridge_stdloc}"
MATCHA_ROOT="${CAMBRIDGE_MATCHA_2DGS_ROOT:-/root/MAtCha/output_cambridge_full_retained_v2}"
REFERENCE_ROOT="${LAFGS_V2_ULFPARITY_REFERENCE_ROOT:-/mnt/pool/sqy/stdloc_lafgs_v2_ulfparity_alternating_20260721}"
EXPERIMENT_ROOT="${LAFGS_V2_ROBUST_INITIALIZER_ROOT:-/mnt/pool/sqy/stdloc_lafgs_v2_robust_initializer_20260722}"

# Formal sparse deployment contract. Do not make these environment-tunable in
# this script; only KCS/GWFF construction varies between its two variants.
LANDMARK_BUDGET="${LAFGS_ROBUST_LANDMARK_BUDGET:-20000}"
NATIVE_KEYPOINTS=2048
LONGEST_EDGE=0
MATCH_THRESHOLD=0
MAX_MATCHES_PER_LANDMARK=0
VALIDATION_RATIO="${LAFGS_ROBUST_VALIDATION_RATIO:-0.2}"
# Keep robust KCS/GWFF ablations on the same sequence-stratified holdout as
# the formal native residual pipeline.  Legacy temporal_block remains an
# explicit opt-in for historical comparisons only.
SPLIT_MODE="${LAFGS_ROBUST_SPLIT_MODE:-stratified_temporal_block}"
SPLIT_SEED="${LAFGS_ROBUST_SPLIT_SEED:-2026}"
TRAIN_SEED="${LAFGS_ROBUST_TRAIN_SEED:-2026}"
SUPPORT_VIEWS="${LAFGS_ROBUST_SUPPORT_VIEWS:-128}"
SUPPORT_SAMPLING="${LAFGS_ROBUST_SUPPORT_SAMPLING:-uniform}"
MIN_VISIBLE_VIEWS="${LAFGS_ROBUST_MIN_VISIBLE_VIEWS:-4}"
MIN_VOTES="${LAFGS_ROBUST_MIN_VOTES:-2}"
MIN_RATE="${LAFGS_ROBUST_MIN_RATE:-0.1}"
VIEW_BINS="${LAFGS_ROBUST_VIEW_BINS:-4}"
MIN_VIEW_BINS="${LAFGS_ROBUST_MIN_VIEW_BINS:-2}"
TRAJECTORY_BINS="${LAFGS_ROBUST_TRAJECTORY_BINS:-4}"
MIN_TRAJECTORY_BINS="${LAFGS_ROBUST_MIN_TRAJECTORY_BINS:-2}"
TRIM_FRACTION="${LAFGS_ROBUST_TRIM_FRACTION:-0.1}"
DESCRIPTOR_MIN_COSINE="${LAFGS_ROBUST_DESCRIPTOR_MIN_COSINE:--1.0}"
TRIM_HIST_BINS="${LAFGS_ROBUST_TRIM_HIST_BINS:-64}"
FUSION_REFERENCE_MODE="${LAFGS_ROBUST_FUSION_REFERENCE_MODE:-mean}"
EVAL_REPROJECTION_PX=12
RESIDUAL_STEPS="${LAFGS_ROBUST_RESIDUAL_STEPS:-5000}"
# Full-resolution camera tensors are large enough that worker-prefetching can
# exhaust /dev/shm when this bootstrap overlaps another formal run.  Match the
# main runner's parent-process default; throughput experiments must opt in.
CAMERA_LOADER_WORKERS="${LAFGS_ROBUST_CAMERA_LOADER_WORKERS:-0}"

if [[ "$LANDMARK_BUDGET" -le 0 ]]; then
  echo "LAFGS_ROBUST_LANDMARK_BUDGET must be positive" >&2
  exit 2
fi
if ! [[ "$CAMERA_LOADER_WORKERS" =~ ^[0-9]+$ ]]; then
  echo "LAFGS_ROBUST_CAMERA_LOADER_WORKERS must be a non-negative integer" >&2
  exit 2
fi
if ! [[ "$TRAIN_SEED" =~ ^[0-9]+$ ]]; then
  echo "LAFGS_ROBUST_TRAIN_SEED must be a non-negative integer" >&2
  exit 2
fi
case "$FUSION_REFERENCE_MODE" in
  mean|weighted_cosine_medoid) ;;
  *) echo "Unsupported LAFGS_ROBUST_FUSION_REFERENCE_MODE: $FUSION_REFERENCE_MODE" >&2; exit 2 ;;
esac

tag_value() {
  local value="$1"
  value="${value//-/m}"
  value="${value//./p}"
  printf '%s' "$value"
}

case "$SCENE" in
  GreatCourt) MATCHA_RUN="GreatCourt_n20_long_masked_retrain_retry" ;;
  KingsCollege) MATCHA_RUN="KingsCollege_n20_long_masked_retrain" ;;
  OldHospital) MATCHA_RUN="OldHospital_n20_long_masked_retrain_retry" ;;
  ShopFacade) MATCHA_RUN="ShopFacade_n20_long_masked_retrain" ;;
  StMarysChurch) MATCHA_RUN="StMarysChurch_n20_long_masked_retrain" ;;
esac

MODEL_ROOT="$REFERENCE_ROOT/matcha_wrappers/$SCENE"
SOURCE_PLY="$MATCHA_ROOT/$MATCHA_RUN/free_gaussians/point_cloud/iteration_30000/point_cloud.ply"
TAG="robustkcs_gwff${LANDMARK_BUDGET}_s${SUPPORT_VIEWS}_${SUPPORT_SAMPLING}_mv$(tag_value "$MIN_VISIBLE_VIEWS")_v$(tag_value "$MIN_VOTES")_r$(tag_value "$MIN_RATE")_vb${VIEW_BINS}m${MIN_VIEW_BINS}_tb${TRAJECTORY_BINS}m${MIN_TRAJECTORY_BINS}_t$(tag_value "$TRIM_FRACTION")_dc$(tag_value "$DESCRIPTOR_MIN_COSINE")_h${TRIM_HIST_BINS}_ref${FUSION_REFERENCE_MODE}_${MASK_POLICY}"
# Historical tags did not encode the validation split, so temporal-block
# outputs could collide with the frozen stratified protocol.  Version the root
# explicitly and verify every reusable artifact below.
ROBUST_PROTOCOL_VERSION="v2_split${SPLIT_MODE}_seed${SPLIT_SEED}_fullres_native_uncapped"
DEFAULT_RUN_ROOT="$EXPERIMENT_ROOT/$SCENE/${TAG}_${ROBUST_PROTOCOL_VERSION}"
if [[ "$TRAIN_SEED" != "2026" ]]; then
  DEFAULT_RUN_ROOT="${DEFAULT_RUN_ROOT}_trainseed${TRAIN_SEED}"
fi
RUN_ROOT="${LAFGS_ROBUST_RUN_ROOT:-$DEFAULT_RUN_ROOT}"
BOOTSTRAP_DIR="${LAFGS_ROBUST_BOOTSTRAP_SOURCE_DIR:-$RUN_ROOT/bootstrap}"
BOOTSTRAP_IDS="$BOOTSTRAP_DIR/sampled_idx.pkl"
BOOTSTRAP_META="$BOOTSTRAP_DIR/landmark_meta.pt"
BOOTSTRAP_STATE="$BOOTSTRAP_DIR/0_lafgs_map_state.pt"
RESIDUAL_DIR="$RUN_ROOT/residual_${RESIDUAL_STEPS}"
RESULT_ROOT="$RUN_ROOT/results"
CONFIG_ROOT="$RUN_ROOT/configs"
LOG_ROOT="$RUN_ROOT/logs"
STDLOC_RESULTS_ROOT="$RUN_ROOT/stdloc_results"
MANIFEST="$RUN_ROOT/study_manifest.json"
RESIDUAL_SELECTION="$RESULT_ROOT/residual_selection_safety.json"
VISIBILITY_CACHE="${LAFGS_ROBUST_VISIBILITY_CACHE_PATH:-$RUN_ROOT/visibility_${LANDMARK_BUDGET}_native.pt}"
BOOTSTRAP_VALIDATION_REF="${LAFGS_ROBUST_BOOTSTRAP_VALIDATION_REF:-}"

DEFAULT_QUERY_CACHE="$RUN_ROOT/query_cache_native_fullres_k${NATIVE_KEYPOINTS}.pt"
if [[ "$SCENE" == "OldHospital" ]]; then
  DEFAULT_QUERY_CACHE="$REFERENCE_ROOT/OldHospital/ulfparity_native20k_s128_k2048_v2/query_cache_native_fullres_k2048.pt"
fi
QUERY_CACHE="${LAFGS_ROBUST_QUERY_CACHE_PATH:-$DEFAULT_QUERY_CACHE}"

export CUDA_VISIBLE_DEVICES="$GPU"
export CUDA_HOME=/usr/local/cuda-11.8
export PATH="/root/miniconda3/envs/cybersim_agent/bin:$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="/root/miniconda3/envs/cybersim_agent/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$REPO_ROOT"
export PYTHONHASHSEED="$TRAIN_SEED"
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export STDLOC_CAMERA_LOADER_WORKERS="$CAMERA_LOADER_WORKERS"
export STDLOC_RESULTS_ROOT

mkdir -p "$RUN_ROOT" "$CONFIG_ROOT" "$LOG_ROOT" "$RESULT_ROOT" "$STDLOC_RESULTS_ROOT"
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

evaluation_ref_for_label() {
  local label="$1"
  if [[ "$label" == "bootstrap" && -n "$BOOTSTRAP_VALIDATION_REF" ]]; then
    printf '%s\n' "$BOOTSTRAP_VALIDATION_REF"
  else
    printf '%s\n' "$RESULT_ROOT/${label}_validation.results_path"
  fi
}

# File existence alone is insufficient because prior robust runs reused the
# same path for temporal_block and stratified_temporal_block.  These checks
# keep state and validation reuse fail-closed under the frozen protocol.
verify_state_protocol() {
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
config = dict(torch.load(state_path, map_location='cpu').get('config', {}))
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

exact('split_mode', expected_split_mode)
numeric('split_seed', int(expected_split_seed))
exact('query_feature_contract', 'native_resized_input')
exact('observation_source', 'native')
numeric('native_sparse_keypoint_count', int(expected_keypoints))
numeric('native_anchor_aux_weight', 0.0)
if bool(config.get('native_outcome_mode')) != bool(int(expected_native_outcome)):
    errors.append('native_outcome_mode does not match this stage')
for key in ('mv_weight', 'local_weight', 'dustbin_weight', 'geometry_weight', 'pose_weight'):
    numeric(key, 0.0)
numeric('retrieval_weight', float(expected_retrieval_weight))
numeric('trust_weight', float(expected_trust_weight))
if errors:
    raise SystemExit(f"Refusing to reuse robust state {label}: " + '; '.join(errors))
print(f'Verified frozen robust state protocol for {label}')
PY
}

verify_eval_protocol() {
  local label="$1"
  local ref="$2"
  require_file "$ref"
  "$PYTHON" - "$label" "$ref" "$SPLIT_MODE" "$SPLIT_SEED" "$LONGEST_EDGE" <<'PY'
import json
import sys
from pathlib import Path

label, ref_path, expected_split_mode, expected_split_seed, expected_longest_edge = sys.argv[1:]
summary_path = Path(Path(ref_path).read_text().strip()) / 'results_summary.json'
if not summary_path.is_file():
    raise SystemExit(f'Missing results_summary.json for {label}: {summary_path}')
protocol = dict(json.loads(summary_path.read_text()).get('evaluation_protocol', {}))
candidate = dict(protocol.get('candidate_split', {}))
errors = []
if protocol.get('evaluation_camera_subset') != 'candidate_validation':
    errors.append('evaluation_camera_subset is not candidate_validation')
if candidate.get('mode') != expected_split_mode:
    errors.append(f"candidate split mode {candidate.get('mode')!r} != {expected_split_mode!r}")
if candidate.get('seed') != int(expected_split_seed):
    errors.append(f"candidate split seed {candidate.get('seed')!r} != {expected_split_seed!r}")
if candidate.get('direct_holdout') is not True:
    errors.append('candidate validation is not a direct holdout')
if protocol.get('longest_edge') != int(expected_longest_edge):
    errors.append(f"longest_edge {protocol.get('longest_edge')!r} != {expected_longest_edge!r}")
if errors:
    raise SystemExit(f"Refusing to reuse robust validation {label}: " + '; '.join(errors))
print(f'Verified frozen robust validation protocol for {label}')
PY
}

verify_bootstrap_eval_binding() {
  local ref="$1"
  local state="$2"
  require_file "$ref"
  require_file "$state"
  "$PYTHON" - "$ref" "$state" <<'PY'
import json
import sys
from pathlib import Path

ref_path, state_path = map(Path, sys.argv[1:])
summary_path = Path(ref_path.read_text().strip()) / "results_summary.json"
if not summary_path.is_file():
    raise SystemExit(f"Missing bootstrap results summary: {summary_path}")
provenance = json.loads(summary_path.read_text()).get("artifact_provenance", {})
active_state = provenance.get("landmark_feature_override_path")
if not active_state:
    raise SystemExit("Bootstrap validation does not record its active feature state")
if Path(active_state).resolve() != state_path.resolve():
    raise SystemExit(
        "Bootstrap validation feature state does not match the requested source: "
        f"evaluation={Path(active_state).resolve()} source={state_path.resolve()}"
    )
print(f"Verified bootstrap validation state binding: {state_path}")
PY
}

write_manifest() {
  "$PYTHON" - "$MANIFEST" <<PY
import json
from pathlib import Path

payload = {
    "schema_version": 2,
    "purpose": "validation_only_robust_kcs_gwff_mask_ablation",
    "test_evaluation_forbidden": True,
    "scene": "${SCENE}",
    "formal_deployment_protocol": {
        "longest_edge": ${LONGEST_EDGE},
        "native_keypoints": ${NATIVE_KEYPOINTS},
        "sparse_frontend": "ulfloc_native",
        "topk": 1,
        "cosine_threshold": ${MATCH_THRESHOLD},
        "max_matches_per_landmark": ${MAX_MATCHES_PER_LANDMARK},
        "reprojection_error_px": ${EVAL_REPROJECTION_PX},
        "candidate_split_mode": "${SPLIT_MODE}",
        "candidate_split_seed": ${SPLIT_SEED},
        "robust_protocol_version": "${ROBUST_PROTOCOL_VERSION}",
    },
    "initializer": {
        "scaffold_mode": "ulf_robust_consensus",
        "initialization_mode": "ulf_robust_geometry",
        "landmark_budget": ${LANDMARK_BUDGET},
        "support_views": ${SUPPORT_VIEWS},
        "support_view_sampling": "${SUPPORT_SAMPLING}",
        "support_mask_policy": "${MASK_POLICY}",
        "minimum_visible_views": ${MIN_VISIBLE_VIEWS},
        "minimum_votes": ${MIN_VOTES},
        "minimum_consensus_rate": ${MIN_RATE},
        "view_bins": ${VIEW_BINS},
        "minimum_distinct_view_bins": ${MIN_VIEW_BINS},
        "trajectory_bins": ${TRAJECTORY_BINS},
        "minimum_distinct_trajectory_bins": ${MIN_TRAJECTORY_BINS},
        "allow_nonconsensus_fallback": False,
        "descriptor_trim_fraction": ${TRIM_FRACTION},
        "descriptor_min_cosine": ${DESCRIPTOR_MIN_COSINE},
        "descriptor_trim_histogram_bins": ${TRIM_HIST_BINS},
        "fusion_reference_mode": "${FUSION_REFERENCE_MODE}",
    },
    "inputs": {
        "model_root": str(Path("${MODEL_ROOT}").resolve()),
        "source_ply": str(Path("${SOURCE_PLY}").resolve()),
        "query_cache": str(Path("${QUERY_CACHE}").resolve()),
    },
    "runtime": {
        "camera_loader_workers": ${CAMERA_LOADER_WORKERS},
    },
    "optimization": {
        "train_seed": ${TRAIN_SEED},
        "bootstrap_source_dir": str(Path("${BOOTSTRAP_DIR}").resolve()),
        "bootstrap_validation_ref": (
            str(Path("${BOOTSTRAP_VALIDATION_REF}").resolve())
            if "${BOOTSTRAP_VALIDATION_REF}" else None
        ),
        "visibility_cache": str(Path("${VISIBILITY_CACHE}").resolve()),
    },
    "residual": {
        "steps": ${RESIDUAL_STEPS},
        "objective": "pure_native_keep_swap_miss_reject",
        "selection": "validation_only_safety",
    },
}
path = Path("${MANIFEST}")
path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY
}

bootstrap() {
  require_file "$SOURCE_PLY"
  require_file "$MODEL_ROOT/artifact_provenance.json"
  write_manifest
  if [[ -f "$BOOTSTRAP_STATE" && -f "$BOOTSTRAP_IDS" && -f "$BOOTSTRAP_META" ]]; then
    verify_state_protocol bootstrap "$BOOTSTRAP_STATE" 0 0 0
    echo "[Robust KCS/GWFF] Reusing bootstrap: $BOOTSTRAP_DIR"
    return
  fi
  run_logged bootstrap \
    "$PYTHON" train_lafgs_map.py \
    --model_path "$MODEL_ROOT" --source_path "$DATA_ROOT/$SCENE" \
    --images processed --data_device cpu --gaussian_type 2dgs --feature_type sp \
    --resolution 1 --longest_edge "$LONGEST_EDGE" --norm_before_render --load_iteration 30000 \
    --query_feature_contract native_resized_input \
    --output_dir "$BOOTSTRAP_DIR" \
    --scaffold_mode ulf_robust_consensus --generated_landmark_path "$BOOTSTRAP_DIR/robust_ids.pkl" \
    --regenerate_scaffold --scaffold_budget "$LANDMARK_BUDGET" --scaffold_seed 2026 \
    --initialization_mode ulf_robust_geometry \
    --ulf_consensus_keypoints "$NATIVE_KEYPOINTS" --ulf_consensus_radius_px 1.0 \
    --ulf_consensus_min_visible_views "$MIN_VISIBLE_VIEWS" --ulf_consensus_min_votes "$MIN_VOTES" \
    --ulf_consensus_min_rate "$MIN_RATE" --ulf_consensus_view_bins "$VIEW_BINS" \
    --ulf_consensus_min_distinct_view_bins "$MIN_VIEW_BINS" \
    --ulf_consensus_trajectory_bins "$TRAJECTORY_BINS" \
    --ulf_consensus_min_distinct_trajectory_bins "$MIN_TRAJECTORY_BINS" \
    --no-ulf_consensus_allow_nonconsensus_fallback \
    --ulf_support_view_sampling "$SUPPORT_SAMPLING" --ulf_support_mask_policy "$MASK_POLICY" \
    --ulf_consensus_max_views "$SUPPORT_VIEWS" --ulf_fusion_max_views "$SUPPORT_VIEWS" \
    --ulf_fusion_min_cosine 0 --ulf_fusion_descriptor_trim_fraction "$TRIM_FRACTION" \
    --ulf_fusion_descriptor_min_cosine "$DESCRIPTOR_MIN_COSINE" \
    --ulf_fusion_trim_histogram_bins "$TRIM_HIST_BINS" \
    --ulf_fusion_reference_mode "$FUSION_REFERENCE_MODE" \
    --query_cache_path "$QUERY_CACHE" --query_cache_policy reuse_or_build \
    --visibility_cache_path "$VISIBILITY_CACHE" --visibility_mode rasterizer \
    --objective hard --observation_source native --native_keypoint_count "$NATIVE_KEYPOINTS" \
    --native_association_radius_px 2 --native_sampling_mode detector_grid \
    --native_anchor_aux_weight 0 --no-native_outcome_mode \
    --steps 0 --save_steps 0 --distill_budget 0 \
    --validation_ratio "$VALIDATION_RATIO" --split_mode "$SPLIT_MODE" --split_seed "$SPLIT_SEED" \
    --train_seed "$TRAIN_SEED" --max_observations 512 --validation_observations 512 \
    --mv_weight 0 --retrieval_weight 0 --trust_weight 0 --local_weight 0 \
    --dustbin_weight 0 --geometry_weight 0 --pose_weight 0 --pose_gradient_mode off
  require_file "$BOOTSTRAP_STATE"
  require_file "$BOOTSTRAP_IDS"
  require_file "$BOOTSTRAP_META"
  verify_state_protocol bootstrap "$BOOTSTRAP_STATE" 0 0 0
}

make_eval_config() {
  local label="$1"
  local state="$2"
  local cfg="$CONFIG_ROOT/${label}_validation.yaml"
  "$PYTHON" scripts/make_stdloc_eval_cfg.py \
    --base_cfg configs/stdloc_cambridge.yaml --output "$cfg" \
    --artifact_model_path "$MODEL_ROOT" \
    --detector_folder ulfloc_native_no_detector --detector_iters 0 \
    --landmark_path "$BOOTSTRAP_IDS" --landmark_meta_path "$BOOTSTRAP_META" \
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
  local state="$2"
  local ref
  ref="$(evaluation_ref_for_label "$label")"
  if [[ -f "$ref" ]]; then
    verify_eval_protocol "$label" "$ref"
    if [[ "$label" == "bootstrap" ]]; then
      verify_bootstrap_eval_binding "$ref" "$BOOTSTRAP_STATE"
    fi
    echo "[Robust KCS/GWFF] Reusing protocol-matched validation evaluation: $label"
    return
  fi
  local cfg
  cfg="$(make_eval_config "$label" "$state")"
  run_logged "${label}_validation" \
    "$PYTHON" stdloc.py \
    --model_path "$MODEL_ROOT" --source_path "$DATA_ROOT/$SCENE" \
    --images processed --data_device cpu --gaussian_type 2dgs --feature_type sp \
    --resolution 1 --longest_edge "$LONGEST_EDGE" --norm_before_render --iteration 30000 \
    --cfg "$cfg" --prefix "lafgs-v2-robust-${SCENE}-${MASK_POLICY}-${label}-validation" --sparse_only \
    --evaluation_camera_subset candidate_validation --candidate_direct_validation_holdout \
    --candidate_validation_ratio "$VALIDATION_RATIO" \
    --candidate_split_mode "$SPLIT_MODE" --candidate_split_seed "$SPLIT_SEED"
  local output_path
  output_path="$(sed -n 's/^Output path: //p' "$LOG_ROOT/${label}_validation.log" | tail -n 1)"
  if [[ -z "$output_path" || ! -f "$output_path/results_summary.json" ]]; then
    echo "Robust KCS/GWFF validation did not produce results_summary.json for $label" >&2
    exit 1
  fi
  printf '%s\n' "$output_path" > "$ref"
  verify_eval_protocol "$label" "$ref"
  if [[ "$label" == "bootstrap" ]]; then
    verify_bootstrap_eval_binding "$ref" "$BOOTSTRAP_STATE"
  fi
}

result_summary() {
  local label="$1"
  local ref
  ref="$(evaluation_ref_for_label "$label")"
  require_file "$ref"
  local directory
  directory="$(<"$ref")"
  require_file "$directory/results_summary.json"
  printf '%s\n' "$directory/results_summary.json"
}

validate() {
  bootstrap
  run_eval bootstrap "$BOOTSTRAP_STATE"
}

verify_native_residual_state() {
  local state="$1"
  require_file "$state"
  verify_state_protocol residual "$state" 1 1 0.02
  "$PYTHON" - "$state" "$TRAIN_SEED" <<'PY'
import json
import sys
from pathlib import Path

import torch

state_path, expected_train_seed = sys.argv[1:]
state = torch.load(state_path, map_location="cpu")
config = state.get("config", {})
if str(config.get("observation_source")) != "native":
    raise SystemExit("robust residual must use native sparse observations")
if not bool(config.get("native_outcome_mode")):
    raise SystemExit("robust residual must enable keep/swap/miss/reject")
for name in ("native_anchor_aux_weight", "mv_weight", "local_weight", "dustbin_weight", "geometry_weight", "pose_weight"):
    if abs(float(config.get(name, 0.0))) > 1e-12:
        raise SystemExit(f"robust residual unexpectedly enables {name}")
if abs(float(config.get("retrieval_weight", 0.0)) - 1.0) > 1e-12:
    raise SystemExit("robust residual must retain unit native retrieval weight")
if abs(float(config.get("trust_weight", 0.0)) - 0.02) > 1e-12:
    raise SystemExit("robust residual must retain trust weight 0.02")
manifest_path = Path(state_path).parent / "reproducibility_manifest.json"
if not manifest_path.is_file():
    raise SystemExit(f"robust residual is missing reproducibility manifest: {manifest_path}")
arguments = json.loads(manifest_path.read_text()).get("arguments", {})
if int(arguments.get("train_seed", -1)) != int(expected_train_seed):
    raise SystemExit(
        "robust residual optimization seed does not match the run namespace: "
        f"manifest={arguments.get('train_seed')!r} expected={expected_train_seed}"
    )
offset = torch.as_tensor(state.get("raw_anchor_offset"), dtype=torch.float32)
if offset.numel() and float(offset.abs().max()) > 1e-12:
    raise SystemExit("pure descriptor residual must not move the surface anchor")
print("Verified robust pure-native residual checkpoint")
PY
}

residual() {
  validate
  local final_state="$RESIDUAL_DIR/${RESIDUAL_STEPS}_lafgs_map_state.pt"
  if [[ -f "$final_state" ]]; then
    verify_native_residual_state "$final_state"
    echo "[Robust KCS/GWFF] Reusing pure-native residual: $RESIDUAL_DIR"
    return
  fi
  run_logged residual \
    "$PYTHON" train_lafgs_map.py \
    --model_path "$MODEL_ROOT" --source_path "$DATA_ROOT/$SCENE" \
    --images processed --data_device cpu --gaussian_type 2dgs --feature_type sp \
    --resolution 1 --longest_edge "$LONGEST_EDGE" --norm_before_render --load_iteration 30000 \
    --query_feature_contract native_resized_input \
    --output_dir "$RESIDUAL_DIR" --scaffold_mode file --landmark_path "$BOOTSTRAP_IDS" \
    --initial_state_path "$BOOTSTRAP_STATE" --initial_state_blend 1 --initial_state_alignment exact \
    --initialization_mode ulf_robust_geometry \
    --query_cache_path "$QUERY_CACHE" --query_cache_policy readonly \
    --visibility_cache_path "$VISIBILITY_CACHE" --visibility_mode rasterizer \
    --objective hard --observation_source native --native_keypoint_count "$NATIVE_KEYPOINTS" \
    --native_association_radius_px 2 --native_sampling_mode detector_grid \
    --generic_proposal_count 0 --generic_proposal_weight 0 --distill_budget 0 \
    --validation_ratio "$VALIDATION_RATIO" --split_mode "$SPLIT_MODE" --split_seed "$SPLIT_SEED" \
    --train_seed "$TRAIN_SEED" --max_observations 512 --validation_observations 512 \
    --native_anchor_aux_weight 0 --native_outcome_mode --native_nce_weight 0 \
    --native_keep_weight 1 --native_keep_margin 0.05 \
    --native_swap_weight 1 --native_swap_margin 0.05 \
    --native_miss_weight 1 --native_miss_margin 0.05 \
    --native_reject_weight 0.05 --native_reject_threshold "$MATCH_THRESHOLD" \
    --mv_weight 0 --retrieval_weight 1 --trust_weight 0.02 --local_weight 0 --dustbin_weight 0 \
    --geometry_weight 0 --pose_weight 0 --pose_gradient_mode off \
    --steps "$RESIDUAL_STEPS" --save_steps 500 1000 2500 "$RESIDUAL_STEPS" \
    --feature_lr 5e-5 --weight_decay 1e-4 --hypothesis_topk 32 \
    --positive_radius_px 2 --negative_radius_px 6 --log_interval 100
  require_file "$final_state"
  verify_native_residual_state "$final_state"
}

residual_validate() {
  residual
  run_eval bootstrap "$BOOTSTRAP_STATE"
  local step
  for step in 500 1000 2500 "$RESIDUAL_STEPS"; do
    local state="$RESIDUAL_DIR/${step}_lafgs_map_state.pt"
    if [[ -f "$state" ]]; then
      run_eval "residual_${step}" "$state"
    fi
  done
}

select_residual() {
  residual_validate
  if [[ -f "$RESIDUAL_SELECTION" ]]; then
    echo "[Robust KCS/GWFF] Reusing residual selection: $RESIDUAL_SELECTION"
    return
  fi
  local command=(
    "$PYTHON" scripts/select_lafgs_map_checkpoint.py
    --control_results "$(result_summary bootstrap)"
    --control_state "$BOOTSTRAP_STATE" --control_tag bootstrap --selection_mode safety
    --min_te_gain_cm 0.02 --metric_tolerance 1e-9 --mean_te_weight 0.05
    --max_recall_2m_drop 0.01 --max_recall_5cm_drop 0.01 --output "$RESIDUAL_SELECTION"
  )
  local step
  for step in 500 1000 2500 "$RESIDUAL_STEPS"; do
    local label="residual_${step}"
    local state="$RESIDUAL_DIR/${step}_lafgs_map_state.pt"
    if [[ -f "$state" && -f "$RESULT_ROOT/${label}_validation.results_path" ]]; then
      command+=(--candidate "$label" "$(result_summary "$label")" "$state")
    fi
  done
  run_logged residual_selection_safety "${command[@]}"
  require_file "$RESIDUAL_SELECTION"
}

case "$MODE" in
  bootstrap) bootstrap ;;
  validate) validate ;;
  residual) residual ;;
  residual_validate) residual_validate ;;
  select_residual) select_residual ;;
  all) select_residual ;;
esac
