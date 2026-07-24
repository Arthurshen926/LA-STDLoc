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
# Keep the runner default identical to the frozen 32K canonical bank.  A 0.1
# rate silently changes the KCS IDs relative to the established r0p01 control.
MIN_RATE="${LAFGS_ROBUST_MIN_RATE:-0.01}"
VIEW_BINS="${LAFGS_ROBUST_VIEW_BINS:-4}"
MIN_VIEW_BINS="${LAFGS_ROBUST_MIN_VIEW_BINS:-2}"
TRAJECTORY_BINS="${LAFGS_ROBUST_TRAJECTORY_BINS:-4}"
MIN_TRAJECTORY_BINS="${LAFGS_ROBUST_MIN_TRAJECTORY_BINS:-2}"
INDEPENDENT_BIN_SCORING="${LAFGS_ROBUST_INDEPENDENT_BIN_SCORING:-1}"
FUSION_VIEW_BINS="${LAFGS_ROBUST_FUSION_VIEW_BINS:-$VIEW_BINS}"
FUSION_EXACT_BIN_BALANCE="${LAFGS_ROBUST_FUSION_EXACT_BIN_BALANCE:-0}"
TRIM_FRACTION="${LAFGS_ROBUST_TRIM_FRACTION:-0.1}"
DESCRIPTOR_MIN_COSINE="${LAFGS_ROBUST_DESCRIPTOR_MIN_COSINE:--1.0}"
TRIM_HIST_BINS="${LAFGS_ROBUST_TRIM_HIST_BINS:-64}"
FUSION_REFERENCE_MODE="${LAFGS_ROBUST_FUSION_REFERENCE_MODE:-mean}"
# Adaptive GWFF is deliberately a separate protocol from a global bottom-
# quantile trim.  It only trims landmarks with an observed incompatible-view
# tail; stable or weakly observed landmarks keep their full support set.
ADAPTIVE_TRIM="${LAFGS_ROBUST_ADAPTIVE_TRIM:-0}"
ADAPTIVE_TRIM_MIN_FRACTION="${LAFGS_ROBUST_ADAPTIVE_TRIM_MIN_FRACTION:-0.0}"
ADAPTIVE_TRIM_MAX_FRACTION="${LAFGS_ROBUST_ADAPTIVE_TRIM_MAX_FRACTION:-0.20}"
ADAPTIVE_TRIM_TAIL_COSINE="${LAFGS_ROBUST_ADAPTIVE_TRIM_TAIL_COSINE:-0.75}"
ADAPTIVE_TRIM_MIN_OBSERVATIONS="${LAFGS_ROBUST_ADAPTIVE_TRIM_MIN_OBSERVATIONS:-4}"
# The formal adaptive schedule uses each landmark's own median/MAD. The
# historical absolute threshold remains available only as a named ablation.
ADAPTIVE_TRIM_MODE="${LAFGS_ROBUST_ADAPTIVE_TRIM_MODE:-relative_mad}"
ADAPTIVE_TRIM_MAD_SCALE="${LAFGS_ROBUST_ADAPTIVE_TRIM_MAD_SCALE:-2.5}"
# Reusing a frozen KCS ID set isolates GWFF changes and avoids paying for an
# identical full-support consensus pass in every fusion-only experiment.
LANDMARK_SOURCE_PATH="${LAFGS_ROBUST_LANDMARK_SOURCE_PATH:-}"
EVAL_REPROJECTION_PX=12
RESIDUAL_STEPS="${LAFGS_ROBUST_RESIDUAL_STEPS:-5000}"
# Residual profiles are part of the experimental protocol, not an incidental
# training detail.  Keep the historical pure native objective as the default
# for initializer ablations, while allowing the formal entrypoint to pin its
# false-attractor-aware profile explicitly.
NATIVE_KEEP_WEIGHT="${LAFGS_ROBUST_NATIVE_KEEP_WEIGHT:-1.0}"
NATIVE_KEEP_MARGIN="${LAFGS_ROBUST_NATIVE_KEEP_MARGIN:-0.05}"
NATIVE_KEEP_LOOSE_WEIGHT="${LAFGS_ROBUST_NATIVE_KEEP_LOOSE_WEIGHT:-0.0}"
NATIVE_KEEP_LOOSE_RADIUS_PX="${LAFGS_ROBUST_NATIVE_KEEP_LOOSE_RADIUS_PX:-4.0}"
NATIVE_KEEP_LOOSE_MARGIN="${LAFGS_ROBUST_NATIVE_KEEP_LOOSE_MARGIN:-0.025}"
NATIVE_SWAP_WEIGHT="${LAFGS_ROBUST_NATIVE_SWAP_WEIGHT:-1.0}"
NATIVE_SWAP_MARGIN="${LAFGS_ROBUST_NATIVE_SWAP_MARGIN:-0.05}"
NATIVE_MISS_WEIGHT="${LAFGS_ROBUST_NATIVE_MISS_WEIGHT:-1.0}"
NATIVE_MISS_MARGIN="${LAFGS_ROBUST_NATIVE_MISS_MARGIN:-0.05}"
NATIVE_REJECT_WEIGHT="${LAFGS_ROBUST_NATIVE_REJECT_WEIGHT:-0.0}"
NATIVE_REJECT_THRESHOLD="${LAFGS_ROBUST_NATIVE_REJECT_THRESHOLD:-$MATCH_THRESHOLD}"
NATIVE_GLOBAL_ATTRACTOR_WEIGHT="${LAFGS_ROBUST_NATIVE_GLOBAL_ATTRACTOR_WEIGHT:-0.0}"
NATIVE_GLOBAL_ATTRACTOR_MIN_INCOMING="${LAFGS_ROBUST_NATIVE_GLOBAL_ATTRACTOR_MIN_INCOMING:-4}"
NATIVE_GLOBAL_ATTRACTOR_SUPPORT_POWER="${LAFGS_ROBUST_NATIVE_GLOBAL_ATTRACTOR_SUPPORT_POWER:-0.5}"
NATIVE_GLOBAL_ATTRACTOR_MAX_SCORE="${LAFGS_ROBUST_NATIVE_GLOBAL_ATTRACTOR_MAX_SCORE:-4.0}"
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
case "$ADAPTIVE_TRIM" in
  0|1) ;;
  *) echo "LAFGS_ROBUST_ADAPTIVE_TRIM must be 0 or 1" >&2; exit 2 ;;
esac
case "$INDEPENDENT_BIN_SCORING" in
  0|1) ;;
  *) echo "LAFGS_ROBUST_INDEPENDENT_BIN_SCORING must be 0 or 1" >&2; exit 2 ;;
esac
case "$FUSION_EXACT_BIN_BALANCE" in
  0|1) ;;
  *) echo "LAFGS_ROBUST_FUSION_EXACT_BIN_BALANCE must be 0 or 1" >&2; exit 2 ;;
esac
if [[ "$ADAPTIVE_TRIM" == "1" ]]; then
  "$PYTHON" - "$TRIM_FRACTION" "$ADAPTIVE_TRIM_MIN_FRACTION" \
    "$ADAPTIVE_TRIM_MAX_FRACTION" "$ADAPTIVE_TRIM_TAIL_COSINE" \
    "$ADAPTIVE_TRIM_MIN_OBSERVATIONS" "$ADAPTIVE_TRIM_MODE" \
    "$ADAPTIVE_TRIM_MAD_SCALE" <<'PY'
import math
import sys

global_fraction, minimum, maximum, tail_cosine, minimum_observations = map(float, sys.argv[1:6])
mode = sys.argv[6]
mad_scale = float(sys.argv[7])
if not math.isclose(global_fraction, 0.0, rel_tol=0.0, abs_tol=1e-12):
    raise SystemExit(
        "adaptive GWFF requires LAFGS_ROBUST_TRIM_FRACTION=0; "
        "do not compose global and landmark-specific trim schedules"
    )
if not (0.0 <= minimum <= maximum < 1.0):
    raise SystemExit("adaptive GWFF fractions must satisfy 0 <= min <= max < 1")
if not -1.0 <= tail_cosine <= 1.0:
    raise SystemExit("adaptive GWFF tail cosine must be in [-1, 1]")
if not minimum_observations.is_integer() or minimum_observations < 1:
    raise SystemExit("adaptive GWFF minimum observations must be a positive integer")
if mode not in {"absolute", "relative_mad"}:
    raise SystemExit("adaptive GWFF mode must be absolute or relative_mad")
if not math.isfinite(mad_scale) or mad_scale <= 0.0:
    raise SystemExit("adaptive GWFF MAD scale must be finite and positive")
PY
fi
if [[ -n "$LANDMARK_SOURCE_PATH" ]]; then
  if [[ ! -f "$LANDMARK_SOURCE_PATH" ]]; then
    echo "LAFGS_ROBUST_LANDMARK_SOURCE_PATH does not exist: $LANDMARK_SOURCE_PATH" >&2
    exit 2
  fi
  LANDMARK_SOURCE_PATH="$(realpath "$LANDMARK_SOURCE_PATH")"
fi
"$PYTHON" - \
  "$NATIVE_KEEP_WEIGHT" "$NATIVE_KEEP_MARGIN" \
  "$NATIVE_KEEP_LOOSE_WEIGHT" "$NATIVE_KEEP_LOOSE_RADIUS_PX" "$NATIVE_KEEP_LOOSE_MARGIN" \
  "$NATIVE_SWAP_WEIGHT" "$NATIVE_SWAP_MARGIN" \
  "$NATIVE_MISS_WEIGHT" "$NATIVE_MISS_MARGIN" \
  "$NATIVE_REJECT_WEIGHT" "$NATIVE_REJECT_THRESHOLD" \
  "$NATIVE_GLOBAL_ATTRACTOR_WEIGHT" "$NATIVE_GLOBAL_ATTRACTOR_MIN_INCOMING" \
  "$NATIVE_GLOBAL_ATTRACTOR_SUPPORT_POWER" "$NATIVE_GLOBAL_ATTRACTOR_MAX_SCORE" <<'PY'
import math
import sys

values = list(map(float, sys.argv[1:]))
(
    keep_weight,
    keep_margin,
    loose_weight,
    loose_radius,
    loose_margin,
    swap_weight,
    swap_margin,
    miss_weight,
    miss_margin,
    reject_weight,
    reject_threshold,
    global_attractor_weight,
    global_attractor_min_incoming,
    global_attractor_support_power,
    global_attractor_max_score,
) = values
if not all(math.isfinite(value) for value in values):
    raise SystemExit("native residual profile values must be finite")
if min(keep_weight, loose_weight, swap_weight, miss_weight, reject_weight) < 0.0:
    raise SystemExit("native residual loss weights must be non-negative")
if min(keep_margin, loose_margin, swap_margin, miss_margin) < 0.0:
    raise SystemExit("native residual ranking margins must be non-negative")
if loose_radius < 2.0:
    raise SystemExit("native loose-keep radius must be at least the 2px positive radius")
if global_attractor_weight < 0.0 or global_attractor_support_power < 0.0:
    raise SystemExit("global-attractor weight and support power must be non-negative")
if not global_attractor_min_incoming.is_integer() or global_attractor_min_incoming < 1:
    raise SystemExit("global-attractor minimum incoming count must be a positive integer")
if global_attractor_max_score <= 0.0:
    raise SystemExit("global-attractor maximum score must be positive")
PY

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
ADAPTIVE_TRIM_TAG="adapt0"
if [[ "$ADAPTIVE_TRIM" == "1" ]]; then
  ADAPTIVE_TRIM_TAG="adapt1_${ADAPTIVE_TRIM_MODE}_mad$(tag_value "$ADAPTIVE_TRIM_MAD_SCALE")_min$(tag_value "$ADAPTIVE_TRIM_MIN_FRACTION")_max$(tag_value "$ADAPTIVE_TRIM_MAX_FRACTION")_tail$(tag_value "$ADAPTIVE_TRIM_TAIL_COSINE")_obs${ADAPTIVE_TRIM_MIN_OBSERVATIONS}"
fi
LANDMARK_SOURCE_TAG=""
if [[ -n "$LANDMARK_SOURCE_PATH" ]]; then
  LANDMARK_SOURCE_TAG="_ids$(sha256sum "$LANDMARK_SOURCE_PATH" | awk '{print substr($1, 1, 12)}')"
fi
RESIDUAL_PROFILE_TAG="nk$(tag_value "$NATIVE_KEEP_WEIGHT")m$(tag_value "$NATIVE_KEEP_MARGIN")_kl$(tag_value "$NATIVE_KEEP_LOOSE_WEIGHT")r$(tag_value "$NATIVE_KEEP_LOOSE_RADIUS_PX")m$(tag_value "$NATIVE_KEEP_LOOSE_MARGIN")_ns$(tag_value "$NATIVE_SWAP_WEIGHT")m$(tag_value "$NATIVE_SWAP_MARGIN")_nm$(tag_value "$NATIVE_MISS_WEIGHT")m$(tag_value "$NATIVE_MISS_MARGIN")_nr$(tag_value "$NATIVE_REJECT_WEIGHT")t$(tag_value "$NATIVE_REJECT_THRESHOLD")_ga$(tag_value "$NATIVE_GLOBAL_ATTRACTOR_WEIGHT")i${NATIVE_GLOBAL_ATTRACTOR_MIN_INCOMING}p$(tag_value "$NATIVE_GLOBAL_ATTRACTOR_SUPPORT_POWER")m$(tag_value "$NATIVE_GLOBAL_ATTRACTOR_MAX_SCORE")"
TAG="robustkcs_gwff${LANDMARK_BUDGET}${LANDMARK_SOURCE_TAG}_s${SUPPORT_VIEWS}_${SUPPORT_SAMPLING}_mv$(tag_value "$MIN_VISIBLE_VIEWS")_v$(tag_value "$MIN_VOTES")_r$(tag_value "$MIN_RATE")_vb${VIEW_BINS}m${MIN_VIEW_BINS}_tb${TRAJECTORY_BINS}m${MIN_TRAJECTORY_BINS}_ib${INDEPENDENT_BIN_SCORING}_fvb${FUSION_VIEW_BINS}eb${FUSION_EXACT_BIN_BALANCE}_t$(tag_value "$TRIM_FRACTION")_dc$(tag_value "$DESCRIPTOR_MIN_COSINE")_h${TRIM_HIST_BINS}_ref${FUSION_REFERENCE_MODE}_${ADAPTIVE_TRIM_TAG}_${MASK_POLICY}"
# Historical tags did not encode the validation split, so temporal-block
# outputs could collide with the frozen stratified protocol.  Version the root
# explicitly and verify every reusable artifact below.
ROBUST_PROTOCOL_VERSION="v4_exact_fusion_bins_split${SPLIT_MODE}_seed${SPLIT_SEED}_fullres_native_uncapped"
DEFAULT_RUN_ROOT="$EXPERIMENT_ROOT/$SCENE/${TAG}_${ROBUST_PROTOCOL_VERSION}"
if [[ "$TRAIN_SEED" != "2026" ]]; then
  DEFAULT_RUN_ROOT="${DEFAULT_RUN_ROOT}_trainseed${TRAIN_SEED}"
fi
RUN_ROOT="${LAFGS_ROBUST_RUN_ROOT:-$DEFAULT_RUN_ROOT}"
BOOTSTRAP_DIR="${LAFGS_ROBUST_BOOTSTRAP_SOURCE_DIR:-$RUN_ROOT/bootstrap}"
BOOTSTRAP_IDS="$BOOTSTRAP_DIR/sampled_idx.pkl"
BOOTSTRAP_META="$BOOTSTRAP_DIR/landmark_meta.pt"
BOOTSTRAP_STATE="$BOOTSTRAP_DIR/0_lafgs_map_state.pt"
RESIDUAL_DIR="$RUN_ROOT/residual_${RESIDUAL_STEPS}_${RESIDUAL_PROFILE_TAG}"
RESIDUAL_LABEL_PREFIX="residual_${RESIDUAL_PROFILE_TAG}"
RESULT_ROOT="$RUN_ROOT/results"
CONFIG_ROOT="$RUN_ROOT/configs"
LOG_ROOT="$RUN_ROOT/logs"
STDLOC_RESULTS_ROOT="$RUN_ROOT/stdloc_results"
MANIFEST="$RUN_ROOT/study_manifest_${RESIDUAL_PROFILE_TAG}.json"
# Pose is the deployment objective.  Keep a small, predeclared recall safety
# constraint, but do not require every diagnostic statistic to be monotonic.
# Version the selection file so an older strict-safety result is never reused.
RESIDUAL_SELECTION="$RESULT_ROOT/residual_selection_performance_v1_${RESIDUAL_PROFILE_TAG}.json"
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
    "$NATIVE_KEYPOINTS" "$VALIDATION_RATIO" "$FUSION_EXACT_BIN_BALANCE" <<'PY'
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
    expected_validation_ratio,
    expected_exact_fusion,
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
numeric('validation_ratio', float(expected_validation_ratio))
numeric('native_anchor_aux_weight', 0.0)
if label == 'bootstrap':
    exact(
        'ulf_fusion_exact_bin_balance',
        bool(int(expected_exact_fusion)),
    )
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

verify_bootstrap_landmark_source() {
  if [[ -z "$LANDMARK_SOURCE_PATH" ]]; then
    return
  fi
  require_file "$LANDMARK_SOURCE_PATH"
  require_file "$BOOTSTRAP_IDS"
  require_file "$BOOTSTRAP_STATE"
  "$PYTHON" - "$LANDMARK_SOURCE_PATH" "$BOOTSTRAP_IDS" "$BOOTSTRAP_STATE" \
    "$LANDMARK_BUDGET" <<'PY'
import pickle
import sys

import torch

source_path, emitted_path, state_path, expected_budget = sys.argv[1:]
with open(source_path, "rb") as handle:
    source = torch.as_tensor(pickle.load(handle), dtype=torch.long).reshape(-1)
with open(emitted_path, "rb") as handle:
    emitted = torch.as_tensor(pickle.load(handle), dtype=torch.long).reshape(-1)
state = torch.load(state_path, map_location="cpu")
state_indices = torch.as_tensor(state.get("landmark_indices"), dtype=torch.long).reshape(-1)
if source.numel() != int(expected_budget):
    raise SystemExit(
        f"reused landmark source count {source.numel()} != requested budget {expected_budget}"
    )
if not torch.equal(source, emitted):
    raise SystemExit("bootstrap emitted landmark IDs differ from the frozen source")
if not torch.equal(source, state_indices):
    raise SystemExit("bootstrap state landmark IDs differ from the frozen source")
print("Verified frozen KCS landmark-ID binding")
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
        "scaffold_mode": (
            "file_reused_kcs" if "${LANDMARK_SOURCE_PATH}" else "ulf_robust_consensus"
        ),
        "source_landmark_path": (
            str(Path("${LANDMARK_SOURCE_PATH}").resolve())
            if "${LANDMARK_SOURCE_PATH}" else None
        ),
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
        "independent_bin_scoring": bool(${INDEPENDENT_BIN_SCORING}),
        "fusion_view_bins": ${FUSION_VIEW_BINS},
        "fusion_exact_bin_balance": bool(${FUSION_EXACT_BIN_BALANCE}),
        "allow_nonconsensus_fallback": False,
        "descriptor_trim_fraction": ${TRIM_FRACTION},
        "descriptor_min_cosine": ${DESCRIPTOR_MIN_COSINE},
        "descriptor_trim_histogram_bins": ${TRIM_HIST_BINS},
        "adaptive_descriptor_trim": bool(${ADAPTIVE_TRIM}),
        "adaptive_descriptor_trim_min_fraction": ${ADAPTIVE_TRIM_MIN_FRACTION},
        "adaptive_descriptor_trim_max_fraction": ${ADAPTIVE_TRIM_MAX_FRACTION},
        "adaptive_descriptor_trim_tail_cosine": ${ADAPTIVE_TRIM_TAIL_COSINE},
        "adaptive_descriptor_trim_min_observations": ${ADAPTIVE_TRIM_MIN_OBSERVATIONS},
        "adaptive_descriptor_trim_mode": "${ADAPTIVE_TRIM_MODE}",
        "adaptive_descriptor_trim_mad_scale": ${ADAPTIVE_TRIM_MAD_SCALE},
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
        "profile_tag": "${RESIDUAL_PROFILE_TAG}",
        "keep_weight": ${NATIVE_KEEP_WEIGHT},
        "keep_margin": ${NATIVE_KEEP_MARGIN},
        "keep_loose_weight": ${NATIVE_KEEP_LOOSE_WEIGHT},
        "keep_loose_radius_px": ${NATIVE_KEEP_LOOSE_RADIUS_PX},
        "keep_loose_margin": ${NATIVE_KEEP_LOOSE_MARGIN},
        "swap_weight": ${NATIVE_SWAP_WEIGHT},
        "swap_margin": ${NATIVE_SWAP_MARGIN},
        "miss_weight": ${NATIVE_MISS_WEIGHT},
        "miss_margin": ${NATIVE_MISS_MARGIN},
        "reject_weight": ${NATIVE_REJECT_WEIGHT},
        "reject_threshold": ${NATIVE_REJECT_THRESHOLD},
        "global_attractor_weight": ${NATIVE_GLOBAL_ATTRACTOR_WEIGHT},
        "global_attractor_min_incoming": ${NATIVE_GLOBAL_ATTRACTOR_MIN_INCOMING},
        "global_attractor_support_power": ${NATIVE_GLOBAL_ATTRACTOR_SUPPORT_POWER},
        "global_attractor_max_score": ${NATIVE_GLOBAL_ATTRACTOR_MAX_SCORE},
        "selection": "validation_only_performance_v1",
        "primary_metric": "median_te_cm + 0.05 * mean_te_cm",
        "max_recall_2m_drop": 0.01,
        "max_recall_5cm_drop": 0.01,
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
    verify_bootstrap_landmark_source
    echo "[Robust KCS/GWFF] Reusing bootstrap: $BOOTSTRAP_DIR"
    return
  fi
  local scaffold_args
  if [[ -n "$LANDMARK_SOURCE_PATH" ]]; then
    scaffold_args=(
      --scaffold_mode file --landmark_path "$LANDMARK_SOURCE_PATH"
      --scaffold_budget "$LANDMARK_BUDGET"
    )
  else
    scaffold_args=(
      --scaffold_mode ulf_robust_consensus
      --generated_landmark_path "$BOOTSTRAP_DIR/robust_ids.pkl"
      --regenerate_scaffold --scaffold_budget "$LANDMARK_BUDGET" --scaffold_seed 2026
    )
  fi
  local fusion_args=(
    --ulf_fusion_min_cosine 0
    --ulf_fusion_view_bins "$FUSION_VIEW_BINS"
    --ulf_fusion_descriptor_trim_fraction "$TRIM_FRACTION"
    --ulf_fusion_descriptor_min_cosine "$DESCRIPTOR_MIN_COSINE"
    --ulf_fusion_trim_histogram_bins "$TRIM_HIST_BINS"
    --ulf_fusion_reference_mode "$FUSION_REFERENCE_MODE"
  )
  if [[ "$FUSION_EXACT_BIN_BALANCE" == "1" ]]; then
    fusion_args+=(--ulf_fusion_exact_bin_balance)
  else
    fusion_args+=(--no-ulf_fusion_exact_bin_balance)
  fi
  if [[ "$ADAPTIVE_TRIM" == "1" ]]; then
    fusion_args+=(
      --ulf_fusion_adaptive_trim
      --ulf_fusion_adaptive_trim_min_fraction "$ADAPTIVE_TRIM_MIN_FRACTION"
      --ulf_fusion_adaptive_trim_max_fraction "$ADAPTIVE_TRIM_MAX_FRACTION"
      --ulf_fusion_adaptive_trim_tail_cosine "$ADAPTIVE_TRIM_TAIL_COSINE"
      --ulf_fusion_adaptive_trim_min_observations "$ADAPTIVE_TRIM_MIN_OBSERVATIONS"
      --ulf_fusion_adaptive_trim_mode "$ADAPTIVE_TRIM_MODE"
      --ulf_fusion_adaptive_trim_mad_scale "$ADAPTIVE_TRIM_MAD_SCALE"
    )
  fi
  local independent_bin_args=(--no-ulf_consensus_independent_bin_scoring)
  if [[ "$INDEPENDENT_BIN_SCORING" == "1" ]]; then
    independent_bin_args=(--ulf_consensus_independent_bin_scoring)
  fi
  run_logged bootstrap \
    "$PYTHON" train_lafgs_map.py \
    --model_path "$MODEL_ROOT" --source_path "$DATA_ROOT/$SCENE" \
    --images processed --data_device cpu --gaussian_type 2dgs --feature_type sp \
    --resolution 1 --longest_edge "$LONGEST_EDGE" --norm_before_render --load_iteration 30000 \
    --query_feature_contract native_resized_input \
    --output_dir "$BOOTSTRAP_DIR" \
    "${scaffold_args[@]}" \
    --initialization_mode ulf_robust_geometry \
    --ulf_consensus_keypoints "$NATIVE_KEYPOINTS" --ulf_consensus_radius_px 1.0 \
    --ulf_consensus_min_visible_views "$MIN_VISIBLE_VIEWS" --ulf_consensus_min_votes "$MIN_VOTES" \
    --ulf_consensus_min_rate "$MIN_RATE" --ulf_consensus_view_bins "$VIEW_BINS" \
    --ulf_consensus_min_distinct_view_bins "$MIN_VIEW_BINS" \
    --ulf_consensus_trajectory_bins "$TRAJECTORY_BINS" \
    --ulf_consensus_min_distinct_trajectory_bins "$MIN_TRAJECTORY_BINS" \
    "${independent_bin_args[@]}" \
    --no-ulf_consensus_allow_nonconsensus_fallback \
    --ulf_support_view_sampling "$SUPPORT_SAMPLING" --ulf_support_mask_policy "$MASK_POLICY" \
    --ulf_consensus_max_views "$SUPPORT_VIEWS" --ulf_fusion_max_views "$SUPPORT_VIEWS" \
    "${fusion_args[@]}" \
    --query_cache_path "$QUERY_CACHE" --query_cache_policy reuse_or_build \
    --visibility_cache_path "$VISIBILITY_CACHE" --visibility_mode rasterizer \
    --objective hard --observation_source native --native_keypoint_count "$NATIVE_KEYPOINTS" \
    --native_association_radius_px 2 --native_sampling_mode detector_grid \
    --native_anchor_aux_weight 0 --no-native_outcome_mode \
    --steps 0 --save_steps 0 --distill_budget 0 \
    --validation_ratio "$VALIDATION_RATIO" --split_mode "$SPLIT_MODE" --split_seed "$SPLIT_SEED" \
    --train_seed "$TRAIN_SEED" --max_observations "$NATIVE_KEYPOINTS" \
    --validation_observations "$NATIVE_KEYPOINTS" \
    --mv_weight 0 --retrieval_weight 0 --trust_weight 0 --local_weight 0 \
    --dustbin_weight 0 --geometry_weight 0 --pose_weight 0 --pose_gradient_mode off
  require_file "$BOOTSTRAP_STATE"
  require_file "$BOOTSTRAP_IDS"
  require_file "$BOOTSTRAP_META"
  verify_state_protocol bootstrap "$BOOTSTRAP_STATE" 0 0 0
  verify_bootstrap_landmark_source
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

verify_eval_config_binding() {
  local cfg="$1"
  local state="$2"
  require_file "$cfg"
  require_file "$state"
  "$PYTHON" - "$cfg" "$state" <<'PY'
from pathlib import Path
import sys

import yaml

cfg_path = Path(sys.argv[1])
expected_state = Path(sys.argv[2]).resolve()
cfg = yaml.safe_load(cfg_path.read_text())
sparse = cfg.get("sparse", {})
actual_value = sparse.get("landmark_feature_override_path")
if not actual_value:
    raise SystemExit("evaluation config is missing landmark_feature_override_path")
actual_state = Path(str(actual_value)).resolve()
if actual_state != expected_state:
    raise SystemExit(
        "evaluation config override mismatch: "
        f"{actual_state} != {expected_state}"
    )
if not actual_state.is_file():
    raise SystemExit(f"evaluation config override does not exist: {actual_state}")
if not bool(sparse.get("override_landmark_features", False)):
    raise SystemExit("evaluation config does not enable landmark feature override")
print("Verified evaluation config checkpoint binding")
PY
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
  verify_eval_config_binding "$cfg" "$state"
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
  # ``stdloc.py`` announces a provisional output path while parsing the config,
  # then emits the timestamped directory that actually contains the result at
  # completion. Prefer the terminal result line so interrupted/retried runs
  # cannot bind a reference to the provisional directory.
  output_path="$(sed -n \
    -e 's/^Result are saved in //p' \
    -e 's/^Results are saved in //p' \
    -e 's/^Output path: //p' \
    "$LOG_ROOT/${label}_validation.log" | tail -n 1)"
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
  "$PYTHON" - "$state" "$TRAIN_SEED" \
    "$NATIVE_KEEP_WEIGHT" "$NATIVE_KEEP_MARGIN" \
    "$NATIVE_KEEP_LOOSE_WEIGHT" "$NATIVE_KEEP_LOOSE_RADIUS_PX" "$NATIVE_KEEP_LOOSE_MARGIN" \
    "$NATIVE_SWAP_WEIGHT" "$NATIVE_SWAP_MARGIN" \
    "$NATIVE_MISS_WEIGHT" "$NATIVE_MISS_MARGIN" \
    "$NATIVE_REJECT_WEIGHT" "$NATIVE_REJECT_THRESHOLD" \
    "$NATIVE_GLOBAL_ATTRACTOR_WEIGHT" "$NATIVE_GLOBAL_ATTRACTOR_MIN_INCOMING" \
    "$NATIVE_GLOBAL_ATTRACTOR_SUPPORT_POWER" "$NATIVE_GLOBAL_ATTRACTOR_MAX_SCORE" <<'PY'
import json
import sys
from pathlib import Path

import torch

(
    state_path,
    expected_train_seed,
    expected_keep_weight,
    expected_keep_margin,
    expected_loose_weight,
    expected_loose_radius,
    expected_loose_margin,
    expected_swap_weight,
    expected_swap_margin,
    expected_miss_weight,
    expected_miss_margin,
    expected_reject_weight,
    expected_reject_threshold,
    expected_global_attractor_weight,
    expected_global_attractor_min_incoming,
    expected_global_attractor_support_power,
    expected_global_attractor_max_score,
) = sys.argv[1:]
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
expected_numeric = {
    "native_keep_weight": expected_keep_weight,
    "native_keep_margin": expected_keep_margin,
    "native_keep_loose_weight": expected_loose_weight,
    "native_keep_loose_radius_px": expected_loose_radius,
    "native_keep_loose_margin": expected_loose_margin,
    "native_swap_weight": expected_swap_weight,
    "native_swap_margin": expected_swap_margin,
    "native_miss_weight": expected_miss_weight,
    "native_miss_margin": expected_miss_margin,
    "native_reject_weight": expected_reject_weight,
    "native_reject_threshold": expected_reject_threshold,
    "native_global_attractor_weight": expected_global_attractor_weight,
    "native_global_attractor_min_incoming": expected_global_attractor_min_incoming,
    "native_global_attractor_support_power": expected_global_attractor_support_power,
    "native_global_attractor_max_score": expected_global_attractor_max_score,
}
for name, expected in expected_numeric.items():
    actual = config.get(name)
    try:
        matches = abs(float(actual) - float(expected)) <= 1e-12
    except (TypeError, ValueError):
        matches = False
    if not matches:
        raise SystemExit(
            f"robust residual profile mismatch for {name}: {actual!r} != {expected!r}"
        )
if float(expected_global_attractor_weight) > 0.0:
    prior = config.get("native_global_attractor_prior")
    if not isinstance(prior, dict) or not bool(prior.get("enabled")):
        raise SystemExit("global-attractor residual is missing its train-only prior")
    prior_path = Path(str(prior.get("path", "")))
    if not prior_path.is_file():
        raise SystemExit(f"global-attractor prior is missing: {prior_path}")
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
    --native_keep_weight "$NATIVE_KEEP_WEIGHT" --native_keep_margin "$NATIVE_KEEP_MARGIN" \
    --native_keep_loose_weight "$NATIVE_KEEP_LOOSE_WEIGHT" \
    --native_keep_loose_radius_px "$NATIVE_KEEP_LOOSE_RADIUS_PX" \
    --native_keep_loose_margin "$NATIVE_KEEP_LOOSE_MARGIN" \
    --native_swap_weight "$NATIVE_SWAP_WEIGHT" --native_swap_margin "$NATIVE_SWAP_MARGIN" \
    --native_miss_weight "$NATIVE_MISS_WEIGHT" --native_miss_margin "$NATIVE_MISS_MARGIN" \
    --native_reject_weight "$NATIVE_REJECT_WEIGHT" --native_reject_threshold "$NATIVE_REJECT_THRESHOLD" \
    --native_global_attractor_weight "$NATIVE_GLOBAL_ATTRACTOR_WEIGHT" \
    --native_global_attractor_min_incoming "$NATIVE_GLOBAL_ATTRACTOR_MIN_INCOMING" \
    --native_global_attractor_support_power "$NATIVE_GLOBAL_ATTRACTOR_SUPPORT_POWER" \
    --native_global_attractor_max_score "$NATIVE_GLOBAL_ATTRACTOR_MAX_SCORE" \
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
      run_eval "${RESIDUAL_LABEL_PREFIX}_${step}" "$state"
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
    --control_state "$BOOTSTRAP_STATE" --control_tag bootstrap --selection_mode performance
    --min_te_gain_cm 0.02 --metric_tolerance 1e-9 --mean_te_weight 0.05
    --max_recall_2m_drop 0.01 --max_recall_5cm_drop 0.01 --output "$RESIDUAL_SELECTION"
  )
  local step
  for step in 500 1000 2500 "$RESIDUAL_STEPS"; do
    local label="${RESIDUAL_LABEL_PREFIX}_${step}"
    local state="$RESIDUAL_DIR/${step}_lafgs_map_state.pt"
    if [[ -f "$state" && -f "$RESULT_ROOT/${label}_validation.results_path" ]]; then
      command+=(--candidate "$label" "$(result_summary "$label")" "$state")
    fi
  done
  run_logged residual_selection_performance "${command[@]}"
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
