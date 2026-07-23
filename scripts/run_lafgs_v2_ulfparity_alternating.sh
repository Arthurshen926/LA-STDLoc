#!/usr/bin/env bash
set -euo pipefail

# Strict ULF-parity bootstrap followed by inference-aligned alternating map
# refinement. This runner deliberately does not reuse the legacy 640px/16K
# distillation pipeline: the 20K KCS/GWFF bank is kept intact through residual,
# bounded BA, and refresh.
#
# Protocol:
#   frozen external RGB 2DGS -> KCS/GWFF 20K bootstrap -> native outcome
#   residual -> fixed-descriptor native GT-clean bounded BA -> short refresh
#
# Every selection is made only on the direct held-out training-camera split.
# The test mode evaluates the selected state exactly once on the official test
# split. There is intentionally no final full-data refit.

if [[ $# -ne 3 ]]; then
  echo "Usage: $0 <scene> <gpu> <prepare|bootstrap|bootstrap_validate|residual|residual_validate|select_residual|ba|ba_validate|select_ba|refresh|refresh_validate|select_final|test|test_ba_selected|all>" >&2
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
  prepare|bootstrap|bootstrap_validate|residual|residual_validate|select_residual|ba|ba_validate|select_ba|refresh|refresh_validate|select_final|test|test_ba_selected|all) ;;
  *) echo "Unsupported mode: $MODE" >&2; exit 2 ;;
esac

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/root/miniconda3/envs/cybersim_agent/bin/python}"
DATA_ROOT="${CAMBRIDGE_DATA_ROOT:-/mnt/pool/sqy/Cambridge_stdloc}"
MATCHA_ROOT="${CAMBRIDGE_MATCHA_2DGS_ROOT:-/root/MAtCha/output_cambridge_full_retained_v2}"
EXPERIMENT_ROOT="${LAFGS_V2_ULFPARITY_ROOT:-/mnt/pool/sqy/stdloc_lafgs_v2_ulfparity_alternating_20260721}"
WRAPPER_ROOT="$EXPERIMENT_ROOT/matcha_wrappers"
SCENE_ROOT="$EXPERIMENT_ROOT/$SCENE"
MODEL_ROOT="$WRAPPER_ROOT/$SCENE"

# Canonical formal deployment protocol. These values are intentionally not
# tunable through this runner: prior 640px and per-landmark-cap evaluations
# changed the deployed candidate distribution by centimetres. Capacity and
# support-view studies may vary below, but every formal map is evaluated with
# this same full-resolution, uncapped native frontend.
FORMAL_PROTOCOL_ID="lafgs_v2_fullres_native_uncapped_v1"
FORMAL_LONGEST_EDGE=0
FORMAL_NATIVE_KEYPOINTS=2048
FORMAL_EVAL_PROFILE="ulfloc_parity"
FORMAL_NATIVE_MATCH_THRESHOLD=0
FORMAL_MAX_MATCHES_PER_LANDMARK=0

# ULF-Loc's public Cambridge sparse setup. Keeping this fixed makes the
# bootstrap gate interpretable rather than a resize or detector ablation.
LANDMARK_BUDGET="${LAFGS_ULF_LANDMARK_BUDGET:-20000}"
SUPPORT_VIEWS="${LAFGS_ULF_SUPPORT_VIEWS:-128}"
SUPPORT_VIEW_SAMPLING="${LAFGS_ULF_SUPPORT_VIEW_SAMPLING:-uniform}"
ULF_PARITY_KCS_MASK_POLICY="${LAFGS_ULF_PARITY_KCS_MASK_POLICY:-rgb_only}"
case "$ULF_PARITY_KCS_MASK_POLICY" in
  rgb_only|deployment_post_filter) ;;
  *)
    echo "Unsupported LAFGS_ULF_PARITY_KCS_MASK_POLICY: $ULF_PARITY_KCS_MASK_POLICY" >&2
    exit 2
    ;;
esac
if [[ -n "${LAFGS_ULF_NATIVE_KEYPOINTS:-}" && "${LAFGS_ULF_NATIVE_KEYPOINTS}" != "$FORMAL_NATIVE_KEYPOINTS" ]]; then
  echo "Formal protocol fixes LAFGS_ULF_NATIVE_KEYPOINTS=$FORMAL_NATIVE_KEYPOINTS" >&2
  exit 2
fi
NATIVE_KEYPOINTS="$FORMAL_NATIVE_KEYPOINTS"
VALIDATION_RATIO="${LAFGS_ULF_VALIDATION_RATIO:-0.2}"
# Formal runs use the sequence-stratified temporal holdout that produced the
# current reference result.  A legacy temporal_block comparison must opt in
# explicitly so it cannot be mixed into a capacity or initializer study.
SPLIT_MODE="${LAFGS_ULF_SPLIT_MODE:-stratified_temporal_block}"
SPLIT_SEED="${LAFGS_ULF_SPLIT_SEED:-2026}"
case "$SPLIT_MODE" in
  random|sequence_block|temporal_block|stratified_temporal_block) ;;
  *)
    echo "Unsupported LAFGS_ULF_SPLIT_MODE: $SPLIT_MODE" >&2
    exit 2
    ;;
esac
# Native-resolution Cambridge frames are large enough that multiple concurrent
# formal runs can exhaust /dev/shm through pin-memory DataLoader workers. The
# reproducible default is parent-process loading; callers can explicitly raise
# this for a single-run throughput experiment.
CAMERA_LOADER_WORKERS="${LAFGS_ULF_CAMERA_LOADER_WORKERS:-0}"
if ! [[ "$CAMERA_LOADER_WORKERS" =~ ^[0-9]+$ ]]; then
  echo "LAFGS_ULF_CAMERA_LOADER_WORKERS must be a non-negative integer" >&2
  exit 2
fi
# Keep the validation partition fixed while allowing an explicitly namespaced
# optimization seed for repeated residual/BA/refresh runs.
TRAIN_SEED="${LAFGS_ULF_TRAIN_SEED:-2026}"
BOOTSTRAP_GATE_CM="${LAFGS_ULF_BOOTSTRAP_GATE_CM:-20.0}"
RESIDUAL_STEPS="${LAFGS_ULF_RESIDUAL_STEPS:-5000}"
BA_STEPS="${LAFGS_ULF_BA_STEPS:-1500}"
REFRESH_STEPS="${LAFGS_ULF_REFRESH_STEPS:-1000}"
BA_TANGENT_BOUND_M="${LAFGS_ULF_BA_TANGENT_BOUND_M:-0.003}"
BA_NORMAL_BOUND_M="${LAFGS_ULF_BA_NORMAL_BOUND_M:-0.0015}"
# Disabled by default so the original position-only BA remains reproducible.
# Positive values enforce a rendered-depth front-surface compatibility gate.
BA_DEPTH_ABS_TOLERANCE="${LAFGS_ULF_BA_DEPTH_ABS_TOLERANCE:-0}"
BA_DEPTH_REL_TOLERANCE="${LAFGS_ULF_BA_DEPTH_REL_TOLERANCE:-0}"
EVAL_REPROJECTION_PX="${LAFGS_ULF_EVAL_REPROJECTION_PX:-12.0}"
# ``ulfloc_native`` calls SuperPoint.detectAndCompute directly, whose native
# NMS radius is four.  This setting is retained only for the unused detector
# frontend compatibility path and must not be interpreted as the ULF NMS.
EVAL_NMS="${LAFGS_ULF_EVAL_NMS:-2}"
PNP_VOXEL_M="${LAFGS_ULF_PNP_VOXEL_M:-1.0}"
TASK_TRANSLATION_SCALE_M="${LAFGS_ULF_TASK_TRANSLATION_SCALE_M:-0.07160573943725686}"
# Native rejection is trained in exactly the deployed cosine-score space. A
# nonzero threshold or landmark cap belongs to a separately named ablation,
# never a formal LaFGS result.
if [[ -n "${LAFGS_ULF_EVAL_PROFILE:-}" && "${LAFGS_ULF_EVAL_PROFILE}" != "$FORMAL_EVAL_PROFILE" ]]; then
  echo "Formal protocol fixes LAFGS_ULF_EVAL_PROFILE=$FORMAL_EVAL_PROFILE" >&2
  exit 2
fi
if [[ -n "${LAFGS_ULF_NATIVE_MATCH_THRESHOLD:-}" && "${LAFGS_ULF_NATIVE_MATCH_THRESHOLD}" != "0" ]]; then
  echo "Formal protocol fixes LAFGS_ULF_NATIVE_MATCH_THRESHOLD=0" >&2
  exit 2
fi
if [[ -n "${LAFGS_ULF_MAX_MATCHES_PER_LANDMARK:-}" && "${LAFGS_ULF_MAX_MATCHES_PER_LANDMARK}" != "0" ]]; then
  echo "Formal protocol fixes LAFGS_ULF_MAX_MATCHES_PER_LANDMARK=0" >&2
  exit 2
fi
EVAL_PROFILE="$FORMAL_EVAL_PROFILE"
NATIVE_MATCH_THRESHOLD="$FORMAL_NATIVE_MATCH_THRESHOLD"
MAX_MATCHES_PER_LANDMARK="$FORMAL_MAX_MATCHES_PER_LANDMARK"

case "$SCENE" in
  GreatCourt) MATCHA_RUN="GreatCourt_n20_long_masked_retrain_retry" ;;
  KingsCollege) MATCHA_RUN="KingsCollege_n20_long_masked_retrain" ;;
  OldHospital) MATCHA_RUN="OldHospital_n20_long_masked_retrain_retry" ;;
  ShopFacade) MATCHA_RUN="ShopFacade_n20_long_masked_retrain" ;;
  StMarysChurch) MATCHA_RUN="StMarysChurch_n20_long_masked_retrain" ;;
esac
SOURCE_PLY="$MATCHA_ROOT/$MATCHA_RUN/free_gaussians/point_cloud/iteration_30000/point_cloud.ply"

NATIVE_MATCH_THRESHOLD_TAG="${NATIVE_MATCH_THRESHOLD//-/m}"
NATIVE_MATCH_THRESHOLD_TAG="${NATIVE_MATCH_THRESHOLD_TAG//./p}"
MAX_MATCHES_PER_LANDMARK_TAG="${MAX_MATCHES_PER_LANDMARK//-/m}"
MAX_MATCHES_PER_LANDMARK_TAG="${MAX_MATCHES_PER_LANDMARK_TAG//./p}"
# The strict formal default removes the residual-stage source-descriptor
# regression.  The optional low-weight anchor auxiliary is exposed explicitly
# so a control cannot be mislabeled as pure-native in its manifest.
PROTOCOL_VERSION="v8_fullres_native_uncapped_pure_native"
# Selection is a safety boundary, not an additional pose-only tuning stage.
# The formal default requires a checkpoint to improve translation while not
# degrading rotation, raw GT cleanliness, inlier cleanliness, or translation
# pose information.  The old deployment-pose selector remains explicit for
# backwards-compatible ablations only.
SELECTION_MODE="${LAFGS_ULF_SELECTION_MODE:-safety}"
case "$SELECTION_MODE" in
  safety|performance) ;;
  *)
    echo "LAFGS_ULF_SELECTION_MODE must be safety or performance; got $SELECTION_MODE" >&2
    exit 2
    ;;
esac
# Control identity is part of the selection protocol.  Bump the namespace so
# a legacy artifact that labelled a residual control as "control_strong" can
# never be reused as the provenance for a new official test.
SELECTION_VERSION="${LAFGS_ULF_SELECTION_VERSION:-v7_${SELECTION_MODE}_control_identity_${PROTOCOL_VERSION}}"
NATIVE_OBSERVATION_SOURCE="${LAFGS_ULF_NATIVE_OBSERVATION_SOURCE:-native}"
NATIVE_ANCHOR_AUX_WEIGHT="${LAFGS_ULF_NATIVE_ANCHOR_AUX_WEIGHT:-0}"
NATIVE_MV_WEIGHT="${LAFGS_ULF_NATIVE_MV_WEIGHT:-0}"
TAG="ulfparity_native${LANDMARK_BUDGET}_s${SUPPORT_VIEWS}_k${NATIVE_KEYPOINTS}_${EVAL_PROFILE}_tau${NATIVE_MATCH_THRESHOLD_TAG}_cap${MAX_MATCHES_PER_LANDMARK_TAG}_${PROTOCOL_VERSION}"
if [[ "$SUPPORT_VIEW_SAMPLING" != "uniform" ]]; then
  TAG="ulfparity_native${LANDMARK_BUDGET}_s${SUPPORT_VIEWS}_${SUPPORT_VIEW_SAMPLING}_k${NATIVE_KEYPOINTS}_${EVAL_PROFILE}_tau${NATIVE_MATCH_THRESHOLD_TAG}_cap${MAX_MATCHES_PER_LANDMARK_TAG}_${PROTOCOL_VERSION}"
fi
if [[ "$ULF_PARITY_KCS_MASK_POLICY" != "rgb_only" ]]; then
  TAG="${TAG}_kcsmask${ULF_PARITY_KCS_MASK_POLICY}"
fi
if [[ "$SPLIT_MODE" != "temporal_block" ]]; then
  TAG="${TAG}_split${SPLIT_MODE}"
fi
if [[ "$TRAIN_SEED" != "2026" ]]; then
  TAG="${TAG}_seed${TRAIN_SEED}"
fi
RUN_ROOT="$SCENE_ROOT/$TAG"
# A residual-objective ablation must start from exactly the same fixed KCS/GWFF
# bank.  Reusing a completed bootstrap avoids rebuilding a different random
# bank and makes the comparison causal.  It is intentionally opt-in: a normal
# formal run still writes its bootstrap under its own run root.
BOOTSTRAP_DIR="${LAFGS_ULF_BOOTSTRAP_SOURCE_DIR:-$RUN_ROOT/bootstrap}"
RESIDUAL_DIR="$RUN_ROOT/residual_${RESIDUAL_STEPS}"
BA_DIR="$RUN_ROOT/ba_${BA_STEPS}"
BA_TANGENT_BOUND_TAG="${BA_TANGENT_BOUND_M//./p}"
BA_NORMAL_BOUND_TAG="${BA_NORMAL_BOUND_M//./p}"
BA_DEPTH_ABS_TAG="${BA_DEPTH_ABS_TOLERANCE//./p}"
BA_DEPTH_REL_TAG="${BA_DEPTH_REL_TOLERANCE//./p}"
BA_DEPTH_GATE_TAG=""
if [[ "$BA_DEPTH_ABS_TOLERANCE" != "0" || "$BA_DEPTH_REL_TOLERANCE" != "0" ]]; then
  BA_DEPTH_GATE_TAG="_dg_a${BA_DEPTH_ABS_TAG}_r${BA_DEPTH_REL_TAG}"
  BA_DIR="${BA_DIR}${BA_DEPTH_GATE_TAG}"
fi
# BA candidates with different association gates are different experiments.
# Namespace every downstream evaluation and selector artifact by the gate so a
# gated BA can never silently reuse an ungated result with the same step tag.
BA_STAGE_TAG="${BA_DEPTH_GATE_TAG:-_ungated}"
# The suffix makes a stale descriptor-only refresh made with parser defaults
# impossible to reuse as though it had preserved the selected BA geometry.
REFRESH_DIR="$RUN_ROOT/refresh_${REFRESH_STEPS}_t${BA_TANGENT_BOUND_TAG}_n${BA_NORMAL_BOUND_TAG}${BA_DEPTH_GATE_TAG}"
VALIDATION_ROOT="$RUN_ROOT/validation"
CONFIG_ROOT="$VALIDATION_ROOT/configs"
LOG_ROOT="$RUN_ROOT/logs"
RESULT_ROOT="$VALIDATION_ROOT/results"
STDLOC_RESULTS_ROOT="$RUN_ROOT/stdloc_results"
# A formal rerun may reuse a cache only when train_lafgs_map.py accepts its
# frontend manifest.  Keeping the cache path overridable avoids duplicating a
# 28GB native-query cache while still failing closed on protocol mismatch.
QUERY_CACHE="${LAFGS_ULF_QUERY_CACHE_PATH:-$RUN_ROOT/query_cache_native_fullres_k${NATIVE_KEYPOINTS}.pt}"
VISIBILITY_CACHE="${LAFGS_ULF_VISIBILITY_CACHE_PATH:-$RUN_ROOT/visibility_${LANDMARK_BUDGET}_native.pt}"
BOOTSTRAP_IDS="$BOOTSTRAP_DIR/sampled_idx.pkl"
BOOTSTRAP_META="$BOOTSTRAP_DIR/landmark_meta.pt"
BOOTSTRAP_STATE="$BOOTSTRAP_DIR/0_lafgs_map_state.pt"
RESIDUAL_SELECTION="$VALIDATION_ROOT/residual_selection_${SELECTION_VERSION}.json"
PROTOCOL_MANIFEST="$RUN_ROOT/protocol_manifest${BA_STAGE_TAG}.json"
BA_SELECTION="$VALIDATION_ROOT/ba_selection_${SELECTION_VERSION}${BA_STAGE_TAG}.json"
FINAL_SELECTION="$VALIDATION_ROOT/final_selection_${SELECTION_VERSION}${BA_STAGE_TAG}.json"

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

validate_native_match_threshold() {
  "$PYTHON" - "$NATIVE_MATCH_THRESHOLD" <<'PY'
import math
import sys

value = float(sys.argv[1])
if not math.isfinite(value) or value < -1.0 or value > 1.0:
    raise SystemExit(
        "LAFGS_ULF_NATIVE_MATCH_THRESHOLD must be a finite cosine score in [-1, 1], "
        f"got {sys.argv[1]!r}"
    )
PY
}

validate_native_match_threshold

validate_max_matches_per_landmark() {
  "$PYTHON" - "$MAX_MATCHES_PER_LANDMARK" <<'PY'
import sys

try:
    value = int(sys.argv[1])
except ValueError as exc:
    raise SystemExit(
        "LAFGS_ULF_MAX_MATCHES_PER_LANDMARK must be a non-negative integer"
    ) from exc
if value < 0:
    raise SystemExit("LAFGS_ULF_MAX_MATCHES_PER_LANDMARK must be non-negative")
if str(value) != sys.argv[1].strip():
    raise SystemExit(
        "LAFGS_ULF_MAX_MATCHES_PER_LANDMARK must be an integer literal"
    )
PY
}

validate_max_matches_per_landmark

validate_support_view_sampling() {
  case "$SUPPORT_VIEW_SAMPLING" in
    uniform|pose_diverse) ;;
    *)
      echo "LAFGS_ULF_SUPPORT_VIEW_SAMPLING must be uniform or pose_diverse, got $SUPPORT_VIEW_SAMPLING" >&2
      exit 2
      ;;
  esac
}

validate_support_view_sampling

validate_native_residual_profile() {
  "$PYTHON" - "$NATIVE_OBSERVATION_SOURCE" "$NATIVE_ANCHOR_AUX_WEIGHT" "$NATIVE_MV_WEIGHT" <<'PY'
import math
import sys

source, anchor_aux, mv_weight = sys.argv[1:]
if source not in {"native", "native_plus_anchor"}:
    raise SystemExit(
        "LAFGS_ULF_NATIVE_OBSERVATION_SOURCE must be native or native_plus_anchor, "
        f"got {source!r}"
    )
try:
    anchor_aux = float(anchor_aux)
    mv_weight = float(mv_weight)
except ValueError as exc:
    raise SystemExit("native residual weights must be numeric") from exc
if not math.isfinite(anchor_aux) or anchor_aux < 0.0:
    raise SystemExit("LAFGS_ULF_NATIVE_ANCHOR_AUX_WEIGHT must be finite and non-negative")
if not math.isfinite(mv_weight) or mv_weight < 0.0:
    raise SystemExit("LAFGS_ULF_NATIVE_MV_WEIGHT must be finite and non-negative")
if source == "native" and (anchor_aux > 0.0 or mv_weight > 0.0):
    raise SystemExit(
        "an anchor auxiliary requires "
        "LAFGS_ULF_NATIVE_OBSERVATION_SOURCE=native_plus_anchor"
    )
if source == "native_plus_anchor" and anchor_aux <= 0.0:
    raise SystemExit(
        "native_plus_anchor requires a positive "
        "LAFGS_ULF_NATIVE_ANCHOR_AUX_WEIGHT"
    )
PY
}

validate_native_residual_profile

validate_formal_pure_native_profile() {
  if [[ "$NATIVE_OBSERVATION_SOURCE" != "native" || "$NATIVE_ANCHOR_AUX_WEIGHT" != "0" || "$NATIVE_MV_WEIGHT" != "0" ]]; then
    echo "This formal runner is pure-native only; use run_lafgs_v2_native_objective_ablation.sh for mixed anchor controls" >&2
    exit 2
  fi
}

validate_formal_pure_native_profile

validate_ba_depth_gate() {
  "$PYTHON" - "$BA_DEPTH_ABS_TOLERANCE" "$BA_DEPTH_REL_TOLERANCE" <<'PY'
import math
import sys

for name, raw in zip(("absolute", "relative"), sys.argv[1:]):
    try:
        value = float(raw)
    except ValueError as exc:
        raise SystemExit(f"BA {name} depth tolerance must be numeric") from exc
    if not math.isfinite(value) or value < 0.0:
        raise SystemExit(
            f"BA {name} depth tolerance must be finite and non-negative, got {raw!r}"
        )
PY
}

validate_ba_depth_gate

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "Required artifact is missing: $1" >&2
    exit 1
  fi
}

require_bank_size() {
  local artifact="$1"
  local expected="$2"
  local count
  count="$($PYTHON - "$artifact" <<'PY'
import pickle
import sys
value = pickle.load(open(sys.argv[1], "rb"))
print(int(value.numel() if hasattr(value, "numel") else len(value)))
PY
)"
  if [[ "$count" != "$expected" ]]; then
    echo "Expected exactly $expected landmarks in $artifact, found $count" >&2
    exit 1
  fi
}

require_anchor_bounds() {
  local state="$1"
  local tangent_bound="$2"
  local normal_bound="$3"
  require_file "$state"
  "$PYTHON" - "$state" "$tangent_bound" "$normal_bound" <<'PY'
import math
import sys

import torch

path, tangent_arg, normal_arg = sys.argv[1:]
state = torch.load(path, map_location="cpu")
config = state.get("config", {})
try:
    tangent_state = float(config["tangent_bound_m"])
    normal_state = float(config["normal_bound_m"])
except (KeyError, TypeError, ValueError) as exc:
    raise SystemExit(f"State has no surface-anchor bound metadata: {path}") from exc
tangent_arg = float(tangent_arg)
normal_arg = float(normal_arg)
if not (
    math.isclose(tangent_state, tangent_arg, rel_tol=0.0, abs_tol=1e-12)
    and math.isclose(normal_state, normal_arg, rel_tol=0.0, abs_tol=1e-12)
):
    raise SystemExit(
        "State surface-anchor bounds do not match the requested stage: "
        f"state=({tangent_state:g}, {normal_state:g}) "
        f"requested=({tangent_arg:g}, {normal_arg:g}) path={path}"
    )
print(
    "Verified surface-anchor bounds: "
    f"tangent={tangent_state:g} normal={normal_state:g} state={path}"
)
PY
}

verify_native_checkpoint_contract() {
  # Check the saved effective contract as well as raw parser values.  This
  # prevents a checkpoint made with an inert parser default from being reused
  # as evidence for a pure-native stage.
  local state="$1"
  local stage="$2"
  local expected_source="$3"
  local expected_anchor_aux="$4"
  local expected_mv="$5"
  local expected_local="$6"
  local expected_dustbin="$7"
  local expected_outcome="$8"
  local expected_retrieval="$9"
  local expected_trust="${10}"
  local expect_reject_contract="${11}"
  require_file "$state"
  "$PYTHON" - \
    "$state" "$stage" "$expected_source" "$expected_anchor_aux" \
    "$expected_mv" "$expected_local" "$expected_dustbin" \
    "$expected_outcome" "$expected_retrieval" "$expected_trust" \
    "$expect_reject_contract" "$NATIVE_MATCH_THRESHOLD" "$LANDMARK_BUDGET" <<'PY'
import math
import sys

import torch

(
    path, stage, expected_source, expected_anchor_aux, expected_mv,
    expected_local, expected_dustbin, expected_outcome, expected_retrieval,
    expected_trust, expect_reject_contract, match_threshold, landmark_budget,
) = sys.argv[1:]
state = torch.load(path, map_location="cpu")
config = state.get("config", {})
if not isinstance(config, dict):
    raise SystemExit(f"{stage}: checkpoint has no config dictionary: {path}")

def require_equal(name, actual, expected):
    if actual != expected:
        raise SystemExit(
            f"{stage}: checkpoint {name}={actual!r}, expected {expected!r}: {path}"
        )

def require_close(name, actual, expected):
    try:
        actual = float(actual)
        expected = float(expected)
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"{stage}: checkpoint {name} is not numeric: {actual!r}") from exc
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12):
        raise SystemExit(
            f"{stage}: checkpoint {name}={actual:g}, expected {expected:g}: {path}"
        )

require_equal("observation_source", config.get("observation_source"), expected_source)
require_equal("query_feature_contract", config.get("query_feature_contract"), "native_resized_input")
require_equal("objective", config.get("objective"), "hard")
require_equal("native_sampling_mode", config.get("native_sampling_mode"), "detector_grid")
require_equal("native_outcome_mode", bool(config.get("native_outcome_mode")), expected_outcome == "true")
require_close("native_anchor_aux_weight", config.get("native_anchor_aux_weight"), expected_anchor_aux)
require_close("mv_weight", config.get("mv_weight"), expected_mv)
require_close("local_weight", config.get("local_weight"), expected_local)
require_close("dustbin_weight", config.get("dustbin_weight"), expected_dustbin)
require_close("retrieval_weight", config.get("retrieval_weight"), expected_retrieval)
require_close("trust_weight", config.get("trust_weight"), expected_trust)
require_close("distill_budget", config.get("distill_budget"), 0)
if int(torch.as_tensor(state.get("landmark_indices", [])).numel()) != int(landmark_budget):
    raise SystemExit(f"{stage}: checkpoint landmark count is not {landmark_budget}: {path}")

contract = config.get("native_auxiliary_contract")
if not isinstance(contract, dict):
    raise SystemExit(f"{stage}: missing native_auxiliary_contract: {path}")
require_equal("contract.observation_source", contract.get("observation_source"), expected_source)
require_equal("contract.native_sampling_mode", contract.get("native_sampling_mode"), "detector_grid")
require_equal("contract.native_outcome_mode", bool(contract.get("native_outcome_mode")), expected_outcome == "true")
expected_pure = expected_source == "native"
require_equal("contract.pure_native", bool(contract.get("pure_native")), expected_pure)
effective = contract.get("effective_anchor_weights", {})
if not isinstance(effective, dict):
    raise SystemExit(f"{stage}: malformed effective_anchor_weights: {path}")
for name, raw_weight in (("mv", expected_mv), ("local", expected_local), ("dustbin", expected_dustbin)):
    require_close(
        f"contract.effective_anchor_weights.{name}",
        effective.get(name),
        float(expected_anchor_aux) * float(raw_weight),
    )

reject_contract = config.get("native_reject_contract", {})
if expect_reject_contract == "true":
    if not isinstance(reject_contract, dict) or not bool(reject_contract.get("enabled", False)):
        raise SystemExit(f"{stage}: native reject contract is required: {path}")
    require_close(
        "native_reject_contract.deployment_match_threshold",
        reject_contract.get("deployment_match_threshold"),
        match_threshold,
    )
elif bool(reject_contract.get("enabled", False)):
    raise SystemExit(f"{stage}: unexpected native reject contract: {path}")

if expected_outcome == "true":
    require_close("native_reject_threshold", config.get("native_reject_threshold"), match_threshold)
print(f"Verified native checkpoint contract: stage={stage} state={path}")
PY
}

ensure_fixed_bank_metadata() {
  # The strict 20K protocol deliberately disables distillation, so the
  # historical distillation-only metadata artifact is absent in runs made
  # before train_lafgs_map.py started emitting fixed-bank metadata.  Evaluation
  # does not consume it while use_landmark_prior=false, but materializing a
  # minimal identity-only record makes the bank contract explicit and lets old
  # bootstrap output be reused without recomputation.
  local state="$1"
  local ids="$2"
  local meta="$3"
  require_file "$state"
  require_file "$ids"
  if [[ -f "$meta" ]]; then
    return
  fi
  "$PYTHON" - "$state" "$ids" "$meta" <<'PY'
import pickle
import sys
from pathlib import Path

import torch

state_path, ids_path, output_path = map(Path, sys.argv[1:])
state = torch.load(state_path, map_location="cpu")
with ids_path.open("rb") as handle:
    ids = torch.as_tensor(pickle.load(handle), dtype=torch.long).reshape(-1)
state_ids = torch.as_tensor(state["landmark_indices"], dtype=torch.long).reshape(-1)
if not torch.equal(ids, state_ids):
    raise SystemExit(
        "Fixed-bank metadata refused: sampled_idx.pkl and map state landmark IDs differ"
    )
output_path.parent.mkdir(parents=True, exist_ok=True)
torch.save(
    {
        "version": 1,
        "landmark_indices": ids,
        "fixed_bank": True,
        "one_time_landmark_distillation": False,
        "feature_dim": int(torch.as_tensor(state["landmark_features"]).shape[1]),
        "state_path": str(state_path.resolve()),
    },
    output_path,
)
print(f"Wrote fixed-bank metadata: {output_path}")
PY
}

run_logged() {
  local stage="$1"
  shift
  printf '%q ' "$@" > "$LOG_ROOT/${stage}.command.sh"
  printf '\n' >> "$LOG_ROOT/${stage}.command.sh"
  "$@" 2>&1 | tee "$LOG_ROOT/${stage}.log"
}

write_protocol_manifest() {
  "$PYTHON" - \
    "$PROTOCOL_MANIFEST" "$SCENE" "$MODEL_ROOT" "$SOURCE_PLY" \
    "$LANDMARK_BUDGET" "$SUPPORT_VIEWS" "$SUPPORT_VIEW_SAMPLING" "$ULF_PARITY_KCS_MASK_POLICY" "$NATIVE_KEYPOINTS" \
    "$VALIDATION_RATIO" "$SPLIT_MODE" "$SPLIT_SEED" "$TRAIN_SEED" "$BOOTSTRAP_GATE_CM" \
    "$BA_TANGENT_BOUND_M" "$BA_NORMAL_BOUND_M" "$NATIVE_MATCH_THRESHOLD" \
    "$BA_DEPTH_ABS_TOLERANCE" "$BA_DEPTH_REL_TOLERANCE" \
    "$QUERY_CACHE" "$VISIBILITY_CACHE" "$BOOTSTRAP_DIR" \
    "$PROTOCOL_VERSION" "$NATIVE_OBSERVATION_SOURCE" \
    "$NATIVE_ANCHOR_AUX_WEIGHT" "$NATIVE_MV_WEIGHT" \
    "$EVAL_PROFILE" "$MAX_MATCHES_PER_LANDMARK" "$SELECTION_MODE" <<'PY'
import json
import os
import sys
from pathlib import Path

(
    output, scene, model_root, source_ply, landmark_budget, support_views,
    support_view_sampling, ulf_parity_kcs_mask_policy, native_keypoints, validation_ratio, split_mode, split_seed, train_seed, bootstrap_gate,
    tangent_bound_m, normal_bound_m, native_match_threshold,
    ba_depth_abs_tolerance, ba_depth_rel_tolerance,
    query_cache, visibility_cache, bootstrap_dir, protocol_version,
    native_observation_source, native_anchor_aux_weight, native_mv_weight,
    evaluation_profile, max_matches_per_landmark, selection_mode,
) = sys.argv[1:]
native_anchor_aux_weight = float(native_anchor_aux_weight)
native_mv_weight = float(native_mv_weight)
max_matches_per_landmark = int(max_matches_per_landmark)
payload = {
    "schema_version": 6,
    "method": f"lafgs_kcs_gwff_native_outcome_bounded_ba_{protocol_version}",
    "formal_protocol": {
        "id": "lafgs_v2_fullres_native_uncapped_v1",
        "longest_edge": 0,
        "query_feature_contract": "native_resized_input",
        "sparse_frontend": "ulfloc_native",
        "native_keypoints": int(native_keypoints),
        "match_topk": 1,
        "cosine_threshold": float(native_match_threshold),
        "max_matches_per_landmark": max_matches_per_landmark,
        "test_selection_forbidden": True,
    },
    "scene": scene,
    "frozen_external_rgb_2dgs": str(Path(model_root).resolve()),
    "source_matcha_ply": str(Path(source_ply).resolve()),
    "bootstrap": {
        "scaffold_mode": "ulf_parity",
        "initialization_mode": "ulf_parity",
        "landmark_budget": int(landmark_budget),
        "support_views": int(support_views),
        "support_view_sampling": support_view_sampling,
        "full_resolution": True,
        "longest_edge": 0,
        "early_distillation": False,
        "kcs_mask_policy": ulf_parity_kcs_mask_policy,
        "gwff_mask_policy": "raw_rgb_encoder_then_mask_observation_accumulation",
        "kcs_visibility": "2dgs_raster_contribution",
        "gwff_sampling": "direct_stride8_grid_sample_zero_padding",
        "state_source": str(Path(bootstrap_dir).resolve()),
    },
    "native_residual": {
        "observation_source": native_observation_source,
        "query_frontend": "SuperPoint.detectAndCompute",
        "query_keypoints": int(native_keypoints),
        "candidate_set": "detached_full_bank_cosine_topk_before_gt_labels",
        "outcomes": ["keep", "swap", "miss", "reject"],
        "source_descriptor_regression_weight": native_mv_weight,
        "anchor_auxiliary_weight": native_anchor_aux_weight,
        "effective_anchor_auxiliary_weight": (
            native_anchor_aux_weight * native_mv_weight
        ),
        "gt_source_injected_into_candidates": False,
        "deployment_match_threshold": float(native_match_threshold),
        "training_reject_threshold": float(native_match_threshold),
        "threshold_contract": "exact_direct_cosine_score_v1",
        "effective_loss_weights": {
            "native_outcome": 1.0,
            "descriptor_trust": 0.02,
            "anchor_mv": native_anchor_aux_weight * native_mv_weight,
            "anchor_local": 0.0,
            "anchor_dustbin": 0.0,
        },
        "pure_native_no_anchor_observations": (
            native_observation_source == "native"
            and native_anchor_aux_weight == 0.0
            and native_mv_weight == 0.0
        ),
    },
    "native_matcher": {
        "profile": evaluation_profile,
        "frontend": "SuperPoint.detectAndCompute",
        "native_superpoint_nms_radius": 4,
        "topk": 1,
        "mnn": False,
        "dual_softmax": False,
        "cosine_threshold": float(native_match_threshold),
        "max_matches_per_landmark": max_matches_per_landmark,
        "per_landmark_hardcap_enabled": max_matches_per_landmark > 0,
    },
    "cache_artifacts": {
        "query_cache_path": str(Path(query_cache).resolve()),
        "visibility_cache_path": str(Path(visibility_cache).resolve()),
        "query_cache_reuse_requires_frontend_manifest_match": True,
    },
    "bounded_ba": {
        "descriptor_frozen": True,
        "geometry": "surface_tangent_normal_bounded_native_gt_clean_association",
        "tangent_parameterization": "radial_tanh_with_analytic_zero_limit",
        "tangent_bound_m": float(tangent_bound_m),
        "normal_bound_m": float(normal_bound_m),
        "depth_abs_tolerance_m": float(ba_depth_abs_tolerance),
        "depth_rel_tolerance": float(ba_depth_rel_tolerance),
        "depth_compatibility_gate": (
            float(ba_depth_abs_tolerance) > 0.0
            or float(ba_depth_rel_tolerance) > 0.0
        ),
        "minimum_distinct_support_views": 3,
    },
    "selection": {
        "validation_ratio": float(validation_ratio),
        "split_mode": split_mode,
        "split_seed": int(split_seed),
        "optimization_seed": int(train_seed),
        "bootstrap_gate_median_te_cm": float(bootstrap_gate),
        "checkpoint_selection_mode": selection_mode,
        "joint_clean_pose_gate_required": selection_mode == "safety",
        "joint_clean_pose_gate": [
            "translation_gain",
            "rotation_not_worse",
            "raw_precision_not_worse",
            "inlier_precision_not_worse",
            "pose_information_not_worse",
        ] if selection_mode == "safety" else [],
        "test_metrics_used_for_selection": False,
        "full_data_refit_after_selection": False,
    },
    "known_protocol_difference": {
        "deployment_native_frontend": "mask_rgb_then_post_detection_valid_mask_filter",
        "kcs_frontend": (
            "mask_rgb_without_post_detection_filter"
            if ulf_parity_kcs_mask_policy == "rgb_only"
            else "mask_rgb_then_post_detection_valid_mask_filter"
        ),
        "semantics_matched": ulf_parity_kcs_mask_policy == "deployment_post_filter",
        "reason": "rgb_only preserves ULF KCS behavior; deployment_post_filter is the explicit deployment-semantic ablation",
    },
    "runtime": {
        "camera_loader_workers": int(
            os.environ.get("STDLOC_CAMERA_LOADER_WORKERS", "4")
        ),
    },
}
path = Path(output)
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
print(json.dumps(payload, indent=2, sort_keys=True))
PY
}

prepare_scene() {
  require_file "$SOURCE_PLY"
  if [[ ! -f "$MODEL_ROOT/artifact_provenance.json" ]]; then
    run_logged prepare \
      "$PYTHON" scripts/audit_cambridge_matcha_2dgs_protocol.py \
      --runs_root "$MATCHA_ROOT" --data_root "$DATA_ROOT" --scenes "$SCENE" \
      --output_json "$RUN_ROOT/matcha_protocol.json" \
      --output_markdown "$RUN_ROOT/matcha_protocol.md" \
      --prepare_wrapper_root "$WRAPPER_ROOT"
  fi
  require_file "$MODEL_ROOT/artifact_provenance.json"
  require_file "$MODEL_ROOT/point_cloud/iteration_30000/point_cloud.ply"
  write_protocol_manifest
}

base_map_args() {
  printf '%s\0' \
    "$PYTHON" train_lafgs_map.py \
    --model_path "$MODEL_ROOT" --source_path "$DATA_ROOT/$SCENE" \
    --images processed --data_device cpu --gaussian_type 2dgs --feature_type sp \
    --resolution 1 --longest_edge 0 --norm_before_render --load_iteration 30000 \
    --query_feature_contract native_resized_input \
    --query_cache_path "$QUERY_CACHE" --visibility_cache_path "$VISIBILITY_CACHE" \
    --visibility_mode rasterizer --objective hard \
    --native_keypoint_count "$NATIVE_KEYPOINTS" \
    --native_association_radius_px 2 --native_sampling_mode detector_grid \
    --generic_proposal_count 0 --generic_proposal_weight 0 \
    --distill_budget 0 --validation_ratio "$VALIDATION_RATIO" \
    --split_mode "$SPLIT_MODE" --split_seed "$SPLIT_SEED" --train_seed "$TRAIN_SEED" \
    --max_observations 512 --validation_observations 512
}

append_base_map_args() {
  # NUL-delimited output avoids accidental shell word splitting in the shared
  # immutable command prefix.
  local -n target="$1"
  local item
  while IFS= read -r -d '' item; do
    target+=("$item")
  done < <(base_map_args)
}

bootstrap() {
  prepare_scene
  if [[ -f "$BOOTSTRAP_STATE" && -f "$BOOTSTRAP_IDS" ]]; then
    require_bank_size "$BOOTSTRAP_IDS" "$LANDMARK_BUDGET"
    ensure_fixed_bank_metadata "$BOOTSTRAP_STATE" "$BOOTSTRAP_IDS" "$BOOTSTRAP_META"
    verify_native_checkpoint_contract "$BOOTSTRAP_STATE" bootstrap native 0 0 0 0 false 0 0 false
    echo "[ULF parity] Reusing bootstrap: $BOOTSTRAP_DIR"
    return
  fi
  local command=()
  append_base_map_args command
  command+=(
    --output_dir "$BOOTSTRAP_DIR"
    --scaffold_mode ulf_parity
    --generated_landmark_path "$BOOTSTRAP_DIR/ulf_parity_20k_ids.pkl"
    --regenerate_scaffold --scaffold_budget "$LANDMARK_BUDGET" --scaffold_seed "$TRAIN_SEED"
    --initialization_mode ulf_parity
    --ulf_consensus_keypoints "$NATIVE_KEYPOINTS" --ulf_consensus_radius_px 1.0
    --ulf_consensus_knn 32 --ulf_consensus_max_views "$SUPPORT_VIEWS"
    --ulf_fusion_max_views "$SUPPORT_VIEWS" --ulf_fusion_min_cosine 0
    --ulf_support_view_sampling "$SUPPORT_VIEW_SAMPLING"
    --ulf_parity_kcs_mask_policy "$ULF_PARITY_KCS_MASK_POLICY"
    --observation_source native --native_anchor_aux_weight 0 --no-native_outcome_mode \
    --query_cache_policy reuse_or_build
    --steps 0 --save_steps 0
    --mv_weight 0 --retrieval_weight 0 --trust_weight 0 --local_weight 0
    --dustbin_weight 0 --geometry_weight 0 --pose_weight 0 --pose_gradient_mode off
    --log_interval 50
  )
  run_logged bootstrap "${command[@]}"
  require_file "$BOOTSTRAP_STATE"
  require_file "$BOOTSTRAP_IDS"
  require_bank_size "$BOOTSTRAP_IDS" "$LANDMARK_BUDGET"
  ensure_fixed_bank_metadata "$BOOTSTRAP_STATE" "$BOOTSTRAP_IDS" "$BOOTSTRAP_META"
  verify_native_checkpoint_contract "$BOOTSTRAP_STATE" bootstrap native 0 0 0 0 false 0 0 false
}

make_eval_config() {
  local label="$1"
  local state="$2"
  local cfg="$CONFIG_ROOT/${label}.yaml"
  require_file "$state"
  require_bank_size "$BOOTSTRAP_IDS" "$LANDMARK_BUDGET"
  "$PYTHON" scripts/make_stdloc_eval_cfg.py \
    --base_cfg configs/stdloc_cambridge.yaml --output "$cfg" \
    --artifact_model_path "$MODEL_ROOT" \
    --detector_folder ulfloc_native_no_detector --detector_iters 0 \
    --landmark_path "$BOOTSTRAP_IDS" --landmark_meta_path "$BOOTSTRAP_META" \
    --landmark_feature_override_path "$state" --override_landmark_features \
    --detect_num "$NATIVE_KEYPOINTS" --nms "$EVAL_NMS" \
    --sparse_query_feature_contract native_resized_input \
    --sparse_frontend ulfloc_native --reprojection_error "$EVAL_REPROJECTION_PX" \
    --match_threshold "$NATIVE_MATCH_THRESHOLD" --match_topk 1 \
    --max_matches_per_landmark "$MAX_MATCHES_PER_LANDMARK" \
    --candidate_frontend_match_policy error --diagnostics \
    --diagnostics_grid_rows 4 --diagnostics_grid_cols 4 \
    --diagnostics_voxel_size "$PNP_VOXEL_M" \
    --diagnostics_task_translation_scale_m "$TASK_TRANSLATION_SCALE_M" \
    --diagnostics_task_rotation_scale_degrees 2.0 \
    > "$LOG_ROOT/${label}_config.json"
  printf '%s\n' "$cfg"
}

run_eval() {
  local label="$1"
  local subset="$2"
  local state="$3"
  local result_ref="$RESULT_ROOT/${label}_${subset}.results_path"
  if [[ -f "$result_ref" && -f "$(<"$result_ref")/results_summary.json" ]]; then
    echo "[ULF parity] Reusing evaluation: $label/$subset"
    return
  fi
  local cfg
  cfg="$(make_eval_config "${label}_${subset}" "$state")"
  local command=(
    "$PYTHON" stdloc.py
    --model_path "$MODEL_ROOT" --source_path "$DATA_ROOT/$SCENE"
    --images processed --data_device cpu --gaussian_type 2dgs --feature_type sp
    --resolution 1 --longest_edge 0 --norm_before_render --iteration 30000
    --cfg "$cfg" --prefix "lafgs-v2-ulfparity-${SCENE}-${label}-${subset}"
    --sparse_only
  )
  if [[ "$subset" == "validation" ]]; then
    command+=(
      --evaluation_camera_subset candidate_validation
      --candidate_direct_validation_holdout
      --candidate_validation_ratio "$VALIDATION_RATIO"
      --candidate_split_mode "$SPLIT_MODE" --candidate_split_seed "$SPLIT_SEED"
    )
  fi
  printf '%q ' "${command[@]}" > "$LOG_ROOT/${label}_${subset}.command.sh"
  printf '\n' >> "$LOG_ROOT/${label}_${subset}.command.sh"
  "${command[@]}" 2>&1 | tee "$LOG_ROOT/${label}_${subset}.log"
  local output_path
  output_path="$(sed -n 's/^Output path: //p' "$LOG_ROOT/${label}_${subset}.log" | tail -n 1)"
  if [[ -z "$output_path" || ! -f "$output_path/results_summary.json" ]]; then
    echo "Evaluation did not create results_summary.json for $label/$subset" >&2
    exit 1
  fi
  printf '%s\n' "$output_path" > "$result_ref"
}

result_summary() {
  local label="$1"
  local subset="$2"
  local ref="$RESULT_ROOT/${label}_${subset}.results_path"
  require_file "$ref"
  local directory
  directory="$(<"$ref")"
  require_file "$directory/results_summary.json"
  printf '%s\n' "$directory/results_summary.json"
}

check_bootstrap_gate() {
  local summary
  summary="$(result_summary bootstrap validation)"
  "$PYTHON" - "$summary" "$BOOTSTRAP_GATE_CM" <<'PY'
import json
import sys

summary = json.load(open(sys.argv[1], "r", encoding="utf-8"))
value = float(summary["sparse"]["median_te"])
gate = float(sys.argv[2])
print(f"Bootstrap validation median TE: {value:.6f} cm; gate: <= {gate:.6f} cm")
if value > gate:
    raise SystemExit(
        "Bootstrap gate failed. Do not start residual: audit KCS/GWFF, map geometry, "
        "masking, and native frontend protocol first."
    )
PY
}

bootstrap_validate() {
  bootstrap
  run_eval bootstrap validation "$BOOTSTRAP_STATE"
  check_bootstrap_gate
}

native_outcome_args() {
  printf '%s\0' \
    --observation_source "$NATIVE_OBSERVATION_SOURCE" \
    --native_anchor_aux_weight "$NATIVE_ANCHOR_AUX_WEIGHT" \
    --native_outcome_mode --native_nce_weight 0 \
    --native_keep_weight 1 --native_keep_margin 0.05 \
    --native_swap_weight 1 --native_swap_margin 0.05 \
    --native_miss_weight 1 --native_miss_margin 0.05 \
    --native_reject_weight 0.05 --native_reject_threshold "$NATIVE_MATCH_THRESHOLD" \
    --mv_weight "$NATIVE_MV_WEIGHT" --retrieval_weight 1 --trust_weight 0.02 --local_weight 0 \
    --dustbin_weight 0 --geometry_weight 0 --pose_weight 0 --pose_gradient_mode off
}

append_native_outcome_args() {
  local -n target="$1"
  local item
  while IFS= read -r -d '' item; do
    target+=("$item")
  done < <(native_outcome_args)
}

residual() {
  bootstrap_validate
  if [[ -f "$RESIDUAL_DIR/${RESIDUAL_STEPS}_lafgs_map_state.pt" ]]; then
    verify_native_checkpoint_contract "$RESIDUAL_DIR/${RESIDUAL_STEPS}_lafgs_map_state.pt" residual native 0 0 0 0 true 1 0.02 true
    echo "[ULF parity] Reusing residual phase: $RESIDUAL_DIR"
    return
  fi
  local command=()
  append_base_map_args command
  append_native_outcome_args command
  command+=(
    --output_dir "$RESIDUAL_DIR" --scaffold_mode file --landmark_path "$BOOTSTRAP_IDS"
    --initial_state_path "$BOOTSTRAP_STATE" --initial_state_blend 1 --initial_state_alignment exact
    --initialization_mode ulf_parity --query_cache_policy readonly
    --steps "$RESIDUAL_STEPS" --save_steps 500 1000 2500 "$RESIDUAL_STEPS"
    --feature_lr 5e-5 --weight_decay 1e-4 --hypothesis_topk 32
    --positive_radius_px 2 --negative_radius_px 6 --log_interval 100
  )
  run_logged residual "${command[@]}"
  require_file "$RESIDUAL_DIR/${RESIDUAL_STEPS}_lafgs_map_state.pt"
  verify_native_checkpoint_contract "$RESIDUAL_DIR/${RESIDUAL_STEPS}_lafgs_map_state.pt" residual native 0 0 0 0 true 1 0.02 true
}

residual_validate() {
  residual
  run_eval bootstrap validation "$BOOTSTRAP_STATE"
  local step
  for step in 500 1000 2500 "$RESIDUAL_STEPS"; do
    local state="$RESIDUAL_DIR/${step}_lafgs_map_state.pt"
    if [[ -f "$state" ]]; then
      run_eval "residual_${step}" validation "$state"
    fi
  done
}

select_stage() {
  local output="$1"
  local control_state="$2"
  local control_label="$3"
  shift 3
  if [[ -f "$output" ]]; then
    echo "[ULF parity] Reusing selection: $output"
    return
  fi
  local command=(
    "$PYTHON" scripts/select_lafgs_map_checkpoint.py
    --control_results "$(result_summary "$control_label" validation)"
    --control_state "$control_state" --control_tag "$control_label" --selection_mode "$SELECTION_MODE"
    --min_te_gain_cm 0.02 --metric_tolerance 1e-9 --mean_te_weight 0.05
    --max_recall_2m_drop 0.01 --max_recall_5cm_drop 0.01 --output "$output"
  )
  while [[ $# -gt 1 ]]; do
    local label="$1"
    local state="$2"
    shift 2
    if [[ -f "$state" && -f "$RESULT_ROOT/${label}_validation.results_path" ]]; then
      command+=(--candidate "$label" "$(result_summary "$label" validation)" "$state")
    fi
  done
  run_logged "$(basename "$output" .json)" "${command[@]}"
  require_file "$output"
}

select_residual() {
  residual_validate
  select_stage "$RESIDUAL_SELECTION" "$BOOTSTRAP_STATE" bootstrap \
    residual_500 "$RESIDUAL_DIR/500_lafgs_map_state.pt" \
    residual_1000 "$RESIDUAL_DIR/1000_lafgs_map_state.pt" \
    residual_2500 "$RESIDUAL_DIR/2500_lafgs_map_state.pt" \
    "residual_${RESIDUAL_STEPS}" "$RESIDUAL_DIR/${RESIDUAL_STEPS}_lafgs_map_state.pt"
}

selected_state() {
  local selection="$1"
  require_file "$selection"
  "$PYTHON" - "$selection" <<'PY'
import json
import sys
payload = json.load(open(sys.argv[1], "r", encoding="utf-8"))
if payload.get("selection_protocol", {}).get("test_metrics_used") is not False:
    raise SystemExit("selection is not validation-only")
print(payload["selected_state"])
PY
}

selected_label() {
  local selection="$1"
  require_file "$selection"
  "$PYTHON" - "$selection" <<'PY'
import json
import sys
payload = json.load(open(sys.argv[1], "r", encoding="utf-8"))
print(payload["selected_tag"])
PY
}

selection_uses_control() {
  local selection="$1"
  require_file "$selection"
  "$PYTHON" - "$selection" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], "r", encoding="utf-8"))
print(
    "1"
    if payload.get("used_control_fallback", payload.get("used_strong_fallback", False))
    else "0"
)
PY
}

ba() {
  select_residual
  local initial_state
  initial_state="$(selected_state "$RESIDUAL_SELECTION")"
  require_file "$initial_state"
  local expect_reject_contract=false
  if [[ "$initial_state" != "$BOOTSTRAP_STATE" ]]; then
    expect_reject_contract=true
  fi
  if [[ -f "$BA_DIR/${BA_STEPS}_lafgs_map_state.pt" ]]; then
    require_anchor_bounds "$BA_DIR/${BA_STEPS}_lafgs_map_state.pt" "$BA_TANGENT_BOUND_M" "$BA_NORMAL_BOUND_M"
    verify_native_checkpoint_contract "$BA_DIR/${BA_STEPS}_lafgs_map_state.pt" bounded_ba native 0 0 0 0 false 0 0 "$expect_reject_contract"
    echo "[ULF parity] Reusing bounded BA: $BA_DIR"
    return
  fi
  local command=()
  append_base_map_args command
  command+=(
    --output_dir "$BA_DIR" --scaffold_mode file --landmark_path "$BOOTSTRAP_IDS"
    --initial_state_path "$initial_state" --initial_state_blend 1 --initial_state_alignment exact
    --initialization_mode ulf_parity --query_cache_policy readonly
    --observation_source native --native_anchor_aux_weight 0 --no-native_outcome_mode \
    --descriptor_end_step -1
    --steps "$BA_STEPS" --save_steps 500 1000 "$BA_STEPS"
    --feature_lr 0 --geometry_lr 0.003 --geometry_start_step 0 --geometry_weight 1
    --geometry_mode native_association --geometry_association_max_reprojection_px 2
    --geometry_association_min_margin 0.02
    --geometry_association_depth_abs_tolerance "$BA_DEPTH_ABS_TOLERANCE"
    --geometry_association_depth_rel_tolerance "$BA_DEPTH_REL_TOLERANCE"
    --geometry_association_min_support_views 3
    --geometry_association_support_observations "$NATIVE_KEYPOINTS"
    --tangent_bound_m "$BA_TANGENT_BOUND_M" --normal_bound_m "$BA_NORMAL_BOUND_M"
    --surface_weight 0.05 --depth_weight 0.25 --reprojection_weight 1
    --mv_weight 0 --retrieval_weight 0 --trust_weight 0 --local_weight 0 --dustbin_weight 0
    --pose_weight 0 --pose_gradient_mode off --log_interval 100
  )
  run_logged "bounded_ba${BA_STAGE_TAG}" "${command[@]}"
  require_file "$BA_DIR/${BA_STEPS}_lafgs_map_state.pt"
  verify_native_checkpoint_contract "$BA_DIR/${BA_STEPS}_lafgs_map_state.pt" bounded_ba native 0 0 0 0 false 0 0 "$expect_reject_contract"
}

ba_validate() {
  ba
  local residual_state residual_label
  residual_state="$(selected_state "$RESIDUAL_SELECTION")"
  residual_label="$(selected_label "$RESIDUAL_SELECTION")"
  run_eval "residual_selected_${residual_label}" validation "$residual_state"
  local step
  for step in 500 1000 "$BA_STEPS"; do
    local state="$BA_DIR/${step}_lafgs_map_state.pt"
    if [[ -f "$state" ]]; then
      run_eval "ba_${step}${BA_STAGE_TAG}" validation "$state"
    fi
  done
}

ba_control_label() {
  local residual_label
  residual_label="$(selected_label "$RESIDUAL_SELECTION")"
  printf 'residual_selected_%s\n' "$residual_label"
}

select_ba() {
  ba_validate
  local control_label control_state
  control_label="$(ba_control_label)"
  control_state="$(selected_state "$RESIDUAL_SELECTION")"
  select_stage "$BA_SELECTION" "$control_state" "$control_label" \
    "ba_500${BA_STAGE_TAG}" "$BA_DIR/500_lafgs_map_state.pt" \
    "ba_1000${BA_STAGE_TAG}" "$BA_DIR/1000_lafgs_map_state.pt" \
    "ba_${BA_STEPS}${BA_STAGE_TAG}" "$BA_DIR/${BA_STEPS}_lafgs_map_state.pt"
}

refresh() {
  select_ba
  if [[ "$(selection_uses_control "$BA_SELECTION")" == "1" ]]; then
    # A rejected BA must not be followed by a descriptor refresh that silently
    # reinterprets the residual state's anchor parameterization.  The final
    # selector will retain this validation-selected control explicitly.
    echo "[ULF parity] Skipping descriptor refresh: BA did not beat the residual control"
    return
  fi
  local initial_state
  initial_state="$(selected_state "$BA_SELECTION")"
  require_file "$initial_state"
  require_anchor_bounds "$initial_state" "$BA_TANGENT_BOUND_M" "$BA_NORMAL_BOUND_M"
  if [[ -f "$REFRESH_DIR/${REFRESH_STEPS}_lafgs_map_state.pt" ]]; then
    require_anchor_bounds "$REFRESH_DIR/${REFRESH_STEPS}_lafgs_map_state.pt" "$BA_TANGENT_BOUND_M" "$BA_NORMAL_BOUND_M"
    verify_native_checkpoint_contract "$REFRESH_DIR/${REFRESH_STEPS}_lafgs_map_state.pt" refresh native 0 0 0 0 true 1 0.02 true
    echo "[ULF parity] Reusing descriptor refresh: $REFRESH_DIR"
    return
  fi
  local command=()
  append_base_map_args command
  append_native_outcome_args command
  command+=(
    --output_dir "$REFRESH_DIR" --scaffold_mode file --landmark_path "$BOOTSTRAP_IDS"
    --initial_state_path "$initial_state" --initial_state_blend 1 --initial_state_alignment exact
    --initialization_mode ulf_parity --query_cache_policy readonly
    --steps "$REFRESH_STEPS" --save_steps 500 "$REFRESH_STEPS"
    --feature_lr 2.5e-5 --weight_decay 1e-4 --hypothesis_topk 32
    --tangent_bound_m "$BA_TANGENT_BOUND_M" --normal_bound_m "$BA_NORMAL_BOUND_M"
    --positive_radius_px 2 --negative_radius_px 6 --log_interval 100
  )
  run_logged refresh "${command[@]}"
  require_file "$REFRESH_DIR/${REFRESH_STEPS}_lafgs_map_state.pt"
  verify_native_checkpoint_contract "$REFRESH_DIR/${REFRESH_STEPS}_lafgs_map_state.pt" refresh native 0 0 0 0 true 1 0.02 true
}

refresh_validate() {
  refresh
  local ba_state ba_label
  ba_state="$(selected_state "$BA_SELECTION")"
  ba_label="$(selected_label "$BA_SELECTION")"
  run_eval "ba_selected_${ba_label}${BA_STAGE_TAG}" validation "$ba_state"
  local step
  for step in 500 "$REFRESH_STEPS"; do
    local state="$REFRESH_DIR/${step}_lafgs_map_state.pt"
    if [[ -f "$state" ]]; then
      run_eval "refresh_${step}${BA_STAGE_TAG}" validation "$state"
    fi
  done
}

refresh_control_label() {
  local ba_label
  ba_label="$(selected_label "$BA_SELECTION")"
  printf 'ba_selected_%s%s\n' "$ba_label" "$BA_STAGE_TAG"
}

select_final() {
  refresh_validate
  local control_label control_state
  control_label="$(refresh_control_label)"
  control_state="$(selected_state "$BA_SELECTION")"
  select_stage "$FINAL_SELECTION" "$control_state" "$control_label" \
    "refresh_500${BA_STAGE_TAG}" "$REFRESH_DIR/500_lafgs_map_state.pt" \
    "refresh_${REFRESH_STEPS}${BA_STAGE_TAG}" "$REFRESH_DIR/${REFRESH_STEPS}_lafgs_map_state.pt"
}

test_selected() {
  select_final
  local final_state final_label
  final_state="$(selected_state "$FINAL_SELECTION")"
  final_label="$(selected_label "$FINAL_SELECTION")"
  # This is the sole official held-out test call in the protocol. No control
  # map is evaluated on test and no output of this invocation is fed back into
  # any selection decision.
  run_eval "selected_${final_label}${BA_STAGE_TAG}" test "$final_state"
  "$PYTHON" - \
    "$RUN_ROOT/final_test_manifest${BA_STAGE_TAG}.json" "$FINAL_SELECTION" \
    "$(result_summary "selected_${final_label}${BA_STAGE_TAG}" test)" "$final_state" <<'PY'
import json
import sys
from pathlib import Path
output, selection, results, state = map(Path, sys.argv[1:])
payload = json.loads(selection.read_text())
if payload.get("selection_protocol", {}).get("test_metrics_used") is not False:
    raise SystemExit("final selection used test metrics")
record = {
    "selection": str(selection.resolve()),
    "selection_was_validation_only": True,
    "selected_tag": payload["selected_tag"],
    "selection_control_tag": payload.get("control_tag", "control_strong"),
    "selected_state": str(state.resolve()),
    "held_out_test_results": str(results.resolve()),
    "test_evaluations_in_this_protocol": 1,
    "full_data_refit_after_selection": False,
}
output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
print(json.dumps(record, indent=2, sort_keys=True))
PY
}

test_ba_selected() {
  # The descriptor refresh is an explicitly separate experiment.  When the
  # validation-only BA selector retains the residual state (or accepts BA),
  # this mode provides the single held-out test permitted by the protocol
  # without silently running that known-independent stage first.
  select_ba
  local selected_state selected_label manifest label
  selected_state="$(selected_state "$BA_SELECTION")"
  selected_label="$(selected_label "$BA_SELECTION")"
  manifest="$RUN_ROOT/ba_selected_test_manifest${BA_STAGE_TAG}.json"
  label="selected_ba_${selected_label}${BA_STAGE_TAG}"

  if [[ -f "$manifest" ]]; then
    echo "[ULF parity] Reusing BA-selected held-out test manifest: $manifest"
    return
  fi

  # No validation artifact can be replaced by a test result.  The selection
  # file is produced before this call and must explicitly attest to that.
  "$PYTHON" - "$BA_SELECTION" <<'PY'
import json
import sys

selection = json.load(open(sys.argv[1], "r", encoding="utf-8"))
if selection.get("selection_protocol", {}).get("test_metrics_used") is not False:
    raise SystemExit("BA selection is not validation-only")
PY

  run_eval "$label" test "$selected_state"
  "$PYTHON" - "$manifest" "$BA_SELECTION" \
    "$(result_summary "$label" test)" "$selected_state" <<'PY'
import json
import sys
from pathlib import Path

output, selection, results, state = map(Path, sys.argv[1:])
payload = json.loads(selection.read_text())
record = {
    "selection": str(selection.resolve()),
    "selection_was_validation_only": True,
    "selected_tag": payload["selected_tag"],
    "selection_control_tag": payload.get("control_tag", "bootstrap"),
    "selected_state": str(state.resolve()),
    "held_out_test_results": str(results.resolve()),
    "test_evaluations_in_this_protocol": 1,
    "final_stage": "ba_safety_selection_without_descriptor_refresh",
    "full_data_refit_after_selection": False,
}
output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
print(json.dumps(record, indent=2, sort_keys=True))
PY
}

case "$MODE" in
  prepare) prepare_scene ;;
  bootstrap) bootstrap ;;
  bootstrap_validate) bootstrap_validate ;;
  residual) residual ;;
  residual_validate) residual_validate ;;
  select_residual) select_residual ;;
  ba) ba ;;
  ba_validate) ba_validate ;;
  select_ba) select_ba ;;
  refresh) refresh ;;
  refresh_validate) refresh_validate ;;
  select_final) select_final ;;
  test) test_selected ;;
  test_ba_selected) test_ba_selected ;;
  all)
    bootstrap_validate
    select_residual
    select_ba
    select_final
    test_selected
    ;;
esac
