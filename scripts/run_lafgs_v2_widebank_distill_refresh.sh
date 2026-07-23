#!/usr/bin/env bash
set -euo pipefail

# Validation-only continuation of the intended ordering:
# wide KCS/GWFF bank -> native residual on the wide bank -> matchability-first
# 16K distillation -> short native descriptor refresh.
#
# This is intentionally separate from the formal test runner.  It consumes a
# fixed residual state chosen before this study, performs all choices on the
# candidate-validation split, and never calls the test evaluation mode.

if [[ $# -ne 3 ]]; then
  echo "Usage: $0 <scene> <gpu> <distill|distill_validate|refresh|refresh_validate|all>" >&2
  exit 2
fi

SCENE="$1"
GPU="$2"
MODE="$3"

case "$SCENE" in
  OldHospital) ;;
  *) echo "This initial wide-bank study currently has a fixed OldHospital source state" >&2; exit 2 ;;
esac
case "$GPU" in
  0|1|2) ;;
  *) echo "GPU must be 0, 1, or 2; got $GPU" >&2; exit 2 ;;
esac
case "$MODE" in
  distill|distill_validate|refresh|refresh_validate|all) ;;
  *) echo "Unsupported mode: $MODE" >&2; exit 2 ;;
esac

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/root/miniconda3/envs/cybersim_agent/bin/python}"
DATA_ROOT="${CAMBRIDGE_DATA_ROOT:-/mnt/pool/sqy/Cambridge_stdloc}"
REFERENCE_ROOT="${LAFGS_V2_ULFPARITY_REFERENCE_ROOT:-/mnt/pool/sqy/stdloc_lafgs_v2_ulfparity_alternating_20260721}"
EXPERIMENT_ROOT="${LAFGS_V2_WIDEBANK_ROOT:-/mnt/pool/sqy/stdloc_lafgs_v2_widebank_distill_20260722}"
MODEL_ROOT="$REFERENCE_ROOT/matcha_wrappers/$SCENE"

LONGEST_EDGE=0
NATIVE_KEYPOINTS=2048
MATCH_THRESHOLD=0
MAX_MATCHES_PER_LANDMARK=0
EVAL_REPROJECTION_PX=12
VALIDATION_RATIO=0.2
# A wide-bank distillation result is comparable to the formal map only when
# both its source residual and its validation split use the same protocol.
SPLIT_MODE="stratified_temporal_block"
SPLIT_SEED=2026
WIDE_BANK_BUDGET="${LAFGS_WIDEBANK_WIDE_BANK_BUDGET:-32000}"
FINAL_BANK_BUDGET="${LAFGS_WIDEBANK_FINAL_BANK_BUDGET:-16384}"
RESIDUAL_STEPS="${LAFGS_WIDEBANK_RESIDUAL_STEPS:-5000}"
REFRESH_STEPS=1000
TANGENT_BOUND_M=0.005
NORMAL_BOUND_M=0.002
HARD_MATCHABILITY_CORE_RATIO="${LAFGS_WIDEBANK_HARD_MATCHABILITY_CORE_RATIO:-0}"
QUALITY_RESERVOIR_MULTIPLIER="${LAFGS_WIDEBANK_QUALITY_RESERVOIR_MULTIPLIER:-0}"
QUALITY_RESERVOIR_SCORE="${LAFGS_WIDEBANK_QUALITY_RESERVOIR_SCORE:-posterior_mean}"
QUALITY_RESERVOIR_WILSON_Z="${LAFGS_WIDEBANK_QUALITY_RESERVOIR_WILSON_Z:-1.96}"
QUERY_CACHE="${LAFGS_WIDEBANK_QUERY_CACHE_PATH:-$REFERENCE_ROOT/OldHospital/ulfparity_native20k_s128_k2048_v2/query_cache_native_fullres_k2048.pt}"
CAMERA_LOADER_WORKERS="${LAFGS_WIDEBANK_CAMERA_LOADER_WORKERS:-0}"

SOURCE_RUN_ROOT="${LAFGS_WIDEBANK_SOURCE_RUN_ROOT:-}"
if [[ -z "$SOURCE_RUN_ROOT" ]]; then
  echo "Set LAFGS_WIDEBANK_SOURCE_RUN_ROOT to a validation-selected wide residual built with ${SPLIT_MODE}" >&2
  exit 2
fi
SOURCE_DIR="$SOURCE_RUN_ROOT/residual_${RESIDUAL_STEPS}"
SOURCE_STATE="${LAFGS_WIDEBANK_SOURCE_STATE:-$SOURCE_DIR/${RESIDUAL_STEPS}_lafgs_map_state.pt}"
SOURCE_IDS="${LAFGS_WIDEBANK_SOURCE_IDS:-$SOURCE_DIR/sampled_idx.pkl}"
# The strict ULF runner and the robust-KCS runner use different, explicitly
# versioned locations for their validation-only selector and protocol
# manifest.  Detect only these two known layouts; anything else still fails
# closed in verify_source below.
DEFAULT_SOURCE_SELECTION="$SOURCE_RUN_ROOT/validation/residual_selection_v7_safety_control_identity_v8_fullres_native_uncapped_pure_native.json"
if [[ ! -f "$DEFAULT_SOURCE_SELECTION" && -f "$SOURCE_RUN_ROOT/results/residual_selection_safety.json" ]]; then
  DEFAULT_SOURCE_SELECTION="$SOURCE_RUN_ROOT/results/residual_selection_safety.json"
fi
DEFAULT_SOURCE_PROTOCOL="$SOURCE_RUN_ROOT/protocol_manifest_ungated.json"
if [[ ! -f "$DEFAULT_SOURCE_PROTOCOL" && -f "$SOURCE_RUN_ROOT/study_manifest.json" ]]; then
  DEFAULT_SOURCE_PROTOCOL="$SOURCE_RUN_ROOT/study_manifest.json"
fi
SOURCE_SELECTION="${LAFGS_WIDEBANK_SOURCE_SELECTION:-$DEFAULT_SOURCE_SELECTION}"
SOURCE_PROTOCOL="${LAFGS_WIDEBANK_SOURCE_PROTOCOL:-$DEFAULT_SOURCE_PROTOCOL}"

if ! [[ "$CAMERA_LOADER_WORKERS" =~ ^[0-9]+$ ]]; then
  echo "LAFGS_WIDEBANK_CAMERA_LOADER_WORKERS must be a non-negative integer" >&2
  exit 2
fi
if ! "$PYTHON" - "$HARD_MATCHABILITY_CORE_RATIO" <<'PY'
import math
import sys

value = float(sys.argv[1])
if not math.isfinite(value) or not 0.0 <= value <= 1.0:
    raise SystemExit(1)
PY
then
  echo "LAFGS_WIDEBANK_HARD_MATCHABILITY_CORE_RATIO must lie in [0, 1]" >&2
  exit 2
fi
if ! "$PYTHON" - "$QUALITY_RESERVOIR_MULTIPLIER" <<'PY'
import math
import sys

value = float(sys.argv[1])
if not math.isfinite(value) or value < 0.0 or (0.0 < value < 1.0):
    raise SystemExit(1)
PY
then
  echo "LAFGS_WIDEBANK_QUALITY_RESERVOIR_MULTIPLIER must be zero or >= 1" >&2
  exit 2
fi
case "$QUALITY_RESERVOIR_SCORE" in
  posterior_mean|wilson_lower) ;;
  *) echo "LAFGS_WIDEBANK_QUALITY_RESERVOIR_SCORE must be posterior_mean or wilson_lower" >&2; exit 2 ;;
esac
if ! "$PYTHON" - "$QUALITY_RESERVOIR_WILSON_Z" <<'PY'
import math
import sys

value = float(sys.argv[1])
if not math.isfinite(value) or value <= 0.0:
    raise SystemExit(1)
PY
then
  echo "LAFGS_WIDEBANK_QUALITY_RESERVOIR_WILSON_Z must be finite and positive" >&2
  exit 2
fi
HARD_CORE_TAG="${HARD_MATCHABILITY_CORE_RATIO/./p}"
QUALITY_RESERVOIR_TAG="${QUALITY_RESERVOIR_MULTIPLIER/./p}"
QUALITY_RESERVOIR_Z_TAG="${QUALITY_RESERVOIR_WILSON_Z/./p}"

# v3 was generated from a temporal-block wide source before the formal split
# was frozen.  Keep the current study in a separate namespace so old distill
# and refresh states cannot be reused merely because their paths exist.
WIDEBANK_PROTOCOL_VERSION="v7_split${SPLIT_MODE}_fullres_native_uncapped_qcore${HARD_CORE_TAG}_qres${QUALITY_RESERVOIR_TAG}_qscore${QUALITY_RESERVOIR_SCORE}_qz${QUALITY_RESERVOIR_Z_TAG}"
RUN_ROOT="${LAFGS_WIDEBANK_RUN_ROOT:-$EXPERIMENT_ROOT/$SCENE/wide${WIDE_BANK_BUDGET}_selected_residual${RESIDUAL_STEPS}_to_${FINAL_BANK_BUDGET}_matchability_first_${WIDEBANK_PROTOCOL_VERSION}}"
DISTILL_DIR="$RUN_ROOT/distill_${FINAL_BANK_BUDGET}"
DISTILL_STATE="$DISTILL_DIR/distilled_lafgs_map_state.pt"
DISTILL_IDS="$DISTILL_DIR/distilled_sampled_idx.pkl"
DISTILL_META="$DISTILL_DIR/landmark_meta.pt"
DISTILL_VISIBILITY_CACHE="$DISTILL_DIR/visibility_${WIDE_BANK_BUDGET}_native.pt"
REFRESH_DIR="$RUN_ROOT/refresh_${REFRESH_STEPS}"
REFRESH_VISIBILITY_CACHE="$REFRESH_DIR/visibility_${FINAL_BANK_BUDGET}_native.pt"
CONFIG_ROOT="$RUN_ROOT/configs"
LOG_ROOT="$RUN_ROOT/logs"
RESULT_ROOT="$RUN_ROOT/results"
STDLOC_RESULTS_ROOT="$RUN_ROOT/stdloc_results"
MANIFEST="$RUN_ROOT/study_manifest.json"

export CUDA_VISIBLE_DEVICES="$GPU"
export CUDA_HOME=/usr/local/cuda-11.8
export PATH="/root/miniconda3/envs/cybersim_agent/bin:$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="/root/miniconda3/envs/cybersim_agent/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$REPO_ROOT"
export PYTHONHASHSEED=2026
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

verify_source() {
  require_file "$MODEL_ROOT/artifact_provenance.json"
  require_file "$SOURCE_STATE"
  require_file "$SOURCE_IDS"
  require_file "$SOURCE_SELECTION"
  require_file "$SOURCE_PROTOCOL"
  require_file "$QUERY_CACHE"
  "$PYTHON" - "$SOURCE_STATE" "$SOURCE_IDS" "$SOURCE_SELECTION" "$SOURCE_PROTOCOL" \
    "$WIDE_BANK_BUDGET" "$SPLIT_MODE" "$SPLIT_SEED" "$NATIVE_KEYPOINTS" <<'PY'
import json
import pickle
import sys
import torch

(
    state_path,
    ids_path,
    selection_path,
    protocol_path,
    expected,
    expected_split,
    expected_seed,
    expected_keypoints,
) = sys.argv[1:]
expected = int(expected)
state = torch.load(state_path, map_location='cpu')
with open(ids_path, 'rb') as handle:
    ids = torch.as_tensor(pickle.load(handle), dtype=torch.long).reshape(-1)
state_ids = torch.as_tensor(state.get('landmark_indices'), dtype=torch.long).reshape(-1)
config = state.get('config', {})
if ids.numel() != expected or not torch.equal(ids, state_ids):
    raise SystemExit('Wide residual state is not exactly aligned with the requested fixed 32K bank')
if str(config.get('observation_source')) != 'native' or not bool(config.get('native_outcome_mode')):
    raise SystemExit('Source state is not a pure-native residual checkpoint')
if config.get('split_mode') != expected_split or int(config.get('split_seed', -1)) != int(expected_seed):
    raise SystemExit(
        'Wide-bank source split does not match this study: '
        f"{config.get('split_mode')!r}/{config.get('split_seed')!r} != {expected_split!r}/{expected_seed!r}"
    )
if config.get('query_feature_contract') != 'native_resized_input':
    raise SystemExit('Wide-bank source does not use the frozen native sparse query contract')
if int(config.get('native_sparse_keypoint_count', -1)) != int(expected_keypoints):
    raise SystemExit('Wide-bank source does not use the frozen native keypoint capacity')
for key, expected_value in {
    'native_anchor_aux_weight': 0.0,
    'mv_weight': 0.0,
    'local_weight': 0.0,
    'dustbin_weight': 0.0,
    'geometry_weight': 0.0,
    'pose_weight': 0.0,
    'retrieval_weight': 1.0,
    'trust_weight': 0.02,
}.items():
    if abs(float(config.get(key, float('nan'))) - expected_value) > 1e-12:
        raise SystemExit(f'Wide-bank source {key} does not match the pure-native protocol')
if float(torch.as_tensor(state.get('raw_anchor_offset')).abs().max().item()) > 1e-12:
    raise SystemExit('This descriptor-distillation study requires a zero-anchor wide residual source')
selection = json.load(open(selection_path))
if selection.get('selection_protocol', {}).get('test_metrics_used') is not False:
    raise SystemExit('Wide-bank source selection must be validation-only')
if selection.get('selected_state') != state_path:
    raise SystemExit('Wide-bank source state does not match the validation-selected residual state')
protocol = json.load(open(protocol_path))
strict_formal = protocol.get('formal_protocol', {})
robust_formal = protocol.get('formal_deployment_protocol', {})
if strict_formal:
    if strict_formal.get('id') != 'lafgs_v2_fullres_native_uncapped_v1':
        raise SystemExit('Wide-bank source does not use the locked full-resolution native protocol')
    protocol_split = protocol.get('selection', {}).get('split_mode')
    protocol_seed = protocol.get('selection', {}).get('split_seed')
    protocol_frontend = strict_formal.get('sparse_frontend')
else:
    # Robust KCS/GWFF writes a study manifest rather than the strict runner's
    # protocol manifest.  It remains eligible only when the full deployment
    # contract is explicitly present and all fields match the frozen one.
    if not robust_formal or protocol.get('test_evaluation_forbidden') is not True:
        raise SystemExit('Wide-bank source has no validation-only formal deployment contract')
    protocol_split = robust_formal.get('candidate_split_mode')
    protocol_seed = robust_formal.get('candidate_split_seed')
    protocol_frontend = robust_formal.get('sparse_frontend')
    strict_formal = robust_formal

for key, expected_value in {
    'longest_edge': 0,
    'native_keypoints': int(expected_keypoints),
    'topk': 1,
    'max_matches_per_landmark': 0,
}.items():
    if int(strict_formal.get(key, -1)) != expected_value:
        raise SystemExit(f'Wide-bank source formal contract {key} does not match {expected_value}')
if abs(float(strict_formal.get('cosine_threshold', float('nan')))) > 1e-12:
    raise SystemExit('Wide-bank source formal contract does not use cosine threshold 0')
if protocol_frontend != 'ulfloc_native':
    raise SystemExit('Wide-bank source formal contract does not use the native ULF frontend')
if protocol_split != expected_split or int(protocol_seed) != int(expected_seed):
    raise SystemExit(
        'Wide-bank source split does not match this study: '
        f"{protocol_split!r}/{protocol_seed!r} != {expected_split!r}/{expected_seed!r}"
    )
print('Verified selected wide-bank native residual source: exact bank, pure-native, zero anchor offset, matching frozen split')
PY
}

write_manifest() {
  "$PYTHON" - "$MANIFEST" <<PY
import json
from pathlib import Path

payload = {
    "schema_version": 2,
    "purpose": "validation_only_widebank_residual_matchability_distill_short_refresh",
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
        "widebank_protocol_version": "${WIDEBANK_PROTOCOL_VERSION}",
    },
    "source": {
        "wide_residual_state": str(Path("${SOURCE_STATE}").resolve()),
        "wide_landmark_ids": str(Path("${SOURCE_IDS}").resolve()),
        "validation_selection": str(Path("${SOURCE_SELECTION}").resolve()),
        "protocol_manifest": str(Path("${SOURCE_PROTOCOL}").resolve()),
        "wide_budget": ${WIDE_BANK_BUDGET},
        "source_residual_steps": ${RESIDUAL_STEPS},
    },
    "distillation": {
        "final_budget": ${FINAL_BANK_BUDGET},
        "statistics_observations": 716,
        "matchability_first": True,
        "minimum_observations": 2,
        "matchability_threshold": 0.5,
        "hard_matchability_core_ratio": ${HARD_MATCHABILITY_CORE_RATIO},
        "quality_reservoir_multiplier": ${QUALITY_RESERVOIR_MULTIPLIER},
        "quality_reservoir_score": "${QUALITY_RESERVOIR_SCORE}",
        "quality_reservoir_wilson_z": ${QUALITY_RESERVOIR_WILSON_Z},
        "quality_reservoir_source": "observed_native_clean_match_statistics",
        "false_top1_max": 0.5,
        "rank_pool_multiplier": 1.5,
        "coverage": "3d_voxel+image_grid+depth+translation_fim",
        "shortfall_policy": "strict_matchability_then_explicit_coverage_fill",
        "coverage_fill_tiers": ["observed_remaining", "unobserved_remaining"],
    },
    "refresh": {"steps": ${REFRESH_STEPS}, "objective": "pure_native_keep_swap_miss_reject"},
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
    --validation_ratio "$VALIDATION_RATIO" --split_mode "$SPLIT_MODE" --split_seed "$SPLIT_SEED" \
    --train_seed 2026 --max_observations 512 --validation_observations 512 \
    --tangent_bound_m "$TANGENT_BOUND_M" --normal_bound_m "$NORMAL_BOUND_M"
}

append_base_map_args() {
  local -n target="$1"
  local item
  while IFS= read -r -d '' item; do
    target+=("$item")
  done < <(base_map_args)
}

validate_bank() {
  local state="$1"
  local ids="$2"
  local expected="$3"
  require_file "$state"
  require_file "$ids"
  "$PYTHON" - "$state" "$ids" "$expected" <<'PY'
import pickle
import sys
import torch

state_path, ids_path, expected = sys.argv[1:]
expected = int(expected)
state = torch.load(state_path, map_location='cpu')
with open(ids_path, 'rb') as handle:
    ids = torch.as_tensor(pickle.load(handle), dtype=torch.long).reshape(-1)
state_ids = torch.as_tensor(state.get('landmark_indices'), dtype=torch.long).reshape(-1)
if ids.numel() != expected or not torch.equal(ids, state_ids):
    raise SystemExit('State/ID alignment or bank capacity is invalid')
print(f'Validated fixed bank: {expected} landmarks')
PY
}

verify_stage_state_protocol() {
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
    raise SystemExit(f"Refusing to reuse wide-bank stage {label}: " + '; '.join(errors))
print(f'Verified frozen wide-bank stage protocol for {label}')
PY
}

verify_quality_reservoir_state() {
  if [[ "$QUALITY_RESERVOIR_MULTIPLIER" == "0" || "$QUALITY_RESERVOIR_MULTIPLIER" == "0.0" ]]; then
    return
  fi
  local state="$1"
  require_file "$state"
  "$PYTHON" - "$state" "$FINAL_BANK_BUDGET" "$QUALITY_RESERVOIR_MULTIPLIER" \
    "$QUALITY_RESERVOIR_SCORE" "$QUALITY_RESERVOIR_WILSON_Z" <<'PY'
import math
import sys

import torch

(
    state_path,
    expected_budget,
    expected_multiplier,
    expected_score,
    expected_wilson_z,
) = sys.argv[1:]
expected_budget = int(expected_budget)
expected_multiplier = float(expected_multiplier)
expected_wilson_z = float(expected_wilson_z)
state = torch.load(state_path, map_location='cpu')
config = dict(state.get('config', {}))
distillation = dict(config.get('distillation', {}))
meta = dict(state.get('selection_meta', {}))
errors = []
if not bool(distillation.get('quality_reservoir_active', False)):
    errors.append('quality reservoir is not marked active')
if not math.isclose(
    float(distillation.get('quality_reservoir_multiplier', float('nan'))),
    expected_multiplier,
    rel_tol=0.0,
    abs_tol=1e-12,
):
    errors.append('quality reservoir multiplier does not match the requested value')
if distillation.get('quality_reservoir_score_mode') != expected_score:
    errors.append('quality reservoir score mode does not match the requested value')
if not math.isclose(
    float(distillation.get('quality_reservoir_wilson_z', float('nan'))),
    expected_wilson_z,
    rel_tol=0.0,
    abs_tol=1e-12,
):
    errors.append('quality reservoir Wilson z does not match the requested value')
reservoir = torch.as_tensor(meta.get('quality_reservoir_indices', []), dtype=torch.long).reshape(-1)
selected = torch.as_tensor(meta.get('final_indices', []), dtype=torch.long).reshape(-1)
if reservoir.numel() <= expected_budget:
    errors.append(
        f'quality reservoir is inert ({reservoir.numel()} candidates for {expected_budget} slots)'
    )
if selected.numel() != expected_budget:
    errors.append(f'final local selection has {selected.numel()} rather than {expected_budget} entries')
reservoir_set = set(reservoir.tolist())
if not set(selected.tolist()).issubset(reservoir_set):
    errors.append('final selection leaks outside the quality reservoir')
if int(distillation.get('coverage_fill_count', -1)) != 0:
    errors.append('quality-reservoir run unexpectedly used global coverage fill')
if errors:
    raise SystemExit('Invalid quality-reservoir distillation: ' + '; '.join(errors))
print(
    'Verified active non-inert quality reservoir: '
    f'{reservoir.numel()} candidates -> {selected.numel()} landmarks'
)
PY
}

verify_validation_protocol() {
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
    raise SystemExit(f"Refusing to reuse wide-bank validation {label}: " + '; '.join(errors))
print(f'Verified frozen wide-bank validation protocol for {label}')
PY
}

native_outcome_args() {
  printf '%s\0' \
    --observation_source native --native_anchor_aux_weight 0 \
    --native_outcome_mode --native_nce_weight 0 \
    --native_keep_weight 1 --native_keep_margin 0.05 \
    --native_swap_weight 1 --native_swap_margin 0.05 \
    --native_miss_weight 1 --native_miss_margin 0.05 \
    --native_reject_weight 0.05 --native_reject_threshold "$MATCH_THRESHOLD" \
    --mv_weight 0 --retrieval_weight 1 --trust_weight 0.02 --local_weight 0 \
    --dustbin_weight 0 --geometry_weight 0 --pose_weight 0 --pose_gradient_mode off
}

append_native_outcome_args() {
  local -n target="$1"
  local item
  while IFS= read -r -d '' item; do
    target+=("$item")
  done < <(native_outcome_args)
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
    verify_validation_protocol "$label" "$ref"
    echo "[wide bank] Reusing protocol-matched validation: $label"
    return
  fi
  local cfg
  cfg="$(make_eval_config "$label" "$ids" "$meta" "$state")"
  run_logged "${label}_validation" \
    "$PYTHON" stdloc.py \
    --model_path "$MODEL_ROOT" --source_path "$DATA_ROOT/$SCENE" \
    --images processed --data_device cpu --gaussian_type 2dgs --feature_type sp \
    --resolution 1 --longest_edge "$LONGEST_EDGE" --norm_before_render --iteration 30000 \
    --cfg "$cfg" --prefix "lafgs-v2-widebank-${SCENE}-${label}-validation" --sparse_only \
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
  verify_validation_protocol "$label" "$ref"
}

distill() {
  verify_source
  write_manifest
  if [[ -f "$DISTILL_STATE" && -f "$DISTILL_IDS" && -f "$DISTILL_META" ]]; then
    validate_bank "$DISTILL_STATE" "$DISTILL_IDS" "$FINAL_BANK_BUDGET"
    verify_stage_state_protocol distill "$DISTILL_STATE" 0 0 0
    verify_quality_reservoir_state "$DISTILL_STATE"
    return
  fi
  local command=()
  append_base_map_args command
  command+=(
    --output_dir "$DISTILL_DIR" --scaffold_mode file --landmark_path "$SOURCE_IDS"
    --initial_state_path "$SOURCE_STATE" --initial_state_blend 1 --initial_state_alignment exact
    --initialization_mode ulf_parity --observation_source native --native_anchor_aux_weight 0
    --no-native_outcome_mode --steps 0 --save_steps 0
    --mv_weight 0 --retrieval_weight 0 --trust_weight 0 --local_weight 0 --dustbin_weight 0
    --geometry_weight 0 --pose_weight 0 --pose_gradient_mode off
    --visibility_cache_path "$DISTILL_VISIBILITY_CACHE"
    --distill_budget "$FINAL_BANK_BUDGET" --distill_require_exact_budget --distill_allow_coverage_fill
    --statistics_observations 716 --statistics_hypothesis_topk 32
    --distill_min_observations 2 --distill_matchability_threshold 0.5 --distill_false_top1_max 0.5
    --distill_proposal_weight 1 --distill_rank_pool_multiplier 1.5
    --distill_matchability_preserve_ratio 0.30 --distill_utility_preserve_ratio 0.35
    --distill_hard_matchability_core_ratio "$HARD_MATCHABILITY_CORE_RATIO"
    --distill_quality_reservoir_multiplier "$QUALITY_RESERVOIR_MULTIPLIER"
    --distill_quality_reservoir_score "$QUALITY_RESERVOIR_SCORE"
    --distill_quality_reservoir_wilson_z "$QUALITY_RESERVOIR_WILSON_Z"
    --distill_grid_size 8 --distill_max_per_grid 512 --distill_depth_bins 8 --distill_max_per_depth_bin 4096
  )
  run_logged distill "${command[@]}"
  validate_bank "$DISTILL_STATE" "$DISTILL_IDS" "$FINAL_BANK_BUDGET"
  verify_stage_state_protocol distill "$DISTILL_STATE" 0 0 0
  verify_quality_reservoir_state "$DISTILL_STATE"
}

distill_validate() {
  distill
  run_eval distill "$DISTILL_IDS" "$DISTILL_META" "$DISTILL_STATE"
}

refresh() {
  distill_validate
  local state="$REFRESH_DIR/${REFRESH_STEPS}_lafgs_map_state.pt"
  if [[ -f "$state" ]]; then
    validate_bank "$state" "$DISTILL_IDS" "$FINAL_BANK_BUDGET"
    verify_stage_state_protocol refresh "$state" 1 1 0.02
    return
  fi
  local command=()
  append_base_map_args command
  append_native_outcome_args command
  command+=(
    --output_dir "$REFRESH_DIR" --scaffold_mode file --landmark_path "$DISTILL_IDS"
    --initial_state_path "$DISTILL_STATE" --initial_state_blend 1 --initial_state_alignment exact
    --initialization_mode ulf_parity --visibility_cache_path "$REFRESH_VISIBILITY_CACHE"
    --steps "$REFRESH_STEPS" --save_steps 500 "$REFRESH_STEPS"
    --feature_lr 2.5e-5 --weight_decay 1e-4 --hypothesis_topk 32
    --positive_radius_px 2 --negative_radius_px 6 --log_interval 100
  )
  run_logged refresh "${command[@]}"
  validate_bank "$state" "$DISTILL_IDS" "$FINAL_BANK_BUDGET"
  verify_stage_state_protocol refresh "$state" 1 1 0.02
}

refresh_validate() {
  refresh
  local step
  for step in 500 "$REFRESH_STEPS"; do
    local state="$REFRESH_DIR/${step}_lafgs_map_state.pt"
    if [[ -f "$state" ]]; then
      run_eval "refresh_${step}" "$DISTILL_IDS" "$DISTILL_META" "$state"
    fi
  done
}

case "$MODE" in
  distill) distill ;;
  distill_validate) distill_validate ;;
  refresh) refresh ;;
  refresh_validate) refresh_validate ;;
  all) refresh_validate ;;
esac
