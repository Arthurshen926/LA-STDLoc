#!/usr/bin/env bash
set -euo pipefail

# Validation-only clean-correspondence protection ablation for the historical
# LaFGS-native deployment matcher. Every variant starts from the exact same
# frozen 20K KCS/GWFF bootstrap and differs only in native keep/reject weights.
# It deliberately has no test mode: a held-out test is reserved for a state
# selected after this validation-only experiment.

if [[ $# -ne 3 ]]; then
  echo "Usage: $0 <scene> <gpu> <keep4_reject002|keep8_reject002|keep4_reject005|keep4_margin010_reject005>" >&2
  exit 2
fi

SCENE="$1"
GPU="$2"
VARIANT="$3"

case "$SCENE" in
  GreatCourt|KingsCollege|OldHospital|ShopFacade|StMarysChurch) ;;
  *) echo "Unsupported Cambridge scene: $SCENE" >&2; exit 2 ;;
esac
case "$GPU" in
  0|1|2) ;;
  *) echo "GPU must be 0, 1, or 2; got $GPU" >&2; exit 2 ;;
esac
case "$VARIANT" in
  keep4_reject002|keep8_reject002|keep4_reject005|keep4_margin010_reject005) ;;
  *) echo "Unsupported protection variant: $VARIANT" >&2; exit 2 ;;
esac

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/root/miniconda3/envs/cybersim_agent/bin/python}"
DATA_ROOT="${CAMBRIDGE_DATA_ROOT:-/mnt/pool/sqy/Cambridge_stdloc}"
REFERENCE_ROOT="${LAFGS_V2_ULFPARITY_REFERENCE_ROOT:-/mnt/pool/sqy/stdloc_lafgs_v2_ulfparity_alternating_20260721}"
EXPERIMENT_ROOT="${LAFGS_V2_NATIVE_ABLATION_ROOT:-/mnt/pool/sqy/stdloc_lafgs_v2_native_protection_20260722}"

LANDMARK_BUDGET="${LAFGS_ULF_LANDMARK_BUDGET:-20000}"
SUPPORT_VIEWS="${LAFGS_ULF_SUPPORT_VIEWS:-128}"
NATIVE_KEYPOINTS="${LAFGS_ULF_NATIVE_KEYPOINTS:-2048}"
VALIDATION_RATIO="${LAFGS_ULF_VALIDATION_RATIO:-0.2}"
SPLIT_MODE="${LAFGS_ULF_SPLIT_MODE:-temporal_block}"
SPLIT_SEED="${LAFGS_ULF_SPLIT_SEED:-2026}"
STEPS="${LAFGS_NATIVE_ABLATION_STEPS:-5000}"
NATIVE_MATCH_THRESHOLD="${LAFGS_NATIVE_ABLATION_MATCH_THRESHOLD:-0.4}"
NATIVE_MATCH_THRESHOLD_TAG="${NATIVE_MATCH_THRESHOLD//-/m}"
NATIVE_MATCH_THRESHOLD_TAG="${NATIVE_MATCH_THRESHOLD_TAG//./p}"
# This is the already-evaluated 20K ULF-initialized deployment control. Its
# matcher uses tau=0.4 and a two-query landmark cap, so it is not the strict
# external ULF matcher control. Do not change its IDs, feature state, query
# cache, or validation split in an objective study.
REFERENCE_TAG="${LAFGS_NATIVE_ABLATION_REFERENCE_TAG:-ulfparity_native20k_s128_k2048_tau0p4_v5_pure_native_explicit_aux0}"

case "$SCENE" in
  GreatCourt) MATCHA_RUN="GreatCourt_n20_long_masked_retrain_retry" ;;
  KingsCollege) MATCHA_RUN="KingsCollege_n20_long_masked_retrain" ;;
  OldHospital) MATCHA_RUN="OldHospital_n20_long_masked_retrain_retry" ;;
  ShopFacade) MATCHA_RUN="ShopFacade_n20_long_masked_retrain" ;;
  StMarysChurch) MATCHA_RUN="StMarysChurch_n20_long_masked_retrain" ;;
esac

REFERENCE_RUN="$REFERENCE_ROOT/$SCENE/$REFERENCE_TAG"
MODEL_ROOT="$REFERENCE_ROOT/matcha_wrappers/$SCENE"
BOOTSTRAP_DIR="$REFERENCE_RUN/bootstrap"
BOOTSTRAP_STATE="$BOOTSTRAP_DIR/0_lafgs_map_state.pt"
BOOTSTRAP_IDS="$BOOTSTRAP_DIR/sampled_idx.pkl"
BOOTSTRAP_META="$BOOTSTRAP_DIR/landmark_meta.pt"
CONTROL_STATE="$REFERENCE_RUN/residual_5000/5000_lafgs_map_state.pt"
REFERENCE_RESIDUAL_MANIFEST="$REFERENCE_RUN/residual_5000/reproducibility_manifest.json"
if [[ ! -f "$REFERENCE_RESIDUAL_MANIFEST" ]]; then
  echo "Reference residual reproducibility manifest is missing: $REFERENCE_RESIDUAL_MANIFEST" >&2
  exit 1
fi
mapfile -t REFERENCE_CACHE_PATHS < <(
  "$PYTHON" - "$REFERENCE_RESIDUAL_MANIFEST" <<'PY'
import json
import sys

manifest = json.load(open(sys.argv[1], "r", encoding="utf-8"))
inputs = manifest.get("inputs", {})
for key in ("query_cache_path", "visibility_cache_path"):
    value = inputs.get(key, {})
    path = value.get("path") if isinstance(value, dict) else None
    if not path:
        raise SystemExit(f"reference manifest does not record {key}")
    print(path)
PY
)
if [[ "${#REFERENCE_CACHE_PATHS[@]}" -ne 2 ]]; then
  echo "Reference residual manifest did not resolve exactly two cache paths" >&2
  exit 1
fi
QUERY_CACHE="${LAFGS_NATIVE_ABLATION_QUERY_CACHE_PATH:-${REFERENCE_CACHE_PATHS[0]}}"
VISIBILITY_CACHE="${LAFGS_NATIVE_ABLATION_VISIBILITY_CACHE_PATH:-${REFERENCE_CACHE_PATHS[1]}}"
mapfile -t CONTROL_RESULT_CANDIDATES < <(
  find "$REFERENCE_RUN/stdloc_results" -type f \
    -path "*-residual_5000-validation-*/results_summary.json" | sort
)
if [[ "${#CONTROL_RESULT_CANDIDATES[@]}" -ne 1 ]]; then
  echo "Expected one immutable pure-native residual validation summary under $REFERENCE_RUN, found ${#CONTROL_RESULT_CANDIDATES[@]}" >&2
  exit 1
fi
CONTROL_RESULTS="${CONTROL_RESULT_CANDIDATES[0]}"

case "$VARIANT" in
  keep4_reject002)
    KEEP_WEIGHT=4
    REJECT_WEIGHT=0.02
    KEEP_MARGIN_DEFAULT=0.05
    ;;
  keep8_reject002)
    KEEP_WEIGHT=8
    REJECT_WEIGHT=0.02
    KEEP_MARGIN_DEFAULT=0.05
    ;;
  keep4_reject005)
    KEEP_WEIGHT=4
    REJECT_WEIGHT=0.05
    KEEP_MARGIN_DEFAULT=0.05
    ;;
  keep4_margin010_reject005)
    KEEP_WEIGHT=4
    REJECT_WEIGHT=0.05
    KEEP_MARGIN_DEFAULT=0.10
    ;;
esac
KEEP_MARGIN="${LAFGS_NATIVE_ABLATION_KEEP_MARGIN:-$KEEP_MARGIN_DEFAULT}"
SWAP_WEIGHT="${LAFGS_NATIVE_ABLATION_SWAP_WEIGHT:-1}"
SWAP_MARGIN="${LAFGS_NATIVE_ABLATION_SWAP_MARGIN:-0.05}"
MISS_WEIGHT="${LAFGS_NATIVE_ABLATION_MISS_WEIGHT:-1}"
MISS_MARGIN="${LAFGS_NATIVE_ABLATION_MISS_MARGIN:-0.05}"
TRUST_WEIGHT="${LAFGS_NATIVE_ABLATION_TRUST_WEIGHT:-0.02}"

RUN_ROOT="$EXPERIMENT_ROOT/$SCENE/${VARIANT}_from_${REFERENCE_TAG}_s${SUPPORT_VIEWS}_k${NATIVE_KEYPOINTS}_tau${NATIVE_MATCH_THRESHOLD_TAG}_v1"
STATE_DIR="$RUN_ROOT/residual_${STEPS}"
LOG_ROOT="$RUN_ROOT/logs"
CONFIG_ROOT="$RUN_ROOT/configs"
RESULT_ROOT="$RUN_ROOT/results"
STDLOC_RESULTS_ROOT="$RUN_ROOT/stdloc_results"
MANIFEST="$RUN_ROOT/protocol_manifest.json"
SELECTION="$RUN_ROOT/validation_selection.json"

export CUDA_VISIBLE_DEVICES="$GPU"
export CUDA_HOME=/usr/local/cuda-11.8
export PATH="/root/miniconda3/envs/cybersim_agent/bin:$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="/root/miniconda3/envs/cybersim_agent/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$REPO_ROOT"
export PYTHONHASHSEED=2026
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export STDLOC_RESULTS_ROOT

mkdir -p "$STATE_DIR" "$LOG_ROOT" "$CONFIG_ROOT" "$RESULT_ROOT" "$STDLOC_RESULTS_ROOT"
cd "$REPO_ROOT"

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "Required immutable input is missing: $1" >&2
    exit 1
  fi
}

validate_numeric_contract() {
  "$PYTHON" - \
    "$NATIVE_MATCH_THRESHOLD" "$KEEP_WEIGHT" "$REJECT_WEIGHT" "$KEEP_MARGIN" \
    "$SWAP_WEIGHT" "$SWAP_MARGIN" "$MISS_WEIGHT" "$MISS_MARGIN" "$TRUST_WEIGHT" <<'PY'
import math
import sys

names = (
    "native match threshold", "keep weight", "reject weight", "keep margin",
    "swap weight", "swap margin", "miss weight", "miss margin", "trust weight",
)
values = [float(value) for value in sys.argv[1:]]
for name, value in zip(names, values):
    if not math.isfinite(value):
        raise SystemExit(f"{name} must be finite")
if not -1.0 <= values[0] <= 1.0:
    raise SystemExit("native match threshold must be a cosine score in [-1, 1]")
if any(value < 0.0 for value in values[1:]):
    raise SystemExit("native objective weights and margins must be non-negative")
PY
}

validate_bootstrap_contract() {
  "$PYTHON" - "$BOOTSTRAP_STATE" "$BOOTSTRAP_IDS" "$LANDMARK_BUDGET" <<'PY'
import pickle
import sys

import torch

state_path, ids_path, expected = sys.argv[1:]
state = torch.load(state_path, map_location="cpu")
config = state.get("config", {})
if config.get("initialization_mode") != "ulf_parity":
    raise SystemExit("reference bootstrap is not an ULF-parity initialization")
if int(config.get("distill_budget", -1)) != 0:
    raise SystemExit("reference bootstrap was distilled before residual training")
ids = pickle.load(open(ids_path, "rb"))
count = int(ids.numel() if hasattr(ids, "numel") else len(ids))
if count != int(expected):
    raise SystemExit(f"reference bootstrap has {count} IDs, expected {expected}")
print(f"Verified immutable ULF-parity bootstrap: {count} landmarks")
PY
}

validate_numeric_contract
require_file "$BOOTSTRAP_STATE"
require_file "$BOOTSTRAP_IDS"
require_file "$BOOTSTRAP_META"
require_file "$QUERY_CACHE"
require_file "$VISIBILITY_CACHE"
require_file "$CONTROL_STATE"
require_file "$CONTROL_RESULTS"
validate_bootstrap_contract

write_manifest() {
  "$PYTHON" - "$MANIFEST" <<PY
import hashlib
import json
from pathlib import Path

def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

payload = {
    "schema_version": 3,
    "purpose": "validation_only_native_clean_correspondence_protection",
    "test_evaluation_forbidden": True,
    "scene": "${SCENE}",
    "variant": "${VARIANT}",
    "immutable_bootstrap": {
        "state": "${BOOTSTRAP_STATE}",
        "state_sha256": sha256("${BOOTSTRAP_STATE}"),
        "landmark_ids": "${BOOTSTRAP_IDS}",
        "landmark_ids_sha256": sha256("${BOOTSTRAP_IDS}"),
        "landmark_meta": "${BOOTSTRAP_META}",
        "query_cache": "${QUERY_CACHE}",
        "visibility_cache": "${VISIBILITY_CACHE}",
        "landmark_count": ${LANDMARK_BUDGET},
        "support_views": ${SUPPORT_VIEWS},
        "early_distillation": False,
        "historical_matcher_profile": "lafgs_native_tau0.4_cap2",
    },
    "control": {
        "tag": "pure_native_keep1_reject005",
        "state": "${CONTROL_STATE}",
        "results": "${CONTROL_RESULTS}",
    },
    "residual": {
        "query_frontend": "SuperPoint.detectAndCompute",
        "candidate_set": "detached_full_bank_cosine_topk_before_gt_labels",
        "gt_source_injected_into_candidates": False,
        "outcomes": ["keep", "swap", "miss", "reject"],
        "observation_source": "native",
        "native_anchor_aux_weight": 0.0,
        "mv_weight": 0.0,
        "trust_weight": ${TRUST_WEIGHT},
        "native_keep_weight": ${KEEP_WEIGHT},
        "native_keep_margin": ${KEEP_MARGIN},
        "native_swap_weight": ${SWAP_WEIGHT},
        "native_swap_margin": ${SWAP_MARGIN},
        "native_miss_weight": ${MISS_WEIGHT},
        "native_miss_margin": ${MISS_MARGIN},
        "native_reject_weight": ${REJECT_WEIGHT},
        "deployment_match_threshold": ${NATIVE_MATCH_THRESHOLD},
        "training_reject_threshold": ${NATIVE_MATCH_THRESHOLD},
        "threshold_contract": "exact_direct_cosine_score_v1",
    },
    "evaluation": {
        "subset": "candidate_validation",
        "ratio": ${VALIDATION_RATIO},
        "split_mode": "${SPLIT_MODE}",
        "split_seed": ${SPLIT_SEED},
        "checkpoints": [500, 1000, 2500, ${STEPS}],
    },
}
path = Path("${MANIFEST}")
path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\\n")
print(path)
PY
}

run_logged() {
  local stage="$1"
  shift
  printf '%q ' "$@" > "$LOG_ROOT/${stage}.command.sh"
  printf '\n' >> "$LOG_ROOT/${stage}.command.sh"
  "$@" 2>&1 | tee "$LOG_ROOT/${stage}.log"
}

train() {
  local final_state="$STATE_DIR/${STEPS}_lafgs_map_state.pt"
  if [[ -f "$final_state" ]]; then
    echo "[native protection] Reusing training state: $final_state"
    return
  fi
  run_logged train \
    "$PYTHON" train_lafgs_map.py \
    --model_path "$MODEL_ROOT" --source_path "$DATA_ROOT/$SCENE" \
    --images processed --data_device cpu --gaussian_type 2dgs --feature_type sp \
    --resolution 1 --longest_edge 0 --norm_before_render --load_iteration 30000 \
    --query_feature_contract native_resized_input \
    --query_cache_path "$QUERY_CACHE" --visibility_cache_path "$VISIBILITY_CACHE" \
    --visibility_mode rasterizer --objective hard --native_keypoint_count "$NATIVE_KEYPOINTS" \
    --native_association_radius_px 2 --native_sampling_mode detector_grid \
    --generic_proposal_count 0 --generic_proposal_weight 0 --distill_budget 0 \
    --validation_ratio "$VALIDATION_RATIO" --split_mode "$SPLIT_MODE" --split_seed "$SPLIT_SEED" \
    --train_seed 2026 --max_observations 512 --validation_observations 512 \
    --observation_source native --native_anchor_aux_weight 0 \
    --native_outcome_mode --native_nce_weight 0 \
    --native_keep_weight "$KEEP_WEIGHT" --native_keep_margin "$KEEP_MARGIN" \
    --native_swap_weight "$SWAP_WEIGHT" --native_swap_margin "$SWAP_MARGIN" \
    --native_miss_weight "$MISS_WEIGHT" --native_miss_margin "$MISS_MARGIN" \
    --native_reject_weight "$REJECT_WEIGHT" --native_reject_threshold "$NATIVE_MATCH_THRESHOLD" \
    --mv_weight 0 --retrieval_weight 1 --trust_weight "$TRUST_WEIGHT" --local_weight 0 \
    --dustbin_weight 0 --geometry_weight 0 --pose_weight 0 --pose_gradient_mode off \
    --output_dir "$STATE_DIR" --scaffold_mode file --landmark_path "$BOOTSTRAP_IDS" \
    --initial_state_path "$BOOTSTRAP_STATE" --initial_state_blend 1 --initial_state_alignment exact \
    --initialization_mode ulf_parity --query_cache_policy readonly \
    --steps "$STEPS" --save_steps 500 1000 2500 "$STEPS" \
    --feature_lr 5e-5 --weight_decay 1e-4 --hypothesis_topk 32 \
    --positive_radius_px 2 --negative_radius_px 6 --log_interval 100
  require_file "$final_state"
}

eval_checkpoint() {
  local step="$1"
  local state="$STATE_DIR/${step}_lafgs_map_state.pt"
  local reference="$RESULT_ROOT/${step}.results_path"
  [[ -f "$state" ]] || return
  if [[ -f "$reference" && -f "$(<"$reference")/results_summary.json" ]]; then
    echo "[native protection] Reusing validation: $VARIANT/$step"
    return
  fi
  local config="$CONFIG_ROOT/${step}.yaml"
  "$PYTHON" scripts/make_stdloc_eval_cfg.py \
    --base_cfg configs/stdloc_cambridge.yaml --output "$config" \
    --artifact_model_path "$MODEL_ROOT" --detector_folder ulfloc_native_no_detector --detector_iters 0 \
    --landmark_path "$BOOTSTRAP_IDS" --landmark_meta_path "$BOOTSTRAP_META" \
    --landmark_feature_override_path "$state" --override_landmark_features \
    --detect_num "$NATIVE_KEYPOINTS" --nms 2 --sparse_query_feature_contract native_resized_input \
    --sparse_frontend ulfloc_native --reprojection_error 12 --match_threshold "$NATIVE_MATCH_THRESHOLD" --match_topk 1 \
    --max_matches_per_landmark 2 --candidate_frontend_match_policy error --diagnostics \
    --diagnostics_grid_rows 4 --diagnostics_grid_cols 4 --diagnostics_voxel_size 1.0 \
    --diagnostics_task_translation_scale_m 0.07160573943725686 \
    --diagnostics_task_rotation_scale_degrees 2.0 \
    > "$LOG_ROOT/${step}_config.json"
  run_logged "eval_${step}" \
    "$PYTHON" stdloc.py \
    --model_path "$MODEL_ROOT" --source_path "$DATA_ROOT/$SCENE" \
    --images processed --data_device cpu --gaussian_type 2dgs --feature_type sp \
    --resolution 1 --longest_edge 0 --norm_before_render --iteration 30000 \
    --cfg "$config" --prefix "lafgs-v2-native-protection-${SCENE}-${VARIANT}-${step}" --sparse_only \
    --evaluation_camera_subset candidate_validation --candidate_direct_validation_holdout \
    --candidate_validation_ratio "$VALIDATION_RATIO" --candidate_split_mode "$SPLIT_MODE" \
    --candidate_split_seed "$SPLIT_SEED"
  local output_path
  output_path="$(sed -n 's/^Output path: //p' "$LOG_ROOT/eval_${step}.log" | tail -n 1)"
  if [[ -z "$output_path" || ! -f "$output_path/results_summary.json" ]]; then
    echo "Validation did not create results_summary.json for $VARIANT/$step" >&2
    exit 1
  fi
  printf '%s\n' "$output_path" > "$reference"
}

result_summary() {
  local step="$1"
  local reference="$RESULT_ROOT/${step}.results_path"
  require_file "$reference"
  local directory
  directory="$(<"$reference")"
  require_file "$directory/results_summary.json"
  printf '%s\n' "$directory/results_summary.json"
}

select_checkpoint() {
  if [[ -f "$SELECTION" ]]; then
    echo "[native protection] Reusing selection: $SELECTION"
    return
  fi
  local command=(
    "$PYTHON" scripts/select_lafgs_map_checkpoint.py
    --control_results "$CONTROL_RESULTS" --control_state "$CONTROL_STATE"
    --control_tag pure_native_keep1_reject005 --selection_mode performance
    --min_te_gain_cm 0.02 --mean_te_weight 0.05
    --max_recall_2m_drop 0.01 --max_recall_5cm_drop 0.01 --output "$SELECTION"
  )
  local step
  for step in 500 1000 2500 "$STEPS"; do
    if [[ -f "$STATE_DIR/${step}_lafgs_map_state.pt" && -f "$RESULT_ROOT/${step}.results_path" ]]; then
      command+=(--candidate "${VARIANT}_${step}" "$(result_summary "$step")" "$STATE_DIR/${step}_lafgs_map_state.pt")
    fi
  done
  run_logged select "${command[@]}"
  require_file "$SELECTION"
}

write_manifest
train
for step in 500 1000 2500 "$STEPS"; do
  eval_checkpoint "$step"
done
select_checkpoint
