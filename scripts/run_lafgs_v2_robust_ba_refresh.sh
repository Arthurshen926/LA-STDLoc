#!/usr/bin/env bash
set -euo pipefail

# Validation-only bounded surface BA and descriptor refresh for a robust
# KCS/GWFF residual selected on the candidate holdout.  This is deliberately
# separate from the strict-ULF runner: it must never reinterpret a robust
# initializer as a parity bootstrap, and it never evaluates the official test
# split.

if [[ $# -ne 3 ]]; then
  echo "Usage: $0 <scene> <gpu> <ba|ba_validate|select_ba|refresh|refresh_validate|select_refresh|terminal_refresh|terminal_refresh_validate|select_terminal_refresh|terminal_all|all>" >&2
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
  ba|ba_validate|select_ba|refresh|refresh_validate|select_refresh|terminal_refresh|terminal_refresh_validate|select_terminal_refresh|terminal_all|all) ;;
  *) echo "Unsupported robust BA mode: $MODE" >&2; exit 2 ;;
esac

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/root/miniconda3/envs/cybersim_agent/bin/python}"
REFERENCE_ROOT="${LAFGS_V2_ULFPARITY_REFERENCE_ROOT:-/mnt/pool/sqy/stdloc_lafgs_v2_ulfparity_alternating_20260721}"
ROBUST_ROOT="${LAFGS_V2_ROBUST_INITIALIZER_ROOT:-/mnt/pool/sqy/stdloc_lafgs_v2_robust_initializer_20260722}"
MODEL_ROOT="$REFERENCE_ROOT/matcha_wrappers/$SCENE"

DEFAULT_SOURCE_ROOT=""
if [[ "$SCENE" == "KingsCollege" ]]; then
  DEFAULT_SOURCE_ROOT="$ROBUST_ROOT/KingsCollege/robustkcs_gwff32000_s0_uniform_mv4_v2_r0p01_vb4m2_tb4m2_t0p1_dcm1p0_h64_support_rgb_only"
fi
SOURCE_ROOT="${LAFGS_ROBUST_BA_SOURCE_ROOT:-$DEFAULT_SOURCE_ROOT}"
if [[ -z "$SOURCE_ROOT" ]]; then
  echo "Set LAFGS_ROBUST_BA_SOURCE_ROOT for $SCENE" >&2
  exit 2
fi

SOURCE_MANIFEST="$SOURCE_ROOT/study_manifest.json"
SOURCE_SELECTION="$SOURCE_ROOT/results/residual_selection_safety.json"
DEFAULT_BOOTSTRAP_DIR="$SOURCE_ROOT/bootstrap"
# Multi-seed residual runs intentionally share a frozen bootstrap instead of
# copying 32K map artifacts into every seed namespace.  Resolve that explicit
# provenance before falling back to the historical colocated layout.
if [[ -f "$SOURCE_MANIFEST" ]]; then
  manifest_bootstrap_dir="$($PYTHON - "$SOURCE_MANIFEST" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], "r", encoding="utf-8"))
path = payload.get("optimization", {}).get("bootstrap_source_dir")
print(path or "")
PY
)"
  if [[ -n "$manifest_bootstrap_dir" ]]; then
    DEFAULT_BOOTSTRAP_DIR="$manifest_bootstrap_dir"
  fi
fi
BOOTSTRAP_DIR="${LAFGS_ROBUST_BA_BOOTSTRAP_DIR:-$DEFAULT_BOOTSTRAP_DIR}"
BOOTSTRAP_IDS="$BOOTSTRAP_DIR/sampled_idx.pkl"
BOOTSTRAP_META="$BOOTSTRAP_DIR/landmark_meta.pt"
SOURCE_PLY="$MODEL_ROOT/point_cloud/iteration_30000/point_cloud.ply"

VALIDATION_RATIO="${LAFGS_ROBUST_BA_VALIDATION_RATIO:-0.2}"
SPLIT_MODE="${LAFGS_ROBUST_BA_SPLIT_MODE:-stratified_temporal_block}"
SPLIT_SEED="${LAFGS_ROBUST_BA_SPLIT_SEED:-2026}"
BA_STEPS="${LAFGS_ROBUST_BA_STEPS:-1500}"
REFRESH_STEPS="${LAFGS_ROBUST_REFRESH_STEPS:-1000}"
TANGENT_BOUND_M="${LAFGS_ROBUST_BA_TANGENT_BOUND_M:-0.003}"
NORMAL_BOUND_M="${LAFGS_ROBUST_BA_NORMAL_BOUND_M:-0.0015}"
DEPTH_ABS_TOLERANCE="${LAFGS_ROBUST_BA_DEPTH_ABS_TOLERANCE:-0}"
DEPTH_REL_TOLERANCE="${LAFGS_ROBUST_BA_DEPTH_REL_TOLERANCE:-0}"
CAMERA_LOADER_WORKERS="${LAFGS_ROBUST_BA_CAMERA_LOADER_WORKERS:-0}"

if ! [[ "$CAMERA_LOADER_WORKERS" =~ ^[0-9]+$ ]]; then
  echo "LAFGS_ROBUST_BA_CAMERA_LOADER_WORKERS must be a non-negative integer" >&2
  exit 2
fi
if [[ "$SPLIT_MODE" != "stratified_temporal_block" ]]; then
  echo "Robust BA requires the source stratified_temporal_block holdout" >&2
  exit 2
fi

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "Required artifact is missing: $1" >&2
    exit 1
  fi
}

require_file "$SOURCE_MANIFEST"
require_file "$SOURCE_SELECTION"
require_file "$BOOTSTRAP_IDS"
require_file "$BOOTSTRAP_META"
require_file "$SOURCE_PLY"
require_file "$MODEL_ROOT/artifact_provenance.json"

LANDMARK_BUDGET="$($PYTHON - "$BOOTSTRAP_IDS" <<'PY'
import pickle
import sys

value = pickle.load(open(sys.argv[1], "rb"))
print(int(value.numel() if hasattr(value, "numel") else len(value)))
PY
)"
SOURCE_QUERY_CACHE="$($PYTHON - "$SOURCE_MANIFEST" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], "r", encoding="utf-8"))
path = payload.get("inputs", {}).get("query_cache")
if not path:
    raise SystemExit("robust source manifest has no query-cache path")
print(path)
PY
)"
QUERY_CACHE="${LAFGS_ROBUST_BA_QUERY_CACHE_PATH:-$SOURCE_QUERY_CACHE}"
DEFAULT_VISIBILITY_CACHE="$SOURCE_ROOT/visibility_${LANDMARK_BUDGET}_native.pt"
# Like the bootstrap IDs, a multi-seed residual can reference the original
# rasterizer cache rather than duplicating it under every seed namespace.
manifest_visibility_cache="$($PYTHON - "$SOURCE_MANIFEST" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], "r", encoding="utf-8"))
path = payload.get("optimization", {}).get("visibility_cache")
print(path or "")
PY
)"
if [[ -n "$manifest_visibility_cache" ]]; then
  DEFAULT_VISIBILITY_CACHE="$manifest_visibility_cache"
fi
VISIBILITY_CACHE="${LAFGS_ROBUST_BA_VISIBILITY_CACHE_PATH:-$DEFAULT_VISIBILITY_CACHE}"

tag_value() {
  local value="$1"
  value="${value//-/m}"
  value="${value//./p}"
  printf '%s' "$value"
}

GEOMETRY_ROOT="${LAFGS_ROBUST_BA_OUTPUT_ROOT:-$SOURCE_ROOT/bounded_ba_${BA_STEPS}_t$(tag_value "$TANGENT_BOUND_M")_n$(tag_value "$NORMAL_BOUND_M")}"
BA_DIR="$GEOMETRY_ROOT/ba_${BA_STEPS}"
REFRESH_DIR="$GEOMETRY_ROOT/refresh_${REFRESH_STEPS}"
# This is intentionally distinct from REFRESH_DIR.  It validates the
# predeclared final BA checkpoint followed by a native descriptor refresh,
# even when an intermediate BA-only checkpoint is not independently safe to
# promote.  The final state is still selected directly against the original
# residual with the same validation-only five-metric gate.
TERMINAL_REFRESH_DIR="$GEOMETRY_ROOT/terminal_ba_${BA_STEPS}_refresh_${REFRESH_STEPS}"
CONFIG_ROOT="$GEOMETRY_ROOT/configs"
LOG_ROOT="$GEOMETRY_ROOT/logs"
RESULT_ROOT="$GEOMETRY_ROOT/results"
STDLOC_RESULTS_ROOT="$GEOMETRY_ROOT/stdloc_results"
MANIFEST="$GEOMETRY_ROOT/protocol_manifest.json"
BA_SELECTION="$RESULT_ROOT/ba_selection_safety.json"
REFRESH_SELECTION="$RESULT_ROOT/refresh_selection_safety.json"
TERMINAL_REFRESH_SELECTION="$RESULT_ROOT/terminal_ba_refresh_selection_safety.json"

export CUDA_VISIBLE_DEVICES="$GPU"
export CUDA_HOME=/usr/local/cuda-11.8
export PATH="/root/miniconda3/envs/cybersim_agent/bin:$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="/root/miniconda3/envs/cybersim_agent/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$REPO_ROOT"
export PYTHONHASHSEED=2026
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export STDLOC_CAMERA_LOADER_WORKERS="$CAMERA_LOADER_WORKERS"
export STDLOC_RESULTS_ROOT

mkdir -p "$GEOMETRY_ROOT" "$CONFIG_ROOT" "$LOG_ROOT" "$RESULT_ROOT" "$STDLOC_RESULTS_ROOT"
cd "$REPO_ROOT"

run_logged() {
  local stage="$1"
  shift
  printf '%q ' "$@" > "$LOG_ROOT/${stage}.command.sh"
  printf '\n' >> "$LOG_ROOT/${stage}.command.sh"
  "$@" 2>&1 | tee "$LOG_ROOT/${stage}.log"
}

selected_state() {
  "$PYTHON" - "$SOURCE_SELECTION" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], "r", encoding="utf-8"))
if payload.get("selection_protocol", {}).get("test_metrics_used") is not False:
    raise SystemExit("robust residual selection used test metrics")
if payload.get("used_control_fallback", payload.get("used_strong_fallback", False)):
    raise SystemExit("robust residual did not pass the validation-only safety gate")
print(payload["selected_state"])
PY
}

selected_label() {
  "$PYTHON" - "$SOURCE_SELECTION" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], "r", encoding="utf-8"))
print(payload["selected_tag"])
PY
}

verify_selected_residual() {
  local state
  state="$(selected_state)"
  require_file "$state"
  require_file "$QUERY_CACHE"
  require_file "$VISIBILITY_CACHE"
  "$PYTHON" - "$SOURCE_MANIFEST" "$SOURCE_SELECTION" "$state" "$BOOTSTRAP_IDS" "$LANDMARK_BUDGET" "$SPLIT_MODE" <<'PY'
import json
import pickle
import sys

import torch

manifest_path, selection_path, state_path, ids_path, expected_count, expected_split = sys.argv[1:]
manifest = json.load(open(manifest_path, "r", encoding="utf-8"))
formal = manifest.get("formal_deployment_protocol", {})
if (
    int(formal.get("longest_edge", -1)) != 0
    or int(formal.get("native_keypoints", -1)) != 2048
    or float(formal.get("cosine_threshold", 1.0)) != 0.0
    or int(formal.get("max_matches_per_landmark", -1)) != 0
):
    raise SystemExit("robust source is not the locked full-resolution uncapped protocol")
selection = json.load(open(selection_path, "r", encoding="utf-8"))
if selection.get("selection_protocol", {}).get("test_metrics_used") is not False:
    raise SystemExit("robust source selection is not validation-only")
if selection.get("used_control_fallback", selection.get("used_strong_fallback", False)):
    raise SystemExit("bounded BA requires an independently accepted residual")
state = torch.load(state_path, map_location="cpu")
config = state.get("config", {})
if str(config.get("split_mode")) != expected_split:
    raise SystemExit("residual split does not match bounded-BA split")
if str(config.get("observation_source")) != "native" or not bool(config.get("native_outcome_mode")):
    raise SystemExit("bounded BA source must be a pure-native outcome residual")
for key in ("native_anchor_aux_weight", "mv_weight", "local_weight", "dustbin_weight", "geometry_weight", "pose_weight"):
    if abs(float(config.get(key, 0.0))) > 1e-12:
        raise SystemExit(f"source residual unexpectedly enables {key}")
if abs(float(config.get("retrieval_weight", 0.0)) - 1.0) > 1e-12:
    raise SystemExit("source residual does not retain native retrieval")
if abs(float(config.get("trust_weight", 0.0)) - 0.02) > 1e-12:
    raise SystemExit("source residual does not retain trust weight 0.02")
raw_anchor = torch.as_tensor(state.get("raw_anchor_offset"), dtype=torch.float32)
if raw_anchor.numel() and float(raw_anchor.abs().max().item()) > 1e-12:
    raise SystemExit("bounded BA must begin from a descriptor-only zero-anchor residual")
with open(ids_path, "rb") as handle:
    ids = torch.as_tensor(pickle.load(handle), dtype=torch.long).reshape(-1)
state_ids = torch.as_tensor(state.get("landmark_indices"), dtype=torch.long).reshape(-1)
if ids.numel() != int(expected_count) or not torch.equal(ids, state_ids):
    raise SystemExit("source residual IDs are not exactly aligned with the fixed bank")
print("Verified accepted robust residual source for bounded BA")
PY
}

write_manifest() {
  local state
  state="$(selected_state)"
  "$PYTHON" - "$MANIFEST" "$state" "$SOURCE_SELECTION" "$SOURCE_MANIFEST" "$BOOTSTRAP_DIR" "$BOOTSTRAP_IDS" "$BOOTSTRAP_META" <<PY
import json
import sys
from pathlib import Path

(
    output,
    selected_state,
    selection,
    source_manifest,
    bootstrap_dir,
    bootstrap_ids,
    bootstrap_meta,
) = map(Path, sys.argv[1:])
payload = {
    "schema_version": 3,
    "purpose": "validation_only_robust_kcs_gwff_bounded_surface_ba_and_refresh",
    "test_evaluation_forbidden": True,
    "scene": "${SCENE}",
    "source": {
        "robust_study_manifest": str(source_manifest.resolve()),
        "residual_selection": str(selection.resolve()),
        "selected_residual_state": str(selected_state.resolve()),
        "selection_was_validation_only": True,
    },
    "bootstrap": {
        "directory": str(bootstrap_dir.resolve()),
        "landmark_ids": str(bootstrap_ids.resolve()),
        "landmark_metadata": str(bootstrap_meta.resolve()),
        "resolution": "manifest_optimization_bootstrap_source_or_colocated_fallback",
    },
    "formal_deployment_protocol": {
        "longest_edge": 0,
        "native_keypoints": 2048,
        "sparse_frontend": "ulfloc_native",
        "topk": 1,
        "cosine_threshold": 0.0,
        "max_matches_per_landmark": 0,
    },
    "bounded_ba": {
        "descriptor_frozen": True,
        "association": "native_top1_gt_reprojection_clean_without_candidate_injection",
        "tangent_bound_m": float("${TANGENT_BOUND_M}"),
        "normal_bound_m": float("${NORMAL_BOUND_M}"),
        "minimum_distinct_support_views": 3,
        "depth_abs_tolerance_m": float("${DEPTH_ABS_TOLERANCE}"),
        "depth_rel_tolerance": float("${DEPTH_REL_TOLERANCE}"),
    },
    "refresh": {
        "enabled_only_after_ba_safety_acceptance": True,
        "objective": "pure_native_keep_swap_miss_reject",
        "steps": int("${REFRESH_STEPS}"),
    },
    "terminal_ba_refresh": {
        "enabled": True,
        "protocol": "predeclared_terminal_ba_coupling",
        "predeclared_ba_checkpoint_step": int("${BA_STEPS}"),
        "initialization": "always_use_terminal_ba_checkpoint_without_intermediate_ba_promotion",
        "objective": "pure_native_keep_swap_miss_reject",
        "geometry_frozen_during_refresh": True,
        "refresh_steps": int("${REFRESH_STEPS}"),
        "final_selection_control": "original_selected_residual",
        "final_selection": "validation_only_direct_five_metric_safety_gate",
        "test_metrics_used": False,
    },
    "runtime": {"camera_loader_workers": int("${CAMERA_LOADER_WORKERS}")},
}
output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY
}

base_map_args() {
  printf '%s\0' \
    "$PYTHON" train_lafgs_map.py \
    --model_path "$MODEL_ROOT" --source_path "${CAMBRIDGE_DATA_ROOT:-/mnt/pool/sqy/Cambridge_stdloc}/$SCENE" \
    --images processed --data_device cpu --gaussian_type 2dgs --feature_type sp \
    --resolution 1 --longest_edge 0 --norm_before_render --load_iteration 30000 \
    --query_feature_contract native_resized_input \
    --query_cache_path "$QUERY_CACHE" --query_cache_policy readonly \
    --visibility_cache_path "$VISIBILITY_CACHE" --visibility_mode rasterizer \
    --objective hard --native_keypoint_count 2048 \
    --native_association_radius_px 2 --native_sampling_mode detector_grid \
    --generic_proposal_count 0 --generic_proposal_weight 0 --distill_budget 0 \
    --validation_ratio "$VALIDATION_RATIO" --split_mode "$SPLIT_MODE" --split_seed "$SPLIT_SEED" \
    --train_seed 2026 --max_observations 512 --validation_observations 512
}

append_base_map_args() {
  local -n target="$1"
  local item
  while IFS= read -r -d '' item; do
    target+=("$item")
  done < <(base_map_args)
}

native_outcome_args() {
  printf '%s\0' \
    --observation_source native --native_anchor_aux_weight 0 \
    --native_outcome_mode --native_nce_weight 0 \
    --native_keep_weight 1 --native_keep_margin 0.05 \
    --native_swap_weight 1 --native_swap_margin 0.05 \
    --native_miss_weight 1 --native_miss_margin 0.05 \
    --native_reject_weight 0.05 --native_reject_threshold 0 \
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
  local state="$2"
  local cfg="$CONFIG_ROOT/${label}.yaml"
  "$PYTHON" scripts/make_stdloc_eval_cfg.py \
    --base_cfg configs/stdloc_cambridge.yaml --output "$cfg" \
    --artifact_model_path "$MODEL_ROOT" \
    --detector_folder ulfloc_native_no_detector --detector_iters 0 \
    --landmark_path "$BOOTSTRAP_IDS" --landmark_meta_path "$BOOTSTRAP_META" \
    --landmark_feature_override_path "$state" --override_landmark_features \
    --detect_num 2048 --nms 2 \
    --sparse_query_feature_contract native_resized_input --sparse_frontend ulfloc_native \
    --reprojection_error 12 --match_threshold 0 --match_topk 1 \
    --max_matches_per_landmark 0 --candidate_frontend_match_policy error --diagnostics \
    --diagnostics_grid_rows 4 --diagnostics_grid_cols 4 --diagnostics_voxel_size 1.0 \
    --diagnostics_task_translation_scale_m 0.07160573943725686 \
    --diagnostics_task_rotation_scale_degrees 2.0 \
    > "$LOG_ROOT/${label}_config.json"
  printf '%s\n' "$cfg"
}

run_eval() {
  local label="$1"
  local state="$2"
  local ref="$RESULT_ROOT/${label}.results_path"
  if [[ -f "$ref" && -f "$(<"$ref")/results_summary.json" ]]; then
    echo "[robust BA] Reusing candidate validation: $label"
    return
  fi
  local cfg
  cfg="$(make_eval_config "$label" "$state")"
  run_logged "${label}_validation" \
    "$PYTHON" stdloc.py \
    --model_path "$MODEL_ROOT" --source_path "${CAMBRIDGE_DATA_ROOT:-/mnt/pool/sqy/Cambridge_stdloc}/$SCENE" \
    --images processed --data_device cpu --gaussian_type 2dgs --feature_type sp \
    --resolution 1 --longest_edge 0 --norm_before_render --iteration 30000 \
    --cfg "$cfg" --prefix "lafgs-v2-robust-ba-${SCENE}-${label}-validation" --sparse_only \
    --evaluation_camera_subset candidate_validation --candidate_direct_validation_holdout \
    --candidate_validation_ratio "$VALIDATION_RATIO" \
    --candidate_split_mode "$SPLIT_MODE" --candidate_split_seed "$SPLIT_SEED"
  local output_path
  output_path="$(sed -n 's/^Output path: //p' "$LOG_ROOT/${label}_validation.log" | tail -n 1)"
  if [[ -z "$output_path" || ! -f "$output_path/results_summary.json" ]]; then
    echo "Candidate validation did not create results_summary.json for $label" >&2
    exit 1
  fi
  printf '%s\n' "$output_path" > "$ref"
}

result_summary() {
  local label="$1"
  local ref="$RESULT_ROOT/${label}.results_path"
  require_file "$ref"
  local directory
  directory="$(<"$ref")"
  require_file "$directory/results_summary.json"
  printf '%s\n' "$directory/results_summary.json"
}

verify_ba_state() {
  local state="$1"
  local initial_state="$2"
  require_file "$state"
  "$PYTHON" - "$state" "$initial_state" "$BOOTSTRAP_IDS" "$SOURCE_PLY" "$TANGENT_BOUND_M" "$NORMAL_BOUND_M" <<'PY'
import math
import pickle
import sys

import torch
from localization_training.surface_anchor import (
    bounded_surface_local_offsets,
    materialize_bounded_surface_anchors,
)
from plyfile import PlyData

state_path, initial_path, ids_path, ply_path, tangent, normal = sys.argv[1:]
tangent = float(tangent)
normal = float(normal)
state = torch.load(state_path, map_location="cpu")
initial = torch.load(initial_path, map_location="cpu")
config = state.get("config", {})
if str(config.get("observation_source")) != "native":
    raise SystemExit("BA must use native observations")
if bool(config.get("native_outcome_mode")):
    raise SystemExit("BA must freeze the descriptor outcome objective")
if float(config.get("geometry_weight", 0.0)) <= 0.0:
    raise SystemExit("BA checkpoint has no active geometry loss")
if str(config.get("geometry_mode")) != "native_association":
    raise SystemExit("BA must use native_association geometry")
if int(config.get("descriptor_end_step", 0)) != -1 or float(config.get("feature_lr", 1.0)) != 0.0:
    raise SystemExit("BA descriptor is not frozen")
if not math.isclose(float(config.get("tangent_bound_m")), tangent, abs_tol=1e-12):
    raise SystemExit("BA tangent bound mismatch")
if not math.isclose(float(config.get("normal_bound_m")), normal, abs_tol=1e-12):
    raise SystemExit("BA normal bound mismatch")
with open(ids_path, "rb") as handle:
    ids = torch.as_tensor(pickle.load(handle), dtype=torch.long).reshape(-1)
state_ids = torch.as_tensor(state.get("landmark_indices"), dtype=torch.long).reshape(-1)
if not torch.equal(ids, state_ids):
    raise SystemExit("BA landmark IDs changed")
if not torch.allclose(
    torch.as_tensor(state.get("landmark_features"), dtype=torch.float32),
    torch.as_tensor(initial.get("landmark_features"), dtype=torch.float32),
    atol=1e-6,
    rtol=1e-6,
):
    raise SystemExit("BA unexpectedly changed descriptors")
vertex = PlyData.read(ply_path).elements[0].data
base_xyz = torch.stack([
    torch.from_numpy(vertex["x"].copy()),
    torch.from_numpy(vertex["y"].copy()),
    torch.from_numpy(vertex["z"].copy()),
], dim=1).float()[ids]
base_rotation = torch.stack([
    torch.from_numpy(vertex["rot_0"].copy()),
    torch.from_numpy(vertex["rot_1"].copy()),
    torch.from_numpy(vertex["rot_2"].copy()),
    torch.from_numpy(vertex["rot_3"].copy()),
], dim=1).float()[ids]
landmark_xyz = torch.as_tensor(state.get("landmark_xyz"), dtype=torch.float32)
if landmark_xyz.shape != base_xyz.shape or not torch.isfinite(landmark_xyz).all():
    raise SystemExit("BA state has invalid landmark xyz")
raw_offset = torch.as_tensor(state.get("raw_anchor_offset"), dtype=torch.float32)
if raw_offset.shape != base_xyz.shape or not torch.isfinite(raw_offset).all():
    raise SystemExit("BA state has invalid raw surface-anchor offsets")
tangent_offset, normal_offset = bounded_surface_local_offsets(
    raw_offset,
    tangent_bound_m=tangent,
    normal_bound_m=normal,
)
maximum_tangent = float(torch.linalg.norm(tangent_offset, dim=1).max().item())
maximum_normal = float(normal_offset.abs().max().item())
if maximum_tangent > tangent + 1e-7:
    raise SystemExit(
        f"BA tangent displacement {maximum_tangent:g} exceeds bound {tangent:g}"
    )
if maximum_normal > normal + 1e-7:
    raise SystemExit(
        f"BA normal displacement {maximum_normal:g} exceeds bound {normal:g}"
    )
reconstructed_xyz = materialize_bounded_surface_anchors(
    base_xyz,
    base_rotation,
    raw_offset,
    tangent_bound_m=tangent,
    normal_bound_m=normal,
)
if not torch.allclose(landmark_xyz, reconstructed_xyz, atol=2e-6, rtol=1e-6):
    raise SystemExit("BA landmark xyz does not match its bounded surface-anchor state")
maximum = float(torch.linalg.norm(landmark_xyz - base_xyz, dim=1).max().item())
bound = math.sqrt(tangent * tangent + normal * normal) + 1e-6
if maximum > bound:
    raise SystemExit(f"BA displacement {maximum:g} exceeds bounded surface limit {bound:g}")
print(
    "Verified bounded BA state: "
    f"max_xyz_displacement_m={maximum:.8g}, "
    f"max_tangent_displacement_m={maximum_tangent:.8g}, "
    f"max_normal_displacement_m={maximum_normal:.8g}"
)
PY
}

verify_refresh_state() {
  local state="$1"
  local ba_state="$2"
  require_file "$state"
  "$PYTHON" - "$state" "$ba_state" "$BOOTSTRAP_IDS" "$TANGENT_BOUND_M" "$NORMAL_BOUND_M" <<'PY'
import math
import pickle
import sys

import torch

state_path, ba_path, ids_path, tangent, normal = sys.argv[1:]
state = torch.load(state_path, map_location="cpu")
ba = torch.load(ba_path, map_location="cpu")
config = state.get("config", {})
if str(config.get("observation_source")) != "native" or not bool(config.get("native_outcome_mode")):
    raise SystemExit("refresh must restore the pure-native outcome objective")
for key in ("native_anchor_aux_weight", "mv_weight", "local_weight", "dustbin_weight", "geometry_weight", "pose_weight"):
    if abs(float(config.get(key, 0.0))) > 1e-12:
        raise SystemExit(f"refresh unexpectedly enables {key}")
if abs(float(config.get("retrieval_weight", 0.0)) - 1.0) > 1e-12:
    raise SystemExit("refresh must retain native retrieval")
if abs(float(config.get("trust_weight", 0.0)) - 0.02) > 1e-12:
    raise SystemExit("refresh must retain trust weight")
if not math.isclose(float(config.get("tangent_bound_m")), float(tangent), abs_tol=1e-12):
    raise SystemExit("refresh tangent bound mismatch")
if not math.isclose(float(config.get("normal_bound_m")), float(normal), abs_tol=1e-12):
    raise SystemExit("refresh normal bound mismatch")
with open(ids_path, "rb") as handle:
    ids = torch.as_tensor(pickle.load(handle), dtype=torch.long).reshape(-1)
if not torch.equal(ids, torch.as_tensor(state.get("landmark_indices"), dtype=torch.long).reshape(-1)):
    raise SystemExit("refresh landmark IDs changed")
if not torch.allclose(
    torch.as_tensor(state.get("landmark_xyz"), dtype=torch.float32),
    torch.as_tensor(ba.get("landmark_xyz"), dtype=torch.float32),
    atol=1e-7,
    rtol=1e-7,
):
    raise SystemExit("descriptor refresh unexpectedly moved the bounded geometry")
print("Verified pure-native descriptor refresh with frozen bounded geometry")
PY
}

select_stage() {
  local output="$1"
  local control_label="$2"
  local control_state="$3"
  shift 3
  if [[ -f "$output" ]]; then
    echo "[robust BA] Reusing selection: $output"
    return
  fi
  local command=(
    "$PYTHON" scripts/select_lafgs_map_checkpoint.py
    --control_results "$(result_summary "$control_label")"
    --control_state "$control_state" --control_tag "$control_label" --selection_mode safety
    --min_te_gain_cm 0.02 --metric_tolerance 1e-9 --output "$output"
  )
  while [[ $# -gt 1 ]]; do
    local label="$1"
    local state="$2"
    shift 2
    if [[ -f "$state" && -f "$RESULT_ROOT/${label}.results_path" ]]; then
      command+=(--candidate "$label" "$(result_summary "$label")" "$state")
    fi
  done
  run_logged "$(basename "$output" .json)" "${command[@]}"
  require_file "$output"
}

selection_state() {
  local selection="$1"
  "$PYTHON" - "$selection" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], "r", encoding="utf-8"))
if payload.get("selection_protocol", {}).get("test_metrics_used") is not False:
    raise SystemExit("stage selection used test metrics")
print(payload["selected_state"])
PY
}

selection_label() {
  local selection="$1"
  "$PYTHON" - "$selection" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], "r", encoding="utf-8"))
print(payload["selected_tag"])
PY
}

selection_uses_control() {
  local selection="$1"
  "$PYTHON" - "$selection" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], "r", encoding="utf-8"))
print("1" if payload.get("used_control_fallback", payload.get("used_strong_fallback", False)) else "0")
PY
}

ba() {
  verify_selected_residual
  write_manifest
  local initial_state
  initial_state="$(selected_state)"
  if [[ -f "$BA_DIR/${BA_STEPS}_lafgs_map_state.pt" ]]; then
    verify_ba_state "$BA_DIR/${BA_STEPS}_lafgs_map_state.pt" "$initial_state"
    echo "[robust BA] Reusing bounded BA: $BA_DIR"
    return
  fi
  local command=()
  append_base_map_args command
  command+=(
    --output_dir "$BA_DIR" --scaffold_mode file --landmark_path "$BOOTSTRAP_IDS"
    --initial_state_path "$initial_state" --initial_state_blend 1 --initial_state_alignment exact
    --initialization_mode ulf_robust_geometry
    --observation_source native --native_anchor_aux_weight 0 --no-native_outcome_mode
    --descriptor_end_step -1 --steps "$BA_STEPS" --save_steps 500 1000 "$BA_STEPS"
    --feature_lr 0 --geometry_lr 0.003 --geometry_start_step 0 --geometry_weight 1
    --geometry_mode native_association --geometry_association_max_reprojection_px 2
    --geometry_association_min_margin 0.02
    --geometry_association_depth_abs_tolerance "$DEPTH_ABS_TOLERANCE"
    --geometry_association_depth_rel_tolerance "$DEPTH_REL_TOLERANCE"
    --geometry_association_min_support_views 3 --geometry_association_support_observations 2048
    --tangent_bound_m "$TANGENT_BOUND_M" --normal_bound_m "$NORMAL_BOUND_M"
    --surface_weight 0.05 --depth_weight 0.25 --reprojection_weight 1
    --mv_weight 0 --retrieval_weight 0 --trust_weight 0 --local_weight 0 --dustbin_weight 0
    --pose_weight 0 --pose_gradient_mode off --log_interval 100
  )
  run_logged bounded_ba "${command[@]}"
  verify_ba_state "$BA_DIR/${BA_STEPS}_lafgs_map_state.pt" "$initial_state"
}

ba_validate() {
  ba
  local initial_state initial_label
  initial_state="$(selected_state)"
  initial_label="$(selected_label)"
  run_eval "residual_selected_${initial_label}" "$initial_state"
  local step
  for step in 500 1000 "$BA_STEPS"; do
    local state="$BA_DIR/${step}_lafgs_map_state.pt"
    if [[ -f "$state" ]]; then
      run_eval "ba_${step}" "$state"
    fi
  done
}

select_ba() {
  ba_validate
  local initial_state initial_label
  initial_state="$(selected_state)"
  initial_label="$(selected_label)"
  select_stage "$BA_SELECTION" "residual_selected_${initial_label}" "$initial_state" \
    ba_500 "$BA_DIR/500_lafgs_map_state.pt" \
    ba_1000 "$BA_DIR/1000_lafgs_map_state.pt" \
    "ba_${BA_STEPS}" "$BA_DIR/${BA_STEPS}_lafgs_map_state.pt"
}

refresh() {
  select_ba
  if [[ "$(selection_uses_control "$BA_SELECTION")" == "1" ]]; then
    echo "[robust BA] Skipping refresh because BA did not beat the selected residual"
    return
  fi
  local initial_state
  initial_state="$(selection_state "$BA_SELECTION")"
  if [[ -f "$REFRESH_DIR/${REFRESH_STEPS}_lafgs_map_state.pt" ]]; then
    verify_refresh_state "$REFRESH_DIR/${REFRESH_STEPS}_lafgs_map_state.pt" "$initial_state"
    echo "[robust BA] Reusing descriptor refresh: $REFRESH_DIR"
    return
  fi
  local command=()
  append_base_map_args command
  append_native_outcome_args command
  command+=(
    --output_dir "$REFRESH_DIR" --scaffold_mode file --landmark_path "$BOOTSTRAP_IDS"
    --initial_state_path "$initial_state" --initial_state_blend 1 --initial_state_alignment exact
    --initialization_mode ulf_robust_geometry
    --steps "$REFRESH_STEPS" --save_steps 500 "$REFRESH_STEPS"
    --feature_lr 2.5e-5 --weight_decay 1e-4 --hypothesis_topk 32
    --tangent_bound_m "$TANGENT_BOUND_M" --normal_bound_m "$NORMAL_BOUND_M"
    --positive_radius_px 2 --negative_radius_px 6 --log_interval 100
  )
  run_logged refresh "${command[@]}"
  verify_refresh_state "$REFRESH_DIR/${REFRESH_STEPS}_lafgs_map_state.pt" "$initial_state"
}

terminal_refresh() {
  # The terminal BA step is fixed before any refresh metric is examined.  This
  # is a coupled BA->descriptor experiment, not a fallback that promotes an
  # individually unsafe BA checkpoint.
  ba
  write_manifest
  local residual_state terminal_ba_state
  residual_state="$(selected_state)"
  terminal_ba_state="$BA_DIR/${BA_STEPS}_lafgs_map_state.pt"
  verify_ba_state "$terminal_ba_state" "$residual_state"
  if [[ -f "$TERMINAL_REFRESH_DIR/${REFRESH_STEPS}_lafgs_map_state.pt" ]]; then
    verify_refresh_state "$TERMINAL_REFRESH_DIR/${REFRESH_STEPS}_lafgs_map_state.pt" "$terminal_ba_state"
    echo "[robust BA] Reusing predeclared terminal-BA descriptor refresh: $TERMINAL_REFRESH_DIR"
    return
  fi
  local command=()
  append_base_map_args command
  append_native_outcome_args command
  command+=(
    --output_dir "$TERMINAL_REFRESH_DIR" --scaffold_mode file --landmark_path "$BOOTSTRAP_IDS"
    --initial_state_path "$terminal_ba_state" --initial_state_blend 1 --initial_state_alignment exact
    --initialization_mode ulf_robust_geometry
    --steps "$REFRESH_STEPS" --save_steps 500 "$REFRESH_STEPS"
    --feature_lr 2.5e-5 --weight_decay 1e-4 --hypothesis_topk 32
    --geometry_lr 0 --geometry_start_step -1
    --tangent_bound_m "$TANGENT_BOUND_M" --normal_bound_m "$NORMAL_BOUND_M"
    --positive_radius_px 2 --negative_radius_px 6 --log_interval 100
  )
  run_logged terminal_ba_refresh "${command[@]}"
  verify_refresh_state "$TERMINAL_REFRESH_DIR/${REFRESH_STEPS}_lafgs_map_state.pt" "$terminal_ba_state"
}

refresh_validate() {
  refresh
  if [[ "$(selection_uses_control "$BA_SELECTION")" == "1" ]]; then
    return
  fi
  local ba_state ba_label
  ba_state="$(selection_state "$BA_SELECTION")"
  ba_label="$(selection_label "$BA_SELECTION")"
  run_eval "ba_selected_${ba_label}" "$ba_state"
  local step
  for step in 500 "$REFRESH_STEPS"; do
    local state="$REFRESH_DIR/${step}_lafgs_map_state.pt"
    if [[ -f "$state" ]]; then
      run_eval "refresh_${step}" "$state"
    fi
  done
}

terminal_refresh_validate() {
  terminal_refresh
  local residual_state residual_label terminal_ba_state
  residual_state="$(selected_state)"
  residual_label="$(selected_label)"
  terminal_ba_state="$BA_DIR/${BA_STEPS}_lafgs_map_state.pt"
  run_eval "residual_selected_${residual_label}" "$residual_state"
  # This is diagnostic-only; the final coupled selector deliberately compares
  # terminal refresh checkpoints to the original residual, not to BA.
  run_eval "terminal_ba_${BA_STEPS}" "$terminal_ba_state"
  local step
  for step in 500 "$REFRESH_STEPS"; do
    local state="$TERMINAL_REFRESH_DIR/${step}_lafgs_map_state.pt"
    if [[ -f "$state" ]]; then
      run_eval "terminal_refresh_${step}" "$state"
    fi
  done
}

select_refresh() {
  refresh_validate
  if [[ "$(selection_uses_control "$BA_SELECTION")" == "1" ]]; then
    echo "[robust BA] Final descriptor state remains the BA selection control"
    return
  fi
  local ba_state ba_label
  ba_state="$(selection_state "$BA_SELECTION")"
  ba_label="$(selection_label "$BA_SELECTION")"
  select_stage "$REFRESH_SELECTION" "ba_selected_${ba_label}" "$ba_state" \
    refresh_500 "$REFRESH_DIR/500_lafgs_map_state.pt" \
    "refresh_${REFRESH_STEPS}" "$REFRESH_DIR/${REFRESH_STEPS}_lafgs_map_state.pt"
}

select_terminal_refresh() {
  terminal_refresh_validate
  local residual_state residual_label
  residual_state="$(selected_state)"
  residual_label="$(selected_label)"
  select_stage "$TERMINAL_REFRESH_SELECTION" "residual_selected_${residual_label}" "$residual_state" \
    terminal_refresh_500 "$TERMINAL_REFRESH_DIR/500_lafgs_map_state.pt" \
    "terminal_refresh_${REFRESH_STEPS}" "$TERMINAL_REFRESH_DIR/${REFRESH_STEPS}_lafgs_map_state.pt"
}

case "$MODE" in
  ba) ba ;;
  ba_validate) ba_validate ;;
  select_ba) select_ba ;;
  refresh) refresh ;;
  refresh_validate) refresh_validate ;;
  select_refresh) select_refresh ;;
  terminal_refresh) terminal_refresh ;;
  terminal_refresh_validate) terminal_refresh_validate ;;
  select_terminal_refresh) select_terminal_refresh ;;
  terminal_all) select_terminal_refresh ;;
  all) select_refresh ;;
esac
