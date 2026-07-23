#!/usr/bin/env bash
set -euo pipefail

# Reproducible validation-only U0--U4 bootstrap matrix from the LaFGS V2
# protocol note.  This wrapper fixes the capacity/support-view factors and
# delegates all map construction and candidate evaluation to the frozen
# full-resolution native runner.  It intentionally never evaluates test.

if [[ $# -ne 3 ]]; then
  echo "Usage: $0 <scene> <gpu> <u0|u1|u2|u3|u4|all>" >&2
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
  u0|u1|u2|u3|u4|all) ;;
  *) echo "Unsupported bootstrap-matrix row: $MODE" >&2; exit 2 ;;
esac

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/root/miniconda3/envs/cybersim_agent/bin/python}"
EXPERIMENT_ROOT="${LAFGS_V2_ULFPARITY_ROOT:-/mnt/pool/sqy/stdloc_lafgs_v2_ulfparity_alternating_20260721}"
MATRIX_ROOT="${LAFGS_V2_BOOTSTRAP_MATRIX_ROOT:-$EXPERIMENT_ROOT/bootstrap_matrix_20260722}"
SPLIT_MODE="${LAFGS_BOOTSTRAP_MATRIX_SPLIT_MODE:-stratified_temporal_block}"
SPLIT_SEED="${LAFGS_BOOTSTRAP_MATRIX_SPLIT_SEED:-2026}"
CAMERA_LOADER_WORKERS="${LAFGS_BOOTSTRAP_MATRIX_CAMERA_LOADER_WORKERS:-0}"
DEFAULT_QUERY_CACHE="/dev/shm/lafgs_v2_query_cache/${SCENE}_native_fullres_k2048.pt"
# OldHospital already has a compatible 28GB native cache on the pool.  Reuse
# it by default so a concurrent KingsCollege matrix cannot exhaust /dev/shm.
if [[ "$SCENE" == "OldHospital" ]]; then
  DEFAULT_QUERY_CACHE="$EXPERIMENT_ROOT/OldHospital/ulfparity_native20k_s128_k2048_v2/query_cache_native_fullres_k2048.pt"
fi
QUERY_CACHE="${LAFGS_BOOTSTRAP_MATRIX_QUERY_CACHE_PATH:-$DEFAULT_QUERY_CACHE}"

if ! [[ "$CAMERA_LOADER_WORKERS" =~ ^[0-9]+$ ]]; then
  echo "LAFGS_BOOTSTRAP_MATRIX_CAMERA_LOADER_WORKERS must be a non-negative integer" >&2
  exit 2
fi
case "$SPLIT_MODE" in
  stratified_temporal_block) ;;
  *)
    echo "The formal U0--U4 matrix requires stratified_temporal_block, got $SPLIT_MODE" >&2
    exit 2
    ;;
esac

cd "$REPO_ROOT"
mkdir -p "$MATRIX_ROOT/$SCENE"

run_row() {
  local row="$1"
  local budget
  local support_views
  local support_sampling
  local tag
  local run_root
  local runner_status=0
  case "$row" in
    u0) budget=16000; support_views=128; support_sampling=uniform ;;
    u1) budget=20000; support_views=128; support_sampling=uniform ;;
    u2) budget=20000; support_views=256; support_sampling=pose_diverse ;;
    u3) budget=20000; support_views=0; support_sampling=uniform ;;
    u4) budget=32000; support_views=0; support_sampling=uniform ;;
    *) echo "Internal unknown bootstrap-matrix row: $row" >&2; exit 2 ;;
  esac

  tag="ulfparity_native${budget}_s${support_views}_k2048_ulfloc_parity_tau0_cap0_v8_fullres_native_uncapped_pure_native"
  if [[ "$support_sampling" != "uniform" ]]; then
    tag="ulfparity_native${budget}_s${support_views}_${support_sampling}_k2048_ulfloc_parity_tau0_cap0_v8_fullres_native_uncapped_pure_native"
  fi
  if [[ "$SPLIT_MODE" != "temporal_block" ]]; then
    tag="${tag}_split${SPLIT_MODE}"
  fi
  run_root="$EXPERIMENT_ROOT/$SCENE/$tag"

  # Clear caller-supplied capacity and bootstrap state so an entry always has
  # the exact factor definition recorded above.
  if env -u LAFGS_ULF_BOOTSTRAP_SOURCE_DIR \
    -u LAFGS_ULF_LANDMARK_BUDGET \
    -u LAFGS_ULF_SUPPORT_VIEWS \
    -u LAFGS_ULF_SUPPORT_VIEW_SAMPLING \
    -u LAFGS_ULF_PARITY_KCS_MASK_POLICY \
    -u LAFGS_ULF_SPLIT_MODE \
    -u LAFGS_ULF_SPLIT_SEED \
    -u LAFGS_ULF_TRAIN_SEED \
    -u LAFGS_ULF_CAMERA_LOADER_WORKERS \
    -u LAFGS_ULF_EVAL_PROFILE \
    -u LAFGS_ULF_NATIVE_MATCH_THRESHOLD \
    -u LAFGS_ULF_MAX_MATCHES_PER_LANDMARK \
    -u LAFGS_ULF_RESIDUAL_STEPS \
    -u LAFGS_ULF_SELECTION_MODE \
    LAFGS_ULF_LANDMARK_BUDGET="$budget" \
    LAFGS_ULF_SUPPORT_VIEWS="$support_views" \
    LAFGS_ULF_SUPPORT_VIEW_SAMPLING="$support_sampling" \
    LAFGS_ULF_PARITY_KCS_MASK_POLICY=rgb_only \
    LAFGS_ULF_SPLIT_MODE="$SPLIT_MODE" \
    LAFGS_ULF_SPLIT_SEED="$SPLIT_SEED" \
    LAFGS_ULF_TRAIN_SEED=2026 \
    LAFGS_ULF_CAMERA_LOADER_WORKERS="$CAMERA_LOADER_WORKERS" \
    LAFGS_ULF_QUERY_CACHE_PATH="$QUERY_CACHE" \
    bash scripts/run_lafgs_v2_ulfparity_alternating.sh "$SCENE" "$GPU" bootstrap_validate
  then
    :
  else
    runner_status=$?
  fi

  "$PYTHON" - "$MATRIX_ROOT/$SCENE/${row}.json" "$row" "$budget" \
    "$support_views" "$support_sampling" "$SPLIT_MODE" "$SPLIT_SEED" \
    "$QUERY_CACHE" "$run_root" "$runner_status" <<'PY'
import json
import sys
from pathlib import Path

path, row, budget, views, sampling, split, seed, cache, run_root, status = sys.argv[1:]
payload = {
    "schema_version": 1,
    "purpose": "validation_only_lafgs_v2_bootstrap_matrix",
    "row": row.upper(),
    "test_evaluation_forbidden": True,
    "formal_protocol": "lafgs_v2_fullres_native_uncapped_v1",
    "landmark_budget": int(budget),
    "support_views": int(views),
    "support_view_sampling": sampling,
    "split_mode": split,
    "split_seed": int(seed),
    "query_cache": cache,
    "run_root": run_root,
    "bootstrap_runner_exit_status": int(status),
    "bootstrap_gate_passed": int(status) == 0,
}
Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY
  if [[ "$runner_status" -ne 0 ]]; then
    echo "Bootstrap-matrix row ${row} did not pass the bootstrap gate; validation artifact was retained." >&2
    return "$runner_status"
  fi
}

if [[ "$MODE" == "all" ]]; then
  failures=0
  for row in u0 u1 u2 u3 u4; do
    if ! run_row "$row"; then
      failures=1
    fi
  done
  exit "$failures"
else
  run_row "$MODE"
fi
