#!/usr/bin/env bash
set -euo pipefail

# Formal sparse LaFGS-V2 entry point.  Keep this separate from the robust
# initializer ablation runner so a default invocation cannot silently fall
# back to an ablation budget or a resized/capped frontend.

if [[ $# -ne 3 ]]; then
  echo "Usage: $0 <scene> <gpu> <bootstrap|validate|residual|residual_validate|select_residual|all>" >&2
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
  bootstrap|validate|residual|residual_validate|select_residual|all) ;;
  *) echo "Unsupported mode: $MODE" >&2; exit 2 ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Do not make semantic protocol fields caller-tunable here.  Storage roots and
# input roots remain configurable through the downstream runner's documented
# environment variables.
export LAFGS_ROBUST_LANDMARK_BUDGET=32000
export LAFGS_ROBUST_VALIDATION_RATIO=0.2
export LAFGS_ROBUST_SPLIT_MODE=stratified_temporal_block
export LAFGS_ROBUST_SPLIT_SEED=2026
export LAFGS_ROBUST_TRAIN_SEED=2026
export LAFGS_ROBUST_SUPPORT_VIEWS=0
export LAFGS_ROBUST_SUPPORT_SAMPLING=uniform
export LAFGS_ROBUST_MIN_VISIBLE_VIEWS=4
export LAFGS_ROBUST_MIN_VOTES=2
export LAFGS_ROBUST_MIN_RATE=0.01
export LAFGS_ROBUST_VIEW_BINS=4
export LAFGS_ROBUST_MIN_VIEW_BINS=2
export LAFGS_ROBUST_TRAJECTORY_BINS=4
export LAFGS_ROBUST_MIN_TRAJECTORY_BINS=2
export LAFGS_ROBUST_TRIM_FRACTION=0.1
export LAFGS_ROBUST_DESCRIPTOR_MIN_COSINE=-1.0
export LAFGS_ROBUST_TRIM_HIST_BINS=64
export LAFGS_ROBUST_FUSION_REFERENCE_MODE=mean
export LAFGS_ROBUST_RESIDUAL_STEPS=5000
export LAFGS_ROBUST_CAMERA_LOADER_WORKERS=0
# This is the selected native residual profile: protect native GT-clean
# correspondences while penalizing train-only false-attractor landmarks.  It
# must be explicit because the general robust-initializer runner defaults to
# a no-attractor ablation profile.
export LAFGS_ROBUST_NATIVE_KEEP_WEIGHT=1.0
export LAFGS_ROBUST_NATIVE_KEEP_MARGIN=0.05
export LAFGS_ROBUST_NATIVE_KEEP_LOOSE_WEIGHT=0.0
export LAFGS_ROBUST_NATIVE_KEEP_LOOSE_RADIUS_PX=4.0
export LAFGS_ROBUST_NATIVE_KEEP_LOOSE_MARGIN=0.025
export LAFGS_ROBUST_NATIVE_SWAP_WEIGHT=1.0
export LAFGS_ROBUST_NATIVE_SWAP_MARGIN=0.05
export LAFGS_ROBUST_NATIVE_MISS_WEIGHT=1.0
export LAFGS_ROBUST_NATIVE_MISS_MARGIN=0.05
export LAFGS_ROBUST_NATIVE_REJECT_WEIGHT=0.05
export LAFGS_ROBUST_NATIVE_REJECT_THRESHOLD=0.0
export LAFGS_ROBUST_NATIVE_GLOBAL_ATTRACTOR_WEIGHT=0.25
export LAFGS_ROBUST_NATIVE_GLOBAL_ATTRACTOR_MIN_INCOMING=4
export LAFGS_ROBUST_NATIVE_GLOBAL_ATTRACTOR_SUPPORT_POWER=0.5
export LAFGS_ROBUST_NATIVE_GLOBAL_ATTRACTOR_MAX_SCORE=4.0

printf '%s\n' '[LaFGS-V2 canonical sparse mainline]'
printf '%s\n' 'full-resolution native frontend; cosine top-1; no landmark match cap'
printf '%s\n' 'strict KCS 32K; RGB-only support KCS; 10% mean-reference GWFF trim'
printf '%s\n' 'false-attractor-aware pure-native 5K residual; validation-only checkpoint selection before test'

exec bash "$SCRIPT_DIR/run_lafgs_v2_robust_initializer_ablation.sh" \
  "$SCENE" "$GPU" support_rgb_only "$MODE"
