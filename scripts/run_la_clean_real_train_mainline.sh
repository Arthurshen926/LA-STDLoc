#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

# Clean LA-STDLoc mainline:
#   all real Cambridge train RGB pseudo-queries
#   full STDLoc teacher cache for sparse/dense diagnostics
#   scratch LA student training
#
# Synthetic RGB, teacher gates, sample selectors, no-reference masks, soft
# reliability weights, and direct depth checks are intentionally disabled here.
# Use scripts/run_la_pseudo_query_pipeline.sh directly for those ablations.

export LA_ENABLE_SYNTHETIC=0
export SYNTHETIC_COUNT=0
export SYNTHETIC_CANDIDATE_MULTIPLIER=1
export TEACHER_CACHE_SOURCES=train_rgb
export TEACHER_CACHE_SPARSE_VALID_MASK=0
export RUN_PSEUDO_QUERY_GATE=0
export RUN_PSEUDO_QUERY_SELECT=0
export PSEUDO_QUERY_SOURCES=train_rgb
export PSEUDO_QUERY_MAX_SYNTHETIC=0
export PSEUDO_QUERY_SELECT_MAX_SYNTHETIC=0
export PSEUDO_QUERY_RELIABILITY_MODE=none
export PSEUDO_QUERY_RELIABILITY_LOSS_MODE=none
export PSEUDO_QUERY_STAGE_OBJECTIVE_MODE=none
export PSEUDO_QUERY_EXCLUDE_SPARSE_FAILURE_STAGES=0
export PSEUDO_QUERY_FILTER_TEACHER_CACHE=0
export PSEUDO_QUERY_REQUIRE_TEACHER_CACHE=1
export PSEUDO_QUERY_ENABLE_TEACHER_GATE=0
export PSEUDO_QUERY_NO_REFERENCE_REGION_WEIGHT=0
export LA_DIRECT_DEPTH_CHECK=0

export LA_TRAIN_MODE=scratch
export RUN_PSEUDO_QUERY_MANIFEST=${RUN_PSEUDO_QUERY_MANIFEST:-1}
export RUN_TEACHER_CACHE=${RUN_TEACHER_CACHE:-1}
export RUN_TRAIN=${RUN_TRAIN:-1}
export RUN_EVAL=${RUN_EVAL:-1}
export RUN_LA_FRONTEND_REFRESH=${RUN_LA_FRONTEND_REFRESH:-1}
export FORCE_LA_FRONTEND_REFRESH=${FORCE_LA_FRONTEND_REFRESH:-0}

export LA_ADAPT_STEPS=${LA_ADAPT_STEPS:-500}
export TRAIN_SEED=${TRAIN_SEED:-0}
export LA_BOOTSTRAP_LANDMARK_NUM=${LA_BOOTSTRAP_LANDMARK_NUM:-4096}
export LA_DETECTOR_LANDMARK_NUM=${LA_DETECTOR_LANDMARK_NUM:-4096}
export LA_DETECTOR_ITERS=${LA_DETECTOR_ITERS:-${LA_ADAPT_STEPS}}

exec "$SCRIPT_DIR/run_la_pseudo_query_pipeline.sh"
