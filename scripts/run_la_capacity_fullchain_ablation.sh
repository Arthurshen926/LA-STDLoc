#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

SCENE=${SCENE:-ShopFacade}
LANDMARK_NUM=${LANDMARK_NUM:-8192}
LA_ADAPT_STEPS=${LA_ADAPT_STEPS:-2000}
TRAIN_SEED=${TRAIN_SEED:-180}
GPU=${GPU:-0}
PSEUDO_QUERY_SOURCE_ROOT=${PSEUDO_QUERY_SOURCE_ROOT:-/mnt/pool/sqy/stdloc_la_mainline_refactor_2000_20260630}
OUT_ROOT=${OUT_ROOT:-/mnt/pool/sqy/stdloc_la_capacity_fullchain_${LA_ADAPT_STEPS}_${LANDMARK_NUM}_seed${TRAIN_SEED}}
FORCE_PSEUDO_QUERY_COPY=${FORCE_PSEUDO_QUERY_COPY:-1}

source_pseudo_dir="$PSEUDO_QUERY_SOURCE_ROOT/$SCENE/pseudo_query"
target_pseudo_dir="$OUT_ROOT/$SCENE/pseudo_query"

if [[ ! -f "$source_pseudo_dir/pseudo_queries.jsonl" ]]; then
  echo "Missing source pseudo-query manifest: $source_pseudo_dir/pseudo_queries.jsonl" >&2
  exit 1
fi
if [[ ! -f "$source_pseudo_dir/pseudo_teacher_cache.pt" ]]; then
  echo "Missing source pseudo teacher cache: $source_pseudo_dir/pseudo_teacher_cache.pt" >&2
  exit 1
fi

if [[ "$FORCE_PSEUDO_QUERY_COPY" == "1" || ! -f "$target_pseudo_dir/pseudo_queries.jsonl" || ! -f "$target_pseudo_dir/pseudo_teacher_cache.pt" ]]; then
  rm -rf "$target_pseudo_dir"
  mkdir -p "$target_pseudo_dir"
  cp -a "$source_pseudo_dir/." "$target_pseudo_dir/"
fi

export SCENES="$SCENE"
export OUT_ROOT
export GPU
export LA_ADAPT_STEPS
export TRAIN_SEED
export LA_BOOTSTRAP_LANDMARK_NUM="$LANDMARK_NUM"
export LA_DETECTOR_LANDMARK_NUM="$LANDMARK_NUM"
export LA_DETECTOR_ITERS=${LA_DETECTOR_ITERS:-$LA_ADAPT_STEPS}

export RUN_PSEUDO_QUERY_MANIFEST=0
export RUN_TEACHER_CACHE=0
export RUN_TEACHER_CACHE_AUDIT=1
export RUN_TRAIN=${RUN_TRAIN:-1}
export RUN_EVAL=${RUN_EVAL:-1}
export RUN_LA_FRONTEND_REFRESH=${RUN_LA_FRONTEND_REFRESH:-1}
export FORCE_LA_FRONTEND_REFRESH=${FORCE_LA_FRONTEND_REFRESH:-1}

echo "Running LA capacity full-chain ablation:"
echo "  scene=$SCENE"
echo "  landmark_num=$LANDMARK_NUM"
echo "  steps=$LA_ADAPT_STEPS"
echo "  seed=$TRAIN_SEED"
echo "  gpu=$GPU"
echo "  out_root=$OUT_ROOT"

exec "$SCRIPT_DIR/run_la_clean_real_train_mainline.sh"
