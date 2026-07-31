#!/usr/bin/env bash
set -euo pipefail

# Low-cost oracle over the frozen A2 correspondence graph.  This runner does
# not train or mutate a map and never enables the A3 hard selector.

if [[ $# -ne 3 ]]; then
  echo "Usage: bash $0 <scene> <gpu> <dump|score|all>" >&2
  exit 2
fi

SCENE="$1"
GPU="$2"
MODE="$3"
case "$SCENE" in
  GreatCourt|KingsCollege|ShopFacade|StMarysChurch) ;;
  *) echo "Unsupported frozen scene: $SCENE" >&2; exit 2 ;;
esac
case "$GPU" in
  0|1) ;;
  *) echo "Group consensus experiments are restricted to GPU 0 or 1" >&2; exit 2 ;;
esac
case "$MODE" in
  dump|score|all) ;;
  *) echo "Unsupported mode: $MODE" >&2; exit 2 ;;
esac

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/root/miniconda3/envs/cybersim_agent/bin/python}"
DATA_ROOT="${CAMBRIDGE_DATA_ROOT:-/mnt/pool/sqy/Cambridge_stdloc}"
FROZEN_ROOT="${LAFGS_V1_MULTISCENE_ROOT:-/mnt/pool/sqy/stdloc_lafgs_v1_frozen_multiscene_20260731}"
ORACLE_ROOT="${LAFGS_GROUP_CONSENSUS_ROOT:-/mnt/pool/sqy/stdloc_lafgs_v1_group_consensus_oracle_20260731}"
SCENE_ROOT="$FROZEN_ROOT/$SCENE"
ROOT="$ORACLE_ROOT/$SCENE"
MODEL_ROOT="$SCENE_ROOT/prior/rgb_matcha_2dgs"
SOURCE_ROOT="$DATA_ROOT/$SCENE"
PLY="$MODEL_ROOT/point_cloud/iteration_30000/point_cloud.ply"
A2_DIR="$SCENE_ROOT/evaluation/A2_family_all/seed2026"
A2_CFG="$A2_DIR/config.yaml"
OUTPUT="$ROOT/dump"
CFG="$OUTPUT/config.yaml"
QUERY_LIST="$ROOT/query_list.json"
SELECTION_REPORT="$ROOT/query_selection.json"
RESULT_POINTER="$OUTPUT/result.path"
REPORT_JSON="$ROOT/group_consensus_oracle.json"
REPORT_MD="$ROOT/group_consensus_oracle.md"

export CUDA_HOME=/usr/local/cuda-11.8
export PATH="/root/miniconda3/envs/cybersim_agent/bin:$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="/root/miniconda3/envs/cybersim_agent/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$REPO_ROOT"
export PYTHONHASHSEED=2026
export STDLOC_CAMERA_LOADER_WORKERS=0
mkdir -p "$ROOT" "$OUTPUT/results"
cd "$REPO_ROOT"

require_file() {
  [[ -f "$1" ]] || {
    echo "Required artifact is missing: $1" >&2
    exit 1
  }
}

select_queries() {
  if [[ -f "$QUERY_LIST" && -f "$SELECTION_REPORT" ]]; then
    return
  fi
  "$PYTHON" scripts/select_lafgs_group_consensus_queries.py \
    --scene-root "$SCENE_ROOT" \
    --output-list "$QUERY_LIST" \
    --output-report "$SELECTION_REPORT" \
    > "$ROOT/query_selection.log"
}

dump_queries() {
  require_file "$A2_CFG"
  require_file "$PLY"
  select_queries
  if [[ -f "$RESULT_POINTER" ]] && \
     [[ -f "$(<"$RESULT_POINTER")/discrete_oracle_dump/manifest.json" ]]; then
    return
  fi
  "$PYTHON" scripts/make_stdloc_eval_cfg.py \
    --base_cfg "$A2_CFG" --output "$CFG" \
    --artifact_model_path "$MODEL_ROOT" \
    --diagnostics_dump_discrete_oracle \
    --diagnostics_oracle_topk 16 \
    > "$OUTPUT/config_build.json"
  "$PYTHON" - "$A2_CFG" "$CFG" <<'PY'
import copy
import sys
import yaml

base = yaml.safe_load(open(sys.argv[1]))
current = yaml.safe_load(open(sys.argv[2]))
for payload in (base, current):
    diagnostics = payload["sparse"].setdefault("diagnostics", {})
    diagnostics.pop("dump_discrete_oracle", None)
    diagnostics.pop("oracle_topk", None)
if base != current:
    raise SystemExit("oracle config changed fields outside diagnostics")
PY
  (
    export CUDA_VISIBLE_DEVICES="$GPU"
    export STDLOC_RESULTS_ROOT="$OUTPUT/results"
    "$PYTHON" stdloc.py \
      --model_path "$MODEL_ROOT" --source_path "$SOURCE_ROOT" \
      --images processed --data_device cpu --gaussian_type 2dgs \
      --sh_degree 3 --feature_type sp --resolution 1 --longest_edge 0 \
      --norm_before_render --iteration 30000 --cfg "$CFG" \
      --prefix "lafgs-group-oracle-$SCENE" \
      --sparse_only --evaluation_camera_subset test \
      --evaluation_camera_list "$QUERY_LIST" \
      --evaluation_camera_list_test_only \
      2>&1 | tee "$OUTPUT/eval.log"
  )
  local result
  result="$(sed -n 's/^Output path: //p' "$OUTPUT/eval.log" | tail -n 1)"
  [[ -n "$result" && -f "$result/discrete_oracle_dump/manifest.json" ]] || {
    echo "Discrete oracle dump failed for $SCENE" >&2
    exit 1
  }
  printf '%s\n' "$result" > "$RESULT_POINTER"
}

score_queries() {
  dump_queries
  local result
  result="$(<"$RESULT_POINTER")"
  "$PYTHON" scripts/evaluate_lafgs_group_consensus_oracle.py \
    --scene "$SCENE" \
    --dump-dir "$result/discrete_oracle_dump" \
    --gaussian-ply "$PLY" \
    --selection-report "$SELECTION_REPORT" \
    --output-json "$REPORT_JSON" \
    --output-markdown "$REPORT_MD" \
    > "$ROOT/group_consensus_oracle.log"
}

case "$MODE" in
  dump) dump_queries ;;
  score) score_queries ;;
  all) score_queries ;;
esac
