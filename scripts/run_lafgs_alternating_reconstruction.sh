#!/usr/bin/env bash
set -euo pipefail

# Two-round inference-aligned LaFGS reconstruction.  Structure is updated by
# fixed-seed hard PnP replay; feature calibration only touches newly activated
# rows.  Geometry remains frozen unless a separately audited geometry M-step is
# explicitly inserted after the second structure gate.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/root/miniconda3/envs/cybersim_agent/bin/python}"
GPU="${LAFGS_ALTERNATING_GPU:-0}"
OUTPUT_ROOT="${LAFGS_ALTERNATING_OUTPUT_ROOT:?Set output root}"
BASE_MAP="${LAFGS_ALTERNATING_BASE_MAP:?Set immutable active base map}"
CANDIDATE_POOL="${LAFGS_ALTERNATING_CANDIDATE_POOL:?Set full candidate pool}"
QUERY_CACHE="${LAFGS_ALTERNATING_QUERY_CACHE:?Set native query cache}"
TRACK_PAYLOAD="${LAFGS_ALTERNATING_TRACK_PAYLOAD:?Set Track-First payload}"
DEPLOYMENT_MASK="${LAFGS_ALTERNATING_DEPLOYMENT_MASK:?Set masks.pkl}"
BASE_ROWS="${LAFGS_ALTERNATING_BASE_ROWS:-50048}"

export CUDA_VISIBLE_DEVICES="$GPU"
export CUDA_HOME=/usr/local/cuda-11.8
export PATH="/root/miniconda3/envs/cybersim_agent/bin:$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="/root/miniconda3/envs/cybersim_agent/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$REPO_ROOT"
export PYTHONHASHSEED=2026
mkdir -p "$OUTPUT_ROOT"
cd "$REPO_ROOT"

"$PYTHON" scripts/run_lafgs_alternating_structure.py \
  --base-map "$BASE_MAP" \
  --candidate-pool-map "$CANDIDATE_POOL" \
  --query-cache "$QUERY_CACHE" \
  --deployment-mask-cache "$DEPLOYMENT_MASK" \
  --output-dir "$OUTPUT_ROOT/round1_structure" \
  --batch-size 32 --max-candidates 160 --seed 2026

"$PYTHON" scripts/train_lafgs_micro_anchor_descriptors.py \
  --anchor-map "$OUTPUT_ROOT/round1_structure/active_map.pt" \
  --track-payload "$TRACK_PAYLOAD" \
  --query-cache "$QUERY_CACHE" \
  --deployment-mask-cache "$DEPLOYMENT_MASK" \
  --output-dir "$OUTPUT_ROOT/round1_refresh" \
  --train-start-row "$BASE_ROWS" \
  --steps 500 --checkpoint-steps 500 \
  --positive-batch-size 256 --guard-batch-size 512 \
  --guard-rows-per-query 128 \
  --positive-margin 0.01 --guard-margin 0.01 \
  --guard-weight 4 --trust-weight 0.2 \
  --learning-rate 0.001 --max-residual-norm 0.05 --seed 2026

"$PYTHON" scripts/build_lafgs_alternating_candidate_pool.py \
  --current-map "$OUTPUT_ROOT/round1_refresh/anchor_map_step_0500.pt" \
  --source-candidate-pool "$CANDIDATE_POOL" \
  --output "$OUTPUT_ROOT/round2_candidate_pool.pt" \
  --round 2

"$PYTHON" scripts/run_lafgs_alternating_structure.py \
  --base-map "$OUTPUT_ROOT/round1_refresh/anchor_map_step_0500.pt" \
  --candidate-pool-map "$OUTPUT_ROOT/round2_candidate_pool.pt" \
  --query-cache "$QUERY_CACHE" \
  --deployment-mask-cache "$DEPLOYMENT_MASK" \
  --output-dir "$OUTPUT_ROOT/round2_structure" \
  --batch-size 32 --max-candidates 64 --seed 2026
