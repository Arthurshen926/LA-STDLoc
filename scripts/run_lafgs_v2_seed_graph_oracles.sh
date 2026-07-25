#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${GPU_ID:-2}"
PYTHON="${PYTHON:-/root/miniconda3/envs/cybersim_agent/bin/python}"
MODEL_ROOT="/mnt/pool/sqy/stdloc_lafgs_v2_ulfparity_alternating_20260721/matcha_wrappers/OldHospital"
LEGACY_ROOT="/mnt/pool/sqy/stdloc_lafgs_v2_assignment_v2_20260724/OldHospital/legacy_best_calibrated_null"
QUERY_CACHE="/mnt/pool/sqy/stdloc_lafgs_v2_ulfparity_alternating_20260721/OldHospital/ulfparity_native20k_s128_k2048_v2/query_cache_native_fullres_k2048.pt"
VISIBILITY_CACHE="/mnt/pool/sqy/stdloc_lafgs_v2_independent_bins_20260724/OldHospital/robustkcs_gwff32000_s0_uniform_mv2_v2_r0p01_vb8m2_tb4m2_ib1_fvb8_t0p1_dcm1p0_h64_refmean_adapt0_deployment_post_filter_v3_independent_bins_splitstratified_temporal_block_seed2026_fullres_native_uncapped/visibility_32000_native.pt"
SUPPORT_GRAPH="/mnt/pool/sqy/stdloc_lafgs_v2_assignment_v2_20260724/OldHospital/adapter_r16_res005/graph_full895.pt"
RUN_ROOT="/mnt/pool/sqy/stdloc_lafgs_v2_seed_graph_20260725/OldHospital"
CONFIG="$RUN_ROOT/legacy_assign_full182_oracle.yaml"
LOG="$RUN_ROOT/legacy_assign_full182_oracle.log"
RESULT_POINTER="$RUN_ROOT/legacy_assign_full182_oracle.results_path"
MATRIX="$RUN_ROOT/c0_c4_assignment_matrix.json"

mkdir -p "$RUN_ROOT" "$RUN_ROOT/results"
export CUDA_HOME=/usr/local/cuda-11.8
export LD_LIBRARY_PATH="/root/miniconda3/envs/cybersim_agent/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export PATH="/root/miniconda3/envs/cybersim_agent/bin:$CUDA_HOME/bin:$PATH"
export PYTHONPATH=/root/STDLoc
export STDLOC_RESULTS_ROOT="$RUN_ROOT/results"

"$PYTHON" - "$LEGACY_ROOT/eval.yaml" "$CONFIG" <<'PY'
import sys
from pathlib import Path

import yaml

source, output = map(Path, sys.argv[1:])
config = yaml.load(source.read_text(), Loader=yaml.FullLoader)
sparse = config["sparse"]
sparse["rerank_use_learned_null"] = False
sparse["sparse_only"] = True
for key in list(sparse):
    if key.startswith("query_context_"):
        del sparse[key]
diagnostics = sparse.setdefault("diagnostics", {})
diagnostics["enabled"] = True
diagnostics["gt_metrics"] = True
diagnostics["dump_discrete_oracle"] = True
diagnostics["oracle_topk"] = 32
output.write_text(yaml.safe_dump(config, sort_keys=False))
PY

if [[ "${PHASE:-all}" != "oracle" ]]; then
  if [[ ! -f "$RESULT_POINTER" ]] || \
     [[ ! -f "$(<"$RESULT_POINTER")/discrete_oracle_dump/manifest.json" ]]; then
    CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON" stdloc.py \
      -m "$MODEL_ROOT" \
      --iteration 30000 \
      --resolution 1 \
      --longest_edge 0 \
      --cfg "$CONFIG" \
      --prefix "lafgs-v2-legacyassign-seed-graph-dump" \
      --sparse_only \
      --evaluation_camera_subset test \
      2>&1 | tee "$LOG"
    OUTPUT_PATH="$(sed -n 's/^Output path: //p' "$LOG" | tail -n 1)"
    if [[ -z "$OUTPUT_PATH" ]] || \
       [[ ! -f "$OUTPUT_PATH/discrete_oracle_dump/manifest.json" ]]; then
      echo "LegacyAssign discrete oracle dump was not produced" >&2
      exit 1
    fi
    printf '%s\n' "$OUTPUT_PATH" > "$RESULT_POINTER"
  fi
fi

RESULT_PATH="$(<"$RESULT_POINTER")"
"$PYTHON" - "$RESULT_PATH/evaluation_protocol.json" <<'PY'
import json
import sys
from pathlib import Path

protocol = json.loads(Path(sys.argv[1]).read_text())
expected = {
    "resolution": 1,
    "longest_edge": 0,
    "evaluation_camera_count": 182,
}
errors = [
    f"{key}={protocol.get(key)!r}, expected {value!r}"
    for key, value in expected.items()
    if protocol.get(key) != value
]
if protocol.get("loaded_image_shapes") != [
    {"height": 1080, "width": 1920, "count": 182}
]:
    errors.append(
        "loaded_image_shapes="
        f"{protocol.get('loaded_image_shapes')!r}, expected full 1080x1920"
    )
if errors:
    raise SystemExit("Protocol mismatch:\n" + "\n".join(errors))
PY

"$PYTHON" scripts/eval_seed_graph_context_oracles.py \
  --dump_dir "$RESULT_PATH/discrete_oracle_dump" \
  --query_cache "$QUERY_CACHE" \
  --visibility_cache "$VISIBILITY_CACHE" \
  --support_graph "$SUPPORT_GRAPH" \
  --output "$MATRIX" \
  --nearest_views 8 \
  --minimum_cohits 2 \
  --seed_count 16 \
  --protect_margin_quantile 0.75 \
  --protect_reliability_quantile 0.75 \
  --confusion_weight 1.0 \
  --delta_values 0.01,0.02,0.05,0.10,0.20 \
  --run_passing_pose
