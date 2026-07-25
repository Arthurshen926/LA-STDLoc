#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${GPU_ID:-2}"
PYTHON="${PYTHON:-/root/miniconda3/envs/cybersim_agent/bin/python}"
MODEL_ROOT="/mnt/pool/sqy/stdloc_lafgs_v2_ulfparity_alternating_20260721/matcha_wrappers/OldHospital"
LEGACY_ROOT="/mnt/pool/sqy/stdloc_lafgs_v2_assignment_v2_20260724/OldHospital/legacy_best_calibrated_null"
RUN_ROOT="/mnt/pool/sqy/stdloc_lafgs_v2_retrieval_width_20260725/OldHospital"
WIDTH="${WIDTH:-8}"
CONFIG="$RUN_ROOT/legacy_assign_top${WIDTH}.yaml"
LOG="$RUN_ROOT/legacy_assign_top${WIDTH}.log"
RESULT_POINTER="$RUN_ROOT/legacy_assign_top${WIDTH}.results_path"

mkdir -p "$RUN_ROOT" "$RUN_ROOT/results"
export CUDA_HOME=/usr/local/cuda-11.8
export LD_LIBRARY_PATH="/root/miniconda3/envs/cybersim_agent/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export PATH="/root/miniconda3/envs/cybersim_agent/bin:$CUDA_HOME/bin:$PATH"
export PYTHONPATH=/root/STDLoc
export STDLOC_RESULTS_ROOT="$RUN_ROOT/results"

"$PYTHON" - "$LEGACY_ROOT/eval.yaml" "$CONFIG" "$WIDTH" <<'PY'
import sys
from pathlib import Path

import yaml

source, output = map(Path, sys.argv[1:3])
width = int(sys.argv[3])
config = yaml.load(source.read_text(), Loader=yaml.FullLoader)
sparse = config["sparse"]
sparse["rerank_use_learned_null"] = False
sparse["rerank_topk"] = width
sparse["rerank_allow_candidate_only_topk_mismatch"] = True
sparse["topk"] = 1
sparse["max_matches_per_keypoint"] = 0
sparse["max_matches_per_landmark"] = 0
for key in list(sparse):
    if key.startswith("query_context_"):
        del sparse[key]
diagnostics = sparse.setdefault("diagnostics", {})
diagnostics["enabled"] = True
diagnostics["gt_metrics"] = True
diagnostics["dump_discrete_oracle"] = False
output.write_text(yaml.safe_dump(config, sort_keys=False))
PY

if [[ ! -f "$RESULT_POINTER" ]] || \
   [[ ! -f "$(<"$RESULT_POINTER")/results.json" ]]; then
  CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON" stdloc.py \
    -m "$MODEL_ROOT" \
    --iteration 30000 \
    --resolution 1 \
    --longest_edge 0 \
    --cfg "$CONFIG" \
    --prefix "lafgs-v2-legacyassign-top${WIDTH}" \
    --sparse_only \
    --evaluation_camera_subset test \
    2>&1 | tee "$LOG"
  OUTPUT_PATH="$(sed -n 's/^Output path: //p' "$LOG" | tail -n 1)"
  if [[ -z "$OUTPUT_PATH" ]] || [[ ! -f "$OUTPUT_PATH/results.json" ]]; then
    echo "Top-${WIDTH} result was not produced" >&2
    exit 1
  fi
  printf '%s\n' "$OUTPUT_PATH" > "$RESULT_POINTER"
fi

RESULT_PATH="$(<"$RESULT_POINTER")"
"$PYTHON" - "$RESULT_PATH" "$WIDTH" <<'PY'
import json
import sys
from pathlib import Path

import numpy as np

result_path = Path(sys.argv[1])
width = int(sys.argv[2])
protocol = json.loads((result_path / "evaluation_protocol.json").read_text())
if not (
    protocol.get("resolution") == 1
    and protocol.get("longest_edge") == 0
    and protocol.get("evaluation_camera_count") == 182
    and protocol.get("loaded_image_shapes")
    == [{"height": 1080, "width": 1920, "count": 182}]
):
    raise SystemExit("full-resolution 182-query protocol mismatch")
records = json.loads((result_path / "results.json").read_text())
te = np.asarray([row["sparse_TE"] for row in records])
ae = np.asarray([row["sparse_AE"] for row in records])

def diagnostic_mean(key):
    values = [
        float(row["sparse"][key])
        for row in records
        if key in row["sparse"]
    ]
    return float(np.mean(values)) if values else None

summary = {
    "rerank_topk": width,
    "query_count": len(records),
    "median_te_cm": float(np.median(te)),
    "mean_te_cm": float(np.mean(te)),
    "p90_te_cm": float(np.quantile(te, 0.9)),
    "recall_5cm_5deg": float(np.mean((te <= 5.0) & (ae <= 5.0))),
    "seq4_median_te_cm": float(np.median(te[:56])),
    "seq8_median_te_cm": float(np.median(te[56:])),
    "raw_retrieval_recall_at_1_given_matchable_2px": diagnostic_mean(
        "sparse_diag_conditional_recall_at_1_given_matchable_2px"
    ),
    "rerank_positive_in_topk_rate": diagnostic_mean(
        "sparse_diag_native_rerank_gt_positive_in_topk_rate"
    ),
    "rerank_selection_accuracy_given_topk_positive": diagnostic_mean(
        "sparse_diag_native_rerank_gt_conditional_selection_accuracy"
    ),
    "rerank_clean_top1_retention": diagnostic_mean(
        "sparse_diag_native_rerank_gt_clean_top1_retention"
    ),
    "beneficial_swap_rate": diagnostic_mean(
        "sparse_diag_native_rerank_gt_beneficial_swap_rate"
    ),
    "harmful_swap_rate": diagnostic_mean(
        "sparse_diag_native_rerank_gt_harmful_swap_rate"
    ),
    "matching_ms": diagnostic_mean("sparse_diag_runtime_matching_ms"),
}
(result_path / "retrieval_width_summary.json").write_text(
    json.dumps(summary, indent=2) + "\n"
)
print(json.dumps(summary, indent=2))
PY
