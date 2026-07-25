#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${GPU_ID:-2}"
PYTHON="${PYTHON:-/root/miniconda3/envs/cybersim_agent/bin/python}"
MODEL_ROOT="/mnt/pool/sqy/stdloc_lafgs_v2_ulfparity_alternating_20260721/matcha_wrappers/OldHospital"
BASE_RESULT="/root/STDLoc/results/lafgs-v2-newmain-p0-oracle-_mnt_pool_sqy_stdloc_lafgs_v2_ulfparity_alternating_20260721_matcha_wrappers_OldHospital-20260724_210421"
BASE_CONFIG="$BASE_RESULT/eval.yaml"
BASE_RESULTS="$BASE_RESULT/results.json"
SHARD="/mnt/pool/sqy/stdloc_lafgs_v2_assignment_v2_20260724/OldHospital/eval_shards/shard0.json"
CONTEXT_STATE="/mnt/pool/sqy/stdloc_lafgs_v2_query_context_20260725/OldHospital/p0_spatial2x2/context_index.pt"
RUN_ROOT="/mnt/pool/sqy/stdloc_lafgs_v2_query_context_20260725/OldHospital/p0_matrix"
CONFIG_ROOT="$RUN_ROOT/configs"
LOG_ROOT="$RUN_ROOT/logs"
RESULT_ROOT="$RUN_ROOT/results"

mkdir -p "$CONFIG_ROOT" "$LOG_ROOT" "$RESULT_ROOT"
export CUDA_HOME=/usr/local/cuda-11.8
export LD_LIBRARY_PATH="/root/miniconda3/envs/cybersim_agent/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export PATH="/root/miniconda3/envs/cybersim_agent/bin:$CUDA_HOME/bin:$PATH"
export PYTHONPATH=/root/STDLoc
export STDLOC_RESULTS_ROOT="$RESULT_ROOT"

make_config() {
  local output="$1"
  local delta="$2"
  local margin="$3"
  local normalization="${4:-absolute}"
  local prior_scale="${5:-0.1}"
  "$PYTHON" - "$BASE_CONFIG" "$output" "$CONTEXT_STATE" "$delta" "$margin" "$normalization" "$prior_scale" <<'PY'
import sys
from pathlib import Path

import yaml

(
    base_path,
    output_path,
    context_path,
    delta,
    margin,
    normalization,
    prior_scale,
) = sys.argv[1:]
config = yaml.load(Path(base_path).read_text(), Loader=yaml.FullLoader)
sparse = config["sparse"]
sparse["query_context_state_path"] = context_path
sparse["query_context_grid_rows"] = 2
sparse["query_context_grid_cols"] = 2
sparse["query_context_nearest_views"] = 8
sparse["query_context_temperature"] = 0.05
sparse["query_context_delta_max"] = float(delta)
sparse["query_context_prior_center"] = 0.1
sparse["query_context_prior_scale"] = float(prior_scale)
sparse["query_context_normalization"] = normalization
sparse["query_context_margin_threshold"] = float(margin)
sparse["diagnostics"]["dump_discrete_oracle"] = False
Path(output_path).write_text(yaml.safe_dump(config, sort_keys=False))
PY
}

run_variant() {
  local label="$1"
  local delta="$2"
  local margin="$3"
  local normalization="${4:-absolute}"
  local prior_scale="${5:-0.1}"
  local config="$CONFIG_ROOT/$label.yaml"
  local log="$LOG_ROOT/$label.log"
  local reference="$RUN_ROOT/$label.results_path"
  if [[ -f "$reference" ]] && [[ -f "$(<"$reference")/results.json" ]]; then
    return
  fi
  make_config "$config" "$delta" "$margin" "$normalization" "$prior_scale"
  CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON" stdloc.py \
    -m "$MODEL_ROOT" \
    --iteration 30000 \
    --resolution 1 \
    --longest_edge 0 \
    --cfg "$config" \
    --prefix "lafgs-v2-query-context-p0-$label" \
    --sparse_only \
    --evaluation_camera_list "$SHARD" \
    --evaluation_camera_list_test_only \
    2>&1 | tee "$log"
  local output_path
  output_path="$(sed -n 's/^Output path: //p' "$log" | tail -n 1)"
  if [[ -z "$output_path" ]] || [[ ! -f "$output_path/results.json" ]]; then
    echo "Missing result for $label" >&2
    exit 1
  fi
  "$PYTHON" - "$output_path/evaluation_protocol.json" "$SHARD" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

protocol_path, camera_list_path = map(Path, sys.argv[1:])
protocol = json.loads(protocol_path.read_text())
expected_list_hash = hashlib.sha256(camera_list_path.read_bytes()).hexdigest()
shapes = protocol.get("loaded_image_shapes", [])
valid = (
    protocol.get("longest_edge") == 0
    and protocol.get("resolution") == 1
    and protocol.get("evaluation_camera_count") == 46
    and protocol.get("evaluation_camera_list_sha256") == expected_list_hash
    and shapes == [{"height": 1080, "width": 1920, "count": 46}]
)
if not valid:
    raise SystemExit(
        "full-resolution P0 protocol mismatch: "
        + json.dumps(protocol, indent=2)
    )
PY
  printf '%s\n' "$output_path" > "$reference"
}

case "${VARIANT:-all}" in
  d001_m0005) run_variant d001_m0005 0.01 0.005 ;;
  d001_m0010) run_variant d001_m0010 0.01 0.010 ;;
  d002_m0005) run_variant d002_m0005 0.02 0.005 ;;
  d002_m0010) run_variant d002_m0010 0.02 0.010 ;;
  gl_d001_m0005)
    run_variant gl_d001_m0005 0.01 0.005 global_lift 1.0
    ;;
  gl_d002_m0005)
    run_variant gl_d002_m0005 0.02 0.005 global_lift 1.0
    ;;
  all)
    run_variant d001_m0005 0.01 0.005
    run_variant d001_m0010 0.01 0.010
    run_variant d002_m0005 0.02 0.005
    run_variant d002_m0010 0.02 0.010
    run_variant gl_d001_m0005 0.01 0.005 global_lift 1.0
    run_variant gl_d002_m0005 0.02 0.005 global_lift 1.0
    ;;
  *) echo "Unknown VARIANT=$VARIANT" >&2; exit 2 ;;
esac

"$PYTHON" - "$BASE_RESULTS" "$SHARD" "$RUN_ROOT" <<'PY'
import json
import sys
from pathlib import Path

import numpy as np

base_results_path, shard_path, run_root = map(Path, sys.argv[1:])
names = set(json.loads(shard_path.read_text()))


def metrics(records):
    te = np.asarray([row["sparse_TE"] for row in records], dtype=np.float64)
    ae = np.asarray([row["sparse_AE"] for row in records], dtype=np.float64)
    diagnostics = [
        row["sparse"] for row in records
    ]

    def diag_mean(key):
        values = [
            float(row[key])
            for row in diagnostics
            if key in row and np.isfinite(float(row[key]))
        ]
        return float(np.mean(values)) if values else None

    return {
        "count": len(records),
        "median_te_cm": float(np.median(te)),
        "mean_te_cm": float(np.mean(te)),
        "p90_te_cm": float(np.quantile(te, 0.9)),
        "recall_5cm_5d": float(np.mean((te <= 5.0) & (ae <= 5.0))),
        "raw_gt_precision_2px": diag_mean(
            "sparse_diag_all_gt_precision_2px"
        ),
        "inlier_gt_precision_2px": diag_mean(
            "sparse_diag_inlier_gt_precision_2px"
        ),
        "conditional_recall_at_1_2px": diag_mean(
            "sparse_diag_conditional_recall_at_1_given_matchable_2px"
        ),
        "matching_ms": diag_mean("sparse_diag_runtime_matching_ms"),
        "context_ambiguous_fraction": diag_mean(
            "sparse_diag_native_query_context_ambiguous_fraction"
        ),
    }


base_records = [
    row
    for row in json.loads(base_results_path.read_text())
    if row["image_name"] in names
]
summary = {"baseline_same_queries": metrics(base_records), "variants": {}}
for reference in sorted(run_root.glob("*.results_path")):
    result_dir = Path(reference.read_text().strip())
    records = json.loads((result_dir / "results.json").read_text())
    summary["variants"][reference.stem] = metrics(records)
(run_root / "pilot_summary.json").write_text(
    json.dumps(summary, indent=2) + "\n"
)
print(json.dumps(summary, indent=2))
PY
