#!/usr/bin/env bash
set -euo pipefail

# Full-test evaluation of the single oracle-gated solver change.  The frozen
# A2 map, descriptors, frontend, candidate count, and test protocol are reused.

if [[ $# -ne 2 ]]; then
  echo "Usage: bash $0 <scene> <gpu>" >&2
  exit 2
fi

SCENE="$1"
GPU="$2"
case "$SCENE" in
  GreatCourt|KingsCollege|ShopFacade|StMarysChurch) ;;
  *) echo "Unsupported frozen scene: $SCENE" >&2; exit 2 ;;
esac
case "$GPU" in
  0|1) ;;
  *) echo "Formal group consensus runs are restricted to GPU 0 or 1" >&2; exit 2 ;;
esac

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/root/miniconda3/envs/cybersim_agent/bin/python}"
DATA_ROOT="${CAMBRIDGE_DATA_ROOT:-/mnt/pool/sqy/Cambridge_stdloc}"
FROZEN_ROOT="${LAFGS_V1_MULTISCENE_ROOT:-/mnt/pool/sqy/stdloc_lafgs_v1_frozen_multiscene_20260731}"
OUTPUT_ROOT="${LAFGS_GROUP_SATURATED_ROOT:-/mnt/pool/sqy/stdloc_lafgs_v1_group_saturated_v2_20260731}"
ORACLE_ROOT="${LAFGS_GROUP_CONSENSUS_ROOT:-/mnt/pool/sqy/stdloc_lafgs_v1_group_consensus_oracle_20260731}"
SCENE_ROOT="$FROZEN_ROOT/$SCENE"
ROOT="$OUTPUT_ROOT/$SCENE"
MODEL_ROOT="$SCENE_ROOT/prior/rgb_matcha_2dgs"
SOURCE_ROOT="$DATA_ROOT/$SCENE"
GATE="$ORACLE_ROOT/formal_cross_scene_gate.json"

export CUDA_HOME=/usr/local/cuda-11.8
export PATH="/root/miniconda3/envs/cybersim_agent/bin:$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="/root/miniconda3/envs/cybersim_agent/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$REPO_ROOT"
export PYTHONHASHSEED=2026
export STDLOC_CAMERA_LOADER_WORKERS=0
mkdir -p "$ROOT"
cd "$REPO_ROOT"

"$PYTHON" - "$GATE" <<'PY'
import json
import sys

gate = json.load(open(sys.argv[1]))
if not gate.get("gate_pass"):
    raise SystemExit(
        "group-saturated solver did not pass the formal cross-scene gate"
    )
if gate.get("solver_versions") != ["group_saturated_poselib_parity_v2"]:
    raise SystemExit("formal gate does not match the compiled solver version")
PY

"$PYTHON" - <<'PY'
from localization_training.dependency_pose_sampler import (
    GROUP_SATURATED_SOLVER_VERSION,
    compiled_backend_available,
    compiled_group_saturated_solver_version,
)
if not compiled_backend_available():
    raise SystemExit("compiled LaFGS PoseLib backend is unavailable")
if compiled_group_saturated_solver_version() != GROUP_SATURATED_SOLVER_VERSION:
    raise SystemExit("compiled group-saturated solver version is stale")
PY

eval_one() {
  local seed="$1"
  local base="$SCENE_ROOT/evaluation/A2_family_all/seed${seed}/config.yaml"
  local output="$ROOT/seed${seed}"
  local cfg="$output/config.yaml"
  if [[ -f "$output/result.path" ]] && \
     [[ -f "$(<"$output/result.path")/results_summary.json" ]]; then
    return
  fi
  [[ -f "$base" ]] || {
    echo "Missing frozen A2 config: $base" >&2
    exit 1
  }
  mkdir -p "$output/results"
  "$PYTHON" scripts/make_stdloc_eval_cfg.py \
    --base_cfg "$base" --output "$cfg" \
    --artifact_model_path "$MODEL_ROOT" \
    --sparse_ransac_seed "$seed" \
    --sparse_solver poselib_group_saturated \
    --group_saturated_cap 8 \
    --group_saturated_surface_voxel_scale_ratio 0.02 \
    --group_saturated_surface_minimum_voxel_size 0.5 \
    --group_saturated_surface_normal_angle_degrees 25 \
    > "$output/config_build.json"
  "$PYTHON" - "$base" "$cfg" <<'PY'
import copy
import sys
import yaml

base = yaml.safe_load(open(sys.argv[1]))
current = yaml.safe_load(open(sys.argv[2]))
allowed = {
    "solver",
    "group_saturated_cap",
    "group_saturated_surface_voxel_scale_ratio",
    "group_saturated_surface_minimum_voxel_size",
    "group_saturated_surface_normal_angle_degrees",
}
for payload in (base, current):
    sparse = payload["sparse"]
    for key in allowed:
        sparse.pop(key, None)
if base != current:
    raise SystemExit("formal solver config changed fields outside the solver contract")
PY
  (
    export CUDA_VISIBLE_DEVICES="$GPU"
    export STDLOC_RESULTS_ROOT="$output/results"
    "$PYTHON" stdloc.py \
      --model_path "$MODEL_ROOT" --source_path "$SOURCE_ROOT" \
      --images processed --data_device cpu --gaussian_type 2dgs \
      --sh_degree 3 --feature_type sp --resolution 1 --longest_edge 0 \
      --norm_before_render --iteration 30000 --cfg "$cfg" \
      --prefix "lafgs-v1-$SCENE-A2-group-saturated-seed$seed" \
      --sparse_only --evaluation_camera_subset test \
      2>&1 | tee "$output/eval.log"
  )
  local result
  result="$(sed -n 's/^Output path: //p' "$output/eval.log" | tail -n 1)"
  [[ -n "$result" && -f "$result/results_summary.json" ]] || {
    echo "Formal solver evaluation failed for $SCENE seed $seed" >&2
    exit 1
  }
  printf '%s\n' "$result" > "$output/result.path"
}

for seed in 2026 2027 2028; do
  eval_one "$seed"
done

"$PYTHON" - "$SCENE_ROOT" "$ROOT" <<'PY'
import json
import sys
from pathlib import Path

import numpy as np

scene_root = Path(sys.argv[1])
root = Path(sys.argv[2])
baseline = json.loads((scene_root / "frozen_results.json").read_text())
payload = {
    "schema": "lafgs_v1_group_saturated_full_test_v2",
    "scene": scene_root.name,
    "solver_contract": {
        "base": "A2_family_all",
        "solver": "poselib_group_saturated",
        "surface_group_cap": 8.0,
        "surface_voxel_scale_ratio": 0.02,
        "surface_minimum_voxel_size_m": 0.5,
        "surface_normal_angle_degrees": 25.0,
        "implementation_version": "group_saturated_poselib_parity_v2",
    },
    "seeds": {},
}
metric_keys = (
    "median_te_cm",
    "mean_te_cm",
    "p90_te_cm",
    "median_ae_deg",
    "mean_ae_deg",
    "recall_2cm_2deg_percent",
    "recall_5cm_5deg_percent",
    "mean_hypotheses",
    "ransac_ms",
    "total_ms",
)
for pointer in sorted(root.glob("seed*/result.path")):
    seed = pointer.parent.name.removeprefix("seed")
    result = Path(pointer.read_text().strip())
    report = json.loads((result / "results_summary.json").read_text())
    rows = json.loads((result / "results.json").read_text())
    te = np.asarray([float(row["sparse_TE"]) for row in rows])
    ae = np.asarray([float(row["sparse_AE"]) for row in rows])
    sparse = report["sparse"]
    diag = report.get("sparse_diagnostics", {})
    current = {
        "query_count": len(rows),
        "median_te_cm": float(np.median(te)),
        "mean_te_cm": float(np.mean(te)),
        "p90_te_cm": float(np.percentile(te, 90)),
        "median_ae_deg": float(np.median(ae)),
        "mean_ae_deg": float(np.mean(ae)),
        "recall_2cm_2deg_percent": 100.0 * float(sparse["recall_2cm_2d"]),
        "recall_5cm_5deg_percent": 100.0 * float(sparse["recall_5cm_5d"]),
        "mean_hypotheses": diag.get(
            "sparse_diag_ransac_actual_hypotheses_mean"
        ),
        "ransac_ms": diag.get("sparse_diag_runtime_ransac_ms_mean"),
        "total_ms": diag.get("sparse_diag_runtime_total_ms_mean"),
        "group_score": diag.get(
            "sparse_diag_group_saturated_score_mean"
        ),
        "supported_groups": diag.get(
            "sparse_diag_group_saturated_supported_groups_mean"
        ),
        "group_ess": diag.get(
            "sparse_diag_group_saturated_group_ess_mean"
        ),
        "max_group_fraction": diag.get(
            "sparse_diag_group_saturated_max_group_fraction_mean"
        ),
        "result_path": str(result),
    }
    baseline_seed = baseline["results"]["A2_family_all"][seed]
    current["delta_vs_a2"] = {
        key: (
            float(current[key]) - float(baseline_seed[key])
            if current.get(key) is not None and baseline_seed.get(key) is not None
            else None
        )
        for key in metric_keys
    }
    payload["seeds"][seed] = current

payload["seed_aggregate"] = {
    key: {
        "mean": float(np.mean([row[key] for row in payload["seeds"].values()])),
        "std": float(np.std([row[key] for row in payload["seeds"].values()])),
    }
    for key in metric_keys
    if all(row.get(key) is not None for row in payload["seeds"].values())
}
baseline_aggregate = baseline["results"]["A2_family_all"]["seed_aggregate"]
payload["aggregate_delta_vs_a2"] = {
    key: (
        payload["seed_aggregate"][key]["mean"]
        - float(baseline_aggregate[key]["mean"])
    )
    for key in metric_keys
    if key in payload["seed_aggregate"] and key in baseline_aggregate
}
(root / "group_saturated_results.json").write_text(
    json.dumps(payload, indent=2) + "\n"
)

aggregate = payload["seed_aggregate"]
delta = payload["aggregate_delta_vs_a2"]
lines = [
    f"# {scene_root.name} Group-Saturated Full Test",
    "",
    "| Metric | Group-saturated | Delta vs A2 |",
    "|---|---:|---:|",
]
for key in metric_keys:
    if key not in aggregate:
        continue
    lines.append(
        f"| {key} | {aggregate[key]['mean']:.4f} | "
        f"{delta.get(key, float('nan')):+.4f} |"
    )
(root / "group_saturated_results.md").write_text("\n".join(lines) + "\n")
PY
