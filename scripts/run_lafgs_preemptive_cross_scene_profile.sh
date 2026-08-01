#!/usr/bin/env bash
set -euo pipefail

# Low-cost exact-preemptive crossover audit on each scene's highest-hypothesis
# frozen A1 queries.  This is a runtime profiler, not a method experiment.

if [[ $# -ne 2 ]]; then
  echo "Usage: bash $0 <scene> <gpu>" >&2
  exit 2
fi
SCENE="$1"
GPU="$2"
case "$SCENE" in
  GreatCourt|KingsCollege|OldHospital|ShopFacade|StMarysChurch) ;;
  *) echo "Unsupported Cambridge scene: $SCENE" >&2; exit 2 ;;
esac
case "$GPU" in 0|1) ;; *) exit 2 ;; esac

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/root/miniconda3/envs/cybersim_agent/bin/python}"
DATA_ROOT="${CAMBRIDGE_DATA_ROOT:-/mnt/pool/sqy/Cambridge_stdloc}"
FROZEN_ROOT="${LAFGS_V1_MULTISCENE_ROOT:-/mnt/pool/sqy/stdloc_lafgs_v1_frozen_multiscene_20260731}"
OUTPUT_ROOT="${LAFGS_PREEMPTIVE_PROFILE_ROOT:-/mnt/pool/sqy/stdloc_lafgs_preemptive_cross_scene_production_20260731}"
QUERY_COUNT="${LAFGS_PREEMPTIVE_PROFILE_QUERY_COUNT:-20}"

SCENE_ROOT="$FROZEN_ROOT/$SCENE"
MODEL_ROOT="$SCENE_ROOT/prior/rgb_matcha_2dgs"
SOURCE_ROOT="$DATA_ROOT/$SCENE"
BOOTSTRAP="$SCENE_ROOT/runs/frozen_v1/bootstrap"
MAP="$SCENE_ROOT/self_localization_reconstruction/anchor_map_step_0175.pt"
METRIC="$SCENE_ROOT/self_localization_reconstruction/metric_state_step_0175.pt"
ROOT="$OUTPUT_ROOT/$SCENE"
QUERY_LIST="$ROOT/high_hypothesis_queries.json"

export CUDA_HOME=/usr/local/cuda-11.8
export PATH="/root/miniconda3/envs/cybersim_agent/bin:$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="/root/miniconda3/envs/cybersim_agent/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$REPO_ROOT"
export STDLOC_CAMERA_LOADER_WORKERS=0
mkdir -p "$ROOT"
cd "$REPO_ROOT"

for path in "$MAP" "$METRIC" "$BOOTSTRAP/sampled_idx.pkl" "$BOOTSTRAP/landmark_meta.pt"; do
  [[ -f "$path" ]] || { echo "Missing frozen A1 artifact: $path" >&2; exit 1; }
done

"$PYTHON" - "$SCENE_ROOT/frozen_results.json" "$QUERY_LIST" "$QUERY_COUNT" <<'PY'
import json
import sys
from pathlib import Path

summary_path, output_path, count = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
summary = json.loads(summary_path.read_text())
seed = summary["results"]["A1_reconstructed"]["2026"]
rows = json.loads((Path(seed["result_path"]) / "results.json").read_text())
rows.sort(
    key=lambda row: float(
        row["sparse"].get("sparse_diag_ransac_actual_hypotheses", -1) or -1
    ),
    reverse=True,
)
payload = {
    "image_names": [row["image_name"] for row in rows[:count]],
    "selection": "highest_frozen_A1_ransac_hypotheses",
    "source": str(summary_path.resolve()),
}
output_path.write_text(json.dumps(payload, indent=2) + "\n")
PY

run_variant() {
  local label="$1"
  local solver="$2"
  local output="$ROOT/$label"
  local cfg="$output/config.yaml"
  mkdir -p "$output/results"
  if [[ -f "$output/result.path" ]] && [[ -f "$(<"$output/result.path")/results.json" ]]; then
    return
  fi
  local extra=()
  if [[ "$solver" == "poselib_preemptive" ]]; then
    extra+=(--preemptive_verification_order low_confidence --preemptive_check_interval 32)
  fi
  "$PYTHON" scripts/make_stdloc_eval_cfg.py \
    --base_cfg configs/stdloc_cambridge.yaml --output "$cfg" \
    --artifact_model_path "$MODEL_ROOT" \
    --detector_folder ulfloc_native_no_detector --detector_iters 0 \
    --landmark_path "$BOOTSTRAP/sampled_idx.pkl" \
    --landmark_meta_path "$BOOTSTRAP/landmark_meta.pt" \
    --detect_num 2048 --nms 2 --sparse_ransac_seed 2026 \
    --sparse_query_feature_contract native_resized_input \
    --reprojection_error 12 --match_threshold 0 --match_topk 1 \
    --max_matches_per_keypoint 0 --max_matches_per_landmark 0 \
    --candidate_frontend_match_policy error \
    --sparse_frontend ulfloc_native_metric \
    --materialized_anchor_map_path "$MAP" --metric_state_path "$METRIC" \
    --sparse_solver "$solver" "${extra[@]}" > "$output/config_build.json"
  (
    export CUDA_VISIBLE_DEVICES="$GPU"
    export STDLOC_RESULTS_ROOT="$output/results"
    "$PYTHON" stdloc.py \
      --model_path "$MODEL_ROOT" --source_path "$SOURCE_ROOT" \
      --images processed --data_device cpu --gaussian_type 2dgs \
      --sh_degree 3 --feature_type sp --resolution 1 --longest_edge 0 \
      --norm_before_render --iteration 30000 --cfg "$cfg" \
      --prefix "lafgs-preemptive-profile-$SCENE-$label" --sparse_only \
      --evaluation_camera_list "$QUERY_LIST" \
      --evaluation_camera_list_test_only 2>&1 | tee "$output/eval.log"
  )
  local result
  result="$(sed -n 's/^Output path: //p' "$output/eval.log" | tail -n 1)"
  [[ -n "$result" && -f "$result/results.json" ]] || exit 1
  printf '%s\n' "$result" > "$output/result.path"
}

run_variant native_poselib poselib
run_variant exact_preemptive poselib_preemptive
"$PYTHON" scripts/evaluate_lafgs_preemptive_parity.py \
  --baseline "$(<"$ROOT/native_poselib/result.path")/results.json" \
  --candidate "$(<"$ROOT/exact_preemptive/result.path")/results.json" \
  --output "$ROOT/parity.json"
