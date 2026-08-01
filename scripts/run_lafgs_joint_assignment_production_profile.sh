#!/usr/bin/env bash
set -euo pipefail

# Full-test production profile for the frozen joint-assignment operating point.
# Accuracy runs intentionally enable GT diagnostics; this profile intentionally
# does not, so retrieval width and timing match deployment.

if [[ $# -ne 3 ]]; then
  echo "Usage: bash $0 <scene> <gpu> <A1-All|S512-PoseSufficient|S1024-Block8>" >&2
  exit 2
fi

SCENE="$1"
GPU="$2"
SELECTOR="$3"
case "$SCENE" in
  GreatCourt|KingsCollege|OldHospital|ShopFacade|StMarysChurch) ;;
  *) echo "Unsupported Cambridge scene: $SCENE" >&2; exit 2 ;;
esac
case "$GPU" in 0|1) ;; *) echo "GPU must be 0 or 1" >&2; exit 2 ;; esac
case "$SELECTOR" in
  A1-All|S512-PoseSufficient|S1024-Block8) ;;
  *) echo "Unsupported selector: $SELECTOR" >&2; exit 2 ;;
esac

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/root/miniconda3/envs/cybersim_agent/bin/python}"
DATA_ROOT="${CAMBRIDGE_DATA_ROOT:-/mnt/pool/sqy/Cambridge_stdloc}"
FROZEN_ROOT="${LAFGS_V1_MULTISCENE_ROOT:-/mnt/pool/sqy/stdloc_lafgs_v1_frozen_multiscene_20260731}"
ASSIGNMENT_ROOT="${LAFGS_JOINT_ASSIGNMENT_P1_ROOT:-/mnt/pool/sqy/stdloc_lafgs_joint_assignment_p1_20260731}"
OUTPUT_ROOT="${LAFGS_JOINT_ASSIGNMENT_PRODUCTION_ROOT:-/mnt/pool/sqy/stdloc_lafgs_joint_assignment_production_20260801}"

SCENE_ROOT="$FROZEN_ROOT/$SCENE"
MODEL_ROOT="$SCENE_ROOT/prior/rgb_matcha_2dgs"
SOURCE_ROOT="$DATA_ROOT/$SCENE"
BOOTSTRAP="$SCENE_ROOT/runs/frozen_v1/bootstrap"
MAP="$SCENE_ROOT/self_localization_reconstruction/anchor_map_step_0175.pt"
METRIC="$SCENE_ROOT/self_localization_reconstruction/metric_state_step_0175.pt"
ASSIGNMENT="$ASSIGNMENT_ROOT/models/heldout_$SCENE/joint_assignment_K8_v4.pt"
ROOT="$OUTPUT_ROOT/$SCENE/$SELECTOR/seed2026"

export CUDA_HOME=/usr/local/cuda-11.8
export PATH="/root/miniconda3/envs/cybersim_agent/bin:$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="/root/miniconda3/envs/cybersim_agent/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$REPO_ROOT"
export PYTHONHASHSEED=2026
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export STDLOC_CAMERA_LOADER_WORKERS=0
mkdir -p "$ROOT"
cd "$REPO_ROOT"

for path in "$MAP" "$METRIC" "$ASSIGNMENT" \
  "$BOOTSTRAP/sampled_idx.pkl" "$BOOTSTRAP/landmark_meta.pt"; do
  [[ -f "$path" ]] || { echo "Missing production artifact: $path" >&2; exit 1; }
done

run_variant() {
  local label="$1"
  local frontend="$2"
  local output="$ROOT/$label"
  local cfg="$output/config.yaml"
  mkdir -p "$output/results"
  if [[ -f "$output/result.path" ]]; then
    local existing
    existing="$(<"$output/result.path")"
    if [[ -f "$existing/results.json" ]]; then
      return
    fi
  fi
  local extra=()
  if [[ "$frontend" == "ulfloc_native_rerank" ]]; then
    extra+=(
      --rerank_state_path "$ASSIGNMENT"
      --rerank_topk 8
      --rerank_patch_radius 2
      --rerank_patch_step_px 8
      --rerank_use_learned_null
      --joint_assignment_fixed_selector "$SELECTOR"
    )
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
    --sparse_frontend "$frontend" \
    --materialized_anchor_map_path "$MAP" --metric_state_path "$METRIC" \
    "${extra[@]}" > "$output/config_build.json"

  "$PYTHON" - "$cfg" "$frontend" <<'PY'
import sys
import yaml

config = yaml.safe_load(open(sys.argv[1]))
sparse = config["sparse"]
diagnostics = sparse.get("diagnostics") or {}
if diagnostics.get("enabled", False) or diagnostics.get("gt_metrics", False):
    raise RuntimeError("production profile must disable diagnostic retrieval")
if sparse.get("frontend") != sys.argv[2]:
    raise RuntimeError("production frontend contract was not materialized")
PY

  (
    export CUDA_VISIBLE_DEVICES="$GPU"
    export STDLOC_RESULTS_ROOT="$output/results"
    "$PYTHON" stdloc.py \
      --model_path "$MODEL_ROOT" --source_path "$SOURCE_ROOT" \
      --images processed --data_device cpu --gaussian_type 2dgs \
      --sh_degree 3 --feature_type sp --resolution 1 --longest_edge 0 \
      --norm_before_render --iteration 30000 --cfg "$cfg" \
      --prefix "lafgs-joint-production-$SCENE-$SELECTOR-$label" \
      --sparse_only --evaluation_camera_subset test \
      2>&1 | tee "$output/eval.log"
  )
  local result
  result="$(sed -n 's/^Output path: //p' "$output/eval.log" | tail -n 1)"
  [[ -n "$result" && -f "$result/results.json" ]] || exit 1
  printf '%s\n' "$result" > "$output/result.path"
}

run_variant A1-production ulfloc_native_metric
run_variant joint-production ulfloc_native_rerank

"$PYTHON" scripts/summarize_joint_assignment_production_profile.py \
  --scene "$SCENE" --selector "$SELECTOR" \
  --baseline "$(<"$ROOT/A1-production/result.path")/results.json" \
  --candidate "$(<"$ROOT/joint-production/result.path")/results.json" \
  --output "$ROOT/profile.json"
