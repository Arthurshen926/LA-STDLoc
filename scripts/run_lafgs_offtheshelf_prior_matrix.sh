#!/usr/bin/env bash
set -euo pipefail

# Official RGB Gaussian prior -> frozen LaFGS paper mainline. Prior training is
# mapping-only and mask-free; localization uses the fixed native-SP A0/A1
# protocol and one PoseLib PnP/RANSAC solve.

if [[ $# -ne 3 ]]; then
  echo "Usage: bash $0 <OldHospital|KingsCollege> <vanilla_3dgs|vanilla_2dgs> <train|export|quality|lafgs|eval|all|status>" >&2
  exit 2
fi

SCENE="$1"
PROFILE="$2"
MODE="$3"
case "$SCENE" in
  OldHospital|KingsCollege) ;;
  *) echo "Unsupported prior-robustness scene: $SCENE" >&2; exit 2 ;;
esac
case "$PROFILE" in
  vanilla_3dgs) GAUSSIAN_TYPE="3dgs" ;;
  vanilla_2dgs) GAUSSIAN_TYPE="2dgs" ;;
  *) echo "Unsupported off-the-shelf prior: $PROFILE" >&2; exit 2 ;;
esac
case "$MODE" in
  train|export|quality|lafgs|eval|all|status) ;;
  *) echo "Unsupported mode: $MODE" >&2; exit 2 ;;
esac

GPU="${LAFGS_OFFTHESHELF_GPU:-2}"
if [[ "$GPU" != "2" ]]; then
  echo "Off-the-shelf prior matrix is restricted to GPU2" >&2
  exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/root/miniconda3/envs/cybersim_agent/bin/python}"
CAMBRIDGE_ROOT="${CAMBRIDGE_DATA_ROOT:-/mnt/pool/sqy/Cambridge_stdloc}"
EXPERIMENT_ROOT="${LAFGS_OFFTHESHELF_ROOT:-/mnt/pool/sqy/stdloc_lafgs_offtheshelf_prior_20260802}"
LAFGS_NAMESPACE="${LAFGS_OFFTHESHELF_LAFGS_NAMESPACE:-lafgs_strict_v2}"
THIRD_PARTY_ROOT="${LAFGS_OFFTHESHELF_THIRD_PARTY_ROOT:-/mnt/pool/sqy/third_party_lafgs_priors}"
MAPPING_DATASET="$EXPERIMENT_ROOT/datasets/${SCENE}_mapping_only_undistorted"
if [[ "$PROFILE" == "vanilla_2dgs" ]]; then
  DATASET="${MAPPING_DATASET}_flat"
else
  DATASET="$MAPPING_DATASET"
fi
PRIOR_ROOT="$EXPERIMENT_ROOT/priors/$SCENE/$PROFILE"
NATIVE_MODEL="$PRIOR_ROOT/native"
MODEL_ROOT="$PRIOR_ROOT/stdloc_model"
LOG_ROOT="$PRIOR_ROOT/logs"
ITERATIONS=30000

if [[ "$PROFILE" == "vanilla_3dgs" ]]; then
  OFFICIAL_REPO="$THIRD_PARTY_ROOT/gaussian-splatting"
  PINNED_COMMIT="54c035f7834b564019656c3e3fcc3646292f727d"
  RASTERIZER_ROOT="$OFFICIAL_REPO/submodules/diff-gaussian-rasterization"
  EXTRA_MODULE_ROOT="$OFFICIAL_REPO/submodules/fused-ssim"
else
  OFFICIAL_REPO="$THIRD_PARTY_ROOT/2d-gaussian-splatting"
  PINNED_COMMIT="335ad612f2e783a4e57b9cbc4d1e167bd599fc98"
  RASTERIZER_ROOT="$OFFICIAL_REPO/submodules/diff-surfel-rasterization"
  EXTRA_MODULE_ROOT=""
fi
SIMPLE_KNN_ROOT="$OFFICIAL_REPO/submodules/simple-knn"

export CUDA_HOME=/usr/local/cuda-11.8
export PATH="/root/miniconda3/envs/cybersim_agent/bin:$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="/root/miniconda3/envs/cybersim_agent/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONHASHSEED=2026
mkdir -p "$PRIOR_ROOT" "$LOG_ROOT"

require_file() {
  [[ -f "$1" ]] || { echo "Required file is missing: $1" >&2; exit 1; }
}

verify_protocol() {
  require_file "$DATASET/mapping_only_manifest.json"
  local actual_commit
  actual_commit="$(git -C "$OFFICIAL_REPO" rev-parse HEAD)"
  [[ "$actual_commit" == "$PINNED_COMMIT" ]] || {
    echo "Official repository commit mismatch: $actual_commit" >&2
    exit 1
  }
  "$PYTHON" - "$DATASET/mapping_only_manifest.json" <<'PY'
import json
import sys

manifest = json.load(open(sys.argv[1]))
assert manifest["semantic_mask_used"] is False
assert manifest["undistortion_used"] is True
assert manifest["target_camera_models"] == ["PINHOLE"]
assert manifest["excluded_test_image_count"] > 0
PY
}

run_logged() {
  local name="$1"
  shift
  printf '%q ' "$@" > "$LOG_ROOT/${name}.command.sh"
  printf '\n' >> "$LOG_ROOT/${name}.command.sh"
  "$@" 2>&1 | tee "$LOG_ROOT/${name}.log"
}

train_prior() {
  verify_protocol
  local ply="$NATIVE_MODEL/point_cloud/iteration_${ITERATIONS}/point_cloud.ply"
  if [[ -f "$ply" ]]; then
    echo "Reusing official prior: $ply"
    return
  fi
  local started finished
  started="$(date +%s)"
  local module_path="$RASTERIZER_ROOT:$SIMPLE_KNN_ROOT:$OFFICIAL_REPO"
  if [[ -n "$EXTRA_MODULE_ROOT" ]]; then
    module_path="$RASTERIZER_ROOT:$SIMPLE_KNN_ROOT:$EXTRA_MODULE_ROOT:$OFFICIAL_REPO"
  fi
  local args=(
    env CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH="$module_path"
    "$PYTHON" "$OFFICIAL_REPO/train.py"
    --source_path "$DATASET" --model_path "$NATIVE_MODEL"
    --images images --resolution 1 --data_device cpu
    --sh_degree 3 --iterations "$ITERATIONS"
    --test_iterations "$ITERATIONS" --save_iterations "$ITERATIONS"
    --quiet
  )
  if [[ "$PROFILE" == "vanilla_3dgs" ]]; then
    args+=(
      --depth_l1_weight_init 0 --depth_l1_weight_final 0
      --disable_viewer
    )
  else
    args+=(--port 6012)
  fi
  run_logged train_official "${args[@]}"
  require_file "$ply"
  finished="$(date +%s)"
  printf '%s\n' "$((finished - started))" > "$PRIOR_ROOT/training_seconds.txt"
}

export_prior() {
  train_prior
  local input_ply="$NATIVE_MODEL/point_cloud/iteration_${ITERATIONS}/point_cloud.ply"
  local output_ply="$MODEL_ROOT/point_cloud/iteration_${ITERATIONS}/point_cloud.ply"
  if [[ ! -f "$MODEL_ROOT/rgb_prior_manifest.json" || ! -f "$output_ply" ]]; then
    run_logged export_prior \
      "$PYTHON" "$REPO_ROOT/scripts/export_rgb_gaussian_prior.py" \
      --input_ply "$input_ply" --output_model "$MODEL_ROOT" \
      --gaussian_type "$GAUSSIAN_TYPE" --sh_degree 3 \
      --source_path "$CAMBRIDGE_ROOT/$SCENE" --images processed \
      --longest_edge 0 --iteration "$ITERATIONS" --prior_kind rgb_only \
      --black_background
  fi
  require_file "$MODEL_ROOT/rgb_prior_manifest.json"
  require_file "$output_ply"
  "$PYTHON" - \
    "$DATASET/mapping_only_manifest.json" \
    "$MODEL_ROOT/rgb_prior_manifest.json" \
    "$PRIOR_ROOT/offtheshelf_prior_protocol.json" \
    "$OFFICIAL_REPO" "$PINNED_COMMIT" "$PROFILE" \
    "$PRIOR_ROOT/training_seconds.txt" <<'PY'
import json
import sys
from pathlib import Path

mapping_path, prior_path, output_path = map(Path, sys.argv[1:4])
official_repo = Path(sys.argv[4]).resolve()
payload = {
    "schema": "lafgs_off_the_shelf_prior_protocol",
    "version": 1,
    "profile": sys.argv[6],
    "official_repository": str(official_repo),
    "official_commit": sys.argv[5],
    "mapping_input": json.loads(mapping_path.read_text()),
    "rgb_prior": json.loads(prior_path.read_text()),
    "controls": {
        "mapping_rgb_only": True,
        "test_rgb_used": False,
        "rgb_prior_semantic_object_sky_mask_used": False,
        "localization_valid_mask_policy": (
            "object_and_sky_and_distortion_v1_fixed_across_priors"
        ),
        "localization_feature_embedding_used": False,
        "scene_specific_detector_used": False,
        "depth_supervision_used": False,
        "task_specific_gaussian_pruning_used": False,
        "lafgs_gradient_to_rgb_gaussian": False,
        "resolution": "1920x1080",
        "iterations": 30000,
        "sh_degree": 3,
    },
    "training_seconds": int(Path(sys.argv[7]).read_text().strip()),
}
output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
print(json.dumps(payload, indent=2, sort_keys=True))
PY
}

evaluate_prior_quality() {
  export_prior
  local evaluation_source="$EXPERIMENT_ROOT/datasets/${SCENE}_official_eval_undistorted"
  require_file "$evaluation_source/evaluation_scene_manifest.json"
  run_logged prior_quality \
    env CUDA_VISIBLE_DEVICES="$GPU" STDLOC_CAMERA_LOADER_WORKERS=0 \
    PYTHONPATH="$REPO_ROOT" \
    "$PYTHON" "$REPO_ROOT/scripts/evaluate_offtheshelf_gaussian_prior.py" \
    --model-root "$MODEL_ROOT" --source "$evaluation_source" \
    --gaussian-type "$GAUSSIAN_TYPE" --sh-degree 3 \
    --iteration "$ITERATIONS" --output "$PRIOR_ROOT/prior_quality.json"
}

run_lafgs() {
  export_prior
  # Rendered depth/alpha are prior-dependent. Keep both cache profiles inside
  # this scene/prior run so no MAtCha or sibling-prior artifact can leak in.
  local lafgs_profile_root="$EXPERIMENT_ROOT/$LAFGS_NAMESPACE/$PROFILE"
  local run_root="$lafgs_profile_root/$SCENE/runs/frozen_v1"
  local full_cache="$run_root/query_cache_native_fullres_k2048.pt"
  local sparse_cache="$run_root/query_cache_native_sparse_teacher.pt"
  env \
    LAFGS_V1_MODEL_ROOT_OVERRIDE="$MODEL_ROOT" \
    LAFGS_V1_PRIOR_PROFILE_OVERRIDE="$PROFILE" \
    LAFGS_V1_CONFIG_OVERRIDE="$REPO_ROOT/configs/lafgs_paper_mainline.yaml" \
    LAFGS_V1_MULTISCENE_ROOT="$lafgs_profile_root" \
    LAFGS_V1_QUERY_CACHE_OVERRIDE="$full_cache" \
    LAFGS_V1_SPARSE_QUERY_CACHE_OVERRIDE="$sparse_cache" \
    LAFGS_EVAL_VARIANTS_OVERRIDE="A0_bootstrap A1_reconstructed" \
    bash "$REPO_ROOT/scripts/run_lafgs_v1_frozen_multiscene.sh" \
      "$SCENE" "$GPU" eval
}

status() {
  printf 'dataset=%s\n' "$DATASET"
  printf 'native_ply=%s\n' "$NATIVE_MODEL/point_cloud/iteration_${ITERATIONS}/point_cloud.ply"
  printf 'model_root=%s\n' "$MODEL_ROOT"
  printf 'lafgs_root=%s\n' "$EXPERIMENT_ROOT/$LAFGS_NAMESPACE/$PROFILE/$SCENE"
  for path in \
    "$DATASET/mapping_only_manifest.json" \
    "$NATIVE_MODEL/point_cloud/iteration_${ITERATIONS}/point_cloud.ply" \
    "$MODEL_ROOT/rgb_prior_manifest.json" \
    "$EXPERIMENT_ROOT/$LAFGS_NAMESPACE/$PROFILE/$SCENE/frozen_results.json"; do
    if [[ -f "$path" ]]; then
      printf 'ready %s\n' "$path"
    else
      printf 'missing %s\n' "$path"
    fi
  done
}

case "$MODE" in
  train) train_prior ;;
  export) export_prior ;;
  quality) evaluate_prior_quality ;;
  lafgs|eval) run_lafgs ;;
  all) evaluate_prior_quality; run_lafgs ;;
  status) status ;;
esac
