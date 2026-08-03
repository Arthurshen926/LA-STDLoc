#!/usr/bin/env bash
set -euo pipefail

# Pure feed-forward AnySplat RGB prior -> frozen LaFGS paper mainline.
# Every Sim(3) is fitted with mapping cameras only. No post optimization or
# localization gradient is allowed to modify the RGB Gaussian prior.

if [[ $# -ne 2 ]]; then
  echo "Usage: bash $0 <OldHospital|KingsCollege> <prepare|infer|align|quality|lafgs|eval|all|status>" >&2
  exit 2
fi

SCENE="$1"
MODE="$2"
case "$SCENE" in
  OldHospital|KingsCollege) ;;
  *) echo "Unsupported AnySplat scene: $SCENE" >&2; exit 2 ;;
esac
case "$MODE" in
  prepare|infer|align|quality|lafgs|eval|all|status) ;;
  *) echo "Unsupported mode: $MODE" >&2; exit 2 ;;
esac

GPU="${LAFGS_ANYSPLAT_GPU:-2}"
if [[ "$GPU" != "2" ]]; then
  echo "AnySplat prior experiment is restricted to GPU2" >&2
  exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STDLOC_PYTHON="${PYTHON:-/root/miniconda3/envs/cybersim_agent/bin/python}"
ANYSPLAT_PYTHON="${ANYSPLAT_PYTHON:-/root/miniconda3/envs/anysplat/bin/python}"
ANYSPLAT_REPO="${ANYSPLAT_REPO:-/mnt/pool/sqy/third_party_lafgs_priors/AnySplat}"
ANYSPLAT_COMMIT="5f5e208a7dd57d52e43ea0d553a95eab526e8775"
ANYSPLAT_MODEL="lhjiang/anysplat"
ANYSPLAT_MODEL_REVISION="d2e8c343672646041ad4ea518184968f94362f01"
HF_HOME="${HF_HOME:-/mnt/pool/sqy/huggingface_cache}"
CAMBRIDGE_ROOT="${CAMBRIDGE_DATA_ROOT:-/mnt/pool/sqy/Cambridge_stdloc}"
EXPERIMENT_ROOT="${LAFGS_OFFTHESHELF_ROOT:-/mnt/pool/sqy/stdloc_lafgs_offtheshelf_prior_20260802}"
LAFGS_NAMESPACE="${LAFGS_OFFTHESHELF_LAFGS_NAMESPACE:-lafgs_strict_v2}"
PRIOR_TAG="${LAFGS_ANYSPLAT_PROFILE:-anysplat_ff}"
DATASET="$EXPERIMENT_ROOT/datasets/${SCENE}_mapping_only_undistorted"
EVAL_DATASET="$EXPERIMENT_ROOT/datasets/${SCENE}_official_eval_undistorted"
PRIOR_ROOT="$EXPERIMENT_ROOT/priors/$SCENE/$PRIOR_TAG"
WINDOW_ROOT="$PRIOR_ROOT/feedforward_input"
RAW_ROOT="$PRIOR_ROOT/raw_windows"
ALIGNED_PLY="$PRIOR_ROOT/aligned/point_cloud.ply"
ALIGNMENT_REPORT="$PRIOR_ROOT/alignment_report.json"
MODEL_ROOT="$PRIOR_ROOT/stdloc_model"
LOG_ROOT="$PRIOR_ROOT/logs"
VIEWS_PER_TRAJECTORY="${LAFGS_ANYSPLAT_VIEWS_PER_TRAJECTORY:-24}"
TRAJECTORY_SEGMENT_SIZE="${LAFGS_ANYSPLAT_TRAJECTORY_SEGMENT_SIZE:-96}"
REQUIRE_COMPLETE_MAPPING_INPUT="${LAFGS_ANYSPLAT_REQUIRE_COMPLETE_MAPPING_INPUT:-0}"
MAXIMUM_PRIMITIVES="${LAFGS_ANYSPLAT_MAXIMUM_PRIMITIVES:-3000000}"
PRESELECTION_MULTIPLIER="${LAFGS_ANYSPLAT_PRESELECTION_MULTIPLIER:-1}"
ITERATION=30000

export CUDA_HOME=/usr/local/cuda-11.8
export PATH="/root/miniconda3/envs/cybersim_agent/bin:$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="/root/miniconda3/envs/cybersim_agent/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONHASHSEED=2026
mkdir -p "$PRIOR_ROOT" "$LOG_ROOT"

require_file() {
  [[ -f "$1" ]] || { echo "Required file is missing: $1" >&2; exit 1; }
}

run_logged() {
  local name="$1"
  shift
  printf '%q ' "$@" > "$LOG_ROOT/${name}.command.sh"
  printf '\n' >> "$LOG_ROOT/${name}.command.sh"
  "$@" 2>&1 | tee "$LOG_ROOT/${name}.log"
}

verify_protocol() {
  require_file "$DATASET/mapping_only_manifest.json"
  require_file "$ANYSPLAT_PYTHON"
  local actual_commit
  actual_commit="$(git -C "$ANYSPLAT_REPO" rev-parse HEAD)"
  [[ "$actual_commit" == "$ANYSPLAT_COMMIT" ]] || {
    echo "AnySplat repository commit mismatch: $actual_commit" >&2
    exit 1
  }
  "$STDLOC_PYTHON" - "$DATASET/mapping_only_manifest.json" <<'PY'
import json
import sys
manifest = json.load(open(sys.argv[1]))
assert manifest["semantic_mask_used"] is False
assert manifest["undistortion_used"] is True
assert manifest["excluded_test_image_count"] > 0
PY
}

prepare_windows() {
  verify_protocol
  if [[ ! -f "$WINDOW_ROOT/windows_manifest.json" ]]; then
    local coverage_args=()
    if [[ "$REQUIRE_COMPLETE_MAPPING_INPUT" == "1" ]]; then
      coverage_args+=(--complete-coverage)
    fi
    run_logged prepare_windows env PYTHONPATH="$REPO_ROOT" \
      "$STDLOC_PYTHON" "$REPO_ROOT/scripts/prepare_anysplat_cambridge_windows.py" \
      --dataset "$DATASET" --output "$WINDOW_ROOT" \
      --views-per-trajectory "$VIEWS_PER_TRAJECTORY" \
      --trajectory-segment-size "$TRAJECTORY_SEGMENT_SIZE" \
      "${coverage_args[@]}"
  fi
  require_file "$WINDOW_ROOT/windows_manifest.json"
  if [[ "$REQUIRE_COMPLETE_MAPPING_INPUT" == "1" ]]; then
    "$STDLOC_PYTHON" - "$WINDOW_ROOT/windows_manifest.json" <<'PY'
import json
import sys
manifest = json.load(open(sys.argv[1]))
assert manifest["selected_image_count"] == manifest["mapping_image_count"], manifest
selected = [
    name
    for window in manifest["windows"]
    for name in window["image_names"]
]
assert len(selected) == len(set(selected)), "mapping images must enter exactly one FF window"
assert min(window["selected_view_count"] for window in manifest["windows"]) >= 3
PY
  fi
}

run_feedforward() {
  prepare_windows
  if [[ ! -f "$RAW_ROOT/feedforward_summary.json" ]]; then
    run_logged feedforward env -u LD_LIBRARY_PATH \
      CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH="$REPO_ROOT" HF_HOME="$HF_HOME" \
      "$ANYSPLAT_PYTHON" "$REPO_ROOT/scripts/run_anysplat_feedforward.py" \
      --anysplat-repo "$ANYSPLAT_REPO" \
      --windows-manifest "$WINDOW_ROOT/windows_manifest.json" \
      --output "$RAW_ROOT" --model-id "$ANYSPLAT_MODEL" \
      --revision "$ANYSPLAT_MODEL_REVISION"
  fi
  require_file "$RAW_ROOT/feedforward_summary.json"
}

align_prior() {
  run_feedforward
  if [[ ! -f "$ALIGNMENT_REPORT" || ! -f "$ALIGNED_PLY" ]]; then
    run_logged align env -u LD_LIBRARY_PATH PYTHONPATH="$REPO_ROOT" \
      "$ANYSPLAT_PYTHON" "$REPO_ROOT/scripts/align_anysplat_feedforward_prior.py" \
      --dataset "$DATASET" \
      --windows-manifest "$WINDOW_ROOT/windows_manifest.json" \
      --raw-root "$RAW_ROOT" --output-ply "$ALIGNED_PLY" \
      --output-report "$ALIGNMENT_REPORT" \
      --maximum-primitives "$MAXIMUM_PRIMITIVES" \
      --preselection-multiplier "$PRESELECTION_MULTIPLIER"
  fi
  require_file "$ALIGNMENT_REPORT"
  require_file "$ALIGNED_PLY"
  local output_ply="$MODEL_ROOT/point_cloud/iteration_${ITERATION}/point_cloud.ply"
  if [[ ! -f "$MODEL_ROOT/rgb_prior_manifest.json" || ! -f "$output_ply" ]]; then
    run_logged export_prior "$STDLOC_PYTHON" \
      "$REPO_ROOT/scripts/export_rgb_gaussian_prior.py" \
      --input_ply "$ALIGNED_PLY" --output_model "$MODEL_ROOT" \
      --gaussian_type 3dgs --sh_degree 0 \
      --source_path "$CAMBRIDGE_ROOT/$SCENE" --images processed \
      --longest_edge 0 --iteration "$ITERATION" --prior_kind rgb_only
  fi
  require_file "$MODEL_ROOT/rgb_prior_manifest.json"
  "$STDLOC_PYTHON" - \
    "$DATASET/mapping_only_manifest.json" \
    "$WINDOW_ROOT/windows_manifest.json" \
    "$RAW_ROOT/feedforward_summary.json" \
    "$ALIGNMENT_REPORT" "$MODEL_ROOT/rgb_prior_manifest.json" \
    "$PRIOR_ROOT/offtheshelf_prior_protocol.json" \
    "$ANYSPLAT_REPO" "$ANYSPLAT_COMMIT" \
    "$LOG_ROOT/align.command.sh" "$PRIOR_TAG" <<'PY'
import json
import sys
from pathlib import Path

mapping, windows, feedforward, alignment, prior, output = map(Path, sys.argv[1:7])
feedforward_payload = json.loads(feedforward.read_text())
feedforward_seconds = float(feedforward_payload["inference_seconds"])
alignment_payload = json.loads(alignment.read_text())
alignment_seconds = alignment_payload.get("alignment_and_fusion_seconds")
alignment_timing_source = "perf_counter"
if alignment_seconds is None:
    align_command = Path(sys.argv[9])
    if align_command.is_file():
        alignment_seconds = max(
            0.0, alignment.stat().st_mtime - align_command.stat().st_mtime
        )
        alignment_timing_source = "log_mtime_estimate"
    else:
        alignment_timing_source = None
model_load_seconds = float(feedforward_payload["model_load_seconds"])
payload = {
    "schema": "lafgs_off_the_shelf_prior_protocol",
    "version": 2,
    "profile": sys.argv[10],
    "official_repository": str(Path(sys.argv[7]).resolve()),
    "official_commit": sys.argv[8],
    "model_id": feedforward_payload["model_id"],
    "model_revision": feedforward_payload["model_revision"],
    "mapping_input": json.loads(mapping.read_text()),
    "feedforward_input": json.loads(windows.read_text()),
    "feedforward_summary": feedforward_payload,
    "mapping_only_sim3_alignment": alignment_payload,
    "rgb_prior": json.loads(prior.read_text()),
    "controls": {
        "mapping_rgb_only": True,
        "test_rgb_used": False,
        "test_pose_used_for_alignment": False,
        "post_optimization_used": False,
        "rgb_prior_semantic_object_sky_mask_used": False,
        "localization_valid_mask_policy": "object_and_sky_and_distortion_v1_fixed_across_priors",
        "localization_feature_embedding_used": False,
        "scene_specific_detector_used": False,
        "depth_supervision_used": False,
        "task_specific_gaussian_pruning_used": False,
        "lafgs_gradient_to_rgb_gaussian": False,
        "downstream_mapping_images": "complete Cambridge mapping split",
        "prior_uses_complete_mapping_split": (
            json.loads(windows.read_text())["selected_image_count"]
            == json.loads(mapping.read_text())["mapping_image_count"]
        ),
        "sh_degree": 0,
    },
    "training_seconds": 0,
    "feedforward_seconds": feedforward_seconds,
    "model_load_seconds": model_load_seconds,
    "alignment_and_fusion_seconds": alignment_seconds,
    "alignment_timing_source": alignment_timing_source,
    "total_prior_seconds": (
        model_load_seconds + feedforward_seconds + float(alignment_seconds)
        if alignment_seconds is not None
        else None
    ),
}
output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
print(json.dumps({
    "profile": payload["profile"],
    "mapping_image_count": payload["mapping_input"]["mapping_image_count"],
    "feedforward_image_count": payload["feedforward_input"]["selected_image_count"],
    "primitive_count": payload["rgb_prior"]["primitive_count"],
    "model_load_seconds": model_load_seconds,
    "feedforward_seconds": feedforward_seconds,
    "alignment_and_fusion_seconds": alignment_seconds,
    "alignment_timing_source": alignment_timing_source,
    "total_prior_seconds": payload["total_prior_seconds"],
}, indent=2, sort_keys=True))
PY
}

evaluate_quality() {
  align_prior
  require_file "$EVAL_DATASET/evaluation_scene_manifest.json"
  if [[ ! -f "$PRIOR_ROOT/prior_quality.json" ]]; then
    run_logged prior_quality env CUDA_VISIBLE_DEVICES="$GPU" \
      STDLOC_CAMERA_LOADER_WORKERS=0 PYTHONPATH="$REPO_ROOT" \
      "$STDLOC_PYTHON" "$REPO_ROOT/scripts/evaluate_offtheshelf_gaussian_prior.py" \
      --model-root "$MODEL_ROOT" --source "$EVAL_DATASET" \
      --gaussian-type 3dgs --sh-degree 0 --iteration "$ITERATION" \
      --output "$PRIOR_ROOT/prior_quality.json"
  fi
}

run_lafgs() {
  align_prior
  local lafgs_profile_root="$EXPERIMENT_ROOT/$LAFGS_NAMESPACE/$PRIOR_TAG"
  local run_root="$lafgs_profile_root/$SCENE/runs/frozen_v1"
  local full_cache="$run_root/query_cache_native_fullres_k2048.pt"
  local sparse_cache="$run_root/query_cache_native_sparse_teacher.pt"
  run_mainline_stage() {
    local stage="$1"
    env LAFGS_V1_MODEL_ROOT_OVERRIDE="$MODEL_ROOT" \
      LAFGS_V1_PRIOR_PROFILE_OVERRIDE=anysplat_ff \
      LAFGS_V1_CONFIG_OVERRIDE="$REPO_ROOT/configs/lafgs_paper_mainline.yaml" \
      LAFGS_V1_MULTISCENE_ROOT="$lafgs_profile_root" \
      LAFGS_V1_QUERY_CACHE_OVERRIDE="$full_cache" \
      LAFGS_V1_SPARSE_QUERY_CACHE_OVERRIDE="$sparse_cache" \
      LAFGS_EVAL_VARIANTS_OVERRIDE="A0_bootstrap A1_reconstructed" \
      bash "$REPO_ROOT/scripts/run_lafgs_v1_frozen_multiscene.sh" \
        "$SCENE" "$GPU" "$stage"
  }
  # The frozen mainline's eval mode deliberately consumes existing immutable
  # artifacts only. Build A0/A1 first when adapting a new RGB prior.
  run_mainline_stage reconstruct
  run_mainline_stage eval
}

status() {
  printf 'dataset=%s\nprior_root=%s\nmodel_root=%s\n' "$DATASET" "$PRIOR_ROOT" "$MODEL_ROOT"
  for path in "$WINDOW_ROOT/windows_manifest.json" \
    "$RAW_ROOT/feedforward_summary.json" "$ALIGNMENT_REPORT" \
    "$MODEL_ROOT/rgb_prior_manifest.json" "$PRIOR_ROOT/prior_quality.json" \
    "$EXPERIMENT_ROOT/$LAFGS_NAMESPACE/$PRIOR_TAG/$SCENE/frozen_results.json"; do
    [[ -f "$path" ]] && printf 'ready %s\n' "$path" || printf 'missing %s\n' "$path"
  done
}

case "$MODE" in
  prepare) prepare_windows ;;
  infer) run_feedforward ;;
  align) align_prior ;;
  quality) evaluate_quality ;;
  lafgs|eval) run_lafgs ;;
  all) evaluate_quality; run_lafgs ;;
  status) status ;;
esac
