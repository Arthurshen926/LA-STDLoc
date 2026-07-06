#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT=${DATA_ROOT:-/mnt/pool/sqy/Cambridge_stdloc}
BASELINE_ROOT=${BASELINE_ROOT:-/mnt/pool/sqy/stdloc_la_full_runs}
OUT_ROOT=${OUT_ROOT:-/mnt/pool/sqy/stdloc_la_pseudo_query_runs}
SCENES=${SCENES:-ShopFacade OldHospital}
CFG=${CFG:-configs/stdloc_cambridge.yaml}
BASELINE_ITERS=${BASELINE_ITERS:-30000}
GPU=${GPU:-0}
NERFBASELINES_BIN=${NERFBASELINES_BIN:-/root/miniconda3/envs/nb/bin/nerfbaselines}
NERFBASELINES_BACKEND=${NERFBASELINES_BACKEND:-conda}
NERFBASELINES_DATA_ROOT=${NERFBASELINES_DATA_ROOT:-/mnt/pool/sqy/stdloc_la_nerfbaselines_datasets}
NERFBASELINES_IMAGES_SOURCE=${NERFBASELINES_IMAGES_SOURCE:-.}
NERFBASELINES_IMAGE_DOWNSCALE_FACTOR=${NERFBASELINES_IMAGE_DOWNSCALE_FACTOR:-1.0}
NERFBASELINES_MAX_IMAGE_WIDTH=${NERFBASELINES_MAX_IMAGE_WIDTH:-0}
RUN_RGB_TEACHER_TRAIN=${RUN_RGB_TEACHER_TRAIN:-0}
RGB_TEACHER_TRAIN_STEPS=${RGB_TEACHER_TRAIN_STEPS:-0}
RGB_TEACHER_LOGGER=${RGB_TEACHER_LOGGER:-tensorboard}
RGB_TEACHER_SAVE_ITERS=${RGB_TEACHER_SAVE_ITERS:-$RGB_TEACHER_TRAIN_STEPS}
RGB_TEACHER_EVAL_FEW_ITERS=${RGB_TEACHER_EVAL_FEW_ITERS:-$RGB_TEACHER_TRAIN_STEPS}
RGB_TEACHER_EVAL_ALL_ITERS=${RGB_TEACHER_EVAL_ALL_ITERS:-999999}
RGB_TEACHER_RENDER_OUTPUT_NAMES=${RGB_TEACHER_RENDER_OUTPUT_NAMES:-color}
RGB_TEACHER_DISABLE_OUTPUT_ARTIFACT=${RGB_TEACHER_DISABLE_OUTPUT_ARTIFACT:-0}
RGB_TEACHER_WILDGAUSSIANS_PRESET=${RGB_TEACHER_WILDGAUSSIANS_PRESET:-}
RGB_TEACHER_WILDGAUSSIANS_SET=${RGB_TEACHER_WILDGAUSSIANS_SET:-}
WILDGAUSSIANS_RENDER_SCALE=${WILDGAUSSIANS_RENDER_SCALE:-0.5}
WILDGAUSSIANS_RENDER_RESOLUTION=${WILDGAUSSIANS_RENDER_RESOLUTION:-}
WILDGAUSSIANS_APPEARANCE_MODE=${WILDGAUSSIANS_APPEARANCE_MODE:-auto}
MATCHA_ROOT=${MATCHA_ROOT:-/root/MAtCha}
MATCHA_RUNS_ROOT=${MATCHA_RUNS_ROOT:-/root/MAtCha/output_cambridge/runs}
MATCHA_MODEL_PATH=${MATCHA_MODEL_PATH:-}
MATCHA_PYTHON_DEFAULT=${MATCHA_PYTHON_DEFAULT:-/root/miniconda3/envs/cybersim_agent/bin/python}
MATCHA_PYTHON=${MATCHA_PYTHON:-}
MATCHA_ITERATION=${MATCHA_ITERATION:-30000}
MATCHA_RENDER_RESOLUTION=${MATCHA_RENDER_RESOLUTION:-}
SYNTHETIC_APPEARANCE_STRATEGY=${SYNTHETIC_APPEARANCE_STRATEGY:-nearest}

LA_ENABLE_SYNTHETIC=${LA_ENABLE_SYNTHETIC:-0}
if [[ -z "${SYNTHETIC_COUNT+x}" ]]; then
  if [[ "$LA_ENABLE_SYNTHETIC" == "1" ]]; then
    SYNTHETIC_COUNT=16
  else
    SYNTHETIC_COUNT=0
  fi
fi
SYNTHETIC_CANDIDATE_MULTIPLIER=${SYNTHETIC_CANDIDATE_MULTIPLIER:-1}
SYNTHETIC_RENDER_COUNT=$((SYNTHETIC_COUNT * SYNTHETIC_CANDIDATE_MULTIPLIER))
SYNTHETIC_SEED=${SYNTHETIC_SEED:-2026}
PSEUDO_QUERY_POSE_SAMPLER=${PSEUDO_QUERY_POSE_SAMPLER:-spatial_offset}
SYNTHETIC_SPATIAL_MIN_OFFSET_RATIO=${SYNTHETIC_SPATIAL_MIN_OFFSET_RATIO:-1.0}
SYNTHETIC_SPATIAL_MAX_OFFSET_RATIO=${SYNTHETIC_SPATIAL_MAX_OFFSET_RATIO:-3.0}
SYNTHETIC_SPATIAL_YAW_DEG=${SYNTHETIC_SPATIAL_YAW_DEG:-20.0}
SYNTHETIC_SPATIAL_HEIGHT_OFFSET_RATIO=${SYNTHETIC_SPATIAL_HEIGHT_OFFSET_RATIO:-0.15}
RENDER_SYNTHETIC_BACKEND=${RENDER_SYNTHETIC_BACKEND:-matcha}
REPAIR_SYNTHETIC_ARTIFACTS=${REPAIR_SYNTHETIC_ARTIFACTS:-0}
SYNTHETIC_ACCEPT_SCORE=${SYNTHETIC_ACCEPT_SCORE:-0.65}
SYNTHETIC_QUALITY_GATE=${SYNTHETIC_QUALITY_GATE:-0}
SYNTHETIC_QA_MAX_MEAN=${SYNTHETIC_QA_MAX_MEAN:-0.60}
SYNTHETIC_QA_MAX_P95=${SYNTHETIC_QA_MAX_P95:--1.0}
SYNTHETIC_QA_MAX_MILD_FRAC=${SYNTHETIC_QA_MAX_MILD_FRAC:-0.85}
SYNTHETIC_QA_MAX_SEVERE_FRAC=${SYNTHETIC_QA_MAX_SEVERE_FRAC:-0.58}
SYNTHETIC_QA_MAX_LOW_DETAIL_MEAN=${SYNTHETIC_QA_MAX_LOW_DETAIL_MEAN:-0.60}
SYNTHETIC_REPAIR_THRESHOLD=${SYNTHETIC_REPAIR_THRESHOLD:-0.35}
MIN_OPACITY_MULTIPLIER=${MIN_OPACITY_MULTIPLIER:-0.15}

RUN_PSEUDO_QUERY_MANIFEST=${RUN_PSEUDO_QUERY_MANIFEST:-1}
RUN_TEACHER_CACHE=${RUN_TEACHER_CACHE:-1}
RUN_TEACHER_CACHE_AUDIT=${RUN_TEACHER_CACHE_AUDIT:-1}
TEACHER_CACHE_MAX=${TEACHER_CACHE_MAX:-0}
if [[ -z "${TEACHER_CACHE_SOURCES+x}" ]]; then
  if [[ "$LA_ENABLE_SYNTHETIC" == "1" ]]; then
    TEACHER_CACHE_SOURCES=train_rgb,synthetic_rgb
  else
    TEACHER_CACHE_SOURCES=train_rgb
  fi
fi
TEACHER_CACHE_SPARSE_VALID_MASK=${TEACHER_CACHE_SPARSE_VALID_MASK:-$LA_ENABLE_SYNTHETIC}
TEACHER_CACHE_SPARSE_VALID_MASK_MODE=${TEACHER_CACHE_SPARSE_VALID_MASK_MODE:-no_reference}
TEACHER_CACHE_SPARSE_VALID_MASK_SOURCES=${TEACHER_CACHE_SPARSE_VALID_MASK_SOURCES:-synthetic_rgb}
TEACHER_CACHE_SPARSE_VALID_MASK_MIN_FRACTION=${TEACHER_CACHE_SPARSE_VALID_MASK_MIN_FRACTION:-0.5}
TEACHER_CACHE_SPARSE_VALID_MASK_CANDIDATE_MULTIPLIER=${TEACHER_CACHE_SPARSE_VALID_MASK_CANDIDATE_MULTIPLIER:-2.0}
TEACHER_CACHE_SPARSE_SUPPORT_SCORE_WEIGHT=${TEACHER_CACHE_SPARSE_SUPPORT_SCORE_WEIGHT:-0.5}
TEACHER_CACHE_SPARSE_SUPPORT_SCORE_MIN_MULTIPLIER=${TEACHER_CACHE_SPARSE_SUPPORT_SCORE_MIN_MULTIPLIER:-0.75}
TEACHER_CACHE_NO_REFERENCE_IMAGE_SCALE=${TEACHER_CACHE_NO_REFERENCE_IMAGE_SCALE:-0.25}
TEACHER_CACHE_NO_REFERENCE_SUPPORT_THRESHOLD=${TEACHER_CACHE_NO_REFERENCE_SUPPORT_THRESHOLD:-0.22}
TEACHER_CACHE_NO_REFERENCE_SUPPORT_DILATE_RADIUS=${TEACHER_CACHE_NO_REFERENCE_SUPPORT_DILATE_RADIUS:-5}
TEACHER_CACHE_NO_REFERENCE_SUPPORT_MIN_AREA=${TEACHER_CACHE_NO_REFERENCE_SUPPORT_MIN_AREA:-24}
TEACHER_CACHE_NO_REFERENCE_INVALID_MIN_AREA=${TEACHER_CACHE_NO_REFERENCE_INVALID_MIN_AREA:-96}
RUN_PSEUDO_QUERY_GATE=${RUN_PSEUDO_QUERY_GATE:-0}
RUN_PSEUDO_QUERY_SELECT=${RUN_PSEUDO_QUERY_SELECT:-0}

RUN_TRAIN=${RUN_TRAIN:-1}
LA_TRAIN_MODE=${LA_TRAIN_MODE:-adapt}
LA_ADAPT_STEPS=${LA_ADAPT_STEPS:-${TRAIN_STEPS:-100}}
TRAIN_STEPS="$LA_ADAPT_STEPS"
LA_LOC_START_ITER=${LA_LOC_START_ITER:-1}
LA_LANDMARK_PATH=${LA_LANDMARK_PATH:-}
TRAIN_SEED=${TRAIN_SEED:-0}
if [[ -z "${PSEUDO_QUERY_SOURCES+x}" ]]; then
  if [[ "$LA_ENABLE_SYNTHETIC" == "1" ]]; then
    PSEUDO_QUERY_SOURCES=train_rgb,synthetic_rgb
  else
    PSEUDO_QUERY_SOURCES=train_rgb
  fi
fi
PSEUDO_QUERY_MAX_SYNTHETIC=${PSEUDO_QUERY_MAX_SYNTHETIC:-0}
PSEUDO_QUERY_SELECT_MAX_SYNTHETIC=${PSEUDO_QUERY_SELECT_MAX_SYNTHETIC:-0}
PSEUDO_QUERY_SELECT_SORT_BY=${PSEUDO_QUERY_SELECT_SORT_BY:-artifact}
PSEUDO_QUERY_MIN_SUPPORT_FRAC=${PSEUDO_QUERY_MIN_SUPPORT_FRAC:-0.0}
PSEUDO_QUERY_MIN_SUPPORT_SCORE=${PSEUDO_QUERY_MIN_SUPPORT_SCORE:--1.0}
PSEUDO_QUERY_REAL_WEIGHT=${PSEUDO_QUERY_REAL_WEIGHT:-2.0}
PSEUDO_QUERY_SYNTHETIC_WEIGHT=${PSEUDO_QUERY_SYNTHETIC_WEIGHT:-1.0}
PSEUDO_QUERY_SAMPLING_MODE=${PSEUDO_QUERY_SAMPLING_MODE:-record_proportional}
PSEUDO_QUERY_RELIABILITY_MODE=${PSEUDO_QUERY_RELIABILITY_MODE:-none}
PSEUDO_QUERY_RELIABILITY_LOSS_MODE=${PSEUDO_QUERY_RELIABILITY_LOSS_MODE:-none}
PSEUDO_QUERY_STAGE_OBJECTIVE_MODE=${PSEUDO_QUERY_STAGE_OBJECTIVE_MODE:-none}
PSEUDO_QUERY_STAGE_STATS_POLICY=${PSEUDO_QUERY_STAGE_STATS_POLICY:-hard}
PSEUDO_QUERY_RELIABILITY_MIN_WEIGHT=${PSEUDO_QUERY_RELIABILITY_MIN_WEIGHT:-0.20}
PSEUDO_QUERY_RELIABILITY_REAL_MIN_WEIGHT=${PSEUDO_QUERY_RELIABILITY_REAL_MIN_WEIGHT:-0.50}
PSEUDO_QUERY_RELIABILITY_SYNTHETIC_MIN_WEIGHT=${PSEUDO_QUERY_RELIABILITY_SYNTHETIC_MIN_WEIGHT:-0.25}
PSEUDO_QUERY_RELIABILITY_MEMORY_MIN_WEIGHT=${PSEUDO_QUERY_RELIABILITY_MEMORY_MIN_WEIGHT:-0.75}
PSEUDO_QUERY_RELIABILITY_STATS_MIN_WEIGHT=${PSEUDO_QUERY_RELIABILITY_STATS_MIN_WEIGHT:-}
PSEUDO_QUERY_RELIABILITY_ERROR_SCALE=${PSEUDO_QUERY_RELIABILITY_ERROR_SCALE:-2.0}
PSEUDO_QUERY_RELIABILITY_INLIER_POWER=${PSEUDO_QUERY_RELIABILITY_INLIER_POWER:-0.5}
PSEUDO_QUERY_RELIABILITY_TEACHER_OK_WEIGHT=${PSEUDO_QUERY_RELIABILITY_TEACHER_OK_WEIGHT:-1.0}
PSEUDO_QUERY_RELIABILITY_DENSE_IMPROVES_WEIGHT=${PSEUDO_QUERY_RELIABILITY_DENSE_IMPROVES_WEIGHT:-0.90}
PSEUDO_QUERY_RELIABILITY_MIXED_WEIGHT=${PSEUDO_QUERY_RELIABILITY_MIXED_WEIGHT:-0.70}
PSEUDO_QUERY_RELIABILITY_DENSE_RESCUES_WEIGHT=${PSEUDO_QUERY_RELIABILITY_DENSE_RESCUES_WEIGHT:-0.55}
PSEUDO_QUERY_RELIABILITY_SPARSE_FAILURE_WEIGHT=${PSEUDO_QUERY_RELIABILITY_SPARSE_FAILURE_WEIGHT:-0.30}
PSEUDO_QUERY_RELIABILITY_DENSE_REGRESSION_WEIGHT=${PSEUDO_QUERY_RELIABILITY_DENSE_REGRESSION_WEIGHT:-0.35}
PSEUDO_QUERY_RELIABILITY_UNKNOWN_WEIGHT=${PSEUDO_QUERY_RELIABILITY_UNKNOWN_WEIGHT:-0.60}
PSEUDO_QUERY_EXCLUDE_SPARSE_FAILURE_STAGES=${PSEUDO_QUERY_EXCLUDE_SPARSE_FAILURE_STAGES:-0}
PSEUDO_QUERY_FILTER_TEACHER_CACHE=${PSEUDO_QUERY_FILTER_TEACHER_CACHE:-0}
PSEUDO_QUERY_REQUIRE_TEACHER_CACHE=${PSEUDO_QUERY_REQUIRE_TEACHER_CACHE:-1}
PSEUDO_QUERY_ENABLE_TEACHER_GATE=${PSEUDO_QUERY_ENABLE_TEACHER_GATE:-0}
PSEUDO_QUERY_TEACHER_MAX_SPARSE_TE=${PSEUDO_QUERY_TEACHER_MAX_SPARSE_TE:-100.0}
PSEUDO_QUERY_TEACHER_MAX_DENSE_TE=${PSEUDO_QUERY_TEACHER_MAX_DENSE_TE:-100.0}
PSEUDO_QUERY_TEACHER_ALLOWED_STAGES=${PSEUDO_QUERY_TEACHER_ALLOWED_STAGES:-}
PSEUDO_QUERY_TEACHER_GATE_SOURCES=${PSEUDO_QUERY_TEACHER_GATE_SOURCES:-$TEACHER_CACHE_SOURCES}
PSEUDO_QUERY_NO_REFERENCE_REGION_WEIGHT=${PSEUDO_QUERY_NO_REFERENCE_REGION_WEIGHT:-$LA_ENABLE_SYNTHETIC}
PSEUDO_QUERY_NO_REFERENCE_REGION_WEIGHT_SOURCES=${PSEUDO_QUERY_NO_REFERENCE_REGION_WEIGHT_SOURCES:-synthetic_rgb}
PSEUDO_QUERY_NO_REFERENCE_REGION_WEIGHT_MIN=${PSEUDO_QUERY_NO_REFERENCE_REGION_WEIGHT_MIN:-0.25}
PSEUDO_QUERY_NO_REFERENCE_REGION_WEIGHT_SUPPORT_POWER=${PSEUDO_QUERY_NO_REFERENCE_REGION_WEIGHT_SUPPORT_POWER:-1.0}
PSEUDO_QUERY_NO_REFERENCE_REGION_WEIGHT_IMAGE_SCALE=${PSEUDO_QUERY_NO_REFERENCE_REGION_WEIGHT_IMAGE_SCALE:-0.25}
LA_DIRECT_DEPTH_CHECK=${LA_DIRECT_DEPTH_CHECK:-0}
RUN_LA_FRONTEND_REFRESH=${RUN_LA_FRONTEND_REFRESH:-0}
FORCE_LA_FRONTEND_REFRESH=${FORCE_LA_FRONTEND_REFRESH:-0}
LA_DETECTOR_FOLDER=${LA_DETECTOR_FOLDER:-detector_la}
LA_DETECTOR_ITERS=${LA_DETECTOR_ITERS:-30000}
LA_DETECTOR_LANDMARK_NUM=${LA_DETECTOR_LANDMARK_NUM:-16384}
LA_DETECTOR_LANDMARK_K=${LA_DETECTOR_LANDMARK_K:-32}
LA_DETECTOR_SAMPLING_MODE=${LA_DETECTOR_SAMPLING_MODE:-localization_aware}
LA_DETECTOR_TARGET_MODE=${LA_DETECTOR_TARGET_MODE:-soft}
LA_DETECTOR_MIN_LOC_OBSERVATIONS=${LA_DETECTOR_MIN_LOC_OBSERVATIONS:-4}
LA_DETECTOR_UTILITY_WEIGHT=${LA_DETECTOR_UTILITY_WEIGHT:-1.0}
LA_DETECTOR_PNP_VOXEL_SIZE=${LA_DETECTOR_PNP_VOXEL_SIZE:-0.25}
LA_DETECTOR_PNP_MAX_PER_VOXEL=${LA_DETECTOR_PNP_MAX_PER_VOXEL:-8}
LA_DETECTOR_PNP_PRESERVE_RATIO=${LA_DETECTOR_PNP_PRESERVE_RATIO:-0.5}
LA_DETECTOR_SOFT_SIGMA=${LA_DETECTOR_SOFT_SIGMA:-1.5}
LA_BOOTSTRAP_DETECTOR_FOLDER=${LA_BOOTSTRAP_DETECTOR_FOLDER:-detector_bootstrap}
LA_BOOTSTRAP_SAMPLING_MODE=${LA_BOOTSTRAP_SAMPLING_MODE:-baseline}
LA_BOOTSTRAP_LANDMARK_NUM=${LA_BOOTSTRAP_LANDMARK_NUM:-$LA_DETECTOR_LANDMARK_NUM}
LA_BOOTSTRAP_LANDMARK_K=${LA_BOOTSTRAP_LANDMARK_K:-$LA_DETECTOR_LANDMARK_K}
FORCE_LA_BOOTSTRAP_LANDMARKS=${FORCE_LA_BOOTSTRAP_LANDMARKS:-0}
EVAL_SPARSE_DETECT_NUM=${EVAL_SPARSE_DETECT_NUM:-}
EVAL_SPARSE_REPROJECTION_ERROR=${EVAL_SPARSE_REPROJECTION_ERROR:-}

RUN_EVAL=${RUN_EVAL:-0}

if [[ -z "${PYTHON:-}" ]]; then
  if [[ -x /root/miniconda3/envs/ulfloc_repro/bin/python ]]; then
    PYTHON=/root/miniconda3/envs/ulfloc_repro/bin/python
  else
    PYTHON=python
  fi
fi
if [[ -z "$MATCHA_PYTHON" ]]; then
  if [[ -x "$MATCHA_PYTHON_DEFAULT" ]]; then
    MATCHA_PYTHON="$MATCHA_PYTHON_DEFAULT"
  else
    MATCHA_PYTHON="$PYTHON"
  fi
fi
if [[ -z "$MATCHA_RENDER_RESOLUTION" ]]; then
  MATCHA_RENDER_RESOLUTION="${WILDGAUSSIANS_RENDER_RESOLUTION:-}"
fi
if [[ -z "$MATCHA_RENDER_RESOLUTION" ]]; then
  MATCHA_RENDER_RESOLUTION=960x540
fi

require_cuda_toolchain() {
  if [[ ! -x "$CUDA_HOME/bin/nvcc" ]]; then
    echo "Missing CUDA nvcc at $CUDA_HOME/bin/nvcc" >&2
    exit 1
  fi
  if [[ ! -f "$CUDA_HOME/include/cuda_runtime.h" && ! -f "$CUDA_HOME/targets/x86_64-linux/include/cuda_runtime.h" ]]; then
    echo "Missing cuda_runtime.h under CUDA_HOME=$CUDA_HOME" >&2
    exit 1
  fi
}

CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-11.8}
require_cuda_toolchain
export CUDA_HOME
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
MATCHA_ENV_LIB="$(dirname "$(dirname "$MATCHA_PYTHON")")/lib"
if [[ "$RENDER_SYNTHETIC_BACKEND" == "matcha" && -d "$MATCHA_ENV_LIB" ]]; then
  export LD_LIBRARY_PATH="$MATCHA_ENV_LIB:$LD_LIBRARY_PATH"
fi
export PYTHONPATH=${PYTHONPATH:-/root/STDLoc}
export CUDA_VISIBLE_DEVICES="$GPU"

mkdir -p "$OUT_ROOT"

make_eval_cfg() {
  local scene=$1
  local artifact_model=$2
  local out_cfg=$3
  local detector_folder=${4:-detector}
  local detector_iters=${5:-30000}
  local eval_cfg_args=()
  if [[ -n "$EVAL_SPARSE_DETECT_NUM" ]]; then
    eval_cfg_args+=(--detect_num "$EVAL_SPARSE_DETECT_NUM")
  fi
  if [[ -n "$EVAL_SPARSE_REPROJECTION_ERROR" ]]; then
    eval_cfg_args+=(--reprojection_error "$EVAL_SPARSE_REPROJECTION_ERROR")
  fi
  "$PYTHON" scripts/make_stdloc_eval_cfg.py \
    --base_cfg "$CFG" \
    --output "$out_cfg" \
    --artifact_model_path "$artifact_model" \
    --detector_folder "$detector_folder" \
    --detector_iters "$detector_iters" \
    "${eval_cfg_args[@]}"
}

for scene in $SCENES; do
  baseline_model="$BASELINE_ROOT/${scene}_baseline"
  scene_out="$OUT_ROOT/$scene"
  teacher_dir="$scene_out/rgb_teacher"
  nb_dataset="$NERFBASELINES_DATA_ROOT/$scene"
  pseudo_dir="$scene_out/pseudo_query"
  train_start_iter=0
  train_load_iteration_args=()
  train_landmark_path="$LA_LANDMARK_PATH"
  train_needs_bootstrap_landmarks=0
  case "$LA_TRAIN_MODE" in
    adapt)
      train_start_iter=$BASELINE_ITERS
      train_load_iteration_args+=(--load_iteration "$BASELINE_ITERS")
      if [[ -z "$train_landmark_path" ]]; then
        train_landmark_path="$baseline_model/detector/sampled_idx.pkl"
      fi
      train_model="$scene_out/student_${TRAIN_STEPS}step_seed${TRAIN_SEED}"
      ;;
    scratch)
      train_start_iter=0
      if [[ -z "$train_landmark_path" ]]; then
        train_landmark_path="$scene_out/student_scratch_${TRAIN_STEPS}step_seed${TRAIN_SEED}/$LA_BOOTSTRAP_DETECTOR_FOLDER/sampled_idx.pkl"
        train_needs_bootstrap_landmarks=1
      fi
      train_model="$scene_out/student_scratch_${TRAIN_STEPS}step_seed${TRAIN_SEED}"
      if [[ "$RUN_EVAL" == "1" && "$RUN_LA_FRONTEND_REFRESH" != "1" ]]; then
        echo "Scratch LA eval requires RUN_LA_FRONTEND_REFRESH=1; baseline frontend indices are not valid for a scratch map." >&2
        exit 1
      fi
      ;;
    *)
      echo "Unsupported LA_TRAIN_MODE=$LA_TRAIN_MODE; expected adapt or scratch." >&2
      exit 1
      ;;
  esac
  manifest="$pseudo_dir/pseudo_queries.jsonl"
  gated_manifest="$pseudo_dir/pseudo_queries_gated.jsonl"
  selected_manifest="$pseudo_dir/pseudo_queries_selected.jsonl"
  gate_summary="$pseudo_dir/pseudo_query_gate_summary.json"
  selection_summary="$pseudo_dir/pseudo_query_selection_summary.json"
  teacher_cache="$pseudo_dir/pseudo_teacher_cache.pt"
  teacher_summary="$pseudo_dir/pseudo_teacher_cache_summary.json"
  eval_cfg="$scene_out/stdloc_baseline_artifacts.yaml"
  student_eval_cfg="$scene_out/stdloc_student_la_artifacts.yaml"
  final_eval_cfg="$eval_cfg"
  end_iter=$((train_start_iter + LA_ADAPT_STEPS))
  scene_matcha_var="MATCHA_MODEL_PATH_${scene}"
  matcha_model_path="${!scene_matcha_var:-${MATCHA_MODEL_PATH:-$MATCHA_RUNS_ROOT/${scene}_n20_long_masked/free_gaussians}}"
  rgb_teacher_checkpoint="${RGB_TEACHER_CHECKPOINT:-}"
  if [[ -z "$rgb_teacher_checkpoint" && "$RGB_TEACHER_TRAIN_STEPS" != "0" ]]; then
    rgb_teacher_checkpoint="$teacher_dir/$scene/checkpoint-$RGB_TEACHER_TRAIN_STEPS"
  fi
  needs_nerfbaselines=0
  if [[ "$RUN_RGB_TEACHER_TRAIN" == "1" ]]; then
    needs_nerfbaselines=1
  elif (( SYNTHETIC_RENDER_COUNT > 0 )) && [[ "$RENDER_SYNTHETIC_BACKEND" == "wildgaussians" ]]; then
    needs_nerfbaselines=1
  fi

  mkdir -p "$teacher_dir" "$pseudo_dir" "$scene_out"
  make_eval_cfg "$scene" "$baseline_model" "$eval_cfg" "detector" "$BASELINE_ITERS"

  if (( SYNTHETIC_RENDER_COUNT > 0 )) && [[ "$RENDER_SYNTHETIC_BACKEND" == "matcha" && ! -d "$matcha_model_path" ]]; then
    echo "Missing MAtCha model for $scene: $matcha_model_path" >&2
    exit 1
  fi

  if [[ "$needs_nerfbaselines" == "1" ]]; then
    "$PYTHON" scripts/prepare_nerfbaselines_colmap_dataset.py \
      --source_path "$DATA_ROOT/$scene" \
      --output "$nb_dataset" \
      --images_source "$NERFBASELINES_IMAGES_SOURCE" \
      --image_downscale_factor "$NERFBASELINES_IMAGE_DOWNSCALE_FACTOR" \
      --max_image_width "$NERFBASELINES_MAX_IMAGE_WIDTH" \
      --force

    rgb_disable_artifact_args=()
    if [[ "$RGB_TEACHER_DISABLE_OUTPUT_ARTIFACT" == "1" ]]; then
      rgb_disable_artifact_args+=(--disable_output_artifact)
    fi
    rgb_checkpoint_args=()
    if [[ -n "$rgb_teacher_checkpoint" ]]; then
      rgb_checkpoint_args+=(--checkpoint "$rgb_teacher_checkpoint")
    fi
    rgb_preset_args=()
    if [[ -n "$RGB_TEACHER_WILDGAUSSIANS_PRESET" ]]; then
      rgb_preset_args+=(--wildgaussians_preset "$RGB_TEACHER_WILDGAUSSIANS_PRESET")
    fi
    rgb_set_args=()
    if [[ -n "$RGB_TEACHER_WILDGAUSSIANS_SET" ]]; then
      IFS=';' read -r -a rgb_custom_sets <<< "$RGB_TEACHER_WILDGAUSSIANS_SET"
      for item in "${rgb_custom_sets[@]}"; do
        if [[ -n "$item" ]]; then
          rgb_set_args+=(--wildgaussians_set "$item")
        fi
      done
    fi

    "$PYTHON" scripts/prepare_rgb_teacher_manifest.py \
      --scene "$scene" \
      --source_path "$nb_dataset" \
      --output "$teacher_dir/rgb_teacher_manifest.json" \
      --output_root "$teacher_dir" \
      --backend wildgaussians \
      "${rgb_checkpoint_args[@]}" \
      --nerfbaselines_bin "$NERFBASELINES_BIN" \
      --nerfbaselines_backend "$NERFBASELINES_BACKEND" \
      --train_steps "$RGB_TEACHER_TRAIN_STEPS" \
      --logger "$RGB_TEACHER_LOGGER" \
      --save_iters "$RGB_TEACHER_SAVE_ITERS" \
      --eval_few_iters "$RGB_TEACHER_EVAL_FEW_ITERS" \
      --eval_all_iters "$RGB_TEACHER_EVAL_ALL_ITERS" \
      --render_output_names "$RGB_TEACHER_RENDER_OUTPUT_NAMES" \
      --render_resolution "$WILDGAUSSIANS_RENDER_RESOLUTION" \
      "${rgb_preset_args[@]}" \
      "${rgb_set_args[@]}" \
      "${rgb_disable_artifact_args[@]}"

    if [[ "$RUN_RGB_TEACHER_TRAIN" == "1" ]]; then
      mapfile -t rgb_train_sets < <("$PYTHON" - "$teacher_dir/rgb_teacher_manifest.json" <<'PY'
import json
import sys

with open(sys.argv[1]) as f:
    payload = json.load(f)
for item in payload["teachers"][0].get("metrics", {}).get("wildgaussians_sets", []):
    print(item)
PY
)
      rgb_train_args=(
        train
        --method wild-gaussians
        --data "$nb_dataset"
        --output "$teacher_dir/$scene"
        --backend "$NERFBASELINES_BACKEND"
        --logger "$RGB_TEACHER_LOGGER"
      )
      if (( RGB_TEACHER_TRAIN_STEPS > 0 )); then
        rgb_train_args+=(
          --set "iterations=$RGB_TEACHER_TRAIN_STEPS"
          --save-iters "$RGB_TEACHER_SAVE_ITERS"
          --eval-few-iters "$RGB_TEACHER_EVAL_FEW_ITERS"
          --eval-all-iters "$RGB_TEACHER_EVAL_ALL_ITERS"
        )
      fi
      for item in "${rgb_train_sets[@]}"; do
        rgb_train_args+=(--set "$item")
      done
      if [[ "$RGB_TEACHER_DISABLE_OUTPUT_ARTIFACT" == "1" ]]; then
        rgb_train_args+=(--disable-output-artifact)
      fi
      "$NERFBASELINES_BIN" "${rgb_train_args[@]}"
      if [[ -d "$rgb_teacher_checkpoint" ]]; then
        "$PYTHON" scripts/prepare_rgb_teacher_manifest.py \
          --scene "$scene" \
          --source_path "$nb_dataset" \
          --output "$teacher_dir/rgb_teacher_manifest.json" \
          --output_root "$teacher_dir" \
          --backend wildgaussians \
          --checkpoint "$rgb_teacher_checkpoint" \
          --nerfbaselines_bin "$NERFBASELINES_BIN" \
          --nerfbaselines_backend "$NERFBASELINES_BACKEND" \
          --train_steps "$RGB_TEACHER_TRAIN_STEPS" \
          --logger "$RGB_TEACHER_LOGGER" \
          --save_iters "$RGB_TEACHER_SAVE_ITERS" \
          --eval_few_iters "$RGB_TEACHER_EVAL_FEW_ITERS" \
          --eval_all_iters "$RGB_TEACHER_EVAL_ALL_ITERS" \
          --render_output_names "$RGB_TEACHER_RENDER_OUTPUT_NAMES" \
          --render_resolution "$WILDGAUSSIANS_RENDER_RESOLUTION" \
          "${rgb_preset_args[@]}" \
          "${rgb_set_args[@]}" \
          "${rgb_disable_artifact_args[@]}"
      fi
    fi
  fi

  repair_args=()
  if [[ "$REPAIR_SYNTHETIC_ARTIFACTS" == "1" ]]; then
    repair_args+=(--repair_synthetic_artifacts)
  fi
  quality_gate_args=()
  if [[ "$SYNTHETIC_QUALITY_GATE" != "1" ]]; then
    quality_gate_args+=(--skip_synthetic_quality_gate)
  fi

  if [[ "$RUN_PSEUDO_QUERY_MANIFEST" == "1" ]]; then
    "$PYTHON" scripts/build_pseudo_query_manifest.py \
      -s "$DATA_ROOT/$scene" \
      -m "$baseline_model" \
      -r 1 -f sp -g 3dgs --images processed --data_device cpu \
      --iteration "$BASELINE_ITERS" \
      --scene_name "$scene" \
      --output "$manifest" \
      --synthetic_count "$SYNTHETIC_RENDER_COUNT" \
      --synthetic_image_root "$pseudo_dir/synthetic_rgb" \
      --synthetic_seed "$SYNTHETIC_SEED" \
      --synthetic_pose_sampler "$PSEUDO_QUERY_POSE_SAMPLER" \
      --synthetic_spatial_min_offset_ratio "$SYNTHETIC_SPATIAL_MIN_OFFSET_RATIO" \
      --synthetic_spatial_max_offset_ratio "$SYNTHETIC_SPATIAL_MAX_OFFSET_RATIO" \
      --synthetic_spatial_yaw_deg "$SYNTHETIC_SPATIAL_YAW_DEG" \
      --synthetic_spatial_height_offset_ratio "$SYNTHETIC_SPATIAL_HEIGHT_OFFSET_RATIO" \
      --render_synthetic_backend "$RENDER_SYNTHETIC_BACKEND" \
      --rgb_teacher_checkpoint "$rgb_teacher_checkpoint" \
      --nerfbaselines_bin "$NERFBASELINES_BIN" \
      --nerfbaselines_backend "$NERFBASELINES_BACKEND" \
      --wildgaussians_render_root "$pseudo_dir/wildgaussians_render" \
      --wildgaussians_output_names "$RGB_TEACHER_RENDER_OUTPUT_NAMES" \
      --wildgaussians_render_scale "$WILDGAUSSIANS_RENDER_SCALE" \
      --wildgaussians_render_resolution "$WILDGAUSSIANS_RENDER_RESOLUTION" \
      --wildgaussians_appearance_mode "$WILDGAUSSIANS_APPEARANCE_MODE" \
      --matcha_model_path "$matcha_model_path" \
      --matcha_root "$MATCHA_ROOT" \
      --matcha_python "$MATCHA_PYTHON" \
      --matcha_render_root "$pseudo_dir/matcha_render" \
      --matcha_iteration "$MATCHA_ITERATION" \
      --matcha_render_resolution "$MATCHA_RENDER_RESOLUTION" \
      --synthetic_appearance_strategy "$SYNTHETIC_APPEARANCE_STRATEGY" \
      --synthetic_accept_score "$SYNTHETIC_ACCEPT_SCORE" \
      --synthetic_qa_max_mean "$SYNTHETIC_QA_MAX_MEAN" \
      --synthetic_qa_max_p95 "$SYNTHETIC_QA_MAX_P95" \
      --synthetic_qa_max_mild_frac "$SYNTHETIC_QA_MAX_MILD_FRAC" \
      --synthetic_qa_max_severe_frac "$SYNTHETIC_QA_MAX_SEVERE_FRAC" \
      --synthetic_qa_max_low_detail_mean "$SYNTHETIC_QA_MAX_LOW_DETAIL_MEAN" \
      --synthetic_repair_threshold "$SYNTHETIC_REPAIR_THRESHOLD" \
      --min_opacity_multiplier "$MIN_OPACITY_MULTIPLIER" \
      "${quality_gate_args[@]}" \
      "${repair_args[@]}"
  elif [[ ! -f "$manifest" && ( "$RUN_TEACHER_CACHE" == "1" || "$RUN_PSEUDO_QUERY_GATE" == "1" || "$RUN_PSEUDO_QUERY_SELECT" == "1" || "$RUN_TRAIN" == "1" ) ]]; then
    echo "Missing pseudo-query manifest for $scene: $manifest" >&2
    exit 1
  fi

  if [[ "$RUN_TEACHER_CACHE" == "1" ]]; then
    teacher_cache_mask_args=()
    if [[ "$TEACHER_CACHE_SPARSE_VALID_MASK" == "1" ]]; then
      teacher_cache_mask_args+=(
        --sparse_valid_mask
        --sparse_valid_mask_mode "$TEACHER_CACHE_SPARSE_VALID_MASK_MODE"
        --sparse_valid_mask_sources "$TEACHER_CACHE_SPARSE_VALID_MASK_SOURCES"
        --sparse_valid_mask_output_dir "$pseudo_dir/sparse_support_masks"
        --sparse_valid_mask_min_fraction "$TEACHER_CACHE_SPARSE_VALID_MASK_MIN_FRACTION"
        --sparse_valid_mask_candidate_multiplier "$TEACHER_CACHE_SPARSE_VALID_MASK_CANDIDATE_MULTIPLIER"
        --sparse_support_score_weight "$TEACHER_CACHE_SPARSE_SUPPORT_SCORE_WEIGHT"
        --sparse_support_score_min_multiplier "$TEACHER_CACHE_SPARSE_SUPPORT_SCORE_MIN_MULTIPLIER"
        --no_reference_image_scale "$TEACHER_CACHE_NO_REFERENCE_IMAGE_SCALE"
        --no_reference_support_threshold "$TEACHER_CACHE_NO_REFERENCE_SUPPORT_THRESHOLD"
        --no_reference_support_dilate_radius "$TEACHER_CACHE_NO_REFERENCE_SUPPORT_DILATE_RADIUS"
        --no_reference_support_min_area "$TEACHER_CACHE_NO_REFERENCE_SUPPORT_MIN_AREA"
        --no_reference_invalid_min_area "$TEACHER_CACHE_NO_REFERENCE_INVALID_MIN_AREA"
      )
    fi
    "$PYTHON" scripts/build_pseudo_teacher_cache.py \
      -s "$DATA_ROOT/$scene" \
      -m "$baseline_model" \
      -r 1 -f sp -g 3dgs --images processed --data_device cpu \
      --iteration "$BASELINE_ITERS" \
      --cfg "$eval_cfg" \
      --manifest "$manifest" \
      --output "$teacher_cache" \
      --summary_json "$teacher_summary" \
      --sources "$TEACHER_CACHE_SOURCES" \
      --max_queries "$TEACHER_CACHE_MAX" \
      "${teacher_cache_mask_args[@]}"
  fi

  if [[ "$RUN_TEACHER_CACHE_AUDIT" == "1" && -f "$manifest" && -f "$teacher_cache" ]]; then
    "$PYTHON" scripts/audit_pseudo_teacher_cache.py \
      --manifest "$manifest" \
      --output "$teacher_cache" \
      --summary_json "$teacher_summary" \
      --sources "$TEACHER_CACHE_SOURCES"
  fi

  train_manifest="$manifest"
  if [[ "$RUN_PSEUDO_QUERY_GATE" == "1" ]]; then
    gate_args=(
      --manifest "$manifest"
      --output "$gated_manifest"
      --summary_json "$gate_summary"
      --synthetic_qa_max_mean "$SYNTHETIC_QA_MAX_MEAN"
      --synthetic_qa_max_p95 "$SYNTHETIC_QA_MAX_P95"
      --synthetic_qa_max_mild_frac "$SYNTHETIC_QA_MAX_MILD_FRAC"
      --synthetic_qa_max_severe_frac "$SYNTHETIC_QA_MAX_SEVERE_FRAC"
      --synthetic_qa_max_low_detail_mean "$SYNTHETIC_QA_MAX_LOW_DETAIL_MEAN"
    )
    if [[ -f "$teacher_cache" && "$PSEUDO_QUERY_ENABLE_TEACHER_GATE" == "1" ]]; then
      gate_args+=(
        --teacher_gate
        --teacher_cache "$teacher_cache"
        --teacher_max_sparse_te "$PSEUDO_QUERY_TEACHER_MAX_SPARSE_TE"
        --teacher_max_dense_te "$PSEUDO_QUERY_TEACHER_MAX_DENSE_TE"
        --teacher_allowed_stages "$PSEUDO_QUERY_TEACHER_ALLOWED_STAGES"
        --teacher_gate_sources "$PSEUDO_QUERY_TEACHER_GATE_SOURCES"
      )
    else
      gate_args+=(--no_teacher_gate)
    fi
    "$PYTHON" scripts/gate_pseudo_query_manifest.py "${gate_args[@]}"
    train_manifest="$gated_manifest"
  fi

  if [[ "$RUN_PSEUDO_QUERY_SELECT" == "1" ]]; then
    select_args=(
      --manifest "$train_manifest"
      --output "$selected_manifest"
      --summary_json "$selection_summary"
      --max_synthetic "$PSEUDO_QUERY_SELECT_MAX_SYNTHETIC"
      --synthetic_sources "synthetic_rgb"
      --sort_by "$PSEUDO_QUERY_SELECT_SORT_BY"
      --min_support_frac "$PSEUDO_QUERY_MIN_SUPPORT_FRAC"
      --min_support_score "$PSEUDO_QUERY_MIN_SUPPORT_SCORE"
    )
    if [[ -f "$teacher_cache" ]]; then
      select_args+=(--teacher_cache "$teacher_cache")
    fi
    "$PYTHON" scripts/select_pseudo_query_pool.py "${select_args[@]}"
    train_manifest="$selected_manifest"
  fi

  if [[ "$RUN_TRAIN" == "1" ]]; then
    if [[ "$LA_TRAIN_MODE" == "adapt" ]]; then
      if [[ ! -d "$train_model" || "${FORCE_TRAIN_COPY:-0}" == "1" ]]; then
        rm -rf "$train_model"
        cp -a "$baseline_model" "$train_model"
      fi
    elif [[ ! -d "$train_model" || "${FORCE_TRAIN_COPY:-0}" == "1" ]]; then
      rm -rf "$train_model"
      mkdir -p "$train_model"
    fi
    if [[ "$LA_TRAIN_MODE" == "scratch" && "$train_needs_bootstrap_landmarks" == "1" ]]; then
      bootstrap_landmark_path="$train_model/$LA_BOOTSTRAP_DETECTOR_FOLDER/sampled_idx.pkl"
      if [[ ! -f "$bootstrap_landmark_path" || "$FORCE_LA_BOOTSTRAP_LANDMARKS" == "1" ]]; then
        "$PYTHON" train_detector.py \
          -s "$DATA_ROOT/$scene" \
          -m "$train_model" \
          -r 1 -f sp -g 3dgs --images processed --data_device cpu \
          --iteration 0 \
          --iterations 1 \
          --detector_folder "$LA_BOOTSTRAP_DETECTOR_FOLDER" \
          --landmark_num "$LA_BOOTSTRAP_LANDMARK_NUM" \
          --landmark_k "$LA_BOOTSTRAP_LANDMARK_K" \
          --sampling_mode "$LA_BOOTSTRAP_SAMPLING_MODE" \
          --landmark_only \
          --test_iterations 1 \
          --save_iterations 1
      fi
      train_landmark_path="$bootstrap_landmark_path"
    fi
    pseudo_filter_args=()
    if [[ "$PSEUDO_QUERY_FILTER_TEACHER_CACHE" == "1" ]]; then
      pseudo_filter_args+=(--pseudo_query_filter_teacher_cache)
    else
      pseudo_filter_args+=(--no-pseudo_query_filter_teacher_cache)
    fi
    pseudo_cache_requirement_args=()
    if [[ "$PSEUDO_QUERY_REQUIRE_TEACHER_CACHE" == "1" ]]; then
      pseudo_cache_requirement_args+=(--pseudo_query_require_teacher_cache)
    else
      pseudo_cache_requirement_args+=(--no-pseudo_query_require_teacher_cache)
    fi
    pseudo_region_weight_args=()
    if [[ "$PSEUDO_QUERY_NO_REFERENCE_REGION_WEIGHT" == "1" ]]; then
      pseudo_region_weight_args+=(--pseudo_query_no_reference_region_weight)
    else
      pseudo_region_weight_args+=(--no-pseudo_query_no_reference_region_weight)
    fi
    pseudo_stage_gate_args=()
    if [[ "$PSEUDO_QUERY_EXCLUDE_SPARSE_FAILURE_STAGES" == "1" ]]; then
      pseudo_stage_gate_args+=(--pseudo_query_exclude_sparse_failure_stages)
    fi
    pseudo_reliability_stats_args=()
    if [[ -n "$PSEUDO_QUERY_RELIABILITY_STATS_MIN_WEIGHT" ]]; then
      pseudo_reliability_stats_args+=(
        --pseudo_query_reliability_stats_min_weight "$PSEUDO_QUERY_RELIABILITY_STATS_MIN_WEIGHT"
      )
    fi
    direct_depth_check_args=()
    if [[ "$LA_DIRECT_DEPTH_CHECK" == "1" ]]; then
      direct_depth_check_args+=(--direct_depth_check)
    fi
    "$PYTHON" train_locaware.py \
      -s "$DATA_ROOT/$scene" \
      -m "$train_model" \
      -r 1 -f sp -g 3dgs --images processed --data_device cpu \
      "${train_load_iteration_args[@]}" \
      --iterations "$end_iter" \
      --train_phase feature \
      --loc_teacher direct \
      --landmark_path "$train_landmark_path" \
      --loc_interval 1 \
      --loc_start_iter "$LA_LOC_START_ITER" \
      --loc_anchors 2048 \
      --loc_direct_weight 0.05 \
      --loc_multiview_weight 0.03 \
      --loc_multiview_temperature 0.07 \
      --loc_multiview_slots 4 \
      --loc_full_bank_weight 0.05 \
      --loc_full_bank_temperature 0.07 \
      --loc_full_bank_hard_negatives 32 \
      --loc_full_bank_margin 0.2 \
      --loc_anchor_weight 0.01 \
      --loc_desc_weight 0.0 \
      --loc_reproj_weight 0.0 \
      --loc_proto_weight 0.0 \
      --loc_rank_weight 0.0 \
      --loc_opacity_weight 0.0 \
      --no-use_loc_opacity \
      "${direct_depth_check_args[@]}" \
      --direct_depth_abs_tolerance 0.001 \
      --direct_depth_rel_tolerance 0.01 \
      --query_mode mixed \
      --mixed_sparse_probability 1.0 \
      --pseudo_query_manifest "$train_manifest" \
      --pseudo_teacher_cache "$teacher_cache" \
      --pseudo_query_sources "$PSEUDO_QUERY_SOURCES" \
      --pseudo_query_max_synthetic "$PSEUDO_QUERY_MAX_SYNTHETIC" \
      --pseudo_query_real_weight "$PSEUDO_QUERY_REAL_WEIGHT" \
      --pseudo_query_synthetic_weight "$PSEUDO_QUERY_SYNTHETIC_WEIGHT" \
      --pseudo_query_sampling_mode "$PSEUDO_QUERY_SAMPLING_MODE" \
      --pseudo_query_reliability_mode "$PSEUDO_QUERY_RELIABILITY_MODE" \
      --pseudo_query_reliability_loss_mode "$PSEUDO_QUERY_RELIABILITY_LOSS_MODE" \
      --pseudo_query_stage_objective_mode "$PSEUDO_QUERY_STAGE_OBJECTIVE_MODE" \
      --pseudo_query_stage_stats_policy "$PSEUDO_QUERY_STAGE_STATS_POLICY" \
      --pseudo_query_reliability_min_weight "$PSEUDO_QUERY_RELIABILITY_MIN_WEIGHT" \
      --pseudo_query_reliability_real_min_weight "$PSEUDO_QUERY_RELIABILITY_REAL_MIN_WEIGHT" \
      --pseudo_query_reliability_synthetic_min_weight "$PSEUDO_QUERY_RELIABILITY_SYNTHETIC_MIN_WEIGHT" \
      --pseudo_query_reliability_memory_min_weight "$PSEUDO_QUERY_RELIABILITY_MEMORY_MIN_WEIGHT" \
      "${pseudo_reliability_stats_args[@]}" \
      --pseudo_query_reliability_error_scale "$PSEUDO_QUERY_RELIABILITY_ERROR_SCALE" \
      --pseudo_query_reliability_inlier_power "$PSEUDO_QUERY_RELIABILITY_INLIER_POWER" \
      --pseudo_query_reliability_teacher_ok_weight "$PSEUDO_QUERY_RELIABILITY_TEACHER_OK_WEIGHT" \
      --pseudo_query_reliability_dense_improves_weight "$PSEUDO_QUERY_RELIABILITY_DENSE_IMPROVES_WEIGHT" \
      --pseudo_query_reliability_mixed_weight "$PSEUDO_QUERY_RELIABILITY_MIXED_WEIGHT" \
      --pseudo_query_reliability_dense_rescues_weight "$PSEUDO_QUERY_RELIABILITY_DENSE_RESCUES_WEIGHT" \
      --pseudo_query_reliability_sparse_failure_weight "$PSEUDO_QUERY_RELIABILITY_SPARSE_FAILURE_WEIGHT" \
      --pseudo_query_reliability_dense_regression_weight "$PSEUDO_QUERY_RELIABILITY_DENSE_REGRESSION_WEIGHT" \
      --pseudo_query_reliability_unknown_weight "$PSEUDO_QUERY_RELIABILITY_UNKNOWN_WEIGHT" \
      "${pseudo_cache_requirement_args[@]}" \
      "${pseudo_filter_args[@]}" \
      "${pseudo_stage_gate_args[@]}" \
      "${pseudo_region_weight_args[@]}" \
      --pseudo_query_no_reference_region_weight_sources "$PSEUDO_QUERY_NO_REFERENCE_REGION_WEIGHT_SOURCES" \
      --pseudo_query_no_reference_region_weight_min "$PSEUDO_QUERY_NO_REFERENCE_REGION_WEIGHT_MIN" \
      --pseudo_query_no_reference_region_weight_support_power "$PSEUDO_QUERY_NO_REFERENCE_REGION_WEIGHT_SUPPORT_POWER" \
      --pseudo_query_no_reference_region_weight_image_scale "$PSEUDO_QUERY_NO_REFERENCE_REGION_WEIGHT_IMAGE_SCALE" \
      --pseudo_query_no_reference_support_threshold "$TEACHER_CACHE_NO_REFERENCE_SUPPORT_THRESHOLD" \
      --pseudo_query_no_reference_support_dilate_radius "$TEACHER_CACHE_NO_REFERENCE_SUPPORT_DILATE_RADIUS" \
      --pseudo_query_no_reference_support_min_area "$TEACHER_CACHE_NO_REFERENCE_SUPPORT_MIN_AREA" \
      --pseudo_query_no_reference_invalid_min_area "$TEACHER_CACHE_NO_REFERENCE_INVALID_MIN_AREA" \
      --pseudo_query_teacher_max_sparse_te "$PSEUDO_QUERY_TEACHER_MAX_SPARSE_TE" \
      --pseudo_query_teacher_max_dense_te "$PSEUDO_QUERY_TEACHER_MAX_DENSE_TE" \
      --pseudo_query_teacher_allowed_stages "$PSEUDO_QUERY_TEACHER_ALLOWED_STAGES" \
      --train_seed "$TRAIN_SEED" \
      --save_iterations "$end_iter" \
      --test_iterations "$end_iter"
  fi

  if [[ "$RUN_LA_FRONTEND_REFRESH" == "1" ]]; then
    train_ply="$train_model/point_cloud/iteration_${end_iter}/point_cloud.ply"
    if [[ ! -f "$train_ply" ]]; then
      echo "Missing trained student point cloud for LA frontend refresh: $train_ply" >&2
      exit 1
    fi
    detector_path="$train_model/$LA_DETECTOR_FOLDER/${LA_DETECTOR_ITERS}_detector.pth"
    landmark_path="$train_model/$LA_DETECTOR_FOLDER/sampled_idx.pkl"
    if [[ ! -f "$detector_path" || ! -f "$landmark_path" || "$FORCE_LA_FRONTEND_REFRESH" == "1" ]]; then
      "$PYTHON" train_detector.py \
        -s "$DATA_ROOT/$scene" \
        -m "$train_model" \
        -r 1 -f sp -g 3dgs --images processed --data_device cpu \
        --iteration "$end_iter" \
        --iterations "$LA_DETECTOR_ITERS" \
        --detector_folder "$LA_DETECTOR_FOLDER" \
        --landmark_num "$LA_DETECTOR_LANDMARK_NUM" \
        --landmark_k "$LA_DETECTOR_LANDMARK_K" \
        --sampling_mode "$LA_DETECTOR_SAMPLING_MODE" \
        --detector_target_mode "$LA_DETECTOR_TARGET_MODE" \
        --min_loc_observations "$LA_DETECTOR_MIN_LOC_OBSERVATIONS" \
        --utility_weight "$LA_DETECTOR_UTILITY_WEIGHT" \
        --pnp_voxel_size "$LA_DETECTOR_PNP_VOXEL_SIZE" \
        --pnp_max_per_voxel "$LA_DETECTOR_PNP_MAX_PER_VOXEL" \
        --pnp_preserve_ratio "$LA_DETECTOR_PNP_PRESERVE_RATIO" \
        --soft_sigma "$LA_DETECTOR_SOFT_SIGMA" \
        --test_iterations "$LA_DETECTOR_ITERS" \
        --save_iterations "$LA_DETECTOR_ITERS"
    fi
    make_eval_cfg "$scene" "$train_model" "$student_eval_cfg" "$LA_DETECTOR_FOLDER" "$LA_DETECTOR_ITERS"
    final_eval_cfg="$student_eval_cfg"
  fi

  if [[ "$RUN_EVAL" == "1" ]]; then
    "$PYTHON" stdloc.py \
      -s "$DATA_ROOT/$scene" \
      -m "$train_model" \
      -r 1 -f sp -g 3dgs --images processed --data_device cpu \
      --iteration "$end_iter" \
      --cfg "$final_eval_cfg" \
      --prefix "pseudo-query-${end_iter}" \
      --sparse_only
  fi
done
