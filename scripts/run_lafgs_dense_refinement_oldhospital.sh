#!/usr/bin/env bash
set -euo pipefail

# Standalone experiment. It intentionally does not alter the LaFGS V2 mainline.
ROOT=${ROOT:-/mnt/pool/sqy/stdloc_lafgs_v2_dense_refinement_20260719}
CLEAN_ROOT=${CLEAN_ROOT:-/mnt/pool/sqy/stdloc_lafgs_v2_clean_matcha_20260719}
SCENE=OldHospital
SCENE_ROOT="$ROOT/$SCENE"
MODEL_ROOT="$CLEAN_ROOT/matcha_wrappers/$SCENE"
DATA_ROOT=${DATA_ROOT:-/mnt/pool/sqy/Cambridge_stdloc}
FINAL_ROOT="$CLEAN_ROOT/$SCENE/final_refit_mean_p98304_s512_rp2"
CFG="$CLEAN_ROOT/$SCENE/configs/mean_p98304_s512_rp2/selected_full_test.yaml"
FIELD_STATE="$FINAL_ROOT/descriptor_3000/3000_lafgs_map_state.pt"
# This is the existing strict candidate holdout.  The final-refit map records
# validation_ratio=0.0, so stdloc.py correctly refuses to repurpose its seen
# training cameras as a direct heldout validation set.
VALID_INPUT="$(cat "$CLEAN_ROOT/$SCENE/results/mean_p98304_s512_rp2/descriptor_3000_validation.results_path")/results.json"
TEST_INPUT="$(cat "$FINAL_ROOT/results/selected_full_test.results_path")/results.json"
# This environment has the prebuilt CUDA 11.8 gsplat extension.  The legacy
# ulfloc_repro environment currently falls back to a JIT build that cannot
# find the host crypt headers on this machine.
PYTHON=${PYTHON:-/root/miniconda3/envs/cybersim_agent/bin/python}
GPU_FIELD=${GPU_FIELD:-1}
GPU_PRIOR=${GPU_PRIOR:-2}
# Keep the depth-consistent field result separate from the earlier prototype,
# whose feature visibility and lifted full-map depth could disagree.
FIELD_LABEL=${FIELD_LABEL:-lafgs_field_depth_consistent}

export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-11.8}
# gsplat lazily builds/loads its CUDA extension.  Reuse the exact toolchain
# used by the clean MAtCha/LaFGS runners rather than relying on host PATH.
export PATH="/root/miniconda3/envs/cybersim_agent/bin:$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="/root/miniconda3/envs/cybersim_agent/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH=/root/STDLoc
export PYTHONHASHSEED=2026
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

mkdir -p "$SCENE_ROOT/logs"
for required in "$PYTHON" "$CFG" "$FIELD_STATE" "$VALID_INPUT" "$TEST_INPUT" "$MODEL_ROOT/point_cloud/iteration_30000/point_cloud.ply"; do
  [[ -e "$required" ]] || { echo "Missing required artifact: $required" >&2; exit 1; }
done

run_mode() {
  local split=$1
  local label=$2
  local mode=$3
  local gpu=$4
  local input=$5
  local output="$SCENE_ROOT/$split/$label"
  local log="$SCENE_ROOT/logs/${split}_${label}.log"
  local extra=()
  if [[ "$mode" == "lafgs_field" ]]; then
    extra+=(--field_state "$FIELD_STATE")
  fi
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" scripts/eval_lafgs_dense_refinement.py \
    -s "$DATA_ROOT/$SCENE" -m "$MODEL_ROOT" --images processed --data_device cpu \
    --gaussian_type 2dgs --feature_type sp --resolution 1 --longest_edge 640 \
    --norm_before_render --iteration 30000 --cfg "$CFG" --input_results "$input" \
    --output_dir "$output" --mode "$mode" --dense_iterations 1 \
    --alpha_min 0.25 --min_depth 0.05 --valid_cell_fraction 0.5 \
    --max_coarse_matches 1024 --max_dense_matches 4096 --checkpoint_every 5 --resume \
    "${extra[@]}" 2>&1 | tee "$log"
}

run_mode validation "$FIELD_LABEL" lafgs_field "$GPU_FIELD" "$VALID_INPUT" &
PID_FIELD=$!
run_mode validation prior_rgb prior_rgb "$GPU_PRIOR" "$VALID_INPUT" &
PID_PRIOR=$!
wait "$PID_FIELD"
wait "$PID_PRIOR"

for mode in "$FIELD_LABEL" prior_rgb; do
  "$PYTHON" scripts/select_lafgs_dense_refinement_gate.py \
    --validation_results "$SCENE_ROOT/validation/$mode/results.json" \
    --output "$SCENE_ROOT/validation/$mode/gate_selection.json" \
    > "$SCENE_ROOT/logs/validation_${mode}_gate.log"
done

run_mode test "$FIELD_LABEL" lafgs_field "$GPU_FIELD" "$TEST_INPUT" &
PID_FIELD=$!
run_mode test prior_rgb prior_rgb "$GPU_PRIOR" "$TEST_INPUT" &
PID_PRIOR=$!
wait "$PID_FIELD"
wait "$PID_PRIOR"

for mode in "$FIELD_LABEL" prior_rgb; do
  "$PYTHON" scripts/select_lafgs_dense_refinement_gate.py \
    --validation_results "$SCENE_ROOT/validation/$mode/results.json" \
    --output "$SCENE_ROOT/validation/$mode/gate_selection.json" \
    --apply_results "$SCENE_ROOT/test/$mode/results.json" \
    --applied_output "$SCENE_ROOT/test/$mode" \
    > "$SCENE_ROOT/logs/test_${mode}_gate_apply.log"
done

echo "Dense-refinement experiment complete: $SCENE_ROOT"
