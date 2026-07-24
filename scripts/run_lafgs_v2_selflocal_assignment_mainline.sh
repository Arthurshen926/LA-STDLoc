#!/usr/bin/env bash
set -euo pipefail

# Required inputs are explicit so this stage cannot silently bind a stale map.
: "${SOURCE_STATE:?Set SOURCE_STATE to the Stage-A wide-bank map state}"
: "${QUERY_CACHE:?Set QUERY_CACHE to the full-resolution native query cache}"
: "${VISIBILITY_CACHE:?Set VISIBILITY_CACHE to the aligned raster visibility cache}"
: "${STATISTICS_RUN:?Set STATISTICS_RUN to corrected train-only native statistics}"
: "${OUTPUT_ROOT:?Set OUTPUT_ROOT}"

GPU="${GPU:-2}"
TOPK="${TOPK:-4}"
FINAL_BUDGET="${FINAL_BUDGET:-24000}"
PYTHON="${PYTHON:-python}"

export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONPATH="${PYTHONPATH:-$(pwd)}"
mkdir -p "$OUTPUT_ROOT"

DISTILLED="$OUTPUT_ROOT/bank_core_reserve_switch_${FINAL_BUDGET}"
"$PYTHON" scripts/redistill_lafgs_bank.py \
  --statistics_run "$STATISTICS_RUN" \
  --source_state "$SOURCE_STATE" \
  --output_dir "$DISTILLED" \
  --budget "$FINAL_BUDGET" \
  --profile core_reserve_switch

DISTILLED_VISIBILITY="$DISTILLED/visibility.pt"
"$PYTHON" scripts/subset_landmark_visibility.py \
  --visibility_cache "$VISIBILITY_CACHE" \
  --source_state "$SOURCE_STATE" \
  --target_state "$DISTILLED/distilled_lafgs_map_state.pt" \
  --output "$DISTILLED_VISIBILITY"

# Train the query-specific teacher on the distilled bank. The explicit cosine
# skip and clean-top1 target preserve already-correct native correspondences.
TEACHER="$OUTPUT_ROOT/protected_one_of_${TOPK}"
"$PYTHON" scripts/train_one_of_k_reranker.py \
  --query_cache "$QUERY_CACHE" \
  --visibility_cache "$DISTILLED_VISIBILITY" \
  --map_state "$DISTILLED/distilled_lafgs_map_state.pt" \
  --landmark_statistics "$DISTILLED/landmark_statistics_full.pt" \
  --global_attractor_statistics \
    "$DISTILLED/distill_global_attractor_prior.pt" \
  --output "$TEACHER.pt" \
  --topk "$TOPK" \
  --patch_radius 2 \
  --patch_step_px 8 \
  --epochs 8 \
  --hidden_dim 64 \
  --global_skip_temperature 0.07

# Teacher is GT-positive gated and removed after this stage. The distilled
# field remains a gated candidate; the default deployment state is the bank
# that already passed direct evaluation.
FIELD_STATE="$OUTPUT_ROOT/protected_teacher_distilled_field.pt"
"$PYTHON" scripts/distill_one_of_k_to_field.py \
  --query_cache "$QUERY_CACHE" \
  --visibility_cache "$DISTILLED_VISIBILITY" \
  --map_state "$DISTILLED/distilled_lafgs_map_state.pt" \
  --reranker_state "$TEACHER.pt" \
  --output "$FIELD_STATE" \
  --epochs 1 \
  --margin 0.02 \
  --max_residual_norm 0.05

printf 'DISTILLED_BANK=%s\n' "$DISTILLED"
printf 'TEACHER_STATE=%s\n' "$TEACHER.pt"
printf 'DEPLOYMENT_STATE=%s\n' \
  "$DISTILLED/distilled_lafgs_map_state.pt"
printf 'CANDIDATE_FIELD_STATE=%s\n' "$FIELD_STATE"
