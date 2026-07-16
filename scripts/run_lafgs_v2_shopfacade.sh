#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-all}"
case "$MODE" in
  prepare|smoke|c0|c1|detector|eval|eval_c0|eval_c1|all) ;;
  *) echo "Usage: $0 <prepare|smoke|c0|c1|detector|eval|eval_c0|eval_c1|all>" >&2; exit 2 ;;
esac

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/root/miniconda3/envs/cybersim_agent/bin/python}"
DATA_ROOT="${CAMBRIDGE_DATA_ROOT:-/mnt/pool/sqy/Cambridge_stdloc}"
MATCHA_ROOT="${CAMBRIDGE_MATCHA_2DGS_ROOT:-/root/MAtCha/output_cambridge_full_retained_v2}"
EXPERIMENT_ROOT="${LAFGS_V2_ROOT:-/mnt/pool/sqy/stdloc_lafgs_v2_20260716}"
MODEL_ROOT="$EXPERIMENT_ROOT/ShopFacade"
SOURCE_RUN="$MATCHA_ROOT/ShopFacade_n20_long_masked_retrain/free_gaussians"
SOURCE_ITERATION=30000
SOURCE_POINT_CLOUD="$SOURCE_RUN/point_cloud/iteration_${SOURCE_ITERATION}"
PROPOSAL_DETECTOR="${LAFGS_V2_PROPOSAL_DETECTOR:-/mnt/pool/sqy/stdloc_matcha_pretrained_mainline_v1_20260714/lafgs_external_matcha/ShopFacade/detector_covsoft_normalized_10000/10000_detector.pth}"
STEPS="${LAFGS_V2_STEPS:-30000}"
DETECTOR_STEPS="${LAFGS_V2_DETECTOR_STEPS:-10000}"
FINAL_BUDGET="${LAFGS_V2_FINAL_BUDGET:-16384}"
MVINIT_VIEWS="${LAFGS_V2_MVINIT_VIEWS:-16}"

export CUDA_VISIBLE_DEVICES=2
export CUDA_HOME=/usr/local/cuda-11.8
export PATH="/root/miniconda3/envs/cybersim_agent/bin:$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:/root/miniconda3/envs/cybersim_agent/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$REPO_ROOT"
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export PYTHONHASHSEED=0

mkdir -p "$EXPERIMENT_ROOT/logs" "$EXPERIMENT_ROOT/evaluations" "$EXPERIMENT_ROOT/configs"
cd "$REPO_ROOT"

prepare() {
  test -f "$SOURCE_POINT_CLOUD/point_cloud.ply"
  mkdir -p "$MODEL_ROOT/point_cloud"
  if [[ ! -e "$MODEL_ROOT/point_cloud/iteration_${SOURCE_ITERATION}" ]]; then
    ln -s "$SOURCE_POINT_CLOUD" "$MODEL_ROOT/point_cloud/iteration_${SOURCE_ITERATION}"
  fi
  cp "$SOURCE_RUN/cfg_args" "$MODEL_ROOT/cfg_args"
  test -f "$PROPOSAL_DETECTOR"
}

run_coreset() {
  local name="$1"
  local synthetic_ratio="$2"
  prepare
  if [[ -f "$MODEL_ROOT/$name/coreset_state.pt" ]]; then
    echo "[LaFGS V2] Reusing completed $name"
    return
  fi
  "$PYTHON" train_coreset_v2.py \
    --model_path "$MODEL_ROOT" \
    --source_path "$DATA_ROOT/ShopFacade" \
    --images processed --data_device cpu --gaussian_type 2dgs --feature_type sp \
    --iteration "$SOURCE_ITERATION" --iterations "$STEPS" \
    --output_folder "$name" --proposal_detector "$PROPOSAL_DETECTOR" \
    --final_budget "$FINAL_BUDGET" \
    --surface_voxel_size 0.05 --surface_normal_bins 0 \
    --mvinit_views "$MVINIT_VIEWS" --mvinit_min_observations 1 \
    --mvinit_chunk_size 32768 \
    --detect_num 2048 --train_keypoints 512 --nms_radius 2 \
    --retrieval_topk 16 --retrieval_chunk_size 8192 \
    --projection_interval 500 --hysteresis 0.05 \
    --descriptor_lr 0.002 --gate_lr 0.01 --temperature 0.07 \
    --coverage_weight 0.5 --budget_weight 1.0 \
    --redundancy_weight 0.1 --redundancy_multiplier 2.0 \
    --trust_weight 0.01 --synthetic_ratio "$synthetic_ratio" \
    --log_interval 500 --seed 2026 \
    2>&1 | tee "$EXPERIMENT_ROOT/logs/${name}.log"
}

run_detector() {
  prepare
  local variant="${1:-coreset_c1}"
  local folder="${variant}_detector"
  if [[ -f "$MODEL_ROOT/$folder/${DETECTOR_STEPS}_detector.pth" ]]; then
    echo "[LaFGS V2] Reusing completed detector"
    return
  fi
  test -f "$MODEL_ROOT/$variant/sampled_idx.pkl"
  "$PYTHON" train_detector.py \
    --model_path "$MODEL_ROOT" --source_path "$DATA_ROOT/ShopFacade" \
    --images processed --data_device cpu --gaussian_type 2dgs --feature_type sp \
    --iteration "$SOURCE_ITERATION" --iterations "$DETECTOR_STEPS" \
    --test_iterations "$DETECTOR_STEPS" --save_iterations "$DETECTOR_STEPS" \
    --detector_folder "$folder" --landmark_num "$FINAL_BUDGET" \
    --precomputed_landmark_path "$MODEL_ROOT/$variant/sampled_idx.pkl" \
    --detector_target_mode hard \
    2>&1 | tee "$EXPERIMENT_ROOT/logs/${variant}_detector.log"
}

run_eval() {
  prepare
  local variant="${1:-coreset_c1}"
  local label="${2:-C2}"
  run_detector "$variant"
  local cfg="$EXPERIMENT_ROOT/configs/${variant}_test.yaml"
  local stable="$EXPERIMENT_ROOT/evaluations/${variant}_test"
  if [[ -f "$stable/results_summary.json" ]]; then
    echo "[LaFGS V2] Reusing completed evaluation"
    return
  fi
  "$PYTHON" scripts/make_stdloc_eval_cfg.py \
    --base_cfg configs/stdloc_cambridge.yaml --output "$cfg" \
    --artifact_model_path "$MODEL_ROOT" \
    --detector_folder "${variant}_detector" --detector_iters "$DETECTOR_STEPS" \
    --landmark_feature_override_path "$variant/final_candidate_teacher_state.pt" \
    --override_landmark_features --detect_num 4096 --nms 2 \
    --reprojection_error 8.306069762524674 --match_threshold 0 --match_topk 1 \
    --max_matches_per_landmark 2 --candidate_frontend_match_policy ignore
  local prefix="lafgs-v2-ShopFacade-${label}-test"
  "$PYTHON" stdloc.py --model_path "$MODEL_ROOT" \
    --source_path "$DATA_ROOT/ShopFacade" --images processed --data_device cpu \
    --gaussian_type 2dgs --feature_type sp --resolution -1 --longest_edge 640 \
    --norm_before_render --iteration "$SOURCE_ITERATION" \
    --cfg "$cfg" --prefix "$prefix" --sparse_only \
    2>&1 | tee "$EXPERIMENT_ROOT/logs/${variant}_eval.log"
  local result
  result="$(find "$REPO_ROOT/results" -maxdepth 1 -type d -name "${prefix}-*" -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)"
  test -f "$result/results_summary.json"
  mkdir -p "$stable"
  cp -r --no-preserve=ownership "$result"/. "$stable"/
  rm -rf "$result"
}

case "$MODE" in
  prepare) prepare ;;
  smoke) LAFGS_V2_STEPS=5 run_coreset coreset_smoke 0 ;;
  c0) run_coreset coreset_c0 0 ;;
  c1) run_coreset coreset_c1 0.15 ;;
  detector) run_detector coreset_c1 ;;
  eval|eval_c1) run_eval coreset_c1 C2 ;;
  eval_c0) run_eval coreset_c0 C0 ;;
  all)
    run_coreset coreset_c0 0
    run_coreset coreset_c1 0.15
    run_eval coreset_c1 C2
    ;;
esac
