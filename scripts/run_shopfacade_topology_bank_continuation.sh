#!/usr/bin/env bash

set -euo pipefail

GPU="${1:?usage: $0 <gpu:1|2> <off|on>}"
VARIANT="${2:?usage: $0 <gpu:1|2> <off|on>}"
[[ "$GPU" == "1" || "$GPU" == "2" ]] || { echo "gpu must be 1 or 2" >&2; exit 2; }
[[ "$VARIANT" == "off" || "$VARIANT" == "on" ]] || { echo "variant must be off or on" >&2; exit 2; }

ROOT="${CAMBRIDGE_MATCHA_MAINLINE_ROOT:-/mnt/pool/sqy/stdloc_matcha_pretrained_mainline_v1_20260714}"
SOURCE="$ROOT/lafgs_external_matcha/ShopFacade"
DATA="${CAMBRIDGE_DATA_ROOT:-/mnt/pool/sqy/Cambridge_stdloc}/ShopFacade"
START=60000
STEPS="${TOPOLOGY_CONTINUATION_STEPS:-200}"
END=$((START + STEPS))
TOPOLOGY_GROWTH_CAP="${TOPOLOGY_GROWTH_CAP:-0.0001}"
TOPOLOGY_TAG="${TOPOLOGY_TAG:-default}"
OUTPUT="$ROOT/topology_bank_continuation/ShopFacade_${VARIANT}_${STEPS}_${TOPOLOGY_TAG}"
LANDMARKS="${TOPOLOGY_LANDMARKS:-$SOURCE/detector_queryjoint_exactset_fim_fixed_6000/sampled_idx.pkl}"
RISK_CFG="${TOPOLOGY_RISK_CFG:-$ROOT/eval_configs/ShopFacade/exactset_fim_fixed_6000_calibrated.yaml}"
TOPOLOGY_RISK_EPSILON="${TOPOLOGY_RISK_EPSILON:-0.0001}"
DIFF_PNP_XYZ_LR="${DIFF_PNP_XYZ_LR:-6.107374653220177e-7}"
DIFF_PNP_MAX_STEP_M="${DIFF_PNP_MAX_STEP_M:-0}"
DIFF_PNP_WEIGHT="${DIFF_PNP_WEIGHT:-0.05}"
LOG="$ROOT/logs/ShopFacade/topology_bank_${VARIANT}_${STEPS}_${TOPOLOGY_TAG}.log"

export CUDA_VISIBLE_DEVICES="$GPU"
export CUDA_HOME=/usr/local/cuda-11.8
export PATH="/root/miniconda3/envs/cybersim_agent/bin:$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:/root/miniconda3/envs/cybersim_agent/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

mkdir -p "$OUTPUT/point_cloud" "$(dirname "$LOG")"
if [[ ! -e "$OUTPUT/point_cloud/iteration_${START}" ]]; then
  ln -s "$SOURCE/point_cloud/iteration_${START}" "$OUTPUT/point_cloud/iteration_${START}"
fi
cp -f "$SOURCE/cfg_args" "$OUTPUT/cfg_args"
[[ -f "$SOURCE/artifact_provenance.json" ]] && cp -f "$SOURCE/artifact_provenance.json" "$OUTPUT/artifact_provenance.json"

TOPOLOGY_ARGS=()
if [[ "$VARIANT" == "on" ]]; then
  TOPOLOGY_ARGS=(
    --enable_topology
    --lafgs_topology_start_iter $((START - 30000 + 50))
    --topology_stats_warmup $((START + 50))
    --topology_update_interval 50
    --topology_min_observations 8
    --topology_split_quantile 0.99
    --topology_ambiguity_quantile 0.90
    --topology_min_repeatability 0.25
    --topology_min_radius 0
    --topology_growth_cap_per_event "$TOPOLOGY_GROWTH_CAP"
    --topology_total_point_budget_ratio 1.001
    --topology_cooldown_iterations 100
    --topology_max_mutation_events 1
    --topology_pose_information_floor 0.05
    --topology_residual_score_floor 0.05
    --topology_restrict_split_to_landmarks
    --topology_risk_commit_policy heldout_pose
    --topology_risk_holdout_size 8
    --topology_risk_holdout_selection pose_stratified
    --topology_risk_epsilon "$TOPOLOGY_RISK_EPSILON"
    --topology_risk_pose_cfg "$RISK_CFG"
    --topology_risk_pose_ae_scale 2
    --topology_risk_pose_te_scale 5
    --topology_risk_pose_inlier_weight 0.05
    --topology_risk_pose_veto_mode r5_r2_tail
  )
else
  TOPOLOGY_ARGS=(--lafgs_topology_start_iter $((END + 1)))
fi

cd "$PYTHONPATH"
python train_lafgs.py \
  -s "$DATA" -m "$OUTPUT" -r 1 -f sp -g 2dgs --images processed --data_device cpu \
  --load_iteration "$START" --iterations "$END" --train_seed 2026 --train_phase full \
  --lafgs_curriculum_base_iter_override 30000 \
  --loc_interval 2 --synthetic_view_ratio 0 --lafgs_stage_schedule pretrained_2dgs \
  --lafgs_stage_bootstrap_until 3000 --lafgs_stage_joint_until 15000 \
  --lafgs_stage_refine_base_weight 0.2 --lafgs_stage_refine_loc_weight 1.5 \
  --lafgs_stage_refine_geometry_anchor_weight 0.05 --landmark_path "$LANDMARKS" \
  --no-lafgs_dense_feature_render --loc_direct_weight 1 --loc_multiview_weight 0.1 \
  --loc_multiview_slots 2 --loc_multiview_memory_device cpu --loc_full_bank_weight 0.1 \
  --loc_full_bank_hard_negatives 64 --loc_full_bank_max_landmarks 16384 \
  --loc_full_bank_pose_information_weight 0.5 --loc_full_bank_pose_information_floor 0.2 \
  --loc_full_bank_pose_information_mode point_jacobian --loc_full_bank_balance_weight 0.25 \
  --loc_clean_hard_negative_weight 0.5 --loc_full_bank_clean_reproj_radius 4 \
  --loc_full_bank_clean_hard_negatives 16 --loc_clean_field_start_iter 15000 \
  --loc_clean_field_full_bank_weight_scale 0.5 --loc_clean_field_clean_hn_weight_scale 1.5 \
  --loc_clean_field_balance_weight 0.5 --loc_clean_field_pose_information_weight 0.5 \
  --lafgs_curriculum --lafgs_diff_pnp_start_iter 3000 --lafgs_geometry_start_iter 15000 \
  --lafgs_diff_pnp_weight "$DIFF_PNP_WEIGHT" --lafgs_diff_pnp_max_correspondences 64 \
  --lafgs_diff_pnp_spatial_grid_size 4 \
  --lafgs_diff_pnp_local_window_radius 4 \
  --lafgs_diff_pnp_geometry_local_window_radius 4 \
  --lafgs_diff_pnp_max_condition_number 100000 \
  --lafgs_diff_pnp_geometry_max_reproj_error 4 \
  --lafgs_diff_pnp_geometry_match_max_reproj_error 4 \
  --lafgs_diff_pnp_allow_geometry_grad \
  --lafgs_diff_pnp_isolate_geometry_grad --allow_raw_xyz_geometry_grad \
  --lafgs_diff_pnp_geometry_xyz_lr "$DIFF_PNP_XYZ_LR" \
  --lafgs_diff_pnp_geometry_max_step_m "$DIFF_PNP_MAX_STEP_M" \
  --lafgs_diff_pnp_geometry_reproj_weight 0.01 \
  --lafgs_diff_pnp_geometry_depth_anchor_weight 0.1 \
  --lafgs_diff_pnp_geometry_match_reproj_weight 0.5 \
  --loc_anchor_lr 0 --geometry_anchor_weight 0.1 \
  --lafgs_geometry_grad_clip_abs 10 --loc_full_checkpoint_mode none \
  "${TOPOLOGY_ARGS[@]}" --save_iterations "$END" --test_iterations "$END" \
  2>&1 | tee "$LOG"
