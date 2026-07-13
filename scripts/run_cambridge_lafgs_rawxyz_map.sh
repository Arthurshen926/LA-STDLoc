#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "Usage: $0 <scene> <gpu> <experiment-root>" >&2
  exit 2
fi

SCENE="$1"
GPU="$2"
EXPERIMENT_ROOT="$3"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/root/miniconda3/envs/cybersim_agent/bin/python}"
DATA_ROOT="${CAMBRIDGE_DATA_ROOT:-/mnt/pool/sqy/Cambridge_stdloc}"
MODEL_ROOT="$EXPERIMENT_ROOT/lafgs_from_sfm/$SCENE"

case "$SCENE" in
  GreatCourt|KingsCollege|OldHospital|ShopFacade|StMarysChurch) ;;
  *) echo "Unsupported Cambridge scene: $SCENE" >&2; exit 2 ;;
esac

export CUDA_HOME=/usr/local/cuda-11.8
export PATH="/root/miniconda3/envs/cybersim_agent/bin:/usr/local/cuda-11.8/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-11.8/lib64:/root/miniconda3/envs/cybersim_agent/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$REPO_ROOT"
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export CUDA_VISIBLE_DEVICES="$GPU"

if [[ -f "$MODEL_ROOT/point_cloud/iteration_30000/point_cloud.ply" ]]; then
  echo "[lafgs-rawxyz] Skip completed map: $MODEL_ROOT"
  exit 0
fi

mkdir -p "$MODEL_ROOT"
cd "$REPO_ROOT"

"$PYTHON" train_lafgs.py \
  -s "$DATA_ROOT/$SCENE" -m "$MODEL_ROOT" \
  -r 1 -f sp -g 2dgs --images processed --data_device cpu \
  --densify_grad_threshold 0.0004 \
  --position_lr_init 0.000016 \
  --scaling_lr 0.001 \
  --iterations 30000 \
  --train_seed 0 \
  --train_phase full \
  --loc_interval 1 \
  --synthetic_view_ratio 0 \
  --synthetic_view_desc_weight 0 \
  --synthetic_view_reproj_weight 0 \
  --lafgs_stage_schedule sfm_from_zero \
  --lafgs_stage_bootstrap_until 3000 \
  --lafgs_stage_joint_until 15000 \
  --lafgs_rgb_densify \
  --lafgs_rgb_densify_until_iter 15000 \
  --lafgs_rgb_densify_child_max_source_drift 0.20 \
  --landmark_path __all__ \
  --lafgs_mvinit_enabled \
  --lafgs_mvinit_max_views 64 \
  --lafgs_mvinit_view_selection uniform \
  --lafgs_mvinit_chunk_size 32768 \
  --lafgs_mvinit_feature_scale 0.5 \
  --loc_full_bank_pose_information_weight 0.5 \
  --loc_full_bank_pose_information_floor 0.2 \
  --loc_full_bank_nearby_as_positive \
  --loc_full_bank_nearby_as_positive_until 15000 \
  --lafgs_curriculum \
  --lafgs_diff_pnp_start_iter 3000 \
  --lafgs_diff_pnp_weight 0.05 \
  --lafgs_diff_pnp_max_correspondences 64 \
  --lafgs_diff_pnp_spatial_grid_size 4 \
  --lafgs_diff_pnp_point_weight_floor 0.05 \
  --lafgs_diff_pnp_local_window_radius 1.25 \
  --lafgs_diff_pnp_geometry_local_window_radius 1.5 \
  --lafgs_diff_pnp_max_condition_number 100000 \
  --lafgs_diff_pnp_geometry_pose_guard_max_loss_increase -1 \
  --lafgs_diff_pnp_geometry_pose_guard_max_loss 5 \
  --lafgs_diff_pnp_geometry_pose_guard_softness 10 \
  --lafgs_diff_pnp_geometry_pose_guard_min_scale 0.05 \
  --lafgs_diff_pnp_feedback_pose_guard_max_loss_increase 30 \
  --lafgs_diff_pnp_feedback_pose_guard_max_loss 5 \
  --lafgs_diff_pnp_feedback_pose_guard_softness 10 \
  --lafgs_diff_pnp_feedback_pose_guard_min_scale 0.05 \
  --lafgs_diff_pnp_allow_geometry_grad \
  --lafgs_diff_pnp_isolate_geometry_grad \
  --allow_raw_xyz_geometry_grad \
  --lafgs_diff_pnp_geometry_xyz_lr 0.000001 \
  --lafgs_geometry_grad_clip_abs 10 \
  --lafgs_diff_pnp_geometry_reproj_weight 0.01 \
  --lafgs_diff_pnp_geometry_depth_anchor_weight 0.1 \
  --loc_anchor_lr 0.00005 \
  --surfel_loc_tangent_bound 0.03 \
  --surfel_loc_normal_bound 0.005 \
  --surfel_loc_radius_floor 1 \
  --surfel_loc_anchor_reg_weight 0.1 \
  --lafgs_diff_pnp_geometry_match_reproj_weight 0.5 \
  --lafgs_diff_pnp_geometry_match_max_reproj_error 2 \
  --lafgs_diff_pnp_geometry_max_reproj_error 4 \
  --loc_full_checkpoint_mode none \
  --save_iterations 5000 10000 20000 30000 \
  --test_iterations 5000 10000 20000 30000
