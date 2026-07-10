#!/usr/bin/env python3
import argparse
import json
import os
import shlex
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path


DEFAULT_SCENES = ["GreatCourt", "KingsCollege", "OldHospital", "ShopFacade", "StMarysChurch"]
DEFAULT_PYTHON = "/root/miniconda3/envs/cybersim_agent/bin/python"
DEFAULT_LAFGS_STEPS = 500


@dataclass
class ScenePlan:
    scene: str
    data_dir: Path
    baseline_model: Path
    lafgs_model: Path
    detector_source: Path
    baseline_iteration: int
    final_iteration: int
    uses_baseline_initialization: bool
    status: str
    missing_reasons: list
    train_baseline_command: list
    train_lafgs_command: list
    train_baseline_detector_command: list
    baseline_eval_cfg_command: list
    baseline_eval_command: list
    train_lafgs_detector_command: list
    lafgs_eval_cfg_command: list
    lafgs_eval_command: list
    checkpoint_iterations: list
    checkpoint_lafgs_detector_commands: list
    checkpoint_lafgs_eval_cfg_commands: list
    checkpoint_lafgs_eval_commands: list

    def to_json(self):
        def convert(value):
            if isinstance(value, Path):
                return str(value)
            if isinstance(value, list):
                return [convert(item) for item in value]
            if isinstance(value, tuple):
                return [convert(item) for item in value]
            if isinstance(value, dict):
                return {str(key): convert(item) for key, item in value.items()}
            return value

        data = asdict(self)
        return convert(data)


def _existing_python():
    path = Path(DEFAULT_PYTHON)
    return str(path) if path.exists() else sys.executable


def _common_data_args(data_dir, gaussian_type="3dgs"):
    return [
        "-s",
        str(data_dir),
        "-r",
        "1",
        "-f",
        "sp",
        "-g",
        str(gaussian_type),
        "--images",
        "processed",
        "--data_device",
        "cpu",
    ]


def _common_train_args():
    return [
        "--densify_grad_threshold",
        "0.0004",
        "--position_lr_init",
        "0.000016",
        "--scaling_lr",
        "0.001",
    ]


def _train_baseline_command(python, data_dir, baseline_model, baseline_iterations, gaussian_type="3dgs"):
    return [
        python,
        "train.py",
        *_common_data_args(data_dir, gaussian_type=gaussian_type),
        *_common_train_args(),
        "-m",
        str(baseline_model),
        "--iterations",
        str(baseline_iterations),
        "--train_detector",
        "--test_iterations",
        str(baseline_iterations),
        "--save_iterations",
        str(baseline_iterations),
        "--test_detector_iterations",
        str(baseline_iterations),
        "--save_detector_iterations",
        str(baseline_iterations),
        "--detector_folder",
        "detector",
    ]


def _train_lafgs_command(
    python,
    data_dir,
    lafgs_model,
    baseline_iterations,
    final_iteration,
    gaussian_type="3dgs",
    load_iteration=None,
    loc_interval=1,
    mvinit_feature_scale=0.5,
    mvinit_max_views=64,
    mvinit_chunk_size=32768,
    diff_pnp_start_iter=1,
    diff_pnp_weight=0.05,
    diff_pnp_max_correspondences=64,
    diff_pnp_spatial_grid_size=4,
    diff_pnp_point_weight_floor=0.05,
    pose_information_weight=0.5,
    pose_information_floor=0.2,
    full_bank_balance_weight=0.0,
    full_bank_balance_grid_size=0,
    full_bank_balance_depth_bins=0,
    full_bank_balance_max_weight=4.0,
    full_bank_clean_hard_negative_weight=0.0,
    full_bank_clean_reproj_radius=4.0,
    full_bank_clean_hard_negatives=16,
    clean_field_start_iter=0,
    clean_field_full_bank_weight_scale=1.0,
    clean_field_clean_hn_weight_scale=1.0,
    clean_field_balance_weight=-1.0,
    clean_field_pose_information_weight=-1.0,
    clean_field_diff_pnp_weight_scale=1.0,
    geometry_pose_guard_max_loss_increase=-1.0,
    geometry_pose_guard_max_loss=5.0,
    geometry_pose_guard_softness=10.0,
    geometry_pose_guard_min_scale=0.05,
    feedback_pose_guard_max_loss_increase=30.0,
    feedback_pose_guard_max_loss=5.0,
    feedback_pose_guard_softness=10.0,
    feedback_pose_guard_min_scale=0.05,
    geometry_local_window_radius=1.5,
    geometry_confidence_threshold=0.0,
    geometry_margin_threshold=0.0,
    geometry_peak_probability_threshold=0.0,
    geometry_max_entropy=0.0,
    geometry_max_reprojection_error=4.0,
    geometry_use_all_correspondences=False,
    geometry_match_reprojection_weight=0.5,
    geometry_match_confidence_threshold=-1.0,
    geometry_match_margin_threshold=-1.0,
    geometry_match_peak_probability_threshold=-1.0,
    geometry_match_max_entropy=-1.0,
    geometry_match_max_reprojection_error=2.0,
    utility_pose_loss_scale=1.0,
    utility_reprojection_error_scale=4.0,
    pnp_local_window_radius=1.25,
    max_condition_number=100_000.0,
    feedback_pose_guard_keep_gt_reprojection=False,
    allow_geometry_grad=True,
    geometry_reprojection_weight=0.01,
    geometry_depth_anchor_weight=0.1,
    geometry_xyz_lr=0.0,
    loc_anchor_lr=5e-5,
    surfel_loc_tangent_bound=0.03,
    surfel_loc_normal_bound=0.005,
    surfel_loc_radius_floor=1.0,
    surfel_loc_anchor_reg_weight=0.0,
    detach_pnp_points=True,
    allow_raw_xyz_geometry_grad=False,
    stage_schedule="none",
    stage_bootstrap_until=3000,
    stage_joint_until=15000,
    rgb_densify=False,
    rgb_densify_until_iter=0,
    rgb_densify_child_max_source_drift=0.0,
    geometry_residual=False,
    geometry_residual_weight=0.0,
    geometry_residual_max_scale_ratio=0.2,
    geometry_grad_clip_abs=0.0,
    landmark_path=None,
    enable_topology=False,
    topology_enable_soft_prune=False,
    full_bank_nearby_as_positive=True,
    full_bank_nearby_as_positive_until=10000,
    save_iterations=None,
    test_iterations=None,
):
    save_iterations = [final_iteration] if save_iterations is None else list(save_iterations)
    test_iterations = save_iterations if test_iterations is None else list(test_iterations)
    command = [
        python,
        "train_lafgs.py",
        *_common_data_args(data_dir, gaussian_type=gaussian_type),
        *_common_train_args(),
        "-m",
        str(lafgs_model),
        "--iterations",
        str(final_iteration),
        "--train_phase",
        "full",
        "--loc_interval",
        str(loc_interval),
        "--synthetic_view_ratio",
        "0.0",
        "--synthetic_view_desc_weight",
        "0.0",
        "--synthetic_view_reproj_weight",
        "0.0",
        "--lafgs_mvinit_enabled",
        "--lafgs_mvinit_max_views",
        str(mvinit_max_views),
        "--lafgs_mvinit_view_selection",
        "uniform",
        "--lafgs_mvinit_chunk_size",
        str(mvinit_chunk_size),
        "--lafgs_mvinit_feature_scale",
        str(mvinit_feature_scale),
        "--loc_full_bank_pose_information_weight",
        str(pose_information_weight),
        "--loc_full_bank_pose_information_floor",
        str(pose_information_floor),
        "--loc_full_bank_balance_weight",
        str(full_bank_balance_weight),
        "--loc_full_bank_balance_grid_size",
        str(full_bank_balance_grid_size),
        "--loc_full_bank_balance_depth_bins",
        str(full_bank_balance_depth_bins),
        "--loc_full_bank_balance_max_weight",
        str(full_bank_balance_max_weight),
        "--loc_full_bank_clean_hard_negative_weight",
        str(full_bank_clean_hard_negative_weight),
        "--loc_full_bank_clean_reproj_radius",
        str(full_bank_clean_reproj_radius),
        "--loc_full_bank_clean_hard_negatives",
        str(full_bank_clean_hard_negatives),
        "--loc_clean_field_start_iter",
        str(clean_field_start_iter),
        "--loc_clean_field_full_bank_weight_scale",
        str(clean_field_full_bank_weight_scale),
        "--loc_clean_field_clean_hn_weight_scale",
        str(clean_field_clean_hn_weight_scale),
        "--loc_clean_field_balance_weight",
        str(clean_field_balance_weight),
        "--loc_clean_field_pose_information_weight",
        str(clean_field_pose_information_weight),
        "--loc_clean_field_diff_pnp_weight_scale",
        str(clean_field_diff_pnp_weight_scale),
        "--lafgs_curriculum",
        "--lafgs_diff_pnp_start_iter",
        str(diff_pnp_start_iter),
        "--lafgs_diff_pnp_weight",
        str(diff_pnp_weight),
        "--lafgs_diff_pnp_max_correspondences",
        str(diff_pnp_max_correspondences),
        "--lafgs_diff_pnp_spatial_grid_size",
        str(diff_pnp_spatial_grid_size),
        "--lafgs_diff_pnp_point_weight_floor",
        str(diff_pnp_point_weight_floor),
        "--lafgs_diff_pnp_local_window_radius",
        str(pnp_local_window_radius),
        "--lafgs_diff_pnp_geometry_xyz_lr",
        str(geometry_xyz_lr),
        "--lafgs_diff_pnp_geometry_reproj_weight",
        str(geometry_reprojection_weight),
        "--lafgs_diff_pnp_geometry_depth_anchor_weight",
        str(geometry_depth_anchor_weight),
        "--loc_anchor_lr",
        str(loc_anchor_lr),
        "--surfel_loc_tangent_bound",
        str(surfel_loc_tangent_bound),
        "--surfel_loc_normal_bound",
        str(surfel_loc_normal_bound),
        "--surfel_loc_radius_floor",
        str(surfel_loc_radius_floor),
        "--surfel_loc_anchor_reg_weight",
        str(surfel_loc_anchor_reg_weight),
        "--lafgs_diff_pnp_geometry_match_reproj_weight",
        str(geometry_match_reprojection_weight),
        "--lafgs_diff_pnp_geometry_match_confidence_threshold",
        str(geometry_match_confidence_threshold),
        "--lafgs_diff_pnp_geometry_match_margin_threshold",
        str(geometry_match_margin_threshold),
        "--lafgs_diff_pnp_geometry_match_peak_probability_threshold",
        str(geometry_match_peak_probability_threshold),
        "--lafgs_diff_pnp_geometry_match_max_entropy",
        str(geometry_match_max_entropy),
        "--lafgs_diff_pnp_geometry_match_max_reproj_error",
        str(geometry_match_max_reprojection_error),
        "--lafgs_diff_pnp_geometry_max_reproj_error",
        str(geometry_max_reprojection_error),
        "--lafgs_diff_pnp_geometry_confidence_threshold",
        str(geometry_confidence_threshold),
        "--lafgs_diff_pnp_geometry_margin_threshold",
        str(geometry_margin_threshold),
        "--lafgs_diff_pnp_geometry_peak_probability_threshold",
        str(geometry_peak_probability_threshold),
        "--lafgs_diff_pnp_geometry_max_entropy",
        str(geometry_max_entropy),
        "--lafgs_diff_pnp_geometry_local_window_radius",
        str(geometry_local_window_radius),
        "--lafgs_diff_pnp_utility_pose_loss_scale",
        str(utility_pose_loss_scale),
        "--lafgs_diff_pnp_utility_reprojection_error_scale",
        str(utility_reprojection_error_scale),
        "--lafgs_diff_pnp_max_condition_number",
        str(max_condition_number),
        "--lafgs_diff_pnp_geometry_pose_guard_max_loss_increase",
        str(geometry_pose_guard_max_loss_increase),
        "--lafgs_diff_pnp_geometry_pose_guard_max_loss",
        str(geometry_pose_guard_max_loss),
        "--lafgs_diff_pnp_geometry_pose_guard_softness",
        str(geometry_pose_guard_softness),
        "--lafgs_diff_pnp_geometry_pose_guard_min_scale",
        str(geometry_pose_guard_min_scale),
        "--lafgs_diff_pnp_feedback_pose_guard_max_loss_increase",
        str(feedback_pose_guard_max_loss_increase),
        "--lafgs_diff_pnp_feedback_pose_guard_max_loss",
        str(feedback_pose_guard_max_loss),
        "--lafgs_diff_pnp_feedback_pose_guard_softness",
        str(feedback_pose_guard_softness),
        "--lafgs_diff_pnp_feedback_pose_guard_min_scale",
        str(feedback_pose_guard_min_scale),
        "--save_iterations",
        *[str(iteration) for iteration in save_iterations],
        "--test_iterations",
        *[str(iteration) for iteration in test_iterations],
    ]
    if load_iteration is None:
        load_iteration = baseline_iterations
    if load_iteration is not None and int(load_iteration) > 0:
        command[command.index("--iterations") : command.index("--iterations")] = [
            "--load_iteration",
            str(load_iteration),
        ]
    insert_before_save = command.index("--save_iterations")
    stage_args = []
    if str(stage_schedule or "none") != "none":
        stage_args.extend(
            [
                "--lafgs_stage_schedule",
                str(stage_schedule),
                "--lafgs_stage_bootstrap_until",
                str(stage_bootstrap_until),
                "--lafgs_stage_joint_until",
                str(stage_joint_until),
            ]
        )
    if rgb_densify:
        stage_args.append("--lafgs_rgb_densify")
        if int(rgb_densify_until_iter or 0) > 0:
            stage_args.extend(["--lafgs_rgb_densify_until_iter", str(rgb_densify_until_iter)])
        if float(rgb_densify_child_max_source_drift or 0.0) > 0.0:
            stage_args.extend(
                [
                    "--lafgs_rgb_densify_child_max_source_drift",
                    str(rgb_densify_child_max_source_drift),
                ]
            )
    if geometry_residual or float(geometry_residual_weight or 0.0) > 0.0:
        stage_args.extend(
            [
                "--lafgs_geometry_residual",
                "--lafgs_geometry_residual_weight",
                str(geometry_residual_weight),
                "--lafgs_geometry_residual_max_scale_ratio",
                str(geometry_residual_max_scale_ratio),
            ]
        )
    if float(geometry_grad_clip_abs or 0.0) > 0.0:
        stage_args.extend(["--lafgs_geometry_grad_clip_abs", str(geometry_grad_clip_abs)])
    if landmark_path:
        stage_args.extend(["--landmark_path", str(landmark_path)])
    if enable_topology:
        stage_args.append("--enable_topology")
    if topology_enable_soft_prune:
        stage_args.append("--topology_enable_soft_prune")
    if stage_args:
        command[insert_before_save:insert_before_save] = stage_args
    if full_bank_nearby_as_positive:
        command[command.index("--lafgs_curriculum") : command.index("--lafgs_curriculum")] = [
            "--loc_full_bank_nearby_as_positive",
            "--loc_full_bank_nearby_as_positive_until",
            str(full_bank_nearby_as_positive_until),
        ]
    if geometry_use_all_correspondences:
        command.insert(
            command.index("--lafgs_diff_pnp_geometry_local_window_radius"),
            "--lafgs_diff_pnp_geometry_use_all_correspondences",
        )
    if feedback_pose_guard_keep_gt_reprojection:
        command.insert(
            command.index("--save_iterations"),
            "--lafgs_diff_pnp_feedback_pose_guard_keep_gt_reprojection",
        )
    if allow_geometry_grad:
        insert_at = command.index("--lafgs_diff_pnp_geometry_xyz_lr")
        command[insert_at:insert_at] = [
            "--lafgs_diff_pnp_allow_geometry_grad",
            "--lafgs_diff_pnp_isolate_geometry_grad",
        ]
    if allow_raw_xyz_geometry_grad:
        command.insert(
            command.index("--lafgs_diff_pnp_geometry_xyz_lr"),
            "--allow_raw_xyz_geometry_grad",
        )
    if detach_pnp_points:
        command.insert(
            command.index("--lafgs_diff_pnp_feedback_pose_guard_max_loss_increase"),
            "--lafgs_diff_pnp_detach_pnp_points",
        )
    return command


def _eval_cfg_command(
    python,
    cfg,
    output,
    artifact_model_path,
    detector_iterations,
    detect_num,
    nms,
    reprojection_error,
    detector_folder="detector",
    diagnostics=True,
    diagnostics_dump_correspondences=False,
    diagnostics_grid_rows=4,
    diagnostics_grid_cols=4,
    diagnostics_voxel_size=0.25,
    geometry_balance=False,
    geometry_balance_grid_rows=4,
    geometry_balance_grid_cols=4,
    geometry_balance_max_per_cell=64,
    geometry_balance_voxel_size=0.25,
    geometry_balance_max_per_voxel=64,
    geometry_balance_max_matches=0,
):
    command = [
        python,
        "scripts/make_stdloc_eval_cfg.py",
        "--base_cfg",
        str(cfg),
        "--output",
        str(output),
        "--artifact_model_path",
        str(artifact_model_path),
        "--detector_folder",
        str(detector_folder),
        "--detector_iters",
        str(detector_iterations),
        "--detect_num",
        str(detect_num),
        "--nms",
        str(nms),
        "--reprojection_error",
        str(reprojection_error),
        "--summary_json",
        str(output.with_name(output.stem + "_summary.json")),
    ]
    if not diagnostics:
        command.append("--no_diagnostics")
    if diagnostics_dump_correspondences:
        command.append("--diagnostics_dump_correspondences")
    command.extend(
        [
            "--diagnostics_grid_rows",
            str(diagnostics_grid_rows),
            "--diagnostics_grid_cols",
            str(diagnostics_grid_cols),
            "--diagnostics_voxel_size",
            str(diagnostics_voxel_size),
        ]
    )
    if geometry_balance:
        command.extend(
            [
                "--geometry_balance",
                "--geometry_balance_grid_rows",
                str(geometry_balance_grid_rows),
                "--geometry_balance_grid_cols",
                str(geometry_balance_grid_cols),
                "--geometry_balance_max_per_cell",
                str(geometry_balance_max_per_cell),
                "--geometry_balance_voxel_size",
                str(geometry_balance_voxel_size),
                "--geometry_balance_max_per_voxel",
                str(geometry_balance_max_per_voxel),
                "--geometry_balance_max_matches",
                str(geometry_balance_max_matches),
            ]
        )
    return command


def _train_detector_command(
    python,
    data_dir,
    model,
    model_iteration,
    detector_folder,
    detector_iterations,
    gaussian_type="3dgs",
    landmark_num=16384,
    landmark_k=32,
    sampling_mode="localization_aware_pnp",
    detector_target_mode="soft",
    min_loc_observations=4,
    utility_weight=1.0,
    pnp_voxel_size=0.25,
    pnp_max_per_voxel=8,
    pnp_preserve_ratio=0.5,
    soft_sigma=1.5,
    coverage_preserve_ratio=0.75,
    coverage_utility_ratio=0.1,
    coverage_high_confidence_ratio=0.1,
    coverage_grid_size=4,
    coverage_max_per_grid=1536,
    coverage_depth_bins=4,
    coverage_max_per_depth_bin=6144,
):
    return [
        python,
        "train_detector.py",
        *_common_data_args(data_dir, gaussian_type=gaussian_type),
        "-m",
        str(model),
        "--iteration",
        str(model_iteration),
        "--iterations",
        str(detector_iterations),
        "--detector_folder",
        str(detector_folder),
        "--landmark_num",
        str(landmark_num),
        "--landmark_k",
        str(landmark_k),
        "--sampling_mode",
        str(sampling_mode),
        "--detector_target_mode",
        str(detector_target_mode),
        "--min_loc_observations",
        str(min_loc_observations),
        "--utility_weight",
        str(utility_weight),
        "--pnp_voxel_size",
        str(pnp_voxel_size),
        "--pnp_max_per_voxel",
        str(pnp_max_per_voxel),
        "--pnp_preserve_ratio",
        str(pnp_preserve_ratio),
        "--soft_sigma",
        str(soft_sigma),
        "--coverage_preserve_ratio",
        str(coverage_preserve_ratio),
        "--coverage_utility_ratio",
        str(coverage_utility_ratio),
        "--coverage_high_confidence_ratio",
        str(coverage_high_confidence_ratio),
        "--coverage_grid_size",
        str(coverage_grid_size),
        "--coverage_max_per_grid",
        str(coverage_max_per_grid),
        "--coverage_depth_bins",
        str(coverage_depth_bins),
        "--coverage_max_per_depth_bin",
        str(coverage_max_per_depth_bin),
        "--test_iterations",
        str(detector_iterations),
        "--save_iterations",
        str(detector_iterations),
    ]


def _eval_command(python, data_dir, model, iteration, cfg, prefix, gaussian_type="3dgs"):
    return [
        python,
        "stdloc.py",
        *_common_data_args(data_dir, gaussian_type=gaussian_type),
        "-m",
        str(model),
        "--iteration",
        str(iteration),
        "--cfg",
        str(cfg),
        "--prefix",
        prefix,
        "--sparse_only",
    ]


def _checkpoint_iterations(baseline_iteration, final_iteration, interval):
    interval = int(interval or 0)
    if interval <= 0:
        return []
    start = int(baseline_iteration) + interval
    final_iteration = int(final_iteration)
    iterations = list(range(start, final_iteration + 1, interval))
    if not iterations or iterations[-1] != final_iteration:
        iterations.append(final_iteration)
    return iterations


def build_scene_plan(
    scene,
    *,
    data_root,
    baseline_root,
    output_root,
    python,
    baseline_iterations,
    lafgs_steps,
    detect_num,
    nms,
    reprojection_error,
    train_missing_baseline,
    force_train,
    skip_train,
    skip_eval,
    eval_baseline,
    cfg,
    gaussian_type="3dgs",
    loc_interval=1,
    mvinit_feature_scale=0.5,
    mvinit_max_views=64,
    mvinit_chunk_size=32768,
    diff_pnp_start_iter=1,
    diff_pnp_weight=0.05,
    diff_pnp_max_correspondences=64,
    diff_pnp_spatial_grid_size=4,
    diff_pnp_point_weight_floor=0.05,
    pose_information_weight=0.5,
    pose_information_floor=0.2,
    full_bank_balance_weight=0.0,
    full_bank_balance_grid_size=0,
    full_bank_balance_depth_bins=0,
    full_bank_balance_max_weight=4.0,
    full_bank_clean_hard_negative_weight=0.0,
    full_bank_clean_reproj_radius=4.0,
    full_bank_clean_hard_negatives=16,
    clean_field_start_iter=0,
    clean_field_full_bank_weight_scale=1.0,
    clean_field_clean_hn_weight_scale=1.0,
    clean_field_balance_weight=-1.0,
    clean_field_pose_information_weight=-1.0,
    clean_field_diff_pnp_weight_scale=1.0,
    geometry_pose_guard_max_loss_increase=-1.0,
    geometry_pose_guard_max_loss=5.0,
    geometry_pose_guard_softness=10.0,
    geometry_pose_guard_min_scale=0.05,
    feedback_pose_guard_max_loss_increase=30.0,
    feedback_pose_guard_max_loss=5.0,
    feedback_pose_guard_softness=10.0,
    feedback_pose_guard_min_scale=0.05,
    geometry_local_window_radius=1.5,
    geometry_confidence_threshold=0.0,
    geometry_margin_threshold=0.0,
    geometry_peak_probability_threshold=0.0,
    geometry_max_entropy=0.0,
    geometry_max_reprojection_error=4.0,
    geometry_use_all_correspondences=False,
    geometry_match_reprojection_weight=0.5,
    geometry_match_confidence_threshold=-1.0,
    geometry_match_margin_threshold=-1.0,
    geometry_match_peak_probability_threshold=-1.0,
    geometry_match_max_entropy=-1.0,
    geometry_match_max_reprojection_error=2.0,
    utility_pose_loss_scale=1.0,
    utility_reprojection_error_scale=4.0,
    pnp_local_window_radius=1.25,
    max_condition_number=100_000.0,
    feedback_pose_guard_keep_gt_reprojection=False,
    allow_geometry_grad=True,
    geometry_reprojection_weight=0.01,
    geometry_depth_anchor_weight=0.1,
    geometry_xyz_lr=0.0,
    loc_anchor_lr=5e-5,
    surfel_loc_tangent_bound=0.03,
    surfel_loc_normal_bound=0.005,
    surfel_loc_radius_floor=1.0,
    surfel_loc_anchor_reg_weight=0.0,
    rgb_densify_child_max_source_drift=0.0,
    geometry_residual=False,
    geometry_residual_weight=0.0,
    geometry_residual_max_scale_ratio=0.2,
    geometry_grad_clip_abs=0.0,
    direct_pnp_xyz_grad=False,
    allow_raw_xyz_geometry_grad=False,
    lafgs_detector_source="lafgs",
    baseline_detector_folder="detector_baseline_covpreserve",
    lafgs_detector_folder="detector_lafgs",
    lafgs_detector_iterations=30000,
    lafgs_detector_landmark_num=16384,
    lafgs_detector_landmark_k=32,
    lafgs_detector_sampling_mode="coverage_preserving",
    lafgs_detector_target_mode="weighted_hard",
    lafgs_detector_min_loc_observations=4,
    lafgs_detector_utility_weight=1.0,
    lafgs_detector_pnp_voxel_size=0.25,
    lafgs_detector_pnp_max_per_voxel=8,
    lafgs_detector_pnp_preserve_ratio=0.5,
    lafgs_detector_soft_sigma=1.5,
    lafgs_detector_coverage_preserve_ratio=0.75,
    lafgs_detector_coverage_utility_ratio=0.1,
    lafgs_detector_coverage_high_confidence_ratio=0.1,
    lafgs_detector_coverage_grid_size=4,
    lafgs_detector_coverage_max_per_grid=1536,
    lafgs_detector_coverage_depth_bins=4,
    lafgs_detector_coverage_max_per_depth_bin=6144,
    eval_diagnostics=True,
    eval_diagnostics_dump_correspondences=False,
    eval_diagnostics_grid_rows=4,
    eval_diagnostics_grid_cols=4,
    eval_diagnostics_voxel_size=0.25,
    eval_geometry_balance=False,
    eval_geometry_balance_grid_rows=4,
    eval_geometry_balance_grid_cols=4,
    eval_geometry_balance_max_per_cell=64,
    eval_geometry_balance_voxel_size=0.25,
    eval_geometry_balance_max_per_voxel=64,
    eval_geometry_balance_max_matches=0,
    checkpoint_eval_interval=0,
    checkpoint_detector_iterations=0,
    lafgs_from_sfm_zero=False,
):
    data_root = Path(data_root)
    baseline_root = Path(baseline_root)
    output_root = Path(output_root)
    cfg = Path(cfg)
    if lafgs_from_sfm_zero:
        if int(diff_pnp_start_iter) <= 1:
            diff_pnp_start_iter = 3000
        if float(geometry_xyz_lr) <= 0.0:
            geometry_xyz_lr = 2.0e-5
        if float(rgb_densify_child_max_source_drift or 0.0) <= 0.0:
            rgb_densify_child_max_source_drift = 2.0
        if not bool(geometry_residual) and float(geometry_residual_weight or 0.0) <= 0.0:
            geometry_residual = True
            geometry_residual_weight = 0.01
        if float(geometry_grad_clip_abs or 0.0) <= 0.0:
            geometry_grad_clip_abs = 1.0
        allow_geometry_grad = True
        allow_raw_xyz_geometry_grad = True
    data_dir = data_root / scene
    baseline_model = baseline_root / f"{scene}_baseline"
    lafgs_model = output_root / scene
    uses_baseline_initialization = not bool(lafgs_from_sfm_zero)
    lafgs_base_iteration = int(baseline_iterations) if uses_baseline_initialization else 0
    final_iteration = lafgs_base_iteration + int(lafgs_steps)
    detector_source = baseline_model / "detector"
    lafgs_detector_source = str(lafgs_detector_source or "lafgs").lower()
    if lafgs_detector_source not in {"baseline", "lafgs"}:
        raise ValueError("lafgs_detector_source must be 'baseline' or 'lafgs'.")
    eval_detector_folder = "detector" if lafgs_detector_source == "baseline" else str(lafgs_detector_folder)
    eval_detector_iterations = (
        int(baseline_iterations) if lafgs_detector_source == "baseline" else int(lafgs_detector_iterations)
    )
    baseline_point_cloud = baseline_model / "point_cloud" / f"iteration_{baseline_iterations}" / "point_cloud.ply"
    detector_path = detector_source / f"{baseline_iterations}_detector.pth"
    sampled_idx = detector_source / "sampled_idx.pkl"
    lafgs_final = lafgs_model / "point_cloud" / f"iteration_{final_iteration}" / "point_cloud.ply"
    checkpoint_iterations = _checkpoint_iterations(lafgs_base_iteration, final_iteration, checkpoint_eval_interval)
    lafgs_save_iterations = checkpoint_iterations or [final_iteration]
    if (
        str(gaussian_type).lower() == "2dgs"
        and bool(allow_geometry_grad)
        and float(geometry_xyz_lr) > 0.0
        and not bool(allow_raw_xyz_geometry_grad)
    ):
        raise ValueError(
            "2DGS raw xyz geometry updates require allow_raw_xyz_geometry_grad=True."
        )

    missing = []
    if not data_dir.is_dir():
        missing.append(f"missing data directory: {data_dir}")
    baseline_point_cloud_ready = baseline_point_cloud.is_file()
    legacy_detector_ready = detector_path.is_file() and sampled_idx.is_file()
    needs_baseline_point_cloud = uses_baseline_initialization or bool(eval_baseline)
    needs_legacy_detector = (
        (uses_baseline_initialization and not skip_train)
        or lafgs_detector_source == "baseline"
    )
    if needs_baseline_point_cloud and not baseline_point_cloud_ready and not train_missing_baseline:
        missing.append(f"missing baseline point cloud: {baseline_point_cloud}")
    if needs_legacy_detector and not legacy_detector_ready and not train_missing_baseline:
        missing.append(f"missing baseline legacy detector artifacts under: {detector_source}")
    needs_lafgs_final = skip_train and (lafgs_detector_source == "lafgs" or not skip_eval)
    if needs_lafgs_final and not lafgs_final.is_file():
        missing.append(f"missing LaFGS point cloud for --skip_train: {lafgs_final}")
    if skip_train and checkpoint_iterations:
        for checkpoint_iteration in checkpoint_iterations:
            checkpoint_ply = lafgs_model / "point_cloud" / f"iteration_{checkpoint_iteration}" / "point_cloud.ply"
            if not checkpoint_ply.is_file():
                missing.append(f"missing LaFGS checkpoint point cloud for --skip_train: {checkpoint_ply}")
    if cfg and not cfg.is_file():
        missing.append(f"missing eval cfg: {cfg}")

    status = "ready"
    if missing:
        status = "skipped_missing_inputs"
    elif skip_train:
        status = "ready_eval_only"
    elif lafgs_final.is_file() and not force_train:
        status = "ready_existing_lafgs"

    eval_dir = lafgs_model / "eval_configs"
    baseline_detector_iterations = int(lafgs_detector_iterations)
    baseline_eval_cfg = eval_dir / f"baseline_{baseline_detector_folder}_det8192_nms2_reproj12.yaml"
    lafgs_eval_cfg = eval_dir / "lafgs_guarded_pnp_det8192_nms2_reproj12.yaml"
    if lafgs_detector_source == "lafgs":
        lafgs_eval_cfg = eval_dir / f"lafgs_guarded_pnp_{lafgs_detector_folder}_det8192_nms2_reproj12.yaml"
    checkpoint_detector_iterations = (
        int(checkpoint_detector_iterations)
        if int(checkpoint_detector_iterations or 0) > 0
        else int(lafgs_detector_iterations)
    )

    checkpoint_lafgs_detector_commands = []
    checkpoint_lafgs_eval_cfg_commands = []
    checkpoint_lafgs_eval_commands = []
    if checkpoint_iterations and not skip_eval:
        for checkpoint_iteration in checkpoint_iterations:
            checkpoint_detector_folder = (
                "detector"
                if lafgs_detector_source == "baseline"
                else f"{lafgs_detector_folder}_ckpt_{checkpoint_iteration}"
            )
            checkpoint_detector_iters = (
                int(baseline_iterations)
                if lafgs_detector_source == "baseline"
                else checkpoint_detector_iterations
            )
            checkpoint_eval_cfg = (
                eval_dir
                / f"lafgs_guarded_pnp_{checkpoint_detector_folder}_det8192_nms2_reproj12.yaml"
            )
            if lafgs_detector_source == "lafgs":
                checkpoint_lafgs_detector_commands.append(
                    _train_detector_command(
                        python,
                        data_dir,
                        lafgs_model,
                        checkpoint_iteration,
                        checkpoint_detector_folder,
                        checkpoint_detector_iterations,
                        gaussian_type=gaussian_type,
                        landmark_num=lafgs_detector_landmark_num,
                        landmark_k=lafgs_detector_landmark_k,
                        sampling_mode=lafgs_detector_sampling_mode,
                        detector_target_mode=lafgs_detector_target_mode,
                        min_loc_observations=lafgs_detector_min_loc_observations,
                        utility_weight=lafgs_detector_utility_weight,
                        pnp_voxel_size=lafgs_detector_pnp_voxel_size,
                        pnp_max_per_voxel=lafgs_detector_pnp_max_per_voxel,
                        pnp_preserve_ratio=lafgs_detector_pnp_preserve_ratio,
                        soft_sigma=lafgs_detector_soft_sigma,
                        coverage_preserve_ratio=lafgs_detector_coverage_preserve_ratio,
                        coverage_utility_ratio=lafgs_detector_coverage_utility_ratio,
                        coverage_high_confidence_ratio=lafgs_detector_coverage_high_confidence_ratio,
                        coverage_grid_size=lafgs_detector_coverage_grid_size,
                        coverage_max_per_grid=lafgs_detector_coverage_max_per_grid,
                        coverage_depth_bins=lafgs_detector_coverage_depth_bins,
                        coverage_max_per_depth_bin=lafgs_detector_coverage_max_per_depth_bin,
                    )
                )
            checkpoint_lafgs_eval_cfg_commands.append(
                _eval_cfg_command(
                    python,
                    cfg,
                    checkpoint_eval_cfg,
                    lafgs_model,
                    checkpoint_detector_iters,
                    detect_num,
                    nms,
                    reprojection_error,
                    detector_folder=checkpoint_detector_folder,
                    diagnostics=eval_diagnostics,
                    diagnostics_dump_correspondences=eval_diagnostics_dump_correspondences,
                    diagnostics_grid_rows=eval_diagnostics_grid_rows,
                    diagnostics_grid_cols=eval_diagnostics_grid_cols,
                    diagnostics_voxel_size=eval_diagnostics_voxel_size,
                    geometry_balance=eval_geometry_balance,
                    geometry_balance_grid_rows=eval_geometry_balance_grid_rows,
                    geometry_balance_grid_cols=eval_geometry_balance_grid_cols,
                    geometry_balance_max_per_cell=eval_geometry_balance_max_per_cell,
                    geometry_balance_voxel_size=eval_geometry_balance_voxel_size,
                    geometry_balance_max_per_voxel=eval_geometry_balance_max_per_voxel,
                    geometry_balance_max_matches=eval_geometry_balance_max_matches,
                )
            )
            checkpoint_lafgs_eval_commands.append(
                _eval_command(
                    python,
                    data_dir,
                    lafgs_model,
                    checkpoint_iteration,
                    checkpoint_eval_cfg,
                    f"lafgs-guarded-pnp-{checkpoint_iteration}-{checkpoint_detector_folder}",
                    gaussian_type=gaussian_type,
                )
            )

    return ScenePlan(
        scene=scene,
        data_dir=data_dir,
        baseline_model=baseline_model,
        lafgs_model=lafgs_model,
        detector_source=detector_source,
        baseline_iteration=lafgs_base_iteration,
        final_iteration=final_iteration,
        uses_baseline_initialization=uses_baseline_initialization,
        status=status,
        missing_reasons=missing,
        train_baseline_command=_train_baseline_command(
            python,
            data_dir,
            baseline_model,
            baseline_iterations,
            gaussian_type=gaussian_type,
        ),
        train_lafgs_command=_train_lafgs_command(
            python,
            data_dir,
            lafgs_model,
            baseline_iterations,
            final_iteration,
            gaussian_type=gaussian_type,
            load_iteration=lafgs_base_iteration if uses_baseline_initialization else 0,
            loc_interval=loc_interval,
            mvinit_feature_scale=mvinit_feature_scale,
            mvinit_max_views=mvinit_max_views,
            mvinit_chunk_size=mvinit_chunk_size,
            diff_pnp_start_iter=diff_pnp_start_iter,
            diff_pnp_weight=diff_pnp_weight,
            diff_pnp_max_correspondences=diff_pnp_max_correspondences,
            diff_pnp_spatial_grid_size=diff_pnp_spatial_grid_size,
            diff_pnp_point_weight_floor=diff_pnp_point_weight_floor,
            pose_information_weight=pose_information_weight,
            pose_information_floor=pose_information_floor,
            full_bank_balance_weight=full_bank_balance_weight,
            full_bank_balance_grid_size=full_bank_balance_grid_size,
            full_bank_balance_depth_bins=full_bank_balance_depth_bins,
            full_bank_balance_max_weight=full_bank_balance_max_weight,
            full_bank_clean_hard_negative_weight=full_bank_clean_hard_negative_weight,
            full_bank_clean_reproj_radius=full_bank_clean_reproj_radius,
            full_bank_clean_hard_negatives=full_bank_clean_hard_negatives,
            clean_field_start_iter=clean_field_start_iter,
            clean_field_full_bank_weight_scale=clean_field_full_bank_weight_scale,
            clean_field_clean_hn_weight_scale=clean_field_clean_hn_weight_scale,
            clean_field_balance_weight=clean_field_balance_weight,
            clean_field_pose_information_weight=clean_field_pose_information_weight,
            clean_field_diff_pnp_weight_scale=clean_field_diff_pnp_weight_scale,
            geometry_pose_guard_max_loss_increase=geometry_pose_guard_max_loss_increase,
            geometry_pose_guard_max_loss=geometry_pose_guard_max_loss,
            geometry_pose_guard_softness=geometry_pose_guard_softness,
            geometry_pose_guard_min_scale=geometry_pose_guard_min_scale,
            feedback_pose_guard_max_loss_increase=feedback_pose_guard_max_loss_increase,
            feedback_pose_guard_max_loss=feedback_pose_guard_max_loss,
            feedback_pose_guard_softness=feedback_pose_guard_softness,
            feedback_pose_guard_min_scale=feedback_pose_guard_min_scale,
            geometry_local_window_radius=geometry_local_window_radius,
            geometry_confidence_threshold=geometry_confidence_threshold,
            geometry_margin_threshold=geometry_margin_threshold,
            geometry_peak_probability_threshold=geometry_peak_probability_threshold,
            geometry_max_entropy=geometry_max_entropy,
            geometry_max_reprojection_error=geometry_max_reprojection_error,
            geometry_use_all_correspondences=geometry_use_all_correspondences,
            geometry_match_reprojection_weight=geometry_match_reprojection_weight,
            geometry_match_confidence_threshold=geometry_match_confidence_threshold,
            geometry_match_margin_threshold=geometry_match_margin_threshold,
            geometry_match_peak_probability_threshold=geometry_match_peak_probability_threshold,
            geometry_match_max_entropy=geometry_match_max_entropy,
            geometry_match_max_reprojection_error=geometry_match_max_reprojection_error,
            utility_pose_loss_scale=utility_pose_loss_scale,
            utility_reprojection_error_scale=utility_reprojection_error_scale,
            pnp_local_window_radius=pnp_local_window_radius,
            max_condition_number=max_condition_number,
            feedback_pose_guard_keep_gt_reprojection=feedback_pose_guard_keep_gt_reprojection,
            allow_geometry_grad=allow_geometry_grad,
            geometry_reprojection_weight=geometry_reprojection_weight,
            geometry_depth_anchor_weight=geometry_depth_anchor_weight,
            geometry_xyz_lr=geometry_xyz_lr,
            loc_anchor_lr=loc_anchor_lr,
            surfel_loc_tangent_bound=surfel_loc_tangent_bound,
            surfel_loc_normal_bound=surfel_loc_normal_bound,
            surfel_loc_radius_floor=surfel_loc_radius_floor,
            surfel_loc_anchor_reg_weight=surfel_loc_anchor_reg_weight,
            detach_pnp_points=not (direct_pnp_xyz_grad or lafgs_from_sfm_zero),
            allow_raw_xyz_geometry_grad=allow_raw_xyz_geometry_grad,
            stage_schedule="sfm_from_zero" if lafgs_from_sfm_zero else "none",
            stage_bootstrap_until=3000,
            stage_joint_until=15000,
            rgb_densify=bool(lafgs_from_sfm_zero),
            rgb_densify_until_iter=15000 if lafgs_from_sfm_zero else 0,
            rgb_densify_child_max_source_drift=rgb_densify_child_max_source_drift,
            geometry_residual=geometry_residual,
            geometry_residual_weight=geometry_residual_weight,
            geometry_residual_max_scale_ratio=geometry_residual_max_scale_ratio,
            geometry_grad_clip_abs=geometry_grad_clip_abs,
            landmark_path="__all__" if lafgs_from_sfm_zero else None,
            enable_topology=bool(lafgs_from_sfm_zero),
            topology_enable_soft_prune=bool(lafgs_from_sfm_zero),
            save_iterations=lafgs_save_iterations,
            test_iterations=lafgs_save_iterations,
        ),
        train_baseline_detector_command=_train_detector_command(
            python,
            data_dir,
            baseline_model,
            baseline_iterations,
            baseline_detector_folder,
            baseline_detector_iterations,
            gaussian_type=gaussian_type,
            landmark_num=lafgs_detector_landmark_num,
            landmark_k=lafgs_detector_landmark_k,
            sampling_mode=lafgs_detector_sampling_mode,
            detector_target_mode=lafgs_detector_target_mode,
            min_loc_observations=lafgs_detector_min_loc_observations,
            utility_weight=lafgs_detector_utility_weight,
            pnp_voxel_size=lafgs_detector_pnp_voxel_size,
            pnp_max_per_voxel=lafgs_detector_pnp_max_per_voxel,
            pnp_preserve_ratio=lafgs_detector_pnp_preserve_ratio,
            soft_sigma=lafgs_detector_soft_sigma,
            coverage_preserve_ratio=lafgs_detector_coverage_preserve_ratio,
            coverage_utility_ratio=lafgs_detector_coverage_utility_ratio,
            coverage_high_confidence_ratio=lafgs_detector_coverage_high_confidence_ratio,
            coverage_grid_size=lafgs_detector_coverage_grid_size,
            coverage_max_per_grid=lafgs_detector_coverage_max_per_grid,
            coverage_depth_bins=lafgs_detector_coverage_depth_bins,
            coverage_max_per_depth_bin=lafgs_detector_coverage_max_per_depth_bin,
        )
        if (
            not lafgs_from_sfm_zero
            or train_missing_baseline
            or (eval_baseline and not skip_eval)
        )
        else [],
        baseline_eval_cfg_command=[]
        if skip_eval or not eval_baseline
        else _eval_cfg_command(
            python,
            cfg,
            baseline_eval_cfg,
            baseline_model,
            baseline_detector_iterations,
            detect_num,
            nms,
            reprojection_error,
            detector_folder=baseline_detector_folder,
            diagnostics=eval_diagnostics,
            diagnostics_dump_correspondences=eval_diagnostics_dump_correspondences,
            diagnostics_grid_rows=eval_diagnostics_grid_rows,
            diagnostics_grid_cols=eval_diagnostics_grid_cols,
            diagnostics_voxel_size=eval_diagnostics_voxel_size,
            geometry_balance=eval_geometry_balance,
            geometry_balance_grid_rows=eval_geometry_balance_grid_rows,
            geometry_balance_grid_cols=eval_geometry_balance_grid_cols,
            geometry_balance_max_per_cell=eval_geometry_balance_max_per_cell,
            geometry_balance_voxel_size=eval_geometry_balance_voxel_size,
            geometry_balance_max_per_voxel=eval_geometry_balance_max_per_voxel,
            geometry_balance_max_matches=eval_geometry_balance_max_matches,
        ),
        baseline_eval_command=[]
        if skip_eval or not eval_baseline
        else _eval_command(
            python,
            data_dir,
            baseline_model,
            baseline_iterations,
            baseline_eval_cfg,
            f"baseline-{baseline_iterations}-{baseline_detector_folder}",
            gaussian_type=gaussian_type,
        ),
        train_lafgs_detector_command=[]
        if lafgs_detector_source != "lafgs"
        else _train_detector_command(
            python,
            data_dir,
            lafgs_model,
            final_iteration,
            lafgs_detector_folder,
            lafgs_detector_iterations,
            gaussian_type=gaussian_type,
            landmark_num=lafgs_detector_landmark_num,
            landmark_k=lafgs_detector_landmark_k,
            sampling_mode=lafgs_detector_sampling_mode,
            detector_target_mode=lafgs_detector_target_mode,
            min_loc_observations=lafgs_detector_min_loc_observations,
            utility_weight=lafgs_detector_utility_weight,
            pnp_voxel_size=lafgs_detector_pnp_voxel_size,
            pnp_max_per_voxel=lafgs_detector_pnp_max_per_voxel,
            pnp_preserve_ratio=lafgs_detector_pnp_preserve_ratio,
            soft_sigma=lafgs_detector_soft_sigma,
            coverage_preserve_ratio=lafgs_detector_coverage_preserve_ratio,
            coverage_utility_ratio=lafgs_detector_coverage_utility_ratio,
            coverage_high_confidence_ratio=lafgs_detector_coverage_high_confidence_ratio,
            coverage_grid_size=lafgs_detector_coverage_grid_size,
            coverage_max_per_grid=lafgs_detector_coverage_max_per_grid,
            coverage_depth_bins=lafgs_detector_coverage_depth_bins,
            coverage_max_per_depth_bin=lafgs_detector_coverage_max_per_depth_bin,
        ),
        lafgs_eval_cfg_command=[]
        if skip_eval
        else _eval_cfg_command(
            python,
            cfg,
            lafgs_eval_cfg,
            lafgs_model,
            eval_detector_iterations,
            detect_num,
            nms,
            reprojection_error,
            detector_folder=eval_detector_folder,
            diagnostics=eval_diagnostics,
            diagnostics_dump_correspondences=eval_diagnostics_dump_correspondences,
            diagnostics_grid_rows=eval_diagnostics_grid_rows,
            diagnostics_grid_cols=eval_diagnostics_grid_cols,
            diagnostics_voxel_size=eval_diagnostics_voxel_size,
            geometry_balance=eval_geometry_balance,
            geometry_balance_grid_rows=eval_geometry_balance_grid_rows,
            geometry_balance_grid_cols=eval_geometry_balance_grid_cols,
            geometry_balance_max_per_cell=eval_geometry_balance_max_per_cell,
            geometry_balance_voxel_size=eval_geometry_balance_voxel_size,
            geometry_balance_max_per_voxel=eval_geometry_balance_max_per_voxel,
            geometry_balance_max_matches=eval_geometry_balance_max_matches,
        ),
        lafgs_eval_command=[]
        if skip_eval
        else _eval_command(
            python,
            data_dir,
            lafgs_model,
            final_iteration,
            lafgs_eval_cfg,
            f"lafgs-guarded-pnp-{final_iteration}-{eval_detector_folder}",
            gaussian_type=gaussian_type,
        ),
        checkpoint_iterations=checkpoint_iterations,
        checkpoint_lafgs_detector_commands=checkpoint_lafgs_detector_commands,
        checkpoint_lafgs_eval_cfg_commands=checkpoint_lafgs_eval_cfg_commands,
        checkpoint_lafgs_eval_commands=checkpoint_lafgs_eval_commands,
    )


def _run(command, *, cwd, env, dry_run):
    printable = " ".join(shlex.quote(str(part)) for part in command)
    print(f"[lafgs-cambridge] {printable}", flush=True)
    if dry_run:
        return
    subprocess.run([str(part) for part in command], cwd=cwd, env=env, check=True)


def _ensure_symlink(source, target, *, dry_run, force):
    source = Path(source)
    target = Path(target)
    print(f"[lafgs-cambridge] link {target} -> {source}", flush=True)
    if dry_run:
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink() or target.exists():
        if os.path.realpath(target) == os.path.realpath(source):
            return
        if not force:
            raise FileExistsError(f"{target} exists and does not point to {source}; use --force-init")
        if target.is_dir() and not target.is_symlink():
            raise IsADirectoryError(f"{target} is a real directory; refusing to replace it")
        target.unlink()
    target.symlink_to(source)


def _env(cuda_visible_devices=None):
    env = os.environ.copy()
    if cuda_visible_devices:
        env["CUDA_VISIBLE_DEVICES"] = str(cuda_visible_devices)
    cuda_home = env.get("CUDA_HOME", "/usr/local/cuda-11.8")
    conda_bin = "/root/miniconda3/envs/cybersim_agent/bin"
    env["CUDA_HOME"] = cuda_home
    env["PATH"] = f"{conda_bin}:{cuda_home}/bin:{env.get('PATH', '')}"
    ld_parts = [f"{cuda_home}/lib64", "/root/miniconda3/envs/cybersim_agent/lib"]
    if env.get("LD_LIBRARY_PATH"):
        ld_parts.append(env["LD_LIBRARY_PATH"])
    env["LD_LIBRARY_PATH"] = ":".join(ld_parts)
    env["PYTHONPATH"] = env.get("PYTHONPATH", "/root/STDLoc")
    env["PYTORCH_CUDA_ALLOC_CONF"] = env.get("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")
    return env


def run_plan(
    plan,
    *,
    cwd,
    dry_run,
    force_init,
    force_train,
    train_missing_baseline,
    skip_train,
    skip_detector_train,
    skip_eval,
    cuda_visible_devices=None,
):
    if plan.missing_reasons and not train_missing_baseline:
        print(f"[lafgs-cambridge] skip {plan.scene}: {'; '.join(plan.missing_reasons)}", flush=True)
        return "skipped"

    env = _env(cuda_visible_devices=cuda_visible_devices)
    baseline_train_iteration = int(plan.train_baseline_command[plan.train_baseline_command.index("--iterations") + 1])
    baseline_ready = (
        (plan.baseline_model / "point_cloud" / f"iteration_{baseline_train_iteration}" / "point_cloud.ply").is_file()
        and (plan.detector_source / f"{baseline_train_iteration}_detector.pth").is_file()
        and (plan.detector_source / "sampled_idx.pkl").is_file()
    )
    if train_missing_baseline and not baseline_ready:
        _run(plan.train_baseline_command, cwd=cwd, env=env, dry_run=dry_run)

    if plan.uses_baseline_initialization:
        _ensure_symlink(
            plan.baseline_model / "point_cloud" / f"iteration_{plan.baseline_iteration}",
            plan.lafgs_model / "point_cloud" / f"iteration_{plan.baseline_iteration}",
            dry_run=dry_run,
            force=force_init,
        )
    legacy_detector_ready = (
        (plan.detector_source / f"{plan.baseline_iteration}_detector.pth").is_file()
        and (plan.detector_source / "sampled_idx.pkl").is_file()
    )
    if plan.uses_baseline_initialization and (legacy_detector_ready or not skip_train):
        _ensure_symlink(plan.detector_source, plan.lafgs_model / "detector", dry_run=dry_run, force=force_init)
    elif plan.uses_baseline_initialization:
        print(
            f"[lafgs-cambridge] skip baseline detector link for {plan.scene}: "
            f"legacy detector not needed for eval-only run",
            flush=True,
        )
    else:
        print(
            f"[lafgs-cambridge] skip baseline initialization links for {plan.scene}: "
            f"LaFGS starts from SfM point cloud",
            flush=True,
        )

    final_ply = plan.lafgs_model / "point_cloud" / f"iteration_{plan.final_iteration}" / "point_cloud.ply"
    if not skip_train and (force_train or not final_ply.is_file()):
        _run(plan.train_lafgs_command, cwd=cwd, env=env, dry_run=dry_run)
    elif final_ply.is_file():
        print(f"[lafgs-cambridge] skip LaFGS train for {plan.scene}: found {final_ply}", flush=True)
    else:
        print(
            f"[lafgs-cambridge] skip LaFGS train for {plan.scene}: --skip_train set but missing {final_ply}",
            flush=True,
        )

    if plan.train_baseline_detector_command:
        baseline_detector_cmd = plan.train_baseline_detector_command
        baseline_detector_folder = baseline_detector_cmd[baseline_detector_cmd.index("--detector_folder") + 1]
        baseline_detector_iterations = int(baseline_detector_cmd[baseline_detector_cmd.index("--iterations") + 1])
        baseline_detector_ready = (
            (plan.baseline_model / baseline_detector_folder / f"{baseline_detector_iterations}_detector.pth").is_file()
            and (plan.baseline_model / baseline_detector_folder / "sampled_idx.pkl").is_file()
        )
        if not skip_detector_train and (force_train or not baseline_detector_ready):
            _run(baseline_detector_cmd, cwd=cwd, env=env, dry_run=dry_run)
        else:
            print(
                f"[lafgs-cambridge] skip baseline detector train for {plan.scene}: "
                f"found {plan.baseline_model / baseline_detector_folder}",
                flush=True,
            )

    lafgs_detector_commands = list(plan.checkpoint_lafgs_detector_commands)
    if plan.train_lafgs_detector_command:
        lafgs_detector_commands.append(plan.train_lafgs_detector_command)
    for detector_cmd in lafgs_detector_commands:
        detector_folder = detector_cmd[detector_cmd.index("--detector_folder") + 1]
        detector_iterations = int(detector_cmd[detector_cmd.index("--iterations") + 1])
        detector_ready = (
            (plan.lafgs_model / detector_folder / f"{detector_iterations}_detector.pth").is_file()
            and (plan.lafgs_model / detector_folder / "sampled_idx.pkl").is_file()
        )
        if not skip_detector_train and (force_train or not detector_ready):
            _run(detector_cmd, cwd=cwd, env=env, dry_run=dry_run)
        else:
            print(
                f"[lafgs-cambridge] skip LaFGS detector train for {plan.scene}: "
                f"found {plan.lafgs_model / detector_folder}",
                flush=True,
            )

    if not skip_eval:
        if not dry_run:
            (plan.lafgs_model / "eval_configs").mkdir(parents=True, exist_ok=True)
        commands = [plan.baseline_eval_cfg_command, plan.baseline_eval_command]
        if plan.checkpoint_lafgs_eval_cfg_commands or plan.checkpoint_lafgs_eval_commands:
            for cfg_command, eval_command in zip(
                plan.checkpoint_lafgs_eval_cfg_commands,
                plan.checkpoint_lafgs_eval_commands,
            ):
                commands.extend([cfg_command, eval_command])
        commands.extend([plan.lafgs_eval_cfg_command, plan.lafgs_eval_command])
        for command in commands:
            if command:
                _run(command, cwd=cwd, env=env, dry_run=dry_run)
    return "done"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run the five-scene Cambridge guarded-PnP LaFGS protocol.")
    parser.add_argument("--data_root", default="/mnt/pool/sqy/Cambridge_stdloc")
    parser.add_argument("--baseline_root", default="/mnt/pool/sqy/stdloc_la_full_runs")
    parser.add_argument("--output_root", default="")
    parser.add_argument("--results_root", default="/root/STDLoc/results")
    parser.add_argument("--cfg", default="configs/stdloc_cambridge.yaml")
    parser.add_argument("--scenes", nargs="+", default=DEFAULT_SCENES)
    parser.add_argument("--python", default=_existing_python())
    parser.add_argument("--baseline_iterations", type=int, default=30000)
    parser.add_argument("--lafgs_steps", type=int, default=DEFAULT_LAFGS_STEPS)
    parser.add_argument("--detect_num", type=int, default=8192)
    parser.add_argument("--nms", type=int, default=2)
    parser.add_argument("--reprojection_error", type=float, default=12.0)
    parser.add_argument("--gaussian_type", choices=["3dgs", "2dgs"], default="3dgs")
    parser.add_argument(
        "--lafgs_from_sfm_zero",
        action="store_true",
        help="Train LaFGS from the SfM point cloud at iteration 0 with the staged 2DGS localization-aware schedule.",
    )
    parser.add_argument("--loc_interval", type=int, default=1)
    parser.add_argument("--mvinit_feature_scale", type=float, default=0.5)
    parser.add_argument("--mvinit_max_views", type=int, default=64)
    parser.add_argument("--mvinit_chunk_size", type=int, default=32768)
    parser.add_argument("--diff_pnp_start_iter", type=int, default=1)
    parser.add_argument("--diff_pnp_weight", type=float, default=0.05)
    parser.add_argument("--diff_pnp_max_correspondences", type=int, default=64)
    parser.add_argument("--diff_pnp_spatial_grid_size", type=int, default=4)
    parser.add_argument("--diff_pnp_point_weight_floor", type=float, default=0.05)
    parser.add_argument("--pose_information_weight", type=float, default=0.5)
    parser.add_argument("--pose_information_floor", type=float, default=0.2)
    parser.add_argument("--full_bank_balance_weight", type=float, default=0.0)
    parser.add_argument("--full_bank_balance_grid_size", type=int, default=0)
    parser.add_argument("--full_bank_balance_depth_bins", type=int, default=0)
    parser.add_argument("--full_bank_balance_max_weight", type=float, default=4.0)
    parser.add_argument("--full_bank_clean_hard_negative_weight", type=float, default=0.0)
    parser.add_argument("--full_bank_clean_reproj_radius", type=float, default=4.0)
    parser.add_argument("--full_bank_clean_hard_negatives", type=int, default=16)
    parser.add_argument("--clean_field_start_iter", type=int, default=0)
    parser.add_argument("--clean_field_full_bank_weight_scale", type=float, default=1.0)
    parser.add_argument("--clean_field_clean_hn_weight_scale", type=float, default=1.0)
    parser.add_argument("--clean_field_balance_weight", type=float, default=-1.0)
    parser.add_argument("--clean_field_pose_information_weight", type=float, default=-1.0)
    parser.add_argument("--clean_field_diff_pnp_weight_scale", type=float, default=1.0)
    parser.add_argument("--geometry_pose_guard_max_loss_increase", type=float, default=-1.0)
    parser.add_argument("--geometry_pose_guard_max_loss", type=float, default=5.0)
    parser.add_argument("--geometry_pose_guard_softness", type=float, default=10.0)
    parser.add_argument("--geometry_pose_guard_min_scale", type=float, default=0.05)
    parser.add_argument("--feedback_pose_guard_max_loss_increase", type=float, default=30.0)
    parser.add_argument("--feedback_pose_guard_max_loss", type=float, default=5.0)
    parser.add_argument("--feedback_pose_guard_softness", type=float, default=10.0)
    parser.add_argument("--feedback_pose_guard_min_scale", type=float, default=0.05)
    parser.add_argument("--geometry_local_window_radius", type=float, default=1.5)
    parser.add_argument("--geometry_confidence_threshold", type=float, default=0.0)
    parser.add_argument("--geometry_margin_threshold", type=float, default=0.0)
    parser.add_argument("--geometry_peak_probability_threshold", type=float, default=0.0)
    parser.add_argument("--geometry_max_entropy", type=float, default=0.0)
    parser.add_argument("--geometry_max_reprojection_error", type=float, default=4.0)
    parser.add_argument(
        "--geometry_selected_correspondences",
        action="store_true",
        help="Use only the selected PnP correspondences for geometry and descriptor-match feedback.",
    )
    parser.add_argument(
        "--geometry_all_correspondences",
        action="store_true",
        help="Use all valid soft correspondences for geometry feedback.",
    )
    parser.add_argument("--geometry_match_reprojection_weight", type=float, default=0.5)
    parser.add_argument("--geometry_match_confidence_threshold", type=float, default=-1.0)
    parser.add_argument("--geometry_match_margin_threshold", type=float, default=-1.0)
    parser.add_argument("--geometry_match_peak_probability_threshold", type=float, default=-1.0)
    parser.add_argument("--geometry_match_max_entropy", type=float, default=-1.0)
    parser.add_argument("--geometry_match_max_reprojection_error", type=float, default=2.0)
    parser.add_argument("--utility_pose_loss_scale", type=float, default=1.0)
    parser.add_argument("--utility_reprojection_error_scale", type=float, default=4.0)
    parser.add_argument("--pnp_local_window_radius", type=float, default=1.25)
    parser.add_argument("--max_condition_number", type=float, default=100_000.0)
    parser.add_argument("--feedback_pose_guard_keep_gt_reprojection", action="store_true")
    parser.add_argument("--no_geometry_grad", action="store_true")
    parser.add_argument("--geometry_reprojection_weight", type=float, default=0.01)
    parser.add_argument("--geometry_depth_anchor_weight", type=float, default=0.1)
    parser.add_argument("--geometry_xyz_lr", type=float, default=0.0)
    parser.add_argument("--loc_anchor_lr", type=float, default=5e-5)
    parser.add_argument("--surfel_loc_tangent_bound", type=float, default=0.03)
    parser.add_argument("--surfel_loc_normal_bound", type=float, default=0.005)
    parser.add_argument("--surfel_loc_radius_floor", type=float, default=1.0)
    parser.add_argument("--surfel_loc_anchor_reg_weight", type=float, default=0.0)
    parser.add_argument("--rgb_densify_child_max_source_drift", type=float, default=0.0)
    parser.add_argument("--geometry_residual", action="store_true")
    parser.add_argument("--geometry_residual_weight", type=float, default=0.0)
    parser.add_argument("--geometry_residual_max_scale_ratio", type=float, default=0.2)
    parser.add_argument("--geometry_grad_clip_abs", type=float, default=0.0)
    parser.add_argument(
        "--direct_pnp_xyz_grad",
        action="store_true",
        help="Let differentiable PnP pose loss backpropagate through soft PnP 3D points to Gaussian xyz.",
    )
    parser.add_argument(
        "--allow_raw_xyz_geometry_grad",
        action="store_true",
        help="Explicitly allow 2DGS geometry feedback to update raw surfel centers.",
    )
    parser.add_argument(
        "--lafgs_detector_source",
        choices=["baseline", "lafgs"],
        default="lafgs",
        help="Use baseline frontend for legacy ablations, or train/evaluate a frontend on the LaFGS map.",
    )
    parser.add_argument("--baseline_detector_folder", default="detector_baseline_covpreserve")
    parser.add_argument("--lafgs_detector_folder", default="detector_lafgs")
    parser.add_argument("--lafgs_detector_iterations", type=int, default=30000)
    parser.add_argument("--lafgs_detector_landmark_num", type=int, default=16384)
    parser.add_argument("--lafgs_detector_landmark_k", type=int, default=32)
    parser.add_argument(
        "--lafgs_detector_sampling_mode",
        choices=[
            "baseline",
            "localization_aware",
            "localization_aware_spatial",
            "localization_aware_global",
            "localization_aware_pnp",
            "coverage_preserving",
        ],
        default="coverage_preserving",
    )
    parser.add_argument("--lafgs_detector_target_mode", choices=["hard", "soft", "weighted_hard"], default="weighted_hard")
    parser.add_argument("--lafgs_detector_min_loc_observations", type=int, default=4)
    parser.add_argument("--lafgs_detector_utility_weight", type=float, default=1.0)
    parser.add_argument("--lafgs_detector_pnp_voxel_size", type=float, default=0.25)
    parser.add_argument("--lafgs_detector_pnp_max_per_voxel", type=int, default=8)
    parser.add_argument("--lafgs_detector_pnp_preserve_ratio", type=float, default=0.5)
    parser.add_argument("--lafgs_detector_soft_sigma", type=float, default=1.5)
    parser.add_argument("--lafgs_detector_coverage_preserve_ratio", type=float, default=0.75)
    parser.add_argument("--lafgs_detector_coverage_utility_ratio", type=float, default=0.1)
    parser.add_argument("--lafgs_detector_coverage_high_confidence_ratio", type=float, default=0.1)
    parser.add_argument("--lafgs_detector_coverage_grid_size", type=int, default=4)
    parser.add_argument("--lafgs_detector_coverage_max_per_grid", type=int, default=1536)
    parser.add_argument("--lafgs_detector_coverage_depth_bins", type=int, default=4)
    parser.add_argument("--lafgs_detector_coverage_max_per_depth_bin", type=int, default=6144)
    parser.add_argument(
        "--checkpoint_eval_interval",
        type=int,
        default=0,
        help="Save/test LaFGS and run localization every N reconstruction iterations after the baseline iteration.",
    )
    parser.add_argument(
        "--checkpoint_detector_iterations",
        type=int,
        default=0,
        help="Detector training iterations for each checkpoint eval; defaults to --lafgs_detector_iterations.",
    )
    parser.add_argument("--no_eval_diagnostics", action="store_true")
    parser.add_argument("--eval_diagnostics_dump_correspondences", action="store_true")
    parser.add_argument("--eval_diagnostics_grid_rows", type=int, default=4)
    parser.add_argument("--eval_diagnostics_grid_cols", type=int, default=4)
    parser.add_argument("--eval_diagnostics_voxel_size", type=float, default=0.25)
    parser.add_argument("--eval_geometry_balance", action="store_true")
    parser.add_argument("--eval_geometry_balance_grid_rows", type=int, default=4)
    parser.add_argument("--eval_geometry_balance_grid_cols", type=int, default=4)
    parser.add_argument("--eval_geometry_balance_max_per_cell", type=int, default=64)
    parser.add_argument("--eval_geometry_balance_voxel_size", type=float, default=0.25)
    parser.add_argument("--eval_geometry_balance_max_per_voxel", type=int, default=64)
    parser.add_argument("--eval_geometry_balance_max_matches", type=int, default=0)
    parser.add_argument("--skip_detector_train", action="store_true")
    parser.add_argument("--train_missing_baseline", action="store_true")
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_eval", action="store_true")
    parser.add_argument("--no_eval_baseline", action="store_true")
    parser.add_argument("--force_train", action="store_true")
    parser.add_argument("--force_init", action="store_true")
    parser.add_argument("--cuda_visible_devices", default="")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--manifest", default="")
    return parser.parse_args(argv)


def main(argv=None):
    raw_argv = sys.argv[1:] if argv is None else list(argv)
    args = parse_args(raw_argv)
    gaussian_type = args.gaussian_type
    explicit_gaussian_type = any(
        item == "--gaussian_type" or str(item).startswith("--gaussian_type=")
        for item in raw_argv
    )
    if args.lafgs_from_sfm_zero and not explicit_gaussian_type:
        gaussian_type = "2dgs"
    lafgs_steps = args.lafgs_steps
    if args.lafgs_from_sfm_zero and int(lafgs_steps) == DEFAULT_LAFGS_STEPS:
        lafgs_steps = 30000
    output_root = Path(
        args.output_root
        or f"/mnt/pool/sqy/stdloc_lafgs_cambridge_guarded_pnp_{datetime.now().strftime('%Y%m%d')}"
    )
    plans = [
        build_scene_plan(
            scene,
            data_root=args.data_root,
            baseline_root=args.baseline_root,
            output_root=output_root,
            python=args.python,
            baseline_iterations=args.baseline_iterations,
            lafgs_steps=lafgs_steps,
            detect_num=args.detect_num,
            nms=args.nms,
            reprojection_error=args.reprojection_error,
            train_missing_baseline=args.train_missing_baseline,
            force_train=args.force_train,
            skip_train=args.skip_train,
            skip_eval=args.skip_eval,
            eval_baseline=not args.no_eval_baseline,
            cfg=args.cfg,
            gaussian_type=gaussian_type,
            loc_interval=args.loc_interval,
            mvinit_feature_scale=args.mvinit_feature_scale,
            mvinit_max_views=args.mvinit_max_views,
            mvinit_chunk_size=args.mvinit_chunk_size,
            diff_pnp_start_iter=args.diff_pnp_start_iter,
            diff_pnp_weight=args.diff_pnp_weight,
            diff_pnp_max_correspondences=args.diff_pnp_max_correspondences,
            diff_pnp_spatial_grid_size=args.diff_pnp_spatial_grid_size,
            diff_pnp_point_weight_floor=args.diff_pnp_point_weight_floor,
            pose_information_weight=args.pose_information_weight,
            pose_information_floor=args.pose_information_floor,
            full_bank_balance_weight=args.full_bank_balance_weight,
            full_bank_balance_grid_size=args.full_bank_balance_grid_size,
            full_bank_balance_depth_bins=args.full_bank_balance_depth_bins,
            full_bank_balance_max_weight=args.full_bank_balance_max_weight,
            full_bank_clean_hard_negative_weight=args.full_bank_clean_hard_negative_weight,
            full_bank_clean_reproj_radius=args.full_bank_clean_reproj_radius,
            full_bank_clean_hard_negatives=args.full_bank_clean_hard_negatives,
            clean_field_start_iter=args.clean_field_start_iter,
            clean_field_full_bank_weight_scale=args.clean_field_full_bank_weight_scale,
            clean_field_clean_hn_weight_scale=args.clean_field_clean_hn_weight_scale,
            clean_field_balance_weight=args.clean_field_balance_weight,
            clean_field_pose_information_weight=args.clean_field_pose_information_weight,
            clean_field_diff_pnp_weight_scale=args.clean_field_diff_pnp_weight_scale,
            geometry_pose_guard_max_loss_increase=args.geometry_pose_guard_max_loss_increase,
            geometry_pose_guard_max_loss=args.geometry_pose_guard_max_loss,
            geometry_pose_guard_softness=args.geometry_pose_guard_softness,
            geometry_pose_guard_min_scale=args.geometry_pose_guard_min_scale,
            feedback_pose_guard_max_loss_increase=args.feedback_pose_guard_max_loss_increase,
            feedback_pose_guard_max_loss=args.feedback_pose_guard_max_loss,
            feedback_pose_guard_softness=args.feedback_pose_guard_softness,
            feedback_pose_guard_min_scale=args.feedback_pose_guard_min_scale,
            geometry_local_window_radius=args.geometry_local_window_radius,
            geometry_confidence_threshold=args.geometry_confidence_threshold,
            geometry_margin_threshold=args.geometry_margin_threshold,
            geometry_peak_probability_threshold=args.geometry_peak_probability_threshold,
            geometry_max_entropy=args.geometry_max_entropy,
            geometry_max_reprojection_error=args.geometry_max_reprojection_error,
            geometry_use_all_correspondences=(
                bool(args.geometry_all_correspondences) and not bool(args.geometry_selected_correspondences)
            ),
            geometry_match_reprojection_weight=args.geometry_match_reprojection_weight,
            geometry_match_confidence_threshold=args.geometry_match_confidence_threshold,
            geometry_match_margin_threshold=args.geometry_match_margin_threshold,
            geometry_match_peak_probability_threshold=args.geometry_match_peak_probability_threshold,
            geometry_match_max_entropy=args.geometry_match_max_entropy,
            geometry_match_max_reprojection_error=args.geometry_match_max_reprojection_error,
            utility_pose_loss_scale=args.utility_pose_loss_scale,
            utility_reprojection_error_scale=args.utility_reprojection_error_scale,
            pnp_local_window_radius=args.pnp_local_window_radius,
            max_condition_number=args.max_condition_number,
            feedback_pose_guard_keep_gt_reprojection=args.feedback_pose_guard_keep_gt_reprojection,
            allow_geometry_grad=not args.no_geometry_grad,
            geometry_reprojection_weight=args.geometry_reprojection_weight,
            geometry_depth_anchor_weight=args.geometry_depth_anchor_weight,
            geometry_xyz_lr=args.geometry_xyz_lr,
            loc_anchor_lr=args.loc_anchor_lr,
            surfel_loc_tangent_bound=args.surfel_loc_tangent_bound,
            surfel_loc_normal_bound=args.surfel_loc_normal_bound,
            surfel_loc_radius_floor=args.surfel_loc_radius_floor,
            surfel_loc_anchor_reg_weight=args.surfel_loc_anchor_reg_weight,
            rgb_densify_child_max_source_drift=args.rgb_densify_child_max_source_drift,
            geometry_residual=args.geometry_residual,
            geometry_residual_weight=args.geometry_residual_weight,
            geometry_residual_max_scale_ratio=args.geometry_residual_max_scale_ratio,
            geometry_grad_clip_abs=args.geometry_grad_clip_abs,
            direct_pnp_xyz_grad=args.direct_pnp_xyz_grad,
            allow_raw_xyz_geometry_grad=args.allow_raw_xyz_geometry_grad,
            lafgs_detector_source=args.lafgs_detector_source,
            baseline_detector_folder=args.baseline_detector_folder,
            lafgs_detector_folder=args.lafgs_detector_folder,
            lafgs_detector_iterations=args.lafgs_detector_iterations,
            lafgs_detector_landmark_num=args.lafgs_detector_landmark_num,
            lafgs_detector_landmark_k=args.lafgs_detector_landmark_k,
            lafgs_detector_sampling_mode=args.lafgs_detector_sampling_mode,
            lafgs_detector_target_mode=args.lafgs_detector_target_mode,
            lafgs_detector_min_loc_observations=args.lafgs_detector_min_loc_observations,
            lafgs_detector_utility_weight=args.lafgs_detector_utility_weight,
            lafgs_detector_pnp_voxel_size=args.lafgs_detector_pnp_voxel_size,
            lafgs_detector_pnp_max_per_voxel=args.lafgs_detector_pnp_max_per_voxel,
            lafgs_detector_pnp_preserve_ratio=args.lafgs_detector_pnp_preserve_ratio,
            lafgs_detector_soft_sigma=args.lafgs_detector_soft_sigma,
            lafgs_detector_coverage_preserve_ratio=args.lafgs_detector_coverage_preserve_ratio,
            lafgs_detector_coverage_utility_ratio=args.lafgs_detector_coverage_utility_ratio,
            lafgs_detector_coverage_high_confidence_ratio=args.lafgs_detector_coverage_high_confidence_ratio,
            lafgs_detector_coverage_grid_size=args.lafgs_detector_coverage_grid_size,
            lafgs_detector_coverage_max_per_grid=args.lafgs_detector_coverage_max_per_grid,
            lafgs_detector_coverage_depth_bins=args.lafgs_detector_coverage_depth_bins,
            lafgs_detector_coverage_max_per_depth_bin=args.lafgs_detector_coverage_max_per_depth_bin,
            checkpoint_eval_interval=args.checkpoint_eval_interval,
            checkpoint_detector_iterations=args.checkpoint_detector_iterations,
            lafgs_from_sfm_zero=args.lafgs_from_sfm_zero,
            eval_diagnostics=not args.no_eval_diagnostics,
            eval_diagnostics_dump_correspondences=args.eval_diagnostics_dump_correspondences,
            eval_diagnostics_grid_rows=args.eval_diagnostics_grid_rows,
            eval_diagnostics_grid_cols=args.eval_diagnostics_grid_cols,
            eval_diagnostics_voxel_size=args.eval_diagnostics_voxel_size,
            eval_geometry_balance=args.eval_geometry_balance,
            eval_geometry_balance_grid_rows=args.eval_geometry_balance_grid_rows,
            eval_geometry_balance_grid_cols=args.eval_geometry_balance_grid_cols,
            eval_geometry_balance_max_per_cell=args.eval_geometry_balance_max_per_cell,
            eval_geometry_balance_voxel_size=args.eval_geometry_balance_voxel_size,
            eval_geometry_balance_max_per_voxel=args.eval_geometry_balance_max_per_voxel,
            eval_geometry_balance_max_matches=args.eval_geometry_balance_max_matches,
        )
        for scene in args.scenes
    ]
    manifest = {
        "output_root": str(output_root),
        "results_root": args.results_root,
        "scenes": [plan.to_json() for plan in plans],
    }
    manifest_path = Path(args.manifest) if args.manifest else output_root / "lafgs_cambridge_guarded_pnp_manifest.json"
    if args.dry_run:
        print(json.dumps(manifest, indent=2), flush=True)
    for plan in plans:
        run_plan(
            plan,
            cwd=Path(__file__).resolve().parents[1],
            dry_run=args.dry_run,
            force_init=args.force_init,
            force_train=args.force_train,
            train_missing_baseline=args.train_missing_baseline,
            skip_train=args.skip_train,
            skip_detector_train=args.skip_detector_train,
            skip_eval=args.skip_eval,
            cuda_visible_devices=args.cuda_visible_devices,
        )
    if not args.dry_run:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2))
        print(f"[lafgs-cambridge] wrote manifest: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
