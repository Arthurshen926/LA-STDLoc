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


@dataclass
class ScenePlan:
    scene: str
    data_dir: Path
    baseline_model: Path
    lafgs_model: Path
    detector_source: Path
    baseline_iteration: int
    final_iteration: int
    status: str
    missing_reasons: list
    train_baseline_command: list
    train_lafgs_command: list
    baseline_eval_cfg_command: list
    baseline_eval_command: list
    lafgs_eval_cfg_command: list
    lafgs_eval_command: list

    def to_json(self):
        data = asdict(self)
        for key, value in list(data.items()):
            if isinstance(value, Path):
                data[key] = str(value)
        return data


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
    mvinit_feature_scale=0.5,
    mvinit_max_views=64,
    mvinit_chunk_size=32768,
    pose_information_weight=0.0,
    pose_information_floor=0.0,
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
    geometry_max_reprojection_error=12.0,
    geometry_use_all_correspondences=True,
    geometry_match_reprojection_weight=0.0,
    geometry_match_confidence_threshold=-1.0,
    geometry_match_margin_threshold=-1.0,
    geometry_match_peak_probability_threshold=-1.0,
    geometry_match_max_entropy=-1.0,
    geometry_match_max_reprojection_error=-1.0,
    utility_pose_loss_scale=1.0,
    utility_reprojection_error_scale=4.0,
    pnp_local_window_radius=0.0,
    max_condition_number=-1.0,
    feedback_pose_guard_keep_gt_reprojection=False,
    allow_geometry_grad=True,
    geometry_reprojection_weight=1.0,
    geometry_depth_anchor_weight=0.0,
    geometry_xyz_lr=0.00002,
    loc_anchor_lr=0.0,
    surfel_loc_tangent_bound=0.0,
    surfel_loc_normal_bound=0.0,
    surfel_loc_anchor_reg_weight=0.0,
    detach_pnp_points=True,
):
    command = [
        python,
        "train_lafgs.py",
        *_common_data_args(data_dir, gaussian_type=gaussian_type),
        *_common_train_args(),
        "-m",
        str(lafgs_model),
        "--load_iteration",
        str(baseline_iterations),
        "--iterations",
        str(final_iteration),
        "--train_phase",
        "full",
        "--loc_interval",
        "8",
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
        "--lafgs_curriculum",
        "--lafgs_diff_pnp_start_iter",
        "400",
        "--lafgs_diff_pnp_weight",
        "0.0005",
        "--lafgs_diff_pnp_max_correspondences",
        "64",
        "--lafgs_diff_pnp_spatial_grid_size",
        "4",
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
        str(final_iteration),
        "--test_iterations",
        str(final_iteration),
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
):
    return [
        python,
        "scripts/make_stdloc_eval_cfg.py",
        "--base_cfg",
        str(cfg),
        "--output",
        str(output),
        "--artifact_model_path",
        str(artifact_model_path),
        "--detector_folder",
        "detector",
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
    mvinit_feature_scale=0.5,
    mvinit_max_views=64,
    mvinit_chunk_size=32768,
    pose_information_weight=0.0,
    pose_information_floor=0.0,
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
    geometry_max_reprojection_error=12.0,
    geometry_use_all_correspondences=True,
    geometry_match_reprojection_weight=0.0,
    geometry_match_confidence_threshold=-1.0,
    geometry_match_margin_threshold=-1.0,
    geometry_match_peak_probability_threshold=-1.0,
    geometry_match_max_entropy=-1.0,
    geometry_match_max_reprojection_error=-1.0,
    utility_pose_loss_scale=1.0,
    utility_reprojection_error_scale=4.0,
    pnp_local_window_radius=0.0,
    max_condition_number=-1.0,
    feedback_pose_guard_keep_gt_reprojection=False,
    allow_geometry_grad=True,
    geometry_reprojection_weight=1.0,
    geometry_depth_anchor_weight=0.0,
    geometry_xyz_lr=0.00002,
    loc_anchor_lr=0.0,
    surfel_loc_tangent_bound=0.0,
    surfel_loc_normal_bound=0.0,
    surfel_loc_anchor_reg_weight=0.0,
    direct_pnp_xyz_grad=False,
):
    data_root = Path(data_root)
    baseline_root = Path(baseline_root)
    output_root = Path(output_root)
    cfg = Path(cfg)
    data_dir = data_root / scene
    baseline_model = baseline_root / f"{scene}_baseline"
    lafgs_model = output_root / scene
    final_iteration = int(baseline_iterations) + int(lafgs_steps)
    detector_source = baseline_model / "detector"
    baseline_point_cloud = baseline_model / "point_cloud" / f"iteration_{baseline_iterations}" / "point_cloud.ply"
    detector_path = detector_source / f"{baseline_iterations}_detector.pth"
    sampled_idx = detector_source / "sampled_idx.pkl"
    lafgs_final = lafgs_model / "point_cloud" / f"iteration_{final_iteration}" / "point_cloud.ply"

    missing = []
    if not data_dir.is_dir():
        missing.append(f"missing data directory: {data_dir}")
    baseline_ready = baseline_point_cloud.is_file() and detector_path.is_file() and sampled_idx.is_file()
    if not baseline_ready and not train_missing_baseline:
        missing.append(f"missing baseline artifacts under: {baseline_model}")
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
    baseline_eval_cfg = eval_dir / "baseline_det8192_nms2_reproj12.yaml"
    lafgs_eval_cfg = eval_dir / "lafgs_guarded_pnp_det8192_nms2_reproj12.yaml"

    return ScenePlan(
        scene=scene,
        data_dir=data_dir,
        baseline_model=baseline_model,
        lafgs_model=lafgs_model,
        detector_source=detector_source,
        baseline_iteration=int(baseline_iterations),
        final_iteration=final_iteration,
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
            mvinit_feature_scale=mvinit_feature_scale,
            mvinit_max_views=mvinit_max_views,
            mvinit_chunk_size=mvinit_chunk_size,
            pose_information_weight=pose_information_weight,
            pose_information_floor=pose_information_floor,
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
            surfel_loc_anchor_reg_weight=surfel_loc_anchor_reg_weight,
            detach_pnp_points=not direct_pnp_xyz_grad,
        ),
        baseline_eval_cfg_command=[]
        if skip_eval or not eval_baseline
        else _eval_cfg_command(
            python,
            cfg,
            baseline_eval_cfg,
            baseline_model,
            baseline_iterations,
            detect_num,
            nms,
            reprojection_error,
        ),
        baseline_eval_command=[]
        if skip_eval or not eval_baseline
        else _eval_command(
            python,
            data_dir,
            baseline_model,
            baseline_iterations,
            baseline_eval_cfg,
            f"baseline-{baseline_iterations}",
            gaussian_type=gaussian_type,
        ),
        lafgs_eval_cfg_command=[]
        if skip_eval
        else _eval_cfg_command(
            python,
            cfg,
            lafgs_eval_cfg,
            lafgs_model,
            baseline_iterations,
            detect_num,
            nms,
            reprojection_error,
        ),
        lafgs_eval_command=[]
        if skip_eval
        else _eval_command(
            python,
            data_dir,
            lafgs_model,
            final_iteration,
            lafgs_eval_cfg,
            f"lafgs-guarded-pnp-{final_iteration}",
            gaussian_type=gaussian_type,
        ),
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
    env["CUDA_HOME"] = cuda_home
    env["PATH"] = f"{cuda_home}/bin:{env.get('PATH', '')}"
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
    skip_eval,
    cuda_visible_devices=None,
):
    if plan.missing_reasons and not train_missing_baseline:
        print(f"[lafgs-cambridge] skip {plan.scene}: {'; '.join(plan.missing_reasons)}", flush=True)
        return "skipped"

    env = _env(cuda_visible_devices=cuda_visible_devices)
    baseline_ready = (
        (plan.baseline_model / "point_cloud" / f"iteration_{plan.baseline_iteration}" / "point_cloud.ply").is_file()
        and (plan.detector_source / f"{plan.baseline_iteration}_detector.pth").is_file()
        and (plan.detector_source / "sampled_idx.pkl").is_file()
    )
    if train_missing_baseline and not baseline_ready:
        _run(plan.train_baseline_command, cwd=cwd, env=env, dry_run=dry_run)

    _ensure_symlink(
        plan.baseline_model / "point_cloud" / f"iteration_{plan.baseline_iteration}",
        plan.lafgs_model / "point_cloud" / f"iteration_{plan.baseline_iteration}",
        dry_run=dry_run,
        force=force_init,
    )
    _ensure_symlink(plan.detector_source, plan.lafgs_model / "detector", dry_run=dry_run, force=force_init)

    final_ply = plan.lafgs_model / "point_cloud" / f"iteration_{plan.final_iteration}" / "point_cloud.ply"
    if not skip_train and (force_train or not final_ply.is_file()):
        _run(plan.train_lafgs_command, cwd=cwd, env=env, dry_run=dry_run)
    else:
        print(f"[lafgs-cambridge] skip LaFGS train for {plan.scene}: found {final_ply}", flush=True)

    if not skip_eval:
        if not dry_run:
            (plan.lafgs_model / "eval_configs").mkdir(parents=True, exist_ok=True)
        for command in (
            plan.baseline_eval_cfg_command,
            plan.baseline_eval_command,
            plan.lafgs_eval_cfg_command,
            plan.lafgs_eval_command,
        ):
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
    parser.add_argument("--lafgs_steps", type=int, default=500)
    parser.add_argument("--detect_num", type=int, default=8192)
    parser.add_argument("--nms", type=int, default=2)
    parser.add_argument("--reprojection_error", type=float, default=12.0)
    parser.add_argument("--gaussian_type", choices=["3dgs", "2dgs"], default="3dgs")
    parser.add_argument("--mvinit_feature_scale", type=float, default=0.5)
    parser.add_argument("--mvinit_max_views", type=int, default=64)
    parser.add_argument("--mvinit_chunk_size", type=int, default=32768)
    parser.add_argument("--pose_information_weight", type=float, default=0.0)
    parser.add_argument("--pose_information_floor", type=float, default=0.0)
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
    parser.add_argument("--geometry_max_reprojection_error", type=float, default=12.0)
    parser.add_argument(
        "--geometry_selected_correspondences",
        action="store_true",
        help="Use only the selected PnP correspondences for geometry and descriptor-match feedback.",
    )
    parser.add_argument("--geometry_match_reprojection_weight", type=float, default=0.0)
    parser.add_argument("--geometry_match_confidence_threshold", type=float, default=-1.0)
    parser.add_argument("--geometry_match_margin_threshold", type=float, default=-1.0)
    parser.add_argument("--geometry_match_peak_probability_threshold", type=float, default=-1.0)
    parser.add_argument("--geometry_match_max_entropy", type=float, default=-1.0)
    parser.add_argument("--geometry_match_max_reprojection_error", type=float, default=-1.0)
    parser.add_argument("--utility_pose_loss_scale", type=float, default=1.0)
    parser.add_argument("--utility_reprojection_error_scale", type=float, default=4.0)
    parser.add_argument("--pnp_local_window_radius", type=float, default=0.0)
    parser.add_argument("--max_condition_number", type=float, default=-1.0)
    parser.add_argument("--feedback_pose_guard_keep_gt_reprojection", action="store_true")
    parser.add_argument("--no_geometry_grad", action="store_true")
    parser.add_argument("--geometry_reprojection_weight", type=float, default=1.0)
    parser.add_argument("--geometry_depth_anchor_weight", type=float, default=0.0)
    parser.add_argument("--geometry_xyz_lr", type=float, default=0.00002)
    parser.add_argument("--loc_anchor_lr", type=float, default=0.0)
    parser.add_argument("--surfel_loc_tangent_bound", type=float, default=0.0)
    parser.add_argument("--surfel_loc_normal_bound", type=float, default=0.0)
    parser.add_argument("--surfel_loc_anchor_reg_weight", type=float, default=0.0)
    parser.add_argument(
        "--direct_pnp_xyz_grad",
        action="store_true",
        help="Let differentiable PnP pose loss backpropagate through soft PnP 3D points to Gaussian xyz.",
    )
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
    args = parse_args(argv)
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
            lafgs_steps=args.lafgs_steps,
            detect_num=args.detect_num,
            nms=args.nms,
            reprojection_error=args.reprojection_error,
            train_missing_baseline=args.train_missing_baseline,
            force_train=args.force_train,
            skip_train=args.skip_train,
            skip_eval=args.skip_eval,
            eval_baseline=not args.no_eval_baseline,
            cfg=args.cfg,
            gaussian_type=args.gaussian_type,
            mvinit_feature_scale=args.mvinit_feature_scale,
            mvinit_max_views=args.mvinit_max_views,
            mvinit_chunk_size=args.mvinit_chunk_size,
            pose_information_weight=args.pose_information_weight,
            pose_information_floor=args.pose_information_floor,
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
            geometry_use_all_correspondences=not args.geometry_selected_correspondences,
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
            surfel_loc_anchor_reg_weight=args.surfel_loc_anchor_reg_weight,
            direct_pnp_xyz_grad=args.direct_pnp_xyz_grad,
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
            skip_eval=args.skip_eval,
            cuda_visible_devices=args.cuda_visible_devices,
        )
    if not args.dry_run:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2))
        print(f"[lafgs-cambridge] wrote manifest: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
