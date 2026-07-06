import argparse
import sys


def build_parser():
    parser = argparse.ArgumentParser(description="LaFGS localization-aware reconstruction defaults")
    parser.add_argument("--lafgs_mv_init_until", type=int, default=0)
    parser.add_argument("--lafgs_mvinit_enabled", action="store_true", default=False)
    parser.add_argument("--lafgs_mvinit_max_views", type=int, default=16)
    parser.add_argument("--lafgs_mvinit_view_selection", choices=["first", "uniform"], default=None)
    parser.add_argument("--lafgs_mvinit_chunk_size", type=int, default=32768)
    parser.add_argument("--lafgs_mvinit_feature_scale", type=float, default=None)
    parser.add_argument("--lafgs_curriculum", action="store_true", default=False)
    parser.add_argument("--lafgs_locrec_start_iter", type=int, default=1)
    parser.add_argument("--lafgs_diff_pnp_start_iter", type=int, default=5_000)
    parser.add_argument("--lafgs_geometry_start_iter", type=int, default=10_000)
    parser.add_argument("--lafgs_topology_start_iter", type=int, default=15_000)
    parser.add_argument("--lafgs_diff_pnp_weight", type=float, default=0.0)
    parser.add_argument(
        "--lafgs_diff_pnp_reprojection_loss_type",
        choices=["smooth_l1", "huber", "cauchy"],
        default=None,
    )
    parser.add_argument("--lafgs_diff_pnp_reprojection_loss_delta", type=float, default=None)
    parser.add_argument("--lafgs_diff_pnp_point_weight_floor", type=float, default=None)
    parser.add_argument("--lafgs_diff_pnp_utility_pose_loss_scale", type=float, default=None)
    parser.add_argument("--lafgs_diff_pnp_utility_reprojection_error_scale", type=float, default=None)
    parser.add_argument("--lafgs_diff_pnp_local_window_radius", type=float, default=None)
    parser.add_argument("--lafgs_diff_pnp_geometry_xyz_lr", type=float, default=None)
    parser.add_argument("--lafgs_diff_pnp_isolate_geometry_grad", action="store_true", default=None)
    parser.add_argument("--lafgs_diff_pnp_geometry_reproj_weight", type=float, default=None)
    parser.add_argument("--lafgs_diff_pnp_geometry_depth_anchor_weight", type=float, default=None)
    parser.add_argument("--lafgs_diff_pnp_geometry_match_reproj_weight", type=float, default=None)
    parser.add_argument("--lafgs_diff_pnp_geometry_match_confidence_threshold", type=float, default=None)
    parser.add_argument("--lafgs_diff_pnp_geometry_match_margin_threshold", type=float, default=None)
    parser.add_argument("--lafgs_diff_pnp_geometry_match_peak_probability_threshold", type=float, default=None)
    parser.add_argument("--lafgs_diff_pnp_geometry_match_max_entropy", type=float, default=None)
    parser.add_argument("--lafgs_diff_pnp_geometry_match_max_reproj_error", type=float, default=None)
    parser.add_argument("--lafgs_diff_pnp_geometry_confidence_threshold", type=float, default=None)
    parser.add_argument("--lafgs_diff_pnp_geometry_margin_threshold", type=float, default=None)
    parser.add_argument("--lafgs_diff_pnp_geometry_peak_probability_threshold", type=float, default=None)
    parser.add_argument("--lafgs_diff_pnp_geometry_max_entropy", type=float, default=None)
    parser.add_argument("--lafgs_diff_pnp_geometry_max_reproj_error", type=float, default=None)
    parser.add_argument("--lafgs_diff_pnp_geometry_use_all_correspondences", action="store_true", default=None)
    parser.add_argument("--lafgs_diff_pnp_geometry_local_window_radius", type=float, default=None)
    parser.add_argument("--lafgs_diff_pnp_max_condition_number", type=float, default=None)
    parser.add_argument("--lafgs_diff_pnp_geometry_pose_guard_max_loss_increase", type=float, default=None)
    parser.add_argument("--lafgs_diff_pnp_geometry_pose_guard_max_loss", type=float, default=None)
    parser.add_argument("--lafgs_diff_pnp_geometry_pose_guard_softness", type=float, default=None)
    parser.add_argument("--lafgs_diff_pnp_geometry_pose_guard_min_scale", type=float, default=None)
    parser.add_argument("--lafgs_diff_pnp_feedback_pose_guard_max_loss_increase", type=float, default=None)
    parser.add_argument("--lafgs_diff_pnp_feedback_pose_guard_max_loss", type=float, default=None)
    parser.add_argument("--lafgs_diff_pnp_feedback_pose_guard_softness", type=float, default=None)
    parser.add_argument("--lafgs_diff_pnp_feedback_pose_guard_min_scale", type=float, default=None)
    parser.add_argument("--lafgs_diff_pnp_feedback_pose_guard_keep_gt_reprojection", action="store_true", default=None)
    parser.add_argument("--lafgs_diff_pnp_detach_pnp_points", action="store_true", default=None)
    if hasattr(argparse, "BooleanOptionalAction"):
        parser.add_argument(
            "--lafgs_diff_pnp_detach_gt_reprojection_points",
            action=argparse.BooleanOptionalAction,
            default=None,
        )
    else:
        parser.add_argument(
            "--lafgs_diff_pnp_detach_gt_reprojection_points",
            dest="lafgs_diff_pnp_detach_gt_reprojection_points",
            action="store_true",
        )
        parser.add_argument(
            "--no-lafgs_diff_pnp_detach_gt_reprojection_points",
            dest="lafgs_diff_pnp_detach_gt_reprojection_points",
            action="store_false",
        )
        parser.set_defaults(lafgs_diff_pnp_detach_gt_reprojection_points=None)
    if hasattr(argparse, "BooleanOptionalAction"):
        parser.add_argument(
            "--lafgs_diff_pnp_use_loc_opacity_weight",
            action=argparse.BooleanOptionalAction,
            default=None,
        )
    else:
        parser.add_argument(
            "--lafgs_diff_pnp_use_loc_opacity_weight",
            dest="lafgs_diff_pnp_use_loc_opacity_weight",
            action="store_true",
        )
        parser.add_argument(
            "--no-lafgs_diff_pnp_use_loc_opacity_weight",
            dest="lafgs_diff_pnp_use_loc_opacity_weight",
            action="store_false",
        )
        parser.set_defaults(lafgs_diff_pnp_use_loc_opacity_weight=None)
    parser.add_argument("--lafgs_geometry_residual", action="store_true", default=False)
    parser.add_argument("--lafgs_synthetic_feature_source", choices=["rgb", "loc_feature"], default=None)
    parser.add_argument("--loc_direct_weight", type=float, default=None)
    parser.add_argument("--loc_multiview_weight", type=float, default=None)
    parser.add_argument("--loc_full_bank_weight", type=float, default=None)
    parser.add_argument("--geometry_anchor_weight", type=float, default=None)
    parser.add_argument("--synthetic_view_ratio", type=float, default=None)
    parser.add_argument("--synthetic_view_desc_weight", type=float, default=None)
    parser.add_argument("--synthetic_view_reproj_weight", type=float, default=None)
    return parser


def _setdefault(args, name, value):
    if not hasattr(args, name) or getattr(args, name) is None:
        setattr(args, name, value)


def _set_if_missing_or_legacy(args, name, value, legacy_values):
    current = getattr(args, name, None)
    if current is None or current in legacy_values:
        setattr(args, name, value)


def _explicit_lafgs_overrides(argv):
    overrides = set()
    for index, item in enumerate(argv):
        normalized_item = item
        if normalized_item.startswith("--no-lafgs_"):
            normalized_item = "--" + normalized_item[len("--no-") :]
        if (
            normalized_item.startswith("--lafgs_")
            or normalized_item.startswith("--synthetic_view_")
            or normalized_item.startswith("--loc_")
            or normalized_item.startswith("--geometry_")
        ):
            name = normalized_item[2:].split("=", 1)[0].replace("-", "_")
            overrides.add(name)
        if item == "--lafgs_synthetic_feature_source":
            if index + 1 < len(argv):
                overrides.add("lafgs_synthetic_feature_source")
        elif item.startswith("--lafgs_synthetic_feature_source="):
            overrides.add("lafgs_synthetic_feature_source")
    return overrides


def lafgs_defaults(args, explicit_overrides=None):
    explicit_overrides = set(explicit_overrides or ())
    _set_if_missing_or_legacy(args, "loc_teacher", "direct", {"dense"})
    _set_if_missing_or_legacy(args, "localization_enabled", True, {False})
    _set_if_missing_or_legacy(args, "use_loc_opacity", True, {False})
    if "loc_direct_weight" not in explicit_overrides:
        _set_if_missing_or_legacy(args, "loc_direct_weight", 1.0, {0.1})
    else:
        _setdefault(args, "loc_direct_weight", 1.0)
    if "loc_multiview_weight" not in explicit_overrides:
        _set_if_missing_or_legacy(args, "loc_multiview_weight", 0.1, {0.05})
    else:
        _setdefault(args, "loc_multiview_weight", 0.1)
    if "loc_full_bank_weight" not in explicit_overrides:
        _set_if_missing_or_legacy(args, "loc_full_bank_weight", 0.1, {0.0})
    else:
        _setdefault(args, "loc_full_bank_weight", 0.1)
    _set_if_missing_or_legacy(args, "loc_full_bank_hard_negatives", 64, {32})
    _setdefault(args, "loc_full_bank_source_mode", "ignore")
    _set_if_missing_or_legacy(args, "direct_depth_check", True, {False})
    if "geometry_anchor_weight" not in explicit_overrides:
        _set_if_missing_or_legacy(args, "geometry_anchor_weight", 0.05, {0.0})
    else:
        _setdefault(args, "geometry_anchor_weight", 0.05)
    _setdefault(args, "geometry_anchor_scale_weight", 0.1)
    _setdefault(args, "geometry_anchor_rotation_weight", 0.1)
    _set_if_missing_or_legacy(args, "loc_opacity_weight", 0.01, {0.0})
    _set_if_missing_or_legacy(args, "loc_opacity_target", 0.35, {0.5})
    if "synthetic_view_ratio" not in explicit_overrides:
        _set_if_missing_or_legacy(args, "synthetic_view_ratio", 0.1, {0.0})
    else:
        _setdefault(args, "synthetic_view_ratio", 0.1)
    if "synthetic_view_desc_weight" not in explicit_overrides:
        _set_if_missing_or_legacy(args, "synthetic_view_desc_weight", 0.25, {0.0})
    else:
        _setdefault(args, "synthetic_view_desc_weight", 0.25)
    if "synthetic_view_reproj_weight" not in explicit_overrides:
        _set_if_missing_or_legacy(args, "synthetic_view_reproj_weight", 0.05, {0.0})
    else:
        _setdefault(args, "synthetic_view_reproj_weight", 0.05)
    if "lafgs_synthetic_feature_source" not in explicit_overrides:
        _set_if_missing_or_legacy(args, "lafgs_synthetic_feature_source", "rgb", {"loc_feature"})
    else:
        _setdefault(args, "lafgs_synthetic_feature_source", "rgb")
    _setdefault(args, "lafgs_mv_init_until", 0)
    _set_if_missing_or_legacy(args, "lafgs_mvinit_enabled", True, {False})
    if "lafgs_mvinit_max_views" not in explicit_overrides:
        _set_if_missing_or_legacy(args, "lafgs_mvinit_max_views", 16, {0})
    else:
        _setdefault(args, "lafgs_mvinit_max_views", 16)
    if "lafgs_mvinit_view_selection" not in explicit_overrides:
        _setdefault(args, "lafgs_mvinit_view_selection", "uniform")
    else:
        _setdefault(args, "lafgs_mvinit_view_selection", "uniform")
    _set_if_missing_or_legacy(args, "lafgs_mvinit_chunk_size", 32768, {0})
    _setdefault(args, "lafgs_mvinit_feature_scale", 1.0)
    _set_if_missing_or_legacy(args, "lafgs_curriculum", True, {False})
    _set_if_missing_or_legacy(args, "lafgs_locrec_start_iter", 1, {0})
    _set_if_missing_or_legacy(args, "lafgs_diff_pnp_start_iter", 5_000, {0})
    _setdefault(args, "lafgs_geometry_start_iter", 10_000)
    _setdefault(args, "lafgs_topology_start_iter", 15_000)
    if "lafgs_diff_pnp_weight" not in explicit_overrides:
        _set_if_missing_or_legacy(args, "lafgs_diff_pnp_weight", 0.05, {0.0})
    else:
        _setdefault(args, "lafgs_diff_pnp_weight", 0.05)
    if "lafgs_diff_pnp_reprojection_loss_type" not in explicit_overrides:
        _setdefault(args, "lafgs_diff_pnp_reprojection_loss_type", "smooth_l1")
    else:
        _setdefault(args, "lafgs_diff_pnp_reprojection_loss_type", "smooth_l1")
    if "lafgs_diff_pnp_reprojection_loss_delta" not in explicit_overrides:
        _setdefault(args, "lafgs_diff_pnp_reprojection_loss_delta", 1.0)
    else:
        _setdefault(args, "lafgs_diff_pnp_reprojection_loss_delta", 1.0)
    if "lafgs_diff_pnp_use_loc_opacity_weight" not in explicit_overrides:
        _setdefault(args, "lafgs_diff_pnp_use_loc_opacity_weight", False)
    else:
        _setdefault(args, "lafgs_diff_pnp_use_loc_opacity_weight", False)
    if "lafgs_diff_pnp_point_weight_floor" not in explicit_overrides:
        _set_if_missing_or_legacy(args, "lafgs_diff_pnp_point_weight_floor", 0.75, {0.0})
    else:
        _setdefault(args, "lafgs_diff_pnp_point_weight_floor", 0.75)
    if "lafgs_diff_pnp_utility_pose_loss_scale" not in explicit_overrides:
        _setdefault(args, "lafgs_diff_pnp_utility_pose_loss_scale", 1.0)
    else:
        _setdefault(args, "lafgs_diff_pnp_utility_pose_loss_scale", 1.0)
    if "lafgs_diff_pnp_utility_reprojection_error_scale" not in explicit_overrides:
        _setdefault(args, "lafgs_diff_pnp_utility_reprojection_error_scale", 4.0)
    else:
        _setdefault(args, "lafgs_diff_pnp_utility_reprojection_error_scale", 4.0)
    if "lafgs_diff_pnp_local_window_radius" not in explicit_overrides:
        _setdefault(args, "lafgs_diff_pnp_local_window_radius", 0.0)
    else:
        _setdefault(args, "lafgs_diff_pnp_local_window_radius", 0.0)
    if "lafgs_diff_pnp_geometry_xyz_lr" not in explicit_overrides:
        _setdefault(args, "lafgs_diff_pnp_geometry_xyz_lr", 0.0)
    else:
        _setdefault(args, "lafgs_diff_pnp_geometry_xyz_lr", 0.0)
    if "lafgs_diff_pnp_isolate_geometry_grad" not in explicit_overrides:
        _setdefault(args, "lafgs_diff_pnp_isolate_geometry_grad", False)
    else:
        _setdefault(args, "lafgs_diff_pnp_isolate_geometry_grad", False)
    if "lafgs_diff_pnp_geometry_reproj_weight" not in explicit_overrides:
        _setdefault(args, "lafgs_diff_pnp_geometry_reproj_weight", 0.0)
    else:
        _setdefault(args, "lafgs_diff_pnp_geometry_reproj_weight", 0.0)
    if "lafgs_diff_pnp_geometry_depth_anchor_weight" not in explicit_overrides:
        _setdefault(args, "lafgs_diff_pnp_geometry_depth_anchor_weight", 0.0)
    else:
        _setdefault(args, "lafgs_diff_pnp_geometry_depth_anchor_weight", 0.0)
    if "lafgs_diff_pnp_geometry_match_reproj_weight" not in explicit_overrides:
        _setdefault(args, "lafgs_diff_pnp_geometry_match_reproj_weight", 0.0)
    else:
        _setdefault(args, "lafgs_diff_pnp_geometry_match_reproj_weight", 0.0)
    if "lafgs_diff_pnp_geometry_match_confidence_threshold" not in explicit_overrides:
        _setdefault(args, "lafgs_diff_pnp_geometry_match_confidence_threshold", -1.0)
    else:
        _setdefault(args, "lafgs_diff_pnp_geometry_match_confidence_threshold", -1.0)
    if "lafgs_diff_pnp_geometry_match_margin_threshold" not in explicit_overrides:
        _setdefault(args, "lafgs_diff_pnp_geometry_match_margin_threshold", -1.0)
    else:
        _setdefault(args, "lafgs_diff_pnp_geometry_match_margin_threshold", -1.0)
    if "lafgs_diff_pnp_geometry_match_peak_probability_threshold" not in explicit_overrides:
        _setdefault(args, "lafgs_diff_pnp_geometry_match_peak_probability_threshold", -1.0)
    else:
        _setdefault(args, "lafgs_diff_pnp_geometry_match_peak_probability_threshold", -1.0)
    if "lafgs_diff_pnp_geometry_match_max_entropy" not in explicit_overrides:
        _setdefault(args, "lafgs_diff_pnp_geometry_match_max_entropy", -1.0)
    else:
        _setdefault(args, "lafgs_diff_pnp_geometry_match_max_entropy", -1.0)
    if "lafgs_diff_pnp_geometry_match_max_reproj_error" not in explicit_overrides:
        _setdefault(args, "lafgs_diff_pnp_geometry_match_max_reproj_error", -1.0)
    else:
        _setdefault(args, "lafgs_diff_pnp_geometry_match_max_reproj_error", -1.0)
    if "lafgs_diff_pnp_geometry_confidence_threshold" not in explicit_overrides:
        _setdefault(args, "lafgs_diff_pnp_geometry_confidence_threshold", 0.0)
    else:
        _setdefault(args, "lafgs_diff_pnp_geometry_confidence_threshold", 0.0)
    if "lafgs_diff_pnp_geometry_margin_threshold" not in explicit_overrides:
        _setdefault(args, "lafgs_diff_pnp_geometry_margin_threshold", 0.0)
    else:
        _setdefault(args, "lafgs_diff_pnp_geometry_margin_threshold", 0.0)
    if "lafgs_diff_pnp_geometry_peak_probability_threshold" not in explicit_overrides:
        _setdefault(args, "lafgs_diff_pnp_geometry_peak_probability_threshold", 0.0)
    else:
        _setdefault(args, "lafgs_diff_pnp_geometry_peak_probability_threshold", 0.0)
    if "lafgs_diff_pnp_geometry_max_entropy" not in explicit_overrides:
        _setdefault(args, "lafgs_diff_pnp_geometry_max_entropy", 0.0)
    else:
        _setdefault(args, "lafgs_diff_pnp_geometry_max_entropy", 0.0)
    if "lafgs_diff_pnp_geometry_max_reproj_error" not in explicit_overrides:
        _setdefault(args, "lafgs_diff_pnp_geometry_max_reproj_error", 0.0)
    else:
        _setdefault(args, "lafgs_diff_pnp_geometry_max_reproj_error", 0.0)
    if "lafgs_diff_pnp_geometry_use_all_correspondences" not in explicit_overrides:
        _setdefault(args, "lafgs_diff_pnp_geometry_use_all_correspondences", False)
    else:
        _setdefault(args, "lafgs_diff_pnp_geometry_use_all_correspondences", False)
    if "lafgs_diff_pnp_geometry_local_window_radius" not in explicit_overrides:
        _setdefault(args, "lafgs_diff_pnp_geometry_local_window_radius", 0.0)
    else:
        _setdefault(args, "lafgs_diff_pnp_geometry_local_window_radius", 0.0)
    if "lafgs_diff_pnp_max_condition_number" not in explicit_overrides:
        _setdefault(args, "lafgs_diff_pnp_max_condition_number", -1.0)
    else:
        _setdefault(args, "lafgs_diff_pnp_max_condition_number", -1.0)
    if "lafgs_diff_pnp_geometry_pose_guard_max_loss_increase" not in explicit_overrides:
        _setdefault(args, "lafgs_diff_pnp_geometry_pose_guard_max_loss_increase", -1.0)
    else:
        _setdefault(args, "lafgs_diff_pnp_geometry_pose_guard_max_loss_increase", -1.0)
    if "lafgs_diff_pnp_geometry_pose_guard_max_loss" not in explicit_overrides:
        _setdefault(args, "lafgs_diff_pnp_geometry_pose_guard_max_loss", -1.0)
    else:
        _setdefault(args, "lafgs_diff_pnp_geometry_pose_guard_max_loss", -1.0)
    if "lafgs_diff_pnp_geometry_pose_guard_softness" not in explicit_overrides:
        _setdefault(args, "lafgs_diff_pnp_geometry_pose_guard_softness", 0.0)
    else:
        _setdefault(args, "lafgs_diff_pnp_geometry_pose_guard_softness", 0.0)
    if "lafgs_diff_pnp_geometry_pose_guard_min_scale" not in explicit_overrides:
        _setdefault(args, "lafgs_diff_pnp_geometry_pose_guard_min_scale", 0.0)
    else:
        _setdefault(args, "lafgs_diff_pnp_geometry_pose_guard_min_scale", 0.0)
    if "lafgs_diff_pnp_feedback_pose_guard_max_loss_increase" not in explicit_overrides:
        _setdefault(args, "lafgs_diff_pnp_feedback_pose_guard_max_loss_increase", -1.0)
    else:
        _setdefault(args, "lafgs_diff_pnp_feedback_pose_guard_max_loss_increase", -1.0)
    if "lafgs_diff_pnp_feedback_pose_guard_max_loss" not in explicit_overrides:
        _setdefault(args, "lafgs_diff_pnp_feedback_pose_guard_max_loss", -1.0)
    else:
        _setdefault(args, "lafgs_diff_pnp_feedback_pose_guard_max_loss", -1.0)
    if "lafgs_diff_pnp_feedback_pose_guard_softness" not in explicit_overrides:
        _setdefault(args, "lafgs_diff_pnp_feedback_pose_guard_softness", 0.0)
    else:
        _setdefault(args, "lafgs_diff_pnp_feedback_pose_guard_softness", 0.0)
    if "lafgs_diff_pnp_feedback_pose_guard_min_scale" not in explicit_overrides:
        _setdefault(args, "lafgs_diff_pnp_feedback_pose_guard_min_scale", 0.0)
    else:
        _setdefault(args, "lafgs_diff_pnp_feedback_pose_guard_min_scale", 0.0)
    if "lafgs_diff_pnp_feedback_pose_guard_keep_gt_reprojection" not in explicit_overrides:
        _setdefault(args, "lafgs_diff_pnp_feedback_pose_guard_keep_gt_reprojection", False)
    else:
        _setdefault(args, "lafgs_diff_pnp_feedback_pose_guard_keep_gt_reprojection", False)
    if "lafgs_diff_pnp_detach_pnp_points" not in explicit_overrides:
        _setdefault(args, "lafgs_diff_pnp_detach_pnp_points", False)
    else:
        _setdefault(args, "lafgs_diff_pnp_detach_pnp_points", False)
    if "lafgs_diff_pnp_detach_gt_reprojection_points" not in explicit_overrides:
        _setdefault(args, "lafgs_diff_pnp_detach_gt_reprojection_points", False)
    else:
        _setdefault(args, "lafgs_diff_pnp_detach_gt_reprojection_points", False)
    _set_if_missing_or_legacy(args, "lafgs_geometry_residual_weight", 1.0, {0.0})
    _set_if_missing_or_legacy(args, "lafgs_geometry_residual_max_scale_ratio", 0.2, {0.0})
    _setdefault(args, "lafgs_geometry_residual", False)
    return args


def _build_locaware_parser():
    from argparse import ArgumentParser

    from arguments import ModelParams, OptimizationParams
    from train_locaware import add_locaware_training_args

    parser = ArgumentParser(description="LaFGS training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    add_locaware_training_args(parser)
    return parser, lp, op


def main(argv=None):
    from train_locaware import training
    from utils.general_utils import safe_state, seed_everything
    import torch

    parser, lp, op = _build_locaware_parser()
    argv = sys.argv[1:] if argv is None else list(argv)
    explicit_overrides = _explicit_lafgs_overrides(argv)
    args = parser.parse_args(argv)
    args = lafgs_defaults(args, explicit_overrides=explicit_overrides)
    args.loc_teacher = "direct"
    args.localization_enabled = True
    args.use_loc_opacity = True
    args.save_iterations.append(args.iterations)
    args.test_iterations.append(args.iterations)

    print("Optimizing LaFGS " + args.model_path)
    safe_state(args.quiet)
    seed_everything(args.train_seed)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), args)
    print("\nLaFGS localization-aware reconstruction training complete.")


if __name__ == "__main__":
    main()
