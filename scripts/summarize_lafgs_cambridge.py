#!/usr/bin/env python3
import argparse
import glob
import json
import os
from pathlib import Path


DEFAULT_SCENES = ["GreatCourt", "KingsCollege", "OldHospital", "ShopFacade", "StMarysChurch"]
METRICS = ("median_te", "median_ae", "recall_5cm_5d", "recall_2cm_2d", "avg_inliers")
TRAINING_KEYS = (
    "episodes",
    "direct_episodes",
    "diff_pnp_episodes",
    "diff_pnp_nonzero_loss_episodes",
    "diff_pnp_used_correspondences_total",
    "diff_pnp_loss_total",
    "diff_pnp_loss_max",
    "diff_pnp_pose_loss_total",
    "diff_pnp_pose_loss_max",
    "diff_pnp_detach_pnp_points_total",
    "diff_pnp_detach_pnp_points_max",
    "diff_pnp_geometry_pose_guard_max_loss_increase_total",
    "diff_pnp_geometry_pose_guard_max_loss_total",
    "diff_pnp_geometry_pose_guard_softness_total",
    "diff_pnp_geometry_pose_guard_min_scale_total",
    "diff_pnp_geometry_pose_guard_scale_total",
    "diff_pnp_geometry_pose_guard_scale_max",
    "diff_pnp_geometry_pose_guard_violation_total",
    "diff_pnp_feedback_pose_guard_passed_total",
    "diff_pnp_feedback_pose_guard_max_loss_increase_total",
    "diff_pnp_feedback_pose_guard_max_loss_total",
    "diff_pnp_feedback_pose_guard_softness_total",
    "diff_pnp_feedback_pose_guard_min_scale_total",
    "diff_pnp_feedback_pose_guard_scale_total",
    "diff_pnp_feedback_pose_guard_scale_max",
    "diff_pnp_feedback_pose_guard_violation_total",
    "diff_pnp_feedback_gt_reprojection_scale_total",
    "diff_pnp_geometry_correspondences_total",
    "diff_pnp_geometry_candidate_count_total",
    "diff_pnp_geometry_depth_anchor_loss_total",
    "diff_pnp_geometry_depth_anchor_weight_total",
    "diff_pnp_geometry_depth_anchor_correspondences_total",
    "diff_pnp_geometry_depth_anchor_candidate_count_total",
    "diff_pnp_geometry_depth_anchor_weight_sum_total",
    "diff_pnp_geometry_depth_anchor_filter_keep_ratio_total",
    "diff_pnp_geometry_depth_anchor_filter_keep_ratio_max",
    "diff_pnp_geometry_match_reprojection_weight_total",
    "diff_pnp_geometry_match_confidence_threshold_total",
    "diff_pnp_geometry_match_margin_threshold_total",
    "diff_pnp_geometry_match_peak_probability_threshold_total",
    "diff_pnp_geometry_match_max_entropy_total",
    "diff_pnp_geometry_match_max_reprojection_error_total",
    "diff_pnp_geometry_match_correspondences_total",
    "diff_pnp_geometry_match_candidate_count_total",
    "diff_pnp_geometry_match_weight_sum_total",
    "diff_pnp_geometry_match_filter_keep_ratio_total",
    "diff_pnp_geometry_match_filter_keep_ratio_max",
    "diff_pnp_geometry_valid_candidate_count_total",
    "diff_pnp_geometry_filter_keep_ratio_total",
    "diff_pnp_geometry_filter_keep_ratio_max",
    "diff_pnp_geometry_candidate_confidence_mean_total",
    "diff_pnp_geometry_candidate_confidence_max_max",
    "diff_pnp_geometry_kept_confidence_mean_total",
    "diff_pnp_geometry_kept_confidence_max_max",
    "diff_pnp_geometry_candidate_margin_mean_total",
    "diff_pnp_geometry_candidate_margin_max_max",
    "diff_pnp_geometry_kept_margin_mean_total",
    "diff_pnp_geometry_kept_margin_max_max",
    "diff_pnp_geometry_candidate_peak_probability_mean_total",
    "diff_pnp_geometry_candidate_peak_probability_min_total",
    "diff_pnp_geometry_candidate_peak_probability_min_max",
    "diff_pnp_geometry_candidate_peak_probability_max_max",
    "diff_pnp_geometry_kept_peak_probability_mean_total",
    "diff_pnp_geometry_kept_peak_probability_min_total",
    "diff_pnp_geometry_kept_peak_probability_min_max",
    "diff_pnp_geometry_kept_peak_probability_max_max",
    "diff_pnp_geometry_candidate_entropy_mean_total",
    "diff_pnp_geometry_candidate_entropy_max_max",
    "diff_pnp_geometry_kept_entropy_mean_total",
    "diff_pnp_geometry_kept_entropy_max_max",
    "diff_pnp_geometry_candidate_reprojection_error_mean_total",
    "diff_pnp_geometry_candidate_reprojection_error_max_max",
    "diff_pnp_geometry_kept_reprojection_error_mean_total",
    "diff_pnp_geometry_kept_reprojection_error_max_max",
    "diff_pnp_geometry_confidence_threshold_total",
    "diff_pnp_geometry_margin_threshold_total",
    "diff_pnp_geometry_peak_probability_threshold_total",
    "diff_pnp_geometry_max_entropy_total",
    "diff_pnp_geometry_max_reprojection_error_total",
    "diff_pnp_geometry_use_all_correspondences_total",
    "diff_pnp_geometry_local_window_radius_total",
    "diff_pnp_condition_guard_max_condition_number_total",
    "diff_pnp_condition_guard_scale_total",
    "diff_pnp_condition_guard_passed_total",
    "geometry_xyz_lr_total",
    "geometry_xyz_lr_max",
    "geometry_xyz_lr_nonzero_episodes",
    "geometry_xyz_grad_abs_max",
    "geometry_xyz_grad_nonzero_episodes",
    "geometry_xyz_full_grad_abs_max",
    "geometry_xyz_isolated_grad_abs_max",
    "geometry_xyz_isolated_grad_episodes",
    "geometry_xyz_step_delta_max",
    "geometry_xyz_step_nonzero_episodes",
    "mvinit_used_views",
    "mvinit_feature_scale",
    "mvinit_mean_observations",
)


def _read_json(path):
    with open(path) as f:
        return json.load(f)


def _real(path):
    return os.path.realpath(str(path))


def latest_result_for_model(results_root, model_path, prefix=""):
    results_root = Path(results_root)
    model_path = _real(model_path)
    best = None
    for summary_path in glob.glob(str(results_root / "**" / "summary.json"), recursive=True):
        parent = Path(summary_path).parent.name
        if prefix and not parent.startswith(prefix):
            continue
        try:
            summary = _read_json(summary_path)
        except Exception:
            continue
        if _real(summary.get("model_path", "")) != model_path:
            continue
        mtime = os.path.getmtime(summary_path)
        if best is None or mtime > best[0]:
            best = (mtime, summary_path, summary)
    if best is None:
        return None, None
    return best[1], best[2]


def _metric(summary, key):
    if not summary:
        return None
    sparse = summary.get("sparse") or {}
    value = sparse.get(key)
    return float(value) if value is not None else None


def _delta(lafgs_value, baseline_value):
    if lafgs_value is None or baseline_value is None:
        return None
    return float(lafgs_value - baseline_value)


def _relative_delta(delta, baseline_value):
    if delta is None or baseline_value in (None, 0):
        return None
    return float(delta / baseline_value)


def _training_summary(model_path):
    path = Path(model_path) / "loc_training_summary.json"
    if not path.exists():
        return {}, None
    try:
        return _read_json(path), str(path)
    except Exception:
        return {}, str(path)


def summarize_scene(
    scene,
    *,
    baseline_root,
    lafgs_root,
    results_root,
    baseline_prefix,
    lafgs_prefix,
    final_iteration,
):
    baseline_model = Path(baseline_root) / f"{scene}_baseline"
    lafgs_model = Path(lafgs_root) / scene
    final_ply = lafgs_model / "point_cloud" / f"iteration_{final_iteration}" / "point_cloud.ply"
    baseline_result_path, baseline_result = latest_result_for_model(results_root, baseline_model, baseline_prefix)
    lafgs_result_path, lafgs_result = latest_result_for_model(results_root, lafgs_model, lafgs_prefix)
    train_summary, train_summary_path = _training_summary(lafgs_model)

    row = {
        "scene": scene,
        "baseline_model": str(baseline_model),
        "lafgs_model": str(lafgs_model),
        "lafgs_final_iteration": int(final_iteration),
        "lafgs_final_ply_exists": final_ply.exists(),
        "baseline_summary_path": baseline_result_path,
        "lafgs_summary_path": lafgs_result_path,
        "training_summary_path": train_summary_path,
        "complete": baseline_result is not None and lafgs_result is not None,
    }
    for key in METRICS:
        baseline_value = _metric(baseline_result, key)
        lafgs_value = _metric(lafgs_result, key)
        row[f"baseline_{key}"] = baseline_value
        row[f"lafgs_{key}"] = lafgs_value
        delta = _delta(lafgs_value, baseline_value)
        row[f"delta_{key}"] = delta
        row[f"relative_delta_{key}"] = _relative_delta(delta, baseline_value)
        if key == "median_te":
            row["baseline_median_te_cm"] = baseline_value
            row["lafgs_median_te_cm"] = lafgs_value
            row["delta_median_te_cm"] = delta
        elif key == "median_ae":
            row["baseline_median_ae_deg"] = baseline_value
            row["lafgs_median_ae_deg"] = lafgs_value
            row["delta_median_ae_deg"] = delta

    for key in TRAINING_KEYS:
        row[key] = train_summary.get(key)

    return row


def summarize_cambridge(
    *,
    scenes,
    baseline_root,
    lafgs_root,
    results_root,
    baseline_prefix,
    lafgs_prefix,
    final_iteration,
):
    rows = [
        summarize_scene(
            scene,
            baseline_root=baseline_root,
            lafgs_root=lafgs_root,
            results_root=results_root,
            baseline_prefix=baseline_prefix,
            lafgs_prefix=lafgs_prefix,
            final_iteration=final_iteration,
        )
        for scene in scenes
    ]
    complete = [row for row in rows if row["complete"]]
    te_deltas = [row["delta_median_te"] for row in complete if row["delta_median_te"] is not None]
    ae_deltas = [row["delta_median_ae"] for row in complete if row["delta_median_ae"] is not None]
    recall_deltas = [row["delta_recall_5cm_5d"] for row in complete if row["delta_recall_5cm_5d"] is not None]
    missing = [row["scene"] for row in rows if not row["complete"]]
    aggregate = {
        "scene_count": len(rows),
        "completed_scene_count": len(complete),
        "missing_result_scenes": missing,
        "improved_te_scene_count": sum(1 for value in te_deltas if value < 0),
        "worsened_te_scene_count": sum(1 for value in te_deltas if value > 0),
        "mean_delta_median_te_cm": sum(te_deltas) / len(te_deltas) if te_deltas else None,
        "mean_delta_median_ae_deg": sum(ae_deltas) / len(ae_deltas) if ae_deltas else None,
        "mean_delta_recall_5cm_5d": sum(recall_deltas) / len(recall_deltas) if recall_deltas else None,
    }
    return {"aggregate": aggregate, "scenes": rows}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Summarize Cambridge baseline vs guarded-PnP LaFGS results.")
    parser.add_argument("--scenes", nargs="+", default=DEFAULT_SCENES)
    parser.add_argument("--baseline_root", default="/mnt/pool/sqy/stdloc_la_full_runs")
    parser.add_argument("--lafgs_root", required=True)
    parser.add_argument("--results_root", default="/root/STDLoc/results")
    parser.add_argument("--baseline_prefix", default="baseline-30000")
    parser.add_argument("--lafgs_prefix", default="lafgs-guarded-pnp")
    parser.add_argument("--final_iteration", type=int, default=30500)
    parser.add_argument("--output", default="")
    parser.add_argument("--require_complete", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    summary = summarize_cambridge(
        scenes=args.scenes,
        baseline_root=args.baseline_root,
        lafgs_root=args.lafgs_root,
        results_root=args.results_root,
        baseline_prefix=args.baseline_prefix,
        lafgs_prefix=args.lafgs_prefix,
        final_iteration=args.final_iteration,
    )
    text = json.dumps(summary, indent=2, sort_keys=True)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n")
    print(text)
    if args.require_complete and summary["aggregate"]["missing_result_scenes"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
