#!/usr/bin/env python3
"""Evaluate a compact LaFGS map with one sparse PoseLib solve per query."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from common.hashing import sha256_file
from common.config import (
    load_scene_calibration,
    load_mainline_config,
    resolve_keypoint_count,
    resolve_reprojection_error_px,
)
from data.datasets import ColmapDataset
from evaluation.bootstrap import materialize_a0
from evaluation.evaluator import evaluate_dataset
from localization.localizer import SparseLocalizer


def _input_artifact_contract(
    map_path: Path, descriptor_state_path: Path, *, descriptor_role: str
) -> dict:
    if descriptor_role not in {"metric_state", "context_state"}:
        raise ValueError("unknown descriptor state role")
    return {
        "map": {
            "path": str(map_path.resolve()),
            "sha256": sha256_file(map_path),
        },
        "descriptor_state": {
            "role": descriptor_role,
            "path": str(descriptor_state_path.resolve()),
            "sha256": sha256_file(descriptor_state_path),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--images", default="processed")
    parser.add_argument("--map", type=Path)
    parser.add_argument("--metric-state", type=Path)
    parser.add_argument(
        "--context-state",
        type=Path,
        help="Mapping-only MCCD artifact; mutually exclusive with --metric-state.",
    )
    parser.add_argument(
        "--scene-calibration",
        type=Path,
        help=(
            "Mapping-only calibration JSON. By default the evaluator uses "
            "scene_calibration.json beside the trained map when present."
        ),
    )
    parser.add_argument(
        "--guided-sampling",
        action="store_true",
        help=(
            "Sort unchanged top-1 correspondences by descriptor margin, "
            "mapping matchability, and map uncertainty for one PoseLib PROSAC solve."
        ),
    )
    parser.add_argument(
        "--group-aware-pose",
        action="store_true",
        help=(
            "Add a bounded distinct-parent AP3P hypothesis supplement to the "
            "single robust-pose wrapper; mapping evidence only until validated."
        ),
    )
    parser.add_argument("--group-field", default="parent_source_track_ids")
    parser.add_argument("--group-hypothesis-samples", type=int, default=32)
    parser.add_argument(
        "--stage-state",
        type=Path,
        help="Evaluate A0 by materializing a Stage-A state with an identity metric.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default="configs/paper_mainline.yaml")
    parser.add_argument("--split", choices=("mapping", "test"), default="test")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--suppress-duplicate-anchors",
        action="store_true",
        help=(
            "Keep only the highest-score query match per landmark before the "
            "single PoseLib solve."
        ),
    )
    parser.add_argument(
        "--assignment-topk",
        type=int,
        default=0,
        help=(
            "Enable sparse maximum-weight query-to-Anchor assignment over exact "
            "global top-K candidates. Zero preserves the frozen independent top-1."
        ),
    )
    parser.add_argument(
        "--assignment-dustbin-score",
        type=float,
        default=-1.0,
        help="Strict minimum cosine score for an assigned real Anchor edge.",
    )
    parser.add_argument(
        "--topk-geometric-feedback",
        action="store_true",
        help=(
            "Enable the V21 query-specific exact Top-64 geometric bundle and "
            "at most one additional PoseLib solve, without sparse LGCV."
        ),
    )
    parser.add_argument(
        "--sparse-lgcv-topk-feedback",
        action="store_true",
        help=(
            "Enable the V22 query-specific exact Top-64 plus sparse LGCV "
            "bundle gate and at most one additional PoseLib solve."
        ),
    )
    parser.add_argument(
        "--pose-conditioned-sparse-refinement",
        action="store_true",
        help=(
            "Enable the V24 GPU pose-conditioned Top-64 joint assignment, "
            "protected first-pass inliers, and at most one additional PoseLib solve."
        ),
    )
    parser.add_argument(
        "--refinement-pose-backend",
        choices=("local", "robust"),
        default="local",
        help=(
            "Pose-conditioned second-stage solver: local keeps the first pose "
            "basin; robust runs one bounded PoseLib re-estimate."
        ),
    )
    parser.add_argument(
        "--feedback-minimum-baseline-inliers",
        type=int,
        default=128,
        help="Inclusive first-pass inlier trigger for either online feedback arm.",
    )
    parser.add_argument(
        "--feedback-maximum-baseline-inliers",
        type=int,
        default=256,
        help=(
            "Exclusive first-pass inlier trigger for either feedback arm; zero "
            "removes the upper bound."
        ),
    )
    parser.add_argument(
        "--feedback-minimum-candidate-inlier-gain",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--feedback-minimum-candidate-relative-inlier-gain",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--feedback-maximum-candidate-ransac-iterations",
        type=int,
        default=0,
        help="Zero disables the candidate PoseLib iteration upper bound.",
    )
    parser.add_argument(
        "--feedback-minimum-baseline-inlier-retention",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--feedback-maximum-protected-median-residual-increase-px",
        type=float,
        default=-1.0,
    )
    parser.add_argument(
        "--feedback-maximum-protected-p90-residual-increase-px",
        type=float,
        default=-1.0,
    )
    parser.add_argument(
        "--feedback-maximum-pose-update-translation-cm",
        type=float,
        default=-1.0,
    )
    parser.add_argument(
        "--feedback-maximum-pose-update-rotation-deg",
        type=float,
        default=-1.0,
    )
    parser.add_argument(
        "--refinement-minimum-changed-inliers", type=int, default=8
    )
    parser.add_argument(
        "--refinement-maximum-score-drop-from-top1", type=float, default=0.03
    )
    parser.add_argument(
        "--refinement-view-direction-slack-deg", type=float, default=15.0
    )
    parser.add_argument(
        "--refinement-maximum-changed-rows", type=int, default=128
    )
    parser.add_argument(
        "--refinement-maximum-changed-to-baseline-inlier-ratio",
        type=float,
        default=0.50,
    )
    parser.add_argument(
        "--refinement-minimum-proposal-count", type=int, default=60
    )
    parser.add_argument(
        "--refinement-minimum-proposal-relative-gain", type=float, default=0.075
    )
    parser.add_argument(
        "--refinement-minimum-changed-inlier-fraction", type=float, default=0.10
    )
    parser.add_argument(
        "--refinement-minimum-changed-inlier-spatial-cells", type=int, default=3
    )
    parser.add_argument(
        "--refinement-maximum-changed-inlier-median-residual-px",
        type=float,
        default=6.0,
    )
    parser.add_argument(
        "--deployment-mode",
        action="store_true",
        help=(
            "Avoid per-stage CUDA synchronization. Total latency remains strict; "
            "frontend/matching use CUDA-event timings."
        ),
    )
    args = parser.parse_args()
    if args.assignment_topk < 0:
        parser.error("--assignment-topk must be zero or positive")
    if args.assignment_topk and (
        args.suppress_duplicate_anchors
        or args.guided_sampling
        or args.group_aware_pose
    ):
        parser.error(
            "--assignment-topk cannot be combined with duplicate suppression "
            "guided sampling, or group-aware pose"
        )
    if sum(
        (
            args.topk_geometric_feedback,
            args.sparse_lgcv_topk_feedback,
            args.pose_conditioned_sparse_refinement,
        )
    ) > 1:
        parser.error(
            "online sparse-refinement modes are separate ablations"
        )
    if (
        args.topk_geometric_feedback
        or args.sparse_lgcv_topk_feedback
        or args.pose_conditioned_sparse_refinement
    ) and (
        args.assignment_topk
        or args.suppress_duplicate_anchors
        or args.guided_sampling
        or args.group_aware_pose
        or args.context_state is not None
    ):
        parser.error(
            "online sparse refinement is a separate shared-metric ablation"
        )
    if args.stage_state:
        if args.map or args.metric_state or args.context_state:
            parser.error("--stage-state cannot be combined with descriptor map options")
        map_path, metric_path = materialize_a0(
            args.stage_state, args.output / "materialized_a0", args.config
        )
    else:
        if args.map is None:
            parser.error("A1 evaluation requires --map")
        if (args.metric_state is None) == (args.context_state is None):
            parser.error(
                "select exactly one descriptor protocol: --metric-state or "
                "--context-state"
            )
        map_path, metric_path = args.map, args.metric_state
    deployment = load_mainline_config(args.config).values["deployment"]
    dataset = ColmapDataset(args.dataset, images=args.images)
    cameras = dataset.split(args.split)
    calibration_cameras = dataset.split("mapping")
    calibration_path = args.scene_calibration
    inferred_calibrations = [map_path.parent / "scene_calibration.json"]
    if args.stage_state is not None:
        inferred_calibrations.insert(
            0, args.stage_state.parent.parent / "scene_calibration.json"
        )
    if calibration_path is None:
        calibration_path = next(
            (path for path in inferred_calibrations if path.is_file()), None
        )
    scene_calibration = (
        load_scene_calibration(calibration_path)
        if calibration_path is not None
        else None
    )
    keypoint_count = resolve_keypoint_count(deployment, calibration_cameras)
    reprojection_error_px = resolve_reprojection_error_px(
        deployment, calibration_cameras, scene_calibration
    )
    descriptor_state_path = (
        args.context_state.resolve()
        if args.context_state is not None
        else metric_path.resolve()
    )
    artifact_contract = _input_artifact_contract(
        map_path,
        descriptor_state_path,
        descriptor_role=(
            "context_state" if args.context_state is not None else "metric_state"
        ),
    )
    localizer = SparseLocalizer(
        map_path,
        metric_path,
        context_state_path=args.context_state,
        device=args.device,
        keypoint_count=keypoint_count,
        nms_radius=int(deployment["nms"]),
        reprojection_error_px=reprojection_error_px,
        confidence=deployment["confidence"],
        max_iterations=deployment["maximum_iterations"],
        min_iterations=deployment["minimum_iterations"],
        seed=args.seed,
        suppress_duplicate_anchors=args.suppress_duplicate_anchors,
        guided_sampling=args.guided_sampling,
        group_aware_pose=args.group_aware_pose,
        group_field=args.group_field,
        group_hypothesis_samples=args.group_hypothesis_samples,
        assignment_topk=args.assignment_topk,
        assignment_dustbin_score=args.assignment_dustbin_score,
        topk_geometric_feedback=args.topk_geometric_feedback,
        sparse_lgcv_topk_feedback=args.sparse_lgcv_topk_feedback,
        pose_conditioned_sparse_refinement=(
            args.pose_conditioned_sparse_refinement
        ),
        refinement_pose_backend=args.refinement_pose_backend,
        feedback_minimum_baseline_inliers=(
            args.feedback_minimum_baseline_inliers
        ),
        feedback_maximum_baseline_inliers=(
            args.feedback_maximum_baseline_inliers
        ),
        feedback_minimum_candidate_inlier_gain=(
            args.feedback_minimum_candidate_inlier_gain
        ),
        feedback_minimum_candidate_relative_inlier_gain=(
            args.feedback_minimum_candidate_relative_inlier_gain
        ),
        feedback_maximum_candidate_ransac_iterations=(
            args.feedback_maximum_candidate_ransac_iterations
        ),
        feedback_minimum_baseline_inlier_retention=(
            args.feedback_minimum_baseline_inlier_retention
        ),
        feedback_maximum_protected_median_residual_increase_px=(
            args.feedback_maximum_protected_median_residual_increase_px
        ),
        feedback_maximum_protected_p90_residual_increase_px=(
            args.feedback_maximum_protected_p90_residual_increase_px
        ),
        feedback_maximum_pose_update_translation_cm=(
            args.feedback_maximum_pose_update_translation_cm
        ),
        feedback_maximum_pose_update_rotation_deg=(
            args.feedback_maximum_pose_update_rotation_deg
        ),
        refinement_maximum_score_drop_from_top1=(
            args.refinement_maximum_score_drop_from_top1
        ),
        refinement_view_direction_slack_deg=(
            args.refinement_view_direction_slack_deg
        ),
        refinement_maximum_changed_rows=(
            args.refinement_maximum_changed_rows
        ),
        refinement_maximum_changed_to_baseline_inlier_ratio=(
            args.refinement_maximum_changed_to_baseline_inlier_ratio
        ),
        refinement_minimum_proposal_count=(
            args.refinement_minimum_proposal_count
        ),
        refinement_minimum_proposal_relative_gain=(
            args.refinement_minimum_proposal_relative_gain
        ),
        refinement_minimum_changed_inliers=(
            args.refinement_minimum_changed_inliers
        ),
        refinement_minimum_changed_inlier_fraction=(
            args.refinement_minimum_changed_inlier_fraction
        ),
        refinement_minimum_changed_inlier_spatial_cells=(
            args.refinement_minimum_changed_inlier_spatial_cells
        ),
        refinement_maximum_changed_inlier_median_residual_px=(
            args.refinement_maximum_changed_inlier_median_residual_px
        ),
        profile_mode=not args.deployment_mode,
    )
    result = evaluate_dataset(
        dataset=dataset,
        localizer=localizer,
        cameras=cameras,
        output=args.output,
    )
    if (
        sha256_file(map_path) != artifact_contract["map"]["sha256"]
        or sha256_file(descriptor_state_path)
        != artifact_contract["descriptor_state"]["sha256"]
    ):
        raise RuntimeError("evaluation input artifact changed while localization ran")
    result["summary"].update(
        {
            "evaluated_split": args.split,
            "random_seed": int(args.seed),
            "input_map_path": artifact_contract["map"]["path"],
            "input_map_sha256": artifact_contract["map"]["sha256"],
            "input_descriptor_state_role": artifact_contract["descriptor_state"][
                "role"
            ],
            "input_descriptor_state_path": artifact_contract["descriptor_state"][
                "path"
            ],
            "input_descriptor_state_sha256": artifact_contract[
                "descriptor_state"
            ]["sha256"],
        }
    )
    # ``evaluate_dataset`` writes its generic summary before the CLI-specific
    # artifact contract is known.  Persist the enriched summary so every real
    # localization result is unambiguously bound to the evaluated map.
    (args.output / "summary.json").write_text(
        json.dumps(result["summary"], indent=2, sort_keys=True) + "\n"
    )
    (args.output / "deployment_contract.json").write_text(
        json.dumps(
            {
                "schema": "lafgs_sparse_deployment_contract",
                "version": 2,
                "keypoint_count": int(keypoint_count),
                "nms_radius": int(deployment["nms"]),
                "ransac_reprojection_px": float(reprojection_error_px),
                "scene_calibration": (
                    str(calibration_path.resolve())
                    if calibration_path is not None
                    else None
                ),
                "calibration_split": "mapping",
                "evaluated_split": args.split,
                "pose_solves": (
                    "one PoseLib RANSAC plus one local nonlinear refinement"
                    if args.pose_conditioned_sparse_refinement
                    and args.refinement_pose_backend == "local"
                    else "one plus at most one bounded PoseLib RANSAC"
                    if args.pose_conditioned_sparse_refinement
                    else
                    "one plus at most one feedback solve"
                    if args.topk_geometric_feedback
                    or args.sparse_lgcv_topk_feedback
                    or args.pose_conditioned_sparse_refinement
                    else 1
                ),
                "duplicate_anchor_suppression": bool(args.suppress_duplicate_anchors),
                "guided_sampling": bool(args.guided_sampling),
                "group_aware_pose": bool(args.group_aware_pose),
                "group_field": args.group_field if args.group_aware_pose else None,
                "group_hypothesis_samples": (
                    int(args.group_hypothesis_samples)
                    if args.group_aware_pose
                    else 0
                ),
                "capacity_assignment": bool(args.assignment_topk > 0),
                "topk_geometric_feedback": bool(args.topk_geometric_feedback),
                "sparse_lgcv_topk_feedback": bool(
                    args.sparse_lgcv_topk_feedback
                ),
                "pose_conditioned_sparse_refinement": bool(
                    args.pose_conditioned_sparse_refinement
                ),
                "refinement_pose_backend": args.refinement_pose_backend,
                "refinement_candidate_pool": (
                    "first_pose_point_projection_frustum"
                    if args.pose_conditioned_sparse_refinement
                    and args.refinement_pose_backend == "robust"
                    else "global_top64"
                    if args.pose_conditioned_sparse_refinement
                    else "disabled"
                ),
                "feedback_minimum_baseline_inliers": int(
                    args.feedback_minimum_baseline_inliers
                ),
                "feedback_maximum_baseline_inliers": int(
                    args.feedback_maximum_baseline_inliers
                ),
                "feedback_minimum_candidate_inlier_gain": int(
                    args.feedback_minimum_candidate_inlier_gain
                ),
                "feedback_minimum_candidate_relative_inlier_gain": float(
                    args.feedback_minimum_candidate_relative_inlier_gain
                ),
                "feedback_maximum_candidate_ransac_iterations": int(
                    args.feedback_maximum_candidate_ransac_iterations
                ),
                "feedback_minimum_baseline_inlier_retention": float(
                    args.feedback_minimum_baseline_inlier_retention
                ),
                "feedback_maximum_protected_median_residual_increase_px": float(
                    args.feedback_maximum_protected_median_residual_increase_px
                ),
                "feedback_maximum_protected_p90_residual_increase_px": float(
                    args.feedback_maximum_protected_p90_residual_increase_px
                ),
                "feedback_maximum_pose_update_translation_cm": float(
                    args.feedback_maximum_pose_update_translation_cm
                ),
                "feedback_maximum_pose_update_rotation_deg": float(
                    args.feedback_maximum_pose_update_rotation_deg
                ),
                "refinement_minimum_changed_inliers": int(
                    args.refinement_minimum_changed_inliers
                ),
                "refinement_maximum_score_drop_from_top1": float(
                    args.refinement_maximum_score_drop_from_top1
                ),
                "refinement_view_direction_slack_deg": float(
                    args.refinement_view_direction_slack_deg
                ),
                "refinement_maximum_changed_rows": int(
                    args.refinement_maximum_changed_rows
                ),
                "refinement_maximum_changed_to_baseline_inlier_ratio": float(
                    args.refinement_maximum_changed_to_baseline_inlier_ratio
                ),
                "refinement_minimum_proposal_count": int(
                    args.refinement_minimum_proposal_count
                ),
                "refinement_minimum_proposal_relative_gain": float(
                    args.refinement_minimum_proposal_relative_gain
                ),
                "refinement_minimum_changed_inlier_fraction": float(
                    args.refinement_minimum_changed_inlier_fraction
                ),
                "refinement_minimum_changed_inlier_spatial_cells": int(
                    args.refinement_minimum_changed_inlier_spatial_cells
                ),
                "refinement_maximum_changed_inlier_median_residual_px": float(
                    args.refinement_maximum_changed_inlier_median_residual_px
                ),
                "assignment_topk": int(args.assignment_topk),
                "assignment_dustbin_score": float(args.assignment_dustbin_score),
                "timing_mode": "deployment" if args.deployment_mode else "profile",
                "descriptor_protocol": (
                    "mccd" if args.context_state is not None else "shared_metric"
                ),
                "input_artifacts": artifact_contract,
                "photometric_canonicalization_contract": (
                    localizer.photometric_canonicalization_contract
                ),
                "context_state": (
                    str(args.context_state.resolve())
                    if args.context_state is not None
                    else None
                ),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
