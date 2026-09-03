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
        "--view-conditioned-anchor-state",
        type=Path,
        help=(
            "Mapping-only V27 descriptor modes, selected by the first pose "
            "inside pose-conditioned sparse refinement."
        ),
    )
    parser.add_argument(
        "--view-conditioned-minimum-concentration", type=float, default=0.0
    )
    parser.add_argument("--view-conditioned-residual-scale", type=float, default=1.0)
    parser.add_argument(
        "--view-conditioned-require-two-valid-modes", action="store_true"
    )
    parser.add_argument(
        "--view-conditioned-score-fusion",
        choices=("replace", "max_with_base"),
        default="replace",
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
        "--confidence-core-progressive-sampling",
        action="store_true",
        help=(
            "Keep the complete retained confidence Core but order it by "
            "cosine score for PoseLib progressive sampling."
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
    parser.add_argument(
        "--keypoint-count-override",
        type=int,
        help="Engineering upper-bound override; mapping-derived default is unchanged.",
    )
    parser.add_argument(
        "--superpoint-subpixel-refinement",
        action="store_true",
        help=(
            "Apply a bounded 3x3 quadratic detector-peak fit before sparse "
            "descriptor sampling and PnP."
        ),
    )
    parser.add_argument(
        "--superpoint-subpixel-geometry-only",
        action="store_true",
        help=(
            "Keep native sparse descriptors bit-exact and refine only the 2D "
            "coordinates passed to PnP."
        ),
    )
    parser.add_argument(
        "--superpoint-subpixel-maximum-offset", type=float, default=0.5
    )
    parser.add_argument("--ransac-confidence-override", type=float)
    parser.add_argument("--ransac-maximum-iterations-override", type=int)
    parser.add_argument("--ransac-minimum-iterations-override", type=int)
    parser.add_argument(
        "--ransac-hypothesis-core-size",
        type=int,
        default=0,
        help=(
            "Experimental first-pose solver: generate robust hypotheses on "
            "this many strongest matches, then rescore and refine on all matches."
        ),
    )
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
    parser.add_argument("--refinement-minimum-changed-inliers", type=int, default=8)
    parser.add_argument("--refinement-projection-gate-px", type=float, default=8.0)
    parser.add_argument(
        "--refinement-uncertainty-projection-gate-px",
        type=float,
        default=0.0,
        help="Zero disables the wider low-inlier projection gate.",
    )
    parser.add_argument(
        "--refinement-uncertainty-maximum-baseline-inliers",
        type=int,
        default=0,
        help="Use the wider projection gate at or below this first-pass count.",
    )
    parser.add_argument(
        "--refinement-maximum-score-drop-from-top1", type=float, default=0.03
    )
    parser.add_argument(
        "--refinement-reliability-adaptive-score-drop",
        action="store_true",
        help=(
            "Allow a wider descriptor score drop only for mapping-reliable "
            "Anchors with strong first-pose reprojection evidence."
        ),
    )
    parser.add_argument(
        "--refinement-reliability-expanded-score-drop", type=float, default=0.10
    )
    parser.add_argument(
        "--refinement-reliability-minimum-matchability-quantile",
        type=float,
        default=0.50,
    )
    parser.add_argument(
        "--refinement-reliability-maximum-uncertainty-quantile",
        type=float,
        default=0.50,
    )
    parser.add_argument(
        "--refinement-reliability-maximum-geometry-cost",
        type=float,
        default=0.50,
    )
    parser.add_argument(
        "--refinement-reliability-minimum-improvement-px",
        type=float,
        default=4.0,
    )
    parser.add_argument(
        "--refinement-view-direction-slack-deg", type=float, default=15.0
    )
    parser.add_argument("--refinement-maximum-changed-rows", type=int, default=128)
    parser.add_argument(
        "--refinement-maximum-changed-to-baseline-inlier-ratio",
        type=float,
        default=0.50,
    )
    parser.add_argument("--refinement-minimum-proposal-count", type=int, default=60)
    parser.add_argument(
        "--refinement-minimum-proposal-relative-gain", type=float, default=0.075
    )
    parser.add_argument(
        "--refinement-active-row-retrieval",
        action="store_true",
        help="Run second-stage descriptor retrieval only for first-pass outliers.",
    )
    parser.add_argument(
        "--refinement-pre-topk-view-filter",
        action="store_true",
        help=(
            "Apply mapping-only viewing-direction and distance support before "
            "the exact second-stage Top-K retrieval."
        ),
    )
    parser.add_argument(
        "--refinement-projection-first-local-candidates",
        action="store_true",
        help=(
            "Rank descriptors only among Anchors projected near each sparse "
            "query keypoint under T0, instead of descriptor Top-K first."
        ),
    )
    parser.add_argument(
        "--refinement-projection-first-radius-px", type=float, default=12.0
    )
    parser.add_argument(
        "--refinement-common-candidate-grid-gate",
        action="store_true",
        help=(
            "Compare the first and second poses on the same sparse Top-K "
            "candidate grid before accepting the second pose."
        ),
    )
    parser.add_argument(
        "--refinement-minimum-common-grid-relative-energy-gain",
        type=float,
        default=0.0,
        help=(
            "Minimum relative robust common-grid energy reduction; -1 records "
            "diagnostics without rejecting a candidate."
        ),
    )
    parser.add_argument(
        "--refinement-progressive-sampling",
        action="store_true",
        help=(
            "Order hard-core and pose-supported correspondences by online "
            "quality and enable PoseLib progressive sampling for the second solve."
        ),
    )
    parser.add_argument(
        "--refinement-allow-soft-inliers",
        action="store_true",
        help=(
            "Allow a bounded subset of high-residual first-pass inliers to "
            "compete for a supported alternative Anchor."
        ),
    )
    parser.add_argument(
        "--refinement-soft-inlier-minimum-residual-px", type=float, default=6.0
    )
    parser.add_argument(
        "--refinement-soft-inlier-maximum-score-drop", type=float, default=0.02
    )
    parser.add_argument(
        "--refinement-soft-inlier-minimum-improvement-px", type=float, default=2.0
    )
    parser.add_argument(
        "--refinement-maximum-soft-inlier-changes", type=int, default=16
    )
    parser.add_argument(
        "--refinement-pose-conditioned-mutual-matching",
        action="store_true",
        help=(
            "Require each geometrically feasible candidate Anchor to select "
            "its highest-scoring query row in the sparse pose-conditioned graph."
        ),
    )
    parser.add_argument(
        "--refinement-set-level-reserve-selection",
        action="store_true",
        help=(
            "Choose a spatially and depth-diverse Reserve bundle at the "
            "existing proposal cap instead of independent lowest cost."
        ),
    )
    parser.add_argument(
        "--refinement-heldout-candidate-validation",
        action="store_true",
        help=(
            "Reserve a deterministic spatial subset of first-pass outlier rows "
            "from the second solve and compare T0/T1 on their strict sparse graph."
        ),
    )
    parser.add_argument(
        "--refinement-spatial-jackknife-diagnostic",
        action="store_true",
        help=(
            "Record leave-one-image-cell local pose stability for T0/T1; "
            "diagnostic only and never authorizes the candidate."
        ),
    )
    parser.add_argument(
        "--refinement-minimum-heldout-relative-energy-gain",
        type=float,
        default=0.0,
        help="Minimum T0-to-T1 held-out strict-assignment energy reduction.",
    )
    parser.add_argument(
        "--refinement-uncertainty-aware-projection",
        action="store_true",
        help=(
            "Project first-pose and full 3D Anchor covariance into pixels and "
            "use a bounded per-edge projection gate."
        ),
    )
    parser.add_argument(
        "--refinement-maximum-uncertainty-projection-gate-px",
        type=float,
        default=12.0,
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
    parser.add_argument(
        "--match-retention-fraction",
        type=float,
        default=1.0,
        help=(
            "Retain this query-local fraction of the highest absolute-cosine "
            "Top-1 correspondences as the first-pass PnP confidence core."
        ),
    )
    parser.add_argument("--minimum-retained-match-count", type=int, default=256)
    parser.add_argument(
        "--minimum-sufficient-confidence-core",
        action="store_true",
        help=(
            "Build Core v2 from score, Top1/Top2 margin, mapping reliability, "
            "uncertainty, and 2D/3D diversity at the requested budget."
        ),
    )
    parser.add_argument(
        "--first-pass-query-cap",
        type=int,
        default=0,
        help=(
            "Restrict T0 to the strongest detector rows while leaving all "
            "extracted rows available to the sparse Reserve; zero disables."
        ),
    )
    parser.add_argument(
        "--refinement-expanded-reserve-maximum-inlier-fraction",
        type=float,
        default=0.0,
        help=(
            "Use detector rows beyond the exact T0 cap only when T0 inliers "
            "divided by retained Core matches do not exceed this query-level "
            "threshold; zero disables adaptive reserve expansion."
        ),
    )
    parser.add_argument(
        "--core-reserve-refinement",
        action="store_true",
        help=(
            "Use high-cosine matches for robust PnP, then admit only "
            "pose-consistent reserve Top-1 rows to one local sparse refinement."
        ),
    )
    parser.add_argument("--core-reserve-reprojection-gate-px", type=float, default=4.0)
    parser.add_argument("--core-reserve-minimum-supported-rows", type=int, default=16)
    parser.add_argument(
        "--final-pose-polish-reprojection-px",
        type=float,
        default=0.0,
        help=(
            "Locally polish the final pose using only current inliers within "
            "this strict pixel radius; zero disables."
        ),
    )
    parser.add_argument(
        "--final-pose-polish-minimum-inliers", type=int, default=64
    )
    args = parser.parse_args()
    if args.assignment_topk < 0:
        parser.error("--assignment-topk must be zero or positive")
    if args.assignment_topk and (
        args.suppress_duplicate_anchors or args.guided_sampling or args.group_aware_pose
    ):
        parser.error(
            "--assignment-topk cannot be combined with duplicate suppression "
            "guided sampling, or group-aware pose"
        )
    if (
        sum(
            (
                args.topk_geometric_feedback,
                args.sparse_lgcv_topk_feedback,
                args.pose_conditioned_sparse_refinement,
            )
        )
        > 1
    ):
        parser.error("online sparse-refinement modes are separate ablations")
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
        parser.error("online sparse refinement is a separate shared-metric ablation")
    if args.view_conditioned_anchor_state is not None and not (
        args.pose_conditioned_sparse_refinement
        and args.metric_state is not None
        and args.context_state is None
    ):
        parser.error(
            "--view-conditioned-anchor-state requires the shared identity metric "
            "and --pose-conditioned-sparse-refinement"
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
    if args.keypoint_count_override is not None:
        if not 512 <= int(args.keypoint_count_override) <= 4096:
            parser.error("--keypoint-count-override must be in [512,4096]")
        keypoint_count = int(args.keypoint_count_override)
    reprojection_error_px = resolve_reprojection_error_px(
        deployment, calibration_cameras, scene_calibration
    )
    ransac_confidence = (
        float(deployment["confidence"])
        if args.ransac_confidence_override is None
        else float(args.ransac_confidence_override)
    )
    ransac_maximum_iterations = (
        int(deployment["maximum_iterations"])
        if args.ransac_maximum_iterations_override is None
        else int(args.ransac_maximum_iterations_override)
    )
    ransac_minimum_iterations = (
        int(deployment["minimum_iterations"])
        if args.ransac_minimum_iterations_override is None
        else int(args.ransac_minimum_iterations_override)
    )
    if not (
        0.5 < ransac_confidence < 1.0
        and 1 <= ransac_minimum_iterations <= ransac_maximum_iterations
        and ransac_maximum_iterations <= 100000
    ):
        parser.error("RANSAC override configuration is invalid")
    if args.ransac_hypothesis_core_size and not (
        64 <= int(args.ransac_hypothesis_core_size) <= keypoint_count
    ):
        parser.error("--ransac-hypothesis-core-size must be zero or in [64,K]")
    if (
        args.superpoint_subpixel_refinement
        and args.superpoint_subpixel_geometry_only
    ) or not 0.0 <= float(args.superpoint_subpixel_maximum_offset) <= 0.5:
        parser.error("SuperPoint subpixel configuration is invalid")
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
    view_conditioned_contract = (
        {
            "path": str(args.view_conditioned_anchor_state.resolve()),
            "sha256": sha256_file(args.view_conditioned_anchor_state),
            "minimum_concentration": float(args.view_conditioned_minimum_concentration),
            "residual_scale": float(args.view_conditioned_residual_scale),
            "require_two_valid_modes": bool(
                args.view_conditioned_require_two_valid_modes
            ),
            "score_fusion": args.view_conditioned_score_fusion,
        }
        if args.view_conditioned_anchor_state is not None
        else None
    )
    localizer = SparseLocalizer(
        map_path,
        metric_path,
        context_state_path=args.context_state,
        view_conditioned_anchor_state_path=args.view_conditioned_anchor_state,
        view_conditioned_minimum_concentration=(
            args.view_conditioned_minimum_concentration
        ),
        view_conditioned_residual_scale=args.view_conditioned_residual_scale,
        view_conditioned_require_two_valid_modes=(
            args.view_conditioned_require_two_valid_modes
        ),
        view_conditioned_score_fusion=args.view_conditioned_score_fusion,
        device=args.device,
        keypoint_count=keypoint_count,
        nms_radius=int(deployment["nms"]),
        subpixel_keypoints=args.superpoint_subpixel_refinement,
        subpixel_geometry_only=args.superpoint_subpixel_geometry_only,
        subpixel_maximum_offset=args.superpoint_subpixel_maximum_offset,
        reprojection_error_px=reprojection_error_px,
        confidence=ransac_confidence,
        max_iterations=ransac_maximum_iterations,
        min_iterations=ransac_minimum_iterations,
        seed=args.seed,
        ransac_hypothesis_core_size=args.ransac_hypothesis_core_size,
        suppress_duplicate_anchors=args.suppress_duplicate_anchors,
        guided_sampling=args.guided_sampling,
        confidence_core_progressive_sampling=(
            args.confidence_core_progressive_sampling
        ),
        group_aware_pose=args.group_aware_pose,
        group_field=args.group_field,
        group_hypothesis_samples=args.group_hypothesis_samples,
        assignment_topk=args.assignment_topk,
        assignment_dustbin_score=args.assignment_dustbin_score,
        topk_geometric_feedback=args.topk_geometric_feedback,
        sparse_lgcv_topk_feedback=args.sparse_lgcv_topk_feedback,
        pose_conditioned_sparse_refinement=(args.pose_conditioned_sparse_refinement),
        refinement_pose_backend=args.refinement_pose_backend,
        feedback_minimum_baseline_inliers=(args.feedback_minimum_baseline_inliers),
        feedback_maximum_baseline_inliers=(args.feedback_maximum_baseline_inliers),
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
        refinement_projection_gate_px=args.refinement_projection_gate_px,
        refinement_uncertainty_projection_gate_px=(
            args.refinement_uncertainty_projection_gate_px
        ),
        refinement_uncertainty_maximum_baseline_inliers=(
            args.refinement_uncertainty_maximum_baseline_inliers
        ),
        refinement_maximum_score_drop_from_top1=(
            args.refinement_maximum_score_drop_from_top1
        ),
        refinement_reliability_adaptive_score_drop=(
            args.refinement_reliability_adaptive_score_drop
        ),
        refinement_reliability_expanded_score_drop=(
            args.refinement_reliability_expanded_score_drop
        ),
        refinement_reliability_minimum_matchability_quantile=(
            args.refinement_reliability_minimum_matchability_quantile
        ),
        refinement_reliability_maximum_uncertainty_quantile=(
            args.refinement_reliability_maximum_uncertainty_quantile
        ),
        refinement_reliability_maximum_geometry_cost=(
            args.refinement_reliability_maximum_geometry_cost
        ),
        refinement_reliability_minimum_improvement_px=(
            args.refinement_reliability_minimum_improvement_px
        ),
        refinement_view_direction_slack_deg=(args.refinement_view_direction_slack_deg),
        refinement_maximum_changed_rows=(args.refinement_maximum_changed_rows),
        refinement_maximum_changed_to_baseline_inlier_ratio=(
            args.refinement_maximum_changed_to_baseline_inlier_ratio
        ),
        refinement_minimum_proposal_count=(args.refinement_minimum_proposal_count),
        refinement_minimum_proposal_relative_gain=(
            args.refinement_minimum_proposal_relative_gain
        ),
        refinement_active_row_retrieval=args.refinement_active_row_retrieval,
        refinement_pre_topk_view_filter=(args.refinement_pre_topk_view_filter),
        refinement_common_candidate_grid_gate=(
            args.refinement_common_candidate_grid_gate
        ),
        refinement_minimum_common_grid_relative_energy_gain=(
            args.refinement_minimum_common_grid_relative_energy_gain
        ),
        refinement_progressive_sampling=args.refinement_progressive_sampling,
        refinement_allow_soft_inliers=args.refinement_allow_soft_inliers,
        refinement_soft_inlier_minimum_residual_px=(
            args.refinement_soft_inlier_minimum_residual_px
        ),
        refinement_soft_inlier_maximum_score_drop=(
            args.refinement_soft_inlier_maximum_score_drop
        ),
        refinement_soft_inlier_minimum_improvement_px=(
            args.refinement_soft_inlier_minimum_improvement_px
        ),
        refinement_maximum_soft_inlier_changes=(
            args.refinement_maximum_soft_inlier_changes
        ),
        refinement_pose_conditioned_mutual_matching=(
            args.refinement_pose_conditioned_mutual_matching
        ),
        refinement_set_level_reserve_selection=(
            args.refinement_set_level_reserve_selection
        ),
        refinement_projection_first_local_candidates=(
            args.refinement_projection_first_local_candidates
        ),
        refinement_projection_first_radius_px=(
            args.refinement_projection_first_radius_px
        ),
        refinement_heldout_candidate_validation=(
            args.refinement_heldout_candidate_validation
        ),
        refinement_spatial_jackknife_diagnostic=(
            args.refinement_spatial_jackknife_diagnostic
        ),
        refinement_minimum_heldout_relative_energy_gain=(
            args.refinement_minimum_heldout_relative_energy_gain
        ),
        refinement_uncertainty_aware_projection=(
            args.refinement_uncertainty_aware_projection
        ),
        refinement_maximum_uncertainty_projection_gate_px=(
            args.refinement_maximum_uncertainty_projection_gate_px
        ),
        refinement_minimum_changed_inliers=(args.refinement_minimum_changed_inliers),
        refinement_minimum_changed_inlier_fraction=(
            args.refinement_minimum_changed_inlier_fraction
        ),
        refinement_minimum_changed_inlier_spatial_cells=(
            args.refinement_minimum_changed_inlier_spatial_cells
        ),
        refinement_maximum_changed_inlier_median_residual_px=(
            args.refinement_maximum_changed_inlier_median_residual_px
        ),
        match_retention_fraction=args.match_retention_fraction,
        minimum_retained_match_count=args.minimum_retained_match_count,
        minimum_sufficient_confidence_core=(args.minimum_sufficient_confidence_core),
        first_pass_query_cap=args.first_pass_query_cap,
        refinement_expanded_reserve_maximum_inlier_fraction=(
            args.refinement_expanded_reserve_maximum_inlier_fraction
        ),
        core_reserve_refinement=args.core_reserve_refinement,
        core_reserve_reprojection_gate_px=(args.core_reserve_reprojection_gate_px),
        core_reserve_minimum_supported_rows=(args.core_reserve_minimum_supported_rows),
        final_pose_polish_reprojection_px=(
            args.final_pose_polish_reprojection_px
        ),
        final_pose_polish_minimum_inliers=(args.final_pose_polish_minimum_inliers),
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
        or (
            view_conditioned_contract is not None
            and sha256_file(view_conditioned_contract["path"])
            != view_conditioned_contract["sha256"]
        )
    ):
        raise RuntimeError("evaluation input artifact changed while localization ran")
    result["summary"].update(
        {
            "evaluated_split": args.split,
            "random_seed": int(args.seed),
            "ransac_confidence": float(ransac_confidence),
            "ransac_maximum_iterations": int(ransac_maximum_iterations),
            "ransac_minimum_iterations": int(ransac_minimum_iterations),
            "ransac_hypothesis_core_size": int(args.ransac_hypothesis_core_size),
            "superpoint_subpixel_refinement": bool(
                args.superpoint_subpixel_refinement
            ),
            "superpoint_subpixel_geometry_only": bool(
                args.superpoint_subpixel_geometry_only
            ),
            "superpoint_subpixel_maximum_offset": float(
                args.superpoint_subpixel_maximum_offset
            ),
            "final_pose_polish_reprojection_px": float(
                args.final_pose_polish_reprojection_px
            ),
            "final_pose_polish_minimum_inliers": int(
                args.final_pose_polish_minimum_inliers
            ),
            "confidence_core_progressive_sampling": bool(
                args.confidence_core_progressive_sampling
            ),
            "input_map_path": artifact_contract["map"]["path"],
            "input_map_sha256": artifact_contract["map"]["sha256"],
            "input_descriptor_state_role": artifact_contract["descriptor_state"][
                "role"
            ],
            "input_descriptor_state_path": artifact_contract["descriptor_state"][
                "path"
            ],
            "input_descriptor_state_sha256": artifact_contract["descriptor_state"][
                "sha256"
            ],
            "view_conditioned_anchor_state": view_conditioned_contract,
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
                "keypoint_count_override": (
                    int(args.keypoint_count_override)
                    if args.keypoint_count_override is not None
                    else None
                ),
                "nms_radius": int(deployment["nms"]),
                "superpoint_subpixel_refinement": bool(
                    args.superpoint_subpixel_refinement
                ),
                "superpoint_subpixel_geometry_only": bool(
                    args.superpoint_subpixel_geometry_only
                ),
                "superpoint_subpixel_maximum_offset": float(
                    args.superpoint_subpixel_maximum_offset
                ),
                "ransac_reprojection_px": float(reprojection_error_px),
                "ransac_confidence": float(ransac_confidence),
                "ransac_maximum_iterations": int(ransac_maximum_iterations),
                "ransac_minimum_iterations": int(ransac_minimum_iterations),
                "ransac_hypothesis_core_size": int(
                    args.ransac_hypothesis_core_size
                ),
                "scene_calibration": (
                    str(calibration_path.resolve())
                    if calibration_path is not None
                    else None
                ),
                "calibration_split": "mapping",
                "evaluated_split": args.split,
                "pose_solves": (
                    "one PoseLib RANSAC plus at most one local nonlinear refinement"
                    if args.core_reserve_refinement
                    else "one PoseLib RANSAC plus one local nonlinear refinement"
                    if args.pose_conditioned_sparse_refinement
                    and args.refinement_pose_backend == "local"
                    else "one plus at most one bounded PoseLib RANSAC"
                    if args.pose_conditioned_sparse_refinement
                    else "one plus at most one feedback solve"
                    if args.topk_geometric_feedback
                    or args.sparse_lgcv_topk_feedback
                    or args.pose_conditioned_sparse_refinement
                    else 1
                ),
                "duplicate_anchor_suppression": bool(args.suppress_duplicate_anchors),
                "guided_sampling": bool(args.guided_sampling),
                "confidence_core_progressive_sampling": bool(
                    args.confidence_core_progressive_sampling
                ),
                "group_aware_pose": bool(args.group_aware_pose),
                "group_field": args.group_field if args.group_aware_pose else None,
                "group_hypothesis_samples": (
                    int(args.group_hypothesis_samples) if args.group_aware_pose else 0
                ),
                "capacity_assignment": bool(args.assignment_topk > 0),
                "topk_geometric_feedback": bool(args.topk_geometric_feedback),
                "sparse_lgcv_topk_feedback": bool(args.sparse_lgcv_topk_feedback),
                "pose_conditioned_sparse_refinement": bool(
                    args.pose_conditioned_sparse_refinement
                ),
                "mapping_view_conditioned_anchor_descriptors": bool(
                    args.view_conditioned_anchor_state is not None
                ),
                "view_conditioned_minimum_concentration": (
                    float(args.view_conditioned_minimum_concentration)
                    if args.view_conditioned_anchor_state is not None
                    else None
                ),
                "view_conditioned_residual_scale": (
                    float(args.view_conditioned_residual_scale)
                    if args.view_conditioned_anchor_state is not None
                    else None
                ),
                "view_conditioned_require_two_valid_modes": bool(
                    args.view_conditioned_require_two_valid_modes
                ),
                "view_conditioned_score_fusion": (
                    args.view_conditioned_score_fusion
                    if args.view_conditioned_anchor_state is not None
                    else None
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
                "refinement_projection_gate_px": float(
                    args.refinement_projection_gate_px
                ),
                "refinement_uncertainty_projection_gate_px": float(
                    args.refinement_uncertainty_projection_gate_px
                ),
                "refinement_uncertainty_maximum_baseline_inliers": int(
                    args.refinement_uncertainty_maximum_baseline_inliers
                ),
                "refinement_maximum_score_drop_from_top1": float(
                    args.refinement_maximum_score_drop_from_top1
                ),
                "refinement_reliability_adaptive_score_drop": bool(
                    args.refinement_reliability_adaptive_score_drop
                ),
                "refinement_reliability_expanded_score_drop": float(
                    args.refinement_reliability_expanded_score_drop
                ),
                "refinement_reliability_minimum_matchability_quantile": float(
                    args.refinement_reliability_minimum_matchability_quantile
                ),
                "refinement_reliability_maximum_uncertainty_quantile": float(
                    args.refinement_reliability_maximum_uncertainty_quantile
                ),
                "refinement_reliability_maximum_geometry_cost": float(
                    args.refinement_reliability_maximum_geometry_cost
                ),
                "refinement_reliability_minimum_improvement_px": float(
                    args.refinement_reliability_minimum_improvement_px
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
                "refinement_active_row_retrieval": bool(
                    args.refinement_active_row_retrieval
                ),
                "refinement_pre_topk_view_filter": bool(
                    args.refinement_pre_topk_view_filter
                ),
                "refinement_common_candidate_grid_gate": bool(
                    args.refinement_common_candidate_grid_gate
                ),
                "refinement_minimum_common_grid_relative_energy_gain": float(
                    args.refinement_minimum_common_grid_relative_energy_gain
                ),
                "refinement_progressive_sampling": bool(
                    args.refinement_progressive_sampling
                ),
                "refinement_allow_soft_inliers": bool(
                    args.refinement_allow_soft_inliers
                ),
                "refinement_soft_inlier_minimum_residual_px": float(
                    args.refinement_soft_inlier_minimum_residual_px
                ),
                "refinement_soft_inlier_maximum_score_drop": float(
                    args.refinement_soft_inlier_maximum_score_drop
                ),
                "refinement_soft_inlier_minimum_improvement_px": float(
                    args.refinement_soft_inlier_minimum_improvement_px
                ),
                "refinement_maximum_soft_inlier_changes": int(
                    args.refinement_maximum_soft_inlier_changes
                ),
                "refinement_pose_conditioned_mutual_matching": bool(
                    args.refinement_pose_conditioned_mutual_matching
                ),
                "refinement_set_level_reserve_selection": bool(
                    args.refinement_set_level_reserve_selection
                ),
                "refinement_projection_first_local_candidates": bool(
                    args.refinement_projection_first_local_candidates
                ),
                "refinement_projection_first_radius_px": float(
                    args.refinement_projection_first_radius_px
                ),
                "refinement_heldout_candidate_validation": bool(
                    args.refinement_heldout_candidate_validation
                ),
                "refinement_spatial_jackknife_diagnostic": bool(
                    args.refinement_spatial_jackknife_diagnostic
                ),
                "refinement_minimum_heldout_relative_energy_gain": float(
                    args.refinement_minimum_heldout_relative_energy_gain
                ),
                "refinement_uncertainty_aware_projection": bool(
                    args.refinement_uncertainty_aware_projection
                ),
                "refinement_maximum_uncertainty_projection_gate_px": float(
                    args.refinement_maximum_uncertainty_projection_gate_px
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
                "match_retention_fraction": float(args.match_retention_fraction),
                "minimum_retained_match_count": int(args.minimum_retained_match_count),
                "minimum_sufficient_confidence_core": bool(
                    args.minimum_sufficient_confidence_core
                ),
                "first_pass_query_cap": int(args.first_pass_query_cap),
                "refinement_expanded_reserve_maximum_inlier_fraction": float(
                    args.refinement_expanded_reserve_maximum_inlier_fraction
                ),
                "core_reserve_refinement": bool(args.core_reserve_refinement),
                "core_reserve_reprojection_gate_px": float(
                    args.core_reserve_reprojection_gate_px
                ),
                "core_reserve_minimum_supported_rows": int(
                    args.core_reserve_minimum_supported_rows
                ),
                "final_pose_polish_reprojection_px": float(
                    args.final_pose_polish_reprojection_px
                ),
                "final_pose_polish_minimum_inliers": int(
                    args.final_pose_polish_minimum_inliers
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
