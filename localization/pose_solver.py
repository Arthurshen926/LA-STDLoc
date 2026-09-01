"""Single standard PoseLib PnP/RANSAC solve used at deployment."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import poselib
import cv2


@dataclass(frozen=True)
class PoseEstimate:
    pose_w2c: np.ndarray
    inliers: np.ndarray
    diagnostics: dict


def camera_intrinsics(
    fov_x: float, fov_y: float, width: int, height: int
) -> np.ndarray:
    focal_x = float(width) / (2.0 * math.tan(float(fov_x) / 2.0))
    focal_y = float(height) / (2.0 * math.tan(float(fov_y) / 2.0))
    return np.asarray(
        [[focal_x, 0.0, width / 2], [0.0, focal_y, height / 2], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )


def poselib_camera(intrinsic: np.ndarray) -> dict:
    """Materialize the immutable PoseLib PINHOLE camera for one calibration."""
    intrinsic = np.asarray(intrinsic)
    return {
        "model": "PINHOLE",
        "width": int(intrinsic[0, 2] * 2),
        "height": int(intrinsic[1, 2] * 2),
        "params": [
            intrinsic[0, 0],
            intrinsic[1, 1],
            intrinsic[0, 2],
            intrinsic[1, 2],
        ],
    }


def solve_absolute_pose(
    points_2d: np.ndarray,
    points_3d: np.ndarray,
    intrinsic: np.ndarray,
    *,
    reprojection_error_px: float = 12.0,
    confidence: float = 0.99999,
    max_iterations: int = 100000,
    min_iterations: int = 1000,
    seed: int = 2026,
    progressive_sampling: bool = False,
    camera: dict | None = None,
) -> PoseEstimate:
    points_2d = np.asarray(points_2d)
    points_3d = np.asarray(points_3d)
    intrinsic = np.asarray(intrinsic)
    if points_2d.shape[0] < 4:
        return PoseEstimate(
            np.eye(4, dtype=np.float32),
            np.empty(0, dtype=np.int64),
            {"iterations": 0, "num_inliers": 0},
        )
    if camera is None:
        camera = poselib_camera(intrinsic)
    pose, info = poselib.estimate_absolute_pose(
        points_2d,
        points_3d,
        camera,
        {
            "max_iterations": int(max_iterations),
            "min_iterations": int(min_iterations),
            "max_reproj_error": float(reprojection_error_px),
            "success_prob": float(confidence),
            "progressive_sampling": bool(progressive_sampling),
            "max_prosac_iterations": int(max_iterations),
            "seed": int(seed),
        },
        {"verbose": False},
    )
    if int(info["num_inliers"]) <= 0:
        return PoseEstimate(
            np.eye(4, dtype=np.float32),
            np.empty(0, dtype=np.int64),
            dict(info),
        )
    pose_w2c = np.concatenate((pose.Rt, np.asarray([[0, 0, 0, 1]])), axis=0).astype(
        np.float32
    )
    inliers = np.flatnonzero(np.asarray(info["inliers"]))
    return PoseEstimate(pose_w2c, inliers, dict(info))


def refine_absolute_pose_from_initial(
    points_2d: np.ndarray,
    points_3d: np.ndarray,
    intrinsic: np.ndarray,
    initial_pose_w2c: np.ndarray,
    optimization_rows: np.ndarray,
    *,
    reprojection_error_px: float = 12.0,
    camera: dict | None = None,
) -> PoseEstimate:
    """Run one local PoseLib non-linear refinement from a trusted first pose.

    The caller chooses the sparse optimization rows.  Inliers are then scored
    over the complete candidate correspondence registry so the return type is
    compatible with the standard robust solver.
    """

    points_2d = np.asarray(points_2d, dtype=np.float64)
    points_3d = np.asarray(points_3d, dtype=np.float64)
    intrinsic = np.asarray(intrinsic, dtype=np.float64)
    initial_pose = np.asarray(initial_pose_w2c, dtype=np.float64)
    rows = np.asarray(optimization_rows, dtype=np.int64).reshape(-1)
    if not (
        points_2d.ndim == 2
        and points_2d.shape[1] == 2
        and points_3d.shape == (points_2d.shape[0], 3)
        and intrinsic.shape == (3, 3)
        and initial_pose.shape == (4, 4)
        and rows.size >= 4
        and int(rows.min()) >= 0
        and int(rows.max()) < points_2d.shape[0]
        and np.unique(rows).size == rows.size
        and np.isfinite(points_2d).all()
        and np.isfinite(points_3d).all()
        and np.isfinite(intrinsic).all()
        and np.isfinite(initial_pose).all()
        and float(reprojection_error_px) > 0.0
    ):
        raise ValueError("local absolute-pose refinement inputs are invalid")
    if camera is None:
        camera = poselib_camera(intrinsic)
    initial = poselib.CameraPose()
    initial.R = initial_pose[:3, :3]
    initial.t = initial_pose[:3, 3]
    refined, info = poselib.refine_absolute_pose(
        points_2d[rows],
        points_3d[rows],
        initial,
        camera,
        {"verbose": False},
    )
    pose_w2c = np.concatenate(
        (refined.Rt, np.asarray([[0.0, 0.0, 0.0, 1.0]])), axis=0
    ).astype(np.float32)
    camera_points = (
        pose_w2c[:3, :3] @ points_3d.T + pose_w2c[:3, 3:4]
    ).T
    depth = camera_points[:, 2]
    projected = np.full_like(points_2d, np.inf)
    valid = depth > 1e-12
    homogeneous = (intrinsic @ camera_points[valid].T).T
    projected[valid] = homogeneous[:, :2] / homogeneous[:, 2:3]
    residual = np.linalg.norm(projected - points_2d, axis=1)
    inliers = np.flatnonzero(residual <= float(reprojection_error_px))
    diagnostics = dict(info)
    diagnostics.update(
        {
            "iterations": int(info.get("iterations", 0)),
            "num_inliers": int(inliers.size),
            "local_refinement": True,
            "optimization_correspondence_count": int(rows.size),
        }
    )
    return PoseEstimate(pose_w2c, inliers, diagnostics)


def solve_group_diverse_absolute_pose(
    points_2d: np.ndarray,
    points_3d: np.ndarray,
    intrinsic: np.ndarray,
    group_ids: np.ndarray,
    *,
    reprojection_error_px: float = 12.0,
    confidence: float = 0.99999,
    max_iterations: int = 100000,
    min_iterations: int = 1000,
    seed: int = 2026,
    group_hypothesis_samples: int = 32,
) -> PoseEstimate:
    """Supplement standard PoseLib with bounded distinct-group AP3P samples.

    The standard PoseLib result remains an explicit candidate and is returned
    byte-for-byte when no group-diverse candidate has a better standard inlier
    score.  This keeps one robust-pose wrapper and changes only hypothesis
    generation; it does not install the unsupported group-capped scorer.
    """

    points_2d = np.asarray(points_2d)
    points_3d = np.asarray(points_3d)
    intrinsic = np.asarray(intrinsic)
    groups = np.asarray(group_ids)
    if groups.dtype.kind not in "iu" or groups.shape != (points_2d.shape[0],):
        raise ValueError("one integer correlation group is required per match")
    if int(group_hypothesis_samples) <= 0:
        raise ValueError("group hypothesis sample count must be positive")
    baseline = solve_absolute_pose(
        points_2d,
        points_3d,
        intrinsic,
        reprojection_error_px=reprojection_error_px,
        confidence=confidence,
        max_iterations=max_iterations,
        min_iterations=min_iterations,
        seed=seed,
    )
    if points_2d.shape[0] < 4:
        return baseline

    # Runtime import avoids a module cycle: the offline oracle uses pose_error
    # from this module, while this bounded deployment experiment reuses its
    # audited sampler and standard scorer.
    from localization.group_consensus import (
        build_group_diverse_hypotheses,
        reprojection_residuals,
        score_hypothesis_residuals,
        select_standard_hypothesis,
    )

    diverse = build_group_diverse_hypotheses(
        points_2d,
        points_3d,
        intrinsic,
        groups,
        sample_count=int(group_hypothesis_samples),
        seed=int(seed),
    )
    if diverse.shape[0] == 0:
        return baseline
    candidates = np.concatenate((baseline.pose_w2c[None], diverse), axis=0)
    residuals = reprojection_residuals(candidates, points_2d, points_3d, intrinsic)
    scores = score_hypothesis_residuals(
        residuals,
        groups,
        threshold_px=float(reprojection_error_px),
    )
    winner = select_standard_hypothesis(scores)
    if winner == 0:
        diagnostics = dict(baseline.diagnostics)
        diagnostics.update(
            {
                "group_diverse_candidates": int(diverse.shape[0]),
                "group_diverse_selected": False,
            }
        )
        return PoseEstimate(baseline.pose_w2c, baseline.inliers, diagnostics)

    selected = candidates[winner]
    inliers = np.flatnonzero(residuals[winner] <= float(reprojection_error_px))
    if inliers.size >= 4:
        camera = {
            "model": "PINHOLE",
            "width": int(intrinsic[0, 2] * 2),
            "height": int(intrinsic[1, 2] * 2),
            "params": [
                intrinsic[0, 0],
                intrinsic[1, 1],
                intrinsic[0, 2],
                intrinsic[1, 2],
            ],
        }
        initial = poselib.CameraPose()
        initial.R = selected[:3, :3]
        initial.t = selected[:3, 3]
        refined, refine_info = poselib.refine_absolute_pose(
            points_2d[inliers], points_3d[inliers], initial, camera, {"verbose": False}
        )
        selected = np.concatenate(
            (refined.Rt, np.asarray([[0, 0, 0, 1]], dtype=np.float64)), axis=0
        )
        refined_residual = reprojection_residuals(
            selected[None], points_2d, points_3d, intrinsic
        )[0]
        inliers = np.flatnonzero(refined_residual <= float(reprojection_error_px))
    else:
        refine_info = {}
    final_candidates = np.stack((baseline.pose_w2c, selected), axis=0)
    final_residuals = reprojection_residuals(
        final_candidates, points_2d, points_3d, intrinsic
    )
    final_scores = score_hypothesis_residuals(
        final_residuals, groups, threshold_px=float(reprojection_error_px)
    )
    if select_standard_hypothesis(final_scores) == 0:
        diagnostics = dict(baseline.diagnostics)
        diagnostics.update(
            {
                "group_diverse_candidates": int(diverse.shape[0]),
                "group_diverse_selected": False,
                "group_diverse_refinement_rejected": True,
            }
        )
        return PoseEstimate(baseline.pose_w2c, baseline.inliers, diagnostics)
    diagnostics = dict(baseline.diagnostics)
    diagnostics.update(
        {
            "num_inliers": int(inliers.size),
            "group_diverse_candidates": int(diverse.shape[0]),
            "group_diverse_selected": True,
            "group_diverse_refinement": dict(refine_info),
        }
    )
    return PoseEstimate(selected.astype(np.float32), inliers, diagnostics)


def solve_pose(
    points_2d,
    points_3d,
    intrinsic,
    *,
    solver: str = "poselib",
    reprojection_error: float = 12.0,
    confidence: float = 0.99999,
    max_iterations: int = 100000,
    min_iterations: int = 1000,
    ransac_seed: int = 2026,
    scores=None,
    return_diagnostics: bool = False,
    **unsupported,
):
    """Compatibility surface for offline evidence code.

    The release pipeline deliberately supports one solver and rejects all
    research-only sampling options instead of silently ignoring them.
    """
    if solver != "poselib":
        raise ValueError("LaFGS release supports only the standard PoseLib solver")
    if scores is not None:
        scores = np.asarray(scores).reshape(-1)
        if scores.shape[0] != np.asarray(points_2d).shape[0]:
            raise ValueError("one retrieval score is required per correspondence")
        # Scores are retained in the offline evidence interface for artifact
        # parity, but standard PoseLib sampling is deliberately score-agnostic.

    def is_disabled(value) -> bool:
        if value is None or value is False:
            return True
        if isinstance(value, (int, float, np.number)):
            return bool(value == 0)
        return False

    active = {
        key: value for key, value in unsupported.items() if not is_disabled(value)
    }
    if active:
        raise ValueError(f"Unsupported custom pose-solver options: {sorted(active)}")
    estimate = solve_absolute_pose(
        points_2d,
        points_3d,
        intrinsic,
        reprojection_error_px=reprojection_error,
        confidence=confidence,
        max_iterations=max_iterations,
        min_iterations=min_iterations,
        seed=ransac_seed,
    )
    if return_diagnostics:
        return estimate.pose_w2c, estimate.inliers, estimate.diagnostics
    return estimate.pose_w2c, estimate.inliers


def pose_error(
    predicted_w2c: np.ndarray, ground_truth_w2c: np.ndarray
) -> tuple[float, float]:
    predicted = np.asarray(predicted_w2c)
    ground_truth = np.asarray(ground_truth_w2c)
    rotation_error = ground_truth[:3, :3] @ predicted[:3, :3].T
    angle_degrees = float(
        np.linalg.norm(cv2.Rodrigues(rotation_error)[0]) * 180.0 / math.pi
    )
    predicted_center = np.linalg.inv(predicted)[:3, 3]
    ground_truth_center = np.linalg.inv(ground_truth)[:3, 3]
    translation_centimeters = float(
        np.linalg.norm(predicted_center - ground_truth_center) * 100.0
    )
    return angle_degrees, translation_centimeters


cal_pose_error = pose_error
