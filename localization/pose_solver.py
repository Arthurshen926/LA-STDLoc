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
