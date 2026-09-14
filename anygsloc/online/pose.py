"""Frozen numerical kernels extracted without algorithm changes."""

from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np
import poselib


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
