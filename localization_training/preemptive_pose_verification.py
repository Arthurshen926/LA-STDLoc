"""PoseLib-parity absolute pose with preemptive residual verification."""

from __future__ import annotations

import numpy as np

try:
    from localization_training import _lafgs_poselib
except ImportError:
    _lafgs_poselib = None


def compiled_preemptive_backend_available() -> bool:
    return bool(
        _lafgs_poselib is not None
        and hasattr(_lafgs_poselib, "solve_preemptive_absolute_pose")
    )


def solve_preemptive_absolute_pose(
    points2d,
    points3d,
    K,
    *,
    verification_priorities=None,
    reprojection_error: float = 12.0,
    confidence: float = 0.99999,
    max_iterations: int = 100000,
    min_iterations: int = 1000,
    progressive_sampling: bool = False,
    max_prosac_iterations: int = 100000,
    check_interval: int = 32,
    seed: int = 0,
):
    """Run the unchanged PoseLib pipeline with safe score pruning only."""

    points2d = np.asarray(points2d, dtype=np.float64).reshape(-1, 2)
    points3d = np.asarray(points3d, dtype=np.float64).reshape(-1, 3)
    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    count = len(points2d)
    if len(points3d) != count:
        raise ValueError("2D and 3D correspondences must align")
    if verification_priorities is None:
        priorities = np.zeros(count, dtype=np.float64)
    else:
        priorities = np.asarray(
            verification_priorities, dtype=np.float64
        ).reshape(-1)
        if len(priorities) != count:
            raise ValueError(
                "verification priorities must align with correspondences"
            )
    if count < 4:
        return np.eye(4, dtype=np.float32), np.empty(0, dtype=np.int32), {
            "iterations": 0,
            "refinements": 0,
            "residual_evaluations": 0,
            "full_residual_evaluations": 0,
            "residual_evaluation_reduction": 0.0,
            "backend": "cpp" if compiled_preemptive_backend_available() else "unavailable",
        }
    if not compiled_preemptive_backend_available():
        raise RuntimeError(
            "preemptive PoseLib verification requires the compiled LaFGS "
            "extension; run scripts/build_lafgs_poselib.sh"
        )
    pose, inliers, diagnostics = (
        _lafgs_poselib.solve_preemptive_absolute_pose(
            points2d,
            points3d,
            K,
            priorities,
            float(reprojection_error),
            float(confidence),
            int(max_iterations),
            int(min_iterations),
            bool(progressive_sampling),
            int(max_prosac_iterations),
            int(check_interval),
            int(seed),
        )
    )
    return (
        np.asarray(pose, dtype=np.float32),
        np.asarray(inliers, dtype=np.int32),
        dict(diagnostics),
    )
