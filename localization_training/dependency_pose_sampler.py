"""Dependency-aware minimal-set RANSAC for calibrated absolute pose."""

from __future__ import annotations

import math

import numpy as np
import poselib


def _diverse_set(
    indices: np.ndarray,
    dependency_groups: np.ndarray,
    image_cells: np.ndarray,
    depth_bins: np.ndarray,
    surface_groups: np.ndarray,
) -> bool:
    return (
        np.unique(dependency_groups[indices]).size >= 3
        and np.unique(image_cells[indices]).size >= 3
        and np.unique(depth_bins[indices]).size >= 2
        and np.unique(surface_groups[indices]).size >= 2
    )


def _reprojection_errors(
    pose,
    points2d: np.ndarray,
    points3d: np.ndarray,
    K: np.ndarray,
) -> np.ndarray:
    rt = np.asarray(pose.Rt, dtype=np.float64)
    camera = points3d @ rt[:, :3].T + rt[:, 3]
    valid = camera[:, 2] > 1e-8
    projected = np.empty_like(points2d)
    projected[:, 0] = K[0, 0] * camera[:, 0] / np.maximum(camera[:, 2], 1e-8) + K[0, 2]
    projected[:, 1] = K[1, 1] * camera[:, 1] / np.maximum(camera[:, 2], 1e-8) + K[1, 2]
    errors = np.linalg.norm(projected - points2d, axis=1)
    errors[~valid] = np.inf
    return errors


def solve_dependency_absolute_pose(
    points2d,
    points3d,
    K,
    *,
    dependency_groups,
    image_cells,
    depth_bins,
    surface_groups,
    reprojection_error: float = 12.0,
    confidence: float = 0.99999,
    max_iterations: int = 8000,
    min_iterations: int = 500,
    seed: int = 0,
):
    """Run one PnP with diversity constraints only on sampled minimal sets."""
    points2d = np.asarray(points2d, dtype=np.float64).reshape(-1, 2)
    points3d = np.asarray(points3d, dtype=np.float64).reshape(-1, 3)
    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    metadata = [
        np.asarray(value, dtype=np.int64).reshape(-1)
        for value in (
            dependency_groups,
            image_cells,
            depth_bins,
            surface_groups,
        )
    ]
    count = points2d.shape[0]
    if points3d.shape[0] != count or any(value.size != count for value in metadata):
        raise ValueError("correspondences and dependency metadata must align")
    if count < 4:
        return np.eye(4, dtype=np.float32), np.empty(0, dtype=np.int32), {
            "iterations": 0,
            "diverse_samples": 0,
            "fallback_samples": 0,
        }
    dependency_groups, image_cells, depth_bins, surface_groups = metadata
    homogeneous = np.concatenate((points2d, np.ones((count, 1))), axis=1)
    bearings = homogeneous @ np.linalg.inv(K).T
    bearings /= np.linalg.norm(bearings, axis=1, keepdims=True).clip(min=1e-12)
    rng = np.random.default_rng(int(seed))
    best_pose = None
    best_inliers = np.empty(0, dtype=np.int64)
    best_cost = np.inf
    target_iterations = int(max_iterations)
    diverse_samples = 0
    fallback_samples = 0
    iteration = 0
    while iteration < max(int(min_iterations), target_iterations):
        sample = None
        for _ in range(32):
            proposed = rng.choice(count, size=4, replace=False)
            if _diverse_set(
                proposed,
                dependency_groups,
                image_cells,
                depth_bins,
                surface_groups,
            ):
                sample = proposed
                diverse_samples += 1
                break
        if sample is None:
            sample = rng.choice(count, size=4, replace=False)
            fallback_samples += 1
        try:
            hypotheses = poselib.p3p(bearings[sample[:3]], points3d[sample[:3]])
        except RuntimeError:
            hypotheses = []
        for pose in hypotheses:
            errors = _reprojection_errors(pose, points2d, points3d, K)
            inliers = np.flatnonzero(errors <= float(reprojection_error))
            cost = float(
                np.minimum(errors * errors, float(reprojection_error) ** 2).sum()
            )
            if inliers.size > best_inliers.size or (
                inliers.size == best_inliers.size and cost < best_cost
            ):
                best_pose, best_inliers, best_cost = pose, inliers, cost
                ratio = inliers.size / count
                if ratio > 0:
                    all_inlier = min(max(ratio**4, 1e-12), 1.0 - 1e-12)
                    required = math.ceil(
                        math.log(1.0 - float(confidence))
                        / math.log(1.0 - all_inlier)
                    )
                    target_iterations = min(int(max_iterations), max(int(min_iterations), required))
        iteration += 1
        if iteration >= int(max_iterations):
            break
    if best_pose is None or best_inliers.size < 4:
        return np.eye(4, dtype=np.float32), np.empty(0, dtype=np.int32), {
            "iterations": iteration,
            "diverse_samples": diverse_samples,
            "fallback_samples": fallback_samples,
        }
    camera = {
        "model": "PINHOLE",
        "width": int(round(K[0, 2] * 2)),
        "height": int(round(K[1, 2] * 2)),
        "params": [K[0, 0], K[1, 1], K[0, 2], K[1, 2]],
    }
    refined, _ = poselib.refine_absolute_pose(
        points2d[best_inliers],
        points3d[best_inliers],
        best_pose,
        camera,
        {"verbose": False},
    )
    errors = _reprojection_errors(refined, points2d, points3d, K)
    best_inliers = np.flatnonzero(errors <= float(reprojection_error))
    w2c = np.eye(4, dtype=np.float32)
    w2c[:3] = np.asarray(refined.Rt, dtype=np.float32)
    return w2c, best_inliers.astype(np.int32), {
        "iterations": iteration,
        "diverse_samples": diverse_samples,
        "fallback_samples": fallback_samples,
    }
