"""Dependency-aware minimal-set RANSAC for calibrated absolute pose."""

from __future__ import annotations

import math

import numpy as np
import poselib

try:
    from localization_training import _lafgs_poselib
except ImportError:
    _lafgs_poselib = None


def compiled_backend_available() -> bool:
    return _lafgs_poselib is not None


def _diverse_set(
    indices: np.ndarray,
    dependency_groups: np.ndarray,
    image_cells: np.ndarray,
    surface_groups: np.ndarray,
    points3d: np.ndarray,
    scene_scale: float,
) -> bool:
    triangle = points3d[indices]
    edges = triangle[[1, 2, 0]] - triangle[[0, 1, 2]]
    maximum_extent = np.linalg.norm(edges, axis=1).max()
    area = 0.5 * np.linalg.norm(
        np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
    )
    return (
        indices.size == 3
        and np.unique(dependency_groups[indices]).size == 3
        and np.unique(image_cells[indices]).size >= 3
        and np.unique(surface_groups[indices]).size >= 2
        and maximum_extent >= 0.02 * scene_scale
        and area >= 1e-4 * scene_scale * scene_scale
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


def _pose_error(pose, ground_truth_w2c: np.ndarray) -> tuple[float, float]:
    estimated = np.eye(4, dtype=np.float64)
    estimated[:3] = np.asarray(pose.Rt, dtype=np.float64)
    ground_truth = np.asarray(ground_truth_w2c, dtype=np.float64).reshape(4, 4)
    relative = estimated[:3, :3] @ ground_truth[:3, :3].T
    cosine = np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0)
    re_deg = float(np.degrees(np.arccos(cosine)))
    estimated_center = -estimated[:3, :3].T @ estimated[:3, 3]
    ground_truth_center = -ground_truth[:3, :3].T @ ground_truth[:3, 3]
    te_cm = float(np.linalg.norm(estimated_center - ground_truth_center) * 100.0)
    return te_cm, re_deg


def solve_dependency_absolute_pose(
    points2d,
    points3d,
    K,
    *,
    dependency_groups,
    image_cells,
    surface_groups,
    sampling_scores=None,
    sampling_margins=None,
    sampling_keypoint_scores=None,
    guided_mixture: float = 0.0,
    guided_rank_power: float = 0.5,
    ground_truth_w2c=None,
    minimal_set_record_limit: int = 0,
    reprojection_error: float = 12.0,
    confidence: float = 0.99999,
    max_iterations: int = 8000,
    min_iterations: int = 500,
    rescue_max_iterations: int = 0,
    rescue_inlier_ratio: float = 0.0,
    seed: int = 0,
    backend: str = "auto",
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
    dependency_groups, image_cells, surface_groups = metadata
    if backend not in {"auto", "cpp", "python"}:
        raise ValueError("backend must be auto, cpp, or python")
    use_cpp = (
        backend != "python"
        and _lafgs_poselib is not None
        and ground_truth_w2c is None
        and int(minimal_set_record_limit) == 0
    )
    if backend == "cpp" and not use_cpp:
        raise RuntimeError(
            "compiled backend unavailable or incompatible with teacher recording"
        )
    if use_cpp:
        w2c, inliers, info = _lafgs_poselib.solve_dependency_absolute_pose(
            points2d,
            points3d,
            K,
            dependency_groups,
            image_cells,
            surface_groups,
            (
                np.asarray(sampling_scores, dtype=np.float64).reshape(-1)
                if sampling_scores is not None
                else np.empty(0, dtype=np.float64)
            ),
            float(guided_mixture),
            float(guided_rank_power),
            float(reprojection_error),
            float(confidence),
            int(max_iterations),
            int(min_iterations),
            int(rescue_max_iterations),
            float(rescue_inlier_ratio),
            int(seed),
        )
        info = dict(info)
        info["minimal_set_records"] = []
        return (
            np.asarray(w2c, dtype=np.float32),
            np.asarray(inliers, dtype=np.int32),
            info,
        )
    scene_center = np.median(points3d, axis=0)
    scene_scale = float(
        np.median(np.linalg.norm(points3d - scene_center, axis=1))
    )
    scene_scale = max(scene_scale, 1e-6)
    homogeneous = np.concatenate((points2d, np.ones((count, 1))), axis=1)
    bearings = homogeneous @ np.linalg.inv(K).T
    bearings /= np.linalg.norm(bearings, axis=1, keepdims=True).clip(min=1e-12)
    rng = np.random.default_rng(int(seed))
    sampling_probability = None
    if sampling_scores is not None and float(guided_mixture) > 0:
        if float(guided_rank_power) <= 0:
            raise ValueError("guided_rank_power must be positive")
        sampling_scores = np.asarray(sampling_scores, dtype=np.float64).reshape(-1)
        if sampling_scores.size != count:
            raise ValueError("sampling scores must align with correspondences")
        score_order = np.argsort(-sampling_scores, kind="stable")
        ranks = np.empty(count, dtype=np.float64)
        ranks[score_order] = np.arange(1, count + 1, dtype=np.float64)
        guided = 1.0 / np.power(ranks, float(guided_rank_power))
        guided /= guided.sum()
        mixture = min(max(float(guided_mixture), 0.0), 1.0)
        sampling_probability = mixture * guided + (1.0 - mixture) / count
    if sampling_margins is not None:
        sampling_margins = np.asarray(
            sampling_margins, dtype=np.float64
        ).reshape(-1)
        if sampling_margins.size != count:
            raise ValueError("sampling margins must align with correspondences")
    if sampling_keypoint_scores is not None:
        sampling_keypoint_scores = np.asarray(
            sampling_keypoint_scores, dtype=np.float64
        ).reshape(-1)
        if sampling_keypoint_scores.size != count:
            raise ValueError("keypoint scores must align with correspondences")
    sampling_cdf = (
        np.cumsum(sampling_probability)
        if sampling_probability is not None
        else None
    )

    def sample_triplet() -> np.ndarray:
        if sampling_cdf is None:
            return rng.choice(count, size=3, replace=False)
        for _ in range(8):
            proposed = np.searchsorted(
                sampling_cdf, rng.random(3), side="right"
            )
            if np.unique(proposed).size == 3:
                return proposed
        return rng.choice(count, size=3, replace=False, p=sampling_probability)
    best_pose = None
    best_inliers = np.empty(0, dtype=np.int64)
    best_cost = np.inf
    base_max_iterations = int(max_iterations)
    hard_max_iterations = max(
        base_max_iterations, int(rescue_max_iterations)
    )
    target_iterations = base_max_iterations
    diverse_samples = 0
    fallback_samples = 0
    local_refinements = 0
    minimal_set_records = []
    record_stride = max(
        int(max_iterations) // max(int(minimal_set_record_limit), 1), 1
    )
    camera = {
        "model": "PINHOLE",
        "width": int(round(K[0, 2] * 2)),
        "height": int(round(K[1, 2] * 2)),
        "params": [K[0, 0], K[1, 1], K[0, 2], K[1, 2]],
    }
    iteration = 0
    while iteration < max(int(min_iterations), target_iterations):
        sample = None
        for _ in range(32):
            proposed = sample_triplet()
            if _diverse_set(
                proposed,
                dependency_groups,
                image_cells,
                surface_groups,
                points3d,
                scene_scale,
            ):
                sample = proposed
                diverse_samples += 1
                break
        if sample is None:
            sample = sample_triplet()
            fallback_samples += 1
        try:
            hypotheses = poselib.p3p(bearings[sample], points3d[sample])
        except RuntimeError:
            hypotheses = []
        sample_best_inliers = -1
        sample_best_pose = None
        sample_best_cost = np.inf
        for pose in hypotheses:
            errors = _reprojection_errors(pose, points2d, points3d, K)
            inliers = np.flatnonzero(errors <= float(reprojection_error))
            cost = float(
                np.minimum(errors * errors, float(reprojection_error) ** 2).sum()
            )
            if inliers.size > sample_best_inliers or (
                inliers.size == sample_best_inliers and cost < sample_best_cost
            ):
                sample_best_inliers = int(inliers.size)
                sample_best_pose = pose
                sample_best_cost = cost
            if inliers.size > best_inliers.size or (
                inliers.size == best_inliers.size and cost < best_cost
            ):
                best_pose, best_inliers, best_cost = pose, inliers, cost
                if best_inliers.size >= 6:
                    try:
                        refined, _ = poselib.refine_absolute_pose(
                            points2d[best_inliers],
                            points3d[best_inliers],
                            best_pose,
                            camera,
                            {"verbose": False},
                        )
                        refined_errors = _reprojection_errors(
                            refined, points2d, points3d, K
                        )
                        refined_inliers = np.flatnonzero(
                            refined_errors <= float(reprojection_error)
                        )
                        refined_cost = float(
                            np.minimum(
                                refined_errors * refined_errors,
                                float(reprojection_error) ** 2,
                            ).sum()
                        )
                        if refined_inliers.size > best_inliers.size or (
                            refined_inliers.size == best_inliers.size
                            and refined_cost < best_cost
                        ):
                            best_pose = refined
                            best_inliers = refined_inliers
                            best_cost = refined_cost
                        local_refinements += 1
                    except RuntimeError:
                        pass
                ratio = best_inliers.size / count
                if ratio > 0:
                    all_inlier = min(max(ratio**3, 1e-12), 1.0 - 1e-12)
                    required = math.ceil(
                        math.log(1.0 - float(confidence))
                        / math.log(1.0 - all_inlier)
                    )
                    target_iterations = min(int(max_iterations), max(int(min_iterations), required))
        if (
            ground_truth_w2c is not None
            and int(minimal_set_record_limit) > 0
            and iteration % record_stride == 0
            and len(minimal_set_records) < int(minimal_set_record_limit)
        ):
            triangle = points3d[sample]
            area = 0.5 * np.linalg.norm(
                np.cross(
                    triangle[1] - triangle[0],
                    triangle[2] - triangle[0],
                )
            )
            extent = max(
                np.linalg.norm(triangle[1] - triangle[0]),
                np.linalg.norm(triangle[2] - triangle[0]),
                np.linalg.norm(triangle[2] - triangle[1]),
            )
            if sample_best_pose is None:
                te_cm, re_deg = None, None
            else:
                te_cm, re_deg = _pose_error(
                    sample_best_pose, ground_truth_w2c
                )
            minimal_set_records.append(
                {
                    "correspondence_indices": sample.astype(np.int32).tolist(),
                    "inlier_count": max(sample_best_inliers, 0),
                    "te_cm": te_cm,
                    "re_deg": re_deg,
                    "correct_basin": bool(
                        te_cm is not None
                        and te_cm <= 50.0
                        and re_deg <= 5.0
                    ),
                    "image_cell_count": int(
                        np.unique(image_cells[sample]).size
                    ),
                    "dependency_count": int(
                        np.unique(dependency_groups[sample]).size
                    ),
                    "surface_count": int(
                        np.unique(surface_groups[sample]).size
                    ),
                    "normalized_extent": float(extent / scene_scale),
                    "normalized_area": float(area / (scene_scale**2)),
                    "sampling_scores": (
                        np.asarray(sampling_scores)[sample].astype(float).tolist()
                        if sampling_scores is not None
                        else None
                    ),
                    "sampling_margins": (
                        sampling_margins[sample].astype(float).tolist()
                        if sampling_margins is not None
                        else None
                    ),
                    "keypoint_scores": (
                        sampling_keypoint_scores[sample].astype(float).tolist()
                        if sampling_keypoint_scores is not None
                        else None
                    ),
                }
            )
        iteration += 1
        current_ratio = best_inliers.size / count
        if (
            iteration >= base_max_iterations
            and target_iterations <= base_max_iterations
            and hard_max_iterations > base_max_iterations
            and current_ratio < float(rescue_inlier_ratio)
        ):
            target_iterations = hard_max_iterations
        if iteration >= hard_max_iterations:
            break
    if best_pose is None or best_inliers.size < 4:
        return np.eye(4, dtype=np.float32), np.empty(0, dtype=np.int32), {
            "iterations": iteration,
            "diverse_samples": diverse_samples,
            "fallback_samples": fallback_samples,
            "local_refinements": local_refinements,
            "minimal_set_records": minimal_set_records,
            "rescue_used": iteration > base_max_iterations,
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
        "local_refinements": local_refinements,
        "minimal_set_records": minimal_set_records,
        "rescue_used": iteration > base_max_iterations,
    }
