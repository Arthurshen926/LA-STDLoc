"""Deployment-identical counterfactual pose supervision."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch

from utils.pose_utils import cal_pose_error, solve_pose


@dataclass(frozen=True)
class ExactCounterfactualConfig:
    reprojection_error_px: float = 12.0
    confidence: float = 0.99999
    maximum_iterations: int = 100000
    minimum_iterations: int = 1000
    seed: int = 2026
    clean_reprojection_px: float = 4.0
    strict_translation_cm: float = 5.0
    basin_translation_cm: float = 50.0
    basin_rotation_degrees: float = 5.0
    maximum_candidates_per_row: int = 8


def _project_errors(
    points3d: np.ndarray,
    points2d: np.ndarray,
    intrinsics: np.ndarray,
    pose_w2c: np.ndarray,
) -> np.ndarray:
    points3d = np.asarray(points3d, dtype=np.float64).reshape(-1, 3)
    points2d = np.asarray(points2d, dtype=np.float64).reshape(-1, 2)
    intrinsics = np.asarray(intrinsics, dtype=np.float64).reshape(3, 3)
    pose = np.asarray(pose_w2c, dtype=np.float64).reshape(4, 4)
    camera = points3d @ pose[:3, :3].T + pose[:3, 3]
    valid = camera[:, 2] > 1e-8
    projected = np.empty_like(points2d)
    projected[:, 0] = (
        intrinsics[0, 0]
        * camera[:, 0]
        / np.maximum(camera[:, 2], 1e-8)
        + intrinsics[0, 2]
    )
    projected[:, 1] = (
        intrinsics[1, 1]
        * camera[:, 1]
        / np.maximum(camera[:, 2], 1e-8)
        + intrinsics[1, 2]
    )
    errors = np.linalg.norm(projected - points2d, axis=1)
    errors[~valid] = np.inf
    return errors


def geometry_diversity_score(
    points3d: np.ndarray,
    dependency_groups: np.ndarray,
    source_groups: np.ndarray,
) -> float:
    """Return a dimensionless, query-local tie breaker."""

    points = np.asarray(points3d, dtype=np.float64).reshape(-1, 3)
    dependency = np.asarray(dependency_groups, dtype=np.int64).reshape(-1)
    sources = np.asarray(source_groups, dtype=np.int64).reshape(-1)
    if not (len(points) == len(dependency) == len(sources)):
        raise ValueError("geometry-diversity inputs must align")
    if not len(points):
        return float("-inf")
    centered = points - np.median(points, axis=0, keepdims=True)
    scale = max(float(np.median(np.linalg.norm(centered, axis=1))), 1e-8)
    normalized = centered / scale
    covariance = normalized.T @ normalized / max(len(normalized), 1)
    sign, logdet = np.linalg.slogdet(
        covariance + np.eye(3, dtype=np.float64) * 1e-6
    )
    spread = float(logdet) if sign > 0 else -100.0
    return (
        float(np.unique(dependency).size)
        + 0.1 * float(np.unique(sources).size)
        + 0.01 * spread
    )


def outcome_order_key(outcome: dict) -> tuple:
    """Encode the documented lexicographic target ordering."""

    valid = bool(outcome["valid"])
    return (
        valid,
        bool(outcome["correct_basin"]) if valid else False,
        bool(outcome["strict_translation_success"]) if valid else False,
        -float(outcome["translation_error_cm"]) if valid else float("-inf"),
        -float(outcome["rotation_error_degrees"]) if valid else float("-inf"),
        -int(outcome["harmful_consensus_count"]) if valid else -10**9,
        float(outcome["geometry_diversity"]) if valid else float("-inf"),
    )


def improves_lexicographically(candidate: dict, baseline: dict) -> bool:
    return outcome_order_key(candidate) > outcome_order_key(baseline)


def solve_counterfactual_pose(
    *,
    points2d: np.ndarray,
    points3d: np.ndarray,
    intrinsics: np.ndarray,
    ground_truth_w2c: np.ndarray,
    dependency_groups: np.ndarray,
    source_groups: np.ndarray,
    config: ExactCounterfactualConfig,
) -> dict:
    """Run the same fixed-seed PoseLib solve used by sparse deployment."""

    points2d = np.asarray(points2d, dtype=np.float64).reshape(-1, 2)
    points3d = np.asarray(points3d, dtype=np.float64).reshape(-1, 3)
    intrinsics = np.asarray(intrinsics, dtype=np.float64).reshape(3, 3)
    ground_truth = np.asarray(
        ground_truth_w2c, dtype=np.float64
    ).reshape(4, 4)
    pose, inliers, diagnostics = solve_pose(
        points2d,
        points3d,
        intrinsics,
        solver="poselib",
        reprojection_error=float(config.reprojection_error_px),
        confidence=float(config.confidence),
        max_iterations=int(config.maximum_iterations),
        min_iterations=int(config.minimum_iterations),
        ransac_seed=int(config.seed),
        return_diagnostics=True,
    )
    inliers = np.asarray(inliers, dtype=np.int64).reshape(-1)
    valid = bool(
        inliers.size >= 4
        and np.isfinite(pose).all()
    )
    if valid:
        rotation_error, translation_error = cal_pose_error(
            pose, ground_truth
        )
        ground_truth_errors = _project_errors(
            points3d, points2d, intrinsics, ground_truth
        )
        harmful = ground_truth_errors > float(
            config.clean_reprojection_px
        )
        harmful_consensus = int(harmful[inliers].sum())
    else:
        rotation_error = translation_error = float("inf")
        harmful_consensus = 0
    return {
        "valid": valid,
        "translation_error_cm": float(translation_error),
        "rotation_error_degrees": float(rotation_error),
        "correct_basin": bool(
            valid
            and translation_error <= float(config.basin_translation_cm)
            and rotation_error <= float(config.basin_rotation_degrees)
        ),
        "strict_translation_success": bool(
            valid
            and translation_error <= float(config.strict_translation_cm)
        ),
        "inlier_count": int(inliers.size),
        "harmful_consensus_count": int(harmful_consensus),
        "hypotheses": diagnostics.get("ransac_actual_hypotheses"),
        "inlier_indices": torch.as_tensor(inliers, dtype=torch.int32),
        "geometry_diversity": float(
            geometry_diversity_score(
                points3d, dependency_groups, source_groups
            )
        ),
    }


def serialize_config(config: ExactCounterfactualConfig) -> dict:
    return asdict(config)
