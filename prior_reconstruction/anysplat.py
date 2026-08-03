"""AnySplat feed-forward prior selection, alignment, and PLY conversion."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable

import numpy as np
from plyfile import PlyData, PlyElement
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class SimilarityTransform:
    """Similarity mapping ``target = scale * rotation @ source + translation``."""

    scale: float
    rotation: np.ndarray
    translation: np.ndarray

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=np.float64)
        return self.scale * (points @ self.rotation.T) + self.translation


def colmap_qvec_to_rotation(qvec: np.ndarray) -> np.ndarray:
    """Convert a COLMAP WXYZ quaternion to its world-to-camera rotation."""

    w, x, y, z = np.asarray(qvec, dtype=np.float64)
    return np.asarray(
        [
            [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * z * x + 2 * w * y],
            [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
            [2 * z * x - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y],
        ]
    )


def _frame_sort_key(name: str) -> tuple[str, int, str]:
    path = Path(name)
    match = re.search(r"(\d+)$", path.stem)
    frame = int(match.group(1)) if match else -1
    trajectory = path.parts[0] if len(path.parts) > 1 else ""
    return trajectory, frame, name


def farthest_point_indices(points: np.ndarray, count: int) -> list[int]:
    """Select a deterministic spatially diverse subset of camera centers."""

    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"camera centers must have shape [N,3], got {points.shape}")
    if count <= 0:
        raise ValueError("count must be positive")
    if len(points) <= count:
        return list(range(len(points)))
    centroid = points.mean(axis=0)
    first = int(np.argmax(np.linalg.norm(points - centroid, axis=1)))
    selected = [first]
    minimum_distance = np.linalg.norm(points - points[first], axis=1)
    minimum_distance[first] = -np.inf
    while len(selected) < count:
        index = int(np.argmax(minimum_distance))
        selected.append(index)
        minimum_distance = np.minimum(
            minimum_distance,
            np.linalg.norm(points - points[index], axis=1),
        )
        minimum_distance[selected] = -np.inf
    return sorted(selected)


def select_trajectory_windows(
    names: Iterable[str],
    centers_by_name: dict[str, np.ndarray],
    views_per_trajectory: int,
    segment_size: int = 0,
    complete_coverage: bool = False,
) -> list[dict[str, object]]:
    """Build pose-diverse pure-FF windows from local mapping trajectory segments."""

    grouped: dict[str, list[str]] = {}
    for name in sorted(names, key=_frame_sort_key):
        if name not in centers_by_name:
            raise KeyError(f"missing mapping pose for {name}")
        trajectory = Path(name).parts[0] if len(Path(name).parts) > 1 else "root"
        grouped.setdefault(trajectory, []).append(name)
    windows = []
    for trajectory, trajectory_names in sorted(grouped.items()):
        if complete_coverage:
            maximum_size = segment_size if segment_size > 0 else views_per_trajectory
            segment_count = int(np.ceil(len(trajectory_names) / maximum_size))
            if len(trajectory_names) < 3:
                raise ValueError(
                    f"trajectory {trajectory} has fewer than three mapping views"
                )
            base, remainder = divmod(len(trajectory_names), segment_count)
            segment_sizes = [
                base + (index < remainder) for index in range(segment_count)
            ]
            if min(segment_sizes) < 3 or max(segment_sizes) > views_per_trajectory:
                raise ValueError(
                    f"cannot partition {trajectory} into 3..{views_per_trajectory} "
                    f"view windows: {segment_sizes}"
                )
            segments = []
            start = 0
            for size in segment_sizes:
                segments.append((start, trajectory_names[start : start + size]))
                start += size
        else:
            size = segment_size if segment_size > 0 else len(trajectory_names)
            segments = [
                (start, trajectory_names[start : start + size])
                for start in range(0, len(trajectory_names), size)
            ]
        for segment_index, (start, segment_names) in enumerate(segments):
            centers = np.stack([centers_by_name[name] for name in segment_names])
            indices = farthest_point_indices(centers, views_per_trajectory)
            selected_names = [segment_names[index] for index in indices]
            window_id = (
                trajectory
                if segment_size <= 0
                else f"{trajectory}_{segment_index:03d}"
            )
            windows.append(
                {
                    "window_id": window_id,
                    "trajectory": trajectory,
                    "trajectory_segment_index": segment_index,
                    "trajectory_segment_start": start,
                    "trajectory_segment_stop": start + len(segment_names),
                    "available_view_count": len(segment_names),
                    "selected_view_count": len(selected_names),
                    "image_names": selected_names,
                }
            )
    return windows


def fit_similarity(source: np.ndarray, target: np.ndarray) -> SimilarityTransform:
    """Fit a proper-rotation Umeyama Sim(3) from paired 3D points."""

    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError(
            f"source and target must have matching [N,3] shapes, got "
            f"{source.shape} and {target.shape}"
        )
    if len(source) < 3:
        raise ValueError("at least three camera centers are required for Sim(3)")
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    source_zero = source - source_center
    target_zero = target - target_center
    source_variance = np.mean(np.sum(source_zero * source_zero, axis=1))
    if source_variance <= 1e-12:
        raise ValueError("source camera centers are degenerate")
    covariance = target_zero.T @ source_zero / len(source)
    u, singular_values, vt = np.linalg.svd(covariance)
    sign = np.ones(3, dtype=np.float64)
    if np.linalg.det(u @ vt) < 0:
        sign[-1] = -1.0
    rotation = u @ np.diag(sign) @ vt
    scale = float(np.sum(singular_values * sign) / source_variance)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(f"invalid Sim(3) scale: {scale}")
    translation = target_center - scale * (rotation @ source_center)
    return SimilarityTransform(scale, rotation, translation)


def fit_similarity_robust(
    source: np.ndarray,
    target: np.ndarray,
    *,
    iterations: int = 5,
    minimum_threshold: float = 0.05,
) -> tuple[SimilarityTransform, np.ndarray, np.ndarray]:
    """Fit Sim(3) with deterministic MAD trimming and return inliers/residuals."""

    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    inliers = np.ones(len(source), dtype=bool)
    transform = fit_similarity(source, target)
    for _ in range(iterations):
        residuals = np.linalg.norm(transform.transform_points(source) - target, axis=1)
        median = float(np.median(residuals[inliers]))
        mad = float(np.median(np.abs(residuals[inliers] - median)))
        threshold = max(float(minimum_threshold), median + 3.5 * 1.4826 * mad)
        updated = residuals <= threshold
        if updated.sum() < 3 or np.array_equal(updated, inliers):
            break
        inliers = updated
        transform = fit_similarity(source[inliers], target[inliers])
    residuals = np.linalg.norm(transform.transform_points(source) - target, axis=1)
    return transform, inliers, residuals


def fit_similarity_from_camera_poses(
    source_centers: np.ndarray,
    target_centers: np.ndarray,
    source_c2w_rotations: np.ndarray,
    target_c2w_rotations: np.ndarray,
    *,
    iterations: int = 5,
    minimum_center_threshold: float = 0.05,
) -> tuple[SimilarityTransform, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fit mapping-only Sim(3), using orientations to resolve path degeneracy."""

    source_centers = np.asarray(source_centers, dtype=np.float64)
    target_centers = np.asarray(target_centers, dtype=np.float64)
    source_rotations = np.asarray(source_c2w_rotations, dtype=np.float64)
    target_rotations = np.asarray(target_c2w_rotations, dtype=np.float64)
    if source_rotations.shape != target_rotations.shape or source_rotations.shape != (
        len(source_centers), 3, 3
    ):
        raise ValueError("camera rotations must match center pairs with shape [N,3,3]")
    rotation_candidates = target_rotations @ np.swapaxes(source_rotations, 1, 2)
    orientation_inliers = np.ones(len(source_centers), dtype=bool)
    global_rotation = Rotation.from_matrix(rotation_candidates).mean().as_matrix()
    for _ in range(iterations):
        angular = Rotation.from_matrix(
            np.swapaxes(global_rotation[None], 1, 2) @ rotation_candidates
        ).magnitude()
        median = float(np.median(angular[orientation_inliers]))
        mad = float(np.median(np.abs(angular[orientation_inliers] - median)))
        threshold = max(np.deg2rad(0.5), median + 3.5 * 1.4826 * mad)
        updated = angular <= threshold
        if updated.sum() < 3 or np.array_equal(updated, orientation_inliers):
            break
        orientation_inliers = updated
        global_rotation = Rotation.from_matrix(
            rotation_candidates[orientation_inliers]
        ).mean().as_matrix()

    rotated_source = source_centers @ global_rotation.T
    center_inliers = np.ones(len(source_centers), dtype=bool)

    def fit_scale_translation(mask: np.ndarray) -> tuple[float, np.ndarray]:
        source = rotated_source[mask]
        target = target_centers[mask]
        source_center = source.mean(axis=0)
        target_center = target.mean(axis=0)
        source_zero = source - source_center
        target_zero = target - target_center
        denominator = float(np.sum(source_zero * source_zero))
        if denominator <= 1e-12:
            raise ValueError("source camera centers are degenerate")
        scale = float(np.sum(source_zero * target_zero) / denominator)
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError(f"invalid orientation-constrained Sim(3) scale: {scale}")
        return scale, target_center - scale * source_center

    scale, translation = fit_scale_translation(center_inliers)
    for _ in range(iterations):
        residuals = np.linalg.norm(
            scale * rotated_source + translation - target_centers,
            axis=1,
        )
        median = float(np.median(residuals[center_inliers]))
        mad = float(np.median(np.abs(residuals[center_inliers] - median)))
        threshold = max(
            float(minimum_center_threshold),
            median + 3.5 * 1.4826 * mad,
        )
        updated = residuals <= threshold
        if updated.sum() < 3 or np.array_equal(updated, center_inliers):
            break
        center_inliers = updated
        scale, translation = fit_scale_translation(center_inliers)
    transform = SimilarityTransform(scale, global_rotation, translation)
    center_residuals = np.linalg.norm(
        transform.transform_points(source_centers) - target_centers,
        axis=1,
    )
    orientation_residuals = Rotation.from_matrix(
        np.swapaxes(global_rotation[None], 1, 2) @ rotation_candidates
    ).magnitude()
    return (
        transform,
        center_inliers,
        center_residuals,
        orientation_inliers,
        orientation_residuals,
    )


def transform_gaussian_moments(
    means: np.ndarray,
    covariances: np.ndarray,
    transform: SimilarityTransform,
) -> tuple[np.ndarray, np.ndarray]:
    means = transform.transform_points(means)
    rotation = transform.rotation
    covariances = (
        transform.scale**2
        * np.einsum("ij,njk,lk->nil", rotation, covariances, rotation)
    )
    return means, covariances


def covariance_to_scale_rotation(
    covariances: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert world covariances to Graphdeco log-scales and WXYZ quaternions."""

    covariances = np.asarray(covariances, dtype=np.float64)
    values, vectors = np.linalg.eigh(covariances)
    order = np.argsort(values, axis=1)[:, ::-1]
    values = np.take_along_axis(values, order, axis=1)
    vectors = np.take_along_axis(vectors, order[:, None, :], axis=2)
    values = np.clip(values, 1e-12, None)
    negative = np.linalg.det(vectors) < 0
    vectors[negative, :, -1] *= -1
    quaternion_xyzw = Rotation.from_matrix(vectors).as_quat()
    quaternion_wxyz = quaternion_xyzw[:, [3, 0, 1, 2]]
    return np.log(np.sqrt(values)), quaternion_wxyz


def probability_to_logit(probability: np.ndarray) -> np.ndarray:
    probability = np.clip(np.asarray(probability, dtype=np.float64), 1e-6, 1 - 1e-6)
    return np.log(probability) - np.log1p(-probability)


def spatial_confidence_coreset(
    means: np.ndarray,
    covariances: np.ndarray,
    opacities: np.ndarray,
    budget: int,
) -> np.ndarray:
    """Select an appearance-agnostic spatial/opacity coreset for FF fusion."""

    means = np.asarray(means, dtype=np.float64)
    covariances = np.asarray(covariances, dtype=np.float64)
    opacities = np.asarray(opacities, dtype=np.float64).reshape(-1)
    if budget <= 0 or len(means) <= budget:
        return np.arange(len(means), dtype=np.int64)
    sample_count = min(len(covariances), 8192)
    sample_indices = np.linspace(
        0, len(covariances) - 1, sample_count, dtype=np.int64
    )
    eigenvalues = np.linalg.eigvalsh(covariances[sample_indices])
    characteristic_scale = float(
        np.median(np.sqrt(np.clip(eigenvalues[:, -1], 1e-12, None)))
    )
    voxel_size = max(characteristic_scale * 1.5, 1e-6)
    keys = np.floor(means / voxel_size).astype(np.int64)
    confidence_order = np.argsort(-opacities, kind="stable")
    _, first = np.unique(keys[confidence_order], axis=0, return_index=True)
    spatial = confidence_order[first]
    spatial = spatial[np.argsort(-opacities[spatial], kind="stable")]
    if len(spatial) >= budget:
        return np.sort(spatial[:budget])
    selected = np.zeros(len(means), dtype=bool)
    selected[spatial] = True
    remaining = confidence_order[~selected[confidence_order]]
    result = np.concatenate((spatial, remaining[: budget - len(spatial)]))
    return np.sort(result)


def write_graphdeco_dc_ply(
    path: Path,
    means: np.ndarray,
    covariances: np.ndarray,
    f_dc: np.ndarray,
    opacity_probability: np.ndarray,
) -> None:
    """Write a semantic-equivalent SH-0 Graphdeco Gaussian PLY."""

    means = np.asarray(means, dtype=np.float32)
    f_dc = np.asarray(f_dc, dtype=np.float32)
    opacity_probability = np.asarray(opacity_probability).reshape(-1)
    if means.shape != f_dc.shape or means.shape[1] != 3:
        raise ValueError(f"means/f_dc must both be [N,3], got {means.shape}/{f_dc.shape}")
    scales, rotations = covariance_to_scale_rotation(covariances)
    names = [
        "x", "y", "z", "nx", "ny", "nz",
        "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
        "scale_0", "scale_1", "scale_2",
        "rot_0", "rot_1", "rot_2", "rot_3",
    ]
    output = np.empty(len(means), dtype=[(name, "<f4") for name in names])
    attributes = np.concatenate(
        (
            means,
            np.zeros_like(means),
            f_dc,
            probability_to_logit(opacity_probability)[:, None],
            scales,
            rotations,
        ),
        axis=1,
    ).astype(np.float32)
    output[:] = list(map(tuple, attributes))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(output, "vertex")], text=False).write(path)
