"""Correlation-aware support scores for absolute-pose hypotheses.

The functions in this module are solver-independent.  They operate on a fixed
correspondence graph and make repeated evidence from one map-side group
saturate instead of counting every match as an independent observation.
"""

from __future__ import annotations

import itertools
import math

import numpy as np


def project_points(points3d, K, pose_w2c):
    points3d = np.asarray(points3d, dtype=np.float64).reshape(-1, 3)
    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    pose_w2c = np.asarray(pose_w2c, dtype=np.float64).reshape(4, 4)
    camera = points3d @ pose_w2c[:3, :3].T + pose_w2c[:3, 3]
    valid = np.isfinite(camera).all(axis=1) & (camera[:, 2] > 1e-8)
    projected = np.full((points3d.shape[0], 2), np.nan, dtype=np.float64)
    projected[valid, 0] = (
        K[0, 0] * camera[valid, 0] / camera[valid, 2] + K[0, 2]
    )
    projected[valid, 1] = (
        K[1, 1] * camera[valid, 1] / camera[valid, 2] + K[1, 2]
    )
    return projected, valid


def reprojection_errors(points2d, points3d, K, pose_w2c):
    points2d = np.asarray(points2d, dtype=np.float64).reshape(-1, 2)
    projected, valid = project_points(points3d, K, pose_w2c)
    errors = np.linalg.norm(projected - points2d, axis=1)
    errors[~valid] = np.inf
    return errors


def soft_msac_support(errors, threshold):
    """Return the support equivalent of a truncated quadratic MSAC cost."""
    threshold = float(threshold)
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("threshold must be finite and positive")
    errors = np.asarray(errors, dtype=np.float64).reshape(-1)
    normalized_sq = np.square(errors / threshold)
    return np.clip(1.0 - normalized_sq, 0.0, 1.0)


def canonicalize_normal_sign(normals):
    """Remove the arbitrary sign of Gaussian surface normals."""
    normals = np.asarray(normals, dtype=np.float64).reshape(-1, 3)
    norm = np.linalg.norm(normals, axis=1, keepdims=True)
    normalized = normals / np.maximum(norm, 1e-12)
    dominant = np.argmax(np.abs(normalized), axis=1)
    sign = np.sign(normalized[np.arange(normalized.shape[0]), dominant])
    sign[sign == 0] = 1.0
    return normalized * sign[:, None]


def robust_scene_scale(xyz):
    xyz = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    if xyz.shape[0] == 0:
        return 1.0
    center = np.median(xyz, axis=0)
    scale = float(np.median(np.linalg.norm(xyz - center, axis=1)))
    return max(scale, 1e-6)


class _DisjointSet:
    def __init__(self, size):
        self.parent = np.arange(int(size), dtype=np.int64)
        self.rank = np.zeros(int(size), dtype=np.int8)

    def find(self, value):
        value = int(value)
        parent = self.parent[value]
        while parent != value:
            grandparent = self.parent[parent]
            self.parent[value] = grandparent
            value = int(parent)
            parent = grandparent
        return value

    def union(self, left, right):
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


def build_surface_component_groups(
    xyz,
    normals,
    *,
    voxel_size=None,
    voxel_scale_ratio=0.02,
    minimum_voxel_size=0.5,
    maximum_normal_angle_degrees=25.0,
):
    """Build connected, approximately coplanar map-side surface components.

    Unlike a raw source-Gaussian ID, a component can represent a large
    correlated facade.  Unlike global plane quantization, connectivity keeps
    parallel but spatially separated surfaces distinct.
    """
    xyz = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    normals = canonicalize_normal_sign(normals)
    if xyz.shape[0] != normals.shape[0]:
        raise ValueError("xyz and normals must align")
    if xyz.shape[0] == 0:
        return np.empty(0, dtype=np.int64), {
            "component_count": 0,
            "voxel_count": 0,
            "voxel_size": float(minimum_voxel_size),
            "scene_scale": 1.0,
        }
    scene_scale = robust_scene_scale(xyz)
    if voxel_size is None:
        voxel_size = max(
            float(minimum_voxel_size),
            float(voxel_scale_ratio) * scene_scale,
        )
    voxel_size = max(float(voxel_size), 1e-6)
    origin = np.min(xyz, axis=0)
    voxel_key = np.floor((xyz - origin) / voxel_size).astype(np.int64)
    unique_key, inverse = np.unique(voxel_key, axis=0, return_inverse=True)
    voxel_count = unique_key.shape[0]

    voxel_normal_sum = np.zeros((voxel_count, 3), dtype=np.float64)
    np.add.at(voxel_normal_sum, inverse, normals)
    voxel_normals = canonicalize_normal_sign(voxel_normal_sum)
    voxel_centers = origin + (unique_key.astype(np.float64) + 0.5) * voxel_size
    key_to_id = {tuple(key): index for index, key in enumerate(unique_key)}
    disjoint = _DisjointSet(voxel_count)
    cosine_limit = math.cos(math.radians(float(maximum_normal_angle_degrees)))
    forward_offsets = [
        offset
        for offset in itertools.product((-1, 0, 1), repeat=3)
        if offset != (0, 0, 0)
        and next(value for value in offset if value != 0) > 0
    ]
    tangent_tolerance = 1.5 * voxel_size
    for left, key in enumerate(unique_key):
        left_normal = voxel_normals[left]
        for offset in forward_offsets:
            right = key_to_id.get(tuple(key + np.asarray(offset)))
            if right is None:
                continue
            right_normal = voxel_normals[right]
            if float(np.dot(left_normal, right_normal)) < cosine_limit:
                continue
            displacement = voxel_centers[right] - voxel_centers[left]
            if max(
                abs(float(np.dot(displacement, left_normal))),
                abs(float(np.dot(displacement, right_normal))),
            ) > tangent_tolerance:
                continue
            disjoint.union(left, right)

    roots = np.asarray([disjoint.find(index) for index in range(voxel_count)])
    _, voxel_component = np.unique(roots, return_inverse=True)
    groups = voxel_component[inverse].astype(np.int64)
    component_sizes = np.bincount(groups)
    return groups, {
        "component_count": int(component_sizes.size),
        "voxel_count": int(voxel_count),
        "voxel_size": float(voxel_size),
        "scene_scale": float(scene_scale),
        "maximum_component_size": int(component_sizes.max(initial=0)),
        "median_component_size": float(
            np.median(component_sizes) if component_sizes.size else 0.0
        ),
    }


def image_cell_ids(points2d, width, height, rows=4, cols=4):
    points2d = np.asarray(points2d, dtype=np.float64).reshape(-1, 2)
    width = max(float(width), 1.0)
    height = max(float(height), 1.0)
    col = np.floor(points2d[:, 0] * int(cols) / width).astype(np.int64)
    row = np.floor(points2d[:, 1] * int(rows) / height).astype(np.int64)
    col = np.clip(col, 0, int(cols) - 1)
    row = np.clip(row, 0, int(rows) - 1)
    return row * int(cols) + col


def image_cell_normalization(cell_ids):
    """Return bounded inverse-sqrt density weights with mean one."""
    cell_ids = np.asarray(cell_ids, dtype=np.int64).reshape(-1)
    if cell_ids.size == 0:
        return np.empty(0, dtype=np.float64)
    _, inverse, counts = np.unique(
        cell_ids, return_inverse=True, return_counts=True
    )
    weights = 1.0 / np.sqrt(counts[inverse].astype(np.float64))
    weights /= max(float(np.mean(weights)), 1e-12)
    return np.clip(weights, 0.25, 4.0)


def saturated_group_support(utilities, group_ids, cap, weights=None):
    utilities = np.asarray(utilities, dtype=np.float64).reshape(-1)
    group_ids = np.asarray(group_ids, dtype=np.int64).reshape(-1)
    if utilities.shape != group_ids.shape:
        raise ValueError("utilities and group_ids must align")
    cap = float(cap)
    if not np.isfinite(cap) or cap <= 0:
        raise ValueError("cap must be finite and positive")
    if weights is not None:
        weights = np.asarray(weights, dtype=np.float64).reshape(-1)
        if weights.shape != utilities.shape:
            raise ValueError("weights and utilities must align")
        utilities = utilities * weights
    if utilities.size == 0:
        return 0.0
    valid_group = group_ids >= 0
    if not bool(np.all(valid_group)):
        group_ids = group_ids.copy()
        next_group = int(group_ids[valid_group].max(initial=-1)) + 1
        group_ids[~valid_group] = np.arange(
            next_group, next_group + int((~valid_group).sum())
        )
    _, inverse = np.unique(group_ids, return_inverse=True)
    group_mass = np.bincount(inverse, weights=utilities)
    return float(np.minimum(group_mass, cap).sum())


def score_pose(
    points2d,
    points3d,
    K,
    pose_w2c,
    threshold,
    *,
    group_ids=None,
    cap=None,
    weights=None,
):
    errors = reprojection_errors(points2d, points3d, K, pose_w2c)
    support = soft_msac_support(errors, threshold)
    if group_ids is None:
        score = float(np.sum(support))
    else:
        score = saturated_group_support(
            support,
            group_ids,
            cap,
            weights=weights,
        )
    return score, errors, support


def support_concentration(utilities, group_ids):
    utilities = np.asarray(utilities, dtype=np.float64).reshape(-1)
    group_ids = np.asarray(group_ids, dtype=np.int64).reshape(-1)
    if utilities.size == 0 or float(utilities.sum()) <= 0:
        return {
            "unique_supported_groups": 0,
            "maximum_group_fraction": 0.0,
            "group_effective_sample_size": 0.0,
        }
    _, inverse = np.unique(group_ids, return_inverse=True)
    mass = np.bincount(inverse, weights=utilities)
    mass = mass[mass > 0]
    probability = mass / mass.sum()
    return {
        "unique_supported_groups": int(mass.size),
        "maximum_group_fraction": float(probability.max(initial=0.0)),
        "group_effective_sample_size": float(
            1.0 / max(float(np.square(probability).sum()), 1e-12)
        ),
    }
