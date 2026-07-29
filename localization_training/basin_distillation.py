"""Utilities for query-specific P3P basin supervision."""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
import poselib
import torch


GOOD_SET = 0
HARMFUL_SET = 1
NEAR_MISS_SET = 2


def expanded_positive_lookup(record: dict) -> dict[int, list[int]]:
    """Expand a CSR positive teacher record into query-row identities."""
    rows = torch.as_tensor(record["query_rows"]).long()
    offsets = torch.as_tensor(record["positive_offsets"]).long()
    indices = torch.as_tensor(record["positive_indices"]).long()
    if offsets.numel() != rows.numel() + 1:
        raise ValueError("positive offsets do not align with query rows")
    output: dict[int, list[int]] = {}
    for index, row in enumerate(rows.tolist()):
        values = indices[offsets[index] : offsets[index + 1]].tolist()
        if values:
            output[int(row)] = [int(value) for value in values]
    return output


def image_cell_ids(keypoints: np.ndarray, width: int, height: int) -> np.ndarray:
    x = np.floor(keypoints[:, 0] / max(int(width), 1) * 4).astype(np.int64)
    y = np.floor(keypoints[:, 1] / max(int(height), 1) * 4).astype(np.int64)
    return np.clip(y, 0, 3) * 4 + np.clip(x, 0, 3)


def is_diverse_triplet(
    indices: np.ndarray,
    points3d: np.ndarray,
    dependency_groups: np.ndarray,
    image_cells: np.ndarray,
    surface_groups: np.ndarray,
) -> bool:
    indices = np.asarray(indices, dtype=np.int64).reshape(-1)
    if indices.size != 3 or np.unique(indices).size != 3:
        return False
    triangle = points3d[indices]
    center = np.median(points3d, axis=0)
    scale = max(
        float(np.median(np.linalg.norm(points3d - center, axis=1))), 1e-6
    )
    edges = triangle[[1, 2, 0]] - triangle[[0, 1, 2]]
    extent = float(np.linalg.norm(edges, axis=1).max())
    area = float(
        0.5
        * np.linalg.norm(
            np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
        )
    )
    return bool(
        np.unique(dependency_groups[indices]).size == 3
        and np.unique(image_cells[indices]).size >= 3
        and np.unique(surface_groups[indices]).size >= 2
        and extent >= 0.02 * scale
        and area >= 1e-4 * scale * scale
    )


def proposal_propensity(pool_size: int, category_probability: float) -> float:
    if int(pool_size) < 3:
        return 0.0
    return float(category_probability) / float(math.comb(int(pool_size), 3))


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


def evaluate_p3p_triplet(
    triplet: np.ndarray,
    points2d: np.ndarray,
    points3d: np.ndarray,
    K: np.ndarray,
    ground_truth_w2c: np.ndarray,
    *,
    basis_points3d: np.ndarray | None = None,
    reprojection_error: float = 12.0,
    correct_translation_cm: float = 50.0,
    correct_rotation_deg: float = 5.0,
) -> dict:
    """Score one P3P basis against the full query correspondence set."""
    triplet = np.asarray(triplet, dtype=np.int64).reshape(3)
    homogeneous = np.concatenate(
        (points2d, np.ones((points2d.shape[0], 1), dtype=np.float64)), axis=1
    )
    bearings = homogeneous @ np.linalg.inv(K).T
    bearings /= np.linalg.norm(bearings, axis=1, keepdims=True).clip(min=1e-12)
    try:
        hypotheses = poselib.p3p(
            bearings[triplet],
            (
                np.asarray(basis_points3d, dtype=np.float64).reshape(3, 3)
                if basis_points3d is not None
                else points3d[triplet]
            ),
        )
    except RuntimeError:
        hypotheses = []
    best = None
    threshold2 = float(reprojection_error) ** 2
    for pose in hypotheses:
        rt = np.asarray(pose.Rt, dtype=np.float64)
        camera = points3d @ rt[:, :3].T + rt[:, 3]
        valid = camera[:, 2] > 1e-8
        projected = np.empty_like(points2d)
        projected[:, 0] = (
            K[0, 0] * camera[:, 0] / np.maximum(camera[:, 2], 1e-8) + K[0, 2]
        )
        projected[:, 1] = (
            K[1, 1] * camera[:, 1] / np.maximum(camera[:, 2], 1e-8) + K[1, 2]
        )
        error2 = np.square(projected - points2d).sum(axis=1)
        error2[~valid] = np.inf
        inlier_count = int((error2 <= threshold2).sum())
        msac_cost = float(np.minimum(error2, threshold2).sum())
        te_cm, re_deg = _pose_error(pose, ground_truth_w2c)
        candidate = (inlier_count, -msac_cost, te_cm, re_deg)
        if best is None or candidate[:2] > best[:2]:
            best = candidate
    if best is None:
        return {
            "valid": False,
            "inlier_count": 0,
            "msac_cost": float(points2d.shape[0] * threshold2),
            "te_cm": float("inf"),
            "re_deg": float("inf"),
            "correct_basin": False,
        }
    inlier_count, negative_cost, te_cm, re_deg = best
    return {
        "valid": True,
        "inlier_count": int(inlier_count),
        "msac_cost": float(-negative_cost),
        "te_cm": float(te_cm),
        "re_deg": float(re_deg),
        "correct_basin": bool(
            te_cm <= float(correct_translation_cm)
            and re_deg <= float(correct_rotation_deg)
        ),
    }


def aggregate_edge_credit(
    set_rows: torch.Tensor,
    set_anchors: torch.Tensor,
    set_types: torch.Tensor,
    correct_basin: torch.Tensor,
    propensities: torch.Tensor,
    severity: torch.Tensor,
    *,
    maximum_inverse_propensity: float = 100.0,
) -> dict:
    """Aggregate query-local edge credit without creating global labels."""
    rows = torch.as_tensor(set_rows).long()
    anchors = torch.as_tensor(set_anchors).long()
    types = torch.as_tensor(set_types).long()
    correct = torch.as_tensor(correct_basin).bool()
    propensity = torch.as_tensor(propensities).float().clamp_min(1e-12)
    severity = torch.as_tensor(severity).float().clamp_min(0)
    inverse = (
        propensity.median().clamp_min(1e-12) / propensity
    ).clamp_max(float(maximum_inverse_propensity))
    if inverse.numel():
        inverse /= inverse.mean().clamp_min(1e-8)
    positive: dict[tuple[int, int], float] = defaultdict(float)
    negative: dict[tuple[int, int], float] = defaultdict(float)
    for set_index in range(rows.shape[0]):
        for position in range(3):
            key = (int(rows[set_index, position]), int(anchors[set_index, position]))
            if bool(correct[set_index]) and int(types[set_index]) != HARMFUL_SET:
                positive[key] += float(inverse[set_index])
            elif int(types[set_index]) == HARMFUL_SET:
                negative[key] += float(inverse[set_index] * severity[set_index])

    def encode(values: dict[tuple[int, int], float]):
        ordered = sorted(values.items())
        return {
            "rows": torch.as_tensor([key[0] for key, _ in ordered], dtype=torch.long),
            "anchors": torch.as_tensor(
                [key[1] for key, _ in ordered], dtype=torch.long
            ),
            "weights": torch.as_tensor(
                [value for _, value in ordered], dtype=torch.float32
            ),
        }

    return {"positive": encode(positive), "negative": encode(negative)}
