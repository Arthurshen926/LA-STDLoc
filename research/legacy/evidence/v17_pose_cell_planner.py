"""Continuous-SE(3) confirmation planning after source-family exhaustion."""

from __future__ import annotations

import hashlib
import math

import torch

from common.v7_contracts import require_view_role, view_role_contract
from evidence.v7_query_planner import camera_centers, trajectory_statistics
from evidence.v9_novel_query_planner import (
    _axis_rotation,
    _pose_from_center_rotation,
    _rotation_angle,
    _visible_proxy,
)


def _prior_separation(
    pose: torch.Tensor,
    *,
    prior_poses: torch.Tensor,
    prior_centers: torch.Tensor,
    baseline: float,
    translation_scale: float,
    rotation_scale_deg: float,
) -> tuple[float, float, float]:
    center = camera_centers(pose[None])[0]
    distance = torch.linalg.norm(prior_centers - center, dim=1) / baseline
    rotation = pose[:3, :3]
    similarity = torch.einsum(
        "ij,nij->n", rotation, prior_poses[:, :3, :3]
    )
    cosine = ((similarity - 1.0) * 0.5).clamp(-1.0, 1.0)
    angle = torch.rad2deg(torch.acos(cosine))
    combined = torch.sqrt(
        (distance / float(translation_scale)).square()
        + (angle / float(rotation_scale_deg)).square()
    )
    closest = int(combined.argmin())
    return (
        float(combined[closest]),
        float(distance[closest]),
        float(angle[closest]),
    )


def plan_v17_pose_cell_confirmation(
    *,
    pose_w2c: torch.Tensor,
    intrinsics: torch.Tensor,
    image_hw: torch.Tensor,
    names: list[str],
    anchor_xyz: torch.Tensor,
    prior_pose_w2c: torch.Tensor,
    prior_source_family_ids: torch.Tensor,
    seed: int,
    maximum_queries: int = 96,
    templates_per_parent: int = 6,
    anchor_projection_stride: int = 32,
    minimum_novel_baselines: float = 0.65,
    maximum_novel_baselines: float = 2.35,
    minimum_view_angle_deg: float = 8.0,
    prior_translation_scale_baselines: float = 0.30,
    prior_rotation_scale_deg: float = 5.0,
    minimum_combined_prior_separation: float = 1.0,
) -> dict:
    """Plan pose-cell-fresh queries while keeping parent-block statistics honest.

    Historical plans eventually consume every mapping-camera parent even though
    each parent represents a continuum of unseen SE(3) poses.  V17 permits a
    previously used parent only when the new pose lies outside every registered
    translation/rotation collision ellipsoid.  One query per parent is kept in
    the new batch, and the parent remains the bootstrap block ID.
    """

    poses = torch.as_tensor(pose_w2c, dtype=torch.float64)
    intrinsic = torch.as_tensor(intrinsics, dtype=torch.float64)
    hw = torch.as_tensor(image_hw, dtype=torch.long)
    anchors = torch.as_tensor(anchor_xyz, dtype=torch.float64).reshape(-1, 3)
    prior = torch.as_tensor(prior_pose_w2c, dtype=torch.float64)
    prior_families = torch.as_tensor(prior_source_family_ids).long().reshape(-1)
    if (
        poses.ndim != 3
        or poses.shape[1:] != (4, 4)
        or intrinsic.shape != (poses.shape[0], 3, 3)
        or hw.shape != (poses.shape[0], 2)
        or len(names) != poses.shape[0]
        or anchors.shape[0] == 0
        or prior.ndim != 3
        or prior.shape[1:] != (4, 4)
        or prior.shape[0] != prior_families.numel()
    ):
        raise ValueError("V17 mapping/prior pose registries differ")
    if prior.shape[0] == 0:
        raise ValueError("V17 pose-cell planning requires a prior pose registry")
    sequence_ids = [str(name).split("/", 1)[0] for name in names]
    statistics = trajectory_statistics(poses, sequence_ids)
    baseline = max(float(statistics["median_adjacent_baseline_m"]), 1e-4)
    centers = camera_centers(poses)
    prior_centers = camera_centers(prior)
    generator = torch.Generator().manual_seed(int(seed))
    parent_order = torch.randperm(poses.shape[0], generator=generator).tolist()
    sampled_anchors = anchors[:: max(int(anchor_projection_stride), 1)]
    scales = (0.95, 1.20, 1.50, 1.85, 2.25, 2.70, 3.10, 3.55)
    yaws = (13.0, 18.0, 24.0, 31.0, 38.0, 45.0, 52.0, 58.0)
    pitches = (6.0, -8.0, 10.0, -12.0, 14.0, -16.0, 18.0, -20.0)
    geometric_candidates = []
    for parent_order_index, parent in enumerate(parent_order):
        c2w = poses[parent, :3, :3].T
        for template_offset in range(int(templates_per_parent)):
            template = (parent_order_index + template_offset + int(seed)) % len(scales)
            sign = -1.0 if (parent + template_offset + int(seed)) % 2 else 1.0
            scale = scales[template]
            vertical = ((parent + 2 * template_offset) % 5 - 2) * 0.16
            forward_sign = -1.0 if template_offset % 3 == 0 else 1.0
            center = (
                centers[parent]
                + sign * 0.72 * scale * baseline * c2w[:, 0]
                + vertical * baseline * c2w[:, 1]
                + forward_sign * 0.82 * scale * baseline * c2w[:, 2]
            )
            yaw = math.radians(sign * yaws[template])
            pitch = math.radians(pitches[template])
            rotation = (
                _axis_rotation(0, pitch)
                @ _axis_rotation(1, yaw)
                @ poses[parent, :3, :3]
            )
            pose = _pose_from_center_rotation(center, rotation)
            distances = torch.linalg.norm(centers - center, dim=1)
            nearest = int(distances.argmin())
            novelty = float(distances[nearest] / baseline)
            angle = _rotation_angle(pose[:3, :3], poses[nearest, :3, :3])
            if (
                novelty < minimum_novel_baselines
                or novelty > maximum_novel_baselines
                or angle < minimum_view_angle_deg
            ):
                continue
            separation, prior_distance, prior_angle = _prior_separation(
                pose,
                prior_poses=prior,
                prior_centers=prior_centers,
                baseline=baseline,
                translation_scale=prior_translation_scale_baselines,
                rotation_scale_deg=prior_rotation_scale_deg,
            )
            if separation < minimum_combined_prior_separation:
                continue
            preliminary_utility = (
                max(0.0, 1.0 - abs(novelty - 1.55) / 1.55)
                + min(angle / 35.0, 1.0)
                + 0.25 * min(separation / 3.0, 1.0)
            )
            geometric_candidates.append(
                {
                    "pose": pose,
                    "parent": parent,
                    "novelty": novelty,
                    "angle": angle,
                    "prior_separation": separation,
                    "prior_distance": prior_distance,
                    "prior_angle": prior_angle,
                    "preliminary_utility": preliminary_utility,
                    "template": template,
                }
            )
    # Visibility projection dominates CPU planning cost.  Rank with the cheap
    # SE(3) terms first, then evaluate only enough parent-unique candidates to
    # construct a healthy reserve instead of raster-proxy testing every
    # template for every mapping camera.
    geometric_candidates.sort(
        key=lambda row: (
            -row["preliminary_utility"],
            row["parent"],
            row["template"],
        )
    )
    visible_candidates = []
    accepted_parents = set()
    for row in geometric_candidates:
        if row["parent"] in accepted_parents:
            continue
        visible, cells = _visible_proxy(
            sampled_anchors,
            row["pose"],
            intrinsic[row["parent"]],
            hw[row["parent"]],
        )
        if visible < 32 or cells < 6:
            continue
        row = {
            **row,
            "visible": visible,
            "cells": cells,
            "utility": row["preliminary_utility"] + min(cells / 16.0, 1.0),
        }
        visible_candidates.append(row)
        accepted_parents.add(row["parent"])
        if len(visible_candidates) == max(int(maximum_queries) * 3, 128):
            break
    visible_candidates.sort(
        key=lambda row: (-row["utility"], row["parent"], row["template"])
    )
    selected = visible_candidates[: int(maximum_queries)]
    if len(selected) < int(maximum_queries):
        raise RuntimeError(
            f"V17 pose-cell gates produced {len(selected)}/{maximum_queries} queries"
        )

    source_rows = torch.tensor([row["parent"] for row in selected], dtype=torch.long)
    pose_tensor = torch.stack([row["pose"] for row in selected])
    pose_cell_ids = [
        hashlib.sha256(pose.contiguous().numpy().tobytes()).hexdigest()
        for pose in pose_tensor
    ]
    output = {
        "schema": "lafgs_v17_pose_cell_confirmation_plan",
        "version": 1,
        **view_role_contract("confirmation_query"),
        "uses_source_mapping_rgb": False,
        "uses_test_queries": False,
        "enters_track_registry": False,
        "enters_anchor_observation_csr": False,
        "enters_descriptor_bank": False,
        "render_protocol": "clean_once_per_pose",
        "loo_used": False,
        "trajectory_interpolation_candidate_count": 0,
        "seed": int(seed),
        "mapping_camera_count": int(poses.shape[0]),
        "query_count": len(selected),
        "trajectory_statistics": statistics,
        "pose_w2c": pose_tensor,
        "intrinsics": intrinsic[source_rows].clone(),
        "image_hw": hw[source_rows].clone(),
        "source_mapping_indices": [[int(row)] for row in source_rows.tolist()],
        # Parent IDs intentionally remain the bootstrap blocks.  Reusing a
        # parent is disclosed instead of pretending fine pose cells are fully
        # independent source families.
        "pose_family_ids": source_rows.clone(),
        "pose_cell_ids": pose_cell_ids,
        "query_kinds": ["pose_cell_novel_se3"] * len(selected),
        "novelty_baselines": torch.tensor([row["novelty"] for row in selected]),
        "nearest_view_angle_deg": torch.tensor([row["angle"] for row in selected]),
        "visible_anchor_proxy_count": torch.tensor(
            [row["visible"] for row in selected]
        ),
        "visible_cell_count": torch.tensor([row["cells"] for row in selected]),
        "nearest_prior_combined_separation": torch.tensor(
            [row["prior_separation"] for row in selected]
        ),
        "nearest_prior_translation_baselines": torch.tensor(
            [row["prior_distance"] for row in selected]
        ),
        "nearest_prior_rotation_deg": torch.tensor(
            [row["prior_angle"] for row in selected]
        ),
        "planner_contract": {
            "trajectory_interpolation": False,
            "geometry_mutation": False,
            "test_pose_distribution": False,
            "freshness_unit": "continuous_se3_pose_cell",
            "statistical_block_unit": "source_mapping_parent",
            "source_parent_reuse_disclosed": True,
            "prior_pose_count": int(prior.shape[0]),
            "prior_source_parent_count": int(torch.unique(prior_families).numel()),
            "source_parent_overlap_count": len(
                set(source_rows.tolist()) & set(prior_families.tolist())
            ),
            "prior_translation_scale_baselines": float(
                prior_translation_scale_baselines
            ),
            "prior_rotation_scale_deg": float(prior_rotation_scale_deg),
            "minimum_combined_prior_separation": float(
                minimum_combined_prior_separation
            ),
            "minimum_novel_baselines": float(minimum_novel_baselines),
            "maximum_novel_baselines": float(maximum_novel_baselines),
            "selection": "pose_cell_freshness_overlap_utility_parent_unique",
        },
    }
    digest = hashlib.sha256()
    digest.update(output["pose_w2c"].contiguous().numpy().tobytes())
    digest.update(str(seed).encode())
    digest.update(b"confirmation_query")
    output["plan_sha256"] = digest.hexdigest()
    require_view_role(output, "confirmation_query")
    return output
