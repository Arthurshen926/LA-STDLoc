"""Mapping-only compact view-support metadata for Projective Anchors."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F


SCHEMA = "lafgs_v24_anchor_view_support"
VERSION = 1


def _camera_centers(mapping_pose_w2c: torch.Tensor) -> torch.Tensor:
    pose = torch.as_tensor(mapping_pose_w2c).float()
    if pose.ndim != 3 or pose.shape[1:] != (4, 4):
        raise ValueError("V24 mapping poses must have shape [Q,4,4]")
    return -torch.einsum("qji,qj->qi", pose[:, :3, :3], pose[:, :3, 3])


def _direction_modes(
    directions: torch.Tensor, *, split_minimum_angle_deg: float
) -> tuple[torch.Tensor, torch.Tensor, int]:
    directions = F.normalize(torch.as_tensor(directions).float(), dim=1)
    mean = F.normalize(directions.mean(dim=0), dim=0)
    mean_angles = torch.rad2deg(
        torch.acos((directions @ mean).clamp(-1.0, 1.0))
    )
    if (
        directions.shape[0] < 4
        or float(mean_angles.max()) < float(split_minimum_angle_deg)
    ):
        modes = torch.stack((mean, mean))
        radii = torch.tensor([float(mean_angles.max()), -1.0])
        return modes, radii, 1

    first = directions[int(torch.argmin(directions @ mean))]
    second = directions[int(torch.argmin(directions @ first))]
    seed_scores = torch.stack((directions @ first, directions @ second), dim=1)
    labels = seed_scores.argmax(dim=1)
    if not bool((labels == 0).any()) or not bool((labels == 1).any()):
        modes = torch.stack((mean, mean))
        radii = torch.tensor([float(mean_angles.max()), -1.0])
        return modes, radii, 1
    centers = []
    radii = []
    for label in (0, 1):
        rows = directions[labels == label]
        center = F.normalize(rows.mean(dim=0), dim=0)
        radius = torch.rad2deg(
            torch.acos((rows @ center).clamp(-1.0, 1.0))
        ).max()
        centers.append(center)
        radii.append(radius)
    return torch.stack(centers), torch.stack(radii), 2


def build_anchor_view_support(
    *,
    anchor_xyz: torch.Tensor,
    observation_offsets: torch.Tensor,
    observation_query_indices: torch.Tensor,
    mapping_pose_w2c: torch.Tensor,
    split_minimum_angle_deg: float = 15.0,
) -> dict:
    """Compress mapping observation rays into at most two modes per Anchor."""

    xyz = torch.as_tensor(anchor_xyz).float().cpu()
    offsets = torch.as_tensor(observation_offsets).long().cpu().reshape(-1)
    query = torch.as_tensor(observation_query_indices).long().cpu().reshape(-1)
    pose = torch.as_tensor(mapping_pose_w2c).float().cpu()
    if (
        xyz.ndim != 2
        or xyz.shape[1] != 3
        or offsets.shape != (xyz.shape[0] + 1,)
        or int(offsets[0]) != 0
        or int(offsets[-1]) != query.numel()
        or bool((offsets[1:] < offsets[:-1]).any())
        or pose.ndim != 3
        or pose.shape[1:] != (4, 4)
        or (query.numel() and (int(query.min()) < 0 or int(query.max()) >= pose.shape[0]))
        or not bool(torch.isfinite(xyz).all())
        or not bool(torch.isfinite(pose).all())
        or float(split_minimum_angle_deg) <= 0.0
    ):
        raise ValueError("V24 Anchor view-support inputs are invalid")
    centers = _camera_centers(pose)
    modes = torch.empty((xyz.shape[0], 2, 3), dtype=torch.float32)
    radii = torch.empty((xyz.shape[0], 2), dtype=torch.float32)
    mode_count = torch.empty(xyz.shape[0], dtype=torch.uint8)
    minimum_distance = torch.empty(xyz.shape[0], dtype=torch.float32)
    maximum_distance = torch.empty(xyz.shape[0], dtype=torch.float32)
    observation_count = offsets[1:] - offsets[:-1]
    if bool((observation_count <= 0).any()):
        raise ValueError("every V24 Anchor needs at least one mapping observation")
    for anchor in range(xyz.shape[0]):
        rows = query[offsets[anchor] : offsets[anchor + 1]]
        rays = centers[rows] - xyz[anchor]
        distances = rays.norm(dim=1)
        if not bool(torch.isfinite(distances).all()) or bool((distances <= 1e-8).any()):
            raise ValueError("V24 mapping observation has an invalid viewing ray")
        anchor_modes, anchor_radii, count = _direction_modes(
            rays / distances[:, None],
            split_minimum_angle_deg=float(split_minimum_angle_deg),
        )
        modes[anchor] = anchor_modes
        radii[anchor] = anchor_radii
        mode_count[anchor] = count
        minimum_distance[anchor] = distances.min()
        maximum_distance[anchor] = distances.max()
    return {
        "schema": SCHEMA,
        "version": VERSION,
        "uses_source_mapping_rgb": False,
        "uses_test_queries": False,
        "map_mutated": False,
        "split_minimum_angle_deg": float(split_minimum_angle_deg),
        "direction_modes": modes,
        "direction_radius_deg": radii,
        "mode_count": mode_count,
        "minimum_distance_m": minimum_distance,
        "maximum_distance_m": maximum_distance,
        "observation_count": observation_count,
    }


def mapping_poses_from_views(views: Sequence) -> torch.Tensor:
    poses = [torch.as_tensor(view.pose_w2c).float().cpu() for view in views]
    if not poses:
        raise ValueError("V24 Anchor view support requires mapping views")
    return torch.stack(poses)
