"""Deterministic, mapping-only novel-view planner for V7 P1."""

from __future__ import annotations

from collections.abc import Sequence
import hashlib
import json
import math
from typing import Any

import torch

from common.v7_contracts import require_view_role, view_role_contract


SCHEMA = "lafgs_v7_novel_view_plan"
VERSION = 1


def camera_centers(pose_w2c: torch.Tensor) -> torch.Tensor:
    pose = torch.as_tensor(pose_w2c, dtype=torch.float64)
    return -(pose[:, :3, :3].transpose(1, 2) @ pose[:, :3, 3, None]).squeeze(2)


def _pose(center: torch.Tensor, rotation_w2c: torch.Tensor) -> torch.Tensor:
    result = torch.eye(4, dtype=torch.float64)
    result[:3, :3] = rotation_w2c
    result[:3, 3] = -(rotation_w2c @ center)
    return result


def _orthogonalize(rotation: torch.Tensor) -> torch.Tensor:
    u, _, vh = torch.linalg.svd(rotation)
    result = u @ vh
    if float(torch.linalg.det(result)) < 0:
        u[:, -1] *= -1
        result = u @ vh
    return result


def _rotation_angle_deg(left: torch.Tensor, right: torch.Tensor) -> float:
    relative = left @ right.T
    cosine = ((torch.trace(relative) - 1.0) / 2.0).clamp(-1.0, 1.0)
    return math.degrees(math.acos(float(cosine)))


def _axis_rotation(axis: int, radians: float) -> torch.Tensor:
    c, s = math.cos(radians), math.sin(radians)
    if axis == 0:
        return torch.tensor([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=torch.float64)
    return torch.tensor([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=torch.float64)


def _stable_digest(value: Any) -> str:
    def normalize(item: Any) -> Any:
        if isinstance(item, torch.Tensor):
            return item.tolist()
        if isinstance(item, dict):
            return {key: normalize(child) for key, child in item.items()}
        if isinstance(item, (list, tuple)):
            return [normalize(child) for child in item]
        return item
    encoded = json.dumps(normalize(value), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def trajectory_statistics(pose_w2c: torch.Tensor, sequence_ids: Sequence[str]) -> dict[str, float]:
    poses = torch.as_tensor(pose_w2c, dtype=torch.float64)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4) or poses.shape[0] < 2:
        raise ValueError("V7 planner requires at least two mapping poses")
    if len(sequence_ids) != poses.shape[0]:
        raise ValueError("sequence IDs must align with mapping poses")
    centers = camera_centers(poses)
    adjacent = [
        index for index in range(poses.shape[0] - 1)
        if sequence_ids[index] == sequence_ids[index + 1]
    ]
    if not adjacent:
        raise ValueError("V7 planner found no within-sequence adjacent poses")
    baselines = torch.tensor([
        float(torch.linalg.norm(centers[index + 1] - centers[index])) for index in adjacent
    ], dtype=torch.float64)
    rotations = torch.tensor([
        _rotation_angle_deg(poses[index, :3, :3], poses[index + 1, :3, :3])
        for index in adjacent
    ], dtype=torch.float64)
    origin = centers.median(dim=0).values
    radii = torch.linalg.norm(centers - origin, dim=1)
    return {
        "adjacent_pair_count": len(adjacent),
        "median_adjacent_baseline_m": float(baselines.median()),
        "median_adjacent_rotation_deg": float(rotations.median()),
        "scene_radius_m": float(radii.median().clamp_min(1e-6)),
        "envelope_min_x": float(centers[:, 0].min()),
        "envelope_max_x": float(centers[:, 0].max()),
        "envelope_min_y": float(centers[:, 1].min()),
        "envelope_max_y": float(centers[:, 1].max()),
        "envelope_min_z": float(centers[:, 2].min()),
        "envelope_max_z": float(centers[:, 2].max()),
    }


def plan_v7_novel_queries(
    *,
    pose_w2c: torch.Tensor,
    intrinsics: torch.Tensor,
    image_hw: torch.Tensor,
    names: Sequence[str],
    role: str,
    seed: int,
    maximum_queries: int = 64,
    near_fraction: float = 0.75,
) -> dict[str, Any]:
    if role not in {"feedback_query", "confirmation_query"}:
        raise ValueError("P1 planner creates feedback or confirmation queries only")
    poses = torch.as_tensor(pose_w2c, dtype=torch.float64)
    intrinsics = torch.as_tensor(intrinsics, dtype=torch.float64)
    image_hw = torch.as_tensor(image_hw, dtype=torch.long)
    if len(names) != poses.shape[0] or intrinsics.shape != (poses.shape[0], 3, 3):
        raise ValueError("mapping camera registry does not align")
    if image_hw.shape != (poses.shape[0], 2) or maximum_queries <= 0:
        raise ValueError("invalid image registry or query budget")
    sequence_ids = [str(name).split("/", 1)[0] for name in names]
    stats = trajectory_statistics(poses, sequence_ids)
    centers = camera_centers(poses)
    pairs = [
        index for index in range(poses.shape[0] - 1)
        if sequence_ids[index] == sequence_ids[index + 1]
    ]
    generator = torch.Generator().manual_seed(int(seed))
    order = torch.randperm(len(pairs), generator=generator).tolist()
    pairs = [pairs[index] for index in order]
    near_budget = min(len(pairs), round(maximum_queries * float(near_fraction)))
    selected: list[torch.Tensor] = []
    sources: list[list[int]] = []
    kinds: list[str] = []
    source_rows: list[int] = []
    baseline = max(stats["median_adjacent_baseline_m"], 1e-4)
    rotation_deg = max(stats["median_adjacent_rotation_deg"], 0.25)
    role_phase = 1.0 if role == "feedback_query" else -1.0
    for rank, left in enumerate(pairs[:near_budget]):
        right = left + 1
        fraction = 0.5 + role_phase * (0.08 if rank % 2 == 0 else -0.08)
        center = torch.lerp(centers[left], centers[right], fraction)
        rotation = _orthogonalize(torch.lerp(poses[left, :3, :3], poses[right, :3, :3], fraction))
        c2w = rotation.T
        center = center + role_phase * (0.12 * baseline) * c2w[:, rank % 2]
        delta = math.radians(min(5.0, 0.5 * rotation_deg)) * (1 if rank % 2 == 0 else -1)
        rotation = _axis_rotation(1, role_phase * delta) @ rotation
        selected.append(_pose(center, rotation))
        sources.append([left, right])
        kinds.append("near_interpolation_perturbation")
        source_rows.append(left)
    remaining = maximum_queries - len(selected)
    for rank, left in enumerate(pairs[near_budget:near_budget + remaining]):
        direction = centers[left + 1] - centers[left]
        direction = direction / torch.linalg.norm(direction).clamp_min(1e-8)
        sign = role_phase * (1.0 if rank % 2 == 0 else -1.0)
        center = centers[left] + sign * 0.45 * baseline * direction
        lateral = poses[left, :3, :3].T[:, 0]
        center = center + sign * 0.25 * baseline * lateral
        delta = math.radians(min(10.0, rotation_deg)) * sign
        rotation = _axis_rotation(rank % 2, delta) @ poses[left, :3, :3]
        selected.append(_pose(center, rotation))
        sources.append([left, left + 1])
        kinds.append("bounded_trajectory_extension")
        source_rows.append(left)
    if not selected:
        raise ValueError("V7 planner generated no novel queries")
    planned = torch.stack(selected)
    distances = torch.cdist(camera_centers(planned), centers)
    if bool((distances.min(dim=1).values <= 1e-9).any()):
        raise ValueError("planned V7 query duplicates a mapping camera center")
    source_rows_tensor = torch.tensor(source_rows, dtype=torch.long)
    contract = view_role_contract(role)
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "version": VERSION,
        **contract,
        "uses_source_mapping_rgb": False,
        "uses_test_queries": False,
        "enters_track_registry": False,
        "enters_anchor_observation_csr": False,
        "enters_descriptor_bank": False,
        "render_protocol": "clean_once_per_pose",
        "seed": int(seed),
        "mapping_camera_count": int(poses.shape[0]),
        "query_count": int(planned.shape[0]),
        "trajectory_statistics": stats,
        "pose_w2c": planned,
        "intrinsics": intrinsics[source_rows_tensor].clone(),
        "image_hw": image_hw[source_rows_tensor].clone(),
        "source_mapping_indices": sources,
        "pose_family_ids": torch.arange(planned.shape[0], dtype=torch.long),
        "query_kinds": kinds,
    }
    payload["plan_sha256"] = _stable_digest(payload)
    require_view_role(payload, role)
    return payload


def plan_v7_test_pose_render_diagnostic(
    *,
    mapping_pose_w2c: torch.Tensor,
    mapping_names: Sequence[str],
    test_pose_w2c: torch.Tensor,
    test_intrinsics: torch.Tensor,
    test_image_hw: torch.Tensor,
    test_names: Sequence[str],
    query_indices: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Materialize an explicitly non-formal test-pose/render diagnostic plan.

    Only camera metadata enters this artifact. Test RGB paths and pixels are not
    accepted by the interface, and the resulting queries can never control a map.
    """

    mapping_poses = torch.as_tensor(mapping_pose_w2c, dtype=torch.float64)
    test_poses = torch.as_tensor(test_pose_w2c, dtype=torch.float64)
    intrinsics = torch.as_tensor(test_intrinsics, dtype=torch.float64)
    image_hw = torch.as_tensor(test_image_hw, dtype=torch.long)
    if mapping_poses.ndim != 3 or mapping_poses.shape[1:] != (4, 4):
        raise ValueError("diagnostic mapping poses must have shape [N,4,4]")
    if test_poses.ndim != 3 or test_poses.shape[1:] != (4, 4):
        raise ValueError("diagnostic test poses must have shape [Q,4,4]")
    count = int(test_poses.shape[0])
    if (
        len(mapping_names) != mapping_poses.shape[0]
        or len(test_names) != count
        or intrinsics.shape != (count, 3, 3)
        or image_hw.shape != (count, 2)
    ):
        raise ValueError("diagnostic camera metadata does not align")
    if query_indices is None:
        query_indices = torch.arange(count, dtype=torch.long)
    query_indices = torch.as_tensor(query_indices).long()
    if query_indices.shape != (count,) or torch.unique(query_indices).numel() != count:
        raise ValueError("diagnostic query indices must be unique")
    mapping_sequences = [str(name).split("/", 1)[0] for name in mapping_names]
    statistics = trajectory_statistics(mapping_poses, mapping_sequences)
    distances = torch.cdist(camera_centers(test_poses), camera_centers(mapping_poses))
    nearest = distances.topk(k=min(2, mapping_poses.shape[0]), largest=False).indices
    contract = view_role_contract("test_pose_render_diagnostic")
    payload: dict[str, Any] = {
        "schema": "lafgs_v7_test_pose_render_diagnostic_plan",
        "version": 1,
        **contract,
        "formal_protocol_eligible": False,
        "transductive_pose_distribution_oracle": True,
        "uses_source_mapping_rgb": False,
        "uses_test_queries": True,
        "uses_test_pose_metadata": True,
        "uses_test_rgb": False,
        "enters_track_registry": False,
        "enters_anchor_observation_csr": False,
        "enters_descriptor_bank": False,
        "render_protocol": "clean_once_per_pose",
        "mapping_camera_count": int(mapping_poses.shape[0]),
        "query_count": count,
        "query_indices": query_indices,
        "trajectory_statistics": statistics,
        "pose_w2c": test_poses,
        "intrinsics": intrinsics,
        "image_hw": image_hw,
        "source_mapping_indices": nearest.tolist(),
        "pose_family_ids": query_indices.clone(),
        "query_kinds": ["exact_test_pose_render_no_test_rgb"] * count,
        "test_camera_names_sha256": hashlib.sha256(
            "\n".join(map(str, test_names)).encode()
        ).hexdigest(),
    }
    payload["plan_sha256"] = _stable_digest(payload)
    require_view_role(payload, "test_pose_render_diagnostic")
    return payload
