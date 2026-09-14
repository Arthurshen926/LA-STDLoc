"""Interpolation-free novel/ambiguity query planning for V9 feedback."""

from __future__ import annotations

from collections.abc import Sequence
import hashlib
import math

import torch

from common.v7_contracts import require_view_role, view_role_contract
from evidence.v7_query_planner import camera_centers, trajectory_statistics


SCHEMA = "lafgs_v9_novel_query_plan"
VERSION = 1


def _pose_from_center_rotation(
    center: torch.Tensor, rotation_w2c: torch.Tensor
) -> torch.Tensor:
    pose = torch.eye(4, dtype=torch.float64)
    pose[:3, :3] = rotation_w2c
    pose[:3, 3] = -(rotation_w2c @ center)
    return pose


def _axis_rotation(axis: int, radians: float) -> torch.Tensor:
    value = torch.eye(3, dtype=torch.float64)
    left, right = ((1, 2), (0, 2), (0, 1))[int(axis)]
    cosine, sine = math.cos(radians), math.sin(radians)
    value[left, left] = cosine
    value[right, right] = cosine
    value[left, right] = -sine
    value[right, left] = sine
    return value


def _look_at(center: torch.Tensor, target: torch.Tensor) -> torch.Tensor | None:
    forward = target - center
    if float(torch.linalg.norm(forward)) < 1e-8:
        return None
    forward = torch.nn.functional.normalize(forward, dim=0)
    up = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64)
    if abs(float(torch.dot(forward, up))) > 0.95:
        up = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
    right = torch.nn.functional.normalize(torch.cross(forward, up, dim=0), dim=0)
    down = torch.cross(forward, right, dim=0)
    return _pose_from_center_rotation(center, torch.stack((right, down, forward)))


def _rotation_angle(left: torch.Tensor, right: torch.Tensor) -> float:
    relative = left @ right.T
    cosine = ((torch.trace(relative) - 1.0) * 0.5).clamp(-1.0, 1.0)
    return math.degrees(math.acos(float(cosine)))


def _visible_proxy(
    xyz: torch.Tensor,
    pose: torch.Tensor,
    intrinsic: torch.Tensor,
    image_hw: torch.Tensor,
) -> tuple[int, int]:
    camera = xyz @ pose[:3, :3].T + pose[:3, 3]
    projected = camera @ intrinsic.T
    uv = projected[:, :2] / projected[:, 2:].clamp_min(1e-8)
    height, width = map(int, image_hw.tolist())
    valid = (
        torch.isfinite(uv).all(1)
        & torch.isfinite(camera[:, 2])
        & (camera[:, 2] > 0)
        & (uv[:, 0] >= 0)
        & (uv[:, 0] < width)
        & (uv[:, 1] >= 0)
        & (uv[:, 1] < height)
    )
    if not bool(valid.any()):
        return 0, 0
    pixels = uv[valid]
    cell_x = (pixels[:, 0] * 4 / max(width, 1)).long().clamp(0, 3)
    cell_y = (pixels[:, 1] * 4 / max(height, 1)).long().clamp(0, 3)
    return int(valid.sum()), int(torch.unique(cell_y * 4 + cell_x).numel())


def plan_v9_novel_queries(
    *,
    pose_w2c: torch.Tensor,
    intrinsics: torch.Tensor,
    image_hw: torch.Tensor,
    names: Sequence[str],
    anchor_xyz: torch.Tensor,
    ambiguity_xyz: torch.Tensor | None,
    role: str,
    seed: int,
    maximum_queries: int = 128,
    candidate_multiplier: int = 8,
    anchor_projection_stride: int = 16,
    minimum_novel_baselines: float = 0.65,
    minimum_view_angle_deg: float = 8.0,
    forbidden_pose_family_ids: Sequence[int] = (),
) -> dict:
    """Plan unseen but map-overlapping views without trajectory interpolation."""

    if role not in {"feedback_query", "confirmation_query"}:
        raise ValueError("V9 planner only creates feedback or confirmation queries")
    poses = torch.as_tensor(pose_w2c, dtype=torch.float64)
    intrinsic = torch.as_tensor(intrinsics, dtype=torch.float64)
    hw = torch.as_tensor(image_hw, dtype=torch.long)
    anchors = torch.as_tensor(anchor_xyz, dtype=torch.float64).reshape(-1, 3)
    ambiguity = torch.as_tensor(
        torch.empty(0, 3) if ambiguity_xyz is None else ambiguity_xyz,
        dtype=torch.float64,
    ).reshape(-1, 3)
    if (
        poses.ndim != 3
        or poses.shape[1:] != (4, 4)
        or intrinsic.shape != (poses.shape[0], 3, 3)
        or hw.shape != (poses.shape[0], 2)
        or len(names) != poses.shape[0]
        or anchors.shape[0] == 0
    ):
        raise ValueError("V9 mapping camera/Anchor registries differ")
    sequence_ids = [str(name).split("/", 1)[0] for name in names]
    statistics = trajectory_statistics(poses, sequence_ids)
    baseline = max(statistics["median_adjacent_baseline_m"], 1e-4)
    centers = camera_centers(poses)
    generator = torch.Generator().manual_seed(int(seed))
    forbidden_families = {int(value) for value in forbidden_pose_family_ids}
    parent_order = torch.randperm(poses.shape[0], generator=generator).tolist()
    role_sign = 1.0 if role == "feedback_query" else -1.0
    proposals: list[tuple[torch.Tensor, int, str, int]] = []
    target_count = int(maximum_queries) * int(candidate_multiplier)
    scales = (0.85, 1.25, 1.75, 2.5)
    yaws = (10.0, 15.0, 22.0, 30.0)
    for order, parent in enumerate(parent_order):
        if parent in forbidden_families:
            continue
        c2w = poses[parent, :3, :3].T
        scale = scales[(order + int(seed)) % len(scales)]
        sign = role_sign * (1.0 if order % 2 == 0 else -1.0)
        center = (
            centers[parent]
            + sign * 0.70 * scale * baseline * c2w[:, 0]
            + ((order % 3) - 1) * 0.25 * baseline * c2w[:, 1]
            + sign * 0.80 * scale * baseline * c2w[:, 2]
        )
        yaw = math.radians(sign * yaws[order % len(yaws)])
        pitch = math.radians(role_sign * (8.0 if order % 2 == 0 else -8.0))
        rotation = _axis_rotation(0, pitch) @ _axis_rotation(1, yaw) @ poses[
            parent, :3, :3
        ]
        proposals.append(
            (_pose_from_center_rotation(center, rotation), parent, "novel_se3", parent)
        )
        if len(proposals) >= target_count:
            break
    if ambiguity.numel():
        ambiguity_order = torch.randperm(
            ambiguity.shape[0], generator=generator
        ).tolist()
        for order, target in enumerate(ambiguity[ambiguity_order]):
            nearest = int(torch.argmin(torch.linalg.norm(centers - target, dim=1)))
            if nearest in forbidden_families:
                continue
            c2w = poses[nearest, :3, :3].T
            sign = role_sign * (1.0 if order % 2 == 0 else -1.0)
            center = (
                centers[nearest]
                + sign * (1.25 + 0.5 * (order % 3)) * baseline * c2w[:, 0]
                - 0.5 * baseline * c2w[:, 2]
                + ((order % 3) - 1) * 0.2 * baseline * c2w[:, 1]
            )
            directed = _look_at(center, target)
            if directed is not None:
                proposals.append((directed, nearest, "ambiguity_directed", nearest))
            if len(proposals) >= target_count * 2:
                break
    sampled_anchors = anchors[:: max(int(anchor_projection_stride), 1)]
    sampled_ambiguity = ambiguity[:: max(int(anchor_projection_stride // 4), 1)]
    candidates = []
    for pose, parent, kind, family in proposals:
        center = camera_centers(pose[None])[0]
        distances = torch.linalg.norm(centers - center, dim=1)
        nearest = int(torch.argmin(distances))
        novelty_distance = float(distances[nearest] / baseline)
        angle = _rotation_angle(pose[:3, :3], poses[nearest, :3, :3])
        if (
            novelty_distance < float(minimum_novel_baselines)
            or angle < float(minimum_view_angle_deg)
        ):
            continue
        visible, cells = _visible_proxy(
            sampled_anchors, pose, intrinsic[parent], hw[parent]
        )
        if visible < 32 or cells < 6:
            continue
        ambiguity_visible = 0
        if sampled_ambiguity.numel():
            ambiguity_visible, _ = _visible_proxy(
                sampled_ambiguity, pose, intrinsic[parent], hw[parent]
            )
        score = (
            min(novelty_distance / 2.5, 1.0)
            + min(angle / 30.0, 1.0)
            + min(cells / 16.0, 1.0)
            + 0.5 * min(ambiguity_visible / 16.0, 1.0)
        )
        candidates.append(
            {
                "pose": pose,
                "parent": parent,
                "family": family,
                "kind": kind,
                "novelty_baselines": novelty_distance,
                "nearest_view_angle_deg": angle,
                "visible_anchor_proxy_count": visible,
                "visible_cell_count": cells,
                "visible_ambiguity_proxy_count": ambiguity_visible,
                "utility": score,
            }
        )
    candidates.sort(
        key=lambda row: (-row["utility"], row["kind"], row["parent"])
    )
    selected = []
    used_families = set()
    kind_counts = {"novel_se3": 0, "ambiguity_directed": 0}
    for row in candidates:
        if row["family"] in used_families:
            continue
        # Maintain ambiguity excitation without reserving any interpolation arm.
        if (
            row["kind"] == "novel_se3"
            and ambiguity.numel()
            and kind_counts["novel_se3"] >= math.ceil(maximum_queries * 0.75)
        ):
            continue
        selected.append(row)
        used_families.add(row["family"])
        kind_counts[row["kind"]] += 1
        if len(selected) == int(maximum_queries):
            break
    # Ambiguity-directed views are preferred, never a hard quota: an imperfect
    # ambiguity proxy must not shrink an otherwise valid novel-view batch.
    if len(selected) < int(maximum_queries):
        for row in candidates:
            if row["family"] in used_families:
                continue
            selected.append(row)
            used_families.add(row["family"])
            kind_counts[row["kind"]] += 1
            if len(selected) == int(maximum_queries):
                break
    if len(selected) < int(maximum_queries):
        raise RuntimeError(
            f"V9 novelty gates produced {len(selected)}/{maximum_queries} queries"
        )
    source_rows = torch.tensor([row["parent"] for row in selected], dtype=torch.long)
    output = {
        "schema": SCHEMA,
        "version": VERSION,
        **view_role_contract(role),
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
        "pose_w2c": torch.stack([row["pose"] for row in selected]),
        "intrinsics": intrinsic[source_rows].clone(),
        "image_hw": hw[source_rows].clone(),
        "source_mapping_indices": [[int(row["parent"])] for row in selected],
        "pose_family_ids": source_rows.clone(),
        "query_kinds": [row["kind"] for row in selected],
        "novelty_baselines": torch.tensor(
            [row["novelty_baselines"] for row in selected]
        ),
        "nearest_view_angle_deg": torch.tensor(
            [row["nearest_view_angle_deg"] for row in selected]
        ),
        "visible_anchor_proxy_count": torch.tensor(
            [row["visible_anchor_proxy_count"] for row in selected]
        ),
        "visible_cell_count": torch.tensor(
            [row["visible_cell_count"] for row in selected]
        ),
        "visible_ambiguity_proxy_count": torch.tensor(
            [row["visible_ambiguity_proxy_count"] for row in selected]
        ),
        "planner_contract": {
            "trajectory_interpolation": False,
            "geometry_mutation": False,
            "test_pose_distribution": False,
            "minimum_novel_baselines": float(minimum_novel_baselines),
            "minimum_view_angle_deg": float(minimum_view_angle_deg),
            "forbidden_pose_family_count": len(forbidden_families),
            "selection": "novelty_overlap_ambiguity_utility_family_unique",
        },
    }
    digest = hashlib.sha256()
    digest.update(output["pose_w2c"].contiguous().numpy().tobytes())
    digest.update(str(seed).encode())
    digest.update(role.encode())
    output["plan_sha256"] = digest.hexdigest()
    require_view_role(output, role)
    return output
