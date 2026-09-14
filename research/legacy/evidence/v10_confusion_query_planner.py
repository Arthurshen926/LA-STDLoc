"""Constrained confusion-aware query planning without interpolation or full look-at."""

from __future__ import annotations

from collections.abc import Sequence
import hashlib
import math

import torch

from common.v7_contracts import require_view_role, view_role_contract
from evidence.v7_query_planner import camera_centers, trajectory_statistics
from evidence.v9_novel_query_planner import (
    _look_at,
    _pose_from_center_rotation,
    _rotation_angle,
    _visible_proxy,
)


SCHEMA = "lafgs_v10_confusion_query_plan"
VERSION = 1


def _limit_rotation(
    parent_w2c: torch.Tensor, target_w2c: torch.Tensor, maximum_angle_deg: float
) -> torch.Tensor:
    relative = target_w2c @ parent_w2c.T
    angle = math.radians(_rotation_angle(target_w2c, parent_w2c))
    if math.degrees(angle) <= float(maximum_angle_deg):
        return target_w2c
    skew = relative - relative.T
    axis = torch.tensor(
        [skew[2, 1], skew[0, 2], skew[1, 0]], dtype=torch.float64
    )
    if float(torch.linalg.norm(axis)) < 1e-8:
        return parent_w2c
    axis = torch.nn.functional.normalize(axis, dim=0)
    limited = math.radians(float(maximum_angle_deg))
    cross = torch.tensor(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ],
        dtype=torch.float64,
    )
    delta = (
        torch.eye(3, dtype=torch.float64)
        + math.sin(limited) * cross
        + (1.0 - math.cos(limited)) * (cross @ cross)
    )
    return delta @ parent_w2c


def _visible_pair(
    pair_xyz: torch.Tensor,
    pose: torch.Tensor,
    intrinsic: torch.Tensor,
    image_hw: torch.Tensor,
) -> bool:
    visible, _ = _visible_proxy(pair_xyz, pose, intrinsic, image_hw)
    return visible == 2


def plan_v10_confusion_queries(
    *,
    pose_w2c: torch.Tensor,
    intrinsics: torch.Tensor,
    image_hw: torch.Tensor,
    names: Sequence[str],
    anchor_xyz: torch.Tensor,
    confusion_pairs: torch.Tensor,
    role: str,
    feedback_stage: str | None,
    seed: int,
    maximum_queries: int = 128,
    forbidden_pose_family_ids: Sequence[int] = (),
    priority_anchor_rows: Sequence[int] = (),
    minimum_novel_baselines: float = 0.9,
    maximum_novel_baselines: float = 2.0,
    maximum_rotation_from_parent_deg: float = 40.0,
    anchor_projection_stride: int = 16,
    maximum_views_per_confusion_pair: int = 4,
) -> dict:
    """Plan bounded novel views that keep a current confusion pair visible."""

    if role not in {"feedback_query", "confirmation_query"}:
        raise ValueError("V10 planner only creates feedback or confirmation queries")
    if role == "feedback_query" and feedback_stage not in {"safety"}:
        raise ValueError("V10 feedback plan must declare the safety stage")
    if role == "confirmation_query" and feedback_stage is not None:
        raise ValueError("confirmation queries cannot have a feedback stage")
    poses = torch.as_tensor(pose_w2c, dtype=torch.float64)
    calibration = torch.as_tensor(intrinsics, dtype=torch.float64)
    hw = torch.as_tensor(image_hw, dtype=torch.long)
    anchors = torch.as_tensor(anchor_xyz, dtype=torch.float64).reshape(-1, 3)
    pairs = torch.as_tensor(confusion_pairs).long().reshape(-1, 2)
    if (
        poses.ndim != 3
        or poses.shape[1:] != (4, 4)
        or calibration.shape != (poses.shape[0], 3, 3)
        or hw.shape != (poses.shape[0], 2)
        or len(names) != poses.shape[0]
        or pairs.numel() == 0
        or int(pairs.min()) < 0
        or int(pairs.max()) >= anchors.shape[0]
    ):
        raise ValueError("V10 planner registries differ")
    sequence_ids = [str(name).split("/", 1)[0] for name in names]
    statistics = trajectory_statistics(poses, sequence_ids)
    baseline = max(statistics["median_adjacent_baseline_m"], 1e-4)
    centers = camera_centers(poses)
    forbidden = {int(value) for value in forbidden_pose_family_ids}
    priority_anchors = {int(value) for value in priority_anchor_rows}
    generator = torch.Generator().manual_seed(int(seed))
    pair_order = torch.randperm(pairs.shape[0], generator=generator).tolist()
    available = [index for index in range(poses.shape[0]) if index not in forbidden]
    sampled_anchors = anchors[:: max(int(anchor_projection_stride), 1)]
    proposals = []
    scales = (1.0, 1.25, 1.5, 1.8)
    signs = (1.0, -1.0)
    for order in pair_order:
        pair_rows = pairs[order]
        pair_xyz = anchors[pair_rows]
        midpoint = pair_xyz.mean(0)
        parent_candidates = sorted(
            available,
            key=lambda index: float(torch.linalg.norm(centers[index] - midpoint)),
        )[:384]
        accepted_parents = 0
        for parent_order, parent in enumerate(parent_candidates):
            if not _visible_pair(pair_xyz, poses[parent], calibration[parent], hw[parent]):
                continue
            c2w = poses[parent, :3, :3].T
            scale = scales[(order + parent_order + seed) % len(scales)]
            sign = signs[(order + parent_order + seed) % len(signs)]
            center = (
                centers[parent]
                + sign * 0.65 * scale * baseline * c2w[:, 0]
                + sign * 0.85 * scale * baseline * c2w[:, 2]
            )
            directed_pose = _look_at(center, midpoint)
            if directed_pose is None:
                continue
            rotation = _limit_rotation(
                poses[parent, :3, :3],
                directed_pose[:3, :3],
                maximum_rotation_from_parent_deg,
            )
            pose = _pose_from_center_rotation(center, rotation)
            distance = float(torch.cdist(camera_centers(pose[None]), centers).min() / baseline)
            parent_angle = _rotation_angle(rotation, poses[parent, :3, :3])
            if not (
                float(minimum_novel_baselines)
                <= distance
                <= float(maximum_novel_baselines)
                and parent_angle <= float(maximum_rotation_from_parent_deg) + 1e-6
                and _visible_pair(pair_xyz, pose, calibration[parent], hw[parent])
            ):
                continue
            visible, cells = _visible_proxy(
                sampled_anchors, pose, calibration[parent], hw[parent]
            )
            if visible < 32 or cells < 6:
                continue
            pair_separation = float(torch.linalg.norm(pair_xyz[0] - pair_xyz[1]))
            utility = (
                min(visible / 512.0, 1.0)
                + min(cells / 16.0, 1.0)
                + min(pair_separation / max(baseline, 1e-4), 1.0)
                + 1.0 - abs(distance - 1.4) / 1.4
                + (10.0 if int(pair_rows[0]) in priority_anchors else 0.0)
            )
            proposals.append(
                {
                    "pose": pose,
                    "parent": parent,
                    "pair_rows": pair_rows,
                    "novelty": distance,
                    "parent_angle": parent_angle,
                    "visible": visible,
                    "cells": cells,
                    "utility": utility,
                    "priority_target": int(pair_rows[0]) in priority_anchors,
                }
            )
            accepted_parents += 1
            if accepted_parents == int(maximum_views_per_confusion_pair):
                break
    proposals.sort(key=lambda row: (-row["utility"], row["parent"]))
    selected = []
    used_families = set()
    pair_counts: dict[tuple[int, int], int] = {}
    for row in proposals:
        pair_key = tuple(sorted(map(int, row["pair_rows"].tolist())))
        if (
            row["parent"] in used_families
            or pair_counts.get(pair_key, 0) >= int(maximum_views_per_confusion_pair)
        ):
            continue
        selected.append(row)
        used_families.add(row["parent"])
        pair_counts[pair_key] = pair_counts.get(pair_key, 0) + 1
        if len(selected) == int(maximum_queries):
            break
    if len(selected) < int(maximum_queries):
        raise RuntimeError(
            f"V10 bounded confusion planner produced {len(selected)}/{maximum_queries} queries"
        )
    source_rows = torch.tensor([row["parent"] for row in selected], dtype=torch.long)
    output = {
        "schema": SCHEMA,
        "version": VERSION,
        **view_role_contract(role),
        "feedback_stage": feedback_stage,
        "uses_source_mapping_rgb": False,
        "uses_test_queries": False,
        "enters_track_registry": False,
        "enters_anchor_observation_csr": False,
        "enters_descriptor_bank": False,
        "render_protocol": "clean_once_per_pose",
        "loo_used": False,
        "trajectory_interpolation_candidate_count": 0,
        "ambiguity_full_look_at": False,
        "seed": int(seed),
        "mapping_camera_count": int(poses.shape[0]),
        "query_count": len(selected),
        "trajectory_statistics": statistics,
        "pose_w2c": torch.stack([row["pose"] for row in selected]),
        "intrinsics": calibration[source_rows].clone(),
        "image_hw": hw[source_rows].clone(),
        "source_mapping_indices": [[int(row["parent"])] for row in selected],
        "pose_family_ids": source_rows.clone(),
        "query_kinds": ["bounded_confusion_excitation"] * len(selected),
        "confusion_anchor_rows": torch.stack([row["pair_rows"] for row in selected]),
        "priority_target_mask": torch.tensor(
            [row["priority_target"] for row in selected], dtype=torch.bool
        ),
        "priority_target_count": sum(int(row["priority_target"]) for row in selected),
        "novelty_baselines": torch.tensor([row["novelty"] for row in selected]),
        "rotation_from_parent_deg": torch.tensor(
            [row["parent_angle"] for row in selected]
        ),
        "visible_anchor_proxy_count": torch.tensor(
            [row["visible"] for row in selected]
        ),
        "visible_cell_count": torch.tensor([row["cells"] for row in selected]),
        "planner_contract": {
            "trajectory_interpolation": False,
            "ambiguity_full_look_at": False,
            "minimum_novel_baselines": float(minimum_novel_baselines),
            "maximum_novel_baselines": float(maximum_novel_baselines),
            "maximum_rotation_from_parent_deg": float(
                maximum_rotation_from_parent_deg
            ),
            "confusion_pair_visible": True,
            "family_unique": True,
            "priority_anchor_count": len(priority_anchors),
            "maximum_views_per_confusion_pair": int(maximum_views_per_confusion_pair),
        },
    }
    digest = hashlib.sha256()
    digest.update(output["pose_w2c"].contiguous().numpy().tobytes())
    digest.update(str(seed).encode())
    digest.update(role.encode())
    output["plan_sha256"] = digest.hexdigest()
    require_view_role(output, role)
    return output
