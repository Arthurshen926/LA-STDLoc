"""Mapping-only sufficiency-guided virtual-camera planning.

The planner proposes poses and an immutable coverage audit.  It never renders
RGB, mutates Tracks, or consumes test queries.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import torch

from features.sampling import unproject_pixels


SCHEMA = "lafgs_sufficiency_guided_virtual_render_plan"
VERSION = 1


@dataclass(frozen=True)
class PlannerPolicy:
    voxel_size_m: float = 0.25
    surface_stride: int = 24
    alpha_minimum: float = 0.2
    target_families: int = 2
    target_view_bins: int = 2
    target_stable_observations: int = 3
    interpolation_fraction: float = 0.5
    perturb_translation_m: float = 0.10
    perturb_yaw_deg: float = 8.0
    boundary_expansion_m: float = 0.20
    envelope_margin_m: float = 0.35
    maximum_parent_distance_m: float = 1.0
    maximum_view_change_deg: float = 55.0
    maximum_artifact_risk: float = 0.65
    maximum_candidates: int = 512
    selected_view_budget: int = 32
    maximum_per_family: int = 1
    coverage_cap: float = 1.0
    parallax_weight: float = 0.20
    appearance_weight: float = 0.10
    risk_weight: float = 0.25


def camera_centers(pose_w2c: torch.Tensor) -> torch.Tensor:
    pose = torch.as_tensor(pose_w2c, dtype=torch.float64)
    return -(pose[:, :3, :3].transpose(1, 2) @ pose[:, :3, 3, None]).squeeze(2)


def _orthogonalize(rotation: torch.Tensor) -> torch.Tensor:
    u, _, vh = torch.linalg.svd(rotation)
    result = u @ vh
    if torch.linalg.det(result) < 0:
        u[:, -1] *= -1
        result = u @ vh
    return result


def _pose_from_center_rotation(center: torch.Tensor, rotation_w2c: torch.Tensor):
    pose = torch.eye(4, dtype=torch.float64)
    pose[:3, :3] = rotation_w2c
    pose[:3, 3] = -(rotation_w2c @ center)
    return pose


def _interpolate_pose(left: torch.Tensor, right: torch.Tensor, fraction: float):
    poses = torch.stack((left, right)).double()
    centers = camera_centers(poses)
    center = torch.lerp(centers[0], centers[1], float(fraction))
    rotation = _orthogonalize(
        torch.lerp(poses[0, :3, :3], poses[1, :3, :3], float(fraction))
    )
    return _pose_from_center_rotation(center, rotation)


def _look_at_pose(center: torch.Tensor, target: torch.Tensor) -> torch.Tensor | None:
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


def _pose_signature(pose: torch.Tensor) -> tuple[int, ...]:
    return tuple(torch.round(torch.as_tensor(pose).reshape(-1) * 1e7).long().tolist())


def generate_candidate_poses(
    pose_w2c: torch.Tensor,
    deficit_xyz: torch.Tensor,
    policy: PlannerPolicy,
) -> dict:
    """Generate a bounded geometry-only pool from declared pose operations."""
    poses = torch.as_tensor(pose_w2c, dtype=torch.float64)
    count = int(poses.shape[0])
    if poses.shape != (count, 4, 4) or count < 2:
        raise ValueError("planner requires at least two 4x4 mapping poses")
    centers = camera_centers(poses)
    distance = torch.cdist(centers, centers)
    distance.fill_diagonal_(torch.inf)
    proposals: list[tuple[torch.Tensor, int, str, tuple[int, ...], float]] = []

    # Geometry-nearest SE(3) interpolation.
    for left in range(count):
        right = int(torch.argmin(distance[left]))
        if left < right:
            proposals.append(
                (
                    _interpolate_pose(
                        poses[left], poses[right], policy.interpolation_fraction
                    ),
                    left,
                    "se3_interpolation",
                    (left, right),
                    0.10,
                )
            )

    yaw = math.radians(float(policy.perturb_yaw_deg))
    yaw_rotation = torch.tensor(
        [[math.cos(yaw), 0.0, math.sin(yaw)], [0.0, 1.0, 0.0],
         [-math.sin(yaw), 0.0, math.cos(yaw)]], dtype=torch.float64
    )
    for parent in range(count):
        c2w = poses[parent, :3, :3].T
        for sign in (-1.0, 1.0):
            center = centers[parent] + sign * float(policy.perturb_translation_m) * c2w[:, 0]
            proposals.append(
                (
                    _pose_from_center_rotation(center, poses[parent, :3, :3]),
                    parent,
                    "small_translation",
                    (parent,),
                    0.20,
                )
            )
            proposals.append(
                (
                    _pose_from_center_rotation(
                        centers[parent], yaw_rotation.pow(1) @ poses[parent, :3, :3]
                        if sign > 0 else yaw_rotation.T @ poses[parent, :3, :3]
                    ),
                    parent,
                    "small_rotation",
                    (parent,),
                    0.25,
                )
            )

    # Boundary expansion along the principal trajectory axis.
    centered = centers - centers.mean(0)
    _, _, vh = torch.linalg.svd(centered)
    axis = vh[0]
    projection = centered @ axis
    for parent, sign in ((int(torch.argmin(projection)), -1.0),
                         (int(torch.argmax(projection)), 1.0)):
        center = centers[parent] + sign * float(policy.boundary_expansion_m) * axis
        proposals.append((
            _pose_from_center_rotation(center, poses[parent, :3, :3]), parent,
            "boundary_expansion", (parent,), 0.45,
        ))

    # Reverse views are high-risk but can expose systematically unseen backsides.
    reverse = torch.diag(torch.tensor([-1.0, 1.0, -1.0], dtype=torch.float64))
    for parent in range(count):
        proposals.append((
            _pose_from_center_rotation(
                centers[parent], reverse @ poses[parent, :3, :3]
            ), parent, "reverse_view", (parent,), 0.60,
        ))

    # Deficit-directed views reuse a nearby source center and only change gaze.
    deficits = torch.as_tensor(deficit_xyz, dtype=torch.float64).reshape(-1, 3)
    if deficits.numel():
        order = torch.argsort(torch.linalg.norm(deficits - centers.mean(0), dim=1),
                              descending=True, stable=True)
        for deficit in deficits[order[: min(32, deficits.shape[0])]]:
            delta = deficit - centers
            distance_to_deficit = torch.linalg.norm(delta, dim=1).clamp_min(1e-8)
            rays = delta / distance_to_deficit[:, None]
            forwards = poses[:, :3, :3].transpose(1, 2)[:, :, 2]
            alignment = (rays * forwards).sum(1)
            normalized_distance = distance_to_deficit / distance_to_deficit.median().clamp_min(1e-8)
            parent = int(torch.argmax(alignment - 0.05 * normalized_distance))
            directed = _look_at_pose(centers[parent], deficit)
            if directed is not None:
                proposals.append((directed, parent, "deficit_directed", (parent,), 0.35))

    lower = centers.amin(0) - float(policy.envelope_margin_m)
    upper = centers.amax(0) + float(policy.envelope_margin_m)
    kept = []
    seen = set()
    for pose, parent, kind, family_members, base_risk in proposals:
        center = camera_centers(pose[None])[0]
        parent_distance = float(torch.linalg.norm(center - centers[parent]))
        forward = pose[:3, :3].T[:, 2]
        parent_forward = poses[parent, :3, :3].T[:, 2]
        angle = math.degrees(math.acos(float(torch.dot(forward, parent_forward).clamp(-1, 1))))
        risk = min(1.0, float(base_risk) + 0.2 * parent_distance /
                   max(float(policy.maximum_parent_distance_m), 1e-8))
        if (
            bool(((center < lower) | (center > upper)).any())
            or parent_distance > float(policy.maximum_parent_distance_m)
            or angle > float(policy.maximum_view_change_deg) and kind != "reverse_view"
            or risk > float(policy.maximum_artifact_risk)
        ):
            continue
        signature = _pose_signature(pose)
        if signature in seen:
            continue
        seen.add(signature)
        family = parent if kind in {"small_translation", "small_rotation"} else count + len(kept)
        kept.append((pose.float(), parent, kind, family, family_members, risk))
    if len(kept) > int(policy.maximum_candidates):
        by_kind: dict[str, list[tuple]] = defaultdict(list)
        for row in kept:
            by_kind[row[2]].append(row)
        kind_order = sorted(by_kind)
        positions = {kind: 0 for kind in kind_order}
        bounded = []
        while len(bounded) < int(policy.maximum_candidates):
            progress = False
            for kind in kind_order:
                position = positions[kind]
                if position < len(by_kind[kind]):
                    bounded.append(by_kind[kind][position])
                    positions[kind] += 1
                    progress = True
                    if len(bounded) == int(policy.maximum_candidates):
                        break
            if not progress:
                break
        kept = bounded
    return {
        "pose_w2c": torch.stack([row[0] for row in kept]) if kept else torch.empty(0, 4, 4),
        "parent_camera_index": torch.tensor([row[1] for row in kept], dtype=torch.long),
        "kind": [row[2] for row in kept],
        "pose_family": torch.tensor([row[3] for row in kept], dtype=torch.long),
        "family_source_cameras": [row[4] for row in kept],
        "artifact_risk": torch.tensor([row[5] for row in kept]),
    }


def greedy_capped_coverage(
    candidate_cells: Sequence[torch.Tensor],
    cell_demand: torch.Tensor,
    pose_family: torch.Tensor,
    *,
    budget: int,
    maximum_per_family: int = 1,
    coverage_cap: float = 1.0,
    parallax: torch.Tensor | None = None,
    appearance: torch.Tensor | None = None,
    artifact_risk: torch.Tensor | None = None,
    parallax_weight: float = 0.2,
    appearance_weight: float = 0.1,
    risk_weight: float = 0.25,
) -> tuple[torch.Tensor, list[dict]]:
    """Deterministic monotone capped-coverage greedy with family capacity."""
    count = len(candidate_cells)
    demand = torch.as_tensor(cell_demand).float().clamp_min(0)
    family = torch.as_tensor(pose_family).long()
    if family.shape != (count,):
        raise ValueError("pose_family must align with candidates")
    parallax = torch.zeros(count) if parallax is None else torch.as_tensor(parallax).float()
    appearance = torch.zeros(count) if appearance is None else torch.as_tensor(appearance).float()
    risk = torch.zeros(count) if artifact_risk is None else torch.as_tensor(artifact_risk).float()
    if any(value.shape != (count,) for value in (parallax, appearance, risk)):
        raise ValueError("candidate utility vectors must align")
    remaining = demand.clone()
    coverage_normalizer = max(float(demand.sum()), 1.0)
    used_family: defaultdict[int, int] = defaultdict(int)
    selected: list[int] = []
    trace: list[dict] = []
    for _ in range(min(int(budget), count)):
        best = None
        for candidate in range(count):
            if candidate in selected or used_family[int(family[candidate])] >= int(maximum_per_family):
                continue
            cells = torch.unique(torch.as_tensor(candidate_cells[candidate]).long())
            if cells.numel() and (int(cells.min()) < 0 or int(cells.max()) >= demand.numel()):
                raise ValueError("candidate coverage cell is outside field")
            coverage = float(torch.minimum(
                remaining[cells], torch.full_like(remaining[cells], float(coverage_cap))
            ).sum()) if cells.numel() else 0.0
            coverage_score = coverage / coverage_normalizer
            modular = (
                float(parallax_weight) * float(parallax[candidate])
                + float(appearance_weight) * float(appearance[candidate])
                + float(risk_weight) * (
                    1.0 - max(0.0, min(1.0, float(risk[candidate])))
                )
            )
            gain = coverage_score + modular
            key = (gain, coverage_score, coverage, -candidate)
            if best is None or key > best[0]:
                best = (key, candidate, cells, coverage, coverage_score, modular)
        if best is None or best[0][0] <= 0:
            break
        _, candidate, cells, coverage, coverage_score, modular = best
        selected.append(candidate)
        used_family[int(family[candidate])] += 1
        if cells.numel():
            remaining[cells] = (remaining[cells] - float(coverage_cap)).clamp_min(0)
        trace.append({
            "candidate_index": candidate,
            "pose_family": int(family[candidate]),
            "coverage_gain": coverage,
            "normalized_coverage_gain": coverage_score,
            "modular_gain": modular,
            "total_gain": coverage_score + modular,
            "remaining_demand": float(remaining.sum()),
        })
    return torch.tensor(selected, dtype=torch.long), trace


def validate_mapping_inputs(query_payload: Mapping, track_payload: Mapping) -> None:
    if query_payload.get("uses_test_queries") is not False:
        raise ValueError("virtual rendering planner requires mapping-only cache")
    if query_payload.get("uses_source_mapping_rgb") is not False:
        raise ValueError("virtual rendering planner requires rendered-RGB evidence")
    if track_payload.get("rendered_rgb_only") is not True:
        raise ValueError("virtual rendering planner requires rendered-RGB-only Tracks")
