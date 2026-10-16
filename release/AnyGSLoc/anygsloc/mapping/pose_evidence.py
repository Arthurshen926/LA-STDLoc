#!/usr/bin/env python3


"""Frozen mainline kernel extracted from anygsloc.mapping.pose_evidence."""

from __future__ import annotations

import math

import torch

from anygsloc.mapping.coverage import IncrementalBipartiteCoverage
from anygsloc.mapping.observability import PoseEvidence
from anygsloc.mapping.pose_information import (
    fisher_contributions,
    pose_jacobian_analytic,
    task_scaled_pose_jacobian,
)


def _ordered_rows(record: dict, query_cache: dict) -> tuple[int, ...]:
    rows = tuple(int(value) for value in record)
    if len(rows) < 2:
        return rows
    score = torch.as_tensor(query_cache["native_scores"])
    return tuple(sorted(rows, key=lambda row: (-float(score[row]), row)))


def _query_row_matchability(cached: dict, rows: tuple[int, ...]) -> float:
    """Return a query-conditioned repeatability factor for one candidate edge."""
    if not rows:
        return 0.0
    score = torch.as_tensor(cached["native_scores"]).float().reshape(-1)
    finite = score[torch.isfinite(score)]
    if finite.numel() < 2:
        return 1.0
    low = torch.quantile(finite, 0.1)
    high = torch.quantile(finite, 0.9)
    selected = score[torch.as_tensor(rows).long()].max()
    normalized = ((selected - low) / (high - low).clamp_min(1e-6)).clamp(0, 1)
    # Do not discard weak detections completely: global track/graph evidence is
    # still informative, while the query factor downweights fragile views.
    return float(0.5 + 0.5 * normalized)


def _project_world_covariance(
    points: torch.Tensor,
    covariance_world: torch.Tensor,
    intrinsic: torch.Tensor,
    pose_w2c: torch.Tensor,
) -> torch.Tensor:
    """Project anisotropic 3D landmark covariance into image coordinates."""
    camera = points @ pose_w2c[:3, :3].T + pose_w2c[:3, 3]
    x, y, z = camera.unbind(dim=1)
    z = z.clamp_min(1e-8)
    dproj = points.new_zeros((points.shape[0], 2, 3))
    dproj[:, 0, 0] = intrinsic[0, 0] / z
    dproj[:, 0, 2] = -intrinsic[0, 0] * x / z.square()
    dproj[:, 1, 1] = intrinsic[1, 1] / z
    dproj[:, 1, 2] = -intrinsic[1, 1] * y / z.square()
    jacobian = dproj @ pose_w2c[:3, :3]
    projected = jacobian @ covariance_world @ jacobian.transpose(1, 2)
    return 0.5 * (projected + projected.transpose(1, 2))


def _build_pose_evidence(
    edges,
    xyz: torch.Tensor,
    matchability: torch.Tensor,
    candidate_covariance: torch.Tensor,
    query_names: list[str],
    query_cache: dict,
    voxel_ids: torch.Tensor,
    *,
    pixel_variance: float,
    task_translation_m: float,
    task_rotation_deg: float,
) -> list[list[PoseEvidence]]:
    query_candidates: list[list[int]] = [[] for _ in query_names]
    for candidate, candidate_edges in enumerate(edges):
        for query in candidate_edges:
            query_candidates[int(query)].append(candidate)
    evidence: list[list[PoseEvidence]] = [[] for _ in edges]
    if candidate_covariance.shape != (len(edges), 3, 3):
        raise ValueError("candidate covariance must have shape [candidate_count,3,3]")
    for query, candidates in enumerate(query_candidates):
        if not candidates:
            continue
        name = query_names[query]
        cached = query_cache[name]
        candidate_tensor = torch.as_tensor(candidates).long()
        points = xyz[candidate_tensor].double()
        K = torch.as_tensor(cached["native_K"]).double()
        pose = torch.as_tensor(cached["pose_w2c"]).double()
        jacobian = task_scaled_pose_jacobian(
            pose_jacobian_analytic(points, K, pose),
            translation_scale=float(task_translation_m),
            rotation_scale=math.radians(float(task_rotation_deg)),
        )
        ones = torch.ones((points.shape[0], 1), dtype=torch.float64)
        depth = (pose @ torch.cat((points, ones), dim=1).T)[2]
        covariance = torch.eye(2, dtype=torch.float64)[None].repeat(
            points.shape[0], 1, 1
        ) * float(pixel_variance)
        selected_covariance = candidate_covariance[candidate_tensor].double()
        finite_covariance = (
            torch.isfinite(selected_covariance)
            .reshape(selected_covariance.shape[0], -1)
            .all(dim=1)
        )
        if bool(finite_covariance.any()):
            covariance[finite_covariance] += _project_world_covariance(
                points[finite_covariance],
                selected_covariance[finite_covariance],
                K,
                pose,
            )
        row_matchability = []
        ordered_rows = []
        for candidate in candidates:
            rows = _ordered_rows(edges[candidate][query], cached)
            ordered_rows.append(rows)
            row_matchability.append(_query_row_matchability(cached, rows))
        query_matchability = matchability[candidate_tensor].double() * torch.as_tensor(
            row_matchability, dtype=torch.float64
        )
        information = fisher_contributions(
            jacobian,
            weights=query_matchability,
            measurement_covariance=covariance,
        )
        keypoints = torch.as_tensor(cached["native_keypoints"]).float()
        height, width = (int(value) for value in cached["native_input_hw"])
        for local, candidate in enumerate(candidates):
            rows = ordered_rows[local]
            if not rows:
                continue
            point = keypoints[rows[0]]
            cell_x = int(torch.floor(point[0] / max(width, 1) * 4).clamp(0, 3))
            cell_y = int(torch.floor(point[1] / max(height, 1) * 4).clamp(0, 3))
            depth_bin = int(torch.floor(torch.log2(depth[local].clamp_min(0.05)) * 2))
            evidence[candidate].append(
                PoseEvidence(
                    query=query,
                    rows=rows,
                    information=information[local],
                    image_cell=cell_y * 4 + cell_x,
                    depth_bin=depth_bin,
                    spatial_voxel=int(voxel_ids[candidate]),
                    matchability=float(query_matchability[local]),
                )
            )
    return evidence


def _initial_pose_state(
    selected: torch.Tensor,
    matching: IncrementalBipartiteCoverage,
    evidence_by_candidate,
    query_count: int,
) -> tuple[
    torch.Tensor,
    list[set[int]],
    list[set[int]],
    list[set[int]],
    list[set[int]],
    list[dict[int, int]],
]:
    information = torch.eye(6, dtype=torch.float64)[None].repeat(query_count, 1, 1)
    information *= 1e-4
    rows = [set() for _ in range(query_count)]
    cells = [set() for _ in range(query_count)]
    depths = [set() for _ in range(query_count)]
    voxels = [set() for _ in range(query_count)]
    selected_set = set(torch.as_tensor(selected).long().tolist())
    assignments = [dict() for _ in range(query_count)]
    evidence_lookup = [
        {item.query: item for item in candidate} for candidate in evidence_by_candidate
    ]
    for query in range(query_count):
        for candidate, row in matching.assignments(query).items():
            if candidate not in selected_set:
                continue
            item = evidence_lookup[candidate].get(query)
            if item is None:
                continue
            information[query] += item.information
            rows[query].add(int(row))
            assignments[query][int(candidate)] = int(row)
            cells[query].add(item.image_cell)
            depths[query].add(item.depth_bin)
            voxels[query].add(item.spatial_voxel)
    return information, rows, cells, depths, voxels, assignments
