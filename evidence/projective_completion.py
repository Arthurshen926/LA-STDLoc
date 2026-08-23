"""Depth-proposed, descriptor-verified, ray-triangulated completion Anchors."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from common.v6_contracts import ANCHOR_CANDIDATE_SCHEMA
from evidence.observation_provider import ObservationProvider
from evidence.projective_reconstruction import reconstruct_projective_anchors
from evidence.triangulation import reciprocal_epipolar_matches
from features.raster_sampling import sample_raster_at_grid_uv


def _unproject(view, rows: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
    uv = view.keypoints[rows].float() + float(view.pixel_center_offset)
    z = depth[rows]
    camera = torch.stack(
        (
            (uv[:, 0] - view.intrinsics[0, 2]) / view.intrinsics[0, 0] * z,
            (uv[:, 1] - view.intrinsics[1, 2]) / view.intrinsics[1, 1] * z,
            z,
        ),
        dim=1,
    )
    return (camera - view.pose_w2c[:3, 3]) @ view.pose_w2c[:3, :3]


@torch.no_grad()
def build_projective_completion(
    observations: ObservationProvider,
    base_association: dict,
    *,
    voxel_size_m: float,
    alpha_minimum: float,
    minimum_similarity: float,
    minimum_margin: float = 0.01,
    maximum_epipolar_error_px: float = 2.0,
    minimum_observations: int = 3,
    minimum_camera_families: int = 2,
    maximum_rows_per_view: int = 256,
    safety_maximum_components: int = 100000,
    eligible_query_indices: torch.Tensor | list[int] | None = None,
    target_query_indices: torch.Tensor | list[int] | None = None,
    excluded_support_query_indices: torch.Tensor | list[int] | None = None,
    device: str = "cuda",
) -> dict:
    """Use Gaussian depth only to propose neighborhoods; deploy only ray xyz."""

    if not float(voxel_size_m) > 0:
        raise ValueError("voxel size must be positive")
    if not 0.0 <= float(minimum_similarity) <= 1.0:
        raise ValueError("minimum similarity must lie in [0,1]")
    if list(observations.names) != list(base_association.get("query_names", ())):
        raise ValueError("completion registry differs from base association")
    used = {
        (int(query), int(row))
        for query, row in zip(
            torch.as_tensor(base_association["tracks"]["query_index"]).tolist(),
            torch.as_tensor(base_association["tracks"]["keypoint_index"]).tolist(),
        )
    }
    xyz_parts = []
    descriptor_parts = []
    score_parts = []
    query_parts = []
    keypoint_parts = []
    eligible_queries = (
        set(range(len(observations)))
        if eligible_query_indices is None
        else {int(value) for value in torch.as_tensor(eligible_query_indices).tolist()}
    )
    if not eligible_queries or min(eligible_queries) < 0 or max(eligible_queries) >= len(observations):
        raise ValueError("completion eligible query registry is empty or invalid")
    target_queries = (
        None
        if target_query_indices is None
        else {int(value) for value in torch.as_tensor(target_query_indices).tolist()}
    )
    if target_queries is not None and (
        not target_queries
        or min(target_queries) < 0
        or max(target_queries) >= len(observations)
    ):
        raise ValueError("completion target query registry is empty or invalid")
    excluded_support_queries = (
        set() if target_queries is None else set(target_queries)
    )
    if excluded_support_query_indices is not None:
        excluded_support_queries.update(
            int(value)
            for value in torch.as_tensor(excluded_support_query_indices).tolist()
        )
    if excluded_support_queries and (
        min(excluded_support_queries) < 0
        or max(excluded_support_queries) >= len(observations)
    ):
        raise ValueError("completion excluded support registry is invalid")
    proposal_queries = set(eligible_queries)
    if target_queries is not None:
        proposal_queries.update(target_queries)
    for query_index in sorted(proposal_queries):
        view = observations.build_view(query_index)
        if view.depth is None and view.keypoint_depth is None:
            raise ValueError("completion requires rendered depth proposals")
        depth = (
            view.keypoint_depth.float()
            if view.keypoint_depth is not None
            else sample_raster_at_grid_uv(view.depth, view.keypoints).float()
        )
        alpha = (
            view.keypoint_alpha.float()
            if view.keypoint_alpha is not None
            else sample_raster_at_grid_uv(view.alpha, view.keypoints).float()
        )
        valid = torch.isfinite(depth) & (depth > 0) & torch.isfinite(alpha) & (
            alpha >= float(alpha_minimum)
        )
        rows = torch.as_tensor(
            [
                row
                for row in torch.nonzero(valid, as_tuple=False).reshape(-1).tolist()
                if (query_index, row) not in used
            ],
            dtype=torch.long,
        )
        if rows.numel() > int(maximum_rows_per_view):
            order = torch.argsort(
                view.detector_scores[rows], descending=True, stable=True
            )
            rows = rows[order[: int(maximum_rows_per_view)]]
        if rows.numel() == 0:
            continue
        xyz_parts.append(_unproject(view, rows, depth))
        descriptor_parts.append(F.normalize(view.descriptors[rows].float(), dim=1))
        score_parts.append(view.detector_scores[rows].float())
        query_parts.append(torch.full((rows.numel(),), query_index, dtype=torch.long))
        keypoint_parts.append(rows)
    if not xyz_parts:
        raise ValueError("no unused render-valid observations for completion")
    proposal_xyz = torch.cat(xyz_parts)
    descriptors = torch.cat(descriptor_parts)
    scores = torch.cat(score_parts)
    query = torch.cat(query_parts)
    keypoint = torch.cat(keypoint_parts)
    voxel = torch.floor(proposal_xyz / float(voxel_size_m)).long()
    _, inverse = torch.unique(voxel, dim=0, sorted=True, return_inverse=True)
    target_voxels = None
    if target_queries is not None:
        target_rows = torch.isin(
            query, torch.tensor(sorted(target_queries), dtype=torch.long)
        )
        target_voxels = set(inverse[target_rows].tolist())
        if not target_voxels:
            raise ValueError("target queries produced no completion seed region")
    groups = []
    bins = torch.as_tensor(base_association["query_bins"]).long()
    views = [observations.build_view(index) for index in range(len(observations))]
    ordered_proposals = torch.argsort(inverse, stable=True)
    voxel_count = int(inverse.max()) + 1
    voxel_offsets = torch.cat(
        (
            torch.zeros(1, dtype=torch.long),
            torch.bincount(inverse, minlength=voxel_count).cumsum(0),
        )
    )
    for group in range(voxel_count):
        if target_voxels is not None and group not in target_voxels:
            continue
        proposal_rows = ordered_proposals[
            voxel_offsets[group] : voxel_offsets[group + 1]
        ]
        support = torch.tensor(
            [
                int(value) not in excluded_support_queries
                and int(value) in eligible_queries
                for value in query[proposal_rows].tolist()
            ],
            dtype=torch.bool,
        )
        proposal_rows = proposal_rows[support]
        if proposal_rows.numel() < int(minimum_observations):
            continue
        unique_queries = torch.unique(query[proposal_rows], sorted=True)
        if unique_queries.numel() < int(minimum_observations):
            continue
        parent = list(range(int(proposal_rows.numel())))

        def find(value: int) -> int:
            while parent[value] != value:
                parent[value] = parent[parent[value]]
                value = parent[value]
            return value

        def union(left: int, right: int) -> None:
            a, b = find(left), find(right)
            if a != b:
                parent[max(a, b)] = min(a, b)

        for left_pos in range(int(unique_queries.numel())):
            left_query = int(unique_queries[left_pos])
            left_local = torch.nonzero(
                query[proposal_rows] == left_query, as_tuple=False
            ).reshape(-1)
            for right_pos in range(left_pos + 1, int(unique_queries.numel())):
                right_query = int(unique_queries[right_pos])
                right_local = torch.nonzero(
                    query[proposal_rows] == right_query, as_tuple=False
                ).reshape(-1)
                source, target, _ = reciprocal_epipolar_matches(
                    descriptors[proposal_rows[left_local]],
                    descriptors[proposal_rows[right_local]],
                    views[left_query].physical_keypoints[keypoint[proposal_rows[left_local]]],
                    views[right_query].physical_keypoints[keypoint[proposal_rows[right_local]]],
                    views[left_query].intrinsics,
                    views[left_query].pose_w2c,
                    views[right_query].intrinsics,
                    views[right_query].pose_w2c,
                    minimum_similarity=float(minimum_similarity),
                    minimum_margin=float(minimum_margin),
                    maximum_epipolar_error_px=float(maximum_epipolar_error_px),
                    epipolar_candidate_topk=4,
                )
                for left_row, right_row in zip(source.tolist(), target.tolist()):
                    union(int(left_local[left_row]), int(right_local[right_row]))
        components: dict[int, list[int]] = {}
        for local in range(int(proposal_rows.numel())):
            components.setdefault(find(local), []).append(local)
        for local_rows in components.values():
            rows = proposal_rows[torch.tensor(local_rows, dtype=torch.long)]
            # The association contract permits one observation per camera.
            order = rows
            for values, descending in (
                (keypoint, False),
                (scores, True),
                (query, False),
            ):
                order = order[
                    torch.argsort(values[order], descending=descending, stable=True)
                ]
            ordered_query = query[order]
            keep = torch.ones(order.numel(), dtype=torch.bool)
            if order.numel() > 1:
                keep[1:] = ordered_query[1:] != ordered_query[:-1]
            rows = order[keep]
            if rows.numel() < int(minimum_observations):
                continue
            if torch.unique(bins[query[rows]]).numel() < int(minimum_camera_families):
                continue
            groups.append(rows)
    if len(groups) > int(safety_maximum_components):
        raise RuntimeError("completion safety cap triggered; refusing silent truncation")
    if not groups:
        raise ValueError("depth proposals produced no descriptor-consistent component")
    track_index = torch.cat(
        [torch.full((rows.numel(),), index, dtype=torch.long) for index, rows in enumerate(groups)]
    )
    flat = torch.cat(groups)
    association = {
        "schema": "projective_association_graph_v2",
        "version": 2,
        "uses_source_mapping_rgb": False,
        "uses_test_queries": False,
        "query_names": list(observations.names),
        "query_bins": bins,
        "tracks": {
            "track_index": track_index,
            "query_index": query[flat],
            "keypoint_index": keypoint[flat],
            "confidence": scores[flat].clamp(0, 1),
            "track_level": torch.ones(len(groups), dtype=torch.long),
        },
        "diagnostics": {"track_count": len(groups)},
        "component_statistics": {
            "identity_reliability": torch.stack(
                [
                    (descriptors[rows] @ F.normalize(descriptors[rows].mean(0), dim=0)).mean()
                    for rows in groups
                ]
            ).clamp(0, 1)
        },
    }
    result = reconstruct_projective_anchors(observations, association)
    result["schema"] = ANCHOR_CANDIDATE_SCHEMA
    result["candidate_kind"] = "depth_proposed_projective_completion"
    result["contract"].update(
        gaussian_depth_role="proposal_neighborhood_only",
        target_queries_seed_regions=target_queries is not None,
        support_queries_restricted=eligible_query_indices is not None,
        target_queries_used_as_anchor_support=False,
        excluded_support_query_count=len(excluded_support_queries),
        reciprocal_local_descriptor_support=True,
        known_pose_epipolar_support=True,
        final_xyz_source="fixed_camera_robust_ray_triangulation",
        safety_cap_behavior="fail_closed_not_truncate",
    )
    return result
