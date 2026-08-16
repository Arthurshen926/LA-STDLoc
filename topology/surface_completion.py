"""Source-image-free Gaussian surface completion from rendered RGB/depth views."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from evidence.observation_provider import ObservationProvider
from evidence.tracks import fuse_projective_anchor_observations


def _unproject_to_world(
    keypoints: torch.Tensor,
    depth: torch.Tensor,
    intrinsic: torch.Tensor,
    pose_w2c: torch.Tensor,
) -> torch.Tensor:
    physical = keypoints.float() + 0.5
    camera = torch.stack(
        (
            (physical[:, 0] - intrinsic[0, 2]) / intrinsic[0, 0] * depth,
            (physical[:, 1] - intrinsic[1, 2]) / intrinsic[1, 1] * depth,
            depth,
        ),
        dim=1,
    )
    return (camera - pose_w2c[:3, 3]) @ pose_w2c[:3, :3]


@torch.inference_mode()
def materialize_gaussian_surface_completion(
    observations: ObservationProvider,
    *,
    voxel_size_m: float,
    maximum_candidates: int,
    maximum_rows_per_view: int,
    alpha_minimum: float,
    minimum_observations: int,
    minimum_views: int,
    minimum_pose_bins: int,
    descriptor_trim_fraction: float,
) -> dict:
    """Cluster rendered surface observations without reciprocal Track matching.

    Geometry comes from the Gaussian-rendered depth at each sparse rendered-RGB
    keypoint.  Identity is a multi-view surface voxel, not a Gaussian primitive
    center and not a ray-triangulated Track.
    """

    if not (float(voxel_size_m) > 0):
        raise ValueError("surface-completion voxel size must be positive")
    if int(maximum_candidates) < 0 or int(maximum_rows_per_view) <= 0:
        raise ValueError("surface-completion capacities are invalid")
    if not (0 <= float(alpha_minimum) <= 1):
        raise ValueError("surface-completion alpha threshold must be in [0,1]")
    if min(int(minimum_observations), int(minimum_views), int(minimum_pose_bins)) <= 0:
        raise ValueError("surface-completion support minima must be positive")

    xyz_rows = []
    descriptor_rows = []
    detector_rows = []
    alpha_rows = []
    query_rows = []
    keypoint_rows = []
    pose_bin_rows = []
    for query_index in range(len(observations)):
        view = observations.build_view(query_index)
        if view.depth is None and view.keypoint_depth is None:
            raise ValueError("surface completion requires rendered depth")
        if view.alpha is None and view.keypoint_alpha is None:
            raise ValueError("surface completion requires rendered alpha")
        if view.valid_mask is None and view.keypoint_validity is None:
            raise ValueError("surface completion requires rendered validity")
        height, width = view.image_hw
        pixels = torch.floor(view.keypoints).long()
        x = pixels[:, 0].clamp(0, width - 1)
        y = pixels[:, 1].clamp(0, height - 1)
        depth = (
            view.keypoint_depth.float()
            if view.keypoint_depth is not None
            else view.depth[y, x].float()
        )
        alpha = (
            view.keypoint_alpha.float()
            if view.keypoint_alpha is not None
            else view.alpha[y, x].float()
        )
        keypoint_valid = (
            view.keypoint_validity.bool()
            if view.keypoint_validity is not None
            else view.valid_mask[y, x].bool()
        )
        valid = (
            keypoint_valid
            & torch.isfinite(depth)
            & (depth > 0)
            & torch.isfinite(alpha)
            & (alpha >= float(alpha_minimum))
            & torch.isfinite(view.detector_scores)
        )
        rows = torch.nonzero(valid, as_tuple=False).reshape(-1)
        if rows.numel() > int(maximum_rows_per_view):
            ranked = torch.argsort(
                view.detector_scores[rows].float(), descending=True, stable=True
            )
            rows = rows[ranked[: int(maximum_rows_per_view)]]
        if rows.numel() == 0:
            continue
        xyz_rows.append(
            _unproject_to_world(
                view.keypoints[rows],
                depth[rows],
                view.intrinsics.float(),
                view.pose_w2c.float(),
            )
        )
        descriptor_rows.append(F.normalize(view.descriptors[rows].float(), dim=1))
        detector_rows.append(view.detector_scores[rows].float().clamp_min(0))
        alpha_rows.append(alpha[rows].clamp(0, 1))
        query_rows.append(torch.full((rows.numel(),), query_index, dtype=torch.long))
        keypoint_rows.append(rows.long())
        pose_bin_rows.append(
            torch.full((rows.numel(),), int(view.pose_bin), dtype=torch.long)
        )
    if not xyz_rows:
        raise ValueError("rendered views contain no legal surface observations")
    xyz = torch.cat(xyz_rows)
    descriptors = torch.cat(descriptor_rows)
    detector = torch.cat(detector_rows)
    alpha = torch.cat(alpha_rows)
    query = torch.cat(query_rows)
    keypoint = torch.cat(keypoint_rows)
    pose_bin = torch.cat(pose_bin_rows)
    if bool((pose_bin < 0).any()):
        raise ValueError("surface completion requires explicit mapping pose bins")
    voxel = torch.floor(xyz / float(voxel_size_m)).long()
    unique_voxel, inverse = torch.unique(voxel, dim=0, sorted=True, return_inverse=True)
    group_count = int(unique_voxel.shape[0])
    observation_count = torch.bincount(inverse, minlength=group_count)
    view_pairs = torch.unique(torch.stack((inverse, query), dim=1), dim=0)
    view_count = torch.bincount(view_pairs[:, 0], minlength=group_count)
    bin_pairs = torch.unique(torch.stack((inverse, pose_bin), dim=1), dim=0)
    pose_bin_count = torch.bincount(bin_pairs[:, 0], minlength=group_count)
    weight = detector * alpha
    weight_sum = torch.zeros(group_count, dtype=torch.float32)
    weight_sum.index_add_(0, inverse, weight)
    eligible = (
        (observation_count >= int(minimum_observations))
        & (view_count >= int(minimum_views))
        & (pose_bin_count >= int(minimum_pose_bins))
        & (weight_sum > 0)
    )
    candidates = torch.nonzero(eligible, as_tuple=False).reshape(-1)
    # Stable lexicographic order: views, pose bins, observations, weight, voxel ID.
    order = candidates
    for values in (weight_sum, observation_count, pose_bin_count, view_count):
        order = order[torch.argsort(values[order], descending=True, stable=True)]
    selected_groups = order[: int(maximum_candidates)]
    selected_xyz = []
    selected_features = []
    selected_covariance = []
    selected_matchability = []
    selected_offsets = [0]
    selected_queries = []
    selected_keypoints = []
    for group in selected_groups.tolist():
        rows = torch.nonzero(inverse == int(group), as_tuple=False).reshape(-1)
        weights = weight[rows].clamp_min(1e-6)
        mean = (xyz[rows] * weights[:, None]).sum(0) / weights.sum()
        centered = xyz[rows] - mean
        covariance = (
            torch.einsum("n,ni,nj->ij", weights, centered, centered) / weights.sum()
        )
        feature = fuse_projective_anchor_observations(
            descriptors[rows],
            pose_bin[rows],
            detector_weight=detector[rows],
            visibility_weight=alpha[rows],
            trim_fraction=float(descriptor_trim_fraction),
        )
        selected_xyz.append(mean)
        selected_covariance.append(covariance)
        selected_features.append(feature)
        selected_matchability.append(
            float(view_count[group])
            / (float(view_count[group]) + 2.0)
            * float(detector[rows].mean())
        )
        selected_queries.append(query[rows])
        selected_keypoints.append(keypoint[rows])
        selected_offsets.append(selected_offsets[-1] + int(rows.numel()))
    count = int(selected_groups.numel())
    empty_features = torch.empty((0, descriptors.shape[1]), dtype=torch.float32)
    return {
        "schema": "lafgs_materialized_anchor_map",
        "version": 1,
        "anchor_ids": torch.arange(count, dtype=torch.long),
        "anchor_xyz": torch.stack(selected_xyz)
        if selected_xyz
        else torch.empty((0, 3)),
        "anchor_features": (
            torch.stack(selected_features) if selected_features else empty_features
        ),
        "source_primitive_ids": torch.full((count,), -1, dtype=torch.long),
        "gaussian_support_component_ids": selected_groups.long(),
        "gaussian_support_voxel_coordinates": unique_voxel[selected_groups].long(),
        "track_cluster_ids": torch.full((count,), -1, dtype=torch.long),
        "anchor_type": torch.zeros(count, dtype=torch.long),
        "dependency_group_ids": torch.arange(count, dtype=torch.long),
        "coarse_dependency_group_ids": torch.arange(count, dtype=torch.long),
        "fine_identity_ids": torch.arange(count, dtype=torch.long),
        "source_dependency_group_ids": selected_groups.long(),
        "anchor_position_covariance": (
            torch.stack(selected_covariance)
            if selected_covariance
            else torch.empty((0, 3, 3))
        ).float(),
        "anchor_matchability": torch.tensor(selected_matchability).float(),
        "base_anchor_count": count,
        "canonical_anchor_count": count,
        "micro_anchor_count": 0,
        "surface_completion_observations": {
            "observation_offsets": torch.tensor(selected_offsets, dtype=torch.long),
            "query_indices": (
                torch.cat(selected_queries)
                if selected_queries
                else torch.empty(0, dtype=torch.long)
            ),
            "keypoint_indices": (
                torch.cat(selected_keypoints)
                if selected_keypoints
                else torch.empty(0, dtype=torch.long)
            ),
        },
        "surface_completion": {
            "schema": "lafgs_gaussian_render_surface_completion",
            "version": 1,
            "identity": "multi_view_rendered_depth_surface_voxel",
            "geometry": "weighted_rendered_depth_unprojection_not_primitive_center",
            "descriptor": "gwff_style_rendered_rgb_observation_fusion",
            "uses_source_mapping_rgb": False,
            "uses_test_queries": False,
            "voxel_size_m": float(voxel_size_m),
            "maximum_rows_per_view": int(maximum_rows_per_view),
            "maximum_candidates": int(maximum_candidates),
            "minimum_observations": int(minimum_observations),
            "minimum_views": int(minimum_views),
            "minimum_pose_bins": int(minimum_pose_bins),
            "alpha_minimum": float(alpha_minimum),
            "descriptor_trim_fraction": float(descriptor_trim_fraction),
            "legal_observation_count": int(xyz.shape[0]),
            "eligible_surface_component_count": int(eligible.sum()),
            "selected_surface_component_count": count,
        },
    }


def surface_completion_selector_inputs(
    surface_map: dict,
    observations: ObservationProvider,
) -> tuple[dict, dict]:
    """Build exact mapping-only matching evidence for completion candidates."""

    count = int(torch.as_tensor(surface_map["anchor_ids"]).numel())
    completion = surface_map.get("surface_completion", {})
    evidence = surface_map.get("surface_completion_observations")
    if (
        completion.get("schema") != "lafgs_gaussian_render_surface_completion"
        or evidence is None
    ):
        raise ValueError("surface map lacks completion observation evidence")
    offsets = torch.as_tensor(evidence["observation_offsets"]).long()
    query = torch.as_tensor(evidence["query_indices"]).long()
    keypoint = torch.as_tensor(evidence["keypoint_indices"]).long()
    if offsets.shape != (count + 1,) or int(offsets[-1]) != query.numel():
        raise ValueError("surface completion observation CSR is inconsistent")
    if query.shape != keypoint.shape:
        raise ValueError("surface completion observation rows do not align")
    positives: list[dict[int, set[int]]] = [dict() for _ in range(len(observations))]
    observation_count = offsets[1:] - offsets[:-1]
    for anchor in range(count):
        start, end = int(offsets[anchor]), int(offsets[anchor + 1])
        for query_index, keypoint_index in zip(
            query[start:end].tolist(), keypoint[start:end].tolist()
        ):
            positives[int(query_index)].setdefault(int(keypoint_index), set()).add(
                anchor
            )
    records = []
    for query_index in range(len(observations)):
        view = observations.build_view(query_index)
        retained = (
            torch.ones(view.keypoints.shape[0], dtype=torch.bool)
            if view.keypoint_validity is None
            else view.keypoint_validity.bool()
        )
        query_rows = torch.nonzero(retained, as_tuple=False).reshape(-1)
        positive_values = []
        positive_offsets = [0]
        for row in query_rows.tolist():
            positive_values.extend(sorted(positives[query_index].get(int(row), ())))
            positive_offsets.append(len(positive_values))
        records.append(
            {
                "query_index": query_index,
                "query_name": view.image_name,
                "query_rows": query_rows,
                "positive_offsets": torch.tensor(positive_offsets, dtype=torch.long),
                "positive_indices": torch.tensor(positive_values, dtype=torch.long),
                "ambiguous_offsets": torch.zeros(
                    query_rows.numel() + 1, dtype=torch.long
                ),
                "ambiguous_indices": torch.empty(0, dtype=torch.long),
            }
        )
    legal = observation_count.long()
    zeros = torch.zeros(count, dtype=torch.long)
    graph = {
        "schema": "lafgs_gaussian_render_surface_completion_function_graph",
        "version": 1,
        "uses_source_mapping_rgb": False,
        "uses_test_queries": False,
        "provenance_opportunity_count": legal.clone(),
        "provenance_legal_hit_strong_count": legal.clone(),
        "provenance_legal_hit_clean_count": legal.clone(),
        "provenance_solver_inlier_gtclean_strong_count": zeros.clone(),
        "provenance_harmful_solver_inlier_count": zeros.clone(),
        "records": [
            {
                "query_index": record["query_index"],
                "query_rows": record["query_rows"].clone(),
            }
            for record in records
        ],
        "evidence_semantics": (
            "exact_multiview_surface_component_observation_support_without_pose_feedback"
        ),
    }
    teacher = {
        "schema": "lafgs_v9_active_map_complete_positive_teacher",
        "version": 1,
        "anchor_count": count,
        "query_names": list(observations.names),
        "records": records,
        "uses_source_mapping_rgb": False,
        "uses_test_queries": False,
        "identity_positive_policy": "same_gaussian_render_surface_component",
    }
    return teacher, graph
