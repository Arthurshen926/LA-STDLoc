"""Pure-ray geometry and view-balanced descriptors for V6 Projective Anchors."""

from __future__ import annotations

import torch

from common.v6_contracts import ANCHOR_CANDIDATE_SCHEMA, ASSOCIATION_GRAPH_SCHEMA
from evidence.observation_provider import ObservationProvider
from evidence.tracks import fuse_projective_anchor_observations
from evidence.triangulation import robust_triangulate_associations


def _gather(values: list[torch.Tensor], query: torch.Tensor, row: torch.Tensor) -> torch.Tensor:
    counts = torch.tensor([item.shape[0] for item in values], dtype=torch.long)
    offsets = torch.cat((counts.new_zeros(1), counts.cumsum(0)))
    return torch.cat(values, dim=0)[offsets[query] + row]


@torch.no_grad()
def reconstruct_projective_anchors(
    observations: ObservationProvider,
    association: dict,
    *,
    maximum_observations: int = 32,
    minimum_views: int = 3,
    minimum_view_bins: int = 2,
    huber_delta_px: float = 2.0,
    triangulation_iterations: int = 3,
    minimum_parallax_deg: float = 1.0,
    parallax_quantile: float = 0.75,
    maximum_reprojection_px: float = 2.0,
    maximum_condition_number: float = 1e6,
    descriptor_trim_fraction: float = 0.2,
) -> dict:
    if association.get("schema") != ASSOCIATION_GRAPH_SCHEMA:
        raise ValueError("V6 reconstruction requires projective association v2")
    if association.get("uses_source_mapping_rgb") is not False or association.get(
        "uses_test_queries"
    ) is not False:
        raise ValueError("association is outside the mapping-only scope")
    if list(observations.names) != list(association["query_names"]):
        raise ValueError("association and observation registries differ")
    tracks = association["tracks"]
    track_index = torch.as_tensor(tracks["track_index"]).long()
    query_index = torch.as_tensor(tracks["query_index"]).long()
    keypoint_index = torch.as_tensor(tracks["keypoint_index"]).long()
    confidence = torch.as_tensor(tracks["confidence"]).float()
    track_count = int(association["diagnostics"]["track_count"])
    views = [observations.build_view(index) for index in range(len(observations))]
    keypoints = [view.keypoints.float() for view in views]
    descriptors = [view.descriptors.float() for view in views]
    detector_scores = [view.detector_scores.float() for view in views]
    uv = _gather(keypoints, query_index, keypoint_index)
    geometry = robust_triangulate_associations(
        landmark_count=track_count,
        landmark_index=track_index,
        query_index=query_index,
        uv=uv,
        confidence=confidence,
        camera_K=torch.stack([view.intrinsics.float() for view in views]),
        pose_w2c=torch.stack([view.pose_w2c.float() for view in views]),
        query_bin=torch.as_tensor(association["query_bins"]).long(),
        rendered_depth=None,
        maximum_observations_per_landmark=int(maximum_observations),
        minimum_views=int(minimum_views),
        minimum_view_bins=int(minimum_view_bins),
        huber_delta_px=float(huber_delta_px),
        iterations=int(triangulation_iterations),
        minimum_parallax_deg=float(minimum_parallax_deg),
        parallax_quantile=float(parallax_quantile),
        maximum_reprojection_px=float(maximum_reprojection_px),
        maximum_condition_number=float(maximum_condition_number),
        maximum_covariance_trace_m2=float("inf"),
        maximum_rendered_depth_residual_m=float("inf"),
        minimum_rendered_depth_observations=0,
        surface_support_enabled=False,
    )
    eligible = torch.as_tensor(geometry["triangulated"]).bool()
    selected = torch.nonzero(eligible, as_tuple=False).reshape(-1)
    if selected.numel() == 0:
        raise ValueError("association graph contains no ray-triangulated Anchor")
    selected_lookup = torch.full((track_count,), -1, dtype=torch.long)
    selected_lookup[selected] = torch.arange(selected.numel())
    selected_observation = selected_lookup[track_index] >= 0
    selected_track = selected_lookup[track_index[selected_observation]]
    selected_query = query_index[selected_observation]
    selected_keypoint = keypoint_index[selected_observation]
    selected_confidence = confidence[selected_observation]
    selected_descriptors = _gather(
        descriptors, selected_query, selected_keypoint
    )
    selected_detector = _gather(
        [value[:, None] for value in detector_scores], selected_query, selected_keypoint
    ).reshape(-1)
    features = []
    offsets = [0]
    csr_query = []
    csr_keypoint = []
    for anchor in range(int(selected.numel())):
        rows = torch.nonzero(selected_track == anchor, as_tuple=False).reshape(-1)
        features.append(
            fuse_projective_anchor_observations(
                selected_descriptors[rows],
                torch.as_tensor(association["query_bins"])[selected_query[rows]],
                detector_weight=selected_detector[rows] * selected_confidence[rows],
                trim_fraction=float(descriptor_trim_fraction),
            )
        )
        csr_query.append(selected_query[rows])
        csr_keypoint.append(selected_keypoint[rows])
        offsets.append(offsets[-1] + int(rows.numel()))
    geometry_reliability = (
        torch.exp(
            -torch.as_tensor(geometry["triangulation_reprojection_median_px"])[selected]
            / max(float(maximum_reprojection_px), 1e-6)
        )
        * (
            torch.as_tensor(geometry["triangulation_parallax_deg"])[selected]
            / max(float(minimum_parallax_deg), 1e-6)
        ).clamp(max=1.0)
    )
    identity_reliability = torch.as_tensor(
        association["component_statistics"]["identity_reliability"]
    )[selected]
    count = int(selected.numel())
    return {
        "schema": ANCHOR_CANDIDATE_SCHEMA,
        "version": 2,
        "uses_source_mapping_rgb": False,
        "uses_test_queries": False,
        "query_names": list(association["query_names"]),
        "query_bins": torch.as_tensor(association["query_bins"]).long(),
        "anchor_ids": torch.arange(count, dtype=torch.long),
        "source_component_ids": selected,
        "anchor_xyz": torch.as_tensor(geometry["triangulated_xyz"])[selected].float(),
        "anchor_features": torch.stack(features).float(),
        "anchor_position_covariance": torch.as_tensor(
            geometry["triangulation_covariance_matrix"]
        )[selected].float(),
        "identity_reliability": identity_reliability.float(),
        "geometry_reliability": geometry_reliability.float(),
        "projective_anchor_observations": {
            "observation_offsets": torch.tensor(offsets, dtype=torch.long),
            "query_indices": torch.cat(csr_query),
            "keypoint_indices": torch.cat(csr_keypoint),
        },
        "geometry_diagnostics": {
            key: torch.as_tensor(value)[selected].clone()
            for key, value in geometry.items()
            if isinstance(value, torch.Tensor) and value.ndim >= 1 and value.shape[0] == track_count
        },
        "contract": {
            "final_xyz_source": "fixed_camera_robust_ray_triangulation",
            "gaussian_depth_used_for_final_xyz": False,
            "gaussian_primitive_center_used": False,
            "one_observation_per_camera": True,
            "continuous_identity_and_geometry_reliability": True,
        },
    }
