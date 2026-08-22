"""One-pass evidence-gated multiview association for V6 Projective Anchors."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from common.v6_contracts import ASSOCIATION_GRAPH_SCHEMA
from evidence.observation_provider import ObservationProvider
from evidence.triangulation import build_cycle_consistent_tracks, camera_pose_bins
from features.raster_sampling import sample_raster_at_grid_uv


def _render_valid_rows(view) -> torch.Tensor:
    validity = view.keypoint_validity
    if validity is None:
        if view.valid_mask is None:
            raise ValueError("V6 association requires render-valid observations")
        validity = sample_raster_at_grid_uv(view.valid_mask, view.keypoints).bool()
    validity = torch.as_tensor(validity).bool().reshape(-1)
    if validity.numel() != view.keypoints.shape[0]:
        raise ValueError("render-valid rows do not align with keypoints")
    if not bool(validity.all()):
        raise ValueError("observation cache contains rows outside the rendered domain")
    return validity


def _component_statistics(
    tracks: dict[str, torch.Tensor],
    descriptors: list[torch.Tensor],
    query_bins: torch.Tensor,
    track_count: int,
) -> dict[str, torch.Tensor]:
    track_index = torch.as_tensor(tracks["track_index"]).long()
    query_index = torch.as_tensor(tracks["query_index"]).long()
    keypoint_index = torch.as_tensor(tracks["keypoint_index"]).long()
    confidence = torch.as_tensor(tracks["confidence"]).float().clamp(0, 1)
    level = torch.as_tensor(tracks["track_level"]).long()
    packed = torch.cat([F.normalize(value.float(), dim=1) for value in descriptors])
    offsets = torch.tensor(
        [0, *torch.tensor([len(value) for value in descriptors]).cumsum(0).tolist()],
        dtype=torch.long,
    )
    observation_descriptor = packed[offsets[query_index] + keypoint_index]
    mean_confidence = torch.zeros(track_count)
    mean_confidence.index_add_(0, track_index, confidence)
    count = torch.bincount(track_index, minlength=track_count).clamp_min(1)
    mean_confidence /= count
    descriptor_consistency = torch.zeros(track_count)
    distinct_families = torch.zeros(track_count, dtype=torch.long)
    for track in range(track_count):
        rows = torch.nonzero(track_index == track, as_tuple=False).reshape(-1)
        prototype = F.normalize(observation_descriptor[rows].mean(0), dim=0)
        descriptor_consistency[track] = (
            observation_descriptor[rows] @ prototype
        ).mean().clamp(0, 1)
        distinct_families[track] = torch.unique(query_bins[query_index[rows]]).numel()
    family_support = (distinct_families.float() / 3.0).clamp(max=1.0)
    cycle_factor = torch.where(level >= 2, 1.0, 0.8)
    reliability = (
        mean_confidence.clamp_min(1e-8)
        * descriptor_consistency.clamp_min(1e-8)
        * family_support.clamp_min(1e-8)
    ).pow(1.0 / 3.0) * cycle_factor
    return {
        "identity_reliability": reliability.clamp(0, 1),
        "mean_edge_confidence": mean_confidence,
        "descriptor_consistency": descriptor_consistency,
        "distinct_camera_family_count": distinct_families,
        "cycle_supported": level >= 2,
    }


@torch.no_grad()
def build_projective_association_graph(
    observations: ObservationProvider,
    *,
    pair_neighbors: int = 6,
    minimum_baseline_m: float = 0.03,
    maximum_baseline_m: float = 5.0,
    maximum_axis_angle_deg: float = 75.0,
    minimum_similarity: float = 0.65,
    minimum_margin: float = 0.01,
    maximum_epipolar_error_px: float = 2.0,
    epipolar_candidate_topk: int = 4,
    minimum_track_views: int = 3,
    view_bins: int = 8,
    view_direction_weight: float = 0.5,
    device: str = "cuda",
) -> dict:
    """Build one graph; no post-hoc repair, parent/child split, or child cap."""

    views = [observations.build_view(index) for index in range(len(observations))]
    for view in views:
        _render_valid_rows(view)
    descriptors = [view.descriptors.float() for view in views]
    keypoints = [view.keypoints.float() for view in views]
    detector_scores = [view.detector_scores.float() for view in views]
    camera_K = torch.stack([view.intrinsics.float() for view in views])
    pose_w2c = torch.stack([view.pose_w2c.float() for view in views])
    image_hw = torch.tensor([view.image_hw for view in views], dtype=torch.long)
    query_bins = camera_pose_bins(
        pose_w2c, int(view_bins), direction_weight=float(view_direction_weight)
    )
    tracks, diagnostics, sidecar = build_cycle_consistent_tracks(
        descriptors=descriptors,
        keypoints=keypoints,
        detector_scores=detector_scores,
        camera_K=camera_K,
        pose_w2c=pose_w2c,
        pair_neighbors=int(pair_neighbors),
        pair_policy="nearest",
        pair_image_hw=image_hw,
        minimum_baseline_m=float(minimum_baseline_m),
        maximum_baseline_m=float(maximum_baseline_m),
        maximum_axis_angle_deg=float(maximum_axis_angle_deg),
        minimum_similarity=float(minimum_similarity),
        minimum_margin=float(minimum_margin),
        maximum_epipolar_error_px=float(maximum_epipolar_error_px),
        epipolar_candidate_topk=int(epipolar_candidate_topk),
        minimum_track_views=int(minimum_track_views),
        require_cycle=True,
        allow_chain_tracks=True,
        return_pair_sidecar=True,
        device=device,
    )
    track_count = int(diagnostics["track_count"])
    statistics = _component_statistics(
        tracks, descriptors, query_bins, track_count
    )
    return {
        "schema": ASSOCIATION_GRAPH_SCHEMA,
        "version": 2,
        "uses_source_mapping_rgb": False,
        "uses_test_queries": False,
        "query_names": list(observations.names),
        "query_bins": query_bins,
        "tracks": tracks,
        "pair_sidecar": sidecar,
        "diagnostics": diagnostics,
        "component_statistics": statistics,
        "contract": {
            "render_valid_observations_required": True,
            "reciprocal_descriptor": True,
            "known_pose_epipolar": True,
            "one_observation_per_camera_per_component": True,
            "posthoc_support_repair": False,
            "parent_child_semantics": False,
            "child_cap": False,
            "cycle_chain_are_confidence_attributes_not_candidate_types": True,
        },
    }
