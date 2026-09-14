"""Descriptor-independent Gaussian-provenance correspondence supervision.

The feedback renderer and the Projective Anchor map share a frozen Gaussian
prior.  This module uses that privileged lineage to build a truth graph which
is deliberately independent of descriptor retrieval.  Descriptor Top-L rows
belong to the competition graph and must never be passed to these routines.

The implementation keeps three pieces separate:

* mapping-observation composition is aggregated into an Anchor signature;
* primitive-to-Anchor inversion enumerates full-map candidates;
* observation transport decides whether the lineage evidence identifies one
  Track/Anchor, an equivalent set, an ambiguous set, or no map support.

No leave-one-out operation is provided.  Calibration must use disjoint mapping
view families and a separately materialized validation split.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch


TRUTH_NONE = 0
TRUTH_UNIQUE = 1
TRUTH_EQUIVALENT = 2
TRUTH_AMBIGUOUS = 3
TRUTH_INVALID = 4

TRUTH_STATUS_NAMES = (
    "NONE",
    "UNIQUE",
    "EQUIVALENT",
    "AMBIGUOUS",
    "INVALID",
)


def _validate_offsets(offsets: torch.Tensor, edge_count: int, row_count: int) -> None:
    offsets = torch.as_tensor(offsets).long().reshape(-1)
    if offsets.shape != (int(row_count) + 1,):
        raise ValueError("CSR offsets do not align with rows")
    if int(offsets[0]) != 0 or int(offsets[-1]) != int(edge_count):
        raise ValueError("CSR offsets do not cover all edges")
    if bool((offsets[1:] < offsets[:-1]).any()):
        raise ValueError("CSR offsets must be monotonic")


def _segment_sum(values: torch.Tensor, rows: torch.Tensor, count: int) -> torch.Tensor:
    output = values.new_zeros((int(count),))
    if values.numel():
        output.scatter_add_(0, rows, values)
    return output


@dataclass(frozen=True)
class TruthAssignmentThresholds:
    """Precision-first operating point for provenance truth assignment."""

    minimum_provenance_overlap: float = 0.20
    minimum_transport_view_families: int = 2
    maximum_transport_median_residual_px: float = 4.0
    minimum_assignment_confidence: float = 0.15
    minimum_top1_top2_margin: float = 0.03
    minimum_top1_top2_ratio: float = 1.15
    maximum_composition_entropy: float = float("inf")
    maximum_relative_depth_spread: float = float("inf")
    minimum_retained_composition_fraction: float = 0.95
    maximum_query_reprojection_px: float = float("inf")
    maximum_query_normalized_depth_residual: float = float("inf")
    maximum_query_projection_std_px: float = float("inf")

    def validate(self) -> None:
        if not 0.0 <= float(self.minimum_provenance_overlap) <= 1.0:
            raise ValueError("minimum provenance overlap must lie in [0, 1]")
        if int(self.minimum_transport_view_families) < 1:
            raise ValueError("transport requires at least one view family")
        if float(self.maximum_transport_median_residual_px) <= 0.0:
            raise ValueError("maximum transport residual must be positive")
        if not 0.0 <= float(self.minimum_assignment_confidence) <= 1.0:
            raise ValueError("minimum assignment confidence must lie in [0, 1]")
        if float(self.minimum_top1_top2_margin) < 0.0:
            raise ValueError("assignment margin must be non-negative")
        if float(self.minimum_top1_top2_ratio) < 1.0:
            raise ValueError("assignment ratio must be at least one")
        if float(self.maximum_composition_entropy) < 0.0:
            raise ValueError("maximum composition entropy must be non-negative")
        if float(self.maximum_relative_depth_spread) < 0.0:
            raise ValueError("maximum relative depth spread must be non-negative")
        if not 0.0 <= float(self.minimum_retained_composition_fraction) <= 1.0:
            raise ValueError("retained composition fraction must lie in [0, 1]")
        if float(self.maximum_query_reprojection_px) <= 0.0:
            raise ValueError("maximum Query reprojection residual must be positive")
        if float(self.maximum_query_normalized_depth_residual) <= 0.0:
            raise ValueError("maximum normalized Query depth residual must be positive")
        if float(self.maximum_query_projection_std_px) <= 0.0:
            raise ValueError("maximum Query projection standard deviation must be positive")


def aggregate_anchor_provenance(
    *,
    observation_offsets: torch.Tensor,
    observation_primitive_ids: torch.Tensor,
    observation_weights: torch.Tensor,
    observation_view_family_ids: torch.Tensor,
    observation_valid: torch.Tensor | None = None,
    minimum_edge_weight: float = 0.0,
) -> dict[str, torch.Tensor | str | int]:
    """Aggregate per-observation Gaussian composition into Anchor signatures.

    Every view family receives equal total weight within an Anchor.  Repeated
    observations from one family therefore cannot dominate the signature.
    Duplicate ``(Anchor, primitive)`` edges are summed and the final signature
    is normalized to unit mass.
    """

    offsets = torch.as_tensor(observation_offsets).long().cpu().reshape(-1)
    primitive_ids = torch.as_tensor(observation_primitive_ids).long().cpu()
    weights = torch.as_tensor(observation_weights).float().cpu()
    families = torch.as_tensor(observation_view_family_ids).long().cpu().reshape(-1)
    if primitive_ids.ndim != 2 or primitive_ids.shape != weights.shape:
        raise ValueError("observation primitive IDs and weights must be [E, K]")
    edge_count = int(primitive_ids.shape[0])
    anchor_count = int(offsets.numel() - 1)
    _validate_offsets(offsets, edge_count, anchor_count)
    if families.numel() != edge_count:
        raise ValueError("observation view families do not align with observations")
    valid_observation = (
        torch.ones(edge_count, dtype=torch.bool)
        if observation_valid is None
        else torch.as_tensor(observation_valid).bool().cpu().reshape(-1)
    )
    if valid_observation.numel() != edge_count:
        raise ValueError("observation validity does not align with observations")
    if edge_count and int(families.min()) < 0:
        raise ValueError("view family IDs must be non-negative")

    observation_anchor = torch.repeat_interleave(
        torch.arange(anchor_count, dtype=torch.long), offsets[1:] - offsets[:-1]
    )
    pair_base = int(families.max()) + 1 if families.numel() else 1
    family_pair = observation_anchor * pair_base + families
    _, family_inverse, observations_per_family = torch.unique(
        family_pair, sorted=True, return_inverse=True, return_counts=True
    )
    family_balance = observations_per_family[family_inverse].float().reciprocal()

    expanded_anchor = observation_anchor[:, None].expand_as(primitive_ids)
    balanced_weights = weights * family_balance[:, None]
    valid = (
        valid_observation[:, None]
        & (primitive_ids >= 0)
        & torch.isfinite(balanced_weights)
        & (balanced_weights > float(minimum_edge_weight))
    )
    if not bool(valid.any()):
        return {
            "schema": "lafgs_anchor_gaussian_provenance",
            "version": 1,
            "anchor_count": anchor_count,
            "anchor_provenance_offsets": torch.zeros(anchor_count + 1, dtype=torch.long),
            "anchor_provenance_primitive_ids": torch.empty(0, dtype=torch.long),
            "anchor_provenance_weights": torch.empty(0, dtype=torch.float32),
            "anchor_provenance_entropy": torch.zeros(anchor_count),
            "anchor_provenance_view_family_count": torch.zeros(anchor_count, dtype=torch.long),
        }

    flat_anchor = expanded_anchor[valid]
    flat_primitive = primitive_ids[valid]
    flat_weight = balanced_weights[valid]
    primitive_base = int(flat_primitive.max()) + 1
    encoded = flat_anchor * primitive_base + flat_primitive
    order = torch.argsort(encoded, stable=True)
    encoded = encoded[order]
    flat_weight = flat_weight[order]
    unique, inverse = torch.unique_consecutive(encoded, return_inverse=True)
    merged_weight = flat_weight.new_zeros((unique.numel(),))
    merged_weight.scatter_add_(0, inverse, flat_weight)
    merged_anchor = torch.div(unique, primitive_base, rounding_mode="floor")
    merged_primitive = unique.remainder(primitive_base)
    total = _segment_sum(merged_weight, merged_anchor, anchor_count)
    normalized = merged_weight / total[merged_anchor].clamp_min(1e-12)
    counts = torch.bincount(merged_anchor, minlength=anchor_count)
    output_offsets = torch.zeros(anchor_count + 1, dtype=torch.long)
    output_offsets[1:] = counts.cumsum(0)
    entropy_terms = -(normalized * normalized.clamp_min(1e-12).log())
    entropy = _segment_sum(entropy_terms, merged_anchor, anchor_count)
    family_anchor = torch.div(
        torch.unique(family_pair[valid_observation], sorted=True),
        pair_base,
        rounding_mode="floor",
    )
    family_count = torch.bincount(family_anchor, minlength=anchor_count)
    return {
        "schema": "lafgs_anchor_gaussian_provenance",
        "version": 1,
        "anchor_count": anchor_count,
        "anchor_provenance_offsets": output_offsets,
        "anchor_provenance_primitive_ids": merged_primitive,
        "anchor_provenance_weights": normalized,
        "anchor_provenance_entropy": entropy,
        "anchor_provenance_view_family_count": family_count,
    }


def build_primitive_anchor_index(anchor_provenance: Mapping) -> dict[str, torch.Tensor | str | int]:
    """Invert an Anchor provenance CSR without using descriptor scores."""

    offsets = torch.as_tensor(anchor_provenance["anchor_provenance_offsets"]).long().cpu()
    primitive = torch.as_tensor(
        anchor_provenance["anchor_provenance_primitive_ids"]
    ).long().cpu()
    weight = torch.as_tensor(anchor_provenance["anchor_provenance_weights"]).float().cpu()
    anchor_count = int(anchor_provenance["anchor_count"])
    _validate_offsets(offsets, primitive.numel(), anchor_count)
    if weight.shape != primitive.shape:
        raise ValueError("Anchor provenance weights do not align with primitive IDs")
    anchors = torch.repeat_interleave(
        torch.arange(anchor_count, dtype=torch.long), offsets[1:] - offsets[:-1]
    )
    order = torch.argsort(primitive, stable=True)
    primitive = primitive[order]
    anchors = anchors[order]
    weight = weight[order]
    unique, counts = torch.unique_consecutive(primitive, return_counts=True)
    inverse_offsets = torch.zeros(unique.numel() + 1, dtype=torch.long)
    inverse_offsets[1:] = counts.cumsum(0)
    return {
        "schema": "lafgs_primitive_anchor_inverted_index",
        "version": 1,
        "anchor_count": anchor_count,
        "primitive_ids": unique,
        "primitive_offsets": inverse_offsets,
        "anchor_rows": anchors,
        "anchor_responsibilities": weight,
    }


def provenance_candidate_graph(
    *,
    query_primitive_ids: torch.Tensor,
    query_weights: torch.Tensor,
    primitive_anchor_index: Mapping,
    query_valid: torch.Tensor | None = None,
    query_composition_entropy: torch.Tensor | None = None,
    query_relative_depth_spread: torch.Tensor | None = None,
    query_retained_composition_fraction: torch.Tensor | None = None,
    maximum_candidates_per_row: int = 256,
) -> dict[str, torch.Tensor | str | int]:
    """Build full-map candidates from shared Gaussian composition only."""

    query_ids = torch.as_tensor(query_primitive_ids).long().cpu()
    query_mass = torch.as_tensor(query_weights).float().cpu()
    if query_ids.ndim != 2 or query_ids.shape != query_mass.shape:
        raise ValueError("query primitive IDs and weights must be [R, K]")
    row_count = int(query_ids.shape[0])
    valid_rows = (
        torch.ones(row_count, dtype=torch.bool)
        if query_valid is None
        else torch.as_tensor(query_valid).bool().cpu().reshape(-1)
    )
    if valid_rows.numel() != row_count:
        raise ValueError("query provenance validity does not align with rows")
    composition_entropy = (
        torch.zeros(row_count)
        if query_composition_entropy is None
        else torch.as_tensor(query_composition_entropy).float().cpu().reshape(-1)
    )
    depth_spread = (
        torch.zeros(row_count)
        if query_relative_depth_spread is None
        else torch.as_tensor(query_relative_depth_spread).float().cpu().reshape(-1)
    )
    retained_fraction = (
        torch.ones(row_count)
        if query_retained_composition_fraction is None
        else torch.as_tensor(query_retained_composition_fraction)
        .float()
        .cpu()
        .reshape(-1)
    )
    if (
        composition_entropy.numel() != row_count
        or depth_spread.numel() != row_count
        or retained_fraction.numel() != row_count
    ):
        raise ValueError("query provenance diagnostics do not align with rows")
    maximum = int(maximum_candidates_per_row)
    if maximum < 0:
        raise ValueError("maximum candidates per row must be non-negative")
    universe = torch.as_tensor(primitive_anchor_index["primitive_ids"]).long().cpu()
    offsets = torch.as_tensor(primitive_anchor_index["primitive_offsets"]).long().cpu()
    anchors = torch.as_tensor(primitive_anchor_index["anchor_rows"]).long().cpu()
    responsibility = torch.as_tensor(
        primitive_anchor_index["anchor_responsibilities"]
    ).float().cpu()
    _validate_offsets(offsets, anchors.numel(), universe.numel())

    output_offsets = [0]
    output_anchor: list[int] = []
    output_bhattacharyya: list[float] = []
    output_intersection: list[float] = []
    for row in range(row_count):
        if not bool(valid_rows[row]):
            output_offsets.append(len(output_anchor))
            continue
        scores: dict[int, list[float]] = defaultdict(lambda: [0.0, 0.0])
        for primitive, mass in zip(query_ids[row].tolist(), query_mass[row].tolist()):
            if primitive < 0 or not torch.isfinite(torch.tensor(mass)) or mass <= 0.0:
                continue
            position = int(torch.searchsorted(universe, int(primitive)))
            if position >= universe.numel() or int(universe[position]) != int(primitive):
                continue
            start, stop = int(offsets[position]), int(offsets[position + 1])
            for anchor, anchor_mass in zip(
                anchors[start:stop].tolist(), responsibility[start:stop].tolist()
            ):
                item = scores[int(anchor)]
                item[0] += float(max(mass * anchor_mass, 0.0) ** 0.5)
                item[1] += float(min(mass, anchor_mass))
        ordered = sorted(
            scores.items(),
            key=lambda item: (-item[1][0], -item[1][1], item[0]),
        )
        if maximum > 0:
            ordered = ordered[:maximum]
        output_anchor.extend(anchor for anchor, _ in ordered)
        output_bhattacharyya.extend(value[0] for _, value in ordered)
        output_intersection.extend(value[1] for _, value in ordered)
        output_offsets.append(len(output_anchor))
    return {
        "schema": "lafgs_descriptor_independent_provenance_candidate_graph",
        "version": 1,
        "row_count": row_count,
        "anchor_count": int(primitive_anchor_index["anchor_count"]),
        "candidate_offsets": torch.tensor(output_offsets, dtype=torch.long),
        "candidate_anchor_rows": torch.tensor(output_anchor, dtype=torch.long),
        "bhattacharyya_overlap": torch.tensor(output_bhattacharyya),
        "intersection_overlap": torch.tensor(output_intersection),
        "query_valid": valid_rows,
        "query_composition_entropy": composition_entropy,
        "query_relative_depth_spread": depth_spread,
        "query_retained_composition_fraction": retained_fraction,
        "uses_descriptor_scores": False,
        "maximum_candidates_per_row": maximum,
        "candidate_enumeration": (
            "complete_primitive_inverted_index" if maximum == 0 else "bounded_overlap_rank"
        ),
    }


def backproject_query_surface(
    keypoints: torch.Tensor,
    depth: torch.Tensor,
    intrinsic: torch.Tensor,
    pose_w2c: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Backproject rendered keypoints to the Gaussian surface in world space."""

    xy = torch.as_tensor(keypoints).double().reshape(-1, 2)
    z = torch.as_tensor(depth).double().reshape(-1)
    calibration = torch.as_tensor(intrinsic).double().reshape(3, 3)
    pose = torch.as_tensor(pose_w2c).double().reshape(4, 4)
    if xy.shape[0] != z.numel():
        raise ValueError("query keypoints and depths do not align")
    valid = torch.isfinite(xy).all(1) & torch.isfinite(z) & (z > 0)
    homogeneous = torch.cat([xy, torch.ones_like(xy[:, :1])], dim=1)
    camera = (torch.linalg.inv(calibration) @ homogeneous.T).T * z[:, None]
    world = (camera - pose[:3, 3]) @ pose[:3, :3]
    world[~valid] = float("nan")
    return world.float(), valid


def query_anchor_geometry_evidence(
    *,
    candidate_graph: Mapping,
    query_keypoints: torch.Tensor,
    query_depth: torch.Tensor,
    query_indices: torch.Tensor,
    anchor_xyz: torch.Tensor,
    anchor_covariance: torch.Tensor,
    query_intrinsics: torch.Tensor,
    query_poses_w2c: torch.Tensor,
    strict_depth_absolute_m: float = 0.25,
    strict_depth_relative: float = 0.05,
    device: str | torch.device = "cpu",
    edge_chunk_size: int = 262144,
) -> dict[str, torch.Tensor | str | int | float]:
    """Measure candidate Anchor consistency in the Query camera.

    Gaussian lineage proposes full-map candidates and observation transport
    recovers Track support.  This independent geometric term prevents nearby
    Tracks on the same Gaussian surface from becoming interchangeable unless
    their triangulated Anchor is also consistent with the Query keypoint and
    rendered depth.  It never consults descriptors or Top-L retrieval.
    """

    offsets = torch.as_tensor(candidate_graph["candidate_offsets"]).long().cpu()
    anchors = torch.as_tensor(candidate_graph["candidate_anchor_rows"]).long().cpu()
    row_count = int(candidate_graph["row_count"])
    _validate_offsets(offsets, anchors.numel(), row_count)
    xy = torch.as_tensor(query_keypoints).float().cpu().reshape(-1, 2)
    depth = torch.as_tensor(query_depth).float().cpu().reshape(-1)
    queries = torch.as_tensor(query_indices).long().cpu().reshape(-1)
    if xy.shape[0] != row_count or depth.numel() != row_count or queries.numel() != row_count:
        raise ValueError("Query geometry rows do not align with candidate rows")
    xyz = torch.as_tensor(anchor_xyz).float().cpu()
    covariance = torch.as_tensor(anchor_covariance).float().cpu()
    anchor_count = int(xyz.shape[0])
    if covariance.shape != (anchor_count, 3, 3):
        raise ValueError("Anchor covariance does not align with Anchor geometry")
    if anchors.numel() and (int(anchors.min()) < 0 or int(anchors.max()) >= anchor_count):
        raise ValueError("candidate Anchor is outside the geometry registry")
    intrinsics = torch.as_tensor(query_intrinsics).float().cpu()
    poses = torch.as_tensor(query_poses_w2c).float().cpu()
    if intrinsics.ndim != 3 or intrinsics.shape[1:] != (3, 3):
        raise ValueError("Query intrinsics must be [Q, 3, 3]")
    if poses.shape != (intrinsics.shape[0], 4, 4):
        raise ValueError("Query poses do not align with Query intrinsics")
    if queries.numel() and (int(queries.min()) < 0 or int(queries.max()) >= poses.shape[0]):
        raise ValueError("Query index is outside the Query camera registry")
    absolute = float(strict_depth_absolute_m)
    relative = float(strict_depth_relative)
    if absolute <= 0.0 or relative <= 0.0:
        raise ValueError("Query depth tolerances must be positive")

    edge_rows = torch.repeat_interleave(
        torch.arange(row_count), offsets[1:] - offsets[:-1]
    )
    reprojection = torch.full((anchors.numel(),), float("inf"))
    normalized_depth = torch.full_like(reprojection, float("inf"))
    projection_std = torch.full_like(reprojection, float("inf"))
    positive_depth = torch.zeros(anchors.numel(), dtype=torch.bool)
    compute_device = torch.device(device)
    chunk_size = max(int(edge_chunk_size), 1)
    for start in range(0, anchors.numel(), chunk_size):
        stop = min(start + chunk_size, anchors.numel())
        rows = edge_rows[start:stop].to(compute_device)
        local_anchors = anchors[start:stop].to(compute_device)
        local_queries = queries[rows.cpu()].to(compute_device)
        local_xy = xy[rows.cpu()].to(compute_device)
        local_depth = depth[rows.cpu()].to(compute_device)
        local_xyz = xyz[local_anchors.cpu()].to(compute_device)
        local_covariance = covariance[local_anchors.cpu()].to(compute_device)
        calibration = intrinsics[local_queries.cpu()].to(compute_device)
        pose = poses[local_queries.cpu()].to(compute_device)
        rotation = pose[:, :3, :3]
        camera = torch.einsum("nij,nj->ni", rotation, local_xyz) + pose[:, :3, 3]
        projected = torch.einsum("nij,nj->ni", calibration, camera)
        uv = projected[:, :2] / projected[:, 2:].clamp_min(1e-8)
        local_reprojection = torch.linalg.norm(uv - local_xy, dim=1)
        camera_covariance = rotation @ local_covariance @ rotation.transpose(-1, -2)
        x, y, z = camera.unbind(1)
        dproj = camera.new_zeros((camera.shape[0], 2, 3))
        dproj[:, 0, 0] = calibration[:, 0, 0] / z.clamp_min(1e-8)
        dproj[:, 0, 2] = (
            -calibration[:, 0, 0] * x / z.square().clamp_min(1e-8)
        )
        dproj[:, 1, 1] = calibration[:, 1, 1] / z.clamp_min(1e-8)
        dproj[:, 1, 2] = (
            -calibration[:, 1, 1] * y / z.square().clamp_min(1e-8)
        )
        pixel_covariance = dproj @ camera_covariance @ dproj.transpose(-1, -2)
        local_projection_std = (
            pixel_covariance.diagonal(dim1=-2, dim2=-1)
            .sum(1)
            .clamp_min(0.0)
            .sqrt()
        )
        depth_std = camera_covariance[:, 2, 2].clamp_min(0.0).sqrt()
        tolerance = torch.maximum(
            torch.full_like(local_depth, absolute), local_depth.abs() * relative
        )
        local_normalized_depth = (local_depth - z).abs() / (
            tolerance + 2.0 * depth_std
        ).clamp_min(1e-8)
        local_positive = (
            torch.isfinite(camera).all(1)
            & torch.isfinite(local_reprojection)
            & torch.isfinite(local_normalized_depth)
            & torch.isfinite(local_projection_std)
            & (z > 0.0)
        )
        reprojection[start:stop] = local_reprojection.cpu()
        normalized_depth[start:stop] = local_normalized_depth.cpu()
        projection_std[start:stop] = local_projection_std.cpu()
        positive_depth[start:stop] = local_positive.cpu()
    return {
        "schema": "lafgs_query_anchor_geometry_evidence",
        "version": 1,
        "candidate_offsets": offsets,
        "candidate_anchor_rows": anchors,
        "query_reprojection_residual_px": reprojection,
        "query_normalized_depth_residual": normalized_depth,
        "query_projection_std_px": projection_std,
        "query_positive_depth": positive_depth,
        "strict_depth_absolute_m": absolute,
        "strict_depth_relative": relative,
        "uses_descriptor_scores": False,
    }


def transport_candidate_graph(
    *,
    candidate_graph: Mapping,
    query_surface_xyz: torch.Tensor,
    anchor_observation_offsets: torch.Tensor,
    observation_query_indices: torch.Tensor,
    observation_keypoint_indices: torch.Tensor,
    observation_enabled: torch.Tensor | None,
    mapping_keypoints: Sequence[torch.Tensor],
    mapping_intrinsics: torch.Tensor,
    mapping_poses_w2c: torch.Tensor,
    mapping_view_family_ids: torch.Tensor,
    maximum_observations_per_candidate: int = 64,
    inlier_residual_px: float = 4.0,
    minimum_candidate_overlap_to_evaluate: float = 0.0,
) -> dict[str, torch.Tensor | str | int]:
    """Transport Query Gaussian surfaces into independent mapping views.

    Candidate generation has already happened through the primitive inverted
    index.  This function consults only the candidate Anchor's original mapping
    observations; it performs no descriptor match and no Top-L lookup.
    """

    candidate_offsets = torch.as_tensor(candidate_graph["candidate_offsets"]).long().cpu()
    candidate_anchors = torch.as_tensor(candidate_graph["candidate_anchor_rows"]).long().cpu()
    candidate_overlap = torch.as_tensor(
        candidate_graph["bhattacharyya_overlap"]
    ).float().cpu()
    row_count = int(candidate_graph["row_count"])
    _validate_offsets(candidate_offsets, candidate_anchors.numel(), row_count)
    if candidate_overlap.shape != candidate_anchors.shape:
        raise ValueError("candidate overlaps do not align with candidate Anchors")
    surface = torch.as_tensor(query_surface_xyz).float().cpu().reshape(-1, 3)
    if surface.shape[0] != row_count:
        raise ValueError("query surfaces do not align with candidate rows")
    anchor_offsets = torch.as_tensor(anchor_observation_offsets).long().cpu()
    observation_queries = torch.as_tensor(observation_query_indices).long().cpu()
    observation_keypoints = torch.as_tensor(observation_keypoint_indices).long().cpu()
    anchor_count = int(anchor_offsets.numel() - 1)
    _validate_offsets(anchor_offsets, observation_queries.numel(), anchor_count)
    if observation_keypoints.shape != observation_queries.shape:
        raise ValueError("mapping observation query/keypoint arrays do not align")
    enabled = (
        torch.ones(observation_queries.numel(), dtype=torch.bool)
        if observation_enabled is None
        else torch.as_tensor(observation_enabled).bool().cpu().reshape(-1)
    )
    if enabled.numel() != observation_queries.numel():
        raise ValueError("mapping observation enable mask does not align")
    intrinsics = torch.as_tensor(mapping_intrinsics).float().cpu()
    poses = torch.as_tensor(mapping_poses_w2c).float().cpu()
    families = torch.as_tensor(mapping_view_family_ids).long().cpu().reshape(-1)
    mapping_count = len(mapping_keypoints)
    if intrinsics.shape != (mapping_count, 3, 3) or poses.shape != (mapping_count, 4, 4):
        raise ValueError("mapping camera geometry does not align with keypoints")
    if families.numel() != mapping_count:
        raise ValueError("mapping view families do not align")
    if observation_queries.numel() and int(observation_queries.max()) >= mapping_count:
        raise ValueError("mapping observation query is outside the camera registry")
    keypoint_counts = torch.tensor(
        [torch.as_tensor(value).reshape(-1, 2).shape[0] for value in mapping_keypoints],
        dtype=torch.long,
    )
    keypoint_offsets = torch.cat(
        [torch.zeros(1, dtype=torch.long), torch.cumsum(keypoint_counts, dim=0)]
    )
    flat_keypoints = torch.cat(
        [torch.as_tensor(value).float().reshape(-1, 2) for value in mapping_keypoints],
        dim=0,
    )
    maximum = max(int(maximum_observations_per_candidate), 1)
    threshold = float(inlier_residual_px)
    if threshold <= 0.0:
        raise ValueError("transport residual threshold must be positive")
    minimum_overlap = float(minimum_candidate_overlap_to_evaluate)
    if minimum_overlap < 0.0:
        raise ValueError("minimum candidate overlap must be non-negative")

    # Select the bounded, family-balanced observation set once per Anchor.  An
    # Anchor can occur in many query candidate graphs; repeating this work per
    # occurrence changes no evidence but is prohibitively expensive.
    observation_anchor = torch.repeat_interleave(
        torch.arange(anchor_count), anchor_offsets[1:] - anchor_offsets[:-1]
    )
    enabled_observations = torch.nonzero(enabled, as_tuple=False).reshape(-1)
    enabled_anchors = observation_anchor[enabled_observations]
    enabled_counts = torch.bincount(enabled_anchors, minlength=anchor_count)
    enabled_offsets = torch.cat(
        [torch.zeros(1, dtype=torch.long), torch.cumsum(enabled_counts, dim=0)]
    )
    selected_mask = torch.ones(enabled_observations.numel(), dtype=torch.bool)
    for anchor in torch.nonzero(
        enabled_counts > maximum, as_tuple=False
    ).reshape(-1).tolist():
        start, stop = int(enabled_offsets[anchor]), int(enabled_offsets[anchor + 1])
        local_positions = torch.arange(start, stop)
        local_observations = enabled_observations[local_positions]
        local_families = families[observation_queries[local_observations]]
        queues = []
        for family in torch.unique(local_families, sorted=True).tolist():
            family_positions = local_positions[local_families == int(family)]
            family_observations = enabled_observations[family_positions]
            family_queries = observation_queries[family_observations]
            queues.append(
                family_positions[torch.argsort(family_queries, stable=True)].tolist()
            )
        selected = []
        cursor = 0
        while len(selected) < maximum:
            advanced = False
            for queue in queues:
                if cursor < len(queue):
                    selected.append(queue[cursor])
                    advanced = True
                    if len(selected) == maximum:
                        break
            if not advanced:
                break
            cursor += 1
        selected_mask[start:stop] = False
        selected_mask[torch.tensor(selected, dtype=torch.long)] = True
    selected_observations = enabled_observations[selected_mask]
    selected_anchors = enabled_anchors[selected_mask]
    selected_counts = torch.bincount(selected_anchors, minlength=anchor_count)
    selected_offsets = torch.cat(
        [torch.zeros(1, dtype=torch.long), torch.cumsum(selected_counts, dim=0)]
    )

    family_counts = torch.zeros(candidate_anchors.numel(), dtype=torch.long)
    observation_counts = torch.zeros(candidate_anchors.numel(), dtype=torch.long)
    best_residual = torch.full((candidate_anchors.numel(),), float("inf"))
    median_residual = torch.full((candidate_anchors.numel(),), float("inf"))
    edge_rows = torch.repeat_interleave(
        torch.arange(row_count), candidate_offsets[1:] - candidate_offsets[:-1]
    )
    eligible = (
        (candidate_overlap >= minimum_overlap)
        & torch.isfinite(surface[edge_rows]).all(1)
        & (selected_counts[candidate_anchors] > 0)
    )
    eligible_edges = torch.nonzero(eligible, as_tuple=False).reshape(-1)
    edge_chunk_size = 16384
    for chunk_start in range(0, eligible_edges.numel(), edge_chunk_size):
        edges = eligible_edges[chunk_start : chunk_start + edge_chunk_size]
        anchors = candidate_anchors[edges]
        counts = selected_counts[anchors]
        prefix = torch.cumsum(counts, dim=0) - counts
        owner = torch.repeat_interleave(torch.arange(edges.numel()), counts)
        within = torch.arange(owner.numel()) - torch.repeat_interleave(prefix, counts)
        local = selected_observations[selected_offsets[anchors[owner]] + within]
        local_queries = observation_queries[local]
        local_keypoints = observation_keypoints[local]
        if bool(
            (
                (local_keypoints < 0)
                | (local_keypoints >= keypoint_counts[local_queries])
            ).any()
        ):
            raise ValueError("mapping observation keypoint is outside its registry")
        target = flat_keypoints[keypoint_offsets[local_queries] + local_keypoints]
        points = surface[edge_rows[edges[owner]]]
        camera = torch.einsum(
            "nij,nj->ni", poses[local_queries, :3, :3], points
        ) + poses[local_queries, :3, 3]
        usable = torch.isfinite(camera).all(1) & (camera[:, 2] > 0.0)
        projected = torch.einsum(
            "nij,nj->ni", intrinsics[local_queries], camera
        )
        uv = projected[:, :2] / projected[:, 2:].clamp_min(1e-8)
        residual = torch.linalg.norm(uv - target, dim=1)
        finite = usable & torch.isfinite(residual)
        padded = torch.full((edges.numel(), maximum), float("inf"))
        padded[owner[finite], within[finite]] = residual[finite]
        ordered = padded.sort(dim=1).values
        finite_counts = torch.isfinite(ordered).sum(1)
        has_finite = finite_counts > 0
        observation_counts[edges] = finite_counts
        best_residual[edges[has_finite]] = ordered[has_finite, 0]
        median_index = ((finite_counts - 1).clamp_min(0) // 2).long()
        median_residual[edges[has_finite]] = ordered[
            has_finite, median_index[has_finite]
        ]
        inlier = finite & (residual <= threshold)
        if bool(inlier.any()):
            family_pairs = torch.unique(
                torch.stack([owner[inlier], families[local_queries[inlier]]], dim=1),
                dim=0,
            )
            local_family_counts = torch.bincount(
                family_pairs[:, 0], minlength=edges.numel()
            )
            family_counts[edges] = local_family_counts
    return {
        "schema": "lafgs_mapping_observation_transport_evidence",
        "version": 1,
        "candidate_offsets": candidate_offsets,
        "candidate_anchor_rows": candidate_anchors,
        "transport_view_family_count": family_counts,
        "transport_observation_count": observation_counts,
        "transport_best_residual_px": best_residual,
        "transport_median_residual_px": median_residual,
        "uses_descriptor_scores": False,
        "inlier_residual_px": threshold,
        "minimum_candidate_overlap_to_evaluate": minimum_overlap,
    }


def assign_provenance_truth(
    *,
    candidate_graph: Mapping,
    transport_evidence: Mapping | None,
    geometry_evidence: Mapping | None = None,
    equivalence_class_ids: torch.Tensor | None = None,
    thresholds: TruthAssignmentThresholds = TruthAssignmentThresholds(),
) -> dict[str, torch.Tensor | str | int | dict]:
    """Classify rows as UNIQUE/EQUIVALENT/AMBIGUOUS/NONE/INVALID."""

    thresholds.validate()
    offsets = torch.as_tensor(candidate_graph["candidate_offsets"]).long().cpu()
    anchors = torch.as_tensor(candidate_graph["candidate_anchor_rows"]).long().cpu()
    overlap = torch.as_tensor(candidate_graph["bhattacharyya_overlap"]).float().cpu()
    valid = torch.as_tensor(candidate_graph["query_valid"]).bool().cpu()
    row_count = int(candidate_graph["row_count"])
    composition_entropy = torch.as_tensor(
        candidate_graph.get("query_composition_entropy", torch.zeros(row_count))
    ).float().cpu()
    depth_spread = torch.as_tensor(
        candidate_graph.get("query_relative_depth_spread", torch.zeros(row_count))
    ).float().cpu()
    retained_fraction = torch.as_tensor(
        candidate_graph.get(
            "query_retained_composition_fraction", torch.ones(row_count)
        )
    ).float().cpu()
    _validate_offsets(offsets, anchors.numel(), row_count)
    if (
        overlap.shape != anchors.shape
        or valid.numel() != row_count
        or composition_entropy.numel() != row_count
        or depth_spread.numel() != row_count
        or retained_fraction.numel() != row_count
    ):
        raise ValueError("candidate graph fields do not align")
    if transport_evidence is None:
        family_count = torch.zeros_like(anchors)
        median_residual = torch.full_like(overlap, float("inf"))
    else:
        transport_anchors = torch.as_tensor(
            transport_evidence["candidate_anchor_rows"]
        ).long().cpu()
        transport_offsets = torch.as_tensor(
            transport_evidence["candidate_offsets"]
        ).long().cpu()
        if not torch.equal(transport_anchors, anchors) or not torch.equal(
            transport_offsets, offsets
        ):
            raise ValueError("transport evidence differs from the candidate graph")
        family_count = torch.as_tensor(
            transport_evidence["transport_view_family_count"]
        ).long().cpu()
        median_residual = torch.as_tensor(
            transport_evidence["transport_median_residual_px"]
        ).float().cpu()
    if geometry_evidence is None:
        query_reprojection = torch.zeros_like(overlap)
        query_normalized_depth = torch.zeros_like(overlap)
        query_projection_std = torch.zeros_like(overlap)
        query_positive_depth = torch.ones_like(anchors, dtype=torch.bool)
    else:
        geometry_anchors = torch.as_tensor(
            geometry_evidence["candidate_anchor_rows"]
        ).long().cpu()
        geometry_offsets = torch.as_tensor(
            geometry_evidence["candidate_offsets"]
        ).long().cpu()
        if not torch.equal(geometry_anchors, anchors) or not torch.equal(
            geometry_offsets, offsets
        ):
            raise ValueError("Query geometry evidence differs from the candidate graph")
        query_reprojection = torch.as_tensor(
            geometry_evidence["query_reprojection_residual_px"]
        ).float().cpu()
        query_normalized_depth = torch.as_tensor(
            geometry_evidence["query_normalized_depth_residual"]
        ).float().cpu()
        query_projection_std = torch.as_tensor(
            geometry_evidence["query_projection_std_px"]
        ).float().cpu()
        query_positive_depth = torch.as_tensor(
            geometry_evidence["query_positive_depth"]
        ).bool().cpu()
        if not (
            query_reprojection.shape
            == query_normalized_depth.shape
            == query_projection_std.shape
            == query_positive_depth.shape
            == overlap.shape
        ):
            raise ValueError("Query geometry evidence fields do not align")
    if equivalence_class_ids is None:
        equivalence = torch.arange(int(candidate_graph.get("anchor_count", 0)))
        if anchors.numel() and equivalence.numel() <= int(anchors.max()):
            equivalence = torch.arange(int(anchors.max()) + 1)
    else:
        equivalence = torch.as_tensor(equivalence_class_ids).long().cpu().reshape(-1)
    if anchors.numel() and (int(anchors.min()) < 0 or int(anchors.max()) >= equivalence.numel()):
        raise ValueError("candidate Anchor is outside the equivalence registry")

    status = torch.full((row_count,), TRUTH_NONE, dtype=torch.int8)
    confidence = torch.zeros(row_count)
    top1_margin = torch.zeros(row_count)
    top1_ratio = torch.ones(row_count)
    truth_offsets = [0]
    truth_anchors: list[int] = []
    for row in range(row_count):
        diagnostic_valid = bool(
            torch.isfinite(composition_entropy[row])
            and torch.isfinite(depth_spread[row])
            and torch.isfinite(retained_fraction[row])
            and float(composition_entropy[row])
            <= float(thresholds.maximum_composition_entropy)
            and float(depth_spread[row])
            <= float(thresholds.maximum_relative_depth_spread)
            and float(retained_fraction[row])
            >= float(thresholds.minimum_retained_composition_fraction)
        )
        if not bool(valid[row]) or not diagnostic_valid:
            status[row] = TRUTH_INVALID
            truth_offsets.append(len(truth_anchors))
            continue
        start, stop = int(offsets[row]), int(offsets[row + 1])
        if stop <= start:
            status[row] = TRUTH_NONE
            truth_offsets.append(len(truth_anchors))
            continue
        local_overlap = overlap[start:stop]
        plausible = local_overlap >= float(thresholds.minimum_provenance_overlap)
        if not bool(plausible.any()):
            status[row] = TRUTH_NONE
            truth_offsets.append(len(truth_anchors))
            continue
        transport_ok = (
            (family_count[start:stop] >= int(thresholds.minimum_transport_view_families))
            & torch.isfinite(median_residual[start:stop])
            & (
                median_residual[start:stop]
                <= float(thresholds.maximum_transport_median_residual_px)
            )
        )
        geometry_ok = (
            query_positive_depth[start:stop]
            & torch.isfinite(query_reprojection[start:stop])
            & torch.isfinite(query_normalized_depth[start:stop])
            & torch.isfinite(query_projection_std[start:stop])
            & (
                query_reprojection[start:stop]
                <= float(thresholds.maximum_query_reprojection_px)
            )
            & (
                query_normalized_depth[start:stop]
                <= float(thresholds.maximum_query_normalized_depth_residual)
            )
            & (
                query_projection_std[start:stop]
                <= float(thresholds.maximum_query_projection_std_px)
            )
        )
        eligible = plausible & transport_ok & geometry_ok
        if not bool(eligible.any()):
            status[row] = TRUTH_AMBIGUOUS
            truth_offsets.append(len(truth_anchors))
            continue
        local_anchor = anchors[start:stop]
        local_class = equivalence[local_anchor]
        residual_quality = torch.exp(
            -0.5
            * (
                median_residual[start:stop]
                / float(thresholds.maximum_transport_median_residual_px)
            ).square()
        )
        local_family_count = family_count[start:stop].float()
        maximum_family_count = local_family_count[eligible].max().clamp_min(1.0)
        family_quality = local_family_count / maximum_family_count
        reprojection_quality = torch.exp(
            -0.5
            * (
                query_reprojection[start:stop]
                / float(thresholds.maximum_query_reprojection_px)
            ).square()
        )
        depth_quality = torch.exp(
            -0.5
            * (
                query_normalized_depth[start:stop]
                / float(thresholds.maximum_query_normalized_depth_residual)
            ).square()
        )
        local_confidence = (
            local_overlap
            * residual_quality
            * family_quality
            * reprojection_quality
            * depth_quality
        )
        grouped: dict[int, tuple[float, list[int]]] = {}
        for column in torch.nonzero(eligible, as_tuple=False).reshape(-1).tolist():
            group = int(local_class[column])
            anchor = int(local_anchor[column])
            score = float(local_confidence[column])
            previous = grouped.get(group)
            if previous is None:
                grouped[group] = (score, [anchor])
            else:
                grouped[group] = (max(previous[0], score), previous[1] + [anchor])
        ordered = sorted(grouped.items(), key=lambda item: (-item[1][0], item[0]))
        _, (best_score, best_anchors) = ordered[0]
        second_score = ordered[1][1][0] if len(ordered) > 1 else 0.0
        margin = best_score - second_score
        ratio = best_score / max(second_score, 1e-12) if second_score > 0 else float("inf")
        confidence[row] = best_score
        top1_margin[row] = margin
        top1_ratio[row] = ratio
        decisive = (
            best_score >= float(thresholds.minimum_assignment_confidence)
            and margin >= float(thresholds.minimum_top1_top2_margin)
            and ratio >= float(thresholds.minimum_top1_top2_ratio)
        )
        if not decisive:
            status[row] = TRUTH_AMBIGUOUS
            truth_offsets.append(len(truth_anchors))
            continue
        best_anchors = sorted(set(best_anchors))
        truth_anchors.extend(best_anchors)
        status[row] = TRUTH_EQUIVALENT if len(best_anchors) > 1 else TRUTH_UNIQUE
        truth_offsets.append(len(truth_anchors))
    counts = {
        name: int((status == code).sum())
        for code, name in enumerate(TRUTH_STATUS_NAMES)
    }
    return {
        "schema": "lafgs_gaussian_provenance_anchor_truth",
        "version": 1,
        "row_count": row_count,
        "truth_status": status,
        "truth_status_names": TRUTH_STATUS_NAMES,
        "truth_offsets": torch.tensor(truth_offsets, dtype=torch.long),
        "truth_anchor_rows": torch.tensor(truth_anchors, dtype=torch.long),
        "assignment_confidence": confidence,
        "top1_top2_confidence_margin": top1_margin,
        "top1_top2_confidence_ratio": top1_ratio,
        "status_counts": counts,
        "uses_descriptor_scores": False,
        "uses_topl_candidates": False,
        "calibration_policy": "disjoint_mapping_view_families_no_loo",
        "thresholds": {
            name: getattr(thresholds, name)
            for name in TruthAssignmentThresholds.__dataclass_fields__
        },
    }


def truth_membership_mask(
    truth: Mapping,
    candidate_anchor_rows: torch.Tensor,
) -> torch.Tensor:
    """Compare a competition graph with truth without changing the truth graph."""

    candidates = torch.as_tensor(candidate_anchor_rows).long().cpu()
    if candidates.ndim != 2 or candidates.shape[0] != int(truth["row_count"]):
        raise ValueError("competition rows do not align with truth rows")
    offsets = torch.as_tensor(truth["truth_offsets"]).long().cpu()
    anchors = torch.as_tensor(truth["truth_anchor_rows"]).long().cpu()
    output = torch.zeros_like(candidates, dtype=torch.bool)
    for row in range(candidates.shape[0]):
        start, stop = int(offsets[row]), int(offsets[row + 1])
        if stop > start:
            output[row] = torch.isin(candidates[row], anchors[start:stop])
    return output


@torch.inference_mode()
def assign_full_map_projection_truth(
    *,
    keypoints: torch.Tensor,
    rendered_depth: torch.Tensor,
    query_indices: torch.Tensor,
    anchor_xyz: torch.Tensor,
    anchor_covariance: torch.Tensor,
    observation_count: torch.Tensor,
    mapping_intrinsics: torch.Tensor,
    mapping_poses_w2c: torch.Tensor,
    equivalence_class_ids: torch.Tensor | None = None,
    strict_reprojection_px: float = 4.0,
    strict_depth_absolute_m: float = 0.25,
    strict_depth_relative: float = 0.05,
    maximum_projection_std_px: float = 2.0,
    minimum_observations: int = 3,
    maximum_truth_anchors_per_row: int = 64,
    device: str | torch.device = "cpu",
) -> dict[str, torch.Tensor | str | int | dict]:
    """Full-map projection-only teacher used solely as the V16 comparator.

    Unlike the old V16 call site this scans the complete Anchor map rather than
    descriptor Top-L.  It intentionally retains the old fixed geometric
    thresholds so the provenance teacher can be compared against that exact
    heuristic family.  It is not used to calibrate provenance thresholds.
    """

    xy = torch.as_tensor(keypoints).float().cpu().reshape(-1, 2)
    depth = torch.as_tensor(rendered_depth).float().cpu().reshape(-1)
    queries = torch.as_tensor(query_indices).long().cpu().reshape(-1)
    row_count = int(xy.shape[0])
    if depth.numel() != row_count or queries.numel() != row_count:
        raise ValueError("projection truth query rows do not align")
    xyz_cpu = torch.as_tensor(anchor_xyz).float().cpu()
    covariance_cpu = torch.as_tensor(anchor_covariance).float().cpu()
    support_cpu = torch.as_tensor(observation_count).long().cpu().reshape(-1)
    anchor_count = int(xyz_cpu.shape[0])
    if covariance_cpu.shape != (anchor_count, 3, 3) or support_cpu.numel() != anchor_count:
        raise ValueError("projection truth Anchor fields do not align")
    intrinsics_cpu = torch.as_tensor(mapping_intrinsics).float().cpu()
    poses_cpu = torch.as_tensor(mapping_poses_w2c).float().cpu()
    if queries.numel() and (
        int(queries.min()) < 0 or int(queries.max()) >= intrinsics_cpu.shape[0]
    ):
        raise ValueError("projection truth query is outside the camera registry")
    if equivalence_class_ids is None:
        equivalence = torch.arange(anchor_count)
    else:
        equivalence = torch.as_tensor(equivalence_class_ids).long().cpu().reshape(-1)
        if equivalence.numel() != anchor_count:
            raise ValueError("projection truth equivalence registry does not align")
    compute_device = torch.device(device)
    xyz = xyz_cpu.to(compute_device)
    covariance = covariance_cpu.to(compute_device)
    support = support_cpu.to(compute_device)
    output: list[list[int]] = [[] for _ in range(row_count)]
    for query in torch.unique(queries, sorted=True).tolist():
        rows_cpu = torch.nonzero(queries == int(query), as_tuple=False).reshape(-1)
        calibration = intrinsics_cpu[query].to(compute_device)
        pose = poses_cpu[query].to(compute_device)
        rotation = pose[:3, :3]
        camera = xyz @ rotation.T + pose[:3, 3]
        projected = camera @ calibration.T
        uv = projected[:, :2] / projected[:, 2:].clamp_min(1e-8)
        camera_covariance = torch.einsum(
            "ab,nbc,dc->nad", rotation, covariance, rotation
        )
        x, y, z = camera.unbind(1)
        dproj = camera.new_zeros((anchor_count, 2, 3))
        dproj[:, 0, 0] = calibration[0, 0] / z.clamp_min(1e-8)
        dproj[:, 0, 2] = -calibration[0, 0] * x / z.square().clamp_min(1e-8)
        dproj[:, 1, 1] = calibration[1, 1] / z.clamp_min(1e-8)
        dproj[:, 1, 2] = -calibration[1, 1] * y / z.square().clamp_min(1e-8)
        pixel_covariance = dproj @ camera_covariance @ dproj.transpose(-1, -2)
        projection_std = (
            pixel_covariance.diagonal(dim1=-2, dim2=-1).sum(1).clamp_min(0).sqrt()
        )
        depth_std = camera_covariance[:, 2, 2].clamp_min(0).sqrt()
        query_xy = xy[rows_cpu].to(compute_device)
        query_depth = depth[rows_cpu].to(compute_device)
        reprojection = torch.cdist(query_xy, uv)
        depth_error = (query_depth[:, None] - camera[:, 2][None]).abs()
        depth_tolerance = torch.maximum(
            torch.full_like(query_depth, float(strict_depth_absolute_m)),
            query_depth.abs() * float(strict_depth_relative),
        )
        positive = (
            torch.isfinite(reprojection)
            & torch.isfinite(depth_error)
            & torch.isfinite(projection_std)[None]
            & torch.isfinite(depth_std)[None]
            & (camera[:, 2] > 0)[None]
            & (support >= int(minimum_observations))[None]
            & (reprojection <= float(strict_reprojection_px))
            & (projection_std <= float(maximum_projection_std_px))[None]
            & (depth_std <= depth_tolerance[:, None])
            & (depth_error <= depth_tolerance[:, None] + 2.0 * depth_std[None])
        )
        normalized = reprojection / max(float(strict_reprojection_px), 1e-8)
        normalized = normalized + depth_error / depth_tolerance[:, None].clamp_min(1e-8)
        normalized = normalized.masked_fill(~positive, float("inf"))
        maximum = min(max(int(maximum_truth_anchors_per_row), 1), anchor_count)
        values, indices = torch.topk(normalized, maximum, dim=1, largest=False)
        for local, global_row in enumerate(rows_cpu.tolist()):
            finite = torch.isfinite(values[local])
            output[global_row] = indices[local, finite].cpu().tolist()

    status = torch.full((row_count,), TRUTH_NONE, dtype=torch.int8)
    offsets = [0]
    anchors: list[int] = []
    for row, candidates in enumerate(output):
        if not torch.isfinite(xy[row]).all() or not torch.isfinite(depth[row]) or depth[row] <= 0:
            status[row] = TRUTH_INVALID
            offsets.append(len(anchors))
            continue
        if not candidates:
            status[row] = TRUTH_NONE
            offsets.append(len(anchors))
            continue
        classes = defaultdict(list)
        for anchor in candidates:
            classes[int(equivalence[anchor])].append(int(anchor))
        if len(classes) != 1:
            status[row] = TRUTH_AMBIGUOUS
            offsets.append(len(anchors))
            continue
        values = sorted(next(iter(classes.values())))
        anchors.extend(values)
        status[row] = TRUTH_EQUIVALENT if len(values) > 1 else TRUTH_UNIQUE
        offsets.append(len(anchors))
    return {
        "schema": "lafgs_full_map_projection_truth_comparator",
        "version": 1,
        "row_count": row_count,
        "truth_status": status,
        "truth_status_names": TRUTH_STATUS_NAMES,
        "truth_offsets": torch.tensor(offsets, dtype=torch.long),
        "truth_anchor_rows": torch.tensor(anchors, dtype=torch.long),
        "status_counts": {
            name: int((status == code).sum())
            for code, name in enumerate(TRUTH_STATUS_NAMES)
        },
        "uses_descriptor_scores": False,
        "uses_topl_candidates": False,
        "heuristic_projection_comparator_only": True,
        "thresholds": {
            "strict_reprojection_px": float(strict_reprojection_px),
            "strict_depth_absolute_m": float(strict_depth_absolute_m),
            "strict_depth_relative": float(strict_depth_relative),
            "maximum_projection_std_px": float(maximum_projection_std_px),
            "minimum_observations": int(minimum_observations),
        },
    }


__all__ = [
    "TRUTH_AMBIGUOUS",
    "TRUTH_EQUIVALENT",
    "TRUTH_INVALID",
    "TRUTH_NONE",
    "TRUTH_STATUS_NAMES",
    "TRUTH_UNIQUE",
    "TruthAssignmentThresholds",
    "aggregate_anchor_provenance",
    "assign_full_map_projection_truth",
    "assign_provenance_truth",
    "backproject_query_surface",
    "build_primitive_anchor_index",
    "provenance_candidate_graph",
    "query_anchor_geometry_evidence",
    "transport_candidate_graph",
    "truth_membership_mask",
]
