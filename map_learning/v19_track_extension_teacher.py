"""Selective novel-view teacher that extends frozen mapping Tracks.

The teacher never reads descriptor Top-L retrieval.  Full-map projection first
proposes geometrically legal Anchors.  The Query surface point is then
transported to the frozen mapping observations of each candidate Track, where
independent view-family agreement and the native (identity) descriptor provide
selective identity evidence.  Feedback rows are never inserted into a Track.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F

from map_learning.v18_provenance_truth import (
    TRUTH_AMBIGUOUS,
    TRUTH_EQUIVALENT,
    TRUTH_INVALID,
    TRUTH_NONE,
    TRUTH_STATUS_NAMES,
    TRUTH_UNIQUE,
)


@dataclass(frozen=True)
class TrackExtensionTier:
    maximum_query_reprojection_px: float
    maximum_query_normalized_depth_residual: float
    maximum_query_projection_std_px: float
    maximum_transport_median_residual_px: float
    minimum_transport_view_families: int
    minimum_descriptor_cosine: float
    minimum_descriptor_view_families: int

    def validate(self) -> None:
        if min(
            self.maximum_query_reprojection_px,
            self.maximum_query_normalized_depth_residual,
            self.maximum_query_projection_std_px,
            self.maximum_transport_median_residual_px,
        ) <= 0.0:
            raise ValueError("Track-extension residual thresholds must be positive")
        if min(
            self.minimum_transport_view_families,
            self.minimum_descriptor_view_families,
        ) < 1:
            raise ValueError("Track-extension family support must be positive")
        if not -1.0 <= self.minimum_descriptor_cosine <= 1.0:
            raise ValueError("Track-extension cosine threshold must lie in [-1, 1]")


TRACK_EXTENSION_TIERS = {
    "tier_a": TrackExtensionTier(1.5, 0.50, 1.0, 2.0, 3, 0.85, 2),
    "tier_b": TrackExtensionTier(2.5, 0.75, 1.5, 3.0, 2, 0.75, 2),
    "tier_c": TrackExtensionTier(4.0, 1.00, 2.0, 4.0, 2, 0.65, 1),
}


@torch.inference_mode()
def prepare_track_observation_bank(
    *,
    anchor_observation_offsets: torch.Tensor,
    observation_query_indices: torch.Tensor,
    observation_keypoint_indices: torch.Tensor,
    observation_enabled: torch.Tensor,
    mapping_keypoints: Sequence[torch.Tensor],
    mapping_descriptors: Sequence[torch.Tensor],
    mapping_view_family_ids: torch.Tensor,
    maximum_observations_per_anchor: int = 48,
) -> dict:
    """Freeze family-balanced Track observations and flattened native features."""

    anchor_offsets = torch.as_tensor(anchor_observation_offsets).long().cpu()
    observation_queries = torch.as_tensor(observation_query_indices).long().cpu()
    observation_keypoints = torch.as_tensor(observation_keypoint_indices).long().cpu()
    enabled = torch.as_tensor(observation_enabled).bool().cpu().reshape(-1)
    families = torch.as_tensor(mapping_view_family_ids).long().cpu()
    anchor_count = int(anchor_offsets.numel() - 1)
    _validate_offsets(anchor_offsets, observation_queries.numel(), anchor_count)
    if observation_keypoints.shape != observation_queries.shape or enabled.numel() != observation_queries.numel():
        raise ValueError("Track-extension observation registry does not align")
    mapping_count = len(mapping_keypoints)
    if len(mapping_descriptors) != mapping_count or families.numel() != mapping_count:
        raise ValueError("Track-extension mapping feature banks do not align")
    descriptor_dim = int(torch.as_tensor(mapping_descriptors[0]).shape[-1])
    keypoint_counts = torch.tensor(
        [torch.as_tensor(value).reshape(-1, 2).shape[0] for value in mapping_keypoints]
    )
    descriptor_counts = torch.tensor(
        [
            torch.as_tensor(value).reshape(-1, descriptor_dim).shape[0]
            for value in mapping_descriptors
        ]
    )
    if not torch.equal(keypoint_counts, descriptor_counts):
        raise ValueError("Track-extension mapping keypoints/descriptors differ")
    keypoint_offsets = torch.cat(
        (torch.zeros(1, dtype=torch.long), keypoint_counts.cumsum(0))
    )
    flat_keypoints = torch.cat(
        [torch.as_tensor(value).float().reshape(-1, 2) for value in mapping_keypoints]
    )
    flat_descriptors = F.normalize(
        torch.cat(
            [
                torch.as_tensor(value).float().reshape(-1, descriptor_dim)
                for value in mapping_descriptors
            ]
        ),
        dim=1,
    )
    observation_anchor = torch.repeat_interleave(
        torch.arange(anchor_count), anchor_offsets[1:] - anchor_offsets[:-1]
    )
    maximum = max(int(maximum_observations_per_anchor), 1)
    by_anchor: list[list[int]] = [[] for _ in range(anchor_count)]
    for observation in torch.nonzero(enabled, as_tuple=False).reshape(-1).tolist():
        by_anchor[int(observation_anchor[observation])].append(observation)
    selected: list[int] = []
    selected_counts = torch.zeros(anchor_count, dtype=torch.long)
    selected_offsets = torch.zeros(anchor_count + 1, dtype=torch.long)
    for anchor, values in enumerate(by_anchor):
        queues: dict[int, list[int]] = defaultdict(list)
        for observation in values:
            queues[int(families[observation_queries[observation]])].append(observation)
        ordered_queues = [queues[key] for key in sorted(queues)]
        local = []
        cursor = 0
        while len(local) < maximum:
            advanced = False
            for queue in ordered_queues:
                if cursor < len(queue):
                    local.append(queue[cursor])
                    advanced = True
                    if len(local) == maximum:
                        break
            if not advanced:
                break
            cursor += 1
        selected.extend(local)
        selected_counts[anchor] = len(local)
        selected_offsets[anchor + 1] = len(selected)
    return {
        "schema": "lafgs_v19_prepared_track_observation_bank",
        "version": 1,
        "anchor_count": anchor_count,
        "mapping_count": mapping_count,
        "descriptor_dim": descriptor_dim,
        "selected_observations": torch.tensor(selected, dtype=torch.long),
        "selected_counts": selected_counts,
        "selected_offsets": selected_offsets,
        "keypoint_offsets": keypoint_offsets,
        "flat_keypoints": flat_keypoints,
        "flat_descriptors": flat_descriptors,
        "maximum_observations_per_anchor": maximum,
    }


def _validate_offsets(offsets: torch.Tensor, edge_count: int, row_count: int) -> None:
    if (
        offsets.shape != (row_count + 1,)
        or int(offsets[0]) != 0
        or int(offsets[-1]) != int(edge_count)
        or bool((offsets[1:] < offsets[:-1]).any())
    ):
        raise ValueError("Track-extension CSR offsets do not align")


@torch.inference_mode()
def full_map_projection_candidate_graph(
    *,
    keypoints: torch.Tensor,
    rendered_depth: torch.Tensor,
    query_indices: torch.Tensor,
    anchor_xyz: torch.Tensor,
    anchor_covariance: torch.Tensor,
    observation_count: torch.Tensor,
    query_intrinsics: torch.Tensor,
    query_poses_w2c: torch.Tensor,
    broad_reprojection_px: float = 4.0,
    broad_depth_absolute_m: float = 0.25,
    broad_depth_relative: float = 0.05,
    broad_normalized_depth_residual: float = 1.0,
    broad_projection_std_px: float = 2.0,
    minimum_observations: int = 3,
    maximum_candidates_per_row: int = 64,
    row_chunk_size: int = 256,
    device: str | torch.device = "cpu",
) -> dict:
    """Enumerate full-map geometric candidates without descriptor retrieval."""

    xy = torch.as_tensor(keypoints).float().cpu().reshape(-1, 2)
    depth = torch.as_tensor(rendered_depth).float().cpu().reshape(-1)
    queries = torch.as_tensor(query_indices).long().cpu().reshape(-1)
    row_count = int(xy.shape[0])
    if depth.numel() != row_count or queries.numel() != row_count:
        raise ValueError("Track-extension Query rows do not align")
    xyz_cpu = torch.as_tensor(anchor_xyz).float().cpu()
    covariance_cpu = torch.as_tensor(anchor_covariance).float().cpu()
    support_cpu = torch.as_tensor(observation_count).long().cpu().reshape(-1)
    anchor_count = int(xyz_cpu.shape[0])
    if (
        covariance_cpu.shape != (anchor_count, 3, 3)
        or support_cpu.numel() != anchor_count
    ):
        raise ValueError("Track-extension Anchor geometry does not align")
    intrinsics = torch.as_tensor(query_intrinsics).float().cpu()
    poses = torch.as_tensor(query_poses_w2c).float().cpu()
    if intrinsics.ndim != 3 or intrinsics.shape[1:] != (3, 3):
        raise ValueError("Track-extension intrinsics must be [Q,3,3]")
    if poses.shape != (intrinsics.shape[0], 4, 4):
        raise ValueError("Track-extension camera registry does not align")
    if queries.numel() and (
        int(queries.min()) < 0 or int(queries.max()) >= intrinsics.shape[0]
    ):
        raise ValueError("Track-extension Query index is outside its registry")
    maximum = max(int(maximum_candidates_per_row), 1)
    chunk_size = max(int(row_chunk_size), 1)
    compute = torch.device(device)
    xyz = xyz_cpu.to(compute)
    covariance = covariance_cpu.to(compute)
    support = support_cpu.to(compute)
    row_candidates: list[torch.Tensor] = [torch.empty(0, dtype=torch.long) for _ in range(row_count)]
    row_reprojection: list[torch.Tensor] = [torch.empty(0) for _ in range(row_count)]
    row_depth: list[torch.Tensor] = [torch.empty(0) for _ in range(row_count)]
    row_std: list[torch.Tensor] = [torch.empty(0) for _ in range(row_count)]
    valid = torch.isfinite(xy).all(1) & torch.isfinite(depth) & (depth > 0.0)
    for query in torch.unique(queries[valid], sorted=True).tolist():
        rows = torch.nonzero(valid & (queries == int(query)), as_tuple=False).reshape(-1)
        calibration = intrinsics[query].to(compute)
        pose = poses[query].to(compute)
        rotation = pose[:3, :3]
        camera = xyz @ rotation.T + pose[:3, 3]
        projected = camera @ calibration.T
        uv = projected[:, :2] / projected[:, 2:].clamp_min(1e-8)
        camera_covariance = torch.einsum(
            "ab,nbc,dc->nad", rotation, covariance, rotation
        )
        x, y, z = camera.unbind(1)
        jacobian = camera.new_zeros((anchor_count, 2, 3))
        jacobian[:, 0, 0] = calibration[0, 0] / z.clamp_min(1e-8)
        jacobian[:, 0, 2] = -calibration[0, 0] * x / z.square().clamp_min(1e-8)
        jacobian[:, 1, 1] = calibration[1, 1] / z.clamp_min(1e-8)
        jacobian[:, 1, 2] = -calibration[1, 1] * y / z.square().clamp_min(1e-8)
        pixel_covariance = jacobian @ camera_covariance @ jacobian.transpose(-1, -2)
        projection_std = (
            pixel_covariance.diagonal(dim1=-2, dim2=-1)
            .sum(1)
            .clamp_min(0.0)
            .sqrt()
        )
        depth_std = camera_covariance[:, 2, 2].clamp_min(0.0).sqrt()
        anchor_usable = (
            torch.isfinite(uv).all(1)
            & torch.isfinite(z)
            & torch.isfinite(projection_std)
            & torch.isfinite(depth_std)
            & (z > 0.0)
            & (support >= int(minimum_observations))
            & (projection_std <= float(broad_projection_std_px))
        )
        for start in range(0, rows.numel(), chunk_size):
            local_rows = rows[start : start + chunk_size]
            local_xy = xy[local_rows].to(compute)
            local_depth = depth[local_rows].to(compute)
            reprojection = torch.cdist(local_xy, uv)
            tolerance = torch.maximum(
                torch.full_like(local_depth, float(broad_depth_absolute_m)),
                local_depth.abs() * float(broad_depth_relative),
            )
            normalized_depth = (local_depth[:, None] - z[None]).abs() / (
                tolerance[:, None] + 2.0 * depth_std[None]
            ).clamp_min(1e-8)
            legal = (
                anchor_usable[None]
                & torch.isfinite(reprojection)
                & torch.isfinite(normalized_depth)
                & (reprojection <= float(broad_reprojection_px))
                & (
                    normalized_depth
                    <= float(broad_normalized_depth_residual)
                )
            )
            cost = (
                reprojection / max(float(broad_reprojection_px), 1e-8)
                + normalized_depth
            ).masked_fill(~legal, float("inf"))
            k = min(maximum, anchor_count)
            values, indices = torch.topk(cost, k, dim=1, largest=False)
            for local, global_row in enumerate(local_rows.tolist()):
                keep = torch.isfinite(values[local])
                selected = indices[local, keep]
                row_candidates[global_row] = selected.cpu()
                row_reprojection[global_row] = reprojection[local, selected].cpu()
                row_depth[global_row] = normalized_depth[local, selected].cpu()
                row_std[global_row] = projection_std[selected].cpu()
    counts = torch.tensor([item.numel() for item in row_candidates], dtype=torch.long)
    offsets = torch.cat((torch.zeros(1, dtype=torch.long), counts.cumsum(0)))
    return {
        "schema": "lafgs_v19_full_map_projection_candidate_graph",
        "version": 1,
        "row_count": row_count,
        "anchor_count": anchor_count,
        "candidate_offsets": offsets,
        "candidate_anchor_rows": torch.cat(row_candidates),
        "query_reprojection_residual_px": torch.cat(row_reprojection),
        "query_normalized_depth_residual": torch.cat(row_depth),
        "query_projection_std_px": torch.cat(row_std),
        "query_valid": valid,
        "uses_descriptor_scores": False,
        "uses_topl_candidates": False,
        "minimum_observations": int(minimum_observations),
        "broad_thresholds": {
            "reprojection_px": float(broad_reprojection_px),
            "depth_absolute_m": float(broad_depth_absolute_m),
            "depth_relative": float(broad_depth_relative),
            "normalized_depth_residual": float(broad_normalized_depth_residual),
            "projection_std_px": float(broad_projection_std_px),
        },
    }


@torch.inference_mode()
def track_observation_consensus(
    *,
    candidate_graph: Mapping,
    query_surface_xyz: torch.Tensor,
    query_descriptors: torch.Tensor,
    anchor_observation_offsets: torch.Tensor,
    observation_query_indices: torch.Tensor,
    observation_keypoint_indices: torch.Tensor,
    observation_enabled: torch.Tensor,
    mapping_keypoints: Sequence[torch.Tensor],
    mapping_descriptors: Sequence[torch.Tensor],
    mapping_intrinsics: torch.Tensor,
    mapping_poses_w2c: torch.Tensor,
    mapping_view_family_ids: torch.Tensor,
    geometric_inlier_px: float = 4.0,
    descriptor_inlier_cosine: float = 0.65,
    maximum_observations_per_candidate: int = 48,
    edge_chunk_size: int = 2048,
    device: str | torch.device = "cpu",
    prepared_observation_bank: Mapping | None = None,
) -> dict:
    """Score candidate Tracks using frozen observations from independent families."""

    offsets = torch.as_tensor(candidate_graph["candidate_offsets"]).long().cpu()
    candidates = torch.as_tensor(candidate_graph["candidate_anchor_rows"]).long().cpu()
    row_count = int(candidate_graph["row_count"])
    _validate_offsets(offsets, candidates.numel(), row_count)
    surface = torch.as_tensor(query_surface_xyz).float().cpu().reshape(-1, 3)
    query_features = F.normalize(
        torch.as_tensor(query_descriptors).float().cpu().reshape(row_count, -1), dim=1
    )
    if surface.shape[0] != row_count:
        raise ValueError("Track-extension Query surfaces do not align")
    anchor_offsets = torch.as_tensor(anchor_observation_offsets).long().cpu()
    observation_queries = torch.as_tensor(observation_query_indices).long().cpu()
    observation_keypoints = torch.as_tensor(observation_keypoint_indices).long().cpu()
    enabled = torch.as_tensor(observation_enabled).bool().cpu().reshape(-1)
    anchor_count = int(anchor_offsets.numel() - 1)
    _validate_offsets(anchor_offsets, observation_queries.numel(), anchor_count)
    if observation_keypoints.shape != observation_queries.shape or enabled.numel() != observation_queries.numel():
        raise ValueError("Track-extension observation registry does not align")
    if candidates.numel() and int(candidates.max()) >= anchor_count:
        raise ValueError("Track-extension candidate is outside the Anchor registry")
    intrinsics = torch.as_tensor(mapping_intrinsics).float().cpu()
    poses = torch.as_tensor(mapping_poses_w2c).float().cpu()
    families = torch.as_tensor(mapping_view_family_ids).long().cpu()
    mapping_count = len(mapping_keypoints)
    if len(mapping_descriptors) != mapping_count or intrinsics.shape != (mapping_count, 3, 3) or poses.shape != (mapping_count, 4, 4) or families.numel() != mapping_count:
        raise ValueError("Track-extension mapping observation banks do not align")
    if prepared_observation_bank is None:
        bank = prepare_track_observation_bank(
            anchor_observation_offsets=anchor_offsets,
            observation_query_indices=observation_queries,
            observation_keypoint_indices=observation_keypoints,
            observation_enabled=enabled,
            mapping_keypoints=mapping_keypoints,
            mapping_descriptors=mapping_descriptors,
            mapping_view_family_ids=families,
            maximum_observations_per_anchor=maximum_observations_per_candidate,
        )
    else:
        bank = prepared_observation_bank
    maximum = int(bank["maximum_observations_per_anchor"])
    if not (
        bank.get("schema") == "lafgs_v19_prepared_track_observation_bank"
        and int(bank["anchor_count"]) == anchor_count
        and int(bank["mapping_count"]) == mapping_count
        and int(bank["descriptor_dim"]) == query_features.shape[1]
        and maximum == max(int(maximum_observations_per_candidate), 1)
    ):
        raise ValueError("prepared Track observation bank differs")
    selected_observations = torch.as_tensor(bank["selected_observations"]).long()
    selected_counts = torch.as_tensor(bank["selected_counts"]).long()
    selected_offsets = torch.as_tensor(bank["selected_offsets"]).long()
    keypoint_offsets = torch.as_tensor(bank["keypoint_offsets"]).long()
    flat_keypoints = torch.as_tensor(bank["flat_keypoints"]).float()
    flat_descriptors = torch.as_tensor(bank["flat_descriptors"]).float()
    transport_family_count = torch.zeros(candidates.numel(), dtype=torch.long)
    descriptor_family_count = torch.zeros(candidates.numel(), dtype=torch.long)
    median_residual = torch.full((candidates.numel(),), float("inf"))
    best_cosine = torch.full((candidates.numel(),), -1.0)
    edge_rows = torch.repeat_interleave(
        torch.arange(row_count), offsets[1:] - offsets[:-1]
    )
    compute = torch.device(device)
    for start in range(0, candidates.numel(), max(int(edge_chunk_size), 1)):
        edges = torch.arange(start, min(start + int(edge_chunk_size), candidates.numel()))
        anchors = candidates[edges]
        counts = selected_counts[anchors]
        usable_edges = edges[counts > 0]
        if usable_edges.numel() == 0:
            continue
        anchors = candidates[usable_edges]
        counts = selected_counts[anchors]
        prefix = torch.cumsum(counts, 0) - counts
        owner = torch.repeat_interleave(torch.arange(usable_edges.numel()), counts)
        within = torch.arange(owner.numel()) - torch.repeat_interleave(prefix, counts)
        observations = selected_observations[selected_offsets[anchors[owner]] + within]
        mapping_query = observation_queries[observations]
        mapping_keypoint = observation_keypoints[observations]
        flat_rows = keypoint_offsets[mapping_query] + mapping_keypoint
        target_xy = flat_keypoints[flat_rows].to(compute)
        target_descriptor = flat_descriptors[flat_rows].to(compute)
        query_row = edge_rows[usable_edges[owner]]
        points = surface[query_row].to(compute)
        calibration = intrinsics[mapping_query].to(compute)
        pose = poses[mapping_query].to(compute)
        camera = torch.einsum("nij,nj->ni", pose[:, :3, :3], points) + pose[:, :3, 3]
        projected = torch.einsum("nij,nj->ni", calibration, camera)
        uv = projected[:, :2] / projected[:, 2:].clamp_min(1e-8)
        residual = torch.linalg.norm(uv - target_xy, dim=1)
        cosine = (query_features[query_row].to(compute) * target_descriptor).sum(1)
        finite = (
            torch.isfinite(camera).all(1)
            & (camera[:, 2] > 0.0)
            & torch.isfinite(residual)
            & torch.isfinite(cosine)
        )
        geometric = finite & (residual <= float(geometric_inlier_px))
        descriptor = geometric & (cosine >= float(descriptor_inlier_cosine))
        padded_residual = torch.full((usable_edges.numel(), maximum), float("inf"), device=compute)
        padded_cosine = torch.full((usable_edges.numel(), maximum), -1.0, device=compute)
        owner_device = owner.to(compute)
        within_device = within.to(compute)
        padded_residual[owner_device[finite], within_device[finite]] = residual[finite]
        padded_cosine[
            owner_device[geometric], within_device[geometric]
        ] = cosine[geometric]
        ordered_residual = padded_residual.sort(1).values
        finite_counts = torch.isfinite(ordered_residual).sum(1)
        has_finite = finite_counts > 0
        median_index = ((finite_counts - 1).clamp_min(0) // 2).long()
        median_residual[usable_edges[has_finite.cpu()]] = ordered_residual[
            has_finite, median_index[has_finite]
        ].cpu()
        best_cosine[usable_edges] = padded_cosine.max(1).values.cpu()
        for mask, output in (
            (geometric, transport_family_count),
            (descriptor, descriptor_family_count),
        ):
            if bool(mask.any()):
                pairs = torch.unique(
                    torch.stack(
                        [owner[mask.cpu()], families[mapping_query[mask.cpu()]]], dim=1
                    ),
                    dim=0,
                )
                output[usable_edges] = torch.bincount(
                    pairs[:, 0], minlength=usable_edges.numel()
                )
    return {
        "schema": "lafgs_v19_track_observation_consensus",
        "version": 1,
        "candidate_offsets": offsets,
        "candidate_anchor_rows": candidates,
        "transport_view_family_count": transport_family_count,
        "transport_median_residual_px": median_residual,
        "descriptor_view_family_count": descriptor_family_count,
        "descriptor_best_cosine": best_cosine,
        "geometric_inlier_px": float(geometric_inlier_px),
        "descriptor_inlier_cosine": float(descriptor_inlier_cosine),
        "uses_deployed_metric": False,
        "native_descriptor_only": True,
    }


def assign_track_extension_truth(
    *,
    candidate_graph: Mapping,
    consensus: Mapping,
    equivalence_class_ids: torch.Tensor,
    tier: TrackExtensionTier,
) -> dict:
    """Assign only when every eligible candidate belongs to one identity class."""

    tier.validate()
    offsets = torch.as_tensor(candidate_graph["candidate_offsets"]).long().cpu()
    candidates = torch.as_tensor(candidate_graph["candidate_anchor_rows"]).long().cpu()
    row_count = int(candidate_graph["row_count"])
    _validate_offsets(offsets, candidates.numel(), row_count)
    if not torch.equal(offsets, torch.as_tensor(consensus["candidate_offsets"]).long().cpu()) or not torch.equal(candidates, torch.as_tensor(consensus["candidate_anchor_rows"]).long().cpu()):
        raise ValueError("Track-extension candidate and consensus graphs differ")
    equivalence = torch.as_tensor(equivalence_class_ids).long().cpu().reshape(-1)
    if candidates.numel() and int(candidates.max()) >= equivalence.numel():
        raise ValueError("Track-extension candidate is outside the identity registry")
    reprojection = torch.as_tensor(candidate_graph["query_reprojection_residual_px"]).float().cpu()
    depth = torch.as_tensor(candidate_graph["query_normalized_depth_residual"]).float().cpu()
    projection_std = torch.as_tensor(candidate_graph["query_projection_std_px"]).float().cpu()
    families = torch.as_tensor(consensus["transport_view_family_count"]).long().cpu()
    transport = torch.as_tensor(consensus["transport_median_residual_px"]).float().cpu()
    descriptor_families = torch.as_tensor(consensus["descriptor_view_family_count"]).long().cpu()
    cosine = torch.as_tensor(consensus["descriptor_best_cosine"]).float().cpu()
    if not (
        reprojection.shape
        == depth.shape
        == projection_std.shape
        == families.shape
        == transport.shape
        == descriptor_families.shape
        == cosine.shape
        == candidates.shape
    ):
        raise ValueError("Track-extension evidence fields do not align")
    eligible = (
        torch.isfinite(reprojection)
        & torch.isfinite(depth)
        & torch.isfinite(projection_std)
        & torch.isfinite(transport)
        & torch.isfinite(cosine)
        & (reprojection <= tier.maximum_query_reprojection_px)
        & (depth <= tier.maximum_query_normalized_depth_residual)
        & (projection_std <= tier.maximum_query_projection_std_px)
        & (transport <= tier.maximum_transport_median_residual_px)
        & (families >= tier.minimum_transport_view_families)
        & (cosine >= tier.minimum_descriptor_cosine)
        & (descriptor_families >= tier.minimum_descriptor_view_families)
    )
    valid_rows = torch.as_tensor(candidate_graph["query_valid"]).bool().cpu()
    status = torch.full((row_count,), TRUTH_NONE, dtype=torch.int8)
    status[~valid_rows] = TRUTH_INVALID
    edge_rows = torch.repeat_interleave(
        torch.arange(row_count), offsets[1:] - offsets[:-1]
    )
    projection_count = offsets[1:] - offsets[:-1]
    class_ids = equivalence[candidates]
    class_min = torch.full((row_count,), torch.iinfo(torch.long).max)
    class_max = torch.full((row_count,), torch.iinfo(torch.long).min)
    if edge_rows.numel():
        class_min.scatter_reduce_(
            0, edge_rows, class_ids, reduce="amin", include_self=True
        )
        class_max.scatter_reduce_(
            0, edge_rows, class_ids, reduce="amax", include_self=True
        )
    projection_unique = (projection_count > 0) & (class_min == class_max)
    projection_ambiguous = valid_rows & (projection_count > 0) & ~projection_unique
    status[projection_ambiguous] = TRUTH_AMBIGUOUS

    # Missing Track-bank support is an abstention, not evidence that a
    # projected identity is false.  Consensus can confirm only rows whose
    # exhaustive projection candidate set was already identity-unique.
    selected_edges = eligible & projection_unique[edge_rows] & valid_rows[edge_rows]
    if bool(selected_edges.any()):
        encoded = edge_rows[selected_edges] * equivalence.numel() + candidates[selected_edges]
        encoded = torch.unique(encoded, sorted=True)
        selected_rows = torch.div(encoded, equivalence.numel(), rounding_mode="floor")
        selected_anchors = encoded.remainder(equivalence.numel())
    else:
        selected_rows = torch.empty(0, dtype=torch.long)
        selected_anchors = torch.empty(0, dtype=torch.long)
    selected_count = torch.bincount(selected_rows, minlength=row_count)
    decisive = valid_rows & projection_unique & (selected_count > 0)
    status[decisive & (selected_count == 1)] = TRUTH_UNIQUE
    status[decisive & (selected_count > 1)] = TRUTH_EQUIVALENT
    truth_offsets = torch.cat(
        (torch.zeros(1, dtype=torch.long), selected_count.cumsum(0))
    )
    return {
        "schema": "lafgs_v19_track_extension_truth",
        "version": 1,
        "row_count": row_count,
        "truth_status": status,
        "truth_status_names": TRUTH_STATUS_NAMES,
        "truth_offsets": truth_offsets,
        "truth_anchor_rows": selected_anchors,
        "status_counts": {
            name: int((status == code).sum())
            for code, name in enumerate(TRUTH_STATUS_NAMES)
        },
        "tier": asdict(tier),
        "uses_descriptor_scores": True,
        "uses_deployed_metric": False,
        "uses_topl_candidates": False,
        "feedback_enters_track_registry": False,
        "reference_source": "mapping_observation_track_membership",
        "reference_available_for_novel_query": False,
    }


__all__ = [
    "TRACK_EXTENSION_TIERS",
    "TrackExtensionTier",
    "assign_track_extension_truth",
    "full_map_projection_candidate_graph",
    "prepare_track_observation_bank",
    "track_observation_consensus",
]
