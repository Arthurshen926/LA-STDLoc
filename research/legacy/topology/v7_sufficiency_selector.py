"""V7's deterministic, provenance-complete map sufficiency selector."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math

import numpy as np
import torch


EdgeRegistry = Sequence[Mapping[int, Sequence[int]]]
SparsePoseInformation = Sequence[Mapping[int, torch.Tensor]]
COMPLETION_LAYERS = ("matching", "image_cell", "view_family", "depth_range")


class CompactEdgeRegistry(Sequence[Mapping[int, Sequence[int]]]):
    """CSR-backed edge registry that does not duplicate million-edge graphs."""

    def __init__(
        self,
        offsets: torch.Tensor,
        query_indices: torch.Tensor,
        identities: torch.Tensor,
    ) -> None:
        self.offsets = torch.as_tensor(offsets).long().cpu().reshape(-1)
        self.query_indices = torch.as_tensor(query_indices).long().cpu().reshape(-1)
        self.identities = torch.as_tensor(identities).long().cpu().reshape(-1)
        if self.offsets.numel() < 2 or int(self.offsets[0]) != 0:
            raise ValueError("compact edge offsets must start at zero")
        if bool((self.offsets[1:] < self.offsets[:-1]).any()):
            raise ValueError("compact edge offsets must be monotone")
        if int(self.offsets[-1]) != self.query_indices.numel():
            raise ValueError("compact edge offsets do not cover query indices")
        if self.identities.numel() != self.query_indices.numel():
            raise ValueError("compact edge identities do not align")

    def __len__(self) -> int:
        return self.offsets.numel() - 1

    def __getitem__(self, anchor: int | slice) -> Mapping[int, Sequence[int]]:
        if isinstance(anchor, slice):
            return [self[index] for index in range(*anchor.indices(len(self)))]
        anchor = int(anchor)
        if anchor < 0:
            anchor += len(self)
        if not 0 <= anchor < len(self):
            raise IndexError(anchor)
        start, end = int(self.offsets[anchor]), int(self.offsets[anchor + 1])
        result: dict[int, list[int]] = {}
        for query, identity in zip(
            self.query_indices[start:end].tolist(),
            self.identities[start:end].tolist(),
        ):
            if int(identity) < 0:
                continue
            result.setdefault(int(query), []).append(int(identity))
        return {query: tuple(sorted(set(values))) for query, values in result.items()}


class LazyPoseInformation(Sequence[Mapping[int, torch.Tensor]]):
    """Compute full-SE(3) Fisher contributions from mapping rays on demand."""

    def __init__(
        self,
        anchor_xyz: torch.Tensor,
        offsets: torch.Tensor,
        query_indices: torch.Tensor,
        intrinsics: torch.Tensor,
        poses_w2c: torch.Tensor,
        *,
        measurement_variance_px2: float = 1.0,
    ) -> None:
        self.anchor_xyz = torch.as_tensor(anchor_xyz).double().cpu()
        self.offsets = torch.as_tensor(offsets).long().cpu().reshape(-1)
        self.query_indices = torch.as_tensor(query_indices).long().cpu().reshape(-1)
        self.intrinsics = torch.as_tensor(intrinsics).double().cpu()
        self.poses_w2c = torch.as_tensor(poses_w2c).double().cpu()
        self.variance = float(measurement_variance_px2)
        if self.anchor_xyz.ndim != 2 or self.anchor_xyz.shape[1] != 3:
            raise ValueError("lazy pose xyz must have shape [N,3]")
        if self.offsets.shape != (self.anchor_xyz.shape[0] + 1,):
            raise ValueError("lazy pose offsets do not align with anchors")
        if int(self.offsets[-1]) != self.query_indices.numel():
            raise ValueError("lazy pose offsets do not cover query indices")
        if self.intrinsics.ndim != 3 or self.intrinsics.shape[1:] != (3, 3):
            raise ValueError("lazy pose intrinsics must have shape [Q,3,3]")
        if self.poses_w2c.shape != (self.intrinsics.shape[0], 4, 4):
            raise ValueError("lazy pose camera registries differ")
        if self.variance <= 0.0:
            raise ValueError("measurement variance must be positive")

    def __len__(self) -> int:
        return self.anchor_xyz.shape[0]

    def __getitem__(self, anchor: int | slice) -> Mapping[int, torch.Tensor]:
        if isinstance(anchor, slice):
            return [self[index] for index in range(*anchor.indices(len(self)))]
        anchor = int(anchor)
        if anchor < 0:
            anchor += len(self)
        if not 0 <= anchor < len(self):
            raise IndexError(anchor)
        start, end = int(self.offsets[anchor]), int(self.offsets[anchor + 1])
        result: dict[int, torch.Tensor] = {}
        xyz = self.anchor_xyz[anchor]
        homogeneous = torch.cat((xyz, xyz.new_ones(1)))
        for query in self.query_indices[start:end].tolist():
            query = int(query)
            camera = (self.poses_w2c[query] @ homogeneous)[:3]
            x, y, z = camera
            if not bool(torch.isfinite(camera).all()) or float(z) <= 1e-8:
                continue
            intrinsic = self.intrinsics[query]
            dproj = camera.new_zeros((2, 3))
            dproj[0, 0] = intrinsic[0, 0] / z
            dproj[0, 2] = -intrinsic[0, 0] * x / z.square()
            dproj[1, 1] = intrinsic[1, 1] / z
            dproj[1, 2] = -intrinsic[1, 1] * y / z.square()
            skew = camera.new_tensor([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
            camera_jacobian = torch.cat(
                (torch.eye(3, dtype=torch.float64), -skew), dim=1
            )
            jacobian = dproj @ camera_jacobian
            contribution = jacobian.T @ jacobian / self.variance
            result[query] = (
                result.get(query, torch.zeros(6, 6, dtype=torch.float64)) + contribution
            )
        return result


class CompactPoseInformation(Sequence[Mapping[int, torch.Tensor]]):
    """CSR-backed full-SE(3) Fisher contributions precomputed per mapping edge."""

    def __init__(
        self,
        offsets: torch.Tensor,
        query_indices: torch.Tensor,
        contributions: torch.Tensor,
    ) -> None:
        self.offsets = torch.as_tensor(offsets).long().cpu().reshape(-1)
        self.query_indices = torch.as_tensor(query_indices).long().cpu().reshape(-1)
        self.contributions = torch.as_tensor(contributions).float().cpu()
        if self.offsets.numel() < 2 or int(self.offsets[0]) != 0:
            raise ValueError("compact pose offsets must start at zero")
        if int(self.offsets[-1]) != self.query_indices.numel():
            raise ValueError("compact pose offsets do not cover query indices")
        if self.contributions.shape != (self.query_indices.numel(), 6, 6):
            raise ValueError("compact pose contributions must have shape [E,6,6]")

    def __len__(self) -> int:
        return self.offsets.numel() - 1

    def __getitem__(self, anchor: int | slice) -> Mapping[int, torch.Tensor]:
        if isinstance(anchor, slice):
            return [self[index] for index in range(*anchor.indices(len(self)))]
        anchor = int(anchor)
        if anchor < 0:
            anchor += len(self)
        if not 0 <= anchor < len(self):
            raise IndexError(anchor)
        start, end = int(self.offsets[anchor]), int(self.offsets[anchor + 1])
        result: dict[int, torch.Tensor] = {}
        for edge in range(start, end):
            query = int(self.query_indices[edge])
            result[query] = (
                result.get(query, torch.zeros(6, 6)) + self.contributions[edge]
            )
        return result


@dataclass(frozen=True)
class EligibilityThresholds:
    minimum_geometry_reliability: float
    minimum_observation_count: int
    minimum_view_family_count: int
    maximum_descriptor_dispersion: float
    maximum_reprojection_error: float
    maximum_covariance_trace: float
    minimum_parallax: float


@dataclass(frozen=True)
class SufficiencyTargets:
    precision_matching_rank: int | Sequence[int]
    completion_matching_rank: int | Sequence[int]
    image_cells: int | Sequence[int]
    view_families: int | Sequence[int]
    depth_ranges: int | Sequence[int]
    pose_logdet: float | Sequence[float]
    pose_minimum_eigenvalue: float | Sequence[float]
    maximum_anchors: int
    pose_damping: float = 1e-6


class _MatchingState:
    def __init__(self, query_count: int, edges: EdgeRegistry) -> None:
        self.edges = edges
        self.selected: set[int] = set()
        self.row_to_anchor = [dict() for _ in range(query_count)]
        self.anchor_to_row = [dict() for _ in range(query_count)]

    def _augment(
        self,
        query: int,
        anchor: int,
        row_to_anchor: dict[int, int],
        anchor_to_row: dict[int, int],
        seen_anchors: set[int],
        seen_rows: set[int],
    ) -> bool:
        if anchor in seen_anchors:
            return False
        seen_anchors.add(anchor)
        for row in self.edges[anchor].get(query, ()):
            row = int(row)
            if row in seen_rows:
                continue
            seen_rows.add(row)
            previous = row_to_anchor.get(row)
            if previous is None or self._augment(
                query,
                previous,
                row_to_anchor,
                anchor_to_row,
                seen_anchors,
                seen_rows,
            ):
                row_to_anchor[row] = anchor
                anchor_to_row[anchor] = row
                return True
        return False

    def would_augment(self, anchor: int, query: int) -> bool:
        return anchor not in self.selected and self._augment(
            int(query),
            int(anchor),
            dict(self.row_to_anchor[int(query)]),
            dict(self.anchor_to_row[int(query)]),
            set(),
            set(),
        )

    def add(self, anchor: int) -> None:
        anchor = int(anchor)
        if anchor in self.selected:
            return
        for query in self.edges[anchor]:
            self._augment(
                int(query),
                anchor,
                self.row_to_anchor[int(query)],
                self.anchor_to_row[int(query)],
                set(),
                set(),
            )
        self.selected.add(anchor)

    @property
    def counts(self) -> np.ndarray:
        return np.asarray([len(value) for value in self.row_to_anchor], dtype=np.int64)


class _UniqueCoverageState:
    def __init__(self, query_count: int, edges: EdgeRegistry) -> None:
        self.edges = edges
        self.selected: set[int] = set()
        self.covered = [set() for _ in range(query_count)]

    def would_augment(self, anchor: int, query: int) -> bool:
        return bool(
            set(self.edges[int(anchor)].get(int(query), ())) - self.covered[int(query)]
        )

    def add(self, anchor: int) -> None:
        anchor = int(anchor)
        if anchor in self.selected:
            return
        for query, identities in self.edges[anchor].items():
            self.covered[int(query)].update(int(value) for value in identities)
        self.selected.add(anchor)

    @property
    def counts(self) -> np.ndarray:
        return np.asarray([len(value) for value in self.covered], dtype=np.int64)


def _vector(value: int | Sequence[int], query_count: int, name: str) -> np.ndarray:
    result = np.broadcast_to(np.asarray(value, dtype=np.int64), (query_count,)).copy()
    if bool((result < 0).any()):
        raise ValueError(f"{name} targets must be non-negative")
    return result


def _float_vector(
    value: float | Sequence[float], query_count: int, name: str
) -> torch.Tensor:
    result = torch.as_tensor(
        np.broadcast_to(np.asarray(value, dtype=np.float64), (query_count,)).copy()
    )
    if not bool(torch.isfinite(result).all()):
        raise ValueError(f"{name} targets must be finite")
    return result


def _aligned_tensor(value: torch.Tensor, count: int, name: str) -> torch.Tensor:
    result = torch.as_tensor(value).reshape(-1)
    if result.numel() != count:
        raise ValueError(f"{name} does not align with candidates")
    return result


def eligibility_mask(
    *,
    geometry_reliability: torch.Tensor,
    observation_count: torch.Tensor,
    view_family_count: torch.Tensor,
    descriptor_dispersion: torch.Tensor,
    reprojection_error: torch.Tensor,
    covariance_trace: torch.Tensor,
    parallax: torch.Tensor,
    render_artifact_supported: torch.Tensor,
    lineage_complete: torch.Tensor,
    thresholds: EligibilityThresholds,
) -> tuple[torch.Tensor, dict[str, int]]:
    """Apply only preregistered hard exclusions; no weighted eligibility score."""

    count = torch.as_tensor(geometry_reliability).numel()
    fields = {
        "geometry_reliability": _aligned_tensor(
            geometry_reliability, count, "geometry reliability"
        ).float(),
        "observation_count": _aligned_tensor(
            observation_count, count, "observation count"
        ).long(),
        "view_family_count": _aligned_tensor(
            view_family_count, count, "view-family count"
        ).long(),
        "descriptor_dispersion": _aligned_tensor(
            descriptor_dispersion, count, "descriptor dispersion"
        ).float(),
        "reprojection_error": _aligned_tensor(
            reprojection_error, count, "reprojection error"
        ).float(),
        "covariance_trace": _aligned_tensor(
            covariance_trace, count, "covariance trace"
        ).float(),
        "parallax": _aligned_tensor(parallax, count, "parallax").float(),
        "render_artifact_supported": _aligned_tensor(
            render_artifact_supported, count, "render artifact support"
        ).bool(),
        "lineage_complete": _aligned_tensor(
            lineage_complete, count, "lineage completeness"
        ).bool(),
    }
    finite = torch.ones(count, dtype=torch.bool)
    for name in (
        "geometry_reliability",
        "descriptor_dispersion",
        "reprojection_error",
        "covariance_trace",
        "parallax",
    ):
        finite &= torch.isfinite(fields[name])
    failures = {
        "non_finite": ~finite,
        "geometry_reliability": fields["geometry_reliability"]
        < thresholds.minimum_geometry_reliability,
        "observation_count": fields["observation_count"]
        < thresholds.minimum_observation_count,
        "view_family_count": fields["view_family_count"]
        < thresholds.minimum_view_family_count,
        "descriptor_dispersion": fields["descriptor_dispersion"]
        > thresholds.maximum_descriptor_dispersion,
        "reprojection_error": fields["reprojection_error"]
        > thresholds.maximum_reprojection_error,
        "covariance_trace": fields["covariance_trace"]
        > thresholds.maximum_covariance_trace,
        "parallax": fields["parallax"] < thresholds.minimum_parallax,
        "render_artifact_support": fields["render_artifact_supported"],
        "lineage_incomplete": ~fields["lineage_complete"],
    }
    eligible = torch.ones(count, dtype=torch.bool)
    for failed in failures.values():
        eligible &= ~failed
    return eligible, {name: int(value.sum()) for name, value in failures.items()}


@torch.inference_mode()
def reconstruct_mapping_candidate_evidence(
    *,
    anchor_xyz: torch.Tensor,
    anchor_features: torch.Tensor,
    observation_offsets: torch.Tensor,
    query_indices: torch.Tensor,
    keypoint_indices: torch.Tensor,
    query_names: Sequence[str],
    query_bins: torch.Tensor,
    rendered_feature_records: Mapping[str, Mapping],
    device: str = "cuda",
    grid_shape: tuple[int, int] = (4, 4),
) -> dict:
    """Reconstruct omitted P3 quality fields from frozen mapping observations."""

    xyz_cpu = torch.as_tensor(anchor_xyz).float().cpu()
    features_cpu = torch.as_tensor(anchor_features).float().cpu()
    offsets = torch.as_tensor(observation_offsets).long().cpu()
    queries = torch.as_tensor(query_indices).long().cpu()
    rows = torch.as_tensor(keypoint_indices).long().cpu()
    bins = torch.as_tensor(query_bins).long().cpu()
    count = xyz_cpu.shape[0]
    edge_count = queries.numel()
    if (
        xyz_cpu.shape != (count, 3)
        or features_cpu.ndim != 2
        or features_cpu.shape[0] != count
    ):
        raise ValueError("candidate xyz and descriptors differ")
    if (
        offsets.shape != (count + 1,)
        or int(offsets[-1]) != edge_count
        or rows.shape != queries.shape
    ):
        raise ValueError("candidate observation CSR differs")
    if list(query_names) != list(rendered_feature_records):
        raise ValueError("mapping feature registry differs from candidate registry")
    if bins.shape != (len(query_names),):
        raise ValueError("mapping query bins differ")
    if (
        edge_count == 0
        or int(queries.min()) < 0
        or int(queries.max()) >= len(query_names)
    ):
        raise ValueError("candidate observations contain invalid mapping queries")

    target = torch.device(device)
    if target.type != "cuda":
        raise ValueError("formal evidence reconstruction requires CUDA")
    xyz = xyz_cpu.to(target)
    features = torch.nn.functional.normalize(features_cpu.to(target), dim=1)
    observation_count = offsets[1:] - offsets[:-1]
    anchor_for_edge_cpu = torch.repeat_interleave(
        torch.arange(count), observation_count
    )
    descriptor_sum = torch.zeros(count, device=target)
    reprojection_sum = torch.zeros(count, device=target)
    ray_sum = torch.zeros(count, 3, device=target)
    invalid_projection_count = torch.zeros(count, device=target)
    image_cells = torch.empty(edge_count, dtype=torch.long)
    depth_ranges = torch.empty(edge_count, dtype=torch.long)
    pose_contributions = torch.empty(edge_count, 6, 6, dtype=torch.float32)
    ordered = torch.argsort(queries, stable=True)
    query_offsets = torch.cat(
        (
            torch.zeros(1, dtype=torch.long),
            torch.bincount(queries, minlength=len(query_names)).cumsum(0),
        )
    )
    grid_rows, grid_columns = map(int, grid_shape)
    if grid_rows <= 0 or grid_columns <= 0:
        raise ValueError("image grid must be positive")

    for query, name in enumerate(query_names):
        positions = ordered[int(query_offsets[query]) : int(query_offsets[query + 1])]
        if positions.numel() == 0:
            continue
        record = rendered_feature_records[name]
        required = {
            "native_keypoints",
            "native_descriptors",
            "native_K",
            "pose_w2c",
            "native_input_hw",
        }
        if not required <= set(record):
            raise ValueError(f"mapping feature record is incomplete: {name}")
        anchor_rows = anchor_for_edge_cpu[positions].to(target)
        keypoint_rows = rows[positions]
        maximum_row = int(keypoint_rows.max())
        keypoints_all = torch.as_tensor(record["native_keypoints"]).float()
        descriptors_all = torch.as_tensor(record["native_descriptors"]).float()
        if (
            maximum_row >= keypoints_all.shape[0]
            or descriptors_all.shape[0] != keypoints_all.shape[0]
        ):
            raise ValueError(
                f"candidate keypoint row is outside mapping record: {name}"
            )
        keypoints = keypoints_all[keypoint_rows].to(target)
        descriptors = torch.nn.functional.normalize(
            descriptors_all[keypoint_rows].to(target), dim=1
        )
        descriptor_sum.index_add_(
            0,
            anchor_rows,
            1.0 - (descriptors * features[anchor_rows]).sum(1).clamp(-1, 1),
        )

        intrinsic = torch.as_tensor(record["native_K"], device=target).float()
        pose = torch.as_tensor(record["pose_w2c"], device=target).float()
        homogeneous = torch.cat(
            (xyz[anchor_rows], torch.ones(anchor_rows.numel(), 1, device=target)), dim=1
        )
        camera = (pose @ homogeneous.T).T[:, :3]
        z = camera[:, 2]
        valid_projection = (
            torch.isfinite(camera).all(1) & torch.isfinite(z) & (z > 1e-8)
        )
        invalid_projection_count.index_add_(0, anchor_rows, (~valid_projection).float())
        safe_z = torch.where(valid_projection, z, torch.ones_like(z))
        projected = torch.stack(
            (
                intrinsic[0, 0] * camera[:, 0] / safe_z + intrinsic[0, 2],
                intrinsic[1, 1] * camera[:, 1] / safe_z + intrinsic[1, 2],
            ),
            dim=1,
        )
        reprojection = torch.linalg.vector_norm(projected - keypoints, dim=1)
        reprojection = torch.where(
            valid_projection, reprojection, torch.full_like(reprojection, 1e6)
        )
        reprojection_sum.index_add_(0, anchor_rows, reprojection)
        dproj = camera.new_zeros((camera.shape[0], 2, 3))
        dproj[:, 0, 0] = intrinsic[0, 0] / safe_z
        dproj[:, 0, 2] = -intrinsic[0, 0] * camera[:, 0] / safe_z.square()
        dproj[:, 1, 1] = intrinsic[1, 1] / safe_z
        dproj[:, 1, 2] = -intrinsic[1, 1] * camera[:, 1] / safe_z.square()
        skew = camera.new_zeros((camera.shape[0], 3, 3))
        skew[:, 0, 1] = -camera[:, 2]
        skew[:, 0, 2] = camera[:, 1]
        skew[:, 1, 0] = camera[:, 2]
        skew[:, 1, 2] = -camera[:, 0]
        skew[:, 2, 0] = -camera[:, 1]
        skew[:, 2, 1] = camera[:, 0]
        identity = torch.eye(3, device=target)[None].expand(camera.shape[0], -1, -1)
        jacobian = dproj @ torch.cat((identity, -skew), dim=2)
        fisher = jacobian.transpose(1, 2) @ jacobian
        fisher[~valid_projection] = 0
        pose_contributions[positions] = fisher.cpu()
        center = -(pose[:3, :3].T @ pose[:3, 3])
        rays = torch.nn.functional.normalize(xyz[anchor_rows] - center, dim=1)
        ray_sum.index_add_(0, anchor_rows, rays)

        height, width = map(int, torch.as_tensor(record["native_input_hw"]).tolist())
        columns = (
            torch.floor(keypoints[:, 0] / width * grid_columns)
            .long()
            .clamp(0, grid_columns - 1)
        )
        grid_y = (
            torch.floor(keypoints[:, 1] / height * grid_rows)
            .long()
            .clamp(0, grid_rows - 1)
        )
        image_cells[positions] = (grid_y * grid_columns + columns).cpu()
        valid_depth = z[valid_projection]
        median_depth = (
            valid_depth.median().clamp_min(1e-6)
            if valid_depth.numel()
            else z.new_tensor(1.0)
        )
        ratio = safe_z / median_depth
        depth_identity = torch.bucketize(
            ratio, torch.tensor([0.5, 1.0, 2.0], device=target)
        )
        depth_identity[~valid_projection] = -1
        depth_ranges[positions] = depth_identity.cpu()

    denominator = observation_count.clamp_min(1).to(target).float()
    descriptor_dispersion = (descriptor_sum / denominator).cpu()
    reprojection_error = (reprojection_sum / denominator).cpu()
    mean_ray_norm = torch.linalg.vector_norm(
        ray_sum / denominator[:, None], dim=1
    ).clamp(0, 1)
    parallax_dispersion_deg = torch.rad2deg(torch.acos(mean_ray_norm)).cpu()
    family = bins[queries]
    family_base = int(bins.max()) + 1
    unique_pairs = torch.unique(anchor_for_edge_cpu * family_base + family)
    view_family_count = torch.bincount(
        torch.div(unique_pairs, family_base, rounding_mode="floor"), minlength=count
    )
    if (
        not bool(torch.isfinite(descriptor_dispersion).all())
        or not bool(torch.isfinite(reprojection_error).all())
        or not bool(torch.isfinite(parallax_dispersion_deg).all())
    ):
        raise ValueError("reconstructed mapping candidate evidence is non-finite")
    return {
        "schema": "lafgs_v7_reconstructed_mapping_candidate_evidence",
        "version": 1,
        "observation_count": observation_count,
        "view_family_count": view_family_count,
        "descriptor_dispersion": descriptor_dispersion,
        "reprojection_error_px_mean": reprojection_error,
        "ray_angular_dispersion_deg": parallax_dispersion_deg,
        "invalid_projection_count": invalid_projection_count.long().cpu(),
        "image_cell_identities": image_cells,
        "depth_range_identities": depth_ranges,
        "view_family_identities": family,
        "pose_information_contributions": pose_contributions,
        "contract": {
            "uses_source_mapping_rgb": False,
            "uses_test_queries": False,
            "descriptor_source": "frozen_rendered_mapping_superpoint_rows",
            "geometry_source": "pure_ray_anchor_xyz_and_mapping_camera_calibration",
            "gaussian_depth_used_for_pnp_xyz": False,
            "render_artifact_support": "upstream_render_valid_before_nms_only",
        },
    }


def select_v7_sufficiency(
    *,
    anchor_ids: torch.Tensor,
    reliability: torch.Tensor,
    eligible: torch.Tensor,
    layer_edges: Mapping[str, EdgeRegistry],
    pose_information: SparsePoseInformation,
    query_count: int,
    targets: SufficiencyTargets,
    previous_active: torch.Tensor | None = None,
    previous_clean_critical: torch.Tensor | None = None,
    active_set_change_fraction: float = 0.10,
) -> dict:
    """Select one active map for initialization and every later update."""

    anchor_ids = torch.as_tensor(anchor_ids).long().reshape(-1)
    count = anchor_ids.numel()
    reliability = _aligned_tensor(reliability, count, "reliability").float()
    eligible = _aligned_tensor(eligible, count, "eligibility").bool()
    if len(set(anchor_ids.tolist())) != count:
        raise ValueError("anchor IDs must be unique")
    if tuple(layer_edges) != COMPLETION_LAYERS:
        raise ValueError(f"layer order must be exactly {COMPLETION_LAYERS}")
    if any(len(edges) != count for edges in layer_edges.values()):
        raise ValueError("layer edge registries do not align with candidates")
    if len(pose_information) != count:
        raise ValueError("pose information does not align with candidates")
    query_count = int(query_count)
    if query_count <= 0 or targets.maximum_anchors <= 0:
        raise ValueError("query count and maximum anchors must be positive")
    if not 0.0 <= float(active_set_change_fraction) <= 1.0:
        raise ValueError("active-set change fraction must be in [0,1]")
    if not bool(torch.isfinite(reliability[eligible]).all()):
        raise ValueError("eligible reliability values must be finite")

    layer_targets = {
        "matching": _vector(targets.completion_matching_rank, query_count, "matching"),
        "image_cell": _vector(targets.image_cells, query_count, "image-cell"),
        "view_family": _vector(targets.view_families, query_count, "view-family"),
        "depth_range": _vector(targets.depth_ranges, query_count, "depth-range"),
    }
    precision_target = _vector(
        targets.precision_matching_rank, query_count, "precision matching"
    )
    if bool((precision_target > layer_targets["matching"]).any()):
        raise ValueError("precision matching target cannot exceed completion target")
    pose_logdet_target = _float_vector(targets.pose_logdet, query_count, "pose logdet")
    pose_minimum_target = _float_vector(
        targets.pose_minimum_eigenvalue, query_count, "pose minimum-eigenvalue"
    )

    states = {
        name: (
            _MatchingState(query_count, edges)
            if name == "matching"
            else _UniqueCoverageState(query_count, edges)
        )
        for name, edges in layer_edges.items()
    }
    sortable_reliability = torch.nan_to_num(reliability, nan=-torch.inf)
    order = sorted(
        range(count),
        key=lambda row: (-float(sortable_reliability[row]), int(anchor_ids[row])),
    )
    selected: list[int] = []
    selected_set: set[int] = set()
    reason: dict[int, str] = {}

    def add(row: int, why: str) -> None:
        if row in selected_set or len(selected) >= int(targets.maximum_anchors):
            return
        selected.append(row)
        selected_set.add(row)
        reason[row] = why
        for state in states.values():
            state.add(row)

    def complete(
        state: _MatchingState | _UniqueCoverageState, target: np.ndarray, why: str
    ) -> None:
        counts = state.counts
        for row in order:
            if bool((counts >= target).all()) or len(selected) >= int(
                targets.maximum_anchors
            ):
                break
            if row in selected_set or not bool(eligible[row]):
                continue
            if any(
                counts[q] < target[q] and state.would_augment(row, q)
                for q in state.edges[row]
            ):
                add(row, why)
                counts = state.counts

    complete(states["matching"], precision_target, "precision_core")
    for layer in COMPLETION_LAYERS:
        complete(states[layer], layer_targets[layer], f"{layer}_completion")

    pose_base = torch.eye(6, dtype=torch.float64).repeat(query_count, 1, 1) * float(
        targets.pose_damping
    )
    for row in selected:
        for query, matrix in pose_information[row].items():
            value = torch.as_tensor(matrix).double()
            if value.shape != (6, 6) or not bool(torch.isfinite(value).all()):
                raise ValueError("pose information matrices must be finite 6x6 tensors")
            pose_base[int(query)] += value

    def pose_scores(matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        symmetric = (matrix + matrix.transpose(-1, -2)) * 0.5
        eigenvalues = torch.linalg.eigvalsh(symmetric)
        return eigenvalues.clamp_min(1e-12).log().sum(-1), eigenvalues[:, 0]

    while len(selected) < int(targets.maximum_anchors):
        logdet, minimum = pose_scores(pose_base)
        deficient = (logdet < pose_logdet_target) | (minimum < pose_minimum_target)
        if not bool(deficient.any()):
            break
        best: tuple[float, float, int, int] | None = None
        for row in order:
            if row in selected_set or not bool(eligible[row]):
                continue
            contributions = pose_information[row]
            if not contributions:
                continue
            affected = sorted(
                int(query) for query in contributions if bool(deficient[int(query)])
            )
            if not affected:
                continue
            affected_tensor = torch.tensor(affected, dtype=torch.long)
            proposed = pose_base[affected_tensor] + torch.stack(
                [torch.as_tensor(contributions[query]).double() for query in affected]
            )
            proposed_logdet, proposed_minimum = pose_scores(proposed)
            current_minimum = minimum[affected_tensor]
            current_logdet = logdet[affected_tensor]
            minimum_gain = torch.minimum(
                proposed_minimum,
                pose_minimum_target[affected_tensor],
            ) - torch.minimum(
                current_minimum,
                pose_minimum_target[affected_tensor],
            )
            logdet_gain = torch.minimum(
                proposed_logdet,
                pose_logdet_target[affected_tensor],
            ) - torch.minimum(
                current_logdet,
                pose_logdet_target[affected_tensor],
            )
            key = (
                float(minimum_gain.clamp_min(0).sum()),
                float(logdet_gain.clamp_min(0).sum()),
                -int(anchor_ids[row]),
                row,
            )
            if (key[0] > 0.0 or key[1] > 0.0) and (best is None or key[:3] > best[:3]):
                best = key
        if best is None:
            break
        row = best[3]
        add(row, "pose_redundancy_completion")
        for query, matrix in pose_information[row].items():
            pose_base[int(query)] += torch.as_tensor(matrix).double()

    desired_set = set(selected)
    previous_rows: set[int] = set()
    critical_rows: set[int] = set()
    applied_changes: list[dict] = []
    budget = None
    if previous_active is not None:
        active_ids = set(torch.as_tensor(previous_active).long().reshape(-1).tolist())
        id_to_row = {int(value): row for row, value in enumerate(anchor_ids.tolist())}
        unknown = active_ids - set(id_to_row)
        if unknown:
            raise ValueError("previous active state contains unknown anchor IDs")
        previous_rows = {id_to_row[value] for value in active_ids}
        if previous_clean_critical is not None:
            critical_ids = set(
                torch.as_tensor(previous_clean_critical).long().reshape(-1).tolist()
            )
            if not critical_ids <= active_ids:
                raise ValueError(
                    "clean-critical anchors must belong to previous active state"
                )
            critical_rows = {id_to_row[value] for value in critical_ids}
            desired_set |= critical_rows
            for row in critical_rows:
                reason.setdefault(row, "previous_clean_critical_protection")
        budget = math.floor(float(active_set_change_fraction) * len(previous_rows))
        transitioned = set(previous_rows)
        removals = sorted(
            previous_rows - desired_set,
            key=lambda row: (
                bool(eligible[row]),
                float(reliability[row]),
                int(anchor_ids[row]),
            ),
        )
        additions = [row for row in selected if row not in previous_rows]
        # Safety-first transition: spend the bounded budget on desired additions
        # before optional removals, avoiding abrupt loss of RANSAC redundancy.
        changes = [("add", row) for row in additions] + [
            ("remove", row) for row in removals
        ]
        for action, row in changes[:budget]:
            if action == "remove":
                transitioned.remove(row)
            else:
                transitioned.add(row)
            applied_changes.append(
                {"action": action, "anchor_id": int(anchor_ids[row])}
            )
        desired_set = transitioned

    final_rows = sorted(desired_set, key=lambda row: int(anchor_ids[row]))
    final_states = {
        name: (
            _MatchingState(query_count, edges)
            if name == "matching"
            else _UniqueCoverageState(query_count, edges)
        )
        for name, edges in layer_edges.items()
    }
    final_pose = torch.eye(6, dtype=torch.float64).repeat(query_count, 1, 1) * float(
        targets.pose_damping
    )
    for row in final_rows:
        for state in final_states.values():
            state.add(row)
        for query, matrix in pose_information[row].items():
            final_pose[int(query)] += torch.as_tensor(matrix).double()
    final_logdet, final_minimum = pose_scores(final_pose)
    return {
        "schema": "lafgs_v7_unified_sufficiency_selection",
        "version": 1,
        "selected_anchor_rows": torch.tensor(final_rows, dtype=torch.long),
        "selected_anchor_ids": anchor_ids[final_rows],
        "primary_selection_reason": {
            int(anchor_ids[row]): reason.get(row, "trust_region_previous_state")
            for row in final_rows
        },
        "eligibility": {
            "eligible_count": int(eligible.sum()),
            "candidate_count": count,
        },
        "targets": {
            "precision_matching_rank": precision_target.tolist(),
            **{name: value.tolist() for name, value in layer_targets.items()},
        },
        "achieved": {
            **{name: final_states[name].counts.tolist() for name in COMPLETION_LAYERS},
            "pose_logdet": final_logdet.tolist(),
            "pose_minimum_eigenvalue": final_minimum.tolist(),
        },
        "unmet": {
            **{
                name: int(
                    np.maximum(layer_targets[name] - final_states[name].counts, 0).sum()
                )
                for name in COMPLETION_LAYERS
            },
            "pose": int(
                (
                    (final_logdet < pose_logdet_target)
                    | (final_minimum < pose_minimum_target)
                ).sum()
            ),
        },
        "trust_region": {
            "initialization_unlimited": previous_active is None,
            "change_fraction": float(active_set_change_fraction),
            "change_budget": budget,
            "applied_change_count": len(applied_changes),
            "applied_changes": applied_changes,
            "previous_clean_critical_protected": sorted(
                int(anchor_ids[row]) for row in critical_rows
            ),
        },
        "contract": {
            "same_api_initialization_and_update": True,
            "hierarchical_not_weighted_sum": True,
            "exact_provenance_reason": True,
            "one_anchor_one_pose_contribution_per_query": True,
            "test_queries_used": False,
            "feedback_descriptors_used": False,
        },
    }
