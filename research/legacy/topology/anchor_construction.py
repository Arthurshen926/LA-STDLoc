"""Unified construction model for projective Track and surface completion Anchors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch

from evidence.observation_provider import ObservationProvider
from evidence.tracks import fuse_track_descriptors


TRACK_KIND = 1
SURFACE_COMPLETION_KIND = 0


def _exact_vector(value, *, name: str, count: int, dtype: torch.dtype) -> torch.Tensor:
    tensor = torch.as_tensor(value)
    if tensor.dtype != dtype or tensor.shape != (count,):
        raise ValueError(
            f"{name} must have exact dtype={dtype} shape={(count,)}, "
            f"got dtype={tensor.dtype} shape={tuple(tensor.shape)}"
        )
    return tensor


@dataclass(frozen=True)
class AnchorCandidateBatch:
    """One provenance-homogeneous batch in the unified candidate registry."""

    xyz: torch.Tensor
    features: torch.Tensor
    source_primitive_ids: torch.Tensor
    track_cluster_ids: torch.Tensor
    anchor_type: torch.Tensor
    parent_identity_ids: torch.Tensor
    correlation_group_ids: torch.Tensor
    covariance: torch.Tensor
    matchability: torch.Tensor
    surface_support_weight: torch.Tensor
    candidate_kind: torch.Tensor
    observation_offsets: torch.Tensor
    observation_query_indices: torch.Tensor
    observation_keypoint_indices: torch.Tensor

    def validate(self) -> None:
        count = int(self.xyz.shape[0]) if self.xyz.ndim == 2 else -1
        if self.xyz.ndim != 2 or self.xyz.shape[1] != 3 or count < 0:
            raise ValueError("candidate xyz must have shape [N,3]")
        if self.features.ndim != 2 or self.features.shape[0] != count:
            raise ValueError("candidate features must have shape [N,D]")
        if (
            not torch.isfinite(self.xyz).all()
            or not torch.isfinite(self.features).all()
        ):
            raise ValueError("candidate geometry/features must be finite")
        for name, value in (
            ("source_primitive_ids", self.source_primitive_ids),
            ("track_cluster_ids", self.track_cluster_ids),
            ("anchor_type", self.anchor_type),
            ("parent_identity_ids", self.parent_identity_ids),
            ("correlation_group_ids", self.correlation_group_ids),
            ("candidate_kind", self.candidate_kind),
        ):
            _exact_vector(value, name=name, count=count, dtype=torch.long)
        for name, value in (
            ("matchability", self.matchability),
            ("surface_support_weight", self.surface_support_weight),
        ):
            _exact_vector(value, name=name, count=count, dtype=torch.float32)
            if not torch.isfinite(value).all():
                raise ValueError(f"{name} must be finite")
        if self.covariance.dtype != torch.float32 or self.covariance.shape != (
            count,
            3,
            3,
        ):
            raise ValueError("candidate covariance must be float32 [N,3,3]")
        offsets = self.observation_offsets
        if offsets.dtype != torch.long or offsets.shape != (count + 1,):
            raise ValueError("observation offsets must be int64 [N+1]")
        if int(offsets[0]) != 0 or bool((offsets[1:] < offsets[:-1]).any()):
            raise ValueError("observation offsets must start at zero and be monotonic")
        edge_count = int(offsets[-1])
        for name, value in (
            ("observation_query_indices", self.observation_query_indices),
            ("observation_keypoint_indices", self.observation_keypoint_indices),
        ):
            if value.dtype != torch.long or value.shape != (edge_count,):
                raise ValueError(f"{name} must be int64 [observation_count]")
            if value.numel() and int(value.min()) < 0:
                raise ValueError(f"{name} cannot contain negative rows")


class AnchorCandidateProvider(Protocol):
    def materialize(self) -> AnchorCandidateBatch: ...


def _selected_track_observations(
    payload: dict, selected_tracks: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    tracks = payload["tracks"]
    all_track = torch.as_tensor(tracks["track_index"]).long()
    all_query = torch.as_tensor(tracks["query_index"]).long()
    all_keypoint = torch.as_tensor(tracks["keypoint_index"]).long()
    counts = []
    queries = []
    keypoints = []
    for track in selected_tracks.tolist():
        rows = torch.nonzero(all_track == int(track), as_tuple=False).reshape(-1)
        counts.append(int(rows.numel()))
        queries.append(all_query[rows])
        keypoints.append(all_keypoint[rows])
    offsets = torch.cat(
        (
            torch.zeros(1, dtype=torch.long),
            torch.tensor(counts, dtype=torch.long).cumsum(0),
        )
    )
    return (
        offsets,
        torch.cat(queries) if queries else torch.empty(0, dtype=torch.long),
        torch.cat(keypoints) if keypoints else torch.empty(0, dtype=torch.long),
    )


class TrackAnchorProvider:
    """Materialize observation-defined Anchors from frozen Track geometry."""

    def __init__(
        self,
        *,
        payload: dict,
        observations: ObservationProvider | dict,
        track_indices: torch.Tensor,
        trim_fraction: float,
        features: torch.Tensor | None = None,
        source_primitive_ids: torch.Tensor | None = None,
        matchability: torch.Tensor | None = None,
    ) -> None:
        self.payload = payload
        self.observations = observations
        self.track_indices = torch.as_tensor(track_indices).long()
        self.trim_fraction = float(trim_fraction)
        self.features = features
        self.source_primitive_ids = source_primitive_ids
        self.matchability = matchability

    def materialize(self) -> AnchorCandidateBatch:
        selected = self.track_indices
        if selected.ndim != 1 or torch.unique(selected).numel() != selected.numel():
            raise ValueError("selected Track IDs must be a unique int64 vector")
        geometry = self.payload["track_geometry"]
        track_count = int(torch.as_tensor(geometry["triangulated_xyz"]).shape[0])
        if selected.numel() and (
            int(selected.min()) < 0 or int(selected.max()) >= track_count
        ):
            raise ValueError("selected Track ID is outside geometry")
        xyz = torch.as_tensor(geometry["triangulated_xyz"])[selected].float()
        features = (
            fuse_track_descriptors(
                payload=self.payload,
                query_cache=self.observations,
                track_indices=selected,
                trim_fraction=self.trim_fraction,
            )
            if self.features is None
            else torch.as_tensor(self.features).float()
        )
        count = int(selected.numel())
        if features.ndim != 2 or features.shape[0] != count:
            raise ValueError("selected Track features do not align")
        covariance = torch.as_tensor(
            geometry.get(
                "triangulation_covariance_matrix",
                torch.full((track_count, 3, 3), float("nan")),
            )
        )[selected].float()
        sources = (
            torch.full((count,), -1, dtype=torch.long)
            if self.source_primitive_ids is None
            else torch.as_tensor(self.source_primitive_ids).long()
        )
        parents = self.payload.get("tracks", {}).get("parent_source_track_ids")
        parents = (
            selected.clone()
            if parents is None
            else torch.as_tensor(parents).long()[selected]
        )
        supported = torch.as_tensor(
            geometry.get(
                "triangulation_surface_supported",
                torch.zeros(track_count, dtype=torch.bool),
            )
        )[selected].bool()
        depth_count = torch.as_tensor(
            geometry.get(
                "triangulation_rendered_depth_observation_count",
                torch.zeros(track_count, dtype=torch.long),
            )
        )[selected].float()
        observation_count = torch.as_tensor(
            geometry.get(
                "triangulation_observation_count",
                torch.ones(track_count, dtype=torch.long),
            )
        )[selected].float()
        support_weight = torch.where(
            supported,
            (depth_count / observation_count.clamp_min(1)).clamp(0, 1),
            torch.zeros_like(depth_count),
        ).float()
        offsets, query_rows, keypoint_rows = _selected_track_observations(
            self.payload, selected
        )
        batch = AnchorCandidateBatch(
            xyz=xyz,
            features=features,
            source_primitive_ids=sources,
            track_cluster_ids=selected.clone(),
            anchor_type=torch.full((count,), 1, dtype=torch.long),
            parent_identity_ids=parents,
            correlation_group_ids=parents.clone(),
            covariance=covariance,
            matchability=(
                torch.ones(count, dtype=torch.float32)
                if self.matchability is None
                else torch.as_tensor(self.matchability).float()
            ),
            surface_support_weight=support_weight,
            candidate_kind=torch.full((count,), TRACK_KIND, dtype=torch.long),
            observation_offsets=offsets,
            observation_query_indices=query_rows,
            observation_keypoint_indices=keypoint_rows,
        )
        batch.validate()
        return batch


class SurfaceCompletionProvider:
    """Bounded non-Track completion from a KCS/GWFF canonical map.

    The primitive center is retained only for explicitly selected completion
    rows; it is never treated as Track/projective identity and is disabled when
    ``maximum_candidates`` is zero.
    """

    def __init__(
        self,
        canonical_map: dict,
        rows: torch.Tensor,
        *,
        maximum_candidates: int,
        matchability: torch.Tensor | None = None,
    ) -> None:
        self.canonical_map = canonical_map
        self.rows = torch.as_tensor(rows).long()
        self.maximum_candidates = int(maximum_candidates)
        self.matchability = matchability

    def materialize(self) -> AnchorCandidateBatch:
        if self.maximum_candidates < 0:
            raise ValueError("surface-completion capacity cannot be negative")
        rows = self.rows[: self.maximum_candidates]
        if rows.ndim != 1 or torch.unique(rows).numel() != rows.numel():
            raise ValueError("surface-completion rows must be unique")
        count_all = int(torch.as_tensor(self.canonical_map["anchor_xyz"]).shape[0])
        if rows.numel() and (int(rows.min()) < 0 or int(rows.max()) >= count_all):
            raise ValueError("surface-completion row is outside canonical map")
        xyz = torch.as_tensor(self.canonical_map["anchor_xyz"])[rows].float()
        features = torch.as_tensor(self.canonical_map["anchor_features"])[rows].float()
        sources = torch.as_tensor(self.canonical_map["source_primitive_ids"])[
            rows
        ].long()
        support_components = self.canonical_map.get("gaussian_support_component_ids")
        if sources.numel() and bool((sources < 0).any()) and support_components is None:
            raise ValueError(
                "surface completion requires Gaussian lineage via primitive or "
                "raster-support identity"
            )
        group_field = next(
            (
                field
                for field in (
                    "source_dependency_group_ids",
                    "coarse_dependency_group_ids",
                    "dependency_group_ids",
                )
                if field in self.canonical_map
            ),
            None,
        )
        if support_components is not None:
            support_components = torch.as_tensor(support_components)[rows].long()
        groups = (
            support_components.clone()
            if support_components is not None
            else (
                sources.clone()
                if group_field is None
                else torch.as_tensor(self.canonical_map[group_field])[rows].long()
            )
        )
        parents = torch.where(sources >= 0, sources, groups)
        count = int(rows.numel())
        covariance = torch.as_tensor(
            self.canonical_map.get(
                "anchor_position_covariance",
                torch.zeros((count_all, 3, 3), dtype=torch.float32),
            )
        )[rows].float()
        surface_observations = self.canonical_map.get("surface_completion_observations")
        if surface_observations is None:
            offsets = torch.zeros(count + 1, dtype=torch.long)
            observation_query = torch.empty(0, dtype=torch.long)
            observation_keypoint = torch.empty(0, dtype=torch.long)
        else:
            all_offsets = torch.as_tensor(
                surface_observations["observation_offsets"]
            ).long()
            all_query = torch.as_tensor(surface_observations["query_indices"]).long()
            all_keypoint = torch.as_tensor(
                surface_observations["keypoint_indices"]
            ).long()
            counts = all_offsets[rows + 1] - all_offsets[rows]
            selected_query = []
            selected_keypoint = []
            for row in rows.tolist():
                start, end = int(all_offsets[row]), int(all_offsets[row + 1])
                selected_query.append(all_query[start:end])
                selected_keypoint.append(all_keypoint[start:end])
            offsets = torch.cat((torch.zeros(1, dtype=torch.long), counts.cumsum(0)))
            observation_query = (
                torch.cat(selected_query)
                if selected_query
                else torch.empty(0, dtype=torch.long)
            )
            observation_keypoint = (
                torch.cat(selected_keypoint)
                if selected_keypoint
                else torch.empty(0, dtype=torch.long)
            )
        batch = AnchorCandidateBatch(
            xyz=xyz,
            features=features,
            source_primitive_ids=sources,
            track_cluster_ids=torch.full((count,), -1, dtype=torch.long),
            anchor_type=torch.zeros(count, dtype=torch.long),
            parent_identity_ids=parents,
            correlation_group_ids=groups,
            covariance=covariance,
            matchability=(
                torch.ones(count, dtype=torch.float32)
                if self.matchability is None
                else torch.as_tensor(self.matchability).float()
            ),
            surface_support_weight=torch.ones(count, dtype=torch.float32),
            candidate_kind=torch.full(
                (count,), SURFACE_COMPLETION_KIND, dtype=torch.long
            ),
            observation_offsets=offsets,
            observation_query_indices=observation_query,
            observation_keypoint_indices=observation_keypoint,
        )
        batch.validate()
        return batch


class UnifiedAnchorConstructor:
    """Concatenate provider batches into the single selector/localizer state."""

    @staticmethod
    def materialize(providers: list[AnchorCandidateProvider]) -> AnchorCandidateBatch:
        batches = [provider.materialize() for provider in providers]
        if not batches:
            raise ValueError("unified Anchor construction requires a provider")
        for batch in batches:
            batch.validate()
        descriptor_dims = {int(batch.features.shape[1]) for batch in batches}
        if len(descriptor_dims) != 1:
            raise ValueError("all Anchor providers must share descriptor dimension")
        counts = [int(batch.xyz.shape[0]) for batch in batches]
        observation_offsets = [0]
        query_rows = []
        keypoint_rows = []
        for batch in batches:
            base = observation_offsets[-1]
            observation_offsets.extend((batch.observation_offsets[1:] + base).tolist())
            query_rows.append(batch.observation_query_indices)
            keypoint_rows.append(batch.observation_keypoint_indices)
        result = AnchorCandidateBatch(
            xyz=torch.cat([batch.xyz for batch in batches]),
            features=torch.cat([batch.features for batch in batches]),
            source_primitive_ids=torch.cat(
                [batch.source_primitive_ids for batch in batches]
            ),
            track_cluster_ids=torch.cat([batch.track_cluster_ids for batch in batches]),
            anchor_type=torch.cat([batch.anchor_type for batch in batches]),
            parent_identity_ids=torch.cat(
                [batch.parent_identity_ids for batch in batches]
            ),
            correlation_group_ids=torch.cat(
                [batch.correlation_group_ids for batch in batches]
            ),
            covariance=torch.cat([batch.covariance for batch in batches]),
            matchability=torch.cat([batch.matchability for batch in batches]),
            surface_support_weight=torch.cat(
                [batch.surface_support_weight for batch in batches]
            ),
            candidate_kind=torch.cat([batch.candidate_kind for batch in batches]),
            observation_offsets=torch.tensor(observation_offsets, dtype=torch.long),
            observation_query_indices=torch.cat(query_rows),
            observation_keypoint_indices=torch.cat(keypoint_rows),
        )
        if int(result.xyz.shape[0]) != sum(counts):
            raise AssertionError(
                "unified Anchor row count changed during concatenation"
            )
        result.validate()
        return result

    @staticmethod
    def attach_to_map(state: dict, batch: AnchorCandidateBatch) -> None:
        """Attach unified semantics after exact legacy materialization."""

        batch.validate()
        count = int(torch.as_tensor(state["anchor_ids"]).numel())
        if count != int(batch.xyz.shape[0]):
            raise ValueError("unified candidate rows do not align with compact map")
        exact = {
            "anchor_xyz": batch.xyz,
            "anchor_features": batch.features,
            "source_primitive_ids": batch.source_primitive_ids,
            "track_cluster_ids": batch.track_cluster_ids,
            "anchor_type": batch.anchor_type,
        }
        for field, expected in exact.items():
            observed = torch.as_tensor(state[field])
            if observed.dtype != expected.dtype or observed.shape != expected.shape:
                raise ValueError(f"legacy/unified {field} dtype or shape differs")
            if not torch.equal(observed, expected):
                raise ValueError(f"legacy/unified {field} values differ")
        state["anchor_parent_identity_ids"] = batch.parent_identity_ids
        state["anchor_correlation_group_ids"] = batch.correlation_group_ids
        state["anchor_surface_support_weight"] = batch.surface_support_weight
        state["anchor_candidate_kind"] = batch.candidate_kind
        state["projective_anchor_observations"] = {
            "schema": "lafgs_projective_anchor_observations",
            "version": 1,
            "observation_offsets": batch.observation_offsets,
            "query_indices": batch.observation_query_indices,
            "keypoint_indices": batch.observation_keypoint_indices,
        }
        state["projective_anchor_construction"] = {
            "schema": "lafgs_gaussian_supported_projective_anchor_construction",
            "version": 1,
            "track_anchor_count": int((batch.candidate_kind == TRACK_KIND).sum()),
            "surface_completion_anchor_count": int(
                (batch.candidate_kind == SURFACE_COMPLETION_KIND).sum()
            ),
            "identity_semantics": "observation_equivalence_class_or_explicit_completion",
            "gaussian_role": "support_visibility_lineage_and_bounded_completion",
            "descriptor_fusion": "gwff_style_projective_observation_fusion",
            "completion_policy": "always_candidate_selected_by_shared_sufficiency",
        }
