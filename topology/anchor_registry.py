"""Compatibility registry for evidence-grounded localization anchors.

The registry is deliberately a view over an existing materialized V3 map.  It
adds orthogonal identity, geometry, evidence, and selection semantics while
preserving every tensor consumed by localization exactly.  It is therefore a
safe migration boundary, not a new topology policy.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch

from topology.geometry_materializer import (
    GEOMETRY_IMAGE_TRIANGULATED,
    GEOMETRY_SURFACE_INITIALIZED,
    GEOMETRY_SURFACE_REGULARIZED,
    materialize_legacy_map_geometry,
)


SCHEMA = "lafgs_evidence_grounded_anchor_registry"
VERSION = 1

IDENTITY_TRACK_VERIFIED = 0
IDENTITY_MULTI_VIEW_SUPPORTED = 1
IDENTITY_WEAK_FALLBACK = 2

SELECTION_PRECISION = 0
SELECTION_MATCHING_COMPLETION = 1
SELECTION_OBSERVABILITY_COMPLETION = 2
SELECTION_LEGACY_UNRESOLVED = 3

OBSERVATION_TRACK = 0
OBSERVATION_POSITIVE_TEACHER = 1


def _aligned_tensor(
    payload: Mapping,
    key: str,
    count: int,
    *,
    default: float = float("nan"),
) -> torch.Tensor:
    if key not in payload:
        return torch.full((count,), default, dtype=torch.float32)
    value = torch.as_tensor(payload[key]).detach().cpu().float().reshape(-1)
    if value.numel() != count:
        raise ValueError(f"{key} does not align with anchor IDs")
    return value.clone()


def _teacher_observations(
    teacher: Mapping,
    anchor_type: torch.Tensor,
    *,
    include_tracks: bool,
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    anchor_rows: list[torch.Tensor] = []
    query_rows: list[torch.Tensor] = []
    keypoint_rows: list[torch.Tensor] = []
    source_rows: list[torch.Tensor] = []
    count = int(anchor_type.numel())
    if int(teacher.get("anchor_count", count)) != count:
        raise ValueError("positive teacher anchor count does not match anchor map")
    for fallback_query, record in enumerate(teacher.get("records", ())):
        offsets = torch.as_tensor(record["positive_offsets"]).long().reshape(-1)
        positives = torch.as_tensor(record["positive_indices"]).long().reshape(-1)
        rows = torch.as_tensor(record["query_rows"]).long().reshape(-1)
        if offsets.numel() != rows.numel() + 1:
            raise ValueError("positive teacher offsets do not align with query rows")
        if offsets.numel() and (
            int(offsets[0]) != 0 or int(offsets[-1]) != positives.numel()
        ):
            raise ValueError("positive teacher CSR offsets are invalid")
        if positives.numel() == 0:
            continue
        if int(positives.min()) < 0 or int(positives.max()) >= count:
            raise ValueError("positive teacher references an invalid anchor")
        repeated_rows = torch.repeat_interleave(
            torch.arange(rows.numel()), offsets[1:] - offsets[:-1]
        )
        keep = torch.ones_like(positives, dtype=torch.bool)
        if not include_tracks:
            keep &= anchor_type[positives] == 0
        positives = positives[keep]
        repeated_rows = repeated_rows[keep]
        if positives.numel() == 0:
            continue
        query_index = int(record.get("query_index", fallback_query))
        anchor_rows.append(positives)
        query_rows.append(torch.full_like(positives, query_index))
        keypoint_rows.append(rows[repeated_rows])
        source_rows.append(
            torch.full_like(positives, OBSERVATION_POSITIVE_TEACHER)
        )
    return anchor_rows, query_rows, keypoint_rows, source_rows


def _track_observations(
    track_payload: Mapping,
    track_cluster_ids: torch.Tensor,
    teacher: Mapping | None,
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    tracks = track_payload.get("tracks")
    if not isinstance(tracks, Mapping):
        return [], [], [], []
    observed_tracks = torch.as_tensor(tracks["track_index"]).long().reshape(-1)
    observed_queries = torch.as_tensor(tracks["query_index"]).long().reshape(-1)
    observed_keypoints = torch.as_tensor(tracks["keypoint_index"]).long().reshape(-1)
    if not (
        observed_tracks.numel()
        == observed_queries.numel()
        == observed_keypoints.numel()
    ):
        raise ValueError("track observation arrays do not align")
    if observed_tracks.numel() == 0:
        return [], [], [], []
    track_count = int(
        torch.as_tensor(
            track_payload["track_geometry"]["triangulated_xyz"]
        ).shape[0]
    )
    if int(observed_tracks.min()) < 0 or int(observed_tracks.max()) >= track_count:
        raise ValueError("track payload references an invalid track")
    row_by_track = torch.full((track_count,), -1, dtype=torch.long)
    selected_rows = torch.nonzero(
        track_cluster_ids >= 0, as_tuple=False
    ).reshape(-1)
    selected_tracks = track_cluster_ids[selected_rows]
    if selected_tracks.numel() and int(selected_tracks.max()) >= track_count:
        raise ValueError("anchor map references an invalid track")
    row_by_track[selected_tracks] = selected_rows
    anchor_rows = row_by_track[observed_tracks]
    keep = anchor_rows >= 0
    if not bool(keep.any()):
        return [], [], [], []

    # Track and teacher artifacts normally share the same query ordering.  If
    # not, remap by immutable query name instead of silently changing identity.
    if teacher is not None and "query_names" in teacher and "query_names" in track_payload:
        teacher_names = list(teacher["query_names"])
        track_names = list(track_payload["query_names"])
        if track_names != teacher_names:
            teacher_index = {name: index for index, name in enumerate(teacher_names)}
            if len(teacher_index) != len(teacher_names):
                raise ValueError("positive teacher query names are not unique")
            try:
                remap = torch.tensor(
                    [teacher_index[name] for name in track_names], dtype=torch.long
                )
            except KeyError as error:
                raise ValueError("track and teacher query registries differ") from error
            observed_queries = remap[observed_queries]
    return (
        [anchor_rows[keep]],
        [observed_queries[keep]],
        [observed_keypoints[keep]],
        [torch.full_like(anchor_rows[keep], OBSERVATION_TRACK)],
    )


def _observation_csr(
    state: Mapping,
    teacher: Mapping | None,
    track_payload: Mapping | None,
) -> dict[str, torch.Tensor]:
    anchor_type = torch.as_tensor(state["anchor_type"]).detach().cpu().long()
    track_ids = torch.as_tensor(state["track_cluster_ids"]).detach().cpu().long()
    count = int(anchor_type.numel())
    anchors: list[torch.Tensor] = []
    queries: list[torch.Tensor] = []
    keypoints: list[torch.Tensor] = []
    sources: list[torch.Tensor] = []
    if track_payload is not None:
        chunks = _track_observations(track_payload, track_ids, teacher)
        for target, values in zip((anchors, queries, keypoints, sources), chunks):
            target.extend(values)
    if teacher is not None:
        chunks = _teacher_observations(
            teacher, anchor_type, include_tracks=track_payload is None
        )
        for target, values in zip((anchors, queries, keypoints, sources), chunks):
            target.extend(values)
    if not anchors:
        return {
            "observation_offsets": torch.zeros(count + 1, dtype=torch.long),
            "observation_query_indices": torch.empty(0, dtype=torch.long),
            "observation_keypoint_indices": torch.empty(0, dtype=torch.long),
            "observation_source_kind": torch.empty(0, dtype=torch.int8),
        }
    triples = torch.stack(
        (
            torch.cat(anchors),
            torch.cat(queries),
            torch.cat(keypoints),
            torch.cat(sources).long(),
        ),
        dim=1,
    )
    triples = torch.unique(triples, dim=0, sorted=True)
    order = torch.argsort(triples[:, 0], stable=True)
    triples = triples[order]
    counts = torch.bincount(triples[:, 0], minlength=count)
    offsets = torch.zeros(count + 1, dtype=torch.long)
    offsets[1:] = torch.cumsum(counts, dim=0)
    return {
        "observation_offsets": offsets,
        "observation_query_indices": triples[:, 1].long(),
        "observation_keypoint_indices": triples[:, 2].long(),
        "observation_source_kind": triples[:, 3].to(torch.int8),
    }


def _distinct_query_counts(observations: Mapping, count: int) -> torch.Tensor:
    offsets = observations["observation_offsets"]
    queries = observations["observation_query_indices"]
    result = torch.zeros(count, dtype=torch.long)
    for row in range(count):
        begin, end = int(offsets[row]), int(offsets[row + 1])
        if end > begin:
            result[row] = torch.unique(queries[begin:end]).numel()
    return result


def _query_registry(
    teacher: Mapping | None,
    track_payload: Mapping | None,
    observations: Mapping,
) -> tuple[list[str], torch.Tensor, str]:
    teacher_names = list(teacher.get("query_names", ())) if teacher is not None else []
    track_names = (
        list(track_payload.get("query_names", ()))
        if track_payload is not None
        else []
    )
    names = teacher_names or track_names
    observed_queries = observations["observation_query_indices"]
    required_count = int(observed_queries.max()) + 1 if observed_queries.numel() else 0
    if not names:
        names = [f"query_{index:06d}" for index in range(required_count)]
    if len(names) < required_count:
        raise ValueError("query registry does not cover all Anchor observations")
    groups = torch.full((len(names),), -1, dtype=torch.long)
    semantics = "unavailable"
    if track_payload is not None and "query_bins" in track_payload:
        track_groups = torch.as_tensor(track_payload["query_bins"]).long().reshape(-1)
        if track_groups.numel() != len(track_names):
            raise ValueError("track query bins do not align with query names")
        if track_names == names:
            groups.copy_(track_groups)
        else:
            index = {name: row for row, name in enumerate(names)}
            if len(index) != len(names):
                raise ValueError("Anchor query names are not unique")
            try:
                target = torch.tensor([index[name] for name in track_names]).long()
            except KeyError as error:
                raise ValueError("track and Anchor query registries differ") from error
            groups[target] = track_groups
        semantics = "track_payload_query_bins"
    return names, groups, semantics


def _geometry_fields(
    state: Mapping,
    track_payload: Mapping | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    materialized = materialize_legacy_map_geometry(state, track_payload)
    return (
        materialized["geometry_mode"],
        materialized["covariance"],
        materialized["surface_evidence"],
        materialized["surface_dependence"],
    )


def _selection_reasons(
    state: Mapping,
    provenance: Mapping | None,
) -> tuple[torch.Tensor, bool]:
    count = int(torch.as_tensor(state["anchor_ids"]).numel())
    reason = torch.full((count,), SELECTION_LEGACY_UNRESOLVED, dtype=torch.int8)
    if provenance is None:
        return reason, False
    required = (
        "track_universe_count",
        "track_core_universe_ids",
        "coverage_track_universe_ids",
        "coverage_gaussian_universe_ids",
        "pose_track_universe_ids",
        "pose_gaussian_universe_ids",
    )
    missing = [key for key in required if key not in provenance]
    if missing:
        raise ValueError(
            "explicit selection provenance is incomplete: " + ", ".join(missing)
        )
    track_ids = torch.as_tensor(state["track_cluster_ids"]).detach().cpu().long()
    base_rows = torch.as_tensor(
        state.get("track_centric_reconstruction", {}).get(
            "base_canonical_rows", torch.empty(0, dtype=torch.long)
        )
    ).detach().cpu().long()
    raw_track_count = provenance["track_universe_count"]
    if isinstance(raw_track_count, bool) or not isinstance(raw_track_count, int):
        raise ValueError("selection track_universe_count must be an integer")
    track_count = int(raw_track_count)
    if track_count < 0:
        raise ValueError("selection track_universe_count must be non-negative")
    is_track = track_ids >= 0
    unified_ids = track_ids.clone()
    if int((~is_track).sum()) != int(base_rows.numel()):
        raise ValueError("base canonical rows do not align with materialized map")
    unified_ids[~is_track] = track_count + base_rows
    if unified_ids.numel() != torch.unique(unified_ids).numel():
        raise ValueError("materialized Anchor selection universe is not unique")
    assignments = (
        (SELECTION_PRECISION, ("track_core_universe_ids",)),
        (
            SELECTION_MATCHING_COMPLETION,
            ("coverage_track_universe_ids", "coverage_gaussian_universe_ids"),
        ),
        (
            SELECTION_OBSERVABILITY_COMPLETION,
            ("pose_track_universe_ids", "pose_gaussian_universe_ids"),
        ),
    )
    selected_groups: list[torch.Tensor] = []
    for value, keys in assignments:
        chunks = []
        for key in keys:
            raw = torch.as_tensor(provenance[key]).detach().cpu()
            if raw.ndim != 1 or raw.dtype == torch.bool or raw.is_floating_point():
                raise ValueError(f"selection provenance {key} must be a 1-D integer tensor")
            chunk = raw.long()
            if chunk.numel() != torch.unique(chunk).numel():
                raise ValueError(f"selection provenance {key} contains duplicates")
            is_track_group = key in {
                "track_core_universe_ids",
                "coverage_track_universe_ids",
                "pose_track_universe_ids",
            }
            if is_track_group and chunk.numel() and (
                int(chunk.min()) < 0 or int(chunk.max()) >= track_count
            ):
                raise ValueError(f"selection provenance {key} is outside Track universe")
            if not is_track_group and chunk.numel() and int(
                chunk.min()
            ) < track_count:
                raise ValueError(
                    f"selection provenance {key} is outside Gaussian universe"
                )
            chunks.append(chunk)
        selected = torch.cat(chunks)
        if selected.numel() != torch.unique(selected).numel():
            raise ValueError("selection provenance groups overlap")
        selected_groups.append(selected)
        if selected.numel():
            reason[torch.isin(unified_ids, selected)] = value
    all_selected = torch.cat(selected_groups)
    if all_selected.numel() != torch.unique(all_selected).numel():
        raise ValueError("selection provenance reasons are not mutually exclusive")
    if not torch.equal(
        torch.sort(all_selected).values, torch.sort(unified_ids).values
    ):
        raise ValueError(
            "explicit selection provenance does not exactly cover final Anchors"
        )
    if bool((reason == SELECTION_LEGACY_UNRESOLVED).any()):
        raise ValueError("explicit selection provenance left unresolved Anchors")
    return reason, True


def build_anchor_registry(
    state: Mapping,
    *,
    teacher: Mapping | None = None,
    track_payload: Mapping | None = None,
    selection_provenance: Mapping | None = None,
) -> dict:
    """Build a zero-behavior-change registry from an existing map artifact."""
    if state.get("schema") != "lafgs_materialized_anchor_map":
        raise ValueError("unsupported anchor map schema")
    required = (
        "anchor_ids",
        "anchor_xyz",
        "anchor_features",
        "source_primitive_ids",
        "track_cluster_ids",
        "anchor_type",
    )
    missing = [key for key in required if key not in state]
    if missing:
        raise ValueError(f"anchor map missing fields: {missing}")
    count = int(torch.as_tensor(state["anchor_ids"]).numel())
    copied = {
        key: torch.as_tensor(state[key]).detach().cpu().clone()
        for key in required
    }
    for key, value in copied.items():
        if value.shape[0] != count:
            raise ValueError(f"{key} does not align with anchor IDs")
    for key in (
        "dependency_group_ids",
        "coarse_dependency_group_ids",
        "fine_identity_ids",
        "source_dependency_group_ids",
    ):
        if key in state:
            copied[key] = torch.as_tensor(state[key]).detach().cpu().clone()

    observations = _observation_csr(state, teacher, track_payload)
    query_names, query_group_ids, query_group_semantics = _query_registry(
        teacher, track_payload, observations
    )
    observation_count = observations["observation_offsets"][1:] - observations[
        "observation_offsets"
    ][:-1]
    distinct_queries = _distinct_query_counts(observations, count)
    legacy_type = copied["anchor_type"].long()
    identity = torch.full((count,), IDENTITY_WEAK_FALLBACK, dtype=torch.int8)
    identity[distinct_queries >= 2] = IDENTITY_MULTI_VIEW_SUPPORTED
    identity[legacy_type == 1] = IDENTITY_TRACK_VERIFIED
    (
        geometry_mode,
        covariance,
        surface_evidence,
        surface_dependence,
    ) = _geometry_fields(
        state, track_payload
    )
    selection_reason, exact_selection = _selection_reasons(
        state, selection_provenance
    )
    gaussian_lineage = copied["source_primitive_ids"].long() >= 0
    evidence_mask = (
        (legacy_type == 1).to(torch.int16)
        | (surface_evidence.to(torch.int16) << 1)
        | (gaussian_lineage.to(torch.int16) << 2)
        | ((observation_count > 0).to(torch.int16) << 3)
    )
    registry = {
        "schema": SCHEMA,
        "version": VERSION,
        "uses_test_queries": False,
        "mapping_only": True,
        "audit_only": True,
        "localization_input": False,
        **copied,
        **observations,
        "query_names": query_names,
        "query_group_ids": query_group_ids,
        "query_group_semantics": query_group_semantics,
        "identity_mode": identity,
        "geometry_mode": geometry_mode,
        "surface_dependence": surface_dependence,
        "selection_reason": selection_reason,
        "evidence_mask": evidence_mask,
        "observation_count": observation_count,
        "distinct_observation_query_count": distinct_queries,
        "anchor_position_covariance": covariance,
        "anchor_reliability": _aligned_tensor(state, "anchor_reliability", count),
        "anchor_matchability": _aligned_tensor(state, "anchor_matchability", count),
        "anchor_alias_risk": _aligned_tensor(state, "anchor_alias_risk", count),
        "compatibility": {
            "source_schema": str(state["schema"]),
            "localization_tensors_preserved_exactly": True,
            "legacy_anchor_type_retained": True,
            "selection_provenance_exact": exact_selection,
            "missing_scores_are_nan": True,
            "geometry_materialization_policy": "v3_p5_compatibility",
            "surface_dependence_changes_localization_tensors": False,
            "legacy_unresolved_is_epistemic_unknown": True,
            "new_pipeline_requires_exact_selection_provenance": True,
        },
        "enums": {
            "identity_mode": {
                "track_verified": IDENTITY_TRACK_VERIFIED,
                "multi_view_supported": IDENTITY_MULTI_VIEW_SUPPORTED,
                "weak_fallback": IDENTITY_WEAK_FALLBACK,
            },
            "geometry_mode": {
                "image_triangulated": GEOMETRY_IMAGE_TRIANGULATED,
                "surface_regularized": GEOMETRY_SURFACE_REGULARIZED,
                "surface_initialized": GEOMETRY_SURFACE_INITIALIZED,
            },
            "selection_reason": {
                "precision": SELECTION_PRECISION,
                "matching_completion": SELECTION_MATCHING_COMPLETION,
                "observability_completion": SELECTION_OBSERVABILITY_COMPLETION,
                "legacy_unresolved": SELECTION_LEGACY_UNRESOLVED,
            },
            "observation_source_kind": {
                "track": OBSERVATION_TRACK,
                "positive_teacher": OBSERVATION_POSITIVE_TEACHER,
            },
            "evidence_mask_bits": {
                "track_observation": 0,
                "surface_prior": 1,
                "gaussian_lineage": 2,
                "real_observation": 3,
            },
        },
    }
    validate_registry_compatibility(registry, state)
    registry["report"] = registry_report(registry)
    return registry


def validate_registry_compatibility(registry: Mapping, state: Mapping) -> None:
    """Fail unless every localization-facing tensor is bitwise unchanged."""
    for key in (
        "anchor_ids",
        "anchor_xyz",
        "anchor_features",
        "source_primitive_ids",
        "track_cluster_ids",
        "anchor_type",
        "dependency_group_ids",
        "coarse_dependency_group_ids",
        "fine_identity_ids",
        "source_dependency_group_ids",
    ):
        if key not in state:
            continue
        if key not in registry or not torch.equal(
            torch.as_tensor(registry[key]).cpu(), torch.as_tensor(state[key]).cpu()
        ):
            raise ValueError(f"registry changed localization tensor: {key}")


def registry_report(registry: Mapping) -> dict:
    identity = torch.as_tensor(registry["identity_mode"])
    geometry = torch.as_tensor(registry["geometry_mode"])
    surface_dependence = torch.as_tensor(
        registry.get(
            "surface_dependence",
            geometry != GEOMETRY_IMAGE_TRIANGULATED,
        )
    ).bool()
    selection = torch.as_tensor(registry["selection_reason"])
    observations = torch.as_tensor(registry["observation_count"])
    return {
        "anchor_count": int(identity.numel()),
        "observation_edge_count": int(observations.sum()),
        "observation_grounded_anchor_count": int((observations > 0).sum()),
        "identity": {
            "track_verified": int((identity == IDENTITY_TRACK_VERIFIED).sum()),
            "multi_view_supported": int(
                (identity == IDENTITY_MULTI_VIEW_SUPPORTED).sum()
            ),
            "weak_fallback": int((identity == IDENTITY_WEAK_FALLBACK).sum()),
        },
        "geometry": {
            "image_triangulated": int(
                (geometry == GEOMETRY_IMAGE_TRIANGULATED).sum()
            ),
            "surface_regularized": int(
                (geometry == GEOMETRY_SURFACE_REGULARIZED).sum()
            ),
            "surface_initialized": int(
                (geometry == GEOMETRY_SURFACE_INITIALIZED).sum()
            ),
            "surface_dependent": int(surface_dependence.sum()),
        },
        "selection": {
            "precision": int((selection == SELECTION_PRECISION).sum()),
            "matching_completion": int(
                (selection == SELECTION_MATCHING_COMPLETION).sum()
            ),
            "observability_completion": int(
                (selection == SELECTION_OBSERVABILITY_COMPLETION).sum()
            ),
            "legacy_unresolved": int(
                (selection == SELECTION_LEGACY_UNRESOLVED).sum()
            ),
        },
        "selection_provenance_exact": bool(
            registry["compatibility"]["selection_provenance_exact"]
        ),
    }
