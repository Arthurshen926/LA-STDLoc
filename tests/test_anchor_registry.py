import torch

from topology.anchor_registry import (
    GEOMETRY_IMAGE_TRIANGULATED,
    GEOMETRY_SURFACE_INITIALIZED,
    IDENTITY_MULTI_VIEW_SUPPORTED,
    IDENTITY_TRACK_VERIFIED,
    SELECTION_MATCHING_COMPLETION,
    SELECTION_OBSERVABILITY_COMPLETION,
    SELECTION_PRECISION,
    build_anchor_registry,
    validate_registry_compatibility,
)


def _state():
    return {
        "schema": "lafgs_materialized_anchor_map",
        "anchor_ids": torch.arange(3),
        "anchor_xyz": torch.tensor(
            [[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [2.0, 0.0, 1.0]]
        ),
        "anchor_features": torch.randn(3, 4),
        "source_primitive_ids": torch.tensor([7, 8, 9]),
        "track_cluster_ids": torch.tensor([1, -1, -1]),
        "anchor_type": torch.tensor([1, 0, 0]),
        "dependency_group_ids": torch.tensor([0, 1, 2]),
        "track_centric_reconstruction": {
            "base_canonical_rows": torch.tensor([2, 4])
        },
    }


def _teacher():
    records = []
    for query in range(2):
        records.append(
            {
                "query_index": query,
                "query_rows": torch.tensor([10 + query]),
                "positive_offsets": torch.tensor([0, 1]),
                "positive_indices": torch.tensor([1]),
            }
        )
    return {"anchor_count": 3, "query_names": ["a", "b"], "records": records}


def _tracks():
    return {
        "query_names": ["a", "b"],
        "tracks": {
            "track_index": torch.tensor([1, 1]),
            "query_index": torch.tensor([0, 1]),
            "keypoint_index": torch.tensor([3, 5]),
        },
        "track_geometry": {
            "triangulated_xyz": torch.zeros(5, 3),
            "triangulation_covariance_matrix": torch.eye(3).repeat(5, 1, 1),
        },
    }


def test_registry_preserves_v3_map_and_separates_semantics():
    state = _state()
    provenance = {
        "track_universe_count": 5,
        "track_core_universe_ids": torch.tensor([1]),
        "coverage_track_universe_ids": torch.empty(0, dtype=torch.long),
        "coverage_gaussian_universe_ids": torch.tensor([7]),
        "pose_track_universe_ids": torch.empty(0, dtype=torch.long),
        "pose_gaussian_universe_ids": torch.tensor([9]),
    }
    registry = build_anchor_registry(
        state,
        teacher=_teacher(),
        track_payload=_tracks(),
        selection_provenance=provenance,
    )
    validate_registry_compatibility(registry, state)
    assert registry["identity_mode"].tolist() == [
        IDENTITY_TRACK_VERIFIED,
        IDENTITY_MULTI_VIEW_SUPPORTED,
        2,
    ]
    assert registry["geometry_mode"].tolist() == [
        GEOMETRY_IMAGE_TRIANGULATED,
        GEOMETRY_SURFACE_INITIALIZED,
        GEOMETRY_SURFACE_INITIALIZED,
    ]
    assert registry["selection_reason"].tolist() == [
        SELECTION_PRECISION,
        SELECTION_MATCHING_COMPLETION,
        SELECTION_OBSERVABILITY_COMPLETION,
    ]
    assert registry["observation_count"].tolist() == [2, 2, 0]
    assert registry["compatibility"]["localization_tensors_preserved_exactly"]


def test_legacy_registry_does_not_invent_selection_provenance():
    registry = build_anchor_registry(_state())
    assert not registry["compatibility"]["selection_provenance_exact"]
    assert registry["report"]["selection"]["legacy_unresolved"] == 3


def test_registry_uses_track_covariance_trace_as_isotropic_fallback():
    tracks = _tracks()
    tracks["track_geometry"].pop("triangulation_covariance_matrix")
    tracks["track_geometry"]["triangulation_covariance_trace"] = torch.full(
        (5,), 0.3
    )
    registry = build_anchor_registry(_state(), track_payload=tracks)
    torch.testing.assert_close(
        registry["anchor_position_covariance"][0], torch.eye(3) * 0.1
    )
