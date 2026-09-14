from __future__ import annotations

import pytest
import torch

from evidence.observation_provider import GaussianRenderObservationProvider
from evidence.tracks import (
    fuse_projective_anchor_observations,
    robust_fuse_track_descriptors,
)
from topology.anchor_construction import (
    SURFACE_COMPLETION_KIND,
    TRACK_KIND,
    SurfaceCompletionProvider,
    TrackAnchorProvider,
    UnifiedAnchorConstructor,
)


def _provider() -> GaussianRenderObservationProvider:
    records = {}
    for index in range(3):
        records[f"seq-{index}/frame.png"] = {
            "native_keypoints": torch.tensor([[index + 0.1, index + 0.2]]),
            "native_descriptors": torch.nn.functional.normalize(
                torch.tensor([[1.0, float(index + 1), 0.5]]), dim=1
            ),
            "native_scores": torch.tensor([0.8]),
            "native_K": torch.eye(3),
            "pose_w2c": torch.eye(4),
            "native_input_hw": [4, 5],
            "native_valid_mask": torch.ones((4, 5), dtype=torch.bool),
            "native_depth": torch.ones((4, 5)),
            "native_alpha": torch.ones((4, 5)),
        }
    return GaussianRenderObservationProvider(
        {"uses_source_mapping_rgb": False, "queries": records},
        query_bins=torch.tensor([0, 1, 2]),
    )


def _track_payload() -> dict:
    return {
        "query_names": list(_provider().names),
        "query_bins": torch.tensor([0, 1, 2]),
        "tracks": {
            "track_index": torch.tensor([0, 0, 1]),
            "query_index": torch.tensor([0, 1, 2]),
            "keypoint_index": torch.tensor([0, 0, 0]),
            "confidence": torch.tensor([0.9, 0.7, 0.6]),
            "parent_source_track_ids": torch.tensor([11, 11]),
        },
        "track_geometry": {
            "triangulated_xyz": torch.tensor([[1.0, 0.0, 2.0], [2.0, 0.0, 3.0]]),
            "triangulation_covariance_matrix": torch.eye(3).repeat(2, 1, 1),
            "triangulation_surface_supported": torch.tensor([True, False]),
            "triangulation_rendered_depth_observation_count": torch.tensor([1, 0]),
            "triangulation_observation_count": torch.tensor([2, 1]),
        },
    }


def _surface_map() -> dict:
    return {
        "anchor_xyz": torch.tensor([[5.0, 1.0, 4.0], [6.0, 1.0, 4.0]]),
        "anchor_features": torch.nn.functional.normalize(
            torch.tensor([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]]), dim=1
        ),
        "source_primitive_ids": torch.tensor([20, 21]),
        "coarse_dependency_group_ids": torch.tensor([7, 7]),
        "anchor_position_covariance": torch.eye(3).repeat(2, 1, 1) * 0.1,
    }


def test_projective_fusion_compatibility_mode_is_exact() -> None:
    descriptors = torch.nn.functional.normalize(torch.rand((5, 8)), dim=1)
    bins = torch.tensor([0, 0, 1, 2, 2])
    confidence = torch.tensor([0.9, 0.8, 0.7, 0.6, 0.5])
    reliability = torch.tensor([1.0, 0.8, 1.0, 0.9, 0.7])
    expected = robust_fuse_track_descriptors(
        descriptors, bins, confidence * reliability, trim_fraction=0.2
    )
    observed = fuse_projective_anchor_observations(
        descriptors,
        bins,
        detector_weight=confidence,
        visibility_weight=reliability,
        trim_fraction=0.2,
    )
    assert torch.equal(observed, expected)


def test_unified_constructor_keeps_track_and_surface_identity_distinct() -> None:
    track = TrackAnchorProvider(
        payload=_track_payload(),
        observations=_provider(),
        track_indices=torch.tensor([0, 1]),
        trim_fraction=0.0,
        source_primitive_ids=torch.tensor([-1, -1]),
        matchability=torch.tensor([0.8, 0.7]),
    )
    surface = SurfaceCompletionProvider(
        _surface_map(),
        torch.tensor([1]),
        maximum_candidates=1,
        matchability=torch.tensor([0.4]),
    )
    batch = UnifiedAnchorConstructor.materialize([track, surface])
    assert batch.xyz.shape == (3, 3)
    assert batch.candidate_kind.tolist() == [
        TRACK_KIND,
        TRACK_KIND,
        SURFACE_COMPLETION_KIND,
    ]
    assert batch.track_cluster_ids.tolist() == [0, 1, -1]
    assert batch.parent_identity_ids.tolist() == [11, 11, 21]
    assert batch.correlation_group_ids.tolist() == [11, 11, 7]
    assert batch.surface_support_weight.tolist() == [0.5, 0.0, 1.0]
    assert batch.observation_offsets.tolist() == [0, 2, 3, 3]

    state = {
        "anchor_ids": torch.arange(3),
        "anchor_xyz": batch.xyz.clone(),
        "anchor_features": batch.features.clone(),
        "source_primitive_ids": batch.source_primitive_ids.clone(),
        "track_cluster_ids": batch.track_cluster_ids.clone(),
        "anchor_type": batch.anchor_type.clone(),
    }
    UnifiedAnchorConstructor.attach_to_map(state, batch)
    assert state["projective_anchor_construction"]["completion_policy"] == (
        "always_candidate_selected_by_shared_sufficiency"
    )
    assert (
        state["projective_anchor_construction"]["surface_completion_anchor_count"] == 1
    )
    assert state["projective_anchor_observations"]["observation_offsets"].tolist() == [
        0,
        2,
        3,
        3,
    ]


def test_surface_completion_is_bounded_and_requires_gaussian_lineage() -> None:
    disabled = SurfaceCompletionProvider(
        _surface_map(), torch.tensor([0, 1]), maximum_candidates=0
    ).materialize()
    assert disabled.xyz.shape[0] == 0
    malformed = _surface_map()
    malformed["source_primitive_ids"][1] = -1
    with pytest.raises(ValueError, match="Gaussian lineage"):
        SurfaceCompletionProvider(
            malformed, torch.tensor([1]), maximum_candidates=1
        ).materialize()


def test_unified_attachment_fails_on_any_legacy_numeric_drift() -> None:
    batch = TrackAnchorProvider(
        payload=_track_payload(),
        observations=_provider(),
        track_indices=torch.tensor([0]),
        trim_fraction=0.0,
    ).materialize()
    state = {
        "anchor_ids": torch.arange(1),
        "anchor_xyz": batch.xyz.clone(),
        "anchor_features": batch.features.clone(),
        "source_primitive_ids": batch.source_primitive_ids.clone(),
        "track_cluster_ids": batch.track_cluster_ids.clone(),
        "anchor_type": batch.anchor_type.clone(),
    }
    state["anchor_xyz"][0, 0] += 1e-6
    with pytest.raises(ValueError, match="anchor_xyz values differ"):
        UnifiedAnchorConstructor.attach_to_map(state, batch)
