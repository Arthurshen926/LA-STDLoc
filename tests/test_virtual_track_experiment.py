import torch

from evidence.virtual_track_experiment import (
    DRY_RUN_THRESHOLDS,
    augment_formal_anchor_map,
    build_map_bound_identity_metric,
    dry_run_passes,
    enforce_one_observation_per_family,
    validate_augmented_mapping_guard,
)
from topology.anchor_construction import AnchorCandidateBatch


def test_duplicate_pose_family_cannot_supply_two_track_observations():
    tracks = {
        "track_index": torch.tensor([0, 0, 0, 1, 1, 1]),
        "query_index": torch.tensor([0, 1, 2, 0, 2, 3]),
        "keypoint_index": torch.tensor([4, 5, 6, 7, 8, 9]),
        "confidence": torch.tensor([0.3, 0.9, 0.7, 0.8, 0.2, 0.5]),
        "track_level": torch.tensor([2, 1], dtype=torch.int8),
    }
    filtered, audit = enforce_one_observation_per_family(
        tracks, torch.tensor([249, 249, 281, 283])
    )
    assert filtered["query_index"].tolist() == [1, 2, 0, 2, 3]
    assert audit["duplicate_family_observation_count"] == 1
    assert audit["maximum_observations_per_track_family"] == 1


def test_dry_run_gate_is_frozen_and_test_independent():
    metrics = dict(DRY_RUN_THRESHOLDS)
    metrics.update(family_contract_passed=True, gt_visible_diagnostic=None)
    passed, failures = dry_run_passes(metrics)
    assert passed and failures == []
    metrics["test_median_translation_cm"] = 10_000  # irrelevant field
    assert dry_run_passes(metrics)[0]
    metrics["new_anchor_count"] = 0
    assert not dry_run_passes(metrics)[0]


def test_virtual_tracks_augment_and_preserve_formal_map_prefix():
    formal = {
        "schema": "lafgs_materialized_anchor_map", "version": 1,
        "anchor_ids": torch.arange(2),
        "anchor_xyz": torch.tensor([[0., 0., 1.], [1., 0., 1.]]),
        "anchor_features": torch.eye(2),
        "source_primitive_ids": torch.tensor([-1, 7]),
        "track_cluster_ids": torch.tensor([5, -1]),
        "anchor_type": torch.tensor([1, 0]),
        "dependency_group_ids": torch.tensor([5, 7]),
        "coarse_dependency_group_ids": torch.tensor([5, 7]),
        "fine_identity_ids": torch.tensor([5, 7]),
        "source_dependency_group_ids": torch.tensor([-1, 7]),
        "parent_source_track_ids": torch.tensor([5, 7]),
        "repair_child_index": torch.tensor([0, 0]),
        "repair_parent_child_count": torch.tensor([1, 1]),
        "anchor_parent_identity_ids": torch.tensor([5, 7]),
        "anchor_correlation_group_ids": torch.tensor([5, 7]),
        "anchor_surface_support_weight": torch.tensor([0., 1.]),
        "anchor_candidate_kind": torch.tensor([1, 0]),
        "base_anchor_count": 1, "canonical_anchor_count": 2,
        "micro_anchor_count": 1, "requested_micro_anchor_budget": 1,
        "projective_anchor_observations": {
            "schema": "lafgs_projective_anchor_observations", "version": 1,
            "observation_offsets": torch.tensor([0, 1, 1]),
            "query_indices": torch.tensor([0]), "keypoint_indices": torch.tensor([3]),
        },
        "projective_anchor_construction": {
            "schema": "lafgs_gaussian_supported_projective_anchor_construction",
            "version": 1, "track_anchor_count": 1,
            "surface_completion_anchor_count": 1,
        },
    }
    virtual = AnchorCandidateBatch(
        xyz=torch.tensor([[2., 0., 1.]]), features=torch.tensor([[.5, .5]]),
        source_primitive_ids=torch.tensor([-1]), track_cluster_ids=torch.tensor([0]),
        anchor_type=torch.tensor([1]), parent_identity_ids=torch.tensor([0]),
        correlation_group_ids=torch.tensor([0]), covariance=torch.eye(3)[None],
        matchability=torch.ones(1), surface_support_weight=torch.zeros(1),
        candidate_kind=torch.ones(1, dtype=torch.long),
        observation_offsets=torch.tensor([0, 1]),
        observation_query_indices=torch.tensor([0]),
        observation_keypoint_indices=torch.tensor([4]),
    )
    registry = {"query_count": 1, "formal_query_count": 2}
    augmented = augment_formal_anchor_map(
        formal, virtual, formal_query_count=2,
        virtual_registry=registry, lineage={"mapping_only": True},
    )
    guard = validate_augmented_mapping_guard(
        augmented, formal, virtual_query_count=1
    )
    assert guard["formal_prefix_preserved"]
    assert guard["augmented_anchor_count"] == 3
    assert augmented["projective_anchor_observations"]["query_indices"].tolist() == [0, 2]
    assert augmented["virtual_anchor_selection_registry"][
        "primary_selection_reasons"
    ] == ["stable_broad_virtual_track_augmentation"]
    augmented["virtual_anchor_selection_registry"]["primary_selection_reasons"] = [
        "evil"
    ]
    import pytest
    with pytest.raises(ValueError, match="selection registry"):
        validate_augmented_mapping_guard(augmented, formal, virtual_query_count=1)


def test_augmented_identity_metric_is_exact_no_learn_and_map_bound():
    payload = build_map_bound_identity_metric(
        map_path="/tmp/exact-map.pt", map_sha256="abc",
        anchor_ids=torch.arange(3), descriptor_dim=2,
        producer={"entrypoint": "unit-test"},
    )
    assert payload["landmark_indices"].tolist() == [0, 1, 2]
    assert payload["metric_config"]["max_residual_norm"] == 0.0
    assert payload["producer"]["virtual_suffix_training_steps"] == 0
    assert all(torch.count_nonzero(value) == 0 for value in payload["metric_state_dict"].values())
    augment_formal_anchor_map,
