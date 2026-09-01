from __future__ import annotations

from copy import deepcopy

import pytest
import torch

from map_learning.v21_identity_calibration import (
    ROLE,
    SCHEMA,
    SEMANTICS,
    VERSION,
    accepted_identity_mask,
    aggregate_record_counts,
    block_descriptor_medoid,
    build_identity_evidence,
    build_query_provisional_record,
    calibration_thresholds,
    mapping_identity_support,
    sha256_json,
    threshold_grid_summary,
    validate_identity_calibration_payload,
    validate_query_provisional_record,
)


SHA = "a" * 64
SHA_B = "b" * 64


def _mapping_support() -> dict:
    # Global observations are [q0, q1], [q2], [q0, q2] for Anchor rows 0..2.
    # Provenance rows deliberately arrive as a permutation.
    return mapping_identity_support(
        target_fine_identity_ids=torch.tensor([10, 20]),
        anchor_fine_identity_ids=torch.tensor([10, 10, 20]),
        observation_offsets=torch.tensor([0, 2, 3, 5]),
        observation_query_indices=torch.tensor([0, 1, 2, 0, 2]),
        provenance_observation_rows=torch.tensor([2, 0, 4, 1, 3]),
        provenance_observation_valid=torch.tensor([True, True, True, True, False]),
        mapping_view_family_ids=torch.tensor([0, 1, 2]),
        family_roles={0: "track_bank", 1: "track_bank", 2: "independent_validation"},
    )


def _evidence() -> dict:
    descriptors = torch.tensor(
        [
            [1.0, 0.0],
            [0.99, 0.01],
            [0.98, 0.02],
            [0.97, 0.03],
            [0.0, 1.0],
        ]
    )
    return build_identity_evidence(
        observation_fine_identity_ids=torch.tensor([10, 10, 10, 10, 20]),
        observation_anchor_rows=torch.tensor([0, 0, 1, 1, 2]),
        observation_descriptors=descriptors,
        observation_query_indices=torch.tensor([0, 1, 2, 3, 4]),
        observation_block_ids=("b0", "b0", "b1", "b2", "b0"),
        observation_sequence_ids=("s0", "s0", "s0", "s1", "s0"),
        wrong_top1_anchor_row=torch.tensor([True, False, True, False, False]),
        wrong_top1_fine_identity=torch.tensor([True, False, True, False, False]),
        baseline_failure=torch.tensor([True, False, True, False, True]),
        baseline_inlier=torch.tensor([True, True, False, False, True]),
        truth_anchor_scores=torch.tensor([0.7, 0.8, 0.7, 0.9, 0.8]),
        winner_scores=torch.tensor([0.9, 0.8, 0.9, 0.9, 0.8]),
        mapping_support=_mapping_support(),
    )


def _thresholds() -> dict:
    return calibration_thresholds(
        minimum_adaptation_blocks=3,
        minimum_adaptation_sequences=1,
        minimum_mapping_observations=2,
        minimum_mapping_families=2,
        minimum_descriptor_medoid_cosine=0.8,
    )


def _query(evidence: dict) -> dict:
    accepted = accepted_identity_mask(evidence, _thresholds())
    return build_query_provisional_record(
        query={
            "query_index": 0,
            "image_name": "seq/frame00001.png",
            "image_sha256": SHA,
            "sequence_id": "seq",
            "frame_index": 1,
            "block_id": "b0",
            "source_record_sha256": SHA_B,
            "pose_w2c_sha256": SHA,
            "keypoints_sha256": SHA,
            "descriptors_sha256": SHA,
            "keypoint_count": 4,
            "baseline_r5": False,
            "diagnostic_unique_query_rows": torch.tensor([0, 1, 2]),
            "diagnostic_unique_anchor_rows": torch.tensor([0, 1, 2]),
            "diagnostic_unique_fine_identity_ids": torch.tensor([10, 10, 20]),
            "winner_anchor_rows": torch.tensor([7, 1, 2]),
            "winner_fine_identity_ids": torch.tensor([99, 10, 20]),
            "baseline_inlier": torch.tensor([True, True, True]),
            "truth_anchor_scores": torch.tensor([0.7, 0.8, 0.8]),
            "winner_scores": torch.tensor([0.9, 0.8, 0.8]),
        },
        evidence=evidence,
        accepted_identities=accepted,
    )


def test_block_medoid_balances_blocks_instead_of_dense_edges() -> None:
    result = block_descriptor_medoid(
        torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.8, 0.2]]),
        ("block_a", "block_a", "block_b"),
    )
    assert result["adaptation_block_count"] == 2
    assert result["adaptation_block_ids"] == ("block_a", "block_b")
    assert result["descriptor_cross_block_defined"] is True
    assert result["descriptor_medoid_min_cosine"] > 0.95


def test_one_block_does_not_claim_cross_block_consistency() -> None:
    result = block_descriptor_medoid(torch.tensor([[1.0, 0.0]]), ("block_a",))
    assert result["descriptor_cross_block_defined"] is False
    assert result["descriptor_medoid_min_cosine"] == -1.0


def test_mapping_support_uses_only_valid_track_bank_families() -> None:
    support = _mapping_support()
    assert support["mapping_track_bank_observation_count"].tolist() == [2, 0]
    assert support["mapping_track_bank_family_count"].tolist() == [2, 0]
    assert support["mapping_all_valid_observation_count"].tolist() == [3, 1]
    assert support["mapping_all_valid_family_count"].tolist() == [3, 1]
    assert support["identity_anchor_row_count"].tolist() == [2, 1]


def test_gate_retains_recurrent_identity_and_positive_only_preservation() -> None:
    evidence = _evidence()
    accepted = accepted_identity_mask(evidence, _thresholds())
    assert evidence["fine_identity_ids"].tolist() == [10, 20]
    assert accepted.tolist() == [True, False]
    query = _query(evidence)
    assert query["accepted_unique_mask"].tolist() == [True, True, False]
    assert query["provisional_positive_offsets"].tolist() == [0, 1, 2, 2, 2]
    assert query["provisional_positive_anchor_rows"].tolist() == [0, 1]
    assert query["promotion_wrong_top1_count"] == 1
    assert query["preservation_correct_top1_count"] == 1
    assert query["negative_anchor_rows"] is None


def test_threshold_grid_reports_failure_queries_without_using_heldout_rows() -> None:
    evidence = _evidence()
    grid = threshold_grid_summary(
        evidence=evidence,
        observation_fine_identity_ids=torch.tensor([10, 10, 10, 10, 20]),
        observation_query_indices=torch.tensor([0, 1, 2, 3, 4]),
        wrong_top1_fine_identity=torch.tensor([True, False, True, False, False]),
        baseline_failure=torch.tensor([True, False, True, False, True]),
        block_minimums=[3],
        sequence_minimums=[1, 2],
        mapping_observation_minimums=[2],
        mapping_family_minimums=[2],
        descriptor_cosine_minimums=[0.8],
    )
    assert len(grid) == 2
    assert grid[0]["accepted_identity_count"] == 1
    assert grid[0]["baseline_failure_wrong_top1_edge_count"] == 2
    assert grid[0]["baseline_failure_query_count"] == 2


def test_payload_is_non_deployable_and_rejects_negative_anchor() -> None:
    evidence = _evidence()
    accepted = accepted_identity_mask(evidence, _thresholds())
    evidence["accepted_identity_mask"] = accepted
    query = _query(evidence)
    source = {"path": "/source", "sha256": SHA, "size_bytes": 1}
    registry = {
        "role": ROLE,
        "registry_sha256": SHA,
        "rows": [
            {
                "ordinal": 0,
                "query_index": 0,
                "image_name": query["image_name"],
                "image_sha256": query["image_sha256"],
                "source_record_sha256": query["source_record_sha256"],
            }
        ],
    }
    counts = aggregate_record_counts([query])
    payload = {
        "schema": SCHEMA,
        "version": VERSION,
        "protocol": "test_adapted",
        "uses_test_queries": True,
        "test_adapted": True,
        "role": ROLE,
        "complete_adaptation_registry_consumed": True,
        "formation_stage": "after_complete_adaptation_before_control_scoring",
        "control_or_confirmation_features_consumed": False,
        "control_or_confirmation_outcomes_consumed": False,
        "source_teacher_mutating_action_authorized": False,
        "provisional_action_positive_only": True,
        "negative_labels_created": False,
        "ambiguous_or_unlabelled_are_negative": False,
        "artifact_writes_map": False,
        "candidate_deployment_authorized": False,
        "heldout_control_required_before_confirmation": True,
        "heldout_confirmation_required_before_deployment": True,
        "semantics": SEMANTICS,
        "correspondence_truth_sha256": SHA,
        "stable_map_sha256": SHA,
        "split_manifest_sha256": SHA,
        "mapping_provenance_sha256": SHA,
        "teacher_validation_sha256": SHA,
        "frontend_shard_registry": registry,
        "frontend_shard_registry_sha256": SHA,
        "anchor_count": 100,
        "descriptor_dim": 2,
        "query_count": 1,
        "accepted_identity_count": 1,
        "candidate_thresholds": _thresholds(),
        "candidate_thresholds_sha256": sha256_json(_thresholds()),
        "threshold_grid": [{"thresholds": _thresholds()}],
        "counts": counts,
        "provisional_action_positive_available": True,
        "quarantined_candidate_generation_allowed": True,
        "identity_evidence": evidence,
        "inputs": {
            "correspondence_truth": source,
            "frontend_caches": [source],
            "stable_map": source,
            "split_manifest": source,
            "mapping_provenance": source,
            "teacher_validation": source,
            "producer_sources": [source],
        },
        "records": [query],
    }
    validate_identity_calibration_payload(payload)
    unsafe = deepcopy(payload)
    unsafe["candidate_deployment_authorized"] = True
    with pytest.raises(ValueError):
        validate_identity_calibration_payload(unsafe)
    negative = deepcopy(query)
    negative["negative_anchor_rows"] = torch.tensor([7])
    with pytest.raises(ValueError):
        validate_query_provisional_record(negative)
