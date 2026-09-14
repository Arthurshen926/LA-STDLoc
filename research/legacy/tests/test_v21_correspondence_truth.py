from __future__ import annotations

from copy import deepcopy

import pytest
import torch

from map_learning.v18_provenance_truth import backproject_query_surface
from map_learning.v19_track_extension_teacher import (
    TrackExtensionTier,
    assign_track_extension_truth,
    full_map_projection_candidate_graph,
    track_observation_consensus,
)
from map_learning.v21_correspondence_truth import (
    SCHEMA,
    SEMANTICS,
    STATUS_AMBIGUOUS,
    STATUS_EQUIVALENT,
    STATUS_NO_TRUTH,
    STATUS_UNIQUE,
    VERSION,
    build_query_truth_record,
    gaussian_row_validity,
    resolve_teacher_action,
    sha256_json,
    status_counts,
    validate_correspondence_payload,
)
from map_learning.v21_test_cache import build_shard_registry, tensor_sha256


SHA = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64


def _teacher(*, tier_c_authorized: bool = True) -> dict:
    common = {
        "maximum_query_reprojection_px": 4.0,
        "maximum_query_normalized_depth_residual": 1.0,
        "maximum_query_projection_std_px": 2.0,
        "maximum_transport_median_residual_px": 4.0,
        "minimum_transport_view_families": 2,
        "minimum_descriptor_cosine": 0.6,
        "minimum_descriptor_view_families": 1,
    }
    return {
        "schema": "lafgs_v19_track_extension_teacher_validation",
        "version": 2,
        "uses_test_queries": False,
        "loo_used": False,
        "feedback_enters_track_registry": False,
        "reference_source": "mapping_observation_track_membership",
        "reference_available_for_novel_query": False,
        "selection_uses_validation": False,
        "authorization_uses_wilson_lower_bound": True,
        "authorization_requires_independent_mapping_families": True,
        "selected_tiers": {
            "tier_b": {
                "thresholds": common,
                "permitted_actions_if_authorized": ["strong_metric_control"],
                "authorized_actions": [],
                "calibration": {"authorized": False},
                "validation": {"decisive_precision": 0.968},
            },
            "tier_c": {
                "thresholds": common,
                "permitted_actions_if_authorized": [
                    "soft_diagnostic",
                    "planner_priority",
                ],
                "authorized_actions": (
                    ["soft_diagnostic", "planner_priority"]
                    if tier_c_authorized
                    else []
                ),
                "calibration": {"authorized": tier_c_authorized},
                "validation": {"decisive_precision": 0.968},
            },
        },
    }


def _frontend_and_support() -> tuple[dict, dict]:
    keypoints = torch.tensor(
        [[1.0, 1.0], [2.0, 1.0], [3.0, 1.0], [4.0, 1.0], [5.0, 1.0]]
    )
    frontend = {
        "query_index": 7,
        "image_name": "seq1/frame00007.png",
        "image_sha256": SHA,
        "sequence_id": "seq1",
        "frame_index": 7,
        "block_id": "adaptation:seq1:000",
        "role": "adaptation",
        "source_record_sha256": SHA_B,
        "pose_w2c_sha256": SHA_C,
        "keypoints": keypoints,
        "descriptors": torch.eye(5, 8),
    }
    support = {
        "query_index": 7,
        "image_name": frontend["image_name"],
        "image_sha256": SHA,
        "source_record_sha256": SHA_B,
        "pose_w2c_sha256": SHA_C,
        "role": "adaptation",
        "keypoint_count": 5,
        "keypoints_sha256": tensor_sha256(keypoints),
        "gaussian_support_valid": torch.ones(5, dtype=torch.bool),
        "gaussian_alpha_at_keypoints": torch.ones(5),
        "gaussian_relative_depth_spread_3x3": torch.zeros(5),
        "gaussian_local_valid_fraction_3x3": torch.ones(5),
    }
    return frontend, support


def _v19_truth() -> dict:
    # UNIQUE, EQUIVALENT, AMBIGUOUS, INVALID, NONE.
    return {
        "truth_status": torch.tensor([1, 2, 3, 4, 0], dtype=torch.int8),
        "truth_offsets": torch.tensor([0, 1, 3, 3, 3, 3]),
        "truth_anchor_rows": torch.tensor([1, 2, 3]),
    }


def test_tier_c_is_planner_diagnostic_not_action_truth() -> None:
    decision = resolve_teacher_action(
        _teacher(), tier_name="tier_c", requested_action="planner_priority"
    )
    assert decision["teacher_authorized"] is True
    assert decision["planner_diagnostic_authorized"] is True
    assert decision["action_authorized"] is False
    assert decision["action_block_reason"] == (
        "planner_diagnostic_is_not_map_or_metric_action"
    )


def test_unapproved_strong_metric_tier_fails_closed() -> None:
    decision = resolve_teacher_action(
        _teacher(), tier_name="tier_b", requested_action="strong_metric_control"
    )
    assert decision["teacher_authorized"] is False
    assert decision["action_authorized"] is False
    assert decision["action_block_reason"] == (
        "requested_action_not_authorized_by_teacher_validation"
    )


def test_query_truth_keeps_diagnostics_but_empties_action_csr() -> None:
    frontend, support = _frontend_and_support()
    record = build_query_truth_record(
        frontend_record=frontend,
        support_record=support,
        v19_truth=_v19_truth(),
        projection_candidate_offsets=torch.tensor([0, 1, 3, 5, 5, 5]),
        geometry_valid=torch.ones(5, dtype=torch.bool),
        action_authorized=False,
        tier_name="tier_c",
        requested_action="planner_priority",
    )
    assert record["diagnostic_truth_status"].tolist() == [
        STATUS_UNIQUE,
        STATUS_EQUIVALENT,
        STATUS_AMBIGUOUS,
        STATUS_NO_TRUTH,
        STATUS_NO_TRUTH,
    ]
    assert record["diagnostic_positive_anchor_rows"].tolist() == [1, 2, 3]
    assert record["truth_status"].tolist() == [
        STATUS_NO_TRUTH,
        STATUS_NO_TRUTH,
        STATUS_AMBIGUOUS,
        STATUS_NO_TRUTH,
        STATUS_NO_TRUTH,
    ]
    assert record["positive_offsets"].tolist() == [0, 0, 0, 0, 0, 0]
    assert record["positive_anchor_rows"].numel() == 0
    assert record["negative_anchor_rows"] is None


def test_authorized_mutating_action_exposes_only_unique_equivalent() -> None:
    frontend, support = _frontend_and_support()
    record = build_query_truth_record(
        frontend_record=frontend,
        support_record=support,
        v19_truth=_v19_truth(),
        projection_candidate_offsets=torch.tensor([0, 1, 3, 5, 5, 5]),
        geometry_valid=torch.ones(5, dtype=torch.bool),
        action_authorized=True,
        tier_name="tier_b",
        requested_action="strong_metric_control",
    )
    assert record["truth_status"].tolist() == [1, 2, 3, 0, 0]
    assert record["positive_offsets"].tolist() == [0, 1, 3, 3, 3, 3]
    assert record["positive_anchor_rows"].tolist() == [1, 2, 3]


def test_gaussian_rejections_abstain_and_never_become_negatives() -> None:
    _, support = _frontend_and_support()
    support["gaussian_alpha_at_keypoints"] = torch.tensor(
        [1.0, 0.1, 1.0, 1.0, 1.0]
    )
    support["gaussian_relative_depth_spread_3x3"] = torch.tensor(
        [0.0, 0.0, 0.2, 0.0, 0.0]
    )
    support["gaussian_local_valid_fraction_3x3"] = torch.tensor(
        [1.0, 1.0, 1.0, 0.5, 1.0]
    )
    support["gaussian_support_valid"][4] = False
    valid = gaussian_row_validity(
        support,
        minimum_alpha=0.2,
        maximum_relative_depth_spread=0.05,
        minimum_local_valid_fraction=1.0,
    )
    assert valid.tolist() == [True, False, False, False, False]


def test_real_v19_full_map_and_track_consensus_certifies_synthetic_track() -> None:
    intrinsic = torch.tensor(
        [[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]]
    )
    pose = torch.eye(4)
    keypoints = torch.tensor([[50.0, 50.0]])
    depth = torch.tensor([2.0])
    surface, surface_valid = backproject_query_surface(
        keypoints, depth, intrinsic, pose
    )
    graph = full_map_projection_candidate_graph(
        keypoints=keypoints,
        rendered_depth=depth,
        query_indices=torch.zeros(1, dtype=torch.long),
        anchor_xyz=torch.tensor([[0.0, 0.0, 2.0]]),
        anchor_covariance=torch.zeros(1, 3, 3),
        observation_count=torch.tensor([3]),
        query_intrinsics=intrinsic[None],
        query_poses_w2c=pose[None],
        device="cpu",
    )
    graph["query_valid"] = graph["query_valid"] & surface_valid
    consensus = track_observation_consensus(
        candidate_graph=graph,
        query_surface_xyz=surface,
        query_descriptors=torch.tensor([[1.0, 0.0]]),
        anchor_observation_offsets=torch.tensor([0, 3]),
        observation_query_indices=torch.tensor([0, 1, 2]),
        observation_keypoint_indices=torch.tensor([0, 0, 0]),
        observation_enabled=torch.ones(3, dtype=torch.bool),
        mapping_keypoints=[keypoints, keypoints, keypoints],
        mapping_descriptors=[
            torch.tensor([[1.0, 0.0]]),
            torch.tensor([[1.0, 0.0]]),
            torch.tensor([[1.0, 0.0]]),
        ],
        mapping_intrinsics=intrinsic.repeat(3, 1, 1),
        mapping_poses_w2c=pose.repeat(3, 1, 1),
        mapping_view_family_ids=torch.tensor([0, 1, 2]),
        device="cpu",
    )
    truth = assign_track_extension_truth(
        candidate_graph=graph,
        consensus=consensus,
        equivalence_class_ids=torch.tensor([0]),
        tier=TrackExtensionTier(1.0, 1.0, 1.0, 1.0, 2, 0.9, 2),
    )
    assert truth["truth_status"].tolist() == [1]
    assert truth["truth_anchor_rows"].tolist() == [0]


def _payload(record: dict, decision: dict) -> dict:
    record = deepcopy(record)
    source = {"path": "/tmp/source.pt", "sha256": SHA, "size_bytes": 1}
    split_row = {
        "query_index": record["query_index"],
        "image_name": record["image_name"],
        "image_sha256": record["image_sha256"],
        "source_record_sha256": record["source_record_sha256"],
        "role": "adaptation",
    }
    registry = build_shard_registry(
        [split_row],
        role="adaptation",
        shard_count=1,
        split_manifest_sha256=SHA,
    )
    record["source_record_sha256"] = registry["rows"][0][
        "source_record_sha256"
    ]
    records = [record]
    counts = status_counts(records)
    diagnostic_counts = status_counts(records, diagnostic=True)
    diagnostic_edges = int(record["diagnostic_positive_anchor_rows"].numel())
    positive_edges = int(record["positive_anchor_rows"].numel())
    diagnostic_rows = diagnostic_counts["UNIQUE"] + diagnostic_counts["EQUIVALENT"]
    gates = {
        "minimum_alpha": 0.2,
        "maximum_relative_depth_spread": 0.05,
        "minimum_local_valid_fraction": 1.0,
        "maximum_projection_candidates_per_row": 64,
        "saturated_candidate_rows": "abstain",
    }
    return {
        "schema": SCHEMA,
        "version": VERSION,
        "protocol": "test_adapted",
        "uses_test_queries": True,
        "test_adapted": True,
        "role": "adaptation",
        "training_consumers_allowed": bool(decision["action_authorized"]),
        "planner_diagnostic_consumers_allowed": bool(
            decision["planner_diagnostic_authorized"]
        ),
        "control_or_confirmation_forbidden": True,
        "negative_labels_created": False,
        "ambiguous_or_unlabelled_are_negative": False,
        "feedback_enters_mapping_track_registry": False,
        "artifact_writes_map": False,
        "exact_poselib_recovery_is_identity_truth": False,
        "semantics": dict(SEMANTICS),
        "stable_map_sha256": SHA,
        "gaussian_support_sha256": SHA,
        "mapping_provenance_sha256": SHA,
        "mapping_feature_cache_sha256": SHA,
        "teacher_validation_sha256": SHA,
        "frontend_shard_registry": registry,
        "frontend_shard_registry_sha256": registry["registry_sha256"],
        "anchor_count": 8,
        "query_count": 1,
        "teacher_action_decision": decision,
        "action_authorized": bool(decision["action_authorized"]),
        "gaussian_geometry_gates": gates,
        "gaussian_geometry_gates_sha256": sha256_json(gates),
        "status_counts": counts,
        "diagnostic_status_counts": diagnostic_counts,
        "diagnostic_positive_edge_count": diagnostic_edges,
        "positive_edge_count": positive_edges,
        "blocked_diagnostic_positive_row_count": (
            diagnostic_rows if not decision["action_authorized"] else 0
        ),
        "blocked_diagnostic_positive_edge_count": (
            diagnostic_edges if not decision["action_authorized"] else 0
        ),
        "blocked_diagnostic_positive_reason": (
            decision["action_block_reason"]
            if diagnostic_rows and not decision["action_authorized"]
            else None
        ),
        "inputs": {
            "stable_map": source,
            "gaussian_support": source,
            "mapping_provenance": source,
            "mapping_feature_cache": source,
            "teacher_validation": source,
            "frontend_caches": [source],
            "producer_sources": [source],
        },
        "records": records,
    }


def test_payload_reports_blocked_diagnostics_and_rejects_hidden_action() -> None:
    frontend, support = _frontend_and_support()
    decision = resolve_teacher_action(
        _teacher(), tier_name="tier_c", requested_action="planner_priority"
    )
    record = build_query_truth_record(
        frontend_record=frontend,
        support_record=support,
        v19_truth=_v19_truth(),
        projection_candidate_offsets=torch.tensor([0, 1, 3, 5, 5, 5]),
        geometry_valid=torch.ones(5, dtype=torch.bool),
        action_authorized=False,
        tier_name="tier_c",
        requested_action="planner_priority",
    )
    payload = _payload(record, decision)
    validate_correspondence_payload(payload)
    assert payload["blocked_diagnostic_positive_row_count"] == 2
    assert payload["blocked_diagnostic_positive_edge_count"] == 3

    tampered = deepcopy(payload)
    tampered["records"][0]["truth_status"][0] = STATUS_UNIQUE
    tampered["records"][0]["positive_offsets"] = torch.tensor(
        [0, 1, 1, 1, 1, 1]
    )
    tampered["records"][0]["positive_anchor_rows"] = torch.tensor([1])
    with pytest.raises(ValueError, match="unauthorized"):
        validate_correspondence_payload(tampered)
