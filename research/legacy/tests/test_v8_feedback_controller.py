import torch

from map_learning.v8_feedback_controller import (
    materialize_quarantined_map,
    propose_feedback_anchor_quarantine,
)


def _record(query: int, family: int, false_id: int, positive_id: int) -> dict:
    return {
        "query_index": query,
        "pose_family_id": family,
        "diagnosis": {
            "category": "precision_deficit",
            "can_drive_map_update": True,
            "translation_error_cm": 2.0,
            "rotation_error_deg": 0.2,
            "precision_diagnostic": {
                "alternative_translation_error_cm": 0.5,
                "alternative_rotation_error_deg": 0.1,
            },
            "descriptor_control_evidence": {
                "positive_anchor_ids": torch.tensor([positive_id]),
                "false_attractor_anchor_ids": torch.tensor([false_id]),
            },
        },
    }


def test_quarantine_requires_repeated_unopposed_pose_evidence() -> None:
    records = [_record(0, 10, 1, 4), _record(1, 11, 1, 5)]
    # Anchor 2 is harmful only once; Anchor 4 is protected by a positive role.
    records += [_record(2, 12, 2, 4), _record(3, 13, 4, 5)]
    proposal = propose_feedback_anchor_quarantine(
        anchor_ids=torch.arange(100),
        feedback_records=records,
        maximum_quarantine_fraction=0.01,
    )
    assert proposal["proposed_anchor_ids"].tolist() == [1]
    assert proposal["actual_quarantine_fraction"] == 0.01


def test_materialized_quarantine_preserves_ids_and_slices_csr() -> None:
    state = {
        "anchor_ids": torch.tensor([8, 9, 10]),
        "anchor_xyz": torch.arange(9).reshape(3, 3),
        "anchor_features": torch.eye(3),
        "canonical_anchor_count": 3,
        "micro_anchor_count": 3,
        "provenance": {"uses_test_queries": False},
        "projective_anchor_observations": {
            "observation_offsets": torch.tensor([0, 2, 3, 5]),
            "query_indices": torch.tensor([0, 1, 2, 3, 4]),
            "keypoint_indices": torch.tensor([5, 6, 7, 8, 9]),
        },
    }
    output, selected = materialize_quarantined_map(state, torch.tensor([1]))
    assert selected.tolist() == [0, 2]
    assert output["anchor_ids"].tolist() == [8, 10]
    csr = output["projective_anchor_observations"]
    assert csr["observation_offsets"].tolist() == [0, 2, 4]
    assert csr["query_indices"].tolist() == [0, 1, 3, 4]
    assert output["provenance"]["v8_feedback_quarantined_source_anchor_ids"].tolist() == [9]
