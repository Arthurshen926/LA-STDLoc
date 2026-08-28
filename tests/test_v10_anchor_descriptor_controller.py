import torch

from map_learning.v10_anchor_descriptor_controller import (
    aggregate_descriptor_action_gain,
    aggregate_group_descriptor_action_gain,
    build_confusion_component_groups,
    propose_actionable_anchor_descriptors,
)


def _record(query: int, family: int, descriptor: torch.Tensor) -> dict:
    return {
        "loo_used": False,
        "can_train_metric": True,
        "query_index": query,
        "pose_family_id": family,
        "training_evidence": {
            "query_descriptors": descriptor[None],
            "positive_anchor_rows": torch.tensor([1]),
            "positive_rank": torch.tensor([1]),
            "negative_scores": torch.tensor([0.72]),
        },
    }


def test_v10_proposes_one_bounded_family_balanced_descriptor() -> None:
    anchors = torch.tensor([[1.0, 0, 0, 0], [0.8, 0.6, 0, 0]])
    descriptors = [
        torch.tensor([0.68, 0.73, 0, 0]),
        torch.tensor([0.67, 0.74, 0, 0]),
        torch.tensor([0.69, 0.72, 0, 0]),
    ]
    proposal = propose_actionable_anchor_descriptors(
        anchor_features=anchors,
        feedback_records=[_record(i, i, value) for i, value in enumerate(descriptors)],
        maximum_update_angle_deg=5.0,
    )
    assert proposal["candidate_count"] == 1
    assert proposal["candidate_anchor_rows"].tolist() == [1]
    assert proposal["candidate_audit"][0]["update_angle_deg"] <= 5.00001
    assert proposal["feedback_descriptor_exact_copy"] is False
    assert proposal["descriptor_count_per_anchor"] == 1


def test_v10_descriptor_action_needs_actual_cross_family_gain() -> None:
    records = [
        {"anchor_row": 3, "pose_family_id": 1, "baseline_task_error": 1.0, "updated_task_error": 0.8, "loo_used": False},
        {"anchor_row": 3, "pose_family_id": 2, "baseline_task_error": 1.0, "updated_task_error": 0.7, "loo_used": False},
        {"anchor_row": 4, "pose_family_id": 1, "baseline_task_error": 1.0, "updated_task_error": 0.5, "loo_used": False},
    ]
    audit = aggregate_descriptor_action_gain(records)
    assert audit["authorized_anchor_rows"].tolist() == [3]


def test_v10_confusion_groups_are_disjoint_and_group_gain_is_causal() -> None:
    records = [
        {
            "loo_used": False,
            "can_train_metric": True,
            "training_evidence": {"positive_anchor_rows": torch.tensor([1, 2, 3])},
        },
        {
            "loo_used": False,
            "can_train_metric": True,
            "training_evidence": {"positive_anchor_rows": torch.tensor([3, 4])},
        },
    ]
    groups = build_confusion_component_groups(
        candidate_anchor_rows=torch.tensor([1, 2, 3, 4, 8]),
        feedback_records=records,
        maximum_group_size=3,
    )
    flattened = [anchor for group in groups for anchor in group]
    assert sorted(flattened) == [1, 2, 3, 4, 8]
    assert len(flattened) == len(set(flattened))
    audit = aggregate_group_descriptor_action_gain(
        [
            {"group_index": 0, "pose_family_id": 1, "baseline_task_error": 1.0, "updated_task_error": 0.8, "loo_used": False},
            {"group_index": 0, "pose_family_id": 2, "baseline_task_error": 1.0, "updated_task_error": 0.7, "loo_used": False},
        ]
    )
    assert audit["authorized_group_indices"].tolist() == [0]
