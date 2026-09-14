import torch

from map_learning.v17_competitive_metric import (
    build_competitive_metric_evidence,
    select_minimum_effective_gain,
)


def _query(query_index: int, family: int) -> dict:
    return {
        "query_index": query_index,
        "pose_family_id": family,
        "keypoints": torch.tensor([[5.0, 5.0], [15.0, 15.0]]),
        "candidate_anchor_rows": torch.tensor([[1, 0], [0, 1]]),
        "candidate_scores": torch.tensor([[0.9, 0.8], [0.9, 0.8]]),
        "certified_positive": torch.tensor([[False, True], [True, False]]),
        "image_hw": torch.tensor([20, 20]),
    }


def test_competitive_metric_repairs_current_winner_and_protects_correct_winner() -> None:
    queries = [_query(0, 10), _query(1, 11)]
    evidence = build_competitive_metric_evidence(
        competition_queries=queries,
        query_descriptors={0: torch.eye(2), 1: torch.eye(2)},
        action_metadata={
            0: {"can_train_metric": True, "actual_task_gain": 1.0},
            1: {"can_train_metric": True, "actual_task_gain": 2.0},
        },
        active_anchor_mask=torch.ones(2, dtype=torch.bool),
    )
    assert evidence["repair_negative_anchor_rows"].tolist() == [1, 1]
    assert evidence["repair_positive_anchor_rows"].tolist() == [0, 0]
    assert evidence["protection_positive_anchor_rows"].tolist() == [0, 0]
    assert evidence["protection_negative_anchor_rows"].tolist() == [1, 1]
    assert evidence["repair_pose_family_count"] == 2
    assert torch.allclose(
        evidence["protection_initial_margin"], torch.tensor([0.1, 0.1])
    )


def test_competitive_metric_rejects_one_family_negative_evidence() -> None:
    query = _query(0, 10)
    try:
        build_competitive_metric_evidence(
            competition_queries=[query],
            query_descriptors={0: torch.eye(2)},
            action_metadata={
                0: {"can_train_metric": True, "actual_task_gain": 1.0}
            },
            active_anchor_mask=torch.ones(2, dtype=torch.bool),
        )
    except RuntimeError as error:
        assert "no active competitive repair pair" in str(error)
    else:
        raise AssertionError("one-family evidence must not authorize metric repair")


def test_minimum_effective_gain_does_not_chase_largest_average_response() -> None:
    def decision(classification: str, net: float, r5: float) -> dict:
        return {
            "classification": classification,
            "hard_safety": {"passed": True},
            "paired_effect": {"net_gain": net},
            "baseline": {"r5_percent": 80.0},
            "candidate": {"r5_percent": r5},
        }

    selected = select_minimum_effective_gain(
        {
            "alpha_0p025_active": decision("DEFAULT_CANDIDATE", 0.3, 81.0),
            "alpha_0p25_active": decision("DEFAULT_CANDIDATE", 500.0, 82.0),
        }
    )
    assert selected == "alpha_0p025_active"
