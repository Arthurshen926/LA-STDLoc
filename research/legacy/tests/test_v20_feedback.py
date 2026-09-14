import pytest
import torch

from map_learning.v18_provenance_truth import (
    TRUTH_AMBIGUOUS,
    TRUTH_NONE,
    TRUTH_UNIQUE,
)
from map_learning.v20_feedback import (
    ROW_NUISANCE,
    ROW_REPAIR,
    ROW_UNKNOWN,
    build_topk_competition_evidence,
    partition_feedback_rows,
)


def _record(query: int, family: int) -> dict:
    status = torch.tensor(
        [
            TRUTH_UNIQUE,
            TRUTH_UNIQUE,
            TRUTH_NONE,
            TRUTH_AMBIGUOUS,
            TRUTH_UNIQUE,
        ]
    )
    policy = partition_feedback_rows(
        row_valid=torch.tensor([True, True, False, False, True]),
        truth_status=status,
        definite_nuisance=torch.tensor([False, False, True, False, False]),
    )
    return {
        "query_index": query,
        "pose_family_id": family,
        "can_train_descriptor": True,
        "actual_query_task_gain": 0.5,
        "query_descriptors": torch.vstack(
            (torch.eye(4), torch.tensor([0.8, 0.6, 0.0, 0.0]))
        ),
        "candidate_anchor_rows": torch.tensor(
            [
                [1, 0, 2],
                [2, 1, 0],
                [1, 0, 2],
                [0, 1, 2],
                [1, 2, 0],
            ]
        ),
        "candidate_scores": torch.tensor(
            [
                [0.90, 0.80, 0.20],
                [0.85, 0.40, 0.30],
                [0.9, 0.2, 0.1],
                [0.7, 0.6, 0.5],
                [0.95, 0.40, 0.30],
            ]
        ),
        "truth": {
            "row_count": 5,
            "truth_status": status,
            "truth_offsets": torch.tensor([0, 1, 2, 2, 2, 3]),
            "truth_anchor_rows": torch.tensor([0, 2, 1]),
        },
        "row_policy": policy,
    }


def test_feedback_row_policy_keeps_all_rows_in_plant_but_abstains_safely() -> None:
    policy = _record(0, 10)["row_policy"]
    assert policy["plant_eligible"].tolist() == [True, True, True, True, True]
    assert policy["row_role"].tolist() == [
        ROW_REPAIR,
        ROW_REPAIR,
        ROW_NUISANCE,
        ROW_UNKNOWN,
        ROW_REPAIR,
    ]
    assert policy["counts"] == {
        "plant": 5,
        "repair": 3,
        "nuisance": 1,
        "unknown": 1,
    }


def test_feedback_row_policy_rejects_valid_nuisance_overlap() -> None:
    with pytest.raises(ValueError, match="render-valid row"):
        partition_feedback_rows(
            row_valid=torch.tensor([True]),
            truth_status=torch.tensor([TRUTH_UNIQUE]),
            definite_nuisance=torch.tensor([True]),
        )


def test_topk_evidence_is_multifamily_listwise_and_excludes_nuisance() -> None:
    evidence = build_topk_competition_evidence(
        records=[_record(0, 10), _record(1, 20)],
        anchor_count=3,
        equivalence_class_ids=torch.arange(3),
    )
    assert evidence["repair_query_descriptors"].shape == (2, 4)
    assert evidence["repair_positive_anchor_rows"].tolist() == [0, 0]
    assert evidence["repair_negative_anchor_rows"].tolist() == [1, 2, 1, 2]
    assert evidence["repair_wrong_winner_anchor_rows"].tolist() == [1, 1]
    assert evidence["repair_wrong_winner_clean_support_family_counts"].tolist() == [2, 2]
    assert evidence["negative_action_anchor_rows"].tolist() == [1]
    assert evidence["protection_positive_anchor_rows"].tolist() == [2, 1, 2, 1]
    assert evidence["counts"] == {
        "decisive": 6,
        "competition_miss": 2,
        "correct_winner": 4,
        "nuisance": 2,
        "unknown": 2,
    }
    assert evidence["nuisance_max_scores"].tolist() == pytest.approx([0.9, 0.9])


def test_topk_evidence_rejects_single_family_wrong_winner() -> None:
    with pytest.raises(RuntimeError, match="cross-family"):
        build_topk_competition_evidence(
            records=[_record(0, 10)],
            anchor_count=3,
            equivalence_class_ids=torch.arange(3),
        )


def test_topk_evidence_serializes_same_class_candidates_as_positives() -> None:
    left = _record(0, 10)
    right = _record(1, 20)
    for record in (left, right):
        record["candidate_anchor_rows"][0] = torch.tensor([3, 1, 0])
        record["candidate_scores"][0] = torch.tensor([0.9, 0.8, 0.7])
    evidence = build_topk_competition_evidence(
        records=[left, right],
        anchor_count=4,
        equivalence_class_ids=torch.tensor([0, 0, 2, 3]),
    )
    assert evidence["repair_positive_anchor_rows"].tolist() == [0, 1, 0, 1]
    assert evidence["repair_negative_anchor_rows"].tolist() == [3, 3]


def test_topk_evidence_rejects_overlapping_roles_and_bad_truth_csr() -> None:
    overlap = _record(0, 10)
    overlap["row_policy"]["unknown"] = overlap["row_policy"]["unknown"].clone()
    overlap["row_policy"]["unknown"][0] = True
    with pytest.raises(ValueError, match="disjoint and exhaustive"):
        build_topk_competition_evidence(
            records=[overlap],
            anchor_count=3,
            equivalence_class_ids=torch.arange(3),
        )

    invalid_truth = _record(0, 10)
    invalid_truth["truth"]["truth_anchor_rows"] = torch.tensor([0, 3, 1])
    with pytest.raises(ValueError, match="truth CSR"):
        build_topk_competition_evidence(
            records=[invalid_truth],
            anchor_count=3,
            equivalence_class_ids=torch.arange(3),
        )
