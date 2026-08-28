import torch

from map_learning.v8_safety_actions import (
    certified_feedback_row_mask,
    evaluate_counterfactual_gaussian_action,
    propose_anchor_quarantine,
)


def test_non_accept_feedback_cannot_update_any_row() -> None:
    certificate = {
        "decision": "UNCERTAIN", "can_drive_map_update": False,
        "row_valid": torch.tensor([True, False, True]),
    }
    assert certified_feedback_row_mask(certificate).tolist() == [False, False, False]


def test_anchor_quarantine_requires_independent_pose_families() -> None:
    proposal = propose_anchor_quarantine(
        anchor_count=4,
        harmful_anchor_rows=torch.tensor([1, 1, 2, 2]),
        pose_family_ids=torch.tensor([3, 4, 5, 5]),
    )
    assert proposal["proposed_quarantine"].tolist() == [False, True, False, False]
    assert proposal["map_mutation_count"] == 0


def test_counterfactual_cleanup_passes_only_as_reversible_quarantine() -> None:
    result = evaluate_counterfactual_gaussian_action(
        baseline_task_error=torch.tensor([1.0, 1.1, 0.8, 0.9]),
        cleaned_task_error=torch.tensor([0.7, 0.8, 0.8, 0.85]),
        pose_family_ids=torch.tensor([0, 1, 0, 1]),
    )
    assert result["decision"] == "PASS"
    assert result["permanent_gaussian_deletion_authorized"] is False


def test_counterfactual_single_family_or_broad_worsening_rolls_back() -> None:
    result = evaluate_counterfactual_gaussian_action(
        baseline_task_error=torch.tensor([1.0, 1.0, 1.0, 1.0]),
        cleaned_task_error=torch.tensor([0.8, 1.2, 1.2, 1.2]),
        pose_family_ids=torch.tensor([0, 0, 0, 0]),
    )
    assert result["decision"] == "ROLLBACK"
    assert result["map_mutation_count"] == 0
