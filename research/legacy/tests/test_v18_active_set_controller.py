import torch

from map_learning.v18_active_set_controller import (
    apply_reversible_active_action,
    propose_current_competition_actions,
    propose_truth_reactivation_actions,
)
from map_learning.v18_provenance_truth import TRUTH_EQUIVALENT, TRUTH_UNIQUE
from scripts.run_v18_truth_aware_controller import _action_proposals


def test_current_graph_proposes_harmful_dominated_and_inactive_actions() -> None:
    records = []
    for query, family in ((4, 10), (7, 11)):
        records.append(
            {
                "query_index": query,
                "pose_family_id": family,
                "candidate_anchor_rows": torch.tensor([[3, 0, 1, 2]]),
                "candidate_scores": torch.tensor([[0.9, 0.8, 0.7, 0.6]]),
                "truth": {
                    "truth_status": torch.tensor([TRUTH_EQUIVALENT]),
                    "truth_offsets": torch.tensor([0, 2]),
                    "truth_anchor_rows": torch.tensor([0, 1]),
                },
            }
        )
    proposals = propose_current_competition_actions(
        records=records,
        anchor_count=4,
    )
    assert proposals["historical_removal_audit_used"] is False
    by_anchor = {
        item["anchor_row"]: item["kind"] for item in proposals["proposals"]
    }
    assert by_anchor == {
        3: "harmful_removal",
        1: "dominated_removal",
        2: "inactive_redundancy_probe",
    }
    reversed_proposals = propose_current_competition_actions(
        records=list(reversed(records)),
        anchor_count=4,
    )
    assert torch.equal(
        reversed_proposals["proposal_anchor_rows"],
        proposals["proposal_anchor_rows"],
    )


def test_active_action_supports_reactivation_and_swap() -> None:
    after, audit = apply_reversible_active_action(
        torch.tensor([True, False, True]),
        deactivate_rows=torch.tensor([2]),
        reactivate_rows=torch.tensor([1]),
    )
    assert after.tolist() == [True, True, False]
    assert audit["before_active_count"] == audit["after_active_count"] == 2
    assert audit["reversible"] is True


def test_reactivation_requires_inactive_truth_across_families() -> None:
    records = []
    for query, family in ((0, 10), (1, 20)):
        records.append(
            {
                "query_index": query,
                "pose_family_id": family,
                "candidate_anchor_rows": torch.tensor([[1, 0, 2]]),
                "current_winner_anchor_rows": torch.tensor([1]),
                "truth": {
                    "truth_status": torch.tensor([TRUTH_UNIQUE]),
                    "truth_offsets": torch.tensor([0, 1]),
                    "truth_anchor_rows": torch.tensor([0]),
                },
            }
        )
    proposals = propose_truth_reactivation_actions(
        records=records,
        anchor_count=3,
        active_anchor_mask=torch.tensor([False, True, True]),
    )
    assert proposals["proposal_count"] == 1
    assert proposals["proposals"][0]["anchor_row"] == 0
    assert proposals["proposals"][0]["kind"] == "truth_reactivation"


def test_unified_controller_does_not_starve_reactivation() -> None:
    records = []
    for query, family in ((0, 10), (1, 20)):
        records.append(
            {
                "query_index": query,
                "pose_family_id": family,
                "candidate_anchor_rows": torch.tensor([[1, 0, 2]]),
                "candidate_scores": torch.tensor([[0.9, 0.8, 0.7]]),
                "current_winner_anchor_rows": torch.tensor([1]),
                "current_winner_available": torch.tensor([True]),
                "truth": {
                    "truth_status": torch.tensor([TRUTH_UNIQUE]),
                    "truth_offsets": torch.tensor([0, 1]),
                    "truth_anchor_rows": torch.tensor([0]),
                },
            }
        )
    proposals = _action_proposals(
        records,
        active=torch.tensor([False, True, True]),
        anchor_count=3,
        maximum_inactive=8,
    )
    assert proposals[0]["kind"] == "truth_reactivation"
    assert proposals[0]["anchor_row"] == 0
