import torch

from map_learning.v15_global_sufficiency import (
    feedback_conditioned_reliability,
    feedback_utility_components,
    size_aware_supervision,
)


def _record(family: int, gain: float = 2.0) -> dict:
    return {
        "certificate_decision": "ACCEPT",
        "pose_family_id": family,
        "can_train_metric": True,
        "clean_protection_evidence": {"positive_anchor_rows": torch.tensor([1, 1])},
        "training_evidence": {
            "positive_anchor_rows": torch.tensor([1, 1, 2]),
            "actual_query_task_gain": gain,
        },
    }


def test_feedback_credit_is_family_unique_and_query_bounded() -> None:
    result = feedback_utility_components(
        [_record(7, 100.0), _record(7, 1.0), _record(8, 2.0)],
        anchor_count=4,
        harmful_anchor_rows=torch.tensor([3]),
    )
    assert result["clean_pose_family_count"].tolist() == [0.0, 2.0, 0.0, 0.0]
    assert result["task_pose_family_count"].tolist() == [0.0, 2.0, 2.0, 0.0]
    assert result["bounded_task_gain"][1].item() == 3.0
    assert result["causally_harmful"].tolist() == [False, False, False, True]


def test_harmful_anchor_is_demoted_but_not_made_ineligible() -> None:
    components = feedback_utility_components(
        [_record(1)], anchor_count=4, harmful_anchor_rows=torch.tensor([3])
    )
    score = feedback_conditioned_reliability(torch.ones(4), components)
    assert score[1] > score[0]
    assert score[3] < score[0]
    assert torch.isfinite(score).all()


def test_map_size_is_an_effect_but_never_excuses_task_regression() -> None:
    supervision = {
        "baseline": {"total_risk": 1.0},
        "candidate": {"total_risk": 0.995},
        "hard_checks": {"safe": True},
        "bootstrap_probability_lower_risk": 0.9,
        "classification": "NO_ACTION",
    }
    result = size_aware_supervision(supervision, compression_fraction=0.25)
    assert result["classification"] == "PARETO_CANDIDATE"
    assert result["task_only_classification"] == "NO_ACTION"

    supervision["candidate"]["total_risk"] = 1.001
    rejected = size_aware_supervision(supervision, compression_fraction=0.25)
    assert rejected["classification"] == "NO_ACTION"
    assert rejected["size_safe_and_better"] is False
