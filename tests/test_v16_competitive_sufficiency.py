import torch

from map_learning.v16_competitive_sufficiency import (
    competitive_reserve_state,
    reserve_transition_is_safe,
)


def _state(active: torch.Tensor) -> dict:
    # B (row 1) is the strongest certified positive. A (row 0) is geometrically
    # redundant but cannot beat wrong competitor C (row 2) after B is removed.
    return competitive_reserve_state(
        candidate_anchor_rows=torch.tensor([[1, 2, 0]]),
        candidate_scores=torch.tensor([[0.90, 0.85, 0.80]]),
        certified_positive=torch.tensor([[True, False, True]]),
        active_anchor_mask=active,
        keypoints=torch.tensor([[20.0, 20.0]]),
        anchor_xyz=torch.tensor(
            [[0.0, 0.0, 5.0], [0.1, 0.0, 5.0], [3.0, 0.0, 5.0]]
        ),
        intrinsic=torch.tensor(
            [[100.0, 0.0, 20.0], [0.0, 100.0, 20.0], [0.0, 0.0, 1.0]]
        ),
        pose_w2c=torch.eye(4),
        image_hw=torch.tensor([40, 40]),
        margin_delta=0.01,
    )


def test_deleting_correct_winner_exposes_wrong_competitor() -> None:
    before = _state(torch.ones(3, dtype=torch.bool))
    after = _state(torch.tensor([True, False, True]))
    assert before["winner_anchor_rows"].tolist() == [1]
    assert before["winner_certified_positive"].tolist() == [True]
    assert before["safe_positive_count_per_row"].tolist() == [1]
    assert after["winner_anchor_rows"].tolist() == [2]
    assert after["winner_certified_positive"].tolist() == [False]
    assert after["safe_positive_count_per_row"].tolist() == [0]
    safe, reasons = reserve_transition_is_safe(before, after)
    assert safe is False
    assert "correct_winner_lost" in reasons


def test_deleting_wrong_competitor_can_increase_positive_reserve() -> None:
    before = _state(torch.ones(3, dtype=torch.bool))
    after = _state(torch.tensor([True, True, False]))
    assert after["winner_anchor_rows"].tolist() == [1]
    assert after["safe_positive_count_per_row"].tolist() == [2]
    safe, reasons = reserve_transition_is_safe(before, after)
    assert safe is True
    assert reasons == []
