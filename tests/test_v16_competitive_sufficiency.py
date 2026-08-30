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


def test_registered_safety_floor_can_consume_redundant_reserve() -> None:
    before = {
        "topl_exhausted": torch.tensor([False]),
        "winner_certified_positive": torch.tensor([True]),
        "anchor_unique_safe_count": 20,
        "spatial_cell_count": 10,
        "pose_logdet": 12.0,
        "pose_minimum_eigenvalue": 4.0,
        "pose_effective_correspondence_count": 18.0,
    }
    after = {
        **before,
        "anchor_unique_safe_count": 16,
        "spatial_cell_count": 8,
        "pose_logdet": 11.7,
        "pose_minimum_eigenvalue": 3.5,
        "pose_effective_correspondence_count": 15.0,
    }
    strict, _ = reserve_transition_is_safe(before, after)
    assert strict is False
    safe, reasons = reserve_transition_is_safe(
        before,
        after,
        minimum_anchor_unique_safe_count=12,
        minimum_spatial_cell_count=6,
        maximum_pose_logdet_drop=0.5,
        minimum_pose_eigenvalue_retention=0.8,
        minimum_effective_correspondence_count=12.0,
    )
    assert safe is True
    assert reasons == []


def test_compressed_start_can_reduce_but_not_add_topl_exhaustion() -> None:
    common = {
        "winner_certified_positive": torch.tensor([False, True]),
        "anchor_unique_safe_count": 4,
        "spatial_cell_count": 3,
        "pose_logdet": 2.0,
        "pose_minimum_eigenvalue": 0.1,
        "pose_effective_correspondence_count": 4.0,
    }
    before = {**common, "topl_exhausted": torch.tensor([True, False])}
    repairing = {**common, "topl_exhausted": torch.tensor([True, False])}
    safe, reasons = reserve_transition_is_safe(
        before,
        repairing,
        minimum_anchor_unique_safe_count=12,
        minimum_spatial_cell_count=6,
        maximum_pose_logdet_drop=0.5,
        minimum_pose_eigenvalue_retention=0.8,
        minimum_effective_correspondence_count=12.0,
    )
    assert safe is True
    assert reasons == []
    regressing = {**common, "topl_exhausted": torch.tensor([True, True])}
    safe, reasons = reserve_transition_is_safe(before, regressing)
    assert safe is False
    assert "new_topl_exhaustion" in reasons


def test_pose_reserve_propagates_certification_confidence() -> None:
    common = {
        "candidate_anchor_rows": torch.tensor([[0], [1]]),
        "candidate_scores": torch.tensor([[0.9], [0.9]]),
        "certified_positive": torch.tensor([[True], [True]]),
        "active_anchor_mask": torch.ones(2, dtype=torch.bool),
        "keypoints": torch.tensor([[10.0, 10.0], [30.0, 20.0]]),
        "anchor_xyz": torch.tensor([[0.0, 0.0, 5.0], [1.0, 0.5, 5.0]]),
        "intrinsic": torch.tensor(
            [[100.0, 0.0, 10.0], [0.0, 100.0, 10.0], [0.0, 0.0, 1.0]]
        ),
        "pose_w2c": torch.eye(4),
        "image_hw": torch.tensor([40, 40]),
    }
    full = competitive_reserve_state(
        **common,
        certification_confidence=torch.ones(2, 1),
        measurement_covariance_px2=torch.eye(2).repeat(2, 1, 1, 1),
    )
    uncertain = competitive_reserve_state(
        **common,
        certification_confidence=torch.tensor([[1.0], [0.1]]),
        measurement_covariance_px2=torch.eye(2).repeat(2, 1, 1, 1),
    )
    assert full["pose_effective_correspondence_count"] == 2.0
    assert uncertain["pose_effective_correspondence_count"] < 2.0
    assert uncertain["pose_logdet"] < full["pose_logdet"]
