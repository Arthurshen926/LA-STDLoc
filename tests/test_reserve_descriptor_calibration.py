import torch

from map_learning.reserve_descriptor_calibration import (
    optimize_bounded_descriptor,
    validate_descriptor,
)


def _evidence(descriptor, competitor):
    return {
        "query": 0,
        "descriptor": torch.as_tensor(descriptor).float(),
        "competitor_score": float(competitor),
        "old_score": 0.0,
    }


def test_bounded_descriptor_moves_toward_positive_and_away_from_negative():
    initial = torch.tensor([1.0, 0.0])
    positive = torch.tensor([[0.8, 0.6], [0.7, 0.7]])
    negative = torch.tensor([[1.0, -0.1], [0.9, -0.2]])
    result, report = optimize_bounded_descriptor(
        initial,
        positive,
        negative,
        maximum_residual_norm=0.1,
        margin=0.03,
        temperature=0.04,
        trust_weight=0.1,
        steps=80,
        learning_rate=0.02,
    )
    assert report["residual_norm"] <= 0.100001
    assert float((positive @ result).mean()) > float((positive @ initial).mean())
    assert float((negative @ result).mean()) < float((negative @ initial).mean())


def test_validation_requires_false_winner_reduction_and_positive_protection():
    initial = torch.tensor([1.0, 0.0])
    candidate = torch.nn.functional.normalize(torch.tensor([1.0, 0.1]), dim=0)
    positives = [_evidence([0.9, 0.1], 0.85), _evidence([0.8, 0.2], 0.78)]
    negatives = [_evidence([0.8, -0.6], 0.79)]
    report = validate_descriptor(
        initial,
        candidate,
        positives,
        negatives,
        maximum_positive_score_drop=0.01,
    )
    assert report["new_positive_wins"] >= report["old_positive_wins"]
    assert report["new_false_wins"] < report["old_false_wins"]
    assert report["accepted"]
