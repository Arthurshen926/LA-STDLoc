import torch

from map_learning.v6_feedback_evaluator import (
    _maximum_matching,
    _positive_score_statistics,
)


def test_maximum_matching_prevents_duplicate_anchor_votes() -> None:
    count, pairs = _maximum_matching([[0, 1], [0], [1, 2]])
    assert count == 3
    assert len({anchor for _, anchor in pairs}) == 3


def test_positive_statistics_match_stable_argsort_with_ties() -> None:
    scores = torch.tensor([0.7, 0.9, 0.9, -0.1, 0.8, 0.9])
    positives = [2, 4, 5]
    best_positive, best_wrong, rank, best_anchor = _positive_score_statistics(
        scores, positives
    )
    order = torch.argsort(scores, descending=True, stable=True)
    oracle_rank = min(
        int(torch.nonzero(order == anchor, as_tuple=False)[0]) + 1
        for anchor in positives
    )
    wrong = scores.clone()
    wrong[torch.tensor(positives)] = -torch.inf
    assert best_positive == float(scores[positives].max())
    assert best_wrong == float(wrong.max())
    assert rank == oracle_rank
    assert best_anchor == 2
