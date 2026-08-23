import torch

from map_learning.v6_feedback_evaluator import (
    _layer_edges,
    _maximum_matching,
    _positive_score_statistics,
)


def test_maximum_matching_prevents_duplicate_anchor_votes() -> None:
    count, pairs = _maximum_matching([[0, 1], [0], [1, 2]])
    assert count == 3
    assert len({anchor for _, anchor in pairs}) == 3


def test_spatial_positive_edges_match_dense_distance_oracle() -> None:
    generator = torch.Generator().manual_seed(91)
    keypoints = torch.rand((37, 2), generator=generator) * 100
    projected = torch.rand((211, 2), generator=generator) * 100
    visible = torch.arange(0, 211, 2, dtype=torch.long)
    radius = 7.0
    actual = _layer_edges(keypoints, projected, visible, radius)
    distance = torch.cdist(keypoints, projected[visible])
    expected = [
        visible[torch.nonzero(distance[row] <= radius).reshape(-1)].tolist()
        for row in range(keypoints.shape[0])
    ]
    assert actual == expected


def test_positive_statistics_match_stable_argsort_with_ties() -> None:
    scores = torch.tensor(
        [
            [0.7, 0.9, 0.9, -0.1, 0.8, 0.9],
            [0.2, 0.1, 0.3, 0.3, -0.5, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    positives_by_row = [[2, 4, 5], [], [3, 5]]
    result = _positive_score_statistics(scores, positives_by_row, chunk_size=1)
    assert set(result) == {0, 2}
    for row, positives in enumerate(positives_by_row):
        if not positives:
            continue
        order = torch.argsort(scores[row], descending=True, stable=True)
        oracle_rank = min(
            int(torch.nonzero(order == anchor, as_tuple=False)[0]) + 1
            for anchor in positives
        )
        wrong = scores[row].clone()
        wrong[torch.tensor(positives)] = -torch.inf
        positive, best_wrong, rank, best_anchor = result[row]
        positive_scores = scores[row, positives]
        expected_positive = positive_scores.max()
        expected_anchor = min(
            anchor
            for anchor in positives
            if scores[row, anchor] == expected_positive
        )
        assert positive == float(expected_positive)
        assert best_wrong == float(wrong.max())
        assert rank == oracle_rank
        assert best_anchor == expected_anchor
