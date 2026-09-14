from __future__ import annotations

import pytest
import torch

from map_learning.v21_true_anchor_rank import exact_global_true_anchor_ranks


def test_exact_global_true_anchor_ranks_reports_strict_and_stable_tie_ranks() -> None:
    anchors = torch.tensor(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ]
    )
    queries = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ]
    )
    truth = torch.tensor([1, 3, 2])

    result = exact_global_true_anchor_ranks(
        query_descriptors=queries,
        anchor_features=anchors,
        true_anchor_rows=truth,
        query_batch_size=2,
        anchor_chunk_size=2,
    )

    # Duplicate row 0 ties row 1: strict rank remains one while the production
    # row-order tie break places the true row second.
    assert result["strict_ranks"].tolist() == [1, 2, 1]
    assert result["lower_row_equal_score_counts"].tolist() == [1, 0, 0]
    assert result["exact_tie_break_ranks"].tolist() == [2, 2, 1]


@pytest.mark.parametrize(
    ("queries", "anchors"),
    [
        (torch.tensor([[0.0, 0.0]]), torch.tensor([[1.0, 0.0]])),
        (torch.tensor([[1.0, 0.0]]), torch.tensor([[0.0, 0.0]])),
    ],
)
def test_exact_global_true_anchor_ranks_rejects_zero_norm_vectors(
    queries: torch.Tensor, anchors: torch.Tensor
) -> None:
    with pytest.raises(ValueError, match="registries differ"):
        exact_global_true_anchor_ranks(
            query_descriptors=queries,
            anchor_features=anchors,
            true_anchor_rows=torch.tensor([0]),
        )


def test_exact_global_true_anchor_ranks_rejects_bad_true_anchor_row() -> None:
    with pytest.raises(ValueError, match="registries differ"):
        exact_global_true_anchor_ranks(
            query_descriptors=torch.tensor([[1.0, 0.0]]),
            anchor_features=torch.tensor([[1.0, 0.0]]),
            true_anchor_rows=torch.tensor([1]),
        )
