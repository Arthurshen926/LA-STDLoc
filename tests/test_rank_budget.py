import torch

from localization_training.rank_budget import (
    csr_best_positive_scores,
    multi_positive_rank_budget_loss,
)


def _geometry(row_count, bank_count):
    query_uv = torch.zeros(row_count, 2)
    bank_uv = torch.full((bank_count, 2), 100.0)
    bank_projected = torch.ones(bank_count, dtype=torch.bool)
    return query_uv, bank_uv, bank_projected


def test_csr_best_positive_uses_all_positives():
    scores = torch.tensor([[0.1, 0.7, 0.4], [0.8, 0.2, 0.3]])
    offsets = torch.tensor([0, 2, 3])
    indices = torch.tensor([0, 1, 2])
    best, valid = csr_best_positive_scores(scores, offsets, indices)
    torch.testing.assert_close(best, torch.tensor([0.7, 0.3]))
    assert valid.tolist() == [True, True]


def test_rank_budget_reports_exact_recall_and_bands():
    scores = torch.tensor(
        [
            [0.9, 0.8, 0.7, 0.6, 0.5],
            [0.1, 0.9, 0.8, 0.7, 0.6],
        ],
        requires_grad=True,
    )
    query_uv, bank_uv, bank_projected = _geometry(2, 5)
    result = multi_positive_rank_budget_loss(
        scores,
        positive_offsets=torch.tensor([0, 1, 2]),
        positive_indices=torch.tensor([0, 0]),
        query_uv=query_uv,
        bank_uv=bank_uv,
        bank_projected=bank_projected,
    )
    assert result.ranks.tolist() == [1, 5]
    assert result.diagnostics["rank_budget_recall_at_1"] == 0.5
    assert result.diagnostics["rank_budget_recall_at_4"] == 0.5
    assert result.diagnostics["rank_budget_recall_at_8"] == 1.0
    result.loss.backward()
    assert torch.isfinite(scores.grad).all()


def test_rank_budget_ignores_near_ambiguous_landmark():
    scores = torch.tensor([[0.5, 0.9, 0.4]], requires_grad=True)
    query_uv = torch.tensor([[0.0, 0.0]])
    bank_uv = torch.tensor([[0.0, 0.0], [3.0, 0.0], [20.0, 0.0]])
    result = multi_positive_rank_budget_loss(
        scores,
        positive_offsets=torch.tensor([0, 1]),
        positive_indices=torch.tensor([0]),
        query_uv=query_uv,
        bank_uv=bank_uv,
        bank_projected=torch.ones(3, dtype=torch.bool),
        negative_radius_px=6.0,
    )
    assert result.ranks.item() == 1


def test_rank_budget_empty_positive_rows_are_excluded():
    scores = torch.tensor([[0.3, 0.2], [0.4, 0.1]], requires_grad=True)
    query_uv, bank_uv, bank_projected = _geometry(2, 2)
    result = multi_positive_rank_budget_loss(
        scores,
        positive_offsets=torch.tensor([0, 0, 1]),
        positive_indices=torch.tensor([1]),
        query_uv=query_uv,
        bank_uv=bank_uv,
        bank_projected=bank_projected,
    )
    assert result.valid_rows.tolist() == [False, True]
    assert result.diagnostics["rank_budget_matchable_count"] == 1


def test_rank_budget_persistently_protects_reference_clean_rows():
    reference = torch.tensor(
        [
            [0.9, 0.8, 0.1],
            [0.2, 0.9, 0.1],
        ]
    )
    scores = torch.tensor(
        [
            [0.7, 0.8, 0.1],
            [0.3, 0.8, 0.1],
        ],
        requires_grad=True,
    )
    query_uv, bank_uv, bank_projected = _geometry(2, 3)
    result = multi_positive_rank_budget_loss(
        scores,
        positive_offsets=torch.tensor([0, 1, 2]),
        positive_indices=torch.tensor([0, 1]),
        query_uv=query_uv,
        bank_uv=bank_uv,
        bank_projected=bank_projected,
        reference_scores=reference,
        reference_clean_weight=2.0,
        reference_clean_margin=0.02,
    )
    assert result.diagnostics["rank_budget_reference_clean_count"] == 2
    assert result.diagnostics["rank_budget_reference_clean_retained_count"] == 1
    assert result.diagnostics["rank_budget_reference_clean_retention"] == 0.5
    assert result.diagnostics["rank_budget_reference_clean_loss"] > 0.0
    result.loss.backward()
    assert scores.grad[0, 0] < 0.0
    assert scores.grad[0, 1] > 0.0
