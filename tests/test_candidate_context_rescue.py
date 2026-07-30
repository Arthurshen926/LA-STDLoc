import torch

from localization_training.candidate_context_rescue import (
    CandidateConditionedContextScorer,
    DirectedConfusionEdge,
    candidate_conditioned_rescue,
    candidate_conditioned_rescue_from_edge_keys,
    directed_edge_keys,
    oracle_acceptance_mask,
)


def test_rescue_only_changes_a_directed_pair_winner():
    candidates = torch.tensor([[10, 11, 12], [20, 21, 22]])
    local = torch.tensor([[0.80, 0.78, 0.70], [0.90, 0.89, 0.88]])
    context = torch.tensor([[0.10, 0.90, 1.00], [0.10, 1.00, 1.00]])
    edges = [
        (DirectedConfusionEdge(correct_anchor=11, confusing_anchor=10),),
        (),
    ]
    indices, scores, diagnostics = candidate_conditioned_rescue(
        candidates,
        local,
        context,
        edges,
        context_weight=0.05,
        maximum_score_delta=0.05,
    )
    assert int(indices[0, 0]) == 11
    assert torch.equal(indices[1], candidates[1])
    assert torch.equal(scores[1], local[1])
    assert diagnostics["changed_row_count"] == 1


def test_rescue_cannot_promote_an_unlabelled_candidate():
    candidates = torch.tensor([[10, 11, 12]])
    local = torch.tensor([[0.80, 0.79, 0.78]])
    context = torch.tensor([[0.00, 0.10, 1.00]])
    edges = [
        (DirectedConfusionEdge(correct_anchor=11, confusing_anchor=10),)
    ]
    indices, _, _ = candidate_conditioned_rescue(
        candidates,
        local,
        context,
        edges,
        context_weight=0.05,
    )
    assert int(indices[0, 0]) == 10
    assert 12 not in indices[0, :1]


def test_strict_oracle_requires_cleaner_two_pixel_measurement():
    accepted = oracle_acceptance_mask(
        torch.tensor([8.0, 1.5, 8.0, 8.0, 1.5]),
        torch.tensor([1.5, 1.0, 3.0, 1.0, 1.0]),
        torch.tensor([True, True, True, False, False]),
        strict_threshold_px=2.0,
    )
    assert accepted.tolist() == [True, False, False, False, False]


def test_candidate_conditioned_scorer_checks_pair_shapes():
    scorer = CandidateConditionedContextScorer(
        context_dim=8, scalar_dim=3, hidden_dim=16
    )
    output = scorer(
        torch.randn(4, 8),
        torch.randn(4, 8),
        torch.randn(4, 8),
        torch.randn(4, 3),
    )
    assert output.shape == (4,)


def test_vectorized_edge_rescue_matches_directed_pair_contract():
    candidates = torch.tensor([[10, 11, 12], [20, 21, 22]])
    local = torch.tensor([[0.80, 0.78, 0.70], [0.90, 0.89, 0.88]])
    context = torch.tensor([[0.10, 0.90, 1.00], [0.10, 1.00, 1.00]])
    keys = directed_edge_keys(
        [{"correct_anchor": 11, "confusing_anchor": 10}],
        anchor_count=32,
    )
    indices, scores, diagnostics = (
        candidate_conditioned_rescue_from_edge_keys(
            candidates,
            local,
            context,
            keys,
            anchor_count=32,
            context_weight=0.05,
        )
    )
    assert int(indices[0, 0]) == 11
    assert torch.equal(indices[1], candidates[1])
    assert torch.equal(scores[1], local[1])
    assert diagnostics["changed_row_count"] == 1

    loop_indices, loop_scores, _ = candidate_conditioned_rescue(
        candidates,
        local,
        context,
        [
            (DirectedConfusionEdge(11, 10),),
            (),
        ],
        context_weight=0.05,
    )
    assert torch.equal(indices, loop_indices)
    assert torch.equal(scores, loop_scores)
