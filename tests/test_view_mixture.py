import torch

from types import SimpleNamespace

import pytest

from evidence.view_mixture import (
    LeaveOneQueryOutViewMixtureMatcher,
    build_view_mixture,
    mixture_scores,
)


def test_view_mixture_requires_independent_bins_per_cluster():
    angles = torch.tensor([0.0, 0.04, 1.0, 1.04])
    descriptors = torch.stack((angles.cos(), angles.sin()), dim=1)
    mixture = build_view_mixture(
        descriptors,
        torch.arange(4),
        torch.ones(4),
        minimum_angle_degrees=10.0,
        minimum_loss_improvement=0.001,
    )
    assert mixture.eligible
    assert mixture.prototypes.shape == (2, 2)
    assert torch.isclose(mixture.priors.sum(), torch.tensor(1.0))


def test_single_mode_falls_back_to_one_prototype():
    descriptor = torch.tensor([[1.0, 0.0], [1.0, 0.01]])
    mixture = build_view_mixture(descriptor, torch.tensor([0, 1]), torch.ones(2))
    assert not mixture.eligible
    assert mixture.prototypes.shape[0] == 1


def test_mixture_aggregation_preserves_one_anchor_score():
    query = torch.tensor([[1.0, 0.0]])
    prototypes = torch.tensor([[[1.0, 0.0], [0.0, 1.0]], [[0.8, 0.6], [0.0, 0.0]]])
    priors = torch.tensor([[0.5, 0.5], [1.0, 0.0]])
    scores = mixture_scores(query, prototypes, priors, temperature=0.05)
    assert scores.shape == (1, 2)
    assert scores[0, 0] > scores[0, 1]
    expected = torch.nn.functional.normalize(query, dim=1) @ torch.nn.functional.normalize(
        prototypes[1, 0][None].float(), dim=1
    ).T
    assert torch.equal(scores[:, 1], expected[:, 0])


def _replay_matcher_for_update(*, budget_extra=1):
    matcher = LeaveOneQueryOutViewMixtureMatcher.__new__(
        LeaveOneQueryOutViewMixtureMatcher
    )
    matcher.device = torch.device("cpu")
    matcher.thresholds = dict(
        minimum_cluster_observations=2,
        minimum_cluster_view_bins=2,
        minimum_angle_degrees=12.0,
        minimum_loss_improvement=0.015,
    )
    matcher.budget_extra = budget_extra
    matcher.base_eligible = {0}
    matcher.maximum_query_local_eligible = 1
    matcher.base_prototypes = torch.tensor(
        [[[1.0, 0.0], [0.0, 0.0]], [[1.0, 0.0], [0.0, 1.0]]]
    )
    matcher.base_priors = torch.tensor([[1.0, 0.0], [0.5, 0.5]])
    matcher.prototypes = matcher.base_prototypes.clone()
    matcher.priors = matcher.base_priors.clone()
    matcher.previous_rows = torch.empty(0, dtype=torch.long)
    # local Track row 0 maps to unified row 1; unified row 0 is surface K1.
    matcher.track_replay = SimpleNamespace(rows_by_query=[[0], []])
    matcher.replay = SimpleNamespace(
        track_rows=torch.tensor([1]),
        query_update=lambda query: (
            (torch.tensor([0, 1]), torch.tensor([[0.0, 1.0], [1.0, 0.0]]))
            if query == 0 else (torch.empty(0, dtype=torch.long), torch.empty(0, 2))
        ),
    )
    descriptors = torch.tensor([[1.0, 0.0], [0.99, 0.01], [0.0, 1.0], [0.01, 0.99]])
    matcher.observations = {
        0: (descriptors, torch.tensor([0, 1, 2, 3]), torch.ones(4), torch.tensor([0, 0, 0, 0]))
    }
    return matcher


def test_query_update_keeps_surface_k1_and_restores_next_query():
    matcher = _replay_matcher_for_update()
    matcher._update(0)
    assert torch.equal(matcher.priors[0], torch.tensor([1.0, 0.0]))
    assert torch.equal(matcher.prototypes[0, 0], torch.tensor([0.0, 1.0]))
    # Current query owns every Track observation, so its K2 row falls to K1.
    assert torch.equal(matcher.priors[1], torch.tensor([1.0, 0.0]))
    matcher._update(1)
    assert torch.equal(matcher.priors[1], torch.tensor([0.5, 0.5]))


def test_query_local_budget_saturation_is_rejected():
    matcher = _replay_matcher_for_update(budget_extra=0)
    matcher.base_eligible = set()
    matcher.track_replay.rows_by_query = [[0], []]
    descriptors, bins, weights, _ = matcher.observations[0]
    matcher.observations[0] = (
        descriptors, bins, weights, torch.tensor([1, 1, 1, 1])
    )
    with pytest.raises(ValueError, match="saturates budget"):
        matcher._update(0)
