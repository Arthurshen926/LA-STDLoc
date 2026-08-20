import torch

from evidence.view_mixture import build_view_mixture, mixture_scores


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
