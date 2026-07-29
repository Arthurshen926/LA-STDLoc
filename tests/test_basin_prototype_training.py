import torch
import torch.nn.functional as F

from localization_training.prototype_optimization import (
    _deployed_pair_scores,
    hyperedge_loss,
    materialize_prototypes,
    teacher_set_scores,
)


def test_prototype_materialization_enforces_bounds():
    features, bias, temperature, bounded = materialize_prototypes(
        torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        torch.tensor([[100.0, 0.0], [0.0, -100.0]]),
        torch.tensor([100.0, -100.0]),
        torch.tensor([100.0, -100.0]),
        maximum_residual=0.05,
        maximum_negative_bias=0.12,
        minimum_temperature=0.85,
        maximum_temperature=1.15,
    )
    assert torch.allclose(torch.linalg.norm(features, dim=1), torch.ones(2))
    assert float(bounded.norm(dim=1).max()) <= 0.050001
    assert bool((bias <= 0).all()) and bool((bias >= -0.12).all())
    assert bool((temperature >= 0.85).all())
    assert bool((temperature <= 1.15).all())


def test_prototype_residual_has_gradient_at_zero_initialization():
    residual = torch.zeros((1, 2), requires_grad=True)
    features, _, _, _ = materialize_prototypes(
        torch.tensor([[1.0, 0.0]]),
        residual,
        torch.zeros(1),
        torch.zeros(1),
        maximum_residual=0.05,
        maximum_negative_bias=0.12,
        minimum_temperature=0.85,
        maximum_temperature=1.15,
    )
    features[0, 1].backward()
    assert residual.grad is not None
    assert residual.grad[0, 1] > 0.9


def test_contrastive_smoothmax_reaches_nonwinning_secondary_mode():
    query = torch.tensor([[1.0, 0.0]])
    prototype = torch.tensor([[0.7, 0.7]], requires_grad=True)
    score = _deployed_pair_scores(
        query=query,
        anchors=torch.tensor([0]),
        bank=F.normalize(torch.tensor([[0.8, 0.6]]), dim=1),
        family_features=F.normalize(prototype, dim=1),
        family_parents=torch.tensor([0]),
        family_bias=torch.zeros(1),
        family_temperature=torch.ones(1),
        smoothmax_temperature=0.05,
    )
    score.backward()
    assert prototype.grad is not None
    assert float(prototype.grad.norm()) > 0


def test_teacher_set_score_uses_secondary_bias():
    query = F.normalize(torch.tensor([[[0.0, 1.0]] * 3]), dim=2)
    score = teacher_set_scores(
        query,
        F.normalize(torch.tensor([[1.0, 0.0]]), dim=1),
        F.normalize(torch.tensor([[0.0, 1.0]]), dim=1),
        torch.tensor([-0.1]),
        torch.ones(1),
        torch.zeros((1, 3), dtype=torch.long),
        torch.zeros((1, 3), dtype=torch.long),
    )
    assert torch.allclose(score, torch.tensor([2.7]))


def test_sibling_loss_prefers_repaired_set():
    _, bad = hyperedge_loss(
        torch.tensor([1.0, 0.0]),
        torch.tensor([1, 2]),
        torch.tensor([0, 3]),
        torch.tensor([100.0, 4.0]),
        torch.tensor([20.0, 1.0]),
        torch.tensor([-1, 0]),
        margin=0.05,
        translation_scale_cm=15.0,
        rotation_scale_deg=2.0,
    )
    _, good = hyperedge_loss(
        torch.tensor([0.0, 1.0]),
        torch.tensor([1, 2]),
        torch.tensor([0, 3]),
        torch.tensor([100.0, 4.0]),
        torch.tensor([20.0, 1.0]),
        torch.tensor([-1, 0]),
        margin=0.05,
        translation_scale_cm=15.0,
        rotation_scale_deg=2.0,
    )
    assert good < bad
