import torch

from localization_training.selection_aware_reconstruction import (
    ROLE_CRITICAL,
    ROLE_HARMFUL,
    ROLE_NEUTRAL,
    SelectionAwareOptimizationConfig,
    SelectionAwareTrainingData,
    bounded_representations,
    build_mode_table,
    selection_aware_ranking_loss,
    selection_role_masks,
    optimize_selection_aware_representations,
    winning_representation,
)


def test_winning_representation_respects_family_bias_and_temperature():
    anchors = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    prototypes = torch.tensor([[0.8, 0.6], [1.0, 0.0]])
    table = build_mode_table(
        anchors,
        prototypes,
        torch.tensor([0, 1]),
        torch.tensor([0.0, -0.2]),
        torch.ones(2),
    )
    representation, score = winning_representation(
        torch.tensor([[0.8, 0.6], [1.0, 0.0]]),
        torch.tensor([0, 1]),
        table,
    )
    assert representation.tolist() == [2, 3]
    assert torch.allclose(score, torch.tensor([1.0, 0.8]), atol=1e-6)


def test_selection_roles_are_disjoint_and_rank_critical_gain():
    roles = selection_role_masks(
        selected=torch.tensor([1, 1, 0, 0, 0], dtype=torch.bool),
        strict_clean=torch.tensor([1, 0, 1, 1, 0], dtype=torch.bool),
        solver_clean=torch.tensor([1, 0, 1, 1, 1], dtype=torch.bool),
        harmful=torch.tensor([0, 1, 0, 0, 0], dtype=torch.bool),
        reserve_gain=torch.tensor([0.0, 0.0, 0.2, 0.8, 1.0]),
        maximum_critical=1,
    )
    assert roles["protected"].tolist() == [True, False, False, False, False]
    assert roles["harmful"].tolist() == [False, True, False, False, False]
    assert roles["critical"].tolist() == [False, False, False, False, True]
    assert not bool(
        (
            roles["protected"]
            & roles["harmful"]
            | roles["protected"]
            & roles["critical"]
            | roles["harmful"]
            & roles["critical"]
        ).any()
    )


def test_bounded_representations_enforces_l2_bound():
    base = torch.eye(3)
    transformed, bounded = bounded_representations(
        base, torch.full_like(base, 100.0), maximum_delta=0.02
    )
    assert torch.all(torch.linalg.norm(bounded, dim=1) <= 0.020001)
    assert torch.allclose(
        torch.linalg.norm(transformed, dim=1), torch.ones(3), atol=1e-6
    )


def test_selection_aware_loss_protects_neutral_and_corrects_active():
    loss, parts = selection_aware_ranking_loss(
        torch.tensor([0.8, 0.2, 0.1]),
        torch.tensor([0.2, 0.3, 0.4]),
        role=torch.tensor([ROLE_NEUTRAL, ROLE_HARMFUL, ROLE_CRITICAL]),
        baseline_margin=torch.tensor([0.7, -0.1, -0.3]),
        weight=torch.ones(3),
        margin=0.05,
        preserve_tolerance=0.02,
        temperature=0.05,
    )
    assert float(loss) > 0
    assert float(parts["preserve"]) > 0
    assert float(parts["active"]) > 0


def test_selection_v2_optimizes_topk_replay_and_basis_hyperedge():
    data = SelectionAwareTrainingData(
        query_features=torch.tensor([[1.0, 0.0]]),
        positive_representation=torch.tensor([0]),
        negative_representation=torch.tensor([1]),
        role=torch.tensor([ROLE_HARMFUL]),
        weight=torch.ones(1),
        baseline_margin=torch.tensor([0.0]),
        diagnostics={},
        replay_query_features=torch.tensor([[1.0, 0.0]]),
        replay_representation_indices=torch.tensor([[0, 1]]),
        replay_baseline_logits=torch.tensor([[0.8, 0.2]]),
        replay_weights=torch.ones(1),
        basis_query_features=torch.tensor(
            [[[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]]
        ),
        basis_positive_representation=torch.tensor([[0, 1, 0]]),
        basis_negative_representation=torch.tensor([[1, 0, 1]]),
        basis_weights=torch.ones(1),
    )
    state = {
        "anchor_features": torch.tensor(
            [[0.8, 0.2], [0.2, 0.8]]
        )
    }
    family = {
        "prototype_features": torch.empty((0, 2)),
        "prototype_bias": torch.empty(0),
        "prototype_temperature": torch.empty(0),
    }
    updated, _, history = optimize_selection_aware_representations(
        state=state,
        family=family,
        data=data,
        config=SelectionAwareOptimizationConfig(
            steps=2,
            batch_size=1,
            learning_rate=0.01,
        ),
        device=torch.device("cpu"),
    )
    assert torch.isfinite(updated["anchor_features"]).all()
    assert "topk_replay_loss" in history[-1]
    assert "basis_hyperedge_loss" in history[-1]
