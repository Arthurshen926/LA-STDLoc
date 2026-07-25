import torch

from localization_training.functional_replay import (
    per_landmark_gradient_conflict,
    protected_functional_replay_loss,
)


def test_functional_replay_detects_and_repairs_identity_forgetting():
    bank = torch.tensor([[0.6, 0.8], [1.0, 0.0], [0.0, 1.0]], requires_grad=True)
    query = torch.tensor([[1.0, 0.0]])
    result = protected_functional_replay_loss(
        bank,
        query,
        protected_landmark_indices=torch.tensor([0]),
        reference_candidate_indices=torch.tensor([[0, 1, 2]]),
        reference_candidate_logits=torch.tensor([[0.9, 0.8, 0.1]]),
        reference_margins=torch.tensor([0.1]),
        temperature=0.05,
    )

    assert result.retained.tolist() == [False]
    assert result.loss > 0.0
    result.loss.backward()
    assert bank.grad[0, 0] < 0.0


def test_functional_replay_zero_drift_has_small_distribution_loss():
    bank = torch.eye(3, requires_grad=True)
    query = torch.tensor([[1.0, 0.0, 0.0]])
    reference_logits = torch.tensor([[1.0, 0.0, 0.0]])
    result = protected_functional_replay_loss(
        bank,
        query,
        protected_landmark_indices=torch.tensor([0]),
        reference_candidate_indices=torch.tensor([[0, 1, 2]]),
        reference_candidate_logits=reference_logits,
        reference_margins=torch.tensor([1.0]),
        temperature=0.1,
    )

    assert result.retained.tolist() == [True]
    torch.testing.assert_close(result.margin_loss, torch.tensor(0.0))
    torch.testing.assert_close(result.distribution_loss, torch.tensor(0.0))


def test_per_landmark_gradient_projection_removes_conflict_only():
    promotion = torch.tensor([[1.0, 0.0], [1.0, 1.0]])
    protection = torch.tensor([[-1.0, 0.0], [1.0, 0.0]])
    projected, conflict = per_landmark_gradient_conflict(
        promotion, protection
    )

    assert conflict.tolist() == [True, False]
    torch.testing.assert_close(projected[0], torch.zeros(2))
    torch.testing.assert_close(projected[1], promotion[1])
