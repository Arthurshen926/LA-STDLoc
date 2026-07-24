import torch

from localization_training.local_assignment import OneOfKAssignmentHead
from scripts.train_one_of_k_reranker import (
    multi_positive_assignment_loss,
    normalized_landmark_statistics,
)


def test_multi_positive_assignment_accepts_any_positive_and_null():
    candidate = torch.tensor(
        [[0.0, 3.0, 2.0], [-1.0, -2.0, -3.0]], requires_grad=True
    )
    null = torch.tensor([-2.0, 2.0], requires_grad=True)
    positive = torch.tensor(
        [[False, True, True], [False, False, False]]
    )
    loss, diagnostics = multi_positive_assignment_loss(
        candidate, null, positive
    )
    assert loss < 0.3
    assert diagnostics["positive_rows"] == 1
    assert diagnostics["null_rows"] == 1
    loss.backward()
    assert candidate.grad is not None


def test_multi_positive_assignment_protects_clean_top1():
    candidate = torch.tensor([[0.0, 4.0]], requires_grad=True)
    null = torch.tensor([-4.0], requires_grad=True)
    positive = torch.tensor([[True, True]])
    protected, diagnostics = multi_positive_assignment_loss(
        candidate,
        null,
        positive,
    )
    unprotected, _ = multi_positive_assignment_loss(
        candidate,
        null,
        positive,
        protect_clean_top1=False,
    )
    assert protected > unprotected
    assert diagnostics["protected_top1_rows"] == 1


def test_global_skip_preserves_cosine_order_at_initialization():
    head = OneOfKAssignmentHead(
        hidden_dim=4,
        feature_dim=5,
        global_skip_scale=10.0,
    )
    for parameter in head.candidate.parameters():
        torch.nn.init.zeros_(parameter)
    features = torch.zeros(1, 2, 5)
    features[0, :, 0] = torch.tensor([0.8, 0.6])
    logits, _ = head(features)
    assert logits.argmax(dim=1).item() == 0
    assert torch.allclose(logits[0], torch.tensor([8.0, 6.0]))


def test_landmark_statistics_are_id_aligned_and_bounded(tmp_path):
    indices = torch.tensor([7, 11])
    statistics = {
        "matchability": torch.tensor([0.2, 0.8]),
        "false_top1_rate": torch.tensor([0.7, 0.1]),
        "cross_view_top1_harmful_switch_rate": torch.tensor([0.5, 0.0]),
        "rescue_utility": torch.tensor([0.0, 10.0]),
        "effective_observation_count": torch.tensor([1.0, 100.0]),
    }
    path = tmp_path / "statistics.pt"
    torch.save(
        {"landmark_indices": indices, "statistics": statistics},
        path,
    )
    features, names = normalized_landmark_statistics(path, indices)
    assert features.shape == (2, 5)
    assert len(names) == 5
    assert bool(((features >= 0.0) & (features <= 1.0)).all())
