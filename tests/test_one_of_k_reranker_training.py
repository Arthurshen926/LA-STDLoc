import torch

from localization_training.local_assignment import OneOfKAssignmentHead
from scripts.train_one_of_k_reranker import (
    ambiguity_gated_positive_mask,
    assignment_error_breakdown,
    multi_positive_assignment_loss,
    normalized_landmark_statistics,
    summarize_assignment_counts,
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


def test_ambiguity_training_target_matches_runtime_gate():
    positives = torch.tensor([[False, True], [False, True], [True, True]])
    scores = torch.tensor([[0.8, 0.5], [0.51, 0.50], [0.9, 0.6]])
    gated = ambiguity_gated_positive_mask(positives, scores, 0.05)
    assert gated.tolist() == [[False, False], [False, True], [True, False]]


def test_assignment_breakdown_separates_beneficial_and_harmful_swaps():
    logits = torch.tensor([[0.0, 1.0], [0.0, 1.0], [1.0, 0.0]])
    null = torch.full((3,), -10.0)
    positives = torch.tensor([[False, True], [True, False], [True, False]])
    scores = torch.tensor([[0.51, 0.50], [0.51, 0.50], [0.51, 0.50]])
    counts = assignment_error_breakdown(
        logits, null, positives, scores, ambiguity_margin_threshold=0.05
    )
    summary = summarize_assignment_counts(counts)
    assert counts["beneficial_swaps"] == 1
    assert counts["harmful_swaps"] == 1
    assert summary["clean_top1_retention"] == 0.5
