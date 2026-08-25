import torch

from localization.matcher import (
    global_cosine_top1,
    global_owner_prototype_top1,
    global_view_mixture_topk,
)


def test_sparse_owner_prototype_changes_appearance_not_anchor_identity() -> None:
    query = torch.tensor([[1.0, 0.0]])
    base = torch.tensor([[0.9, 0.1], [0.0, 1.0]])
    result = global_owner_prototype_top1(
        query,
        base,
        torch.tensor([[1.0, 0.0]]),
        torch.tensor([1]),
        anchor_descriptors_normalized=False,
    )
    assert result.anchor_indices.tolist() == [1]
    assert torch.allclose(result.scores, torch.ones(1))


def test_empty_sparse_owner_prototype_is_exact_baseline() -> None:
    query = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    base = torch.tensor([[0.9, 0.1], [0.0, 1.0]])
    expected = global_cosine_top1(query, base)
    actual = global_owner_prototype_top1(
        query, base, torch.empty((0, 2)), torch.empty(0, dtype=torch.long)
    )
    assert torch.equal(actual.anchor_indices, expected.anchor_indices)
    assert torch.equal(actual.scores, expected.scores)


def test_single_prototype_rows_are_exact_cosine_compatibility():
    query = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    anchors = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]])
    prototypes = torch.zeros(3, 2, 2)
    prototypes[:, 0] = anchors
    priors = torch.tensor([[1.0, 0.0]] * 3)
    baseline = global_cosine_top1(query, anchors, chunk_size=8)
    mixture = global_view_mixture_topk(query, prototypes, priors, topk=1)
    assert torch.equal(mixture.anchor_indices[:, 0], baseline.anchor_indices)
    assert torch.equal(mixture.scores[:, 0], baseline.scores)


def test_all_k1_evaluator_path_is_bitwise_exact_after_single_normalization():
    generator = torch.Generator().manual_seed(2026)
    query = torch.nn.functional.normalize(
        torch.nn.functional.normalize(
            torch.randn(100, 32, generator=generator), dim=1
        ),
        dim=1,
    )
    bank = torch.nn.functional.normalize(
        torch.randn(1000, 32, generator=generator), dim=1
    )
    prototypes = torch.zeros(1000, 2, 32)
    prototypes[:, 0] = bank
    priors = torch.tensor([[1.0, 0.0]] * 1000)
    expected_scores, expected_indices = torch.topk(query @ bank.T, k=1, dim=1)
    actual = global_view_mixture_topk(query, prototypes, priors, topk=1)
    assert torch.equal(actual.anchor_indices, expected_indices)
    assert torch.equal(actual.scores, expected_scores)


def test_two_prototypes_still_emit_one_anchor_identity():
    query = torch.tensor([[0.0, 1.0]])
    prototypes = torch.tensor([[[1.0, 0.0], [0.0, 1.0]], [[0.8, 0.2], [0.0, 0.0]]])
    priors = torch.tensor([[0.5, 0.5], [1.0, 0.0]])
    result = global_view_mixture_topk(query, prototypes, priors, topk=1, temperature=0.05)
    assert result.anchor_indices.shape == (1, 1)
    assert result.anchor_indices.item() == 0
