import torch

from localization.matcher import global_cosine_top1, global_view_mixture_topk


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


def test_two_prototypes_still_emit_one_anchor_identity():
    query = torch.tensor([[0.0, 1.0]])
    prototypes = torch.tensor([[[1.0, 0.0], [0.0, 1.0]], [[0.8, 0.2], [0.0, 0.0]]])
    priors = torch.tensor([[0.5, 0.5], [1.0, 0.0]])
    result = global_view_mixture_topk(query, prototypes, priors, topk=1, temperature=0.05)
    assert result.anchor_indices.shape == (1, 1)
    assert result.anchor_indices.item() == 0
