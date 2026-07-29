import pytest
import torch
import torch.nn.functional as F

from localization_training.full_primitive_retrieval import (
    chunked_exact_topk,
    chunked_exact_topk_dual_prototype,
    chunked_exact_topk_family_prototype,
    conditional_core_reserve_topk,
    suppress_redundant_hypotheses,
)


def test_chunked_exact_topk_matches_dense_cosine():
    torch.manual_seed(4)
    query = torch.randn(7, 5)
    features = torch.randn(23, 5)
    expected_scores, expected_indices = torch.topk(
        F.normalize(query, dim=1) @ F.normalize(features, dim=1).T, 4, dim=1
    )
    result = chunked_exact_topk(query, features, topk=4, chunk_size=6)
    assert torch.equal(result.indices, expected_indices)
    assert torch.allclose(result.scores, expected_scores, atol=1e-6)
    assert result.chunks == 4


def test_dual_prototype_retrieval_uses_max_score_per_anchor():
    query = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    primary = torch.tensor([[1.0, 0.0], [-1.0, 0.0]])
    secondary = torch.tensor([[0.0, 1.0], [0.0, -1.0]])
    result = chunked_exact_topk_dual_prototype(
        query,
        primary,
        secondary,
        torch.tensor([True, False]),
        topk=2,
        chunk_size=1,
    )
    assert result.indices[:, 0].tolist() == [0, 0]
    assert torch.allclose(result.scores[:, 0], torch.ones(2))


def test_family_prototype_retrieval_returns_unique_geometry_anchors():
    query = torch.tensor([[0.0, 1.0], [1.0, 1.0]])
    primary = torch.tensor([[1.0, 0.0], [-1.0, 0.0], [0.7, 0.7]])
    prototypes = torch.tensor(
        [[0.0, 1.0], [0.1, 1.0], [-0.7, 0.7]]
    )
    parents = torch.tensor([0, 0, 1])
    result = chunked_exact_topk_family_prototype(
        query,
        primary,
        prototypes,
        parents,
        topk=3,
        chunk_size=2,
    )
    assert result.indices[0, 0].item() == 0
    assert set(result.indices[0].tolist()) == {0, 1, 2}
    assert result.indices[1].unique().numel() == 3
    assert torch.allclose(result.scores[0, 0], torch.tensor(1.0), atol=1e-6)


def test_family_prototype_bias_prevents_weak_secondary_activation():
    query = torch.tensor([[0.0, 1.0]])
    primary = torch.tensor([[1.0, 0.0], [0.2, 0.98]])
    prototypes = torch.tensor([[0.0, 1.0]])
    parents = torch.tensor([0])
    uncalibrated = chunked_exact_topk_family_prototype(
        query, primary, prototypes, parents, topk=1
    )
    calibrated = chunked_exact_topk_family_prototype(
        query,
        primary,
        prototypes,
        parents,
        prototype_bias=torch.tensor([-0.05]),
        prototype_temperature=torch.ones(1),
        topk=1,
    )
    assert uncalibrated.indices.item() == 0
    assert calibrated.indices.item() == 1


def test_family_prototype_temperature_must_be_positive():
    with pytest.raises(ValueError, match="temperature"):
        chunked_exact_topk_family_prototype(
            torch.tensor([[0.0, 1.0]]),
            torch.tensor([[1.0, 0.0]]),
            torch.tensor([[0.0, 1.0]]),
            torch.tensor([0]),
            prototype_temperature=torch.tensor([0.0]),
        )


def test_family_prototype_bias_must_be_non_positive():
    with pytest.raises(ValueError, match="non-positive"):
        chunked_exact_topk_family_prototype(
            torch.tensor([[0.0, 1.0]]),
            torch.tensor([[1.0, 0.0]]),
            torch.tensor([[0.0, 1.0]]),
            torch.tensor([0]),
            prototype_bias=torch.tensor([0.01]),
        )


def test_conditional_reserve_only_changes_ambiguous_core_rows():
    query = torch.tensor([[1.0, 0.0], [0.7, 0.7]])
    features = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [0.7, 0.7], [-1.0, 0.0]]
    )
    result, ambiguous, margin = conditional_core_reserve_topk(
        query,
        features,
        torch.tensor([True, True, False, False]),
        margin_threshold=0.1,
        topk=2,
        chunk_size=2,
    )
    assert ambiguous.tolist() == [False, True]
    assert result.indices[0].tolist() == [0, 1]
    assert result.indices[1, 0].item() == 2
    assert margin[0] > margin[1]


def test_surface_suppression_keeps_distinct_voxels_and_sources():
    scores = torch.tensor([[0.9, 0.8, 0.7, 0.6]])
    indices = torch.tensor([[0, 1, 2, 3]])
    xyz = torch.tensor(
        [[0.01, 0.0, 0.0], [0.02, 0.0, 0.0], [0.2, 0.0, 0.0], [0.4, 0.0, 0.0]]
    )
    source = torch.tensor([10, 11, 10, 12])
    kept_scores, kept_indices, ratio = suppress_redundant_hypotheses(
        scores,
        indices,
        xyz,
        output_topk=2,
        voxel_size=0.1,
        source_indices=source,
    )
    assert kept_indices.tolist() == [[0, 3]]
    assert torch.allclose(kept_scores, torch.tensor([[0.9, 0.6]]))
    assert ratio == 0.5
