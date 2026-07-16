import torch
import torch.nn.functional as F

from localization_training.full_primitive_retrieval import (
    chunked_exact_topk,
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
