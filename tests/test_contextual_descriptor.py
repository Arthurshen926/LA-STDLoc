import torch
import torch.nn.functional as F

from localization_training.contextual_descriptor import (
    BoundedDualContextEncoder,
    flatten_context,
    fuse_local_and_context,
    joint_local_context_similarity,
    multiscale_dense_query_context,
    multiscale_map_3d_context,
    multiscale_sparse_query_context,
)


def test_sparse_context_excludes_center_descriptor():
    descriptors = torch.eye(4)
    keypoints = torch.tensor(
        [[0.0, 0.0], [2.0, 0.0], [20.0, 0.0], [40.0, 0.0]]
    )
    scores = torch.ones(4)
    context = multiscale_sparse_query_context(
        descriptors,
        keypoints,
        scores,
        radii_px=(3.0,),
        maximum_neighbors=3,
        chunk_size=2,
    )
    assert context.shape == (4, 1, 4)
    assert context[0, 0, 0] == 0
    assert context[0, 0, 1] > 0.99
    assert torch.equal(context[2, 0], torch.zeros(4))


def test_dense_context_keeps_native_pixel_center_contract():
    feature_map = torch.zeros(1, 2, 2, 2)
    feature_map[:, 0] = 1.0
    keypoints = torch.tensor([[3.5, 3.5], [11.5, 11.5]])
    context = multiscale_dense_query_context(
        feature_map, keypoints, radii_cells=(0, 1)
    )
    assert context.shape == (2, 2, 2)
    assert torch.allclose(context[..., 0], torch.ones(2, 2))


def test_3d_context_is_translation_invariant_and_excludes_center():
    features = F.normalize(torch.eye(4), dim=1)
    xyz = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [3.0, 0.0, 0.0], [8.0, 0.0, 0.0]]
    )
    first = multiscale_map_3d_context(
        features, xyz, neighbor_counts=(1, 2), chunk_size=2
    )
    second = multiscale_map_3d_context(
        features, xyz + 100.0, neighbor_counts=(1, 2), chunk_size=2
    )
    assert torch.allclose(first, second, atol=1e-6)
    assert first[0, 0, 0] == 0
    assert first[0, 0, 1] > 0.99


def test_fused_descriptor_is_fixed_size_and_normalized():
    local = F.normalize(torch.randn(5, 4), dim=1)
    context = flatten_context(torch.randn(5, 3, 4))
    fused = fuse_local_and_context(local, context, context_weight=0.2)
    assert fused.shape == (5, 16)
    assert torch.allclose(fused.norm(dim=1), torch.ones(5), atol=1e-6)


def test_joint_similarity_matches_materialized_fusion():
    generator = torch.Generator().manual_seed(13)
    query_local = torch.randn(5, 8, generator=generator)
    map_local = torch.randn(7, 8, generator=generator)
    query_context = torch.randn(5, 6, generator=generator)
    map_context = torch.randn(7, 6, generator=generator)
    weight = 0.17
    expected = fuse_local_and_context(
        query_local, query_context, context_weight=weight
    ) @ fuse_local_and_context(
        map_local, map_context, context_weight=weight
    ).T
    actual = joint_local_context_similarity(
        query_local,
        map_local,
        query_context,
        map_context,
        context_weight=weight,
    )
    assert torch.allclose(actual, expected, atol=1e-6)


def test_dual_context_encoder_starts_shared_and_bounds_residual():
    model = BoundedDualContextEncoder(
        input_dim=12,
        output_dim=4,
        rank=3,
        maximum_residual=0.1,
    )
    value = torch.randn(7, 12)
    query, query_residual = model.query(value)
    mapped, map_residual = model.map(value)
    assert torch.allclose(query, mapped)
    assert torch.equal(query_residual, torch.zeros_like(query_residual))
    with torch.no_grad():
        model.query.up.weight.fill_(100.0)
    _, query_residual = model.query(value)
    assert bool((query_residual.norm(dim=1) <= 0.100001).all())
    assert torch.equal(map_residual, torch.zeros_like(map_residual))
