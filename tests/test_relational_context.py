import torch
import torch.nn.functional as F

from localization_training.relational_context import (
    AsymmetricBoundedDualContextEncoder,
    QueryAmbiguityGate,
    relational_map_3d_context,
    relational_sparse_query_context,
)


def test_query_context_preserves_neighbor_direction():
    descriptors = F.normalize(torch.eye(3, 4), dim=1)
    scores = torch.ones(3)
    left = torch.tensor([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]])
    right = torch.tensor([[0.0, 0.0], [-10.0, 0.0], [0.0, 10.0]])
    left_context = relational_sparse_query_context(
        descriptors, left, scores, neighbor_count=2
    )
    right_context = relational_sparse_query_context(
        descriptors, right, scores, neighbor_count=2
    )
    assert not torch.allclose(left_context[0], right_context[0])


def test_map_context_is_translation_invariant_and_uses_surface_frame():
    features = F.normalize(torch.eye(4), dim=1)
    xyz = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ]
    )
    tangent_x = torch.tensor([[1.0, 0.0, 0.0]]).repeat(4, 1)
    tangent_y = torch.tensor([[0.0, 1.0, 0.0]]).repeat(4, 1)
    normals = torch.tensor([[0.0, 0.0, 1.0]]).repeat(4, 1)
    kwargs = {
        "source_ids": torch.arange(4),
        "track_ids": torch.arange(4),
        "surface_scale": torch.ones(4),
        "neighbor_count": 2,
        "candidate_multiplier": 1,
    }
    context = relational_map_3d_context(
        features, xyz, tangent_x, tangent_y, normals, **kwargs
    )
    translated = relational_map_3d_context(
        features,
        xyz + torch.tensor([20.0, -3.0, 5.0]),
        tangent_x,
        tangent_y,
        normals,
        **kwargs,
    )
    swapped_frame = relational_map_3d_context(
        features, xyz, tangent_y, tangent_x, normals, **kwargs
    )
    assert torch.allclose(context, translated, atol=1e-6)
    assert not torch.allclose(context, swapped_frame)


def test_asymmetric_encoder_accepts_distinct_input_dimensions():
    model = AsymmetricBoundedDualContextEncoder(
        query_input_dim=12,
        map_input_dim=16,
        output_dim=8,
        rank=4,
    )
    query, query_residual = model.query(torch.randn(3, 12))
    map_value, map_residual = model.map(torch.randn(5, 16))
    assert query.shape == (3, 8)
    assert map_value.shape == (5, 8)
    assert query_residual.shape == query.shape
    assert map_residual.shape == map_value.shape


def test_query_ambiguity_gate_is_bounded_and_checks_alignment():
    gate = QueryAmbiguityGate(context_dim=8, hidden_dim=4)
    value = gate(torch.randn(3, 8), torch.rand(3))
    assert value.shape == (3,)
    assert bool(((value >= 0.0) & (value <= 1.0)).all())
    try:
        gate(torch.randn(3, 8), torch.rand(2))
    except ValueError as error:
        assert "must align" in str(error)
    else:
        raise AssertionError("misaligned gate inputs must be rejected")
