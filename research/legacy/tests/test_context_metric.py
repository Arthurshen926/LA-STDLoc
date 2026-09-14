import torch

from map_learning.context_metric import (
    MapConsistentContextAdapter,
    dense_context_tokens,
)
from map_learning.context_metric_crossfit import prepare_training_records


def test_context_adapter_is_exact_identity_at_initialization():
    torch.manual_seed(3)
    adapter = MapConsistentContextAdapter(
        descriptor_dim=4,
        hidden_dim=8,
        context_kernels=(3, 7),
        maximum_residual_norm=0.1,
    )
    raw = torch.randn(5, 4)
    tokens = torch.randn(5, 3, 4)

    adapted, residual = adapter(raw, tokens)

    torch.testing.assert_close(residual, torch.zeros_like(residual))
    torch.testing.assert_close(adapted, torch.nn.functional.normalize(raw, dim=1))


def test_context_adapter_enforces_hard_residual_bound():
    torch.manual_seed(4)
    adapter = MapConsistentContextAdapter(
        descriptor_dim=4,
        hidden_dim=8,
        context_kernels=(3,),
        maximum_residual_norm=0.05,
    )
    with torch.no_grad():
        adapter.context_head[-1].weight.normal_()
        adapter.context_head[-1].bias.normal_()

    _, residual = adapter(torch.randn(7, 4), torch.randn(7, 2, 4))

    assert bool((residual.norm(dim=1) <= 0.050001).all())


def test_context_adapter_keeps_radial_trust_gradient_near_boundary():
    adapter = MapConsistentContextAdapter(
        descriptor_dim=4,
        hidden_dim=8,
        context_kernels=(3,),
        maximum_residual_norm=0.05,
    )
    with torch.no_grad():
        adapter.context_head[-1].bias.fill_(10.0)
    raw = torch.randn(7, 4)
    tokens = torch.randn(7, 2, 4)

    _, residual = adapter(raw, tokens)
    residual.square().sum(dim=1).mean().backward()

    gradient = adapter.context_head[-1].bias.grad
    assert gradient is not None
    assert bool(torch.isfinite(gradient).all())
    assert float(gradient.abs().sum()) > 0.0


def test_legacy_hard_clip_remains_available_for_old_artifacts():
    adapter = MapConsistentContextAdapter(
        descriptor_dim=4,
        hidden_dim=8,
        context_kernels=(3,),
        maximum_residual_norm=0.05,
        residual_parameterization="hard_clip_v1",
    )
    with torch.no_grad():
        adapter.context_head[-1].bias.fill_(10.0)

    _, residual = adapter(torch.randn(3, 4), torch.randn(3, 2, 4))

    torch.testing.assert_close(
        residual.norm(dim=1),
        torch.full((3,), 0.05),
        atol=1e-6,
        rtol=0.0,
    )


def test_dense_context_tokens_preserve_constant_descriptor_direction():
    direction = torch.tensor([1.0, 2.0, -1.0])
    feature_map = direction[:, None, None].expand(3, 4, 5).clone()
    keypoints = torch.tensor([[0.0, 0.0], [19.0, 15.0]])

    tokens = dense_context_tokens(
        feature_map,
        keypoints,
        (32, 40),
        valid_mask=torch.ones(4, 5, dtype=torch.bool),
        kernels=(3, 7, 15),
    )

    expected = torch.nn.functional.normalize(direction, dim=0)
    assert tokens.shape == (2, 4, 3)
    torch.testing.assert_close(tokens, expected.expand_as(tokens))


def test_training_records_drop_unsupported_positive_anchors():
    teacher = {
        "anchor_count": 4,
        "records": [
            {
                "query_rows": torch.tensor([2, 5]),
                "positive_offsets": torch.tensor([0, 2, 3]),
                "positive_indices": torch.tensor([0, 3, 2]),
                "ambiguous_offsets": torch.tensor([0, 1, 1]),
                "ambiguous_indices": torch.tensor([3]),
            }
        ],
    }

    records, report = prepare_training_records(
        teacher=teacher,
        support_query_indices=[0],
        anchor_indices=torch.tensor([0, 2]),
        maximum_positives=2,
        maximum_ignored=2,
    )

    assert records[0]["positives"].tolist() == [[0, -1], [1, -1]]
    assert records[0]["ignored"].tolist() == [[-1, -1], [-1, -1]]
    assert records[0]["matchable_rows"].tolist() == [0, 1]
    assert report["training_positive_pair_count"] == 2


def test_zero_context_mode_is_a_pointwise_control():
    from map_learning.context_metric import context_from_cached_query

    cached = {
        "native_descriptors": torch.randn(3, 4),
        "native_keypoints": torch.randn(3, 2),
        "native_input_hw": [16, 16],
    }

    raw, tokens = context_from_cached_query(
        cached,
        torch.tensor([0, 2]),
        device=torch.device("cpu"),
        kernels=(3, 7),
        context_mode="zero",
    )

    assert raw.shape == (2, 4)
    assert tokens.shape == (2, 3, 4)
    assert not bool(tokens.any())
