import torch

from localization_training.local_context_adapter import (
    LocalContextMetricAdapter,
    pool_local_query_context,
)


def test_local_context_pool_preserves_row_alignment():
    feature_map = torch.zeros(2, 4, 4)
    feature_map[0] = 1.0
    context = pool_local_query_context(
        feature_map,
        torch.tensor([[0.0, 0.0], [2.0, 2.0]]),
        (4, 4),
        radius=1,
        step_px=1.0,
    )
    assert context.shape == (2, 2)
    assert torch.isfinite(context).all()


def test_adapter_initializes_as_identity_and_bounds_residual():
    adapter = LocalContextMetricAdapter(
        descriptor_dim=4, rank=2, max_residual_norm=0.05
    )
    sparse = torch.nn.functional.normalize(torch.randn(3, 4), dim=1)
    context = torch.nn.functional.normalize(torch.randn(3, 4), dim=1)
    adapted, residual = adapter(sparse, context)
    assert torch.allclose(adapted, sparse)
    with torch.no_grad():
        adapter.up.weight.fill_(10.0)
    _, residual = adapter(sparse, context)
    assert bool((torch.linalg.norm(residual, dim=1) <= 0.050001).all())
