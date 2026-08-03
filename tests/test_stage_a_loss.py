import torch

from map_learning.stage_a_loss import (
    DetectorFreeObservationBatch,
    descriptor_trust_loss,
    hard_hypothesis_retrieval_loss,
    materialize_descriptor_residual,
)


def _batch():
    return DetectorFreeObservationBatch(
        source_indices=torch.tensor([0, 1, 0, -1]),
        query_features=torch.nn.functional.normalize(
            torch.tensor([
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.8, 1.0, 0.0],
                [0.4, 0.0, 0.6, 1.0],
                [0.0, 0.0, 1.0, 0.0],
            ]), dim=-1,
        ),
        query_uv=torch.tensor([[10., 10.], [20., 10.], [10., 10.], [100., 100.]]),
        source_depth=torch.ones(4),
        bank_uv=torch.tensor([[10., 10.], [20., 10.], [30., 30.], [40., 40.]]),
        bank_depth=torch.ones(4),
        bank_projected=torch.ones(4, dtype=torch.bool),
        bank_visible=torch.ones(4, dtype=torch.bool),
    )


def test_keep_swap_miss_reject_are_disjoint_and_differentiable():
    features = torch.eye(4, requires_grad=True)
    output = hard_hypothesis_retrieval_loss(
        features, _batch(), hypothesis_topk=2, native_outcome_mode=True,
        native_nce_weight=0, native_keep_margin=1.1, native_reject_weight=1,
    )
    diagnostics = output.diagnostics
    assert [diagnostics[f"retrieval_native_{name}_count"] for name in
            ("keep", "swap", "miss", "reject")] == [1, 1, 1, 1]
    output.loss.backward()
    assert torch.isfinite(features.grad).all()


def test_descriptor_residual_and_trust_are_bounded():
    initial = torch.tensor([[1.0, 0.0]])
    result = materialize_descriptor_residual(
        initial, torch.tensor([[0.0, 10.0]]), max_residual_norm=0.1
    )
    assert torch.allclose(result.norm(dim=1), torch.ones(1))
    assert result[0, 1] < 0.11
    assert descriptor_trust_loss(initial, initial).item() == 0
