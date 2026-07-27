from pathlib import Path

import torch

from localization_training.shared_metric import SharedLowRankMetric
from scripts.build_lafgs_v7_track_centric_maps import (
    _normalized_log_score,
    _voxel_diverse_order,
)
from scripts.train_lafgs_v7_online_metric import (
    _bounded_anchor_features,
    _multi_positive_list_loss,
)


def test_shared_metric_starts_as_identity_and_is_bounded():
    metric = SharedLowRankMetric(
        descriptor_dim=4, rank=2, max_residual_norm=0.05
    )
    value = torch.randn(5, 4)
    output, residual = metric(value)
    torch.testing.assert_close(output, torch.nn.functional.normalize(value))
    assert float(residual.norm(dim=1).max()) == 0.0
    with torch.no_grad():
        metric.up.weight.fill_(10.0)
    _, residual = metric(value)
    assert float(residual.norm(dim=1).max()) <= 0.050001


def test_voxel_order_takes_spatial_representatives_before_duplicates():
    xyz = torch.tensor(
        [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [2.0, 0.0, 0.0]]
    )
    order = _voxel_diverse_order(
        xyz, torch.tensor([4.0, 3.0, 2.0]), voxel_size=1.0
    )
    assert order.tolist() == [0, 2, 1]
    assert torch.isfinite(_normalized_log_score(torch.zeros(3))).all()


def test_listwise_loss_rewards_any_positive_and_penalizes_harmful_mass():
    query = torch.tensor([[1.0, 0.0]])
    bank = torch.tensor([[1.0, 0.0], [0.8, 0.6], [-1.0, 0.0]])
    positives = torch.tensor([[0, -1]])
    clean, _, _ = _multi_positive_list_loss(
        query,
        bank,
        positives,
        topk=3,
        temperature=0.1,
        harmful_prior=None,
        harmful_weight=1.0,
        harmful_indices=torch.full((1, 1), -1),
    )
    harmful, _, _ = _multi_positive_list_loss(
        query,
        bank,
        positives,
        topk=3,
        temperature=0.1,
        harmful_prior=None,
        harmful_weight=1.0,
        harmful_indices=torch.tensor([[1]]),
    )
    assert float(harmful) > float(clean)


def test_anchor_residual_is_bounded():
    raw = torch.nn.functional.normalize(torch.randn(4, 8), dim=1)
    adapted, residual = _bounded_anchor_features(
        raw, torch.full_like(raw, 10.0), maximum=0.02
    )
    assert float(residual.norm(dim=1).max()) <= 0.020001
    torch.testing.assert_close(adapted.norm(dim=1), torch.ones(4))


def test_metric_frontend_uses_native_sparse_dispatch():
    source = (Path(__file__).parents[1] / "stdloc.py").read_text()
    native_dispatch = source.split(
        "sparse_result = self.loc_sparse_ulfloc_native", 1
    )[0].rsplit("def localize(", 1)[-1]
    assert '"ulfloc_native_metric"' in native_dispatch
