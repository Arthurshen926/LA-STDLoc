import torch
import pytest

from localization.matcher import global_cosine_top1
from localization.localizer import load_shared_metric
from map_learning.metric import SharedLowRankMetric
from map_learning.trainer import _update_group_dro_weights, full_refresh_interval


def test_shared_metric_starts_as_identity_and_is_bounded():
    metric = SharedLowRankMetric(descriptor_dim=8, rank=2, max_residual_norm=0.05)
    descriptor = torch.nn.functional.normalize(torch.randn(5, 8), dim=1)
    transformed, residual = metric(descriptor)
    assert torch.allclose(transformed, descriptor)
    assert torch.equal(residual, torch.zeros_like(residual))


def test_global_matcher_has_no_landmark_cap():
    query = torch.tensor([[1., 0.], [1., 0.], [0., 1.]])
    bank = torch.tensor([[1., 0.], [0., 1.]])
    matches = global_cosine_top1(query, bank)
    assert matches.anchor_indices.tolist() == [0, 0, 1]
    assert matches.keypoint_indices.tolist() == [0, 1, 2]


def test_metric_state_must_match_anchor_registry(tmp_path):
    metric = SharedLowRankMetric(descriptor_dim=8, rank=2)
    state = {
        "landmark_indices": torch.tensor([10, 11]),
        "metric_config": metric.export_config(),
        "metric_state_dict": metric.state_dict(),
    }
    path = tmp_path / "metric.pt"
    torch.save(state, path)
    with pytest.raises(ValueError, match="does not align"):
        load_shared_metric(
            path,
            anchor_ids=torch.tensor([11, 10]),
            device=torch.device("cpu"),
        )


def test_full_refresh_interval_covers_every_rotating_shard():
    interval = full_refresh_interval(176, 7)
    refreshes = 1 + (176 - 1) // interval
    assert refreshes >= 7


def test_group_dro_update_cannot_collapse_to_one_trajectory_group():
    weights = torch.ones(8) / 8
    risk = torch.tensor([1000.0, 0, 0, 0, 0, 0, 0, 0])
    updated = _update_group_dro_weights(
        weights, risk, eta=0.03, maximum_uniform_ratio=3.0
    )
    assert updated.sum() == pytest.approx(1.0)
    assert float(updated.max()) <= 3.0 / 8.0 + 1e-6
    assert torch.all(updated > 0)


def test_uncapped_group_dro_update_preserves_frozen_behavior():
    weights = torch.tensor([0.2, 0.3, 0.5])
    risk = torch.tensor([1.0, 4.0, 9.0])
    expected = weights * torch.exp(0.03 * risk)
    expected /= expected.sum()
    actual = _update_group_dro_weights(
        weights, risk, eta=0.03, maximum_uniform_ratio=1e9
    )
    assert torch.allclose(actual, expected)
