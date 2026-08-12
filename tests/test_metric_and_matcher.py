import torch
import pytest

from localization.matcher import (
    Top1Matches,
    global_cosine_top1,
    global_cosine_top2,
    suppress_duplicate_anchor_matches,
    suppress_duplicate_entity_matches,
)
from localization.localizer import load_shared_metric
from map_learning.metric import SharedLowRankMetric
from map_learning.trainer import (
    _update_group_dro_weights,
    full_refresh_interval,
    limit_training_records,
)
from common.config import load_mainline_config


def test_shared_metric_starts_as_identity_and_is_bounded():
    metric = SharedLowRankMetric(descriptor_dim=8, rank=2, max_residual_norm=0.05)
    descriptor = torch.nn.functional.normalize(torch.randn(5, 8), dim=1)
    transformed, residual = metric(descriptor)
    assert torch.allclose(transformed, descriptor)
    assert torch.equal(residual, torch.zeros_like(residual))


def test_global_matcher_has_no_landmark_cap():
    query = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    bank = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    matches = global_cosine_top1(query, bank)
    assert matches.anchor_indices.tolist() == [0, 0, 1]
    assert matches.keypoint_indices.tolist() == [0, 1, 2]


def test_global_top2_returns_exact_margin_candidates():
    query = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    bank = torch.tensor([[1.0, 0.0], [0.8, 0.2], [0.0, 1.0]])
    matches = global_cosine_top2(query, bank, chunk_size=2)
    assert matches.anchor_indices.tolist() == [[0, 1], [2, 1]]
    assert torch.all(matches.scores[:, 0] >= matches.scores[:, 1])


def test_duplicate_anchor_suppression_keeps_best_and_query_order():
    matches = Top1Matches(
        keypoint_indices=torch.tensor([2, 5, 9, 12, 20]),
        anchor_indices=torch.tensor([4, 7, 4, 8, 7]),
        scores=torch.tensor([0.8, 0.7, 0.9, 0.6, 0.65]),
    )
    retained = suppress_duplicate_anchor_matches(matches)
    assert retained.keypoint_indices.tolist() == [5, 9, 12]
    assert retained.anchor_indices.tolist() == [7, 4, 8]
    assert retained.scores.tolist() == pytest.approx([0.7, 0.9, 0.6])


def test_duplicate_anchor_suppression_breaks_score_ties_stably():
    matches = Top1Matches(
        keypoint_indices=torch.tensor([3, 6, 8]),
        anchor_indices=torch.tensor([2, 2, 5]),
        scores=torch.tensor([0.5, 0.5, 0.4]),
    )
    retained = suppress_duplicate_anchor_matches(matches)
    assert retained.keypoint_indices.tolist() == [3, 8]


def test_duplicate_entity_suppression_keeps_isolated_anchors_distinct():
    matches = Top1Matches(
        keypoint_indices=torch.tensor([0, 1, 2, 3, 4]),
        anchor_indices=torch.tensor([0, 1, 2, 2, 3]),
        scores=torch.tensor([0.8, 0.9, 0.7, 0.6, 0.5]),
    )
    retained = suppress_duplicate_entity_matches(
        matches, torch.tensor([4, 4, -1, -1])
    )
    assert retained.keypoint_indices.tolist() == [1, 2, 4]
    assert retained.anchor_indices.tolist() == [1, 2, 3]


def test_duplicate_entity_suppression_breaks_ties_stably():
    matches = Top1Matches(
        keypoint_indices=torch.tensor([5, 8]),
        anchor_indices=torch.tensor([0, 1]),
        scores=torch.tensor([0.5, 0.5]),
    )
    retained = suppress_duplicate_entity_matches(matches, torch.tensor([0, 0]))
    assert retained.keypoint_indices.tolist() == [5]
    assert retained.anchor_indices.tolist() == [0]


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


def test_deployment_row_limit_keeps_nested_detector_prefix():
    records = [
        {
            "deployment_rows": torch.tensor([2, 7, 12]),
            "cache_rows": torch.tensor([2, 7, 12]),
            "positives": torch.tensor([[0], [1], [2]]),
            "ignored_anchors": torch.tensor([[-1], [-1], [-1]]),
            "matchable": torch.tensor([True, False, True]),
            "group": 3,
        }
    ]
    limited, report = limit_training_records(records, 8)
    assert limited[0]["cache_rows"].tolist() == [2, 7]
    assert limited[0]["positives"].tolist() == [[0], [1]]
    assert limited[0]["group"] == 3
    assert report == {
        "deployment_row_limit": 8,
        "deployment_rows_before": 3,
        "deployment_rows_after": 2,
    }


def test_group_dro_update_cannot_collapse_to_one_trajectory_group():
    weights = torch.ones(8) / 8
    risk = torch.tensor([1000.0, 0, 0, 0, 0, 0, 0, 0])
    updated = _update_group_dro_weights(
        weights, risk, eta=0.03, maximum_uniform_ratio=3.0
    )
    assert updated.sum() == pytest.approx(1.0)
    assert float(updated.max()) <= 3.0 / 8.0 + 1e-6
    assert torch.all(updated > 0)


def test_paper_mainline_enables_capped_group_dro():
    config = load_mainline_config("configs/paper_mainline.yaml").values
    assert config["reconstruction"]["group_dro_max_weight_ratio"] == 3.0


def test_uncapped_group_dro_update_preserves_frozen_behavior():
    weights = torch.tensor([0.2, 0.3, 0.5])
    risk = torch.tensor([1.0, 4.0, 9.0])
    expected = weights * torch.exp(0.03 * risk)
    expected /= expected.sum()
    actual = _update_group_dro_weights(
        weights, risk, eta=0.03, maximum_uniform_ratio=1e9
    )
    assert torch.allclose(actual, expected)
