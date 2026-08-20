import itertools

import torch
import pytest

from localization.matcher import (
    TopKMatches,
    Top1Matches,
    global_cosine_top1,
    global_cosine_top2,
    global_cosine_topk,
    maximum_weight_anchor_assignment,
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


def _legacy_global_match(query, bank, *, topk, chunk_size):
    """Pre-acceleration kernel retained as a strict non-tie test oracle."""
    query = torch.nn.functional.normalize(query.float(), dim=1)
    scores = query.new_full((query.shape[0], topk), -torch.inf)
    indices = torch.zeros(
        (query.shape[0], topk), dtype=torch.long, device=query.device
    )
    for start in range(0, bank.shape[0], chunk_size):
        stop = min(start + chunk_size, bank.shape[0])
        chunk = torch.nn.functional.normalize(bank[start:stop].float(), dim=1)
        chunk_scores = query @ chunk.T
        chunk_indices = torch.arange(start, stop, device=query.device)[None].expand(
            query.shape[0], -1
        )
        merged_scores = torch.cat((scores, chunk_scores), dim=1)
        merged_indices = torch.cat((indices, chunk_indices), dim=1)
        scores, positions = torch.topk(merged_scores, topk, dim=1)
        indices = torch.gather(merged_indices, 1, positions)
    return scores, indices


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


@pytest.mark.parametrize("chunk_size", [1, 2, 7, 31])
def test_exact_top1_acceleration_matches_legacy_non_tie_oracle(chunk_size):
    generator = torch.Generator().manual_seed(20260820)
    query = torch.randn(19, 16, generator=generator)
    bank = torch.randn(67, 16, generator=generator)
    expected_scores, expected_indices = _legacy_global_match(
        query, bank, topk=1, chunk_size=chunk_size
    )
    actual = global_cosine_top1(query, bank, chunk_size=chunk_size)
    assert torch.equal(actual.anchor_indices[:, None], expected_indices)
    assert torch.equal(actual.scores[:, None], expected_scores)


@pytest.mark.parametrize("chunk_size", [1, 2, 7, 31])
def test_exact_top2_acceleration_matches_legacy_non_tie_oracle(chunk_size):
    generator = torch.Generator().manual_seed(20260820)
    query = torch.randn(19, 16, generator=generator)
    bank = torch.randn(67, 16, generator=generator)
    expected_scores, expected_indices = _legacy_global_match(
        query, bank, topk=2, chunk_size=chunk_size
    )
    actual = global_cosine_top2(query, bank, chunk_size=chunk_size)
    assert torch.equal(actual.anchor_indices, expected_indices)
    assert torch.equal(actual.scores, expected_scores)


@pytest.mark.parametrize("chunk_size", [1, 2, 3, 8])
def test_exact_matcher_ties_are_chunk_independent_and_anchor_stable(chunk_size):
    query = torch.tensor([[1.0, 0.0], [0.0, 0.0], [-0.0, 1.0]])
    bank = torch.tensor(
        [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [-1.0, 0.0], [0.0, 1.0]]
    )
    top1 = global_cosine_top1(query, bank, chunk_size=chunk_size)
    top2 = global_cosine_top2(query, bank, chunk_size=chunk_size)
    assert top1.anchor_indices.tolist() == [0, 0, 5]
    assert top2.anchor_indices.tolist() == [[0, 1], [0, 1], [5, 0]]


def test_pre_normalized_bank_reuses_exact_historical_matching_values():
    generator = torch.Generator().manual_seed(7)
    query = torch.randn(13, 32, generator=generator)
    raw_bank = torch.randn(71, 32, generator=generator)
    cached_bank = torch.nn.functional.normalize(raw_bank.float(), dim=1)
    expected = global_cosine_top1(query, raw_bank, chunk_size=11)
    actual = global_cosine_top1(
        query,
        cached_bank,
        chunk_size=11,
        anchor_descriptors_normalized=True,
    )
    assert torch.equal(actual.anchor_indices, expected.anchor_indices)
    assert torch.equal(actual.scores, expected.scores)


def test_global_topk_is_exact_across_chunks():
    query = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    bank = torch.tensor([[1.0, 0.0], [0.8, 0.2], [0.4, 0.6], [0.0, 1.0], [-1.0, 0.0]])
    matches = global_cosine_topk(query, bank, topk=3, chunk_size=2)
    dense = (
        torch.nn.functional.normalize(query, dim=1)
        @ torch.nn.functional.normalize(bank, dim=1).T
    )
    expected_scores, expected_indices = torch.topk(dense, 3, dim=1)
    assert torch.equal(matches.anchor_indices, expected_indices)
    assert torch.allclose(matches.scores, expected_scores)


def test_capacity_assignment_can_fall_back_to_second_candidate():
    candidates = TopKMatches(
        keypoint_indices=torch.tensor([4, 7]),
        anchor_indices=torch.tensor([[0, 1], [0, 1]]),
        scores=torch.tensor([[0.90, 0.80], [0.85, 0.10]]),
    )
    result = maximum_weight_anchor_assignment(candidates, dustbin_score=0.0)
    assert result.matches.keypoint_indices.tolist() == [4, 7]
    assert result.matches.anchor_indices.tolist() == [1, 0]
    assert result.matches.scores.tolist() == pytest.approx([0.8, 0.85])
    assert result.reassigned_query_count == 1
    assert result.top1_collision_count == 1
    assert result.unmatched_query_count == 0


def test_capacity_assignment_uses_private_dustbin_and_strict_threshold():
    candidates = TopKMatches(
        keypoint_indices=torch.tensor([0, 1, 2]),
        anchor_indices=torch.tensor([[0, 1], [0, 2], [3, 4]]),
        scores=torch.tensor([[0.8, 0.7], [0.6, 0.5], [0.2, 0.1]]),
    )
    result = maximum_weight_anchor_assignment(candidates, dustbin_score=0.6)
    assert result.matches.keypoint_indices.tolist() == [0]
    assert result.matches.anchor_indices.tolist() == [0]
    assert result.unmatched_query_count == 2
    assert result.eligible_edge_count == 2


def test_capacity_assignment_rejects_duplicate_candidate_edges():
    candidates = TopKMatches(
        keypoint_indices=torch.tensor([0]),
        anchor_indices=torch.tensor([[2, 2]]),
        scores=torch.tensor([[0.8, 0.7]]),
    )
    with pytest.raises(ValueError, match="unique per row"):
        maximum_weight_anchor_assignment(candidates, dustbin_score=0.0)


def test_capacity_assignment_matches_small_graph_exhaustive_oracle():
    generator = torch.Generator().manual_seed(20260819)
    for query_count in range(1, 5):
        for _ in range(20):
            anchor_count = 5
            topk = 3
            anchors = torch.stack(
                [
                    torch.randperm(anchor_count, generator=generator)[:topk]
                    for _ in range(query_count)
                ]
            )
            scores = torch.sort(
                torch.rand((query_count, topk), generator=generator) * 1.4 - 0.4,
                dim=1,
                descending=True,
            ).values
            dustbin = 0.1
            candidates = TopKMatches(
                keypoint_indices=torch.arange(query_count),
                anchor_indices=anchors,
                scores=scores,
            )
            result = maximum_weight_anchor_assignment(candidates, dustbin_score=dustbin)
            returned = {
                int(row): (int(anchor), float(score))
                for row, anchor, score in zip(
                    result.matches.keypoint_indices,
                    result.matches.anchor_indices,
                    result.matches.scores,
                )
            }
            actual = sum(
                returned.get(row, (-1, dustbin))[1] for row in range(query_count)
            )
            best = -float("inf")
            choices = [
                [-1]
                + [rank for rank in range(topk) if float(scores[row, rank]) > dustbin]
                for row in range(query_count)
            ]
            for ranks in itertools.product(*choices):
                selected = [
                    int(anchors[row, rank])
                    for row, rank in enumerate(ranks)
                    if rank >= 0
                ]
                if len(selected) != len(set(selected)):
                    continue
                objective = sum(
                    dustbin if rank < 0 else float(scores[row, rank])
                    for row, rank in enumerate(ranks)
                )
                best = max(best, objective)
            assert actual == pytest.approx(best, abs=1e-6)


def test_capacity_assignment_rejects_unsorted_candidate_scores():
    candidates = TopKMatches(
        keypoint_indices=torch.tensor([0]),
        anchor_indices=torch.tensor([[2, 3]]),
        scores=torch.tensor([[0.7, 0.8]]),
    )
    with pytest.raises(ValueError, match="rank-sorted"):
        maximum_weight_anchor_assignment(candidates, dustbin_score=0.0)


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
    retained = suppress_duplicate_entity_matches(matches, torch.tensor([4, 4, -1, -1]))
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
