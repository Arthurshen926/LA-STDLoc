from __future__ import annotations

import torch
import numpy as np
import pytest

from localization import pose_solver
from localization.matcher import global_cosine_top1
from localization.localizer import _query_level_feedback_count
from map_learning.v24_anchor_view_support import build_anchor_view_support
from map_learning.v24_pose_conditioned_sparse_refinement import (
    build_pose_visible_topk,
    changed_inlier_spatial_cell_count,
    compare_poses_on_heldout_candidate_graph,
    compare_poses_on_common_candidate_grid,
    default_config,
    runtime_config,
    select_pose_conditioned_rows,
    spatial_jackknife_pose_stability,
)


def _topk(
    rows: list[list[int]], scores: list[list[float]]
) -> tuple[torch.Tensor, torch.Tensor]:
    padded_rows = [row + [row[0]] * (64 - len(row)) for row in rows]
    padded_scores = [score + [score[0] - 1.0] * (64 - len(score)) for score in scores]
    return torch.tensor(padded_rows), torch.tensor(padded_scores)


def test_query_level_reserve_expands_only_for_weak_t0() -> None:
    ranks = torch.arange(2560)
    count, expanded = _query_level_feedback_count(
        ranks,
        extracted_count=2560,
        first_pass_query_cap=2048,
        baseline_inlier_count=490,
        first_pass_match_count=1400,
        expanded_reserve_maximum_inlier_fraction=0.35,
    )
    assert (count, expanded) == (2560, True)

    count, expanded = _query_level_feedback_count(
        ranks,
        extracted_count=2560,
        first_pass_query_cap=2048,
        baseline_inlier_count=491,
        first_pass_match_count=1400,
        expanded_reserve_maximum_inlier_fraction=0.35,
    )
    assert (count, expanded) == (2048, False)


def test_query_level_reserve_rejects_non_prefix_detector_registry() -> None:
    ranks = torch.tensor([0, 2, 3, 1])
    with pytest.raises(ValueError, match="exact prefix"):
        _query_level_feedback_count(
            ranks,
            extracted_count=4,
            first_pass_query_cap=2,
            baseline_inlier_count=1,
            first_pass_match_count=2,
            expanded_reserve_maximum_inlier_fraction=0.5,
        )


def test_joint_selector_preserves_inliers_and_reserves_their_anchor() -> None:
    # Under identity K/pose, XYZ x/z,y/z are pixel coordinates.
    xyz = torch.tensor(
        [
            [0.0, 0.0, 1.0],  # protected row-0 baseline
            [20.0, 0.0, 1.0],  # row-1 bad baseline
            [1.0, 0.0, 1.0],  # row-1 correct alternative
            [30.0, 0.0, 1.0],  # row-2 bad baseline
        ]
    )
    candidates, scores = _topk(
        [[0, 2], [1, 2, 0], [3, 2]],
        [[1.0, 0.99], [1.0, 0.99, 0.99], [1.0, 0.95]],
    )
    result = select_pose_conditioned_rows(
        keypoints=torch.tensor([[0.0, 0.0], [1.0, 0.0], [1.1, 0.0]]),
        topk_anchor_rows=candidates,
        topk_scores=scores,
        baseline_inlier_rows=torch.tensor([0]),
        anchor_xyz=xyz,
        intrinsic=torch.eye(3),
        baseline_pose_w2c=torch.eye(4),
        config=default_config(),
    )
    # Row 0 is protected. Rows 1 and 2 compete for Anchor 2; the lower joint
    # cost row wins. Anchor 0 cannot be stolen because a protected row owns it.
    assert result["anchor_rows"].tolist() == [0, 2, 3]
    assert result["changed_query_rows"].tolist() == [1]
    assert result["duplicate_candidate_owner_rejection_count"] == 1


def test_joint_selector_requires_geometric_improvement_and_descriptor_support() -> None:
    xyz = torch.tensor(
        [
            [10.0, 0.0, 1.0],
            [9.5, 0.0, 1.0],  # only 0.5 px better: rejected
            [0.0, 0.0, 1.0],  # geometric fit but score drop too large
        ]
    )
    candidates, scores = _topk([[0, 1, 2]], [[1.0, 0.99, 0.80]])
    result = select_pose_conditioned_rows(
        keypoints=torch.tensor([[0.0, 0.0]]),
        topk_anchor_rows=candidates,
        topk_scores=scores,
        baseline_inlier_rows=torch.empty(0, dtype=torch.long),
        anchor_xyz=xyz,
        intrinsic=torch.eye(3),
        baseline_pose_w2c=torch.eye(4),
    )
    assert result["changed_query_rows"].numel() == 0
    assert result["anchor_rows"].tolist() == [0]


def test_reliability_authorizes_only_strong_geometry_score_drop_expansion() -> None:
    xyz = torch.tensor(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],  # reliable alternative for row 1
            [20.0, 0.0, 1.0],  # bad row-1 baseline
            [30.0, 0.0, 1.0],
        ]
    )
    candidates, scores = _topk(
        [[0], [2, 1]],
        [[1.0], [1.0, 0.93]],
    )
    common = dict(
        keypoints=torch.tensor([[0.0, 0.0], [1.0, 0.0]]),
        topk_anchor_rows=candidates,
        topk_scores=scores,
        baseline_inlier_rows=torch.tensor([0]),
        anchor_xyz=xyz,
        intrinsic=torch.eye(3),
        baseline_pose_w2c=torch.eye(4),
        anchor_matchability=torch.tensor([0.8, 0.9, 0.1, 0.2]),
        anchor_uncertainty=torch.tensor([0.2, 0.1, 2.0, 1.0]),
    )
    fixed = select_pose_conditioned_rows(
        **common,
        config=runtime_config(maximum_score_drop_from_top1=0.03),
    )
    adaptive = select_pose_conditioned_rows(
        **common,
        config=runtime_config(
            maximum_score_drop_from_top1=0.03,
            reliability_adaptive_score_drop=True,
            reliability_expanded_score_drop=0.10,
        ),
    )

    assert fixed["changed_query_rows"].numel() == 0
    assert adaptive["changed_query_rows"].tolist() == [1]
    assert adaptive["reliability_expanded_budget_edge_count"] == 1
    assert adaptive["reliability_expanded_selected_row_count"] == 1


def test_reliability_expansion_rejects_low_quality_anchor() -> None:
    xyz = torch.tensor(
        [[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [20.0, 0.0, 1.0]]
    )
    candidates, scores = _topk([[0], [2, 1]], [[1.0], [1.0, 0.93]])
    result = select_pose_conditioned_rows(
        keypoints=torch.tensor([[0.0, 0.0], [1.0, 0.0]]),
        topk_anchor_rows=candidates,
        topk_scores=scores,
        baseline_inlier_rows=torch.tensor([0]),
        anchor_xyz=xyz,
        intrinsic=torch.eye(3),
        baseline_pose_w2c=torch.eye(4),
        anchor_matchability=torch.tensor([0.8, 0.1, 0.9]),
        anchor_uncertainty=torch.tensor([0.2, 2.0, 0.1]),
        config=runtime_config(
            maximum_score_drop_from_top1=0.03,
            reliability_adaptive_score_drop=True,
            reliability_expanded_score_drop=0.10,
        ),
    )

    assert result["changed_query_rows"].numel() == 0
    assert result["reliability_expanded_budget_edge_count"] == 0


def test_pose_conditioned_mutual_matching_keeps_anchor_best_query() -> None:
    # Both rows can geometrically claim Anchor 2.  Joint cost prefers row 0,
    # while the mutual descriptor check correctly lets Anchor 2 choose row 1.
    xyz = torch.tensor(
        [
            [10.0, 0.0, 1.0],
            [10.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [4.0, 0.0, 1.0],
        ]
    )
    candidates, scores = _topk(
        [[0, 2], [1, 2], [3]],
        [[1.0, 0.98], [1.0, 0.99], [1.0]],
    )
    common = dict(
        keypoints=torch.tensor([[0.0, 0.0], [2.0, 0.0], [4.0, 0.0]]),
        topk_anchor_rows=candidates,
        topk_scores=scores,
        baseline_inlier_rows=torch.tensor([2]),
        anchor_xyz=xyz,
        intrinsic=torch.eye(3),
        baseline_pose_w2c=torch.eye(4),
    )
    ordinary = select_pose_conditioned_rows(**common)
    mutual = select_pose_conditioned_rows(
        **common,
        config=runtime_config(pose_conditioned_mutual_matching=True),
    )

    assert ordinary["changed_query_rows"].tolist() == [0]
    assert mutual["changed_query_rows"].tolist() == [1]
    assert mutual["mutual_candidate_matching_enabled"] is True
    assert mutual["mutual_candidate_rejected_edge_count"] == 1


def test_heldout_rows_are_excluded_from_pose_proposals() -> None:
    count = 21
    candidates = torch.arange(64).repeat(count, 1)
    scores = torch.linspace(1.0, 0.0, 64).repeat(count, 1)
    xyz = torch.stack(
        (torch.arange(64).float() * 10.0, torch.zeros(64), torch.ones(64)),
        dim=1,
    )
    result = select_pose_conditioned_rows(
        keypoints=torch.zeros((count, 2)),
        topk_anchor_rows=candidates,
        topk_scores=scores,
        baseline_inlier_rows=torch.tensor([20]),
        anchor_xyz=xyz,
        intrinsic=torch.eye(3),
        baseline_pose_w2c=torch.eye(4),
        image_hw=(100, 100),
        config=runtime_config(heldout_candidate_validation=True),
    )

    assert result["heldout_validation_query_rows"].tolist() == list(range(20))
    assert result["heldout_validation_edge_count"] >= 40
    assert result["changed_query_rows"].numel() == 0


def test_heldout_strict_assignment_prefers_better_independent_pose() -> None:
    rows = 6
    xy = torch.stack((torch.arange(rows).float(), torch.zeros(rows)), dim=1)
    candidates = torch.stack((torch.arange(rows), torch.arange(rows) + rows), dim=1)
    xyz = torch.zeros((rows * 2, 3))
    xyz[:rows, 0] = torch.arange(rows).float() + 1.0
    xyz[rows:, 0] = torch.arange(rows).float() - 1.0
    xyz[:, 2] = 1.0
    candidate_pose = torch.eye(4)
    candidate_pose[0, 3] = 1.0
    result = compare_poses_on_heldout_candidate_graph(
        keypoints=xy,
        candidate_anchor_rows=candidates,
        candidate_scores=torch.tensor([[1.0, 0.99]]).repeat(rows, 1),
        candidate_edge_mask=torch.ones((rows, 2), dtype=torch.bool),
        anchor_xyz=xyz,
        intrinsic=torch.eye(3),
        baseline_pose_w2c=torch.eye(4),
        candidate_pose_w2c=candidate_pose,
        maximum_score_drop_from_top1=0.03,
        robust_scale_px=1.0,
    )

    assert result["strict_one_to_one"] is True
    assert result["solver_rows_used"] is False
    assert result["candidate_energy"] < result["baseline_energy"]
    assert result["relative_energy_gain"] > 0


def test_runtime_projection_gate_can_expand_the_pose_basin() -> None:
    xyz = torch.tensor([[0.0, 0.0, 1.0], [20.0, 0.0, 1.0], [10.0, 0.0, 1.0]])
    candidates, scores = _topk([[0], [1, 2]], [[1.0], [1.0, 0.99]])
    common = dict(
        keypoints=torch.tensor([[0.0, 0.0], [0.0, 0.0]]),
        topk_anchor_rows=candidates,
        topk_scores=scores,
        baseline_inlier_rows=torch.tensor([0]),
        anchor_xyz=xyz,
        intrinsic=torch.eye(3),
        baseline_pose_w2c=torch.eye(4),
    )
    narrow = select_pose_conditioned_rows(
        **common, config=runtime_config(projection_gate_px=8.0)
    )
    expanded = select_pose_conditioned_rows(
        **common, config=runtime_config(projection_gate_px=12.0)
    )

    assert narrow["changed_query_rows"].numel() == 0
    assert expanded["changed_query_rows"].tolist() == [1]


def test_projection_first_retrieval_recovers_descriptor_rank_beyond_topk() -> None:
    features = torch.zeros((66, 2))
    features[:64, 0] = 1.0
    features[64] = torch.tensor([0.8, 0.6])
    features[65, 0] = 1.0
    xyz = torch.tensor([[20.0, 0.0, 1.0]]).repeat(66, 1)
    xyz[64] = torch.tensor([0.0, 0.0, 1.0])
    result = build_pose_visible_topk(
        query_descriptors=torch.tensor([[1.0, 0.0]]),
        query_keypoints=torch.tensor([[0.0, 0.0]]),
        normalized_anchor_features=features,
        baseline_anchor_rows=torch.tensor([65]),
        baseline_scores=torch.tensor([1.0]),
        anchor_xyz=xyz,
        intrinsic=torch.eye(3),
        baseline_pose_w2c=torch.eye(4),
        image_hw=(100, 100),
        projection_first_local_candidates=True,
        projection_first_radius_px=4.0,
    )
    matches = result["matches"]
    assert matches.anchor_indices[0, 0].item() == 65
    assert matches.anchor_indices[0, 1].item() == 64
    assert matches.scores[0, 1].item() == pytest.approx(0.8)
    assert result["projection_local_edge_count"] == 1


def test_set_level_reserve_selection_respects_existing_capacity() -> None:
    rows = 7
    xyz = torch.zeros((rows * 2, 3))
    xyz[:rows, 0] = 20.0
    xyz[rows:, 0] = torch.arange(rows).float()
    xyz[:, 2] = 1.0
    candidates, scores = _topk(
        [[index, rows + index] for index in range(rows)],
        [[1.0, 0.99] for _ in range(rows)],
    )
    result = select_pose_conditioned_rows(
        keypoints=torch.stack((torch.arange(rows).float(), torch.zeros(rows)), dim=1),
        topk_anchor_rows=candidates,
        topk_scores=scores,
        baseline_inlier_rows=torch.tensor([0]),
        anchor_xyz=xyz,
        intrinsic=torch.eye(3),
        baseline_pose_w2c=torch.eye(4),
        image_hw=(10, 10),
        config=runtime_config(
            maximum_changed_rows=2,
            maximum_changed_to_baseline_inlier_ratio=2.0,
            set_level_reserve_selection=True,
        ),
    )
    assert result["set_level_reserve_selection_enabled"] is True
    assert result["changed_query_rows"].numel() == 2
    assert result["capacity_rejection_count"] == 4


def test_full_anchor_covariance_expands_only_bounded_pixel_gate() -> None:
    xyz = torch.tensor(
        [
            [0.0, 0.0, 5.0],
            [1.0, 0.0, 5.0],
            [0.0, 1.0, 5.0],
            [1.0, 1.0, 5.0],
            [-1.0, 0.0, 5.0],
            [0.0, -1.0, 5.0],
            [1.0, 0.0, 1.0],
            [0.1, 0.0, 1.0],
        ]
    )
    keypoints = torch.tensor(
        [
            [0.0, 0.0],
            [20.0, 0.0],
            [0.0, 20.0],
            [20.0, 20.0],
            [-20.0, 0.0],
            [0.0, -20.0],
            [0.0, 0.0],
        ]
    )
    candidates, scores = _topk(
        [[0], [1], [2], [3], [4], [5], [6, 7]],
        [[1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0, 0.99]],
    )
    covariance = torch.zeros((8, 3, 3))
    covariance[7, 0, 0] = 0.01
    common = dict(
        keypoints=keypoints,
        topk_anchor_rows=candidates,
        topk_scores=scores,
        baseline_inlier_rows=torch.arange(6),
        anchor_xyz=xyz,
        intrinsic=torch.tensor([[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 1.0]]),
        baseline_pose_w2c=torch.eye(4),
        anchor_position_covariance=covariance,
    )
    fixed = select_pose_conditioned_rows(**common)
    adaptive = select_pose_conditioned_rows(
        **common,
        config=runtime_config(
            uncertainty_aware_projection=True,
            maximum_uncertainty_projection_gate_px=12.0,
        ),
    )

    assert fixed["changed_query_rows"].numel() == 0
    assert adaptive["changed_query_rows"].tolist() == [6]
    assert adaptive["projection_gate_p90_px"] <= 12.0
    assert adaptive["expanded_projection_edge_count"] > 0


def test_bounded_soft_inlier_can_replace_only_a_high_residual_inlier() -> None:
    xyz = torch.tensor(
        [
            [7.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
        ]
    )
    candidates, scores = _topk(
        [[0, 1], [2, 3]],
        [[1.0, 0.99], [1.0, 0.99]],
    )
    result = select_pose_conditioned_rows(
        keypoints=torch.tensor([[0.0, 0.0], [0.0, 0.0]]),
        topk_anchor_rows=candidates,
        topk_scores=scores,
        baseline_inlier_rows=torch.tensor([0, 1]),
        anchor_xyz=xyz,
        intrinsic=torch.eye(3),
        baseline_pose_w2c=torch.eye(4),
        config=runtime_config(
            allow_soft_inliers=True,
            soft_inlier_minimum_baseline_residual_px=6.0,
            soft_inlier_maximum_score_drop=0.02,
            soft_inlier_minimum_reprojection_improvement_px=2.0,
            maximum_soft_inlier_changes=1,
        ),
    )

    assert result["anchor_rows"].tolist() == [1, 2]
    assert result["soft_inlier_candidate_row_count"] == 1
    assert result["soft_inlier_changed_row_count"] == 1
    assert result["hard_core_inlier_row_count"] == 1


def test_changed_inlier_support_counts_spatially_distinct_cells() -> None:
    count = changed_inlier_spatial_cell_count(
        keypoints=torch.tensor(
            [[10.0, 10.0], [90.0, 10.0], [10.0, 90.0], [90.0, 90.0]]
        ),
        changed_query_rows=torch.tensor([0, 1, 2, 3]),
        candidate_inlier_rows=torch.tensor([0, 1, 3]),
        image_hw=(100, 100),
        grid_shape=(2, 2),
    )
    assert count == 3


def test_common_candidate_grid_prefers_pose_with_better_shared_explanation() -> None:
    candidates, scores = _topk(
        [[0, 3], [1, 4], [2, 5]],
        [[1.0, 0.99], [1.0, 0.99], [1.0, 0.99]],
    )
    # Rank-0 points explain the first pose; rank-1 points explain a candidate
    # translated by +1 on x. Row zero is hard core, so both poses must retain
    # its baseline Anchor while the other rows compare the same candidate grid.
    xyz = torch.tensor(
        [
            [0.0, 0.0, 1.0],
            [4.0, 0.0, 1.0],
            [5.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [2.0, 0.0, 1.0],
        ]
    )
    candidate_pose = torch.eye(4)
    candidate_pose[0, 3] = 1.0
    result = compare_poses_on_common_candidate_grid(
        keypoints=torch.tensor([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]),
        topk_anchor_rows=candidates,
        topk_scores=scores,
        baseline_inlier_rows=torch.tensor([0]),
        anchor_xyz=xyz,
        intrinsic=torch.eye(3),
        baseline_pose_w2c=torch.eye(4),
        candidate_pose_w2c=candidate_pose,
        maximum_score_drop_from_top1=0.03,
        robust_scale_px=2.0,
    )

    assert result["candidate_energy"] < result["baseline_energy"]
    assert result["relative_energy_gain"] > 0


def test_mapping_view_support_rejects_an_unobserved_candidate_side() -> None:
    xyz = torch.tensor([[10.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    candidates, scores = _topk([[0, 1]], [[1.0, 0.99]])
    support = {
        "schema": "lafgs_v24_anchor_view_support",
        "uses_test_queries": False,
        "direction_modes": torch.tensor(
            [[[0.0, 0.0, -1.0]] * 2, [[0.0, 0.0, 1.0]] * 2]
        ),
        "direction_radius_deg": torch.tensor([[5.0, -1.0], [5.0, -1.0]]),
        "mode_count": torch.ones(2, dtype=torch.long),
        "minimum_distance_m": torch.ones(2) * 0.5,
        "maximum_distance_m": torch.ones(2) * 2.0,
    }
    result = select_pose_conditioned_rows(
        keypoints=torch.tensor([[0.0, 0.0]]),
        topk_anchor_rows=candidates,
        topk_scores=scores,
        baseline_inlier_rows=torch.empty(0, dtype=torch.long),
        anchor_xyz=xyz,
        intrinsic=torch.eye(3),
        baseline_pose_w2c=torch.eye(4),
        anchor_view_support=support,
    )
    assert result["view_support_available"] is True
    assert result["changed_query_rows"].numel() == 0
    assert result["view_support_rejected_edge_count"] >= 1


def test_anchor_view_support_is_mapping_only_and_aligned() -> None:
    poses = torch.eye(4).repeat(2, 1, 1)
    poses[1, 0, 3] = -1.0  # second camera centre is +1 on x
    support = build_anchor_view_support(
        anchor_xyz=torch.tensor([[0.0, 0.0, 2.0]]),
        observation_offsets=torch.tensor([0, 2]),
        observation_query_indices=torch.tensor([0, 1]),
        mapping_pose_w2c=poses,
    )
    assert support["uses_test_queries"] is False
    assert support["direction_modes"].shape == (1, 2, 3)
    assert support["observation_count"].tolist() == [2]
    assert support["minimum_distance_m"][0] > 0


def test_pose_visible_topk_keeps_global_winner_and_localizes_pool() -> None:
    generator = torch.Generator().manual_seed(4)
    anchors = torch.nn.functional.normalize(
        torch.randn(80, 8, generator=generator), dim=1
    )
    query = torch.nn.functional.normalize(torch.randn(3, 8, generator=generator), dim=1)
    baseline = global_cosine_top1(query, anchors, anchor_descriptors_normalized=True)
    xyz = torch.stack(
        (
            torch.linspace(-0.5, 0.5, 80),
            torch.linspace(-0.25, 0.25, 80),
            torch.ones(80) * 4.0,
        ),
        dim=1,
    )
    result = build_pose_visible_topk(
        query_descriptors=query,
        normalized_anchor_features=anchors,
        baseline_anchor_rows=baseline.anchor_indices,
        baseline_scores=baseline.scores,
        anchor_xyz=xyz,
        intrinsic=torch.tensor(
            [[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]]
        ),
        baseline_pose_w2c=torch.eye(4),
        image_hw=(100, 100),
    )

    matches = result["matches"]
    assert result["visible_anchor_count"] == 80
    assert result["candidate_pool_anchor_count"] == 80
    assert not result["global_fallback"]
    assert torch.equal(matches.anchor_indices[:, 0], baseline.anchor_indices)
    assert all(row.unique().numel() == 64 for row in matches.anchor_indices)


def test_pose_visible_topk_scores_only_requested_query_rows() -> None:
    generator = torch.Generator().manual_seed(12)
    anchors = torch.nn.functional.normalize(
        torch.randn(80, 8, generator=generator), dim=1
    )
    query = torch.nn.functional.normalize(torch.randn(3, 8, generator=generator), dim=1)
    baseline = global_cosine_top1(query, anchors, anchor_descriptors_normalized=True)
    xyz = torch.stack(
        (torch.linspace(-0.5, 0.5, 80), torch.zeros(80), torch.ones(80) * 4.0),
        dim=1,
    )

    result = build_pose_visible_topk(
        query_descriptors=query,
        normalized_anchor_features=anchors,
        baseline_anchor_rows=baseline.anchor_indices,
        baseline_scores=baseline.scores,
        anchor_xyz=xyz,
        intrinsic=torch.tensor(
            [[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]]
        ),
        baseline_pose_w2c=torch.eye(4),
        image_hw=(100, 100),
        retrieval_query_rows=torch.tensor([1]),
    )

    matches = result["matches"]
    assert result["retrieval_query_count"] == 1
    assert matches.anchor_indices[0].unique().numel() == 1
    assert matches.anchor_indices[1].unique().numel() == 64
    assert matches.anchor_indices[2].unique().numel() == 1


def test_pose_visible_topk_prefilters_mapping_unsupported_anchors() -> None:
    generator = torch.Generator().manual_seed(19)
    count = 100
    anchors = torch.nn.functional.normalize(
        torch.randn(count, 8, generator=generator), dim=1
    )
    query = torch.nn.functional.normalize(torch.randn(2, 8, generator=generator), dim=1)
    baseline = global_cosine_top1(query, anchors, anchor_descriptors_normalized=True)
    xyz = torch.stack(
        (
            torch.linspace(-0.5, 0.5, count),
            torch.zeros(count),
            torch.ones(count) * 4.0,
        ),
        dim=1,
    )
    supported_direction = torch.nn.functional.normalize(-xyz, dim=1)
    modes = supported_direction[:, None, :].repeat(1, 2, 1)
    modes[70:] *= -1.0
    support = {
        "schema": "lafgs_v24_anchor_view_support",
        "uses_test_queries": False,
        "direction_modes": modes,
        "direction_radius_deg": torch.tensor([[5.0, -1.0]]).repeat(count, 1),
        "mode_count": torch.ones(count, dtype=torch.long),
        "minimum_distance_m": torch.ones(count),
        "maximum_distance_m": torch.ones(count) * 8.0,
    }

    result = build_pose_visible_topk(
        query_descriptors=query,
        normalized_anchor_features=anchors,
        baseline_anchor_rows=baseline.anchor_indices,
        baseline_scores=baseline.scores,
        anchor_xyz=xyz,
        intrinsic=torch.tensor(
            [[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]]
        ),
        baseline_pose_w2c=torch.eye(4),
        image_hw=(100, 100),
        anchor_view_support=support,
        prefilter_mapping_view_support=True,
        view_direction_slack_deg=0.0,
    )

    assert result["view_supported_anchor_count"] == 70
    assert result["candidate_pool_anchor_count"] == 70
    assert not result["view_support_prefilter_fallback"]


def test_view_conditioned_max_fusion_adds_evidence_without_new_owner() -> None:
    generator = torch.Generator().manual_seed(29)
    count = 80
    anchors = torch.nn.functional.normalize(
        torch.randn(count, 8, generator=generator), dim=1
    )
    query = torch.nn.functional.normalize(torch.randn(1, 8, generator=generator), dim=1)
    baseline = global_cosine_top1(query, anchors, anchor_descriptors_normalized=True)
    target = (int(baseline.anchor_indices[0]) + 1) % count
    xyz = torch.stack(
        (torch.linspace(-0.5, 0.5, count), torch.zeros(count), torch.ones(count) * 4),
        dim=1,
    )
    directions = torch.nn.functional.normalize(-xyz, dim=1)
    modes = directions[:, None].repeat(1, 2, 1)
    support = {
        "schema": "lafgs_v24_anchor_view_support",
        "uses_test_queries": False,
        "direction_modes": modes,
        "direction_radius_deg": torch.ones(count, 2) * 10,
        "mode_count": torch.ones(count, dtype=torch.long) * 2,
        "minimum_distance_m": torch.ones(count),
        "maximum_distance_m": torch.ones(count) * 8,
    }
    mode_features = anchors[:, None].repeat(1, 2, 1)
    mode_features[target] = query[0]
    result = build_pose_visible_topk(
        query_descriptors=query,
        normalized_anchor_features=anchors,
        baseline_anchor_rows=baseline.anchor_indices,
        baseline_scores=baseline.scores,
        anchor_xyz=xyz,
        intrinsic=torch.tensor(
            [[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]]
        ),
        baseline_pose_w2c=torch.eye(4),
        image_hw=(100, 100),
        anchor_view_support=support,
        view_conditioned_descriptor_state={
            "mode_features": mode_features,
            "mode_valid": torch.ones(count, 2, dtype=torch.bool),
            "mode_concentration": torch.ones(count, 2),
        },
        view_conditioned_require_two_valid_modes=True,
        view_conditioned_score_fusion="max_with_base",
    )
    matches = result["matches"]
    target_column = torch.nonzero(
        matches.anchor_indices[0] == target, as_tuple=False
    ).item()
    assert torch.isclose(matches.scores[0, target_column], torch.tensor(1.0))
    assert matches.anchor_indices[0].unique().numel() == 64


def test_spatial_jackknife_deletes_each_occupied_image_cell(monkeypatch) -> None:
    def fake_refine(
        points_2d,
        points_3d,
        intrinsic,
        initial_pose_w2c,
        optimization_rows,
        **kwargs,
    ):
        pose = np.asarray(initial_pose_w2c).copy()
        pose[0, 3] += float(np.mean(optimization_rows)) * 1e-5
        return pose_solver.PoseEstimate(
            pose, np.asarray(optimization_rows), {"iterations": 1}
        )

    monkeypatch.setattr(pose_solver, "refine_absolute_pose_from_initial", fake_refine)
    xy = np.asarray(
        [[10.0, 10.0]] * 5
        + [[90.0, 10.0]] * 5
        + [[10.0, 90.0]] * 5
        + [[90.0, 90.0]] * 5
    )
    result = spatial_jackknife_pose_stability(
        keypoints=xy,
        points_3d=np.column_stack((xy, np.ones(20))),
        pose_w2c=np.eye(4),
        inlier_rows=np.arange(20),
        intrinsic=np.eye(3),
        image_hw=(100, 100),
        reprojection_error_px=12.0,
        grid_shape=(2, 2),
        minimum_remaining_inliers=12,
    )
    assert result["valid_group_count"] == 4
    assert result["removed_inlier_count_maximum"] == 5
    assert result["normalized_task_update_p90"] > 0


def test_local_pose_refinement_scores_the_complete_correspondence_set(
    monkeypatch,
) -> None:
    def identity_refinement(points_2d, points_3d, initial, camera, options):
        assert points_2d.shape[0] == 4
        return initial, {"iterations": 2}

    monkeypatch.setattr(
        pose_solver.poselib, "refine_absolute_pose", identity_refinement
    )
    points_3d = np.asarray(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [4.0, 4.0, 1.0],
        ]
    )
    points_2d = points_3d[:, :2].copy()
    result = pose_solver.refine_absolute_pose_from_initial(
        points_2d,
        points_3d,
        np.eye(3),
        np.eye(4),
        np.asarray([0, 1, 2, 3]),
        reprojection_error_px=0.1,
    )
    assert result.inliers.tolist() == [0, 1, 2, 3, 4]
    assert result.diagnostics["optimization_correspondence_count"] == 4
