from __future__ import annotations

import torch
import numpy as np

from localization import pose_solver
from localization.matcher import global_cosine_top1
from map_learning.v24_anchor_view_support import build_anchor_view_support
from map_learning.v24_pose_conditioned_sparse_refinement import (
    build_pose_visible_topk,
    changed_inlier_spatial_cell_count,
    default_config,
    select_pose_conditioned_rows,
)


def _topk(rows: list[list[int]], scores: list[list[float]]) -> tuple[torch.Tensor, torch.Tensor]:
    padded_rows = [row + [row[0]] * (64 - len(row)) for row in rows]
    padded_scores = [score + [score[0] - 1.0] * (64 - len(score)) for score in scores]
    return torch.tensor(padded_rows), torch.tensor(padded_scores)


def test_joint_selector_preserves_inliers_and_reserves_their_anchor() -> None:
    # Under identity K/pose, XYZ x/z,y/z are pixel coordinates.
    xyz = torch.tensor(
        [
            [0.0, 0.0, 1.0],   # protected row-0 baseline
            [20.0, 0.0, 1.0],  # row-1 bad baseline
            [1.0, 0.0, 1.0],   # row-1 correct alternative
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
            [9.5, 0.0, 1.0],   # only 0.5 px better: rejected
            [0.0, 0.0, 1.0],   # geometric fit but score drop too large
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
    query = torch.nn.functional.normalize(
        torch.randn(3, 8, generator=generator), dim=1
    )
    baseline = global_cosine_top1(
        query, anchors, anchor_descriptors_normalized=True
    )
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
