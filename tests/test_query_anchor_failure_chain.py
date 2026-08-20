from __future__ import annotations

import numpy as np

from evaluation.query_anchor_failure_chain import (
    classify_failure_chain,
    descriptor_recall_diagnostics,
    geometry_diagnostics,
    maximum_cardinality_minimum_distance_matching,
    nearby_anchor_edges,
    project_gt_visible_anchors,
)


def _scene():
    intrinsic = np.asarray(
        [[100.0, 0.0, 50.0], [0.0, 100.0, 40.0], [0.0, 0.0, 1.0]]
    )
    xyz = np.asarray(
        [
            [-0.5, -0.4, 4.0], [0.5, -0.4, 4.1], [0.6, 0.5, 4.3],
            [-0.6, 0.5, 4.2], [0.0, 0.0, 3.4], [0.2, -0.1, 5.0],
        ]
    )
    pose = np.eye(4)
    camera = xyz @ pose[:3, :3].T + pose[:3, 3]
    uvh = camera @ intrinsic.T
    uv = uvh[:, :2] / uvh[:, 2:]
    return intrinsic, xyz, pose, uv


def test_gt_visibility_applies_alpha_depth_and_positive_depth() -> None:
    intrinsic, xyz, pose, uv = _scene()
    alpha = np.ones((80, 100))
    depth = np.zeros((80, 100))
    for point, z in zip(uv, xyz[:, 2]):
        depth[int(round(point[1])), int(round(point[0]))] = z
    alpha[int(round(uv[1, 1])), int(round(uv[1, 0]))] = 0.0
    depth[int(round(uv[2, 1])), int(round(uv[2, 0]))] += 1.0
    result = project_gt_visible_anchors(
        xyz, intrinsic, pose, image_size=(100, 80),
        rendered_alpha=alpha, rendered_depth=depth,
        alpha_minimum=0.05, depth_abs_tolerance_m=0.1,
        depth_relative_tolerance=0.0,
    )
    assert result.in_frame.all()
    assert result.visible.tolist() == [True, False, False, True, True, True]


def test_maximum_matching_prioritizes_cardinality_then_distance() -> None:
    # q0 can take a0 or a1; q1 can only take a0. A greedy q0->a0 would have
    # rank one, while the exact matching must return q0->a1 and q1->a0.
    matching = maximum_cardinality_minimum_distance_matching(
        np.asarray([0, 0, 1]), np.asarray([0, 1, 0]), np.asarray([0.1, 0.2, 0.1]),
        query_count=2,
    )
    assert matching.rank == 2
    assert matching.query_rows.tolist() == [0, 1]
    assert matching.anchor_rows.tolist() == [1, 0]


def test_detector_edges_and_geometry_are_solvable_for_spread_points() -> None:
    intrinsic, xyz, pose, uv = _scene()
    q, a, d = nearby_anchor_edges(
        uv + np.asarray([0.2, -0.1]), uv, np.ones(len(xyz), dtype=bool),
        radius_px=1.0,
    )
    matching = maximum_cardinality_minimum_distance_matching(q, a, d, query_count=len(uv))
    assert matching.rank == len(xyz)
    geometry = geometry_diagnostics(
        uv, xyz, intrinsic, pose, image_size=(100, 80), seed=2026,
    )
    assert geometry["grid_coverage_4x4"] >= 4
    assert geometry["information_rank"] == 6
    assert geometry["pnp_geometry_solvable"] is True


def test_descriptor_recall_splits_track_and_surface_and_reports_margin() -> None:
    scores = np.asarray(
        [[0.9, 0.8, 0.7, 0.6], [0.9, 0.85, 0.4, 0.3]], dtype=np.float64
    )
    top = np.argsort(-scores, axis=1)
    diagnostics = descriptor_recall_diagnostics(
        scores, top,
        positive_query_rows=np.asarray([0, 1]),
        positive_anchor_rows=np.asarray([1, 2]),
        anchor_type=np.asarray([1, 1, 0, 0]),
        ks=(1, 2, 4),
    )
    assert diagnostics["all"]["recall_percent"] == {
        "r_at_1": 0.0, "r_at_2": 50.0, "r_at_4": 100.0
    }
    assert diagnostics["track"]["eligible_query_row_count"] == 1
    assert diagnostics["surface_completion"]["eligible_query_row_count"] == 1
    assert diagnostics["all"]["best_positive_minus_best_wrong_margin"]["mean"] < 0


def test_failure_categories_are_layer_ordered() -> None:
    common = dict(
        visible_geometry_solvable=True, detector_matching_rank=8,
        oracle_matching_rank=8, top32_positive_matching_rank=8,
        oracle_pose_correct=True, deployed_pose_correct=False,
    )
    assert classify_failure_chain(**{**common, "visible_geometry_solvable": False}).startswith("L1")
    assert classify_failure_chain(**{**common, "detector_matching_rank": 3}).startswith("L2")
    assert classify_failure_chain(**{**common, "top32_positive_matching_rank": 3}).startswith("L3")
    assert classify_failure_chain(**common).startswith("L4")
    assert classify_failure_chain(**{**common, "oracle_pose_correct": False}).startswith("L5")
