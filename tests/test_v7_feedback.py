import inspect

import numpy as np
import pytest
import torch

from localization.matcher import global_cosine_top1
from localization.pose_solver import solve_absolute_pose
from map_learning.v7_feedback import (
    DiagnosticRegistry,
    V7PoseEstimate,
    V7Top1Matches,
    V7LocalizationResult,
    diagnose_feedback_query,
    localize_rgb_query,
    _global_cosine_top1,
    _solve_standard_poselib,
)


def _result(*, pose=None, registry=True):
    xyz = torch.tensor([[0.0, 0.0, 2.0], [0.1, 0.0, 2.0]])
    return V7LocalizationResult(
        keypoints=torch.tensor([[50.0, 50.0], [55.0, 50.0]]),
        descriptors=torch.eye(2),
        scores=torch.ones(2),
        matches=V7Top1Matches(torch.arange(2), torch.arange(2), torch.ones(2)),
        pose=V7PoseEstimate(
            np.eye(4, dtype=np.float32) if pose is None else pose,
            np.arange(2),
            {},
        ),
        intrinsic=np.array(
            [[100.0, 0.0, 50.5], [0.0, 100.0, 50.5], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        ),
        runtime_ms={},
        active_anchor_ids=torch.arange(2),
        active_anchor_xyz=xyz,
        diagnostic_registry=(
            DiagnosticRegistry(torch.arange(2), xyz, torch.ones(2, dtype=torch.bool))
            if registry
            else None
        ),
    )


def _certificate(decision="ACCEPT"):
    return {
        "decision": decision,
        "can_drive_map_update": decision == "ACCEPT",
        "row_valid": torch.ones(2, dtype=torch.bool),
    }


def _rasters():
    return torch.ones(1, 2, 2), torch.full((1, 2, 2), 2.0)


def test_localize_interface_cannot_receive_oracle_inputs() -> None:
    assert tuple(inspect.signature(localize_rgb_query).parameters) == (
        "rgb",
        "intrinsics",
        "map_state",
    )


def test_v7_top1_is_exactly_current_global_top1() -> None:
    generator = torch.Generator().manual_seed(17)
    query = torch.randn(19, 256, generator=generator)
    anchors = torch.nn.functional.normalize(
        torch.randn(67, 256, generator=generator), dim=1
    )
    expected = global_cosine_top1(query, anchors, anchor_descriptors_normalized=True)
    actual = _global_cosine_top1(query, anchors)
    assert torch.equal(actual.keypoint_indices, expected.keypoint_indices)
    assert torch.equal(actual.anchor_indices, expected.anchor_indices)
    assert torch.equal(actual.scores, expected.scores)


def test_v7_poselib_is_exactly_current_standard_wrapper() -> None:
    intrinsic = np.array([[300.0, 0.0, 160.0], [0.0, 300.0, 120.0], [0.0, 0.0, 1.0]])
    points_3d = np.array(
        [[x, y, 4.0 + 0.1 * x] for x in (-1.0, -0.5, 0.5, 1.0) for y in (-0.5, 0.5)]
    )
    camera = points_3d.T
    points_2d = (intrinsic @ camera).T
    points_2d = points_2d[:, :2] / points_2d[:, 2:3]
    expected = solve_absolute_pose(
        points_2d,
        points_3d,
        intrinsic,
        reprojection_error_px=12.0,
        confidence=0.99999,
        max_iterations=100000,
        min_iterations=1000,
        seed=2026,
        progressive_sampling=False,
    )
    actual = _solve_standard_poselib(
        points_2d,
        points_3d,
        intrinsic,
        reprojection_error_px=12.0,
        confidence=0.99999,
        maximum_iterations=100000,
        minimum_iterations=1000,
        seed=2026,
    )
    assert np.array_equal(actual.pose_w2c, expected.pose_w2c)
    assert np.array_equal(actual.inliers, expected.inliers)


@pytest.mark.parametrize("decision", ["UNCERTAIN", "REJECT"])
def test_nonaccept_is_unreliable_and_cannot_update(decision) -> None:
    alpha, depth = _rasters()
    diagnosis = diagnose_feedback_query(
        _result(), torch.eye(4), alpha, depth, _certificate(decision)
    )
    assert diagnosis["category"] == "unreliable_query"
    assert diagnosis["can_drive_map_update"] is False


def test_accept_success_is_nominal() -> None:
    alpha, depth = _rasters()
    diagnosis = diagnose_feedback_query(
        _result(),
        torch.eye(4),
        alpha,
        depth,
        _certificate(),
        minimum_oracle_correspondences=2,
    )
    assert diagnosis["category"] == "nominal_success"
    assert diagnosis["oracle_used_online"] is False


def test_missing_candidate_support_is_coverage_deficit() -> None:
    bad = np.eye(4, dtype=np.float32)
    bad[0, 3] = 1.0
    result = _result(pose=bad)
    result.diagnostic_registry.eligible[:] = False
    alpha, depth = _rasters()
    diagnosis = diagnose_feedback_query(
        result,
        torch.eye(4),
        alpha,
        depth,
        _certificate(),
        minimum_oracle_correspondences=2,
    )
    assert diagnosis["category"] == "coverage_deficit"


def test_sufficient_oracle_support_routes_representation_deficit() -> None:
    bad = np.eye(4, dtype=np.float32)
    bad[0, 3] = 1.0
    alpha, depth = _rasters()
    diagnosis = diagnose_feedback_query(
        _result(pose=bad),
        torch.eye(4),
        alpha,
        depth,
        _certificate(),
        minimum_oracle_correspondences=2,
    )
    assert diagnosis["category"] == "representation_deficit"
