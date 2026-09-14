import torch

from evidence.rendered_track_support import local_depth_spread, pair_support_evidence
from topology.matching_coverage import track_candidate_edges


def _camera(tx: float = 0.0):
    pose = torch.eye(4)
    pose[0, 3] = tx
    intrinsic = torch.tensor([[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]])
    return intrinsic, pose


def _evidence(**overrides):
    intrinsic, left_pose = _camera()
    _, right_pose = _camera(-0.1)
    values = {
        "left_uv": torch.tensor([[50.0, 50.0]]),
        "right_uv": torch.tensor([[45.0, 50.0]]),
        "left_depth": torch.tensor([2.0]),
        "right_depth": torch.tensor([2.0]),
        "left_alpha": torch.tensor([1.0]),
        "right_alpha": torch.tensor([1.0]),
        "left_valid": torch.tensor([True]),
        "right_valid": torch.tensor([True]),
        "left_uncertainty": torch.tensor([0.0]),
        "right_uncertainty": torch.tensor([0.0]),
        "left_intrinsic": intrinsic,
        "right_intrinsic": intrinsic,
        "left_pose_w2c": left_pose,
        "right_pose_w2c": right_pose,
    }
    values.update(overrides)
    return pair_support_evidence(**values)


def test_consistent_support_edge_has_unit_weight_and_is_retained():
    result = _evidence()
    assert result["hard_reject"].tolist() == [False]
    assert result["high_confidence_support"].tolist() == [True]
    torch.testing.assert_close(
        result["cycle_error_px"], torch.zeros(1), atol=1e-5, rtol=0
    )
    assert float(result["soft_weight"][0]) > 0.999


def test_uncertain_expected_depth_never_hard_rejects_projective_edge():
    result = _evidence(
        right_depth=torch.tensor([5.0]),
        right_uncertainty=torch.tensor([10.0]),
    )
    assert result["high_confidence_support"].tolist() == [False]
    assert result["hard_reject"].tolist() == [False]
    assert 0.25 <= float(result["soft_weight"][0]) <= 1.0


def test_joint_high_confidence_cycle_and_depth_conflict_is_rejected():
    result = _evidence(
        right_uv=torch.tensor([[80.0, 50.0]]),
        right_depth=torch.tensor([5.0]),
    )
    assert result["high_confidence_support"].tolist() == [True]
    assert result["hard_reject"].tolist() == [True]


def test_local_depth_spread_marks_edges_and_holes_uncertain():
    depth = torch.full((5, 5), 2.0)
    alpha = torch.ones((5, 5))
    alpha[:, :2] = 0.0
    spread = local_depth_spread(
        depth,
        alpha,
        torch.tensor([[2.0, 2.0], [0.0, 2.0]]),
        alpha_minimum=0.05,
        radius=1,
    )
    assert float(spread[0]) == 0.0
    assert torch.isinf(spread[1])


def test_track_candidate_edges_use_only_certified_observations():
    payload = {
        "track_geometry": {"triangulated": torch.tensor([True, True])},
        "tracks": {
            "track_index": torch.tensor([0, 0, 1]),
            "query_index": torch.tensor([0, 1, 0]),
            "keypoint_index": torch.tensor([3, 4, 5]),
            "coverage_certified": torch.tensor([True, False, False]),
        },
    }
    assert track_candidate_edges(payload) == [{0: (3,)}, {}]
