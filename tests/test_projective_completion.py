import pytest
import torch

from evidence.projective_completion import build_projective_completion
from evidence.triangulation import reciprocal_epipolar_matches


def test_projective_completion_rejects_invalid_voxel_size() -> None:
    with pytest.raises(ValueError):
        build_projective_completion(
            None,
            {},
            voxel_size_m=0.0,
            alpha_minimum=0.05,
            minimum_similarity=0.65,
        )


def test_single_candidate_reciprocal_match_has_no_runner_up() -> None:
    intrinsic = torch.tensor(
        [[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]]
    )
    pose_a = torch.eye(4)
    pose_b = torch.eye(4)
    pose_b[0, 3] = -1.0
    source, target, confidence = reciprocal_epipolar_matches(
        torch.tensor([[1.0, 0.0]]),
        torch.tensor([[1.0, 0.0]]),
        torch.tensor([[50.0, 50.0]]),
        torch.tensor([[30.0, 50.0]]),
        intrinsic,
        pose_a,
        intrinsic,
        pose_b,
        minimum_similarity=0.65,
        minimum_margin=0.01,
        maximum_epipolar_error_px=2.0,
        epipolar_candidate_topk=4,
    )
    assert torch.equal(source, torch.tensor([0]))
    assert torch.equal(target, torch.tensor([0]))
    assert torch.isfinite(confidence).all()
