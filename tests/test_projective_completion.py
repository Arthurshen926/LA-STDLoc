import pytest
import torch

from evidence.observation_provider import GaussianRenderObservationProvider
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


def test_completion_uses_target_only_to_seed_support_region() -> None:
    point = torch.tensor([0.0, 0.0, 5.0])
    intrinsic = torch.tensor(
        [[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]]
    )
    queries = {}
    names = []
    for index, center_x in enumerate((0.0, 0.5, 1.0, 1.5)):
        name = f"q{index}"
        names.append(name)
        pose = torch.eye(4)
        pose[0, 3] = -center_x
        camera = point @ pose[:3, :3].T + pose[:3, 3]
        physical = (camera @ intrinsic.T)[:2] / camera[2]
        queries[name] = {
            "native_keypoints": (physical - 0.5)[None],
            "native_descriptors": torch.tensor([[1.0, 0.0]]),
            "native_scores": torch.tensor([1.0]),
            "native_K": intrinsic,
            "pose_w2c": pose,
            "native_input_hw": torch.tensor([100, 100]),
            "native_depth_at_keypoints": torch.tensor([camera[2]]),
            "native_alpha_at_keypoints": torch.tensor([1.0]),
        }
    provider = GaussianRenderObservationProvider(
        {"uses_source_mapping_rgb": False, "queries": queries},
        query_names=names,
    )
    association = {
        "query_names": names,
        "query_bins": torch.arange(4),
        "tracks": {
            "query_index": torch.empty(0, dtype=torch.long),
            "keypoint_index": torch.empty(0, dtype=torch.long),
        },
    }
    completion = build_projective_completion(
        provider,
        association,
        voxel_size_m=0.5,
        alpha_minimum=0.05,
        minimum_similarity=0.65,
        minimum_observations=3,
        minimum_camera_families=2,
        target_query_indices=[0],
        excluded_support_query_indices=[0],
        device="cpu",
    )
    support = torch.as_tensor(
        completion["projective_anchor_observations"]["query_indices"]
    )
    assert 0 not in support.tolist()
    assert set(support.tolist()) == {1, 2, 3}
    assert completion["contract"]["target_queries_used_as_anchor_support"] is False
