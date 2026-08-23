import pytest
import torch

from evidence.observation_provider import GaussianRenderObservationProvider
from evidence.projective_loo import LeaveOneQueryOutProjectiveMap


def test_loo_rejects_non_projective_map() -> None:
    provider = GaussianRenderObservationProvider(
        {"uses_source_mapping_rgb": False, "queries": {"q": {}}},
        validate_all=False,
    )
    with pytest.raises(ValueError, match="registries"):
        LeaveOneQueryOutProjectiveMap(
            {"v6_mapping_query_names": [], "anchor_ids": torch.tensor([0])},
            provider,
        )


def test_loo_retriangulates_from_remaining_physical_pixel_centers() -> None:
    point = torch.tensor([0.0, 0.0, 5.0])
    K = torch.tensor([[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]])
    queries = {}
    names = []
    for index, center_x in enumerate((0.0, 0.5, 1.0, 1.5)):
        name = f"q{index}"
        names.append(name)
        pose = torch.eye(4)
        pose[0, 3] = -center_x
        camera = point @ pose[:3, :3].T + pose[:3, 3]
        physical = (camera @ K.T)[:2] / camera[2]
        queries[name] = {
            "native_keypoints": (physical - 0.5)[None],
            "native_descriptors": torch.tensor([[1.0, 0.0]]),
            "native_scores": torch.tensor([1.0]),
            "native_K": K,
            "pose_w2c": pose,
            "native_input_hw": torch.tensor([100, 100]),
        }
    provider = GaussianRenderObservationProvider(
        {"uses_source_mapping_rgb": False, "queries": queries},
        query_names=names,
    )
    state = {
        "anchor_ids": torch.tensor([0]),
        "anchor_xyz": point[None],
        "anchor_features": torch.tensor([[1.0, 0.0]]),
        "v6_mapping_query_names": names,
        "v6_mapping_query_bins": torch.arange(4),
        "projective_anchor_construction": {
            "final_xyz_source": "fixed_camera_robust_ray_triangulation"
        },
        "projective_anchor_observations": {
            "observation_offsets": torch.tensor([0, 4]),
            "query_indices": torch.arange(4),
            "keypoint_indices": torch.zeros(4, dtype=torch.long),
        },
    }
    update = LeaveOneQueryOutProjectiveMap(state, provider).query_update(0)
    assert update["valid"].tolist() == [True]
    assert torch.allclose(update["anchor_xyz"][0], point, atol=1e-4, rtol=0)

    neighborhood = LeaveOneQueryOutProjectiveMap(state, provider).query_update(
        0, excluded_queries=[0, 1]
    )
    assert neighborhood["excluded_queries"].tolist() == [0, 1]
    assert neighborhood["valid"].tolist() == [False]

    purged = LeaveOneQueryOutProjectiveMap(
        state, provider, affected_anchor_policy="purge"
    ).query_update(0, excluded_queries=[0, 1])
    assert purged["anchor_rows"].tolist() == [0]
    assert purged["valid"].tolist() == [False]
    assert purged["contract"]["query_descriptor_loo"] is True
    assert purged["contract"]["query_geometry_loo"] is True
    assert purged["contract"]["affected_anchor_policy"] == "purge"
