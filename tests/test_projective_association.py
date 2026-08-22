from __future__ import annotations

import pytest
import torch

from evidence.observation_provider import GaussianRenderObservationProvider
from evidence.projective_association import _component_statistics, _render_valid_rows


def _provider(valid: bool = True) -> GaussianRenderObservationProvider:
    queries = {}
    for index in range(3):
        queries[f"view{index}"] = {
            "native_keypoints": torch.tensor([[2.0, 2.0]]),
            "native_descriptors": torch.tensor([[1.0, 0.0]]),
            "native_scores": torch.tensor([0.8]),
            "native_K": torch.eye(3),
            "pose_w2c": torch.eye(4),
            "native_input_hw": [5, 5],
            "native_valid_keypoint_mask": torch.tensor([valid]),
        }
    return GaussianRenderObservationProvider(
        {"uses_source_mapping_rgb": False, "queries": queries}
    )


def test_association_rejects_invalid_cached_observation() -> None:
    with pytest.raises(ValueError):
        _render_valid_rows(_provider(False).build_view(0))


def test_identity_reliability_is_continuous_not_a_track_type() -> None:
    tracks = {
        "track_index": torch.tensor([0, 0, 0]),
        "query_index": torch.tensor([0, 1, 2]),
        "keypoint_index": torch.tensor([0, 0, 0]),
        "confidence": torch.tensor([0.8, 0.9, 1.0]),
        "track_level": torch.tensor([2]),
    }
    stats = _component_statistics(
        tracks,
        [torch.tensor([[1.0, 0.0]]) for _ in range(3)],
        torch.tensor([0, 1, 2]),
        1,
    )
    assert 0.0 < float(stats["identity_reliability"][0]) <= 1.0
    assert stats["distinct_camera_family_count"].tolist() == [3]
