import torch

from evidence.tracks import fuse_track_descriptors
from scripts.materialize_rendered_track_appearance_ensemble import (
    _camera_response,
    _keypoint_validity,
    robust_appearance_fusion,
)


def test_appearance_fusion_rejects_one_descriptor_outlier():
    good = torch.tensor([1.0, 0.0, 0.0])
    descriptors = torch.stack(
        [good, good, good, good, good, good, torch.tensor([0.0, 1.0, 0.0])]
    )[:, None]
    fused, dispersion, reliability = robust_appearance_fusion(descriptors)
    torch.testing.assert_close(fused[0], good)
    assert float(dispersion[0]) == 0.0
    assert float(reliability[0]) == 1.0


def test_camera_response_is_deterministic_and_bounded():
    image = torch.tensor([[[0.2]], [[0.5]], [[0.9]]])
    recipe = {
        "exposure": 1.2,
        "gamma": 0.85,
        "white_balance": (1.08, 1.0, 0.92),
        "contrast": 1.05,
    }
    first = _camera_response(image, recipe)
    second = _camera_response(image, recipe)
    torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)
    assert bool(((first >= 0.0) & (first <= 1.0)).all())


def test_alpha_validity_erodes_render_boundary():
    alpha = torch.ones((9, 9))
    alpha[:, :2] = 0.0
    keypoints = torch.tensor([[1.0, 4.0], [2.0, 4.0], [4.0, 4.0]])
    valid, sampled = _keypoint_validity(
        alpha, keypoints, alpha_minimum=0.05, erosion_radius=1
    )
    assert valid.tolist() == [False, False, True]
    assert sampled.tolist() == [0.0, 1.0, 1.0]


def test_track_fusion_uses_valid_rows_and_appearance_reliability():
    payload = {
        "query_names": ["q0", "q1", "q2"],
        "query_bins": torch.tensor([0, 1, 2]),
        "tracks": {
            "track_index": torch.zeros(3, dtype=torch.long),
            "query_index": torch.arange(3),
            "keypoint_index": torch.zeros(3, dtype=torch.long),
            "confidence": torch.ones(3),
        },
    }
    cache = {
        "queries": {
            "q0": {
                "native_descriptors": torch.tensor([[1.0, 0.0]]),
                "native_valid_keypoint_mask": torch.tensor([True]),
                "native_appearance_reliability": torch.tensor([1.0]),
            },
            "q1": {
                "native_descriptors": torch.tensor([[0.0, 1.0]]),
                "native_valid_keypoint_mask": torch.tensor([False]),
                "native_appearance_reliability": torch.tensor([1.0]),
            },
            "q2": {
                "native_descriptors": torch.tensor([[1.0, 0.0]]),
                "native_valid_keypoint_mask": torch.tensor([True]),
                "native_appearance_reliability": torch.tensor([0.5]),
            },
        }
    }
    fused = fuse_track_descriptors(
        payload=payload,
        query_cache=cache,
        track_indices=torch.tensor([0]),
        trim_fraction=0.0,
    )
    torch.testing.assert_close(fused[0], torch.tensor([1.0, 0.0]))
