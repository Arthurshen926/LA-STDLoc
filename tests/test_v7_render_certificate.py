import copy

import pytest
import torch

from common.v7_contracts import tensor_tree_equal
from evidence.v7_render_certificate import (
    CertificateThresholds,
    certify_v7_render,
    extreme_distortion_row_mask,
    rgb_structure_support_mask,
    render_quality_pixel_masks,
)


def _normal(size: int = 40):
    yy, xx = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij")
    texture = 0.3 + 0.4 * (((xx // 2 + yy // 2) % 2).float())
    rgb = texture[None].repeat(3, 1, 1)
    alpha = torch.ones(size, size)
    depth = torch.full((size, size), 4.0)
    keypoints = torch.tensor([[10.0, 10.0], [20.0, 20.0], [30.0, 30.0]])
    return dict(
        rgb=rgb, alpha=alpha, depth=depth, keypoints=keypoints,
        nearest_mapping_distance_m=0.2, median_adjacent_baseline_m=0.5,
        source_family_support=2, expected_median_depth_m=4.0,
    )


def test_normal_render_is_accepted_after_full_rgb_detection() -> None:
    result = certify_v7_render(**_normal())
    assert result["decision"] == "ACCEPT"
    assert result["can_drive_map_update"] is True
    assert result["detector_input"] == "complete_unmasked_rgb"
    assert result["row_valid"].tolist() == [True, True, True]


@pytest.mark.parametrize("failure", ["alpha_hole", "black", "nan_depth", "negative_depth", "curtain", "outside"])
def test_known_extreme_artifacts_are_rejected(failure: str) -> None:
    values = _normal()
    if failure == "alpha_hole":
        values["alpha"].zero_()
    elif failure == "black":
        values["rgb"].zero_()
        values["alpha"].zero_()
    elif failure == "nan_depth":
        values["depth"].fill_(float("nan"))
    elif failure == "negative_depth":
        values["depth"].fill_(-1.0)
    elif failure == "curtain":
        values["depth"].fill_(0.1)
    elif failure == "outside":
        values["nearest_mapping_distance_m"] = 3.0
    result = certify_v7_render(**values)
    assert result["decision"] == "REJECT"
    assert result["can_drive_map_update"] is False


def test_local_artifact_marks_rows_without_rejecting_whole_render() -> None:
    values = _normal()
    values["artifact_row_mask"] = torch.tensor([False, True, False])
    result = certify_v7_render(**values)
    assert result["decision"] == "ACCEPT"
    assert result["can_drive_map_update"] is True
    assert result["row_valid"].tolist() == [True, False, True]
    assert result["row_uncertain"].tolist() == [False, True, False]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_missing_artifact_raster_preserves_cuda_row_device() -> None:
    values = {
        key: value.cuda() if isinstance(value, torch.Tensor) else value
        for key, value in _normal().items()
    }
    result = certify_v7_render(**values)
    assert result["row_valid"].is_cuda


def test_reject_and_uncertain_certification_cannot_mutate_map_state() -> None:
    state = {"anchor_ids": torch.arange(3), "anchor_features": torch.eye(3)}
    frozen = copy.deepcopy(state)
    rejected = _normal()
    rejected["alpha"].zero_()
    uncertain = _normal()
    uncertain["source_family_support"] = 0
    assert certify_v7_render(**rejected)["can_drive_map_update"] is False
    assert certify_v7_render(**uncertain)["can_drive_map_update"] is False
    assert tensor_tree_equal(state, frozen)


def test_distortion_audit_only_flags_extreme_post_detector_rows() -> None:
    distortion = torch.ones(40, 40)
    distortion[20, 20] = 100.0
    keypoints = torch.tensor([[10.0, 10.0], [20.0, 20.0], [30.0, 30.0]])
    assert extreme_distortion_row_mask(distortion, keypoints).tolist() == [
        False,
        True,
        False,
    ]


def test_distortion_audit_does_not_call_a_broad_secondary_mode_extreme() -> None:
    distortion = torch.ones(40, 40)
    distortion[20:, :] = 2.0
    distortion[20, 20] = 100.0
    keypoints = torch.tensor([[10.0, 10.0], [30.0, 30.0], [20.0, 20.0]])
    assert extreme_distortion_row_mask(distortion, keypoints).tolist() == [
        False,
        False,
        True,
    ]


def test_rgb_structure_support_is_post_detector_and_local() -> None:
    size = 80
    yy, xx = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij")
    textured = 0.3 + 0.4 * (((xx // 2 + yy // 2) % 2).float())
    textured[:, size // 2 :] = 0.5
    rgb = textured[None].repeat(3, 1, 1)
    support, score = rgb_structure_support_mask(rgb)
    assert support.shape == (size, size)
    assert score.shape == (size, size)
    assert bool(support[20, 20])
    assert not bool(support[20, 70])

    values = _normal(size)
    values["rgb"] = rgb
    values["keypoints"] = torch.tensor([[20.0, 20.0], [70.0, 20.0]])
    values["thresholds"] = CertificateThresholds(
        minimum_valid_keypoint_fraction=0.40,
    )
    result = certify_v7_render(**values)
    assert result["row_valid"].tolist() == [True, False]
    assert result["row_reasons"]["low_rgb_structure_support"].tolist() == [
        False,
        True,
    ]


def test_pixel_quality_export_separates_invalid_from_uncertain() -> None:
    values = _normal(80)
    values["rgb"][:, 30:50, 30:50] = 0.5
    masks = render_quality_pixel_masks(
        rgb=values["rgb"], alpha=values["alpha"], depth=values["depth"]
    )
    assert bool(masks["invalid"][40, 40])
    assert bool(masks["uncertain"][0, 0])
    assert not bool(masks["invalid"][0, 0])
    assert not bool((masks["valid"] & masks["invalid"]).any())
