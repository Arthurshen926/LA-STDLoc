import copy

import pytest
import torch

from common.v7_contracts import tensor_tree_equal
from evidence.v7_render_certificate import (
    certify_v7_render,
    extreme_distortion_row_mask,
)


def _normal(size: int = 40):
    rgb = torch.full((3, size, size), 0.5)
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
