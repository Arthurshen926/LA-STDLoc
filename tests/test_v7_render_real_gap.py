import numpy as np
import torch

from evidence.v7_render_certificate import CertificateThresholds
from evidence.v7_render_real_gap import (
    feather_support,
    mutual_spatial_pairs,
    projected_match_correctness,
    sample_pixel_mask,
    shared_support_mask,
    signed_translation_bias,
    summarize_pose_rows,
)


def test_shared_support_and_sampling_exclude_border_and_holes():
    rgb = torch.rand(3, 32, 40)
    alpha = torch.ones(32, 40)
    depth = torch.ones(32, 40) * 5
    alpha[10:15, 10:15] = 0
    mask = shared_support_mask(
        rgb,
        alpha,
        depth,
        thresholds=CertificateThresholds(
            border_fraction=0.05,
            rgb_structure_support_threshold=0.01,
        ),
    )
    rows = sample_pixel_mask(mask, torch.tensor([[0.0, 0.0], [12.0, 12.0], [25.0, 20.0]]))
    assert rows.tolist() == [False, False, True]


def test_feather_support_is_bounded_and_soft_at_boundary():
    mask = torch.zeros(21, 21, dtype=torch.bool)
    mask[6:15, 6:15] = True
    soft = feather_support(mask, reference_px=108)
    assert soft.shape == mask.shape
    assert bool(((soft >= 0) & (soft <= 1)).all())
    assert 0 < float(soft[5, 10]) < 1
    assert float(soft[10, 10]) > float(soft[5, 10])


def test_mutual_spatial_pairs_reject_nonmutual_and_far_rows():
    left = torch.tensor([[0.0, 0.0], [0.2, 0.0], [10.0, 10.0]])
    right = torch.tensor([[0.1, 0.0], [20.0, 20.0]])
    lrow, rrow, distance = mutual_spatial_pairs(
        left, right, maximum_distance_px=1.0
    )
    assert lrow.tolist() == [0]
    assert rrow.tolist() == [0]
    assert torch.allclose(distance, torch.tensor([0.1]))


def test_projected_correctness_uses_anchor_geometry_and_pixel_center():
    intrinsic = torch.tensor([[10.0, 0.0, 5.0], [0.0, 10.0, 5.0], [0.0, 0.0, 1.0]])
    xyz = torch.tensor([[0.0, 0.0, 2.0], [2.0, 0.0, 2.0]])
    keypoints = torch.tensor([[4.5, 4.5], [4.5, 4.5]])
    correct = projected_match_correctness(
        keypoints,
        xyz,
        torch.eye(4),
        intrinsic,
        maximum_reprojection_px=1.0,
    )
    assert correct.tolist() == [True, False]


def test_pose_summary_and_signed_bias_are_directional():
    gt = np.eye(4)
    predicted = np.eye(4)
    predicted[0, 3] = -0.01
    bias = signed_translation_bias(predicted, gt)
    assert np.allclose(bias, [1.0, 0.0, 0.0])
    summary = summarize_pose_rows(
        [
            {"translation_error_cm": 1.0, "rotation_error_deg": 0.1},
            {"translation_error_cm": 9.0, "rotation_error_deg": 0.2},
        ]
    )
    assert summary["median_translation_cm"] == 5.0
    assert summary["recall_5cm_5deg_percent"] == 50.0
