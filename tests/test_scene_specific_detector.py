import pytest
import torch

from evidence.scene_detector_supervision import (
    IGNORE_LABEL,
    NEGATIVE_LABEL,
    POSITIVE_LABEL,
    build_tri_state_heatmap,
    build_feedback_match_heatmap,
    build_pose_contribution_weights,
    project_visible_clean_anchors,
    spatially_balance_points,
)
from features.scene_specific_detector import (
    SceneSpecificDetector,
    detector_metrics,
    fuse_scene_reliability,
    protected_scene_candidate_indices,
    mean_candidate_reliability,
    tri_state_detector_loss,
)


def test_detector_head_preserves_requested_image_shape() -> None:
    head = SceneSpecificDetector(feature_dim=8, hidden_dim=4)
    output = head(torch.randn(2, 8, 5, 7), output_hw=(40, 56))
    assert output.shape == (2, 40, 56)


def test_tri_state_loss_ignores_unlabelled_pixels() -> None:
    logits = torch.tensor([[0.0, 0.0, 100.0]], requires_grad=True)
    labels = torch.tensor([[POSITIVE_LABEL, NEGATIVE_LABEL, IGNORE_LABEL]])
    loss = tri_state_detector_loss(logits, labels, positive_weight=1.0)
    loss.backward()
    assert logits.grad[0, 2] == 0
    assert logits.grad[0, 0] < 0
    assert logits.grad[0, 1] > 0


def test_projected_positive_requires_depth_and_v2_support() -> None:
    xyz = torch.tensor([[0.0, 0.0, 2.0], [1.0, 0.0, 2.0], [0.0, 0.0, -1.0]])
    K = torch.tensor([[10.0, 0.0, 10.0], [0.0, 10.0, 10.0], [0.0, 0.0, 1.0]])
    depth = torch.full((20, 20), 2.0)
    support = torch.ones(20, 20, dtype=torch.bool)
    support[10, 15] = False
    uv, rows = project_visible_clean_anchors(
        anchor_xyz=xyz,
        clean_anchor_mask=torch.ones(3, dtype=torch.bool),
        intrinsic=K,
        pose_w2c=torch.eye(4),
        rendered_depth=depth,
        valid_pixel_mask=support,
    )
    assert rows.tolist() == [0]
    assert torch.allclose(uv, torch.tensor([[10.0, 10.0]]))


def test_heatmap_does_not_turn_uncovered_content_into_negative() -> None:
    invalid = torch.zeros(32, 32, dtype=torch.bool)
    invalid[:8, :8] = True
    labels = build_tri_state_heatmap(
        image_hw=(32, 32),
        positive_uv=torch.tensor([[20.0, 20.0]]),
        invalid_pixel_mask=invalid,
        output_stride=8,
        positive_radius_px=5.0,
    )
    assert labels.shape == (4, 4)
    assert labels[0, 0] == NEGATIVE_LABEL
    assert labels[2, 2] == POSITIVE_LABEL
    assert labels[0, 3] == IGNORE_LABEL


def test_uncertain_pixels_override_positive_and_negative() -> None:
    invalid = torch.ones(16, 16, dtype=torch.bool)
    uncertain = torch.zeros_like(invalid)
    uncertain[8:, 8:] = True
    labels = build_tri_state_heatmap(
        image_hw=(16, 16),
        positive_uv=torch.tensor([[12.0, 12.0]]),
        invalid_pixel_mask=invalid,
        uncertain_pixel_mask=uncertain,
        output_stride=8,
        positive_radius_px=5.0,
    )
    assert labels[1, 1] == IGNORE_LABEL


def test_spatial_balancing_keeps_best_target_per_cell() -> None:
    uv = torch.tensor([[2.0, 2.0], [3.0, 3.0], [18.0, 18.0]])
    selected = spatially_balance_points(
        uv, torch.tensor([0.1, 0.9, 0.5]), image_hw=(20, 20), grid_hw=(2, 2)
    )
    assert selected.tolist() == [1, 2]


def test_detector_metric_reports_positive_negative_separation() -> None:
    metrics = detector_metrics(
        torch.tensor([[3.0, -3.0, 0.0]]),
        torch.tensor([[POSITIVE_LABEL, NEGATIVE_LABEL, IGNORE_LABEL]]),
    )
    assert metrics["separation"] > 0.8


def test_invalid_detector_labels_are_rejected() -> None:
    with pytest.raises(ValueError, match="-1, 0, or 1"):
        tri_state_detector_loss(torch.zeros(1), torch.tensor([2]))


def test_scene_reliability_preserves_native_zero_response() -> None:
    native = torch.tensor([[0.0, 0.2, 0.4]])
    fused = fuse_scene_reliability(native, torch.tensor([[100.0, 0.0, -100.0]]))
    assert fused[0, 0] == 0
    assert torch.allclose(fused[0, 1], torch.tensor(0.1))
    assert fused[0, 2] < 1e-10


def test_feedback_heatmap_labels_only_actionable_detector_outcomes() -> None:
    labels = build_feedback_match_heatmap(
        image_hw=(32, 32),
        keypoints=torch.tensor([[4.0, 4.0], [12.0, 4.0], [20.0, 4.0], [28.0, 4.0]]),
        reprojection_error_px=torch.tensor([2.0, 20.0, 8.0, 1.0]),
        row_valid=torch.tensor([True, True, True, False]),
        row_uncertain=torch.tensor([False, False, False, False]),
    )
    assert labels[0].tolist() == [POSITIVE_LABEL, NEGATIVE_LABEL, IGNORE_LABEL, NEGATIVE_LABEL]


def test_feedback_heatmap_positive_wins_same_cell_and_uncertain_is_ignored() -> None:
    labels = build_feedback_match_heatmap(
        image_hw=(16, 16),
        keypoints=torch.tensor([[1.0, 1.0], [6.0, 6.0], [12.0, 4.0]]),
        reprojection_error_px=torch.tensor([30.0, 1.0, 30.0]),
        row_valid=torch.tensor([True, True, True]),
        row_uncertain=torch.tensor([False, False, True]),
    )
    assert labels[0, 0] == POSITIVE_LABEL
    assert labels[0, 1] == IGNORE_LABEL


def test_pose_contribution_weights_are_bounded_analytic_not_loo() -> None:
    labels = torch.tensor([[POSITIVE_LABEL, POSITIVE_LABEL], [NEGATIVE_LABEL, IGNORE_LABEL]])
    weights = build_pose_contribution_weights(
        labels=labels,
        keypoints=torch.tensor([[2.0, 2.0], [10.0, 2.0], [2.0, 10.0]]),
        image_hw=(16, 16),
        reprojection_error_px=torch.tensor([1.0, 2.0, 30.0]),
        camera_depth=torch.tensor([2.0, 4.0, 3.0]),
        match_margin=torch.tensor([0.01, 0.02, 0.20]),
    )
    assert weights.shape == labels.shape
    assert bool((weights[labels >= 0] > 0).all())
    assert weights[1, 1] == 0


def test_weighted_detector_loss_requires_aligned_non_negative_weights() -> None:
    logits = torch.zeros(2)
    labels = torch.tensor([POSITIVE_LABEL, NEGATIVE_LABEL])
    weighted = tri_state_detector_loss(
        logits, labels, positive_weight=1.0, sample_weight=torch.tensor([2.0, 1.0])
    )
    assert torch.isfinite(weighted)
    with pytest.raises(ValueError, match="non-negative"):
        tri_state_detector_loss(logits, labels, sample_weight=torch.tensor([1.0, -1.0]))


def test_scene_reliability_strength_zero_is_native_and_one_is_original_rule() -> None:
    native = torch.tensor([[0.2, 0.4]])
    logits = torch.tensor([[2.0, -2.0]])
    assert torch.equal(fuse_scene_reliability(native, logits, strength=0.0), native)
    assert torch.allclose(
        fuse_scene_reliability(native, logits, strength=1.0), native * torch.sigmoid(logits)
    )


def test_protected_scene_candidates_bound_the_replacement_budget() -> None:
    indices = protected_scene_candidate_indices(
        keypoints=torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]),
        native_scores=torch.tensor([0.9, 0.8, 0.7, 0.6]),
        detector_logits=torch.tensor([[0.0, 0.0, -10.0, 10.0]]),
        output_count=3,
        protected_native_count=2,
    )
    assert indices.tolist() == [0, 1, 3]


def test_mean_candidate_reliability_samples_only_native_points() -> None:
    value = mean_candidate_reliability(
        torch.tensor([[0.0, 0.0], [2.0, 0.0]]),
        torch.tensor([[0.0, -10.0, 2.0]]),
    )
    assert torch.allclose(value, (torch.sigmoid(torch.tensor(0.0)) + torch.sigmoid(torch.tensor(2.0))) / 2)
