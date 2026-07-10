import torch


def _camera():
    K = torch.tensor(
        [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )
    return K, torch.eye(4, dtype=torch.float32)


def _point_at_pixel(x, y, depth=10.0):
    return [x * depth / 10.0, y * depth / 10.0, depth]


def test_sparse_candidate_teacher_exposes_false_positive_and_recovers_false_negative():
    from localization_training.sparse_candidate_teacher import build_sparse_candidate_batch

    feature_map = torch.zeros(2, 5, 5)
    feature_map[:, 1, 1] = torch.tensor([1.0, 0.0])
    heatmap = torch.zeros(1, 5, 5)
    heatmap[0, 1, 1] = 0.9
    landmark_xyz = torch.tensor(
        [
            _point_at_pixel(1.5, 1.5),
            _point_at_pixel(3.5, 3.5),
        ]
    )
    # The query descriptor prefers landmark 1, while GT projection says landmark 0.
    landmark_features = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    K, pose = _camera()

    batch = build_sparse_candidate_batch(
        feature_map,
        heatmap,
        landmark_features,
        landmark_xyz,
        K,
        pose,
        detect_num=1,
        nms_radius=0,
        positive_radius_px=0.25,
        hard_negatives=1,
        match_margin=0.0,
    )

    assert batch.diagnostics["predicted_pair_count"] == 1
    assert batch.diagnostics["predicted_correct_count"] == 0
    assert batch.diagnostics["keypoint_gt_coverage"] == 1.0
    assert batch.diagnostics["false_negative_count"] == 1
    assert batch.diagnostics["recovered_positive_count"] == 1
    assert batch.pair_landmark_idx.tolist() == [1, 0]
    assert batch.pair_labels.tolist() == [0.0, 1.0]
    assert batch.hard_negative_logits.numel() == 1
    assert batch.assignment_positive_similarity.shape == (1,)
    assert batch.assignment_negative_similarity.shape == (1, 1)
    assert batch.assignment_negative_mask.tolist() == [[True]]
    assert batch.diagnostics["assignment_top1_accuracy"] == 0.0


def test_sparse_candidate_topk_exposes_and_recovers_second_rank_positive():
    from localization_training.sparse_candidate_teacher import (
        build_sparse_candidate_batch,
        sparse_candidate_losses,
    )

    feature_map = torch.zeros(2, 5, 5)
    feature_map[:, 1, 1] = torch.tensor([1.0, 0.0])
    heatmap = torch.zeros(1, 5, 5)
    heatmap[0, 1, 1] = 0.9
    landmark_xyz = torch.tensor(
        [
            _point_at_pixel(3.5, 3.5),
            _point_at_pixel(1.5, 1.5),
        ]
    )
    landmark_features = torch.tensor(
        [[1.0, 0.0], [0.8, 0.6]], requires_grad=True
    )
    K, pose = _camera()

    batch = build_sparse_candidate_batch(
        feature_map,
        heatmap,
        landmark_features,
        landmark_xyz,
        K,
        pose,
        detect_num=1,
        nms_radius=0,
        match_mode="topk",
        match_topk=2,
        positive_radius_px=0.25,
        negative_radius_px=1.0,
        hard_negatives=1,
    )

    assert batch.pair_scorer_landmark_idx.tolist() == [0, 1]
    assert batch.pair_scorer_labels.tolist() == [0.0, 1.0]
    assert batch.diagnostics["top1_correct_keypoint_count"] == 0
    assert batch.diagnostics["predicted_correct_keypoint_count"] == 1
    assert batch.diagnostics["predicted_correct_reprojection_error_mean"] == 0.0
    assert batch.diagnostics["predicted_correct_reprojection_error_p95"] == 0.0
    assert batch.diagnostics["topk_rescued_keypoint_count"] == 1
    assert batch.diagnostics["false_negative_count"] == 0
    losses = sparse_candidate_losses(batch)
    losses.matcher_assignment.backward()
    assert torch.isfinite(losses.matcher_assignment)
    assert landmark_features.grad is not None


def test_sparse_candidate_offset_target_reduces_endpoint_error_and_backpropagates():
    from localization_training.sparse_candidate_teacher import (
        build_sparse_candidate_batch,
        sparse_candidate_losses,
    )

    feature_map = torch.zeros(2, 5, 5)
    feature_map[:, 1, 1] = torch.tensor([1.0, 0.0])
    heatmap = torch.zeros(1, 5, 5)
    heatmap[0, 1, 1] = 0.9
    offset_map = torch.zeros(2, 5, 5)
    offset_map[0, 1, 1] = 0.25
    offset_map.requires_grad_()
    landmark_xyz = torch.tensor([_point_at_pixel(2.0, 1.5)])
    landmark_features = torch.tensor([[1.0, 0.0]])
    K, pose = _camera()

    batch = build_sparse_candidate_batch(
        feature_map,
        heatmap,
        landmark_features,
        landmark_xyz,
        K,
        pose,
        detect_num=1,
        nms_radius=0,
        positive_radius_px=1.0,
        keypoint_offset_map=offset_map,
    )
    losses = sparse_candidate_losses(batch)
    losses.detector_offset.backward()

    assert torch.allclose(batch.detector_offset_targets, torch.tensor([[0.5, 0.0]]))
    assert batch.diagnostics["detector_offset_target_norm_mean"] == 0.5
    assert batch.diagnostics["detector_offset_endpoint_error_mean"] == 0.25
    assert offset_map.grad is not None
    assert offset_map.grad[0, 1, 1] < 0


def test_sparse_candidate_offset_can_target_actual_top1_match():
    from localization_training.sparse_candidate_teacher import build_sparse_candidate_batch

    feature_map = torch.zeros(2, 5, 5)
    feature_map[:, 1, 1] = torch.tensor([1.0, 0.0])
    heatmap = torch.zeros(1, 5, 5)
    heatmap[0, 1, 1] = 0.9
    landmarks = torch.tensor(
        [
            _point_at_pixel(1.6, 1.5),
            _point_at_pixel(2.0, 1.5),
        ]
    )
    landmark_features = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    K, pose = _camera()

    nearest = build_sparse_candidate_batch(
        feature_map,
        heatmap,
        landmark_features,
        landmarks,
        K,
        pose,
        detect_num=1,
        nms_radius=0,
        positive_radius_px=1.0,
        keypoint_offset_map=torch.zeros(2, 5, 5),
        detector_offset_target_source="geometric_nearest",
    )
    matched = build_sparse_candidate_batch(
        feature_map,
        heatmap,
        landmark_features,
        landmarks,
        K,
        pose,
        detect_num=1,
        nms_radius=0,
        positive_radius_px=1.0,
        keypoint_offset_map=torch.zeros(2, 5, 5),
        detector_offset_target_source="matched_top1",
    )

    assert torch.allclose(nearest.detector_offset_targets, torch.tensor([[0.1, 0.0]]))
    assert torch.allclose(matched.detector_offset_targets, torch.tensor([[0.5, 0.0]]))


def test_sparse_candidate_rejects_invalid_offset_map_shape():
    import pytest

    from localization_training.sparse_candidate_teacher import build_sparse_candidate_batch

    feature_map = torch.zeros(2, 5, 5)
    heatmap = torch.ones(1, 5, 5)
    K, pose = _camera()
    with pytest.raises(ValueError, match="2xHxW"):
        build_sparse_candidate_batch(
            feature_map,
            heatmap,
            torch.tensor([[1.0, 0.0]]),
            torch.tensor([_point_at_pixel(1.5, 1.5)]),
            K,
            pose,
            detect_num=1,
            keypoint_offset_map=torch.zeros(1, 5, 5),
        )


def test_sparse_candidate_losses_backpropagate_to_landmark_feature_and_detector_score():
    from localization_training.sparse_candidate_teacher import (
        build_sparse_candidate_batch,
        sparse_candidate_losses,
    )

    feature_map = torch.zeros(2, 5, 5)
    feature_map[:, 1, 1] = torch.tensor([1.0, 0.0])
    heatmap = torch.full((1, 5, 5), 0.01, requires_grad=True)
    with torch.no_grad():
        heatmap[0, 1, 1] = 0.9
    landmark_xyz = torch.tensor(
        [
            _point_at_pixel(1.5, 1.5),
            _point_at_pixel(3.5, 3.5),
        ]
    )
    landmark_features = torch.tensor(
        [[0.2, 0.8], [0.9, 0.1]], requires_grad=True
    )
    K, pose = _camera()
    batch = build_sparse_candidate_batch(
        feature_map,
        heatmap,
        landmark_features,
        landmark_xyz,
        K,
        pose,
        detect_num=1,
        nms_radius=0,
        positive_radius_px=0.25,
        hard_negatives=1,
        match_margin=0.3,
    )
    losses = sparse_candidate_losses(batch)
    total = losses.pair + losses.hard_negative + losses.assignment + losses.detector_match
    total.backward()

    assert landmark_features.grad is not None
    assert torch.count_nonzero(landmark_features.grad).item() > 0
    assert heatmap.grad is not None
    assert heatmap.grad[0, 1, 1].abs().item() > 0


def test_row_assignment_loss_directly_penalizes_wrong_query_ranking():
    from localization_training.sparse_candidate_teacher import _row_assignment_loss

    mask = torch.tensor([[True, True]])
    well_ranked = _row_assignment_loss(
        torch.tensor([0.9]),
        torch.tensor([[0.2, 0.1]]),
        mask,
        temperature=0.1,
        margin=0.05,
    )
    misranked = _row_assignment_loss(
        torch.tensor([0.1]),
        torch.tensor([[0.8, 0.7]]),
        mask,
        temperature=0.1,
        margin=0.05,
    )

    assert misranked > well_ranked


def test_sparse_candidate_geometry_and_coverage_losses_are_finite():
    from localization_training.sparse_candidate_teacher import (
        build_sparse_candidate_batch,
        sparse_candidate_losses,
    )

    pixels = [(1, 1), (6, 1), (1, 6), (6, 6), (3, 3), (4, 4)]
    channels = len(pixels)
    feature_map = torch.zeros(channels, 8, 8)
    heatmap = torch.zeros(1, 8, 8, requires_grad=True)
    landmark_features = torch.eye(channels, requires_grad=True)
    landmark_xyz = []
    for channel, (x, y) in enumerate(pixels):
        feature_map[channel, y, x] = 1.0
        with torch.no_grad():
            heatmap[0, y, x] = 0.9 - channel * 0.01
        landmark_xyz.append(_point_at_pixel(x + 0.5, y + 0.5, depth=8.0 + channel))
    K, pose = _camera()
    batch = build_sparse_candidate_batch(
        feature_map,
        heatmap,
        landmark_features,
        torch.tensor(landmark_xyz),
        K,
        pose,
        detect_num=len(pixels),
        nms_radius=0,
        positive_radius_px=0.25,
        hard_negatives=2,
        match_margin=0.3,
        grid_rows=2,
        grid_cols=2,
        depth_bins=3,
    )
    losses = sparse_candidate_losses(batch)

    assert batch.diagnostics["predicted_gt_precision"] == 1.0
    assert batch.diagnostics["positive_grid_occupancy"] == 4
    assert batch.diagnostics["positive_depth_occupancy"] == 3
    assert torch.isfinite(losses.geometry_set)
    assert torch.isfinite(losses.coverage)
    assert torch.isfinite(losses.matcher_translation_info)
    assert batch.diagnostics["geometry_condition"] > 0
    assert batch.diagnostics["matcher_translation_min_eig"] > 0


def test_sparse_candidate_teacher_mnn_matches_eval_semantics():
    from localization_training.sparse_frontend import match_score_matrix

    score = torch.tensor([[0.9, 0.8], [0.85, 0.1]])
    matches = match_score_matrix(score, mode="mnn", threshold=0.0)

    assert matches.keypoint_idx.tolist() == [0]
    assert matches.landmark_idx.tolist() == [0]


def test_calibrated_pair_threshold_preserves_recall_floor_and_rejects_negatives():
    from localization_training.sparse_candidate_teacher import calibrate_binary_threshold

    logits = torch.tensor([4.0, 3.0, 2.0, 1.0, 0.0, -1.0])
    labels = torch.tensor([1.0, 0.0, 1.0, 1.0, 0.0, 0.0])
    calibrated = calibrate_binary_threshold(
        logits,
        labels,
        torch.ones(6, dtype=torch.bool),
        min_recall=2.0 / 3.0,
    )

    accepted = logits > calibrated["threshold"]
    assert calibrated["recall"] >= 2.0 / 3.0
    assert accepted.sum().item() == calibrated["accepted_count"]
    assert labels[accepted].sum().item() == calibrated["correct_count"]
    assert calibrated["accepted_count"] < logits.numel()


def test_sparse_candidate_teacher_ignores_two_to_six_pixel_ambiguity_band():
    from localization_training.sparse_candidate_teacher import (
        build_sparse_candidate_batch,
        sparse_candidate_losses,
    )

    feature_map = torch.zeros(2, 8, 8)
    feature_map[:, 1, 1] = torch.tensor([1.0, 0.0])
    heatmap = torch.zeros(1, 8, 8)
    heatmap[0, 1, 1] = 0.9
    landmark_xyz = torch.tensor([_point_at_pixel(5.5, 1.5)])
    landmark_features = torch.tensor([[1.0, 0.0]], requires_grad=True)
    K, pose = _camera()

    batch = build_sparse_candidate_batch(
        feature_map,
        heatmap,
        landmark_features,
        landmark_xyz,
        K,
        pose,
        detect_num=1,
        nms_radius=0,
        positive_radius_px=2.0,
        negative_radius_px=6.0,
        hard_negatives=1,
    )
    losses = sparse_candidate_losses(batch)

    assert batch.diagnostics["predicted_ambiguous_count"] == 1
    assert batch.pair_valid_mask.tolist() == [False]
    assert batch.hard_negative_logits.numel() == 0
    assert losses.pair.item() == 0.0


def test_sparse_candidate_teacher_builds_multi_positive_geometric_set():
    from localization_training.sparse_candidate_teacher import (
        build_sparse_candidate_batch,
        sparse_candidate_losses,
    )

    feature_map = torch.zeros(2, 10, 10)
    feature_map[:, 1, 1] = torch.tensor([1.0, 0.0])
    heatmap = torch.zeros(1, 10, 10)
    heatmap[0, 1, 1] = 0.9
    landmark_xyz = torch.tensor(
        [
            _point_at_pixel(1.5, 1.5),
            _point_at_pixel(1.7, 1.5),
            _point_at_pixel(8.5, 8.5),
        ]
    )
    landmark_features = torch.tensor(
        [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]], requires_grad=True
    )
    K, pose = _camera()

    batch = build_sparse_candidate_batch(
        feature_map,
        heatmap,
        landmark_features,
        landmark_xyz,
        K,
        pose,
        detect_num=1,
        nms_radius=0,
        positive_radius_px=0.3,
        negative_radius_px=2.0,
        max_positives=4,
        hard_negatives=1,
    )
    losses = sparse_candidate_losses(batch)

    assert batch.multi_positive_mask.tolist() == [[True, True, False, False]]
    assert batch.diagnostics["multi_positive_count"] == 2
    assert batch.diagnostics["multi_positive_per_positive_row"] == 2.0
    assert torch.isfinite(losses.dustbin_assignment)


def test_multi_positive_dustbin_loss_trains_explicit_unmatched_score():
    from localization_training.sparse_candidate_teacher import _multi_positive_dustbin_loss

    dustbin = torch.tensor(0.5, requires_grad=True)
    loss = _multi_positive_dustbin_loss(
        torch.tensor([[0.9, 0.0], [0.0, 0.0]], requires_grad=True),
        torch.tensor([[True, False], [False, False]]),
        torch.tensor([[0.2], [0.8]], requires_grad=True),
        torch.tensor([[True], [True]]),
        dustbin,
        temperature=0.1,
        margin=0.05,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert dustbin.grad is not None
    assert dustbin.grad.abs().item() > 0


def test_sparse_candidate_pair_scorer_loss_backpropagates_to_scorer():
    from localization_training.pair_scorer import SparsePairScorer
    from localization_training.sparse_candidate_teacher import (
        build_sparse_candidate_batch,
        sparse_candidate_losses,
    )

    feature_map = torch.zeros(2, 8, 8)
    feature_map[:, 1, 1] = torch.tensor([1.0, 0.0])
    feature_map[:, 6, 6] = torch.tensor([0.0, 1.0])
    heatmap = torch.zeros(1, 8, 8)
    heatmap[0, 1, 1] = 0.9
    heatmap[0, 6, 6] = 0.8
    landmarks = torch.tensor(
        [_point_at_pixel(1.5, 1.5), _point_at_pixel(3.5, 3.5)]
    )
    landmark_features = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    scorer = SparsePairScorer(hidden_dim=8)
    K, pose = _camera()

    batch = build_sparse_candidate_batch(
        feature_map,
        heatmap,
        landmark_features,
        landmarks,
        K,
        pose,
        detect_num=2,
        nms_radius=0,
        positive_radius_px=0.5,
        negative_radius_px=2.0,
        hard_negatives=1,
        pair_scorer=scorer,
    )
    losses = sparse_candidate_losses(batch)
    losses.pair_scorer.backward()

    assert batch.pair_scorer_logits.shape == (2,)
    assert torch.isfinite(losses.pair_scorer)
    assert any(parameter.grad is not None for parameter in scorer.parameters())


def test_pair_scorer_assignment_directly_ranks_positive_within_keypoint():
    from localization_training.sparse_candidate_teacher import (
        _grouped_multi_positive_assignment_loss,
    )

    labels = torch.tensor([1.0, 0.0, 0.0, 1.0])
    valid = torch.ones(4, dtype=torch.bool)
    groups = torch.tensor([0, 0, 1, 1])
    well_ranked = _grouped_multi_positive_assignment_loss(
        torch.tensor([3.0, 0.0, 0.0, 3.0]), labels, valid, groups
    )
    misranked = _grouped_multi_positive_assignment_loss(
        torch.tensor([0.0, 3.0, 3.0, 0.0]), labels, valid, groups
    )

    assert misranked > well_ranked


def test_reprojection_assignment_prefers_subpixel_positive():
    from localization_training.sparse_candidate_teacher import (
        _grouped_reprojection_assignment_loss,
    )

    labels = torch.tensor([1.0, 1.0, 0.0])
    errors = torch.tensor([0.2, 1.8, 8.0])
    valid = torch.ones(3, dtype=torch.bool)
    groups = torch.zeros(3, dtype=torch.long)
    precise_positive_ranked = _grouped_reprojection_assignment_loss(
        torch.tensor([3.0, 0.0, -1.0]),
        labels,
        errors,
        valid,
        groups,
        sigma_px=1.0,
    )
    coarse_positive_ranked = _grouped_reprojection_assignment_loss(
        torch.tensor([0.0, 3.0, -1.0]),
        labels,
        errors,
        valid,
        groups,
        sigma_px=1.0,
    )

    assert precise_positive_ranked < coarse_positive_ranked


def test_reprojection_assignment_backpropagates_and_ignores_ambiguity():
    from localization_training.sparse_candidate_teacher import (
        _grouped_reprojection_assignment_loss,
    )

    logits = torch.tensor([0.0, 20.0, 1.0], requires_grad=True)
    loss = _grouped_reprojection_assignment_loss(
        logits,
        torch.tensor([1.0, 0.0, 0.0]),
        torch.tensor([0.5, 4.0, 8.0]),
        torch.tensor([True, False, True]),
        torch.zeros(3, dtype=torch.long),
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert logits.grad is not None
    assert logits.grad[1] == 0
    assert logits.grad[0] < 0
    assert logits.grad[2] > 0


def test_translation_schur_loss_is_finite_and_backpropagates_to_weights():
    from localization_training.sparse_candidate_teacher import _translation_schur_loss

    torch.manual_seed(4)
    jacobian = torch.randn(12, 2, 6)
    weights = torch.full((12,), 0.5, requires_grad=True)
    loss, diagnostics = _translation_schur_loss(jacobian, weights)
    loss.backward()

    assert torch.isfinite(loss)
    assert weights.grad is not None
    assert torch.isfinite(weights.grad).all()
    assert diagnostics["translation_min_eig"] > 0


def test_binary_detector_targets_keep_correct_matches_at_probability_one():
    from localization_training.sparse_candidate_teacher import build_sparse_candidate_batch

    feature_map = torch.zeros(2, 5, 5)
    feature_map[:, 1, 1] = torch.tensor([1.0, 0.0])
    heatmap = torch.zeros(1, 5, 5)
    heatmap[0, 1, 1] = 0.9
    landmark_xyz = torch.tensor([_point_at_pixel(1.5, 1.5)])
    K, pose = _camera()
    batch = build_sparse_candidate_batch(
        feature_map,
        heatmap,
        torch.tensor([[1.0, 0.0]]),
        landmark_xyz,
        K,
        pose,
        detect_num=1,
        nms_radius=0,
        positive_radius_px=0.25,
        detector_target_source="predicted_correct",
        detector_binary_target=True,
    )

    assert batch.detector_targets.tolist() == [1.0]
    assert batch.detector_loss_weights[0] > 0
    assert batch.diagnostics["detector_positive_target_mean"] == 1.0
