import torch


def _camera():
    K = torch.tensor(
        [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )
    return K, torch.eye(4, dtype=torch.float32)


def _point_at_pixel(x, y, depth=10.0):
    return [x * depth / 10.0, y * depth / 10.0, depth]


def test_candidate_assignment_translation_fisher_weights_are_finite_and_matchability_aware():
    from localization_training.sparse_candidate_teacher import (
        _assignment_information_weights,
    )

    points = torch.tensor(
        [
            [-1.0, -0.5, 4.0],
            [1.0, -0.5, 4.5],
            [-0.8, 0.7, 5.0],
            [1.2, 0.8, 6.0],
            [0.0, 0.0, 7.0],
            [0.4, -1.0, 9.0],
        ]
    )
    K = torch.tensor(
        [[320.0, 0.0, 160.0], [0.0, 320.0, 120.0], [0.0, 0.0, 1.0]]
    )
    positive = torch.tensor([0.9, 0.7, 0.4, 0.2, 0.8, 0.5])
    negatives = torch.tensor(
        [
            [0.1, 0.0],
            [0.6, 0.5],
            [0.5, 0.3],
            [0.8, 0.7],
            [0.2, 0.1],
            [0.5, 0.4],
        ]
    )
    negative_mask = torch.ones_like(negatives, dtype=torch.bool)

    f4_weights, f4_diagnostics = _assignment_information_weights(
        points,
        K,
        torch.eye(4),
        positive,
        negatives,
        negative_mask,
        mode="conditional_translation",
        blend=1.0,
        floor=0.1,
        normalization="quantile",
    )
    f5_weights, f5_diagnostics = _assignment_information_weights(
        points,
        K,
        torch.eye(4),
        positive,
        negatives,
        negative_mask,
        mode="conditional_translation",
        blend=1.0,
        floor=0.1,
        normalization="quantile",
        use_matchability=True,
        matchability_floor=0.05,
        uncertainty_entropy_scale=2.0,
        match_temperature=0.1,
    )

    assert torch.isfinite(f4_weights).all()
    assert torch.isfinite(f5_weights).all()
    assert torch.all(f4_weights >= 0.1)
    assert torch.all(f5_weights >= 0.1)
    assert not torch.allclose(f4_weights, f5_weights)
    assert f4_diagnostics["assignment_pose_information_mode_id"] == 4.0
    assert f5_diagnostics["assignment_pose_information_uses_matchability"] == 1.0
    assert f5_diagnostics["assignment_pose_information_matchability_mean"] < 1.0
    assert f5_diagnostics["assignment_pose_information_sigma_mean"] > 1.0
    assert f5_diagnostics["assignment_fisher_translation_min_eigenvalue"] > 0.0


def test_candidate_assignment_row_weights_change_the_optimized_objective():
    from localization_training.sparse_candidate_teacher import _row_assignment_loss

    positive = torch.tensor([1.0, -1.0], requires_grad=True)
    negative = torch.tensor([[0.0], [0.0]], requires_grad=True)
    mask = torch.ones_like(negative, dtype=torch.bool)
    unweighted = _row_assignment_loss(
        positive,
        negative,
        mask,
        temperature=1.0,
        margin=0.0,
    )
    weighted = _row_assignment_loss(
        positive,
        negative,
        mask,
        temperature=1.0,
        margin=0.0,
        weights=torch.tensor([1.0, 0.1]),
    )

    assert weighted < unweighted
    weighted.backward()
    assert torch.isfinite(positive.grad).all()
    assert torch.isfinite(negative.grad).all()


def test_counterfactual_pairwise_loss_pushes_gt_above_actual_top1():
    from localization_training.sparse_candidate_teacher import (
        _counterfactual_pairwise_assignment_loss,
    )

    positive = torch.tensor([0.1, 0.4], requires_grad=True)
    negative = torch.tensor([0.8, 0.2], requires_grad=True)
    loss = _counterfactual_pairwise_assignment_loss(
        positive,
        negative,
        torch.tensor([True, False]),
        torch.tensor([1.0, 1.0]),
        temperature=0.1,
        margin=0.05,
    )
    loss.backward()

    assert loss.item() > 0.0
    assert positive.grad[0] < 0.0
    assert negative.grad[0] > 0.0
    assert positive.grad[1] == 0.0
    assert negative.grad[1] == 0.0


def test_quota_displacement_returns_lowest_retained_pair():
    from localization_training.sparse_candidate_teacher import (
        _quota_displaced_candidate_indices,
    )

    displaced = _quota_displaced_candidate_indices(
        torch.tensor([0, 0, 0, 1]),
        torch.tensor([0.9, 0.7, 0.2, 0.8]),
        torch.tensor([True, True, False, True]),
        torch.tensor([0, 1, 2]),
        2,
    )

    assert displaced.tolist() == [1, -1, -1]


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
        counterfactual_enabled=True,
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
    assert batch.counterfactual_assignment_valid_mask.tolist() == [True]
    assert batch.counterfactual_positive_similarity.shape == (1,)
    assert batch.counterfactual_negative_similarity.shape == (1,)
    assert batch.counterfactual_assignment_weights.item() > 0.0
    assert batch.diagnostics["counterfactual_swap_target_count"] == 1


def test_counterfactual_assignment_missed_mode_only_repairs_ambiguity_band():
    from localization_training.sparse_candidate_teacher import build_sparse_candidate_batch

    feature_map = torch.zeros(2, 5, 5)
    feature_map[:, 1, 1] = torch.tensor([1.0, 0.0])
    heatmap = torch.zeros(1, 5, 5)
    heatmap[0, 1, 1] = 0.9
    landmark_xyz = torch.tensor(
        [_point_at_pixel(1.5, 1.5), _point_at_pixel(3.5, 3.5)]
    )
    landmark_features = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    K, pose = _camera()
    common = dict(
        detect_num=1,
        nms_radius=0,
        positive_radius_px=0.25,
        hard_negatives=1,
        match_margin=0.0,
        counterfactual_enabled=True,
        counterfactual_target_mode="assignment_missed",
    )

    missed = build_sparse_candidate_batch(
        feature_map,
        heatmap,
        landmark_features,
        landmark_xyz,
        K,
        pose,
        negative_radius_px=4.0,
        **common,
    )
    already_supervised = build_sparse_candidate_batch(
        feature_map,
        heatmap,
        landmark_features,
        landmark_xyz,
        K,
        pose,
        negative_radius_px=2.0,
        **common,
    )

    assert missed.counterfactual_assignment_valid_mask.tolist() == [True]
    assert missed.diagnostics["counterfactual_swap_eligible_target_count"] == 1
    assert already_supervised.counterfactual_assignment_valid_mask.tolist() == [False]
    assert already_supervised.diagnostics["counterfactual_swap_eligible_target_count"] == 0


def test_map_information_geometry_mask_rejects_invisible_projectable_candidate():
    from localization_training.sparse_candidate_teacher import build_sparse_candidate_batch

    feature_map = torch.zeros(2, 5, 5)
    feature_map[:, 1, 1] = torch.tensor([1.0, 0.0])
    heatmap = torch.zeros(1, 5, 5)
    heatmap[0, 1, 1] = 0.9
    landmark_xyz = torch.tensor([_point_at_pixel(1.5, 1.5)])
    landmark_features = torch.tensor([[1.0, 0.0]])
    K, pose = _camera()

    batch = build_sparse_candidate_batch(
        feature_map,
        heatmap,
        landmark_features,
        landmark_xyz,
        K,
        pose,
        visible_mask=torch.tensor([False]),
        detect_num=1,
        nms_radius=0,
        positive_radius_px=0.25,
    )

    assert batch.pair_scorer_labels.tolist() == [0.0]
    assert batch.map_candidate_geometry_valid_mask.tolist() == [False]
    assert batch.diagnostics["map_candidate_invisible_projectable_count"] == 1
    assert (
        batch.diagnostics["map_candidate_invisible_projectable_under_4px_count"]
        == 1
    )


def test_map_information_geometry_mask_matches_final_landmark_quota():
    from localization_training.sparse_candidate_teacher import build_sparse_candidate_batch

    feature_map = torch.zeros(2, 5, 5)
    heatmap = torch.zeros(1, 5, 5)
    for x in (0, 2, 4):
        feature_map[:, 1, x] = torch.tensor([1.0, 0.0])
        heatmap[0, 1, x] = 0.9
    landmark_xyz = torch.tensor([_point_at_pixel(2.5, 1.5)])
    landmark_features = torch.tensor([[1.0, 0.0]])
    K, pose = _camera()

    batch = build_sparse_candidate_batch(
        feature_map,
        heatmap,
        landmark_features,
        landmark_xyz,
        K,
        pose,
        detect_num=3,
        nms_radius=0,
        positive_radius_px=3.0,
        map_max_matches_per_landmark=2,
    )

    assert batch.diagnostics["predicted_pair_count"] == 3
    assert batch.diagnostics["map_candidate_after_landmark_quota_count"] == 2
    assert batch.diagnostics["map_candidate_landmark_quota_removed_count"] == 1
    assert int(batch.map_candidate_geometry_valid_mask.sum().item()) == 2


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


def test_multi_positive_assignment_accepts_any_geometric_positive():
    from localization_training.sparse_candidate_teacher import (
        _multi_positive_assignment_loss,
    )

    positive_similarity = torch.tensor([[0.4, 0.9]], requires_grad=True)
    negative_similarity = torch.tensor([[0.7]], requires_grad=True)
    both_positive = _multi_positive_assignment_loss(
        positive_similarity,
        torch.tensor([[True, True]]),
        negative_similarity,
        torch.tensor([[True]]),
        temperature=0.1,
        margin=0.0,
        weights=torch.tensor([1.0]),
    )
    nearest_only = _multi_positive_assignment_loss(
        positive_similarity,
        torch.tensor([[True, False]]),
        negative_similarity,
        torch.tensor([[True]]),
        temperature=0.1,
        margin=0.0,
        weights=torch.tensor([1.0]),
    )
    both_positive.backward()

    assert both_positive.item() < nearest_only.item()
    assert positive_similarity.grad is not None
    assert negative_similarity.grad is not None
    assert torch.isfinite(positive_similarity.grad).all()


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


def test_measurement_accepted_detector_targets_use_calibrated_pair_threshold():
    from localization_training.pair_measurement import PairMeasurementHead
    from localization_training.sparse_candidate_teacher import build_sparse_candidate_batch

    feature_map = torch.zeros(2, 5, 5)
    feature_map[:, 1, 1] = torch.tensor([1.0, 0.0])
    heatmap = torch.zeros(1, 5, 5)
    heatmap[0, 1, 1] = 0.9
    landmark_xyz = torch.tensor([_point_at_pixel(1.5, 1.5)])
    K, pose = _camera()
    measurement = PairMeasurementHead(
        descriptor_dim=2,
        patch_radius=0,
        hidden_dim=4,
    )
    common = dict(
        query_feature_map=feature_map,
        detector_heatmap=heatmap,
        landmark_features=torch.tensor([[1.0, 0.0]]),
        landmark_xyz=landmark_xyz,
        K=K,
        pose_gt_w2c=pose,
        detect_num=1,
        nms_radius=0,
        positive_radius_px=0.25,
        detector_target_source="measurement_accepted_correct",
        detector_binary_target=True,
        pair_measurement_head=measurement,
    )

    accepted = build_sparse_candidate_batch(
        **common,
        pair_measurement_accept_threshold=0.0,
    )
    rejected = build_sparse_candidate_batch(
        **common,
        pair_measurement_accept_threshold=100.0,
    )

    assert accepted.detector_targets.tolist() == [1.0]
    assert rejected.detector_targets.tolist() == [0.0]
