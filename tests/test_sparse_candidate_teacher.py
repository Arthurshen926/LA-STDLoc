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
    assert batch.diagnostics["geometry_condition"] > 0


def test_sparse_candidate_teacher_mnn_matches_eval_semantics():
    from localization_training.sparse_candidate_teacher import _final_matches

    score = torch.tensor([[0.9, 0.8], [0.85, 0.1]])
    image_idx, landmark_idx = _final_matches(score, mode="mnn", threshold=0.0)

    assert image_idx.tolist() == [0]
    assert landmark_idx.tolist() == [0]
