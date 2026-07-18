import torch


def _observations():
    from localization_training.detector_free_map import DetectorFreeObservationBatch

    return DetectorFreeObservationBatch(
        source_indices=torch.tensor([0]),
        query_features=torch.tensor([[1.0, 0.0]]),
        query_uv=torch.tensor([[10.0, 10.0]]),
        source_depth=torch.tensor([2.0]),
        bank_uv=torch.tensor(
            [[10.0, 10.0], [10.5, 10.0], [30.0, 30.0]]
        ),
        bank_depth=torch.tensor([2.0, 2.0, 2.0]),
        bank_projected=torch.tensor([True, True, True]),
        bank_visible=torch.tensor([True, True, True]),
    )


def test_descriptor_residual_is_bounded_and_normalized():
    from localization_training.detector_free_map import (
        materialize_descriptor_residual,
    )

    initial = torch.tensor([[1.0, 0.0]])
    residual = torch.tensor([[0.0, 10.0]])
    result = materialize_descriptor_residual(
        initial,
        residual,
        max_residual_norm=0.1,
    )
    assert torch.allclose(torch.linalg.norm(result, dim=-1), torch.ones(1))
    assert result[0, 1] < 0.11


def test_hard_hypothesis_treats_nearby_anchor_as_positive():
    from localization_training.detector_free_map import (
        hard_hypothesis_retrieval_loss,
    )

    features = torch.tensor(
        [[1.0, 0.0], [0.99, 0.01], [1.0, 0.0]],
        requires_grad=True,
    )
    output = hard_hypothesis_retrieval_loss(
        features,
        _observations(),
        hypothesis_topk=3,
        positive_radius_px=2.0,
        negative_radius_px=6.0,
    )
    assert output.diagnostics["retrieval_positive_count_mean"] >= 2.0
    assert output.diagnostics["retrieval_negative_count_mean"] >= 1.0
    output.loss.backward()
    assert features.grad is not None
    assert torch.isfinite(features.grad).all()


def test_hard_hypothesis_penalizes_current_false_top_match():
    from localization_training.detector_free_map import (
        hard_hypothesis_retrieval_loss,
    )

    observations = _observations()
    features = torch.tensor(
        [[0.7, 0.7], [0.6, 0.8], [0.99, 0.10]],
        requires_grad=True,
    )
    before = hard_hypothesis_retrieval_loss(
        features,
        observations,
        hypothesis_topk=2,
    )
    assert before.diagnostics["retrieval_top1_gt_precision_2px"] == 0.0
    before.loss.backward()
    assert features.grad[2].abs().sum() > 0


def test_balanced_observation_builder_uses_depth_visibility():
    from localization_training.detector_free_map import (
        build_detector_free_observations,
    )

    bank_xyz = torch.tensor(
        [
            [0.0, 0.0, 2.0],
            [0.5, 0.0, 2.0],
            [0.0, 0.0, -1.0],
        ]
    )
    feature_map = torch.zeros(2, 8, 8)
    feature_map[0] = 1.0
    K = torch.tensor([[4.0, 0.0, 4.0], [0.0, 4.0, 4.0], [0.0, 0.0, 1.0]])
    depth = torch.full((8, 8), 2.0)
    alpha = torch.ones(8, 8)
    batch = build_detector_free_observations(
        bank_xyz,
        feature_map,
        K,
        torch.eye(4),
        target_depth=depth,
        target_alpha=alpha,
    )
    assert batch.source_indices.tolist() == [0, 1]
    assert batch.bank_visible.tolist() == [True, True, False]


def test_observation_builder_accepts_rasterizer_visibility_override():
    from localization_training.detector_free_map import (
        build_detector_free_observations,
    )

    bank_xyz = torch.tensor([[0.0, 0.0, 2.0], [0.5, 0.0, 3.0]])
    feature_map = torch.zeros(2, 8, 8)
    feature_map[0] = 1.0
    K = torch.tensor([[4.0, 0.0, 4.0], [0.0, 4.0, 4.0], [0.0, 0.0, 1.0]])
    depth = torch.full((8, 8), 2.0)
    batch = build_detector_free_observations(
        bank_xyz,
        feature_map,
        K,
        torch.eye(4),
        target_depth=depth,
        target_alpha=torch.ones(8, 8),
        bank_visibility_mask=torch.tensor([True, True]),
    )
    assert batch.bank_visible.tolist() == [True, True]
    assert batch.source_indices.tolist() == [0, 1]


def test_observation_adaptive_trust_protects_weak_landmarks():
    from localization_training.detector_free_map import (
        observation_adaptive_trust_weights,
    )

    weights = observation_adaptive_trust_weights(torch.tensor([0, 2, 20]))
    assert weights[0] > weights[1] > weights[2]
    assert torch.allclose(weights.mean(), torch.tensor(1.0))


def test_jittered_observations_resample_frozen_query_feature():
    from localization_training.detector_free_map import (
        DetectorFreeObservationBatch,
        jitter_detector_free_observations,
    )

    feature_map = torch.zeros(2, 5, 5)
    feature_map[0] = 1.0
    observations = DetectorFreeObservationBatch(
        source_indices=torch.tensor([0]),
        query_features=torch.tensor([[1.0, 0.0]]),
        query_uv=torch.tensor([[2.0, 2.0]]),
        source_depth=torch.tensor([2.0]),
        bank_uv=torch.tensor([[2.0, 2.0]]),
        bank_depth=torch.tensor([2.0]),
        bank_projected=torch.tensor([True]),
        bank_visible=torch.tensor([True]),
        query_feature_map=feature_map,
    )
    jittered = jitter_detector_free_observations(
        observations,
        standard_deviation=1.0,
        maximum=0.5,
        generator=torch.Generator().manual_seed(3),
    )
    assert jittered.query_uv.shape == (1, 2)
    assert torch.linalg.norm(jittered.query_uv - observations.query_uv) <= 0.5001
    assert torch.allclose(jittered.query_features, torch.tensor([[1.0, 0.0]]))


def test_score_proposals_follow_frozen_nms_and_require_nearby_visible_anchor():
    from localization_training.detector_free_map import (
        DetectorFreeObservationBatch,
        build_score_proposal_observations,
    )

    feature_map = torch.zeros(2, 8, 8)
    feature_map[0] = 1.0
    feature_map[:, 4, 5] = torch.tensor([0.0, 1.0])
    observations = DetectorFreeObservationBatch(
        source_indices=torch.tensor([0]),
        query_features=torch.tensor([[1.0, 0.0]]),
        query_uv=torch.tensor([[1.0, 1.0]]),
        source_depth=torch.tensor([2.0]),
        bank_uv=torch.tensor([[1.0, 1.0], [5.0, 4.0], [7.0, 7.0]]),
        bank_depth=torch.tensor([2.0, 3.0, 4.0]),
        bank_projected=torch.tensor([True, True, True]),
        bank_visible=torch.tensor([True, True, False]),
        query_feature_map=feature_map,
    )
    score_map = torch.zeros(8, 8)
    score_map[4, 5] = 0.9
    score_map[7, 7] = 0.8
    proposals = build_score_proposal_observations(
        observations,
        score_map,
        max_proposals=8,
        nms_radius=1,
        positive_search_radius_px=1.0,
    )
    assert proposals.source_indices.tolist() == [1]
    assert proposals.query_uv.tolist() == [[5.0, 4.0]]
    assert torch.allclose(proposals.query_features, torch.tensor([[0.0, 1.0]]))
    assert proposals.source_depth.tolist() == [3.0]


def test_background_dustbin_loss_backpropagates_to_score_and_features():
    from localization_training.detector_free_map import (
        DetectorFreeObservationBatch,
        background_dustbin_loss,
    )

    feature_map = torch.zeros(2, 8, 8)
    feature_map[0] = 1.0
    observations = DetectorFreeObservationBatch(
        source_indices=torch.tensor([0]),
        query_features=torch.tensor([[1.0, 0.0]]),
        query_uv=torch.tensor([[1.0, 1.0]]),
        source_depth=torch.tensor([2.0]),
        bank_uv=torch.tensor([[1.0, 1.0], [6.0, 6.0]]),
        bank_depth=torch.tensor([2.0, 2.0]),
        bank_projected=torch.tensor([True, True]),
        bank_visible=torch.tensor([True, False]),
        query_feature_map=feature_map,
    )
    features = torch.tensor([[1.0, 0.0], [0.5, 0.5]], requires_grad=True)
    dustbin = torch.tensor(0.5, requires_grad=True)
    loss, diagnostics = background_dustbin_loss(
        features,
        observations,
        dustbin,
        sample_count=4,
        exclusion_radius_px=2.0,
        generator=torch.Generator().manual_seed(5),
    )
    assert diagnostics["dustbin_background_count"] > 0
    loss.backward()
    assert torch.isfinite(features.grad).all()
    assert torch.isfinite(dustbin.grad)
    assert dustbin.grad.abs() > 0


def test_dustbin_can_use_valid_no_anchor_regions_when_alpha_is_opaque():
    from localization_training.detector_free_map import (
        DetectorFreeObservationBatch,
        background_dustbin_loss,
    )

    feature_map = torch.zeros(2, 8, 8)
    feature_map[0] = 1.0
    observations = DetectorFreeObservationBatch(
        source_indices=torch.tensor([0]),
        query_features=torch.tensor([[1.0, 0.0]]),
        query_uv=torch.tensor([[1.0, 1.0]]),
        source_depth=torch.tensor([2.0]),
        bank_uv=torch.tensor([[1.0, 1.0]]),
        bank_depth=torch.tensor([2.0]),
        bank_projected=torch.tensor([True]),
        bank_visible=torch.tensor([True]),
        query_feature_map=feature_map,
        target_alpha_map=torch.ones(8, 8),
        query_valid_mask=torch.ones(8, 8, dtype=torch.bool),
    )
    features = torch.tensor([[1.0, 0.0]], requires_grad=True)
    dustbin = torch.tensor(0.2, requires_grad=True)
    loss, diagnostics = background_dustbin_loss(
        features,
        observations,
        dustbin,
        sample_count=8,
        exclusion_radius_px=2.0,
        allow_no_anchor=True,
        generator=torch.Generator().manual_seed(7),
    )
    assert diagnostics["dustbin_background_count"] > 0
    assert diagnostics["dustbin_no_anchor_count"] > 0
    loss.backward()
    assert dustbin.grad is not None
    assert torch.isfinite(dustbin.grad)


def test_local_correlation_prefers_descriptor_with_center_peak():
    from localization_training.detector_free_map import (
        DetectorFreeObservationBatch,
        local_correlation_peak_loss,
    )

    feature_map = torch.zeros(2, 7, 7)
    feature_map[0] = 1.0
    feature_map[:, 3, 3] = torch.tensor([0.0, 1.0])
    observations = DetectorFreeObservationBatch(
        source_indices=torch.tensor([0]),
        query_features=torch.tensor([[0.0, 1.0]]),
        query_uv=torch.tensor([[3.0, 3.0]]),
        source_depth=torch.tensor([2.0]),
        bank_uv=torch.tensor([[3.0, 3.0]]),
        bank_depth=torch.tensor([2.0]),
        bank_projected=torch.tensor([True]),
        bank_visible=torch.tensor([True]),
        query_feature_map=feature_map,
    )
    center_descriptor = torch.tensor([[0.0, 1.0]], requires_grad=True)
    offset_descriptor = torch.tensor([[1.0, 0.0]])
    center_loss, diagnostics = local_correlation_peak_loss(
        center_descriptor,
        observations,
        radius=1,
        target_sigma=0.25,
    )
    offset_loss, _ = local_correlation_peak_loss(
        offset_descriptor,
        observations,
        radius=1,
        target_sigma=0.25,
    )
    assert center_loss < offset_loss
    assert diagnostics["local_center_top1_ratio"] == 1.0
    center_loss.backward()
    assert torch.isfinite(center_descriptor.grad).all()

    observations.query_uv = torch.tensor([[0.0, 0.0]])
    observations.query_features = torch.tensor([[1.0, 0.0]])
    boundary_loss, _ = local_correlation_peak_loss(
        torch.tensor([[1.0, 0.0]]),
        observations,
        radius=1,
        target_sigma=0.25,
    )
    assert torch.isfinite(boundary_loss)


def test_pose_layer_has_finite_geometry_gradient_near_correct_rotation():
    from localization_training.detector_free_map import (
        DetectorFreeObservationBatch,
        LocalSoftCorrespondenceOutput,
        pose_layer_loss,
    )
    from localization_training.pose_refiner import project_points, se3_exp

    grid = torch.cartesian_prod(
        torch.linspace(-0.5, 0.5, 5),
        torch.linspace(-0.4, 0.4, 4),
    )
    points = torch.cat([grid, torch.full((grid.shape[0], 1), 3.0)], dim=1)
    points.requires_grad_(True)
    K = torch.tensor([[100.0, 0.0, 40.0], [0.0, 100.0, 30.0], [0.0, 0.0, 1.0]])
    gt_pose = torch.eye(4)
    target_uv, _ = project_points(points.detach(), K, gt_pose)
    observations = DetectorFreeObservationBatch(
        source_indices=torch.arange(points.shape[0]),
        query_features=torch.zeros(points.shape[0], 2),
        query_uv=target_uv,
        source_depth=torch.full((points.shape[0],), 3.0),
        bank_uv=target_uv,
        bank_depth=torch.full((points.shape[0],), 3.0),
        bank_projected=torch.ones(points.shape[0], dtype=torch.bool),
        bank_visible=torch.ones(points.shape[0], dtype=torch.bool),
        K=K,
        pose_w2c=gt_pose,
    )
    local = LocalSoftCorrespondenceOutput(
        expected_uv=target_uv,
        confidence=torch.ones(points.shape[0]),
        entropy=torch.zeros(points.shape[0]),
        valid=torch.ones(points.shape[0], dtype=torch.bool),
    )
    perturbation = torch.tensor([0.005, -0.003, 0.002, 0.001, -0.001, 0.001])
    loss, diagnostics = pose_layer_loss(
        points,
        observations,
        local,
        se3_exp(perturbation),
        min_points=6,
        max_points=32,
        max_condition_number=1e8,
    )
    assert diagnostics["pose_layer_active"] == 1.0
    assert torch.isfinite(loss)
    loss.backward()
    assert points.grad is not None
    assert torch.isfinite(points.grad).all()
