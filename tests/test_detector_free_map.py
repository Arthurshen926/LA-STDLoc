import torch


def test_descriptor_stage_can_be_explicitly_disabled():
    from localization_training.detector_free_map import descriptor_losses_active

    assert descriptor_losses_active(1, -1) is False
    assert descriptor_losses_active(1000, -1) is False
    assert descriptor_losses_active(1, 0) is True
    assert descriptor_losses_active(1000, 0) is True
    assert descriptor_losses_active(10, 10) is True
    assert descriptor_losses_active(11, 10) is False


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


def test_hard_hypothesis_never_injects_missed_source():
    from localization_training.detector_free_map import (
        hard_hypothesis_retrieval_loss,
    )

    observations = _observations()
    features = torch.tensor(
        [[0.0, 1.0], [0.1, 0.9], [0.9, 0.1]],
        requires_grad=True,
    )
    output = hard_hypothesis_retrieval_loss(
        features,
        observations,
        hypothesis_topk=1,
        missed_positive_weight=1.0,
        missed_positive_margin=0.1,
    )
    diagnostics = output.diagnostics
    assert diagnostics["retrieval_candidate_count_mean"] == 1.0
    assert diagnostics["retrieval_candidate_source_injected"] == 0.0
    assert diagnostics["retrieval_source_recall_at_hypothesis_k"] == 0.0
    assert diagnostics["retrieval_missed_source_count"] == 1
    assert diagnostics["retrieval_missed_positive_count"] == 1
    assert diagnostics["retrieval_valid_loss_rows"] == 0
    assert diagnostics["retrieval_missed_positive_loss"] > 0
    output.loss.backward()
    assert features.grad[:2].abs().sum() > 0
    assert features.grad[2].abs().sum() > 0


def test_missed_source_is_not_a_miss_when_equivalent_surfel_is_retrieved():
    from localization_training.detector_free_map import (
        hard_hypothesis_retrieval_loss,
    )

    observations = _observations()
    features = torch.tensor(
        [[0.0, 1.0], [1.0, 0.0], [0.9, 0.1]],
        requires_grad=True,
    )
    output = hard_hypothesis_retrieval_loss(
        features,
        observations,
        hypothesis_topk=1,
        missed_positive_weight=1.0,
        missed_positive_margin=0.1,
    )
    diagnostics = output.diagnostics
    assert diagnostics["retrieval_source_recall_at_hypothesis_k"] == 0.0
    assert diagnostics["retrieval_gt_recall_at_hypothesis_k"] == 1.0
    assert diagnostics["retrieval_missed_source_count"] == 1
    assert diagnostics["retrieval_missed_positive_count"] == 0
    assert diagnostics["retrieval_missed_positive_loss"] == 0.0


def test_hard_hypothesis_reports_true_global_recall_at_fixed_k():
    from localization_training.detector_free_map import (
        hard_hypothesis_retrieval_loss,
    )

    observations = _observations()
    features = torch.tensor(
        [[0.7, 0.7], [0.6, 0.8], [1.0, 0.0]],
        requires_grad=True,
    )
    output = hard_hypothesis_retrieval_loss(
        features,
        observations,
        hypothesis_topk=1,
    )
    assert output.diagnostics["retrieval_source_recall_at_1"] == 0.0
    assert output.diagnostics["retrieval_source_recall_at_4"] == 1.0


def test_native_outcome_loss_separates_keep_swap_miss_and_reject():
    from localization_training.detector_free_map import (
        DetectorFreeObservationBatch,
        hard_hypothesis_retrieval_loss,
    )

    observations = DetectorFreeObservationBatch(
        source_indices=torch.tensor([0, 1, 0, -1]),
        query_features=torch.nn.functional.normalize(
            torch.tensor(
                [
                    [1.0, 0.0, 0.0, 0.0],       # keep: landmark 0
                    [0.0, 0.8, 1.0, 0.0],       # swap: 2 -> 1
                    [0.4, 0.0, 0.6, 1.0],       # miss: 3/2, then 0
                    [0.0, 0.0, 1.0, 0.0],       # reject: no legal map point
                ]
            ),
            dim=-1,
        ),
        query_uv=torch.tensor(
            [[10.0, 10.0], [20.0, 10.0], [10.0, 10.0], [100.0, 100.0]]
        ),
        source_depth=torch.ones(4),
        bank_uv=torch.tensor(
            [[10.0, 10.0], [20.0, 10.0], [30.0, 30.0], [40.0, 40.0]]
        ),
        bank_depth=torch.ones(4),
        bank_projected=torch.ones(4, dtype=torch.bool),
        bank_visible=torch.ones(4, dtype=torch.bool),
    )
    features = torch.eye(4, requires_grad=True)
    output = hard_hypothesis_retrieval_loss(
        features,
        observations,
        hypothesis_topk=2,
        native_outcome_mode=True,
        native_nce_weight=0.0,
        native_keep_weight=1.0,
        native_keep_margin=1.1,
        native_swap_weight=1.0,
        native_swap_margin=0.05,
        native_miss_weight=1.0,
        native_miss_margin=0.05,
        native_reject_weight=1.0,
        native_reject_threshold=0.5,
    )
    diagnostics = output.diagnostics
    assert diagnostics["retrieval_native_outcome_mode"] == 1.0
    assert diagnostics["retrieval_native_keep_count"] == 1
    assert diagnostics["retrieval_native_swap_count"] == 1
    assert diagnostics["retrieval_native_miss_count"] == 1
    assert diagnostics["retrieval_native_reject_count"] == 1
    assert diagnostics["retrieval_native_keep_loss"] > 0.0
    assert diagnostics["retrieval_native_swap_loss"] > 0.0
    assert diagnostics["retrieval_native_miss_loss"] > 0.0
    assert diagnostics["retrieval_native_reject_loss"] > 0.0
    assert diagnostics["retrieval_candidate_source_injected"] == 0.0
    output.loss.backward()
    assert features.grad is not None
    assert torch.isfinite(features.grad).all()


def test_native_attractor_penalty_targets_repeated_wrong_top1_landmark():
    from localization_training.detector_free_map import (
        DetectorFreeObservationBatch,
        hard_hypothesis_retrieval_loss,
    )

    observations = DetectorFreeObservationBatch(
        source_indices=torch.tensor([0, 0]),
        query_features=torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
        query_uv=torch.tensor([[10.0, 10.0], [10.5, 10.0]]),
        source_depth=torch.ones(2),
        bank_uv=torch.tensor([[10.0, 10.0], [30.0, 30.0], [40.0, 40.0]]),
        bank_depth=torch.ones(3),
        bank_projected=torch.ones(3, dtype=torch.bool),
        bank_visible=torch.ones(3, dtype=torch.bool),
    )
    # Landmark 2 is the same wrong top-1 for both otherwise matchable rows.
    features = torch.tensor(
        [[0.7, 0.7], [0.0, 1.0], [0.99, 0.10]], requires_grad=True
    )
    output = hard_hypothesis_retrieval_loss(
        features,
        observations,
        hypothesis_topk=1,
        native_outcome_mode=True,
        native_nce_weight=0.0,
        native_keep_weight=0.0,
        native_swap_weight=0.0,
        native_miss_weight=0.0,
        native_reject_weight=0.0,
        native_attractor_weight=1.0,
        native_attractor_margin=0.05,
    )
    diagnostics = output.diagnostics
    assert diagnostics["retrieval_native_attractor_count"] == 2
    assert diagnostics["retrieval_native_attractor_unique_count"] == 1
    assert diagnostics["retrieval_native_attractor_max_count"] == 2
    assert diagnostics["retrieval_native_attractor_loss"] > 0.0
    output.loss.backward()
    assert features.grad[2].abs().sum() > 0.0


def test_native_global_attractor_prior_reweights_only_known_false_targets():
    from localization_training.detector_free_map import (
        DetectorFreeObservationBatch,
        hard_hypothesis_retrieval_loss,
    )

    observations = DetectorFreeObservationBatch(
        source_indices=torch.tensor([0, 0]),
        query_features=torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
        query_uv=torch.tensor([[10.0, 10.0], [10.5, 10.0]]),
        source_depth=torch.ones(2),
        bank_uv=torch.tensor([[10.0, 10.0], [30.0, 30.0], [40.0, 40.0]]),
        bank_depth=torch.ones(3),
        bank_projected=torch.ones(3, dtype=torch.bool),
        bank_visible=torch.ones(3, dtype=torch.bool),
    )
    features = torch.tensor(
        [[0.7, 0.7], [0.0, 1.0], [0.99, 0.10]], requires_grad=True
    )
    output = hard_hypothesis_retrieval_loss(
        features,
        observations,
        hypothesis_topk=1,
        native_outcome_mode=True,
        native_keep_weight=0.0,
        native_swap_weight=0.0,
        native_miss_weight=0.0,
        native_reject_weight=0.0,
        native_global_attractor_weight=1.0,
        native_global_attractor_scores=torch.tensor([0.0, 0.0, 3.0]),
    )
    diagnostics = output.diagnostics
    assert diagnostics["retrieval_native_global_attractor_count"] == 2
    assert diagnostics["retrieval_native_global_attractor_unique_count"] == 1
    assert diagnostics["retrieval_native_global_attractor_score_mean"] == 3.0
    assert diagnostics["retrieval_native_global_attractor_loss"] > 0.0
    output.loss.backward()
    assert features.grad[2].abs().sum() > 0.0


def test_native_outcome_ignores_measurement_limited_top1_globally():
    from localization_training.detector_free_map import (
        DetectorFreeObservationBatch,
        hard_hypothesis_retrieval_loss,
    )

    observations = DetectorFreeObservationBatch(
        source_indices=torch.tensor([0]),
        query_features=torch.tensor([[1.0, 0.0]]),
        query_uv=torch.tensor([[10.0, 10.0]]),
        source_depth=torch.ones(1),
        bank_uv=torch.tensor([[10.0, 10.0], [15.0, 10.0], [30.0, 30.0]]),
        bank_depth=torch.ones(3),
        bank_projected=torch.ones(3, dtype=torch.bool),
        bank_visible=torch.ones(3, dtype=torch.bool),
    )
    features = torch.tensor(
        [[0.0, 1.0], [1.0, 0.0], [0.8, 0.2]], requires_grad=True
    )
    output = hard_hypothesis_retrieval_loss(
        features,
        observations,
        hypothesis_topk=1,
        positive_radius_px=2.0,
        negative_radius_px=8.0,
        native_outcome_mode=True,
        native_nce_weight=0.0,
        native_keep_weight=1.0,
        native_swap_weight=1.0,
        native_miss_weight=1.0,
        native_reject_weight=1.0,
        native_attractor_weight=1.0,
    )
    diagnostics = output.diagnostics
    assert diagnostics["retrieval_native_ambiguous_top1_count"] == 1
    assert diagnostics["retrieval_native_keep_count"] == 0
    assert diagnostics["retrieval_native_swap_count"] == 0
    assert diagnostics["retrieval_native_miss_count"] == 0
    assert diagnostics["retrieval_native_reject_count"] == 0
    assert diagnostics["retrieval_native_reject_enabled"] == 1.0
    assert diagnostics["retrieval_native_reject_weight"] == 1.0
    assert diagnostics["retrieval_native_attractor_count"] == 0
    assert output.loss.item() == 0.0


def test_native_outcome_api_defaults_to_forced_top1_without_reject():
    import inspect

    from localization_training.detector_free_map import (
        hard_hypothesis_retrieval_loss,
    )

    assert (
        inspect.signature(hard_hypothesis_retrieval_loss)
        .parameters["native_reject_weight"]
        .default
        == 0.0
    )


def test_native_loose_keep_preserves_a_2_to_4_pixel_top1():
    from localization_training.detector_free_map import (
        DetectorFreeObservationBatch,
        hard_hypothesis_retrieval_loss,
    )

    observations = DetectorFreeObservationBatch(
        source_indices=torch.tensor([-1]),
        query_features=torch.tensor([[1.0, 0.0]]),
        query_uv=torch.tensor([[13.0, 10.0]]),
        source_depth=torch.zeros(1),
        bank_uv=torch.tensor([[10.0, 10.0], [40.0, 40.0]]),
        bank_depth=torch.ones(2),
        bank_projected=torch.ones(2, dtype=torch.bool),
        bank_visible=torch.ones(2, dtype=torch.bool),
    )
    features = torch.tensor([[1.0, 0.0], [0.9, 0.1]], requires_grad=True)
    output = hard_hypothesis_retrieval_loss(
        features,
        observations,
        hypothesis_topk=2,
        native_outcome_mode=True,
        native_nce_weight=0.0,
        native_keep_weight=0.0,
        native_keep_loose_weight=1.0,
        native_keep_loose_radius_px=4.0,
        native_keep_loose_margin=1.0,
        native_swap_weight=0.0,
        native_miss_weight=0.0,
        native_reject_weight=0.0,
    )
    assert output.diagnostics["retrieval_native_keep_loose_count"] == 1
    assert output.diagnostics["retrieval_native_keep_loose_loss"] > 0.0
    output.loss.backward()
    assert torch.isfinite(features.grad).all()


def test_random_retrieval_does_not_treat_unmatched_as_last_landmark():
    from localization_training.detector_free_map import (
        DetectorFreeObservationBatch,
        random_negative_retrieval_loss,
    )

    observations = _observations()
    observations = DetectorFreeObservationBatch(
        source_indices=torch.tensor([0, -1]),
        query_features=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        query_uv=torch.tensor([[10.0, 10.0], [30.0, 30.0]]),
        source_depth=torch.tensor([2.0, 0.0]),
        bank_uv=observations.bank_uv,
        bank_depth=observations.bank_depth,
        bank_projected=observations.bank_projected,
        bank_visible=observations.bank_visible,
    )
    features = torch.tensor(
        [[1.0, 0.0], [0.99, 0.01], [0.0, 1.0]],
        requires_grad=True,
    )
    output = random_negative_retrieval_loss(
        features,
        observations,
        negative_count=2,
        generator=torch.Generator().manual_seed(4),
    )
    assert output.diagnostics["retrieval_query_count"] == 1
    assert output.diagnostics["retrieval_unmatched_query_count"] == 1
    output.loss.backward()
    assert torch.isfinite(features.grad).all()


def test_local_soft_correspondence_invalidates_unmatched_rows():
    from localization_training.detector_free_map import (
        DetectorFreeObservationBatch,
        local_soft_correspondences,
    )

    feature_map = torch.zeros(2, 5, 5)
    feature_map[0] = 1.0
    observations = DetectorFreeObservationBatch(
        source_indices=torch.tensor([0, -1]),
        query_features=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        query_uv=torch.tensor([[2.0, 2.0], [3.0, 3.0]]),
        source_depth=torch.tensor([2.0, 0.0]),
        bank_uv=torch.tensor([[2.0, 2.0], [3.0, 3.0]]),
        bank_depth=torch.tensor([2.0, 2.0]),
        bank_projected=torch.tensor([True, True]),
        bank_visible=torch.tensor([True, True]),
        query_feature_map=feature_map,
    )
    features = torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)
    output = local_soft_correspondences(features, observations, radius=1)
    assert output.valid.tolist() == [True, False]
    assert output.confidence[1].item() == 0.0
    output.expected_uv[0].sum().backward()
    assert features.grad[1].abs().sum() == 0


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


def test_observation_builder_keeps_base_measurement_fixed_from_prediction():
    from localization_training.detector_free_map import (
        build_detector_free_observations,
        multiview_descriptor_loss,
    )

    base_xyz = torch.tensor([[0.0, 0.0, 2.0]])
    predicted_xyz = torch.tensor([[0.1, 0.0, 2.0]], requires_grad=True)
    feature_map = torch.zeros(2, 12, 12)
    feature_map[0] = 1.0
    K = torch.tensor(
        [[20.0, 0.0, 6.0], [0.0, 20.0, 6.0], [0.0, 0.0, 1.0]]
    )
    batch = build_detector_free_observations(
        base_xyz,
        feature_map,
        K,
        torch.eye(4),
        prediction_bank_xyz=predicted_xyz,
        target_depth=torch.full((12, 12), 2.0),
        target_alpha=torch.ones(12, 12),
    )
    assert torch.allclose(batch.query_uv, torch.tensor([[6.0, 6.0]]))
    assert torch.allclose(batch.base_bank_uv, torch.tensor([[6.0, 6.0]]))
    assert torch.allclose(batch.bank_uv, torch.tensor([[7.0, 6.0]]))
    descriptor = torch.tensor([[0.0, 1.0]], requires_grad=True)
    loss = multiview_descriptor_loss(descriptor, batch)
    loss.backward()
    assert descriptor.grad is not None and descriptor.grad.abs().sum() > 0
    assert predicted_xyz.grad is None


def test_native_sparse_observations_preserve_detector_coordinates_and_unmatched_rows():
    from localization_training.detector_free_map import (
        build_native_sparse_observations,
    )

    # K has a half-pixel principal point so a point on the optical axis projects
    # to the detector grid coordinate (4, 4) after the sparse +0.5 convention.
    bank_xyz = torch.tensor([[0.0, 0.0, 2.0], [0.5, 0.0, 2.0]])
    K = torch.tensor(
        [[10.0, 0.0, 4.5], [0.0, 10.0, 4.5], [0.0, 0.0, 1.0]]
    )
    keypoints = torch.tensor([[4.0, 4.0], [0.0, 0.0]])
    descriptors = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    batch = build_native_sparse_observations(
        bank_xyz,
        keypoints,
        descriptors,
        torch.tensor([1.0, 0.5]),
        K,
        torch.eye(4),
        image_size=(10, 10),
        bank_visibility_mask=torch.tensor([True, True]),
        max_observations=2,
        positive_radius_px=0.75,
        unmatched_fraction=0.5,
    )

    # The query is an actual detector proposal, not the base anchor projection.
    assert torch.equal(batch.query_uv, keypoints)
    assert batch.source_indices.tolist() == [0, -1]
    assert torch.allclose(batch.query_features, descriptors)
    assert torch.allclose(batch.base_bank_uv[0], torch.tensor([4.0, 4.0]))
    assert batch.query_feature_map is None


def test_native_sparse_observations_expose_all_geometric_positives_as_csr():
    from localization_training.detector_free_map import (
        build_native_sparse_observations,
        hard_hypothesis_retrieval_loss,
    )

    bank_xyz = torch.tensor(
        [
            [0.0, 0.0, 2.0],
            [0.01, 0.0, 2.0],
            [0.5, 0.0, 2.0],
        ]
    )
    K = torch.tensor(
        [[50.0, 0.0, 16.5], [0.0, 50.0, 16.5], [0.0, 0.0, 1.0]]
    )
    batch = build_native_sparse_observations(
        bank_xyz,
        torch.tensor([[16.0, 16.0]]),
        torch.tensor([[1.0, 0.0]]),
        torch.tensor([1.0]),
        K,
        torch.eye(4),
        image_size=(32, 32),
        bank_visibility_mask=torch.ones(3, dtype=torch.bool),
        positive_radius_px=1.0,
        max_observations=1,
    )

    assert batch.source_indices.tolist() == [0]
    assert batch.positive_offsets.tolist() == [0, 2]
    assert batch.positive_indices.tolist() == [0, 1]
    assert torch.all(batch.positive_reprojection_errors <= 1.0)

    retrieval = hard_hypothesis_retrieval_loss(
        torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]]),
        batch,
        hypothesis_topk=3,
        positive_radius_px=1.0,
        negative_radius_px=2.0,
        native_outcome_mode=True,
    )
    assert retrieval.diagnostics["retrieval_positive_multiplicity_p50"] == 2.0
    assert retrieval.diagnostics["retrieval_multi_positive_query_fraction"] == 1.0


def test_native_sparse_observation_default_and_coverage_counts():
    import inspect

    from localization_training.detector_free_map import (
        build_native_sparse_observations,
    )

    assert (
        inspect.signature(build_native_sparse_observations)
        .parameters["max_observations"]
        .default
        == 2048
    )
    bank_xyz = torch.tensor([[0.0, 0.0, 2.0]])
    K = torch.tensor(
        [[50.0, 0.0, 16.5], [0.0, 50.0, 16.5], [0.0, 0.0, 1.0]]
    )
    batch = build_native_sparse_observations(
        bank_xyz,
        torch.tensor([[16.0, 16.0], [4.0, 4.0], [40.0, 40.0]]),
        torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]),
        torch.ones(3),
        K,
        torch.eye(4),
        image_size=(32, 32),
        bank_visibility_mask=torch.ones(1, dtype=torch.bool),
    )

    assert batch.native_input_count == 3
    assert batch.native_valid_count == 2
    assert batch.native_selected_count == 2
    assert batch.configured_max_observations == 2048


def test_local_group_identity_assignment_prefers_distinct_identity():
    from localization_training.detector_free_map import (
        local_group_identity_assignment_loss,
    )

    teacher = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    distinct = teacher.clone().requires_grad_()
    collapsed = torch.tensor(
        [[1.0, 0.0], [1.0, 0.0]], requires_grad=True
    )
    indices = torch.arange(2)
    uv = torch.tensor([[4.0, 4.0], [12.0, 4.0]])
    groups = torch.zeros(2, dtype=torch.long)
    distinct_loss, diagnostics = local_group_identity_assignment_loss(
        distinct,
        teacher,
        indices,
        uv,
        groups,
        temperature=0.1,
        positive_radius_px=2.0,
    )
    collapsed_loss, _ = local_group_identity_assignment_loss(
        collapsed,
        teacher,
        indices,
        uv,
        groups,
        temperature=0.1,
        positive_radius_px=2.0,
    )
    collapsed_loss.backward()

    assert distinct_loss < collapsed_loss
    assert diagnostics["native_semidense_local_identity_group_count"] == 1
    assert collapsed.grad is not None
    assert collapsed.grad.abs().sum() > 0


def test_global_margin_preservation_penalizes_only_margin_degradation():
    from localization_training.detector_free_map import (
        build_native_sparse_observations,
        global_margin_preservation_loss,
    )

    bank_xyz = torch.tensor([[0.0, 0.0, 2.0], [0.2, 0.0, 2.0]])
    K = torch.tensor(
        [[50.0, 0.0, 16.5], [0.0, 50.0, 16.5], [0.0, 0.0, 1.0]]
    )
    observations = build_native_sparse_observations(
        bank_xyz,
        torch.tensor([[16.0, 16.0]]),
        torch.tensor([[1.0, 0.0]]),
        torch.ones(1),
        K,
        torch.eye(4),
        image_size=(32, 32),
        bank_visibility_mask=torch.ones(2, dtype=torch.bool),
        max_observations=1,
    )
    reference = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    degraded = torch.tensor(
        [[0.8, 0.6], [0.7, 0.714]], requires_grad=True
    )
    loss, diagnostics = global_margin_preservation_loss(
        degraded,
        reference,
        observations,
        torch.tensor([0]),
        torch.tensor([0]),
    )
    loss.backward()

    assert loss > 0
    assert diagnostics["native_semidense_margin_preserve_count"] == 1
    assert diagnostics["native_semidense_margin_preserve_violation_rate"] == 1.0
    assert degraded.grad is not None
    assert degraded.grad.abs().sum() > 0


def test_protected_semidense_v2_routes_high_precision_to_margin_only():
    from localization_training.detector_free_map import (
        build_native_sparse_observations,
        native_semidense_neighborhood_loss,
    )

    bank_xyz = torch.tensor([[0.0, 0.0, 2.0], [0.2, 0.0, 2.0]])
    K = torch.tensor(
        [[50.0, 0.0, 16.5], [0.0, 50.0, 16.5], [0.0, 0.0, 1.0]]
    )
    dense = torch.zeros(2, 4, 4)
    dense[0] = 1.0
    observations = build_native_sparse_observations(
        bank_xyz,
        torch.tensor([[16.0, 16.0]]),
        torch.tensor([[1.0, 0.0]]),
        torch.ones(1),
        K,
        torch.eye(4),
        image_size=(32, 32),
        bank_visibility_mask=torch.ones(2, dtype=torch.bool),
        query_feature_map=dense,
        max_observations=1,
    )
    reference = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    degraded = torch.tensor(
        [[0.8, 0.6], [0.7, 0.714]], requires_grad=True
    )
    loss, diagnostics = native_semidense_neighborhood_loss(
        degraded,
        bank_xyz,
        torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]),
        observations,
        protected_v2=True,
        measurement_min_reprojection_px=2.0,
        measurement_max_reprojection_px=8.0,
        margin_preservation_weight=1.0,
        reference_bank_features=reference,
    )
    loss.backward()

    assert diagnostics["native_semidense_measurement_limited_count"] == 0
    assert diagnostics["native_semidense_protected_high_precision_count"] == 1
    assert diagnostics["native_semidense_teacher_pair_count"] == 0
    assert diagnostics["native_semidense_margin_preserve_count"] == 1
    assert loss > 0
    assert degraded.grad is not None
    assert degraded.grad.abs().sum() > 0


def test_protected_semidense_v2_excludes_current_query_protected_neighbors():
    from localization_training.detector_free_map import (
        build_native_sparse_observations,
        native_semidense_neighborhood_loss,
    )

    bank_xyz = torch.tensor([[0.0, 0.0, 2.0], [0.12, 0.0, 2.0]])
    K = torch.tensor(
        [[50.0, 0.0, 16.5], [0.0, 50.0, 16.5], [0.0, 0.0, 1.0]]
    )
    dense = torch.zeros(2, 4, 4)
    dense[0] = 1.0
    observations = build_native_sparse_observations(
        bank_xyz,
        torch.tensor([[16.0, 16.0], [19.0, 16.0]]),
        torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
        torch.ones(2),
        K,
        torch.eye(4),
        image_size=(32, 32),
        bank_visibility_mask=torch.ones(2, dtype=torch.bool),
        query_feature_map=dense,
        max_observations=2,
    )
    features = torch.tensor(
        [[0.0, 1.0], [1.0, 0.0]], requires_grad=True
    )
    loss, diagnostics = native_semidense_neighborhood_loss(
        features,
        bank_xyz,
        torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]),
        observations,
        protected_v2=True,
        measurement_min_reprojection_px=2.0,
        measurement_max_reprojection_px=8.0,
        neighbors_per_anchor=2,
        surface_max_distance_m=0.15,
        surface_normal_cosine=0.95,
        surface_point_plane_m=0.03,
    )
    loss.backward()

    assert diagnostics["native_semidense_measurement_limited_count"] == 1
    assert diagnostics["native_semidense_protected_high_precision_count"] == 1
    assert diagnostics["native_semidense_protected_neighbor_excluded_count"] == 1
    assert diagnostics["native_semidense_teacher_pair_count"] == 1
    assert features.grad is not None
    assert features.grad[0].abs().sum() > 0
    assert torch.allclose(features.grad[1], torch.zeros_like(features.grad[1]))


def test_protected_semidense_accepts_any_csr_positive_surface():
    from localization_training.detector_free_map import (
        build_native_sparse_observations,
        native_semidense_neighborhood_loss,
    )

    # Landmark 0 is the nearest reprojection positive but lies on a different
    # depth surface. Landmark 1 is another legal positive on the same surface
    # as the current 2--8 px top-1 landmark 2.
    bank_xyz = torch.tensor(
        [[0.0, 0.0, 5.0], [0.06, 0.0, 2.0], [0.12, 0.0, 2.0]]
    )
    K = torch.tensor(
        [[50.0, 0.0, 16.5], [0.0, 50.0, 16.5], [0.0, 0.0, 1.0]]
    )
    dense = torch.zeros(2, 4, 4)
    dense[0] = 1.0
    observations = build_native_sparse_observations(
        bank_xyz,
        torch.tensor([[16.0, 16.0]]),
        torch.tensor([[1.0, 0.0]]),
        torch.ones(1),
        K,
        torch.eye(4),
        image_size=(32, 32),
        bank_visibility_mask=torch.ones(3, dtype=torch.bool),
        query_feature_map=dense,
        positive_radius_px=2.0,
        max_observations=1,
    )
    assert observations.source_indices.tolist() == [0]
    assert set(observations.positive_indices.tolist()) == {0, 1}
    features = torch.tensor(
        [[0.0, 1.0], [0.0, 1.0], [1.0, 0.0]], requires_grad=True
    )
    _, diagnostics = native_semidense_neighborhood_loss(
        features,
        bank_xyz,
        torch.tensor([[0.0, 0.0, 1.0]]).repeat(3, 1),
        observations,
        protected_v2=True,
        measurement_min_reprojection_px=2.0,
        measurement_max_reprojection_px=8.0,
        neighbors_per_anchor=1,
        surface_max_distance_m=0.15,
        surface_normal_cosine=0.95,
        surface_point_plane_m=0.03,
    )

    assert diagnostics["native_semidense_csr_surface_reference"] == 1.0
    assert diagnostics["native_semidense_measurement_limited_count"] == 1
    assert diagnostics["native_semidense_anchor_is_deployment_top1"] == 1.0


def test_native_semidense_teacher_updates_same_descriptor_field():
    from localization_training.detector_free_map import (
        build_native_sparse_observations,
        native_semidense_neighborhood_loss,
    )

    bank_xyz = torch.tensor([[0.0, 0.0, 2.0], [0.04, 0.0, 2.0]])
    K = torch.tensor(
        [[50.0, 0.0, 16.5], [0.0, 50.0, 16.5], [0.0, 0.0, 1.0]]
    )
    dense = torch.zeros(2, 4, 4)
    dense[0] = torch.tensor(
        [
            [1.0, 1.0, 0.5, 0.0],
            [1.0, 1.0, 0.5, 0.0],
            [0.5, 0.5, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )
    dense[1] = 1.0 - dense[0]
    dense = torch.nn.functional.normalize(dense, dim=0)
    observations = build_native_sparse_observations(
        bank_xyz,
        torch.tensor([[16.0, 16.0]]),
        torch.tensor([[1.0, 0.0]]),
        torch.tensor([1.0]),
        K,
        torch.eye(4),
        image_size=(32, 32),
        bank_visibility_mask=torch.ones(2, dtype=torch.bool),
        query_valid_mask=torch.ones(32, 32, dtype=torch.bool),
        query_feature_map=dense,
        positive_radius_px=2.0,
        max_observations=1,
    )
    features = torch.tensor(
        [[1.0, 0.0], [0.8, 0.2]], requires_grad=True
    )
    loss, diagnostics = native_semidense_neighborhood_loss(
        features,
        bank_xyz,
        torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]),
        observations,
        positive_radius_px=2.0,
        max_anchors=1,
        neighbors_per_anchor=2,
        neighborhood_radius_m=0.1,
        local_radius_px=8,
        target_sigma_px=2.0,
        pose_safe_max_delete_gain_m=0.0,
        pose_safe_teacher_pairs=True,
        lgcv_weight=0.25,
    )
    loss.backward()

    assert diagnostics["native_semidense_clean_anchor_count"] == 1
    assert diagnostics["native_semidense_teacher_pair_count"] == 2
    assert diagnostics["native_semidense_unique_landmarks"] == 2
    assert diagnostics["native_semidense_lgcv_loss"] == 0.0
    assert diagnostics["native_semidense_pair_pose_safe_enabled"] == 1.0
    assert torch.isfinite(loss)
    assert features.grad is not None
    assert features.grad.abs().sum() > 0


def test_group_geometric_consistency_penalizes_direction_scale_and_orientation():
    from localization_training.detector_free_map import (
        group_geometric_consistency_loss,
    )

    target = torch.tensor(
        [[4.0, 4.0], [8.0, 4.0], [4.0, 8.0], [8.0, 8.0]]
    )
    exact = target.clone().requires_grad_()
    exact_loss, exact_diagnostics = group_geometric_consistency_loss(
        exact,
        target,
        torch.zeros(4, dtype=torch.long),
    )
    distorted = target.clone()
    distorted[1] += torch.tensor([1.5, 1.0])
    distorted[2] += torch.tensor([0.5, -1.0])
    distorted.requires_grad_()
    distorted_loss, diagnostics = group_geometric_consistency_loss(
        distorted,
        target,
        torch.zeros(4, dtype=torch.long),
    )

    assert exact_loss.item() < 1e-7
    assert distorted_loss > exact_loss
    assert exact_diagnostics["native_semidense_lgcv_group_count"] == 1
    assert diagnostics["native_semidense_lgcv_edge_count"] == 6
    assert diagnostics["native_semidense_lgcv_triangle_count"] > 0
    distorted_loss.backward()
    assert distorted.grad is not None
    assert distorted.grad.abs().sum() > 0


def test_counterfactual_pose_gate_rejects_bias_improving_deletions():
    from localization_training.detector_free_map import (
        counterfactual_pose_safe_mask,
    )
    from localization_training.pose_refiner import project_points

    xy = torch.cartesian_prod(
        torch.linspace(-0.8, 0.8, 4),
        torch.linspace(-0.5, 0.5, 3),
    )
    points = torch.cat(
        [xy, torch.linspace(2.5, 5.0, xy.shape[0])[:, None]], dim=1
    )
    K = torch.tensor(
        [[300.0, 0.0, 160.0], [0.0, 300.0, 120.0], [0.0, 0.0, 1.0]]
    )
    pose = torch.eye(4)
    observed, _ = project_points(points, K, pose)
    observed = observed - 0.5
    observed[-1] += torch.tensor([2.0, -1.5])

    safe, diagnostics = counterfactual_pose_safe_mask(
        points,
        torch.arange(points.shape[0]),
        observed,
        K,
        pose,
        maximum_delete_gain_m=0.0,
        minimum_correspondences=6,
    )

    assert diagnostics["native_semidense_pose_safe_evaluable"] == 1.0
    assert diagnostics["native_semidense_pose_harmful_count"] > 0
    assert diagnostics["native_semidense_pose_delete_gain_max_m"] > 0.0
    assert safe.sum() < safe.numel()


def test_native_association_geometry_updates_only_predicted_surface_anchor():
    from dataclasses import replace

    from localization_training.detector_free_map import (
        build_native_sparse_observations,
        native_association_geometry_losses,
    )

    base_xyz = torch.tensor([[0.0, 0.0, 2.0]])
    predicted_xyz = torch.tensor([[0.02, 0.0, 2.0]], requires_grad=True)
    K = torch.tensor(
        [[50.0, 0.0, 8.5], [0.0, 50.0, 8.5], [0.0, 0.0, 1.0]]
    )
    observations = build_native_sparse_observations(
        base_xyz,
        torch.tensor([[8.0, 8.0]]),
        torch.tensor([[1.0, 0.0]]),
        torch.tensor([1.0]),
        K,
        torch.eye(4),
        image_size=(16, 16),
        prediction_bank_xyz=predicted_xyz,
        target_depth=torch.full((16, 16), 2.0),
        target_alpha=torch.ones(16, 16),
        bank_visibility_mask=torch.tensor([True]),
        max_observations=1,
        positive_radius_px=1.0,
    )
    # BA must reproject ``predicted_xyz`` itself rather than receiving its
    # gradient only through the observation batch assembled before the loss.
    observations = replace(
        observations,
        bank_uv=observations.bank_uv.detach(),
        bank_depth=observations.bank_depth.detach(),
        bank_projected=observations.bank_projected.detach(),
    )
    descriptor = torch.tensor([[1.0, 0.0]], requires_grad=True)
    raw_offset = torch.zeros_like(predicted_xyz, requires_grad=True)
    _, depth_loss, reprojection_loss, diagnostics = native_association_geometry_losses(
        predicted_xyz,
        raw_offset,
        descriptor,
        observations,
        max_reprojection_error_px=1.0,
    )
    assert diagnostics["native_geometry_clean_correspondences"] == 1
    (depth_loss + reprojection_loss).backward()
    assert predicted_xyz.grad is not None
    assert predicted_xyz.grad.abs().sum() > 0
    assert descriptor.grad is None


def test_native_association_geometry_requires_supported_landmark():
    from localization_training.detector_free_map import (
        build_native_sparse_observations,
        native_association_geometry_losses,
    )

    xyz = torch.tensor([[0.0, 0.0, 2.0]], requires_grad=True)
    K = torch.tensor(
        [[50.0, 0.0, 8.5], [0.0, 50.0, 8.5], [0.0, 0.0, 1.0]]
    )
    observations = build_native_sparse_observations(
        xyz.detach(),
        torch.tensor([[8.0, 8.0]]),
        torch.tensor([[1.0, 0.0]]),
        torch.tensor([1.0]),
        K,
        torch.eye(4),
        image_size=(16, 16),
        prediction_bank_xyz=xyz,
        target_depth=torch.full((16, 16), 2.0),
        target_alpha=torch.ones(16, 16),
        bank_visibility_mask=torch.tensor([True]),
        max_observations=1,
        positive_radius_px=1.0,
    )
    raw_offset = torch.zeros_like(xyz, requires_grad=True)
    _, depth_loss, reprojection_loss, diagnostics = native_association_geometry_losses(
        xyz,
        raw_offset,
        torch.tensor([[1.0, 0.0]]),
        observations,
        max_reprojection_error_px=1.0,
        landmark_support_mask=torch.tensor([False]),
    )
    assert diagnostics["native_geometry_clean_before_support"] == 1
    assert diagnostics["native_geometry_clean_correspondences"] == 0
    assert diagnostics["native_geometry_support_eligible_landmarks"] == 0
    assert depth_loss.item() == 0.0
    assert reprojection_loss.item() == 0.0


def test_native_association_depth_gate_rejects_back_surface_match():
    from localization_training.detector_free_map import (
        build_native_sparse_observations,
        native_association_geometry_losses,
    )

    xyz = torch.tensor([[0.0, 0.0, 2.0]], requires_grad=True)
    K = torch.tensor(
        [[50.0, 0.0, 8.5], [0.0, 50.0, 8.5], [0.0, 0.0, 1.0]]
    )
    observations = build_native_sparse_observations(
        xyz.detach(),
        torch.tensor([[8.0, 8.0]]),
        torch.tensor([[1.0, 0.0]]),
        torch.tensor([1.0]),
        K,
        torch.eye(4),
        image_size=(16, 16),
        prediction_bank_xyz=xyz,
        target_depth=torch.full((16, 16), 3.0),
        target_alpha=torch.ones(16, 16),
        bank_visibility_mask=torch.tensor([True]),
        max_observations=1,
        positive_radius_px=1.0,
    )
    raw_offset = torch.zeros_like(xyz, requires_grad=True)
    _, depth_loss, reprojection_loss, diagnostics = native_association_geometry_losses(
        xyz,
        raw_offset,
        torch.tensor([[1.0, 0.0]]),
        observations,
        max_reprojection_error_px=1.0,
        depth_abs_tolerance=0.001,
        depth_rel_tolerance=0.01,
    )
    assert diagnostics["native_geometry_depth_gate_enabled"] == 1.0
    assert diagnostics["native_geometry_depth_gate_valid_count"] == 1
    assert diagnostics["native_geometry_depth_gate_rejected_count"] == 1
    assert diagnostics["native_geometry_clean_correspondences"] == 0
    assert depth_loss.item() == 0.0
    assert reprojection_loss.item() == 0.0


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
    measurement_uv = target_uv.clone().requires_grad_()
    confidence = torch.ones(points.shape[0], requires_grad=True)
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
        expected_uv=measurement_uv,
        confidence=confidence,
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
    assert measurement_uv.grad is None
    assert confidence.grad is None


def test_bounded_geometry_uses_independent_measurement_and_only_updates_xyz():
    from localization_training.detector_free_map import (
        bounded_geometry_losses,
        build_detector_free_observations,
    )

    base_xyz = torch.tensor([[0.0, 0.0, 2.0]])
    predicted_xyz = torch.tensor([[0.02, 0.0, 2.0]], requires_grad=True)
    feature_map = torch.zeros(2, 16, 16)
    feature_map[0] = 1.0
    feature_map[:, 8, 8] = torch.tensor([0.0, 1.0])
    K = torch.tensor(
        [[50.0, 0.0, 8.0], [0.0, 50.0, 8.0], [0.0, 0.0, 1.0]]
    )
    observations = build_detector_free_observations(
        base_xyz,
        feature_map,
        K,
        torch.eye(4),
        prediction_bank_xyz=predicted_xyz,
        target_depth=torch.full((16, 16), 2.0),
        target_alpha=torch.ones(16, 16),
    )
    descriptor = torch.tensor([[0.0, 1.0]], requires_grad=True)
    raw_offset = torch.zeros_like(predicted_xyz, requires_grad=True)
    _, depth, reprojection, _, _ = bounded_geometry_losses(
        predicted_xyz,
        raw_offset,
        descriptor,
        observations,
        local_radius=1,
    )
    (depth + reprojection).backward()
    assert predicted_xyz.grad is not None
    assert predicted_xyz.grad.abs().sum() > 0
    assert descriptor.grad is None


def test_pose_layer_feature_mode_routes_gradient_only_to_measurement():
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
    K = torch.tensor(
        [[100.0, 0.0, 40.0], [0.0, 100.0, 30.0], [0.0, 0.0, 1.0]]
    )
    gt_pose = torch.eye(4)
    target_uv, _ = project_points(points.detach(), K, gt_pose)
    measurement_uv = (target_uv + torch.tensor([0.15, -0.1])).requires_grad_()
    confidence = torch.ones(points.shape[0], requires_grad=True)
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
        expected_uv=measurement_uv,
        confidence=confidence,
        entropy=torch.zeros(points.shape[0]),
        valid=torch.ones(points.shape[0], dtype=torch.bool),
    )
    perturbation = torch.tensor(
        [0.005, -0.003, 0.002, 0.001, -0.001, 0.001]
    )
    loss, diagnostics = pose_layer_loss(
        points,
        observations,
        local,
        se3_exp(perturbation),
        min_points=6,
        max_points=32,
        max_condition_number=1e8,
        gradient_mode="feature",
    )
    assert diagnostics["pose_layer_feature_gradient_mode"] == 1.0
    loss.backward()
    assert measurement_uv.grad is not None
    assert measurement_uv.grad.abs().sum() > 0
    assert confidence.grad is not None
    assert confidence.grad.abs().sum() > 0
    assert points.grad is None


def test_overlap_initial_state_aligns_descriptors_and_offsets_by_id(tmp_path):
    import torch

    from train_lafgs_map import _load_initial_features

    state_path = tmp_path / "state.pt"
    torch.save(
        {
            "landmark_indices": torch.tensor([5, 2]),
            "landmark_features": torch.tensor([[0.0, 1.0], [1.0, 0.0]]),
            "raw_anchor_offset": torch.tensor(
                [[5.0, 5.1, 5.2], [2.0, 2.1, 2.2]]
            ),
            "mvinit_observation_count": torch.tensor([50, 20]),
        },
        state_path,
    )
    fallback = torch.tensor(
        [[0.0, -1.0], [-1.0, 0.0], [1.0, 1.0]]
    )

    features, aligned_state, matched = _load_initial_features(
        state_path,
        torch.tensor([2, 5, 7]),
        2,
        torch.device("cpu"),
        fallback_features=fallback,
        alignment="overlap",
    )

    assert matched == 2
    assert torch.allclose(features[0], torch.tensor([1.0, 0.0]))
    assert torch.allclose(features[1], torch.tensor([0.0, 1.0]))
    assert torch.allclose(
        features[2],
        torch.nn.functional.normalize(fallback[2], dim=0),
    )
    assert torch.allclose(
        aligned_state["raw_anchor_offset"],
        torch.tensor(
            [
                [2.0, 2.1, 2.2],
                [5.0, 5.1, 5.2],
                [0.0, 0.0, 0.0],
            ]
        ),
    )
    assert aligned_state["_raw_anchor_offset_alignment_valid"] is True
    assert aligned_state["_mvinit_observation_count_alignment_valid"] is True
    assert torch.equal(
        aligned_state["mvinit_observation_count"],
        torch.tensor([20, 50, 0]),
    )


def test_nonzero_initial_anchor_offset_cannot_change_saved_bounds():
    import pytest

    from localization_training.surface_anchor import (
        validate_surface_anchor_resume_bounds,
    )

    state = {
        "_raw_anchor_offset_alignment_valid": True,
        "raw_anchor_offset": torch.tensor([[0.4, -0.2, 0.1]]),
        "config": {"tangent_bound_m": 0.003, "normal_bound_m": 0.0015},
    }
    validate_surface_anchor_resume_bounds(
        state,
        tangent_bound_m=0.003,
        normal_bound_m=0.0015,
    )
    with pytest.raises(ValueError, match="nonzero raw_anchor_offset"):
        validate_surface_anchor_resume_bounds(
            state,
            tangent_bound_m=0.005,
            normal_bound_m=0.002,
        )

    zero_state = {
        **state,
        "raw_anchor_offset": torch.zeros(1, 3),
    }
    validate_surface_anchor_resume_bounds(
        zero_state,
        tangent_bound_m=0.005,
        normal_bound_m=0.002,
    )


def test_inactive_phase_gradient_clear_prevents_adamw_weight_decay():
    import torch

    from train_lafgs_map import _clear_inactive_phase_gradients

    residual = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
    anchor = torch.nn.Parameter(torch.tensor([0.5]))
    optimizer = torch.optim.AdamW(
        [
            {"params": [residual], "lr": 0.1, "weight_decay": 0.5},
            {"params": [anchor], "lr": 0.1, "weight_decay": 0.0},
        ]
    )
    before_residual = residual.detach().clone()
    before_anchor = anchor.detach().clone()
    (residual.sum() * 0.0 + anchor.sum() * 0.0).backward()

    _clear_inactive_phase_gradients(
        residual,
        anchor,
        None,
        descriptor_update_active=False,
        geometry_update_active=False,
        dustbin_update_active=False,
    )
    optimizer.step()

    assert residual.grad is None
    assert anchor.grad is None
    assert torch.equal(residual.detach(), before_residual)
    assert torch.equal(anchor.detach(), before_anchor)
