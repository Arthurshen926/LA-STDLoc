import unittest

import torch


class DirectLandmarkTeacherTest(unittest.TestCase):
    def test_stochastic_full_bank_is_bounded_and_keeps_all_positives(self):
        from localization_training.direct_landmark_teacher import (
            sample_stochastic_full_bank,
        )

        torch.manual_seed(7)
        bank = torch.arange(10000)
        positives = torch.tensor([9999, 17, 250, 17])
        sampled = sample_stochastic_full_bank(bank, positives, max_landmarks=512)

        self.assertLessEqual(sampled.numel(), 512)
        self.assertEqual(torch.unique(sampled).numel(), sampled.numel())
        self.assertTrue(torch.isin(torch.unique(positives), sampled).all())

    def test_multiview_memory_indices_follow_source_lineage_after_row_reorder(self):
        from localization_training.direct_landmark_teacher import (
            stable_landmark_memory_indices,
        )

        gaussians = type(
            "FakeGaussians",
            (),
            {"loc_source_index": torch.tensor([10, 20, 20, 30])},
        )()

        stable = stable_landmark_memory_indices(
            gaussians,
            torch.tensor([2, 0, 3, 1]),
        )

        self.assertEqual(stable.tolist(), [20, 10, 30, 20])

        partial = stable_landmark_memory_indices(
            gaussians,
            torch.tensor([0, 9]),
        )
        self.assertEqual(partial.tolist(), [10, 9])

    def test_direct_teacher_updates_multiview_memory_by_source_not_current_row(self):
        from localization_training.direct_landmark_teacher import (
            LandmarkObservationMemory,
            direct_landmark_teacher,
        )

        class FakeGaussians:
            def __init__(self):
                self._xyz = torch.tensor(
                    [[0.0, 0.0, 4.0], [0.5, 0.0, 4.0], [-0.5, 0.0, 4.0]],
                    dtype=torch.float32,
                )
                self._loc_feature = torch.nn.Parameter(
                    torch.tensor(
                        [[[1.0, 0.0]], [[0.0, 1.0]], [[1.0, 1.0]]],
                        dtype=torch.float32,
                    )
                )
                self.loc_source_index = torch.tensor([10, 20, 20])

            @property
            def get_xyz(self):
                return self._xyz

            @property
            def get_loc_feature(self):
                return self._loc_feature

        memory = LandmarkObservationMemory(
            torch.tensor([10, 20]),
            feature_dim=2,
            slots=2,
            device="cpu",
        )
        output = direct_landmark_teacher(
            FakeGaussians(),
            torch.ones(2, 16, 16),
            pose_gt_w2c=torch.eye(4),
            fovx=0.7610127542247298,
            fovy=0.7610127542247298,
            landmark_indices=torch.tensor([0, 1, 2]),
            target_depth=torch.full((16, 16), 4.0),
            alpha_threshold=0.0,
            depth_abs_tolerance=0.05,
            depth_rel_tolerance=0.01,
            max_landmarks=None,
            multiview_memory=memory,
            multiview_temperature=0.1,
        )

        self.assertEqual(output.diagnostics["multiview_memory_key_count"], 3)
        self.assertEqual(output.diagnostics["multiview_memory_unique_source_count"], 2)
        self.assertEqual(output.diagnostics["multiview_memory_shared_source_count"], 1)
        self.assertEqual(memory.positive_count(torch.tensor([10, 20])).tolist(), [1, 1])

    def test_depth_consistency_filters_occluded_landmarks(self):
        from localization_training.direct_landmark_teacher import (
            filter_depth_consistent_landmarks,
            project_landmarks_to_query,
        )

        xyz = torch.tensor([[0.0, 0.0, 4.0], [0.5, 0.0, 4.0]], dtype=torch.float32)
        pose = torch.eye(4, dtype=torch.float32)
        K = torch.tensor([[20.0, 0.0, 8.0], [0.0, 20.0, 8.0], [0.0, 0.0, 1.0]], dtype=torch.float32)

        uv, depth, valid = project_landmarks_to_query(xyz, K, pose, height=16, width=16)
        target_depth = torch.full((16, 16), 4.0, dtype=torch.float32)
        target_depth[8, 10] = 2.0

        consistent = filter_depth_consistent_landmarks(
            uv,
            depth,
            valid,
            target_depth,
            abs_tolerance=0.05,
            rel_tolerance=0.01,
        )

        self.assertTrue(consistent[0].item())
        self.assertFalse(consistent[1].item())

    def test_direct_loss_and_stats_use_query_observation_as_prototype(self):
        from localization_training.direct_landmark_teacher import direct_landmark_teacher

        class FakeGaussians:
            def __init__(self):
                self._xyz = torch.tensor([[0.0, 0.0, 4.0], [0.5, 0.0, 4.0]], dtype=torch.float32)
                self._loc_feature = torch.nn.Parameter(
                    torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]], dtype=torch.float32)
                )

            @property
            def get_xyz(self):
                return self._xyz

            @property
            def get_loc_feature(self):
                return self._loc_feature

        query_feature_map = torch.zeros(2, 16, 16, dtype=torch.float32)
        query_feature_map[:, 8, 8] = torch.tensor([1.0, 0.0])
        query_feature_map[:, 8, 10] = torch.tensor([1.0, 0.0])
        target_depth = torch.full((16, 16), 4.0, dtype=torch.float32)

        out = direct_landmark_teacher(
            FakeGaussians(),
            query_feature_map,
            pose_gt_w2c=torch.eye(4),
            fovx=0.7610127542247298,
            fovy=0.7610127542247298,
            landmark_indices=torch.tensor([0, 1]),
            target_depth=target_depth,
            alpha_threshold=0.0,
            depth_abs_tolerance=0.05,
            depth_rel_tolerance=0.01,
            max_landmarks=None,
        )

        self.assertEqual(out.loc_visible_idx.tolist(), [0, 1])
        self.assertGreater(out.loss.item(), 0.45)
        self.assertTrue(torch.allclose(out.stats["prototype"][0], torch.tensor([1.0, 0.0])))
        self.assertTrue(torch.allclose(out.stats["prototype"][1], torch.tensor([1.0, 0.0])))
        self.assertGreater(out.stats["positive_prob"][0].item(), out.stats["positive_prob"][1].item())

    def test_multiview_contrastive_uses_memory_positives_and_ignores_nearby_projection_negatives(self):
        from localization_training.direct_landmark_teacher import (
            LandmarkObservationMemory,
            multiview_contrastive_landmark_loss,
        )

        landmark_indices = torch.tensor([10, 20, 30])
        memory = LandmarkObservationMemory(landmark_indices, feature_dim=2, slots=2, device="cpu")
        memory.update(
            torch.tensor([10, 20]),
            torch.tensor(
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                ]
            ),
        )

        gaussian_features = torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
            ],
            requires_grad=True,
        )
        query_features = torch.tensor(
            [
                [0.8, 0.2],
                [0.0, 1.0],
                [1.0, 0.0],
            ]
        )
        uv = torch.tensor(
            [
                [8.0, 8.0],
                [30.0, 30.0],
                [8.5, 8.0],
            ]
        )

        loss = multiview_contrastive_landmark_loss(
            gaussian_features,
            query_features,
            torch.tensor([10, 20, 30]),
            target_uv=uv,
            memory=memory,
            temperature=0.1,
            ignore_radius=1.0,
        )
        loss_without_ignore = multiview_contrastive_landmark_loss(
            gaussian_features,
            query_features,
            torch.tensor([10, 20, 30]),
            target_uv=uv,
            memory=memory,
            temperature=0.1,
            ignore_radius=0.0,
        )

        self.assertLess(loss.item(), loss_without_ignore.item())
        loss.backward()
        self.assertIsNotNone(gaussian_features.grad)
        self.assertGreater(memory.positive_count(torch.tensor([10])).item(), 0)

    def test_observation_memory_preserves_view_diversity_and_quality(self):
        from localization_training.direct_landmark_teacher import LandmarkObservationMemory

        memory = LandmarkObservationMemory(
            torch.tensor([10]),
            feature_dim=2,
            slots=2,
            device="cpu",
            view_similarity_threshold=0.95,
        )
        memory.update(
            torch.tensor([10]),
            torch.tensor([[1.0, 0.0]]),
            view_directions=torch.tensor([[1.0, 0.0, 0.0]]),
            confidences=torch.tensor([0.5]),
        )
        memory.update(
            torch.tensor([10]),
            torch.tensor([[0.0, 1.0]]),
            view_directions=torch.tensor([[0.99, 0.01, 0.0]]),
            confidences=torch.tensor([0.4]),
        )
        features, valid = memory.lookup(torch.tensor([10]))
        self.assertTrue(torch.allclose(features[0, 0], torch.tensor([1.0, 0.0])))
        self.assertEqual(valid.sum().item(), 1)

        memory.update(
            torch.tensor([10]),
            torch.tensor([[0.0, 1.0]]),
            view_directions=torch.tensor([[1.0, 0.0, 0.0]]),
            confidences=torch.tensor([0.9]),
        )
        memory.update(
            torch.tensor([10]),
            torch.tensor([[0.7, 0.7]]),
            view_directions=torch.tensor([[0.0, 1.0, 0.0]]),
            confidences=torch.tensor([0.6]),
        )

        features, valid = memory.lookup(torch.tensor([10]))
        self.assertEqual(valid.sum().item(), 2)
        self.assertTrue(torch.allclose(features[0, 0], torch.tensor([0.0, 1.0])))
        self.assertTrue(torch.allclose(features[0, 1], torch.nn.functional.normalize(torch.tensor([0.7, 0.7]), dim=0)))

    def test_full_bank_bimnn_loss_uses_complete_landmark_bank_negatives(self):
        from localization_training.direct_landmark_teacher import full_bank_bimnn_loss

        query = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
        aligned_bank = torch.tensor(
            [
                [1.0, 0.0],
                [0.7, 0.7],
                [0.0, 1.0],
            ],
            dtype=torch.float32,
            requires_grad=True,
        )
        confused_bank = torch.tensor(
            [
                [0.0, 1.0],
                [1.0, 0.0],
                [0.0, 1.0],
            ],
            dtype=torch.float32,
            requires_grad=True,
        )
        positives = torch.tensor([0, 2])

        aligned = full_bank_bimnn_loss(query, aligned_bank, positives, temperature=0.2)
        confused = full_bank_bimnn_loss(query, confused_bank, positives, temperature=0.2)

        self.assertLess(aligned.item(), confused.item())
        aligned.backward()
        self.assertIsNotNone(aligned_bank.grad)

    def test_clean_reprojection_hard_negative_penalizes_far_repeated_matches(self):
        from localization_training.direct_landmark_teacher import clean_reprojection_hard_negative_loss

        query = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
        confused_bank = torch.tensor(
            [
                [1.0, 0.0],
                [0.99, 0.01],
                [0.0, 1.0],
            ],
            dtype=torch.float32,
            requires_grad=True,
        )
        clean_bank = torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [0.0, 1.0],
            ],
            dtype=torch.float32,
            requires_grad=True,
        )
        positive = torch.tensor([0])
        query_uv = torch.tensor([[8.0, 8.0]], dtype=torch.float32)
        bank_uv = torch.tensor(
            [
                [8.0, 8.0],
                [30.0, 30.0],
                [32.0, 32.0],
            ],
            dtype=torch.float32,
        )

        confused = clean_reprojection_hard_negative_loss(
            query,
            confused_bank,
            positive,
            query_uv,
            bank_uv,
            reprojection_radius=4.0,
            hard_negative_topk=1,
            margin=0.2,
        )
        clean = clean_reprojection_hard_negative_loss(
            query,
            clean_bank,
            positive,
            query_uv,
            bank_uv,
            reprojection_radius=4.0,
            hard_negative_topk=1,
            margin=0.2,
        )

        self.assertGreater(confused.item(), clean.item() + 0.1)
        confused.backward()
        self.assertIsNotNone(confused_bank.grad)

    def test_clean_reprojection_hard_negative_ignores_nearby_projected_aliases(self):
        from localization_training.direct_landmark_teacher import clean_reprojection_hard_negative_loss

        query = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
        bank = torch.tensor(
            [
                [1.0, 0.0],
                [0.99, 0.01],
                [0.0, 1.0],
            ],
            dtype=torch.float32,
        )
        query_uv = torch.tensor([[8.0, 8.0]], dtype=torch.float32)
        bank_uv = torch.tensor(
            [
                [8.0, 8.0],
                [9.0, 8.0],
                [30.0, 30.0],
            ],
            dtype=torch.float32,
        )

        loss = clean_reprojection_hard_negative_loss(
            query,
            bank,
            torch.tensor([0]),
            query_uv,
            bank_uv,
            reprojection_radius=4.0,
            hard_negative_topk=1,
            margin=0.2,
        )

        self.assertLess(loss.item(), 0.05)

    def test_geometry_balance_weights_downweight_repeated_cells_and_depth_bins(self):
        from localization_training.direct_landmark_teacher import geometry_balance_weights

        uv = torch.tensor(
            [
                [4.0, 4.0],
                [5.0, 4.0],
                [6.0, 4.0],
                [28.0, 28.0],
            ],
            dtype=torch.float32,
        )
        depth = torch.tensor([4.0, 4.1, 4.2, 20.0], dtype=torch.float32)

        weights = geometry_balance_weights(
            uv,
            depth=depth,
            image_size=(32, 32),
            grid_size=4,
            depth_bins=2,
            max_weight=4.0,
        )

        self.assertGreater(weights[-1].item(), weights[:3].mean().item())
        self.assertAlmostEqual(float(weights.mean().item()), 1.0, places=5)

    def test_direct_teacher_reports_clean_hard_negative_separate_from_full_bank_loss(self):
        from localization_training.direct_landmark_teacher import direct_landmark_teacher

        class FakeGaussians:
            def __init__(self):
                self._xyz = torch.tensor(
                    [
                        [0.0, 0.0, 4.0],
                        [1.4, 1.4, 4.0],
                    ],
                    dtype=torch.float32,
                )
                self._loc_feature = torch.nn.Parameter(
                    torch.tensor(
                        [
                            [[1.0, 0.0]],
                            [[0.99, 0.01]],
                        ],
                        dtype=torch.float32,
                    )
                )

            @property
            def get_xyz(self):
                return self._xyz

            @property
            def get_loc_feature(self):
                return self._loc_feature

        query_feature_map = torch.zeros(2, 64, 64, dtype=torch.float32)
        query_feature_map[:, 32, 32] = torch.tensor([1.0, 0.0])
        query_feature_map[:, 54, 54] = torch.tensor([0.99, 0.01])
        kwargs = dict(
            gaussians=FakeGaussians(),
            query_feature_map=query_feature_map,
            pose_gt_w2c=torch.eye(4),
            fovx=0.9272952180016123,
            fovy=0.9272952180016123,
            landmark_indices=torch.tensor([0]),
            full_bank_indices=torch.tensor([0, 1]),
            max_landmarks=1,
            full_bank_hard_negative_topk=1,
            full_bank_hard_negative_margin=0.2,
            full_bank_clean_reproj_radius=4.0,
            full_bank_clean_hard_negatives=1,
        )

        base = direct_landmark_teacher(**kwargs, full_bank_clean_hard_negative_weight=0.0)
        clean = direct_landmark_teacher(**kwargs, full_bank_clean_hard_negative_weight=1.0)

        self.assertTrue(torch.allclose(clean.full_bank_loss, base.full_bank_loss, atol=1e-6))
        self.assertGreater(clean.clean_hard_negative_loss.item(), 0.0)
        self.assertGreater(clean.diagnostics["full_bank_clean_hard_negative_loss"], 0.0)

    def test_full_bank_bimnn_loss_ignores_sibling_source_false_negatives(self):
        from localization_training.direct_landmark_teacher import full_bank_bimnn_loss

        query = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
        bank = torch.tensor(
            [
                [1.0, 0.0],
                [0.99, 0.01],
                [0.0, 1.0],
            ],
            dtype=torch.float32,
        )
        positives = torch.tensor([0])
        ignore_mask = torch.tensor([[False, True, False]])

        with_false_negative = full_bank_bimnn_loss(
            query,
            bank,
            positives,
            temperature=0.2,
            hard_negative_topk=1,
        )
        ignored = full_bank_bimnn_loss(
            query,
            bank,
            positives,
            temperature=0.2,
            hard_negative_topk=1,
            ignore_bank_mask=ignore_mask,
        )

        self.assertLess(ignored.item(), with_false_negative.item())

    def test_full_bank_descriptor_stats_ignore_mask_removes_nearby_false_negative(self):
        from localization_training.direct_landmark_teacher import full_bank_descriptor_stats

        query = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
        bank = torch.tensor(
            [
                [1.0, 0.0],
                [0.99, 0.01],
                [0.0, 1.0],
            ],
            dtype=torch.float32,
        )
        positives = torch.tensor([0])
        ignore_mask = torch.tensor([[False, True, False]])

        _, margin_without_ignore, _ = full_bank_descriptor_stats(query, bank, positives, temperature=0.2)
        _, margin_with_ignore, _ = full_bank_descriptor_stats(
            query,
            bank,
            positives,
            temperature=0.2,
            ignore_bank_mask=ignore_mask,
        )

        self.assertGreater(margin_with_ignore.item(), margin_without_ignore.item() + 4.0)

    def test_full_bank_bimnn_loss_and_stats_support_multi_positive_mask(self):
        from localization_training.direct_landmark_teacher import full_bank_bimnn_loss, full_bank_descriptor_stats

        query = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
        bank = torch.tensor(
            [
                [0.0, 1.0],
                [1.0, 0.0],
                [0.5, 0.5],
            ],
            dtype=torch.float32,
        )
        positives = torch.tensor([0])
        positive_mask = torch.tensor([[True, True, False]])

        single_positive = full_bank_bimnn_loss(query, bank, positives, temperature=0.2)
        multi_positive = full_bank_bimnn_loss(
            query,
            bank,
            positives,
            temperature=0.2,
            positive_bank_mask=positive_mask,
        )
        single_prob, single_margin, _ = full_bank_descriptor_stats(query, bank, positives, temperature=0.2)
        multi_prob, multi_margin, _ = full_bank_descriptor_stats(
            query,
            bank,
            positives,
            temperature=0.2,
            positive_bank_mask=positive_mask,
        )

        self.assertLess(multi_positive.item(), single_positive.item())
        self.assertGreater(multi_prob.item(), single_prob.item())
        self.assertGreater(multi_margin.item(), single_margin.item())

    def test_full_bank_descriptor_stats_chunking_matches_full_matrix(self):
        from localization_training.direct_landmark_teacher import full_bank_descriptor_stats

        torch.manual_seed(7)
        query = torch.randn(5, 4)
        bank = torch.randn(17, 4)
        positives = torch.tensor([0, 3, 8, 12, 16])
        ignore_mask = torch.zeros((5, 17), dtype=torch.bool)
        ignore_mask[0, 1:4] = True
        ignore_mask[2, 7] = True
        positive_mask = torch.zeros((5, 17), dtype=torch.bool)
        positive_mask[1, 3] = True
        positive_mask[1, 4] = True
        positive_mask[4, 15] = True

        full = full_bank_descriptor_stats(
            query,
            bank,
            positives,
            temperature=0.2,
            ignore_bank_mask=ignore_mask,
            positive_bank_mask=positive_mask,
        )
        chunked = full_bank_descriptor_stats(
            query,
            bank,
            positives,
            temperature=0.2,
            ignore_bank_mask=ignore_mask,
            positive_bank_mask=positive_mask,
            chunk_size=6,
        )

        for full_value, chunked_value in zip(full, chunked):
            self.assertTrue(torch.allclose(full_value, chunked_value, atol=1e-6))

    def test_full_bank_bimnn_loss_chunking_matches_loss_and_gradients(self):
        from localization_training.direct_landmark_teacher import full_bank_bimnn_loss

        torch.manual_seed(19)
        query = torch.randn(5, 4)
        bank = torch.randn(17, 4)
        positives = torch.tensor([0, -1, 8, 12, 16])
        ignore_mask = torch.zeros((5, 17), dtype=torch.bool)
        ignore_mask[0, 1:4] = True
        ignore_mask[2, 7] = True
        positive_mask = torch.zeros((5, 17), dtype=torch.bool)
        positive_mask[0, 1] = True
        positive_mask[3, 13] = True
        weights = torch.tensor([1.0, 3.0, 0.5, 1.5, 2.0])

        def loss_and_gradients(chunk_size, precomputed=False):
            current_query = query.clone().requires_grad_(True)
            current_bank = bank.clone().requires_grad_(True)
            shared_scores = None
            if precomputed:
                shared_scores = torch.nn.functional.normalize(current_query, dim=-1) @ (
                    torch.nn.functional.normalize(current_bank, dim=-1).T
                )
            loss = full_bank_bimnn_loss(
                current_query,
                current_bank,
                positives,
                temperature=0.2,
                hard_negative_topk=3,
                hard_negative_margin=0.15,
                weights=weights,
                ignore_bank_mask=ignore_mask,
                positive_bank_mask=positive_mask,
                chunk_size=chunk_size,
                query_bank_scores=shared_scores,
            )
            gradients = torch.autograd.grad(loss, (current_query, current_bank))
            return loss, gradients

        full_loss, full_gradients = loss_and_gradients(None)
        chunked_loss, chunked_gradients = loss_and_gradients(2)
        shared_loss, shared_gradients = loss_and_gradients(2, precomputed=True)

        torch.testing.assert_close(chunked_loss, full_loss, atol=1e-6, rtol=1e-6)
        for chunked, full in zip(chunked_gradients, full_gradients):
            torch.testing.assert_close(chunked, full, atol=2e-6, rtol=2e-5)
        torch.testing.assert_close(shared_loss, full_loss, atol=1e-6, rtol=1e-6)
        for shared, full in zip(shared_gradients, full_gradients):
            torch.testing.assert_close(shared, full, atol=2e-6, rtol=2e-5)

    def test_full_bank_precomputed_scores_and_uv_distances_are_equivalent(self):
        from localization_training.direct_landmark_teacher import (
            clean_reprojection_hard_negative_loss,
            full_bank_descriptor_stats,
        )

        torch.manual_seed(23)
        query = torch.randn(4, 8)
        bank = torch.randn(13, 8)
        positives = torch.tensor([0, -1, 7, 12])
        query_uv = torch.randn(4, 2) * 5.0
        bank_uv = torch.randn(13, 2) * 5.0
        scores = torch.nn.functional.normalize(query, dim=-1) @ (
            torch.nn.functional.normalize(bank, dim=-1).T
        )
        uv_distances = torch.cdist(query_uv, bank_uv)

        baseline_stats = full_bank_descriptor_stats(query, bank, positives, temperature=0.2)
        shared_stats = full_bank_descriptor_stats(
            query,
            bank,
            positives,
            temperature=0.2,
            query_bank_scores=scores,
        )
        for baseline, shared in zip(baseline_stats, shared_stats):
            torch.testing.assert_close(shared, baseline, atol=1e-6, rtol=1e-6)

        baseline_clean = clean_reprojection_hard_negative_loss(
            query,
            bank,
            positives,
            query_uv,
            bank_uv,
            reprojection_radius=2.0,
            hard_negative_topk=3,
        )
        shared_clean = clean_reprojection_hard_negative_loss(
            query,
            bank,
            positives,
            query_uv,
            bank_uv,
            reprojection_radius=2.0,
            hard_negative_topk=3,
            query_bank_scores=scores,
            query_bank_uv_distances=uv_distances,
        )
        torch.testing.assert_close(shared_clean, baseline_clean, atol=1e-6, rtol=1e-6)

    def test_direct_teacher_can_ignore_full_bank_3d_and_uv_nearby_false_negatives(self):
        from localization_training.direct_landmark_teacher import direct_landmark_teacher

        class FakeGaussians:
            def __init__(self):
                self._xyz = torch.tensor(
                    [
                        [0.0, 0.0, 4.0],
                        [0.05, 0.0, 4.0],
                        [0.0, 0.0, 5.0],
                        [0.5, 0.0, 4.0],
                    ],
                    dtype=torch.float32,
                )
                self._loc_feature = torch.nn.Parameter(
                    torch.tensor(
                        [
                            [[1.0, 0.0]],
                            [[0.99, 0.01]],
                            [[0.98, 0.02]],
                            [[0.0, 1.0]],
                        ],
                        dtype=torch.float32,
                    )
                )

            @property
            def get_xyz(self):
                return self._xyz

            @property
            def get_loc_feature(self):
                return self._loc_feature

        query_feature_map = torch.zeros(2, 16, 16, dtype=torch.float32)
        query_feature_map[:, 8, 8] = torch.tensor([1.0, 0.0])
        target_depth = torch.full((16, 16), 4.0, dtype=torch.float32)
        gaussians = FakeGaussians()

        out_without_ignore = direct_landmark_teacher(
            gaussians,
            query_feature_map,
            pose_gt_w2c=torch.eye(4),
            fovx=0.7610127542247298,
            fovy=0.7610127542247298,
            landmark_indices=torch.tensor([0]),
            target_depth=target_depth,
            alpha_threshold=0.0,
            depth_abs_tolerance=0.05,
            depth_rel_tolerance=0.01,
            max_landmarks=None,
            full_bank_indices=torch.tensor([0, 1, 2, 3]),
            full_bank_temperature=0.2,
            full_bank_hard_negative_topk=1,
        )
        out_with_ignore = direct_landmark_teacher(
            gaussians,
            query_feature_map,
            pose_gt_w2c=torch.eye(4),
            fovx=0.7610127542247298,
            fovy=0.7610127542247298,
            landmark_indices=torch.tensor([0]),
            target_depth=target_depth,
            alpha_threshold=0.0,
            depth_abs_tolerance=0.05,
            depth_rel_tolerance=0.01,
            max_landmarks=None,
            full_bank_indices=torch.tensor([0, 1, 2, 3]),
            full_bank_temperature=0.2,
            full_bank_hard_negative_topk=1,
            full_bank_ignore_3d_radius=0.1,
            full_bank_ignore_uv_radius=1.0,
        )

        self.assertLess(out_with_ignore.full_bank_loss.item(), out_without_ignore.full_bank_loss.item())
        self.assertGreater(
            out_with_ignore.stats["margin"].item(),
            out_without_ignore.stats["margin"].item(),
        )
        self.assertEqual(out_with_ignore.diagnostics["full_bank_query_count"], 1)
        self.assertEqual(out_with_ignore.diagnostics["full_bank_bank_count"], 4)
        self.assertEqual(out_with_ignore.diagnostics["full_bank_valid_positive_count"], 1)
        self.assertEqual(out_with_ignore.diagnostics["full_bank_ignore_negative_count"], 2)
        self.assertEqual(out_with_ignore.diagnostics["full_bank_effective_negative_count"], 1)
        self.assertAlmostEqual(out_with_ignore.diagnostics["full_bank_ignore_negative_ratio"], 2.0 / 3.0)

    def test_direct_teacher_can_treat_nearby_full_bank_entries_as_multi_positives(self):
        from localization_training.direct_landmark_teacher import direct_landmark_teacher

        class FakeGaussians:
            def __init__(self):
                self._xyz = torch.tensor(
                    [
                        [0.0, 0.0, 4.0],
                        [0.05, 0.0, 4.0],
                        [0.0, 0.0, 5.0],
                        [0.5, 0.0, 4.0],
                    ],
                    dtype=torch.float32,
                )
                self._loc_feature = torch.nn.Parameter(
                    torch.tensor(
                        [
                            [[0.0, 1.0]],
                            [[1.0, 0.0]],
                            [[0.99, 0.01]],
                            [[0.5, 0.5]],
                        ],
                        dtype=torch.float32,
                    )
                )

            @property
            def get_xyz(self):
                return self._xyz

            @property
            def get_loc_feature(self):
                return self._loc_feature

        query_feature_map = torch.zeros(2, 16, 16, dtype=torch.float32)
        query_feature_map[:, 8, 8] = torch.tensor([1.0, 0.0])
        target_depth = torch.full((16, 16), 4.0, dtype=torch.float32)
        gaussians = FakeGaussians()

        single_positive = direct_landmark_teacher(
            gaussians,
            query_feature_map,
            pose_gt_w2c=torch.eye(4),
            fovx=0.7610127542247298,
            fovy=0.7610127542247298,
            landmark_indices=torch.tensor([0]),
            target_depth=target_depth,
            alpha_threshold=0.0,
            depth_abs_tolerance=0.05,
            depth_rel_tolerance=0.01,
            max_landmarks=None,
            full_bank_indices=torch.tensor([0, 1, 2, 3]),
            full_bank_temperature=0.2,
            full_bank_hard_negative_topk=1,
            full_bank_ignore_3d_radius=0.1,
            full_bank_ignore_uv_radius=1.0,
        )
        multi_positive = direct_landmark_teacher(
            gaussians,
            query_feature_map,
            pose_gt_w2c=torch.eye(4),
            fovx=0.7610127542247298,
            fovy=0.7610127542247298,
            landmark_indices=torch.tensor([0]),
            target_depth=target_depth,
            alpha_threshold=0.0,
            depth_abs_tolerance=0.05,
            depth_rel_tolerance=0.01,
            max_landmarks=None,
            full_bank_indices=torch.tensor([0, 1, 2, 3]),
            full_bank_temperature=0.2,
            full_bank_hard_negative_topk=1,
            full_bank_ignore_3d_radius=0.1,
            full_bank_ignore_uv_radius=1.0,
            full_bank_nearby_as_positive=True,
        )

        self.assertLess(multi_positive.full_bank_loss.item(), single_positive.full_bank_loss.item())
        self.assertEqual(multi_positive.diagnostics["full_bank_positive_count"], 3)
        self.assertEqual(multi_positive.diagnostics["full_bank_extra_positive_count"], 2)
        self.assertEqual(multi_positive.diagnostics["full_bank_effective_negative_count"], 1)

    def test_direct_teacher_responsibility_source_mode_keeps_sibling_losers_as_negatives(self):
        from localization_training.direct_landmark_teacher import direct_landmark_teacher

        class FakeGaussians:
            def __init__(self):
                self._xyz = torch.tensor(
                    [
                        [0.0, 0.0, 4.0],
                        [0.05, 0.0, 4.0],
                        [0.5, 0.0, 4.0],
                    ],
                    dtype=torch.float32,
                )
                self.loc_source_index = torch.tensor([10, 10, 20])
                self._loc_feature = torch.nn.Parameter(
                    torch.tensor(
                        [
                            [[1.0, 0.0]],
                            [[0.99, 0.01]],
                            [[0.0, 1.0]],
                        ],
                        dtype=torch.float32,
                    )
                )

            @property
            def get_xyz(self):
                return self._xyz

            @property
            def get_loc_feature(self):
                return self._loc_feature

        query_feature_map = torch.zeros(2, 16, 16, dtype=torch.float32)
        query_feature_map[:, 8, 8] = torch.tensor([1.0, 0.0])
        target_depth = torch.full((16, 16), 4.0, dtype=torch.float32)
        gaussians = FakeGaussians()

        ignored = direct_landmark_teacher(
            gaussians,
            query_feature_map,
            pose_gt_w2c=torch.eye(4),
            fovx=0.7610127542247298,
            fovy=0.7610127542247298,
            landmark_indices=torch.tensor([0]),
            target_depth=target_depth,
            alpha_threshold=0.0,
            depth_abs_tolerance=0.05,
            depth_rel_tolerance=0.01,
            max_landmarks=None,
            full_bank_indices=torch.tensor([0, 1, 2]),
            full_bank_temperature=0.2,
            full_bank_hard_negative_topk=1,
            full_bank_source_mode="ignore",
        )
        responsibility = direct_landmark_teacher(
            gaussians,
            query_feature_map,
            pose_gt_w2c=torch.eye(4),
            fovx=0.7610127542247298,
            fovy=0.7610127542247298,
            landmark_indices=torch.tensor([0]),
            target_depth=target_depth,
            alpha_threshold=0.0,
            depth_abs_tolerance=0.05,
            depth_rel_tolerance=0.01,
            max_landmarks=None,
            full_bank_indices=torch.tensor([0, 1, 2]),
            full_bank_temperature=0.2,
            full_bank_hard_negative_topk=1,
            full_bank_source_mode="responsibility",
        )

        self.assertGreater(responsibility.full_bank_loss.item(), ignored.full_bank_loss.item())
        self.assertEqual(ignored.diagnostics["full_bank_source_ignore_count"], 1)
        self.assertEqual(ignored.diagnostics["full_bank_source_negative_count"], 0)
        self.assertEqual(responsibility.diagnostics["full_bank_source_ignore_count"], 0)
        self.assertEqual(responsibility.diagnostics["full_bank_source_negative_count"], 1)

    def test_child_responsibility_feature_mode_keeps_best_child_per_source(self):
        from localization_training.direct_landmark_teacher import child_responsibility_keep_mask

        selected_full_idx = torch.tensor([0, 1, 2])
        source_index = torch.tensor([10, 10, 20])
        gaussian_features = torch.tensor(
            [
                [0.0, 1.0],
                [1.0, 0.0],
                [0.0, 1.0],
            ],
            dtype=torch.float32,
        )
        query_features = torch.tensor(
            [
                [1.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
            ],
            dtype=torch.float32,
        )

        keep = child_responsibility_keep_mask(
            selected_full_idx,
            source_index,
            gaussian_features,
            query_features,
            mode="feature",
        )

        self.assertEqual(keep.tolist(), [False, True, True])

    def test_direct_teacher_child_responsibility_updates_only_best_child_per_source(self):
        from localization_training.direct_landmark_teacher import direct_landmark_teacher

        class FakeGaussians:
            def __init__(self):
                self._xyz = torch.tensor(
                    [
                        [0.0, 0.0, 4.0],
                        [0.05, 0.0, 4.0],
                        [0.5, 0.0, 4.0],
                    ],
                    dtype=torch.float32,
                )
                self.loc_source_index = torch.tensor([10, 10, 20])
                self._loc_feature = torch.nn.Parameter(
                    torch.tensor(
                        [
                            [[0.0, 1.0]],
                            [[1.0, 0.0]],
                            [[0.0, 1.0]],
                        ],
                        dtype=torch.float32,
                    )
                )

            @property
            def get_xyz(self):
                return self._xyz

            @property
            def get_loc_feature(self):
                return self._loc_feature

        query_feature_map = torch.zeros(2, 16, 16, dtype=torch.float32)
        query_feature_map[:, 8, 8] = torch.tensor([1.0, 0.0])
        query_feature_map[:, 8, 10] = torch.tensor([1.0, 0.0])
        query_feature_map[:, 8, 12] = torch.tensor([0.0, 1.0])

        out = direct_landmark_teacher(
            FakeGaussians(),
            query_feature_map,
            pose_gt_w2c=torch.eye(4),
            fovx=0.7610127542247298,
            fovy=0.7610127542247298,
            landmark_indices=torch.tensor([0, 1, 2]),
            target_depth=torch.full((16, 16), 4.0, dtype=torch.float32),
            alpha_threshold=0.0,
            depth_abs_tolerance=0.05,
            depth_rel_tolerance=0.01,
            max_landmarks=None,
            child_responsibility_mode="feature",
        )

        self.assertEqual(out.loc_visible_idx.tolist(), [1, 2])
        self.assertEqual(out.diagnostics["child_responsibility_candidate_count"], 3)
        self.assertEqual(out.diagnostics["child_responsibility_kept_count"], 2)
        self.assertEqual(out.diagnostics["child_responsibility_dropped_count"], 1)

    def test_direct_teacher_child_responsibility_competes_before_anchor_limit(self):
        from localization_training.direct_landmark_teacher import direct_landmark_teacher

        class FakeGaussians:
            def __init__(self):
                self._xyz = torch.tensor(
                    [
                        [0.0, 0.0, 4.0],
                        [0.05, 0.0, 4.0],
                    ],
                    dtype=torch.float32,
                )
                self.loc_source_index = torch.tensor([10, 10])
                self._loc_feature = torch.nn.Parameter(
                    torch.tensor(
                        [
                            [[0.0, 1.0]],
                            [[1.0, 0.0]],
                        ],
                        dtype=torch.float32,
                    )
                )

            @property
            def get_xyz(self):
                return self._xyz

            @property
            def get_loc_feature(self):
                return self._loc_feature

        query_feature_map = torch.zeros(2, 16, 16, dtype=torch.float32)
        query_feature_map[:, 8, 8] = torch.tensor([1.0, 0.0])

        out = direct_landmark_teacher(
            FakeGaussians(),
            query_feature_map,
            pose_gt_w2c=torch.eye(4),
            fovx=0.7610127542247298,
            fovy=0.7610127542247298,
            landmark_indices=torch.tensor([0, 1]),
            target_depth=torch.full((16, 16), 4.0, dtype=torch.float32),
            alpha_threshold=0.0,
            depth_abs_tolerance=0.05,
            depth_rel_tolerance=0.01,
            max_landmarks=1,
            child_responsibility_mode="feature",
        )

        self.assertEqual(out.loc_visible_idx.tolist(), [1])
        self.assertEqual(out.diagnostics["child_responsibility_candidate_count"], 2)
        self.assertEqual(out.diagnostics["child_responsibility_kept_count"], 1)
        self.assertEqual(out.diagnostics["child_responsibility_dropped_count"], 1)

    def test_limit_valid_indices_can_stratify_by_projection_grid(self):
        from localization_training.direct_landmark_teacher import _limit_valid_indices

        valid = torch.ones(12, dtype=torch.bool)
        uv = torch.tensor([[1.0, 1.0]] * 10 + [[9.0, 1.0], [15.0, 15.0]])

        keep = _limit_valid_indices(valid, max_landmarks=3, uv=uv, image_size=(16, 16), grid_size=2)

        self.assertIn(10, keep.tolist())
        self.assertIn(11, keep.tolist())
        self.assertEqual(keep.numel(), 3)

    def test_anchor_loss_penalizes_descriptor_drift_from_baseline(self):
        from localization_training.direct_landmark_teacher import descriptor_anchor_loss

        current = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        baseline = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
        weights = torch.tensor([0.0, 1.0])

        loss = descriptor_anchor_loss(current, baseline, weights=weights)

        self.assertAlmostEqual(loss.item(), 1.0)

    def test_direct_teacher_downweights_landmarks_in_artifact_regions(self):
        from localization_training.direct_landmark_teacher import direct_landmark_teacher

        class FakeGaussians:
            def __init__(self):
                self._xyz = torch.tensor([[0.0, 0.0, 4.0], [0.5, 0.0, 4.0]], dtype=torch.float32)
                self._loc_feature = torch.nn.Parameter(
                    torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]], dtype=torch.float32)
                )

            @property
            def get_xyz(self):
                return self._xyz

            @property
            def get_loc_feature(self):
                return self._loc_feature

        query_feature_map = torch.zeros(2, 16, 16, dtype=torch.float32)
        query_feature_map[:, 8, 8] = torch.tensor([1.0, 0.0])
        query_feature_map[:, 8, 10] = torch.tensor([1.0, 0.0])
        target_depth = torch.full((16, 16), 4.0, dtype=torch.float32)
        artifact_map = torch.ones(16, 16, dtype=torch.float32)
        artifact_map[:, 9:] = 0.1

        unweighted = direct_landmark_teacher(
            FakeGaussians(),
            query_feature_map,
            pose_gt_w2c=torch.eye(4),
            fovx=0.7610127542247298,
            fovy=0.7610127542247298,
            landmark_indices=torch.tensor([0, 1]),
            target_depth=target_depth,
            alpha_threshold=0.0,
            depth_abs_tolerance=0.05,
            depth_rel_tolerance=0.01,
            max_landmarks=None,
        )
        weighted = direct_landmark_teacher(
            FakeGaussians(),
            query_feature_map,
            pose_gt_w2c=torch.eye(4),
            fovx=0.7610127542247298,
            fovy=0.7610127542247298,
            landmark_indices=torch.tensor([0, 1]),
            target_depth=target_depth,
            alpha_threshold=0.0,
            depth_abs_tolerance=0.05,
            depth_rel_tolerance=0.01,
            max_landmarks=None,
            artifact_weight_map=artifact_map,
        )

        self.assertLess(weighted.desc_loss.item(), unweighted.desc_loss.item() * 0.35)
        self.assertAlmostEqual(weighted.diagnostics["artifact_region_weight_min"], 0.1, places=3)
        self.assertEqual(weighted.diagnostics["artifact_region_weighted_count"], 1)

    def test_artifact_combined_mean_scales_direct_teacher_loss(self):
        from localization_training.direct_landmark_teacher import direct_landmark_teacher

        class FakeGaussians:
            def __init__(self):
                self._xyz = torch.tensor([[0.0, 0.0, 4.0], [0.5, 0.0, 4.0]], dtype=torch.float32)
                self._loc_feature = torch.nn.Parameter(
                    torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]], dtype=torch.float32)
                )

            @property
            def get_xyz(self):
                return self._xyz

            @property
            def get_loc_feature(self):
                return self._loc_feature

        query_feature_map = torch.zeros(2, 16, 16, dtype=torch.float32)
        query_feature_map[:, 8, 8] = torch.tensor([1.0, 0.0])
        query_feature_map[:, 8, 10] = torch.tensor([1.0, 0.0])
        target_depth = torch.full((16, 16), 4.0, dtype=torch.float32)
        artifact_map = torch.ones(16, 16, dtype=torch.float32) * 0.25

        unscaled = direct_landmark_teacher(
            FakeGaussians(),
            query_feature_map,
            pose_gt_w2c=torch.eye(4),
            fovx=0.7610127542247298,
            fovy=0.7610127542247298,
            landmark_indices=torch.tensor([0, 1]),
            target_depth=target_depth,
            alpha_threshold=0.0,
            depth_abs_tolerance=0.05,
            depth_rel_tolerance=0.01,
            max_landmarks=None,
            artifact_weight_map=artifact_map,
            artifact_image_weight=0.5,
            artifact_weight_combine_mode="product",
            artifact_loss_scale_mode="none",
        )
        scaled = direct_landmark_teacher(
            FakeGaussians(),
            query_feature_map,
            pose_gt_w2c=torch.eye(4),
            fovx=0.7610127542247298,
            fovy=0.7610127542247298,
            landmark_indices=torch.tensor([0, 1]),
            target_depth=target_depth,
            alpha_threshold=0.0,
            depth_abs_tolerance=0.05,
            depth_rel_tolerance=0.01,
            max_landmarks=None,
            artifact_weight_map=artifact_map,
            artifact_image_weight=0.5,
            artifact_weight_combine_mode="product",
            artifact_loss_scale_mode="combined_mean",
        )

        self.assertAlmostEqual(scaled.diagnostics["artifact_teacher_loss_scale"], 0.125, places=4)
        self.assertAlmostEqual(
            scaled.desc_loss.item(),
            unscaled.desc_loss.item() * 0.125,
            places=5,
        )
        self.assertLess(scaled.loss.item(), unscaled.loss.item() * 0.2)

    def test_artifact_loss_scale_none_preserves_legacy_weighted_mean(self):
        from localization_training.direct_landmark_teacher import direct_landmark_teacher

        class FakeGaussians:
            def __init__(self):
                self._xyz = torch.tensor([[0.0, 0.0, 4.0], [0.5, 0.0, 4.0]], dtype=torch.float32)
                self._loc_feature = torch.nn.Parameter(
                    torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]], dtype=torch.float32)
                )

            @property
            def get_xyz(self):
                return self._xyz

            @property
            def get_loc_feature(self):
                return self._loc_feature

        query_feature_map = torch.zeros(2, 16, 16, dtype=torch.float32)
        query_feature_map[:, 8, 8] = torch.tensor([1.0, 0.0])
        query_feature_map[:, 8, 10] = torch.tensor([1.0, 0.0])
        target_depth = torch.full((16, 16), 4.0, dtype=torch.float32)
        artifact_map = torch.ones(16, 16, dtype=torch.float32)
        artifact_map[:, 9:] = 0.1

        legacy = direct_landmark_teacher(
            FakeGaussians(),
            query_feature_map,
            pose_gt_w2c=torch.eye(4),
            fovx=0.7610127542247298,
            fovy=0.7610127542247298,
            landmark_indices=torch.tensor([0, 1]),
            target_depth=target_depth,
            alpha_threshold=0.0,
            depth_abs_tolerance=0.05,
            depth_rel_tolerance=0.01,
            max_landmarks=None,
            artifact_weight_map=artifact_map,
        )
        explicit_none = direct_landmark_teacher(
            FakeGaussians(),
            query_feature_map,
            pose_gt_w2c=torch.eye(4),
            fovx=0.7610127542247298,
            fovy=0.7610127542247298,
            landmark_indices=torch.tensor([0, 1]),
            target_depth=target_depth,
            alpha_threshold=0.0,
            depth_abs_tolerance=0.05,
            depth_rel_tolerance=0.01,
            max_landmarks=None,
            artifact_weight_map=artifact_map,
            artifact_image_weight=0.5,
            artifact_weight_combine_mode="product",
            artifact_loss_scale_mode="none",
        )

        self.assertAlmostEqual(explicit_none.desc_loss.item(), legacy.desc_loss.item(), places=6)
        self.assertAlmostEqual(explicit_none.diagnostics["artifact_teacher_loss_scale"], 1.0)

    def test_direct_teacher_can_return_multiview_loss_and_update_memory(self):
        from localization_training.direct_landmark_teacher import LandmarkObservationMemory, direct_landmark_teacher

        class FakeGaussians:
            def __init__(self):
                self._xyz = torch.tensor([[0.0, 0.0, 4.0], [0.5, 0.0, 4.0]], dtype=torch.float32)
                self._loc_feature = torch.nn.Parameter(
                    torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]], dtype=torch.float32)
                )

            @property
            def get_xyz(self):
                return self._xyz

            @property
            def get_loc_feature(self):
                return self._loc_feature

        query_feature_map = torch.zeros(2, 16, 16, dtype=torch.float32)
        query_feature_map[:, 8, 8] = torch.tensor([1.0, 0.0])
        query_feature_map[:, 8, 10] = torch.tensor([0.0, 1.0])
        memory = LandmarkObservationMemory(torch.tensor([0, 1]), feature_dim=2, slots=2, device="cpu")

        out = direct_landmark_teacher(
            FakeGaussians(),
            query_feature_map,
            pose_gt_w2c=torch.eye(4),
            fovx=0.7610127542247298,
            fovy=0.7610127542247298,
            landmark_indices=torch.tensor([0, 1]),
            target_depth=torch.full((16, 16), 4.0, dtype=torch.float32),
            alpha_threshold=0.0,
            depth_abs_tolerance=0.05,
            depth_rel_tolerance=0.01,
            max_landmarks=None,
            multiview_memory=memory,
            multiview_temperature=0.1,
        )

        self.assertGreaterEqual(out.multiview_loss.item(), 0.0)
        self.assertEqual(memory.positive_count(torch.tensor([0, 1])).tolist(), [1, 1])

    def test_direct_teacher_returns_full_bank_and_anchor_losses(self):
        from localization_training.direct_landmark_teacher import direct_landmark_teacher

        class FakeGaussians:
            def __init__(self):
                self._xyz = torch.tensor(
                    [
                        [0.0, 0.0, 4.0],
                        [0.5, 0.0, 4.0],
                        [0.0, 0.5, 4.0],
                    ],
                    dtype=torch.float32,
                )
                self._loc_feature = torch.nn.Parameter(
                    torch.tensor(
                        [
                            [[1.0, 0.0]],
                            [[0.0, 1.0]],
                            [[0.7, 0.7]],
                        ],
                        dtype=torch.float32,
                    )
                )

            @property
            def get_xyz(self):
                return self._xyz

            @property
            def get_loc_feature(self):
                return self._loc_feature

        query_feature_map = torch.zeros(2, 16, 16, dtype=torch.float32)
        query_feature_map[:, 8, 8] = torch.tensor([1.0, 0.0])
        query_feature_map[:, 8, 10] = torch.tensor([0.0, 1.0])
        baseline_features = torch.tensor(
            [
                [[1.0, 0.0]],
                [[1.0, 0.0]],
                [[0.7, 0.7]],
            ],
            dtype=torch.float32,
        )

        out = direct_landmark_teacher(
            FakeGaussians(),
            query_feature_map,
            pose_gt_w2c=torch.eye(4),
            fovx=0.7610127542247298,
            fovy=0.7610127542247298,
            landmark_indices=torch.tensor([0, 1]),
            target_depth=torch.full((16, 16), 4.0, dtype=torch.float32),
            alpha_threshold=0.0,
            depth_abs_tolerance=0.05,
            depth_rel_tolerance=0.01,
            max_landmarks=None,
            full_bank_indices=torch.tensor([0, 1, 2]),
            full_bank_temperature=0.2,
            anchor_features=baseline_features,
        )

        self.assertGreater(out.full_bank_loss.item(), 0.0)
        self.assertGreater(out.anchor_loss.item(), 0.4)
        self.assertIn("full_bank_positive_prob", out.stats)
        self.assertIn("anchor_loss", out.stats)

    def test_direct_teacher_supports_matchability_weighted_translation_fisher(self):
        from localization_training.direct_landmark_teacher import direct_landmark_teacher

        class FakeGaussians:
            def __init__(self):
                self._xyz = torch.tensor(
                    [
                        [-0.8, -0.5, 4.0],
                        [0.8, -0.5, 4.5],
                        [-0.7, 0.6, 5.0],
                        [0.9, 0.7, 6.0],
                        [0.0, 0.0, 7.0],
                        [0.3, -0.8, 8.0],
                    ],
                    dtype=torch.float32,
                )
                self._loc_feature = torch.nn.Parameter(torch.eye(6, dtype=torch.float32)[:, None])

            @property
            def get_xyz(self):
                return self._xyz

            @property
            def get_loc_feature(self):
                return self._loc_feature

        gaussians = FakeGaussians()
        feature_map = torch.zeros(6, 32, 32, dtype=torch.float32)
        focal = 16.0 / torch.tan(torch.tensor(0.7610127542247298 / 2.0)).item()
        for index, point in enumerate(gaussians.get_xyz):
            x = int(round(focal * point[0].item() / point[2].item() + 16.0))
            y = int(round(focal * point[1].item() / point[2].item() + 16.0))
            feature_map[index, y, x] = 1.0

        out = direct_landmark_teacher(
            gaussians,
            feature_map,
            pose_gt_w2c=torch.eye(4),
            fovx=0.7610127542247298,
            fovy=0.7610127542247298,
            landmark_indices=torch.arange(6),
            max_landmarks=None,
            full_bank_indices=torch.arange(6),
            full_bank_temperature=0.1,
            full_bank_pose_information_weight=1.0,
            full_bank_pose_information_floor=0.1,
            full_bank_pose_information_mode="conditional_translation",
            full_bank_pose_information_normalization="quantile",
            full_bank_fisher_use_matchability=True,
            full_bank_fisher_matchability_floor=0.05,
            full_bank_fisher_uncertainty_entropy_scale=2.0,
        )

        self.assertEqual(out.diagnostics["pose_information_mode_id"], 4.0)
        self.assertEqual(out.diagnostics["pose_information_uses_matchability"], 1.0)
        self.assertGreater(out.diagnostics["pose_information_translation_min_eigenvalue"], 0.0)
        self.assertGreater(out.diagnostics["pose_information_effective_count"], 0.0)
        self.assertGreaterEqual(out.diagnostics["pose_information_weight_min"], 0.1)
        self.assertTrue(torch.isfinite(out.full_bank_loss))
        out.full_bank_loss.backward()
        self.assertTrue(torch.isfinite(gaussians._loc_feature.grad).all())


if __name__ == "__main__":
    unittest.main()
