import unittest

import torch


class DirectLandmarkTeacherTest(unittest.TestCase):
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

    def test_anchor_loss_penalizes_descriptor_drift_from_baseline(self):
        from localization_training.direct_landmark_teacher import descriptor_anchor_loss

        current = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        baseline = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
        weights = torch.tensor([0.0, 1.0])

        loss = descriptor_anchor_loss(current, baseline, weights=weights)

        self.assertAlmostEqual(loss.item(), 1.0)

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


if __name__ == "__main__":
    unittest.main()
