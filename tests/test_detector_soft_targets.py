import unittest
from unittest import mock

import torch


class DetectorSoftTargetsTest(unittest.TestCase):
    def test_weighted_hard_gt_map_uses_landmark_utility_only_as_loss_weight(self):
        from train_detector import generate_weighted_hard_gt_map, utility_weighted_detector_loss

        dtype = torch.float32
        feature_map = torch.zeros(4, 16, 16, dtype=dtype)
        xyz = torch.tensor([[0.0, 0.0, 4.0], [0.4, 0.0, 4.0]], dtype=dtype)
        pose = torch.eye(4, dtype=dtype)
        K = torch.tensor([[20.0, 0.0, 8.0], [0.0, 20.0, 8.0], [0.0, 0.0, 1.0]], dtype=dtype)
        utility = torch.tensor([0.2, 1.0], dtype=dtype)

        gt, weight = generate_weighted_hard_gt_map(
            xyz,
            feature_map,
            pose,
            K,
            utility=utility,
        )

        self.assertEqual(gt.shape, (1, 16, 16))
        self.assertEqual(gt[0, 8, 8].item(), 1.0)
        self.assertEqual(gt[0, 8, 10].item(), 1.0)
        self.assertEqual(gt[0, 8, 9].item(), 0.0)
        self.assertGreater(weight[0, 8, 10].item(), weight[0, 8, 8].item())
        good_logits = torch.logit(gt.clamp(1e-4, 1 - 1e-4))
        bad_logits = torch.zeros_like(gt)
        good = utility_weighted_detector_loss(good_logits, gt, weight_map=weight)
        bad = utility_weighted_detector_loss(bad_logits, gt, weight_map=weight)
        self.assertLess(good.item(), bad.item())

    def test_weighted_hard_gt_map_keeps_low_utility_landmarks(self):
        from train_detector import generate_weighted_hard_gt_map

        dtype = torch.float32
        feature_map = torch.zeros(4, 32, 32, dtype=dtype)
        xyz = torch.tensor(
            [
                [-0.8, 0.0, 4.0],
                [0.0, 0.0, 4.0],
                [0.8, 0.0, 4.0],
            ],
            dtype=dtype,
        )
        pose = torch.eye(4, dtype=dtype)
        K = torch.tensor([[20.0, 0.0, 16.0], [0.0, 20.0, 16.0], [0.0, 0.0, 1.0]], dtype=dtype)
        utility = torch.tensor([0.9, 0.05, 1.0], dtype=dtype)

        gt, weight = generate_weighted_hard_gt_map(
            xyz,
            feature_map,
            pose,
            K,
            utility=utility,
        )

        self.assertEqual(gt[0, 16, 12].item(), 1.0)
        self.assertEqual(gt[0, 16, 16].item(), 1.0)
        self.assertEqual(gt[0, 16, 20].item(), 1.0)
        self.assertGreater(weight[0, 16, 20].item(), weight[0, 16, 16].item())

    def test_weighted_hard_gt_map_avoids_full_image_meshgrid_allocation(self):
        from train_detector import generate_weighted_hard_gt_map

        dtype = torch.float32
        feature_map = torch.zeros(4, 24, 24, dtype=dtype)
        xyz = torch.tensor([[0.0, 0.0, 4.0]], dtype=dtype)
        pose = torch.eye(4, dtype=dtype)
        K = torch.tensor([[20.0, 0.0, 12.0], [0.0, 20.0, 12.0], [0.0, 0.0, 1.0]], dtype=dtype)

        with mock.patch("torch.meshgrid", side_effect=AssertionError("full image grid allocated")):
            gt, _ = generate_weighted_hard_gt_map(xyz, feature_map, pose, K)

        self.assertEqual(gt[0, 12, 12].item(), 1.0)

    def test_weighted_hard_gt_map_does_not_splat_neighbors(self):
        from train_detector import generate_weighted_hard_gt_map

        dtype = torch.float32
        feature_map = torch.zeros(4, 24, 24, dtype=dtype)
        xyz = torch.tensor([[0.0, 0.0, 4.0]], dtype=dtype)
        pose = torch.eye(4, dtype=dtype)
        K = torch.tensor([[20.0, 0.0, 12.0], [0.0, 20.0, 12.0], [0.0, 0.0, 1.0]], dtype=dtype)

        gt, _ = generate_weighted_hard_gt_map(xyz, feature_map, pose, K)

        self.assertEqual(gt[0, 12, 12].item(), 1.0)
        self.assertEqual(gt[0, 12, 13].item(), 0.0)

    def test_feature_extraction_for_detector_targets_is_detached(self):
        from train_detector import extract_normalized_feature_map

        class FakeFeatureExtractor(torch.nn.Module):
            def forward(self, image):
                return {"feature_map": image * 2.0}

        image = torch.ones(1, 8, 8, requires_grad=True)
        feature_map = extract_normalized_feature_map(
            FakeFeatureExtractor(),
            image,
            size=(8, 8),
        )

        self.assertEqual(feature_map.shape, (1, 8, 8))
        self.assertFalse(feature_map.requires_grad)

    def test_render_visible_mask_cache_keeps_masks_off_cuda_graph(self):
        from train_detector import render_visible_mask_from_cache, store_render_visible_mask

        cache = {}
        mask = torch.tensor([True, False, True])
        store_render_visible_mask(cache, "frame", mask)

        self.assertEqual(cache["frame"].device.type, "cpu")
        self.assertFalse(cache["frame"].requires_grad)
        restored = render_visible_mask_from_cache(cache, "frame", torch.device("cpu"))
        self.assertEqual(restored.device.type, "cpu")
        self.assertTrue(torch.equal(restored, mask))
        self.assertIsNone(render_visible_mask_from_cache(cache, "missing", torch.device("cpu")))

    def test_detector_target_soft_mode_does_not_require_landmark_meta(self):
        from train_detector import build_detector_target_map

        class FakeGaussians:
            @property
            def get_xyz(self):
                return torch.tensor([[0.0, 0.0, 4.0]], dtype=torch.float32)

        feature_map = torch.zeros(4, 24, 24)
        pose = torch.eye(4)
        K = torch.tensor([[20.0, 0.0, 12.0], [0.0, 20.0, 12.0], [0.0, 0.0, 1.0]])
        idx = torch.tensor([0])

        gt_map, soft_target, weight_map = build_detector_target_map(
            FakeGaussians(),
            feature_map,
            idx,
            pose,
            K,
            detector_target_mode="soft",
            landmark_meta=None,
            soft_sigma=1.0,
        )

        self.assertTrue(soft_target)
        self.assertEqual(gt_map[0, 12, 12].item(), 1.0)
        self.assertGreater(gt_map[0, 12, 13].item(), 0.0)
        self.assertLess(gt_map[0, 12, 13].item(), gt_map[0, 12, 12].item())
        self.assertEqual(weight_map[0, 12, 12].item(), 1.0)

    def test_soft_detector_target_keeps_hard_peaks_and_moves_utility_to_weights(self):
        from train_detector import build_detector_target_map

        class FakeGaussians:
            @property
            def get_xyz(self):
                return torch.tensor(
                    [
                        [-0.4, 0.0, 4.0],
                        [0.0, 0.0, 4.0],
                        [0.4, 0.0, 4.0],
                    ],
                    dtype=torch.float32,
                )

        feature_map = torch.zeros(4, 24, 24)
        pose = torch.eye(4)
        K = torch.tensor([[20.0, 0.0, 12.0], [0.0, 20.0, 12.0], [0.0, 0.0, 1.0]])
        idx = torch.tensor([0, 1, 2])
        landmark_meta = {"utility": torch.tensor([0.1, 10.0, 1.0])}

        gt_map, soft_target, weight_map = build_detector_target_map(
            FakeGaussians(),
            feature_map,
            idx,
            pose,
            K,
            detector_target_mode="soft",
            landmark_meta=landmark_meta,
            soft_sigma=1.0,
        )

        self.assertTrue(soft_target)
        self.assertEqual(gt_map[0, 12, 10].item(), 1.0)
        self.assertEqual(gt_map[0, 12, 12].item(), 1.0)
        self.assertEqual(gt_map[0, 12, 14].item(), 1.0)
        self.assertGreater(gt_map[0, 12, 11].item(), 0.0)
        self.assertGreater(weight_map[0, 12, 11].item(), 1.0)
        self.assertGreater(weight_map[0, 12, 12].item(), weight_map[0, 12, 10].item())

    def test_soft_detector_target_penalizes_bad_geometric_landmark_quality(self):
        from train_detector import build_detector_target_map

        class FakeGaussians:
            @property
            def get_xyz(self):
                return torch.tensor(
                    [
                        [-0.4, 0.0, 4.0],
                        [0.0, 0.0, 4.0],
                        [0.4, 0.0, 4.0],
                    ],
                    dtype=torch.float32,
                )

        feature_map = torch.zeros(4, 24, 24)
        pose = torch.eye(4)
        K = torch.tensor([[20.0, 0.0, 12.0], [0.0, 20.0, 12.0], [0.0, 0.0, 1.0]])
        idx = torch.tensor([0, 1, 2])
        landmark_meta = {
            "utility": torch.tensor([1.0, 1.0, 1.0]),
            "reproj_error": torch.tensor([1.0, 64.0, 2.0]),
            "information": torch.tensor([1.0, 1.0, 0.25]),
        }

        _, soft_target, weight_map = build_detector_target_map(
            FakeGaussians(),
            feature_map,
            idx,
            pose,
            K,
            detector_target_mode="soft",
            landmark_meta=landmark_meta,
            soft_sigma=1.0,
        )

        self.assertTrue(soft_target)
        self.assertGreater(weight_map[0, 12, 10].item(), weight_map[0, 12, 12].item())
        self.assertGreater(weight_map[0, 12, 10].item(), weight_map[0, 12, 14].item())

    def test_detector_landmark_quality_uses_clean_pose_and_balance_meta(self):
        from train_detector import detector_landmark_quality_from_meta

        meta = {
            "utility": torch.tensor([1.0, 1.0, 1.0]),
            "raw_gt_precision_2px": torch.tensor([0.9, 0.2, 0.9]),
            "inlier_gt_precision_2px": torch.tensor([0.8, 0.8, 0.2]),
            "pose_min_eig": torch.tensor([1.0, 1.0, 0.1]),
            "depth_balance": torch.tensor([1.0, 0.25, 1.0]),
            "spatial_balance": torch.tensor([1.0, 1.0, 0.25]),
        }

        quality = detector_landmark_quality_from_meta(meta, 3)

        self.assertGreater(quality[0].item(), quality[1].item())
        self.assertGreater(quality[0].item(), quality[2].item())

    def test_final_candidate_quality_requires_clean_pose_and_balanced_landmarks(self):
        from train_detector import final_candidate_quality_from_meta

        meta = {
            "utility": torch.tensor([10.0, 10.0, 10.0, 0.2]),
            "reproj_error": torch.tensor([0.5, 32.0, 0.5, 0.5]),
            "information": torch.tensor([1.0, 1.0, 0.05, 1.0]),
            "spatial_balance": torch.tensor([1.0, 1.0, 1.0, 0.25]),
            "depth_balance": torch.tensor([1.0, 1.0, 1.0, 0.25]),
        }

        quality, components = final_candidate_quality_from_meta(meta, 4)

        self.assertGreater(quality[0].item(), quality[1].item())
        self.assertGreater(quality[0].item(), quality[2].item())
        self.assertGreater(quality[0].item(), quality[3].item())
        self.assertIn("candidate_cleanliness", components)
        self.assertIn("pose_info_contribution", components)
        self.assertIn("candidate_balance", components)

    def test_coverage_sampling_uses_final_candidate_quality_for_pose_quota(self):
        from localization_training.landmark_distill import coverage_preserving_sample
        from train_detector import final_candidate_quality_from_meta

        n = 6
        xyz = torch.stack(
            [torch.arange(n, dtype=torch.float32), torch.zeros(n), torch.ones(n) * 4.0],
            dim=1,
        )
        base_score = torch.ones(n)
        meta = {
            "utility": torch.tensor([20.0, 18.0, 16.0, 0.2, 0.1, 0.1]),
            "reproj_error": torch.tensor([48.0, 40.0, 36.0, 0.5, 0.5, 0.5]),
            "information": torch.tensor([1.0, 1.0, 1.0, 1.0, 0.2, 0.1]),
            "spatial_balance": torch.ones(n),
            "depth_balance": torch.ones(n),
        }
        quality, _ = final_candidate_quality_from_meta(meta, n)

        sampled, sample_meta = coverage_preserving_sample(
            xyz,
            base_score,
            quality,
            num=4,
            min_observations=torch.ones(n, dtype=torch.bool),
            base_preserve_ratio=0.0,
            utility_preserve_ratio=1.0,
            voxel_size=0.01,
            max_per_voxel=99,
        )

        self.assertEqual(int(sample_meta["source_pose_useful_count"].item()), 4)
        self.assertIn(3, set(sampled.tolist()))
        self.assertNotIn(0, set(sampled.tolist()))

    def test_detector_landmark_quality_infers_balance_from_coverage_meta(self):
        from train_detector import detector_landmark_quality_from_meta

        meta = {
            "utility": torch.ones(4),
            "coverage_uv": torch.tensor(
                [
                    [4.0, 4.0],
                    [5.0, 5.0],
                    [6.0, 6.0],
                    [80.0, 80.0],
                ]
            ),
            "coverage_image_size": torch.tensor([100, 100]),
            "coverage_grid_size": torch.tensor(2),
            "coverage_depth": torch.tensor([2.0, 2.1, 2.2, 10.0]),
            "coverage_depth_bins": torch.tensor(2),
        }

        quality = detector_landmark_quality_from_meta(meta, 4)

        self.assertGreater(quality[3].item(), quality[0].item())
        self.assertGreater(quality[3].item(), quality[1].item())
        self.assertGreater(quality[3].item(), quality[2].item())

    def test_detector_landmark_quality_accepts_cuda_coverage_meta(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA is not available")
        from train_detector import detector_landmark_quality_from_meta

        meta = {
            "utility": torch.ones(2, device="cuda"),
            "coverage_uv": torch.tensor([[4.0, 4.0], [80.0, 80.0]], device="cuda"),
            "coverage_image_size": torch.tensor([100, 100], device="cuda"),
            "coverage_grid_size": torch.tensor(2, device="cuda"),
            "coverage_depth": torch.tensor([2.0, 10.0], device="cuda"),
            "coverage_depth_bins": torch.tensor(2, device="cuda"),
        }

        quality = detector_landmark_quality_from_meta(meta, 2)

        self.assertEqual(quality.device.type, "cuda")
        self.assertTrue(torch.isfinite(quality).all())

    def test_soft_detector_target_backward_survives_non_inplace_masking(self):
        from train_detector import detector_target_loss, generate_soft_gt_map

        xyz = torch.tensor(
            [[0.0, 0.0, 4.0], [0.4, 0.0, 4.0]],
            dtype=torch.float32,
            requires_grad=True,
        )
        feature_map = torch.zeros(4, 24, 24)
        pose = torch.eye(4)
        K = torch.tensor([[20.0, 0.0, 12.0], [0.0, 20.0, 12.0], [0.0, 0.0, 1.0]])

        gt_map, weight_map = generate_soft_gt_map(
            xyz,
            feature_map,
            pose,
            K,
            utility=torch.tensor([1.0, 2.0]),
            soft_sigma=1.0,
        )
        gt_map = gt_map * torch.ones_like(gt_map, dtype=torch.bool)
        logits = torch.zeros_like(gt_map, requires_grad=True)
        loss = detector_target_loss(logits, gt_map, soft_target=True, weight_map=weight_map)

        loss.backward()

        self.assertIsNotNone(logits.grad)
        self.assertFalse(gt_map.requires_grad)
        self.assertIsNone(xyz.grad)

    def test_weighted_hard_detector_target_is_explicit_conservative_mode(self):
        from train_detector import build_detector_target_map

        class FakeGaussians:
            @property
            def get_xyz(self):
                return torch.tensor(
                    [
                        [-0.4, 0.0, 4.0],
                        [0.0, 0.0, 4.0],
                    ],
                    dtype=torch.float32,
                )

        feature_map = torch.zeros(4, 24, 24)
        pose = torch.eye(4)
        K = torch.tensor([[20.0, 0.0, 12.0], [0.0, 20.0, 12.0], [0.0, 0.0, 1.0]])
        idx = torch.tensor([0, 1])
        landmark_meta = {"utility": torch.tensor([0.1, 10.0])}

        gt_map, weighted_target, weight_map = build_detector_target_map(
            FakeGaussians(),
            feature_map,
            idx,
            pose,
            K,
            detector_target_mode="weighted_hard",
            landmark_meta=landmark_meta,
        )

        self.assertTrue(weighted_target)
        self.assertEqual(gt_map[0, 12, 10].item(), 1.0)
        self.assertEqual(gt_map[0, 12, 12].item(), 1.0)
        self.assertEqual(gt_map[0, 12, 11].item(), 0.0)
        self.assertGreater(weight_map[0, 12, 12].item(), weight_map[0, 12, 10].item())

    def test_detector_loss_uses_focal_for_soft_targets(self):
        from train_detector import detector_target_loss

        target = torch.zeros(1, 8, 8)
        target[0, 4, 4] = 1.0
        logits = torch.zeros_like(target)

        with mock.patch("train_detector.utility_weighted_detector_loss", return_value=torch.tensor(3.0)) as focal:
            loss = detector_target_loss(logits, target, soft_target=True)

        focal.assert_called_once()
        self.assertEqual(loss.item(), 3.0)

    def test_detector_observed_mask_intersects_coverage_observations(self):
        from train_detector import detector_sampling_observed_mask

        loc_observation_count = torch.tensor([4, 4, 4, 2])
        coverage_stats = {"observed": torch.tensor([True, False, True, True])}

        observed = detector_sampling_observed_mask(
            loc_observation_count,
            min_loc_observations=4,
            coverage_stats=coverage_stats,
        )

        self.assertEqual(observed.tolist(), [True, False, True, False])

    def test_detector_parser_accepts_isolation_sampling_modes(self):
        from train_detector import build_arg_parser

        parser = build_arg_parser()
        args = parser.parse_args(["--sampling_mode", "localization_aware_global"])
        self.assertEqual(args.sampling_mode, "localization_aware_global")

        args = parser.parse_args(["--sampling_mode", "localization_aware_spatial"])
        self.assertEqual(args.sampling_mode, "localization_aware_spatial")

    def test_detector_parser_accepts_pnp_aware_sampling_mode(self):
        from train_detector import build_arg_parser

        parser = build_arg_parser()
        args = parser.parse_args(["--sampling_mode", "localization_aware_pnp"])

        self.assertEqual(args.sampling_mode, "localization_aware_pnp")

    def test_detector_parser_accepts_coverage_preserving_sampling_and_weighted_hard_target(self):
        from train_detector import build_arg_parser

        parser = build_arg_parser()
        args = parser.parse_args(
            [
                "--sampling_mode",
                "coverage_preserving",
                "--detector_target_mode",
                "weighted_hard",
                "--coverage_preserve_ratio",
                "0.5",
                "--coverage_utility_ratio",
                "0.25",
                "--coverage_high_confidence_ratio",
                "0.25",
                "--coverage_grid_size",
                "4",
                "--coverage_max_per_grid",
                "2",
                "--coverage_depth_bins",
                "3",
                "--coverage_max_per_depth_bin",
                "3",
                "--candidate_reprojection_error_scale",
                "2.0",
                "--candidate_cleanliness_weight",
                "1.5",
                "--candidate_pose_info_weight",
                "2.0",
                "--candidate_balance_weight",
                "0.75",
                "--candidate_reliability_weight",
                "0.1",
                "--candidate_utility_weight",
                "0.0",
            ]
        )

        self.assertEqual(args.sampling_mode, "coverage_preserving")
        self.assertEqual(args.detector_target_mode, "weighted_hard")
        self.assertAlmostEqual(args.coverage_preserve_ratio, 0.5)
        self.assertAlmostEqual(args.coverage_utility_ratio, 0.25)
        self.assertAlmostEqual(args.coverage_high_confidence_ratio, 0.25)
        self.assertEqual(args.coverage_grid_size, 4)
        self.assertEqual(args.coverage_max_per_grid, 2)
        self.assertEqual(args.coverage_depth_bins, 3)
        self.assertEqual(args.coverage_max_per_depth_bin, 3)
        self.assertAlmostEqual(args.candidate_reprojection_error_scale, 2.0)
        self.assertAlmostEqual(args.candidate_cleanliness_weight, 1.5)
        self.assertAlmostEqual(args.candidate_pose_info_weight, 2.0)
        self.assertAlmostEqual(args.candidate_balance_weight, 0.75)
        self.assertAlmostEqual(args.candidate_reliability_weight, 0.1)
        self.assertAlmostEqual(args.candidate_utility_weight, 0.0)

    def test_coverage_preserving_sample_keeps_stable_landmarks_and_limits_utility_cluster(self):
        from localization_training.landmark_distill import coverage_preserving_sample

        xyz = torch.tensor(
            [
                [0.0, 0.0, 4.0],
                [1.0, 0.0, 4.0],
                [0.0, 1.0, 4.0],
                [1.0, 1.0, 4.0],
                [5.00, 5.00, 4.0],
                [5.05, 5.00, 4.0],
                [5.00, 5.05, 4.0],
                [5.05, 5.05, 4.0],
            ],
            dtype=torch.float32,
        )
        base_score = torch.tensor([0.99, 0.98, 0.97, 0.96, 0.1, 0.1, 0.1, 0.1])
        utility = torch.tensor([0.1, 0.1, 0.1, 0.1, 10.0, 9.0, 8.0, 7.0])

        sampled, meta = coverage_preserving_sample(
            xyz,
            base_score,
            utility,
            num=6,
            k=1,
            min_observations=torch.ones(8, dtype=torch.bool),
            base_preserve_ratio=0.5,
            utility_preserve_ratio=0.25,
            voxel_size=0.25,
            max_per_voxel=1,
        )

        sampled_set = set(sampled.tolist())
        self.assertTrue({0, 1, 2}.issubset(sampled_set))
        self.assertLessEqual(len(sampled_set & {4, 5, 6, 7}), 2)
        self.assertEqual(int(meta["source_visible_stable_count"].item()), 3)
        self.assertEqual(int(meta["source_pose_useful_count"].item()), 2)

    def test_coverage_preserving_sample_balances_2d_grid_depth_and_high_confidence_quota(self):
        from localization_training.landmark_distill import coverage_preserving_sample

        xyz = torch.tensor(
            [
                [0.0, 0.0, 4.0],
                [1.0, 0.0, 4.0],
                [0.0, 1.0, 4.0],
                [1.0, 1.0, 4.0],
                [5.00, 5.00, 2.0],
                [5.05, 5.00, 2.1],
                [5.10, 5.00, 2.2],
                [5.15, 5.00, 2.3],
                [2.0, 5.0, 9.0],
                [2.0, 6.0, 10.0],
                [8.0, 0.0, 6.0],
                [8.0, 1.0, 6.5],
            ],
            dtype=torch.float32,
        )
        uv = torch.tensor(
            [
                [4.0, 4.0],
                [20.0, 4.0],
                [4.0, 20.0],
                [20.0, 20.0],
                [2.0, 2.0],
                [3.0, 2.0],
                [2.0, 3.0],
                [3.0, 3.0],
                [28.0, 28.0],
                [30.0, 30.0],
                [12.0, 12.0],
                [28.0, 4.0],
            ],
            dtype=torch.float32,
        )
        depth = xyz[:, 2]
        base_score = torch.tensor([0.99, 0.98, 0.97, 0.96, 0.1, 0.1, 0.1, 0.1, 0.4, 0.3, 0.05, 0.04])
        utility = torch.tensor([0.1, 0.1, 0.1, 0.1, 10.0, 9.0, 8.0, 7.0, 0.2, 0.2, 0.05, 0.05])
        high_confidence = torch.tensor([0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.2, 12.0, 11.0, 0.1, 0.1])

        sampled, meta = coverage_preserving_sample(
            xyz,
            base_score,
            utility,
            num=8,
            min_observations=torch.ones(12, dtype=torch.bool),
            base_preserve_ratio=0.5,
            utility_preserve_ratio=0.25,
            high_confidence=high_confidence,
            high_confidence_ratio=0.25,
            uv=uv,
            image_size=(32, 32),
            grid_size=4,
            max_per_grid=2,
            depth=depth,
            depth_bins=3,
            max_per_depth_bin=4,
            voxel_size=0.25,
            max_per_voxel=1,
        )

        sampled_set = set(sampled.tolist())
        self.assertTrue({8, 9}.issubset(sampled_set))
        self.assertLessEqual(len(sampled_set & {4, 5, 6, 7}), 2)
        self.assertEqual(int(meta["source_high_confidence_count"].item()), 2)
        self.assertIn("coverage_uv", meta)
        self.assertIn("coverage_depth", meta)

    def test_coverage_preserving_reserves_pose_useful_quota_before_stable_points_saturate_caps(self):
        from localization_training.landmark_distill import coverage_preserving_sample

        n = 6
        xyz = torch.stack(
            [torch.arange(n, dtype=torch.float32), torch.zeros(n), torch.ones(n) * 4.0],
            dim=1,
        )
        uv = torch.tensor([[2.0, 2.0]] * n, dtype=torch.float32)
        base_score = torch.tensor([6.0, 5.0, 4.0, 3.0, 0.1, 0.1])
        utility = torch.tensor([0.1, 0.1, 0.1, 0.1, 10.0, 9.0])

        sampled, meta = coverage_preserving_sample(
            xyz,
            base_score,
            utility,
            num=4,
            min_observations=torch.ones(n, dtype=torch.bool),
            base_preserve_ratio=0.75,
            utility_preserve_ratio=0.25,
            uv=uv,
            image_size=(32, 32),
            grid_size=4,
            max_per_grid=2,
            voxel_size=0.01,
            max_per_voxel=99,
        )

        sampled_set = set(sampled.tolist())
        self.assertEqual(int(meta["source_pose_useful_count"].item()), 1)
        self.assertIn(4, sampled_set)
        self.assertLessEqual(int(sampled.numel()), 2)

    def test_coverage_preserving_reserves_high_confidence_quota_before_stable_points_saturate_caps(self):
        from localization_training.landmark_distill import coverage_preserving_sample

        n = 6
        xyz = torch.stack(
            [torch.arange(n, dtype=torch.float32), torch.zeros(n), torch.ones(n) * 4.0],
            dim=1,
        )
        uv = torch.tensor([[2.0, 2.0]] * n, dtype=torch.float32)
        base_score = torch.tensor([6.0, 5.0, 4.0, 3.0, 0.1, 0.1])
        utility = torch.tensor([0.1, 0.1, 0.1, 0.1, 10.0, 0.1])
        high_confidence = torch.tensor([0.1, 0.1, 0.1, 0.1, 0.1, 10.0])

        sampled, meta = coverage_preserving_sample(
            xyz,
            base_score,
            utility,
            num=5,
            min_observations=torch.ones(n, dtype=torch.bool),
            base_preserve_ratio=0.6,
            utility_preserve_ratio=0.2,
            high_confidence=high_confidence,
            high_confidence_ratio=0.2,
            uv=uv,
            image_size=(32, 32),
            grid_size=4,
            max_per_grid=3,
            voxel_size=0.01,
            max_per_voxel=99,
        )

        sampled_set = set(sampled.tolist())
        self.assertEqual(int(meta["source_pose_useful_count"].item()), 1)
        self.assertEqual(int(meta["source_high_confidence_count"].item()), 1)
        self.assertIn(4, sampled_set)
        self.assertIn(5, sampled_set)
        self.assertLessEqual(int(sampled.numel()), 3)

    def test_coverage_preserving_sample_balances_clustered_stable_quota(self):
        from localization_training.landmark_distill import coverage_preserving_sample

        n = 14
        xyz = torch.stack(
            [torch.arange(n, dtype=torch.float32), torch.zeros(n), torch.ones(n) * 4.0],
            dim=1,
        )
        uv = torch.tensor(
            [[2.0, 2.0]] * 8
            + [[20.0, 2.0], [22.0, 2.0], [2.0, 20.0], [2.0, 22.0], [22.0, 22.0], [24.0, 22.0]],
            dtype=torch.float32,
        )
        base_score = torch.tensor(
            [14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]
        )
        utility = torch.zeros(n)

        sampled, _ = coverage_preserving_sample(
            xyz,
            base_score,
            utility,
            num=8,
            min_observations=torch.ones(n, dtype=torch.bool),
            base_preserve_ratio=0.75,
            utility_preserve_ratio=0.0,
            uv=uv,
            image_size=(32, 32),
            grid_size=4,
            max_per_grid=2,
            voxel_size=0.01,
            max_per_voxel=99,
        )

        cell_x = (uv[sampled, 0].clamp(0, 31) / 32 * 4).floor().long().clamp(0, 3)
        cell_y = (uv[sampled, 1].clamp(0, 31) / 32 * 4).floor().long().clamp(0, 3)
        cell = cell_y * 4 + cell_x
        counts = torch.bincount(cell, minlength=16)

        self.assertLessEqual(int(counts.max().item()), 2)

    def test_coverage_preserving_fallback_keeps_grid_cap_when_relaxing_voxel_cap(self):
        from localization_training.landmark_distill import coverage_preserving_sample

        n = 8
        xyz = torch.zeros(n, 3, dtype=torch.float32)
        xyz[:, 2] = 4.0
        uv = torch.tensor(
            [
                [2.0, 2.0],
                [3.0, 2.0],
                [2.0, 3.0],
                [3.0, 3.0],
                [20.0, 2.0],
                [22.0, 2.0],
                [2.0, 20.0],
                [22.0, 22.0],
            ],
            dtype=torch.float32,
        )
        base_score = torch.tensor([8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0])
        utility = torch.zeros(n)

        sampled, _ = coverage_preserving_sample(
            xyz,
            base_score,
            utility,
            num=6,
            min_observations=torch.ones(n, dtype=torch.bool),
            base_preserve_ratio=1.0,
            utility_preserve_ratio=0.0,
            uv=uv,
            image_size=(32, 32),
            grid_size=4,
            max_per_grid=2,
            depth=xyz[:, 2],
            depth_bins=2,
            max_per_depth_bin=6,
            voxel_size=0.25,
            max_per_voxel=1,
        )

        cell_x = (uv[sampled, 0].clamp(0, 31) / 32 * 4).floor().long().clamp(0, 3)
        cell_y = (uv[sampled, 1].clamp(0, 31) / 32 * 4).floor().long().clamp(0, 3)
        cell = cell_y * 4 + cell_x
        counts = torch.bincount(cell, minlength=16)

        self.assertEqual(int(sampled.numel()), 6)
        self.assertLessEqual(int(counts.max().item()), 2)

    def test_coverage_preserving_does_not_break_grid_cap_to_force_requested_count(self):
        from localization_training.landmark_distill import coverage_preserving_sample

        n = 8
        xyz = torch.stack(
            [torch.arange(n, dtype=torch.float32), torch.zeros(n), torch.ones(n) * 4.0],
            dim=1,
        )
        uv = torch.tensor([[2.0, 2.0]] * n, dtype=torch.float32)
        base_score = torch.arange(float(n), 0.0, -1.0)
        utility = torch.zeros(n)

        sampled, meta = coverage_preserving_sample(
            xyz,
            base_score,
            utility,
            num=6,
            min_observations=torch.ones(n, dtype=torch.bool),
            base_preserve_ratio=1.0,
            utility_preserve_ratio=0.0,
            uv=uv,
            image_size=(32, 32),
            grid_size=4,
            max_per_grid=2,
            voxel_size=0.01,
            max_per_voxel=99,
        )

        self.assertEqual(int(sampled.numel()), 2)
        self.assertEqual(int(meta["source_fallback_count"].item()), 0)
        self.assertEqual(int(meta["coverage_requested_count"].item()), 6)
        self.assertEqual(int(meta["coverage_underfill_count"].item()), 4)

    def test_coverage_preserving_can_explicitly_allow_unbalanced_fallback(self):
        from localization_training.landmark_distill import coverage_preserving_sample

        n = 8
        xyz = torch.stack(
            [torch.arange(n, dtype=torch.float32), torch.zeros(n), torch.ones(n) * 4.0],
            dim=1,
        )
        uv = torch.tensor([[2.0, 2.0]] * n, dtype=torch.float32)
        base_score = torch.arange(float(n), 0.0, -1.0)
        utility = torch.zeros(n)

        sampled, meta = coverage_preserving_sample(
            xyz,
            base_score,
            utility,
            num=6,
            min_observations=torch.ones(n, dtype=torch.bool),
            base_preserve_ratio=1.0,
            utility_preserve_ratio=0.0,
            uv=uv,
            image_size=(32, 32),
            grid_size=4,
            max_per_grid=2,
            voxel_size=0.01,
            max_per_voxel=99,
            allow_unbalanced_fallback=True,
        )

        self.assertEqual(int(sampled.numel()), 6)
        self.assertEqual(int(meta["source_fallback_count"].item()), 4)
        self.assertEqual(int(meta["coverage_underfill_count"].item()), 0)

    def test_coverage_preserving_meta_records_geometry_diagnostics(self):
        from localization_training.landmark_distill import coverage_preserving_sample

        n = 8
        xyz = torch.zeros(n, 3, dtype=torch.float32)
        xyz[:, 2] = torch.tensor([2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
        uv = torch.tensor(
            [
                [2.0, 2.0],
                [3.0, 2.0],
                [2.0, 3.0],
                [3.0, 3.0],
                [20.0, 2.0],
                [22.0, 2.0],
                [2.0, 20.0],
                [22.0, 22.0],
            ],
            dtype=torch.float32,
        )
        base_score = torch.arange(float(n), 0.0, -1.0)
        utility = torch.zeros(n)

        _, meta = coverage_preserving_sample(
            xyz,
            base_score,
            utility,
            num=6,
            min_observations=torch.ones(n, dtype=torch.bool),
            base_preserve_ratio=1.0,
            utility_preserve_ratio=0.0,
            uv=uv,
            image_size=(32, 48),
            grid_size=4,
            max_per_grid=2,
            depth=xyz[:, 2],
            depth_bins=2,
            max_per_depth_bin=6,
            voxel_size=0.25,
            max_per_voxel=1,
        )

        self.assertTrue(torch.equal(meta["coverage_image_size"], torch.tensor([32, 48], device=base_score.device)))
        self.assertAlmostEqual(float(meta["coverage_depth_min"].item()), 2.0)
        self.assertAlmostEqual(float(meta["coverage_depth_max"].item()), 9.0)
        self.assertIn("source_relaxed_fill_count", meta)
        self.assertIn("source_fallback_count", meta)
        accounted = (
            int(meta["source_visible_stable_count"].item())
            + int(meta["source_pose_useful_count"].item())
            + int(meta["source_high_confidence_count"].item())
            + int(meta["source_fill_count"].item())
            + int(meta["source_relaxed_fill_count"].item())
            + int(meta["source_fallback_count"].item())
        )
        self.assertEqual(accounted, 6)

    def test_detector_parser_accepts_landmark_only_bootstrap(self):
        from train_detector import build_arg_parser

        parser = build_arg_parser()
        args = parser.parse_args(["--landmark_only", "--iteration", "0", "--detector_folder", "detector_bootstrap"])

        self.assertTrue(args.landmark_only)
        self.assertEqual(args.iteration, 0)
        self.assertEqual(args.detector_folder, "detector_bootstrap")

    def test_detector_parser_accepts_precomputed_landmark_path(self):
        from train_detector import build_arg_parser

        parser = build_arg_parser()
        args = parser.parse_args(["--precomputed_landmark_path", "/tmp/sample.pkl"])

        self.assertEqual(args.precomputed_landmark_path, "/tmp/sample.pkl")

    def test_detector_parser_accepts_sparse_candidate_teacher_controls(self):
        from train_detector import build_arg_parser

        parser = build_arg_parser()
        args = parser.parse_args(
            [
                "--sparse_candidate_teacher",
                "--candidate_teacher_optimize_features",
                "--candidate_teacher_detector_init_path",
                "/tmp/detector.pth",
                "--candidate_teacher_detect_num",
                "4096",
                "--candidate_teacher_assignment_weight",
                "1.5",
                "--candidate_teacher_assignment_margin",
                "0.08",
                "--candidate_teacher_geometry_weight",
                "0.2",
                "--candidate_teacher_support_query_split",
                "--candidate_teacher_split_mode",
                "temporal_block",
            ]
        )

        self.assertTrue(args.sparse_candidate_teacher)
        self.assertTrue(args.candidate_teacher_optimize_features)
        self.assertEqual(args.candidate_teacher_detect_num, 4096)
        self.assertEqual(args.candidate_teacher_assignment_weight, 1.5)
        self.assertEqual(args.candidate_teacher_assignment_margin, 0.08)
        self.assertEqual(args.candidate_teacher_geometry_weight, 0.2)
        self.assertTrue(args.candidate_teacher_support_query_split)

    def test_precomputed_detector_landmarks_are_loaded_and_validated(self):
        import pickle
        import tempfile
        from pathlib import Path

        from train_detector import load_precomputed_detector_landmarks

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sampled_idx.pkl"
            with path.open("wb") as handle:
                pickle.dump([4, 2, 1], handle)

            sampled = load_precomputed_detector_landmarks(str(path), point_count=8)

        self.assertEqual(sampled.dtype, torch.long)
        self.assertEqual(sampled.tolist(), [4, 2, 1])

    def test_precomputed_detector_landmarks_can_be_moved_to_device(self):
        import pickle
        import tempfile
        from pathlib import Path

        from train_detector import load_precomputed_detector_landmarks

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sampled_idx.pkl"
            with path.open("wb") as handle:
                pickle.dump([1, 3], handle)

            sampled = load_precomputed_detector_landmarks(
                str(path),
                point_count=4,
                device=torch.device("cpu"),
            )

        self.assertEqual(sampled.device.type, "cpu")
        self.assertEqual(sampled.tolist(), [1, 3])

    def test_detector_model_defaults_are_filled_for_new_model_without_cfg(self):
        from argparse import Namespace

        from train_detector import fill_missing_model_defaults

        args = Namespace(source_path="/data/scene", model_path="/tmp/new_model")
        fill_missing_model_defaults(args)

        self.assertEqual(args.sh_degree, 3)
        self.assertEqual(args.feature_type, "")
        self.assertEqual(args.gaussian_type, "3dgs")
        self.assertEqual(args.images, "images")
        self.assertEqual(args.resolution, -1)
        self.assertEqual(args.data_device, "cuda")
        self.assertTrue(args.white_background)

    def test_detector_model_defaults_do_not_override_existing_cfg_values(self):
        from argparse import Namespace

        from train_detector import fill_missing_model_defaults

        args = Namespace(
            source_path="/data/scene",
            model_path="/tmp/model",
            sh_degree=2,
            feature_type="sp",
            gaussian_type="2dgs",
            images="processed",
            resolution=1,
            data_device="cpu",
            white_background=False,
        )
        fill_missing_model_defaults(args)

        self.assertEqual(args.sh_degree, 2)
        self.assertEqual(args.feature_type, "sp")
        self.assertEqual(args.gaussian_type, "2dgs")
        self.assertEqual(args.images, "processed")
        self.assertEqual(args.resolution, 1)
        self.assertEqual(args.data_device, "cpu")
        self.assertFalse(args.white_background)

    def test_empty_detector_landmark_sample_fails_before_training(self):
        from train_detector import validate_detector_sampled_indices

        with self.assertRaisesRegex(ValueError, "sampled 0 detector landmarks"):
            validate_detector_sampled_indices(
                torch.empty(0, dtype=torch.long),
                sampling_mode="localization_aware",
                min_loc_observations=4,
            )

    def test_nonempty_detector_landmark_sample_is_returned_as_long_tensor(self):
        from train_detector import validate_detector_sampled_indices

        sampled = validate_detector_sampled_indices(
            [3, 1, 2],
            sampling_mode="baseline",
            min_loc_observations=0,
        )

        self.assertEqual(sampled.dtype, torch.long)
        self.assertEqual(sampled.tolist(), [3, 1, 2])


if __name__ == "__main__":
    unittest.main()
