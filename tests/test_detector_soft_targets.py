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
        self.assertEqual(gt_map[0, 12, 13].item(), 0.0)
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
        self.assertEqual(gt_map[0, 12, 11].item(), 0.0)
        self.assertEqual(weight_map[0, 12, 11].item(), 1.0)
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

    def test_detector_parser_accepts_isolation_sampling_modes(self):
        from train_detector import build_arg_parser

        parser = build_arg_parser()
        args = parser.parse_args(["--sampling_mode", "localization_aware_global"])
        self.assertEqual(args.sampling_mode, "localization_aware_global")

        args = parser.parse_args(["--sampling_mode", "localization_aware_spatial"])
        self.assertEqual(args.sampling_mode, "localization_aware_spatial")


if __name__ == "__main__":
    unittest.main()
