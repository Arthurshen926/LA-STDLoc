import unittest
from unittest import mock

import torch


class DetectorSoftTargetsTest(unittest.TestCase):
    def test_soft_gt_map_uses_landmark_utility_and_focal_loss_prefers_target(self):
        from train_detector import generate_soft_gt_map, utility_weighted_detector_loss

        dtype = torch.float32
        feature_map = torch.zeros(4, 16, 16, dtype=dtype)
        xyz = torch.tensor([[0.0, 0.0, 4.0], [0.4, 0.0, 4.0]], dtype=dtype)
        pose = torch.eye(4, dtype=dtype)
        K = torch.tensor([[20.0, 0.0, 8.0], [0.0, 20.0, 8.0], [0.0, 0.0, 1.0]], dtype=dtype)
        utility = torch.tensor([0.2, 1.0], dtype=dtype)

        gt = generate_soft_gt_map(
            xyz,
            feature_map,
            pose,
            K,
            utility=utility,
            sigma=1.0,
        )

        self.assertEqual(gt.shape, (1, 16, 16))
        self.assertGreater(gt[0, 8, 10].item(), gt[0, 8, 8].item())
        good_logits = torch.logit(gt.clamp(1e-4, 1 - 1e-4))
        bad_logits = torch.zeros_like(gt)
        good = utility_weighted_detector_loss(good_logits, gt)
        bad = utility_weighted_detector_loss(bad_logits, gt)
        self.assertLess(good.item(), bad.item())

    def test_soft_gt_map_can_limit_targets_to_highest_utility_landmarks(self):
        from train_detector import generate_soft_gt_map

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

        gt = generate_soft_gt_map(
            xyz,
            feature_map,
            pose,
            K,
            utility=utility,
            sigma=0.75,
            max_landmarks=2,
        )

        self.assertGreater(gt[0, 16, 12].item(), 0.1)
        self.assertLess(gt[0, 16, 16].item(), 0.01)
        self.assertGreater(gt[0, 16, 20].item(), 0.1)

    def test_soft_gt_map_avoids_full_image_meshgrid_allocation(self):
        from train_detector import generate_soft_gt_map

        dtype = torch.float32
        feature_map = torch.zeros(4, 24, 24, dtype=dtype)
        xyz = torch.tensor([[0.0, 0.0, 4.0]], dtype=dtype)
        pose = torch.eye(4, dtype=dtype)
        K = torch.tensor([[20.0, 0.0, 12.0], [0.0, 20.0, 12.0], [0.0, 0.0, 1.0]], dtype=dtype)

        with mock.patch("torch.meshgrid", side_effect=AssertionError("full image grid allocated")):
            gt = generate_soft_gt_map(xyz, feature_map, pose, K, sigma=1.0)

        self.assertGreater(gt[0, 12, 12].item(), 0.0)

    def test_soft_gt_map_limits_splat_to_three_sigma_radius(self):
        from train_detector import generate_soft_gt_map

        dtype = torch.float32
        feature_map = torch.zeros(4, 24, 24, dtype=dtype)
        xyz = torch.tensor([[0.0, 0.0, 4.0]], dtype=dtype)
        pose = torch.eye(4, dtype=dtype)
        K = torch.tensor([[20.0, 0.0, 12.0], [0.0, 20.0, 12.0], [0.0, 0.0, 1.0]], dtype=dtype)

        gt = generate_soft_gt_map(xyz, feature_map, pose, K, sigma=1.0)

        self.assertGreater(gt[0, 12, 15].item(), 0.0)
        self.assertEqual(gt[0, 12, 16].item(), 0.0)

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
        self.assertTrue(torch.equal(restored, mask))
        self.assertIsNone(render_visible_mask_from_cache(cache, "missing", torch.device("cpu")))


if __name__ == "__main__":
    unittest.main()
