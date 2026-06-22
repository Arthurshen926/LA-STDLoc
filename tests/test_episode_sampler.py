import os
import tempfile
import unittest

import torch


class EpisodeSamplerTest(unittest.TestCase):
    def test_pose_noise_and_sparse_cache_round_trip(self):
        from localization_training.episode_sampler import (
            SparsePoseCache,
            apply_pose_noise,
            sample_noise_from_distribution,
        )

        pose = torch.eye(4)
        noisy = apply_pose_noise(pose, torch.tensor([0.1, -0.2, 0.3, 0.01, 0.02, -0.03]))
        self.assertEqual(noisy.shape, (4, 4))
        self.assertFalse(torch.allclose(noisy, pose))

        errors = {
            "translation": torch.tensor([0.01, 0.2, 0.4]),
            "rotation_deg": torch.tensor([0.5, 2.0, 5.0]),
        }
        sampled = sample_noise_from_distribution(errors, quantile=0.5, generator=torch.Generator().manual_seed(0))
        self.assertEqual(sampled.shape, (6,))
        self.assertLessEqual(torch.linalg.norm(sampled[:3]).item(), 0.21)

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "cache.pt")
            cache = SparsePoseCache(path)
            cache.update("image_a", noisy, inliers=12, ae=1.5, te=23.0)
            cache.save()
            restored = SparsePoseCache(path)
            restored.load()
            item = restored.get("image_a")
            self.assertEqual(item["inliers"], 12)
            self.assertTrue(torch.allclose(item["pose_w2c"], noisy))


if __name__ == "__main__":
    unittest.main()
