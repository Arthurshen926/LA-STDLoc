import unittest

import torch


class LandmarkDistillTest(unittest.TestCase):
    def test_localization_aware_sample_prefers_high_utility(self):
        from localization_training.landmark_distill import localization_aware_sample

        xyz = torch.arange(10, dtype=torch.float32)[:, None].repeat(1, 3)
        base_score = torch.zeros(10)
        utility = torch.arange(10, dtype=torch.float32)
        sampled, meta = localization_aware_sample(
            xyz,
            base_score,
            utility,
            num=3,
            k=4,
            min_observations=torch.ones(10, dtype=torch.bool),
        )

        self.assertEqual(sampled.numel(), 3)
        self.assertTrue(set(sampled.tolist()).issubset({7, 8, 9}))
        self.assertIn("utility", meta)
        self.assertEqual(meta["indices"].shape, sampled.shape)


if __name__ == "__main__":
    unittest.main()
