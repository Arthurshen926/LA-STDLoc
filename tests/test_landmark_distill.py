import unittest
from unittest import mock

import torch


class LandmarkDistillTest(unittest.TestCase):
    def test_localization_aware_sample_prefers_high_utility(self):
        from localization_training.landmark_distill import localization_aware_sample

        xyz = torch.arange(10, dtype=torch.float32)[:, None].repeat(1, 3)
        base_score = torch.zeros(10)
        utility = torch.arange(10, dtype=torch.float32)
        with mock.patch("torch.randperm", return_value=torch.tensor([0, 4, 8, 1, 2, 3, 5, 6, 7, 9])):
            sampled, meta = localization_aware_sample(
                xyz,
                base_score,
                utility,
                num=3,
                k=4,
                min_observations=torch.ones(10, dtype=torch.bool),
            )

        self.assertEqual(sampled.numel(), 3)
        self.assertEqual(set(sampled.tolist()), {3, 6, 9})
        self.assertIn("utility", meta)
        self.assertEqual(meta["indices"].shape, sampled.shape)

    def test_localization_aware_sample_keeps_spatial_neighborhood_coverage(self):
        from localization_training.landmark_distill import localization_aware_sample

        xyz = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.1, 0.0, 0.0],
                [0.2, 0.0, 0.0],
                [10.0, 0.0, 0.0],
                [10.1, 0.0, 0.0],
                [20.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        )
        base_score = torch.zeros(6)
        utility = torch.tensor([100.0, 90.0, 80.0, 10.0, 9.0, 11.0])

        with mock.patch("torch.randperm", return_value=torch.tensor([0, 3, 5, 1, 2, 4])):
            sampled, _ = localization_aware_sample(
                xyz,
                base_score,
                utility,
                num=3,
                k=2,
                min_observations=torch.ones(6, dtype=torch.bool),
            )

        self.assertEqual(set(sampled.tolist()), {0, 3, 5})

    def test_localization_aware_sample_can_reproduce_global_topk_ablation(self):
        from localization_training.landmark_distill import localization_aware_sample

        xyz = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.1, 0.0, 0.0],
                [0.2, 0.0, 0.0],
                [10.0, 0.0, 0.0],
                [10.1, 0.0, 0.0],
                [20.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        )
        base_score = torch.zeros(6)
        utility = torch.tensor([100.0, 90.0, 80.0, 10.0, 9.0, 11.0])

        sampled, _ = localization_aware_sample(
            xyz,
            base_score,
            utility,
            num=3,
            k=2,
            min_observations=torch.ones(6, dtype=torch.bool),
            spatial=False,
        )

        self.assertEqual(set(sampled.tolist()), {0, 1, 2})


if __name__ == "__main__":
    unittest.main()
