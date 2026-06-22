import unittest

import torch


class LocAwareLossesTest(unittest.TestCase):
    def test_hard_negative_ranking_penalizes_confusing_neighbor(self):
        from localization_training.losses import hard_negative_ranking_loss

        loc_features = torch.tensor(
            [
                [1.0, 0.0],
                [0.9, 0.1],
                [0.0, 1.0],
            ]
        )
        prototypes = torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [0.0, 1.0],
            ]
        )
        loss = hard_negative_ranking_loss(loc_features, prototypes, margin=0.2)
        self.assertGreater(loss.item(), 0.0)

        separated = torch.eye(3)
        loss_ok = hard_negative_ranking_loss(separated, separated, margin=0.2)
        self.assertLess(loss_ok.item(), 1e-5)

    def test_hard_negative_ranking_supports_bounded_sampling(self):
        from localization_training.losses import hard_negative_ranking_loss

        torch.manual_seed(0)
        loc_features = torch.randn(5000, 8)
        prototypes = loc_features + 0.01 * torch.randn_like(loc_features)

        loss = hard_negative_ranking_loss(
            loc_features,
            prototypes,
            margin=0.2,
            max_samples=128,
            max_negatives=256,
        )

        self.assertEqual(loss.ndim, 0)
        self.assertTrue(torch.isfinite(loss))

    def test_geometry_anchor_loss_increases_with_drift(self):
        from localization_training.losses import geometry_anchor_loss

        current = {
            "xyz": torch.tensor([[0.1, 0.0, 0.0], [0.0, 0.0, 0.0]]),
            "scaling": torch.tensor([[0.0, 0.1, 0.0], [0.0, 0.0, 0.0]]),
            "rotation": torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]),
        }
        anchor = {
            "xyz": torch.zeros(2, 3),
            "scaling": torch.zeros(2, 3),
            "rotation": torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]),
        }

        drift = geometry_anchor_loss(current, anchor)
        none = geometry_anchor_loss(anchor, anchor)
        self.assertGreater(drift.item(), none.item())
        self.assertEqual(none.item(), 0.0)


if __name__ == "__main__":
    unittest.main()
