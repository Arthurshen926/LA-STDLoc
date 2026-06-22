import unittest

import torch


class DenseTeacherLossTest(unittest.TestCase):
    def test_symmetric_descriptor_loss_prefers_identity_matches(self):
        from localization_training.losses import symmetric_descriptor_loss

        rendered = torch.eye(4)
        query = torch.eye(4)
        good = symmetric_descriptor_loss(rendered, query, temperature=0.05)
        bad = symmetric_descriptor_loss(rendered, torch.roll(query, shifts=1, dims=0), temperature=0.05)

        self.assertLess(good.item(), 1e-3)
        self.assertGreater(bad.item(), good.item() + 5.0)

    def test_fine_reprojection_loss_recovers_local_peak(self):
        from localization_training.losses import fine_reprojection_loss

        feature_map = torch.zeros(3, 7, 7)
        rendered = torch.tensor([[1.0, 0.0, 0.0]])
        target_uv = torch.tensor([[4.0, 3.0]])
        feature_map[:, 3, 4] = rendered[0]

        loss, stats = fine_reprojection_loss(
            rendered,
            feature_map,
            target_uv,
            window_radius=1,
            temperature=0.01,
        )

        self.assertLess(loss.item(), 1e-4)
        self.assertTrue(torch.allclose(stats["pred_uv"], target_uv, atol=1e-3))
        self.assertGreater(stats["positive_prob"].item(), 0.99)


if __name__ == "__main__":
    unittest.main()
