import unittest

import torch


class PoseInformationTest(unittest.TestCase):
    def test_information_scores_are_positive_and_report_condition(self):
        from localization_training.pose_information import (
            compute_pose_information,
            pose_jacobian_analytic,
            pose_jacobian_numeric,
        )

        dtype = torch.float64
        points = torch.tensor(
            [
                [-1.0, -0.5, 5.0],
                [1.0, -0.5, 5.0],
                [-1.0, 0.5, 5.0],
                [1.0, 0.5, 5.0],
                [0.0, 0.0, 4.0],
            ],
            dtype=dtype,
        )
        K = torch.tensor(
            [[180.0, 0.0, 80.0], [0.0, 180.0, 60.0], [0.0, 0.0, 1.0]],
            dtype=dtype,
        )
        pose = torch.eye(4, dtype=dtype)

        J = pose_jacobian_numeric(points, K, pose)
        analytic = pose_jacobian_analytic(points, K, pose)
        self.assertEqual(J.shape, (5, 2, 6))
        self.assertTrue(torch.isfinite(J).all())
        self.assertTrue(torch.allclose(analytic, J, rtol=2e-3, atol=2e-3))

        info = compute_pose_information(points, K, pose, weights=torch.ones(5, dtype=dtype))
        self.assertEqual(info.scores.shape, (5,))
        self.assertTrue((info.scores > 0).all())
        self.assertTrue(torch.isfinite(info.condition_number))
        self.assertGreater(info.logdet.item(), 0.0)


if __name__ == "__main__":
    unittest.main()
