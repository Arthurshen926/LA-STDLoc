import unittest

import torch


class PoseRefinerTest(unittest.TestCase):
    def test_weighted_gauss_newton_reduces_reprojection_error(self):
        from localization_training.pose_refiner import (
            project_points,
            reprojection_rmse,
            weighted_gauss_newton_refine,
        )

        dtype = torch.float64
        K = torch.tensor(
            [[120.0, 0.0, 64.0], [0.0, 120.0, 48.0], [0.0, 0.0, 1.0]],
            dtype=dtype,
        )
        points = torch.tensor(
            [
                [-0.4, -0.2, 4.0],
                [0.4, -0.1, 4.2],
                [-0.2, 0.35, 4.5],
                [0.35, 0.3, 4.1],
                [0.0, 0.0, 3.8],
                [0.6, 0.15, 5.0],
            ],
            dtype=dtype,
        )
        pose_gt = torch.eye(4, dtype=dtype)
        pose_init = torch.eye(4, dtype=dtype)
        pose_init[0, 3] = 0.12
        pose_init[1, 3] = -0.08

        target_uv, valid = project_points(points, K, pose_gt)
        self.assertTrue(valid.all())
        before = reprojection_rmse(points, target_uv, K, pose_init)

        refined, info = weighted_gauss_newton_refine(
            points,
            target_uv,
            K,
            pose_init,
            weights=torch.ones(points.shape[0], dtype=dtype),
            num_iterations=5,
            damping=1e-3,
        )
        after = reprojection_rmse(points, target_uv, K, refined)

        self.assertLess(after.item(), before.item() * 0.2)
        self.assertEqual(info["iterations"], 5)
        self.assertTrue(torch.isfinite(refined).all())


if __name__ == "__main__":
    unittest.main()
