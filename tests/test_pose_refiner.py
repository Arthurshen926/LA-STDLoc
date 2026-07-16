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

    def test_task_scaled_gauss_newton_is_scene_scale_invariant(self):
        from localization_training.pose_refiner import (
            project_points,
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
        pose_init[:3, 3] = torch.tensor([0.12, -0.08, 0.04], dtype=dtype)
        target_uv, _ = project_points(points, K, pose_gt)
        rotation_scale = torch.deg2rad(torch.tensor(2.0, dtype=dtype))
        parameter_scale = torch.tensor(
            [0.1, 0.1, 0.1, rotation_scale, rotation_scale, rotation_scale],
            dtype=dtype,
        )

        refined_m, info_m = weighted_gauss_newton_refine(
            points,
            target_uv,
            K,
            pose_init,
            num_iterations=5,
            parameter_scale=parameter_scale,
        )

        world_scale = 100.0
        scaled_pose_init = pose_init.clone()
        scaled_pose_init[:3, 3] *= world_scale
        scaled_parameter_scale = parameter_scale.clone()
        scaled_parameter_scale[:3] *= world_scale
        refined_cm, info_cm = weighted_gauss_newton_refine(
            points * world_scale,
            target_uv,
            K,
            scaled_pose_init,
            num_iterations=5,
            parameter_scale=scaled_parameter_scale,
        )

        torch.testing.assert_close(
            refined_m[:3, 3], refined_cm[:3, 3] / world_scale, rtol=1e-4, atol=1e-6
        )
        torch.testing.assert_close(
            info_m["condition_number"], info_cm["condition_number"], rtol=2e-3, atol=1e-3
        )
        self.assertGreater(
            abs(
                torch.log10(info_m["raw_condition_number"])
                - torch.log10(info_cm["raw_condition_number"])
            ).item(),
            1.0,
        )


if __name__ == "__main__":
    unittest.main()
