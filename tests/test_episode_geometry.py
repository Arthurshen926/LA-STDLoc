import unittest

import torch


class EpisodeGeometryTest(unittest.TestCase):
    def test_unproject_then_project_matches_known_transform(self):
        from localization_training.correspondence import (
            project_world_to_pixels,
            unproject_pixels,
        )

        dtype = torch.float64
        uv = torch.tensor([[32.0, 16.0], [40.0, 20.0]], dtype=dtype)
        depth = torch.tensor([4.0, 6.0], dtype=dtype)
        K = torch.tensor(
            [[80.0, 0.0, 32.0], [0.0, 80.0, 16.0], [0.0, 0.0, 1.0]],
            dtype=dtype,
        )

        pose_init = torch.eye(4, dtype=dtype)
        pose_gt = torch.eye(4, dtype=dtype)
        pose_gt[0, 3] = 0.25

        world = unproject_pixels(uv, depth, K, pose_init)
        reproj, valid = project_world_to_pixels(world, K, pose_gt)

        expected_cam = (pose_gt @ torch.cat([world, torch.ones(2, 1, dtype=dtype)], dim=1).T)[:3].T
        expected_uv = torch.stack(
            [
                K[0, 0] * expected_cam[:, 0] / expected_cam[:, 2] + K[0, 2],
                K[1, 1] * expected_cam[:, 1] / expected_cam[:, 2] + K[1, 2],
            ],
            dim=1,
        )

        self.assertTrue(valid.all())
        self.assertTrue(torch.allclose(reproj, expected_uv, atol=1e-9))


if __name__ == "__main__":
    unittest.main()
