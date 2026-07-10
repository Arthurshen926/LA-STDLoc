import unittest

import torch


class GeometrySelectorTest(unittest.TestCase):
    def test_balanced_selector_enforces_2d_grid_and_3d_voxel_quotas(self):
        from localization_training.geometry_selector import GeometryBalancedSelector

        p2d = torch.tensor(
            [
                [2.0, 2.0],
                [3.0, 3.0],
                [18.0, 2.0],
                [2.0, 18.0],
                [18.0, 18.0],
                [19.0, 19.0],
            ]
        )
        p3d = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.1, 0.1, 0.0],
                [2.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
                [2.0, 2.0, 0.0],
                [2.1, 2.1, 0.0],
            ]
        )
        scores = torch.tensor([0.99, 0.98, 0.8, 0.7, 0.6, 0.5])

        selector = GeometryBalancedSelector(
            image_width=20,
            image_height=20,
            grid_rows=2,
            grid_cols=2,
            max_per_cell=1,
            voxel_size=1.0,
            max_per_voxel=1,
            max_matches=4,
        )

        selected = selector.select(p2d, p3d, scores)

        self.assertEqual(selected.tolist(), [0, 2, 3, 4])

    def test_balanced_selector_with_no_global_limit_still_enforces_quotas(self):
        from localization_training.geometry_selector import GeometryBalancedSelector

        p2d = torch.tensor([[0.0, 0.0], [1.0, 1.0], [8.0, 8.0]])
        p3d = torch.tensor([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [2.0, 2.0, 0.0]])
        scores = torch.tensor([0.1, 0.9, 0.5])

        selector = GeometryBalancedSelector(
            image_width=10,
            image_height=10,
            grid_rows=1,
            grid_cols=1,
            max_per_cell=1,
            voxel_size=1.0,
            max_per_voxel=1,
            max_matches=0,
        )

        selected = selector.select(p2d, p3d, scores)

        self.assertEqual(selected.tolist(), [1])

    def test_pose_informative_inlier_selection_prefers_spatial_baseline(self):
        from localization_training.geometry_selector import GeometryBalancedSelector

        p3d = torch.tensor(
            [
                [0.0, 0.0, 4.0],
                [0.1, 0.0, 4.0],
                [0.2, 0.0, 4.0],
                [2.0, 0.0, 4.0],
                [0.0, 2.0, 4.0],
            ]
        )
        inliers = torch.tensor([0, 1, 2, 3, 4])
        scores = torch.tensor([0.99, 0.98, 0.97, 0.7, 0.6])
        pose_w2c = torch.eye(4)
        intrinsic = torch.tensor(
            [
                [100.0, 0.0, 50.0],
                [0.0, 100.0, 50.0],
                [0.0, 0.0, 1.0],
            ]
        )

        selector = GeometryBalancedSelector(
            image_width=100,
            image_height=100,
            max_matches=0,
            post_max_matches=3,
        )

        selected = selector.select_pose_informative_inliers(
            p3d,
            pose_w2c,
            intrinsic,
            inliers,
            scores=scores,
        )

        self.assertEqual(selected.tolist(), [0, 3, 4])


if __name__ == "__main__":
    unittest.main()
