import unittest

import numpy as np

from localization_training.scene_normalization import compute_scene_normalization


class SceneNormalizationTests(unittest.TestCase):
    def test_capacity_and_steps_adapt_to_scene_coverage(self):
        small_positions = np.stack(
            [np.linspace(0.0, 5.0, 231), np.zeros(231), np.zeros(231)], axis=1
        )
        large_positions = np.stack(
            [np.linspace(0.0, 25.0, 895), np.zeros(895), np.zeros(895)], axis=1
        )
        surfels = np.full(1000, 0.005)

        small = compute_scene_normalization(
            small_positions, 1202378, surfels, (854, 480)
        )
        large = compute_scene_normalization(
            large_positions, 817587, surfels, (854, 480)
        )

        self.assertEqual(small.landmark_count, 16384)
        self.assertEqual(large.landmark_count, 32768)
        self.assertEqual(small.train_detect_num, 4096)
        self.assertEqual(large.train_detect_num, 4096)
        self.assertEqual(small.full_bank_landmark_count, 16384)
        self.assertEqual(large.full_bank_landmark_count, 16384)
        self.assertLess(small.detector_steps, large.detector_steps)
        self.assertLess(small.candidate_steps, large.candidate_steps)
        self.assertLess(small.mvinit_views, large.mvinit_views)

    def test_metric_and_pixel_parameters_scale_without_scene_names(self):
        positions = np.asarray([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=float)
        base = compute_scene_normalization(
            positions, 10000, np.full(100, 0.01), (640, 480), field_steps=10000
        )
        scaled = compute_scene_normalization(
            positions * 10.0,
            10000,
            np.full(100, 0.1),
            (1280, 960),
            field_steps=10000,
        )

        self.assertAlmostEqual(
            scaled.translation_scale_m / base.translation_scale_m, 10.0, places=5
        )
        self.assertAlmostEqual(
            scaled.surfel_tangent_bound_m / base.surfel_tangent_bound_m,
            10.0,
            places=5,
        )
        self.assertEqual(base.bootstrap_steps, 1000)
        self.assertEqual(base.joint_steps, 5000)
        self.assertEqual(base.pixel_scale, scaled.pixel_scale)
        self.assertAlmostEqual(
            base.geometry_xyz_lr,
            base.surfel_tangent_bound_m * 1e-5,
        )
        self.assertAlmostEqual(
            base.loc_anchor_lr,
            base.surfel_tangent_bound_m * 1e-4,
        )


if __name__ == "__main__":
    unittest.main()
