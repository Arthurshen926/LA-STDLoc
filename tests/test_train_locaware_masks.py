import unittest
from argparse import ArgumentParser
from types import SimpleNamespace

import torch


class TrainLocawareMaskTest(unittest.TestCase):
    def test_resize_bool_mask_matches_target_hw(self):
        from train_locaware import _resize_bool_mask

        mask = torch.ones(1, 8, 8, dtype=torch.bool)
        resized = _resize_bool_mask(mask, (4, 6))

        self.assertEqual(resized.shape, (1, 4, 6))
        self.assertEqual(resized.dtype, torch.bool)

    def test_geometry_anchor_refreshes_after_topology_point_count_change(self):
        from train_locaware import _refresh_geometry_anchor_if_point_count_changed

        class FakeGaussians:
            def __init__(self, n):
                self._xyz = torch.ones(n, 3)
                self._scaling = torch.ones(n, 3)
                self._rotation = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(n, 1)

            @property
            def get_xyz(self):
                return self._xyz

        old_anchor = {
            "xyz": torch.zeros(2, 3),
            "scaling": torch.zeros(2, 3),
            "rotation": torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(2, 1),
        }

        refreshed = _refresh_geometry_anchor_if_point_count_changed(FakeGaussians(3), old_anchor)
        self.assertEqual(refreshed["xyz"].shape[0], 3)
        self.assertTrue(torch.equal(refreshed["xyz"], torch.ones(3, 3)))

        same = _refresh_geometry_anchor_if_point_count_changed(FakeGaussians(3), refreshed)
        self.assertIs(same, refreshed)

    def test_locaware_parser_defaults_disable_loc_opacity_and_support_boolean_override(self):
        from train_locaware import add_locaware_training_args

        parser = ArgumentParser()
        add_locaware_training_args(parser)

        default_args = parser.parse_args([])
        self.assertFalse(default_args.use_loc_opacity)

        enabled = parser.parse_args(["--use_loc_opacity"])
        self.assertTrue(enabled.use_loc_opacity)

        disabled = parser.parse_args(["--use_loc_opacity", "--no-use_loc_opacity"])
        self.assertFalse(disabled.use_loc_opacity)

    def test_feature_phase_trains_only_localization_feature(self):
        from train_locaware import _set_phase_lrs

        class FakeGaussians:
            def __init__(self):
                self.optimizer = SimpleNamespace(
                    param_groups=[
                        {"name": "xyz", "lr": 0.01},
                        {"name": "loc_feature", "lr": 0.02},
                        {"name": "loc_opacity", "lr": 0.03},
                    ]
                )

        gaussians = FakeGaussians()
        args = SimpleNamespace(
            geometry_xyz_lr_mult=0.05,
            geometry_scale_lr_mult=0.1,
            geometry_rotation_lr_mult=0.1,
        )

        _set_phase_lrs(gaussians, "feature", args)

        lrs = {group["name"]: group["lr"] for group in gaussians.optimizer.param_groups}
        self.assertEqual(lrs["xyz"], 0.0)
        self.assertEqual(lrs["loc_opacity"], 0.0)
        self.assertEqual(lrs["loc_feature"], 0.02)

    def test_locaware_parser_accepts_direct_teacher_controls(self):
        from train_locaware import add_locaware_training_args

        parser = ArgumentParser()
        add_locaware_training_args(parser)
        args = parser.parse_args(
            [
                "--loc_teacher",
                "direct",
                "--landmark_path",
                "/tmp/sampled_idx.pkl",
                "--direct_depth_check",
                "--direct_depth_abs_tolerance",
                "0.05",
                "--direct_depth_rel_tolerance",
                "0.02",
            ]
        )

        self.assertEqual(args.loc_teacher, "direct")
        self.assertEqual(args.landmark_path, "/tmp/sampled_idx.pkl")
        self.assertTrue(args.direct_depth_check)
        self.assertEqual(args.direct_depth_abs_tolerance, 0.05)
        self.assertEqual(args.direct_depth_rel_tolerance, 0.02)

    def test_locaware_parser_accepts_mixed_query_controls(self):
        from train_locaware import add_locaware_training_args

        parser = ArgumentParser()
        add_locaware_training_args(parser)
        defaults = parser.parse_args([])
        self.assertEqual(defaults.mixed_sparse_probability, 0.5)
        self.assertEqual(defaults.pose_noise_sampling, "empirical")

        args = parser.parse_args(
            [
                "--query_mode",
                "mixed",
                "--mixed_sparse_probability",
                "0.25",
                "--pose_noise_sampling",
                "quantile",
            ]
        )
        self.assertEqual(args.query_mode, "mixed")
        self.assertEqual(args.mixed_sparse_probability, 0.25)
        self.assertEqual(args.pose_noise_sampling, "quantile")

    def test_locaware_parser_defaults_topology_to_split_only(self):
        from train_locaware import add_locaware_training_args

        parser = ArgumentParser()
        add_locaware_training_args(parser)
        defaults = parser.parse_args([])
        self.assertFalse(defaults.topology_enable_soft_prune)
        self.assertFalse(defaults.topology_enable_physical_prune)

        args = parser.parse_args(["--topology_enable_soft_prune", "--topology_enable_physical_prune"])
        self.assertTrue(args.topology_enable_soft_prune)
        self.assertTrue(args.topology_enable_physical_prune)


if __name__ == "__main__":
    unittest.main()
