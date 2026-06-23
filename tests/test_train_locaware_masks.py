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

    def test_current_landmark_indices_expand_topology_split_children_from_source_ids(self):
        from train_locaware import _current_landmark_indices_from_source_index

        class FakeGaussians:
            def __init__(self):
                self.loc_source_index = torch.tensor([0, 5, 5, 7, 9])

            @property
            def get_xyz(self):
                return torch.zeros(5, 3)

        current = _current_landmark_indices_from_source_index(
            torch.tensor([5, 7]),
            FakeGaussians(),
        )

        self.assertEqual(current.tolist(), [1, 2, 3])

    def test_current_landmark_indices_fallback_without_source_tracking(self):
        from train_locaware import _current_landmark_indices_from_source_index

        class FakeGaussians:
            @property
            def get_xyz(self):
                return torch.zeros(10, 3)

        current = _current_landmark_indices_from_source_index(
            torch.tensor([2, 4]),
            FakeGaussians(),
        )

        self.assertEqual(current.tolist(), [2, 4])

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

    def test_locaware_parser_accepts_external_localization_state(self):
        from train_locaware import add_locaware_training_args

        parser = ArgumentParser()
        add_locaware_training_args(parser)

        args = parser.parse_args(["--localization_state_path", "/tmp/labels.pt"])

        self.assertEqual(args.localization_state_path, "/tmp/labels.pt")

    def test_external_localization_state_restores_after_scene_load(self):
        from train_locaware import _restore_external_localization_state

        class FakeGaussians:
            def __init__(self):
                self.restored = None

            def restore_localization_state(self, state):
                self.restored = state

        state = {"loc_observation_count": torch.tensor([1, 2, 3])}
        gaussians = FakeGaussians()

        _restore_external_localization_state(gaussians, state)

        self.assertTrue(torch.equal(gaussians.restored["loc_observation_count"], torch.tensor([1, 2, 3])))

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

    def test_phase_lr_restore_uses_base_lr_after_feature_freeze(self):
        from train_locaware import _set_phase_lrs

        class FakeGaussians:
            def __init__(self):
                self.optimizer = SimpleNamespace(
                    param_groups=[
                        {"name": "xyz", "lr": 0.01},
                        {"name": "scaling", "lr": 0.02},
                        {"name": "rotation", "lr": 0.03},
                        {"name": "loc_feature", "lr": 0.04},
                        {"name": "loc_opacity", "lr": 0.05},
                        {"name": "opacity", "lr": 0.06},
                    ]
                )

        gaussians = FakeGaussians()
        args = SimpleNamespace(
            geometry_xyz_lr_mult=0.5,
            geometry_scale_lr_mult=0.25,
            geometry_rotation_lr_mult=0.1,
        )

        _set_phase_lrs(gaussians, "feature", args)
        _set_phase_lrs(gaussians, "geometry", args)

        lrs = {group["name"]: group["lr"] for group in gaussians.optimizer.param_groups}
        self.assertEqual(lrs["xyz"], 0.005)
        self.assertEqual(lrs["scaling"], 0.005)
        self.assertEqual(lrs["rotation"], 0.003)
        self.assertEqual(lrs["loc_feature"], 0.04)
        self.assertEqual(lrs["loc_opacity"], 0.05)
        self.assertEqual(lrs["opacity"], 0.0)

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

    def test_locaware_parser_defaults_dense_responsibility_gates_off(self):
        from train_locaware import add_locaware_training_args

        parser = ArgumentParser()
        add_locaware_training_args(parser)

        defaults = parser.parse_args([])
        configured = parser.parse_args(
            [
                "--loc_responsibility_opacity_weight",
                "0.5",
                "--loc_responsibility_depth_weight",
                "0.25",
            ]
        )

        self.assertEqual(defaults.loc_responsibility_opacity_weight, 0.0)
        self.assertEqual(defaults.loc_responsibility_depth_weight, 0.0)
        self.assertEqual(configured.loc_responsibility_opacity_weight, 0.5)
        self.assertEqual(configured.loc_responsibility_depth_weight, 0.25)

    def test_locaware_parser_accepts_v03_full_bank_controls(self):
        from train_locaware import add_locaware_training_args

        parser = ArgumentParser()
        add_locaware_training_args(parser)
        args = parser.parse_args(
            [
                "--loc_full_bank_weight",
                "0.3",
                "--loc_full_bank_temperature",
                "0.11",
                "--loc_full_bank_hard_negatives",
                "16",
                "--loc_full_bank_margin",
                "0.4",
                "--loc_anchor_weight",
                "0.02",
            ]
        )

        self.assertEqual(args.loc_full_bank_weight, 0.3)
        self.assertEqual(args.loc_full_bank_temperature, 0.11)
        self.assertEqual(args.loc_full_bank_hard_negatives, 16)
        self.assertEqual(args.loc_full_bank_margin, 0.4)
        self.assertEqual(args.loc_anchor_weight, 0.02)

    def test_locaware_parser_accepts_dense_responsibility_kl_controls(self):
        from train_locaware import add_locaware_training_args

        parser = ArgumentParser()
        add_locaware_training_args(parser)
        defaults = parser.parse_args([])
        args = parser.parse_args(
            [
                "--loc_dense_kl_weight",
                "0.05",
                "--loc_dense_kl_temperature",
                "0.09",
                "--loc_responsibility_topk",
                "6",
            ]
        )

        self.assertEqual(defaults.loc_responsibility_topk, 32)
        self.assertEqual(args.loc_dense_kl_weight, 0.05)
        self.assertEqual(args.loc_dense_kl_temperature, 0.09)
        self.assertEqual(args.loc_responsibility_topk, 6)

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

    def test_locaware_parser_accepts_physical_prune_thresholds(self):
        from train_locaware import add_locaware_training_args

        parser = ArgumentParser()
        add_locaware_training_args(parser)
        args = parser.parse_args(
            [
                "--topology_physical_rgb_threshold",
                "0.1",
                "--topology_physical_loc_threshold",
                "0.2",
                "--topology_physical_utility_threshold",
                "-1.5",
            ]
        )

        self.assertEqual(args.topology_physical_rgb_threshold, 0.1)
        self.assertEqual(args.topology_physical_loc_threshold, 0.2)
        self.assertEqual(args.topology_physical_utility_threshold, -1.5)

    def test_locaware_parser_accepts_topology_split_gate_controls(self):
        from train_locaware import add_locaware_training_args

        parser = ArgumentParser()
        add_locaware_training_args(parser)
        args = parser.parse_args(
            [
                "--topology_min_repeatability",
                "0.05",
                "--topology_min_radius",
                "0.5",
            ]
        )

        self.assertEqual(args.topology_min_repeatability, 0.05)
        self.assertEqual(args.topology_min_radius, 0.5)


if __name__ == "__main__":
    unittest.main()
