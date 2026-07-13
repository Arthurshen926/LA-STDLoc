import unittest
import random as py_random
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

    def test_feature_map_normalization_is_inplace(self):
        import torch.nn.functional as F

        from train_locaware import _normalize_feature_map_inplace

        feature_map = torch.randn(4, 3, 5)
        expected = F.normalize(feature_map.clone(), p=2, dim=0)
        data_ptr = feature_map.data_ptr()

        normalized = _normalize_feature_map_inplace(feature_map)

        self.assertEqual(normalized.data_ptr(), data_ptr)
        self.assertTrue(torch.allclose(normalized, expected, atol=1e-6))

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

    def test_geometry_anchor_refreshes_equal_count_mutation_by_node_id(self):
        from train_locaware import (
            _capture_geometry_anchor,
            _refresh_geometry_anchor_if_point_count_changed,
        )

        class FakeGaussians:
            def __init__(self, xyz, node_ids):
                self._xyz = xyz
                self._scaling = xyz + 100.0
                self._rotation = torch.stack(
                    [
                        torch.tensor([float(value), 1.0, 0.0, 0.0])
                        for value in xyz[:, 0]
                    ]
                )
                self.loc_node_id = node_ids

            @property
            def get_xyz(self):
                return self._xyz

        anchor = _capture_geometry_anchor(
            FakeGaussians(
                torch.tensor([[10.0, 0.0, 0.0], [20.0, 0.0, 0.0], [30.0, 0.0, 0.0]]),
                torch.tensor([100, 101, 102]),
            )
        )
        refreshed = _refresh_geometry_anchor_if_point_count_changed(
            FakeGaussians(
                torch.tensor([[200.0, 0.0, 0.0], [300.0, 0.0, 0.0], [400.0, 0.0, 0.0]]),
                torch.tensor([101, 102, 103]),
            ),
            anchor,
        )

        self.assertEqual(refreshed["node_ids"].tolist(), [101, 102, 103])
        self.assertTrue(
            torch.equal(
                refreshed["xyz"],
                torch.tensor([[200.0, 0.0, 0.0], [300.0, 0.0, 0.0], [400.0, 0.0, 0.0]]),
            )
        )
        self.assertTrue(
            torch.equal(
                refreshed["scaling"],
                torch.tensor([[300.0, 100.0, 100.0], [400.0, 100.0, 100.0], [500.0, 100.0, 100.0]]),
            )
        )

    def test_geometry_step_delta_skips_point_count_changes(self):
        from train_locaware import _record_geometry_optimizer_diagnostics

        class FakeGaussians:
            def __init__(self):
                self._xyz = torch.tensor([[1000.0, 0.0, 0.0], [1001.0, 0.0, 0.0]])
                self.optimizer = SimpleNamespace(param_groups=[{"name": "xyz", "lr": 0.0}])

        summary = {}
        _record_geometry_optimizer_diagnostics(
            summary,
            FakeGaussians(),
            phase="full",
            xyz_before=torch.tensor([[0.0, 0.0, 0.0]]),
            record_lr_grad=False,
        )

        self.assertEqual(summary["geometry_xyz_step_point_count_changed"], 1)
        self.assertEqual(summary["geometry_xyz_step_delta_skipped_point_count_changed"], 1)
        self.assertNotIn("geometry_xyz_step_delta_max", summary)

    def test_lafgs_geometry_residual_diagnostics_are_recorded_to_summary(self):
        from train_locaware import _record_lafgs_geometry_residual_diagnostics

        teacher_out = SimpleNamespace(diagnostics={})
        summary = {}

        _record_lafgs_geometry_residual_diagnostics(
            summary,
            teacher_out,
            torch.tensor(0.25),
            {
                "over_limit_count": 3,
                "max_residual_norm": 1.5,
                "max_allowed_norm": 0.4,
            },
        )

        self.assertEqual(teacher_out.diagnostics["lafgs_geometry_residual_loss"], 0.25)
        self.assertEqual(summary["direct_diag_lafgs_geometry_residual_loss_total"], 0.25)
        self.assertEqual(summary["direct_diag_lafgs_geometry_residual_over_limit_count_total"], 3.0)
        self.assertEqual(summary["direct_diag_lafgs_geometry_residual_max_norm_max"], 1.5)
        self.assertEqual(summary["direct_diag_lafgs_geometry_residual_max_allowed_min"], 0.4)

    def test_rgb_densify_child_outlier_prune_removes_only_far_children(self):
        from train_locaware import _prune_lafgs_rgb_densify_child_outliers

        class FakeGaussians:
            def __init__(self):
                self._xyz = torch.tensor(
                    [
                        [10.0, 0.0, 0.0],
                        [0.2, 0.0, 0.0],
                        [5.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0],
                    ],
                    dtype=torch.float32,
                )
                self.loc_source_xyz = torch.zeros(4, 3)
                self.loc_birth_iteration = torch.tensor([0, 100, 100, 100])
                self.loc_source_index = torch.tensor([0, 0, 0, 0])
                self.pruned_mask = None

            @property
            def get_xyz(self):
                return self._xyz

            def prune_points(self, mask):
                self.pruned_mask = mask.detach().cpu()
                keep = ~mask.cpu()
                self._xyz = self._xyz[keep]
                self.loc_source_xyz = self.loc_source_xyz[keep]
                self.loc_birth_iteration = self.loc_birth_iteration[keep]
                self.loc_source_index = self.loc_source_index[keep]

        gaussians = FakeGaussians()

        stats = _prune_lafgs_rgb_densify_child_outliers(gaussians, max_source_drift=2.0)

        self.assertEqual(stats["pruned"], 1)
        self.assertEqual(stats["child_count"], 3)
        self.assertTrue(torch.equal(gaussians.pruned_mask, torch.tensor([False, False, True, False])))
        self.assertEqual(gaussians.get_xyz.shape[0], 3)
        self.assertTrue(torch.equal(gaussians.loc_birth_iteration, torch.tensor([0, 100, 100])))

    def test_final_geometry_delta_aligns_by_source_index_and_birth(self):
        from train_locaware import _record_final_geometry_delta_summary

        class FakeGaussians:
            def __init__(self):
                self._xyz = torch.tensor(
                    [
                        [10.1, 0.0, 0.0],
                        [20.2, 0.0, 0.0],
                        [99.0, 0.0, 0.0],
                    ],
                    dtype=torch.float32,
                )
                self.loc_source_index = torch.tensor([1, 0, 0])
                self.loc_birth_iteration = torch.tensor([0, 0, 500])
                self.loc_source_xyz = torch.tensor(
                    [
                        [10.0, 0.0, 0.0],
                        [20.0, 0.0, 0.0],
                        [20.0, 0.0, 0.0],
                    ],
                    dtype=torch.float32,
                )

            @property
            def get_xyz(self):
                return self._xyz

            def get_localization_xyz(self):
                return self._xyz

        reference = {
            "raw_xyz": torch.tensor([[20.0, 0.0, 0.0], [10.0, 0.0, 0.0]], dtype=torch.float32),
            "loc_xyz": torch.tensor([[20.0, 0.0, 0.0], [10.0, 0.0, 0.0]], dtype=torch.float32),
        }
        summary = {}

        _record_final_geometry_delta_summary(summary, FakeGaussians(), reference)

        self.assertEqual(summary["geometry_point_count_changed"], True)
        self.assertEqual(summary["geometry_source_aligned_delta_count"], 3)
        self.assertEqual(summary["geometry_birth0_delta_count"], 2)
        self.assertEqual(summary["geometry_child_delta_count"], 1)
        self.assertAlmostEqual(summary["raw_xyz_delta_from_initial_max"], 0.2, places=5)
        self.assertAlmostEqual(summary["loc_xyz_delta_from_initial_max"], 0.2, places=5)
        self.assertAlmostEqual(summary["raw_xyz_child_delta_from_source_max"], 79.0, places=5)

    def test_feature_anchor_refresh_aligns_by_stable_node_id_not_row_order(self):
        from train_locaware import (
            _capture_feature_anchor,
            _feature_anchor_tensor,
            _refresh_feature_anchor_if_point_count_changed,
        )

        class FakeGaussians:
            def __init__(self, features, node_ids):
                self._loc_feature = features.reshape(features.shape[0], 1, features.shape[1])
                self.loc_node_id = node_ids

            @property
            def get_xyz(self):
                return torch.zeros(self._loc_feature.shape[0], 3)

            @property
            def get_loc_feature(self):
                return self._loc_feature

        anchor = _capture_feature_anchor(
            FakeGaussians(
                torch.tensor([[10.0, 0.0], [20.0, 0.0], [30.0, 0.0]]),
                torch.tensor([100, 101, 102]),
            )
        )
        refreshed = _refresh_feature_anchor_if_point_count_changed(
            FakeGaussians(
                torch.tensor([[200.0, 0.0], [300.0, 0.0], [400.0, 0.0]]),
                torch.tensor([101, 102, 103]),
            ),
            anchor,
        )

        aligned = _feature_anchor_tensor(refreshed).reshape(3, 2)
        self.assertTrue(torch.equal(aligned[0], torch.tensor([20.0, 0.0])))
        self.assertTrue(torch.equal(aligned[1], torch.tensor([30.0, 0.0])))
        self.assertTrue(torch.equal(aligned[2], torch.tensor([400.0, 0.0])))

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

    def test_frozen_child_feature_gradients_are_masked_by_parent_age(self):
        from train_locaware import _mask_frozen_child_loc_feature_gradients

        class FakeGaussians:
            def __init__(self):
                self._loc_feature = torch.nn.Parameter(torch.zeros(4, 1, 2))
                self._loc_feature.grad = torch.ones_like(self._loc_feature)
                self.loc_parent_node_id = torch.tensor([-1, 10, 10, 11])
                self.last_topology_iteration = torch.tensor([0, 90, 40, 99])

        gaussians = FakeGaussians()

        frozen_count = _mask_frozen_child_loc_feature_gradients(
            gaussians,
            iteration=100,
            freeze_steps=10,
        )

        self.assertEqual(frozen_count, 2)
        self.assertTrue(torch.equal(gaussians._loc_feature.grad[0], torch.ones(1, 2)))
        self.assertTrue(torch.equal(gaussians._loc_feature.grad[1], torch.zeros(1, 2)))
        self.assertTrue(torch.equal(gaussians._loc_feature.grad[2], torch.ones(1, 2)))
        self.assertTrue(torch.equal(gaussians._loc_feature.grad[3], torch.zeros(1, 2)))

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

    def test_locaware_parser_accepts_update3_topology_schedule_controls(self):
        from train_locaware import add_locaware_training_args

        parser = ArgumentParser()
        add_locaware_training_args(parser)

        defaults = parser.parse_args([])
        args = parser.parse_args(
            [
                "--topology_max_mutation_events",
                "1",
                "--loc_child_feature_freeze_steps",
                "100",
                "--loc_full_bank_nearby_as_positive_until",
                "30625",
            ]
        )

        self.assertEqual(defaults.topology_max_mutation_events, 0)
        self.assertEqual(defaults.loc_child_feature_freeze_steps, 0)
        self.assertEqual(defaults.loc_full_bank_nearby_as_positive_until, 0)
        self.assertEqual(args.topology_max_mutation_events, 1)
        self.assertEqual(args.loc_child_feature_freeze_steps, 100)
        self.assertEqual(args.loc_full_bank_nearby_as_positive_until, 30625)

    def test_lafgs_nearby_positive_until_uses_relative_step_for_checkpoint_resume(self):
        from argparse import Namespace

        from train_locaware import full_bank_nearby_as_positive_active

        args = Namespace(
            loc_full_bank_nearby_as_positive=True,
            loc_full_bank_nearby_as_positive_until=10000,
        )

        self.assertTrue(
            full_bank_nearby_as_positive_active(args, iteration=30001, lafgs_step=1)
        )
        self.assertFalse(
            full_bank_nearby_as_positive_active(args, iteration=40001, lafgs_step=10001)
        )

        args.loc_full_bank_nearby_as_positive_until = 0
        self.assertTrue(
            full_bank_nearby_as_positive_active(args, iteration=50000, lafgs_step=20000)
        )

    def test_sfm_from_zero_resume_uses_absolute_training_step(self):
        from argparse import Namespace

        from train_locaware import (
            lafgs_curriculum_base_iteration,
            lafgs_curriculum_step,
            lafgs_stage_loss_weights,
        )

        args = Namespace(
            lafgs_stage_schedule="sfm_from_zero",
            lafgs_stage_bootstrap_until=3000,
            lafgs_stage_joint_until=15000,
            lafgs_stage_bootstrap_base_weight=1.0,
            lafgs_stage_bootstrap_loc_weight=0.15,
            lafgs_stage_bootstrap_geometry_anchor_weight=0.05,
            lafgs_stage_joint_base_weight=0.5,
            lafgs_stage_joint_loc_weight=1.0,
            lafgs_stage_joint_geometry_anchor_weight=0.05,
            lafgs_stage_refine_base_weight=0.15,
            lafgs_stage_refine_loc_weight=1.5,
            lafgs_stage_refine_geometry_anchor_weight=0.02,
            base_loss_weight=1.0,
            loc_loss_weight=1.0,
            geometry_anchor_weight=0.0,
        )

        base_iteration = lafgs_curriculum_base_iteration(args, scene_loaded_iter=5000)
        resumed_step = lafgs_curriculum_step(iteration=5001, base_iteration=base_iteration)

        self.assertEqual(base_iteration, 0)
        self.assertEqual(resumed_step, 5001)
        self.assertEqual(lafgs_stage_loss_weights(args, resumed_step)["stage"], "joint")

    def test_sfm_from_zero_resume_skips_multiview_initialization(self):
        from argparse import Namespace

        from train_locaware import lafgs_should_run_multiview_initialization

        sfm_args = Namespace(
            lafgs_stage_schedule="sfm_from_zero",
            lafgs_mvinit_enabled=True,
            lafgs_mvinit_max_views=64,
        )
        legacy_args = Namespace(
            lafgs_stage_schedule="none",
            lafgs_mvinit_enabled=True,
            lafgs_mvinit_max_views=64,
        )

        self.assertTrue(lafgs_should_run_multiview_initialization(sfm_args, first_iter=0))
        self.assertFalse(lafgs_should_run_multiview_initialization(sfm_args, first_iter=5000))
        self.assertTrue(lafgs_should_run_multiview_initialization(legacy_args, first_iter=30000))

    def test_locaware_parser_accepts_sfm_from_zero_stage_and_rgb_densify_controls(self):
        from train_locaware import add_locaware_training_args, lafgs_stage_loss_weights

        parser = ArgumentParser()
        add_locaware_training_args(parser)

        defaults = parser.parse_args([])
        self.assertEqual(defaults.lafgs_stage_schedule, "none")
        self.assertFalse(defaults.lafgs_rgb_densify)

        args = parser.parse_args(
            [
                "--lafgs_stage_schedule",
                "sfm_from_zero",
                "--lafgs_stage_bootstrap_until",
                "3000",
                "--lafgs_stage_joint_until",
                "15000",
                "--lafgs_rgb_densify",
                "--lafgs_rgb_densify_until_iter",
                "15000",
                "--lafgs_rgb_densify_child_max_source_drift",
                "2.5",
            ]
        )

        self.assertEqual(args.lafgs_stage_schedule, "sfm_from_zero")
        self.assertTrue(args.lafgs_rgb_densify)
        self.assertEqual(args.lafgs_rgb_densify_until_iter, 15000)
        self.assertEqual(args.lafgs_rgb_densify_child_max_source_drift, 2.5)
        self.assertEqual(lafgs_stage_loss_weights(args, 1000)["stage"], "bootstrap")
        self.assertGreater(
            lafgs_stage_loss_weights(args, 20000)["loc"],
            lafgs_stage_loss_weights(args, 20000)["base"],
        )

    def test_landmark_indices_can_bootstrap_from_all_current_points(self):
        from train_locaware import _load_landmark_indices

        indices = _load_landmark_indices("/unused", "__all__", point_count=5)

        self.assertEqual(indices.tolist(), [0, 1, 2, 3, 4])

    def test_sfm_from_zero_phase_lrs_keep_rgb_scaffold_trainable(self):
        from argparse import Namespace

        from train_locaware import _set_phase_lrs

        class FakeGaussians:
            def __init__(self):
                self.optimizer = SimpleNamespace(
                    param_groups=[
                        {"name": "xyz", "lr": 0.1},
                        {"name": "f_dc", "lr": 0.2},
                        {"name": "f_rest", "lr": 0.3},
                        {"name": "opacity", "lr": 0.4},
                        {"name": "scaling", "lr": 0.5},
                        {"name": "rotation", "lr": 0.6},
                        {"name": "loc_feature", "lr": 0.7},
                        {"name": "loc_opacity", "lr": 0.8},
                    ]
                )

        gaussians = FakeGaussians()
        args = Namespace(
            gaussian_type="2dgs",
            lafgs_stage_schedule="sfm_from_zero",
            allow_raw_xyz_geometry_grad=True,
            loc_overlay_mode="none",
            loc_anchor_lr=0.0,
            surfel_loc_tangent_bound=0.0,
            surfel_loc_normal_bound=0.0,
            use_loc_opacity=True,
            lafgs_diff_pnp_allow_geometry_grad=False,
            lafgs_diff_pnp_geometry_xyz_lr=0.0,
            geometry_xyz_lr_mult=1.0,
            geometry_scale_lr_mult=1.0,
            geometry_rotation_lr_mult=1.0,
        )

        _set_phase_lrs(gaussians, "locrec", args)
        lr_by_name = {group["name"]: group["lr"] for group in gaussians.optimizer.param_groups}

        self.assertGreater(lr_by_name["xyz"], 0.0)
        self.assertGreater(lr_by_name["f_dc"], 0.0)
        self.assertGreater(lr_by_name["scaling"], 0.0)
        self.assertGreater(lr_by_name["loc_feature"], 0.0)

    def test_locaware_pseudo_query_defaults_are_train_rgb_mainline(self):
        from train_locaware import add_locaware_training_args

        parser = ArgumentParser()
        add_locaware_training_args(parser)

        defaults = parser.parse_args([])
        opt_in = parser.parse_args(
            [
                "--pseudo_query_sources",
                "train_rgb,synthetic_rgb",
                "--pseudo_query_sampling_mode",
                "source_balanced",
            ]
        )

        self.assertEqual(defaults.pseudo_query_sources, "train_rgb")
        self.assertEqual(defaults.pseudo_query_sampling_mode, "record_proportional")
        self.assertTrue(defaults.pseudo_query_require_teacher_cache)
        self.assertEqual(opt_in.pseudo_query_sources, "train_rgb,synthetic_rgb")
        self.assertEqual(opt_in.pseudo_query_sampling_mode, "source_balanced")

        opt_out_cache = parser.parse_args(["--no-pseudo_query_require_teacher_cache"])
        self.assertFalse(opt_out_cache.pseudo_query_require_teacher_cache)

    def test_query_artifact_filter_loads_scene_split_and_severity_names(self):
        import csv
        import tempfile
        from pathlib import Path

        from train_locaware import _load_query_artifact_filter_names

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "artifact_candidates.csv"
            with path.open("w", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=["scene", "split", "image_name", "gate_severity"],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "scene": "ShopFacade",
                        "split": "heldout_query_sample",
                        "image_name": "seq2/frame00146.png",
                        "gate_severity": "mild",
                    }
                )
                writer.writerow(
                    {
                        "scene": "ShopFacade",
                        "split": "final_test_sample",
                        "image_name": "seq1/frame00028.png",
                        "gate_severity": "severe",
                    }
                )
                writer.writerow(
                    {
                        "scene": "OldHospital",
                        "split": "heldout_query_sample",
                        "image_name": "seq5/frame00011.png",
                        "gate_severity": "severe",
                    }
                )

            names = _load_query_artifact_filter_names(
                str(path),
                scene_name="ShopFacade",
                severities="mild,severe",
                splits="heldout_query_sample",
            )

        self.assertEqual(names, {"seq2/frame00146.png"})

    def test_query_artifact_filter_removes_only_query_cameras_by_image_name(self):
        from train_locaware import _filter_query_cameras_by_artifacts

        cameras = [
            SimpleNamespace(image_name="seq2/frame00001.png"),
            SimpleNamespace(image_name="seq2/frame00146.png"),
            SimpleNamespace(image_name="seq2/frame00002.png"),
        ]

        filtered, removed = _filter_query_cameras_by_artifacts(
            cameras,
            {"seq2/frame00146.png"},
        )

        self.assertEqual([camera.image_name for camera in filtered], ["seq2/frame00001.png", "seq2/frame00002.png"])
        self.assertEqual(removed, ["seq2/frame00146.png"])

    def test_query_artifact_filter_does_not_match_basename_across_sequence_prefixes(self):
        from train_locaware import _filter_query_cameras_by_artifacts

        cameras = [
            SimpleNamespace(image_name="seq1/frame00001.png"),
            SimpleNamespace(image_name="seq1/frame00146.png"),
            SimpleNamespace(image_name="seq1/frame00002.png"),
        ]

        filtered, removed = _filter_query_cameras_by_artifacts(
            cameras,
            {"seq2/frame00146.png"},
        )

        self.assertEqual(
            [camera.image_name for camera in filtered],
            ["seq1/frame00001.png", "seq1/frame00146.png", "seq1/frame00002.png"],
        )
        self.assertEqual(removed, [])

    def test_locaware_parser_accepts_render_artifact_weighting_args(self):
        from train_locaware import add_locaware_training_args

        parser = ArgumentParser()
        add_locaware_training_args(parser)

        defaults = parser.parse_args([])
        args = parser.parse_args(
            [
                "--render_artifact_weight_path",
                "/tmp/artifacts.csv",
                "--render_artifact_weight_targets",
                "teacher,risk",
                "--render_artifact_weight_mild",
                "0.7",
                "--render_artifact_weight_severe",
                "0.3",
            ]
        )

        self.assertEqual(defaults.render_artifact_weight_path, "")
        self.assertEqual(defaults.render_artifact_weight_targets, "teacher")
        self.assertEqual(defaults.render_artifact_weight_severities, "severe")
        self.assertEqual(defaults.render_artifact_weight_mode, "severity")
        self.assertAlmostEqual(defaults.render_artifact_weight_mild, 1.0)
        self.assertAlmostEqual(defaults.render_artifact_weight_severe, 0.70)
        self.assertAlmostEqual(defaults.render_artifact_weight_continuous_min, 0.70)
        self.assertAlmostEqual(defaults.render_artifact_weight_continuous_power, 1.0)
        self.assertEqual(args.render_artifact_weight_path, "/tmp/artifacts.csv")
        self.assertEqual(args.render_artifact_weight_targets, "teacher,risk")
        self.assertAlmostEqual(args.render_artifact_weight_mild, 0.7)
        self.assertAlmostEqual(args.render_artifact_weight_severe, 0.3)

    def test_locaware_parser_accepts_topology_risk_commit_policy(self):
        from train_locaware import add_locaware_training_args

        parser = ArgumentParser()
        add_locaware_training_args(parser)

        defaults = parser.parse_args([])
        reject = parser.parse_args(["--topology_risk_commit_policy", "reject_all"])
        accept = parser.parse_args(["--topology_risk_commit_policy", "accept_all"])
        heldout = parser.parse_args(
            [
                "--topology_risk_commit_policy",
                "heldout_descriptor",
                "--topology_risk_holdout_size",
                "3",
                "--topology_risk_holdout_selection",
                "pose_stratified",
                "--topology_risk_epsilon",
                "0.05",
                "--topology_risk_ci_z",
                "1.96",
                "--topology_risk_min_ci_samples",
                "4",
                "--topology_risk_desc_weight",
                "1.5",
                "--topology_risk_full_bank_weight",
                "0.25",
                "--topology_risk_reproj_weight",
                "0.1",
                "--topology_risk_anchors",
                "64",
            ]
        )
        heldout_pose = parser.parse_args(
            [
                "--topology_risk_commit_policy",
                "heldout_pose",
                "--topology_risk_pose_cfg",
                "pose.yaml",
                "--topology_risk_pose_ae_weight",
                "2.0",
                "--topology_risk_pose_te_weight",
                "3.0",
                "--topology_risk_pose_inlier_weight",
                "0.5",
                "--topology_risk_pose_ae_scale",
                "10.0",
                "--topology_risk_pose_te_scale",
                "50.0",
                "--topology_risk_pose_inlier_scale",
                "200.0",
                "--topology_risk_pose_veto_mode",
                "r5_r2_tail",
                "--topology_risk_pose_r5_miss_weight",
                "4.0",
                "--topology_risk_pose_r2_miss_weight",
                "2.0",
                "--topology_risk_pose_tail_fail_weight",
                "8.0",
                "--topology_risk_pose_cvar_weight",
                "0.75",
                "--topology_risk_pose_cvar_fraction",
                "0.25",
                "--topology_risk_pose_r2_tolerance",
                "0.125",
                "--topology_risk_pose_tail_tolerance",
                "0.25",
            ]
        )

        self.assertEqual(defaults.topology_risk_commit_policy, "off")
        self.assertEqual(defaults.topology_risk_holdout_size, 4)
        self.assertEqual(defaults.topology_risk_holdout_selection, "prefix")
        self.assertEqual(defaults.topology_risk_epsilon, 0.0)
        self.assertEqual(defaults.topology_risk_ci_z, 0.0)
        self.assertEqual(defaults.topology_risk_min_ci_samples, 2)
        self.assertEqual(defaults.topology_risk_pose_ae_weight, 1.0)
        self.assertEqual(defaults.topology_risk_pose_te_weight, 1.0)
        self.assertEqual(defaults.topology_risk_pose_inlier_weight, 0.0)
        self.assertEqual(defaults.topology_risk_pose_cfg, "")
        self.assertEqual(defaults.topology_risk_pose_ae_scale, 5.0)
        self.assertEqual(defaults.topology_risk_pose_te_scale, 200.0)
        self.assertEqual(defaults.topology_risk_pose_inlier_scale, 100.0)
        self.assertEqual(defaults.topology_risk_pose_veto_mode, "off")
        self.assertEqual(defaults.topology_risk_pose_r5_miss_weight, 0.0)
        self.assertEqual(defaults.topology_risk_pose_r2_miss_weight, 0.0)
        self.assertEqual(defaults.topology_risk_pose_tail_fail_weight, 0.0)
        self.assertEqual(defaults.topology_risk_pose_cvar_weight, 0.0)
        self.assertEqual(defaults.topology_risk_pose_cvar_fraction, 0.25)
        self.assertEqual(defaults.topology_risk_pose_r2_tolerance, 0.0)
        self.assertEqual(defaults.topology_risk_pose_tail_tolerance, 0.0)
        self.assertEqual(reject.topology_risk_commit_policy, "reject_all")
        self.assertEqual(accept.topology_risk_commit_policy, "accept_all")
        self.assertEqual(heldout.topology_risk_commit_policy, "heldout_descriptor")
        self.assertEqual(heldout.topology_risk_holdout_size, 3)
        self.assertEqual(heldout.topology_risk_holdout_selection, "pose_stratified")
        self.assertEqual(heldout.topology_risk_epsilon, 0.05)
        self.assertEqual(heldout.topology_risk_ci_z, 1.96)
        self.assertEqual(heldout.topology_risk_min_ci_samples, 4)
        self.assertEqual(heldout.topology_risk_desc_weight, 1.5)
        self.assertEqual(heldout.topology_risk_full_bank_weight, 0.25)
        self.assertEqual(heldout.topology_risk_reproj_weight, 0.1)
        self.assertEqual(heldout.topology_risk_anchors, 64)
        self.assertEqual(heldout_pose.topology_risk_commit_policy, "heldout_pose")
        self.assertEqual(heldout_pose.topology_risk_pose_cfg, "pose.yaml")
        self.assertEqual(heldout_pose.topology_risk_pose_ae_weight, 2.0)
        self.assertEqual(heldout_pose.topology_risk_pose_te_weight, 3.0)
        self.assertEqual(heldout_pose.topology_risk_pose_inlier_weight, 0.5)
        self.assertEqual(heldout_pose.topology_risk_pose_ae_scale, 10.0)
        self.assertEqual(heldout_pose.topology_risk_pose_te_scale, 50.0)
        self.assertEqual(heldout_pose.topology_risk_pose_inlier_scale, 200.0)
        self.assertEqual(heldout_pose.topology_risk_pose_veto_mode, "r5_r2_tail")
        self.assertEqual(heldout_pose.topology_risk_pose_r5_miss_weight, 4.0)
        self.assertEqual(heldout_pose.topology_risk_pose_r2_miss_weight, 2.0)
        self.assertEqual(heldout_pose.topology_risk_pose_tail_fail_weight, 8.0)
        self.assertEqual(heldout_pose.topology_risk_pose_cvar_weight, 0.75)
        self.assertEqual(heldout_pose.topology_risk_pose_cvar_fraction, 0.25)
        self.assertEqual(heldout_pose.topology_risk_pose_r2_tolerance, 0.125)
        self.assertEqual(heldout_pose.topology_risk_pose_tail_tolerance, 0.25)

    def test_select_risk_cameras_supports_prefix_strided_and_pose_stratified_modes(self):
        from train_locaware import _select_risk_cameras

        cameras = list(range(10))

        self.assertEqual(_select_risk_cameras(cameras, 4, "prefix"), [0, 1, 2, 3])
        self.assertEqual(_select_risk_cameras(cameras, 4, "strided"), [0, 3, 6, 9])
        self.assertEqual(_select_risk_cameras(cameras, 1, "strided"), [5])
        self.assertEqual(_select_risk_cameras(cameras, 20, "strided"), cameras)
        self.assertEqual(_select_risk_cameras([], 4, "strided"), [])
        self.assertEqual(_select_risk_cameras(cameras, 4, "pose_stratified"), [0, 3, 6, 9])

        shuffled_pose_cameras = [
            SimpleNamespace(camera_center=torch.tensor([5.0, 0.0, 0.0])),
            SimpleNamespace(camera_center=torch.tensor([0.0, 0.0, 0.0])),
            SimpleNamespace(camera_center=torch.tensor([4.0, 0.0, 0.0])),
            SimpleNamespace(camera_center=torch.tensor([1.0, 0.0, 0.0])),
            SimpleNamespace(camera_center=torch.tensor([3.0, 0.0, 0.0])),
            SimpleNamespace(camera_center=torch.tensor([2.0, 0.0, 0.0])),
        ]
        selected = _select_risk_cameras(shuffled_pose_cameras, 3, "pose_stratified")
        self.assertEqual([float(cam.camera_center[0]) for cam in selected], [0.0, 2.0, 5.0])

        with self.assertRaises(ValueError):
            _select_risk_cameras(cameras, 4, "unknown")

    def test_heldout_pose_risk_score_penalizes_pose_error_and_rewards_inliers(self):
        from train_locaware import _pose_risk_from_sparse_metrics

        args = SimpleNamespace(
            topology_risk_pose_ae_weight=2.0,
            topology_risk_pose_te_weight=3.0,
            topology_risk_pose_inlier_weight=0.5,
            topology_risk_pose_ae_scale=10.0,
            topology_risk_pose_te_scale=100.0,
            topology_risk_pose_inlier_scale=200.0,
        )

        risk = _pose_risk_from_sparse_metrics(ae_deg=5.0, te_cm=20.0, inliers=50, args=args)

        self.assertAlmostEqual(risk, 2.0 * 0.5 + 3.0 * 0.2 - 0.5 * 0.25)

    def test_heldout_pose_risk_penalizes_recall_misses_and_tail_failures(self):
        from train_locaware import _pose_risk_from_sparse_metrics

        args = SimpleNamespace(
            topology_risk_pose_ae_weight=0.0,
            topology_risk_pose_te_weight=0.0,
            topology_risk_pose_inlier_weight=0.0,
            topology_risk_pose_ae_scale=10.0,
            topology_risk_pose_te_scale=100.0,
            topology_risk_pose_inlier_scale=200.0,
            topology_risk_pose_r5_ae_threshold=5.0,
            topology_risk_pose_r5_te_threshold=5.0,
            topology_risk_pose_r2_ae_threshold=2.0,
            topology_risk_pose_r2_te_threshold=2.0,
            topology_risk_pose_tail_ae_threshold=10.0,
            topology_risk_pose_tail_te_threshold=500.0,
            topology_risk_pose_r5_miss_weight=4.0,
            topology_risk_pose_r2_miss_weight=2.0,
            topology_risk_pose_tail_fail_weight=8.0,
        )

        self.assertAlmostEqual(
            _pose_risk_from_sparse_metrics(ae_deg=1.0, te_cm=1.0, inliers=0, args=args),
            0.0,
        )
        self.assertAlmostEqual(
            _pose_risk_from_sparse_metrics(ae_deg=6.0, te_cm=1.0, inliers=0, args=args),
            6.0,
        )
        self.assertAlmostEqual(
            _pose_risk_from_sparse_metrics(ae_deg=11.0, te_cm=1.0, inliers=0, args=args),
            14.0,
        )

    def test_pose_risk_aggregation_adds_tail_cvar(self):
        from train_locaware import _aggregate_pose_risk_values

        args = SimpleNamespace(
            topology_risk_pose_cvar_weight=2.0,
            topology_risk_pose_cvar_fraction=0.5,
        )

        risk = _aggregate_pose_risk_values([0.1, 0.2, 1.0, 2.0], args)

        self.assertAlmostEqual(risk, 0.825 + 2.0 * 1.5)

    def test_heldout_risk_evaluator_accepts_only_when_trial_risk_decreases(self):
        from train_locaware import HeldoutRiskCommitEvaluator

        class FakeGaussians:
            def __init__(self):
                self.value = "baseline"
                self.restored = False

        class FakeProposal:
            iteration = 10
            candidate_count = 2
            split_mask = torch.tensor([True, False])
            physical_prune_mask = torch.tensor([False, False])

        gaussians = FakeGaussians()
        proposal = FakeProposal()

        def capture(model):
            return {"value": model.value}

        def restore(model, state):
            model.value = state["value"]
            model.restored = True

        def apply_trial(model, _proposal):
            model.value = "trial"

        scores = {"baseline": 1.0, "trial": 0.8}
        evaluator = HeldoutRiskCommitEvaluator(
            score_fn=lambda model: scores[model.value],
            apply_trial_fn=apply_trial,
            capture_state_fn=capture,
            restore_state_fn=restore,
            epsilon=0.05,
            reason_prefix="unit",
        )

        decision = evaluator(proposal, gaussians)

        self.assertTrue(decision["accepted"])
        self.assertEqual(decision["reason"], "unit_decreased")
        self.assertAlmostEqual(decision["baseline_risk"], 1.0)
        self.assertAlmostEqual(decision["trial_risk"], 0.8)
        self.assertAlmostEqual(decision["delta_risk"], -0.2)
        self.assertEqual(gaussians.value, "baseline")
        self.assertTrue(gaussians.restored)

        gaussians.restored = False
        scores["trial"] = 0.98
        decision = evaluator(proposal, gaussians)

        self.assertFalse(decision["accepted"])
        self.assertEqual(decision["reason"], "unit_not_decreased")
        self.assertEqual(gaussians.value, "baseline")
        self.assertTrue(gaussians.restored)

    def test_heldout_risk_evaluator_applies_metric_veto_after_risk_drop(self):
        from train_locaware import HeldoutRiskCommitEvaluator

        class FakeGaussians:
            value = "baseline"

        class FakeProposal:
            iteration = 10
            candidate_count = 2
            split_mask = torch.tensor([True, False])
            physical_prune_mask = torch.tensor([False, False])

        scores = {
            "baseline": {"risk": 1.0, "metrics": {"r5_count": 4}},
            "trial": {"risk": 0.8, "metrics": {"r5_count": 3}},
        }

        def metric_gate(baseline, trial):
            return False, "r5_decreased", {"r5_delta": trial["r5_count"] - baseline["r5_count"]}

        evaluator = HeldoutRiskCommitEvaluator(
            score_fn=lambda model: scores[model.value],
            apply_trial_fn=lambda model, proposal: setattr(model, "value", "trial"),
            capture_state_fn=lambda model: {"value": model.value},
            restore_state_fn=lambda model, state: setattr(model, "value", state["value"]),
            epsilon=0.05,
            reason_prefix="unit",
            metric_gate_fn=metric_gate,
        )

        decision = evaluator(FakeProposal(), FakeGaussians())

        self.assertFalse(decision["accepted"])
        self.assertEqual(decision["reason"], "unit_r5_decreased")
        self.assertAlmostEqual(decision["delta_risk"], -0.2)
        self.assertEqual(decision["risk_r5_delta"], -1)

    def test_heldout_risk_evaluator_logs_metric_deltas_for_scalar_rejections(self):
        from train_locaware import HeldoutRiskCommitEvaluator

        class FakeGaussians:
            value = "baseline"

        class FakeProposal:
            iteration = 10
            candidate_count = 2
            split_mask = torch.tensor([True, False])
            physical_prune_mask = torch.tensor([False, False])

        scores = {
            "baseline": {
                "risk": 1.0,
                "metrics": {"count": 4, "r5_count": 2, "r2_count": 1, "tail_fail_count": 2},
            },
            "trial": {
                "risk": 1.1,
                "metrics": {"count": 4, "r5_count": 3, "r2_count": 1, "tail_fail_count": 1},
            },
        }

        def metric_gate(baseline, trial):
            return True, "", {
                "metric_count": min(baseline["count"], trial["count"]),
                "r5_delta": trial["r5_count"] - baseline["r5_count"],
                "r2_delta": trial["r2_count"] - baseline["r2_count"],
                "tail_fail_delta": trial["tail_fail_count"] - baseline["tail_fail_count"],
            }

        evaluator = HeldoutRiskCommitEvaluator(
            score_fn=lambda model: scores[model.value],
            apply_trial_fn=lambda model, proposal: setattr(model, "value", "trial"),
            capture_state_fn=lambda model: {"value": model.value},
            restore_state_fn=lambda model, state: setattr(model, "value", state["value"]),
            epsilon=0.05,
            reason_prefix="unit",
            metric_gate_fn=metric_gate,
        )

        decision = evaluator(FakeProposal(), FakeGaussians())

        self.assertFalse(decision["accepted"])
        self.assertEqual(decision["reason"], "unit_not_decreased")
        self.assertAlmostEqual(decision["delta_risk"], 0.1)
        self.assertEqual(decision["risk_metric_count"], 4)
        self.assertEqual(decision["risk_r5_delta"], 1)
        self.assertEqual(decision["risk_r2_delta"], 0)
        self.assertEqual(decision["risk_tail_fail_delta"], -1)

    def test_heldout_risk_evaluator_uses_paired_ucb_when_enabled(self):
        from train_locaware import HeldoutRiskCommitEvaluator

        class FakeGaussians:
            value = "baseline"

        class FakeProposal:
            iteration = 10
            candidate_count = 2
            split_mask = torch.tensor([True, False])
            physical_prune_mask = torch.tensor([False, False])

        values = {
            "baseline": [1.0, 1.0, 1.0, 1.0],
            "trial": [0.8, 0.8, 0.8, 0.8],
        }
        evaluator = HeldoutRiskCommitEvaluator(
            score_fn=lambda model: values[model.value],
            apply_trial_fn=lambda model, proposal: setattr(model, "value", "trial"),
            capture_state_fn=lambda model: {"value": model.value},
            restore_state_fn=lambda model, state: setattr(model, "value", state["value"]),
            epsilon=0.05,
            ci_z=1.96,
            min_ci_samples=4,
            reason_prefix="unit",
        )

        gaussians = FakeGaussians()
        decision = evaluator(FakeProposal(), gaussians)
        self.assertTrue(decision["accepted"])
        self.assertEqual(decision["reason"], "unit_ucb_decreased")
        self.assertAlmostEqual(decision["delta_risk"], -0.2)
        self.assertAlmostEqual(decision["delta_risk_ucb"], -0.2)
        self.assertEqual(decision["risk_sample_count"], 4)
        self.assertEqual(gaussians.value, "baseline")

        values["trial"] = [0.1, 1.1, 0.9, 0.9]
        decision = evaluator(FakeProposal(), gaussians)
        self.assertFalse(decision["accepted"])
        self.assertEqual(decision["reason"], "unit_ucb_not_decreased")
        self.assertLess(decision["delta_risk"], 0.0)
        self.assertGreater(decision["delta_risk_ucb"], -0.05)
        self.assertEqual(gaussians.value, "baseline")

    def test_heldout_risk_evaluator_rejects_ci_when_sample_count_is_too_small(self):
        from train_locaware import HeldoutRiskCommitEvaluator

        class FakeGaussians:
            value = "baseline"

        class FakeProposal:
            iteration = 10
            candidate_count = 2
            split_mask = torch.tensor([True, False])
            physical_prune_mask = torch.tensor([False, False])

        values = {
            "baseline": [1.0, 1.0],
            "trial": [0.8, 0.8],
        }
        evaluator = HeldoutRiskCommitEvaluator(
            score_fn=lambda model: values[model.value],
            apply_trial_fn=lambda model, proposal: setattr(model, "value", "trial"),
            capture_state_fn=lambda model: {"value": model.value},
            restore_state_fn=lambda model, state: setattr(model, "value", state["value"]),
            epsilon=0.05,
            ci_z=1.96,
            min_ci_samples=4,
            reason_prefix="unit",
        )

        decision = evaluator(FakeProposal(), FakeGaussians())
        self.assertFalse(decision["accepted"])
        self.assertEqual(decision["reason"], "unit_ci_insufficient")
        self.assertEqual(decision["risk_sample_count"], 2)

    def test_pose_recall_tail_veto_rejects_r5_and_tail_regressions(self):
        from train_locaware import _pose_recall_tail_veto

        args = SimpleNamespace(
            topology_risk_pose_veto_mode="r5_r2_tail",
            topology_risk_pose_r2_tolerance=0.25,
            topology_risk_pose_tail_tolerance=0.0,
        )
        baseline = {"count": 4, "r5_count": 3, "r2_count": 2, "tail_fail_count": 1}

        ok, reason, details = _pose_recall_tail_veto(
            baseline,
            {"count": 4, "r5_count": 2, "r2_count": 2, "tail_fail_count": 1},
            args,
        )
        self.assertFalse(ok)
        self.assertEqual(reason, "r5_decreased")
        self.assertEqual(details["r5_delta"], -1)

        ok, reason, details = _pose_recall_tail_veto(
            baseline,
            {"count": 4, "r5_count": 3, "r2_count": 1, "tail_fail_count": 1},
            args,
        )
        self.assertTrue(ok)
        self.assertEqual(reason, "")
        self.assertEqual(details["r2_delta"], -1)

        ok, reason, details = _pose_recall_tail_veto(
            baseline,
            {"count": 4, "r5_count": 3, "r2_count": 2, "tail_fail_count": 2},
            args,
        )
        self.assertFalse(ok)
        self.assertEqual(reason, "tail_increased")
        self.assertEqual(details["tail_fail_delta"], 1)

    def test_heldout_risk_evaluator_restores_rng_state_after_scoring(self):
        from train_locaware import HeldoutRiskCommitEvaluator

        class FakeGaussians:
            value = "baseline"

        class FakeProposal:
            iteration = 10
            candidate_count = 2
            split_mask = torch.tensor([True, False])
            physical_prune_mask = torch.tensor([False, False])

        def score_fn(model):
            torch.rand(3)
            py_random.random()
            return 1.0 if model.value == "baseline" else 0.9

        evaluator = HeldoutRiskCommitEvaluator(
            score_fn=score_fn,
            apply_trial_fn=lambda model, proposal: setattr(model, "value", "trial"),
            capture_state_fn=lambda model: {"value": model.value},
            restore_state_fn=lambda model, state: setattr(model, "value", state["value"]),
            epsilon=0.0,
            reason_prefix="unit",
        )

        torch.manual_seed(1234)
        py_random.seed(5678)
        expected_torch_state = torch.get_rng_state()
        expected_python_state = py_random.getstate()

        decision = evaluator(FakeProposal(), FakeGaussians())
        self.assertTrue(decision["accepted"])

        torch_after = torch.rand(4)
        python_after = py_random.random()
        torch.set_rng_state(expected_torch_state)
        py_random.setstate(expected_python_state)
        self.assertTrue(torch.equal(torch_after, torch.rand(4)))
        self.assertEqual(python_after, py_random.random())

    def test_heldout_risk_evaluator_rejects_nonfinite_trial_risk_and_restores(self):
        from train_locaware import HeldoutRiskCommitEvaluator

        class FakeGaussians:
            value = "baseline"

        class FakeProposal:
            iteration = 5
            candidate_count = 1
            split_mask = torch.tensor([True])
            physical_prune_mask = torch.tensor([False])

        restored = []

        evaluator = HeldoutRiskCommitEvaluator(
            score_fn=lambda model: 1.0 if model.value == "baseline" else float("nan"),
            apply_trial_fn=lambda model, proposal: setattr(model, "value", "trial"),
            capture_state_fn=lambda model: {"value": model.value},
            restore_state_fn=lambda model, state: (setattr(model, "value", state["value"]), restored.append(True)),
            epsilon=0.0,
            reason_prefix="unit",
        )

        gaussians = FakeGaussians()
        decision = evaluator(FakeProposal(), gaussians)

        self.assertFalse(decision["accepted"])
        self.assertEqual(decision["reason"], "unit_nonfinite")
        self.assertEqual(gaussians.value, "baseline")
        self.assertEqual(restored, [True])

    def test_locaware_state_clone_preserves_parameters_for_restore(self):
        from train_locaware import _clone_tensor_tree

        param = torch.nn.Parameter(torch.ones(2, 3), requires_grad=True)
        cloned = _clone_tensor_tree((param, {"buffer": torch.zeros(1)}))

        self.assertIsInstance(cloned[0], torch.nn.Parameter)
        self.assertIsNot(cloned[0], param)
        self.assertTrue(cloned[0].requires_grad)
        self.assertTrue(torch.equal(cloned[0].detach(), param.detach()))
        self.assertFalse(isinstance(cloned[1]["buffer"], torch.nn.Parameter))

    def test_locaware_parser_accepts_descriptor_overlay_controls(self):
        from train_locaware import add_locaware_training_args

        parser = ArgumentParser()
        add_locaware_training_args(parser)

        defaults = parser.parse_args([])
        args = parser.parse_args(
            [
                "--loc_overlay_mode",
                "descriptor",
                "--loc_overlay_lr",
                "0.003",
                "--loc_overlay_active_logit",
                "2.5",
                "--loc_overlay_max_residual_norm",
                "0.2",
                "--loc_overlay_normalize",
                "--loc_overlay_reg_weight",
                "0.01",
            ]
        )

        self.assertEqual(defaults.loc_overlay_mode, "none")
        self.assertEqual(defaults.loc_overlay_max_residual_norm, 0.0)
        self.assertFalse(defaults.loc_overlay_normalize)
        self.assertEqual(defaults.loc_overlay_reg_weight, 0.0)
        self.assertEqual(args.loc_overlay_mode, "descriptor")
        self.assertAlmostEqual(args.loc_overlay_lr, 0.003)
        self.assertAlmostEqual(args.loc_overlay_active_logit, 2.5)
        self.assertAlmostEqual(args.loc_overlay_max_residual_norm, 0.2)
        self.assertTrue(args.loc_overlay_normalize)
        self.assertAlmostEqual(args.loc_overlay_reg_weight, 0.01)

    def test_locaware_parser_accepts_child_responsibility_controls(self):
        from train_locaware import add_locaware_training_args

        parser = ArgumentParser()
        add_locaware_training_args(parser)

        defaults = parser.parse_args([])
        args = parser.parse_args(
            [
                "--loc_child_responsibility_mode",
                "feature",
                "--loc_child_responsibility_start_iter",
                "30625",
            ]
        )

        self.assertEqual(defaults.loc_child_responsibility_mode, "none")
        self.assertEqual(defaults.loc_child_responsibility_start_iter, 0)
        self.assertEqual(args.loc_child_responsibility_mode, "feature")
        self.assertEqual(args.loc_child_responsibility_start_iter, 30625)

    def test_locaware_parser_accepts_full_bank_source_mode(self):
        from train_locaware import add_locaware_training_args

        parser = ArgumentParser()
        add_locaware_training_args(parser)

        defaults = parser.parse_args([])
        args = parser.parse_args(["--loc_full_bank_source_mode", "responsibility"])

        self.assertEqual(defaults.loc_full_bank_source_mode, "ignore")
        self.assertEqual(args.loc_full_bank_source_mode, "responsibility")

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

    def test_diff_pnp_phase_can_unlock_xyz_only_when_geometry_grad_enabled(self):
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
                    ]
                )

        args = SimpleNamespace(
            use_loc_opacity=True,
            lafgs_diff_pnp_allow_geometry_grad=False,
            geometry_xyz_lr_mult=0.05,
            geometry_scale_lr_mult=0.1,
            geometry_rotation_lr_mult=0.1,
        )
        gaussians = FakeGaussians()

        _set_phase_lrs(gaussians, "diff_pnp", args)
        lrs = {group["name"]: group["lr"] for group in gaussians.optimizer.param_groups}
        self.assertEqual(lrs["xyz"], 0.0)
        self.assertEqual(lrs["scaling"], 0.0)
        self.assertEqual(lrs["rotation"], 0.0)
        self.assertEqual(lrs["loc_feature"], 0.04)
        self.assertEqual(lrs["loc_opacity"], 0.05)

        args.lafgs_diff_pnp_allow_geometry_grad = True
        _set_phase_lrs(gaussians, "diff_pnp", args)
        lrs = {group["name"]: group["lr"] for group in gaussians.optimizer.param_groups}
        self.assertEqual(lrs["xyz"], 0.0005)
        self.assertEqual(lrs["scaling"], 0.0)
        self.assertEqual(lrs["rotation"], 0.0)
        self.assertEqual(lrs["loc_feature"], 0.04)
        self.assertEqual(lrs["loc_opacity"], 0.05)

    def test_diff_pnp_geometry_can_use_absolute_xyz_lr_override(self):
        from train_locaware import _set_phase_lrs

        class FakeGaussians:
            def __init__(self):
                self.optimizer = SimpleNamespace(
                    param_groups=[
                        {"name": "xyz", "lr": 2.6e-7},
                        {"name": "scaling", "lr": 0.02},
                        {"name": "rotation", "lr": 0.03},
                        {"name": "loc_feature", "lr": 0.04},
                    ]
                )

        args = SimpleNamespace(
            use_loc_opacity=False,
            lafgs_diff_pnp_allow_geometry_grad=True,
            lafgs_diff_pnp_geometry_xyz_lr=2.0e-5,
            geometry_xyz_lr_mult=0.005,
            geometry_scale_lr_mult=0.1,
            geometry_rotation_lr_mult=0.1,
        )
        gaussians = FakeGaussians()

        _set_phase_lrs(gaussians, "diff_pnp", args)

        lrs = {group["name"]: group["lr"] for group in gaussians.optimizer.param_groups}
        self.assertAlmostEqual(lrs["xyz"], 2.0e-5)
        self.assertEqual(lrs["scaling"], 0.0)
        self.assertEqual(lrs["rotation"], 0.0)
        self.assertEqual(lrs["loc_feature"], 0.04)

    def test_geometry_phase_with_diff_pnp_grad_uses_absolute_xyz_lr_override(self):
        from train_locaware import _set_phase_lrs

        class FakeGaussians:
            def __init__(self):
                self.optimizer = SimpleNamespace(
                    param_groups=[
                        {"name": "xyz", "lr": 2.6e-7},
                        {"name": "scaling", "lr": 0.02},
                        {"name": "rotation", "lr": 0.03},
                        {"name": "loc_feature", "lr": 0.04},
                        {"name": "loc_opacity", "lr": 0.05},
                    ]
                )

        args = SimpleNamespace(
            use_loc_opacity=True,
            lafgs_diff_pnp_allow_geometry_grad=True,
            lafgs_diff_pnp_geometry_xyz_lr=2.0e-5,
            geometry_xyz_lr_mult=0.005,
            geometry_scale_lr_mult=0.0,
            geometry_rotation_lr_mult=0.0,
        )
        gaussians = FakeGaussians()

        _set_phase_lrs(gaussians, "geometry", args)

        lrs = {group["name"]: group["lr"] for group in gaussians.optimizer.param_groups}
        self.assertAlmostEqual(lrs["xyz"], 2.0e-5)
        self.assertEqual(lrs["scaling"], 0.0)
        self.assertEqual(lrs["rotation"], 0.0)
        self.assertEqual(lrs["loc_feature"], 0.04)
        self.assertEqual(lrs["loc_opacity"], 0.05)

    def test_isolated_diff_pnp_geometry_backward_replaces_only_xyz_grad(self):
        from train_locaware import _backward_with_optional_isolated_xyz_grad

        class FakeGaussians:
            def __init__(self):
                self._xyz = torch.nn.Parameter(torch.tensor([1.0]))

        gaussians = FakeGaussians()
        loc_feature = torch.nn.Parameter(torch.tensor([2.0]))
        rgb_feature = torch.nn.Parameter(torch.tensor([3.0]))
        summary = {}

        nonloc_loss = 7.0 * gaussians._xyz + 11.0 * rgb_feature
        diff_pnp_loss = 3.0 * gaussians._xyz + 5.0 * loc_feature
        total_loss = nonloc_loss + diff_pnp_loss

        _backward_with_optional_isolated_xyz_grad(
            total_loss,
            diff_pnp_loss,
            gaussians,
            isolate_xyz_grad=True,
            summary=summary,
        )

        self.assertTrue(torch.allclose(gaussians._xyz.grad, torch.tensor([3.0])))
        self.assertTrue(torch.allclose(loc_feature.grad, torch.tensor([5.0])))
        self.assertTrue(torch.allclose(rgb_feature.grad, torch.tensor([11.0])))
        self.assertAlmostEqual(summary["geometry_xyz_full_grad_abs_max"], 10.0)
        self.assertAlmostEqual(summary["geometry_xyz_isolated_grad_abs_max"], 3.0)
        self.assertEqual(summary["geometry_xyz_isolated_grad_episodes"], 1)

    def test_isolated_diff_pnp_geometry_backward_clears_xyz_grad_without_pnp_loss(self):
        from train_locaware import _backward_with_optional_isolated_xyz_grad

        class FakeGaussians:
            def __init__(self):
                self._xyz = torch.nn.Parameter(torch.tensor([1.0]))

        gaussians = FakeGaussians()
        rgb_feature = torch.nn.Parameter(torch.tensor([3.0]))
        summary = {}

        total_loss = 7.0 * gaussians._xyz + 11.0 * rgb_feature
        no_pnp_loss = torch.tensor(0.0)

        _backward_with_optional_isolated_xyz_grad(
            total_loss,
            no_pnp_loss,
            gaussians,
            isolate_xyz_grad=True,
            summary=summary,
        )

        self.assertIsNone(gaussians._xyz.grad)
        self.assertTrue(torch.allclose(rgb_feature.grad, torch.tensor([11.0])))
        self.assertAlmostEqual(summary["geometry_xyz_full_grad_abs_max"], 7.0)
        self.assertAlmostEqual(summary["geometry_xyz_isolated_grad_abs_max"], 0.0)

    def test_isolated_diff_pnp_geometry_backward_keeps_geometry_regularizer_grad(self):
        from train_locaware import _backward_with_optional_isolated_xyz_grad

        class FakeGaussians:
            def __init__(self):
                self._xyz = torch.nn.Parameter(torch.tensor([1.0]))

        gaussians = FakeGaussians()
        loc_feature = torch.nn.Parameter(torch.tensor([2.0]))
        rgb_feature = torch.nn.Parameter(torch.tensor([3.0]))
        summary = {}

        nonloc_loss = 7.0 * gaussians._xyz + 11.0 * rgb_feature
        diff_pnp_loss = 3.0 * gaussians._xyz + 5.0 * loc_feature
        geometry_regularizer = 13.0 * gaussians._xyz
        total_loss = nonloc_loss + diff_pnp_loss + geometry_regularizer

        _backward_with_optional_isolated_xyz_grad(
            total_loss,
            diff_pnp_loss,
            gaussians,
            isolate_xyz_grad=True,
            summary=summary,
            isolated_xyz_regularizer_loss=geometry_regularizer,
        )

        self.assertTrue(torch.allclose(gaussians._xyz.grad, torch.tensor([16.0])))
        self.assertTrue(torch.allclose(loc_feature.grad, torch.tensor([5.0])))
        self.assertTrue(torch.allclose(rgb_feature.grad, torch.tensor([11.0])))
        self.assertAlmostEqual(summary["geometry_xyz_full_grad_abs_max"], 23.0)
        self.assertAlmostEqual(summary["geometry_xyz_isolated_grad_abs_max"], 16.0)

    def test_isolated_diff_pnp_geometry_backward_keeps_rgb_scaffold_and_drops_other_xyz_grad(self):
        from train_locaware import _backward_with_optional_isolated_xyz_grad

        class FakeGaussians:
            def __init__(self):
                self._xyz = torch.nn.Parameter(torch.tensor([1.0]))

        gaussians = FakeGaussians()
        rgb_feature = torch.nn.Parameter(torch.tensor([2.0]))
        loc_feature = torch.nn.Parameter(torch.tensor([3.0]))
        rgb_scaffold = 7.0 * gaussians._xyz + 11.0 * rgb_feature
        direct_feature_loss = 13.0 * gaussians._xyz + 17.0 * loc_feature
        diff_pnp_loss = 3.0 * gaussians._xyz + 5.0 * loc_feature
        total_loss = rgb_scaffold + direct_feature_loss + diff_pnp_loss

        _backward_with_optional_isolated_xyz_grad(
            total_loss,
            diff_pnp_loss,
            gaussians,
            isolate_xyz_grad=True,
            isolated_xyz_scaffold_loss=rgb_scaffold,
        )

        self.assertTrue(torch.allclose(gaussians._xyz.grad, torch.tensor([10.0])))
        self.assertTrue(torch.allclose(rgb_feature.grad, torch.tensor([11.0])))
        self.assertTrue(torch.allclose(loc_feature.grad, torch.tensor([22.0])))

    def test_lafgs_geometry_gradient_clip_clamps_geometry_params(self):
        from train_locaware import _clip_lafgs_geometry_gradients

        class FakeGaussians:
            def __init__(self):
                self._xyz = torch.nn.Parameter(torch.zeros(2))
                self._loc_anchor_offset = torch.nn.Parameter(torch.zeros(2))
                self._scaling = torch.nn.Parameter(torch.zeros(2))
                self._rotation = torch.nn.Parameter(torch.zeros(2))
                self._xyz.grad = torch.tensor([0.5, -2.0])
                self._loc_anchor_offset.grad = torch.tensor([100.0, -0.25])
                self._scaling.grad = torch.tensor([0.1, 0.2])
                self._rotation.grad = None

        gaussians = FakeGaussians()
        summary = {}

        clipped = _clip_lafgs_geometry_gradients(gaussians, max_abs=1.0, summary=summary)

        self.assertEqual(clipped, 2)
        self.assertTrue(torch.equal(gaussians._xyz.grad, torch.tensor([0.5, -1.0])))
        self.assertTrue(torch.equal(gaussians._loc_anchor_offset.grad, torch.tensor([1.0, -0.25])))
        self.assertTrue(torch.equal(gaussians._scaling.grad, torch.tensor([0.1, 0.2])))
        self.assertEqual(summary["geometry_grad_clip_events"], 1)
        self.assertEqual(summary["geometry_grad_clip_param_events"], 2)
        self.assertEqual(summary["geometry_grad_clip_xyz_events"], 1)
        self.assertEqual(summary["geometry_grad_clip_loc_anchor_offset_events"], 1)
        self.assertEqual(summary["geometry_grad_clip_loc_anchor_offset_before_abs_max"], 100.0)

    def test_locaware_parser_accepts_diff_pnp_isolated_geometry_grad(self):
        from train_locaware import add_locaware_training_args

        parser = ArgumentParser()
        add_locaware_training_args(parser)

        defaults = parser.parse_args([])
        enabled = parser.parse_args(["--lafgs_diff_pnp_isolate_geometry_grad"])

        self.assertFalse(defaults.lafgs_diff_pnp_isolate_geometry_grad)
        self.assertTrue(enabled.lafgs_diff_pnp_isolate_geometry_grad)

    def test_locaware_parser_accepts_lafgs_mvinit_view_selection(self):
        from train_locaware import add_locaware_training_args

        parser = ArgumentParser()
        add_locaware_training_args(parser)

        defaults = parser.parse_args([])
        configured = parser.parse_args(["--lafgs_mvinit_view_selection", "uniform"])

        self.assertEqual(defaults.lafgs_mvinit_view_selection, "first")
        self.assertEqual(configured.lafgs_mvinit_view_selection, "uniform")

    def test_locaware_parser_accepts_lafgs_mvinit_feature_scale(self):
        from train_locaware import add_locaware_training_args

        parser = ArgumentParser()
        add_locaware_training_args(parser)

        defaults = parser.parse_args([])
        configured = parser.parse_args(["--lafgs_mvinit_feature_scale", "0.5"])

        self.assertEqual(defaults.lafgs_mvinit_feature_scale, 1.0)
        self.assertEqual(configured.lafgs_mvinit_feature_scale, 0.5)

    def test_locaware_parser_accepts_diff_pnp_absolute_pose_guard_caps(self):
        from train_locaware import add_locaware_training_args

        parser = ArgumentParser()
        add_locaware_training_args(parser)

        defaults = parser.parse_args([])
        configured = parser.parse_args(
            [
                "--lafgs_diff_pnp_geometry_pose_guard_max_loss",
                "2.0",
                "--lafgs_diff_pnp_feedback_pose_guard_max_loss",
                "3.0",
            ]
        )

        self.assertEqual(defaults.lafgs_diff_pnp_geometry_pose_guard_max_loss, -1.0)
        self.assertEqual(defaults.lafgs_diff_pnp_feedback_pose_guard_max_loss, -1.0)
        self.assertEqual(configured.lafgs_diff_pnp_geometry_pose_guard_max_loss, 2.0)
        self.assertEqual(configured.lafgs_diff_pnp_feedback_pose_guard_max_loss, 3.0)

    def test_locaware_parser_accepts_diff_pnp_soft_feedback_guard(self):
        from train_locaware import add_locaware_training_args

        parser = ArgumentParser()
        add_locaware_training_args(parser)

        defaults = parser.parse_args([])
        configured = parser.parse_args(
            [
                "--lafgs_diff_pnp_feedback_pose_guard_softness",
                "10.0",
                "--lafgs_diff_pnp_feedback_pose_guard_min_scale",
                "0.05",
            ]
        )

        self.assertEqual(defaults.lafgs_diff_pnp_feedback_pose_guard_softness, 0.0)
        self.assertEqual(defaults.lafgs_diff_pnp_feedback_pose_guard_min_scale, 0.0)
        self.assertEqual(configured.lafgs_diff_pnp_feedback_pose_guard_softness, 10.0)
        self.assertEqual(configured.lafgs_diff_pnp_feedback_pose_guard_min_scale, 0.05)

    def test_locaware_parser_accepts_diff_pnp_geometry_local_candidate_pool(self):
        from train_locaware import add_locaware_training_args

        parser = ArgumentParser()
        add_locaware_training_args(parser)

        defaults = parser.parse_args([])
        configured = parser.parse_args(
            [
                "--lafgs_diff_pnp_geometry_use_all_correspondences",
                "--lafgs_diff_pnp_geometry_local_window_radius",
                "1.5",
            ]
        )

        self.assertFalse(defaults.lafgs_diff_pnp_geometry_use_all_correspondences)
        self.assertEqual(defaults.lafgs_diff_pnp_geometry_local_window_radius, 0.0)
        self.assertTrue(configured.lafgs_diff_pnp_geometry_use_all_correspondences)
        self.assertEqual(configured.lafgs_diff_pnp_geometry_local_window_radius, 1.5)

    def test_locaware_parser_accepts_diff_pnp_geometry_soft_guard(self):
        from train_locaware import add_locaware_training_args

        parser = ArgumentParser()
        add_locaware_training_args(parser)

        defaults = parser.parse_args([])
        configured = parser.parse_args(
            [
                "--lafgs_diff_pnp_geometry_pose_guard_softness",
                "10.0",
                "--lafgs_diff_pnp_geometry_pose_guard_min_scale",
                "0.05",
            ]
        )

        self.assertEqual(defaults.lafgs_diff_pnp_geometry_pose_guard_softness, 0.0)
        self.assertEqual(defaults.lafgs_diff_pnp_geometry_pose_guard_min_scale, 0.0)
        self.assertEqual(configured.lafgs_diff_pnp_geometry_pose_guard_softness, 10.0)
        self.assertEqual(configured.lafgs_diff_pnp_geometry_pose_guard_min_scale, 0.05)

    def test_geometry_local_window_requests_projected_uv_without_main_local_window(self):
        from train_locaware import _diff_pnp_needs_projected_uv

        args = SimpleNamespace(
            lafgs_diff_pnp_local_window_radius=0.0,
            lafgs_diff_pnp_geometry_local_window_radius=1.5,
        )

        self.assertTrue(_diff_pnp_needs_projected_uv(args))

    def test_feature_phase_with_descriptor_overlay_trains_overlay_and_freezes_base_feature(self):
        from train_locaware import _set_phase_lrs

        class FakeGaussians:
            def __init__(self):
                self.optimizer = SimpleNamespace(
                    param_groups=[
                        {"name": "xyz", "lr": 0.01},
                        {"name": "loc_feature", "lr": 0.02},
                        {"name": "loc_overlay_feature", "lr": 0.03},
                        {"name": "loc_overlay_active_logit", "lr": 0.04},
                        {"name": "loc_opacity", "lr": 0.05},
                    ]
                )

        gaussians = FakeGaussians()
        args = SimpleNamespace(
            loc_overlay_mode="descriptor",
            geometry_xyz_lr_mult=0.05,
            geometry_scale_lr_mult=0.1,
            geometry_rotation_lr_mult=0.1,
        )

        _set_phase_lrs(gaussians, "feature", args)

        lrs = {group["name"]: group["lr"] for group in gaussians.optimizer.param_groups}
        self.assertEqual(lrs["xyz"], 0.0)
        self.assertEqual(lrs["loc_feature"], 0.0)
        self.assertEqual(lrs["loc_opacity"], 0.0)
        self.assertEqual(lrs["loc_overlay_feature"], 0.03)
        self.assertEqual(lrs["loc_overlay_active_logit"], 0.04)

    def test_descriptor_overlay_configuration_uses_direct_landmark_sources(self):
        from train_locaware import _configure_descriptor_overlay

        class FakeGaussians:
            def __init__(self):
                self.initialized = None
                self.added_lr = None

            def init_descriptor_overlay(
                self,
                source_indices,
                init_active_logit=0.0,
                max_residual_norm=0.0,
                normalize=False,
            ):
                self.initialized = (
                    source_indices.clone(),
                    init_active_logit,
                    max_residual_norm,
                    normalize,
                )

            def add_descriptor_overlay_to_optimizer(self, lr=None):
                self.added_lr = lr

        gaussians = FakeGaussians()
        args = SimpleNamespace(
            loc_overlay_mode="descriptor",
            loc_overlay_lr=0.003,
            loc_overlay_active_logit=2.5,
            loc_overlay_max_residual_norm=0.2,
            loc_overlay_normalize=True,
        )

        configured = _configure_descriptor_overlay(
            gaussians,
            args,
            direct_landmark_indices=torch.tensor([5, 7]),
        )

        self.assertTrue(configured)
        self.assertTrue(torch.equal(gaussians.initialized[0], torch.tensor([5, 7])))
        self.assertAlmostEqual(gaussians.initialized[1], 2.5)
        self.assertAlmostEqual(gaussians.initialized[2], 0.2)
        self.assertTrue(gaussians.initialized[3])
        self.assertAlmostEqual(gaussians.added_lr, 0.003)

    def test_descriptor_overlay_regularizer_uses_gated_residual_norm(self):
        from train_locaware import _descriptor_overlay_regularizer

        class FakeGaussians:
            def __init__(self):
                self._loc_overlay_feature = torch.tensor([[[3.0, 4.0]], [[0.0, 0.0]]])
                self._loc_overlay_active_logit = torch.full((2, 1, 1), 20.0)

            def _has_descriptor_overlay(self):
                return True

        regularizer = _descriptor_overlay_regularizer(FakeGaussians())

        self.assertTrue(torch.allclose(regularizer, torch.tensor(12.5), atol=1e-4))

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
                "--loc_full_bank_ignore_3d_radius",
                "0.25",
                "--loc_full_bank_ignore_uv_radius",
                "2.5",
                "--loc_full_bank_nearby_as_positive",
                "--loc_anchor_weight",
                "0.02",
            ]
        )

        self.assertEqual(args.loc_full_bank_weight, 0.3)
        self.assertEqual(args.loc_full_bank_temperature, 0.11)
        self.assertEqual(args.loc_full_bank_hard_negatives, 16)
        self.assertEqual(args.loc_full_bank_margin, 0.4)
        self.assertEqual(args.loc_full_bank_ignore_3d_radius, 0.25)
        self.assertEqual(args.loc_full_bank_ignore_uv_radius, 2.5)
        self.assertTrue(args.loc_full_bank_nearby_as_positive)
        self.assertEqual(args.loc_anchor_weight, 0.02)

    def test_locaware_parser_accepts_clean_full_bank_controls(self):
        from train_locaware import add_locaware_training_args

        parser = ArgumentParser()
        add_locaware_training_args(parser)
        args = parser.parse_args(
            [
                "--loc_full_bank_balance_weight",
                "0.75",
                "--loc_full_bank_balance_grid_size",
                "4",
                "--loc_full_bank_balance_depth_bins",
                "3",
                "--loc_full_bank_balance_max_weight",
                "5.0",
                "--loc_full_bank_clean_hard_negative_weight",
                "0.5",
                "--loc_clean_hard_negative_weight",
                "0.6",
                "--loc_full_bank_clean_reproj_radius",
                "3.0",
                "--loc_full_bank_clean_hard_negatives",
                "8",
                "--loc_clean_field_start_iter",
                "500",
                "--loc_clean_field_full_bank_weight_scale",
                "0.25",
                "--loc_clean_field_clean_hn_weight_scale",
                "8.0",
                "--loc_clean_field_balance_weight",
                "0.9",
                "--loc_clean_field_pose_information_weight",
                "0.8",
                "--loc_clean_field_diff_pnp_weight_scale",
                "4.0",
            ]
        )

        self.assertEqual(args.loc_full_bank_balance_weight, 0.75)
        self.assertEqual(args.loc_full_bank_balance_grid_size, 4)
        self.assertEqual(args.loc_full_bank_balance_depth_bins, 3)
        self.assertEqual(args.loc_full_bank_balance_max_weight, 5.0)
        self.assertEqual(args.loc_full_bank_clean_hard_negative_weight, 0.5)
        self.assertEqual(args.loc_clean_hard_negative_weight, 0.6)
        self.assertEqual(args.loc_full_bank_clean_reproj_radius, 3.0)
        self.assertEqual(args.loc_full_bank_clean_hard_negatives, 8)
        self.assertEqual(args.loc_clean_field_start_iter, 500)
        self.assertEqual(args.loc_clean_field_full_bank_weight_scale, 0.25)
        self.assertEqual(args.loc_clean_field_clean_hn_weight_scale, 8.0)
        self.assertEqual(args.loc_clean_field_balance_weight, 0.9)
        self.assertEqual(args.loc_clean_field_pose_information_weight, 0.8)
        self.assertEqual(args.loc_clean_field_diff_pnp_weight_scale, 4.0)

    def test_clean_field_stage_controls_apply_after_start_iter(self):
        from train_locaware import _clean_field_stage_controls

        args = SimpleNamespace(
            loc_full_bank_clean_hard_negative_weight=0.5,
            loc_clean_hard_negative_weight=-1.0,
            loc_full_bank_balance_weight=0.25,
            loc_full_bank_pose_information_weight=0.2,
            lafgs_diff_pnp_weight=0.05,
            loc_clean_field_start_iter=100,
            loc_clean_field_full_bank_weight_scale=0.2,
            loc_clean_field_clean_hn_weight_scale=6.0,
            loc_clean_field_balance_weight=0.9,
            loc_clean_field_pose_information_weight=0.8,
            loc_clean_field_diff_pnp_weight_scale=4.0,
        )

        early = _clean_field_stage_controls(args, 50)
        late = _clean_field_stage_controls(args, 100)

        self.assertFalse(early["active"])
        self.assertEqual(early["full_bank_weight_scale"], 1.0)
        self.assertEqual(early["clean_hn_weight"], 0.5)
        self.assertEqual(early["balance_weight"], 0.25)
        self.assertEqual(early["pose_information_weight"], 0.2)
        self.assertEqual(early["diff_pnp_weight"], 0.05)
        self.assertTrue(late["active"])
        self.assertEqual(late["full_bank_weight_scale"], 0.2)
        self.assertEqual(late["clean_hn_weight"], 3.0)
        self.assertEqual(late["balance_weight"], 0.9)
        self.assertEqual(late["pose_information_weight"], 0.8)
        self.assertEqual(late["diff_pnp_weight"], 0.2)

        args.loc_clean_hard_negative_weight = 0.7
        overridden = _clean_field_stage_controls(args, 100)
        self.assertAlmostEqual(overridden["clean_hn_weight"], 4.2)

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

    def test_locaware_parser_accepts_dense_miss_hit_rank_controls(self):
        from train_locaware import add_locaware_training_args

        parser = ArgumentParser()
        add_locaware_training_args(parser)
        defaults = parser.parse_args([])
        args = parser.parse_args(
            [
                "--loc_dense_rank_weight",
                "0.03",
                "--loc_dense_rank_margin",
                "0.4",
                "--loc_dense_rank_teacher_confidence",
                "0.7",
                "--loc_dense_rank_miss_topk",
                "3",
            ]
        )

        self.assertEqual(defaults.loc_dense_rank_weight, 0.0)
        self.assertEqual(defaults.loc_dense_rank_margin, 0.2)
        self.assertEqual(defaults.loc_dense_rank_teacher_confidence, 0.0)
        self.assertEqual(defaults.loc_dense_rank_miss_topk, 1)
        self.assertEqual(args.loc_dense_rank_weight, 0.03)
        self.assertEqual(args.loc_dense_rank_margin, 0.4)
        self.assertEqual(args.loc_dense_rank_teacher_confidence, 0.7)
        self.assertEqual(args.loc_dense_rank_miss_topk, 3)

    def test_locaware_parser_accepts_synthetic_view_controls(self):
        from train_locaware import add_locaware_training_args

        parser = ArgumentParser()
        add_locaware_training_args(parser)
        defaults = parser.parse_args([])
        args = parser.parse_args(
            [
                "--synthetic_view_ratio",
                "0.15",
                "--synthetic_view_candidates",
                "3",
                "--synthetic_view_alpha_min",
                "0.35",
                "--synthetic_view_alpha_max",
                "0.65",
                "--synthetic_view_min_observability",
                "0.25",
                "--synthetic_view_desc_weight",
                "0.05",
                "--synthetic_view_reproj_weight",
                "0.01",
            ]
        )

        self.assertEqual(defaults.synthetic_view_ratio, 0.0)
        self.assertEqual(defaults.synthetic_view_candidates, 1)
        self.assertEqual(defaults.synthetic_view_alpha_min, 0.35)
        self.assertEqual(defaults.synthetic_view_alpha_max, 0.65)
        self.assertEqual(defaults.synthetic_view_min_observability, 0.0)
        self.assertEqual(defaults.synthetic_view_desc_weight, 0.0)
        self.assertEqual(defaults.synthetic_view_reproj_weight, 0.0)
        self.assertEqual(args.synthetic_view_ratio, 0.15)
        self.assertEqual(args.synthetic_view_candidates, 3)
        self.assertEqual(args.synthetic_view_alpha_min, 0.35)
        self.assertEqual(args.synthetic_view_alpha_max, 0.65)
        self.assertEqual(args.synthetic_view_min_observability, 0.25)
        self.assertEqual(args.synthetic_view_desc_weight, 0.05)
        self.assertEqual(args.synthetic_view_reproj_weight, 0.01)

    def test_pseudo_query_teacher_cache_filter_is_disabled_by_default(self):
        from train_locaware import add_locaware_training_args

        parser = ArgumentParser()
        add_locaware_training_args(parser)

        defaults = parser.parse_args([])
        strict = parser.parse_args(
            [
                "--pseudo_query_filter_teacher_cache",
                "--pseudo_query_teacher_allowed_stages",
                "teacher_ok",
            ]
        )

        self.assertFalse(defaults.pseudo_query_filter_teacher_cache)
        self.assertEqual(defaults.pseudo_query_teacher_allowed_stages, "")
        self.assertTrue(strict.pseudo_query_filter_teacher_cache)
        self.assertEqual(strict.pseudo_query_teacher_allowed_stages, "teacher_ok")

    def test_pseudo_query_no_reference_region_weight_downweights_blank_regions(self):
        from la_artifacts.pseudo_query import PseudoQueryRecord
        from train_locaware import _pseudo_query_no_reference_region_weight

        image = torch.zeros(3, 32, 32)
        pattern = torch.linspace(0.0, 1.0, 16).reshape(1, 16).repeat(16, 1)
        image[:, 8:24, 8:24] = pattern
        record = PseudoQueryRecord(
            query_id="synthetic_rgb:synthetic/structured.png",
            scene="ShopFacade",
            source="synthetic_rgb",
            image_name="synthetic/structured.png",
            image_path="",
            pose_w2c=torch.eye(4).tolist(),
            fovx=0.8,
            fovy=0.6,
            width=32,
            height=32,
        )

        weight_map, summary = _pseudo_query_no_reference_region_weight(
            record,
            image,
            enabled=True,
            allowed_sources={"synthetic_rgb"},
            min_weight=0.25,
        )

        self.assertIsNotNone(weight_map)
        self.assertEqual(tuple(weight_map.shape), (32, 32))
        self.assertEqual(summary["enabled"], True)
        self.assertEqual(summary["mode"], "no_reference_region_weight")
        self.assertLess(float(weight_map[:6].mean().item()), 0.10)
        self.assertGreater(float(weight_map[10:22, 10:22].mean().item()), 0.25)

    def test_pseudo_query_no_reference_region_weight_respects_source_filter(self):
        from la_artifacts.pseudo_query import PseudoQueryRecord
        from train_locaware import _pseudo_query_no_reference_region_weight

        record = PseudoQueryRecord(
            query_id="train_rgb:seq/frame.png",
            scene="ShopFacade",
            source="train_rgb",
            image_name="seq/frame.png",
            image_path="",
            pose_w2c=torch.eye(4).tolist(),
            fovx=0.8,
            fovy=0.6,
            width=8,
            height=8,
        )

        weight_map, summary = _pseudo_query_no_reference_region_weight(
            record,
            torch.ones(3, 8, 8),
            enabled=True,
            allowed_sources={"synthetic_rgb"},
        )

        self.assertIsNone(weight_map)
        self.assertEqual(summary["enabled"], False)
        self.assertEqual(summary["reason"], "source_not_enabled")

    def test_pseudo_query_reliability_softly_downweights_bad_teacher_cache(self):
        from la_artifacts.pseudo_query import PseudoQueryRecord
        from la_artifacts.pseudo_query_training import (
            pseudo_query_reliability_decision,
            pseudo_teacher_cache_reliability_stats,
        )

        args = SimpleNamespace(
            pseudo_query_reliability_mode="soft",
            pseudo_query_reliability_min_weight=0.20,
            pseudo_query_reliability_real_min_weight=0.50,
            pseudo_query_reliability_synthetic_min_weight=0.25,
            pseudo_query_reliability_memory_min_weight=0.75,
            pseudo_query_reliability_error_scale=2.0,
            pseudo_query_reliability_inlier_power=0.5,
            pseudo_query_reliability_teacher_ok_weight=1.0,
            pseudo_query_reliability_dense_improves_weight=0.90,
            pseudo_query_reliability_mixed_weight=0.70,
            pseudo_query_reliability_dense_rescues_weight=0.55,
            pseudo_query_reliability_sparse_failure_weight=0.30,
            pseudo_query_reliability_dense_regression_weight=0.35,
            pseudo_query_reliability_unknown_weight=0.60,
        )
        real_record = PseudoQueryRecord(
            query_id="train_rgb:seq/frame.png",
            scene="OldHospital",
            source="train_rgb",
            image_name="seq/frame.png",
            image_path="",
            pose_w2c=torch.eye(4).tolist(),
            fovx=0.8,
            fovy=0.6,
            width=8,
            height=8,
        )
        synthetic_record = PseudoQueryRecord(
            query_id="synthetic_rgb:synthetic/frame.png",
            scene="OldHospital",
            source="synthetic_rgb",
            image_name="synthetic/frame.png",
            image_path="",
            pose_w2c=torch.eye(4).tolist(),
            fovx=0.8,
            fovy=0.6,
            width=8,
            height=8,
        )
        ok_item = {"failure_stage": "teacher_ok", "te": 4.0, "dense_te": 3.0, "inliers": 200, "source": "train_rgb"}
        bad_item = {"failure_stage": "sparse_failure", "te": 80.0, "dense_te": 75.0, "inliers": 20, "source": "train_rgb"}
        synthetic_bad_item = {
            "failure_stage": "sparse_failure",
            "te": 80.0,
            "dense_te": 75.0,
            "inliers": 20,
            "source": "synthetic_rgb",
            "sparse_valid_mask": {"support_frac": 0.2, "valid_frac": 0.3},
        }
        stats = pseudo_teacher_cache_reliability_stats(
            SimpleNamespace(items={"ok": ok_item, "bad": bad_item, "synthetic_bad": synthetic_bad_item})
        )

        ok_decision = pseudo_query_reliability_decision(real_record, ok_item, stats, args)
        bad_decision = pseudo_query_reliability_decision(real_record, bad_item, stats, args)
        synthetic_bad_decision = pseudo_query_reliability_decision(synthetic_record, synthetic_bad_item, stats, args)

        self.assertGreater(ok_decision["weight"], bad_decision["weight"])
        self.assertGreaterEqual(bad_decision["weight"], args.pseudo_query_reliability_real_min_weight)
        self.assertGreaterEqual(synthetic_bad_decision["weight"], args.pseudo_query_reliability_synthetic_min_weight)
        self.assertFalse(bad_decision["update_memory"])
        self.assertFalse(synthetic_bad_decision["update_stats"])
        self.assertTrue(ok_decision["update_memory"])

    def test_pseudo_query_reliability_none_keeps_mainline_unweighted(self):
        from la_artifacts.pseudo_query import PseudoQueryRecord
        from la_artifacts.pseudo_query_training import pseudo_query_reliability_decision

        record = PseudoQueryRecord(
            query_id="synthetic_rgb:synthetic/frame.png",
            scene="ShopFacade",
            source="synthetic_rgb",
            image_name="synthetic/frame.png",
            image_path="",
            pose_w2c=torch.eye(4).tolist(),
            fovx=0.8,
            fovy=0.6,
            width=8,
            height=8,
        )
        args = SimpleNamespace(pseudo_query_reliability_mode="none")
        item = {"failure_stage": "sparse_failure", "te": 500.0, "dense_te": 500.0, "inliers": 5}

        decision = pseudo_query_reliability_decision(record, item, {}, args)

        self.assertFalse(decision["enabled"])
        self.assertEqual(decision["weight"], 1.0)
        self.assertTrue(decision["update_memory"])
        self.assertTrue(decision["update_stats"])

    def test_soft_pseudo_query_reliability_scales_loc_loss(self):
        from train_locaware import _scale_loc_loss_by_pseudo_reliability

        loss = torch.tensor(10.0)
        args = SimpleNamespace(pseudo_query_reliability_loss_mode="soft")
        scaled = _scale_loc_loss_by_pseudo_reliability(loss, {"weight": 0.35}, args)

        self.assertTrue(torch.allclose(scaled, torch.tensor(3.5)))

    def test_disabled_pseudo_query_reliability_keeps_loc_loss_unscaled(self):
        from train_locaware import _scale_loc_loss_by_pseudo_reliability

        loss = torch.tensor(10.0)
        args = SimpleNamespace(pseudo_query_reliability_loss_mode="none")
        scaled = _scale_loc_loss_by_pseudo_reliability(loss, {"weight": 0.35}, args)

        self.assertTrue(torch.allclose(scaled, loss))

    def test_stage_aware_direct_policy_treats_failure_modes_differently(self):
        from train_locaware import _pseudo_query_stage_direct_loss_policy

        args = SimpleNamespace(pseudo_query_stage_objective_mode="direct")

        rescue = _pseudo_query_stage_direct_loss_policy(
            {"stage": "dense_rescues_sparse", "update_memory": True, "update_stats": True},
            args,
        )
        sparse_fail = _pseudo_query_stage_direct_loss_policy(
            {"stage": "sparse_failure", "update_memory": True, "update_stats": True},
            args,
        )

        self.assertTrue(rescue["enabled"])
        self.assertLess(rescue["desc"], rescue["full_bank"])
        self.assertGreaterEqual(rescue["multiview"], rescue["full_bank"])
        self.assertTrue(rescue["update_memory"])

        self.assertTrue(sparse_fail["enabled"])
        self.assertLess(sparse_fail["desc"], 0.5)
        self.assertEqual(sparse_fail["multiview"], 0.0)
        self.assertGreater(sparse_fail["full_bank"], sparse_fail["desc"])
        self.assertEqual(sparse_fail["anchor"], 1.0)
        self.assertFalse(sparse_fail["update_memory"])
        self.assertFalse(sparse_fail["update_stats"])

    def test_stage_aware_direct_policy_scales_loss_components_before_reliability(self):
        from train_locaware import _compose_direct_loc_loss

        args = SimpleNamespace(
            loc_direct_weight=1.0,
            loc_multiview_weight=1.0,
            loc_full_bank_weight=1.0,
            loc_anchor_weight=1.0,
            pseudo_query_stage_objective_mode="direct",
        )

        loss, policy = _compose_direct_loc_loss(
            torch.tensor(1.0),
            torch.tensor(1.0),
            torch.tensor(1.0),
            torch.tensor(1.0),
            {"stage": "sparse_failure", "update_memory": True, "update_stats": True},
            args,
        )

        self.assertTrue(policy["enabled"])
        self.assertTrue(torch.allclose(loss, torch.tensor(2.0)))

        disabled_args = SimpleNamespace(
            loc_direct_weight=1.0,
            loc_multiview_weight=1.0,
            loc_full_bank_weight=1.0,
            loc_anchor_weight=1.0,
            pseudo_query_stage_objective_mode="none",
        )
        disabled_loss, disabled_policy = _compose_direct_loc_loss(
            torch.tensor(1.0),
            torch.tensor(1.0),
            torch.tensor(1.0),
            torch.tensor(1.0),
            {"stage": "sparse_failure", "update_memory": True, "update_stats": True},
            disabled_args,
        )

        self.assertFalse(disabled_policy["enabled"])
        self.assertTrue(torch.allclose(disabled_loss, torch.tensor(4.0)))

    def test_clean_hard_negative_loss_is_not_scaled_by_full_bank_stage_scale(self):
        from train_locaware import _compose_direct_loc_loss

        args = SimpleNamespace(
            loc_direct_weight=0.0,
            loc_multiview_weight=0.0,
            loc_full_bank_weight=10.0,
            loc_clean_hard_negative_weight=3.0,
            loc_anchor_weight=0.0,
            pseudo_query_stage_objective_mode="none",
        )

        loss, _ = _compose_direct_loc_loss(
            torch.tensor(0.0),
            torch.tensor(0.0),
            torch.tensor(2.0),
            torch.tensor(0.0),
            None,
            args,
            full_bank_weight_scale=0.1,
            loc_clean_hard_negative_loss=torch.tensor(5.0),
            clean_hard_negative_weight=3.0,
        )

        # Full-bank is scaled: 10 * 0.1 * 2 = 2. Clean HN remains independent: 3 * 5 = 15.
        self.assertTrue(torch.allclose(loss, torch.tensor(17.0)))

    def test_pseudo_query_stage_source_diagnostics_are_one_hot(self):
        from la_artifacts.pseudo_query import PseudoQueryRecord
        from train_locaware import _pseudo_query_stage_source_diagnostics

        record = PseudoQueryRecord(
            query_id="synthetic_rgb:synthetic/frame.png",
            scene="OldHospital",
            source="synthetic_rgb",
            image_name="synthetic/frame.png",
            image_path="",
            pose_w2c=torch.eye(4).tolist(),
            fovx=0.8,
            fovy=0.6,
            width=8,
            height=8,
        )

        diagnostics = _pseudo_query_stage_source_diagnostics(
            record,
            {"stage": "dense_rescues_sparse"},
        )

        self.assertEqual(diagnostics["pseudo_query_source_synthetic_rgb"], 1.0)
        self.assertEqual(diagnostics["pseudo_query_source_train_rgb"], 0.0)
        self.assertEqual(diagnostics["pseudo_query_source_other"], 0.0)
        self.assertEqual(diagnostics["pseudo_query_stage_dense_rescues_sparse"], 1.0)
        self.assertEqual(diagnostics["pseudo_query_stage_sparse_failure"], 0.0)
        self.assertEqual(diagnostics["pseudo_query_stage_unknown"], 0.0)

    def test_pseudo_query_stage_source_diagnostics_defaults_missing_stage_to_unknown(self):
        from la_artifacts.pseudo_query import PseudoQueryRecord
        from train_locaware import _pseudo_query_stage_source_diagnostics

        record = PseudoQueryRecord(
            query_id="train_rgb:seq/frame.png",
            scene="ShopFacade",
            source="train_rgb",
            image_name="seq/frame.png",
            image_path="",
            pose_w2c=torch.eye(4).tolist(),
            fovx=0.8,
            fovy=0.6,
            width=8,
            height=8,
        )

        diagnostics = _pseudo_query_stage_source_diagnostics(record, {})

        self.assertEqual(diagnostics["pseudo_query_source_train_rgb"], 1.0)
        self.assertEqual(diagnostics["pseudo_query_source_synthetic_rgb"], 0.0)
        self.assertEqual(diagnostics["pseudo_query_stage_unknown"], 1.0)
        self.assertEqual(diagnostics["pseudo_query_stage_teacher_ok"], 0.0)

    def test_pseudo_query_stage_source_diagnostics_include_cross_tab_terms(self):
        from la_artifacts.pseudo_query import PseudoQueryRecord
        from train_locaware import _pseudo_query_stage_source_diagnostics

        record = PseudoQueryRecord(
            query_id="synthetic_rgb:synthetic/frame.png",
            scene="OldHospital",
            source="synthetic_rgb",
            image_name="synthetic/frame.png",
            image_path="",
            pose_w2c=torch.eye(4).tolist(),
            fovx=0.8,
            fovy=0.6,
            width=8,
            height=8,
        )

        diagnostics = _pseudo_query_stage_source_diagnostics(
            record,
            {"stage": "dense_rescues_sparse"},
        )

        cross_keys = [
            key
            for key in diagnostics
            if key.startswith("pseudo_query_source_stage_")
        ]
        active_cross_keys = [key for key in cross_keys if diagnostics[key] == 1.0]

        self.assertEqual(
            active_cross_keys,
            ["pseudo_query_source_stage_synthetic_rgb_dense_rescues_sparse"],
        )
        self.assertEqual(
            diagnostics["pseudo_query_source_stage_train_rgb_dense_rescues_sparse"],
            0.0,
        )
        self.assertEqual(
            diagnostics["pseudo_query_source_stage_synthetic_rgb_sparse_failure"],
            0.0,
        )

    def test_pseudo_query_manifest_alignment_requires_cache_coverage_without_quality_gate(self):
        from la_artifacts.pseudo_query import PseudoQueryManifest, PseudoQueryRecord, PseudoTeacherCache
        from train_locaware import _align_pseudo_manifest_to_teacher_cache

        def record(name, source="train_rgb", teacher_cache_key=""):
            return PseudoQueryRecord(
                query_id=f"{source}:{name}",
                scene="ShopFacade",
                source=source,
                image_name=name,
                image_path="",
                pose_w2c=torch.eye(4).tolist(),
                fovx=0.8,
                fovy=0.6,
                width=8,
                height=8,
                teacher_cache_key=teacher_cache_key,
            )

        manifest = PseudoQueryManifest(
            version=1,
            records=[
                record("ok.png"),
                record("failed_but_cached.png"),
                record("missing.png"),
                record("synthetic/ok.png", source="synthetic_rgb", teacher_cache_key="synthetic:custom"),
            ],
        )
        cache = PseudoTeacherCache(
            {
                "train_rgb:ok.png": {"failed": False, "failure_stage": "teacher_ok"},
                "train_rgb:failed_but_cached.png": {"failed": True, "failure_stage": "sparse_failure"},
                "synthetic:custom": {"failed": False, "failure_stage": "dense_regression_after_good_sparse"},
            }
        )

        aligned, summary = _align_pseudo_manifest_to_teacher_cache(
            manifest,
            cache,
            enabled=True,
        )

        self.assertEqual([row.image_name for row in aligned.records], ["ok.png", "failed_but_cached.png", "synthetic/ok.png"])
        self.assertEqual(summary["before"], 4)
        self.assertEqual(summary["after"], 3)
        self.assertEqual(summary["dropped_missing_teacher_cache"], 1)
        self.assertEqual(summary["quality_filtered"], 0)

    def test_missing_pseudo_teacher_cache_is_optional_for_default_mainline(self):
        from train_locaware import _load_training_pose_cache

        args = SimpleNamespace(
            sparse_pose_cache=None,
            pseudo_teacher_cache="/tmp/does-not-exist/pseudo_teacher_cache.pt",
            pseudo_query_manifest="",
            pseudo_query_require_teacher_cache=True,
            pseudo_query_filter_teacher_cache=False,
            pseudo_query_reliability_mode="none",
        )

        self.assertIsNone(_load_training_pose_cache(args))

    def test_pseudo_query_manifest_requires_teacher_cache_path_by_default(self):
        from train_locaware import _load_training_pose_cache

        args = SimpleNamespace(
            sparse_pose_cache=None,
            pseudo_teacher_cache="/tmp/does-not-exist/pseudo_teacher_cache.pt",
            pseudo_query_manifest="/tmp/pseudo_queries.jsonl",
            pseudo_query_require_teacher_cache=True,
            pseudo_query_filter_teacher_cache=False,
            pseudo_query_reliability_mode="none",
            pseudo_query_stage_objective_mode="none",
        )

        with self.assertRaisesRegex(FileNotFoundError, "Pseudo teacher cache is required"):
            _load_training_pose_cache(args)

    def test_pseudo_query_manifest_can_opt_out_of_teacher_cache_requirement(self):
        from train_locaware import _load_training_pose_cache

        args = SimpleNamespace(
            sparse_pose_cache=None,
            pseudo_teacher_cache="/tmp/does-not-exist/pseudo_teacher_cache.pt",
            pseudo_query_manifest="/tmp/pseudo_queries.jsonl",
            pseudo_query_require_teacher_cache=False,
            pseudo_query_filter_teacher_cache=False,
            pseudo_query_reliability_mode="none",
            pseudo_query_stage_objective_mode="none",
        )

        self.assertIsNone(_load_training_pose_cache(args))

    def test_missing_required_pseudo_teacher_cache_raises_clear_error(self):
        from train_locaware import _load_training_pose_cache

        args = SimpleNamespace(
            sparse_pose_cache=None,
            pseudo_teacher_cache="/tmp/does-not-exist/pseudo_teacher_cache.pt",
            pseudo_query_manifest="",
            pseudo_query_require_teacher_cache=True,
            pseudo_query_filter_teacher_cache=True,
            pseudo_query_reliability_mode="none",
        )

        with self.assertRaisesRegex(FileNotFoundError, "Pseudo teacher cache is required"):
            _load_training_pose_cache(args)

    def test_pseudo_query_stage_objective_requires_teacher_cache(self):
        from train_locaware import _load_training_pose_cache

        args = SimpleNamespace(
            sparse_pose_cache=None,
            pseudo_teacher_cache="/tmp/does-not-exist/pseudo_teacher_cache.pt",
            pseudo_query_require_teacher_cache=True,
            pseudo_query_filter_teacher_cache=False,
            pseudo_query_reliability_mode="none",
            pseudo_query_stage_objective_mode="direct",
            pseudo_query_manifest="/tmp/pseudo_queries.jsonl",
            query_mode="noise",
            mixed_sparse_probability=0.0,
        )

        with self.assertRaisesRegex(FileNotFoundError, "Pseudo teacher cache is required"):
            _load_training_pose_cache(args)

    def test_pseudo_query_sparse_init_requires_teacher_cache(self):
        from train_locaware import _load_training_pose_cache

        args = SimpleNamespace(
            sparse_pose_cache=None,
            pseudo_teacher_cache="/tmp/does-not-exist/pseudo_teacher_cache.pt",
            pseudo_query_manifest="/tmp/pseudo_queries.jsonl",
            pseudo_query_filter_teacher_cache=False,
            pseudo_query_reliability_mode="none",
            query_mode="mixed",
            mixed_sparse_probability=1.0,
        )

        with self.assertRaisesRegex(FileNotFoundError, "Pseudo teacher cache is required"):
            _load_training_pose_cache(args)

    def test_synthetic_view_uses_synthetic_specific_dense_weights(self):
        from train_locaware import _dense_loss_weights_for_episode

        args = SimpleNamespace(
            loc_desc_weight=0.0,
            loc_reproj_weight=0.0,
            loc_dense_kl_weight=0.02,
            loc_dense_rank_weight=0.02,
            synthetic_view_desc_weight=0.05,
            synthetic_view_reproj_weight=0.01,
        )

        real_weights = _dense_loss_weights_for_episode(args, synthetic_view_used=False)
        synthetic_weights = _dense_loss_weights_for_episode(args, synthetic_view_used=True)

        self.assertEqual(real_weights["desc"], 0.0)
        self.assertEqual(real_weights["reproj"], 0.0)
        self.assertEqual(synthetic_weights["desc"], 0.05)
        self.assertEqual(synthetic_weights["reproj"], 0.01)
        self.assertEqual(synthetic_weights["kl"], 0.02)
        self.assertEqual(synthetic_weights["rank"], 0.02)

    def test_locaware_parser_accepts_selective_dense_kl_controls(self):
        from train_locaware import add_locaware_training_args

        parser = ArgumentParser()
        add_locaware_training_args(parser)
        defaults = parser.parse_args([])
        args = parser.parse_args(
            [
                "--loc_dense_pose_gate",
                "--loc_dense_pose_gate_min_te",
                "1.5",
                "--loc_dense_pose_gate_min_ae",
                "0.2",
                "--loc_dense_attr_cosine_threshold",
                "0.8",
                "--loc_dense_attr_entropy_threshold",
                "0.3",
                "--loc_dense_min_positive_prob",
                "0.6",
                "--loc_dense_max_reproj_error",
                "2.0",
                "--loc_dense_min_eligible_anchors",
                "4",
            ]
        )

        self.assertFalse(defaults.loc_dense_pose_gate)
        self.assertEqual(defaults.loc_dense_attr_cosine_threshold, -1.0)
        self.assertTrue(args.loc_dense_pose_gate)
        self.assertEqual(args.loc_dense_pose_gate_min_te, 1.5)
        self.assertEqual(args.loc_dense_pose_gate_min_ae, 0.2)
        self.assertEqual(args.loc_dense_attr_cosine_threshold, 0.8)
        self.assertEqual(args.loc_dense_attr_entropy_threshold, 0.3)
        self.assertEqual(args.loc_dense_min_positive_prob, 0.6)
        self.assertEqual(args.loc_dense_max_reproj_error, 2.0)
        self.assertEqual(args.loc_dense_min_eligible_anchors, 4)

    def test_dense_pose_gate_requires_cached_dense_pose_improvement(self):
        from train_locaware import _dense_pose_improvement_weight

        improved = {"te": 20.0, "ae": 2.0, "dense_te": 10.0, "dense_ae": 1.0}
        worse = {"te": 20.0, "ae": 2.0, "dense_te": 21.0, "dense_ae": 1.0}
        missing = {"te": 20.0, "ae": 2.0}

        self.assertEqual(_dense_pose_improvement_weight(improved, min_te=1.0, min_ae=0.1), 1.0)
        self.assertEqual(_dense_pose_improvement_weight(worse, min_te=1.0, min_ae=0.1), 0.0)
        self.assertEqual(_dense_pose_improvement_weight(missing, min_te=1.0, min_ae=0.1), 0.0)

    def test_locaware_parser_accepts_dense_advantage_gate_controls(self):
        from train_locaware import add_locaware_training_args

        parser = ArgumentParser()
        add_locaware_training_args(parser)
        defaults = parser.parse_args([])
        args = parser.parse_args(
            [
                "--loc_dense_advantage_gate",
                "--loc_dense_advantage_min_te",
                "2.0",
                "--loc_dense_advantage_min_ae",
                "0.2",
                "--loc_dense_advantage_te_scale",
                "8.0",
                "--loc_dense_advantage_ae_scale",
                "0.8",
            ]
        )

        self.assertFalse(defaults.loc_dense_advantage_gate)
        self.assertEqual(defaults.loc_dense_advantage_te_scale, 10.0)
        self.assertTrue(args.loc_dense_advantage_gate)
        self.assertEqual(args.loc_dense_advantage_min_te, 2.0)
        self.assertEqual(args.loc_dense_advantage_min_ae, 0.2)
        self.assertEqual(args.loc_dense_advantage_te_scale, 8.0)
        self.assertEqual(args.loc_dense_advantage_ae_scale, 0.8)

    def test_locaware_parser_defaults_to_no_pseudo_query_stage_gate(self):
        from train_locaware import add_locaware_training_args

        parser = ArgumentParser()
        add_locaware_training_args(parser)
        defaults = parser.parse_args([])
        args = parser.parse_args(["--pseudo_query_exclude_sparse_failure_stages"])

        self.assertFalse(defaults.pseudo_query_exclude_sparse_failure_stages)
        self.assertTrue(args.pseudo_query_exclude_sparse_failure_stages)

    def test_dense_advantage_gate_returns_continuous_pose_weight(self):
        from train_locaware import _dense_pose_advantage_weight

        improved = {"te": 20.0, "ae": 2.0, "dense_te": 12.0, "dense_ae": 1.3}
        no_te_advantage = {"te": 20.0, "ae": 2.0, "dense_te": 20.5, "dense_ae": 1.0}
        missing = {"te": 20.0, "ae": 2.0}

        weight = _dense_pose_advantage_weight(
            improved,
            min_te=2.0,
            min_ae=0.2,
            te_scale=10.0,
            ae_scale=1.0,
        )

        self.assertGreater(weight, 0.0)
        self.assertLess(weight, 1.0)
        self.assertAlmostEqual(weight, 0.5)
        self.assertEqual(
            _dense_pose_advantage_weight(
                no_te_advantage,
                min_te=2.0,
                min_ae=0.2,
                te_scale=10.0,
                ae_scale=1.0,
            ),
            0.0,
        )
        self.assertEqual(_dense_pose_advantage_weight(missing), 0.0)


    def test_locaware_parser_accepts_mixed_query_controls(self):
        from train_locaware import add_locaware_training_args

        parser = ArgumentParser()
        add_locaware_training_args(parser)
        defaults = parser.parse_args([])
        self.assertEqual(defaults.mixed_sparse_probability, 0.5)
        self.assertEqual(defaults.pose_noise_sampling, "empirical")
        self.assertEqual(defaults.train_seed, 0)
        self.assertEqual(defaults.query_split_mode, "random")

        args = parser.parse_args(
            [
                "--train_seed",
                "13",
                "--query_mode",
                "mixed",
                "--mixed_sparse_probability",
                "0.25",
                "--pose_noise_sampling",
                "quantile",
                "--query_split_mode",
                "temporal_block",
            ]
        )
        self.assertEqual(args.train_seed, 13)
        self.assertEqual(args.query_mode, "mixed")
        self.assertEqual(args.mixed_sparse_probability, 0.25)
        self.assertEqual(args.pose_noise_sampling, "quantile")
        self.assertEqual(args.query_split_mode, "temporal_block")

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

    def test_locaware_parser_requires_explicit_override_for_untrained_loc_opacity_prune(self):
        from train_locaware import add_locaware_training_args

        parser = ArgumentParser()
        add_locaware_training_args(parser)

        defaults = parser.parse_args([])
        override = parser.parse_args(["--topology_allow_untrained_loc_opacity_prune"])

        self.assertFalse(defaults.topology_allow_untrained_loc_opacity_prune)
        self.assertTrue(override.topology_allow_untrained_loc_opacity_prune)

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
