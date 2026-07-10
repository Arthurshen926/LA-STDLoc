import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts.run_lafgs_cambridge_guarded_pnp import (
    DEFAULT_SCENES,
    _env,
    build_scene_plan,
    parse_args,
    run_plan,
)
from scripts.summarize_lafgs_cambridge import summarize_cambridge


class LafgsCambridgeExperimentPlanTest(unittest.TestCase):
    def test_runner_env_prefers_conda_bin_before_cuda_bin(self):
        with mock.patch.dict("os.environ", {"PATH": "/usr/bin", "CUDA_HOME": "/usr/local/cuda-11.8"}, clear=True):
            env = _env()

        path_parts = env["PATH"].split(":")
        self.assertEqual(path_parts[0], "/root/miniconda3/envs/cybersim_agent/bin")
        self.assertEqual(path_parts[1], "/usr/local/cuda-11.8/bin")

    def test_runner_cli_accepts_custom_baseline_detector_folder(self):
        args = parse_args(
            [
                "--baseline_detector_folder",
                "detector_baseline_covpreserve_bugfix2_1000",
            ]
        )

        self.assertEqual(args.baseline_detector_folder, "detector_baseline_covpreserve_bugfix2_1000")

    def test_eval_only_custom_detectors_do_not_require_legacy_baseline_detector(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_root = root / "data"
            baseline_root = root / "baseline"
            output_root = root / "lafgs"
            cfg = root / "cfg.yaml"
            scene = "ShopFacade"
            baseline_detector_folder = "detector_baseline_custom"
            lafgs_detector_folder = "detector_lafgs_custom"
            baseline_model = baseline_root / f"{scene}_baseline"
            lafgs_model = output_root / scene

            (data_root / scene).mkdir(parents=True)
            (baseline_model / "point_cloud" / "iteration_30000").mkdir(parents=True)
            (baseline_model / "point_cloud" / "iteration_30000" / "point_cloud.ply").write_text("ply\n")
            (baseline_model / baseline_detector_folder).mkdir(parents=True)
            (baseline_model / baseline_detector_folder / "30000_detector.pth").write_text("det")
            (baseline_model / baseline_detector_folder / "sampled_idx.pkl").write_text("idx")
            (lafgs_model / "point_cloud" / "iteration_30500").mkdir(parents=True)
            (lafgs_model / "point_cloud" / "iteration_30500" / "point_cloud.ply").write_text("ply\n")
            (lafgs_model / lafgs_detector_folder).mkdir(parents=True)
            (lafgs_model / lafgs_detector_folder / "30000_detector.pth").write_text("det")
            (lafgs_model / lafgs_detector_folder / "sampled_idx.pkl").write_text("idx")
            cfg.write_text("cfg")

            plan = build_scene_plan(
                scene,
                data_root=data_root,
                baseline_root=baseline_root,
                output_root=output_root,
                python="python",
                baseline_iterations=30000,
                lafgs_steps=500,
                detect_num=8192,
                nms=2,
                reprojection_error=12.0,
                train_missing_baseline=False,
                force_train=False,
                skip_train=True,
                skip_eval=False,
                eval_baseline=True,
                cfg=cfg,
                baseline_detector_folder=baseline_detector_folder,
                lafgs_detector_folder=lafgs_detector_folder,
            )

            self.assertEqual(plan.status, "ready_eval_only")
            self.assertEqual(plan.missing_reasons, [])
            self.assertIn(
                f"--detector_folder {baseline_detector_folder}",
                " ".join(plan.baseline_eval_cfg_command),
            )
            self.assertIn(
                f"--detector_folder {lafgs_detector_folder}",
                " ".join(plan.lafgs_eval_cfg_command),
            )

            with mock.patch("scripts.run_lafgs_cambridge_guarded_pnp._ensure_symlink") as link_mock:
                run_plan(
                    plan,
                    cwd=root,
                    dry_run=True,
                    force_init=False,
                    force_train=False,
                    train_missing_baseline=False,
                    skip_train=True,
                    skip_detector_train=True,
                    skip_eval=True,
                )

            link_targets = [call.args[1] for call in link_mock.call_args_list]
            self.assertIn(lafgs_model / "point_cloud" / "iteration_30000", link_targets)
            self.assertNotIn(lafgs_model / "detector", link_targets)

    def test_eval_only_requires_matching_lafgs_final_iteration(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_root = root / "data"
            baseline_root = root / "baseline"
            output_root = root / "lafgs"
            cfg = root / "cfg.yaml"
            scene = "ShopFacade"
            baseline_model = baseline_root / f"{scene}_baseline"
            lafgs_model = output_root / scene

            (data_root / scene).mkdir(parents=True)
            (baseline_model / "point_cloud" / "iteration_30000").mkdir(parents=True)
            (baseline_model / "point_cloud" / "iteration_30000" / "point_cloud.ply").write_text("ply\n")
            (baseline_model / "detector").mkdir(parents=True)
            (baseline_model / "detector" / "30000_detector.pth").write_text("det")
            (baseline_model / "detector" / "sampled_idx.pkl").write_text("idx")
            (lafgs_model / "point_cloud" / "iteration_30500").mkdir(parents=True)
            (lafgs_model / "point_cloud" / "iteration_30500" / "point_cloud.ply").write_text("ply\n")
            cfg.write_text("cfg")

            plan = build_scene_plan(
                scene,
                data_root=data_root,
                baseline_root=baseline_root,
                output_root=output_root,
                python="python",
                baseline_iterations=30000,
                lafgs_steps=30000,
                detect_num=8192,
                nms=2,
                reprojection_error=12.0,
                train_missing_baseline=False,
                force_train=False,
                skip_train=True,
                skip_eval=False,
                eval_baseline=True,
                cfg=cfg,
            )

            self.assertEqual(plan.status, "skipped_missing_inputs")
            self.assertTrue(
                any("missing LaFGS point cloud" in reason and "iteration_60000" in reason for reason in plan.missing_reasons)
            )

    def test_guarded_pnp_plan_uses_all_cambridge_scenes_and_no_synthetic_views(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_root = root / "data"
            baseline_root = root / "baseline"
            output_root = root / "lafgs"
            for scene in DEFAULT_SCENES:
                (data_root / scene).mkdir(parents=True)
                model = baseline_root / f"{scene}_baseline"
                (model / "point_cloud" / "iteration_30000").mkdir(parents=True)
                (model / "point_cloud" / "iteration_30000" / "point_cloud.ply").write_text("ply\n")
                (model / "detector").mkdir(parents=True)
                (model / "detector" / "30000_detector.pth").write_text("detector")
                (model / "detector" / "sampled_idx.pkl").write_text("idx")

            plans = [
                build_scene_plan(
                    scene,
                    data_root=data_root,
                    baseline_root=baseline_root,
                    output_root=output_root,
                    python="python",
                    baseline_iterations=30000,
                    lafgs_steps=500,
                    detect_num=8192,
                    nms=2,
                    reprojection_error=12.0,
                    train_missing_baseline=False,
                    force_train=False,
                    skip_train=False,
                    skip_eval=False,
                    eval_baseline=True,
                    cfg="configs/stdloc_cambridge.yaml",
                    geometry_confidence_threshold=0.02,
                    geometry_margin_threshold=0.002,
                    geometry_peak_probability_threshold=0.8,
                    geometry_max_entropy=0.35,
                    geometry_max_reprojection_error=4.0,
                    geometry_match_reprojection_weight=0.5,
                    geometry_match_peak_probability_threshold=0.8,
                    geometry_match_max_entropy=0.4,
                    geometry_match_max_reprojection_error=2.0,
                    pnp_local_window_radius=1.25,
                    max_condition_number=100000.0,
                    eval_diagnostics_grid_rows=3,
                    eval_diagnostics_grid_cols=5,
                    eval_diagnostics_voxel_size=0.5,
                )
                for scene in DEFAULT_SCENES
            ]

            self.assertEqual([plan.scene for plan in plans], DEFAULT_SCENES)
            for plan in plans:
                train_cmd = " ".join(plan.train_lafgs_command)
                self.assertIn("--synthetic_view_ratio 0.0", train_cmd)
                self.assertIn("--synthetic_view_desc_weight 0.0", train_cmd)
                self.assertIn("--synthetic_view_reproj_weight 0.0", train_cmd)
                self.assertNotIn("--lafgs_diff_pnp_feedback_pose_guard_keep_gt_reprojection", train_cmd)
                self.assertIn("--lafgs_diff_pnp_allow_geometry_grad", train_cmd)
                self.assertIn("--lafgs_diff_pnp_geometry_pose_guard_max_loss 5.0", train_cmd)
                self.assertIn("--lafgs_diff_pnp_geometry_pose_guard_softness 10.0", train_cmd)
                self.assertIn("--lafgs_diff_pnp_geometry_pose_guard_min_scale 0.05", train_cmd)
                self.assertNotIn("--lafgs_diff_pnp_geometry_use_all_correspondences", train_cmd)
                self.assertIn("--lafgs_diff_pnp_geometry_local_window_radius 1.5", train_cmd)
                self.assertIn("--lafgs_diff_pnp_utility_pose_loss_scale 1.0", train_cmd)
                self.assertIn("--lafgs_diff_pnp_utility_reprojection_error_scale 4.0", train_cmd)
                self.assertIn("--lafgs_diff_pnp_geometry_confidence_threshold 0.02", train_cmd)
                self.assertIn("--lafgs_diff_pnp_geometry_margin_threshold 0.002", train_cmd)
                self.assertIn("--lafgs_diff_pnp_geometry_peak_probability_threshold 0.8", train_cmd)
                self.assertIn("--lafgs_diff_pnp_geometry_max_entropy 0.35", train_cmd)
                self.assertIn("--lafgs_diff_pnp_geometry_max_reproj_error 4.0", train_cmd)
                self.assertIn("--lafgs_diff_pnp_geometry_match_reproj_weight 0.5", train_cmd)
                self.assertIn("--lafgs_diff_pnp_geometry_match_peak_probability_threshold 0.8", train_cmd)
                self.assertIn("--lafgs_diff_pnp_geometry_match_max_entropy 0.4", train_cmd)
                self.assertIn("--lafgs_diff_pnp_geometry_match_max_reproj_error 2.0", train_cmd)
                self.assertIn("--lafgs_diff_pnp_local_window_radius 1.25", train_cmd)
                self.assertIn("--lafgs_diff_pnp_max_condition_number 100000.0", train_cmd)
                self.assertIn("--lafgs_diff_pnp_feedback_pose_guard_max_loss 5.0", train_cmd)
                self.assertIn("--lafgs_diff_pnp_feedback_pose_guard_softness 10.0", train_cmd)
                self.assertIn("--lafgs_diff_pnp_feedback_pose_guard_min_scale 0.05", train_cmd)
                self.assertIn("--lafgs_mvinit_max_views 64", train_cmd)
                self.assertIn("--lafgs_mvinit_feature_scale 0.5", train_cmd)
                self.assertIn("--loc_full_bank_nearby_as_positive", train_cmd)
                self.assertIn("--loc_full_bank_nearby_as_positive_until 10000", train_cmd)
                self.assertEqual(plan.detector_source, baseline_root / f"{plan.scene}_baseline" / "detector")
                baseline_detector_cmd = " ".join(plan.train_baseline_detector_command)
                self.assertIn("train_detector.py", baseline_detector_cmd)
                self.assertIn("--iteration 30000", baseline_detector_cmd)
                self.assertIn("--detector_folder detector_baseline_covpreserve", baseline_detector_cmd)
                self.assertIn("--sampling_mode coverage_preserving", baseline_detector_cmd)
                self.assertIn("--detector_target_mode weighted_hard", baseline_detector_cmd)
                detector_cmd = " ".join(plan.train_lafgs_detector_command)
                self.assertIn("train_detector.py", detector_cmd)
                self.assertIn("--iteration 30500", detector_cmd)
                self.assertIn("--detector_folder detector_lafgs", detector_cmd)
                self.assertIn("--sampling_mode coverage_preserving", detector_cmd)
                self.assertIn("--detector_target_mode weighted_hard", detector_cmd)
                self.assertIn("--coverage_preserve_ratio 0.75", detector_cmd)
                self.assertIn("--coverage_utility_ratio 0.1", detector_cmd)
                self.assertIn("--coverage_high_confidence_ratio 0.1", detector_cmd)
                self.assertIn("--coverage_grid_size 4", detector_cmd)
                self.assertIn("--coverage_max_per_grid 1536", detector_cmd)
                self.assertIn("--coverage_depth_bins 4", detector_cmd)
                self.assertIn("--coverage_max_per_depth_bin 6144", detector_cmd)
                baseline_eval_cfg_cmd = " ".join(plan.baseline_eval_cfg_command)
                self.assertIn("--detector_folder detector_baseline_covpreserve", baseline_eval_cfg_cmd)
                self.assertIn("--detector_iters 30000", baseline_eval_cfg_cmd)
                eval_cfg_cmd = " ".join(plan.lafgs_eval_cfg_command)
                self.assertIn("--detector_folder detector_lafgs", eval_cfg_cmd)
                self.assertIn("--detector_iters 30000", eval_cfg_cmd)
                self.assertIn("--diagnostics_grid_rows 3", eval_cfg_cmd)
                self.assertIn("--diagnostics_grid_cols 5", eval_cfg_cmd)
                self.assertIn("--diagnostics_voxel_size 0.5", eval_cfg_cmd)
                self.assertEqual(plan.final_iteration, 30500)

    def test_guarded_pnp_plan_legacy_lafgs_eval_can_reuse_baseline_detector_explicitly(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_root = root / "data"
            baseline_root = root / "baseline"
            output_root = root / "lafgs"
            scene = "ShopFacade"
            (data_root / scene).mkdir(parents=True)
            model = baseline_root / f"{scene}_baseline"
            (model / "point_cloud" / "iteration_30000").mkdir(parents=True)
            (model / "point_cloud" / "iteration_30000" / "point_cloud.ply").write_text("ply\n")
            (model / "detector").mkdir(parents=True)
            (model / "detector" / "30000_detector.pth").write_text("detector")
            (model / "detector" / "sampled_idx.pkl").write_text("idx")

            plan = build_scene_plan(
                scene,
                data_root=data_root,
                baseline_root=baseline_root,
                output_root=output_root,
                python="python",
                baseline_iterations=30000,
                lafgs_steps=500,
                detect_num=8192,
                nms=2,
                reprojection_error=12.0,
                train_missing_baseline=False,
                force_train=False,
                skip_train=False,
                skip_eval=False,
                eval_baseline=True,
                cfg="configs/stdloc_cambridge.yaml",
                lafgs_detector_source="baseline",
            )

            self.assertEqual(plan.train_lafgs_detector_command, [])
            self.assertIn("--detector_folder detector", " ".join(plan.lafgs_eval_cfg_command))
            self.assertIn("lafgs-guarded-pnp-30500-detector", plan.lafgs_eval_command)

    def test_runner_trains_baseline_retrained_detector_even_when_lafgs_reuses_legacy_detector(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_root = root / "data"
            baseline_root = root / "baseline"
            output_root = root / "lafgs"
            scene = "ShopFacade"
            (data_root / scene).mkdir(parents=True)
            model = baseline_root / f"{scene}_baseline"
            (model / "point_cloud" / "iteration_30000").mkdir(parents=True)
            (model / "point_cloud" / "iteration_30000" / "point_cloud.ply").write_text("ply\n")
            (model / "detector").mkdir(parents=True)
            (model / "detector" / "30000_detector.pth").write_text("detector")
            (model / "detector" / "sampled_idx.pkl").write_text("idx")

            plan = build_scene_plan(
                scene,
                data_root=data_root,
                baseline_root=baseline_root,
                output_root=output_root,
                python="python",
                baseline_iterations=30000,
                lafgs_steps=500,
                detect_num=8192,
                nms=2,
                reprojection_error=12.0,
                train_missing_baseline=False,
                force_train=False,
                skip_train=True,
                skip_eval=True,
                eval_baseline=True,
                cfg="configs/stdloc_cambridge.yaml",
                lafgs_detector_source="baseline",
            )

            with mock.patch("scripts.run_lafgs_cambridge_guarded_pnp._run") as run_mock:
                run_plan(
                    plan,
                    cwd=root,
                    dry_run=True,
                    force_init=False,
                    force_train=False,
                    train_missing_baseline=False,
                    skip_train=True,
                    skip_detector_train=False,
                    skip_eval=True,
                )

        commands = [" ".join(call.args[0]) for call in run_mock.call_args_list]
        self.assertTrue(
            any("--detector_folder detector_baseline_covpreserve" in command for command in commands),
            commands,
        )

    def test_guarded_pnp_plan_keeps_gt_reprojection_only_when_explicitly_requested(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_root = root / "data"
            baseline_root = root / "baseline"
            output_root = root / "lafgs"
            scene = "GreatCourt"
            (data_root / scene).mkdir(parents=True)
            model = baseline_root / f"{scene}_baseline"
            (model / "point_cloud" / "iteration_30000").mkdir(parents=True)
            (model / "point_cloud" / "iteration_30000" / "point_cloud.ply").write_text("ply\n")
            (model / "detector").mkdir(parents=True)
            (model / "detector" / "30000_detector.pth").write_text("detector")
            (model / "detector" / "sampled_idx.pkl").write_text("idx")

            plan = build_scene_plan(
                scene,
                data_root=data_root,
                baseline_root=baseline_root,
                output_root=output_root,
                python="python",
                baseline_iterations=30000,
                lafgs_steps=500,
                detect_num=8192,
                nms=2,
                reprojection_error=12.0,
                train_missing_baseline=False,
                force_train=False,
                skip_train=False,
                skip_eval=False,
                eval_baseline=True,
                cfg="configs/stdloc_cambridge.yaml",
                feedback_pose_guard_keep_gt_reprojection=True,
            )

            train_cmd = " ".join(plan.train_lafgs_command)
            self.assertIn("--lafgs_diff_pnp_feedback_pose_guard_keep_gt_reprojection", train_cmd)

    def test_guarded_pnp_plan_can_use_selected_geometry_correspondences(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_root = root / "data"
            baseline_root = root / "baseline"
            output_root = root / "lafgs"
            scene = "OldHospital"
            (data_root / scene).mkdir(parents=True)
            model = baseline_root / f"{scene}_baseline"
            (model / "point_cloud" / "iteration_30000").mkdir(parents=True)
            (model / "point_cloud" / "iteration_30000" / "point_cloud.ply").write_text("ply\n")
            (model / "detector").mkdir(parents=True)
            (model / "detector" / "30000_detector.pth").write_text("detector")
            (model / "detector" / "sampled_idx.pkl").write_text("idx")

            plan = build_scene_plan(
                scene,
                data_root=data_root,
                baseline_root=baseline_root,
                output_root=output_root,
                python="python",
                baseline_iterations=30000,
                lafgs_steps=500,
                detect_num=8192,
                nms=2,
                reprojection_error=12.0,
                train_missing_baseline=False,
                force_train=False,
                skip_train=False,
                skip_eval=False,
                eval_baseline=True,
                cfg="configs/stdloc_cambridge.yaml",
                geometry_use_all_correspondences=False,
                geometry_match_reprojection_weight=0.25,
            )

            train_cmd = " ".join(plan.train_lafgs_command)
            self.assertIn("--lafgs_diff_pnp_geometry_match_reproj_weight 0.25", train_cmd)
            self.assertNotIn("--lafgs_diff_pnp_geometry_use_all_correspondences", train_cmd)

    def test_guarded_pnp_plan_builds_long_run_checkpoint_eval_commands(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_root = root / "data"
            baseline_root = root / "baseline"
            output_root = root / "lafgs"
            cfg = root / "cfg.yaml"
            scene = "ShopFacade"
            (data_root / scene).mkdir(parents=True)
            model = baseline_root / f"{scene}_baseline"
            (model / "point_cloud" / "iteration_30000").mkdir(parents=True)
            (model / "point_cloud" / "iteration_30000" / "point_cloud.ply").write_text("ply\n")
            (model / "detector").mkdir(parents=True)
            (model / "detector" / "30000_detector.pth").write_text("detector")
            (model / "detector" / "sampled_idx.pkl").write_text("idx")
            cfg.write_text("cfg")

            plan = build_scene_plan(
                scene,
                data_root=data_root,
                baseline_root=baseline_root,
                output_root=output_root,
                python="python",
                baseline_iterations=30000,
                lafgs_steps=10000,
                detect_num=8192,
                nms=2,
                reprojection_error=12.0,
                train_missing_baseline=False,
                force_train=False,
                skip_train=False,
                skip_eval=False,
                eval_baseline=True,
                cfg=cfg,
                gaussian_type="2dgs",
                loc_interval=2,
                checkpoint_eval_interval=1000,
                checkpoint_detector_iterations=1000,
                lafgs_detector_folder="detector_lafgs_a3_10k",
                diff_pnp_weight=0.0001,
                geometry_reprojection_weight=0.01,
                geometry_depth_anchor_weight=0.1,
                loc_anchor_lr=5e-7,
                surfel_loc_tangent_bound=0.03,
                surfel_loc_normal_bound=0.005,
            )

            train_cmd = " ".join(plan.train_lafgs_command)
            self.assertEqual(plan.final_iteration, 40000)
            self.assertIn("--loc_interval 2", train_cmd)
            self.assertIn("--iterations 40000", train_cmd)
            self.assertIn("--lafgs_diff_pnp_weight 0.0001", train_cmd)
            self.assertIn("--lafgs_diff_pnp_geometry_reproj_weight 0.01", train_cmd)
            self.assertIn("--lafgs_diff_pnp_geometry_depth_anchor_weight 0.1", train_cmd)
            self.assertIn("--loc_anchor_lr 5e-07", train_cmd)
            self.assertIn("--surfel_loc_tangent_bound 0.03", train_cmd)
            self.assertIn("--surfel_loc_normal_bound 0.005", train_cmd)
            self.assertIn("--save_iterations 31000 32000 33000 34000 35000 36000 37000 38000 39000 40000", train_cmd)
            self.assertIn("--test_iterations 31000 32000 33000 34000 35000 36000 37000 38000 39000 40000", train_cmd)

            self.assertEqual(len(plan.checkpoint_lafgs_detector_commands), 10)
            self.assertEqual(len(plan.checkpoint_lafgs_eval_cfg_commands), 10)
            self.assertEqual(len(plan.checkpoint_lafgs_eval_commands), 10)
            first_detector = " ".join(plan.checkpoint_lafgs_detector_commands[0])
            first_eval_cfg = " ".join(plan.checkpoint_lafgs_eval_cfg_commands[0])
            first_eval = " ".join(plan.checkpoint_lafgs_eval_commands[0])
            self.assertIn("--iteration 31000", first_detector)
            self.assertIn("--iterations 1000", first_detector)

    def test_guarded_pnp_default_profile_uses_non_tiny_feedback_and_radius_floor(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_root = root / "data"
            baseline_root = root / "baseline"
            output_root = root / "lafgs"
            cfg = root / "cfg.yaml"
            scene = "ShopFacade"
            (data_root / scene).mkdir(parents=True)
            model = baseline_root / f"{scene}_baseline"
            (model / "point_cloud" / "iteration_30000").mkdir(parents=True)
            (model / "point_cloud" / "iteration_30000" / "point_cloud.ply").write_text("ply\n")
            (model / "detector").mkdir(parents=True)
            (model / "detector" / "30000_detector.pth").write_text("detector")
            (model / "detector" / "sampled_idx.pkl").write_text("idx")
            cfg.write_text("cfg")

            plan = build_scene_plan(
                scene,
                data_root=data_root,
                baseline_root=baseline_root,
                output_root=output_root,
                python="python",
                baseline_iterations=30000,
                lafgs_steps=500,
                detect_num=8192,
                nms=2,
                reprojection_error=12.0,
                train_missing_baseline=False,
                force_train=False,
                skip_train=False,
                skip_eval=False,
                eval_baseline=True,
                cfg=cfg,
                gaussian_type="2dgs",
            )

            train_cmd = " ".join(plan.train_lafgs_command)
            self.assertIn("--loc_interval 1", train_cmd)
            self.assertIn("--lafgs_diff_pnp_weight 0.05", train_cmd)
            self.assertIn("--lafgs_diff_pnp_geometry_match_reproj_weight 0.5", train_cmd)
            self.assertIn("--lafgs_diff_pnp_geometry_match_max_reproj_error 2.0", train_cmd)
            self.assertIn("--loc_anchor_lr 5e-05", train_cmd)
            self.assertIn("--surfel_loc_tangent_bound 0.03", train_cmd)
            self.assertIn("--surfel_loc_normal_bound 0.005", train_cmd)
            self.assertIn("--surfel_loc_radius_floor 1.0", train_cmd)

    def test_runner_executes_checkpoint_curve_and_final_detector_eval(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_root = root / "data"
            baseline_root = root / "baseline"
            output_root = root / "lafgs"
            cfg = root / "cfg.yaml"
            scene = "ShopFacade"
            (data_root / scene).mkdir(parents=True)
            baseline_model = baseline_root / f"{scene}_baseline"
            (baseline_model / "point_cloud" / "iteration_30000").mkdir(parents=True)
            (baseline_model / "point_cloud" / "iteration_30000" / "point_cloud.ply").write_text("ply\n")
            (baseline_model / "detector").mkdir(parents=True)
            (baseline_model / "detector" / "30000_detector.pth").write_text("detector")
            (baseline_model / "detector" / "sampled_idx.pkl").write_text("idx")
            lafgs_model = output_root / scene
            for iteration in (31000, 32000):
                (lafgs_model / "point_cloud" / f"iteration_{iteration}").mkdir(parents=True)
                (lafgs_model / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply").write_text("ply\n")
            cfg.write_text("cfg")

            plan = build_scene_plan(
                scene,
                data_root=data_root,
                baseline_root=baseline_root,
                output_root=output_root,
                python="python",
                baseline_iterations=30000,
                lafgs_steps=2000,
                detect_num=8192,
                nms=2,
                reprojection_error=12.0,
                train_missing_baseline=False,
                force_train=False,
                skip_train=True,
                skip_eval=False,
                eval_baseline=True,
                cfg=cfg,
                checkpoint_eval_interval=1000,
                checkpoint_detector_iterations=1000,
                lafgs_detector_folder="detector_lafgs_final",
                lafgs_detector_iterations=30000,
            )

            with mock.patch("scripts.run_lafgs_cambridge_guarded_pnp._run") as run_mock:
                run_plan(
                    plan,
                    cwd=root,
                    dry_run=True,
                    force_init=False,
                    force_train=False,
                    train_missing_baseline=False,
                    skip_train=True,
                    skip_detector_train=False,
                    skip_eval=False,
                )

        commands = [" ".join(call.args[0]) for call in run_mock.call_args_list]
        self.assertTrue(any("--detector_folder detector_lafgs_final_ckpt_31000" in command for command in commands))
        self.assertTrue(any("--detector_folder detector_lafgs_final_ckpt_32000" in command for command in commands))
        self.assertTrue(any("--detector_folder detector_lafgs_final --landmark_num" in command for command in commands))
        self.assertTrue(any("--prefix lafgs-guarded-pnp-31000-detector_lafgs_final_ckpt_31000" in command for command in commands))
        self.assertTrue(any("--prefix lafgs-guarded-pnp-32000-detector_lafgs_final_ckpt_32000" in command for command in commands))
        self.assertTrue(any("--prefix lafgs-guarded-pnp-32000-detector_lafgs_final" in command for command in commands))

    def test_guarded_pnp_plan_can_use_2dgs_surface_bootstrap(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_root = root / "data"
            baseline_root = root / "baseline"
            output_root = root / "lafgs"
            scene = "OldHospital"
            (data_root / scene).mkdir(parents=True)
            model = baseline_root / f"{scene}_baseline"
            (model / "point_cloud" / "iteration_30000").mkdir(parents=True)
            (model / "point_cloud" / "iteration_30000" / "point_cloud.ply").write_text("ply\n")
            (model / "detector").mkdir(parents=True)
            (model / "detector" / "30000_detector.pth").write_text("detector")
            (model / "detector" / "sampled_idx.pkl").write_text("idx")

            plan = build_scene_plan(
                scene,
                data_root=data_root,
                baseline_root=baseline_root,
                output_root=output_root,
                python="python",
                baseline_iterations=30000,
                lafgs_steps=500,
                detect_num=8192,
                nms=2,
                reprojection_error=12.0,
                train_missing_baseline=False,
                force_train=False,
                skip_train=False,
                skip_eval=False,
                eval_baseline=True,
                cfg="configs/stdloc_cambridge.yaml",
                gaussian_type="2dgs",
            )

            self.assertIn("-g", plan.train_lafgs_command)
            self.assertEqual(plan.train_lafgs_command[plan.train_lafgs_command.index("-g") + 1], "2dgs")
            self.assertEqual(plan.train_baseline_command[plan.train_baseline_command.index("-g") + 1], "2dgs")
            self.assertEqual(plan.lafgs_eval_command[plan.lafgs_eval_command.index("-g") + 1], "2dgs")

    def test_guarded_pnp_plan_enables_pose_information_weight_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_root = root / "data"
            baseline_root = root / "baseline"
            output_root = root / "lafgs"
            scene = "OldHospital"
            (data_root / scene).mkdir(parents=True)
            model = baseline_root / f"{scene}_baseline"
            (model / "point_cloud" / "iteration_30000").mkdir(parents=True)
            (model / "point_cloud" / "iteration_30000" / "point_cloud.ply").write_text("ply\n")
            (model / "detector").mkdir(parents=True)
            (model / "detector" / "30000_detector.pth").write_text("detector")
            (model / "detector" / "sampled_idx.pkl").write_text("idx")

            plan = build_scene_plan(
                scene,
                data_root=data_root,
                baseline_root=baseline_root,
                output_root=output_root,
                python="python",
                baseline_iterations=30000,
                lafgs_steps=500,
                detect_num=8192,
                nms=2,
                reprojection_error=12.0,
                train_missing_baseline=False,
                force_train=False,
                skip_train=False,
                skip_eval=False,
                eval_baseline=True,
                cfg="configs/stdloc_cambridge.yaml",
            )

            train_cmd = " ".join(plan.train_lafgs_command)
            self.assertIn("--loc_full_bank_pose_information_weight 0.5", train_cmd)
            self.assertIn("--loc_full_bank_pose_information_floor 0.2", train_cmd)

    def test_guarded_pnp_plan_can_enable_clean_field_objective(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_root = root / "data"
            baseline_root = root / "baseline"
            output_root = root / "lafgs"
            scene = "OldHospital"
            (data_root / scene).mkdir(parents=True)
            model = baseline_root / f"{scene}_baseline"
            (model / "point_cloud" / "iteration_30000").mkdir(parents=True)
            (model / "point_cloud" / "iteration_30000" / "point_cloud.ply").write_text("ply\n")
            (model / "detector").mkdir(parents=True)
            (model / "detector" / "30000_detector.pth").write_text("detector")
            (model / "detector" / "sampled_idx.pkl").write_text("idx")

            plan = build_scene_plan(
                scene,
                data_root=data_root,
                baseline_root=baseline_root,
                output_root=output_root,
                python="python",
                baseline_iterations=30000,
                lafgs_steps=500,
                detect_num=8192,
                nms=2,
                reprojection_error=12.0,
                train_missing_baseline=False,
                force_train=False,
                skip_train=False,
                skip_eval=False,
                eval_baseline=True,
                cfg="configs/stdloc_cambridge.yaml",
                full_bank_balance_weight=0.5,
                full_bank_balance_grid_size=4,
                full_bank_balance_depth_bins=3,
                full_bank_balance_max_weight=5.0,
                full_bank_clean_hard_negative_weight=0.35,
                full_bank_clean_reproj_radius=3.0,
                full_bank_clean_hard_negatives=12,
                clean_field_start_iter=300,
                clean_field_full_bank_weight_scale=0.25,
                clean_field_clean_hn_weight_scale=8.0,
                clean_field_balance_weight=0.9,
                clean_field_pose_information_weight=0.8,
                clean_field_diff_pnp_weight_scale=4.0,
            )

            train_cmd = " ".join(plan.train_lafgs_command)
            self.assertIn("--loc_full_bank_balance_weight 0.5", train_cmd)
            self.assertIn("--loc_full_bank_balance_grid_size 4", train_cmd)
            self.assertIn("--loc_full_bank_balance_depth_bins 3", train_cmd)
            self.assertIn("--loc_full_bank_balance_max_weight 5.0", train_cmd)
            self.assertIn("--loc_full_bank_clean_hard_negative_weight 0.35", train_cmd)
            self.assertIn("--loc_full_bank_clean_reproj_radius 3.0", train_cmd)
            self.assertIn("--loc_full_bank_clean_hard_negatives 12", train_cmd)
            self.assertIn("--loc_clean_field_start_iter 300", train_cmd)
            self.assertIn("--loc_clean_field_full_bank_weight_scale 0.25", train_cmd)
            self.assertIn("--loc_clean_field_clean_hn_weight_scale 8.0", train_cmd)
            self.assertIn("--loc_clean_field_balance_weight 0.9", train_cmd)
            self.assertIn("--loc_clean_field_pose_information_weight 0.8", train_cmd)
            self.assertIn("--loc_clean_field_diff_pnp_weight_scale 4.0", train_cmd)

    def test_runner_cli_accepts_clean_field_objective(self):
        args = parse_args(
            [
                "--full_bank_balance_weight",
                "0.5",
                "--full_bank_balance_grid_size",
                "4",
                "--full_bank_balance_depth_bins",
                "3",
                "--full_bank_balance_max_weight",
                "5.0",
                "--full_bank_clean_hard_negative_weight",
                "0.35",
                "--full_bank_clean_reproj_radius",
                "3.0",
                "--full_bank_clean_hard_negatives",
                "12",
                "--clean_field_start_iter",
                "300",
                "--clean_field_full_bank_weight_scale",
                "0.25",
                "--clean_field_clean_hn_weight_scale",
                "8.0",
                "--clean_field_balance_weight",
                "0.9",
                "--clean_field_pose_information_weight",
                "0.8",
                "--clean_field_diff_pnp_weight_scale",
                "4.0",
            ]
        )

        self.assertEqual(args.full_bank_balance_weight, 0.5)
        self.assertEqual(args.full_bank_balance_grid_size, 4)
        self.assertEqual(args.full_bank_balance_depth_bins, 3)
        self.assertEqual(args.full_bank_balance_max_weight, 5.0)
        self.assertEqual(args.full_bank_clean_hard_negative_weight, 0.35)
        self.assertEqual(args.full_bank_clean_reproj_radius, 3.0)
        self.assertEqual(args.full_bank_clean_hard_negatives, 12)
        self.assertEqual(args.clean_field_start_iter, 300)
        self.assertEqual(args.clean_field_full_bank_weight_scale, 0.25)
        self.assertEqual(args.clean_field_clean_hn_weight_scale, 8.0)
        self.assertEqual(args.clean_field_balance_weight, 0.9)
        self.assertEqual(args.clean_field_pose_information_weight, 0.8)
        self.assertEqual(args.clean_field_diff_pnp_weight_scale, 4.0)

    def test_guarded_pnp_plan_can_override_localization_utility_scales(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_root = root / "data"
            baseline_root = root / "baseline"
            output_root = root / "lafgs"
            scene = "GreatCourt"
            (data_root / scene).mkdir(parents=True)
            model = baseline_root / f"{scene}_baseline"
            (model / "point_cloud" / "iteration_30000").mkdir(parents=True)
            (model / "point_cloud" / "iteration_30000" / "point_cloud.ply").write_text("ply\n")
            (model / "detector").mkdir(parents=True)
            (model / "detector" / "30000_detector.pth").write_text("detector")
            (model / "detector" / "sampled_idx.pkl").write_text("idx")

            plan = build_scene_plan(
                scene,
                data_root=data_root,
                baseline_root=baseline_root,
                output_root=output_root,
                python="python",
                baseline_iterations=30000,
                lafgs_steps=500,
                detect_num=8192,
                nms=2,
                reprojection_error=12.0,
                train_missing_baseline=False,
                force_train=False,
                skip_train=False,
                skip_eval=False,
                eval_baseline=True,
                cfg="configs/stdloc_cambridge.yaml",
                utility_pose_loss_scale=2.5,
                utility_reprojection_error_scale=8.0,
            )

            train_cmd = " ".join(plan.train_lafgs_command)
            self.assertIn("--lafgs_diff_pnp_utility_pose_loss_scale 2.5", train_cmd)
            self.assertIn("--lafgs_diff_pnp_utility_reprojection_error_scale 8.0", train_cmd)

    def test_guarded_pnp_plan_can_enable_direct_pose_grad_to_pnp_points(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_root = root / "data"
            baseline_root = root / "baseline"
            output_root = root / "lafgs"
            scene = "GreatCourt"
            (data_root / scene).mkdir(parents=True)
            model = baseline_root / f"{scene}_baseline"
            (model / "point_cloud" / "iteration_30000").mkdir(parents=True)
            (model / "point_cloud" / "iteration_30000" / "point_cloud.ply").write_text("ply\n")
            (model / "detector").mkdir(parents=True)
            (model / "detector" / "30000_detector.pth").write_text("detector")
            (model / "detector" / "sampled_idx.pkl").write_text("idx")

            default_plan = build_scene_plan(
                scene,
                data_root=data_root,
                baseline_root=baseline_root,
                output_root=output_root,
                python="python",
                baseline_iterations=30000,
                lafgs_steps=500,
                detect_num=8192,
                nms=2,
                reprojection_error=12.0,
                train_missing_baseline=False,
                force_train=False,
                skip_train=False,
                skip_eval=False,
                eval_baseline=True,
                cfg="configs/stdloc_cambridge.yaml",
            )
            direct_plan = build_scene_plan(
                scene,
                data_root=data_root,
                baseline_root=baseline_root,
                output_root=output_root,
                python="python",
                baseline_iterations=30000,
                lafgs_steps=500,
                detect_num=8192,
                nms=2,
                reprojection_error=12.0,
                train_missing_baseline=False,
                force_train=False,
                skip_train=False,
                skip_eval=False,
                eval_baseline=True,
                cfg="configs/stdloc_cambridge.yaml",
                direct_pnp_xyz_grad=True,
            )

            self.assertIn("--lafgs_diff_pnp_detach_pnp_points", default_plan.train_lafgs_command)
            self.assertNotIn("--lafgs_diff_pnp_detach_pnp_points", direct_plan.train_lafgs_command)

    def test_runner_can_build_pose_only_diff_pnp_command(self):
        from scripts.run_lafgs_cambridge_guarded_pnp import build_scene_plan

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_root = root / "data"
            baseline_root = root / "baseline"
            output_root = root / "lafgs"
            scene = "OldHospital"
            (data_root / scene).mkdir(parents=True)
            model = baseline_root / f"{scene}_baseline"
            (model / "point_cloud" / "iteration_30000").mkdir(parents=True)
            (model / "point_cloud" / "iteration_30000" / "point_cloud.ply").write_text("ply\n")
            (model / "detector").mkdir(parents=True)
            (model / "detector" / "30000_detector.pth").write_text("detector")
            (model / "detector" / "sampled_idx.pkl").write_text("idx")

            plan = build_scene_plan(
                scene,
                data_root=data_root,
                baseline_root=baseline_root,
                output_root=output_root,
                python="python",
                baseline_iterations=30000,
                lafgs_steps=500,
                detect_num=8192,
                nms=2,
                reprojection_error=12.0,
                train_missing_baseline=False,
                force_train=False,
                skip_train=False,
                skip_eval=False,
                eval_baseline=True,
                cfg="configs/stdloc_cambridge.yaml",
                allow_geometry_grad=False,
                geometry_reprojection_weight=0.0,
                geometry_xyz_lr=0.0,
            )

        command = plan.train_lafgs_command
        self.assertIn("--lafgs_diff_pnp_geometry_reproj_weight", command)
        self.assertEqual(command[command.index("--lafgs_diff_pnp_geometry_reproj_weight") + 1], "0.0")
        self.assertNotIn("--lafgs_diff_pnp_allow_geometry_grad", command)
        self.assertNotIn("--lafgs_diff_pnp_isolate_geometry_grad", command)
        self.assertIn("--lafgs_diff_pnp_geometry_xyz_lr", command)
        self.assertEqual(command[command.index("--lafgs_diff_pnp_geometry_xyz_lr") + 1], "0.0")

    def test_runner_requires_explicit_raw_xyz_geometry_grad_for_2dgs(self):
        from scripts.run_lafgs_cambridge_guarded_pnp import build_scene_plan

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_root = root / "data"
            baseline_root = root / "baseline"
            output_root = root / "lafgs"
            scene = "ShopFacade"
            (data_root / scene).mkdir(parents=True)
            model = baseline_root / f"{scene}_baseline"
            (model / "point_cloud" / "iteration_30000").mkdir(parents=True)
            (model / "point_cloud" / "iteration_30000" / "point_cloud.ply").write_text("ply\n")
            (model / "detector").mkdir(parents=True)
            (model / "detector" / "30000_detector.pth").write_text("detector")
            (model / "detector" / "sampled_idx.pkl").write_text("idx")

            common = dict(
                scene=scene,
                data_root=data_root,
                baseline_root=baseline_root,
                output_root=output_root,
                python="python",
                baseline_iterations=30000,
                lafgs_steps=500,
                detect_num=8192,
                nms=2,
                reprojection_error=12.0,
                train_missing_baseline=False,
                force_train=False,
                skip_train=False,
                skip_eval=False,
                eval_baseline=True,
                cfg="configs/stdloc_cambridge.yaml",
                gaussian_type="2dgs",
                geometry_xyz_lr=0.00002,
            )

            with self.assertRaisesRegex(ValueError, "allow_raw_xyz_geometry_grad"):
                build_scene_plan(**common)

            plan = build_scene_plan(**common, allow_raw_xyz_geometry_grad=True)

        command = plan.train_lafgs_command
        self.assertIn("--allow_raw_xyz_geometry_grad", command)
        self.assertEqual(command[command.index("--lafgs_diff_pnp_geometry_xyz_lr") + 1], "2e-05")

    def test_runner_can_build_sfm_from_zero_staged_lafgs_command(self):
        from scripts.run_lafgs_cambridge_guarded_pnp import build_scene_plan

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_root = root / "data"
            baseline_root = root / "baseline"
            output_root = root / "lafgs"
            cfg = root / "cfg.yaml"
            scene = "ShopFacade"
            (data_root / scene).mkdir(parents=True)
            cfg.write_text("cfg")

            plan = build_scene_plan(
                scene,
                data_root=data_root,
                baseline_root=baseline_root,
                output_root=output_root,
                python="python",
                baseline_iterations=30000,
                lafgs_steps=30000,
                detect_num=8192,
                nms=2,
                reprojection_error=12.0,
                train_missing_baseline=False,
                force_train=False,
                skip_train=False,
                skip_eval=False,
                eval_baseline=False,
                cfg=cfg,
                gaussian_type="2dgs",
                lafgs_from_sfm_zero=True,
                checkpoint_eval_interval=5000,
            )

        command = plan.train_lafgs_command
        command_text = " ".join(command)
        self.assertEqual(plan.status, "ready")
        self.assertEqual(plan.baseline_iteration, 0)
        self.assertEqual(plan.final_iteration, 30000)
        self.assertEqual(plan.checkpoint_iterations, [5000, 10000, 15000, 20000, 25000, 30000])
        self.assertNotIn("--load_iteration", command)
        self.assertIn("--iterations 30000", command_text)
        self.assertIn("--lafgs_stage_schedule sfm_from_zero", command_text)
        self.assertIn("--lafgs_rgb_densify", command_text)
        self.assertIn("--lafgs_rgb_densify_until_iter 15000", command_text)
        self.assertIn("--lafgs_rgb_densify_child_max_source_drift 2.0", command_text)
        self.assertIn("--lafgs_geometry_residual", command)
        self.assertIn("--lafgs_geometry_residual_weight 0.01", command_text)
        self.assertIn("--lafgs_geometry_grad_clip_abs 1.0", command_text)
        self.assertIn("--landmark_path __all__", command_text)
        self.assertIn("--allow_raw_xyz_geometry_grad", command)
        self.assertIn("--lafgs_diff_pnp_allow_geometry_grad", command)
        self.assertIn("--enable_topology", command)
        self.assertIn("--topology_enable_soft_prune", command)
        self.assertIn("--lafgs_diff_pnp_geometry_xyz_lr 2e-05", command_text)
        self.assertEqual(plan.train_baseline_detector_command, [])

    def test_runner_can_build_depth_anchored_geometry_command(self):
        from scripts.run_lafgs_cambridge_guarded_pnp import build_scene_plan

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scene = "ShopFacade"
            data_root = root / "data"
            baseline_root = root / "baseline"
            output_root = root / "out"
            cfg = root / "cfg.yaml"
            (data_root / scene).mkdir(parents=True)
            (baseline_root / f"{scene}_baseline" / "point_cloud" / "iteration_30000").mkdir(parents=True)
            (baseline_root / f"{scene}_baseline" / "detector").mkdir(parents=True)
            (baseline_root / f"{scene}_baseline" / "point_cloud" / "iteration_30000" / "point_cloud.ply").write_text("ply")
            (baseline_root / f"{scene}_baseline" / "detector" / "30000_detector.pth").write_text("det")
            (baseline_root / f"{scene}_baseline" / "detector" / "sampled_idx.pkl").write_text("idx")
            cfg.write_text("cfg")

            plan = build_scene_plan(
                scene,
                data_root=data_root,
                baseline_root=baseline_root,
                output_root=output_root,
                python="python",
                baseline_iterations=30000,
                lafgs_steps=500,
                detect_num=8192,
                nms=2,
                reprojection_error=12.0,
                train_missing_baseline=False,
                force_train=False,
                skip_train=False,
                skip_eval=False,
                eval_baseline=True,
                cfg=cfg,
                geometry_depth_anchor_weight=0.25,
            )

        command = plan.train_lafgs_command
        self.assertIn("--lafgs_diff_pnp_geometry_depth_anchor_weight", command)
        self.assertEqual(
            command[command.index("--lafgs_diff_pnp_geometry_depth_anchor_weight") + 1],
            "0.25",
        )

    def test_runner_can_build_strict_pose_improvement_guard_command(self):
        from scripts.run_lafgs_cambridge_guarded_pnp import build_scene_plan

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scene = "OldHospital"
            data_root = root / "data"
            baseline_root = root / "baseline"
            output_root = root / "out"
            cfg = root / "cfg.yaml"
            (data_root / scene).mkdir(parents=True)
            (baseline_root / f"{scene}_baseline" / "point_cloud" / "iteration_30000").mkdir(parents=True)
            (baseline_root / f"{scene}_baseline" / "detector").mkdir(parents=True)
            (baseline_root / f"{scene}_baseline" / "point_cloud" / "iteration_30000" / "point_cloud.ply").write_text("ply")
            (baseline_root / f"{scene}_baseline" / "detector" / "30000_detector.pth").write_text("det")
            (baseline_root / f"{scene}_baseline" / "detector" / "sampled_idx.pkl").write_text("idx")
            cfg.write_text("cfg")

            plan = build_scene_plan(
                scene,
                data_root=data_root,
                baseline_root=baseline_root,
                output_root=output_root,
                python="python",
                baseline_iterations=30000,
                lafgs_steps=500,
                detect_num=8192,
                nms=2,
                reprojection_error=12.0,
                train_missing_baseline=False,
                force_train=False,
                skip_train=False,
                skip_eval=False,
                eval_baseline=True,
                cfg=cfg,
                geometry_pose_guard_max_loss_increase=0.0,
                geometry_pose_guard_softness=0.0,
                geometry_pose_guard_min_scale=0.0,
                feedback_pose_guard_max_loss_increase=0.0,
                feedback_pose_guard_softness=0.0,
                feedback_pose_guard_min_scale=0.0,
            )

        command = plan.train_lafgs_command
        self.assertIn("--lafgs_diff_pnp_geometry_pose_guard_max_loss_increase", command)
        self.assertEqual(
            command[command.index("--lafgs_diff_pnp_geometry_pose_guard_max_loss_increase") + 1],
            "0.0",
        )
        self.assertIn("--lafgs_diff_pnp_feedback_pose_guard_max_loss_increase", command)
        self.assertEqual(
            command[command.index("--lafgs_diff_pnp_feedback_pose_guard_max_loss_increase") + 1],
            "0.0",
        )
        self.assertEqual(
            command[command.index("--lafgs_diff_pnp_geometry_pose_guard_softness") + 1],
            "0.0",
        )
        self.assertEqual(
            command[command.index("--lafgs_diff_pnp_feedback_pose_guard_min_scale") + 1],
            "0.0",
        )


class LafgsCambridgeSummaryTest(unittest.TestCase):
    def test_summary_reports_deltas_training_diagnostics_and_missing_scenes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            results = root / "results"
            baseline_root = root / "baseline"
            lafgs_root = root / "lafgs"

            for model_root, suffix in ((baseline_root, "_baseline"), (lafgs_root, "")):
                for scene in ("KingsCollege", "OldHospital"):
                    model = model_root / f"{scene}{suffix}"
                    (model / "point_cloud" / "iteration_30500").mkdir(parents=True)
                    if suffix == "_baseline":
                        (model / "point_cloud" / "iteration_30000").mkdir(parents=True)
                    else:
                        (model / "loc_training_summary.json").write_text(
                            json.dumps(
                                {
                                    "diff_pnp_episodes": 3,
                                    "diff_pnp_feedback_pose_guard_passed_total": 2,
                                    "diff_pnp_feedback_gt_reprojection_scale_total": 3,
                                    "diff_pnp_geometry_correspondences_total": 7,
                                    "diff_pnp_geometry_match_correspondences_total": 5,
                                    "diff_pnp_geometry_match_reprojection_weight_total": 1.5,
                                    "diff_pnp_geometry_match_peak_probability_threshold_total": 2.4,
                                    "diff_pnp_geometry_filter_keep_ratio_total": 1.25,
                                    "diff_pnp_geometry_kept_confidence_mean_total": 0.09,
                                    "diff_pnp_geometry_kept_reprojection_error_mean_total": 2.0,
                                    "geometry_xyz_isolated_grad_abs_max": 0.01,
                                    "diff_pnp_detach_pnp_points_total": 1.0,
                                    "diff_pnp_detach_pnp_points_max": 1.0,
                                    "diff_pnp_pose_loss_total": 12.5,
                                    "diff_pnp_pose_loss_max": 4.0,
                                    "diff_pnp_geometry_pose_guard_max_loss_increase_total": 0.0,
                                    "diff_pnp_feedback_pose_guard_max_loss_increase_total": 0.0,
                                    "geometry_xyz_full_grad_abs_max": 0.03,
                                    "geometry_xyz_step_delta_skipped_point_count_changed": 2,
                                    "geometry_source_aligned_delta_count": 120,
                                    "geometry_birth0_delta_count": 100,
                                    "geometry_child_delta_count": 20,
                                    "raw_xyz_delta_from_initial_all_sources_max": 1.5,
                                    "raw_xyz_child_delta_from_source_max": 1.4,
                                    "direct_diag_pose_information_weight_mean_total": 1.2,
                                    "direct_diag_pose_information_weight_mean_max": 0.7,
                                    "direct_diag_pose_information_weight_min_min": 0.25,
                                    "direct_diag_full_bank_clean_hard_negative_loss_total": 0.09,
                                    "direct_diag_full_bank_clean_hard_negative_weight_total": 0.7,
                                    "direct_diag_full_bank_balance_weight_mean_total": 1.1,
                                    "direct_diag_full_bank_balance_weight_min_min": 0.2,
                                    "direct_diag_full_bank_balance_weight_max_max": 2.3,
                                    "direct_diag_surfel_loc_anchor_reg_loss_total": 0.004,
                                    "direct_diag_surfel_loc_tangent_bound_max": 0.2,
                                    "direct_diag_surfel_loc_normal_bound_max": 0.05,
                                }
                            )
                        )

            def write_result(parent, model_path, te, ae, recall, inliers):
                parent.mkdir(parents=True)
                (parent / "summary.json").write_text(
                    json.dumps(
                        {
                            "model_path": str(model_path),
                            "sparse": {
                                "median_te": te,
                                "median_ae": ae,
                                "recall_5cm_5d": recall,
                                "recall_2cm_2d": recall / 2,
                                "avg_inliers": inliers,
                            },
                        }
                    )
                )

            write_result(
                results / "baseline-30000-kings",
                baseline_root / "KingsCollege_baseline",
                20.0,
                0.5,
                0.1,
                100,
            )
            write_result(
                results / "lafgs-guarded-pnp-kings",
                lafgs_root / "KingsCollege",
                15.0,
                0.4,
                0.2,
                120,
            )
            write_result(
                results / "baseline-30000-old",
                baseline_root / "OldHospital_baseline",
                18.0,
                0.3,
                0.05,
                90,
            )
            write_result(
                results / "lafgs-guarded-pnp-old",
                lafgs_root / "OldHospital",
                19.0,
                0.35,
                0.04,
                80,
            )

            summary = summarize_cambridge(
                scenes=["KingsCollege", "OldHospital", "GreatCourt"],
                baseline_root=baseline_root,
                lafgs_root=lafgs_root,
                results_root=results,
                baseline_prefix="baseline-30000",
                lafgs_prefix="lafgs-guarded-pnp",
                final_iteration=30500,
            )

            self.assertEqual(summary["aggregate"]["scene_count"], 3)
            self.assertEqual(summary["aggregate"]["completed_scene_count"], 2)
            self.assertEqual(summary["aggregate"]["improved_te_scene_count"], 1)
            self.assertEqual(summary["aggregate"]["missing_result_scenes"], ["GreatCourt"])
            kings = next(row for row in summary["scenes"] if row["scene"] == "KingsCollege")
            self.assertEqual(kings["delta_median_te_cm"], -5.0)
            self.assertEqual(kings["diff_pnp_episodes"], 3)
            self.assertEqual(kings["diff_pnp_feedback_pose_guard_passed_total"], 2)
            self.assertEqual(kings["diff_pnp_geometry_match_correspondences_total"], 5)
            self.assertEqual(kings["diff_pnp_geometry_match_reprojection_weight_total"], 1.5)
            self.assertEqual(kings["diff_pnp_geometry_match_peak_probability_threshold_total"], 2.4)
            self.assertEqual(kings["diff_pnp_geometry_filter_keep_ratio_total"], 1.25)
            self.assertEqual(kings["diff_pnp_geometry_kept_confidence_mean_total"], 0.09)
            self.assertEqual(kings["diff_pnp_geometry_kept_reprojection_error_mean_total"], 2.0)
            self.assertEqual(kings["diff_pnp_detach_pnp_points_total"], 1.0)
            self.assertEqual(kings["diff_pnp_detach_pnp_points_max"], 1.0)
            self.assertEqual(kings["diff_pnp_pose_loss_total"], 12.5)
            self.assertEqual(kings["diff_pnp_pose_loss_max"], 4.0)
            self.assertEqual(kings["diff_pnp_geometry_pose_guard_max_loss_increase_total"], 0.0)
            self.assertEqual(kings["diff_pnp_feedback_pose_guard_max_loss_increase_total"], 0.0)
            self.assertEqual(kings["geometry_xyz_full_grad_abs_max"], 0.03)
            self.assertEqual(kings["geometry_xyz_step_delta_skipped_point_count_changed"], 2)
            self.assertEqual(kings["geometry_source_aligned_delta_count"], 120)
            self.assertEqual(kings["geometry_birth0_delta_count"], 100)
            self.assertEqual(kings["geometry_child_delta_count"], 20)
            self.assertEqual(kings["raw_xyz_delta_from_initial_all_sources_max"], 1.5)
            self.assertEqual(kings["raw_xyz_child_delta_from_source_max"], 1.4)
            self.assertEqual(kings["direct_diag_pose_information_weight_mean_total"], 1.2)
            self.assertEqual(kings["direct_diag_pose_information_weight_mean_max"], 0.7)
            self.assertEqual(kings["direct_diag_pose_information_weight_min_min"], 0.25)
            self.assertEqual(kings["direct_diag_full_bank_clean_hard_negative_loss_total"], 0.09)
            self.assertEqual(kings["direct_diag_full_bank_clean_hard_negative_weight_total"], 0.7)
            self.assertEqual(kings["direct_diag_full_bank_balance_weight_mean_total"], 1.1)
            self.assertEqual(kings["direct_diag_full_bank_balance_weight_min_min"], 0.2)
            self.assertEqual(kings["direct_diag_full_bank_balance_weight_max_max"], 2.3)
            self.assertEqual(kings["direct_diag_surfel_loc_anchor_reg_loss_total"], 0.004)
            self.assertEqual(kings["direct_diag_surfel_loc_tangent_bound_max"], 0.2)
            self.assertEqual(kings["direct_diag_surfel_loc_normal_bound_max"], 0.05)
            old = next(row for row in summary["scenes"] if row["scene"] == "OldHospital")
            self.assertEqual(old["delta_median_te_cm"], 1.0)


if __name__ == "__main__":
    unittest.main()
