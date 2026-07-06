import json
import tempfile
import unittest
from pathlib import Path

from scripts.run_lafgs_cambridge_guarded_pnp import (
    DEFAULT_SCENES,
    build_scene_plan,
)
from scripts.summarize_lafgs_cambridge import summarize_cambridge


class LafgsCambridgeExperimentPlanTest(unittest.TestCase):
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
                self.assertEqual(plan.detector_source, baseline_root / f"{plan.scene}_baseline" / "detector")
                self.assertEqual(plan.final_iteration, 30500)

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

    def test_guarded_pnp_plan_keeps_pose_information_weight_off_by_default(self):
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
            self.assertIn("--loc_full_bank_pose_information_weight 0.0", train_cmd)
            self.assertIn("--loc_full_bank_pose_information_floor 0.0", train_cmd)

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
                                    "direct_diag_pose_information_weight_mean_total": 1.2,
                                    "direct_diag_pose_information_weight_mean_max": 0.7,
                                    "direct_diag_pose_information_weight_min_min": 0.25,
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
            self.assertEqual(kings["direct_diag_pose_information_weight_mean_total"], 1.2)
            self.assertEqual(kings["direct_diag_pose_information_weight_mean_max"], 0.7)
            self.assertEqual(kings["direct_diag_pose_information_weight_min_min"], 0.25)
            self.assertEqual(kings["direct_diag_surfel_loc_anchor_reg_loss_total"], 0.004)
            self.assertEqual(kings["direct_diag_surfel_loc_tangent_bound_max"], 0.2)
            self.assertEqual(kings["direct_diag_surfel_loc_normal_bound_max"], 0.05)
            old = next(row for row in summary["scenes"] if row["scene"] == "OldHospital")
            self.assertEqual(old["delta_median_te_cm"], 1.0)


if __name__ == "__main__":
    unittest.main()
