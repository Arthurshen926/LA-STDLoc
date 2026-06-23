import re
import unittest
import importlib.util
from argparse import Namespace
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_locaware_cambridge_full.sh"
V02_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_locaware_v02_shopfacade.sh"
V03_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_locaware_v03_shopfacade.sh"
V03_MULTISCENE_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_locaware_v03_multiscene.sh"
V03_TOPOLOGY_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_locaware_v03_topology_full.sh"
DENSE_KL_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_densekl_v03_cambridge.sh"
PREPARE_BASELINES_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "prepare_cambridge_baseline_artifacts.sh"
GEOM_2X2_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_geometry_balance_2x2_shopfacade.sh"
DESCRIPTOR_DIAG_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "diagnose_sparse_descriptors.py"
INLIER_DIAG_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "diagnose_sparse_inliers.py"
REMAP_TOPOLOGY_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "remap_topology_landmarks.py"
DENSE_RESP_DIAG_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "diagnose_dense_responsibility.py"


class FullRunScriptArgsTest(unittest.TestCase):
    def _command_blocks(self, command_name):
        text = SCRIPT.read_text()
        pattern = re.compile(rf'"\$PYTHON" {re.escape(command_name)} \\\n(?P<body>.*?)(?=\n"\$PYTHON"|\nif |\nelse|\nfi|\Z)', re.S)
        return [match.group("body") for match in pattern.finditer(text)]

    def _load_descriptor_diag_module(self):
        spec = importlib.util.spec_from_file_location("descriptor_diag_script", DESCRIPTOR_DIAG_SCRIPT)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def _load_inlier_diag_module(self):
        spec = importlib.util.spec_from_file_location("inlier_diag_script", INLIER_DIAG_SCRIPT)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def _load_dense_resp_diag_module(self):
        spec = importlib.util.spec_from_file_location("dense_resp_diag_script", DENSE_RESP_DIAG_SCRIPT)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_eval_commands_do_not_receive_training_only_args(self):
        for command_name in ("stdloc.py", "cache_sparse_poses.py"):
            with self.subTest(command=command_name):
                blocks = self._command_blocks(command_name)
                self.assertTrue(blocks, f"{command_name} is not invoked by the full run script")
                for block in blocks:
                    self.assertIn('"${DATA_ARGS[@]}"', block)
                    self.assertNotIn('"${TRAIN_ARGS[@]}"', block)
                    self.assertNotIn('"${COMMON_ARGS[@]}"', block)

    def test_la_training_phases_are_resume_safe(self):
        text = SCRIPT.read_text()
        for phase_end in ("FEATURE_END", "GEOMETRY_END", "TOPOLOGY_END", "CLOSED_LOOP_END"):
            with self.subTest(phase_end=phase_end):
                self.assertIn(f'if ! point_cloud_exists "$LA_MODEL" "${phase_end}"; then', text)

    def test_v02_script_keeps_sparse_pipeline_fixed(self):
        text = V02_SCRIPT.read_text()
        self.assertIn("--loc_teacher direct", text)
        self.assertIn("--no-use_loc_opacity", text)
        self.assertIn("--loc_multiview_weight 0.05", text)
        self.assertIn("--support_query_split", text)
        self.assertIn("--loc_proto_weight 0.0", text)
        self.assertIn("--loc_rank_weight 0.0", text)
        self.assertIn("--direct_depth_check", text)
        self.assertIn('sparse["detector_model_path"] = baseline_model', text)
        self.assertIn('sparse["landmark_model_path"] = baseline_model', text)
        self.assertIn("--prefix \"phase-v02-${checkpoint}\"", text)

    def test_v02_script_does_not_default_to_invalid_e3_fixed_baseline_indices(self):
        text = V02_SCRIPT.read_text()
        self.assertIn("RUN_E3=${RUN_E3:-0}", text)
        self.assertNotIn("phase-e3-40k-fixed-baseline-sparse", text)
        self.assertIn("phase-e3-40k-baseline-hard-sparse", text)

    def test_v03_script_runs_feature_only_full_bank_anchor_without_topology(self):
        text = V03_SCRIPT.read_text()
        self.assertIn("--loc_teacher direct", text)
        self.assertIn("--loc_full_bank_weight", text)
        self.assertIn("--loc_full_bank_hard_negatives", text)
        self.assertIn("--loc_anchor_weight", text)
        self.assertIn("--loc_opacity_weight 0.0", text)
        self.assertIn("--no-use_loc_opacity", text)
        self.assertIn("--support_query_split", text)
        self.assertNotIn("--enable_topology", text)
        self.assertIn('sparse["detector_model_path"] = baseline_model', text)
        self.assertIn('sparse["landmark_model_path"] = baseline_model', text)

    def test_v03_script_accepts_query_split_seed_for_multiseed_runs(self):
        text = V03_SCRIPT.read_text()
        self.assertIn("V03_QUERY_SPLIT_SEED=${V03_QUERY_SPLIT_SEED:-2025}", text)
        self.assertIn("V03_TRAIN_SEED=${V03_TRAIN_SEED:-0}", text)
        self.assertIn("V03_QUERY_SPLIT_MODE=${V03_QUERY_SPLIT_MODE:-random}", text)
        self.assertIn('--train_seed "$V03_TRAIN_SEED"', text)
        self.assertIn('--query_split_seed "$V03_QUERY_SPLIT_SEED"', text)
        self.assertIn('--query_split_mode "$V03_QUERY_SPLIT_MODE"', text)
        self.assertNotIn("--query_split_seed 2025", text)

    def test_v03_multiscene_script_dispatches_scene_seed_matrix_safely(self):
        text = V03_MULTISCENE_SCRIPT.read_text()
        self.assertIn("SCENES=${SCENES:-ShopFacade KingsCollege OldHospital}", text)
        self.assertIn("TRAIN_SEEDS=${TRAIN_SEEDS:-0 1 2}", text)
        self.assertIn("QUERY_SPLIT_SEEDS=${QUERY_SPLIT_SEEDS:-2025 2026 2027}", text)
        self.assertIn('BASELINE_MODEL="$BASELINE_ROOT/${scene}_baseline"', text)
        self.assertIn('if [[ ! -d "$BASELINE_MODEL" ]]; then', text)
        self.assertIn('if [[ ! -f "$BASELINE_MODEL/detector/30000_detector.pth" || ! -f "$BASELINE_MODEL/detector/sampled_idx.pkl" ]]; then', text)
        self.assertIn("run_locaware_v03_shopfacade.sh", text)
        self.assertIn('V03_TRAIN_SEED="$train_seed"', text)
        self.assertIn('V03_QUERY_SPLIT_SEED="$query_split_seed"', text)
        self.assertIn('MODEL_ROOT="$MODEL_ROOT/${scene}/train_seed_${train_seed}/query_split_${query_split_seed}"', text)

    def test_train_locaware_seeds_after_safe_state_from_cli_arg(self):
        text = (Path(__file__).resolve().parents[1] / "train_locaware.py").read_text()

        self.assertIn("--train_seed", text)
        self.assertNotIn("seed_everything(2025)", text)
        self.assertLess(text.index("safe_state(args.quiet)"), text.index("seed_everything(args.train_seed)"))

    def test_v03_topology_script_matches_v03_direct_objective_by_default(self):
        self.assertTrue(V03_TOPOLOGY_SCRIPT.exists(), "v0.3 topology rerun script is missing")
        text = V03_TOPOLOGY_SCRIPT.read_text()
        self.assertIn("diagnose_sparse_inliers.py", text)
        self.assertIn("--label_state_output", text)
        self.assertIn("--label_state_reset", text)
        self.assertIn("--localization_state_path", text)
        self.assertIn("TRAIN_PHASE=${TRAIN_PHASE:-feature}", text)
        self.assertIn("LOC_TEACHER=${LOC_TEACHER:-direct}", text)
        self.assertIn('TOPOLOGY_MUTATION_MODE=${TOPOLOGY_MUTATION_MODE:-split_only}', text)
        self.assertIn("TOPOLOGY_USE_LOC_OPACITY=${TOPOLOGY_USE_LOC_OPACITY:-0}", text)
        self.assertIn("TOPOLOGY_PROTECT_LANDMARKS=${TOPOLOGY_PROTECT_LANDMARKS:-0}", text)
        self.assertIn("TOPOLOGY_REMAP_MODE=${TOPOLOGY_REMAP_MODE:-source_distance}", text)
        self.assertIn("TOPOLOGY_DENSE_DESC_WEIGHT=${TOPOLOGY_DENSE_DESC_WEIGHT:-0.0}", text)
        self.assertIn("TOPOLOGY_DENSE_REPROJ_WEIGHT=${TOPOLOGY_DENSE_REPROJ_WEIGHT:-0.0}", text)
        self.assertIn('--train_phase "$TRAIN_PHASE"', text)
        self.assertIn('--loc_teacher "$LOC_TEACHER"', text)
        self.assertIn("--loc_direct_weight", text)
        self.assertIn("--loc_multiview_weight", text)
        self.assertIn("--loc_full_bank_weight", text)
        self.assertIn("--loc_anchor_weight", text)
        self.assertIn("--direct_depth_check", text)
        self.assertIn("LOC_OPACITY_ARGS=(--no-use_loc_opacity --loc_opacity_weight 0.0)", text)
        self.assertIn('"${LOC_OPACITY_ARGS[@]}"', text)
        self.assertIn('--loc_desc_weight "$TOPOLOGY_DENSE_DESC_WEIGHT"', text)
        self.assertIn('--loc_reproj_weight "$TOPOLOGY_DENSE_REPROJ_WEIGHT"', text)
        self.assertIn("--enable_topology", text)
        self.assertIn('case "$TOPOLOGY_MUTATION_MODE" in', text)
        self.assertIn("no_mutation)", text)
        self.assertIn("split_only)", text)
        self.assertIn("soft_prune)", text)
        self.assertIn("soft_prune_only)", text)
        self.assertIn("physical_prune)", text)
        self.assertIn("physical_prune_only)", text)
        self.assertIn("TOPOLOGY_DISABLE_SPLIT=${TOPOLOGY_DISABLE_SPLIT:-0}", text)
        self.assertIn("--topology_disable_split", text)
        self.assertIn("TOPOLOGY_DENSE_DESC_WEIGHT=1.0", text)
        self.assertIn("TOPOLOGY_DENSE_REPROJ_WEIGHT=0.1", text)
        self.assertIn('TOPOLOGY_ARGS+=(--topology_enable_physical_prune)', text)
        self.assertIn("--topology_allow_untrained_loc_opacity_prune", text)
        self.assertIn("remap_topology_landmarks.py", text)
        self.assertIn("--topology_loc_state", text)
        self.assertIn("--output_sampled_idx", text)
        self.assertIn("--remap_mode", text)
        self.assertIn("stdloc.py", text)
        self.assertIn('sparse["landmark_path"] = topology_landmark_path', text)
        self.assertIn('sparse["landmark_model_path"] = topology_model', text)
        self.assertIn("--prefix \"phase-v03-topology-${TOPOLOGY_END}\"", text)

    def test_dense_kl_script_runs_dense_teacher_without_topology(self):
        text = DENSE_KL_SCRIPT.read_text()
        self.assertIn("--loc_teacher dense", text)
        self.assertIn("--loc_dense_kl_weight", text)
        self.assertIn("--loc_responsibility_topk", text)
        self.assertIn("--loc_responsibility_opacity_weight", text)
        self.assertIn("--loc_responsibility_depth_weight", text)
        self.assertIn("--no-use_loc_opacity", text)
        self.assertNotIn("--enable_topology", text)
        self.assertIn("diagnose_dense_responsibility.py", text)
        self.assertIn('sparse["detector_model_path"] = baseline_model', text)
        self.assertIn('sparse["landmark_model_path"] = baseline_model', text)
        self.assertIn("--sparse_only", text)

    def test_prepare_baseline_artifacts_script_builds_missing_scene_baselines(self):
        text = PREPARE_BASELINES_SCRIPT.read_text()
        self.assertIn("SOURCE_ROOT=${SOURCE_ROOT:-/mnt/pool/sqy/ulfloc_repro_20260607}", text)
        self.assertIn("TARGET_ROOT=${TARGET_ROOT:-/mnt/pool/sqy/stdloc_la_full_runs}", text)
        self.assertIn("SCENES=${SCENES:-ShopFacade KingsCollege OldHospital}", text)
        self.assertIn("SKIP_DETECTOR=${SKIP_DETECTOR:-0}", text)
        self.assertIn("REQUIRE_LOC_FEATURE=${REQUIRE_LOC_FEATURE:-1}", text)
        self.assertIn("TRAIN_MISSING_BASELINE=${TRAIN_MISSING_BASELINE:-0}", text)
        self.assertIn("FORCE_BASELINE_TRAIN=${FORCE_BASELINE_TRAIN:-0}", text)
        self.assertIn("ply_has_loc_feature()", text)
        self.assertIn("train_full_baseline()", text)
        self.assertIn('SOURCE_MODEL="$SOURCE_ROOT/$scene"', text)
        self.assertIn('TARGET_MODEL="$TARGET_ROOT/${scene}_baseline"', text)
        self.assertIn('SOURCE_PLY="$SOURCE_MODEL/point_cloud/iteration_${BASELINE_ITERS}/point_cloud.ply"', text)
        self.assertIn('cp -a "$SOURCE_MODEL" "$TARGET_MODEL"', text)
        self.assertIn('"$PYTHON" train.py', text)
        self.assertIn("--train_detector", text)
        self.assertIn("--test_detector_iterations", text)
        self.assertIn("--save_detector_iterations", text)
        self.assertIn('if [[ "$REQUIRE_LOC_FEATURE" == "1" ]] && ! ply_has_loc_feature "$SOURCE_PLY"; then', text)
        self.assertIn('if [[ "$TRAIN_MISSING_BASELINE" == "1" ]]; then', text)
        self.assertIn("source point cloud lacks loc_* feature fields", text)
        self.assertIn('if [[ "$SKIP_DETECTOR" == "1" ]]; then', text)
        self.assertIn("train_detector.py", text)
        self.assertIn("--sampling_mode baseline", text)
        self.assertIn("--detector_target_mode hard", text)
        self.assertIn("--detector_folder detector", text)
        self.assertIn('TARGET_PLY="$TARGET_MODEL/point_cloud/iteration_${BASELINE_ITERS}/point_cloud.ply"', text)
        detector_ready_pos = text.index('if [[ -f "$TARGET_MODEL/detector/${DETECTOR_ITERS}_detector.pth" && -f "$TARGET_MODEL/detector/sampled_idx.pkl" ]]; then')
        source_loc_check_pos = text.index('if [[ "$REQUIRE_LOC_FEATURE" == "1" ]] && ! ply_has_loc_feature "$SOURCE_PLY"; then')
        self.assertLess(detector_ready_pos, source_loc_check_pos)
        self.assertIn('if [[ ! -f "$TARGET_PLY" ]]; then', text)

    def test_geometry_balance_2x2_script_runs_original_and_balanced_selectors(self):
        text = GEOM_2X2_SCRIPT.read_text()
        self.assertIn('geometry_balance"', text)
        self.assertIn('"enabled": False', text)
        self.assertIn('"enabled": True', text)
        self.assertIn('"post": {', text)
        self.assertIn('"score_weight": float(post_score_weight)', text)
        self.assertIn("phase-2x2-baseline-original", text)
        self.assertIn("phase-2x2-baseline-balanced", text)
        self.assertIn("phase-2x2-la-original", text)
        self.assertIn("phase-2x2-la-balanced", text)
        self.assertIn('sparse["detector_model_path"] = baseline_model', text)
        self.assertIn('sparse["landmark_model_path"] = baseline_model', text)

    def test_train_locaware_applies_topology_before_saving_same_iteration_checkpoint(self):
        text = (Path(__file__).resolve().parents[1] / "train_locaware.py").read_text()

        topology_pos = text.index("topology_controller.update(gaussians, scene.cameras_extent, iteration)")
        save_pos = text.index("scene.save(iteration)")

        self.assertLess(topology_pos, save_pos)

    def test_descriptor_diagnostics_script_reports_level1_metrics(self):
        text = DESCRIPTOR_DIAG_SCRIPT.read_text()
        self.assertIn("descriptor_alignment_metrics", text)
        self.assertIn("--baseline_model_path", text)
        self.assertIn("--max_images", text)
        self.assertIn("positive_cosine_mean", text)
        self.assertIn("mnn_precision", text)

    def test_descriptor_diagnostics_limits_iterable_cameras(self):
        module = self._load_descriptor_diag_module()
        cameras = (idx for idx in range(4))
        self.assertEqual(module._limit_cameras(cameras, 2), [0, 1])

    def test_descriptor_diagnostics_tolerates_missing_optional_overrides(self):
        module = self._load_descriptor_diag_module()

        updated = module._ensure_optional_args(Namespace())

        self.assertIsNone(updated.landmark_model_path)
        self.assertIsNone(updated.baseline_model_path)
        self.assertEqual(updated.baseline_iteration, 30000)

    def test_sparse_inlier_diagnostics_script_reports_level3_metrics(self):
        text = INLIER_DIAG_SCRIPT.read_text()
        self.assertIn("summarize_landmark_value", text)
        self.assertIn("--reprojection_error", text)
        self.assertIn("spearman_utility_inlier_rate", text)
        self.assertIn("visible_count", text)
        self.assertIn("inlier_count", text)
        self.assertIn("--label_state_output", text)
        self.assertIn("--label_state_reset", text)
        self.assertIn("add_sparse_match_label_stats", text)
        self.assertIn("calibrate_landmark_quality", text)
        self.assertIn("--calibrated_utility_output", text)
        self.assertIn("--calibration_target", text)
        self.assertIn("--calibration_split_mode", text)
        self.assertIn("--calibration_eval_fraction", text)
        self.assertIn("spearman_calibrated_correct_rate", text)
        self.assertIn("calibration_heldout", text)

    def test_sparse_inlier_diagnostics_limits_iterable_cameras(self):
        module = self._load_inlier_diag_module()
        cameras = (idx for idx in range(5))
        self.assertEqual(module._limit_cameras(cameras, 3), [0, 1, 2])

    def test_sparse_inlier_diagnostics_tolerates_missing_optional_overrides(self):
        module = self._load_inlier_diag_module()
        config = {"sparse": {}}

        updated = module._apply_sparse_overrides(config, Namespace())

        self.assertTrue(updated["sparse"]["sparse_only"])

    def test_sparse_inlier_diagnostics_builds_block_calibration_holdout(self):
        module = self._load_inlier_diag_module()

        roles = module._calibration_image_roles(5, mode="block", eval_fraction=0.4)

        self.assertEqual(roles, ["train", "train", "train", "eval", "eval"])

    def test_sparse_inlier_diagnostics_uses_target_specific_calibration_denominator(self):
        module = self._load_inlier_diag_module()
        visible = module.torch.tensor([10, 8])
        matched = module.torch.tensor([4, 5])
        correct = module.torch.tensor([2, 4])
        inlier = module.torch.tensor([1, 3])

        correct_positive, correct_trial, correct_mask = module._calibration_target_counts(
            "correct",
            visible,
            matched,
            correct,
            inlier,
        )
        inlier_positive, inlier_trial, inlier_mask = module._calibration_target_counts(
            "inlier",
            visible,
            matched,
            correct,
            inlier,
        )

        self.assertTrue(module.torch.equal(correct_positive, correct))
        self.assertTrue(module.torch.equal(correct_trial, visible))
        self.assertTrue(module.torch.equal(correct_mask, visible > 0))
        self.assertTrue(module.torch.equal(inlier_positive, inlier))
        self.assertTrue(module.torch.equal(inlier_trial, matched))
        self.assertTrue(module.torch.equal(inlier_mask, matched > 0))

    def test_remap_topology_landmarks_script_uses_source_index_mapping(self):
        text = REMAP_TOPOLOGY_SCRIPT.read_text()
        self.assertIn("remap_sampled_indices_from_source_index", text)
        self.assertIn("--source_sampled_idx", text)
        self.assertIn("--topology_loc_state", text)
        self.assertIn("--output_sampled_idx", text)
        self.assertIn("--remap_score_source", text)
        self.assertIn("--remap_mode", text)
        self.assertIn("--max_source_distance", text)
        self.assertIn("loc_source_xyz", text)
        self.assertIn("loc_current_xyz", text)
        self.assertIn("missing_count", text)

    def test_dense_responsibility_diagnostics_script_reports_reconstruction_metrics(self):
        text = DENSE_RESP_DIAG_SCRIPT.read_text()
        self.assertIn("dense_localization_teacher", text)
        self.assertIn("responsibility_reconstruction_mean_cosine", text)
        self.assertIn("responsibility_reconstruction_p10_cosine", text)
        self.assertIn("--responsibility_topk", text)
        self.assertIn("--responsibility_opacity_weight", text)
        self.assertIn("--responsibility_depth_weight", text)
        self.assertIn("--anchor_count", text)
        self.assertIn("--output", text)

    def test_dense_responsibility_diagnostics_summarizes_images_by_valid_anchor_count(self):
        module = self._load_dense_resp_diag_module()
        images = [
            {
                "anchor_count": 128,
                "responsibility_reconstruction_mean_cosine": 0.5,
                "responsibility_reconstruction_p10_cosine": 0.2,
                "responsibility_reconstruction_min_cosine": -0.1,
                "responsibility_reconstruction_valid_anchor_count": 10,
            },
            {
                "anchor_count": 64,
                "responsibility_reconstruction_mean_cosine": 0.8,
                "responsibility_reconstruction_p10_cosine": 0.6,
                "responsibility_reconstruction_min_cosine": 0.1,
                "responsibility_reconstruction_valid_anchor_count": 30,
            },
        ]

        summary = module._summarize_images(images)

        self.assertEqual(summary["image_count"], 2)
        self.assertEqual(summary["total_valid_anchor_count"], 40)
        self.assertAlmostEqual(summary["mean_anchor_count"], 96.0)
        self.assertAlmostEqual(summary["mean_responsibility_reconstruction_mean_cosine"], 0.725)
        self.assertAlmostEqual(summary["mean_responsibility_reconstruction_p10_cosine"], 0.5)
        self.assertAlmostEqual(summary["min_responsibility_reconstruction_min_cosine"], -0.1)


if __name__ == "__main__":
    unittest.main()
