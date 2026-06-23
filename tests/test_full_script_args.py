import re
import unittest
import importlib.util
from argparse import Namespace
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_locaware_cambridge_full.sh"
V02_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_locaware_v02_shopfacade.sh"
DESCRIPTOR_DIAG_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "diagnose_sparse_descriptors.py"
INLIER_DIAG_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "diagnose_sparse_inliers.py"


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

    def test_sparse_inlier_diagnostics_limits_iterable_cameras(self):
        module = self._load_inlier_diag_module()
        cameras = (idx for idx in range(5))
        self.assertEqual(module._limit_cameras(cameras, 3), [0, 1, 2])

    def test_sparse_inlier_diagnostics_tolerates_missing_optional_overrides(self):
        module = self._load_inlier_diag_module()
        config = {"sparse": {}}

        updated = module._apply_sparse_overrides(config, Namespace())

        self.assertTrue(updated["sparse"]["sparse_only"])


if __name__ == "__main__":
    unittest.main()
