import tempfile
import unittest
from pathlib import Path

from scripts.audit_cambridge_matcha_2dgs_protocol import (
    ensure_symlink,
    markdown_report,
    normalize_image_name,
    prepare_wrapper,
    read_ply_header,
    sha256_file,
)


class CambridgeMatcha2DGSProtocolTests(unittest.TestCase):
    def test_normalize_image_name_preserves_cambridge_sequence(self):
        self.assertEqual(normalize_image_name("seq2/frame00001.png"), "seq2/frame00001.png")
        self.assertEqual(normalize_image_name("seq2__frame00001"), "seq2/frame00001.png")
        self.assertEqual(
            normalize_image_name("/tmp/images/seq2__frame00001.png"),
            "seq2/frame00001.png",
        )

    def test_read_native_2dgs_ply_header(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "point_cloud.ply"
            path.write_bytes(
                b"ply\nformat binary_little_endian 1.0\n"
                b"element vertex 7\nproperty float x\nproperty float scale_0\n"
                b"property float scale_1\nproperty float mip_filter\nend_header\n"
            )
            count, properties = read_ply_header(path)
        self.assertEqual(count, 7)
        self.assertEqual(properties, ["x", "scale_0", "scale_1", "mip_filter"])

    def test_wrapper_links_only_source_iteration(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source_iteration = source / "point_cloud" / "iteration_30000"
            source_iteration.mkdir(parents=True)
            source_ply = source_iteration / "point_cloud.ply"
            source_ply.write_bytes(b"ply")
            (source / "input.ply").write_bytes(b"input")
            wrapper_root = root / "wrappers"
            summary = {
                "model_path": str(source),
                "point_cloud": str(source_ply),
                "point_cloud_sha256": sha256_file(source_ply),
                "rgb_geometry_only": True,
                "protocol_role": "external_rgb_2dgs_fixed_geometry_baseline",
                "selected_camera_count": 20,
                "full_training_camera_count": 231,
                "uses_full_cambridge_training_split": False,
                "strict_from_sfm_iteration0_equivalent": False,
            }
            wrapper = Path(prepare_wrapper("ShopFacade", summary, wrapper_root, root / "data"))
            point_cloud_root = wrapper / "point_cloud"
            self.assertFalse(point_cloud_root.is_symlink())
            self.assertTrue((point_cloud_root / "iteration_30000").is_symlink())
            self.assertEqual((point_cloud_root / "iteration_30000").resolve(), source_iteration)
            self.assertIn("gaussian_type='2dgs'", (wrapper / "cfg_args").read_text())

    def test_broken_wrapper_symlink_can_move_to_persistent_target(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            transient = root / "transient"
            transient.mkdir()
            persistent = root / "persistent"
            persistent.mkdir()
            link = root / "artifact"
            link.symlink_to(transient, target_is_directory=True)
            transient.rmdir()

            ensure_symlink(link, persistent)

            self.assertTrue(link.is_symlink())
            self.assertEqual(link.resolve(), persistent)

    def test_strict_runner_pins_historical_r2_learning_rates(self):
        runner = Path(__file__).parents[1] / "scripts" / "run_cambridge_strict_2dgs_scene.sh"
        text = runner.read_text()

        self.assertIn('R2_FOLDER="detector_strict2dgs_R2_flr2e4_2000"', text)
        self.assertIn("--candidate_teacher_feature_lr 0.0002", text)
        self.assertIn("--candidate_teacher_detector_lr 0.0001", text)
        self.assertIn("--candidate_teacher_hard_negatives 8", text)
        self.assertIn("--candidate_teacher_assignment_temperature 0.05", text)
        self.assertIn("--candidate_teacher_assignment_margin 0.05", text)
        self.assertIn("--train_seed 0", text)
        self.assertIn("--save_iterations 5000 10000 20000 30000", text)
        self.assertIn("--test_iterations 5000 10000 20000 30000", text)

    def test_report_distinguishes_fixed_geometry_from_strict_from_sfm(self):
        report = markdown_report(
            {
                "scenes": {
                    "ShopFacade": {
                        "vertex_count": 100,
                        "is_native_2dgs_schema": True,
                        "loc_feature_count": 0,
                        "selected_camera_count": 20,
                        "full_training_camera_count": 231,
                        "uses_full_cambridge_training_split": False,
                        "selected_position_error_m": {"max": 0.0},
                        "selected_rotation_error_deg": {"max": 0.0},
                        "all_output_position_error_m": {"max": 0.0},
                    }
                }
            }
        )

        self.assertIn("20/231", report)
        self.assertIn("valid fixed-geometry native-2DGS baselines", report)
        self.assertIn("not strict replacements", report)


if __name__ == "__main__":
    unittest.main()
