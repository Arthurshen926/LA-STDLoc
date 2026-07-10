import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image


class ValidSupportMaskAuditScriptsTest(unittest.TestCase):
    def test_image_audit_writes_summary_jsonl_and_contact_sheet(self):
        from scripts.audit_valid_support_masks import audit_images

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            clean = root / "clean.png"
            artifact = root / "artifact.png"
            Image.fromarray(np.full((32, 40, 3), 128, dtype=np.uint8)).save(clean)
            image = np.full((32, 40, 3), 128, dtype=np.uint8)
            image[:, :12] = 0
            Image.fromarray(image).save(artifact)

            summary = audit_images([clean, artifact], root / "audit", image_scale=1.0, visual_max=2)

            self.assertEqual(summary["aggregate"]["count"], 2)
            self.assertTrue(Path(summary["records_jsonl"]).exists())
            self.assertTrue(Path(summary["contact_sheet"]).exists())
            rows = [json.loads(line) for line in Path(summary["records_jsonl"]).read_text().splitlines()]
            self.assertEqual(len(rows), 2)
            self.assertIn("valid_frac", rows[0]["metrics"])

    def test_matcha_camera_json_converts_to_render_record_payloads(self):
        from scripts.audit_matcha_2dgs_valid_support import matcha_camera_rows_to_render_records

        rows = [
            {
                "id": 7,
                "img_name": "seq1__frame00001",
                "width": 100,
                "height": 50,
                "fx": 80.0,
                "fy": 70.0,
                "position": [1.0, 2.0, 3.0],
                "rotation": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            }
        ]

        records = matcha_camera_rows_to_render_records(rows, scene="ShopFacade", max_records=1)

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["query_id"], "matcha_2dgs:ShopFacade:seq1__frame00001")
        self.assertEqual(records[0]["width"], 100)
        self.assertEqual(records[0]["height"], 50)
        self.assertGreater(records[0]["fovx"], 0.0)
        self.assertGreater(records[0]["fovy"], 0.0)
        self.assertEqual(len(records[0]["pose_w2c"]), 4)
        self.assertEqual(len(records[0]["pose_w2c"][0]), 4)

    def test_matcha_run_discovery_supports_retained_v2_names(self):
        from scripts.audit_matcha_2dgs_valid_support import discover_matcha_scene_runs

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model = root / "OldHospital_n20_long_masked_retrain_retry" / "free_gaussians"
            (model / "point_cloud" / "iteration_30000").mkdir(parents=True)
            (model / "cameras.json").write_text("[]")

            runs = discover_matcha_scene_runs(root)

            self.assertIn("OldHospital", runs)
            self.assertEqual(runs["OldHospital"], model)

    def test_matcha_audit_script_help_runs_when_executed_as_file(self):
        result = subprocess.run(
            [sys.executable, "scripts/audit_matcha_2dgs_valid_support.py", "--help"],
            cwd=Path(__file__).resolve().parents[1],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--runs_root", result.stdout)

    def test_matcha_renderer_adds_2dgs_submodule_paths(self):
        from scripts.render_matcha_records import _add_matcha_to_path

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            gs_root = root / "2d-gaussian-splatting"
            for submodule in ("diff-surfel-rasterization", "simple-knn", "tetra-triangulation"):
                (gs_root / "submodules" / submodule).mkdir(parents=True)

            before = list(sys.path)
            try:
                _add_matcha_to_path(root)
                self.assertIn(str(gs_root), sys.path)
                self.assertIn(str(gs_root / "submodules" / "diff-surfel-rasterization"), sys.path)
                self.assertIn(str(gs_root / "submodules" / "simple-knn"), sys.path)
            finally:
                sys.path[:] = before

    def test_matcha_subprocess_env_prepends_conda_lib(self):
        from scripts.audit_matcha_2dgs_valid_support import matcha_subprocess_env

        with tempfile.TemporaryDirectory() as tmpdir:
            env_root = Path(tmpdir) / "env"
            python_path = env_root / "bin" / "python"
            (env_root / "bin").mkdir(parents=True)
            (env_root / "lib").mkdir()
            python_path.write_text("")

            env = matcha_subprocess_env(python_path, base_env={"LD_LIBRARY_PATH": "/existing"})

            self.assertTrue(env["LD_LIBRARY_PATH"].startswith(str(env_root / "lib")))
            self.assertIn("/existing", env["LD_LIBRARY_PATH"])


if __name__ == "__main__":
    unittest.main()
