import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import yaml


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "make_stdloc_eval_cfg.py"


class MakeStdlocEvalCfgTest(unittest.TestCase):
    def test_writes_detector_landmark_artifact_paths_and_sparse_overrides(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            base_cfg = tmp / "base.yaml"
            out_cfg = tmp / "out.yaml"
            base_cfg.write_text(
                yaml.dump(
                    {
                        "sparse": {
                            "detect_num": 2048,
                            "reprojection_error": 12.0,
                            "detector_path": "detector/30000_detector.pth",
                        },
                        "dense": {},
                    }
                )
            )

            proc = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--base_cfg",
                    str(base_cfg),
                    "--output",
                    str(out_cfg),
                    "--artifact_model_path",
                    "/tmp/model",
                    "--detector_folder",
                    "detector_la",
                    "--detector_iters",
                    "123",
                    "--candidate_teacher_state_path",
                    "detector_la/candidate_teacher_state.pt",
                    "--detect_num",
                    "4096",
                    "--reprojection_error",
                    "8.0",
                    "--nms",
                    "2",
                    "--geometry_balance",
                    "--geometry_balance_max_per_cell",
                    "8",
                    "--geometry_balance_max_matches",
                    "512",
                    "--diagnostics_grid_rows",
                    "3",
                    "--diagnostics_grid_cols",
                    "5",
                    "--diagnostics_voxel_size",
                    "0.5",
                    "--summary_json",
                    str(tmp / "summary.json"),
                ],
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            self.assertEqual(proc.returncode, 0, proc.stderr)
            cfg = yaml.load(out_cfg.read_text(), Loader=yaml.FullLoader)
            self.assertEqual(cfg["model_path"], "/tmp/model")
            sparse = cfg["sparse"]
            self.assertEqual(sparse["detector_path"], "detector_la/123_detector.pth")
            self.assertEqual(sparse["landmark_path"], "detector_la/sampled_idx.pkl")
            self.assertEqual(sparse["landmark_meta_path"], "detector_la/landmark_meta.pt")
            self.assertEqual(sparse["detector_model_path"], "/tmp/model")
            self.assertEqual(sparse["landmark_model_path"], "/tmp/model")
            self.assertEqual(sparse["landmark_meta_model_path"], "/tmp/model")
            self.assertEqual(
                sparse["candidate_teacher_state_path"],
                "detector_la/candidate_teacher_state.pt",
            )
            self.assertEqual(sparse["candidate_teacher_state_model_path"], "/tmp/model")
            self.assertEqual(sparse["detect_num"], 4096)
            self.assertEqual(float(sparse["reprojection_error"]), 8.0)
            self.assertEqual(sparse["nms"], 2)
            self.assertFalse(sparse["use_landmark_prior"])
            self.assertTrue(sparse["diagnostics"]["enabled"])
            self.assertTrue(sparse["diagnostics"]["gt_metrics"])
            self.assertEqual(sparse["diagnostics"]["grid_rows"], 3)
            self.assertEqual(sparse["diagnostics"]["grid_cols"], 5)
            self.assertEqual(float(sparse["diagnostics"]["voxel_size"]), 0.5)
            self.assertTrue(sparse["geometry_balance"]["enabled"])
            self.assertEqual(sparse["geometry_balance"]["max_per_cell"], 8)
            self.assertEqual(sparse["geometry_balance"]["max_matches"], 512)

            summary = json.loads((tmp / "summary.json").read_text())
            self.assertEqual(summary["output"], str(out_cfg))
            self.assertEqual(summary["detector_path"], "detector_la/123_detector.pth")
            self.assertEqual(
                summary["candidate_teacher_state_path"],
                "detector_la/candidate_teacher_state.pt",
            )
            self.assertEqual(summary["nms"], 2)
            self.assertTrue(summary["diagnostics"]["enabled"])
            self.assertEqual(summary["diagnostics"]["grid_rows"], 3)
            self.assertEqual(summary["diagnostics"]["grid_cols"], 5)
            self.assertEqual(float(summary["diagnostics"]["voxel_size"]), 0.5)
            self.assertTrue(summary["geometry_balance"]["enabled"])


if __name__ == "__main__":
    unittest.main()
