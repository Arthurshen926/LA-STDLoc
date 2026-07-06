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
                    "--detect_num",
                    "4096",
                    "--reprojection_error",
                    "8.0",
                    "--nms",
                    "2",
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
            self.assertEqual(sparse["detect_num"], 4096)
            self.assertEqual(float(sparse["reprojection_error"]), 8.0)
            self.assertEqual(sparse["nms"], 2)
            self.assertFalse(sparse["use_landmark_prior"])

            summary = json.loads((tmp / "summary.json").read_text())
            self.assertEqual(summary["output"], str(out_cfg))
            self.assertEqual(summary["detector_path"], "detector_la/123_detector.pth")
            self.assertEqual(summary["nms"], 2)


if __name__ == "__main__":
    unittest.main()
