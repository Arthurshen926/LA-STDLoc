import importlib.util
import unittest
from pathlib import Path

import torch


def load_script():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "audit_pose_forward_render_artifacts.py"
    spec = importlib.util.spec_from_file_location("audit_pose_forward_render_artifacts", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PoseForwardAuditTest(unittest.TestCase):
    def test_parse_forward_offsets_keeps_zero_candidate_first(self):
        module = load_script()

        offsets = module.parse_forward_offsets("0.10,0,0.20")

        self.assertEqual(offsets, [0.0, 0.1, 0.2])

    def test_forward_pose_moves_camera_center_along_local_positive_z(self):
        module = load_script()
        pose = torch.eye(4, dtype=torch.float32)

        moved = module.forward_pose_w2c(pose, 0.2)
        center = torch.linalg.inv(moved)[:3, 3]

        self.assertTrue(torch.allclose(center, torch.tensor([0.0, 0.0, 0.2]), atol=1e-6))
        self.assertTrue(torch.allclose(moved[:3, 3], torch.tensor([0.0, 0.0, -0.2]), atol=1e-6))

    def test_build_parser_reuses_model_params_longest_edge(self):
        module = load_script()

        args = module.build_parser().parse_args(
            [
                "-s",
                "/tmp/source",
                "-m",
                "/tmp/model",
                "-f",
                "sp",
                "--iteration",
                "32000",
                "--output_csv",
                "/tmp/out.csv",
            ]
        )

        self.assertEqual(args.longest_edge, 640)


if __name__ == "__main__":
    unittest.main()
