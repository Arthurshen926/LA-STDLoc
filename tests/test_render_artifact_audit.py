import importlib.util
import math
import unittest
from pathlib import Path
from types import SimpleNamespace


def load_script():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "audit_render_artifacts.py"
    spec = importlib.util.spec_from_file_location("audit_render_artifacts", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RenderArtifactAuditTest(unittest.TestCase):
    def test_default_thresholds_classify_known_render_artifact_rows(self):
        module = load_script()
        thresholds = module.ArtifactThresholds()

        severe = module.classify_artifact_severity(
            {
                "psnr_mean_matched": 12.6065,
                "ssim": 0.3415,
                "residual_frac_025": 0.2136,
                "alpha_cov_05": 0.9004,
                "mean_abs_bias": 0.0162,
            },
            thresholds,
        )
        mild = module.classify_artifact_severity(
            {
                "psnr_mean_matched": 16.5457,
                "ssim": 0.5073,
                "residual_frac_025": 0.0738,
                "alpha_cov_05": 0.9909,
                "mean_abs_bias": 0.0045,
            },
            thresholds,
        )
        clean = module.classify_artifact_severity(
            {
                "psnr_mean_matched": 16.2282,
                "ssim": 0.5928,
                "residual_frac_025": 0.0903,
                "alpha_cov_05": 0.9958,
                "mean_abs_bias": 0.0098,
            },
            thresholds,
        )

        self.assertEqual(severe, "severe")
        self.assertEqual(mild, "mild")
        self.assertEqual(clean, "none")

    def test_candidate_rows_keep_exact_sequence_paths(self):
        module = load_script()
        rows = [
            {
                "scene": "OldHospital",
                "split": "heldout_query_sample",
                "image_name": "seq2/frame00124.png",
                "psnr_mean_matched": 13.2794,
                "ssim": 0.4038,
                "residual_frac_025": 0.1902,
                "alpha_cov_05": 0.8486,
                "mean_abs_bias": 0.0641,
            },
            {
                "scene": "OldHospital",
                "split": "heldout_query_sample",
                "image_name": "seq1/frame00124.png",
                "psnr_mean_matched": 17.0,
                "ssim": 0.70,
                "residual_frac_025": 0.05,
                "alpha_cov_05": 0.98,
                "mean_abs_bias": 0.01,
            },
        ]

        candidates = module.candidate_rows(rows, severities={"mild", "severe"})

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["image_name"], "seq2/frame00124.png")
        self.assertEqual(candidates[0]["gate_severity"], "severe")

    def test_select_audit_cameras_sorts_before_sequence_query_split_when_requested(self):
        module = load_script()
        cameras = [
            SimpleNamespace(image_name="seq2/frame00005.png"),
            SimpleNamespace(image_name="seq1/frame00001.png"),
            SimpleNamespace(image_name="seq2/frame00001.png"),
            SimpleNamespace(image_name="seq2/frame00002.png"),
        ]

        selected = module.select_audit_cameras(
            train_cameras=cameras,
            test_cameras=[],
            split="heldout_query_sample",
            support_query_split=True,
            query_holdout_ratio=0.5,
            query_split_seed=2026,
            query_split_mode="sequence_block",
            support_query_sort_by_name=True,
        )

        self.assertEqual(
            [camera.image_name for camera in selected],
            ["seq2/frame00001.png", "seq2/frame00002.png", "seq2/frame00005.png"],
        )

    def test_psnr_value_accepts_non_contiguous_tensors(self):
        module = load_script()
        import torch

        rendered = torch.zeros(3, 4, 5).transpose(1, 2)
        target = torch.ones_like(rendered) * 0.5

        value = module.psnr_value(rendered, target)

        self.assertTrue(math.isfinite(value))
        self.assertAlmostEqual(value, 6.0206, places=3)

    def test_region_weight_sidecar_path_preserves_sequence_name(self):
        module = load_script()
        import tempfile
        import torch

        with tempfile.TemporaryDirectory() as tmp:
            rel_path = module._write_region_weight_map(
                tmp,
                "OldHospital",
                "heldout_query_sample",
                "seq2/frame00124.png",
                torch.ones(4, 4),
                {"image_width": 16, "image_height": 16},
            )
            saved = Path(tmp) / rel_path

            self.assertEqual(rel_path, "OldHospital/heldout_query_sample/seq2/frame00124.pt")
            self.assertTrue(saved.exists())


if __name__ == "__main__":
    unittest.main()
