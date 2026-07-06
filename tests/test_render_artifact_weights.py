import csv
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


class RenderArtifactWeightsTest(unittest.TestCase):
    def test_load_weight_lookup_filters_scene_split_and_uses_exact_paths(self):
        from localization_training.render_artifacts import load_artifact_weight_lookup

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "candidates.csv"
            with path.open("w", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=["scene", "split", "image_name", "gate_severity"],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "scene": "OldHospital",
                        "split": "heldout_query_sample",
                        "image_name": "seq2/frame00124.png",
                        "gate_severity": "severe",
                    }
                )
                writer.writerow(
                    {
                        "scene": "OldHospital",
                        "split": "heldout_query_sample",
                        "image_name": "seq1/frame00124.png",
                        "gate_severity": "mild",
                    }
                )
                writer.writerow(
                    {
                        "scene": "ShopFacade",
                        "split": "heldout_query_sample",
                        "image_name": "seq2/frame00124.png",
                        "gate_severity": "mild",
                    }
                )
                writer.writerow(
                    {
                        "scene": "OldHospital",
                        "split": "final_test_sample",
                        "image_name": "seq3/frame00009.png",
                        "gate_severity": "severe",
                    }
                )

            lookup = load_artifact_weight_lookup(
                str(path),
                scene_name="OldHospital",
                splits="heldout_query_sample",
                mild_weight=0.65,
                severe_weight=0.25,
            )

        self.assertAlmostEqual(lookup.weight_for_name("seq2/frame00124.png"), 0.25)
        self.assertAlmostEqual(lookup.weight_for_name("seq1/frame00124.png"), 0.65)
        self.assertAlmostEqual(lookup.weight_for_name("frame00124.png"), 1.0)
        self.assertEqual(lookup.summary(), {"mild": 1, "severe": 1})

    def test_metric_quality_weight_maps_render_quality_to_soft_weight(self):
        from localization_training.render_artifacts import metric_quality_weight

        severe_weight = metric_quality_weight(
            {
                "psnr_mean_matched": 12.0,
                "ssim": 0.35,
                "residual_frac_025": 0.25,
                "alpha_cov_05": 0.80,
                "mean_abs_bias": 0.08,
            },
            mild_weight=0.65,
            severe_weight=0.25,
        )
        clean_weight = metric_quality_weight(
            {
                "psnr_mean_matched": 18.0,
                "ssim": 0.70,
                "residual_frac_025": 0.02,
                "alpha_cov_05": 0.98,
                "mean_abs_bias": 0.01,
            },
            mild_weight=0.65,
            severe_weight=0.25,
        )

        self.assertLess(severe_weight, 0.65)
        self.assertGreaterEqual(severe_weight, 0.25)
        self.assertAlmostEqual(clean_weight, 1.0)

    def test_continuous_quality_weight_uses_metric_distance_not_only_severity(self):
        from localization_training.render_artifacts import continuous_quality_weight

        clean = {
            "psnr_mean_matched": 18.0,
            "ssim": 0.72,
            "residual_frac_025": 0.02,
            "alpha_cov_05": 0.98,
            "mean_abs_bias": 0.01,
        }
        borderline = {
            "psnr_mean_matched": 15.0,
            "ssim": 0.55,
            "residual_frac_025": 0.10,
            "alpha_cov_05": 0.85,
            "mean_abs_bias": 0.04,
        }
        bad = {
            "psnr_mean_matched": 12.0,
            "ssim": 0.35,
            "residual_frac_025": 0.25,
            "alpha_cov_05": 0.70,
            "mean_abs_bias": 0.08,
        }

        clean_weight = continuous_quality_weight(clean, min_weight=0.70)
        borderline_weight = continuous_quality_weight(borderline, min_weight=0.70)
        bad_weight = continuous_quality_weight(bad, min_weight=0.70)

        self.assertAlmostEqual(clean_weight, 1.0)
        self.assertGreater(borderline_weight, bad_weight)
        self.assertGreaterEqual(bad_weight, 0.70)
        self.assertLess(bad_weight, 1.0)

    def test_continuous_quality_weight_does_not_saturate_all_severe_rows(self):
        from localization_training.render_artifacts import continuous_quality_weight

        barely_severe = {
            "psnr_mean_matched": 13.40,
            "ssim": 0.55,
            "residual_frac_025": 0.11,
            "alpha_cov_05": 0.98,
            "mean_abs_bias": 0.01,
        }
        extreme_severe = {
            "psnr_mean_matched": 11.40,
            "ssim": 0.28,
            "residual_frac_025": 0.30,
            "alpha_cov_05": 0.65,
            "mean_abs_bias": 0.14,
        }

        barely_weight = continuous_quality_weight(barely_severe, min_weight=0.70)
        extreme_weight = continuous_quality_weight(extreme_severe, min_weight=0.70)

        self.assertGreater(barely_weight, extreme_weight + 0.05)
        self.assertGreater(barely_weight, 0.70)
        self.assertGreaterEqual(extreme_weight, 0.70)

    def test_load_weight_lookup_can_use_continuous_mode(self):
        from localization_training.render_artifacts import load_artifact_weight_lookup

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "candidates.csv"
            with path.open("w", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "scene",
                        "split",
                        "image_name",
                        "gate_severity",
                        "psnr_mean_matched",
                        "ssim",
                        "alpha_cov_05",
                        "residual_frac_025",
                        "mean_abs_bias",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "scene": "OldHospital",
                        "split": "heldout_query_sample",
                        "image_name": "seq1/frame00001.png",
                        "gate_severity": "severe",
                        "psnr_mean_matched": "14.0",
                        "ssim": "0.54",
                        "alpha_cov_05": "0.97",
                        "residual_frac_025": "0.14",
                        "mean_abs_bias": "0.03",
                    }
                )
                writer.writerow(
                    {
                        "scene": "OldHospital",
                        "split": "heldout_query_sample",
                        "image_name": "seq1/frame00002.png",
                        "gate_severity": "severe",
                        "psnr_mean_matched": "11.0",
                        "ssim": "0.30",
                        "alpha_cov_05": "0.70",
                        "residual_frac_025": "0.25",
                        "mean_abs_bias": "0.09",
                    }
                )

            lookup = load_artifact_weight_lookup(
                str(path),
                scene_name="OldHospital",
                splits="heldout_query_sample",
                severities="severe",
                mode="continuous",
                continuous_min_weight=0.70,
            )

        self.assertGreater(
            lookup.weight_for_name("seq1/frame00001.png"),
            lookup.weight_for_name("seq1/frame00002.png"),
        )
        self.assertGreaterEqual(lookup.weight_for_name("seq1/frame00002.png"), 0.70)
        self.assertLess(lookup.weight_for_name("seq1/frame00001.png"), 1.0)

    def test_lookup_weights_camera_objects_by_image_name(self):
        from localization_training.render_artifacts import ArtifactWeightLookup

        lookup = ArtifactWeightLookup(
            {"seq2/frame00124.png": 0.25},
            {"seq2/frame00124.png": "severe"},
        )

        self.assertAlmostEqual(lookup.weight_for_camera(SimpleNamespace(image_name="seq2/frame00124.png")), 0.25)
        self.assertAlmostEqual(lookup.weight_for_camera(SimpleNamespace(image_name="seq1/frame00124.png")), 1.0)

    def test_combine_artifact_confidence_modes(self):
        import torch

        from localization_training.render_artifacts import combine_artifact_confidence

        local = torch.tensor([1.0, 0.5, 0.25], dtype=torch.float32)

        product = combine_artifact_confidence(local, image_weight=0.8, mode="product")
        minimum = combine_artifact_confidence(local, image_weight=0.8, mode="min")
        unchanged = combine_artifact_confidence(local, image_weight=0.8, mode="none")

        torch.testing.assert_close(product, torch.tensor([0.8, 0.4, 0.2]))
        torch.testing.assert_close(minimum, torch.tensor([0.8, 0.5, 0.25]))
        torch.testing.assert_close(unchanged, local)

        with self.assertRaises(ValueError):
            combine_artifact_confidence(local, image_weight=0.8, mode="unsupported")

    def test_region_weight_lookup_loads_manifest_and_samples_uv(self):
        import torch

        from localization_training.render_artifacts import load_artifact_region_weight_lookup

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            map_path = root / "maps" / "seq2" / "frame00124.pt"
            map_path.parent.mkdir(parents=True)
            weight_map = torch.ones(4, 4, dtype=torch.float32)
            weight_map[:, 2:] = 0.2
            torch.save({"weight": weight_map, "image_width": 16, "image_height": 16}, map_path)

            manifest = root / "manifest.csv"
            with manifest.open("w", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=["scene", "split", "image_name", "gate_severity", "region_weight_path"],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "scene": "OldHospital",
                        "split": "heldout_query_sample",
                        "image_name": "seq2/frame00124.png",
                        "gate_severity": "severe",
                        "region_weight_path": str(map_path.relative_to(root)),
                    }
                )

            lookup = load_artifact_region_weight_lookup(
                str(manifest),
                scene_name="OldHospital",
                splits="heldout_query_sample",
                severities="severe",
            )
            weights = lookup.sample_weights_for_name(
                "seq2/frame00124.png",
                torch.tensor([[2.0, 8.0], [13.0, 8.0]], dtype=torch.float32),
                image_size=(16, 16),
            )

        self.assertGreater(weights[0].item(), 0.95)
        self.assertLess(weights[1].item(), 0.35)
        self.assertEqual(lookup.summary(), {"severe": 1})

    def test_region_weight_lookup_resolves_relative_paths_from_explicit_root(self):
        import torch

        from localization_training.render_artifacts import load_artifact_region_weight_lookup

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            log_dir = root / "logs"
            map_root = root / "maps"
            log_dir.mkdir()
            map_path = map_root / "OldHospital" / "heldout_query_sample" / "seq2" / "frame00124.pt"
            map_path.parent.mkdir(parents=True)
            torch.save(torch.ones(4, 4, dtype=torch.float32) * 0.4, map_path)

            manifest = log_dir / "manifest.csv"
            with manifest.open("w", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=["scene", "split", "image_name", "gate_severity", "region_weight_path"],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "scene": "OldHospital",
                        "split": "heldout_query_sample",
                        "image_name": "seq2/frame00124.png",
                        "gate_severity": "severe",
                        "region_weight_path": str(map_path.relative_to(map_root)),
                    }
                )

            lookup = load_artifact_region_weight_lookup(
                str(manifest),
                root=str(map_root),
                scene_name="OldHospital",
                splits="heldout_query_sample",
                severities="severe",
            )

        self.assertEqual(lookup.path_for_name("seq2/frame00124.png"), str(map_path))
        self.assertEqual(lookup.summary(), {"severe": 1})

    def test_local_artifact_weight_map_keeps_clean_regions_high(self):
        import torch

        from localization_training.render_artifacts import local_artifact_weight_map

        rendered = torch.zeros(3, 16, 16, dtype=torch.float32)
        target = torch.zeros_like(rendered)
        rendered[:, 8:, 8:] = 1.0

        weight_map = local_artifact_weight_map(
            rendered,
            target,
            output_size=4,
            min_weight=0.25,
            power=1.0,
        )

        self.assertGreater(weight_map[0, 0].item(), 0.95)
        self.assertLess(weight_map[-1, -1].item(), 0.35)


if __name__ == "__main__":
    unittest.main()
