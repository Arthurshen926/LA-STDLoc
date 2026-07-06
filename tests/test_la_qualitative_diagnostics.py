import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from la_diagnostics.qualitative import BatchInputs, generate_qualitative_report


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_image(path, color):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 48), color=color).save(path)


class LAQualitativeDiagnosticsTest(unittest.TestCase):
    def test_report_records_batch_outputs_and_sample_groups(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_root = root / "images"
            _write_image(image_root / "seq8" / "frame00032.png", (220, 40, 40))
            _write_image(image_root / "seq8" / "frame00043.png", (40, 180, 80))
            _write_image(image_root / "seq8" / "frame00028.png", (40, 80, 220))
            _write_image(image_root / "seq1" / "frame00175.png", (220, 180, 40))
            _write_image(image_root / "seq1" / "frame00130.png", (120, 120, 220))

            baseline_results = root / "baseline_results.json"
            current_results = root / "current_results.json"
            _write_json(
                baseline_results,
                [
                    {"image_name": "seq8/frame00032.png", "sparse_TE": 100.0, "sparse_AE": 1.0},
                    {"image_name": "seq8/frame00043.png", "sparse_TE": 70.0, "sparse_AE": 0.7},
                    {"image_name": "seq8/frame00028.png", "sparse_TE": 30.0, "sparse_AE": 0.3},
                ],
            )
            _write_json(
                current_results,
                [
                    {"image_name": "seq8/frame00032.png", "sparse_TE": 120.0, "sparse_AE": 1.2},
                    {"image_name": "seq8/frame00043.png", "sparse_TE": 40.0, "sparse_AE": 0.4},
                    {"image_name": "seq8/frame00028.png", "sparse_TE": 55.0, "sparse_AE": 0.55},
                ],
            )

            audit_csv = root / "artifact_audit.csv"
            _write_csv(
                audit_csv,
                [
                    {
                        "scene": "OldHospital",
                        "split": "heldout_query_sample",
                        "image_name": "seq1/frame00175.png",
                        "psnr_mean_matched": "11.4",
                        "ssim": "0.54",
                        "residual_frac_025": "0.41",
                        "alpha_cov_05": "0.69",
                        "mean_abs_bias": "0.20",
                        "gate_severity": "severe",
                    },
                    {
                        "scene": "OldHospital",
                        "split": "heldout_query_sample",
                        "image_name": "seq1/frame00130.png",
                        "psnr_mean_matched": "18.0",
                        "ssim": "0.80",
                        "residual_frac_025": "0.02",
                        "alpha_cov_05": "1.0",
                        "mean_abs_bias": "0.01",
                        "gate_severity": "mild",
                    },
                ],
            )
            region_root = root / "region_maps"
            region_path = region_root / "OldHospital" / "heldout_query_sample" / "seq1" / "frame00175.npy"
            region_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(region_path, np.array([[0.25, 0.5], [0.75, 1.0]], dtype=np.float32))
            region_manifest = root / "region_manifest.csv"
            _write_csv(
                region_manifest,
                [
                    {
                        "scene": "OldHospital",
                        "split": "heldout_query_sample",
                        "image_name": "seq1/frame00175.png",
                        "gate_severity": "severe",
                        "region_weight_path": "OldHospital/heldout_query_sample/seq1/frame00175.npy",
                        "region_weight_min": "0.25",
                        "region_weight_mean": "0.625",
                        "region_weight_weighted_frac": "0.75",
                    }
                ],
            )

            out_dir = root / "report"
            summary = generate_qualitative_report(
                BatchInputs(
                    batch_name="p20_oldhospital",
                    scene="OldHospital",
                    current_results=current_results,
                    baseline_results=baseline_results,
                    artifact_audit_csv=audit_csv,
                    region_manifest_csv=region_manifest,
                    region_weight_root=region_root,
                    image_root=image_root,
                    output_dir=out_dir,
                    registry_path=out_dir / "registry.jsonl",
                    top_k=2,
                    notes="unit test batch",
                )
            )

            self.assertEqual(summary["batch_name"], "p20_oldhospital")
            self.assertEqual(summary["final_test"]["paired_count"], 3)
            self.assertEqual(summary["artifact_teacher"]["severity_counts"]["severe"], 1)
            for name in [
                "summary.json",
                "sample_flow.csv",
                "index.md",
                "final_test_worst.png",
                "final_test_improved_regressed.png",
                "artifact_teacher_severe.png",
                "artifact_teacher_flagged.png",
                "registry.jsonl",
            ]:
                self.assertTrue((out_dir / name).exists(), name)

            with (out_dir / "sample_flow.csv").open(newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            groups = {row["group"] for row in rows}
            self.assertIn("final_worst", groups)
            self.assertIn("final_improved", groups)
            self.assertIn("final_regressed", groups)
            self.assertIn("artifact_severe", groups)
            self.assertIn("artifact_mild", groups)
            improved = next(row for row in rows if row["group"] == "final_improved")
            self.assertEqual(improved["image_name"], "seq8/frame00043.png")
            self.assertEqual(float(improved["delta_te"]), 30.0)

            registry_lines = (out_dir / "registry.jsonl").read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(registry_lines), 1)
            self.assertEqual(json.loads(registry_lines[0])["batch_name"], "p20_oldhospital")

    def test_artifact_teacher_and_final_test_rows_keep_distinct_stages(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_json(
                root / "current.json",
                [{"image_name": "seq8/frame00032.png", "sparse_TE": 120.0, "sparse_AE": 1.2}],
            )
            _write_csv(
                root / "audit.csv",
                [
                    {
                        "scene": "OldHospital",
                        "split": "heldout_query_sample",
                        "image_name": "seq1/frame00175.png",
                        "psnr_mean_matched": "11.4",
                        "ssim": "0.54",
                        "residual_frac_025": "0.41",
                        "alpha_cov_05": "0.69",
                        "mean_abs_bias": "0.20",
                        "gate_severity": "severe",
                    }
                ],
            )

            out_dir = root / "report"
            generate_qualitative_report(
                BatchInputs(
                    batch_name="split_check",
                    scene="OldHospital",
                    current_results=root / "current.json",
                    artifact_audit_csv=root / "audit.csv",
                    output_dir=out_dir,
                    top_k=1,
                )
            )

            with (out_dir / "sample_flow.csv").open(newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            final_row = next(row for row in rows if row["stage"] == "final_test")
            artifact_row = next(row for row in rows if row["stage"] == "artifact_teacher")
            self.assertEqual(final_row["image_name"], "seq8/frame00032.png")
            self.assertEqual(final_row["gate_severity"], "")
            self.assertEqual(artifact_row["image_name"], "seq1/frame00175.png")
            self.assertEqual(artifact_row["split"], "heldout_query_sample")
            self.assertEqual(artifact_row["sparse_te"], "")

    def test_registry_upserts_same_batch_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_json(
                root / "current.json",
                [{"image_name": "seq8/frame00032.png", "sparse_TE": 120.0, "sparse_AE": 1.2}],
            )
            registry = root / "registry.jsonl"
            for note in ["first", "second"]:
                generate_qualitative_report(
                    BatchInputs(
                        batch_name="same_batch",
                        scene="OldHospital",
                        current_results=root / "current.json",
                        output_dir=root / "report",
                        registry_path=registry,
                        top_k=1,
                        notes=note,
                    )
                )

            lines = registry.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 1)
            self.assertEqual(json.loads(lines[0])["notes"], "second")

    def test_region_pt_dict_weight_is_supported(self):
        import torch

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_json(
                root / "current.json",
                [{"image_name": "seq8/frame00032.png", "sparse_TE": 120.0, "sparse_AE": 1.2}],
            )
            _write_csv(
                root / "audit.csv",
                [
                    {
                        "scene": "OldHospital",
                        "split": "heldout_query_sample",
                        "image_name": "seq1/frame00175.png",
                        "psnr_mean_matched": "11.4",
                        "ssim": "0.54",
                        "residual_frac_025": "0.41",
                        "alpha_cov_05": "0.69",
                        "mean_abs_bias": "0.20",
                        "gate_severity": "severe",
                    }
                ],
            )
            region_root = root / "region_maps"
            region_path = region_root / "OldHospital" / "heldout_query_sample" / "seq1" / "frame00175.pt"
            region_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"weight": torch.tensor([[0.25, 0.75], [1.0, 0.5]])}, region_path)
            _write_csv(
                root / "manifest.csv",
                [
                    {
                        "scene": "OldHospital",
                        "split": "heldout_query_sample",
                        "image_name": "seq1/frame00175.png",
                        "gate_severity": "severe",
                        "region_weight_path": "OldHospital/heldout_query_sample/seq1/frame00175.pt",
                    }
                ],
            )

            out_dir = root / "report"
            generate_qualitative_report(
                BatchInputs(
                    batch_name="pt_dict",
                    scene="OldHospital",
                    current_results=root / "current.json",
                    artifact_audit_csv=root / "audit.csv",
                    region_manifest_csv=root / "manifest.csv",
                    region_weight_root=region_root,
                    output_dir=out_dir,
                    top_k=1,
                )
            )

            with (out_dir / "sample_flow.csv").open(newline="", encoding="utf-8") as f:
                artifact_row = next(row for row in csv.DictReader(f) if row["stage"] == "artifact_teacher")
            self.assertEqual(float(artifact_row["region_weight_min"]), 0.25)
            self.assertAlmostEqual(float(artifact_row["region_weight_mean"]), 0.625)

    def test_cli_generates_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_json(
                root / "current.json",
                [{"image_name": "seq8/frame00032.png", "sparse_TE": 120.0, "sparse_AE": 1.2}],
            )
            out_dir = root / "report"
            script = Path(__file__).resolve().parents[1] / "scripts" / "run_la_qualitative_diagnostics.py"
            argv = [
                sys.executable,
                str(script),
                "--batch_name",
                "cli_batch",
                "--scene",
                "OldHospital",
                "--current_results",
                str(root / "current.json"),
                "--output_dir",
                str(out_dir),
                "--top_k",
                "1",
            ]
            import subprocess

            completed = subprocess.run(argv, check=False, text=True, capture_output=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertTrue((out_dir / "summary.json").exists())
            self.assertIn("cli_batch", (out_dir / "index.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
