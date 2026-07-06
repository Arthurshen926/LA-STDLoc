import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch


class ArtifactDetectorRepairTest(unittest.TestCase):
    def test_valid_mask_erodes_boundaries_and_filters_small_components(self):
        from la_artifacts.valid_mask import ArtifactValidMaskBuilder, ArtifactValidMaskConfig

        score = torch.ones(10, 10) * 0.9
        score[1:8, 1:8] = 0.1
        score[8, 8] = 0.1

        result = ArtifactValidMaskBuilder(
            ArtifactValidMaskConfig(
                max_artifact_score=0.3,
                erosion_radius=1,
                min_component_area=9,
            )
        ).build(score)

        self.assertFalse(bool(result.mask[1, 1]))
        self.assertTrue(bool(result.mask[4, 4]))
        self.assertFalse(bool(result.mask[8, 8]))
        self.assertEqual(result.summary["component_count"], 1)
        self.assertAlmostEqual(result.summary["valid_frac"], 25 / 100, places=4)
        self.assertEqual(result.components[0]["bbox_xyxy"], [2, 2, 7, 7])

    def test_valid_mask_can_filter_points_and_downsample_to_feature_grid(self):
        from la_artifacts.valid_mask import ArtifactValidMaskBuilder, ArtifactValidMaskConfig

        score = torch.ones(8, 8) * 0.9
        score[:4, :4] = 0.1
        result = ArtifactValidMaskBuilder(
            ArtifactValidMaskConfig(
                max_artifact_score=0.3,
                erosion_radius=0,
                min_component_area=1,
            )
        ).build(score)

        points = torch.tensor([[1.0, 1.0], [6.0, 6.0], [3.9, 0.2]])
        keep = result.valid_points(points)
        feature_mask = result.to_feature_mask((2, 2), min_valid_fraction=0.5)

        torch.testing.assert_close(keep, torch.tensor([True, False, True]))
        torch.testing.assert_close(
            feature_mask,
            torch.tensor([[True, False], [False, False]]),
        )

    def test_build_manifest_helper_writes_synthetic_valid_mask_metadata(self):
        from la_artifacts.detector import ArtifactEvidence
        from la_artifacts.pseudo_query import PseudoQueryRecord
        from scripts.build_pseudo_query_manifest import _write_synthetic_valid_mask

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            record = PseudoQueryRecord(
                query_id="synthetic_rgb:synthetic/000000.png",
                scene="ShopFacade",
                source="synthetic_rgb",
                image_name="synthetic/000000.png",
                image_path=str(tmp / "synthetic" / "000000.png"),
                pose_w2c=torch.eye(4).tolist(),
                fovx=0.8,
                fovy=0.6,
                width=8,
                height=8,
            )
            score = torch.ones(8, 8) * 0.9
            score[:4, :4] = 0.1
            evidence = ArtifactEvidence(score_map=score, summary={"artifact_score_mean": 0.7})

            result = _write_synthetic_valid_mask(
                record,
                evidence,
                mask_root=tmp / "valid_masks",
                max_artifact_score=0.3,
                erosion_radius=0,
                min_component_area=1,
            )

            mask_path = Path(record.meta["artifact_valid_mask"]["mask_path"])
            self.assertTrue(mask_path.exists())
            self.assertEqual(result.summary["component_count"], 1)
            self.assertAlmostEqual(record.meta["artifact_valid_mask"]["valid_frac"], 0.25, places=4)

    def test_valid_mask_eval_summarizes_keypoint_retention(self):
        from la_artifacts.valid_mask import ArtifactValidMask
        from scripts.evaluate_valid_masks import _summarize_keypoint_retention

        valid = torch.zeros(8, 8, dtype=torch.bool)
        valid[:4, :4] = True
        result = ArtifactValidMask(
            mask=valid,
            score_map=torch.zeros(8, 8),
            components=[],
            summary={"valid_frac": 0.25},
        )
        points = torch.tensor([[1.0, 1.0], [3.8, 0.2], [6.0, 6.0]])

        summary = _summarize_keypoint_retention(result, points)

        self.assertEqual(summary["keypoint_count"], 3)
        self.assertEqual(summary["valid_keypoint_count"], 2)
        self.assertAlmostEqual(summary["valid_keypoint_frac"], 2 / 3, places=4)

    def test_valid_mask_eval_writes_comparison_sheet(self):
        from PIL import Image

        from scripts.evaluate_valid_masks import ValidMaskVisualRecord, build_valid_mask_contact_sheet

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            image_path = tmp / "query.png"
            ref_path = tmp / "ref.png"
            out_path = tmp / "sheet.jpg"
            Image.new("RGB", (16, 12), color=(120, 120, 120)).save(image_path)
            Image.new("RGB", (16, 12), color=(80, 80, 80)).save(ref_path)
            mask = torch.zeros(12, 16, dtype=torch.bool)
            mask[:, :8] = True
            score = torch.ones(12, 16) * 0.8
            score[:, :8] = 0.1
            row = ValidMaskVisualRecord(
                query_id="synthetic_rgb:query.png",
                source="synthetic_rgb",
                image_path=str(image_path),
                reference_path=str(ref_path),
                score_map=score,
                valid_mask=mask,
                keypoints_xy=torch.tensor([[2.0, 2.0], [12.0, 8.0]]),
                valid_keypoints=torch.tensor([True, False]),
                metrics={"valid_frac": 0.5, "valid_keypoint_frac": 0.5},
            )

            build_valid_mask_contact_sheet([row], out_path, max_records=1)

            self.assertTrue(out_path.exists())
            self.assertGreater(out_path.stat().st_size, 0)

    def test_teacher_cache_helper_builds_sparse_valid_mask_for_allowed_source(self):
        from la_artifacts.pseudo_query import PseudoQueryRecord
        from scripts.build_pseudo_teacher_cache import _build_sparse_valid_mask_for_record

        record = PseudoQueryRecord(
            query_id="synthetic_rgb:synthetic/000000.png",
            scene="ShopFacade",
            source="synthetic_rgb",
            image_name="synthetic/000000.png",
            image_path="",
            pose_w2c=torch.eye(4).tolist(),
            fovx=0.8,
            fovy=0.6,
            width=8,
            height=8,
        )
        query = torch.ones(3, 16, 16) * 0.5

        mask, summary = _build_sparse_valid_mask_for_record(
            record,
            query,
            enabled=True,
            allowed_sources={"synthetic_rgb"},
            output_dir="",
            max_artifact_score=0.45,
            erosion_radius=0,
            min_component_area=1,
        )

        self.assertIsNotNone(mask)
        self.assertEqual(tuple(mask.shape), (16, 16))
        self.assertIn("valid_frac", summary)
        self.assertTrue(summary["enabled"])

    def test_teacher_cache_helper_can_build_no_reference_support_mask(self):
        from la_artifacts.pseudo_query import PseudoQueryRecord
        from scripts.build_pseudo_teacher_cache import _build_sparse_guidance_for_record

        record = PseudoQueryRecord(
            query_id="synthetic_rgb:synthetic/structured.png",
            scene="ShopFacade",
            source="synthetic_rgb",
            image_name="synthetic/structured.png",
            image_path="",
            pose_w2c=torch.eye(4).tolist(),
            fovx=0.8,
            fovy=0.6,
            width=32,
            height=32,
        )
        query = torch.full((3, 32, 32), 0.5)
        query[:, :, 14:18] = 0.95

        valid_mask, support_score, summary = _build_sparse_guidance_for_record(
            record,
            query,
            enabled=True,
            allowed_sources={"synthetic_rgb"},
            mode="no_reference",
            no_reference_support_threshold=0.22,
            no_reference_support_dilate_radius=1,
            no_reference_support_min_area=1,
        )

        self.assertIsNotNone(valid_mask)
        self.assertIsNotNone(support_score)
        self.assertTrue(summary["enabled"])
        self.assertEqual(summary["mode"], "no_reference")
        self.assertIn("support_frac", summary)
        self.assertEqual(tuple(valid_mask.shape), (32, 32))
        self.assertEqual(tuple(support_score.shape), (32, 32))
        self.assertGreater(support_score[:, 13:19].float().mean().item(), 0.2)
        self.assertLess(support_score[:, :6].float().mean().item(), 0.2)

    def test_teacher_cache_records_dense_valid_mask_diagnostics(self):
        script = Path(__file__).resolve().parents[1] / "scripts" / "build_pseudo_teacher_cache.py"
        text = script.read_text()

        self.assertIn('"dense_valid_mask_enabled"', text)
        self.assertIn('"dense_valid_mask_valid_cells"', text)
        self.assertIn('"dense_valid_mask_valid_frac"', text)

    def test_teacher_cache_helper_skips_disallowed_source(self):
        from la_artifacts.pseudo_query import PseudoQueryRecord
        from scripts.build_pseudo_teacher_cache import _build_sparse_valid_mask_for_record

        record = PseudoQueryRecord(
            query_id="train_rgb:seq/frame.png",
            scene="ShopFacade",
            source="train_rgb",
            image_name="seq/frame.png",
            image_path="",
            pose_w2c=torch.eye(4).tolist(),
            fovx=0.8,
            fovy=0.6,
            width=8,
            height=8,
        )

        mask, summary = _build_sparse_valid_mask_for_record(
            record,
            torch.zeros(3, 8, 8),
            enabled=True,
            allowed_sources={"synthetic_rgb"},
        )

        self.assertIsNone(mask)
        self.assertFalse(summary["enabled"])
        self.assertEqual(summary["reason"], "source_not_enabled")

    def test_pseudo_query_quality_gate_can_be_disabled_for_no_reference_mainline(self):
        from scripts.build_pseudo_query_manifest import _synthetic_quality_gate_from_args

        args = SimpleNamespace(
            skip_synthetic_quality_gate=True,
            synthetic_accept_score=0.65,
            synthetic_qa_max_mean=0.60,
            synthetic_qa_max_p95=-1.0,
            synthetic_qa_max_mild_frac=0.85,
            synthetic_qa_max_severe_frac=0.58,
            synthetic_qa_max_low_detail_mean=0.60,
        )

        self.assertIsNone(_synthetic_quality_gate_from_args(args))

    def test_matcha_renderer_preserves_manifest_pose_and_intrinsics_payload(self):
        from la_artifacts.pseudo_query import PseudoQueryRecord
        from scripts.render_matcha_records import _record_payload

        pose = torch.eye(4)
        pose[0, 3] = 1.25
        record = PseudoQueryRecord(
            query_id="synthetic_rgb:synthetic/000003.png",
            scene="ShopFacade",
            source="synthetic_rgb",
            image_name="synthetic/000003.png",
            image_path="/tmp/synthetic/000003.png",
            pose_w2c=pose.tolist(),
            fovx=0.9,
            fovy=0.7,
            width=640,
            height=360,
        )

        payload = _record_payload(record, index=3)

        self.assertEqual(payload["index"], 3)
        self.assertEqual(payload["query_id"], record.query_id)
        self.assertEqual(payload["pose_w2c"], pose.tolist())
        self.assertEqual(payload["width"], 640)
        self.assertEqual(payload["height"], 360)
        self.assertAlmostEqual(payload["fovx"], 0.9)
        self.assertAlmostEqual(payload["fovy"], 0.7)

    def test_matcha_renderer_loads_multiline_jsonl_manifest(self):
        from scripts.render_matcha_records import _load_manifest_records

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "records.jsonl"
            rows = [
                {"query_id": "synthetic_rgb:synthetic/000000.png"},
                {"query_id": "synthetic_rgb:synthetic/000001.png"},
            ]
            path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")

            loaded = _load_manifest_records(path)

            self.assertEqual([row["query_id"] for row in loaded], [row["query_id"] for row in rows])

    def test_matcha_backend_consumes_external_rendered_frames(self):
        from PIL import Image

        from la_artifacts.pseudo_query import PseudoQueryRecord
        from scripts import build_pseudo_query_manifest as manifest_builder

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            record = PseudoQueryRecord(
                query_id="synthetic_rgb:synthetic/000000.png",
                scene="ShopFacade",
                source="synthetic_rgb",
                image_name="synthetic/000000.png",
                image_path=str(tmp / "synthetic_rgb" / "000000.png"),
                pose_w2c=torch.eye(4).tolist(),
                fovx=0.8,
                fovy=0.6,
                width=8,
                height=6,
            )

            def fake_run(command, check):
                self.assertIn("--manifest", command)
                output_index = command.index("--output_dir") + 1
                frame_dir = Path(command[output_index])
                frame_dir.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (8, 6), color=(10, 20, 30)).save(frame_dir / "000000.png")
                return SimpleNamespace(returncode=0)

            original_run = manifest_builder.subprocess.run
            try:
                manifest_builder.subprocess.run = fake_run
                rendered = manifest_builder._render_synthetic_records_matcha(
                    [record],
                    model_path=str(tmp / "matcha_model"),
                    render_root=str(tmp / "matcha_render"),
                    matcha_root="/root/MAtCha",
                    matcha_python="python",
                    iteration=30000,
                    quality_gate=None,
                    valid_mask_root="",
                )
            finally:
                manifest_builder.subprocess.run = original_run

            self.assertEqual(len(rendered), 1)
            self.assertTrue(Path(rendered[0].image_path).exists())
            self.assertEqual(rendered[0].width, 8)
            self.assertEqual(rendered[0].height, 6)
            self.assertEqual(rendered[0].repair_action, "matcha_render")
            self.assertEqual(rendered[0].meta["render_backend"], "matcha")
            self.assertEqual(rendered[0].reason, "ok")

    def test_detector_combines_rgb_feature_and_alpha_scores(self):
        from la_artifacts.detector import ArtifactDetector, ArtifactDetectorConfig

        rendered_rgb = torch.zeros(3, 4, 4)
        target_rgb = torch.zeros(3, 4, 4)
        target_rgb[:, :, 2:] = 1.0
        rendered_feature = torch.zeros(2, 4, 4)
        target_feature = torch.zeros(2, 4, 4)
        rendered_feature[0] = 1.0
        target_feature[0, :, :2] = 1.0
        target_feature[1, :, 2:] = 1.0
        alpha = torch.ones(4, 4)
        alpha[:, 3:] = 0.0

        detector = ArtifactDetector(
            ArtifactDetectorConfig(
                rgb_weight=1.0,
                feature_weight=1.0,
                alpha_weight=1.0,
                rgb_residual_start=0.1,
                rgb_residual_stop=0.9,
                feature_residual_start=0.1,
                feature_residual_stop=0.9,
                alpha_threshold=0.5,
            )
        )
        evidence = detector.detect(
            rendered_rgb=rendered_rgb,
            target_rgb=target_rgb,
            rendered_feature=rendered_feature,
            target_feature=target_feature,
            alpha=alpha,
        )

        self.assertEqual(tuple(evidence.score_map.shape), (4, 4))
        self.assertGreater(evidence.score_map[:, 3].mean().item(), evidence.score_map[:, 0].mean().item())
        self.assertIn("artifact_score_mean", evidence.summary)
        self.assertGreater(evidence.summary["artifact_mild_frac"], 0.0)

    def test_gaussian_scores_from_contributors(self):
        from la_artifacts.detector import ArtifactDetector

        ids = torch.tensor([[0, 1], [1, 2]])
        weights = torch.tensor([[0.75, 0.25], [0.25, 0.75]])
        anchor_scores = torch.tensor([0.2, 0.8])
        scores = ArtifactDetector().gaussian_scores_from_contributors(
            ids,
            weights,
            anchor_scores,
            gaussian_count=3,
        )

        torch.testing.assert_close(scores, torch.tensor([0.2, 0.5, 0.8]))

    def test_detector_rejects_low_texture_render_without_target(self):
        from la_artifacts.detector import ArtifactDetector, ArtifactDetectorConfig

        rendered_rgb = torch.full((3, 16, 16), 0.5)
        detector = ArtifactDetector(ArtifactDetectorConfig(low_texture_std_threshold=0.03))
        evidence = detector.detect(rendered_rgb=rendered_rgb)

        self.assertGreater(evidence.summary["artifact_score_mean"], 0.9)
        self.assertIn("low_texture_mean", evidence.summary)

    def test_detector_rejects_smooth_low_gradient_render_without_target(self):
        from la_artifacts.detector import ArtifactDetector, ArtifactDetectorConfig

        ramp = torch.linspace(0.35, 0.45, 128)[None, None, :].expand(3, 128, 128)
        detector = ArtifactDetector(
            ArtifactDetectorConfig(
                low_texture_std_threshold=0.03,
                low_texture_grad_threshold=0.002,
            )
        )
        evidence = detector.detect(rendered_rgb=ramp)

        self.assertGreater(evidence.summary["artifact_score_mean"], 0.5)

    def test_detector_flags_local_low_detail_render_without_target(self):
        from la_artifacts.detector import ArtifactDetector, ArtifactDetectorConfig

        rendered_rgb = torch.full((3, 64, 64), 0.5)
        rendered_rgb[:, 20:44, 20:44] = 0.8
        detector = ArtifactDetector(
            ArtifactDetectorConfig(
                low_texture_std_threshold=0.001,
                low_texture_grad_threshold=0.0,
                low_detail_grad_threshold=0.035,
                low_detail_pool=15,
            )
        )
        evidence = detector.detect(rendered_rgb=rendered_rgb)

        self.assertGreater(evidence.summary["low_detail_mean"], 0.7)
        self.assertGreater(evidence.summary["artifact_score_mean"], 0.7)

    def test_repair_suppresses_contributors_and_opacity(self):
        from la_artifacts.repair import ArtifactRepair, ArtifactRepairConfig

        repair = ArtifactRepair(ArtifactRepairConfig(min_opacity_multiplier=0.2))
        scores = torch.tensor([0.0, 0.5, 1.0])
        multipliers = repair.gaussian_opacity_multiplier(scores)
        torch.testing.assert_close(multipliers, torch.tensor([1.0, 0.6, 0.2]))

        ids = torch.tensor([[0, 1, 2]])
        weights = torch.tensor([[0.2, 0.3, 0.5]])
        repaired, diagnostics = repair.suppress_contributor_weights(ids, weights, scores)

        self.assertAlmostEqual(float(repaired.sum().item()), 1.0, places=5)
        self.assertLess(float(repaired[0, 2].item()), float(weights[0, 2].item()))
        self.assertGreater(diagnostics["repair_suppressed_count"], 0)


class PseudoQueryManifestTest(unittest.TestCase):
    def test_visualization_resolves_spatial_offset_nearest_train_reference(self):
        from la_artifacts.pseudo_query import PseudoQueryRecord
        from scripts.visualize_pseudo_query_pipeline import _resolve_reference

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train_image = root / "processed" / "seq2" / "frame00184.png"
            train_image.parent.mkdir(parents=True, exist_ok=True)
            train_image.write_bytes(b"fake")
            record = PseudoQueryRecord(
                query_id="synthetic_rgb:synthetic/000000.png",
                scene="ShopFacade",
                source="synthetic_rgb",
                image_name="synthetic/000000.png",
                image_path="",
                pose_w2c=torch.eye(4).tolist(),
                fovx=0.8,
                fovy=0.6,
                width=8,
                height=6,
                nearest_train_image="synthetic_spatial_offset/seq2/frame00184.png__offset2.100__yaw-4.9",
            )

            resolved = _resolve_reference(record, root, images="processed")

            self.assertEqual(resolved, train_image)

    def test_manifest_roundtrip_and_camera_materialization(self):
        from la_artifacts.pseudo_query import PseudoQueryManifest, PseudoQueryRecord
        from PIL import Image

        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "synthetic.png"
            Image.new("RGB", (8, 6), color=(255, 0, 0)).save(image_path)
            record = PseudoQueryRecord(
                query_id="synthetic_rgb:synthetic.png",
                scene="ShopFacade",
                source="synthetic_rgb",
                image_name="synthetic.png",
                image_path=str(image_path),
                pose_w2c=torch.eye(4).tolist(),
                fovx=0.8,
                fovy=0.6,
                width=8,
                height=6,
                accepted=True,
            )
            manifest_path = Path(tmp) / "manifest.jsonl"
            PseudoQueryManifest(1, [record]).save_jsonl(manifest_path)
            loaded = PseudoQueryManifest.load(manifest_path)

            self.assertEqual(loaded.source_counts(), {"synthetic_rgb:accepted": 1})
            camera = loaded.records[0].to_camera(device="cpu")
            self.assertEqual(tuple(camera.original_image.shape), (3, 6, 8))
            self.assertEqual(camera.image_name, "synthetic.png")

    def test_teacher_cache_roundtrip(self):
        from la_artifacts.pseudo_query import PseudoTeacherCache

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cache.pt"
            cache = PseudoTeacherCache({"train_rgb:seq/frame.png": {"inliers": 12}})
            cache.save(path)
            loaded = PseudoTeacherCache.load(path)

            self.assertEqual(loaded.get("train_rgb:seq/frame.png")["inliers"], 12)

    def test_teacher_cache_audit_rewrites_stale_summary_paths_and_checks_coverage(self):
        from la_artifacts.pseudo_query import PseudoQueryManifest, PseudoQueryRecord, PseudoTeacherCache
        from scripts.audit_pseudo_teacher_cache import audit_teacher_cache

        def record(query_id):
            return PseudoQueryRecord(
                query_id=query_id,
                scene="ShopFacade",
                source="train_rgb",
                image_name=query_id.split(":", 1)[1],
                image_path="",
                pose_w2c=torch.eye(4).tolist(),
                fovx=0.8,
                fovy=0.6,
                width=8,
                height=6,
                teacher_cache_key=query_id,
            )

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            manifest_path = tmp / "pseudo_queries.jsonl"
            cache_path = tmp / "pseudo_teacher_cache.pt"
            summary_path = tmp / "pseudo_teacher_cache_summary.json"
            PseudoQueryManifest(
                1,
                [
                    record("train_rgb:ok.png"),
                    record("train_rgb:missing.png"),
                ],
            ).save_jsonl(manifest_path)
            PseudoTeacherCache(
                {
                    "train_rgb:ok.png": {"failure_stage": "teacher_ok"},
                    "train_rgb:extra.png": {"failure_stage": "sparse_failure"},
                }
            ).save(cache_path)
            summary_path.write_text(
                json.dumps(
                    {
                        "manifest": "/old/root/pseudo_queries.jsonl",
                        "output": "/old/root/pseudo_teacher_cache.pt",
                        "sparse_valid_mask": {"enabled": False},
                    }
                )
            )

            summary = audit_teacher_cache(
                manifest_path=manifest_path,
                cache_path=cache_path,
                summary_json=summary_path,
                sources=["train_rgb"],
            )
            written = json.loads(summary_path.read_text())

            self.assertEqual(summary["manifest"], str(manifest_path.resolve()))
            self.assertEqual(summary["output"], str(cache_path.resolve()))
            self.assertEqual(written["manifest"], str(manifest_path.resolve()))
            self.assertEqual(written["output"], str(cache_path.resolve()))
            self.assertEqual(summary["count"], 2)
            self.assertEqual(summary["manifest_count"], 2)
            self.assertEqual(summary["source_counts"], {"train_rgb": 2})
            self.assertEqual(summary["stage_counts"], {"sparse_failure": 1, "teacher_ok": 1})
            self.assertEqual(summary["cache_coverage"]["missing_cache_count"], 1)
            self.assertEqual(summary["cache_coverage"]["extra_cache_count"], 1)
            self.assertEqual(summary["cache_coverage"]["missing_cache_keys"], ["train_rgb:missing.png"])
            self.assertEqual(summary["cache_coverage"]["extra_cache_keys"], ["train_rgb:extra.png"])
            self.assertEqual(summary["sparse_valid_mask"], {"enabled": False})

    def test_teacher_cache_audit_cli_imports_repo_packages_without_pythonpath(self):
        from la_artifacts.pseudo_query import PseudoQueryManifest, PseudoQueryRecord, PseudoTeacherCache

        query_id = "train_rgb:ok.png"
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            manifest_path = tmp / "pseudo_queries.jsonl"
            cache_path = tmp / "pseudo_teacher_cache.pt"
            summary_path = tmp / "pseudo_teacher_cache_summary.json"
            PseudoQueryManifest(
                1,
                [
                    PseudoQueryRecord(
                        query_id=query_id,
                        scene="ShopFacade",
                        source="train_rgb",
                        image_name="ok.png",
                        image_path="",
                        pose_w2c=torch.eye(4).tolist(),
                        fovx=0.8,
                        fovy=0.6,
                        width=8,
                        height=6,
                        teacher_cache_key=query_id,
                    )
                ],
            ).save_jsonl(manifest_path)
            PseudoTeacherCache({query_id: {"failure_stage": "teacher_ok"}}).save(cache_path)

            script = Path(__file__).resolve().parents[1] / "scripts" / "audit_pseudo_teacher_cache.py"
            result = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--manifest",
                    str(manifest_path),
                    "--output",
                    str(cache_path),
                    "--summary_json",
                    str(summary_path),
                    "--sources",
                    "train_rgb",
                ],
                cwd=tmp,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(summary_path.read_text())["count"], 1)

    def test_manifest_filters_records_by_teacher_cache_pose_quality(self):
        from la_artifacts.pseudo_query import PseudoQueryManifest, PseudoQueryRecord, PseudoTeacherCache

        def record(name):
            return PseudoQueryRecord(
                query_id=name,
                scene="ShopFacade",
                source="synthetic_rgb",
                image_name=name.split(":", 1)[1],
                image_path="",
                pose_w2c=torch.eye(4).tolist(),
                fovx=0.8,
                fovy=0.6,
                width=8,
                height=6,
                teacher_cache_key=name,
            )

        manifest = PseudoQueryManifest(1, [record("synthetic_rgb:good.png"), record("synthetic_rgb:bad.png")])
        cache = PseudoTeacherCache(
            {
                "synthetic_rgb:good.png": {"te": 30.0, "dense_te": 20.0, "failure_stage": "teacher_ok"},
                "synthetic_rgb:bad.png": {"te": 3000.0, "dense_te": 2500.0, "failure_stage": "sparse_failure"},
            }
        )

        filtered = manifest.filter_by_teacher_cache(cache, max_sparse_te=100.0, max_dense_te=100.0)

        self.assertEqual([row.query_id for row in filtered.records], ["synthetic_rgb:good.png"])

    def test_synthetic_quality_gate_rejects_bad_local_artifact_metrics(self):
        from la_artifacts.quality_gate import SyntheticQualityGate, SyntheticQualityGateConfig

        gate = SyntheticQualityGate(
            SyntheticQualityGateConfig(
                max_artifact_mean=0.65,
                max_artifact_p95=0.99,
                max_artifact_mild_frac=0.85,
                max_artifact_severe_frac=0.50,
            )
        )

        decision = gate.evaluate_summary(
            {
                "artifact_score_mean": 0.42,
                "artifact_score_p95": 0.97,
                "artifact_mild_frac": 0.70,
                "artifact_severe_frac": 0.61,
            }
        )

        self.assertFalse(decision.accepted)
        self.assertIn("artifact_severe_frac", decision.failures)
        self.assertEqual(decision.reason, "synthetic_quality_rejected")

    def test_teacher_cache_gate_rejects_bad_stage_even_when_pose_error_passes(self):
        from la_artifacts.pseudo_query import PseudoQueryManifest, PseudoQueryRecord, PseudoTeacherCache

        def record(name):
            return PseudoQueryRecord(
                query_id=name,
                scene="ShopFacade",
                source="synthetic_rgb",
                image_name=name.split(":", 1)[1],
                image_path="",
                pose_w2c=torch.eye(4).tolist(),
                fovx=0.8,
                fovy=0.6,
                width=8,
                height=6,
                teacher_cache_key=name,
            )

        manifest = PseudoQueryManifest(
            1,
            [
                record("synthetic_rgb:ok.png"),
                record("synthetic_rgb:mixed.png"),
                record("synthetic_rgb:missing.png"),
            ],
        )
        cache = PseudoTeacherCache(
            {
                "synthetic_rgb:ok.png": {"te": 4.0, "dense_te": 3.0, "failure_stage": "teacher_ok"},
                "synthetic_rgb:mixed.png": {"te": 4.0, "dense_te": 3.0, "failure_stage": "mixed_or_uncertain"},
            }
        )

        filtered, summary = manifest.gate_by_teacher_cache(
            cache,
            max_sparse_te=100.0,
            max_dense_te=100.0,
            allowed_stages=["teacher_ok"],
        )

        self.assertEqual([row.query_id for row in filtered.records], ["synthetic_rgb:ok.png"])
        self.assertEqual(summary["accepted_count"], 1)
        self.assertEqual(summary["rejected_reasons"]["teacher_stage_rejected"], 1)
        self.assertEqual(summary["rejected_reasons"]["missing_teacher_cache"], 1)

    def test_gate_manifest_file_writes_accepted_pool_and_rejection_summary(self):
        from la_artifacts.pseudo_query import PseudoQueryManifest, PseudoQueryRecord, PseudoTeacherCache
        from scripts.gate_pseudo_query_manifest import gate_manifest_file

        def record(name, source, artifact_summary):
            image_name = name.split(":", 1)[1]
            return PseudoQueryRecord(
                query_id=name,
                scene="ShopFacade",
                source=source,
                image_name=image_name,
                image_path="",
                pose_w2c=torch.eye(4).tolist(),
                fovx=0.8,
                fovy=0.6,
                width=8,
                height=6,
                teacher_cache_key=name,
                meta={"artifact_summary": artifact_summary},
            )

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            manifest_path = tmp / "manifest.jsonl"
            cache_path = tmp / "cache.pt"
            output_path = tmp / "gated.jsonl"
            summary_path = tmp / "summary.json"
            manifest = PseudoQueryManifest(
                1,
                [
                    record("train_rgb:seq/frame.png", "train_rgb", {}),
                    record(
                        "synthetic_rgb:synthetic/good.png",
                        "synthetic_rgb",
                        {
                            "artifact_score_mean": 0.45,
                            "artifact_mild_frac": 0.60,
                            "artifact_severe_frac": 0.20,
                            "low_detail_mean": 0.45,
                        },
                    ),
                    record(
                        "synthetic_rgb:synthetic/bad.png",
                        "synthetic_rgb",
                        {
                            "artifact_score_mean": 0.45,
                            "artifact_mild_frac": 0.60,
                            "artifact_severe_frac": 0.75,
                            "low_detail_mean": 0.45,
                        },
                    ),
                ],
            )
            manifest.save_jsonl(manifest_path)
            PseudoTeacherCache(
                {
                    "train_rgb:seq/frame.png": {"te": 3.0, "dense_te": 3.0, "failure_stage": "teacher_ok"},
                    "synthetic_rgb:synthetic/good.png": {"te": 3.0, "dense_te": 3.0, "failure_stage": "teacher_ok"},
                    "synthetic_rgb:synthetic/bad.png": {"te": 3.0, "dense_te": 3.0, "failure_stage": "teacher_ok"},
                }
            ).save(cache_path)

            summary = gate_manifest_file(
                manifest_path,
                output_path,
                summary_json=summary_path,
                teacher_cache_path=cache_path,
                synthetic_qa_max_mean=0.60,
                synthetic_qa_max_mild_frac=0.85,
                synthetic_qa_max_severe_frac=0.58,
                synthetic_qa_max_low_detail_mean=0.60,
                teacher_allowed_stages=["teacher_ok"],
            )

            gated = PseudoQueryManifest.load(output_path)
            self.assertEqual(
                [row.query_id for row in gated.accepted().records],
                ["train_rgb:seq/frame.png", "synthetic_rgb:synthetic/good.png"],
            )
            self.assertEqual(summary["initial_counts"], {"synthetic_rgb:accepted": 2, "train_rgb:accepted": 1})
            self.assertEqual(summary["final_counts"], {"synthetic_rgb:accepted": 1, "synthetic_rgb:rejected": 1, "train_rgb:accepted": 1})
            self.assertTrue(summary_path.exists())

    def test_build_manifest_helper_records_full_synthetic_quality_decision(self):
        from la_artifacts.pseudo_query import PseudoQueryRecord
        from la_artifacts.quality_gate import SyntheticQualityGate, SyntheticQualityGateConfig
        from scripts.build_pseudo_query_manifest import _apply_synthetic_quality_gate

        record = PseudoQueryRecord(
            query_id="synthetic_rgb:synthetic/bad.png",
            scene="ShopFacade",
            source="synthetic_rgb",
            image_name="synthetic/bad.png",
            image_path="",
            pose_w2c=torch.eye(4).tolist(),
            fovx=0.8,
            fovy=0.6,
            width=8,
            height=6,
        )
        summary = {
            "artifact_score_mean": 0.55,
            "artifact_mild_frac": 0.70,
            "artifact_severe_frac": 0.70,
            "low_detail_mean": 0.55,
        }
        gate = SyntheticQualityGate(SyntheticQualityGateConfig(max_artifact_mean=0.60, max_artifact_severe_frac=0.58))

        _apply_synthetic_quality_gate(record, summary, gate)

        self.assertFalse(record.accepted)
        self.assertEqual(record.reason, "synthetic_quality_rejected")
        self.assertEqual(record.meta["artifact_summary"]["artifact_severe_frac"], 0.70)
        self.assertIn("synthetic_quality_gate", record.meta)

    @unittest.skipIf(not torch.cuda.is_available(), "requires CUDA")
    def test_synthetic_manifest_render_uses_rgb_only_when_repair_is_disabled(self):
        from la_artifacts.detector import ArtifactDetector
        from la_artifacts.pseudo_query import PseudoQueryRecord
        from scripts import build_pseudo_query_manifest as builder

        calls = []

        def fake_render(*args, **kwargs):
            calls.append(kwargs.get("rgb_only"))
            return {
                "render": torch.rand(3, 8, 8, device="cuda"),
                "alphas": torch.ones(8, 8, device="cuda"),
            }

        record = PseudoQueryRecord(
            query_id="synthetic_rgb:synthetic/000000.png",
            scene="ShopFacade",
            source="synthetic_rgb",
            image_name="synthetic/000000.png",
            image_path="",
            pose_w2c=torch.eye(4).tolist(),
            fovx=0.8,
            fovy=0.6,
            width=8,
            height=8,
        )

        class Gaussians:
            @property
            def get_xyz(self):
                return torch.zeros(1, 3, device="cuda")

        with tempfile.TemporaryDirectory() as tmp:
            record.image_path = str(Path(tmp) / "000000.png")
            original_render = builder.render_from_pose_gsplat
            try:
                builder.render_from_pose_gsplat = fake_render
                builder._render_synthetic_records(
                    [record],
                    Gaussians(),
                    torch.zeros(3, device="cuda"),
                    ArtifactDetector(),
                    repair=None,
                )
            finally:
                builder.render_from_pose_gsplat = original_render

        self.assertEqual(calls, [True])

    def test_gate_manifest_can_apply_teacher_cache_only_to_selected_sources(self):
        from la_artifacts.pseudo_query import PseudoQueryManifest, PseudoQueryRecord, PseudoTeacherCache
        from scripts.gate_pseudo_query_manifest import gate_manifest_file

        def record(name, source):
            return PseudoQueryRecord(
                query_id=name,
                scene="ShopFacade",
                source=source,
                image_name=name.split(":", 1)[1],
                image_path="",
                pose_w2c=torch.eye(4).tolist(),
                fovx=0.8,
                fovy=0.6,
                width=8,
                height=6,
                teacher_cache_key=name,
                meta={
                    "artifact_summary": {
                        "artifact_score_mean": 0.1,
                        "artifact_mild_frac": 0.0,
                        "artifact_severe_frac": 0.0,
                        "low_detail_mean": 0.1,
                    }
                },
            )

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            manifest_path = tmp / "manifest.jsonl"
            cache_path = tmp / "cache.pt"
            output_path = tmp / "gated.jsonl"
            PseudoQueryManifest(
                1,
                [
                    record("train_rgb:seq/frame.png", "train_rgb"),
                    record("synthetic_rgb:synthetic/frame.png", "synthetic_rgb"),
                ],
            ).save_jsonl(manifest_path)
            PseudoTeacherCache(
                {
                    "synthetic_rgb:synthetic/frame.png": {
                        "te": 3.0,
                        "dense_te": 3.0,
                        "failure_stage": "mixed_or_uncertain",
                    }
                }
            ).save(cache_path)

            summary = gate_manifest_file(
                manifest_path,
                output_path,
                teacher_cache_path=cache_path,
                teacher_gate=True,
                teacher_allowed_stages=["teacher_ok"],
                teacher_gate_sources=["synthetic_rgb"],
            )

            gated = PseudoQueryManifest.load(output_path)
            self.assertEqual([row.query_id for row in gated.accepted().records], ["train_rgb:seq/frame.png"])
            self.assertEqual(summary["teacher_cache_gate"]["rejected_reasons"], {"teacher_stage_rejected": 1})

    def test_gate_manifest_defaults_to_artifact_only_when_teacher_cache_is_present(self):
        from la_artifacts.pseudo_query import PseudoQueryManifest, PseudoQueryRecord, PseudoTeacherCache
        from scripts.gate_pseudo_query_manifest import gate_manifest_file

        record = PseudoQueryRecord(
            query_id="synthetic_rgb:synthetic/frame.png",
            scene="ShopFacade",
            source="synthetic_rgb",
            image_name="synthetic/frame.png",
            image_path="",
            pose_w2c=torch.eye(4).tolist(),
            fovx=0.8,
            fovy=0.6,
            width=8,
            height=6,
            teacher_cache_key="synthetic_rgb:synthetic/frame.png",
            meta={
                "artifact_summary": {
                    "artifact_score_mean": 0.1,
                    "artifact_mild_frac": 0.0,
                    "artifact_severe_frac": 0.0,
                    "low_detail_mean": 0.1,
                }
            },
        )

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            manifest_path = tmp / "manifest.jsonl"
            cache_path = tmp / "cache.pt"
            output_path = tmp / "gated.jsonl"
            PseudoQueryManifest(1, [record]).save_jsonl(manifest_path)
            PseudoTeacherCache(
                {
                    "synthetic_rgb:synthetic/frame.png": {
                        "te": 500.0,
                        "dense_te": 500.0,
                        "failure_stage": "sparse_failure",
                    }
                }
            ).save(cache_path)

            summary = gate_manifest_file(
                manifest_path,
                output_path,
                teacher_cache_path=cache_path,
            )

            gated = PseudoQueryManifest.load(output_path)
            self.assertEqual([row.query_id for row in gated.accepted().records], ["synthetic_rgb:synthetic/frame.png"])
            self.assertFalse(summary["teacher_cache_gate"]["enabled"])
            self.assertEqual(summary["teacher_cache_gate"].get("rejected_reasons", {}), {})

    def test_pseudo_query_sampler_can_sample_record_proportionally_to_pool_sizes(self):
        from la_artifacts.pseudo_query import PseudoQueryRecord, PseudoQuerySampler

        def record(query_id, source):
            return PseudoQueryRecord(
                query_id=query_id,
                scene="ShopFacade",
                source=source,
                image_name=query_id.split(":", 1)[1],
                image_path="",
                pose_w2c=torch.eye(4).tolist(),
                fovx=0.8,
                fovy=0.6,
                width=8,
                height=6,
                teacher_cache_key=query_id,
            )

        rows = [record(f"train_rgb:seq/{idx:05d}.png", "train_rgb") for idx in range(231)]
        rows += [record(f"synthetic_rgb:synthetic/{idx:05d}.png", "synthetic_rgb") for idx in range(5)]

        source_sampler = PseudoQuerySampler(
            rows,
            real_weight=2.0,
            synthetic_weight=1.0,
            seed=7,
            sampling_mode="source_balanced",
        )
        proportional_sampler = PseudoQuerySampler(
            rows,
            real_weight=2.0,
            synthetic_weight=1.0,
            seed=7,
            sampling_mode="record_proportional",
        )
        default_sampler = PseudoQuerySampler(
            rows,
            real_weight=2.0,
            synthetic_weight=1.0,
            seed=7,
        )

        def synthetic_fraction(sampler, draws=2000):
            count = 0
            for _ in range(draws):
                count += 1 if sampler.sample_record().source == "synthetic_rgb" else 0
            return count / float(draws)

        self.assertGreater(synthetic_fraction(source_sampler), 0.25)
        frac = synthetic_fraction(proportional_sampler)
        self.assertGreater(frac, 0.005)
        self.assertLess(frac, 0.03)
        default_frac = synthetic_fraction(default_sampler)
        self.assertGreater(default_frac, 0.005)
        self.assertLess(default_frac, 0.03)

    def test_select_pseudo_query_pool_caps_synthetic_by_artifact_quality_by_default(self):
        from la_artifacts.pseudo_query import PseudoQueryManifest, PseudoQueryRecord, PseudoTeacherCache
        from scripts.select_pseudo_query_pool import select_pseudo_query_pool_file

        def record(name, source, artifact_score=0.0):
            return PseudoQueryRecord(
                query_id=name,
                scene="ShopFacade",
                source=source,
                image_name=name.split(":", 1)[1],
                image_path="",
                pose_w2c=torch.eye(4).tolist(),
                fovx=0.8,
                fovy=0.6,
                width=8,
                height=6,
                teacher_cache_key=name,
                artifact_score=artifact_score,
                meta={
                    "artifact_summary": {
                        "artifact_score_mean": artifact_score,
                        "artifact_mild_frac": artifact_score,
                        "artifact_severe_frac": artifact_score,
                        "low_detail_mean": artifact_score,
                    }
                },
            )

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            manifest_path = tmp / "gated.jsonl"
            cache_path = tmp / "cache.pt"
            output_path = tmp / "selected.jsonl"
            summary_path = tmp / "summary.json"
            PseudoQueryManifest(
                1,
                [
                    record("train_rgb:seq/frame.png", "train_rgb"),
                    record("synthetic_rgb:synthetic/good_b.png", "synthetic_rgb", 0.20),
                    record("synthetic_rgb:synthetic/good_a.png", "synthetic_rgb", 0.10),
                    record("synthetic_rgb:synthetic/extra.png", "synthetic_rgb", 0.05),
                ],
            ).save_jsonl(manifest_path)
            PseudoTeacherCache(
                {
                    "synthetic_rgb:synthetic/good_b.png": {"te": 1.0, "dense_te": 1.0, "failure_stage": "teacher_ok"},
                    "synthetic_rgb:synthetic/good_a.png": {"te": 5.0, "dense_te": 3.0, "failure_stage": "teacher_ok"},
                    "synthetic_rgb:synthetic/extra.png": {"te": 90.0, "dense_te": 90.0, "failure_stage": "sparse_failure"},
                }
            ).save(cache_path)

            summary = select_pseudo_query_pool_file(
                manifest_path,
                output_path,
                summary_json=summary_path,
                teacher_cache_path=cache_path,
                max_synthetic=2,
            )

            selected = PseudoQueryManifest.load(output_path)
            self.assertEqual(
                [row.query_id for row in selected.accepted().records],
                [
                    "train_rgb:seq/frame.png",
                    "synthetic_rgb:synthetic/extra.png",
                    "synthetic_rgb:synthetic/good_a.png",
                ],
            )
            rejected = {row.query_id: row for row in selected.records if not row.accepted}
            self.assertEqual(rejected["synthetic_rgb:synthetic/good_b.png"].reason, "synthetic_pool_not_selected")
            self.assertEqual(summary["synthetic_selected"], 2)
            self.assertEqual(json.loads(summary_path.read_text())["synthetic_rejected_by_cap"], 1)

    def test_select_pseudo_query_pool_can_prioritize_no_reference_support(self):
        from la_artifacts.pseudo_query import PseudoQueryManifest, PseudoQueryRecord, PseudoTeacherCache
        from scripts.select_pseudo_query_pool import select_pseudo_query_pool_file

        def record(name, source):
            return PseudoQueryRecord(
                query_id=name,
                scene="ShopFacade",
                source=source,
                image_name=name.split(":", 1)[1],
                image_path="",
                pose_w2c=torch.eye(4).tolist(),
                fovx=0.8,
                fovy=0.6,
                width=8,
                height=6,
                teacher_cache_key=name,
            )

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            manifest_path = tmp / "manifest.jsonl"
            cache_path = tmp / "cache.pt"
            output_path = tmp / "selected.jsonl"
            PseudoQueryManifest(
                1,
                [
                    record("train_rgb:seq/frame.png", "train_rgb"),
                    record("synthetic_rgb:synthetic/low_support.png", "synthetic_rgb"),
                    record("synthetic_rgb:synthetic/high_support_b.png", "synthetic_rgb"),
                    record("synthetic_rgb:synthetic/high_support_a.png", "synthetic_rgb"),
                ],
            ).save_jsonl(manifest_path)
            PseudoTeacherCache(
                {
                    "synthetic_rgb:synthetic/low_support.png": {
                        "sparse_support_score_prior_score_mean": 0.15,
                        "sparse_valid_mask": {"support_frac": 0.95},
                    },
                    "synthetic_rgb:synthetic/high_support_b.png": {
                        "sparse_support_score_prior_score_mean": 0.80,
                        "sparse_valid_mask": {"support_frac": 0.60},
                    },
                    "synthetic_rgb:synthetic/high_support_a.png": {
                        "sparse_support_score_prior_score_mean": 0.80,
                        "sparse_valid_mask": {"support_frac": 0.90},
                    },
                }
            ).save(cache_path)

            select_pseudo_query_pool_file(
                manifest_path,
                output_path,
                teacher_cache_path=cache_path,
                max_synthetic=2,
                sort_by="support",
            )

            selected = PseudoQueryManifest.load(output_path)
            self.assertEqual(
                [row.query_id for row in selected.accepted().records],
                [
                    "train_rgb:seq/frame.png",
                    "synthetic_rgb:synthetic/high_support_a.png",
                    "synthetic_rgb:synthetic/high_support_b.png",
                ],
            )
            chosen = {
                row.query_id: row.meta["synthetic_pool_selection"]
                for row in selected.accepted().records
                if row.source == "synthetic_rgb"
            }
            self.assertEqual(chosen["synthetic_rgb:synthetic/high_support_a.png"]["support_score_mean"], 0.80)
            self.assertEqual(chosen["synthetic_rgb:synthetic/high_support_a.png"]["support_frac"], 0.90)

    def test_select_pseudo_query_pool_rejects_obviously_low_support_synthetic(self):
        from la_artifacts.pseudo_query import PseudoQueryManifest, PseudoQueryRecord, PseudoTeacherCache
        from scripts.select_pseudo_query_pool import select_pseudo_query_pool_file

        def record(name, source):
            return PseudoQueryRecord(
                query_id=name,
                scene="ShopFacade",
                source=source,
                image_name=name.split(":", 1)[1],
                image_path="",
                pose_w2c=torch.eye(4).tolist(),
                fovx=0.8,
                fovy=0.6,
                width=8,
                height=6,
                teacher_cache_key=name,
            )

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            manifest_path = tmp / "manifest.jsonl"
            cache_path = tmp / "cache.pt"
            output_path = tmp / "selected.jsonl"
            PseudoQueryManifest(
                1,
                [
                    record("train_rgb:seq/frame.png", "train_rgb"),
                    record("synthetic_rgb:synthetic/low_support.png", "synthetic_rgb"),
                    record("synthetic_rgb:synthetic/clean.png", "synthetic_rgb"),
                ],
            ).save_jsonl(manifest_path)
            PseudoTeacherCache(
                {
                    "synthetic_rgb:synthetic/low_support.png": {
                        "sparse_support_score_prior_score_mean": 0.20,
                        "sparse_valid_mask": {"support_frac": 0.30},
                    },
                    "synthetic_rgb:synthetic/clean.png": {
                        "sparse_support_score_prior_score_mean": 0.55,
                        "sparse_valid_mask": {"support_frac": 0.85},
                    },
                }
            ).save(cache_path)

            summary = select_pseudo_query_pool_file(
                manifest_path,
                output_path,
                teacher_cache_path=cache_path,
                max_synthetic=2,
                min_support_frac=0.5,
            )

            selected = PseudoQueryManifest.load(output_path)
            self.assertEqual(
                [row.query_id for row in selected.accepted().records],
                ["train_rgb:seq/frame.png", "synthetic_rgb:synthetic/clean.png"],
            )
            low = next(row for row in selected.records if row.query_id.endswith("low_support.png"))
            self.assertEqual(low.reason, "synthetic_pool_low_support")
            self.assertEqual(summary["synthetic_rejected_by_support"], 1)

    def test_artifact_sort_does_not_use_teacher_pose_error_as_tiebreaker(self):
        from la_artifacts.pseudo_query import PseudoQueryManifest, PseudoQueryRecord, PseudoTeacherCache
        from scripts.select_pseudo_query_pool import select_pseudo_query_pool_file

        def record(name):
            return PseudoQueryRecord(
                query_id=name,
                scene="ShopFacade",
                source="synthetic_rgb",
                image_name=name.split(":", 1)[1],
                image_path="",
                pose_w2c=torch.eye(4).tolist(),
                fovx=0.8,
                fovy=0.6,
                width=8,
                height=6,
                accepted=True,
                artifact_score=0.0,
                teacher_cache_key=name,
            )

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            manifest_path = tmp / "manifest.jsonl"
            cache_path = tmp / "cache.pt"
            output_path = tmp / "selected.jsonl"
            PseudoQueryManifest(
                1,
                [
                    record("synthetic_rgb:synthetic/a.png"),
                    record("synthetic_rgb:synthetic/b.png"),
                ],
            ).save_jsonl(manifest_path)
            PseudoTeacherCache(
                {
                    "synthetic_rgb:synthetic/a.png": {"te": 1000.0, "dense_te": 1000.0},
                    "synthetic_rgb:synthetic/b.png": {"te": 1.0, "dense_te": 1.0},
                }
            ).save(cache_path)

            select_pseudo_query_pool_file(
                manifest_path,
                output_path,
                teacher_cache_path=cache_path,
                max_synthetic=1,
                sort_by="artifact",
            )

            selected = PseudoQueryManifest.load(output_path)
            self.assertEqual(
                [row.query_id for row in selected.accepted().records],
                ["synthetic_rgb:synthetic/a.png"],
            )


class RgbTeacherManifestTest(unittest.TestCase):
    def test_rgb_teacher_manifest_roundtrip(self):
        from la_artifacts.rgb_teacher import RgbTeacherManifest, RgbTeacherSpec

        with tempfile.TemporaryDirectory() as tmp:
            spec = RgbTeacherSpec(
                scene="ShopFacade",
                source_path="/data/ShopFacade",
                backend="wildgaussians",
                checkpoint="",
                output_root=tmp,
                status="planned",
            )
            path = Path(tmp) / "rgb_teacher.json"
            RgbTeacherManifest.single(spec).save(path)
            loaded = RgbTeacherManifest.load(path)

            self.assertEqual(loaded.teachers[0].scene, "ShopFacade")
            self.assertEqual(loaded.teachers[0].backend, "wildgaussians")

    def test_wildgaussians_commands_accept_external_nerfbaselines_binary(self):
        from la_artifacts.rgb_teacher import (
            normalize_render_resolution,
            wildgaussians_render_command_template,
            wildgaussians_train_command,
        )

        train = wildgaussians_train_command(
            "/data/ShopFacade",
            "/out/rgb_teacher",
            "ShopFacade",
            nerfbaselines_bin="/opt/nb/bin/nerfbaselines",
            nerfbaselines_backend="conda",
            train_steps=5,
            logger="none",
            disable_output_artifact=True,
        )
        render = wildgaussians_render_command_template(
            "",
            nerfbaselines_bin="/opt/nb/bin/nerfbaselines",
            nerfbaselines_backend="conda",
        )

        self.assertEqual(train[0], "/opt/nb/bin/nerfbaselines")
        self.assertIn("--backend", train)
        self.assertIn("conda", train)
        self.assertIn("--set", train)
        self.assertIn("iterations=5", train)
        self.assertIn("--disable-output-artifact", train)
        self.assertEqual(render[0], "/opt/nb/bin/nerfbaselines")
        self.assertIn("render-trajectory", render)
        self.assertIn("--output-names", render)
        self.assertIn("{checkpoint}", render)

        render_with_resolution = wildgaussians_render_command_template(
            "/ckpt",
            nerfbaselines_bin="/opt/nb/bin/nerfbaselines",
            nerfbaselines_backend="conda",
            resolution="960x540",
        )
        self.assertIn("--resolution", render_with_resolution)
        self.assertIn("960x540", render_with_resolution)
        self.assertEqual(normalize_render_resolution("960X540"), "960x540")

        with self.assertRaises(ValueError):
            normalize_render_resolution("960")
        with self.assertRaises(ValueError):
            normalize_render_resolution("0x540")

    def test_wildgaussians_train_command_accepts_config_sets(self):
        from la_artifacts.rgb_teacher import wildgaussians_train_command

        train = wildgaussians_train_command(
            "/data/ShopFacade",
            "/out/rgb_teacher",
            "ShopFacade",
            train_steps=15000,
            logger="none",
            config_sets=[
                "appearance_enabled=false",
                "uncertainty_mode=disabled",
                "densify_until_iter=7000",
            ],
        )

        self.assertIn("iterations=15000", train)
        self.assertIn("appearance_enabled=false", train)
        self.assertIn("uncertainty_mode=disabled", train)
        self.assertIn("densify_until_iter=7000", train)

    def test_wildgaussians_train_command_accepts_explicit_output_path(self):
        from la_artifacts.rgb_teacher import wildgaussians_train_command

        train = wildgaussians_train_command(
            "/data/OldHospital",
            "/out/rgb_teacher",
            "OldHospital",
            train_steps=30000,
            output_path="/out/control/OldHospital_wg_noapp_30k",
        )

        output_index = train.index("--output") + 1
        self.assertEqual(train[output_index], "/out/control/OldHospital_wg_noapp_30k")

    def test_rgb_teacher_cambridge_stable_preset_keeps_appearance_and_stopdens(self):
        from scripts.prepare_rgb_teacher_manifest import WILDGAUSSIANS_PRESETS

        stable_sets = WILDGAUSSIANS_PRESETS["cambridge_stable_v1"]["sets"]
        app_dino_sets = WILDGAUSSIANS_PRESETS["cambridge_app_dino_v1"]["sets"]
        legacy_sets = WILDGAUSSIANS_PRESETS["cambridge_legacy_noapp_nounc_v1"]["sets"]
        oldhospital_sets = WILDGAUSSIANS_PRESETS["oldhospital_noapp_nosky_30k_v1"]["sets"]

        self.assertIn("appearance_enabled=true", stable_sets)
        self.assertIn("uncertainty_mode=disabled", stable_sets)
        self.assertIn("densify_until_iter=7000", stable_sets)
        self.assertIn("appearance_enabled=true", app_dino_sets)
        self.assertIn("uncertainty_mode=dino", app_dino_sets)
        self.assertIn("appearance_enabled=false", legacy_sets)
        self.assertIn("uncertainty_mode=disabled", legacy_sets)
        self.assertEqual(WILDGAUSSIANS_PRESETS["oldhospital_noapp_nosky_30k_v1"]["train_steps"], 30000)
        self.assertIn("appearance_enabled=false", oldhospital_sets)
        self.assertIn("uncertainty_mode=disabled", oldhospital_sets)
        self.assertIn("densify_until_iter=7000", oldhospital_sets)
        self.assertNotIn("num_sky_gaussians=50000", oldhospital_sets)

    def test_nerfbaselines_trajectory_from_records_uses_c2w_and_pinhole_intrinsics(self):
        from la_artifacts.rgb_teacher import nerfbaselines_trajectory_from_records

        record = {
            "pose_w2c": [
                [1.0, 0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0, 2.0],
                [0.0, 0.0, 1.0, 3.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            "fovx": 1.0,
            "fovy": 0.8,
            "width": 640,
            "height": 480,
        }
        trajectory = nerfbaselines_trajectory_from_records([record])
        frame = trajectory["frames"][0]

        self.assertEqual(trajectory["format"], "nerfbaselines-v1")
        self.assertEqual(trajectory["camera_model"], "pinhole")
        self.assertEqual(trajectory["image_size"], [640, 480])
        self.assertEqual(len(frame["pose"]), 12)
        self.assertAlmostEqual(frame["pose"][3], -1.0, places=5)
        self.assertAlmostEqual(frame["pose"][7], -2.0, places=5)
        self.assertAlmostEqual(frame["pose"][11], -3.0, places=5)
        self.assertEqual(len(frame["intrinsics"]), 4)
        self.assertEqual(frame["appearance_weights"], [])

    def test_nerfbaselines_trajectory_from_records_can_scale_render_resolution(self):
        from la_artifacts.rgb_teacher import nerfbaselines_trajectory_from_records

        record = {
            "pose_w2c": torch.eye(4).tolist(),
            "fovx": 1.0,
            "fovy": 0.8,
            "width": 640,
            "height": 480,
        }
        base = nerfbaselines_trajectory_from_records([record])
        scaled = nerfbaselines_trajectory_from_records([record], image_scale=0.5)

        self.assertEqual(scaled["image_size"], [320, 240])
        self.assertAlmostEqual(scaled["frames"][0]["intrinsics"][0], base["frames"][0]["intrinsics"][0] * 0.5)
        self.assertAlmostEqual(scaled["frames"][0]["intrinsics"][1], base["frames"][0]["intrinsics"][1] * 0.5)

    def test_nerfbaselines_trajectory_can_use_record_appearance_weights(self):
        from la_artifacts.rgb_teacher import nerfbaselines_trajectory_from_records

        base = {
            "pose_w2c": torch.eye(4).tolist(),
            "fovx": 1.0,
            "fovy": 0.8,
            "width": 640,
            "height": 480,
        }
        records = [
            {
                **base,
                "meta": {
                    "wildgaussians_appearance_train_indices": [3],
                    "wildgaussians_appearance_weights": [1.0],
                },
            },
            {
                **base,
                "meta": {
                    "wildgaussians_appearance_train_indices": [3, 4],
                    "wildgaussians_appearance_weights": [0.25, 0.75],
                },
            },
        ]
        trajectory = nerfbaselines_trajectory_from_records(records, appearance_mode="record")

        self.assertEqual(trajectory["appearances"], [{"embedding_train_index": 3}, {"embedding_train_index": 4}])
        self.assertEqual(trajectory["frames"][0]["appearance_weights"], [1.0, 0.0])
        self.assertEqual(trajectory["frames"][1]["appearance_weights"], [0.25, 0.75])

    def test_resolve_wildgaussians_appearance_mode_auto_uses_checkpoint_config(self):
        from la_artifacts.rgb_teacher import resolve_wildgaussians_appearance_mode

        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp)
            (checkpoint / "config.yaml").write_text("appearance_enabled: false\n")
            self.assertEqual(resolve_wildgaussians_appearance_mode("auto", checkpoint, records=[{}]), "none")

            (checkpoint / "config.yaml").write_text("appearance_enabled: true\n")
            record = {
                "meta": {
                    "wildgaussians_appearance_train_indices": [3],
                    "wildgaussians_appearance_weights": [1.0],
                }
            }
            self.assertEqual(resolve_wildgaussians_appearance_mode("auto", checkpoint, records=[record]), "record")

    def test_apply_wildgaussians_appearance_strategy_can_choose_nearest_endpoint(self):
        from la_artifacts.pseudo_query import PseudoQueryRecord, apply_wildgaussians_appearance_strategy

        record = PseudoQueryRecord(
            query_id="synthetic_rgb:synthetic/000000.png",
            scene="ShopFacade",
            source="synthetic_rgb",
            image_name="synthetic/000000.png",
            image_path="",
            pose_w2c=torch.eye(4).tolist(),
            fovx=0.8,
            fovy=0.6,
            width=8,
            height=6,
            synthetic_alpha=0.75,
            meta={
                "wildgaussians_appearance_train_indices": [10, 11],
                "wildgaussians_appearance_weights": [0.25, 0.75],
            },
        )

        apply_wildgaussians_appearance_strategy([record], "nearest")

        self.assertEqual(record.meta["wildgaussians_appearance_strategy"], "nearest")
        self.assertEqual(record.meta["wildgaussians_appearance_train_indices"], [11])
        self.assertEqual(record.meta["wildgaussians_appearance_weights"], [1.0])

    def test_nerfbaselines_trajectory_rejects_invalid_render_scale(self):
        from la_artifacts.rgb_teacher import nerfbaselines_trajectory_from_records

        record = {
            "pose_w2c": torch.eye(4).tolist(),
            "fovx": 1.0,
            "fovy": 0.8,
            "width": 640,
            "height": 480,
        }
        with self.assertRaises(ValueError):
            nerfbaselines_trajectory_from_records([record], image_scale=0.0)

    def test_wildgaussians_checkpoint_health_flags_nan_critical_tensors(self):
        from la_artifacts.rgb_teacher_health import check_wildgaussians_checkpoint

        with tempfile.TemporaryDirectory() as tmp:
            ckpt = Path(tmp)
            torch.save(
                {
                    "xyz": torch.zeros(2, 3),
                    "features_dc": torch.zeros(2, 3),
                    "features_rest": torch.zeros(2, 45),
                    "scales": torch.zeros(2, 3),
                    "rotations": torch.zeros(2, 4),
                    "embeddings": torch.zeros(2, 24),
                    "appearance_embeddings": torch.full((2, 32), float("nan")),
                    "appearance_mlp.mlp.0.weight": torch.zeros(2, 2),
                    "opacities": torch.zeros(2, 1),
                },
                ckpt / "chkpnt-10.pth",
            )

            health = check_wildgaussians_checkpoint(ckpt)

            self.assertFalse(health.ok)
            self.assertIn("appearance_embeddings", health.reason)

    def test_wildgaussians_checkpoint_health_allows_inf_sky_opacities(self):
        from la_artifacts.rgb_teacher_health import check_wildgaussians_checkpoint

        with tempfile.TemporaryDirectory() as tmp:
            ckpt = Path(tmp)
            opacities = torch.zeros(10, 1)
            opacities[:4] = float("inf")
            torch.save(
                {
                    "xyz": torch.zeros(10, 3),
                    "features_dc": torch.zeros(10, 3),
                    "features_rest": torch.zeros(10, 45),
                    "scales": torch.zeros(10, 3),
                    "rotations": torch.zeros(10, 4),
                    "embeddings": torch.zeros(10, 24),
                    "appearance_embeddings": torch.zeros(2, 32),
                    "appearance_mlp.mlp.0.weight": torch.zeros(2, 2),
                    "opacities": opacities,
                },
                ckpt / "chkpnt-10.pth",
            )

            health = check_wildgaussians_checkpoint(ckpt)

            self.assertTrue(health.ok, health.reason)


class NerfBaselinesDatasetStagingTest(unittest.TestCase):
    def test_downscale_target_size_uses_factor_or_max_width(self):
        from scripts.prepare_nerfbaselines_colmap_dataset import _target_image_size

        self.assertEqual(_target_image_size((1920, 1080), image_downscale_factor=2.0), (960, 540))
        self.assertEqual(_target_image_size((1920, 1080), max_image_width=960), (960, 540))
        self.assertEqual(_target_image_size((640, 480), max_image_width=960), (640, 480))

    def test_downscale_target_size_rejects_conflicting_resize_modes(self):
        from scripts.prepare_nerfbaselines_colmap_dataset import _target_image_size

        with self.assertRaises(ValueError):
            _target_image_size((1920, 1080), image_downscale_factor=2.0, max_image_width=960)


class NerfBaselinesVisualsTest(unittest.TestCase):
    def test_predictions_grid_pairs_gt_and_rendered_images(self):
        import tarfile

        from PIL import Image

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            src = tmp_path / "src"
            (src / "gt-color" / "seq1").mkdir(parents=True)
            (src / "color" / "seq1").mkdir(parents=True)
            Image.new("RGB", (8, 4), color=(255, 0, 0)).save(src / "gt-color" / "seq1" / "frame00001.png")
            Image.new("RGB", (8, 4), color=(0, 0, 255)).save(src / "color" / "seq1" / "frame00001.png")

            tar_path = tmp_path / "predictions.tar.gz"
            with tarfile.open(tar_path, "w:gz") as tar:
                tar.add(src / "gt-color" / "seq1" / "frame00001.png", arcname="gt-color/seq1/frame00001.png")
                tar.add(src / "color" / "seq1" / "frame00001.png", arcname="color/seq1/frame00001.png")

            from la_artifacts.nerfbaselines_visuals import build_predictions_grid

            out_path = tmp_path / "grid.png"
            summary = build_predictions_grid(tar_path, out_path, sample_count=1, columns=1)

            self.assertEqual(summary["pairs"], 1)
            self.assertTrue(out_path.exists())
            with Image.open(out_path) as grid:
                self.assertEqual(grid.size, (16, 4))


if __name__ == "__main__":
    unittest.main()
