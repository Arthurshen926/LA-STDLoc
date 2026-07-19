import json
import tempfile
import unittest
from pathlib import Path


class SelectLaFGSCleanInitializationTest(unittest.TestCase):
    def _report(self, tag, te, ae, protocol="protocol-a"):
        record = {
            "tag": tag,
            "state": f"/tmp/{tag}.pt",
            "results_summary": f"/tmp/{tag}.json",
            "evaluation_camera_subset": "candidate_validation",
            "evaluation_camera_count": 20,
            "evaluation_protocol_sha256": protocol,
            "metrics": {
                "median_te_cm": te,
                "median_ae_deg": ae,
                "raw_gt_precision_2px": 0.2,
                "inlier_gt_precision_2px": 0.8,
                "translation_pose_info_logdet": 12.0,
            },
        }
        return {
            "selection_protocol": {"test_metrics_used": False},
            "control": record,
            "candidates": [],
            "selected_tag": tag,
            "selected_state": record["state"],
        }

    def test_selects_lowest_validation_translation_error(self):
        from scripts.select_lafgs_clean_initialization import (
            load_initialization_candidate,
            select_initialization,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            medoid = root / "medoid.json"
            mean = root / "mean.json"
            medoid.write_text(json.dumps(self._report("medoid_control", 4.0, 0.2)))
            mean.write_text(json.dumps(self._report("mean_control", 3.0, 0.3)))
            selected = select_initialization(
                [
                    load_initialization_candidate("medoid", medoid),
                    load_initialization_candidate("mean", mean),
                ]
            )
        self.assertEqual(selected["selected_initialization"], "mean")
        self.assertFalse(selected["selection_protocol"]["test_metrics_used"])

    def test_rejects_mismatched_validation_protocols(self):
        from scripts.select_lafgs_clean_initialization import (
            load_initialization_candidate,
            select_initialization,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "first.json"
            second = root / "second.json"
            first.write_text(json.dumps(self._report("first", 4.0, 0.2)))
            second.write_text(json.dumps(self._report("second", 3.0, 0.3, "protocol-b")))
            with self.assertRaisesRegex(ValueError, "different validation protocols"):
                select_initialization(
                    [
                        load_initialization_candidate("first", first),
                        load_initialization_candidate("second", second),
                    ]
                )

    def test_uses_the_inner_primary_score_for_global_selection(self):
        from scripts.select_lafgs_clean_initialization import (
            load_initialization_candidate,
            select_initialization,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            mean = root / "mean.json"
            medoid = root / "medoid.json"
            mean_payload = self._report("mean", 3.0, 0.2)
            medoid_payload = self._report("medoid", 2.9, 0.3)
            for payload, score in ((mean_payload, 2.0), (medoid_payload, 2.1)):
                payload["selection_protocol"].update(
                    {
                        "selection_mode": "performance",
                        "primary_metric": "median_te_cm + mean_te_weight * mean_te_cm",
                        "mean_te_weight": 0.05,
                        "max_recall_2m_drop": 0.01,
                        "max_recall_5cm_drop": 0.01,
                    }
                )
                payload["control"]["primary_score"] = score
            mean.write_text(json.dumps(mean_payload))
            medoid.write_text(json.dumps(medoid_payload))
            selected = select_initialization(
                [
                    load_initialization_candidate("mean", mean),
                    load_initialization_candidate("medoid", medoid),
                ]
            )
        self.assertEqual(selected["selected_initialization"], "mean")
        self.assertEqual(
            selected["selection_protocol"]["checkpoint_selection_mode"],
            "performance",
        )
