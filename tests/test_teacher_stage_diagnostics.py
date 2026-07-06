import csv
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from la_diagnostics.teacher_stage import (
    build_teacher_stage_records,
    classify_teacher_stage,
    selected_image_names_from_sample_flow,
    summarize_stage_records,
)


class TeacherStageDiagnosticsTest(unittest.TestCase):
    def test_classifies_sparse_failure_before_dense(self):
        stage = classify_teacher_stage(
            sparse_te=120.0,
            dense_te=118.0,
            sparse_inliers=120,
            sparse_correct_rate=0.01,
        )
        self.assertEqual(stage, "sparse_failure")

    def test_classifies_dense_regression_after_good_sparse(self):
        stage = classify_teacher_stage(
            sparse_te=3.0,
            dense_te=30.0,
            sparse_inliers=120,
            sparse_correct_rate=0.4,
        )
        self.assertEqual(stage, "dense_regression_after_good_sparse")

    def test_build_records_merges_sparse_match_diagnostics(self):
        results = [
            {
                "image_name": "seq1/frame00001.png",
                "sparse_TE": 3.0,
                "sparse_AE": 0.2,
                "dense_TE": 30.0,
                "dense_AE": 1.5,
                "sparse": {"inliers": 100},
                "dense": [{"inliers": 20}],
            }
        ]
        sparse_diag = {
            "seq1/frame00001.png": {
                "matches": 50,
                "correct_matches": 25,
                "inliers": 30,
            }
        }

        records = build_teacher_stage_records(results, sparse_diag)

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["failure_stage"], "dense_regression_after_good_sparse")
        self.assertEqual(records[0]["sparse_correct_rate"], 0.5)
        self.assertEqual(records[0]["dense_delta_te"], 27.0)

    def test_summary_counts_failure_stages(self):
        records = [
            {"failure_stage": "sparse_failure", "sparse_te": 120.0, "dense_te": 118.0, "dense_delta_te": -2.0},
            {
                "failure_stage": "dense_regression_after_good_sparse",
                "sparse_te": 3.0,
                "dense_te": 30.0,
                "dense_delta_te": 27.0,
            },
        ]

        summary = summarize_stage_records(records)

        self.assertEqual(summary["count"], 2)
        self.assertEqual(summary["failure_stage_counts"]["sparse_failure"], 1)
        self.assertEqual(summary["failure_stage_counts"]["dense_regression_after_good_sparse"], 1)

    def test_selects_image_names_from_sample_flow_groups(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample_flow.csv"
            with path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["group", "image_name"])
                writer.writeheader()
                writer.writerow({"group": "final_worst", "image_name": "seq8/frame00032.png"})
                writer.writerow({"group": "artifact_severe", "image_name": "seq1/frame00175.png"})
                writer.writerow({"group": "final_regressed", "image_name": "seq8/frame00031.png"})
                writer.writerow({"group": "final_worst", "image_name": "seq8/frame00032.png"})

            names = selected_image_names_from_sample_flow(path, ["final_worst", "final_regressed"])

            self.assertEqual(names, ["seq8/frame00032.png", "seq8/frame00031.png"])

    def test_stage_script_reads_missing_optional_args_like_cfg_merge(self):
        from scripts.diagnose_stdloc_teacher_stages import _read_image_names

        args = Namespace(
            image_names=["seq8/frame00032.png"],
            sample_flow_csv=None,
            sample_flow_groups=["final_worst"],
            max_images=8,
        )

        self.assertEqual(_read_image_names(args), ["seq8/frame00032.png"])

    def test_stage_script_jsonable_result_handles_nested_arrays(self):
        import numpy as np

        from scripts.diagnose_stdloc_teacher_stages import _jsonable_result

        row = {
            "image_name": "seq8/frame00032.png",
            "sparse": {"pose_w2c": np.eye(4), "inliers": np.int64(10)},
            "dense": [{"pose_w2c": np.eye(4) * 2, "inliers": np.int64(5)}],
        }

        converted = _jsonable_result(row)

        json.dumps(converted)
        self.assertEqual(converted["sparse"]["pose_w2c"][0][0], 1.0)
        self.assertEqual(converted["dense"][0]["pose_w2c"][0][0], 2.0)
        self.assertEqual(converted["sparse"]["inliers"], 10)


if __name__ == "__main__":
    unittest.main()
