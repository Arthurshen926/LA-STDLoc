import json
import tempfile
import unittest
from pathlib import Path

import torch

from la_artifacts.pseudo_query import PseudoQueryManifest, PseudoQueryRecord
from la_artifacts.pseudo_query_training import pseudo_query_reliability_decision


class PseudoQueryABTests(unittest.TestCase):
    def _record(self, query_id, source, image_name, image_path, tx=0.0):
        pose = torch.eye(4)
        pose[0, 3] = float(tx)
        return PseudoQueryRecord(
            query_id=query_id,
            scene="ShopFacade",
            source=source,
            image_name=image_name,
            image_path=image_path,
            pose_w2c=pose.tolist(),
            fovx=0.8,
            fovy=0.6,
            width=640,
            height=360,
            teacher_cache_key=query_id,
        )

    def test_backend_render_manifest_rewrites_only_synthetic_paths_and_preserves_pose(self):
        from scripts.render_pseudo_query_manifest import prepare_records_for_backend

        train = self._record(
            "train_rgb:seq1/frame00001.png",
            "train_rgb",
            "seq1/frame00001.png",
            "/data/seq1/frame00001.png",
        )
        synthetic = self._record(
            "synthetic_rgb:synthetic/000000.png",
            "synthetic_rgb",
            "synthetic/000000.png",
            "/old/synthetic/000000.png",
            tx=1.25,
        )

        all_records, render_records = prepare_records_for_backend(
            [train, synthetic],
            synthetic_image_root="/tmp/backend_synthetic",
        )

        self.assertEqual(len(all_records), 2)
        self.assertEqual(len(render_records), 1)
        self.assertEqual(all_records[0].image_path, "/data/seq1/frame00001.png")
        self.assertEqual(render_records[0].image_path, "/tmp/backend_synthetic/synthetic/000000.png")
        self.assertEqual(render_records[0].pose_w2c, synthetic.pose_w2c)
        self.assertEqual(render_records[0].query_id, synthetic.query_id)
        self.assertEqual(render_records[0].teacher_cache_key, synthetic.teacher_cache_key)

    def test_ab_compare_flags_pose_mismatch_and_summarizes_deltas(self):
        from scripts.compare_pseudo_query_backends import compare_backend_runs

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            key = "synthetic_rgb:synthetic/000000.png"
            left_record = self._record(key, "synthetic_rgb", "synthetic/000000.png", str(tmp / "left.png"), tx=0.0)
            right_record = self._record(key, "synthetic_rgb", "synthetic/000000.png", str(tmp / "right.png"), tx=0.2)
            left_manifest = tmp / "left.jsonl"
            right_manifest = tmp / "right.jsonl"
            PseudoQueryManifest(version=1, records=[left_record]).save_jsonl(left_manifest)
            PseudoQueryManifest(version=1, records=[right_record]).save_jsonl(right_manifest)

            left_cache = tmp / "left.pt"
            right_cache = tmp / "right.pt"
            torch.save(
                {
                    "items": {
                        key: {
                            "failure_stage": "sparse_failure",
                            "inliers": 10,
                            "dense_inliers": 20,
                            "te": 30.0,
                            "dense_te": 40.0,
                            "ae": 1.0,
                            "dense_ae": 2.0,
                            "failed": True,
                        }
                    }
                },
                left_cache,
            )
            torch.save(
                {
                    "items": {
                        key: {
                            "failure_stage": "teacher_ok",
                            "inliers": 15,
                            "dense_inliers": 35,
                            "te": 20.0,
                            "dense_te": 25.0,
                            "ae": 0.5,
                            "dense_ae": 1.5,
                            "failed": False,
                        }
                    }
                },
                right_cache,
            )

            summary = compare_backend_runs(
                left_label="wg",
                left_manifest=left_manifest,
                left_cache=left_cache,
                right_label="matcha",
                right_manifest=right_manifest,
                right_cache=right_cache,
                source="synthetic_rgb",
            )

        self.assertFalse(summary["pose_alignment"]["same_pose_all"])
        self.assertAlmostEqual(summary["pose_alignment"]["max_pose_abs_diff"], 0.2, places=6)
        self.assertEqual(summary["backends"]["wg"]["stage_counts"], {"sparse_failure": 1})
        self.assertEqual(summary["backends"]["matcha"]["stage_counts"], {"teacher_ok": 1})
        deltas = summary["pairwise_delta_right_minus_left"]
        self.assertEqual(deltas["inliers"]["mean"], 5.0)
        self.assertEqual(deltas["dense_te"]["mean"], -15.0)
        top_examples = summary["pairwise_top_examples"]
        self.assertEqual(top_examples["right_te_improves_most"][0]["key"], key)
        self.assertEqual(top_examples["right_inliers_improves_most"][0]["delta"], 5.0)

    def test_soft_reliability_can_keep_stats_updates_below_memory_threshold(self):
        from argparse import Namespace

        record = self._record(
            "train_rgb:seq1/frame00001.png",
            "train_rgb",
            "seq1/frame00001.png",
            "/data/seq1/frame00001.png",
        )
        item = {
            "source": "train_rgb",
            "failure_stage": "sparse_failure",
            "te": 30.0,
            "dense_te": 40.0,
            "inliers": 10,
        }
        stats = {
            "__global__": {"median_final_te": 10.0, "median_inliers": 100.0},
            "train_rgb": {"median_final_te": 10.0, "median_inliers": 100.0},
        }
        args = Namespace(
            pseudo_query_reliability_mode="soft",
            pseudo_query_reliability_teacher_ok_weight=1.0,
            pseudo_query_reliability_dense_improves_weight=0.95,
            pseudo_query_reliability_mixed_weight=0.60,
            pseudo_query_reliability_dense_rescues_weight=0.70,
            pseudo_query_reliability_sparse_failure_weight=0.25,
            pseudo_query_reliability_dense_regression_weight=0.25,
            pseudo_query_reliability_unknown_weight=0.50,
            pseudo_query_reliability_error_scale=1.5,
            pseudo_query_reliability_inlier_power=0.75,
            pseudo_query_reliability_min_weight=0.20,
            pseudo_query_reliability_real_min_weight=0.45,
            pseudo_query_reliability_synthetic_min_weight=0.25,
            pseudo_query_reliability_memory_min_weight=0.80,
            pseudo_query_reliability_stats_min_weight=0.0,
        )

        decision = pseudo_query_reliability_decision(record, item, stats, args)

        self.assertLess(decision["weight"], args.pseudo_query_reliability_memory_min_weight)
        self.assertFalse(decision["update_memory"])
        self.assertTrue(decision["update_stats"])

    def test_stage_direct_soft_stats_policy_keeps_sparse_failure_stats(self):
        from argparse import Namespace
        from train_locaware import _pseudo_query_stage_direct_loss_policy

        args = Namespace(
            pseudo_query_stage_objective_mode="direct",
            pseudo_query_stage_stats_policy="soft",
        )
        reliability = {
            "stage": "sparse_failure",
            "update_memory": False,
            "update_stats": True,
        }

        policy = _pseudo_query_stage_direct_loss_policy(reliability, args)

        self.assertFalse(policy["update_memory"])
        self.assertTrue(policy["update_stats"])


if __name__ == "__main__":
    unittest.main()
