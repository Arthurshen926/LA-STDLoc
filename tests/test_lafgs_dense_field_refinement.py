import unittest

import torch


class DenseFieldRefinementTest(unittest.TestCase):
    def test_sparse_initialization_only_overwrites_shared_support(self):
        from scripts.train_lafgs_dense_field_refinement import (
            _merge_sparse_initialization,
        )

        support = {
            "indices": torch.tensor([1, 3, 5]),
            "features": torch.eye(3, dtype=torch.float32),
        }
        sparse = {
            "indices": torch.tensor([3, 4]),
            "features": torch.tensor(
                [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float32
            ),
        }

        merged, count = _merge_sparse_initialization(support, sparse, point_count=8)

        self.assertEqual(count, 1)
        self.assertTrue(torch.allclose(merged[0], support["features"][0]))
        self.assertTrue(torch.allclose(merged[1], sparse["features"][0]))

    def test_curriculum_interpolates_and_clamps(self):
        from types import SimpleNamespace
        from scripts.train_lafgs_dense_field_refinement import _curriculum

        args = SimpleNamespace(
            jitter_warmup_steps=10,
            jitter_translation_start_m=0.01,
            jitter_translation_end_m=0.11,
            jitter_rotation_start_deg=0.1,
            jitter_rotation_end_deg=1.1,
        )

        self.assertEqual(_curriculum(0, args), (0.01, 0.1))
        middle = _curriculum(5, args)
        self.assertAlmostEqual(middle[0], 0.06)
        self.assertAlmostEqual(middle[1], 0.6)
        self.assertEqual(_curriculum(20, args), (0.11, 1.1))

    def test_sparse_seed_loader_rejects_holdout_and_filters_basin(self):
        import json
        import tempfile
        from pathlib import Path

        from scripts.train_lafgs_dense_field_refinement import (
            _filter_sparse_seed_poses,
            _load_sparse_seed_poses,
        )

        records = [
            {
                "image_name": "train.png",
                "sparse": {"pose_w2c": torch.eye(4).tolist()},
            }
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "results.json"
            path.write_text(json.dumps(records))
            seeds, manifest = _load_sparse_seed_poses(
                path,
                allowed_names={"train.png"},
                excluded_names={"heldout.png"},
            )

        self.assertEqual(set(seeds), {"train.png"})
        self.assertEqual(manifest["train_seed_count"], 1)
        cache = {"train.png": {"pose_w2c": torch.eye(4)}}
        filtered, stats = _filter_sparse_seed_poses(
            seeds,
            cache,
            max_translation_m=0.01,
            max_rotation_deg=0.01,
        )
        self.assertEqual(set(filtered), {"train.png"})
        self.assertEqual(stats["eligible_count"], 1)
