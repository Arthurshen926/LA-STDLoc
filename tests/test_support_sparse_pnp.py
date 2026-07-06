import unittest
from types import SimpleNamespace

import numpy as np
import torch


class SupportSparsePnpTest(unittest.TestCase):
    def test_support_mask_prioritizes_structured_keypoints_without_selector(self):
        from scripts.evaluate_support_sparse_pnp import select_support_keypoints

        # Candidate ids are already sorted by detector score. The support mask
        # covers the right half of the feature grid, so right-half ids should be
        # promoted without using the artifact valid-mask selector.
        support = torch.zeros(4, 4, dtype=torch.bool)
        support[:, 2:] = True
        candidates = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7])

        selected, metrics = select_support_keypoints(
            candidates,
            support,
            height=4,
            width=4,
            target_count=4,
            refill=True,
        )

        torch.testing.assert_close(selected, torch.tensor([2, 3, 6, 7]))
        self.assertEqual(metrics["support_selected_keypoints"], 4)
        self.assertEqual(metrics["support_refill_keypoints"], 0)
        self.assertAlmostEqual(metrics["support_mask_frac"], 0.5)

    def test_support_score_prior_softly_reranks_keypoint_scores(self):
        from stdloc import apply_sparse_support_score_prior

        scores = torch.tensor([[0.90, 0.80], [0.70, 0.60]])
        support_score = torch.tensor([[0.0, 0.0], [1.0, 1.0]])

        adjusted, metrics = apply_sparse_support_score_prior(
            scores,
            support_score,
            weight=1.0,
            min_multiplier=0.5,
        )

        self.assertLess(adjusted[0, 0].item(), adjusted[1, 0].item())
        self.assertGreater(adjusted[0, 1].item(), 0.0)
        self.assertAlmostEqual(metrics["sparse_support_score_prior_weight"], 1.0)
        self.assertAlmostEqual(metrics["sparse_support_score_prior_multiplier_mean"], 1.25)

    def test_support_score_prior_accepts_singleton_score_channel(self):
        from stdloc import apply_sparse_support_score_prior

        scores = torch.ones(1, 2, 2)
        support_score = torch.tensor([[0.0, 1.0], [0.0, 1.0]])

        adjusted, _ = apply_sparse_support_score_prior(
            scores,
            support_score,
            weight=0.5,
            min_multiplier=0.75,
        )

        self.assertEqual(tuple(adjusted.shape), (1, 2, 2))
        self.assertLess(adjusted[0, 0, 0].item(), adjusted[0, 0, 1].item())

    def test_dense_query_valid_mask_suppresses_invalid_query_cells(self):
        from stdloc import apply_dense_query_valid_mask_to_corr

        corr = torch.arange(16, dtype=torch.float32).reshape(1, 4, 4)
        valid_mask = torch.zeros(4, 4, dtype=torch.bool)
        valid_mask[:2, :2] = True

        masked, metrics = apply_dense_query_valid_mask_to_corr(
            corr,
            valid_mask,
            query_height=2,
            query_width=2,
            min_fraction=0.5,
        )

        torch.testing.assert_close(masked[:, 0, :], corr[:, 0, :])
        self.assertTrue(torch.all(masked[:, 1:, :] < -1e8))
        self.assertEqual(metrics["dense_valid_mask_valid_cells"], 1)
        self.assertAlmostEqual(metrics["dense_valid_mask_valid_frac"], 0.25)

    def test_stdloc_localize_passes_sparse_guidance_to_dense_stage(self):
        from stdloc import STDLoc

        loc = object.__new__(STDLoc)
        loc.config = {"sparse": {"sparse_only": False}, "dense": {"iters": 1}}
        loc.feature_extractor = SimpleNamespace()
        loc.get_feature_map = lambda query_image: (
                torch.zeros(1, 2, 2),
                torch.zeros(1, 4, 4),
        )
        seen = {}

        def fake_sparse(query_feature_map, fovx, fovy, valid_mask=None, support_score=None):
            return {"pose_w2c": np.eye(4, dtype=np.float32), "inliers": 8}

        def fake_dense(coarse, fine, pose_w2c, fovx, fovy, valid_mask=None, support_score=None):
            seen["valid_mask"] = valid_mask
            seen["support_score"] = support_score
            return {"pose_w2c": pose_w2c, "inliers": 6}

        loc.loc_sparse = fake_sparse
        loc.loc_dense = fake_dense
        valid_mask = torch.ones(4, 4, dtype=torch.bool)
        support_score = torch.ones(4, 4)

        loc.localize(
            torch.zeros(3, 4, 4),
            0.8,
            0.6,
            sparse_valid_mask=valid_mask,
            sparse_support_score=support_score,
        )

        self.assertIs(seen["valid_mask"], valid_mask)
        self.assertIs(seen["support_score"], support_score)

    def test_teacher_cache_support_mask_score_mode_uses_binary_support_and_soft_score(self):
        from scripts.build_pseudo_teacher_cache import _build_sparse_guidance_for_record

        record = SimpleNamespace(source="synthetic_rgb", query_id="synthetic_rgb:toy.png")
        image = torch.full((3, 32, 32), 0.5)
        image[:, :, 14:18] = 0.95

        valid_mask, support_score, summary = _build_sparse_guidance_for_record(
            record,
            image,
            enabled=True,
            allowed_sources={"synthetic_rgb"},
            mode="support_mask_score",
            no_reference_image_scale=1.0,
            no_reference_support_threshold=0.2,
            no_reference_support_dilate_radius=1,
            no_reference_support_min_area=4,
            no_reference_invalid_min_area=16,
        )

        self.assertEqual(summary["mode"], "support_mask_score")
        self.assertIsNotNone(support_score)
        self.assertEqual(tuple(valid_mask.shape), (32, 32))
        self.assertEqual(tuple(support_score.shape), (32, 32))
        self.assertGreater(valid_mask[:, 13:19].float().mean().item(), 0.5)
        self.assertLess(valid_mask[:, :6].float().mean().item(), 0.2)
        self.assertGreater(summary["valid_frac"], 0.95)
        self.assertGreater(summary["support_frac"], 0.0)

    def test_support_ablation_summary_reports_sparse_deltas(self):
        from scripts.evaluate_support_sparse_pnp import summarize_support_ablation

        rows = [
            {
                "baseline": {"inliers": 10, "matches": 50, "sparse_te": 100.0, "sparse_ae": 4.0, "failed": False},
                "support": {"inliers": 12, "matches": 48, "sparse_te": 80.0, "sparse_ae": 3.0, "failed": False},
            },
            {
                "baseline": {"inliers": 0, "matches": 0, "sparse_te": 9999.0, "sparse_ae": 180.0, "failed": True},
                "support": {"inliers": 8, "matches": 40, "sparse_te": 200.0, "sparse_ae": 5.0, "failed": False},
            },
        ]

        summary = summarize_support_ablation(rows)

        self.assertEqual(summary["count"], 2)
        self.assertEqual(summary["support_wins_te"], 2)
        self.assertEqual(summary["support_rescues"], 1)
        self.assertAlmostEqual(summary["baseline_avg_inliers"], 5.0)
        self.assertAlmostEqual(summary["support_avg_inliers"], 10.0)
        self.assertAlmostEqual(summary["delta_avg_inliers"], 5.0)


if __name__ == "__main__":
    unittest.main()
