import importlib.util
from pathlib import Path
import unittest
from unittest import mock

import torch
import torch.nn.functional as F


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "eval_lafgs_dense_refinement.py"
SPEC = importlib.util.spec_from_file_location("dense_eval", MODULE_PATH)
dense_eval = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(dense_eval)


class LocalDenseMatchingTest(unittest.TestCase):
    def test_prior_rgb_encoder_resolution_uses_original_image(self):
        self.assertEqual(
            dense_eval.prior_rgb_encoder_resolution((360, 640), 45, 80),
            (360, 640),
        )
        self.assertEqual(
            dense_eval.prior_rgb_encoder_resolution(None, 45, 80),
            (45, 80),
        )

    def test_prior_rgb_uses_high_res_encoder_and_feature_grid_geometry(self):
        calls = []

        def fake_render(_, __, ___, ____, width, height, **kwargs):
            calls.append((width, height, kwargs.get("rgb_only")))
            return {
                "render": torch.ones(3, height, width),
                "depth": torch.ones(1, height, width),
                "rend_alpha": torch.ones(1, height, width),
            }

        class IdentityFeatureExtractor:
            def __call__(self, image):
                return {"feature_map": image}

        with mock.patch.object(dense_eval, "render_from_pose_gsplat", fake_render):
            result = dense_eval._render_representation(
                None,
                IdentityFeatureExtractor(),
                "prior_rgb",
                torch.eye(4),
                1.0,
                1.0,
                45,
                80,
                prior_rgb_source_image_size=(360, 640),
            )

        rendered_features, depth, *_ = result
        self.assertEqual(calls, [(640, 360, True), (80, 45, True)])
        self.assertEqual(tuple(rendered_features.shape), (3, 45, 80))
        self.assertEqual(tuple(depth.shape), (45, 80))

    def test_prior_rgb_ulfloc_schedule_renders_at_feature_grid_then_encodes_source_size(self):
        render_calls = []
        encoder_inputs = []

        def fake_render(_, __, ___, ____, width, height, **kwargs):
            render_calls.append((width, height, kwargs.get("rgb_only")))
            return {
                "render": torch.ones(3, height, width),
                "depth": torch.ones(1, height, width),
                "rend_alpha": torch.ones(1, height, width),
            }

        class CapturingFeatureExtractor:
            def __call__(self, image):
                encoder_inputs.append(tuple(image.shape))
                return {"feature_map": image}

        with mock.patch.object(dense_eval, "render_from_pose_gsplat", fake_render):
            result = dense_eval._render_representation(
                None,
                CapturingFeatureExtractor(),
                "prior_rgb",
                torch.eye(4),
                1.0,
                1.0,
                45,
                80,
                prior_rgb_source_image_size=(360, 640),
                prior_rgb_render_resolution="feature",
            )

        rendered_features, depth, *_ = result
        self.assertEqual(render_calls, [(80, 45, True), (80, 45, True)])
        self.assertEqual(encoder_inputs, [(1, 3, 360, 640)])
        self.assertEqual(tuple(rendered_features.shape), (3, 45, 80))
        self.assertEqual(tuple(depth.shape), (45, 80))

    def test_local_matching_recovers_a_known_pixel_shift(self):
        generator = torch.Generator().manual_seed(7)
        channels, height, width = 64, 16, 20
        query = F.normalize(
            torch.randn(channels, height, width, generator=generator), p=2, dim=0
        )
        rendered = torch.zeros_like(query)
        valid = torch.zeros(height, width, dtype=torch.bool)
        shift_x, shift_y = 2, -1
        for y in range(2, height - 2):
            for x in range(2, width - 3):
                rendered[:, y, x] = query[:, y + shift_y, x + shift_x]
                valid[y, x] = True

        matches, diagnostics = dense_eval.build_local_dense_matches(
            query,
            rendered,
            valid,
            radius_px=3,
            anchor_stride=2,
            temperature=0.07,
            batch_size=32,
            min_similarity=-1.0,
            max_dense_matches=0,
            correspondence_mode="hard",
        )
        self.assertIsNotNone(matches)
        query_xy, rendered_xy, _ = matches
        self.assertGreaterEqual(diagnostics["fine_matches"], 4)
        expected = torch.tensor([shift_x, shift_y], dtype=query_xy.dtype)
        self.assertTrue(torch.all(query_xy - rendered_xy == expected))

    def test_soft_local_matching_preserves_subpixel_expected_shift(self):
        generator = torch.Generator().manual_seed(11)
        channels, height, width = 64, 12, 16
        query = F.normalize(
            torch.randn(channels, height, width, generator=generator), p=2, dim=0
        )
        rendered = torch.zeros_like(query)
        valid = torch.zeros(height, width, dtype=torch.bool)
        for y in range(2, height - 2):
            for x in range(2, width - 2):
                rendered[:, y, x] = query[:, y, x]
                valid[y, x] = True
        matches, _ = dense_eval.build_local_dense_matches(
            query,
            rendered,
            valid,
            radius_px=1,
            anchor_stride=1,
            temperature=0.01,
            batch_size=32,
            min_similarity=-1.0,
            max_dense_matches=0,
            correspondence_mode="soft",
        )
        query_xy, rendered_xy, _ = matches
        self.assertLess(float((query_xy - rendered_xy).abs().max()), 1e-3)

    def test_local_matching_excludes_invalid_query_window_cells(self):
        generator = torch.Generator().manual_seed(19)
        channels, height, width = 32, 10, 12
        query = F.normalize(
            torch.randn(channels, height, width, generator=generator), p=2, dim=0
        )
        rendered = query.clone()
        rendered_valid = torch.ones(height, width, dtype=torch.bool)
        query_valid = torch.zeros(height, width, dtype=torch.bool)
        query_valid[:, : width // 2] = True

        matches, _ = dense_eval.build_local_dense_matches(
            query,
            rendered,
            rendered_valid,
            radius_px=0,
            anchor_stride=1,
            temperature=0.07,
            batch_size=64,
            min_similarity=-1.0,
            max_dense_matches=0,
            correspondence_mode="hard",
            query_valid=query_valid,
        )

        self.assertIsNotNone(matches)
        query_xy, _, _ = matches
        self.assertTrue(torch.all(query_xy[:, 0] < width // 2))

    def test_single_valid_candidate_has_finite_zero_confidence(self):
        """A masked local window must not become an infinite-margin match."""
        query = F.normalize(torch.ones(8, 3, 3), p=2, dim=0)
        rendered = query.clone()
        rendered_valid = torch.ones(3, 3, dtype=torch.bool)
        query_valid = torch.zeros(3, 3, dtype=torch.bool)
        query_valid[1, 1] = True

        matches, diagnostics = dense_eval.build_local_dense_matches(
            query,
            rendered,
            rendered_valid,
            radius_px=1,
            anchor_stride=1,
            temperature=0.07,
            batch_size=32,
            min_similarity=-1.0,
            max_dense_matches=0,
            correspondence_mode="soft",
            query_valid=query_valid,
        )

        self.assertIsNotNone(matches)
        _, _, score = matches
        self.assertTrue(torch.isfinite(score).all())
        self.assertTrue(torch.all(score == 0))
        self.assertEqual(diagnostics["local_score_finite_fraction"], 1.0)

    def test_global_matching_excludes_invalid_query_cells(self):
        generator = torch.Generator().manual_seed(23)
        channels, height, width = 32, 8, 8
        query = F.normalize(
            torch.randn(channels, height, width, generator=generator), p=2, dim=0
        )
        rendered = query.clone()
        query_coarse = F.normalize(
            F.interpolate(query[None], size=(2, 2), mode="bilinear", align_corners=False)[0],
            p=2,
            dim=0,
        )
        rendered_valid = torch.ones(height, width, dtype=torch.bool)
        query_valid = torch.ones(height, width, dtype=torch.bool)
        query_valid[0, 0] = False

        matches, _ = dense_eval.build_dense_matches(
            query,
            query_coarse,
            rendered,
            rendered_valid,
            coarse_temperature=0.1,
            fine_temperature=0.1,
            coarse_threshold=0.0,
            fine_threshold=0.0,
            valid_cell_fraction=0.5,
            max_coarse_matches=0,
            max_dense_matches=0,
            query_valid=query_valid,
        )

        self.assertIsNotNone(matches)
        query_xy, _, _ = matches
        self.assertFalse(bool(((query_xy[:, 0] == 0) & (query_xy[:, 1] == 0)).any()))

    def test_ulfloc_geometric_support_keeps_similarity_and_rejects_reflection(self):
        query_xy = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
                [2.0, 0.0],
                [0.0, 2.0],
                [2.0, 1.0],
            ]
        )
        common = {
            "neighbors": 4,
            "angle_thresh_cos": 0.9659,
            "scale_thresh": 0.1,
            "scale_limit": 3.0,
        }
        identity = dense_eval._ulfloc_geometric_support(query_xy, query_xy, **common)
        reflected = dense_eval._ulfloc_geometric_support(
            query_xy,
            query_xy * torch.tensor([1.0, -1.0]),
            **common,
        )
        self.assertTrue(torch.all(identity > 4.0))
        self.assertTrue(torch.all(reflected == 0.0))

    def test_ulfloc_matching_recovers_identity_coarse_to_fine_pairs(self):
        generator = torch.Generator().manual_seed(29)
        channels, height, width = 32, 16, 24
        query = F.normalize(
            torch.randn(channels, height, width, generator=generator), p=2, dim=0
        )
        coarse = F.normalize(
            F.interpolate(query[None], size=(2, 3), mode="bilinear", align_corners=False)[0],
            p=2,
            dim=0,
        )
        matches, diagnostics = dense_eval.build_ulfloc_dense_matches(
            query,
            coarse,
            query.clone(),
            torch.ones(height, width, dtype=torch.bool),
            coarse_temperature=0.1,
            fine_temperature=0.1,
            coarse_threshold=0.0,
            fine_threshold=0.0,
            valid_cell_fraction=0.5,
            max_coarse_matches=0,
            max_dense_matches=0,
            geometric_filter=True,
            geometric_neighbors=3,
            geometric_support_threshold=0.0,
            geometric_angle_cos=0.9659,
            geometric_scale_threshold=0.1,
            geometric_scale_limit=3.0,
        )
        self.assertIsNotNone(matches)
        query_xy, rendered_xy, _ = matches
        self.assertGreaterEqual(diagnostics["coarse_matches"], 4)
        self.assertTrue(torch.all(query_xy == rendered_xy))

    def test_prior_gn_weights_downweight_seed_pose_outlier(self):
        points = torch.tensor([[0.0, 0.0, 2.0], [0.0, 0.0, 2.0]])
        target = torch.tensor([[0.0, 0.0], [3.0, 0.0]])
        intrinsic = torch.eye(3)
        pose = torch.eye(4)
        weights, residual, valid = dense_eval._prior_gn_weights(
            points,
            target,
            intrinsic,
            pose,
            torch.ones(2),
            robust_delta_px=0.75,
        )
        self.assertTrue(valid.all())
        self.assertAlmostEqual(float(residual[0]), 0.0)
        self.assertGreater(float(weights[0]), float(weights[1]))


if __name__ == "__main__":
    unittest.main()
