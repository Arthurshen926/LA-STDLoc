import tempfile
import unittest
from pathlib import Path

import torch


class ValidSupportMaskPackageTest(unittest.TestCase):
    def test_no_reference_builder_exports_valid_and_support_masks(self):
        from valid_support_mask import NoReferenceValidSupportMaskBuilder

        image = torch.full((3, 48, 64), 0.5)
        image[:, 10:30, 20:44] = 0.5
        image[:, 10:30, 21:44:2] = 0.95

        result = NoReferenceValidSupportMaskBuilder().build(image)

        self.assertEqual(tuple(result.valid_mask.shape), (48, 64))
        self.assertEqual(tuple(result.support_mask.shape), (48, 64))
        self.assertGreater(result.summary["valid_frac"], 0.99)
        self.assertGreater(result.support_mask[:, 20:44].float().mean().item(), 0.1)

    def test_no_reference_builder_detects_obvious_blank_artifact_regions(self):
        from valid_support_mask import NoReferenceValidSupportMaskBuilder

        image = torch.full((3, 64, 64), 0.55)
        image[:, :, :16] = 0.0

        result = NoReferenceValidSupportMaskBuilder().build(image)

        self.assertLess(result.valid_mask[:, :16].float().mean().item(), 0.25)
        self.assertGreater(result.valid_mask[:, 24:].float().mean().item(), 0.95)
        self.assertGreater(result.summary["invalid_frac"], 0.15)

    def test_score_valid_mask_keeps_clean_connected_regions(self):
        from valid_support_mask import ScoreValidMaskBuilder, ScoreValidMaskConfig

        score = torch.ones((32, 32))
        score[8:24, 8:24] = 0.1

        result = ScoreValidMaskBuilder(
            ScoreValidMaskConfig(max_artifact_score=0.4, erosion_radius=1, min_component_area=16)
        ).build(score)

        self.assertGreater(result.mask.float().mean().item(), 0.1)
        self.assertEqual(result.summary["component_count"], 1)
        self.assertGreater(result.valid_points(torch.tensor([[16.0, 16.0]])).item(), 0)
        self.assertFalse(result.valid_points(torch.tensor([[2.0, 2.0]])).item())

    def test_save_bundle_writes_portable_png_outputs(self):
        from valid_support_mask import NoReferenceValidSupportMaskBuilder, save_mask_bundle_pngs

        result = NoReferenceValidSupportMaskBuilder().build(torch.full((3, 16, 16), 0.5))
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = save_mask_bundle_pngs(result, Path(tmpdir) / "frame_000")

            self.assertEqual(set(paths), {"valid_mask", "support_mask", "invalid_score", "support_score"})
            for value in paths.values():
                self.assertTrue(Path(value).exists())


if __name__ == "__main__":
    unittest.main()
