import tempfile
import unittest
from pathlib import Path

import torch


class NoReferenceValidMaskTest(unittest.TestCase):
    def test_uniform_midtone_image_stays_valid_but_has_low_support(self):
        from la_artifacts.no_reference_valid_mask import NoReferenceValidMaskBuilder

        image = torch.full((3, 32, 32), 0.5)

        result = NoReferenceValidMaskBuilder().build(image)

        self.assertGreater(result.summary["valid_frac"], 0.98)
        self.assertLess(result.summary["support_frac"], 0.05)

    def test_structured_region_creates_support_without_invalidating_flat_region(self):
        from la_artifacts.no_reference_valid_mask import NoReferenceValidMaskBuilder

        image = torch.full((3, 48, 48), 0.5)
        image[:, :, 20:28] = 0.9

        result = NoReferenceValidMaskBuilder().build(image)

        self.assertGreater(result.support_mask[:, 18:30].float().mean().item(), 0.15)
        self.assertGreater(result.valid_mask.float().mean().item(), 0.95)

    def test_blank_dark_border_is_invalid_but_center_remains_valid(self):
        from la_artifacts.no_reference_valid_mask import NoReferenceValidMaskBuilder

        image = torch.full((3, 40, 40), 0.55)
        image[:, :8, :] = 0.0
        image[:, -8:, :] = 0.0

        result = NoReferenceValidMaskBuilder().build(image)

        self.assertLess(result.valid_mask[:8].float().mean().item(), 0.2)
        self.assertLess(result.valid_mask[-8:].float().mean().item(), 0.2)
        self.assertGreater(result.valid_mask[12:28, 12:28].float().mean().item(), 0.95)

    def test_point_support_and_validity_are_separate(self):
        from la_artifacts.no_reference_valid_mask import NoReferenceValidMaskBuilder

        image = torch.full((3, 32, 32), 0.5)
        image[:, :, 15:17] = 0.95
        result = NoReferenceValidMaskBuilder().build(image)
        points = torch.tensor([[16.0, 16.0], [2.0, 2.0]])

        support = result.support_points(points)
        valid = result.valid_points(points)

        torch.testing.assert_close(valid, torch.tensor([True, True]))
        self.assertTrue(bool(support[0]))
        self.assertFalse(bool(support[1]))

    def test_save_outputs_valid_and_support_masks(self):
        from la_artifacts.no_reference_valid_mask import NoReferenceValidMaskBuilder, save_no_reference_valid_mask_pngs

        image = torch.full((3, 24, 24), 0.5)
        image[:, :, 10:14] = 0.9
        result = NoReferenceValidMaskBuilder().build(image)

        with tempfile.TemporaryDirectory() as tmp:
            paths = save_no_reference_valid_mask_pngs(result, Path(tmp) / "sample")

            self.assertTrue(Path(paths["valid_mask"]).exists())
            self.assertTrue(Path(paths["support_mask"]).exists())
            self.assertTrue(Path(paths["support_score"]).exists())

    def test_no_reference_eval_contact_sheet_uses_separate_masks(self):
        from PIL import Image

        from scripts.evaluate_no_reference_valid_masks import (
            NoReferenceVisualRecord,
            build_no_reference_contact_sheet,
            summarize_point_masks,
        )

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            image_path = tmp / "query.png"
            out_path = tmp / "sheet.jpg"
            Image.new("RGB", (16, 12), color=(120, 120, 120)).save(image_path)
            valid = torch.ones(12, 16, dtype=torch.bool)
            support = torch.zeros(12, 16, dtype=torch.bool)
            support[:, :8] = True
            points = torch.tensor([[2.0, 2.0], [12.0, 8.0]])

            summary = summarize_point_masks(valid, support, points)
            self.assertEqual(summary["valid_point_count"], 2)
            self.assertEqual(summary["support_point_count"], 1)

            build_no_reference_contact_sheet(
                [
                    NoReferenceVisualRecord(
                        query_id="synthetic_rgb:query.png",
                        image_path=str(image_path),
                        valid_mask=valid,
                        support_mask=support,
                        invalid_score=torch.zeros(12, 16),
                        support_score=support.float(),
                        points_xy=points,
                        valid_points=torch.tensor([True, True]),
                        support_points=torch.tensor([True, False]),
                        metrics=summary,
                    )
                ],
                out_path,
                max_records=1,
            )
            self.assertTrue(out_path.exists())
            self.assertGreater(out_path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
