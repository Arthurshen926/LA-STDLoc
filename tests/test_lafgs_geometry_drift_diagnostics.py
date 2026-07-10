import unittest

import torch

from scripts.diagnose_lafgs_geometry_drift import (
    compute_geometry_drift_summary,
    summarize_distances,
)


class LafgsGeometryDriftDiagnosticsTest(unittest.TestCase):
    def test_source_aware_drift_ignores_reorder_and_measures_child_offset(self):
        reference_xyz = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [100.0, 0.0, 0.0],
            ]
        )
        current_xyz = torch.tensor(
            [
                [100.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.1, 0.0, 0.0],
            ]
        )
        loc_state = {
            "loc_source_index": torch.tensor([1, 0, 0]),
            "loc_source_xyz": torch.tensor(
                [
                    [100.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                ]
            ),
            "loc_birth_iteration": torch.tensor([0, 0, 5000]),
        }

        summary = compute_geometry_drift_summary(
            current_xyz=current_xyz,
            loc_state=loc_state,
            reference_xyz=reference_xyz,
            iteration=10000,
        )

        self.assertGreater(summary["row_index_drift"]["mean"], 60.0)
        self.assertAlmostEqual(summary["source_xyz_drift"]["max"], 0.1, places=6)
        self.assertAlmostEqual(summary["reference_source_index_drift"]["max"], 0.1, places=6)
        self.assertEqual(summary["source_index"]["unique_source_count"], 2)
        self.assertEqual(summary["source_index"]["max_children_per_source"], 2)
        self.assertEqual(summary["birth_iteration"]["born_after_reference_count"], 1)

    def test_summarize_distances_handles_empty_input(self):
        summary = summarize_distances(torch.empty(0))

        self.assertEqual(summary["count"], 0)
        self.assertIsNone(summary["mean"])
        self.assertIsNone(summary["max"])


if __name__ == "__main__":
    unittest.main()
