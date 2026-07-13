import json
import tempfile
import unittest
from pathlib import Path

from scripts.summarize_cambridge_strict_2dgs import latest_result, paired_bootstrap


def metrics(values):
    return {
        "query_count": len(values),
        "per_query": {
            f"query-{index}": {"sparse_TE": value}
            for index, value in enumerate(values)
        },
    }


class CambridgeStrict2DGSSummaryTests(unittest.TestCase):
    def test_marginal_median_and_paired_delta_median_are_distinct(self):
        baseline = metrics([0.0, 100.0, 101.0])
        candidate = metrics([50.0, 51.0, 102.0])

        result = paired_bootstrap(
            baseline,
            candidate,
            seed=2026,
            samples=100,
            batch_size=16,
        )

        self.assertEqual(result["marginal_median_delta_cm"], -49.0)
        self.assertEqual(result["paired_median_delta_cm"], 1.0)
        self.assertAlmostEqual(result["paired_mean_delta_cm"], 2.0 / 3.0)

    def test_latest_result_is_scoped_to_artifact_model_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expected_model = root / "experiment-a" / "lafgs_from_sfm" / "ShopFacade"
            other_model = root / "experiment-b" / "lafgs_from_sfm" / "ShopFacade"
            expected_result = root / "strict2dgs-lafgs-baseline-ShopFacade-a"
            other_result = root / "strict2dgs-lafgs-baseline-ShopFacade-b"
            expected_result.mkdir()
            other_result.mkdir()
            (expected_result / "summary.json").write_text(
                json.dumps({"model_path": str(expected_model)})
            )
            (other_result / "summary.json").write_text(
                json.dumps({"model_path": str(other_model)})
            )

            selected = latest_result(
                root,
                "lafgs",
                "ShopFacade",
                "baseline",
                model_path=expected_model,
            )

            self.assertEqual(selected, expected_result)


if __name__ == "__main__":
    unittest.main()
