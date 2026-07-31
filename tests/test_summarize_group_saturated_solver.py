import unittest

from scripts.summarize_lafgs_group_saturated_solver import summarize


def _report(scene, baseline_te, candidate_te, catastrophic_regressed=0):
    rows = []
    for index, (baseline, candidate) in enumerate(
        zip(baseline_te, candidate_te)
    ):
        rows.append(
            {
                "image_name": f"query-{index}",
                "seed": 2026,
                "baseline_te_cm": float(baseline),
                "baseline_ae_deg": 0.1,
                "te_cm": float(candidate),
                "ae_deg": 0.1,
            }
        )

    def summary(values):
        values = sorted(float(value) for value in values)
        return {
            "query_count": len(values),
            "median_te_cm": sum(values) / len(values),
            "mean_te_cm": sum(values) / len(values),
            "p90_te_cm": values[-1],
            "recall_5cm_5deg_percent": (
                100.0 * sum(value <= 5.0 for value in values) / len(values)
            ),
        }

    return {
        "scene": scene,
        "solver_version": "group_saturated_poselib_parity_v2",
        "primary_seed_comparison": {
            "seed": 2026,
            "lower_te_count": sum(
                candidate < baseline
                for baseline, candidate in zip(baseline_te, candidate_te)
            ),
            "higher_te_count": sum(
                candidate > baseline
                for baseline, candidate in zip(baseline_te, candidate_te)
            ),
            "catastrophic_recovered_count": 0,
            "catastrophic_regressed_count": catastrophic_regressed,
        },
        "summary_by_seed": {"2026": {"all": summary(candidate_te)}},
        "queries": rows,
    }


class GroupSaturatedFormalGateTest(unittest.TestCase):
    def test_requires_three_improved_scenes_and_safe_greatcourt(self):
        reports = [
            _report("GreatCourt", [2.0, 8.0], [1.0, 7.0]),
            _report("KingsCollege", [2.0, 8.0], [1.0, 7.0]),
            _report("ShopFacade", [2.0, 8.0], [1.0, 7.0]),
            _report("StMarysChurch", [2.0, 8.0], [2.5, 9.0]),
        ]
        payload = summarize(reports)
        self.assertTrue(payload["gate_pass"])
        self.assertEqual(payload["improved_scene_count"], 3)
        self.assertTrue(payload["solver_version_consistent"])

    def test_rejects_a_new_greatcourt_catastrophe(self):
        reports = [
            _report(
                "GreatCourt",
                [2.0, 8.0],
                [1.0, 7.0],
                catastrophic_regressed=1,
            ),
            _report("KingsCollege", [2.0, 8.0], [1.0, 7.0]),
            _report("ShopFacade", [2.0, 8.0], [1.0, 7.0]),
            _report("StMarysChurch", [2.0, 8.0], [1.0, 7.0]),
        ]
        payload = summarize(reports)
        self.assertFalse(payload["gate_pass"])
        self.assertFalse(payload["greatcourt_non_regression"])


if __name__ == "__main__":
    unittest.main()
