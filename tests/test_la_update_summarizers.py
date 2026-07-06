import importlib.util
import unittest
from pathlib import Path


def load_script(name):
    script_path = Path(__file__).resolve().parents[1] / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class LAUpdateSummarizerTest(unittest.TestCase):
    def test_dense_summary_parses_new_and_legacy_seed_layouts(self):
        module = load_script("summarize_la_update2_dense_long.py")
        root = Path("/tmp/la_dense")

        parsed = module.parse_model_path(
            root
            / "models"
            / "pose_gate_100"
            / "ShopFacade"
            / "train_seed_0"
            / "query_split_2025"
            / "ShopFacade_densekl_from_30500",
            root,
        )
        self.assertEqual(parsed["tag"], "pose_gate_100")
        self.assertEqual(parsed["scene"], "ShopFacade")
        self.assertEqual(parsed["train_seed"], 0)
        self.assertEqual(parsed["query_split_seed"], 2025)
        self.assertFalse(parsed["legacy_seed_layout"])

        legacy = module.parse_model_path(
            root / "models" / "pose_gate_100" / "ShopFacade" / "seed_2025" / "ShopFacade_densekl_from_30500",
            root,
        )
        self.assertEqual(legacy["train_seed"], None)
        self.assertEqual(legacy["query_split_seed"], 2025)
        self.assertTrue(legacy["legacy_seed_layout"])

    def test_topology_summary_parses_new_and_legacy_seed_layouts(self):
        module = load_script("summarize_la_update2_long_closure.py")
        root = Path("/tmp/la_topology")

        parsed = module.parse_model_path(
            root
            / "models"
            / "core_no_mutation_500"
            / "OldHospital"
            / "train_seed_2"
            / "query_split_2027"
            / "OldHospital_v03_topology_from_30500",
            root,
        )
        self.assertEqual(parsed["tag"], "core_no_mutation_500")
        self.assertEqual(parsed["scene"], "OldHospital")
        self.assertEqual(parsed["train_seed"], 2)
        self.assertEqual(parsed["query_split_seed"], 2027)
        self.assertFalse(parsed["legacy_seed_layout"])

        legacy = module.parse_model_path(
            root
            / "models"
            / "core_no_mutation_500"
            / "OldHospital"
            / "seed_2027"
            / "OldHospital_v03_topology_from_30500",
            root,
        )
        self.assertEqual(legacy["train_seed"], None)
        self.assertEqual(legacy["query_split_seed"], 2027)
        self.assertTrue(legacy["legacy_seed_layout"])

    def test_topology_summary_parses_update3_p0_tags(self):
        module = load_script("summarize_la_update2_long_closure.py")

        family, mode, steps = module.parse_tag("p0_S1_500")

        self.assertEqual(family, "p0")
        self.assertEqual(mode, "S1")
        self.assertEqual(steps, 500)

    def test_topology_risk_event_summary_tracks_rejected_metric_diagnostics(self):
        module = load_script("summarize_topology_risk_events.py")
        log_text = """
[Topology] iter=32005 candidates=1519 physical_prune=0 requested_split=8 parent_removed=0 children_added=0 points=342918->342918 risk_accepted=False risk_reason=heldout_pose_not_decreased risk_baseline=0.415939 risk_trial=0.416951 risk_delta=0.001012 risk_metric_count=8 risk_r5_delta=0 risk_r2_delta=0 risk_tail_fail_delta=0
[Topology] iter=32010 candidates=1702 physical_prune=0 requested_split=9 parent_removed=9 children_added=18 points=342918->342927 risk_accepted=True risk_reason=heldout_pose_decreased risk_baseline=0.414889 risk_trial=0.413780 risk_delta=-0.001110 risk_metric_count=8 risk_r5_delta=1 risk_r2_delta=0 risk_tail_fail_delta=-1
[Topology] iter=32015 candidates=1800 physical_prune=0 requested_split=9 parent_removed=0 children_added=0 points=342927->342927 risk_accepted=False risk_reason=heldout_pose_not_decreased risk_baseline=0.5 risk_trial=0.49 risk_delta=-0.01
"""

        events = module.parse_topology_events(log_text)
        summary = module.summarize_events(events)

        self.assertEqual(len(events), 3)
        self.assertEqual(events[0]["iter"], 32005)
        self.assertFalse(events[0]["risk_accepted"])
        self.assertEqual(events[0]["risk_metric_count"], 8)
        self.assertEqual(summary["events"], 3)
        self.assertEqual(summary["accepted"], 1)
        self.assertEqual(summary["rejected"], 2)
        self.assertEqual(summary["children_added_total"], 18)
        self.assertEqual(summary["rejected_with_metric_count"], 1)
        self.assertEqual(summary["rejected_metric_missing"], 1)
        self.assertEqual(summary["accepted_metric_delta"]["r5"], 1)
        self.assertEqual(summary["accepted_metric_delta"]["tail_fail"], -1)
        self.assertEqual(summary["rejected_metric_delta"]["r5"], 0)
        self.assertEqual(summary["reason_counts"]["heldout_pose_not_decreased"], 2)


if __name__ == "__main__":
    unittest.main()
