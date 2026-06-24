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


if __name__ == "__main__":
    unittest.main()
