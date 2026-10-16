"""Tiny, no-CUDA contracts for conditional native B8 continuation."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
import prefix_gate as p
from test_projection_cpu import raw_fixture
from project_a_reference import project_records, validate_record

class PrefixContracts(unittest.TestCase):
    def make(self):
        ref = {validate_record(r): r for r in project_records(raw_fixture())}
        return p.PrefixGate(ref, p.POLICY), ref
    def fill(self, gate, ref):
        for k in range(1, 9):
            gate.before_step(k)
            gate.observe(ref[(k, False)], ref[(8, True)]["exact_digest"] if k == 8 else None)
    def test_budget_pass_is_durable_before_step9(self):
        gate, ref = self.make()
        with self.assertRaises(RuntimeError):
            gate.before_step(9)
        self.fill(gate, ref)
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "gate.json"
            self.assertTrue(gate.decide(600., 30., 20., path, {"cpu_fixture": True}))
            self.assertEqual(json.loads(path.read_text())["verdict"], "PASS")
            gate.before_step(9)
    def test_insufficient_budget_persists_stop_and_blocks_step9(self):
        gate, ref = self.make()
        self.fill(gate, ref)
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "gate.json"
            self.assertFalse(gate.decide(900., 100., 30., path, {}))
            self.assertEqual(json.loads(path.read_text())["comparison_verdict"], "PASS")
            with self.assertRaises(RuntimeError):
                gate.before_step(9)
    def test_missing_duplicate_mismatch_or_endpoint_blocks(self):
        for mode in ("missing", "duplicate", "difference", "endpoint"):
            gate, ref = self.make()
            with self.assertRaises(RuntimeError):
                if mode == "missing": gate.observe(ref[(2, False)])
                elif mode == "duplicate":
                    gate.observe(ref[(1, False)])
                    gate.observe(ref[(1, False)])
                elif mode == "difference":
                    bad = copy.deepcopy(ref[(1, False)])
                    bad["rng_digest"] = "b" * 64
                    gate.observe(bad)
                else:
                    for k in range(1, 9): gate.observe(ref[(k, False)])
            self.assertFalse(gate.passed)
    def test_invalid_or_unmeasured_forecast_rejected(self):
        for timings in ((0, 1, 1), (500, 0, 2), (500, 2, 0), (1, 2, 3),
                        (float("nan"), 2, 3)):
            with self.assertRaises(RuntimeError): p.forecast(*timings, p.POLICY)
        bad = dict(p.POLICY, historical_prefix_seconds=100)
        with self.assertRaises(RuntimeError): p.forecast(500, 2, 3, bad)
    def test_ceiling_is_strict_and_failed_write_never_opens(self):
        elapsed = 3380 - 2186 * 1.10 - 2 * 1.25 * 20 - 120
        self.assertEqual(p.forecast(elapsed, 10, 10, p.POLICY)["verdict"], "STOP_AT_8_BUDGET")
        gate, ref = self.make()
        self.fill(gate, ref)
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "exists"
            path.write_text("do not overwrite")
            with self.assertRaises(FileExistsError): gate.decide(500, 10, 10, path, {})
            self.assertFalse(gate.passed)
