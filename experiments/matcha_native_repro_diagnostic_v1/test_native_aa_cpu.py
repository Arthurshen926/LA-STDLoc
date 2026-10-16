"""No GPU, tiny protocol/argv/runtime contracts; original lifecycle suite is reused."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
import native_aa_launcher as u

class NativeAAContracts(unittest.TestCase):
    def fixture(self):
        return {"schema": "matcha_native_aa_launch_v1", "stage": "native_aa_n8",
            "pair_id": u.PAIR, "replicate": "u1", "physical_gpu": 0, "logical_gpu": 0,
            "visible_count": 1, "uuid": u.ASSIGNED, "deadline_seconds": 1200,
            "forecast_seconds": 900, "output_root": str(u.RUNS/(u.PAIR+"_u1")),
            "references": {k: {"path": "/fixture/"+k, "sha256": "0"*64} for k in u.base.REFS},
            "frozen_files": {str(p): "0"*64 for p in u.SOURCE_FILES}, "environment": {}}
    def test_only_two_fixed_fresh_outputs_single_uuid_and_short_deadline(self):
        spec = self.fixture()
        u.validate_spec(spec)
        for key, value in (("uuid", "other"), ("physical_gpu", 1), ("logical_gpu", 1),
                ("replicate", "u3"), ("deadline_seconds", 3500), ("forecast_seconds", 901),
                ("output_root", "/tmp/other"), ("stage", "prefix")):
            bad = copy.deepcopy(spec); bad[key] = value
            with self.assertRaises(RuntimeError): u.validate_spec(bad)
        spec["replicate"] = "u2"; spec["output_root"] = str(u.RUNS/(u.PAIR+"_u2"))
        u.validate_spec(spec)
    def test_original_args_horizon_and_exactly_one_endpoint(self):
        original = ["/original", "-m", "/old", "--iterations", "30000", "--seed", "0"]
        argv = u.native_argv(self.fixture(), {"argv": original}, "/fixture/fresh")
        self.assertEqual(original, ["/original", "-m", "/old", "--iterations", "30000", "--seed", "0"])
        self.assertEqual(argv[0], str(u.V1/"train_with_charts_observer.py"))
        self.assertEqual(argv[-2:], ["--exact_stop", "8"])
        for flag in ("--exact_windows", "--exact_reference_trace", "--exact_resume", "--exact_prefix_spec",
                     "--early_stop_iteration", "--start_checkpoint"):
            self.assertNotIn(flag, argv)
    def test_actual_runtime_flag_mismatch_is_invalid_control(self):
        runtime = {"threads":32, "interop":16, "deterministic":False, "warn_only":False,
            "cudnn_benchmark":False, "cudnn_deterministic":False, "cudnn_tf32":True, "matmul_tf32":False}
        proof = {"runtime":runtime, "autocast_enabled":False, "cpu_autocast_enabled":False,
                 "diagnostic_guard_imported":False}
        u.validate_runtime_flags(proof)
        for key in runtime:
            bad = copy.deepcopy(proof)
            bad["runtime"][key] = 1 if key in ("threads","interop") else not runtime[key]
            with self.assertRaises(RuntimeError): u.validate_runtime_flags(bad)
        for key in ("autocast_enabled","cpu_autocast_enabled","diagnostic_guard_imported"):
            bad = copy.deepcopy(proof); bad[key] = True
            with self.assertRaises(RuntimeError): u.validate_runtime_flags(bad)
    def test_shared_supervisor_functions_are_original_objects(self):
        import native_n8_launcher as base
        self.assertIs(u.base, base)
        self.assertEqual(Path(base.__file__).resolve(), u.V1/"native_n8_launcher.py")
        self.assertNotIn("diagnostic_cpu", __import__("sys").modules)
    def test_resources_reject_insufficient_ram_or_disk(self):
        u.resource_preflight(96*1024**3,64*1024**3)
        for values in ((95*1024**3,64*1024**3),(96*1024**3,63*1024**3)):
            with self.assertRaises(RuntimeError): u.resource_preflight(*values)
