"""Persist only micro-CPU evidence; this is not a GPU acceptance receipt."""
import hashlib
import json
from pathlib import Path
import time
import unittest
import test_cpu_contracts
import test_launcher_cpu
import test_diagnostic_cpu
import test_projection_cpu
import test_prefix_gate_cpu
import test_b_launcher_cpu
import torch
root = Path(__file__).resolve().parent
suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromModule(module)
                           for module in (test_cpu_contracts, test_launcher_cpu, test_diagnostic_cpu, test_projection_cpu, test_prefix_gate_cpu, test_b_launcher_cpu))
start = time.perf_counter()
result = unittest.TextTestRunner(verbosity=2).run(suite)
report = {
    "schema": "matcha_exact_resume_micro_cpu_v1",
    "tests_run": result.testsRun, "failures": len(result.failures),
    "errors": len(result.errors), "passed": result.wasSuccessful(),
    "test_body_seconds": time.perf_counter() - start,
    "cuda_initialized": torch.cuda.is_initialized(),
    "visible_cuda_devices": torch.cuda.device_count(),
    "fixture_gaussians": 3, "fixture_image_size": [2, 2],
    "gpu_equivalence_executed": False,
    "production_input_read": False,
    "real_signal_and_lease_cpu_tests": True,
    "files": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
              for p in root.glob("*.py")},
}
(root / "cpu_validation.json").write_text(json.dumps(report, indent=2, sort_keys=True))
print(json.dumps(report, sort_keys=True))
raise SystemExit(0 if result.wasSuccessful() and not torch.cuda.is_initialized() else 1)
