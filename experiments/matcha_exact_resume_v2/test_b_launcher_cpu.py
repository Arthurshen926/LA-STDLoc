"""B execution-scope/argv negatives, without GPU or actual inputs."""
import copy
import json
import tempfile
from test_projection_cpu import raw_fixture
from project_a_reference import project_records
from pathlib import Path
import unittest
import native_b_prefix_launcher as b
import native_n8_launcher as base
class BLauncherContracts(unittest.TestCase):
    def fixture(self):
        return {"schema": "matcha_B8_conditional_prefix_launch_v1",
            "stage": "native_B8_conditional15101", "physical_gpu": 0, "logical_gpu": 0,
            "visible_count": 1, "uuid": "GPU-55a12bad-1a5e-1538-5097-3b422bdb7ef1",
            "deadline_seconds": 3500, "forecast_seconds": 900,
            "output_root": str(b.RUNS / "native_b_conditional_prefix_CPU_fixture"),
            "references": {k: {"path": "/cpu-fixture/" + k, "sha256": "0" * 64} for k in b.REFS},
            "frozen_files": {str(b.ROOT / k): "0" * 64 for k in b.REQUIRED_FILES},
            "environment": {}, "promotion_policy": copy.deepcopy(b.POLICY)}
    def test_b_spec_rejects_wrong_gpu_stage_budget_source_and_policy(self):
        value = self.fixture()
        b.validate_spec(value)
        for k, v in (("physical_gpu", 1), ("logical_gpu", 1), ("visible_count", 2),
                     ("uuid", "GPU-unknown"), ("deadline_seconds", 3600),
                     ("forecast_seconds", 3380), ("stage", "long_unconditional"),
                     ("output_root", "/tmp/escape")):
            bad = copy.deepcopy(value); bad[k] = v
            with self.assertRaises(RuntimeError): b.validate_spec(bad)
            self.assertNotEqual(base.spec_digest(value), base.spec_digest(bad))
        for key in ("promotion_policy", "references", "frozen_files"):
            bad = copy.deepcopy(value); bad[key] = {}
            with self.assertRaises(RuntimeError): b.validate_spec(bad)
    def test_real_jsonl_reference_passes_full_frozen_verifier(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            def put(name, value):
                path = root / (name + ".json")
                path.write_text(json.dumps(value))
                return {"path": str(path), "sha256": b.sha(path)}
            records = project_records(raw_fixture())
            trajectory = root / "trajectory.jsonl"
            trajectory.write_text("".join(json.dumps(r) + "\n" for r in records))
            refs = {"a_reference": {"path": str(trajectory), "sha256": b.sha(trajectory)}}
            refs["training_plan"] = put("plan", {"env_candidate": {}, "executable": b.PYTHON,
                "cwd": "/root/MAtCha", "seed": 0, "argv": ["/root/MAtCha/2d-gaussian-splatting/train_with_charts.py", "--iterations", "30000"]})
            refs["input_review"] = put("input", {"schema": "matcha_resume_input_review_v1",
                "verdict": "PASS", "training_plan_sha256": refs["training_plan"]["sha256"]})
            refs["a_calibration_review"] = put("calibration", {"verdict": "PASS", "step8_full_digest": records[-1]["exact_digest"]})
            refs["projection_receipt"] = put("projection", {"schema": "matcha_native_a_projection_v2",
                "projection_path": str(trajectory), "projection_sha256": b.sha(trajectory),
                "calibration_review_sha256": refs["a_calibration_review"]["sha256"]})
            refs["cpu_benchmark"] = put("benchmark", {"cpu_fixture": True})
            refs["binding"] = put("binding", {"input_receipt_sha256": refs["input_review"]["sha256"],
                "files": {v["path"]: v["sha256"] for v in refs.values()}})
            refs["source_review"] = {"path": "/placeholder", "sha256": "0" * 64}
            spec = {"references": refs, "frozen_files": {}, "environment": {}}
            refs["source_review"] = put("review", {"schema": "matcha_exact_launcher_review_v1",
                "verdict": "PASS", "frozen_files": {}, "spec_body_sha256": base.spec_digest(spec)})
            b.verify_frozen(spec, None)
            trajectory.write_text(trajectory.read_text() + json.dumps(records[-1]) + "\n")
            with self.assertRaises(RuntimeError): b.verify_frozen(spec, None)

    def test_host_memory_and_disk_reserve_fail_closed(self):
        b.resource_preflight(96 * 1024**3, 64 * 1024**3)
        for ram, disk in ((95 * 1024**3, 100 * 1024**3), (100 * 1024**3, 63 * 1024**3)):
            with self.assertRaises(RuntimeError): b.resource_preflight(ram, disk)

    def test_b_argv_preserves_horizon_and_has_only_three_full_endpoints(self):
        spec = self.fixture()
        original = ["/native/trainer", "-m", "/old-output", "--iterations", "30000", "--seed", "0"]
        argv = b.native_argv(spec, {"argv": original}, spec["output_root"], "/fixture/spec.json")
        self.assertEqual(original, ["/native/trainer", "-m", "/old-output", "--iterations", "30000", "--seed", "0"])
        self.assertEqual(argv[argv.index("--iterations")+1], "30000")
        self.assertEqual(argv[argv.index("--exact_stop")+1], "15101")
        i = argv.index("--exact_save")
        self.assertEqual(argv[i+1:i+4], ["8", "14999", "15101"])
        self.assertNotIn("--exact_windows", argv)
        self.assertNotIn("--early_stop_iteration", argv)
        self.assertNotIn("--start_checkpoint", argv)
        self.assertEqual(argv[argv.index("--exact_reference_trace")+1], spec["references"]["a_reference"]["path"])
