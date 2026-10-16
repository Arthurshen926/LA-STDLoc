"""Tiny CPU proofs for read-only diagnostic guard; no GPU initialization."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import copy
import hashlib
from pathlib import Path
import random
import tempfile
import threading
import unittest
import numpy as np
import torch
import diagnostic_cpu as dc
import exact_resume as er

class DiagnosticContracts(unittest.TestCase):
    def setUp(self):
        self.native_threads = torch.get_num_threads()
        self.native_interop = torch.get_num_interop_threads()
        self.rng = er.rng_state()
        dc.drain_records()
    def tearDown(self):
        torch.set_num_threads(self.native_threads)
        er.restore_rng(self.rng)
        self.assertEqual(torch.get_num_interop_threads(), self.native_interop)
        self.assertFalse(torch.cuda.is_initialized())
    def fixture(self):
        bits = np.array([0x80000000, 0x7fc00123, 0x3f800000, 0xff800000,
                         0, 0x7fc00321], dtype=np.uint32).view(np.float32)
        t = torch.from_numpy(bits.copy()).reshape(2, 3).transpose(0, 1)
        return {"tensor": t, "numpy": bits[::2], "none": None, "empty": [],
                "tuple": (1, -0.0, {"ordered": [3, 2, 1]})}
    def test_nested_guard_restores_threads_rng_and_exact_bytes(self):
        torch.set_num_threads(2)
        value = self.fixture()
        before = dc.rng_bytes()
        expected = er.pack(value)
        digest = er.semantic_digest(expected)
        with dc.diagnostic_threads("outer"):
            self.assertEqual(torch.get_num_threads(), 1)
            with dc.diagnostic_threads("inner"):
                actual = er.pack(value)
                self.assertEqual(torch.get_num_threads(), 1)
                self.assertEqual(er.semantic_digest(actual), digest)
                self.assertTrue(er.equal(actual, expected))
            self.assertEqual(torch.get_num_threads(), 1)
        self.assertEqual(torch.get_num_threads(), 2)
        self.assertEqual(dc.rng_bytes(), before)
        records = dc.drain_records()
        self.assertEqual(len(records), 1)
        self.assertTrue(records[0]["completed"])
    def test_exception_nested_path_restores_native_configuration(self):
        torch.set_num_threads(2)
        before = dc.rng_bytes()
        with self.assertRaisesRegex(ValueError, "sentinel"):
            with dc.diagnostic_threads("outer_error"):
                with dc.diagnostic_threads("inner_error"):
                    raise ValueError("sentinel")
        self.assertEqual(torch.get_num_threads(), 2)
        self.assertEqual(dc.rng_bytes(), before)
        self.assertFalse(dc.drain_records()[0]["completed"])
    def test_rng_change_is_detected_without_hiding_it(self):
        torch.set_num_threads(2)
        before = dc.rng_bytes()
        with self.assertRaisesRegex(RuntimeError, "changed RNG"):
            with dc.diagnostic_threads("illegal_draw"):
                random.random()
        self.assertEqual(torch.get_num_threads(), 2)
        self.assertNotEqual(dc.rng_bytes(), before)
    def test_non_main_thread_is_rejected_before_changes(self):
        torch.set_num_threads(2)
        errors = []
        def work():
            try:
                with dc.diagnostic_threads():
                    errors.append("entered")
            except RuntimeError as exc:
                errors.append(str(exc))
        t = threading.Thread(target=work)
        t.start()
        t.join(timeout=2)
        self.assertFalse(t.is_alive())
        self.assertEqual(len(errors), 1)
        self.assertIn("main thread", errors[0])
        self.assertEqual(torch.get_num_threads(), 2)
    def test_serialized_archive_and_readback_bytes_identical(self):
        torch.set_num_threads(2)
        value = er.pack(self.fixture())
        before = dc.rng_bytes()
        with tempfile.TemporaryDirectory() as root:
            a = er.atomic_save.__wrapped__(Path(root) / "native", value)
            b = er.atomic_save(Path(root) / "guarded", value)
            self.assertEqual(a["sha256"], b["sha256"])
            self.assertEqual(a["bytes"], b["bytes"])
            av = er.read_atomic(a["path"])
            bv = er.read_atomic(b["path"])
            self.assertTrue(er.equal(av, bv))
            self.assertEqual(er.semantic_digest(av), er.diagnostic_digest(bv))
        self.assertEqual(torch.get_num_threads(), 2)
        self.assertEqual(dc.rng_bytes(), before)
    def test_tiny_model_adam_digest_matches_unguarded_capture(self):
        from test_cpu_contracts import example
        model, _, _, _, _ = example()
        torch.set_num_threads(2)
        before = dc.rng_bytes()
        baseline = er.capture_model.__wrapped__(model)
        guarded = er.capture_model(model)
        self.assertTrue(er.equal(baseline, guarded))
        self.assertEqual(er.semantic_digest(baseline), er.semantic_digest(guarded))
        self.assertEqual(torch.get_num_threads(), 2)
        self.assertEqual(dc.rng_bytes(), before)

if __name__ == "__main__":
    unittest.main(verbosity=2)
