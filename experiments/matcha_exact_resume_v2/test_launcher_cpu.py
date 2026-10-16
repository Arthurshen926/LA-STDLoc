"""Real, isolated CPU signal/lease contracts. No GPU calls or scientific inputs."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import unittest
import native_n8_launcher as launch

class LauncherContracts(unittest.TestCase):
    def run_isolated(self, body):
        script = "import sys\nsys.path.insert(0, " + repr(str(Path(__file__).parent)) + ")\n"
        script += textwrap.dedent(body)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
        done = subprocess.run([sys.executable, "-c", script], env=env,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              text=True, timeout=10)
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)

    def test_review_binds_entire_spec_except_own_reference(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            def put(name, value):
                p = root / (name + ".json")
                p.write_text(json.dumps(value))
                return {"path": str(p), "sha256": launch.sha(p)}
            plan = {"env_candidate": {}, "executable": launch.PYTHON, "cwd": "/root/MAtCha",
                    "seed": 0, "argv": ["/root/MAtCha/2d-gaussian-splatting/train_with_charts.py",
                                       "--iterations", "30000"]}
            refs = {"training_plan": put("plan", plan)}
            refs["input_review"] = put("cpu_fixture_input_review",
                {"schema": "matcha_resume_input_review_v1", "verdict": "PASS",
                 "training_plan_sha256": refs["training_plan"]["sha256"]})
            refs["binding"] = put("cpu_fixture_binding",
                {"input_receipt_sha256": refs["input_review"]["sha256"], "files": {}})
            refs["source_review"] = {"path": "/placeholder", "sha256": "0" * 64}
            spec = {"references": refs, "frozen_files": {}, "environment": {},
                    "uuid": "CPU_fixture_UUID", "output_root": str(root / "fresh"),
                    "forecast_seconds": 20}
            digest = launch.spec_digest(spec)
            refs["source_review"] = put("cpu_fixture_source_review",
                {"schema": "matcha_exact_launcher_review_v1", "verdict": "PASS",
                 "frozen_files": {}, "spec_body_sha256": digest})
            self.assertEqual(launch.spec_digest(spec), digest)
            launch.verify_frozen(spec, None)
            for field, value in (("uuid", "another"), ("output_root", str(root / "other")),
                                 ("forecast_seconds", 21)):
                bad = copy.deepcopy(spec)
                bad[field] = value
                self.assertNotEqual(launch.spec_digest(bad), digest)
                with self.assertRaisesRegex(RuntimeError, "source/execution review"):
                    launch.verify_frozen(bad, None)
            for key in ("training_plan", "binding", "input_review"):
                bad = copy.deepcopy(spec)
                bad["references"][key]["path"] += ".changed"
                self.assertNotEqual(launch.spec_digest(bad), digest)

    def test_real_signal_in_popen_ownership_window(self):
        self.run_isolated("""
            import os, signal, subprocess, sys, time
            import native_n8_launcher as L
            for sig in L.TERMINATION_SIGNALS:
                signal.signal(sig, L.interruption)
            real_popen = subprocess.Popen
            p = None
            def injecting_popen(*a, **kw):
                child = real_popen(*a, **kw)
                # Exact race: child exists, caller has not received its handle.
                os.kill(os.getpid(), signal.SIGTERM)
                return child
            try:
                with L.blocked_termination_signals():
                    p = injecting_popen([sys.executable, "-c", "import time; time.sleep(5)"],
                                        start_new_session=True)
                assert L.STOP_REQUESTED == signal.SIGTERM
                try:
                    L.wait_owned(p, time.monotonic() + 2)
                    raise AssertionError("interruption was not fail-stop")
                except RuntimeError as exc:
                    assert "interrupted" in str(exc)
            finally:
                with L.blocked_termination_signals():
                    if p is not None:
                        L.cleanup_owned_group(p, term_timeout=.05, kill_timeout=1)
            assert p is not None and p.poll() is not None
            try:
                os.killpg(p.pid, 0)
                raise AssertionError("owned group escaped")
            except ProcessLookupError:
                pass
        """)

    def test_real_second_signal_during_term_kill_drain(self):
        self.run_isolated("""
            import os, signal, subprocess, sys, threading, time
            import native_n8_launcher as L
            for sig in L.TERMINATION_SIGNALS:
                signal.signal(sig, L.interruption)
            code = "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); print('ready',flush=True); time.sleep(5)"
            p = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE,
                                 text=True, start_new_session=True)
            assert p.stdout.readline().strip() == "ready"
            L.interruption(signal.SIGTERM, None)
            timer = threading.Timer(.02, lambda: os.kill(os.getpid(), signal.SIGALRM))
            timer.start()
            drained = False
            try:
                with L.blocked_termination_signals():
                    L.cleanup_owned_group(p, term_timeout=.08, kill_timeout=1)
                    drained = True
                timer.join()
            finally:
                with L.blocked_termination_signals():
                    L.cleanup_owned_group(p, term_timeout=.05, kill_timeout=1)
            assert drained and p.returncode == -signal.SIGKILL
            assert L.STOP_REQUESTED == signal.SIGALRM
            try:
                os.killpg(p.pid, 0)
                raise AssertionError("owned group escaped")
            except ProcessLookupError:
                pass
        """)

    def test_real_wait_deadline_terminates_owned_child(self):
        self.run_isolated("""
            import os, subprocess, sys, time
            import native_n8_launcher as L
            p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(5)"],
                                 start_new_session=True)
            try:
                try:
                    L.wait_owned(p, time.monotonic() + .03, poll_seconds=.01)
                    raise AssertionError("deadline accepted")
                except RuntimeError as exc:
                    assert "deadline" in str(exc)
            finally:
                with L.blocked_termination_signals():
                    L.cleanup_owned_group(p, term_timeout=.05, kill_timeout=1)
            assert p.poll() is not None
        """)

    def test_real_lease_inherited_until_child_exits(self):
        self.run_isolated("""
            import os, subprocess, sys, tempfile
            import native_n8_launcher as L
            with tempfile.TemporaryDirectory() as d:
                p = None
                try:
                    with L.gpu_lease("CPU_fixture_UUID", d) as lease:
                        p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(5)"],
                                             pass_fds=(lease.fileno(),), start_new_session=True)
                    # Parent closed its fd; child still owns the same file description.
                    try:
                        with L.gpu_lease("CPU_fixture_UUID", d):
                            raise AssertionError("second owner obtained live lease")
                    except BlockingIOError:
                        pass
                finally:
                    if p is not None:
                        with L.blocked_termination_signals():
                            L.cleanup_owned_group(p, term_timeout=.05, kill_timeout=1)
                with L.gpu_lease("CPU_fixture_UUID", d):
                    pass
        """)

if __name__ == "__main__":
    unittest.main(verbosity=2)
