"""Two separately reviewed native seed0 N8 controls; shared reviewed supervisor primitives."""
import argparse
import json
import os
from pathlib import Path
import runpy
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
V1 = ROOT.parent / "matcha_exact_resume"
sys.path.insert(0, str(V1))
import native_n8_launcher as base
RUNS, PYTHON = base.RUNS, base.PYTHON
ASSIGNED = "GPU-55a12bad-1a5e-1538-5097-3b422bdb7ef1"
PAIR = "native_aa_stmary_20261012_v1"
SOURCE_FILES = tuple(V1 / name for name in base.REQUIRED_FILES) + (Path(__file__).resolve(),)
require, sha, write_new = base.require, base.sha, base.write_new

def validate_spec(spec):
    require(set(spec) == {"schema", "stage", "pair_id", "replicate", "physical_gpu", "logical_gpu",
        "visible_count", "uuid", "deadline_seconds", "forecast_seconds", "output_root", "references",
        "frozen_files", "environment"}, "native AA spec fields")
    require(spec["schema"] == "matcha_native_aa_launch_v1" and spec["stage"] == "native_aa_n8" and
            spec["pair_id"] == PAIR and spec["replicate"] in ("u1", "u2"), "unregistered control")
    require((spec["physical_gpu"], spec["logical_gpu"], spec["visible_count"], spec["uuid"]) ==
            (0, 0, 1, ASSIGNED), "unassigned GPU")
    require(spec["deadline_seconds"] == 1200 and spec["forecast_seconds"] == 900, "short-job budget")
    require(Path(spec["output_root"]) == RUNS / (PAIR + "_" + spec["replicate"]), "unique fixed fresh output")
    require(set(spec["references"]) == set(base.REFS), "reference fields")
    for item in spec["references"].values():
        require(set(item) == {"path", "sha256"} and Path(item["path"]).is_absolute() and
                len(item["sha256"]) == 64, "reference format")
    require(set(spec["frozen_files"]) == {str(p) for p in SOURCE_FILES}, "original code plus thin launcher only")
    return spec

def verify_frozen(spec, deadline):
    docs = base.verify_frozen(spec, deadline)
    require(docs["source_review"]["approval_scope"] == "native_u1_u2_only_n8_no_guard_no_promotion",
            "wrong independent approval scope")
    return docs

def native_argv(spec, plan, root):
    argv = list(plan["argv"])
    argv[0] = str(V1 / "train_with_charts_observer.py")
    argv[argv.index("-m") + 1] = str(Path(root) / "model")
    return argv + ["--exact_manifest", spec["references"]["binding"]["path"],
                   "--exact_dir", str(Path(root) / "state"), "--exact_stop", "8"]

def validate_runtime_flags(proof):
    expected = {"threads": 32, "interop": 16, "deterministic": False, "warn_only": False,
        "cudnn_benchmark": False, "cudnn_deterministic": False,
        "cudnn_tf32": True, "matmul_tf32": False}
    require(all(proof["runtime"].get(k) == v for k, v in expected.items()), "native runtime flags differ from A")
    require(not proof["autocast_enabled"] and not proof["cpu_autocast_enabled"] and
            not proof["diagnostic_guard_imported"], "AMP/guard appeared")

def runtime_proof(spec, torch):
    import exact_resume as original
    require(Path(original.__file__).resolve() == V1 / "exact_resume.py", "non-original helper imported")
    require("diagnostic_cpu" not in sys.modules, "diagnostic guard is forbidden")
    proof = {"context": base.context_proof(spec, torch), "runtime": original.runtime_binding(),
            "autocast_enabled": torch.is_autocast_enabled(),
            "cpu_autocast_enabled": torch.is_autocast_cpu_enabled(),
            "autocast_gpu_dtype": str(torch.get_autocast_gpu_dtype()),
            "autocast_cpu_dtype": str(torch.get_autocast_cpu_dtype()),
            "grad_enabled": torch.is_grad_enabled(), "diagnostic_guard_imported": False,
            "original_helper_path": str(Path(original.__file__).resolve())}
    validate_runtime_flags(proof)
    return proof

def resource_preflight(memory_available=None, disk_free=None):
    if memory_available is None:
        values = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
        memory_available = int(values["MemAvailable"].split()[0]) * 1024
    if disk_free is None:
        disk_free = shutil.disk_usage(RUNS).free
    require(memory_available >= 96 * 1024**3 and disk_free >= 64 * 1024**3, "insufficient RAM/disk")
    return {"available_host_ram_bytes": memory_available, "free_disk_bytes": disk_free,
            "minimum_host_ram_bytes": 96 * 1024**3, "minimum_disk_bytes": 64 * 1024**3}

def child(spec_path, run_root):
    signal.pthread_sigmask(signal.SIG_UNBLOCK, base.TERMINATION_SIGNALS)
    spec = validate_spec(json.loads(Path(spec_path).read_text()))
    root = Path(run_root)
    require(root.resolve() == Path(spec["output_root"]).resolve(), "child output binding")
    owner = json.loads((root / "launch.json").read_text())
    require(owner["spec_sha256"] == sha(spec_path) and owner["parent_pid"] == os.getppid(), "unbound child")
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == ASSIGNED, "child visibility")
    lease_fd = int(os.environ.get("MATCHA_LEASE_FD", "-1"))
    require(lease_fd >= 0, "inherited lease missing")
    held, expected = os.fstat(lease_fd), (RUNS / ("." + ASSIGNED + ".lease")).stat()
    require((held.st_dev, held.st_ino) == (expected.st_dev, expected.st_ino), "lease identity")
    verify_frozen(spec, owner["started_monotonic"] + 1080)
    import torch
    torch.cuda.set_device(0)
    implicit = torch.zeros(1).cuda()
    require(implicit.device == torch.device("cuda:0"), "implicit CUDA assignment")
    del implicit
    before = runtime_proof(spec, torch)
    write_new(root / "runtime_before.json", before)
    plan = json.loads(Path(spec["references"]["training_plan"]["path"]).read_text())
    argv = native_argv(spec, plan, root)
    sys.argv = argv
    started = time.monotonic()
    try:
        runpy.run_path(argv[0], run_name="__main__")
    finally:
        write_new(root / "runtime_after.json", runtime_proof(spec, torch))
    records = [json.loads(line) for line in (root / "state/trace_from_1.jsonl").read_text().splitlines()]
    require([(r["iteration"], "exact_digest" in r) for r in records] ==
            [(k, False) for k in range(1, 9)] + [(8, True)], "incomplete/truncated observation")
    require((root / "state/step_00008/manifest.json").is_file(), "missing complete endpoint8")
    perf = [json.loads(line) for line in (root / "state/performance.jsonl").read_text().splitlines()]
    require(perf[-1]["event"] == "stop" and perf[-1]["iteration"] == 8, "not safely stopped8")
    verify_frozen(spec, owner["started_monotonic"] + 1080)
    write_new(root / "native_aa_child_complete.json", {"schema": "matcha_native_aa_child_v1",
        "replicate": spec["replicate"], "training_seconds": time.monotonic() - started,
        "native_horizon": 30000, "observed_terminal": 8, "guard_used": False,
        "comparison_executed": False, "production_continuation_allowed": False})

def launch(spec_path):
    started = time.monotonic()
    spec_path = Path(spec_path).resolve()
    spec = validate_spec(json.loads(spec_path.read_text()))
    deadline = started + 1200
    base.check_interruption()
    verify_frozen(spec, deadline - 120)
    with base.gpu_lease(ASSIGNED) as lease:
        base.verify_assignment(spec)
        base.check_interruption()
        root = base.claim_output(spec["output_root"])
        write_new(root / "resource_preflight.json", resource_preflight())
        write_new(root / "launch.json", {"schema": "matcha_native_aa_launch_record_v1",
            "spec_sha256": sha(spec_path), "spec_body_sha256": base.spec_digest(spec),
            "parent_pid": os.getpid(), "spec": spec, "started_monotonic": started,
            "deadline_seconds": 1200, "cleanup_reserve_seconds": 120})
        env = os.environ.copy()
        env.update(spec["environment"])
        env.update(CUDA_VISIBLE_DEVICES=ASSIGNED, ANYGSLOC_GPU_PHYSICAL="0", MATCHA_LEASE_FD=str(lease.fileno()))
        command = [PYTHON, str(Path(__file__).resolve()), "--child", str(spec_path), "--run-root", str(root)]
        process, verdict = None, "FAILED"
        try:
            with (root / "stdout.log").open("x") as out, (root / "stderr.log").open("x") as err:
                base.check_interruption()
                require(time.monotonic() < deadline - 120, "preflight deadline")
                with base.blocked_termination_signals():
                    process = subprocess.Popen(command, cwd="/root/MAtCha", env=env, stdout=out, stderr=err,
                        start_new_session=True, pass_fds=(lease.fileno(),))
                code = base.wait_owned(process, deadline - 120)
                require(code == 0, "native AA child failed")
            base.check_interruption()
            complete = json.loads((root / "native_aa_child_complete.json").read_text())
            require(complete["observed_terminal"] == 8 and complete["guard_used"] is False, "invalid control")
            for name in ("runtime_before.json", "runtime_after.json"):
                proof = json.loads((root / name).read_text())
                ctx = proof["context"]
                base.validate_context(spec, ctx["actual_driverapi_context_uuid"], ctx["visible_count"], ctx["logical_gpu"])
                require(not proof["autocast_enabled"] and not proof["cpu_autocast_enabled"] and
                        not proof["diagnostic_guard_imported"], "AMP/guard appeared")
            verdict = "NATIVE_N8_COMPLETE_PENDING_PAIR_COMPARISON"
        finally:
            with base.blocked_termination_signals():
                if process is not None:
                    base.cleanup_owned_group(process)
                write_new(root / "launcher_result.json", {"schema": "matcha_native_aa_result_v1",
                    "verdict": verdict, "replicate": spec["replicate"],
                    "elapsed_seconds": time.monotonic() - started, "native_horizon": 30000,
                    "terminal": 8 if verdict != "FAILED" else None,
                    "comparison_executed": False, "production_continuation_allowed": False})
        base.check_interruption()
        require(time.monotonic() < deadline, "total1200s exceeded")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path)
    parser.add_argument("--child", type=Path)
    parser.add_argument("--run-root", type=Path)
    args = parser.parse_args()
    if args.child:
        require(args.spec is None and args.run_root is not None, "invalid child invocation")
        child(args.child, args.run_root)
    else:
        require(args.spec is not None and args.run_root is None, "missing spec")
        for signum in base.TERMINATION_SIGNALS:
            signal.signal(signum, base.interruption)
        signal.setitimer(signal.ITIMER_REAL, 1130)
        try:
            launch(args.spec)
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
