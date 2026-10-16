"""Single assigned GPU0: B8 A/A, then conditional measured-budget prefix15101."""
import argparse
import json
import os
from pathlib import Path
import runpy
import signal
import shutil
import subprocess
import sys
import time
import uuid
import native_n8_launcher as base
from prefix_gate import POLICY

ROOT, RUNS, PYTHON = base.ROOT, base.RUNS, base.PYTHON
REQUIRED_FILES = base.REQUIRED_FILES + ("native_b_prefix_launcher.py", "prefix_gate.py")
REFS = base.REFS + ("a_reference", "projection_receipt", "a_calibration_review", "cpu_benchmark")
require, sha, write_new = base.require, base.sha, base.write_new

def validate_spec(spec):
    require(set(spec) == {"schema", "stage", "physical_gpu", "logical_gpu", "visible_count",
        "uuid", "deadline_seconds", "forecast_seconds", "output_root", "references",
        "frozen_files", "environment", "promotion_policy"}, "B spec fields")
    require(spec["schema"] == "matcha_B8_conditional_prefix_launch_v1" and
            spec["stage"] == "native_B8_conditional15101", "only registered B8 conditional15101")
    require((spec["physical_gpu"], spec["logical_gpu"], spec["visible_count"]) == (0, 0, 1),
            "only physical0/one-visible logical0")
    require(spec["uuid"] == "GPU-55a12bad-1a5e-1538-5097-3b422bdb7ef1", "unassigned physical0 UUID")
    require(spec["deadline_seconds"] == 3500 and spec["forecast_seconds"] == 900,
            "registered conservative B8 calibration forecast")
    require(spec["promotion_policy"] == POLICY, "unreviewed conditional policy")
    out = Path(spec["output_root"])
    require(out.is_absolute() and out.parent == RUNS and out.name.startswith("native_b_conditional_prefix_"),
            "output outside fresh B root")
    require(set(spec["references"]) == set(REFS), "B reference coverage")
    for item in spec["references"].values():
        require(set(item) == {"path", "sha256"} and Path(item["path"]).is_absolute() and
                len(item["sha256"]) == 64, "reference format")
    require({str(ROOT / name) for name in REQUIRED_FILES} <= set(spec["frozen_files"]),
            "missing B execution source")
    require(isinstance(spec["environment"], dict), "missing runtime environment")
    return spec

def verify_frozen(spec, deadline):
    docs = base.verify_frozen(spec, deadline)
    receipt = docs["projection_receipt"]
    require(receipt["schema"] == "matcha_native_a_projection_v2" and
            receipt["projection_path"] == spec["references"]["a_reference"]["path"] and
            receipt["projection_sha256"] == spec["references"]["a_reference"]["sha256"] and
            receipt["calibration_review_sha256"] == spec["references"]["a_calibration_review"]["sha256"],
            "projected A evidence links differ")
    import project_a_reference as projection
    raw = [json.loads(line) for line in Path(spec["references"]["a_reference"]["path"]).read_text().splitlines()]
    keys = [projection.validate_record(r) for r in raw]
    require(len(keys) == len(set(keys)), "duplicate A projection")
    projection.validate_projected_records(dict(zip(keys, raw)))
    require(docs["a_calibration_review"]["verdict"] == "PASS" and
            docs["a_calibration_review"]["step8_full_digest"] == raw[-1]["exact_digest"], "A8 receipt differs")
    for key in ("a_reference", "projection_receipt", "a_calibration_review", "cpu_benchmark"):
        item = spec["references"][key]
        require(docs["binding"]["files"].get(item["path"]) == item["sha256"], "unbound B evidence")
    return docs

def native_argv(spec, plan, run_root, spec_path):
    argv = list(plan["argv"])
    argv[0] = str(ROOT / "train_with_charts_observer.py")
    argv[argv.index("-m") + 1] = str(Path(run_root) / "model")
    return argv + ["--exact_manifest", spec["references"]["binding"]["path"],
        "--exact_dir", str(Path(run_root) / "state"), "--exact_stop", "15101",
        "--exact_save", "8", "14999", "15101",
        "--exact_reference_trace", spec["references"]["a_reference"]["path"],
        "--exact_prefix_spec", str(Path(spec_path).resolve())]

def load_controller_spec(spec_path, manifest_path, run_root):
    spec_path, run_root = Path(spec_path).resolve(), Path(run_root).resolve()
    spec = validate_spec(json.loads(spec_path.read_text()))
    require(Path(spec["output_root"]).resolve() == run_root, "controller output differs")
    owner = json.loads((run_root / "launch.json").read_text())
    require(owner["spec_sha256"] == sha(spec_path) and owner["parent_pid"] == os.getppid() and
            owner["spec_body_sha256"] == base.spec_digest(spec), "controller unbound launch")
    require(spec["references"]["binding"]["path"] == str(Path(manifest_path).resolve()) and
            spec["references"]["binding"]["sha256"] == sha(manifest_path), "controller binding differs")
    verify_frozen(spec, owner["started_monotonic"] + 3380)
    return spec, owner

def child(spec_path, run_root):
    signal.pthread_sigmask(signal.SIG_UNBLOCK, base.TERMINATION_SIGNALS)
    spec = validate_spec(json.loads(Path(spec_path).read_text()))
    require(Path(run_root).resolve() == Path(spec["output_root"]).resolve(), "child output binding")
    run_root = Path(run_root)
    owner = json.loads((run_root / "launch.json").read_text())
    require(owner["spec_sha256"] == sha(spec_path) and owner["parent_pid"] == os.getppid(), "unbound child")
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == spec["uuid"], "child visibility")
    lease_fd = int(os.environ.get("MATCHA_LEASE_FD", "-1"))
    require(lease_fd >= 0, "child GPU lease missing")
    held, expected = os.fstat(lease_fd), (RUNS / ("." + spec["uuid"] + ".lease")).stat()
    require((held.st_dev, held.st_ino) == (expected.st_dev, expected.st_ino), "child lease identity")
    for name in REQUIRED_FILES:
        require(sha(ROOT / name) == spec["frozen_files"][str(ROOT / name)], "child source changed")
    import torch
    torch.cuda.set_device(0)
    implicit = torch.zeros(1).cuda()
    require(implicit.device == torch.device("cuda:0"), "implicit CUDA assignment")
    before = base.context_proof(spec, torch)
    del implicit
    write_new(run_root / "context_before.json", before)
    plan = json.loads(Path(spec["references"]["training_plan"]["path"]).read_text())
    argv = native_argv(spec, plan, run_root, spec_path)
    sys.path.insert(0, str(ROOT))
    sys.argv = argv
    started = time.monotonic()
    try:
        runpy.run_path(argv[0], run_name="__main__")
    finally:
        write_new(run_root / "context_after.json", base.context_proof(spec, torch))
    gate = json.loads((run_root / "state/B8_conditional_prefix_gate.json").read_text())
    require(gate["comparison_verdict"] == "PASS", "A/A did not complete")
    terminal = 15101 if gate["verdict"] == "PASS" else 8
    require(gate["verdict"] in ("PASS", "STOP_AT_8_BUDGET"), "unknown gate outcome")
    require((run_root / ("state/step_%05d/manifest.json" % terminal)).is_file(), "terminal state missing")
    if terminal == 15101:
        require((run_root / "state/step_14999/manifest.json").is_file(), "14999 checkpoint missing")
    performance = [json.loads(line) for line in (run_root / "state/performance.jsonl").read_text().splitlines()]
    require(performance[-1]["event"] == "stop" and performance[-1]["iteration"] == terminal, "truncated training")
    verify_frozen(spec, owner["started_monotonic"] + 3380)
    write_new(run_root / "b_child_complete.json", {"schema": "matcha_B_conditional_child_v1",
        "training_seconds": time.monotonic() - started, "native_horizon": 30000,
        "observed_terminal": terminal, "a_a_equivalence_passed": True,
        "resume_equivalence_passed": False, "gate_verdict": gate["verdict"]})

def resource_preflight(memory_available=None, disk_free=None):
    if memory_available is None:
        values = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
        memory_available = int(values["MemAvailable"].split()[0]) * 1024
    if disk_free is None:
        disk_free = shutil.disk_usage(RUNS).free
    require(memory_available >= 96 * 1024**3 and disk_free >= 64 * 1024**3,
            "insufficient96GiB RAM /64GiB disk reserve")
    return {"available_host_ram_bytes": memory_available, "free_disk_bytes": disk_free,
            "minimum_host_ram_bytes": 96 * 1024**3, "minimum_disk_bytes": 64 * 1024**3}

def launch(spec_path):
    started = time.monotonic()
    spec_path = Path(spec_path).resolve()
    spec = validate_spec(json.loads(spec_path.read_text()))
    deadline = started + spec["deadline_seconds"]
    base.check_interruption()
    verify_frozen(spec, deadline - 120)
    with base.gpu_lease(spec["uuid"]) as lease:
        base.verify_assignment(spec)
        base.check_interruption()
        root = base.claim_output(spec["output_root"])
        write_new(root / "resource_preflight.json", resource_preflight())
        write_new(root / "launch.json", {"schema": "matcha_B_conditional_launch_v1",
            "spec_sha256": sha(spec_path), "spec_body_sha256": base.spec_digest(spec),
            "parent_pid": os.getpid(), "spec": spec, "started_monotonic": started,
            "deadline_seconds": 3500, "cleanup_reserve_seconds": 120})
        env = os.environ.copy()
        env.update(spec["environment"])
        env.update(CUDA_VISIBLE_DEVICES=spec["uuid"], ANYGSLOC_GPU_PHYSICAL="0",
                   MATCHA_LEASE_FD=str(lease.fileno()))
        command = [PYTHON, str(Path(__file__).resolve()), "--child", str(spec_path), "--run-root", str(root)]
        process, verdict, terminal = None, "FAILED", None
        try:
            with (root / "stdout.log").open("x") as out, (root / "stderr.log").open("x") as err:
                base.check_interruption()
                require(time.monotonic() < deadline - 120, "preflight deadline")
                with base.blocked_termination_signals():
                    process = subprocess.Popen(command, cwd="/root/MAtCha", env=env, stdout=out,
                        stderr=err, start_new_session=True, pass_fds=(lease.fileno(),))
                code = base.wait_owned(process, deadline - 120)
                require(code == 0, "B child failed")
            base.check_interruption()
            complete = json.loads((root / "b_child_complete.json").read_text())
            terminal = complete["observed_terminal"]
            for name in ("context_before.json", "context_after.json"):
                proof = json.loads((root / name).read_text())
                base.validate_context(spec, proof["actual_driverapi_context_uuid"], proof["visible_count"], proof["logical_gpu"])
            verdict = "B_PREFIX15101_COMPLETE_PENDING_RESUME" if terminal == 15101 else "B_AA_PASS_STOP8_BUDGET"
        finally:
            with base.blocked_termination_signals():
                if process is not None:
                    base.cleanup_owned_group(process)
                write_new(root / "launcher_result.json", {"schema": "matcha_B_conditional_result_v1",
                    "verdict": verdict, "elapsed_seconds": time.monotonic() - started,
                    "native_horizon": 30000, "terminal": terminal,
                    "a_a_equivalence_passed": verdict != "FAILED", "resume_equivalence_passed": False})
        base.check_interruption()
        require(time.monotonic() < deadline, "total3500s deadline exceeded")

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
        require(args.spec is not None and args.run_root is None, "missing B spec")
        for signum in base.TERMINATION_SIGNALS:
            signal.signal(signum, base.interruption)
        signal.setitimer(signal.ITIMER_REAL, 3430)
        try:
            launch(args.spec)
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
