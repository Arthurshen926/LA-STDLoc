"""Single-UUID, fresh-output, bounded native N8 trial-A launcher.

No stage other than native_trial_a_n8 is executable here. No GPU work occurs on import.
Reference provenance: execute_prior_driver_assigned_v2.py SHA
9b9acbcf50c438b586cab36ea6ecb79892535b8e1761a29d6c9a69b7451e3fa9.
"""
import argparse
import contextlib
import fcntl
import ctypes
import hashlib
import json
import os
from pathlib import Path
import runpy
import signal
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parent
RUNS = Path("/mnt/pool/sqy/anygsloc_prior_full_compact_20261011_v1/matcha_exact_runs")
PYTHON = "/root/miniconda3/envs/cybersim_agent/bin/python"
REQUIRED_FILES = ("native_n8_launcher.py", "exact_resume.py", "continuation_gate.py",
                  "train_with_charts_observer.py", "train_with_charts_exact.py",
                  "freeze_binding.py")
REFS = ("training_plan", "binding", "input_review", "source_review")
STOP_REQUESTED = None
TERMINATION_SIGNALS = {signal.SIGTERM, signal.SIGINT, signal.SIGHUP, signal.SIGALRM}

def require(ok, message):
    if not ok:
        raise RuntimeError("N8-launch: " + message)

def sha(path, deadline=None):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            check_interruption()
            require(deadline is None or time.monotonic() < deadline, "deadline during hashing")
            h.update(block)
    return h.hexdigest()

def validate_spec(spec):
    require(set(spec) == {"schema", "stage", "physical_gpu", "logical_gpu", "visible_count",
        "uuid", "deadline_seconds", "forecast_seconds", "output_root", "references",
        "frozen_files", "environment"}, "spec fields")
    require(spec["schema"] == "matcha_native_n8_launch_v1" and
            spec["stage"] == "native_trial_a_n8", "only trial-A N8 is allowed")
    require((spec["physical_gpu"], spec["logical_gpu"], spec["visible_count"]) == (0, 0, 1),
            "only physical0 / one-visible logical0 allowed")
    require(isinstance(spec["uuid"], str) and spec["uuid"].startswith("GPU-") and
            str(uuid.UUID(spec["uuid"][4:])) == spec["uuid"][4:], "invalid assigned UUID")
    require(spec["deadline_seconds"] == 3500 and 0 < spec["forecast_seconds"] < 3380,
            "deadline or forecast margin")
    out = Path(spec["output_root"])
    require(out.is_absolute() and out.parent == RUNS and out.name.startswith("native_n8_trial_a"),
            "output outside isolated N8 trial-A root")
    require(set(spec["references"]) == set(REFS), "missing frozen references")
    for item in spec["references"].values():
        require(set(item) == {"path", "sha256"} and Path(item["path"]).is_absolute() and
                len(item["sha256"]) == 64, "reference format")
    required = {str(ROOT / name) for name in REQUIRED_FILES}
    require(required <= set(spec["frozen_files"]), "missing launcher/source binding")
    require(isinstance(spec["environment"], dict) and all(
        key in {"PYTHONPATH", "PATH", "LD_LIBRARY_PATH", "PYTORCH_CUDA_ALLOC_CONF",
                "OMP_NUM_THREADS", "MKL_NUM_THREADS", "PYTHONHASHSEED", "CUBLAS_WORKSPACE_CONFIG"}
        and isinstance(value, str) for key, value in spec["environment"].items()), "unapproved environment key")
    return spec

def spec_digest(spec):
    body = dict(spec)
    body["references"] = {k: v for k, v in spec["references"].items() if k != "source_review"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

@contextlib.contextmanager
def blocked_termination_signals():
    previous = signal.pthread_sigmask(signal.SIG_BLOCK, TERMINATION_SIGNALS)
    try:
        yield
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, previous)

@contextlib.contextmanager
def gpu_lease(assigned_uuid, directory=RUNS):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    handle = (directory / ("." + assigned_uuid + ".lease")).open("a+")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield handle
    finally:
        handle.close()

def verify_frozen(spec, deadline):
    for item in spec["references"].values():
        require(sha(item["path"], deadline) == item["sha256"], "changed reference: " + item["path"])
    for path, expected in spec["frozen_files"].items():
        require(sha(path, deadline) == expected, "changed execution source: " + path)
    docs = {key: json.loads(Path(value["path"]).read_text()) for key, value in spec["references"].items()}
    review = docs["source_review"]
    require(review["schema"] == "matcha_exact_launcher_review_v1" and review["verdict"] == "PASS" and
            review["frozen_files"] == spec["frozen_files"] and
            review["spec_body_sha256"] == spec_digest(spec), "source/execution review missing or stale")
    inp = docs["input_review"]
    require(inp["schema"] == "matcha_resume_input_review_v1" and inp["verdict"] == "PASS" and
            inp["training_plan_sha256"] == spec["references"]["training_plan"]["sha256"],
            "independent real input review missing")
    require(docs["binding"]["input_receipt_sha256"] == spec["references"]["input_review"]["sha256"],
            "binding input receipt differs")
    for path, expected in spec["frozen_files"].items():
        require(docs["binding"]["files"].get(path) == expected, "execution source absent from binding")
    plan = docs["training_plan"]
    require(spec["environment"] == plan["env_candidate"], "frozen native runtime environment differs")
    require(plan["executable"] == PYTHON and plan["cwd"] == "/root/MAtCha" and plan["seed"] == 0,
            "frozen recipe executable/cwd/seed changed")
    argv = plan["argv"]
    require(argv[0] == "/root/MAtCha/2d-gaussian-splatting/train_with_charts.py" and
            argv[argv.index("--iterations") + 1] == "30000" and
            not any(a.startswith("--exact_") or a in ("--start_checkpoint", "--early_stop_iteration")
                    for a in argv), "native30000 recipe changed")
    return docs

def nvidia_inventory(check_output=subprocess.check_output):
    raw = check_output(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"], text=True, timeout=15)
    return {int(line.split(",")[0]): line.split(",")[1].strip() for line in raw.splitlines() if line.strip()}

def verify_assignment(spec, check_output=subprocess.check_output):
    inventory = nvidia_inventory(check_output)
    require(inventory.get(0) == spec["uuid"], "physical0 UUID differs from assignment")
    raw = check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"],
                       text=True, timeout=15)
    require(not any(line.split(",")[0].strip() == spec["uuid"] for line in raw.splitlines() if line.strip()),
            "assigned GPU0 is occupied")
    return inventory

def context_uuid():
    class CUuuid(ctypes.Structure):
        _fields_ = [("bytes", ctypes.c_ubyte * 16)]
    lib = ctypes.CDLL("libcuda.so.1")
    lib.cuCtxGetDevice.argtypes = [ctypes.POINTER(ctypes.c_int)]
    lib.cuCtxGetDevice.restype = ctypes.c_int
    lib.cuDeviceGetUuid.argtypes = [ctypes.POINTER(CUuuid), ctypes.c_int]
    lib.cuDeviceGetUuid.restype = ctypes.c_int
    device = ctypes.c_int()
    require(lib.cuCtxGetDevice(ctypes.byref(device)) == 0, "no actual CUDA context")
    value = CUuuid()
    require(lib.cuDeviceGetUuid(ctypes.byref(value), device) == 0, "actual context UUID unavailable")
    return "GPU-" + str(uuid.UUID(bytes=bytes(value.bytes)))

def validate_context(spec, actual, visible, logical):
    require(actual == spec["uuid"] and visible == 1 and logical == 0,
            "actual DriverAPI context is not assigned physical0/logical0")

def context_proof(spec, torch):
    actual = context_uuid()
    validate_context(spec, actual, torch.cuda.device_count(), torch.cuda.current_device())
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == spec["uuid"], "visibility changed")
    return {"pid": os.getpid(), "physical_gpu": 0, "logical_gpu": 0, "visible_count": 1,
            "assigned_uuid": spec["uuid"], "actual_driverapi_context_uuid": actual,
            "numeric_pid_nvml_attribution_used": False,
            "allocated_bytes": torch.cuda.memory_allocated(0)}

def write_new(path, value):
    with Path(path).open("x") as f:
        json.dump(value, f, indent=2, sort_keys=True)
        f.flush()
        os.fsync(f.fileno())

def claim_output(root):
    root = Path(root)
    root.parent.mkdir(parents=True, exist_ok=True)
    root.mkdir(exist_ok=False)
    return root

def native_argv(spec, plan, run_root):
    argv = list(plan["argv"])
    argv[0] = str(ROOT / "train_with_charts_observer.py")
    argv[argv.index("-m") + 1] = str(Path(run_root) / "model")
    return argv + ["--exact_manifest", spec["references"]["binding"]["path"],
                   "--exact_dir", str(Path(run_root) / "state"),
                   "--exact_stop", "8", "--exact_windows", "1:8"]

def interruption(signum, frame):
    # Defer raising until normal flow. A handler must never interrupt Popen ownership,
    # cleanup entry, a second signal's cleanup, or the result-file write.
    global STOP_REQUESTED
    STOP_REQUESTED = signum

def check_interruption():
    require(STOP_REQUESTED is None, "interrupted; terminating owned child group")

def wait_owned(process, deadline, poll_seconds=0.2):
    while True:
        check_interruption()
        remaining = deadline - time.monotonic()
        require(remaining > 0, "child wait deadline")
        try:
            return process.wait(timeout=min(poll_seconds, remaining))
        except subprocess.TimeoutExpired:
            pass

def child(spec_path, run_root):
    # Popen inherited the parent's temporary critical-section mask across exec.
    signal.pthread_sigmask(signal.SIG_UNBLOCK, TERMINATION_SIGNALS)
    spec = validate_spec(json.loads(Path(spec_path).read_text()))
    require(Path(run_root).resolve() == Path(spec["output_root"]).resolve(), "child output binding")
    run_root = Path(run_root)
    owner = json.loads((run_root / "launch.json").read_text())
    require(owner["spec_sha256"] == sha(spec_path) and owner["parent_pid"] == os.getppid(),
            "child was not launched by its bound parent")
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == spec["uuid"], "child visibility")
    lease_fd = int(os.environ.get("MATCHA_LEASE_FD", "-1"))
    require(lease_fd >= 0, "child GPU lease missing")
    held, expected = os.fstat(lease_fd), (RUNS / ("." + spec["uuid"] + ".lease")).stat()
    require((held.st_dev, held.st_ino) == (expected.st_dev, expected.st_ino), "child lease identity")
    for name in REQUIRED_FILES:
        require(sha(ROOT / name) == spec["frozen_files"][str(ROOT / name)], "child execution source changed")
    import torch
    torch.cuda.set_device(0)
    implicit = torch.zeros(1).cuda()
    require(implicit.device == torch.device("cuda:0"), "implicit CUDA is not logical0")
    before = context_proof(spec, torch)
    del implicit
    write_new(run_root / "context_before.json", before)
    plan = json.loads(Path(spec["references"]["training_plan"]["path"]).read_text())
    argv = native_argv(spec, plan, run_root)
    sys.path.insert(0, str(ROOT))
    sys.argv = argv
    started = time.monotonic()
    try:
        runpy.run_path(argv[0], run_name="__main__")
    finally:
        after = context_proof(spec, torch)
        write_new(run_root / "context_after.json", after)
    require((run_root / "state/step_00008/manifest.json").is_file(), "N8 checkpoint missing")
    write_new(run_root / "n8_child_complete.json", {"schema": "matcha_n8_child_complete_v1",
        "training_seconds": time.monotonic() - started, "native_horizon": 30000,
        "observed_terminal": 8, "before": before, "after": after})

def cleanup_owned_group(process, killpg=os.killpg, term_timeout=15, kill_timeout=10):
    # Only the group created by this launcher's own start_new_session child.
    require(process.pid > 1 and process.pid != os.getpgrp(), "invalid owned process group")
    try:
        killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=term_timeout)
    except subprocess.TimeoutExpired:
        try:
            killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=kill_timeout)
    # Catch orphan descendants after their group leader exits.
    try:
        killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass

def launch(spec_path):
    started = time.monotonic()
    spec_path = Path(spec_path).resolve()
    spec = validate_spec(json.loads(spec_path.read_text()))
    deadline = started + spec["deadline_seconds"]
    check_interruption()
    verify_frozen(spec, deadline - 120)
    with gpu_lease(spec["uuid"]) as lease:
        verify_assignment(spec)
        check_interruption()
        root = claim_output(spec["output_root"])
        write_new(root / "launch.json", {"schema": "matcha_n8_launch_record_v1",
            "spec_sha256": sha(spec_path), "spec_body_sha256": spec_digest(spec),
            "parent_pid": os.getpid(), "spec": spec,
            "deadline_seconds": 3500, "cleanup_reserve_seconds": 120})
        env = os.environ.copy()
        env.update(spec["environment"])
        env["CUDA_VISIBLE_DEVICES"] = spec["uuid"]
        env["ANYGSLOC_GPU_PHYSICAL"] = "0"
        env["MATCHA_LEASE_FD"] = str(lease.fileno())
        command = [PYTHON, str(Path(__file__).resolve()), "--child", str(spec_path),
                   "--run-root", str(root)]
        process = None
        verdict = "FAILED"
        try:
            with (root / "stdout.log").open("x") as out, (root / "stderr.log").open("x") as err:
                check_interruption()
                require(time.monotonic() < deadline - 120, "preflight exhausted deadline")
                # Block termination until the newly created group's handle is owned.
                with blocked_termination_signals():
                    process = subprocess.Popen(command, cwd="/root/MAtCha", env=env, stdout=out,
                        stderr=err, start_new_session=True, pass_fds=(lease.fileno(),))
                code = wait_owned(process, deadline - 120)
                require(code == 0, "N8 child failed")
            check_interruption()
            require((root / "n8_child_complete.json").is_file(), "missing child completion")
            for name in ("context_before.json", "context_after.json"):
                proof = json.loads((root / name).read_text())
                validate_context(spec, proof["actual_driverapi_context_uuid"], proof["visible_count"], proof["logical_gpu"])
            verdict = "N8_COMPLETE_NOT_EQUIVALENCE"
        finally:
            # A second signal cannot interrupt draining the group we created.
            with blocked_termination_signals():
                if process is not None:
                    cleanup_owned_group(process)
                write_new(root / "launcher_result.json", {"schema": "matcha_n8_launcher_result_v1",
                    "verdict": verdict, "elapsed_seconds": time.monotonic() - started,
                    "native_horizon": 30000, "terminal": 8, "gpu_equivalence_passed": False,
                    "long_prefix_or_resume_launched": False})
        check_interruption()
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
        require(args.spec is not None and args.run_root is None, "missing N8 spec")
        for signum in TERMINATION_SIGNALS:
            signal.signal(signum, interruption)
        signal.setitimer(signal.ITIMER_REAL, 3430)
        try:
            launch(args.spec)
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
