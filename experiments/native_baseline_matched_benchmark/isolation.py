"""Finite per-job release gate and continuous evidence. No CUDA imports."""
import os
from pathlib import Path
import subprocess
import threading
import time
from .common import read, ref, verify, sha

ULF_ENV_ALIAS_GPU0_UUID = "GPU-55a12bad-1a5e-1538-5097-3b422bdb7ef1"

def check_lease(value, contract_sha, protocol_sha, now=None):
    now = time.time() if now is None else now
    required = dict(approved=True, scope="cambridge_matched100_formal_timing",
                    contract_sha256=contract_sha, protocol_sha256=protocol_sha,
                    all_training_accuracy_finished=True, cpu_io_quiet=True)
    for key, expected in required.items():
        if value.get(key) != expected:
            raise ValueError("GPU release missing or mismatched: " + key)
    if (not value.get("approval_reference") or not value.get("owner_handover")
        or not str(value.get("gpu_uuid", "")).startswith("GPU-")
        or not value.get("valid_from_unix", float("inf")) <= now
        or value.get("valid_until_unix", -1) < now + 3060
        or value["valid_until_unix"] - value["valid_from_unix"] > 7200):
        raise ValueError("invalid/expiring physical GPU release")
    return value["gpu_uuid"]

def runtime_gate(c, runtime, initialized_context=None):
    verify(runtime["lease_ref"])
    verify(runtime["contract_ref"])
    value = read(runtime["lease_ref"]["path"])
    # At run time only expiry is required; launch reserved the complete3000s budget.
    if time.time() > value["valid_until_unix"] or value["contract_sha256"] != runtime["contract_ref"]["sha256"]:
        raise ValueError("release expired or changed")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != runtime["gpu_uuid"]:
        # Frozen ULF top-level code rewrites the environment string to "0"
        # before model loading. Only the monitor may accept that literal alias
        # after model loading, backed by its actual main-thread context proof.
        expected = dict(case=c["case"], round=c["round"], contract=runtime["contract_ref"],
                        visible_device_count=1, logical_device=0,
                        actual_context_uuid=runtime["gpu_uuid"])
        if (os.environ.get("CUDA_VISIBLE_DEVICES") != "0"
                or runtime["gpu_uuid"] != ULF_ENV_ALIAS_GPU0_UUID
                or c.get("cell", {}).get("family") != "native"
                or c.get("cell", {}).get("method") != "ULFLoc"
                or initialized_context != expected):
            raise ValueError("single released physical GPU UUID required")
    if runtime["contract_ref"] != ref(runtime["contract_path"]) or c != read(runtime["contract_path"]):
        raise ValueError("runtime contract detached")

def sample():
    gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=uuid,name,memory.used,utilization.gpu",
                                  "--format=csv,noheader,nounits"], text=True, timeout=10)
    apps = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
                                   "--format=csv,noheader,nounits"], text=True, timeout=10)
    return dict(unix=time.time(), gpu_csv=gpu, compute_apps_csv=apps,
                loadavg=list(os.getloadavg()), diskstats=Path("/proc/diskstats").read_text(),
                proc_stat=Path("/proc/stat").read_text(), self_io=Path("/proc/self/io").read_text())

def applications(value):
    rows = []
    for row in value["compute_apps_csv"].splitlines():
        columns = row.split(",")
        if len(columns) >= 2:
            rows.append((columns[0].strip(), int(columns[1].strip())))
    return rows

def assert_idle(value, uuid):
    if applications(value):
        raise RuntimeError("another training/accuracy/compute process still exists on a physical GPU")
    if uuid not in [row.split(",")[0].strip() for row in value["gpu_csv"].splitlines()]:
        raise ValueError("released GPU absent")

def physical_context_uuid():
    """DriverAPI primitive already used by the reviewed resource gate; current thread."""
    import ctypes
    import uuid
    cuda = ctypes.CDLL("libcuda.so.1")
    ordinal = ctypes.c_int()
    if cuda.cuCtxGetDevice(ctypes.byref(ordinal)) != 0:
        raise ValueError("cuCtxGetDevice failed: no actual CUDA context")
    value = (ctypes.c_ubyte * 16)()
    function = getattr(cuda, "cuDeviceGetUuid_v2", cuda.cuDeviceGetUuid)
    if function(ctypes.byref(value), ordinal) != 0:
        raise ValueError("cuDeviceGetUuid failed")
    return "GPU-" + str(uuid.UUID(bytes=bytes(value)))

def context_proof(runtime, torch_module=None, driver_uuid=physical_context_uuid):
    if torch_module is None:
        import torch as torch_module
    visible = torch_module.cuda.device_count()
    current = torch_module.cuda.current_device()
    actual = driver_uuid()
    if visible != 1 or current != 0 or actual != runtime["gpu_uuid"]:
        raise ValueError("actual single CUDA context does not match released physical UUID")
    c = read(runtime["contract_path"])
    return dict(case=c["case"], round=c["round"], contract=runtime["contract_ref"],
                visible_device_count=visible, logical_device=current, actual_context_uuid=actual)

class Monitor:
    """Starts sampling once first initialized native/Any query is ready.
    Pre-launch rejects compute on every GPU; initialization observes one process.
    Actual ownership of CUDA execution is proven separately by Driver context UUID.
    CPU/I/O quietness is parent-attested and host telemetry retained, not auto-certified.
    """
    def __init__(self, runtime):
        self.runtime = runtime
        self.samples = []
        self.observed_pids = None
        self.error = None
        self.stop = threading.Event()
        self.thread = None
        self.context_proofs = {}
        self.closed_clean = False

    def observe(self):
        value = sample()
        value["cuda_visible_devices_environment"] = os.environ.get("CUDA_VISIBLE_DEVICES")
        self.samples.append(value)
        rows = applications(value)
        if self.observed_pids is None:
            if len(rows) != 1 or rows[0][0] != self.runtime["gpu_uuid"]:
                raise RuntimeError("cannot identify sole initialized GPU process")
            self.observed_pids = set(rows)
        elif set(rows) - self.observed_pids:
            raise RuntimeError("competing GPU compute process observed")
        runtime_gate(read(self.runtime["contract_path"]), self.runtime,
                     initialized_context=self.context_proofs.get("initialization"))

    def work(self):
        while not self.stop.wait(1):
            try:
                self.observe()
            except BaseException as exc:
                self.error = repr(exc)
                return

    def check(self):
        if self.thread is None:
            self.context_proofs["initialization"] = context_proof(self.runtime)
            self.observe()
            self.thread = threading.Thread(target=self.work, daemon=True)
            self.thread.start()
        if self.error:
            raise RuntimeError("isolation monitoring failed: " + self.error)
        if time.time() > read(self.runtime["lease_ref"]["path"])["valid_until_unix"]:
            raise ValueError("physical GPU release expired")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.stop.set()
        if self.thread:
            self.thread.join(timeout=25)
            if self.thread.is_alive():
                raise RuntimeError("isolation monitor did not terminate")
        if self.error:
            raise RuntimeError("isolation monitoring failed during close: " + self.error)
        if args and args[0] is not None:
            return False
        self.observe()
        self.context_proofs["completion"] = context_proof(self.runtime)
        self.closed_clean = True
