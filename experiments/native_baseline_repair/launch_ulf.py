"""Bounded ULF-only launcher. Does not alter native training code or parameters."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import time

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def read(path):
    return json.loads(Path(path).read_text())

def check_ref(ref):
    p = Path(ref["path"])
    if p.stat().st_size != ref["bytes"] or sha(p) != ref["sha256"]:
        raise ValueError("artifact drift: " + str(p))

def nofile_pair(soft, hard, required=4096):
    if hard != resource.RLIM_INFINITY and hard < required:
        raise ValueError("NOFILE hard limit too low")
    return (soft if soft == resource.RLIM_INFINITY else max(soft, required)), hard

def check_review(review, protocol_sha):
    if (review.get("complete") is not True or review.get("verdict") != "PASS"
            or review.get("protocol_sha256") != protocol_sha
            or review.get("approved_methods") != ["ULFLoc"]):
        raise ValueError("ULF training independent review missing or mismatched")

def check_lease(lease, protocol_sha, semantic, now=None, min_remaining_seconds=0):
    now = time.time() if now is None else now
    if (lease.get("approved") is not True
            or lease.get("scope") != "ulf_stmary_native_init_training"
            or lease.get("protocol_sha256") != protocol_sha
            or semantic not in lease.get("mapping_semantics", [])
            or not lease.get("approval_reference") or not lease.get("owner_handover")
            or not str(lease.get("gpu_uuid", "")).startswith("GPU-")
            or not lease.get("valid_from_unix", float("inf")) <= now <= lease.get("valid_until_unix", -1)):
        raise ValueError("missing or invalid GPU lease")
    if lease["valid_until_unix"] - lease["valid_from_unix"] > 7200:
        raise ValueError("GPU lease exceeds two hours")
    if lease["valid_until_unix"] < now + min_remaining_seconds:
        raise ValueError("GPU lease cannot cover full Runner timeout and launch margin")
    return lease["gpu_uuid"]

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--mapping-semantics", choices=("off", "on"), required=True)
    ap.add_argument("--lease", type=Path, required=True)
    a = ap.parse_args()
    root = a.root.resolve()
    if (root / "delegation.json").exists():
        raise ValueError("delegated task reuse is forbidden")
    pp = root / "training_protocol_v4.json"
    protocol = read(pp); ph = sha(pp)
    ready = read(root / "training_cpu_preflight_v4.json")
    if not ready.get("complete") or ready.get("protocol_sha256") != ph:
        raise ValueError("CPU preflight mismatch")
    check_review(read(root / "independent_training_preflight_review_v4.json"), ph)
    lease_sha = sha(a.lease)
    uuid = check_lease(read(a.lease), ph, a.mapping_semantics)
    if sha(a.lease) != lease_sha:
        raise ValueError("GPU lease changed during initial read")
    for path, digest in protocol["native_code_sha256"].items():
        if sha(path) != digest:
            raise ValueError("code drift: " + path)
    for ref in protocol["inputs"].values():
        check_ref(ref)
    for ref in protocol["evidence"]:
        check_ref(ref)
    if not os.path.samefile(protocol["source"], protocol["canonical_source"]):
        raise ValueError("source alias changed")
    key = "ULFLoc_StMarysChurch_mapping_semantics_" + a.mapping_semantics + "_train"
    spec = protocol["jobs"][key]
    if spec.get("execution_ready") is not True or spec["timeout_seconds"] != 3400:
        raise ValueError("unapproved training stage")
    if Path(spec["fresh_model_path"]).exists() or Path(spec["fresh_receipt_path"]).exists():
        raise FileExistsError("model or receipt already exists; refuse overwrite")
    gpu_rows = subprocess.check_output(["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"], text=True).splitlines()
    if uuid not in [x.strip() for x in gpu_rows]:
        raise ValueError("leased GPU missing")
    apps = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"], text=True)
    if any(x.split(",")[0].strip() == uuid for x in apps.splitlines()):
        raise RuntimeError("leased GPU is occupied")
    fresh_lease = read(a.lease)
    if sha(a.lease) != lease_sha:
        raise ValueError("GPU lease changed during preflight")
    if check_lease(fresh_lease, ph, a.mapping_semantics, min_remaining_seconds=3460) != uuid:
        raise ValueError("GPU lease identity changed")
    previous_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
    new_limit = nofile_pair(*previous_limit)
    resource.setrlimit(resource.RLIMIT_NOFILE, new_limit)
    env = os.environ.copy(); env.update(spec["new_job"]["env"])
    env["CUDA_VISIBLE_DEVICES"] = uuid
    for key_env in ("ULFLOC_POSTPROCESS_PLY", "ULFLOC_POSTPROCESS_INDICES"):
        env.pop(key_env, None)
    out = root / "launches" / key
    out.mkdir(parents=True, exist_ok=False)
    command = spec["new_job"]["command"]
    record = {"protocol_sha256": ph, "stage": key, "command": command,
              "gpu_uuid": uuid, "unix": time.time(), "pid": os.getpid(),
              "nofile_before": previous_limit, "nofile_after": new_limit,
              "lease": {"path": str(a.lease), "sha256": lease_sha},
              "launcher_sha256": sha(__file__), "runner_timeout_seconds": 3400}
    (out / "launch.json").write_text(json.dumps(record, indent=2))
    print(json.dumps({"launch": str(out), "gpu_uuid": uuid, "stage": key}), flush=True)
    with (out / "native.log").open("xb", buffering=0) as log:
        os.dup2(log.fileno(), 1); os.dup2(log.fileno(), 2)
        os.chdir(spec["new_job"]["cwd"])
        os.execvpe(command[0], command, env)

if __name__ == "__main__":
    main()
