"""Runner-owned STDLoc staged launch. Requires independent review and GPU lease."""
import argparse
import json
import os
from pathlib import Path
import resource
import subprocess
import time

from std_exact import atomic_json, sha
from std_build import PROTOCOL_SHA

STAGES = {"gaussian_1": (0, 10000), "gaussian_2": (10000, 20000),
          "gaussian_3": (20000, 30000), "detector": (30000, 30000),
          "reference": (0, 10032), "probe_a": (10000, 10032), "probe_b": (10000, 10032)}
DIAGNOSTICS = {"reference", "probe_a", "probe_b"}


def read(path):
    return json.loads(Path(path).read_text())


def require_ref(ref):
    path = Path(ref["path"])
    if path.stat().st_size != ref["bytes"] or sha(path) != ref["sha256"]:
        raise ValueError("artifact drift: " + str(path))


def check_review(review, build_sha, semantic, stage):
    if (review.get("complete") is not True or review.get("verdict") != "PASS"
            or review.get("protocol_sha256") != PROTOCOL_SHA
            or review.get("build_sha256") != build_sha
            or review.get("scope") != "std_exact_cpu_preflight"
            or review.get("approved_method") != "STDLoc"
            or review.get("approved_scene") != "StMarysChurch"
            or semantic not in review.get("approved_mapping_semantics", [])
            or stage not in review.get("approved_stages", [])
            or review.get("approved_training_contract") != "state_exact_native_30000_plus_detector"
            or not review.get("reviewer") or not review.get("approval_reference")):
        raise ValueError("independent state-exact review missing or mismatched")
    if not review.get("evidence"):
        raise ValueError("independent review lacks CPU evidence")
    for ref in review["evidence"]:
        require_ref(ref)


def check_lease(lease, build_sha, semantic, stage, now=None):
    now = time.time() if now is None else now
    if (lease.get("approved") is not True or lease.get("scope") != "std_exact_native_stage"
            or lease.get("protocol_sha256") != PROTOCOL_SHA
            or lease.get("build_sha256") != build_sha
            or lease.get("mapping_semantics") != semantic
            or lease.get("stage") != stage
            or not lease.get("approval_reference") or not lease.get("owner_handover")
            or not str(lease.get("gpu_uuid", "")).startswith("GPU-")
            or not lease.get("valid_from_unix", float("inf")) <= now
            or lease.get("valid_until_unix", -1) < now + 3460):
        raise ValueError("missing, mismatched or too-short GPU lease")
    if lease["valid_until_unix"] - lease["valid_from_unix"] > 7200:
        raise ValueError("GPU lease exceeds two hours")
    return lease["gpu_uuid"]


def nofile_pair(soft, hard):
    if hard != resource.RLIM_INFINITY and hard < 4096:
        raise ValueError("NOFILE hard limit too low")
    return (soft if soft == resource.RLIM_INFINITY else max(4096, soft)), hard


def plan(root, semantic, stage):
    root = Path(root).resolve()
    protocol_path = root / "training_protocol_v4.json"
    if sha(protocol_path) != PROTOCOL_SHA:
        raise ValueError("protocol v4 drift")
    protocol = read(protocol_path)
    build_path = root / "std_execution/build.json"
    build = read(build_path)
    build_sha = sha(build_path)
    if (build.get("complete") is not True or build.get("protocol_sha256") != PROTOCOL_SHA
            or not build.get("native_bytes_after_strip_identical")
            or not build.get("native_ast_after_strip_identical")):
        raise ValueError("invalid exact build")
    for path, expected in {**protocol["native_code_sha256"], **build["file_sha256"]}.items():
        if sha(path) != expected:
            raise ValueError("source drift: " + path)
    for ref in protocol["inputs"].values():
        require_ref(ref)
    for ref in protocol["evidence"]:
        require_ref(ref)
    if (root / "delegation.json").exists():
        raise ValueError("delegated task reuse forbidden")
    if not os.path.samefile(protocol["source"], protocol["canonical_source"]):
        raise ValueError("source alias changed")
    key = "STDLoc_StMarysChurch_mapping_semantics_" + semantic + "_train"
    spec = protocol["jobs"][key]
    # v4 intentionally remains blocked for unsplit execution. This adapter has
    # separate, hash-bound admission and never executes that unsplit command.
    if spec["execution_ready"] is not False or spec["timeout_seconds"] != 3400:
        raise ValueError("unexpected base protocol status")
    diagnostic = stage in DIAGNOSTICS
    base = root / "std_execution" / ("diagnostics" if diagnostic else "production") / semantic
    run_dir = base / stage
    if run_dir.exists():
        raise FileExistsError("stage directory already exists; no blind retry: " + str(run_dir))
    start, stop = STAGES[stage]
    model = base / "model" if diagnostic else Path(spec["fresh_model_path"])
    if start == 0 and model.exists():
        raise FileExistsError("initial model already exists")
    if Path(spec["fresh_receipt_path"]).exists():
        raise FileExistsError("final receipt already exists")
    if start:
        previous = "reference" if diagnostic else {10000: "gaussian_1", 20000: "gaussian_2", 30000: "gaussian_3"}[start]
        previous_dir = base / previous
        done = read(previous_dir / "stage_complete.json")
        if (done.get("status") != "complete" or done.get("exact_stage") != previous
                or done.get("build_sha256") != build_sha
                or done.get("probe_only") != diagnostic):
            raise ValueError("preceding stage not complete")
        resume = str(previous_dir / ("checkpoint_10000" if diagnostic else "checkpoint_" + str(start)))
        record = read(Path(resume) / "complete.json")
        verified = read(Path(resume) / "roundtrip_verified.json")
        if not verified.get("complete") or verified.get("state_digest") != record.get("state_digest"):
            raise ValueError("preceding save/load roundtrip not verified")
        if not record.get("complete") or record["iteration"] != start:
            raise ValueError("preceding checkpoint incomplete")
    else:
        resume = None
    command = list(spec["new_job"]["command"])
    command[command.index("--receipt") + 1] = str(run_dir / "native_receipt.json")
    command[command.index("-m") + 1] = str(model)
    # All original scientific flags are unchanged, including iterations=30000.
    native_begin = command.index("--") + 1
    if command[native_begin:].count("--iterations") != 1 or command[command.index("--iterations")+1] != "30000":
        raise ValueError("native schedule drift")
    if command[command.index("-s")+1] != protocol["source"] or "cambridge" not in protocol["source"]:
        raise ValueError("native Cambridge alias must be preserved")
    identity = {"protocol_sha256": PROTOCOL_SHA, "build_sha256": build_sha,
                "mapping_semantics": semantic, "model_path": str(model),
                "source_alias": protocol["source"], "probe_only": diagnostic}
    config = {"schema": "std_exact_stage_v1", "stage": stage, "start": start, "stop": stop,
              "identity": identity, "probe_only": diagnostic, "resume": resume,
              "run_dir": str(run_dir), "wrapper_command": command,
              "native_trainer": str(Path(spec["new_job"]["cwd"]) / "train.py"),
              "generated_trainer": build["generated_trainer"],
              "code_sha256": {**protocol["native_code_sha256"], **build["file_sha256"]},
              "stage_receipt": str(run_dir / "native_receipt.json"),
              "final_receipt": spec["fresh_receipt_path"],
              "checkpoint_at": 10000 if stage == "reference" else (-1 if stage in {"probe_a", "probe_b", "detector"} else stop),
              "checkpoint_out": str(run_dir / ("checkpoint_10000" if diagnostic else "checkpoint_" + str(stop))),
              "trace_start": 10001 if diagnostic else 30001,
              "runner_timeout_seconds": 3400}
    return config, spec["new_job"]["env"], spec["new_job"]["cwd"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--mapping-semantics", choices=("off", "on"), required=True)
    parser.add_argument("--stage", choices=tuple(STAGES), required=True)
    parser.add_argument("--lease", type=Path)
    parser.add_argument("--cpu-plan-only", action="store_true")
    args = parser.parse_args()
    config, native_env, cwd = plan(args.root, args.mapping_semantics, args.stage)
    if args.cpu_plan_only:
        print(json.dumps(config, indent=2))
        return
    root = args.root.resolve()
    build_sha = config["identity"]["build_sha256"]
    check_review(read(root / "std_execution/independent_review.json"), build_sha, args.mapping_semantics, args.stage)
    if args.lease is None:
        raise ValueError("explicit stage GPU lease required")
    lease_sha = sha(args.lease)
    uuid = check_lease(read(args.lease), build_sha, args.mapping_semantics, args.stage)
    rows = subprocess.check_output(["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"], text=True)
    if uuid not in [line.strip() for line in rows.splitlines()]:
        raise ValueError("leased GPU missing")
    apps = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"], text=True)
    if any(line.split(",")[0].strip() == uuid for line in apps.splitlines()):
        raise RuntimeError("leased GPU occupied")
    if sha(args.lease) != lease_sha or check_lease(read(args.lease), build_sha, args.mapping_semantics, args.stage) != uuid:
        raise ValueError("GPU lease changed during preflight")
    run_dir = Path(config["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=False)
    config_path = run_dir / "config.json"
    atomic_json(config_path, config)
    previous = resource.getrlimit(resource.RLIMIT_NOFILE)
    updated = nofile_pair(*previous)
    resource.setrlimit(resource.RLIMIT_NOFILE, updated)
    env = os.environ.copy()
    env.update(native_env)
    env["CUDA_VISIBLE_DEVICES"] = uuid
    env["STD_EXACT_CONFIG"] = str(config_path)
    env["STD_EXACT_CONFIG_SHA256"] = sha(config_path)
    for key in ("ULFLOC_POSTPROCESS_PLY", "ULFLOC_POSTPROCESS_INDICES"):
        env.pop(key, None)
    python = config["wrapper_command"][0]
    entry = str(Path(__file__).resolve().with_name("std_entry.py"))
    atomic_json(run_dir / "launch.json", {"stage": args.stage, "gpu_uuid": uuid,
                "lease_sha256": lease_sha, "build_sha256": build_sha,
                "pid": os.getpid(), "unix": time.time(), "nofile_before": previous,
                "nofile_after": updated, "runner_timeout_seconds": 3400,
                "command": [python, entry]})
    print(json.dumps({"stage": args.stage, "run_dir": str(run_dir), "gpu_uuid": uuid}), flush=True)
    with (run_dir / "native.log").open("xb", buffering=0) as log:
        os.dup2(log.fileno(), 1)
        os.dup2(log.fileno(), 2)
        os.chdir(cwd)
        os.execvpe(python, [python, entry], env)


if __name__ == "__main__":
    main()
