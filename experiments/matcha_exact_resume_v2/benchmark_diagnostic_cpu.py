"""CPU-only same-content benchmark of already produced A checkpoints."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import argparse
import contextlib
import copy
import gc
import hashlib
import json
from pathlib import Path
import resource
import time
import numpy as np
import torch
import diagnostic_cpu as dc
import exact_resume as er

def clone_tree(v):
    if isinstance(v, torch.Tensor):
        if v.device.type != "cpu":
            raise RuntimeError("benchmark permits only CPU-loaded snapshots")
        return v.detach().clone(memory_format=torch.preserve_format)
    if isinstance(v, np.ndarray):
        return v.copy(order="K")
    if isinstance(v, dict):
        return {k: clone_tree(x) for k, x in v.items()}
    if isinstance(v, list):
        return [clone_tree(x) for x in v]
    if isinstance(v, tuple):
        return tuple(clone_tree(x) for x in v)
    return copy.deepcopy(v)

def layout(v):
    result = {"torch_bytes": 0, "torch_noncontiguous_bytes": 0,
              "torch_tensor_count": 0, "numpy_bytes": 0}
    def visit(x):
        if isinstance(x, torch.Tensor):
            n = x.numel() * x.element_size()
            result["torch_bytes"] += n
            result["torch_tensor_count"] += 1
            if not x.is_contiguous():
                result["torch_noncontiguous_bytes"] += n
        elif isinstance(x, np.ndarray):
            result["numpy_bytes"] += x.nbytes
        elif isinstance(x, dict):
            for y in x.values():
                visit(y)
        elif isinstance(x, (tuple, list)):
            for y in x:
                visit(y)
    visit(v)
    return result

def publish(root, name, value):
    with (root / name).open("x") as f:
        json.dump(value, f, indent=2, sort_keys=True)
        f.flush()
        os.fsync(f.fileno())

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--a-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    torch.set_num_threads(32)
    original_interop = torch.get_num_interop_threads()
    start = time.monotonic()
    trace = [json.loads(x) for x in (args.a_root / "state/trace_from_1.jsonl").read_text().splitlines()]
    expected_a8 = next(x["exact_digest"] for x in trace if x["iteration"] == 8 and "exact_digest" in x)
    evidence = {"schema": "matcha_diagnostic_cpu_same_content_v1",
                "gpu_initialized": False, "native_threads": 32,
                "native_interop": original_interop, "records": [],
                "a_root": str(args.a_root), "expected_a8_digest": expected_a8,
                "limits": "CPU-loaded full snapshot clone and typed raw-byte SHA only. No GPU transfer, numerical training, large serialization or large direct-equality scan is benchmarked. Tiny contracts separately cover equality and serialization."}
    expected_static = None
    for kind, path in (("dynamic", args.a_root / "state/step_00008"),
                       ("static", args.a_root / "state/static")):
        info = json.loads((path / "manifest.json").read_text())
        p = path / "state.pt"
        verify_started = time.perf_counter()
        if p.stat().st_size != info["bytes"] or er.file_sha(p) != info["sha256"]:
            raise RuntimeError("actual A artifact differs from its manifest")
        file_verify_seconds = time.perf_counter() - verify_started
        load_started = time.perf_counter()
        with dc.diagnostic_threads("benchmark_cpu_load"):
            value = torch.load(p, map_location="cpu", weights_only=False)
        load_seconds = time.perf_counter() - load_started
        with dc.diagnostic_threads("benchmark_source_digest"):
            expected = er.semantic_digest(value)
        if kind == "dynamic":
            expected_static = value["static_digest"]
            with dc.diagnostic_threads("benchmark_A8_digest"):
                actual_a8 = er.semantic_digest({k: value[k] for k in er.COMPARISON_FIELDS})
            if actual_a8 != expected_a8:
                raise RuntimeError("CPU-loaded A8 model/Adam/RNG/cache-reference digest differs")
            evidence["actual_a8_digest"] = actual_a8
        elif expected != expected_static:
            raise RuntimeError("actual full static payload differs from A8 static byte digest")
        evidence[kind] = {"path": str(path), "file_bytes": info["bytes"],
                          "manifest_file_sha256": info["sha256"],
                          "source_semantic_sha256": expected,
                          "file_verification_seconds": file_verify_seconds,
                          "cpu_load_seconds": load_seconds,
                          "layout": layout(value)}
        for mode in ("native32", "guard1"):
            if torch.get_num_threads() != 32 or torch.get_num_interop_threads() != original_interop:
                raise RuntimeError("native thread config leaked before benchmark")
            before = dc.rng_bytes()
            started = time.perf_counter()
            manager = dc.diagnostic_threads("same_content_" + kind) if mode == "guard1" else contextlib.nullcontext()
            with manager:
                clone_started = time.perf_counter()
                actual = clone_tree(value)
                clone_seconds = time.perf_counter() - clone_started
                digest_started = time.perf_counter()
                digest = er.semantic_digest(actual)
                digest_seconds = time.perf_counter() - digest_started
            total = time.perf_counter() - started
            if (digest != expected or dc.rng_bytes() != before or
                    torch.get_num_threads() != 32 or torch.get_num_interop_threads() != original_interop):
                raise RuntimeError("byte digest, RNG or native thread restoration differs")
            record = {"kind": kind, "mode": mode, "clone_seconds": clone_seconds,
                      "digest_seconds": digest_seconds, "total_seconds": total,
                      "semantic_sha256": digest, "full_typed_byte_hash_equal": True,
                      "rng_unchanged": True, "native_threads_restored": True,
                      "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024}
            evidence["records"].append(record)
            publish(args.output, kind + "_" + mode + ".json", record)
            print(json.dumps(record, sort_keys=True), flush=True)
            del actual
            gc.collect()
        del value
        gc.collect()
    evidence["elapsed_seconds"] = time.monotonic() - start
    evidence["gpu_initialized"] = torch.cuda.is_initialized()
    evidence["visible_cuda_devices"] = torch.cuda.device_count()
    if evidence["gpu_initialized"] or evidence["visible_cuda_devices"]:
        raise RuntimeError("CPU-only benchmark accessed CUDA")
    evidence["code_sha256"] = {p.name: er.file_sha(p) for p in
        (Path(__file__), Path(er.__file__), Path(dc.__file__))}
    evidence["verdict"] = "PASS_CPU_SAME_CONTENT_ONLY"
    publish(args.output, "report.json", evidence)
    print(json.dumps({"report": str(args.output / "report.json"),
                      "elapsed_seconds": evidence["elapsed_seconds"], "verdict": evidence["verdict"]}), flush=True)

if __name__ == "__main__":
    main()
