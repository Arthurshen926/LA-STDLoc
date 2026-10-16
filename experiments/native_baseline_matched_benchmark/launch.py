"""Launcher only: validates CPU review, all30 assets and one exact physical-GPU lease."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import resource
import sys
from .common import *
from .plan import validate_round
from .isolation import check_lease, sample, assert_idle
from experiments.native_baseline_repair.query_plan import validate_contract as validate_accuracy
from experiments.native_baseline_repair.launch_ulf import nofile_pair

def check_review(review, protocol_ref):
    if not (review.get("complete") is True and review.get("verdict") == "PASS"
            and review.get("scope") == "matched_timing_cpu"
            and review.get("protocol_sha256") == protocol_ref["sha256"]
            and review.get("no_gpu_run_performed") is True):
        raise ValueError("new matched adapter independent CPU review required")

def verify_scene_images(snapshot, source, names):
    by_name = {item["name"]: item for item in snapshot["queries"]}
    if len(by_name) != len(snapshot["queries"]):
        raise ValueError("duplicate frozen snapshot query")
    for name in names:
        if sha(Path(source) / "processed" / name) != by_name[name]["image_sha256"]:
            raise ValueError("selected query RGB changed: " + name)

def verify_any_cameras(cameras, snapshot, expected_names):
    import math
    import numpy as np
    if [camera.image_name for camera in cameras] != expected_names:
        raise ValueError("Any full camera population/order changed")
    frozen = {q["name"]: q for q in snapshot["queries"]}
    for camera in cameras:
        q = frozen[camera.image_name]
        if not np.array_equal(np.asarray(camera.pose_w2c, dtype=np.float32), np.asarray(q["pose_w2c"], dtype=np.float32)):
            raise ValueError("Any query pose differs from frozen snapshot")
        if [camera.height, camera.width] != q["image_hw"]:
            raise ValueError("Any query dimensions changed")
        fx = camera.width / (2 * math.tan(camera.fov_x / 2))
        fy = camera.height / (2 * math.tan(camera.fov_y / 2))
        expected = np.asarray(q["intrinsic"], dtype=np.float32)
        actual = np.array([[fx, 0, camera.width/2], [0, fy, camera.height/2], [0, 0, 1]], dtype=np.float32)
        if not np.allclose(actual, expected, rtol=1e-6, atol=1e-4):
            raise ValueError("Any query intrinsics changed")

def preflight(contract_path, lease_path):
    contract_path = Path(contract_path)
    c = read(contract_path)
    root = Path(c["protocol"]["path"]).parent
    p = validate_round(c, root)
    review_path = root / "independent_review.json"
    check_review(read(review_path), c["protocol"])
    lease_ref = ref(lease_path)
    lease = read(lease_path)
    uuid = check_lease(lease, sha(contract_path), c["protocol"]["sha256"])
    qp = read(p["accuracy_protocol"]["path"])
    scene = qp["scenes"][c["cell"]["scene"]]
    verify(c["cell"]["accuracy_reference"])
    if "accuracy_results" in c["cell"]:
        verify(c["cell"]["accuracy_results"])
    verify(scene["reference_snapshot"])
    for value in scene["inputs"].values():
        verify(value)
    verify_scene_images(read(scene["reference_snapshot"]["path"]), scene["source"],
                        c["selection"]["measured_query_order"])
    if c["cell"]["family"] == "native":
        verify(c["cell"]["accuracy_contract"])
        ac = read(c["cell"]["accuracy_contract"]["path"])
        validate_accuracy(ac, qp, Path(p["accuracy_protocol"]["path"]))
        for path, digest in qp["code_sha256"].items():
            if sha(path) != digest:
                raise ValueError("native frozen code changed: " + path)
        for value in [*ac["model_assets"].values(), ac["wrapper"], ac["native_entry"], ac["native_config"]]:
            verify(value)
        executable = ac["executable"]
        env_updates = ac["env"]
    else:
        ap = read(p["any_protocol"]["path"])
        item = ap["scenes"][c["cell"]["scene"]]
        if c["cell"]["map"] != item["maps"][c["cell"]["variant"]]:
            raise ValueError("Any map detached from original benchmark")
        verify(c["cell"]["map"])
        for path, digest in ap["code_sha256"].items():
            full = Path(path) if Path(path).is_absolute() else REPO / path
            if sha(full) != digest:
                raise ValueError("Any frozen code changed")
        # Query bytes, masks, calibration and poses at the actual Any destination.
        verify(item["reference_snapshot"])
        any_snapshot = read(item["reference_snapshot"]["path"])
        if sha(item["scene_info"]["mask_path"]) != any_snapshot["mask_sha256"]:
            raise ValueError("Any precomputed mask archive changed")
        dataset = item["scene_info"]["dataset"]
        if isinstance(dataset, str):
            verify_scene_images(read(scene["reference_snapshot"]["path"]), dataset,
                                c["selection"]["measured_query_order"])
        else:
            raise ValueError("Any dataset layout requires explicit source binding")
        from anygsloc.data.datasets import ColmapDataset
        actual = ColmapDataset(dataset, load_masks=False)
        verify_any_cameras(actual.split("test"), any_snapshot, item["scene_info"]["test_names"])
        executable = "/root/miniconda3/envs/g4splat/bin/python"
        env_updates = {}
    runtime = dict(contract_path=str(contract_path.resolve()), contract_ref=ref(contract_path),
                   lease_ref=lease_ref, gpu_uuid=uuid, review=ref(review_path))
    return c, runtime, executable, env_updates

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--lease", type=Path, required=True)
    args = parser.parse_args()
    c, runtime, executable, updates = preflight(args.contract, args.lease)
    root = Path(c["protocol"]["path"]).parent
    lock = os.open(str(root / "physical_gpu.lock"), os.O_CREAT | os.O_RDWR, 0o600)
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    os.set_inheritable(lock, True)  # Kernel releases on normal exit, failure or Runner kill.
    before = sample()
    assert_idle(before, runtime["gpu_uuid"])
    verify(runtime["lease_ref"])
    check_lease(read(args.lease), runtime["contract_ref"]["sha256"], c["protocol"]["sha256"])
    verify(runtime["contract_ref"])
    output = Path(c["output"])
    output.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    env.update(updates)
    env.update(CUDA_VISIBLE_DEVICES=runtime["gpu_uuid"], PYTHONPATH=str(REPO) + os.pathsep + env.get("PYTHONPATH", ""),
               OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4")
    for key in ("ULFLOC_POSTPROCESS_PLY", "ULFLOC_POSTPROCESS_INDICES", "ANYGSLOC_DELEGATE_EXECUTOR"):
        env.pop(key, None)
    runtime.update(before=before, lock_fd=lock, process_mode="fresh process, one cell, one round",
                   executable=executable, runner_timeout_seconds=3000)
    write_new(output / "launch.json", runtime)
    env["MATCHED_LAUNCH_SHA256"] = sha(output / "launch.json")
    resource.setrlimit(resource.RLIMIT_NOFILE, nofile_pair(*resource.getrlimit(resource.RLIMIT_NOFILE)))
    command = [executable, "-m", "experiments.native_baseline_matched_benchmark.worker",
               "--contract", str(args.contract.resolve())]
    print(json.dumps(dict(case=c["case"], round=c["round"], command=command, gpu_uuid=runtime["gpu_uuid"])), flush=True)
    with (output / "native.log").open("xb", buffering=0) as stream:
        os.dup2(stream.fileno(), 1)
        os.dup2(stream.fileno(), 2)
        os.chdir(REPO)
        os.execvpe(executable, command, env)

if __name__ == "__main__":
    main()
