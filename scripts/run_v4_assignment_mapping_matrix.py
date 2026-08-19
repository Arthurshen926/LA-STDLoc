#!/usr/bin/env python3
"""Run the frozen V4 full-mapping LOO assignment audit on all 24 scenes.

This coordinator never opens the test split.  Each job replays the same compact
map and leave-one-query-out descriptor bank with either the deployed independent
Top-1 rule or one globally shared capacity-feasible Top-K rule.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


PYTHON = Path("/root/miniconda3/envs/g4splat/bin/python")
VARIANTS = {
    "top1": (),
    "assignment_k4": ("--assignment-topk", "4", "--assignment-dustbin-score", "-1"),
    "assignment_k8": ("--assignment-topk", "8", "--assignment-dustbin-score", "-1"),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _scene_inputs(scene_root: Path) -> dict[str, Path]:
    result_path = scene_root / "single_seed_result.json"
    result = json.loads(result_path.read_text())
    artifacts = result["artifacts"]
    paths = {
        "map": Path(artifacts["training_map"]).resolve(),
        "metric": Path(artifacts["metric"]).resolve(),
        "track_payload": Path(artifacts["repaired_track"]).resolve(),
        "teacher": (
            scene_root
            / "surface_unified_training"
            / "rendered_track_positive_teacher.pt"
        ).resolve(),
        "query_cache": Path(artifacts["appearance_cache"]).resolve(),
        "scene_calibration": Path(artifacts["scene_calibration"]).resolve(),
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"scene misses frozen assignment inputs: {missing}")
    return paths


def _jobs(source_state: dict, only: set[str] | None) -> list[dict]:
    scenes = source_state.get("scenes", {})
    if len(scenes) != 24 or any(row.get("status") != "done" for row in scenes.values()):
        raise ValueError("source V4 matrix must contain 24 completed scenes")
    output = []
    for variant in VARIANTS:
        for key in sorted(scenes):
            if only and key not in only and key.split("/", 1)[1] not in only:
                continue
            source_root = Path(scenes[key]["output"]).resolve()
            output.append(
                {
                    "key": f"{key}/{variant}",
                    "scene_key": key,
                    "family": key.split("/", 1)[0],
                    "scene": key.split("/", 1)[1],
                    "variant": variant,
                    "source_root": str(source_root),
                }
            )
    if not output:
        raise ValueError("no assignment matrix jobs selected")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-matrix", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--code-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--gpus", default="0,1,2")
    parser.add_argument("--max-workers", type=int, default=3)
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--only", default="")
    args = parser.parse_args()

    code_root = args.code_root.resolve()
    source_state = json.loads(args.source_matrix.resolve().read_text())
    selected = {item for item in args.only.split(",") if item} or None
    jobs = _jobs(source_state, selected)
    gpus = [item.strip() for item in args.gpus.split(",") if item.strip()]
    workers = min(max(int(args.max_workers), 1), len(gpus))
    if not gpus:
        raise ValueError("at least one GPU is required")
    if int(args.cpu_threads) <= 0:
        raise ValueError("CPU thread count must be positive")

    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    state_path = root / "matrix_state.json"
    state = (
        json.loads(state_path.read_text())
        if state_path.exists()
        else {
            "schema": "lafgs_v4_assignment_mapping_loo_matrix",
            "version": 1,
            "uses_test_queries": False,
            "source_matrix": str(args.source_matrix.resolve()),
            "code_root": str(code_root),
            "variants": {
                name: {"arguments": list(arguments)}
                for name, arguments in VARIANTS.items()
            },
            "jobs": {},
        }
    )
    for job in jobs:
        state["jobs"].setdefault(
            job["key"], {**job, "status": "pending", "attempts": 0}
        )
    _atomic_json(state_path, state)

    pending = [job for job in jobs if state["jobs"][job["key"]].get("status") != "done"]
    active: dict[int, tuple[dict, subprocess.Popen, object, str]] = {}
    while pending or active:
        while pending and len(active) < workers:
            occupied = {entry[3] for entry in active.values()}
            gpu = next((value for value in gpus if value not in occupied), None)
            if gpu is None:
                break
            job = pending.pop(0)
            source_root = Path(job["source_root"])
            inputs = _scene_inputs(source_root)
            output = root / job["family"] / job["scene"] / job["variant"]
            report = output / "full_mapping_loo_report.json"
            if output.exists() and not report.is_file():
                raise RuntimeError(f"partial job output must be quarantined: {output}")
            if report.is_file():
                state["jobs"][job["key"]].update(
                    {"status": "done", "output": str(output)}
                )
                _atomic_json(state_path, state)
                continue
            arguments = [
                "--map",
                str(inputs["map"]),
                "--expected-map-sha256",
                _sha256(inputs["map"]),
                "--metric-state",
                str(inputs["metric"]),
                "--expected-metric-sha256",
                _sha256(inputs["metric"]),
                "--track-payload",
                str(inputs["track_payload"]),
                "--expected-track-payload-sha256",
                _sha256(inputs["track_payload"]),
                "--teacher",
                str(inputs["teacher"]),
                "--expected-teacher-sha256",
                _sha256(inputs["teacher"]),
                "--query-cache",
                str(inputs["query_cache"]),
                "--expected-query-cache-sha256",
                _sha256(inputs["query_cache"]),
                "--scene-calibration",
                str(inputs["scene_calibration"]),
                "--expected-scene-calibration-sha256",
                _sha256(inputs["scene_calibration"]),
                "--output-dir",
                str(output),
                "--descriptor-trim-fraction",
                "0.2",
                "--seed",
                "2026",
                "--device",
                "cuda:0",
                "--cpu-threads",
                str(args.cpu_threads),
                *VARIANTS[job["variant"]],
            ]
            command = [
                str(PYTHON),
                "-u",
                "-m",
                "scripts.evaluate_rendered_track_fullmap",
                *arguments,
            ]
            log_path = root / job["family"] / job["scene"] / f"{job['variant']}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log = log_path.open("ab")
            env = dict(os.environ)
            env.update(
                {
                    "PYTHONPATH": str(code_root),
                    "CUDA_VISIBLE_DEVICES": gpu,
                    "OMP_NUM_THREADS": str(args.cpu_threads),
                    "MKL_NUM_THREADS": str(args.cpu_threads),
                    "PYTHONUNBUFFERED": "1",
                }
            )
            process = subprocess.Popen(
                command,
                cwd=code_root,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            state["jobs"][job["key"]].update(
                {
                    "status": "running",
                    "attempts": int(state["jobs"][job["key"]].get("attempts", 0)) + 1,
                    "gpu": gpu,
                    "pid": process.pid,
                    "output": str(output),
                    "command": command,
                    "started": time.time(),
                }
            )
            _atomic_json(state_path, state)
            active[process.pid] = (job, process, log, gpu)
        if not active:
            continue
        time.sleep(2.0)
        for pid, (job, process, log, _gpu) in list(active.items()):
            returncode = process.poll()
            if returncode is None:
                continue
            log.close()
            output = Path(state["jobs"][job["key"]]["output"])
            complete = (output / "full_mapping_loo_report.json").is_file()
            state["jobs"][job["key"]].update(
                {
                    "status": "done" if returncode == 0 and complete else "failed",
                    "returncode": int(returncode),
                    "pid": None,
                    "finished": time.time(),
                }
            )
            _atomic_json(state_path, state)
            del active[pid]
    failures = [
        key for key, row in state["jobs"].items() if row.get("status") != "done"
    ]
    state["finished"] = time.time()
    _atomic_json(state_path, state)
    if failures:
        raise SystemExit(f"assignment matrix completed with failures: {failures}")


if __name__ == "__main__":
    main()
