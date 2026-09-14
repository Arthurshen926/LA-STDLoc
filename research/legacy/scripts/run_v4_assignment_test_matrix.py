#!/usr/bin/env python3
"""Run one frozen official-test evaluation for the mapping-selected assignment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


PYTHON = Path("/root/miniconda3/envs/g4splat/bin/python")


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mapping-summary", type=Path, required=True)
    parser.add_argument("--source-matrix", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--code-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--gpus", default="0,1,2")
    parser.add_argument("--max-workers", type=int, default=3)
    args = parser.parse_args()
    summary = json.loads(args.mapping_summary.resolve().read_text())
    if (
        summary.get("schema") != "lafgs_v4_assignment_mapping_loo_summary"
        or summary.get("uses_test_queries") is not False
        or summary.get("authorizes_one_frozen_24_scene_test_matrix") is not True
    ):
        raise ValueError("mapping summary does not authorize official-test evaluation")
    variant = summary.get("selected_variant")
    if variant not in {"assignment_k4", "assignment_k8"}:
        raise ValueError("mapping-selected assignment variant is unsupported")
    topk = int(variant.rsplit("k", 1)[-1])
    source = json.loads(args.source_matrix.resolve().read_text())
    scenes = source.get("scenes", {})
    if len(scenes) != 24 or any(row.get("status") != "done" for row in scenes.values()):
        raise ValueError("source V4 matrix must contain 24 completed scenes")
    gpus = [item.strip() for item in args.gpus.split(",") if item.strip()]
    workers = min(max(int(args.max_workers), 1), len(gpus))
    if not gpus:
        raise ValueError("at least one GPU is required")
    code_root = args.code_root.resolve()
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    state_path = root / "matrix_state.json"
    state = (
        json.loads(state_path.read_text())
        if state_path.exists()
        else {
            "schema": "lafgs_v4_assignment_official_test_matrix",
            "version": 1,
            "mapping_summary": str(args.mapping_summary.resolve()),
            "selected_variant": variant,
            "assignment_topk": topk,
            "assignment_dustbin_score": -1.0,
            "seed": 2026,
            "jobs": {},
        }
    )
    jobs = []
    for key, scene in sorted(scenes.items()):
        family, name = key.split("/", 1)
        source_root = Path(scene["output"])
        result = json.loads((source_root / "single_seed_result.json").read_text())
        artifacts = result["artifacts"]
        job = {
            "key": key,
            "family": family,
            "scene": name,
            "dataset": str(Path(scene["dataset"]).resolve()),
            "map": str(Path(artifacts["training_map"]).resolve()),
            "metric": str(Path(artifacts["metric"]).resolve()),
            "scene_calibration": str(Path(artifacts["scene_calibration"]).resolve()),
        }
        for label in ("map", "metric", "scene_calibration"):
            path = Path(job[label])
            if not path.is_file():
                raise FileNotFoundError(path)
            job[f"{label}_sha256"] = _sha256(path)
        state["jobs"].setdefault(key, {**job, "status": "pending", "attempts": 0})
        jobs.append(job)
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
            output = root / job["family"] / job["scene"]
            summary_path = output / "summary.json"
            if output.exists() and not summary_path.is_file():
                raise RuntimeError(f"partial official-test output: {output}")
            if summary_path.is_file():
                state["jobs"][job["key"]].update(
                    {"status": "done", "output": str(output)}
                )
                _atomic_json(state_path, state)
                continue
            command = [
                str(PYTHON),
                "-u",
                "-m",
                "scripts.evaluate",
                "--dataset",
                job["dataset"],
                "--images",
                "processed",
                "--map",
                job["map"],
                "--metric-state",
                job["metric"],
                "--scene-calibration",
                job["scene_calibration"],
                "--output",
                str(output),
                "--split",
                "test",
                "--device",
                "cuda:0",
                "--seed",
                "2026",
                "--assignment-topk",
                str(topk),
                "--assignment-dustbin-score",
                "-1",
            ]
            log_path = root / job["family"] / f"{job['scene']}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log = log_path.open("ab")
            env = dict(os.environ)
            env.update({"PYTHONPATH": str(code_root), "CUDA_VISIBLE_DEVICES": gpu})
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
                    "pid": process.pid,
                    "gpu": gpu,
                    "attempts": int(state["jobs"][job["key"]].get("attempts", 0)) + 1,
                    "command": command,
                    "output": str(output),
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
            complete = (output / "summary.json").is_file()
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
        raise SystemExit(f"official-test matrix completed with failures: {failures}")


if __name__ == "__main__":
    main()
