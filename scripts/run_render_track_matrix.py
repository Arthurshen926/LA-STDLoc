#!/usr/bin/env python3
"""Resumable one-seed V4 rendered-Track benchmark over Cambridge/7Scenes/12Scenes."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time


SCENES = [
    *[("Cambridge", name, "/mnt/pool/sqy/Cambridge_stdloc/{name}", "/mnt/pool/sqy/stdloc_lafgs_v1_frozen_multiscene_20260731/{name}/prior/rgb_matcha_2dgs/point_cloud/iteration_30000/point_cloud.ply") for name in ("GreatCourt", "KingsCollege", "OldHospital", "ShopFacade", "StMarysChurch")],
    *[("7Scenes", name, "/mnt/pool/sqy/datasets/7Scenes_pgt_full_reference_v5/{name}", "/mnt/pool/sqy/datasets/7Scenes_pgt_full_reference_v5/{name}/lafgs_prior_v1/point_cloud/iteration_30000/point_cloud.ply") for name in ("chess", "fire", "heads", "office", "pumpkin", "redkitchen", "stairs")],
    *[("12Scenes", name, "/mnt/pool/sqy/datasets/12Scenes_pgt_full_reference_v5/{name}", "/mnt/pool/sqy/datasets/12Scenes_pgt_full_reference_v5/{name}/lafgs_prior_v1/point_cloud/iteration_30000/point_cloud.ply") for name in ("apt1_kitchen", "apt1_living", "apt2_bed", "apt2_kitchen", "apt2_living", "apt2_luke", "office1_gates362", "office1_gates381", "office1_lounge", "office1_manolis", "office2_5a", "office2_5b")],
]


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def scene_records(args: argparse.Namespace) -> list[dict]:
    selected = set(args.only.split(",")) if args.only else None
    records = []
    for family, name, dataset, prior in SCENES:
        key = f"{family}/{name}"
        if selected and key not in selected and name not in selected:
            continue
        records.append(
            {
                "family": family,
                "scene": name,
                "key": key,
                "dataset": dataset.format(name=name),
                "prior": prior.format(name=name),
            }
        )
    if not records:
        raise SystemExit("--only did not match a known scene")
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--code-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--gpus", default="1,2")
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument("--only", default="")
    parser.add_argument(
        "--output-override",
        action="append",
        default=[],
        metavar="KEY=PATH",
        help="Use an existing/resumable scene root for one key (for example Cambridge/KingsCollege=/mnt/pool/...).",
    )
    args = parser.parse_args()
    args.output_root = args.output_root.expanduser().resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)
    gpus = [item.strip() for item in args.gpus.split(",") if item.strip()]
    if not gpus:
        raise SystemExit("at least one GPU is required")
    workers = max(1, min(int(args.max_workers), len(gpus)))
    records = scene_records(args)
    overrides = {}
    for item in args.output_override:
        if "=" not in item:
            raise SystemExit(f"invalid --output-override {item!r}; expected KEY=PATH")
        key, value = item.split("=", 1)
        overrides[key] = str(Path(value).expanduser().resolve())
    state_path = args.output_root / "matrix_state.json"
    state = {}
    if state_path.exists():
        state = json.loads(state_path.read_text())
    state.setdefault("schema", "lafgs_v4_render_track_matrix_state")
    state.setdefault("version", 1)
    state.setdefault("seed", int(args.seed))
    state.setdefault("code_root", str(args.code_root.resolve()))
    state.setdefault("scenes", {})
    for record in records:
        state["scenes"].setdefault(record["key"], {**record, "status": "pending", "attempts": 0})
    atomic_json(state_path, state)

    pending = [r for r in records if state["scenes"][r["key"]].get("status") != "done"]
    active: dict[int, tuple[dict, subprocess.Popen, object]] = {}
    next_index = 0
    while pending or active:
        while pending and len(active) < workers:
            record = pending.pop(0)
            key = record["key"]
            scene_root = Path(
                overrides.get(
                    record["key"],
                    str(args.output_root / record["family"] / record["scene"]),
                )
            )
            scene_root.mkdir(parents=True, exist_ok=True)
            gpu = gpus[len(active) % len(gpus)]
            log_path = scene_root / "matrix_worker.stdout.log"
            log = log_path.open("ab")
            command = [
                "/root/miniconda3/envs/g4splat/bin/python", "-u",
                str(args.code_root.resolve() / "scripts/run_render_track_single_scene.py"),
                "--scene", record["scene"], "--dataset", record["dataset"], "--prior", record["prior"],
                "--output", str(scene_root), "--code-root", str(args.code_root.resolve()),
                "--gpu", gpu, "--seed", str(args.seed),
            ]
            state["scenes"][key].update({"status": "running", "gpu": gpu, "attempts": int(state["scenes"][key].get("attempts", 0)) + 1, "command": command, "started": time.time()})
            state["scenes"][key]["output"] = str(scene_root)
            atomic_json(state_path, state)
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, cwd=args.code_root.resolve(), start_new_session=True)
            active[process.pid] = (record, process, log)
            next_index += 1
        time.sleep(2.0)
        for pid, (record, process, log) in list(active.items()):
            returncode = process.poll()
            if returncode is None:
                continue
            log.close()
            key = record["key"]
            result_path = Path(state["scenes"][key].get("output", str(args.output_root / key.split("/")[0] / record["scene"]))) / "single_seed_result.json"
            state["scenes"][key].update({"status": "done" if returncode == 0 and result_path.exists() else "failed", "returncode": returncode, "finished": time.time()})
            atomic_json(state_path, state)
            del active[pid]
    state["finished"] = time.time()
    atomic_json(state_path, state)
    failed = [key for key, value in state["scenes"].items() if value.get("status") != "done"]
    if failed:
        raise SystemExit(f"matrix completed with failures: {failed}")


if __name__ == "__main__":
    main()
