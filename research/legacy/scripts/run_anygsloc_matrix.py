#!/usr/bin/env python3
"""Audit or execute the frozen AnyGSLoc paper experiment matrix."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any

from common.config import ANYGSLOC_SCHEMA, load_mainline_config


PYTHON = Path("/root/miniconda3/envs/g4splat/bin/python")


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def complete_cell(cell: dict[str, Any]) -> dict[str, Any]:
    cell = dict(cell)
    family, scene = cell["family"], cell["scene"]
    if family == "7Scenes":
        root = Path("/mnt/pool/sqy/datasets/7Scenes_pgt_full_reference_v5") / scene
        cell.update(
            dataset=str(root),
            gaussian_ply=str(root / "lafgs_prior_v1/point_cloud/iteration_30000/point_cloud.ply"),
            gaussian_type="2dgs",
            sh_degree=3,
        )
    elif family == "12Scenes":
        root = Path("/mnt/pool/sqy/datasets/12Scenes_pgt_full_reference_v5") / scene
        cell.update(
            dataset=str(root),
            gaussian_ply=str(root / "lafgs_prior_v1/point_cloud/iteration_30000/point_cloud.ply"),
            gaussian_type="2dgs",
            sh_degree=3,
        )
    cell["key"] = f"{family}/{scene}"
    return cell


def load_cells(matrix: Path, group: str, only: set[str]) -> tuple[dict, list[dict]]:
    payload = json.loads(matrix.read_text())
    if payload.get("schema") != "anygsloc_paper_experiment_matrix":
        raise ValueError("unsupported AnyGSLoc experiment matrix")
    if any(payload["scope"].get(key) is not False for key in (
        "offline_self_localization_feedback",
        "descriptor_or_metric_training",
        "test_query_map_adaptation",
    )):
        raise ValueError("experiment matrix enables a forbidden feedback mechanism")
    config_path = Path(payload["formal_config"])
    if not config_path.is_absolute():
        config_path = (matrix.resolve().parents[1] / config_path).resolve()
    if load_mainline_config(config_path).values["schema"] != ANYGSLOC_SCHEMA:
        raise ValueError("matrix does not bind the AnyGSLoc formal config")
    cells = [complete_cell(cell) for cell in payload[group]]
    if only:
        cells = [cell for cell in cells if cell["key"] in only or cell["scene"] in only]
    if not cells:
        raise ValueError("no experiment cells selected")
    keys = [cell["key"] for cell in cells]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate experiment cell")
    payload["resolved_config"] = str(config_path)
    return payload, cells


def audit_cell(cell: dict[str, Any]) -> dict[str, Any]:
    dataset = Path(cell["dataset"])
    source = Path(cell.get("prior_manifest", cell.get("gaussian_ply", "")))
    errors = []
    if not dataset.is_dir():
        errors.append(f"missing dataset: {dataset}")
    if not source.is_file():
        errors.append(f"missing prior: {source}")
    existing = cell.get("existing_base")
    existing_complete = False
    if existing:
        existing_root = Path(existing)
        existing_complete = all(
            path.is_file()
            for path in (
                existing_root / "projective_map/projective_anchor_map.pt",
                existing_root / "projective_map/identity_metric.pt",
                existing_root / "projective_map/report.json",
                existing_root / "evaluation/baseline_seed2026/summary.json",
            )
        )
    return {**cell, "input_errors": errors, "existing_base_complete": existing_complete}


def scene_command(cell: dict[str, Any], *, output_root: Path, config: Path, gpu: str) -> list[str]:
    command = [
        str(PYTHON), "-u", "-m", "scripts.run_anygsloc_scene",
        "--scene", cell["scene"], "--dataset", cell["dataset"],
        "--output", str(output_root / cell["family"] / cell["scene"]),
        "--config", str(config), "--gpu", gpu,
    ]
    if "prior_manifest" in cell:
        command.extend(("--prior-manifest", cell["prior_manifest"]))
    else:
        command.extend(
            (
                "--gaussian-ply", cell["gaussian_ply"],
                "--gaussian-type", cell["gaussian_type"],
                "--sh-degree", str(cell["sh_degree"]),
            )
        )
        if cell.get("white_background", False):
            command.append("--white-background")
    return command


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, default=Path("configs/anygsloc_experiments.json"))
    parser.add_argument("--group", choices=("primary_24_scene", "prior_robustness"), default="primary_24_scene")
    parser.add_argument("--only", default="")
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--reuse-existing-base", action="store_true")
    args = parser.parse_args()
    selected = {item for item in args.only.split(",") if item}
    payload, cells = load_cells(args.matrix.resolve(), args.group, selected)
    audited = [audit_cell(cell) for cell in cells]
    output_root = Path(payload["output_root"]).resolve()
    selection_identity = ",".join(sorted(row["key"] for row in audited))
    selection_tag = (
        "all"
        if not selected
        else hashlib.sha256(selection_identity.encode()).hexdigest()[:12]
    )
    audit_path = output_root / f"{args.group}_{selection_tag}_input_audit.json"
    audit = {
        "schema": "anygsloc_experiment_input_audit",
        "version": 1,
        "group": args.group,
        "selection_identity": selection_identity,
        "selection_tag": selection_tag,
        "cell_count": len(audited),
        "valid_cell_count": sum(not row["input_errors"] for row in audited),
        "cells": audited,
    }
    atomic_json(audit_path, audit)
    print(json.dumps(audit, indent=2, sort_keys=True))
    invalid = [row["key"] for row in audited if row["input_errors"]]
    if invalid:
        raise SystemExit(f"invalid experiment inputs: {invalid}")
    if args.audit_only:
        return
    gpus = [item for item in args.gpus.split(",") if item]
    workers = min(max(1, args.max_workers), len(gpus))
    pending = [
        row for row in audited
        if not (args.reuse_existing_base and row["existing_base_complete"])
    ]
    state_path = output_root / f"{args.group}_{selection_tag}_state.json"
    state = {
        "schema": "anygsloc_experiment_state",
        "version": 1,
        "selection_identity": selection_identity,
        "selection_tag": selection_tag,
        "started": time.time(),
        "cells": {},
    }

    def execute(index_cell: tuple[int, dict[str, Any]]) -> tuple[str, int, list[str]]:
        index, cell = index_cell
        gpu = gpus[index % len(gpus)]
        command = scene_command(cell, output_root=output_root, config=Path(payload["resolved_config"]), gpu=gpu)
        result = subprocess.run(command, cwd=Path(__file__).resolve().parents[1])
        return cell["key"], result.returncode, command

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(execute, pair) for pair in enumerate(pending)]
        for future in as_completed(futures):
            key, returncode, command = future.result()
            state["cells"][key] = {"returncode": returncode, "command": command}
            atomic_json(state_path, state)
    for row in audited:
        if args.reuse_existing_base and row["existing_base_complete"]:
            state["cells"][row["key"]] = {"status": "reused_existing_base", "path": row["existing_base"]}
    state["finished"] = time.time()
    atomic_json(state_path, state)
    failed = [key for key, value in state["cells"].items() if value.get("returncode", 0)]
    if failed:
        raise SystemExit(f"experiment cells failed: {failed}")


if __name__ == "__main__":
    main()
