#!/usr/bin/env python3
"""Run the frozen eight-scene assignment rejection panel efficiently."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess

from scripts.run_v4_assignment_mapping_matrix import PYTHON, _scene_inputs, _sha256


def common_arguments(scene_root: Path) -> list[str]:
    inputs = _scene_inputs(scene_root)
    return [
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
        "--descriptor-trim-fraction",
        "0.2",
    ]


def run_checked(command: list[str], *, code_root: Path, log: Path, env: dict) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("ab") as stream:
        result = subprocess.run(
            command,
            cwd=code_root,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if result.returncode != 0:
        raise RuntimeError(f"panel command failed ({result.returncode}): {log}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-matrix", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--code-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--preregistration",
        type=Path,
        default=Path(
            "docs/evidence/v4_assignment_rejection_panel_preregistration.json"
        ),
    )
    parser.add_argument("--gpus", default="0,1,2")
    parser.add_argument("--replay-workers", type=int, default=2)
    parser.add_argument("--candidate-workers", type=int, default=4)
    parser.add_argument("--pose-workers", type=int, default=4)
    parser.add_argument(
        "--reuse-sidecar",
        action="append",
        default=[],
        metavar="FAMILY/SCENE=PATH",
        help="Reuse an exact source-validated sidecar instead of rematerializing it.",
    )
    args = parser.parse_args()
    code_root = args.code_root.resolve()
    root = args.output_root.resolve()
    prereg_path = args.preregistration.resolve()
    prereg = json.loads(prereg_path.read_text())
    source = json.loads(args.source_matrix.resolve().read_text())
    if prereg.get("uses_test_queries") is not False:
        raise ValueError("panel preregistration must be mapping-only")
    scenes = list(prereg["hard_scenes"])
    source_scenes = source.get("scenes", {})
    if any(
        scene not in source_scenes or source_scenes[scene].get("status") != "done"
        for scene in scenes
    ):
        raise ValueError("panel source scenes are not all complete")
    for scene in scenes:
        family, name = scene.split("/", 1)
        baseline = (
            args.baseline_root.resolve()
            / family
            / name
            / "top1"
            / "full_mapping_loo_report.json"
        )
        if not baseline.is_file():
            raise FileNotFoundError(f"panel misses frozen Top-1 baseline: {baseline}")
    if root.exists():
        raise FileExistsError(root)
    root.mkdir(parents=True)
    (root / "preregistration.json").write_bytes(prereg_path.read_bytes())
    gpus = [value.strip() for value in args.gpus.split(",") if value.strip()]
    if not gpus:
        raise ValueError("panel needs at least one GPU for sidecar materialization")
    common_by_scene = {
        scene: common_arguments(Path(source_scenes[scene]["output"]))
        for scene in scenes
    }
    reusable = {}
    for specification in args.reuse_sidecar:
        scene, separator, value = specification.partition("=")
        if not separator or scene not in scenes or scene in reusable:
            raise ValueError(f"invalid reusable sidecar specification: {specification}")
        path = Path(value).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        reusable[scene] = path
    sidecar_by_scene = {
        scene: reusable.get(
            scene,
            root
            / scene.split("/", 1)[0]
            / scene.split("/", 1)[1]
            / "mapping_loo_top8.pt",
        )
        for scene in scenes
    }

    def materialize_partition(gpu: str, assigned: list[str]) -> None:
        for scene in assigned:
            if scene in reusable:
                continue
            family, name = scene.split("/", 1)
            sidecar = sidecar_by_scene[scene]
            command = [
                str(PYTHON),
                "-u",
                "-m",
                "scripts.materialize_v4_mapping_topk_sidecar",
                *common_by_scene[scene],
                "--topk",
                str(prereg["shared_topk"]),
                "--output",
                str(sidecar),
                "--device",
                "cuda:0",
                "--cpu-threads",
                "2",
            ]
            env = dict(os.environ)
            env.update(
                {
                    "PYTHONPATH": str(code_root),
                    "CUDA_VISIBLE_DEVICES": gpu,
                    "OMP_NUM_THREADS": "2",
                    "MKL_NUM_THREADS": "2",
                }
            )
            run_checked(
                command,
                code_root=code_root,
                log=root / family / name / "mapping_loo_top8.log",
                env=env,
            )

    partitions = [scenes[index :: len(gpus)] for index in range(len(gpus))]
    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        futures = [
            pool.submit(materialize_partition, gpu, partition)
            for gpu, partition in zip(gpus, partitions)
            if partition
        ]
        for future in futures:
            future.result()

    for scene in scenes:
        family, name = scene.split("/", 1)
        sidecar = sidecar_by_scene[scene]
        run_checked(
            [
                str(PYTHON),
                "-u",
                "-m",
                "scripts.audit_v4_mapping_topk_headroom",
                "--sidecar",
                str(sidecar),
                "--expected-sidecar-sha256",
                _sha256(sidecar),
                "--output",
                str(root / family / name / "topk_headroom.json"),
                "--cpu-threads",
                "2",
            ],
            code_root=code_root,
            log=root / family / name / "topk_headroom.log",
            env={
                **os.environ,
                "PYTHONPATH": str(code_root),
                "CUDA_VISIBLE_DEVICES": "",
            },
        )

    def replay_partition(assigned: list[str]) -> None:
        for scene in assigned:
            family, name = scene.split("/", 1)
            sidecar = sidecar_by_scene[scene]
            command = [
                str(PYTHON),
                "-u",
                "-m",
                "scripts.evaluate_v4_mapping_topk_candidates",
                *common_by_scene[scene],
                "--sidecar",
                str(sidecar),
                "--expected-sidecar-sha256",
                _sha256(sidecar),
                "--preregistration",
                str(root / "preregistration.json"),
                "--output-root",
                str(root / family / name / "candidate_batch"),
                "--candidate-workers",
                str(args.candidate_workers),
                "--pose-workers-per-candidate",
                str(args.pose_workers),
                "--device",
                "cpu",
                "--cpu-threads",
                "1",
            ]
            env = dict(os.environ)
            env.update(
                {
                    "PYTHONPATH": str(code_root),
                    "CUDA_VISIBLE_DEVICES": "",
                    "OMP_NUM_THREADS": "1",
                    "MKL_NUM_THREADS": "1",
                }
            )
            run_checked(
                command,
                code_root=code_root,
                log=root / family / name / "candidate_batch.log",
                env=env,
            )

    replay_workers = min(max(int(args.replay_workers), 1), len(scenes))
    replay_partitions = [
        scenes[index::replay_workers] for index in range(replay_workers)
    ]
    with ThreadPoolExecutor(max_workers=replay_workers) as pool:
        futures = [
            pool.submit(replay_partition, partition) for partition in replay_partitions
        ]
        for future in futures:
            future.result()

    summary_path = root / "summary.json"
    subprocess.run(
        [
            str(PYTHON),
            "-m",
            "scripts.summarize_v4_assignment_rejection_panel",
            "--panel-root",
            str(root),
            "--baseline-root",
            str(args.baseline_root.resolve()),
            "--preregistration",
            str(root / "preregistration.json"),
            "--output",
            str(summary_path),
        ],
        cwd=code_root,
        env={**os.environ, "PYTHONPATH": str(code_root)},
        check=True,
    )


if __name__ == "__main__":
    main()
