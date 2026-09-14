#!/usr/bin/env python3
"""Run nested compact/16K/20K pure-sparse capacity factors."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil
import subprocess
import sys

import torch

from common.config import load_mainline_config
from map_learning.pipeline import resolve_prior_ply, train_compact_map


CAPACITY_SPECS = "16000:8000:broad,20000:10000:broad"


def _run(*arguments: object) -> None:
    subprocess.run([str(value) for value in arguments], check=True)


def _factor_config(root: Path, supplied: Path | None) -> Path:
    if supplied is not None:
        return supplied.resolve()
    matches = sorted(root.glob("factor_config_k*.yaml"))
    if len(matches) != 1:
        raise ValueError(
            f"Expected one materialized keypoint config in {root}, found {matches}"
        )
    return matches[0].resolve()


def _manifest(root: Path) -> dict[str, Path]:
    path = root / "pipeline_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        key: Path(value).expanduser().resolve()
        for key, value in json.loads(path.read_text()).items()
    }


def _assert_nested(first: Path, second: Path) -> dict:
    states = [
        torch.load(path, map_location="cpu", weights_only=False)
        for path in (first, second)
    ]
    tracks = [
        set(
            torch.as_tensor(state["track_centric_reconstruction"]["track_indices"])
            .long()
            .tolist()
        )
        for state in states
    ]
    bases = [
        set(
            torch.as_tensor(
                state["track_centric_reconstruction"]["base_canonical_rows"]
            )
            .long()
            .tolist()
        )
        for state in states
    ]
    report = {
        "track_16k_subset_of_20k": tracks[0].issubset(tracks[1]),
        "base_16k_subset_of_20k": bases[0].issubset(bases[1]),
        "track_counts": [len(value) for value in tracks],
        "base_counts": [len(value) for value in bases],
    }
    if not all(report[key] for key in report if key.endswith("subset_of_20k")):
        raise RuntimeError(f"Capacity maps are not nested: {report}")
    return report


def _reuse_pipeline_evaluation(source: Path, destination: Path) -> bool:
    required = ("summary.json", "results.json", "deployment_contract.json")
    if not all((source / name).is_file() for name in required):
        return False
    destination.mkdir(parents=True, exist_ok=True)
    for name in required:
        shutil.copy2(source / name, destination / name)
    return True


def _evaluate(
    *,
    dataset: Path,
    map_path: Path,
    metric_state: Path,
    calibration: Path,
    config: Path,
    output: Path,
    device: str,
    seeds: list[int],
    workers: int,
) -> dict:
    def evaluate_seed(seed: int) -> tuple[str, dict]:
        seed_output = output / f"evaluation_seed{seed}"
        summary = seed_output / "summary.json"
        if not summary.is_file():
            _run(
                sys.executable,
                "scripts/evaluate.py",
                "--dataset",
                dataset,
                "--map",
                map_path,
                "--metric-state",
                metric_state,
                "--scene-calibration",
                calibration,
                "--output",
                seed_output,
                "--config",
                config,
                "--device",
                device,
                "--seed",
                seed,
            )
        contract = json.loads(
            (seed_output / "deployment_contract.json").read_text()
        )
        if (
            contract["pose_solves"] != 1
            or contract.get("duplicate_anchor_suppression", False)
            or contract.get("guided_sampling", False)
        ):
            raise RuntimeError(f"Non-baseline sparse deployment: {contract}")
        return str(seed), json.loads(summary.read_text())

    # PoseLib evaluation is CPU-heavy and keeps only a small SuperPoint model on
    # the GPU. Independent seeds can therefore share one GPU without changing
    # either their random state or output contract.
    max_workers = min(max(int(workers), 1), len(seeds))
    if max_workers == 1:
        evaluated = [evaluate_seed(seed) for seed in seeds]
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            evaluated = list(executor.map(evaluate_seed, seeds))
    return dict(evaluated)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline-root", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--prior", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--gaussian-type", choices=("2dgs", "3dgs"), required=True)
    parser.add_argument("--sh-degree", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seeds", default="2026,2027,2028")
    parser.add_argument("--evaluation-workers", type=int, default=3)
    parser.add_argument("--provenance-shards", type=int, default=4)
    parser.add_argument("--observation-shards", type=int, default=4)
    args = parser.parse_args()

    root = args.pipeline_root.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = _factor_config(root, args.config)
    values = load_mainline_config(config).values
    deployment = values["deployment"]
    if (
        deployment["global_topk"] != 1
        or deployment["max_matches_per_keypoint"] != 0
        or deployment["max_matches_per_landmark"] != 0
        or deployment["pose_solves"] != 1
        or deployment["solver"] != "poselib"
    ):
        raise ValueError("Factor matrix requires the frozen pure-sparse contract")

    artifacts = _manifest(root)
    calibration = json.loads(artifacts["scene_calibration"].read_text())
    parameters = calibration["parameters"]
    capacity_dir = output / "capacity_maps"
    report_path = capacity_dir / "track_centric_build.json"
    if not report_path.is_file():
        _run(
            sys.executable,
            "-m",
            "topology.track_core",
            "--canonical-map",
            artifacts["canonical_map"],
            "--function-graph",
            artifacts["function_graph"],
            "--track-payload",
            artifacts["track_payload"],
            "--query-cache",
            artifacts["query_cache"],
            "--output-dir",
            capacity_dir,
            "--specs",
            CAPACITY_SPECS,
            "--base-selection",
            "group_balanced",
            "--base-voxel-size",
            parameters["base_voxel_m"],
            "--dependency-voxel-size",
            parameters["dependency_voxel_m"],
            "--descriptor-trim-fraction",
            values["adaptive"]["descriptor_trim_fraction"],
        )
    build = json.loads(report_path.read_text())
    by_budget = {
        int(key.split("_", 1)[0][1:]): Path(value["path"]).resolve()
        for key, value in build["maps"].items()
    }
    if set(by_budget) != {16000, 20000}:
        raise RuntimeError(f"Unexpected capacity maps: {by_budget}")
    nestedness = _assert_nested(by_budget[16000], by_budget[20000])
    (output / "capacity_nestedness.json").write_text(
        json.dumps(nestedness, indent=2, sort_keys=True) + "\n"
    )

    seeds = [int(value) for value in args.seeds.split(",") if value]
    result = {"compact": {}, "16000": {}, "20000": {}}
    reused_pipeline_seed = None
    if 2026 in seeds and "evaluation" in artifacts:
        if _reuse_pipeline_evaluation(
            artifacts["evaluation"], output / "compact" / "evaluation_seed2026"
        ):
            reused_pipeline_seed = 2026
    result["compact"] = _evaluate(
        dataset=args.dataset.resolve(),
        map_path=artifacts["trained_map"],
        metric_state=artifacts["metric_state"],
        calibration=artifacts["scene_calibration"],
        config=config,
        output=output / "compact",
        device=args.device,
        seeds=seeds,
        workers=args.evaluation_workers,
    )
    prior_ply = resolve_prior_ply(args.prior)
    for budget in (16000, 20000):
        branch = output / f"map_{budget:05d}"
        trained = train_compact_map(
            compact_map=by_budget[budget],
            function_graph=artifacts["function_graph"],
            track_payload=artifacts["track_payload"],
            query_cache=artifacts["query_cache"],
            prior_ply=prior_ply,
            gaussian_type=args.gaussian_type,
            sh_degree=args.sh_degree,
            output=branch / "map_learning",
            config=config,
            provenance_shards=args.provenance_shards,
            observation_shards=args.observation_shards,
            scene_calibration=artifacts["scene_calibration"],
        )
        result[str(budget)] = _evaluate(
            dataset=args.dataset.resolve(),
            map_path=trained["trained_map"],
            metric_state=trained["metric_state"],
            calibration=Path(trained["trained_map"]).parent / "scene_calibration.json",
            config=config,
            output=branch,
            device=args.device,
            seeds=seeds,
            workers=args.evaluation_workers,
        )
    payload = {
        "schema": "lafgs_pure_sparse_keypoint_capacity_factor",
        "version": 1,
        "keypoints": int(deployment["keypoints"]),
        "capacity_specs": CAPACITY_SPECS,
        "nestedness": nestedness,
        "group_dro_max_weight_ratio": float(
            values["reconstruction"]["group_dro_max_weight_ratio"]
        ),
        "reused_pipeline_evaluation_seed": reused_pipeline_seed,
        "results": result,
    }
    (output / "factor_matrix.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
