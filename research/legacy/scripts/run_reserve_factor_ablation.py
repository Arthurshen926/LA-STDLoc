#!/usr/bin/env python3
"""Train/evaluate causal Track/Gaussian reserve factors with one protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import torch

from map_learning.pipeline import train_compact_map


def _run(*arguments: object) -> None:
    subprocess.run([str(value) for value in arguments], check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline-root", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--gaussian-type", choices=("2dgs", "3dgs"), required=True)
    parser.add_argument("--sh-degree", type=int, default=3)
    parser.add_argument(
        "--factors", default="track_only_final,all_coverage,full"
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--provenance-shards", type=int, default=4)
    parser.add_argument("--observation-shards", type=int, default=4)
    parser.add_argument(
        "--fixed-metric-gate-only",
        action="store_true",
        help="Hold the learned descriptor metric fixed to isolate topology causality.",
    )
    parser.add_argument("--splits", default="mapping,test")
    parser.add_argument(
        "--materialize-only",
        action="store_true",
        help="Record selection provenance and factor maps without evaluating them.",
    )
    args = parser.parse_args()

    root = args.pipeline_root.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    artifacts = {
        key: Path(value).resolve()
        for key, value in json.loads(
            (root / "pipeline_manifest.json").read_text()
        ).items()
    }
    topology = output / "topology_with_provenance"
    build_report = topology / "adaptive_distillation_build.json"
    if not build_report.is_file():
        _run(
            sys.executable,
            "-m",
            "topology.adaptive_distillation",
            "--canonical-map",
            artifacts["canonical_map"],
            "--function-graph",
            artifacts["function_graph"],
            "--complete-positive-teacher",
            artifacts["positive_teacher"],
            "--track-payload",
            artifacts["track_payload"],
            "--query-cache",
            artifacts["query_cache"],
            "--output-dir",
            topology,
            "--config",
            args.config.resolve(),
        )
    build = json.loads(build_report.read_text())
    factor_dir = output / "factor_maps"
    factor_report_path = factor_dir / "reserve_factor_maps.json"
    factor_source_map = (
        artifacts["trained_map"] if args.fixed_metric_gate_only else Path(build["map"])
    ).resolve()
    materialized_source = None
    if factor_report_path.is_file():
        materialized_source = Path(
            json.loads(factor_report_path.read_text())["source_map"]
        ).resolve()
    if materialized_source != factor_source_map:
        _run(
            sys.executable,
            "-m",
            "topology.reserve_factor",
            "--source-map",
            factor_source_map,
            "--canonical-map",
            artifacts["canonical_map"],
            "--track-payload",
            artifacts["track_payload"],
            "--selection-provenance",
            build["selection_provenance"]["path"],
            "--output-dir",
            factor_dir,
            "--factors",
            args.factors,
        )
    factors = json.loads(factor_report_path.read_text())["factors"]
    if args.materialize_only:
        report = {
            "schema": "lafgs_reserve_causal_factor_ablation",
            "version": 1,
            "changes_default_mainline": False,
            "status": "factor_maps_materialized",
            "selection_provenance": build["selection_provenance"],
            "results": factors,
        }
        (output / "reserve_factor_ablation_report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n"
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        return
    results = {}
    source_metric = None
    if args.fixed_metric_gate_only:
        source_metric = torch.load(
            artifacts["metric_state"], map_location="cpu", weights_only=False
        )
    for name, factor in factors.items():
        branch = output / name
        if args.fixed_metric_gate_only:
            branch.mkdir(parents=True, exist_ok=True)
            factor_state = torch.load(
                factor["map"], map_location="cpu", weights_only=False
            )
            fixed_metric = dict(source_metric)
            fixed_metric["landmark_indices"] = torch.as_tensor(
                factor_state["anchor_ids"]
            ).long()
            fixed_metric["map_path"] = str(Path(factor["map"]).resolve())
            metric_path = branch / "fixed_metric_state.pt"
            torch.save(fixed_metric, metric_path)
            trained = {
                "trained_map": Path(factor["map"]).resolve(),
                "metric_state": metric_path,
            }
        else:
            trained = train_compact_map(
                compact_map=factor["map"],
                function_graph=artifacts["function_graph"],
                track_payload=artifacts["track_payload"],
                query_cache=artifacts["query_cache"],
                prior_ply=artifacts["prior_ply"],
                gaussian_type=args.gaussian_type,
                sh_degree=args.sh_degree,
                output=branch / "map_learning",
                config=args.config.resolve(),
                provenance_shards=args.provenance_shards,
                observation_shards=args.observation_shards,
                scene_calibration=artifacts["scene_calibration"],
                refresh_all_ransac_shards=True,
            )
        results[name] = {
            "anchor_count": int(factor["anchor_count"]),
            "track_count": int(factor["track_count"]),
            "gaussian_count": int(factor["gaussian_count"]),
        }
        for split in [value for value in args.splits.split(",") if value]:
            destination = branch / f"{split}_seed{args.seed}"
            if not (destination / "summary.json").is_file():
                _run(
                    sys.executable,
                    "scripts/evaluate.py",
                    "--dataset",
                    args.dataset.resolve(),
                    "--map",
                    trained["trained_map"],
                    "--metric-state",
                    trained["metric_state"],
                    "--scene-calibration",
                    artifacts["scene_calibration"],
                    "--config",
                    args.config.resolve(),
                    "--output",
                    destination,
                    "--split",
                    split,
                    "--seed",
                    args.seed,
                    "--device",
                    args.device,
                )
            results[name][split] = json.loads(
                (destination / "summary.json").read_text()
            )
    report = {
        "schema": "lafgs_reserve_causal_factor_ablation",
        "version": 1,
        "changes_default_mainline": False,
        "fixed_metric_gate_only": bool(args.fixed_metric_gate_only),
        "selection_provenance": build["selection_provenance"],
        "results": results,
    }
    (output / "reserve_factor_ablation_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
