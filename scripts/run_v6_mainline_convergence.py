#!/usr/bin/env python3
"""Run fixed-stage V6 convergence without metric hard-gate acceptance."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from common.hashing import sha256_file


def _run(command: list[str], *, root: Path) -> None:
    subprocess.run(command, cwd=root, check=True)


def _evaluate(
    args: argparse.Namespace,
    *,
    root: Path,
    map_path: Path,
    map_sha: str,
    metric_path: Path,
    metric_sha: str,
    output: Path,
) -> tuple[Path, dict]:
    _run(
        [
            sys.executable,
            str(root / "scripts/evaluate_v6_self_localization.py"),
            "--map", str(map_path),
            "--expected-map-sha256", map_sha,
            "--metric", str(metric_path),
            "--expected-metric-sha256", metric_sha,
            "--observation-cache", str(args.observation_cache),
            "--expected-observation-cache-sha256",
            args.expected_observation_cache_sha256,
            "--output-dir", str(output),
            "--device", args.device,
            "--cpu-threads", str(args.cpu_threads),
            "--positive-radius-px", str(args.positive_radius_px),
            "--alpha-minimum", str(args.alpha_minimum),
            "--required-rank", str(args.required_rank),
            "--loo-pose-neighbors", str(args.loo_pose_neighbors),
            "--ransac-reprojection-px", str(args.ransac_reprojection_px),
            "--seed", str(args.seed),
        ],
        root=root,
    )
    summary_path = output / "summary.json"
    return summary_path, json.loads(summary_path.read_text())


def _propose(
    args: argparse.Namespace,
    *,
    root: Path,
    arm: str,
    map_path: Path,
    map_sha: str,
    feedback_summary: dict,
    output: Path,
) -> dict:
    command = [
        sys.executable,
        str(root / "scripts/propose_v6_round.py"),
        "--arm", arm,
        "--map", str(map_path),
        "--expected-map-sha256", map_sha,
        "--observation-cache", str(args.observation_cache),
        "--expected-observation-cache-sha256",
        args.expected_observation_cache_sha256,
        "--feedback", str(feedback_summary["feedback_path"]),
        "--expected-feedback-sha256", feedback_summary["feedback_sha256"],
        "--output-dir", str(output),
        "--device", args.device,
        "--descriptor-trust-region", str(args.descriptor_trust_region),
        "--descriptor-margin", str(args.descriptor_margin),
        "--descriptor-temperature", str(args.descriptor_temperature),
        "--descriptor-learning-rate", str(args.descriptor_learning_rate),
        "--descriptor-epochs", str(args.descriptor_epochs),
        "--descriptor-batch-size", str(args.descriptor_batch_size),
        "--descriptor-maximum-triplets-per-query",
        str(args.descriptor_maximum_triplets_per_query),
        "--maximum-anchors", str(args.selection_maximum_anchors),
        "--matching-target", str(args.required_rank),
        "--pose-logdet-target", str(args.pose_logdet_target),
    ]
    if arm == "reconstruction":
        command.extend(
            [
                "--association-graph", str(args.association_graph),
                "--expected-association-graph-sha256",
                args.expected_association_graph_sha256,
            ]
        )
    _run(command, root=root)
    return json.loads((output / "proposal.json").read_text())


def run(args: argparse.Namespace) -> dict:
    root = Path(__file__).resolve().parents[1]
    dirty = subprocess.run(
        ["git", "status", "--porcelain=v1"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if dirty:
        raise RuntimeError("V6 convergence runner requires a clean worktree")
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if int(args.descriptor_rounds) < 1:
        raise ValueError("at least one descriptor round is required")
    if bool(args.run_reconstruction) and (
        args.association_graph is None
        or args.expected_association_graph_sha256 is None
    ):
        raise ValueError("reconstruction requires the SHA-bound association graph")
    args.output_dir.mkdir(parents=True)

    map_path = args.map.resolve()
    map_sha = args.expected_map_sha256
    metric_path = args.metric.resolve()
    metric_sha = args.expected_metric_sha256
    stages = []

    summary_path, summary = _evaluate(
        args,
        root=root,
        map_path=map_path,
        map_sha=map_sha,
        metric_path=metric_path,
        metric_sha=metric_sha,
        output=args.output_dir / "stage_00_baseline",
    )
    stages.append(
        {
            "stage": "baseline",
            "map": str(map_path),
            "map_sha256": map_sha,
            "summary": str(summary_path),
            "metrics": summary["summary"],
            "failure_layer_counts": summary["failure_layer_counts"],
        }
    )

    for round_index in range(int(args.descriptor_rounds)):
        proposal = _propose(
            args,
            root=root,
            arm="descriptor_loss",
            map_path=map_path,
            map_sha=map_sha,
            feedback_summary=summary,
            output=args.output_dir / f"stage_{len(stages):02d}_descriptor_proposal",
        )
        if proposal.get("proposal_available") is False:
            break
        map_path = Path(proposal["output"]["map"])
        map_sha = proposal["output"]["map_sha256"]
        metric_path = Path(proposal["output"]["metric"])
        metric_sha = proposal["output"]["metric_sha256"]
        summary_path, summary = _evaluate(
            args,
            root=root,
            map_path=map_path,
            map_sha=map_sha,
            metric_path=metric_path,
            metric_sha=metric_sha,
            output=args.output_dir / f"stage_{len(stages):02d}_descriptor_evaluation",
        )
        stages.append(
            {
                "stage": f"descriptor_round_{round_index + 1}",
                "map": str(map_path),
                "map_sha256": map_sha,
                "summary": str(summary_path),
                "metrics": summary["summary"],
                "failure_layer_counts": summary["failure_layer_counts"],
            }
        )

    if args.run_reconstruction and int(summary["failure_layer_counts"].get("L1", 0)):
        proposal = _propose(
            args,
            root=root,
            arm="reconstruction",
            map_path=map_path,
            map_sha=map_sha,
            feedback_summary=summary,
            output=args.output_dir / f"stage_{len(stages):02d}_reconstruction_proposal",
        )
        if proposal.get("proposal_available") is not False:
            map_path = Path(proposal["output"]["map"])
            map_sha = proposal["output"]["map_sha256"]
            metric_path = Path(proposal["output"]["metric"])
            metric_sha = proposal["output"]["metric_sha256"]
            summary_path, summary = _evaluate(
                args,
                root=root,
                map_path=map_path,
                map_sha=map_sha,
                metric_path=metric_path,
                metric_sha=metric_sha,
                output=args.output_dir / f"stage_{len(stages):02d}_reconstruction_evaluation",
            )
            stages.append(
                {
                    "stage": "reconstruction",
                    "map": str(map_path),
                    "map_sha256": map_sha,
                    "summary": str(summary_path),
                    "metrics": summary["summary"],
                    "failure_layer_counts": summary["failure_layer_counts"],
                }
            )

    if args.run_selection:
        proposal = _propose(
            args,
            root=root,
            arm="selection",
            map_path=map_path,
            map_sha=map_sha,
            feedback_summary=summary,
            output=args.output_dir / f"stage_{len(stages):02d}_selection_proposal",
        )
        map_path = Path(proposal["output"]["map"])
        map_sha = proposal["output"]["map_sha256"]
        metric_path = Path(proposal["output"]["metric"])
        metric_sha = proposal["output"]["metric_sha256"]
        summary_path, summary = _evaluate(
            args,
            root=root,
            map_path=map_path,
            map_sha=map_sha,
            metric_path=metric_path,
            metric_sha=metric_sha,
            output=args.output_dir / f"stage_{len(stages):02d}_selection_evaluation",
        )
        stages.append(
            {
                "stage": "selection_after_fresh_feedback",
                "map": str(map_path),
                "map_sha256": map_sha,
                "summary": str(summary_path),
                "metrics": summary["summary"],
                "failure_layer_counts": summary["failure_layer_counts"],
            }
        )

    result = {
        "schema": "lafgs_v6_mainline_convergence_run",
        "version": 1,
        "uses_source_mapping_rgb": False,
        "uses_test_queries": False,
        "automatic_hard_gate_acceptance": False,
        "stages": stages,
        "final_map": str(map_path),
        "final_map_sha256": map_sha,
        "final_metric": str(metric_path),
        "final_metric_sha256": metric_sha,
        "online_protocol": "native_superpoint_global_top1_one_standard_poselib",
    }
    temporary = args.output_dir / f".run.{os.getpid()}.tmp"
    try:
        temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, args.output_dir / "run.json")
    finally:
        temporary.unlink(missing_ok=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", type=Path, required=True)
    parser.add_argument("--expected-map-sha256", required=True)
    parser.add_argument("--metric", type=Path, required=True)
    parser.add_argument("--expected-metric-sha256", required=True)
    parser.add_argument("--observation-cache", type=Path, required=True)
    parser.add_argument("--expected-observation-cache-sha256", required=True)
    parser.add_argument("--association-graph", type=Path)
    parser.add_argument("--expected-association-graph-sha256")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--positive-radius-px", type=float, default=2.0)
    parser.add_argument("--alpha-minimum", type=float, default=0.05)
    parser.add_argument("--required-rank", type=int, default=16)
    parser.add_argument("--loo-pose-neighbors", type=int, default=3)
    parser.add_argument("--ransac-reprojection-px", type=float, default=4.0)
    parser.add_argument("--descriptor-rounds", type=int, default=1)
    parser.add_argument("--descriptor-trust-region", type=float, default=0.05)
    parser.add_argument("--descriptor-margin", type=float, default=0.05)
    parser.add_argument("--descriptor-temperature", type=float, default=0.04)
    parser.add_argument("--descriptor-learning-rate", type=float, default=0.02)
    parser.add_argument("--descriptor-epochs", type=int, default=5)
    parser.add_argument("--descriptor-batch-size", type=int, default=8192)
    parser.add_argument("--descriptor-maximum-triplets-per-query", type=int, default=128)
    parser.add_argument("--run-reconstruction", action="store_true")
    parser.add_argument("--run-selection", action="store_true")
    parser.add_argument("--selection-maximum-anchors", type=int, default=20000)
    parser.add_argument("--pose-logdet-target", type=float, default=0.0)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
