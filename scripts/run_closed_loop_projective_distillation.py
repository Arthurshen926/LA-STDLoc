#!/usr/bin/env python3
"""Run the bounded V6 map-feedback loop from one frozen Projective map."""

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


def _evaluation_command(
    args: argparse.Namespace,
    *,
    root: Path,
    map_path: Path,
    map_sha: str,
    metric_path: Path,
    metric_sha: str,
    output: Path,
) -> list[str]:
    return [
        sys.executable,
        str(root / "scripts/evaluate_v6_self_localization.py"),
        "--map", str(map_path),
        "--expected-map-sha256", map_sha,
        "--metric", str(metric_path),
        "--expected-metric-sha256", metric_sha,
        "--observation-cache", str(args.observation_cache),
        "--expected-observation-cache-sha256", args.expected_observation_cache_sha256,
        "--output-dir", str(output),
        "--device", args.device,
        "--cpu-threads", str(args.cpu_threads),
        "--positive-radius-px", str(args.positive_radius_px),
        "--alpha-minimum", str(args.alpha_minimum),
        "--required-rank", str(args.required_rank),
        "--ransac-reprojection-px", str(args.ransac_reprojection_px),
        "--seed", str(args.seed),
    ]


def _proposal_command(
    args: argparse.Namespace,
    *,
    root: Path,
    arm: str,
    map_path: Path,
    map_sha: str,
    feedback_path: Path,
    feedback_sha: str,
    output: Path,
) -> list[str]:
    command = [
        sys.executable,
        str(root / "scripts/propose_v6_round.py"),
        "--arm", arm,
        "--map", str(map_path),
        "--expected-map-sha256", map_sha,
        "--observation-cache", str(args.observation_cache),
        "--expected-observation-cache-sha256", args.expected_observation_cache_sha256,
        "--feedback", str(feedback_path),
        "--expected-feedback-sha256", feedback_sha,
        "--output-dir", str(output),
        "--device", args.device,
        "--descriptor-trust-region", str(args.descriptor_trust_region),
        "--maximum-anchors", str(args.maximum_anchor_count),
        "--matching-target", str(args.matching_target),
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
    return command


def run(args: argparse.Namespace) -> dict:
    root = Path(__file__).resolve().parents[1]
    dirty = subprocess.run(
        ["git", "status", "--porcelain=v1"], cwd=root, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    if dirty:
        raise RuntimeError("V6 closed-loop runner requires a clean worktree")
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if bool(args.association_graph) != bool(args.expected_association_graph_sha256):
        raise ValueError("association graph path and expected SHA must be supplied together")
    args.output_dir.mkdir(parents=True)
    map_path = args.map.resolve()
    metric_path = args.metric.resolve()
    map_sha = args.expected_map_sha256
    metric_sha = args.expected_metric_sha256
    seen = {map_sha}
    rounds = []
    for round_index in range(3):
        round_dir = args.output_dir / f"round_{round_index:02d}"
        round_dir.mkdir()
        baseline_dir = round_dir / "baseline"
        _run(
            _evaluation_command(
                args, root=root, map_path=map_path, map_sha=map_sha,
                metric_path=metric_path, metric_sha=metric_sha, output=baseline_dir,
            ),
            root=root,
        )
        baseline_summary_path = baseline_dir / "summary.json"
        baseline_payload = json.loads(baseline_summary_path.read_text())
        feedback_path = Path(baseline_payload["feedback_path"])
        feedback_sha = baseline_payload["feedback_sha256"]
        if sum(int(value) for value in baseline_payload["failure_layer_counts"].values()) == 0:
            decision = {
                "schema": "closed_loop_distillation_round_v1",
                "version": 1,
                "uses_source_mapping_rgb": False,
                "uses_test_queries": False,
                "round_index": round_index,
                "baseline_map_sha256": map_sha,
                "baseline_summary_sha256": sha256_file(baseline_summary_path),
                "decisions": [],
                "accepted_arm": None,
                "accepted_map_sha256": None,
                "stop": True,
                "stop_reason": "no_feedback_deficit",
            }
            decision_path = round_dir / "acceptance.json"
            decision_path.write_text(
                json.dumps(decision, indent=2, sort_keys=True) + "\n"
            )
            rounds.append(decision)
            break
        arms = ["descriptor", "selection"]
        if (
            int(baseline_payload["failure_layer_counts"].get("L1", 0)) > 0
            and args.association_graph is not None
        ):
            arms.append("reconstruction")
        candidate_rows = []
        for arm in arms:
            proposal_dir = round_dir / f"proposal_{arm}"
            _run(
                _proposal_command(
                    args, root=root, arm=arm, map_path=map_path, map_sha=map_sha,
                    feedback_path=feedback_path, feedback_sha=feedback_sha,
                    output=proposal_dir,
                ),
                root=root,
            )
            proposal = json.loads((proposal_dir / "proposal.json").read_text())
            if proposal.get("proposal_available") is False:
                continue
            evaluation_dir = round_dir / f"evaluation_{arm}"
            _run(
                _evaluation_command(
                    args,
                    root=root,
                    map_path=Path(proposal["output"]["map"]),
                    map_sha=proposal["output"]["map_sha256"],
                    metric_path=Path(proposal["output"]["metric"]),
                    metric_sha=proposal["output"]["metric_sha256"],
                    output=evaluation_dir,
                ),
                root=root,
            )
            candidate_rows.append((arm, proposal, evaluation_dir / "summary.json"))
        decision_path = round_dir / "acceptance.json"
        command = [
            sys.executable,
            str(root / "scripts/accept_v6_round.py"),
            "--round-index", str(round_index),
            "--baseline-summary", str(baseline_summary_path),
            "--expected-baseline-summary-sha256", sha256_file(baseline_summary_path),
            "--baseline-map-sha256", map_sha,
            "--maximum-anchor-count", str(args.maximum_anchor_count),
            "--maximum-online-latency-ms", str(args.maximum_online_latency_ms),
            "--output", str(decision_path),
        ]
        for state_hash in sorted(seen):
            command.extend(["--seen-state-sha256", state_hash])
        for arm, proposal, summary_path in candidate_rows:
            command.extend(
                [
                    "--arm", arm,
                    "--candidate-summary", str(summary_path),
                    "--expected-candidate-summary-sha256", sha256_file(summary_path),
                    "--candidate-map-sha256", proposal["output"]["map_sha256"],
                ]
            )
        _run(command, root=root)
        decision = json.loads(decision_path.read_text())
        rounds.append(decision)
        if decision["stop"]:
            break
        winner = next(
            row for row in candidate_rows if row[0] == decision["accepted_arm"]
        )[1]
        map_path = Path(winner["output"]["map"])
        metric_path = Path(winner["output"]["metric"])
        map_sha = winner["output"]["map_sha256"]
        metric_sha = winner["output"]["metric_sha256"]
        seen.add(map_sha)
    result = {
        "schema": "lafgs_v6_closed_loop_projective_run",
        "version": 1,
        "uses_source_mapping_rgb": False,
        "uses_test_queries": False,
        "producer_git_commit": commit,
        "maximum_rounds": 3,
        "rounds": rounds,
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
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--positive-radius-px", type=float, default=2.0)
    parser.add_argument("--alpha-minimum", type=float, default=0.05)
    parser.add_argument("--required-rank", type=int, default=4)
    parser.add_argument("--ransac-reprojection-px", type=float, default=4.0)
    parser.add_argument("--descriptor-trust-region", type=float, default=0.05)
    parser.add_argument("--matching-target", type=int, default=4)
    parser.add_argument("--pose-logdet-target", type=float, default=0.0)
    parser.add_argument("--maximum-anchor-count", type=int, required=True)
    parser.add_argument("--maximum-online-latency-ms", type=float, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
