#!/usr/bin/env python3
"""Apply V6 hard guards and lexicographic risk to one proposal panel."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from common.hashing import sha256_file
from map_learning.closed_loop_distillation import accept_candidate


def _summary(path: Path, expected: str) -> tuple[dict, str]:
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"summary SHA differs for {path}")
    payload = json.loads(path.read_text())
    summary = payload.get("summary")
    if not isinstance(summary, dict):
        raise ValueError("evaluation summary is missing")
    return summary, actual


def run(args: argparse.Namespace) -> dict:
    if not 0 <= int(args.round_index) < 3:
        raise ValueError("V6 permits only round indices 0, 1, and 2")
    baseline, baseline_sha = _summary(
        args.baseline_summary, args.expected_baseline_summary_sha256
    )
    seen = set(args.seen_state_sha256)
    if args.baseline_map_sha256 not in seen:
        seen.add(args.baseline_map_sha256)
    decisions = []
    for arm, path, expected, state_hash in zip(
        args.arm,
        args.candidate_summary,
        args.expected_candidate_summary_sha256,
        args.candidate_map_sha256,
    ):
        candidate, summary_sha = _summary(path, expected)
        decision = accept_candidate(
            baseline,
            candidate,
            seen_state_hashes=seen,
            candidate_state_hash=state_hash,
            maximum_anchor_count=args.maximum_anchor_count,
            maximum_online_latency_ms=args.maximum_online_latency_ms,
        )
        decisions.append(
            {
                "arm": arm,
                "candidate_summary_sha256": summary_sha,
                "candidate_map_sha256": state_hash,
                **decision,
            }
        )
    accepted = [row for row in decisions if row["accepted"]]
    winner = min(accepted, key=lambda row: tuple(row["candidate_risk"])) if accepted else None
    result = {
        "schema": "closed_loop_distillation_round_v1",
        "version": 1,
        "uses_source_mapping_rgb": False,
        "uses_test_queries": False,
        "round_index": int(args.round_index),
        "baseline_map_sha256": args.baseline_map_sha256,
        "baseline_summary_sha256": baseline_sha,
        "decisions": decisions,
        "accepted_arm": None if winner is None else winner["arm"],
        "accepted_map_sha256": None if winner is None else winner["candidate_map_sha256"],
        "stop": winner is None or int(args.round_index) == 2,
        "stop_reason": (
            "no_accepted_proposal"
            if winner is None
            else "maximum_rounds_reached"
            if int(args.round_index) == 2
            else None
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, args.output)
    finally:
        temporary.unlink(missing_ok=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round-index", type=int, required=True)
    parser.add_argument("--baseline-summary", type=Path, required=True)
    parser.add_argument("--expected-baseline-summary-sha256", required=True)
    parser.add_argument("--baseline-map-sha256", required=True)
    parser.add_argument("--arm", action="append", required=True)
    parser.add_argument("--candidate-summary", action="append", type=Path, required=True)
    parser.add_argument("--expected-candidate-summary-sha256", action="append", required=True)
    parser.add_argument("--candidate-map-sha256", action="append", required=True)
    parser.add_argument("--seen-state-sha256", action="append", default=[])
    parser.add_argument("--maximum-anchor-count", type=int, required=True)
    parser.add_argument("--maximum-online-latency-ms", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    lengths = {
        len(args.arm), len(args.candidate_summary),
        len(args.expected_candidate_summary_sha256), len(args.candidate_map_sha256),
    }
    if len(lengths) != 1:
        raise ValueError("candidate arm arguments must align")
    run(args)


if __name__ == "__main__":
    main()
