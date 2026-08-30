#!/usr/bin/env python3
"""Finalize a frozen V18 action through disjoint control and confirmation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from common.hashing import sha256_file
from map_learning.v18_control_gate import gate_closed_loop_action


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-arm", required=True)
    parser.add_argument("--stable-map", type=Path, required=True)
    parser.add_argument("--candidate-map", type=Path, required=True)
    parser.add_argument("--control-decision", type=Path, required=True)
    parser.add_argument("--confirmation-decision", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    control = json.loads(args.control_decision.read_text())
    confirmation = (
        None
        if args.confirmation_decision is None
        else json.loads(args.confirmation_decision.read_text())
    )
    gate = gate_closed_loop_action(
        candidate_arm=args.candidate_arm,
        control_decision=control,
        confirmation_decision=confirmation,
    )
    selected = args.candidate_map if gate["formal_deployment_authorized"] else args.stable_map
    output = {
        **gate,
        "selected_map": str(selected.resolve()),
        "selected_map_sha256": sha256_file(selected),
        "inputs": {
            "stable_map": str(args.stable_map.resolve()),
            "stable_map_sha256": sha256_file(args.stable_map),
            "candidate_map": str(args.candidate_map.resolve()),
            "candidate_map_sha256": sha256_file(args.candidate_map),
            "control_decision": str(args.control_decision.resolve()),
            "control_decision_sha256": sha256_file(args.control_decision),
            "confirmation_decision": (
                None
                if args.confirmation_decision is None
                else str(args.confirmation_decision.resolve())
            ),
            "confirmation_decision_sha256": (
                None
                if args.confirmation_decision is None
                else sha256_file(args.confirmation_decision)
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
