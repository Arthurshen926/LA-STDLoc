#!/usr/bin/env python3
"""Compare a completed P9 paired probe and write a valid GO/STOP Pair Gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

from common.hashing import sha256_file
from evidence.fixed_pair_matcher_ceiling import (
    pair_gate_report,
    validate_pair_gate_report,
)
from scripts.fixed_pair_matcher_ceiling_common import (
    atomic_json_save_fresh,
    configure_formal_cpu_runtime,
    load_completion,
    producer_identity,
    validate_fresh_file_output,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--completion", type=Path, required=True)
    parser.add_argument("--expected-completion-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def run(args: argparse.Namespace) -> dict:
    configure_formal_cpu_runtime()
    completed = load_completion(
        path=args.completion,
        expected_file_sha256=args.expected_completion_sha256,
    )
    output = validate_fresh_file_output(
        args.output,
        protected=[completed["path"], completed["probe_path"]],
    )
    if output.name != "p9_fixed_pair_matcher_ceiling_pair_gate.json":
        raise ValueError("P9 scene Pair Gate output must use its fixed filename")
    producer = producer_identity(
        entrypoint="python -m scripts.compare_fixed_pair_matcher_ceiling"
    )
    report = pair_gate_report(
        probe=completed["probe"],
        probe_path=str(completed["probe_path"]),
        probe_sha256=completed["probe_sha256"],
        completion_path=str(completed["path"]),
        completion_sha256=completed["sha256"],
        producer_identity=producer,
        compiled_identity=producer["compiled_identity"],
        parent_stairs_gate=completed["feature_cache"]["payload"]["inputs"].get(
            "parent_stairs_gate"
        ),
    )
    completion_identity = completed["payload"].get("compiled_identity")
    feature_identity = (
        completed["feature_cache"]["payload"]
        .get("producer_identity", {})
        .get("compiled_identity")
    )
    if not (
        completion_identity
        == feature_identity
        == completed["probe"]["producer_identity"].get("compiled_identity")
        == producer["compiled_identity"]
    ):
        raise ValueError("P9 feature/probe/comparator compiled identities differ")
    if completed["probe"]["scene"] == "greatcourt":
        parent = report["parent_stairs_gate"]
        if (
            not isinstance(parent, dict)
            or parent.get("scientific_projection", {}).get("scene_pair_gate_passed")
            is not True
        ):
            raise ValueError("GreatCourt P9 gate lacks its passing Stairs parent")
    elif report["parent_stairs_gate"] is not None:
        raise ValueError("Stairs P9 gate cannot have a parent scene gate")
    for path, digest in (
        (completed["path"], completed["sha256"]),
        (completed["probe_path"], completed["probe_sha256"]),
    ):
        if sha256_file(path) != digest:
            raise RuntimeError("P9 Pair-Gate input changed during comparison")
    atomic_json_save_fresh(
        report,
        output,
        validator=lambda value: validate_pair_gate_report(
            value, expected_scene=completed["probe"]["scene"]
        ),
    )
    return {
        **report,
        "output": str(output),
        "output_sha256": sha256_file(output),
    }


def main(argv: Sequence[str] | None = None) -> None:
    report = run(build_parser().parse_args(argv))
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["scene_pair_gate_passed"]:
        raise SystemExit(2)


def entrypoint(argv: Sequence[str] | None = None) -> None:
    try:
        main(argv)
    except SystemExit:
        raise
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    entrypoint()
