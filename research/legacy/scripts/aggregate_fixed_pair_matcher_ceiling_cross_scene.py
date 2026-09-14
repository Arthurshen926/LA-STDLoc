#!/usr/bin/env python3
"""Aggregate the exact Stairs/GreatCourt P9 Pair Gates without Track work."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
from pathlib import Path
import sys

from common.hashing import sha256_file
from scripts.fixed_pair_matcher_ceiling_common import (
    atomic_json_save_fresh,
    configure_formal_cpu_runtime,
    load_scene_gate,
    producer_identity,
    validate_fresh_file_output,
)


CROSS_SCENE_SCHEMA = "lafgs_p9_fixed_pair_matcher_ceiling_cross_scene_pair_gate"
CROSS_SCENE_VERSION = 1
CROSS_SCENE_FILENAME = "cross_scene_fixed_pair_matcher_ceiling_gate.json"
CROSS_SCENE_REQUIRED_KEYS = {
    "schema",
    "version",
    "mapping_only",
    "uses_test_queries",
    "valid",
    "inputs",
    "compiled_identity",
    "producer_identity",
    "same_producer_source_hashes",
    "same_runtime_and_backend",
    "scene_pair_gate_passed",
    "both_scene_pair_gate_passed",
    "advance_to_track_implementation_review",
    "authorizes_real_track_run",
    "advance_to_pose",
    "authorizes_test",
    "changes_method_default",
    "decision",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stairs-pair-gate", type=Path, required=True)
    parser.add_argument("--expected-stairs-pair-gate-sha256", required=True)
    parser.add_argument("--greatcourt-pair-gate", type=Path, required=True)
    parser.add_argument("--expected-greatcourt-pair-gate-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _gate_input(gate: Mapping) -> dict:
    return {
        "path": str(gate["path"]),
        "sha256": gate["sha256"],
        "scientific_projection": dict(gate["scientific_projection"]),
    }


def validate_cross_scene_report(payload: Mapping) -> dict:
    """Validate the review-only two-scene decision and all authority flags."""
    passed_by_scene = payload.get("scene_pair_gate_passed")
    if (
        not CROSS_SCENE_REQUIRED_KEYS.issubset(payload)
        or payload.get("schema") != CROSS_SCENE_SCHEMA
        or payload.get("version") != CROSS_SCENE_VERSION
        or payload.get("mapping_only") is not True
        or payload.get("uses_test_queries") is not False
        or payload.get("valid") is not True
        or not isinstance(payload.get("producer_identity"), Mapping)
        or payload.get("compiled_identity")
        != payload["producer_identity"].get("compiled_identity")
        or payload.get("same_producer_source_hashes") is not True
        or payload.get("same_runtime_and_backend") is not True
        or not isinstance(passed_by_scene, Mapping)
        or set(passed_by_scene) != {"stairs", "greatcourt"}
        or any(not isinstance(value, bool) for value in passed_by_scene.values())
        or payload.get("authorizes_real_track_run") is not False
        or payload.get("advance_to_pose") is not False
        or payload.get("authorizes_test") is not False
        or payload.get("changes_method_default") is not False
    ):
        raise ValueError("P9 cross-scene Pair Gate is structurally invalid")
    passed = passed_by_scene["stairs"] and passed_by_scene["greatcourt"]
    if (
        payload.get("both_scene_pair_gate_passed") is not passed
        or payload.get("advance_to_track_implementation_review") is not passed
        or payload.get("decision")
        != (
            "GO_TO_FIXED_PAIR_TRACK_IMPLEMENTATION_REVIEW"
            if passed
            else "STOP_BEFORE_FIXED_PAIR_TRACK_IMPLEMENTATION"
        )
    ):
        raise ValueError("P9 cross-scene Pair Gate decision is stale")
    inputs = payload.get("inputs")
    if not isinstance(inputs, Mapping) or set(inputs) != {
        "stairs_scene_pair_gate",
        "greatcourt_scene_pair_gate",
    }:
        raise ValueError("P9 cross-scene Pair Gate inputs are incomplete")
    for scene, key in (
        ("stairs", "stairs_scene_pair_gate"),
        ("greatcourt", "greatcourt_scene_pair_gate"),
    ):
        value = inputs[key]
        if (
            not isinstance(value, Mapping)
            or value.get("scientific_projection", {}).get("scene") != scene
            or value.get("scientific_projection", {}).get("scene_pair_gate_passed")
            is not passed_by_scene[scene]
            or value.get("scientific_projection", {}).get("compiled_identity")
            != payload["compiled_identity"]
        ):
            raise ValueError("P9 cross-scene input projection is inconsistent")
    return {"both_scene_pair_gate_passed": passed}


def cross_scene_report(
    *, stairs: Mapping, greatcourt: Mapping, producer: Mapping
) -> dict:
    """Build one fail-closed cross-scene report from validated scene gates."""
    stairs_payload = stairs["payload"]
    greatcourt_payload = greatcourt["payload"]
    expected_parent = _gate_input(stairs)
    if greatcourt_payload.get("parent_stairs_gate") != expected_parent:
        raise ValueError(
            "P9 GreatCourt Pair Gate is not bound to this exact Stairs gate"
        )

    producers = (
        stairs_payload["producer_identity"],
        greatcourt_payload["producer_identity"],
        producer,
    )
    compiled = {value.get("compiled_identity") for value in producers}
    sources = [value.get("source_file_sha256") for value in producers]
    runtimes = [value.get("runtime") for value in producers]
    if len(compiled) != 1 or any(value != sources[0] for value in sources[1:]):
        raise ValueError("P9 scene gates do not share one compiled producer source set")
    if any(value != runtimes[0] for value in runtimes[1:]):
        raise ValueError("P9 scene gates do not share one runtime/backend")

    passed_by_scene = {
        "stairs": stairs_payload["scene_pair_gate_passed"],
        "greatcourt": greatcourt_payload["scene_pair_gate_passed"],
    }
    passed = passed_by_scene["stairs"] and passed_by_scene["greatcourt"]
    report = {
        "schema": CROSS_SCENE_SCHEMA,
        "version": CROSS_SCENE_VERSION,
        "mapping_only": True,
        "uses_test_queries": False,
        "valid": True,
        "inputs": {
            "stairs_scene_pair_gate": _gate_input(stairs),
            "greatcourt_scene_pair_gate": _gate_input(greatcourt),
        },
        "compiled_identity": producer["compiled_identity"],
        "producer_identity": dict(producer),
        "same_producer_source_hashes": True,
        "same_runtime_and_backend": True,
        "scene_pair_gate_passed": passed_by_scene,
        "both_scene_pair_gate_passed": passed,
        "advance_to_track_implementation_review": passed,
        "authorizes_real_track_run": False,
        "advance_to_pose": False,
        "authorizes_test": False,
        "changes_method_default": False,
        "decision": (
            "GO_TO_FIXED_PAIR_TRACK_IMPLEMENTATION_REVIEW"
            if passed
            else "STOP_BEFORE_FIXED_PAIR_TRACK_IMPLEMENTATION"
        ),
    }
    validate_cross_scene_report(report)
    return report


def run(args: argparse.Namespace) -> dict:
    configure_formal_cpu_runtime()
    stairs = load_scene_gate(
        path=args.stairs_pair_gate,
        expected_file_sha256=args.expected_stairs_pair_gate_sha256,
        expected_scene="stairs",
        require_pass=True,
    )
    greatcourt = load_scene_gate(
        path=args.greatcourt_pair_gate,
        expected_file_sha256=args.expected_greatcourt_pair_gate_sha256,
        expected_scene="greatcourt",
        require_pass=None,
    )
    output = validate_fresh_file_output(
        args.output, protected=[stairs["path"], greatcourt["path"]]
    )
    if output.name != CROSS_SCENE_FILENAME:
        raise ValueError("P9 cross-scene Pair Gate output must use its fixed filename")
    producer = producer_identity(
        entrypoint=(
            "python -m scripts.aggregate_fixed_pair_matcher_ceiling_cross_scene"
        )
    )
    report = cross_scene_report(stairs=stairs, greatcourt=greatcourt, producer=producer)
    for gate in (stairs, greatcourt):
        if sha256_file(gate["path"]) != gate["sha256"]:
            raise RuntimeError("P9 scene Pair Gate changed during aggregation")
    atomic_json_save_fresh(report, output, validator=validate_cross_scene_report)
    return {**report, "output": str(output), "output_sha256": sha256_file(output)}


def main(argv: Sequence[str] | None = None) -> None:
    report = run(build_parser().parse_args(argv))
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["both_scene_pair_gate_passed"]:
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
