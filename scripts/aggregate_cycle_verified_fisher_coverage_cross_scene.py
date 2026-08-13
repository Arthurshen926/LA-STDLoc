#!/usr/bin/env python3
"""Aggregate both recursively validated P8 coverage-V2 Stage-B gates."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
from typing import Sequence

from common.hashing import sha256_file
from scripts.compare_cycle_verified_fisher_coverage_mechanism import (
    BASE_GATE_NAMES,
    STAIRS_RETENTION_GATE_NAMES,
    validate_stage_b_gate,
)
from scripts.cycle_verified_fisher_cli_common import (
    atomic_json_save,
    validate_output_target,
)
from scripts.cycle_verified_fisher_coverage_track_common import (
    artifact_schema_contract,
    cross_b_producer_identity,
    require_clean_identity,
    required_artifact_keys,
)


SCHEMA = "lafgs_cycle_verified_fisher_coverage_cross_scene_stage_b_gate"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stairs-stage-b-gate", type=Path, required=True)
    parser.add_argument("--expected-stairs-stage-b-gate-sha256", required=True)
    parser.add_argument("--greatcourt-stage-b-gate", type=Path, required=True)
    parser.add_argument("--expected-greatcourt-stage-b-gate-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def run(args: argparse.Namespace) -> dict:
    scenes = {
        "stairs": validate_stage_b_gate(
            scene="stairs",
            path=args.stairs_stage_b_gate,
            expected_sha256=args.expected_stairs_stage_b_gate_sha256,
        ),
        "greatcourt": validate_stage_b_gate(
            scene="greatcourt",
            path=args.greatcourt_stage_b_gate,
            expected_sha256=args.expected_greatcourt_stage_b_gate_sha256,
        ),
    }
    if scenes["stairs"]["path"] == scenes["greatcourt"]["path"]:
        raise ValueError("Cross-scene Stage-B requires two distinct scene gates")
    stairs_payload = scenes["stairs"]["payload"]
    greatcourt_payload = scenes["greatcourt"]["payload"]
    if (
        stairs_payload["stage_a"]["cross_scene_gate"]
        != greatcourt_payload["stage_a"]["cross_scene_gate"]
    ):
        raise ValueError("Stage-B gates bind different cross-scene Stage-A roots")
    expected_greatcourt_parent = {
        "path": str(scenes["greatcourt"]["path"]),
        "sha256": scenes["greatcourt"]["sha256"],
    }
    if (
        stairs_payload["paired_track"].get("greatcourt_stage_b_parent")
        != expected_greatcourt_parent
        or greatcourt_payload["paired_track"].get("greatcourt_stage_b_parent")
        is not None
    ):
        raise ValueError("Stairs Track does not bind this GreatCourt Stage-B Pass")
    if (
        stairs_payload["compiled_identity"]
        != greatcourt_payload["compiled_identity"]
    ):
        raise ValueError("Stage-B gates differ outside preregistered scene calibration")
    if (
        stairs_payload["paired_track"]["track_producer_identity"]
        != greatcourt_payload["paired_track"]["track_producer_identity"]
    ):
        raise ValueError("Both scenes must use the same Track producer identity")
    if (
        stairs_payload["stage_b_producer_identity"]
        != greatcourt_payload["stage_b_producer_identity"]
    ):
        raise ValueError("Both scenes must use the same Stage-B producer identity")
    if (
        set(stairs_payload["stage_b"]["base_gates"]) != BASE_GATE_NAMES
        or set(greatcourt_payload["stage_b"]["base_gates"]) != BASE_GATE_NAMES
        or set(stairs_payload["stage_b"]["stairs_v1_retention_gates"])
        != STAIRS_RETENTION_GATE_NAMES
        or set(stairs_payload["stage_b"]["stairs_control_parity_gate"])
        != {"v1_nearest_control_scientific_projection_exact"}
        or type(
            stairs_payload["stage_b"]["stairs_control_parity_gate"][
                "v1_nearest_control_scientific_projection_exact"
            ]
        )
        is not bool
        or greatcourt_payload["stage_b"]["stairs_control_parity_gate"] != {}
        or greatcourt_payload["stage_b"]["stairs_v1_retention_gates"] != {}
        or stairs_payload["stairs_v1_reference"] is None
        or greatcourt_payload["stairs_v1_reference"] is not None
    ):
        raise ValueError("Cross-scene base/Stairs-retention gate separation changed")
    passed_by_scene = {
        scene: value["payload"]["scene_specific_mechanism_pass"]
        for scene, value in scenes.items()
    }
    if any(type(value) is not bool for value in passed_by_scene.values()):
        raise ValueError("Cross-scene Stage-B outcomes are not boolean")
    passed = all(passed_by_scene.values())
    producer = cross_b_producer_identity()
    require_clean_identity(producer, label="P8 coverage-V2 cross-Stage-B producer")
    report = {
        "schema": SCHEMA,
        "version": 1,
        "uses_test_queries": False,
        "mapping_only": True,
        "valid": True,
        "policy": "cycle_verified_fisher_coverage",
        "inputs": {
            scene: {"path": str(value["path"]), "sha256": value["sha256"]}
            for scene, value in scenes.items()
        },
        "cross_scene_stage_a_gate": deepcopy(
            stairs_payload["stage_a"]["cross_scene_gate"]
        ),
        "compiled_identity": deepcopy(stairs_payload["compiled_identity"]),
        "track_producer_identity": deepcopy(
            stairs_payload["paired_track"]["track_producer_identity"]
        ),
        "stage_b_producer_identity": deepcopy(
            stairs_payload["stage_b_producer_identity"]
        ),
        "cross_stage_b_producer_identity": producer,
        "base_gates": {
            scene: deepcopy(value["payload"]["stage_b"]["base_gates"])
            for scene, value in scenes.items()
        },
        "stairs_v1_retention_gates": deepcopy(
            stairs_payload["stage_b"]["stairs_v1_retention_gates"]
        ),
        "stairs_control_parity_gate": deepcopy(
            stairs_payload["stage_b"]["stairs_control_parity_gate"]
        ),
        "scene_stage_b_passed": passed_by_scene,
        "both_scene_stage_b_passed": passed,
        "advance_to_v2_aware_fullchain_lineage_implementation": passed,
        "authorizes_existing_fullchain": False,
        "advance_to_mapping_pose": False,
        "authorizes_test": False,
        "changes_method_default": False,
        "decision": (
            "GO_TO_V2_AWARE_FULLCHAIN_LINEAGE_IMPLEMENTATION"
            if passed
            else "STOP_BEFORE_FULLCHAIN_LINEAGE_IMPLEMENTATION"
        ),
    }
    contract = artifact_schema_contract("cross_scene_stage_b_gate")
    if (
        set(report) != required_artifact_keys("cross_scene_stage_b_gate")
        or report["schema"] != contract["schema"]
        or report["version"] != contract["version"]
        or report["decision"]
        not in {contract["pass_decision"], contract["stop_decision"]}
        or report["authorizes_existing_fullchain"]
        is not contract["authorizes_existing_fullchain"]
        or report["advance_to_mapping_pose"]
        is not contract["advance_to_mapping_pose"]
        or report["authorizes_test"] is not contract["authorizes_test"]
    ):
        raise RuntimeError("Compiled cross-scene Stage-B output schema changed")
    output = validate_output_target(
        args.output, protected_paths=[value["path"] for value in scenes.values()]
    )
    if output.exists():
        raise FileExistsError("P8 V2 cross-Stage-B output already exists")
    if any(sha256_file(value["path"]) != value["sha256"] for value in scenes.values()):
        raise RuntimeError("A scene Stage-B gate changed before aggregation")
    atomic_json_save(report, output, overwrite=False)
    return {**report, "output": str(output), "output_sha256": sha256_file(output)}


def main(argv: Sequence[str] | None = None) -> None:
    report = run(build_parser().parse_args(argv))
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["both_scene_stage_b_passed"]:
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
