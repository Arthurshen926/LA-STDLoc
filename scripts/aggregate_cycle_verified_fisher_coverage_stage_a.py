#!/usr/bin/env python3
"""Aggregate both P8 V2 Stage-A gates before authorizing Track work."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

from common.hashing import sha256_file
from evidence.cycle_verified_fisher import (
    assert_verified_cycle_table_exact,
    materialize_verified_cycle_table,
)
from scripts.compare_cycle_verified_fisher_coverage_stage_a import (
    STAIRS_V1_SELECTION_CONTRACT,
    _comparison,
    _gates,
)
from scripts.cycle_verified_fisher_cli_common import (
    SCENE_CONTRACTS,
    V2_FROZEN_SOURCE_CONTRACTS,
    atomic_json_save,
    assert_selection_metrics,
    attest_file,
    evaluate_pair_subsets_from_verified_table,
    load_coverage_selection,
    load_mapping_cache,
    load_probe,
    load_proposals,
    load_selection,
    load_verified_cycle_table,
    selection_pairs,
    validate_output_target,
    validate_probe_proposal_lineage,
    validate_v2_frozen_source_contract,
)


SCHEMA = "lafgs_cycle_verified_fisher_coverage_cross_scene_stage_a_gate"
PER_SCENE_SCHEMA = "lafgs_cycle_verified_fisher_coverage_stage_a_gate"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stairs-stage-a-gate", type=Path, required=True)
    parser.add_argument("--expected-stairs-stage-a-gate-sha256", required=True)
    parser.add_argument("--greatcourt-stage-a-gate", type=Path, required=True)
    parser.add_argument("--expected-greatcourt-stage-a-gate-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _load_scene_gate(*, scene: str, path: Path, expected_sha256: str) -> dict:
    path = attest_file(path, expected_sha256, label=f"{scene} P8 V2 Stage-A gate")
    payload = json.loads(path.read_text())
    gates = payload.get("gates")
    passed = payload.get("stage_a_passed") is True
    expected_decision = (
        "SCENE_STAGE_A_PASS_REQUIRES_OTHER_SCENE"
        if passed
        else "STOP_BEFORE_TRACK_REUSE"
    )
    expected_contract = {"scene": scene, **SCENE_CONTRACTS[scene]}
    expected_sources = {"scene": scene, **V2_FROZEN_SOURCE_CONTRACTS[scene]}
    if (
        payload.get("schema") != PER_SCENE_SCHEMA
        or int(payload.get("version", -1)) != 1
        or payload.get("uses_test_queries") is not False
        or payload.get("mapping_only") is not True
        or payload.get("valid") is not True
        or payload.get("policy") != "cycle_verified_fisher_coverage"
        or payload.get("scene_contract") != expected_contract
        or payload.get("frozen_source_contract") != expected_sources
        or payload.get("verified_geometry_independently_rematerialized_exact")
        is not True
        or not isinstance(gates, dict)
        or not gates
        or not all(isinstance(value, bool) for value in gates.values())
        or passed != all(gates.values())
        or payload.get("requires_other_scene") is not True
        or payload.get("requires_v2_aware_track_lineage_implementation") is not True
        or payload.get("advance_to_reuse_only_track_build") is not False
        or payload.get("decision") != expected_decision
    ):
        raise ValueError(f"{scene} P8 V2 Stage-A gate is invalid")
    inputs = payload.get("inputs")
    required_inputs = {
        "query_cache",
        "pair_proposals",
        "pair_match_probe",
        "verified_cycle_table",
        "pair_selection",
    }
    if scene == "stairs":
        required_inputs.add("stairs_v1_pair_selection")
    if not isinstance(inputs, dict) or set(inputs) != required_inputs:
        raise ValueError(f"{scene} P8 V2 Stage-A input registry is invalid")
    for name, reference in inputs.items():
        if (
            not isinstance(reference, dict)
            or not reference.get("path")
            or not isinstance(reference.get("sha256"), str)
        ):
            raise ValueError(f"{scene} Stage-A {name} reference is incomplete")
        referenced = Path(str(reference["path"])).expanduser().resolve()
        if not referenced.is_file() or sha256_file(referenced) != reference["sha256"]:
            raise ValueError(f"{scene} Stage-A {name} reference changed")
    return {
        "path": path,
        "sha256": sha256_file(path),
        "payload": payload,
        "passed": passed,
    }


def _artifact_matches(reference: dict, artifact: dict) -> bool:
    return (
        Path(str(reference.get("path", ""))).expanduser().resolve()
        == artifact["path"]
        and reference.get("sha256") == artifact["sha256"]
        and (
            "content_sha256" not in artifact
            or reference.get("content_sha256") == artifact["content_sha256"]
        )
        and (
            "mapping_scope" not in artifact
            or reference.get("mapping_scope") == artifact["mapping_scope"]
        )
    )


def _replay_scene_gate(scene_gate: dict) -> list[dict]:
    """Recursively replay a per-scene gate before granting Track authority."""
    payload = scene_gate["payload"]
    scene = payload["scene_contract"]["scene"]
    contract = SCENE_CONTRACTS[scene]
    source = V2_FROZEN_SOURCE_CONTRACTS[scene]
    inputs = payload["inputs"]
    cache_reference = inputs["query_cache"]
    mapping_scope = cache_reference.get("mapping_scope")
    scope_kwargs = {}
    if isinstance(mapping_scope, dict) and mapping_scope.get("mode") == (
        "mapping_sparse_refresh_equivalence_v2"
    ):
        equivalence = mapping_scope.get("equivalence_report", {})
        scope_kwargs = {
            "mapping_scope_equivalence": Path(str(equivalence.get("path", ""))),
            "expected_mapping_scope_equivalence_sha256": equivalence.get("sha256"),
        }
    cache = load_mapping_cache(
        path=Path(str(cache_reference["path"])),
        expected_file_sha256=source["query_cache_sha256"],
        expected_query_names_sha256=source["query_names_sha256"],
        expected_mapping_keypoints=contract["mapping_keypoints"],
        expected_nms_radius=contract["nms_radius"],
        **scope_kwargs,
    )
    proposals_reference = inputs["pair_proposals"]
    proposals = load_proposals(
        path=Path(str(proposals_reference["path"])),
        expected_file_sha256=source["proposals_sha256"],
        expected_content_sha256=source["proposals_content_sha256"],
        cache=cache,
        expected_mapping_keypoints=contract["mapping_keypoints"],
        expected_nms_radius=contract["nms_radius"],
        expected_pair_budget=contract["pair_budget"],
        expected_candidate_pair_count=contract["candidate_pair_count"],
        expected_candidate_components=contract["candidate_component_count"],
    )
    probe_reference = inputs["pair_match_probe"]
    probe = load_probe(
        path=Path(str(probe_reference["path"])),
        expected_file_sha256=source["probe_sha256"],
        expected_content_sha256=source["probe_content_sha256"],
        cache=cache,
        expected_mapping_keypoints=contract["mapping_keypoints"],
        expected_nms_radius=contract["nms_radius"],
        expected_candidate_pair_count=contract["candidate_pair_count"],
    )
    validate_probe_proposal_lineage(probe=probe, proposals=proposals)
    validate_v2_frozen_source_contract(
        scene=scene, cache=cache, probe=probe, proposals=proposals
    )
    table_reference = inputs["verified_cycle_table"]
    table = load_verified_cycle_table(
        path=Path(str(table_reference["path"])),
        expected_file_sha256=table_reference["sha256"],
        expected_content_sha256=table_reference["content_sha256"],
        probe=probe,
        expected_maximum_reprojection_error_px=2.0,
    )
    rematerialized = materialize_verified_cycle_table(
        pair_match_probe=probe["payload"],
        keypoints=cache["keypoints"],
        camera_K=cache["camera_K"],
        pose_w2c=cache["pose_w2c"],
        maximum_reprojection_error_px=2.0,
    )
    assert_verified_cycle_table_exact(table["payload"], rematerialized)
    selection_reference = inputs["pair_selection"]
    selection = load_coverage_selection(
        path=Path(str(selection_reference["path"])),
        expected_file_sha256=selection_reference["sha256"],
        expected_content_sha256=selection_reference["content_sha256"],
        probe=probe,
        coverage_reference_pairs=proposals["nearest_pairs"],
        verified_cycle_table=table,
        expected_pair_budget=contract["pair_budget"],
    )
    stairs_v1 = None
    stairs_v1_record = None
    if scene == "stairs":
        reference = inputs["stairs_v1_pair_selection"]
        if (
            reference.get("sha256") != STAIRS_V1_SELECTION_CONTRACT["sha256"]
            or reference.get("content_sha256")
            != STAIRS_V1_SELECTION_CONTRACT["content_sha256"]
        ):
            raise ValueError("Stairs Stage-A gate changed the frozen V1 guard")
        stairs_v1_record = load_selection(
            path=Path(str(reference["path"])),
            expected_file_sha256=STAIRS_V1_SELECTION_CONTRACT["sha256"],
            expected_content_sha256=STAIRS_V1_SELECTION_CONTRACT["content_sha256"],
            probe=probe,
            expected_pair_budget=contract["pair_budget"],
        )
    artifacts = {
        "query_cache": cache,
        "pair_proposals": proposals,
        "pair_match_probe": probe,
        "verified_cycle_table": table,
        "pair_selection": selection,
    }
    if stairs_v1_record is not None:
        artifacts["stairs_v1_pair_selection"] = stairs_v1_record
    if any(
        not _artifact_matches(inputs[name], artifact)
        for name, artifact in artifacts.items()
    ):
        raise ValueError(f"{scene} Stage-A recursive input lineage differs")
    subsets = {
        "control": proposals["nearest_pairs"],
        "variant": selection_pairs(selection["payload"]),
    }
    if stairs_v1_record is not None:
        subsets["stairs_v1"] = selection_pairs(stairs_v1_record["payload"])
    evaluations = evaluate_pair_subsets_from_verified_table(
        probe=probe["payload"],
        verified_cycle_table=table["payload"],
        subsets=subsets,
    )
    control, variant = evaluations["control"], evaluations["variant"]
    stairs_v1 = evaluations.get("stairs_v1")
    assert_selection_metrics(selection["payload"], variant)
    gates = _gates(
        selection=selection["payload"],
        control=control,
        variant=variant,
        expected_pair_budget=contract["pair_budget"],
        expected_candidate_pair_count=contract["candidate_pair_count"],
        expected_candidate_components=contract["candidate_component_count"],
        stairs_v1=stairs_v1,
    )
    comparisons = {
        name: _comparison(control[name], variant[name])
        for name in (
            "confidence_weighted_fisher_utility_sum",
            "completed_verified_keypoint_triangle_count",
            "completed_verified_triangle_camera_count",
        )
    }
    if (
        payload.get("control") != control
        or payload.get("variant") != variant
        or payload.get("stairs_v1") != stairs_v1
        or payload.get("gates") != gates
        or payload.get("comparisons") != comparisons
        or payload.get("stage_a_passed") is not all(gates.values())
        or payload.get("candidate_universe_verified_camera")
        != table["payload"]["candidate_verified_camera"]
    ):
        raise ValueError(f"{scene} Stage-A gate does not replay exactly")
    attestations = [
        {"label": name, "path": value["path"], "sha256": value["sha256"]}
        for name, value in artifacts.items()
    ]
    mapping_scope = cache.get("mapping_scope", {})
    equivalence = (
        mapping_scope.get("equivalence_report")
        if isinstance(mapping_scope, dict)
        else None
    )
    if isinstance(equivalence, dict):
        attestations.append(
            {
                "label": "mapping_scope_equivalence",
                "path": Path(str(equivalence["path"])).resolve(),
                "sha256": equivalence["sha256"],
            }
        )
    if any(
        sha256_file(value["path"]) != value["sha256"]
        for value in attestations
    ):
        raise RuntimeError(f"{scene} Stage-A input changed during recursive replay")
    return attestations


def run(args: argparse.Namespace) -> dict:
    scenes = {
        "stairs": _load_scene_gate(
            scene="stairs",
            path=args.stairs_stage_a_gate,
            expected_sha256=args.expected_stairs_stage_a_gate_sha256,
        ),
        "greatcourt": _load_scene_gate(
            scene="greatcourt",
            path=args.greatcourt_stage_a_gate,
            expected_sha256=args.expected_greatcourt_stage_a_gate_sha256,
        ),
    }
    replayed = {
        scene: _replay_scene_gate(value) for scene, value in scenes.items()
    }
    output_target = validate_output_target(
        args.output,
        protected_paths=[
            value["path"] for value in scenes.values()
        ]
        + [
            attestation["path"]
            for attestations in replayed.values()
            for attestation in attestations
        ],
    )
    passed = all(value["passed"] for value in scenes.values())
    report = {
        "schema": SCHEMA,
        "version": 1,
        "uses_test_queries": False,
        "mapping_only": True,
        "valid": True,
        "policy": "cycle_verified_fisher_coverage",
        "scene_stage_a_passed": {
            scene: value["passed"] for scene, value in scenes.items()
        },
        "both_scene_stage_a_passed": passed,
        "advance_to_v2_aware_reuse_only_track_build": passed,
        "authorizes_existing_v1_track_runner": False,
        "decision": (
            "GO_TO_V2_AWARE_REUSE_ONLY_TRACK_BUILD"
            if passed
            else "STOP_BEFORE_TRACK_REUSE"
        ),
        "inputs": {
            scene: {"path": str(value["path"]), "sha256": value["sha256"]}
            for scene, value in scenes.items()
        },
    }
    if any(
        sha256_file(value["path"]) != value["sha256"]
        for value in scenes.values()
    ) or any(
        sha256_file(attestation["path"]) != attestation["sha256"]
        for attestations in replayed.values()
        for attestation in attestations
    ):
        raise RuntimeError("A P8 V2 cross-scene input changed before authorization")
    output = atomic_json_save(report, output_target, overwrite=bool(args.overwrite))
    report["output"] = str(output)
    report["output_sha256"] = sha256_file(output)
    return report


def main(argv: Sequence[str] | None = None) -> None:
    report = run(build_parser().parse_args(argv))
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["both_scene_stage_a_passed"]:
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
