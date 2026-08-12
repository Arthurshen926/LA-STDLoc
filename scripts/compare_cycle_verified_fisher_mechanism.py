#!/usr/bin/env python3
"""Compare P8 Stage-B Tracks after an independently passed Stage-A gate."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Sequence

from common.hashing import sha256_file
from evidence.cycle_verified_fisher import CONTROL_POLICY_NAME, POLICY_NAME
from scripts.cycle_verified_fisher_cli_common import (
    atomic_json_save,
    attest_file,
    load_mapping_cache,
    load_probe,
    load_proposals,
    load_selection,
    load_stage_a_gate,
    load_track_factor,
    selection_pairs,
    validate_output_target,
    validate_probe_proposal_lineage,
    validate_scene_contract,
)
from scripts.run_track_pair_factor import _track_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", choices=("stairs", "greatcourt"), required=True)
    parser.add_argument("--query-cache", type=Path, required=True)
    parser.add_argument("--expected-query-cache-sha256", required=True)
    parser.add_argument("--proposals", type=Path, required=True)
    parser.add_argument("--expected-proposals-sha256", required=True)
    parser.add_argument("--expected-proposals-content-sha256", required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--expected-probe-sha256", required=True)
    parser.add_argument("--expected-probe-content-sha256", required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--expected-selection-sha256", required=True)
    parser.add_argument("--expected-selection-content-sha256", required=True)
    parser.add_argument("--stage-a-gate", type=Path, required=True)
    parser.add_argument("--expected-stage-a-gate-sha256", required=True)
    parser.add_argument("--control-factor", type=Path, required=True)
    parser.add_argument("--expected-control-factor-sha256", required=True)
    parser.add_argument("--control-report", type=Path, required=True)
    parser.add_argument("--expected-control-report-sha256", required=True)
    parser.add_argument("--variant-factor", type=Path, required=True)
    parser.add_argument("--expected-variant-factor-sha256", required=True)
    parser.add_argument("--variant-report", type=Path, required=True)
    parser.add_argument("--expected-variant-report-sha256", required=True)
    parser.add_argument("--expected-query-names-sha256", required=True)
    parser.add_argument("--expected-mapping-keypoints", type=int, required=True)
    parser.add_argument("--expected-nms-radius", type=int, required=True)
    parser.add_argument("--expected-pair-budget", type=int, required=True)
    parser.add_argument("--expected-candidate-pair-count", type=int, required=True)
    parser.add_argument("--expected-candidate-components", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _load_report(
    *,
    path: Path,
    expected_sha256: str,
    factor: dict,
    expected_policy: str,
    cache: dict,
    mapping_keypoints: int,
    nms_radius: int,
    pair_budget: int,
) -> dict:
    path = attest_file(path, expected_sha256, label=f"{expected_policy} report")
    payload = json.loads(path.read_text())
    if (
        payload.get("schema") != "lafgs_pair_policy_track_factor"
        or int(payload.get("version", -1)) != 1
        or payload.get("uses_test_queries") is not False
        or payload.get("reuse_only") is not True
        or payload.get("pair_policy") != expected_policy
        or int(payload.get("mapping_keypoint_factor", -1)) != int(mapping_keypoints)
        or int(payload.get("mapping_nms_radius", -1)) != int(nms_radius)
        or int(payload.get("exact_pair_budget", -1)) != int(pair_budget)
        or int(payload.get("mapping_query_count", -1)) != len(cache["names"])
        or payload.get("query_names_sha256") != cache["query_names_sha256"]
        or Path(str(payload.get("artifact", ""))).resolve() != factor["path"]
        or payload.get("artifact_sha256") != factor["sha256"]
        or payload.get("pair_policy_parameters")
        != factor["payload"].get("pair_policy_parameters")
    ):
        raise ValueError(f"{expected_policy} report differs from its reuse-only factor")
    inputs = payload.get("inputs")
    query_cache = inputs.get("query_cache") if isinstance(inputs, dict) else None
    if (
        not isinstance(query_cache, dict)
        or Path(str(query_cache.get("path", ""))).resolve() != cache["path"]
        or query_cache.get("sha256") != cache["sha256"]
        or not isinstance(payload.get("track"), dict)
        or inputs != factor["payload"].get("input_lineage")
    ):
        raise ValueError(f"{expected_policy} report lacks exact cache/Track evidence")
    expected_track = _track_report(
        factor["payload"]["tracks"],
        factor["payload"]["track_geometry"],
        query_count=len(cache["names"]),
    )
    if payload["track"] != expected_track:
        raise ValueError(f"{expected_policy} report Track metrics differ from factor")
    return {"path": path, "sha256": sha256_file(path), "payload": payload}


def _finite_number(payload: dict, *path: str) -> float:
    value = payload
    for key in path:
        if not isinstance(value, dict) or key not in value:
            raise ValueError(f"Missing mechanism metric: {'/'.join(path)}")
        value = value[key]
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Mechanism metric is not finite: {'/'.join(path)}")
    return result


def _track_metrics(report: dict, *, allow_undefined_covariance: bool) -> dict:
    result = {
        "triangulated_tracks": _finite_number(
            report, "track", "triangulated_track_count"
        ),
        "broad_eligible_tracks": _finite_number(
            report, "track", "broad_eligible_track_count"
        ),
        "high_confidence_tracks": _finite_number(
            report, "track", "high_confidence_track_count"
        ),
        "mapping_query_with_broad_track_fraction": _finite_number(
            report, "track", "mapping_query_with_broad_track_fraction"
        ),
    }
    covariance = report.get("track", {}).get(
        "triangulated_covariance_trace_m2", {}
    ).get("p90")
    if covariance is None and allow_undefined_covariance:
        result["triangulated_covariance_p90_m2"] = None
    elif covariance is None:
        raise ValueError("Control triangulated covariance p90 is undefined")
    else:
        result["triangulated_covariance_p90_m2"] = float(covariance)
        if not math.isfinite(result["triangulated_covariance_p90_m2"]):
            raise ValueError("Triangulated covariance p90 is not finite")
    if any(
        result[name] < 0
        for name in (
            "triangulated_tracks",
            "broad_eligible_tracks",
            "high_confidence_tracks",
        )
    ):
        raise ValueError("Track count metrics must be non-negative")
    coverage = result["mapping_query_with_broad_track_fraction"]
    if coverage < 0.0 or coverage > 1.0:
        raise ValueError("Mapping-query Track coverage must lie in [0, 1]")
    if (
        result["triangulated_covariance_p90_m2"] is not None
        and result["triangulated_covariance_p90_m2"] < 0.0
    ):
        raise ValueError("Triangulated covariance p90 must be non-negative")
    return result


def _comparison(control: float, variant: float | None) -> dict:
    if variant is None:
        return {"control": control, "variant": None, "delta": None, "ratio": None}
    return {
        "control": control,
        "variant": variant,
        "delta": variant - control,
        "ratio": None if control == 0 else variant / control,
    }


def _stage_b_gates(*, control: dict, variant: dict) -> dict[str, bool]:
    return {
        "triangulated_tracks_retain_98pct": variant["triangulated_tracks"]
        >= 0.98 * control["triangulated_tracks"],
        "broad_eligible_tracks_retain_98pct": variant["broad_eligible_tracks"]
        >= 0.98 * control["broad_eligible_tracks"],
        "high_confidence_tracks_retain_98pct": variant["high_confidence_tracks"]
        >= 0.98 * control["high_confidence_tracks"],
        "triangulated_covariance_p90_not_worse_5pct": (
            variant["triangulated_covariance_p90_m2"] is not None
            and variant["triangulated_covariance_p90_m2"]
            <= 1.05 * control["triangulated_covariance_p90_m2"]
        ),
        "broad_mapping_query_coverage_not_lower": variant[
            "mapping_query_with_broad_track_fraction"
        ]
        >= control["mapping_query_with_broad_track_fraction"],
        "control_probe_rows_reused": True,
        "variant_probe_rows_reused": True,
        "same_probe_matcher_contract": True,
    }


def _validate_reuse_lineage(
    *,
    factor: dict,
    role: str,
    probe: dict,
    proposals: dict,
    selection: dict,
    stage_a: dict,
) -> None:
    diagnostics = factor.get("diagnostics")
    sidecar_policy = factor.get("pair_sidecar", {}).get("policy", {})
    lineage = factor.get("input_lineage")
    matcher = probe["payload"]["matcher"]
    if (
        not isinstance(diagnostics, dict)
        or int(diagnostics.get("track_pair_matches_reused", -1)) != 1
        or sidecar_policy.get("uses_precomputed_pair_matches") is not True
        or not isinstance(lineage, dict)
        or factor.get("pair_policy_parameters", {}).get("reuse_only") is not True
        or factor.get("pair_policy_parameters", {}).get("probe_matcher") != matcher
        or lineage.get("probe_matcher") != matcher
    ):
        raise ValueError(f"{role} factor does not attest same-probe matcher reuse")
    expected = {
        "pair_proposals": proposals,
        "pair_match_probe": probe,
        "pair_selection": selection,
        "stage_a_gate": stage_a,
    }
    for name, artifact in expected.items():
        entry = lineage.get(name)
        if (
            not isinstance(entry, dict)
            or Path(str(entry.get("path", ""))).resolve() != artifact["path"]
            or entry.get("sha256") != artifact["sha256"]
            or (
                "content_sha256" in artifact
                and entry.get("content_sha256") != artifact["content_sha256"]
            )
        ):
            raise ValueError(f"{role} factor {name} lineage differs")
    pair = factor["pair_sidecar"]["pair"]
    observed_pairs = list(
        zip(
            pair["left_query_index"].long().tolist(),
            pair["right_query_index"].long().tolist(),
        )
    )
    expected_pairs = (
        proposals["nearest_pairs"]
        if role == "control"
        else selection_pairs(selection["payload"])
    )
    expected_role = (
        "attested_nearest_same_probe_control"
        if role == "control"
        else "cycle_verified_fisher_selection"
    )
    if (
        observed_pairs != expected_pairs
        or lineage.get("pair_subset_role") != expected_role
        or factor.get("pair_policy_parameters", {}).get("pair_subset_role")
        != expected_role
    ):
        raise ValueError(f"{role} factor pair subset differs from its attestation")


def run(args: argparse.Namespace) -> dict:
    contract = validate_scene_contract(
        scene=args.scene,
        mapping_keypoints=args.expected_mapping_keypoints,
        nms_radius=args.expected_nms_radius,
        pair_budget=args.expected_pair_budget,
        candidate_pair_count=args.expected_candidate_pair_count,
        candidate_component_count=args.expected_candidate_components,
    )
    cache = load_mapping_cache(
        path=args.query_cache,
        expected_file_sha256=args.expected_query_cache_sha256,
        expected_query_names_sha256=args.expected_query_names_sha256,
        expected_mapping_keypoints=args.expected_mapping_keypoints,
        expected_nms_radius=args.expected_nms_radius,
    )
    proposals = load_proposals(
        path=args.proposals,
        expected_file_sha256=args.expected_proposals_sha256,
        expected_content_sha256=args.expected_proposals_content_sha256,
        cache=cache,
        expected_mapping_keypoints=args.expected_mapping_keypoints,
        expected_nms_radius=args.expected_nms_radius,
        expected_pair_budget=args.expected_pair_budget,
        expected_candidate_pair_count=args.expected_candidate_pair_count,
        expected_candidate_components=args.expected_candidate_components,
    )
    probe = load_probe(
        path=args.probe,
        expected_file_sha256=args.expected_probe_sha256,
        expected_content_sha256=args.expected_probe_content_sha256,
        cache=cache,
        expected_mapping_keypoints=args.expected_mapping_keypoints,
        expected_nms_radius=args.expected_nms_radius,
        expected_candidate_pair_count=args.expected_candidate_pair_count,
    )
    validate_probe_proposal_lineage(probe=probe, proposals=proposals)
    selection = load_selection(
        path=args.selection,
        expected_file_sha256=args.expected_selection_sha256,
        expected_content_sha256=args.expected_selection_content_sha256,
        probe=probe,
        expected_pair_budget=args.expected_pair_budget,
    )
    stage_a = load_stage_a_gate(
        path=args.stage_a_gate,
        expected_file_sha256=args.expected_stage_a_gate_sha256,
        cache=cache,
        proposals=proposals,
        probe=probe,
        selection=selection,
        require_go=True,
    )
    factor_common = {
        "expected_query_names": cache["names"],
        "expected_query_names_sha256": cache["query_names_sha256"],
        "expected_query_cache_path": cache["path"],
        "expected_query_cache_sha256": cache["sha256"],
        "expected_mapping_keypoints": args.expected_mapping_keypoints,
        "expected_nms_radius": args.expected_nms_radius,
        "expected_pair_budget": args.expected_pair_budget,
    }
    control_factor = load_track_factor(
        path=args.control_factor,
        expected_file_sha256=args.expected_control_factor_sha256,
        expected_policy=CONTROL_POLICY_NAME,
        **factor_common,
    )
    variant_factor = load_track_factor(
        path=args.variant_factor,
        expected_file_sha256=args.expected_variant_factor_sha256,
        expected_policy=POLICY_NAME,
        **factor_common,
    )
    _validate_reuse_lineage(
        factor=control_factor["payload"],
        role="control",
        probe=probe,
        proposals=proposals,
        selection=selection,
        stage_a=stage_a,
    )
    _validate_reuse_lineage(
        factor=variant_factor["payload"],
        role="variant",
        probe=probe,
        proposals=proposals,
        selection=selection,
        stage_a=stage_a,
    )
    if control_factor["payload"]["pair_policy_parameters"]["probe_matcher"] != (
        variant_factor["payload"]["pair_policy_parameters"]["probe_matcher"]
    ):
        raise ValueError("Control and variant factors use different matcher contracts")
    control_report = _load_report(
        path=args.control_report,
        expected_sha256=args.expected_control_report_sha256,
        factor=control_factor,
        expected_policy=CONTROL_POLICY_NAME,
        cache=cache,
        mapping_keypoints=args.expected_mapping_keypoints,
        nms_radius=args.expected_nms_radius,
        pair_budget=args.expected_pair_budget,
    )
    variant_report = _load_report(
        path=args.variant_report,
        expected_sha256=args.expected_variant_report_sha256,
        factor=variant_factor,
        expected_policy=POLICY_NAME,
        cache=cache,
        mapping_keypoints=args.expected_mapping_keypoints,
        nms_radius=args.expected_nms_radius,
        pair_budget=args.expected_pair_budget,
    )
    if (
        control_report["payload"].get("probe_matcher") != probe["payload"]["matcher"]
        or variant_report["payload"].get("probe_matcher")
        != probe["payload"]["matcher"]
    ):
        raise ValueError("Track reports do not bind the same probe matcher")

    control = _track_metrics(
        control_report["payload"], allow_undefined_covariance=False
    )
    variant = _track_metrics(
        variant_report["payload"], allow_undefined_covariance=True
    )
    comparisons = {name: _comparison(control[name], variant[name]) for name in control}
    gates = _stage_b_gates(control=control, variant=variant)
    passed = all(gates.values())
    artifacts = {
        "query_cache": cache,
        "pair_proposals": proposals,
        "pair_match_probe": probe,
        "pair_selection": selection,
        "stage_a_gate": stage_a,
        "control_factor": control_factor,
        "control_report": control_report,
        "variant_factor": variant_factor,
        "variant_report": variant_report,
    }
    output_target = validate_output_target(
        args.output, protected_paths=[value["path"] for value in artifacts.values()]
    )
    for artifact in artifacts.values():
        if sha256_file(artifact["path"]) != artifact["sha256"]:
            raise RuntimeError("A frozen input changed during Stage-B comparison")
    report = {
        "schema": "lafgs_cycle_verified_fisher_mechanism_gate",
        "version": 2,
        "uses_test_queries": False,
        "mapping_only": True,
        "valid": True,
        "policy": POLICY_NAME,
        "scene_contract": contract,
        "stage_a": {
            "gate_path": str(stage_a["path"]),
            "gate_sha256": stage_a["sha256"],
            "passed": True,
        },
        "stage_b": {
            "comparisons": comparisons,
            "gates": gates,
            "passed": passed,
        },
        "mechanism_gate_passed": passed,
        "advance_to_fullchain_mapping_pose": passed,
        "decision": "GO_TO_FULLCHAIN" if passed else "STOP_BEFORE_FULLCHAIN",
        "inputs": {
            name: {
                "path": str(value["path"]),
                "sha256": value["sha256"],
                **(
                    {"content_sha256": value["content_sha256"]}
                    if "content_sha256" in value
                    else {}
                ),
            }
            for name, value in artifacts.items()
        },
        "limitations": [
            "A two-scene mechanism pass authorizes fullchain/mapping-pose "
            "validation; it is not a pose claim."
        ],
    }
    output = atomic_json_save(report, output_target, overwrite=bool(args.overwrite))
    report["output"] = str(output)
    report["output_sha256"] = sha256_file(output)
    return report


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    report = run(args)
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["mechanism_gate_passed"]:
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
