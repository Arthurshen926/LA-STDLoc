#!/usr/bin/env python3
"""Fail closed on variant Track lineage at canonical or compact evidence stage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from common.artifacts import verify_evidence_graph_contract
from common.hashing import sha256_file


def _load(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def _same_artifact(value, expected: Path) -> bool:
    path = Path(str(value)).expanduser().resolve()
    return path == expected or (
        path.is_file() and sha256_file(path) == sha256_file(expected)
    )


def audit_canonical(
    *, evidence_contract: Path, expected_track_payload: Path
) -> dict:
    contract = json.loads(evidence_contract.read_text())
    verify_evidence_graph_contract(contract)
    expected_track_payload = expected_track_payload.resolve()
    track = contract["artifacts"]["track_payload"]
    valid = (
        Path(track["path"]).resolve() == expected_track_payload
        and track["sha256"] == sha256_file(expected_track_payload)
    )
    return {
        "schema": "lafgs_pair_policy_fullchain_lineage_audit",
        "version": 1,
        "stage": "canonical_evidence",
        "uses_test_queries": False,
        "valid": bool(valid),
        "evidence_contract": str(evidence_contract.resolve()),
        "evidence_contract_sha256": sha256_file(evidence_contract),
        "expected_track_payload": str(expected_track_payload),
        "expected_track_payload_sha256": sha256_file(expected_track_payload),
    }


def audit_compact(
    *, compact_map: Path, provenance: Path, teacher: Path,
    expected_track_payload: Path,
) -> dict:
    compact_map = compact_map.resolve()
    provenance = provenance.resolve()
    teacher = teacher.resolve()
    expected_track_payload = expected_track_payload.resolve()
    state = _load(compact_map)
    provenance_payload = _load(provenance)
    teacher_payload = _load(teacher)
    checks = {
        "compact_map_track_payload": _same_artifact(
            state.get("provenance", {}).get("track_payload", ""),
            expected_track_payload,
        ),
        "provenance_track_payload": _same_artifact(
            provenance_payload.get("config", {}).get("track_payload", ""),
            expected_track_payload,
        ),
        "teacher_track_payload": _same_artifact(
            teacher_payload.get("track_payload", ""), expected_track_payload
        ),
        "provenance_anchor_map": _same_artifact(
            provenance_payload.get("anchor_map", ""), compact_map
        ),
        "teacher_anchor_map": _same_artifact(
            teacher_payload.get("anchor_map", ""), compact_map
        ),
        "teacher_raster_provenance": _same_artifact(
            teacher_payload.get("raster_provenance", ""), provenance
        ),
    }
    return {
        "schema": "lafgs_pair_policy_fullchain_lineage_audit",
        "version": 1,
        "stage": "compact_evidence",
        "uses_test_queries": False,
        "valid": all(checks.values()),
        "checks": checks,
        "expected_track_payload": str(expected_track_payload),
        "expected_track_payload_sha256": sha256_file(expected_track_payload),
        "artifacts": {
            "compact_map": {"path": str(compact_map), "sha256": sha256_file(compact_map)},
            "raster_provenance": {"path": str(provenance), "sha256": sha256_file(provenance)},
            "complete_positive_teacher": {"path": str(teacher), "sha256": sha256_file(teacher)},
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("canonical", "compact"), required=True)
    parser.add_argument("--expected-track-payload", type=Path, required=True)
    parser.add_argument("--evidence-contract", type=Path)
    parser.add_argument("--compact-map", type=Path)
    parser.add_argument("--raster-provenance", type=Path)
    parser.add_argument("--complete-positive-teacher", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.stage == "canonical":
        if args.evidence_contract is None:
            parser.error("canonical stage requires --evidence-contract")
        report = audit_canonical(
            evidence_contract=args.evidence_contract,
            expected_track_payload=args.expected_track_payload,
        )
    else:
        required = (
            args.compact_map,
            args.raster_provenance,
            args.complete_positive_teacher,
        )
        if any(value is None for value in required):
            parser.error("compact stage requires map, provenance, and teacher")
        report = audit_compact(
            compact_map=args.compact_map,
            provenance=args.raster_provenance,
            teacher=args.complete_positive_teacher,
            expected_track_payload=args.expected_track_payload,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["valid"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
