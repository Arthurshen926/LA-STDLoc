#!/usr/bin/env python3
"""Apply frozen risk supervision to V20 sparse-map replay shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from common.hashing import sha256_file
from map_learning.v13_risk_supervisor import supervise_candidate, validate_records


def aggregate(args: argparse.Namespace) -> dict:
    if args.output.exists():
        raise FileExistsError(args.output)
    payloads = [json.loads(path.read_text()) for path in args.shards]
    if any(
        item.get("schema") != "lafgs_v20_sparse_map_replay_shard"
        or item.get("uses_test_queries") is not False
        or item.get("loo_used") is not False
        or item.get("plant_row_policy") != "all_detected_rows"
        or item.get("descriptor_training_safe") is not True
        or not isinstance(item.get("analysis_only"), bool)
        for item in payloads
    ):
        raise ValueError("invalid V20 sparse-map replay shard")
    expected = int(payloads[0]["shard_count"])
    if (
        any(int(item["shard_count"]) != expected for item in payloads)
        or sorted(int(item["shard_index"]) for item in payloads)
        != list(range(expected))
    ):
        raise ValueError("V20 replay shard registry is incomplete")
    identity = payloads[0]["input"]
    arm = payloads[0]["candidate_arm"]
    authorized = bool(payloads[0]["strong_feedback_authorized"])
    analysis_only = bool(payloads[0]["analysis_only"])
    if analysis_only and authorized:
        raise ValueError("V20 analysis-only replay cannot carry formal authorization")
    evaluation_role = str(payloads[0].get("evaluation_role", ""))
    expected_role = "feedback_query" if args.phase == "control" else "confirmation_query"
    if evaluation_role != expected_role:
        raise ValueError("V20 replay phase uses the wrong certified view role")
    if any(
        item["input"] != identity
        or item["candidate_arm"] != arm
        or bool(item["strong_feedback_authorized"]) != authorized
        or bool(item["analysis_only"]) != analysis_only
        or item.get("evaluation_role") != evaluation_role
        for item in payloads[1:]
    ):
        raise ValueError("V20 replay shards have different frozen actions")
    raw_records = [record for item in payloads for record in item["records"]]
    query_indices = [int(record["query_index"]) for record in raw_records]
    source_record_sha256s = [
        str(record.get("source_record_sha256", "")) for record in raw_records
    ]
    if len(set(query_indices)) != len(query_indices):
        raise ValueError("V20 replay queries overlap across shards")
    if (
        len(set(source_record_sha256s)) != len(source_record_sha256s)
        or any(len(value) != 64 for value in source_record_sha256s)
    ):
        raise ValueError("V20 replay source-record registry is invalid")
    certified_path = Path(identity["certified_batch"]).resolve()
    if sha256_file(certified_path) != identity["certified_batch_sha256"]:
        raise ValueError("V20 replay certified-batch SHA256 differs")
    certified = json.loads(certified_path.read_text())
    if not (
        certified.get("schema") == "lafgs_v14_observer_split_certified_view"
        and certified.get("version") == 1
        and certified.get("view_role") == expected_role
        and certified.get("uses_test_queries") is False
        and certified.get("map_mutation_count") == 0
    ):
        raise ValueError("V20 replay certified-batch contract differs")
    expected_accept: dict[int, tuple[str, int]] = {}
    for item in certified.get("records", []):
        if item.get("decision") != "ACCEPT":
            continue
        source_path = Path(item.get("path", "")).resolve()
        source_sha = str(item.get("sha256", ""))
        if not source_path.is_file() or sha256_file(source_path) != source_sha:
            raise ValueError("V20 certified source-record SHA256 differs")
        source = torch.load(source_path, map_location="cpu", weights_only=False)
        query_index = int(item["query_index"])
        if not (
            source.get("schema") == "lafgs_v7_certified_clean_render"
            and source.get("version") == 1
            and source.get("view_role") == expected_role
            and int(source.get("query_index", -1)) == query_index
            and source.get("certificate", {}).get("decision") == "ACCEPT"
        ):
            raise ValueError("V20 certified source-record identity differs")
        if query_index in expected_accept:
            raise ValueError("V20 certified ACCEPT query registry is not unique")
        expected_accept[query_index] = (
            source_sha,
            int(source["pose_family_id"]),
        )
    actual_accept = {
        int(record["query_index"]): (
            str(record["source_record_sha256"]),
            int(record["pose_family_id"]),
        )
        for record in raw_records
    }
    if (
        len(expected_accept)
        != int(certified.get("decision_counts", {}).get("ACCEPT", -1))
        or actual_accept != expected_accept
        or any(
            int(item.get("source_query_count", -1))
            != len(certified.get("records", []))
            or int(item.get("accepted_query_count", -1))
            != len(item.get("records", []))
            for item in payloads
        )
    ):
        raise ValueError("V20 replay does not exactly cover certified ACCEPT queries")
    pose_families = sorted(
        {int(record["pose_family_id"]) for record in raw_records}
    )
    if not raw_records or not pose_families:
        raise ValueError("V20 replay has no accepted query families")
    records = validate_records(raw_records, [arm])
    decision = supervise_candidate(records, arm, seed=20260831)
    if args.phase == "control":
        if args.selected_arm is not None:
            raise ValueError("V20 control selects its arm; do not pass selected arm")
        selected = (
            arm
            if authorized
            and decision["classification"]
            in {"DEFAULT_CANDIDATE", "PARETO_CANDIDATE"}
            else None
        )
        route = "ADVANCE_TO_CONFIRMATION" if selected else "NO_ACTION"
    else:
        if args.selected_arm != arm:
            raise ValueError("V20 confirmation must replay the control-selected arm")
        if not authorized:
            raise ValueError("V20 confirmation cannot deploy an unauthorized teacher arm")
        selected = arm
        route = decision["classification"]
    output = {
        "schema": "lafgs_v20_sparse_map_decision",
        "version": 1,
        "phase": args.phase,
        "uses_test_queries": False,
        "loo_used": False,
        "confirmation_can_train_or_select": False,
        "strong_feedback_authorized": authorized,
        "analysis_only": analysis_only,
        "accepted_query_count": len(records),
        "evaluation_role": evaluation_role,
        "evaluation_query_indices": sorted(query_indices),
        "evaluation_source_record_sha256s": sorted(source_record_sha256s),
        "evaluation_pose_family_ids": pose_families,
        "input": identity,
        "candidate_arm": arm,
        "decision_report": decision,
        "selected_arm": selected,
        "decision": route,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards", type=Path, nargs="+", required=True)
    parser.add_argument("--phase", choices=("control", "confirmation"), required=True)
    parser.add_argument("--selected-arm")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = aggregate(args)
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
