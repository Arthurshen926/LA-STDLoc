#!/usr/bin/env python3
"""Expose an Observer split as a sealed certified-render view."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from common.hashing import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observer-batch", type=Path, required=True)
    parser.add_argument("--view-role", choices=("feedback_query", "confirmation_query"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    source_path = args.observer_batch.resolve()
    source = json.loads(source_path.read_text())
    if not (
        source.get("schema") == "lafgs_v9_no_loo_causal_feedback_batch"
        and source.get("uses_test_queries") is False
        and source.get("loo_used") is False
        and source.get("accepted_query_row_policy") == "v2_row_valid_only"
    ):
        raise ValueError("view requires corrected no-LOO Observer records")
    certified_path = Path(source["input"]["certified_batch"]).resolve()
    if sha256_file(certified_path) != source["input"]["certified_batch_sha256"]:
        raise ValueError("Observer source certified-batch SHA256 differs")
    certified = json.loads(certified_path.read_text())
    required_observer_role = (
        "heldout_control"
        if args.view_role == "feedback_query"
        else "confirmation_observer"
    )
    if (
        certified.get("view_role") != args.view_role
        or source.get("role") != required_observer_role
    ):
        raise ValueError("Observer role cannot materialize the requested view")
    records = []
    decisions = {"ACCEPT": 0, "UNCERTAIN": 0, "REJECT": 0}
    families = set()
    for item in source["records"]:
        observed_path = Path(item["path"])
        if sha256_file(observed_path) != item["sha256"]:
            raise ValueError("Observer record SHA256 differs")
        observed = torch.load(observed_path, map_location="cpu", weights_only=False)
        render_path = Path(observed["source_record"])
        if sha256_file(render_path) != observed["source_record_sha256"]:
            raise ValueError("certified render SHA256 differs")
        decision = str(observed["certificate_decision"])
        decisions[decision] += 1
        families.add(int(observed["pose_family_id"]))
        records.append(
            {
                "query_index": int(observed["query_index"]),
                "decision": decision,
                "path": str(render_path.resolve()),
                "sha256": observed["source_record_sha256"],
            }
        )
    output = {
        "schema": "lafgs_v14_observer_split_certified_view",
        "version": 1,
        "view_role": args.view_role,
        "observer_role": source["role"],
        "source_view_role": certified["view_role"],
        "uses_test_queries": False,
        "map_mutation_count": 0,
        "accepted_query_row_policy": "v2_row_valid_only",
        "query_count": len(records),
        "pose_family_count": len(families),
        "decision_counts": decisions,
        "input": {
            "observer_batch": str(source_path),
            "observer_batch_sha256": sha256_file(source_path),
        },
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"query_count": len(records), "decision_counts": decisions}, indent=2))


if __name__ == "__main__":
    main()
