#!/usr/bin/env python3
"""Materialize V10 actionable one-descriptor-per-Anchor candidates."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from common.hashing import sha256_file
from map_learning.v10_anchor_descriptor_controller import (
    propose_actionable_anchor_descriptors,
)


def _save(payload: dict, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feedback-batches", type=Path, nargs="+", required=True)
    parser.add_argument("--map", type=Path, required=True)
    parser.add_argument("--expected-map-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    map_path = args.map.resolve()
    map_sha = sha256_file(map_path)
    if map_sha != args.expected_map_sha256:
        raise ValueError("V10 M0 SHA256 differs")
    state = torch.load(map_path, map_location="cpu", weights_only=False)
    records = []
    batch_inputs = []
    seen_queries = set()
    for input_path in args.feedback_batches:
        batch_path = input_path.resolve()
        batch = json.loads(batch_path.read_text())
        if not (
            batch.get("schema") == "lafgs_v9_no_loo_causal_feedback_batch"
            and batch.get("loo_used") is False
            and batch.get("uses_test_queries") is False
            and batch.get("input", {}).get("map_sha256") == map_sha
        ):
            raise ValueError("V10 proposal input violates the no-LOO M0 contract")
        batch_inputs.append({"path": str(batch_path), "sha256": sha256_file(batch_path)})
        for item in batch["records"]:
            query = int(item["query_index"])
            if query in seen_queries:
                raise ValueError("V10 feedback shards overlap")
            seen_queries.add(query)
            path = Path(item["path"]).resolve()
            if sha256_file(path) != item["sha256"]:
                raise ValueError("V10 feedback record SHA256 differs")
            records.append(torch.load(path, map_location="cpu", weights_only=False))
    proposal = propose_actionable_anchor_descriptors(
        anchor_features=state["anchor_features"], feedback_records=records
    )
    proposal["input"] = {
        "map": str(map_path),
        "map_sha256": map_sha,
        "feedback_batches": batch_inputs,
    }
    args.output_dir.mkdir(parents=True)
    proposal_path = args.output_dir / "anchor_descriptor_proposal.pt"
    _save(proposal, proposal_path)
    report = {
        "schema": "lafgs_v10_anchor_descriptor_proposal_report",
        "version": 1,
        "loo_used": False,
        "uses_test_queries": False,
        "candidate_count": proposal["candidate_count"],
        "rejection_counts": proposal["rejection_counts"],
        "output": {
            "proposal": str(proposal_path.resolve()),
            "proposal_sha256": sha256_file(proposal_path),
        },
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
