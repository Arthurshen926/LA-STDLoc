#!/usr/bin/env python3
"""Build bounded disjoint V10 descriptor groups from feedback co-occurrence."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from common.hashing import sha256_file
from map_learning.v10_anchor_descriptor_controller import (
    build_confusion_component_groups,
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
    parser.add_argument("--proposal", type=Path, required=True)
    parser.add_argument("--maximum-group-size", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    proposal_path = args.proposal.resolve()
    proposal = torch.load(proposal_path, map_location="cpu", weights_only=False)
    records = []
    batch_inputs = []
    for batch_input in args.feedback_batches:
        batch_path = batch_input.resolve()
        batch = json.loads(batch_path.read_text())
        batch_inputs.append({"path": str(batch_path), "sha256": sha256_file(batch_path)})
        records.extend(
            torch.load(item["path"], map_location="cpu", weights_only=False)
            for item in batch["records"]
        )
    groups = build_confusion_component_groups(
        candidate_anchor_rows=proposal["candidate_anchor_rows"],
        feedback_records=records,
        maximum_group_size=args.maximum_group_size,
    )
    payload = {
        "schema": "lafgs_v10_confusion_descriptor_groups",
        "version": 1,
        "loo_used": False,
        "uses_test_queries": False,
        "maximum_group_size": args.maximum_group_size,
        "group_count": len(groups),
        "groups": groups,
        "proposal": str(proposal_path),
        "proposal_sha256": sha256_file(proposal_path),
        "feedback_batches": batch_inputs,
    }
    _save(payload, args.output)
    print(
        json.dumps(
            {
                "schema": payload["schema"],
                "group_count": len(groups),
                "group_sizes": [len(group) for group in groups],
                "output": str(args.output.resolve()),
                "output_sha256": sha256_file(args.output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
