#!/usr/bin/env python3
"""Merge independently rendered certificate shards into one sealed registry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from common.hashing import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifests", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    payloads = [json.loads(path.read_text()) for path in args.manifests]
    frozen = ("schema", "version", "view_role", "uses_test_queries", "map_mutation_count")
    if any(tuple(item.get(key) for key in frozen) != tuple(payloads[0].get(key) for key in frozen) for item in payloads[1:]):
        raise ValueError("certified render shards violate the same-role contract")
    if payloads[0].get("uses_test_queries") is not False:
        raise ValueError("V13 confirmation cannot contain test queries")
    records = sorted(
        [record for payload in payloads for record in payload["records"]],
        key=lambda row: int(row["query_index"]),
    )
    if len({int(row["query_index"]) for row in records}) != len(records):
        raise ValueError("certified render shards overlap")
    decisions = {name: 0 for name in ("ACCEPT", "UNCERTAIN", "REJECT")}
    for row in records:
        decisions[row["decision"]] += 1
    output = dict(payloads[0])
    output.update(
        {
            "schema": "lafgs_v13_merged_certified_render_batch",
            "query_count": len(records),
            "decision_counts": decisions,
            "records": records,
            "render_shards": [
                {"path": str(path.resolve()), "sha256": sha256_file(path)}
                for path in args.manifests
            ],
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: output[key] for key in ("query_count", "decision_counts")}, indent=2))


if __name__ == "__main__":
    main()
