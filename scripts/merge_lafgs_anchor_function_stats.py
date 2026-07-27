#!/usr/bin/env python3
"""Merge disjoint query shards of LaFGS anchor functional statistics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


SUM_FIELDS = (
    "winner_count",
    "clean_winner_count",
    "harm_winner_count",
    "inlier_count",
    "clean_topk_count",
    "clean_topk_query_count",
    "unique_topk_count",
    "image_bin_support_count",
    "sequence_clean_support",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    shards = [
        torch.load(path, map_location="cpu", weights_only=False)
        for path in args.inputs
    ]
    reference = shards[0]
    output = {
        key: value
        for key, value in reference.items()
        if key not in SUM_FIELDS
        and key not in {"query_support_bits", "query_indices", "query_diagnostics"}
    }
    for shard in shards[1:]:
        for key in (
            "anchor_count",
            "query_count_total",
            "sequence_names",
            "source_primitive_ids",
            "track_cluster_ids",
            "anchor_type",
        ):
            left, right = reference[key], shard[key]
            if torch.is_tensor(left):
                if not torch.equal(left, right):
                    raise ValueError(f"shards disagree on {key}")
            elif left != right:
                raise ValueError(f"shards disagree on {key}")
    for key in SUM_FIELDS:
        output[key] = sum(
            (torch.as_tensor(shard[key]) for shard in shards),
            torch.zeros_like(torch.as_tensor(reference[key])),
        )
    support = torch.zeros_like(reference["query_support_bits"])
    for shard in shards:
        support |= torch.as_tensor(shard["query_support_bits"])
    output["query_support_bits"] = support
    output["query_indices"] = torch.cat(
        [torch.as_tensor(shard["query_indices"]) for shard in shards]
    ).sort().values
    output["query_diagnostics"] = sorted(
        [
            item
            for shard in shards
            for item in shard["query_diagnostics"]
        ],
        key=lambda item: item["query_index"],
    )
    output["schema"] = "lafgs_anchor_function_stats"
    output["version"] = 1
    output["input_shards"] = [
        str(Path(path).resolve()) for path in args.inputs
    ]
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, path)
    summary = {
        "anchor_count": int(output["anchor_count"]),
        "query_count": int(output["query_indices"].numel()),
        "never_winner": int((output["winner_count"] == 0).sum()),
        "used_but_never_clean": int(
            (
                (output["winner_count"] > 0)
                & (output["clean_winner_count"] == 0)
            ).sum()
        ),
        "majority_harmful": int(
            (
                output["harm_winner_count"]
                > output["clean_winner_count"]
            ).sum()
        ),
        "zero_unique_support": int(
            (output["unique_topk_count"] == 0).sum()
        ),
        "inlier_supported": int((output["inlier_count"] > 0).sum()),
    }
    path.with_suffix(".json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
