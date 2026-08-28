#!/usr/bin/env python3
"""Replay actual delete-one-Anchor effects on no-LOO V9 feedback queries."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import torch

from common.hashing import sha256_file
from map_learning.v9_causal_feedback import standard_pose_replay


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feedback-batches", type=Path, nargs="+", required=True)
    parser.add_argument("--map", type=Path, required=True)
    parser.add_argument("--minimum-pose-families", type=int, default=2)
    parser.add_argument("--maximum-candidates", type=int, default=1000)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.shard_count <= 0 or not 0 <= args.shard_index < args.shard_count:
        parser.error("invalid shard index/count")
    map_path = args.map.resolve()
    map_sha = sha256_file(map_path)
    state = torch.load(map_path, map_location="cpu", weights_only=False)
    xyz = torch.as_tensor(state["anchor_xyz"]).float()
    records = []
    source_batches = []
    seen = set()
    for batch_input in args.feedback_batches:
        batch_path = batch_input.resolve()
        batch = json.loads(batch_path.read_text())
        if not (
            batch.get("schema") == "lafgs_v9_no_loo_causal_feedback_batch"
            and batch.get("loo_used") is False
            and batch.get("uses_test_queries") is False
            and batch.get("input", {}).get("map_sha256") == map_sha
        ):
            raise ValueError("removal audit input violates the V9 no-LOO contract")
        source_batches.append({"path": str(batch_path), "sha256": sha256_file(batch_path)})
        for item in batch["records"]:
            query = int(item["query_index"])
            if query in seen:
                raise ValueError("causal observer shards overlap")
            seen.add(query)
            path = Path(item["path"])
            if sha256_file(path) != item["sha256"]:
                raise ValueError("causal observer record SHA256 differs")
            records.append(torch.load(path, map_location="cpu", weights_only=False))

    candidate_families: dict[int, set[int]] = defaultdict(set)
    candidate_queries: dict[int, set[int]] = defaultdict(set)
    candidate_gain: dict[int, float] = defaultdict(float)
    for record in records:
        if not record["can_train_metric"]:
            continue
        evidence = record["training_evidence"]
        family = int(record["pose_family_id"])
        query = int(record["query_index"])
        gain = float(evidence["actual_query_task_gain"])
        for anchor in torch.unique(
            torch.as_tensor(evidence["negative_anchor_rows"]).long()
        ).tolist():
            candidate_families[int(anchor)].add(family)
            candidate_queries[int(anchor)].add(query)
            candidate_gain[int(anchor)] += gain
    candidates = [
        anchor
        for anchor, families in candidate_families.items()
        if len(families) >= int(args.minimum_pose_families)
    ]
    candidates.sort(
        key=lambda anchor: (
            -candidate_gain[anchor],
            -len(candidate_families[anchor]),
            anchor,
        )
    )
    candidates = candidates[: int(args.maximum_candidates)]
    candidates = [
        anchor
        for index, anchor in enumerate(candidates)
        if index % args.shard_count == args.shard_index
    ]
    candidate_set = set(candidates)
    removal_records = []
    affected_query_count = 0
    for record_index, record in enumerate(records):
        if record["certificate_decision"] != "ACCEPT":
            continue
        topk = torch.as_tensor(record["topk_anchor_rows"]).long()
        affected = sorted(set(topk[:, 0].tolist()) & candidate_set)
        if not affected:
            continue
        affected_query_count += 1
        source = torch.load(
            record["source_record"], map_location="cpu", weights_only=False
        )
        keypoints = torch.as_tensor(source["keypoints"]).float() + 0.5
        for anchor in affected:
            replacement = topk[:, 0].clone()
            rows = torch.nonzero(replacement == anchor, as_tuple=False).reshape(-1)
            for row in rows.tolist():
                alternatives = topk[row][topk[row] != anchor]
                if alternatives.numel() == 0:
                    raise RuntimeError("Top-K removal replay has no replacement")
                replacement[row] = alternatives[0]
            removed = standard_pose_replay(
                keypoints=keypoints,
                anchor_rows=replacement,
                anchor_xyz=xyz,
                intrinsic=source["intrinsics"],
                ground_truth_w2c=source["pose_w2c"],
            )
            removal_records.append(
                {
                    "anchor_row": anchor,
                    "query_index": int(record["query_index"]),
                    "pose_family_id": int(record["pose_family_id"]),
                    "replaced_correspondence_count": int(rows.numel()),
                    "baseline_task_error": float(record["baseline"]["task_error"]),
                    "removed_task_error": float(removed["task_error"]),
                    "removed_translation_error_cm": float(
                        removed["translation_error_cm"]
                    ),
                    "removed_rotation_error_deg": float(removed["rotation_error_deg"]),
                    "loo_used": False,
                }
            )
        if (record_index + 1) % 16 == 0:
            print(
                f"V9 removal shard {args.shard_index}: scanned "
                f"{record_index + 1}/{len(records)}",
                flush=True,
            )
    output = {
        "schema": "lafgs_v9_actual_removal_replay_shard",
        "version": 1,
        "loo_used": False,
        "uses_test_queries": False,
        "map_mutation_count": 0,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "source_candidate_count_before_cap": len(candidate_families),
        "audited_candidate_count": len(candidates),
        "affected_query_count": affected_query_count,
        "source_batches": source_batches,
        "map": str(map_path),
        "map_sha256": map_sha,
        "records": removal_records,
    }
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps({key: value for key, value in output.items() if key != "records"}, indent=2)
    )


if __name__ == "__main__":
    main()
