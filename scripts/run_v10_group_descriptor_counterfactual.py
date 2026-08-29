#!/usr/bin/env python3
"""Run exact bounded-group descriptor matching and PoseLib counterfactuals."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from common.hashing import sha256_file
from map_learning.v9_causal_feedback import standard_pose_replay


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feedback-batches", type=Path, nargs="+", required=True)
    parser.add_argument("--proposal", type=Path, required=True)
    parser.add_argument("--groups", type=Path, required=True)
    parser.add_argument("--map", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
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
    features = F.normalize(torch.as_tensor(state["anchor_features"]).float(), dim=1)
    xyz = torch.as_tensor(state["anchor_xyz"]).float()
    proposal_path = args.proposal.resolve()
    proposal = torch.load(proposal_path, map_location="cpu", weights_only=False)
    descriptor_by_row = {
        int(row): F.normalize(
            torch.as_tensor(proposal["candidate_descriptors"])[index].float(), dim=0
        )
        for index, row in enumerate(
            torch.as_tensor(proposal["candidate_anchor_rows"]).long().tolist()
        )
    }
    groups_path = args.groups.resolve()
    group_state = torch.load(groups_path, map_location="cpu", weights_only=False)
    if not (
        group_state.get("schema") == "lafgs_v10_confusion_descriptor_groups"
        and group_state.get("loo_used") is False
        and group_state.get("proposal_sha256") == sha256_file(proposal_path)
    ):
        raise ValueError("V10 group registry differs from the descriptor proposal")
    selected_groups = [
        (index, group)
        for index, group in enumerate(group_state["groups"])
        if index % args.shard_count == args.shard_index
    ]
    device = torch.device(args.device)
    feedback_records = []
    source_batches = []
    for batch_input in args.feedback_batches:
        batch_path = batch_input.resolve()
        batch = json.loads(batch_path.read_text())
        source_batches.append({"path": str(batch_path), "sha256": sha256_file(batch_path)})
        feedback_records.extend(
            torch.load(item["path"], map_location="cpu", weights_only=False)
            for item in batch["records"]
        )
    output_records = []
    for index, record in enumerate(feedback_records):
        if record["certificate_decision"] != "ACCEPT":
            continue
        source = torch.load(
            record["source_record"], map_location="cpu", weights_only=False
        )
        source_query_rows = torch.as_tensor(record["source_query_rows"]).long()
        query = F.normalize(
            torch.as_tensor(source["descriptors"])[source_query_rows]
            .to(device=device)
            .float(),
            dim=1,
        )
        query_cpu = query.cpu()
        topk = torch.as_tensor(record["topk_anchor_rows"]).long()
        current = topk[:, 0]
        keypoints = torch.as_tensor(source["keypoints"])[source_query_rows].float() + 0.5
        if query.shape[0] != topk.shape[0]:
            raise ValueError("filtered Observer rows do not align with Top-K")
        for group_index, group in selected_groups:
            group_rows = torch.tensor(sorted(group), dtype=torch.long)
            membership = torch.isin(topk, group_rows)
            non_group_rank = (~membership).float().argmax(1).long()
            query_rows = torch.arange(topk.shape[0])
            base_rows = topk[query_rows, non_group_rank]
            base_scores = (query_cpu * features[base_rows]).sum(1)
            group_descriptors = torch.stack(
                [descriptor_by_row[int(row)] for row in group_rows.tolist()]
            ).to(device)
            group_scores = (query @ group_descriptors.T).cpu()
            best_group_scores, best_group_columns = group_scores.max(1)
            best_group_rows = group_rows[best_group_columns]
            wins = (best_group_scores > base_scores) | (
                (best_group_scores == base_scores) & (best_group_rows < base_rows)
            )
            updated = base_rows.clone()
            updated[wins] = best_group_rows[wins]
            changed = updated != current
            if not bool(changed.any()):
                continue
            pose = standard_pose_replay(
                keypoints=keypoints,
                anchor_rows=updated,
                anchor_xyz=xyz,
                intrinsic=source["intrinsics"],
                ground_truth_w2c=source["pose_w2c"],
            )
            output_records.append(
                {
                    "group_index": group_index,
                    "query_index": int(record["query_index"]),
                    "pose_family_id": int(record["pose_family_id"]),
                    "changed_correspondence_count": int(changed.sum()),
                    "baseline_task_error": float(record["baseline"]["task_error"]),
                    "updated_task_error": float(pose["task_error"]),
                    "loo_used": False,
                }
            )
        if (index + 1) % 16 == 0:
            print(
                f"V10 group replay shard {args.shard_index}: "
                f"{index + 1}/{len(feedback_records)}",
                flush=True,
            )
    output = {
        "schema": "lafgs_v10_group_descriptor_counterfactual_shard",
        "version": 1,
        "loo_used": False,
        "uses_test_queries": False,
        "map_mutation_count": 0,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "group_indices": [index for index, _ in selected_groups],
        "affected_pair_count": len(output_records),
        "map": str(map_path),
        "map_sha256": map_sha,
        "proposal": str(proposal_path),
        "proposal_sha256": sha256_file(proposal_path),
        "groups": str(groups_path),
        "groups_sha256": sha256_file(groups_path),
        "source_batches": source_batches,
        "records": output_records,
    }
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps({key: value for key, value in output.items() if key != "records"}, indent=2)
    )


if __name__ == "__main__":
    main()
