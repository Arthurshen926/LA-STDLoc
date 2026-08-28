#!/usr/bin/env python3
"""Run exact single-Anchor descriptor matching and PoseLib counterfactuals."""

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
    anchor_features = F.normalize(torch.as_tensor(state["anchor_features"]).float(), dim=1)
    anchor_xyz = torch.as_tensor(state["anchor_xyz"]).float()
    proposal_path = args.proposal.resolve()
    proposal = torch.load(proposal_path, map_location="cpu", weights_only=False)
    if not (
        proposal.get("schema") == "lafgs_v10_actionable_anchor_descriptor_proposal"
        and proposal.get("loo_used") is False
        and proposal.get("input", {}).get("map_sha256") == map_sha
    ):
        raise ValueError("single-action replay proposal differs from V10 M0")
    all_candidate_rows = torch.as_tensor(proposal["candidate_anchor_rows"]).long()
    all_candidate_descriptors = F.normalize(
        torch.as_tensor(proposal["candidate_descriptors"]).float(), dim=1
    )
    shard_selection = torch.arange(all_candidate_rows.numel()).remainder(
        args.shard_count
    ) == args.shard_index
    candidate_rows = all_candidate_rows[shard_selection]
    candidate_descriptors = all_candidate_descriptors[shard_selection]
    device = torch.device(args.device)
    candidate_descriptors_device = candidate_descriptors.to(device)

    feedback_records = []
    source_batches = []
    seen_queries = set()
    for batch_input in args.feedback_batches:
        batch_path = batch_input.resolve()
        batch = json.loads(batch_path.read_text())
        if not (
            batch.get("schema") == "lafgs_v9_no_loo_causal_feedback_batch"
            and batch.get("loo_used") is False
            and batch.get("input", {}).get("map_sha256") == map_sha
        ):
            raise ValueError("counterfactual feedback batch violates V10 contract")
        source_batches.append({"path": str(batch_path), "sha256": sha256_file(batch_path)})
        for item in batch["records"]:
            query = int(item["query_index"])
            if query in seen_queries:
                raise ValueError("counterfactual feedback shards overlap")
            seen_queries.add(query)
            path = Path(item["path"]).resolve()
            if sha256_file(path) != item["sha256"]:
                raise ValueError("counterfactual feedback record SHA256 differs")
            feedback_records.append(
                torch.load(path, map_location="cpu", weights_only=False)
            )

    output_records = []
    for index, record in enumerate(feedback_records):
        if record["certificate_decision"] != "ACCEPT":
            continue
        source = torch.load(
            record["source_record"], map_location="cpu", weights_only=False
        )
        query = F.normalize(
            torch.as_tensor(source["descriptors"], device=device).float(), dim=1
        )
        candidate_scores = (query @ candidate_descriptors_device.T).cpu()
        topk = torch.as_tensor(record["topk_anchor_rows"]).long()
        current = topk[:, 0]
        current_scores = (query.cpu() * anchor_features[current]).sum(1)
        second = topk[:, 1]
        second_scores = (query.cpu() * anchor_features[second]).sum(1)
        keypoints = torch.as_tensor(source["keypoints"]).float() + 0.5
        for column, anchor in enumerate(candidate_rows.tolist()):
            updated = current.clone()
            score = candidate_scores[:, column]
            owns = current == int(anchor)
            updated[owns] = second[owns]
            updated_score = torch.where(owns, second_scores, current_scores)
            wins = (score > updated_score) | (
                (score == updated_score) & (int(anchor) < updated)
            )
            updated[wins] = int(anchor)
            changed = updated != current
            if not bool(changed.any()):
                continue
            pose = standard_pose_replay(
                keypoints=keypoints,
                anchor_rows=updated,
                anchor_xyz=anchor_xyz,
                intrinsic=source["intrinsics"],
                ground_truth_w2c=source["pose_w2c"],
            )
            output_records.append(
                {
                    "anchor_row": int(anchor),
                    "query_index": int(record["query_index"]),
                    "pose_family_id": int(record["pose_family_id"]),
                    "changed_correspondence_count": int(changed.sum()),
                    "gained_correspondence_count": int((wins & ~owns).sum()),
                    "lost_correspondence_count": int((owns & ~wins).sum()),
                    "baseline_task_error": float(record["baseline"]["task_error"]),
                    "updated_task_error": float(pose["task_error"]),
                    "updated_translation_error_cm": float(pose["translation_error_cm"]),
                    "updated_rotation_error_deg": float(pose["rotation_error_deg"]),
                    "loo_used": False,
                }
            )
        if (index + 1) % 16 == 0:
            print(
                f"V10 descriptor replay shard {args.shard_index}: "
                f"{index + 1}/{len(feedback_records)}",
                flush=True,
            )
    output = {
        "schema": "lafgs_v10_single_descriptor_counterfactual_shard",
        "version": 1,
        "loo_used": False,
        "uses_test_queries": False,
        "map_mutation_count": 0,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "candidate_count": int(candidate_rows.numel()),
        "candidate_anchor_rows": candidate_rows.tolist(),
        "affected_pair_count": len(output_records),
        "map": str(map_path),
        "map_sha256": map_sha,
        "proposal": str(proposal_path),
        "proposal_sha256": sha256_file(proposal_path),
        "source_batches": source_batches,
        "records": output_records,
    }
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps({key: value for key, value in output.items() if key != "records"}, indent=2)
    )


if __name__ == "__main__":
    main()
