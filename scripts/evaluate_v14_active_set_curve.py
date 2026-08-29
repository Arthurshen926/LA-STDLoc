#!/usr/bin/env python3
"""Exact Top-1 + PoseLib evaluation of delete-only active-set candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from common.hashing import sha256_file
from localization.matcher import global_cosine_top1
from map_learning.v9_causal_feedback import standard_pose_replay


def _pose_json(result: dict) -> dict:
    result = dict(result)
    result["pose_w2c"] = result["pose_w2c"].tolist()
    return result


def _parse_candidate(value: str) -> tuple[str, Path]:
    name, separator, path = value.partition("=")
    if not separator or not name or not path or not name.replace("_", "").isalnum():
        raise argparse.ArgumentTypeError("candidate must be NAME=/absolute/map.pt")
    return name, Path(path)


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observer-batch", type=Path, required=True)
    parser.add_argument("--baseline-map", type=Path, required=True)
    parser.add_argument("--candidate", type=_parse_candidate, action="append", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.shard_count <= 0 or not 0 <= args.shard_index < args.shard_count:
        parser.error("invalid shard index/count")
    if len({name for name, _ in args.candidate}) != len(args.candidate):
        parser.error("candidate names must be unique")

    observer_path = args.observer_batch.resolve()
    observer = json.loads(observer_path.read_text())
    if not (
        observer.get("schema") == "lafgs_v9_no_loo_causal_feedback_batch"
        and observer.get("uses_test_queries") is False
        and observer.get("loo_used") is False
        and observer.get("accepted_query_row_policy") == "v2_row_valid_only"
    ):
        raise ValueError("evaluation requires corrected no-LOO Observer records")
    target = torch.device(args.device)

    def load_map(path: Path) -> dict:
        path = path.resolve()
        payload = torch.load(path, map_location="cpu", weights_only=False)
        return {
            "path": path,
            "sha256": sha256_file(path),
            "ids": torch.as_tensor(payload["anchor_ids"]).long(),
            "xyz": torch.as_tensor(payload["anchor_xyz"]).float(),
            "features": F.normalize(
                torch.as_tensor(payload["anchor_features"], device=target).float(), dim=1
            ),
        }

    baseline = load_map(args.baseline_map)
    candidates = {name: load_map(path) for name, path in args.candidate}
    baseline_ids = set(baseline["ids"].tolist())
    if any(not set(item["ids"].tolist()).issubset(baseline_ids) for item in candidates.values()):
        raise ValueError("active-set candidate is not a subset of the baseline map")

    selected = [
        row
        for index, row in enumerate(observer["records"])
        if index % args.shard_count == args.shard_index
    ]
    records = []
    for local_index, item in enumerate(selected):
        record_path = Path(item["path"])
        if sha256_file(record_path) != item["sha256"]:
            raise ValueError("Observer record SHA256 differs")
        observed = torch.load(record_path, map_location="cpu", weights_only=False)
        if observed["certificate_decision"] != "ACCEPT":
            continue
        source_path = Path(observed["source_record"])
        if sha256_file(source_path) != observed["source_record_sha256"]:
            raise ValueError("source feedback record SHA256 differs")
        source = torch.load(source_path, map_location="cpu", weights_only=False)
        rows = torch.as_tensor(observed["source_query_rows"]).long()
        query = F.normalize(
            torch.as_tensor(source["descriptors"])[rows].to(target).float(), dim=1
        )
        keypoints = torch.as_tensor(source["keypoints"])[rows].float() + 0.5

        def replay(map_state: dict) -> dict:
            matches = global_cosine_top1(
                query, map_state["features"], anchor_descriptors_normalized=True
            )
            pose = _pose_json(
                standard_pose_replay(
                    keypoints=keypoints,
                    anchor_rows=matches.anchor_indices.cpu(),
                    anchor_xyz=map_state["xyz"],
                    intrinsic=source["intrinsics"],
                    ground_truth_w2c=source["pose_w2c"],
                )
            )
            pose["matched_anchor_ids"] = map_state["ids"][matches.anchor_indices.cpu()].tolist()
            return pose

        baseline_pose = replay(baseline)
        row = {
            "query_index": int(observed["query_index"]),
            "pose_family_id": int(observed["pose_family_id"]),
            "valid_query_row_count": int(rows.numel()),
            "baseline": baseline_pose,
        }
        baseline_match_ids = torch.as_tensor(baseline_pose.pop("matched_anchor_ids")).long()
        for name, map_state in candidates.items():
            pose = replay(map_state)
            candidate_ids = torch.as_tensor(pose.pop("matched_anchor_ids")).long()
            pose["top1_changed_count"] = int((candidate_ids != baseline_match_ids).sum())
            row[name] = pose
        records.append(row)
        if (local_index + 1) % 8 == 0:
            print(f"V14 active-set shard {args.shard_index}: {local_index + 1}/{len(selected)}", flush=True)

    output = {
        "schema": "lafgs_v14_active_set_curve_shard",
        "version": 1,
        "uses_test_queries": False,
        "loo_used": False,
        "accepted_query_row_policy": "v2_row_valid_only",
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "candidate_arms": list(candidates),
        "input": {
            "observer_batch": str(observer_path),
            "observer_batch_sha256": sha256_file(observer_path),
            "baseline_map": str(baseline["path"]),
            "baseline_map_sha256": baseline["sha256"],
            "candidates": {
                name: {"path": str(item["path"]), "sha256": item["sha256"]}
                for name, item in candidates.items()
            },
        },
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
