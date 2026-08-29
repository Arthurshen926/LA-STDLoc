#!/usr/bin/env python3
"""Exact multi-map replay on a sealed V2-certified novel-query batch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from common.hashing import sha256_file
from localization.matcher import global_cosine_top1
from map_learning.v8_safety_actions import certified_feedback_row_mask
from map_learning.v9_causal_feedback import standard_pose_replay


def _parse_candidate(value: str) -> tuple[str, Path]:
    name, separator, path = value.partition("=")
    if not separator or not name or not path or not name.replace("_", "").isalnum():
        raise argparse.ArgumentTypeError("candidate must be NAME=/absolute/map.pt")
    return name, Path(path)


def _pose_json(result: dict) -> dict:
    result = dict(result)
    result["pose_w2c"] = result["pose_w2c"].tolist()
    return result


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--certified-batch", type=Path, required=True)
    parser.add_argument("--expected-role", choices=("feedback_query", "confirmation_query"), required=True)
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

    batch_path = args.certified_batch.resolve()
    batch = json.loads(batch_path.read_text())
    if not (
        batch.get("view_role") == args.expected_role
        and batch.get("uses_test_queries") is False
        and batch.get("map_mutation_count") == 0
        and batch.get("schema")
        in {"lafgs_v7_certified_clean_render_batch", "lafgs_v13_merged_certified_render_batch"}
    ):
        raise ValueError("V15 evaluation requires a sealed V2-certified batch")
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
                torch.as_tensor(payload["anchor_features"], device=target).float(),
                dim=1,
            ),
        }

    baseline = load_map(args.baseline_map)
    candidates = {name: load_map(path) for name, path in args.candidate}
    baseline_ids = set(baseline["ids"].tolist())
    if any(not set(item["ids"].tolist()).issubset(baseline_ids) for item in candidates.values()):
        raise ValueError("V15 candidate is not an active subset of the frozen M0")

    selected = [
        item
        for index, item in enumerate(batch["records"])
        if index % args.shard_count == args.shard_index
    ]
    records = []
    for index, item in enumerate(selected):
        record_path = Path(item["path"]).resolve()
        if sha256_file(record_path) != item["sha256"]:
            raise ValueError("certified render record SHA256 differs")
        source = torch.load(record_path, map_location="cpu", weights_only=False)
        if source["certificate"]["decision"] != "ACCEPT":
            continue
        valid = certified_feedback_row_mask(source["certificate"])
        rows = torch.nonzero(valid, as_tuple=False).reshape(-1)
        query = F.normalize(
            torch.as_tensor(source["descriptors"])[rows].to(target).float(), dim=1
        )
        keypoints = torch.as_tensor(source["keypoints"])[rows].float() + 0.5

        def replay(map_state: dict) -> tuple[dict, torch.Tensor]:
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
            return pose, map_state["ids"][matches.anchor_indices.cpu()]

        baseline_pose, baseline_match_ids = replay(baseline)
        row = {
            "query_index": int(source["query_index"]),
            "pose_family_id": int(source["pose_family_id"]),
            "valid_query_row_count": int(rows.numel()),
            "baseline": baseline_pose,
        }
        for name, map_state in candidates.items():
            pose, candidate_ids = replay(map_state)
            pose["top1_changed_count"] = int((candidate_ids != baseline_match_ids).sum())
            row[name] = pose
        records.append(row)
        if (index + 1) % 8 == 0 or index + 1 == len(selected):
            print(
                f"V15 certified curve shard {args.shard_index}: "
                f"{index + 1}/{len(selected)}",
                flush=True,
            )

    output = {
        "schema": "lafgs_v14_active_set_curve_shard",
        "version": 2,
        "uses_test_queries": False,
        "loo_used": False,
        "accepted_query_row_policy": "v2_row_valid_only",
        "evaluation_role": args.expected_role,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "candidate_arms": list(candidates),
        "input": {
            "certified_batch": str(batch_path),
            "certified_batch_sha256": sha256_file(batch_path),
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

