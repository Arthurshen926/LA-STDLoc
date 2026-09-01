#!/usr/bin/env python3
"""Audit full/active-pool sufficiency against V19 truth and Top-L competition."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path

import torch

from common.hashing import sha256_file
from map_learning.v19_full_pool_sufficiency import (
    audit_full_pool_sufficiency_rows,
)
from map_learning.v9_causal_feedback import standard_pose_replay


def _atomic_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _oracle_pose(
    *,
    truth: dict,
    keypoints: torch.Tensor,
    active_mask: torch.Tensor,
    anchor_xyz: torch.Tensor,
    intrinsic: torch.Tensor,
    pose_w2c: torch.Tensor,
) -> tuple[dict | None, int]:
    offsets = torch.as_tensor(truth["truth_offsets"]).long()
    anchors = torch.as_tensor(truth["truth_anchor_rows"]).long()
    rows = []
    selected = []
    for row in range(int(truth["row_count"])):
        start, stop = int(offsets[row]), int(offsets[row + 1])
        if stop <= start:
            continue
        available = anchors[start:stop][active_mask[anchors[start:stop]]]
        if available.numel():
            rows.append(row)
            selected.append(int(available[0]))
    if len(rows) < 4:
        return None, len(rows)
    return (
        standard_pose_replay(
            keypoints=keypoints[torch.tensor(rows)],
            anchor_rows=torch.tensor(selected),
            anchor_xyz=anchor_xyz,
            intrinsic=intrinsic,
            ground_truth_w2c=pose_w2c,
        ),
        len(rows),
    )


def _success(pose: dict | None) -> bool:
    return bool(
        pose is not None
        and float(pose["translation_error_cm"]) < 5.0
        and float(pose["rotation_error_deg"]) < 5.0
    )


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-shards", nargs="+", type=Path, required=True)
    parser.add_argument("--observer-manifests", nargs="+", type=Path, required=True)
    parser.add_argument("--full-map", type=Path, required=True)
    parser.add_argument("--active-map", type=Path)
    parser.add_argument("--tier", default="tier_c")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    state = torch.load(args.full_map, map_location="cpu", weights_only=False)
    anchor_ids = torch.as_tensor(state["anchor_ids"]).long()
    active_mask = torch.ones(anchor_ids.numel(), dtype=torch.bool)
    active_path = None
    if args.active_map is not None:
        active_path = args.active_map.resolve()
        active_state = torch.load(active_path, map_location="cpu", weights_only=False)
        active_ids = torch.as_tensor(active_state["anchor_ids"]).long()
        if not bool(torch.isin(active_ids, anchor_ids).all()):
            raise ValueError("V19 active map is not a full-map subset")
        active_mask = torch.isin(anchor_ids, active_ids)
    equivalence = torch.as_tensor(
        state.get("fine_identity_ids", torch.arange(anchor_ids.numel()))
    ).long()

    observer_records = {}
    observer_inputs = []
    for manifest_path in args.observer_manifests:
        manifest = json.loads(manifest_path.read_text())
        if not (
            manifest.get("schema") == "lafgs_v9_no_loo_causal_feedback_batch"
            and manifest.get("uses_test_queries") is False
            and manifest.get("loo_used") is False
        ):
            raise ValueError("V19 full-pool observer contract differs")
        observer_inputs.append(
            {"path": str(manifest_path.resolve()), "sha256": sha256_file(manifest_path)}
        )
        for item in manifest["records"]:
            observer_records[int(item["query_index"])] = Path(item["path"]).resolve()

    teacher_records = []
    teacher_inputs = []
    authorization = None
    validation_sha = None
    for shard_path in args.teacher_shards:
        shard = torch.load(shard_path, map_location="cpu", weights_only=False)
        if not (
            shard.get("schema") == "lafgs_v19_novel_track_extension_shard"
            and shard.get("uses_test_queries") is False
            and shard.get("loo_used") is False
            and shard.get("reference_available_for_novel_query") is False
        ):
            raise ValueError("V19 full-pool teacher shard contract differs")
        if authorization is None:
            authorization = dict(shard["tier_action_authorization"])
            validation_sha = shard["inputs"]["teacher_validation_sha256"]
        elif (
            authorization != shard["tier_action_authorization"]
            or validation_sha != shard["inputs"]["teacher_validation_sha256"]
        ):
            raise ValueError("V19 teacher shards use different calibration")
        teacher_records.extend(shard["records"])
        teacher_inputs.append(
            {"path": str(shard_path.resolve()), "sha256": sha256_file(shard_path)}
        )
    if args.tier not in authorization:
        raise ValueError("requested V19 teacher tier is absent")
    candidate_pool_deficit_authorized = bool(authorization.get("tier_a", False))

    totals = defaultdict(int)
    records = []
    for teacher in sorted(teacher_records, key=lambda item: int(item["query_index"])):
        query_index = int(teacher["query_index"])
        observer_path = observer_records.get(query_index)
        if observer_path is None:
            raise ValueError("V19 teacher Query has no competition record")
        observer = torch.load(observer_path, map_location="cpu", weights_only=False)
        source_path = Path(teacher["source_record"])
        source = torch.load(source_path, map_location="cpu", weights_only=False)
        teacher_rows = torch.as_tensor(teacher["source_query_rows"]).long()
        observer_rows = torch.as_tensor(observer["source_query_rows"]).long()
        lookup = {int(row): local for local, row in enumerate(observer_rows.tolist())}
        if not all(int(row) in lookup for row in teacher_rows.tolist()):
            raise ValueError("V19 teacher rows are absent from frozen competition")
        local = torch.tensor([lookup[int(row)] for row in teacher_rows.tolist()])
        retrieved = torch.as_tensor(observer["topk_anchor_rows"]).long()[local]
        truth = teacher["truth_tiers"][args.tier]
        row_audit = audit_full_pool_sufficiency_rows(
            truth=truth,
            projection_candidate_graph=teacher["projection_candidate_graph"],
            retrieved_anchor_rows=retrieved,
            active_anchor_mask=active_mask,
            equivalence_class_ids=equivalence,
            candidate_pool_deficit_authorized=candidate_pool_deficit_authorized,
        )
        keypoints = torch.as_tensor(source["keypoints"]).float()[teacher_rows] + 0.5
        full_oracle, full_count = _oracle_pose(
            truth=truth,
            keypoints=keypoints,
            active_mask=torch.ones_like(active_mask),
            anchor_xyz=state["anchor_xyz"],
            intrinsic=source["intrinsics"],
            pose_w2c=source["pose_w2c"],
        )
        active_oracle, active_count = _oracle_pose(
            truth=truth,
            keypoints=keypoints,
            active_mask=active_mask,
            anchor_xyz=state["anchor_xyz"],
            intrinsic=source["intrinsics"],
            pose_w2c=source["pose_w2c"],
        )
        baseline_success = bool(observer["baseline_success"])
        full_success = _success(full_oracle)
        active_success = _success(active_oracle)
        for key, value in row_audit["counts"].items():
            totals[key] += int(value)
        totals["query_count"] += 1
        totals["baseline_success_count"] += int(baseline_success)
        totals["full_pool_oracle_success_count"] += int(full_success)
        totals["active_map_oracle_success_count"] += int(active_success)
        totals["full_pool_oracle_recovery_count"] += int(
            not baseline_success and full_success
        )
        totals["active_map_selection_loss_query_count"] += int(
            full_success and not active_success
        )
        totals["full_pool_oracle_geometry_set_failure_count"] += int(
            full_oracle is not None and not full_success
        )
        totals["insufficient_decisive_truth_query_count"] += int(full_oracle is None)
        records.append(
            {
                "query_index": query_index,
                "pose_family_id": int(teacher["pose_family_id"]),
                "row_audit": row_audit,
                "baseline_pose": observer["baseline"],
                "full_pool_oracle_pose": full_oracle,
                "active_map_oracle_pose": active_oracle,
                "full_pool_oracle_correspondence_count": full_count,
                "active_map_oracle_correspondence_count": active_count,
            }
        )
    artifact = {
        "schema": "lafgs_v19_full_pool_sufficiency_audit",
        "version": 1,
        "uses_test_queries": False,
        "loo_used": False,
        "teacher_tier": args.tier,
        "teacher_tier_action_authorized": bool(authorization[args.tier]),
        "candidate_pool_deficit_authorized": candidate_pool_deficit_authorized,
        "active_map_is_full_map": args.active_map is None,
        "totals": dict(totals),
        "records": records,
        "inputs": {
            "teacher_shards": teacher_inputs,
            "observer_manifests": observer_inputs,
            "full_map": str(args.full_map.resolve()),
            "full_map_sha256": sha256_file(args.full_map),
            "active_map": None if active_path is None else str(active_path),
            "active_map_sha256": None if active_path is None else sha256_file(active_path),
            "teacher_validation_sha256": validation_sha,
        },
    }
    _atomic_save(artifact, args.output.resolve())
    report = {key: value for key, value in artifact.items() if key != "records"}
    report["output"] = str(args.output.resolve())
    report["output_sha256"] = sha256_file(args.output)
    args.output.with_suffix(".json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
