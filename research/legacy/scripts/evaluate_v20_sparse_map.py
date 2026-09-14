#!/usr/bin/env python3
"""Exact paired Top-1 + PoseLib replay for a V20 sparse descriptor map."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from common.hashing import sha256_file
from localization.matcher import global_cosine_top1
from map_learning.v20_sparse_descriptor import audit_materialized_sparse_action
from map_learning.v9_causal_feedback import standard_pose_replay


def _json_pose(pose: dict) -> dict:
    value = dict(pose)
    value["pose_w2c"] = value["pose_w2c"].tolist()
    return value


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--certified-batch", type=Path, required=True)
    parser.add_argument("--baseline-map", type=Path, required=True)
    parser.add_argument("--candidate-map", type=Path, required=True)
    parser.add_argument(
        "--expected-role",
        choices=("feedback_query", "confirmation_query"),
        required=True,
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--allow-analysis-only",
        action="store_true",
        help=(
            "Evaluate a clean nonzero teacher-unauthorized proposal without "
            "granting deployment authority."
        ),
    )
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.shard_count <= 0 or not 0 <= args.shard_index < args.shard_count:
        parser.error("invalid V20 shard index/count")

    batch_path = args.certified_batch.resolve()
    batch = json.loads(batch_path.read_text())
    role = batch.get("view_role", batch.get("role"))
    baseline_path = args.baseline_map.resolve()
    baseline_sha = sha256_file(baseline_path)
    if not (
        batch.get("schema") == "lafgs_v14_observer_split_certified_view"
        and batch.get("version") == 1
        and role == args.expected_role
        and batch.get("uses_test_queries") is False
        and batch.get("loo_used", False) is False
        and batch.get("map_mutation_count") == 0
        and batch.get("accepted_query_row_policy") == "v2_row_valid_only"
    ):
        raise ValueError("V20 replay requires a sealed V14 non-test view")
    observer_batch_path = Path(batch["input"]["observer_batch"]).resolve()
    if sha256_file(observer_batch_path) != batch["input"]["observer_batch_sha256"]:
        raise ValueError("V20 replay Observer-batch SHA256 differs")
    observer_batch = json.loads(observer_batch_path.read_text())
    observer_role = observer_batch.get("role")
    source_certified_path = Path(
        observer_batch.get("input", {}).get("certified_batch", "")
    ).resolve()
    source_certified_sha = str(
        observer_batch.get("input", {}).get("certified_batch_sha256", "")
    )
    if sha256_file(source_certified_path) != source_certified_sha:
        raise ValueError("V20 Observer source certified-batch SHA256 differs")
    source_certified = json.loads(source_certified_path.read_text())
    if not (
        observer_batch.get("schema")
        == "lafgs_v9_no_loo_causal_feedback_batch"
        and observer_batch.get("version") == 2
        and observer_batch.get("uses_test_queries") is False
        and observer_batch.get("loo_used") is False
        and observer_batch.get("accepted_query_row_policy")
        == "v2_row_valid_only"
        and observer_batch.get("training_rows_are_alternative_pose_entered_only")
        is True
        and observer_batch.get("clean_protection_has_explicit_query_rows") is True
        and observer_batch.get("input", {}).get("map_sha256") == baseline_sha
        and source_certified.get("view_role") == args.expected_role
        and (
            (args.expected_role == "feedback_query" and observer_role == "heldout_control")
            or (
                args.expected_role == "confirmation_query"
                and observer_role == "confirmation_observer"
            )
        )
    ):
        raise ValueError("V20 replay Observer split role or lineage differs")
    candidate_path = args.candidate_map.resolve()
    baseline = torch.load(baseline_path, map_location="cpu", weights_only=False)
    candidate = torch.load(candidate_path, map_location="cpu", weights_only=False)
    baseline_ids = torch.as_tensor(baseline["anchor_ids"]).long()
    candidate_ids = torch.as_tensor(candidate["anchor_ids"]).long()
    action = candidate.get("v20_sparse_descriptor_action", {})
    training_status = action.get("training_status")
    status_allows_replay = training_status == "REQUIRES_EXACT_POSE_CONTROL" or (
        args.allow_analysis_only
        and training_status == "ANALYSIS_ONLY_TEACHER_NOT_AUTHORIZED"
    )
    if not (
        baseline.get("schema") == "lafgs_materialized_anchor_map"
        and candidate.get("schema") == "lafgs_materialized_anchor_map"
        and torch.equal(baseline_ids, candidate_ids)
        and torch.equal(
            torch.as_tensor(baseline["anchor_xyz"]),
            torch.as_tensor(candidate["anchor_xyz"]),
        )
        and action.get("schema") == "lafgs_v20_sparse_descriptor_action"
        and action.get("baseline_map_sha256") == sha256_file(baseline_path)
        and action.get("query_descriptor_action") == "native_unchanged"
        and action.get("mode") == "positive_only"
        and status_allows_replay
        and action.get("positive_objective") == "per_positive_listwise_mean"
        and action.get("clean_protection_passed") is True
        and action.get("materialized_action_audit", {}).get("passed") is True
        and action.get("positive_win_nonregression_passed") is True
        and float(action.get("post_training_action_scale", 0.0)) > 0.0
    ):
        raise ValueError("V20 candidate map is not a descriptor-only Full-M0 action")
    arm = str(action["arm"])
    baseline_raw = torch.as_tensor(baseline["anchor_features"])
    candidate_raw = torch.as_tensor(candidate["anchor_features"])
    selected_rows = torch.as_tensor(action["selected_anchor_rows"]).long().reshape(-1)
    action_scales = torch.as_tensor(
        action.get("per_anchor_action_scales", [])
    ).float().reshape(-1)
    reported_angles = torch.as_tensor(
        action.get("per_anchor_observed_angle_deg", [])
    ).float().reshape(-1)
    if (
        baseline_raw.shape != candidate_raw.shape
        or baseline_raw.shape[0] != baseline_ids.numel()
        or baseline_raw.ndim != 2
        or not bool(torch.isfinite(baseline_raw.float()).all())
        or not bool(torch.isfinite(candidate_raw.float()).all())
        or bool((torch.linalg.norm(baseline_raw.float(), dim=1) <= 1e-8).any())
        or bool((torch.linalg.norm(candidate_raw.float(), dim=1) <= 1e-8).any())
        or selected_rows.numel() == 0
        or selected_rows.numel() > 4096
        or action_scales.numel() != selected_rows.numel()
        or reported_angles.numel() != selected_rows.numel()
        or not bool(torch.isfinite(action_scales).all())
        or bool((action_scales <= 0.0).any())
        or bool((action_scales > 1.0).any())
        or torch.unique(selected_rows).numel() != selected_rows.numel()
        or int(selected_rows.min()) < 0
        or int(selected_rows.max()) >= baseline_ids.numel()
    ):
        raise ValueError("V20 candidate descriptor bank contract differs")
    global_seed = float(action.get("global_seed_action_scale", 0.0))
    if (
        not 0.0 < global_seed <= float(action_scales.min())
        or
        abs(
            float(action_scales.min())
            - float(action["post_training_action_scale"])
        )
        > 1e-8
        or abs(
            float(action_scales.max())
            - float(action["maximum_applied_anchor_scale"])
        )
        > 1e-8
    ):
        raise ValueError("V20 candidate per-Anchor action scales are not bound")
    selected_mask = torch.zeros(baseline_ids.numel(), dtype=torch.bool)
    selected_mask[selected_rows] = True
    if not torch.equal(baseline_raw[~selected_mask], candidate_raw[~selected_mask]):
        raise ValueError("V20 candidate changed an unselected Anchor descriptor")
    selected_native = F.normalize(baseline_raw[selected_rows].float(), dim=1)
    selected_candidate = F.normalize(candidate_raw[selected_rows].float(), dim=1)
    selected_angles = torch.rad2deg(
        torch.acos(
            (selected_native * selected_candidate).sum(1).clamp(-1.0, 1.0)
        )
    )
    if (
        not torch.allclose(
            selected_angles.cpu(), reported_angles, atol=2e-3, rtol=1e-4
        )
        or float(selected_angles.max())
        > float(action["maximum_angle_deg"]) + 0.05
    ):
        raise ValueError("V20 candidate exceeds its descriptor angular cap")
    target = torch.device(args.device)
    evidence_path = Path(action["evidence"]).resolve()
    if sha256_file(evidence_path) != action["evidence_sha256"]:
        raise ValueError("V20 candidate evidence SHA256 differs")
    evidence = torch.load(evidence_path, map_location="cpu", weights_only=False)
    if not (
        evidence.get("schema") == "lafgs_v20_topk_competition_evidence"
        and int(evidence.get("version", 0)) >= 2
        and evidence.get("uses_test_queries") is False
        and evidence.get("loo_used") is False
        and evidence.get("inputs", {}).get("anchor_map_sha256") == baseline_sha
        and evidence.get("strong_feedback_authorized")
        == action.get("strong_feedback_authorized")
        and evidence.get("inputs", {}).get("design_batch_sha256")
        == action.get("design_batch_sha256")
        and sorted(int(value) for value in evidence.get("design_query_indices", []))
        == list(action.get("design_query_indices", []))
        and sorted(
            int(value) for value in evidence.get("design_pose_family_ids", [])
        )
        == list(action.get("design_pose_family_ids", []))
        and sorted(
            str(value)
            for value in evidence.get("design_source_record_sha256s", [])
        )
        == list(action.get("design_source_record_sha256s", []))
    ):
        raise ValueError("V20 candidate evidence lineage differs")
    materialized_audit = audit_materialized_sparse_action(
        baseline_anchor_features=baseline_raw,
        candidate_anchor_features=candidate_raw,
        selected_anchor_rows=selected_rows,
        evidence=evidence,
        clean_margin_slack=float(action["clean_margin_slack"]),
        maximum_angle_deg=float(action["maximum_angle_deg"]),
        device=target,
    )
    if not materialized_audit["passed"]:
        raise ValueError("V20 materialized candidate fails its clean action audit")
    baseline_features = F.normalize(
        baseline_raw.to(device=target).float(), dim=1
    )
    candidate_features = F.normalize(
        candidate_raw.to(device=target).float(), dim=1
    )
    xyz = torch.as_tensor(baseline["anchor_xyz"]).float()

    selected = [
        item
        for index, item in enumerate(batch["records"])
        if index % args.shard_count == args.shard_index
    ]
    records = []
    for local_index, item in enumerate(selected):
        source_path = Path(item["path"]).resolve()
        if sha256_file(source_path) != item["sha256"]:
            raise ValueError("V20 certified query record SHA256 differs")
        source = torch.load(source_path, map_location="cpu", weights_only=False)
        if int(source["query_index"]) != int(item["query_index"]):
            raise ValueError("V20 certified query index differs from its record")
        certificate = source["certificate"]
        if certificate["decision"] != "ACCEPT":
            continue
        # Plant replay deliberately retains every detected row.  The V2-valid
        # mask controls repair supervision, not deployed localization input.
        query = F.normalize(
            torch.as_tensor(source["descriptors"], device=target).float(), dim=1
        )
        keypoints = torch.as_tensor(source["keypoints"]).float() + 0.5
        if (
            query.ndim != 2
            or query.shape[1] != baseline_features.shape[1]
            or keypoints.shape != (query.shape[0], 2)
            or not bool(torch.isfinite(query).all())
            or not bool(torch.isfinite(keypoints).all())
        ):
            raise ValueError("V20 certified Query rows are invalid")
        baseline_match = global_cosine_top1(
            query, baseline_features, anchor_descriptors_normalized=True
        )
        candidate_match = global_cosine_top1(
            query, candidate_features, anchor_descriptors_normalized=True
        )
        baseline_pose = _json_pose(
            standard_pose_replay(
                keypoints=keypoints,
                anchor_rows=baseline_match.anchor_indices.cpu(),
                anchor_xyz=xyz,
                intrinsic=source["intrinsics"],
                ground_truth_w2c=source["pose_w2c"],
            )
        )
        candidate_pose = _json_pose(
            standard_pose_replay(
                keypoints=keypoints,
                anchor_rows=candidate_match.anchor_indices.cpu(),
                anchor_xyz=xyz,
                intrinsic=source["intrinsics"],
                ground_truth_w2c=source["pose_w2c"],
            )
        )
        valid = torch.as_tensor(certificate["row_valid"]).bool()
        candidate_pose["top1_changed_count"] = int(
            (candidate_match.anchor_indices != baseline_match.anchor_indices).sum()
        )
        records.append(
            {
                "query_index": int(source["query_index"]),
                "pose_family_id": int(source["pose_family_id"]),
                "source_record_sha256": str(item["sha256"]),
                "plant_query_row_count": int(query.shape[0]),
                "repair_eligible_source_row_count": int(valid.sum()),
                "invalid_or_unknown_source_row_count": int((~valid).sum()),
                "baseline": baseline_pose,
                arm: candidate_pose,
            }
        )
        if (local_index + 1) % 4 == 0 or local_index + 1 == len(selected):
            print(
                f"V20 {args.expected_role} shard {args.shard_index}: "
                f"{local_index + 1}/{len(selected)}",
                flush=True,
            )
    output = {
        "schema": "lafgs_v20_sparse_map_replay_shard",
        "version": 1,
        "uses_test_queries": False,
        "loo_used": False,
        "plant_row_policy": "all_detected_rows",
        "repair_row_policy": "v2_valid_decisive_only",
        "evaluation_role": args.expected_role,
        "candidate_arm": arm,
        "strong_feedback_authorized": bool(action["strong_feedback_authorized"]),
        "descriptor_training_safe": bool(materialized_audit["passed"]),
        "materialized_action_audit": materialized_audit,
        "analysis_only": training_status != "REQUIRES_EXACT_POSE_CONTROL",
        "shard_index": int(args.shard_index),
        "shard_count": int(args.shard_count),
        "source_query_count": len(batch["records"]),
        "accepted_query_count": len(records),
        "input": {
            "certified_batch": str(batch_path),
            "certified_batch_sha256": sha256_file(batch_path),
            "observer_batch": str(observer_batch_path),
            "observer_batch_sha256": sha256_file(observer_batch_path),
            "baseline_map": str(baseline_path),
            "baseline_map_sha256": sha256_file(baseline_path),
            "candidate_map": str(candidate_path),
            "candidate_map_sha256": sha256_file(candidate_path),
        },
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
