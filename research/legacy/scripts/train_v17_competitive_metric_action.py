#!/usr/bin/env python3
"""Train a bounded metric from the V16 active winner/runner-up state."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from common.hashing import sha256_file
from map_learning.v17_competitive_metric import build_competitive_metric_evidence
from map_learning.v9_metric_controller import metric_artifact, train_v9_shared_metric


def _save(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--competitive-state", type=Path, required=True)
    parser.add_argument("--design-batch", type=Path, required=True)
    parser.add_argument("--baseline-map", type=Path, required=True)
    parser.add_argument("--active-set-map", type=Path, required=True)
    parser.add_argument("--expected-baseline-map-sha256", required=True)
    parser.add_argument("--minimum-negative-pose-families", type=int, default=2)
    parser.add_argument("--maximum-repair-rows-per-query", type=int, default=256)
    parser.add_argument("--maximum-protection-rows-per-query", type=int, default=256)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--maximum-residual-norm", type=float, default=0.05)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--clean-protection-weight", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=1720260829)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)

    map_path = args.baseline_map.resolve()
    map_sha = sha256_file(map_path)
    if map_sha != args.expected_baseline_map_sha256:
        raise ValueError("V17 frozen baseline map SHA256 differs")
    baseline = torch.load(map_path, map_location="cpu", weights_only=False)
    active_path = args.active_set_map.resolve()
    active_map = torch.load(active_path, map_location="cpu", weights_only=False)
    baseline_ids = torch.as_tensor(baseline["anchor_ids"]).long()
    active_ids = torch.as_tensor(active_map["anchor_ids"]).long()
    if not set(active_ids.tolist()).issubset(set(baseline_ids.tolist())):
        raise ValueError("V17 active map is not a delete-only subset of M0")
    active_mask = torch.isin(baseline_ids, active_ids)

    state_path = args.competitive_state.resolve()
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    if not (
        state.get("schema") == "lafgs_v16_topl_competitive_state"
        and state.get("uses_test_queries") is False
        and state.get("loo_used") is False
    ):
        raise ValueError("V17 requires the sealed V16 competition state")
    design_path = args.design_batch.resolve()
    design = json.loads(design_path.read_text())
    if not (
        design.get("schema") == "lafgs_v9_no_loo_causal_feedback_batch"
        and design.get("role") == "controller_design"
        and design.get("uses_test_queries") is False
        and design.get("loo_used") is False
        and design.get("accepted_query_row_policy") == "v2_row_valid_only"
    ):
        raise ValueError("V17 requires the frozen design-only V2-valid observer batch")
    design_records = {int(item["query_index"]): item for item in design["records"]}
    descriptors = {}
    metadata = {}
    for query in state["queries"]:
        query_index = int(query["query_index"])
        item = design_records[query_index]
        record_path = Path(item["path"]).resolve()
        if sha256_file(record_path) != item["sha256"]:
            raise ValueError("V17 design observer record SHA256 differs")
        record = torch.load(record_path, map_location="cpu", weights_only=False)
        source_path = Path(record["source_record"]).resolve()
        if sha256_file(source_path) != record["source_record_sha256"]:
            raise ValueError("V17 source render record SHA256 differs")
        source = torch.load(source_path, map_location="cpu", weights_only=False)
        rows = torch.as_tensor(record["source_query_rows"]).long()
        if rows.numel() != torch.as_tensor(query["keypoints"]).shape[0]:
            raise ValueError("V17 competition/source row contract differs")
        descriptors[query_index] = torch.as_tensor(source["descriptors"])[rows].float()
        metadata[query_index] = {
            "can_train_metric": bool(record["can_train_metric"]),
            "actual_task_gain": float(record["actual_task_gain"]),
            "category": str(record["category"]),
        }

    evidence = build_competitive_metric_evidence(
        competition_queries=state["queries"],
        query_descriptors=descriptors,
        action_metadata=metadata,
        active_anchor_mask=active_mask,
        minimum_negative_pose_families=args.minimum_negative_pose_families,
        maximum_repair_rows_per_query=args.maximum_repair_rows_per_query,
        maximum_protection_rows_per_query=args.maximum_protection_rows_per_query,
    )
    metric, training_report = train_v9_shared_metric(
        anchor_features=baseline["anchor_features"],
        query_descriptors=evidence["repair_query_descriptors"],
        positive_anchor_rows=evidence["repair_positive_anchor_rows"],
        negative_anchor_rows=evidence["repair_negative_anchor_rows"],
        sample_weights=evidence["repair_sample_weights"],
        clean_query_descriptors=evidence["protection_query_descriptors"],
        clean_positive_anchor_rows=evidence["protection_positive_anchor_rows"],
        clean_negative_anchor_rows=evidence["protection_negative_anchor_rows"],
        clean_initial_margin=evidence["protection_initial_margin"],
        clean_sample_weights=evidence["protection_sample_weights"],
        rank=args.rank,
        maximum_residual_norm=args.maximum_residual_norm,
        steps=args.steps,
        learning_rate=args.learning_rate,
        clean_protection_weight=args.clean_protection_weight,
        seed=args.seed,
        device=args.device,
    )
    training_report = {
        **training_report,
        "schema": "lafgs_v17_active_competitive_metric_training_report",
        "active_winner_runner_up_aligned": True,
        "repair_pose_family_count": evidence["repair_pose_family_count"],
        "protection_pose_family_count": evidence["protection_pose_family_count"],
    }
    args.output_dir.mkdir(parents=True)
    evidence_path = args.output_dir / "competitive_metric_evidence.pt"
    _save(evidence, evidence_path)
    artifact = metric_artifact(
        metric,
        anchor_ids=baseline_ids,
        map_path=str(map_path),
        map_sha256=map_sha,
        training_report=training_report,
    )
    artifact.update(
        {
            "training_protocol": "v17_active_competitive_winner_runner_up",
            "competitive_state": str(state_path),
            "competitive_state_sha256": sha256_file(state_path),
            "active_set_map": str(active_path),
            "active_set_map_sha256": sha256_file(active_path),
        }
    )
    metric_path = args.output_dir / "shared_metric.pt"
    _save(artifact, metric_path)
    report = {
        "schema": "lafgs_v17_active_competitive_metric_action_report",
        "version": 1,
        "status": "PROPOSAL",
        "uses_test_queries": False,
        "loo_used": False,
        "feedback_descriptors_copied_into_map": False,
        "geometry_mutation_count": 0,
        "repair_row_count": int(evidence["repair_positive_anchor_rows"].numel()),
        "protection_row_count": int(
            evidence["protection_positive_anchor_rows"].numel()
        ),
        "repair_pose_family_count": evidence["repair_pose_family_count"],
        "protection_pose_family_count": evidence["protection_pose_family_count"],
        "authorized_negative_anchor_count": evidence[
            "authorized_negative_anchor_count"
        ],
        "training": training_report,
        "inputs": {
            "baseline_map": str(map_path),
            "baseline_map_sha256": map_sha,
            "active_set_map": str(active_path),
            "active_set_map_sha256": sha256_file(active_path),
            "competitive_state": str(state_path),
            "competitive_state_sha256": sha256_file(state_path),
            "design_batch": str(design_path),
            "design_batch_sha256": sha256_file(design_path),
        },
        "output": {
            "metric": str(metric_path.resolve()),
            "metric_sha256": sha256_file(metric_path),
            "evidence": str(evidence_path.resolve()),
            "evidence_sha256": sha256_file(evidence_path),
        },
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
