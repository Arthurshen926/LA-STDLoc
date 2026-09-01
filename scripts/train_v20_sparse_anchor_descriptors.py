#!/usr/bin/env python3
"""Train and materialize a sparse map-side V20 descriptor proposal."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from common.hashing import sha256_file
from map_learning.metric import SharedLowRankMetric
from map_learning.v20_sparse_descriptor import (
    audit_materialized_sparse_action,
    train_sparse_anchor_descriptors,
)


def _save(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _jsonable(value: object) -> object:
    if isinstance(value, torch.Tensor):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _identity_metric(
    *,
    anchor_ids: torch.Tensor,
    map_path: Path,
    map_sha256: str,
    arm: str,
    photometric_contract: dict | None,
) -> dict:
    metric = SharedLowRankMetric(
        descriptor_dim=256, rank=1, max_residual_norm=0.0
    )
    with torch.no_grad():
        for parameter in metric.parameters():
            parameter.zero_()
    return {
        "schema": "lafgs_shared_metric_state",
        "version": 1,
        "protocol": "v20_sparse_anchor_native_query_identity",
        "step": 0,
        "metric_config": metric.export_config(),
        "metric_state_dict": metric.state_dict(),
        # The localization metric registry is keyed by the map's Anchor IDs,
        # not by descriptor-bank row number.  Full V2 M0 currently uses a
        # contiguous registry, but preserving the actual IDs keeps this shim
        # correct for every valid materialized Anchor map.
        "landmark_indices": torch.as_tensor(anchor_ids).long().cpu().clone(),
        "map_path": str(map_path),
        "map_sha256": str(map_sha256),
        "photometric_canonicalization_contract": photometric_contract,
        "loo_used": False,
        "feedback_descriptors_copied_into_map": False,
        "deployment_arm": arm,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-map", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=("positive_only", "positive_and_repeated_negative"),
        default="positive_only",
    )
    parser.add_argument("--maximum-angle-deg", type=float, default=5.0)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--clean-margin-slack", type=float, default=0.002)
    parser.add_argument("--clean-protection-weight", type=float, default=4.0)
    parser.add_argument("--angular-regularization-weight", type=float, default=0.01)
    parser.add_argument("--minimum-repair-margin-gain", type=float, default=0.001)
    parser.add_argument("--minimum-coordinate-ranking-gain", type=float, default=1e-5)
    parser.add_argument("--maximum-selected-anchor-count", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)

    map_path = args.baseline_map.resolve()
    map_sha = sha256_file(map_path)
    baseline = torch.load(map_path, map_location="cpu", weights_only=False)
    evidence_path = args.evidence.resolve()
    evidence = torch.load(evidence_path, map_location="cpu", weights_only=False)
    if not (
        baseline.get("schema") == "lafgs_materialized_anchor_map"
        and evidence.get("schema") == "lafgs_v20_topk_competition_evidence"
        and int(evidence.get("version", 0)) >= 2
        and evidence.get("uses_test_queries") is False
        and evidence.get("loo_used") is False
        and evidence.get("inputs", {}).get("anchor_map_sha256") == map_sha
    ):
        raise ValueError("V20 sparse descriptor input contract differs")
    design_rows = evidence.get("per_query", [])
    used_design_queries = {int(item["query_index"]) for item in design_rows}
    used_design_families = {int(item["pose_family_id"]) for item in design_rows}
    design_queries = sorted(
        {int(value) for value in evidence.get("design_query_indices", [])}
    )
    design_families = sorted(
        {int(value) for value in evidence.get("design_pose_family_ids", [])}
    )
    design_source_record_sha256s = sorted(
        str(value)
        for value in evidence.get("design_source_record_sha256s", [])
    )
    design_batch_sha = str(evidence.get("inputs", {}).get("design_batch_sha256", ""))
    if (
        not design_queries
        or not design_families
        or not used_design_queries.issubset(design_queries)
        or not used_design_families.issubset(design_families)
        or len(design_source_record_sha256s) != len(design_queries)
        or len(set(design_source_record_sha256s))
        != len(design_source_record_sha256s)
        or any(len(value) != 64 for value in design_source_record_sha256s)
        or len(design_batch_sha) != 64
    ):
        raise ValueError("V20 design split lineage is incomplete")
    anchor_ids = torch.as_tensor(baseline["anchor_ids"]).long()
    native_features = torch.as_tensor(baseline["anchor_features"])
    if native_features.shape != (anchor_ids.numel(), 256):
        raise ValueError("V20 requires one 256D descriptor per Anchor")

    updated_features, training = train_sparse_anchor_descriptors(
        anchor_features=native_features,
        evidence=evidence,
        mode=args.mode,
        maximum_angle_deg=float(args.maximum_angle_deg),
        steps=int(args.steps),
        batch_size=int(args.batch_size),
        learning_rate=float(args.learning_rate),
        temperature=float(args.temperature),
        clean_margin_slack=float(args.clean_margin_slack),
        clean_protection_weight=float(args.clean_protection_weight),
        angular_regularization_weight=float(
            args.angular_regularization_weight
        ),
        minimum_repair_margin_gain=float(args.minimum_repair_margin_gain),
        minimum_coordinate_ranking_gain=float(
            args.minimum_coordinate_ranking_gain
        ),
        maximum_selected_anchor_count=int(args.maximum_selected_anchor_count),
        seed=int(args.seed),
        device=args.device,
        strong_feedback_authorized=bool(evidence["strong_feedback_authorized"]),
    )
    arm = f"sparse_{args.mode}_angle_{float(args.maximum_angle_deg):g}"
    candidate = dict(baseline)
    candidate["anchor_features"] = updated_features
    materialized_audit = audit_materialized_sparse_action(
        baseline_anchor_features=baseline["anchor_features"],
        candidate_anchor_features=candidate["anchor_features"],
        selected_anchor_rows=training["selected_anchor_rows"],
        evidence=evidence,
        clean_margin_slack=float(args.clean_margin_slack),
        maximum_angle_deg=float(args.maximum_angle_deg),
        device=args.device,
    )
    training["materialized_action_audit"] = materialized_audit
    if not materialized_audit["passed"]:
        training["clean_protection_passed"] = False
        training["deployment_status"] = "REJECTED_MATERIALIZED_ACTION_AUDIT"
    candidate["v20_sparse_descriptor_action"] = {
        "schema": "lafgs_v20_sparse_descriptor_action",
        "version": 1,
        "arm": arm,
        "mode": args.mode,
        "query_descriptor_action": "native_unchanged",
        "selected_anchor_rows": training["selected_anchor_rows"],
        "maximum_angle_deg": float(args.maximum_angle_deg),
        "per_anchor_observed_angle_deg": training[
            "per_anchor_observed_angle_deg"
        ],
        "baseline_map": str(map_path),
        "baseline_map_sha256": map_sha,
        "evidence": str(evidence_path),
        "evidence_sha256": sha256_file(evidence_path),
        "strong_feedback_authorized": bool(
            evidence["strong_feedback_authorized"]
        ),
        "clean_protection_passed": bool(training["clean_protection_passed"]),
        "materialized_action_audit": materialized_audit,
        "positive_win_nonregression_passed": bool(
            training["positive_win_nonregression_passed"]
        ),
        "post_training_action_scale": float(
            training["post_training_action_scale"]
        ),
        "global_seed_action_scale": float(training["global_seed_action_scale"]),
        "per_anchor_action_scales": training["per_anchor_action_scales"],
        "maximum_applied_anchor_scale": float(
            training["maximum_applied_anchor_scale"]
        ),
        "clean_margin_slack": float(training["clean_margin_slack"]),
        "positive_objective": training["positive_objective"],
        "training_status": training["deployment_status"],
        "requires_exact_pose_control": True,
        "design_batch_sha256": design_batch_sha,
        "design_query_indices": design_queries,
        "design_pose_family_ids": design_families,
        "design_source_record_sha256s": design_source_record_sha256s,
    }
    args.output_dir.mkdir(parents=True)
    candidate_path = (args.output_dir / "candidate_anchor_map.pt").resolve()
    _save(candidate, candidate_path)
    candidate_sha = sha256_file(candidate_path)
    metric_path = (args.output_dir / "identity_metric.pt").resolve()
    _save(
        _identity_metric(
            anchor_ids=anchor_ids,
            map_path=candidate_path,
            map_sha256=candidate_sha,
            arm=arm,
            photometric_contract=candidate.get(
                "photometric_canonicalization_contract"
            ),
        ),
        metric_path,
    )
    report = {
        "schema": "lafgs_v20_sparse_anchor_descriptor_action_report",
        "version": 1,
        "status": training["deployment_status"],
        "uses_test_queries": False,
        "loo_used": False,
        "arm": arm,
        "training": _jsonable(training),
        "design_split": {
            "batch_sha256": design_batch_sha,
            "query_indices": design_queries,
            "pose_family_ids": design_families,
            "source_record_sha256s": design_source_record_sha256s,
        },
        "inputs": {
            "baseline_map": str(map_path),
            "baseline_map_sha256": map_sha,
            "evidence": str(evidence_path),
            "evidence_sha256": sha256_file(evidence_path),
        },
        "outputs": {
            "candidate_map": str(candidate_path),
            "candidate_map_sha256": candidate_sha,
            "identity_metric": str(metric_path),
            "identity_metric_sha256": sha256_file(metric_path),
        },
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
