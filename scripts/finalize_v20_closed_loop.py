#!/usr/bin/env python3
"""Fail-closed deployment gate for one frozen V20 sparse descriptor action."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from common.hashing import sha256_file
from map_learning.metric import validate_zero_identity_metric
from map_learning.v20_sparse_descriptor import audit_materialized_sparse_action


def finalize(args: argparse.Namespace) -> dict:
    if args.output.exists():
        raise FileExistsError(args.output)
    action = json.loads(args.action_report.read_text())
    control = json.loads(args.control_decision.read_text())
    confirmation = json.loads(args.confirmation_decision.read_text())
    if not (
        action.get("schema") == "lafgs_v20_sparse_anchor_descriptor_action_report"
        and action.get("uses_test_queries") is False
        and action.get("loo_used") is False
        and control.get("schema") == "lafgs_v20_sparse_map_decision"
        and control.get("phase") == "control"
        and control.get("uses_test_queries") is False
        and control.get("evaluation_role") == "feedback_query"
        and confirmation.get("schema") == "lafgs_v20_sparse_map_decision"
        and confirmation.get("phase") == "confirmation"
        and confirmation.get("uses_test_queries") is False
        and confirmation.get("evaluation_role") == "confirmation_query"
    ):
        raise ValueError("V20 finalization input contract differs")
    arm = str(action["arm"])
    candidate_map = Path(action["outputs"]["candidate_map"]).resolve()
    candidate_metric = Path(action["outputs"]["identity_metric"]).resolve()
    candidate_map_sha = sha256_file(candidate_map)
    candidate_metric_sha = sha256_file(candidate_metric)
    evidence_path = Path(action["inputs"]["evidence"]).resolve()
    evidence_sha = sha256_file(evidence_path)
    candidate_payload = torch.load(
        candidate_map, map_location="cpu", weights_only=False
    )
    metric_payload = torch.load(
        candidate_metric, map_location="cpu", weights_only=False
    )
    candidate_action = candidate_payload.get("v20_sparse_descriptor_action", {})
    candidate_selected = torch.as_tensor(
        candidate_action.get("selected_anchor_rows", [])
    ).long().reshape(-1).tolist()
    reported_selected = [
        int(value)
        for value in action.get("training", {}).get("selected_anchor_rows", [])
    ]
    candidate_scales = torch.as_tensor(
        candidate_action.get("per_anchor_action_scales", [])
    ).float().reshape(-1)
    reported_scales = torch.as_tensor(
        action.get("training", {}).get("per_anchor_action_scales", [])
    ).float().reshape(-1)
    candidate_angles = torch.as_tensor(
        candidate_action.get("per_anchor_observed_angle_deg", [])
    ).float().reshape(-1)
    reported_angles = torch.as_tensor(
        action.get("training", {}).get("per_anchor_observed_angle_deg", [])
    ).float().reshape(-1)
    stable_map = args.stable_map.resolve()
    stable_metric = args.stable_metric.resolve()
    baseline_map_sha = sha256_file(stable_map)
    stable_payload = torch.load(
        stable_map, map_location="cpu", weights_only=False
    )
    stable_metric_payload = torch.load(
        stable_metric, map_location="cpu", weights_only=False
    )
    candidate_ids = torch.as_tensor(candidate_payload.get("anchor_ids"))
    stable_ids = torch.as_tensor(stable_payload.get("anchor_ids"))
    candidate_xyz = torch.as_tensor(candidate_payload.get("anchor_xyz"))
    stable_xyz = torch.as_tensor(stable_payload.get("anchor_xyz"))
    candidate_features = torch.as_tensor(candidate_payload.get("anchor_features"))
    stable_features = torch.as_tensor(stable_payload.get("anchor_features"))
    if not (
        action["outputs"]["candidate_map_sha256"] == candidate_map_sha
        and action["outputs"]["identity_metric_sha256"] == candidate_metric_sha
        and action["inputs"]["evidence_sha256"] == evidence_sha
        and Path(candidate_action.get("evidence", "")).resolve() == evidence_path
        and candidate_action.get("evidence_sha256") == evidence_sha
        and control["input"]["candidate_map_sha256"] == candidate_map_sha
        and confirmation["input"]["candidate_map_sha256"] == candidate_map_sha
        and control.get("candidate_arm") == arm
        and confirmation.get("candidate_arm") == arm
        and candidate_payload.get("schema") == "lafgs_materialized_anchor_map"
        and stable_payload.get("schema") == "lafgs_materialized_anchor_map"
        and candidate_ids.dtype == torch.long
        and stable_ids.dtype == torch.long
        and candidate_ids.ndim == 1
        and torch.equal(candidate_ids, stable_ids)
        and candidate_xyz.shape == stable_xyz.shape
        and torch.equal(candidate_xyz, stable_xyz)
        and candidate_features.shape == stable_features.shape
        and candidate_features.ndim == 2
        and candidate_action.get("schema")
        == "lafgs_v20_sparse_descriptor_action"
        and candidate_action.get("arm") == arm
        and candidate_action.get("mode") == "positive_only"
        and candidate_action.get("training_status") == action.get("status")
        and candidate_action.get("query_descriptor_action") == "native_unchanged"
        and candidate_action.get("positive_objective")
        == action.get("training", {}).get("positive_objective")
        and candidate_action.get("strong_feedback_authorized")
        == action.get("training", {}).get("strong_feedback_authorized")
        and candidate_action.get("clean_protection_passed")
        == action.get("training", {}).get("clean_protection_passed")
        and candidate_action.get("materialized_action_audit")
        == action.get("training", {}).get("materialized_action_audit")
        and candidate_action.get("materialized_action_audit", {}).get("passed")
        is True
        and candidate_action.get("positive_win_nonregression_passed")
        is True
        and float(candidate_action.get("post_training_action_scale", 0.0))
        == float(
            action.get("training", {}).get("post_training_action_scale", -1.0)
        )
        and float(candidate_action.get("global_seed_action_scale", 0.0))
        == float(
            action.get("training", {}).get("global_seed_action_scale", -1.0)
        )
        and float(candidate_action.get("clean_margin_slack", -1.0))
        == float(action.get("training", {}).get("clean_margin_slack", -2.0))
        and candidate_selected == reported_selected
        and torch.equal(candidate_scales, reported_scales)
        and torch.equal(candidate_angles, reported_angles)
        and float(candidate_action.get("maximum_applied_anchor_scale", 0.0))
        == float(
            action.get("training", {}).get("maximum_applied_anchor_scale", -1.0)
        )
        and metric_payload.get("schema") == "lafgs_shared_metric_state"
        and metric_payload.get("protocol")
        == "v20_sparse_anchor_native_query_identity"
        and metric_payload.get("map_sha256") == candidate_map_sha
        and Path(metric_payload.get("map_path", "")).resolve() == candidate_map
        and metric_payload.get("deployment_arm") == arm
        and action.get("inputs", {}).get("baseline_map_sha256")
        == baseline_map_sha
        and candidate_action.get("baseline_map_sha256") == baseline_map_sha
        and control.get("input", {}).get("baseline_map_sha256")
        == baseline_map_sha
        and confirmation.get("input", {}).get("baseline_map_sha256")
        == baseline_map_sha
    ):
        raise ValueError("V20 decisions do not bind the frozen candidate artifacts")
    selected_tensor = torch.as_tensor(candidate_selected).long()
    exact_angles = torch.rad2deg(
        torch.acos(
            (
                F.normalize(stable_features[selected_tensor].float(), dim=1)
                * F.normalize(candidate_features[selected_tensor].float(), dim=1)
            )
            .sum(1)
            .clamp(-1.0, 1.0)
        )
    )
    if (
        candidate_angles.numel() != selected_tensor.numel()
        or not torch.allclose(
            candidate_angles, exact_angles, atol=2e-3, rtol=1e-4
        )
        or float(exact_angles.max())
        > float(candidate_action.get("maximum_angle_deg", 0.0)) + 0.05
        or float(candidate_action.get("maximum_angle_deg", -1.0))
        != float(action.get("training", {}).get("maximum_angle_deg", -2.0))
    ):
        raise ValueError("V20 candidate descriptor angles are not artifact-bound")
    evidence_payload = torch.load(
        evidence_path, map_location="cpu", weights_only=False
    )
    if not (
        evidence_payload.get("inputs", {}).get("anchor_map_sha256")
        == baseline_map_sha
        and evidence_payload.get("strong_feedback_authorized") is True
        and evidence_payload.get("inputs", {}).get("design_batch_sha256")
        == candidate_action.get("design_batch_sha256")
        and sorted(
            str(value)
            for value in evidence_payload.get(
                "design_source_record_sha256s", []
            )
        )
        == sorted(
            str(value)
            for value in candidate_action.get(
                "design_source_record_sha256s", []
            )
        )
    ):
        raise ValueError("V20 final evidence lineage differs")
    exact_materialized_audit = audit_materialized_sparse_action(
        baseline_anchor_features=stable_features,
        candidate_anchor_features=candidate_features,
        selected_anchor_rows=selected_tensor,
        evidence=evidence_payload,
        clean_margin_slack=float(candidate_action["clean_margin_slack"]),
        maximum_angle_deg=float(candidate_action["maximum_angle_deg"]),
        device="cpu",
    )
    if not exact_materialized_audit["passed"]:
        raise ValueError("V20 final candidate fails materialized clean audit")
    validate_zero_identity_metric(
        metric_payload,
        descriptor_dim=int(candidate_features.shape[1]),
        landmark_indices=candidate_ids,
        map_path=str(candidate_map),
        map_sha256=candidate_map_sha,
        allowed_protocols={"v20_sparse_anchor_native_query_identity"},
    )
    if not (
        metric_payload.get("loo_used") is False
        and metric_payload.get("feedback_descriptors_copied_into_map") is False
        and metric_payload.get("deployment_arm") == arm
        and metric_payload.get("photometric_canonicalization_contract")
        == candidate_payload.get("photometric_canonicalization_contract")
    ):
        raise ValueError("V20 candidate identity metric behavior differs")
    validate_zero_identity_metric(
        stable_metric_payload,
        descriptor_dim=int(stable_features.shape[1]),
        landmark_indices=stable_ids,
        map_path=str(stable_map),
        map_sha256=baseline_map_sha,
        allowed_protocols={
            "v6_identity_shared_metric",
            "rendered_track_map_bound_identity",
            "v20_sparse_anchor_native_query_identity",
        },
    )
    design = action.get("design_split", {})
    design_queries = [int(value) for value in design.get("query_indices", [])]
    candidate_design_queries = [
        int(value)
        for value in candidate_action.get("design_query_indices", [])
    ]
    control_queries = [
        int(value) for value in control.get("evaluation_query_indices", [])
    ]
    confirmation_queries = [
        int(value)
        for value in confirmation.get("evaluation_query_indices", [])
    ]
    design_sources = [
        str(value) for value in design.get("source_record_sha256s", [])
    ]
    candidate_design_sources = [
        str(value)
        for value in candidate_action.get("design_source_record_sha256s", [])
    ]
    control_sources = [
        str(value)
        for value in control.get("evaluation_source_record_sha256s", [])
    ]
    confirmation_sources = [
        str(value)
        for value in confirmation.get(
            "evaluation_source_record_sha256s", []
        )
    ]
    design_families = {int(value) for value in design.get("pose_family_ids", [])}
    control_families = {
        int(value) for value in control.get("evaluation_pose_family_ids", [])
    }
    confirmation_families = {
        int(value)
        for value in confirmation.get("evaluation_pose_family_ids", [])
    }
    design_batch_sha = str(design.get("batch_sha256", ""))
    control_batch_sha = str(
        control.get("input", {}).get("certified_batch_sha256", "")
    )
    confirmation_batch_sha = str(
        confirmation.get("input", {}).get("certified_batch_sha256", "")
    )
    if (
        candidate_action.get("design_batch_sha256") != design_batch_sha
        or candidate_design_queries != design_queries
        or candidate_design_sources != design_sources
        or not design_queries
        or not control_queries
        or not confirmation_queries
        or len(set(design_queries)) != len(design_queries)
        or len(set(control_queries)) != len(control_queries)
        or len(set(confirmation_queries)) != len(confirmation_queries)
        or not design_sources
        or not control_sources
        or not confirmation_sources
        or any(
            len(value) != 64
            for values in (design_sources, control_sources, confirmation_sources)
            for value in values
        )
        or len(set(design_sources)) != len(design_sources)
        or len(set(control_sources)) != len(control_sources)
        or len(set(confirmation_sources)) != len(confirmation_sources)
        or set(design_sources) & set(control_sources)
        or set(design_sources) & set(confirmation_sources)
        or set(control_sources) & set(confirmation_sources)
        or {
            int(value)
            for value in candidate_action.get("design_pose_family_ids", [])
        }
        != design_families
        or not design_families
        or not control_families
        or not confirmation_families
        or design_families & control_families
        or design_families & confirmation_families
        or control_families & confirmation_families
        or len({design_batch_sha, control_batch_sha, confirmation_batch_sha}) != 3
        or any(
            len(value) != 64
            for value in (
                design_batch_sha,
                control_batch_sha,
                confirmation_batch_sha,
            )
        )
    ):
        raise ValueError("V20 design/control/confirmation splits are not disjoint")
    decision = confirmation["decision_report"]
    authorized = bool(
        action["training"]["strong_feedback_authorized"]
        and action["training"].get("mode") == "positive_only"
        and action["training"].get("positive_objective")
        == "per_positive_listwise_mean"
        and action["training"].get("clean_protection_passed") is True
        and action["training"].get("materialized_action_audit", {}).get("passed")
        is True
        and action["training"].get("positive_win_nonregression_passed")
        is True
        and int(action["training"].get("protection_row_count", 0)) > 0
        and float(action["training"].get("post_training_action_scale", 0.0)) > 0.0
        and action.get("status") == "REQUIRES_EXACT_POSE_CONTROL"
        and control.get("strong_feedback_authorized") is True
        and control.get("analysis_only") is False
        and confirmation.get("strong_feedback_authorized") is True
        and confirmation.get("analysis_only") is False
        and control.get("selected_arm") == arm
        and control.get("decision") == "ADVANCE_TO_CONFIRMATION"
        and confirmation.get("selected_arm") == arm
        and decision.get("classification")
        in {"DEFAULT_CANDIDATE", "PARETO_CANDIDATE"}
        and decision.get("hard_safety", {}).get("passed") is True
        and float(decision.get("paired_effect", {}).get("net_gain", 0.0)) > 0.0
        and float(
            decision.get("bootstrap", {}).get(
                "probability_candidate_lower_risk", 0.0
            )
        )
        >= 0.95
    )
    output = {
        "schema": "lafgs_v20_closed_loop_deployment_decision",
        "version": 1,
        "uses_test_queries": False,
        "loo_used": False,
        "candidate_arm": arm,
        "formal_deployment_authorized": authorized,
        "decision": "DEPLOY_CANDIDATE" if authorized else "RETAIN_STABLE",
        "deployment": {
            "map": str(candidate_map if authorized else stable_map),
            "map_sha256": (
                candidate_map_sha if authorized else sha256_file(stable_map)
            ),
            "metric": str(candidate_metric if authorized else stable_metric),
            "metric_sha256": (
                candidate_metric_sha if authorized else sha256_file(stable_metric)
            ),
            "map_mutation": (
                "sparse_anchor_descriptors" if authorized else "none"
            ),
            "query_descriptor_action": "native_unchanged",
        },
        "inputs": {
            "action_report_sha256": sha256_file(args.action_report),
            "control_decision_sha256": sha256_file(args.control_decision),
            "confirmation_decision_sha256": sha256_file(
                args.confirmation_decision
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action-report", type=Path, required=True)
    parser.add_argument("--control-decision", type=Path, required=True)
    parser.add_argument("--confirmation-decision", type=Path, required=True)
    parser.add_argument("--stable-map", type=Path, required=True)
    parser.add_argument("--stable-metric", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = finalize(args)
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
