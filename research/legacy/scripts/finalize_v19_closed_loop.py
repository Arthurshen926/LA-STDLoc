#!/usr/bin/env python3
"""Freeze the V19 observer/controller decision without implicit fallbacks."""

from __future__ import annotations

import argparse
import json
from argparse import Namespace
from pathlib import Path

import torch

from common.hashing import sha256_file


def finalize_closed_loop(args: Namespace) -> dict:
    """Resolve the frozen V19 candidate without an implicit identity fallback.

    A confirmed metric is deployable only when the exact scaled metric artifact
    is supplied and declares the control-selected arm.  Rejected runs retain the
    stable identity metric and do not require a candidate artifact.
    """

    if args.output.exists():
        raise FileExistsError(args.output)
    candidate_metric = getattr(args, "candidate_metric", None)
    candidate_metric_path = (
        None if candidate_metric is None else Path(candidate_metric).resolve()
    )

    teacher = torch.load(
        args.teacher_validation, map_location="cpu", weights_only=False
    )
    full_audit = torch.load(
        args.full_pool_audit, map_location="cpu", weights_only=False
    )
    compressed_audit = torch.load(
        args.compressed_pool_audit, map_location="cpu", weights_only=False
    )
    control = json.loads(args.metric_control.read_text())
    confirmation = json.loads(args.metric_confirmation.read_text())
    if not (
        teacher.get("schema") == "lafgs_v19_track_extension_teacher_validation"
        and teacher.get("selection_uses_validation") is False
        and full_audit.get("schema") == "lafgs_v19_full_pool_sufficiency_audit"
        and compressed_audit.get("schema")
        == "lafgs_v19_full_pool_sufficiency_audit"
        and full_audit.get("uses_test_queries") is False
        and compressed_audit.get("uses_test_queries") is False
        and control.get("phase") == "control"
        and confirmation.get("phase") == "confirmation"
        and control.get("uses_test_queries") is False
        and confirmation.get("uses_test_queries") is False
    ):
        raise ValueError("V19 finalization input contract differs")
    selected_arm = control.get("selected_arm")
    confirmation_decision = confirmation.get("decisions", {}).get(selected_arm, {})
    paired = confirmation_decision.get("paired_effect", {})
    bootstrap = confirmation_decision.get("bootstrap", {})
    metric_authorized = bool(
        selected_arm is not None
        and confirmation.get("selected_arm") == selected_arm
        and confirmation_decision.get("classification")
        in {"DEFAULT_CANDIDATE", "PARETO_CANDIDATE"}
        and confirmation_decision.get("hard_safety", {}).get("passed") is True
        and float(paired.get("net_gain", 0.0)) > 0.0
        and float(bootstrap.get("probability_candidate_lower_risk", 0.0)) >= 0.95
    )
    candidate_metric_sha = None
    if metric_authorized:
        if candidate_metric_path is None:
            raise ValueError(
                "confirmed V19 metric requires the exact candidate metric artifact"
            )
        candidate = torch.load(
            candidate_metric_path, map_location="cpu", weights_only=False
        )
        if not (
            candidate.get("schema") == "lafgs_shared_metric_state"
            and candidate.get("loo_used") is False
            and candidate.get("deployment_arm") == selected_arm
        ):
            raise ValueError("V19 candidate metric does not bind the selected arm")
        candidate_metric_sha = sha256_file(candidate_metric_path)

    tier_a = teacher["selected_tiers"]["tier_a"]
    tier_b = teacher["selected_tiers"]["tier_b"]
    anchor_addition_authorized = bool(
        tier_a["authorized_actions"]
        and full_audit["candidate_pool_deficit_authorized"]
        and int(full_audit["totals"]["certified_candidate_pool_deficit"]) > 0
    )
    compressed_selection_loss = int(
        compressed_audit["totals"]["active_map_selection_loss_query_count"]
    )
    deployment = {
        "stable_map": str(args.stable_map.resolve()),
        "stable_map_sha256": sha256_file(args.stable_map),
        "metric": (
            str(candidate_metric_path) if metric_authorized else "identity"
        ),
        "metric_sha256": candidate_metric_sha,
        "map_mutation": "none",
        "reason": (
            "confirmed_metric_deployed"
            if metric_authorized
            else "fail_closed_teacher_and_confirmation_gates_preserve_full_v2_m0"
        ),
    }
    output = {
        "schema": "lafgs_v19_closed_loop_deployment_decision",
        "version": 2,
        "uses_test_queries": False,
        "loo_used": False,
        "observer": {
            "destructive_truth_authorized": bool(tier_a["authorized_actions"]),
            "strong_metric_truth_authorized": bool(tier_b["authorized_actions"]),
            "soft_diagnostic_authorized": bool(
                teacher["selected_tiers"]["tier_c"]["authorized_actions"]
            ),
        },
        "controller": {
            "compressed_map_authorized": compressed_selection_loss == 0,
            "compressed_map_selection_loss_query_count": compressed_selection_loss,
            "anchor_addition_authorized": anchor_addition_authorized,
            "mapping_identity_metric_authorized": metric_authorized,
            "selected_metric_arm": selected_arm,
        },
        "deployment": deployment,
        "inputs": {
            "teacher_validation_sha256": sha256_file(args.teacher_validation),
            "full_pool_audit_sha256": sha256_file(args.full_pool_audit),
            "compressed_pool_audit_sha256": sha256_file(args.compressed_pool_audit),
            "metric_control_sha256": sha256_file(args.metric_control),
            "metric_confirmation_sha256": sha256_file(args.metric_confirmation),
            "candidate_metric_sha256": candidate_metric_sha,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-validation", type=Path, required=True)
    parser.add_argument("--full-pool-audit", type=Path, required=True)
    parser.add_argument("--compressed-pool-audit", type=Path, required=True)
    parser.add_argument("--metric-control", type=Path, required=True)
    parser.add_argument("--metric-confirmation", type=Path, required=True)
    parser.add_argument("--stable-map", type=Path, required=True)
    parser.add_argument(
        "--candidate-metric",
        type=Path,
        help="exact scaled metric artifact; required only for confirmed deployment",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = finalize_closed_loop(args)
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
