#!/usr/bin/env python3
"""Archive the confirmed-versus-Full V17 joint metric for analysis."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from common.hashing import sha256_file
from map_learning.metric import SharedLowRankMetric


def _save(payload: dict, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-map", type=Path, required=True)
    parser.add_argument("--active-set-map", type=Path, required=True)
    parser.add_argument("--source-metric", type=Path, required=True)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--control-decision", type=Path, required=True)
    parser.add_argument("--confirmation-decision", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if not 0.0 < args.alpha <= 1.0:
        parser.error("V17 alpha must be in (0, 1]")

    baseline_path = args.baseline_map.resolve()
    active_path = args.active_set_map.resolve()
    source_path = args.source_metric.resolve()
    baseline = torch.load(baseline_path, map_location="cpu", weights_only=False)
    active = torch.load(active_path, map_location="cpu", weights_only=False)
    baseline_ids = torch.as_tensor(baseline["anchor_ids"]).long()
    active_ids = torch.as_tensor(active["anchor_ids"]).long()
    if not set(active_ids.tolist()).issubset(set(baseline_ids.tolist())):
        raise ValueError("confirmed V17 active set is not delete-only")
    source = torch.load(source_path, map_location="cpu", weights_only=False)
    if not (
        source.get("protocol") == "v9_no_loo_causal_shared_metric"
        and source.get("training_protocol")
        == "v17_active_competitive_winner_runner_up"
        and source.get("loo_used") is False
        and torch.equal(torch.as_tensor(source["landmark_indices"]).long(), baseline_ids)
    ):
        raise ValueError("source metric is not the sealed V17 competition proposal")

    arm = "alpha_" + f"{args.alpha:g}".replace(".", "p") + "_active"
    control = json.loads(args.control_decision.read_text())
    confirmation = json.loads(args.confirmation_decision.read_text())
    if not (
        control.get("schema") == "lafgs_v17_minimum_intervention_gain_curve"
        and control.get("selected_arm") == arm
        and control.get("decision") == "ADVANCE_TO_CONFIRMATION"
        and confirmation.get("schema")
        == "lafgs_v17_repeated_confirmation_decision"
        and confirmation.get("selected_arm") == arm
        and confirmation.get("decision")
        in {"PARETO_CANDIDATE", "DEFAULT_CANDIDATE"}
    ):
        raise ValueError("V17 action was not control-selected and independently confirmed")

    config = dict(source["metric_config"])
    config["max_residual_norm"] = float(config["max_residual_norm"] * args.alpha)
    metric = SharedLowRankMetric(**config)
    state = {key: value.clone() for key, value in source["metric_state_dict"].items()}
    state["up.weight"] = state["up.weight"] * float(args.alpha)
    metric.load_state_dict(state, strict=True)
    artifact = {
        "schema": "lafgs_shared_metric_state",
        "version": 1,
        "protocol": "v17_competitive_minimum_intervention",
        "step": int(source["step"]),
        "metric_config": metric.export_config(),
        "metric_state_dict": metric.state_dict(),
        "landmark_indices": active_ids.clone(),
        "map_path": str(active_path),
        "map_sha256": sha256_file(active_path),
        "source_metric": str(source_path),
        "source_metric_sha256": sha256_file(source_path),
        "gain": float(args.alpha),
        "loo_used": False,
        "feedback_descriptors_copied_into_map": False,
        "deployment_status": "PARETO_ARCHIVE_NOT_DEFAULT",
    }
    args.output_dir.mkdir(parents=True)
    output_path = args.output_dir / "shared_metric.pt"
    _save(artifact, output_path)
    report = {
        "schema": "lafgs_v17_confirmed_action_artifact",
        "version": 1,
        "status": "PARETO_ARCHIVE_NOT_DEFAULT",
        "baseline_anchor_count": int(baseline_ids.numel()),
        "active_anchor_count": int(active_ids.numel()),
        "removed_anchor_count": int(baseline_ids.numel() - active_ids.numel()),
        "metric_gain": float(args.alpha),
        "maximum_residual_norm": float(config["max_residual_norm"]),
        "uses_test_queries": False,
        "loo_used": False,
        "control_decision": str(args.control_decision.resolve()),
        "control_decision_sha256": sha256_file(args.control_decision),
        "confirmation_decision": str(args.confirmation_decision.resolve()),
        "confirmation_decision_sha256": sha256_file(args.confirmation_decision),
        "output_metric": str(output_path.resolve()),
        "output_metric_sha256": sha256_file(output_path),
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
