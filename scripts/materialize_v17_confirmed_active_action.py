#!/usr/bin/env python3
"""Materialize the confirmed V16 active set with an identity metric."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from common.hashing import sha256_file
from map_learning.metric import SharedLowRankMetric


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--active-set-map", type=Path, required=True)
    parser.add_argument("--attribution-decision", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    decision = json.loads(args.attribution_decision.read_text())
    if not (
        decision.get("schema") == "lafgs_v17_repeated_confirmation_decision"
        and decision.get("selected_arm") == "active"
        and decision.get("decision")
        in {"PARETO_CANDIDATE", "DEFAULT_CANDIDATE"}
        and decision.get("uses_test_queries") is False
    ):
        raise ValueError("active-only map was not confirmed by repeated confirmation")
    map_path = args.active_set_map.resolve()
    state = torch.load(map_path, map_location="cpu", weights_only=False)
    anchor_ids = torch.as_tensor(state["anchor_ids"]).long()
    features = torch.as_tensor(state["anchor_features"])
    if features.ndim != 2 or features.shape[0] != anchor_ids.numel():
        raise ValueError("active map descriptors do not align with Anchor IDs")
    metric = SharedLowRankMetric(
        descriptor_dim=int(features.shape[1]), rank=1, max_residual_norm=0.0
    )
    with torch.no_grad():
        for parameter in metric.parameters():
            parameter.zero_()
    artifact = {
        "schema": "lafgs_shared_metric_state",
        "version": 1,
        "protocol": "v17_confirmed_competitive_active_identity",
        "step": 0,
        "metric_config": metric.export_config(),
        "metric_state_dict": metric.state_dict(),
        "landmark_indices": anchor_ids.clone(),
        "map_path": str(map_path),
        "map_sha256": sha256_file(map_path),
        "photometric_canonicalization_contract": None,
        "loo_used": False,
        "feedback_descriptors_copied_into_map": False,
        "descriptor_action": "rollback_to_identity_after_attribution",
        "deployment_status": "CONFIRMED_PARETO_CANDIDATE",
    }
    args.output_dir.mkdir(parents=True)
    output_path = args.output_dir / "identity_metric.pt"
    temporary = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
    try:
        torch.save(artifact, temporary)
        os.replace(temporary, output_path)
    finally:
        temporary.unlink(missing_ok=True)
    report = {
        "schema": "lafgs_v17_confirmed_active_action",
        "version": 1,
        "status": "CONFIRMED_PARETO_CANDIDATE",
        "uses_test_queries": False,
        "loo_used": False,
        "anchor_count": int(anchor_ids.numel()),
        "map": str(map_path),
        "map_sha256": sha256_file(map_path),
        "descriptor_action": "identity_no_update",
        "attribution_decision": str(args.attribution_decision.resolve()),
        "attribution_decision_sha256": sha256_file(args.attribution_decision),
        "metric": str(output_path.resolve()),
        "metric_sha256": sha256_file(output_path),
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
