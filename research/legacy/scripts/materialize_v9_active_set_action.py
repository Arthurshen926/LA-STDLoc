#!/usr/bin/env python3
"""Aggregate actual removal gains and materialize a delete-only V9 proposal."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import torch

from common.hashing import sha256_file
from map_learning.v8_feedback_controller import materialize_quarantined_map
from map_learning.v9_causal_feedback import aggregate_actual_removal_gain
from topology.v6_anchor_map import identity_metric_state


def _save(payload: dict, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-shards", type=Path, nargs="+", required=True)
    parser.add_argument("--map", type=Path, required=True)
    parser.add_argument("--maximum-removal-fraction", type=float, default=0.01)
    parser.add_argument("--minimum-median-actual-gain", type=float, default=0.001)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    map_path = args.map.resolve()
    map_sha = sha256_file(map_path)
    payloads = [json.loads(path.read_text()) for path in args.replay_shards]
    if any(
        payload.get("schema") != "lafgs_v9_actual_removal_replay_shard"
        or payload.get("loo_used") is not False
        or payload.get("uses_test_queries") is not False
        or payload.get("map_sha256") != map_sha
        for payload in payloads
    ):
        raise ValueError("invalid V9 actual-removal replay shard")
    shard_count = int(payloads[0]["shard_count"])
    if sorted(int(payload["shard_index"]) for payload in payloads) != list(
        range(shard_count)
    ):
        raise ValueError("actual-removal shard registry is incomplete")
    records = [record for payload in payloads for record in payload["records"]]
    audit = aggregate_actual_removal_gain(
        records, minimum_median_gain=args.minimum_median_actual_gain
    )
    state = torch.load(map_path, map_location="cpu", weights_only=False)
    maximum = math.floor(
        len(torch.as_tensor(state["anchor_ids"])) * args.maximum_removal_fraction
    )
    authorized = torch.sort(audit["authorized_anchor_rows"][:maximum]).values
    candidate, _ = materialize_quarantined_map(state, authorized)
    candidate["provenance"] = {
        **dict(candidate.get("provenance", {})),
        "v9_no_loo_actual_removal_gain_action": True,
        "loo_used": False,
        "uses_test_queries": False,
        "feedback_descriptors_copied_into_map": False,
    }
    args.output_dir.mkdir(parents=True)
    audit_path = args.output_dir / "actual_removal_audit.pt"
    _save(audit, audit_path)
    output_map = args.output_dir / "projective_anchor_map.pt"
    _save(candidate, output_map)
    output_metric = args.output_dir / "identity_metric.pt"
    _save(
        identity_metric_state(
            candidate,
            map_path=str(output_map.resolve()),
            map_sha256=sha256_file(output_map),
        ),
        output_metric,
    )
    report = {
        "schema": "lafgs_v9_delete_only_active_set_action",
        "version": 1,
        "status": "PROPOSAL",
        "loo_used": False,
        "uses_test_queries": False,
        "actual_removal_pose_gain_required": True,
        "minimum_median_actual_task_gain": args.minimum_median_actual_gain,
        "positive_cumulative_actual_task_gain_required": True,
        "source_anchor_count": int(torch.as_tensor(state["anchor_ids"]).numel()),
        "removed_anchor_count": int(authorized.numel()),
        "removed_fraction": float(authorized.numel() / len(state["anchor_ids"])),
        "geometry_mutation_count": 0,
        "anchor_addition_count": 0,
        "output": {
            "audit": str(audit_path.resolve()),
            "audit_sha256": sha256_file(audit_path),
            "map": str(output_map.resolve()),
            "map_sha256": sha256_file(output_map),
            "metric": str(output_metric.resolve()),
            "metric_sha256": sha256_file(output_metric),
        },
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
