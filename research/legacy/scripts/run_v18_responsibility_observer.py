#!/usr/bin/env python3
"""Materialize operation-level responsibility on provenance-truth feedback."""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import torch

from common.hashing import sha256_file
from map_learning.v18_responsibility_observer import (
    decompose_correspondence_responsibility,
)


def _atomic_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design-batch", type=Path, required=True)
    parser.add_argument("--feedback-truth", type=Path, required=True)
    parser.add_argument(
        "--allow-certified-topl-fallback",
        action="store_true",
        help="explicitly audit legacy V16 Top-L geometric truth",
    )
    parser.add_argument("--baseline-map", type=Path, required=True)
    parser.add_argument("--active-set-map", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--maximum-individual-rows", type=int, default=32)
    parser.add_argument("--minimum-task-gain", type=float, default=0.01)
    parser.add_argument("--minimum-anchor-pose-families", type=int, default=2)
    parser.add_argument("--progress-interval", type=int, default=10)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if int(args.minimum_anchor_pose_families) < 2:
        parser.error("map-side responsibility requires at least two pose families")

    design = json.loads(args.design_batch.read_text())
    feedback = torch.load(args.feedback_truth, map_location="cpu", weights_only=False)
    baseline = torch.load(args.baseline_map, map_location="cpu", weights_only=False)
    if not (
        design.get("schema") == "lafgs_v9_no_loo_causal_feedback_batch"
        and design.get("role") == "controller_design"
        and design.get("uses_test_queries") is False
        and design.get("loo_used") is False
        and (
            feedback.get("schema") == "lafgs_v18_feedback_provenance_truth_batch"
            or (
                args.allow_certified_topl_fallback
                and feedback.get("schema")
                == "lafgs_v18_feedback_certified_projection_truth_batch"
                and feedback.get("explicit_fallback_required") is True
                and feedback.get("descriptor_independent_full_map_truth") is False
            )
        )
        and feedback.get("uses_test_queries") is False
        and feedback.get("loo_used") is False
    ):
        raise ValueError("V18 responsibility inputs violate the no-test design contract")
    baseline_ids = torch.as_tensor(baseline["anchor_ids"]).long()
    active_mask = torch.ones(baseline_ids.numel(), dtype=torch.bool)
    active_path = None
    if args.active_set_map is not None:
        active_path = args.active_set_map.resolve()
        active_map = torch.load(active_path, map_location="cpu", weights_only=False)
        active_ids = torch.as_tensor(active_map["anchor_ids"]).long()
        if not bool(torch.isin(active_ids, baseline_ids).all()):
            raise ValueError("V18 responsibility active map is not a baseline subset")
        active_mask = torch.isin(baseline_ids, active_ids)
    truth_by_query = {
        int(record["query_index"]): record for record in feedback["records"]
    }
    records = []
    anchor_families: dict[int, set[int]] = defaultdict(set)
    totals = defaultdict(int)
    accepted = []
    for item in design["records"]:
        observed = torch.load(item["path"], map_location="cpu", weights_only=False)
        if observed["certificate_decision"] == "ACCEPT":
            accepted.append((item, observed))
    for completed, (_item, observed) in enumerate(accepted, start=1):
        query_index = int(observed["query_index"])
        truth_record = truth_by_query.get(query_index)
        if truth_record is None:
            raise ValueError("V18 truth does not cover every ACCEPT design query")
        source_path = Path(observed["source_record"]).resolve()
        if sha256_file(source_path) != observed["source_record_sha256"]:
            raise ValueError("V18 responsibility source SHA256 differs")
        source = torch.load(source_path, map_location="cpu", weights_only=False)
        source_rows = torch.as_tensor(observed["source_query_rows"]).long()
        candidates = torch.as_tensor(observed["topk_anchor_rows"]).long()
        scores = torch.as_tensor(observed["topk_scores"]).float()
        available = active_mask[candidates]
        # Deleted Anchors remain in the registry but move behind every active
        # candidate.  This exactly reproduces delete-only global ranking as
        # long as the frozen Top-L contains an active candidate.
        order = torch.argsort(
            scores.masked_fill(~available, -torch.inf),
            dim=1,
            descending=True,
            stable=True,
        )
        candidates = candidates.gather(1, order)
        scores = scores.gather(1, order)
        result = decompose_correspondence_responsibility(
            keypoints=torch.as_tensor(source["keypoints"]).float()[source_rows] + 0.5,
            candidate_anchor_rows=candidates,
            candidate_scores=scores,
            truth=truth_record["truth"],
            anchor_xyz=baseline["anchor_xyz"],
            intrinsic=source["intrinsics"],
            pose_w2c=source["pose_w2c"],
            maximum_individual_rows=int(args.maximum_individual_rows),
            minimum_task_gain=float(args.minimum_task_gain),
        )
        family = int(observed["pose_family_id"])
        for anchor in torch.as_tensor(
            result["anchor_suppressible_anchor_rows"]
        ).tolist():
            anchor_families[int(anchor)].add(family)
        totals["wrong_decisive_row_count"] += int(result["wrong_decisive_row_count"])
        totals["audited_wrong_row_count"] += int(result["audited_wrong_row_count"])
        totals["coverage_limited_row_count"] += int(result["coverage_limited_row_count"])
        totals["row_suppressible_count"] += int(
            torch.as_tensor(result["row_suppressible_query_rows"]).numel()
        )
        totals["metric_controllable_count"] += int(
            torch.as_tensor(result["metric_controllable_query_rows"]).numel()
        )
        totals["geometry_limited_query_count"] += int(result["geometry_limited"])
        records.append(
            {
                "query_index": query_index,
                "pose_family_id": family,
                "source_query_rows": source_rows,
                "candidate_anchor_rows": candidates,
                "candidate_scores": scores,
                "truth": truth_record["truth"],
                "responsibility": result,
            }
        )
        if completed % max(int(args.progress_interval), 1) == 0 or completed == len(accepted):
            print(
                json.dumps(
                    {
                        "completed_queries": completed,
                        "accepted_queries": len(accepted),
                        **totals,
                    }
                ),
                flush=True,
            )
    cross_family = sorted(
        anchor
        for anchor, families in anchor_families.items()
        if len(families) >= int(args.minimum_anchor_pose_families)
    )
    artifact = {
        "schema": "lafgs_v18_operation_responsibility_batch",
        "version": 1,
        "uses_test_queries": False,
        "loo_used": False,
        "truth_source": feedback.get("truth_source", "gaussian_provenance"),
        "certified_topl_fallback": bool(
            feedback.get("schema")
            == "lafgs_v18_feedback_certified_projection_truth_batch"
        ),
        "active_set_conditioned": args.active_set_map is not None,
        "accepted_query_count": len(records),
        "totals": dict(totals),
        "cross_family_anchor_suppressible_rows": torch.tensor(
            cross_family, dtype=torch.long
        ),
        "cross_family_anchor_suppressible_count": len(cross_family),
        "minimum_anchor_pose_families": int(args.minimum_anchor_pose_families),
        "records": records,
        "inputs": {
            "design_batch": str(args.design_batch.resolve()),
            "design_batch_sha256": sha256_file(args.design_batch),
            "feedback_truth": str(args.feedback_truth.resolve()),
            "feedback_truth_sha256": sha256_file(args.feedback_truth),
            "baseline_map": str(args.baseline_map.resolve()),
            "baseline_map_sha256": sha256_file(args.baseline_map),
            "active_set_map": None if active_path is None else str(active_path),
            "active_set_map_sha256": (
                None if active_path is None else sha256_file(active_path)
            ),
        },
    }
    _atomic_save(artifact, args.output.resolve())
    report = {key: value for key, value in artifact.items() if key != "records"}
    report["cross_family_anchor_suppressible_rows"] = cross_family
    report["output"] = str(args.output.resolve())
    report["output_sha256"] = sha256_file(args.output)
    args.output.with_suffix(".json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
