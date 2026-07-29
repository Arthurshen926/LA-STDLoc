#!/usr/bin/env python3
"""Audit whether a family update fixes the confusion edges it trained on."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def _dynamic_lookup(dynamic: dict) -> dict[tuple[str, int], dict]:
    lookup = {}
    for name, record in zip(dynamic["query_names"], dynamic["records"]):
        rows = torch.as_tensor(record["query_rows"]).long()
        anchors = torch.as_tensor(record["top1_anchor_indices"]).long()
        errors = torch.as_tensor(record["gt_reprojection_errors_px"]).float()
        inliers = torch.as_tensor(record["ransac_inlier_mask"]).bool()
        for index, row in enumerate(rows.tolist()):
            lookup[(str(name), int(row))] = {
                "anchor": int(anchors[index]),
                "error": float(errors[index]),
                "inlier": bool(inliers[index]),
            }
    return lookup


def _query_pose(dynamic: dict) -> dict[str, dict]:
    return {
        str(name): {
            "te_cm": float(record["te_cm"]),
            "re_deg": float(record["re_deg"]),
            "hypotheses": int(record.get("hypotheses", 0)),
        }
        for name, record in zip(dynamic["query_names"], dynamic["records"])
    }


def _pose_summary(names: set[str], lookup: dict[str, dict]) -> dict:
    values = np.asarray([lookup[name]["te_cm"] for name in names])
    hypotheses = np.asarray([lookup[name]["hypotheses"] for name in names])
    return {
        "query_count": len(names),
        "median_te_cm": float(np.median(values)),
        "mean_te_cm": float(np.mean(values)),
        "p90_te_cm": float(np.percentile(values, 90)),
        "recall_5cm_percent": float(100.0 * np.mean(values <= 5.0)),
        "mean_hypotheses": float(np.mean(hypotheses)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confusion-graph", required=True)
    parser.add_argument("--contrastive-evidence", required=True)
    parser.add_argument("--baseline-dynamic", required=True)
    parser.add_argument("--candidate-dynamic", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    graph = torch.load(
        args.confusion_graph, map_location="cpu", weights_only=False
    )
    evidence = torch.load(
        args.contrastive_evidence, map_location="cpu", weights_only=False
    )
    baseline = torch.load(
        args.baseline_dynamic, map_location="cpu", weights_only=False
    )
    candidate = torch.load(
        args.candidate_dynamic, map_location="cpu", weights_only=False
    )
    targeted_pairs = {
        (int(positive), int(negative))
        for record in evidence["records"]
        for positive, negative in zip(
            torch.as_tensor(
                record["hard_negative_positive_indices"]
            ).tolist(),
            torch.as_tensor(record["hard_negative_indices"]).tolist(),
        )
    }
    events = [
        event
        for event in graph["events"]
        if (
            int(event["correct_anchor"]),
            int(event["confusing_anchor"]),
        )
        in targeted_pairs
    ]
    baseline_lookup = _dynamic_lookup(baseline)
    candidate_lookup = _dynamic_lookup(candidate)
    counts = {
        "correct_family": 0,
        "original_confusing_family": 0,
        "other_family": 0,
        "missing": 0,
    }
    baseline_clean = 0
    candidate_clean = 0
    baseline_inlier_clean = 0
    candidate_inlier_clean = 0
    changed = 0
    for event in events:
        key = (str(event["query_name"]), int(event["query_row"]))
        before = baseline_lookup.get(key)
        after = candidate_lookup.get(key)
        if before is None or after is None:
            counts["missing"] += 1
            continue
        changed += int(before["anchor"] != after["anchor"])
        if after["anchor"] == int(event["correct_anchor"]):
            counts["correct_family"] += 1
        elif after["anchor"] == int(event["confusing_anchor"]):
            counts["original_confusing_family"] += 1
        else:
            counts["other_family"] += 1
        baseline_clean += int(before["error"] <= 2.0)
        candidate_clean += int(after["error"] <= 2.0)
        baseline_inlier_clean += int(before["inlier"] and before["error"] <= 2.0)
        candidate_inlier_clean += int(after["inlier"] and after["error"] <= 2.0)
    valid_count = len(events) - counts["missing"]
    targeted_queries = {str(event["query_name"]) for event in events}
    baseline_pose = _query_pose(baseline)
    candidate_pose = _query_pose(candidate)
    output = {
        "schema": "lafgs_confusion_delta_audit",
        "version": 1,
        "targeted_confusion_pair_count": len(targeted_pairs),
        "targeted_event_count": len(events),
        "targeted_query_count": len(targeted_queries),
        "assignment_counts": counts,
        "assignment_change_rate": changed / max(valid_count, 1),
        "correct_family_switch_rate": counts["correct_family"]
        / max(valid_count, 1),
        "original_confusion_retention_rate": counts[
            "original_confusing_family"
        ]
        / max(valid_count, 1),
        "event_clean_2px_rate_before": baseline_clean
        / max(valid_count, 1),
        "event_clean_2px_rate_after": candidate_clean
        / max(valid_count, 1),
        "event_inlier_clean_2px_rate_before": baseline_inlier_clean
        / max(valid_count, 1),
        "event_inlier_clean_2px_rate_after": candidate_inlier_clean
        / max(valid_count, 1),
        "targeted_pose_before": _pose_summary(
            targeted_queries, baseline_pose
        ),
        "targeted_pose_after": _pose_summary(
            targeted_queries, candidate_pose
        ),
        "provenance": {
            key: str(Path(value).resolve())
            for key, value in vars(args).items()
            if key != "output"
        },
    }
    path = Path(args.output).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
