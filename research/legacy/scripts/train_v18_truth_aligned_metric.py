#!/usr/bin/env python3
"""Train the shared low-rank metric from V18 operation responsibility."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from common.hashing import sha256_file
from map_learning.v18_competitive_metric import (
    build_truth_aligned_metric_evidence,
)
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
    parser.add_argument("--responsibility", type=Path, required=True)
    parser.add_argument("--baseline-map", type=Path, required=True)
    parser.add_argument("--active-set-map", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--minimum-repair-pose-families", type=int, default=2)
    parser.add_argument("--maximum-repair-rows-per-query", type=int, default=256)
    parser.add_argument("--maximum-protection-rows-per-query", type=int, default=256)
    parser.add_argument("--minimum-replace-task-gain", type=float, default=0.01)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--maximum-residual-norm", type=float, default=0.05)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--clean-protection-weight", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=1820260829)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)

    responsibility = torch.load(
        args.responsibility, map_location="cpu", weights_only=False
    )
    baseline_path = args.baseline_map.resolve()
    active_path = args.active_set_map.resolve()
    baseline = torch.load(baseline_path, map_location="cpu", weights_only=False)
    active_map = torch.load(active_path, map_location="cpu", weights_only=False)
    if not (
        responsibility.get("schema") == "lafgs_v18_operation_responsibility_batch"
        and responsibility.get("uses_test_queries") is False
        and responsibility.get("loo_used") is False
        and responsibility["inputs"]["baseline_map_sha256"]
        == sha256_file(baseline_path)
    ):
        raise ValueError("V18 metric responsibility contract differs")
    baseline_ids = torch.as_tensor(baseline["anchor_ids"]).long()
    active_ids = torch.as_tensor(active_map["anchor_ids"]).long()
    if not bool(torch.isin(active_ids, baseline_ids).all()):
        raise ValueError("V18 metric active map is not a delete-only baseline subset")
    active_mask = torch.isin(baseline_ids, active_ids)

    descriptors = {}
    keypoints = {}
    image_hw = {}
    feedback = torch.load(
        responsibility["inputs"]["feedback_truth"],
        map_location="cpu",
        weights_only=False,
    )
    truth_sources = {
        int(record["query_index"]): record for record in feedback["records"]
    }
    fallback_sources = {}
    if any("source_record" not in record for record in feedback["records"]):
        design_path = Path(feedback["inputs"]["design_batch"]).resolve()
        if sha256_file(design_path) != feedback["inputs"]["design_batch_sha256"]:
            raise ValueError("V18 metric fallback design SHA256 differs")
        design = json.loads(design_path.read_text())
        for item in design["records"]:
            observed = torch.load(item["path"], map_location="cpu", weights_only=False)
            if observed["certificate_decision"] == "ACCEPT":
                fallback_sources[int(observed["query_index"])] = observed
    for record in responsibility["records"]:
        query = int(record["query_index"])
        source_rows = torch.as_tensor(record["source_query_rows"]).long()
        # The source path is retained in the provenance-truth input rather than
        # copied into this observer artifact; recover it from the frozen design
        # observer record registry stored beside the responsibility records.
        matching = truth_sources.get(query)
        if matching is None:
            raise ValueError("V18 metric cannot recover a feedback source record")
        if "source_record" not in matching:
            matching = fallback_sources.get(query)
        if matching is None or "source_record" not in matching:
            raise ValueError("V18 metric fallback cannot recover a source record")
        source_path = Path(matching["source_record"]).resolve()
        if sha256_file(source_path) != matching["source_record_sha256"]:
            raise ValueError("V18 metric feedback source SHA256 differs")
        source = torch.load(source_path, map_location="cpu", weights_only=False)
        descriptors[query] = torch.as_tensor(source["descriptors"])[source_rows].float()
        keypoints[query] = torch.as_tensor(source["keypoints"])[source_rows].float() + 0.5
        image_hw[query] = torch.as_tensor(source["image_hw"]).long()
    evidence = build_truth_aligned_metric_evidence(
        responsibility_records=responsibility["records"],
        query_descriptors=descriptors,
        query_keypoints=keypoints,
        query_image_hw=image_hw,
        active_anchor_mask=active_mask,
        minimum_repair_pose_families=int(args.minimum_repair_pose_families),
        maximum_repair_rows_per_query=int(args.maximum_repair_rows_per_query),
        maximum_protection_rows_per_query=int(args.maximum_protection_rows_per_query),
        minimum_replace_task_gain=float(args.minimum_replace_task_gain),
    )
    metric, training = train_v9_shared_metric(
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
        rank=int(args.rank),
        maximum_residual_norm=float(args.maximum_residual_norm),
        steps=int(args.steps),
        learning_rate=float(args.learning_rate),
        clean_protection_weight=float(args.clean_protection_weight),
        seed=int(args.seed),
        device=args.device,
    )
    training = {
        **training,
        "schema": "lafgs_v18_truth_aligned_metric_training",
        "provenance_truth_used": responsibility.get("truth_source")
        in {None, "gaussian_provenance"},
        "truth_source": responsibility.get("truth_source", "gaussian_provenance"),
        "counterfactual_gain_weighted": True,
        "pose_family_balanced": True,
    }
    args.output_dir.mkdir(parents=True)
    evidence_path = args.output_dir / "metric_evidence.pt"
    _save(evidence, evidence_path)
    metric_state = metric_artifact(
        metric,
        anchor_ids=baseline_ids,
        map_path=str(baseline_path),
        map_sha256=sha256_file(baseline_path),
        training_report=training,
    )
    metric_state.update(
        {
            "training_protocol": "v18_truth_counterfactual_family_balanced",
            "active_set_map": str(active_path),
            "active_set_map_sha256": sha256_file(active_path),
        }
    )
    metric_path = args.output_dir / "shared_metric.pt"
    _save(metric_state, metric_path)
    report = {
        "schema": "lafgs_v18_truth_aligned_metric_action_report",
        "version": 1,
        "status": "PROPOSAL_REQUIRES_GAIN_CURVE_CONFIRMATION",
        "uses_test_queries": False,
        "loo_used": False,
        "feedback_descriptors_copied_into_map": False,
        "repair_row_count": int(evidence["repair_positive_anchor_rows"].numel()),
        "protection_row_count": int(
            evidence["protection_positive_anchor_rows"].numel()
        ),
        "repair_pose_family_count": int(evidence["repair_pose_family_count"]),
        "protection_pose_family_count": int(
            evidence["protection_pose_family_count"]
        ),
        "training": training,
        "outputs": {
            "metric": str(metric_path.resolve()),
            "metric_sha256": sha256_file(metric_path),
            "evidence": str(evidence_path.resolve()),
            "evidence_sha256": sha256_file(evidence_path),
        },
        "inputs": {
            "responsibility": str(args.responsibility.resolve()),
            "responsibility_sha256": sha256_file(args.responsibility),
            "baseline_map": str(baseline_path),
            "baseline_map_sha256": sha256_file(baseline_path),
            "active_set_map": str(active_path),
            "active_set_map_sha256": sha256_file(active_path),
        },
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
