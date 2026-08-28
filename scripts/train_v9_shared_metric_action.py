#!/usr/bin/env python3
"""Train and materialize the bounded V9 shared-metric ranking proposal."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path

import torch

from common.hashing import sha256_file
from map_learning.v9_metric_controller import (
    metric_artifact,
    train_v9_shared_metric,
    transform_map_anchor_features,
)


def _save(payload: dict, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feedback-batches", type=Path, nargs="+", required=True)
    parser.add_argument("--map", type=Path, required=True)
    parser.add_argument("--expected-map-sha256", required=True)
    parser.add_argument("--minimum-pose-families", type=int, default=2)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--maximum-residual-norm", type=float, default=0.05)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    map_path = args.map.resolve()
    map_sha = sha256_file(map_path)
    if map_sha != args.expected_map_sha256:
        raise ValueError("fixed V2 map SHA256 differs")
    state = torch.load(map_path, map_location="cpu", weights_only=False)

    batch_inputs = []
    records = []
    seen_queries = set()
    for batch_path_input in args.feedback_batches:
        batch_path = batch_path_input.resolve()
        batch = json.loads(batch_path.read_text())
        if not (
            batch.get("schema") == "lafgs_v9_no_loo_causal_feedback_batch"
            and batch.get("loo_used") is False
            and batch.get("uses_test_queries") is False
            and batch.get("map_mutation_count") == 0
            and batch.get("input", {}).get("map_sha256") == map_sha
        ):
            raise ValueError("metric input violates immutable no-LOO feedback contract")
        batch_inputs.append({"path": str(batch_path), "sha256": sha256_file(batch_path)})
        for item in batch["records"]:
            query_key = int(item["query_index"])
            if query_key in seen_queries:
                raise ValueError("causal observer shards overlap")
            seen_queries.add(query_key)
            path = Path(item["path"]).resolve()
            if sha256_file(path) != item["sha256"]:
                raise ValueError("causal feedback record SHA256 differs")
            record = torch.load(path, map_location="cpu", weights_only=False)
            if record.get("loo_used") is not False:
                raise ValueError("LOO evidence is forbidden in the metric action")
            records.append(record)

    negative_families: dict[int, set[int]] = defaultdict(set)
    for record in records:
        if not record["can_train_metric"]:
            continue
        family = int(record["pose_family_id"])
        for row in torch.as_tensor(
            record["training_evidence"]["negative_anchor_rows"]
        ).tolist():
            negative_families[int(row)].add(family)
    authorized_negatives = {
        row
        for row, families in negative_families.items()
        if len(families) >= int(args.minimum_pose_families)
    }
    query_parts = []
    positive_parts = []
    negative_parts = []
    weight_parts = []
    clean_query_parts = []
    clean_positive_parts = []
    clean_negative_parts = []
    clean_margin_parts = []
    contributing_families = set()
    for record in records:
        clean = record["clean_protection_evidence"]
        if torch.as_tensor(clean["query_descriptors"]).numel():
            clean_query_parts.append(torch.as_tensor(clean["query_descriptors"]).float())
            clean_positive_parts.append(torch.as_tensor(clean["positive_anchor_rows"]).long())
            clean_negative_parts.append(torch.as_tensor(clean["negative_anchor_rows"]).long())
            clean_margin_parts.append(torch.as_tensor(clean["initial_margin"]).float())
        if not record["can_train_metric"]:
            continue
        evidence = record["training_evidence"]
        negatives = torch.as_tensor(evidence["negative_anchor_rows"]).long()
        keep = torch.tensor(
            [int(row) in authorized_negatives for row in negatives.tolist()],
            dtype=torch.bool,
        )
        if not bool(keep.any()):
            continue
        contributing_families.add(int(record["pose_family_id"]))
        query_parts.append(torch.as_tensor(evidence["query_descriptors"]).float()[keep])
        positive_parts.append(torch.as_tensor(evidence["positive_anchor_rows"]).long()[keep])
        negative_parts.append(negatives[keep])
        gain = max(float(evidence["actual_query_task_gain"]), 0.01)
        weight_parts.append(torch.full((int(keep.sum()),), gain))
    if not query_parts:
        raise RuntimeError("no causal ranking pair survived the two-family gate")
    query = torch.cat(query_parts)
    positive = torch.cat(positive_parts)
    negative = torch.cat(negative_parts)
    weights = torch.cat(weight_parts)
    clean_query = torch.cat(clean_query_parts) if clean_query_parts else torch.empty(0, 256)
    clean_positive = torch.cat(clean_positive_parts) if clean_positive_parts else torch.empty(0, dtype=torch.long)
    clean_negative = torch.cat(clean_negative_parts) if clean_negative_parts else torch.empty(0, dtype=torch.long)
    clean_margin = torch.cat(clean_margin_parts) if clean_margin_parts else torch.empty(0)
    metric, training_report = train_v9_shared_metric(
        anchor_features=state["anchor_features"],
        query_descriptors=query,
        positive_anchor_rows=positive,
        negative_anchor_rows=negative,
        sample_weights=weights,
        clean_query_descriptors=clean_query,
        clean_positive_anchor_rows=clean_positive,
        clean_negative_anchor_rows=clean_negative,
        clean_initial_margin=clean_margin,
        rank=args.rank,
        maximum_residual_norm=args.maximum_residual_norm,
        steps=args.steps,
        device=args.device,
    )
    transformed = transform_map_anchor_features(
        metric, state["anchor_features"], device=args.device
    )
    candidate = dict(state)
    candidate["anchor_features"] = transformed
    candidate["provenance"] = {
        **dict(state.get("provenance", {})),
        "v9_no_loo_shared_metric_action": True,
        "v9_feedback_batches": batch_inputs,
        "v9_metric_training_pose_family_count": len(contributing_families),
        "feedback_descriptors_copied_into_map": False,
        "feedback_queries_enter_mapping_csr": False,
        "loo_used": False,
        "uses_test_queries": False,
    }
    args.output_dir.mkdir(parents=True)
    output_map = args.output_dir / "projective_anchor_map.pt"
    _save(candidate, output_map)
    output_map_sha = sha256_file(output_map)
    output_metric = args.output_dir / "shared_metric.pt"
    _save(
        metric_artifact(
            metric,
            anchor_ids=candidate["anchor_ids"],
            map_path=str(output_map.resolve()),
            map_sha256=output_map_sha,
            training_report=training_report,
        ),
        output_metric,
    )
    report = {
        "schema": "lafgs_v9_shared_metric_action_report",
        "version": 1,
        "status": "PROPOSAL",
        "loo_used": False,
        "uses_test_queries": False,
        "feedback_descriptors_copied_into_map": False,
        "geometry_mutation_count": 0,
        "anchor_addition_count": 0,
        "anchor_deletion_count": 0,
        "source_map": str(map_path),
        "source_map_sha256": map_sha,
        "causal_negative_anchor_count": len(authorized_negatives),
        "training_pose_family_count": len(contributing_families),
        "training": training_report,
        "output": {
            "map": str(output_map.resolve()),
            "map_sha256": output_map_sha,
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
