#!/usr/bin/env python3
"""Materialize the frozen V16 certified Top-L truth as an explicit fallback.

This artifact is intentionally *not* a descriptor-independent full-map truth
graph.  It preserves the previously validated V16 geometric certification so
the V18 controller mechanisms can be evaluated after the provenance teacher
replacement gate fails.  Consumers must opt in to this fallback explicitly.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from common.hashing import sha256_file
from map_learning.v16_competitive_sufficiency import certify_topl_relations
from map_learning.v18_provenance_truth import (
    TRUTH_EQUIVALENT,
    TRUTH_INVALID,
    TRUTH_NONE,
    TRUTH_STATUS_NAMES,
    TRUTH_UNIQUE,
)


def _atomic_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _truth_from_positive_mask(
    candidates: torch.Tensor,
    positive: torch.Tensor,
    valid_rows: torch.Tensor,
) -> dict:
    candidate_rows = torch.as_tensor(candidates).long()
    positive_mask = torch.as_tensor(positive).bool()
    valid = torch.as_tensor(valid_rows).bool().reshape(-1)
    if candidate_rows.shape != positive_mask.shape or candidate_rows.shape[0] != valid.numel():
        raise ValueError("certified projection truth rows do not align")
    status = torch.full((valid.numel(),), TRUTH_NONE, dtype=torch.int8)
    offsets = [0]
    anchors: list[int] = []
    for row in range(valid.numel()):
        if not bool(valid[row]):
            status[row] = TRUTH_INVALID
            offsets.append(len(anchors))
            continue
        values = sorted(set(candidate_rows[row, positive_mask[row]].tolist()))
        if not values:
            status[row] = TRUTH_NONE
        elif len(values) == 1:
            status[row] = TRUTH_UNIQUE
            anchors.extend(values)
        else:
            status[row] = TRUTH_EQUIVALENT
            anchors.extend(values)
        offsets.append(len(anchors))
    return {
        "schema": "lafgs_v18_certified_topl_projection_truth",
        "version": 1,
        "row_count": int(valid.numel()),
        "truth_status": status,
        "truth_status_names": TRUTH_STATUS_NAMES,
        "truth_offsets": torch.tensor(offsets, dtype=torch.long),
        "truth_anchor_rows": torch.tensor(anchors, dtype=torch.long),
        "status_counts": {
            name: int((status == code).sum())
            for code, name in enumerate(TRUTH_STATUS_NAMES)
        },
        "uses_descriptor_scores": False,
        "uses_topl_candidates": True,
        "fallback_only": True,
    }


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design-batch", type=Path, required=True)
    parser.add_argument("--baseline-map", type=Path, required=True)
    parser.add_argument("--mapping-evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--progress-interval", type=int, default=10)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    design = json.loads(args.design_batch.read_text())
    baseline = torch.load(args.baseline_map, map_location="cpu", weights_only=False)
    evidence = torch.load(
        args.mapping_evidence, map_location="cpu", weights_only=False
    )
    anchor_count = int(torch.as_tensor(baseline["anchor_ids"]).numel())
    if not (
        design.get("schema") == "lafgs_v9_no_loo_causal_feedback_batch"
        and design.get("role") == "controller_design"
        and design.get("uses_test_queries") is False
        and design.get("loo_used") is False
        and evidence.get("schema")
        == "lafgs_v7_reconstructed_mapping_candidate_evidence"
        and int(evidence.get("candidate_count", -1)) == anchor_count
    ):
        raise ValueError("V18 certified fallback inputs differ from the frozen plant")

    records = []
    diagnostics = {
        "truth_row_count": 0,
        "unique_or_equivalent_count": 0,
        "none_count": 0,
        "invalid_count": 0,
    }
    accepted = []
    for item in design["records"]:
        observed = torch.load(item["path"], map_location="cpu", weights_only=False)
        if observed["certificate_decision"] == "ACCEPT":
            accepted.append(observed)
    for completed, observed in enumerate(accepted, start=1):
        source_path = Path(observed["source_record"]).resolve()
        if sha256_file(source_path) != observed["source_record_sha256"]:
            raise ValueError("V18 certified fallback source SHA256 differs")
        source = torch.load(source_path, map_location="cpu", weights_only=False)
        rows = torch.as_tensor(observed["source_query_rows"]).long()
        keypoints = torch.as_tensor(source["keypoints"])[rows].float() + 0.5
        candidates = torch.as_tensor(observed["topk_anchor_rows"]).long()
        relations = certify_topl_relations(
            keypoints=keypoints,
            candidate_anchor_rows=candidates,
            anchor_xyz=baseline["anchor_xyz"],
            anchor_covariance=baseline["anchor_position_covariance"],
            observation_count=evidence["observation_count"],
            view_family_count=evidence["view_family_count"],
            pose_w2c=source["pose_w2c"],
            intrinsic=source["intrinsics"],
            alpha=source["alpha_float16"],
            depth=source["depth_float16"],
            surface_median_depth=source.get("surface_median_depth_float16"),
            row_valid=torch.ones(rows.numel(), dtype=torch.bool),
        )
        supported = (
            relations["positive"] | relations["ambiguous"] | relations["negative"]
        ).any(1)
        truth = _truth_from_positive_mask(
            candidates, relations["positive"], supported
        )
        status = torch.as_tensor(truth["truth_status"]).long()
        diagnostics["truth_row_count"] += int(rows.numel())
        diagnostics["unique_or_equivalent_count"] += int(
            ((status == TRUTH_UNIQUE) | (status == TRUTH_EQUIVALENT)).sum()
        )
        diagnostics["none_count"] += int((status == TRUTH_NONE).sum())
        diagnostics["invalid_count"] += int((status == TRUTH_INVALID).sum())
        records.append(
            {
                "query_index": int(observed["query_index"]),
                "pose_family_id": int(observed["pose_family_id"]),
                "source_query_rows": rows,
                "truth": truth,
            }
        )
        if completed % max(int(args.progress_interval), 1) == 0 or completed == len(accepted):
            print(
                json.dumps(
                    {
                        "completed_queries": completed,
                        "accepted_queries": len(accepted),
                        **diagnostics,
                    }
                ),
                flush=True,
            )

    artifact = {
        "schema": "lafgs_v18_feedback_certified_projection_truth_batch",
        "version": 1,
        "uses_test_queries": False,
        "loo_used": False,
        "truth_source": "legacy_v16_certified_topl_geometry_fallback",
        "descriptor_independent_full_map_truth": False,
        "controller_replacement_authorized": False,
        "explicit_fallback_required": True,
        "accepted_query_count": len(records),
        "diagnostics": diagnostics,
        "records": records,
        "inputs": {
            "design_batch": str(args.design_batch.resolve()),
            "design_batch_sha256": sha256_file(args.design_batch),
            "baseline_map": str(args.baseline_map.resolve()),
            "baseline_map_sha256": sha256_file(args.baseline_map),
            "mapping_evidence": str(args.mapping_evidence.resolve()),
            "mapping_evidence_sha256": sha256_file(args.mapping_evidence),
        },
    }
    _atomic_save(artifact, args.output.resolve())
    report = {key: value for key, value in artifact.items() if key != "records"}
    report["output"] = str(args.output.resolve())
    report["output_sha256"] = sha256_file(args.output)
    args.output.with_suffix(".json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
