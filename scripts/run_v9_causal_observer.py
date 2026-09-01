#!/usr/bin/env python3
"""Run no-LOO exact Top-K causal diagnosis on certified V9 feedback queries."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F

from common.hashing import sha256_file
from localization.matcher import global_cosine_topk
from map_learning.v9_causal_feedback import (
    anchor_unique_spatial_correspondences,
    first_correct_topk_replacement,
    require_no_loo_feedback_contract,
    standard_pose_replay,
    topk_geometric_correctness,
)
from map_learning.v8_safety_actions import certified_feedback_row_mask


def _require_sha(path: Path, expected: str | None, label: str) -> str:
    actual = sha256_file(path)
    if expected is not None and actual != expected:
        raise ValueError(f"{label} SHA256 differs")
    return actual


def _save(payload: dict, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _pose_entered_training_evidence(
    *,
    evidence: dict,
    source_descriptors: torch.Tensor,
    selected_query_rows: torch.Tensor,
    authorized: bool,
    task_gain: float,
) -> dict:
    """Keep only wrong→right rows that actually entered the alternative pose."""

    changed = torch.as_tensor(evidence["changed_query_rows"]).long().reshape(-1)
    selected = torch.as_tensor(selected_query_rows).long().reshape(-1)
    columns = {
        name: torch.as_tensor(evidence[name]).reshape(-1)
        for name in (
            "positive_anchor_rows",
            "negative_anchor_rows",
            "positive_rank",
            "positive_scores",
            "negative_scores",
        )
    }
    if any(value.numel() != changed.numel() for value in columns.values()):
        raise ValueError("V9 changed-row evidence columns do not align")
    if changed.numel() and torch.unique(changed).numel() != changed.numel():
        raise ValueError("V9 changed Query rows are duplicated")
    entered = torch.isin(changed, selected)
    keep = entered if authorized else torch.zeros_like(entered)
    rows = changed[keep]
    descriptors = torch.as_tensor(source_descriptors)
    if rows.numel() and (
        int(rows.min()) < 0 or int(rows.max()) >= descriptors.shape[0]
    ):
        raise ValueError("V9 pose-entered Query row is outside descriptors")
    return {
        "query_rows": rows,
        "query_descriptors": descriptors[rows].clone(),
        "positive_anchor_rows": columns["positive_anchor_rows"][keep],
        "negative_anchor_rows": columns["negative_anchor_rows"][keep],
        "positive_rank": columns["positive_rank"][keep],
        "positive_scores": columns["positive_scores"][keep],
        "negative_scores": columns["negative_scores"][keep],
        "alternative_pose_entered_mask": torch.ones(
            rows.numel(), dtype=torch.bool
        ),
        "candidate_changed_query_rows": changed,
        "candidate_changed_alternative_pose_entered_mask": entered,
        "actual_query_task_gain": task_gain,
    }


def _clean_protection_evidence(
    *,
    clean_rows: torch.Tensor,
    source_descriptors: torch.Tensor,
    candidate_rows: torch.Tensor,
    candidate_scores: torch.Tensor,
) -> dict:
    """Serialize clean row IDs and descriptor/top1/top2 in identical order."""

    rows = torch.as_tensor(clean_rows).long().reshape(-1)
    descriptors = torch.as_tensor(source_descriptors)
    candidates = torch.as_tensor(candidate_rows).long()
    scores = torch.as_tensor(candidate_scores).float()
    if (
        candidates.ndim != 2
        or candidates.shape != scores.shape
        or candidates.shape[1] < 2
        or descriptors.shape[0] != candidates.shape[0]
    ):
        raise ValueError("V9 clean protection source rows do not align")
    if rows.numel() and (
        int(rows.min()) < 0
        or int(rows.max()) >= descriptors.shape[0]
        or torch.unique(rows).numel() != rows.numel()
    ):
        raise ValueError("V9 clean protection rows are invalid or duplicated")
    return {
        "query_rows": rows.clone(),
        "query_descriptors": descriptors[rows].clone(),
        "positive_anchor_rows": candidates[rows, 0].clone(),
        "negative_anchor_rows": candidates[rows, 1].clone(),
        "initial_margin": (scores[rows, 0] - scores[rows, 1]).clone(),
    }


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--certified-batch", type=Path, required=True)
    parser.add_argument(
        "--expected-view-role",
        choices=("feedback_query", "confirmation_query"),
        default="feedback_query",
    )
    parser.add_argument("--expected-certified-batch-sha256")
    parser.add_argument("--map", type=Path, required=True)
    parser.add_argument("--expected-map-sha256")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--topk", type=int, default=64)
    parser.add_argument("--chunk-size", type=int, default=8192)
    parser.add_argument("--minimum-correspondences", type=int, default=16)
    parser.add_argument("--minimum-spatial-cells", type=int, default=6)
    parser.add_argument("--minimum-changed-rows", type=int, default=8)
    parser.add_argument("--minimum-task-gain", type=float, default=0.01)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.shard_count <= 0 or not 0 <= args.shard_index < args.shard_count:
        parser.error("invalid shard index/count")
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    batch_path = args.certified_batch.resolve()
    map_path = args.map.resolve()
    batch_sha = _require_sha(
        batch_path, args.expected_certified_batch_sha256, "certified batch"
    )
    map_sha = _require_sha(map_path, args.expected_map_sha256, "fixed V2 map")
    batch = json.loads(batch_path.read_text())
    if not (
        batch.get("view_role") == args.expected_view_role
        and batch.get("uses_test_queries") is False
        and batch.get("map_mutation_count") == 0
    ):
        raise ValueError("V9 observer requires immutable non-test feedback renders")
    plan_path = Path(batch["input"]["query_plan"])
    _require_sha(plan_path, batch["input"]["query_plan_sha256"], "query plan")
    plan = torch.load(plan_path, map_location="cpu", weights_only=False)
    require_no_loo_feedback_contract(plan)
    if plan.get("trajectory_interpolation_candidate_count") != 0:
        raise ValueError("trajectory interpolation is forbidden in V9")
    state = torch.load(map_path, map_location="cpu", weights_only=False)
    if state.get("provenance", {}).get("mapping_source") != (
        "gaussian_render_v2_filtered_before_projective_association"
    ):
        raise ValueError("V9 observer is not bound to the accepted V2 rebuild")
    xyz_cpu = torch.as_tensor(state["anchor_xyz"]).float().cpu()
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("V9 exact global Top-K observation requires CUDA")
    features = F.normalize(
        torch.as_tensor(state["anchor_features"], device=device).float(), dim=1
    )

    args.output_dir.mkdir(parents=True)
    records_dir = args.output_dir / "records"
    records_dir.mkdir()
    registry = []
    category_counts: Counter[str] = Counter()
    authorized_query_count = 0
    training_row_count = 0
    selected_records = [
        item
        for index, item in enumerate(batch["records"])
        if index % args.shard_count == args.shard_index
    ]
    for index, item in enumerate(selected_records):
        source_path = Path(item["path"]).resolve()
        source_sha = _require_sha(source_path, item["sha256"], "render record")
        source = torch.load(source_path, map_location="cpu", weights_only=False)
        require_no_loo_feedback_contract(source)
        accepted = source["certificate"]["decision"] == "ACCEPT"
        if accepted:
            local_valid = certified_feedback_row_mask(source["certificate"])
            if local_valid.numel() != torch.as_tensor(source["descriptors"]).shape[0]:
                raise ValueError("V2 row-valid mask does not align with descriptors")
            source_query_rows = torch.nonzero(local_valid, as_tuple=False).reshape(-1)
            if source_query_rows.numel() < args.minimum_correspondences:
                raise ValueError("ACCEPT query exposes too few V2-valid rows")
        else:
            source_query_rows = torch.arange(
                torch.as_tensor(source["descriptors"]).shape[0]
            )
        source_descriptors = torch.as_tensor(source["descriptors"])[source_query_rows]
        descriptors = F.normalize(
            source_descriptors.to(device=device).float(), dim=1
        )
        topk = global_cosine_topk(
            descriptors,
            features,
            topk=args.topk,
            chunk_size=args.chunk_size,
            anchor_descriptors_normalized=True,
        )
        candidate_rows = topk.anchor_indices.cpu()
        candidate_scores = topk.scores.cpu()
        keypoints = torch.as_tensor(source["keypoints"])[source_query_rows].float().cpu()
        baseline = standard_pose_replay(
            keypoints=keypoints + 0.5,
            anchor_rows=candidate_rows[:, 0],
            anchor_xyz=xyz_cpu,
            intrinsic=source["intrinsics"],
            ground_truth_w2c=source["pose_w2c"],
        )
        baseline_success = bool(
            baseline["translation_error_cm"] < 5.0
            and baseline["rotation_error_deg"] < 5.0
        )
        clean_rows = torch.empty(0, dtype=torch.long)
        evidence = {
            "supported_query_rows": torch.empty(0, dtype=torch.long),
            "changed_query_rows": torch.empty(0, dtype=torch.long),
            "positive_anchor_rows": torch.empty(0, dtype=torch.long),
            "negative_anchor_rows": torch.empty(0, dtype=torch.long),
            "positive_rank": torch.empty(0, dtype=torch.long),
            "positive_scores": torch.empty(0),
            "negative_scores": torch.empty(0),
        }
        alternative = None
        selected_query_rows = torch.empty(0, dtype=torch.long)
        spatial_cells = 0
        if accepted:
            correct = topk_geometric_correctness(
                keypoints=keypoints,
                candidate_anchor_rows=candidate_rows,
                anchor_xyz=xyz_cpu,
                pose_w2c=source["pose_w2c"],
                intrinsic=source["intrinsics"],
                alpha=source["alpha_float16"],
                depth=source["depth_float16"],
                row_valid=torch.ones(keypoints.shape[0], dtype=torch.bool),
            )
            evidence = first_correct_topk_replacement(
                candidate_rows, candidate_scores, correct
            )
            clean_rows = torch.nonzero(correct[:, 0], as_tuple=False).reshape(-1)
            if clean_rows.numel() > 256:
                clean_rows = clean_rows[:256]
            supported = evidence["supported_query_rows"]
            if supported.numel():
                first_rank = correct.float().argmax(1).long()
                selected_anchor_rows = candidate_rows[
                    supported, first_rank[supported]
                ]
                selected_scores = candidate_scores[supported, first_rank[supported]]
                unique = anchor_unique_spatial_correspondences(
                    keypoints=keypoints[supported],
                    anchor_rows=selected_anchor_rows,
                    scores=selected_scores,
                    image_hw=source["image_hw"],
                )
                selected_query_rows = supported[unique["selected_rows"]]
                spatial_cells = unique["spatial_cell_count"]
                if (
                    selected_query_rows.numel() >= args.minimum_correspondences
                    and spatial_cells >= args.minimum_spatial_cells
                    and evidence["changed_query_rows"].numel()
                    >= args.minimum_changed_rows
                ):
                    alternative = standard_pose_replay(
                        keypoints=keypoints[selected_query_rows] + 0.5,
                        anchor_rows=unique["anchor_rows"],
                        anchor_xyz=xyz_cpu,
                        intrinsic=source["intrinsics"],
                        ground_truth_w2c=source["pose_w2c"],
                    )
        task_gain = (
            float("nan")
            if alternative is None
            else baseline["task_error"] - alternative["task_error"]
        )
        alternative_success = bool(
            alternative is not None
            and alternative["inlier_count"] > 0
            and alternative["translation_error_cm"] < 5.0
            and alternative["rotation_error_deg"] < 5.0
        )
        precision_authorized = bool(
            accepted
            and baseline_success
            and alternative is not None
            and alternative["inlier_count"] > 0
            and task_gain >= args.minimum_task_gain
        )
        recovery_authorized = bool(
            accepted
            and not baseline_success
            and alternative_success
            and task_gain >= args.minimum_task_gain
        )
        authorized = bool(
            args.expected_view_role == "feedback_query"
            and (precision_authorized or recovery_authorized)
        )
        if not accepted:
            category = "unreliable_query"
        elif precision_authorized:
            category = "causal_precision_deficit"
        elif recovery_authorized:
            category = "causal_recoverable_failure"
        elif baseline_success:
            category = "nominal_success"
        else:
            category = "unresolved_failure"
        category_counts[category] += 1
        if authorized:
            authorized_query_count += 1
        training_evidence = _pose_entered_training_evidence(
            evidence=evidence,
            source_descriptors=source_descriptors,
            selected_query_rows=selected_query_rows,
            authorized=authorized,
            task_gain=task_gain,
        )
        training_row_count += int(training_evidence["query_rows"].numel())
        clean_protection = _clean_protection_evidence(
            clean_rows=clean_rows,
            source_descriptors=source_descriptors,
            candidate_rows=candidate_rows,
            candidate_scores=candidate_scores,
        )
        record = {
            "schema": "lafgs_v9_no_loo_causal_feedback_record",
            "version": 2,
            "query_index": int(source["query_index"]),
            "pose_family_id": int(source["pose_family_id"]),
            "source_record": str(source_path),
            "source_record_sha256": source_sha,
            "certificate_decision": source["certificate"]["decision"],
            "source_query_rows": source_query_rows.clone(),
            "invalid_source_row_count": int(
                torch.as_tensor(source["descriptors"]).shape[0]
                - source_query_rows.numel()
            ),
            "baseline": baseline,
            "alternative": alternative,
            "baseline_success": baseline_success,
            "alternative_success": alternative_success,
            "actual_task_gain": task_gain,
            "category": category,
            "can_train_metric": authorized,
            "topk_anchor_rows": candidate_rows,
            "topk_scores": candidate_scores.to(torch.float16),
            "training_evidence": training_evidence,
            "clean_protection_evidence": clean_protection,
            "spatial_cell_count": spatial_cells,
            "loo_used": False,
            "enters_track_registry": False,
            "enters_anchor_observation_csr": False,
            "enters_descriptor_bank": False,
            "feedback_descriptors_copied_into_map": False,
            "map_mutation_count": 0,
        }
        path = records_dir / f"query_{int(source['query_index']):04d}.pt"
        _save(record, path)
        registry.append(
            {
                "query_index": record["query_index"],
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "category": category,
                "can_train_metric": authorized,
            }
        )
        if (index + 1) % 8 == 0 or index + 1 == len(selected_records):
            print(
                f"V9 causal observer shard {args.shard_index}: "
                f"{index + 1}/{len(selected_records)}",
                flush=True,
            )
    manifest = {
        "schema": "lafgs_v9_no_loo_causal_feedback_batch",
        "version": 2,
        "status": "PASS",
        "role": (
            "observer_pool"
            if args.expected_view_role == "feedback_query"
            else "confirmation_observer"
        ),
        "source_view_role": args.expected_view_role,
        "confirmation_can_train_or_select": False,
        "loo_used": False,
        "trajectory_interpolation_used": False,
        "feedback_descriptors_copied_into_map": False,
        "uses_test_queries": False,
        "map_mutation_count": 0,
        "topk": args.topk,
        "accepted_query_row_policy": "v2_row_valid_only",
        "training_rows_are_alternative_pose_entered_only": True,
        "clean_protection_has_explicit_query_rows": True,
        "query_count": len(registry),
        "source_query_count": len(batch["records"]),
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "category_counts": dict(category_counts),
        "authorized_metric_query_count": authorized_query_count,
        "authorized_metric_training_row_count": training_row_count,
        "input": {
            "certified_batch": str(batch_path),
            "certified_batch_sha256": batch_sha,
            "map": str(map_path),
            "map_sha256": map_sha,
        },
        "records": registry,
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
