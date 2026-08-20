#!/usr/bin/env python3
"""Measure exact GT-positive Top-K and matching-rank headroom without PoseLib."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from common.hashing import sha256_file
from localization.matcher import TopKMatches, maximum_weight_anchor_assignment
from topology.assignment_replay import validate_mapping_topk
from topology.deployment_revision import _csr_contains_at_rows


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        json.loads(temporary.read_text())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sidecar", type=Path, required=True)
    parser.add_argument("--expected-sidecar-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cpu-threads", type=int, default=2)
    args = parser.parse_args()
    if int(args.cpu_threads) < 1:
        raise ValueError("CPU thread count must be positive")
    torch.set_num_threads(int(args.cpu_threads))
    sidecar_path = args.sidecar.resolve()
    if sha256_file(sidecar_path) != args.expected_sidecar_sha256:
        raise ValueError("mapping Top-K sidecar SHA differs")
    sidecar = torch.load(sidecar_path, map_location="cpu", weights_only=False)
    validate_mapping_topk(sidecar)
    if sidecar.get("uses_test_queries") is not False:
        raise ValueError("headroom audit requires mapping-only candidates")
    repository = Path(__file__).resolve().parents[1]
    identity = sidecar.get("producer_identity", {})
    if identity.get("worktree_clean") is not True:
        raise ValueError("mapping Top-K sidecar producer was not clean")
    for relative, expected in identity.get("source_sha256", {}).items():
        if sha256_file(repository / relative) != expected:
            raise ValueError(f"mapping Top-K producer source differs: {relative}")
    teacher_path = Path(sidecar["inputs"]["teacher"]).resolve()
    if sha256_file(teacher_path) != sidecar["input_sha256"]["teacher"]:
        raise ValueError("mapping teacher SHA differs")
    teacher = torch.load(teacher_path, map_location="cpu", weights_only=False)
    if list(teacher.get("query_names", ())) != list(sidecar["query_names"]):
        raise ValueError("sidecar and teacher query registries differ")

    topk = int(sidecar["topk"])
    ranks = tuple(value for value in (1, 2, 4, 8) if value <= topk)
    totals = {rank: 0 for rank in ranks}
    positive_rows = 0
    row_count = 0
    oracle_rank = 0
    top1_collisions = 0
    per_query = []
    for query_index, (candidate, record) in enumerate(
        zip(sidecar["records"], teacher["records"])
    ):
        indices = torch.as_tensor(candidate["anchor_indices"]).long()
        local_rows = torch.as_tensor(candidate["teacher_local_rows"]).long()
        correct_by_rank = []
        has_positive = None
        for rank in range(topk):
            correct, current_has_positive = _csr_contains_at_rows(
                record, "positive", local_rows, indices[:, rank]
            )
            correct_by_rank.append(correct)
            if has_positive is None:
                has_positive = current_has_positive
            elif not torch.equal(has_positive, current_has_positive):
                raise RuntimeError("positive-row registry changed across ranks")
        correct_matrix = torch.stack(correct_by_rank, dim=1)
        query_positive = int(has_positive.sum())
        cumulative = correct_matrix.cumsum(dim=1).clamp_max(1).bool()
        query_hits = {
            rank: int(cumulative[:, rank - 1][has_positive].sum()) for rank in ranks
        }
        # Move all correct edges ahead of dustbin-valued incorrect edges while
        # retaining stable candidate-rank order within each class.  Unit real
        # edge utility then makes the sparse optimum maximum-cardinality.
        oracle_order = torch.argsort(
            correct_matrix.float(), dim=1, descending=True, stable=True
        )
        oracle_indices = torch.gather(indices, 1, oracle_order)
        oracle_scores = torch.gather(correct_matrix.float(), 1, oracle_order)
        oracle = maximum_weight_anchor_assignment(
            TopKMatches(
                keypoint_indices=torch.arange(indices.shape[0]),
                anchor_indices=oracle_indices,
                scores=oracle_scores,
            ),
            dustbin_score=0.0,
        )
        query_oracle = int(oracle.matches.anchor_indices.numel())
        query_collisions = int(indices.shape[0] - torch.unique(indices[:, 0]).numel())
        row_count += int(indices.shape[0])
        positive_rows += query_positive
        oracle_rank += query_oracle
        top1_collisions += query_collisions
        for rank in ranks:
            totals[rank] += query_hits[rank]
        per_query.append(
            {
                "query_index": query_index,
                "image_name": candidate["image_name"],
                "row_count": int(indices.shape[0]),
                "positive_row_count": query_positive,
                "positive_hits_at_k": {str(k): query_hits[k] for k in ranks},
                "maximum_correct_matching_rank": query_oracle,
                "top1_collision_count": query_collisions,
            }
        )
    recall = {
        str(rank): (100.0 * totals[rank] / positive_rows if positive_rows else 0.0)
        for rank in ranks
    }
    output = {
        "schema": "lafgs_v4_mapping_topk_headroom_audit",
        "version": 1,
        "uses_test_queries": False,
        "sidecar": str(sidecar_path),
        "sidecar_sha256": args.expected_sidecar_sha256,
        "query_count": len(per_query),
        "row_count": row_count,
        "positive_row_count": positive_rows,
        "positive_recall_percent_at_k": recall,
        "maximum_correct_matching_rank_sum": oracle_rank,
        "maximum_correct_matching_rank_percent": (
            100.0 * oracle_rank / positive_rows if positive_rows else 0.0
        ),
        "top1_collision_count": top1_collisions,
        "top8_headroom_over_top1_percentage_points": (
            recall.get("8", recall[str(ranks[-1])]) - recall["1"]
        ),
        "queries": per_query,
    }
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output.resolve(), output)
    print(
        json.dumps(
            {key: value for key, value in output.items() if key != "queries"},
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
