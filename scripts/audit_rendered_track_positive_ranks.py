#!/usr/bin/env python3
"""Audit positive-anchor ranks using mapping-sequence crossfit only.

The audit answers whether correct ray-triangulated geometry is absent from the
descriptor shortlist or merely loses global Top-1.  Every held sequence uses
the already materialized fold map and metric whose descriptors exclude that
sequence.  No test query or source mapping RGB is accepted.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F

from localization.localizer import load_shared_metric
from scripts.evaluate_rendered_track_crossfit import _sequence_name


def positive_rank_hits(
    ranked: torch.Tensor,
    positive_offsets: torch.Tensor,
    positive_indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return first positive rank (one based) and positive-row mask."""
    ranked = torch.as_tensor(ranked).long()
    offsets = torch.as_tensor(positive_offsets).long()
    positives = torch.as_tensor(positive_indices).long()
    if ranked.ndim != 2 or offsets.shape != (ranked.shape[0] + 1,):
        raise ValueError("ranked rows and positive CSR do not align")
    counts = offsets[1:] - offsets[:-1]
    if bool((counts < 0).any()) or int(offsets[0]) != 0:
        raise ValueError("positive CSR offsets are invalid")
    if int(offsets[-1]) != int(positives.numel()):
        raise ValueError("positive CSR terminal offset is invalid")
    row_ids = torch.repeat_interleave(torch.arange(ranked.shape[0]), counts)
    membership = torch.zeros_like(ranked, dtype=torch.bool)
    if positives.numel():
        # A teacher row usually has few positives.  Compare only entries that
        # share a row rather than materializing a dense row-by-anchor mask.
        for rank in range(ranked.shape[1]):
            matched = positives == ranked[row_ids, rank]
            membership[row_ids[matched], rank] = True
    has_positive = counts > 0
    sentinel = ranked.shape[1] + 1
    first = torch.full((ranked.shape[0],), sentinel, dtype=torch.long)
    hit = membership.any(dim=1)
    first[hit] = membership[hit].to(torch.int64).argmax(dim=1) + 1
    return first, has_positive


def _atomic_json(payload: dict, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@torch.inference_mode()
def run(args) -> dict:
    cache_payload = torch.load(args.query_cache, map_location="cpu", weights_only=False)
    if (
        cache_payload.get("uses_source_mapping_rgb") is not False
        or cache_payload.get("uses_test_queries") is not False
    ):
        raise ValueError("rank audit requires rendered mapping-only cache")
    cache = cache_payload.get("queries", cache_payload)
    fold_reports = []
    total_positive_rows = 0
    total_hits = {rank: 0 for rank in args.ranks}
    for fold_dir in sorted(
        path for path in args.crossfit_dir.iterdir() if path.is_dir()
    ):
        held_sequence = fold_dir.name
        state = torch.load(
            fold_dir / "anchor_map.pt", map_location="cpu", weights_only=False
        )
        teacher = torch.load(
            fold_dir / "positive_teacher.pt", map_location="cpu", weights_only=False
        )
        metric = load_shared_metric(
            fold_dir / "metric_state.pt",
            anchor_ids=torch.as_tensor(state["anchor_ids"]).long(),
            device=torch.device(args.device),
        )
        bank = F.normalize(
            torch.as_tensor(state["anchor_features"])
            .float()
            .to(torch.device(args.device)),
            dim=1,
        )
        maximum_rank = min(max(args.ranks), int(bank.shape[0]))
        fold_positive_rows = 0
        fold_hits = {rank: 0 for rank in args.ranks}
        per_query = []
        for query_index, (name, record) in enumerate(
            zip(teacher["query_names"], teacher["records"])
        ):
            if _sequence_name(name) != held_sequence:
                continue
            rows = torch.as_tensor(record["query_rows"]).long()
            descriptors = F.normalize(
                torch.as_tensor(cache[name]["native_descriptors"])[rows].float(), dim=1
            ).to(torch.device(args.device))
            adapted, _ = metric(descriptors)
            ranked = torch.topk(adapted @ bank.T, k=maximum_rank, dim=1).indices.cpu()
            first, has_positive = positive_rank_hits(
                ranked,
                record["positive_offsets"],
                record["positive_indices"],
            )
            positive_count = int(has_positive.sum())
            fold_positive_rows += positive_count
            query_hits = {}
            for rank in args.ranks:
                hits = int(((first <= rank) & has_positive).sum())
                fold_hits[rank] += hits
                query_hits[str(rank)] = hits
            per_query.append(
                {
                    "query_index": query_index,
                    "query_name": name,
                    "positive_row_count": positive_count,
                    "hits": query_hits,
                }
            )
        if not fold_positive_rows:
            raise RuntimeError(f"held sequence {held_sequence} has no positive rows")
        total_positive_rows += fold_positive_rows
        for rank in args.ranks:
            total_hits[rank] += fold_hits[rank]
        fold_reports.append(
            {
                "held_sequence": held_sequence,
                "positive_row_count": fold_positive_rows,
                "hit_count": {str(rank): fold_hits[rank] for rank in args.ranks},
                "hit_fraction": {
                    str(rank): fold_hits[rank] / fold_positive_rows
                    for rank in args.ranks
                },
                "queries": per_query,
            }
        )
        print(
            json.dumps(
                {
                    key: value
                    for key, value in fold_reports[-1].items()
                    if key != "queries"
                },
                sort_keys=True,
            ),
            flush=True,
        )
    report = {
        "schema": "lafgs_rendered_track_positive_rank_audit",
        "version": 1,
        "uses_source_mapping_rgb": False,
        "uses_test_queries": False,
        "ranks": args.ranks,
        "folds": fold_reports,
        "combined": {
            "positive_row_count": total_positive_rows,
            "hit_count": {str(rank): total_hits[rank] for rank in args.ranks},
            "hit_fraction": {
                str(rank): total_hits[rank] / total_positive_rows for rank in args.ranks
            },
        },
    }
    _atomic_json(report, args.output)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crossfit-dir", type=Path, required=True)
    parser.add_argument("--query-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ranks", default="1,4,16,64")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    args.crossfit_dir = args.crossfit_dir.resolve()
    args.query_cache = args.query_cache.resolve()
    args.output = args.output.resolve()
    args.ranks = sorted({int(value) for value in args.ranks.split(",")})
    if not args.ranks or min(args.ranks) < 1:
        raise ValueError("ranks must be positive")
    if args.output.exists():
        raise FileExistsError(args.output)
    print(json.dumps(run(args), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
