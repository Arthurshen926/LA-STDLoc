#!/usr/bin/env python3
"""Audit repeated top-1 assignments by localization-anchor provenance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from localization.localizer import load_shared_metric
from map_learning.trainer import _csr_first_k


def _dense_csr(record: dict, prefix: str) -> torch.Tensor:
    offsets = torch.as_tensor(record[f"{prefix}_offsets"]).long()
    counts = offsets[1:] - offsets[:-1]
    width = max(int(counts.max().item()), 1)
    return _csr_first_k(offsets, record[f"{prefix}_indices"], width)


def _safe_ratio(numerator: torch.Tensor, denominator: torch.Tensor) -> float:
    return float(numerator.sum() / denominator.sum().clamp_min(1))


def _category_summary(
    mask: torch.Tensor,
    *,
    winner: torch.Tensor,
    correct: torch.Tensor,
    false: torch.Tensor,
    collision_rows: torch.Tensor,
    collision_queries: torch.Tensor,
    margin_sum: torch.Tensor,
) -> dict[str, float | int]:
    category_winners = winner[mask]
    category_false = false[mask]
    category_collision = collision_rows[mask]
    return {
        "anchor_count": int(mask.sum()),
        "winner_count": int(category_winners.sum()),
        "correct_winner_count": int(correct[mask].sum()),
        "false_winner_count": int(category_false.sum()),
        "false_winner_rate": _safe_ratio(category_false, category_winners),
        "collision_anchor_count": int((category_collision > 0).sum()),
        "collision_row_count": int(category_collision.sum()),
        "collision_query_count": int(collision_queries[mask].sum()),
        "collision_fraction_of_false": _safe_ratio(
            category_collision, category_false
        ),
        "collision_margin_mean": _safe_ratio(
            margin_sum[mask], category_collision
        ),
    }


@torch.inference_mode()
def audit(
    *,
    map_path: Path,
    metric_state_path: Path,
    teacher_path: Path,
    query_cache_path: Path,
    deployment_row_limit: int,
    device: torch.device,
) -> dict:
    state = torch.load(map_path, map_location="cpu", weights_only=False)
    teacher = torch.load(teacher_path, map_location="cpu", weights_only=False)
    cache_payload = torch.load(
        query_cache_path, map_location="cpu", weights_only=False
    )
    cache = cache_payload.get("queries", cache_payload)
    names = list(teacher["query_names"])
    anchor_ids = torch.as_tensor(state["anchor_ids"]).long()
    anchor_type = torch.as_tensor(state["anchor_type"]).long()
    bank = F.normalize(
        torch.as_tensor(state["anchor_features"]).float().to(device), dim=1
    )
    metric = load_shared_metric(
        metric_state_path, anchor_ids=anchor_ids, device=device
    )
    count = int(anchor_ids.numel())
    counters = {
        name: torch.zeros(count, dtype=torch.float64)
        for name in (
            "winner",
            "correct",
            "false",
            "collision_rows",
            "collision_queries",
            "margin_sum",
        )
    }
    query_collision_count = 0
    query_count = len(names)
    for query_index, (name, record) in enumerate(
        zip(names, teacher["records"]), start=1
    ):
        rows = torch.as_tensor(record["query_rows"]).long()
        positions = torch.arange(rows.numel())
        if int(deployment_row_limit) > 0:
            keep = rows < int(deployment_row_limit)
            rows = rows[keep]
            positions = positions[keep]
        descriptors = F.normalize(
            torch.as_tensor(cache[name]["native_descriptors"]).float()[rows], dim=1
        ).to(device)
        adapted, _ = metric(descriptors)
        scores, indices = torch.topk(adapted @ bank.T, k=2, dim=1)
        winners = indices[:, 0].cpu()
        margins = (scores[:, 0] - scores[:, 1]).cpu().double()
        positives = _dense_csr(record, "positive")[positions]
        ambiguous = _dense_csr(record, "ambiguous")[positions]
        is_correct = (
            (positives == winners[:, None]) & (positives >= 0)
        ).any(dim=1)
        is_ambiguous = (
            (ambiguous == winners[:, None]) & (ambiguous >= 0)
        ).any(dim=1)
        has_positive = (positives >= 0).any(dim=1)
        is_false = has_positive & ~is_correct & ~is_ambiguous
        ones = torch.ones(winners.numel(), dtype=torch.float64)
        counters["winner"].index_add_(0, winners, ones)
        counters["correct"].index_add_(
            0, winners[is_correct], torch.ones(int(is_correct.sum()), dtype=torch.float64)
        )
        counters["false"].index_add_(
            0, winners[is_false], torch.ones(int(is_false.sum()), dtype=torch.float64)
        )
        false_winners = winners[is_false]
        false_margins = margins[is_false]
        if false_winners.numel():
            unique, inverse, local_counts = torch.unique(
                false_winners, return_inverse=True, return_counts=True
            )
            repeated = local_counts[inverse] >= 2
            if bool(repeated.any()):
                query_collision_count += 1
                collision_winners = false_winners[repeated]
                counters["collision_rows"].index_add_(
                    0,
                    collision_winners,
                    torch.ones(collision_winners.numel(), dtype=torch.float64),
                )
                counters["margin_sum"].index_add_(
                    0, collision_winners, false_margins[repeated]
                )
                repeated_anchors = unique[local_counts >= 2]
                counters["collision_queries"].index_add_(
                    0,
                    repeated_anchors,
                    torch.ones(repeated_anchors.numel(), dtype=torch.float64),
                )
        if query_index % 100 == 0 or query_index == query_count:
            print(
                json.dumps(
                    {
                        "event": "alias_provenance_audit",
                        "queries_complete": query_index,
                        "query_count": query_count,
                    }
                ),
                flush=True,
            )

    categories = {
        "all": torch.ones(count, dtype=torch.bool),
        "gaussian_reserve": anchor_type == 0,
        "track_anchor": anchor_type == 1,
    }
    return {
        "schema": "lafgs_alias_provenance_audit",
        "version": 1,
        "uses_test_queries": False,
        "query_count": query_count,
        "queries_with_repeated_false_assignment": int(query_collision_count),
        "deployment_row_limit": int(deployment_row_limit),
        "categories": {
            name: _category_summary(mask, **counters)
            for name, mask in categories.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", type=Path, required=True)
    parser.add_argument("--metric-state", type=Path, required=True)
    parser.add_argument("--complete-positive-teacher", type=Path, required=True)
    parser.add_argument("--query-cache", type=Path, required=True)
    parser.add_argument("--deployment-row-limit", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    report = audit(
        map_path=args.map.resolve(),
        metric_state_path=args.metric_state.resolve(),
        teacher_path=args.complete_positive_teacher.resolve(),
        query_cache_path=args.query_cache.resolve(),
        deployment_row_limit=args.deployment_row_limit,
        device=torch.device(args.device),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
