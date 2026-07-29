#!/usr/bin/env python3
"""Audit whether basin supervision changed the field rather than the solver."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", required=True)
    parser.add_argument("--basin-teacher", required=True)
    parser.add_argument("--query-cache", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    state = torch.load(args.map, map_location="cpu", weights_only=False)
    teacher = torch.load(
        args.basin_teacher, map_location="cpu", weights_only=False
    )
    cache_payload = torch.load(
        args.query_cache, map_location="cpu", weights_only=False
    )
    cache = cache_payload.get("queries", cache_payload)
    device = torch.device("cuda")
    bank = F.normalize(torch.as_tensor(state["anchor_features"]).float(), dim=1).to(
        device
    )
    good_margins = []
    good_top1 = []
    blame_margins = []
    blame_weights = []
    for record in teacher["records"]:
        name = record["query_name"]
        descriptors = F.normalize(
            torch.as_tensor(cache[name]["native_descriptors"]).float(), dim=1
        ).to(device)
        types = torch.as_tensor(record["set_types"]).long()
        correct = torch.as_tensor(record["correct_basin"]).bool()
        good = torch.nonzero(
            correct & ((types == 0) | (types == 2)), as_tuple=False
        ).reshape(-1)
        if good.numel():
            rows = torch.as_tensor(record["set_query_rows"]).long()[good].reshape(-1)
            anchors = (
                torch.as_tensor(record["set_anchor_indices"]).long()[good].reshape(-1)
            )
            query = descriptors[rows.to(device)]
            score = query @ bank.T
            positive = score.gather(1, anchors.to(device)[:, None]).reshape(-1)
            top2 = torch.topk(score, k=2, dim=1)
            top1_anchor = top2.indices[:, 0]
            negative = torch.where(
                top1_anchor == anchors.to(device),
                top2.values[:, 1],
                top2.values[:, 0],
            )
            good_margins.extend((positive - negative).cpu().tolist())
            good_top1.extend((top1_anchor == anchors.to(device)).cpu().tolist())
        rows = torch.as_tensor(record["blame_rows"]).long()
        if rows.numel():
            query = descriptors[rows.to(device)]
            harmful = torch.as_tensor(
                record["blame_harmful_anchors"]
            ).long().to(device)
            positive = torch.as_tensor(
                record["blame_positive_anchors"]
            ).long().to(device)
            margin = torch.einsum("bd,bd->b", query, bank[positive]) - torch.einsum(
                "bd,bd->b", query, bank[harmful]
            )
            blame_margins.extend(margin.cpu().tolist())
            blame_weights.extend(
                torch.as_tensor(record["blame_weights"]).float().tolist()
            )
    good_array = np.asarray(good_margins, dtype=np.float64)
    blame_array = np.asarray(blame_margins, dtype=np.float64)
    weights = np.asarray(blame_weights, dtype=np.float64)
    summary = {
        "schema": "lafgs_basin_field_audit",
        "map": str(Path(args.map).resolve()),
        "anchor_count": int(bank.shape[0]),
        "good_edge_count": int(good_array.size),
        "good_top1_rate_percent": float(np.mean(good_top1) * 100.0),
        "good_margin_mean": float(good_array.mean()),
        "good_margin_median": float(np.median(good_array)),
        "blame_edge_count": int(blame_array.size),
        "blame_positive_win_rate_percent": float(
            np.mean(blame_array > 0) * 100.0
        ),
        "blame_margin_mean": float(blame_array.mean()),
        "blame_margin_median": float(np.median(blame_array)),
        "blame_weighted_margin_mean": float(
            np.average(blame_array, weights=np.maximum(weights, 1e-8))
        ),
    }
    Path(args.output).write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
