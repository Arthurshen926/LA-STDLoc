#!/usr/bin/env python3
"""Build compact maps from query-level threshold rescue attribution."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from scripts.build_lafgs_lgo_candidate_maps import _materialize


def _selected_rows(state: dict) -> torch.Tensor:
    return torch.as_tensor(
        state["functional_pruning"]["selected_source_rows"]
    ).long()


def main() -> None:
    torch.set_num_threads(1)
    parser = argparse.ArgumentParser()
    parser.add_argument("--anchor-map", required=True)
    parser.add_argument("--compact-map", required=True)
    parser.add_argument("--function-graph", required=True)
    parser.add_argument("--baselines", required=True)
    parser.add_argument("--groups", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--budgets", nargs="+", type=int, default=[32500, 33000, 34000]
    )
    parser.add_argument("--r5-weight", type=float, default=4.0)
    parser.add_argument("--regression-weight", type=float, default=5.0)
    parser.add_argument("--tail-clip-m", type=float, default=0.25)
    args = parser.parse_args()
    state = torch.load(
        args.anchor_map, map_location="cpu", weights_only=False
    )
    compact = torch.load(
        args.compact_map, map_location="cpu", weights_only=False
    )
    graph = torch.load(
        args.function_graph, map_location="cpu", weights_only=False
    )
    baselines = torch.load(
        args.baselines, map_location="cpu", weights_only=False
    )
    groups = torch.load(
        args.groups, map_location="cpu", weights_only=False
    )["groups"]
    count = int(graph["anchor_count"])
    compact_rows = _selected_rows(compact)
    compact_mask = torch.zeros(count, dtype=torch.bool)
    compact_mask[compact_rows] = True
    graph_records = {
        int(record["query_index"]): record
        for record in graph["records"]
    }
    states = {
        int(budget): {
            int(record["query_index"]): record
            for record in payload["records"]
        }
        for budget, payload in baselines["states"].items()
    }
    if 30000 not in states or 40000 not in states:
        raise ValueError("30K and 40K LGO baselines are required")

    rescue_credit = torch.zeros(count)
    regression_credit = torch.zeros(count)
    clean_switch_count = torch.zeros(count, dtype=torch.int64)
    harmful_switch_count = torch.zeros(count, dtype=torch.int64)
    rescued_queries = 0
    regressed_queries = 0
    for query_index, record in graph_records.items():
        base30 = states[30000][query_index]
        base40 = states[40000][query_index]
        error30 = float(base30["translation_error_m"])
        error40 = float(base40["translation_error_m"])
        positions30 = torch.as_tensor(base30["positions"]).long()
        positions40 = torch.as_tensor(base40["positions"]).long()
        valid30 = torch.as_tensor(base30["valid"]).bool()
        valid40 = torch.as_tensor(base40["valid"]).bool()
        indices = torch.as_tensor(record["top_indices"]).long()
        changed = valid40 & (
            ~valid30 | (positions30 != positions40)
        )
        if not changed.any():
            continue
        chosen40 = indices[changed].gather(
            1, positions40[changed, None]
        ).squeeze(1)
        flags = torch.as_tensor(record["legal_flags"])[changed].gather(
            1, positions40[changed, None]
        ).squeeze(1)
        clean = (flags & 4) != 0
        harmful = ~clean
        improvement = float(
            np.clip(
                error30 - error40,
                -args.tail_clip_m,
                args.tail_clip_m,
            )
        )
        crossed_good = error30 > 0.05 and error40 <= 0.05
        crossed_bad = error30 <= 0.05 and error40 > 0.05
        if crossed_good:
            rescued_queries += 1
        if crossed_bad:
            regressed_queries += 1
        clean_rows = torch.unique(chosen40[clean])
        harmful_rows = torch.unique(chosen40[harmful])
        if clean_rows.numel():
            credit = max(improvement, 0.0) + (
                float(args.r5_weight) if crossed_good else 0.0
            )
            rescue_credit[clean_rows] += credit / clean_rows.numel()
            clean_switch_count[clean_rows] += 1
        if harmful_rows.numel():
            penalty = max(-improvement, 0.0) + (
                float(args.regression_weight) if crossed_bad else 0.0
            )
            regression_credit[harmful_rows] += (
                penalty / harmful_rows.numel()
            )
            harmful_switch_count[harmful_rows] += 1

    group_scores = {}
    for group in groups:
        rows = torch.as_tensor(group["rows"]).long()
        score = float(
            rescue_credit[rows].sum()
            - regression_credit[rows].sum()
        )
        score += 0.05 * np.log1p(
            int(group["gtclean_inlier_count"])
        )
        score -= 0.10 * np.log1p(
            int(group["harmful_consensus_count"])
        )
        group_scores[int(group["group_id"])] = score
    ranked = sorted(
        groups,
        key=lambda group: (
            group_scores[int(group["group_id"])]
            / max(np.sqrt(int(group["size"])), 1.0),
            group_scores[int(group["group_id"])],
            -int(group["group_id"]),
        ),
        reverse=True,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "schema": "lafgs_query_rescue_maps",
        "version": 1,
        "rescued_mapping_query_count_30k_to_40k": rescued_queries,
        "regressed_mapping_query_count_30k_to_40k": regressed_queries,
        "anchors_with_rescue_credit": int((rescue_credit > 0).sum()),
        "anchors_with_regression_credit": int(
            (regression_credit > 0).sum()
        ),
        "maps": {},
        "config": vars(args),
    }
    for budget in args.budgets:
        active = compact_mask.clone()
        active_count = int(active.sum())
        selected_groups = []
        for group in ranked:
            if group_scores[int(group["group_id"])] <= 0:
                continue
            rows = torch.as_tensor(group["rows"]).long()
            new_rows = rows[~active[rows]]
            if not new_rows.numel():
                continue
            if active_count + int(new_rows.numel()) > budget:
                continue
            active[new_rows] = True
            active_count += int(new_rows.numel())
            selected_groups.append(int(group["group_id"]))
            if active_count >= budget:
                break
        rows = torch.nonzero(active).reshape(-1)
        label = f"query_rescue_{int(rows.numel())}"
        metadata = {
            "schema": "lafgs_v5_active_map",
            "version": 1,
            "operation": "query_threshold_rescue",
            "requested_budget": budget,
            "selected_group_ids": selected_groups,
        }
        path = output_dir / f"{label}.pt"
        torch.save(_materialize(state, rows, metadata), path)
        report["maps"][label] = {
            **metadata,
            "path": str(path),
            "anchor_count": int(rows.numel()),
        }
    report_path = output_dir / "query_rescue_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
