#!/usr/bin/env python3
"""Materialize provenance-aware rescue, retire and swap map proposals."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from scripts.build_lafgs_lgo_candidate_maps import _materialize


PROFILES = {
    "balanced": {
        "clean": 2.0,
        "harm": 2.5,
        "support": 0.25,
        "purity": 0.15,
    },
    "r5_rescue": {
        "clean": 3.0,
        "harm": 2.0,
        "support": 0.35,
        "purity": 0.10,
    },
    "consensus_clean": {
        "clean": 1.5,
        "harm": 4.0,
        "support": 0.15,
        "purity": 0.20,
    },
}


def _selected_rows(state: dict) -> torch.Tensor:
    metadata = state.get("functional_pruning", {})
    if "selected_source_rows" in metadata:
        return torch.as_tensor(metadata["selected_source_rows"]).long()
    return torch.arange(
        int(torch.as_tensor(state["anchor_xyz"]).shape[0]),
        dtype=torch.long,
    )


def _score(group: dict, profile: dict) -> float:
    opportunity = max(int(group["opportunity_count"]), 1)
    clean = int(group["gtclean_inlier_count"]) / opportunity
    harm = int(group["harmful_consensus_count"]) / opportunity
    support = np.log1p(int(group["support_edge_count"])) / max(
        np.sqrt(int(group["size"])), 1.0
    )
    purity = 0.5 * (
        float(group.get("source_purity", 0.0))
        + float(group.get("track_purity", 0.0))
    )
    return (
        profile["clean"] * clean
        - profile["harm"] * harm
        + profile["support"] * support
        + profile["purity"] * purity
    )


def _grow(
    active: torch.Tensor,
    groups: list[dict],
    target: int,
    scores: dict[int, float],
) -> tuple[torch.Tensor, list[int]]:
    selected = []
    active_count = int(active.sum())
    order = sorted(
        groups,
        key=lambda group: (
            scores[int(group["group_id"])]
            / max(np.sqrt(int(group["size"])), 1.0),
            scores[int(group["group_id"])],
            -int(group["group_id"]),
        ),
        reverse=True,
    )
    for group in order:
        rows = torch.as_tensor(group["rows"]).long()
        new_rows = rows[~active[rows]]
        if new_rows.numel() == 0:
            continue
        if active_count + int(new_rows.numel()) > target:
            continue
        active[new_rows] = True
        active_count += int(new_rows.numel())
        selected.append(int(group["group_id"]))
        if active_count >= target:
            break
    return active, selected


def _shrink(
    active: torch.Tensor,
    protected: torch.Tensor,
    groups: list[dict],
    target: int,
    scores: dict[int, float],
) -> tuple[torch.Tensor, list[int]]:
    retired = []
    active_count = int(active.sum())
    order = sorted(
        groups,
        key=lambda group: (
            scores[int(group["group_id"])]
            / max(np.sqrt(int(group["size"])), 1.0),
            scores[int(group["group_id"])],
            int(group["group_id"]),
        ),
    )
    for group in order:
        rows = torch.as_tensor(group["rows"]).long()
        removable = rows[active[rows] & ~protected[rows]]
        if removable.numel() == 0:
            continue
        if active_count - int(removable.numel()) < target:
            continue
        active[removable] = False
        active_count -= int(removable.numel())
        retired.append(int(group["group_id"]))
        if active_count <= target:
            break
    return active, retired


def main() -> None:
    torch.set_num_threads(1)
    parser = argparse.ArgumentParser()
    parser.add_argument("--anchor-map", required=True)
    parser.add_argument("--compact-32k-map", required=True)
    parser.add_argument("--balanced-40k-map", required=True)
    parser.add_argument("--groups", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--budgets", nargs="+", type=int, default=[34000, 36000, 38000]
    )
    parser.add_argument("--swap-fraction", type=float, default=0.04)
    args = parser.parse_args()
    state = torch.load(
        args.anchor_map, map_location="cpu", weights_only=False
    )
    compact = torch.load(
        args.compact_32k_map, map_location="cpu", weights_only=False
    )
    balanced = torch.load(
        args.balanced_40k_map, map_location="cpu", weights_only=False
    )
    group_payload = torch.load(
        args.groups, map_location="cpu", weights_only=False
    )
    groups = group_payload["groups"]
    count = int(torch.as_tensor(state["anchor_xyz"]).shape[0])
    compact_rows = _selected_rows(compact)
    balanced_rows = _selected_rows(balanced)
    compact_mask = torch.zeros(count, dtype=torch.bool)
    balanced_mask = torch.zeros(count, dtype=torch.bool)
    compact_mask[compact_rows] = True
    balanced_mask[balanced_rows] = True
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {"schema": "lafgs_v5_bidirectional_maps", "maps": {}}

    for profile_name, weights in PROFILES.items():
        scores = {
            int(group["group_id"]): _score(group, weights)
            for group in groups
        }
        for budget in args.budgets:
            rescue, added = _grow(
                compact_mask.clone(), groups, budget, scores
            )
            rescue_rows = torch.nonzero(rescue).reshape(-1)
            label = f"rescue_{profile_name}_{int(rescue_rows.numel())}"
            metadata = {
                "schema": "lafgs_v5_active_map",
                "version": 1,
                "operation": "32k_core_rescue",
                "profile": profile_name,
                "requested_budget": budget,
                "selected_group_ids": added,
            }
            path = output_dir / f"{label}.pt"
            if not path.exists():
                torch.save(
                    _materialize(state, rescue_rows, metadata), path
                )
            report["maps"][label] = {
                **metadata,
                "path": str(path),
                "anchor_count": int(rescue_rows.numel()),
            }

            protected = compact_mask.clone()
            retire, removed = _shrink(
                balanced_mask.clone(),
                protected,
                groups,
                budget,
                scores,
            )
            retire_rows = torch.nonzero(retire).reshape(-1)
            label = f"retire_{profile_name}_{int(retire_rows.numel())}"
            metadata = {
                "schema": "lafgs_v5_active_map",
                "version": 1,
                "operation": "40k_harmful_retire",
                "profile": profile_name,
                "requested_budget": budget,
                "retired_group_ids": removed,
            }
            path = output_dir / f"{label}.pt"
            if not path.exists():
                torch.save(
                    _materialize(state, retire_rows, metadata), path
                )
            report["maps"][label] = {
                **metadata,
                "path": str(path),
                "anchor_count": int(retire_rows.numel()),
            }

            swap_count = max(
                int(round(budget * float(args.swap_fraction))), 1
            )
            swap, removed = _shrink(
                rescue.clone(),
                compact_mask,
                groups,
                max(int(rescue.sum()) - swap_count, int(compact_mask.sum())),
                scores,
            )
            excluded = {
                group_id for group_id in removed
            }
            add_groups = [
                group
                for group in groups
                if int(group["group_id"]) not in excluded
            ]
            swap, swapped_in = _grow(
                swap, add_groups, budget, scores
            )
            swap_rows = torch.nonzero(swap).reshape(-1)
            label = f"swap_{profile_name}_{int(swap_rows.numel())}"
            metadata = {
                "schema": "lafgs_v5_active_map",
                "version": 1,
                "operation": "rescue_retire_swap",
                "profile": profile_name,
                "requested_budget": budget,
                "retired_group_ids": removed,
                "rescue_group_ids": swapped_in,
            }
            path = output_dir / f"{label}.pt"
            if not path.exists():
                torch.save(
                    _materialize(state, swap_rows, metadata), path
                )
            report["maps"][label] = {
                **metadata,
                "path": str(path),
                "anchor_count": int(swap_rows.numel()),
            }
    report_path = output_dir / "candidate_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(f"wrote {len(report['maps'])} candidates -> {report_path}")


if __name__ == "__main__":
    main()
