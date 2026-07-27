#!/usr/bin/env python3
"""Build grouped functional-pruning candidates across multiple map budgets."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch


PROFILES = {
    "balanced": {
        "inlier": 4.0,
        "unique": 5.0,
        "clean": 2.0,
        "sequence": 2.0,
        "image_bin": 0.4,
        "harm": 3.0,
        "wrong": 4.0,
        "parent_replace": 3.0,
        "group_rank": 1.5,
    },
    "coverage": {
        "inlier": 2.0,
        "unique": 8.0,
        "clean": 2.0,
        "sequence": 4.0,
        "image_bin": 0.8,
        "harm": 2.0,
        "wrong": 3.0,
        "parent_replace": 2.0,
        "group_rank": 2.0,
    },
    "harm_averse": {
        "inlier": 5.0,
        "unique": 4.0,
        "clean": 2.0,
        "sequence": 2.0,
        "image_bin": 0.3,
        "harm": 7.0,
        "wrong": 9.0,
        "parent_replace": 4.0,
        "group_rank": 1.5,
    },
}


def _popcount_rows(bits: np.ndarray) -> np.ndarray:
    return np.unpackbits(bits, axis=1).sum(axis=1)


def _support_signatures(bits: torch.Tensor) -> list[str]:
    array = bits.numpy()
    signatures = []
    for row in array:
        signatures.append(hashlib.blake2b(row.tobytes(), digest_size=8).hexdigest())
    return signatures


def _family_groups(source: np.ndarray, track: np.ndarray) -> list[str]:
    source_count = Counter(source.tolist())
    groups = []
    for index, (source_id, track_id) in enumerate(zip(source, track)):
        if source_count[int(source_id)] > 1:
            groups.append(f"source:{int(source_id)}")
        elif int(track_id) >= 0:
            groups.append(f"track:{int(track_id)}")
        else:
            groups.append(f"single:{index}")
    return groups


def _parent_replacement(
    source: np.ndarray,
    anchor_type: np.ndarray,
    support_bits: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, list[dict]]:
    groups = defaultdict(list)
    for index, source_id in enumerate(source.tolist()):
        groups[int(source_id)].append(index)
    replaceable = np.zeros(source.shape[0], dtype=bool)
    diagnostics = []
    for source_id, rows in groups.items():
        parents = [row for row in rows if anchor_type[row] == 0]
        children = [row for row in rows if anchor_type[row] != 0]
        if not parents or not children:
            continue
        child_union = np.bitwise_or.reduce(support_bits[children], axis=0)
        for parent in parents:
            parent_bits = support_bits[parent]
            parent_count = int(np.unpackbits(parent_bits).sum())
            covered = int(
                np.unpackbits(parent_bits & child_union).sum()
            )
            ratio = covered / max(parent_count, 1)
            is_replaceable = parent_count == 0 or ratio >= threshold
            replaceable[parent] = is_replaceable
            diagnostics.append(
                {
                    "source_primitive_id": source_id,
                    "parent_row": parent,
                    "child_count": len(children),
                    "parent_query_support": parent_count,
                    "children_cover_parent": covered,
                    "coverage_ratio": ratio,
                    "replaceable": is_replaceable,
                }
            )
    return replaceable, diagnostics


def _group_rank_penalty(
    base_score: np.ndarray, groups: list[str]
) -> np.ndarray:
    members = defaultdict(list)
    for index, group in enumerate(groups):
        members[group].append(index)
    rank = np.zeros(base_score.shape[0], dtype=np.float64)
    for rows in members.values():
        ordered = sorted(rows, key=lambda row: (-base_score[row], row))
        for position, row in enumerate(ordered):
            rank[row] = np.log1p(position)
    return rank


def _materialize(state: dict, rows: torch.Tensor, metadata: dict) -> dict:
    count = int(state["anchor_xyz"].shape[0])
    output = {}
    for key, value in state.items():
        if torch.is_tensor(value) and value.ndim and value.shape[0] == count:
            output[key] = value[rows]
        else:
            output[key] = value
    output["anchor_ids"] = torch.arange(rows.numel(), dtype=torch.long)
    output["canonical_anchor_count"] = int(rows.numel())
    anchor_type = torch.as_tensor(output["anchor_type"])
    output["base_anchor_count"] = int((anchor_type == 0).sum())
    output["micro_anchor_count"] = int((anchor_type != 0).sum())
    output["requested_micro_anchor_budget"] = output["micro_anchor_count"]
    output["functional_pruning"] = metadata
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--anchor-map", required=True)
    parser.add_argument("--function-stats", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--budgets", type=int, nargs="+", default=[45000, 40000, 35000, 30000])
    parser.add_argument(
        "--profiles",
        nargs="+",
        choices=sorted(PROFILES),
        default=["balanced", "coverage", "harm_averse"],
    )
    parser.add_argument("--parent-coverage-threshold", type=float, default=0.9)
    args = parser.parse_args()

    state = torch.load(args.anchor_map, map_location="cpu", weights_only=False)
    stats = torch.load(
        args.function_stats, map_location="cpu", weights_only=False
    )
    count = int(state["anchor_xyz"].shape[0])
    if int(stats["anchor_count"]) != count:
        raise ValueError("anchor map and functional statistics do not align")

    source = torch.as_tensor(stats["source_primitive_ids"]).numpy()
    track = torch.as_tensor(stats["track_cluster_ids"]).numpy()
    anchor_type = torch.as_tensor(stats["anchor_type"]).numpy()
    support_bits = torch.as_tensor(stats["query_support_bits"]).numpy()
    support_query_count = _popcount_rows(support_bits)
    sequence_count = (
        torch.as_tensor(stats["sequence_clean_support"]) > 0
    ).sum(dim=1).numpy()
    parent_replaceable, parent_diagnostics = _parent_replacement(
        source,
        anchor_type,
        support_bits,
        args.parent_coverage_threshold,
    )
    family_groups = _family_groups(source, track)
    support_signatures = _support_signatures(
        torch.as_tensor(stats["query_support_bits"])
    )

    raw = {
        key: torch.as_tensor(stats[key]).numpy().astype(np.float64)
        for key in (
            "winner_count",
            "clean_winner_count",
            "harm_winner_count",
            "inlier_count",
            "clean_topk_query_count",
            "unique_topk_count",
            "image_bin_support_count",
        )
    }
    wrong = (
        (raw["winner_count"] > 0)
        & (raw["harm_winner_count"] > raw["clean_winner_count"])
    ).astype(np.float64)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "schema": "lafgs_functional_pruning_curve",
        "version": 1,
        "anchor_count": count,
        "budgets": args.budgets,
        "profiles": {},
        "class_counts": {
            "never_winner": int((raw["winner_count"] == 0).sum()),
            "used_but_incorrect": int(
                (
                    (raw["winner_count"] > 0)
                    & (raw["clean_winner_count"] == 0)
                ).sum()
            ),
            "majority_harmful": int(wrong.sum()),
            "zero_unique_support": int(
                (raw["unique_topk_count"] == 0).sum()
            ),
            "inlier_supported": int(
                (raw["inlier_count"] > 0).sum()
            ),
            "parent_replaceable": int(parent_replaceable.sum()),
            "query_supported": int((support_query_count > 0).sum()),
        },
        "parent_diagnostics": parent_diagnostics,
    }

    for profile_name in args.profiles:
        weights = PROFILES[profile_name]
        positive = (
            weights["inlier"] * np.log1p(raw["inlier_count"])
            + weights["unique"] * np.log1p(raw["unique_topk_count"])
            + weights["clean"] * np.log1p(raw["clean_topk_query_count"])
            + weights["sequence"] * np.log1p(sequence_count)
            + weights["image_bin"]
            * np.log1p(raw["image_bin_support_count"])
        )
        negative = (
            weights["harm"] * np.log1p(raw["harm_winner_count"])
            + weights["wrong"] * wrong
            + weights["parent_replace"] * parent_replaceable.astype(float)
        )
        preliminary = positive - negative
        family_rank = _group_rank_penalty(preliminary, family_groups)
        support_rank = _group_rank_penalty(
            preliminary, support_signatures
        )
        group_rank = family_rank + support_rank
        score = preliminary - weights["group_rank"] * group_rank
        order = np.lexsort((np.arange(count), -score))
        profile_report = {
            "weights": weights,
            "score_quantiles": {
                str(q): float(np.quantile(score, q))
                for q in (0.0, 0.1, 0.5, 0.9, 1.0)
            },
            "maps": {},
        }
        profile_dir = output_dir / profile_name
        profile_dir.mkdir(parents=True, exist_ok=True)
        for budget in sorted(set(args.budgets), reverse=True):
            budget = min(max(int(budget), 1), count)
            selected_np = np.sort(order[:budget])
            selected = torch.as_tensor(selected_np, dtype=torch.long)
            selected_mask = np.zeros(count, dtype=bool)
            selected_mask[selected_np] = True
            metadata = {
                "schema": "lafgs_functional_pruned_map",
                "version": 1,
                "profile": profile_name,
                "budget": budget,
                "source_anchor_map": str(Path(args.anchor_map).resolve()),
                "function_stats": str(Path(args.function_stats).resolve()),
                "selected_source_rows": selected,
                "weights": weights,
            }
            path = profile_dir / f"functional_{budget:05d}.pt"
            torch.save(_materialize(state, selected, metadata), path)
            profile_report["maps"][str(budget)] = {
                "path": str(path.resolve()),
                "base_count": int((anchor_type[selected_np] == 0).sum()),
                "micro_count": int((anchor_type[selected_np] != 0).sum()),
                "retained_inlier_support_fraction": float(
                    raw["inlier_count"][selected_np].sum()
                    / max(raw["inlier_count"].sum(), 1.0)
                ),
                "retained_unique_support_fraction": float(
                    raw["unique_topk_count"][selected_np].sum()
                    / max(raw["unique_topk_count"].sum(), 1.0)
                ),
                "retained_harm_fraction": float(
                    raw["harm_winner_count"][selected_np].sum()
                    / max(raw["harm_winner_count"].sum(), 1.0)
                ),
                "retained_parent_replaceable": int(
                    parent_replaceable[selected_np].sum()
                ),
                "retired_used_anchor_count": int(
                    (
                        (~selected_mask)
                        & (raw["winner_count"] > 0)
                    ).sum()
                ),
            }
        report["profiles"][profile_name] = profile_report

    (output_dir / "functional_pruning_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["class_counts"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
