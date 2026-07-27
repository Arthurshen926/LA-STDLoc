#!/usr/bin/env python
"""Build cross-fold-stable query-group rescue and retire active maps."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.cluster import KMeans

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.build_lafgs_lgo_candidate_maps import _materialize


def _selected_rows(state, count):
    rows = state.get("functional_pruning", {}).get("selected_source_rows")
    if rows is None:
        return torch.arange(count, dtype=torch.long)
    return torch.as_tensor(rows, dtype=torch.long)


def _pose_features(cache_queries, names, diagnostics):
    centers = []
    directions = []
    difficulty = []
    for name, diag in zip(names, diagnostics):
        pose = torch.as_tensor(cache_queries[name]["pose_w2c"]).double().numpy()
        rotation = pose[:3, :3]
        centers.append(-(rotation.T @ pose[:3, 3]))
        directions.append(rotation.T @ np.asarray([0.0, 0.0, 1.0]))
        inliers = float(diag["solver_inlier_count"])
        harmful = float(diag["harmful_solver_inlier_count"])
        difficulty.append(
            [
                np.log1p(inliers),
                harmful / max(inliers, 1.0),
                float(diag["legal_top1_2px_rate"]),
                float(diag["legal_top64_4px_rate"]),
            ]
        )
    features = np.concatenate(
        [
            np.asarray(centers),
            np.asarray(directions) * 2.0,
            np.asarray(difficulty),
        ],
        axis=1,
    )
    mean = features.mean(axis=0)
    scale = features.std(axis=0)
    return (features - mean) / np.maximum(scale, 1e-6)


def _query_group_block_folds(names, groups, fold_count):
    folds = np.zeros(len(names), dtype=np.int64)
    by_group = {}
    for index, (name, group) in enumerate(zip(names, groups)):
        by_group.setdefault(int(group), []).append((name, index))
    for rows in by_group.values():
        rows.sort()
        for rank, (_, index) in enumerate(rows):
            folds[index] = min(
                int(rank * fold_count / max(len(rows), 1)), fold_count - 1
            )
    return folds


def _aggregate(graph, groups, folds, group_count, fold_count):
    anchor_count = int(graph["anchor_count"])
    clean = torch.zeros(
        (anchor_count, group_count, fold_count), dtype=torch.int32
    )
    harmful = torch.zeros_like(clean)
    opportunity = torch.zeros_like(clean)
    group_query_count = torch.zeros(group_count, dtype=torch.int32)
    for query_index, record in enumerate(graph["records"]):
        group = int(groups[query_index])
        fold = int(folds[query_index])
        group_query_count[group] += 1
        top1 = torch.as_tensor(record["top_indices"]).long()[:, 0]
        flags = torch.as_tensor(record["legal_flags"]).to(torch.uint8)[:, 0]
        inlier = torch.as_tensor(record["solver_inlier"]).bool()
        legal2 = (flags & 2) != 0
        legal4 = (flags & 4) != 0
        ones = torch.ones_like(top1, dtype=torch.int32)
        opportunity[:, group, fold].index_add_(0, top1, ones)
        if bool((inlier & legal2).any()):
            ids = top1[inlier & legal2]
            clean[:, group, fold].index_add_(
                0, ids, torch.ones_like(ids, dtype=torch.int32)
            )
        if bool((inlier & ~legal4).any()):
            ids = top1[inlier & ~legal4]
            harmful[:, group, fold].index_add_(
                0, ids, torch.ones_like(ids, dtype=torch.int32)
            )
    return clean, harmful, opportunity, group_query_count


def _first_two_active_positions(indices, active_mask):
    active = active_mask[indices]
    positions = torch.arange(indices.shape[1])[None].expand_as(indices)
    sentinel = torch.full_like(positions, indices.shape[1])
    ranked = torch.where(active, positions, sentinel)
    first_two = torch.topk(
        ranked, k=2, dim=1, largest=False, sorted=True
    ).values
    return first_two[:, 0], first_two[:, 1]


def _aggregate_assignment_transitions(
    graph,
    groups,
    folds,
    group_count,
    fold_count,
    compact_mask,
    balanced_mask,
):
    anchor_count = int(graph["anchor_count"])
    rescue_clean = torch.zeros(
        (anchor_count, group_count, fold_count), dtype=torch.int32
    )
    rescue_harm = torch.zeros_like(rescue_clean)
    rescue_opportunity = torch.zeros_like(rescue_clean)
    retire_clean = torch.zeros_like(rescue_clean)
    retire_harm = torch.zeros_like(rescue_clean)
    retire_opportunity = torch.zeros_like(rescue_clean)
    group_query_count = torch.zeros(group_count, dtype=torch.int32)
    for query_index, record in enumerate(graph["records"]):
        group = int(groups[query_index])
        fold = int(folds[query_index])
        group_query_count[group] += 1
        indices = torch.as_tensor(record["top_indices"]).long()
        flags = torch.as_tensor(record["legal_flags"]).to(torch.uint8)
        legal2 = (flags & 2) != 0
        legal4 = (flags & 4) != 0
        positions = torch.arange(indices.shape[1])[None].expand_as(indices)

        compact_first, _ = _first_two_active_positions(
            indices, compact_mask
        )
        compact_valid = compact_first < indices.shape[1]
        compact_row = torch.arange(indices.shape[0])[compact_valid]
        compact_position = compact_first[compact_valid]
        compact_legal2 = legal2[
            compact_row, compact_position
        ][:, None]
        compact_legal4 = legal4[
            compact_row, compact_position
        ][:, None]
        candidate_mask = (
            positions[compact_valid] < compact_position[:, None]
        ) & ~compact_mask[indices[compact_valid]]
        candidate_ids = indices[compact_valid][candidate_mask]
        candidate_clean = (
            legal2[compact_valid]
            & ~compact_legal2
            & candidate_mask
        )[candidate_mask]
        candidate_harm = (
            ~legal4[compact_valid]
            & compact_legal4
            & candidate_mask
        )[candidate_mask]
        ones = torch.ones_like(candidate_ids, dtype=torch.int32)
        rescue_opportunity[:, group, fold].index_add_(
            0, candidate_ids, ones
        )
        rescue_clean[:, group, fold].index_add_(
            0, candidate_ids, candidate_clean.to(torch.int32)
        )
        rescue_harm[:, group, fold].index_add_(
            0, candidate_ids, candidate_harm.to(torch.int32)
        )

        first, second = _first_two_active_positions(
            indices, balanced_mask
        )
        valid = second < indices.shape[1]
        row = torch.arange(indices.shape[0])[valid]
        old_position = first[valid]
        next_position = second[valid]
        old_ids = indices[row, old_position]
        old_legal2 = legal2[row, old_position]
        old_legal4 = legal4[row, old_position]
        next_legal2 = legal2[row, next_position]
        next_legal4 = legal4[row, next_position]
        retire_opportunity[:, group, fold].index_add_(
            0, old_ids, torch.ones_like(old_ids, dtype=torch.int32)
        )
        retire_clean[:, group, fold].index_add_(
            0,
            old_ids,
            ((~old_legal4) & next_legal2).to(torch.int32),
        )
        retire_harm[:, group, fold].index_add_(
            0,
            old_ids,
            (old_legal2 & ~next_legal4).to(torch.int32),
        )
    return (
        rescue_clean,
        rescue_harm,
        rescue_opportunity,
        retire_clean,
        retire_harm,
        retire_opportunity,
        group_query_count,
    )


def _scores(clean, harmful, opportunity, group_query_count, harm_weight):
    evidence = clean.float() - float(harm_weight) * harmful.float()
    fold_sign = torch.sign(evidence)
    stable_positive = (fold_sign > 0).sum(dim=2) >= 2
    stable_negative = (fold_sign < 0).sum(dim=2) >= 2
    group_evidence = evidence.sum(dim=2)
    group_opportunity = opportunity.sum(dim=2).float().clamp_min(1.0)
    rate = group_evidence / group_opportunity.sqrt()
    difficulty_weight = (
        group_query_count.float().clamp_min(1.0).rsqrt()
    )
    rescue = (
        torch.relu(rate)
        * stable_positive.float()
        * difficulty_weight[None]
    ).sum(dim=1)
    rescue -= (
        torch.relu(-rate) * stable_negative.float()
    ).amax(dim=1)
    retire = (
        torch.relu(-rate) * stable_negative.float()
    ).sum(dim=1)
    retire -= (
        torch.relu(rate) * stable_positive.float()
    ).amax(dim=1)
    return rescue, retire, stable_positive, stable_negative


def _save_map(state, rows, path, metadata):
    rows = torch.as_tensor(rows, dtype=torch.long)
    torch.save(_materialize(state, rows, metadata), path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor_map", required=True)
    parser.add_argument("--compact_map", required=True)
    parser.add_argument("--balanced_map", required=True)
    parser.add_argument("--function_graph", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--group_count", type=int, default=24)
    parser.add_argument("--fold_count", type=int, default=3)
    parser.add_argument("--harm_weight", type=float, default=2.0)
    parser.add_argument(
        "--budgets", nargs="+", type=int, default=[34000, 36000, 38000]
    )
    args = parser.parse_args()
    state = torch.load(args.anchor_map, map_location="cpu", weights_only=False)
    compact = torch.load(
        args.compact_map, map_location="cpu", weights_only=False
    )
    balanced = torch.load(
        args.balanced_map, map_location="cpu", weights_only=False
    )
    graph = torch.load(
        args.function_graph, map_location="cpu", weights_only=False
    )
    cache = torch.load(
        graph["query_cache"], map_location="cpu", weights_only=False
    )["queries"]
    features = _pose_features(
        cache, graph["query_names"], graph["query_diagnostics"]
    )
    del cache
    groups = KMeans(
        n_clusters=int(args.group_count),
        random_state=2026,
        n_init=20,
    ).fit_predict(features)
    folds = _query_group_block_folds(
        graph["query_names"], groups, int(args.fold_count)
    )
    anchor_count = int(graph["anchor_count"])
    compact_rows = _selected_rows(compact, anchor_count)
    balanced_rows = _selected_rows(balanced, anchor_count)
    compact_mask = torch.zeros(anchor_count, dtype=torch.bool)
    balanced_mask = torch.zeros(anchor_count, dtype=torch.bool)
    compact_mask[compact_rows] = True
    balanced_mask[balanced_rows] = True
    (
        rescue_clean,
        rescue_harm,
        rescue_opportunity,
        retire_clean,
        retire_harm,
        retire_opportunity,
        group_query_count,
    ) = _aggregate_assignment_transitions(
        graph,
        groups,
        folds,
        int(args.group_count),
        int(args.fold_count),
        compact_mask,
        balanced_mask,
    )
    rescue, _, rescue_stable_positive, _ = _scores(
        rescue_clean,
        rescue_harm,
        rescue_opportunity,
        group_query_count,
        args.harm_weight,
    )
    retire, _, retire_stable_positive, _ = _scores(
        retire_clean,
        retire_harm,
        retire_opportunity,
        group_query_count,
        args.harm_weight,
    )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "schema": "lafgs_v6_query_group_maps",
        "version": 1,
        "group_count": int(args.group_count),
        "fold_count": int(args.fold_count),
        "harm_weight": float(args.harm_weight),
        "group_query_count": group_query_count.tolist(),
        "maps": {},
    }
    for budget in args.budgets:
        rescue_order = torch.argsort(rescue, descending=True)
        add = rescue_order[
            ~compact_mask[rescue_order] & (rescue[rescue_order] > 0)
        ][: max(int(budget) - int(compact_mask.sum()), 0)]
        rescue_mask = compact_mask.clone()
        rescue_mask[add] = True
        rescue_rows = torch.nonzero(rescue_mask).reshape(-1)
        label = f"qgroup_rescue_{int(rescue_rows.numel())}"
        metadata = {
            "schema": "lafgs_v6_query_group_active_map",
            "operation": "cross_fold_query_group_rescue",
            "requested_budget": int(budget),
            "selected_source_rows": rescue_rows,
        }
        path = output / f"{label}.pt"
        _save_map(state, rescue_rows, path, metadata)
        report["maps"][label] = {
            "path": str(path),
            "anchor_count": int(rescue_rows.numel()),
            "stable_positive_added": int(
                rescue_stable_positive[add].any(dim=1).sum()
            ),
        }

        removable = balanced_mask & ~compact_mask & (retire > 0)
        remove_order = torch.argsort(retire, descending=True)
        remove = remove_order[removable[remove_order]][
            : max(int(balanced_mask.sum()) - int(budget), 0)
        ]
        retire_mask = balanced_mask.clone()
        retire_mask[remove] = False
        retire_rows = torch.nonzero(retire_mask).reshape(-1)
        label = f"qgroup_retire_{int(retire_rows.numel())}"
        metadata = {
            "schema": "lafgs_v6_query_group_active_map",
            "operation": "cross_fold_query_group_retire",
            "requested_budget": int(budget),
            "selected_source_rows": retire_rows,
        }
        path = output / f"{label}.pt"
        _save_map(state, retire_rows, path, metadata)
        report["maps"][label] = {
            "path": str(path),
            "anchor_count": int(retire_rows.numel()),
            "stable_positive_removals": int(
                retire_stable_positive[remove].any(dim=1).sum()
            ),
        }
    (output / "query_group_report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
