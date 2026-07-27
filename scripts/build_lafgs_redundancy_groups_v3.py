#!/usr/bin/env python3
"""Build bounded multi-edge functional communities for Active Map V5."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from scripts.build_lafgs_redundancy_groups import (
    _anchor_bands,
    _support_edges,
    _support_slices,
)


class _BoundedDisjointSet:
    def __init__(self, size: int):
        self.parent = np.arange(size, dtype=np.int32)
        self.size = np.ones(size, dtype=np.int32)

    def find(self, value: int) -> int:
        root = value
        while self.parent[root] != root:
            root = int(self.parent[root])
        while self.parent[value] != value:
            parent = int(self.parent[value])
            self.parent[value] = root
            value = parent
        return root

    def union(self, left: int, right: int, maximum: int) -> bool:
        left, right = self.find(left), self.find(right)
        if left == right:
            return True
        if int(self.size[left] + self.size[right]) > maximum:
            return False
        if self.size[left] < self.size[right]:
            left, right = right, left
        self.parent[right] = left
        self.size[left] += self.size[right]
        return True


def _pair_intersections(
    packed: np.ndarray, band: np.ndarray
) -> Counter[tuple[int, int]]:
    by_keypoint = defaultdict(list)
    for value in packed:
        anchor = int(value >> np.uint64(32))
        keypoint = int(value & np.uint64(0xFFFFFFFF))
        by_keypoint[keypoint].append(anchor)
    intersections: Counter[tuple[int, int]] = Counter()
    for anchors in by_keypoint.values():
        unique = sorted(set(anchors))
        for left, right in combinations(unique, 2):
            if band[left] == band[right]:
                intersections[(left, right)] += 1
    return intersections


def _confusion_pairs(
    graph: dict,
    relevant: np.ndarray,
    band: np.ndarray,
    topk: int,
    margin: float,
) -> Counter[tuple[int, int]]:
    counts: Counter[tuple[int, int]] = Counter()
    relevant_t = torch.from_numpy(relevant)
    band_t = torch.from_numpy(band)
    for record in graph["records"]:
        indices = torch.as_tensor(record["top_indices"]).long()[:, :topk]
        scores = torch.as_tensor(record["top_scores"]).float()[:, :topk]
        eligible = relevant_t[indices]
        eligible &= (scores[:, :1] - scores) <= float(margin)
        for left_pos, right_pos in combinations(range(indices.shape[1]), 2):
            keep = eligible[:, left_pos] & eligible[:, right_pos]
            keep &= (
                band_t[indices[:, left_pos]]
                == band_t[indices[:, right_pos]]
            )
            left = indices[:, left_pos][keep]
            right = indices[:, right_pos][keep]
            low = torch.minimum(left, right)
            high = torch.maximum(left, right)
            packed = (low.to(torch.int64) << 32) | high.to(torch.int64)
            values, frequencies = torch.unique(
                packed, return_counts=True
            )
            for value, frequency in zip(
                values.tolist(), frequencies.tolist()
            ):
                counts[
                    (int(value >> 32), int(value & 0xFFFFFFFF))
                ] += int(frequency)
    return counts


def main() -> None:
    torch.set_num_threads(1)
    parser = argparse.ArgumentParser()
    parser.add_argument("--anchor-map", required=True)
    parser.add_argument("--function-graph", required=True)
    parser.add_argument("--budget-maps", nargs=4, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--support-jaccard", type=float, default=0.45)
    parser.add_argument("--support-containment", type=float, default=0.75)
    parser.add_argument("--minimum-shared-keypoints", type=int, default=2)
    parser.add_argument("--descriptor-cosine", type=float, default=0.72)
    parser.add_argument("--spatial-radius-m", type=float, default=0.75)
    parser.add_argument("--confusion-topk", type=int, default=4)
    parser.add_argument("--confusion-score-margin", type=float, default=0.04)
    parser.add_argument("--minimum-confusions", type=int, default=3)
    parser.add_argument("--max-legal-edges-per-keypoint", type=int, default=4)
    parser.add_argument("--max-group-size", type=int, default=64)
    args = parser.parse_args()

    state = torch.load(
        args.anchor_map, map_location="cpu", weights_only=False
    )
    graph = torch.load(
        args.function_graph, map_location="cpu", weights_only=False
    )
    count = int(torch.as_tensor(state["anchor_xyz"]).shape[0])
    band, selected = _anchor_bands(count, args.budget_maps)
    relevant = band > 0
    rows = np.flatnonzero(relevant)
    local_of = {int(row): local for local, row in enumerate(rows)}
    packed = _support_edges(
        graph, relevant, args.max_legal_edges_per_keypoint
    )
    offsets, support_values = _support_slices(packed, count)
    support_count = np.diff(offsets)
    support_pairs = _pair_intersections(packed, band)
    confusion_pairs = _confusion_pairs(
        graph,
        relevant,
        band,
        args.confusion_topk,
        args.confusion_score_margin,
    )

    source = torch.as_tensor(state["source_primitive_ids"]).numpy()
    track = torch.as_tensor(state["track_cluster_ids"]).numpy()
    xyz = torch.as_tensor(state["anchor_xyz"]).float().numpy()
    features = F.normalize(
        torch.as_tensor(state["anchor_features"]).float(), dim=1
    ).numpy()
    dsu = _BoundedDisjointSet(len(rows))
    accepted_edges = Counter()

    lineage_edges = []
    for label_name, labels in (("source", source), ("track", track)):
        members = defaultdict(list)
        for row in rows:
            label = int(labels[row])
            if label_name == "track" and label < 0:
                continue
            members[(int(band[row]), label)].append(int(row))
        for family in members.values():
            root = family[0]
            for row in family[1:]:
                lineage_edges.append((root, row, label_name))
    for left, right, edge_type in lineage_edges:
        if dsu.union(
            local_of[left], local_of[right], args.max_group_size
        ):
            accepted_edges[edge_type] += 1

    candidate_edges = []
    all_pairs = set(support_pairs) | set(confusion_pairs)
    for left, right in all_pairs:
        intersection = int(support_pairs.get((left, right), 0))
        union = int(
            support_count[left] + support_count[right] - intersection
        )
        jaccard = intersection / max(union, 1)
        containment = intersection / max(
            min(support_count[left], support_count[right]), 1
        )
        confusion = int(confusion_pairs.get((left, right), 0))
        cosine = float(features[left] @ features[right])
        distance = float(np.linalg.norm(xyz[left] - xyz[right]))
        support_ok = (
            intersection >= args.minimum_shared_keypoints
            and (
                jaccard >= args.support_jaccard
                or containment >= args.support_containment
            )
        )
        confusion_ok = confusion >= args.minimum_confusions
        appearance_ok = (
            cosine >= args.descriptor_cosine
            or distance <= args.spatial_radius_m
        )
        if not appearance_ok or not (support_ok or confusion_ok):
            continue
        score = (
            2.0 * jaccard
            + containment
            + np.log1p(confusion)
            + max(cosine, 0.0)
            + float(distance <= args.spatial_radius_m)
        )
        candidate_edges.append(
            (
                score,
                left,
                right,
                support_ok,
                confusion_ok,
                jaccard,
                containment,
                confusion,
            )
        )
    candidate_edges.sort(reverse=True)
    for (
        _,
        left,
        right,
        support_ok,
        confusion_ok,
        _,
        _,
        _,
    ) in candidate_edges:
        if dsu.union(
            local_of[left], local_of[right], args.max_group_size
        ):
            if support_ok:
                accepted_edges["support"] += 1
            if confusion_ok:
                accepted_edges["confusion"] += 1

    components = defaultdict(list)
    for row in rows:
        components[dsu.find(local_of[int(row)])].append(int(row))
    groups = []
    harmful = torch.as_tensor(
        graph["provenance_harmful_solver_inlier_count"]
    )
    clean = torch.as_tensor(
        graph["provenance_solver_inlier_gtclean_4px_count"]
    )
    opportunity = torch.as_tensor(
        graph["provenance_opportunity_count"]
    )
    for component in components.values():
        group_rows = torch.tensor(sorted(component), dtype=torch.int32)
        row_long = group_rows.long()
        source_values = source[row_long.numpy()].tolist()
        source_purity = max(Counter(source_values).values()) / len(
            source_values
        )
        track_values = [
            value
            for value in track[row_long.numpy()].tolist()
            if value >= 0
        ]
        track_purity = (
            max(Counter(track_values).values()) / len(track_values)
            if track_values
            else 0.0
        )
        groups.append(
            {
                "group_id": len(groups),
                "band": int(band[int(group_rows[0])]),
                "rows": group_rows,
                "size": int(group_rows.numel()),
                "support_edge_count": int(
                    support_count[row_long].sum()
                ),
                "supported_anchor_count": int(
                    (support_count[row_long] > 0).sum()
                ),
                "harmful_consensus_count": int(harmful[row_long].sum()),
                "gtclean_inlier_count": int(clean[row_long].sum()),
                "opportunity_count": int(opportunity[row_long].sum()),
                "source_purity": float(source_purity),
                "track_purity": float(track_purity),
            }
        )
    groups.sort(
        key=lambda group: (
            int(group["band"]),
            int(group["rows"][0]),
        )
    )
    for group_id, group in enumerate(groups):
        group["group_id"] = group_id

    sizes = np.asarray([group["size"] for group in groups])
    output = {
        "schema": "lafgs_redundancy_groups",
        "version": 3,
        "anchor_map": str(Path(args.anchor_map).resolve()),
        "function_graph": str(Path(args.function_graph).resolve()),
        "budget_maps": {
            str(budget): path
            for budget, path in zip(
                sorted(selected), sorted(args.budget_maps)
            )
        },
        "anchor_band": torch.from_numpy(band),
        "support_offsets": torch.from_numpy(offsets),
        "support_keypoint_ids": torch.from_numpy(
            support_values.astype(np.int64)
        ),
        "groups": groups,
        "accepted_edges": dict(accepted_edges),
        "config": vars(args),
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, output_path)
    print(
        json.dumps(
            {
                "group_count": len(groups),
                "singleton_count": int((sizes == 1).sum()),
                "multi_anchor_group_count": int((sizes > 1).sum()),
                "group_size_mean": float(sizes.mean()),
                "group_size_p90": float(np.quantile(sizes, 0.9)),
                "group_size_max": int(sizes.max()),
                "accepted_edges": dict(accepted_edges),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
