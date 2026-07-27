#!/usr/bin/env python3
"""Construct keypoint-support redundancy groups for Active Map V4."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


class _DisjointSet:
    def __init__(self, size: int):
        self.parent = np.arange(size, dtype=np.int32)
        self.rank = np.zeros(size, dtype=np.int8)

    def find(self, value: int) -> int:
        root = value
        while self.parent[root] != root:
            root = int(self.parent[root])
        while self.parent[value] != value:
            parent = int(self.parent[value])
            self.parent[value] = root
            value = parent
        return root

    def union(self, left: int, right: int) -> None:
        left = self.find(left)
        right = self.find(right)
        if left == right:
            return
        if self.rank[left] < self.rank[right]:
            left, right = right, left
        self.parent[right] = left
        if self.rank[left] == self.rank[right]:
            self.rank[left] += 1


def _selected_rows(path: str) -> np.ndarray:
    state = torch.load(path, map_location="cpu", weights_only=False)
    metadata = state.get("functional_pruning", {})
    if "selected_source_rows" not in metadata:
        raise ValueError(f"{path} is not a functional-pruning map")
    return torch.as_tensor(
        metadata["selected_source_rows"]
    ).long().numpy()


def _anchor_bands(count: int, budget_maps: list[str]) -> tuple[np.ndarray, dict]:
    selected = {
        int(Path(path).stem.rsplit("_", 1)[-1]): _selected_rows(path)
        for path in budget_maps
    }
    required = {30000, 35000, 40000, 45000}
    if set(selected) != required:
        raise ValueError("budget maps must provide 30K, 35K, 40K and 45K")
    masks = {}
    for budget, rows in selected.items():
        mask = np.zeros(count, dtype=bool)
        mask[rows] = True
        masks[budget] = mask
    if not (
        np.all(masks[30000] <= masks[35000])
        and np.all(masks[35000] <= masks[40000])
        and np.all(masks[40000] <= masks[45000])
    ):
        raise ValueError("budget maps are not nested")
    band = np.full(count, -1, dtype=np.int8)
    band[masks[30000]] = 0
    band[masks[35000] & ~masks[30000]] = 1
    band[masks[40000] & ~masks[35000]] = 2
    band[masks[45000] & ~masks[40000]] = 3
    band[~masks[45000]] = 4
    return band, selected


def _support_edges(graph: dict, relevant: np.ndarray, max_edges: int):
    packed = []
    relevant_t = torch.from_numpy(relevant)
    for record in graph["records"]:
        indices = torch.as_tensor(record["top_indices"]).long()
        flags = torch.as_tensor(record["legal_flags"])
        legal = (flags & 4) != 0
        legal &= relevant_t[indices]
        if max_edges > 0:
            positions = torch.arange(indices.shape[1])[None]
            legal_rank = torch.where(
                legal, positions, torch.full_like(positions, indices.shape[1])
            )
            keep_position = torch.topk(
                legal_rank, k=min(max_edges, indices.shape[1]),
                dim=1, largest=False
            ).indices
            keep = torch.zeros_like(legal)
            keep.scatter_(1, keep_position, True)
            legal &= keep
        if not legal.any():
            continue
        keypoint_ids = torch.as_tensor(
            record["global_keypoint_ids"]
        ).long()[:, None].expand_as(indices)
        packed.append(
            (indices[legal].to(torch.int64) << 32)
            | (keypoint_ids[legal] & 0xFFFFFFFF)
        )
    if not packed:
        return np.empty(0, dtype=np.uint64)
    values = torch.unique(torch.cat(packed)).numpy().astype(np.uint64)
    return np.sort(values)


def _support_slices(
    packed: np.ndarray, anchor_count: int
) -> tuple[np.ndarray, np.ndarray]:
    anchors = (packed >> np.uint64(32)).astype(np.int64)
    counts = np.bincount(anchors, minlength=anchor_count)
    offsets = np.concatenate(([0], np.cumsum(counts)))
    return offsets, (packed & np.uint64(0xFFFFFFFF)).astype(np.uint32)


def _jaccard(
    offsets: np.ndarray, values: np.ndarray, left: int, right: int
) -> float:
    a = values[offsets[left] : offsets[left + 1]]
    b = values[offsets[right] : offsets[right + 1]]
    if not len(a) and not len(b):
        return 1.0
    intersection = np.intersect1d(a, b, assume_unique=True).size
    return intersection / max(len(a) + len(b) - intersection, 1)


def _minhash(
    offsets: np.ndarray, values: np.ndarray, rows: np.ndarray
) -> np.ndarray:
    prime = np.uint64(4294967311)
    coefficients = (
        (np.uint64(2654435761), np.uint64(1013904223)),
        (np.uint64(2246822519), np.uint64(3266489917)),
        (np.uint64(3266489917), np.uint64(668265263)),
        (np.uint64(374761393), np.uint64(1274126177)),
    )
    signatures = np.full(
        (len(rows), len(coefficients)),
        np.iinfo(np.uint64).max,
        dtype=np.uint64,
    )
    for local, row in enumerate(rows):
        support = values[offsets[row] : offsets[row + 1]].astype(
            np.uint64
        )
        if not len(support):
            continue
        for column, (a, b) in enumerate(coefficients):
            signatures[local, column] = np.min(
                (a * support + b) % prime
            )
    return signatures


def _split_component(
    rows: list[int], support_count: np.ndarray, max_size: int
) -> list[list[int]]:
    ordered = sorted(rows, key=lambda row: (-support_count[row], row))
    return [
        ordered[start : start + max_size]
        for start in range(0, len(ordered), max_size)
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--anchor-map", required=True)
    parser.add_argument("--function-graph", required=True)
    parser.add_argument("--budget-maps", nargs=4, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--support-jaccard", type=float, default=0.6)
    parser.add_argument("--descriptor-cosine", type=float, default=0.75)
    parser.add_argument("--spatial-radius-m", type=float, default=0.5)
    parser.add_argument("--max-legal-edges-per-keypoint", type=int, default=8)
    parser.add_argument("--max-group-size", type=int, default=64)
    args = parser.parse_args()

    state = torch.load(
        args.anchor_map, map_location="cpu", weights_only=False
    )
    graph = torch.load(
        args.function_graph, map_location="cpu", weights_only=False
    )
    count = int(state["anchor_xyz"].shape[0])
    if int(graph["anchor_count"]) != count:
        raise ValueError("map and function graph do not align")
    band, selected = _anchor_bands(count, args.budget_maps)
    relevant = band > 0
    packed = _support_edges(
        graph, relevant, args.max_legal_edges_per_keypoint
    )
    offsets, support_values = _support_slices(packed, count)
    support_count = np.diff(offsets)
    rows = np.flatnonzero(relevant)
    local_of = {int(row): index for index, row in enumerate(rows)}
    dsu = _DisjointSet(len(rows))
    source = torch.as_tensor(state["source_primitive_ids"]).numpy()
    track = torch.as_tensor(state["track_cluster_ids"]).numpy()
    xyz = torch.as_tensor(state["anchor_xyz"]).float().numpy()
    features = F.normalize(
        torch.as_tensor(state["anchor_features"]).float(), dim=1
    ).numpy()

    for labels in (source, track):
        members = defaultdict(list)
        for row in rows:
            label = int(labels[row])
            if labels is track and label < 0:
                continue
            members[(int(band[row]), label)].append(int(row))
        for group_rows in members.values():
            root = local_of[group_rows[0]]
            for row in group_rows[1:]:
                dsu.union(root, local_of[row])

    signatures = _minhash(offsets, support_values, rows)
    buckets = defaultdict(list)
    for local, row in enumerate(rows):
        if support_count[row] == 0:
            continue
        for columns in ((0, 1), (2, 3)):
            buckets[
                (
                    int(band[row]),
                    int(signatures[local, columns[0]]),
                    int(signatures[local, columns[1]]),
                )
            ].append(int(row))
    compared = set()
    for bucket_rows in buckets.values():
        if len(bucket_rows) > 128:
            bucket_rows = sorted(
                bucket_rows,
                key=lambda row: (-support_count[row], row),
            )[:128]
        for left_index, left in enumerate(bucket_rows):
            for right in bucket_rows[left_index + 1 :]:
                pair = (min(left, right), max(left, right))
                if pair in compared:
                    continue
                compared.add(pair)
                jaccard = _jaccard(
                    offsets, support_values, left, right
                )
                if jaccard < float(args.support_jaccard):
                    continue
                cosine = float(features[left] @ features[right])
                distance = float(np.linalg.norm(xyz[left] - xyz[right]))
                if (
                    cosine >= float(args.descriptor_cosine)
                    or distance <= float(args.spatial_radius_m)
                ):
                    dsu.union(local_of[left], local_of[right])

    no_support_buckets = defaultdict(list)
    for row in rows[support_count[rows] == 0]:
        voxel = tuple(np.floor(xyz[row] / 0.5).astype(int).tolist())
        descriptor_hash = int(
            sum((features[row, index] > 0) << index for index in range(8))
        )
        no_support_buckets[
            (int(band[row]), voxel, descriptor_hash)
        ].append(int(row))
    for bucket_rows in no_support_buckets.values():
        for start in range(0, len(bucket_rows), 16):
            block = bucket_rows[start : start + 16]
            root = local_of[block[0]]
            for row in block[1:]:
                dsu.union(root, local_of[row])

    components = defaultdict(list)
    for row in rows:
        components[dsu.find(local_of[int(row)])].append(int(row))
    groups = []
    for component_rows in components.values():
        by_band = defaultdict(list)
        for row in component_rows:
            by_band[int(band[row])].append(row)
        for band_id, band_rows in by_band.items():
            for split in _split_component(
                band_rows, support_count, args.max_group_size
            ):
                group_rows = torch.as_tensor(split, dtype=torch.int32)
                opportunities = torch.as_tensor(
                    graph["candidate_opportunity_count"]
                )[group_rows.long()]
                harmful = torch.as_tensor(
                    graph["harmful_solver_inlier_count"]
                )[group_rows.long()]
                clean = torch.as_tensor(
                    graph["solver_inlier_gtclean_4px_count"]
                )[group_rows.long()]
                groups.append(
                    {
                        "group_id": len(groups),
                        "band": band_id,
                        "rows": group_rows,
                        "size": len(split),
                        "support_edge_count": int(
                            support_count[split].sum()
                        ),
                        "supported_anchor_count": int(
                            (support_count[split] > 0).sum()
                        ),
                        "harmful_consensus_count": int(harmful.sum()),
                        "gtclean_inlier_count": int(clean.sum()),
                        "opportunity_count": int(opportunities.sum()),
                    }
                )
    band_names = {
        1: "30k_to_35k",
        2: "35k_to_40k",
        3: "40k_to_45k",
        4: "45k_to_50k",
    }
    output = {
        "schema": "lafgs_redundancy_groups",
        "version": 1,
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
        "support_keypoint_ids": torch.from_numpy(support_values.astype(np.int64)),
        "groups": groups,
        "config": vars(args),
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, output_path)
    summary = {
        "group_count": len(groups),
        "support_edge_count": int(len(support_values)),
        "anchors_with_keypoint_support": int(
            (support_count[rows] > 0).sum()
        ),
        "bands": {
            band_names[band_id]: {
                "anchor_count": int((band == band_id).sum()),
                "group_count": sum(
                    int(group["band"]) == band_id for group in groups
                ),
            }
            for band_id in band_names
        },
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
