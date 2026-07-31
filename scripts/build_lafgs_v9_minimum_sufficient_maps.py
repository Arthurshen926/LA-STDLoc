#!/usr/bin/env python3
"""Build Track-core maps with an automatically sized query multi-cover reserve."""

from __future__ import annotations

import argparse
import heapq
import json
from pathlib import Path

import numpy as np
import torch

try:
    from scripts.build_lafgs_v7_track_centric_maps import (
        _base_utility,
        _eligible_tracks,
        _materialize,
        _select_capacity_limited_tracks,
        _track_quality,
    )
except ModuleNotFoundError:
    from build_lafgs_v7_track_centric_maps import (
        _base_utility,
        _eligible_tracks,
        _materialize,
        _select_capacity_limited_tracks,
        _track_quality,
    )
from localization_training.micro_anchors import fuse_track_descriptors
try:
    from scripts.build_lafgs_v9_complete_positive_teacher import (
        _query_index_remap,
    )
except ModuleNotFoundError:
    from build_lafgs_v9_complete_positive_teacher import _query_index_remap


_EVENT_SHIFT = 32


def _event_id(query_index: int, keypoint_index: int) -> int:
    return (int(query_index) << _EVENT_SHIFT) | int(keypoint_index)


def _event_query(event: int) -> int:
    return int(event) >> _EVENT_SHIFT


def _positive_events_by_anchor(
    teacher: dict, candidate_count: int
) -> list[set[int]]:
    """Invert complete CSR positives into anchor-to-query/keypoint events."""
    events = [set() for _ in range(int(candidate_count))]
    for record in teacher["records"]:
        query = int(record["query_index"])
        rows = torch.as_tensor(record["query_rows"]).long()
        offsets = torch.as_tensor(record["positive_offsets"]).long()
        indices = torch.as_tensor(record["positive_indices"]).long()
        counts = offsets[1:] - offsets[:-1]
        nonempty = counts > 0
        if not bool(nonempty.any()):
            continue
        repeated_rows = torch.repeat_interleave(rows[nonempty], counts[nonempty])
        valid = (indices >= 0) & (indices < int(candidate_count))
        for anchor, row in zip(
            indices[valid].tolist(), repeated_rows[valid].tolist()
        ):
            events[int(anchor)].add(_event_id(query, row))
    return events


def _track_core_events(
    payload: dict,
    selected_tracks: torch.Tensor,
    query_index_remap: torch.Tensor | None = None,
) -> set[int]:
    track_count = int(
        torch.as_tensor(payload["track_geometry"]["triangulated"]).numel()
    )
    selected = torch.zeros(track_count, dtype=torch.bool)
    selected[torch.as_tensor(selected_tracks).long()] = True
    observations = payload["tracks"]
    observation_track = torch.as_tensor(observations["track_index"]).long()
    observation_query = torch.as_tensor(observations["query_index"]).long()
    observation_keypoint = torch.as_tensor(
        observations["keypoint_index"]
    ).long()
    keep = selected[observation_track]
    result = set()
    for query, keypoint in zip(
        observation_query[keep].tolist(),
        observation_keypoint[keep].tolist(),
    ):
        target_query = (
            int(query_index_remap[int(query)])
            if query_index_remap is not None
            else int(query)
        )
        result.add(_event_id(target_query, keypoint))
    return result


def _query_weights(query_groups: torch.Tensor) -> np.ndarray:
    groups = torch.as_tensor(query_groups).long()
    sizes = torch.bincount(groups).float().clamp_min(1)
    reference = sizes.mean()
    return torch.sqrt(reference / sizes[groups]).numpy()


def _event_gain(
    events: set[int],
    covered: set[int],
    deficits: np.ndarray,
    weights: np.ndarray,
) -> float:
    return float(
        sum(
            weights[_event_query(event)]
            for event in events
            if event not in covered and deficits[_event_query(event)] > 0
        )
    )


def greedy_query_multicover(
    events_by_anchor: list[set[int]],
    core_events: set[int],
    query_groups: torch.Tensor,
    *,
    minimum_rows_per_query: int,
    utility: torch.Tensor,
    eligible: torch.Tensor | None = None,
    maximum_reserve: int | None = None,
) -> tuple[torch.Tensor, dict]:
    """Select the smallest greedy reserve satisfying per-query row coverage.

    Sequence-size-normalized gains make a positive from a short trajectory
    worth more than another duplicate from a heavily represented trajectory.
    """
    query_groups = torch.as_tensor(query_groups).long()
    query_count = int(query_groups.numel())
    if len(events_by_anchor) != int(torch.as_tensor(utility).numel()):
        raise ValueError("events and utility must describe the same anchors")
    eligible = (
        torch.ones(len(events_by_anchor), dtype=torch.bool)
        if eligible is None
        else torch.as_tensor(eligible).bool()
    )
    if eligible.numel() != len(events_by_anchor):
        raise ValueError("eligible mask must describe every candidate anchor")

    available = [set() for _ in range(query_count)]
    for event in core_events:
        available[_event_query(event)].add(event)
    for anchor, events in enumerate(events_by_anchor):
        if not bool(eligible[anchor]):
            continue
        for event in events:
            available[_event_query(event)].add(event)
    targets = np.asarray(
        [
            min(int(minimum_rows_per_query), len(query_events))
            for query_events in available
        ],
        dtype=np.int64,
    )
    covered = set(core_events)
    core_counts = np.zeros(query_count, dtype=np.int64)
    for event in core_events:
        core_counts[_event_query(event)] += 1
    deficits = np.maximum(targets - core_counts, 0)
    weights = _query_weights(query_groups)
    utility_values = torch.as_tensor(utility).float().numpy()

    heap = []
    for anchor, events in enumerate(events_by_anchor):
        if not bool(eligible[anchor]) or not events:
            continue
        gain = _event_gain(events, covered, deficits, weights)
        if gain > 0:
            heapq.heappush(
                heap, (-gain, -float(utility_values[anchor]), anchor)
            )

    selected = []
    while int(deficits.sum()) > 0 and heap:
        if maximum_reserve is not None and len(selected) >= maximum_reserve:
            break
        negative_gain, negative_utility, anchor = heapq.heappop(heap)
        gain = _event_gain(
            events_by_anchor[anchor], covered, deficits, weights
        )
        if not np.isclose(gain, -negative_gain, rtol=0.0, atol=1e-9):
            if gain > 0:
                heapq.heappush(
                    heap, (-gain, negative_utility, anchor)
                )
            continue
        if gain <= 0:
            break
        selected.append(anchor)
        for event in events_by_anchor[anchor]:
            query = _event_query(event)
            if event in covered or deficits[query] <= 0:
                continue
            covered.add(event)
            deficits[query] -= 1

    achieved = targets - deficits
    report = {
        "minimum_rows_per_query": int(minimum_rows_per_query),
        "reserve_count": len(selected),
        "target_event_count": int(targets.sum()),
        "achieved_event_count": int(achieved.sum()),
        "unmet_query_count": int((deficits > 0).sum()),
        "unmet_event_count": int(deficits.sum()),
        "core_coverage_p10": float(np.percentile(core_counts, 10)),
        "core_coverage_median": float(np.median(core_counts)),
        "final_coverage_p10": float(np.percentile(achieved, 10)),
        "final_coverage_median": float(np.median(achieved)),
    }
    return torch.as_tensor(selected, dtype=torch.long), report


def _parse_specs(raw_specs: str) -> list[tuple[int, str]]:
    specs = []
    for raw in raw_specs.split(","):
        count, tier = raw.split(":")
        specs.append((int(count), tier))
    return specs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-map", required=True)
    parser.add_argument("--function-graph", required=True)
    parser.add_argument("--complete-positive-teacher", required=True)
    parser.add_argument("--track-payload", required=True)
    parser.add_argument("--query-cache", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--track-cores",
        default="8000:broad,10000:broad,12000:relaxed,14000:relaxed",
    )
    parser.add_argument("--minimum-rows-per-query", type=int, default=96)
    parser.add_argument("--maximum-harmful-rate", type=float, default=0.25)
    parser.add_argument("--maximum-reserve", type=int, default=16000)
    parser.add_argument("--descriptor-trim-fraction", type=float, default=0.2)
    parser.add_argument("--dependency-voxel-size", type=float, default=0.5)
    args = parser.parse_args()

    canonical_path = Path(args.canonical_map).resolve()
    graph_path = Path(args.function_graph).resolve()
    teacher_path = Path(args.complete_positive_teacher).resolve()
    payload_path = Path(args.track_payload).resolve()
    query_path = Path(args.query_cache).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    canonical = torch.load(
        canonical_path, map_location="cpu", weights_only=False
    )
    graph = torch.load(graph_path, map_location="cpu", weights_only=False)
    teacher = torch.load(
        teacher_path, map_location="cpu", weights_only=False
    )
    payload = torch.load(
        payload_path, map_location="cpu", weights_only=False
    )
    query_cache = torch.load(
        query_path, map_location="cpu", weights_only=False
    )
    base_count = int(canonical["base_anchor_count"])
    if int(teacher["anchor_count"]) != int(
        torch.as_tensor(canonical["anchor_xyz"]).shape[0]
    ):
        raise ValueError("complete teacher does not align with canonical map")
    payload_to_teacher = _query_index_remap(
        payload["query_names"], teacher["query_names"]
    )
    query_groups = torch.empty_like(torch.as_tensor(payload["query_bins"]))
    query_groups[payload_to_teacher] = torch.as_tensor(payload["query_bins"])

    print("Inverting complete positive teacher", flush=True)
    events_by_base = _positive_events_by_anchor(teacher, base_count)
    utility = _base_utility(graph, base_count)
    opportunity = torch.as_tensor(
        graph["provenance_opportunity_count"][:base_count]
    ).float()
    harmful = torch.as_tensor(
        graph["provenance_harmful_solver_inlier_count"][:base_count]
    ).float()
    harmful_rate = harmful / opportunity.clamp_min(1)
    eligible_base = (
        torch.as_tensor(
            graph["provenance_legal_hit_2px_count"][:base_count]
        )
        > 0
    ) & (harmful_rate <= float(args.maximum_harmful_rate))

    geometry = payload["track_geometry"]
    quality = _track_quality(geometry)
    specs = _parse_specs(args.track_cores)
    selected_by_spec = {}
    selected_union = []
    capacity_by_spec = {}
    for count, tier in specs:
        tier_mask = _eligible_tracks(geometry, tier)
        selected, capacity = _select_capacity_limited_tracks(
            quality, tier_mask, count
        )
        selected_by_spec[(count, tier)] = selected
        selected_union.append(selected)
        capacity_by_spec[(count, tier)] = capacity
        if selected.numel() < count:
            print(
                f"{tier} Track core is capacity-limited: "
                f"requested={count} eligible={int(tier_mask.sum())}; "
                "preserving the quality gate and using the realized core",
                flush=True,
            )

    unique_tracks = torch.unique(torch.cat(selected_union), sorted=False)
    print(f"Fusing {unique_tracks.numel()} Track descriptors", flush=True)
    original_threads = torch.get_num_threads()
    try:
        # Thousands of tiny medoid reductions are slower with a large OpenMP
        # pool because thread-launch overhead dominates the arithmetic.
        torch.set_num_threads(1)
        fused = fuse_track_descriptors(
            payload=payload,
            query_cache=query_cache,
            track_indices=unique_tracks,
            trim_fraction=args.descriptor_trim_fraction,
        )
    finally:
        torch.set_num_threads(original_threads)
    feature_by_track = {
        int(track): fused[row]
        for row, track in enumerate(unique_tracks.tolist())
    }

    summary = {
        "schema": "lafgs_v9_minimum_sufficient_map_build",
        "canonical_map": str(canonical_path),
        "function_graph": str(graph_path),
        "complete_positive_teacher": str(teacher_path),
        "track_payload": str(payload_path),
        "query_cache": str(query_path),
        "selection_split": "all_mapping_train",
        "selection_query_count": int(query_groups.numel()),
        "maps": {},
    }
    for track_count, tier in specs:
        tracks = selected_by_spec[(track_count, tier)]
        core_events = _track_core_events(
            payload, tracks, query_index_remap=payload_to_teacher
        )
        reserve, coverage = greedy_query_multicover(
            events_by_base,
            core_events,
            query_groups,
            minimum_rows_per_query=args.minimum_rows_per_query,
            utility=utility,
            eligible=eligible_base,
            maximum_reserve=args.maximum_reserve,
        )
        features = torch.stack(
            [feature_by_track[int(track)] for track in tracks.tolist()]
        )
        budget = int(tracks.numel() + reserve.numel())
        state = _materialize(
            canonical,
            payload,
            tracks,
            features,
            reserve,
            budget=budget,
            quality_tier=tier,
            source_map=canonical_path,
            payload_path=payload_path,
            dependency_voxel_size=args.dependency_voxel_size,
        )
        state["track_centric_reconstruction"].update(
            {
                "schema": "lafgs_v9_minimum_sufficient_map",
                "complete_positive_teacher": str(teacher_path),
                "automatic_base_reserve": True,
                "coverage": coverage,
            }
        )
        tag = (
            f"core{track_count:05d}_{tier}_"
            f"qrows{args.minimum_rows_per_query:03d}_"
            f"total{budget:05d}"
        )
        path = output_dir / f"minimum_sufficient_{tag}.pt"
        torch.save(state, path)
        summary["maps"][tag] = {
            "path": str(path),
            **capacity_by_spec[(track_count, tier)],
            "track_count": int(tracks.numel()),
            "base_reserve_count": int(reserve.numel()),
            "total_count": budget,
            "coverage": coverage,
        }
        print(
            f"{tag}: reserve={reserve.numel()} "
            f"unmet_queries={coverage['unmet_query_count']}",
            flush=True,
        )
    report = output_dir / "minimum_sufficient_build.json"
    report.write_text(json.dumps(summary, indent=2) + "\n")
    print(report)


if __name__ == "__main__":
    main()
