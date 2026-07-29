#!/usr/bin/env python3
"""Build the directed anchor-family confusion graph from real localization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from localization_training.confusion_evidence import (
    ConfusionGraphConfig,
    build_anchor_family_confusion_graph,
)
from localization_training.shared_metric import SharedLowRankMetric


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", required=True)
    parser.add_argument("--metric-state", required=True)
    parser.add_argument("--family-state", required=True)
    parser.add_argument("--dynamic-outcomes", required=True)
    parser.add_argument("--complete-positive-teacher", required=True)
    parser.add_argument("--query-cache", required=True)
    parser.add_argument("--track-payload", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--minimum-occurrences", type=int, default=2)
    parser.add_argument("--minimum-trajectories", type=int, default=1)
    parser.add_argument("--harmful-threshold-px", type=float, default=12.0)
    parser.add_argument("--clean-threshold-px", type=float, default=4.0)
    parser.add_argument("--maximum-events-per-edge", type=int, default=64)
    args = parser.parse_args()

    state = torch.load(args.map, map_location="cpu", weights_only=False)
    metric_payload = torch.load(
        args.metric_state, map_location="cpu", weights_only=False
    )
    metric = SharedLowRankMetric(**metric_payload["metric_config"])
    metric.load_state_dict(metric_payload["metric_state_dict"])
    family = torch.load(
        args.family_state, map_location="cpu", weights_only=False
    )
    dynamic = torch.load(
        args.dynamic_outcomes, map_location="cpu", weights_only=False
    )
    positives = torch.load(
        args.complete_positive_teacher, map_location="cpu", weights_only=False
    )
    cache_payload = torch.load(
        args.query_cache, map_location="cpu", weights_only=False
    )
    cache = cache_payload.get("queries", cache_payload)
    track = torch.load(
        args.track_payload, map_location="cpu", weights_only=False
    )
    query_bins = {
        str(name): int(value)
        for name, value in zip(
            track["query_names"],
            torch.as_tensor(track["query_bins"]).tolist(),
        )
    }

    def progress(completed: int, total: int) -> None:
        if completed % 25 == 0:
            print(json.dumps({"completed": completed, "total": total}), flush=True)

    config = ConfusionGraphConfig(
        minimum_occurrences=args.minimum_occurrences,
        minimum_trajectories=args.minimum_trajectories,
        harmful_threshold_px=args.harmful_threshold_px,
        clean_threshold_px=args.clean_threshold_px,
        maximum_events_per_edge=args.maximum_events_per_edge,
    )
    graph = build_anchor_family_confusion_graph(
        state=state,
        metric=metric,
        family=family,
        dynamic=dynamic,
        positives=positives,
        cache=cache,
        query_bins=query_bins,
        config=config,
        device=torch.device(args.device),
        progress=progress,
    )
    graph["provenance"] = {
        key: str(Path(value).resolve())
        for key, value in {
            "map": args.map,
            "metric_state": args.metric_state,
            "family_state": args.family_state,
            "dynamic_outcomes": args.dynamic_outcomes,
            "complete_positive_teacher": args.complete_positive_teacher,
            "query_cache": args.query_cache,
            "track_payload": args.track_payload,
        }.items()
    }
    path = Path(args.output).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(graph, path)
    path.with_suffix(".json").write_text(
        json.dumps(
            {
                "schema": graph["schema"],
                "version": graph["version"],
                "summary": graph["summary"],
                "config": graph["config"],
                "provenance": graph["provenance"],
                "edges": graph["edges"],
            },
            indent=2,
        )
        + "\n"
    )
    print(path)


if __name__ == "__main__":
    main()
