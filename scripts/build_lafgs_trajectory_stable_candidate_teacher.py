#!/usr/bin/env python3
"""Build a training-only trajectory-stable candidate promotion teacher."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from localization_training.shared_metric import SharedLowRankMetric
from localization_training.trajectory_stable_candidate import (
    TrajectoryStableCandidateConfig,
    build_trajectory_stable_candidate_teacher,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", required=True)
    parser.add_argument("--metric-state", required=True)
    parser.add_argument("--dynamic-outcomes", required=True)
    parser.add_argument("--complete-positive-teacher", required=True)
    parser.add_argument("--query-cache", required=True)
    parser.add_argument("--track-payload", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--topk", type=int, default=16)
    parser.add_argument("--minimum-observations", type=int, default=2)
    parser.add_argument("--minimum-trajectories", type=int, default=2)
    parser.add_argument("--minimum-view-bins", type=int, default=2)
    parser.add_argument("--single-trajectory-minimum-observations", type=int, default=3)
    parser.add_argument("--single-trajectory-minimum-view-bins", type=int, default=3)
    parser.add_argument("--maximum-score-gap", type=float, default=0.20)
    parser.add_argument("--promotion-weight", type=float, default=2.0)
    parser.add_argument("--harmful-inlier-multiplier", type=float, default=1.5)
    args = parser.parse_args()

    state = torch.load(args.map, map_location="cpu", weights_only=False)
    metric_payload = torch.load(args.metric_state, map_location="cpu", weights_only=False)
    metric = SharedLowRankMetric(**metric_payload["metric_config"])
    metric.load_state_dict(metric_payload["metric_state_dict"])
    positives = torch.load(
        args.complete_positive_teacher, map_location="cpu", weights_only=False
    )
    dynamic = torch.load(args.dynamic_outcomes, map_location="cpu", weights_only=False)
    cache_payload = torch.load(args.query_cache, map_location="cpu", weights_only=False)
    cache = cache_payload.get("queries", cache_payload)
    track = torch.load(args.track_payload, map_location="cpu", weights_only=False)
    query_bins = {
        str(name): int(view_bin)
        for name, view_bin in zip(
            track["query_names"], torch.as_tensor(track["query_bins"]).tolist()
        )
    }
    config = TrajectoryStableCandidateConfig(
        topk=args.topk,
        minimum_observations=args.minimum_observations,
        minimum_trajectories=args.minimum_trajectories,
        minimum_view_bins=args.minimum_view_bins,
        single_trajectory_minimum_observations=(
            args.single_trajectory_minimum_observations
        ),
        single_trajectory_minimum_view_bins=(
            args.single_trajectory_minimum_view_bins
        ),
        maximum_score_gap=args.maximum_score_gap,
        promotion_weight=args.promotion_weight,
        harmful_inlier_multiplier=args.harmful_inlier_multiplier,
    )

    def progress(completed: int, total: int) -> None:
        if completed % 25 == 0 or completed == total:
            print(json.dumps({"completed": completed, "query_count": total}), flush=True)

    output = build_trajectory_stable_candidate_teacher(
        state=state,
        metric=metric,
        positives=positives,
        dynamic=dynamic,
        cache=cache,
        query_bins=query_bins,
        config=config,
        device=torch.device(args.device),
        progress=progress,
    )
    path = Path(args.output).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, path)
    path.with_suffix(".json").write_text(
        json.dumps(
            {
                "schema": output["schema"],
                "version": output["version"],
                "summary": output["summary"],
                "config": output["config"],
                "stable_support": output["stable_support"],
            },
            indent=2,
        )
        + "\n"
    )
    print(path)


if __name__ == "__main__":
    main()
