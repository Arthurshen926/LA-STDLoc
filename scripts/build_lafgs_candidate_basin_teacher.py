#!/usr/bin/env python3
"""Build top-K one/two-edge repair Basin supervision for appearance families."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from localization_training.artifact_contract import sha256_file
from localization_training.candidate_basin_teacher import (
    CandidateBasinConfig,
    build_candidate_basin_teacher,
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
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--topk", type=int, default=16)
    parser.add_argument("--good-sets-per-query", type=int, default=8)
    parser.add_argument("--harmful-sets-per-query", type=int, default=6)
    parser.add_argument("--repairs-per-query", type=int, default=8)
    parser.add_argument("--legal-candidates-per-row", type=int, default=4)
    parser.add_argument("--two-edge-beam", type=int, default=8)
    parser.add_argument("--adaptive-budget-factor", type=int, default=2)
    parser.add_argument("--high-tail-threshold-cm", type=float, default=0.0)
    parser.add_argument("--clean-threshold-px", type=float, default=4.0)
    parser.add_argument("--harmful-threshold-px", type=float, default=12.0)
    parser.add_argument("--minimum-harmful-inliers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--query-start", type=int, default=0)
    parser.add_argument("--query-limit", type=int, default=0)
    args = parser.parse_args()

    paths = {
        "map": Path(args.map).resolve(),
        "metric_state": Path(args.metric_state).resolve(),
        "family_state": Path(args.family_state).resolve(),
        "dynamic_outcomes": Path(args.dynamic_outcomes).resolve(),
        "complete_positive_teacher": Path(
            args.complete_positive_teacher
        ).resolve(),
        "query_cache": Path(args.query_cache).resolve(),
    }
    state = torch.load(paths["map"], map_location="cpu", weights_only=False)
    metric_payload = torch.load(
        paths["metric_state"], map_location="cpu", weights_only=False
    )
    family = torch.load(
        paths["family_state"], map_location="cpu", weights_only=False
    )
    dynamic = torch.load(
        paths["dynamic_outcomes"], map_location="cpu", weights_only=False
    )
    positives = torch.load(
        paths["complete_positive_teacher"],
        map_location="cpu",
        weights_only=False,
    )
    cache_payload = torch.load(
        paths["query_cache"], map_location="cpu", weights_only=False
    )
    metric = SharedLowRankMetric(**metric_payload["metric_config"])
    metric.load_state_dict(metric_payload["metric_state_dict"])
    config = CandidateBasinConfig(
        **{
            key: getattr(args, key)
            for key in CandidateBasinConfig.__dataclass_fields__
        }
    )

    def progress(completed: int, query_count: int, totals: dict) -> None:
        if completed % 10 == 0:
            print(
                json.dumps(
                    {
                        "completed": completed,
                        "query_count": query_count,
                        **totals,
                    }
                ),
                flush=True,
            )

    output = build_candidate_basin_teacher(
        state=state,
        metric=metric,
        family=family,
        dynamic=dynamic,
        positives=positives,
        cache=cache_payload.get("queries", cache_payload),
        config=config,
        device=torch.device(args.device),
        progress=progress,
    )
    output["config"] = vars(args)
    output["artifacts"] = {
        key: {"path": str(path), "sha256": sha256_file(path)}
        for key, path in paths.items()
    }
    path = Path(args.output).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(output, temporary)
    os.replace(temporary, path)
    path.with_suffix(".json").write_text(
        json.dumps(
            {
                "schema": output["schema"],
                "version": output["version"],
                "summary": output["summary"],
                "config": output["config"],
                "artifacts": output["artifacts"],
            },
            indent=2,
        )
        + "\n"
    )
    print(path)


if __name__ == "__main__":
    main()
