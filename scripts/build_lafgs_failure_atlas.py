#!/usr/bin/env python3
"""Build a view-conditioned failure atlas and active render-view manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from localization_training.failure_atlas import (
    FailureAtlasConfig,
    build_failure_atlas,
    plan_failure_conditioned_views,
)
from localization_training.shared_metric import SharedLowRankMetric


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", required=True)
    parser.add_argument("--metric-state", required=True)
    parser.add_argument(
        "--family-state",
        default="",
        help="Optional appearance family. Omit for the frozen A1 base-metric matcher.",
    )
    parser.add_argument("--dynamic-outcomes", required=True)
    parser.add_argument("--complete-positive-teacher", required=True)
    parser.add_argument("--query-cache", required=True)
    parser.add_argument("--track-payload", required=True)
    parser.add_argument("--basin-teacher", default="")
    parser.add_argument("--output", required=True)
    parser.add_argument("--render-manifest", default="")
    parser.add_argument("--scene", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--topk", type=int, default=16)
    parser.add_argument("--clean-threshold-px", type=float, default=2.0)
    parser.add_argument(
        "--coarse-clean-threshold-px", type=float, default=4.0
    )
    parser.add_argument("--harmful-threshold-px", type=float, default=12.0)
    parser.add_argument("--minimum-matchable-rate", type=float, default=0.08)
    parser.add_argument(
        "--minimum-render-topk-recall", type=float, default=0.25
    )
    parser.add_argument("--assignment-gap", type=float, default=0.08)
    parser.add_argument("--risk-tail-quantile", type=float, default=0.75)
    parser.add_argument("--maximum-planned-views", type=int, default=64)
    parser.add_argument("--maximum-views-per-source", type=int, default=2)
    parser.add_argument(
        "--maximum-views-per-trajectory", type=int, default=16
    )
    parser.add_argument(
        "--maximum-views-per-component", type=int, default=8
    )
    parser.add_argument("--interpolation-alphas", default="0.35,0.5,0.65")
    parser.add_argument(
        "--planner-mode",
        choices=("adjacent", "viewpoint_completion"),
        default="viewpoint_completion",
    )
    parser.add_argument("--partner-candidates", type=int, default=4)
    parser.add_argument("--minimum-normalized-view-gap", type=float, default=0.75)
    parser.add_argument(
        "--maximum-normalized-pair-distance", type=float, default=6.0
    )
    parser.add_argument("--maximum-pair-rotation-degrees", type=float, default=55.0)
    parser.add_argument("--view-gap-weight", type=float, default=0.75)
    parser.add_argument("--anchor-coverage-weight", type=float, default=0.5)
    parser.add_argument("--artifact-risk-weight", type=float, default=0.75)
    args = parser.parse_args()

    state = torch.load(args.map, map_location="cpu", weights_only=False)
    metric_payload = torch.load(
        args.metric_state, map_location="cpu", weights_only=False
    )
    family = (
        torch.load(args.family_state, map_location="cpu", weights_only=False)
        if args.family_state
        else None
    )
    dynamic = torch.load(
        args.dynamic_outcomes, map_location="cpu", weights_only=False
    )
    positives = torch.load(
        args.complete_positive_teacher,
        map_location="cpu",
        weights_only=False,
    )
    cache_payload = torch.load(
        args.query_cache, map_location="cpu", weights_only=False
    )
    cache = cache_payload.get("queries", cache_payload)
    track = torch.load(
        args.track_payload, map_location="cpu", weights_only=False
    )
    query_bins = {
        str(name): int(group)
        for name, group in zip(
            track["query_names"],
            torch.as_tensor(track["query_bins"]).tolist(),
        )
    }
    basin_teacher = (
        torch.load(
            args.basin_teacher, map_location="cpu", weights_only=False
        )
        if args.basin_teacher
        else None
    )
    metric = SharedLowRankMetric(**metric_payload["metric_config"])
    metric.load_state_dict(metric_payload["metric_state_dict"])
    config = FailureAtlasConfig(
        topk=args.topk,
        clean_threshold_px=args.clean_threshold_px,
        coarse_clean_threshold_px=args.coarse_clean_threshold_px,
        harmful_threshold_px=args.harmful_threshold_px,
        minimum_matchable_rate=args.minimum_matchable_rate,
        minimum_render_topk_recall=args.minimum_render_topk_recall,
        assignment_gap=args.assignment_gap,
        risk_tail_quantile=args.risk_tail_quantile,
        maximum_planned_views=args.maximum_planned_views,
        maximum_views_per_source=args.maximum_views_per_source,
        maximum_views_per_trajectory=args.maximum_views_per_trajectory,
        maximum_views_per_component=args.maximum_views_per_component,
        interpolation_alphas=tuple(
            float(value)
            for value in args.interpolation_alphas.split(",")
            if value.strip()
        ),
        planner_mode=args.planner_mode,
        partner_candidates=args.partner_candidates,
        minimum_normalized_view_gap=args.minimum_normalized_view_gap,
        maximum_normalized_pair_distance=args.maximum_normalized_pair_distance,
        maximum_pair_rotation_degrees=args.maximum_pair_rotation_degrees,
        view_gap_weight=args.view_gap_weight,
        anchor_coverage_weight=args.anchor_coverage_weight,
        artifact_risk_weight=args.artifact_risk_weight,
    )

    def progress(completed: int, query_count: int) -> None:
        if completed % 25 == 0:
            print(
                json.dumps(
                    {"completed": completed, "query_count": query_count}
                ),
                flush=True,
            )

    atlas = build_failure_atlas(
        state=state,
        metric=metric,
        family=family,
        dynamic=dynamic,
        positives=positives,
        cache=cache,
        query_bins=query_bins,
        basin_teacher=basin_teacher,
        config=config,
        device=torch.device(args.device),
        progress=progress,
    )
    planned = plan_failure_conditioned_views(
        atlas=atlas, cache=cache, config=config
    )
    atlas["planned_views"] = planned
    atlas["summary"]["planned_view_count"] = len(planned)
    path = Path(args.output).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(atlas, path)
    json_payload = {
        "schema": atlas["schema"],
        "version": atlas["version"],
        "matcher": atlas["matcher"],
        "summary": atlas["summary"],
        "config": atlas["config"],
        "cells": atlas["cells"],
        "planned_views": [
            {
                **record,
                "K": torch.as_tensor(record["K"]).tolist(),
            }
            for record in planned
        ],
    }
    path.with_suffix(".json").write_text(
        json.dumps(json_payload, indent=2) + "\n"
    )
    manifest_path = Path(
        args.render_manifest
        or path.with_name(path.stem + "_render_manifest.jsonl")
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w") as stream:
        for index, record in enumerate(planned):
            image_name = f"failure_render/{index:06d}.png"
            stream.write(
                json.dumps(
                    {
                        "query_id": record["query_id"],
                        "scene": str(args.scene),
                        "source": "failure_render",
                        "image_name": image_name,
                        "image_path": "",
                        "pose_w2c": record["pose_w2c"],
                        "fovx": record["fovx"],
                        "fovy": record["fovy"],
                        "width": record["width"],
                        "height": record["height"],
                        "accepted": False,
                        "reason": "not_rendered",
                        "artifact_score": 0.0,
                        "repair_action": "none",
                        "nearest_train_image": record["source_query"],
                        "synthetic_alpha": record["synthetic_alpha"],
                        "teacher_cache_key": record["query_id"],
                        "meta": {
                            key: (
                                torch.as_tensor(value).tolist()
                                if key == "K"
                                else value
                            )
                            for key, value in record.items()
                            if key
                            not in {
                                "query_id",
                                "pose_w2c",
                                "fovx",
                                "fovy",
                                "width",
                                "height",
                            }
                        },
                    },
                    sort_keys=True,
                )
                + "\n"
            )
    print(path)
    print(manifest_path)


if __name__ == "__main__":
    main()
