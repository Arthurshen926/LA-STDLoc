#!/usr/bin/env python3
"""Plan confusion-conditioned Gaussian views from real assignment failures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from localization_training.confusion_evidence import (
    ConfusionViewPlanningConfig,
    filter_confusion_graph_by_context_oracle,
    plan_confusion_conditioned_views,
    plan_reference_guided_confusion_views,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confusion-graph", required=True)
    parser.add_argument("--map", required=True)
    parser.add_argument("--query-cache", required=True)
    parser.add_argument("--track-payload", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--scene", default="")
    parser.add_argument("--context-oracle", default="")
    parser.add_argument(
        "--context-oracle-method",
        default="O1_cross_trajectory_2d",
    )
    parser.add_argument(
        "--minimum-context-oracle-positive-fraction",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--minimum-context-oracle-records",
        type=int,
        default=2,
    )
    parser.add_argument("--maximum-planned-views", type=int, default=64)
    parser.add_argument("--maximum-edges", type=int, default=256)
    parser.add_argument("--maximum-events-per-edge", type=int, default=3)
    parser.add_argument("--maximum-pose-neighbors", type=int, default=4)
    parser.add_argument("--maximum-views-per-edge", type=int, default=2)
    parser.add_argument("--maximum-views-per-source", type=int, default=2)
    parser.add_argument(
        "--maximum-views-per-trajectory", type=int, default=16
    )
    parser.add_argument("--minimum-edge-occurrences", type=int, default=5)
    parser.add_argument("--minimum-edge-trajectories", type=int, default=1)
    parser.add_argument("--maximum-neighbor-scale", type=float, default=4.0)
    parser.add_argument("--maximum-view-angle-deg", type=float, default=40.0)
    parser.add_argument("--image-margin-px", type=float, default=16.0)
    parser.add_argument("--interpolation-alphas", default="0.35,0.5,0.65")
    parser.add_argument(
        "--planning-policy",
        choices=("interpolation", "reference_guided_arc"),
        default="interpolation",
    )
    parser.add_argument("--arc-yaw-degrees", default="-10,-5,5,10")
    parser.add_argument("--arc-vertical-fractions", default="0")
    parser.add_argument("--minimum-pose-novelty", type=float, default=0.15)
    parser.add_argument(
        "--maximum-safe-envelope-scale", type=float, default=3.0
    )
    parser.add_argument("--context-neighbor-count", type=int, default=16)
    parser.add_argument(
        "--context-separation-weight", type=float, default=8.0
    )
    parser.add_argument("--pose-novelty-weight", type=float, default=0.5)
    parser.add_argument("--diversity-weight", type=float, default=0.5)
    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=1,
        help="Small planner kernels are faster without large OpenMP teams.",
    )
    args = parser.parse_args()
    torch.set_num_threads(max(int(args.cpu_threads), 1))

    graph = torch.load(
        args.confusion_graph, map_location="cpu", weights_only=False
    )
    oracle_filter = None
    if args.context_oracle:
        oracle = json.loads(Path(args.context_oracle).read_text())
        graph, oracle_filter = filter_confusion_graph_by_context_oracle(
            graph,
            oracle,
            method=args.context_oracle_method,
            minimum_positive_fraction=(
                args.minimum_context_oracle_positive_fraction
            ),
            minimum_records=args.minimum_context_oracle_records,
        )
    state = torch.load(args.map, map_location="cpu", weights_only=False)
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
    config = ConfusionViewPlanningConfig(
        maximum_planned_views=args.maximum_planned_views,
        maximum_edges=args.maximum_edges,
        maximum_events_per_edge=args.maximum_events_per_edge,
        maximum_pose_neighbors=args.maximum_pose_neighbors,
        maximum_views_per_edge=args.maximum_views_per_edge,
        maximum_views_per_source=args.maximum_views_per_source,
        maximum_views_per_trajectory=args.maximum_views_per_trajectory,
        minimum_edge_occurrences=args.minimum_edge_occurrences,
        minimum_edge_trajectories=args.minimum_edge_trajectories,
        maximum_neighbor_scale=args.maximum_neighbor_scale,
        maximum_view_angle_deg=args.maximum_view_angle_deg,
        image_margin_px=args.image_margin_px,
        interpolation_alphas=tuple(
            float(value)
            for value in args.interpolation_alphas.split(",")
            if value.strip()
        ),
        arc_yaw_degrees=tuple(
            float(value)
            for value in args.arc_yaw_degrees.split(",")
            if value.strip()
        ),
        arc_vertical_fractions=tuple(
            float(value)
            for value in args.arc_vertical_fractions.split(",")
            if value.strip()
        ),
        minimum_pose_novelty=args.minimum_pose_novelty,
        maximum_safe_envelope_scale=args.maximum_safe_envelope_scale,
        context_neighbor_count=args.context_neighbor_count,
        context_separation_weight=args.context_separation_weight,
        pose_novelty_weight=args.pose_novelty_weight,
        diversity_weight=args.diversity_weight,
    )
    planner = (
        plan_reference_guided_confusion_views
        if args.planning_policy == "reference_guided_arc"
        else plan_confusion_conditioned_views
    )
    planned = planner(
        confusion_graph=graph,
        state=state,
        cache=cache,
        query_bins=query_bins,
        config=config,
    )
    path = Path(args.output).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as stream:
        for index, record in enumerate(planned):
            stream.write(
                json.dumps(
                    {
                        "query_id": record["query_id"],
                        "scene": str(args.scene),
                        "source": "confusion_render",
                        "image_name": f"confusion_render/{index:06d}.png",
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
    summary = {
        "schema": "lafgs_confusion_conditioned_view_plan",
        "version": 1,
        "confusion_graph": str(Path(args.confusion_graph).resolve()),
        "map": str(Path(args.map).resolve()),
        "query_cache": str(Path(args.query_cache).resolve()),
        "query_count": len(planned),
        "cross_trajectory_count": sum(
            bool(record["cross_trajectory"]) for record in planned
        ),
        "targeted_edge_count": len(
            {int(record["edge_index"]) for record in planned}
        ),
        "planning_policy": str(args.planning_policy),
        "context_oracle_filter": oracle_filter,
        "mean_pose_novelty": (
            sum(float(record.get("pose_novelty", 0.0)) for record in planned)
            / len(planned)
            if planned
            else 0.0
        ),
        "mean_projected_context_separation": (
            sum(
                float(record.get("projected_context_separation", 0.0))
                for record in planned
            )
            / len(planned)
            if planned
            else 0.0
        ),
        "config": vars(args),
    }
    path.with_suffix(".json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
