#!/usr/bin/env python3
"""Plan diversity-constrained Gaussian views from an existing Failure Atlas."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from localization_training.failure_atlas import (
    FailureAtlasConfig,
    plan_failure_conditioned_views,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atlas", required=True)
    parser.add_argument("--query-cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--scene", default="")
    parser.add_argument("--maximum-planned-views", type=int, default=64)
    parser.add_argument("--maximum-views-per-source", type=int, default=2)
    parser.add_argument(
        "--maximum-views-per-trajectory", type=int, default=16
    )
    parser.add_argument(
        "--maximum-views-per-component", type=int, default=8
    )
    parser.add_argument("--interpolation-alphas", default="0.35,0.5,0.65")
    args = parser.parse_args()

    atlas = torch.load(args.atlas, map_location="cpu", weights_only=False)
    cache_payload = torch.load(
        args.query_cache, map_location="cpu", weights_only=False
    )
    cache = cache_payload.get("queries", cache_payload)
    config = FailureAtlasConfig(
        maximum_planned_views=args.maximum_planned_views,
        maximum_views_per_source=args.maximum_views_per_source,
        maximum_views_per_trajectory=args.maximum_views_per_trajectory,
        maximum_views_per_component=args.maximum_views_per_component,
        interpolation_alphas=tuple(
            float(value)
            for value in args.interpolation_alphas.split(",")
            if value.strip()
        ),
    )
    planned = plan_failure_conditioned_views(
        atlas=atlas,
        cache=cache,
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
                        "source": "failure_render",
                        "image_name": f"failure_render/{index:06d}.png",
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
        "schema": "lafgs_failure_conditioned_view_plan",
        "version": 1,
        "atlas": str(Path(args.atlas).resolve()),
        "query_count": len(planned),
        "config": {
            "maximum_planned_views": args.maximum_planned_views,
            "maximum_views_per_source": args.maximum_views_per_source,
            "maximum_views_per_trajectory": (
                args.maximum_views_per_trajectory
            ),
            "maximum_views_per_component": args.maximum_views_per_component,
            "interpolation_alphas": list(config.interpolation_alphas),
        },
    }
    path.with_suffix(".json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(path)


if __name__ == "__main__":
    main()
