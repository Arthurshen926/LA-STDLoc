#!/usr/bin/env python3
"""Run the frozen LaFGS reconstruction pipeline from an imported RGB prior."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from data.datasets import ColmapDataset
from evaluation.evaluator import evaluate_dataset
from localization.localizer import SparseLocalizer
from common.config import (
    load_scene_calibration,
    load_mainline_config,
    resolve_keypoint_count,
    resolve_reprojection_error_px,
)
from map_learning.pipeline import (
    build_bootstrap_and_tracks,
    build_evidence,
    distill_compact_map,
    resolve_prior_ply,
    train_compact_map,
    write_pipeline_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--prior", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gaussian-type", choices=("2dgs", "3dgs"), required=True)
    parser.add_argument("--sh-degree", type=int, default=3)
    parser.add_argument("--config", default="configs/paper_mainline.yaml")
    parser.add_argument("--valid-masks", default="")
    parser.add_argument("--function-graph-shards", type=int, default=1)
    parser.add_argument("--provenance-shards", type=int, default=1)
    parser.add_argument("--observation-shards", type=int, default=1)
    parser.add_argument("--pose-scoring-shards", type=int, default=1)
    parser.add_argument("--evaluate", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    artifacts = build_bootstrap_and_tracks(
        dataset=args.dataset,
        prior=args.prior,
        output=args.output,
        gaussian_type=args.gaussian_type,
        sh_degree=args.sh_degree,
        config=args.config,
    )
    prior_ply = resolve_prior_ply(args.prior)
    evidence = build_evidence(
        base_state=artifacts["base_state"],
        track_payload=artifacts["track_payload"],
        query_cache=artifacts["query_cache"],
        prior_ply=prior_ply,
        gaussian_type=args.gaussian_type,
        sh_degree=args.sh_degree,
        visibility_cache=artifacts["visibility_cache"],
        output=args.output / "evidence",
        config=args.config,
        valid_masks=args.valid_masks or None,
        function_graph_shards=args.function_graph_shards,
        provenance_shards=args.provenance_shards,
        observation_shards=args.observation_shards,
    )
    compact_map = distill_compact_map(
        canonical_map=evidence["canonical_map"],
        function_graph=evidence["function_graph"],
        positive_teacher=evidence["positive_teacher"],
        track_payload=artifacts["track_payload"],
        query_cache=artifacts["query_cache"],
        output=args.output / "topology",
        config=args.config,
        pose_scoring_shards=args.pose_scoring_shards,
    )
    trained = train_compact_map(
        compact_map=compact_map,
        function_graph=evidence["function_graph"],
        track_payload=artifacts["track_payload"],
        query_cache=artifacts["query_cache"],
        prior_ply=prior_ply,
        gaussian_type=args.gaussian_type,
        sh_degree=args.sh_degree,
        output=args.output / "map_learning",
        config=args.config,
        valid_masks=args.valid_masks or None,
        provenance_shards=args.provenance_shards,
        observation_shards=args.observation_shards,
    )
    outputs = {
        **artifacts,
        **evidence,
        **trained,
        "prior_ply": prior_ply,
        "compact_map": compact_map,
    }
    if args.evaluate:
        deployment = load_mainline_config(args.config).values["deployment"]
        dataset = ColmapDataset(args.dataset, images="processed")
        test_cameras = dataset.split("test")
        mapping_cameras = dataset.split("mapping")
        calibration_path = Path(trained["trained_map"]).parent / "scene_calibration.json"
        scene_calibration = load_scene_calibration(calibration_path)
        keypoint_count = resolve_keypoint_count(deployment, mapping_cameras)
        reprojection_error_px = resolve_reprojection_error_px(
            deployment, mapping_cameras, scene_calibration
        )
        localizer = SparseLocalizer(
            trained["trained_map"],
            trained["metric_state"],
            device=args.device,
            keypoint_count=keypoint_count,
            reprojection_error_px=reprojection_error_px,
            confidence=deployment["confidence"],
            max_iterations=deployment["maximum_iterations"],
            min_iterations=deployment["minimum_iterations"],
            seed=2026,
        )
        result = evaluate_dataset(
            dataset=dataset,
            localizer=localizer,
            cameras=test_cameras,
            output=args.output / "evaluation",
        )
        (args.output / "evaluation" / "deployment_contract.json").write_text(
            json.dumps(
                {
                    "schema": "lafgs_sparse_deployment_contract",
                    "version": 1,
                    "keypoint_count": int(keypoint_count),
                    "ransac_reprojection_px": float(reprojection_error_px),
                    "scene_calibration": str(calibration_path.resolve()),
                    "calibration_split": "mapping",
                    "evaluated_split": "test",
                    "pose_solves": 1,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        print(json.dumps(result["summary"], indent=2))
        outputs["evaluation"] = args.output / "evaluation"
    write_pipeline_manifest(args.output, outputs)


if __name__ == "__main__":
    main()
