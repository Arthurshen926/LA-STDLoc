#!/usr/bin/env python3
"""Run the frozen LaFGS reconstruction pipeline from an imported RGB prior."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Sequence

from common.anchor_registry_artifact import materialize_anchor_registry
from common.hashing import sha256_file
from common.pipeline_completion import write_pipeline_completion
from data.datasets import ColmapDataset
from evaluation.evaluator import evaluate_dataset
from localization.localizer import SparseLocalizer
from common.config import (
    load_scene_calibration,
    load_mainline_config,
    materialize_keypoint_factor_config,
    materialize_mapping_keypoint_config,
    materialize_surface_track_config,
    resolve_keypoint_count,
    resolve_reprojection_error_px,
)
from map_learning.pipeline import (
    build_bootstrap_and_tracks,
    build_evidence,
    configure_build_timing,
    distill_compact_map,
    finalize_build_timing,
    resolve_prior_ply,
    timed_pipeline_stage,
    train_compact_map,
    write_pipeline_manifest,
)


def build_parser() -> argparse.ArgumentParser:
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
    parser.add_argument(
        "--gpu-workers-per-device",
        type=int,
        default=0,
        help=(
            "Bound concurrent shard processes per visible GPU; zero preserves "
            "historical scheduling. Use one for long memory-heavy shards."
        ),
    )
    parser.add_argument(
        "--evaluate",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Explicitly opt into reading and evaluating the test split.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--keypoints",
        type=int,
        choices=(1024, 2048),
        help=(
            "Lock detector density for mapping observations, reconstruction, "
            "and sparse deployment in a materialized factor config."
        ),
    )
    parser.add_argument(
        "--surface-supported-tracks",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable the cross-fitted weak-axis surface geometry factor.",
    )
    parser.add_argument(
        "--mapping-keypoints",
        type=int,
        choices=(1024, 2048),
        help=(
            "Lock mapping-observation density only; sparse deployment keeps "
            "the config's independent density policy."
        ),
    )
    return parser


def _validate_arguments(
    args: argparse.Namespace, parser: argparse.ArgumentParser
) -> dict[str, object]:
    experimental_factors = {
        "joint_keypoints": args.keypoints,
        "mapping_keypoints": args.mapping_keypoints,
        "surface_supported_tracks": bool(args.surface_supported_tracks),
    }
    if args.keypoints is not None and args.mapping_keypoints is not None:
        parser.error("--mapping-keypoints cannot be combined with joint --keypoints")
    if args.evaluate and any(bool(value) for value in experimental_factors.values()):
        parser.error("experimental factor flags and --evaluate are mutually exclusive")
    if args.gpu_workers_per_device < 0:
        parser.error("--gpu-workers-per-device cannot be negative")
    args.output = args.output.expanduser().resolve()
    if args.output.exists():
        parser.error(
            "--output must be a fresh, nonexistent root; quarantine partial or stale runs"
        )
    return experimental_factors


def run(args: argparse.Namespace, *, experimental_factors: dict[str, object]) -> dict:
    # Atomically claim the run root.  This closes the preflight/write TOCTOU
    # window and makes direct API callers obey the same fresh-root contract.
    args.output.mkdir(parents=True, exist_ok=False)
    if int(args.gpu_workers_per_device) > 0:
        os.environ["LAFGS_GPU_WORKERS_PER_DEVICE"] = str(
            int(args.gpu_workers_per_device)
        )
    configure_build_timing(args.output / "build_timing.json")
    config = Path(args.config)
    if args.keypoints is not None:
        config = materialize_keypoint_factor_config(
            config,
            args.output / f"factor_config_k{args.keypoints}.yaml",
            args.keypoints,
        )
    if args.mapping_keypoints is not None:
        config = materialize_mapping_keypoint_config(
            config,
            args.output / f"factor_config_mapping_k{args.mapping_keypoints}.yaml",
            args.mapping_keypoints,
        )
    if args.surface_supported_tracks:
        config = materialize_surface_track_config(
            config,
            args.output / "factor_config_surface_tracks.yaml",
        )
    with timed_pipeline_stage("bootstrap_and_tracks"):
        artifacts = build_bootstrap_and_tracks(
            dataset=args.dataset,
            prior=args.prior,
            output=args.output,
            gaussian_type=args.gaussian_type,
            sh_degree=args.sh_degree,
            config=config,
        )
    prior_ply = resolve_prior_ply(args.prior)
    with timed_pipeline_stage("canonical_evidence"):
        evidence = build_evidence(
            base_state=artifacts["base_state"],
            track_payload=artifacts["track_payload"],
            query_cache=artifacts["query_cache"],
            prior_ply=prior_ply,
            gaussian_type=args.gaussian_type,
            sh_degree=args.sh_degree,
            visibility_cache=artifacts["visibility_cache"],
            output=args.output / "evidence",
            config=config,
            valid_masks=args.valid_masks or None,
            function_graph_shards=args.function_graph_shards,
            provenance_shards=args.provenance_shards,
            observation_shards=args.observation_shards,
            scene_calibration=artifacts.get("scene_calibration"),
        )
    with timed_pipeline_stage("selector"):
        compact_map = distill_compact_map(
            canonical_map=evidence["canonical_map"],
            function_graph=evidence["function_graph"],
            positive_teacher=evidence["positive_teacher"],
            track_payload=artifacts["track_payload"],
            query_cache=artifacts["query_cache"],
            output=args.output / "topology",
            config=config,
            pose_scoring_shards=args.pose_scoring_shards,
        )
    with timed_pipeline_stage("compact_evidence_and_metric"):
        trained = train_compact_map(
            compact_map=compact_map,
            function_graph=evidence["function_graph"],
            track_payload=artifacts["track_payload"],
            query_cache=artifacts["query_cache"],
            prior_ply=prior_ply,
            gaussian_type=args.gaussian_type,
            sh_degree=args.sh_degree,
            output=args.output / "map_learning",
            config=config,
            valid_masks=args.valid_masks or None,
            rebuild_function_graph=True,
            function_graph_shards=args.function_graph_shards,
            provenance_shards=args.provenance_shards,
            observation_shards=args.observation_shards,
            scene_calibration=artifacts.get("scene_calibration"),
        )
    outputs = {
        **artifacts,
        **evidence,
        **trained,
        "prior_ply": prior_ply,
        "compact_map": compact_map,
    }
    if args.valid_masks:
        outputs["valid_masks"] = Path(args.valid_masks).expanduser().resolve()
    if args.evaluate:
        deployment = load_mainline_config(config).values["deployment"]
        dataset = ColmapDataset(args.dataset, images="processed")
        test_cameras = dataset.split("test")
        mapping_cameras = dataset.split("mapping")
        calibration_path = (
            Path(trained["trained_map"]).parent / "scene_calibration.json"
        )
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
            nms_radius=int(deployment["nms"]),
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
                    "nms_radius": int(deployment["nms"]),
                    "ransac_reprojection_px": float(reprojection_error_px),
                    "scene_calibration": str(calibration_path.resolve()),
                    "calibration_split": "mapping",
                    "evaluated_split": "test",
                    "pose_solves": 1,
                    "duplicate_anchor_suppression": False,
                    "guided_sampling": False,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        print(json.dumps(result["summary"], indent=2))
        outputs["evaluation"] = args.output / "evaluation"
    selection_provenance = args.output / "topology/adaptive_selection_provenance.pt"
    scene_calibration = trained.get("scene_calibration")
    if scene_calibration is None:
        raise RuntimeError(
            "new pipeline completion requires an explicit mapping-only calibration"
        )
    registry_parents = {
        "trained_map": Path(trained["trained_map"]),
        "compact_map": Path(compact_map),
        "positive_teacher": Path(trained["compact_positive_teacher"]),
        "track_payload": Path(artifacts["track_payload"]),
        "query_cache": Path(artifacts["query_cache"]),
        "raster_provenance": Path(trained["compact_provenance"]),
        "selection_provenance": selection_provenance,
        "scene_calibration": Path(scene_calibration),
        "metric_state": Path(trained["metric_state"]),
        "config": Path(config).expanduser().resolve(),
        "gaussian_ply": Path(prior_ply),
    }
    locked_parents = {
        name: (path, sha256_file(path)) for name, path in registry_parents.items()
    }
    with timed_pipeline_stage("anchor_registry"):
        registry_result = materialize_anchor_registry(
            parents=locked_parents,
            output=Path(trained["trained_map"]).parent / "neutral_anchor_registry.pt",
            contract_output=(
                Path(trained["trained_map"]).parent
                / "neutral_anchor_registry.contract.json"
            ),
            require_pipeline_parents=True,
        )
    outputs["anchor_registry"] = registry_result["registry"]
    outputs["anchor_registry_contract"] = registry_result["contract"]
    outputs["selection_provenance"] = selection_provenance
    outputs["config"] = Path(config).expanduser().resolve()
    outputs["build_timing"] = finalize_build_timing()
    manifest = write_pipeline_manifest(args.output, outputs)
    completion = write_pipeline_completion(
        output=args.output,
        artifacts=outputs,
        pipeline_manifest=manifest,
        anchor_registry_contract=registry_result["contract"],
        config=config,
        evaluation_requested=bool(args.evaluate),
        experimental_factors=experimental_factors,
    )
    print(
        json.dumps(
            {
                "pipeline_completion": completion["path"],
                "pipeline_completion_sha256": completion["sha256"],
                "uses_test_queries": completion["uses_test_queries"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return completion


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    experimental_factors = _validate_arguments(args, parser)
    run(args, experimental_factors=experimental_factors)


if __name__ == "__main__":
    main()
