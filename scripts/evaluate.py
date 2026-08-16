#!/usr/bin/env python3
"""Evaluate a compact LaFGS map with one sparse PoseLib solve per query."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from common.config import (
    load_scene_calibration,
    load_mainline_config,
    resolve_keypoint_count,
    resolve_reprojection_error_px,
)
from data.datasets import ColmapDataset
from evaluation.bootstrap import materialize_a0
from evaluation.evaluator import evaluate_dataset
from localization.localizer import SparseLocalizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--images", default="processed")
    parser.add_argument("--map", type=Path)
    parser.add_argument("--metric-state", type=Path)
    parser.add_argument(
        "--context-state",
        type=Path,
        help="Mapping-only MCCD artifact; mutually exclusive with --metric-state.",
    )
    parser.add_argument(
        "--scene-calibration",
        type=Path,
        help=(
            "Mapping-only calibration JSON. By default the evaluator uses "
            "scene_calibration.json beside the trained map when present."
        ),
    )
    parser.add_argument(
        "--guided-sampling",
        action="store_true",
        help=(
            "Sort unchanged top-1 correspondences by descriptor margin, "
            "mapping matchability, and map uncertainty for one PoseLib PROSAC solve."
        ),
    )
    parser.add_argument(
        "--group-aware-pose",
        action="store_true",
        help=(
            "Add a bounded distinct-parent AP3P hypothesis supplement to the "
            "single robust-pose wrapper; mapping evidence only until validated."
        ),
    )
    parser.add_argument("--group-field", default="parent_source_track_ids")
    parser.add_argument("--group-hypothesis-samples", type=int, default=32)
    parser.add_argument(
        "--stage-state",
        type=Path,
        help="Evaluate A0 by materializing a Stage-A state with an identity metric.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default="configs/paper_mainline.yaml")
    parser.add_argument("--split", choices=("mapping", "test"), default="test")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--suppress-duplicate-anchors",
        action="store_true",
        help=(
            "Keep only the highest-score query match per landmark before the "
            "single PoseLib solve."
        ),
    )
    args = parser.parse_args()
    if args.stage_state:
        if args.map or args.metric_state or args.context_state:
            parser.error("--stage-state cannot be combined with descriptor map options")
        map_path, metric_path = materialize_a0(
            args.stage_state, args.output / "materialized_a0", args.config
        )
    else:
        if args.map is None:
            parser.error("A1 evaluation requires --map")
        if (args.metric_state is None) == (args.context_state is None):
            parser.error(
                "select exactly one descriptor protocol: --metric-state or "
                "--context-state"
            )
        map_path, metric_path = args.map, args.metric_state
    deployment = load_mainline_config(args.config).values["deployment"]
    dataset = ColmapDataset(args.dataset, images=args.images)
    cameras = dataset.split(args.split)
    calibration_cameras = dataset.split("mapping")
    calibration_path = args.scene_calibration
    inferred_calibrations = [map_path.parent / "scene_calibration.json"]
    if args.stage_state is not None:
        inferred_calibrations.insert(
            0, args.stage_state.parent.parent / "scene_calibration.json"
        )
    if calibration_path is None:
        calibration_path = next(
            (path for path in inferred_calibrations if path.is_file()), None
        )
    scene_calibration = (
        load_scene_calibration(calibration_path)
        if calibration_path is not None
        else None
    )
    keypoint_count = resolve_keypoint_count(deployment, calibration_cameras)
    reprojection_error_px = resolve_reprojection_error_px(
        deployment, calibration_cameras, scene_calibration
    )
    localizer = SparseLocalizer(
        map_path,
        metric_path,
        context_state_path=args.context_state,
        device=args.device,
        keypoint_count=keypoint_count,
        nms_radius=int(deployment["nms"]),
        reprojection_error_px=reprojection_error_px,
        confidence=deployment["confidence"],
        max_iterations=deployment["maximum_iterations"],
        min_iterations=deployment["minimum_iterations"],
        seed=args.seed,
        suppress_duplicate_anchors=args.suppress_duplicate_anchors,
        guided_sampling=args.guided_sampling,
        group_aware_pose=args.group_aware_pose,
        group_field=args.group_field,
        group_hypothesis_samples=args.group_hypothesis_samples,
    )
    result = evaluate_dataset(
        dataset=dataset,
        localizer=localizer,
        cameras=cameras,
        output=args.output,
    )
    (args.output / "deployment_contract.json").write_text(
        json.dumps(
            {
                "schema": "lafgs_sparse_deployment_contract",
                "version": 1,
                "keypoint_count": int(keypoint_count),
                "nms_radius": int(deployment["nms"]),
                "ransac_reprojection_px": float(reprojection_error_px),
                "scene_calibration": (
                    str(calibration_path.resolve())
                    if calibration_path is not None
                    else None
                ),
                "calibration_split": "mapping",
                "evaluated_split": args.split,
                "pose_solves": 1,
                "duplicate_anchor_suppression": bool(args.suppress_duplicate_anchors),
                "guided_sampling": bool(args.guided_sampling),
                "group_aware_pose": bool(args.group_aware_pose),
                "group_field": args.group_field if args.group_aware_pose else None,
                "group_hypothesis_samples": (
                    int(args.group_hypothesis_samples) if args.group_aware_pose else 0
                ),
                "descriptor_protocol": (
                    "mccd" if args.context_state is not None else "shared_metric"
                ),
                "context_state": (
                    str(args.context_state.resolve())
                    if args.context_state is not None
                    else None
                ),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
