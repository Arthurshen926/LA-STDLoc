#!/usr/bin/env python3
"""Evaluate a frozen render-only closed-loop map on real test RGB."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from common.config import (
    load_mainline_config,
    load_scene_calibration,
    resolve_keypoint_count,
    resolve_reprojection_error_px,
)
from common.hashing import sha256_file
from common.evaluation_code import mapping_pose_evaluation_code_identity
from data.datasets import ColmapDataset
from evaluation.evaluator import evaluate_dataset
from localization.localizer import SparseLocalizer


def _require_sha(path: Path, expected: str, label: str) -> str:
    actual = sha256_file(path)
    if actual != str(expected):
        raise ValueError(f"{label} SHA differs: expected {expected}, got {actual}")
    return actual


def load_closed_loop_selection(path: Path, expected_sha256: str) -> dict:
    path = path.resolve()
    artifact_sha256 = _require_sha(path, expected_sha256, "closed-loop selection")
    selection = json.loads(path.read_text())
    if (
        selection.get("schema") != "lafgs_rendered_track_closed_loop_selection"
        or selection.get("valid") is not True
        or selection.get("uses_source_mapping_rgb") is not False
        or selection.get("uses_test_queries") is not False
    ):
        raise ValueError("closed-loop selection schema or construction split differs")
    authorization = selection.get("authorization", {})
    if (
        authorization.get("mapping_selection_complete") is not True
        or authorization.get("test_may_be_used_only_for_frozen_final_evaluation")
        is not True
        or authorization.get("test_may_change_map_or_selection") is not False
    ):
        raise ValueError("closed-loop selection does not authorize frozen test")
    artifacts = selection.get("selected_artifacts", {})
    hashes = selection.get("selected_artifact_sha256", {})
    required = {"map", "metric", "teacher", "query_cache", "scene_calibration"}
    if not required.issubset(artifacts) or not required.issubset(hashes):
        raise ValueError("closed-loop selection misses a selected artifact")
    resolved = {}
    for role in sorted(required):
        artifact = Path(str(artifacts[role])).resolve()
        _require_sha(artifact, str(hashes[role]), f"selected {role}")
        resolved[role] = artifact
    return {
        "path": path,
        "sha256": artifact_sha256,
        "payload": selection,
        "artifacts": resolved,
    }


def run(args: argparse.Namespace) -> dict:
    evaluation_code = mapping_pose_evaluation_code_identity(require_clean=True)
    closed_loop = load_closed_loop_selection(
        args.closed_loop_selection, args.expected_closed_loop_selection_sha256
    )
    paths = closed_loop["artifacts"]
    if args.output.exists():
        raise FileExistsError(args.output)
    deployment = load_mainline_config(args.config).values["deployment"]
    dataset = ColmapDataset(args.dataset, images=args.images)
    test_cameras = dataset.split("test")
    mapping_cameras = dataset.split("mapping")
    scene_calibration = load_scene_calibration(paths["scene_calibration"])
    keypoint_count = resolve_keypoint_count(deployment, mapping_cameras)
    reprojection_error_px = resolve_reprojection_error_px(
        deployment, mapping_cameras, scene_calibration
    )
    localizer = SparseLocalizer(
        paths["map"],
        paths["metric"],
        device=args.device,
        keypoint_count=keypoint_count,
        nms_radius=int(deployment["nms"]),
        reprojection_error_px=reprojection_error_px,
        confidence=deployment["confidence"],
        max_iterations=deployment["maximum_iterations"],
        min_iterations=deployment["minimum_iterations"],
        seed=int(args.seed),
    )
    result = evaluate_dataset(
        dataset=dataset,
        localizer=localizer,
        cameras=test_cameras,
        output=args.output,
    )
    _require_sha(closed_loop["path"], closed_loop["sha256"], "closed-loop selection")
    for role, path in paths.items():
        _require_sha(
            path,
            closed_loop["payload"]["selected_artifact_sha256"][role],
            f"selected {role}",
        )
    if mapping_pose_evaluation_code_identity(require_clean=True) != evaluation_code:
        raise RuntimeError("test evaluation code identity changed during execution")
    contract = {
        "schema": "lafgs_rendered_track_closed_loop_test_contract",
        "version": 1,
        "uses_source_mapping_rgb": False,
        "evaluation_code": evaluation_code,
        "evaluated_split": "test",
        "test_used_for_construction_selection_or_calibration": False,
        "closed_loop_selection": str(closed_loop["path"]),
        "closed_loop_selection_sha256": closed_loop["sha256"],
        "closed_loop_decision": closed_loop["payload"]["decision"],
        "selected_label": closed_loop["payload"]["selected_label"],
        "selected_artifacts": {role: str(path) for role, path in paths.items()},
        "selected_artifact_sha256": closed_loop["payload"]["selected_artifact_sha256"],
        "seed": int(args.seed),
        "deployment": {
            "keypoint_count": int(keypoint_count),
            "nms_radius": int(deployment["nms"]),
            "ransac_reprojection_px": float(reprojection_error_px),
            "pose_solves_per_query": 1,
            "one_global_top1_per_query_row": True,
        },
        "summary": result["summary"],
    }
    contract_path = args.output / "rendered_track_closed_loop_contract.json"
    contract_path.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n")
    return contract


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--closed-loop-selection", type=Path, required=True)
    parser.add_argument("--expected-closed-loop-selection-sha256", required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--images", default="processed")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default="configs/paper_mainline.yaml")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
