#!/usr/bin/env python3
"""Run formal V6 mapping self-localization and serialize L1--L4 feedback."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess

import torch

from common.hashing import sha256_file
from common.v6_contracts import (
    FEEDBACK_VERSION,
    RENDER_OBSERVATION_SCHEMA,
    ordered_query_registry_sha256,
    require_schema,
)
from common.v6_pipeline_contract import (
    FORMAL_FEEDBACK_CANDIDATE_ARMS,
    resolve_v6_feedback_calibration_map_lineage,
    validate_v6_feedback_calibration_binding,
    validate_v6_feedback_scene_calibration,
)
from evidence.observation_provider import GaussianRenderObservationProvider
from map_learning.v6_feedback_evaluator import evaluate_query_local_feedback
from topology.v6_anchor_map import validate_v6_identity_metric


_SOURCE_PATHS = (
    "scripts/evaluate_v6_self_localization.py",
    "common/v6_contracts.py",
    "common/v6_pipeline_contract.py",
    "evidence/observation_provider.py",
    "map_learning/v6_feedback_evaluator.py",
    "map_learning/self_localization_feedback.py",
    "evidence/projective_loo.py",
    "evidence/projective_reconstruction.py",
    "localization/matcher.py",
    "localization/pose_solver.py",
    "topology/layered_sufficiency.py",
    "topology/pose_information.py",
)


def _producer() -> dict:
    root = Path(__file__).resolve().parents[1]
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain=v1"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if dirty:
        raise RuntimeError("V6 feedback evaluator requires a clean worktree")
    return {
        "git_commit": commit,
        "worktree_clean": True,
        "source_sha256": {path: sha256_file(root / path) for path in _SOURCE_PATHS},
        "torch_version": torch.__version__,
    }


def _require(path: Path, expected: str, label: str) -> str:
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"{label} SHA differs")
    return actual


def run(args: argparse.Namespace) -> dict:
    if int(args.cpu_threads) < 1:
        raise ValueError("CPU thread count must be positive")
    torch.set_num_threads(int(args.cpu_threads))
    os.environ["OMP_NUM_THREADS"] = str(int(args.cpu_threads))
    os.environ["MKL_NUM_THREADS"] = str(int(args.cpu_threads))
    producer = _producer()
    paths = {
        "map": args.map.resolve(),
        "metric": args.metric.resolve(),
        "observation_cache": args.observation_cache.resolve(),
        "scene_calibration": args.scene_calibration.resolve(),
        "feedback_calibration_binding": args.feedback_calibration_binding.resolve(),
    }
    hashes = {
        "map": _require(paths["map"], args.expected_map_sha256, "map"),
        "metric": _require(paths["metric"], args.expected_metric_sha256, "metric"),
        "observation_cache": _require(
            paths["observation_cache"],
            args.expected_observation_cache_sha256,
            "observation cache",
        ),
        "scene_calibration": _require(
            paths["scene_calibration"],
            args.expected_scene_calibration_sha256,
            "scene calibration",
        ),
        "feedback_calibration_binding": _require(
            paths["feedback_calibration_binding"],
            args.expected_feedback_calibration_binding_sha256,
            "feedback calibration binding",
        ),
    }
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    state = torch.load(paths["map"], map_location="cpu", weights_only=False)
    metric = torch.load(paths["metric"], map_location="cpu", weights_only=False)
    cache = torch.load(
        paths["observation_cache"], map_location="cpu", weights_only=False
    )
    calibration = json.loads(paths["scene_calibration"].read_text())
    calibration_binding = json.loads(
        paths["feedback_calibration_binding"].read_text()
    )
    if not isinstance(calibration, dict):
        raise ValueError("scene calibration is not a JSON object")
    if not isinstance(calibration_binding, dict):
        raise ValueError("feedback calibration binding is not a JSON object")
    require_schema(cache, RENDER_OBSERVATION_SCHEMA, label="V6 observations")
    cache_names = list(cache.get("query_names", cache.get("queries", {})))
    query_registry_sha256 = ordered_query_registry_sha256(cache_names)
    if list(state.get("v6_mapping_query_names", ())) != cache_names:
        raise ValueError("V6 map and observation query registries differ")
    calibration_map_lineage = resolve_v6_feedback_calibration_map_lineage(
        state=state,
        current_map_sha256=hashes["map"],
        candidate_parent_map_sha256=args.candidate_parent_map_sha256,
        candidate_arm=args.candidate_arm,
    )
    calibrated_ransac_reprojection_px = validate_v6_feedback_scene_calibration(
        calibration,
        query_count=len(cache_names),
    )
    if float(args.ransac_reprojection_px) != calibrated_ransac_reprojection_px:
        raise ValueError(
            "RANSAC threshold differs from scene calibration: "
            f"{float(args.ransac_reprojection_px)!r} != "
            f"{calibrated_ransac_reprojection_px!r}"
        )
    validate_v6_feedback_calibration_binding(
        calibration_binding,
        map_sha256=calibration_map_lineage[
            "calibration_binding_source_map_sha256"
        ],
        observation_cache_sha256=hashes["observation_cache"],
        calibration_sha256=hashes["scene_calibration"],
        query_registry_sha256=query_registry_sha256,
        query_count=len(cache_names),
    )
    validate_v6_identity_metric(
        metric,
        state=state,
        map_path=str(paths["map"]),
        map_sha256=hashes["map"],
    )
    observations = GaussianRenderObservationProvider(cache, query_names=cache_names)
    result = evaluate_query_local_feedback(
        state=state,
        observations=observations,
        source_map_sha256=hashes["map"],
        query_cache_sha256=hashes["observation_cache"],
        scene_calibration_sha256=hashes["scene_calibration"],
        feedback_calibration_binding_sha256=hashes[
            "feedback_calibration_binding"
        ],
        device=torch.device(args.device),
        positive_radius_px=args.positive_radius_px,
        alpha_minimum=args.alpha_minimum,
        required_rank=args.required_rank,
        ransac_reprojection_px=args.ransac_reprojection_px,
        seed=args.seed,
        loo_pose_neighbors=args.loo_pose_neighbors,
        required_visibility_rank=args.required_visibility_rank,
        required_detectable_rank=args.required_detectable_rank,
        loo_affected_anchor_policy=args.loo_affected_anchor_policy,
        pose_logdet_target=args.pose_logdet_target,
        pose_min_eigenvalue_target=args.pose_min_eigenvalue_target,
    )
    result["producer"] = producer
    result["input_sha256"] = hashes
    result["map_path"] = str(paths["map"])
    result["contract"].update(calibration_map_lineage)
    temporary = args.output_dir / f".feedback.{os.getpid()}.tmp"
    output = args.output_dir / "feedback.pt"
    try:
        torch.save(result, temporary)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    summary = {
        "schema": "lafgs_v6_query_local_feedback_summary",
        "version": FEEDBACK_VERSION,
        "uses_source_mapping_rgb": False,
        "uses_test_queries": False,
        "summary": result["summary"],
        "descriptor_validation_summary": result["descriptor_validation_summary"],
        "independent_mapping_validation_summary": result[
            "independent_mapping_validation_summary"
        ],
        "descriptor_training_replay_summary": result[
            "descriptor_training_replay_summary"
        ],
        "descriptor_gradient_reuse_summary": result[
            "descriptor_gradient_reuse_summary"
        ],
        "reconstruction_target_replay_summary": result[
            "reconstruction_target_replay_summary"
        ],
        "selection_training_replay_summary": result[
            "selection_training_replay_summary"
        ],
        "failure_layer_counts": result["feedback"]["failure_layer_counts"],
        "failure_layer_counts_are_overlapping": result["feedback"][
            "failure_layer_counts_are_overlapping"
        ],
        "failure_query_count": result["feedback"]["failure_query_count"],
        "multi_layer_failure_query_count": result["feedback"][
            "multi_layer_failure_query_count"
        ],
        "contract": result["contract"],
        "feedback_path": str(output.resolve()),
        "feedback_sha256": sha256_file(output),
        "producer": producer,
        "input_sha256": hashes,
        "cpu_threads": int(args.cpu_threads),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", type=Path, required=True)
    parser.add_argument("--expected-map-sha256", required=True)
    parser.add_argument("--metric", type=Path, required=True)
    parser.add_argument("--expected-metric-sha256", required=True)
    parser.add_argument("--observation-cache", type=Path, required=True)
    parser.add_argument("--expected-observation-cache-sha256", required=True)
    parser.add_argument("--scene-calibration", type=Path, required=True)
    parser.add_argument("--expected-scene-calibration-sha256", required=True)
    parser.add_argument("--feedback-calibration-binding", type=Path, required=True)
    parser.add_argument(
        "--expected-feedback-calibration-binding-sha256", required=True
    )
    parser.add_argument("--candidate-parent-map-sha256")
    parser.add_argument("--candidate-arm", choices=FORMAL_FEEDBACK_CANDIDATE_ARMS)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--positive-radius-px", type=float, default=2.0)
    parser.add_argument("--alpha-minimum", type=float, default=0.05)
    parser.add_argument("--required-rank", type=int, default=16)
    parser.add_argument("--required-visibility-rank", type=int, default=4)
    parser.add_argument("--required-detectable-rank", type=int, default=16)
    parser.add_argument("--loo-pose-neighbors", type=int, default=3)
    parser.add_argument(
        "--loo-affected-anchor-policy",
        choices=("purge", "rebuild"),
        default="rebuild",
        help="rebuild is required for descriptor identity supervision; purge is diagnostic-only",
    )
    parser.add_argument("--ransac-reprojection-px", type=float, required=True)
    parser.add_argument("--pose-logdet-target", type=float, default=0.0)
    parser.add_argument("--pose-min-eigenvalue-target", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=2026)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
