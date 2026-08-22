#!/usr/bin/env python3
"""Create one independent V6 descriptor/selection/reconstruction proposal arm."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess

import torch

from common.hashing import sha256_file
from common.v6_contracts import ASSOCIATION_GRAPH_SCHEMA, FEEDBACK_SCHEMA, require_schema
from evidence.observation_provider import GaussianRenderObservationProvider
from evidence.projective_completion import build_projective_completion
from map_learning.v6_proposals import descriptor_only_proposal, selection_only_proposal
from topology.v6_anchor_map import (
    identity_metric_state,
    materialize_projective_anchor_map,
    merge_projective_candidates,
    projective_candidates_from_map,
)


def _clean_commit() -> str:
    root = Path(__file__).resolve().parents[1]
    dirty = subprocess.run(
        ["git", "status", "--porcelain=v1"], cwd=root, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    if dirty:
        raise RuntimeError("V6 proposal producer must be clean")
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True,
        capture_output=True, text=True,
    ).stdout.strip()


def _load(path: Path, expected: str, label: str) -> tuple[dict, str]:
    path = path.resolve()
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"{label} SHA differs")
    return torch.load(path, map_location="cpu", weights_only=False), actual


def _save(value: dict, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _jsonable(value):
    if isinstance(value, torch.Tensor):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def run(args: argparse.Namespace) -> dict:
    commit = _clean_commit()
    state, map_sha = _load(args.map, args.expected_map_sha256, "map")
    cache, cache_sha = _load(
        args.observation_cache,
        args.expected_observation_cache_sha256,
        "observation cache",
    )
    evaluation, feedback_sha = _load(
        args.feedback,
        args.expected_feedback_sha256,
        "feedback",
    )
    feedback = evaluation.get("feedback", evaluation)
    require_schema(feedback, FEEDBACK_SCHEMA, label="feedback")
    if feedback["input_sha256"]["map"] != map_sha:
        raise ValueError("feedback is not bound to the source map")
    observations = GaussianRenderObservationProvider(cache)
    arm = args.arm
    selection_report = None
    unavailable_reason = None
    if arm in {"descriptor", "descriptor_selection"}:
        proposal = descriptor_only_proposal(
            state, observations, feedback, trust_region=args.descriptor_trust_region
        )
        if arm == "descriptor_selection":
            proposal, selection_report = selection_only_proposal(
                proposal,
                feedback,
                maximum_anchors=args.maximum_anchors,
                matching_target=args.matching_target,
                pose_logdet_target=args.pose_logdet_target,
            )
    elif arm == "selection":
        proposal, selection_report = selection_only_proposal(
            state,
            feedback,
            maximum_anchors=args.maximum_anchors,
            matching_target=args.matching_target,
            pose_logdet_target=args.pose_logdet_target,
        )
    elif arm == "reconstruction":
        association, _ = _load(
            args.association_graph,
            args.expected_association_graph_sha256,
            "association graph",
        )
        require_schema(
            association, ASSOCIATION_GRAPH_SCHEMA, label="association graph"
        )
        l1_queries = [
            record["query_index"]
            for record in feedback["records"]
            if record["failure_layer"] == "L1"
        ]
        if not l1_queries:
            raise ValueError("reconstruction proposal has no L1 query")
        try:
            completion = build_projective_completion(
                observations,
                association,
                voxel_size_m=args.completion_voxel_size_m,
                alpha_minimum=args.alpha_minimum,
                minimum_similarity=args.completion_minimum_similarity,
                minimum_margin=args.minimum_margin,
                maximum_epipolar_error_px=args.maximum_epipolar_error_px,
                minimum_observations=args.minimum_views,
                minimum_camera_families=args.minimum_camera_families,
                maximum_rows_per_view=args.completion_maximum_rows_per_view,
                safety_maximum_components=args.completion_safety_maximum_components,
                eligible_query_indices=l1_queries,
                device=args.device,
            )
        except ValueError as error:
            if str(error) != "depth proposals produced no descriptor-consistent component":
                raise
            proposal = None
            unavailable_reason = "no_descriptor_consistent_projective_completion"
        if unavailable_reason is None:
            merged = merge_projective_candidates(
                [projective_candidates_from_map(state), completion]
            )
            proposal = materialize_projective_anchor_map(
                merged,
                lineage={
                    **dict(state.get("provenance", {})),
                    "v6_parent_map_sha256": map_sha,
                    "v6_reconstruction_feedback_sha256": feedback_sha,
                    "v6_l1_query_count": len(l1_queries),
                },
            )
    else:
        raise ValueError(f"unknown proposal arm {arm}")
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    if proposal is None:
        report = {
            "schema": "lafgs_v6_round_proposal",
            "version": 1,
            "uses_source_mapping_rgb": False,
            "uses_test_queries": False,
            "arm": arm,
            "proposal_available": False,
            "unavailable_reason": unavailable_reason,
            "producer_git_commit": commit,
            "input_sha256": {
                "map": map_sha,
                "observation_cache": cache_sha,
                "feedback": feedback_sha,
            },
            "anchor_count": int(torch.as_tensor(state["anchor_ids"]).numel()),
        }
        (args.output_dir / "proposal.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n"
        )
        return report
    map_path = args.output_dir / "proposal_map.pt"
    _save(proposal, map_path)
    proposal_sha = sha256_file(map_path)
    metric = identity_metric_state(
        proposal, map_path=str(map_path.resolve()), map_sha256=proposal_sha
    )
    metric_path = args.output_dir / "identity_metric.pt"
    _save(metric, metric_path)
    report = {
        "schema": "lafgs_v6_round_proposal",
        "version": 1,
        "uses_source_mapping_rgb": False,
        "uses_test_queries": False,
        "arm": arm,
        "proposal_available": True,
        "producer_git_commit": commit,
        "input_sha256": {
            "map": map_sha,
            "observation_cache": cache_sha,
            "feedback": feedback_sha,
        },
        "output": {
            "map": str(map_path.resolve()),
            "map_sha256": proposal_sha,
            "metric": str(metric_path.resolve()),
            "metric_sha256": sha256_file(metric_path),
        },
        "anchor_count": int(torch.as_tensor(proposal["anchor_ids"]).numel()),
        "selection_report": _jsonable(selection_report),
    }
    (args.output_dir / "proposal.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("descriptor", "selection", "descriptor_selection", "reconstruction"), required=True)
    parser.add_argument("--map", type=Path, required=True)
    parser.add_argument("--expected-map-sha256", required=True)
    parser.add_argument("--observation-cache", type=Path, required=True)
    parser.add_argument("--expected-observation-cache-sha256", required=True)
    parser.add_argument("--feedback", type=Path, required=True)
    parser.add_argument("--expected-feedback-sha256", required=True)
    parser.add_argument("--association-graph", type=Path)
    parser.add_argument("--expected-association-graph-sha256")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--descriptor-trust-region", type=float, default=0.05)
    parser.add_argument("--maximum-anchors", type=int, default=20000)
    parser.add_argument("--matching-target", type=int, default=4)
    parser.add_argument("--pose-logdet-target", type=float, default=0.0)
    parser.add_argument("--completion-voxel-size-m", type=float, default=0.05)
    parser.add_argument("--alpha-minimum", type=float, default=0.05)
    parser.add_argument("--completion-minimum-similarity", type=float, default=0.7)
    parser.add_argument("--minimum-margin", type=float, default=0.01)
    parser.add_argument("--maximum-epipolar-error-px", type=float, default=2.0)
    parser.add_argument("--minimum-views", type=int, default=3)
    parser.add_argument("--minimum-camera-families", type=int, default=2)
    parser.add_argument("--completion-maximum-rows-per-view", type=int, default=256)
    parser.add_argument("--completion-safety-maximum-components", type=int, default=100000)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
