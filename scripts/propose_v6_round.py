#!/usr/bin/env python3
"""Create one independent V6 descriptor/selection/reconstruction proposal arm."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

import torch

from common.hashing import sha256_file
from common.v6_contracts import (
    ASSOCIATION_GRAPH_SCHEMA,
    DESCRIPTOR_SPLIT_SCHEMA,
    FEEDBACK_SCHEMA,
    RENDER_OBSERVATION_SCHEMA,
    ordered_query_registry_sha256,
    require_mapping_only,
    require_schema,
)
from evidence.observation_provider import GaussianRenderObservationProvider
from evidence.projective_completion import build_projective_completion
from map_learning.v6_proposals import (
    descriptor_loss_proposal,
    descriptor_only_proposal,
    selection_only_proposal,
)
from topology.v6_anchor_map import (
    compact_projective_deployment_map,
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


def _distillation_summary(value):
    if not isinstance(value, dict):
        return None
    result = dict(value)
    for rows_field, count_field in (
        ("updated_anchor_rows", "updated_anchor_count"),
        ("round_updated_anchor_rows", "round_updated_anchor_count"),
    ):
        rows = result.pop(rows_field, None)
        if rows is not None:
            result.setdefault(count_field, int(torch.as_tensor(rows).numel()))
    return _jsonable(result)


def _validate_proposal_inputs(
    *,
    state: dict,
    cache: dict,
    feedback: dict,
    map_sha: str,
    cache_sha: str,
) -> None:
    require_schema(cache, RENDER_OBSERVATION_SCHEMA, label="observation cache")
    require_schema(feedback, FEEDBACK_SCHEMA, label="feedback")
    require_mapping_only(state.get("provenance", {}), label="source map")
    if state.get("provenance", {}).get("v6_compact_deployment_export") is True:
        raise ValueError("compact deployment maps cannot produce another V6 proposal")
    input_sha = feedback.get("input_sha256", {})
    if input_sha.get("map") != map_sha:
        raise ValueError("feedback is not bound to the source map")
    if input_sha.get("query_cache") != cache_sha:
        raise ValueError("feedback is not bound to the observation cache")
    cache_names = list(cache.get("query_names", cache.get("queries", {})))
    if list(feedback.get("query_names", ())) != cache_names:
        raise ValueError("feedback and observation cache registries differ")


def _load_query_indices(
    path: Path | None,
    expected_sha256: str | None,
    *,
    feedback_sha256: str | None = None,
    query_names: list[str] | None = None,
    require_source_feedback_match: bool = False,
) -> tuple[list[int] | None, str | None]:
    if path is None:
        if expected_sha256 is not None:
            raise ValueError("descriptor training split SHA has no input file")
        return None, None
    if expected_sha256 is None:
        raise ValueError("descriptor training split requires an expected SHA")
    serialized = path.read_bytes()
    actual_sha256 = hashlib.sha256(serialized).hexdigest()
    if actual_sha256 != expected_sha256:
        raise ValueError("descriptor training split SHA differs")
    value = json.loads(serialized)
    if not isinstance(value, dict):
        raise ValueError("descriptor training split must be a bound document")
    require_schema(value, DESCRIPTOR_SPLIT_SCHEMA, label="descriptor training split")
    if (
        require_source_feedback_match
        and value.get("source_feedback_sha256") != feedback_sha256
    ):
        raise ValueError("descriptor training split is not bound to the feedback")
    if query_names is None or value.get(
        "query_names_sha256"
    ) != ordered_query_registry_sha256(query_names):
        raise ValueError("descriptor training split query registry differs")
    training = value.get("training_query_indices")
    validation = value.get("validation_query_indices")
    if not isinstance(training, list) or not isinstance(validation, list):
        raise ValueError("descriptor training split indices must be lists")
    rows = [int(row) for row in training]
    held_out = [int(row) for row in validation]
    expected_rows = list(range(len(query_names)))
    if (
        rows != sorted(set(rows))
        or held_out != sorted(set(held_out))
        or sorted(rows + held_out) != expected_rows
    ):
        raise ValueError("descriptor training split is not an exact partition")
    return rows, actual_sha256


def _attach_reconstruction_distillation(
    proposal: dict,
    state: dict,
    completion: dict,
    *,
    target_query_indices: list[int],
    excluded_support_query_indices: list[int],
) -> None:
    for field in ("v6_descriptor_distillation", "v6_selection_distillation"):
        report = state.get(field)
        if isinstance(report, dict):
            proposal[field] = dict(report)
    prior = state.get("v6_reconstruction_distillation")
    prior_targets = torch.as_tensor(
        prior.get("target_query_indices", ()) if isinstance(prior, dict) else (),
        dtype=torch.long,
    ).reshape(-1)
    targets = torch.unique(
        torch.cat(
            (prior_targets, torch.tensor(target_query_indices, dtype=torch.long))
        ),
        sorted=True,
    )
    prior_excluded = torch.as_tensor(
        prior.get("excluded_support_query_indices", ())
        if isinstance(prior, dict)
        else (),
        dtype=torch.long,
    ).reshape(-1)
    excluded = torch.unique(
        torch.cat(
            (
                prior_excluded,
                torch.tensor(excluded_support_query_indices, dtype=torch.long),
            )
        ),
        sorted=True,
    )
    proposal["v6_reconstruction_distillation"] = {
        "schema": "lafgs_v6_target_seeded_projective_completion",
        "version": 1,
        "target_query_indices": targets,
        "round_target_query_indices": torch.tensor(
            target_query_indices, dtype=torch.long
        ),
        "excluded_support_query_indices": excluded,
        "round_excluded_support_query_indices": torch.tensor(
            excluded_support_query_indices, dtype=torch.long
        ),
        "target_query_depth_used_for_seed_region": True,
        "target_queries_used_as_anchor_support": False,
        "final_xyz_source": "fixed_camera_robust_ray_triangulation",
        "completion_contract": dict(completion.get("contract", {})),
        "reconstruction_round": int(
            prior.get("reconstruction_round", 0)
            if isinstance(prior, dict)
            else 0
        )
        + 1,
    }


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
    _validate_proposal_inputs(
        state=state,
        cache=cache,
        feedback=feedback,
        map_sha=map_sha,
        cache_sha=cache_sha,
    )
    observations = GaussianRenderObservationProvider(cache)
    arm = args.arm
    descriptor_training_queries = None
    descriptor_training_split_sha = None
    if arm == "descriptor":
        proposal = descriptor_only_proposal(
            state,
            observations,
            feedback,
            trust_region=args.descriptor_trust_region,
        )
    elif arm in {"descriptor_loss", "descriptor_selection", "selection"}:
        (
            descriptor_training_queries,
            descriptor_training_split_sha,
        ) = _load_query_indices(
            args.descriptor_training_query_indices,
            args.expected_descriptor_training_query_indices_sha256,
            feedback_sha256=feedback_sha,
            query_names=list(feedback["query_names"]),
            require_source_feedback_match=not any(
                isinstance(state.get(field), dict)
                for field in (
                    "v6_descriptor_distillation",
                    "v6_reconstruction_distillation",
                    "v6_selection_distillation",
                )
            ),
        )
    selection_report = None
    unavailable_reason = None
    association_sha = None
    if arm == "descriptor":
        pass
    elif arm in {"descriptor_loss", "descriptor_selection"}:
        try:
            proposal = descriptor_loss_proposal(
                state,
                observations,
                feedback,
                trust_region=args.descriptor_trust_region,
                margin=args.descriptor_margin,
                temperature=args.descriptor_temperature,
                learning_rate=args.descriptor_learning_rate,
                epochs=args.descriptor_epochs,
                batch_size=args.descriptor_batch_size,
                maximum_triplets_per_query=args.descriptor_maximum_triplets_per_query,
                clean_fraction=args.descriptor_clean_fraction,
                clean_weight=args.descriptor_clean_weight,
                trust_weight=args.descriptor_trust_weight,
                training_query_indices=descriptor_training_queries,
                device=args.device,
            )
        except ValueError as error:
            if str(error) != "feedback contains no trainable descriptor triplets":
                raise
            proposal = None
            unavailable_reason = "no_trainable_l3_descriptor_triplets"
        if arm == "descriptor_selection" and proposal is not None:
            proposal, selection_report = selection_only_proposal(
                proposal,
                feedback,
                maximum_anchors=args.maximum_anchors,
                visibility_target=args.visibility_target,
                detectability_target=args.detectability_target,
                matching_target=args.matching_target,
                pose_logdet_target=args.pose_logdet_target,
                pose_min_eigenvalue_target=args.pose_min_eigenvalue_target,
                training_query_indices=descriptor_training_queries,
            )
        if proposal is not None and descriptor_training_split_sha is not None:
            report = dict(proposal["v6_descriptor_distillation"])
            prior_split_shas = list(
                report.get("training_split_artifact_sha256s", ())
            )
            if descriptor_training_split_sha not in prior_split_shas:
                prior_split_shas.append(descriptor_training_split_sha)
            report["training_split_artifact_sha256s"] = prior_split_shas
            proposal["v6_descriptor_distillation"] = report
    elif arm == "selection":
        proposal, selection_report = selection_only_proposal(
            state,
            feedback,
            maximum_anchors=args.maximum_anchors,
            visibility_target=args.visibility_target,
            detectability_target=args.detectability_target,
            matching_target=args.matching_target,
            pose_logdet_target=args.pose_logdet_target,
            pose_min_eigenvalue_target=args.pose_min_eigenvalue_target,
            training_query_indices=descriptor_training_queries,
        )
    elif arm == "reconstruction":
        if (
            args.association_graph is None
            or args.expected_association_graph_sha256 is None
        ):
            raise ValueError("reconstruction requires a SHA-bound association graph")
        association, association_sha = _load(
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
            if "L1"
            in record.get("failure_layers", (record.get("failure_layer"),))
        ]
        if not l1_queries:
            raise ValueError("reconstruction proposal has no L1 query")
        excluded_support_queries = sorted(
            {
                int(query_index)
                for record in feedback["records"]
                if "L1"
                in record.get("failure_layers", (record.get("failure_layer"),))
                for query_index in torch.as_tensor(
                    record.get("excluded_query_indices", (record["query_index"],))
                ).tolist()
            }
        )
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
                target_query_indices=l1_queries,
                excluded_support_query_indices=excluded_support_queries,
                device=args.device,
            )
        except ValueError as error:
            unavailable = {
                "no unused render-valid observations for completion": (
                    "no_unused_render_valid_completion_observations"
                ),
                "target queries produced no completion seed region": (
                    "no_target_completion_seed_region"
                ),
                "depth proposals produced no descriptor-consistent component": (
                    "no_descriptor_consistent_projective_completion"
                ),
                "association graph contains no ray-triangulated Anchor": (
                    "no_ray_triangulated_completion_anchor"
                ),
            }
            if str(error) not in unavailable:
                raise
            proposal = None
            unavailable_reason = unavailable[str(error)]
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
            _attach_reconstruction_distillation(
                proposal,
                state,
                completion,
                target_query_indices=l1_queries,
                excluded_support_query_indices=excluded_support_queries,
            )
    else:
        raise ValueError(f"unknown proposal arm {arm}")
    if (
        proposal is not None
        and selection_report is not None
        and descriptor_training_split_sha is not None
    ):
        report = dict(proposal["v6_selection_distillation"])
        prior_split_shas = list(report.get("training_split_artifact_sha256s", ()))
        if descriptor_training_split_sha not in prior_split_shas:
            prior_split_shas.append(descriptor_training_split_sha)
        report["training_split_artifact_sha256s"] = prior_split_shas
        proposal["v6_selection_distillation"] = report
    proposal_configuration = {
        "device": args.device,
        "descriptor_trust_region": float(args.descriptor_trust_region),
        "descriptor_margin": float(args.descriptor_margin),
        "descriptor_temperature": float(args.descriptor_temperature),
        "descriptor_learning_rate": float(args.descriptor_learning_rate),
        "descriptor_epochs": int(args.descriptor_epochs),
        "descriptor_batch_size": int(args.descriptor_batch_size),
        "descriptor_maximum_triplets_per_query": int(
            args.descriptor_maximum_triplets_per_query
        ),
        "descriptor_clean_fraction": float(args.descriptor_clean_fraction),
        "descriptor_clean_weight": float(args.descriptor_clean_weight),
        "descriptor_trust_weight": float(args.descriptor_trust_weight),
        "maximum_anchors": int(args.maximum_anchors),
        "visibility_target": int(args.visibility_target),
        "detectability_target": int(args.detectability_target),
        "matching_target": int(args.matching_target),
        "pose_logdet_target": float(args.pose_logdet_target),
        "pose_min_eigenvalue_target": float(args.pose_min_eigenvalue_target),
        "completion_voxel_size_m": float(args.completion_voxel_size_m),
        "alpha_minimum": float(args.alpha_minimum),
        "completion_minimum_similarity": float(
            args.completion_minimum_similarity
        ),
        "minimum_margin": float(args.minimum_margin),
        "maximum_epipolar_error_px": float(args.maximum_epipolar_error_px),
        "minimum_views": int(args.minimum_views),
        "minimum_camera_families": int(args.minimum_camera_families),
        "completion_maximum_rows_per_view": int(
            args.completion_maximum_rows_per_view
        ),
        "completion_safety_maximum_components": int(
            args.completion_safety_maximum_components
        ),
    }
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    if proposal is None:
        report = {
            "schema": "lafgs_v6_round_proposal",
            "version": 2,
            "uses_source_mapping_rgb": False,
            "uses_test_queries": False,
            "arm": arm,
            "proposal_available": False,
            "unavailable_reason": unavailable_reason,
            "producer_git_commit": commit,
            "configuration": proposal_configuration,
            "input_sha256": {
                "map": map_sha,
                "observation_cache": cache_sha,
                "feedback": feedback_sha,
                **(
                    {
                        "mapping_training_query_indices": (
                            descriptor_training_split_sha
                        ),
                        "descriptor_training_query_indices": (
                            descriptor_training_split_sha
                        )
                    }
                    if descriptor_training_split_sha is not None
                    else {}
                ),
                **(
                    {"association_graph": association_sha}
                    if association_sha is not None
                    else {}
                ),
            },
            "anchor_count": int(torch.as_tensor(state["anchor_ids"]).numel()),
        }
        (args.output_dir / "proposal.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n"
        )
        return report
    provenance = dict(proposal.get("provenance", {}))
    history = list(provenance.get("v6_proposal_history", ()))
    history.append(
        {
            "arm": arm,
            "producer_git_commit": commit,
            "parent_map_sha256": map_sha,
            "observation_cache_sha256": cache_sha,
            "feedback_sha256": feedback_sha,
            "mapping_training_split_sha256": descriptor_training_split_sha,
            "descriptor_training_split_sha256": descriptor_training_split_sha,
            "association_graph_sha256": association_sha,
        }
    )
    proposal["provenance"] = {
        **provenance,
        "uses_source_mapping_rgb": False,
        "uses_test_queries": False,
        "v6_parent_map_sha256": map_sha,
        "v6_latest_feedback_sha256": feedback_sha,
        "v6_latest_observation_cache_sha256": cache_sha,
        "v6_latest_producer_git_commit": commit,
        "v6_latest_proposal_arm": arm,
        "v6_proposal_history": history,
    }
    map_path = args.output_dir / "proposal_map.pt"
    _save(proposal, map_path)
    proposal_sha = sha256_file(map_path)
    metric = identity_metric_state(
        proposal, map_path=str(map_path.resolve()), map_sha256=proposal_sha
    )
    metric_path = args.output_dir / "identity_metric.pt"
    _save(metric, metric_path)
    deployment = compact_projective_deployment_map(proposal)
    deployment_map_path = args.output_dir / "deployment_map.pt"
    _save(deployment, deployment_map_path)
    deployment_map_sha = sha256_file(deployment_map_path)
    deployment_metric = identity_metric_state(
        deployment,
        map_path=str(deployment_map_path.resolve()),
        map_sha256=deployment_map_sha,
    )
    deployment_metric_path = args.output_dir / "deployment_identity_metric.pt"
    _save(deployment_metric, deployment_metric_path)
    report = {
        "schema": "lafgs_v6_round_proposal",
        "version": 2,
        "uses_source_mapping_rgb": False,
        "uses_test_queries": False,
        "arm": arm,
        "proposal_available": True,
        "producer_git_commit": commit,
        "configuration": proposal_configuration,
        "input_sha256": {
            "map": map_sha,
            "observation_cache": cache_sha,
            "feedback": feedback_sha,
            **(
                {
                    "mapping_training_query_indices": descriptor_training_split_sha,
                    "descriptor_training_query_indices": descriptor_training_split_sha,
                }
                if descriptor_training_split_sha is not None
                else {}
            ),
            **(
                {"association_graph": association_sha}
                if association_sha is not None
                else {}
            ),
        },
        "output": {
            "map": str(map_path.resolve()),
            "map_sha256": proposal_sha,
            "metric": str(metric_path.resolve()),
            "metric_sha256": sha256_file(metric_path),
            "deployment_map": str(deployment_map_path.resolve()),
            "deployment_map_sha256": deployment_map_sha,
            "deployment_metric": str(deployment_metric_path.resolve()),
            "deployment_metric_sha256": sha256_file(deployment_metric_path),
            "training_checkpoint_size_bytes": map_path.stat().st_size,
            "deployment_map_size_bytes": deployment_map_path.stat().st_size,
        },
        "anchor_count": int(torch.as_tensor(proposal["anchor_ids"]).numel()),
        "selection_report": _jsonable(selection_report),
        "descriptor_distillation": _distillation_summary(
            proposal.get("v6_descriptor_distillation")
        ),
        "reconstruction_distillation": _jsonable(
            proposal.get("v6_reconstruction_distillation")
        ),
    }
    (args.output_dir / "proposal.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arm",
        choices=(
            "descriptor",
            "descriptor_loss",
            "selection",
            "descriptor_selection",
            "reconstruction",
        ),
        required=True,
    )
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
    parser.add_argument("--descriptor-margin", type=float, default=0.05)
    parser.add_argument("--descriptor-temperature", type=float, default=0.04)
    parser.add_argument("--descriptor-learning-rate", type=float, default=0.02)
    parser.add_argument("--descriptor-epochs", type=int, default=5)
    parser.add_argument("--descriptor-batch-size", type=int, default=8192)
    parser.add_argument("--descriptor-maximum-triplets-per-query", type=int, default=128)
    parser.add_argument("--descriptor-clean-fraction", type=float, default=0.25)
    parser.add_argument("--descriptor-clean-weight", type=float, default=0.25)
    parser.add_argument("--descriptor-trust-weight", type=float, default=0.1)
    parser.add_argument(
        "--mapping-training-query-indices",
        "--descriptor-training-query-indices",
        dest="descriptor_training_query_indices",
        type=Path,
    )
    parser.add_argument(
        "--expected-mapping-training-query-indices-sha256",
        "--expected-descriptor-training-query-indices-sha256",
        dest="expected_descriptor_training_query_indices_sha256",
    )
    parser.add_argument("--maximum-anchors", type=int, default=20000)
    parser.add_argument("--visibility-target", type=int, default=4)
    parser.add_argument("--detectability-target", type=int, default=16)
    parser.add_argument("--matching-target", type=int, default=16)
    parser.add_argument("--pose-logdet-target", type=float, default=0.0)
    parser.add_argument("--pose-min-eigenvalue-target", type=float, default=0.0)
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
