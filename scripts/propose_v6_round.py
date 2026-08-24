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
from map_learning.v6_association_repair import association_repair_proposal
from map_learning.v6_control_actions import control_oriented_descriptor_proposal
from map_learning.v6_proposals import (
    descriptor_loss_proposal,
    descriptor_only_proposal,
    geometry_consensus_descriptor_feedback,
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
            raise ValueError("mapping training split SHA has no input file")
        return None, None
    if expected_sha256 is None:
        raise ValueError("mapping training split requires an expected SHA")
    serialized = path.read_bytes()
    actual_sha256 = hashlib.sha256(serialized).hexdigest()
    if actual_sha256 != expected_sha256:
        raise ValueError("mapping training split SHA differs")
    value = json.loads(serialized)
    if not isinstance(value, dict):
        raise ValueError("mapping training split must be a bound document")
    require_schema(value, DESCRIPTOR_SPLIT_SCHEMA, label="mapping training split")
    if (
        require_source_feedback_match
        and value.get("source_feedback_sha256") != feedback_sha256
    ):
        raise ValueError("mapping training split is not bound to the feedback")
    if query_names is None or value.get(
        "query_names_sha256"
    ) != ordered_query_registry_sha256(query_names):
        raise ValueError("mapping training split query registry differs")
    training = value.get("training_query_indices")
    validation = value.get("validation_query_indices")
    if not isinstance(training, list) or not isinstance(validation, list):
        raise ValueError("mapping training split indices must be lists")
    rows = [int(row) for row in training]
    held_out = [int(row) for row in validation]
    expected_rows = list(range(len(query_names)))
    if (
        rows != sorted(set(rows))
        or held_out != sorted(set(held_out))
        or sorted(rows + held_out) != expected_rows
    ):
        raise ValueError("mapping training split is not an exact partition")
    return rows, actual_sha256


def _attach_reconstruction_distillation(
    proposal: dict,
    state: dict,
    completion: dict,
    *,
    target_query_indices: list[int],
    excluded_support_query_indices: list[int],
    training_query_indices: list[int],
    query_count: int,
    training_split_sha256: str,
) -> None:
    for field in ("v6_descriptor_distillation", "v6_selection_distillation"):
        report = state.get(field)
        if isinstance(report, dict):
            proposal[field] = dict(report)
    prior = state.get("v6_reconstruction_distillation")
    query_count = int(query_count)
    if query_count < 1:
        raise ValueError("reconstruction query registry must be non-empty")

    def registry(value, *, label: str, non_empty: bool = False) -> torch.Tensor:
        rows = torch.as_tensor(value, dtype=torch.long).reshape(-1)
        if (
            (non_empty and rows.numel() == 0)
            or rows.numel() != torch.unique(rows).numel()
            or (rows.numel() and (int(rows.min()) < 0 or int(rows.max()) >= query_count))
        ):
            raise ValueError(f"{label} query registry is invalid")
        return torch.sort(rows).values

    round_training = registry(
        training_query_indices,
        label="round reconstruction training",
        non_empty=True,
    )
    round_targets = registry(
        target_query_indices,
        label="round reconstruction target",
    )
    round_excluded = registry(
        excluded_support_query_indices,
        label="round reconstruction excluded support",
    )
    if (
        round_targets.numel()
        and not bool(torch.isin(round_targets, round_training).all())
    ) or (
        round_excluded.numel()
        and not bool(torch.isin(round_excluded, round_training).all())
    ):
        raise ValueError("reconstruction targets/support exclusions must be training-only")
    all_queries = torch.arange(query_count, dtype=torch.long)
    round_validation = all_queries[~torch.isin(all_queries, round_training)]
    prior_training_explicit = True
    prior_training = torch.empty(0, dtype=torch.long)
    if isinstance(prior, dict):
        prior_training_value = prior.get("training_query_indices")
        if prior_training_value is None:
            # Legacy reconstruction reports cannot establish an arm-level holdout.
            prior_training = all_queries
            prior_training_explicit = False
        else:
            prior_training = registry(
                prior_training_value,
                label="prior reconstruction training",
                non_empty=True,
            )
            prior_training_explicit = bool(
                prior.get("training_query_registry_explicit", False)
            )
    cumulative_training = torch.unique(
        torch.cat((prior_training, round_training)), sorted=True
    )
    cumulative_validation = all_queries[
        ~torch.isin(all_queries, cumulative_training)
    ]
    prior_targets = torch.as_tensor(
        prior.get("target_query_indices", ()) if isinstance(prior, dict) else (),
        dtype=torch.long,
    ).reshape(-1)
    prior_targets = registry(prior_targets, label="prior reconstruction target")
    if (
        prior_targets.numel()
        and not bool(torch.isin(prior_targets, prior_training).all())
    ):
        raise ValueError("prior reconstruction targets are outside its training split")
    targets = torch.unique(torch.cat((prior_targets, round_targets)), sorted=True)
    prior_excluded = torch.as_tensor(
        prior.get("excluded_support_query_indices", ())
        if isinstance(prior, dict)
        else (),
        dtype=torch.long,
    ).reshape(-1)
    prior_excluded = registry(
        prior_excluded, label="prior reconstruction excluded support"
    )
    if (
        prior_excluded.numel()
        and not bool(torch.isin(prior_excluded, prior_training).all())
    ):
        raise ValueError(
            "prior reconstruction support exclusions are outside its training split"
        )
    excluded = torch.unique(torch.cat((prior_excluded, round_excluded)), sorted=True)
    contract = dict(completion.get("contract", {}))
    if (
        contract.get("target_queries_seed_regions") is not True
        or contract.get("support_queries_restricted") is not True
        or contract.get("target_queries_used_as_anchor_support") is not False
    ):
        raise ValueError("completion does not preserve the reconstruction split contract")
    split_shas = list(
        prior.get("training_split_artifact_sha256s", ())
        if isinstance(prior, dict)
        else ()
    )
    all_split_shas = [*split_shas, training_split_sha256]
    if any(
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value.lower())
        for value in all_split_shas
    ):
        raise ValueError("reconstruction training split SHA registry is invalid")
    if isinstance(prior, dict) and prior_training_explicit and not split_shas:
        raise ValueError("prior reconstruction training split SHA registry is missing")
    if training_split_sha256 not in split_shas:
        split_shas.append(training_split_sha256)
    proposal["v6_reconstruction_distillation"] = {
        "schema": "lafgs_v6_target_seeded_projective_completion",
        "version": 2,
        "target_query_indices": targets,
        "round_target_query_indices": round_targets,
        "excluded_support_query_indices": excluded,
        "round_excluded_support_query_indices": round_excluded,
        "training_query_indices": cumulative_training,
        "round_training_query_indices": round_training,
        "validation_query_indices": cumulative_validation,
        "round_validation_query_indices": round_validation,
        "eligible_support_query_indices": cumulative_training,
        "round_eligible_support_query_indices": round_training,
        "training_query_registry_explicit": prior_training_explicit,
        "training_split_artifact_sha256s": split_shas,
        "round_training_split_artifact_sha256": training_split_sha256,
        "arm_level_holdout_query_indices": cumulative_validation,
        "round_arm_level_holdout_query_indices": round_validation,
        "validation_queries_used_as_target_seed_or_support": False,
        "target_query_depth_used_for_seed_region": True,
        "target_queries_used_as_anchor_support": False,
        "final_xyz_source": "fixed_camera_robust_ray_triangulation",
        "completion_contract": contract,
        "reconstruction_round": int(
            prior.get("reconstruction_round", 0)
            if isinstance(prior, dict)
            else 0
        )
        + 1,
    }


def _reconstruction_training_scope(
    feedback: dict,
    training_query_indices: list[int],
) -> dict[str, list[int]]:
    """Resolve training-only L1 targets and neighbor exclusions."""

    records = list(feedback.get("records", ()))
    query_count = len(records)
    training = [int(value) for value in training_query_indices]
    if (
        not training
        or training != sorted(set(training))
        or training[0] < 0
        or training[-1] >= query_count
    ):
        raise ValueError("reconstruction training query registry is invalid")
    training_set = set(training)
    targets = []
    excluded = set()
    for query_index in training:
        record = records[query_index]
        if int(record.get("query_index", query_index)) != query_index:
            raise ValueError("reconstruction feedback query registry differs")
        if "L1" not in record.get(
            "failure_layers", (record.get("failure_layer"),)
        ):
            continue
        targets.append(query_index)
        neighbors = torch.as_tensor(
            record.get("excluded_query_indices", (query_index,)), dtype=torch.long
        ).reshape(-1)
        if neighbors.numel() and (
            int(neighbors.min()) < 0 or int(neighbors.max()) >= query_count
        ):
            raise ValueError("reconstruction excluded-neighbor registry is invalid")
        excluded.update(int(value) for value in neighbors.tolist() if value in training_set)
    validation = sorted(set(range(query_count)) - training_set)
    return {
        "training_query_indices": training,
        "validation_query_indices": validation,
        "target_query_indices": targets,
        "excluded_support_query_indices": sorted(excluded),
    }


def _training_split_input_sha(arm: str, split_sha256: str | None) -> dict[str, str]:
    if split_sha256 is None:
        return {}
    result = {"mapping_training_query_indices": split_sha256}
    result[
        "reconstruction_training_query_indices"
        if arm == "reconstruction"
        else "descriptor_training_query_indices"
    ] = split_sha256
    return result


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
    elif arm in {
        "descriptor_control",
        "descriptor_loss",
        "descriptor_selection",
        "selection",
        "reconstruction",
    }:
        (
            descriptor_training_queries,
            descriptor_training_split_sha,
        ) = _load_query_indices(
            args.descriptor_training_query_indices,
            args.expected_descriptor_training_query_indices_sha256,
            feedback_sha256=feedback_sha,
            query_names=list(feedback["query_names"]),
            require_source_feedback_match=(
                arm == "reconstruction"
                or not any(
                    isinstance(state.get(field), dict)
                    for field in (
                        "v6_descriptor_distillation",
                        "v6_reconstruction_distillation",
                        "v6_selection_distillation",
                    )
                )
            ),
        )
    selection_report = None
    reconstruction_scope = None
    unavailable_reason = None
    association_sha = None
    reconstruction_strategy = getattr(args, "reconstruction_strategy", "completion")
    repair_minimum_descriptor_similarity = float(
        getattr(args, "repair_minimum_descriptor_similarity", 0.9)
    )
    repair_maximum_xyz_distance_m = float(
        getattr(args, "repair_maximum_xyz_distance_m", 0.02)
    )
    repair_minimum_query_evidence = int(
        getattr(args, "repair_minimum_query_evidence", 5)
    )
    if arm == "descriptor":
        pass
    elif arm == "descriptor_control":
        try:
            proposal = control_oriented_descriptor_proposal(
                state,
                observations,
                feedback,
                training_query_indices=descriptor_training_queries,
                trust_region=args.descriptor_trust_region,
                margin=args.descriptor_margin,
                reprojection_error_px=args.ransac_reprojection_px,
                maximum_candidates_per_query=(
                    args.control_maximum_candidates_per_query
                ),
                maximum_correction_set_size=(
                    args.control_maximum_correction_set_size
                ),
                beam_width=args.control_beam_width,
            )
            if descriptor_training_split_sha is not None:
                proposal["v6_descriptor_distillation"][
                    "training_split_artifact_sha256s"
                ] = [descriptor_training_split_sha]
        except ValueError as error:
            if str(error) not in {
                "feedback contains no controllable pose correction sets",
                "controllable correction sets have no joint trust-region action",
                (
                    "controllable correction sets violate clean protection or "
                    "trust region"
                ),
            }:
                raise
            proposal = None
            unavailable_reason = str(error).replace(" ", "_")
    elif arm in {"descriptor_loss", "descriptor_selection"}:
        try:
            descriptor_feedback = feedback
            geometry_consensus_triplet_count = 0
            if args.descriptor_positive_mode == "exact_or_geometry":
                descriptor_feedback, geometry_consensus_triplet_count = (
                    geometry_consensus_descriptor_feedback(feedback)
                )
            proposal = descriptor_loss_proposal(
                state,
                observations,
                descriptor_feedback,
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
                pose_critical_weight=args.descriptor_pose_critical_weight,
                tail_query_weight=args.descriptor_tail_query_weight,
                training_query_indices=descriptor_training_queries,
                allow_geometry_compatible_positives=(
                    args.descriptor_positive_mode == "exact_or_geometry"
                ),
                loss_mode=getattr(args, "descriptor_loss_mode", "pairwise"),
                consensus_count_target=getattr(
                    args, "descriptor_consensus_count_target", 16.0
                ),
                consensus_cell_target=getattr(
                    args, "descriptor_consensus_cell_target", 4.0
                ),
                consensus_count_weight=getattr(
                    args, "descriptor_consensus_count_weight", 1.0
                ),
                consensus_cell_weight=getattr(
                    args, "descriptor_consensus_cell_weight", 1.0
                ),
                device=args.device,
            )
            proposal["v6_descriptor_distillation"][
                "geometry_consensus_source_triplet_count_all_queries"
            ] = geometry_consensus_triplet_count
            proposal["v6_descriptor_distillation"][
                "geometry_consensus_source_triplet_count_training_queries"
            ] = sum(
                int(
                    descriptor_feedback["records"][query_index].get(
                        "geometry_consensus_weak_positive_triplet_count", 0
                    )
                )
                for query_index in (
                    range(len(descriptor_feedback["records"]))
                    if descriptor_training_queries is None
                    else descriptor_training_queries
                )
            )
        except ValueError as error:
            if str(error) != "feedback contains no trainable descriptor triplets":
                raise
            proposal = None
            unavailable_reason = "no_trainable_l3_descriptor_triplets"
        if arm == "descriptor_selection" and proposal is not None:
            proposal, selection_report = selection_only_proposal(
                proposal,
                observations,
                feedback,
                maximum_anchors=args.maximum_anchors,
                visibility_target=args.visibility_target,
                detectability_target=args.detectability_target,
                matching_target=args.matching_target,
                pose_logdet_target=args.pose_logdet_target,
                pose_min_eigenvalue_target=args.pose_min_eigenvalue_target,
                pose_information_chunk_size=(
                    args.selection_pose_information_chunk_size
                ),
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
            observations,
            feedback,
            maximum_anchors=args.maximum_anchors,
            visibility_target=args.visibility_target,
            detectability_target=args.detectability_target,
            matching_target=args.matching_target,
            pose_logdet_target=args.pose_logdet_target,
            pose_min_eigenvalue_target=args.pose_min_eigenvalue_target,
            pose_information_chunk_size=args.selection_pose_information_chunk_size,
            training_query_indices=descriptor_training_queries,
        )
    elif arm == "reconstruction":
        if (
            descriptor_training_queries is None
            or descriptor_training_split_sha is None
        ):
            raise ValueError(
                "reconstruction requires a SHA-bound mapping training split"
            )
        reconstruction_scope = _reconstruction_training_scope(
            feedback, descriptor_training_queries
        )
        if reconstruction_strategy in (
            "association_repair",
            "association_repair_global",
        ):
            deploy_repair_globally = reconstruction_strategy.endswith("_global")
            try:
                proposal, repair_report = association_repair_proposal(
                    state,
                    observations,
                    feedback,
                    training_query_indices=descriptor_training_queries,
                    training_split_sha256=descriptor_training_split_sha,
                    lineage={
                        **dict(state.get("provenance", {})),
                        "v6_parent_map_sha256": map_sha,
                        "v6_reconstruction_feedback_sha256": feedback_sha,
                        "v6_reconstruction_training_split_sha256": (
                            descriptor_training_split_sha
                        ),
                        "v6_reconstruction_strategy": reconstruction_strategy,
                    },
                    minimum_descriptor_similarity=(
                        repair_minimum_descriptor_similarity
                    ),
                    maximum_xyz_distance_m=repair_maximum_xyz_distance_m,
                    minimum_query_evidence=repair_minimum_query_evidence,
                    minimum_views=args.minimum_views,
                    minimum_view_bins=args.minimum_camera_families,
                    maximum_reprojection_px=args.maximum_epipolar_error_px,
                    deploy_rule_globally=deploy_repair_globally,
                )
                reconstruction_scope = {
                    **reconstruction_scope,
                    "strategy": reconstruction_strategy,
                    "selected_pair_count": int(
                        repair_report["selected_pair_count"]
                    ),
                    "successful_pair_count": int(
                        repair_report["successful_pair_count"]
                    ),
                }
            except ValueError as error:
                unavailable = {
                    "association repair selected no fragmented Track pair": (
                        "no_training_evidenced_association_repair_pair"
                    ),
                    "association graph contains no ray-triangulated Anchor": (
                        "no_ray_triangulated_association_repair_anchor"
                    ),
                }
                if str(error) not in unavailable:
                    raise
                proposal = None
                unavailable_reason = unavailable[str(error)]
        else:
            if (
                args.association_graph is None
                or args.expected_association_graph_sha256 is None
            ):
                raise ValueError(
                    "completion reconstruction requires a SHA-bound association graph"
                )
            association, association_sha = _load(
                args.association_graph,
                args.expected_association_graph_sha256,
                "association graph",
            )
            require_schema(
                association, ASSOCIATION_GRAPH_SCHEMA, label="association graph"
            )
            l1_queries = reconstruction_scope["target_query_indices"]
            excluded_support_queries = reconstruction_scope[
                "excluded_support_query_indices"
            ]
            if not l1_queries:
                proposal = None
                unavailable_reason = "no_training_split_l1_query"
            else:
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
                        safety_maximum_components=(
                            args.completion_safety_maximum_components
                        ),
                        eligible_query_indices=descriptor_training_queries,
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
                        "v6_reconstruction_training_split_sha256": (
                            descriptor_training_split_sha
                        ),
                        "v6_reconstruction_training_query_count": len(
                            descriptor_training_queries
                        ),
                        "v6_reconstruction_holdout_query_count": len(
                            reconstruction_scope["validation_query_indices"]
                        ),
                        "v6_l1_query_count": len(l1_queries),
                    },
                )
                _attach_reconstruction_distillation(
                    proposal,
                    state,
                    completion,
                    target_query_indices=l1_queries,
                    excluded_support_query_indices=excluded_support_queries,
                    training_query_indices=descriptor_training_queries,
                    query_count=len(feedback["records"]),
                    training_split_sha256=descriptor_training_split_sha,
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
        "descriptor_pose_critical_weight": float(
            args.descriptor_pose_critical_weight
        ),
        "descriptor_tail_query_weight": float(args.descriptor_tail_query_weight),
        "ransac_reprojection_px": float(
            getattr(args, "ransac_reprojection_px", 4.0)
        ),
        "control_maximum_candidates_per_query": int(
            getattr(args, "control_maximum_candidates_per_query", 24)
        ),
        "control_maximum_correction_set_size": int(
            getattr(args, "control_maximum_correction_set_size", 8)
        ),
        "control_beam_width": int(getattr(args, "control_beam_width", 4)),
            "descriptor_positive_mode": str(
                getattr(args, "descriptor_positive_mode", "exact")
            ),
            "descriptor_loss_mode": str(
                getattr(args, "descriptor_loss_mode", "pairwise")
            ),
            "descriptor_consensus_count_target": float(
                getattr(args, "descriptor_consensus_count_target", 16.0)
            ),
            "descriptor_consensus_cell_target": float(
                getattr(args, "descriptor_consensus_cell_target", 4.0)
            ),
            "descriptor_consensus_count_weight": float(
                getattr(args, "descriptor_consensus_count_weight", 1.0)
            ),
            "descriptor_consensus_cell_weight": float(
                getattr(args, "descriptor_consensus_cell_weight", 1.0)
            ),
        "maximum_anchors": int(args.maximum_anchors),
        "visibility_target": int(args.visibility_target),
        "detectability_target": int(args.detectability_target),
        "matching_target": int(args.matching_target),
        "pose_logdet_target": float(args.pose_logdet_target),
        "pose_min_eigenvalue_target": float(args.pose_min_eigenvalue_target),
        "selection_pose_information_chunk_size": int(
            args.selection_pose_information_chunk_size
        ),
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
        "reconstruction_mapping_training_split_required": arm == "reconstruction",
        "reconstruction_training_only_targets_seeds_and_support": (
            arm == "reconstruction"
        ),
        "reconstruction_arm_level_holdout": arm == "reconstruction",
        "reconstruction_strategy": reconstruction_strategy,
        "repair_minimum_descriptor_similarity": repair_minimum_descriptor_similarity,
        "repair_maximum_xyz_distance_m": repair_maximum_xyz_distance_m,
        "repair_minimum_query_evidence": repair_minimum_query_evidence,
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
                **_training_split_input_sha(arm, descriptor_training_split_sha),
                **(
                    {"association_graph": association_sha}
                    if association_sha is not None
                    else {}
                ),
            },
            "anchor_count": int(torch.as_tensor(state["anchor_ids"]).numel()),
            "reconstruction_training_scope": reconstruction_scope,
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
            "descriptor_training_split_sha256": (
                descriptor_training_split_sha
                if arm in {"descriptor_loss", "descriptor_selection", "selection"}
                else None
            ),
            "reconstruction_training_split_sha256": (
                descriptor_training_split_sha if arm == "reconstruction" else None
            ),
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
            **_training_split_input_sha(arm, descriptor_training_split_sha),
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
        "reconstruction_training_scope": reconstruction_scope,
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
            "descriptor_control",
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
    parser.add_argument("--descriptor-pose-critical-weight", type=float, default=0.0)
    parser.add_argument("--descriptor-tail-query-weight", type=float, default=0.0)
    parser.add_argument("--ransac-reprojection-px", type=float, required=True)
    parser.add_argument(
        "--control-maximum-candidates-per-query", type=int, default=24
    )
    parser.add_argument(
        "--control-maximum-correction-set-size", type=int, default=8
    )
    parser.add_argument("--control-beam-width", type=int, default=4)
    parser.add_argument(
        "--descriptor-positive-mode",
        choices=("exact", "exact_or_geometry"),
        default="exact",
    )
    parser.add_argument(
        "--descriptor-loss-mode",
        choices=("pairwise", "set_consensus"),
        default="pairwise",
    )
    parser.add_argument("--descriptor-consensus-count-target", type=float, default=16.0)
    parser.add_argument("--descriptor-consensus-cell-target", type=float, default=4.0)
    parser.add_argument("--descriptor-consensus-count-weight", type=float, default=1.0)
    parser.add_argument("--descriptor-consensus-cell-weight", type=float, default=1.0)
    parser.add_argument(
        "--mapping-training-query-indices",
        "--descriptor-training-query-indices",
        dest="descriptor_training_query_indices",
        type=Path,
        help=(
            "SHA-bound mapping train/validation split; required by reconstruction "
            "and used by descriptor/selection arms"
        ),
    )
    parser.add_argument(
        "--expected-mapping-training-query-indices-sha256",
        "--expected-descriptor-training-query-indices-sha256",
        dest="expected_descriptor_training_query_indices_sha256",
        help="Expected SHA256 of the mapping train/validation split",
    )
    parser.add_argument("--maximum-anchors", type=int, default=20000)
    parser.add_argument("--visibility-target", type=int, default=4)
    parser.add_argument("--detectability-target", type=int, default=16)
    parser.add_argument("--matching-target", type=int, default=16)
    parser.add_argument("--pose-logdet-target", type=float, default=0.0)
    parser.add_argument("--pose-min-eigenvalue-target", type=float, default=0.0)
    parser.add_argument(
        "--selection-pose-information-chunk-size", type=int, default=4096
    )
    parser.add_argument("--completion-voxel-size-m", type=float, default=0.05)
    parser.add_argument("--alpha-minimum", type=float, default=0.05)
    parser.add_argument("--completion-minimum-similarity", type=float, default=0.7)
    parser.add_argument("--minimum-margin", type=float, default=0.01)
    parser.add_argument("--maximum-epipolar-error-px", type=float, default=2.0)
    parser.add_argument("--minimum-views", type=int, default=3)
    parser.add_argument("--minimum-camera-families", type=int, default=2)
    parser.add_argument("--completion-maximum-rows-per-view", type=int, default=256)
    parser.add_argument("--completion-safety-maximum-components", type=int, default=100000)
    parser.add_argument(
        "--reconstruction-strategy",
        choices=("completion", "association_repair", "association_repair_global"),
        default="completion",
    )
    parser.add_argument(
        "--repair-minimum-descriptor-similarity", type=float, default=0.9
    )
    parser.add_argument("--repair-maximum-xyz-distance-m", type=float, default=0.02)
    parser.add_argument("--repair-minimum-query-evidence", type=int, default=5)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
