#!/usr/bin/env python3
"""Calibrate and validate Gaussian-provenance Anchor truth without LOO."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import itertools
import json
import multiprocessing
import os
from pathlib import Path

import torch

from common.hashing import sha256_file
from map_learning.v18_provenance_truth import (
    TRUTH_EQUIVALENT,
    TRUTH_UNIQUE,
    TruthAssignmentThresholds,
    aggregate_anchor_provenance,
    assign_full_map_projection_truth,
    assign_provenance_truth,
    backproject_query_surface,
    build_primitive_anchor_index,
    provenance_candidate_graph,
    query_anchor_geometry_evidence,
    transport_candidate_graph,
)


def _atomic_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _balanced_family_roles(families: torch.Tensor, seed: int) -> dict[int, str]:
    unique = torch.unique(torch.as_tensor(families).long(), sorted=True).tolist()
    if len(unique) < 5:
        raise ValueError("V18 truth validation requires at least five pose families")
    ranked = sorted(
        unique,
        key=lambda family: hashlib.sha256(
            f"{int(seed)}:{int(family)}".encode()
        ).digest(),
    )
    design_count = max(1, round(0.60 * len(ranked)))
    calibration_count = max(1, round(0.20 * len(ranked)))
    if design_count + calibration_count >= len(ranked):
        design_count = len(ranked) - calibration_count - 1
    roles = {}
    for offset, family in enumerate(ranked):
        if offset < design_count:
            role = "signature_design"
        elif offset < design_count + calibration_count:
            role = "threshold_calibration"
        else:
            role = "independent_validation"
        roles[int(family)] = role
    return roles


def _sample_rows(mask: torch.Tensor, maximum: int, seed: int) -> torch.Tensor:
    rows = torch.nonzero(mask, as_tuple=False).reshape(-1)
    if rows.numel() <= int(maximum):
        return rows
    generator = torch.Generator().manual_seed(int(seed))
    return rows[torch.randperm(rows.numel(), generator=generator)[: int(maximum)]].sort().values


def _gather_keypoints(
    rows: torch.Tensor,
    observation_queries: torch.Tensor,
    observation_keypoints: torch.Tensor,
    mapping_keypoints: list[torch.Tensor],
    pixel_center_offsets: torch.Tensor,
) -> torch.Tensor:
    result = torch.empty((rows.numel(), 2), dtype=torch.float32)
    selected_queries = observation_queries[rows]
    selected_keypoints = observation_keypoints[rows]
    for query in torch.unique(selected_queries, sorted=True).tolist():
        local = torch.nonzero(selected_queries == int(query), as_tuple=False).reshape(-1)
        indices = selected_keypoints[local]
        result[local] = (
            torch.as_tensor(mapping_keypoints[query]).float()[indices]
            + float(pixel_center_offsets[query])
        )
    return result


def _backproject_mixed_queries(
    *,
    keypoints: torch.Tensor,
    depth: torch.Tensor,
    queries: torch.Tensor,
    intrinsics: torch.Tensor,
    poses_w2c: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    surface = torch.empty((keypoints.shape[0], 3), dtype=torch.float32)
    valid = torch.zeros(keypoints.shape[0], dtype=torch.bool)
    for query in torch.unique(queries, sorted=True).tolist():
        local = torch.nonzero(queries == int(query), as_tuple=False).reshape(-1)
        xyz, local_valid = backproject_query_surface(
            keypoints[local],
            depth[local],
            intrinsics[query],
            poses_w2c[query],
        )
        surface[local] = xyz
        valid[local] = local_valid
    return surface, valid


def _truth_metrics(
    truth: dict,
    ground_truth_anchor: torch.Tensor,
    equivalence: torch.Tensor,
) -> dict[str, float | int]:
    status = torch.as_tensor(truth["truth_status"]).long()
    offsets = torch.as_tensor(truth["truth_offsets"]).long()
    predicted = torch.as_tensor(truth["truth_anchor_rows"]).long()
    gt = torch.as_tensor(ground_truth_anchor).long()
    decisive = (status == TRUTH_UNIQUE) | (status == TRUTH_EQUIVALENT)
    correct = torch.zeros(gt.numel(), dtype=torch.bool)
    for row in torch.nonzero(decisive, as_tuple=False).reshape(-1).tolist():
        start, stop = int(offsets[row]), int(offsets[row + 1])
        correct[row] = bool(
            (equivalence[predicted[start:stop]] == equivalence[gt[row]]).any()
        )
    decisive_count = int(decisive.sum())
    correct_count = int((correct & decisive).sum())
    wrong_count = decisive_count - correct_count
    return {
        "row_count": int(gt.numel()),
        "decisive_assignment_count": decisive_count,
        "correct_assignment_count": correct_count,
        "wrong_assignment_count": wrong_count,
        "decisive_coverage": float(decisive_count / max(gt.numel(), 1)),
        "decisive_precision": float(correct_count / max(decisive_count, 1)),
        "wrong_unique_assignment_rate": float(wrong_count / max(decisive_count, 1)),
    }


def _fake_provenance_only_transport(graph: dict) -> dict:
    count = torch.as_tensor(graph["candidate_anchor_rows"]).numel()
    return {
        "candidate_offsets": graph["candidate_offsets"],
        "candidate_anchor_rows": graph["candidate_anchor_rows"],
        "transport_view_family_count": torch.full((count,), 127, dtype=torch.long),
        "transport_median_residual_px": torch.zeros(count),
    }


_TRIAL_CONTEXT: dict = {}


def _evaluate_threshold_trial(values: tuple[float | int, ...]) -> dict:
    context = _TRIAL_CONTEXT
    base_values = values[:6]
    thresholds = TruthAssignmentThresholds(
        *base_values,
        maximum_composition_entropy=float(context["maximum_composition_entropy"]),
        maximum_relative_depth_spread=float(context["maximum_relative_depth_spread"]),
        maximum_query_reprojection_px=float(values[6]),
        maximum_query_normalized_depth_residual=float(values[7]),
        maximum_query_projection_std_px=2.0,
    )
    truth = assign_provenance_truth(
        candidate_graph=context["graph"],
        transport_evidence=context["transport"],
        geometry_evidence=context["geometry"],
        equivalence_class_ids=context["equivalence"],
        thresholds=thresholds,
    )
    metrics = _truth_metrics(
        truth, context["ground_truth_anchor"], context["equivalence"]
    )
    return {
        "thresholds": {
            name: getattr(thresholds, name)
            for name in TruthAssignmentThresholds.__dataclass_fields__
        },
        "metrics": metrics,
    }


def _choose_thresholds(
    *,
    graph: dict,
    transport: dict,
    geometry: dict,
    ground_truth_anchor: torch.Tensor,
    equivalence: torch.Tensor,
    minimum_precision: float,
    maximum_composition_entropy: float,
    maximum_relative_depth_spread: float,
) -> tuple[TruthAssignmentThresholds, dict, list[dict]]:
    values = list(
        itertools.product(
            (0.05, 0.10, 0.15, 0.20, 0.30),
            (1, 2, 3),
            (1.5, 2.5, 4.0, 6.0),
            (0.05, 0.10, 0.15),
            (0.01, 0.03),
            (1.05, 1.15),
            (1.0, 2.0, 4.0),
            (0.5, 1.0),
        )
    )
    global _TRIAL_CONTEXT
    _TRIAL_CONTEXT = {
        "graph": graph,
        "transport": transport,
        "geometry": geometry,
        "ground_truth_anchor": ground_truth_anchor,
        "equivalence": equivalence,
        "maximum_composition_entropy": maximum_composition_entropy,
        "maximum_relative_depth_spread": maximum_relative_depth_spread,
    }
    worker_count = min(16, max(os.cpu_count() or 1, 1), len(values))
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=worker_count,
            mp_context=multiprocessing.get_context("fork"),
        ) as executor:
            trials = list(executor.map(_evaluate_threshold_trial, values, chunksize=2))
    finally:
        torch.set_num_threads(previous_threads)
        _TRIAL_CONTEXT = {}
    eligible = [
        item
        for item in trials
        if item["metrics"]["decisive_assignment_count"] > 0
        and item["metrics"]["decisive_precision"] >= float(minimum_precision)
    ]
    pool = eligible or trials
    selected = max(
        pool,
        key=lambda item: (
            item["metrics"]["decisive_coverage"],
            item["metrics"]["decisive_precision"],
            -item["metrics"]["wrong_assignment_count"],
        ),
    )
    thresholds = TruthAssignmentThresholds(**selected["thresholds"])
    return thresholds, selected, trials


def _prepare_split(
    *,
    rows: torch.Tensor,
    primitive_ids: torch.Tensor,
    weights: torch.Tensor,
    valid: torch.Tensor,
    rendered_depth: torch.Tensor,
    composition_entropy: torch.Tensor,
    relative_depth_spread: torch.Tensor,
    retained_composition_fraction: torch.Tensor,
    observation_queries: torch.Tensor,
    observation_keypoints: torch.Tensor,
    observation_anchor: torch.Tensor,
    mapping_keypoints: list[torch.Tensor],
    mapping_offsets: torch.Tensor,
    mapping_intrinsics: torch.Tensor,
    mapping_poses: torch.Tensor,
    inverse: dict,
    anchor_observation_offsets: torch.Tensor,
    mapping_families: torch.Tensor,
    design_observation_mask: torch.Tensor,
    anchor_xyz: torch.Tensor,
    anchor_covariance: torch.Tensor,
    maximum_candidates: int,
    transport_residual_px: float,
    minimum_transport_overlap: float,
    geometry_device: str,
) -> dict:
    query_ids = primitive_ids[rows]
    query_weights = weights[rows]
    query_valid = valid[rows]
    graph = provenance_candidate_graph(
        query_primitive_ids=query_ids,
        query_weights=query_weights,
        primitive_anchor_index=inverse,
        query_valid=query_valid,
        query_composition_entropy=composition_entropy[rows],
        query_relative_depth_spread=relative_depth_spread[rows],
        query_retained_composition_fraction=retained_composition_fraction[rows],
        maximum_candidates_per_row=int(maximum_candidates),
    )
    queries = observation_queries[rows]
    keypoints = _gather_keypoints(
        rows,
        observation_queries,
        observation_keypoints,
        mapping_keypoints,
        mapping_offsets,
    )
    surface, surface_valid = _backproject_mixed_queries(
        keypoints=keypoints,
        depth=rendered_depth[rows],
        queries=queries,
        intrinsics=mapping_intrinsics,
        poses_w2c=mapping_poses,
    )
    graph["query_valid"] &= surface_valid
    geometry = query_anchor_geometry_evidence(
        candidate_graph=graph,
        query_keypoints=keypoints,
        query_depth=rendered_depth[rows],
        query_indices=queries,
        anchor_xyz=anchor_xyz,
        anchor_covariance=anchor_covariance,
        query_intrinsics=mapping_intrinsics,
        query_poses_w2c=mapping_poses,
        device=geometry_device,
    )
    transport = transport_candidate_graph(
        candidate_graph=graph,
        query_surface_xyz=surface,
        anchor_observation_offsets=anchor_observation_offsets,
        observation_query_indices=observation_queries,
        observation_keypoint_indices=observation_keypoints,
        observation_enabled=design_observation_mask,
        mapping_keypoints=[
            torch.as_tensor(value).float() + float(mapping_offsets[index])
            for index, value in enumerate(mapping_keypoints)
        ],
        mapping_intrinsics=mapping_intrinsics,
        mapping_poses_w2c=mapping_poses,
        mapping_view_family_ids=mapping_families,
        inlier_residual_px=float(transport_residual_px),
        minimum_candidate_overlap_to_evaluate=float(minimum_transport_overlap),
    )
    return {
        "rows": rows,
        "queries": queries,
        "keypoints": keypoints,
        "depth": rendered_depth[rows],
        "ground_truth_anchor_rows": observation_anchor[rows],
        "candidate_graph": graph,
        "transport": transport,
        "geometry": geometry,
    }


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor-map", type=Path, required=True)
    parser.add_argument("--mapping-provenance", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split-seed", type=int, default=1820260829)
    parser.add_argument("--maximum-calibration-rows", type=int, default=4000)
    parser.add_argument("--maximum-validation-rows", type=int, default=4000)
    parser.add_argument(
        "--maximum-candidates",
        type=int,
        default=0,
        help="0 enumerates every provenance-linked Anchor",
    )
    parser.add_argument("--minimum-calibration-precision", type=float, default=0.99)
    parser.add_argument("--transport-residual-px", type=float, default=8.0)
    parser.add_argument("--projection-device", default="cuda")
    parser.add_argument("--diagnostic-acceptance-quantile", type=float, default=0.95)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not 0.5 <= float(args.diagnostic_acceptance_quantile) < 1.0:
        parser.error("diagnostic acceptance quantile must lie in [0.5, 1)")
    state = torch.load(args.anchor_map, map_location="cpu", weights_only=False)
    raw = torch.load(args.mapping_provenance, map_location="cpu", weights_only=False)
    if not (
        raw.get("schema") == "lafgs_v18_mapping_observation_gaussian_provenance"
        and raw.get("uses_test_queries") is False
        and raw.get("loo_used") is False
        and raw.get("descriptor_independent") is True
        and raw.get("full_gaussian_prior_evaluated") is True
        and raw.get("full_depth_ordered_compositing") is True
        and float(raw.get("minimum_retained_composition_mass", 0.0)) >= 0.95
    ):
        raise ValueError("V18 mapping provenance contract differs")
    observations = state["projective_anchor_observations"]
    anchor_offsets = torch.as_tensor(observations["observation_offsets"]).long()
    observation_queries = torch.as_tensor(observations["query_indices"]).long()
    observation_keypoints = torch.as_tensor(observations["keypoint_indices"]).long()
    edge_count = int(observation_queries.numel())
    raw_rows = torch.as_tensor(raw["observation_rows"]).long()
    if raw_rows.numel() != edge_count or not torch.equal(raw_rows.sort().values, torch.arange(edge_count)):
        raise ValueError("V18 evaluation requires complete mapping provenance, not a partial shard")
    k = int(torch.as_tensor(raw["observation_primitive_ids"]).shape[1])
    primitive_ids = torch.full((edge_count, k), -1, dtype=torch.long)
    weights = torch.zeros((edge_count, k), dtype=torch.float32)
    valid = torch.zeros(edge_count, dtype=torch.bool)
    rendered_depth = torch.full((edge_count,), float("nan"))
    composition_entropy = torch.full((edge_count,), float("nan"))
    relative_depth_spread = torch.full((edge_count,), float("nan"))
    retained_composition_fraction = torch.zeros(edge_count)
    primitive_ids[raw_rows] = torch.as_tensor(raw["observation_primitive_ids"]).long()
    weights[raw_rows] = torch.as_tensor(raw["observation_weights"]).float()
    valid[raw_rows] = torch.as_tensor(raw["observation_valid"]).bool()
    rendered_depth[raw_rows] = torch.as_tensor(raw["observation_rendered_depth"]).float()
    composition_entropy[raw_rows] = torch.as_tensor(
        raw["observation_composition_entropy"]
    ).float()
    relative_depth_spread[raw_rows] = torch.as_tensor(
        raw["observation_relative_depth_spread_3x3"]
    ).float()
    retained_composition_fraction[raw_rows] = torch.as_tensor(
        raw["observation_retained_composition_fraction"]
    ).float()
    anchor_count = int(torch.as_tensor(state["anchor_ids"]).numel())
    observation_anchor = torch.repeat_interleave(
        torch.arange(anchor_count), anchor_offsets[1:] - anchor_offsets[:-1]
    )
    mapping_families = torch.as_tensor(raw["mapping_view_family_ids"]).long()
    family_roles = _balanced_family_roles(mapping_families, args.split_seed)
    roles = [family_roles[int(family)] for family in mapping_families.tolist()]
    query_design = torch.tensor([role == "signature_design" for role in roles])
    query_calibration = torch.tensor([role == "threshold_calibration" for role in roles])
    query_validation = torch.tensor([role == "independent_validation" for role in roles])
    raw_design_observation = query_design[observation_queries] & valid
    calibration_observation = query_calibration[observation_queries] & valid
    validation_observation = query_validation[observation_queries] & valid
    if not bool(
        raw_design_observation.any()
        and calibration_observation.any()
        and validation_observation.any()
    ):
        raise RuntimeError("V18 disjoint family split produced an empty role")
    calibration_rows = _sample_rows(
        calibration_observation, args.maximum_calibration_rows, args.split_seed + 1
    )
    validation_rows = _sample_rows(
        validation_observation, args.maximum_validation_rows, args.split_seed + 2
    )
    diagnostic_quantile = float(args.diagnostic_acceptance_quantile)
    calibration_entropy = composition_entropy[calibration_rows]
    calibration_spread = relative_depth_spread[calibration_rows]
    finite_entropy = calibration_entropy[torch.isfinite(calibration_entropy)]
    finite_spread = calibration_spread[torch.isfinite(calibration_spread)]
    if finite_entropy.numel() == 0 or finite_spread.numel() == 0:
        raise RuntimeError("V18 calibration has no finite provenance diagnostics")
    maximum_entropy = float(torch.quantile(finite_entropy, diagnostic_quantile))
    maximum_depth_spread = float(torch.quantile(finite_spread, diagnostic_quantile))
    minimum_retained_mass = max(
        0.95, float(raw["minimum_retained_composition_mass"])
    )
    design_observation = (
        raw_design_observation
        & torch.isfinite(composition_entropy)
        & torch.isfinite(relative_depth_spread)
        & (retained_composition_fraction >= minimum_retained_mass)
        & (composition_entropy <= maximum_entropy)
        & (relative_depth_spread <= maximum_depth_spread)
    )
    if not bool(design_observation.any()):
        raise RuntimeError("V18 diagnostic gates removed every signature-design observation")
    print(
        json.dumps(
            {
                "stage": "clean_signature_design_observations",
                "raw_count": int(raw_design_observation.sum()),
                "clean_count": int(design_observation.sum()),
                "minimum_retained_composition_mass": minimum_retained_mass,
                "maximum_composition_entropy": maximum_entropy,
                "maximum_relative_depth_spread_3x3": maximum_depth_spread,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    anchor_provenance = aggregate_anchor_provenance(
        observation_offsets=anchor_offsets,
        observation_primitive_ids=primitive_ids,
        observation_weights=weights,
        observation_view_family_ids=mapping_families[observation_queries],
        observation_valid=design_observation,
    )
    print(json.dumps({"stage": "anchor_provenance_ready"}), flush=True)
    inverse = build_primitive_anchor_index(anchor_provenance)
    print(json.dumps({"stage": "primitive_anchor_index_ready"}), flush=True)
    common = {
        "primitive_ids": primitive_ids,
        "weights": weights,
        "valid": valid,
        "rendered_depth": rendered_depth,
        "composition_entropy": composition_entropy,
        "relative_depth_spread": relative_depth_spread,
        "retained_composition_fraction": retained_composition_fraction,
        "observation_queries": observation_queries,
        "observation_keypoints": observation_keypoints,
        "observation_anchor": observation_anchor,
        "mapping_keypoints": raw["mapping_keypoints"],
        "mapping_offsets": torch.as_tensor(raw["mapping_pixel_center_offset"]).float(),
        "mapping_intrinsics": torch.as_tensor(raw["mapping_intrinsics"]).float(),
        "mapping_poses": torch.as_tensor(raw["mapping_poses_w2c"]).float(),
        "inverse": inverse,
        "anchor_observation_offsets": anchor_offsets,
        "mapping_families": mapping_families,
        "design_observation_mask": design_observation,
        "anchor_xyz": state["anchor_xyz"],
        "anchor_covariance": state["anchor_position_covariance"],
        "maximum_candidates": args.maximum_candidates,
        "transport_residual_px": args.transport_residual_px,
        "minimum_transport_overlap": 0.05,
        "geometry_device": args.projection_device,
    }
    calibration = _prepare_split(rows=calibration_rows, **common)
    print(json.dumps({"stage": "calibration_graph_ready"}), flush=True)
    equivalence = torch.as_tensor(state.get("fine_identity_ids", torch.arange(anchor_count))).long()
    thresholds, selected_trial, trials = _choose_thresholds(
        graph=calibration["candidate_graph"],
        transport=calibration["transport"],
        geometry=calibration["geometry"],
        ground_truth_anchor=calibration["ground_truth_anchor_rows"],
        equivalence=equivalence,
        minimum_precision=args.minimum_calibration_precision,
        maximum_composition_entropy=maximum_entropy,
        maximum_relative_depth_spread=maximum_depth_spread,
    )
    print(json.dumps({"stage": "thresholds_calibrated"}), flush=True)
    validation = _prepare_split(rows=validation_rows, **common)
    print(json.dumps({"stage": "validation_graph_ready"}), flush=True)
    provenance_transport_truth = assign_provenance_truth(
        candidate_graph=validation["candidate_graph"],
        transport_evidence=validation["transport"],
        geometry_evidence=validation["geometry"],
        equivalence_class_ids=equivalence,
        thresholds=thresholds,
    )
    provenance_only_truth = assign_provenance_truth(
        candidate_graph=validation["candidate_graph"],
        transport_evidence=_fake_provenance_only_transport(validation["candidate_graph"]),
        equivalence_class_ids=equivalence,
        thresholds=thresholds,
    )
    observation_count = anchor_offsets[1:] - anchor_offsets[:-1]
    projection_truth = assign_full_map_projection_truth(
        keypoints=validation["keypoints"],
        rendered_depth=validation["depth"],
        query_indices=validation["queries"],
        anchor_xyz=state["anchor_xyz"],
        anchor_covariance=state["anchor_position_covariance"],
        observation_count=observation_count,
        mapping_intrinsics=raw["mapping_intrinsics"],
        mapping_poses_w2c=raw["mapping_poses_w2c"],
        equivalence_class_ids=equivalence,
        device=args.projection_device,
    )
    print(json.dumps({"stage": "projection_baseline_ready"}), flush=True)
    report = {
        "schema": "lafgs_v18_provenance_truth_validation",
        "version": 1,
        "uses_test_queries": False,
        "loo_used": False,
        "full_gaussian_prior_evaluated": True,
        "full_depth_ordered_compositing": True,
        "full_provenance_anchor_enumeration": int(args.maximum_candidates) == 0,
        "minimum_retained_composition_mass": float(
            raw["minimum_retained_composition_mass"]
        ),
        "split_policy": "disjoint_mapping_view_family_hash_60_20_20",
        "split_seed": int(args.split_seed),
        "mapping_query_role_counts": {
            role: roles.count(role)
            for role in ("signature_design", "threshold_calibration", "independent_validation")
        },
        "mapping_view_family_policy": raw["mapping_view_family_policy"],
        "mapping_view_family_registry": raw["mapping_view_family_registry"],
        "mapping_view_family_role_counts": {
            role: sum(value == role for value in family_roles.values())
            for role in (
                "signature_design",
                "threshold_calibration",
                "independent_validation",
            )
        },
        "mapping_observation_role_counts": {
            "signature_design": int(design_observation.sum()),
            "signature_design_before_diagnostic_gates": int(
                raw_design_observation.sum()
            ),
            "threshold_calibration": int(calibration_observation.sum()),
            "independent_validation": int(validation_observation.sum()),
        },
        "calibration": selected_trial,
        "calibration_trial_count": len(trials),
        "provenance_diagnostic_calibration": {
            "acceptance_quantile": diagnostic_quantile,
            "maximum_composition_entropy": maximum_entropy,
            "maximum_relative_depth_spread_3x3": maximum_depth_spread,
            "policy": "threshold_calibration_view_families_only",
        },
        "selected_thresholds": {
            name: getattr(thresholds, name)
            for name in TruthAssignmentThresholds.__dataclass_fields__
        },
        "validation": {
            "projection_only_full_map": _truth_metrics(
                projection_truth,
                validation["ground_truth_anchor_rows"],
                equivalence,
            ),
            "provenance_only": _truth_metrics(
                provenance_only_truth,
                validation["ground_truth_anchor_rows"],
                equivalence,
            ),
            "provenance_plus_transport": _truth_metrics(
                provenance_transport_truth,
                validation["ground_truth_anchor_rows"],
                equivalence,
            ),
        },
        "controller_replacement_authorized": False,
        "replacement_gate": {
            "minimum_validation_precision": float(args.minimum_calibration_precision),
            "must_add_correct_assignments_without_adding_wrong_assignments": True,
            "must_have_nonzero_coverage": True,
        },
        "inputs": {
            "anchor_map": str(args.anchor_map.resolve()),
            "anchor_map_sha256": sha256_file(args.anchor_map),
            "mapping_provenance": str(args.mapping_provenance.resolve()),
            "mapping_provenance_sha256": sha256_file(args.mapping_provenance),
        },
    }
    provenance_metrics = report["validation"]["provenance_plus_transport"]
    projection_metrics = report["validation"]["projection_only_full_map"]
    report["controller_replacement_authorized"] = bool(
        provenance_metrics["decisive_assignment_count"] > 0
        and provenance_metrics["decisive_precision"] >= float(args.minimum_calibration_precision)
        and provenance_metrics["correct_assignment_count"]
        > projection_metrics["correct_assignment_count"]
        and provenance_metrics["wrong_assignment_count"]
        <= projection_metrics["wrong_assignment_count"]
    )
    artifact = {
        **report,
        "anchor_provenance": anchor_provenance,
        "primitive_anchor_index": inverse,
        "signature_design_observation_mask": design_observation,
        "calibration_trials": trials,
        "validation_rows": validation_rows,
        "validation_ground_truth_anchor_rows": validation["ground_truth_anchor_rows"],
        "projection_truth": projection_truth,
        "provenance_only_truth": provenance_only_truth,
        "provenance_plus_transport_truth": provenance_transport_truth,
    }
    _atomic_save(artifact, args.output.resolve())
    report["output"] = str(args.output.resolve())
    report["output_sha256"] = sha256_file(args.output)
    report_path = args.output.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
