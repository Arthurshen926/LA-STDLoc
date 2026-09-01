#!/usr/bin/env python3
"""Calibrate/validate a selective Track-extension teacher by mapping families."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import itertools
import json
import math
import os
from pathlib import Path

import torch

from common.hashing import sha256_file
from map_learning.v18_provenance_truth import (
    TRUTH_EQUIVALENT,
    TRUTH_UNIQUE,
    backproject_query_surface,
)
from map_learning.v19_track_extension_teacher import (
    TrackExtensionTier,
    assign_track_extension_truth,
    full_map_projection_candidate_graph,
    track_observation_consensus,
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
        raise ValueError("V19 Track teacher requires at least five mapping families")
    ranked = sorted(
        unique,
        key=lambda family: hashlib.sha256(
            f"v19-track-extension:{int(seed)}:{int(family)}".encode()
        ).digest(),
    )
    design_count = max(1, round(0.60 * len(ranked)))
    calibration_count = max(1, round(0.20 * len(ranked)))
    if design_count + calibration_count >= len(ranked):
        design_count = len(ranked) - calibration_count - 1
    return {
        int(family): (
            "track_bank"
            if index < design_count
            else (
                "threshold_calibration"
                if index < design_count + calibration_count
                else "independent_validation"
            )
        )
        for index, family in enumerate(ranked)
    }


def _sample_rows(mask: torch.Tensor, maximum: int, seed: int) -> torch.Tensor:
    rows = torch.nonzero(mask, as_tuple=False).reshape(-1)
    if rows.numel() <= int(maximum):
        return rows
    generator = torch.Generator().manual_seed(int(seed))
    return rows[
        torch.randperm(rows.numel(), generator=generator)[: int(maximum)]
    ].sort().values


def _gather_rows(
    *,
    rows: torch.Tensor,
    observation_queries: torch.Tensor,
    observation_keypoints: torch.Tensor,
    mapping_keypoints: list[torch.Tensor],
    mapping_descriptors: list[torch.Tensor],
    mapping_offsets: torch.Tensor,
    mapping_depth: torch.Tensor,
    mapping_intrinsics: torch.Tensor,
    mapping_poses: torch.Tensor,
) -> dict:
    queries = observation_queries[rows]
    keypoint_rows = observation_keypoints[rows]
    keypoints = torch.empty((rows.numel(), 2))
    descriptors = torch.empty((rows.numel(), 256))
    for query in torch.unique(queries, sorted=True).tolist():
        local = torch.nonzero(queries == int(query), as_tuple=False).reshape(-1)
        indices = keypoint_rows[local]
        keypoints[local] = (
            torch.as_tensor(mapping_keypoints[query]).float()[indices]
            + float(mapping_offsets[query])
        )
        descriptors[local] = torch.as_tensor(mapping_descriptors[query]).float()[
            indices
        ]
    surface = torch.empty((rows.numel(), 3))
    surface_valid = torch.zeros(rows.numel(), dtype=torch.bool)
    for query in torch.unique(queries, sorted=True).tolist():
        local = torch.nonzero(queries == int(query), as_tuple=False).reshape(-1)
        xyz, valid = backproject_query_surface(
            keypoints[local],
            mapping_depth[rows[local]],
            mapping_intrinsics[query],
            mapping_poses[query],
        )
        surface[local] = xyz
        surface_valid[local] = valid
    return {
        "queries": queries,
        "keypoints": keypoints,
        "descriptors": descriptors,
        "depth": mapping_depth[rows],
        "surface": surface,
        "surface_valid": surface_valid,
    }


def _truth_metrics(
    truth: dict,
    ground_truth_anchor: torch.Tensor,
    equivalence: torch.Tensor,
    rows: torch.Tensor,
) -> dict:
    status = torch.as_tensor(truth["truth_status"]).long()
    offsets = torch.as_tensor(truth["truth_offsets"]).long()
    predicted = torch.as_tensor(truth["truth_anchor_rows"]).long()
    gt = torch.as_tensor(ground_truth_anchor).long()
    selected = torch.as_tensor(rows).long()
    decisive = (status == TRUTH_UNIQUE) | (status == TRUTH_EQUIVALENT)
    prediction_rows = torch.repeat_interleave(
        torch.arange(status.numel()), offsets[1:] - offsets[:-1]
    )
    correct_edges = (
        equivalence[predicted] == equivalence[gt[prediction_rows]]
    )
    correct_rows = torch.bincount(
        prediction_rows[correct_edges], minlength=status.numel()
    ) > 0
    selected_decisive = decisive[selected]
    decisive_count = int(selected_decisive.sum())
    correct_count = int((correct_rows[selected] & selected_decisive).sum())
    wrong_count = decisive_count - correct_count
    return {
        "row_count": int(selected.numel()),
        "decisive_assignment_count": decisive_count,
        "correct_assignment_count": correct_count,
        "wrong_assignment_count": wrong_count,
        "decisive_coverage": float(decisive_count / max(selected.numel(), 1)),
        "decisive_precision": float(correct_count / max(decisive_count, 1)),
    }


def _wilson_lower_bound(correct: int, total: int, z: float = 1.6448536269514722) -> float:
    """One-sided Wilson lower bound (95% by default) for decisive precision."""

    if total <= 0:
        return 0.0
    probability = float(correct) / float(total)
    denominator = 1.0 + z * z / total
    centre = probability + z * z / (2.0 * total)
    radius = z * math.sqrt(
        probability * (1.0 - probability) / total + z * z / (4.0 * total * total)
    )
    return float((centre - radius) / denominator)


def _calibration_safety(
    *,
    truth: dict,
    ground_truth_anchor: torch.Tensor,
    equivalence: torch.Tensor,
    rows: torch.Tensor,
    row_families: torch.Tensor,
    target_precision: float,
    minimum_decisive_assignments: int,
    minimum_active_families: int,
    minimum_assignments_per_family: int = 10,
) -> dict:
    pooled = _truth_metrics(truth, ground_truth_anchor, equivalence, rows)
    family_metrics = {}
    active_families = 0
    for family in torch.unique(row_families[rows], sorted=True).tolist():
        family_rows = rows[row_families[rows] == int(family)]
        metrics = _truth_metrics(
            truth, ground_truth_anchor, equivalence, family_rows
        )
        metrics["precision_wilson_lower_95"] = _wilson_lower_bound(
            metrics["correct_assignment_count"],
            metrics["decisive_assignment_count"],
        )
        family_metrics[int(family)] = metrics
        active_families += int(
            metrics["decisive_assignment_count"]
            >= int(minimum_assignments_per_family)
        )
    pooled["precision_wilson_lower_95"] = _wilson_lower_bound(
        pooled["correct_assignment_count"],
        pooled["decisive_assignment_count"],
    )
    pooled["active_family_count"] = active_families
    pooled["family_metrics"] = family_metrics
    pooled["authorization_requirements"] = {
        "target_precision_wilson_lower_95": float(target_precision),
        "minimum_decisive_assignments": int(minimum_decisive_assignments),
        "minimum_active_families": int(minimum_active_families),
        "minimum_assignments_per_family": int(minimum_assignments_per_family),
    }
    pooled["authorized"] = bool(
        pooled["precision_wilson_lower_95"] >= float(target_precision)
        and pooled["decisive_assignment_count"]
        >= int(minimum_decisive_assignments)
        and active_families >= int(minimum_active_families)
    )
    return pooled


def _candidate_tiers() -> list[TrackExtensionTier]:
    tiers = []
    for reprojection, depth, transport, families, cosine, descriptor_families in itertools.product(
        (1.0, 2.0, 3.0, 4.0),
        (0.50, 0.75, 1.00),
        (1.5, 2.5, 4.0),
        (2, 3),
        (0.60, 0.70, 0.80, 0.90),
        (1, 2),
    ):
        tiers.append(
            TrackExtensionTier(
                maximum_query_reprojection_px=reprojection,
                maximum_query_normalized_depth_residual=depth,
                maximum_query_projection_std_px=min(2.0, reprojection),
                maximum_transport_median_residual_px=transport,
                minimum_transport_view_families=families,
                minimum_descriptor_cosine=cosine,
                minimum_descriptor_view_families=descriptor_families,
            )
        )
    return tiers


def _select_tiers(
    *,
    graph: dict,
    consensus: dict,
    equivalence: torch.Tensor,
    ground_truth: torch.Tensor,
    calibration_rows: torch.Tensor,
    validation_rows: torch.Tensor,
    evaluation_families: torch.Tensor,
) -> tuple[dict, int]:
    trials = []
    for tier in _candidate_tiers():
        truth = assign_track_extension_truth(
            candidate_graph=graph,
            consensus=consensus,
            equivalence_class_ids=equivalence,
            tier=tier,
        )
        trials.append({"tier": tier, "truth": truth})
    selected = {}
    tier_contracts = {
        "tier_a": (0.99, 300, 2, ["destructive_map_control"]),
        "tier_b": (0.97, 200, 2, ["strong_metric_control"]),
        "tier_c": (0.90, 50, 1, ["soft_diagnostic", "planner_priority"]),
    }
    for name, (
        target,
        minimum_assignments,
        minimum_families,
        permitted_actions,
    ) in tier_contracts.items():
        evaluated = []
        for item in trials:
            safety = _calibration_safety(
                truth=item["truth"],
                ground_truth_anchor=ground_truth,
                equivalence=equivalence,
                rows=calibration_rows,
                row_families=evaluation_families,
                target_precision=target,
                minimum_decisive_assignments=minimum_assignments,
                minimum_active_families=minimum_families,
            )
            evaluated.append({**item, "calibration": safety})
        eligible = [item for item in evaluated if item["calibration"]["authorized"]]
        pool = eligible or [
            item
            for item in evaluated
            if item["calibration"]["decisive_assignment_count"] > 0
        ]
        choice = max(
            pool,
            key=lambda item: (
                item["calibration"]["decisive_coverage"]
                if eligible
                else item["calibration"]["precision_wilson_lower_95"],
                item["calibration"]["active_family_count"],
                item["calibration"]["decisive_precision"],
                item["calibration"]["decisive_coverage"],
            ),
        )
        selected[name] = {
            "target_precision": target,
            "target_met_on_calibration": bool(choice["calibration"]["authorized"]),
            "permitted_actions_if_authorized": permitted_actions,
            "authorized_actions": (
                permitted_actions if choice["calibration"]["authorized"] else []
            ),
            "thresholds": choice["tier"].__dict__,
            "calibration": choice["calibration"],
            "validation": _truth_metrics(
                choice["truth"], ground_truth, equivalence, validation_rows
            ),
            "truth": choice["truth"],
        }
    return selected, len(trials)


def _reference_edge_diagnostics(
    *,
    graph: dict,
    consensus: dict,
    ground_truth: torch.Tensor,
    equivalence: torch.Tensor,
    rows: torch.Tensor,
) -> dict:
    offsets = torch.as_tensor(graph["candidate_offsets"]).long()
    candidates = torch.as_tensor(graph["candidate_anchor_rows"]).long()
    reprojection = torch.as_tensor(graph["query_reprojection_residual_px"]).float()
    depth = torch.as_tensor(graph["query_normalized_depth_residual"]).float()
    families = torch.as_tensor(consensus["transport_view_family_count"]).long()
    residual = torch.as_tensor(consensus["transport_median_residual_px"]).float()
    descriptor_families = torch.as_tensor(
        consensus["descriptor_view_family_count"]
    ).long()
    cosine = torch.as_tensor(consensus["descriptor_best_cosine"]).float()
    found_rows = 0
    values = defaultdict(list)
    for row in torch.as_tensor(rows).long().tolist():
        start, stop = int(offsets[row]), int(offsets[row + 1])
        if stop <= start:
            continue
        local = torch.nonzero(
            equivalence[candidates[start:stop]] == equivalence[ground_truth[row]],
            as_tuple=False,
        ).reshape(-1)
        if local.numel() == 0:
            continue
        found_rows += 1
        edge = start + int(local[0])
        values["query_reprojection_px"].append(float(reprojection[edge]))
        values["query_normalized_depth"].append(float(depth[edge]))
        values["transport_view_families"].append(float(families[edge]))
        values["transport_median_residual_px"].append(float(residual[edge]))
        values["descriptor_view_families"].append(
            float(descriptor_families[edge])
        )
        values["descriptor_best_cosine"].append(float(cosine[edge]))
    diagnostics = {
        "row_count": int(torch.as_tensor(rows).numel()),
        "reference_in_projection_candidates_count": found_rows,
        "reference_in_projection_candidates_fraction": float(
            found_rows / max(torch.as_tensor(rows).numel(), 1)
        ),
    }
    quantiles = torch.tensor([0.0, 0.25, 0.5, 0.75, 0.9, 1.0])
    for name, raw in values.items():
        tensor = torch.tensor(raw)
        finite = tensor[torch.isfinite(tensor)]
        diagnostics[name] = {
            "finite_count": int(finite.numel()),
            "quantiles_0_25_50_75_90_100": (
                [] if finite.numel() == 0 else torch.quantile(finite, quantiles).tolist()
            ),
        }
    return diagnostics


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor-map", type=Path, required=True)
    parser.add_argument("--mapping-provenance", type=Path, required=True)
    parser.add_argument("--mapping-feature-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split-seed", type=int, default=1920260830)
    parser.add_argument("--maximum-calibration-rows", type=int, default=4000)
    parser.add_argument("--maximum-validation-rows", type=int, default=4000)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    state = torch.load(args.anchor_map, map_location="cpu", weights_only=False)
    provenance = torch.load(
        args.mapping_provenance, map_location="cpu", weights_only=False
    )
    cache = torch.load(
        args.mapping_feature_cache, map_location="cpu", weights_only=False
    )
    records = cache.get("queries", cache)
    names = list(state["v6_mapping_query_names"])
    if not (
        provenance.get("schema")
        == "lafgs_v18_mapping_observation_gaussian_provenance"
        and provenance.get("uses_test_queries") is False
        and provenance.get("loo_used") is False
        and cache.get("uses_test_queries") is False
        and cache.get("uses_source_mapping_rgb") is False
        and names == list(records)
        and names == list(provenance["mapping_query_names"])
    ):
        raise ValueError("V19 Track-extension frozen input contract differs")
    observations = state["projective_anchor_observations"]
    anchor_offsets = torch.as_tensor(observations["observation_offsets"]).long()
    observation_queries = torch.as_tensor(observations["query_indices"]).long()
    observation_keypoints = torch.as_tensor(observations["keypoint_indices"]).long()
    edge_count = int(observation_queries.numel())
    provenance_rows = torch.as_tensor(provenance["observation_rows"]).long()
    if (
        provenance_rows.numel() != edge_count
        or not torch.equal(
            provenance_rows.sort().values, torch.arange(edge_count)
        )
    ):
        raise ValueError("V19 Track teacher requires complete mapping provenance")
    observation_valid = torch.zeros(edge_count, dtype=torch.bool)
    observation_depth = torch.full((edge_count,), float("nan"))
    observation_valid[provenance_rows] = torch.as_tensor(
        provenance["observation_valid"]
    ).bool()
    observation_depth[provenance_rows] = torch.as_tensor(
        provenance["observation_rendered_depth"]
    ).float()
    anchor_count = int(torch.as_tensor(state["anchor_ids"]).numel())
    observation_anchor = torch.repeat_interleave(
        torch.arange(anchor_count), anchor_offsets[1:] - anchor_offsets[:-1]
    )
    mapping_keypoints = [
        torch.as_tensor(records[name]["native_keypoints"]).float()
        for name in names
    ]
    mapping_descriptors = [
        torch.as_tensor(records[name]["native_descriptors"]).float()
        for name in names
    ]
    mapping_offsets = torch.as_tensor(
        provenance["mapping_pixel_center_offset"]
    ).float()
    mapping_families = torch.as_tensor(
        provenance["mapping_view_family_ids"]
    ).long()
    roles = _balanced_family_roles(mapping_families, args.split_seed)
    query_roles = [roles[int(family)] for family in mapping_families.tolist()]
    track_bank_mask = torch.tensor(
        [query_roles[int(query)] == "track_bank" for query in observation_queries]
    ) & observation_valid
    calibration_mask = torch.tensor(
        [
            query_roles[int(query)] == "threshold_calibration"
            for query in observation_queries
        ]
    ) & observation_valid
    validation_mask = torch.tensor(
        [
            query_roles[int(query)] == "independent_validation"
            for query in observation_queries
        ]
    ) & observation_valid
    calibration_rows = _sample_rows(
        calibration_mask, args.maximum_calibration_rows, args.split_seed + 1
    )
    validation_rows = _sample_rows(
        validation_mask, args.maximum_validation_rows, args.split_seed + 2
    )
    rows = torch.cat((calibration_rows, validation_rows))
    gathered = _gather_rows(
        rows=rows,
        observation_queries=observation_queries,
        observation_keypoints=observation_keypoints,
        mapping_keypoints=mapping_keypoints,
        mapping_descriptors=mapping_descriptors,
        mapping_offsets=mapping_offsets,
        mapping_depth=observation_depth,
        mapping_intrinsics=provenance["mapping_intrinsics"],
        mapping_poses=provenance["mapping_poses_w2c"],
    )
    observation_count = anchor_offsets[1:] - anchor_offsets[:-1]
    graph = full_map_projection_candidate_graph(
        keypoints=gathered["keypoints"],
        rendered_depth=gathered["depth"],
        query_indices=gathered["queries"],
        anchor_xyz=state["anchor_xyz"],
        anchor_covariance=state["anchor_position_covariance"],
        observation_count=observation_count,
        query_intrinsics=provenance["mapping_intrinsics"],
        query_poses_w2c=provenance["mapping_poses_w2c"],
        device=args.device,
    )
    graph["query_valid"] &= gathered["surface_valid"]
    consensus = track_observation_consensus(
        candidate_graph=graph,
        query_surface_xyz=gathered["surface"],
        query_descriptors=gathered["descriptors"],
        anchor_observation_offsets=anchor_offsets,
        observation_query_indices=observation_queries,
        observation_keypoint_indices=observation_keypoints,
        observation_enabled=track_bank_mask,
        mapping_keypoints=[
            value + float(mapping_offsets[index])
            for index, value in enumerate(mapping_keypoints)
        ],
        mapping_descriptors=mapping_descriptors,
        mapping_intrinsics=provenance["mapping_intrinsics"],
        mapping_poses_w2c=provenance["mapping_poses_w2c"],
        mapping_view_family_ids=mapping_families,
        device=args.device,
    )
    equivalence = torch.as_tensor(
        state.get("fine_identity_ids", torch.arange(anchor_count))
    ).long()
    ground_truth = observation_anchor[rows]
    calibration_local = torch.arange(calibration_rows.numel())
    validation_local = torch.arange(calibration_rows.numel(), rows.numel())
    evaluation_families = mapping_families[gathered["queries"]]
    selected, candidate_trial_count = _select_tiers(
        graph=graph,
        consensus=consensus,
        equivalence=equivalence,
        ground_truth=ground_truth,
        calibration_rows=calibration_local,
        validation_rows=validation_local,
        evaluation_families=evaluation_families,
    )
    artifact = {
        "schema": "lafgs_v19_track_extension_teacher_validation",
        "version": 1,
        "uses_test_queries": False,
        "loo_used": False,
        "feedback_enters_track_registry": False,
        "reference_source": "mapping_observation_track_membership",
        "reference_available_for_novel_query": False,
        "selection_uses_validation": False,
        "authorization_uses_wilson_lower_bound": True,
        "authorization_requires_independent_mapping_families": True,
        "split_policy": "disjoint_mapping_view_family_hash_60_20_20",
        "split_seed": int(args.split_seed),
        "family_roles": roles,
        "row_counts": {
            "track_bank": int(track_bank_mask.sum()),
            "threshold_calibration": int(calibration_rows.numel()),
            "independent_validation": int(validation_rows.numel()),
        },
        "candidate_trial_count": candidate_trial_count,
        "reference_edge_diagnostics": {
            "calibration": _reference_edge_diagnostics(
                graph=graph,
                consensus=consensus,
                ground_truth=ground_truth,
                equivalence=equivalence,
                rows=calibration_local,
            ),
            "validation": _reference_edge_diagnostics(
                graph=graph,
                consensus=consensus,
                ground_truth=ground_truth,
                equivalence=equivalence,
                rows=validation_local,
            ),
        },
        "selected_tiers": selected,
        "evaluation_rows": rows,
        "ground_truth_anchor_rows": ground_truth,
        "evaluation_view_family_ids": evaluation_families,
        "candidate_graph": graph,
        "consensus": consensus,
        "inputs": {
            "anchor_map": str(args.anchor_map.resolve()),
            "anchor_map_sha256": sha256_file(args.anchor_map),
            "mapping_provenance": str(args.mapping_provenance.resolve()),
            "mapping_provenance_sha256": sha256_file(args.mapping_provenance),
            "mapping_feature_cache": str(args.mapping_feature_cache.resolve()),
            "mapping_feature_cache_sha256": sha256_file(
                args.mapping_feature_cache
            ),
        },
    }
    _atomic_save(artifact, args.output.resolve())
    report = {
        key: value
        for key, value in artifact.items()
        if key
        not in {
            "selected_tiers",
            "evaluation_rows",
            "ground_truth_anchor_rows",
            "evaluation_view_family_ids",
            "candidate_graph",
            "consensus",
        }
    }
    report["selected_tiers"] = {
        name: {key: value for key, value in item.items() if key != "truth"}
        for name, item in selected.items()
    }
    report["output"] = str(args.output.resolve())
    report["output_sha256"] = sha256_file(args.output)
    args.output.with_suffix(".json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
