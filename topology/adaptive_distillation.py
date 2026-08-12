#!/usr/bin/env python3
"""Adaptive Track/Gaussian topology distillation for the LaFGS V2 mainline."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path

import torch

from common.calibration import (
    derive_adaptive_parameters,
    derive_mapping_statistics,
)
from common.config import load_mainline_config
from evidence.tracks import fuse_track_descriptors
from map_learning.observations import _query_index_remap
from topology.dynamic_reserve import PoseEvidence, spatial_voxel_ids
from topology.geometry_materializer import (
    materialize_track_geometry_compatibility,
)
from topology.matching_coverage import (
    IncrementalBipartiteCoverage,
    base_candidate_edges,
    query_weights_from_groups,
    track_candidate_edges,
)
from topology.pose_information import (
    fisher_contributions,
    pose_jacobian_analytic,
    task_scaled_pose_jacobian,
)
from topology.sufficiency_selector import CompatibilitySufficiencySelector
from topology.track_core import (
    _base_utility,
    _graph_counter,
    _materialize,
    _track_quality,
    _track_source_ids,
)


def _adaptive_track_eligibility(
    geometry: dict,
    *,
    median_px: float,
    p90_px: float,
    covariance_m2: float,
    broad: bool,
) -> torch.Tensor:
    factor = 5.0 if broad else 1.0
    parallax = 0.5 if broad else 1.0
    xyz = torch.as_tensor(geometry["triangulated_xyz"]).float()
    return (
        torch.as_tensor(geometry["triangulated"]).bool()
        & torch.isfinite(xyz).all(dim=1)
        & (
            torch.as_tensor(geometry["triangulation_distinct_view_bin_count"])
            >= 2
        )
        & (
            torch.as_tensor(geometry["triangulation_reprojection_median_px"])
            <= float(median_px) * (4.0 / 3.0 if broad else 1.0)
        )
        & (
            torch.as_tensor(geometry["triangulation_reprojection_p90_px"])
            <= float(p90_px) * (5.0 / 3.0 if broad else 1.0)
        )
        & (
            torch.as_tensor(geometry["triangulation_covariance_trace"])
            <= float(covariance_m2) * factor
        )
        & (
            torch.as_tensor(geometry["triangulation_parallax_deg"])
            >= parallax
        )
    )


def _image_only_core_eligibility(
    geometry: dict,
    *,
    median_px: float,
    p90_px: float,
    covariance_m2: float,
) -> torch.Tensor:
    """Keep surface-promoted tracks out of the immutable Track Core."""
    if "triangulation_image_only_covariance_trace" not in geometry:
        return _adaptive_track_eligibility(
            geometry,
            median_px=median_px,
            p90_px=p90_px,
            covariance_m2=covariance_m2,
            broad=False,
        )
    image_geometry = dict(geometry)
    image_geometry["triangulation_covariance_trace"] = geometry[
        "triangulation_image_only_covariance_trace"
    ]
    image_geometry["triangulation_reprojection_median_px"] = geometry[
        "triangulation_image_only_reprojection_median_px"
    ]
    image_geometry["triangulation_reprojection_p90_px"] = geometry[
        "triangulation_image_only_reprojection_p90_px"
    ]
    return _adaptive_track_eligibility(
        image_geometry,
        median_px=median_px,
        p90_px=p90_px,
        covariance_m2=covariance_m2,
        broad=False,
    )


def _deployment_track_geometry(
    geometry: dict,
    image_only_core: torch.Tensor,
) -> dict:
    """Deploy surface fusion only for tracks promoted out of the reserve.

    A track that already passes the image-only core gate does not need prior
    depth to become a localization landmark.  Retaining its image-only point
    and uncertainty prevents the frozen RGB prior from perturbing reliable
    PnP geometry.  Surface-fused geometry remains available to weak tracks and
    must still pass the matching/pose reserve selection.
    """
    return materialize_track_geometry_compatibility(
        geometry, image_only_core
    )


def _mean_track_confidence(payload: dict) -> torch.Tensor:
    track_count = int(
        torch.as_tensor(payload["track_geometry"]["triangulated"]).numel()
    )
    observations = payload["tracks"]
    track = torch.as_tensor(observations["track_index"]).long()
    confidence = torch.as_tensor(observations["confidence"]).float().clamp(0, 1)
    total = torch.zeros(track_count)
    count = torch.zeros(track_count)
    total.index_add_(0, track, confidence)
    count.index_add_(0, track, torch.ones_like(confidence))
    return total / count.clamp_min(1)


def _candidate_matchability(
    payload: dict, graph: dict, base_count: int, track_threshold_px: float
) -> torch.Tensor:
    geometry = payload["track_geometry"]
    confidence = _mean_track_confidence(payload)
    level = torch.as_tensor(geometry["track_confidence_level"]).long()
    level_factor = torch.where(
        level >= 2,
        torch.ones_like(confidence),
        torch.where(level == 1, 0.8 * torch.ones_like(confidence), 0.6 * torch.ones_like(confidence)),
    )
    reprojection = torch.as_tensor(
        geometry["triangulation_reprojection_median_px"]
    ).float()
    track_probability = confidence * level_factor * torch.exp(
        -reprojection / max(float(track_threshold_px), 1e-6)
    )
    opportunity = torch.as_tensor(
        graph["provenance_opportunity_count"][:base_count]
    ).float()
    legal = _graph_counter(
        graph,
        "provenance_legal_hit_strong_count",
        "provenance_legal_hit_2px_count",
    )[:base_count].float()
    harmful = torch.as_tensor(
        graph["provenance_harmful_solver_inlier_count"][:base_count]
    ).float()
    base_probability = ((legal + 1.0) / (opportunity + 2.0)) * (
        1.0 - harmful / opportunity.clamp_min(1)
    ).clamp_min(0)
    return torch.cat(
        (track_probability.clamp(0.02, 0.98), base_probability.clamp(0.02, 0.98))
    )


def _ordered_rows(record: dict, query_cache: dict) -> tuple[int, ...]:
    rows = tuple(int(value) for value in record)
    if len(rows) < 2:
        return rows
    score = torch.as_tensor(query_cache["native_scores"])
    return tuple(sorted(rows, key=lambda row: (-float(score[row]), row)))


def _query_row_matchability(cached: dict, rows: tuple[int, ...]) -> float:
    """Return a query-conditioned repeatability factor for one candidate edge."""
    if not rows:
        return 0.0
    score = torch.as_tensor(cached["native_scores"]).float().reshape(-1)
    finite = score[torch.isfinite(score)]
    if finite.numel() < 2:
        return 1.0
    low = torch.quantile(finite, 0.1)
    high = torch.quantile(finite, 0.9)
    selected = score[torch.as_tensor(rows).long()].max()
    normalized = ((selected - low) / (high - low).clamp_min(1e-6)).clamp(0, 1)
    # Do not discard weak detections completely: global track/graph evidence is
    # still informative, while the query factor downweights fragile views.
    return float(0.5 + 0.5 * normalized)


def _project_world_covariance(
    points: torch.Tensor,
    covariance_world: torch.Tensor,
    intrinsic: torch.Tensor,
    pose_w2c: torch.Tensor,
) -> torch.Tensor:
    """Project anisotropic 3D landmark covariance into image coordinates."""
    camera = points @ pose_w2c[:3, :3].T + pose_w2c[:3, 3]
    x, y, z = camera.unbind(dim=1)
    z = z.clamp_min(1e-8)
    dproj = points.new_zeros((points.shape[0], 2, 3))
    dproj[:, 0, 0] = intrinsic[0, 0] / z
    dproj[:, 0, 2] = -intrinsic[0, 0] * x / z.square()
    dproj[:, 1, 1] = intrinsic[1, 1] / z
    dproj[:, 1, 2] = -intrinsic[1, 1] * y / z.square()
    jacobian = dproj @ pose_w2c[:3, :3]
    projected = jacobian @ covariance_world @ jacobian.transpose(1, 2)
    return 0.5 * (projected + projected.transpose(1, 2))


def _build_pose_evidence(
    edges,
    xyz: torch.Tensor,
    matchability: torch.Tensor,
    track_covariance: torch.Tensor,
    query_names: list[str],
    query_cache: dict,
    voxel_ids: torch.Tensor,
    *,
    pixel_variance: float,
    task_translation_m: float,
    task_rotation_deg: float,
) -> list[list[PoseEvidence]]:
    query_candidates: list[list[int]] = [[] for _ in query_names]
    for candidate, candidate_edges in enumerate(edges):
        for query in candidate_edges:
            query_candidates[int(query)].append(candidate)
    evidence: list[list[PoseEvidence]] = [[] for _ in edges]
    track_count = int(track_covariance.shape[0])
    for query, candidates in enumerate(query_candidates):
        if not candidates:
            continue
        name = query_names[query]
        cached = query_cache[name]
        candidate_tensor = torch.as_tensor(candidates).long()
        points = xyz[candidate_tensor].double()
        K = torch.as_tensor(cached["native_K"]).double()
        pose = torch.as_tensor(cached["pose_w2c"]).double()
        jacobian = task_scaled_pose_jacobian(
            pose_jacobian_analytic(points, K, pose),
            translation_scale=float(task_translation_m),
            rotation_scale=math.radians(float(task_rotation_deg)),
        )
        ones = torch.ones((points.shape[0], 1), dtype=torch.float64)
        depth = (pose @ torch.cat((points, ones), dim=1).T)[2]
        covariance = torch.eye(2, dtype=torch.float64)[None].repeat(
            points.shape[0], 1, 1
        ) * float(pixel_variance)
        track_mask = candidate_tensor < track_count
        if bool(track_mask.any()):
            selected_covariance = track_covariance[
                candidate_tensor[track_mask]
            ].double()
            if selected_covariance.ndim == 1:
                selected_covariance = torch.diag_embed(
                    (selected_covariance / 3.0)[:, None].expand(-1, 3)
                )
            covariance[track_mask] += _project_world_covariance(
                points[track_mask], selected_covariance, K, pose
            )
        row_matchability = []
        ordered_rows = []
        for candidate in candidates:
            rows = _ordered_rows(edges[candidate][query], cached)
            ordered_rows.append(rows)
            row_matchability.append(_query_row_matchability(cached, rows))
        query_matchability = matchability[candidate_tensor].double() * torch.as_tensor(
            row_matchability, dtype=torch.float64
        )
        information = fisher_contributions(
            jacobian,
            weights=query_matchability,
            measurement_covariance=covariance,
        )
        keypoints = torch.as_tensor(cached["native_keypoints"]).float()
        height, width = (int(value) for value in cached["native_input_hw"])
        for local, candidate in enumerate(candidates):
            rows = ordered_rows[local]
            if not rows:
                continue
            point = keypoints[rows[0]]
            cell_x = int(torch.floor(point[0] / max(width, 1) * 4).clamp(0, 3))
            cell_y = int(torch.floor(point[1] / max(height, 1) * 4).clamp(0, 3))
            depth_bin = int(torch.floor(torch.log2(depth[local].clamp_min(0.05)) * 2))
            evidence[candidate].append(
                PoseEvidence(
                    query=query,
                    rows=rows,
                    information=information[local],
                    image_cell=cell_y * 4 + cell_x,
                    depth_bin=depth_bin,
                    spatial_voxel=int(voxel_ids[candidate]),
                    matchability=float(query_matchability[local]),
                )
            )
    return evidence


def _initial_pose_state(
    selected: torch.Tensor,
    matching: IncrementalBipartiteCoverage,
    evidence_by_candidate,
    query_count: int,
) -> tuple[
    torch.Tensor,
    list[set[int]],
    list[set[int]],
    list[set[int]],
    list[set[int]],
    list[dict[int, int]],
]:
    information = torch.eye(6, dtype=torch.float64)[None].repeat(query_count, 1, 1)
    information *= 1e-4
    rows = [set() for _ in range(query_count)]
    cells = [set() for _ in range(query_count)]
    depths = [set() for _ in range(query_count)]
    voxels = [set() for _ in range(query_count)]
    selected_set = set(torch.as_tensor(selected).long().tolist())
    assignments = [dict() for _ in range(query_count)]
    evidence_lookup = [
        {item.query: item for item in candidate} for candidate in evidence_by_candidate
    ]
    for query in range(query_count):
        for candidate, row in matching.assignments(query).items():
            if candidate not in selected_set:
                continue
            item = evidence_lookup[candidate].get(query)
            if item is None:
                continue
            information[query] += item.information
            rows[query].add(int(row))
            assignments[query][int(candidate)] = int(row)
            cells[query].add(item.image_cell)
            depths[query].add(item.depth_bin)
            voxels[query].add(item.spatial_voxel)
    return information, rows, cells, depths, voxels, assignments


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-map", required=True)
    parser.add_argument("--function-graph", required=True)
    parser.add_argument("--complete-positive-teacher", required=True)
    parser.add_argument("--track-payload", required=True)
    parser.add_argument("--query-cache", required=True)
    parser.add_argument("--alias-risk-audit")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", default="configs/paper_mainline.yaml")
    args = parser.parse_args()

    canonical_path = Path(args.canonical_map).resolve()
    graph_path = Path(args.function_graph).resolve()
    teacher_path = Path(args.complete_positive_teacher).resolve()
    payload_path = Path(args.track_payload).resolve()
    query_path = Path(args.query_cache).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config = load_mainline_config(args.config).values
    if int(config["version"]) < 2:
        raise ValueError("adaptive distillation requires a V2 mainline config")
    policy = config["adaptive"]

    canonical = torch.load(canonical_path, map_location="cpu", weights_only=False)
    graph = torch.load(graph_path, map_location="cpu", weights_only=False)
    teacher = torch.load(teacher_path, map_location="cpu", weights_only=False)
    payload = torch.load(payload_path, map_location="cpu", weights_only=False)
    query_payload = torch.load(query_path, map_location="cpu", weights_only=False)
    query_cache = query_payload.get("queries", query_payload)
    statistics = derive_mapping_statistics(
        query_payload,
        payload,
        track_residual_quantile=float(
            policy.get("ransac_track_residual_quantile", 0.975)
        ),
    )
    parameters = derive_adaptive_parameters(statistics, policy)
    calibration = {
        "schema": "lafgs_mapping_only_scene_calibration",
        "version": 2,
        "statistics": asdict(statistics),
        "parameters": asdict(parameters),
        "policy": dict(policy),
        "uses_test_queries": False,
    }
    (output_dir / "scene_calibration.json").write_text(
        json.dumps(calibration, indent=2, sort_keys=True) + "\n"
    )

    base_count = int(canonical["base_anchor_count"])
    track_count = int(
        torch.as_tensor(payload["track_geometry"]["triangulated"]).numel()
    )
    payload_to_teacher = _query_index_remap(
        payload["query_names"], teacher["query_names"]
    )
    query_groups = torch.empty_like(torch.as_tensor(payload["query_bins"]))
    query_groups[payload_to_teacher] = torch.as_tensor(payload["query_bins"])
    track_edges = track_candidate_edges(
        payload, query_index_remap=payload_to_teacher
    )
    base_edges = base_candidate_edges(teacher, base_count)
    edges = [*track_edges, *base_edges]
    alias_risk = None
    alias_risk_contract = None
    if args.alias_risk_audit:
        alias_path = Path(args.alias_risk_audit).resolve()
        alias = torch.load(alias_path, map_location="cpu", weights_only=False)
        if alias.get("schema") != "lafgs_all_candidate_alias_risk_audit":
            raise ValueError("unsupported all-candidate alias-risk artifact")
        if alias.get("uses_test_queries") is not False:
            raise ValueError("selector alias risk must be mapping-only")
        universe_ids = torch.as_tensor(alias["candidate_universe_ids"]).long()
        values = torch.as_tensor(alias["risk"]["alias_risk"]).float()
        if universe_ids.numel() != values.numel():
            raise ValueError("alias-risk IDs and values do not align")
        if universe_ids.numel() and (
            int(universe_ids.min()) < 0 or int(universe_ids.max()) >= len(edges)
        ):
            raise ValueError("alias-risk candidate is outside selector universe")
        if torch.unique(universe_ids).numel() != universe_ids.numel():
            raise ValueError("alias-risk candidate IDs are not unique")
        alias_risk = torch.full((len(edges),), float("nan"))
        alias_risk[universe_ids] = values
        alias_risk_contract = {
            "path": str(alias_path),
            "risk_definition": alias["risk_definition"],
            "candidate_count": int(universe_ids.numel()),
            "supported_candidate_count": int(torch.isfinite(values).sum()),
            "selection_scope": "matching_completion_equal_gain_tiebreak",
            "unknown_policy": "after_supported_risk",
        }

    geometry = payload["track_geometry"]
    quality = _track_quality(geometry)
    medium = _adaptive_track_eligibility(
        geometry,
        median_px=parameters.track_reprojection_median_px,
        p90_px=parameters.track_reprojection_p90_px,
        covariance_m2=parameters.track_covariance_trace_m2,
        broad=False,
    )
    broad = _adaptive_track_eligibility(
        geometry,
        median_px=parameters.track_reprojection_median_px,
        p90_px=parameters.track_reprojection_p90_px,
        covariance_m2=parameters.track_covariance_trace_m2,
        broad=True,
    )
    image_only_core = _image_only_core_eligibility(
        geometry,
        median_px=parameters.track_reprojection_median_px,
        p90_px=parameters.track_reprojection_p90_px,
        covariance_m2=parameters.track_covariance_trace_m2,
    )
    deployment_geometry = _deployment_track_geometry(geometry, image_only_core)
    deployment_quality = _track_quality(deployment_geometry)
    quality = torch.where(image_only_core, deployment_quality, quality)
    deployment_payload = dict(payload)
    deployment_payload["track_geometry"] = deployment_geometry
    order = torch.argsort(quality, descending=True, stable=True)
    medium_order = order[(medium & image_only_core)[order]]
    selector = CompatibilitySufficiencySelector(
        edges,
        len(teacher["query_names"]),
        track_candidate_count=track_count,
    )
    core, core_report = selector.select_precision(
        medium_order,
        parameters.matching_rows_target,
        minimum_count=int(policy["track_core_minimum"]),
        maximum_count=min(int(policy["track_core_maximum"]), int(medium_order.numel())),
        check_interval=int(policy["track_core_check_interval"]),
    )

    opportunity = torch.as_tensor(
        graph["provenance_opportunity_count"][:base_count]
    ).float()
    harmful = torch.as_tensor(
        graph["provenance_harmful_solver_inlier_count"][:base_count]
    ).float()
    harmful_rate = harmful / opportunity.clamp_min(1)
    base_eligible = (
        _graph_counter(
            graph,
            "provenance_legal_hit_strong_count",
            "provenance_legal_hit_2px_count",
        )[:base_count]
        > 0
    ) & (harmful_rate <= float(policy["maximum_harmful_rate"]))
    core_mask = torch.zeros(track_count, dtype=torch.bool)
    core_mask[core] = True
    reserve_track_ids = torch.nonzero(broad & ~core_mask, as_tuple=False).reshape(-1)
    reserve_base_ids = torch.nonzero(base_eligible, as_tuple=False).reshape(-1) + track_count
    reserve_candidates = torch.cat((reserve_track_ids, reserve_base_ids))
    base_utility = _base_utility(graph, base_count)
    utility = torch.cat((quality, base_utility))
    coverage_selected, matching, coverage_report = selector.complete_matching(
        reserve_candidates,
        utility,
        query_groups,
        requested_rows_per_query=parameters.matching_rows_target,
        maximum_reserve=int(policy["coverage_reserve_maximum"]),
        alias_risk=alias_risk,
    )
    selected = selector.compatibility_materialization_ids

    track_xyz = torch.as_tensor(
        deployment_geometry["triangulated_xyz"]
    ).float()
    base_xyz = torch.as_tensor(canonical["anchor_xyz"][:base_count]).float()
    xyz = torch.cat((track_xyz, base_xyz))
    finite_tracks = torch.isfinite(track_xyz).all(dim=1)
    track_sources = torch.full((track_count,), -1, dtype=torch.long)
    finite_track_indices = torch.nonzero(
        finite_tracks, as_tuple=False
    ).reshape(-1)
    track_sources[finite_track_indices] = _track_source_ids(
        canonical, deployment_payload, finite_track_indices
    )
    source_ids = torch.cat(
        (
            track_sources,
            torch.as_tensor(canonical["source_primitive_ids"][:base_count]).long(),
        )
    )
    voxel_ids = spatial_voxel_ids(xyz, parameters.dependency_voxel_m)
    matchability = _candidate_matchability(
        deployment_payload,
        graph,
        base_count,
        parameters.track_reprojection_median_px,
    )
    if "triangulation_covariance_matrix" in deployment_geometry:
        track_covariance = torch.as_tensor(
            deployment_geometry["triangulation_covariance_matrix"]
        ).float()
        covariance_model = "anisotropic_triangulation_covariance_projected_by_Jx"
    else:
        track_covariance = torch.as_tensor(
            deployment_geometry["triangulation_covariance_trace"]
        ).float().clamp_min(0)
        covariance_model = "isotropic_trace_fallback_projected_by_Jx"
    selected_mask = torch.zeros(len(edges), dtype=torch.bool)
    selected_mask[selected] = True
    pose_candidates = reserve_candidates[
        ~selected_mask[reserve_candidates]
        & (
            matchability[reserve_candidates]
            >= float(policy["pose_minimum_matchability"])
        )
    ]
    pose_relevant = selected_mask.clone()
    pose_relevant[pose_candidates] = True
    pose_edges = [
        candidate_edges if bool(pose_relevant[index]) else {}
        for index, candidate_edges in enumerate(edges)
    ]
    pose_evidence = _build_pose_evidence(
        pose_edges,
        xyz,
        matchability,
        track_covariance,
        list(teacher["query_names"]),
        query_cache,
        voxel_ids,
        pixel_variance=max(parameters.positive_radius_px, 0.5) ** 2,
        task_translation_m=parameters.task_translation_m,
        task_rotation_deg=parameters.task_rotation_deg,
    )
    initial = _initial_pose_state(
        selected, matching, pose_evidence, len(teacher["query_names"])
    )
    pose_selected, pose_report = selector.complete_observability(
        pose_evidence,
        *initial[:5],
        pose_candidates,
        source_ids,
        voxel_ids,
        maximum_additions=int(policy["pose_reserve_maximum"]),
        query_weights=torch.from_numpy(query_weights_from_groups(query_groups)),
        maximum_per_source=int(policy["maximum_per_source"]),
        maximum_per_voxel=int(policy["maximum_per_spatial_voxel"]),
        minimum_relative_gain=float(policy["pose_minimum_relative_gain"]),
        minimum_objective_relative_gain=float(
            policy.get("pose_minimum_objective_relative_gain", 0.0)
        ),
        minimum_additions=int(policy.get("pose_minimum_additions", 0)),
        image_diversity_weight=float(policy["image_diversity_weight"]),
        depth_diversity_weight=float(policy["depth_diversity_weight"]),
        voxel_diversity_weight=float(policy["voxel_diversity_weight"]),
        initial_assignments=initial[5],
    )
    pose_report["matchability_model"] = (
        "query_specific_detector_repeatability_x_candidate_global_reliability"
    )
    pose_report["covariance_model"] = covariance_model
    selected = selector.compatibility_materialization_ids
    selected_tracks = selected[selected < track_count]
    selected_base = selected[selected >= track_count] - track_count
    selected_tracks = selected_tracks[torch.argsort(quality[selected_tracks], descending=True)]
    selected_base = selected_base[torch.argsort(base_utility[selected_base], descending=True)]

    original_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        track_features = fuse_track_descriptors(
            payload=payload,
            query_cache=query_payload,
            track_indices=selected_tracks,
            trim_fraction=float(policy["descriptor_trim_fraction"]),
        )
    finally:
        torch.set_num_threads(original_threads)
    budget = int(selected_tracks.numel() + selected_base.numel())
    selection_provenance = {
        "track_core_universe_ids": core.clone(),
        "coverage_track_universe_ids": coverage_selected[
            coverage_selected < track_count
        ].clone(),
        "coverage_gaussian_universe_ids": coverage_selected[
            coverage_selected >= track_count
        ].clone(),
        "pose_track_universe_ids": pose_selected[
            pose_selected < track_count
        ].clone(),
        "pose_gaussian_universe_ids": pose_selected[
            pose_selected >= track_count
        ].clone(),
    }
    surface_supported = torch.as_tensor(
        geometry.get(
            "triangulation_surface_supported",
            torch.zeros(track_count, dtype=torch.bool),
        )
    ).bool()
    surface_promoted = surface_supported & ~image_only_core
    selection_provenance_path = output_dir / "adaptive_selection_provenance.pt"
    unified_selection_path = output_dir / "unified_sufficiency_selection.pt"
    torch.save(
        {
            "schema": "lafgs_adaptive_selection_provenance",
            "version": 1,
            "track_universe_count": track_count,
            **selection_provenance,
        },
        selection_provenance_path,
    )
    torch.save(selector.artifact(), unified_selection_path)
    state = _materialize(
        canonical,
        deployment_payload,
        selected_tracks,
        track_features,
        selected_base,
        budget=budget,
        quality_tier="adaptive_matching_feasible",
        source_map=canonical_path,
        payload_path=payload_path,
        dependency_voxel_size=parameters.dependency_voxel_m,
        separate_spatial_dependency=True,
    )
    state["track_centric_reconstruction"].update(
        {
            "schema": "lafgs_v2_adaptive_topology",
            "calibration": calibration,
            "track_core": core_report,
            "matching_feasible_coverage": coverage_report,
            "dynamic_pose_reserve": pose_report,
            "track_core_count": int(core.numel()),
            "coverage_reserve_count": int(coverage_selected.numel()),
            "pose_reserve_count": int(pose_selected.numel()),
            "selection_provenance": selection_provenance,
            "unified_sufficiency_selection": selector.artifact(),
            "all_candidate_alias_risk": alias_risk_contract,
            "final_track_count": int(selected_tracks.numel()),
            "final_base_count": int(selected_base.numel()),
            "reserve_candidate_pool": "leftover_tracks_plus_gaussian_base",
            "geometry_policy": "hybrid_triangulated_tracks_and_canonical_gaussian_reserve",
            "track_core_geometry_source": "image_only_triangulation",
            "surface_geometry_scope": "functionally_selected_promoted_reserve",
            "surface_supported_candidate_count": int(surface_supported.sum()),
            "surface_promoted_reserve_candidate_count": int(surface_promoted.sum()),
            "surface_promoted_selected_count": int(
                surface_promoted[selected_tracks].sum()
            ),
        }
    )
    map_path = output_dir / f"adaptive_compact_total{budget:05d}.pt"
    torch.save(state, map_path)
    report = {
        "schema": "lafgs_v2_adaptive_distillation_build",
        "version": 1,
        "map": str(map_path),
        "anchor_count": budget,
        "track_count": int(selected_tracks.numel()),
        "base_count": int(selected_base.numel()),
        "calibration": calibration,
        "track_core": core_report,
        "coverage": coverage_report,
        "pose_reserve": pose_report,
        "selection_provenance": {
            "path": str(selection_provenance_path),
            "track_core_count": int(core.numel()),
            "coverage_track_count": int(
                selection_provenance["coverage_track_universe_ids"].numel()
            ),
            "coverage_gaussian_count": int(
                selection_provenance["coverage_gaussian_universe_ids"].numel()
            ),
            "pose_track_count": int(
                selection_provenance["pose_track_universe_ids"].numel()
            ),
            "pose_gaussian_count": int(
                selection_provenance["pose_gaussian_universe_ids"].numel()
            ),
        },
        "unified_sufficiency_selection": {
            "path": str(unified_selection_path),
            "policy": "v3_compatibility",
            "selected_count": int(selector.selected_ids.numel()),
            "behavior_change_authorized": False,
        },
        "all_candidate_alias_risk": alias_risk_contract,
        "surface_supported": {
            "candidate_count": int(surface_supported.sum()),
            "promoted_reserve_candidate_count": int(surface_promoted.sum()),
            "promoted_selected_count": int(
                surface_promoted[selected_tracks].sum()
            ),
            "core_requires_image_only_geometry": True,
            "track_core_geometry_source": "image_only_triangulation",
            "surface_geometry_scope": "functionally_selected_promoted_reserve",
        },
    }
    report_path = output_dir / "adaptive_distillation_build.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"map": str(map_path), "anchor_count": budget}, indent=2))


if __name__ == "__main__":
    main()
