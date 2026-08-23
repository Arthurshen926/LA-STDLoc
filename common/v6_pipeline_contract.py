"""Fail-closed contract for the Gaussian-render-to-V6 deployment pipeline."""

from __future__ import annotations

from collections.abc import Mapping

import torch

from common.v6_contracts import (
    ASSOCIATION_GRAPH_SCHEMA,
    RENDER_OBSERVATION_SCHEMA,
    require_mapping_only,
    require_schema,
)


_MAP_SCHEMA = "lafgs_materialized_anchor_map"
_OBSERVATION_CSR_SCHEMA = "lafgs_projective_anchor_observations"
_MATERIALIZATION_REPORT_SCHEMA = "lafgs_v6_projective_map_materialization_report"


def _require_true(value: bool, message: str) -> None:
    if value is not True:
        raise ValueError(message)


def _require_false(value: bool, message: str) -> None:
    if value is not False:
        raise ValueError(message)


def validate_v6_pipeline_inputs(
    *,
    state: Mapping,
    observation_cache: Mapping,
    observation_cache_sha256: str,
    map_sha256: str,
    association_graph: Mapping | None = None,
    association_graph_sha256: str | None = None,
    materialization_report: Mapping | None = None,
) -> dict:
    """Validate that a feedback run starts from the complete V6 map path.

    A compatible tensor layout is not enough: the map must be produced from
    Gaussian-render-valid observations, unified projective association, and
    fixed-camera ray triangulation.  The returned dictionary is compact and
    safe to persist in a runner report.
    """

    require_schema(
        observation_cache,
        RENDER_OBSERVATION_SCHEMA,
        label="V6 Gaussian-render observation cache",
    )
    _require_true(
        observation_cache.get("uses_rendered_depth"),
        "V6 observations must include proposal-only rendered depth",
    )
    _require_false(
        observation_cache.get("uses_gaussian_geometry_for_triangulation"),
        "Gaussian geometry cannot supply final triangulation coordinates",
    )
    extraction = observation_cache.get("configuration", {}).get(
        "render_valid_observation_extraction", {}
    )
    if extraction.get("score_gate_stage") != "before_native_nms_and_topk":
        raise ValueError("render validity must be applied before native NMS/Top-K")

    if state.get("schema") != _MAP_SCHEMA:
        raise ValueError(f"V6 map schema differs: {state.get('schema')}")
    provenance = state.get("provenance", {})
    require_mapping_only(provenance, label="V6 initial map")
    if provenance.get("mapping_source") != "gaussian_render_valid_projective_v6":
        raise ValueError("V6 map is not bound to the Gaussian-render projective path")
    if provenance.get("v6_observation_cache_sha256") != observation_cache_sha256:
        raise ValueError("V6 map and observation cache SHA lineage differ")
    _require_false(
        provenance.get("uses_gaussian_geometry_for_triangulation"),
        "V6 map final xyz cannot use Gaussian geometry",
    )

    construction = state.get("projective_anchor_construction", {})
    required_construction = {
        "final_xyz_source": "fixed_camera_robust_ray_triangulation",
        "gaussian_depth_role": "proposal_and_visibility_only",
        "direct_gaussian_surface_anchor": False,
        "posthoc_support_repair": False,
        "parent_child_semantics": False,
    }
    for field, expected in required_construction.items():
        if construction.get(field) != expected:
            raise ValueError(
                f"V6 map construction contract differs for {field}: "
                f"{construction.get(field)} != {expected}"
            )

    observations = state.get("projective_anchor_observations")
    if not isinstance(observations, Mapping) or observations.get(
        "schema"
    ) != _OBSERVATION_CSR_SCHEMA:
        raise ValueError("V6 map lacks exact projective observation identity CSR")
    offsets = torch.as_tensor(observations.get("observation_offsets", ())).long()
    query_indices = torch.as_tensor(observations.get("query_indices", ())).long()
    keypoint_indices = torch.as_tensor(observations.get("keypoint_indices", ())).long()
    anchor_count = int(torch.as_tensor(state.get("anchor_ids", ())).numel())
    if offsets.ndim != 1 or offsets.numel() != anchor_count + 1:
        raise ValueError("V6 observation CSR offsets do not align with Anchors")
    if query_indices.shape != keypoint_indices.shape or query_indices.ndim != 1:
        raise ValueError("V6 observation identity rows do not align")
    if offsets.numel() and (
        int(offsets[0]) != 0
        or int(offsets[-1]) != int(query_indices.numel())
        or bool((offsets[1:] < offsets[:-1]).any())
    ):
        raise ValueError("V6 observation CSR offsets are invalid")
    cache_names = list(
        observation_cache.get("query_names", observation_cache.get("queries", {}))
    )
    if list(state.get("v6_mapping_query_names", ())) != cache_names:
        raise ValueError("V6 map and observation query registries differ")
    if query_indices.numel() and (
        int(query_indices.min()) < 0 or int(query_indices.max()) >= len(cache_names)
    ):
        raise ValueError("V6 observation identity query rows are invalid")

    association_bound = association_graph is not None
    if association_bound != (association_graph_sha256 is not None):
        raise ValueError("association graph and SHA must be supplied together")
    if association_graph is not None:
        require_schema(
            association_graph,
            ASSOCIATION_GRAPH_SCHEMA,
            label="V6 unified association graph",
        )
        contract = association_graph.get("contract", {})
        required_association = {
            "render_valid_observations_required": True,
            "reciprocal_descriptor": True,
            "known_pose_epipolar": True,
            "one_observation_per_camera_per_component": True,
            "posthoc_support_repair": False,
            "parent_child_semantics": False,
            "child_cap": False,
            "cycle_chain_are_confidence_attributes_not_candidate_types": True,
        }
        for field, expected in required_association.items():
            if contract.get(field) != expected:
                raise ValueError(
                    f"V6 association contract differs for {field}: "
                    f"{contract.get(field)} != {expected}"
                )

    report_bound = materialization_report is not None
    if materialization_report is not None:
        require_schema(
            materialization_report,
            _MATERIALIZATION_REPORT_SCHEMA,
            label="V6 materialization report",
        )
        report_input = materialization_report.get("input", {})
        report_output = materialization_report.get("output", {})
        if report_input.get("sha256") != observation_cache_sha256:
            raise ValueError("materialization report observation SHA differs")
        if report_output.get("map_sha256") != map_sha256:
            raise ValueError("materialization report map SHA differs")
        if association_graph_sha256 is not None and report_output.get(
            "association_graph_sha256"
        ) != association_graph_sha256:
            raise ValueError("materialization report association SHA differs")
        contracts = materialization_report.get("contracts", {})
        for field in (
            "render_valid_before_nms",
            "unified_association_once",
            "final_xyz_pure_ray",
            "online_protocol_unchanged",
        ):
            _require_true(
                contracts.get(field),
                f"materialization report does not certify {field}",
            )

    return {
        "schema": "lafgs_v6_full_pipeline_input_contract",
        "version": 1,
        "uses_source_mapping_rgb": False,
        "uses_test_queries": False,
        "observation_cache_sha256": observation_cache_sha256,
        "map_sha256": map_sha256,
        "association_graph_sha256": association_graph_sha256,
        "association_graph_bound": association_bound,
        "materialization_report_bound": report_bound,
        "mapping_query_count": len(cache_names),
        "anchor_count": anchor_count,
        "exact_identity_observation_count": int(query_indices.numel()),
        "render_valid_before_nms": True,
        "unified_association": association_bound,
        "final_xyz_pure_ray": True,
        "gaussian_depth_role": "proposal_and_visibility_only",
        "online_protocol": "native_superpoint_global_top1_one_standard_poselib",
    }
