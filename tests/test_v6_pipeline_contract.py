import pytest
import torch

from common.v6_contracts import ordered_query_registry_sha256
from common.v6_pipeline_contract import validate_v6_pipeline_inputs


def _inputs():
    cache = {
        "schema": "render_observation_cache_v2",
        "version": 2,
        "uses_source_mapping_rgb": False,
        "uses_test_queries": False,
        "uses_rendered_depth": True,
        "uses_gaussian_geometry_for_triangulation": False,
        "queries": {"q0": {}},
        "configuration": {
            "render_valid_observation_extraction": {
                "score_gate_stage": "before_native_nms_and_topk"
            }
        },
    }
    state = {
        "schema": "lafgs_materialized_anchor_map",
        "anchor_ids": torch.tensor([3]),
        "v6_mapping_query_names": ["q0"],
        "provenance": {
            "uses_source_mapping_rgb": False,
            "uses_test_queries": False,
            "uses_gaussian_geometry_for_triangulation": False,
            "mapping_source": "gaussian_render_valid_projective_v6",
            "v6_observation_cache_sha256": "cache",
        },
        "projective_anchor_construction": {
            "final_xyz_source": "fixed_camera_robust_ray_triangulation",
            "gaussian_depth_role": "proposal_and_visibility_only",
            "direct_gaussian_surface_anchor": False,
            "posthoc_support_repair": False,
            "parent_child_semantics": False,
        },
        "projective_anchor_observations": {
            "schema": "lafgs_projective_anchor_observations",
            "observation_offsets": torch.tensor([0, 1]),
            "query_indices": torch.tensor([0]),
            "keypoint_indices": torch.tensor([7]),
        },
    }
    association = {
        "schema": "projective_association_graph_v2",
        "uses_source_mapping_rgb": False,
        "uses_test_queries": False,
        "contract": {
            "render_valid_observations_required": True,
            "reciprocal_descriptor": True,
            "known_pose_epipolar": True,
            "one_observation_per_camera_per_component": True,
            "posthoc_support_repair": False,
            "parent_child_semantics": False,
            "child_cap": False,
            "cycle_chain_are_confidence_attributes_not_candidate_types": True,
        },
    }
    report = {
        "schema": "lafgs_v6_projective_map_materialization_report",
        "uses_source_mapping_rgb": False,
        "uses_test_queries": False,
        "input": {"sha256": "cache"},
        "output": {"map_sha256": "map", "association_graph_sha256": "assoc"},
        "contracts": {
            "render_valid_before_nms": True,
            "unified_association_once": True,
            "final_xyz_pure_ray": True,
            "online_protocol_unchanged": True,
        },
    }
    return cache, state, association, report


def test_full_v6_pipeline_contract_accepts_bound_gaussian_projective_map():
    cache, state, association, report = _inputs()
    result = validate_v6_pipeline_inputs(
        state=state,
        observation_cache=cache,
        observation_cache_sha256="cache",
        map_sha256="map",
        association_graph=association,
        association_graph_sha256="assoc",
        materialization_report=report,
    )
    assert result["exact_identity_observation_count"] == 1
    assert result["unified_association"] is True
    assert result["association_graph_self_bound"] is False
    assert result["final_xyz_pure_ray"] is True
    assert result["ordered_query_registry_sha256"] == (
        ordered_query_registry_sha256(["q0"])
    )


def test_full_v6_pipeline_contract_accepts_new_self_bound_association():
    cache, state, association, _ = _inputs()
    association["input_sha256"] = {"observation_cache": "cache"}
    result = validate_v6_pipeline_inputs(
        state=state,
        observation_cache=cache,
        observation_cache_sha256="cache",
        map_sha256="map",
        association_graph=association,
        association_graph_sha256="assoc",
    )
    assert result["association_graph_self_bound"] is True
    assert result["materialization_report_bound"] is False


def test_full_v6_pipeline_contract_rejects_unbound_legacy_association():
    cache, state, association, _ = _inputs()
    with pytest.raises(ValueError, match="lacks mapping-only observation lineage"):
        validate_v6_pipeline_inputs(
            state=state,
            observation_cache=cache,
            observation_cache_sha256="cache",
            map_sha256="map",
            association_graph=association,
            association_graph_sha256="assoc",
        )


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (
            lambda cache, state, association, report: state["provenance"].update(
                mapping_source="legacy_compatible_map"
            ),
            "Gaussian-render projective path",
        ),
        (
            lambda cache, state, association, report: state[
                "projective_anchor_construction"
            ].update(final_xyz_source="gaussian_depth"),
            "final_xyz_source",
        ),
        (
            lambda cache, state, association, report: cache["configuration"][
                "render_valid_observation_extraction"
            ].update(score_gate_stage="after_topk"),
            "before native NMS/Top-K",
        ),
        (
            lambda cache, state, association, report: report["output"].update(
                association_graph_sha256="wrong"
            ),
            "association SHA differs",
        ),
    ],
)
def test_full_v6_pipeline_contract_rejects_partial_or_unbound_paths(
    mutator, message
):
    cache, state, association, report = _inputs()
    mutator(cache, state, association, report)
    with pytest.raises(ValueError, match=message):
        validate_v6_pipeline_inputs(
            state=state,
            observation_cache=cache,
            observation_cache_sha256="cache",
            map_sha256="map",
            association_graph=association,
            association_graph_sha256="assoc",
            materialization_report=report,
        )
