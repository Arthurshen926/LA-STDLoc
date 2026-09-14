import torch

from topology.anchor_equivalence import (
    PAIR_GAUSSIAN_GAUSSIAN,
    PAIR_TRACK_GAUSSIAN,
    anchor_functional_evidence,
    audit_component_ids,
    build_equivalence_candidates,
    equivalence_edge_masks,
    summarize_equivalence_audit,
)
from topology.anchor_registry import SCHEMA as REGISTRY_SCHEMA


def _registry():
    # A0 and A1 share two exact observations; A1 and A2 share one.  A3 is
    # descriptor-identical to A0 but has no shared observation and must not pair.
    return {
        "schema": REGISTRY_SCHEMA,
        "anchor_ids": torch.arange(4),
        "anchor_type": torch.tensor([1, 0, 0, 0]),
        "anchor_xyz": torch.tensor(
            [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.2, 0.0, 0.0], [5.0, 0.0, 0.0]]
        ),
        "anchor_features": torch.tensor(
            [[1.0, 0.0], [0.8, 0.2], [0.0, 1.0], [1.0, 0.0]]
        ),
        "source_primitive_ids": torch.tensor([7, 7, 8, 9]),
        "coarse_dependency_group_ids": torch.tensor([1, 1, 1, 2]),
        "anchor_position_covariance": torch.eye(3).repeat(4, 1, 1) * 0.01,
        "observation_offsets": torch.tensor([0, 3, 6, 8, 9]),
        "observation_query_indices": torch.tensor([0, 1, 2, 0, 1, 3, 3, 4, 9]),
        "observation_keypoint_indices": torch.tensor([4, 5, 6, 4, 5, 7, 7, 8, 4]),
        "observation_count": torch.tensor([3, 3, 2, 1]),
        "query_group_ids": torch.tensor([0, 1, 1, 2, 3, 4, 5, 6, 7, 8]),
        "query_group_semantics": "synthetic",
    }


def test_candidates_require_exact_shared_observation():
    candidates = build_equivalence_candidates(_registry())
    assert candidates["anchor_left"].tolist() == [0, 1]
    assert candidates["anchor_right"].tolist() == [1, 2]
    assert candidates["shared_observation_count"].tolist() == [2, 1]
    assert candidates["shared_query_count"].tolist() == [2, 1]
    assert candidates["shared_query_group_count"].tolist() == [2, 1]
    assert candidates["pair_type"].tolist() == [
        PAIR_TRACK_GAUSSIAN,
        PAIR_GAUSSIAN_GAUSSIAN,
    ]
    assert candidates["same_source_lineage"].tolist() == [True, False]
    assert candidates["covariance_complete"].tolist() == [True, True]
    assert torch.isfinite(candidates["mahalanobis_squared"]).all()
    # Descriptor-identical A0/A3 is correctly absent.
    assert 3 not in candidates["anchor_right"].tolist()


def test_audit_reports_components_and_never_claims_a_merge():
    registry = _registry()
    candidates = build_equivalence_candidates(registry)
    report = summarize_equivalence_audit(
        registry, candidates, distance_scale_m=0.15
    )
    assert report["candidate_pairs"]["count"] == 2
    assert report["calibrated_triage_graph"]["edge_count"] == 2
    assert report["calibrated_triage_graph"]["component_count"] == 1
    assert report["calibrated_triage_graph"]["largest_component_size"] == 3
    assert report["calibrated_triage_graph"]["is_merge_decision"] is False
    assert report["independent_support_graph"]["edge_count"] == 1
    assert report["independent_support_graph"]["anchor_count"] == 2
    assert report["independent_support_graph"][
        "one_identity_upper_bound_reduction"
    ] == 1
    masks = equivalence_edge_masks(candidates, distance_scale_m=0.15)
    component_ids = audit_component_ids(
        candidates, masks["independent_support"], anchor_count=4
    )
    assert component_ids.tolist() == [0, 0, -1, -1]
    assert report["strict_equivalence_readiness"][
        "ready_for_mahalanobis_calibration"
    ]


def test_functional_evidence_maps_only_known_base_rows():
    registry = _registry()
    state = {
        "track_centric_reconstruction": {
            "base_canonical_rows": torch.tensor([2, 4, 5])
        }
    }
    graph = {
        "provenance_harmful_solver_inlier_count": torch.tensor([0, 0, 3, 0, 1, 0]),
        "provenance_opportunity_count": torch.tensor([0, 0, 9, 0, 7, 2]),
    }
    evidence = anchor_functional_evidence(registry, state, graph)
    assert evidence["known_harmful_count"].tolist() == [-1, 3, 1, 0]
    assert evidence["known_opportunity_count"].tolist() == [-1, 9, 7, 2]
