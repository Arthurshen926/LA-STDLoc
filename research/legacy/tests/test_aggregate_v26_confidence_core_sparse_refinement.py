from __future__ import annotations

import pytest

from scripts.aggregate_v26_confidence_core_sparse_refinement import (
    _validate_query_registry,
    _validate_selected_contract,
)


def test_query_registry_requires_exact_names_and_poses() -> None:
    row = {"image_name": "seq/frame.png", "gt_pose_w2c": [[1.0]]}
    _validate_query_registry([row], [dict(row)])
    with pytest.raises(ValueError, match="query registries"):
        _validate_query_registry(
            [row], [{"image_name": "other.png", "gt_pose_w2c": [[1.0]]}]
        )


def test_selected_contract_binds_confidence_core_accounting() -> None:
    contract = {
        "pose_conditioned_sparse_refinement": True,
        "refinement_pose_backend": "robust",
        "refinement_candidate_pool": "first_pose_point_projection_frustum",
        "refinement_active_row_retrieval": True,
        "refinement_pre_topk_view_filter": True,
        "core_reserve_refinement": False,
        "minimum_retained_match_count": 256,
        "refinement_pose_conditioned_mutual_matching": True,
        "match_retention_fraction": 0.65,
        "refinement_projection_gate_px": 8.0,
        "refinement_uncertainty_projection_gate_px": 0.0,
        "refinement_uncertainty_maximum_baseline_inliers": 0,
        "refinement_minimum_proposal_count": 75,
        "refinement_minimum_proposal_relative_gain": 0.2,
    }
    config = {
        "pose_conditioned_mutual_matching": True,
        "match_retention_fraction": 0.65,
        "minimum_retained_match_count": 256,
        "projection_gate_px": 8.0,
        "wider_projection_gate_px": 0.0,
        "wider_projection_maximum_inliers": 0,
        "minimum_proposal_count": 75,
        "minimum_proposal_relative_gain": 0.2,
    }
    row = {
        "top1_match_count": 100,
        "retained_match_count": 65,
        "score_filtered_match_count": 35,
        "core_reserve_refinement": 0,
        "sparse_feedback_baseline_comparison_inliers": 20,
    }
    _validate_selected_contract(contract, [row], config, "scene")
    bad = dict(row, score_filtered_match_count=34)
    with pytest.raises(ValueError, match="diagnostics"):
        _validate_selected_contract(contract, [bad], config, "scene")
