import torch
import pytest

from common.v6_contracts import exact_identity_positive_contract
from map_learning.self_localization_feedback import (
    active_failure_layers,
    build_self_localization_feedback,
    classify_failure_layer,
)


def test_failure_layers_are_hierarchical() -> None:
    assert (
        classify_failure_layer(
            visible_rank=0,
            detectable_rank=0,
            matching_rank=0,
            required_rank=4,
            pose_information_sufficient=False,
            pose_success=False,
        )
        == "L1"
    )
    assert (
        classify_failure_layer(
            visible_rank=4,
            detectable_rank=3,
            matching_rank=0,
            required_rank=4,
            pose_information_sufficient=False,
            pose_success=False,
        )
        == "L2"
    )
    assert (
        classify_failure_layer(
            visible_rank=4,
            detectable_rank=4,
            matching_rank=3,
            required_rank=4,
            pose_information_sufficient=False,
            pose_success=False,
        )
        == "L3"
    )
    assert (
        classify_failure_layer(
            visible_rank=4,
            detectable_rank=4,
            matching_rank=4,
            required_rank=4,
            pose_information_sufficient=False,
            pose_success=True,
        )
        == "L4"
    )
    assert active_failure_layers(
        visible_rank=1,
        detectable_rank=2,
        matching_rank=3,
        required_rank=4,
        pose_information_sufficient=False,
        pose_success=False,
    ) == ("L1", "L2", "L3", "L4")


def test_feedback_serializes_required_counterfactual_fields() -> None:
    result = build_self_localization_feedback(
        query_names=["q"],
        required_rank=1,
        source_map_sha256="a" * 64,
        query_cache_sha256="b" * 64,
        positive_identity_contract=exact_identity_positive_contract(),
        records=[
            {
                "image_name": "q",
                "visible_rank": 1,
                "detectable_rank": 1,
                "matching_rank": 1,
                "correct_anchor_rank": 1,
                "winner_anchor": 2,
                "best_positive_score": 0.9,
                "best_wrong_score": 0.8,
                "pose_information_sufficient": True,
                "pose_information_contribution": 1.0,
                "pose_success": True,
                "query_geometry_loo": True,
                "query_rows": torch.tensor([0]),
                "winner_anchor_ids": torch.tensor([2]),
                "winner_scores": torch.tensor([0.9]),
                "top1_exact_identity_correct_mask": torch.tensor([True]),
                "top1_geometry_compatible_ambiguous_mask": torch.tensor([False]),
                "top1_identity_projective_incompatible_mask": torch.tensor([False]),
                "top1_negative_mask": torch.tensor([False]),
                "exact_identity_pairs": torch.tensor([[0, 2]]),
                "active_identity_pairs": torch.tensor([[0, 2]]),
                "exact_identity_positive_pairs": torch.tensor([[0, 2]]),
                "identity_positive_count": 1,
                "identity_active_count": 1,
                "identity_lineage_count": 1,
                "geometry_ambiguous_count": 0,
                "descriptor_triplets": torch.tensor([[0, 2, 3, 1]]),
                "descriptor_triplet_harmful_inlier_mask": torch.tensor([False]),
                "descriptor_triplet_pose_weights": torch.tensor([0.0]),
                "descriptor_triplet_legal_pair_clean_mask": torch.tensor([True]),
                "descriptor_identity_supervision_available": True,
                "visible_anchor_ids": torch.tensor([2, 3]),
                "visible_anchor_image_cells": torch.tensor([0, 5]),
                "clean_inlier_pose_anchor_ids": torch.tensor([2]),
                "clean_inlier_pose_query_rows": torch.tensor([0]),
                "clean_inlier_pose_reprojection_errors_px": torch.tensor([0.1]),
                "clean_inlier_pose_information": torch.eye(6).unsqueeze(0),
                "pose_information_anchor_unique": True,
                "estimated_pose_w2c": torch.eye(4),
                "te_cm": 1.0,
                "ae_deg": 2.0,
            }
        ],
    )
    assert result["success_count"] == 1
    assert result["version"] == 3
    assert result["failure_layer_counts_are_overlapping"] is True
    assert result["failure_query_count"] == 0
    assert result["query_descriptor_loo_count"] == 1
    assert result["affected_anchor_policies"] == ["rebuild"]
    assert "affected_anchor_rebuild" in result["deployment_protocol"]
    assert result["records"][0]["query_raw_geometry_observation_loo"] is True
    assert result["records"][0]["query_candidate_topology_loo"] is True
    assert result["records"][0]["positive_wrong_margin"] > 0
    assert result["records"][0]["legal_positive_exists"] is True
    assert result["records"][0]["te_cm"] == 1.0
    assert torch.allclose(
        result["records"][0]["estimated_pose_w2c"],
        torch.eye(4, dtype=torch.float64),
    )
    assert result["identity_positive_count"] == 1
    assert result["identity_supervision_unavailable_query_count"] == 0
    assert result["records"][0]["identity_supervision_available"] is True
    assert result["pose_information_anchor_unique"] is True
    assert result["records"][0]["visible_anchor_image_cells"].tolist() == [0, 5]
    assert result["visibility_evidence_contract"] == {
        "edge_identity": "query_image_grid_cell",
        "grid_shape": [4, 4],
        "raw_visible_anchor_count_is_not_visibility_rank": True,
    }

    purge_record = dict(result["records"][0])
    purge_record["pose_information_sufficient"] = True
    purge_record["descriptor_identity_supervision_available"] = False
    purge_record["affected_anchor_policy"] = "purge"
    with pytest.raises(
        ValueError, match="diagnostic purge feedback cannot contain descriptor triplets"
    ):
        build_self_localization_feedback(
            query_names=["q"],
            required_rank=1,
            source_map_sha256="a" * 64,
            query_cache_sha256="b" * 64,
            positive_identity_contract=exact_identity_positive_contract(),
            records=[purge_record],
        )

    purge_record["descriptor_triplets"] = torch.empty((0, 4), dtype=torch.long)
    purge_record["descriptor_triplet_harmful_inlier_mask"] = torch.empty(
        0, dtype=torch.bool
    )
    purge_record["descriptor_triplet_pose_weights"] = torch.empty(0)
    purge_record["descriptor_triplet_legal_pair_clean_mask"] = torch.empty(
        0, dtype=torch.bool
    )
    purge = build_self_localization_feedback(
        query_names=["q"],
        required_rank=1,
        source_map_sha256="a" * 64,
        query_cache_sha256="b" * 64,
        positive_identity_contract=exact_identity_positive_contract(),
        records=[purge_record],
    )
    assert purge["descriptor_identity_supervision_available"] is False
    assert purge["records"][0]["descriptor_triplets"].shape == (0, 4)
