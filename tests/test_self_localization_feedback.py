from map_learning.self_localization_feedback import (
    active_failure_layers,
    build_self_localization_feedback,
    classify_failure_layer,
)


def test_failure_layers_are_hierarchical() -> None:
    assert classify_failure_layer(visible_rank=0, detectable_rank=0, matching_rank=0, required_rank=4, pose_information_sufficient=False, pose_success=False) == "L1"
    assert classify_failure_layer(visible_rank=4, detectable_rank=3, matching_rank=0, required_rank=4, pose_information_sufficient=False, pose_success=False) == "L2"
    assert classify_failure_layer(visible_rank=4, detectable_rank=4, matching_rank=3, required_rank=4, pose_information_sufficient=False, pose_success=False) == "L3"
    assert classify_failure_layer(visible_rank=4, detectable_rank=4, matching_rank=4, required_rank=4, pose_information_sufficient=False, pose_success=True) == "L4"
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
        records=[{
            "image_name": "q", "visible_rank": 1, "detectable_rank": 1,
            "matching_rank": 1, "correct_anchor_rank": 1, "winner_anchor": 2,
            "best_positive_score": 0.9, "best_wrong_score": 0.8,
            "pose_information_sufficient": True, "pose_information_contribution": 1.0,
            "pose_success": True, "query_geometry_loo": True,
        }],
    )
    assert result["success_count"] == 1
    assert result["version"] == 2
    assert result["failure_layer_counts_are_overlapping"] is True
    assert result["failure_query_count"] == 0
    assert result["query_descriptor_loo_count"] == 1
    assert result["affected_anchor_policies"] == ["rebuild"]
    assert "affected_anchor_rebuild" in result["deployment_protocol"]
    assert result["records"][0]["query_raw_geometry_observation_loo"] is True
    assert result["records"][0]["query_candidate_topology_loo"] is True
    assert result["records"][0]["positive_wrong_margin"] > 0
