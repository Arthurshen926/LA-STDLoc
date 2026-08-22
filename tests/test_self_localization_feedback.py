from map_learning.self_localization_feedback import (
    build_self_localization_feedback,
    classify_failure_layer,
)


def test_failure_layers_are_hierarchical() -> None:
    assert classify_failure_layer(visible_rank=0, detectable_rank=0, matching_rank=0, required_rank=4, pose_information_sufficient=False, pose_success=False) == "L1"
    assert classify_failure_layer(visible_rank=4, detectable_rank=3, matching_rank=0, required_rank=4, pose_information_sufficient=False, pose_success=False) == "L2"
    assert classify_failure_layer(visible_rank=4, detectable_rank=4, matching_rank=3, required_rank=4, pose_information_sufficient=False, pose_success=False) == "L3"
    assert classify_failure_layer(visible_rank=4, detectable_rank=4, matching_rank=4, required_rank=4, pose_information_sufficient=False, pose_success=True) == "L4"


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
    assert result["records"][0]["positive_wrong_margin"] > 0
