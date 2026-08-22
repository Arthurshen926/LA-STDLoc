import torch

from topology.layered_sufficiency import select_layered_sufficiency


def test_layered_selector_meets_each_rank_before_pose() -> None:
    edges = [
        {0: (0,)},
        {0: (1,)},
        {1: (0,)},
        {1: (1,)},
    ]
    information = torch.zeros(2, 4, 6, 6)
    for query in range(2):
        for anchor in range(4):
            information[query, anchor] = torch.eye(6) * (0.2 + 0.1 * anchor)
    result = select_layered_sufficiency(
        layer_edges={name: edges for name in ("visibility", "detectability", "matching")},
        reliability=torch.tensor([1.0, 0.9, 0.8, 0.7]),
        pose_information=information,
        matching_target=1,
        pose_logdet_target=-20.0,
        maximum_anchors=4,
    )
    assert all(value == 0 for value in result["unmet"].values())
    reasons = [row["reason"] for row in result["trace"]]
    assert reasons[0] == "visibility_sufficiency"
    assert result["contract"]["hierarchical_not_weighted_sum"] is True


def test_pose_stage_stops_when_only_unreachable_query_is_deficient() -> None:
    edges = [{0: (0,)}, {1: (0,)}]
    information = [
        {0: torch.eye(6, dtype=torch.float64)},
        {0: torch.eye(6, dtype=torch.float64), 1: torch.zeros((6, 6), dtype=torch.float64)},
    ]
    result = select_layered_sufficiency(
        layer_edges={name: edges for name in ("visibility", "detectability", "matching")},
        reliability=torch.tensor([1.0, 0.9]),
        pose_information=information,
        matching_target=0,
        pose_logdet_target=0.0,
        maximum_anchors=2,
    )
    assert result["selected_anchor_rows"].tolist() == [0]
    assert result["unmet"]["pose"] == 1
