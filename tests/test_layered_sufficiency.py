import torch

from topology.layered_sufficiency import (
    select_layered_sufficiency,
    visibility_image_cells,
)


def test_visibility_image_cells_collapses_clustered_anchors() -> None:
    cells = visibility_image_cells(
        torch.tensor([[1.0, 1.0], [20.0, 20.0], [80.0, 20.0], [99.0, 99.0]]),
        image_hw=(100, 100),
    )
    assert cells.tolist() == [0, 0, 3, 15]
    assert torch.unique(cells).numel() == 3


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


def test_layer_targets_are_independent_and_visibility_counts_image_cells() -> None:
    # Candidates 0 and 1 occupy the same visibility cell.  Candidate 2 is
    # needed for the second cell even though detectability/matching need only
    # one feasible row.
    visibility = [{0: (0,)}, {0: (0,)}, {0: (1,)}]
    detectable = [{0: (0,)}, {0: (1,)}, {0: (2,)}]
    matching = [{0: (0,)}, {0: (1,)}, {0: (2,)}]
    result = select_layered_sufficiency(
        layer_edges={
            "visibility": visibility,
            "detectability": detectable,
            "matching": matching,
        },
        reliability=torch.tensor([1.0, 0.9, 0.8]),
        pose_information=torch.zeros(1, 3, 6, 6),
        visibility_target=2,
        detectability_target=1,
        matching_target=1,
        pose_logdet_target=-200.0,
        maximum_anchors=3,
    )
    assert result["selected_anchor_rows"].tolist() == [0, 2]
    assert result["layer_counts"] == {
        "visibility": [2],
        "detectability": [2],
        "matching": [2],
    }
    assert result["layer_targets"] == {
        "visibility": [2],
        "detectability": [1],
        "matching": [1],
    }


def test_pose_stage_can_target_minimum_eigenvalue_and_logdet() -> None:
    information = torch.zeros(1, 2, 6, 6, dtype=torch.float64)
    information[0, 0] = torch.diag(
        torch.tensor([1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1e-6])
    )
    information[0, 1] = torch.eye(6, dtype=torch.float64) * 0.1
    empty_edges = [{}, {}]
    result = select_layered_sufficiency(
        layer_edges={
            name: empty_edges for name in ("visibility", "detectability", "matching")
        },
        reliability=torch.tensor([1.0, 0.9]),
        pose_information=information,
        visibility_target=0,
        detectability_target=0,
        matching_target=0,
        pose_logdet_target=-100.0,
        pose_min_eigenvalue_target=0.05,
        maximum_anchors=1,
    )
    assert result["selected_anchor_rows"].tolist() == [1]
    assert result["pose_min_eigenvalue"][0] >= 0.05
    assert result["unmet"]["pose"] == 0
    assert result["unmet"]["pose_logdet"] == 0
    assert result["unmet"]["pose_min_eigenvalue"] == 0
