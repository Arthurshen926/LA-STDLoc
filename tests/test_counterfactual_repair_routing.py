import torch

from localization_training.counterfactual_repair_routing import (
    ROUTE_FAMILY,
    ROUTE_PRIMARY,
    ROUTE_REJECT,
    RepairRoutingConfig,
    assign_descriptor_consistent_repair_routes,
    assign_repair_routes,
    _repair_targets,
)


def test_event_support_routing_uses_repair_evidence_not_anchor_history():
    routes = assign_repair_routes(
        [
            (10, "seq1"),
            (10, "seq2"),
            (11, "seq1"),
            (11, "seq1"),
            (12, "seq1"),
            (-1, "seq1"),
        ],
        RepairRoutingConfig(),
    )
    assert routes == [
        ROUTE_PRIMARY,
        ROUTE_PRIMARY,
        ROUTE_FAMILY,
        ROUTE_FAMILY,
        ROUTE_REJECT,
        ROUTE_REJECT,
    ]


def test_descriptor_consistent_routing_splits_modes_and_rejects_single_trajectory():
    routes, clusters = assign_descriptor_consistent_repair_routes(
        [
            (10, "seq1"),
            (10, "seq2"),
            (11, "seq1"),
            (11, "seq2"),
            (11, "seq1"),
            (11, "seq2"),
            (12, "seq1"),
            (12, "seq1"),
        ],
        torch.tensor(
            [
                [1.0, 0.0],
                [0.95, 0.05],
                [1.0, 0.0],
                [0.95, 0.05],
                [0.0, 1.0],
                [0.05, 0.95],
                [1.0, 0.0],
                [0.95, 0.05],
            ]
        ),
        RepairRoutingConfig(minimum_descriptor_cosine=0.8),
    )
    assert routes[:2] == [ROUTE_PRIMARY, ROUTE_PRIMARY]
    assert routes[2:6] == [ROUTE_FAMILY] * 4
    assert clusters[2] == clusters[3]
    assert clusters[4] == clusters[5]
    assert clusters[2] != clusters[4]
    assert routes[6:] == [ROUTE_REJECT, ROUTE_REJECT]


def test_complete_link_routing_does_not_bridge_incompatible_modes():
    routes, clusters = assign_descriptor_consistent_repair_routes(
        [(10, "seq1"), (10, "seq2"), (10, "seq3")],
        torch.tensor(
            [
                [1.0, 0.0],
                [0.8, 0.6],
                [0.8, -0.6],
            ]
        ),
        RepairRoutingConfig(
            minimum_descriptor_cosine=0.75,
            minimum_primary_repair_trajectories=3,
        ),
    )
    assert routes.count(ROUTE_PRIMARY) == 0
    assert routes.count(ROUTE_FAMILY) == 2
    assert routes.count(ROUTE_REJECT) == 1
    assert len({value for value in clusters if value >= 0}) == 1


def test_exact_counterfactual_targets_reject_unaccepted_rows():
    rows, targets = _repair_targets(
        {
            "query_rows": torch.tensor([3, 5]),
            "target_anchor_indices": torch.tensor([7, 9]),
            "accepted": torch.tensor([True, False]),
        }
    )
    assert rows.tolist() == [3, 5]
    assert targets.tolist() == [7, -1]
