import torch

from localization_training.counterfactual_repair_routing import (
    ROUTE_FAMILY,
    ROUTE_PRIMARY,
    ROUTE_REJECT,
    RepairRoutingConfig,
    assign_descriptor_consistent_repair_routes,
    assign_repair_routes,
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
