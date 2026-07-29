import torch
import torch.nn.functional as F

from localization_training.appearance_family import (
    bitwise_union,
    collapse_duplicate_conflicts,
    discover_spherical_modes,
    robust_family_prototype,
)


def test_bitwise_union_preserves_combined_provenance():
    assert bitwise_union(torch.tensor([1, 3, 4, 5], dtype=torch.uint8)) == 7


def test_collapse_conflicts_keeps_prototypes_but_not_geometry_rows():
    base = {
        "anchor_xyz": torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        "source_primitive_ids": torch.tensor([4, 5]),
        "anchor_features": torch.eye(2),
    }
    conflict = {
        "anchor_xyz": torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        ),
        "source_primitive_ids": torch.tensor([4, 5, 4]),
        "anchor_features": torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
        ),
        "basin_conflict_anchors": {
            "source_anchor_rows": torch.tensor([0]),
            "query_groups": torch.tensor([2]),
            "observation_counts": torch.tensor([5]),
        },
    }
    families = collapse_duplicate_conflicts(base, conflict)
    assert len(families) == 1
    assert families[0]["source_anchor"] == 0
    assert torch.allclose(
        families[0]["prototype"],
        torch.tensor([2**-0.5, 2**-0.5]),
    )


def test_robust_family_prototype_trims_descriptor_outlier():
    descriptors = torch.tensor(
        [[1.0, 0.0], [0.99, 0.01], [0.98, -0.02], [-1.0, 0.0]]
    )
    prototype = robust_family_prototype(
        descriptors, torch.ones(4), trim_fraction=0.25
    )
    assert prototype[0] > 0.99
    assert abs(float(prototype[1])) < 0.02


def test_spherical_mode_discovery_rejects_outlier_and_finds_mode():
    primary = torch.tensor([1.0, 0.0, 0.0])
    primary_observations = F.normalize(
        torch.tensor(
            [[1.0, 0.02, 0.0], [1.0, -0.03, 0.0], [1.0, 0.0, 0.02]]
        ),
        dim=1,
    )
    secondary = F.normalize(
        torch.tensor(
            [
                [0.1, 1.0, 0.0],
                [0.08, 1.0, 0.02],
                [0.12, 1.0, -0.02],
                [0.1, 0.98, 0.01],
            ]
        ),
        dim=1,
    )
    modes = discover_spherical_modes(
        torch.cat((primary_observations, secondary, torch.tensor([[0.0, 0.0, 1.0]]))),
        primary,
        maximum_modes=3,
        minimum_cluster_size=3,
        minimum_separation=0.08,
        trim_fraction=0.2,
    )
    assert len(modes) == 1
    assert modes[0]["observation_count"] == 4
    assert modes[0]["prototype"][1] > 0.98
