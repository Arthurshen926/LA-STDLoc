import torch

from scripts.build_lafgs_basin_family_prototypes import (
    collapse_duplicate_conflicts,
    robust_family_prototype,
)


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
