import torch

from scripts.build_lafgs_basis_aware_map import (
    materialize_subset,
    select_basis_safe_retirements,
)


def _record(sets, types, correct, harmful=(), positive=(), weights=()):
    return {
        "set_anchor_indices": torch.tensor(sets, dtype=torch.long).reshape(-1, 3),
        "set_types": torch.tensor(types),
        "correct_basin": torch.tensor(correct),
        "blame_harmful_anchors": torch.tensor(harmful, dtype=torch.long),
        "blame_positive_anchors": torch.tensor(positive, dtype=torch.long),
        "blame_weights": torch.tensor(weights, dtype=torch.float32),
    }


def test_basis_selector_retires_harmful_only_and_preserves_good_basis():
    teacher = {
        "records": [
            _record(
                [[0, 1, 2], [3, 4, 5]],
                [0, 1],
                [True, False],
                harmful=[3],
                positive=[0],
                weights=[2.0],
            )
        ]
    }
    selected, report = select_basis_safe_retirements(
        teacher,
        6,
        maximum_retirements=3,
        minimum_bases_per_query=1,
        harmful_weight=1,
        blame_weight=1,
        good_weight=2,
    )
    assert 3 not in selected.tolist()
    assert {0, 1, 2}.issubset(set(selected.tolist()))
    assert report["retired_good_incidence"] == 0


def test_materialized_subset_updates_track_and_base_lineage():
    state = {
        "anchor_xyz": torch.randn(5, 3),
        "anchor_features": torch.randn(5, 4),
        "anchor_ids": torch.arange(5),
        "track_centric_reconstruction": {
            "track_anchor_count": 3,
            "track_indices": torch.tensor([10, 11, 12]),
            "base_canonical_rows": torch.tensor([20, 21]),
        },
    }
    output = materialize_subset(state, torch.tensor([0, 2, 4]), "map.pt")
    assert output["anchor_xyz"].shape[0] == 3
    assert output["track_centric_reconstruction"]["track_indices"].tolist() == [
        10,
        12,
    ]
    assert output["track_centric_reconstruction"][
        "base_canonical_rows"
    ].tolist() == [21]
