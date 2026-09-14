from pathlib import Path

import torch

from topology.reserve_factor import (
    factor_universe_ids,
    materialize_factor,
    remap_teacher_to_factor,
)


def test_reserve_factor_membership_is_causal_and_nested():
    provenance = {
        "track_core_universe_ids": torch.tensor([0, 1]),
        "coverage_track_universe_ids": torch.tensor([2]),
        "coverage_gaussian_universe_ids": torch.tensor([10, 11]),
        "pose_track_universe_ids": torch.tensor([3]),
        "pose_gaussian_universe_ids": torch.tensor([12]),
    }
    assert factor_universe_ids(provenance, "core").tolist() == [0, 1]
    assert factor_universe_ids(provenance, "track_coverage").tolist() == [0, 1, 2]
    assert factor_universe_ids(provenance, "all_coverage").tolist() == [0, 1, 2, 10, 11]
    assert factor_universe_ids(provenance, "track_pose").tolist() == [0, 1, 2, 10, 11, 3]
    assert factor_universe_ids(provenance, "track_only_final").tolist() == [0, 1, 2, 3]
    assert factor_universe_ids(provenance, "full").tolist() == [0, 1, 2, 10, 11, 3, 12]


def test_reserve_factor_preserves_learned_source_state():
    source = {
        "anchor_ids": torch.arange(4),
        "anchor_type": torch.tensor([1, 1, 0, 0]),
        "track_cluster_ids": torch.tensor([0, 2, -1, -1]),
        "anchor_features": torch.tensor([[10.0], [20.0], [30.0], [40.0]]),
        "anchor_xyz": torch.arange(12, dtype=torch.float32).reshape(4, 3),
        "source_primitive_ids": torch.tensor([4, 5, 6, 7]),
        "dependency_group_ids": torch.tensor([8, 9, 10, 11]),
        "v7_metric_raw_features": torch.tensor([[1.0], [2.0], [3.0], [4.0]]),
        "track_centric_reconstruction": {
            "track_indices": torch.tensor([0, 2]),
            "base_canonical_rows": torch.tensor([1, 3]),
            "dependency_voxel_size": 0.1,
        },
        "provenance": {},
    }
    payload = {"track_geometry": {"triangulated": torch.ones(4, dtype=torch.bool)}}
    factor = materialize_factor(
        source=source,
        canonical={},
        payload=payload,
        universe_ids=torch.tensor([2, 7]),
        source_path=Path("trained.pt"),
        payload_path=Path("tracks.pt"),
        factor="track_only_final",
    )
    assert factor["anchor_features"].tolist() == [[20.0], [40.0]]
    assert factor["anchor_xyz"].tolist() == [
        [3.0, 4.0, 5.0],
        [9.0, 10.0, 11.0],
    ]
    assert factor["v7_metric_raw_features"].tolist() == [[2.0], [4.0]]
    assert factor["factor_source_rows"].tolist() == [1, 3]
    assert factor["track_centric_reconstruction"]["track_indices"].tolist() == [2]
    assert factor["track_centric_reconstruction"]["base_canonical_rows"].tolist() == [3]


def test_complete_positive_teacher_is_filtered_and_reindexed():
    teacher = {
        "anchor_count": 4,
        "records": [
            {
                "positive_offsets": torch.tensor([0, 2, 3]),
                "positive_indices": torch.tensor([0, 2, 3]),
                "ambiguous_offsets": torch.tensor([0, 1, 3]),
                "ambiguous_indices": torch.tensor([1, 2, 3]),
            }
        ],
        "diagnostics": {"exact_track_positive_count": 2},
    }
    remapped = remap_teacher_to_factor(
        teacher, {"factor_source_rows": torch.tensor([2, 3])}
    )
    record = remapped["records"][0]
    assert record["positive_offsets"].tolist() == [0, 1, 2]
    assert record["positive_indices"].tolist() == [0, 1]
    assert record["ambiguous_offsets"].tolist() == [0, 0, 2]
    assert record["ambiguous_indices"].tolist() == [0, 1]
