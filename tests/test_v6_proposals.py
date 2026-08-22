import torch

from topology.v6_anchor_map import subset_projective_anchor_map


def test_subset_rebuilds_projective_csr() -> None:
    state = {
        "anchor_ids": torch.arange(3),
        "anchor_xyz": torch.arange(9).reshape(3, 3),
        "anchor_features": torch.eye(3),
        "source_primitive_ids": torch.full((3,), -1),
        "track_cluster_ids": torch.arange(3),
        "anchor_type": torch.ones(3, dtype=torch.long),
        "dependency_group_ids": torch.arange(3),
        "coarse_dependency_group_ids": torch.arange(3),
        "fine_identity_ids": torch.arange(3),
        "anchor_parent_identity_ids": torch.arange(3),
        "anchor_correlation_group_ids": torch.arange(3),
        "anchor_position_covariance": torch.eye(3).repeat(3, 1, 1),
        "anchor_matchability": torch.ones(3),
        "anchor_candidate_kind": ["a", "b", "c"],
        "projective_anchor_observations": {
            "observation_offsets": torch.tensor([0, 1, 3, 4]),
            "query_indices": torch.tensor([0, 0, 1, 2]),
            "keypoint_indices": torch.tensor([1, 2, 3, 4]),
        },
    }
    selected = subset_projective_anchor_map(state, torch.tensor([0, 2]))
    assert selected["anchor_ids"].tolist() == [0, 1]
    assert selected["projective_anchor_observations"]["observation_offsets"].tolist() == [0, 1, 2]
    assert selected["projective_anchor_observations"]["query_indices"].tolist() == [0, 2]
