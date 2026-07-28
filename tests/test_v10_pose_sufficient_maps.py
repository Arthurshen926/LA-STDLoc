import torch

from scripts.build_lafgs_v10_pose_sufficient_maps import (
    greedy_pose_reserve,
)


def test_pose_reserve_balances_queries_and_respects_source_limit():
    candidates = [
        [(0, 2.0), (1, 1.0)],
        [(0, 2.0), (2, 1.5)],
        [(3, 2.0)],
    ]
    sources = torch.tensor([10, 11, 12, 10])
    voxels = torch.arange(4)

    selected = greedy_pose_reserve(
        candidates,
        sources,
        voxels,
        budget=3,
        minimum_queries_per_anchor_set=1,
    )

    assert selected.tolist()[0] == 0
    assert 3 not in selected.tolist()
    assert set(selected.tolist()) == {0, 1, 2}
