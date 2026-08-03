import torch

from topology.distillation import greedy_query_multicover
from topology.coverage_reserve import greedy_pose_reserve


def test_query_multicover_selects_complementary_anchors():
    events = [{0}, {1}, {(1 << 32)}, {(1 << 32) | 1}]
    selected, report = greedy_query_multicover(
        events, set(), torch.tensor([0, 1]), minimum_rows_per_query=2,
        utility=torch.tensor([4., 3., 2., 1.]),
    )
    assert selected.tolist() == [0, 1, 2, 3]
    assert report["unmet_query_count"] == 0


def test_pose_reserve_respects_source_capacity():
    selected = greedy_pose_reserve(
        [[(0, 3.), (1, 2.)], [(0, 2.), (2, 4.)]],
        source_ids=torch.tensor([0, 0, 1]),
        voxel_ids=torch.tensor([0, 1, 2]),
        budget=2, maximum_per_source=1,
    )
    assert len(selected) == 2
    assert torch.unique(torch.tensor([0, 0, 1])[selected]).numel() == 2
