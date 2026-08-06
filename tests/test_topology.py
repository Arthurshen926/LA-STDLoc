import torch

from topology.distillation import greedy_query_multicover
from topology.coverage_reserve import greedy_pose_reserve
from topology.dynamic_reserve import (
    PoseEvidence,
    greedy_dynamic_pose_reserve,
    spatial_voxel_ids,
)
from topology.matching_coverage import (
    IncrementalBipartiteCoverage,
    greedy_matching_reserve,
)


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


def test_matching_coverage_does_not_count_one_anchor_twice():
    edges = [{0: (0, 1)}, {0: (2,)}]
    state = IncrementalBipartiteCoverage(1, edges)
    state.add(0)
    assert state.counts.tolist() == [1]
    state.add(1)
    assert state.counts.tolist() == [2]


def test_matching_coverage_uses_augmenting_reassignment():
    # Anchor 0 initially takes row 0; anchor 1 can force it to row 1.
    edges = [{0: (0, 1)}, {0: (0,)}]
    state = IncrementalBipartiteCoverage(1, edges)
    state.add(0)
    assert state.add(1) == 1
    assert state.counts.tolist() == [2]


def test_matching_reserve_caps_target_at_feasible_rank():
    edges = [{0: (0, 1)}, {0: (0, 1)}]
    selected, _, report = greedy_matching_reserve(
        edges,
        [],
        [0, 1],
        torch.tensor([2.0, 1.0]),
        torch.tensor([0]),
        requested_rows_per_query=4,
        maximum_reserve=4,
    )
    assert selected.numel() == 2
    assert report["feasible_target_count"] == 2
    assert report["unmet_query_count"] == 0


def test_spatial_voxels_do_not_depend_on_source_identity():
    xyz = torch.tensor([[0.1, 0.1, 0.1], [0.2, 0.1, 0.1], [1.2, 0.1, 0.1]])
    groups = spatial_voxel_ids(xyz, 1.0)
    assert groups[0] == groups[1]
    assert groups[0] != groups[2]


def test_dynamic_pose_reserve_updates_full_information_and_stops_naturally():
    eye = torch.eye(6, dtype=torch.float64)
    evidence = [
        [PoseEvidence(0, (0,), eye, 0, 0, 0)],
        [PoseEvidence(0, (1,), eye, 0, 0, 0)],
    ]
    selected, report = greedy_dynamic_pose_reserve(
        evidence,
        initial_information=eye[None] * 1e-4,
        initial_used_rows=[set()],
        initial_image_cells=[set()],
        initial_depth_bins=[set()],
        initial_spatial_voxels=[set()],
        candidates=[0, 1],
        source_ids=torch.tensor([0, 1]),
        voxel_ids=torch.tensor([0, 1]),
        maximum_additions=2,
        minimum_relative_gain=0.0,
        image_diversity_weight=0,
        depth_diversity_weight=0,
        voxel_diversity_weight=0,
    )
    assert selected.tolist() == [0, 1]
    assert report["selection_is_dynamic"] is True
    assert report["objective"].startswith("task_scaled_full_se3")
