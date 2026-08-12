import itertools
import random

import torch

from topology.adaptive_distillation import (
    _deployment_track_geometry,
    _image_only_core_eligibility,
    _project_world_covariance,
)

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


def test_matching_coverage_uses_alias_risk_before_utility_on_equal_gain():
    edges = [{0: (0,)}, {0: (1,)}]
    selected, _, report = greedy_matching_reserve(
        edges,
        initial=[],
        candidates=[0, 1],
        utility=torch.tensor([10.0, 1.0]),
        query_groups=torch.tensor([0]),
        requested_rows_per_query=1,
        maximum_reserve=1,
        alias_risk=torch.tensor([0.9, 0.1]),
    )
    assert selected.tolist() == [1]
    assert report["alias_risk_tiebreak_enabled"]


def test_incremental_matching_matches_bruteforce_on_random_small_graphs():
    generator = random.Random(2026)
    for _ in range(25):
        candidate_count = generator.randint(1, 6)
        row_count = generator.randint(1, 5)
        edges = []
        for _candidate in range(candidate_count):
            rows = tuple(
                row for row in range(row_count) if generator.random() < 0.5
            )
            edges.append({0: rows} if rows else {})
        state = IncrementalBipartiteCoverage(1, edges)
        for candidate in range(candidate_count):
            state.add(candidate)
        optimum = 0
        choices = [(-1, *edges[candidate].get(0, ())) for candidate in range(candidate_count)]
        for assignment in itertools.product(*choices):
            used = [row for row in assignment if row >= 0]
            if len(used) == len(set(used)):
                optimum = max(optimum, len(used))
        assert state.counts.tolist() == [optimum]


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


def test_anisotropic_landmark_covariance_is_projected_with_full_jacobian():
    point = torch.tensor([[1.0, 0.5, 4.0]], dtype=torch.float64)
    covariance = torch.diag(
        torch.tensor([0.01, 0.04, 0.09], dtype=torch.float64)
    )[None]
    intrinsic = torch.tensor(
        [[800.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]],
        dtype=torch.float64,
    )
    projected = _project_world_covariance(
        point, covariance, intrinsic, torch.eye(4, dtype=torch.float64)
    )
    assert projected.shape == (1, 2, 2)
    assert projected[0, 0, 0] != projected[0, 1, 1]
    assert projected[0, 0, 1] > 0


def test_surface_fusion_cannot_promote_a_track_directly_into_core():
    geometry = {
        "triangulated": torch.tensor([True, True]),
        "triangulated_xyz": torch.ones((2, 3)),
        "triangulation_distinct_view_bin_count": torch.tensor([2, 2]),
        "triangulation_reprojection_median_px": torch.tensor([0.5, 0.5]),
        "triangulation_reprojection_p90_px": torch.tensor([1.0, 1.0]),
        "triangulation_covariance_trace": torch.tensor([0.01, 0.01]),
        "triangulation_image_only_reprojection_median_px": torch.tensor([0.5, 0.5]),
        "triangulation_image_only_reprojection_p90_px": torch.tensor([1.0, 1.0]),
        "triangulation_image_only_covariance_trace": torch.tensor([0.01, 1.0]),
        "triangulation_parallax_deg": torch.tensor([2.0, 2.0]),
    }
    eligible = _image_only_core_eligibility(
        geometry, median_px=1.0, p90_px=2.0, covariance_m2=0.1
    )
    assert eligible.tolist() == [True, False]


def test_image_stable_core_deploys_image_only_geometry():
    geometry = {
        "triangulated_xyz": torch.tensor([[1.1, 0.0, 0.0], [2.1, 0.0, 0.0]]),
        "triangulation_image_only_xyz": torch.tensor(
            [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]
        ),
        "triangulation_covariance_trace": torch.tensor([0.01, 0.01]),
        "triangulation_image_only_covariance_trace": torch.tensor([0.1, 0.2]),
    }
    deployed = _deployment_track_geometry(
        geometry, torch.tensor([True, False])
    )
    assert torch.allclose(
        deployed["triangulated_xyz"],
        torch.tensor([[1.0, 0.0, 0.0], [2.1, 0.0, 0.0]]),
    )
    assert torch.allclose(
        deployed["triangulation_covariance_trace"],
        torch.tensor([0.1, 0.01]),
    )
    assert torch.equal(
        geometry["triangulated_xyz"],
        torch.tensor([[1.1, 0.0, 0.0], [2.1, 0.0, 0.0]]),
    )


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


def test_pose_assignment_uses_augmenting_reassignment():
    eye = torch.eye(6, dtype=torch.float64)
    evidence = [
        [PoseEvidence(0, (0, 1), eye, 0, 0, 0)],
        [PoseEvidence(0, (0,), eye, 1, 0, 1)],
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
        minimum_relative_gain=0,
        image_diversity_weight=0,
        depth_diversity_weight=0,
        voxel_diversity_weight=0,
    )
    assert selected.tolist() == [0, 1]
    assert report["augmenting_row_assignment"] is True
    assert report["row_reassignment_count"] >= 1


def test_pose_reserve_uses_objective_relative_natural_stop():
    eye = torch.eye(6, dtype=torch.float64)
    evidence = [[PoseEvidence(0, (0,), eye, 0, 0, 0)]]
    evidence += [
        [PoseEvidence(0, (index,), eye * 1e-8, 0, 0, index)]
        for index in range(1, 5)
    ]
    selected, report = greedy_dynamic_pose_reserve(
        evidence,
        initial_information=eye[None] * 1e-4,
        initial_used_rows=[set()],
        initial_image_cells=[set()],
        initial_depth_bins=[set()],
        initial_spatial_voxels=[set()],
        candidates=list(range(5)),
        source_ids=torch.arange(5),
        voxel_ids=torch.arange(5),
        maximum_additions=5,
        minimum_relative_gain=0,
        minimum_objective_relative_gain=0.01,
        minimum_additions=1,
        image_diversity_weight=0,
        depth_diversity_weight=0,
        voxel_diversity_weight=0,
    )
    assert selected.tolist() == [0]
    assert report["stop_reason"] == "objective_relative_marginal_gain"
