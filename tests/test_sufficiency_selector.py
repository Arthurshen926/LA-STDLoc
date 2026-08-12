import torch

from topology.dynamic_reserve import PoseEvidence
from topology.sufficiency_selector import CompatibilitySufficiencySelector


def test_compatibility_selector_uses_one_state_and_primary_reason():
    edges = [
        {0: (0,), 1: (0,)},
        {0: (1,)},
        {1: (1,)},
        {0: (2,), 1: (2,)},
    ]
    selector = CompatibilitySufficiencySelector(edges, query_count=2)
    core, _ = selector.select_precision(
        torch.tensor([0, 1]),
        target_rows=1,
        minimum_count=1,
        maximum_count=2,
        check_interval=1,
    )
    coverage, matching, report = selector.complete_matching(
        torch.tensor([1, 2, 3]),
        torch.tensor([4.0, 3.0, 2.0, 1.0]),
        torch.tensor([0, 1]),
        requested_rows_per_query=2,
        maximum_reserve=3,
    )
    artifact = selector.artifact()
    assert core.tolist() == [0]
    assert coverage.tolist() == [3]
    assert selector.selected_ids.tolist() == [0, 3]
    assert torch.equal(
        selector.compatibility_materialization_ids,
        torch.unique(
            torch.cat((core, coverage)), sorted=False
        ),
    )
    assert matching.counts.tolist() == [2, 2]
    assert report["unmet_query_count"] == 0
    assert artifact["primary_selection_reasons"] == [
        "precision",
        "matching_completion",
    ]
    assert artifact["single_selected_state"]
    assert not artifact["behavior_change_authorized"]


def test_compatibility_selector_requires_stage_order():
    selector = CompatibilitySufficiencySelector([{0: (0,)}], query_count=1)
    try:
        selector.complete_matching(
            torch.tensor([0]),
            torch.ones(1),
            torch.zeros(1, dtype=torch.long),
            requested_rows_per_query=1,
            maximum_reserve=1,
        )
    except ValueError as error:
        assert "requires precision" in str(error)
    else:
        raise AssertionError("matching completion accepted an invalid stage order")


def test_compatibility_selector_appends_observability_to_same_trace():
    selector = CompatibilitySufficiencySelector(
        [{0: (0,)}, {0: (1,)}], query_count=1, track_candidate_count=1
    )
    selector.select_precision(
        torch.tensor([0]),
        target_rows=1,
        minimum_count=1,
        maximum_count=1,
        check_interval=1,
    )
    selector.complete_matching(
        torch.tensor([1]),
        torch.ones(2),
        torch.zeros(1, dtype=torch.long),
        requested_rows_per_query=1,
        maximum_reserve=0,
    )
    evidence = [
        [],
        [
            PoseEvidence(
                query=0,
                rows=(1,),
                information=torch.eye(6, dtype=torch.float64),
                image_cell=1,
                depth_bin=1,
                spatial_voxel=1,
                matchability=1.0,
            )
        ],
    ]
    selected, _ = selector.complete_observability(
        evidence,
        torch.eye(6, dtype=torch.float64)[None] * 1e-4,
        [{0}],
        [{0}],
        [{0}],
        [{0}],
        torch.tensor([1]),
        torch.tensor([0, 1]),
        torch.tensor([0, 1]),
        maximum_additions=1,
        minimum_additions=0,
        minimum_relative_gain=0.0,
        minimum_objective_relative_gain=0.0,
        image_diversity_weight=0.0,
        depth_diversity_weight=0.0,
        voxel_diversity_weight=0.0,
        initial_assignments=[{0: 0}],
    )
    assert selected.tolist() == [1]
    artifact = selector.artifact()
    assert artifact["selected_universe_ids"].tolist() == [0, 1]
    assert torch.equal(
        artifact["compatibility_materialization_ids"],
        torch.unique(
            torch.cat((torch.tensor([0]), selected)), sorted=False
        ),
    )
    assert artifact["primary_selection_reasons"][-1] == "observability_completion"
    assert artifact["candidate_partitions"] == {
        "track_evidence_count": 1,
        "surface_evidence_count": 1,
    }


def test_compatibility_materialization_replays_stagewise_unique_order():
    edges = [{0: (1,)}, {0: (2,)}, {0: (0,)}, {0: (3,)}]
    selector = CompatibilitySufficiencySelector(edges, query_count=1)
    core, _ = selector.select_precision(
        torch.tensor([2, 0]),
        target_rows=2,
        minimum_count=2,
        maximum_count=2,
        check_interval=1,
    )
    coverage, matching, _ = selector.complete_matching(
        torch.tensor([1]),
        torch.tensor([1.0, 2.0, 3.0, 0.0]),
        torch.zeros(1, dtype=torch.long),
        requested_rows_per_query=3,
        maximum_reserve=1,
    )
    evidence = [
        [],
        [],
        [],
        [
            PoseEvidence(
                query=0,
                rows=(3,),
                information=torch.eye(6, dtype=torch.float64),
                image_cell=3,
                depth_bin=3,
                spatial_voxel=3,
                matchability=1.0,
            )
        ],
    ]
    pose, _ = selector.complete_observability(
        evidence,
        torch.eye(6, dtype=torch.float64)[None] * 1e-4,
        [set(matching.assignments(0).values())],
        [set()],
        [set()],
        [set()],
        torch.tensor([3]),
        torch.arange(4),
        torch.arange(4),
        maximum_additions=1,
        minimum_additions=0,
        minimum_relative_gain=0.0,
        minimum_objective_relative_gain=0.0,
        image_diversity_weight=0.0,
        depth_diversity_weight=0.0,
        voxel_diversity_weight=0.0,
        initial_assignments=[matching.assignments(0)],
    )
    expected = torch.unique(
        torch.cat((torch.unique(torch.cat((core, coverage)), sorted=False), pose)),
        sorted=False,
    )
    one_shot = torch.unique(torch.cat((core, coverage, pose)), sorted=False)
    assert not torch.equal(expected, one_shot)
    assert torch.equal(selector.compatibility_materialization_ids, expected)
