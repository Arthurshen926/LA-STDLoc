import numpy as np

from topology.crossfit_swap_revision import (
    _select_rank_feasible_pairs,
    temporal_crossfit_split,
)


def test_temporal_crossfit_uses_disjoint_contiguous_blocks():
    names = [f"seq1/frame{index:05d}.png" for index in range(16)]
    selection, gate, report = temporal_crossfit_split(names, block_count=4)
    assert set(selection).isdisjoint(gate)
    assert sorted(selection + gate) == list(range(16))
    assert selection == list(range(4)) + list(range(8, 12))
    assert gate == list(range(4, 8)) + list(range(12, 16))
    assert report["uses_test_queries"] is False


def test_swap_selection_rejects_a_matching_rank_regression():
    edges = [
        {0: (0,)},
        {0: (1,)},
        {0: (0,)},
        {0: (2,)},
    ]
    proposals = [
        (1, 2, 3.0, 3, 0.1),
        (1, 3, 2.0, 2, 0.1),
    ]
    accepted, report = _select_rank_feasible_pairs(
        proposals,
        selected={0, 1},
        edges=edges,
        query_count=1,
        matching_rows_target=2,
        maximum_swaps=1,
    )
    assert [(pair[0], pair[1]) for pair in accepted] == [(1, 3)]
    assert report["required_rank_unmet_query_count"] == 0
    assert np.isclose(report["matching_rank_after_p10"], 2.0)
