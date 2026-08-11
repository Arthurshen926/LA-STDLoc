import torch

from topology.identity_mode_revision import (
    _fit_secondary_medoid,
    _remap_csr,
    interleaved_inner_fold_assignments,
)


def test_secondary_mode_requires_recurrent_view_support():
    observations = [
        {
            "query_index": index,
            "raw": torch.tensor([0.0, 1.0]),
            "adapted": torch.tensor([0.0, 1.0]),
        }
        for index in range(6)
    ]
    mode = _fit_secondary_medoid(
        observations,
        torch.tensor([1.0, 0.0]),
        assignment_margin=0.01,
        minimum_observations=5,
        minimum_views=5,
    )
    assert mode is not None
    assert mode["assigned_view_count"] == 6
    assert torch.allclose(mode["adapted"], torch.tensor([0.0, 1.0]))


def test_teacher_csr_remap_retires_and_adds_alias():
    offsets = torch.tensor([0, 2, 3])
    indices = torch.tensor([0, 2, 1])
    old_to_new = torch.tensor([0, -1, 1])
    alias_lookup = torch.tensor([-1, -1, 2])
    revised_offsets, revised_indices = _remap_csr(
        offsets, indices, old_to_new, alias_lookup, new_count=3
    )
    assert torch.equal(revised_offsets, torch.tensor([0, 3, 3]))
    assert torch.equal(revised_indices, torch.tensor([0, 1, 2]))


def test_inner_folds_interleave_only_outer_discovery_queries():
    names = [f"seq-01/frame-{index:06d}.png" for index in range(12)]
    selection = list(range(8))
    outer = {
        name: (0 if index < 8 else 1) for index, name in enumerate(names)
    }
    assignments, report = interleaved_inner_fold_assignments(
        names, selection, outer, inner_fold_count=4
    )
    assert [assignments[names[index]] for index in selection] == [
        0,
        1,
        2,
        3,
        0,
        1,
        2,
        3,
    ]
    assert all(name not in assignments for name in names[8:])
    assert report["fold_query_counts"] == [2, 2, 2, 2]
    assert report["uses_outer_gate_queries"] is False
