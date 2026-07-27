import torch

from scripts.build_lafgs_function_graph_v3 import (
    _candidate_provenance_mass,
)
from scripts.build_lafgs_raster_provenance_cache import (
    _anchor_source_csr,
)
from scripts.build_lafgs_redundancy_groups_v3 import (
    _BoundedDisjointSet,
)


def test_anchor_source_csr_recovers_track_family():
    state = {
        "anchor_xyz": torch.zeros(3, 3),
        "source_primitive_ids": torch.tensor([5, 6, 7]),
        "track_cluster_ids": torch.tensor([-1, 0, -1]),
        "anchor_ids": torch.tensor([0, 1, 2]),
    }
    track = {
        "assignment": {
            "track_landmark_offsets": torch.tensor([0, 2]),
            "track_landmark_indices": torch.tensor([0, 1]),
            "track_landmark_costs": torch.tensor([0.0, 1.0]),
        },
        "landmark_indices": torch.tensor([10, 11]),
    }
    offsets, ids, weights = _anchor_source_csr(state, track, None)
    assert offsets.tolist() == [0, 1, 3, 4]
    assert ids.tolist() == [5, 10, 11, 7]
    assert torch.isclose(weights[1:3].sum(), torch.tensor(1.0))


def test_candidate_provenance_mass_uses_source_family_weights():
    indices = torch.tensor([[0, 1]])
    primitive_ids = torch.tensor([[10, 11, 12]])
    primitive_mass = torch.tensor([[0.5, 0.25, 0.25]])
    source_ids = torch.tensor([[10, -1], [11, 12]])
    source_weights = torch.tensor([[1.0, 0.0], [0.25, 0.75]])
    mass = _candidate_provenance_mass(
        indices,
        primitive_ids,
        primitive_mass,
        source_ids,
        source_weights,
        8,
        torch.device("cpu"),
    )
    assert torch.allclose(mass, torch.tensor([[0.5, 0.25]]))


def test_bounded_dsu_never_exceeds_maximum_group_size():
    dsu = _BoundedDisjointSet(5)
    assert dsu.union(0, 1, 2)
    assert not dsu.union(0, 2, 2)
    assert dsu.union(2, 3, 3)
    assert dsu.union(2, 4, 3)
    assert not dsu.union(0, 2, 3)
