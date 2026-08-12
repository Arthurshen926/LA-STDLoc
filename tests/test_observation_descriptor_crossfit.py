import torch

from map_learning.metric import SharedLowRankMetric
from topology.observation_descriptor_crossfit import (
    SCHEMA,
    audit_crossfit_observation_descriptors,
)


def _registry(names):
    return {
        "schema": "lafgs_evidence_grounded_anchor_registry",
        "version": 1,
        "anchor_ids": torch.arange(3),
        "anchor_features": torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]
        ),
        "anchor_type": torch.tensor([1, 0, 0]),
        "observation_offsets": torch.tensor([0, 0, 4, 8]),
        "observation_query_indices": torch.tensor([0, 1, 2, 3, 0, 1, 2, 3]),
        "observation_keypoint_indices": torch.tensor([0, 0, 0, 0, 1, 1, 1, 1]),
        "query_names": names,
        "query_group_ids": torch.tensor([0, 1, 0, 1]),
    }


def _cache(names):
    return {
        "queries": {
            name: {
                "native_descriptors": torch.tensor([[0.0, 1.0], [-1.0, 0.0]]),
                "native_scores": torch.ones(2),
            }
            for name in names
        }
    }


def _metric():
    return SharedLowRankMetric(descriptor_dim=2, rank=1, max_residual_norm=0.05)


def test_crossfit_blocks_single_trajectory_without_fabricating_folds():
    names = [f"seq-00/{value}.png" for value in range(4)]
    result = audit_crossfit_observation_descriptors(
        _registry(names), _cache(names), _metric()
    )
    assert result["schema"] == SCHEMA
    assert result["crossfit_available"] is False
    assert result["blocker"] == "requires_at_least_two_mapping_trajectories"
    assert result["deployment_descriptor_mutated"] is False


def test_crossfit_uses_disjoint_trajectories_and_marks_stable_surface():
    names = [
        "seq-00/a.png",
        "seq-00/b.png",
        "seq-01/c.png",
        "seq-01/d.png",
    ]
    result = audit_crossfit_observation_descriptors(
        _registry(names),
        _cache(names),
        _metric(),
        device="cpu",
        score_chunk=2,
    )
    assert result["crossfit_available"] is True
    assert result["fold_trajectory_labels"] == {0: ["seq-00"], 1: ["seq-01"]}
    assert result["bidirectional_eligible_mask"].tolist() == [False, True, True]
    assert result["stable_surface_mask"].tolist() == [False, True, True]
    assert all(value["held_out_observation_count"] == 4 for value in result["report"]["directions"])
    assert all(value["r1_delta"] == 0.0 for value in result["report"]["directions"])
    assert result["uses_test_queries"] is False


def test_crossfit_accepts_an_explicit_nonempty_trajectory_partition():
    names = [
        "seq-00/a.png",
        "seq-00/b.png",
        "seq-01/c.png",
        "seq-01/d.png",
    ]
    result = audit_crossfit_observation_descriptors(
        _registry(names),
        _cache(names),
        _metric(),
        fold_a_trajectories=("seq-01",),
        device="cpu",
        score_chunk=2,
    )
    assert result["fold_policy"] == "explicit_trajectory_partition"
    assert result["fold_trajectory_labels"] == {0: ["seq-01"], 1: ["seq-00"]}
