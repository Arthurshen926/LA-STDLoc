import torch
import pytest

from topology.deployment_revision import (
    select_revision,
    subset_map_and_metric,
    subset_teacher,
)


def _teacher():
    return {
        "anchor_count": 3,
        "query_names": ["q"],
        "records": [
            {
                "query_index": 0,
                "query_rows": torch.tensor([0, 1]),
                "positive_offsets": torch.tensor([0, 1, 2]),
                "positive_indices": torch.tensor([0, 1]),
                "ambiguous_offsets": torch.tensor([0, 0, 0]),
                "ambiguous_indices": torch.empty(0, dtype=torch.long),
            }
        ],
        "diagnostics": {
            "positive_rows": 2,
            "strong_pair_count": 2,
            "ambiguous_pair_count": 0,
        },
    }


def test_revision_prunes_harmful_noncritical_anchor_without_losing_rank(tmp_path):
    counters = {
        name: torch.zeros(3, dtype=torch.float64)
        for name in (
            "winner_count",
            "correct_winner_count",
            "false_attractor_count",
            "ambiguous_winner_count",
            "clean_inlier_count",
            "harmful_inlier_count",
            "counterfactual_clean_gain",
            "information_deletion_loss",
            "tail_nonimproving_winner_count",
        )
    }
    counters["false_attractor_count"][2] = 4
    counters["harmful_inlier_count"][2] = 2
    counters["counterfactual_clean_gain"][2] = 3
    pruned, report = select_revision(
        _teacher(),
        {"counters": counters},
        matching_rows_target=2,
        maximum_prune_fraction=0.5,
    )
    assert pruned.tolist() == [2]
    assert report["matching_constraint"]["unmet_query_count"] == 0


def test_revision_subsets_teacher_and_map_consistently(tmp_path):
    teacher = _teacher()
    keep = torch.tensor([True, False, True])
    revised_teacher = subset_teacher(
        teacher,
        keep,
        tmp_path / "map.pt",
        source_anchor_type=torch.tensor([1, 0, 0]),
    )
    assert revised_teacher["anchor_count"] == 2
    assert revised_teacher["records"][0]["positive_indices"].tolist() == [0]
    state = {
        "anchor_ids": torch.arange(3),
        "anchor_xyz": torch.randn(3, 3),
        "anchor_features": torch.randn(3, 4),
        "anchor_type": torch.tensor([1, 0, 0]),
        "track_cluster_ids": torch.tensor([10, -1, -1]),
        "base_anchor_count": 2,
        "micro_anchor_count": 1,
        "requested_micro_anchor_budget": 1,
        "canonical_anchor_count": 3,
        "track_centric_reconstruction": {
            "track_indices": torch.tensor([10]),
            "base_canonical_rows": torch.tensor([7, 8]),
        },
    }
    metric = {"landmark_indices": torch.arange(3)}
    revised_map, revised_metric = subset_map_and_metric(
        state, metric, keep, output_map=tmp_path / "map.pt"
    )
    assert revised_map["anchor_ids"].tolist() == [0, 1]
    assert revised_map["track_centric_reconstruction"]["track_indices"].tolist() == [10]
    assert revised_metric["landmark_indices"].tolist() == [0, 1]
    assert revised_metric["map_path"] == str((tmp_path / "map.pt").resolve())


def test_revision_teacher_refuses_to_remove_track_core(tmp_path):
    with pytest.raises(ValueError, match="Track Core"):
        subset_teacher(
            _teacher(),
            torch.tensor([False, True, True]),
            tmp_path / "map.pt",
            source_anchor_type=torch.tensor([1, 1, 0]),
        )
