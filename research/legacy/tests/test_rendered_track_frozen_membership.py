import torch

from scripts.materialize_rendered_track_frozen_membership import (
    transfer_frozen_membership,
)


def _candidate() -> dict:
    count = 4
    return {
        "schema": "lafgs_materialized_anchor_map",
        "anchor_ids": torch.arange(count),
        "anchor_xyz": torch.arange(count * 3).reshape(count, 3).float(),
        "anchor_features": torch.eye(count),
        "source_primitive_ids": torch.full((count,), -1),
        "track_cluster_ids": torch.tensor([4, 2, 3, 1]),
        "anchor_type": torch.ones(count, dtype=torch.long),
    }


def test_frozen_membership_chooses_first_quality_child_and_preserves_source_order():
    source = {"track_cluster_ids": torch.tensor([9, 7, 5])}
    repaired = {"tracks": {"source_track_index": torch.tensor([0, 7, 7, 5, 9])}}
    output, diagnostics = transfer_frozen_membership(source, repaired, _candidate())
    # candidate child 4 -> source 9, child 2 -> source 7, child 3 -> source 5;
    # the lower-quality second child 1 for source 7 is not selected.
    assert output["track_cluster_ids"].tolist() == [4, 2, 3]
    assert output["anchor_xyz"][:, 0].tolist() == [0.0, 3.0, 6.0]
    assert diagnostics["retained_source_count"] == 3
    assert diagnostics["missing_source_count"] == 0


def test_frozen_membership_records_missing_source_without_filling():
    source = {"track_cluster_ids": torch.tensor([9, 8])}
    repaired = {"tracks": {"source_track_index": torch.tensor([0, 7, 7, 5, 9])}}
    output, diagnostics = transfer_frozen_membership(source, repaired, _candidate())
    assert output["track_cluster_ids"].tolist() == [4]
    assert diagnostics["missing_source_track_ids"] == [8]
    assert (
        output["rendered_track_frozen_membership"]["runs_sufficiency_selector"] is False
    )


def test_frozen_membership_rejects_duplicate_source_selection():
    source = {"track_cluster_ids": torch.tensor([9, 9])}
    repaired = {"tracks": {"source_track_index": torch.tensor([0, 7, 7, 5, 9])}}
    try:
        transfer_frozen_membership(source, repaired, _candidate())
    except ValueError as error:
        assert "unique" in str(error)
    else:
        raise AssertionError("duplicate frozen source membership was accepted")


def test_frozen_membership_can_keep_complementary_sibling_view_bins():
    source = {"track_cluster_ids": torch.tensor([7])}
    repaired = {
        "query_bins": torch.tensor([0, 1]),
        "tracks": {
            "parent_source_track_ids": torch.tensor([0, 7, 7, 5, 9]),
            "track_index": torch.tensor([2, 1]),
            "query_index": torch.tensor([0, 1]),
        },
    }
    output, diagnostics = transfer_frozen_membership(
        source, repaired, _candidate(), maximum_children_per_source=2
    )
    assert output["track_cluster_ids"].tolist() == [2, 1]
    assert diagnostics["selected_child_count"] == 2
    assert diagnostics["multi_child_source_count"] == 1
    assert (
        output["rendered_track_frozen_membership"]["maximum_children_per_source"] == 2
    )


def test_complementary_child_bins_are_grouped_in_one_observation_pass():
    source = {"track_cluster_ids": torch.tensor([7])}
    repaired = {
        "query_bins": torch.tensor([0, 1, 2]),
        "tracks": {
            "parent_source_track_ids": torch.tensor([0, 7, 7, 5, 9]),
            "track_index": torch.tensor([1, 2, 1, 2]),
            "query_index": torch.tensor([0, 1, 0, 2]),
        },
    }
    output, _ = transfer_frozen_membership(
        source, repaired, _candidate(), maximum_children_per_source=2
    )
    assert output["track_cluster_ids"].tolist() == [2, 1]
