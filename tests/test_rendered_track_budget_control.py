import copy

import pytest
import torch

from scripts.materialize_rendered_track_budget_control import budget_prefix_map


def _fixture():
    count = 5
    state = {
        "schema": "lafgs_materialized_anchor_map",
        "anchor_ids": torch.arange(count),
        "anchor_xyz": torch.arange(count * 3).reshape(count, 3).float(),
        "anchor_features": torch.eye(count),
        "track_cluster_ids": torch.tensor([9, 4, 7, 2, 8]),
        "canonical_anchor_count": count,
        "micro_anchor_count": count,
        "requested_micro_anchor_budget": count,
        "base_anchor_count": 0,
        "track_centric_reconstruction": {
            "track_indices": torch.tensor([9, 4, 7, 2, 8]),
            "base_canonical_rows": torch.empty(0, dtype=torch.long),
        },
    }
    selection = {
        "schema": "lafgs_unified_sufficiency_selection",
        "selected_universe_ids": torch.tensor([9, 4, 7, 2, 8]),
        "primary_selection_reasons": [
            "precision",
            "precision",
            "precision",
            "matching_completion",
            "observability_completion",
        ],
        "reports": {"precision": {"realized_count": 3}},
    }
    return state, selection


def test_budget_control_is_exact_precision_prefix():
    state, selection = _fixture()
    output = budget_prefix_map(state, selection, 3)
    assert torch.equal(output["anchor_ids"], torch.arange(3))
    assert torch.equal(output["track_cluster_ids"], torch.tensor([9, 4, 7]))
    assert torch.equal(output["anchor_xyz"], state["anchor_xyz"][:3])
    assert output["canonical_anchor_count"] == 3
    assert output["track_centric_reconstruction"]["track_indices"].tolist() == [9, 4, 7]


@pytest.mark.parametrize("mutation", ["trace", "reason", "reported_count"])
def test_budget_control_rejects_noncausal_prefix(mutation):
    state, selection = _fixture()
    selection = copy.deepcopy(selection)
    if mutation == "trace":
        selection["selected_universe_ids"][0] = 100
    elif mutation == "reason":
        selection["primary_selection_reasons"][1] = "matching_completion"
    else:
        selection["reports"]["precision"]["realized_count"] = 1
    with pytest.raises(ValueError):
        budget_prefix_map(state, selection, 3)


def test_budget_control_rejects_noncontiguous_source_ids():
    state, selection = _fixture()
    state["anchor_ids"][1] = 11
    with pytest.raises(ValueError, match="contiguous"):
        budget_prefix_map(state, selection, 3)
